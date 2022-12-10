We study the role of intrinsic motivation as an exploration bias for reinforcement learning in sparse-reward synergistic tasks, which are tasks where multiple agents must work together to achieve a goal they could not individually.

Our key idea is that a good guiding principle for intrinsic motivation in synergistic tasks is to take actions which affect the world in ways that would not be achieved if the agents were acting on their own.

Thus, we propose to incentivize agents to take (joint) actions whose effects cannot be predicted via a composition of the predicted effect for each individual agent.

We study two instantiations of this idea, one based on the true states encountered, and another based on a dynamics model trained concurrently with the policy.

While the former is simpler, the latter has the benefit of being analytically differentiable with respect to the action taken.

We validate our approach in robotic bimanual manipulation tasks with sparse rewards; we find that our approach yields more efficient learning than both 1) training with only the sparse reward and 2) using the typical surprise-based formulation of intrinsic motivation, which does not bias toward synergistic behavior.

Videos are available on the project webpage: https://sites.google.com/view/iclr2020-synergistic.

Consider a multi-agent environment such as a team of robots working together to play soccer.

It is critical for a joint policy within such an environment to produce synergistic behavior, allowing multiple agents to work together to achieve a goal which they could not achieve individually.

How should agents learn such synergistic behavior efficiently?

A naive strategy would be to learn policies jointly and hope that synergistic behavior emerges.

However, learning policies from sparse, binary rewards is very challenging -exploration is a huge bottleneck when positive reinforcement is infrequent and rare.

In sparse-reward multi-agent environments where synergistic behavior is critical, exploration is an even bigger issue due to the much larger action space.

A common approach for handling the exploration bottleneck in reinforcement learning is to shape the reward using intrinsic motivation, as was first proposed by Schmidhuber (1991) .

This has been shown to yield improved performance across a variety of domains, such as robotic control tasks (Oudeyer et al., 2007) and Atari games (Bellemare et al., 2016; Pathak et al., 2017) .

Typically, intrinsic motivation is formulated as the agent's prediction error regarding some aspects of the world; shaping the reward with such an error term incentivizes the agent to take actions that "surprise it," and is intuitively a useful heuristic for exploration.

But is this a good strategy for encouraging synergistic behavior in multi-agent settings?

Although synergistic behavior may be difficult to predict, it could be equally difficult to predict the effects of certain single-agent behaviors; this formulation of intrinsic motivation as "surprise" does not specifically favor the emergence of synergy.

In this paper, we study an alternative strategy for employing intrinsic motivation to encourage synergistic behavior in multi-agent tasks.

Our method is based on the simple insight that synergistic behavior leads to effects which would not be achieved if the individual agents were acting alone.

So, we propose to reward agents for joint actions that lead to different results compared to if those same actions were done by the agents individually, in a sequential composition.

For instance, consider the task of twisting open a water bottle, which requires two hands (agents): one to hold the base in place, and another to twist the cap.

Only holding the base in place would not effect any change in Figure 1 : An overview of our approach to incentivizing synergistic behavior via intrinsic motivation.

A heavy red bar (requiring two arms to lift) rests on a table, and the policy π θ suggests for arms A and B to lift the bar from opposite ends.

A composition of pretrained single-agent forward models, f A and f B , predicts the resulting state to be one where the bar is only partially lifted, since neither f A nor f B has ever encountered states where the bar is lifted during training.

A forward model trained on the complete two-agent environment, f joint , correctly predicts that the bar is fully lifted, very different from the compositional prediction.

We train π θ to prefer actions such as these, as a way to bias toward synergistic behavior.

Note that differentiating this intrinsic reward with respect to the action taken does not require differentiating through the environment.

the bottle's pose, while twisting the cap without holding the bottle in place would cause the entire bottle to twist, rather than just the cap.

Here, holding with one hand and subsequently twisting with the other would not open the bottle, but holding and twisting concurrently would.

Based on this intuition, we propose a formulation for intrinsic motivation that leverages the difference between the true effect of an action and the composition of individual-agent predicted effects.

We then present a second formulation that instead uses the discrepancy of predictions between a joint and a compositional prediction model.

While the latter formulation requires training a forward model alongside learning the control strategy, it has the benefit of being analytically differentiable with respect to the action taken.

We later show that this can be leveraged within the policy gradient framework, in order to obtain improved sample complexity over using the policy gradient as-is.

As our experimental point of focus, we study five simulated robotic tasks: four bimanual manipulation (bottle opening, ball pickup, corkscrew rotating, and bar pickup) and multi-agent locomotion (ant push).

All tasks have sparse rewards: 1 if the goal is achieved and 0 otherwise.

These tasks were chosen both because they require synergistic behavior, and because they represent challenging control problems for modern state-of-the-art deep reinforcement learning algorithms (Levine et al., 2016; Lillicrap et al., 2015; Gu et al., 2017; Mnih et al., 2016; .

Across all tasks, we find that shaping the reward via our formulation of intrinsic motivation yields more efficient learning than both 1) training with only the sparse reward signal and 2) shaping the reward via the more standard single-agent formulation of intrinsic motivation as "surprise," which does not explicitly encourage synergistic behavior.

We view this work as a step toward general-purpose synergistic multi-agent reinforcement learning.

Prediction error as intrinsic motivation.

The idea of motivating an agent to reach areas of the state space which yield high model prediction error was first proposed by Schmidhuber (1991) .

Generally, this reward obeys the form f (x) −f (x) , i.e. the difference between the predicted and actual value of some function computed on the current state, the taken action, etc. (Barto, 2013; Oudeyer et al., 2007; Bellemare et al., 2016) ; intrinsic motivation can even be used on its own when no extrinsic reward is provided (Pathak et al., 2017; Burda et al., 2018; Haber et al., 2018) .

A separate line of work studies how agents can synthesize a library of skills via intrinsic motivation in the absence of extrinsic rewards (Eysenbach et al., 2018) .

Recent work has also studied the use of surprise-based reward to solve gentle manipulation tasks, with the novel idea of rewarding the agent for errors in its own predictions of the reward function (Huang et al., 2019) .

In this paper, we will propose formulations of intrinsic motivation that are geared toward multi-agent synergistic tasks.

Exploration in multi-agent reinforcement learning.

The problem of efficient exploration in multi-agent settings has received significant attention over the years.

Lookahead-based exploration (Carmel & Markovitch, 1999 ) is a classic strategy; it rewards an agent for exploration that reduces its uncertainty about the models of other agents in the environment.

More recently, social motivation has been proposed as a general principle for guiding exploration (Jaques et al., 2018) : agents should prefer actions that most strongly influence the policies of other agents.

LOLA (Foerster et al., 2018) , though not quite an exploration strategy, follows a similar paradigm: an agent should reason about the impact of its actions on how other agents learn.

Our work approaches the problem from a different angle that incentivizes synergy: we reward agents for taking actions to affect the world in ways that would not be achieved if the agents were acting alone.

Bimanual manipulation.

The field of bimanual, or dual-arm, robotic manipulation has a rich history (Smith et al., 2012) as an interesting problem across several areas, including hardware design, model-based control, and reinforcement learning.

Model-based control strategies for this task often draw on hybrid force-position control theory (Raibert et al., 1981) , and rely on analytical models of the environment dynamics, usually along with assumptions on how the dynamics can be approximately decomposed into terms corresponding to the two arms (Hsu, 1993; Xi et al., 1996) .

On the other hand, learning-based strategies for this task often leverage human demonstrations to circumvent the challenge of exploration (Zollner et al., 2004; Gribovskaya & Billard, 2008; Kroemer et al., 2015) .

In this work, we describe an exploration strategy based on intrinsic motivation, enabling us to solve synergistic tasks such as bimanual manipulation via model-free reinforcement learning.

Our goal is to enable learning for synergistic tasks in settings with sparse extrinsic rewards.

A central hurdle in such scenarios is the exploration bottleneck: there is a large space of possible action sequences that the agents must explore in order to see rewards.

In the absence of intermediate extrinsic rewards to guide this exploration, one can instead rely on intrinsic rewards that bias the exploratory behavior toward "interesting" actions, a notion which we will formalize.

To accomplish any synergistic task, the agents must work together to affect the environment in ways that would not occur if they were working individually.

In Section 3.1, we present a formulation for intrinsic motivation that operationalizes this insight and allows guiding the exploration toward synergistic behavior, consequently learning the desired tasks more efficiently.

In Section 3.2, we present a second formulation that is (partially) differentiable, making learning even more efficient by allowing us to compute analytical gradients with respect to the action taken.

Finally, in Section 3.3 we show how our formulations can be used to efficiently learn task policies.

Problem Setup.

Each of the tasks we consider can be formulated as a two-agent finite-horizon MDP (Puterman, 1994) .

We denote the environment as E, and the agents as A and B. We assume a state s ∈ S can be partitioned as s := We focus on settings where the reward function of this MDP is binary and sparse, yielding reward r extrinsic (s) = 1 only when s achieves some desired goal configuration.

Learning in such a setup corresponds to acquiring a (parameterized) policy π θ that maximizes the expected proportion of times that a goal configuration is achieved by following π θ .

Unfortunately, exploration guided only by a sparse reward is challenging; we propose to additionally bias it via an intrinsic reward function.

Lets ∼ E(s, a) be a next state resulting from executing action a in state s. We wish to formulate an intrinsic reward function r intrinsic (s, a,s) that encourages synergistic actions and can thereby enable more efficient learning.

Our problem setup and proposed approach are easily extended to settings with more than two agents.

Details, with accompanying experimental results, are provided in Appendix E.

We want to encourage actions that affect the environment in ways that would not occur if the agents were acting individually.

To formalize this notion, we note that a "synergistic" action is one where the agents acting together is crucial to the outcome; so, we should expect a different outcome if the corresponding actions were executed sequentially, with each individual agent acting at a time.

Our key insight is that we can leverage this difference between the true outcome of an action and the expected outcome with individual agents acting sequentially as a reward signal.

We can capture the latter via a composition of forward prediction models for the effects of actions by individual agents acting separately.

Concretely, let f A :

be a singleagent prediction model that regresses to the next environment state resulting from A (resp.

B) taking an action in isolation.

1 We define our first formulation of intrinsic reward, r intrinsic 1 (s, a,s), by measuring the prediction error ofs env using a composition of these single-agent prediction models:

For synergistic actions a, the prediction f composed (s, a) will likely be quite different froms env .

In practice, we pretrain f A and f B using data of random interactions in instantiations of the environment E with only a single active agent.

This implies that the agents have already developed an understanding of the effects of acting alone before being placed in multi-agent environments that require synergistic behavior.

Note that while random interactions sufficed to learn useful prediction models f A and f B in our experiments, this is not essential to the formulation, and one could leverage alternative single-agent exploration strategies to collect interaction samples instead.

The reward r intrinsic 1 (s, a,s) presented above encourages actions that have a synergistic effect.

However, note that this "measurement of synergy" for action a in state s requires explicitly observing the outcomes of executing a in the environment.

In contrast, when humans reason about synergistic tasks such as twisting open a bottle cap while holding the bottle base, we judge whether actions will have a synergistic effect without needing to execute them to make this judgement.

Not only is the non-dependence of the intrinsic reward ons scientifically interesting, but it is also practically desirable.

Specifically, the term f composed (s, a) is analytically differentiable with respect to a (assuming that one uses differentiable regressors f A and f B , such as neural networks), buts env is not, sincē s depends on a via the black-box environment.

If we can reformulate the intrinsic reward to be analytically differentiable with respect to a, we can leverage this for more sample-efficient learning.

To this end, we observe that our formulation rewards actions where the expected outcome under the compositional prediction differs from the outcome when the agents act together.

While we used the observed states as the indication of "outcome when the agents act together," we could instead use a predicted outcome here.

We therefore additionally train a joint prediction model f joint : S × A → S env that, given the states and actions of both agents, and the environment state, predicts the next environment state.

We then define our second formulation of intrinsic reward, r intrinsic 2 (s, a, ·), using the disparity between the predictions of the joint and compositional models:

Note that there is no dependence ons.

At first, this formulation may seem less efficient than r intrinsic 1 , since f joint can at best only matchs env , and requires being trained on data.

However, we note that this formulation makes the intrinsic reward analytically differentiable with respect to the action a executed; we can leverage this within the learning algorithm to obtain more informative gradient updates, as we discuss further in the next section.

Relation to Curiosity.

Typical approaches to intrinsic motivation (Stadie et al., 2015; Pathak et al., 2017) , which reward an agent for "doing what surprises it," take on the form r intrinsic non-synergistic (s, a,s) = f joint (s, a) −s env .

These curiosity-based methods will encourage the system to keep finding new behavior that surprises it, and thus can be seen as a technique for curiosity-driven skill discovery.

In contrast, we are focused on synergistic multi-agent tasks with an extrinsic (albeit sparse) reward, so our methods for intrinsic motivation are not intended to encourage a diversity of learned behaviors, but rather to bias exploration to enable sample-efficient learning for a given task.

1 As the true environment dynamics are stochastic, it can be useful to consider probabilistic regressors f .

However, recent successful applications of model-based reinforcement learning Clavera et al., 2018) have used deterministic regressors, modeling just the maximum likelihood transitions.

We simultaneously learn the joint prediction model f joint and the task policy π θ .

We train π θ via reinforcement learning to maximize the expected total shaped reward r full = r intrinsic i∈{1,2} + λ · r extrinsic across an episode.

Concurrently, we make dual-purpose use of the transition samples {(s, a,s)} collected during the interactions with the environment to train f joint , by minimizing the loss f joint (s, a) −s env .

This simultaneous training of f joint and π θ , as was also done by Stadie et al. (2015) , obviates the need for collecting additional samples to pretrain f joint and ensures that the joint prediction model is trained using the "interesting" synergistic actions being explored.

Full pseudocode is provided in Appendix A.

Our second intrinsic reward formulation allows us to leverage differentiability with respect to the action taken to make learning via policy gradient methods more efficient.

Recall that any policy gradient algorithm (Schulman et al., 2017; Williams, 1992) performs gradient ascent with respect to policy parameters θ on the expected reward over trajectories:

T is the horizon.

We show in Appendix B that the gradient can be written as:

the intuition is that we should not consider the effects of a t here since it gets accounted for by the second term.

In practice, however, we opt to treat the policy gradient algorithm as a black box, and simply add (estimates of) the gradients given by the second term to the gradients yielded by the black-box algorithm.

While this leads to double-counting certain gradients (those of the expected reward at each timestep with respect to the action at that timestep), our preliminary experiments found this to minimally affect training, and make the implementation more convenient as one can leverage an off-the-shelf optimizer like PPO (Schulman et al., 2017) .

Our primary contribution is the design of new intrinsic rewards that are used in conjunction with extrinsic rewards in multi-agent sparse-reward synergistic tasks.

We consider both bimanual manipulation tasks and a multi-agent locomotion task, all of which require synergistic behavior, as our testbed.

We establish the utility of our proposed formulations by comparing to baselines that don't use any intrinsic rewards, or use alternative intrinsic reward formulations.

We also consider ablations of our method that help us understand the different intrinsic reward formulations, and the impact of partial differentiability.

We consider four bimanual manipulation tasks: bottle opening, ball pickup, corkscrew rotating, and bar pickup 2 .

Furthermore, we consider a multi-agent locomotion task: ant push (loosely inspired by a domain considered by Nachum et al. (2019) ).

All tasks involve sparse rewards, and require effective use of both agents to be solved.

We simulate all tasks in MuJoCo (Todorov et al., 2012) .

Now, we describe the tasks, state representations, and action spaces.

Environments.

The four manipulation tasks are set up with 2 Sawyer arms at opposite ends of a table, and an object placed on the table surface.

Three of these tasks are visualized in Figure 2 , alongside the multi-agent locomotion task.

Figure 2 : Screenshots of three of our manipulation tasks and the ant push task.

From left to right: ball lifting, corkscrew rotating, bar pickup, ant push.

These tasks are all designed to require two agents.

We learn policies for these tasks given only sparse binary rewards, by encouraging synergistic behavior via intrinsic motivation.

• Bottle Opening: The goal is to rotate a cuboidal bottle cap, relative to a cuboidal bottle base, by 90

• .

The bottle is modeled as two cuboids on top of one another, connected via a hinge joint, such that in the absence of opposing torques, both cuboids rotate together.

We vary the location and size of the bottle across episodes.

• Ball Pickup:

The goal is to lift a slippery ball by 25cm.

The ball slips out when a single arm tries to lift it.

We vary the location and coefficient of friction of the ball across episodes.

• Corkscrew Rotating: The goal is to rotate a corkscrew relative to its base by 180

• .

The corkscrew is modeled as a handle attached to a base via a hinge joint, such that in the absence of opposing torques, both rotate together.

We vary the location and size of the corkscrew across episodes.

• Bar Pickup: The goal is to lift a long heavy bar by 25cm.

The bar is too heavy to be lifted by a single arm.

We vary the location and density of the bar across episodes.

• Ant Push: Two ants and a large block are placed in an environment.

The goal is for the ants to move the block to a particular region.

To control the block precisely, the ants need to push it together, as they will often topple over when trying to push the block by themselves.

We provide experimental results for a three-agent version of ant push in Appendix E.

State Representation.

The internal state of each agent consists of proprioceptive features: joint positions, joint velocities, and (for manipulation tasks) the end effector pose.

The environment state consists of the current timestep, geometry information for the object, and the object pose.

We use a simple Euclidean metric over the state space.

All forward models predict the change in the object's world frame pose, via an additive offset for the 3D position and a Hamilton product for the quaternion representing the orientation.

Action Space.

To facilitate learning within these environments, we provide the system with a discrete library of generic skills, each parameterized by some (learned) continuous parameters.

Therefore, our stochastic policy π θ maps a state to 1) a distribution over skills for agent A to use, 2) a distribution over skills for agent B to use, 3) means and variances of independent Gaussian distributions for every continuous parameter of skills for A, and 4) means and variances of independent Gaussian distributions for every continuous parameter of skills for B. These skills can either be hand-designed (Wolfe et al., 2010; Srivastava et al., 2014) or learned from demonstration (Kroemer et al., 2015) ; as this is not the focus of our paper, we opt to simply hand-design them.

Further details for manipulation tasks.

If we cannot find an inverse kinematics solution for achieving a skill, it is not executed, though it still consumes a timestep.

While executing a skill, if the arms are about to collide with each other, we attempt to bring back the joint positions to what they were before execution.

In either of these cases, the reward is 0.

See Appendix C for more details on these environments.

Network Architecture.

All forward models and the policy are 4-layer fully connected neural networks with 64-unit hidden layers, ReLU activations, and a multi-headed output to capture both the actor and the critic.

Bimanual manipulation tasks are built on the Surreal Robotics Suite (Fan et al., 2018) .

For all tasks, training is parallelized across 50 workers.

Training Details.

Our proposed synergistic intrinsic rewards rely on forward models f A , f B , and f joint .

We pretrain the single-agent model f A (resp.

f B ) on 10 5 samples of experience with a random policy of only agent A (resp.

B) acting.

Note that this pretraining does not use any extrinsic reward, and therefore the number of steps under the extrinsic reward is comparable across all the approaches.

The joint model f joint and policy π θ start from scratch, and are optimized concurrently.

We than to rely only on the extrinsic, sparse reward signal.

Also, typical formulations of intrinsic motivation as surprise do not work well for synergistic tasks because they encourage the system to affect the environment in ways it cannot currently predict, while our approach encourages the system to affect the environment in ways neither agent would if acting on its own, which is a useful bias for learning synergistic behavior.

set the trade-off coefficient λ = 10.

We use the stable baselines (Hill et al., 2018) implementation of proximal policy optimization (PPO) (Schulman et al., 2017) as our policy gradient algorithm.

We use clipping parameter 0.2, entropy loss coefficient 0.01, value loss function coefficient 0.5, gradient clip threshold 0.5, number of steps 10, number of minibatches per update 4, number of optimization epochs per update 4, and Adam (Kingma & Ba, 2014) with learning rate 0.001.

• Random policy: We randomly choose a skill and parameterization for each agent, at every step.

This baseline serves as a sanity check to ensure that our use of skills does not trivialize the tasks.

• Separate-agent surprise: This baseline simultaneously executes two independent single-agent curiosity policies that are pretrained to maximize the "surprise" rewards f A (s, a) −s env and f B (s, a) −s env respectively.

• Extrinsic reward only: This baseline uses only extrinsic sparse rewards r extrinsic , without shaping.

• Non-synergistic surprise: We learn a joint two-agent policy to optimize for the extrinsic reward and the joint surprise: r full = r intrinsic non-synergistic + λ · r extrinsic .

This encourages curiosity-driven skill discovery but does not explicitly encourage synergistic multi-agent behavior.

Figure 3 shows task success rates as a function of the number of interaction samples for the different methods on each environment.

We plot average success rate over 5 random seeds using solid lines, and shade standard deviations.

Now, we summarize our three key takeaways.

1) Synergistic intrinsic rewards boost sample efficiency.

The tasks we consider are hard and our use of parameterized skills does not trivialize the tasks.

Furthermore, these tasks require coordination among the two agents, and so Separate-agent surprise policies do not perform well.

Given enough training samples, Extrinsic reward only policies start to perform decently well.

However, our Figure 4 : Top: Non-synergistic surprise baseline with varying amounts of pretraining for the joint model f joint .

We see that pretraining this joint model does not yield much improvement in performance, and remains significantly worse than our method (brown curve).

This is sensible since the baseline does not explicitly encourage synergistic behavior, as we do.

Bottom: Ablation showing the impact of using analytical gradients on sample efficiency.

r use of synergistic intrinsic rewards to shape the extrinsic rewards from the environment accelerates learning, solving the task consistently with up to 5× fewer samples in some cases.

2) Synergistic intrinsic rewards perform better than non-synergistic intrinsic rewards.

Policies that use our synergistic intrinsic rewards also work better than the Non-synergistic surprise baseline.

This is primarily because the baseline policies learn to exploit the joint model rather than to behave synergistically.

This also explains why Non-synergistic surprise used together with extrinsic reward hurts task performance (green vs. red curve in Figure 3 ).

Past experiments with such surprise models have largely been limited to games, where progress is correlated with continued exploration (Burda et al., 2018) ; solving robotic tasks often involves more than just surprise-driven exploration.

Figure 4 (top) gives additional results showing that our method's competitive advantage over this baseline persists even if we allow the baseline additional interactions to pretrain the joint prediction model f joint without using any extrinsic reward (similar to our method's pretraining for f composed ).

3) Analytical gradients boost sample efficiency.

In going from r intrinsic 1 (compositional prediction error) to r intrinsic 2 (prediction disparity), we changed two things: 1) the reward function and 2) how it is optimized (we used Equation 1 to leverage the partial differentiability of r intrinsic 2 ).

We conduct an ablation to disentangle the impact of these two changes.

Figure 4 (bottom) presents learning curves for using r intrinsic 2 without analytical gradients, situated in comparison to the previously shown results.

When we factor out the difference due to optimization and compare r requires training an extra model f joint concurrently with the policy, which at best could match the trues env .

Leveraging the analytical gradients, though, affords r intrinsic 2 more sample-efficient optimization (brown vs. purple curve), making it a better overall choice.

We have also tried using our formulation of intrinsic motivation without extrinsic reward (λ = 0); qualitatively, the agents learn to act synergistically, but in ways that do not solve the "task," which is sensible since the task is unknown to the agents.

See the project webpage for videos of these results.

Furthermore, in Appendix D we provide a plot of policy performance versus various settings of λ.

In this work, we presented a formulation of intrinsic motivation that encourages synergistic behavior, and allows efficiently learning sparse-reward tasks such as bimanual manipulation and multi-agent locomotion.

We observed significant benefits compared to non-synergistic forms of intrinsic motivation.

Our formulation relied on encouraging actions whose effects would not be achieved by individual agents acting in isolation.

It would be beneficial to extend this notion further, and explicitly encourage action sequences, not just individual actions, whose effects would not be achieved by individual agents.

Furthermore, while our intrinsic reward encouraged synergistic behavior in the single policy being learned, it would be interesting to extend it to learn a diverse set of policies, and thereby discover a broad set of synergistic skills over the course of training.

Finally, it would be good to extend the domains to involve more complicated object types, such as asymmetric or deformable ones; especially for deformable objects, it may be important to engineer better state representations since these objects do not have a natural notion of 6D pose.

Here is full pseudocode of our training algorithm described in Section 3.3:

Algorithm TRAIN-SYNERGISTIC-POLICY(π θ , M, n, α) , the objective to be optimized can be written as:

We will write ∇ θ J(θ) in a particular way.

Letτ t = s 0 , a 0 , s 1 , a 1 , ..., s t be a random variable denoting trajectories up to timestep t, but excluding a t .

We have:

where we have used the fact that trajectories up to timestep t have no dependence on the future s t+1 , a t+1 , ..., s T , and we have split up the expectation.

Now, observe that the inner expectation,

, is dependent on θ since the a t are sampled from the policy π θ ; intuitively, this expression represents the expected reward of s t with respect to the stochasticity in the current policy.

To make this dependence explicit, let us define r

.

Then:

where in the second line, we used both the product rule and the REINFORCE trick (Williams, 1992

In the second line, we have used the facts thatτ t and the extrinsic sparse reward do not depend on a t .

Note that our inverse kinematics feasibility checks allow the system to learn to rule out end effector poses which are impossible to reach, since these cause no change in the state other than consuming a timestep, and generate 0 reward.

We provide additional details about the action space of the ant push environment.

We pre-train four skills: moving up, down, left, and right on the plane.

Each skill has one continuous parameter specifying an amount to move.

Thus at each timestep, an agent's policy must select both which direction to move and how much to move in that direction.

All training hyperparameters are unchanged from the manipulation tasks.

C.3 POLICY ARCHITECTURE Figure 5 shows a diagram of our policy architecture.

Figure 5: The policy π θ maps a state to 1) a categorical distribution over skills for A, 2) a categorical distribution over skills for B, 3) means and variances of independent Gaussian distributions for every continuous parameter of skills for A, and 4) means and variances of independent Gaussian distributions for every continuous parameter of skills for B. To sample from the policy, we first sample skills for A and B, then sample all necessary continuous parameters for the chosen skills from the Gaussian distributions.

Altogether, the two skills and two sets of parameters form an action, which can be fed into the forward models for prediction.

We conducted an experiment to study the impact of the trade-off coefficient λ on the performance of the learned policy.

When λ = 0, no extrinsic reward is used, so the agents learn to act synergistically, but in ways that do not solve the "task," which is sensible since the task is unknown to them.

Our experiments reported in the main text used λ = 10. .

Each result is averaged across 5 random seeds, with standard deviations shown.

We can infer that once λ reaches a high enough value for the extrinsic rewards to outscale the intrinsic rewards when encountered, the agents will be driven toward behavior that yields extrinsic rewards.

These extrinsic, sparse rewards are only provided when the task is successfully completed.

It is straightforward to extend our formulation and proposed approach to more than two agents.

Without loss of generality, suppose there are three agents A, B, and C. One issue is that as the number of agents increases, the ordering of the application of single-agent forward models within f composed becomes increasingly important.

To address this, we also tried evaluating f composed as an average across the predictions given by all six possible orderings of application, but we did not find this to make much difference in the results.

We leave a thorough treatment of this important question to future work.

We tested our proposed approach on a three-agent version of the ant push environment, and found that it continues to provide a useful bias.

In three-agent ant push, we give harder goal regions for the ants to push the blocks to than in two-agent ant push; these regions were chosen by hand to make all three ants be required to coordinate to solve these tasks, rather than just two as before.

We leave for future work the question of how many agents our method can scale to before performance starts to deteriorate.

Figure 7 : Left: Screenshot of three-agent version of ant push environment.

Right: Learning curves for this environment.

Each curve depicts an average across 5 random seeds, with standard deviations shaded.

In this three-agent environment, taking random actions almost never leads to success due to the exponentially lower likelihood of finding a valid sequence of joint actions leading to the goal, and so using only extrinsic reward does not perform well.

It is apparent that our proposed bias toward synergistic behavior is a useful form of intrinsic motivation for guiding exploration in this environment.

<|TLDR|>

@highlight

We propose a formulation of intrinsic motivation that is suitable as an exploration bias in multi-agent sparse-reward synergistic tasks, by encouraging agents to affect the world in ways that would not be achieved if they were acting individually.

@highlight

The paper focuses on using intrinsic motivation to improve the exploration process of reinforcement learning agents in tasks that require multi-agent to achieve.
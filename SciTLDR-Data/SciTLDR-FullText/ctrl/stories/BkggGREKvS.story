A central challenge in multi-agent reinforcement learning is the induction of coordination between agents of a team.

In this work, we investigate how to promote inter-agent coordination using policy regularization and discuss two possible avenues respectively based on inter-agent modelling and synchronized sub-policy selection.

We test each approach in four challenging continuous control tasks with sparse rewards and compare them against three baselines including MADDPG, a state-of-the-art multi-agent reinforcement learning algorithm.

To ensure a fair comparison, we rely on a thorough hyper-parameter selection and training methodology that allows a fixed hyper-parameter search budget for each algorithm and environment.

We consequently assess both the hyper-parameter sensitivity, sample-efficiency and asymptotic performance of each learning method.

Our experiments show that the proposed methods lead to significant improvements on cooperative problems.

We further analyse the effects of the proposed regularizations on the behaviors learned by the agents.

Multi-Agent Reinforcement Learning (MARL) refers to the task of training an agent to maximize its expected return by interacting with an environment that contains other learning agents.

It represents a challenging branch of Reinforcement Learning (RL) with interesting developments in recent years (Hernandez-Leal et al., 2018) .

A popular framework for MARL is the use of a Centralized Training and a Decentralized Execution (CTDE) procedure (Lowe et al., 2017; Iqbal & Sha, 2019; Foerster et al., 2019; Rashid et al., 2018) .

It is typically implemented by training critics that approximate the value of the joint observations and actions, which are used to train actors restricted to the observation of a single agent.

Such critics, if exposed to coordinated joint actions leading to high returns, can steer the agents' policies toward these highly rewarding behaviors.

However, these approaches depend on the agents luckily stumbling on these actions in order to grasp their benefit.

Thus, it might fail in scenarios where coordination is unlikely to occur by chance.

We hypothesize that in such scenarios, coordination-promoting inductive biases on the policy search could help discover coordinated behaviors more efficiently and supersede task-specific reward shaping and curriculum learning.

In this work, we explore two different priors for successful coordination and use these to regularize the learned policies.

The first avenue, TeamReg, assumes that an agent must be able to predict the behavior of its teammates in order to coordinate with them.

The second, CoachReg, supposes that coordinating agents individually recognize different situations and synchronously use different subpolicies to react to them.

In the following sections we show how to derive practical regularization terms from these premises and meticulously evaluate them 1 .

Our contributions are twofold.

First, we propose two novel approaches that aim at promoting coordination in multi-agent systems.

Our methods augment CTDE MARL algorithms with additional multi-agent objectives that act as regularizers and are optimized jointly with the main return-maximization objective.

Second, we design two new sparse-reward cooperative tasks in the multi-agent particle environment (Mordatch & Abbeel, 2018) .

We use them along with two standard multi-agent tasks to present a detailed evaluation of our approaches against three different baselines.

Finally, we validate our methods' key components by performing an ablation study.

Our experiments suggest that our TeamReg objective provides a dense learning signal that helps to guide the policy towards coordination in the absence of external reward, eventually leading it to the discovery of high performing team strategies in a number of cooperative tasks.

Similarly, by enforcing synchronous sub-policy selections, CoachReg enables to fine-tune a sub-behavior for each recognized situation yielding significant improvements on the overall performance.

In this work we consider the framework of Markov Games (Littman, 1994) , a multi-agent extension of Markov Decision Processes (MDPs) with N independent agents.

A Markov Game is defined by the tuple S, T , P, {O

.

S, T , and P respectively are the set of all possible states, the transition function and the initial state distribution.

While these are global properties of the environment, O i , A i and R i are individually defined for each agent i.

They are respectively the observation functions, the sets of all possible actions and the reward functions.

At each time-step t, the global state of the environment is given by s t ∈ S and every agent's individual action vector is denoted by a Agents aim at maximizing their expected discounted return E T t=0 γ t r i t over the time horizon T , where γ ∈ [0, 1] is a discount factor.

MADDPG (Lowe et al., 2017) is an adaptation of the Deep Deterministic Policy Gradient algorithm (DDPG) (Lillicrap et al., 2015) to the multi-agent setting.

It allows the training of cooperating and competing decentralized policies through the use of a centralized training procedure.

In this framework, each agent i possesses its own deterministic policy µ i for action selection and critic Q i for state-action value estimation, which are respectively parametrized by θ i and φ i .

All parametric models are trained off-policy from previous transitions ζ t := (o t , a t , r t , o t+1 ) uniformly sampled from a replay buffer D. of all N agents.

Each centralized critic is trained to estimate the expected return for a particular agent i using the Deep Q-Network (DQN) (Mnih et al., 2015) loss:

For a given set of weights w, we define its target counterpartw, updated fromw ← τ w + (1 − τ )w where τ is a hyper-parameter.

Each policy is updated to maximize the expected discounted return of the corresponding agent i :

Many works in MARL consider explicit communication channels between the agents and distinguish between communicative actions (e.g. broadcasting a given message) and physical actions (e.g. moving in a given direction) (Foerster et al., 2016; Mordatch & Abbeel, 2018; Lazaridou et al., 2016) .

Consequently, they often focus on the emergence of language, considering tasks where the agents must discover a common communication protocol in order to succeed.

Deriving a successful communication protocol can already be seen as coordination in the communicative action space and can enable, to some extent, successful coordination in the physical action space (Ahilan & Dayan, 2019 ).

Yet, explicit communication is not a necessary condition for coordination as agents can rely on physical communication (Mordatch & Abbeel, 2018; Gupta et al., 2017) .

Approaches to shape RL agents' behaviors with respect to other agents have also been explored.

Strouse et al. (2018) use the mutual information between the agent's policy and a goal-independent policy to shape the agent's behavior towards hiding or spelling out its current goal.

However, this approach is only applicable for tasks with an explicit goal representation and is not specifically intended for coordination.

Jaques et al. (2019) approximate the direct causal effect between agent's actions and use it as an intrinsic reward to encourage social empowerment.

This approximation relies on each agent learning a model of other agents' policies to predict its effect on them.

In general, this type of behavior prediction can be referred to as agent modelling (or opponent modelling) and has been used in previous work to enrich representations (Hernandez-Leal et al., 2019) , to stabilise the learning dynamics (He et al., 2016) or to classify the opponent's play style (Schadd et al., 2007) .

In our work, agent modelling is extended to derive a novel incentive toward team-predictable behaviors.

Finally, Barton et al. (2018) propose convergent cross mapping (CCM) to measure the degree of effective coordination between two agents.

Although this may represent an interesting avenue for behavior analysis, it fails to provide a tool for effectively enforcing coordination as CCM must be computed over long time series which makes it an impractical learning signal for singlestep temporal difference methods.

In this work, we design two coordination-driven multi-agent approaches that do not rely on the existence of explicit communication channels and allow to carry the learned coordinated behaviors at test time, when all agents act in a decentralized fashion.

Intuitively, coordination can be defined as an agent's behavior being informed by the one of another agent, i.e. structure in the agents' interactions.

Namely, a team where agents act independently of one another would not be coordinated.

To promote such structure, our proposed methods rely on team-objectives as regularizers of the common policy gradient update.

In this regard, our approach is closely related to General Value Functions and Auxiliary tasks (Sutton & Barto, 2018) used in Deep RL to learn efficient representations (Jaderberg et al., 2019) .

However, this work's novelty lies in the explicit bias of agents' policy towards either predictability for their teammates or synchronous sub-policy selection.

Pseudocodes of our implementations are provided in Appendix C (see Algorithms 1 and 2).

The structure of coordinated interactions can be leveraged to attain a certain degree of predictability of one agent's behavior with respect to its teammate(s).

We hypothesize that the reciprocal also holds i.e. that promoting agents' predictability could foster such team structure and lead to more coordinated behaviors.

This assumption is cast into the decentralized framework by training agents to predict their teammates' actions given only their own observation.

For continuous control, the loss is defined as the Mean Squared Error (MSE) between the predicted and true actions of the teammates, yielding a teammate-modelling secondary objective.

While the previous work of Hernandez-Leal et al. (2019) focus on stationary, non-learning teammates and exclusively use this approach to learn richer internal representations, we propose to extend this objective to drive the teammates' behaviors closer to the prediction by leveraging a differentiable action selection mechanism.

We call team-spirit this novel objective J i,j T S between agents i and j:

whereμ i,j is the policy head of agent i trying to predict the action of agent j. The total gradient for a given agent i becomes:

where λ 1 and λ 2 are hyper-parameters that respectively weight how well an agent should predict its teammates' actions, and how predictable an agent should be for its teammates.

We call TeamReg this dual regularization from team-spirit objectives.

In order to foster structured agents interactions, this method aims at teaching the agents to recognize different situations and synchronously select corresponding sub-behaviors.

Firstly, to enable explicit sub-behavior selection, we propose policy masks that modulate the agents' policy.

A policy mask u j is a one-hot vector of size K with its j th component set to one.

In practice, we use policy masks to perform dropout (Srivastava et al., 2014) in a structured manner oñ h 1 ∈ R M , the pre-activations of the first hidden layer h 1 of the policy network π.

To do so, we construct the vector u j , which is the concatenation of C copies of u j , in order to reach the dimensionality M = C * K. The element-wise product u j h 1 is then performed and only the units ofh 1 at indices m modulo K = j are kept for m = 0, . . .

, M −1.

In our contribution, each agent i generates e i t , its own policy mask, from its observation o i t .

Here, a simple linear layer l i is used to produce a categorical probability distribution p i (e i t |o i t ) from which the one-hot vector is sampled:

To our knowledge, while this method draws similarity to the options and hierarchical frameworks (Sutton & Barto, 2018; Ahilan & Dayan, 2019) and to policy dropout for exploration (Xie et al., 2018) , it is the first to introduce an agent induced modulation of the policy network by a structured dropout that is decentralized at evaluation and without an explicit communication channel.

Although the policy masking mechanism enables the agent to swiftly switch between sub-policies it does not encourage the agents to synchronously modulate their behavior.

To promote synchronization we introduce the coach entity, parametrized by ψ, which learns to produce policy-masks e c t from the joint observations, i.e. p c (e i c |o t ; ψ).

The coach is used at training time only and drives the agents toward synchronously selecting the same behavior mask.

In other words, the coach is trained to output masks that (1) yield high returns when used by the agents and (2) are predictable by the agents.

Similarly, each agent is regularized so that (1) its private mask matches the coach's mask and (2) it derives efficient behavior when using the coach's mask.

At evaluation time, the coach is removed and the agents only rely on their own policy masks.

The policy gradient loss when agent i is provided with the coach's mask is given by:

The difference between the mask of agent i and the coach's one is measured from the Kullback-Leibler divergence:

The total gradient for agent i is:

In order to propagate gradients through the sampled policy mask we reparametrized the categorical distribution using the Gumbel-softmax trick (Jang et al., 2017 ) with a temperature of 1.

We call this coordinated sub-policy selection regularization CoachReg and illustrate it in Figure 2 .

All of our tasks are based on the OpenAI multi-agent particle environments (Mordatch & Abbeel, 2018) .

SPREAD and CHASE were introduced by (Lowe et al., 2017) .

We use SPREAD as is but with sparse rewards only.

CHASE is modified with a prey controlled by repulsion forces and only the predators are learnable, as we wish to focus on coordination in cooperative tasks.

Finally we introduce COMPROMISE and BOUNCE where agents are explicitly tied together.

While nonzero return can be achieved in these tasks by selfish agents, they all benefit from coordinated strategies and optimal return can only be achieved by agents working closely together.

Figure 3 presents visualizations and a brief description of all four tasks.

A detailed description is provided in Appendix A. In all tasks, agents receive as observation their own global position and velocity as well as the relative position of other entities.

Note that work showcasing experiments on this environment often use discrete action spaces and (dense) reward shaping (e.g. the proximity with the objective) (Iqbal & Sha, 2019; Lowe et al., 2017; Jiang & Lu, 2018) .

However, in our experiments, agents learn with continuous action spaces and from sparse rewards.

The proposed methods offer a way to incorporate new inductive biases in CTDE multi-agent policy search algorithms.

In this work, we evaluate them by extending MADDPG, a state of the art algorithm widely used in the MARL litterature.

We compare against vanilla MADDPG as well as two of its variations in the four cooperative multi-agent tasks described in Section 5.

The first variation (DDPG) is the single-agent counterpart of MADDPG (decentralized training).

The second (MADDPG + sharing) shares the policy and value-function models across agents.

To offer a fair comparison between all methods, the hyper-parameter search routine is the same for each algorithm and environment (see Appendix D.1).

For each search-experiment (one per algorithm per environment), 50 randomly sampled hyper-parameter configurations each using 3 training seeds (total of 150 runs) are used to train the models for 15, 000 episodes.

For each algorithm-environment pair, we then select the best hyper-parameter configuration for the final comparison and retrain them on 10 seeds for twice as long.

We give more details about the training setup and model selection in Appendix B and D.2.

The results of the hyperparameter searches are given in Appendix D.5.

From the average learning curves reported in Figure 4 we observe that CoachReg significantly improves performance on three environments (SPREAD, BOUNCE and COMPROMISE) and performs on par with the baselines on the last one (CHASE).

The same can be said for TeamReg, except on COMPROMISE, the only task with an adversarial component, where it significantly underperforms compared to the other algorithms.

We discuss this specific case in Section 6.3.

Finally, parameter sharing is the best performing choice on CHASE, yet this superiority is restricted to this task where the optimal play is to move symmetrically and squeeze the prey into a corner.

Additionally to our two proposed algorithms and the three baselines, we present results for two ablated versions of our methods.

The first ablation (MADDPG + agent modelling) is similar to TeamReg but with λ 2 = 0, which results in only enforcing agent modelling (i.e. agent predictability is not encouraged).

The second ablation (MADDPG + policy mask) is structurally equivalent to CoachReg, but with λ 1,2,3 = 0, which means that agents still predict and apply a mask to their own policy, but synchronicity is not encouraged.

Figure 12 and 13 (Appendix D.6) present the results of the corresponding hyper-parameter search and Figure 5 shows the learning curves for our full regularization approaches, their respective ablated versions and MADDPG.

The use of unsynchronized policy masks might result in swift and unpredictable behavioral changes and make it difficult for agents to perform together and coordinate.

Experimentally, "MADDPG + policy mask" performs similarly or worse than MADDPG on all but one environment, and never outperforms the full CoachReg approach.

However, policy masks alone seem enough to succeed on SPREAD, which is about selecting a landmark from a set.

Regarding "MADDPG + agent modelling", it does not drastically improve on MADDPG apart from on the SPREAD environment, and the full TeamReg approach shows improvement over its ablated version except on the COMPROMISE task, which we discuss in Section 6.3.

First, we investigate the reason for TeamReg's poor performance on COMPROMISE.

Then, we analyse how TeamReg might be helpful in other environments.

COMPROMISE is the only task with a competitive component (and the only one in which agents do not share their rewards).

The two agents being linked, a good policy has both agents reach their landmark successively (maybe by simply having both agents navigate towards the closest landmark).

However, if one agent never reaches for its landmark, the optimal strategy for the other one becomes to drag it around and always go for its own, leading to a strong imbalance in the return cumulated by both agents.

While this scenario very rarely occurs for the other algorithms, we found TeamReg to often lead to such domination cases (see Figure 14 in Appendix E).

Figure 6 depicts the agents' performance difference for every 150 runs of the hyperparameter search for TeamReg and the baselines, and shows that (1) TeamReg is the only algorithm that does lead to large imbalances in performance between the two agents and (2) that these cases where one agent becomes dominant are all associated with high values of λ 2 , which drives the agents to behave in a predictable fashion to one another.

However, the dominated agent eventually gets exposed more and more to sparse reward gathered by being dragged (by chance) onto its own landmark, picks up the goal of the task and starts pulling in its own direction, which causes the average return over agents to drop as we see in Figure 4 .

This experiment demonstrates that using a predictability-based team-regularization in a competitive task can be harmful; quite understandably, you might not want to optimize an objective that aims at making your behavior predictable to your opponent.

On SPREAD and BOUNCE, TeamReg significantly improves the performance over the baselines.

We aim to analyze here the effects of λ 2 on cooperative tasks and investigate if it does make the agent modelling task more successful (by encouraging the agent to be predictable).

To this end, we compare the best performing hyper-parameter configuration for TeamReg on the SPREAD environment with its ablated versions.

The average return and team-spirit loss defined in Section 4.1 are presented in Figure 7 for these three experiments.

Initially, due to the weight initialization, the predicted and actual actions both have relatively small norms yielding small values of team-spirit loss.

As training goes on (∼1000 episodes), the norms of the action-vector increase and the regularization loss becomes more important.

As expected, λ 2 OF F |λ 1 OF F leads to the highest team-spirit loss as it is not trained to predict the actions of other agents correctly.

When using only the agent-modelling objective (λ 1 ON ), the agents significantly decrease the team-spirit loss, but it never reaches values as low as when using the full TeamReg objective.

Finally, when also pushing agents to be predictable (λ 2 ON ), the agents best predict each others' actions and performance is also improved.

We also notice that the team-spirit loss increases when performance starts to improve i.e. when agents start to master the task (∼8000 episodes).

Indeed, once the reward maximisation signals becomes stronger, the relative importance of the second task is reduced.

We hypothesize that being predictable with respect to one-another may push agents to explore in a more structured and informed manner in the absence of reward signal, as similarly pursued by intrinsic motivation approaches (Chentanez et al., 2005) .

(a) The ball is on the left side of the target, agents both select the purple policy mask In this section we aim at experimentally verifying that CoachReg yields the desired behavior: agents synchronously alternating between varied sub-policies.

A special attention is given when the sub-policies are interpretable.

To this end we record and analyze the agents' policy masks on 100 different episodes for each task.

From the collected masks, we reconstructed the empirical mask distribution of each agent (see Figure 15 in Appendix F.1) whose entropy provides an indication of the mask diversity used by a given agent.

Figure 9 (a) shows the mean entropy for each environment compared to the entropy of Categorical Uniform Distributions of size k (k-CUD).

It shows that, on all the environments, agents use at least two distinct masks by having non-zero entropy.

In addition, agents tend to alternate between masks with more variety (close to uniformly switching between 3 masks) on SPREAD (where there are 3 agents and 3 goals) than on the other environments (comprised of 2 agents).

To test if agents are synchronously selecting the same policy mask at test time (without a coach), we compute the Hamming proximity between the agents' mask sequences with 1 − D h where D h is the Hamming distance, i.e. the number of timesteps where the two sequences are different divided by the total number of timesteps.

From Figure 9 (b) we observe that agents are producing similar mask sequences.

Notably, their mask sequences are significantly more similar that the ones of two agent randomly choosing between two masks at each timestep.

Finally, we observe that some settings result in the agents coming up with interesting strategies, like the one depicted in Figure 8 where the agents alternate between two subpolicies depending on the position of the target.

More cases where the agents change sub-policies during an episode are presented in Appendix F.1.

These results indicate that, in addition to improving the performance on coordination tasks, CoachReg indeed yields the expected behaviors.

An interesting following work would be to use entropy regularization to increase the mask usage variety and mutual information to further disentangle sub-policies.

Stability across hyper-parameter configurations is a recurring challenge in Deep RL.

The average performance for each sampled configuration allow to empirically evaluate the robustness of an algorithm w.r.t.

its hyper-parameters.

We share the full results of the hyper-parameter searches in Figures 10, 11 , 12 and 13 in Appendix D.5 and D.6.

Figure 11 shows that while most algorithms can perform reasonably well with the correct configuration, our proposed coordination regularizers can improve robustness to hyper-parameter despite the fact that they have more hyper-parameters to search over.

Such robustness can be of great value with limited computational budgets.

To assess how the proposed methods perform when using an greater number of agents, we present additional experiments for which the number of agents in the SPREAD task is gradually increased from three to six agents.

The results presented in Figure 18 (Appendix G) show that the performance benefits provided by our methods hold when the number of agents is increased.

Strikingly, we also note how quickly the performance of all methods drop when the number of agents rises.

Indeed, with each new agent, the coordination problem becomes more and more difficult, and that might explain why our methods that promote coordination maintain a higher degree of performance in the case of 4 agents.

Nonetheless, estimating the value function also becomes increasingly challenging as the input space grows exponentially with the number of agents.

In the sparse reward setting, the complexity of the task soon becomes too difficult and none of the algorithms is able to solve it with six agents.

In this work we introduced two policy regularization methods to promote multi-agent coordination within the CTDE framework: TeamReg, which is based on inter-agent action predictability and CoachReg that relies on synchronized behavior selection.

A thorough empirical evaluation of these methods showed that they significantly improve asymptotic performances on cooperative multiagent tasks.

Interesting avenues for future work would be to study the proposed regularizations on other policy search methods as well as to combine both incentives and investigate how the two coordinating objectives interact.

Finally, a limitation of the current formulation is that it relies on single-step metrics, which simplifies off-policy learning but also limits the longer-term coordination opportunities.

A promising direction is thus to explore model-based planning approaches to promote long-term multi-agent interactions.

A TASKS DESCRIPTIONS SPREAD (Figure 3a ): In this environment, there are 3 agents (small orange circles) and 3 landmarks (bigger gray circles).

At every timestep, agents receive a team-reward r t = n − c where n is the number of landmarks occupied by at least one agent and c the number of collisions occurring at that timestep.

To maximize their return, agents must therefore spread out and cover all landmarks.

Initial agents' and landmarks' positions are random.

Termination is triggered when the maximum number of timesteps is reached.

BOUNCE (Figure 3b ): In this environment, two agents (small orange circles) are linked together with a spring that pulls them toward each other when stretched above its relaxation length.

At episode's mid-time a ball (smaller black circle) falls from the top of the environment.

Agents must position correctly so as to have the ball bounce on the spring towards the target (bigger beige circle), which turns yellow if the ball's bouncing trajectory passes through it.

They receive a team-reward of r t = 0.1 if the ball reflects towards the side walls, r t = 0.2 if the ball reflects towards the top of the environment, and r t = 10 if the ball reflects towards the target.

At initialisation, the target's and ball's vertical position is fixed, their horizontal positions are random.

Agents' initial positions are also random.

Termination is triggered when the ball is bounced by the agents or when the maximum number of timesteps is reached.

COMPROMISE (Figure 3c ): In this environment, two agents (small orange circles) are linked together with a spring that pulls them toward each other when stretched above its relaxation length.

They both have a distinct assigned landmark (light gray circle for light orange agent, dark gray circle for dark orange agent), and receive a reward of r t = 10 when they reach it.

Once a landmark is reached by its corresponding agent, the landmark is randomly relocated in the environment.

Initial positions of agents and landmark are random.

Termination is triggered when the maximum number of timesteps is reached.

CHASE (Figure 3d ): In this environment, two predators (orange circles) are chasing a prey (turquoise circle).

The prey moves with respect to a scripted policy consisting of repulsion forces from the walls and predators.

At each timestep, the learning agents (predators) receive a teamreward of r t = n where n is the number of predators touching the prey.

The prey has a greater max speed and acceleration than the predators.

Therefore, to maximize their return, the two agents must coordinate in order to squeeze the prey into a corner or a wall and effectively trap it there.

Termination is triggered when the maximum number of time steps is reached.

In all of our experiments, we use the Adam optimizer (Kingma & Ba, 2014) to perform parameter updates.

All models (actors, critics and coach) are parametrized by feedforward networks containing two hidden layers of 128 units.

We use the Rectified Linear Unit (ReLU) (Nair & Hinton, 2010) as activation function and layer normalization (Ba et al., 2016) on the pre-activations unit to stabilize the learning.

We use a buffer-size of 10 6 entries and a batch-size of 1024.

We collect 100 transitions by interacting with the environment for each learning update.

For all tasks in our hyper-parameter searches, we train the agents for 15, 000 episodes of 100 steps and then re-train the best configuration for each algorithm-environment pair for twice as long (30, 000 episodes) to ensure full convergence for the final evaluation.

The scale of the exploration noise is kept constant for the first half of the training time and then decreases linearly to 0 until the end of training.

We use a discount factor γ of 0.95 and a gradient clipping threshold of 0.5 in all experiments.

Finally for CoachReg, we fixed K to 4 meaning that agents could choose between 4 sub-policies.

Since policies' hidden layers are of size 128 the corresponding value for C is 32.

(1) and (2)

end for Update all target weights end for (8) and (7) Update actor with

Update all target weights end for

We perform searches over the following hyper-parameters: the learning rate of the actor α θ , the learning rate of the critic ω φ relative to the actor (α φ = ω φ * α θ ), the target-network soft-update parameter τ and the initial scale of the exploration noise η noise for the Ornstein-Uhlenbeck noise generating process (Uhlenbeck & Ornstein, 1930) as used by Lillicrap et al. (2015) .

When using TeamReg and CoachReg, we additionally search over the regularization weights λ 1 , λ 2 and λ 3 .

The learning rate of the coach is always equal to the actor's learning rate (i.e. α θ = α ψ ), motivated by their similar architectures and learning signals and in order to reduce the search space.

Table 1 shows the ranges from which values for the hyper-parameters are drawn uniformly during the searches.

Table 1 : Ranges for hyper-parameter search, the log base is 10

During training, a policy is evaluated on a set of 10 different episodes every 100 learning steps.

At the end of the training, the model at the best evaluation iteration is saved as the best version of the policy for this training, and is re-evaluated on 100 different episodes to have a better assessment of its final performance.

The performance of a hyper-parameter configuration is defined as the average performance (across seeds) of the policies learned using this set of hyper-parameter values.

Tables 2, 3 , 4, and 5 shows the best hyper-parameters found by the random searches for each of the environments and each of the algorithms.

Tables 6, 7, 8, and 9 shows the best hyper-parameters found by the random searches for each of the environments and each of the ablated algorithms.

The performance of each parameter configuration is reported in Figure 10 yielding the performance distribution across hyper-parameters configurations for each algorithm on each task.

The same distributions are depicted in Figure 11 using box-and-whisker plot.

It can be seen that TeamReg and CoachReg both boost the performance of the third quartile, suggesting an increase in the robustness across hyper-parameter.

Figure 14 : Learning curves for TeamReg and the three baselines on COMPROMISE.

We see that while both agents remain equally performant as they improve at the task for the baseline algorithms, TeamReg tends to make one agent much stronger than the other one.

This domination is optimal as long as the other agent remains docile, as the dominant agent can gather much more reward than if it had to compromise.

However, when the dominated agent finally picks up the task, the dominant agent that has learned a policy that does not compromise see its return dramatically go down and the mean over agents overall then remains lower than for the baselines.

We depict on Figure 15 the mask distribution of each agent for each (seed, environment) experiment.

Firstly, in most of the experiments, agents use at least 2 different masks.

Secondly, for a given experiments, agents' distributions are very similar, suggesting that they are using the same masks in the same situations and that they are therefore synchronized.

Finally, agents collapse more to using only one mask on CHASE, where they also display more dissimilarity between one another.

This may explain why CHASE is the only task where CoachReg does not improve performance.

Indeed, on CHASE, agents do not seem synchronized nor leveraging multiple sub-policies which are the priors to coordination behind CoachReg.

In brief, we observe that CoachReg is less effective in enforcing those priors to coordination of CHASE, an environment where it does not boost nor harm performance.

Figure 15: Agent's policy mask distributions.

For each (seed, environment) we collected the masks of each agents on 100 episodes.

We render here some episodes roll-outs, the agents synchronously switch between policy masks during an episode.

In addition, the whole group selects the same mask as the one that would have been suggested by the coach.

Figure 16: Visualization sequences on two different environments.

An agent's color represent its current policy mask.

For informative purposes the policy mask that the coach would have produced if these situations would have happened during training is displayed next to the frame's timestep.

Agents synchronously switch between the available policy masks.

As in Subsection 6.4 we report the mean entropy of the mask distribution and the mean Hamming proximity for the ablated "MADDPG + policy mask" and compare it to the full CoachReg.

With "MADDPG + policy mask" agents are not incentivized to use the same masks.

Therefore, in order to assess if they synchronously change policy masks, we computed, for each agent pair, seed and environment, the Hamming proximity for every possible masks equivalence (mask 3 of agent 1 corresponds to mask 0 of agent 2, etc.) and selected the equivalence that maximised the Hamming proximity between the two sequences.

We can observe that while "MADDPG + policy mask" agents display a more diverse mask usage, their selection is less synchronized than with CoachReg.

This is easily understandable as the coach will tend to reduce diversity in order to have all the agents agree on a common mask, on the other hand this agreement enables the agents to synchronize their mask selection.

To this regard, it should be noted that "MADDPG + policy mask" agents are more synchronized that agents independently sampling their masks from k-CUD, suggesting that, even in the absence of the coach, agents tend to synchronize their mask selection.

(a) (b) Figure 17 : Entropy of the policy mask distributions for each task and method, averaged over agents and training seeds.

H max,k is the entropy of a k-CUD.

(b) Hamming Proximity between the policy mask sequence of each agent averaged across agent pairs and seeds.

rand k stands for agents independently sampling their masks from k-CUD.

Error bars are SE across seeds.

We varied the number of agents present in the SPREAD task from three to six.

For each algorithm we used the best performing hyper-parameter configuration from the hyper-parameter search performed on SPREAD with three agents and trained on ten different random seeds.

Results are shown in Figure 18 .

As expected the task becomes more complicated when the number of agents increases and no algorithm succeeds at the task with six agents.

This difficulty is likely to be exacerbated by the sparse reward setting.

However, the proposed methods still outperform the baselines showing that they do not disproportionately suffer from the increased regularization pressure of additional agents.

<|TLDR|>

@highlight

We propose regularization objectives for multi-agent RL algorithms that foster coordination on cooperative tasks.

@highlight

This paper proposes two methods of biasing agents towards learning coordinated behaviours and evaluates both rigorously across multi-agent domains of suitable complexity.

@highlight

This paper proposes two methods building upon MADDPG to encourage collaboration amongst decentralized MARL agents.
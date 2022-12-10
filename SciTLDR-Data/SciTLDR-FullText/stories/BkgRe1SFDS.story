Efficiently learning to solve tasks in complex environments is a key challenge for reinforcement learning (RL) agents.

We propose to decompose a complex environment using a task-agnostic world graphs, an abstraction that accelerates learning by enabling agents to focus exploration on a subspace of the environment.

The nodes of a world graph are important waypoint states and edges represent feasible traversals between them.

Our framework has two learning phases: 1) identifying world graph nodes and edges by training a binary recurrent variational auto-encoder (VAE) on trajectory data and 2) a hierarchical RL framework that leverages structural and connectivity knowledge from the learned world graph to bias exploration towards task-relevant waypoints and regions.

We show that our approach significantly accelerates RL on a suite of challenging 2D grid world tasks: compared to baselines, world graph integration doubles achieved rewards on simpler tasks, e.g. MultiGoal, and manages to solve more challenging tasks, e.g. Door-Key, where baselines fail.

Many real-world applications, e.g., self-driving cars and in-home robotics, require an autonomous agent to execute different tasks within a single environment that features, e.g. high-dimensional state space, complex world dynamics or structured layouts.

In these settings, model-free reinforcement learning (RL) agents often struggle to learn efficiently, requiring a large amount of experience collections to converge to optimal behaviors.

Intuitively, an agent could learn more efficiently by focusing its exploration in task-relevant regions, if it has knowledge of the high-level structure of the environment.

We propose a method to 1) learn and 2) use an environment decomposition in the form of a world graph, a task-agnostic abstraction.

World graph nodes are waypoint states, a set of salient states that can summarize agent trajectories and provide meaningful starting points for efficient exploration (Chatzigiorgaki & Skodras, 2009; Jayaraman et al., 2018; Ghosh et al., 2018) .

The directed and weighted world graph edges characterize feasible traversals among the waypoints.

To leverage the world graph, we model hierarchical RL (HRL) agents where a high-level policy chooses a waypoint state as a goal to guide exploration towards task-relevant regions, and a low-level policy strives to reach the chosen goals.

Our framework consists of two phases.

In the task-agnostic phase, we obtain world graphs by training a recurrent variational auto-encoder (VAE) (Chung et al., 2015; Gregor et al., 2015; Kingma & Welling, 2013) with binary latent variables (Nalisnick & Smyth, 2016) over trajectories collected using a random walk policy (Ha & Schmidhuber, 2018 ) and a curiosity-driven goal-conditioned policy (Ghosh et al., 2018; Nair et al., 2018) .

World graph nodes are states that are most frequently selected by the binary latent variables, while edges are inferred from empirical transition statistics between neighboring waypoints.

In the task-specific phase, taking advantage of the learned world graph for structured exploration, we efficiently train an HRL model (Taylor & Stone, 2009 ).

In summary, our main contributions are:

• A task-agnostic unsupervised approach to learn world graphs, using a recurrent VAE with binary latent variables and a curiosity-driven goal-conditioned policy.

• An HRL scheme for the task-specific phase that features multi-goal selection (Wide-thenNarrow) and navigation via world graph traversal.

4.

On its traversal course to wide goal, agent hits final target and exits.

: waypoints selected by the manager : waypoints initiates traversal : trajectories directly from worker actions : exit point : agent : final goal from manager close to selected waypoints : trajectories from world graph traversal

Figure 1: Top Left: overall pipeline of our 2-phase framework.

Top Right (world graph discovery): a subgraph exemplifies traversal between waypoint states (in blue), see Section 3 for more details.

Bottom (Hierarhical RL): an example rollout from our proposed HRL policy with Wide-then-Narrow Manager instructions and world graph traversals, solving a challenging Door-Key task, see Section 4 for more details.

• Empirical evaluations on multiple tasks in complex 2D grid worlds to validate that our framework produces descriptive world graphs and significantly improves both sample efficiency and final performance on these tasks over baselines, especially thanks to transfer learning from the unsupervised phase and world graph traversal.

An understanding of the environment and its dynamics is essential for effective planning and control in model-based RL.

For example, a robotics agent often locates or navigates by interpreting a map (Lowry et al., 2015; Thrun, 1998; Angeli et al., 2008) .

Our exploration strategy draws inspiration from active localization, where robots are actively guided to investigate unfamiliar regions (Fox et al., 1998; Li et al., 2016) .

Besides mapping, recent works (Azar et al., 2019; Ha & Schmidhuber, 2018; Guo et al., 2018) learn to represent the world with generative latent states (Tian & Gong, 2017; Haarnoja et al., 2018; Racanière et al., 2017) .

If the latent dynamics are also extrapolated, the latent states can assist planning (Mnih et al., 2016a; Hafner et al., 2018) or model-based RL (Gregor & Besse, 2018; Kaiser et al., 2019) .

While also aiming to model the world, we approach this as abstracting both the structure and dynamics of the environment in a graph representation, where nodes are states from the environment and edges encode actionable efficient transitions between nodes.

Existing works (Metzen, 2013; Mannor et al., 2004; Eysenbach et al., 2019; Entezari et al., 2010) have shown benefits of such graph abstractions but typically select nodes only subject to a good coverage the observed state space.

Instead, we identify a parsimonious subset of states that can summarize trajectories and provide more useful intermediate landmarks, i.e. waypoints, for navigating complex environments.

Our method for estimating waypoint states can be viewed as performing automatic (sub)goal discovery.

Subgoal and subpolicy learning are two major approaches to identify a set of temporally-extended actions, "skills", that allow agents to efficiently learn to solve complex tasks.

Subpolicy learning identifies policies useful to solve RL tasks, such as option-based methods (Daniel et al., 2016; Bacon et al., 2017) and subtask segmentations (Pertsch et al., 2019; Kipf et al., 2018) .

Subgoal learning, on the other hand, identifies "important states" to reach (Şimşek et al., 2005) .

Previous works consider various definitions of "important" states: frequently visited states during successful task completions (Digney, 1998; McGovern & Barto, 2001) , states introducing the most novel information (Goyal et al., 2019) , bottleneck states connecting densely-populated regions (Chen , 2007;Şimşek et al., 2005) , or environment-specific heuristics (Ecoffet et al., 2019) .

Our work draws intuition from unsupervised temporal segmentation (Chatzigiorgaki & Skodras, 2009; Jayaraman et al., 2018) and imitation learning (Abbeel & Ng, 2004; Hussein et al., 2017) .

We define "important" states (waypoints) as the most critical states in recovering action sequences generated by some agent, which indicates that these states contain the richest information about the executed policy (Azar et al., 2019) .

We propose a method for learning a world graph G w , a task-agnostic abstraction of an environment that captures its high-level structure and dynamics.

In this work, the primary use of world graphs is to accelerate reinforcement learning of downstream tasks.

The nodes of G w , denoted by a set of waypoints states s p ∈ V p , are generically "important" for accomplishing tasks within the environment, and therefore useful as starting points for exploration.

Our method identifies such waypoint states from interactions with the environment.

In addition, we embed feasible transitions between nearby waypoint states as the edges of G w .

In this work, we define important states in the context of learning G w (see Section 2 for alternative definitions).

That is, we wish to discover a small set of states that, when used as world graph nodes, concisely summarize the structure and dynamics of the environment.

Below, we describe 1) how to collect state-action trajectories and an unsupervised learning objective to identify world graph nodes, and 2) how the graph's edges (i.e., how to transition between nodes) are formed from trajectories.

The structure and dynamics of an environment are implicit in the state-action trajectories observed during exploration.

To identify world graph nodes from such data, we train a recurrent variational autoencoder (VAE) that, given a sequence of state-action pairs, identifies a subset of the states in the sequence from which the full action sequence can be reconstructed (Figure 2 ).

In particular, the VAE infers binary latent variables that controls whether each state in the sequence is used by the generative decoder, i.e., whether a state is "important" or not.

Binary Latent VAE The VAE consists of an inference, a generative and a prior network.

These are structured as follows: the input to the inference network q φ is a trajectory of state-action pairs observed from the environment τ ={(s t , a t )} T t=0 , with s={s t } T t=0 and a={a t } T t=0 denoting the state and action sequences respectively.

The output of the inference network is the approximated posterior over a sequence z={z t } T t=0 of binary latent variables, denoted as q φ (z|a, s).

The generative network p θ computes a distribution over the full action sequence a using the masked state sequence, where s t is masked if z t =0 (we fix z 0 =z T =1 during training), denoted as p θ (a|s, z).

Finally, a state-conditioned p ψ (z t |s t ) given by the prior network p ψ for each s t encodes the empirical average probability that state s t is activated for reconstruction.

This choice encourages inference to select within a consistent subset of states for use in action reconstruction.

In particular, the waypoint Algorithm 1: Identifying waypoint states V p and learning a goal-conditioned policy π g Result: Waypoint states V p and a goal-conditioned policy π g Initialize network parameters for the recurrent variational inference model V Initialize network parameters for the goal-conditioned policy π g Initialize V p with the initial position of the agent, i.e. V p = {s 0 = (1, 1)} while VAE reconstruction error has not converged do for n ← 1 to N do Sample random waypoint s p ∈ V p Navigate agent to s p and perform T -step rollout using a randow walk policy:

T Navigate agent to s p and perform T -step rollout using π g with goal g n : τ

Re-label π g rewards with action reconstruction error as curiosity bonus:

end Perform policy gradient update of π g using τ π and r π Update V using τ r and τ π Update V p as set of states with largest prior mean αs αs+βs .

end states V p are chosen as the states with the largest prior means and during training, once every few iterations, V p is updated based on the current prior network.

Objective Formally, we optimize the VAE using the following evidence lower bound (ELBO):

To ensure differentiablity, we apply a continuous relaxation over the discrete z t .

We use the Beta distribution p ψ (z t ) = Beta(α t , β t ) for the prior and the Hard Kumaraswamy distribution q ψ (z t |a, z) = HardKuma(α t ,β t ) for the approximate posterior, which resembles the Beta distribution but is outside the exponential family (Bastings et al., 2019) .

This choice allows us to sample 0s and 1s without sacrificing differentiability, accomplished via the stretch-and-rectify procedure (Bastings et al., 2019; Louizos et al., 2017) and the reparametrization trick (Kingma & Welling, 2013) .

Lastly, to prevent the trivial solution of using all states for reconstruction, we use a secondary objective L 0 to regularize the L 0 norm of z at a targeted value µ 0 (Louizos et al., 2017; Bastings et al., 2019) , the desired number of selected states out of T steps, e.g. for when T = 25, we set µ 0 = 5, meaning ideally 5 out of 25 states are activated for action reconstruction.

Another term L T to encourage temporal separation between selected states by targeting the number of 0/1 switches among z at 2µ 0 :

See Appendix A for details on training the VAE with binary z t , including integration of the Hard Kumaraswamy distribution and how to regularize the statistics of z.

Naturally, the latent structure learned by the VAE depends on the trajectories used to train it.

Hence, collecting a rich set of trajectories is crucial.

Here, we propose a strategy to bootstrap a useful set of trajectories by alternately exploring the environment based on the current iteration's V p and updating the VAE and V p , repeating this cycle until the action reconstruction accuracy plateaus (Algorithm 1).

During exploration, we use action replay to navigate the agent to a state drawn from the current iteration's V p .

Although resetting via action replay assumes our underneath environment to be deterministic, in cases where this resetting strategy is infeasible, it may be modified so long as to allow the exploration starting points to expand as the agent discovers more of its environment.

For each such starting point, we collect two rollouts.

In the first rollout, we perform a random walk to explore the nearby region.

In the second rollout, we perform actions using a goal-conditioned policy π g (GCP), setting the final state reached by the random walk as the goal.

Both rollouts are used for trianing the VAE and the latter is also used for training π g .

GCP provides a venue to integrate intrinsic motivation, such as curiosity (Burda et al., 2018; Achiam & Sastry, 2017; Pathak et al., 2017; Azar et al., 2019) to generate more diverse rollouts.

Specifically, we use the action reconstruction error of the VAE as an intrinsic reward signal when training π g .

This choice of curioisty also prevents the VAE from collapsing to the simple behaviors of a vanilla π g .

The final stage is to construct the edges of G w , which should ideally capture the environment dynamics, i.e. how to transition between waypoint states.

Once VAE training is complete and V p is fixed, we collect random walk rollouts from each of the waypoints s p ∈ V p to estimate the underlying adjacency matrix (Biggs, 1993) .

More precisely, we claim a directed edge s p → s q if there exists a random walk trajectory from s p to s q that does not intersect a third waypoint.

We also consider paths taken by π g (starting at s p and setting s q as the goal) and keep the shortest observed path from s p to s q as a world graph edge transition.

We use the action sequence length of the edge transition between adjacent waypoints as the weight of the edge.

As shown experimentally, a key benefit of our approach is the ability to plan over G w .

To navigate from one waypoint to another, we can use dynamic programming (Sutton, 1998; Feng et al., 2004) to output the optimal traversal of the graph.

World graphs present a high-level, task-agnostic abstraction of the environment through waypoints and feasible transition routes between them.

A key example of world graph applications for taskspecific RL is structured exploration: instead of exploring the entire environment, RL agents can use world graphs to quickly identify task-relevant regions and bias low-level exploration to these regions.

Our framework to leverage world graphs for structured exploration consists of two parts:

1.

Hierarchical RL wherein the high-level policy selects subgoals from V p .

2.

Traversals using world graph edges.

Formally, an RL agent learning to solve a task is formulated as a Markov Decision Process: at time t, the agent is in a state s t , executes an action a t via a policy π(a t |s t ) and receives a rewards r t .

The agent's goal is to maximize its cumulative expected return R = E (st,at)∼π,p,p0 t≥0 γ t r t , where p(s t+1 |s t , a t ), p 0 (s 0 ) are the transition and initial state distributions.

To incorporate world graphs with RL, we use a hierarchical approach based on the Feudal Network (FN) (Dayan & Hinton, 1993; Vezhnevets et al., 2017) , depicted in Figure 3 .

A standard FN

Collect randomly spawned balls, each ball gives +1 reward.

To end an episode, the agent has to exit at a designated point.

Balls are located randomly, dense reward.

Agents receive a single reward r ≤ 1 proportional to the number of balls collected upon exiting.

Balls are located randomly, sparse reward.

Spawn lava blocks at random locations each time step that immediately terminates the episode if stepped on.

Stochastic environment.

Multiple objects: lava and balls are randomly located, dense reward.

Agent has to pick up a key to open a door (reward +1) and reach the exit point on the other side (reward +1).

Walls, door and key are located randomly.

Agents have additional actions: pick and toggle.

Table 1 : An overview of tasks used to evaluate the benefit of using world graphs.

Visualizations can be found in Appendix D.

decomposes the policy of the agent into two separate policies that receive distinct streams of reward: a high-level policy ("Manager") learns to propose subgoals; a low-level policy ("Worker") receives subgoals from the Manager as inputs and is rewarded for taking actions in the environment that reach the subgoals.

The Manager receives the environment reward defined by the task and therefore must learn to emit subgoals that lead to task completion.

The Manager and Worker do not share weights and operate at different temporal resolutions: the Manager only outputs a new subgoal if either the Worker reaches the chosen one or a subgoal horizon c is exceeded.

For all our experiments, policies are trained using advantage actor-critic (A2C), an on-policy RL algorithm (Wu & Tian, 2016; Pane et al., 2016; Mnih et al., 2016b) .

To ease optimization, the feature extraction layers of the Manager and Worker that encode s t are initialized with the corresponding layers from π g , the GCP learned during world graph discovery phase.

More details are in Appendix B.

To incorporate the world graph, we introduce a Manager policy that factorizes subgoal selection as follows: a wide policy π w (g w t |s t ) selects a waypoint state as the wide goal g w ∈ V p , and a narrow policy π n (g n t |s t , g

The wide-then-narrow subgoal format simplifies the search space for the Manager policy.

Using waypoints as wide goals also makes it possible to leverage the edges of the world graph for planning and executing the planned traversals.

This process breaks down as follows:

1.

When to Traverse: When the agent encounters a waypoint state s t ∈ V p , a "traversal" is initiated if s t has a feasible connection in G w to the active wide goal g w t .

2.

Planning: Upon triggering a traversal, the optimal traversal route from the initiating state to g w t is estimated from the G w edge weights using classic dynamic programming planning (Sutton, 1998; Feng et al., 2004) .

This yields a sequence of intermediate waypoint states.

3. Execution: Execution of graph traversals depends on the nature of the environment.

If deterministic, the agent simply follows the action sequences given by the edges of the traversal.

Otherwise, the agent uses the pretrained GCP π g to sequentially reach each of the intermediate waypoint states along the traversal (we fine-tune π g in parallel where applicable).

If the agent fails to reach the next waypoint state within a certain time limit, it stops its current pursuit and a new (g w , g n ) pair is received from the Manager.

World graph traversal allows the Manager to assign task-relevant wide goals g w that can be far away from the agent yet still reachable, which consequentially accelerates learning by focusing exploration around the task-relevant region near g w .

We now assess each component of our framework on a set of challenging 2D grid worlds.

Our ablation studies demonstrate the following benefits of our framework: • It improves sample efficiency and performance over the baseline HRL model.

• It benefits tasks varying in envirionment scale, task type, reward structure, and stochasticity.

•

The identified waypoints provide superior world representations for solving downstream tasks, as compared to graphs using randomly selected states as nodes.

Implementation details, snippets of the tasks and mazes are in Appendix C-D.

For our ablation studies, we construct 2D grid worlds of increasing sizes (small, medium and large) along with challenging tasks with different reward structures, levels of stochasticity and logic (summarized in Table 1 ).

In all tasks, every action taken by the agent receives a negative reward penalty.

We follow a rigorous evaluation protocol (Wu et al., 2017; Ostrovski et al., 2017; Henderson et al., 2018) : each experiment is repeated with 3 training seeds.

10 additional validation seeds are used to pick the model with the best reward performance.

This model is then tested on 100 testing seeds.

We report mean reward and standard deviation.

We ablate each of the following components in our framework and compare against non-hierarchical (A2C) and hierarchical baselines (FN):

1. initializing the feature extraction layers of the Manager and Worker from π g , 2. applying Wide-then-Narrow Manager (WN) goal instruction, and 3. allowing the Worker to traverse along G w .

Results are shown in Table 2 .

In sum, each component improves performance over the baselines.

Wide and narrow goals Using two goal types is a highly effective way to structure the Manager instructions and enables the Worker to differentiate the transition and local task-solving phases.

We note that for small MultiGoal, agents do not benefit much from G w traversal: it can rely solely on the guidance from WN goals to master both phases.

However with increasing maze size, the Worker struggles to master traversals on its own and thus fails solving the tasks.

World Graph Traversal As conjectured in Section 4.3, the performance gain of our framework can be explained by the larger range and more targeted exploration strategy.

In addition, the Worker We see that 1) traversal speeds up convergence, 2) V rand gives higher variance and slightly worse performance than Vp.

Right: comparing with or without πg initialization on Vp, all models use WN.

We see that initializing the task-specific phase with the task-agnostic goal-conditioned policy significantly boosts learning.

does not have to learn long distance transitions with the aid of G w traversals.

Figure 4 confirms that G w traversal speeds up convergence and its effect becomes more evident with larger mazes.

Note that the graph learning stage only need 2.4K iterations to converge.

Even when taking these additional environment interactions into account, G w traversal still exhibits superior sample efficiency, not to mention that the graph is shared among all tasks.

Moreover, solving Door-Key involves a complex combination of sub-tasks: find and pick up the key, reach and open the door and finally exit.

With limited reward feedback, this is particularly difficult to learn.

The ability to traverse along G w enables longer-horizon planning on top of the waypoints, thanks to which the agents boost the success rate on medium Door-Key from 0.56±0.02 to 0.75±0.06.

To highlight the benefit of establishing the waypoints learned by the VAE as nodes for G w , we compare against results using a G w constructed around randomly selected states (V rand ).

The edges of the random-node graph are formed in the same way as described in Section 3.3 and its feature extractor is also initialized from π g .

Although granting knowledge acquired during the unsupervised phase to V rand is unfair to V p , deploying both initialization and traversal while only varying V rand and V p isolates the effect from the nodes to the best extent.

The comparative results (in Table 3 , learning curves for MultiGoal in Figure 4 ) suggest V p generally outperforms V rand .

Door-Key is the only task in which the two matches.

However, V rand exhibits a large variance, implying that certain sets of random states can be suitable for this task, but using learned waypoints gives strong performance more consistently.

Initialization with GCP Initializing the weights of the Worker and Manager feature extractors from π g (learned during the task-agnostic phase) consistently benefits learning.

In fact, we observe that models starting from scratch fail on almost all tasks within the maximal number of training iterations, unless coupled with G w traversal, which is still inferior to using π g -initialization.

Particularly, for the small MultiGoal-Stochastic environment, there is a high chance that a lava square blocks traversal; therefore, without the environment knowledge from π g transferred by weight initialization, the interference created by the episode-terminating lava prevents the agent from learning the task.

We have shown that world graphs are powerful environment abstractions, which, in particular, are capable of accelerating reinforcement learning.

Future works may extend their applications to more challenging RL setups, such as real-world multi-task learning and navigation.

It is also interesting to generalize the proposed framework to learn dynamic world graphs for evolving environments, and applying world graphs to multi-agent problems, where agents become part of the world graphs of other agents.

As illustrated in the main text, the main objective for the recurrent VAE is the following evidence lower bound with derivation:

log p(a|s) = log p(a|s, z)dz = log p(a|s, z)p(z|s) q(z|a, s) q(z|a, s) dz = log p(a|s, z) p(z|s) q(z|a, s) q(z|a, s)dz

The inference network q ψ takes in the trajectories of state-action pairs τ and at each time step approximates the posterior of the corresponding latent variable z t .

The prior network p ψ takes the state s t at each time step and outputs the state-conditioned prior p ψ (s t ).

We choose Beta as the prior distribution and the Hard Kuma as the approximated posterior to relax the discrete latent variables to continuous surrogates.

The Kuma distribution Kuma(α, β) highly resembles the Beta Distribution in shape but does not come from the exponential family.

Similar to Beta, the Kuma distribution also ranges from bimodal (when α ≈ β) to unimodal (α/β → 0 or α/β → ∞).

Also, when α = 1 or β = 1, Kuma(α, β) = Beta(α, β).

We observe empirically better performance when we fix β = 1 for the Kuma approximated posterior.

One major advantage of the Kuma distribution is its simple Cumulative Distribution Function (CDF):

It is therefore amendable to the reparametrization trick (Kingma & Welling, 2013; Rezende et al., 2014; Maddison et al., 2016) by sampling from uniform distribution u ∼ U(0, 1):

Lastly, the KL-divergence between the Kuma and Beta distributions can be approximated in closed form (Nalisnick & Smyth, 2016) :

where Ψ is the Digamma function, γ the Euler constant, and the approximation uses the first few terms of the Taylor series expansion.

We take the first 5 terms here.

Next, we make the Kuma distribution "hard" by following the steps in Bastings et al. (2019) .

First stretch the support to (r = 0 − 1 , l = 1 + 2 ), 1 , 2 > 0, and the resulting CDF distribution takes the form:

Then, the non-eligible probabilities for 0's and 1's are attained by rectifying all samples below 0 to 0 and above 1 to 1, and other value as it is, that is

Lastly, we impose two additional regularization terms L and L T on the approximated posteriors.

As described in the main text, L prevents the model from selecting all states to reconstruct {a t }

by restraining the expected L 0 norm of z = (z 1 · · · z T −1 ) to approximately be at a targeted value µ 0 (Louizos et al., 2017; Bastings et al., 2019) .

In other words, this objective adds the constraint that there should be µ 0 of activated z t = 1 given a sequence of length T .

The other term L T encourages temporally isolated activation of z t , meaning the number of transition between 0 and 1 among z t 's should roughly be 2µ 0 .

Note that both expectations in Equation 2 have closed forms for HardKuma.

Lagrangian Relaxation.

The overall optimization objective consists of action sequence reconstruction, KL-divergence between the posterior and prior, L 0 and L T (Equation 12).

We tune the objective weights λ i using Lagrangian relaxation (Higgins et al., 2017; Bastings et al., 2019; Bertsekas, 1999) , treating λ i 's as learnable parameters and performing alternative optimization between λ i 's and the model parameters.

We observe that as long as their initialization is within a reasonable range, λ i 's converge to a local optimum:

We observe this approach to produce efficient and stable mini-batch training.

Optimizing composite neural networks like HRL (Co-Reyes et al., 2018 ) is sensitive to weight initialization (Mishkin & Matas, 2015; Le et al., 2015) , due to its complexity and lack of clear supervision at various levels.

Therefore, taking inspiration from prevailing pre-training procedures in computer vision (Russakovsky et al., 2015; Donahue et al., 2014) and NLP (Devlin et al., 2018; Radford et al., 2019) , we take advantage of the weights learned by π g during world graph discovery when initializing the Worker and Manager policies for downstream HRL, as π g has already implicitly embodied much environment dynamics information.

More specifically, we extract the weights of the feature extractor, i.e. the state encoder, and use them as the initial weights for the state encoders of the HRL policies.

Our empirical results demonstrate that such weight initialization consistently improves performance and validates the value of skill/knowledge transfer from GCP (Taylor & Stone, 2009; Barreto et al., 2017) .

Model code folder including all architecture details is shared in comment.

Our models are optimized with Adam (Kingma & Ba, 2014) using mini-batches of size 128, thus spawning 128 asynchronous agents to explore.

We use an initial learning rate of 0.0001, with = 0.001, β 1 = 0.9, β 2 = 0.999; gradients are clipped to 40 for inference and generation nets.

For HardKuma, we set l = −0.1 and r = 1.1.

The maximum sequence length for BiLSTM is 25.

The total number of training iterations is 3600 and model usually converges around 2400 iterations.

We train the prior, inference, and generation networks end-to-end.

We initialize λ i 's (see Lagrangian Relaxation) to be λ 1 = 0.01 (KL-divergence),λ 2 = 0.06 (L 0 ), λ 3 = 0.02 (L T ).

After each update of the latent model, we update λ i 's, whose initial learning rate is 0.0005, by maximizing the original objective in a similar way as using Lagrangian Multiplier.

At the end of optimization, λ i 's converge to locally optimal values.

For example, with the medium maze, λ 1 = 0.067 for the KL-term, λ 2 = 0.070 for the L 0 and λ 3 = 0.051 for the L T term.

The total number of waypoints |V p | is set to be 20% of the size of the full state space.

The procedure of the Manager and the Worker in sending/receiving orders using either traversal paths among V p from replay buffer for deterministic environments or with π g for stochastic ones follows:

1.

The Manager gives a wide-narrow subgoal pair (g w , g n ).

2.

The agent takes action based on the Worker policy π ω conditioned on (g w , g n ) and reaches a new state s .

If s ∈ V p , g w has not yet met, and there exists a valid path basing on the edge paths from the world graph s → g w , agent then either follows replay actions or π g to reach g w .

If π g still does not reach desired destination in a certain steps, then stop the agent wherever it stands; also π g can be finetuned here.

3.

The Worker receives positive reward for reaching g w for the first time.

4.

If agent reaches g n , the Worker also receives positive rewards and terminates this horizon.

5.

The Worker receives negative for every action taken except for during traversal; the Manager receives negative reward for every action taken including traversal.

6.

When either g n is reached or the maximum time step for this horizon is met, the Manager renews its subgoal pair.

The training of the Worker policy π ω follows the same A2C algorithm as π g .

The training of the Manager policy π m also follows a similar procedure but as it operates at a lower temporal resolution, its value function regresses against the t m -step discounted reward where t m covers all actions and rewards generated from the Worker.

When using the Wide-then-Narrow instruction, the policy gradient for the Manager policy π m becomes: E (st,at)∼π,p,p0 [A m,t ∇ log (π ω (g w,t |s t ) π n (g n,t |s t , g w,t , s w,t ))] + ∇ [H (π ω ) + H (π n (·|g w,t ))] , where A m,t is the Manager's advantage at time t. Also, for Manager, as the size of the action space scales linearly with |S|, the exact entropy for the π m can easily become intractable.

Essentially there are O |V| × (N 2 ) possible actions.

To calculate the entropy exactly, all of them has to be summed, making it easily computationally intractable: H = w∈V wn∈sw π n (w n |s w , s t )π ω (w|s t ) log ∇π n (w n |s w , s t )π ω (w|s t ).

Thus in practice we resort to an effective alternative H (π ω ) + H (π n (·|g w,t )).

Psuedo-code for Manager training is in Algorithm 2.

For training the HRL policies, we inherit most hyperparameters from those used when training π g , as the Manager and the Worker both share similar architectures with π g .

The hyperparameters used when training π g follow those from Shang et al. (2019) .

Because the tasks used in HRL experiments are more difficult than the generic goal-reaching task, we set the maximal number of training iterations to 100K abd training is stopped early if model performance reaches a plateau.

The rollout steps for each iteration is 60.

Hyperparameters specific to HRL are the horizon c = 20 and the size of the Manager's local attention range (that is, the neighborhood around g w within which g n is selected), which are N = 5 for small and medium mazes, and N = 7 for the large maze.

@highlight

We learn a task-agnostic world graph abstraction of the environment and show how using it for structured exploration can significantly accelerate downstream task-specific RL.
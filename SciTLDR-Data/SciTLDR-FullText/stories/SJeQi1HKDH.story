Animals develop novel skills not only through the interaction with the environment but also from the influence of the others.

In this work we model the social influence into the scheme of reinforcement learning, enabling the agents to learn both from the environment and from their peers.

Specifically, we first define a metric to measure the distance between policies then quantitatively derive the definition of uniqueness.

Unlike previous precarious joint optimization approaches, the social uniqueness motivation in our work is imposed as a constraint to encourage the agent to learn a policy different from the existing agents while still solve the primal task.

The resulting algorithm, namely Interior Policy Differentiation (IPD), brings about performance improvement as well as a collection of policies that solve a given task with distinct behaviors

The paradigm of Reinforcement Learning (RL), inspired by cognition and animal studies (Thorndike, 2017; Schultz et al., 1997) , can be described as learning by interacting with the environment to maximize a cumulative reward (Sutton et al., 1998) .

From the perspective of ecology, biodiversity as well as the development of various skills are crucial to the continuation and evolution of species (Darwin, 1859; Pianka, 1970) .

Thus the behavioral diversity becomes a rising topic in RL.

Previous works have tried to encourage the emergence of behavioral diversity in RL with two approaches: The first approach is to design interactive environments which contain sufficient richness and diversity.

For example, Heess et al. (2017) show that rich environments enable agents to learn different locomotion skills even using the standard RL algorithms.

Yet designing a complex environment requires manual efforts, and the diversity is limited by the obstacle classes.

The second approach to increase behavioral diversity is to motivate agents to explore beyond just maximizing the reward for the given task.

Zhang et al. (2019) proposed to maximize a heuristically defined novelty metric between policies through task-novelty joint optimization, but the final performance of agents is not guaranteed.

In this work, we address the topic of policy differentiation in RL, i.e., to improve the diversity of RL agents while keeping their ability to solve the primal task.

We draw the inspiration from the Social Influence in animal society (Rogoff, 1990; Ryan & Deci, 2000; van Schaik & Burkart, 2011; Henrich, 2017; Harari, 2014) and formulate the concept of social influence in the reinforcement learning paradigm.

Our learning scheme is illustrated in Fig 1.

The target agent not only learns to interact with the environment to maximize the reward but also differentiate the actions it takes in order to be different from other existing agents.

Since the social influence often acts on people passively as a sort of peer pressure, we implement the social influence in terms of social uniqueness motivation (Chan et al., 2012) and consider it as a constrained optimization problem.

In the following of our work, we first define a rigorous policy distance metric in the policy space to compare the similarity of the agents.

Then we develop an optimization constraint using the proposed metric, which brings immediate rather than episodic feedback in the learning process.

A novel method, namely Interior Policy Differentiation (IPD), is further I should learn to run as fast as I can I should try to be different Figure 1 : The illustration of learning with social influence.

Instead of focusing only on the primal task, an additional constraint is introduced to the target agent, motivating it to not only perform well in the primal task but also take actions differently to other existing agents.

proposed as a better solution for the constrained policy optimization problem.

We benchmark our method on several locomotion tasks and show it can learn various diverse and well-behaved policies for the given tasks based on the standard Proximal Policy Optimization (PPO) algorithm (Schulman et al., 2017) .

Intrinsic motivation methods.

The Variational Information Maximizing Exploration (VIME) method is designed by Houthooft et al. (2016) to tackle the sparse reward problems.

In VIME, an intrinsic reward term based on the maximization of information gains is added to contemporary RL algorithms to encourage exploration.

The curiosity-driven methods, proposed by Pathak et al. (2017) and Burda et al. (2018a) define intrinsic rewards according to prediction errors of neural networks.

i.e., when taking previous unseen states as inputs, networks trained with previous states will tend to predict with low accuracy, so that such prediction errors can be viewed as rewards.

Burda et al. (2018b) proposed Random Network Distillation (RND) to quantify intrinsic reward by prediction differences between a fixed random initialized network and another randomly initialized network trained with previous state information.

Liu et al. (2019) proposed Competitive Experience Replay (CER), in which they use two actors and a centralized critic, and defined an intrinsic reward by the state coincidence of two actors.

The values of intrinsic rewards are fixed to be ±1 for the two actors separately.

All of those approaches leverage the weighted sum of the external rewards, i.e., the primal rewards provided by environments, and intrinsic rewards that provided by different heuristics.

A challenging problem is the trade-off between external rewards and intrinsic rewards.

The Task-Novelty Bisector (TNB) learning method introduced by Zhang et al. (2019) aims to solve such problem by jointly optimize the extrinsic rewards and intrinsic rewards.

Specifically, TNB updates the policy in the direction of the angular bisector of the two gradients, i.e., gradients of the extrinsic and intrinsic objective functions.

However, the foundation of such joint optimization is not solid.

Besides, creating an extra intrinsic reward function and evaluating the novelty of states or policies always requires additional neural networks such as auto-encoders.

Thus extra computation expenses are needed (Zhang et al., 2019) .

Diverse behaviors from rich environments and algorithms.

Heess et al. (2017) introduce the Distributed Proximal Policy Optimization (DPPO) method and enable agents with simulated bodies to learn complex locomotion skills in a diverse set of challenging environments.

Although the learning reward they utilize is straightforward, the skills their policy learned are quite impressive and effective in traveling terrains and obstacles.

Their work shows that rich environments can encourage the emergence of different locomotion behaviors, but extra manual efforts are required in designing such environments.

The research of Such et al. (2018) shows that different RL algorithms may converge to different policies for the same task.

The authors find that algorithms based on policy gradient tend to converge to the same local optimum in the game of Pitfall, while off-policy and value-based algorithms are prone to learn sophisticated strategies.

On the contrary, in this paper, we are more interested in how to learn different policies through a single learning algorithm and learn the capability of avoiding local optimum. (Schulman et al., 2015) , Kurutach et al. (2018) maintains model uncertainty given the data collected from the environment via an ensemble of deep neural networks.

To encourage the emergence of behavioral diversity in RL, we first define a metric to measure the difference between policies, which is the foundation for the later algorithm we propose.

We denote the learned policies as {π θi ; θ i ∈ Θ, i = 1, 2, ...}, wherein θ i represents parameters of the i-th policy, Θ denotes the whole parameter space.

In the following, we omit π and denote a policy π θi as θ i for simplicity unless stated otherwise.

Mathematically, a metric should satisfy three important properties, namely the identity, the symmetry as well as the triangle inequality.

Definition 1 A metric space is an ordered pair (M, d) where M is a set and d is a metric on M , i.e., a function d : M × M → R such that for any x, y, z ∈ M , the following holds:

We use the Total Variance Divergence D T V (Schulman et al., 2015) to measure the distance between policies.

Concretely, for discrete probability distributions p and q, this distance is defined as

is a metric on Θ, thus (Θ, D ρ T V ) is a metric space.

2003; Fuglede & Topsoe, 2004; Villani, 2008) , and similar results can be get

3 It can be extended to continuous state and action spaces by replacing the sums with integrals.

4 The factor 1 2

in Schulman et al. (2015) is omitted in our work for conciseness.

Consequently, to motivate RL with the social uniqueness, we hope our method can maximize the uniqueness of a new policy, i.e., max θ U(θ|Θ ref ), where the Θ ref includes all the existing policies.

In practice, the calculation of D ρ T V (θ i , θ j ) is based on Monte Carlo estimation.

i.e., we need to sample s from ρ(s).

Although in finite state space we can get precise estimation after establishing ergodicity, problem arises when we are facing continuous state cases.

i.e. it is difficult to efficiently get enough samples.

Formally, we denote the domain of ρ(s) as S and denote the domain of ρ θ (s) as S θ ⊂ S, where ρ θ (s) := ρ(s|s ∼ θ) and in finite time horizon problems ρ(s|s ∼ θ) = P (s 0 = s|θ) + P (s 1 = s|θ) + ... + P (s T = s|θ).

As we only care about the reachable regions, the domain S can be divided

In order to improve the sample efficiency, we propose to approximate

, where θ is a certain fixed behavior policy that irrelevant to θ i , θ j .

Such approximation requires a necessary condition:

The domain of possible states are similar between different policies:

When such condition holds, we can use ρ(s|s ∼ θ) as our choice of ρ(s), and the properties in Definition 1 still holds.

In practice, the Condition 1 always holds as we can ensure this by adding sufficiently large noise on θ, while the permitted state space is always limited.

And for more general cases, to satisfy the properties in Definition 1, we must sample s from S θ ∪ S θj , accordingly,

where N represents random action when a policy have never been trained or visited such state domain.

Plugging Eq.(4) into Eq. (2), the objective function of policy differentiation is

While the first two terms are related to the policy θ, the last term is only related to the domain S θ .

If we enable sufficient exploration in training as well as in the initialization of θ, the last term will disappear (i.e. S θj ⊂ S θ ).

Hence we can also use D

Proposition 1 (Unbiased Single Trajectory Estimation) The estimation of ρ θ (s) using a single trajectory τ is unbiased.

The proof of Proposition 1 is in Appendix B. Given the definition of uniqueness and a practically unbiased sampling method, the next step is to develop an efficient learning algorithm.

In the traditional RL paradigm, maximizing the expectation of cumulative rewards g = t=0 γ t r t is commonly used as the objective.

i.e. max θ∈Θ E τ ∼θ [g] , where τ ∼ θ denotes a trajectory τ sampled from the policy θ using Monte Carlo methods.

To improve the behavioral diversity of different agents, the learning objective must take both reward from the primal task and the policy uniqueness into consideration.

Previous approaches (Houthooft et al., 2016; Pathak et al., 2017; Burda et al., 2018a; b; Liu et al., 2019) often directly write the weighted sum of the reward from the primal task and the intrinsic reward g int = t=0 γ t r int,t , where r int,t denotes the intrinsic reward (e.g.,

as the uniqueness reward in our case) as follows,

where 0 < α < 1 is a weight parameter.

Such an objective is sensitive to the selection of α as well as the formulation of r int .

For example, in our case formulating the intrinsic reward r int as

will result in significantly different results.

Besides, a trade-off arises in the selection of α: while a large α may undermine the contribution of intrinsic reward, a small α could ignore the importance of the reward, leading to the failure of agent in solving the primal task.

To tackle these issues, we draw inspiration from the observation that social uniqueness motivates people in passive ways.

In other words, it plays more like a constraint rather than an additional target.

Therefore, we change the multi-objective optimization problem in Eq.(6) into a constrained optimization problem as:

where r 0 is a threshold indicating minimal permitted uniqueness, and r int,t denotes a moving average of r int,t .

Further discussion on the selection of r 0 will be deliberated in Appendix D.

From the perspective of optimization, Eq.(6) can be viewed as a penalty method which replaces the constrained optimization problem in Eq.(7) with the penalty term r int and the penalty coefficient 1−α α > 0, where the difficulty lies in the selection of α.

The work of Zhang et al. (2019) ) tackles this challenge by the Task Novel Bisector (TNB) in the form of Feasible Direction Methods (FDMs) (Zoutendijk, 1960) .

As a heuristic approximation, that approach requires reward shaping and intensive emphasis on r int,t .

Instead, in this work we propose to solve the constrained optimization problem Eq.(7) by resembling the Interior Point Methods (IPMs) (Potra & Wright, 2000; Dantzig & Thapa, 2006) .

In vanilla IPMs, the constrained optimization problem in Eq. (7) is solved by reforming it to an unconstrained form with an additional barrier term in the objective as

The limit of Eq.(8) when α → 0 then leads to the solution of Eq.(7).

Readers please refer to Appendix G for more discussion on the correspondence between those novel policy seeking methods and constrained optimization methods.

However, directly applying the IPMs is computationally challenging and numerically unstable, especially when α is small.

Luckily, in our proposed RL paradigm where the behavior of an agent is influenced by its peers, a more natural way can be used.

Precisely, since the learning process is based on sampled transitions, we can simply bound the collected transitions in the feasible region by permitting previous trained M policies θ i ∈ Θ ref , i = 1, 2, ..., M sending termination signals during the training process of new agents.

In other words, we implicitly bound the feasible region by terminating any new agent that steps outside it.

Consequently, during the training process, all valid samples we collected are inside the feasible region, which means these samples are less likely to appear in previously trained policies.

At the end of the training, we then naturally obtain a new policy that has sufficient uniqueness.

In this way, we no longer need to consider the trade-off problem between intrinsic and extrinsic rewards deliberately.

The learning process of our method is thus more robust and no longer suffer from objective inconsistency.

As our formulation of the constrained optimization problem Eq. (7) is inspired by IPMs, we name our approach as Interior Policy Differentiation (IPD) method.

The MuJoCo environment We demonstrate our proposed method on the OpenAI Gym where the physics engine is based on MuJoCo (Brockman et al., 2016; Todorov et al., 2012) .

Concretely, we test on three locomotion environments, the Hopper-v3 (11 observations and 3 actions), Walker2d-v3 (11 observations and 2 actions), and HalfCheetah-v3 (17 observations and 6 actions).

In our experiments, all the environment parameters are set as default values.

Uniqueness beyond intrinsic stochasticity Experiments in Henderson et al. (2018) show that policies that perform differently can be produced by simply selecting different random seeds before training.

Before applying our method to improve behavior diversity, we firstly benchmark how much uniqueness can be generated from the stochasticity in the training process of vanilla RL algorithms as well as the random weight initialization.

In this work, we mainly demonstrate our proposed method based on PPO (Schulman et al., 2017) .

The extension to other popular algorithms is straightforward.

We also compare our proposed method with the TNB and weighted sum reward (WSR) approaches as different ways to combine the goal of the task and the uniqueness motivation (Zhang et al., 2019) .

More implementation details are depicted in Appendix D.

According to Theorem 2, the uniqueness r int in equation (7) under our uniqueness metric can be unbiased approximated by

i.e., we utilize the metric directly in learning new policies instead of applying any kind of reshaping.

We implement WSR, TNB, and our method in the same experimental settings and for each method, 10 different policies are trained and try to be unique with regard to all previously trained policies sequentially.

Concretely, the 1st policy is trained by ordinary PPO without any social influence.

The 2nd policy should be different from 1st policy, and the 3rd should be different from the previous two policies, and so on.

Fig.2 shows the qualitative results of our method.

We visualize the motion of agents by drawing multiple frames representing the pose of agents at different time steps in the same row.

The horizontal interval between consecutive frames is proportional to the velocity of agents.

The settings of the frequency of highlighted frames and the correlation between interval and velocity are fixed for each environment.

The visualization starts from the beginning of each episode and therefore the readers can get sense of the process of acceleration as well as the pattern of motion of agents clearly.

Fig. 3 shows our experimental results in terms of uniqueness (the x-axis) and the performance (the y-axis).

Policies in the upper right are the more unique ones with higher performance.

In Hopper and HalfCheetah, our proposed method distinctively outperforms other methods.

In Walker2d, both WSR and our method work well in improving the uniqueness of policies, but none of the three methods can find way to surpass the performance of PPO apparently.

Detailed comparison on the task related rewards are carried out in Table 1 .

A box figure depicting the performance of each trained policy and their reward gaining curve are disposed in Fig.5 and Fig.6 in Appendix C. And Fig.7 in Appendix C provides more detailed results from the view of uniqueness.

In addition to averaged reward, we also use success rate as another metrics to compare the performance of different approaches.

In this work, we consider a policy is success when its performance is at least as good as the averaged performance of policies trained without social influences.

To be specific, we use the averaged final performance of PPO as the baseline.

If a new policy, which aims at performing differently to solve the same task, surpasses the baseline during its training process, it will be regarded as a successful policy.

Through the success rate, we know the policy does not learn unique behavior at the expense of performance.

Table 1 shows the success rate of all the methods, including the PPO baseline.

The results show that our method can always surpass the average baseline during training.

Thus the performance of our method can always be insured.

In our experiments, we observed noticeable performance improvements in the Hopper and the HalfCheetah environments.

For the environment of Hopper, in many cases, the agents trained with PPO tend to learn a policy that jumps as far as possible and then fall to the ground and terminate this episode (please refer to Fig.11 in Appendix E).

Our proposed method can prevent new policies from always falling into the same local minimum.

After the first policy being trapped in a local minimum, the following policies will try other approaches to avoid the same behavior, explore other feasible action patterns, and thereafter the performance may get improved.

Such property shows that our method can be a helpful enhancement of the traditional RL scheme, which can be epitomized as policies could make mistakes, but they should explore more instead of hanging around the same local minimum.

The similar feature attributes to the reward growth in the environment of HalfCheetah.

Moreover, we can illuminate the performance improvement of HalfCheetah from another perspective.

The environment of HalfCheetah is quite different from the other two for there is no explicit termination signal in its default settings (i.e., no explicit action like falling to the ground would trigger termination).

At the beginning of the learning process, an agent will act randomly, resulting in massive repeat, trivial samples as well as large control costs.

In our learning scheme, since the agent also interacts with the peers, it can receive termination signals from the peers to prevent wasting too much effort acting randomly.

During the learning process in our method, an agent will first learn to terminate itself as soon as possible to avoid heavy control costs by imitating previous policies and then learns to behave differently to pursue higher reward.

From this point of view, such learning process can be regarded as a kind of implicit curriculum.

As the number of policies learned with social influence grows, the difficulty of finding a unique policy may also increase.

Later policies must keep away from all previous solutions.

The results of our ablation study on how the performance changes under different scales of social influence (i.e., the number of peers) is shown in Fig. 4 , where the thresholds are selected according to our previous ablation study in Sec. D. The performance decrease is more obvious in Hopper than the other two environments for the action space of Hopper is only 3 dimensional.

Thus the number of possible diverse policies can be discovered is limited.

In this work, we develop an efficient approach to motivate RL to learn diverse strategies inspired by social influence.

After defining the distance between policies, we introduce the definition of policy uniqueness.

Regarding the problem as constrained optimization problem, our proposed method, Interior Policy Differentiation (IPD), draws the key insight of the Interior Point Methods.

And our experimental results demonstrate IPD can learn various well-behaved policies, and our approach can help agents to avoid local minimum and can be interpreted as a kind of implicit curriculum learning in certain cases.

The first two properties are obviously guaranteed by D ρ T V .

As for the triangle inequality, Figure 7: Maximal and minimal between policy uniqueness in Hopper, Walker2d and HalfCheetah environments.

The results are averaged over all possible combinations of 10 policies.

As TNB and WSR optimize the uniqueness reward directly, their uniqueness sometimes can exceed our proposed method.

However, such direct optimization will lead to decreasing in task related performance as cost.

To tackle the trade-off problem, carefully hyper-parameter tuning and reward shaping is always a must.

Detailed comparison on the task related rewards are carried out in Table 1 D IMPLEMENTATION DETAILS Calculation of D T V We use deterministic part of policies in the calculation of D T V , i.e., we remove the Gaussian noise on the action space in PPO and use

Network Structure We use MLP with 2 hidden layers as our actor models in PPO.

The first hidden layer is fixed to have 32 units.

Our ablation study on the choice of unit number in the second layer is detailed in Table.

2, Table3 and Fig.8 .

Moreover, we choose to use 10, 64 and 256 hidden units for the three tasks respectively in all of the main experiments, after taking the success rate (Table.

2), performance (Table.

3) and computation expense (i.e. the preference to use less unit when the other two factors are similar) into consideration.

Training Timesteps We fix the training timesteps in our experiments.

The timesteps are fixed to be 1M in Hopper-v3, 1.6M for Walker2d-v3 and 3M for HalfCheetah.

Threshold Selection In our proposed method, we can control the magnitude of policy uniqueness flexibly by adjusting the constraint threshold r 0 .

Choosing different thresholds will lead to different policy behaviors.

Concretely., a larger threshold may drive the agent to perform more differently while smaller threshold imposes a lighter constraint on the behavior of the agent.

Intuitively, a larger threshold will lead to relatively poor performance for the learning algorithm is less likely to find a feasible solution to Eq.(7).

Besides, we do not use constraints in the form of Eq. (7) as we need not force every single action of a new agent to be different from others.

Instead, we are more care about the long term differences.

Therefore, we use the cumulative uniqueness as constraints,

We test our method with different choices of threshold values.

The performance of agents under different thresholds are shown in Fig. 9 and more detailed analysis of their success rate is presented in Table.

2.

F IMPLEMENTATION OF EQ.

(7) We do not use constraints in the form of Eq.(7) as we need not force every single action of a new agent to be different from others.

Instead, we are more care about the long term differences.

Therefore, we use the cumulative uniqueness as constraints.

Moreover, the constraints can be applied after the first t S timesteps (e.g. t S = 20) for the consideration of similar starting sequences.

We note here, the WSR, TNB and IPD methods correspond to three approaches in constrained optimization problem.

For simplicity, we consider Eq.(9) with a more concise notion g int,t − g 0,t ≥ 0, where g int,t = t t=0 r int,t , i.e., max

As the optimization of policy is based on batches of trajectory samples and is implemented with stochastic gradient descent, Eq.(10) can be further simplified as:

where g t (θ) denotes the average over a trajectory.

The Penalty Method considers the constraints of Eq.(11) by putting constraint g(θ) into a penalty term, and then solve the unconstrained problem

using an iterative manner, and the limit when α → 0 lead to the solution of the primal constrained problem.

As an approximation, WSR choose a fixed weight term α, and use the gradient of ∇ θ f + 1−α α ∇ θ g instead of ∇ θ f + 1−α α ∇ θ min{g(θ), 0}, thus the final solution will intensely rely on the selection of α.

The Taylor series of g(θ) at pointθ is g(θ + λ p) = g(θ) + ∇ θ g(θ)

T λ p + O(||λ p||)

The Feasible Direction Method (FDM) considers the constraints of Eq.(11) by first finding a direction p satisfies

so that for small λ, we have

and g(θ + λ p) = g(θ) + λ∇ θ g(θ) T p > 0 if g(θ) > 0

The TNB method, by using the bisector of gradients ∇ θ f and ∇ θ g, select p to be

Clearly, Eq.(17) satisfies Eq. (14), but it is more strict than Eq. (14) as the ∇ θ g term always exists during the optimization of TNB.

In TNB, the learning stride is fixed to be

, leading to problem when ∇ θ f → 0, which shows the final optimization result will heavily rely on the selection of g. i.e., the shape of g is crucial for the success of TNB.

or use the barrier term of −α log g(θ) instead:

where α, the barrier factor, is a small positive number.

As α is small, the barrier term will introduce only minuscule influence on the objective.

On the other hand, when θ get closer to the barrier, the objective will increase fast.

It is clear that the solution of the objective with barrier term will get closer to the primal objective as α getting smaller.

Thus in practice, such methods will choose a sequence of {α i } such that 0 < α i < α k+1 and α i → 0 as k → ∞ The limit of Eq. (18) Directly applying this method is computationally challenging and numerically unstable, especially when α is small.

A more natural way can be used: since the learning process is based on sampled transitions, we can simply bound the collected transitions in the feasible region by permitting previous trained M policies θ i ∈ Θ ref , i = 1, 2, ..., M sending termination signals during the training process of new agents.

In other words, we implicitly bound the feasible region by terminating any new agent that steps outside it.

Consequently, during the training process, all valid samples we collected are inside the feasible region, which means these samples are less likely to appear in previously trained policies.

At the end of the training, we then naturally obtain a new policy that has sufficient uniqueness.

In this way, we no longer need to consider the trade-off problem between intrinsic and extrinsic rewards deliberately.

The learning process of our method is thus more robust and no longer suffer from objective inconsistency.

Algorithm.1 shows the pseudo code of IPD based on PPO, where the blue lines show the addition to primal PPO algorithm.

@highlight

A new RL algorithm called Interior Policy Differentiation is proposed to learn a collection of diverse policies for a given primal task.
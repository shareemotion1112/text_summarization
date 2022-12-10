Genetic algorithms have been widely used in many practical optimization problems.

Inspired by natural selection, operators, including mutation, crossover and selection, provide effective heuristics for search and black-box optimization.

However, they have not been shown useful for deep reinforcement learning, possibly due to the catastrophic consequence of parameter crossovers of neural networks.

Here, we present Genetic Policy Optimization (GPO), a new genetic algorithm for sample-efficient deep policy optimization.

GPO uses imitation learning for policy crossover in the state space and applies policy gradient methods for mutation.

Our experiments on MuJoCo tasks show that GPO as a genetic algorithm is able to provide superior performance over the state-of-the-art policy gradient methods and achieves comparable or higher sample efficiency.

Reinforcement learning (RL) has recently demonstrated significant progress and achieves state-ofthe-art performance in games BID14 , locomotion control BID12 , visual-navigation BID29 , and robotics BID11 .

Among these successes, deep neural networks (DNNs) are widely used as powerful functional approximators to enable signal perception, feature extraction and complex decision making.

For example, in continuous control tasks, the policy that determines which action to take is often parameterized by a deep neural network that takes the current state observation or sensor measurements as input.

In order to optimize such policies, various policy gradient methods BID15 BID19 BID7 have been proposed to estimate gradients approximately from rollout trajectories.

The core idea of these policy gradient methods is to take advantage of the temporal structure in the rollout trajectories to construct a Monte Carlo estimator of the gradient of the expected return.

In addition to the popular policy gradient methods, other alternative solutions, such as those for black-box optimization or stochastic optimization, have been recently studied for policy optimization.

Evolution strategies (ES) is a class of stochastic optimization techniques that can search the policy space without relying on the backpropagation of gradients.

At each iteration, ES samples a candidate population of parameter vectors ("genotypes") from a probability distribution over the parameter space, evaluates the objective function ("fitness") on these candidates, and constructs a new probability distribution over the parameter space using the candidates with the high fitness.

This process is repeated iteratively until the objective is maximized.

Covariance matrix adaptation evolution strategy (CMA-ES; BID5 ) and recent work from BID18 are examples of this procedure.

These ES algorithms have also shown promising results on continuous control tasks and Atari games, but their sample efficiency is often not comparable to the advanced policy gradient methods, because ES is black-box and thus does not fully exploit the policy network architectures or the temporal structure of the RL problems.

Very similar to ES, genetic algorithms (GAs) are a heuristic search technique for search and optimization.

Inspired by the process of natural selection, GAs evolve an initial population of genotypes by repeated application of three genetic operators -mutation, crossover and selection.

One of the main differences between GA and ES is the use of the crossover operator in GA, which is able to provide higher diversity of good candidates in the population.

However, the crossover operator is often performed on the parameter representations of two parents, thus making it unsuitable for nonlinear neural networks.

The straightforward crossover of two neural networks by exchanging their parameters can often destroy the hierarchical relationship of the networks and thus cause a catastrophic drop in performance.

NeuroEvolution of Augmenting Topologies (NEAT; BID24 a) ), which evolves neural networks through evolutionary algorithms such as GA, provides a solution to exchange and augment neurons but has found limited success when used as a method of policy search in deep RL for high-dimensional tasks.

A major challenge to making GAs work for policy optimization is to design a good crossover operator which efficiently combines two parent policies represented by neural networks and generates an offspring that takes advantage of both parents.

In addition, a good mutation operator is needed as random perturbations are often inefficient for high-dimensional policies.

In this paper, we present Genetic Policy Optimization (GPO), a new genetic algorithm for sampleefficient deep policy optimization.

There are two major technical advances in GPO.

First, instead of using parameter crossover, GPO applies imitation learning for policy crossovers in the state space.

The state-space crossover effectively combines two parent policies into an offspring or child policy that tries to mimic its best parent in generating similar state visitation distributions.

Second, GPO applies advanced policy gradient methods for mutation.

By randomly rolling out trajectories and performing gradient descent updates, this mutation operator is more efficient than random parameter perturbations and also maintains sufficient genetic diversity.

Our experiments on several continuous control tasks show that GPO as a genetic algorithm is able to provide superior performance over the state-of-the-art policy gradient methods and achieves comparable or higher sample efficiency.

In the standard RL setting, an agent interacts with an environment E modeled as a Markov Decision Process (MDP).

At each discrete time step t, the agent observes a state s t and choose an action a t ∈ A using a policy π(a t |s t ), which is a mapping from states to a distribution over possible actions.

Here we consider high-dimensional, continuous state and action spaces.

After performing the action a t , the agent collects a scalar reward r(s t , a t ) ∈ R at each time step.

The goal in reinforcement learning is to learn a policy which maximizes the expected sum of (discounted) rewards starting from the initial state.

Formally, the objective is DISPLAYFORM0 where the states s t are sampled from the environment E using an unknown system dynamics model p(s t+1 |s t , a t ) and an initial state distribution p(s 0 ), the actions a t are sampled from the policy π(a t |s t ) and γ ∈ (0, 1] is the discount factor.

Policy-based RL methods search for an optimum policy directly in the policy space.

One popular approach is to parameterize the policy π(a t |s t ; θ) with θ, express the objective J(π(a t |s t ; θ)) as a function of θ and perform gradient descent methods to optimize it.

The REINFORCE algorithm BID28 calculates an unbiased estimation of the gradient ∇ θ J(θ) using the likelihood ratio trick.

Specifically, REINFORCE updates the policy parameters in the direction of the following approximation to policy gradient DISPLAYFORM0 ∇ θ log π(a t |s t ; θ)R t based on a single rollout trajectory, where DISPLAYFORM1 is the discounted sum of rewards from time step t. The advantage actor-critic (A2C) algorithm (Sutton & Barto; BID15 uses the state value function (or critic) to reduce the variance in the above gradient estimation.

The contribution to the gradient at time step t is ∇ θ log π(a t |s t ; θ) DISPLAYFORM2 In practice, multiple rollouts are performed to get the policy gradient, and V π (s t ) is learned using a function approximator.

High variance in policy gradient estimates can sometimes lead to large, destructive updates to the policy parameters.

Trust-region methods such as TRPO BID19 avoid this by restricting the amount by which an update is allowed to change the policy.

TRPO is a second order algorithm that solves an approximation to a constrained optimization problem using conjugate gradient.

Proximal policy optimization (PPO) algorithm BID20 is an approximation to TRPO that relies only on first order gradients.

The PPO objective penalizes the Kullback-Leibler (KL) divergence change between the policy before the update (π θ old ) and the policy at the current step (π θ ).

The penalty weight β is adaptive and adjusted based on observed change in KL divergence after multiple policy update steps have been performed using the same batch of data.

DISPLAYFORM3 whereÊ t [...] indicates the empirical average over a finite batch of samples, andÂ t is the advantage estimation.

BID20 propose another objective based on clipping of the likelihood ratio, but we use the adaptive-KL objective due to its better empirical performance BID7 BID4 .

There is growing interest in using evolutionary algorithms as a policy search procedure in RL.

We provide a brief summary; a detailed survey is provided by BID27 .

Neuroevolution is the process of using evolutionary algorithms to generate neural network weights and topology.

Among early applications of neuroevolution algorithms to control tasks are SANE BID16 and ESP BID3 .

NEAT has also been successfully applied for policy optimization BID23 .

NEAT provides a rich representation for genotypes, and tracks the historical origin of every gene to allow for principled crossover between networks of disparate topologies.

In BID2 , the authors introduce an algorithm based on cooperative co-evolution (CoSyNE) and compare it favorably against Q-learning and policy-gradient based RL algorithms.

They do crossover between fixed topology networks using a multi-point strategy at the granularity of network layers (weight segments).

HyperNEAT (D'Ambrosio & Stanley, 2007) , which extends on NEAT and uses CPPN-based indirect encoding, has been used to learn to play Atari games from raw game pixels BID6 .Recently, BID18 proposed a version of Evolution Strategies (ES) for black-box policy optimization.

At each iteration k, the algorithm samples candidate parameter vectors (policies) using a fixed covariance Gaussian N (0, σ 2 I) perturbation on the mean vector m (k) .

The mean vector is then updated in the direction of the weighted average of the perturbations, where weight is proportional to the fitness of the candidate.

CMA-ES has been used to learn neural network policies for reinforcement learning (CMA-NeuroES, BID8 BID9 ).

CMA-ES samples candidate parameter vectors using a Gaussian N (0, C (k) ) perturbation on the mean vector m (k) .

The covariance matrix and the mean vector for the next iteration are then calculated using the candidates with high fitness.

Cross-Entropy methods use similar ideas and have been found to work reasonably well in simple environments BID26 .In this work, we consider policy networks of fixed topology.

Existing neuroevolution algorithms perform crossover between parents by copying segments-single weight or layer(s)-of DNN parameters from either of the parents.

Also, mutation is generally done by random perturbations of the weights, although more rigorous approaches have been proposed BID5 BID21 BID10 .

In this work, we use policy gradient algorithms for efficient mutation of high-dimensional policies, and also depart from prior work in implementing the crossover operator.3 GENETIC POLICY OPTIMIZATION 3.1 OVERALL ALGORITHM Our procedure for policy optimization proceeds by evolving the policies (genotypes) through a series of selection, crossover and mutation operators (Algorithm 1).

We start with an ensemble of for tuple(π x , π y ) ∈ parents set do 7: DISPLAYFORM0 add π c to children DISPLAYFORM1 end for 10:population ← children 11: until k steps of genetic optimization Figure 1 : Different crossover strategies for neural network policies.

State-visitation distribution plot next to each policy depicts the slice of state-space where that policy gives high returns.

In a naïve approach like parameter-space crossover (shown in bottom-right), edge weights are copied from the parent network to create the offspring.

Our proposed state-space crossover operator, instead, aims to achieve the behavior shown in bottom-left.policies initialized with random parameters.

In line 3, we mutate each of the policies separately by performing a few iterations of updates on the policy parameters.

Any standard policy gradient method, such as PPO or A2C, can be used for mutation.

In line 4, we create a set of parents using a selection procedure guided by a fitness function.

Each element of this set is a policy-pair (π x , π y ) that is used in the reproduction (crossover) step to produce a new child policy (π c ).

This is done in line 7 by mixing the policies of the parents.

In line 10, we obtain the population for the next generation by collecting all the newly created children.

The algorithm terminates after k rounds of optimization.

We consider policies that are parameterized using deep neural networks of fixed architectures.

If the policy is Gaussian, as is common for many robotics and locomotion tasks BID1 , then the network outputs the mean and the standard-deviation of each action in the action-space.

Combining two DNN policies such that the final child policy possibly absorbs the best traits of both the parents is non-trivial.

Figure 1 illustrates different crossover strategies.

The figure includes neural network policies along with the state-visitation distribution plots (in a 2D space) corresponding to some high return rollouts using that policy.

The two parent networks are shown in the top half of the figure.

The state-visitation distributions are made non-overlapping to indicate that the parents policies have good state-to-action mapping for disparate local regions of the state-space.

A naïve approach is to do crossover in the parameter space (bottom-right in figure) .

In this approach, a DNN child policy is created by copying over certain edge weights from either of the parents.

The crossover could be at the granularity of multiple DNN layers, a single layer of edges or even a single edge (e.g. NEAT BID24 ).

However, this type of crossover is expected to yield a low-performance composition due to the complex non-linear interactions between policyparameters and the expected policy return.

For the same reason, the state-visitation distribution of the child doesn't hold any semblance to that of either of the parents.

The bottom-left part of the figure shows the outcome of an ideal crossover in state-space.

The state-visitation distribution of the child includes regions from both the parents, leading to better performance (in expectation) than either of them.

In this work, we propose a new crossover operator that utilizes imitation learning to combine the best traits from both parents and generate a high-performance child or offspring policy.

So this crossover is not done directly in the parameter space but in the behavior or the state visitation space.

We quantify the effect of these two types of crossovers in Section 4 by mixing several DNN pairs and measuring the policy performance in a simulated environment.

Imitation learning can broadly be categorized into behavioral cloning, where the agent is trained to match the action of the expert in a given state using supervised learning, and inverse reinforcement learning, where the agent infers a cost function for the environment using expert demonstrations and then learns an optimal policy for that cost function.

We use behavioral cloning in this paper, and all our references to imitation learning should be taken to mean that.

Our second contribution is in utilizing policy gradient algorithms for mutation of neural network weights in lieu of the Gaussian perturbations used in prior work on evolutionary algorithms for policy search.

Because of the randomness in rollout samples, the policy-gradient mutation operator also maintains sufficient genetic diversity in the population.

This helps our overall genetic algorithm achieve similar or higher sample efficiency compared to the state-of-the-art policy gradient methods.

This section details the three genetic operators.

We use different subscripts for different policies.

The corresponding parameters of the neural network are sub-scripted with the same letter (e.g. θ x for π x ).We also use π x and π θx interchangeably.

This operator mixes two input policies π x and π y in statespace and produces a new child policy π c .

The three policies have identical network architecture.

The child policy is learned using a two-step procedure.

A schematic of the methodology is shown in FIG1 .

Firstly, we train a two-level policy π H (a|s) = π S (parent = x|s)π x (a|s) + π S (parent = y|s)π y (a|s) which, given an observation, first chooses between π x and π y , and then outputs the action distribution of the chosen parent.

π S is a binary policy which is trained using trajectories from the parents involved in the crossover.

In our implementation, we reuse the trajectories from the parents' previous mutation phase rather than generating new samples.

The training objective for π S is weighted maximum likelihood, where normalized trajectory returns are used as weights.

For-mally, given two parents π x and π y , the log loss is given by: DISPLAYFORM0 where p x := π S (parent = x|s), w s is the weight assigned to the trajectory which contained the state s, and D is the set of parent trajectories.

This hierarchical reinforcement learning step acts a medium of knowledge transfer from the parents to the child.

We use only high-reward trajectories from π x and π y as data samples for training π S to avoid transfer of negative behavior.

It is possible to further refine π S by running a few iterations of any policy-gradient algorithm, but we find that the maximum likelihood approach works well in practice and can also avoid extra rollout samples.

Next, to distill the information from π H into a policy with the same architecture as the parents, we use imitation learning to train a child policy π c .

We use trajectories from π H (expert) as supervised data and train π c to predict the expert action under the state distribution induced by the expert.

The surrogate loss for imitation learning is: DISPLAYFORM1 where d * is the state-visitation distribution induced by π H .

To avoid compounding errors due to state distribution mismatch between the expert and the student, we adopt the Dataset Aggregation (DAgger) algorithm BID17 .

Our training dataset D is initialized with trajectories from the expert.

After iteration i of training, we sample some trajectories from the current student (π (i) c ), label the actions in these trajectories using the expert and form a dataset D i .

Training for iteration i + 1 then uses {D ∪ D 1 . . .

∪ D i } to minimize the KL-divergence loss.

This helps to achieve a policy that performs well under its own induced state distribution.

The direction of KL-divergence in Equation 2 encourages high entropy in π c , and empirically, we found this to be marginally better than the reverse direction.

For policies with Gaussian actions, the KL has a closed form and therefore the surrogate loss is easily optimized using a first order method.

In experiments, we found that this crossover operator is very efficient in terms of sample complexity, as it requires only a small size of rollout samples.

More implementation details can be found in Appendix 6.3.

This operator modifies (in parallel) each policy of the input policy ensemble by running some iterations of a policy gradient algorithm.

The policies have different initial parameters and are updated with high-variance gradients estimated using rollout trajectories.

This leads to sufficient genetic diversity and good exploration of the state-space, especially in the initial rounds of GPO.

For two popular policy gradient algorithms-PPO and A2C-the gradients for policy π i are calculated as DISPLAYFORM0 whereÊ i,t [...] indicates the empirical average over a finite batch of samples from π i , andÂ t is the advantage.

We use an MLP to model the critic baseline V π (s t ) for advantage estimation.

PPO does multiple updates on the policy π θi using the same batch of data collected using π θ (old) i , whereas A2C does only a single update.

During mutation, a policy π i can also use data samples from other similar policies in the ensemble for off-policy learning.

A larger data-batch (generally) leads to a better estimate of the gradient and stabilizes learning in policy gradient methods.

When using data-sharing, the gradients for π i are DISPLAYFORM1 where S i ≡ {j | KL[π i , π j ] < before the start of current round of mutation} contains similar policies to π i (including π i ).

Given a set of m policies and a fitness function, this operator returns a set of policy-couples for use in the crossover step.

From all possible m 2 couples, the ones with maximum fitness are selected.

The fitness function f (π x , π y ) can be defined according two criteria, as below.• Performance fitness as sum of expected returns of both policies, i.e. f (π x , π y ) DISPLAYFORM0 • Diversity fitness as KL-divergence between policies, i.e. f (π x , π y ) DISPLAYFORM1 While the first variant favors couples with high cumulative performance, the second variant explicitly encourages crossover between diverse (high KL divergence) parents.

A linear combination provides a trade-off of these two measures of fitness that can vary during the genetic optimization process.

In the early rounds, a relatively higher weight could be provided to KL-driven fitness to encourage exploration of the state-space.

The weight could be annealed with rounds of Algorithm 1 for encouraging high-performance policies.

For our experiments, we use a simple variant where we put all the weight on the performance fitness for all rounds, and rely on the randomness in the starting seed for different policies in the ensemble for diversity in the initial rounds.

In this section, we conduct experiments to measure the efficacy and robustness of the proposed GPO algorithm on a set of continuous control benchmarks.

We begin by describing the experimental setup and our policy representation.

We then analyze the effect of our crossover operator.

This is followed by learning curves for the simulated environments, comparison with baselines and ablations.

We conclude with a discussion on the quality of policies learned by GPO and scalability issues.

All our experiments are done using the OpenAI rllab framework BID1 .

We benchmark 9 continuous-control locomotion tasks based on the MuJoCo physics simulator 1 .

All our control policies are Gaussian, with the mean parameterized by a neural network of two hidden layers (64 hidden units each), and linear units for the final output layer.

The diagonal co-variance matrix is learnt as a parameter, independent of the input observation, similar to BID19 .

The binary policy (π S ) used for crossover has two hidden layers (32 hidden units each), followed by a softmax.

The value-function baseline used for advantage estimation also has two hidden layers (32 hidden units each).

All neural networks use tanh as the non-linearity at the hidden units.

We show results with PPO and A2C as policy gradient algorithms for mutation.

PPO performs 10 steps of full-batch gradient descent on the policy parameters using the same collected batch of simulation data, while A2C does a single descent step.

Other hyperparameters are in Appendix 6.3.

To measure the efficacy of our crossover operator, we run GPO on the HalfCheetah environment, and plot the performance of all the policies involved in 8 different crossovers that occur in the first (a) Average episode reward for the child policies after state-space crossover (left) and parameter-space crossover (right), compared to the performance of the parents.

All bars are normalized to the first parent in each crossover.

Policies are trained on the HalfCheetah environment.(b) State visitation distribution for high reward rollouts from policies trained on the HalfCheetah environment.

From left to right -first parent, second parent, child policy using state-space crossover, child policy using parameter-space crossover.

The number above each subplot is the average episode reward for 100 rollouts from the corresponding policy.

Table 1 : Mean and standard-error for final performance of GPO and baselines using PPO.round of Algorithm 1.

FIG2 shows the average episode reward for the parent policies and their corresponding child.

All bars are normalized to the first parent in each crossover.

The left subplot depicts state-space crossover.

We observe that in many cases, the child either maintains or improves on the better parent.

This is in contrast to the right subplot where parameter-space crossover breaks the information structure contained in either of the parents to create a child with very low performance.

To visualize the state-space crossover better, in FIG2 we plot the state-visitation distribution for high reward rollouts from all policies involved in one of the crossovers.

All states are projected from a 20 dimensional space (for HalfCheetah) into a 2D space by t-SNE BID13 .

Notwithstanding artifacts due to dimensionality reduction, we observe that high reward rollouts from the child policy obtained with state-space crossover visit regions frequented by both the parents, unlike the parameter-space crossover (rightmost subplot) where the policy mostly meanders in regions for which neither of the parents have strong supervision.

In this subsection, we compare the performance of policies trained using GPO with those trained with standard policy gradient algorithms.

GPO is run for 12 rounds (Algorithm 1) with a population size of 8, and simulates 8 million timesteps in total for each environment (1 million steps per candidate policy).

We compare with two baselines which use the same amount of data.

The first baseline algorithm, Single, trains 8 independent policies with policy gradient using 1 million timesteps each, and selects the policy with the maximum performance at the end of training.

Unlike GPO, these policies do not participate in state-space crossover or interact in any way.

The second baseline algorithm, Joint, trains a single policy with policy gradient using 8 million timesteps.

Both Joint and Single do the same number of gradient update steps on the policy parameters, but each gradient step in Joint uses 8 times the batch-size.

For all methods, we replicate 8 runs with different seeds and show the mean and standard error.

FIG3 plots the moving average of per-episode reward when training with PPO as the policy gradient method for all algorithms.

The x-axis is the total timesteps of simulation, including the data required for DAgger imitation learning in the crossover step.

We observe that GPO achieves better performance than Single is almost all environments.

Joint is a more challenging baseline since each gradient step uses a larger batch-size, possibly leading to well-informed, low-variance gradient estimates.

Nonetheless, GPO reaches a much better score for environments such as Walker2D and HalfCheetah, and also their more difficult (hilly) versions.

We believe this is due to better exploration and exploitation by the nature of the genetic algorithm.

The performance at the end of training is shown in Table 1 .

Results with A2C as the policy gradient method are in Appendix 6.1.

With A2C, GPO beats the baselines in all but one environments.

In summary, these results indicate that, with the new crossover and mutation operators, genetic algorithms could be an alternative policy optimization approach that competes with the state-of-the-arts policy gradient methods.

Our policy optimization procedure uses crossover, select and mutate operators on an ensemble of policies over multiple rounds.

In this section, we perform ablation studies to measure the impact on performance when only certain operator(s) are applied.

FIG4 shows the results and uses the following symbols:• Crossover (C) -Presence of the symbol indicates state-space crossover using imitation learning; otherwise a simple crossover is done by copying the parameters of the stronger parent to the offspring policy.• Select (S) -Presence of the symbol denotes use of a fitness function (herein performancefitness, Section 3.3.3) to select policy-pairs to breed; otherwise random selection is used.• Data-sharing during Mutate (M) -Mutation in GPO is done using policy-gradient, and policies in the ensemble share batch samples with other similar policies (Section 3.3.2).

We use this symbol when sharing is enabled; otherwise omit it.

In FIG4 , we refer to setting where none of the components {C, S, M} are used as Base and apply components over it.

We also show Single which trains an ensemble of policies that do not interact in any manner.

We note that each of the components applied in isolation provide some improvement over Single, with data-sharing (M) having the highest benefit.

Also, combining two components generally leads to better performance than using either of the constituents alone.

Finally, using all components results in GPO and it gives the best performance.

The normalized numbers are mentioned in the figure caption.

The SELECTION operator selects high-performing individuals for crossover in every round of Algorithm 1.

Natural selection weeds out poorly-performing policies during the optimization process.

In FIG5 , we measure the average episode reward for each of the policies in the ensemble at the final round of GPO.

We compare this with the final performance of the 8 policies trained using the Single baseline.

We conclude that the GPO policies are more robust.

In FIG6 , we experiment with varying the population size for GPO.

All the policies in this experiment use the same batch-size for the gradient steps and do the same number of gradient steps.

Performance improves by increasing the population size suggesting that GPO is a scalable optimization procedure.

Moreover, the MUTATE and CROSSOVER genetic operators lend themselves perfectly to multiprocessor parallelism.

We presented Genetic Policy Optimization (GPO), a new approach to deep policy optimization which combines ideas from evolutionary algorithms and reinforcement learning.

First, GPO does efficient policy crossover in state space using imitation learning.

Our experiments show the benefits of crossover in state-space over parameter-space for deep neural network policies.

Second, GPO mutates the policy weights by using advanced policy gradient algorithms instead of random perturbations.

We conjecture that the noisy gradient estimates used by policy gradient methods offer sufficient genetic diversity, while providing a strong learning signal.

Our experiments on several MuJoCo locomotion tasks show that GPO has superior performance over the state-of-the-art policy gradient methods and achieves comparable or higher sample efficiency.

Future advances in policy gradient methods and imitation learning will also likely improve the performance of GPO for challenging RL tasks.

Table 2 : Mean and standard-error for final performance of GPO and baselines using A2C.

We use the OpenAI rllab framework, including the MuJoCo environments provided therein, for all our experiments.

There are subtle differences in the environments included in rllab and OpenAI Gym repositories 2 in terms of the coefficients used for different reward components and aliveness bonuses.

In FIG7 , we compare GPO and Joint baseline using PPO on three Gym environments, for two different time-horizon values (512, 1024).

The crossover stage is divided into two phases -a) training the binary policy (π S ) and 2) imitation learning.

For training π S , the dataset consists of trajectories from the parents involved in the crossover.

We combine the trajectories from the parents' previous mutation phase, and filter based on trajectory rewards.

For our experiments, we select top 60% trajectories from the pool, although we did not find the final GPO performance to be very sensitive to this hyperparameter.

We do 100 epochs of supervised training with Adam (mini-batch=64, learning rate=5e-4) on the loss defined in Equation 1.After training π S , we obtain expert trajectories (5k transitions) from π H .

Imitation learning is done in a loop which is run for 10 iterations.

In each iteration i, we form the dataset D i using existing expert trajectories, plus new trajectories (500 transitions) sampled from the student (child) policy π (i) c with the actions labelled by the expert.

Therefore, the ratio of student to expert transitions in D i is linearly increased from 0 to 1 over the 10 iterations.

We then update the student by minimizing the KL divergence objective over D i .

We do 10 epochs of training with Adam (mini-batch=64, learning rate=5e-4) in each iteration.

Table 3 shows the wall-clock time (in minutes), averaged over 3 runs, for GPO and the Joint baseline.

All runs use the same number of simulation timesteps, and are done on an Intel Xeon machine 3 with 12 cores.

For GPO, Mutate takes the major chunk of total time.

This is partially due to the fact that data-sharing between ensemble policies leads to communication overheads.

Having said that, our current implementation based on Python Multiprocessing module and file-based sharing in Unix leaves much on the table in terms of improving the efficiency for Mutate, for example by using MPI.

Joint trains 1 policy with 8× the number of samples as each policy in the GPO ensemble.

However, sample-collection exploits 8-way multi-core parallelism by simulating multiple independent environments in separate processes.

Table 3 : Average wall-clock time (in minutes) for GPO and Joint.

Figure 10: Effect of GPO population size when using same number of timesteps.

In FIG6 , we ran Walker2D with different population-size values, and compared performance.

All policies used the same batch-size and number of gradient steps, making the total number of simulation timesteps grow with the population-size.

In Figure 10 , we show results for the same environment but reduce the batch-size for each gradient step in proportion to the increase in populationsize.

Therefore, all experiments here use equal simulation timesteps.

We observe that the samplecomplexity for a population of 32 is quite competitive with our default GPO value of 8.

@highlight

Genetic algorithms based approach for optimizing deep neural network policies

@highlight

The authors present an algorithm for training ensembles of policy networks that regularly mixes different policies in the ensemble together.

@highlight

This paper proposes a genetic algorithm inspired policy optimization method, which mimics the mutation and the crossover operators over policy networks.
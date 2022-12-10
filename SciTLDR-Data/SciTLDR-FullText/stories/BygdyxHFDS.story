Exploration is a key component of successful reinforcement learning, but optimal approaches are computationally intractable, so researchers have focused on hand-designing mechanisms based on exploration bonuses and intrinsic reward, some inspired by curious behavior in natural systems.

In this work, we propose a strategy for encoding curiosity algorithms as programs in a domain-specific language and searching, during a meta-learning phase, for algorithms that enable RL agents to perform well in new domains.

Our rich language of programs, which can combine neural networks with other building blocks including nearest-neighbor modules and can choose its own loss functions, enables the expression of highly generalizable programs that perform well in domains as disparate as grid navigation with image input, acrobot, lunar lander, ant and hopper.

To  make this approach feasible, we develop several pruning techniques, including learning to predict a program's success based on its syntactic properties.

We demonstrate the effectiveness of the approach empirically, finding curiosity strategies that are similar to those in published literature, as well as novel strategies that are competitive with them and generalize well.

Figure 1: Our RL agent is augmented with a curiosity module, obtained by meta-learning over a complex space of programs, which computes a pseudo-reward r at every time step.

When an agent is learning to behave online, via reinforcement learning (RL), it is critical that it both explores its domain and exploits its rewards effectively.

In very simple problems, it is possible to solve the problem optimally, using techniques of Bayesian decision theory (Ghavamzadeh et al., 2015) .

However, these techniques do not scale at all well and are not effectively applicable to the problems addressable by modern deep RL, with large state and action spaces and sparse rewards.

This difficulty has left researchers the task of designing good exploration strategies for RL systems in complex environments.

One way to think of this problem is in terms of curiosity or intrisic motivation: constructing reward signals that augment or even replace the extrinsic reward from the domain, which induce the RL agent to explore their domain in a way that results in effective longer-term learning and behavior (Pathak et al., 2017; Burda et al., 2018; Oudeyer, 2018) .

The primary difficulty with this approach is that researchers are hand-designing these strategies: it is difficult for humans to systematically consider the space of strategies or to tailor strategies for the distribution of environments an agent might be expected to face.

We take inspiration from the curious behavior observed in young humans and other animals and hypothesize that curiosity is a mechanism found by evolution that encourages meaningful exploration early in agent's life in order to expose it to experiences that enable it to learn to obtain high rewards over the course of its lifetime.

We propose to formulate the problem of generating curious behavior as one of meta-learning: an outer loop, operating at "evolutionary" scale will search over a space of algorithms for generating curious behavior by dynamically adapting the agent's reward signal, and the inner loop will perform standard reinforcement learning using the adapted reward signal.

This process is illustrated in figure 1; note that the aggregate agent, outlined in gray, has the standard interface of an RL agent.

The inner RL algorithm is continually adapting to its input stream of states and rewards, attempting to learn a policy that optimizes the discounted sum of proxy rewards k≥0 γ k r t+k .

The outer "evolutionary" search is attempting to find a program for the curiosity module, so to optimize the agent's lifetime return T t=0 r t , or another global objective like the mean performance on the last few trials.

Although it is, in principle, possible to discover a complete, integrated algorithm for the entire curious learning agent in the gray box, that is a much more complex search problem that is currently computationally infeasible.

We are relying on the assumption that the foundational methods for reinforcement learning, including those based on temporal differencing and policy gradient, are fundamentally sound and can serve as the behavior-learning basis for our agents.

It is important to note, though, that the internal RL algorithm in our architecture must be able to tolerate a nonstationary reward signal, which may necessitate minor algorithmic changes or, at least, different hyperparameter values.

In this meta-learning setting, our objective is to find a curiosity module that works well given a distribution of environments from which we can sample at meta-learning time.

If the environment distribution is relatively low-variance (the tasks are all quite similar) then it might suffice to search over a relatively simple space of curiosity strategies (most trivially, the in an -greedy exploration strategy).

Meta-RL has been widely explored recently, in some cases with a focus on reducing the amount of experience needed by initializing the RL algorithm well (Finn et al., 2017; Clavera et al., 2019) and, in others, for efficient exploration (Duan et al., 2016; Wang et al., 2017) .

The environment distributions in these cases have still been relatively low-diversity, mostly limited to variations of the same task, such as exploring different mazes or navigating terrains of different slopes.

We would like to discover curiosity mechanisms that can generalize across a much broader distribution of environments, even those with different state and action spaces: from image-based games, to joint-based robotic control tasks.

To do that, we perform meta-learning in a rich, combinatorial, open-ended space of programs.

This paper makes three novel contributions.

We focus on a regime of meta-reinforcement-learning in which the possible environments the agent might face are dramatically disparate and in which the agent's lifetime is very long.

This is a substantially different setting than has been addressed in previous work on meta-RL and it requires substantially different techniques for representation and search.

We represent meta-learned curiosity strategies in a rich, combinatorial space of programs rather than in a fixed-dimensional numeric parameter space.

The programs are represented in a domain-specific language (DSL) which includes sophisticated building blocks including neural networks complete with gradient-descent mechanisms, learned objective functions, ensembles, buffers, and other regressors.

This language is rich enough to represent many previously reported hand-designed exploration algorithms.

We believe that by performing meta-RL in such a rich space of mechanisms, we will be able to discover highly general, fundamental curiosity-based exploration methods.

This generality means that a relatively computationally expensive meta-learning process can be amortized over the lifetimes of many agents in a wide variety of environments.

We make the search over programs feasible with relatively modest amounts of computation.

It is a daunting search problem to find a good solution in a combinatorial space of programs, where evaluating a single potential solution requires running an RL algorithm for up to millions of time steps.

We address this problem in multiple ways.

By including environments of substantially different difficulty and character, we can evaluate candidate programs first on relatively simple and short-horizon domains: if they don't perform well in those domains, they are pruned early, which saves a significant amount of computation time.

In addition, we predict the performance of an algorithm from its structure and operations, thus trying the most promising algorithms early in our search.

Finally, we also monitor the learning curve of agents and stop unpromising programs before they reach all T environment steps.

We demonstrate the effectiveness of the approach empirically, finding curiosity strategies that are similar to those in published literature, as well as novel strategies that are competitive with them and generalize well.

Let us assume we have an agent equipped with an RL algorithm (such as DQN or PPO, with all hyperparameters specified), A, which receives states and rewards from and outputs actions to an environment E, generating a stream of experienced transitions e(A; E) t = (s t , a t , r t , s t+1 ).

The agent continually learns a policy π(t) : s t → a t , which will change in time as described by algorithm A; so π(t) = A(e 1:t−1 ) and thus a t ∼ A(e 1:t−1 )(s t ).

Although this need not be the case, we can think of A as an algorithm that tries to maximize the discounted reward i γ i r t+i , γ < 1 and that, at any time-step t, always takes the greedy action that maximizes its estimated expected discounted reward.

To add exploration to this policy, we include a curiosity module C that has access to the stream of state transitions e t experienced by the agent and that, at every time-step t, outputs a proxy reward r t .

We connect this module so that the original RL agent receives these modified rewards, thus observing e(A, C; E) t = (s t , a t , r t = C(e 1:t−1 ), s t+1 ), without having access to the original r t .

Now, even though the inner RL algorithm acts in a purely exploitative manner with respect to r, it may efficiently explore in the outer environment.

Our overall goal is to design a curiosity module C that induces the agent to maximize T t=0 r t , for some number of total time-steps T or some other global goal, like final episode performance.

In an episodic problem, T will span many episodes.

More formally, given a single environment E, RL algorithm A, and curiosity module C, we can see the triplet (environment, curiosity module, agent) as a dynamical system that induces state transitions for the environment, and learning updates for the curiosity module and the agent.

Our objective is to find C that maximizes the expected original reward obtained by the composite system in the environment.

Note that the expectation is over two different distributions at different time scales: there is an "outer" expectation over environments E, and in "inner" expectation over the rewards received by the composite system in that environment, so our final objective is:

In science and computing, mathematical language has been very successful in describing varied phenomena and powerful algorithms with short descriptions.

As Valiant points out: "the power [of mathematics and algorithms] comes from the implied generality, that knowledge of one equation alone will allow one to make accurate predictions about a host of situations not even conceived when the equation was first written down" (Valiant, 2013) .

Therefore, in order to obtain curiosity modules that can generalize over a very broad range of tasks and that are sophisticated enough to provide exploration guidance over very long horizons, we describe them in terms of general programs in a domain-specific language.

Algorithms in this language will map a history of (s t+1 , a t , r t ) triples into a proxy reward r t .

Inspired by human-designed systems that compute and use intrinsic rewards, and to simplify the search, we decompose the curiosity module into two components: the first, I, outputs an intrinsic reward value i t based on the current experienced transition (s t , a t , s t+1 ) (and past transitions (s 1:t−1 , a 1:t−1 ) indirectly through its memory); the second, χ, takes the current time-step t, the actual reward r t , and the intrinsic reward i t (and, if it chooses to store them, their histories) and combines them to yield the proxy reward r t .

To ease generalization across different timescales, in practice, before feeding t into χ we normalize it by the total length of the agent's lifetime, T .

We draw both programs from the same basic class.

Fundamentally, they consist of a directed acyclic graph (DAG) of modules with polymorphically typed inputs and outputs.

There are four classes of modules:

• Input modules (shown in blue), drawn from the set {s t , a t , s t+1 } for the I module and from the set {i t , r t } for the χ module.

They have no inputs, and their outputs have the type corresponding to the types of states and actions in whatever domain they are applied to, or the reals numbers for rewards.

Figure 2: Example diagrams of published algorithms covered by our language (larger figures in the appendix).

The green box represents the output of the intrinsic curiosity function, the pink box is the loss to be minimized.

Pink arcs represent paths and networks along which gradients flow back from the minimizer to update parameters.

• Buffer and parameter modules (shown in gray) of two kinds: FIFO queues that provide as output a finite list of the k most recent inputs, and neural network weights initialized at random at the start of the program and which may (pink border) or may not get updated via back-propagation depending on the computation graph.

• Functional modules (shown in white), which compute output values given input from their parent modules.

• Update modules (shown in pink), which are functional modules (such as k-NearestNeighbor) that either add variables to buffers or modules which add real-valued outputs to a global loss that will provide error signals for gradient descent.

A single node in the DAG is designated as the output node (shown in green): the output of this node is considered to be the output of the entire program, but it need not be a leaf node of the DAG.

On each call to a program (corresponding to one time-step of the system) the current input values and parameter values are propagated through the functional modules, and the output node's output is saved, to be yielded as the output of the whole program.

Before the call terminates, the FIFO buffers are updated and the adjustable parameters are updated via gradient descent using the Adam optimizer Kingma & Ba (2014) .

Most operations are differentiable and thus able to propagate gradient backwards.

Some operations are not differentiable such as buffers (to avoid backpropagating through time) and "Detach" whose purpose is stopping the gradient from flowing back.

In practice, we have multiple copies of the same agent running at the same time, with both a shared policy and shared curiosity module.

Thus, we execute multiple reward predictions on a batch and then update on a batch.

A crucial, and possibly somewhat counter-intuitive, aspect of these programs is their use of neural network weight updates via gradient descent as a form of memory.

In the parameter update step, all adjustable parameters are decremented by the gradient of the sum of the outputs of the loss modules, with respect to the parameters.

This type of update allows the program to, for example, learn to make some types of predictions, online, and use the quality of those predictions in a state to modulate the proxy reward for visiting that state (as is done, for example, in random network distillation (RND) (Burda et al., 2018) ).

Programs representing several published designs for curiosity modules that perform internal gradient descent, including inverse features (Pathak et al., 2017) , RND (Burda et al., 2018) , and ensemble predictive variance (Pathak et al., 2019) , are shown in figure 2 (and bigger versions can be found in appendix A.3).

We can also represent algorithms similar to novelty search (Lehman & Stanley, 2008) and EX 2 (Fu et al., 2017) , which include buffers and nearest neighbor regression modules.

Details on the data types and module library are given in appendix A.

Key to our program search are polymorphic data types: the inputs and outputs to each module are typed, but the instantiation of some types, and thus of some operations, depends on the environment.

We have the four types: reals R, state space of the given environment S, action space of the given environment A and feature space F, used for intermediate computations and always set to R 32 in our current implementation.

For example, a neural network module going from S to F will be instantiated as a convolutional neural network if S is an image and as a fully connected neural network of the appropriate dimension if S is a vector.

Similarly, if we are measuring an error in action space A we use mean-squared error for continuous action spaces and negative log-likelihood for discrete action spaces.

This facility means that the same curiosity program can be applied, independent of whether states are represented as images or vectors, or whether the actions are discrete or continuous, or the dimensionality of either.

This type of abstraction enables our meta-learning approach to discover curiosity modules that generalize radically, applying not just to new tasks, but to tasks with substantially different input and output spaces than the tasks they were trained on.

To clarify the semantics of these programs, we walk through the operation of the RND program in figure 2.

Its only input is s t+1 , which might be an image or an input vector, which is processed by two NNs with parameters Θ 1 and Θ 2 , respectively.

The structure of the NNs (and, hence, the dimensions of the Θ i ) depends on the type of s t+1 : if s t+1 is an image, then they are CNNs, otherwise a fully connected networks.

Each NN outputs a 32-dimensional vector; the L 2 distance between these vectors is the output of the program on this iteration, and is also the input to a loss module.

So, given an input s t+1 , the output intrinsic reward is large if the two NNs generate different outputs and small otherwise.

After each forward pass, the weights in Θ 2 are updated to minimize the loss while Θ 1 remains constant, which causes the trainable NN to mimic the output of the randomly initialized NN.

As the program's ability to predict the output of the randomized NN on an input improves, the intrinsic reward for visiting that state decreases, driving the agent to visit new states.

To limit the search space and prioritize short, meaningful programs we limit the total number of modules of the computation graph to 7.

Our language is expressive enough to describe many (but far from all) curiosity mechanisms in the existing literature, as well as many other potential alternatives, but the expressiveness leads to a very large search space.

Additionally, removing or adding a single operation can drastically change the behavior of a program, making the objective function nonsmooth and, therefore, the space hard to search.

In the next section we explore strategies for speeding up the search over tens of thousands of programs.

We wish to find curiosity programs that work effectively in a wide range of environments, from simple to complex.

However, evaluating tens of thousands of programs in the most expensive environments would consume decades of GPU computation.

Therefore, we have designed multiple strategies for quickly discarding less promising programs and focusing more computation on a few promising programs.

In doing so, we take inspiration from efforts in the AutoML community (Hutter et al., 2018) .

We divide these pruning efforts into three categories: simple tests that are independent of running the program in any environment, "filtering" by ruling out some programs based on poor performance in simple environments, and "meta-meta-RL" learning to predict which programs will perform well based on syntactic features.

Many programs are obviously bad curiosity programs.

We have developed two heuristics to immediately prune these programs without an expensive evaluation.

• Checking that programs are not duplicates.

Since our language is highly expressive, there are many non-obvious ways of getting equivalent programs.

To find duplicates, we designed a randomized test where we identically seed two programs, feed them both identical fake environment data for tens of steps and check whether their outputs are identical.

This test may, with low probability, prune a program that is not an exact duplicate, but since there is a very near neighbor under consideration, it is not very harmful to do so.

• Checking that the loss functions cannot be minimized independently of the input data.

Many programs optimize some loss depending on neural network regressors.

If we treat inputs as uncontrollable variables and networks as having the ability to become any possible function, then for every variable, we can determine whether neural networks can be optimized to minimize it, independently of the input data.

For example, if our loss function is |N N θ (s)| 2 the neural network can learn to make it 0 by disregarding s and optimizing the weights θ to 0.

We discard any program that has this property.

Our ultimate goal is to find algorithms that perform well on many different environments, both simple and complex.

We make two key observations.

First, there may be only tens of reasonable programs that perform well on all environments but hundreds of thousands of programs that perform poorly.

Second, there are some environments that are solvable in a few hundred steps while others require tens of millions.

Therefore, a key idea in our search is to try many programs in cheap environments and only a few promising candidates in the most expensive environments.

This was inspired by the effective use of sequential halving (Karnin et al., 2013) in hyper-parameter optimization (Jamieson & Talwalkar, 2016) .

By pruning programs aggressively, we may be losing multiple programs that perform well on complex environments.

However, by definition, these programs will tend to be less general and robust than those that succeed in all environments.

Moreover, we seek generalization not only for its own sake, but also to ease the search since, even if we only cared about the most expensive environment, performing the complete search only in this environment would be impractical.

Perhaps surprisingly, we find that we can predict program performance directly from program structure.

Our search process bootstraps an initial training set of (program structure, program performance) pairs, then uses this training set to select the most promising next programs to evaluate.

We encode each program's structure with features representing how many times each operation is used, thus having as many features as number of operations in our vocabulary.

We use a k-nearestneighbor regressor, with k = 10.

We then try the most promising programs and update the regressor with their results.

Finally, we add an -greedy exploration policy to make sure we explore all the search space.

Even though the correlation between predictions and actual values is only moderately high (0.54 on a holdout test), this is enough to discover most of the top programs searching only half of the program space, which is our ultimate goal.

Results are shown in appendix C.

We can also prune algorithms during the training process of the RL agent.

In particular, at any point during the meta-search, we use the top K current best programs as benchmarks for all T timesteps.

Then, during the training of a new candidate program we compare its current performance at time t with the performance at time t of the top K programs and stop the run if its performance is significantly lower.

If the program is not pruned and reaches the final time-step T with one of the top K performances, it becomes part of the benchmark for the future programs.

Our RL agent uses PPO (Schulman et al., 2017) based on the implementation by Kostrikov (2018) in PyTorch (Paszke et al., 2017) .

Our code, which can be found at https://bit.

ly/meta-learning-curiosity-algs, is meant to take in any OpenAI gym environment (Brockman et al., 2016) with a specification of the desired exploration horizon T .

We evaluate each curiosity algorithm for multiple trials, using a seed dependent on the trial but independent of the algorithm, which leads to the PPO weights and curiosity data-structures being initialized identically on the same trials for all algorithms.

As is common in PPO, we run multiple rollouts (5, except for MuJoCo which only has 1), with independent experiences but shared policy and curiosity modules.

Curiosity predictions and updates are batched across these rollouts, but not across time.

PPO policy updates are batched both across rollouts and multiple timesteps.

We start by searching for a good intrinsic curiosity program I in a purely exploratory environment, designed by Chevalier-Boisvert et al. (2018) , which is an image-based grid world where agents navigate in an image of a 2D room either by moving forward in the pixel grid or rotating left or right.

We optimize the total number of distinct pixels visited across the agent's lifetime.

This allows us to evaluate intrinsic reward programs in a fast and simple environment, without worrying about combining it with external reward.

To bias towards simple, interpretable algorithms and keep the search space manageable, we search for programs with at most 7 operations.

We first discard duplicate and invalid programs, as described in section 3.1, resulting in about 52,000 programs.

We then randomly split the programs across 4 machines, each with 8 Nvidia Tesla K80 GPUs for 10 hours.

Each machine tries to find the highest-scoring 625 programs in its section of the search space and prunes programs whose partial learning curve is statistically significantly lower than the current top 625 programs.

To do so, after every episode of every trial, we check whether the mean performance of the current program is below the mean performance (at that point during the trial) of the top 625 programs minus two standard deviations of their performance minus one standard deviation of our estimate of the mean of the current program.

In this way we account for both inter-program variability among the top 625 programs and intra-program variability among multiple trials of the same program.

We use a 10-nearest-neighbor regressor to predict program performance and choose the next program to evaluate with an -greedy strategy, choosing the best predicted program 90% of the time and a random program 10% of the time.

By doing this, we try the most promising programs early in our search.

This is important for two reasons: first, we only try 26,000 programs, half of the whole search space, which we estimated from earlier results (shown in figure 8 in the appendix) would be enough to get 88% of the top 1% of programs.

Second, the earlier we run our best programs, the higher the bar for later programs, thus allowing us to prune them earlier, further saving computation time.

Searching through this space took a total of 13 GPU days.

As shown in figure 9 in the appendix, we find that most programs perform relatively poorly, with a long tail of programs that are statistically significantly better, comprising roughly 0.5% of the whole program space.

The highest scoring program (a few other programs have lower average performance but are statistically equivalent) is surprisingly simple and meaningful, comprised of only 5 operations, even though the limit was 7.

This program, which we will call Top, is shown in figure 3 ; it uses a single neural network (a CNN or MLP depending on the type of state) to predict the action from s t+1 and then compares its predictions based on s t with its predictions based on s t+1 , generating high intrinsic reward when the difference is large.

The action prediction loss module either computes a softmax followed by NLL loss or appends zeros to the action to match dimensions and applies MSE loss, depending on the type of the action space.

Note that this is not the same as rewarding taking a different action in the previous time-step.

To the best of our knowledge, the algorithm represented by this program has not been proposed before, although its simplicity makes us think it may have.

The network predicting the action is learning to imitate the policy learned by the internal RL agent, because the curiosity module does not have direct access to the RL agent's internal state.

We show correlation between program performance in gridworld and performance in harder environments (lunar lander on the left, acrobot on the right), using the top 2,000 programs in gridworld.

Performance is evaluated using mean reward across all learning episodes, averaged over trials (two trials for acrobot / lunar lander and five for gridworld).We can see that almost all intrinsic curiosity programs that had statistically significant performance for grid world also do well on the other two environments.

In green, we show the performance of three published works; in increasing gridworld performance: disagreement (Pathak et al., 2019) , inverse features (Pathak et al., 2017) and random distillation (Burda et al., 2018) .

Many of the highest-scoring programs are small variations on Top, including versions that predict the action from s t instead of s t+1 .

Of the top 16 programs, 13 are variants of Top and 3 are variants of an interesting program that is more difficult to understand because it does a combination of random network distillation and state-transition prediction, with some weight sharing, shown in figure 11 in the appendix.

Our reward combiner was developed in lunar lander (the simplest environment with meaningful extrinsic reward) based on the best program among a preliminary set of 16,000 programs (which resembled Random Network Distillation, its computation graph is shown in appendix E).

Among a set of 2478 candidates (with 5 or less operations) the best reward combiner was r t = (1+it−t/T )·it+t/T ·rt 1+it .

Notice that for 0 < i t 1 (usually the case) this is approximately r t = i 2 t + (1 − t/T )i t + (t/T )r t , which is a down-scaled version of intrinsic reward plus a linear interpolation that ranges from all intrinsic reward at t = 0 to all extrinsic reward at t = T .

In future work, we hope to co-adapt the search for intrinsic reward programs and combiners as well as find multiple reward combiners.

Given the fixed reward combiner and the list of 2,000 selected programs found in the image-based grid world, we evaluate the programs on both lunar lander and acrobot, in their discrete action space versions.

Notice that both environments have much longer horizons than the image-based grid world (37,500 and 50,000 vs 2,500) and they have vector-based inputs, not image-based.

The results in figure 4 show good correlation between performance on grid world and on each of the new environments.

Especially interesting is that, for both environments, when intrinsic reward in grid world is above 370 (the start of the statistically significant performances), performance on the other two environments is also good in more than 90% of cases.

Finally, we evaluate the 16 best programs on grid world (most of which also did well on lunar lander and acrobot) on two MuJoCo environments (Todorov et al., 2012) : hopper and ant.

These environments have more than an order of magnitude longer exploration horizon than acrobot and lunar lander, exploring for 500K time-steps, as well as continuous action-spaces instead of discrete.

Ant Hopper Baseline algorithms [-95.3, -39.9 Table 1 : Meta-learned algorithms perform significantly better than constant rewards and statistically equivalently to published algorithms found by human researchers (see 2).

The table shows the confidence interval (one standard deviation) for the mean performance (across trials, across algorithms) for each algorithm category.

Performance is defined as mean episode reward for all episodes.

We then compare the best 16 programs on grid world to four weak baselines (constant 0,-1,1 intrinsic reward and Gaussian noise reward) and the three published algorithms expressible in our language (shown in figure 2 ).

We run two trials for each algorithm and pool all results in each category to get a confidence interval for the mean of that category.

All trials used the reward combiner found on lunar lander.

For both environments we find that the performance of our top programs is statistically equivalent to published work and significantly better than the weak baselines, confirming that we meta-learned good curiosity programs.

Note that we meta-trained our intrinsic curiosity programs only on one environment (GridWorld) and showed they generalized well to other, very different, environments.

Adding more more metatraining tasks would be as simple as standardising the performance within each task (to make results comparable) and then select the programs with best mean performance.

We chose to only meta-train on a single, simple, task because it (surprisingly!) already gave great results; highlighting the broad generalization of meta-learning program representations.

In some regards our work is similar to neural architecture search (NAS) (Stanley & Miikkulainen, 2002; Elsken et al., 2018; Pham et al., 2018; Zoph & Le, 2016) or hyperparameter optimization for deep networks (Mendoza et al., 2016) , which aim at finding the best neural network architecture and hyper-parameters for a particular task.

However, in contrast to most (but not all, see ) NAS work, we want to generalize to many environments instead of just one.

Moreover, we search over programs, which include non-neural operations and data structures, rather than just neural-network architectures, and decide what loss functions to use for training.

Our work also resembles work in the AutoML community (Hutter et al., 2018 ) that searches in a space of programs, for example in the case of SAT solving (KhudaBukhsh et al., 2009) or auto-sklearn (Feurer et al., 2015) .

Although we take inspiration from ideas in that community (Jamieson & Talwalkar, 2016; Li et al., 2016) , our algorithms also specify their own optimization objectives (vs being specified by the user) which need to work well in syncrony with an expensive deep RL algorithm.

There has been work on meta-learning with genetic programming (Schmidhuber, 1987) , searching over mathematical operations within neural networks (Ramachandran et al., 2017; Gaier & Ha, 2019) , searching over programs to solve games (Wilson et al., 2018; Kelly & Heywood, 2017; and to optimize neural network weights (Bengio et al., 1995; Bello et al., 2017) , and neural networks that learn programs (Reed & De Freitas, 2015; Pierrot et al., 2019) .

In contrast, our work uses neural networks as basic operations within larger algorithms.

Finally, modular metalearning (Alet et al., 2018) trains the weights of small neural modules and transfers to new tasks by searching for a good composition of modules using a relatively simple composition scheme; as such, it can be seen as a (restricted) dual of our approach.

There has been much interesting work in designing intrinsic curiosity algorithms.

We take inspiration from many of them to design our domain-specific language.

In particular, we rely on the idea of using neural network training as an implicit memory, which scales well to millions of time-steps, as well as buffers and nearest-neighbour regressors.

As we showed in figure 2 we can represent several prominent curiosity algorithms.

We can also generate meaningful algorithms similar to novelty search (Lehman & Stanley, 2008) and EX 2 (Fu et al., 2017) ; which include buffers and nearest neighbours.

However, there are many exploration algorithm classes that we do not cover, such as those focusing on generating goals (Srivastava et al., 2013; Kulkarni et al., 2016; Florensa et al., 2018) , learning progress (Oudeyer et al., 2007; Schmidhuber, 2008; Azar et al., 2019) , generating diverse skills (Eysenbach et al., 2018) , stochastic neural networks (Florensa et al., 2017; Fortunato et al., 2017) , count-based exploration (Tang et al., 2017) or object-based curiosity measures (Forestier & Oudeyer, 2016) .

Finally, part of our motivation stems from Taïga et al. (2019) showing that some bonus-based curiosity algorithms have trouble generalising to new environments.

Related work on parametric-based meta-RL and efforts to increase its generalization can be found in appendix B. More relevant to our work, there have been research efforts on meta-learning exploration policies.

Duan et al. (2016); Wang et al. (2017) learn an LSTM that explores an environment for one episode, retains its hidden state and is spawned in a second episode in the same environment; by training the network to maximize the reward in the second episode alone it learns to explore efficiently in the first episode.

improves their exploration and that of Finn et al. (2017) by considering the importance of sampling in RL policies.

combine gradient-based meta-learning with a learned latent exploration space in which they add structured noise for meaningful exploration.

Closer to our formulation, Zheng et al. (2018) parametrize an intrinsic reward function which influences policy-gradient updates in a differentiable manner, allowing them to backpropagate through a single step of the policy-gradient update to optimize the intrinsic reward function for a single task.

In contrast to all three of these methods, we search over algorithms, which will allows us to generalize more broadly and to consider the effect of exploration on up to 10 5 − 10 6 time-steps instead of the 10 2 − 10 3 of previous work.

Finally, Chiang et al. (2019); have a setting similar to ours where they modify reward functions over the entire agent's lifetime, but instead of searching over intrinsic curiosity algorithms they tune the parameters of a hand-designed reward function.

Probably closest to our work, in evolved policy gradients meta-learn a neural network that computes a loss function based on interactions of the agent with an environment.

The weights of this network are optimized via evolution strategies to efficiently optimize new policies from scratch to satisfy new goals.

They show that they can generalize more broadly than MAML and RL 2 by meta-training a robot to go to different positions to the east of the start location and then meta-test making the robot quickly learn to go to a location to the west.

In contrast, we showed that by meta-learning programs, we can generalize between radically different environments, not just goal variations of a single environment.

For all methods transferring parametric representations, it is not clear how one would adapt the learned neural networks to an unseen environment with different action dimensionality or a different observation space.

In contrast, algorithms leverage polymorphic data types that adapt the neural networks to the environment they are running in.

In this work we show that programs are a powerful, succinct, representation for algorithms for generating curious exploration, and these programs can be meta-learned efficiently via active search.

Results from this work are two-fold.

First, by construction, algorithms resulting from this search will have broad generalization and will thus be a useful default for RL settings, where reliability is key.

Second, the algorithm search code will be open-sourced to facilitate further research on exploration algorithms based on new ideas or building blocks, which can be added to the search.

In addition, we note that the approach of meta-learning programs instead of network weights may have further applications beyond finding curiosity algorithms, such as meta-learning optimization algorithms or even meta-learning meta-learning algorithms.

We have the following types.

Note that S and A get defined differently for every environment.

• R: real numbers such as r t or the dot-product between two vectors.

• R + : numbers guaranteed to be positive, such as the distance between two vectors.

The only difference to our program search between R and R + is in pruning programs that can optimize objectives without looking at the data.

For R + we check whether they can optimize down to 0, for R we check whether they can optimize to arbitrarily negative values.

• state space S: the environment state, such as a matrix of pixels or a vector with robot joint values.

The particular form of this type is adapted to each environment.

• action space A: either a 1-hot description of the action or the action itself.

The particular form of this type is adapted to each environment.

• feature-space F = R 32 : a space mostly useful to work with neural network embeddings.

For simplicity, we only have a single feature space.

• List [X] : for each type we may also have a list of elements of that type.

All operations that take a particular type as input can also be applied to lists of elements of that type by mapping the function to every element in the list.

Lists also support extra operations such as average or variance.

Note that X stands for the option of being F or A.

|a|+|c| .

RunningNorm keeps track of the variance of the input and normalizes by that variance.

A.3 TWO OTHER PUBLISHED ALGORITHMS COVERED BY OUR DSL

Most work on meta-RL has focused on learning transferable feature representations or parameter values for quickly adapting to new tasks (Finn et al., 2017; Finn, 2018; Clavera et al., 2019) or improving performance on a single task (Xu et al., 2018; Veeriah et al., 2019) .

However, the range of variability between tasks is typically limited to variations of the same goal (such as moving at different speeds or to different locations) or generalizing to different environment variations (such as different mazes or different terrain slopes).

There have been some attempts to broaden the spectrum of generalization, showing transfer between Atari games thanks to modularity (Fernando et al., 2017; Rusu et al., 2016) or proper pretraining (Parisotto et al., 2015) .

However, as noted by Nichol et al. (2018) , Atari games are too different to get big gains with current feature-transfer methods; they instead suggest using different levels of the game Sonic to benchmark generalization.

Moreover, Yu et al. (2019) recently proposed a benchmark of many tasks.

Wang et al. (2019) automatically generate different terrains for a bipedal walker and transfer policies between terrains, showing that this is more effective than learning a policy on hard terrains from scratch; similar to our suggestion in section 3.2.

In contrast to these methods, we aim at generalization between completely different environments, even between environments that do not share the same state and action spaces.

Figure 8: Predicting algorithm performance allows us to find the best programs faster.

We investigate the number of the top 1% of programs found vs. the number of programs evaluated, and observe that the optimized search (in blue) finds 88% of the best programs after only evaluating 50% of the programs (highlighted in green).

The naive search order would have only found 50% of the best programs at that point.

Figure 9 : In black, mean performance across 5 trials for all 26,000 programs evaluated (out of their finished trials).

In green mean plus one standard deviation for the mean estimate and in red one minus one standard deviation for the mean estimate.

On the right, you can see program means form roughly a gaussian distribution of very big noise (thus probably not significant) with a very small (between 0.5% and 1% of programs) long tail of programs with statistically significant performance (their red dots are much higher than almost all green dots), composed of algorithms leading to good exploration.

Figure 10: Top variant in preliminary search on grid world; variant on random network distillation using an ensemble of trained networks instead of a single one.

Figure 11 : Good algorithm found by our search (3 of the top 16 programs on grid world are variants of this program).

On its left part it does random network distillation but does not use that error as a reward.

Instead it does an extra prediction based on the state transition on the right and compares both predictions.

Notice that, to make both predictions, the same F → F network was used to map from the query to the target, thus sharing the weights between both predictions.

@highlight

Meta-learning curiosity algorithms by searching through a rich space of programs yields novel mechanisms that generalize across very different reinforcement-learning domains.
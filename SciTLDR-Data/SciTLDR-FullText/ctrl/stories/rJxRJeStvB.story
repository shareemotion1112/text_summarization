Can the success of reinforcement learning methods for simple combinatorial optimization problems be extended to multi-robot sequential assignment planning?

In addition to the challenge of achieving near-optimal performance in large problems, transferability to an unseen number of robots and tasks is another key challenge for real-world applications.

In this paper, we suggest a method that achieves the first success in both challenges for robot/machine scheduling problems.

Our method comprises of three components.

First, we show any robot scheduling problem can be expressed as a random probabilistic graphical model (PGM).

We develop a mean-field inference method for random PGM and use it for Q-function inference.

Second, we show that transferability can be achieved by carefully designing two-step sequential encoding of problem state.

Third, we resolve the computational scalability issue of fitted Q-iteration by suggesting a heuristic auction-based Q-iteration fitting method enabled by transferability we achieved.

We apply our method to discrete-time, discrete space problems (Multi-Robot Reward Collection (MRRC)) and scalably achieve 97% optimality with transferability.

This optimality is maintained under stochastic contexts.

By extending our method to continuous time, continuous space formulation, we claim to be the first learning-based method with scalable performance in any type of multi-machine scheduling problems; our method scalability achieves comparable performance to popular metaheuristics in Identical parallel machine scheduling (IPMS) problems.

Suppose that we are given a set of robots and seek to serve a set of spatially distributed tasks.

A reward is given for serving each task promptly -resulting in a time-decaying reward collection problem -or when completing the entire set of tasks -resulting in a makespan minimization problem.

As the capability to control and route individual robots has increased [Li (2017) ], efficient orchestration of robots arises as an important remaining concern for such problems.

Multi-robot planning problems.

In this paper, we focus on orchestration problems that can be formulated as robot planning problems.

A key assumption in such orchestration problems is that we are given information on the "duration of time required for an assigned robot to complete a task".

This duration may be deterministic (e.g. as in a Traveling Salesman Problem (TSP) or Vehicle Routing Problem (VRP)) or random with given probability distribution (c.f., [Omidshafiei et al. (2017) ]).

1 .

We call this duration the task completion time.

Due to their combinatorial nature, robot planning problems suffer from exponential computational complexity.

Even in the context of single-robot scheduling problems (e.g., TSP) scalability is a concern.

Planning for multiple robots exacerbates the scalability issue.

While scalable heuristic methods have been developed for various deterministic multi-robot planning problems (c.f., [Rossi Proposed methods.

In the seminal paper [Dai et al. (2017) ], the authors observed that combinatorial optimization problems such as TSP can be formulated as sequential decision making problems.

Decision making in such a sequential framework relies on an estimate of future costs Q(s, a) for an existing task sequence s and candidate next task a. With this estimate, given the prior decisions s at each decision step, they select the next task a to minimize the future cost estimate. [Dai et al. (2017) ]'s solution framework relies on the following three assumptions.

1) For each combinatorial optimization problem, one can heuristically choose how to induce a graph representation of (s, a).

In the case of TSP, the paper induces a fully connected graph for every possible next task.

2) This induced graph representation can be considered as a probabilistic graphical model (PGM) [Koller & Friedman (2009) ].

This PGM can be used with a graph-based mean-field inference method called structure2vec [Dai et al. (2016) ] to infer Q(s, a) for use in combinatorial optimization problems.

3) Inference of Q(s, a) can be learned by the reinforcement framework called fitted Q-iteration.

We create a solution framework to achieve scalability and transferability for multi-robot planning that builds in numerous directions upon the foundation of [Dai et al. (2017) ] as follows: 1.

State representation and mean-field inference theory for random PGM.

Instead of heuristically inducing a PGM, we show that a robot scheduling problem exactly induces a random PGM.

Since there exists no mean-field inference theory for random PGM, we develop the theory and corresponding new structure2vec iteration.

2.

Sequential encoding of information for transferability.

To achieve transferability in terms of the number of robots and tasks, we carefully design a two-step hierarchical mean-field inference [Ranganath et al. (2015) ].

Each step is designed to infer certain information.

The first step is designed to infer each task's relative graphical distance from the robots.

The second step is designed to infer Q(s, a) (a here refers to a joint assignment of robots).

While the first step is by its nature transferable to any number of tasks and robots, the transferability in inference of the second step is achieved by the scale-free characteristic of fitted Q-iteration [van Hasselt et al. (2015) ].

That is, the relative magnitudes of Q(s, a) values are sufficient to select an action a. 3.

Auction-based assignment.

Even if we can infer Q(s, a) precisely, the computation time required to select an action a using the maximum Q(s, a) operation exponentially increases as robots and tasks increase.

To resolve this issue, we suggest a heuristic auction that is enabled by the transferability of our Q(s, a) inference.

Even though this heuristic auction selects a with only polynomial computational complexity, it provides surprisingly good choices for a. (In fact, this heuristic auction increases the performance empirically relative to using the max operation.)

time ?? i to complete -we call this the processsing time.

This time is the same independent of which machine serves the task.

We incorporate one popular extension and allow 'sequence-dependent setup times'.

In this case, a machine must conduct a setup prior to serving each task.

The duration of this setup depends on the current task i and the task j that was previously served on that machine -we call this the setup time.

The completion time for each task is thus the sum of the setup time and processing time.

Under this setting, we solve the IPMS problem for make-span minimization as discussed in [Kurz et al. (2001) ].

That is, we seek to minimize the total time spent from the start time to the completion of the last task.

The IPMS formulation resembles our MRRC formulation in continuous-time and continuous-space and we relegate the detailed formulation to Appendix B.

In Section 2, we formulated multi-robot/machine planning problems as sequential joint assignment decision problems.

As in [Dai et al. (2017) ], we will select a joint assignment using a Q-function based policy.

Since we thus choose action a t with the largest inferred Q(s t , a t ) value in state s t , the development of a Q(s t , a t ) inference method is a key issue.

Toward this end and motivated by these robot planning problems, we provide new results in random PGM-based mean-field inference methods and a subsequent extension of the graph-neural network based inference method called structure2vec [Dai et al. (2016) ] in Section 3.1.

In Section 3.2, we discuss how a careful encoding of information using the extended structure2vec of Section 3.1 enables precise and transferable Q(s t , a t ) inference.

Since the computational complexity required to identify the best joint assignment is exponential with respect to the number of robots and tasks, Section 3.3 discusses how the transferability of our Q(s t , a t ) inference method enables a good action choice heuristic with polynomial computational complexity.

PGM.

Given random variables {X k }, suppose that joint distribution of {X k } can be factored as PGM-based mean-field inference.

One popular use of this PGM information is PGM-based mean-field inference.

In mean-field inference, we find a surrogate distribution Q(X 1 , . . .

, X n ) = i Q i (x i ) that has smallest Kullback-Leibler distance to original joint distribution P (X 1 , . . .

, X n ).

We then use this surrogate distribution to solve the original inference problem. [Koller & Friedman (2009)] shows that when we are given PGM information, {Q i (x i )} can be analytically computed by a fixed point equation.

Despite that this usefulness, in most inference problems it is unrealistic to assume we know or can infer probability distributions of a PGM.

This limitation was addressed in [Dai et al. (2016) ] using a method called structure2vec.

. [Dai et al. (2016) ] suggests that an inference problem with graph-structured data (e.g. a molecule classification problem) can be seen as a particular PGM structure that consists of two types of random variables.

One type of random variables {X k } is one that serves as input of inference problem (e.g. X k denotes atomic number of atom k).

Another type of random variables {H k } is latent random variable where H k is a latent random variable related to X k .

Existence of probabilistic relationships among {H k } are assumed heuristically from graph structure of data.

Then the particular PGM structure they assume is

, where V denotes the set of vertex indexes.

The goal of mean-field inference problem is to find a surrogate distribution Q k (h k ) for posterior marginal P ({h k }|{x k }).

However, we can't compute {Q k (h k )} since we are not given ?? (H k |H i ) nor ?? (H k |X k ).

To overcome this limitation, [Dai et al. (2016) ] develops a method called structure2vec that only requires the structure of the PGM for mean-field inference.

structure2vec embeds the mean-field inference procedure, i.e. fixed point iteration on {Q k (h k )}, into fixed point iterations of neural networks on vectors {?? k }.

Derivation of such fixed point iterations of neural networks can be found in Dai et al. (2016) and can be written as?? k = ?? W 1 x k + W 2 j =k?? j where ?? denotes Relu function and W denotes parameters of neural networks.

Robot scheduling as random PGM-based mean-field inference.

All applications of structure2vec in [Dai et al. (2016; 2017) ] heuristically decide the structure of PGM of each data point from its graph structure.

The key observation we make is that inference problems in robot scheduling exactly induce a 'random' PGM structure (to be precise, a 'random' Bayesian Network).

Given that we start from state s t and action a t , consider a random experiment "sequential decision making using policy ??".

In this experiment, we can define an event as 'How robots serve all the remaining tasks in which sequence'.

We call one such event a 'scenario'.

For each task t i ??? T t , define a random variable X i as 'a characteristic of task t i ' (e.g. when task i is served).

Given a scenario, the relationships among {X i } satisfy as a Bayesian Network.

For details, see Appendix C)

Note that we do not know which scenario will occur from time t and thus do not know which PGM will be realized.

Besides, the inference of probability of each scenario is challenging.

Putting aside this problem for a while, we first define a 'random PGM' and 'semi-cliques'.

Denote the set of all random variables in the inference problem as X = {X i }.

A random PGM is a probabilistic model of how a PGM is randomly chosen from a set of all possible PGMs on X 4 .

Next, denote the set of all possible probabilistic relationships on X as C X .

We call them 'semi-cliques'.

In robot scheduling problem, a semi-clique D ij ??? C X is a conditional dependence X i |X j .

The semi-clique D ij presents as an actual clique if and only if the robot which finishes task t i chooses task t j as the next task.

We will now prove that we don't have to infer the probability of each scenario, i.e. random PGM model itself.

The following theorem for mean-field inference with random PGM is an extension of mean-field inference with PGM [Koller & Friedman (2009)] and suggests that only a simple inference task is required: inference of the presence probability of each semi-cliques.

Theorem 1.

Random PGM based mean field inference Suppose we are given a random PGM on X = {X k }.

Also, assume that we know presence probability {p m } for all semi-cliques

where Z k is a normalizing constant and ?? m is the clique potential for clique m.

From this new result, we can develop the structure2vec inference method for random PGM.

As in [Dai et al. (2016) ], we restrict our discussion to when every semi-clique is between two random variables.

In this case, a semi-clique can be written as D ij with its presence probability p ij .

Lemma 1.

Structure2vec for random PGM.

Suppose we are given a random PGM model with X = {X k }.

Also, assume that we know presence probability {p ij } for all semi-cliques C X = {D ij }.

The fixed point iteration in Theorem 1 for posterior marginal P ({H k }|{x k }) can be embedded in a nonlinear function mapping with embedding vector?? k as

Proof of Thorem 1 and lemma 1.

For brevity, proofs are relegated to the Appendix D and E.

Corollary 1.

For a robot scheduling problem with set of tasks t i ??? T t , the random PGM representation for structure2vec in lemma 1 is ((T t , E {p ij } inference procedure employed in this paper is as follows.

Denote ages of task i, j as age i , age j .

Note that if we generate M samples of ij as {e

is an unbiased and consistent estimator of E[f ( ij , age i , age j )].

For each sample k, for each task i and task j, we form a vector of u k ij = (e k ij , age i , age j ) and compute

We obtain {p ij } from {g ij } using softmax.

Algorithm details are in Appendix F. In this section, we show how Q(s t , a t ) can be precisely and transeferably inferred using a two-step structure2vec inference method (For theoretical justifications on hierarchical variational inference, see Ranganath et al. (2015) ).

We here assume that we are given (T t , E T T t ) and inferred {p ij } so that Corollary 1 can be applied.

For brevity, we illustrate the inference procedure for the special case when task completion time is deterministic (Appendix G illustrates how we can combine random sampling to inference procedure to deal with task completion times as a random variable).

Step 1.

Distance Embedding.

The output vectors {?? 1 k } of structure2vec embeds a local graph information around that vector node [Dai et al. (2016) ].

We here focus on embedding information of robot locations around a task node and thus infer each task's 'relative graphical distance' from robots around it.

As the input of first structure2vec ({x k } in lemma 1), we only use robot assignment information (if t k is an assigned task, we set x k as 'task completion time of assignment'; if t k is not an assigned task:, we set x k = 0).

This procedure is illustrated in Figure 1 .

According to [Dai et al. (2016) ], the output vectors {?? 1 k } of structure2vec will include sufficient information about the relative graphical distance from all robots to each task.

Step 2.

Value Embedding.

The second step is designed to infer 'How much value is likely in the local graph around each task'.

Remind that vectors {?? 1 k }, output vectors of the first step, carries information about the relative graphical distance from all robots to each task.

We concatenate 'age' of each tasks {age k } to each corresponding vector in {?? 1 k } and use the resulting graph as an input ({x k } in lemma 1) of second structure2vec, as illustrated in Figure 1 .

Again, vectors {?? 2 k } of the output graph of second structure2vec operation embeds a local graph structure around each node.

Our intuition is that {?? 2 k } includes sufficient information about 'How much value is likely in the local graph around each task'.

Step 3.

Computing Q(s t , a t ).

To infer Q(s t , a t ), we aggregate the embedding vectors for all nodes, i.e.,?? 2 = k?? 2 k to get one vector?? 2 which embeds the 'value likeliness' of the global graph.

We then use a layer of neural network to map?? 2 into Q(s t , a t ).

The detailed algorithm of above whole procedure (combined with random task completion times) is illustrated in Appendix G.

Why are each inference steps transferable?

For the first step, it is trivial; the inference problem is a scale-free task.

In the second step, the 'value likeliness' will be underestimated or overestimated according to the ratio of (number of robots/number of tasks) in a local graph: underestimated if the ratio in training environment is smaller than the ratio in the testing environment; overestimated otherwise.

The key idea solving this problem is that this over/under-estimation does not matter in Q-function based action decision [van Hasselt et al. (2015) ] as long as the order of Q-function value among actions are the same.

While analytic justification of this order invariance is beyond this paper's scope, the fact that there is no over/underestimation issue in the first step inference problem helps this justification.

In Q-function based action choice, at each time-step t, we find an action with largest Q(s t , a t ).

We call this action choice operation 'max-operation'.

The problem in max-operation in the multi-robot setting is that the number of computation exponentially increases as the number of robots and tasks increases.

In this section, we show that transferability of Q-function inference enables designing an efficient heuristic auction that replaces max operation.

We call it auction-based policy(ADP) and denote it as ?? Q ?? , where Q ?? indicates that we compute ?? Q ?? using current Q ?? estimator.

At time-step t, a state s t is a graph G t = (R t ??? T t , E t ) as defined in section 2.1.

Our ADP, ?? Q ?? , finds an action a t (which is a matching in bipartite graph ((R t ??? T t ), E RT t ) of graph G t ) through iterations between two phases: the bidding phase and the consensus phase.

We start with a bidding phase.

All robots initially know the matching determined in previous iterations.

We denote this matching as Y, a bipartite subgraph of ((R t ???T t ), E RT t ).

When making a bid, a robot r i ignores all other unassigned robots.

For example, suppose robot r i considers t j for bidding.

For r i , Y ??? ij is a proper action (according to definition in section 2.1) in a 'unassigned robot-ignored' problem.

Robot r i thus can compute Q(s t , Y ??? ritj ) of 'unassigned robot-ignored' problem for all unassigned task t j .

If task t * is with the highest value, robot r i bids { rit * , Q(s t , Y ??? rit * )} to auctioneer.

Since number of robots ignored by r i is different at each iteration, transferability of Q-function inference plays key role.

The consensus phase is simple.

The auctioneer finds the bid with the best value, say { * , bid value with * }.

Then auctioneer updates everyone's Y as Y ??? { * }.

These bidding and consensus phases are iterated until we can't add an edge to Y anymore.

Then the central decision maker chooses Y as ?? Q ?? (s k ).

One can easily verify that the computational complexity of computing ?? Q ?? is O (|L R | |L T |), which is only polynomial.

While theoretical performance guarantee of this heuristic auction is out of this paper's scope, in section 5 we show that empirically this heuristic achieves near-optimal performance.

4 LEARNING ALGORITHM

In fitted Q-iteration, we fit ?? of Q ?? (s t , a t ) with stored data using Bellman optimality equation.

That is, chooses ?? that makes

small.

Note that every update of ?? needs at least one max-operation.

To solve this issue, we suggest a learning framework we call auction-fitted Q-iteration.

What we do is simple: when we update ??, we use auction-based policy(ADP) defined in section 3.3 instead of max-operation.

That is, we seek the parameter ?? that minimizes

How can we conduct exploration in Auction-fitted Q-iteration framework?

Unfortunately, we can't use -greedy method since such randomly altered assignment is very likely to cause a catastrophic result in problems with combinatorial nature.

In this paper, we suggest that parameter space exploration [Plappert et al. (2017) ] can be applied.

Recall that we use Q ?? (s k , a k ) to get policy ?? Q ?? (s k ).

Note that ?? denotes all neural network parameters used in the structure2vec iterations introduced in Section 5.

Since Q ?? (s k , a k ) is parametrized by ??, exploration with ?? Q ?? (s k ) can be performed by exploration with parameter ??.

Such exploration in parameter space has been introduced in the policy gradient RL literature.

While this method was originally developed for policy gradient based methods, exploration in parameter space can be particularly useful in auction-fitted Q-iteration.

The detailed application is as follows.

When conducting exploration, apply a random perturbation on the neural network parameters ?? in structure2vec.

The resulting a perturbation in the Q-function used for decision making via the auction-based policy ?? Q ?? (s k ) throughout that problem.

Similarly, when conducting exploitation, the current surrogate Q-function is used throughout the problem.

Updates for the surrogate Q-function may only occur after each problem is complete (and typically after a group of problems).

For MRRC, we conduct a simulation experiment for a discrete time, discrete state environment.

We use maze (see Figure 1 ) generator of UC Berkeley CS188 Pacman project [Neller et al. (2010) ] to generate large size mazes.

We generated a new maze for every training and testing experiments.

Under the deterministic environment, the robot succeeds its movement 100%.

Under stochastic environment, a robot succeeds its intended movement in 55% on the grid with dots and for every other direction 15% each; on the grid without dots, the rates are 70% and 10%.

As described in section 2, routing problems are already solved.

That is, each robot knows how to optimally (in expectation) reach a task.

To find an optimal routing policy, we use Dijkstra's algorithm for deterministic environments and dynamic programming for stochastic environments.

The central assignment decision maker has enough samples of task completion time for every possible route.

We consider two reward rules: Linearly decaying rewards obey f (age) = 200 ??? age until reaching 0, where age is the task age when served; For nonlinearly decaying rewards, f (t) = ?? t for ?? = 0.99.

Initial age of tasks were uniformly distributed in the interval [0, 100] .

Performance test.

We tested the performance under four environments: deterministic/linear rewards, deterministic/nonlinear rewards, stochastic/linear rewards, stochastic/nonlinear rewards.

There are three baselines used for performance test: exact baseline, heuristic baseline, and indirect baseline.

For the experiment with deterministic with linearly decaying rewards, an exact optimal solution for mixed-integer exists and can be used as a baseline.

We solve this program using Gurobi with 60-min cut to get the baseline.

We also implemented the most up-to-date heuristic for MRRC in [Ekici & Retharekar (2013) ].

For any other experiments with nonlinearly decaying rewards or stochastic environment, such an exact optimal solution or other heuristics methods does not exist.

In these cases, we should be conservative when talking about performance.

Our strategy is to construct a indirect baseline using a universally applicable algorithm called Sequential greedy algorithm (SGA) [Han-Lim Choi et al. (2009)] .

SGA is a polynomial-time task allocation algorithm that shows decent scalable performance to both linear and non-linear rewards.

For stochastic environments, we use mean task completion time for task allocation and re-allocate the whole tasks at every timesteps.

We construct our indirect baseline as 'ratio between our method and SGA for experiments with deterministic-linearly decaying rewards'.

Showing that this ratio is maintained for stochastic environments in both linear/nonlinear rewards suffices our purpose.

Table 1 shows experiment results for (# of robots, # of tasks) = (2, 20), (3, 20) , (3, 30), (5, 30), (5, 40), (8, 40), (8, 50); For linear/deterministic rewards, our proposed method achieves nearoptimality (all above 95% optimality).

While there is no exact or comparable performance baseline for experiments under other environments, indirect baseline (%SGA) at least shows that our method does not lose %SGA for stochastic environments compared with %SGA for deterministic environments in both linear and nonlinear rewards.

Scalability test.

We count the training requirements for 93% optimality for seven problem sizes (# of robots N R , # of tasks N T ) = (2, 20), (3, 20) , (5, 30), (5, 40), (8, 40), (8, 50) with deterministic/linearly decaying rewards (we can compare optimality only in this case).

As we can see in Table  2 , the training requirement shown not to scale as problem size increases.

Transferability test.

Suppose that we trained our learning algorithm with problems of three robots and 30 Tasks.

We can claim transferability of our algorithm if our algorithm achieves similar performance for testing with problems of 8 robots and 50 tasks when compared with the algorithm specifically trained with problems of 8 robots and 50 tasks, the same size as testing.

Table 3 shows our comprehensive experiment to test transeferability.

The results in the diagonals (where training size and testing size is the same) becomes a baseline, and we can compare how the networks trained with different problem size did well compare to those results.

We could see that lower-direction transfer tests (trained with larger size problem and tested with smaller size problems) shows only a small loss in performance.

For upper-direction transfer tests (trained with smaller size problem and tested with larger size problem), the performance loss was up 4 percent.

Ablation study.

There are three components in our proposed method: 1) a careful encoding of information using two-layers of structure2vec, 2) new structure2vec equation with random PGM and 3) an auction-based assignment.

Each component was removed from the full method and tested to check the necessity of the component.

We test the performance in a deterministic/linearly decaying rewards (so that there is an optimal solution available for comparison).

The experimental results are shown in Figure 2 .

While the full method requires more training steps, only the full method achieves near-optimal performance.

For IPMS, we test it with continuous time, continuous state environment.

While there have been many learning-based methods proposed for (single) robot scheduling problems, to the best our knowledge our method is the first learning method to claim scalable performance among machinescheduling problems.

Hence, in this case, we focus on showing comparable performance for large problems, instead of attempting to show the superiority of our method compared with heuristics specifically designed for IPMS (actually no heuristic was specifically designed to solve our exact problem (makespan minimization, sequence-dependent setup with no restriction on setup times)) For each task, processing times is determined using uniform [16, 64] .

For every (task i, task j) ordered pair, a unique setup time is determined using uniform [0, 32] .

As illustrated in section 2, we want to minimize make-span.

As a benchmark for IPMS, we use Google OR-Tools library Google (2012).

This library provides metaheuristics such as Greedy Descent, Guided Local Search, Simulated Annealing, Tabu Search.

We compare our algorithm's result with the heuristic with the best result for each experiment.

We consider cases with 3, 5, 7, 10 machines and 50, 75, 100 jobs.

The results are provided in Table 4 .

Makespan obtained by our method divided by the makespan obtained in the baseline is provided.

Although our method has limitations in problems with a small number of tasks, it shows comparable performance to a large number of tasks and shows its value as the first learning-based machine scheduling method that achieves scalable performance.

We presented a learning-based method that achieves the first success for multi-robot/machine scheduling problems in both challenges: scalable performance and tranferability.

We identified that robot scheduling problems have an exact representation as random PGM.

We developed a meanfield inference theory for random PGM and extended structure2vec method of Dai et al. (2016) .

To overcome the limitations of fitted Q-iteration, a heuristic auction that was enabled by transferability is suggested.

Through experimental evaluation, we demonstrate our method's success for MRRC problems under a deterministic/stochastic environment.

Our method also claims to be the first learning-based algorithm that achieves scalable performance among machine scheduling algorithms; our method achieves a comparable performance in a scalable manner.

Our method for MRRC problems can be easily extended to ride-sharing problems or package delivery problems.

Given a set of all user requests to serve, those problems can be formulated as a MRRC problem.

For both ride-sharing and package delivery, it is reasonable to assume that the utility of a user depends on when she is completely serviced.

We can model how the utility of a user decreases over time since when it appears and set the objective function of problems as maximizing total collected user utility.

Now consider a task 'deliver user (or package) from A to B'.

This is actually a task "Move to location A and then move to location B".

If we know the completion time distribution of each move (as we did for MRRC), the task completion time is simply the sum of two random variables corresponding to task completion time distribution of the moves in the task.

Indeed, ride-sharing or package delivery problems are of such tasks (We can ignore charging moves for simplicity, and also we don't have to consider simple relocation of vehicles or robots since we don't consider random customer arrivals).

Therefore, both ride-sharing problems and package delivery problems can be formulated as MRRC problems.

A MRRC WITH CONTINUOUS STATE/CONTINUOUS TIME SPACE FORMULATION, OR WITH SETUP TIME AND PROCESSING TIME

In continuous state/continuous time space formulation, the initial location and ending location of robots and tasks are arbitrary on R 2 .

At every moment at least a robot finishes a task, we make assignment decision for a free robot(s).

We call this moments as 'decision epochs' and express them as an ordered set (t 1 , t 2 , . . .

, t k , . . . ).

Abusing this notation slightly, we use (??) t k = (??) k .

Task completion time can consist of three components: travel time, setup time and processing time.

While a robot in the travel phase or setup phase may be reassigned to other tasks, we can't reassign a robot in the processing phase.

Under these assumptions, at each decision epoch robot r i is given a set of tasks it can assign itself: if it is in the traveling phase or setup phase, it can be assigned to any tasks or not assigned; if it is in the processing phase, it must be reassigned to its unfinished task.

This problem can be cast as a Markov Decision Problem (MDP) whose state, action, and reward are defined as follows:

R k is the set of all robots and T k is the set of all tasks; The set of directed edges

where a directed edge

is a random variable which denotes task completion time of robot i in R k to service task j in T k and a directed edge titj ??? E T T k denotes a task completion time of a robot which just finished serving task i in T k to service task j in T k .

E RT k contains information about each robot's possible assignments:

, where E ri t is a singleton set if robot i is in the processing phase and it must be assigned to its unfinished task, and otherwise it is the set of possible assignments from robot r i to remaining tasks that are not in the processing phase.

Action.

The action a k at decision epoch k is the joint assignment of robots given the current state s k = G k .

The feasible action should satisfy the two constraints: No two robots can be assigned to a task; some robots may not be assigned when number of robots are more than remaining tasks.

To best address those restrictions, we define an action a k at time t as a maximal bipartite matching in bipartite sub-graph ((R k ??? T k ), E RT k ) of graph G k .

For example, robot i in R k is matched with task j in T k in an action a k if we assign robot i to task j at decision epoch t. We denote the set of all possible actions at epoch k as A k .

Reward.

In MRRC, Each task has an arbitrarily determined initial age.

At each decision epoch, the age of each task increases by one.

When a task is serviced, a reward is determined only by its age when serviced.

Denote this reward rule as R(k).

One can easily see that whether a task is served at epoch k is completely determined by s k , a k and s k+1 .

Therefore, we can denote the reward we get with s k , a k and s k+1 as R(s k , a k , s k+1 ).

Objective.

We can now define an assignment policy ?? as a function that maps a state s k to action a k .

Given s 0 initial state, an MRRC problem can be expressed as a problem of finding an optimal assignment policy ?? * such that

As written in 2.2, IPMS is a problem defined in continuous state/continuous time space.

Machines are all identical, but processing times of tasks are all different.

In this paper, we discuss IPMS with 'sequence-dependent setup time'.

A machine's setup time required for servicing a task i is determined by its previously served task j.

In this case, the task completion time is the sum of setup time and processing time.

Under this setting, we solve IPMS problem for make-span minimization objective discussed in [Kurz et al. (2001) ] (The constraints are different in this problem though); That is, minimizing total time spent from start to end to finish all tasks.

Every time there is a finished task, we make assignment decision for a free machine.

We call this times as 'decision epochs' and express them as an ordered set (t 1 , t 2 , . . .

, t k , . . . ).

Abusing this notation slightly, we use (??) t k = (??) k .

Task completion time for a machine to a task consists of two components: processing time and setup time.

While a machine in setup phase may be reassigned to another task, we can't reassign a machine in the processing phase.

Under these assumptions, at each epoch, a machine r i is given a set of tasks it can assign: if it is in the setup phase, it can be assigned to any tasks or not assigned; if it is in the processing phase, it must be reassigned to its unfinished task.

This problem can be cast as a Markov Decision Problem (MDP) whose state, action, and reward are defined as follows:

State.

State s k at decision epoch k is a directed graph G k = (R k ??? T k , E k ): R k is the set of all machines and T k is the set of all tasks; The set of directed edges

where a directed edge ritj ??? E RT k is a random variable which denotes task completion time of machine i in R k to service task j in T k and a directed edge titj ??? E T T k denotes a task completion time of a machine which just finished serving task i in T k to service task j in T k .

E RT k contains information about each robot's possible assignments: E RT k = ??? i E ri k , where E ri k is a singleton set if machine i is in the processing phase and it must be assigned to its unfinished task, and otherwise it is the set of possible assignments from machine r i to remaining tasks that are not in the processing phase.

Action.

Defined the same as MRRC with continuous state/time space.

Reward.

In IPMS, time passes between decision epoch t and decision epoch t + 1.

Denote this time as T t .

One can easily see that T t is completely determined by s k , a k and s k+1 .

Therefore, we can denote the reward we get with s k , a k and s k+1 as T (s k , a k , s k+1 ).

Objective.

We can now define an assignment policy ?? as a function that maps a state s k to action a k .

Given s 0 initial state, an MRRC problem can be expressed as a problem of finding an optimal assignment policy ?? * such that

T (s k , a k , s k+1 ) |s 0 .

Here we analytically show that robot scheduling problem randomly induces a random Bayesian Network from state s t .

Given starting state s t and action a t , a person can repeat a random experiment of "sequential decision making using policy ??".

In this random experiment, we can define events '

How robots serve all remaining tasks in which sequence'.

We call such an event a 'scenario'.

For example, suppose that at time-step t we are given robots {A, B}, tasks {1, 2, 3, 4, 5}, and policy ??.

One possible scenario S * can be {robot A serves task 3 ??? 1 ??? 2 and robot B serves task 5 ??? 4}. Define random variable X k a task characteristic, e.g. 'The time when task k is serviced'.

The question is, 'Given a scenario S * , what is the relationship among random variables {X k }'?

Recall that in our sequential decision making formulation we are given all the 'task completion time' information in the s t description.

Note that, task completion time is only dependent on the previous task and assigned task.

In our example above, under scenario S * 'when task 2 is served' is only dependent on 'when task 1 is served'.

That is, P (X 2 |X 1 , X 3 , S * ) = P (X 2 |X 1 , S * ).

This relationship is called 'conditional independence'.

Given a scenario S * , every relationship among {X i |S * } can be expressed using this kind of relationship among random variables.

A graph with this special relationship is called 'Bayesian Network' [Koller & Friedman (2009) We first define necessary definitions for our proof.

In a random PGM, a PGM is chosen among all possible PGMs on {X k } and semi-cliques C. Denote the set of all possible factorization as F = {S 1 , S 2 , ..., S N } where a factorization with index k is denoted as S k ??? C. Suppose we are given P ({S = S m }).

<|TLDR|>

@highlight

RL can solve (stochastic) multi-robot/scheduling problems scalably and transferably using graph embedding
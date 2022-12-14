Although challenging, strategy profile evaluation in large connected learner networks is crucial for enabling the next wave of machine learning applications.

Recently, $\alpha$-Rank, an evolutionary algorithm, has been proposed as a solution for ranking joint policy profiles in multi-agent systems.

$\alpha$-Rank claimed scalability through a polynomial time implementation with respect to the total number of pure strategy profiles.

In this paper, we formally prove that such a claim is not grounded.

In fact, we show that $\alpha$-Rank exhibits an exponential complexity in number of agents, hindering its application beyond a small finite number of joint profiles.

Realizing such a limitation, we contribute by proposing a scalable evaluation protocol that we title  $\alpha^{\alpha}$-Rank.

Our method combines evolutionary dynamics with stochastic optimization and double oracles for \emph{truly} scalable ranking with linear (in number of agents) time and memory complexities.

Our contributions allow us, for the first time, to conduct large-scale evaluation experiments of multi-agent systems, where we show successful results on large joint strategy profiles with sizes in the  order of $\mathcal{O}(2^{25})$ (i.e., $\approx \text{$33$ million strategies}$) -- a setting not evaluable using current techniques.

Scalable policy evaluation and learning have been long-standing challenges in multi-agent reinforcement learning (MARL) with two difficulties obstructing progress.

First, joint-strategy spaces exponentially explode when a large number of strategic decision-makers is considered, and second, the underlying game dynamics may exhibit cyclic behavior (e.g. the game of Rock-Paper-Scissor) rendering an appropriate evaluation criteria non-trivial.

Focusing on the second challenge, much work in multi-agent systems followed a game-theoretic treatment proposing fixed-points, e.g., Nash (Nash et al., 1950) equilibrium, as potentially valid evaluation metrics.

Though appealing, such measures are normative only when prescribing behaviors of perfectly rational agents -an assumption rarely met in reality Grau-Moya et al. (2018) ; Wen et al. (2019) .

In fact, many game dynamics have been proven not converge to any fixed-point equilibria (Hart & Mas-Colell, 2003; Viossat, 2007) , but rather to limit cycles (Palaiopanos et al., 2017; Bowling & Veloso, 2001) .

Apart from these aforementioned inconsistencies, solving for a Nash equilibrium even for "simple" settings, e.g. two-player games is known to be PPAD-complete (Chen & Deng, 2005 ) -a demanding complexity class when it comes to computational requirements.

To address some of the above limitations, recently proposed ??-Rank as a graph-based game-theoretic solution to multi-agent evaluation.

??-Rank adopts Markov Conley Chains to highlight the presence of cycles in game dynamics, and attempts to compute stationary distributions as a mean for strategy profile ranking.

Though successful in small-scale applications, ??-Rank severely suffers in scalability contrary to polynomial time claims made in .

In fact, we show that ??-Rank exhibits exponential time and memory complexities shedding light on the small-scale empirical study conducted in , whereby the largest reported game included only four agents with four available strategies each.

In this work, we put forward ?? ?? -Rank as a scalable alternative for multi-agent evaluation with linear time and memory demands.

Our method combines numerical optimization with evolutionary game theory for a scalable solver capable of handling large joint spaces with millions of strategy profiles.

To handle even larger profiles, e.g., tens to hundreds of millions, we further introduce an oracle Figure 1: Example of population based evaluation on N = 3 learners each with 3 strategies and 5 copies.

a) Each population obtains a fitness value P i depending on the strategies chosen, b) mutation strategy (red star), and c) population either selecting original strategy, or adopting the novel strategy.

( McMahan et al., 2003) mechanism transforming joint evaluation into a sequence of incremental sub-games with varying sizes.

Given our algorithmic advancements, we justify our claims in a largescale empirical study involving systems with O(2 25 ) possible strategy profiles.

We first demonstrate the computation advantages of ?? ?? -Rank on varying size stochastic matrices against other implementations in Numpy, PyTorch, and OpenSpiel .

With these successes, we then consider experiments unsolvable by current techniques.

Precisely, we evaluate multi-agent systems in self-driving and Ising model scenarios each exhibiting a prohibitively-large strategy space (i.e., order of thousands for the former, and tens of millions for the latter).

Here, we again show that ?? ?? -Rank is capable of recovering correct strategy ranking in such complex domains.

In ??-Rank, strategy profiles of N agents are evaluated through an evolutionary process of mutation and selection.

Initially, agent populations are constructed by creating multiple copies of each learner i ??? {1, . . .

, N } assuming that all agents (in one population) execute the same unified policy.

With this, ??-Rank then simulates a multi-agent game played by randomly sampled learners from each population.

Upon game termination, each participating agent receives a payoff to be used in policy mutation and selection after its return to the population.

Here, the agent is faced with a probabilistic choice between switching to the mutation policy, continuing to follow its current policy, or randomly selecting a novel policy (other than the previous two) from the pool.

This process repeats with the goal of determining an evolutionary strong profile that spreads across the population of agents.

Each of the above three phases is demonstrated in Fig. 1 on a simple example of three agents -depicted by different symbols -each equipped with three strategies -depicted by the colors.

Mathematical Formalisation, Notation, and Definitions:

We next formalize the process posed by ??-Rank, which will lead to its limitations, and also pave the way for our own proposed solution.

We consider N agents with each agent i having access to a set of strategies of size s i .

At round k of the evaluation process, we denote the strategy profile for agent i by S

th allowed policy of the learner.

X represents the set of states and A i is the set of actions for agent i. With this, we define a joint strategy profile for all participating agents as policies belonging to the joint strategy pool,

i and j i ??? {1, . . .

, s i }.

To evaluate performance, we assume each agent is additionally equipped with a payoff (reward) function P

is the pool of joint strategies so to accommodate the effect of other learners on the i th player performance further complicating the evaluation process.

Finally, given a joint profile ??

joint , we define the corresponding joint payoff to be the collection of all individual payoff functions, i.e., P

After attaining rewards from the environment, each agent returns to its population and faces a choice between switching to a mutation policy, exploring a novel policy, or sticking to the current one Such a choice is probabilistic and defined proportional to rewards.

Precisely, agent i adopts

i,c , ??

with ?? ??? R + denoting an exploration parameter 1 , ??

i representing policies followed by other agents at round k, and ?? ??? R + an intensity ranking parameter.

As noted in , one can relate the above switching process to a random walk on a Markov chain with states defined as elements in S [k] joint and transition probabilities through payoff functions.

In particular, each entry in the transition probability matrix T ??? R

refers to the probability of one agent switching from one policy in a relation to attained payoffs.

Precisely, consider any two joint strategy profiles ?? [k] joint and?? [k] joint that differ in only one individual strategy for the i th agent, i.e., there exists a unique agent such that ??

???i defining the probability that one copy of agent i with strategy ?? [k] i,a invades the population with all other agents (in that population) playing??

???i , such a probability is formalized as:

, and 1 /m otherwise,

with m being the size of the population.

So far, we presented relevant derivations for the (??

joint ) entry of the state transition matrix when exactly the i th agent differs in exactly one strategy.

Having one policy change, however, only represents a subset of allowed variations, where two more cases need to be considered.

Now we restrict out attention to variations in joint policies involving more than two individual strategies, i.e., ??

joint ,??

[k] joint = 0.

Consequently, the remaining event of self-transitions can be thus written as

.

Summarising the above three cases, we can then write the (??

joint )'s entry of the Markov chain's transition matrix as:

The goal in ??-Rank is to establish an ordering in policy profiles dependent on evolutionary stability of each joint strategy.

In other words, higher ranked strategies are these that are prevalent in populations with higher average times.

Formally, such a notion can be easily derived as the limiting vector

v 0 of our Markov chain when evolving from an initial distribution v 0 .

Knowing that the limiting vector is a stationary distribution, one can calculate strategy rankings as the solution to the following eigenvector problem:

Limitations of ??-Rank: Though the work in seeks to determine a solution to the above problem, it is worth mentioning that ??-Rank suffers from one major drawback-the scalability-that we remedy in this paper.

We note that the solution methodology in ??-Rank is in fact unscalable to settings involving more than a hand-full of agents.

Particularly, authors claim polynomial complexities of their solution to the problem in Eqn.

3.

Though polynomial, such a complexity, however, is polynomial in an exponential search space, i.e., the space of joint strategy profiles.

As such, the polynomial complexity claim is not grounded, and need to be investigated.

In short, ??-Rank exhibits an exponential (in terms of the number of agents) complexity for determining a ranking, thus rendering it inapplicable to settings involving more than a small amount of agents.

In what comes next, we first discuss traditional approaches that could help solve the Eqn.

3; soon we realize an off-the-shelve solution is unavailable.

Hence, we commence to propose an efficient evaluation algorithm, i.e., ?? ?? -Rank, based on stochastic optimization with suitable complexities and rigorous theoretical guarantees.

At the end, we propose a search heuristic to further scale up our method by introducing oracles and we name it by ?? ?? -Oracle.

The problem of computing stationary distributions is a long-standing classical problem from linear algebra.

Various techniques including power method, PageRank, eigenvalue decomposition, and mirror descent can be utilized for solving the problem in Eqn.

3.

As we demonstrate next, any such implementation scales exponentially in the number of learners, as we summarize in Table 1 .

Power Method.

One of the most common approaches to computing the solution in Eqn.

3 is the power method.

Power method computes the stationary vector v k by constructing a sequence {v j } j???0 from a non-zero initial vector v 0 by applying

Though viable, we first note that the power method exhibits an exponential memory complexity in terms of the number of agents.

To formally derive the bound, define n to represent the total number of joint strategy profiles (i.e., n = |S

Analyzing its time complexity, on the other hand, requires a careful consideration that links convergence rates with the resulting graph topology of the Markov chain.

Precisely, the convergence rate of the power method is dictated by the second-smallest eigenvalue of the normalized Laplacian, L G , of the graph, G, associated to the Markov chain in Section 2, i.e., v joint and transition probability matrix T [k] .

The second-smallest eigenvalue of the normalized Laplacian of the graph associated with the Markov chain is given by:

, with s i denoting the number of strategies of agent i.

Due to space constraints, the full proof of the above lemma is refrained to Appendix A.1.

The importance of Lemma A.1 is that the resultant time complexity of the power method is also exponential of the form O n ?? log (

PageRank.

Inspired by ranking web-pages on the internet, one can consider PageRank (Page et al., 1999) for computing the solution to the eigenvalue problem in Eqn.

3. Applied to our setting, we first realize that the memory is analogous to the power method that is O(m size ) = O n

Eigenvalue Decomposition.

Apart from the above, we can also consider the problem as a standard eigenvalue decomposition task (also what the original ??-Rank is implemented according to ) and adopt the method in Coppersmith & Winograd (1990) to compute the stationary distribution.

Unfortunately, state-of-the-art techniques for eigenvalue decomposition also require exponential memory and exhibit a time complexity of the form

Clearly, these bounds restrict ??-Rank to small number of agents N .

Mirror Descent.

The ordered subsets mirror descent (Ben-Tal et al., 2001 ) requires at each iteration a projection on standard n???dimensional simplex:

As stated in the paper, the computing of this projection requires O(n log n) time.

In our setting, n = N i=1 s i is the total number of joint strategy profiles.

Hence, the projection step is exponential in the number of agents N .

This makes mirror descent inapplicable for ??-Rank when N is large.

Rather than seeking an exact solution to the problem in Eqn.

3, one can consider approximate solvers by defining a constraint optimization objective:

The constrained objective in Eqn.

4 simply seeks a vector x minimizing the distance between x, itself, and

while ensuring that x lies on an ndimensional simplex (i.e., x T 1 = 1, and x 0).

Due to time and memory complexities required for computing exact solutions, we focus on determining an approximate vectorx defined to be the solution to the following relaxed problem of Eqn.

4:

The optimization problem in Eqn.

5 can be solved using a barrier-like technique that we detail below.

Before that, it is instructive to clarify the connection between the original and the relaxed problems Proposition: [Connections to Markov Chain]

Letx be a solution to the relaxed optimization problem in Eqn.

5.

Then,x/||x||1 = v [k] is the stationary distribution to the Markov chain in Section 2.

Importantly, the above proposition allows us to focus on solving the problem in Eqn.

5 which only exhibits inequality constraints.

Problems of this nature can be solved by considering a barrier function leading to an unconstrained finite sum minimization problem.

To do so, denoting b

[k]

i to be the i th row of T [k] ,T ??? I, we can write

.

Introducing logarithmic barrier-functions, with ?? > 0 being a penalty parameter, we arrive at

Eqn.

6 is a standard finite minimization problem that can be solved using any off-the-shelve stochastic optimization algorithm, e.g., stochastic gradients, ADAM (Kingma & Ba, 2014) among others.

A stochastic gradient execution involves sampling a strategy profile i t ??? [1, . . .

, n] at iteration t, and then executing a descent step:

, with ??? x f it (x t ) being a sub-sampled gradient of Eqn.

6, and ?? being a scheduled penalty parameter with ?? t+1 = ??t /?? for some ?? > 1,

See Phase I in Algorithm 1 for the pseudo-code.

We can further derive a convergence theorem of:

Theorem: [Convergence of Barrier Method]

Letx ?? be the output of a gradient algorithm descending in the objective in Eqn.

6, after T iterations, then

where expectation is taken w.r.t.

all randomness of a stochastic gradient implementation, and ?? > 1 is a decay-rate for ??, i.e., ??

, penalty parameter ?? ??? O( ), ?? decay rate ?? > 1, total number of joint strategy profiles n, and a constraint relaxation term ??.

Oracle Parameters: initialize a subset of strategy pools for all agents {S [0] i } by randomly sampling from {S i } 2: Set outer iteration count k = 0 3: while stopping criteria do:

Phase I: Scalable Policy Evaluation (Section 3.2):

for t = 0 ??? T ??? 1 do:

Uniformly sample one strategy profile i

Update solution x

10:

Phase II (if turned on): Scalable Policy Evaluation with Oracle (Section 3.3):

for each agent i do:

12:

Compute the best-response strategy ?? * i by solving Eqn.

8.

Update strategy pools for each agent i as S

Set k = k + 1 15: Return: Best performing strategy profile ?? joint, across all agents.

The proof of the above theorem (see the full proof in Appendix A.2) is interesting by itself, a more important aspect is the memory and time complexity implications posed by our algorithm.

Theorem A.2 implies that after T = O( 1 / 2 ) iterations with being a precision parameter, our algorithm outputs a vectorx ?? 0 such that

Moreover, one can easily see 3 that after T steps, the overall time and memory complexities of our update rules are given by

eventually leads to a memory complexity of Table.

1).

Hence, our algorithm is able to achieve an exponential reduction, in terms of number of agents, in both memory and time complexities.

So far, we have presented scalable multi-agent evaluations through stochastic optimization.

We can further boost scalability (to tens of millions of joint profiles) of our method by introducing an oracle mechanism.

The heuristic of oracles was first introduced in solving large-scale zero-sum matrix games (McMahan et al., 2003) .

The idea is to first create a restricted sub-game in which all players are only allowed to play a restricted number of strategies, which are then expanded by adding incorporating each of the players' best-responses to opponents; the sub-game will be replayed with agents' augmented strategy pools before a new round of best responses is found.

The worse-case scenario of introducing oracles would be to solve the original evaluation problem in full size.

The best response is assumed to be given by an oracle that can be simply implemented by a grid search.

Precisely, given the top-rank profile ??

at iteration k, the goal for agent i is to select 4 the optimal ?? * i from the pre-defined strategy pool S i to maximize the reward

with x [k]

h denoting the state, u

???i,h ) denoting the actions from agent i and the opponents, respectively.

The heuristic of solving the full game from restricted sub-games is crucial especially when it is prohibitively expensive to list all joint-strategy profiles, e.g., in scenarios involving tens-of-millions of joint profiles.

For a complete exposition, we summarize the pseudo-code in Algorithm 1.

In the first phase, vanilla ?? ?? -Rank is executed (lines 4-9), while in the second (lines 11 -13), ?? ?? -Rank with Oracle (if turned on) is computed.

To avoid any confusion, we refer to the latter as ?? ?? -Oracle.

Note that even though in the two-player zero-sum games, the oracle algorithm (McMahan et al., 2003) is guaranteed to converge to the minimax equilibrium.

Providing valid convergence guarantees for ?? ?? -Oracle is an interesting direction for future work.

In this paper, we rather demonstrate the effectiveness of such an approach in a large-scale empirical study as shown in Section 4.

In this section, we evaluate the scalability properties of ?? ?? -Rank 5 .

Precisely, we demonstrate that our method is capable of successfully recovering optimal policies in self-driving car simulations and in the Ising model where strategy spaces are in the order of up to tens-of-millions of possible strategies.

We note that these sizes are well beyond the capabilities of state-of-the-art methods, e.g., ??-Rank ) that considers at maximum four agents with four strategies, or AlphaStar which handles about 600 strategies as detailed in Vinyals et al. (2019) .

Sparsity Data Structures.

During the implementation phase, we realised that the transition probability, T [k] , of the Markov chain induces a sparsity pattern (each row and column in

2) that if exploited can lead to significant speed-ups.

To fully leverage such sparsity, we tailored a novel data structure for sparse storage and computations needed by Algorithm 1.

More details can be found in Appendix B.1.

Correctness of Ranking Results.

Before conducting large-scale sophisticated experiments, it is instructive to validate the correctness of our results on the simple cases especially those reported by .

We therefore test on three normal-form games.

Due to space constraints, we refrain the full description of these tasks to Appendix B.2.

Fig. 2 shows that, in fact, results generated by ?? ?? -Rank, the Phase I of Algorithm 1, are consistent with ??-Rank's results.

Complexity Results on Random Matrices.

We measured the time and memory needed by our method for computing the stationary distribution with varying sizes of simulated random matrices.

Baselines includes eigenvalue decomposition from Numpy, optimization tools in PyTorch, and ??-Rank from OpenSpiel .

For our algorithm we terminated execution with gradient norms being below a predefined threshold of 0.01.

According to Fig. 3 , ?? ?? -Rank can achieve three orders of magnitude reduction compared to eigenvalue decomposition in terms of time.

Most importantly, the performance gap keeps developing with the increasing matrix size.

Autonomous Driving on Highway: High-way (Leurent, 2018) provides an environment for simulating self-driving scenarios with social vehicles designed to mimic real-world traffic flow as strategy pools.

We conducted a ranking experiment involving 5 agents each with 5 strategies, i.e. a strategy space in the order of O(5 5 ) (3125 possible strategy profiles).

Agent strategies varied between "rational" and "dangerous" drivers, which we encoded using different reward functions during training (complete details of defining reward functions can be found in Appendix C.2).

Under this setting, we know, upfront, that optimal profile corresponds to all agents is five rational drivers.

Cars trained using value-iteration and rewards averaged from 200 test trails were reported.

Due to the size of the strategy space, we considered both ?? ?? -Rank and ?? ?? -Oracle.

We set ?? ?? -Oracle to run 200 iterations of gradient updates in solving the top-rank strategy profile (Phase I in Algorithm 1).

Results depicted in Fig. 4 (a) clearly demonstrate that both our implementations are capable of recovering the correct highest ranking strategy profile.

We also note that though such sizes are feasible using ??-Rank and the power-method, our results achieve 4 orders of magnitude reduction in total number of iterations.

Ising Model Experiment: The Ising model (Ising, 1925) is the model for describing ferromagnetism in statistical mechanics.

It assumes a system of magnetic spins, where each spin a j is either an upspin, ???, or down-spin, ???. The system energy is defined by E(a, h)

with h j and ?? being constant coefficients.

The probability of one set of spin configuration is P (a) = exp(???E(a,h)/?? ) a exp(???E(a,h)/?? ) where ?? is the environmental temperature.

Finding the equilibrium of the system is notoriously hard because it is needed to enumerate all possible configurations in computing P (a).Traditional approaches include Markov Chain Monte Carlo (MCMC).

An interesting phenomenon is the phase change, i.e., the spins will reach an equilibrium in the low temperatures, with the increasing ?? , such equilibrium will suddenly break and the system becomes chaotic.

Here we try to observe the phase change through multi-agent evaluation methods.

We assume each spins as an agent, and the reward to be r j = h j a j + ?? 2 k =j a j a k , and set ?? = 1 ?? to build the link between Eqn.

1 and P (a).

We consider the top-rank strategy profile from ?? ?? -Oracle as the system equilibrium and compare it against the ground truth from MCMC.

We consider a five-by-five 2D model which induces a prohibitively-large strategy space of size 2 25 (tens of millions) to which the existing baselines, including ?? ?? -Rank on the single machine, are inapplicable.

Fig. 4(b) illustrates that our method identifies the same phase change as what MCMC suggests.

We show an example of how ?? ?? -Oracle's top-ranked profile finds the system equilibrium in Fig. 4 (c) at ?? = 1.

In this paper, we demonstrated that the approach in exhibits exponential time and memory complexities.

We then proposed ?? ?? -Rank as a scalable solution for multi-agent evaluation with linear time and memory demands.

In a set of experiments, we demonstrated that our method is truly scalable capable of handling large strategy spaces.

There are a lot of interesting avenues for future research.

First, we plan to theoretically analyze convergence properties of the resulting oracle algorithm, and further introduce policy learning through oracles.

Second, we plan take our method to the real-world by conducting multi-robot experiments.

joint and transition probability matrix T [k] .

The second-smallest eigenvalue of the normalized Laplacian of the graph associated with the Markov chain is given by:

, with s i denoting the number of strategies of agent i.

Proof :

For simplicity we drop round index k in the below derivation.

Notice, the underlying graph for the constructed Markov Chain can be represented as a Cartesian product of N complete graphs

Indeed, two vertices ?? [k] ,?? [k] ??? G are connected by the edge if and if only these joint strategy profiles differ in at most one individual strategy, i.e ???!i ??? {1, . . .

, N } :

???i }.Hence, the spectral properties of G can be described in terms of spectral properties of K si as follows (Barik et al., 2015) :

) is the i th eigenvalue of the unnormalized Laplacian of the complete graph K sj and ?? i,j is the corresponding eigenvector 7 .

The spectrum of unnormalized Laplacian of the complete graph K si is given by Spectr(K si ) = {0, s i ??? 1} and the only eigenvector corresponding to zero eigenvalue is 1 ??? R si .

Therefore, the minimum non-zero eigenvalue of unnormalized Laplacian of G is given by min i s i ??? 1.

Finally, due to the fact that G is a regular graph (with degree of each node is equal to N i=1 s i ??? N + 1), the smallest non-zero eigenvalue of the normalized Laplacian of G is given by

Giving this result, the overall time complexity of Power Method is bounded by O n ?? log

= O (log n).

As for the memory complexity, Power Method requires has the same requirements as PageRank algorithm.

8 These results imply that Power Method scales exponentially with number of agents N , and therefore, inapplicable when N is large.

Theorem: [Convergence of Barrier Method]

Letx ?? be the output of a gradient algorithm descending in the objective in Eqn.

6, after T iterations, then

where expectation is taken w.r.t.

all randomness of a stochastic gradient implementation, and ?? > 1 is a decay-rate for ??, i.e., ?? Sample i t ??? [1, . . .

, n] and compute:

The above result implies given precision parameter > 0, after T = O 1 2 iterations, Algorithm 2 outputs vectorx ?? 0 such that:

Hence, by tuning parameters ?? and one can approximate a stationary distribution vector ?? [k] .

Algorithm 2 starts with uniform distribution vector x 0 = 1 n 1 and at step t it updates the previous iterative x t by a rule given in line 6.

Let

9 .

Then, for j / ??? N t all entries [x t+1 ] j are equal to each other and updated as:

Given value x T t 1, the above computation takes O(1) time and space.

For entries j ??? N t , all entries [x t+1 ] j might be different ( worst case ) and, therefore, update The transitional probability matrix T in ?? ?? -Rank is sparse; each row and column in T [k] contains N i=1 s i ??? N + 1 non-zero elements (see Section 3.2).

To fully leverage such sparsity, we design a new data structure (see Fig. 5 ) for the storage and computation.

Compared to standard techniques (e.g., COO, CSR, and CRS 10 ) that store (row, column, value) of a sparse vector, our data structure adopts a more efficient protocol that stores (defaults, positions, biases) leading to improvements in computational efficiency, which gives us additional advantages in computational efficiency.

We reload the operations for such data structure including addition, scalar multiplication, dot product, element-wise square root, L1 norm.

We show the example of addition in Fig. 5 .

Our algorithm provides the expected ranking in all three normal-form games shown in Fig. 6 , which is consistent with the results in ??-Rank .

Battle of sexes.

Battle of sexes is an asymmetric game R OM = [ .

As it is a single-population game, we adopt the transitional probability matrix of Eqn.

11 in .

Such game has the inherent structure that Rock/Paper/Scissor is equally likely to be invaded by a mutant, e.g., the scissor population will always be fixated by the rock population, therefore, our method suggests the long-term survival rate for all three strategies are the same (

For all of our experiments, the gradient updates include two phases: warm-up phase and Adam (Kingma & Ba, 2014) phase.

In the warm-up phase, we used standard stochastic gradient descent; after that, we replace SGD with Adam till the convergence.

In practice, we find this yields faster convergence than normal stochastic gradient descent.

As our algorithm does column sampling for the stochastic matrix (i.e. batch size equals to one), adding momentum term intuitively help stabilize the learning.

We also implement infinite ?? , when calculating transition matrix (or its column), where our noise term is set to be 0.01.

For most of our experiments that involve ?? ?? -rank, we set the terminating condition to be, when the gradient norm is less than 10 ???9 .

However, for Random Matrix experiment, we set the terminating gradient norm to be 10 ???2

??? Learning rate to be in between 15 -17

??? Alpha (ranking intensity) to be in between 1 -2.5

??? Number of Population to be between 25 -55 (in integer)

For all of the Adam experiments, after the warmup-step we chooses to decay ?? and ?? by 0.999 for each time steps, where we have ?? to always be 0.1.

Similarly, ?? starts at the value 0.5.

However, in speed and memory experiment, we chooses the decay to be 0.9 Collision Reward is calculated when agent collided with either social car or other agents.

All of our value iteration agents are based on Leurent (2018) environment discretization, which represents the environment in terms of time to collision MDP, taking into account that the other agents are moving in constant speed.

For all experiments, we run value-iteration for 200 steps with the discounting factor of 0.99.

For each controllable cars, the default speed is randomized to be between 10 to 25, while the social cars, the speed are randomized to be between 23 to 25.

We define five types of driving behaviors (one rational + four dangerous) by letting each controlled car have a different ego reward function during training (though the reward we report is the environmental reward which cannot be changed).

By setting this, we can make sure, at upfront, the best joint-strategy strategy should be all cars to drive rationally.

<|TLDR|>

@highlight

We provide a scalable solution to multi-agent evaluation with linear rate complexity in both time and memory in terms of number of agents
Machine learning workloads are often expensive to train, taking weeks to converge.

The current generation of frameworks relies on custom back-ends in order to achieve efficiency, making it impractical to train models on less common hardware where no such back-ends exist.

Knossos builds on recent work that avoids the need for hand-written libraries, instead compiles machine learning models in much the same way one would compile other kinds of software.

In order to make the resulting code efficient, the Knossos complier directly optimises the abstract syntax tree of the program.

However in contrast to traditional compilers that employ hand-written optimisation passes, we take a rewriting approach driven by the $A^\star$ search algorithm and a learn value function that evaluates future potential cost reduction of taking various rewriting actions to the program.

We show that Knossos can automatically learned optimisations that past compliers had to implement by hand.

Furthermore, we demonstrate that Knossos can achieve wall time reduction compared to a hand-tuned compiler on a suite of machine learning programs, including basic linear algebra and convolutional networks.

The Knossos compiler has minimal dependencies and can be used on any architecture that supports a \Cpp toolchain.

Since cost model the proposed algorithm optimises can be tailored to a particular hardware architecture, the proposed approach can potentially applied to a variety of hardware.

While the development of any kind of software can benefit from compliers able to produce fast code, runtime efficiency is particularity important for modern machine learning.

In particular, because modern models they can take weeks to train (OpenAI, 2018) , complier optimisations that lead to execution speed-ups are of huge value.

In parallel, machine learning is being deployed on a variety of diverse devices ranging from wearables to huge clusters clusters of powerful GPUs.

Since each architecture has different performance profile and requires different code optimisations, it is difficult to provide tooling that works fast on all of them.

Traditionally, the tension between performance and interoperability is resolved by machine learning frameworks (Paszke et al., 2017; Abadi et al., 2016) .

In these frameworks, while code execution is outsourced to hardware-specific back-ends such as XLA (XLA authors, 2016) .

While this approach has seen huge initial success, the cost of providing customised back-ends for each target architecture is prohibitive.

Moreover, the frameworks also custom front-ends that require the programmer to specify the model being trained as a compute graph.

Since the compute graph has semantics separate from the host programming language, this process is often error-prone and time-consuming.

In order to address these obstacles, a new generation of tools has recently appeared that transform machine learning code using the same techniques that have been used for compiling traditional software.

The need for a separate front-end API for machine learning operations is eliminated by including automatic differentiation as a first-class feature of the complied language (Innes et al., 2019; Frostig et al., 2018) .

Instead of custom back-ends, modern machine learning compliers use an intermediate representation and perform extensive code optimisations (Innes et al., 2019; Frostig et al., 2018; van Merrienboer et al., 2018; Wei et al., 2018; Sotoudeh et al., 2019; Rotem et al., 2018) .

In addition, program optimisation is being modelled as a machine learning task itself, with the complier learning how to perform rewrites (Chen et al., 2018b; a) .

in mind.

We formalize program optimisation as a finite-horizon Markov Decision Process (MDP), with the reward signal determined by the cost of executing a program.

By solving this MDP, we are able to produce fast code tailor-made for any given task and architecture, without relying on backend-specific hand-written libraries.

Knossos works by re-writing programs written in an intermediate representation (IR).

Akin to JAX (Frostig et al., 2018) and Zygote (Innes et al., 2019) , all Knossos functions are potentially differentiable, avoiding the syntactic awkwardness that arises from embedding a differentiable program in a host language.

The IR can then be transpiled, allowing it to run on any platform that supports a C ++ toolchain.

This allows Knossos code to be seamlessly deployed on specialized or embedded hardware without the need of manual tuning, both for training and for deployment of models, enabling a much broader user base than competing approaches.

To our knowledge, Knossos is the first compiler that combines RL-based program optimisation, firstclass support for deep learning primitives and the ability to target any architecture supporting the C ++ toolchain.

We defer detailed scope comparisons with prior work to Section 4.

We empirically demonstrate the benefits of our program optimisation in Section 5, showing that Knossos was able to automatically learn loop fusion, a type of compiler optimisation that previously had to be applied manually.

We model code optimisation as a finite-horizon Markov Decision Process (MDP).

An MDP is defined (Puterman, 2014; Sutton & Barto, 2018) as a tuple (S, A, T, R, H, p 0 ), where S denotes the state space, A denotes the action space, T denotes the transition dynamics, R denotes the rewards, H is the maximum time budget allowed to solve the problem (the horizon) and p 0 is a fixed probability distribution over initial states.

We provide a detailed description of the states, transitions and rewards later on this section.

States and transitions An MDP state s = (e s , t s ) consists of a Knossos program (or expression) e ??? E and the remaining time budget t s ??? [0, 1, . . . , H] (i.e., the number of remaining steps), where H is the maximum budget.

Any state with t s = 0 is terminating.

The initial state distribution p 0 models the expressions that the RL agent is likely to be asked to optimize.

A sample Knossos expression is shown in Fig. 1a .

The action set A corresponds to different possible ways of rewriting the same expression (see Fig. 1b ).

The transition function T : S ?? A ??? S returns the next state after taking an action.

For example, the first rule in Fig. 1b says that adding zero to any expression can be simplified to the expression itself.

Once the action is chosen, the transition is deterministic.

Because rewrite rules can be applied to different subexpressions, we specify A using generic rewrite rules, which are applied by pattern matching.

There are over 50 rules like this -we provide the details in Appendix C. An essential feature of the rewrites is that they do not change the meaning of the program, i.e. by simplifying from one expression to another we also implicitly generate a proof that the expressions are equivalent.

The RL agent maintains a policy ??(a|s), which defines the probability of taking an action in state s given there are t s steps remaining till the total time budget is exhausted.

A policy ?? generates rollouts ?? ?? .

A rollout ?? ?? is defined as a sequence of states, actions and rewards obtained from the MDP ?? ?? = (s 1 , a 1 , r 1 , s 2 , a 2 , r 2 , . . .

s H , r H ).

Since the policy ?? can be stochastic, it is modelled as a random variable.

The goal of RL agent is to find an optimal policy ?? = arg max ?? J ?? , which attains the best possible return.

The return is defined as

R(s t , s t+1 ) .

Given a policy and the number of timesteps t remaining till the end of episode, we define a value function V (s) = E ?? ts???1 i=0 R(s i , s i+1 ) s 0 = s , where t s denotes the remaining time budget at state s.

The optimal value function V is defined as the value function of an optimal policy ?? .

We assume access to a function c(s), which provides the cost model, i.e. the computational cost of running e s , the expression represented by state s = (e s , t s ), on representative inputs.

While developing a perfect cost models is theoretically impossible due to the intractability of the halting problem (Turing, 1937) , very good cost models exist for the particular subset of programs that compliers are asked to optimise.

The ideal cost model c B would correspond to the run-time of the program on typical inputs, but evaluating costs by benchmarking is very computationally intensive.

In practice, one can often find a surrogate cost function such that for most initial programs s 0 , the state that is reachable from s 0 and minimizes the surrogate cost function c agrees with that for the ideal cost function c B , that is,

which is much easier to acquire.

In other words, the cost function c does not have to produce the same run-time but the same minimum over programs.

We show experimentally in Section 5 that it is indeed possible to reduce the wall clock time of running a program by optimising such a proxy cost model.

Knossos has a modular architecture, making it easy to change the cost function.

This makes it possible to quickly re-tune Knossos programs for any target hardware.

We stress that the formalism allows us to find optimisations even in case getting to the optimized version of the code requires using intermediate programs of higher cost (see Fig. 8 ).

Our reward function is based on this cost model.

The rewards R(s 1 , s 2 ) = c(s 2 ) ??? c(s 1 ) correspond to the attained reduction in cost when rewriting expression e s1 into e s2 .

This formulation ensures that return J ?? equals the total cost reduction attained along the length of the rollout ?? .

Similarly, the value function corresponds to the expected cost reduction under the current policy.

Since our MDP includes a 'no-op' rewrite rule that allows us to keep the current expression and hence the cost, the optimal value function is monotonic in t i.e.

V ((e, t )) ??? V ((e, t)) for any e, t ??? t.

3 TRAINING THE RL AGENT

Hard and Easy Aspects of Rewriting There are two main ways in which the task of rewriting expressions is more challenging than typical RL benchmarks.

First, the allowed set of actions not only changes from state to state, but grows with the size of the expression.

This makes exploration hard.

Second, the states of the MDP, which correspond to the expressions being rewritten, are represented as graphs, whose size and topology varies as optimisation progresses.

This is unlike traditional deep Reinforcement Learning (Mnih et al., 2013) , which learns either from pixels or from data of fixed shape.

While the rewriting task has many features that make it difficult, it is also easier than many traditional RL tasks for three reasons.

First, MDP transitions are completely deterministic.

Second, the task has a large degree of locality in the sense that the performance of a program can often be substantially improved by optimising its parts separately.

Third, we can generate state transitions in any order convenient to us, as opposed to the traditional RL setting, where we are constrained by the order imposed by the environment.

Overall, we have a problem similar to traditional planning, but which requires us to generalise well in order to obtain competitive solutions.

To do this, Knossos uses a custom RL algorithm, based on A search supported by value function learned with a graph neural networks (Algorithm 1).

We describe how to obtain the heuristic in Section 3.2, and the search algorithm in Section 3.1.

Empirical estimate of maximum cost reduction achievable from s end for return C, t, V target end function

We use the A algorithm (Hart et al., 1968) both to train the compiler and to deploy it.

A maintains two priority queues.

One queue (O) stores the frontier, i.e. states from which transitions have not been explored yet.

The other one (C) stores the states visited so far and is used to avoid exploring the same path twice.

The states are explored in the order induced by the A heuristic, which in our case corresponds to the learned value functionV , obtained from previous iterations.

In particular, node priority is set as follows:

Here,V (s) is the estimated future cost reduction obtained from state s within t remaining timesteps.

The quantity c(s 0 ) ??? c(s) corresponds to the cost reduction that has already been achieved by time t, measured against the cost of the initial expression.

Thus, f (s) is an estimate of the maximum possible cost improvement from a trajectory passing through state s at time t.

After the search, we compute the empirical estimate of the maximum cost reduction achievable (V target (s)) for each visited state.

The estimated value of s with t s timesteps is the maximum cost reduction found from s within t s steps.

DISTANCE(s, s ) in Algorithm 2 is the number of steps required to reach s from s. The algorithm stops after the value function was evaluated a set number of times.

In the code this is represented with the function TERM-CONDITION.

A is well-suited for the rewriting task because it exploits its characteristic features.

In particular, it exploits determinism by assuming that a cost reduction achievable once can always be achieved again.

It exploits the availability of reset by considering nodes in the order defined by the heuristic function.

It exploits locality by preferring re-writes that need a small number of rule applications.

Before deciding on A , we also performed experiments with Monte Carlo Tree Search (MCTS).

MCTS does not make use of reset and had worse empirical performance (see Appendix D for details).

States in the Knossos MDP correspond to computation graphs.

In order to apply deep RL to these graphs, we need to be able to construct differentiable embeddings of them.

To do this, we employ Graph Neural Networks based on Gated Recurrent Units (Li et al., 2016) .

During the forward pass, the GNN begins with an initial embedding of the graph nodes.

It then iteratively applies a diffusion process to the graph.

At each step, the obtained representation is fed into a gated recurrent unit (GRU).

The process implicitly encodes the edge structure of the graph in the obtained representation.

We represent a Knossos expression as a graph.

The graph nodes correspond to subexpressions (see Fig. 1a ).

The graph edges are of two kinds.

The first kind of edges connects the nodes with their parents.

In addition, we use another kind of edges, which is used to explicitly provide the information that two subexpressions are identical.

See Table 2b for a list of all edge types.

Edges can be directed or undirected, with the directed edges going in opposite ways considered different.

To compute the value function for an expression e and time budget t, we start from computing the initial node embedding h 0 v ??? R d for all node v ??? N (e), where N (e) is the set of vertices in expression e.

The initial node embedding consists of a one-hot encoding of the node type (constant, variable, etc) followed by zero padding.

This embedding is then fed into the following recurrent computation (see Fig. 2a ):

where p ??? P indexes different edge types, A p (v) is the set of neighbors of node v with respect to the pth edge type.

We choose the message function m p to be a single dense layer for each edge type p and the aggregation operator ??? as the sum of all incoming messages.

We use the GRU cell (Cho et al., 2014) as the recurrent unit f .

The final node embedding h Finally the value of expression e is computed by taking a weighted sum of the final node embedding h T v and passing through a dense layer as follows:

where g :

H are all one-layer dense networks, ?? denotes the sigmoid function, and

We train the above GNN to approximate the optimal value function V .

LetV (s) = V (e s )[t s ] the value function V computed for expression e s and time budget t s .

To track an approximate lower bound of the optimal value function V , we minimize the loss l(

is defined in Algorithm 2 and corresponds to the best cost improvement obtained with the current policy in t s steps.

Normalization by t s is introduced to ease optimisation by ensuring that target values for all outputs of V are in a similar magnitude.

Thus the value function estimat?? V (s) can be obtained from per-step value estimateV (s) asV (s) = t s ??V (s).

For the loss function l we use the Huber loss.

Details about the optimiser used to minimize the loss l are given in Appendix B. In the pseudocode in Algorithm 1, this optimisation is represented with the function FIT.

Knossos builds on a long tradition of compiler technology.

Similarly to traditional compliers (Santos & Peyton-Jones, 1992; Lattner & Adve, 2004) and the more recent deep learning compliers such as Myia (van Merrienboer et al., 2018) , DLVM (Wei et al., 2018) , ISAM (Sotoudeh et al., 2019) and GLOW (Rotem et al., 2018) , Knossos uses an intermediate representation to optimize programs.

However, while these approaches rely on layers of hand-coded optimisation heuristics, Knossos learns the algorithm used to optimize its programs.

In this respect, Knossos is a spiritual successor of benchmark-driven hardware-agnostic optimisation approaches in computational linear algebra (Padua, 2011) and signal processing (Frigo & Johnson, 1998) .

However, unlike these approaches, Knossos is a fully-fledged complier, and can optimize arbitrary programs.

Moreover, thanks to its Reinforcement Learning-driven optimizer, Knossos has an advantage over existing approaches that attempt to learn how to optimize arbitrary code.

For example, Bunel et al. (2017) learns parameters of a code optimizer with a hard-coded hierarchy.

REGAL (Paliwal et al., 2019) only learns the hyper-parameters for a fixed genetic algorithm that preforms the actual optimisation.

The TVM compiler (Chen et al., 2018a ) learns a cost model over programs, but uses simple simulated annealing to perform the optimisation.

Similarly, Chen et al. (2018b) handles only index summation expressions and again relies on simulated annealing.

LIFT (Steuwer et al., 2017) defines an intermediate language suited for expressing numerical computation, but focuses on providing the right set of rewrite rules rather than on the program optimisation process itself.

In Section 5, we demonstrate that the RL optimizer used by Knossos outperforms this approach by a large margin.

Knossos is also related to JAX (Frostig et al., 2018) , which performs just-in-time compilation of Python code using the XLA backend (XLA authors, 2016) .

Knossos differs from JAX in two ways.

First, it uses efficient RL code optimisation, which is architecture-agnostic.

In fact, since Knossos generates C ++ code, it supports a much broader variety of target architectures.

Also, unlike JAX, it makes use of the benefits of a statically typed languages.

In terms of scope, Knossos is also similar to Zygote for Julia (Innes et al., 2019) .

However, unlike these compliers, Knossos makes use of an RL-driven code optimizer.

Since Knossos provides first class support for automatic differentiation, it is also related to established deep learning frameworks (Maclaurin et al., 2015; Abadi et al., 2016; Paszke et al., 2017) .

However, unlike Knossos, these frameworks do not learn how to optimize code, instead relying on manually-prepared back-ends.

Moreover, using them either requires meta-programming, where the user has to use a high-level language to specify the desired computation graph using constructions external to the language (Abadi et al., 2016) , or is constrained to a restricted subset of the language (Paszke et al., 2017) .

In contrast, the Knossos language can be used directly, without manually specifying computation graph constructs or restricting oneself to an allowed subset of the language.

In parallel, the idea of automated rewriting to achieve a given objective was explored in the context of automated theorem provers.

This is conceptually related to our approach since finding an equivalence between formulae is the same as finding a proof that they are equal.

However, recent work in this space has substantial differences in scope.

In particular, state-of-the-art work that searches for refutational proofs in first-order logic (Zombori et al., 2019; Kaliszyk et al., 2018) uses hardcoded features and cannot learn any new ones.

Also, the optimal objective is very different.

While a mathematical proof is only correct when completely reduced to a tautology, we are satisfied with simplifying an expression by a certain margin, not necessarily in the most optimal way possible.

For the Reinforcement Learning part, our algorithm differs from standard techniques in that it has a much larger action space and a state space that consists of graphs, which makes the application of traditional RL algorithms like DQN (Mnih et al., 2013) , A2C (Mnih et al., 2016) and PPO (Schulman et al., 2017) ineffective.

AlphaGo , which also performs a search over a large state space, but differs from Knossos in that it learns for pixel observations and uses an action space of bounded size.

Reinforcement Learning has also been applied to expression rewriting and scheduling problems (Chen & Tian, 2019) .

However, since this approach used actor-critic RL that does not exploit reset, it less well-suited for compilation tasks as described in Section 3.

We evaluated Knossos in three settings.

First, to understand how close and reliably we can achieve the best optimisation, we applied Knossos to a manually curated set of arithmetic expressions, where we know the best available sequence of rewrites.

Second, we applied Knossos to a set of linear algebraic operations, which are representative of typical workloads in numerical computing.

Third, Float (Vec n (Vec l Float))))) (let (beta (get45 var0)) (let (mat_b (get35 var0)) (let (mat_a (get25 var0)) (let (alpha (get15 var0)) (let (mat_c (get55 var0)) (let (mat_x (build n (lam (var4 : Integer) (build l (lam (k : Integer) (sumbuild m (lam (var5 : Integer) (mul (index var4 (index var5 mat_a)) (index var5 (index k mat_b)))))))))) (let (mat_x_6 (build n (lam (var2 : Integer) (build m (lam (var3 : Integer) (mul alpha (index var2 (index var3 mat_x)))))))) (let (mat_y (build n (lam (var6 : Integer) (build m (lam (var1 : Integer) (mul beta (index var6 (index var1 mat_c)))))))) (build n (lam (i : Integer) (build m (lam (j : Integer) (add (index i (index j mat_x_6)) (index i (index j mat_y)))))))))))))))

(def gemm (Vec n (Vec l Float)) ((var0 : (Tuple Float (Vec n (Vec m Float)) (Vec m (Vec l Float)) Float (Vec n (Vec l Float))))) (let (beta (get45 var0)) (let (mat_b (get35 var0)) (let (mat_a (get25 var0)) (let (alpha (get15 var0)) (let (mat_c (get55 var0)) (build n (lam (i : Integer) (build l (lam (j : Integer) (add (mul alpha (sumbuild m (lam (var5 : Integer) (mul (index i (index var5 mat_a)) (index var5 (index j mat_b)))))) (mul beta (index j (index i mat_c)))))))))))))) cost=7070100 10 8 10 10 we compare it to a hand-written rule-based transpiler of the Knossos IL, which we call ksc.

Both Knossos and ksc output C ++ , which is compiled to binary using gcc with optimisation enabled, ensuring a fair comparison.

We describe the results below.

While arithmetic expressions are simple, optimising them is not always a simple task.

Figure 3 shows an example of two similar arithmetic expressions.

Although they look very similar, they require different optimisation strategy to reach to the optimal form.

The left expression gets to optimal by an arithmetic simplification (??x to a denominator and a numerator) but the right expression gets to optimal by a common subexpression elimination.

It is difficult for a rule-based compiler to distinguish the two and optimise such similar expressions using different strategies.

To test Knossos on arithmetic expressions, we used a training set of 36 arithmetic expressions and a test set of 12 different ones.

The details of the experimental setup are given in Appendix B. In this setting, we pick 6 expressions randomly from a training set to train in each epoch.

We ran training for 30 epochs and running 10 repetitions for each experiment with different random seeds.

Search depth was limited to 10 and the termination condition in A was set to 5000 evaluations of the value function.

See Appendix B for the full details including network parameters.

We show the results in Figure 5a .

It can be seen from the figure that Knossos achieved the oracle cost for all expressions.

We also performed an ablation, comparing Knossos to A algorithm (shown as NoGNN) that does not perform the GNN recurrence in equation 4.

As a baseline, we compared to greedy best-first search, which picks a next state to explore greedily without using the value function f (s) := c(s 0 ) ??? c(s).

We also show a comparison to random search and the initial cost of the expression, before any optimisation.

Bootstrap Mode Similarly to a traditional complier, where we are given a concrete program to optimize, the expressions used to evaluate Knossos in this benchmark were the same ones that we used used during training.

Even in this setup, Knossos still generalises, but it does it across sub-expressions of the expressions in the training set.

We tested that on 8 expressions, training for 30 epochs.

Other experimental setup is the same as Arithmetic Expressions.

Figure 5a shows the comparison of the minimum cost achieved by each agent.

It can be seen from the figure that Knossos achieved the best possible cost for all expressions.

Linear Algebra Primitives Numerical linear algebra is fundamental to most calculations in scientific computing and machine learning.

Primitives such as vector multiplication, plane rotation, matrix multiplications and similar primitives often represent the most time-consuming part of the given computation.

To evaluate the performance of Knossos on in this setting, we trained on a set of 11 such linear algebra primitives and evaluated on General Matrix Multiplication (GEMM).

We trained for 5 epochs, each of which included optimisation of cost of 6 primitives.

Search depth was limited to 30 and the termination condition in A was set to 5000 evaluations of the value function.

Figure 6a shows the cost of GEMM.

The plot shows results for 10 independent runs of the Knossos code optimizer on the same input source file.

We used an augmented set of training rules, which included vector operations (see Table 4 in Appendix).

Because of the complexity of the task, we split the search into two phases of 15 steps each.

The training phases differ in the set of allowed rules.

In the first phase, we only allow rules that result in large changes to the cost (Table 4 ).

In the second phase, we allow all rules.

The shaded area represents one standard deviation across the runs of Knossos.

Results show that Knossos produced code of lower cost than the output of the traditional ksc complier according to our cost model.

We also performed a benchmark using wall clock time, shown in Fig. 7a, again showing an improvement.

In addition, we performed a qualitative evaluation of the output in Fig. 4 .

In the program obtained by ksc (middle listing), three temporary variables mat x, mat x 6, and mat y corresponding to the result of A??B, ????mat x, and ?? ??C, respectively, are created.

In the output of Knossos (bottom listing), all the temporary variables are gone.

Hence, Knossos has discovered a form of loop fusion -the type of optimisation that previously had to be built into a complier by a laborious manual process.

Convolutional Network In order to evaluate Knossos on workloads characteristic of modern machine learning pipelines, we also evaluated Knossos on a computer vision task.

We optimize a code for training a convolutional deep network on the MNIST dataset (LeCun, 1998) (Vec l (Vec n Float))) ((var0 : (Tuple (Vec k (Vec l (Vec kn Float))) (Vec l (Vec n Float)) (Vec k (Vec n Float))))) (let ((kernels (get13 var0)) (image (get23 var0)) (d$r (get$3$3 var0))) (sumbuild k (lam (ki : Integer) (let (a_6 (index ki d$r)) (sumbuild n (lam (ni : Integer) (let (a_8 (index ni a_6)) (let (a_7 (build kn (lam (var1 : Integer) a_8))) (sumbuild kn (lam (kni : Integer) (let (a_10 (index kni a_7)) (let (a_11 (build l (lam (sum$i : Integer) a_10))) (sumbuild l (lam (li : Integer) (let (noi (sub (add ni (div kn 2)) kni)) (let (outside_image (or (gt 0 noi) (gte noi n))) (add (if outside_image (tuple (constVec k (constVec l (constVec kn 0.0))) (constVec l (constVec n 0.0))) (tuple (constVec k (constVec l (constVec kn 0.0))) (deltaVec l li (deltaVec n noi (mul (index kni (index li (index ki kernels))) (index li a_11)))))) (tuple (deltaVec k ki (deltaVec l li (deltaVec kn kni (mul (if outside_image 0.0 (index noi (index li image))) (index li a_11))))) (constVec l (constVec n 0.0))))))))))))))))))) cost=102267214109.0 (def rev$conv1d (Tuple (Vec k (Vec l (Vec kn Float))) (Vec l (Vec n Float))) (var0 : (Vec k (Vec l (Vec kn Float))) (Vec l (Vec n Float)) (Vec k (Vec n Float))) (let ((kernels (get13 var0)) (image (get23 var0)) (d$r (get$3$3 var0))) (sumbuild k (lam (ki : Integer) (sumbuild n (lam (ni : Integer) (sumbuild kn (lam (kni : Integer) (sumbuild l (lam (li : Integer) (let (noi (sub (add ni (div kn 2)) kni)) (let (outside_image (or (gt 0 (sub (add ni (div kn 2)) kni)) (gte (sub (add ni (div kn 2)) kni) n))) (add (if (or (gt 0 (sub (add ni (div kn 2)) kni)) (gte (sub (add ni (div kn 2)) kni) n)) (tuple (constVec k (constVec l (constVec kn 0.0))) (constVec l (constVec n 0.0))) (tuple (constVec k (constVec l (constVec kn 0.0))) (deltaVec l li (deltaVec n noi (mul (index kni (index li (index ki kernels))) (index li (build l (lam (sum$i : Integer) (index ni (index ki d$r)))))))))) (tuple (deltaVec k ki (deltaVec l li (deltaVec kn kni (mul (if outside_image 0.0 (index noi (index li image))) (index li (build l (lam (var0 : Integer) (index ni (index ki d$r))))))))) (constVec l (constVec n 0.0)))))))))))))))) cost=163955001999.0 (def rev$conv1d (Tuple (Vec k (Vec l (Vec kn Float))) (Vec l (Vec n Float))) (var0 : (Vec k (Vec l (Vec kn Float))) (Vec l (Vec n Float)) (Vec k (Vec n Float))) (let ((kernels (get13 var0)) (image (get23 var0)) (d$r (get$3$3 var0))) (add (sumbuild k (lam (var6 : Integer) (sumbuild n (lam (var5 : Integer) (sumbuild kn (lam (var7 : Integer) (sumbuild l (lam (var8 : Integer) (if (or (gt 0 (sub (add var5 (div kn 2)) var7)) (gte (sub (add var5 (div kn 2)) var7) n)) (tuple (constVec k (constVec l (constVec kn 0.0))) (constVec l (constVec n 0.0))) (tuple (constVec k (constVec l (constVec kn 0.0))) (deltaVec l var8 (deltaVec n (sub (add var5 (div kn 2)) var7) (mul (index var7 (index var8 (index var6 kernels))) (let (sum$i var8) (index var5 (index var6 d$r)))))))))))))))) (tuple (build k (lam (var4 : Integer) (sumbuild n (lam (var3 : Integer) (build l (lam (var1 : Integer) (build kn (lam (var2 : Integer) (mul (if (or (gt 0 (sub (add var3 (div kn 2)) var2)) (gte (sub (add var3 (div kn 2)) var2) n)) 0.0 (index (sub (add var3 (div kn 2)) var2) (index var1 image))) (let (var0 var1) (index var3 (index var4 d$r)))))))))))) (constVec l (constVec n 0.0))))) represents a typical implementation of a deep learning algorithm and contains primitives such as dense layers, convolutional layers, pooling layers, and so on.

While MNIST is a basic benchmark, we stress that the goal of Knossos was code optimisation as opposed to the computer vision task itself.

We trained on 5 expressions and evaluated on a reverse mode of a convolutional layer.

We fixed the search depth to 40.

The termination condition in A was set to 30000 evaluations of the value function.

We used an augmented set of training rules and split the search into two phases of 20 steps each, allowing rules that result in large changes to the cost in the first phase and all rules in the second phase.

Results are shown in Figure 6b for the cost model and Figure 7b for the wall clock time.

The shaded area represents the standard deviation across the runs of Knossos and the resulting binary.

As above, the Knossos optimizer produced code that outperformed the baseline.

We have demonstrated that Knossos is capable of producing code that is faster than the output of a traditional complier.

Moreover, unlike traditional compliers, Knossos does not rely on hand-crafted optimisation passes that are very laborious to implement.

Instead, traditional optimisation passes are replaced by atomic rewrite rules that can be combined in many ways.

In fact, in our benchmark of linear algebra primitives, Knossos was able to automatically discover loop fusion, an optimisation strategy long known to complier designers.

Knossos code in our experiments can perform both training and inference and can be run on any hardware supporting the C ++ toolchain, including inexpensive embedded devices.

We have introduced Knossos, a new complier targetting machine learning and numerical computation.

Thanks to its automatic code optimisation, Knossos produces binaries that achieve better run-times than a traditional, rule-based complier.

Knossos can deal with complex code generated by automatic differentiation and automatically discover optimisations that previously required careful complier design.

We believe that Knossos will pave the way towards a new generation of future compliers, which will crucially rely on automatically inferring the correct optimisations.

It also has a LISP-like surface syntax, which we used to implement our programs.

In the future, we plan to provide transpilers, allowing for the compilation of code written in other languages into Knossos.

We provide a sample Knossos program in Figure 4 .In order to facilitate Machine Learning workloads, the Knossos IL has native support for automatic differentiation.

We use a new unified view of automatic differentiation as generalised transposition (Elliott, 2018) .

Rather than having an explicit distinction between forward mode and reverse mode AD, Knossos uses uses a type system together with a set of consistent rewrite rules.

Whenever the gradient operator is used as part of a Knossos algorithm, the complier first generates a syntax tree corresponding to the differentiated program and then applies rewrites to optimize the cost of its execution.

This means that the resulting AD algorithm is tailor-made and optimized with that exact use case in mind.

This is in contrast to systems such as PyTorch, which have hard-coded routines for backward-mode AD.

From the perspective of the user, this process is completely transparent in the sense that taking gradients can be applied to any piece of Knossos code.

While the details of this process are beyond the scope of this paper, from the perspective of this work, the important feature of AD is that it corresponds to a transformation of the abstract syntax tree.

The resulting AST can then be optimised in the same way as any other code.

We now describe the parameters used to perform the experiments reported on in the paper.

The parameters used by A in the four tasks described in Sec. 5 are listed in Tab.

1.

The hyper-parameters for the value network training are given in Tab.

2.

In the Graph Neural Network, initial node features are one-hot vectors that represent the node types.

The used node types are: constant, variable, let, if, tuple, select, +, -, *, /, exp, log, ==, >, >=, or, build, apply, lam, sum, sumbuild, constVec, and deltaVec.

Edge types are listed in Tab.

2b.

The auxiliary edge type"is-identical" is inserted to identify identical subexpressions.

It was added so that it is easier to learn re-writes that rely on matching expressions.

The GNN was implemented using a sparse adjacency matrix instead of dense matrix in order to conserve GPU memory in settings where some expressions grow beyond > 10000 nodes during training.

We ran the GNN recursion 10 times.

For optimization we used the Adam optimizer with learning rate 0.0001 and set the dropout rate zero for GNN and 0.2 for the MLP.

We list the basic rule set used in the arithmetic expressions benchmark in Tab.

3.

The additional rewrite rules used in basic linear algebra and convolutional neural network are given in Tab.

4.

In addition to A search, we compare the performance of Monte Carlo Tree Search (Browne et al., 2012) using the UCT formula (Auer et al., 2002; Kocsis & Szepesv??ri, 2006) .

In order to disambiguate across subtly different versions of the algorithm, we describe it below.

Each iteration of MCTS consists of four steps (Algorithm 3).

1. Selection: Starting from the root, a tree policy is recursively descends through the tree until it reaches a leaf node.

2. Expansion: A child node is added to expand the tree.

3. Simulation:

A rollout policy is applied from the new node until the end of the episode.

4. Back-up: The simulation result is backed up through the selected nodes to update their statistic.

The tree policy ?? t and rollout policy ?? r are defined as follows.

?? t (a|s) = arg max a???A X(s ) n(s ) + ?? ln n(s) n(s )+1

(6) ?? r (a|s) = softmax a???A (R(s, a) + V (s ), ??)

Here, n(s) is a visitation count of state s, and ?? is a constant to control the exploration bonus.

X(s )/n(s ) is the average cost reduction achieved by a set of trajectories which passed through n(s )+1 ) is reduced.

This way, the agent is encouraged to try a diverse set of actions.

We evaluated the performance of A search and MCTS for both training and test.

The experimental setup is the same as the Generalisation to Unseen Data experiment in Section 5 except for the used search algorithm.

For MCTS, we used ?? = 5.0 and ?? = 0.5 for both training and test.

Figure 9a shows the results of running all possible combinations of search algorithms when used for training and test in various configurations.

Overall, using A for both training and test achieved the best performance.

In particular, when we fixed the algorithm used during test to A and varied the training algorithm between A and MCTS, A achieved a significantly lower the total minimum cost than MCTS.

Similarly, when we fixed the algorithm used for training to A and compared the performance during testing A achieved significantly lower cost than MCTS again.

Train/Test expression set (div (div 1.0 x) (add 1.0 (div 1.0 x))) (add (div (div 1.0 x) (add (div 1.0 x) 1.0)) (div 1.0 x)) (add (div (div 1.0 x) (add (div 1.0 x) 2.0)) (div 1.0 x)) (mul (div x y) (div x y)) (div (mul (div x y) x) y) (add (div (mul x y) (add 1.0 (mul x y))) (mul x y)) (add (div 1.0 (add 1.0 (mul x y))) (mul x y)) (div (mul x y) (add 1.0 (mul x y))) Figure 10 : List of expressions in training set for Linear Algebra Primitives.

@highlight

We combine A* search with reinforcement learning to speed up machine learning code
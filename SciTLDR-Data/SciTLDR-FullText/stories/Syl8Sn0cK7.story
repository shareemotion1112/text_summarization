We study a general formulation of program synthesis called syntax-guided synthesis(SyGuS) that concerns synthesizing a program that follows a given grammar and satisfies a given logical specification.

Both the logical specification and the grammar have complex structures and can vary from task to task, posing significant challenges for learning across different tasks.

Furthermore, training data is often unavailable for domain specific synthesis tasks.

To address these challenges, we propose a meta-learning framework that learns a transferable policy from only weak supervision.

Our framework consists of three components: 1) an encoder, which embeds both the logical specification and grammar at the same time using a graph neural network; 2) a grammar adaptive policy network which enables learning a transferable policy; and 3) a reinforcement learning algorithm that jointly trains the embedding and adaptive policy.

We evaluate the framework on 214 cryptographic circuit synthesis tasks.

It solves 141 of them in the out-of-box solver setting, significantly outperforming a similar search-based approach but without learning, which solves only 31.

The result is comparable to two state-of-the-art classical synthesis engines, which solve 129 and 153 respectively.

In the meta-solver setting, the framework can efficiently adapt to unseen tasks and achieves speedup ranging from 2x up to 100x.

Program synthesis concerns automatically generating a program that satisfies desired functional requirements.

Promising results have been demonstrated by applying this approach to problems in diverse domains, such as spreadsheet data manipulation for end-users BID21 , intelligent tutoring for students , and code auto-completion for programmers BID19 , among many others.

In a common formulation posed by BID3 called syntax-guided synthesis (SyGuS), the program synthesizer takes as input a logical formula φ and a grammar G, and produces as output a program in G that satisfies φ.

In this formulation, φ constitutes a semantic specification that describes the desired functional requirements, and G is a syntactic specification that constrains the space of possible programs.

The SyGuS formulation has been targeted by a variety of program synthesizers based on discrete techniques such as constraint solving BID36 , enumerative search BID5 , and stochastic search BID37 .

A key limitation of these synthesizers is that they do not bias their search towards likely programs.

This in turn hinders their efficiency and limits the kinds of programs they are able to synthesize.

It is well known that likely programs have predictable patterns BID23 BID1 .

As a result, recent works have leveraged neural networks for program synthesis.

However, they are limited in two aspects.

First, they do not target general SyGuS tasks; more specifically:• They assume a fixed grammar (i.e., syntactic specification G) across tasks.

For example, BID39 learn loop invariants for program verification, but the grammar of loop invariants is fixed across different programs to be verified.

• The functional requirements (i.e., semantic specification φ) are omitted, in applications that concern identifying semantically similar programs BID34 BID0 BID2 , or presumed to be input-output examples BID33 BID7 BID16 BID11 BID13 BID43 BID42 BID35 .In contrast, the SyGuS formulation allows the grammar G to vary across tasks, thereby affording flexibility to enforce different syntactic requirements in each task.

It also allows to specify functional requirements in a manner more general than input-output examples, by allowing the semantic specification φ to be a logical formula (e.g., f (x) = 2x instead of f (1) = 2 ∧ f (3) = 6).

As a result, the general SyGuS setting necessitates the ability to capture common patterns across different specifications and grammars.

A second limitation of existing approaches is that they rely on strong supervision on the generated program BID33 BID7 BID11 .

However, in SyGuS tasks, ground truth programs f are not readily available; instead, a checker is provided that verifies whether f satisfies φ.

In this paper, we propose a framework that is general in that it makes few assumptions on specific grammars or constraints, and has meta-learning capability that can be utilized in solving unseen tasks more efficiently.

The key contributions we make are (1) a joint graph representation of both syntactic and semantic constraints in each task that is learned by a graph neural network model; (2) a grammar adaptive policy network that generalizes across different grammars and guides the search for the desired program; and (3) a reinforcement learning training method that enables learning transferable representation and policy with weak supervision.

We demonstrate our meta-learning framework on a challenging and practical instance of the SyGuS problem that concerns synthesizing cryptographic circuits that are provably free of side-channel attacks BID17 .

In our experiments, we first compare the framework in an out-of-box solver setting against a similar search-based approach and two state-of-the-art classical solvers developed in the formal methods community.

Then we demonstrate its capability as a meta-solver that can efficiently adapt to unseen tasks, and compare it to the out-of-box version.

The Syntax-Guided Synthesis (SyGuS) problem is to synthesize a function f that satisfies two kinds of constraints:• a syntactic constraint specified by a context-free grammar (CFG) G, and • a semantic constraint specified by a formula φ built from symbols in a background theory T along with f .One example of the SyGuS problem is cryptographic circuit synthesis BID17 .

The goal is to synthesize a side-channel free cryptographic circuit by following the given CFG (syntactic constraint) while ensuring that the synthesized circuit is equivalent to the original circuit (semantic constraint).

In this example, the grammar is designed to avoid side-channel attacks, whereas the original circuit is created only for functional correctness and thus is vulnerable to such attacks.

We henceforth use this problem as an illustrative example but note that our proposed method is not limited to this specific SyGuS problem.

We investigate how to efficiently synthesize the function f .

Specifically, given a dataset of N tasks DISPLAYFORM0 , we address the following two tasks:• learning an algorithm A θ : (φ, G) → f parameterized by θ that can find the function f i for (φ i , G i ) ∈ D; • given a new task set D , adapt the above learned algorithm A θ and execute it on new tasks in D .This setting poses two difficulties in learning.

First, the ground truth target function f is not readily available, making it difficult to formulate as a supervised learning problem.

Second, the constraint φ is typically verified using an SAT or SMT solver, and this solver in turns expects the generated f to be complete.

This means the weak supervision signal will only be given after the entire program is generated.

Thus, it is natural to formulate A θ as a reinforcement learning algorithm.

Since each instance (φ i , G i ) ∈ D is an independent task with different syntactic and semantic constraints, the key to success is the design of such meta-learner, which we elaborate in Sec 3.

This section presents our meta-solver model for solving the two problems formulated in Sec 2.

We first introduce formal notation in Sec 3.1.

To enable the transfer of knowledge across tasks with different syntactic and semantic constraints, we propose a representation framework in Sec 3.2 to jointly encode the two kinds of constraints.

The representation needs to be general enough to encode constraints with different specifications.

Lastly, we introduce the Grammar Adaptive Policy Network in Sec 3.3 that executes a program generation policy while automatically adapting to different grammars encoded in each task specification.

We formally define key concepts in the SyGuS problem formulation as follows.semantic spec φ: The spec itself is a program written using some grammar.

In our case, the grammar used in spec φ is different from the grammar G that specifies the syntax of the output program.

However, in many practical cases the tokens (i.e., the dictionary of terminal symbols) may be shared across the input spec and the output program.

DISPLAYFORM0 Here V denotes the nonterminal tokens, while Σ represents the terminal tokens.

s is a special token that denotes the start of the language, and the language is generated according to the production rules defined in R. For a given non-terminal, the associated production rules can be written as α → β 1 |β 2 . . .

|β nα , where n α is the branching factor for non-terminal α ∈ V, and β i = u 1 u 2 . . .

u |βi| ∈ (V Σ) * .

Each production rule α → β i ∈ R represents a way of expanding the grammar tree, by attaching nodes u 1 , u 2 , . . . , u |βi| to node α.

The expansion is repeated until all the leaf nodes are terminals.

Output function f : The output is a program in the language generated by G. A valid output f must satisfy both the syntactic constraints specified by G and the semantic constraints specified by φ.

Different from traditional neural program synthesis tasks, where the program grammar and vocabulary is fixed, each individual task in our setting has its own form of grammar and semantic specification.

Thus in the program generation phase (which we will elucidate in Sec 3.3), one cannot assume a fixed CFG and use a tree decoder like in BID26 and BID11 .

To enable such generalization across different grammars, the information about the CFG for each task needs to be captured in the task representation.

Since the semantic spec program φ and the CFG G have rich structural information, it is natural to use graphs for their representation.

Representing the programs using graphs has been successfully used in many programming language domains.

In our work, we further extend the approach by BID2 with respect to the following aspects:• Instead of only representing the semantic spec program φ as a graph, we propose to jointly represent it along with the grammar G.• To allow information exchange between the two graphs, we leverage the idea of Static Single Assignment (SSA) form in compiler design.

That is, the same variable (token) that may be assigned (defined) at many different places should be viewed differently, but on the other hand, these variations correspond to the same original thing.

Specifically, we introduce global nodes for shared tokens and global links connecting these globally shared nodes and local nodes that (re)define corresponding tokens.

The overall representation framework is described in FIG0 .

To construct the graph, we first build the abstract syntax tree (AST) for the semantic spec program φ, according to its own grammar (typically different from the output grammar G).

To represent the grammar G, we associate each symbol in V Σ with a node representation.

Furthermore, for a non-terminal α and its corresponding production rules α → β 1 |β 2 . . . |β nα , we create additional nodes α i for each substitute β i .

The purpose is to enable grammar adaptive program generation, which we elaborate in Sec 3.3.

As a simplification, we merge all nodes α i representing β i that is a single terminal token into one node.

Finally, the global nodes for shared tokens in Σ are created to link together the shared variable and operator nodes.

This enables information exchange between the syntactic and semantics specifications.

To encode the joint graph G(φ, G), we use graph neural networks to get the vector representation for each node in the graph.

Specifically, for each node v ∈ G, we use the following parameterization for one step of message passing style update: DISPLAYFORM0 Lastly, {h DISPLAYFORM1 are the set of node embeddings.

Here N (v) is the set of neighbor nodes of v, and e u,v denotes the type of edge that links the node u and v. We parameterize F in a way similar to GGNN BID28 , i.e., F (h t , e) = σ(W e t h t ) where we use different matrices W ∈ R d×d for different edge types and different propagation steps t. We sum over all the node embeddings to get the global graph embedding h(G).In addition to the node embeddings and global graph embedding, we also obtain the embedding matrix for each non-terminal node.

Specifically, given node α, we will have the embedding matrix H α ∈ R nα×d , where the ith row of DISPLAYFORM2 α is the embedding of node α i that corresponds to substitution β i .

This enables the grammar adaptive tree expansion in Sec 3.3.

To enable the meta-solver to generalize across different tasks, both the task representation and program generation policy should be shared.

We perform task conditional program generation for this purpose.

Overall the generation is implemented using tree recursive generation, in the depth-first search (DFS) order.

However, to handle different grammars specified in each task, we propose to use the grammar adaptive policy network.

The key idea is to make the policy parameterized by decision embedding, rather than a fixed set of parameters.

This mechanism is inspired by the pointer network BID44 and graph algorithm learning BID14 .Specifically, suppose we are at the decision step t and try to expand the non-terminal node α t .

For different tasks, the non-terminals may not be the same; furthermore, the number of ways to expand a certain non-terminal can also be different.

As a result, we cannot simply have a parameterized layer W h αt to calculate the logits of multinomial distribution.

Rather, we use the embedding matrix H αt ∈ R nα t ×d to perform decision for this time step.

This embedding matrix is obtained as described in Sec 3.2.

DISPLAYFORM0

• Global graph embedding:• Embedding of production rules DISPLAYFORM0 ......

• Global graph embedding:• Embedding of production rules d1 -> d2 AND d2

Sampled production rule: DISPLAYFORM0 Figure 2: Generating solution using the grammar adaptive policy network.

This figure shows one step of policy roll-out, which demonstrates how the same policy network handles different tasks with different grammar G 1 and G 2 .Now we are able to build our policy network in an auto-regressive way.

Specifically, the policy π(f |φ, G) can be parameterized as: DISPLAYFORM1 Here the probability of each action (in other words, each tree expansion decision) is defined as π(a t |h(G), DISPLAYFORM2 , where s t ∈ R d is the context vector that captures the state of h(G) and T (t−1) .

In our implementation, s t is tracked by a LSTM decoder whose hidden state is updated by the embedding of the chosen action h αt .

The initial state s 0 is obtained by passing graph embedding h(G) through a dense layer with matching size.

In this section, we present a reinforcement learning framework for the meta-solver.

Formally, let θ denote the parameters of graph embedding and adaptive policy network.

For a given pair of instances (φ, G), we learn a policy π θ (f |φ, G) parameterized by θ that generates f such that φ ≡ f .Reward design: The RL episode starts by accepting the representation of tuple φ, G as initial observation.

During the episode, the model executes a sequence of actions to expand non-terminals in f , and finishes the episode when f is complete.

Upon finishing, the SAT solver is invoked and will return a binary flag indicating whether f satisfies φ or not.

An obvious reward design would be directly using the binary value as the episode return.

However, this leads to a high variance in returns as well as a highly non-smooth loss surface.

Here, we propose to smooth the reward as follows: for each specification φ we maintain a test case buffer B φ that stores all input examples observed so far.

Each time the SAT solver is invoked for φ, if f passes then a full reward of 1 is given, otherwise the solver will generate a counter-example b besides the binary flag.

We then sample interpolated examples around b which we denote the set asB b .

Then the reward is given as the fractions of examples in B φ andB b where f has the equivalent output as φ r = DISPLAYFORM0 At the end of the episode, the buffer is updated as B φ ← B φ ∪B b for next time usage.

In the extreme case where all inputs can be enumerated, e.g. binary or discrete values, it reduces to computing the fraction of passed examples over the entire test case set.

This is implemented in our experiment on the cryptographic circuit synthesis task.

In the meta-learning setting, the framework learns to represent a set of different programs and navigate the generation process under different constraints.

We utilize the Advantage Actor-Critic (A2C) for model training.

Given a training set D, a minibatch of instances are sampled from D for each epoch.

For each instance φ i , G i , the model performs a complete rollout using policy π θ (f |φ i , G i ).The actor-critic method computes the gradients w.r.t to θ of each instance as DISPLAYFORM0 where γ denotes the discounting factor and V (s t ; ω) is a state value estimator parameterized by ω.

In our implementation, this is modeled as a standard MLP with scalar output.

It is learned to fit the expected return, i.e., min ω E |f | t=1 γ t r − V (s t ; ω) .

Gradients obtained from each instance are averaged over the minibatch before applying to the parameter.

Figure 3 : An example of a circuit synthesis task from the 2017 SyGuS competition.

Given the original program specification which is represented as an abstract syntax tree (left), the solver is tasked to synthesize a new circuit f (right).

The synthesis process is specified by the syntactic constraint G (top), and the semantic constraint (bottom) specifies that f must have functionality equivalent to the original program.

We evaluate the our framework 1 on cryptographic circuit synthesis tasks BID17 which constitute a challenging benchmark suite from the general track of the SyGuS Competition (2017).

The dataset contains 214 tasks, each of which is a pair of logical specification, describing the correct functionality, and a context free grammar, describing the timing constraints for input signals.

The goal is to find an equivalent logical expression which is required to follow the given context free grammar in order to avoid potential timing channel vulnerabilities.

Figure 3 shows an illustrative example.

Each synthesis task has a different logical specification as well as timing constraints, and both the logical specification and context free grammar varies from task to task, posing a significant challenge in representation learning.

As a result, this suite of tasks serves as an ideal testbed for our learning framework and its capability to generalize to unseen specifications and grammars.

The experiments are conducted in two learning settings.

First, we test our framework as an outof-box solver, which means the training set D and testing set D are the same and contain only one instance.

In other words, the framework is tasked to solve only one instance at a time.

This test-on-train setting serves to investigate the capacity of our framework in representation and policy learning, as the model can arbitrarily "exploit" the problem without worrying about overfitting.

This setting also enables us to compare our framework to classical solvers developed in the formal methods community.

As those solvers do not utilize learning-based strategies, it is sensible to also limit our framework not to carry over prior knowledge from a separate training set.

Second, we evaluate the model as a meta-solver which is trained over a training set D, and finetuned on each of the new tasks in a separate set D .

In this setting, we aim to demonstrate that our Table 1 : Number of instances solved using: 1) EUSolver, 2) CVC4, 3) ESymbolic, and 4) Out-ofBox Solver.

For each solver, the maximum time in solving an instance and the average and median time over all solved instances are also shown below.

framework is capable of learning a transferable representation and policy in order to efficiently adapt to unseen tasks.

In the out-of-box solver setting, we compare our framework against solvers built based on two classical approaches: a SAT/SMT constraint solving based approach and a search based approach.

For the former, we choose CVC4 BID36 , which is the state-of-the-art SMT constraint solver; for the latter, we choose EUSolver BID5 , which is the winner of the SyGuS 2017 Competition BID4 .

Furthermore, we build a search based solver as baseline, ESymbolic, which systematically expands non-terminals in a predefined order (e.g. depth-first-search) and effectively prunes away partially generated candidates by reducing it to 2QBF BID6 satisfiability check.

ESymbolic can be viewed as a generalization of EUSolver by replacing the carefully designed domain-specific heuristics (e.g. indistinguishability and unification) with 2QBF.In order to make the comparison fair, we run all solvers on the same platform with a single core CPU available 2 , even though our framework could take advantage of hardware accelerations, for instance, via GPUs and TPUs.

We measure the performance of each solver by counting the number of instances it can solve given a 6 hours limit spent on each task.

It is worth noting that comparing running time only gives a limited view of the solvers' performance.

Although the hardware is the same, the software implementation can make many differences.

For instance, CVC4 is carefully redesigned and re-implemented in C++ as the successor of CVC3 (Barrett et al., 2003) , which has been actively improved for more than a decade.

To our best knowledge, the design and implementation of EUSolver is directly guided by and heavily tuned according to SyGuS benchmarks.

In contrast, our framework is a proof-of-concept prototype implemented in Python and has not yet been tuned for running time performance.

In Table 1 , we summarize the total number of instances solved by each solver as well as the maximum, average and median running time spent on solved instances.

In terms of the absolute number of solved instances, our framework is not yet as good as EUSolver, which is equipped with specialized heuristics.

However, EUSolver fails to solve 4 instances that are only solved by our framework.

All instances solved by CVC4 and ESymbolic are a strict subset of instances solved by EUSolver.

Thus, besides being a new promising approach, our framework already plays a supplementary role for improving the current state-of-the-art.

Compared with the state-of-the-art CVC4 solver, our framework has smaller maximum time but higher average and median time usage.

This suggests that our framework excels at solving difficult instances with better efficiency.

This observation is further confirmed in FIG3 , where we plot the time usage along with the number of instances solved.

This suggests that canonical solvers such as CVC4 are efficient in solving simple instances, but have inferior scalability compared to our dynamically adapted approach when the problem becomes more difficult, where we can see a steeper increase in time usages by CVC4 in solving 110 and more instances.

Though EUSolver has superior scalability, it is achieved by a number of heuristics that are manually designed and iteratively improved by experts with the same benchmark on hand.

In contrast, our framework learns a policy to solve hard instances from scratch on the fly without requiring training data at all.

We next evaluate whether our framework is capable of learning transferable knowledge across different synthesis tasks.

We randomly split the 214 circuits synthesis tasks into two sets: 150 tasks for training and the rest 64 tasks for testing.

The meta-solver is then trained on the training set for 35000 epochs using methods introduced in Sec 4.1.

For each epoch, a batch of 10 tasks are sampled.

The gradients of each task are averaged and applied to the model parameters using Adam optimizer.

In testing phase, the trained meta-solver is finetuned on each task in the testing set until either a correct program is synthesized or timeout occurs.

This process is similar to the setting in Sec 5.1 but with smaller learning rate and exploration.

We compare the trained meta-solver with the out-of-box solver in solving tasks in the test set.

Out of 64 testing tasks, the out-of-box solver and meta-solver can solve 36 and 37 tasks, respectively.

Besides the additional task solved, the performance is also greatly improved by meta-solver, which is shown in FIG4 .

Table 5 (a) shows the accumulated number of candidates generated to successfully solve various ratios of testing tasks.

We see that the number of explored candidates by meta-solver is significantly reduced: for 40% of testing tasks (i.e., 66% of solved tasks), meta-learning enable 4x reduction on average.

The accumulated reduction for all solved tasks (60% of testing tasks) is not that significant.

This is because meta-learning improve dramatically for most (relatively) easy tasks but helps slightly for a few hard tasks, which actually dominate the number of generated candidates.

FIG4 (b) shows the speedup distribution over the 36 commonly solved tasks.

Meta-solver achieves at least 2x speedup for most benchmarks, orders of magnitude improvement for 10 out of 36 unseen tasks, and solves one task that is not solvable without meta-learning.

We survey work on symbolic program synthesis, neural program synthesis, and neural induction.

Symbolic program synthesis.

Automatically synthesizing a program from its specification was first posed by BID30 .

It received renewed attention with advances in SAT and SMT solvers BID41 and found application in problems in various domains as surveyed by BID22 .

In this context, SyGuS BID3 was proposed as a common format to express these problems.

Several implementations of SyGuS solvers exist, including by constraint solving BID36 , divide-and-conquer BID5 , and stochastic MCMC search BID37 , in addition to various domain-specific algorithms.

A number of probabilistic techniques have been proposed to accelerate these solvers by modeling syntactic aspects of programs.

These include PHOG BID27 , log-bilinear tree-traversal models BID29 , and graph-based statistical models .Neural program synthesis.

Several recent works have used neural networks to accelerate the discovery of desired programs.

These include DeepCoder BID7 , Bayou BID31 , RobustFill BID16 , Differentiable FORTH BID10 , neurosymbolic program synthesis BID33 BID11 , neural-guided deductive search BID43 , learning context-free parsers BID13 , and learning program invariants BID39 .

The syntactic specification in these approaches is fixed by defining a domain-specific language upfront.

Also, with the exception of BID39 , the semantic specification takes the form of input-output examples.

Broadly, these works have difficulty with symbolic constraints, and are primarily concerned with avoiding overfitting, coping with few examples, and tolerating noisy examples.

Our work relaxes both these kinds of specifications to target the general SyGuS formulation.

Recently BID18 propose gradually bootstrapping domain-specific languages for neurally-guided Bayesian program learning, while our work concerns learning programs that use similar grammars, which may or may not be incremental.

Neural program induction.

Another body of work includes techniques in which the neural network is itself the computational substrate.

These include neural Turing machines BID20 ) that can learn simple copying/sorting programs, the neural RAM model BID25 to learn pointer manipulation and dereferencing, the neural GPU model BID24 to learn complex operations like binary multiplication, and BID12 's work to incorporate recursion.

These approaches have fundamental problems regarding verifying and interpreting the output of neural networks.

In contrast, we propose tightly integrating a neural learner with a symbolic verifier so that we obtain the scalability and flexibility of neural learning and the correctness guarantees of symbolic verifiers.

We proposed a framework to learn a transferable representation and strategy in solving a general formulation of program synthesis, i.e. syntax-guided synthesis (SyGuS).

Compared to previous work on neural synthesis, our framework is capable of handling tasks where 1) the grammar and semantic specification varies from task to task, and 2) the supervision is weak.

Specifically, we introduced a graph neural network that can learn a joint representation over different pairs of syntactic and semantic specifications; we implemented a grammar adaptive network that enables program generation to be conditioned on the specific task; and finally, we proposed a meta-learning method based on the Advantage Actor-Critic (A2C) framework.

We compared our framework empirically against one baseline following a similar search fashion and two classical synthesis engines.

Under the outof-box solver setting with limited computational resources and without any prior knowledge from training, our framework is able to solve 141 of 214 tasks, significantly outperforming the baseline ESymbolic by 110.

In terms of the absolute number of solved tasks, the performance is comparable to two state-of-the-art solvers, CVC4 and EUSolver, which solve 129 and 153 respectively.

However, the two state-of-the-art solvers failed on 4 tasks solved by our framework.

When trained as a meta-solver, our framework is capable of accelerating the solving process by 2× to 100×.

@highlight

We propose a meta-learning framework that learns a transferable policy from only weak supervision to solve synthesis tasks with different logical specifications and grammars.
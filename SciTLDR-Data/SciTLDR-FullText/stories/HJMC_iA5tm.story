We present NeuroSAT, a message passing neural network that learns to solve SAT problems after only being trained as a classifier to predict satisfiability.

Although it is not competitive with state-of-the-art SAT solvers, NeuroSAT can solve problems that are substantially larger and more difficult than it ever saw during training by simply running for more iterations.

Moreover, NeuroSAT generalizes to novel distributions; after training only on random SAT problems, at test time it can solve SAT problems encoding graph coloring, clique detection, dominating set, and vertex cover problems, all on a range of distributions over small random graphs.

The propositional satisfiability problem (SAT) is one of the most fundamental problems of computer science.

BID4 showed that the problem is NP-complete, which means that searching for any kind of efficiently-checkable certificate in any context can be reduced to finding a satisfying assignment of a propositional formula.

In practice, search problems arising from a wide range of domains such as hardware and software verification, test pattern generation, planning, scheduling, and combinatorics are all routinely solved by constructing an appropriate SAT problem and then calling a SAT solver BID9 .

Modern SAT solvers based on backtracking search are extremely well-engineered and have been able to solve problems of practical interest with millions of variables BID2 ).We consider the question: can a neural network learn to solve SAT problems?

To answer, we develop a novel message passing neural network (MPNN) BID21 BID16 BID8 , NeuroSAT, and train it as a classifier to predict satisfiability on a dataset of random SAT problems.

We provide NeuroSAT with only a single bit of supervision for each SAT problem that indicates whether or not the problem is satisfiable.

When making a prediction about a new SAT problem, we find that NeuroSAT guesses unsatisfiable with low confidence until it finds a solution, at which point it converges and guesses satisfiable with very high confidence.

The solution itself can almost always be automatically decoded from the network's activations, making NeuroSAT an end-to-end SAT solver.

See Figure 1 for an illustration of the train and test regimes.

Although it is not competitive with state-of-the-art SAT solvers, NeuroSAT can solve SAT problems that are substantially larger and more difficult than it ever saw during training by simply performing more iterations of message passing.

Despite only running for a few dozen iterations during training, at test time NeuroSAT continues to find solutions to harder problems after hundreds and even thousands of iterations.

The learning process has yielded not a traditional classifier but rather a procedure that can be run indefinitely to search for solutions to problems of varying difficulty.

SAT problem P Output: 1 {P is satisfiable} Figure 1: We train NeuroSAT to predict whether SAT problems are satisfiable, providing only a single bit of supervision for each problem.

At test time, when NeuroSAT predicts satisfiable, we can almost always extract a satisfying assignment from the network's activations.

The problems at test time can also be substantially larger, more difficult, and even from entirely different domains than the problems seen during training.

Moreover, NeuroSAT generalizes to entirely new domains.

Since NeuroSAT operates on SAT problems and since SAT is NP-complete, NeuroSAT can be queried on SAT problems encoding any kind of search problem for which certificates can be checked in polynomial time.

Although we train it using only problems from a single random problem generator, at test time it can solve SAT problems encoding graph coloring, clique detection, dominating set, and vertex cover problems, all on a range of distributions over small random graphs.

The same neural network architecture can also be used to help construct proofs for unsatisfiable problems.

When we train it on a different dataset in which every unsatisfiable problem contains a small contradiction (call this trained model NeuroUNSAT), it learns to detect these contradictions instead of searching for satisfying assignments.

Just as we can extract solutions from NeuroSAT's activations, we can extract the variables involved in the contradiction from NeuroUNSAT's activations.

When the number of variables involved in the contradiction is small relative to the total number of variables, knowing which variables are involved in the contradiction can enable constructing a resolution proof more efficiently.

Background.

A formula of propositional logic is a boolean expression built using the constants true (1) and false (0), variables, negations, conjunctions, and disjunctions.

A formula is satisfiable provided there exists an assignment of boolean values to its variables such that the formula evaluates to 1.

For example, the formula ( DISPLAYFORM0 is satisfiable because it will evaluate to 1 under every assignment that does not map x 1 , x 2 and x 3 to the same value.

For every formula, there exists an equisatisfiable formula in conjunctive normal form (CNF), expressed as a conjunction of disjunctions of (possibly negated) variables.

1 Each conjunct of a formula in CNF is called a clause, and each (possibly negated) variable within a clause is called a literal.

The formula above is equivalent to the CNF formula (x 1 ∨ x 2 ∨ x 3 ) ∧ (¬x 1 ∨ ¬x 2 ∨ ¬x 3 ), which we can represent more concisely as {1|2|3, 1|2|3}. A formula in CNF has a satisfying assignment if and only if it has an assignment such that every clause has at least one literal mapped to 1.

A SAT problem is a formula in CNF, where the goal is to determine if the formula is satisfiable, and if so, to produce a satisfying assignment of truth values to variables.

We use n to denote the number of of variables in a SAT problem, and m to denote the number of clauses.

Classification task.

For a SAT problem P , we define φ(P ) to be true if and only if P is satisfiable.

Our first goal is to learn a classifier that approximates φ.

Given a distribution Ψ over SAT problems, we can construct datasets D train and D test with examples of the form (P, φ(P )) by sampling problems P ∼ Ψ and computing φ(P ) using an existing SAT solver.

At test time, we get only the problem P and the goal is to predict φ(P ), i.e. to determine if P is satisfiable.

Ultimately we care about the solving task, which also includes finding solutions to satisfiable problems.

A SAT problem has a simple syntactic structure and therefore could be encoded into a vector space using standard methods such as an RNN.

However, the semantics of propositional logic induce rich invariances that such a syntactic method would ignore, such as permutation invariance and negation invariance.

Specifically, the satisfiability of a formula is not affected by permuting the variables (e.g. swapping x 1 and x 2 throughout the formula), by permuting the clauses (e.g. swapping the first clause with the second clause), or by permuting the literals within a clause (e.g. replacing the clause 1|2 with 2|1.

The satisfiability of a formula is also not affected by negating every literal corresponding to a given variable (e.g. negating all occurrences of x 1 in {1|2, 1|3} to yield {1|2, 1|3}).We now describe our neural network architecture, NeuroSAT, that enforces both permutation invariance and negation invariance.

We encode a SAT problem as an undirected graph with one node for every literal, one node for every clause, an edge between every literal and every clause it appears in, and a different type of edge between each pair of complementary literals (e.g. between x i and x i ).

NeuroSAT iteratively refines a vector space embedding for each node by passing "messages" back and forth along the edges of the graph as described in BID8 .

At every time step, we have an embedding for every literal and every clause.

An iteration consists of two stages.

First, each clause receives messages from its neighboring literals and updates its embedding accordingly.

Next, each literal receives messages from its neighboring clauses as well as from its complement, then updates its embedding accordingly.

FIG1 provides a high-level illustration of the architecture.

More formally, our model is parameterized by two vectors (L init , C init ), three multilayer perceptrons (L msg , C msg , L vote ) and two layer-norm LSTMs BID0 BID12 (L u , C u ).

At every time step t, we have a matrix L (t) ∈ R 2n×d whose ith row contains the embedding for the literal i and a matrix C (t) ∈ R m×d whose jth row contains the embedding for the clause c j , which we initialize by tiling L init and C init respectively.

We also have hidden states DISPLAYFORM0 DISPLAYFORM1 h ∈ R m×d for L u and C u respectively, both initialized to zero matrices.

Let M be the (bipartite) adjacency matrix defined by M (i, j) = 1 { i ∈ c j } and let Flip be the operator that takes a matrix L and swaps each row of L with the row corresponding to the literal's negation.

A single iteration consists of applying the following two updates: DISPLAYFORM2 , which contains a single scalar for each literal (the literal's vote), and then we compute the average of the literal votes DISPLAYFORM3 We train the network to minimize the sigmoid cross-entropy loss between the logit y (T ) and the true label φ(P ).Our architecture enforces permutation invariance by operating on nodes and edges according to the topology of the graph without any additional ordering over nodes or edges.

Likewise, it enforces negation invariance by treating all literals the same no matter whether they originated as a positive or negative occurrence of a variable.

as NeuroSAT runs on a satisfiable problem from SR(20).

For clarity, we reshape each L (t) * to be an R n×2 matrix so that each literal is paired with its complement; specifically, the ith row contains the scalar votes for x i and x i .

Here white represents zero, blue negative and red positive.

For several iterations, almost every literal is voting unsat with low confidence (light blue).

Then a few scattered literals start voting sat for the next few iterations, but not enough to affect the mean vote.

Suddenly there is a phase transition and all the literals (and hence the network as a whole) start to vote sat with very high confidence (dark red).

After the phase transition, the vote for each literal converges and the network stops evolving.

We stress that none of the learned parameters depend on the size of the SAT problem and that a single model can be trained and tested on problems of arbitrary and varying sizes.

At both train and test time, the input to the model is simply any bipartite adjacency matrix M over any number of literals and clauses.

The learned parameters only determine how each individual literal and clause behaves in terms of its neighbors in the graph.

Variation in problem size is handled by the aggregation operators: we sum the outgoing messages of each of a node's neighbors to form the incoming message, and we take the mean of the literal votes at the end of message passing to form the logit y (T ) .

We want our neural network to be able to classify (and ultimately solve) SAT problems from a variety of domains that it never trained on.

One can easily construct distributions over SAT problems for which it would be possible to predict satisfiability with perfect accuracy based only on crude statistics; however, a neural network trained on such a distribution would be unlikely to generalize to problems from other domains.

To force our network to learn something substantive, we create a distribution SR(n) over pairs of random SAT problems on n variables with the following property: one element of the pair is satisfiable, the other is unsatisfiable, and the two differ by negating only a single literal occurrence in a single clause.

To generate a random clause on n variables, SR(n) first samples a small integer k (with mean 5) 2 then samples k variables uniformly at random without replacement, and finally negates each one with independent probability 50%.

It continues to generate clauses c i in this fashion, adding them to the SAT problem, and then querying a traditional SAT solver (we used Minisat BID23 ), until adding the clause c m finally makes the problem unsatisfiable.

Since {c 1 , . . .

, c m−1 } had a satisfying assignment, negating a single literal in c m must yield a satisfiable problem {c 1 , . . .

, c m−1 , c m }.

The pair ({c 1 , . . . , c m−1 , c m }, {c 1 , . . . , c m−1 , c m }) are a sample from SR(n).

Although our ultimate goal is to solve SAT problems arising from a variety of domains, we begin by training NeuroSAT as a classifier to predict satisfiability on SR(40).

Problems in SR(40) are small enough to be solved efficiently by modern SAT solvers-a fact we rely on to generate the problems-but the classification problem is highly non-trivial from a machine learning perspective.

Each problem has 40 variables and over 200 clauses on average, and the positive and negative examples differ by negating only a single literal occurrence out of a thousand.

We were unable to train an LSTM on a many-hot encoding of clauses (specialized to problems with 40 variables) to predict with >50% accuracy on its training set.

Even the canonical SAT solver MiniSAT BID23 needs to backjump 3 almost ten times on average, and needs to perform over a hundred primitive logical inferences (i.e. unit propagations) to solve each problem.

We instantiated the NeuroSAT architecture described in §3 with d = 128 dimensions for the literal embeddings, the clause embeddings, and all the hidden units; 3 hidden layers and a linear output layer for each of the MLPs L msg , C msg , and L vote ; and rectified linear units for all non-linearities.

We regularized by the 2 norm of the parameters scaled by 10 −10 , and performed T = 26 iterations of message passing on every problem.

We trained our model using the ADAM optimizer BID13 ) with a learning rate of 2 × 10 −5 , clipping the gradients by global norm with clipping ratio 0.65 BID20 .

We batched multiple problems together, with each batch containing up to 12,000 nodes (i.e. literals plus clauses).

To accelerate the learning, we sampled the number of variables n uniformly from between 10 and 40 during training (i.e. we trained on SR FIG4 )), though we only evaluate on SR(40).

We trained on millions of problems.

After training, NeuroSAT is able to classify the test set correctly with 85% accuracy.

In the next section, we examine how NeuroSAT manages to do so and show how we can decode solutions to satisfiable problems from its activations.

Note: for the entire rest of the paper, NeuroSAT refers to the specific trained model that has only been trained on SR(U(10, 40)).

Let us try to understand what NeuroSAT (trained on SR(U(10, 40))) is computing as it runs on new problems at test time.

For a given run, we can compute and visualize the 2n-dimensional vector of literal votes DISPLAYFORM0 ) at every iteration t. Figure 3 illustrates the sequence of literal votes DISPLAYFORM1 as NeuroSAT runs on a satisfiable problem from SR(20).

For clarity, we reshape each L (t) * to be an R n×2 matrix so that each literal is paired with its complement; specifically, the ith row contains the scalar votes for x i and x i .

Here white represents zero, blue negative and red positive.

For several iterations, almost every literal is voting unsat with low confidence (light blue).

Then a few scattered literals start voting sat for the next few iterations, but not enough to affect the mean vote.

Suddenly, there is a phase transition and all the literals (and hence the network as a whole) start to vote sat with very high confidence (dark red).

After the phase transition, the vote for each literal converges and the network stops evolving.

NeuroSAT seems to exhibit qualitatively similar behavior on every satisfiable problem that it predicts correctly.

The problems for which NeuroSAT guesses unsat are similar except without the phase change: it continues to guess unsat with low-confidence for as many iterations as NeuroSAT runs for.

NeuroSAT never becomes highly confident that a problem is unsat, and it almost never guesses sat on an unsat problem.

These results suggest that NeuroSAT searches for a certificate of satisfiability, and that it only guesses sat once it has found one.

Let us look more carefully at the literal votes L (24) * from Figure 3 after convergence.

Note that most of the variables have one literal vote distinctly darker than the other.

Moreover, the dark votes are all approximately equal to each other, and the light votes are all approximately equal to each other as well.

Thus the votes seem to encode one bit for each variable.

It turns out that these bits encode a satisfying assignment in this case, but they do not do so reliably in general.

Recall from §3 that NeuroSAT projects the higher dimensional literal embeddings L (T ) ∈ R 2n×d to the literal votes L (T ) * using the MLP L vote .

FIG4 illustrates the two-dimensional PCA embeddings for Lto L (26) (skipping every other time step) as NeuroSAT runs on a satisfiable problem from SR(40).

Blue and red dots indicate literals that are set to 0 and 1 in the satisfying assignment that it eventually finds, respectively.

The blue and red dots cannot be linearly separated until the phase transition at the end, at which point they form two distinct clusters according to the satisfying assignment.

We FIG4 ).

It almost never guesses sat on unsatisfiable problems.

On satisfiable problems, it correctly guesses sat 73% of the time, and we can decode a satisfying assignment for 70% of the satisfiable problems by clustering the literal embeddings L (T ) as described in §6.observe a similar clustering almost every time the network guesses sat.

Thus the literal votes L (T ) * only ever encode the satisfying assignment by chance, when the projection L vote happens to preserve this clustering.

Our analysis suggests a more reliable way to decode solutions from NeuroSAT's internal activations: 2-cluster L (T ) to get cluster centers ∆ 1 and ∆ 2 , partition the variables according to the predicate DISPLAYFORM2 , and then try both candidate assignments that result from mapping the partitions to truth values.

This decoding procedure (using k-means to find the two cluster centers) successfully decodes a satisfying assignment for over 70% of the satisfiable problems in the SR(40) test set.

TAB0 summarizes the results when training on SR(U(10, 40)) and testing on SR(40). (12) to L (26) (skipping every other time step) as NeuroSAT runs on a satisfiable problem from SR(40).

Blue and red dots indicate literals that are set to 0 and 1 in the satisfying assignment that it eventually finds, respectively.

We see that the blue and red dots are mixed up and cannot be linearly separated until the phase transition at the end, at which point they form two distinct clusters according to the satisfying assignment.

Recall that at training time, NeuroSAT is only given a single bit of supervision for each SAT problem.

Moreover, the positive and negative examples in the dataset differ only by the placement of a single edge.

NeuroSAT has learned to search for satisfying assignments solely to explain that single bit of supervision.

Even though we only train NeuroSAT on SR(U(10, 40)), it is able to solve SAT problems sampled from SR(n) for n much larger than 40 by simply running for more iterations of message passing.

Figure 5 shows NeuroSAT's success rate on SR(n) for a range of n as a function of the number of iterations T .

For n = 200, there are 2 160 times more possible assignments to the variables than any problem it saw during training, and yet it can solve 25% of the satisfiable problems in SR(200) by running for four times more iterations than it performed during training.

On the other hand, when restricted to the number of iterations it was trained with, it solves under 10% of them.

Thus we see that its ability to solve bigger and harder problems depends on the fact that the dynamical system it has learned encodes generic procedural knowledge that can operate effectively over a wide range of time frames.

Figure 5: NeuroSAT's success rate on SR(n) for a range of n as a function of the number of iterations T .

Even though we only train NeuroSAT on SR(40) and below, it is able to solve SAT problems sampled from SR(n) for n much larger than 40 by simply running for more iterations.

Figure 6 : Example graph from the Forest-Fire distribution.

The graph has a coloring for k ≥ 5, a clique for k ≤ 3, a dominating set for k ≥ 3, and a vertex cover for k ≥ 6.

However, these properties are not perceptually obvious and require deliberate computation to determine.

Every problem in NP can be reduced to SAT in polynomial time, and SAT problems arising from different domains may have radically different structural and statistical properties.

Even though NeuroSAT has learned to search for satisfying assignments on problems from SR(n), we may still find that the dynamical system it has learned only works properly on problems similar to those it was trained on.

To assess NeuroSAT's ability to extrapolate to different classes of problems, we generated problems in several other domains and then encoded them all into SAT problems (using standard encodings).

In particular, we started by generating one hundred graphs from each of six different random graph distributions (Barabasi, Erdös-Renyi, Forest-Fire, Random-k-Regular, Random-Static-Power-Law, and Random-Geometric).

4 We found parameters for the random graph generators such that each graph has ten nodes and seventeen edges on average.

For each graph in each collection, we generated graph coloring problems (3 ≤ k ≤ 5), dominating-set problems (2 ≤ k ≤ 4)), clique-detection problems (3 ≤ k ≤ 5), and vertex cover problems (4 ≤ k ≤ 6).5 We chose the range of k for each problem to include the threshold for most of the graphs while avoiding trivial problems such as 2-clique.

As before, we used Minisat BID23 to determine satisfiability.

Figure 6 shows an example graph from the distribution.

Note that the trained network does not know anything a priori about these tasks; the generated SAT problems need to encode not only the graphs themselves but also formal descriptions of the tasks to be solved.

Out of the 7,200 generated problems, we kept only the 4,888 satisfiable problems.

On average these problems contained over two and a half times as many clauses as the problems in SR(40).

We ran NeuroSAT for 512 iterations on each of them and found that we could successfully decode solutions for 85% of them.

In contrast, Survey Propagation (SP) BID3 , the canonical (learning-free) message passing algorithm for satisfiability, does not on its own converge to a satisfying assignment on any of these problems.

6 This suggests that NeuroSAT has not simply found a way to approximate SP, but rather has synthesized a qualitatively different algorithm.

NeuroSAT (trained on SR(U(10, 40))) can find satisfying assignments but is not helpful in constructing proofs of unsatisfiability.

When it runs on an unsatisfiable problem, it keeps searching for a satisfying assignment indefinitely and non-systematically.

However, when we train the same architecture on a dataset in which each unsatisfiable problem has a small subset of clauses that are already unsatisfiable (called an unsat core), it learns to detect these unsat cores instead of searching for satisfying assignments.

The literals involved in the unsat core can be decoded from its internal activations.

When the number of literals involved in the unsat core is small relative to the total number of literals, knowing the literals involved in the unsat core can enable constructing a resolution proof more efficiently.

We generated a new distribution SRC(n, u) that is similar to SR(n) except that every unsatisfiable problem contains a small unsat core.

Here n is the number of variables as before, and u is an unsat core over x 1 , . . .

, x k (k < n) that can be made into a satisfiable set of clauses u by negating a single literal.

We sample a pair from SRC(n, u) as follows.

First, we initialize a problem with u , and then we sample clauses (over x 1 to x n ) just as we did for SR(n) until the problem becomes unsatisfiable.

We can now negate a literal in the final clause to get a satisfiable problem p s , and then we can swap u for u in p s to get p u , which is unsatisfiable since it contains the unsat core u. We created train and test datasets from SRC(40, u) with u sampled at random for each problem from a collection of three unsat cores ranging from three clauses to nine clauses: the unsat core R from BID14 , and the two unsat cores resulting from encoding the pigeonhole principles PP(2, 1) and PP(3, 2).

We trained our architecture on this dataset, and we refer to the trained model as NeuroUNSAT.

FIG1 ).

In both cases, the literals in the first six rows are involved in the unsat core.

In 7a, NeuroUNSAT inspects the modified core u of the satisfiable problem but concludes that it does not match the pattern.

In 7b, NeuroUNSAT finds the unsat core u and votes unsat with high confidence (dark blue).NeuroUNSAT is able to predict satisfiability on the test set with 100% accuracy.

Upon inspection, it seems to do so by learning to recognize the unsat cores.

FIG6 shows NeuroUNSAT running on a pair of problems from SRC(30, PP(3, 2)).

In both cases, the literals in the first six rows are involved in the unsat core.

In FIG6 , NeuroUNSAT inspects the modified core u of the satisfiable problem but concludes that it does not match the pattern exactly.

In FIG6 , NeuroUNSAT finds the unsat core u and votes unsat with high confidence (dark blue).

As in §6, the literals involved in the unsat core can sometimes be decoded from the literal votes L (T ) * , but it is more reliable to 2-cluster the higher-dimensional literal embeddings L (T ) .

On the test set, the small number of literals involved in the unsat core end up in their own cluster 98% of the time.

Note that we do not expect NeuroUNSAT to generalize to arbitary unsat cores: as far as we know it is simply memorizing a collection of specific subgraphs, and there is no evidence it has learned a generic procedure to prove unsat.

There have been many attempts over the years to apply statistical learning to various aspects of the SAT problem: restart strategies BID11 ), branching heuristics (Liang et al., 2016 BID10 BID7 , parameter tuning BID22 , and solver selection BID26 .

None of these approaches use neural networks, and instead make use of both generic graph features and features extracted from the runs of SAT solvers.

Moreover, these approaches are designed to assist existing solvers and do not aim to solve SAT problems on their own.

From the machine learning perspective, the closest work to ours is BID19 , which showed that an MPNN can be trained to predict the unique solutions of Sudoku puzzles.

We believe that their network's success is an instance of the phenomenon we study in this paper, namely that MPNNs can synthesize local search algorithms for constraint satisfaction problems.

BID6 present a neural network architecture that can learn to predict whether one propositional formula entails another by randomly sampling and evaluating candidate assignments.

Unlike NeuroSAT, their network does not perform heuristic search and can only work on simple problems for which random guessing is tractable.

There have also been several recent papers showing that various neural network architectures can learn good heuristics for NP-hard combinatorial optimization problems BID25 BID1 BID5 ; however, finding low-cost solutions to optimization problems requires less precise reasoning than finding satisfying assignments.

Our main motivation has been scientific: to better understand the extent to which neural networks are capable of precise, logical reasoning.

Our work has definitively established that neural networks can learn to perform discrete search on their own without the help of hard-coded search procedures, even after only end-to-end training with minimal supervision.

We found this result surprising and think it constitutes an important contribution to the community's evolving understanding of the capabilities and limitations of neural networks.

Although not our primary concern, we also hope that our findings eventually lead to improvements in practical SAT solving.

As we stressed early on, as an end-to-end SAT solver the trained NeuroSAT system discussed in this paper is still vastly less reliable than the state-of-the-art.

We concede that we see no obvious path to beating existing SAT solvers.

One approach might be to continue to train NeuroSAT as an end-to-end solver on increasingly difficult problems.

A second approach might be to use a system like NeuroSAT to help guide decisions within a more traditional SAT solver, though it is not clear that NeuroSAT provides any useful information before it finds a satisfying assignment.

However, as we discussed in §8, when we trained our architecture on different data it learned an entirely different procedure.

In a separate experiment omitted for space reasons, we also trained our architecture to predict whether there is a satisfying assignment involving each individual literal in the problem and found that it was able to predict these bits with high accuracy as well.

Unlike NeuroSAT, it made both type I and type II errors, had no discernable phase transition, and could make reasonable predictions within only a few rounds.

We believe that architectures descended from NeuroSAT will be able to learn very different mechanisms and heuristics depending on the data they are trained on and the details of their objective functions.

We are cautiously optimistic that a descendant of NeuroSAT will one day lead to improvements to the state-of-the-art.

@highlight

We train a graph network to predict boolean satisfiability and show that it learns to search for solutions, and that the solutions it finds can be decoded from its activations.

@highlight

The paper describes a general neural network architecture for predicting satisfiability

@highlight

This paper presents the NeuroSAT architecture which uses a deep message passing neural net for predicting the satisfiability of CNF instances
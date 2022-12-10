We consider the learning of algorithmic tasks by mere observation of input-output pairs.

Rather than studying this as a black-box discrete regression problem with no assumption whatsoever on the input-output mapping, we concentrate on tasks that are amenable to the principle of divide and conquer, and study what are its implications in terms of learning.

This principle creates a powerful inductive bias that we leverage with neural architectures that are defined recursively and dynamically, by learning two scale- invariant atomic operations: how to split a given input into smaller sets, and how to merge two partially solved tasks into a larger partial solution.

Our model can be trained in weakly supervised environments, namely by just observing input-output pairs, and in even weaker environments, using a non-differentiable reward signal.

Moreover, thanks to the dynamic aspect of our architecture, we can incorporate the computational complexity as a regularization term that can be optimized by backpropagation.

We demonstrate the flexibility and efficiency of the Divide- and-Conquer Network on several combinatorial and geometric tasks: convex hull, clustering, knapsack and euclidean TSP.

Thanks to the dynamic programming nature of our model, we show significant improvements in terms of generalization error and computational complexity.

Algorithmic tasks can be described as discrete input-output mappings defined over variable-sized inputs, but this "black-box" vision hides all the fundamental questions that explain how the task can be optimally solved and generalized to arbitrary inputs.

Indeed, many tasks have some degree of scale invariance or self-similarity, meaning that there is a mechanism to solve it that is somehow independent of the input size.

This principle is the basis of recursive solutions and dynamic programming, and is ubiquitous in most areas of discrete mathematics, from geometry to graph theory.

In the case of images and audio signals, invariance principles are also critical for success: CNNs exploit both translation invariance and scale separation with multilayer, localized convolutional operators.

In our scenario of discrete algorithmic tasks, we build our model on the principle of divide and conquer, which provides us with a form of parameter sharing across scales akin to that of CNNs across space or RNNs across time.

Whereas CNN and RNN models define algorithms with linear complexity, attention mechanisms BID1 generally correspond to quadratic complexity, with notable exceptions BID0 .

This can result in a mismatch between the intrinsic complexity required to solve a given task and the complexity that is given to the neural network to solve it, which Figure 1: Divide and Conquer Network.

The split phase is determined by a dynamic neural network S θ that splits each incoming set into two disjoint sets: {X j+1,l , X j+1,l+1 } = S θ (X j,m ), with X j,m = X j+1,l X j+1,l+1 .

The merge phase is carried out by another neural network M φ that combines two partial solutions into a solution of the coarser scale: Y j,m = M φ (Y j+1,l , Y j+1,l+1 ); see Section 3 for more details.

may impact its generalization performance.

Our motivation is that learning cannot be 'complete' until these complexities match, and we start this quest by first focusing on problems for which the intrinsic complexity is well known and understood.

Our Divide-and-Conquer Networks (DiCoNet ) contain two modules: a split phase that is applied recursively and dynamically to the input in a coarse-to-fine way to create a hierarchical partition encoded as a binary tree; and a merge phase that traces back that binary tree in a fine-to-coarse way by progressively combining partial solutions; see Figure 1 .

Each of these phases is parametrized by a single neural network that is applied recursively at each node of the tree, enabling parameter sharing across different scales and leading to good sample complexity and generalisation.

In this paper, we attempt to incorporate the scale-invariance prior with the desiderata to only require weak supervision.

In particular, we consider two setups: learning from input-output pairs, and learning from a non-differentiable reward signal.

Since our split block is inherently discrete, we resort to policy gradient to train the split parameters, while using standard backpropagation for the merge phase; see Section 5.

An important benefit of our framework is that the architecture is dynamically determined, which suggests using the computational complexity as a regularization term.

As shown in the experiments, computational complexity is a good proxy for generalisation error in the context of discrete algorithmic tasks.

We demonstrate our model on algorithmic and geometric tasks with some degree of scale self-similarity: planar convex-hull, k-means clustering, Knapsack Problem and euclidean TSP.

Our numerical results on these tasks reaffirm the fact that whenever the structure of the problem has scale invariance, exploiting it leads to improved generalization and computational complexity over non-recursive approaches.

Using neural networks to solve algorithmic tasks is an active area of current research, but its models can be traced back to context free grammars BID8 .

In particular, dynamic learning appears in works such as BID20 and BID25 .

The current research in the area is dominated by RNNs BID14 BID11 , LSTMs BID13 , sequence-to-sequence neural models BID29 , attention mechanisms BID27 BID0 and explicit external memory models BID28 BID23 BID10 BID30 .

We refer the reader to BID14 and references therein for a more exhaustive and detailed account of related work.

Amongst these works, we highlight some that are particularly relevant to us.

Neural GPU BID15 defines a neural architecture that acts convolutionally with respect to the input and is applied iteratively o(n) times, where n is the input size.

It leads to fixed computational machines with total Θ(n 2 ) complexity.

Neural Programmer-Interpreters BID21 ) introduce a compositional model based on a LSTM that can learn generic programs.

It is trained with full supervision using execution traces.

Directly related, BID4 incorporates recursion into the NPI to enhance its capacity and provide learning certificates in the setup where recursive execution traces are available for supervision.

Hierarchical attention mechanisms have been explored in BID0 .

They improve the complexity of the model from o(n 2 ) of traditional attention to o(n log n), similarly as our models.

Finally, Pointer Networks BID27 ;a) modify classic attention mechanisms to make them amenable to adapt to variable inputdependent outputs, and illustrate the resulting models on geometric algorithmic tasks.

They belong to the Θ(n 2 ) category class.3 PROBLEM SETUP

We consider tasks consisting in a mapping T between a variable-sized input set X = {x 1 , . . .

, x n }, x j ∈ X into an ordered set Y = {y 1 , . . . , y m(n) }, y j ∈ Y. This setup includes problems where the output size m(n) differs from the input size n, and also problems where Y is a labeling of input elements.

In particular, we will study in detail the case where Y ⊆ X (and in particular Y ⊆ X ).We are interested in tasks that are self-similar across scales, meaning that if we consider the recursive decomposition of T as DISPLAYFORM0 where S splits the input into smaller sets, and M merges the solved corresponding sub-problems, then both M and S are significantly easier to approximate with data-driven models.

In other words, the solution of the task for a certain size n is easier to estimate as a function of the partial solutions T (S j (X)) than directly from the input.

Under this assumption, the task T can thus be solved by first splitting the input into s strictly smaller subsets S j (X), solving T on each of these subsets, and then appropriately merging the corresponding outputs together.

In order words, T can be solved by recursion.

A particularly simple and illustrative case is the binary setup with s = 2 and S 1 (X) ∩ S 2 (X) = ∅, that we will adopt in the following for simplicity.

Our first goal is to learn how to perform T for any size n, by observing only input-output example DISPLAYFORM0 Throughout this work, we will make the simplifying assumption of binary splitting (s = 2), although our framework extends naturally to more general versions.

Given an input set X associated with output Y , we first define a split phase that breaks X into a disjoint partition tree P(X): DISPLAYFORM1 and X = X 1,0 X 1,1 .

This partition tree is obtained by recursively applying a trainable binary split module S θ : DISPLAYFORM2 DISPLAYFORM3 Here, J indicates the number of scales or depth of recursion that our model applies for a given input X, and S θ is a neural network that takes a set as input and produces a binary, disjoint partition as output.

Eq. (3) thus defines a hierarchical partitioning of the input that can be visualized as a binary tree; see Figure 1 .

This binary tree is data-dependent and will therefore vary for each input example, dictated by the current choice of parameters for S θ .The second phase of the model takes as input the binary tree partition P(X) determined by the split phase and produces an estimateŶ .

We traverse upwards the dynamic computation tree determined by the split phase using a second trainable block, the merge module M φ : DISPLAYFORM4 DISPLAYFORM5 Here we have denoted byM the atomic block that transforms inputs at the leaves of the split tree, and M φ is a neural network that takes as input two (possibly ordered) inputs and merges them into another (possibly ordered) output.

In the setup where Y ⊆ X, we further impose that DISPLAYFORM6 , to guarantee that the computation load does not diverge with J.

Another setup we can address with (1) consists in problems where one can assign a cost (or reward) to a given partitioning of an input set.

In that case, Y encodes the labels assigned to each input element.

We also assume that the reward function has some form of self-similarity, in the sense that one can relate the reward associated to subsets of the input to the total reward.

In that case, (3) is used to map an input X to a partition P(X), determined by the leaves of the tree {X J,k } k , that is evaluated by an external black-box returning a cost L(P(X)).

For instance, one may wish to perform graph coloring satisfying a number of constraints.

In that case, the cost function would assign L(P(X)) = 0 if P(X) satisfies the constraints, and L(P(X)) = |X| otherwise.

In its basic form, since P(X) belongs to a discrete space of set partitions of size super-exponential in |X| and the cost is non-differentiable, optimizing L(P(X)) over the partitions of X is in general intractable.

However, for tasks with some degree of self-similarity, one can expect that the combinatorial explosion can be avoided.

Indeed, if the cost function L is subadditive, i.e., DISPLAYFORM0 then the hierarchical splitting from (3) can be used as an efficient greedy strategy, since the right hand side acts as a surrogate upper bound that depends only on smaller sets.

In our case, since the split phase is determined by a single block S θ that is recursively applied, this setup can be cast as a simple fixed-horizon (J steps) Markov Decision Process, that can be trained with standard policy gradient methods; see Section 5.

Besides the prospect of better generalization, the recursion (1) also enables the notion of computational complexity regularization.

Indeed, in tasks that are scale invariant the decomposition in terms of M and S is not unique in general.

For example, in the sorting task with n input elements, one may select the largest element of the array and query the sorting task on the remaining n − 1 elements, but one can also attempt to break the input set into two subsets of similar size using a pivot, and query the sorting on each of the two subsets.

Both cases reveal the scale invariance of the problem, but the latter leads to optimal computational complexity ( Θ(n log n) ) whereas the former does not (Θ(n 2 )).

Therefore, in a trainable divide-and-conquer architecture, one can regularize the split operation to minimize computational complexity; see Appendix A.

The split block S θ receives as input a variable-sized set X = (x 1 , . . .

, x n ) and produces a binary partition X = X 0 X 1 .

We encode such partition with binary labels z 1 . . .

z n , z m ∈ {0, 1}, m ≤ n. These labels are sampled from probabilities p θ (z m = 1 | X) that we now describe how to parametrize.

Since the model is defined over sets, we use an architecture that certifies that p θ (z m = 1 | X) are invariant by permutation of the input elements.

The Set2set model BID26 ) constructs a nonlinear set representation by cascading R layers of DISPLAYFORM0 with m ≤ n , r ≤ R , h DISPLAYFORM1 The parameters of S θ are thus θ = {B 0 , B 1,r , B 2,r , b}. In order to avoid covariate shifts given by varying input set distributions and sizes, we consider a normalization of the input that standardizes the input variables x j and feeds the mean and variance µ(X) = (µ 0 , σ) to the first layer.

If the input has some structure, for instance X is the set of vertices of a graph, a simple generalization of the above model is to estimate a graph structure specific to each layer: DISPLAYFORM2 where A (r) is a similarity kernel computed as a symmetric function of current hidden variables: DISPLAYFORM3 .

This corresponds to the so-called graph neural networks BID22 BID6 or neural message passing BID9 .Finally, the binary partition tree P(X) is constructed recursively by first computing p θ (z | X), then sampling from the corresponding distributions to obtain X = X 0 X 1 , and then applying S θ recursively on X 0 and X 1 until the partition tree leaves have size smaller than a predetermined constant, or the number of scales reaches a maximum value J. We denote the resulting distribution over tree partitions by P(X) ∼ S θ (X).

The merge block M φ takes as input a pair of sequences Y 0 , Y 1 and produces an output sequence O. Motivated by our applications, we describe first an architecture for this module in the setup where the output sequence is indexed by elements from the input sequences, although our framework can be extended to more general setups seamlessly.

in Section C.Given an input sequence Y , the merge module computes a stochastic matrix Γ Y (where each row is a probability distribution) such that the output O is expressed by binarizing its entries and multiplying it by the input: DISPLAYFORM0 Since we are interested in weakly supervised tasks, the target output only exists at the coarsest scale of the partition tree.

We thus also consider a generative version M g φ of the merge block that uses its own predictions in order to sample an output sequence.

The initial merge operation at the finest scaleM is defined as the previous merge module applied to the input (X J,k , ∅).

This merge module operation can be instantiated with Pointer Networks BID27 and with Graph Neural Networks/ Neural Message Passing BID9 BID16 BID3 .Pointer Networks We consider a Pointer Network (PtrNet) BID27 to our inputoutput interface as our merge block M φ .

A PtrNet is an auto-regressive model for tasks where the output sequence is a permutation of a subsequence of the input.

The model encodes each input sequence Y q = (x 1,q , . . .

, x nq,q ), q = 0, 1, into a global representation e q := e q,nq , q = 0, 1, by sequentially computing e 1,q , . . .

, e nq,q with an RNN.

Then, another RNN decodes the output sequence with initial state d 0 = ρ(A 0 e 0 + A 1 e 1 ), as described in detail in Appendix D. The trainable parameters φ regroup to the RNN encoder and decoder parameters.

Graph Neural Networks Another use case of our Divide and Conquer Networks are problems formulated as paths on a graph, such as convex hulls or the travelling salesman problem.

A path on a graph of n nodes can be seen as a binary signal over the n × n edge matrix.

Leveraging recent incarnations of Graph Neural Networks/ Neural Message Passing that consider both node and edge hidden features BID9 BID16 BID3 , the merge module can be instantiated with a GNN mapping edge-based features from a bipartite graph representing two partial solutions Y 0 , Y 1 , to the edge features encoding the merged solution.

Specifically, we consider the following update equations: DISPLAYFORM1 where ϕ is a symmetric, non-negative function parametrized with a neural network.

Given a partition tree P(X) = {X j,k } j,k , we perform a merge operation at each node (j, k).

The merge operation traverses the tree in a fine-to-coarse fashion.

At the leaves of the tree, the sets DISPLAYFORM0 , and, while j > 0, these outputs are recursively transformed along the binary tree as DISPLAYFORM1 , Y j+1,2k+1 ) , 0 < j < J , using the auto-regressive version, until we reach the scale with available targets:Ŷ = M φ (Y 1,0 , Y 1,1 ) .

At test-time, without ground-truth outputs, we replace the last M φ by its generative version M g φ .

The recursive merge defined at (4.2.2) can be viewed as a factorized attention mechanism over the input partition.

Indeed, the merge module outputs (21) include the stochastic matrix Γ = (p 1 , . . .

, p S ) whose rows are the p s probability distributions over the indexes.

The number of rows of this matrix is the length of the output sequence and the number of columns is the length of the input sequence.

Since the merge blocks are cascaded by connecting each others outputs as inputs to the next block, given a hierarchical partition of the input P(X), the overall mapping can be written aŝ DISPLAYFORM0 It follows that the recursive merge over the binary tree is a specific reparametrization of the global permutation matrix, in which the permutation matrix has been decomposed into a product of permutations dictated by the binary tree, indicating our belief that many routing decisions are done locally within the original set.

The model is trained with maximum likelihood using the product of the non-binarized stochastic matrices.

Lastly, in order to avoid singularities we need to enforce that log p s,ts is well-defined and therefore that p s,ts > 0.

We thus regularize the quantization step (21) by replacing 0, 1 with 1/J , 1 − n 1/J respectively.

We also found useful to binarize the stochastic matrices at fine scales when the model is close to convergence, so gradients are only sent at coarsest scale.

For simplicity, we use the notation p φ (Y | P(X)) = J j=0 Γ j = M φ (P(X)), where now the matricesΓ j are not binarized.

This section describes how the model parameters {θ, φ} are estimated under two different learning paradigms.

Given a training set of pairs {(X l , Y l )} l≤L , we consider the loss DISPLAYFORM0 Section 4.2 explained how the merge phase M φ is akin to a structured attention mechanism.

Equations (10) show that, thanks to the parameter sharing and despite the quantizations affecting the finest leaves of the tree, the gradient DISPLAYFORM1 is well-defined and non-zero almost everywhere.

However, since the split parameters are separated from the targets through a series of discrete sampling steps, the same is not true for ∇ θ log p θ,φ (Y | X).We therefore resort to the identity used extensively in policy gradient methods.

For arbitrary F defined over partitions X , and denoting by f θ (X ) the probability density of the random partition S θ (X), we have DISPLAYFORM2 Since the split variables at each node of the tree are conditionally independent given its parent, we can compute log f θ (P(X)) as DISPLAYFORM3 By plugging F (P(X)) = log p φ (Y | P(X)) we thus obtain an efficient estimation of DISPLAYFORM4 From FORMULA0 , it is straightforward to train our model in a regime where a given partition P(X) of an input set is evaluated by a black-box system producing a reward R(P(X)).

Indeed, in that case, the loss becomes DISPLAYFORM5 which can be minimized using (13) with F (P(X)) = R(P(X)).

We present experiments on three representative algorithmic tasks: convex hull, clustering and knapsack.

We also report additional experiments on the Travelling Salesman Problem in the Appendix.

The hyperparameters used for each experiment can be found at the Appendix.

The convex hull of a set of n points X = {x 1 , . . .

, x n } is defined as the extremal set of points of the convex polytope with minimum area that contains them all.

The planar (2d) convex hull is a well known task in discrete geometry and the optimal algorithm complexity is achieved using divide and conquer strategies by exploiting the self-similarity of the problem.

The strategy for this task consists of splitting the set of points into two disjoint subsets and solving the problem recursively for each.

If the partition is balanced enough, the overall complexity of the algorithm amounts to Θ(n log n).

The split phase usually takes Θ(n log n) because each node involves a median computation to make the balanced partition property hold.

The merge phase can be done in linear time on the total number of points of the two recursive solutions, which scales logarithmically with the total number of points when sampled uniformly inside a polytope BID7 .

Scales go fine-to-coarse from left to right.

Left:

Split has already converged using the rewards coming from the merge.

It gives disjoint partitions to ease the merge work.

Right: DiCoNet with random split phase.

Although the performance degrades due to the non-optimal split strategy, the model is able to output the correct convex hull for most of the cases.

We test the DiCoNet on the setting consisting of n points sampled in the unit square [0, 1] 2 ⊂ R 2 .

This is the same setup as BID27 .

The training dataset has size sampled uniformly from 6 to 50.

The training procedure is the following; we first train the baseline pointer network until convergence.

Then, we initialize the DiCoNet merge parameters with the baseline and train both split and merge blocks.

We use this procedure in order to save computational time for the experiments, however, we observe convergence even when the DiCoNet parameters are initialized from scratch.

We supervise the merge block with the product of the continuous Γ matrices.

For simplicity, instead of defining the depth of the tree dynamically depending on the average size of the partition, we fix it to 0 for 6-12, 1 for 12-25 and 2 for 25-50; see FIG1 and Table 1 Table 1 : ConvHull test accuracy results with the baseline PtrNet and different setups of the DiCoNet .

The scale J has been set to 3 for n=100 and 4 for n=200.

At row 2 we observe that when the split block is not trained we get worse performance than the baseline, however, the generalization error shrinks faster on the baseline.

When both blocks are trained jointly, we clearly outperform the baseline.

In Row 3 the split is only trained with REINFORCE, and row 4 when we add the computational regularization term (See Supplementary) enforcing shallower trees.

We tackle the task of clustering a set of n points with the DiCoNet in the setting described in (14).

The problem consists in finding k clusters of the data with respect to the Euclidean distance in R d .

The problem reduces to solving the following combinatorial problem over input partitions P(X): DISPLAYFORM0 where σ 2 i is the variance of each subset of the partition P(X), and n i its cardinality.

We only consider the split block for this task because the combinatorial problem is over input partitions.

We use a GNN (6) for the split block.

The graph is created from the points in R d by taking DISPLAYFORM1 2 ) as weights and instantiating the embeddings with the euclidean coordinates.

We test the model in two different datasets.

The first one, which we call "Gaussian", is constructed by sampling k points in the unit square of dimension d, then sampling n k points from gaussians of variance 10 −3 centered at each of the k points.

The second one is constructed by picking 3x3x3 random patches of the RGB images from the CIFAR-10 dataset.

The baseline is a modified version of the split block in which instead of computing binary probabilities we compute a final softmax of dimensionality k in order to produce a labelling over the input.

We compare its performance with the DiCoNet with binary splits and log k scales where we only train with the reward of the output partition at the leaves of the tree, hence, DiCoNet is optimizing k-means FORMULA0 and not a recursive binary version of it.

We show the corresponding cost ratio with Lloyd's and recursive Lloyd's (binary Lloyd's applied recursively); see Table 2 .

In this case, no split regularization has been added to enforce balanced partitions.

Given a set of n items, each with weight w i ≥ 0 and value v i ∈ R, the 0-1 Knapsack problem consists in selecting the subset of the input set that maximizes the total value, so that the total weight does not exceed a given limit: Table 2 : We have used n = 20 · k points for the Gaussian dataset and n = 500 for the CIFAR-10 patches.

The baseline performs better than the DiCoNet when the number of clusters is small but DiCoNet scales better with the number of clusters.

When Lloyd's performs much better than its recursive version ("Gaussian" with d = 10), we observe that DiCoNet performance is between the two.

This shows that although having a recursive structure, DiCoNet is acting like a mixture of both algorithms, in other words, it is doing better than applying binary clustering at each scale.

DiCoNet achieves the best results in the CIFAR-10 patches dataset, where Lloyd's and its recursive version perform similarly with respect to the k-means cost.

DISPLAYFORM0 It is a well-known NP-hard combinatorial optimization problem, which can be solved exactly with dynamic programming using O(nW ) operations, referred as 'pseudo-polynomial' time in the literature.

For a given approximation error > 0, one can use dynamic programming to obtain a polynomial time approximation within a factor 1 − of the optimal solution BID18 .

A remarkable greedy algorithm proposed by Dantzig sorts the input elements according to the ratios ρ i = vi wi and picks them in order until the maximum allowed weight is attained.

Recently, authors considered LSTM-based models to approximate knapsack problems BID2 .We instantiate our DiCoNet in this problem as follows.

We use a GNN architecture as our split module, which is configured to select a subset of the input that fills a fraction α of the target capacity W .

In other words, the GNN split module accepts as input a problem instance {(x 1 , w 1 ), . . .

, (x n , w n )} and outputs a probability vector (p 1 , . . . , p n ).

We sample from the resulting multinomial distribution without replacement until the captured total weight reaches αW .

We then fill the rest of the capacity (1 − α)W recursively, feeding the remaining unpicked elements to the same GNN module.

We do this a number J of times, and in the last call we fill all the capacity, not just the α fraction.

The overall DiCoNet model is illustrated in Figure 3 .We generate 20000 problem instances of size n = 50 to train the model, and evaluate its performance on new instances of size n = 50, 100, 200.

The weights and the values of the elements follow a uniform distribution over [0, 1] , and the capacities are chosen from a uniform distribution over [0.2 n, 0.3 n].

This dataset is similar to the one in BID2 , but has a slightly variable capacity, which we hope will help the model to generalize better.

We choose α = 0.5 in our experiments.

We train the model using REINFORCE (13), and to reduce the gradient variances we consider as baseline the expected reward, approximated by the average of a group of samplings.

Table 3 reports the performance results, measured with the ratio Vopt Vout (so the lower the better).

The baseline model is a GNN which selects the elements using a non-recursive architecture, trained using Reinforce.

We verify how the non-recursive model quickly deteriorates as n increases.

On the other hand, the DiCoNet model performs significatively better than the considered alternatives, even for lengths n = 100.

However, we observe that the Dantzig greedy algorithm eventually outperforms the DiCoNet for sufficiently large input n = 200, suggesting that further improvements may come from relaxing the scale invariance assumption, or by incorporating extra prior knowledge of the task.

This approach of the knapsack problem does not perform as good as BID2 in obtaining the best approximation.

However, we have presented an algorithm that relies on a rather simple 5-layer GNN, applied a fixed number of times (with quadratic complexity respect to n, whereas the pointer network-based LSTM model is cubic), which has proven to have other strengths, such as the ability to generalize to larger ns that the one used for training.

Thus, this approach illustrates well the aim of the DiCoNet .

We believe that it would also be interesting for future work to combine the strengths of both aproaches.

Table 3 : Performance Ratios of different models trained with n = 50 (and using 3 splits in the DiCoNet ) and tested for n ∈ {50, 100, 200} (and different number of splits).

We report the number of splits that give better performances at each n for the DiCoNet .

Note that for n = 50 the model does best with 3 splits, the same as in training, but with larger n more splits give better solutions, as would be desired.

Observe that even for n = 50, the DiCoNet architecture significatively outperforms the non-recursive model, highlighting the highly constrained nature of the problem, in which decisions over an element are highly dependent on previously chosen elements.

Although the DiCoNet clearly outperforms the baseline and the Dantzig algorithm for n ≤ 100, its performance eventually degrades at n = 200; see text.

Figure 3: DiCoNet Architecture for the Knapsack problem.

A GNN Split module selects a subset of input elements until a fraction α of the allowed budget is achieved; then the remaining elements are fed back recursively into the same Split module, until the total weight fills the allowed budget.

We have presented a novel neural architecture that can discover and exploit scale invariance in discrete algorithmic tasks, and can be trained with weak supervision.

Our model learns how to split large inputs recursively, then learns how to solve each subproblem and finally how to merge partial solutions.

The resulting parameter sharing across multiple scales yields improved generalization and sample complexity.

Due to the generality of the DiCoNet , several very different problems have been tackled, some with large and others with weak scale invariance.

In all cases, our inductive bias leads to better generalization and computational complexity.

An interesting perspective is to relate our scale invariance with the growing paradigm of meta-learning; that is, to what extent one could supervise the generalization across problem sizes.

In future work, we plan to extend the results of the TSP by increasing the number of splits J, by refining the supervised DiCoNet model with the non-differentiable TSP cost, and by exploring higher-order interactions using Graph Neural Networks defined over graph hierarchies BID17 .

We also plan to experiment on other NP-hard combinatorial tasks.

The TSP is a prime instance of a NP-hard combinatorial optimization task.

Due to its important practical applications, several powerful heuristics exist in the metric TSP case, in which edges satisfy the triangular inequality.

This motivates data-driven models to either generalize those heuristics to general settings, or improve them.

Data-driven approaches to the TSP can be formulated in two different ways.

First, one can use both the input graph and the ground truth TSP cycle to train the model to predict the ground truth.

Alternatively, one can consider only the input graph and train the model to minimize the cost of the predicted cycle.

The latter is more natural since it optimizes the TSP cost directly, but the cost of the predicted cycle is not differentiable w.r.t model parameters.

Some authors have successfully used reinforcement learning techniques to address this issue BID5 , BID2 , although the models suffer from generalization to larger problem instances.

Here we concentrate on that generalization aspect and therefore focus on the supervised setting using ground truth cycles.

We compare a baseline model that used the formulation of TSP as a Quadratic Assignment Problem to develop a Graph Neural Network BID19 ) with a DiCoNet model that considers split and merge modules given by separate GNNs.

More precisely, the split module is a GNN that receives a graph and outputs binary probabilities for each node.

This module is applied recursively until a fixed scale J and the baseline is used at every final sub-graph of the partition, resulting in signals over the edges encoding possible partial solutions of the TSP.

Finally, the merge module is another GNN that receives a pair of signals encoded as matrices from its leaves and returns a signal over the edges of the complete union graph.

As in BID19 , we generated 20k training examples and tested on 1k other instances.

Each one generated by uniformly sampling DISPLAYFORM0 We build a complete graph with Table 4 : DiCoNet has been trained for n = 20 nodes and only one scale (J = 1).

We used the pre-trained baseline for n = 10 as model on both leaves.

BS1 and BS2 correspond to the baseline trained for n = 10 and n = 20 nodes respectively.

Although for small n, both baselines outperform the DiCoNet , the scale invariance prior of the DiCoNet is leveraged at larger scales resulting in better results and scalability.

DISPLAYFORM1 In Table 4 we report preliminary results of the DiCoNet for the TSP problem.

Although DiCoNet and BS2 have both been trained on graphs of the same size, the dynamic model outperforms the baseline due to its powerful prior on scale invariance.

Although the scale invariance of the TSP is not as clear as in the previous problems (it is not straightforward how to use two TSP partial solutions to build a larger one), we observe that some degree of scale invariance it is enough in order to improve on scalability.

As in previous experiments, the joint work of the split and merge phase is essential to construct the final solution.

The merge block M φ takes as input a pair of sequences Y 0 , Y 1 and produces an output sequence O. We describe first the architecture for this module, and then explain on how it is modified to perform the finest scale computationM φ .

We modify a Pointer Network (PtrNet) BID27 to our input-output interface as our merge block M φ .

A PtrNet is an auto-regressive model for tasks where the output sequence is a permutation of a subsequence of the input.

The model encodes each input sequence Y q = (x 1,q , . . .

, x nq,q ), q = 0, 1, into a global representation e q := e q,nq , q = 0, 1, by sequentially computing e 1,q , . . .

, e nq,q with an RNN.

Then, another RNN decodes the output sequence with initial state d 0 = ρ(A 0 e 0 + A 1 e 1 ).

The trainable parameters φ regroup to the RNN encoder and decoder parameters.

Suppose first that one has a target sequence T = (t 1 . . .

t S ) for the output of the merge.

In that case, we use a conditional autoregressive model of the form e q,i = f enc (e q,i−1 , y q,i ) i = 1, . . .

, n q , q = 0, 1 , DISPLAYFORM0 The conditional probability of the target given the inputs is computed by performing attention over the embeddings e q,i and interpreting the attention as a probability distribution over the input indexes: leading to Γ = (p 1 , . . .

, p S ) .

The output O is expressed in terms of Γ by binarizing its entries and multiplying it by the input: DISPLAYFORM1 However, since we are interested in weakly supervised tasks, the target output only exists at the coarsest scale of the partition tree.

We thus also consider a generative version M g φ of the merge block that uses its own predictions in order to sample an output sequence.

Indeed, in that case, we replace equation FORMULA0 by e q,i = f enc (e q,i−1 , y q,i ) i = 1, . . .

, n q , q = 0, 1 , d s = f dec (d s−1 , y os−1 ) s = 1, . . . , Swhere o s is computed as o s = x arg max ps , s ≤ S. The initial merge operation at the finest scaleM is defined as the previous merge module applied to the input (X J,k , ∅).

We describe next how the successive merge blocks are connected so that the whole system can be evaluated and run.

<|TLDR|>

@highlight

Dynamic model that learns divide and conquer strategies by weak supervision.

@highlight

Proposes to add new inductive bias to neural network architecture by using a divide and conquer strategy.

@highlight

This paper studies problems that can be solved using a dynamic programming approach, and proposes a neural network architecture to solve such problems that beats sequence to sequence baselines.

@highlight

The paper proposes a unique network architecture that can learn divide-and-conquer strategies to solve algorithmic tasks.
The advance of node pooling operations in Graph Neural Networks (GNNs) has lagged behind the feverish design of new message-passing techniques, and pooling remains an important and challenging endeavor for the design of deep architectures.

In this paper, we propose a pooling operation for GNNs that leverages a differentiable unsupervised loss based on the minCut optimization objective.

For each node, our method learns a soft cluster assignment vector that depends on the node features, the target inference task (e.g., a graph classification loss), and, thanks to the minCut objective, also on the connectivity structure of the graph.

Graph pooling is obtained by applying the matrix of assignment vectors to the adjacency matrix and the node features.

We validate the effectiveness of the proposed pooling method on a variety of supervised and unsupervised tasks.

A fundamental component in deep convolutional neural networks is the pooling operation, which replaces the output of convolutions with local summaries of nearby points and is usually implemented by maximum or average operations (Lee et al., 2016) .

State-of-the-art architectures alternate convolutions, which extrapolate local patterns irrespective of the specific location on the input signal, and pooling, which lets the ensuing convolutions capture aggregated patterns.

Pooling allows to learn abstract representations in deeper layers of the network by discarding information that is superfluous for the task, and keeps model complexity under control by limiting the growth of intermediate features.

Graph Neural Networks (GNNs) extend the convolution operation from regular domains, such as images or time series, to data with arbitrary topologies and unordered structures described by graphs (Battaglia et al., 2018) .

The development of pooling strategies for GNNs, however, has lagged behind the design of newer and more effective message-passing (MP) operations (Gilmer et al., 2017) , such as graph convolutions, mainly due to the difficulty of defining an aggregated version of the original graph that supports the pooled signal.

A naïve pooling strategy in GNNs is to average all nodes features (Li et al., 2016) , but it has limited flexibility since it does not extract local summaries of the graph structure, and no further MP operations can be applied afterwards.

An alternative approach consists in pre-computing coarsened versions of the original graph and then fit the data to these deterministic structures (Bruna et al., 2013) .

While this aggregation accounts for the connectivity of the graph, it ignores task-specific objectives as well as the node features.

In this paper, we propose a differentiable pooling operation implemented as a neural network layer, which can be seamlessly combined with other MP layers (see Fig. 1 ).

The parameters in the pooling layer are learned by combining the task-specific loss with an unsupervised regularization term, which optimizes a continuous relaxation of the normalized minCUT objective.

The minCUT identifies dense graph components, where the nodes features become locally homogeneous after the message-passing.

By gradually aggregating these components, the GNN learns to distil global properties from the graph.

The proposed minCUT pooling operator (minCUTpool) yields partitions that 1) cluster together nodes which have similar features and are strongly connected on the graph, and 2) take into account the objective of the downstream task.

minCUT Pooling Message-passing Figure 1 : A deep GNN architecture where message-passing is followed by minCUT pooling.

Given a graph G = {V, E}, |V| = N , and the associated adjacency matrix A ∈ R N ×N , the K-way normalized minCUT (simply referred to as minCUT) aims at partitioning V in K disjoint subsets by removing the minimum volume of edges.

The problem is equivalent to maximizing

where the numerator counts the edge volume within each cluster, and the denominator counts the edges between the nodes in a cluster and the rest of the graph (Shi & Malik, 2000) .

Let C ∈ R N ×K be a cluster assignment matrix, so that C i,j = 1 if node i belongs to cluster j, and 0 otherwise.

The minCUT problem can be expressed as

where D = diag(A1 N ) is the degree matrix (Dhillon et al., 2004) .

Since problem (2) is NP-hard, it is usually recast in a relaxed formulation that can be solved in polynomial time and guarantees a near-optimal solution (Yu & Shi, 2003) :

While the optimization problem (3) is still non-convex, there exists an optimal solution Q * = U K O, where U K ∈ R N ×K contains the eigenvectors of A corresponding to the K largest eigenvalues, and O ∈ R K×K is an orthogonal transformation (Ikebe et al., 1987) .

Since the elements of Q * are real values rather than binary cluster indicators, the spectral clustering (SC) approach can be used to find discrete cluster assignments.

In SC, the rows of Q * are treated as node representations embedded in the eigenspace of the Laplacian, and are clustered together with standard algorithms such as k-means (Von Luxburg, 2007) .

One of the main limitations of SC lies in the computation of the spectrum of A, which has a memory complexity of O(N 2 ) and a computational complexity of O(N 3 ).

This prevents its applicability to large datasets.

To deal with such scalability issues, the constrained optimization in (3) can be solved by gradient descent algorithms that refine the solution by iterating operations whose individual complexity is O(N 2 ), or even O(N ) (Han & Filippone, 2017) .

Those algorithms search the solution on the manifold induced by the orthogonality constraint on the columns of Q, by performing gradient updates along the geodesics (Wen & Yin, 2013; Collins et al., 2014) .

Alternative approaches rely on the QR factorization to constrain the space of feasible solutions (Damle et al., 2016) , and alleviate the cost O(N 3 ) of the factorization by ensuring that orthogonality holds only on one minibatch at a time (Shaham et al., 2018) .

Other works based on neural networks include an autoencoder trained to map the ith row of the Laplacian to the ith components of the first K eigenvectors, to avoid the spectral decomposition (Tian et al., 2014) .

Yi et al. (2017) use a soft orthogonality constraint to learn spectral embeddings as a volumetric reparametrization of a precomputed Laplacian eigenbase.

Shaham et al. (2018) ; Kampffmeyer et al. (2019) propose differentiable loss functions to partition generic data and process out-of-sample data at inference time.

Nazi et al. (2019) generate balanced node partitions with a GNN, but adopt an optimization that does not encourage cluster assignments to be orthogonal.

Many approaches have been proposed to process graphs with neural networks, including recurrent architectures (Scarselli et al., 2009; Li et al., 2016) or convolutional operations inspired by filters used in graph signal processing (Defferrard et al., 2016; .

Since our focus is on graph pooling, we base our GNN implementation on a simple MP operation, which combines the features of each node with its 1st-order neighbors.

To account for the initial node features, it is possible to introduce self-loops by adding a (scaled) identity matrix to the diagonal of A (Kipf & Welling, 2017).

Since our pooling will modify the structure of the adjacency matrix, we prefer a MP implementation that leaves the original A unaltered and accounts for the initial node features by means of skip connections.

N ×N be the symmetrically normalized adjacency matrix and X ∈ R N ×F the matrix containing the node features.

The output of the MP layer is

where Θ M P = {W m , W s } are the trainable weights relative to the mixing and skip component of the layer, respectively.

The minCUT pooling strategy computes a cluster assignment matrix S ∈ R N ×K by means of a multi-layer perceptron, which maps each node feature x i into the ith row of S:

where Θ P ool = {W 1 ∈ R F ×H , W 2 ∈ R H×K } are trainable parameters.

The softmax function guarantees that s i,j ∈ [0, 1] and enforces the constraints S1 K = 1 N inherited from the optimization problem in (2).

The parameters Θ M P and Θ P ool are jointly optimized by minimizing the usual task-specific loss, as well as an unsupervised loss L u , which is composed of two terms

where · F indicates the Frobenius norm.

The cut loss term, L c , evaluates the minCUT given by the cluster assignment S, and is bounded by −1 ≤ L c ≤ 0.

Minimizing L c encourages strongly connected nodes to be clustered together, since the inner product s i , s j increases whenã i,j is large.

L c has a single maximum, reached when the numerator T r(

This occurs if, for each pair of connected nodes (i.e.,ã i,j > 0), the cluster assignments are orthogonal (i.e., s i , s j = 0).

L c reaches its minimum, −1, when T r(S TÃ S) = T r(S TD S).

This occurs when in a graph with K disconnected components the cluster assignments are equal for all the nodes in the same component and orthogonal to the cluster assignments of nodes in different components.

However, L c is a non-convex function and its minimization can lead to local minima or degenerate solutions.

For example, given a connected graph, a trivial optimal solution is the one that assigns all nodes to the same cluster.

As a consequence of the continuous relaxation, another degenerate minimum occurs when the cluster assignments are all uniform, that is, all nodes are equally assigned to all clusters.

This problem is exacerbated by prior message-passing operations, which make the node features more uniform.

The orthogonality loss term, L o , penalizes the degenerate minima of L c by encouraging the cluster assignments to be orthogonal and the clusters to be of similar size.

Since the two matrices in L o have unitary norm it is easy to see that 0 ≤ L o ≤ 2.

Therefore, L o does not dominate over L c and the two terms can be safely summed directly (see Fig. 4 for an example).

I K can be interpreted as a (rescaled) clustering matrix I K =Ŝ TŜ , whereŜ assigns exactly N/K points to each cluster.

The value of the Frobenius norm between clustering matrices is not dominated by the performance on the largest clusters (Law et al., 2017) and, thus, can be used to optimize intra-cluster variance.

Contrarily to SC methods that search for feasible solutions only within the space of orthogonal matrices, L o only introduces a soft constraint that could be violated during the learning procedure.

Since L c is non-convex, the violation compromises the theoretical guarantee of convergence to the optimum of (3).

However, we note that:

1. the cluster assignments S are well initialized: after the MP operation, the features of the connected vertices become similar and, since the MLP is a smooth function (Nelles, 2013) , it yields similar cluster assignments for those vertices; 2. in the GNN architecture, the minCUT objective is a regularization term and, therefore, a solution which is sub-optimal for (3) could instead be adequate for the specific objective of the downstream task; 3.

optimizing the task-specific loss helps the GNN to avoid the degenerate minima of L c .

The coarsened version of the adjacency matrix and the graph signal are computed as

where the entry x pool i,j in X pool ∈ R K×F is the weighted average value of feature j among the elements in cluster i. A pool ∈ R K×K is a symmetric matrix, whose entries a

are the total number of edges between the nodes in the cluster i, while a pool i,j is the number of edges between cluster i and j. Since A pool corresponds to the numerator of L c in (7), the trace maximization yields clusters with many internal connections and weakly connected to each other.

Hence, A pool will be a diagonal-dominant matrix, which describes a graph with self-loops much stronger than any other connection.

Because self-loops hamper the propagation across adjacent nodes in the MP operations following the pooling layer, we compute the new adjacency matrixÃ pool by zeroing the diagonal and by applying the degree normalization

where diag(·) returns the matrix diagonal.

The proposed method is straightforward to implement: the cluster assignments, the loss, graph coarsening, and feature pooling are all computed with standard linear algebra operations.

There are several differences between minCUTpool and classic SC methods.

SC partitions the graph based on the Laplacian, but does not account for the node features.

Instead, the cluster assignments s i found by minCUTpool depend on x i , which works well if connected nodes have similar features.

This is a reasonable assumption in GNNs since, even in disassortative graphs (i.e., networks where dissimilar nodes are likely to be connected (Newman, 2003) ), the features tend to become similar due to the MP operations.

Another difference is that SC handles a single graph and is not conceived for tasks with multiple graphs to be partitioned independently.

Instead, thanks to the independence of the model parameters from the number of nodes N and from the graph spectrum, minCUTpool can generalize to outof-sample data.

This feature is fundamental in problems such as graph classification, where each sample is a graph with a different structure, and allows to train the model on small graphs and process larger ones at inference time.

Finally, minCUTpool directly uses the soft cluster assignments rather than performing k-means afterwards.

Trainable pooling methods.

Similarly to our method, these approaches learn how to generate coarsened version of the graph through differentiable functions, which take as input the nodes features X and are parametrized by weights optimized on the task at hand.

Diffpool (Ying et al., 2018 ) is a pooling module that includes two parallel MP layers: one to compute the new node features X (t+1) and another to generate the cluster assignments S. Diffpool implements an unsupervised loss that consists of two terms.

First, the link prediction term A − SS T F minimizes the Frobenius norm of the difference between the adjacency and the Gram matrix of the cluster assignments, encouraging nearby nodes to be clustered together.

The second term 1 N N i=1 H(S i ) minimizes the entropy of the cluster assignments to make them alike to one-hot vectors.

Like minCUTpool, Diffpool clusters the vertices of annotated graphs, but yields completely different partitions, since it computes differently the clustering assignments, the coarsened adjacency matrix and, most importantly, the unsupervised loss.

In Diffpool, such a loss shows pathological behaviors that are discussed later in the experiments.

The approach dubbed Top-K pooling (Hongyang Gao, 2019; Lee et al., 2019) , learns a projection vector that is applied to each node feature to obtain a score.

The nodes with the K highest scores are retained, the others are dropped.

Since the top-K selection is not differentiable, the scores are also used as a gate/attention for the node features, letting the projection vector to be trained with backpropagation.

Top-K is memory efficient as it avoids generating cluster assignments.

To prevent A from becoming disconnected after nodes removal, Top-K drops the rows and the columns from A 2 and uses it as the new adjacency matrix.

However, computing A 2 costs O(N 2 ) and it is inefficient to implement with sparse operations.

Topological pooling methods.

These methods pre-compute a pyramid of coarsened graphs, only taking into account the topology (A), but not the node features (X).

During training, the node features are pooled with standard procedures and are fit into these deterministic graph structures.

These methods are less flexible, but provide a stronger bias that can prevent degenerate solutions (e.g., coarsened graphs collapsing in a single node).

The approach proposed by Bruna et al. (2013) , which has been adopted also in other GNN architectures (Defferrard et al., 2016; Fey et al., 2018) , exploits GRACLUS (Dhillon et al., 2004), a hierarchical algorithm based on SC.

At each pooling level l, GRACLUS indetifies the pairs of maximally similar nodes i l and j l to be clustered together into a new vertex k (l+1) .

At inference phase, max-pooling is used to determine which node in the pair is kept.

Fake vertices are added so that the number of nodes can be halved each time, but this injects noisy information in the graph.

Node decimation is a method originally proposed in graph signal processing literature (Shuman et al., 2016) , which as been adapted also for GNNs (Simonovsky & Komodakis, 2017) .

The nodes are partitioned in two sets, according to the signs of the Laplacian eigenvector associated to the largest eigenvalue.

One of the two sets is dropped, reducing the number of nodes each time approximately by half.

Kron reduction is used to compute a pyramid of coarsened Laplacians from the remaining nodes.

A procedure proposed in Gama et al. (2018) diffuses a signal from designated nodes on the graph and stores the observed sequence of diffused components.

The resulting stream of information is interpreted as a time signal, where standard CNN pooling is applied.

We also mention a pooling operation for coarsening binary unweighted graphs by aggregating maximal cliques (Luzhnica et al., 2019) .

Nodes assigned to the same clique are summarized by max or average pooling and become a new node in the coarsened graph.

We consider both supervised and unsupervised tasks, and compare minCUTpool with other GNN pooling strategies.

The Appendix provides further details on the experiments and a schematic depiction of the architectures used in each task.

In addition, the Appendix reports two additional experiments: i) graph reconstruction by means of an Auto Encoder with bottleneck, implemented with pooling and un-pooling layers, ii) an architecture with pooling for graph regression.

To study the effectiveness of the proposed loss, we perform different node clustering tasks with a simple GNN composed of a single MP layer followed by a pooling layer.

The GNN is trained by minimizing L u only, so that its effect is evaluated without the "interference" of a supervised loss.

Clustering on synthetic networks We consider two simple graphs: the first is a network with 6 communities and the second is a regular grid.

The adjacency matrix A is binary and the features X are the 2-D node coordinates.

Fig. 2 depicts the node partitions generated by SC (a, d), Diffpool (b, e), and minCUTpool (c, f).

Cluster indexes for Diffpool and minCUTpool are obtained by taking the argmax of S row-wise.

Compared to SC, Diffpool and minCUTpool leverage the information contained in X. minCUTpool generates very accurate and balanced partitions, demonstrating that the cluster assignment matrix S is well formed.

On the other hand, Diffpool assigns some nodes to the wrong community in the first example, and produces an imbalanced partition of the grid.

Image segmentation Given an image, we build a Region Adjacency Graph (Trémeau & Colantoni, 2000) using as nodes the regions generated by an oversegmentation procedure (Felzenszwalb & Huttenlocher, 2004) .

The SC technique used in this example is the recursive normalized cut (Shi & Malik, 2000) , which recursively clusters the nodes until convergence.

For Diffpool and minCUTpool, we include node features consisting of the average and total color in each oversegmented region.

We set the number of desired clusters to K = 4.

The results in Fig. 3 show that minCUTpool yields a more precise segmentation.

On the other hand, SC and Diffpool aggregate wrong regions and, in addition, SC finds too many segments.

Clustering on citation networks We cluster the nodes of three popular citation networks: Cora, Citeseer, and Pubmed.

The nodes are documents represented by sparse bag-of-words feature vectors stored in X and the binary undirected edges in A indicate citation links between documents.

Each node i is labeled with the document class y i .

Once the training is over, to test the quality of the partitions generated by each method we check the agreement between the cluster assignments and the true class labels.

Tab.

1 reports the Completeness Score CS(ỹ, y) = 1 − , where H(·) is the entropy.

The GNN architecture configured with minCUTpool achieves a higher NMI score than SC, which does not account for the node features X when generating the partitions.

Our pooling operation outperforms also Diffpool, since the minimization of the unsupervised loss in Diffpool yields degenerate solutions.

The pathological behavior is shown in Fig. 4 , which depicts the evolution of the NMI scores as the unsupervised losses in Diffpool and minCUTpool are minimized in training.

In this task, the i-th datum is a graph with N i nodes represented by a pair {A i , X i } and must be associated to the correct label y i .

We test the models on different graph classification datasets.

For featureless graphs, we used the node degree information and the clustering coefficient as surrogate node features.

We evaluate model performance with a 10-fold train/test split, using 10% of the training set in each fold as validation for early stopping.

We adopt a fixed network architecture, MP(32)-pool-MP(32)-pool-MP(32)-GlobalAvgPool-softmax, where MP is the message-passing operation in (4) with 32 hidden units.

The pooling module is implemented either by Graclus, Decimation pooling, Top-K, SAGPool (Lee et al., 2019) , Diffpool, or the proposed minCUTpool.

Each pooling method is configured to drop half of the nodes in a graph (K = N/2 in Top-K, Diffpool, and minCUTpool).

As baselines, we consider the popular Weisfeiler-Lehman (WL) graph kernel (Shervashidze et al., 2011) , a network with only MP layers (Flat), and a fully connected network (Dense).

Tab.

2 reports the classification results, highlighting those that are significantly better (p-value < 0.05 w.r.t.

the method with the highest mean accuracy).

The comparison with Flat helps to understand if a pooling operation is useful or not.

The results of Dense, instead, help to quantify how much additional information is brought by the graph structure, with respect to the node features alone.

It can be seen that minCUTpool obtains always equal or better results with respect to every other GNN architecture.

On the other hand, some pooling procedures do not always improve the performance compared to the Flat baseline, making them not advisable to use in some cases.

The WL kernel generally performs worse than the GNNs, except for the Mutagenicity dataset.

This is probably because Mutagenicity has smaller graphs than the other datasets, and the adopted GNN architecture is overparametrized for this task.

Interestingly, in some dataset such as Proteins and COLLAB it is possible to obtain fairly good classification accuracy with the Dense architecture, meaning that the graph structure only adds limited information.

TopK DiffPool minCUT Graclus Decim.

Graclus and Decimation are understandably the fastest methods, since the coarsened graphs are precomputed.

Among the differentiable pooling methods, minCUTpool is faster than Diffpool, which uses a slower MP layer rather than a MLP to compute cluster assignments, and than Top-K, which computes the square of A at every forward pass.

We proposed a pooling layer for GNNs that coarsens a graph by taking into account both the the connectivity structure and the node features.

The layer optimizes a regularization term based on the minCUT objective, which is minimized in conjunction with the task-specific loss to produce node partitions that are optimal for the task at hand.

We tested the effectiveness of our pooling strategy on unsupervised node clustering tasks, by optimizing only the unsupervised clustering loss, as well as supervised graph classification tasks on several popular benchmark datasets.

Results show that minCUTpool performs significantly better than existing pooling strategies for GNNs.

To compare the amount of information retained by the pooling layers in the coarsened graphs, we train an autoencoder (AE) to reconstruct a input graph signal X from its pooled version.

The AE architecture is MP(32)-MP(32)-pool-unpool-MP(32)-MP(32)-MP, and is trained by minimizing the mean squared error between the original and the reconstructed graph signal, X − X rec 2 .

All the pooling operations are configured to retain 25% of the original nodes.

In Diffpool and minCUTpool, the unpool step is simply implemented by transposing the original pooling operations

Top-K does not generate a cluster assignment matrix, but returns a binary mask m = {0, 1}

N that indicates the nodes to drop (0) or to retain (1).

Therefore, an upsamplig matrix U is built by dropping the columns of the identity matrix I N that correspond to a 0 in m, U = [I N ] :,m==1 .

The unpooling operation is performed by replacing S with U in (9), and the resulting upscaled graph is a version of the original graph with zeroes in correspondence of the dropped nodes.

6 and 7 report the original graph signal X (the node features are the 2-D coordinates of the nodes) and the reconstruction X rec obtained by using the different pooling methods, for a ring graph and a regular grid graph.

The reconstruction produced by Diffpool is worse for the ring graph, but is almost perfect for the grid graph, while minCUTpool yields good results in both cases.

On the other hand, Top-K clearly fails in generating a coarsened representation that maintains enough information from the original graph.

This experiment highlights a major issue in Top-K pooling, which retains the nodes associated to the highest K values of a score vector s, computed by projecting the node features onto a trainable vector p: s = Xp.

Nodes that are connected on the graph usually share similar features, and their similarity further increases after the MP operations, which combine the features of neighboring nodes.

Retaining the nodes associated to the top K scores in s corresponds to keeping those nodes that are alike and highly connected, as it can be seen in Fig. 6-7 .

Therefore, Top-K discards entire portions of the graphs, which might contain important information.

This explains why Top-K fails to recover the original graph signal when used as bottleneck for the AE, and yields the worse performance among all GNN methods in the graph classification task.

The QM9 chemical database is a collection of ≈135k small organic molecules, associated to continuous labels describing several geometric, energetic, electronic, and thermodynamic properties 1 .

Each molecule in the dataset is represented as a graph {A i , X i }, where atoms are associated to nodes, and edges represent chemical bonds.

The atomic number of each atom (one-hot encoded; C, N, F, O) is taken as node feature and the type of bond (one-hot encoded; single, double, triple, aromatic) can be used as edge attribute.

In this experiment, we ignore the edge attributes in order to use all pooling algorithms without modifications.

The purpose of this experiment is to compare the trainable pooling methods also on a graph regression task, but it must be intended as a proof of concept.

In fact, the graphs in this dataset are extremely small (the average number of nodes is 8) and, therefore, a pooling operation is arguably not necessary.

We consider a GNN with architecture MP(32)-pool-MP(32)-GlobalAvgPool-Dense, where pool is implemented by Top-K, Diffpool, or minCUTpool.

The network is trained to predict a given chemical property from the input molecular graphs.

Performance is evaluated with a 10-fold cross-validation, using 10% of the training set for validation in each split.

The GNNs are trained for 50 epochs, using Adam with learning rate 5e-4, batch size 32, and ReLU activations.

We use the mean squared error (MSE) as supervised loss.

The MSE obtained on the prediction of each property for different pooling methods is reported in Tab.

3.

As expected, the flat baseline with no pooling operation (MP(32)-MP(32)-GlobalAvgPoolDense) yields a lower error in most cases.

Contrarily to the graph classification and the AE task, Top-K achieves better results than Diffpool in average.

Once again, minCUTpool significantly outperforms the other methods on each regression task and, in one case, also the flat baseline.

Table 3 : MSE on the graph regression task.

The best results with a statistical significance of p < 0.05 are highlighted: the best overall are in bold, the best among pooling methods are underlined.

For the WL kernel, we used the implementation provided in the GraKeL library 2 .

The pooling strategy based on Graclus, is taken from the ChebyNets repository 3 .

Diffpool and minCUTpool are configured with 16 hidden neurons with linear activations in the MLP and MP layer, respectively used to compute the cluster assignment matrix S. The MP layer used to compute the propagated node features X (1) uses an ELU activation in both architectures.

The learning rate for Adam is 5e-4, and the models are trained for 10000 iterations.

The details of the citation networks dataset are reported in Tab.

4.

We train the GNN architectures with Adam, an L 2 penalty loss with weight 1e-4, and 16 hidden units (H) both in the MLP of minCUTpool and in the internal MP of Diffpool.

Mutagenicity, Proteins, DD, COLLAB, and Reddit-2k are datasets representing real-world graphs and are taken from the repository of benchmark datasets for graph kernels 4 .

Bench-easy and Bench-hard 5 are datasets where the node features X and the adjacency matrix A are completely uninformative if considered alone.

Hence, algorithms that account only for the node features or the graph structure will fail to classify the graphs.

Since Bench-easy and Bench-hard come with a train/validation/test split, the 10-fold split is not necessary to evaluate the performance.

The statistics of all the datasets are reported in Tab.

5.

Fig. 8 reports the schematic representation of the minCUTpool layer; Fig. 9 the GNN architecture used in the clustering and segmentation tasks; Fig. 10 the GNN architecture used in the graph classification task; Fig. 12 the GNN architecture used in the graph regression task; Fig. 11 the graph autoencoder used in the graph signal reconstruction task.

<|TLDR|>

@highlight

A new pooling layer for GNNs that learns how to pool nodes, according to their features, the graph connectivity, and the dowstream task objective.
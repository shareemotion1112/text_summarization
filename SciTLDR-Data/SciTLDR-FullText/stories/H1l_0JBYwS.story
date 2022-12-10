Spectral embedding is a popular technique for the representation of graph data.

Several regularization techniques have been proposed to improve the quality of the embedding with respect to downstream tasks like clustering.

In this paper, we explain on a simple block model the impact of the complete graph regularization, whereby a constant is added to all entries of the adjacency matrix.

Specifically, we show that the regularization forces the spectral embedding  to focus on  the  largest blocks, making the representation less sensitive to noise or outliers.

We illustrate these results on both  on both synthetic and real data, showing how regularization improves standard clustering scores.

Spectral embedding is a standard technique for the representation of graph data (Ng et al., 2002; Belkin & Niyogi, 2002) .

Given the adjacency matrix A ∈ R n×n + of the graph, it is obtained by solving either the eigenvalue problem:

or the generalized eigenvalue problem:

where D = diag(A1 n ) is the degree matrix, with 1 n the all-ones vector of dimension n, L = D − A is the Laplacian matrix of the graph, Λ ∈ R k×k is the diagonal matrix of the k smallest (generalized) eigenvalues of L and X ∈ R n×k is the corresponding matrix of (generalized) eigenvectors.

In this paper, we only consider the generalized eigenvalue problem, whose solution is given by the spectral decomposition of the normalized Laplacian matrix L norm = I − D −1/2 AD −1/2 (Luxburg, 2007) .

The spectral embedding can be interpreted as equilibrium states of some physical systems (Snell & Doyle, 2000; Spielman, 2007; Bonald et al., 2018) , a desirable property in modern machine learning.

However, it tends to produce poor results on real datasets if applied directly on the graph (Amini et al., 2013) .

One reason is that real graphs are most often disconnected due to noise or outliers in the dataset.

In order to improve the quality of the embedding, two main types of regularization have been proposed.

The first artificially increases the degree of each node by a constant factor (Chaudhuri et al., 2012; Qin & Rohe, 2013) , while the second adds a constant to all entries of the original adjacency matrix (Amini et al., 2013; Joseph et al., 2016; Zhang & Rohe, 2018) .

In the practically interesting case where the original adjacency matrix A is sparse, the regularized adjacency matrix is dense but has a so-called sparse + low rank structure, enabling the computation of the spectral embedding on very large graphs (Lara, 2019) .

While (Zhang & Rohe, 2018) explains the effects of regularization through graph conductance and (Joseph et al., 2016) through eigenvector perturbation on the Stochastic Block Model, there is no simple interpretation of the benefits of graph regularization.

In this paper, we show on a simple block model that the complete graph regularization forces the spectral embedding to separate the blocks in decreasing order of size, making the embedding less sensitive to noise or outliers in the data.

Indeed, (Zhang & Rohe, 2018) identified that, without regularization, the cuts corresponding to the first dimensions of the spectral embedding tend to separate small sets of nodes, so-called dangling sets, loosely connected to the rest of the graph.

Our work shows more explicitly that regularization forces the spectral embedding to focus on the largest clusters.

Moreover, our analysis involves some explicit characterization of the eigenvalues, allowing us to quantify the impact of the regularization parameter.

The rest of this paper is organized as follows.

Section 2 presents block models and an important preliminary result about their aggregation.

Section 3 presents the main result of the paper, about the regularization of block models, while Section 4 extends this result to bipartite graphs.

Section 5 presents the experiments and Section 6 concludes the paper.

Let A ∈ R n×n + be the adjacency matrix of an undirected, weight graph, that is a symmetric matrix such that A ij > 0 if and only if there is an edge between nodes i and j, with weight A ij .

Assume that the n nodes of the graph can be partitioned into K blocks of respective sizes n 1 , . . .

, n K so that any two nodes of the same block have the same neighborhood, i.e., the corresponding rows (or columns) of A are the same.

Without any loss of generality, we assume that the matrix A has rank K. We refer to such a graph as a block model.

Let Z ∈ R n×K be the associated membership matrix, with Z ij = 1 if index i belongs to block j and 0 otherwise.

We denote by W = Z T Z ∈ R K×K the diagonal matrix of block sizes.

.

This is the adjacency matrix of the aggregate graph, where each block of the initial graph is replaced by a single node; two nodes in this graph are connected by an edge of weight equal to the total weight of edges between the corresponding blocks in the original graph.

We denote byD = diag(Ā1 K ) the degree matrix and byL =D −Ā the Laplacian matrix of the aggregate graph.

The following result shows that the solution to the generalized eigenvalue problem (2) follows from that of the aggregate graph: Proposition 1.

Let x be a solution to the generalized eigenvalue problem:

Then either Z T x = 0 and λ = 1 or x = Zy where y is a solution to the generalized eigenvalue problem:L y = λDy.

Proof.

Consider the following reformulation of the generalized eigenvalue problem (3):

Since the rank of A is equal to K, there are n − K eigenvectors x associated with the eigenvalue λ = 1, each satisfying Z T x = 0.

By orthogonality, the other eigenvectors satisfy x = Zy for some vector y ∈ R K .

We get:

Thus y is a solution to the generalized eigenvalue problem (4).

Let A be the adjacency matrix of some undirected graph.

We consider a regularized version of the graph where an edge of weight α is added between all pairs of nodes, for some constant α > 0.

The corresponding adjacency matrix is given by:

where J = 1 n 1 T n is the all-ones matrix of same dimension as A. We denote by D α = diag(A α 1 n ) the corresponding degree matrix and by L α = D α − A α the Laplacian matrix.

We first consider a simple block model where the graph consists of K disjoint cliques of respective sizes n 1 > n 2 > · · · > n K nodes, with n K ≥ 1.

In this case, we have A = ZZ T , where Z is the membership matrix.

The objective of this section is to demonstrate that, in this setting, the k-th dimension of the spectral embedding isolates the k − 1 largest cliques from the rest of the graph, for any k ∈ {2, . . .

, K} Lemma 1. Let λ 1 ≤ λ 2 ≤ . . .

≤ λ n be the eigenvalues associated with the generalized eigenvalue problem:

Proof.

Since the Laplacian matrix L α is positive semi-definite, all eigenvalues are non-negative (Chung, 1997) .

We know that the eigenvalue 0 has multiplicity 1 on observing that the regularized graph is connected.

Now for any vector x,

so that the matrix A α is positive semi-definite.

In view of (5), this shows that λ ≤ 1 for any eigenvalue λ.

The proof then follows from Proposition 1, on observing that the eigenvalue 1 has multiplicity n − K.

Lemma 2.

Let x be a solution to the generalized eigenvalue problem (6) with λ ∈ (0, 1).

There exists some s ∈ {+1, −1} such that for each node i in block j,

Proof.

In view of Proposition 1, we have x = Zy where y is a solution to the generalized eigenvalue problem of the aggregate graph, with adjacency matrix:

where I K is the identity matrix of dimension K × K. We deduce the degree matrix:

and the Laplacian matrix:L

The generalized eigenvalue problem associated with the aggregate graph is:

After multiplication by W −1 , we get:

Observing that

and since W = diag(n 1 , . . . , n K ),

The result then follows from the fact that x = Zy.

Lemma 3.

The K smallest eigenvalues satisfy:

where for all j = 1, . . .

, K, µ j = αn αn + n j .

Proof.

We know from Lemma 1 that the K smallest eigenvalues are in [0, 1).

Let x be a solution to the generalized eigenvalue problem (6) with λ ∈ (0, 1).

We know that x = Zy where y is an eigenvector associated with the same eigenvalue λ for the aggregate graph.

Since 1 K is an eigenvector for the eigenvalue 0, we have y

We then deduce from (7) and (8)

This condition cannot be satisfied if λ < µ 1 or λ > µ K as the terms of the sum would be either all positive or all negative.

Now let y be another eigenvector for the aggregate graph, with y TD α y = 0, for the eigenvalue λ ∈ (0, 1).

By the same argument, we get:

with λ ∈ {µ 1 , . . .

, µ K }.

This condition cannot be satisfied if λ and λ are in the same interval (µ j , µ j+1 ) for some j as the terms in the sum would be all positive.

There are K − 1 eigenvalues in (0, 1) for K − 1 such intervals, that is one eigenvalue per interval.

The main result of the paper is the following, showing that the k − 1 largest cliques of the original graph can be recovered from the spectral embedding of the regularized graph in dimension k. Theorem 1.

Let X be the spectral embedding of dimension k, as defined by (2), for some k in the set {2, . . .

, K}. Then sign(X) gives the k − 1 largest blocks of the graph.

Proof.

Let x be the j-th column of the matrix X, for some j ∈ {2, . . .

, k}. In view of Lemma 3, this is the eigenvector associated with eigenvalue λ j ∈ (µ j−1 , µ j ), so that

In view of Lemma 2, all entries of x corresponding to blocks of size n 1 , n 2 . . . , n j−1 have the same sign, the other having the opposite sign.

Theorem 1 can be extended in several ways.

First, the assumption of distinct block sizes can easily be relaxed.

If there are L distinct values of block sizes, say m 1 , . . .

, m L blocks of sizes n 1 > . . .

> n L , there are L distinct values for the thresholds µ j and thus L distinct values for the eigenvalues λ j in [0, 1), the multiplicity of the j-th smallest eigenvalue being equal to m j .

The spectral embedding in dimension k still gives k − 1 cliques of the largest sizes.

Second, the graph may have edges between blocks.

Taking A = ZZ T + εJ for instance, for some parameter ε ≥ 0, the results are exactly the same, with α replaced by +α.

A key observation is that regularization really matters when ε → 0, in which case the initial graph becomes disconnected and, in the absence of regularization, the spectral embedding may isolate small connected components of the graph.

In particular, the regularization makes the spectral embedding much less sensitive to noise, as will be demonstrated in the experiments.

Finally, degree correction can be added by varying the node degrees within blocks.

Taking A = θZZ T θ, for some arbitrary diagonal matrix θ with positive entries, similar results can be obtained under the regularization A α = A + αθJθ.

Interestingly, the spectral embedding in dimension k then recovers the k − 1 largest blocks in terms of normalized weight, the ratio of the total weight of the block to the number of nodes in the block.

Let B = R n×m + be the biadjacency matrix of some bipartite graph with respectively n, m nodes in each part, i.e., B ij > 0 if and only if there is an edge between node i in the first part of the graph and node j in the second part of the graph, with weight B ij .

This is an undirected graph of n + m nodes with adjacency matrix:

The spectral embedding of the graph (2) can be written in terms of the biadjacency matrix as follows:

where X 1 , X 2 are the embeddings of each part of the graph, with respective dimensions n × k and

In particular, the spectral embedding of the graph follows from the generalized SVD of the biadjacency matrix B.

The complete regularization adds edges between all pairs of nodes, breaking the bipartite structure of the graph.

Another approach consists in applying the regularization to the biadjacency matrix, i.e., in considering the regularized bipartite graph with biadjacency matrix:

B α = B + αJ, where J = 1 n 1 T m is here the all-ones matrix of same dimension as B. The spectral embedding of the regularized graph is that associated with the adjacency matrix:

As in Section 3, we consider a block model so that the biadjacency matrix B is block-diagonal with all-ones block matrices on the diagonal.

Each part of the graph consists of K groups of nodes of respective sizes n 1 > . . .

> n K and m 1 > . . .

> m K , with nodes of block j in the first part connected only to nodes of block j in the second part, for all j = 1, . . .

, K.

We consider the generalized eigenvalue problem (6) associated with the above matrix A α .

In view of (9), this is equivalent to the generalized SVD of the regularized biadjacency matrix B α .

We have the following results, whose proofs are deferred to the appendix: Lemma 4.

Let λ 1 ≤ λ 2 ≤ . . .

≤ λ n be the eigenvalues associated with the generalized eigenvalue problem (6).

We have λ 1 = 0 < λ 2 ≤ . . .

≤ λ K < λ K+1 = . . .

= λ n−2K < . . .

< λ n = 2.

Lemma 5.

Let x be a solution to the generalized eigenvalue problem (6) with λ ∈ (0, 1).

There exists s 1 , s 2 ∈ {+1, −1} such that for each node i in block j of part p ∈ {1, 2},

Lemma 6.

The K smallest eigenvalues satisfy:

Published as a conference paper at ICLR 2020

Theorem 2.

Let X be the spectral embedding of dimension k, as defined by (2), for some k in the set {2, . . .

, K}. Then sign(X) gives the k − 1 largest blocks of each part of the graph.

Like Theorem 1, the assumption of decreasing block sizes can easily be relaxed.

Assume that block pairs are indexed in decreasing order of µ j .

Then the spectral embedding of dimension k gives the k − 1 first block pairs for that order.

It is interesting to notice that the order now depends on α: when α → 0 + , the block pairs j of highest value ( n nj + m mj ) −1 (equivalently, highest harmonic mean of proportions of nodes in each part of the graph) are isolated first; when α → +∞, the block pairs j of highest value nj mj nm (equivalently, the highest geometric mean of proportions of nodes in each part of the graph) are isolated first.

The results also extend to non-block diagonal biadjacency matrices B and degree-corrected models, as for Theorem 1.

We now illustrate the impact of regularization on the quality of spectral embedding.

We focus on a clustering task, using both synthetic and real datasets where the ground-truth clusters are known.

In all experiments, we skip the first dimension of the spectral embedding as it is not informative (the corresponding eigenvector is the all-ones vector, up to some multiplicative constant).

The code to reproduce these experiments is available online 1 .

We first illustrate the theoretical results of the paper with a toy graph consisting of 3 cliques of respective sizes 5, 3, 2.

We compute the spectral embeddings in dimension 1, using the second smallest eigenvalue.

Denoting by Z the membership matrix, we get X ≈ Z(−0.08, 0.11, 0.05) T for α = 1, showing that the embedding isolates the largest cluster; this is not the case in the absence of regularization, where X ≈ Z(0.1, −0.1, 0.41) T .

This section describes the datasets used in our experiments.

All graphs are considered as undirected.

Table 1 presents the main features of the graphs.

Stochastic Block-Model (SBM) We generate 100 instances of the same stochastic block model (Holland et al., 1983) .

There are 100 blocks of size 20, with intra-block edge probability set to 0.5 for the first 50 blocks and 0.05 for the other blocks.

The inter-block edge probability is set to 0.001 Other sets of parameters can be tested using the code available online.

The ground-truth cluster of each node corresponds to its block.

This dataset consists of around 18000 newsgroups posts on 20 topics.

This defines a weighted bipartite graph between documents and words.

The label of each document corresponds to the topic. (Haruechaiyasak & Damrongrat, 2008) .

This is the graph of hyperlinks between a subset of Wikipedia pages.

The label of each page is its category (e.g., countries, mammals, physics).

We consider a large set of metrics from the clustering literature.

All metrics are upper-bounded by 1 and the higher the score the better.

Homogeneity (H), Completeness (C) and V-measure score (V) (Rosenberg & Hirschberg, 2007) .

Supervised metrics.

A cluster is homogeneous if all its data points are members of a single class in the ground truth.

A clustering is complete if all the members of a class in the ground truth belong to the same cluster in the prediction.

Harmonic mean of homogeneity and completeness.

Adjusted Rand Index (ARI) (Hubert & Arabie, 1985) .

Supervised metric.

This is the corrected for chance version of the Rand Index which is itself an accuracy on pairs of samples.

Adjusted Mutual Information (AMI) (Vinh et al., 2010) Supervised metric.

Adjusted for chance version of the mutual information.

Fowlkes-Mallows Index (FMI) (Fowlkes & Mallows, 1983) .

Supervised metric.

Geometric mean between precision and recall on the edge classification task, as described for the ARI.

Modularity (Q) (Newman, 2006) .

Unsupervised metric.

Fraction of edges within clusters compared to that is some null model where edges are shuffled at random.

Normalized Standard Deviation (NSD) Unsupervised metric.

1 minus normalized standard deviation in cluster size.

All graphs are embedded in dimension 20, with different regularization parameters.

To compare the impact of this parameter across different datasets, we use a relative regularization parameter (w/n 2 )α, where w = 1 T n A1 n is the total weight of the graph.

We use the K-Means algorithm with to cluster the nodes in the embedding space.

The parameter K is set to the ground-truth number of clusters (other experiments with different values of K are reported in the Appendix).

We use the Scikit-learn (Pedregosa et al., 2011) implementation of KMeans and the metrics, when available.

The spectral embedding and the modularity are computed with the Scikit-network package, see the documentation for more details 2 .

We report the results in Table 2 for relative regularization parameter α = 0, 0.1, 1, 10.

We see that the regularization generally improves performance, the optimal value of α depending on both the dataset and the score function.

As suggested by Lemma 3, the optimal value of the regularization parameter should depend on the distribution of cluster sizes, on which we do not have any prior knowledge.

To test the impact of noise on the spectral embedding, we add isolated nodes with self loop to the graph and compare the clustering performance with and without regularization.

The number of isolated nodes is given as a fraction of the initial number of nodes in the graph.

Scores are computed only on the initial nodes.

The results are reported in Table 3 for the Wikipedia for Schools dataset.

We observe that, in the absence of regularization, the scores drop even with only 1% noise.

The computed clustering is a trivial partition with all initial nodes in the same cluster.

This means that the 20 first dimensions of the spectral embedding focus on the isolated nodes.

On the other hand, the scores remain approximately constant in the regularized case, which suggests that regularization makes the embedding robust to this type of noise.

In this paper, we have provided a simple explanation for the well-known benefits of regularization on spectral embedding.

Specifically, regularization forces the embedding to focus on the largest clusters, making the embedding more robust to noise.

This result was obtained through the explicit characterization of the embedding for a simple block model, and extended to bipartite graphs.

An interesting perspective of our work is the extension to stochastic block models, using for instance the concentration results proved in (Lei et al., 2015; Le et al., 2017) .

Another problem of interest is the impact of regularization on other downstream tasks, like link prediction.

Finally, we would like to further explore the impact of the regularization parameter, exploiting the theoretical results presented in this paper.

We provide of proof of Theorem 2 as well as a complete set of experimental results.

The proof of Theorem 2 follows the same workflow as that of Theorem 1.

Let Z 1 ∈ R n×K and Z 2 ∈ R m×K be the left and right membership matrices for the block matrix B ∈ R n×m .

The aggregated matrix isB = Z T 1 BZ 2 ∈ R K×K .

The diagonal matrices of block sizes are

We have the equivalent of Proposition 1: Proposition 2.

Let x 1 , x 2 be a solution to the generalized singular value problem:

x 2 = 0 and σ = 0 or x 1 = Z 1 y 1 and x 2 = Z 2 y 2 where y 1 , y 2 is a solution to the generalized singular value problem:

B y 2 = σD 1 y 1 , B T y 1 = σD 2 y 2 .

Proof.

Since the rank of B is equal to K, there are n−K pairs of singular vectors (x 1 , x 2 ) associated with the singular values 0, each satisfying Z T 1 x 1 = 0 and Z T 2 x 2 = 0.

By orthogonality, the other pairs of singular vectors satisfy x 1 = Z 1 y 1 and x 2 = Z 2 y 2 for some vectors y 1 , y 2 ∈ R K .

By replacing these in the original generalized singular value problem, we get that (y 1 , y 2 ) is a solution to the generalized singular value problem for the aggregate graph.

In the following, we focus on the block model described in Section 4, where

Proof of Lemma 4.

The generalized eigenvalue problem (6) associated with the regularized matrix A α is equivalent to the generalized SVD of the regularized biadjacency matrix B α :

In view of Proposition 2, the singular value σ = 0 has multiplicity n − K, meaning that the eigenvalue λ = 1 has multiplicity n − K. Since the graph is connected, the eigenvalue 0 has multiplicity 1.

The proof then follows from the observation that if (x 1 , x 2 ) is a pair of singular vectors for the singular value σ, then the vectors x = (x 1 , ±x 2 )

T are eigenvectors for the eigenvalues 1 − σ, 1 + σ.

Proof of Lemma 5.

By Proposition 2, we can focus on the generalized singular value problem for the aggregate graph:

we have:

Observing that J K W 1 y 1 ∝ 1 K and J K W 2 y 2 ∝ 1 K , we get:

As two diagonal matrices commute, we obtain:

for some constants η 1 , η 2 , and

Letting s 1 = −sign(η 1 (n j + αn) + η 2 m j ) and s 2 = −sign(η 1 n j + η 2 (m j + αm)), we get:

and the result follows from the fact that x 1 = Z 1 y 1 and x 2 = Z 2 y 2 .

Proof of Lemma 6.

The proof is the same as that of Lemma 3, where the threshold values follow from Lemma 5:

Proof of Theorem 2.

Let x be the j-th column of the matrix X, for some j ∈ {2, . . . , k}. In view of Lemma 6, this is the eigenvector associated with eigenvalue λ j ∈ (µ j−1 , µ j ).

In view of Lemma 4, all entries of x corresponding to blocks of size n 1 , n 2 . . .

, n j−1 have the same sign, the other having the opposite sign.

In this section, we present more extensive experimental results.

Tables 4 and 5 present results for the same experiment as in Table 2 but for different values of K, namely K = 2 (bisection of the graph) and K = K truth /2 (half of the ground-truth value).

As for K = K true , regularization generally improves clustering performance.

However, the optimal value of α remains both dataset dependent and metric dependent.

Note that, for the NG and WS datasets, the clustering remains trivial in the case K = 2, one cluster containing all the nodes, until a certain amount of regularization.

Table 6 presents the different scores for both types of regularization on the NG dataset.

As we can see, preserving the bipartite structure of the graph leads to slightly better performance.

Finally, Table 7 shows the impact of regularization in the presence of noise for the NG dataset.

The conclusions are similar as for the WS dataset: regularization makes the spectral embedding much more robust to noise.

@highlight

Graph regularization forces spectral embedding to focus on the largest clusters, making the representation less sensitive to noise. 
Learning distributed representations for nodes in graphs is a crucial primitive in network analysis with a wide spectrum of applications.

Linear graph embedding methods learn such representations by optimizing the likelihood of both positive and negative edges while constraining the dimension of the embedding vectors.

We argue that the generalization performance of these methods is not due to the dimensionality constraint as commonly believed, but rather the small norm of embedding vectors.

Both theoretical and empirical evidence are provided to support this argument: (a) we prove that the generalization error of these methods can be bounded by limiting the norm of vectors, regardless of the embedding dimension; (b) we show that the generalization performance of linear graph embedding methods is correlated with the norm of embedding vectors, which is small due to the early stopping of SGD and the vanishing gradients.

We performed extensive experiments to validate our analysis and showcased the importance of proper norm regularization in practice.

Graphs have long been considered as one of the most fundamental structures that can naturally represent interactions between numerous real-life objects (e.g., the Web, social networks, proteinprotein interaction networks).

Graph embedding, whose goal is to learn distributed representations for nodes while preserving the structure of the given graph, is a fundamental problem in network analysis that underpins many applications.

A handful of graph embedding techniques have been proposed in recent years BID10 BID15 BID2 , along with impressive results in applications like link prediction, text classification BID14 , and gene function prediction BID18 .Linear graph embedding methods preserve graph structures by converting the inner products of the node embeddings into probability distributions with a softmax function BID10 BID15 BID2 .

Since the exact softmax objective is computationally expensive to optimize, the negative sampling technique BID8 is often used in these methods: instead of optimizing the softmax objective function, we try to maximize the probability of positive instances while minimizing the probability of some randomly sampled negative instances.

It has been shown that by using this negative sampling technique, these graph embedding methods are essentially computing a factorization of the adjacency (or proximity) matrix of graph BID7 .

Hence, it is commonly believed that the key to the generalization performance of these methods is the dimensionality constraint.

However, in this paper we argue that the key factor to the good generalization of these embedding methods is not the dimensionality constraint, but rather the small norm of embedding vectors.

We provide both theoretical and empirical evidence to support this argument:??? Theoretically, we analyze the generalization error of two linear graph embedding hypothesis spaces (restricting embedding dimension/norm), and show that only the norm-restricted hypothesis class can theoretically guarantee good generalization in typical parameter settings.??? Empirically, we show that the success of existing linear graph embedding methods BID10 BID15 BID2 are due to the early stopping of stochastic gradient descent (SGD), which implicitly restricts the norm of embedding vectors.

Furthermore, with prolonged SGD execution and no proper norm regularization, the embedding vectors can severely overfit the training data.

The rest of this paper is organized as follows.

In Section 2, we review the definition of graph embedding problem and the general framework of linear graph embedding.

In Section 3, we present both theoretical and empirical evidence to support our argument that the generalization of embedding vectors is determined by their norm.

In Section 4, we present additional experimental results for a hinge-loss linear graph embedding variant, which further support our argument.

In Section 5, we discuss the new insights that we gained from previous results.

Finally in Section 6, we conclude our paper.

Details of the experiment settings, algorithm pseudo-codes, theorem proofs and the discussion of other related work can all be found in the appendix.

We consider a graph G = (V, E), where V is the set of nodes in G, and E is the set of edges between the nodes in V .

For any two nodes u, v ??? V , an edge (u, v) ??? E if u and v are connected, and we assume all edges are unweighted and undirected for simplicity 1 .

The task of graph embedding is to learn a D-dimensional vector representation x u for each node u ??? V such that the structure of G can be maximally preserved.

These embedding vectors can then be used as features for subsequent applications (e.g., node label classification or link prediction).

Linear graph embedding BID15 BID2 ) is one of the two major approaches for computing graph embeddings 2 .

These methods use the inner products of embedding vectors to capture the likelihood of edge existence, and are appealing to practitioners due to their simplicity and good empirical performance.

Formally, given a node u and its neighborhood N + (u) 3 , the probability of observing node v being a neighbor of u is defined as: DISPLAYFORM0 By minimizing the KL-divergence between the embedding-based distribution and the actual neighborhood distribution, the overall objective function is equivalent to: DISPLAYFORM1 Unfortunately, it is quite problematic to optimize this objective function directly, as the softmax term involves normalizing over all vertices.

To address this issue, the negative sampling BID8 technique is used to avoid computing gradients over the full softmax function.

Intuitively, the negative sampling technique can be viewed as randomly selecting a set of nodes N ??? (u) that are not connected to each node u as its negative neighbors.

The embedding vectors are then learned by minimizing the following objective function instead:1 All linear graph embedding methods discussed in this paper can be generalized to weighted case by multiplying the weight to the corresponding loss function of each edge.

The directed case is usually handled by associating each node with two embedding vectors for incoming and outgoing edges respectively, which is equivalent as learning embedding on a transformed undirected bipartite graph.2 The other major approach is to use deep neural network structure to compute the embedding vectors, see the discussion of other related works in the appendix for details.3 Note that N+(u) can be either the set of direct neighbors in the original graph G BID15 , or an expanded neighborhood based on measures like random walk BID2 .

DISPLAYFORM2 where ??(x) = 1/(1 + e ???x ) is the standard logistic function.

Although the embedding vectors learned through negative sampling do have good empirical performance, there is very few theoretical analysis of such technique that explains the good empirical performance.

The most well-known analysis of negative sampling was done by BID7 , which claims that the embedding vectors are approximating a low-rank factorization of the PMI (Pointwise Mutual Information) matrix.

More specifically, the key discovery of BID7 is that when the embedding dimension is large enough, the optimal solution to Eqn (1) recovers exactly the PMI matrix (up to a shifted constant, assuming the asymptotic case where DISPLAYFORM0 Based on this result, BID7 suggest that optimizing Eqn (1) under the dimensionality constraint is equivalent as computing a low-rank factorization of the shifted PMI matrix.

This is currently the mainstream opinion regarding the intuition behind negative sampling.

Although Levy and Goldberg only analyzed negative sampling in the context of word embedding, it is commonly believed that the same conclusion also holds for graph embedding BID11 .

As explained in Section 2.3, it is commonly believed that linear graph embedding methods are approximating a low-rank factorization of PMI matrices.

As such, people often deem the dimensionality constraint of embedding vectors as the key factor to good generalization BID15 BID2 .

However, due to the sparsity of real-world networks, the explanation of Levy & Goldberg is actually very counter-intuitive in the graph embedding setting: the average node degree usually only ranges from 10 to 100, which is much less than the typical value of embedding dimension (usually in the range of 100 ??? 400).

Essentially, this means that in the context of graph embedding, the total number of free parameters is larger than the total number of training data points, which makes it intuitively very unlikely that the negative sampling model (i.e., Eqn (1)) can inherently guarantee the generalization of embedding vectors in such scenario, and it is much more plausible if the observed good empirical performance is due to some other reason.

In this paper, we provide a different explanation to the good empirical performance of linear graph embedding methods: we argue that the good generalization of linear graph embedding vectors is due to their small norm, which is in turn caused by the vanishing gradients during the stochastic gradient descent (SGD) optimization procedure.

We provide the following evidence to support this argument:??? In Section 3.1, we theoretically analyze the generalization error of two linear graph embedding variants: one has the standard dimensionality constraints, while the other restricts the vector norms.

Our analysis shows that: -The embedding vectors can generalize well to unseen data if their average squared l 2 norm is small, and this is always true regardless of the embedding dimension choice.

-Without norm regularization, the embedding vectors can severely overfit the training data if the embedding dimension is larger than the average node degree.??? In Section 3.2, we provide empirical evidence that the generalization of linear graph embedding is determined by vector norm instead of embedding dimension.

We show that: -In practice, the average norm of the embedding vectors is small due to the early stopping of SGD and the vanishing gradients.

-The generalization performance of embedding vectors starts to drop when the average norm of embedding vectors gets large.

-The dimensionality constraint is only helpful when the embedding dimension is very small (around 5 ??? 10) and there is no norm regularization.

In this section, we present a generalization error analysis of linear graph embedding based on the uniform convergence framework BID0 , which bounds the maximum difference between the training and generalization error over the entire hypothesis space.

We assume the following statistical model for graph generation: there exists an unknown probability distribution Q over the Cartesian product V ?? U of two vertex sets V and U .

Each sample (a, b) from Q denotes an edge connecting a ??? V and b ??? U .The set of (positive) training edges E + consists of the first m i.i.d.

samples from the distribution Q, and the negative edge set E ??? consists of i.i.d.

samples from the uniform distribution U over V ?? U .

The goal is to use these samples to learn a model that generalizes well to the underlying distribution Q. We allow either V = U for homogeneous graphs or V ??? U = ??? for bipartite graphs.

DISPLAYFORM0 to be the collection of all training data, and we assume that data points in E ?? are actually sampled from a combined distribution P over V ?? U ?? {??1} that generates both positive and negative edges.

Using the above notations, the training error L t (x) and generalization error L g (x) of embedding x : (U ??? V ) ??? R D are defined as follows: DISPLAYFORM1 In the uniform convergence framework, we try to prove the following statement: DISPLAYFORM2 over all possible embeddings x in the hypothesis space H. If the above uniform convergence statement is true, then minimizing the training error L t (x) would naturally lead to small generalization error L g (x) with high probability.

Now we present our first technical result, which follows the above framework and bounds the generalization error of linear graph embedding methods with norm constraints: DISPLAYFORM3 be the embedding for nodes in the graph.

Then for any bounded 1-Lipschitz loss function l : R ??? [0, B] and C U , C V > 0, with probability 1 ??? ?? (over the sampling of E ?? ), the following inequality holds DISPLAYFORM4 where ||A ?? || 2 is the spectral norm of the randomized adjacency matrix A ?? defined as follows: DISPLAYFORM5 The proof can be found in the appendix.

Intuitively, Theeorem 1 states that with sufficient norm regularization, linear graph embedding can generalize well regardless of the embedding dimension (note that term D does not appear in Eqn (2) at all).

Theorem 1 also characterizes the importance of choosing proper regularization in lnorm restricted inear graph embedding: in Eqn (2), both the training error term DISPLAYFORM6 are dependent on the value of C U and C V .

With larger C values (i.e., weak norm regularization), the training error would be smaller due to the less restrictive hypothesis space, but the gap term would larger, meaning that the model will likely overfit the training data.

Meanwhile, smaller C values (i.e., strong norm regularization) would lead to more restrictive models, which will not overfit but have larger training error as trade-off.

Therefore, choosing the most proper norm regularization is the key to achieving optimal generalization performance.

A rough estimate of E ?? ||A ?? || 2 can be found in the appendix for interested readers.

On the other hand, if we restrict only the embedding dimension (i.e., no norm regularization on embedding vectors), and the embedding dimension is larger than the average degree of the graph, then it is possible for the embedding vectors to severely overfit the training data.

The following example demonstrates this possibility on a d-regular graph, in which the embedding vectors can always achieve zero training error even when the edge labels are randomly placed: Claim 1.

Let G = (V, E) be a d-regular graph with n vertices and m = nd/2 labeled edges (with labels y i ??? {??1}): DISPLAYFORM7 DISPLAYFORM8 The proof can be found in the appendix.

In other words, without norm regularization, the number of training samples required for learning D-dimensional embedding vectors is at least ???(nD).Considering the fact that many large-scale graphs are sparse (with average degree < 20) and the default embedding dimension commonly ranges from 100 to 400, it is highly unlikely that the the dimensionality constraint by itself could lead to good generalization performance.

In this section, we present several sets of experimental results for the standard linear graph embedding, which collectively suggest that the generalization of these methods are actually determined by vector norm instead of embedding dimension.

Experiment Setting: We use stochastic gradient descent (SGD) to minimize the following objective: DISPLAYFORM0 Here E + is the set of edges in the training graph, and E ??? is the set of negative edges with both ends sampled uniformly from all vertices.

The SGD learning rate is standard: ?? t = (t + c) ???1/2 .

Three different datasets are used in the experiments: Tweet, BlogCatalog and YouTube, and their details can be found in the appendix.

The default embedding dimension is D = 100 for all experiments unless stated otherwise.

FIG2 shows the average l 2 norm of the embedding vectors during the first 50 SGD epochs (with varying value of ?? r ).

As we can see, the average norm of embedding vectors increases consistently after each epoch, but the increase rate gets slower as time progresses.

In practice, the SGD procedure is often stopped after 10 ??? 50 epochs (especially for large scale graphs with millions of vertices 4 ), and the relatively early stopping time would naturally result in small vector norm.

The Vanishing Gradients: FIG3 shows the average l 2 norm of the stochastic gradients ???L/???x u during the first 50 SGD epochs: DISPLAYFORM1 From the figure, we can see that the stochastic gradients become smaller during the later stage of SGD, which is consistent with our earlier observation in FIG2 .

This phenomenon can be intuitively explained as follows: after a few SGD epochs, most of the training data points have already been well fitted by the embedding vectors, which means that most of the coefficients ??(??x T u x v ) in Eqn (3) will be close to 0 afterwards, and as a result the stochastic gradients will be small in the following epochs.

FIG4 shows the generalization performance of embedding vectors during the first 50 SGD epochs, in which we depicts the resulting average precision (AP) score 5 for link prediction and F1 score for node label classification.

As we can see, the generalization performance of embedding vectors starts to drop after 5 ??? 20 epochs when ?? r is small, indicating that they are overfitting the training dataset afterwards.

The generalization performance is worst near the end of SGD execution when ?? r = 0, which coincides with the fact that embedding vectors in that case also have the largest norm among all settings.

Thus, FIG4 and FIG2 collectively suggest that the generalization of linear graph embedding is determined by vector norm.

FIG6 shows the generalization AP score on Tweet dataset with varying value of ?? r and embedding dimension D after 50 epochs.

As we can see in FIG6 , without any norm regularization (?? r = 0), the embedding vectors will overfit the training dataset for any D greater than 10, which is consistent with our analysis in Claim 1.

On the other hand, with larger ?? r , the impact of embedding dimension choice is significantly less noticeable, indicating that the primary factor for generalization is the vector norm in such scenarios.

5 Average Precision (AP) evaluates the performance on ranking problems: we first compute the precision and recall value at every position in the ranked sequence, and then view the precision p(r) as a function of recall r. The average precision is then computed as AveP =

In this section, we present the experimental results for a non-standard linear graph embedding formulation, which optimizes the following objective: DISPLAYFORM0 By replacing logistic loss with hinge-loss, it is now possible to apply the dual coordinate descent (DCD) method BID4 for optimization, which circumvents the issue of vanishing gradients in SGD, allowing us to directly observe the impact of norm regularization.

More specifically, consider all terms in Eqn (4) that are relevant to a particular vertex u: DISPLAYFORM1 in which we defined DISPLAYFORM2 Since Eqn (5) takes the same form as a soft-margin linear SVM objective, with x u being the linear coefficients and (x i , y i ) being training data, it allows us to use any SVM solver to optimize Eqn (5), and then apply it asynchronously on the graph vertices to update their embeddings.

The pseudo-code for the optimization procedure using DCD can be found in the appendix.

Impact of Regularization Coefficient: FIG8 shows the generalization performance of embedding vectors obtained from DCD procedure (??? 20 epochs).

As we can see, the quality of embeddings vectors is very bad when ?? r ??? 0, indicating that proper norm regularization is necessary for generalization.

The value of ?? r also affects the gap between training and testing performance, which is consistent with our analysis that ?? r controls the model capacity of linear graph embedding.

The choice of embedding dimension D on the other hand is not very impactful as demonstrated in FIG9 : as long as D is reasonably large (??? 30), the exact choice has very little effect on the generalization performance.

Even with extremely large embedding dimension setting (D = 1600).

These results are consistent with our theory that the generalization of linear graph embedding is primarily determined by the norm constraints.

So far, we have seen many pieces of evidence supporting our argument, suggesting that the generalization of embedding vectors in linear graph embedding is determined by the vector norm.

Intuitively, it means that these embedding methods are trying to embed the vertices onto a small sphere centered around the origin point.

The radius of the sphere controls the model capacity, and choosing proper embedding dimension allows us to control the trade-off between the expressive power of the model and the computation efficiency.

Note that the connection between norm regularization and generalization performance is actually very intuitive.

To see this, let us consider the semantic meaning of embedding vectors: the probability of any particular edge (u, v) being positive is equal to DISPLAYFORM0 As we can see, this probability value is determined by three factors: DISPLAYFORM1 , the cosine similarity between x u and x v , evaluates the degree of agreement between the directions of x u and x v .???

||x u || 2 and ||x v || 2 on the other hand, reflects the degree of confidence we have regarding the embedding vectors of u and v.

Therefore, by restricting the norm of embedding vectors, we are limiting the confidence level that we have regarding the embedding vectors, which is indeed intuitively helpful for preventing overfitting.

It is worth noting that our results in this paper do not invalidate the analysis of BID7 , but rather clarifies on some key points: as pointed out by BID7 , linear graph embedding methods are indeed approximating the factorization of PMI matrices.

However, as we have seen in this paper, the embedding vectors are primarily constrained by their norm instead of embedding dimension, which implies that the resulting factorization is not really a standard low-rank one, but rather a low-norm factorization: DISPLAYFORM2 The low-norm factorization represents an interesting alternative to the standard low-rank factorization, and our current understanding of such factorization is still very limited.

Given the empirical success of linear graph embedding methods, it would be really helpful if we can have a more in-depth analysis of such factorization, to deepen our understanding and potentially inspire new algorithms.

We have shown that the generalization of linear graph embedding methods are not determined by the dimensionality constraint but rather the norm of embedding vectors.

We proved that limiting the norm of embedding vectors would lead to good generalization, and showed that the generalization of existing linear graph embedding methods is due to the early stopping of SGD and vanishing gradients.

We experimentally investigated the impact embedding dimension choice, and demonstrated that such choice only matters when there is no norm regularization.

In most cases, the best generalization performance is obtained by choosing the optimal value for the norm regularization coefficient, and in such case the impact of embedding dimension case is negligible.

Our findings combined with the analysis of BID7 suggest that linear graph embedding methods are probably computing a low-norm factorization of the PMI matrix, which is an interesting alternative to the standard low-rank factorization and calls for further study.

We use the following three datasets in our experiments:??? Tweet is an undirected graph that encodes keyword co-occurrence relationships using Twitter data: we collected ???1.1 million English tweets using Twitter's Streaming API during 2014 August, and then extracted the most frequent 10,000 keywords as graph nodes and their co-occurrences as edges.

All nodes with more than 2,000 neighbors are removed as stop words.

There are 9,913 nodes and 681,188 edges in total.??? BlogCatalog BID19 ) is an undirected graph that contains the social relationships between BlogCatalog users.

It consists of 10,312 nodes and 333,983 undirected edges, and each node belongs to one of the 39 groups.??? YouTube BID9 ) is a social network among YouTube users.

It includes 500,000 nodes and 3,319,221 undirected edges 6 .For each positive edge in training and testing datasets, we randomly sampled 4 negative edges, which are used for learning the embedding vectors (in training dataset) and evaluating average precision (in testing dataset).

In all experiments, ?? + = 1, ?? ??? = 0.03, which achieves the optimal generalization performance according to cross-validation.

All initial coordinates of embedding vectors are uniformly sampled form [???0.1, 0.1].

In the early days of graph embedding research, graphs are only used as the intermediate data model for visualization BID6 or non-linear dimension reduction BID16 BID1 .

Typically, the first step is to construct an affinity graph from the features of the data points, and then the low-dimensional embedding of graph vertices are computed by finding the eigenvectors of the affinity matrix.

For more recent graph embedding techniques, apart from the linear graph embedding methods discussed in this paper, there are also methods BID17 BID5 BID3 that explore the option of using deep neural network structures to compute the embedding vectors.

These methods typically try to learn a deep neural network model that takes the raw features of graph vertices to compute their low-dimensional embedding vectors: SDNE BID17 uses the adjacency list of vertices as input to predict their Laplacian Eigenmaps; GCN BID5 aggregates the output of neighboring vertices in previous layer to serve as input to the current layer (hence the name "graph convolutional network"); GraphSage BID3 extends GCN by allowing other forms of aggregator (i.e., in addition to the mean aggregator in GCN).

Interestingly though, all these methods use only 2 or 3 neural network layers in their experiments, and there is also evidence suggesting that using higher number of layer would result in worse generalization performance BID5 .

Therefore, it still feels unclear to us whether the deep neural network structure is really helpful in the task of graph embedding.

Prior to our work, there are some existing research works suggesting that norm constrained graph embedding could generalize well.

BID13 studied the problem of computing norm constrained matrix factorization, and reported superior performance compared to the standard lowrank matrix factorization on several tasks.

Given the connection between matrix factorization and linear graph embedding BID7 , the results in our paper is not really that surprising.

Since E ?? consists of i.i.d.

samples from P, by the uniform convergence theorem BID0 BID12 , with probability 1 ??? ??: DISPLAYFORM0 is the hypothesis set, and R(H C U ,C V ) is the empirical Rademacher Complexity of H C U ,C V , which has the following explicit form: DISPLAYFORM1 Here ?? a,b are i.i.d.

Rademacher random variables: Pr(?? a,b = 1) = Pr(?? a,b = ???1) = 0.5.

Since l is 1-Lipschitz, based on the Contraction Lemma BID12 , we have: DISPLAYFORM2 Let us denote X U as the |U |d dimensional vector obtained by concatenating all vectors x u , and X V as the |V |d dimensional vector obtained by concatenating all vectors x v : DISPLAYFORM3 Then we have: DISPLAYFORM4 where A ??? B represents the Kronecker product of A and B, and ||A|| 2 represents the spectral norm of A (i.e., the largest singular value of A).Finally, since ||A ??? I|| 2 = ||A|| 2 , we get the desired result in Theorem 1.

We provide the sketch of a constructive proof here.

Once we have repeated the above procedure for every node in V , it is easy to see that all the constraints yx Now let us assume that the graph G is generated from a Erdos-Renyi model (i.e., the probability of any pair u, v being directed connected is independent), then we have: DISPLAYFORM0 where e ij is the boolean random variable indicating whether (i, j) ??? E.By Central Limit Theorem, where m is the expected number of edges, and n is the total number of vertices.

Then we have, DISPLAYFORM1 for all ||x|| 2 = ||y|| 2 = 1.

Now let S be an -net of the unit sphere in n dimensional Euclidean space, which has roughly O( ???n ) total number of points.

Consider any unit vector x, y ??? R n , and let x S , y S be the closest point of x, y in S, then: By union bound, the probability that at least one pair of x S , y S ??? S satisfying y T S A ?? x S ??? t is at most:Pr(???x S , y S ??? S : y T S A ?? x S ??? t) ??? O( ???2n e ??? t 2 n 2 2m ) Let = 1/n, t = 8m ln n/n, then the above inequality becomes:Pr(???x S , y S ??? S : y T S A ?? x S ??? t) ??? O(e ???n ln n )Since ???x S , y S ??? S, y T S A ?? x S < t implies that sup ||x||2=||y||2=1,x,y???R n y T A ?? x < t + 2 n + 2 n Therefore, we estimate ||A ?? || 2 to be of order O( m ln n/n).

Algorithm 1 shows the full pseudo-code of the DCD method for optimizing the hinge-loss variant of linear graph embedding learning.

<|TLDR|>

@highlight

We argue that the generalization of linear graph embedding is not due to the dimensionality constraint but rather the small norm of embedding vectors.

@highlight

The authors show that the generalization error of linear graph embedding methods is bounded by the norm of embedding vectors rather than dimensionality constraints

@highlight

The authors propose a theoretical bound on the generalization performance of learning graph embeddings and argue that the norm of the coordinates determines the success of the learnt representation.
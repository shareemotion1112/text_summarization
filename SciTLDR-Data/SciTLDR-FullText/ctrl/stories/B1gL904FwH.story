To deal simultaneously with both, the attributed network embedding and clustering, we propose a new model.

It exploits both content and structure information, capitalising on their simultaneous use.

The proposed model relies on the approximation of the relaxed continuous embedding solution by the true discrete clustering one.

Thereby, we show that incorporating an embedding representation provides simpler and more interpretable solutions.

Experiment results demonstrate that the proposed algorithm performs better, in terms of clustering and embedding, than the state-of-art algorithms, including deep learning methods devoted to similar tasks for attributed network datasets with different proprieties.

In recent years, Attributed Networks (AN) (Qi et al., 2012) have been used to model a large variety of real-world networks, such as academic and health care networks where both node links and attributes/features are available for analysis.

Unlike plain networks in which only node links and dependencies are observed, with AN, each node is associated with a valuable set of features.

More recently, the learning representation has received a significant amount of attention as an important aim in many applications including social networks, academic citation networks and proteinprotein interaction networks.

Hence, Attributed network Embedding (ANE) (Cai et al., 2018; Yan et al., 2007; aims to seek a continuous low-dimensional matrix representation for nodes in a network, such that original network topological structure and node attribute proximity can be preserved in the new low-dimensional embedding.

Although, many approaches have emerged with Network Embedding (NE), the research on ANE (Attributed Network Embedding) is still remains to be explored .

Unlike NE that learns from plain networks, ANE aims to capitalize both the proximity information of the network and the affinity of node attributes.

Note that, due to the heterogeneity of the two information sources, it is difficult for the existing NE algorithms to be directly applied to ANE.

To sum up, the learned representation has been shown to be helpful in many learning tasks such as network clustering (Wang et al., 2017) , nodes visualization (Dai et al., 2018) , nodes classification (Zhu et al., 2007; Dai et al., 2018; Huang et al., 2017) and link prediction (Singh & Gordon, 2008; Pan et al., 2018) .

Therefore ANE is a challenging research problem due to the high-dimensionality, sparsity and non-linearity of the graph data.

Frequently, embedding and clustering are used to better understand the content and structure information from the clusters.

Many approaches have been proposed for the learning representation and clustering tasks (Bock, 1987; De Soete & Carroll, 1994; Vichi & Kiers, 2001; Yamamoto & Hwang, 2014) without available information from a network.

With ANE we start to list some works (Pan et al., 2018; Guo et al., 2019) where the proposed algorithms show the benefits of clustering.

Although existing AN clustering has been widely applied, they may achieve poor performance due to the following drawbacks: -High risk of severe deviation of approximate continuous embedding solution from the good discrete clustering.

-Information loss among separate independent stages, i.e., continuous embedding generation and embedding discretization.

These problems result from the sequential way where the learned representation is obtained before obtaining clusters using a technical clustering.

This is implicitly due to the fact that the two tasks do not aim to reach the same objective and, in addition, are carried out separately.

To remedy this weakness.

we propose a novel simultaneous ANE and clustering scheme which jointly; (1) learns embedding from both network and attributes information (2) learns continuous embedding and discrete clustering labels.

Specifically, we explicitly enforce a discrete transformation on the intermediate continuous labels (embedding) , which leads to a tractable optimization problem with a discrete solution.

The key challenge is to know how to integrate the information of both node links and attributes for simultaneous node representation learning and discrete node clustering.

In order to alleviate the influence caused by the information loss during the relaxation of sequential clustering, then to recover a discrete clustering solution, we use a smooth transformation (e.g., rotation) from the relaxed continuous embedding to a discrete solution.

In this sense, the continuous embedding only serves as an intermediate product.

To our best knowledge, the adoption of simultaneous attributed network embedding and clustering in a unified learning framework which has not been adequately investigated yet.

The goal of this work is to conduct investigations along this direction by considering matrix decomposition as the embedding framework.

The rest of the paper is organized as follows.

Section 2 introduces the AN embedding problem and clustering for community detection.

Section 3 provides a sound Simultaneous Attributed Network Embedding and Clustering (SANEC) framework for embedding and clustering.

Section 4 is devoted to numerical experiments.

Finally, the conclusion summarizes the advantages of our contribution.

In this section, we descibe the Simultaneous Attributes Network Embedding and Clustering (SANEC) method.

We will present the formulation of an objective function and an effective algorithm for data embedding and clustering.

But first, we show how to construct two matrices S and M integrating both types of information -content and structure information-to reach our goal.

An attributed network G = (V, E, X) consists of V the set of nodes, E ⊆ V × V the set of links, and X = [x 1 , x2, . . .

, x n ] where n = |V | and x i ∈ R d is the feature/attribute vector of the node v i .

Formally, the graph can be represented by two types of information, the content information X ∈ R n×d and the structure information A ∈ R n×n , where A is an adjacency matrix of G and a ij = 1 if e ij ∈ E otherwise 0; we consider that each node is a neighbor of itself, then we set a ii = 1 for all nodes.

Thereby, we model the nodes proximity by an (n × n) transition matrix W given by W = D −1 A, where D is the degree matrix of A defined by d ii = n i =1 a i i .

In order to exploit additional information about nodes similarity from X, we preprocessed the above dataset X to produce similarity graph input W X of size (n×n); we construct a K-Nearest-Neighbor (KNN) graph.

To this end, we use the heat kernel and L 2 distance, KNN neighborhood mode with K = 15 and we set the width of the neighborhood σ = 1.

Note that any appropriate distance or dissimilarity measure can be used.

Finally we combine in an (n × n) matrix S, nodes proximity from both content information X and structure information W. In this way, we propose to perturb the similarity W by adding the similarity from W X ; we choose to take S defined by

In Figure 3 .1, we illustrate the impact of W X by applying Multidimensional scaling on both W and S. Note that with S, the sparsity is overcome by the presence of W X .

We will see later the interest of its use in S. As we aim to perform clustering, we propose to integrate it in the formulation of a new data representation by assuming that nodes with the same label tend to similar social relations and similar node attributes.

This idea is inspired by the fact that, the labels are strongly influenced by both content and structure information and inherently correlated to these both information sources.

This reminds us that the idea behind the Canonical Discriminant Analysis (CDA) which is a dimension-reduction technique related to principal component analysis (PCA) and canonical correlation (Goodfellow et al., 2016) .

Given groups of observations with measurements on attributes, CDA derives a linear combination of the variables that has the highest possible multiple correlation with the groups.

It can be viewed as a particular PCA where the observations belonging to a same group are replaced by their centroid.

Thereby the new data representation referred to as M = (m ij ) of size (n × d) can be considered as a multiplicative integration of both W and X by replacing each node by the centroid of their neighborhood (barycenter):

In Figure 3 .1, it is interesting to point out the impact of W in the formulation of M. We apply CDA on X and M and indicate the seven true clusters of the Cora dataset.

This leads to show clusters better separated with M than with X and therefore that W already does a good job without clustering.

In this way, given a graph G, a graph clustering aims to partition the nodes in G into k disjoint clusters {C 1 , C 2 , ..., C k }, so that: (1) nodes within the same cluster are close to each other while nodes in different clusters are distant in terms of graph structure; and (2) the nodes within the same cluster are more likely to have similar attribute values.

Let k be the number of clusters and the number of components into which the data is embedded.

With M and S, the SANEC method that we propose aims to obtain the maximally informative embedding according to the clustering structure in the attributed network data.

Therefore, the proposed objective function to optimize is given by

where G = (g ij ) of size (n × k) is a cluster membership matrix, B = (b ij ) of size (n × k) is the embedding matrix and Z = (z ij ) of size (k × k) is an orthonormal rotation matrix which most closely maps B to G ∈ {0, 1} n×k .

Q ∈ R d×k is the features embedding matrix.

Finally, The parameter λ is a non-negative value and can be viewed as a regularization parameter.

The intuition behind the factorization of M and S is to encourage the nodes with similar proximity, those with higher similarity in both matrices, to have closer representations in the latent space given by B. In doing so, the optimisation of (3) leads to a clustering of the nodes into k clusters given by G. Note that, both tasks -embedding and clustering-are performed simultaneously and supported by Z; it is the key to attaining good embedding while taking into account the clustering structure.

To infer the latent factor matrices Z, B, Q and G, we shall derive an alternating optimization algorithm.

To this end, we rely on the following proposition.

Proof.

We first expand the matrix norm of the left term of (4)

In a similar way, we obtain from the two terms of the right term of (4)

and

Due also to B B = I, we have

Summing the two terms of (6) and (7 ) leads to the left term of (4).

Compute Z. Fixing G and B the problem which arises in (3) is equivalent to min Z S − GZB 2 .

From proposition 1., we deduce that

which can be reduced to max Z T r(G SBZ) s.t.

Z Z = I. As proved in page 29 of ten Berge (1993), let UΣV be the SVD for G SB, then

We can observe that this problem turns out to be similar to the well known orthogonal Procrustes problem (Schonemann, 1966) .

Compute Q. Given G, Z and B, the opimization problem (3) is equivalent to min Q M − BQ 2 , and we get

Thereby Q is somewhere an embedding of attributes.

Compute B. Given G, Q and Z, the problem (3) is equivalent to

In the same manner for the computation of Z, letÛΣV be the SVD for (M Q + λSGZ), we get

It is important to emphasize that, at each step, B exploits the information from the matrices Q, G, and Z. This highlights one of the aspects of the simultaneity of embedding and clustering.

Compute G: Finally, given B, Q and Z, the problem (3) is equivalent to min G SB − GZ 2 .

As G is a cluster membership matrix, its computation is done as follows: We fix Q, Z, B. LetB = SB and calculate

3.4 SANEC ALGORITHM In summary, the steps of the SANEC algorithm relying on S referred to as SANEC S can be deduced in Algorithm 1.

The convergence of SANEC S is guaranteed and depends on the initialization to reach only a local optima.

Hence, we start the algorithm several times and select the best result which minimizes the objective function (3).

Algorithm 1 : SANEC S algorithm Input: M and S from structure matrix W and content matrix X ; Initialize: B, Q and Z with arbitrary orthonormal matrix; repeat (a) -Compute G using (13) (b) -Compute B using (12) (c) -Compute Q using (11) (d) -Compute Z using (10) until convergence Output: G: clustering matrix, Z: rotation matrix, B: nodes embedding and Q: attributes embedding

In our work we focus on different clustering methods.

In the sequel, we evaluate the SANEC algorithm with some competitive methods including recent deep learning methods.

We compare our method with both embedding based approaches as well as approaches directly for graph clustering.

We consider classical methods and also deep learning methods; they differ in the use of available information.

Some of them rely only on X such as K-means and others more recent on X and W. Graph Encoder (Tian et al., 2014) learns graph embedding for spectral graph clustering.

DNGR (Cao et al., 2016) trains a stacked denoising autoencoder for graph embedding.

RTM (Chang & Blei, 2009 ) learns the topic distributions of each document from both text and citation.

RMSC (Xia et al., 2014 ) employs a multi-view learning approach for graph clustering.

TADW (Yang et al., 2015) applies matrix factorization for network representation learning.

DeepWalk (Perozzi et al., 2014 ) is a network representation approach which encodes social relations into a continuous embedding space.

Spectral Clustering (Tang & Liu, 2011 ) is a widely used approach for learning social embedding.

GAE (Kipf & Welling, 2016 ) is an autoencoder-based unsupervised framework for attributed network data embedding.

VGAE (Kipf & Welling, 2016 ) is a variational graph autoencoder approach for graph embedding with both node links and node attributes information.

ARGA (Pan et al., 2018) is the most recent adversarially regularized autoencoder algorithm which uses graph autoencoder to learn the embedding.

ARVGA (Pan et al., 2018) algorithm, which uses a variational graph autoencoder to learn the embedding.

For embedding based approaches, we first learn the graph embedding, and then perform k-means clustering algorithm based on the embedding.

For our proposed method, we set the regularization parameter λ ∈ {0, 10 −6 , 10 −3 , 10 −1 , 10 0 , 10 1 , 10 3 }, and choose the best values as the final results.

The performances of clustering methods are evaluated using real-world datasets commonly tested with ANE where the clusters are known.

Specifically, we consider three public citation network data sets, Citeseer, Cora and Wiki, which contain sparse bag-of-words feature vector for each document and a list of citation links between documents.

Each document has a class label.

We treat documents as nodes and the citation links as the edges.

The characteristics of the used datasets are summarized in Table 1 .

The balance coefficient is defined as the ratio of the number of documents in the smallest class to the number of documents in the largest class while nz denotes the percentage of sparsity.

With the SANEC model, the parameter λ controls the role of the second term ||S − GZB || 2 in (3).

To measure its impact on the clustering performance of SANEC S , we vary λ in {0, 10 −6 , 10 −3 , 10 −1 , 10 0 , 10 1 , 10 3 }.

The performances in terms of accuracy (Acc), normalized mutual information (NMI) and adjusted rand index are depicted in Figure 4 ; a Acc, NMI or ARI corresponds to a better clustering result.

First we note that with λ = 0 we rely only on min B,Q M − BQ 2 s.t.

B B = I. In this case, we observe poor results in terms of quality of clustering, this leads to show the impact of the second term in (3).

This quality increases according to λ and we get better performance on all datasets with small values of λ.

Around 10 −2 , it is worthy to note that the clustering result is stable and less sensitive to λ.

However, the performance of SANEC degrades sharply for λ greater than 10, this can be explained by the fact that the initialization of G which is random and can often be far from the real solution.

Through, many experiments on the three datasets and others not reported here, we choose to take λ = 10 −2 .

Compared to the true available clusters, in our experiments the clustering performance is assessed by ACC NMI and ARI.

We repeat the experiments 50 times and the averages (mean) are reported in Table 2 ; the best performance for each dataset is highlighted in bold.

First, we observe the high performances of methods integrating information from W. For instance, RTM and RMSC are better than classical methods using only either M or W. On the other hand, all methods including deep learning algorithms relying on M and W are better yet.

However, regarding SANEC with both versions relying on W, referred to as SANEC W or S referred to as SANEC S , we note high performances for all the datasets and with SANEC S , we remark the impact of W X ; it learns lowdimensional representations while suits the clustering structure.

To go further in our investigation and given the sparsity of X we proceeded to standardization tf-idf followed by L 2 , as it is often used to process document-term matrices; see e.g;, (Salah & Nadif, 2017; , while in the construction of W X we used the cosine metric.

In Figure 4 are reported the results where we observe a slight improvement.

The SANEC model, through B, offers an embedding from which we can also observe a 2d or 3d structure into clusters.

To illustrate the quality of embedding, we consider the three attributed network datasets described above and we use UMAP for data visualization (McInnes et al., 2018) .

The UMAP algorithm leads to dimension reduction based on manifold learning techniques and ideas from topological data analysis.

As in the construction of W X , the number of neighbors that we have chosen with UMAP is equal to 30.

In this paper, we proposed a novel matrix decomposition framework for simultaneous attributed network data embedding and clustering.

Unlike known methods that combine the objective function of AN embedding and the objective function of clustering separately, we proposed a new single framework to perform SANEC S for AN embedding and nodes clustering.

We showed that the optimized objective function can be decomposed into three terms, the first is the objective function of a kind of PCA applied to M, the second is the graph embedding criterion in a low-dimensional space, and the third is the clustering criterion.

We also integrated a discrete rotation functionality, which allows a smooth transformation from the relaxed continuous embedding to a discrete solution, and guarantees a tractable optimization problem with a discrete solution.

Thereby, we developed an effective algorithm capitalizing on learning representation and clustering.

The obtained results show the advantages of combining both tasks over other approaches.

SANEC S outperforms the all recent methods devoted to the same tasks including deep learning methods which require deep models pretraining.

The proposed framework offers several perspectives and investigations.

We have noted that the construction of M and S is important, it highlights the introduction of W. As for the W X we have observed that it is fundamental as it makes possible to link the information from X to the network; this has been verified by many experiments.

Finally, as we have stressed that Q is an embedding of attributes, this suggests to consider also a simultaneously ANE and co-clustering.

<|TLDR|>

@highlight

This paper propose a novel matrix decomposition framework for simultaneous attributed network data embedding and clustering.

@highlight

This paper proposes an algorithm to perform jointly attribute network embedding and clustering together.
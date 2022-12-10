Detecting communities or the modular structure of real-life networks (e.g. a social network or a product purchase network) is an important task because the way a network functions is often determined by its communities.

The traditional approaches to community detection involve modularity-based approaches, which generally speaking, construct partitions based on heuristics that seek to maximize the ratio of the edges within the partitions to those between them.

Node embedding approaches, which represent each node in a graph as a real-valued vector, transform the problem of community detection in a graph to that of clustering a set of vectors.

Existing node embedding approaches are primarily based on first initiating uniform random walks from each node to construct a context of a node and then seeks to make the vector representation of the node close to its context.

However, standard node embedding approaches do not directly take into account the community structure of a network while constructing the context around each node.

To alleviate this, we explore two different threads of work.

First, we investigate the use of biased random walks (specifically, maximum entropy based walks) to obtain more centrality preserving embedding of nodes, which we hypothesize may lead to more effective clusters in the embedded space.

Second, we propose a community structure aware node embedding approach where we incorporate modularity-based partitioning heuristics into the objective function of node embedding.

We demonstrate that our proposed approach for community detection outperforms a number of modularity-based baselines as well as K-means on a standard node-embedded vector space (specifically, node2vec) on a wide range of real-life networks of different sizes and densities.

Partitioning a network (graph) into communities usually leads to better analyzing the functionality of the network and is of immense practical interest for real-world networks, because such communities potentially represent organizational units in social networks, scientific disciplines in authorshipcitation academic publications networks, or functional units in biological networks (e.g. proteinprotein interactions) (Girvan & Newman, 2002; Newman & Girvan, 2004; Waltman & Van Eck, 2013) .

A network community represents a set of nodes with a relatively dense set of connections between its members and relatively sparse connections between its member nodes and the ones outside the community.

Traditional approaches of community detection incrementally construct a community (set of nodes) by employ an objective function that seeks to maximize its internal connectivity and minimize the number of external edges (Newman & Girvan, 2004; Newman, 2006; Blondel et al., 2008; PratPérez et al., 2014) .

Graph representation learning approaches such as (Perozzi et al., 2014; Grover & Leskovec, 2016) represent each node of a graph as a real-valued vector seeking to preserve the correlation between the topological properties of the discrete graph with the distance measures in the embedded metric space.

For example, the vectors corresponding to a pair of nodes in the embedded space is usually close (low distance or high inner product similarity) if it is likely to visit a node of the pair with a random walk started at the other one.

However, a major limitation of the random walk based node representation learning approach is that a random walk may span across the community from which it stared with, which eventually could lead to representing nodes from different communities in close proximity in the embedding space.

This in turn can may not result in effective community detection on application of a standard clustering algorithm, e.g. K-means, in the space of embedded node vectors.

Ideally speaking, for effective community detection with a clustering algorithm operating on the embedded space of node vectors, a node embedding algorithm should preserve the community structure from the discrete space of the sets of nodes to the continuous space of real-valued vectors as perceived with the conventional definitions of the distance metric (e.g. l 2 distance) and the inner product between pairs of vectors denoting the similarity between them.

In other words, a central (hub) node of a community in the discrete graph representation should be transformed in the embedded space in such a way so that it contains other vectors, corresponding to the nodes of the other members in the community, in its close neighborhood.

In our study, we investigate two methods to achieve such a transformation.

Our Contributions First, in contrast to the uniform random walk (URW) based contextualization of nodes in standard node embedding approaches, such as node2vec (Grover & Leskovec, 2016) and DeepWalk (Perozzi et al., 2014) , we investigate a maximum-entropy based biased random walk (MERW) Sinatra et al. (2011) , where in contrast to URW, the transition probabilities are non-local, i.e., they depend on the structure of the entire graph.

Alternately, in our second proposed approach, we investigate if traditional approaches to community detection that operate on a discrete graph (adjacency matrix), e.g. modularity-heuristic (Clauset et al., 2004) or InfoMap (Rosvall & Bergstrom, 2008) , can be useful to contextualize a node for the purpose of obtaining its embedded representation.

In other words, while training a classifier for a node vector that learns to predict its context, we favour those cases where the context nodes are a part of the same community as that of the current node, as predicted by a modularity-based heuristic).

We also investigate a combination of the two different community aware embedding approaches, i.e. employing MERW to first contextualize the nodes and then using the weighted training based on the modularity heuristic.

The rest of the paper is organized as follows.

We first review the literature on community detection and node embedding.

We then describe the details about the MERW-based node embedding and community-structure aware node embedding.

Next, we describe the setup of our experiments, which is followed by a presentation and analysis of the results.

Finally, we conclude the paper with directions for future work.

In this section, we provide a brief overview of the community detection and node representation learning approaches.

In this section, we review a number of combinatorial approaches to community detection.

Each combinatorial approach has the common underlying principle of first constructing an initial partition of an input graph into a set of sub-graphs (communities) and then refining the partition at every iterative step.

Among a number of possible ways to modify a current partition, the one that maximizes a global objective function is chosen.

The global objective, in turn, is computed by aggregating the local objectives over and across the constituent sub-graphs.

Clauset et al. (2004) defines modularity as an intrinsic measure of how effectively, with respect to its topology, a graph (network) is partitioned into a given set of communities.

More formally, given a partition of a graph G = (V, E) into p communities, i.e. given an assigned community (label) c v ∈ {1, . . .

, p} for each node v ∈ V , the modularity, Q is defined as the expected ratio of the number of intra-community edges to the total number of edges, the expectation being computed with respect to the random case of assigning the nodes to arbitrary communities.

More specifically,

where A vw denotes the adjacency relation between nodes v and w, i.e. A vw = 1 if (v, w) ∈ E; k v denotes the number of edges incident on a node v; I(c v , c w ) indicates if nodes v and w are a part of the same community.

A high value of Q in Equation 1 represents a substantial deviation of the fraction of intra-community edges to the total number of edges from what one would expect for a randomized network, and Clauset et al. (2004) suggests that a value above 0.3 is often a good indicator of significant community structure in a network.

The 'CNM' (Clauset Newman Moore) algorithm (Newman & Girvan, 2004) proposes a greedy approach that seeks to optimise the modularity score (Equation 1).

Concretely speaking, it starts with an initial state of node being assigned to a distinct singleton community, seeking to refine the current assignment at every iteration by merging a pair of communities that yields the maximum improvement of the modularity score.

The algorithm proceeds until it is impossible to find a pair of communities which if merged yields an improvement in the modularity score.

The 'Louvain' or the 'Multilevel' algorithm (Blondel et al., 2008) involves first greedily assigning nodes to communities, favoring local optimizations of modularity, and then repeating the algorithm on a coarser network constructed from the communities found in the first step.

These two steps are repeated until no further modularity increasing reassignments are found.

'SCDA' (Scalable Community Detection Algorithm) (Prat-Pérez et al., 2014) detects disjoint communities in networks by maximizing WCC, a recently proposed community metric Prat-Pérez et al. (2012) based on triangle structures within a community.

SCD implements a two-phase procedure that combines different strategies.

In the first phase, SCD uses the clustering coefficient as an heuristic to obtain a preliminary partition of the graph.

In the second phase, SCD refines the initial partition by moving vertices between communities as long as the WCC of the communities increase.

Jiang & Singh (2010) proposed a scalable algorithm -'SPICi' ('Speed and Performance In Clustering' and pronounced as 'spicy'), which constructs communities of nodes by first greedily starting from local seed sets of nodes with high degrees, and then adding those nodes to a cluster that maximize a two-fold objective of the density and the adjacency of nodes within the cluster.

The underlying principle of SPICi is similar to that of 'DPClus' (Altaf-Ul-Amin et al., 2006), the key differences being SPICi exploits a simpler cluster expansion approach, uses a different seed selection criterion and incorporates interaction confidences.

Newman (2006) proposed 'LEADE' (Leading Eigenvector) applies a spectral decomposition of the modularity matrix M , defined as

The leading eigenvector of the modularity matrix is used to split the graph into two sub-graphs so as to maximize modularity improvement.

The process is then recursively applied on each sub-graph until the modularity value cannot be improved further.

'LPA' (Label Propagation Algorithm) (Raghavan et al., 2007) relies on the assumption that each node of a network is assigned to the same community as the majority of its neighbours.

The algorithm starts with initialising a distinct label (community) for each node in the network.

Each node, visited in a random order, then takes the label of the majority of its neighbours.

The iteration stops when the label assignments cannot be changed further.

Rosvall & Bergstrom (2008) proposed the 'InfoMap' algorithm, which relies on finding the optimal encoding of a network based on maximizing the information needed to compress the movement of a random walker across communities on the one hand, whereas minimizing the code length to represent this information.

The algorithm makes uses of the core idea that random walks initiated from a node which is central to a community is less likely to visit a node of a different community.

Huffman encoding of such nodes, hence, are likely to be shorter.

The 'WalkTrap' algorithm (Pons & Latapy, 2005 ) is a hierarchical agglomerating clustering (HAC) algorithm using an idea similar to InfoMap that short length random walks tend to visit only the nodes within a single community.

The distance metric that the algorithm uses for the purpose of HAC between two sets of nodes is the distance between the probability distributions of nodes visited by random walks initiated from member nodes of the two sets.

Different from the existing work in combinatorial approaches to community detection, in our work, we propose a framework to integrate a combinatorial approach within the framework of an embedding approach (specifically, node2vec).

In contrast to the combinatorial approaches which directly work on the discrete space (vertices and edges) of a graph, G = (V, E), an embedding approach transforms each node of a graph, u, into a real-valued vector, u, seeking to preserve the topological structure of the nodes.

Formally,

The transformation function θ is learned with the help of noise contrastive estimation, i.e., the objective is to make the similarity (inner product) between vectors for nodes u and v higher if v lies in the neighborhood of u, and to be of a value small if v does not belong to the neighborhood of u (e.g. v being a randomly sampled node from the graph).

Formally,

where y denotes a binary response variable to train the likelihood function, where N (u) denotes the neighborhood of node u, and the negative component in the likelihood function refers to the randomly sampled noise.

Popular approaches to learn the transformation function of Equation 3 includes node2vec (Grover & Leskovec, 2016) and DeepWalk (Perozzi et al., 2014) , which differ in the way the neighborhood function, N (u), is defined.

While DeepWalk uses a uniform random walk to constitute the neighborhood or context of a node, node2vec uses a biased random walk (with a relative importance to depth-first or breadth-first traversals).

A transformation of the nodes as real-valued vectors then allows the application of relatively simple (but effective) clustering approaches, such as K-means, to partition the embedding space of nodes into distinct clusters.

This is because in contrast to the discrete space, the vector space is equipped with a metric function which allows to compute distance (or equivalently similarity) between any pair of nodes (as opposed to the discrete case).

Cavallari et al. (2017) proposed an expectation-maximization (EM) based approach to iteratively refine a current community assignment (initialized randomly) using node embeddings.

The objective was to ensure that the embedded vectors of each community fits a Gaussian mixture model, or in other words, the embedded space results in relatively disjoint convex clusters.

In contrast to (Cavallari et al., 2017) , our method does not involve a feedback-based EM step.

Wang et al. (2016) proposed to include an additional term in the objective of the transformation function (Equation 3) corresponding to the second order similarity between the neighborhoods of two nodes.

Different to (Wang et al., 2016) , which seeks to obtain a general purpose embedding of graphs, we rather focus only on the community detection problem.

Let P ∈ R |V |×|V | denote the stochastic transition matrix of a graph G = (V, E), where P uv denotes the probability of visiting node v in sequence after visiting node u. In a standard uniform random walk (URW), this probability is given by

where k u denotes the degree of node u. In other words, Equation 5 indicates that there is a equal likelihood of choosing a node v as the next node in sequence from the neighbors of node u.

Maximal-entropy random walk (MERW) is characterized by a stochastic matrix that maximises entropy of a set of paths (node sequences) with a given length and end-points (Ochab & Burda, 2013) .

It leads to the following stochastic matrix.

where λ denotes the largest eigenvalue of the adjacency matrix A, and ψ v and ψ u refer to the v th and the u th components of the corresponding eigenvector.

Parry (1964) applied the FrobeniusPerron theorem to prove that the probability of visiting a node u n after n time steps starting from node u 1 depends only on the number of steps and the two ending points, but is independent of the intermediate nodes, i.e.

Consequently, the choice of the next node to visit in MERW is based on uniformly selecting the node from alternative paths of a given length and end-points.

Delvenne & Libert (2011) shows that the stationary distribution attained by MERW better preserves centrality than URW, thus resulting in random walks that tend to be more local as shown in (Burda et al., 2009 ).

In the context of our problem, MERW based random walk initiated from a node of a community is more likely to remain within the confinements of the same community, as compared to URW.

Standard node embedding approaches, such as node2vec, uses URW to construct the set of contexts for a node for the purpose of learning its representation.

We hypothesize that replacing the URW based neighborhood function to a MERW one results in less likelihood of including a node v in the neighborhood of u, i.e. N (u).

This results in a low likelihood of including the term P (y = 1|u, v) of Equation 4, which corresponds to associating nodes across two different communities, as a positive example while training node representations.

In this section, we describe a two-step approach to node embedding that is likely to preserve the community structure of the discrete space of an input graph in the output embedded space as well.

The first step involves applying a combinatorial community detection algorithm that operates on the discrete input space to obtain an optimal partition, as per the objective function of the combinatorial approach, e.g. modularity (Clauset et al., 2004) or InfoMap (Rosvall & Bergstrom, 2008) .

Formally,

EXPERIMENT SETUP DATASETS Real-world networks with ground-truth communities The experiments are performed over three small scale standard networks, viz., Zacharys karate club network, bottlenose dolphin network, and American college football network.

Along with We have also tested the experiments on three real world networks viz.

Amazon, Youtube and DBLP Yang & Leskovec (2015) ; Harenberg et al. (2014) ; Leskovec & Krevl (2014) .

These networks are undirected and unweighted and they are selected from different application domains.

The overview of these networks are presented Table 1 .

Amazon 1 is an online commercial network for purchasing products.

Here nodes represent products and an edge exists between two products, if they are frequently purchased together.

Each product (i.e. node) belongs to one or more product categories.

Each ground-truth community is defined using hierarchically nested product categories that share a common function Yang & Leskovec (2015) .

Youtube is a website to share videos and considered as a social network.

Each user in the Youtube network is considered as a node and the friendship between two users is denoted as edge.

Moreover, an user can create a group where other Youtube users can be a member through their friendship.

These user created groups are considered as ground-truth communities Yang & Leskovec (2015) .

DBLP is a bibliographic network of Computer Science publications.

Here nodes represent authors and an edge between two nodes represent co-authorship.

Ground-truth communities are defined as sets of authors who published in the same journal or conference Yang & Leskovec (2015) .

The networks described above have several connected components and each connected component consisting of more than 3 nodes are considered as a separate ground-truth community.

Leskovec, et al. observed that the average goodness metric of the top k communities first remain flat with increasing k, but then after approximately 5000 communities, degrades rapidly Yang & Leskovec (2015) .

Therefore they have implemented some community detection algorithms using different goodness metrics on the top 5000 communities of some of the networks described above.

Eventually, they obtained nice results in terms of finding communities in those networks.

Following the same idea we have used only the top 5000 ground-truth communities of each of these networks in the experimental evaluation.

Artificial networks with ground-truth communities Furthermore, we use the LancichinettiFortunato-Radicchi (LFR) networks Lancichinetti et al. (2008) with ground-truth to study the behavior of a proposed community detection algorithm and to compare the performance across various competitive algorithms.

The LFR model involve with a set of parameters which controls the network topology.

In this model, degree distribution and community size distribution follow power laws with exponents γ and β, respectively.

Furthermore, we can also specify the other parameters such as number of vertices n, average degree k avg , maximum degree k max , minimum community size c min , maximum community size c max , and mixing parameter µ. We vary these parameters depending on our experimental needs.

The critical parameter is the mixing parameter µ, which indicates the proportion of relationships a node shares with other communities.

Six artificial networks are produced for experimental evaluations using the following parameter setting, γ = −2,β = −1 µ = 0.01 as mentioned by Lancichinetti et al. Lancichinetti et al. (2008) .

Table 2 provides the details of the other parameters to produce these artificial networks.

The results presented on these networks are the average of 100 runs to reduce the effect of random assumptions.

Omega Index.

The Omega Index Collins & Dent (1988) is an Adjusted Rand Index (ARI) Hubert & Arabie (1985) generalization applicable to overlapping clusters.

It is based on counting the number of pairs of elements occurring in exactly the same number of clusters as in the number of categories and adjusted to the expected number of such pairs.

Formally, given the ground-truth clustering C consisting of categories c i ∈ C and formed clusters c i ∈ C :

The observed agreement is:

where J (J) is the is the maximal number of categories (clusters) in which a pair of elements occurred, A j is the number of pairs of elements occurring in exactly j categories and exactly j clusters, and P = N (N − 1)/2 is the total number of pairs given a total of N elements (nodes of the network being clustered).

The expected agreement is:

where P j (P j ) is the total number of pairs of elements assigned to exactly j categories (clusters).

Mean F1 Score.

The Average F 1 score (F 1a) is a commonly used metric to measure the accuracy of clustering algorithms Yang & Leskovec (2013; 2015) ; Prat-Pérez et al. (2014) .

F 1a is defined as the average of the weighted F 1 scores of a) the best matching ground-truth clusters to the formed clusters and b) the best matching formed clusters to the ground-truth clusters.

Formally, given the ground-truth clustering C consisting of clusters c i ∈ C (called categories) and clusters c i ∈ C formed by the evaluating clustering algorithm:

where

and g(x i , Y ) = {argmax y F 1(x, y)|y ∈ Y } in which F 1(x, y) is the F1 score of the respective clusters.

Normalized Mutual Information (NMI).

Mutual Information (MI) is evaluated by taking all pairs of clusters from the formed and ground-truth clustering respectively and counts the number of common elements in each pair.

Formally, given the ground-truth clustering C consisting of clusters c ∈ C and the formed clusters c ∈ C, mutual information is defined as:

where p(c , c) is the normalized number of common elements in the pair of (category, cluster), p(c ) and p(c) is the normalized number of elements in the categories and formed clusters respectively.

The normalization is performed using the total number of elements in the clustering, i.e. the number of nodes in the input network.

There is no upper bound for I(C , C), so for easier interpretation and comparisons a normalized mutual information that ranges from 0 to 1 is desirable.

There are two ways in which normalization is normally done Strehl & Ghosh (2002) ; Esquivel & Rosvall (2012) .

In the first one, the mutual information is divided by the average of the entropies, while in the second one, it is divided by the maximum of the entropies.

These are defined as follows:

where H(X) = − x∈X p(x) log 2 p(x) is the entropy of the clustering X.

Modularity.

The modularity of a graph compares the presence of each intra-cluster edge of the graph with the probability that that edge would exist in a random graph Newman & Girvan (2004); Blondel et al. (2008) .

Although modularity has been shown to have a resolution limit Fortunato & Barthelemy (2007), some of the most popular clustering algorithms use it as an objective function Waltman & Van Eck (2013); Blondel et al. (2008) .

Modularity is defined as k (e kk −a 2 k ) where e kk , the probability of intra-cluster edges in cluster C k , and a k , the probability of either an intra-cluster edge in cluster C k or of an inter-cluster edge incident on cluster C k .

The higher the values of the four performance indices, omega index, modularity, normalized mutual information and average F1 score, the better the quality of the detected communities.

EXPERIMENTAL RESULTS

@highlight

A community preserving node embedding algorithm that results in more effective detection of communities with a clustering on the embedded space
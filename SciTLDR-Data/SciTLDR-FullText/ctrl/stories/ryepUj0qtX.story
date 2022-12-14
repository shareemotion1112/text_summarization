Network Embeddings (NEs) map the nodes of a given network into $d$-dimensional Euclidean space $\mathbb{R}^d$. Ideally, this mapping is such that 'similar' nodes are mapped onto nearby points, such that the NE can be used for purposes such as link prediction (if 'similar' means being 'more likely to be connected') or classification (if 'similar' means 'being more likely to have the same label').

In recent years various methods for NE have been introduced, all following a similar strategy: defining a notion of similarity between nodes (typically some distance measure within the network), a distance measure in the embedding space, and a loss function that penalizes large distances for similar nodes and small distances for dissimilar nodes.



A difficulty faced by existing methods is that certain networks are fundamentally hard to embed due to their structural properties: (approximate) multipartiteness, certain degree distributions, assortativity, etc.

To overcome this, we introduce a conceptual innovation to the NE literature and propose to create \emph{Conditional Network Embeddings} (CNEs); embeddings that maximally add information with respect to given structural properties (e.g. node degrees, block densities, etc.).

We use a simple Bayesian approach to achieve this, and propose a block stochastic gradient descent algorithm for fitting it efficiently.



We demonstrate that CNEs are superior for link prediction and multi-label classification when compared to state-of-the-art methods, and this without adding significant mathematical or computational complexity.

Finally, we illustrate the potential of CNE for network visualization.

Network Embeddings (NEs) map nodes into d-dimensional Euclidean space R d such that an ordinary distance measure allows for meaningful comparisons between nodes.

Embeddings directly enable the use of a variety of machine learning methods (classification, clustering, etc.) on networks, explaining their exploding popularity.

NE approaches typically have three components (Hamilton et al., 2017) : (1) A measure of similarity between nodes.

E.g. nodes can be deemed more similar if they are adjacent, have strongly overlapping neighborhoods, or are otherwise close to each other (link and path-based measures) BID18 BID20 , or if they have similar functional properties (structural measures) BID19 .

FORMULA3 A metric in the embedding space.

(3) A loss function comparing similarity between node pairs in the network with the proximity of their embeddings.

A good NE is then one for which the average loss is small.

A problem with all NE approaches is that networks are fundamentally more expressive than embeddings in Euclidean spaces.

Consider for example a bipartite network G = (V, U, E) with V, U two disjoint sets of nodes and E ??? V ?? U the set of links.

It is in general impossible to find an embedding in R d such that v ??? V and u ??? U are close for all (v, u) ??? E, while all pairs v, v ??? V are far from each other, as well as all pairs u, u ??? U .

To a lesser extent, this problem will persist in approximately bipartite networks, or more generally (approximately) k-partite networks such as networks derived from stochastic block models.

1 This shows that first-order similarity (i.e. adjacency) in networks cannot be modeled well using a NE.

Similar difficulties exist for second-order proximity (i.e. neighborhood overlap) and other node similarity notions.

A more subtle example is a network with a power law degree distribution.

A first-order similarity NE will tend to embed high degree nodes towards the center (to be close to lots of other nodes), while the low degree nodes will be on the periphery.

Yet, this effect reduces the embedding's degrees of freedom for representing similarity independent of node degree.

CNE: the idea To address these limitations of NEs, we propose a principled probabilistic approachdubbed Conditional Network Embedding (CNE)-that allows optimizing embeddings w.r.t.

certain prior knowledge about the network, formalized as a prior distribution over the links.

This prior knowledge may be derived from the network itself such that no external information is required.

A combined representation of a prior based on structural information and a Euclidean embedding makes it possible to overcome the problems highlighted in the examples above.

For example, nodes in different blocks of an approximately k-partite network need not be particularly distant from each other if they are a priori known to belong to the same block (and hence are unlikely or impossible to be connected a priori).

Similarly, high degree nodes need not be embedded near the center of the point cloud if they are known to have high degree, as it is then known that they are connected to many other nodes.

The embedding can thus focus on encoding which nodes in particular it is connected to.

CNE is also potentially useful for network visualization, with the ability to filter out certain information by using it as a prior.

For example, suppose the nodes in a network represent people working in a company with a matrix-structure (vertical being units or departments, horizontal contents such as projects) and links represent whether they interact a lot.

If we know the vertical structure, we can construct an embedding where the prior is the vertical structure.

The information that the embedding will try to capture corresponds to the horizontal structure.

The embedding can then be used in downstream analysis, e.g., to discover clusters that correspond to teams in the horizontal structure.

Contributions and outline Our contributions can be summarized as follows:??? This paper introduces the concept of NE conditional on certain prior knowledge about the network.??? Section 2 presents CNE ('Conditional Network Embedding'), which realizes this idea by using Bayes rule to combine a prior distribution for the network with a probabilistic model for the Euclidean embedding conditioned on the network.

This yields the posterior probability for the network conditioned on the embedding, which can be maximized to yield a maximum likelihood embedding.

Section 2.2 describes a scalable algorithm based on block stochastic gradient descent.??? Section 3 reports on extensive experiments, comparing with state-of-the-art baselines on link prediction and multi-label classification, on commonly used benchmark networks.

These experiments show that CNE's link prediction accuracy is consistently superior.

For multi-label classification CNE is consistently best on the Macro-F 1 score and best or second best on the Micro-F 1 score.

These results are achieved with considerably lower-dimensional embeddings than the baselines.

A case study also demonstrates the usefulness of CNE in exploratory data analysis of networks.??? Section 4 gives a brief overview of related work, before concluding the paper in Section 5.??? All code, including code for repeating the experiments, and links to the datasets are available at:https://bitbucket.org/ghentdatascience/cne.

Section 2.1 introduces the probabilistic model used by CNE, and Section 2.2 describes an algorithm for optimizing it to find an optimal CNE.

Before doing that, let us introduce some notation.

An undirected network is denoted G = (V, E) where V is a set of n = |V | nodes and E ??? V 2 is the set of links (also known as edges).

A link is denoted by an unordered node pair {i, j} ??? E. Let?? denote the network's adjacency matrix, with element?? ij = 1 for {i, j} ??? E and?? ij = 0 otherwise.

The goal of NE (and thus of CNE) is to find a mapping f : DISPLAYFORM0

The newly proposed method CNE aims to find an embedding X that is maximally informative about the given network G, formalized as a Maximum Likelihood (ML) estimation problem: DISPLAYFORM0 Innovative about CNE is that we do not postulate the likelihood function P (G|X) directly, as is common in ML estimation.

Instead, we use a generic approach to derive prior distributions for the network P (G), and we postulate the density function for the data conditional on the network p(X|G).

This allows one to introduce any prior knowledge about the network into the formulation, through a simple application of Bayes rule DISPLAYFORM1 .

The consequence is that the embedding will not need to represent any information that is already represented by the prior P (G).Section 2.1.1 describes how a broad class of prior information types can be modeled for use by CNE.

Section 2.1.2 describes a possible conditional distribution (albeit an improper one), the one we used for the particular CNE method in this paper.

Section 2.1.3 describes the posterior distribution.

We wish to be able to model a broad class of prior knowledge types in the form of a manageable prior probability distribution P (G) for the network.

Let us first focus on three common types of prior knowledge: knowledge about the overall network density, knowledge about the individual node degrees, and knowledge about the edge density within or between particular subsets of the nodes (e.g. for multipartite networks).

Each of these can be expressed as sets of constraints on the expectations of the sum of various subsets S ??? V 2 of elements from the adjacency matrix: E {i,j}???S a ij = {i,j}???S?? ij , where the expectation is taken w.r.t.

the sought prior distribution P (G).

In the 1 st case, S = V 2 ; in the 2 nd case, S = {(i, j)|j ??? V, j = i} for information on the degree of node i; and in the 3 rd case S = {(i, j)|i ??? A, j ??? B, i = j} for specified sets A, B ??? V .Such constraints do not determine P (G) fully, so we determine P (G) as the distribution with maximum entropy from all distributions satisfying all these constraints.

Adriaens et al. (2017); van Leeuwen et al. (2016) showed that finding this distribution is a convex optimization problem that can be solved efficiently, particularly for sparse networks.

They also showed that the resulting distribution is a product of independent Bernoulli distributions, one for each element of the adjacency matrix: DISPLAYFORM0 where P ij ??? [0, 1] is the probability that {i, j} is linked in the network under this distribution.

They showed that all these P ij can be expressed in terms of a limited number of parameters, namely the unique Lagrange multipliers for the prior knowledge constraints in the maximum entropy problem.

In practice, the number of such unique Lagrange multipliers is far smaller than n.

The three cases discussed above are merely examples of how constraints on the expectation of subsets of the elements of the adjacency matrix can be useful in practice.

For example, if nodes are ordered in some way (e.g. according to time), it could be used to express the fact that nodes are connected only to nodes that are not too distant in that ordering.

Moreover, the above results continue to hold for constraints that are on weighted linear combinations of elements of the adjacency matrix.

This makes it possible to express other kinds of prior knowledge, e.g. on the relation between connectedness and distance in a node order (if provided), or on the network's (degree) assortativity.

A detailed discussion and empirical analysis of such alternatives is deferred to further work.

We now move on to postulating the conditional density P (X|G).

Clearly, any rotation or translation of an embedding should be considered equally good, as we are only interested in distances between pairs of nodes in the embedding.

Thus, the pairwise distances between points, denoted as d ij DISPLAYFORM0 , must form a set of sufficient statistics.

The density should also reflect the fact that connected node pairs tend to be embedded to nearby points, while disconnected node pairs tend to be embedded to more distant points.

Let us focus initially on the marginal density of d ij conditioned on G. The proposed model assumes that given a ij (i.e. knowledge of whether {i, j} ??? E or not), d ij is conditionally independent of the rest of the adjacency matrix.

More specifically, we model the conditional distribution for the distances d ij given {i, j} ??? E as half-normal N + (Leone et al., 1961) with spread parameter ?? 1 > 0: DISPLAYFORM1 and the distribution of distances d kl with {k, l} ??? E as half-normal with spread parameter ?? 2 > ?? 1 : DISPLAYFORM2 The choice of 0 < ?? 1 < ?? 2 will ensure the embedding reflects the neighborhood proximity of the network.

Indeed, the differences between the embedded nodes that are not connected in the network are expected to be larger than the differences between the embedding of connected nodes.

Without losing generality (as it merely fixes the scale), we set ?? 1 = 1 through out this paper.

It is clear that the distances d ij cannot be independent of each other (e.g. the triangle inequality entails a restriction of the range of d ij given the values of d ik and d jk for some k).

Nevertheless, akin to Naive Bayes, we still model the joint distribution of all distances (and thus of the embedding X up to a rotation/translation) as the product of the marginal densities for all pairwise distances: DISPLAYFORM3 This is an improper density function, due to the constraints imposed by Euclidean geometry.

Indeed, certain combinations of pairwise distances should be assigned a probability 0 as they are geometrically impossible.

As a result, p(X|G) is also not properly normalized.

Yet, even though p(X|G) is improper, it can still be used to derive a properly normalized posterior for G as detailed next.

The (also improper) marginal density p(X) can now be computed as: DISPLAYFORM0 We now have all ingredients to compute the posterior of the network conditioned on the embedding by a simple application of Bayes' rule:

DISPLAYFORM1 This is the likelihood function to be maximized in order to get the ML embedding.

Note that, although it was derived using the improper density function p(X|G), thanks to the normalization with the (equally improper) p(X), this is indeed a properly normalized distribution.

Maximizing the likelihood function P (G|X) is a non-convex optimization problem.

We propose to solve it using a block stochastic gradient descent approach, explained below.

The gradient of the likelihood function (Eq. 6) with respect to the embedding x i of node i is: DISPLAYFORM0 As DISPLAYFORM1 < 0, the first summation pulls the embedding of node i towards embeddings of the nodes it is connected to in G. Moreover, if the current prediction of the link P (a ij = 1|X) is small (i.e., if P (a ij = 0|X) is large), the pulling effect will be larger.

Similarly, the second summation pushes x i away from the embeddings of unconnected nodes, and more strongly so if the current prediction of a link between these two unconnected nodes P (a ij = 1|X) is larger.

The magnitudes of the gradient terms are also affected by parameter ?? 2 and prior P (G): a large ?? 2 gives stronger push and pulling effect.

In our quantitative experiments we always set ?? 2 = 2.Computing this gradient w.r.t.

a particular node's embedding requires computing the pairwise differences between n proposed d-dim embedding vectors, with time complexity O(n 2 d) and space complexity O(nd).

This is computationally demanding for mainstream hardware even for networks of sizes of the order n = 1000 and dimensionalities of the order d = 10, and prohibitive beyond that.

To address this issue, we approximate both summations in the objective by sampling k < n/2 terms from each.

This amounts to uniformly sampling k nodes from the set of connected nodes (where a ij = 1), and k from the set of unconnected nodes (where a ij = 0).

Note that each of the terms is bound in norm by the diameter of the embedding, as the other factors are bound by 1 for ?? 1 = 1, ?? 1 < ?? 2 .

If the diameter were bounded, a simple application of Hoeffding's inequality would demonstrate that this average is sharply concentrated around its expectation, and is thus a suitable approximation.

Although there is no prior bound that holds with guarantee on the diameter of the embedding, this does shed some light on why this approach works well in practice.

The choice of k will in practice be motivated by computational constraints.

In our experiments we set it equal or similar to the largest degree, such that the first term is computed exactly.

We first evaluate the network representation obtained by CNE on downstream tasks typically used for evaluating NE methods: link prediction for links and multi-label classification for nodes.

Then, we illustrate how to use CNE to visually explore multi-relational data.

For the quantitative evaluations, we compare CNE against a panel of state-of-the-art baselines for NE: Deepwalk BID18 , LINE BID20 , node2vec , metapath2vec++ , and struc2vec BID19 .

TAB0 lists the networks used in the experiments.

A brief discussion of the methods and the networks is given in the supplement.

For all methods we used their default parameter settings reported in the original papers and with d = 128.

For node2vec, the hyperparameters p and q are tuned over a grid p, q ??? {0.25, 0.05, 1, 2, 4} using 10-fold cross validation.

We repeat our experiments for 10 times with different random seeds.

The final scores are averaged over the 10 repetitions.

In link prediction, we randomly remove 50% of the links of the network while keeping it connected.

The remaining network is thus used for training the embedding, while the removed links (positive links, labeled 1) are used as a part of the test set.

Then, the test set is topped up by an equal number of negative links (labeled 0) randomly drawn from the original network.

In each repetition of the experiment, the node indices are shuffled so as to obtain different train-test splits.

We compare CNE with other methods based on the area under the ROC curve (AUC).

The methods are evaluated against all datasets mentioned in the previous section.

CNE typically works well with small dimensionality d and sample size k. In this experiment we set d = 8 and k = 50.

Only for the two largest networks (arXiv and Gowalla), we increase the dimensionality to d = 16 to reduce underfitting.

To calculate AUC, we first compute the posterior P (a ij = 1|X train ) of the test links based on the embedding X train learned on the training network.

Then the AUC score is computed by comparing the posterior probability of the test links and their true labels.

In this task we first compare CNE against four simple baselines : Common Neighbors (|N(i) ??? N(j)|), Jaccard Similarity ( DISPLAYFORM0 log |N(t)| ), and Preferential Attachment (|N(i)| ?? |N(j)|).

These baselines are neighborhood based node similarity measures.

We first compute pairwise similarity on the training network.

Then from the computed similarities we obtain scores for testing links as the similarity between the two ending nodes.

Those scores are then used to compute the AUC against the true labels.

For the NE baselines, we perform link prediction using logistic regression based on the link representation derived from the node embedding X train .

The link representation is computed by applying the Hadamard operator (element wise multiplication) on the node representation x i and x j , which is reported to give good results .

Then the AUC score is computed by comparing the link probability (from logistic regression) of the test links with their true labels.

Results The link prediction results are shown in TAB1 .

Even with a uniform prior (i.e. prior knowledge only on the overall density), CNE performs better than all baselines on 5 of the 7 networks.

With a degree prior, however, CNE outperforms all baselines on all networks.

We attribute this to the fact that the degree prior encodes information which is hard to encode using a metric embedding alone.

For the multi-relational dataset studentdb, metapath2vec++, which is designed for heterogeneous data, outperforms other baselines but not CNE (regardless of the prior information).

Moreover, CNE is capable of encoding the knowledge of the block structure of this multi-relational network as a prior, with each block corresponding to one node type.

Doing this improves the AUC further by 3.91% versus CNE with degree prior (from 94.39% to 98.30%; i.e., a 70% reduction in error).In terms of runtime, over the seven datasets CNE is fastest in two cases, 12% slower than the fastest (metapath2vec++) in one case, and takes approximately twice as long in the four other cases (also metapath2vec++).

Detailed runtime results can be found in the supplementary material.

We performed multi-label classification on the following networks: BlogCatalog, PPI, and Wikipedia.

Detailed results are given in the supplement, while TAB2 contains an excerpt of the results.

All baselines are evaluated in a standard logistic regression (LR) setup BID18 .When using logistic regression also on the CNE embeddings, CNE performs on-par, but not particularly well (row CNE-LR).

This should not be a surprise though, as potentially relevant information encoded by the prior (the degrees) will not be reflected in the embedding.

However, multi-label classification can easily be cast as a link prediction problem, by adding to the network a node for each label, with a link to each node to which the label applies.

Predicting a label for a node then amounts to predicting a link to that label node.

To evaluate this strategy, we train an embedding on the original network plus half the label links, while the other half of the label links is held out for testing.

For the baselines, this link prediction setup does not lead to consistent improvements (see supplement), but for CNE it does (row CNE-LP, where LP stands for Link Prediction, in TAB2 ).

On Micro-F 1 it is best or once close second best (after LINE with LR, see TAB2 ), and on Macro-F 1 it greatly outperforms any other method, suggesting improved performance mainly on the less frequent labels.

Here we qualitatively evaluate CNE's ability to facilitate visual exploration of multi-relational data, and how a suitable choice of the prior can help with this.

To this end, we use CNE to embed the studentdb dataset directly into 2-dimensional space.

As a larger ?? 2 in general appears to give better visual separation between node clusters, we set ?? 2 = 15.For comparison, we first apply CNE with uniform prior (overall network density).

The resulting embedding FIG0 clearly separates bachelor student/courses/program nodes (upper) from the master's nodes (lower).

Also observe that the embedding is strongly affected by the node degrees (coded as marker size = log degree): high degree nodes flock together in the center.

E.g., these are students who interact with many other smaller degree nodes (courses/programs).

Although there are no direct links between program nodes (green) and course nodes (blue), the students (red) that connect them are pulling courses towards the corresponding program and pushing away other courses.

Next, we encode the individual node degrees as prior.

As in this case the degree information is known, the embedding in addition shows the courses grouped around different programs, e.g.: "Bachelor Program" is close to course "Calculus"; "Master Program Computer Network" is close to course "Seminar Computer Network"; "Master Program Database" is close to course "Database Security"; "Master Program Software Engineering" is close to courses "Software Testing".Thus, although this last evaluation remains qualitative and preliminary, it confirms that CNE with a suitable prior can create embeddings that clearly convey information in addition to the given prior.

NE methods typically have three components (Hamilton et al., 2017): (1) A similarity measure between nodes, (2) A metric in embedding space, (3) A loss function comparing proximity between nodes in embedding space with the similarity in the network.

Early NE methods such as Laplacian Eigenmaps BID3 , Graph factorization BID2 , GraRep BID5 , and HOPE (Ou et al., 2016) optimize mean-squared-error loss between Euclidean distance or inner product based proximity and link based (adjacency matrix) similarity in the network.

Recently, a few NE methods define node similarity based on paths.

Those paths are generated using either the adjacency matrix (LINE, Tang et al., 2015) or random walks (Deepwalk, BID18 , node2vec, Grover & Leskovec 2016 , methapath2vec++, Dong et al. 2017 , and struc2vec Ribeiro et al. 2017 .

Path based embedding methods typically use inner products as proximity measure in the embedding space and optimize a cross-entropy loss.

The recent struc2vec method BID19 uses a node similarity measure that explicitly builds on structural network properties.

CNE, unlike the aforementioned methods, unifies the proximity in embeddings space and node similarity using a probabilistic measure.

This allows CNE to find a more informative ML embedding.

The question of how to visualize networks on digital screens has been studied for a long time.

Recently there has been an uplift in methods to embed networks in a 'small' number of dimensions, where small means small as compared to the number of nodes, yet typically much larger than two.

These methods enable most machine learning methods to readily apply to tasks on networks, such as node classification or network partitioning.

Popular methods include node2vec , where for example the default output dimensionality is 128.

It is not designed for direct use in visualization, and typically one would fit a higher-dimensional embedding and then apply dimensionality reduction, such as PCA (Peason, 1901) or t-SNE (Maaten & Hinton, 2008) to visualize the data.

CNE finds meaningful 2-d embeddings that can be visualized directly.

Besides, CNE gives a visualization that conveys maximum information in addition to prior knowledge about the network.

The literature on NE has so far considered embeddings as tools that are used on their own.

Yet, Euclidean embeddings are unable to accurately reflect certain kinds of network topologies, such that this approach is inevitably limited.

We proposed the notion of Conditional Network Embeddings (CNEs), which seeks an embedding of a network that maximally adds information with respect to certain given prior knowledge about the network.

This prior knowledge can encode information about the network that cannot be represented well by means of an embedding.

We implemented this conceptually novel idea in a new algorithm based on a simple probabilistic model for the joint of the data and the network, which scales similarly to state-of-the-art NE approaches.

The empirical evaluation of this algorithm confirms our intuition that the combination of structural prior knowledge and a Euclidean embedding is extremely powerful.

This is confirmed empirically for both the tasks of link prediction and multi-label classification, where CNE outperforms a range of state-of-the-art baselines on a wide range of networks.

In our future work we intend to investigate other models implementing the idea of conditional NEs, alternative and more scalable optimization strategies, as well as the use of other types of structural information as prior knowledge on the network.

Denote the Euclidean distance between two points as d ij ||x i ??? x j || 2 .

The derivative of d ij with respect to embedding x i of node i reads: DISPLAYFORM0 Then the derivative of the log posterior with respect to x i is given by: DISPLAYFORM1 , we can compute the partial derivative ??? log(P (G|X)) ???dij for {i, j} ??? E as: DISPLAYFORM2 Similarly, the partial derivative DISPLAYFORM3 for {i, j} / ??? E reads: DISPLAYFORM4 The partial derivatives ???Nmn,??Pmn ???dij are nonzero only when m = i and n = j, which gives the final gradient:??? xi log (P (G|X)) = 2 j:{i,j}???E Figure 1: The posterior distribution P (a ij = 1|X) and P (a ij = 0|X) with different prior probability P ij and ?? 2 2 DERIVING THE LOG PROBABILITY OF POSTERIOR P (G|X) DISPLAYFORM5 DISPLAYFORM6 3 EFFECTS OF THE ?? 1 AND ?? 2 PARAMETERS CNE seeks the embedding X that maximizes the likelihood P (G|X) for given G. To understand the effect of parameter ?? 1 and ?? 2 we plot the posterior P (a ij = 1|X) as well as P (a ij = 0|X) in FIG0 .

The plot shows a large ?? 2 corresponds to more extreme minima of the objective function (Fig1a), thus results in stronger push and pulling effect in the optimization.

Large link probability in the network prior further strengthen the pushing and pulling effects FIG0 .

The flat area in FIG0 (?? 2 = 10) allows connected nodes to keep some small distance from each other, and larger ?? 2 also allows larger corrections to the prior probabilities (both FIG0 , but also makes the optimization problem harder.

We used the following baselines in the experiments:??? Deepwalk BID18 : This embedding algorithm learns embedding based on the similarities between nodes.

The proximities are measured by random walks.

The transition probability of walking from one node to all its neighbors are the same and are based on one-hop connectivity.??? LINE BID20 : Instead of random walks, this algorithm defines similarity between nodes based on first and second order adjacencies of the given network.??? node2vec : This is again based on random walks.

In addition to its predecessors, it offers two parameters p, q that interpolates the importance of BFS and DFS like random walk in the learning.??? metapath2vec++ : This approach is developed for heterogeneous NE, namely, the nodes belong to different node types.

methapath2vec++ performs random walks by hopping from a node form one type to a node from another type.

It also utilizes the node type information in the softmax based objective function.??? struc2vec BID19 : The method first measures the structural information by computing pairwise similarity between nodes using a range of neighborhood sizes.

This results in a multilayer weighted graph where the edge weights on the same layer are derived from the node similarity computed on one neighborhood size.

Then the embedding is constructed by a random walk strategy that navigates the multilayer graph.

We used the following commonly used benchmark networks in the experiments:??? Facebook BID15 : In this network, nodes are the users and links represent the friendships between the users.

The network has 4,039 nodes and 88,234 links.??? arXiv ASTRO-PH BID15 : In this network nodes represent authors of papers submitted to arXiv.

The links represents the collaborations: two authors are connected if they co-authored at least one paper.

The network has 18,722 nodes and 198,110 links.??? studentdb : This is a snapshot of the student database from the University of Antwerp's Computer Science department.

There are 403 nodes that belong to one of the following node types including: course, student, professor, program, track, contract, and room.

There 3429 links that are the binary relationships between the nodes: student-in-track, student-in-program, student-in-contract, student-take-course, professorteach-course, course-in-room.

The database schema is given in FIG2 .???

Gowalla : This is a undirected location-based friendship network.

The network has 196,591 nodes, 950,327 links.??? BlogCatalog BID22 ): This social network contains nodes representing bloggers and links representing their relations with other bloggers.

The labels are the bloggers' interests inferred from the meta data.

The network has 10,312 nodes, 333,983 links, and 39 labels (used for multi-label classifications).??? Protein-Protein Interactions (PPI) : A subnetwork of the PPI network for Homo Sapiens.

The subnetwork has 3,890 nodes, 76,584 links, and 50 labels.??? Wikipedia BID16 : This network contains nodes representing words and links representing the co-occurrence of words in Wikipedia pages.

The labels represents the inferred Part-of-Speech tags BID21 .

The network has 4,777 nodes, 184,812 links, and 40 different labels.

In the multi-label classification setting, each node is assigned one or more labels.

For training, 50% of the nodes and all their labels are used for training.

The labels of the remaining nodes need to be predicted.

We train CNE and baselines based on the full network.

Then 50% of the nodes are randomly selected to train a L2 regularized logistic regression classifier.

The regularization strength parameter of the classifier is trained with 10-fold cross-validation (CV) on the training data.

We report the Macro-F 1 and Micro-F 1 based on the predictions.

For the logistic regression classifier (sklearn, BID17 we require every fold to have at least one positive and one negative label and we removed the labels that occur fewer than 10 times (number of folds in CV) in the data.

The detailed results of this approach based on logistic regression are shown in the upper half of TAB0 .

For CNE (written as CNE-LR to emphasize logistic regression was used for classifying), the embeddings are obtained with d = 32 and k = 150 (without optimizing).

Somewhat surprisingly, CNE still performs in line with the state-of-the-art graph embedded methods, although without improving on them (on BlogCatalog, CNE performs third out of five methods, in PPI and Wikipedia it performs fourth out of five).

This is surprising, given the fact that CNE yields embeddings that, by design, do not reflect certain information about the nodes that may be useful in classifying (here, their degree).Multi-label classification can however be cast as a link prediction problem-a task we know CNE performs well at.

To do this, we insert a node into the network corresponding to each of labels, and link the original nodes to the label nodes if they have that label.

We can then employ link prediction, exactly as in the link prediction case (training on the full network, but with only 50% of the edges between original nodes and label nodes, and the other half for testing), to do multi-label classification.

For CNE, besides a degree prior, we can encode a 'block' prior which encodes the average connectivity between original nodes-original nodes, original nodes-labels, and labels-labels (which is zero, as labels are not connected to each other).

Note that this approach means that also neighborhood-based link prediction methods can be used for multi-label classification.

The detailed results of this link prediction approach to multi-label classification are shown in the lower half of TAB0 .

CNE-LP (block+degree) (with LP to indicate it is based on link prediction) consistently outperforms all baselines on Macro-F 1 , while on Micro-F 1 it is best on two datasets (BlogCatalog and PPI), and close second-best on one (Wikipedia).

We note that while the benefit of this link prediction approach to multi-label classification is clear (and unsurprising) for CNE, there is no consistent benefit to other methods.

This shows that the superior performance of CNE-LP for multi-label classification is not (or at least not exclusively) thanks to the link prediction approach, but at least in part also thanks to a more informative embedding when considered in combination with the prior.

We compare the runtime (in second) of CNE with other baselines in this section.

We use the parameters settings in link prediction task for all methods.

Namely, for CNE, we set d = 8 (For arXiv k = 16 to reduce underfitting) and k = 50.

We set stopping criterion of CNE ||??? X || ??? < 10 ???2 or maxIter < 250 (whichever is met first).

These stopping criteria yield embeddings with the same performance in link prediction tasks as reported in the paper.

For other methods, we use the default setting as reported in their original paper.

The hyper-parameters p, q of node2vec are tuned using TAB1 summarizes the runtime of all methods against all datasets we used in the paper.

Over the seven datasets CNE is fastest in two cases, 12% slower than the fastest in one case (metapath2vec++), and approximately twice slower in the four other cases (also metapath2vec++).

<|TLDR|>

@highlight

We introduce a network embedding method that accounts for prior information about the network, yielding superior empirical performance.

@highlight

The paper proposed to use a prior distribution to constraint the network embedding, for the formulation this paper used very restricted Gaussian distributions.

@highlight

Proposes learning unsupervised node embeddings by considering the structural properties of networks.
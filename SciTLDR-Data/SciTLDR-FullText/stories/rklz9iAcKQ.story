We present Deep Graph Infomax (DGI), a general approach for learning node representations within graph-structured data in an unsupervised manner.

DGI relies on maximizing mutual information between patch representations and corresponding high-level summaries of graphs---both derived using established graph convolutional network architectures.

The learnt patch representations summarize subgraphs centered around nodes of interest, and can thus be reused for downstream node-wise learning tasks.

In contrast to most prior approaches to unsupervised learning with GCNs, DGI does not rely on random walk objectives, and is readily applicable to both transductive and inductive learning setups.

We demonstrate competitive performance on a variety of node classification benchmarks, which at times even exceeds the performance of supervised learning.

encoder models based on graph convolutions BID10 , it is unclear whether randomwalk objectives actually provide any useful signal, as these encoders already enforce an inductive bias that neighboring nodes have similar representations.

In this work, we propose an alternative objective for unsupervised graph learning that is based upon mutual information, rather than random walks.

Recently, scalable estimation of mutual information was made both possible and practical through Mutual Information Neural Estimation (MINE, Belghazi et al., 2018) , which relies on training a statistics network as a classifier of samples coming from the joint distribution of two random variables and their product of marginals.

Following on MINE, introduced Deep InfoMax (DIM) for learning representations of highdimensional data.

DIM trains an encoder model to maximize the mutual information between a high-level "global" representation and "local" parts of the input (such as patches of an image).

This encourages the encoder to carry the type of information that is present in all locations (and thus are globally relevant), such as would be the case of a class label.

DIM relies heavily on convolutional neural network structure in the context of image data, and to our knowledge, no work has applied mutual information maximization to graph-structured inputs.

Here, we adapt ideas from DIM to the graph domain, which can be thought of as having a more general type of structure than the ones captured by convolutional neural networks.

In the following sections, we introduce our method called Deep Graph Infomax (DGI).

We demonstrate that the representation learned by DGI is consistently competitive on both transductive and inductive classification tasks, often outperforming both supervised and unsupervised strong baselines in our experiments.

Contrastive methods.

An important approach for unsupervised learning of representations is to train an encoder to be contrastive between representations that capture statistical dependencies of interest and those that do not.

For example, a contrastive approach may employ a scoring function, training the encoder to increase the score on "real" input (a.k.a, positive examples) and decrease the score on "fake" input (a.k.a., negative samples).

Contrastive methods are central to many popular word-embedding methods BID6 BID27 BID26 , but they are found in many unsupervised algorithms for learning representations of graphstructured input as well.

There are many ways to score a representation, but in the graph literature the most common techniques use classification BID32 BID13 BID24 BID16 , though other scoring functions are used BID9 BID2 .

DGI is also contrastive in this respect, as our objective is based on classifying local-global pairs and negative-sampled counterparts.

Sampling strategies.

A key implementation detail to contrastive methods is how to draw positive and negative samples.

The prior work above on unsupervised graph representation learning relies on a local contrastive loss (enforcing proximal nodes to have similar embeddings).

Positive samples typically correspond to pairs of nodes that appear together within short random walks in the graph-from a language modelling perspective, effectively treating nodes as words and random walks as sentences.

Recent work by BID2 uses node-anchored sampling as an alternative.

The negative sampling for these methods is primarily based on sampling of random pairs, with recent work adapting this approach to use a curriculum-based negative sampling scheme (with progressively "closer" negative examples; BID45 or introducing an adversary to select the negative examples BID3 .Predictive coding.

Contrastive predictive coding (CPC, Oord et al., 2018) is another method for learning deep representations based on mutual information maximization.

Like the models above, CPC is also contrastive, in this case using an estimate of the conditional density (in the form of noise contrastive estimation, BID14 as the scoring function.

However, unlike our approach, CPC and the graph methods above are all predictive: the contrastive objective effectively trains a predictor between structurally-specified parts of the input (e.g., between neighboring node pairs or between a node and its neighborhood).

Our approach differs in that we contrast global / local parts of a graph simultaneously, where the global variable is computed from all local variables.

To the best of our knowledge, the sole prior works that instead focuses on contrasting "global" and "local" representations on graphs do so via (auto-)encoding objectives on the adjacency matrix BID41 ) and incorporation of community-level constraints into node embeddings BID42 .

Both methods rely on matrix factorization-style losses and are thus not scalable to larger graphs.

In this section, we will present the Deep Graph Infomax method in a top-down fashion: starting with an abstract overview of our specific unsupervised learning setup, followed by an exposition of the objective function optimized by our method, and concluding by enumerating all the steps of our procedure in a single-graph setting.

We assume a generic graph-based unsupervised machine learning setup: we are provided with a set of node features, X = { x 1 , x 2 , . . .

, x N }, where N is the number of nodes in the graph and x i ∈ R F represents the features of node i. We are also provided with relational information between these nodes in the form of an adjacency matrix, A ∈ R N ×N .

While A may consist of arbitrary real numbers (or even arbitrary edge features), in all our experiments we will assume the graphs to be unweighted, i.e. A ij = 1 if there exists an edge i → j in the graph and A ij = 0 otherwise.

Our objective is to learn an encoder, E : DISPLAYFORM0 for each node i.

These representations may then be retrieved and used for downstream tasks, such as node classification.

Here we will focus on graph convolutional encoders-a flexible class of node embedding architectures, which generate node representations by repeated aggregation over local node neighborhoods BID10 .

A key consequence is that the produced node embeddings, h i , summarize a patch of the graph centered around node i rather than just the node itself.

In what follows, we will often refer to h i as patch representations to emphasize this point.

Our approach to learning the encoder relies on maximizing local mutual information-that is, we seek to obtain node (i.e., local) representations that capture the global information content of the entire graph, represented by a summary vector, s.

In order to obtain the graph-level summary vectors, s, we leverage a readout function, R : DISPLAYFORM0 , and use it to summarize the obtained patch representations into a graph-level representation; i.e., s = R(E(X, A)).As a proxy for maximizing the local mutual information, we employ a discriminator, D : DISPLAYFORM1 represents the probability scores assigned to this patch-summary pair (should be higher for patches contained within the summary).Negative samples for D are provided by pairing the summary s from (X, A) with patch representations h j of an alternative graph, ( X, A).

In a multi-graph setting, such graphs may be obtained as other elements of a training set.

However, for a single graph, an explicit (stochastic) corruption function, C : DISPLAYFORM2 is required to obtain a negative example from the original graph, i.e. ( X, A) = C(X, A).

The choice of the negative sampling procedure will govern the specific kinds of structural information that is desirable to be captured as a byproduct of this maximization.

For the objective, we follow the intuitions from Deep InfoMax (DIM, Hjelm et al., 2018) and use a noise-contrastive type objective with a standard binary cross-entropy (BCE) loss between the samples from the joint (positive examples) and the product of marginals (negative examples).

Following their work, we use the following objective DISPLAYFORM3 This approach effectively maximizes mutual information between h i and s, based on the JensenShannon divergence 2 between the joint and the product of marginals.

As all of the derived patch representations are driven to preserve mutual information with the global graph summary, this allows for discovering and preserving similarities on the patch-level-for example, distant nodes with similar structural roles (which are known to be a strong predictor for many node classification tasks; BID8 .

Note that this is a "reversed" version of the argument given by : for node classification, our aim is for the patches to establish links to similar patches across the graph, rather than enforcing the summary to contain all of these similarities (however, both of these effects should in principle occur simultaneously).

We now provide some intuition that connects the classification error of our discriminator to mutual information maximization on graph representations.

DISPLAYFORM0 be a set of node representations drawn from an empirical probability distribution of graphs, p(X), with finite number of elements, |X|, such that p( DISPLAYFORM1 be a deterministic readout function on graphs and s (k) = R(X (k) ) be the summary vector of the k-th graph, with marginal distribution p( s).

The optimal classifier between the joint distribution p(X, s) and the product of marginals p(X)p( s), assuming class balance, has an error rate upper bounded by DISPLAYFORM2 the set of all graphs in the input set that are mapped to s (k) by R, i.e. DISPLAYFORM3 As R(·) is deterministic, samples from the joint, (X (k) , s (k) ) are drawn from the product of marginals with probability p( s (k) )p(X (k) ), which decomposes into: DISPLAYFORM4 This probability ratio is maximized at 1 when DISPLAYFORM5 .

The probability of drawing any sample of the joint from the product of marginals is then bounded above by DISPLAYFORM6 .

As the probability of drawing ( DISPLAYFORM7 , we know that classifying these samples as coming from the joint has a lower error than classifying them as coming from the product of marginals.

The error rate of such a classifier is then the probability of drawing a sample from the joint as a sample from product of marginals under the mixture probability, which we can bound by Err ≤ DISPLAYFORM8 , with the upper bound achieved, as above, when R(·) is injective for all elements of {X (k) }.It may be useful to note that DISPLAYFORM9 The first result is obtained via a trivial application of Jensen's inequality, while the other extreme is reached only in the edge case of a constant readout function (when every example from the joint is also an example from the product of marginals, so no classifier performs better than chance).

Corollary 1.

From now on, assume that the readout function used, R, is injective.

Assume the number of allowable states in the space of s, | s|, is greater than or equal to |X|.

Then, for s , the optimal summary under the classification error of an optimal classifier between the joint and the product of marginals, it holds that | s | = |X|.Proof.

By injectivity of R, we know that s = argmin s Err * .

As the upper error bound, Err * , is a simple geometric sum, we know that this is minimized when p( s (k) ) is uniform.

As R(·) is deterministic, this implies that each potential summary state would need to be used at least once.

Combined with the condition | s| ≥ |X|, we conclude that the optimum has | s | = |X|.

Theorem 1.

s = argmax s MI(X; s), where MI is mutual information.

Proof.

This follows from the fact that the mutual information is invariant under invertible transforms.

As | s | = |X| and R is injective, it has an inverse function, R −1.

It follows then that, for any s, MI(X; s) ≤ H(X) = MI(X; X) = MI(X; R(X)) = MI(X; s ), where H is entropy.

Theorem 1 shows that for finite input sets and suitable deterministic functions, minimizing the classification error in the discriminator can be used to maximize the mutual information between the input and output.

However, as was shown in , this objective alone is not enough to learn useful representations.

As in their work, we discriminate between the global summary vector and local high-level representations.

DISPLAYFORM10 be the neighborhood of the node i in the k-th graph that collectively maps to its high-level features, DISPLAYFORM11 where n is the neighborhood function that returns the set of neighborhood indices of node i for graph X (k) , and E is a deterministic encoder function.

Let us assume that DISPLAYFORM12 Proof.

Given our assumption of |X i | = | s|, there exists an inverse X i = R −1 ( s), and therefore DISPLAYFORM13 ) mapping s to h i .

The optimal classifier between the joint p( h i , s) and the product of marginals p( h i )p( s) then has (by Lemma 1) an error rate upper bound of DISPLAYFORM14 .

Therefore (as in Corollary 1), for the optimal DISPLAYFORM15 which by the same arguments as in Theorem 1 maximizes the mutual information between the neighborhood and high-level features, MI(X DISPLAYFORM16 This motivates our use of a classifier between samples from the joint and the product of marginals, and using the binary cross-entropy (BCE) loss to optimize this classifier is well-understood in the context of neural network optimization.

Assuming the single-graph setup (i.e., (X, A) provided as input), we will now summarize the steps of the Deep Graph Infomax procedure:1.

Sample a negative example by using the corruption function: ( X, A) ∼ C(X, A).2.

Obtain patch representations, h i for the input graph by passing it through the encoder: DISPLAYFORM0 3.

Obtain patch representations, h j for the negative example by passing it through the encoder: DISPLAYFORM1 .

Summarize the input graph by passing its patch representations through the readout function: s = R(H).

This algorithm is fully summarized by Figure 1 .

Figure 1 : A high-level overview of Deep Graph Infomax.

Refer to Section 3.4 for more details.

DISPLAYFORM0

We have assessed the benefits of the representation learnt by the DGI encoder on a variety of node classification tasks (transductive as well as inductive), obtaining competitive results.

In each case, DGI was used to learn patch representations in a fully unsupervised manner, followed by evaluating the node-level classification utility of these representations.

This was performed by directly using these representations to train and test a simple linear (logistic regression) classifier.

We follow the experimental setup described in BID23 and BID15 on the following benchmark tasks: (1) classifying research papers into topics on the Cora, Citeseer and Pubmed citation networks BID35 ; (2) predicting the community structure of a social network modeled with Reddit posts; and (3) classifying protein roles within protein-protein interaction (PPI) networks BID49 , requiring generalisation to unseen networks.

Further information on the datasets may be found in TAB0 and Appendix A.

For each of three experimental settings (transductive learning, inductive learning on large graphs, and multiple graphs), we employed distinct encoders and corruption functions appropriate to that setting (described below).Transductive learning.

For the transductive learning tasks (Cora, Citeseer and Pubmed), our encoder is a one-layer Graph Convolutional Network (GCN) model BID23 , with the following propagation rule: DISPLAYFORM0 whereÂ = A + I N is the adjacency matrix with inserted self-loops andD is its corresponding degree matrix; i.e.

D ii = jÂ ij .

For the nonlinearity, σ, we have applied the parametric ReLU (PReLU) function BID17 , and Θ ∈ R F ×F is a learnable linear transformation applied to every node, with F = 512 features being computed (specially, F = 256 on Pubmed due to memory limitations).The corruption function used in this setting is designed to encourage the representations to properly encode structural similarities of different nodes in the graph; for this purpose, C preserves the original adjacency matrix ( A = A), whereas the corrupted features, X, are obtained by row-wise shuffling of X. That is, the corrupted graph consists of exactly the same nodes as the original graph, but they are located in different places in the graph, and will therefore receive different patch representations.

We demonstrate DGI is stable to other choices of corruption functions in Appendix C, but we find those that preserve the graph structure result in the strongest features.

Inductive learning on large graphs.

For inductive learning, we may no longer use the GCN update rule in our encoder (as the learned filters rely on a fixed and known adjacency matrix); instead, we apply the mean-pooling propagation rule, as used by GraphSAGE-GCN BID15 : DISPLAYFORM1 with parameters defined as in Equation 3.

Note that multiplying byD DISPLAYFORM2 actually performs a normalized sum (hence the mean-pooling).

While Equation 4 explicitly specifies the adjacency and degree matrices, they are not needed: identical inductive behaviour may be observed by a constant attention mechanism across the node's neighbors, as used by the Const-GAT model BID39 .For Reddit, our encoder is a three-layer mean-pooling model with skip connections BID18 : DISPLAYFORM3 where is featurewise concatenation (i.e. the central node and its neighborhood are handled separately).

We compute F = 512 features in each MP layer, with the PReLU activation for σ.

Given the large scale of the dataset, it will not fit into GPU memory entirely.

Therefore, we use the subsampling approach of BID15 , where a minibatch of nodes is first selected, and then a subgraph centered around each of them is obtained by sampling node neighborhoods with replacement.

Specifically, we sample 10, 10 and 25 neighbors at the first, second and third level, respectively-thus, each subsampled patch has 1 + 10 + 100 + 2500 = 2611 nodes.

Only the computations necessary for deriving the central node i's patch representation, h i , are performed.

These representations are then used to derive the summary vector, s, for the minibatch FIG0 ).

We used minibatches of 256 nodes throughout training.

To define our corruption function in this setting, we use a similar approach as in the transductive tasks, but treat each subsampled patch as a separate graph to be corrupted (i.e., we row-wise shuffle the feature matrices within a subsampled patch).

Note that this may very likely cause the central node's features to be swapped out for a sampled neighbor's features, further encouraging diversity in the negative samples.

The patch representation obtained in the central node is then submitted to the discriminator.

Inductive learning on multiple graphs.

For the PPI dataset, inspired by previous successful supervised architectures BID39 , our encoder is a three-layer mean-pooling model with dense skip connections BID18 BID20 : DISPLAYFORM4 DISPLAYFORM5 DISPLAYFORM6 where W skip is a learnable projection matrix, and MP is as defined in Equation 4.

We compute F = 512 features in each MP layer, using the PReLU activation for σ.

In this multiple-graph setting, we opted to use randomly sampled training graphs as negative examples (i.e., our corruption function simply samples a different graph from the training set).

We found this method to be the most stable, considering that over 40% of the nodes have all-zero features in this dataset.

To further expand the pool of negative examples, we also apply dropout BID36 to the input features of the sampled graph.

We found it beneficial to standardize the learnt embeddings across the training set prior to providing them to the logistic regression model.

Readout, discriminator, and additional training details.

Across all three experimental settings, we employed identical readout functions and discriminator architectures.

For the readout function, we use a simple averaging of all the nodes' features: DISPLAYFORM7 where σ is the logistic sigmoid nonlinearity.

While we have found this readout to perform the best across all our experiments, we assume that its power will diminish with the increase in graph size, and in those cases, more sophisticated readout architectures such as set2vec BID40 or DiffPool BID46 ) are likely to be more appropriate.

The discriminator scores summary-patch representation pairs by applying a simple bilinear scoring function (similar to the scoring used by Oord et al. (2018)): DISPLAYFORM8 Here, W is a learnable scoring matrix and σ is the logistic sigmoid nonlinearity, used to convert scores into probabilities of ( h i , s) being a positive example.

All models are initialized using Glorot initialization BID11 ) and trained to maximize the mutual information provided in Equation 1 on the available nodes (all nodes for the transductive, and training nodes only in the inductive setup) using the Adam SGD optimizer BID22 with an initial learning rate of 0.001 (specially, 10 DISPLAYFORM9 on Reddit).

On the transductive datasets, we use an early stopping strategy on the observed training loss, with a patience of 20 epochs 3 .

On the inductive datasets we train for a fixed number of epochs (150 on Reddit, 20 on PPI).

The results of our comparative evaluation experiments are summarized in TAB1 For the transductive tasks, we report the mean classification accuracy (with standard deviation) on the test nodes of our method after 50 runs of training (followed by logistic regression), and reuse the metrics already reported in BID23 for the performance of DeepWalk and GCN, as well as Label Propagation (LP) BID48 and Planetoid BID44 )-a representative supervised random walk method.

Specially, we provide results for training the logistic regression on raw input features, as well as DeepWalk with the input features concatenated.

Avg.

pooling 0.958 ± 0.001 0.969 ± 0.002For the inductive tasks, we report the micro-averaged F 1 score on the (unseen) test nodes, averaged after 50 runs of training, and reuse the metrics already reported in BID15 for the other techniques.

Specifically, as our setup is unsupervised, we compare against the unsupervised GraphSAGE approaches.

We also provide supervised results for two related architecturesFastGCN BID5 and Avg.

pooling ).Our results demonstrate strong performance being achieved across all five datasets.

We particularly note that the DGI approach is competitive with the results reported for the GCN model with the supervised loss, even exceeding its performance on the Cora and Citeseer datasets.

We assume that these benefits stem from the fact that, indirectly, the DGI approach allows for every node to have access to structural properties of the entire graph, whereas the supervised GCN is limited to only two-layer neighborhoods (by the extreme sparsity of the training signal and the corresponding threat of overfitting).

It should be noted that, while we are capable of outperforming equivalent supervised encoder architectures, our performance still does not surpass the current supervised transductive state of the art (which is held by methods such as GraphSGAN BID7 ).

We further observe that the DGI method successfully outperformed all the competing unsupervised GraphSAGE approaches on the Reddit and PPI datasets-thus verifying the potential of methods based on local mutual information maximization in the inductive node classification domain.

Our Reddit results are competitive with the supervised state of the art, whereas on PPI the gap is still large-we believe this can be attributed to the extreme sparsity of available node features (over 40% of the nodes having all-zero features), that our encoder heavily relies on.

We note that a randomly initialized graph convolutional network may already extract highly useful features and represents a strong baseline-a well-known fact, considering its links to the Weisfeiler- Lehman graph isomorphism test BID43 , that have already been highlighted and analyzed by BID23 and BID15 .

As such, we also provide, as Random-Init, the logistic regression performance on embeddings obtained from a randomly initialized encoder.

Besides demonstrating that DGI is able to further improve on this strong baseline, it particularly reveals that, on the inductive datasets, previous random walk-based negative sampling methods may have been ineffective for learning appropriate features for the classification task.

Lastly, it should be noted that deeper encoders correspond to more pronounced mixing between recovered patch representations, reducing the effective variability of our positive/negative examples' pool.

We believe that this is the reason why shallower architectures performed better on some of the datasets.

While we cannot say that these trends will hold in general, with the DGI loss function we generally found benefits from employing wider, rather than deeper models.

We performed a diverse set of analyses on the embeddings learnt by the DGI algorithm in order to better understand the properties of DGI.

We focus our analysis exclusively on the Cora dataset (as it has the smallest number of nodes, significantly aiding clarity).A standard set of "evolving" t-SNE plots BID25 of the embeddings is given in FIG2 .

As expected given the quantitative results, the learnt embeddings' 2D projections exhibit discernible clustering in the 2D projected space (especially compared to the raw features and Random-Init), which respects the seven topic classes of Cora.

The projection obtains a Silhouette score (Rousseeuw, 1987) of 0.234, which compares favorably with the previous reported score of 0.158 for Embedding Propagation BID9 .We ran further analyses, revealing insights into DGI's mechanism of learning, isolating biased embedding dimensions for pushing the negative example scores down and using the remainder to encode useful information about positive examples.

We leverage these insights to retain competitive performance to the supervised GCN even after half the dimensions are removed from the patch representations provided by the encoder.

These-and several other-qualitative and ablation studies can be found in Appendix B.

We have presented Deep Graph Infomax (DGI), a new approach for learning unsupervised representations on graph-structured data.

By leveraging local mutual information maximization across the graph's patch representations, obtained by powerful graph convolutional architectures, we are able to obtain node embeddings that are mindful of the global structural properties of the graph.

This enables competitive performance across a variety of both transductive and inductive classification tasks, at times even outperforming relevant supervised architectures.

Transductive learning.

We utilize three standard citation network benchmark datasets-Cora, Citeseer and Pubmed BID35 -and closely follow the transductive experimental setup of BID44 .

In all of these datasets, nodes correspond to documents and edges to (undirected) citations.

Node features correspond to elements of a bag-of-words representation of a document.

Each node has a class label.

We allow for only 20 nodes per class to be used for training-however, honouring the transductive setup, the unsupervised learning algorithm has access to all of the nodes' feature vectors.

The predictive power of the learned representations is evaluated on 1000 test nodes.

Inductive learning on large graphs.

We use a large graph dataset (231,443 nodes and 11,606,919 edges) of Reddit posts created during September 2014 (derived and preprocessed as in BID15 ).

The objective is to predict the posts' community ("subreddit"), based on the GloVe embeddings of their content and comments BID31 , as well as metrics such as score or number of comments.

Posts are linked together in the graph if the same user has commented on both.

Reusing the inductive setup of BID15 , posts made in the first 20 days of the month are used for training, while the remaining posts are used for validation or testing and are invisible to the training algorithm.

Inductive learning on multiple graphs.

We make use of a protein-protein interaction (PPI) dataset that consists of graphs corresponding to different human tissues BID49 .

The dataset contains 20 graphs for training, 2 for validation and 2 for testing.

Critically, testing graphs remain completely unobserved during training.

To construct the graphs, we used the preprocessed data provided by BID15 .

Each node has 50 features that are composed of positional gene sets, motif gene sets and immunological signatures.

There are 121 labels for each node set from gene ontology, collected from the Molecular Signatures Database BID37 , and a node can possess several labels simultaneously.

Visualizing discriminator scores.

After obtaining the t-SNE visualizations, we turned our attention to the discriminator-and visualized the scores it attached to various nodes, for both the positive and a (randomly sampled) negative example FIG3 .

From here we can make an interesting observation-within the "clusters" of the learnt embeddings on the positive Cora graph, only a handful of "hot" nodes are selected to receive high discriminator scores.

This suggests that there may be a clear distinction between embedding dimensions used for discrimination and classification, which we more thoroughly investigate in the next paragraph.

In addition, we may observe that, as expected, the model is unable to find any strong structure within a negative example.

Lastly, a few negative examples achieve high discriminator scores-a phenomenon caused by the existence of DISPLAYFORM0 Figure 6: Classification performance (in terms of test accuracy of logistic regression; left) and discriminator performance (in terms of number of poorly discriminated positive/negative examples; right) on the learnt DGI embeddings, after removing a certain number of dimensions from the embedding-either starting with most distinguishing (p ↑) or least distinguishing (p ↓).low-degree nodes in Cora (making the probability of a node ending up in an identical context it had in the positive graph non-negligible).Impact and role of embedding dimensions.

Guided by the previous result, we have visualized the embeddings for the top-scoring positive and negative examples ( FIG4 ).

The analysis revealed existence of distinct dimensions in which both the positive and negative examples are strongly biased.

We hypothesize that, given the random shuffling, the average expected activation of a negative example is zero, and therefore strong biases are required to "push" the example down in the discriminator.

The positive examples may then use the remaining dimensions to both counteract this bias and encode patch similarity.

To substantiate this claim, we order the 512 dimensions based on how distinguishable the positive and negative examples are in them (using p-values obtained from a t-test as a proxy).

We then remove these dimensions from the embedding, respecting this order-either starting from the most distinguishable (p ↑) or least distinguishable dimensions (p ↓)-monitoring how this affects both classification and discriminator performance (Figure 6 ).

The observed trends largely support our hypothesis: if we start by removing the biased dimensions first (p ↓), the classification performance holds up for much longer (allowing us to remove over half of the embedding dimensions while remaining competitive to the supervised GCN), and the positive examples mostly remain correctly discriminated until well over half the dimensions are removed.

Here, we consider alternatives to our corruption function, C, used to produce negative graphs.

We generally find that, for the node classification task, DGI is stable and robust to different strategies.

However, for learning graph features towards other kinds of tasks, the design of appropriate corruption strategies remains an area of open research.

Our corruption function described in Section 4.2 preserves the original adjacency matrix ( A = A) but corrupts the features, X, via row-wise shuffling of X. In this case, the negative graph is constrained to be isomorphic to the positive graph, which should not have to be mandatory.

We can instead produce a negative graph by directly corrupting the adjacency matrix.

Therefore, we first consider an alternative corruption function C which preserves the features ( X = X) but instead adds or removes edges from the adjacency matrix ( A = A).

This is done by sampling, i.i.d., a switch parameter Σ ij , which determines whether to corrupt the adjacency matrix at position (i, j).

Assuming a given corruption rate, ρ, we may define C as performing the following operations: DISPLAYFORM0 DISPLAYFORM1 where ⊕ is the XOR (exclusive OR) operation.

This alternative strategy produces a negative graph with the same features, but different connectivity.

Here, the corruption rate of ρ = 0 corresponds to an unchanged adjacency matrix (i.e. the positive and negative graphs are identical in this case).

In this regime, learning is impossible for the discriminator, and the performance of DGI is in line with a randomly initialized DGI model.

At higher rates of noise, however, DGI produces competitive embeddings.

We also consider simultaneous feature shuffling ( X = X) and adjacency matrix perturbation ( A = A), both as described before.

We find that DGI still learns useful features under this compound corruption strategy-as expected, given that feature shuffling is already equivalent to an (isomorphic) adjacency matrix perturbation.

From both studies, we may observe that a certain lower bound on the positive graph perturbation rate is required to obtain competitive node embeddings for the classification task on Cora.

Furthermore, the features learned for downstream node classification tasks are most powerful when the negative graph has similar levels of connectivity to the positive graph.

The classification performance peaks when the graph is perturbed to a reasonably high level, but remains sparse; i.e. the mixing between the separate 1-step patches is not substantial, and therefore the pool of negative examples is still diverse enough.

Classification performance is impacted only marginally at higher rates of corruption-corresponding to dense negative graphs, and thus a less rich negative example pool-but still considerably outperforming the unsupervised baselines we have considered.

This could be seen as further motivation for relying solely on feature shuffling, without adjacency perturbations-given that feature shuffling is a trivial way to guarantee a diverse set of negative examples, without incurring significant computational costs per epoch.

Classification accuracy for (X, A) corruption X = X GCN Figure 8 : DGI is stable and robust under a corruption function that modifies both the feature matrix (X = X) and the adjacency matrix ( A = A) on the Cora dataset.

Corruption functions that preserve sparsity (ρ ≈ 1 N ) perform the best.

However, DGI still performs well even with large disruptions (where edges are added or removed with probabilities approaching 1).

N.B. log scale used for ρ.

@highlight

A new method for unsupervised representation learning on graphs, relying on maximizing mutual information between local and global representations in a graph. State-of-the-art results, competitive with supervised learning.
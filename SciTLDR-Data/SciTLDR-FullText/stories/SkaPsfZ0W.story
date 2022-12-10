Graph Convolutional Networks (GCNs) are a recently proposed architecture which has had success in semi-supervised learning on graph-structured data.

At the same time, unsupervised learning of graph embeddings has benefited from the information contained in random walks.

In this paper we propose a model, Network of GCNs (N-GCN), which marries these two lines of work.

At its core, N-GCN trains multiple instances of GCNs over node pairs discovered at different distances in random walks, and learns a combination of the instance outputs which optimizes the classification objective.

Our experiments show that our proposed N-GCN model achieves state-of-the-art performance on all of the challenging node classification tasks we consider: Cora, Citeseer, Pubmed, and PPI.

In addition, our proposed method has other desirable properties, including generalization to recently proposed semi-supervised learning methods such as GraphSAGE, allowing us to propose N-SAGE, and resilience to adversarial input perturbations.

Semi-supervised learning on graphs is important in many real-world applications, where the goal is to recover labels for all nodes given only a fraction of labeled ones.

Some applications include social networks, where one wishes to predict user interests, or in health care, where one wishes to predict whether a patient should be screened for cancer.

In many such cases, collecting node labels can be prohibitive.

However, edges between nodes can be easier to obtain, either using an explicit graph (e.g. social network) or implicitly by calculating pairwise similarities (e.g. using a patient-patient similarity kernel, BID19 .Convolutional Neural Networks BID16 learn location-invariant hierarchical filters, enabling significant improvements on Computer Vision tasks BID15 BID23 BID12 .

This success has motivated researchers BID7 to extend convolutions from spatial (i.e. regular lattice) domains to graph-structured (i.e. irregular) domains, yielding a class of algorithms known as Graph Convolutional Networks (GCNs).Formally, we are interested in semi-supervised learning where we are given a graph G = (V, E) with N = |V| nodes; adjacency matrix A; and matrix X ∈ R N ×F of node features.

Labels for only a subset of nodes V L ⊂ V observed.

In general, |V L | |V|.

Our goal is to recover labels for all unlabeled nodes V U = V − V L , using the feature matrix X, the known labels for nodes in V L , and the graph G. In this setting, one treats the graph as the "unsupervised" and labels of V L as the "supervised" portions of the data.

Depicted in FIG2 , our model for semi-supervised node classification builds on the GCN module proposed by BID14 , which operates on the normalized adjacency matrixÂ, as in GCN(Â), whereÂ = D , and D is diagonal matrix of node degrees.

Our proposed extension of GCNs is inspired by the recent advancements in random walk based graph embeddings (e.g. BID22 BID9 BID1 .

We make a Network of GCN modules (N-GCN), feeding each module a different power ofÂ, as in {GCN(Â 0 ), GCN(Â 1 ), GCN(Â 2 ), . . . }.

The k-th power contains statistics from the k-th step of a random walk on the graph.

Therefore, our N-GCN model is able to combine information from various step-sizes.

We then combine the output of all GCN modules into a classification sub-network, and we jointly train all GCN modules and the classification sub-network on the upstream objective, Model architecture, whereÂ is the normalized normalized adjacency matrix, I is the identity matrix, X is node features matrix, and × is matrix-matrix multiply operator.

We calculate K powers of theÂ, feeding each power into r GCNs, along with X. The output of all K × r GCNs can be concatenated along the column dimension, then fed into fully-connected layers, outputting C channels per node, where C is size of label space.

We calculate cross entropy error, between rows prediction N × C with known labels, and use them to update parameters of classification subnetwork and all GCNs.

Right: pre-relu activations after the first fully-connected layer of a 2-layer classification sub-network.

Activations are PCA-ed to 50 dimensions then visualized using t-SNE.semi-supervised node classification.

Weights of the classification sub-network give us insight on how the N-GCN model works.

For instance, in the presence of input perturbations, we observe that the classification sub-network weights shift towards GCN modules utilizing higher powers of the adjacency matrix, effectively widening the "receptive field" of the (spectral) convolutional filters.

We achieve state-of-the-art on several semi-supervised graph learning tasks, showing that explicit random walks enhance the representational power of vanilla GCN's.

The rest of this paper is organized as follows.

Section 2 reviews background work that provides the foundation for this paper.

In Section 3, we describe our proposed method, followed by experimental evaluation in Section 4.

We compare our work with recent closely-related methods in Section 5.

Finally, we conclude with our contributions and future work in Section 6.

Traditional label propagation algorithms BID24 BID5 learn a model that transforms node features into node labels and uses the graph to add a regularizer term: DISPLAYFORM0 where f : R N ×d0 → R N ×C is the model, ∆ is the graph Laplacian, and λ ∈ R is the regularization coefficient hyperparameter.

Graph Convolution BID7 generalizes convolution from Euclidean domains to graphstructured data.

Convolving a "filter" over a signal on graph nodes can be calculated by transforming both the filter and the signal to the Fourier domain, multiplying them, and then transforming the result back into the discrete domain.

The signal transform is achieved by multiplying with the eigenvectors of the graph Laplacian.

The transformation requires a quadratic eigendecomposition of the symmetric Laplacian; however, the low-rank approximation of the eigendecomposition can be calculated using truncated Chebyshev polynomials BID11 .

For instance, BID14 calculates a rank-1 approximation of the decomposition.

They propose a multi-layer Graph Convolutional Networks (GCNs) for semi-supervised graph learning.

Every layer computes the transformation: DISPLAYFORM0 where H (l) ∈ R N ×d l is the input activation matrix to the l-th hidden layer with row H (l)i containing a d l -dimensional feature vector for vertex i ∈ V, and W (l) ∈ R d l ×d l+1 is the layer's trainable weights.

The first hidden layer H (0) is set to the input features X. A softmax on the last layer is used to classify labels.

All layers use the same "normalized adjacency"Â, obtained by the "renormalization trick" utilized by BID14 DISPLAYFORM1 Eq. (2) is a first order approximation of convolving filter W (l) over signal H (l) BID11 BID14 .

The left-multiplication withÂ averages node features with their direct neighbors; this signal is then passed through a non-linearity function σ(·) (e.g, ReLU(z) = max(0, z)).

Successive layers effectively diffuse signals from nodes to neighbors.

Two-layer GCN model can be defined in terms of vertex features X and normalized adjacencyÂ as: DISPLAYFORM2 where the GCN parameters θ = W (0) , W (1) are trained to minimize the cross-entropy error over labeled examples.

The output of the GCN model is a matrix R N ×C , where N is the number of nodes and C is the number of labels.

Each row contains the label scores for one node, assuming there are C classes.

Node Embedding methods represent graph nodes in a continuous vector space.

They learn a dictionary Z ∈ R N ×d , with one d-dimensional embedding per node.

Traditional methods use the adjacency matrix to learn embeddings.

For example, Eigenmaps BID4 calculates the following constrained optimization: DISPLAYFORM0 where I is identity vector.

Skipgram models on text corpora BID20 inspired modern graph embedding methods, which simulate random walks to learn node embeddings BID22 BID9 .

Each random walk generates a sequence of nodes.

Sequences are converted to textual paragraphs, and are passed to a word2vec-style embedding learning algorithm BID20 .

As shown in Abu- BID1 , this learning-by-simulation is equivalent, in expectation, to the decomposition of a random walk co-occurrence statistics matrix D. The expectation on D can be written as: DISPLAYFORM1 where T = D −1 A is the row-normalized transition matrix (a.k.a right-stochastic adjacency matrix), and Q is a "context distribution" that is determined by random walk hyperparameters, such as the length of the random walk.

The expectation therefore weights the importance of one node on another as a function of how well-connected they are, and the distance between them.

The main difference between traditional node embedding methods and random walk methods is the optimization criteria: the former minimizes a loss on representing the adjacency matrix A (see Eq. 4), while the latter minimizes a loss on representing random walk co-occurrence statistics D.

Graph Convolutional Networks and random walk graph embeddings are individually powerful.

BID14 uses GCNs for semi-supervised node classification.

Instead of following tradi-tional methods that use the graph for regularization (e.g. Eq. 4), BID14 use the adjacency matrix for training and inference, effectively diffusing information across edges at all GCN layers (see Eq. 6).

Separately, recent work has showed that random walk statistics can be very powerful for learning an unsupervised representation of nodes that can preserve the structure of the graph BID22 BID9 BID1 .Under special conditions, it is possible for the GCN model to learn random walks.

In particular, consider a two-layer GCN defined in Eq. 6 with the assumption that first-layer activation is identity as σ(z) = z, and weight W (0) is an identity matrix (either explicitly set or learned to satisfy the upstream objective).

Under these two identity conditions, the model reduces to: DISPLAYFORM0 whereÂ 2 can be expanded as: DISPLAYFORM1 By multiplying the adjacency A with the transition matrix T before normalization, the GCN is effectively doing a one-step random walk.

The special conditions described above are not true in practice.

Although stacking hidden GCN layers allows information to flow through graph edges, this flow is indirect as the information goes through feature reduction (matrix multiplication) and a non-linearity (activation function σ(·)).

Therefore, the vanilla GCN cannot directly learn high powers ofÂ, and could struggle with modeling information across distant nodes.

We hypothesize that making the GCN directly operate on random walk statistics will allow the network to better utilize information across distant nodes, in the same way that node embedding methods (e.g. DeepWalk, BID22 ) operating on D are superior to traditional embedding methods operating on the adjacency matrix (e.g. Eigenmaps, BID4 ).

Therefore, in addition to feeding onlyÂ to the GCN model as proposed by BID14 (see Eq. 6), we propose to feed a K-degree polynomial ofÂ to K instantiations of GCN.

Generalizing Eq. FORMULA7 gives: DISPLAYFORM0 We also defineÂ 0 to be the identity matrix.

Similar to BID14 , we add selfconnections and convert directed graphs to undirected ones, makingÂ and henceÂ k symmetric matrices.

The eigendecomposition of symmetric matrices is real.

Therefore, the low-rank approximation of the eigendecomposition BID11 is still valid, and a one layer of BID14 utilizingÂ k should still approximate multiplication in the Fourier domain.

Consider DISPLAYFORM0 where the v-th row describes a latent representation of that particular GCN for node v ∈ V, and where C k is the latent dimensionality.

Though C k can be different for each GCN, we set all C k to be the same for simplicity.

We then combine the output of all K GCN and feed them into a classification sub-network, allowing us to jointly train all GCNs and the classification sub-network via backpropagation.

This should allow the classification sub-network to choose features from the various GCNs, effectively allowing the overall model to learn a combination of features using the raw (normalized) adjacency, different steps of random walks, and the input features X (as they are multiplied by identityÂ 0 ).

From a deep learning prospective, it is intuitive to represent the classification network as a fullyconnected layer.

We can concatenate the output of the K GCNs along the column dimension, i.e. concatenating all GCN(X, DISPLAYFORM0 We add a fully-connected layer f fc : R N ×C K → R N ×C , with trainable parameter matrix W fc ∈ R C K ×C , written as: DISPLAYFORM1 The classifier parameters W fc are jointly trained with GCN parameters θ = {θ (0) , θ (1) , . . . }.

We use subscript fc on N-GCN to indicate the classification network is a fully-connected layer.

We also propose a classification network based on "softmax attention", which learns a convex combination of the GCN instantiations.

Our attention model (N-GCN a ) is parametrized by vector m ∈ R K , one scalar for each GCN.

It can be written as: DISPLAYFORM0 where m is output of a softmax: m = softmax( m).This softmax attention is similar to "Mixture of Experts" model, especially if we set the number of output channels for all GCNs equal to the number of classes, as in C 0 = C 1 = · · · = C. This allows us to add cross entropy loss terms on all GCN outputs in addition to the loss applied at the output NGCN, forcing all GCN's to be independently useful.

It is possible to set the m ∈ R K parameter vector "by hand" using the validation split, especially for reasonable K such as K ≤ 6.

One possible choice might be setting m 0 to some small value and remaining m 1 , . . .

, m K−1 to the harmonic series 1 k ; another choice may be linear decay DISPLAYFORM1 .

These are respectively similar to the context distributions of GloVe BID21 and word2vec BID20 BID17 .

We note that if on average a node's information is captured by its direct or nearby neighbors, then the output of GCNs consuming lower powers ofÂ should be weighted highly.

We minimize the cross entropy between our model output and the known training labels Y as: DISPLAYFORM0 where • is Hadamard product, and diag(V L ) denotes a diagonal matrix, with entry at (i, i) set to 1 if i ∈ V L and 0 otherwise.

In addition, we can apply intermediate supervision for the NGCN a to attempt make all GCN become independently useful, yielding minimization objective: DISPLAYFORM1

To simplify notation, our N-GCN derivations (e.g. Eq. 9) assume that there is one GCN perÂ power.

However, our implementation feeds everyÂ to r GCN modules, as shown in FIG2 .

In addition to vanilla GCNs (e.g. BID14 , our derivation also applies to other graph models including GraphSAGE (SAGE, BID10 .

Algorithm 1 shows a generalization that allows us to make a network of arbitrary graph models (e.g. GCN, SAGE, or others).

Algorithm 2 shows pseudo-code for the vanilla GCN.

Finally, Algorithm 3 defines our full Network of GCN model (N-GCN) by plugging Algorithm 2 into Algorithm 1.

Similarly, we list the algorithms for SAGE and Network of SAGE (N-SAGE) in the Appendix.

We can recover the original algorithms GCN BID14 and SAGE BID10 , respectively, by using Algorithms 3 (N-GCN) and 5 (N-SAGE, listed in Appendix) with r = 1, K = 1, identity CLASSIFIERFN, and modifying line 2 in Algorithm 1 to P ←Â. Moreover, we can recover original DCNN BID2 by calling Algorithm 3 with L = 1, r = 1, modifying line 3 toÂ ← D −1 A, and keeping K > 1 as their proposed model operates on the power series of the transition matrix i.e. unmodified random walks, like ours.

Require:Â is a normalization of A 1: function NETWORK(GRAPHMODELFN,Â, X, L, r = 4, K = 6, CLASSIFIERFN=FCLAYER) 2: DISPLAYFORM0 for k = 1 to K do 5:for i = 1 to r do 6:GraphModels.append(GRAPHMODELFN(P, X, L)) DISPLAYFORM1 return CLASSIFIERFN(GraphModels)Algorithm 2 GCN BID14 Require:Â is a normalization of DISPLAYFORM2 Z ← X 3: DISPLAYFORM3 return NETWORK(GCNMODEL,Â, X, L)

We follow the experimental setup by BID14 and BID25 , including the provided dataset splits (train, validation, test) produced by BID25 .

We experiment on three citation graph datasets: Pubmed, Citeseer, Cora, and a biological graph: Protein-Protein Interactions (PPI).

We choose the aforementioned datasets because they are available online and are used by our baselines.

The citation datasets are prepared by BID25 , and the PPI dataset is prepared by BID10 .

Table 1 summarizes dataset statistics.

Each node in the citation datasets represents an article published in the corresponding journal.

An edge between two nodes represents a citation from one article to another, and a label represents the subject of the article.

Each dataset contains a binary Bag-of-Words (BoW) feature vector for each node.

The BoW are extracted from the article abstract.

Therefore, the task is to predict the subject of articles, given the BoW of their abstract and the citations to other (possibly labeled) articles.

Following BID25 and BID14 , we use 20 nodes per class for training, 500 (overall) nodes for validation, and 1000 nodes for evaluation.

We note that the validation set is larger than training |V L | for these datasets!The PPI graph, as processed and described by BID10 , consists of 24 disjoint subgraphs, each corresponding to a different human tissue.

20 of those subgraphs are used for training, 2 for validation, and 2 for testing, as partitioned by BID10 .

For the citation datasets, we copy baseline numbers from BID14 .

These include label propagation (LP, BID26 ); semi-supervised embedding (SemiEmb, BID24 ); manifold regularization (ManiReg, BID6 ); skip-gram graph embeddings (DeepWalk Perozzi et al., 2014) ; Iterative Classification Algorithm (ICA, BID18 ; Planetoid BID25 ; vanilla GCN BID14 .

For PPI, we copy baseline numbers from BID10 , which include GraphSAGE with LSTM aggregation (SAGE-LSTM) and GraphSAGE with pooling aggregation (SAGE).

Further, for all datasets, we use our implementation to run baselines DCNN BID2 , GCN BID14 BID6 60.1 59.5 70.7 -(b) SemiEmb BID24 59.6 59.0 71.1 -(c) LP BID26 45.3 68.0 63.0 -(d) DeepWalk BID22 43.2 67.2 65.3 -(e) ICA BID18 69.1 75.1 73.9 -(f) Planetoid BID25 64.7 75.7 77.2 -(g) GCN BID14 70 Table 2 : Node classification performance (% accuracy for the first three, citation datasets, and f1 micro-averaged for multiclass PPI), using data splits of BID25 ; BID14 and BID10 .

We report the test accuracy corresponding to the run with the highest validation accuracy.

Results in rows (a) through (g) are copied from BID14 , rows (h) and (i) from BID10 , and (j) through (l) are generated using our code since we can recover other algorithms as explained in Section 3.6.

Rows (m) and (n) are our models.

Entries with "-" indicate that authors from whom we copied results did not run on those datasets.

Nonetheless, we run all datasets using our implementation of the most-competitive baselines.and SAGE (with pooling aggregation, BID10 , as these baselines can be recovered as special cases of our algorithm, as explained in Section 3.6.

We use TensorFlow BID0 to implement our methods, which we use to also measure the performance of baselines GCN, SAGE, and DCNN.

For our methods and baselines, all GCN and SAGE modules that we train are 2 layers, where the first outputs 16 dimensions per node and the second outputs the number of classes (dataset-dependent).

DCNN baseline has one layer and outputs 16 dimensions per node, and its channels (one per transition matrix power) are concatenated into a fully-connected layer that outputs the number of classes.

We use 50% dropout and L2 regularization of 10 −5 for all of the aforementioned models.

Table 2 shows node classification accuracy results.

We run 20 different random initializations for every model (baselines and ours), train using Adam optimizer BID3 with learning rate of 0.01 for 600 steps, capturing the model parameters at peak validation accuracy to avoid overfitting.

For our models, we sweep our hyperparameters r, K, and choice of classification subnetwork ∈ {fc, a}. For baselines and our models, we choose the model with the highest accuracy on validation set, and use it to record metrics on the test set in Table 2 .

Table 3 : Node classification accuracy (in %) for our largest dataset (Pubmed) as we vary size of training data |V| C ∈ {5, 10, 20, 100}. We report mean and standard deviations on 10 runs.

We use a different random seed for every run (i.e. selecting different labeled nodes), but the same 10 random seeds across models.

Convolution-based methods (e.g. SAGE) work well with few training examples, but unmodified random walk methods (e.g. DCNN) work well with more training data.

Our methods combine convolution and random walks, making them work well in both conditions.

Table 2 shows that N-GCN outperforms GCN BID14 and N-SAGE improves on SAGE for all datasets, showing that unmodified random walks indeed help in semi-supervised node classification.

Finally, our proposed models acheive state-of-the-art on all datasets.

We analyze the impact of K and r on classification accuracy in FIG3 .

We note that adding random walks by specifically setting K > 1 improves model accuracy due to the additional information, not due to increased model capacity.

Contrast K = 1, r > 1 (i.e. mixture of GCNs, no random walks) with K > 1, r = 1 (i.e. N-GCN on random walks): in both scenarios, the model has more capacity, but the latter shows better performance.

The same holds for SAGE, as shown in Appendix.

We test our method under feature noise perturbations by removing node features at random.

This is practical, as article authors might forget to include relevant terms in the article abstract, and more generally not all nodes will have the same amount of detailed information.

Figure 3 shows that when features are removed, methods utilizing unmodified random walks: N-GCN, N-SAGE, and DCNN, outperform convolutional methods including GCN and SAGE.

Moreover, the performance gap widens as we remove more features.

This suggests that our methods can somewhat recover removed features by directly pulling-in features from nearby and distant neighbors.

We visualize in Figure 4 the attention weights as a function of % features removed.

With little feature removal, there is some weight onÂ 0 , and the attention weights forÂ 1 ,Â 2 , . . .

follow some decay function.

Maliciously dropping features causes our model to shift its attention weights towards higher powers ofÂ.

The field of graph learning algorithms is quickly evolving.

We review work most similar to ours.

BID8 define graph convolutions as a K-degree polynomial of the Laplacian, where the polynomial coefficients are learned.

In their setup, the K-th degree Laplacian is a sparse square matrix where entry at (i, j) will be zero if nodes i and j are more than K hops apart.

Their sparsity analysis also applies here.

A minor difference is the adjacency normalization.

We useÂ whereas they use the Laplacian defined as I −Â. RaisingÂ to power K will produce a square matrix with entry (i, j) being the probability of random walker ending at node i after K steps from node j. The major difference is the order of random walk versus non-linearity.

In particular, their model calculates learns a linear combination of K-degree polynomial and pass through classifier function g, as in g( k q k A k ), while our (e.g. N-GCN) model calculates k q k g( A k ), where A isÂ in our model and I −Â in theirs, and our g can be a GCN module.

In fact, BID8 is also similar to work by Abu-El-Haija et al. FORMULA0 , as they both learn polynomial coefficients to some normalized adjacency matrix.

BID2 propose DCNN, which calculates powers of the transition matrix and keeps each power in a separate channel until the classification sub-network at the end.

Their model is therefore similar to our work in that it also falls under k q k g( A k ).

However, where their model multiplies features with each power A k once, our model makes use of GCN's BID14 ) that multiply by A k at every GCN layer (see Eq. 2).

Thus, DCNN model BID2 ) is a special case of ours, when GCN module contains only one layer, as explained in Section 3.6.

In this paper, we propose a meta-model that can run arbitrary Graph Convolution models, such as GCN BID14 and SAGE BID10 , on the output of random walks.

Traditional Graph Convolution models operate on the normalized adjacency matrix.

We make multiple instantiations of such models, feeding each instantiation a power of the adjacency matrix, and then concatenating the output of all instances into a classification sub-network.

Our model, Network of GCNs (and similarly, Network of SAGE), is end-to-end trainable, and is able to directly learn information across near or distant neighbors.

We inspect the distribution of parameter weights in our classification sub-network, which reveal to us that our model is effectively able to circumvent adversarial perturbations on the input by shifting weights towards model instances consuming higher powers of the adjacency matrix.

For future work, we plan to extend our methods to a stochastic implementation and tackle other (larger) graph datasets.

7.1 ALGORITHM FOR NETWORK OF SAGE Algorithms 4 and 5, respectively, define SAGE Hamilton et al. (2017) and Network of SAGE (N-SAGE).

Algorithm 4 assumes mean-pool aggregation by BID10 , which performs on-par to their top performer max-pool aggregation.

Further, Algorithm 4 operates in full-batch while BID10 offer a stochastic implementation with edge sampling.

Nonetheless, their proposed stochastic implementation should be wrapped in a network, though we would need a way to approximate (e.g. sample entries) from denseÂ k as k increases.

We leave this as future work.

Algorithm 4 SAGE Model BID10 Require:Â is a normalization of DISPLAYFORM0 Z ← X 3: DISPLAYFORM1 return NETWORK(SAGEMODEL,Â, X, 2)Using SAGE with mean-pooling aggregation is very similar to a vanilla GCN model but with three differences.

First, the choice of adjacency normalization ( DISPLAYFORM2 2 ).

Second, the skip connections in line 4, which concatenates the features with the adjacency-multiplied (i.e. diffused) features.

We believe this is analogous in intuition of incorporatingÂ 0 in our model, which keeps the original features.

Third, the use of node-wise L2 feature normalization at line 5, which is equivalent to applying a layernorm transformation J. BID13 .

Nonetheless, it is worth noting BID10 's formulation of SAGE is flexible to allow different aggregations, such as max-pooling or LSTM, which further deviates SAGE from GCN.

Earlier, in Table 2 , we showed the test performance corresponding to the model performing best on the validation split.

The number of labeled nodes are small, and such model selection is important to avoid overfitting.

For example, there can be up to 10% relative test accuracy difference when training the same model architecture but with different random seed.

In this section, we programatically sweep hyperparameters r, K, choice of classification network (∈ {fc, a}), and whether or not we enableÂ 0 , for both N-GCN and N-SAGE models.

The settings when (K = 1, r = 1, andÂ 0 disabled), correspond to the vanilla base model.

Further, the settings when (K = 1, r > 1, andÂ 0 disabled), correspond to an ensemble of the base model.

These cases are outperformed when K > 1, showing that unmodified random walks indeed help these convolutional methods perform better, by gathering information from nearby and distant nodes.

The automatically generated tables are shown below: Table 4 : N-GCN a results on Citeseer dataset, withÂ 0 disabled.

Top-left entry corresponds to vanilla GCN.

Left column corresponds to ensemble of GCN models.

DISPLAYFORM0 DISPLAYFORM1 K = 5 r = 1 78.1 ± 0.339 79.6 ± 0.293 79.8 ± 0.189 79.7 ± 0.170 79.6 ± 0.243 r = 2 77.3 ± 0.125 79.7 ± 0.171 79.6 ± 0.189 79.6 ± 0.138 79.9 ± 0.177 r = 4 77.3 ± 0.287 79.5 ± 0.396 79.5 ± 0.219 79.7 ± 0.149 79.9 ± 0.189 Table 5 : N-GCN a results on Citeseer dataset, withÂ 0 enabled.

DISPLAYFORM2 -78.6 ± 0.723 78.7 ± 0.407 78.7 ± 0.530 78.0 ± 0.690 r = 2 78.5 ± 0.353 77.9 ± 0.234 78.5 ± 0.724 78.8 ± 0.562 79.1 ± 0.267 r = 4 78.4 ± 0.499 78.4 ± 0.716 78.9 ± 0.306 78.9 ± 0.385 79.0 ± 0.228 Table 6 : N-GCN fc results on Citeseer dataset, withÂ 0 disabled.

Left column corresponds to ensemble of GCN models.

DISPLAYFORM3 K = 5 r = 1 76.5 ± 1.490 78.2 ± 1.290 79.2 ± 1.061 78.5 ± 0.963 78.7 ± 1.384 r = 2 76.1 ± 1.118 77.1 ± 1.152 78.8 ± 1.479 79.4 ± 0.754 78.7 ± 0.612 r = 4 76.0 ± 0.770 77.2 ± 0.785 78.7 ± 0.716 78.7 ± 0.953 79.0 ± 0.313 Table 7 : N-GCN fc results on Citeseer dataset, withÂ 0 enabled.

Table 9 : N-SAGE a results on Citeseer dataset, withÂ 0 enabled.

DISPLAYFORM4 DISPLAYFORM5 DISPLAYFORM6 -76.3 ± 1.545 76.7 ± 1.098 78.0 ± 1.427 77.3 ± 1.038 r = 2 76.6 ± 1.196 77.3 ± 1.309 77.8 ± 0.746 77.5 ± 0.836 77.5 ± 0.298 r = 4 76.5 ± 0.602 78.1 ± 1.239 77.6 ± 0.287 76.9 ± 0.472 77.7 ± 1.119 Table 10 : N-SAGE fc results on Citeseer dataset, withÂ 0 disabled.

Left column corresponds to ensemble of SAGE models.

DISPLAYFORM7 K = 5 r = 1 72.9 ± 0.972 75.9 ± 0.922 75.5 ± 0.499 76.6 ± 1.641 76.8 ± 0.589 r = 2 75.3 ± 0.879 76.1 ± 1.237 76.6 ± 0.579 76.4 ± 0.383 76.2 ± 0.626 r = 4 75.3 ± 1.730 76.4 ± 1.186 76.6 ± 0.576 76.8 ± 0.450 77.4 ± 0.712 Table 11 : N-SAGE fc results on Citeseer dataset, withÂ 0 enabled.

Table 12 : N-GCN a results on Cora dataset, withÂ 0 disabled.

Top-left entry corresponds to vanilla GCN.

Left column corresponds to ensemble of GCN models.

DISPLAYFORM8 DISPLAYFORM9 K = 5 r = 1 78.1 ± 0.339 79.6 ± 0.293 79.8 ± 0.189 79.7 ± 0.170 79.6 ± 0.243 r = 2 77.3 ± 0.125 79.7 ± 0.171 79.6 ± 0.189 79.6 ± 0.138 79.9 ± 0.177 r = 4 77.3 ± 0.287 79.5 ± 0.396 79.5 ± 0.219 79.7 ± 0.149 79.9 ± 0.189 Table 13 : N-GCN a results on Cora dataset, withÂ 0 enabled.

DISPLAYFORM10 -78.6 ± 0.723 78.7 ± 0.407 78.7 ± 0.530 78.0 ± 0.690 r = 2 78.5 ± 0.353 77.9 ± 0.234 78.5 ± 0.724 78.8 ± 0.562 79.1 ± 0.267 r = 4 78.4 ± 0.499 78.4 ± 0.716 78.9 ± 0.306 78.9 ± 0.385 79.0 ± 0.228 Table 14 : N-GCN fc results on Cora dataset, withÂ 0 disabled.

Left column corresponds to ensemble of GCN models.

DISPLAYFORM11 K = 5 r = 1 76.5 ± 1.490 78.2 ± 1.290 79.2 ± 1.061 78.5 ± 0.963 78.7 ± 1.384 r = 2 76.1 ± 1.118 77.1 ± 1.152 78.8 ± 1.479 79.4 ± 0.754 78.7 ± 0.612 r = 4 76.0 ± 0.770 77.2 ± 0.785 78.7 ± 0.716 78.7 ± 0.953 79.0 ± 0.313 Table 15 : N-GCN fc results on Cora dataset, withÂ 0 enabled.

Table 16 : N-SAGE a results on Cora dataset, withÂ 0 disabled.

Top-left entry corresponds to vanilla SAGE.

Left column corresponds to ensemble of SAGE models.

Table 17 : N-SAGE a results on Cora dataset, withÂ 0 enabled.

DISPLAYFORM12 DISPLAYFORM13 DISPLAYFORM14 -76.3 ± 1.545 76.7 ± 1.098 78.0 ± 1.427 77.3 ± 1.038 r = 2 76.6 ± 1.196 77.3 ± 1.309 77.8 ± 0.746 77.5 ± 0.836 77.5 ± 0.298 r = 4 76.5 ± 0.602 78.1 ± 1.239 77.6 ± 0.287 76.9 ± 0.472 77.7 ± 1.119 TAB3 : N-SAGE fc results on Cora dataset, withÂ 0 disabled.

Left column corresponds to ensemble of SAGE models.

DISPLAYFORM15 K = 5 r = 1 72.9 ± 0.972 75.9 ± 0.922 75.5 ± 0.499 76.6 ± 1.641 76.8 ± 0.589 r = 2 75.3 ± 0.879 76.1 ± 1.237 76.6 ± 0.579 76.4 ± 0.383 76.2 ± 0.626 r = 4 75.3 ± 1.730 76.4 ± 1.186 76.6 ± 0.576 76.8 ± 0.450 77.4 ± 0.712 Table 19 : N-SAGE fc results on Cora dataset, withÂ 0 enabled.

Table 20 : N-GCN a results on Pubmed dataset, withÂ 0 disabled.

Top-left entry corresponds to vanilla GCN.

Left column corresponds to ensemble of GCN models.

K = 1 K = 2 K = 3 K = 4 K = 5 r = 1 78.1 ± 0.339 79.6 ± 0.293 79.8 ± 0.189 79.7 ± 0.170 79.6 ± 0.243 r = 2 77.3 ± 0.125 79.7 ± 0.171 79.6 ± 0.189 79.6 ± 0.138 79.9 ± 0.177 r = 4 77.3 ± 0.287 79.5 ± 0.396 79.5 ± 0.219 79.7 ± 0.149 79.9 ± 0.189 Table 21 : N-GCN a results on Pubmed dataset, withÂ 0 enabled.

DISPLAYFORM16 DISPLAYFORM17 -78.6 ± 0.723 78.7 ± 0.407 78.7 ± 0.530 78.0 ± 0.690 r = 2 78.5 ± 0.353 77.9 ± 0.234 78.5 ± 0.724 78.8 ± 0.562 79.1 ± 0.267 r = 4 78.4 ± 0.499 78.4 ± 0.716 78.9 ± 0.306 78.9 ± 0.385 79.0 ± 0.228 Table 22 : N-GCN fc results on Pubmed dataset, withÂ 0 disabled.

Left column corresponds to ensemble of GCN models.

DISPLAYFORM18 K = 5 r = 1 76.5 ± 1.490 78.2 ± 1.290 79.2 ± 1.061 78.5 ± 0.963 78.7 ± 1.384 r = 2 76.1 ± 1.118 77.1 ± 1.152 78.8 ± 1.479 79.4 ± 0.754 78.7 ± 0.612 r = 4 76.0 ± 0.770 77.2 ± 0.785 78.7 ± 0.716 78.7 ± 0.953 79.0 ± 0.313 Table 23 : N-GCN fc results on Pubmed dataset, withÂ 0 enabled.

Table 25 : N-SAGE a results on Pubmed dataset, withÂ 0 enabled.

DISPLAYFORM19 DISPLAYFORM20 -76.3 ± 1.545 76.7 ± 1.098 78.0 ± 1.427 77.3 ± 1.038 r = 2 76.6 ± 1.196 77.3 ± 1.309 77.8 ± 0.746 77.5 ± 0.836 77.5 ± 0.298 r = 4 76.5 ± 0.602 78.1 ± 1.239 77.6 ± 0.287 76.9 ± 0.472 77.7 ± 1.119 Table 26 : N-SAGE fc results on Pubmed dataset, withÂ 0 disabled.

Left column corresponds to ensemble of SAGE models.

K = 1 K = 2 K = 3 K = 4 K = 5 r = 1 72.9 ± 0.972 75.9 ± 0.922 75.5 ± 0.499 76.6 ± 1.641 76.8 ± 0.589 r = 2 75.3 ± 0.879 76.1 ± 1.237 76.6 ± 0.579 76.4 ± 0.383 76.2 ± 0.626 r = 4 75.3 ± 1.730 76.4 ± 1.186 76.6 ± 0.576 76.8 ± 0.450 77.4 ± 0.712 Table 27 : N-SAGE fc results on Pubmed dataset, withÂ 0 enabled.

@highlight

We make a network of Graph Convolution Networks, feeding each a different power of the adjacency matrix, combining all their representation into a classification sub-network, achieving state-of-the-art on semi-supervised node classification.

@highlight

Proposes a new network of GCNs with two approaches: a fully connected layer on top of stacked features and attention mechanism that uses scalar weight per GCN.

@highlight

Presents a Network of Graph Convolutional Networks that uses random walk statistics to extract information from near and distant neighbors in the graph
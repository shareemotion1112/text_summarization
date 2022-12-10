Graph Neural Networks (GNNs) are a class of deep models that operates on data with arbitrary topology and order-invariant structure represented as graphs.

We introduce an efficient memory layer for GNNs that can learn to jointly perform graph representation learning and graph pooling.

We also introduce two new networks based on our memory layer: Memory-Based Graph Neural Network (MemGNN) and Graph Memory Network (GMN) that can learn hierarchical graph representations by coarsening the graph throughout the layers of memory.

The experimental results demonstrate that the proposed models achieve state-of-the-art results in six out of seven graph classification and regression benchmarks.

We also show that the learned representations could correspond to chemical features in the molecule data.

Graph Neural Networks (GNNs) (Wu et al., 2019; Zhou et al., 2018; are a class of deep architectures that operate on data with arbitrary topology represented as graphs such as social networks (Kipf & Welling, 2016) , knowledge graphs (Schlichtkrull et al., 2018) , molecules (Duvenaud et al., 2015) , point clouds (Hassani & Haley, 2019) , and robots .

Unlike regular-structured inputs with spatial locality such as grids (e.g., images and volumetric data) and sequences (e.g., speech and text), GNN inputs are variable-size graphs consisting of permutationinvariant nodes and interactions among them.

GNNs such as Gated GNN (GGNN) (Li et al., 2015) , Message Passing Neural Network (MPNN) (Gilmer et al., 2017) , Graph Convolutional Network (GCN) (Kipf & Welling, 2016) , and Graph Attention Network (GAT) (Velikovi et al., 2018) learn node embeddings through an iterative process of transferring, transforming, and aggregating the node embeddings from topological neighbors.

Each iteration expands the receptive field by one hop and after k iterations the nodes within k hops influence the node embeddings of one another.

GNNs are shown to learn better representations compared to random walks (Grover & Leskovec, 2016; Perozzi et al., 2014) , matrix factorization (Belkin & Niyogi, 2002; Ou et al., 2016) , kernel methods (Shervashidze et al., 2011; Kriege et al., 2016) , and probabilistic graphical models (Dai et al., 2016) .

These models, however, cannot learn hierarchical representation as they do not exploit the graph compositionality.

Recent work such as Differentiable Pooling (DiffPool) (Ying et al., 2018) , TopKPool (Gao & Ji, 2019) , and Self-Attention Graph Pooling (SAGPool) (Lee et al., 2019) define parametric graph pooling layers that let models learn hierarchical graph representation by stacking interleaved layers of GNN and pooling layers.

These layers cluster nodes in the latent space such that the clusters are meaningful with respect to the task.

These clusters might be communities in a social network or potent functional groups within a chemical dataset.

Nevertheless, these models are not efficient as they require an iterative process of message passing after each pooling layer.

In this paper, we introduce a memory layer for joint graph representation learning and graph coarsening that consists of a multi-head array of memory keys and a convolution operator to aggregate the soft cluster assignments from different heads.

The queries to a memory layer are node embeddings from the previous layer and the outputs are the node embeddings of the coarsened graph.

The memory layer does not explicitly require connectivity information and unlike GNNs relies on the global information rather than local topology.

These properties make them more efficient and improve their performance.

We also introduce two networks based on the proposed layer: Memory-based Graph Neural Network (MemGNN) and Graph Memory Network (GMN).

MemGNN consists of a GNN that learns the initial node embeddings, and a stack of memory layers that learns hierarchical graph representation up to the global graph embedding.

GMN, on the other hand, learns the hierarchical representation purely based on memory layers and hence does not require message passing.

Memory Augmented Neural Networks (MANNs) utilize external memory with differentiable read-write operators allowing them to explicitly access the past experiences and are shown to enhance reinforcement learning (Pritzel et al., 2017) , meta learning (Santoro et al., 2016) , few-shot learning (Vinyals et al., 2016) , and multi-hop reasoning .

Unlike RNNs, in which the memory is represented within their hidden states, the decoupled memory in MANNs lets them to store and retrieve longer term memories with less parameters.

The memory can be implemented as a key-value memory such as neural episodic control (Pritzel et al., 2017) and product-key memory layers (Lample et al., 2019) or as a array-structured memory such as Neural Turing Machine (NTM) (Graves et al., 2014) , prototypical networks (Snell et al., 2017) , memory networks , and Sparse Access Memory (SAM) (Rae et al., 2016) .

Our memory layer consists of a multi-head array of memory keys.

Graph Neural Networks (GNNs) use message passing to learn node embeddings over graphs.

GraphSAGE (Hamilton et al., 2017b) learns embedding by sampling and aggregating neighbor nodes whereas GAT (Velikovi et al., 2018) uses attention to aggregate embeddings from all neighbors.

GCN models extend the convolution to arbitrary topology.

Spectral GCNs (Bruna et al., 2014; Defferrard et al., 2016; Kipf & Welling, 2016) use spectral filters over graph Laplacian to define the convolution in the Fourier domain.

These models are less efficient compared to spatial GCNs (Schlichtkrull et al., 2018; Ma et al., 2019) which directly define the convolution on graph patches centered on nodes.

Our memory layer uses a feed-forward network to learn the node embeddings.

Graph pooling can be done globally or hierarchically.

In former, node embeddings are aggregated into a graph embedding using arithmetic operators such as sum or max (Hamilton et al., 2017a; Kipf & Welling, 2016) or set neural networks such as Set2Set (Vinyals et al., 2015) and SortPool (Morris et al., 2019) .

In latter, graphs are coarsened in each layer to capture the hierarchical structure.

Non-parametric methods such as clique pooling (Luzhnica et al., 2019) , kNN pooling (Wang et al., 2018) , and Graclus (Dhillon et al., 2007) rely on topological information and are efficient, but are outperformed by parametric models such as edge contraction pooling (Diehl, 2019) .

DiffPool (Ying et al., 2018) trains two parallel GNNs to compute node embeddings and cluster assignments using a combination of classification loss, link prediction loss, and entropy loss, whereas Mincut pool (Bianchi et al., 2019) trains a sequence of a GNN and an MLP using classification loss and the minimum cut objective.

TopKPool (Cangea et al., 2018; Gao & Ji, 2019 ) computes a node score by learning a projection vector and then drops all the nodes except the top scoring nodes.

SAGPool (Lee et al., 2019) extends the TopKPool by using graph convolutions to take neighbor node features into account.

We use a clustering-friendly distribution to compute the attention scores between nodes and clusters.

We define a memory layer M (l) : R n l ×d l −→ R n l+1 ×d l+1 in layer l as a parametric function that takes in n l query vectors of size d l and generates n l+1 query vectors of size d l+1 such that n l+1 < n l .

The input and output queries represent the node features of the input graph and the coarsened graph, respectively.

The memory layer learns to jointly coarsen the input nodes (i.e., pooling) and transform their features (i.e., representation learning).

As shown in Figure 1 , it consists of arrays of memory keys (i.e., multi-head memory) and a convolutional layer.

Assuming |h| memory heads, a shared input query is compared against all the keys in each head resulting in |h| attention matrices which are then aggregated into a single attention matrix using the convolution layer.

In a content addressable memory (Graves et al., 2014; Sukhbaatar et al., 2015; , the task of attending to memory (i.e., addressing scheme) is formulated as computing the similarly between memory keys to a given query q. Specifically, the attention weight of key k j for query q is defined as w j = sof tmax(d(q, k j )) where d is a similarity measure, typically Euclidean distance or cosine similarity (Rae et al., 2016) .

The soft read operation on memory is defined as a weighted average over the memory keys: r = j w j k j .

Figure 1 : The proposed Architecture for hierarchical graph representation learning using the introduced memory layer.

The query network projects the initial node attributes to a query embedding space and each memory layer jointly coarsens the input queries and transforms them into a new query space.

In this work, we treat the input queries Q (l) ∈ R n l ×d l as the node embeddings of an input graph and treat the keys K (l) ∈ R n l+1 ×d l as the cluster centroids of the queries.

To satisfy this assumption, we impose a clustering-friendly distribution as the distance metric between keys and a query.

Following (Xie et al., 2016; Maaten & Hinton, 2008) , we use the Student's t-distribution as a kernel to measure the normalized similarity between query q i and key k j as follows:

where C ij is the normalized score between query q i and key k j (i.e., probability of assigning node i to cluster j or attention score between query q i and memory key k j ) and τ is the degree of freedom of the Student's t-distribution (i.e., temperature).

To increase the model capacity, we model the memory keys as a multi-head array.

Applying a shared input query against the memory keys produces a tensor of cluster assignments [C

|h| ] ∈ R |h|×n l+1 ×n l where |h| denotes the number of heads.

To aggregate the heads into a single assignment matrix, we treat the heads and the matrix rows and columns as depth, height, and width in standard convolution analogy and apply a convolution operator over them.

Because there is no spatial structure, we use [1 × 1] convolution to aggregate the information across heads and therefore the convolution behaves as a weighted pooling that reduces the heads to a single matrix.

The aggregated assignment matrix is computed as follows:

where Γ φ is a [1 × 1] convolutional operator parametrized by φ, || is the concatenation operator, and C (l) is the aggregated soft assignment matrix.

A memory read generates a value matrix V (l) ∈ R n l+1 ×d l that represents the coarsened node embeddings in the same space as the input queries and is defined as the product of the soft assignment scores and the original queries as follows:

The value matrix is fed to a single layer neural network consisting of a weight matrix

and a LeakyReLU activation function to project the coarsened embeddings from R n l+1 ×d l into R n l+1 ×d l+1 representing the output queries Q (l+1) :

Thanks to these parametrized transformations, a memory layer can jointly learn the node embeddings and coarsens the graph end-to-end.

The computed queries Q (l+1) are the input queries to the subsequent memory layer M (l+1) .

For graph classification, one can simply stack layers of memory up to the level where the input graph is coarsened into a single node representing the global graph embedding and then feed it to a fully-connected layer to predict the graph class as follows:

where Q 0 = f q (g) is the initial query embedding 1 generated by the query network f q over graph g. We introduce two architectures based on the memory layer: GMN and MemGNN.

These two architectures are different in the way that the query network is implemented.

More specifically, GMN uses a feed-forward network for initializing the query: f q (g) = FFN θ (g), whereas MemGNN implements the query network as a message passing GNN: f q (g) = GNN θ (g).

A GMN is a stack of memory layers on top of a query network f q (g) that generates the initial query embeddings without any message passing.

Similar to set neural networks (Vinyals et al., 2015) and transformers (Vaswani et al., 2017) , graph nodes in a GMN are treated as a permutation-invariant set of embeddings.

The query network projects the initial node attributes into an embedding space that represents the initial query space.

Assume a training set D = [g 1 , g 2 , ..., g N ] of N graphs where each graph is represented as g = (A, X, Y ) and A ∈ {0, 1} n×n denotes the adjacency matrix, X ∈ R n×din is the initial node attribute, and Y ∈ R n is the graph label.

Considering that the GMN model treats a graph as a set of permutation-invariant nodes and does not use message passing, and also considering that the memory layers do not rely on connectivity information, the topological information of each node should be somehow encoded into its initial embedding.

Inspired by transformers (Vaswani et al., 2017) , we encode this information along with the initial attribute into the initial query embeddings using a query network f q implemented as a two-layer feed-forward neural network:

where W 0 ∈ R n×din and W 1 ∈ R 2din×d0 are the parameters of the query networks, and || is the concatenation operator.

Unlike the GMN architecture, the query network in MemGNN relies on the iterative process of passing messages and aggregating them to compute the initial query Q 0 :

where query network G θ is an arbitrary parameterized message passing GNN (Gilmer et al., 2017; Li et al., 2015; Kipf & Welling, 2016; Velikovi et al., 2018) .

In our implementation of MemGNN, we use a modified variant of GAT (Velikovi et al., 2018) .

Specifically, we introduce an extension to the original GAT model called edge-based GAT (e-GAT) and use it as the query network.

Unlike GAT, e-GAT learns attention weights not only from the neighbor nodes but also from the input edge attributes.

This is especially important for data containing edge information (e.g., various bonds among atoms represented as edges in molecule datasets).

In an e-GAT layer, attention score between two neighbor nodes is computed as follows.

where

i→j denote the embedding of node i and the embedding of the edge connecting node i to its one-hop neighbor node j in layer l, respectively.

W n and W e are trainable node and edge weights and W is the parameter of a single-layer feed-forward network that computes the attention score.

We jointly train the model using two loss functions: a supervised classification loss and an unsupervised clustering loss.

The supervised loss denoted as L ent is defined as the cross-entropy loss between the predicted and true graph class labels.

The unsupervised clustering loss is inspired by deep clustering methods (Razavi et al., 2019; Xie et al., 2016; Aljalbout et al., 2018) .

It encourages the model to learn clustering-friendly embeddings in the latent space by urging it to learn from high confidence assignments with the help of an auxiliary target distribution.

The unsupervised loss is defined as the Kullback-Leibler (KL) divergence loss between the soft assignments C (l) and the auxiliary distribution P (l) as follows:

For the target distributions P (l) , we use the distribution proposed in (Xie et al., 2016) which normalizes the loss contributions and improves the cluster purity while emphasizing on the samples with higher confidence.

This distribution is defined as follows:

We define the total loss as follows where L is the number of memory layers and λ is a scalar weight.

We initialize the model parameters, the keys, and the queries randomly and optimize them jointly with respect to L using mini-batch stochastic gradient descent.

To stabilize the training, the gradients of L ent are back-propagated batch-wise while the gradients of L KL .

This technique has also been applied in (Hassani & Haley, 2019; Caron et al., 2018) to avoid trivial solutions in deep clustering problem.

We use nine graph benchmarks including seven classification and two regression datasets to evaluate the proposed method.

These datasets are commonly used in both graph kernel (Borgwardt & Kriegel, 2005; Yanardag & Vishwanathan, 2015; Shervashidze et al., 2009; Ying et al., 2018; Shervashidze et al., 2011; Kriege et al., 2016) and GNN (Cangea et al., 2018; Ying et al., 2018; Lee et al., 2019; Gao & Ji, 2019) literature.

The summary of these datasets is as follows (i.e., first two benchmarks are regression tasks and the rest are classification tasks):

ESOL (Delaney, 2004) contains water solubility data for compounds.

Lipophilicity (Gaulton et al., 2016) contains experimental results of octanol/water distribution of compounds.

Bace (Subramanian et al., 2016) provides quantitative binding results for a set of inhibitors of human β-secretase 1 (BACE-1).

DD (Dobson & Doig, 2003 ) is used to distinguish enzyme structures from non-enzymes.

Enzymes (Schomburg et al., 2004 ) is for predicting functional classes of enzymes.

Proteins (Dobson & Doig, 2003 ) is used to predict the protein function from structure.

COLLAB (Yanardag & Vishwanathan, 2015) is for predicting the field of a researcher given her For more information about the datasets and implementation details refer to Appendix A.2 and A.1, repectively.

To evaluate the performance of our models on DD, Enzymes, Proteins, and COLLAB datasets, we follow the experimental protocol in (Ying et al., 2018) and perform 10-fold cross-validation and report the mean accuracy over all folds.

We also report the performance of four kernel-based methods including Graphlet (Shervashidze et al., 2009) , shortest path (Borgwardt & Kriegel, 2005) , Weisfeiler-Lehman (WL) (Shervashidze et al., 2011) , and WL Optimal Assignment (Kriege et al., 2016) , and ten deep models.

The results shown in Table 1 suggest that: (i) our models significantly improve the performance on DD, Enzymes, and Proteins datasets by absolute margins of 14.49%, 4.75%, and 0.49% accuracy, respectively, (ii) both proposed models achieve better performance on these three datasets compared to the baselines, (iii) MemGNN outperforms GMN on COLLAB whereas GMN achieves better result on the Enzymes, Proteins, and DD datasets.

On COLLAB, our models are outperformed by a variant of DiffPpool (i.e., diffpool-det) (Ying et al., 2018) and WL Optimal Assignment (Kriege et al., 2016) .

The former is a GNN augmented with deterministic clustering algorithm 2 , whereas the latter is a graph kernel method.

We speculate that because of the high edge-to-node ratio of COLLAB, these augmentations help in extracting near-optimal cliques.

For the ESOL and Lipophilicity datasets, we follow the evaluation protocol in (Wu et al., 2018) and report the Root-Mean-Square Error (RMSE) for these regression benchmarks.

Considering that these datasets contain initial edge attributes (refer to Appendix A.2 for further details), we train the MemGNN model and compare the results to the baseline models reported in (Wu et al., 2018) including graph-based methods such as GCN, MPNN, Directed Acyclic Graph (DAG) based models, Weave as well as other conventional methods such as Kernel Ridge Regression (KRR) and Influence Relevance Voting (IRV).

Tables 2 and 5 show that our MemGNN model achieves state-ofthe-art results by absolute margin of 0.04 and 0.1 RMSE on ESOL and Lipophilicity benchmarks, respectively.

For further details on these datasets and the baselines see (Wu et al., 2018) .

We also achieve state-of-the-art results on the Bace, Reddit-Binary, and Tox21 datasets.

For more details see Appendix A.3.

To investigate the effect of the proposed e-GAT model, we train our MemGNN model using both GAT and e-GAT models as the query network.

Considering that the ESOL, Lipophilicity, and BACE datasets contain edge attributes, we use them as the benchmarks.

Since nodes have richer features compared to edges, we set the node and edge feature dimensions to 16 and 4, respectively.

The comparative performance evaluation of the two models on the ESOL dataset is shown in Appendix A.4 demonstrating that e-GAT achieves better results on the validation set in each epoch compared to the standard GAT model.

We observed the same effect on Lipophilicity and BACE datasets too.

To investigate the effect of the topological embeddings on the GMN model, we evaluated three initial topological features including adjacency matrix, normalized adjacency matrix, and Random Walk with Restart (RWR).

For further details on RWR, see section A.5.

The results suggested that using the adjacency matrix as the initial feature achieves the best performance.

For instance, 10-fold cross validation accuracy of a GMN model trained on ENZYMES with adjacency matrix, normalized adjacency matrix, and RWR is 78.66%, 77.16%, and 77.33%, respectively.

We investigated two methods to down-sample the neighbors in dense datasets such as COLLAB (i.e., average of 66 neighbors per node) to enhance the memory and computation.

The first method randomly selects 10% of the edges whereas the second method ranks the neighbors based on their RWR scores with respect to the center node and then keeps the top 10% of the edges.

We trained the MemGNN model on COLLAB using both sampling methods which resulted in 73.9% and 73.1% 10-fold cross validation accuracy for random and RWR-based sampling methods respectively, suggesting that random sampling performs slightly better than a random walk sampling.

We stipulate that although keys represent the clusters, the number of keys is not necessarily proportional to the number of the nodes in the input graphs.

In fact, datasets with smaller graphs might have more meaningful clusters to capture.

For example, molecules are comprised of numerous functional groups and yet the average number of nodes in the ESOL dataset is 13.3.

Moreover, our experiments show that for ENZYMES with average number of 32.69 nodes, the best performance is achieved with 10 keys whereas for the ESOL dataset 64 keys results in the best performance.

In ESOL 8, 64, and 160 keys result in RMSE of 0.56, 0.52, and 0.54, respectively.

We also observed that keeping the number of parameters fixed, increasing the number of memory heads improves the performance.

For instance, when the model is trained on ESOL with 160 keys and 1 head, it achieves RMSE of 0.54, whereas when trained with 32 keys of 5 heads, the same model achieves RMSE of 0.53.

Intuitively, the memory keys represent the cluster centroids and enhance the model performance by capturing meaningful structures and coarsening the graph.

To investigate this intuition, we used the learned keys to interpret the knowledge learned by the models through visualizations.

Figure 2 visualizes the learned clusters over atoms (i.e., atoms with same color are within the same cluster) indicating that the clusters mainly consist of meaningful chemical substructures such as a carbon chain and a Hydroxyl group (OH) (i.e., Figure 2a) , as well as a Carboxyl group (COOH) and a benzene ring (i.e., Figure 2b ).

From a chemical perspective, Hydroxyl and Carboxyl groups, and carbon chains have a significant impact on the solubility of the molecule in water or lipid.

This confirms that the network has learned chemical features that are essential for determining the molecule solubility.

It is noteworthy that we tried initializing the memory keys using K-Means algorithm over the initial node embeddings to warm-start them but we did not observe any significant improvement over the randomly selected keys.

We proposed an efficient memory layer and two deep models for hierarchical graph representation learning.

We evaluated the proposed models on nine graph classification and regression tasks and achieved state-of-the-art results on eight of them.

We also experimentally showed that the learned representations can capture the well-known chemical features of the molecules.

Our study indicated that node attributes concatenated with corresponding topological embeddings in combination with one or more memory layers achieves notable results without using message passing.

We also showed that for the topological embeddings, the binary adjacency matrix is sufficient and thus no further preprocessing step is required for extracting them.

Finally, we showed that although connectivity information is not directly imposed on the model, the memory layer can process node embeddings and properly cluster and aggregate the learned embeddings.

Limitations: In section 4.2, we discussed that on the COLLAB dataset, kernel methods or deep models augmented with deterministic clustering algorithm achieve better performance compared to our models.

Analyzing samples in this dataset suggests that in graphs with dense communities, such as cliques, our model lacks the ability to properly detect these dense sub-graphs.

Moreover, the results of the DD dataset reveals that our MemGNN model outperforms the GMN model which implies that we need message passing to perform better on this dataset.

We speculate that this is because the DD dataset relies more on local information.

The most important features to train an SVM on this dataset are surface features which have local behavior.

This suggest that for data with strong local interactions, message passing is required to improve the performance.

Future Directions: We are planning to introduce a model based on the MemGNN and GMN architectures that can perform node classification by attending to the node embeddings and centroids of the clusters from different layers of hierarchy that the node belongs to.

We are also planning to investigate the representation learning capabilities of the proposed models in self-supervised setting.

A APPENDIX

We implemented the model with PyTorch (Paszke et al., 2017) and optimized it using Adam (Kingma & Ba, 2014) optimizer.

We trained the model for a maximum number of 2000 epochs and decayed the learning rate every 500 epochs by 0.5.

The model uses batch-normalization (Ioffe & Szegedy, 2015) , skip-connections, LeakyRelu activation functions, and dropout (Srivastava et al., 2014) for regularization.

We decided the hidden dimension and number of model parameters using random hyper-parameter search strategy.

The best performing hyper-parameters for the datasets are shown in Table 3 .

A.2 DATASET STATISTICS Table 4 indicates the statistics of the datasets we used for graph classification and regression tasks.

In addition to results discussed in section 4.2, we report the evaluation results on three other graph classification benchmarks including BACE (Subramanian et al., 2016 ), Tox21 (Challenge, 2014 , and Reddit-Binary (Yanardag & Vishwanathan, 2015) datasets.

The BACE dataset provides quantitative binding results for a set of inhibitors of human β-secretase 1 (BACE-1), whereas the is Tox21 dataset is for predicting toxicity on 12 different targets.

We follow the evaluation protocol in (Wu et al., 2018) and report the Area Under the Curve Receiver Operating Characteristics (AUC-ROC) measure for this task.

Moreover, considering that the BACE and Tox21 datasets contain initial edge attributes, we train the MemGNN model and compare its performance to the baseline models reported in (Wu et al., 2018) .

The results shown in Table 5 suggest that our model achieves stateof-the-art results by absolute margin of 4.0% AUC-ROC on BACE benchmark.

The results also suggest that our model is competitive with the state-of-the-art GCN model on the Tox21 dataset.

Reddit-Binary dataset is for predicting the type of community (i.e., question-answer-based or discussion-based communities) given a graph of online discussion threads.

In this dataset, nodes represent the users and edges denote interaction between users.

To evaluate the performance of our models on this dataset, we follow the experimental protocol in (Ying et al., 2018) and perform 10-fold cross-validation to evaluate the model performance and report the mean accuracy over all folds.

The results reported in Table 6 show that the introduced GMN model achieves state-of-the-art accuracy by absolute margin of 0.44%.

In section 4.3.1, we introduced e-GAT.

Figure 3a illustrates the RMSE on the validation set of the ESOL dataset for a MemGNN model using GAT and e-GAT implementation as its query network, respectively.

Since ESOL is a regression benchmark, we also plotted the R 2 score in Figure 3b .

As shown in these figures, e-GAT performs better compared to GAT on both metrics.

Consider a weighted or unweighted graph G. A random agent starts traversing the graph from node i and iteratively walks towards its neighbors with a probability proportional to the edge weight that 0.859 ± 0.000 0.907 ± 0.000 Table 6 : Mean validation accuracy over 10-folds.

Method REDDIT-BINARY Graphlet (Shervashidze et al., 2009) 78.04 WL (Shervashidze et al., 2011) 68.2 DiffPool (Ying et al., 2018) 85.95 TopKPool (Gao & Ji, 2019) 74.70 SAGPool (Lee et al., 2019) 73.90 GMN (ours)

86.39 MemGNN (ours) 85.55 connects them.

The agent also may restart the traverse from the starting node i with probability p. Eventually, the agent will stop at node j with a probability called relevance score of node j with respect to node i (Tong et al., 2006) .

The relevance score of node i with every other node of the graph is summarized in t i in equation 12 where t i is the RWR corresponding to node i, p is the restart probability, A is the normalized adjacency matrix and e i is the one-hot vector representing node i. t i = pÃ t i + (1 − p) e i

Solving this linear system results in r i defined in equation 13

Now we can nominate the nodes with higher relevance score w.r.t.

node i for receiving messages in an MeMGNN or use them as topological embeddings in a GMN.

Note that the restart probability in equation 13 defines how far the agent can walk from the source node and therefore the t i can represent whether local or global structures around the node i.

We used p = 0.5 in our studies.

Calculating the inverse of the adjacency matrix of a big graph is costly.

Although we could exactly compute it for all of our datasets but there are existing methods to make the estimation more efficient (Tong et al., 2006) .

Figure 5 : Clusters learned by a MeMGNN for ESOL and LIPO dataset.

Chemical groups like OH (hydroxyl group), CCl3, COOH (carboxyl group), CO (ketone group) as well as benzene rings have been recognized during the learning procedure.

These chemical groups are highly active and have a great impact on the solubility of molecules.

@highlight

We introduce an efficient memory layer that can learn representation and coarsen input graphs simultaneously without relying on message passing.
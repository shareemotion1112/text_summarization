Graph Convolution Network (GCN) has been recognized as one of the most effective graph models for semi-supervised learning, but it extracts merely the first-order or few-order neighborhood information through information propagation, which suffers performance drop-off for deeper structure.

Existing approaches that deal with the higher-order neighbors tend to take advantage of adjacency matrix power.

In this paper, we assume a seemly trivial condition that the higher-order neighborhood information may be similar to that of the first-order neighbors.

Accordingly, we present an unsupervised approach to describe such similarities and learn the weight matrices of higher-order neighbors automatically through Lasso that minimizes the feature loss between the first-order and higher-order neighbors, based on which we formulate the new convolutional filter for GCN to learn the better node representations.

Our model, called higher-order weighted GCN (HWGCN), has achieved the state-of-the-art results on a number of node classification tasks over Cora, Citeseer and Pubmed datasets.

Convolutional neural networks (CNNs) have made great achievements on a wide range of image tasks including image classification (Simonyan & Zisserman, 2015; Szegedy et al., 2015; Huang et al., 2017) , object detection (Girshick et al., 2014; Redmon et al., 2016; Liu et al., 2016; Dai et al., 2016; Lin et al., 2017) , semantic segmentation (Long et al., 2015; Badrinarayanan et al., 2017; Chen et al., 2017) , etc.

Due to the fact that their underlying data representation has a grid-like structure, CNNs perform highly effective on image processing, and can thus capture local patterns by compressing the hypothesis space and using local filters to learn the parameters.

However, lots of real-word data cannot be represented as grid-like structure.

For example, social networks and biological networks are usually represented as graphs instead of grid-like structure, while the data defined on 3D meshes is important for many graphical applications.

As a result, there is an increasing number of fields that focus on studying non-Euclidean structured data.

To address such challenge, inspired by the great success of applying CNNs to computer vision tasks, many research efforts have been devoted to a paradigm shift in graph learning that generalizes convolutions to the graph domain.

More specifically, the graph structure is encoded using a convolutional neural network model to operate the neighborhood of each node in graphs.

In general, attempts in this direction can be categorized into non-spectral (spatial) approaches and spectral approaches.

While recent works are making progress on these two lines of research respectively, here, we focus on the extension of the graph convolution spectral filter.

In this respect, the base work that applies a localized neighbor filter to achieve convolutional architecture is the graph convolutional network (GCN) (Kipf & Welling, 2017) .

However, GCN merely considers the first-order neighbors, resting on which multiply layers are directly stacked to learn the multi-scale information, while it has been observed in many experiments that deeper GCN could not improve the performance and even performs worse (Kipf & Welling, 2017) .

In other words, such convolutional filter limits the representational capacity of the model (Abu-El-Haija et al., 2019) .

In this work, we propose a new model, HWGCN, for convolutional filter formulation that is capable of mixing its neighborhood information at different orders to capture the expressive representations from the graph.

Considering that convolution kernels of different sizes may extract different aspects or information from the input images, similarly, the size of convolutional filter plays a very important role for neighborhood mixing in graph convolutions.

Researchers have recently made some attempts to deal with higher-order neighbors for the convolutional filter (Abu-El-Haija et al., 2019; Liao et al., 2019) .

Instead of using adjacency matrix power with potential information overlap at different orders, in our proposed graph model HWGCN, we bring an important insight to leverage node features in addition to graph structure for convolutional filter formulation, which allows a refined architecture to code better with neighbor selection at different distances, and thus learn better node representations from first-order and higher-order neighbors.

Our contributions are four-fold.

Firstly, we analyze the GCN and demonstrate the importance of similarity between first-order and higher-order information.

Secondly, we build the convolutional filters with first-order and higher-order neighbors rather than the local neighborhood considered in previous work.

Thirdly, we leverage Lasso and the information of node features and graph structure to minimize the feature loss between the first-order and higher-order neighbors to effectively aggregate the higher-order information in a weighted, orthogonal, and unsupervised fashion, unlike existing models that merely utilize graph structure.

Fourthly, we conduct comprehensive experimental studies on a number of datasets, which demonstrate that HWGCN can achieve the state-of-the-art results in terms of classification accuracy.

Let G = (V, E, A) be a graph, where V is the set of vertices {v 1 , · · · , v n } with |V| = n, E is the set of edges, and A is its first-order neighbor matrix, also called the adjacency matrix, where A ∈ R n×n and A ij = {0, 1}, i.e., if (v i , v j ) ∈ E, then A ij = 1; otherwise, A ij = 0.

Based on the adjacency matrix A, the diagonal degree matrix D can be defined as

T ∈ R n×c , where for each node v ∈ V, its feature

is a c 0 -dimensional row vector.

In addition, since the Graph Convolution Network (GCN) proposed by Kipf & Welling (2017) is exploited as a base model to facilitate the analysis and understanding of our further proposed approach, we would like to briefly present its architecture here.

The graph convolutional layer is defined as:

where H (i−1) and H (i) are the input and output activations for layer i (i≥1),Ã is a symmetrically normalized adjacency matrix with self-connections A + I, I is the identity matrix,D is the diagonal degree matrix ofÃ, σ is the non-linear activation function (e.g., ReLU), and Θ (i) ∈ R ci−1×ci is the learnable weight matrix for layer i. Given H (0) = X, the GCN model with l layers can be thus defined as:

Here, we are interested in semi-supervised node classification tasks.

To train a GCN model on such a task with c l class labels, the softmax function normalizes the final output matrix Z ∈ R n×c l , where each row represents the probability of c l labels for a node.

The cross-entropy loss can be accordingly evaluated between all the zs and the corresponding nodes with known labels and the weights can be calculated with back propagation using some gradient descent optimization algorithms (e.g., Adam (Kingma & Ba, 2015) ).

In this section, we discuss the details of how we present k th -order adjacency matrices and weight matrices to leverage Lasso, node features and graph structure simultaneously for the convolutional filter formulation, and how the HWGCN model benefits from such elaborated filters with neighborhood information.

Formulating a convolutional filter should allow GCN models to learn the node representation differences among the neighbors at different orders.

However, the GCN proposed by Kipf & Welling (2017) simplified the Graph Laplacian by restricting the filters to merely operate the first-order neighbors around each node, which fails to capture this kind of semantics, even when stacked over multiple layers (Abu-El-Haija et al., 2019; Zhou et al., 2018) .

To put it into perspective, we evaluate the performance of GCN of different layers on Cora, and the results are illustrated in Table 1 .

We can observe that deeper GCN models are unable to improve the performance of node classification and even harm the prediction.

That is to say, stacking multiple layers with one-hop message propagation is not necessary to yield an advantage to learn the latent information from higher-order neighbors.

From the point of view of the convolution operations over the images, the filters of different neighborhood sizes generally contribute to greater flexibility and thus better performance on various computer vision tasks, while such approximations into graph domain have been scarce with some exceptions that few non-spectral approaches attempted to take advantage of larger yet fixed filter size (Atwood & Towsley, 2016; Niepert et al., 2016) which are somewhat unsatisfying for spectral filter extension; Abu-El-Haija et al. (2019) designed MixHop to mix the feature representations of higherorder neighbors in one graph convolution layer.

Since A is a nonnegative matrix, if A ij > 0, then its matrix power A k ij would be positive, such that A and A k have non-zero elements in the same positions of matrices.

This implies, the layer output may impose the lower-order information on higher orders and increase the feature correlations.

To further explain this, we present our Theorem 1 with proof and analysis left in Appendix A. Theorem 1.

Let A p and A q denote the adjacency matrix A multiplied by itself p and q times respectively where p, q ∈ N * and p < q, then A p • A q = 0 if there are two walks between node v i and v j where the length of two walks are p and q respectively.

To this end, we would like to formulate a convolutional filter using first-and higher-order neighbors, so that the lower-order neighborhood information will be distinctively mixed with higher orders, while not overlapping from each other.

We first introduce the concept of k th -order adjacency matrix.

Definition 1.

k th -order adjacency matrix.

Given a graph G = (V, E, A), we use the shortest path distance to determine the order between each pair of nodes.

Let the shortest path distance between node v i and node v j be d ij , and k th -order adjacency matrix be A (k) ∈ R n×n , so that the element

Based on the definition, the k th -order adjacency matrix A (k) can be accordingly denoted as follows:

Corollary 1.

Let A (p) and A (q) are p th -order and q th -order adjacency matrices respectively where p, q ∈ N * and p = q, then

The proof of Corollary 1 can be found in Appendix B. Given the adjacency matrices of different orders, a naive solution to formulate the filter is to add all the k th -order adjacency matrices A (k)

(1 < k ≤ K)

to A. However, this solution could generate an extremely dense matrix to propagate the noisy information from increasing number of expanded neighbors over layers, and yet make no distinction among neighbors at different orders.

Note that, different neighbors (i.e., at different orders or different positions in the same order) contribute to the node semantics differently.

Thus, a more sophisticated solution to formulate the filter is to assign different weights to higher-order neighbors specifying their layer-wise and node-wise importances, so that each non-zero element in k th -order adjacency matrix will be represented as a weight.

Following k th -order adjacency matrix's definition, we introduce k th -order weight matrix as:

Definition 2.

k th -order weight matrix.

Let the shortest path distance between node v i and node v j be d ij , and k th -order weight matrix be

Accordingly, the k th -order weight matrix W (k) can be formulated as follows:

As such, k th -order weight matrix has several significant properties: (1) W (p) • W (q) = 0 for any pair of p th -order and q th -order weight matrices; (2) w k ij is learnable to specify the importance of each higher-order neighbor; (3) W (k) is more sparse than A (k) since w k ij could be minimized to zero using some optimizations.

Considering that A approximates the most direct and effective layer-wise modeling capacity (Kipf & Welling, 2017) , we therefore propose to add all the k th -order weight matrices W (k) (2 < k ≤ K) to A, so that the graph Laplacian can be formed asD

,W = W +I andD w is the degress matrix ofW .

The difference between the first-order graph convolution and our proposed graph convolution model with two layers is shown in Figure 1 .

We will discuss how we adopt Lasso idea to determine the value of each element of the k th -order weight matrix in section 3.2.

In statistics, feature selection is to select a subset of relevant explanatory variables to describe a response variable (Fonti & Belitser, 2017) .

Robert Tibshirani proposed the Lasso (Tibshirani, 1996) to perform this process using 1 -norm which can shrink some coefficients and set others to zero.

Inspired by its effectiveness on variable selection, here we apply Lasso for higher-order neighbor selection.

As we can see, the first-order neighbors have a close relationship and share some significant commonalities with the central node, and thus have less noisy labels, which acts as a basic and important assumption to GCN.

Therefore, we use the first-order neighbors as the observed features and extract the higher-order neighbors with feature vectors paralleled to the aggregated feature vectors of first-order neighbors in n-dimensional space.

Accordingly we propose the following method:

where the affinity matrixS =D

.

L i can be transformed to an optimization problem as follows (Breiman, 1995) :

λ is the parameter that controls the strength of the penalty, where the larger the penalty value of λ, the greater the amount of shrinkage.

When λ goes to infinity, t becomes 0 and all coefficients shrink to 0.

Note that, the sum of coefficients n j=1 |W (k) ij | controls the potential scale of the feature vector, and thus we need to effectively restrict its value.

Due to its penalty nature, we can set λ to a large value and shift the sum (i.e., n j=1 |W (k) ij |) to a constant value α instead of the cross validation for λ.

In this respect, we can obtain the l 1 optimization problem as follows:

where α i is the sum of the scale coefficients for k th -order neighbors of node i.

Considering that the number of higher-order neighbors may be different from that of first-order neighbors, it is necessary for us to perform scale transformation for the higher-order neighbors.

Given the scale of the aggregated feature vector of first-order neighbors, we can naturally leverage it to control the scale of feature vector for higher-order neighbors.

To this end, we propose the following loss function:

where

i is the scale coefficient for node i. By solving

as follows:

Accordingly, L i has the following form: Stellato et al. (2017) proposed the OSQP which is a fast method (Banjac et al., 2017; based on ADMM (Alternating Direction Method of Multipliers) to solve this quadratic problem.

We can assign different weights to different neighbors by solving this optimization problem.

Since the Pubmed dataset in table 3 is a large graph and each node has too many higher-order neighbors that may mix features from unrelated neighbors, we sort by the weight of the neighbor nodes, and select a certain proportion of neighbor nodes to form the final neighbor weight matrix.

This may improve the performance and reduce training time.

The proportion of neighbor nodes for Pubmed is as follows:

In this section, we evaluate the performance of our proposed model HWGCN on a number of datasets.

We perform experiments on the graph-based benchmark node classification tasks with the random splits and fixed split of each dataset, and compare our model with a wide variety of previous approaches and state-of-the-art baselines.

We test our model on three citation network benchmark datasets: Cora, Citeseer and Pubmed Kipf & Welling, 2017) -in all of these datasets, nodes represent documents and edges denote citation links; node features correspond to elements of a bag-of-words representation of a document i.e., 0/1 values indicating the absence/presence of a certain word, while each node has a class label (Veličković et al., 2018) .

The dataset statistics are summarized in Table 3 .

We compare our approach against some previous methods and state-of-the-art baselines, including: label propagation (LP) (Zhu et al., 2003) , semi-supervised embedding (SemiEmb) (Weston et al., 2012) , manifold regularization (ManiReg) (Belkin et al., 2006) , skip-gram based graph embeddings (DeepWalk) (Perozzi et al., 2014) , iterative classification algorithm (ICA) (Lu & Getoor, 2003) , and Planteoid on fixed-split datasets; multi-layer perceptron, i.e., MLP (without adjacency matrix), graph attention networks (GAT) (Veličković et al., 2018) , plain GCN (Kipf & Welling, 2017) and MixHop (Abu-El-Haija et al., 2019) on both fixed-split and random-split datasets.

We closely follow the experimental setup in Kipf & Welling (2017) to implement our model for evaluation.

The model HWGCN, a two-layer GCN structure with 16 hidden units, is trained through 200 maximum epochs using Adam (Kingma & Ba, 2015) with 0.01 initial learning rate, 5 × 10 −4 L2 regularization on the weights, and 0.5 dropout rate for the input and hidden layers.

The parameter settings of GAT and MixHop are directly taken from (Veličković et al., 2018) and (Abu-El-Haija et al., 2019).

For datasets, we adjust the random splits of Cora, Citeseer, and Pubmed respectively to align with three different scenarios: 5, 10 or 20 instances for each class are randomly sampled as training data while another 500 and 1000 instances are selected as validation and test data.

In addition, in order to fairly assess the benefits of our convolutional filter formulation mechanism with higher-order neighbor information, and to achieve the best performance, we further compare the results by increasingly adding different numbers of weight matrices W (k) to A (i.e., A+

s.t.

k ≥ 2) to determine the effective neighborhood distance.

Accordingly, we conduct such experiments using 20 instances for each class as training data and 1000 instances as test data.

The results are illustrated in Figure 2 , from which we can observe that, mixing neighborhood information at different distances in a graph show different performances for node classification; we obtain the best results with adding weight matrices to k ∈ {4, 5, 6} for Cora, k = 6 for Citeseer and k = 5 for Pubmed respectively.

Based on this observation, we therefore set k = 6 for Cora, k = 6 for Citeseer, and k = 5 for Pubmed in the following experiments.

The results of our comparative evaluation experiments on random-split datasets are summarized in Table 4 .

We report the mean classification accuracy on the test nodes of our method after 50 runs over random dataset splits with 5, 10 or 20 labels for per class.

From the results, we can see that our proposed model HWGCN successfully outperforms previous approaches and state-of-the-art baselines in most cases on random splits, where the best result for each column has been highlighted in bold.

More specifically, compared to the best performances from MLP, GAT, GCN and MixHop, HWGCN manages to improve the accuracy by a margin of 0.1% to 0.9% on Cora with respect to different data splits.

For Citeseer, and Pubmed which is a larger graph with more nodes, though MixHop that mixes neighborhood information at different distance performs slightly better when the training size is 5 per class (on Citeseer) and 10 per class (on Pubmed), HWGCN is still able to outperform the spectral methods such as GCN and GAT that limit to operating first-order neighborhood.

It is worth noting that we do not complicate our model to achieve such superior performance, which is trained as the same architecture as GCN.

The success of our model lies in the proper consideration and accommodation of higher-order neighborhood information, and the advantage of weight matrix formulation to decrease the noises and thus improve the expressiveness of node representations.

We also conduct experiments on fixed split and report the mean accuracy of 100 runs.

Note that, for comparison purposes, we directly take the results of previous approaches including ManiReg, SemiEmb, LP, DeepWalk, ICA, Planteoid and MLP already reported in the original GCN paper (Kipf & Welling, 2017) and GAT from paper (Veličković et al., 2018) .

The results are summarized in Table 5 .

Though it slightly falls behind GAT on Cora and Citeseer, and Mixhop on Pubmed, HWGCN still achieves state-of-the-art performance, which is better than GCN.

Furthermore, based on the comprehensive results in Table 4 and 5, we can see that our model using Lasso to select relevant higher-order neighbors is beneficial to alleviate the problem of overfitting.

For non-spectral approaches that generalize convolutions to the graph domain, convolutions are directly defined on the graph, operating on spatially close neighbors (Zhou et al., 2018) .

Duve- (2015) proposed convolutional networks to learn molecular fingerprints where information flows between neighbors in the graph.

Atwood & Towsley (2016) proposed the diffusionconvolution neural networks (DCNNs), which propagates features based on the transition matrix power series.

Both approaches use a different number of neighbors among all nodes to extract local features.

By contrast, Niepert et al. (2016) extracted local features using fixed number of neighbors for each of the nodes, while Monti et al. (2017) proposed a unified framework allowing to generalize CNN architectures to non-Euclidean domains.

In addition, some researchers applied the pooling operation on graph.

For example, designed a novel SortPooling layer to filter the outputs of the convolutional layer, and Gao & Ji (2019) also proposed novel graph pooling and unpooling, which allows neighbor selection for the central node.

The aforementioned approaches define the convolution and pooling by performing aggregation and filtering respectively over the neighbors of each node, yielding impressive performance on node, link and graph classification.

Some researchers also define the convolution operations based on the spectral formulation, which are called spectral approaches.

Bruna et al. (2014) generalized the convolution operator in the nonEuclidean domain by exploiting the spectrum of the graph Laplacian; this approach involves intense computation and uses non-spatially localized filters.

Defferrard et al. (2016) proposed to approximate the filters by computing the Chebyshev polynomial recurrently, generating fast and localized spectral filters.

Kipf & Welling (2017) further introduced the graph convolutional network (GCN) via limiting the spectral filters to first-order neighbors for each node.

As mentioned earlier, GCN is the model on which our work is based.

The most relevant work to our approach is MixHop (Abu-El-Haija et al., 2019), which repeatedly mixed feature representations of neighbors at various distances.

The major distinction is that we formulate the convolutional filter in a more sophisticated way to leverage node features and graph structure from higher-order neighbors while avoiding potential neighborhood information overlaps, as has been analyzed in more details in Section 3.

The original GCN updates the state of nodes by the aggregation of feature information from directly neighboring nodes in every convolutional layer, but fails to learn the higher-order neighborhood information through operating multiply layers; its performance suffers a drop-off when it adjusts the number of layers over two.

To address this, some recent research efforts have been conducted on mixing neighborhood information at different distances to improve the expressive power of graph convolutions, which are promising yet limiting to adjacency matrix power.

In this paper, we propose a novel model HWGCN to formulate the convolutional filter to regularize first-order and higherorder neighbors in a weighted and orthogonal fashion, where node features and graph structure are leveraged to minimize feature loss through Lasso, extract relevant higher-order neighborhood information, and thus learn better node representations.

Our method is a generic framework which can be further applied to various graph convolution network models.

The experimental results based on the three standard citation network benchmark dataset demonstrate state-of-the-art performance being achieved, which match or outperform other baselines.

To prove Theorem 1, We first give Lemma 1 with proof as follows:

is the number of walks from v i to v j in G of length k where 1 ≤ i ≤ j ≤ |V|.

Proof.

We prove it by induction.

When k = 1, if A ij = 1, there is a walk from v i to v j of length 1.

We assume by induction that (A k−1 ) ij is the number of walks from v i to v j of length k − 1, and we

Based on the induction hypothesis, (A k−1 ) im is the number of walks of length k − 1 between v i and v m .

A mj = 1 if m and j connect, and (A k−1 ) im A mj is the number of walks from v i to v j of length k with v m as their penultimate vertex.

By summing the number of walks with all vertices in graph G as their penultimate vertex, we can accordingly get the number of walks from v i to v j .

As we have proved Lemma 1, we further provide the proof of Theorem 1 as follows:

is the number of walks from v i to v j in G of length k as stated in Lemma 1, if there are two walks between node v i and v j where the length of two walks are p and q respectively, then (A p ) ij > 0 and (A q ) ij > 0.

Therefore, we have (

By the theory, we can find that different matrix powers may have non-zero elements in the same position of matrix.

Especially when there is a circle in graph as shown in Figure 3 , the corresponding elements appear cyclically in higher-order matrices.

This accordingly results in information overlap between lower-order matrix and higher-order matrix.

We also report the accuracy curve for training size ∈ {5, 10} per class on the three datasets.

The results are illustrated in Figure 4 and Figure 5 for another two random splits, and Figure 6 for fixed split.

We replace the distance matrix with matrix powers and report the accuracy curve for training size ∈ {5, 10, 20} per class on random splits of the three datasets.

And the results are illustrated in Figure 7 , Figure 8 and Figure 9 .

Lasso with l 1 -norm can get the sparse solution, and here we provide the weight absolute value statistics for W (k) in Table 6, Table 7 and Table 8 .

From the results, we can observe that large weight absolute values only take up a small portion.

@highlight

We propose HWGCN to mix the relevant neighborhood information at different orders to better learn node representations.

@highlight

The authors propose a variant of GCN, HWGCN, to consider convolution beyond 1-step neighbors, which is comparable to state-of-the-art methods.
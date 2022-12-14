Graph Neural Networks as a combination of Graph Signal Processing and Deep Convolutional Networks shows great power in pattern recognition in non-Euclidean domains.

In this paper, we propose a new method to deploy two pipelines based on the duality of a graph to improve accuracy.

By exploring the primal graph and its dual graph where nodes and edges can be treated as one another, we have exploited the benefits of both vertex features and edge features.

As a result, we have arrived at a framework that has great potential in both semisupervised and unsupervised learning.

Convolutional Neural Networks (CNNs) (Lecun et al. (1998) ) has been very successfully used for automated feature extraction in Euclidean domains, especially for computer vision, such as 2D image classification, object detection, etc.

However, many real-life data has a non-Euclidean graph structure in nature, from which we want to investigate the underlying relations among different objects by utilizing the representation of nodes and edges.

Recently, research on applying the generalization of Convolutional Neural Networks to the non-Euclidean domains has attracted growing attention.

As a result, a branch of research on Geometric Deep Learning (Bruna et al. (2013) ) based on that has been ignited.

Previous works including ChebNet (Defferrard et al. (2016) ) and GCN (Kipf & Welling (2017) ) have demonstrated strong results in solving problems in semi-supervised learning where the labels of only a few objects are given, and we want to find out the labels of other objects through their inner connections.

Current methods generalizing convolution operations include both spatial and spectral domains (Bruna et al. (2013) ).

The spatial one deals with each node directly in the vertex domain while the spectral one takes a further step in converting signals via graph Fourier transform into the spectral domain.

However, one critical weakness would be the fact that the interchangeable and complementary nature between nodes and edges are generally ignored in previous research.

As a result, the duality of the graph is not fully utilized.

If we treat those edges in the original, or known as the primal graph, as the nodes in the new graph, and original nodes as edges, we can arrive at a new graph that further exploits the benefits of edge features.

In such a way, we are able to get both the primal graph and the dual graph (Monti et al. (2018) ).

By combining both the vertex features and the edge features, we will be able to solve wider range of problems and achieve better performance.

In this paper, we propose a new approach to transform the primal graph into its dual form and have implemented two pipelines based on these two forms of graph to improve the accuracy and the performance.

With two pipelines, we also exploited a path to make the model wider instead of merely deeper.

Meanwhile, we have developed a new framework that can be applied later on both semi-supervised learning and unsupervised learning.

Graph-based semi-supervised learning aims to annotate data from a small amount of label data on a graph.

To learn the vectors that can recover the labels of the training data as well as distinguish data with different labels, conventionally, graph Laplacian regularizer gives penalty between sampling based on graph Laplacian matrix (Zhu et al. (2003) ; Ando & Zhang (2007) ; Weston et al. (2012) ).

Sample-based method takes random walk to get samples from the context of data points in order to propagate information (Perozzi et al. (2014) ; Yang et al. (2016) ; Grover & Leskovec (2016) ).

Graph Convolutional Networks generalize the operation of convolution from grid data to graph data (Wu et al. (2019) ).

After the emergence of the spectral-based convolutional networks on graph (Bruna et al. (2013) ), ChebNet (Defferrard et al. (2016) ) approximate the filters by Chebyshev polynomials according to the Laplacian eigendecomposition.

GCN (Kipf & Welling (2017) ) simplifies ChebNet by introducing its first-order approximation and can be viewed as a spatial-based perspective, which requires vertices in the graph to propagate their information to the neighbors.

MoNet (Monti et al. (2017) ) is a spatial-based method, of which convolution is defined as a Gaussian mixture of the candidates.

GAT (Veli??kovi?? et al. (2017) ) applies the attention mechanism to the graph network.

DGI (Veli??kovi?? et al. (2018) ) proposes a framework to learn the unsupervised representations on graph-structured data by maximizing location mutual information.

We refer to Zhou et al. (2018) ; Xu et al. (2018); Battaglia et al. (2018) ; Wu et al. (2019) as a more comprehensive and thorough review on graph neural networks.

Dual approaches on graph networks usually unlike the above mono-methods, apply mixed methods to study graph networks.

DGCN (Zhuang & Ma (2018) ) makes a balance between the spatialbased domain and spectral-based domain by regularizing their mutual information.

GIN (Yu et al. (2018) ) proposes a dual-path from graph convolution on texts and another network on images to gather cross-modal information into a common semantic space.

DPGCNN (Monti et al. (2018) ) extends the classification on vertices to edges by considering the attention mechanism on both.

Our study follows this path, which classifies vertices from the relationship between them (edges) and regularization from the mutual information between classification on both vertices and edges.

3.1 PRELIMINARIES Let G = {V, E, A} denote a graph, where V = {1, . . .

, N } is the set of nodes with |V| = N , E is the set of edges, and A = (A (i,j)???V = 0) ??? R N ??N is the adjacency matrix.

When G is undirected then A is symmetric with A i,j = A j,i , G is an undirected graph, otherwise a directed graph.

The Laplacian matrix, also acts a propagation matrix, has the combinatorial form as L = D ??? A ??? R N ??N , and its normalized form is

N ??N is the degree matrix of graph G with d(i) = j???V A i,j and I ??? R N ??N is the identity matrix.

In some literature, the random walk Laplacian L rw = I ??? D ???1 A is employed to directed graph G.

Let L = U ??U T be the eigendecomposition, where U ??? R N ??N is composed of orthonormal eigenbasis and ?? = diag(?? 0 , ?? 1 , . . .

, ?? N ???1 ) is a diagonal matrix of eigenvalues which denotes frequencies of graph G , and ?? i and u i form an eigenpair.

The convolutional operator * G on the graph signal x is defined by

wheref = U T x and?? = U T g are regarded as the graph Fourier transform of graph signal x and graph filter g, respectively; f = U (??) is the inverse graph Fourier transform, and is the Hadamard product.

?? = diag(?? 0 , ?? ?? ?? ,?? N ???1 ) behaves as spectral filter coefficients.

Graph convolution can be approximated by polynomial filters, the k-th order form is

where

Based on the above approximation, ChebNet (Defferrard et al. (2016) ) further introduces Chebyshev polynomials into graph filters of the convolutional layers for the sake of computational efficiency.

Chebyshev polynomials are recursively expressed as T i (x) = 2xT i???1 (x) ??? T i???2 (x) with T 0 (x) = 1 and T 1 (x) = x. The graph filter then becomes

whereL = 2/?? max L ???

I denotes the scaled normalized Laplacian for all eigenvalues ?? i ??? [???1, 1] and ?? i is trainable parameter.

Graph Convolutional Network (GCN) (Kipf & Welling (2017)) is a variant of ChebNet which only takes first two terms of Equation (3).

By setting the coefficients ?? 0 and ?? 1 as ?? = ?? 0 = ????? 1 and with ?? max = 2, the convolution operator in convolution layer of GCN is induced as g In graph theory, The definition of the dual varies according to the choice of embedding of the graph G. For planar graphs generally, there may be multiple dual graphs, depending on the choice of planar embedding of the graph.

In this work, we follow the most common definition.

Given a plane graph G = {V, E A}, which is designated as the primal graph, the dual graph?? = {??? = E,???,??} is a graph that has a vertex (or node) for each edge of G. The dual graph?? has an edge whenever two edges of G share at least one common vertex.

To be clarified, the vertices (i, j) and (j, i) of dual graph?? converted from a undirected graph are regarded as the same.

Fig.1 shows the conversion from primal graph to its dual counterpart.

When vertices of the primal graph embed features (or signals in terminology of spectral graph theory), the features of a dual node can be obtained by applying a specified functions to its corresponding primal nodes' features, i.e. the simplest applicable function is to calculate the distance between the features of two nodes.

In addition, if the edges of primal graph possess features or attributes, we also take them into account as the their inherited features of dual nodes.

Take node (1, 2) of dual graph in Fig.1b) as an example, its feature is obtained by performing the element-wise subtraction to the feature vectors of nodes 0 and 3 of primal graph in Fig.1a) , i.e.

[1, 1, 0] T ??? [1, 0, 1] T = [0, ???1, 1] T .

The Twin Graph Convolutional Networks (TwinGCN) proposed in this work consists of two pipelines.

Both pipelines are built with the same architecture as GCN, and contain two convolution layers in each pipeline, as shown in Fig.2 .

The upper pipeline acts exactly as GCN; however, the lower one takes the dual featuresX derived from primal features X as its inputs (as described in section 3.3), the predictions or outputs in dual vertex domain (i.e. edge domain in primal) is then aggregated to primal vertex domain.

The goal of introducing a dual pipeline into the model is that we desire to utilize the predictions on the dual node (edges in primal graph) to affect the predictions on primal nodes since the knowledge about those neighbors of a node can be propagated through edges.

For the purpose of training the dual pipeline, we also need to get the labels of dual nodes.

Let us take an example, given a dual node (i, j) (corresponds to an edge in primal graph), primal node i has label ?? and j has label ??, then dual node (i, j) is assigned with a label (??, ??).

One thing worth mentioned is that TwinGCN's convolution layers are not limited to those used in GCN, they can be replaced with other types of convolution layer, such as ChebNet, GWNN (Xu et al. (2019) ), etc.

The convolution layers in the pipelines perform graph convolution operations with shared weights as learnable parameters, mathematically expressed as

where H (l) is the activation in l-th layer, W (l) is learnable weights in that layer.

?? represents nonlinear activation function, e.g. ReLU.

For the task of semi-supervised node classification, the loss function is defined as

where Y L is set of node labels for L ??? V labeled node set, F denotes the number of labels of the nodes, and Z is predicted outcome, a softmax of the output of the network.

In order to take effect of dual pipeline on the prediction of primal pipeline, we adopt KullbackLeibler Divergence (D KL ) as a regularization term in training.

Suppose that P (Y |X) is predictions by primal pipeline and P (?? |X) = P (?? |X) is the derived predictions obtained through an aggregation from the predictions on dual labels by dual pipeline to primal label predictions.

X is derived from X as aforementioned (Section 3.3).

We first calculate the joint probability matrix P (Y,?? ) of two matrices P (Y |X) and P (?? |X)

we further get the marginal probabilities of P (Y ) and P (?? ) from P (Y,?? ).

KullbackLeibler Divergence D KL is evaluated by

finally, we attains the loss function as 3 illustrates a fast algorithm deriving primal predictions from predictions of dual pipeline.

It is conducted by introducing two special incidence matrices.

The matrix at the left hand side (N ?? M , N = |V| and M = |E|) is an incidence matrix in which the rows represent primal nodes, each column depicts whether a primal node in a row has an incidence in the dual node represented by this column.

The rightmost matrix is the incidence matrix of primal labels presenting in dual labels with dimension of L 2 ?? L. Although these two matrices are extremely sparse when node number is very large (we store them in compressed form), by taking advantage of GPU's powerful computing capability, the delicate sparse matrix multiplication subroutine, e.g. Nvidia's cuSARSE, runs much faster than codes with loops for lumping the incidences.

In this section, we evaluate the performance of TwinGCN, we mainly focus on semi-supervised node classification in current work.

Actually, TwinGCN also support unsupervised learning by changing the loss functions which we will fulfill in future work.

We conduct experiments on three benchmark datasets and follow existing studies (Defferrard et al. (2016) ; Kipf & Welling (2017); Xu et al. (2019) etc.)

The datasets include Cora, Citeseer, and Pubmed (Sen et al. (2008) ).

All these three datasets are collected from their corresponding citation networks, the nodes represent documents and edges are the citations.

Table 4 .1 shows details of these datasets.

Label rate indicates the portion of the available labeled nodes used for training.

The training process takes 20 labeled samples for each class for every dataset.

Since both pipelines of our proposed architecture work with graph convolution based on spectral graph theory, we use recent works, such as ChebNet (Defferrard et al. (2016) ) GCN (Kipf & Welling (2017)), and GWNN (Xu et al. (2019) ), etc.

These models maintain the same graph Lapalacian base structure, unlike some other methods take partial graph structure, e.g. FastGCN (Chen et al. (2018) ) applies Monte Carlo importance sampling on edges.

however, this kind of method only guarantees the convergence as the sample size goes to infinity.

For the sake of consistency for comparison, the hyper-parameters for training are kept the same for primal pipeline as other models.

The primal are composed with two graph convolution layers with 16 hidden units and applied with ReLU non-linear activations.

Loss is evaluated with the softmax function.

Dropout (Srivastava et al. (2014) ) of primal pipeline is set to p = 0.5 for the primal.

We use the Adam optimizer (Kingma & Ba (2015)) for optimizing the weights with an initial learning rate lr = 0.01.

As the dual graph is normally much bigger than the counterpart primal graph, its adjacency/Laplacian matrix and the number of dual nodes becomes quadratically larger, e.g. N nodes with N ?? (N ??? 1) edges in a fully-connected graph.

Therefore, to avoid overfitting on dual pipeline, we set its dropout rate higher than 70%.

We also introduce a sampling rate to extract a small fraction from the total dual node labels.

Having a large number of edges in the primal graph also means a large number of dual nodes.

In such situation, the performance will be degraded severely.

The quantitative comparison among different models is given in Table 4 .4.

For node classification, TwinGCN achieves similar results or outperforms with some datasets.

The performance gain comes from the aggregation of knowledge propagated through edges (or dual nodes) trained by dual pipeline.

However, primal pipeline only will ignore the dependency between labels of nodes.

Fig.4a ) illustrate that when compared to the GCN, TwinGCN bearing two pipelines converges slower but achieves a higher accuracy as the number of epoch increases.

This is because that we have two pipelines through mutual interaction.

In Fig.4b ), we observe that two loss curves of traditional GCN and TwinGCN have very similar decreasing trends.

However, the loss curve of TwinGCN is slightly above GCN because the loss of TwinGCN is the summation of both primal and dual pipelines.

To test whether the introduced dual pipeline and regularization improve the basic GCN pipeline, we conducted controlled experiments to make comparison among GCN, GCNs with pipelines on original graph and dual graph and TwinGCN(GCNs with both pipelines and regularization by KLdivergence).

Method Cora Citeseer Pubmed GCN 81.5% ?? 0.3% 70.8% ?? 0.1% 78.8% ?? 0.1% GCN(double-pipeline) 81.6% ?? 0.4% 72.5% ?? 1.0% 79.8% ?? 2.3% TwinGCN 83.0% ?? 1.3% 72.5 ?? 0.8% 79.5% ?? 1.2%

In this work, we propose the TwinGCN with parallel pipelines working on both the primal graph and its dual graph, respectively.

TwinGCN achieves the state-of-the-art performance in semisupervised learning tasks.

Moreover, TwinGCN's ability is not limited to this, we can extend its power/utilization into unsupervised learning by altering its loss functions.

Use unnumbered third level headings for the acknowledgments.

All acknowledgments, including those to funding agencies, go at the end of the paper.

<|TLDR|>

@highlight

A primal dual graph neural network model for semi-supervised learning
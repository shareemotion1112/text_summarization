Graph neural networks have shown promising results on representing and analyzing diverse graph-structured data such as social, citation, and protein interaction networks.

Existing approaches commonly suffer from the oversmoothing issue, regardless of whether policies are edge-based or node-based for neighborhood aggregation.

Most methods also focus on transductive scenarios for fixed graphs, leading to poor generalization performance for unseen graphs.

To address these issues, we propose a new graph neural network model that considers both edge-based neighborhood relationships and node-based entity features, i.e. Graph Entities with Step Mixture via random walk (GESM).

GESM employs a mixture of various steps through random walk to alleviate the oversmoothing problem and attention to use node information explicitly.

These two mechanisms allow for a weighted neighborhood aggregation which considers the properties of entities and relations.

With intensive experiments, we show that the proposed GESM achieves state-of-the-art or comparable performances on four benchmark graph datasets comprising transductive and inductive learning tasks.

Furthermore, we empirically demonstrate the significance of considering global information.

The source code will be publicly available in the near future.

Graphs are universal data representations that exist in a wide variety of real-world problems, such as analyzing social networks (Perozzi et al., 2014; Jia et al., 2017) , forecasting traffic flow (Manley, 2015; Yu et al., 2017) , and recommending products based on personal preferences (Page et al., 1999; Kim et al., 2019) .

Owing to breakthroughs in deep learning, recent graph neural networks (GNNs) (Scarselli et al., 2008) have achieved considerable success on diverse graph problems by collectively aggregating information from graph structures Xu et al., 2018; Gao & Ji, 2019) .

As a result, much research in recent years has focused on how to aggregate the feature representations of neighbor nodes so that the dependence of graphs is effectively utilized.

The majority of studies have predominantly depended on edges to aggregate the neighboring nodes' features.

These edge-based methods are premised on the concept of relational inductive bias within graphs (Battaglia et al., 2018) , which implies that two connected nodes have similar properties and are more likely to share the same label (Kipf & Welling, 2017) .

While this approach leverages graphs' unique property of capturing relations, it appears less capable of generalizing to new or unseen graphs (Wu et al., 2019b) .

To improve the neighborhood aggregation scheme, some studies have incorporated node information; They fully utilize node information and reduce the effects of relational (edge) information.

A recent approach, graph attention networks (GAT), employs the attention mechanism so that weights used for neighborhood aggregation differ according to the feature of nodes (Veli??kovi?? et al., 2018) .

This approach has yielded impressive performance and has shown promise in improving generalization for unseen graphs.

Regardless of neighborhood aggregation schemes, most methods, however, suffer from a common problem where neighborhood information is considered to a limited degree (Klicpera et al., 2019) .

For example, graph convolutional networks (GCNs) (Kipf & Welling, 2017) only operate on data that are closely connected due to oversmoothing , which indicates the "washing out" of remote nodes' features via averaging.

Consequently, information becomes localized and access to global information is restricted (Xu et al., 2018) , leading to poor performance on datasets in which only a small portion is labeled .

In order to address the aforementioned issues, we propose a novel method, Graph Entities with Step Mixture via random walk (GESM), which considers information from all nodes in the graph and can be generalized to new graphs by incorporating random walk and attention.

Random walk enables our model to be applicable to previously unseen graph structures, and a mixture of random walks alleviates the oversmoothing problem, allowing global information to be included during training.

Hence, our method can be effective, particularly for nodes in the periphery or a sparsely labeled dataset.

The attention mechanism also advances our model by considering node information for aggregation.

This enhances the generalizability of models to diverse graph structures.

To validate our approach, we conducted experiments on four standard benchmark datasets: Cora, Citeseer, and Pubmed, which are citation networks for transductive learning, and protein-protein interaction (PPI) for inductive learning, in which test graphs remain unseen during training.

In addition to these experiments, we verified whether our model uses information of remote nodes by reducing the percentage of labeled data.

The experimental results demonstrate the superior performance of GESM on inductive learning as well as transductive learning for datasets.

Moreover, our model achieved enhanced accuracy for datasets with reduced label rates, indicating the contribution of global information.

The key contributions of our approach are as follows:

??? We present graphs with step mixture via random walk, which can adaptively consider local and global information, and demonstrate its effectiveness through experiments on public benchmark datasets with few labels.

??? We propose Graph Entities with Step Mixture via random walk (GESM), an advanced model which incorporates attention, and experimentally show that it is applicable to both transductive and inductive learning tasks, for both nodes and edges are utilized for the neighborhood aggregation scheme.

??? We empirically demonstrate the importance of propagation steps by analyzing its effect on performance in terms of inference time and accuracy.

Step-0 Step-1 Step- Figure 1: Random walk propagation procedure.

From left to right are step-0, step-1, step-2, and stepinfinite.

The values in each node indicate the distribution of a random walk.

In the leftmost picture, only the starting node has a value of 100, and all other nodes are initialized to zero.

As the number of steps increases, values spread throughout the graph and converge to some extent.

Random walk, which is a widely used method in graph theory, mathematically models how node information propagates throughout the graph.

As shown in Figure 1 , random walk refers to randomly moving to neighbor nodes from the starting node in a graph.

For a given graph, the transition matrix P , which describes the probabilities of transition, can be formulated as follows:

where A denotes the adjacency matrix of the graph, and D the diagonal matrix with a degree of nodes.

The probability of moving from one node to any of its neighbors is equal, and the sum of the probabilities of moving to a neighboring node adds up to one.

Let u t be the distribution of the random walk at step t (u 0 represents the starting distribution).

The t step random walk distribution is equal to multiplying P , the transition matrix, t times.

In other words,

The entries of the transition matrix are all positive numbers, and each column sums up to one, indicating that P is a matrix form of the Markov chain with steady-state.

One of the eigenvalues is equal to 1, and its eigenvector is a steady-state (Strang, 1993) .

Therefore, even if the transition matrix is infinitely multiplied, convergence is guaranteed.

The attention mechanism was introduced in sequence-to-sequence modeling to solve long-term dependency problems that occur in machine translation (Bahdanau et al., 2015) .

The key idea of attention is allowing the model to learn and focus on what is important by examining features of the hidden layer.

In the case of GNNs (Scarselli et al., 2008) , GATs (Veli??kovi?? et al., 2018) achieved stateof-the-art performance by using the attention mechanism.

Because the attention mechanism considers the importance of each neighboring node, node features are given more emphasis than structural information (edges) during the propagation process.

Consequently, using attention is advantageous for training and testing graphs with different node features but the same structures (edges).

Given the many benefits of attention, we incorporate the attention mechanism to our model to fully utilize node information.

The attention mechanism enables different importance values to be assigned to nodes of the same neighborhood, so combining attention with mixture-step random walk allows our model to adaptively highlight features with salient information in a global scope.

Let G = (V, E) be a graph, where V and E denote the sets of nodes and edges, respectively.

Nodes are represented as a feature matrix X ??? R n??f , where n and f respectively denote the number of nodes and the input dimension per node.

A label matrix is Y ??? R n??c with the number of classes c, and a learnable weight matrix is denoted by W .

The adjacency matrix of graph G is represented as A ??? R n??n .

The addition of self-loops to the adjacency matrix is A = A + I n , and the column normalized matrix of A is?? = AD ???1 with?? 0 = I n .

Most graph neural networks suffer from the oversmoothing issue along with localized aggregation.

Although JK-Net (Xu et al., 2018) tried to handle oversmoothing by utilizing GCN blocks with mulitple propagation, it could not completely resolve the issue as shown in Figure 4b .

We therefore propose Graph

Step Mixture (GSM), which not only separates the node embedding and propagation process but also tackles oversmoothing and localized aggregation issues through a mixture of random walk steps.

GSM has a simple structure that is composed of three stages, as shown in Figure 2 .

Input X passes through a fully connected layer with a nonlinear activation.

The output is then multiplied by a normalized adjacency matrix?? for each random walk step that is to be considered.

The results for each step are concatenated and fed into another fully connected layer, giving the final output.

The entire propagation process of GSM can be formulated as:

where is the concatenation operation, s is the maximum number of steps considered for aggregation, and?? k is the normalized adjacency matrix?? multiplied k times.

As can be seen from Equation 3, weights are shared across nodes.

In our method, the adjacency matrix?? is an asymmetric matrix, which is generated by random walks and flexible to arbitrary graphs.

On the other hand, prior methods such as JK-Net (Xu et al., 2018) and MixHop (Abu-El-Haija et al., 2019) , use a symmetric Laplacian adjacency matrix, which limits graph structures to given fixed graphs.

(a) Traditional global aggregation scheme

Step-1

Step-2

Step-3 (b) Our step-mixture scheme For the concatenation operation, localized sub-graphs are concatenated with global graphs, which allows the neural network to adaptively select global and local information through learning (see Figure 3) .

While traditional graph convolution methods consider aggregated information within three steps by A(A(AXW (0) )W (1) )W (2) , our method can take all previous aggregations into account

To develop our base model which depends on edge information, we additionally adopt the attention mechanism so that node information is emphasized for aggregation, i.e., Graph Entity

Step Mixture (GESM).

We simply modify the nonlinear transformation of the first fully connected layer in GSM by replacing it with the attention mechanism denoted by H multi (see Equations 3 and 4).

As described in Equation 4, we employ multi-head attention, where H multi is the concatenation of m attention layers and ?? is the coefficient of attention computed using concatenated features of nodes and its neighboring nodes.

By incorporating attention to our base model, we can avoid or ignore noisy parts of the graph, providing a guide for random walk (Lee et al., 2018) .

Utilizing attention can also improve combinatorial generalization for inductive learning, where training and testing graphs are completely different.

In particular, datasets with the same structure but different node information can benefit from our method because these datasets can only be distinguished by node information.

Focusing on node features for aggregation can thus provide more reliable results in inductive learning.

The time complexity of our base model is O(s ?? l ?? h), where s is the maximum number of steps considered for aggregation, l is the number of non-zero entries in the adjacency matrix, and h is the hidden feature dimension.

As suggested by Abu-El-Haija et al. (2019), we can assume h << l under realistic assumptions.

Our model complexity is, therefore, highly efficient with time complexity O(s ?? l), which is on par with vanilla GCN (Kipf & Welling, 2017) .

Transductive learning.

We utilize three benchmark datasets for node classification: Cora, Citeseer, and Pubmed (Sen et al., 2008) .

These three datasets are citation networks, in which the nodes represent documents and the edges correspond to citation links.

The edge configuration is undirected, and the feature of each node consists of word representations of a document.

Detailed statistics of the datasets are described in Table 1.

For experiments on datasets with the public label rate, we follow the transductive experimental setup of Yang et al. (2016) .

Although all of the nodes' feature vectors are accessible, only 20 node labels per class are used for training.

Accordingly, 5.1% for Cora, 3.6% for Citeseer, and 0.3% for Pubmed can be learned.

In addition to experiments with public label rate settings, we conducted experiments using datasets where labels were randomly split into a smaller set for training.

To check whether our model can propagate node information to the entire graph, we reduced the label rate of Cora to 3% and 1%, Citeseer to 1% and 0.5%, Pubmed to 0.1%, and followed the experimental settings of for these datasets with low label rates.

For all experiments, we report the results using 1,000 test nodes and use 500 validation nodes.

Inductive learning.

We use the protein-protein interaction PPI dataset (Zitnik & Leskovec, 2017) ,which is preprocessed by Veli??kovi?? et al. (2018) .

As detailed in Table 1 , the PPI dataset consists of 24 different graphs, where 20 graphs are used for training, 2 for validation, and 2 for testing.

The test set remains completely unobserved during training.

Each node is multi-labeled with 121 labels and 50 features regarding gene sets and immunological signatures.

For transductive learning, we compare our model with numbers of state-of-the-art models according to the results reported in the corresponding papers.

Our model is compared with baseline models specified in (Veli??kovi?? et al., 2018) such as label propagation (LP) (Xiaojin & Zoubin, 2002) , graph embeddings via random walk (DeepWalk) (Perozzi et al., 2014) , and Planetoid (Yang et al., 2016) .

We also compare our model with models that use self-supervised learning (Union) , learnable graph convolution (LGCN) (Gao et al., 2018) , GCN based multi-hop neighborhood mixing (JK-GCN and MixHop) (Xu et al., 2018; Abu-El-Haija et al., 2019) , multi-scale graph convolutional networks (AdaLNet) (Liao et al., 2019) and maximal entropy transition (PAN) (Ma et al., 2019) .

We further include models that utilize teleport term during propagation APPNP (Klicpera et al., 2019) , conduct convolution via spectral filters such as ChebyNet (Defferrard et al., 2016) , GCN (Kipf & Welling, 2017) , SGC (Wu et al., 2019a) , and GWNN (Xu et al., 2019) and models that adopt attention between nodes, such as GAT (Veli??kovi?? et al., 2018) and AGNN (Thekumparampil et al., 2018) .

For inductive learning tasks, we compare our model against four baseline models.

This includes graphs that use sampling and aggregation (GraphSAGE-LSTM) (Hamilton et al., 2017) , and jumping-knowledge (JK-LSTM) (Xu et al., 2018) , along with GAT and LGCN which are used in the transductive setting.

Regarding the hyperparameters of our transductive learning models, we used different settings for datasets with the public split and random split.

We set the dropout probability such that 0.3 of the data were kept for the public split and 0.6 were kept for the random split.

We set the number of multi-head m = 8 for GESM.

The size of the hidden layer h ??? {64, 512} and the maximum number of steps used for aggregation s ??? {10, 30} were adjusted for each dataset.

We trained for a maximum of 300 epochs with L2 regularization ?? = 0.003 and learning rate lr = 0.001.

We report the average classification accuracy of 20 runs.

For inductive learning, the size of all hidden layers was the same with h = 256 for both GSM, which consisted of two fully connected layers at the beginning and GESM.

We set the number of steps s = 10 for GSM, and s = 5, m = 15 for GESM.

L2 regularization and dropout were not used for inductive learning (Veli??kovi?? et al., 2018) .

We trained our models for a maximum of 2,000 epochs with learning rate lr = 0.008.

The evaluation metric was the micro-F1 score, and we report the averaged results of 10 runs.

For all the models, the nonlinearity function of the first fully connected layer was an exponential linear unit (ELU) (Clevert et al., 2016) .

Our models were initialized using Glorot initialization (Glorot & Bengio, 2010) and were trained to minimize the cross-entropy loss using the Adam (Kingma & Ba, 2015) .

We employed an early stopping strategy based on the loss and accuracy of the validation sets, with a patience of 100 epochs.

Results on benchmark datasets.

Table 2 summarizes the comparative evaluation experiments for transductive and inductive learning tasks.

In general, not only are there a small number of methods that can perform on both transductive and inductive learning tasks, but the performance of such methods is not consistently high.

Our methods, however, are ranked in the top-3 for every task, indicating that our method can be applied to any task with large predictive power.

For transductive learning tasks, the experimental results of our methods are higher than or equivalent to those of other methods.

As can be identified from the table, our base model GSM, which is computationally efficient and simple, outperforms many existing baseline models.

These results indicate the significance of considering both global and local information and using random walks.

It can also be observed that GESM yielded more impressive results than GSM, suggesting the importance of considering node information in the aggregation process.

For the inductive learning task, our base model GSM, which employs an edge-based aggregation method, does not invariably obtain the highest accuracy.

However, our model with attention, GESM, significantly improves performance of GSM by learning the importance of neighborhood nodes, and surpasses the results of GAT, despite the fact that GAT consists of more attention layers.

These results for unseen graphs are in good agreement with results shown by Veli??kovi?? et al. (2018) , in which reducing the influence of structural information improved generalization.

Results on datasets with low label rates.

To demonstrate that our methods can consider global information, we experimented on sparse datasets with low label rates of transductive learning datasets.

As indicated in Table 3 , our models show remarkable performance even on the dataset with low label rates.

In particular, we can further observe the superiority of our methods by inspecting Table 2 and 3, in which our methods trained on only 3% of the Cora dataset outperformed some other methods trained on 5.1% of the data.

Because both GSM and GESM showed enhanced accuracy, it could be speculated that using a mixture of random walks played a key role in the experiments; the improved results can be explained by our methods adaptively selecting node information from local and global neighborhoods, and allowing peripheral nodes to receive information.

Oversmoothing and Accuracy.

As shown in Figure 4a , GCN (Kipf & Welling, 2017) , SGC (Wu et al., 2019a) , and GAT (Veli??kovi?? et al., 2018) suffer from oversmoothing.

GCN and GAT show severe degradation in accuracy after the 8th step; The accuracy of SGC does not drop as much as GCN and GAT but nevertheless gradually decreases as the step size increases.

The proposed GSM, unlike the others, maintains its performance without any degradation, because no rank loss (Luan et al., 2019) occurs and oversmoothing is overcome by step mixture.

Interestingly, JK-Net (Xu et al., 2018) also keeps the training accuracy regardless of the step size by using GGN blocks with multiple steps according to Figure 4a .

We further compared the test accuracy of GSM with JK-Net, a similar approach to our model, in regards to the step size.

To investigate the adaptability to larger steps of GSM and JK-Net, we concatenated features after the 10th step.

As shown in Figure 4b , GSM outperforms JK-Net, even though both methods use concatenation to alleviate the oversmoothing issue.

These results are in line with the fact that JK-Net obtains global information similar to GCN or GAT.

Consequently, the larger the step, the more difficult it is for JK-Net to maintain performance.

GSM, on the other hand, maintains a steady performance, which confirms that GSM does not collapse even for large step sizes.

Test Accuracy GSM public (5.1%) 3% 1% GESM public (5.1%) 3% 1% We also observe the effect on accuracy as the number of steps increases under three labeling conditions for GSM and GESM.

As represented in Figure 5 , it is evident that considering remote nodes can contribute to the increase in accuracy.

By taking into account more data within a larger neighborhood, our model can make reliable decisions, resulting in improved performance.

Inspection of the figure also indicates that the accuracy converges faster for datasets with higher label rates, presumably because a small number of walk steps can be used to explore the entire graph.

Moreover, the addition of attention benefits performance in terms of higher accuracy and faster convergence.

Inference time.

As shown in Figure 6 , the computational complexity of all models increases linearly as the step size increases.

We can observe that the inference time of GSM is faster than GCN (Kipf & Welling, 2017) especially when the number of steps is large.

The inference time of GESM is much faster than GAT (Veli??kovi?? et al., 2018) while providing higher accuracies and stable results (see Appendix A).

Our methods are both fast and accurate due to the sophisticated design with a mixture of random walk steps.

Embedding Visualization.

Figure 7 visualizes the hidden features of Cora from our models by using Figure 7 : t-SNE plot of the last hidden layer trained on the Cora dataset.

the t-SNE algorithm (Maaten & Hinton, 2008) .

The figure illustrates the difference between edgebased and node-based aggregation.

While the nodes are closely clustered in the result from GSM, they are scattered in that of GESM.

According to the results in Table 2 , more closely clustered GSM does not generally produce better results than loosely clustered GESM, which supports findings that the attention mechanism aids models to ignore or avoid noisy information in graphs (Lee et al., 2018) .

Table 5 : Average test set accuracy and standard deviation over 100 random train/validation/test splits with 20 runs.

Top-3 results for each column are highlighted in bold, and top-1 values are underlined.

Coauthor CS Coauthor Physics Amazon Computers Amazon Photo MLP 88.3 ?? 0.7 88.9 ?? 1.1 44.9 ?? 5.8 69.6 ?? 3.8 LogReg 86.4 ?? 0.9 86.7 ?? 1.5 64.1 ?? 5.7 73.0 ?? 6.5 LP (Xiaojin & Zoubin, 2002) 73.6 ?? 3.9 86.6 ?? 2.0 70.8 ?? 8.1 72.6 ?? 11.1 GCN (Kipf & Welling, 2017) 91.1 ?? 0.5 92.8 ?? 1.0 82.6 ?? 2.4 91.2 ?? 1.2 GraphSAGE (Hamilton et al., 2017) 91.3 ?? 2.8

93.0 ?? 0.8 82.4 ?? 1.8 91.4 ?? 1.3 GAT (Veli??kovi?? et al., 2018) 90.5 ?? 0.6 92.5 ?? 0.9 78.0 ?? 19.0 85.7 ?? 20.3 GSM (our base model) 91.8 ?? 0.4 93.3 ?? 0.6 79.2 ?? 2.0 89.3 ?? 1.9 GESM (GSM+attention)

92.0 ?? 0.5 93.7 ?? 0.6 79.3 ?? 1.7 90.0 ?? 2.0

For an in-depth verification of overfitting, we extended our experiments to four types of new node classification datasets.

Coauthor CS and Coauthor Physics are co-authorship graphs from the KDD Cup 2016 challenge 1 , in which nodes are authors, features represent the article keyword for each author's paper, and class labels indicate each author's most active research areas.

Amazon Computers and Amazon Photo are co-purchase graphs of Amazon, where nodes represent the items, and edges indicate that items have been purchased together.

The node features are bag-of-words of product reviews, and class labels represent product categories.

Detailed statistics of the datasets are described in Table 4 and we followed the experimental setup of Shchur et al. (2018) .

We used the same values for each hyperparameter (unified size: 64, step size: 15, multi-head for GAT and GESM: 8) without tuning.

The results in Table 5 prove that our proposed methods do not overfit to a particular dataset.

Moreover, in comparison to GAT, the performance of GESM is more accurate, and more stable.

We visualized the distribution of attention vectors.

Figure 8a plots the distribution of neighbors with equal importance and Figure 8b displays the distribution of attention weighted neighbors that we trained with GESM.

Although both figures look similar to some degree, we can conjecture that GESM slightly adjusts the weight values, contributing to improved performance.

<|TLDR|>

@highlight

Simple and effective graph neural network with mixture of random walk steps and attention
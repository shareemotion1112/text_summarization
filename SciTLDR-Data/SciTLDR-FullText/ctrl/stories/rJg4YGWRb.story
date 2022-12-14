Recently popularized graph neural networks achieve the state-of-the-art accuracy on a number of standard benchmark datasets for graph-based semi-supervised learning, improving significantly over existing approaches.

These architectures alternate between a propagation layer that aggregates the hidden states of the local neighborhood and a fully-connected layer.

Perhaps surprisingly, we show that a linear model, that removes all the intermediate fully-connected layers, is still able to achieve a performance comparable to the state-of-the-art models.

This significantly reduces the number of parameters, which is critical for semi-supervised learning where number of labeled examples are small.

This in turn allows a room for designing more innovative propagation layers.

Based on this insight, we propose a novel graph neural network that removes all the intermediate fully-connected layers, and replaces the propagation layers with attention mechanisms that respect the structure of the graph.

The attention mechanism allows us to learn a dynamic and adaptive local summary of the neighborhood to achieve more accurate predictions.

In a number of experiments on benchmark citation networks datasets, we demonstrate that our approach outperforms competing methods.

By examining the attention weights among neighbors, we show that our model provides some interesting insights on how neighbors influence each other.

One of the major bottlenecks in applying machine learning in practice is collecting sizable and reliable labeled data, essential for accurate predictions.

One way to overcome the problem of limited labeled data is semi-supervised learning, using additional unlabeled data that might be freely available.

In this paper, we are interested in a scenario when this additional unlabeled data is available in a form of a graph.

The graph provides underlying pairwise relations among the data points, both labeled and unlabeled.

Of particular interest are those applications where the presence or absence of an edge between two data points is determined by nature, for instance as a result of human activities or natural relations.

As a concrete example, consider a citation network.

Each node in the graph is a published research paper, associated with a bag-of-words feature vector.

An (directed) edge indicates a citation link.

Presence of an edge indicates that the authors of a paper have consciously determined to refer to the other paper, and hence captures some underlying relation that might not be inferred from the bag-of-words feature vectors alone.

Such external graph data are available in several applications of interest, such as classifying users connected via a social network, items and customers connected by purchase history, users and movies connected by viewing history, and entities in a knowledge graph connected by relationships.

In this paper, we are interested in the setting where the graph is explicitly given and represents additional information not present in the feature vectors.

The goal of such graph-based semi-supervised learning problems is to classify the nodes in a graph using a small subset of labeled nodes and all the node features.

There is a long line of literature on this topic since BID6 which seeks graph cuts that preserve the known labels and BID50 which uses graph Laplacian to regularize the nearby nodes to have similar labels.

However, BID27 recently demonstrated that the existing approaches can be significantly improved upon on a number of standard benchmark datasets, using an innovative neural network architecture on graph-based data known collectively as graph neural networks.

Inspired by this success, we seek to understand the reason behind the power of graph neural networks, to guide our design of a novel architecture for semi-supervised learning on graphs.

To this end, we first found that a linear classifier of multinomial logistic regression achieves the accuracy comparable to the best known graph neural network.

This linear classifier removes all intermediate non-linear activation layers, and only keeps the linear propagation function from neighbors in graph neural networks.

This suggests the importance of aggregation information form the neighbors in the graph.

This further motivates us to design a new way of aggregating neighborhood information through attention mechanism since, intuitively, neighbors might not be equally important.

This proposed attention-based graph neural network captures this intuition and (a) greatly reduces the model complexity, with only a single scalar parameter at each intermediate layer; (b) discovers dynamically and adaptively which nodes are relevant to the target node for classification; and (c) improves upon state-of-the-art methods in terms of accuracy on standard benchmark datasets.

Further, the learned attention strengths provide some form of interpretability.

They provide insights on why a particular prediction is made on a target node and which neighbors are more relevant in making that decision.

Given a graph G(V, E) with a set of n nodes V and a set of edges E, we let X i ??? R dx denote the feature vector at node i and let Y i denote the true label.

We use Y L to denote the labels that are revealed to us for a subset L ??? V .

We let X = [X 1 , . . .

, X n ] denote all features, labeled and unlabeled.

Traditionally, semi-supervised learning using both labeled and un-labled data has been solved using two different approaches -Graph Laplacian based algorithms solving for locally consistent solutions BID48 and Expectation Maximization based algorithms BID33 where true-labels of the unlabeled data points are considered as the latent variables of a generative model.

Based on the assumption that nearby nodes in a graph are more likely to have the same labels, the graph information has been used as explicit regularization: DISPLAYFORM0 where L label = i???L l(Y i , f (X i )) is the standard supervised loss for some loss functions l and L G is the graph-based regularization, for example DISPLAYFORM1 2 , which is called the graph Laplacian regularization.

Earlier approaches are non-parametric and searches over all f considering it as a look-up table.

Most popular one is the Label Propagation BID49 that forces the estimated labels to agree in the labeled instances and uses weighted graph Laplacian.

This innovative formulation admits a closed form solution which makes it practically attractive with very low computationally cost.

ManiReg BID4 replaces supervised loss with that of a support vector machine.

ICA BID30 generalizes LP by allowing more general local updates.

A more thorough survey on using non-neural network methods for semi-supervised learning can be found in (Chapelle et al., 2009 ).More recent approaches are parametric, using deep neural networks.

SemiEmb BID42 was the first to use a deep neural network to model f (x) and minimize the above loss.

Planetoid BID46 significantly improves upon the existing graph regularization approaches by replacing the regularization by another loss based on skip-grams (defined below).

In a slightly different context, BID10 show that the accuracy of these approaches can be further improved by bootstrapping these models sequentially.

Unsupervised node embedding for semi-supervised learning.

Several approaches have been proposed to embed the nodes in some latent Euclidean space using only the connectivity in graph G. Once the embedding is learned, standard supervised learning is applied on those embedded features to train a model.

Inspired by the success of word2vec BID28 , several approaches define "skip-grams" on graphs as the neighborhood (context) of a node on the graph and tries to maximize the posterior probability of observing those skip-grams.

DeepWalk BID35 and node2vec BID22 use random walks as skip-grams, LINE BID40 uses local proximities, LASAGNE BID18 uses the Personalized PageRank random walk.

Graph2Gauss (A. BID0 represents a node as a Gaussian distribution, and minimizes the divergence between connected pairs.

BID45 provide a post-processing scheme that takes any node embedding and attempts to improve it by by taking the weighted sum of the given embeddings with Personalized PageRank weights.

The strength of these approaches is universality, as the node embedding does not depend on the particular task at hand (and in particular the features or the labels).

However, as they do not use the node features and the training only happens after embedding, they cannot meet the performance of the state-of-the-art approaches (see DeepWalk in Table 2 ).Graph Neural Network (GNN).

Graph neural networks are extensions of neural networks to structured data encoded as a graph.

Originally introduced as extensions of recurrent neural networks, GNNs apply recurrent layers to each node with additional local averaging layer BID20 BID36 .

However, as the weights are shared across all nodes, GNNs can also be interpreted as extensions of convolutional neural networks on a 2D grid to general graphs.

Typically, a message aggregation step followed by some neural network architecture is iteratively applied.

The model parameters are trained on (semi-)supervised examples with labels.

We give a typical example of a GNN in Section 3, but several diverse variations have been proposed in BID9 BID17 BID29 BID24 Sukhbaatar et al., 2016; BID15 BID13 BID2 BID32 BID23 BID37 .

GNNs have been successfully applied in diverse applications such as molecular activation prediction BID19 , community detection BID8 , matrix completion , combinatorial optimization BID14 BID34 , and detecting similar binary codes BID44 .In particular, for the benchmark datasets that we consider in this paper, BID27 proposed a simple but powerful architecture called Graph Convolutional Network (GCN) that achieves the state-of-the-art accuracy.

In the following section, (a) we show that the performance of GCN can be met by a linear classifier; and (b) use this insight to introduce novel graph neural networks that compare favourably against the state-of-the-art approaches on benchmark datasets.

In this section, we propose a novel Graph Neural Network (GNN) model which we call Attentionbased Graph Neural Network (AGNN), and compare its performance to state-of-the-art models on benchmark citation networks in Section 5.

We seek a model Z = f (X, A) ??? R n??dy that predicts at each node one of the d y classes.

Z ic is the estimated probability that the label at node i ??? [n] is c ??? [d y ] given the features X and the graph A. The data features X ??? R n??dx has at each row d x features for each node, and A ??? {0, 1} n??n is the adjacency matrix of G.The forward pass in a typical GNN alternates between a propagation layer and a single layer perceptron.

Let t be the layer index.

We use H (t) ??? R n??d h to denote the current (hidden) states, with the i-th row H (t) i as the d h dimensional hidden state of node i. A propagation layer with respect to a propagation matrix P ??? R n??n is defined as DISPLAYFORM0 For example, the natural random walk DISPLAYFORM1 j .

The neighborhood of node i is denoted by N (i), and D = diag(A1).

This is a simple local averaging common in consensus or random walk based approaches.

Typical propagation layer respects the adjacency pattern in A, performing a variation of such local averaging.

GNNs encode the graph structure of A into the model via this propagation layer, which can be also interpreted as performing a graph convolution operation as discussed in BID27 .

Next, a single layer perceptron is applied on each node separately and the weights W (t) are shared across all the nodes: DISPLAYFORM2 where W (t) ??? R d h t+1 ??d h t is the weight matrix and ??(??) is an entry-wise activation function.

This weight sharing reduces significantly the number of parameters to be trained, and encodes the invariance property of graph data, i.e. two nodes that are far apart but have the similar neighboring features and structures should be classified similarly.

There are several extensions to this model as discussed in the previous section, but this standard graph neural network has proved powerful in several problems over graphs, e.g. BID8 BID14 .Graph Convolutional Network (GCN).

BID27 introduced a simple but powerful architecture, and achieved the state-of-the-art performance in benchmark citation networks (see Table 2 ).

GCN is a special case of GNN which stacks two layers of specific propagation and perceptron: DISPLAYFORM3 with a choice of P =D ???1/2??D???1/2 , where?? = A+I, I is the identity matrix,D = diag(??1) and 1 is the all-ones vector.

ReLU(a) = max{0, a} is an entry-wise rectified linear activation function, and DISPLAYFORM4 is applied rowwise.

Hence, the output is the predicted likelihoods on the d y dimensional probability simplex.

The weights W (0) and W (1) are trained to minimize the cross-entropy loss over all labeled examples L: DISPLAYFORM5 Graph Linear Network (GLN).

To better understand GCN, we remove the intermediate nonlinear activation units from GCN, which gives Graph Linear Network defined as DISPLAYFORM6 with the same choice of P =D ???1/2??D???1/2 as in GCN.

The weights W (0) and W (1) have the same dimensions as GCN and are trained on a cross entropy loss in (2).

The two propagation layers simply take (linear) local average of the raw features weighted by their degrees, and at the output layer a simple linear classifier (multinomial logistic regression) is applied.

This allows us to separate the gain in the linear propagation layer and the non-linear perceptron layer.

Table 2 , we show that, perhaps surprisingly, GLN achieves an accuracy comparable to the that of the best GNN, and sometimes better.

This suggests that, for citation networks, the strength of the general GNN architectures is in the propagation layer and not in the perceptron layer.

On the other hand, the propagation layers are critical in achieving the desired performance, as is suggested in Table 2 .

There are significant gaps in accuracy for those approaches not using the graph, i.e. T-SVM, and also those that use the graph differently, such as Label Propagation (LP) and Planetoid.

Based on this observation, we propose replacing the propagation layer of GLN with an attention mechanism and test it on the benchmark datasets.

The original propagation layer in GCN and several other graph neural networks such as BID15 BID2 BID31 ) use a static (does not change over the layers) and non-adaptive (does not take into account the states of the nodes) propagation, e.g. P ij = 1/ |N (i)| |N (j)|.

Such propagations are not able to capture which neighbor is more relevant to classifying a target node, which is critical in real data where not all edges imply the same types or strengths of relations.

We need novel dynamic and adaptive propagation layers, capable of capturing the relevance of different edges, which leads to more complex graph neural networks with more parameters.

However, training such complex models is challenging in the semi-supervised setting, as the typical number of samples we have for each class is small; it is 20 in the standard benchmark dataset.

This is evidenced in Table 2 where more complex graph neural network models by BID41 , BID31 , and do not improve upon the simple GCN.On the other hand, our experiments with GLN suggests that we can remove all the perceptron layers and focus only on improving the propagation layers.

To this end, we introduce a novel Attentionbased Graph Neural Network (AGNN).

AGNN is simple; it only has a single scalar parameter ?? (t) at each intermediate layer.

AGNN captures relevance; the proposed attention mechanism over neighbors in (5) learns which neighbors are more relevant and weighs their contributions accordingly.

This builds on the long line of successes of attention mechanisms in summarizing long sentences or large images, by capturing which word or part-of-image is most relevant BID43 BID21 BID3 .

Particularly, we use the attention formulation similar to the one used in BID21 .

2 It only has one parameter and we found this is important for successfully training the model when the number of labels is small as in our semi-supervised learning setting.

We start with a word-embedding layer that maps a bag-of-words representation of a document into an averaged word embedding, and the word embedding W (0) ??? R dx??d h is to be trained as a part of the model: DISPLAYFORM0 This is followed by layers of attention-guided propagation layers parameterized by ?? (t) ??? R at each layer, DISPLAYFORM1 where the propagation matrix P (t) ??? R n??n is also a function of the input states H (t) and is zero for absent edges such that the output row-vector of node i is DISPLAYFORM2 and cos(x, y) = x T y/ x y with the L 2 norm x , for t ??? {1, . . . , } and an integer .

Here is the number of propagation layers.

Note that the new propagation above is dynamic; propagation changes over the layers with differing ?? (t) and also the hidden states.

It is also adaptive; it learns to weight more relevant neighbors higher.

We add the self-loop in the propagation to ensure that the features and the hidden states of the node itself are not lost in the propagation process.

The output layer has a weight DISPLAYFORM3 The weights W (0) , W (1) , and ?? (t) 's are trained on a cross entropy loss in (2).

To ease the notations, we have assumed that the input feature vectors to the first and last layers are augmented with a scalar constant of one, so that the standard bias term can be included in the parameters W (0) and W (1) .The softmax function at attention ensures that the propagation layer P (t) row-sums to one.

The attention from node j to node i is DISPLAYFORM4 DISPLAYFORM5 j ) which captures how relevant j is to i, as measured by the cosine of the angle between the corresponding hidden states.

We show how we can interpret the attentions in Section 5.2 and show that the attention selects neighbors with the same class to be more relevant.

On the standard benchmark datasets on citation networks, we show in Section 5 that this architecture achieves the best performance in Table 2 .Here we note that independently from this work attention over sets has been proposed as "neighborhood attention" BID16 BID25 for a different application.

The main difference of AGNN with respect to these work is the fact that in AGNN attention is computed over a neighborhood of a node on a graph, whereas in these work attention over set of all entities is used to construct a "soft neighborhood".

On standard benchmark datasets of three citation networks, we test our proposed AGNN model on semi-supervised learning tasks.

We test on a fixed split of labeled/validation/test sets from BID46 and compare against baseline methods in Table 2 .

We also test it on random splits of the same sizes in TAB5 , and random splits with larger number of labeled nodes in Table 4 .Benchmark Datasets.

A citation network dataset consists of documents as nodes and citation links as directed edges.

Each node has a human annotated topic from a finite set of classes and a feature vector.

We consider three datasets 3 .

For CiteSeer and Cora datasets, the feature vector has binary entries indicating the presence/absence of the corresponding word from a dictionary.

For PubMed dataset, the feature vector has real-values entries indicating Term Frequency-Inverse Document Frequency (TF-IDF) of the corresponding word from a dictionary.

Although the networks are directed, we use undirected versions of the graphs for all experiments, as is common in all baseline approaches.

Nodes Edges Classes Features TAB5 We train and test only the two models we propose: GLN for comparisons and our proposed AGNN model.

We do not use the validation set labels in training, but use them for optimizing hyperparameters like dropout rate, learning rate, and L 2 -regularization factor.

For AGNN, we use a fixed number of d h = 16 units in the hidden layers and use 4 propagation layers ( = 4) for CiteSeer and Pubmed and 3 propagation layers ( = 3) for Cora as defined in (7).

For GLN, we use 2 propagation layers as defined in (1).

We row-normalize the input feature vectors, as is standard in the literature.

The tables below show the average accuracy with the standard error over 100 training instances with random weight initializations.

We implement our model on TensorFlow BID1 , and the computational complexity of evaluating AGNN is DISPLAYFORM0 Detailed desription of the experiments is provided in Appendix B.

Fixed data splits.

In this first experiment, we use the fixed data splits from BID46 as they are the standard benchmark data splits in literature.

All experiments are run on the same fixed split of 20 labeled nodes for each class, 500 nodes for validation, 1,000 nodes for test, and the rest of nodes as unlabeled data.

Perhaps surprisingly, the linear classifier GLN we proposed in (3) achieves performance comparable to or exceeding the state-of-the-art performance of GCN.

This leads to our novel attention-based model AGNN defined in (6), which achieves the best accuracy on all datasets with a gap larger than the standard error.

The classification accuracy of all the baseline methods are collected from BID46 BID27 BID31 BID10 BID41 .In semi-supervised learning on graphs, it is critical to utilize both the structure of the graph and the node features.

Methods not using all the given data achieve performance far from the stateof-the-art.

BID30 69.1 75.1 73.9 ManiReg BID4 60.1 59.5 70.7 SemiEmb BID42 59.6 59.0 71.1 DCNN BID2 76.8 73.0 Planetoid BID46 64.7 75.7 77.2 MoNet BID31 81.7 78.8 Graph-CNN 76.3 DynamicFilter BID41 81.6 79.0 Bootstrap BID10 53.6 78.4 78.8 GCN BID27 70 Table 2 : Classification accuracy with a fixed split of data from BID46 .A breakthrough result of Planetoid by BID46 significantly improved upon the existing skip-gram based method of DeepWalk and node2vec and the Laplacian regularized methods of ManiReg and SemiEmb.

BID27 was the first to apply a graph neural network to citation datasets, and achieved the state-of-the-art performance with GCN.

Other variations of graph neural networks immediately followed, achieving comparable performance with MoNet, Graph-CNN, and DynamicFilter.

Bootstrap uses a Laplacian regularized approach of BID47 as a sub-routine with bootstrapping to feed high-margin predictions as seeds.

Random splits.

Next, following the setting of BID10 , we run experiments keeping the same size in labeled, validation, and test sets as in Table 2 , but now selecting those nodes uniformly at random.

This, along with the fact that different topics have different number of nodes in it, means that the labels might not be spread evenly across the topics.

For 20 such randomly drawn dataset splits, average accuracy is shown in TAB5 with the standard error.

As we do not force equal number of labeled data for each class, we observe that the performance degrades for all methods compared to Table 2 , except for DeepWalk.

AGNN achieves the best performance consistently.

Here, we note that BID27 does a similar but different experiment using GCN, where random labeled nodes are evenly spread across topics so that each topic has exactly 20 labeled examples.

As this difference in sampling might affect the accuracy, we do not report those results in this table.

Method CiteSeer Cora PubMed DeepWalk BID35 47.2 70.2 72.0 node2vec BID22 Larger training set.

Following the setting of , we run experiments with larger number of labeled data on Cora dataset.

We perform k-fold cross validation experiments for k = 3 and 10, by uniformly and randomly dividing the nodes into k equal sized partitions and then performing k runs of training by masking the labels of each of the k partitions followed by validation on the masked nodes.

Finally the average validation accuracy across k runs is reported.

We run 10 trials of this experiment and reports the mean and standard error of the average k-fold validation accuracy.

Compared to Table 2 , the performance increases with the size of the training set, and AGNN consistently outperforms the current state-of-the-art architecture for this experiment.

Method 3-fold Split 10-fold split Graph-CNN Table 4 : Classification accuracy with larger sets of labelled nodes.

One useful aspect of incorporating attention into a model is that it provides some form of interpretation capability BID3 .

The learned P (t) ij 's in Eq. FORMULA13 represent the attention from node j to node i, and provide insights on how relevant node j is in classifying node i. In FIG0 , we provide statistics of this attention over all adjacent pairs of nodes for Cora and CiteSeer datasets.

We refer to Figure 3 for similar statistics on PubMed.

In FIG0 , we show average attention from a node in topic c 2 (column) to a node in topic c 1 (row), which we call the relevance from c 2 to c 1 and is defined as DISPLAYFORM0 for edge-wise relevance score defined as DISPLAYFORM1 where |N (i)| is the degree of node i, and S c1,c2 = {(i, j) ??? E s and Y i = c 1 , Y j = c 2 } where E s = E ??? {(i, i) for i ??? V } is the edge set augmented with self-loops to include all the attentions learned.

If we are not using any attention, then the typical propagation will be uniform P ij = 1/(|N (i)| + 1), in which case the above normalized attention is zero.

We are measuring for each edge the variation of attention P ij from uniform 1/(|N (i)| + 1) as a multiplicative error, normalized by 1/(|N (i)| + 1).

We believe this is the right normalization, as attention should be measure in relative strength to others in the same neighborhood, and not in absolute additive differences.

We are measuring this multiplicative variation of the attention, averaged over the ordered pairs of classes.

FIG0 shows the relevance score for CiteSeer and Cora datasets. (PubMed is shown in Appendix A.) For both datasets, the diagonal entries are dominant indicating that the attention is learning to put more weight to those in the same class.

A higher value of Relevance(c 2 ??? c 1 ) indicates that, on average, a node in topic c 1 pays more attention to a neighbor in topic c 2 than neighbors from other topics.

For CiteSeer dataset FIG0 , we are showing the average attention at the first propagation layer, P (t=1) , for illustration.

In the off-diagonals, the most influential relations are HCI???Agents, Agents???ML, Agents???HCI, and ML???Agents, and the least influential relations are AI???IR and DB???ML.

Note that these are papers in computer science from late 90s to early 2000s.

For Cora dataset FIG0 , we are showing the relevance score of the second propagation layer, P (t=2) , for illustration.

In the off-diagonals, the most influential relations are CB???PM, PM???CB, Rule???PM, and PM???Rule, and the least influential relations are GA???PM and PM???RL.

This dataset has papers in computer science from the 90s.

We note that these relations are estimated solely based on the available datasets for that period of time and might not accurately reflect the relations for the entire academic fields.

We also consider these relations as a static property in this analysis.

If we have a larger corpus over longer period of time, it is possible to learn the influence conditioned on the period and visualize how these relations change.

Next, we analyze the edges with high and low relevance scores.

We remove the self-loops and then sort the edges according to the relevance score defined in Eq. (9).

We take the top 100 and bottom 100 edges and with respect to their relevance scores, and report the fraction of the edges which are connecting nodes from the same class.

TAB7 shows the result on the benchmark datasets for the relevance scores calculated using the last propagation layer.

This suggests that our architecture learns to put higher attention between nodes of the same class.

Finally, we analyze those nodes in the test sets that were mistaken by GCN but correctly classified by AGNN, and show how our attention mechanism weighted the contribution of its local neighborhood, and show three illustrative examples in Figure 2 .

More examples of this local attention network (including the legends for the color coding of the topics) are provided in Appendix A. We show a entire 2-hop neighborhood of a target node (marked by a thick outline) from the test set of the fixed data splits of Citeseer, Cora, or Pubmed.

The colors denote the true classes of the nodes (including the target) in the target's neighborhood, some of which are unknown to the models at the training time.

The radius of a node j is proportional to the attention to the target node i aggregated over all the layers, i.e. (P (t=4) P (t=3) P (t=2) P (t=1) ) ij for CiteSeer.

The size of the target node reflects its self-attention defined in a similar way.

The first example on the left is node 8434 from PubMed.

AGNN correctly classifies the target node as light blue, whereas GCN mistakes it for yellow, possibly because it is connected to more yellow nodes.

Not only has the attention mechanism learned to put more weight to its light blue 1-hop neighbor, but put equally heavy weights to a path of light blue neighbors some of which are not immediately connected to the target node.

The second example in the middle is node 1580 from PubMed.

AGNN correctly classifies it as yellow, whereas GCN mistakes it for a red, possibly because it only has two neighbors.

Not only has the attention mechanism learned to put more weight to the yellow neighbor, but it has weighted the yellow neighbor (who is connected to many yellow nodes and perhaps has more reliable hidden states representing the true yellow class) even more than itself.

The last example on the right is node 1512 from CiteSeer.

AGNN correctly classifies it as light blue, whereas GCN mistakes it for a white.

This is a special example as those two nodes are completely isolated.

Due to the static and non-adaptive propagation of GCN, it ends up giving the same prediction for such isolated pairs.

If the pair has two different true classes, then it always fails on at least on one of them (in this case the light blue node).

However, AGNN is more flexible in adapting to such graph topology and puts more weight to the target node itself, correctly classifying both.

In this paper, we present an attention-based graph neural network model for semi-supervised classification on a graph.

We demonstrate that our method consistently outperforms competing methods on the standard benchmark citation network datasets.

We also show that the learned attention also provides interesting insights on how neighbors influence each other.

In training, we have tried more

PubMed 1580 CiteSeer 1512Figure 2: We show three selected target nodes in the test set that are mistaken by GCN but correctly classified by AGNN.

We denote this target node by the node with a thick outline (node 8434 from PubMed on the left, node 1580 from PubMed in the middle, and node 1512 from CiteSeer on the right).

We show the strength of attention from a node in the 2-hop neighborhood to the target node by the size of the corresponding node.

Colors represent the hidden true classes (nodes with the same color belong to the same topic).

None of the nodes in the figure was in the training set, hence none of the colors were revealed.

Still, we observe that AGNN has managed to put more attention to those nodes in the same (hidden) classes, allowing the trained model to find the correct labels.complex attention models.

However, due to the increased model complexity the training was not stable and does not give higher accuracy.

We believe that for semi-supervised setting with such a limited number of labeled examples, reducing model complexity is important.

Note that we are able to train deeper (4-layers) models compared to a shallower (2-layers) model of GCN, in part due to the fact that we remove the non-linear layers and reduce the model complexity significantly.

In comparison, deeper GCN models are known to be unstable and do not give the performance of shallower GCNs BID27 .

PubMed dataset has only 3 classes, and the relevance score is shown in the figure below.

In this section we will list all the choices made in training and tuning of hyper-parameters.

The parameters are chosen as to maximize the validation.

All the models use Adam optimization algorithm with full-batchs, as standard in other works on GNNs BID27 .

We also a weight decay term to the objective function for all the learnable weights.

We add dropout to the first and last layers of all models.

In TAB9 .

we show the hyper-parameters used in training the AGNN models for various settings and datasets.

For Cora dataset the architecture consist of = 3 propagation layers, but the first prop- agation layer P (t=1) is non-trainable and the variable ?? (t=1) value is fixed at zero.

While training these AGNN models we maintain the validation accuracy for each iteration and finally choose the trained model parameters from the iteration where average validation accuracy of previous 4 epochs is maximized.

For the k-fold cross validation setting we take the epoch with maximum validation accuracy.

FORMULA8 , we use the same hyper-parameters as GCN BID27 for all the experimental settings: hidden dimension of 16, learning rate of 0.01, weight decay of 5??10 ???4 , dropout of 0.5, 200 epochs and early stopping criteria with a window size of 10.

In this section we provide experimental results justifying the choice of number of propagation layers for each dataset.

We use the same data split and hyper-parameters (except number of propagation layer) as in the fixed data splits setting in Section 5.1.

Similar to other settings the first propagation layer for Cora dataset is non-trainable with ?? (t=1) = 0.

Tables 7 gives the average (over 10 trials) testing accuracy respectively for various choices of number of propagation layers.

We note that different datasets require different number of propagation layers for best performance.

BID35 47.2 70.2 72.0 node2vec BID22 Here we provide the performance of GCN on random splits and larger training set dataset splits from Section 5.1.

We relegate these tables to the appendix since they were not present in BID27 .

We conducted the experiments with the same hyper-parameters as chosen by BID27 for the fixed split.

In TAB13 we provide average testing accuracy and standard error over 20 and 10 runs on random splits and larger training set respectively.

Method 3-fold Split 10-fold split Graph-CNN Table 9 : Classification accuracy with larger sets of labelled nodes.

<|TLDR|>

@highlight

We propose a novel attention-based interpretable Graph Neural Network architecture which outperforms the current state-of-the-art Graph Neural Networks in standard benchmark datasets

@highlight

The authors propose two extensions of GCNs, by removing intermediate non-linearities from the GCN computation and adding an attention mechanism in the aggregation layer.

@highlight

The paper proposes a semi supervised learning algorithm for graph node classification with is inspired from Graph Neural Networks.
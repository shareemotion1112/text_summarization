Predicting properties of nodes in a graph is an important problem with applications in a variety of domains.

Graph-based Semi Supervised Learning (SSL) methods aim to address this problem by labeling a small subset of the nodes as seeds, and then utilizing the graph structure to predict label scores for the rest of the nodes in the graph.

Recently, Graph Convolutional Networks (GCNs) have achieved impressive performance on the graph-based SSL task.

In addition to label scores, it is also desirable to have a confidence score associated with them.

Unfortunately, confidence estimation in the context of GCN has not been previously explored.

We fill this important gap in this paper and propose ConfGCN, which estimates labels scores along with their confidences jointly in GCN-based setting.

ConfGCN uses these estimated confidences to determine the influence of one node on another during neighborhood aggregation, thereby acquiring anisotropic capabilities.

Through extensive analysis and experiments on standard benchmarks, we find that ConfGCN is able to significantly outperform state-of-the-art baselines.

We have made ConfGCN’s source code available to encourage reproducible research.

Graphs are all around us, ranging from citation and social networks to knowledge graphs.

Predicting properties of nodes in such graphs is often desirable.

For example, given a citation network, we may want to predict the research area of an author.

Making such predictions, especially in the semisupervised setting, has been the focus of graph-based semi-supervised learning (SSL) BID25 .

In graph-based SSL, a small set of nodes are initially labeled.

Starting with such supervision and while utilizing the rest of the graph structure, the initially unlabeled nodes are labeled.

Conventionally, the graph structure has been incorporated as an explicit regularizer which enforces a smoothness constraint on the labels estimated on nodes BID36 BID2 BID31 .

Recently proposed Graph Convolutional Networks (GCN) BID8 BID14 provide a framework to apply deep neural networks to graphstructured data.

GCNs have been employed successfully for improving performance on tasks such as semantic role labeling , machine translation BID1 , relation extraction BID28 BID35 , event extraction BID20 , shape segmentation BID34 , and action recognition BID12 .

GCN formulations for graph-based SSL have also attained state-of-the-art performance BID14 BID18 BID29 .

In this paper, we also focus on the task of graphbased SSL using GCNs.

GCN iteratively estimates embedding of nodes in the graph by aggregating embeddings of neighborhood nodes, while backpropagating errors from a target loss function.

Finally, the learned node embeddings are used to estimate label scores on the nodes.

In addition to the label scores, it is desirable to also have confidence estimates associated with them.

Such confidence scores may be used to determine how much to trust the label scores estimated on a given node.

While methods to estimate label score confidence in non-deep graph-based SSL has been previously proposed BID21 , confidence-based GCN is still unexplored.

Figure 1: Label prediction on node a by Kipf-GCN and ConfGCN (this paper).

L 0 is a's true label.

Shade intensity of a node reflects the estimated score of label L 1 assigned to that node.

Since Kipf-GCN is not capable of estimating influence of one node on another, it is misled by the dominant label L 1 in node a's neighborhood and thereby making the wrong assignment.

ConfGCN, on the other hand, estimates confidences (shown by bars) over the label scores, and uses them to increase influence of nodes b and c to estimate the right label on a. Please see Section 1 for details.

In order to fill this important gap, we propose ConfGCN, a GCN framework for graph-based SSL.

ConfGCN jointly estimates label scores on nodes, along with confidences over them.

One of the added benefits of confidence over node's label scores is that they may be used to subdue irrelevant nodes in a node's neighborhood, thereby controlling the number of effective neighbors for each node.

In other words, this enables anisotropic behavior in GCNs.

Let us explain this through the example shown in Figure 1 .

In this figure, while node a has true label L 0 (white), it is incorrectly classified as L 1 (black) by Kipf-GCN (Kipf & Welling, 2016) 2 .

This is because Kipf-GCN suffers from limitations of its neighborhood aggregation scheme BID32 .

For example, Kipf-GCN has no constraints on the number of nodes that can influence the representation of a given target node.

In a k-layer Kipf-GCN model, each node is influenced by all the nodes in its k-hop neighborhood.

However, in real world graphs, nodes are often present in heterogeneous neighborhoods, i.e., a node is often surrounded by nodes of other labels.

For example, in Figure 1 , node a is surrounded by three nodes (d, e, and f ) which are predominantly labeled L 1 , while two nodes (b and c) are labeled L 0 .

Please note that all of these are estimated label scores during GCN learning.

In this case, it is desirable that node a is more influenced by nodes b and c than the other three nodes.

However, since Kipf-GCN doesn't discriminate among the neighboring nodes, it is swayed by the majority and thereby estimating the wrong label L 1 for node a.

ConfGCN is able to overcome this problem by estimating confidences on each node's label scores.

In Figure 1 , such estimated confidences are shown by bars, with white and black bars denoting confidences in scores of labels L 0 and L 1 , respectively.

ConfGCN uses these label confidences to subdue nodes d, e, f since they have low confidence for their label L 1 (shorter black bars), whereas nodes b and c are highly confident about their labels being L 0 (taller white bars).

This leads to higher influence of b and c during aggregation, and thereby ConfGCN correctly predicting the true label of node a as L 0 with high confidence.

This clearly demonstrates the benefit of label confidences and their utility in estimating node influences.

Graph Attention Networks (GAT) BID29 , a recently proposed method also provides a mechanism to estimate influences by allowing nodes to attend to their neighborhood.

However, as we shall see in Section 6, ConfGCN, through its use of label confidences, is significantly more effective.

Our contributions in this paper are as follows.• We propose ConfGCN, a Graph Convolutional Network (GCN) framework for semisupervised learning which models label distribution and their confidences for each node in the graph.

To the best of our knowledge, this is the first confidence-enabled formulation of GCNs.• ConfGCN utilize label confidences to estimate influence of one node on another in a labelspecific manner during neighborhood aggregation of GCN learning.• Through extensive evaluation on multiple real-world datasets, we demonstrate ConfGCN effectiveness over state-of-the-art baselines.

ConfGCN's source code and datasets used in the paper are made publicly available 3 to foster reproducible research.

Semi-Supervised learning (SSL) on graphs: SSL on graphs is the problem of classifying nodes in a graph, where labels are available only for a small fraction of nodes.

Conventionally, the graph structure is imposed by adding an explicit graph-based regularization term in the loss function BID36 BID31 BID2 .

Recently, implicit graph regularization via learned node representation has proven to be more effective.

This can be done either sequentially or in an end to end fashion.

Methods like DeepWalk BID22 , node2vec BID11 , and LINE BID27 first learn graph representations via sampled random walk on the graph or breadth first search traversal and then use the learned representation for node classification.

On the contrary, Planetoid BID33 learns node embedding by jointly predicting the class labels and the neighborhood context in the graph.

Recently, BID14 employs Graph Convolutional Networks (GCNs) to learn node representations.

The generalization of Convolutional Neural Networks to non-euclidean domains is proposed by BID6 which formulates the spectral and spatial construction of GCNs.

This is later improved through an efficient localized filter approximation BID8 .

BID14 provide a first-order formulation of GCNs and show its effectiveness for SSL on graphs.

propose GCNs for directed graphs and provide a mechanism for edge-wise gating to discard noisy edges during aggregation.

This is further improved by BID29 which allows nodes to attend to their neighboring nodes, implicitly providing different weights to different nodes.

BID18 propose Graph Partition Neural Network (GPNN), an extension of GCNs to learn node representations on large graphs.

GPNN first partitions the graph into subgraphs and then alternates between locally and globally propagating information across subgraphs.

An extensive survey of GCNs and their applications can be found in BID5 .

The natural idea of incorporating confidence in predictions has been explored by BID16 for the task of active learning.

BID15 proposes a confidence based framework for classification problems, where the classifier consists of two regions in the predictor space, one for confident classifications and other for ambiguous ones.

In representation learning, uncertainty (inverse of confidence) is first utilized for word embeddings by BID30 .

BID0 further extend this idea to learn hierarchical word representation through encapsulation of probability distributions.

BID21 propose TACO (Transduction Algorithm with COnfidence), the first graph based method which learns label distribution along with its uncertainty for semi-supervised node classification.

BID3 embeds graph nodes as Gaussian distribution using ranking based framework which allows to capture uncertainty of representation.

They update node embeddings to maintain neighborhood ordering, i.e. 1-hop neighbors are more similar to 2-hop neighbors and so on.

Gaussian embeddings have been used for collaborative filtering (Dos BID9 and topic modelling BID7 as well.

Let G = (V, E, X ) be an undirected graph, where V = V l ∪ V u is the union of labeled (V l ) and unlabeled (V u ) nodes in the graph with cardinalities n l and n u , E is the set of edges and X ∈ R (n l +nu)×d is the input node features.

The actual label of a node v is denoted by a one-hot vector Y v ∈ R m , where m is the number of classes.

Given G and seed labels Y ∈ R n l ×m , the goal is to predict the labels of the unlabeled nodes.

To incorporate confidence, we additionally estimate label distribution µ v ∈ R m and a diagonal co-variance matrix Σ v ∈ R m×m , ∀v ∈ V. Here, µ v,i denotes the score of label i on node v, while (Σ v ) ii denotes the variance in the estimation of µ v,i .

In other words, (Σ −1 v ) ii is ConfGCN's confidence in µ v,i .

In this section, we give a brief overview of Graph Convolutional Networks (GCNs) for undirected graphs as proposed by BID14 .

Given a graph G = (V, E, X ) as defined Section 3, the node representation after a single layer of GCN can be defined as DISPLAYFORM0 where, W ∈ R d×d denotes the model parameters, A is the adjacency matrix andD ii = j (A + I) ij .

f is any activation function, we have used ReLU, f (x) = max(0, x) in this paper.

Equation 1 can also be written as DISPLAYFORM1 Here, b ∈ R d denotes bias, N (v) = {u : {u, v} ∈ E} corresponds to immediate neighbors of v in graph G and h v is the obtained representation of node v.

For capturing multi-hop dependencies between nodes, multiple GCN layers can be stacked on top of one another.

The representation of node v after k th layer of GCN is given as DISPLAYFORM2 where, W k , b k denote the layer specific parameters of GCN.

Following BID21 , ConfGCN uses co-variance matrix based symmetric Mahalanobis distance for defining distance between two nodes in the graph.

Formally, for any two given nodes u and v, with label distributions µ u and µ v and co-variance matrices Σ u and Σ v , distance between them is defined as follows.

DISPLAYFORM0 Characteristic of the above distance metric is that if either of Σ u or Σ v has large eigenvalues, then the distance will be low irrespective of the closeness of µ u and µ v .

On the other hand, if Σ u and Σ v both have low eigenvalues, then it requires µ u and µ v to be close for their distance to be low.

Given the above properties, we define r uv , the influence score of node u on its neighboring node v during GCN aggregation, as follows.

DISPLAYFORM1 .This influence score gives more relevance to neighboring nodes with highly confident similar label, while reducing importance of nodes with low confident label scores.

This results in ConfGCN acquiring anisotropic capability during neighborhood aggregation.

For a node v, ConfGCN's equation for updating embedding at the k-th layer is thus defined as follows.

DISPLAYFORM2 The final node representation obtained from ConfGCN is used for predicting labels of the nodes in the graph as follows.

Ŷ DISPLAYFORM3 where, K denotes the number of ConfGCN's layers.

Finally, in order to learn label scores {µ v } and co-variance matrices {Σ v } jointly with other parameters {W k , b k }, following Orbach & Crammer (2012), we include the following three terms in ConfGCN's objective function.

For enforcing neighboring nodes to be close to each other, we include L smooth defined as DISPLAYFORM4 To impose the desirable property that the label distribution of nodes in V l should be close to their input label distribution, we incorporate L label defined as DISPLAYFORM5 Here, for input labels, we assume a fixed uncertainty of 1 γ I ∈ R L×L , where γ > 0.

We also include the following regularization term, L reg to constraint the co-variance matrix to be finite and positive.

DISPLAYFORM6 for some η > 0.

The first term increases monotonically with the eigenvalues of Σ and the second term prevents them from becoming zero.

Additionally in ConfGCN, we include the L const in the objective, to push the label distribution (µ) close to the final model prediction (Ŷ ).

DISPLAYFORM7 Finally, we include the standard cross-entropy loss for semi-supervised multi-class classification over all the labeled nodes (V l ).

DISPLAYFORM8 The final objective for optimization is the linear combination of the above defined terms.

DISPLAYFORM9 where, λ i ∈ R, are the weights of the terms in the objective.

We optimize L({W k , b k , µ v , Σ v }) using stochastic gradient descent.

We hypothesize that all the terms help in improving ConfGCN's performance and we validate this in Section 7.4.

For evaluating the effectiveness of ConfGCN, we evaluate on several semi-supervised classification benchmarks.

Following the experimental setup of BID14 BID18 , we evaluate on Cora, Citeseer, and Pubmed datasets BID23 .

The dataset statistics is summarized in Table 1 .

Label mismatch denotes the fraction of edges between nodes with different labels in the training data.

The benchmark datasets commonly used for semi-supervised classification task have substantially low label mismatch rate.

In order to examine models on datasets with more heterogeneous neighborhoods, we also evaluate on Cora-ML dataset BID4 .All the four datasets are citation networks, where each document is represented using bag-of-words features in the graph with undirected citation links between documents.

The goal is to classify documents into one of the predefined classes.

We use the data splits used by BID33 and follow similar setup for Cora-ML dataset.

Following BID14 , additional 500 labeled nodes are used for hyperparameter tuning.

Table 1 : Details of the datasets used in the paper.

Please refer Section 6.1 for more details.

Citeseer Cora Pubmed Cora ML LP BID36 45.3 68.0 63.0 -ManiReg BID2 60.1 59.5 70.7 -SemiEmb BID31 59.6 59.0 71.1 -Feat BID33 57.2 57.4 69.8 -DeepWalk BID22 43.2 67.2 65.3 -GGNN BID17 68.1 77.9 77.2 -Planetoid BID33 64.9 75.7 75.7 -Kipf-GCN BID14 70.3 81.5 79.0 51.6 G-GCN 71.1 82.0 77.3 50.4 GPNN BID18 69.7 81.8 79.3 60.6 GAT BID29 72.5 83.0 79.0 54.9ConfGCN (this paper) 73.9 83.5 80.7 80.9 Table 2 : Performance comparison of several methods for semi-supervised node classification on multiple benchmark datasets.

ConfGCN performs consistently better across all the datasets.

Baseline method performances on Citeseer, Cora and Pubmed datasets are taken from Liao et al. FORMULA0 ; BID29 .

We consider only the top performing baseline methods on these datasets for evaluation on the Cora-ML dataset.

Please refer Section 7.1 for details.

We use the same data splits as described in BID33 , with a test set of 1000 labeled nodes for testing the prediction accuracy of ConfGCN and a validation set of 500 labeled nodes for optimizing the hyperparameters.

The model is trained using Adam (Kingma & Ba, 2014) with a learning rate of 0.01.

The weight matrices along with µ are initialized using Xavier initialization BID10 and Σ matrix is initialized with identity.

For evaluating ConfGCN, we compare against the following baselines:• Feat BID33 takes only node features as input and ignores the graph structure.• ManiReg BID2 ) is a framework for providing data-dependent geometric regularization.• SemiEmb BID31 augments deep architectures with semi-supervised regularizers to improve their training.• LP BID36 ) is an iterative iterative label propagation algorithm which propagates a nodes labels to its neighboring unlabeled nodes according to their proximity.• DeepWalk BID22 learns node features by treating random walks in a graph as the equivalent of sentences.• Planetoid BID33 provides a transductive and inductive framework for jointly predicting class label and neighborhood context of a node in the graph.• GCN BID14 ) is a variant of convolutional neural networks used for semisupervised learning on graph-structured data.• G-GCN ) is a variant of GCN with edge-wise gating to discard noisy edges during aggregation.• GGNN BID17 is a generalization of RNN framework which can be used for graphstructured data. .

On xaxis we have quartiles of (a) node entropy and (b) degree, i.e., each bin has 25% of the samples in sorted order.

Overall, we observe that the performance of Kipf-GCN and GAT degrades with the increase in node entropy and degree.

In contrast, ConfGCN is able to avoid such degradation due to its estimation and use of confidence scores.

Refer Section 7.2 for details.• GPNN BID18 ) is a graph partition based algorithm which propagates information after partitioning large graphs into smaller subgraphs.• GAT BID29 ) is a graph attention based method which provides different weights to different nodes by allowing nodes to attend to their neighborhood.

In this section, we attempt to answer the following questions:Q1.

How does ConfGCN compare against the existing methods semi-supervised node classification task? (Section 7.1) Q2.

How does the performance of methods vary with increasing node degree and label mismatch? (Section 7.2) Q3.

What is the effect of ablating different terms in ConfGCN's loss function? (Section 7.4) Q4.

How does increasing the number of layers effects ConfGCN's performance? (Section 7.3)

The evaluation results for semi-supervised node classification are summarized in Table 2 .

Results of all other baseline methods on Cora, Citeseer and Pubmed datasets are taken from BID18 BID29 directly.

Overall, we find that ConfGCN outperforms all existing approaches consistently across all the datasets.

We observe that on the more noisy and challenging Cora-ML dataset, ConfGCN performs considerably better, giving nearly 20% absolute increase in accuracy compared to the previous state-of-the-art method.

This can be attributed to ConfGCN's ability to model nodes' label distribution along with the confidence scores which subdues the effect of noisy nodes during neighborhood aggregation.

The lower performance of G-GCN compared to Kipf-GCN on Cora-ML shows that calculating edge-wise gating scores using the hidden representation of nodes is not much helpful in suppressing noisy neighborhood nodes as the representations lack label information or are over averaged or unstable.

Similar reasoning holds for GAT for its poor performance on Cora-ML.

In this section, we provide an analysis of the performance of Kipf-GCN, GAT and ConfGCN for node classification on Cora-ML dataset which has higher label mismatch rate.

We use neighborhood label entropy to quantify label mismatch, which for a node u is defined as follows.

DISPLAYFORM0 Here, label(v) is the true label of node v. The neighborhood label entropy of a node increases with label mismatch amongst its neighbors.

The problem of node classification becomes difficult with increase in node degree, therefore, we also evaluate the performance of methods with increasing node degree.

The results are summarized in FIG1 .

We find that the performance of both Kipf-GCN and GAT decreases with increase in node entropy and degree.

On the contrary, ConfGCN's performance remains consistent and does not degrade with increase in entropy or degree.

This shows that ConfGCN is able to use the label distributions and confidence effectively to subdue irrelevant nodes during aggregation.

Recently, BID32 highlighted an unusual behavior of Kipf-GCN where its performance degraded significantly with increasing number of layers.

For comparison, we evaluate the performance of Kipf-GCN and ConfGCN on citeseer dataset with increasing number of convolutional layers.

The results are summarized in Figure 3 .

We observe that Kipf-GCN's performance degrades drastically with increasing number of layers, whereas ConfGCN's decrease in performance is more gradual.

We also note that ConfGCN outperforms Kipf-GCN at all layer levels.

In this section, we evaluate the different ablated version of ConfGCN by cumulatively eliminating terms from its objective function as defined in Section 5.

The results on citeseer dataset are summarized in Figure 4 .

Overall, we find that ConfGCN performs best when all the terms in its loss function (Equation 5) are included.

In this paper we present ConfGCN, a confidence based Graph Convolutional Network which estimates label scores along with their confidences jointly in GCN-based setting.

In ConfGCN, the influence of one node on another during aggregation is determined using the estimated confidences and label scores, thus inducing anisotropic behavior to GCN.

We demonstrate the effectiveness of ConfGCN against recent methods for semi-supervised node classification task and analyze its performance in different settings.

We make ConfGCN's source code available.

<|TLDR|>

@highlight

We propose a confidence based Graph Convolutional Network for Semi-Supervised Learning.
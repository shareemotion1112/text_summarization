Graph convolutional neural networks have recently shown great potential for the task of zero-shot learning.

These models are highly sample efficient as related concepts in the graph structure share statistical strength allowing generalization to new classes when faced with a lack of data.

However, we find that the extensive use of Laplacian smoothing at each layer in current approaches can easily dilute the knowledge from distant nodes and consequently decrease the performance in zero-shot learning.

In order to still enjoy the benefit brought by the graph structure while preventing the dilution of knowledge from distant nodes, we propose a Dense Graph Propagation (DGP) module with carefully designed direct links among distant nodes.

DGP allows us to exploit the hierarchical graph structure of the knowledge graph through additional connections.

These connections are added based on a node's relationship to its ancestors and descendants.

A weighting scheme is further used to weigh their contribution depending on the distance to the node.

Combined with finetuning of the representations in a two-stage training approach our method outperforms state-of-the-art zero-shot learning approaches.

With the ever-growing supply of image data, from an ever-expanding number of classes, there is an increasing need to use prior knowledge to classify images from unseen classes into correct categories based on semantic relationships between seen and unseen classes.

This task is called zero-shot image classification.

To obtain satisfactory performance on this task, it is crucial to model precise class relationships based on prior class knowledge.

Previously prior knowledge has been incorporated in form of semantic descriptions of classes, such as attributes BID0 BID27 BID18 or word embeddings BID29 BID10 , or by using semantic relations such as knowledge graphs BID23 BID26 BID28 BID19 .

Approaches that use knowledge graphs are less-explored and generally are based on the assumption that unknown classes can exploit similarity to known classes.

Recently the benefit of hybrid approaches that combine knowledge graph and semantic class descriptions has been illustrated BID31 .The current state-of-the-art approach BID31 processes knowledge graphs by making use of recent developments in applying neural network techniques to non-euclidean spaces, such as graph and manifold spaces BID1 .

A deep graph convolutional neural network (GCN) BID13 ) is used and the problem is phrased as weight regression, where the GCN is trained to regress classifier weights for each class.

GCNs balance model complexity and expressiveness with a simple scalable model relying on the idea of message passing, i.e. nodes pass knowledge to their neighbors.

However, these models were originally designed for classification tasks, albeit semi-supervised, an arguably simpler task than regression.

In recent work, it has been shown that GCNs perform a form of Laplacian smoothing, where feature representations will become more similar as depth increases leading to easier classification BID16 .

In the regression setting, instead, the aim is to exchange information between nodes in the graph and extensive smoothing is not desired as it dilutes information and does not allow for accurate regression.

For instance, in a connected graph all features in a GCN with n layers will converge to the same representation as n → ∞ under some conditions, hence washing out all information BID16 .

Here, graph propagation represents the knowledge that a node receives in a single layer for previous approaches.

b) Proposed dense graph propagation for node 'Cat'.

The node receives knowledge from all its descendants during the descendant phase (blue arrows) and its ancestors during the ancestor phase (red arrows).

This leads to a densely connected graph where knowledge can directly propagate between related nodes.

Weights α k are used to weigh nodes that are k-hops away from a given node.

We, therefore, argue that this approach is not ideal for the task of zero-shot learning and that the number of layers in the graph should be small in order to avoid smoothing.

We illustrate this phenomenon in practice, by showing that a shallow GCN consistently outperforms previously reported results.

We employ a model-of-models framework by training the method to predict a set of logistic regression classifier for each class on top of a set of extracted features produced by a CNN.

Choosing a small number of layers, however, has the effect that knowledge will not propagate well through the graph.

A 1-layer GCN for instance only considers neighbors that are two hops away in the graph such that only immediate neighbors influence a given node.

Thus, we propose a dense connectivity scheme, where nodes are connected directly to descendants/ancestors in order to include distant information.

These connections allow us to propagate information without many smoothing operations but leads to the problem that all descendants/ancestors are weighed equally when computing the regression weight vector for a given class.

However, intuitively, nodes closer to a given node should have higher importance.

To remedy this, we extend this framework by adding a weighting scheme that considers the distance between nodes in order to weigh the contribution of different nodes.

Making use of shared weights based on the distance also has the advantage that it only adds a minimal amount of additional parameters, is computationally efficient, and provides a balance between increasing flexibility of the model and keeping it restrictive enough to allow good predictions for the nodes of the unseen classes.

FIG0 illustrates the difference in the way knowledge is propagated in this proposed Dense Graph Propagation (DGP) module compared to a GCN layer.

To allow the feature extraction stage of the pre-trained CNN to adjust to the newly learned classifiers we propose a two-phase training scheme.

In the first step, the DGP is trained to predict the last layer CNN weights.

In the second phase, we replace the last layer weights of the CNN with the weights predicted by the DGP, freeze the weights and finetune the remaining weights of the CNN by optimizing the cross entropy classification loss on the seen classes.

Our contributions can be summarized as follows.

We present• an analysis of our intuitions for zero-shot learning and illustrate how these intuitions can be combined to design a DGP that outperforms previous zero-shot learning results.

• our DGP module, which explicitly exploits the hierarchical structure of the knowledge graph in order to perform zero-shot learning by more efficiently propagating knowledge through the proposed dense connectivity structure.• a novel weighting scheme for the dense model based on the distance between nodes.• experimental results on various splits of the 21K ImageNet dataset, a commonly used largescale dataset for zero-shot learning.

We obtain relative improvements of more than 50% over the previously reported best results.

Figure 2: DGP is trained to predict classifier weights W for each node/class in a graph.

These weights are extracted from the final layer of a pre-trained ResNet.

The graph is constructed from a knowledge graph and each node is represented by its word embedding (semantic information).

The network consists of two phases, a descendant phase where each node receives knowledge form its descendants and an ancestor phase, where it receives knowledge from its ancestors.

Graph convolutional networks are a class of graph neural networks, based on local graph operators BID2 BID6 BID13 .

Their advantage is that their graph structure allows the sharing of statistical strength between classes making these methods highly sample efficient.

After being introduced in BID2 , they were extended with an efficient filtering approach based on recurrent Chebyshev polynomials, reducing their computational complexity to the equivalent of the commonly used CNNs in image processing operating on regular grids BID6 .

BID13 further proposed simplifications to improve scalability and robustness and applied their approach to semi-supervised learning on graphs.

Their approach is termed graph convolutional network (GCN).Zero-shot learning has in recent years been considered from various set of viewpoints such as manifold alignment BID8 , linear auto-encoder BID14 , and low-rank embedded dictionary learning approaches , using semantic relationships based on attributes BID21 BID29 BID10 and relations in knowledge graphs BID31 BID20 BID26 BID23 .

One of the early works BID15 proposed a method based on the idea of a model-of-model class approach, where a model is trained to predict models based on their description.

Each class is modeled as a function of its description.

This idea has recently been used in another work in BID31 , the work most similar to our own, where a graph convolutional neural network is trained to predict logistic regression classifiers on top of pre-trained CNN features.

BID31 proposed to use GCNs BID13 to predict a set of logistic regression classifiers, one for each class, on top of pre-trained CNN features in order to predict unseen classes.

Their approach has yielded impressive performance on a set of zero-shot learning tasks and can, to the author's knowledge be considered to be the current state-of-the-art.

Here we first formalize the problem of zero-shot learning and provide information on how a GCN model can be utilized for the task of zero-shot learning and then describe our proposed model DGP.

Our zero-shot learning framework to address this task is illustrated in Figure 2 .

We train our DGP, to predict the last layer CNN weights for each class/concept.

Zero-shot classification aims to predict the class labels of a set of test data points to a set of classes C te .

However, unlike in common supervised classification, the test data set points have to be assigned to previously unseen classes, given a L dimensional semantic representation vector z ∈ R L per class C and a set of training data points D tr = {( X i , c i ) i = 1, ..., N }, where X i denotes the i-th training image and c i ∈ C tr the corresponding class label.

Here C denotes the set of all classes and C te and C tr the test and training classes, respectively.

Note that training and test classes are disjoint C te ∩ C tr = ∅ for the zero-shot learning task.

In this work, we perform zero-shot classification by using the word embedding of the class labels and the knowledge graph to predict classifiers for each unknown class in form of last layer CNN weights.

Given a graph with N nodes and with C input features per node, X ∈ R N ×C is used to denote the feature matrix.

Here each node represents a distinct concept/class in the classification task and each concept is represented by a word vector of the class name.

The connections between the classes in the knowledge graph are encoded in form of a symmetric adjacency matrix A ∈ R N ×N , which also includes self-loops.

We employ a simple propagation rule to perform convolutions on the graph DISPLAYFORM0 where H (l) represents the activations in the l th layer and Θ ∈ R C×F denotes the trainable weight matrix for layer l.

For the first layer, DISPLAYFORM1 N ×N , which normalizes rows in A to ensure that the scale of the feature representations is not modified by A. Similar to previous work done on graph convolutional neural networks, this propagation rule can be interpreted as a spectral convolution BID13 .The model is trained to predict the classifier weights for the seen classes by optimizing the loss DISPLAYFORM2 where W ∈ R M ×L denotes the prediction of the GCN for the known classes and therefore corresponds to the M rows of the GCN output, which correspond to the training classes.

M denotes the number of training classes and L denotes the dimensionality of the weight vectors.

The ground truth weights are obtained by extracting the last layer weights of a pre-trained CNN and denoted as W ∈ R M ×L .

During testing, the features of new images are extracted from the CNN and the classifiers predicted by the GCN are used to classify the features.

Our DGP for zero-shot learning aims to utilize the hierarchical graph structure for the zero-shot learning task and avoids the dilution of knowledge by intermediate nodes.

This is achieved using a dense graph connectivity scheme consisting of two phases, namely the descendant propagation phase and the ancestor propagation phase.

This two-phase approach further enables the model to learn separate relations between a node and its ancestors and a node and its descendants.

Appendix B provides empirical evidence for this choice.

Unlike in the GCN, we do not use the knowledge graph relations directly as an adjacency graph to include information from neighbors further away.

We do therefore not suffer from the problem of knowledge being washed out due to averaging over the graph.

Instead, we introduce two separate connectivity patterns, one where nodes are connected to all their ancestors and one where nodes are connected to all descendants.

We utilize two adjacency matrices A a ∈ R N ×N that denotes the connections from nodes to their ancestors and adjacency matrix A d that denotes the connections from nodes to their descendants.

Note, A d = A T a .

Unlike in previous approaches, this connectivity pattern allows nodes direct access to knowledge in their extended neighborhood as opposed to knowledge that has been modified by intermediate nodes.

Note that both these adjacency matrices include self-loops.

The connection pattern is illustrated in FIG0 .

The same propagation rule as in Equation 1 is applied consecutively for the two connectivity patterns leading to the overall DGP propagation rule DISPLAYFORM0 Distance weighting scheme In order to allow DGP to weigh the contribution of various neighbors in the dense graph, we propose a weighting scheme that weighs a given nodes neighbors based on the graph distance from the node.

Note, the distance is computed on the knowledge graph and not the dense graph.

We use w a = {w DISPLAYFORM1 to denote the weights for the ancestor and the descendant propagation phase, respectively.

w DISPLAYFORM2

Training of the proposed model is done in two stages, where the first stage trains the DGP to predict the last layer weights of a pre-trained CNN using Equation 2.

Note, W , in this case, contains the M rows of H, which correspond to the training classes.

In order to allow the feature representation of the CNN to adapt to the new class classifiers, we train the CNN by optimizing the cross-entropy classification loss on the seen classes in a second stage.

During this stage, the last layer weights are fixed to the predicted weights of the training classes in the DGP and only the feature representation is updated.

This can be viewed as utilizing the DGP as a constraint for the CNN, as we indirectly incorporate the graph information in order to constrain the CNN output space.

We use a ResNet-50 BID11 model that has been pre-trained on the ImageNet 2012 dataset.

Following BID31 , we use the GloVe text model BID25 , which has been trained on the Wikipedia dataset, as the feature representation of our concepts in the graph.

The DGP model consists of two layers as illustrated in Equation 3 with feature dimensions of 2048 and the final output dimension corresponds to the number of weights in the last layer of the ResNet-50 architecture, 2049 for weights and bias.

Following the observation of BID31 , we perform L2-Normalization on the outputs as it regularizes the outputs into similar ranges.

Similarly, we also normalize the ground truth weights produced by the CNN.

We further make use of Dropout BID30 ) with a dropout rate of 0.5 in each layer.

The model is trained for 3000 epochs with a learning rate of 0.001 and weight decay of 0.0005 using Adam BID12 .

We make use of leaky ReLUs with a negative slope of 0.2.

The number of values per phase K was set to 4 as additional weights had diminishing returns.

The proposed DGP model is implemented in PyTorch BID24 and training and testing are performed on a GTX 1080Ti GPU.

Finetuning is done for 20 epochs using SGD with a learning rate of 0.0001 and momentum of 0.9.

We performed a comparative evaluation of the DGP against previous state-of-the-art on the ImageNet dataset BID7 , the largest commonly used dataset for zero-shot learning.

In our work, we follow the train/test split suggested by BID10 , who proposed to use the 21K ImageNet dataset for zero-shot evaluation.

They define three tasks in increasing difficulty, denoted as "2-hops", "3-hops" and "All".

Hops refer to the distance that classes are away from the ImageNet 2012 1K classes in the ImageNet hierarchy and thus is a measure of how far unseen classes are away from seen classes.

"2-hops" contains all the classes within two hops from the seen classes and consists of roughly 1.5K classes, while "3-hops" contains about 7.8K classes.

FORMULA0 we further evaluate the performance when training categories are included as potential labels.

Note that since the only difference is the number of classes during testing, the model does not have to be retrained.

We denote the splits as "2-hops+1K", "3-hops+1K", "All+1K".

We compare our DGP to the following approaches: Devise BID10 linearly maps visual information in form of features extracted by a convolutional neural network to the semantic word-embedding space.

The transformation is learned using a hinge ranking loss.

Classification is performed by assigning the visual features to the class of the nearest word-embedding.

ConSE BID22 projects image features into a semantic word embedding space as a convex combination of the T closest seen classes semantic embedding weighted by the probabilities that the image belongs to the seen classes.

The probabilities are predicted using a pre-trained convolutional classifier.

Similar to Devise, ConSE assigns images to the nearest classes in the embedding space.

EXEM BID4 creates visual class exemplars by averaging the PCA projections of images belonging to the same seen class.

A kernel-based regressor is then learned to map a semantic embedding vector to the class exemplar.

For zero-shot learning visual exemplars can be predicted for the unseen classes using the learned regressor and images can be assigned using nearest neighbor classification.

SYNC ) aligns a semantic space (e.g., the word-embedding space) with a visual model space, adds a set of phantom object classes in order to connect seen and unseen classes, and derives new embeddings as a convex combination of these phantom classes.

GCNZ BID31 represents the current state of the art and is the approach most related to our proposed DGP.

A GCN is trained to predict last layer weights of a convolutional neural network.

Guided by experimental evidence (see Appendix C) and our intuition that extensive smoothing is a disadvantage for the weight regression in the task of zero-shot learning we add as another baseline, a single-hidden-layer GCN (SGCN) with non-symmetric normalization (D −1 A) (as defined in Equation 1).

Note, GCNZ made use of a symmetric normalization (D −1/2 AD −1/2 ) but our experimental evaluation indicates that the difference is negligible (see Appendix D).

It further yields a better baseline as our proposed DGP also utilizes the non-symmetric normalization.

As DGP, our SGCN model makes use of the proposed two-stage finetuning approach.

Quantitative results for the comparison on the ImageNet datasets are shown in TAB0 .

Compared to previous results such as ConSE , EXEM BID4 , and GCNZ BID31 our proposed methods outperform the previous results with a considerable margin, achieving, for instance, more than 50% relative improvement for Top-1 accuracy on the 21K ImageNet "All" dataset.

We observe that our methods especially outperform the baseline models on the "All" task, illustrating the potential of our methods to more efficiently propagate knowledge.

DGP also achieves consistent improvements over the SGCN model.

We observed that finetuning consistently improved performance for both models in all our experiments.

Ablation studies that highlight the impact of finetuning and weighting of neighbors for the 2-hop scenario can be found in TAB2 .

DGP(-wf) is used to denote the accuracy that is achieved after training the DGP model without weighting (adding no weights in Equation 4) and without finetuning.

DGP(-w) and DGP(-f) are used to denote the results for DGP without weighting and DGP without finetuning, respectively.

We further report the accuracy achieved by the SGCN model without finetuning (SGCN(-f) ).

We observe that the proposed weighting scheme, which allows distant neighbors to have less impact, is crucial for the dense approach.

Further, finetuning the model consistently leads to improved results.

The results are stable over multiple runs and we include variance information for multiple runs in Appendix E.Qualitative results of DGP and the SGCN are shown in FIG6 .

Example images from unseen test classes are displayed and we compare the results of our proposed DGP and the SGCN to results produced by a pre-trained ResNet.

Note, ResNet can only predict training classes while the others predict classes not seen in training.

For comparison, we also provide results for our re-implementation of GCNZ.

We observe that the SGCN and DGP generally provide coherent top-5 results.

All methods struggle to predict the opener and tend to predict some type of plane instead, however, DGP does include opener in the top-5 results.

We further observe that the prediction task on this dataset for zero-shot learning is difficult as it contains classes of fine granularity, such as many different types of squirrels, planes, and furniture.

Additional examples are provided in the appendix.

Testing including training classifiers.

Following the example of BID10 BID22 BID31 , we also report the results when including both training labels and testing labels as potential labels during classification of the zero-shot examples.

Results are shown in Table 2 .

For the baselines, we include two implementations of ConSE, one that uses AlexNet as a backbone BID22 and one that uses ResNet-50 BID31 .

Compared to TAB0 , we observe that the accuracy is considerably lower, but the SGCN and DGP still outperform the previous state-of-the-art approach GCNZ.

SGCN outperforms DGP for low k in the Top-k accuracy measure especially for the 2-hops setting, while DGP outperforms SGCN for larger k. We observe that DGP tends to favor prediction to the closest training classes for its Top-1 prediction (see TAB3 ).

However, this is not necessarily a drawback and is a well-known tradeoff between performing well on the unseen classes and the seen classes, which are not considered in this setting.

In the next paragraph, we will evaluate the model's performance on the seen classes.

This tradeoff can be controlled by including a novelty detector, which predicts if an image comes from the seen or unseen classes as done in BID29 and then assigns it to the zero-shot classifier or a classifier trained on the seen classes.

Another approach is calibrated stacking , which rescales the prediction scores of the known classes.

Zero-shot learning models should perform well not only on unseen but also on seen classes.

To put the zero-shot performance into perspective, we perform experiments where we analyze how the model's performance on the original 1000 seen classes is affected by domain shift as additional unseen classes (all 2-hop classes) are introduced.

TAB3 shows the results when the model is tested on the validation dataset from ImageNet 2012.

We compare the performance to our re-implementation of the GCNZ model with ResNet-50 backbone and also the performance from the original ResNet-50 model, which is trained only on the seen classes.

It can be observed that both our methods outperform GCNZ on Hit@1 and Hit@2 accuracy.

Analysis of weighting scheme In order to validate our intuition that weighting allows our approach to weigh distance neighbors less, we can inspect the learned weighting.

For the first stage the weights are Note, the first value corresponds to self-attention, the second to the 1-hop neighbors, and so forth.

For the first stage, ancestors aggregate information mainly from their immediate descendants to later distribute it to their descendants.

Further, we observe that distant neighbors have less impact in the final stage.

Scalability.

To obtain good scalability it is important that the adjacency matrix A is a sparse matrix so that the complexity of computing D −1 AXΘ is linearly proportional to the number of edges present in A. Our approach utilizes the structure of knowledge graphs, where entities only have few ancestors and descendants, to ensure this.

The adjacency matrix for the ImageNet hierarchy used in our experiments, for instance, has a density of 9.3 × 10 −5 , while our dense connections only increase the density of the adjacency matrix to 19.1 × 10 −5 .

With regards to the number of parameters, the SGCN consist of 4,810,752 weights.

DGP increases the number of trainable parameters by adding 2 × (K + 1) additional weights.

However, as K = 4 in our experiments, this difference in the number of parameters is negligible.

In contrast to previous approaches using graph convolutional neural networks for zero-shot learning, we illustrate that the task of zero-shot learning benefits from shallow networks.

Further, to avoid the lack of information propagation between distant nodes in shallow models, we propose DGP, which exploits the hierarchical structure of the knowledge graph by adding a dense connection scheme.

Experiments illustrate the ability of the proposed methods, outperforming previous state-of-the-art methods for zero-shot learning.

In future work, we aim to investigate the potential of more advanced weighting mechanisms to further improve the performance of DGP compared to the SGCN.

The inclusion of additional semantic information for settings where these are available for a subset of nodes is another future direction.

A QUALITATIVE RESULTS Figure 4 and 5 provide further qualitative results of our finetuned Graph Propagation Module GPM and Dense Graph Propagation Module DGP compared to a standard ResNet and GCNZ, our reimplementation of BID31 .upright, grand piano, organ, accordion, barbershop piano, spinet, keyboard instrument, concert grand, baby grand piano, spinet, concert grand, baby grand, keyboard instrument piano, baby grand, concert grand, spinet, keyboard instrument B TWO-PHASE PROPAGATION TAB4 illustrates the benefit of a two-phase directed propagation rule where ancestors and descendants are considered individually compared to two consecutive updates using the full adjacency matrix in the dense method.

C ANALYSIS OF NUMBER OF LAYERS TAB5 illustrates the drop in performance that is caused by using additional hidden layers in the GCN for the 2-hops experiment.

All hidden layers have dimensionality of 2048 with 0.5 dropout.

TAB6 explains the performance difference between our SGCN, our reimplementation of GCNZ and the reported results in BID31 .

Note, unless otherwise stated training is performed for 3000 epochs.

Non-symmetric normalization (D −1 A) is denoted as non-sym in the normalization column, while a symmetric normalization (D −1/2 AD −1/2 ) is denoted as sym.

No finetuning has been performed for SGCN in these results.

TAB7 shows the mean and std for 3 runs for the 2-hops and All dataset.

It can clearly be observed that as the number of classes increases (2-hops to all), results become more stable.

<|TLDR|>

@highlight

We rethink the way information can be exploited more efficiently in the knowledge graph in order to improve performance on the Zero-Shot Learning task and propose a dense graph propagation (DGP) module for this purpose.

@highlight

This authors propose a solution to the problem of over-smoothing in Graph conv networks by allowing dense propagation between all related nodes, weighted by the mutual distance.

@highlight

Proposes a novel graph convolutional neural network to tackle the problem of zero-shot classification by using relational structures between classes as input of graph convolutional networks to learn classifiers of unseen classes
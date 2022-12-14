We study the robustness to symmetric label noise of GNNs training procedures.

By combining the nonlinear neural message-passing models (e.g. Graph Isomorphism Networks, GraphSAGE, etc.) with loss correction methods, we present a noise-tolerant approach for the graph classification task.

Our experiments show that test accuracy can be improved under the artificial symmetric noisy setting.

Large datasets are beneficial to modern machine learning models, especially neural networks.

Many studies have shown that the accuracy of machine learning models grows log-linear to the amount of training data BID9 .

Currently, complex machine learning models can only achieve superhuman classification results when trained with a very large dataset.

However, large datasets are usually expensive to collect and create exact label.

One solution to create large datasets is crowdsourcing, but this approach introduces a higher level of labeling error into the datasets as well as requires a lot of human resources BID1 .

As a consequence, neural networks are prone to very high generalization error under noisy label data.

Figure 1 demonstrate the accuracy results of a graph neural network trained on MUTAG dataset.

Training accuracies tend to remain high while testing accuracies degrades as more label noise is added to the training data.

Figure 1: GIN model trained with increasing symmetric label noise.

The generalization gap increases as more noise is introduced to the training labels.

Graph neural network (GNN) is a new class of neural networks which learn from graphstructured data.

Typically, GNNs classify graph vertices or the whole graph itself.

Given the input as the graph structure and data (e.g. feature vectors) on each vertex, GNNs training aim to learn a predictive model for classification.

This new class of neural networks enables end-toend learning from a wider range of data format.

In order to build large scale GNNs, it requires large and clean datasets.

Since graph data is arguably harder to label than image data both at vertex-level or graph-level, graph neural networks should have a mechanism to adapt to training label error or noise.

In this paper, we take the noise-correction approach to train a graph neural network with noisy labels.

We study two state-of-the-art graph neural network models: Graph Isomorphism Network BID7 and GraphSAGE BID2 .

Both of these models are trained under symmetric artificial label noise and tested on uncorrupted testing data.

We then apply label noise estimation and loss correction techniques BID5 BID9 to propose our denoising graph neural network model (D-GNN).

Notations and Assumption Let G = (V, E, X) be a graph with vertex set V , edge set E and vertex feature vector matrix X ??? R |V |??f , where f is the dimensionality of vertex features.

Our task is graph classification with noisy labels.

Given a set of graphs: {G 1 , G 2 , . . .

, G N }, their labels {??? 1 ,??? 2 , . . .

,??? N } ??? 2 m , we aim to learn a neural network model for graph label prediction: y G = f (G).

We assume that the training data is corrupted by a noise process N , N i,j is the probability label i being corrupted to label j. We further assume N is symmetric, which corresponds to the symmetric label noise setting.

Noise matrix N is unknown, so we estimate N by learning correction matrix C from the noisy training data.

The most modern approach to the graph classification problem is to learn a graphlevel feature vector h G .

There are several ways to learn h G .

GCN approach by BID3 approximates the Fourier transformation of signals (feature vectors) on graphs to learn representations of a special vertex to use as the representative for the graph.

Similar approaches can be founded in the context of compressive sensing.

To overcome the disadvantages of GCN-like methods such as memory consumption and scalability, the nonlinear neural message passing method is proposed.

GraphSAGE BID2 proposes an algorithm consists of two operations: aggregate and pooling.

aggregate step computes the information on each vertex using the local neighborhood, then pooling computes the output for each vertex.

These vector outputs are then used in classification at vertex-level or graph-level.

More recently, GIN BID7 model generalizes the concept in GraphSAGE to propose a unified message-passing framework for graph classification.

Surrogate Loss Using an alternative loss function to deal with noisy label data is a common practice in the weakly supervised learning literature BID4 BID0 BID1 BID5 BID9 .

We apply the backward loss correction procedure to graph neural network: DISPLAYFORM0 .

This loss can be intuitively understood as going backward one step in the noise process C BID6 .We study the symmetric noise setting where label i is corrupted to label j with the same probability for j to i (N i,j = N j,i ) BID0 .

We use a m ?? m symmetric Markov matrix N to describe the noisy process with m labels.

Furthermore, to simplify the experiment settings, with a given n we set: Matrix N above can be interpreted as all labels are kept with probability 0.8 and corrupted to other labels with probability 0.2 (summation of off-diagonal elements in a row).

DISPLAYFORM1

Formaly we define our graph neural network model as the message passing approach proposed by BID7 .

The feature vector h v of a vertex V at k-th hop (or layer) is given by AGGREGATE and COMBINE functions: DISPLAYFORM0 N (v) denotes the neighborhood set of vertex v; and k ??? [K] is the predefined number of "layers" corresponding to network's perceptive field.

The final representation of graph G is calculated using a READOUT function.

Then, we train the neural network by optimizing the surrogate backward loss.

DISPLAYFORM1 D-GNN is different from GIN only at the surrogate loss function as described above.

To train a D-GNN model, we first train a GIN model on the noisy data for estimating C, then we train D-GNN using the estimated correction matrix.

We train our D-GNN model using three different noise estimator: Conservative (D-GNN-C), Anchors (D-GNN-A), and Exact (D-GNN-E).

The exact loss correction is introduced for comparison purposes.

The hyperparameters of our models are set similar to GIN model in the previous paragraph.

For conservative and anchor correction matrix estimation, we train two models on the same noisy dataset: The first model is without loss correction and the second model is trained using the correction matrix from the first model.

For all neural network models, we use the ReLU activation unit as the nonlinearity.

We test our framework on the set of well-studied 9 datasets for the graph classification task: 4 bioinformatics datasets (MUTAG, PTC, NCI1, PROTEINS), and 5 social network datasets (COLLAB, IMDB-BINARY, IMDB-MULTI, REDDIT-BINARY, REDDIT-MULTI5K) BID8 .

We follow the preprocessing suggested by BID7 to use onehot encoding as vertex degrees for social networks (except REDDIT datasets).

TAB0 gives the overview of each dataset.

Since these datasets have exact label for each graph, we introduce symmetric label noise artificially.

Conservative Estimation We estimate the corruption probability by the Conservative Estimator described in the previous sections.

For each noise configuration, we train the original neural network (GIN) on the noisy data and use the neural response to fill each row of the correction matrix C. Table 2 gives an overview of how well the conservative estimation matrix diverges from the correct noise matrix.

The matrix norm C ??? N is the p-norm with p = 1.

Anchor Estimation We follow the noise estimation method introduced in BID6 FIG2 ) to estimate the noise probability using an unseen set of samples.

These anchor samples are assumed to have the correct labels, hence they can be used to estimate the noise matrix according to the expressivity assumption.

In our experiments, these samples are taken from the testing data (one per class).

Table 2 demonstrates the similarity results.

Table 2 : Norm distance between conservative correction matrix estimation C c and C a compared with true noise matrix N when n = 0.2 Exact Assumption In this experiment setting, we assume that the noise matrix is exactly known from some other estimation process.

In practice, such an assumption might not be realistic.

However, under the symmetric noise assumption, the diagonal of the correction matrix C can be tuned as a hyperparameter.

DISPLAYFORM0

We compare our model with the original Graph Isomorphism Network (GIN) BID7 .

The hyperparameters are fixed across all datasets as follow: epochs=20, num layers=5, num mlp layers=2, batch size=64.

We keep these hyperparameters fixed for all datasets since the similar trend of accuracy degradation is observed independently of hyperparameter tuning.

Besides GIN, we consider GraphSAGE model BID2 under the same noisy setting.

We use the default setting for GraphSAGE as suggested in the original paper.

We fix the noise rate at 20% for the experiments in TAB2 and report the mean accuracy after 10 fold cross validation run.

The worst performance variance of our model is the conservative estimation model.

Due to the overestimation of softmax unit within the cross-entropy loss, the model's confidence to all training data is close to 1.0.

Such overconfidence leads to wrong correction matrix estimation, which in turn leads to worse performance (Table 2) .

In contrast to D-GNN-C, D-GNN-A and D-GNN-E have consistently outperformed the original model.

Such improvement comes from the fact that the correction matrix C is correctly approximated.

FIG2 suggests that the D-GNN-C model might work well under the higher label noise settings.

In this paper, we have introduced the use of loss correction for Graph Neural Networks to deal with symmetric graph label noise.

We experimented on two different practical noise estimatation methods and compare them to the case when we know the exact noise matrix.

Our empirical results show some improvement on noise tolerant when the correction matrix C is correctly estimated.

In practice, we can consider C as a hyperparameter and tune it following some clean validation data.

<|TLDR|>

@highlight

We apply loss correction to graph neural networks to train a more robust to noise model.

@highlight

This paper introduces loss correction for Graph Neural Networks to deal with symmetric graph label noise, focused on a graph classification task.

@highlight

This paper proposes the use of a noise correction loss in the context of graph neural networks to deal with noisy labels.
Graph Neural Networks (GNNs) for prediction tasks like node classification or edge prediction have received increasing attention in recent machine learning from graphically structured data.

However, a large quantity of labeled graphs is difficult to obtain, which significantly limit the true success of GNNs.

Although active learning has been widely studied for addressing label-sparse issues with other data types like text, images, etc., how to make it effective over graphs is an open question for research.

In this paper, we present the investigation on active learning with GNNs for node classification tasks.

Specifically, we propose a new method, which uses node feature propagation followed by K-Medoids clustering of the nodes for instance selection in active learning.

With a theoretical bound analysis we justify the design choice of our approach.

In our experiments on four benchmark dataset, the proposed method outperforms other representative baseline methods consistently and significantly.

Graph Neural Networks (GNN) (Kipf & Welling, 2016; Veličković et al., 2017; Hamilton et al., 2017; Wu et al., 2019) have been widely applied in many supervised and semi-supervised learning scenarios such as node classifications, edge predictions and graph classifications over the past few years.

Though GNN frameworks are effective at fusing both the feature representations of nodes and the connectivity information, people are longing for enhancing the learning efficiency of such frameworks using limited annotated nodes.

This property is in constant need as the budget for labeling is usually far less than the total number of nodes.

For example, in biological problems where a graph represents the chemical structure (Gilmer et al., 2017; Jin et al., 2018 ) of a certain drug assembled through atoms, it is not easy to obtain a detailed analysis of the function for each atom since getting expert labeling advice is very expensive.

On the other hand, people can carefully design a small "seeding pool" so that by selecting "representative" nodes or atoms as the training set, a GNN can be trained to get an automatic estimation of the functions for all the remaining unlabeled ones.

Active Learning (AL) (Settles, 2009; Bodó et al., 2011) , following this lead, provides solutions that select "informative" examples as the initial training set.

While people have proposed various methods for active learning on graphs (Bilgic et al., 2010; Kuwadekar & Neville, 2011; Moore et al., 2011; Rattigan et al., 2007) , active learning for GNN has received relatively few attention in this area.

Cai et al. (2017) and Gao et al. (2018) are two major works that study active learning for GNN.

The two papers both use three kinds of metrics to evaluate the training samples, namely uncertainty, information density, and graph centrality.

The first two metrics make use of the GNN representations learnt using both node features and the graph; while they might be reasonable with a good (well-trained) GNN model, the metrics are not informative when the label budget is limited and/or the network weights are under-trained so that the learned representation is not good.

On the other hand, graph centrality ignores the node features and might not get the real informative nodes.

Further, methods proposed in Cai et al. (2017) ; Gao et al. (2018) only combine the scores using simple linear weighted-sum, which do not solve these problems principally.

We propose a method specifically designed for GNN that naturally avoids the problems of methods above 1 .

Our method select the nodes based on node features propagated through the graph structure, 1 Our code will be released upon acceptance.

making it less sensitive to inaccuracies of representation learnt by under-trained models.

Then we cluster the nodes using K-Medoids clustering; K-Medoids is similar to the conventional K-Means, but constrains the centers to be real nodes in the graph.

Theoretical results and practical experiments prove the strength of our algorithm.

• We perform a theoretical analysis for our method and study the relation between its classification loss and the geometry of the propagated node features.

• We show the advantage of our method over Coreset (Sener & Savarese, 2017) by comparing the bounds.

We also conjecture that similar bounds are not achievable if we use raw unpropagated node features.

• We compare our method with several AL methods and obtain the best performance over all benchmark datasets.

Active Learning (AL) aims at interactively choosing data points from the training pool to maximize model performances, and has been widely studied both in theory (Beygelzimer et al., 2008; Hanneke, 2014) and practice (Settles, 2009; Shen et al., 2017) .

Recently, Sener & Savarese (2017) proposes to compute a Coreset over the last-layer activation of a convolutional neural network.

The method is designed for general-purpose neural networks, and does not take the graph structure into account.

Early works on AL with graph-structured data (Dasarathy et al., 2015; Mac Aodha et al., 2014) study non-parametric classification models with graph regularization.

More recent works analyze active sampling under the graph signal processing framework (Ortega et al., 2018; Chen et al., 2016) .

However, most of these works have focused on the denoising setting where the signal is smooth over the graphs and labels are noisy versions of node features.

Similarly, optimal experimental design (Pukelsheim, 2006; Allen-Zhu et al., 2017) can also apply to graph data but primarily deals with linear regression problems, instead of nonlinear classification with discrete labels.

Graph Neural Networks (GNNs) (Hamilton et al., 2017; Veličković et al., 2017; Kipf & Welling, 2016) are the emerging frameworks in the recent years when people try to model graph-structured data.

Most of the GNN variants follow a multi-layer paradigm.

In each layer, the network performs a message passing scheme, so that the feature representation of a node in the next layer could be some neighborhood aggregation from its previous layer.

The final feature of a single node thus comprises of the information from a multi-hop neighborhood, and is usually universal and "informative" to be used for multiple tasks.

Recent works show the effectiveness of using GNNs in the AL setting.

Cai et al. (2017) , for instance, proposes to linearly combine uncertainty, graph centrality and information density scores and obtains the optimal performance.

Gao et al. (2018) further improves the result by using learnable combination of weights with multi-armed bandit techniques.

Instead of combining different metrics, in this paper, we approach the problem by clustering propagated node features.

We show that our one-step active design outperforms existing methods based on learnt network represenations, in the small label setting, while not degrading in performance for larger amounts of labeled data.

In this section, we describe a formal definition for the problem of graph-based active learning under the node classification setting and introduce a uniform set of notations for the rest of the paper.

We are given a large graph G = (V, E), where each node v ∈ V is associated with a feature vector x v ∈ X ⊆ R d , and a label y v ∈ Y = {1, 2, ..., C}. Let V = {1, 2, ..., n}, we denote the input features as a matrix X ∈ R n×d , where each row represents a node, and the labels as a vector Y = (y 1 , ..., y n ).

We also consider a loss function l(M|G, X, Y ) that computes the loss over the inputs (G, X, Y ) for a model M that maps G, X to a prediction vectorŶ ∈ Y n .

Following previous works on GNN (Cai et al., 2017; Hamilton et al., 2017) , we consider the inductive learning setting; i.e., a small part of Y is revealed to the algorithm, and we wish to minimize the loss on the whole graph l(M|G, X, Y ).

Specifically, an active learning algorithm A is initially given the graph G and feature matrix X. In step t of operation, it selects a subset s t ⊆

[n] = {1, 2, ..., n}, and obtains y i for every i ∈ s t .

We assume y i is drawn randomly according to a distribution P y|xi supported on Y; we use η c (v) = Pr[y = c|v] to denote the probability that y = c given node v, and

T .

Then A uses G, X and y i for i ∈ s 0 ∪ s 1 ∪ · · · ∪ s t as the training set to train a model, using training algorithm M. The trained model is denoted as M At .

If M is the same for all active learning strategies, we can slightly abuse the notation A t = M At to emphasize the focus of active learning algorithms.

A general goal of active learning is then to minimize the loss under a given budget b: min

where the randomness is over the random choices of Y and A. We focus on M being the Graph Neural Networks and their variants elaborated in detail in the following part.

Graph Neural Networks define a multi-layer feature propagation process similar to Multi-Layer Perceptrons (MLPs).

Denote the k-th layer representation matrix of all nodes as X (k) , and X (0) ∈ R n×d are the input node features.

Graph Neural Networks (GNNs) differ in their ways of defining the recursive function f for the next-layer representation:

where Θ k is the parameter for the k-th layer.

Naturally, the input X satisfies X (0) = X by definition.

Graph Convolution Network (GCN).

A GCN (Kipf & Welling, 2016 ) has a specific form of the function f as:

where ReLU is the element-wise rectified-linear unit activation function (Nair & Hinton, 2010) , Θ k is the parameter matrix used for transforming the size of feature representations to a different dimension and S is the normalized adjacency matrix.

Specifically, S is defined as:

where A is the original adjacency matrix associated with graph G and D is the diagonal degree matrix of A. Intuitively, this operation updates node embeddings by the aggregation of their neighbors.

The added identity matrix I (equivalent to adding self-loops to G) acts in a similar spirit to the residual links (He et al., 2016) in MLPs that bypasses shallow-layer representations to deep layers.

By applying this operation in a multi-layer fashion, a GCN encourages nodes that are locally related to share similar deep-layer embeddings and prediction results thereafter.

For the classification task, it is normal to stack a linear transformation along with a softmax function to the representation in the final layer, so that each class could have a prediction score.

That is,

where softmax(x) = exp(x)/ C c=1 exp(x c ) which makes the prediction scores have unit sum of 1 for all classes, and K is the total number of layers.

We use the GCN structure as the fixed unified model M for all the following discussed AL strategies A.

Traditionally, active learning algorithms choose one instance at a time for labeling, i.e., with |s t | = 1.

However, for modern datasets where the numbers of training instances are very large, it would be extremely costly if we re-train the entire system each time when a new label is obtained.

Hence we focus on the "batched" one-step active learning setting (Contardo et al., 2017) , and select the informative nodes once and for all when the algorithm starts.

This is also called the optimal experimental design in the literature (Pukelsheim, 2006; Allen-Zhu et al., 2017) .

Aiming to select the b most representative nodes as the batch, our target (1) becomes: min

The node selection algorithm is described in Section 4.1, followed by the loss bound analysis in Section 4.2, and the comparison with a closely related algorithm (K-Center in Coreset (Sener & Savarese, 2017) ) in Section 4.3.

Input: Node representation matrix X, graph structure matrix G and budget We describe a generic active learning framework using distance-based clustering in Algorithm 1.

It acts in two major steps: 1) computing a distance matrix or function d X,G using the node feature representations X and the graph structure G; 2) applying clustering with b centers over this distance matrix, and from each cluster select the node closest to the center of the cluster.

After receiving the labels (given by matrix Y ) of the selected nodes, we train a graph neural network, specifically GCN, based on X, G and Y for the node classification task.

Generally speaking, different options for the two steps above would yield different performance in the down-stream prediction tasks; we detail and justify our choices below and in subsequent sections.

Distance Function.

Previous methods (Sener & Savarese, 2017; Cai et al., 2017; Gao et al., 2018) commonly use network representations to compute the distance, i.e.,

for some specific k. While this can be helpful in a well-trained network, the representations are quite inaccurate in initial stages of training and such distance function might not select the representatitive nodes.

Differently, we define the pairwise node distance using the L 2 norm of the difference between the corresponding propagated node features:

where (M ) i denotes the i-th row of matrix M , and recall that K is the total number of layers.

Intuitively, this removes the effect of untrained parameters on the distance, while still taking the graph structure into account.

Clustering Method.

Two commonly used methods are K-Means (Cai et al., 2017; Gao et al., 2018) and K-Center (Sener & Savarese, 2017) 2 .

We propose to apply the K-Medoids clustering.

K-Medoids problem is similar to K-Means, but the center it selects must be real sample nodes from the dataset.

This is critical for active learning, since we cannot try to label the unreal cluster centers produced by K-Means.

Also, we show in Section 4.3 that K-Medoids can obtain a more favorable loss bound than K-Center.

We call our method FeatProp, to emphasize the active learning strategy via node feature propagation over the input graph, which is the major difference from other node selection methods.

Recall that we use (S K X)

i − (S K X) j 2 to approximate the pairwise distances between the hidden representations of nodes in GCN.

Intuitively, representation S K X resembles the output of a simplified GCN (Wu et al., 2019) by dropping all activation functions and layer-related parameters in the original structure, which introduces a strong inductive bias.

In other words, the selected nodes could possibly contribute to the stabilization of model parameters during the training phase of GCN.

The following theorem formally shows that using K-Medoids with propagated features can lead to a low classification loss: Theorem 1 (informal).

Suppose that the label vector Y is sampled independently from the distribution y i ∼ η(i), and the loss function l is bounded by [−L, L].

Then under mild assumptions, there exists a constant c 0 such that with probability 1 − δ the expected classification loss of A t satisfies

To understand Theorem 1, notice that the first term

the target loss of K-Medoids (sum of point-center distances), and the second term

quickly decays with n, where n is the total number of nodes in graph G. Therefore the classification loss of A 0 on the entire graph G is mostly dependent on the K-Medoids loss.

In practice, we can utilize existing robust initialization algorithms such as Partitioning Around Medoids (PAM) to approximate the optimal solution for K-Medoids clustering.

The assumptions we made in Theorem 1 are pretty standard in the literature, and we illustrate the details in the appendix.

While our results share some common characteristics with Sener et al. (Sener & Savarese, 2017) , our proof is more involved in the sense that it relates to the translated features (S K X) i − (S K X) j 2 instead of the raw features (X) i − (X) j 2 .

In fact, we conjecture that using raw feature clustering selection for GCN will not result in a similar bound as in (8): this is because GCN uses the matrix S to diffuse the raw features across all nodes in V , and the final predictions of node i will also depend on its neighbors as well as the raw feature (X) i .

We could see a clearer comparison in practice in Section 5.2.

Figure 1: Visualization of Theorem 1.

Consider the set of selected points s and the remaining points in the dataset [n]\s.

K-Medoids corresponds to the mean of all red segments in the figure, whereas K-Center corresponds to the max of all red segments in the figure.

In this subsection we provide justifications on using the K-Medoids clustering method as opposed to Coreset (Sener & Savarese, 2017) .

The Coreset approach aims to find a δ-cover of the training set.

In the context of using propagated features, this means solving

We can show a similar theorem as Theorem 1 for the Coreset approach: Theorem 2.

Under the same assumptions as in Theorem 1, with probability 1 − δ the expected classification loss of A t satisfies

It is easy to see that RHS of Eqn. (8) is smaller than RHS of Eqn.

(9), since

In other words, K-Medoids can obtain a better bound than the K-Center method (see Figure 1 for a graphical illustration).

We observe superior performance of K-Medoid clustering over K-Center clustering in our experiments as well (see Section 5.2).

We evaluate the node classification performance of our selection method on the Cora, Citeseer, and PubMed network datasets (Yang et al., 2016) .

We further supplement our experiment with an even denser network dataset CoraFull (Bojchevski & Günnemann, 2017) to illustrate the performance differences of the comparing approaches on a large-scale setting.

We evaluate the Macro-F1 of the methods over the full set of nodes.

The sizes of the budgets are fixed for all benchmark datasets.

Specifically, we choose to select 10, 20, 40, 80 and 160 nodes as the budget sizes.

After selecting the nodes, a two-layer GCN 3 , with 16 hidden neurons, is trained as the prediction model.

We use the Adam (Kingma & Ba, 2014) optimizer with a learning rate of 0.01 and weight decay of 5 × 10 −4 .

All the other hyperparameters are kept as in the default setting (β 1 = 0.9, β 2 = 0.999).

To guarantee the convergence of the GCN, the model trained after 200 epochs is used to evaluate the metric on the whole set.

We compared the following methods:

• Random:

Choosing the nodes uniformly from the whole vertex set.

• Degree:

Choosing the nodes with the largest degrees.

Note that this method does not consider the information of node features.

• Uncertainty: Similar to the methods in Joshi et al. (2009), we put the nodes with maxentropy into the pool of instances.

• Coreset (Sener & Savarese, 2017) : This method performs a K-Center clustering over the last hidden representations in the network.

If time allows (on Cora and Citeseer), a robust mixture integer programming method as in Sener & Savarese (2017) (dubbed CoresetMIP) is adopted.

We also apply a time-efficient approximation version (Coreset-greedy) for all of the datasets.

The center nodes are then selected into the pool.

• AGE (Cai et al., 2017) : This method linearly combines three metrics -graph centrality, information density, and uncertainty and select nodes with the highest scores.

• ANRMAB (Gao et al., 2018) : This method enhances AGE by learning the combination weights of metrics through an exponential multi-arm-bandit updating rule.

• FeatProp: This is our method.

We perform a K-Medoids clustering to the propogated features (Eqn.

(7)), where X is the input node features.

In the experiment, we adopts an efficient approximated K-Medoids algorithm which performs K-Means until convergence and select nodes cloesest to centers into the pool.

In our experiments, we start with a small set of nodes (5 nodes) sampled uniformly at random from the dataset as the initial pool.

We run all experiments with 5 different random seeds and report the averaged classification accuracy as the metric.

We plot the accuracy vs the number of labeled points.

For approaches (Uncertainty, Coreset, AGE and ANRMAB) that require the current status/hidden representations from the classification model, a fully-trained model built from the previous budget pool is returned.

For example, if the current budget is 40, the model trained from 20 examples selected by the same AL method is used.

Figure 2 , our method outperforms all the other baseline methods in most of the compared settings.

It is noticeable that AGE and ANRMAB which use uncertainty score as their sub-component can achieve better performances than Uncertainty and are the second best methods in most of the cases.

We also show an averaged Macro-F1 with standard deviation across different number of labeled nodes in Table 3 .

It is interesting to find that our method has the second smallest standard deviation (Degree is deterministic in terms of node selection and the variance only comes from the training process) among all methods.

We conjecture that this is due to the fact that other methods building upon uncertainty may suffer from highly variant model parameters at the beginning phase with very limited labeled nodes.

Efficiency.

We also compare the time expenses between our method and Coreset, which also involves a clustering sub-routine (K-Center), in Table 2 .

It is noticeable that in order to make Coreset more stable, CoresetMIP uses an extreme excess of time comparing to Coreset-greedy in the same setting.

An interesting fact we could observe in Figure 2 is that CoresetMIP and Coreset-greedy do not have too much performance difference on Citeseer, and Coreset-greedy is even better than CoresetMIP on Cora.

This is quite different from the result in image classification tasks with CNNs (Sener & Savarese, 2017) .

This phenomenon distinguishes the difference between graph node classification with traditional classification problems.

We conjecture that this is partially due to the fact that the nodes no longer preserve independent embeddings after the GCN structure, which makes the original analysis of Coreset not applicable.

Ablation study.

It is crucial to select the proper distance function and clustering subroutine for FeatProp (Line 1 and Line 2 in Algorithm 1).

As is discussed in Section 4.3, we test the differences with the variant of using the L2 distance from the final layer of GCN as the distance function and the one by setting K-Medoids choice with a K-Center replacement.

We compare these algorithms in Figure 3 .

As is demonstrated in the figure, the K-Center version (blue line) has a lower accuracy than the original FeatProp approach.

This observation is compatible with our analysis in Section 4.3 as K-Medoids comes with a tighter bound than K-Center in terms of the classification loss.

Furthermore, as final layer representations are very sensitive to the small budget case, we observe that the network representation version (orange line) also generally shows a much deteriorated performance at the beginning stage.

Though FeatProp is tailored for GCNs, we could also test the effectiveness of our algorithm over other GNN frameworks.

Specifically, we compare the methods over a Simplified Graph Convolution (SGC) (Wu et al., 2019 ) and obtain similar observations.

Due to the space limit, we put the detailed results in the appendix.

We study the active learning problem in the node classification task for Graph Convolution Networks (GCNs).

We propose a propagated node feature selection approach (FeatProp) to comply with the specific structure of GCNs and give a theoretical result characterizing the relation between its classification loss and the geometry of the propagated node features.

Our empirical experiments also show that FeatProp outperforms the state-of-the-art AL methods consistently on most benchmark datasets.

Note that FeatProp only focuses on sampling representative points in a meaningful (graph) representation, while uncertainty-based methods select the active nodes from a different criterion guided by labels, how to combine that category of methods with FeatProp in a principled way remains an open and yet interesting problem for us to explore.

We also evaluate the methods using the metric of Micro-F1 in Table 4

C be the prediction for node i under input G, X, and (M) i,c be the c-th element of (M) i (i.e., the prediction for class c).

In order to show Theorem 1, we make the following assumptions: Assumption 1.

We assume A 0 overfits to the training data.

Specifically, we assume the following two conditions: i) A 0 has zero training loss on s 0 ; ii) for any unlabeled data (x i , x j ) with i ∈ s 0 and j ∈ s 0 , we have (A 0 ) i,yj ≤ (A 0 ) j,yj and (A 0 ) i,c ≥ (A 0 ) j,c for all c = y j .

The second condition states that A 0 achieves a high confidence on trained samples and low confidence on unseen samples.

We also assume that the class probabilities are given by a ground truth GCN; i.e., there exists a GCN M * that predicts Pr[Y i = c] on the entire training set.

This is a common assumption in the literature, and (Du et al., 2018) shows that gradient descent provably achieves zero training loss and a precise prediction in polynomial time.

Assumption 2.

We assume l is Lipschitz with constant λ and bounded in [−L, L] .

The loss function is naturally Lipschitz for many common loss functions such as hinge loss, mean squared error, and cross-entropy if the model output is bounded.

This assumption is widely used in DL theory (e.g., Allen-Zhu et al. (2018); Du et al. (2018) ).

Assumption 3.

We assume that there exists a constant α such that the sum of input weights of every neuron is less than α.

Namely, we assume i |(Θ K ) i,j | ≤ α.

This assumption is also present in (Sener & Savarese, 2017) .

We note that one can make i |(Θ K ) i,j | arbitrarily small without changing the network prediction; this is because dividing all input weights by a constant t will also divide the output by a constant t. Assumption 4.

We assume that ReLU function activates with probability 1/2.

This is a common assumption in analyzing the loss surface of neural networks, and is also used in (Choromanska et al., 2015; Kawaguchi, 2016; Xu et al., 2018) .

This assumption also aligns with observations in practice that usually half of all the ReLU neurons can activate.

With these assumptions in place, we are able to prove Theorem 1.

Theorem 1 (restated).

Suppose Assumptions 1-4 hold, and the label vector Y is sampled independently from the distribution y v ∼ η(v) for every v ∈ V .

Then with probability 1 − δ the expected classification loss of A t satisfies

Proof.

Fix y j for j ∈ s 0 and therefore the resulting model A 0 .

Let i ∈ V \ s 0 be any node and j ∈ s 0 .

We have

(11) For the first term we have

The last inequality holds from the Lipschitz continuity of l. Now from Assumption 1, we have

otherwise.

Now taking the expection w.r.t the randomness in ReLU we have

Here E σ represents taking the expectation w.r.t ReLU.

Now for (12) we have

The inequality follows from (13).

Now for the second loss in (11) we use the property that M * computes the ground truth:

We now use the fact that ReLU activates with probability 1/2, and compute the expectation:

Here E σ means that we compute the expectation w.r.t randomness in σ (ReLU) in M * .

The last inequality follows from definition of α, and that l ∈ [−L, L].

Combining the two parts to (11) and let j = argmin (S K X) i − (S K X) j , we obtain

Consider the following process: we first get G, X (fixed data) as input, which induces η(i) for i ∈ [n].

Note that M * gives the ground truth η(i) for every i so distributions η(i) ≡ η X,G (i) are fixed once we obtain G, X 4 .

Then the algorithm A choose the set s 0 to label.

After that, we randomly sample y j ∼ η(j) for j ∈ s 0 and use the labels to train model A 0 .

At last, we randomly sample y i ∼ η(i) and obtain loss l(A 0 |G, X, Y ).

Note that the sampling of all y i for i ∈ V \ s 0 is after we fix the model A 0 , and knowing exact values of y j for j ∈ s 0 does not give any information of y i (since η(i) is only determined by G, X).

Now we use Hoeffding's inequality (Theorem 3) with Z i = l((A 0 ) i , y i ); we have −L ≤ Z i ≤ L by our assumption, and recall that |V \ s 0 | = n − b. Let δ be the RHS of (17), we have that with probability 1 − δ,

Now plug in (14), multiply both sides by (n − b) and rearrange.

We obtain that i∈V \s 0

Now note that since the random draws of y i is completely irrelevant with training of A 0 , we can also sample y i together with y j for j ∈ s 0 after receiving G, X and before the training of A 0 (A does not have access to the labels anyway).

So (16) holds for the random drawings of all y's.

Now divide both sides of (16) by n and use (15), we have

The same proof as Theorem 1 applies for Theorem 2 using the max of distances instead of averaging.

We therefore omit the details here.

We attach the Hoeffding's inequality here for the completeness of our paper.

4 To make a rigorous argument, we get the activation of M * in this step, meaning that we pass through the randomness of σ in M * .

<|TLDR|>

@highlight

This paper introduces a clustering-based active learning algorithm on graphs.
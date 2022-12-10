Graph Neural Networks (GNNs) have received tremendous attention recently due to their power in handling graph data for different downstream tasks across different application domains.

The key of GNN is its graph convolutional filters, and recently various kinds of filters are designed.

However, there still lacks in-depth analysis on (1) Whether there exists a best filter that can perform best on all graph data; (2) Which graph properties will influence the optimal choice of graph filter; (3) How to design appropriate filter adaptive to the graph data.

In this paper, we focus on addressing the above three questions.

We first propose a novel assessment tool to evaluate the effectiveness of graph convolutional filters for a given graph.

Using the assessment tool, we find out that there is no single filter as a `silver bullet' that perform the best on all possible graphs.

In addition, different graph structure properties will influence the optimal graph convolutional filter's design choice.

Based on these findings, we develop Adaptive Filter Graph Neural Network (AFGNN), a simple but powerful model that can adaptively learn task-specific filter.

For a given graph, it leverages graph filter assessment as regularization and learns to combine from a set of base filters.

Experiments on both synthetic and real-world benchmark datasets demonstrate that our proposed model can indeed learn an appropriate filter and perform well on graph tasks.

Graph Neural Networks (GNNs) are a family of powerful tools for representation learning on graph data, which has been drawing more and more attention over the past several years.

GNNs can obtain informative node representations for a graph of arbitrary size and attributes, and has shown great effectiveness in graph-related down-stream applications, such as node classification (Kipf & Welling, 2017) , graph classification (Wu et al., 2019b) , graph matching (Bai et al., 2019) , recommendation systems (Ying et al., 2018) , and knowledge graphs (Schlichtkrull et al., 2018) .

As GNNs have superior performance in graph-related tasks, the question as to what makes GNNs so powerful is naturally raised.

Note that GNNs adopt the concept of the convolution operation into graph domain.

To obtain a representation of a specific node in a GNN, the node aggregates representations of its neighbors with a convolutional filter.

For a task related to graph topology, the convolutional filter can help GNN nodes to get better task-specific representations (Xu et al., 2019) .

Therefore, it is the filter that makes GNNs powerful, and thus the key to designing robust and accurate GNNs is to design proper graph convolutional filters.

Recently, many GNN architectures are proposed (Zhou et al., 2018) with their own graph filter designs.

However, none of them have properly answered the following fundamental questions of GNNs: (1) Is there a best filter that works for all graphs?

(2) If not, what are the properties of graph structure that will influence the performance of graph convolutional filters?

(3) Can we design an algorithm to adaptively find the appropriate filter for a given graph?

In this paper, we focus on addressing the above three questions for semi-supervised node classification task.

Inspired by studies in Linear Discriminant Analysis (LDA), we propose a Graph Filter Discriminant (GFD) Score metric to measure the power of a graph convolutional filter in discriminating node representations of different classes on a specific graph.

We have analyzed all the existing GNNs' filters with this assessment method to answer the three aforementioned questions.

We found that no single filter design can achieve optimal results on all possible graphs.

In other words, for different graph data, we should adopt different graph convolutional filters to achieve optimal performance.

We then experimentally and theoretically analyze how graph structure properties influence the optimal choice of graph convolutional filters.

Based on all of our findings, we propose the Adaptive Filter Graph Neural Network (AF-GNN), which can adaptively learn a proper model for the given graph.

We use the Graph Filter Discriminant Score (GFD) as a an extra loss term to guide the network to learn a good data-specific filter, which is a linear combination of a set of base filters.

We show that the proposed Adaptive Filter can better capture graph topology and separate features on both real-world datasets and synthetic datasets.

We highlight our main contributions as follows:

• We propose an assessment tool: Graph Filter Discriminant Score, to analyze the effectiveness of graph convolutional filters.

Using this tool, we find that no best filter can work for all graphs, the optimal choice of a graph convolutional filter depends on the graph data.

• We propose Adaptive Filter Graph Neural Network that can adaptively learn a proper filter for a specific graph using the GFD Score as guidance.

• We show that the proposed model can find better filters and achieve better performance compared to existing GNNs, on both real-word and newly created benchmark datasets.

Semi-Supervised Node Classification.

Let Y be the class assignment vector for all the nodes in V. C indicates the total number of classes, and Y v ∈ {1, · · · , C} indicates the class that node v belongs to.

The goal of semi-supervised node classification is to learn a mapping function f : V → {1, · · · , C} using the labeled nodes, and predict the class labels for the unlabeled nodes, i.e., Y v = f (v), by leveraging both node features X and graph structure A.

Graph Data Generator.

Intuitively, semi-supervised node classification requires both node features (X) and the graph structure (A) to be correlated to the intrinsic node labels (Y ) to some extent.

To systematically analyze the performance of different GNN filters, we test their performance under different graph data with different properties, i.e., graphs with different X, A, Y .

Intuitively, both graph topology and node features have to be correlated with the node labels, if including both can enhance the performance of node classification task.

To better understand the roles played by each component, we assume the graphical model to generate a graph data is as described in Fig. 1(a) .

To better disclose the relationship between different graph filters and properties of different graph data, we further make assumptions on how X and A are generated when Y is given, as it is difficult to obtain those properties from real-world data.

Therefore, we study simulated data to support a thorough analysis.

We now describe the generation of Y , X|Y , and A|Y respectively.

Generating Y : Each node is randomly assigned with a class label with probability proportional to its class size.

We assume each class c is associated with n c nodes.

Generating X|Y : We assume that node features are sampled from a distribution determined by their corresponding labels.

For example, we can sample node features of class c from a multivariate Gaussian distribution with the parameters conditioned on class c:

.

For another example, we can sample node features of class c from a circular distribution with radius r c and noise noise c conditioned on c.

Generating A|Y : We follow the most classic class-aware graph generator, i.e. stochastic block model (SBM, Holland et al. (1983) ), to generate graph structure conditioned on class labels.

SBM has several simple assumptions that (1) edges are generated via Bernoulli distributions independently and (2) the parameter of the Bernoulli distribution is determined by the classes of the corresponding pair of nodes v i and v j , i.e., A ij |Y i , Y j ∼ Ber(p YiYj ), where p YiYj is a parameter determined by the two corresponding classes.

In a simple two-class case, p = p 11 = p 22 denotes the probability that the linked pair belongs to the same class, while q = p 12 = p 21 denotes the probability that the linked pair belongs to different classes.

We call p+q 2 the "density of graph", which controls the overall connectivity of a graph, and we call |p − q| the "density gap", which controls how closely the graph generated by SBM correlates with labels.

We assume p ≥ q in all the following sections.

Degree Corrected SBM (DCSBM, Karrer & Newman (2011) ), which is a variation of SBM, adds another parameter γ to control the "power-law coefficient" of degree distribution among nodes.

Figure 1 Graph Convolutional Filters.

By examining various GNN designs, we find that most of the GNN operators can fit into a unified framework, i.e., for the l-th layer:

which describes the three-step process that involves: (1) a graph convolutional operation (can also be regarded as feature propagation or feature smoothing) denoted as F(G)H (l−1) , (2) a linear transformation denoted by multiplying W , and (3) a non-linear transformation denoted by σ(·).

Clearly, the graph convolutional operation F(G)H (l−1) is the key step that helps GNNs to improve performance.

Thus, to design a good GNN, a powerful graph convolutional filter F(G) is crucial.

We analyze the effectiveness of graph filters for existing GNNs in the following.

The work of GCN (Kipf & Welling, 2017) first adopts the convolutional operation on graphs and use the filter F(G) =D −1/2ÃD−1/2 .

Here,Ã = A + I is the self-augmented adjacency matrix, andD = diag(d 1 , ...,d n ) is the corresponding degree matrix, whered i = n j=1Ã ij .

Some studies (Wu et al., 2019a; Maehara, 2019 ) use a filter F(G) = (D −1/2ÃD−1/2 ) k that is similar in form to GCN's filter, but with a pre-defined exponent k greater than one.

This would help a node to obtain information from its further neighbors without redundant computation cost.

Several other studies propose to use sampling to speed up GNN training (Chen et al., 2018b; Hamilton et al., 2017; Chen et al., 2018a) ), which can be considered as a sparser version of GCN's filter.

Another set of GNNs consider using a learnable graph convolutional filter.

For example, Xu et al. (2019) and Chiang et al. (2019) both propose to use F(G) = A+ I where is a learnable parameter to augment self-loop skip connection.

Graph Attention Networks (Velickovic et al., 2018) proposes to assign attention weight to different nodes in a neighborhood, which can be considered as a flexible learnable graph convolutional filter.

Their graph filters applied on a feature matrix X can be considered as: ∀i, j,

where N i is the neighborhood of node i, α is a learnable weight vector, and || indicates concatenation.

In this section, we introduce a novel assessment tool for analyzing graph convolutional filters.

We first review the Fisher score, which is widely used in Linear Discriminant Analysis to quantify the linear separability of two sets of features.

With the Fisher score, we propose the Graph Filter Discriminant Score metric to evaluate the graph convolutional filter on how well it can separate nodes in different classes.

Fisher Score.

When coming to non-graph data, the Fisher Score (Fisher, 1936 ) is used to assess the linear separability between two classes.

Given two classes of features X (i) and X (j) , the Fisher Score is defined as the ratio of the variance between the classes (inter-class distance) to the variance within the classes (inner-class distance) under the best linear projection w of the original feature:

where µ (i) and µ (j) denotes the mean vector of X (i) and X (j) respectively, Σ (i) and Σ (j) denotes the variance of X (i) and X (j) respectively, and w denotes the linear projection vector which we can understand as a rotation of the coordinate system, and the max w operation is to find the best direction in which these two class of nodes are most separable.

As the numerator of J indicates interclass distance and the denominator of J indicates inner-class distance a larger value of J indicates higher separability.

Note that for given features, we can directly get the closed form solution of the optimal w, with which Fisher Score could be deformed as:

The detailed proof is provided in appendix A.2.

Graph Filter Discriminant Score.

As mentioned before, the key component that empowers GNNs is the graph convolutional filter F(G).

Intuitively, an effective filter should make the representations of nodes in different classes more separable.

Therefore, we propose to use Fisher Scores of the node representations before and after applying the graph convolutional filter in order to evaluate this filter.

For each pair of classes (i, j), we define their Fisher Difference as

, which is the difference of their Fisher Score of representations after applying the filter F(G) and their Fisher Score of initial representations.

We then define the GFD Score for the filter F(G) with respect to feature matrix X as follows:

where n c is the number of nodes in class c. Note that the GFD Score is a weighted sum of the Fisher Difference for each pair of classes.

Intuitively, the larger the GFD score, the more effective is this corresponding filter to increase the separability of node features.

The Fisher Score can be extended to evaluate non-linearly separable data in addition to linearly separable data.

We claim the rationale of such measure by showing that the graph convolution can actually help non-linearly separable data to be linearly separable if the graph filter is chosen properly for a given graph.

As shown in Figure 2 (a)∼(d), if we use a proper filter, the convolutional operation can transform three circular distributions, which are non-linearly separable, into three linearly separable clusters.

Moreover, as shown in Figure 2 (e)∼(h), even if the original features of different classes are sampled from the same distribution, the proper graph convolutional filter can help to linearly separate the data.

This phenomenon shows that if the graph structure (A) is correlated with the task (Y ), a proper filter alone is powerful enough to empower GNNs with non-linearity, without any non-linear activation.

This phenomenon is also supported by the promising result of SGC (Wu et al., 2019a) , which removes all the non-linear activations in the GCN architecture.

Therefore, we claim that the proposed GFD is a reasonable metric to evaluate a graph filter's effectiveness, and a good graph filter for a given graph should have a higher GFD score on that graph.

With the help of the assessment tool, we now examine existing filters and try to answer the two fundamental questions: (1) Is there a best filter that works for all graphs?

(2) If not, what are the properties of graph data that will influence the performance of graph convolutional filters?

The GFD Score we introduced in the above section can be applied to any filter on any given graph.

From Table 3 , we can see that most of the current GNNs fall into the following filter family: {(Â) k }, where the baseÂ is a normalized adjacency matrix, and k is the order of the filter.

Note that there are some other variants of GNN filters that do not fall into this family, for example, GAT, but the analysis is similar.

Without loss of generality, we focus on analyzing this filter family.

The two main components of this filter family are the normalization strategy (Â) and the order to use (k).

For (h) F isher = 17.8018

Figure 2: Each row corresponds to a graph.

The i-th column corresponds to the feature distribution after applying filter (D −1/2ÃD−1/2 ) i−1 .

Both graphs include three classes of same size and has structure generated by SBM (p = 0.6, q = 0.03).

The first graph's feature follows a circular distribution with radius = 1, 0.9, 0.8 and Gaussian noise = 0.02 for each class.

The second graph's feature follows a circular distribution with radius = 1 and Gaussian noise = 0.02 for all classes.

simplicity, we study the roles of these two components separately, using our assessing tool to show whether there exists an optimal choice of filter for different graph data.

If an optimal choice does not exist, we determine the factors that will influence our choice of component.

Through the analysis, we choose SBM and DCSBM introduced previously to generate the structures of synthetic graphs, and choose multivariate Gaussian distributions to generate features of synthetic graphs.

We focusing on the structure properties that influence the optimal choice of filter.

We enumerate the hyper-parameters to generate graphs with different structure properties, including the power law coefficient (γ) that controls the power law degree distribution of the graph, label ratio ( n1 n2 ) that indicates how balance are the classes of this graph, density ( p+q 2 ) that indicates the overall connectivity of the graph, and density gap (|p − q|) that indicates structural separability of the graph.

As these properties are significant for real-world graphs, our generated synthetic graphs can cover a large range of possible graph properties, and are representative for analyzing different filters.

Analyzing Filter's Normalization Strategy.

We consider three normalization strategies, including row normalization D −1 A, column normalization AD −1 , and symmetric normalization D −1/2 AD −1/2 .

We calculate GFD scores of these three graph filters for graphs generated with different parameters.

As shown in Figure 3 , no single normalization strategy is optimal for all graphs.

Here we give an empirical explanation to this phenomenon.

Note that, with the same order, each filter has the same receptive field, and different normalization strategies affect only on how to assign weights to the neighboring nodes.

The row normalization strategy simply takes the mean of features of the node's neighbors.

Clearly, this would help to keep every node's new representations in the same range.

On the contrary, column normalization and symmetric normalization, might keep a larger representation for higher-degree nodes.

Using a column-normalized adjacency matrix as the base of the graph convolutional filter is similar to the PageRank algorithm.

While a node propagates its features to neighbors, this normalization strategy takes its degree into account.

Thus, column normalization can be helpful when the when node degree plays an important role for classification.

Symmetric normalization combines the properties from both the row normalization and the column normalization.

Even in the case where row normalization and column normalization do not perform well, symmetric normalization still leads to promising performance.

We now examine which graph properties influence our choice of the optimal normalization strategy, which may vary per graph.

We find that power law coefficient γ is an important factor that influences the choice of normalization.

As shown in Figure 3 , when power-law coefficient γ decreases (graph's extent of power-law grows), row normalization tends to have better performance.

This is because row normalization helps to keep node representations in the same range, so that large representations of high degree nodes can be avoided.

Therefore, it prevents nodes with similar degrees getting closer to each other and messing the classification tasks where node degrees are not important.

We also find that the label ratio ( n1 n2 ) matters.

As shown in Figure 3 , when the size of each class becomes more imbalanced, column normalization tends to work better.

This is because column normalization better leverages degree property during representation smoothing, as nodes in largesize classes tend to have larger representation since they are more likely to have higher degree.

This can help nodes within different classes become more separable.

We then analyze what would be the best order for filters.

With a highorder filter, a node can obtain information from its further neighbors, and thus the amount of information it receives during the feature propagation increases.

But do we always need more information under any circumstances?

The answer is no. Still, we find that, for different graphs, the order that results in the best performance would be different 1 .

Since there is no best filter order for all the cases, we explore the factors that can influence the choice of order.

We find that the density of graph and density gap between two classes have a big impact.

As shown in Figure 4 , when the density or density gap increases, the filter with higher order tends to be a better choice.

We provide an intuitive explanation for this phenomenon as follows.

Note that the feature propagation scheme is based on the assumption that nodes in the same class have a closer connection.

On one hand, when the density increases, the connections between nodes are closer.

Therefore, high-order filters can help gather richer information and thus reduce the variance of the obtained new node representations, so it helps nodes in the same class get smoother representations.

On the other hand, when the density gap decreases, for a node, the size of neighbors within the same class becomes similar to the size of neighbors within different classes.

Thus conducting high-order graph convolution operations will mix the representations of all nodes regardless of classes, which will make node classification more difficult.

Based on previous analysis, we now answer the last question: Can we design an algorithm to adaptively find the appropriate filter for a given graph?

We develop a simple but powerful model, the Adaptive Filter Graph Neural Network (AFGNN).

For a given graph, AFGNNs can learn to combine an effective filter from a set of filter bases, guided by GDF Scores.

Adaptive Filter Graph Neural Network (AFGNN).

For simplicity, we only consider finding the optimal filter for one family of graph convolutional filters:

where k is the maximum order.

Note that, we also include the identity matrix, which serves as a skip-connection, to maintain the original feature representation.

Based on our previous analysis, for graphs that are not closely correlated to tasks (i.e., small density gap in SBM), the identity matrix will outperform all the other convolutional filters.

We denote the above 3k + 1 filters as

, the l-th layer of AFGNN is defined as a learnable linear combination of these filter bases:

where ψ (l) is the learnable vector to combine base filters and α (l) is its softmax-normalized version.

Comparing to GNNs with fixed filters such as GCN and SGC, our proposed AFGNN can adaptively learn a filter based on any given graph.

As we have shown that no single fixed filter can perform optimally for all graphs, we conclude that an adaptive filter has more capacity to learn better representations.

Comparing to other GNNs with learnable filters such as GAT, AFGNN is computationally cheaper and achieves similar or better performance on most existing benchmarks and our synthetic datasets (as shown in the experiment section).

We leave expanding the base filter family and adding more complex filters such as GAT into our filter bases as future work.

Training Loss.

To train this AFGNN model, we can simply optimize the whole model via any downstream tasks, i.e., node classification.

However, as most of the semi-supervised node classification datasets only contain limited training data, the enlarged filter space will make the model prone to over-fitting.

Thus, we decide to add the GFD Score as an loss term into the training loss to guide the optimization of filter weights, i.e., ψ (l) and to prevent overfitting:

where L CE is the cross-entropy loss of the node classification, and L GF D is defined as the cumulative negation of GFD Score for the learned adaptive filter F AF GN N (G) (l) at each layer with respect to its input representation H (l−1) .

During the training, we minimize L to learn the proper model.

With a different choice of the weight λ for GFD loss, we can categorize our model into: AFGNN 0 : With λ = 0, the model is only trained by L CE , which might be prone to over-fitting when data is not sufficient.

AFGNN 1 : With λ = 1, the model is trained by both L CE and L GF D simultaneously.

AFGNN ∞ : This case is not exactly λ = ∞, and the training process is different from the other two cases.

We implement the training iteratively: we optimize the combination of base filters by training only with GFD loss L GF D , then we optimize the linear transofrmation parameter W l s with classification loss L CE .

Note that the input feature H (0) = X is invariant, we can pre-train the optimal filter for first layer and fix it.

Dataset We first evaluate AFGNN on three widely used benchmark datasets: Cora, Citeseer, and Pubmed (Sen et al., 2008) .

As these datasets are not sensitive enough to differentiate the models, we need more powerful datasets that can evaluate the pros and cons of each model.

Based on our findings in section 3.2, we generate two synthetic benchmarks called SmallGap and SmallRatio.

SmallGap corresponds to the case in which the density gap of the graph is close to 1.

This indicates that the graph structure does not correlate much to the task, thus I would be the best filter in this case.

SmallRatio corresponds to the case in which the label ratio is small, i.e. the size of one class is clearly smaller than the other, and column normalization AD −1 is the best normalization 2 .

Baselines and Settings.

We compare against 5 baselines, including GCN, GIN, SGC, GFNN, and GAT.

To make fair comparisons, for all the baseline GNNs, we set the number of layers (or orders) to be 2, and tune the parameters including learning rate, weight decay, and number of epochs 3 .

For all the benchmark datasets, we follow the data split convention 2 .

For the synthetic dataset, we conduct 5-fold cross-validation, randomly split the nodes into 5 groups of the same size, take one group as the training set, one as the validation set and the remaining three as the test set.

Each time we pick Classification Performance.

As is shown in Table 1 , our proposed AFGNN ∞ model can consistently achieve competitive test accuracy.

On Pubmed, SmallGap, and SmallRatio, AFGNN ∞ can achieve the best results among all the baseline models.

On Cora and Citeseer, though GAT outperforms our proposed model a little bit, however, as shown in Table 6 ,7, GAT takes a longer time to train and converge, and has more memory cost as well.

Also, when the given graph is simple, GAT would suffer unavoidable overfitting problem.

We further compare our AFGNN 0 , AFGNN 1 , AFGNN ∞ to examine the role of GFD loss.

The AFGNN 0 performs quite poorly on all the datasets, implying that the larger search space of the filter without GFD loss is prone to over-fitting, while AFGNN 1 and AFGNN ∞ perform much better.

Also, AGFNN ∞ has superior performance compared to AFGNN 1 , which indicates the GFD Score is indeed a very powerful assessment tool.

Graph Filter Discriminant Analysis.

We are also interested to see whether the proposed method can indeed learn the best combination of filters from the base filter family.

To do so, we calculate the GFD Score of the first-layer filter learned by AFGNN 0 , AFGNN 1 , AFGNN ∞ and the seven base filters on the test set for each dataset.

For the AFGNN models, the filter is trained with the training set for each dataset.

Table 2 4 and Figure 5 show the results, we can see that our proposed method can indeed learn a combined filter on all the datasets.

Specifically, in all the benchmark datasets, the best base filter is (D −1Ã ) 2 , and our proposed adaptive filter not only picks out the best base filter but also learns a better combination.

For the two synthetic datasets, where I and (ÃD −1 ) 2 are the best filters, our algorithm can also learn to pick them out.

We thereby conclude that the proposed GFD loss can help find an appropriate filter for a given dataset.

Understanding the graph convolutional filters in GNNs is very important, as it can help to determine whether a GNN will work on a given graph, and can provide important guidance for GNN design.

In our paper, we focus on the semi-supervised node classification task.

We first propose the Graph Filter Discriminant Score as an assessment tool for graph convolutional filter evaluation, and then apply this GFD Score to analyze a family of existing filters as a case study.

Using this tool, we learn that no single fixed filter can produce optimal results on all graphs.

We then develop a simple but powerful GNN model: Adapative Filter Graph Neural Network, which can learn to combine a family of filters and obtain a task-specific powerful filter.

We also propose to add the negative GFD Score as an extra component to the objective function, it can act as a guidance for the model to learn a more effective filter.

Experiments show that our approach outperforms many existing GNNs on both benchmark and synthetic graphs.

Graph Convolutional Filters (Velickovic et al., 2018 ) F(G) = Q, where Q is parametric attention function of X and A Table 3 summarized the graph filters for existing GNNs.

Proof According to the conclusions in linear discriminant analysis, the maximum separation occurs when w ∝ (

Note that, when we want to apply this fisher linear discriminant score in our problem, the linear transformation part in our classifier (and also the linear transformation part in GNN) will help to find the best w. Thus, we can directly plug the optimum solution w * = c(

) into this formula, here c is a scalar.

Then, we'll have:

Thus we completed the proof.

A.3.1 EXAMPLES OF "NO BEST NORMALIZATION STRATEGY FOR ALL" Figure 6 provides two examples to show there is no best normalization strategy for all graphs.

For both examples, we fix the order of filter to be 2.

The first row shows a case in which row normalization is better than the other two.

The corresponding graph contains 2 classes of nodes with size 500.

The graph structure is generated by DCSBM with p = 0.3, q = 0.05, power law coefficient γ = −0.9.

The features for two classes satisfy multivariate distribution with an identity co-variance matrix, and with mean (0.2,0.2) and (0,0) respectively.

In this example, we can clearly see that with other two normalization strategy, some high-degree hubs show up in the upper right corner from both class, which is harmful for classification.

We generate this example to illustrate the benefit of row normalization because row normalization would be very helpful for a graph with power law degree distribution, which contains some nodes with unusually large degree (those nodes are called hubs), since it can help avoid those hubs obtaining larger representations and thus be mis-classified.

The second row shows a case in which column normalization is better than the other two.

The corresponding graph contains 2 classes of nodes with size 900 and 100 respectively.

The graph structure is generated by SBM with p = 0.3, q = 0.2.

The features for two classes satisfy multivariate distribution with an identity co-variance matrix, and with mean (-0.2,-0.2) and (0.2,0.2) respectively.

We generate this example to illustrate the benefit of column normalization because under this case, we should consider taking more degree information into consideration.

Therefore, column normalization would be more helpful.

Figure 7 provides two examples to show there is no best order for all graphs.

For both examples, we fix the normalization strategy to be row normalization, and varies order to be 2, 4, 6.

The first row shows a case in which small order is better than the large ones.

The corresponding graph contains 2 classes of nodes with same size 500.

The graph structure is generated by SBM with p = 0.215, q = 0.2.

The features for two classes satisfy multivariate distribution with an identity co-variance matrix, and with mean (0.5,0.5) and (0,0) respectively.

The second row shows a case in which large order is better than the smaller ones.

The corresponding graph contains 2 classes of nodes with same size 500.

The graph structure is generated by SBM with p = 0.75, q = 0.6.

The features for two classes satisfy multivariate distribution with an identity co-variance matrix, and with mean (0.5,0.5) and (0,0) respectively.

For the curves indicating how powerlaw coefficient influence the choice of normalization in Figure  3 , we generate the corresponding graphs structure by DCSBM with fixed p = 0.3, q = 0.2 and varies the powerlaw coefficient from -0.3 to 0.

The graph contains two classes of nodes, and is of size 400 and 600 for each class respectively.

The feature for each class satisfies multivariate normal distribution with identity co-variance matrix, and with mean (0,0) and (0.2,0.2).

For the curves indicating how label ratio influence the choice of normalization in Figure 3 , we generate the corresponding graphs structure by SBM with fixed p = 0.3, q = 0.1 and varies the label ratio.

The graph contains a total number of 1000 nodes in two classes.

The feature for each class satisfies multivariate normal distribution with identity co-variance matrix, and with mean (0,0) and (0.5,0.5).

For the curves indicating how density influence the choice of normalization in Figure 4 , we generate the corresponding graphs structure by SBM with fixed density gap p/q = 1.5 and varies the density by varying q. The graph contains two classes of node of size 500.

The feature for each class satisfies multivariate normal distribution with identity co-variance matrix, and with mean (0,0) and (0.5,0.5).

For the curves indicating how density gap influence the choice of normalization in Figure 4 , we generate the corresponding graphs structure by SBM with fixed density p + q = 0.6 and varies the density gap.

The graph contains two classes of node of size 500.

The feature for each class satisfies multivariate normal distribution with identity co-variance matrix, and with mean (-0.2,-0.2) and (0.2,0.2).

The following flowchart (Figure 8 ) describes the process of how a one-layer AFGNN tackle node classification task.

We reduced the dimension of feature by t-SNE (Maaten & Hinton (2008)).

We annotate the filter and the GFD Score in title of each subfigure.

Note that, identity also corresponds to the initial feature.

The figure is the feature representation obtained after conduct graph convolution operation once with the corresponding filter.

We use three benchmark dataset: Cora, Citeseer and Pubmed for the node classification task.

Their statictics are in table4.

Beside number of nodes, edges, classes, the dimension of feature, and the data split strategy, we also show the class ratio variance, which can indicates if this dataset is imbalance or not, density gap, which indicates the dependency of structure and labels, and density, which indicates the overall connectivity of a graph.

We provide the degree distribution in Figure 10 , and we can clearly find that these benchmark datasets has power law degree distribution.

Nodes  2708  3327  19717  Edges  5429  4732  44338  Classes  7  6  3  Feature  1433  3703  500  Train  140  120  60  Validation  500  500  500  Test  1000  1000 We tune the number of epochs based on convergence performance.

For learning rate and weight decay, we follows the parameter setting provides by the corresponding public implementations unless we find better parameters.

The tuned parameters can be found in our code resource.

We report the accuracy of node classification task for baseline models on Cora, Citeseer, and Pubmed provided by corresponding literature.

Since GIN (Xu et al., 2019) is not originally evaluated on node classification task, we do not have the reported number here.

The results is in Table 5 .

A.9 TIME AND MEMORY COST COMPARISON Both our AFGNN model and GAT model have a learnable filter.

We provide time and memory complexity comparison on benchmark datasets here to compare these two models.

As shown in Table 6 , GAT's time cost is at least three times of AFGNN's time cost on both Cora and Citeseer dataset.

As shown in Table 8 : Performance on OAG SmallRatio Dataset because it requires too much memory cost and is not able to run on GPU.

Therefore, AFGNN needs less time and memory cost than GAT.

We generate a real-world dataset with imbalanced classes to justify hard cases may exist in realworld datasets.

We download a large scale academic graph called Open Academic Graph (OAG), and choose two fields that have a large disparity in the number of papers: (1) "History of ideas", which consists of 1041 papers; (2) "Public history", which consists of 150 papers.

Obviously this two classes are imbalanced, and fall in the large label ratio gap problem.

We run supplementary experiment on the generated OAG graph, the experiment setting remains the same as experiment settings for synthetic graphs.

Table 8 shows the experiment results.

To evaluate the models, we compare their F1 score for each class, the weighted average F1 score (micro F1), and the average F1 score (macro F1).

Our AFGNN ∞ model shows superior performance on this dataset.

<|TLDR|>

@highlight

Propose an assessment framework to analyze and learn graph convolutional filter
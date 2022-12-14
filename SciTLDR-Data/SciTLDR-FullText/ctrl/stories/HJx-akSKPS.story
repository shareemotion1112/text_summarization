In this paper, we study a new graph learning problem: learning to count subgraph isomorphisms.

Although the learning based approach is inexact, we are able to generalize to count large patterns and data graphs in polynomial time compared to the exponential time of the original NP-complete problem.

Different from other traditional graph learning problems such as node classification and link prediction, subgraph isomorphism counting requires more global inference to oversee the whole graph.

To tackle this problem, we propose a dynamic intermedium attention memory network (DIAMNet) which augments different representation learning architectures and iteratively attends pattern and target data graphs to memorize different subgraph isomorphisms for the global counting.

We develop both small graphs (<= 1,024 subgraph isomorphisms in each) and large graphs (<= 4,096 subgraph isomorphisms in each) sets to evaluate different models.

Experimental results show that learning based subgraph isomorphism counting can help reduce the time complexity with acceptable accuracy.

Our DIAMNet can further improve existing representation learning models for this more global problem.

Graphs are general data structures widely used in many applications, including social network analysis, molecular structure analysis, natural language processing and knowledge graph modeling, etc.

Learning with graphs has recently drawn much attention as neural network approaches to representation learning have been proven to be effective for complex data structures (Niepert et al., 2016; Kipf & Welling, 2017; Hamilton et al., 2017b; Schlichtkrull et al., 2018; Velickovic et al., 2018; Xu et al., 2019) .

Most of existing graph representation learning algorithms focus on problems such as node classification, linking prediction, community detection, etc. (Hamilton et al., 2017a) .

These applications are of more local decisions for which a learning algorithm can usually make inferences by inspecting the local structure of a graph.

For example, for the node classification problem, after several levels of neighborhood aggregation, the node representation may be able to incorporate sufficient higher-order neighborhood information to discriminate different classes (Xu et al., 2019) .

In this paper, we study a more global learning problem: learning to count subgraph isomorphisms (counting examples are shown as Figure 1 ).

Although subgraph isomorphism is the key to solve graph representation learning based applications (Xu et al., 2019) , tasks of identifying or counting subgraph isomorphisms themselves are also significant and may support broad applications, such as bioinformatics (Milo et al., 2002; Alon et al., 2008) , chemoinformatics (Huan et al., 2003) , and online social network analysis (Kuramochi & Karypis, 2004) .

For example, in a social network, we can solve search queries like "groups of people who like X and visited Y-city/state."

In a knowledge graph, we can answer questions like "how many languages are there in Africa speaking by people living near the banks of the Nile River?"

Many pattern mining algorithms or graph database indexing based approaches have been proposed to tackle subgraph isomorphism problems (Ullmann, 1976; Cordella et al., 2004; He & Singh, 2008; Han et al., 2013; Carletti et al., 2018) .

However, these approaches cannot be applied to large-scale graphs because of the exponential time complexity.

Thanks to the powerful graph representation learning models which can effectively capture local structural information, we can use a learning algorithm to learn how to count subgraph isomorphisms from a lot of examples.

Then the algorithm can scan a large graph and memorize all necessary local information based on a query pattern graph.

In this case, although learning based approaches can be inexact, we can roughly estimate the range of the number of subgraph isomorphism.

This can already help many applications that do not require exact match or need a more efficient pre- processing step.

To this end, in addition to trying different representation learning architectures, we develop a dynamic intermedium attention memory network (DIAMNet) to iteratively attend the query pattern and the target data graph to memorize different local subgraph isomorphisms for global counting.

To evaluate the learning effectiveness and efficiency, we develop a small (??? 1,024 subgraph isomorphisms in each graph) and a large (??? 4,096 subgraph isomorphisms in each graph) dataset and evaluate different neural network architectures.

Our main contributions are as follows.

??? To our best knowledge, this is the first work to model the subgraph isomorphism counting problem as a learning problem, for which both the training and prediction time complexities are polynomial.

??? We exploit the representation power of different deep neural network architectures in an end-toend learning framework.

In particular, we provide universal encoding methods for both sequence models and graph models, and upon them we introduce a dynamic intermedium attention memory network to address the more global inference problem for counting.

??? We conduct extensive experiments on developed datasets which demonstrate that our framework can achieve good results on both relatively large graphs and large patterns compared to existing studies.

Subgraph Isomophism Problems.

Given a pattern graph and a data graph, the subgraph isomorphism search aims to find all occurrences of the pattern in the data graph with bijection mapping functions.

Subgraph isomorphism is an NP-complete problem among different types of graph matching problems (monomorphism, isomorphism, and subgraph isomorphism).

Most subgraph isomorphism algorithms are based on backtracking.

They first obtain a series of candidate vertices and update a mapping table, then recursively revoke their own subgraph searching functions to match one vertex or one edge at a time.

Ullmann's algorithm (Ullmann, 1976) , VF2 (Cordella et al., 2004) , and GraphQL (He & Singh, 2008) belong to this type of algorithms.

However, it is still hard to perform search when either the pattern or the data graph grows since the search space grows exponentially as well.

Some other algorithms are designed based on graph-index, such as gIndex (Yan et al., 2004) , which can be used as filters to prune out many unnecessary graphs.

However, graph-index based algorithms have a problem that the time and space in indexing also increase exponentially with the growth of the graphs (Sun et al., 2012) .

TurboISO (Han et al., 2013) and VF3 (Carletti et al., 2018) add some weak rules to find candidate subregions and then call the recursive match procedure on subregions.

These weak rules can significantly reduce the searching space in most cases.

Graph Representation Learning.

Graph (or network) representation learning can be directly learning an embedding vector of each graph node (Perozzi et al., 2014; Tang et al., 2015; Grover & Leskovec, 2016) .

This approach is not easy to generalize to unseen nodes.

On the other hand, graph neural networks (GNNs) (Battaglia et al., 2018) provide a solution to representation learning for nodes which can be generalized to new graphs and unseen nodes.

Many graph neural networks have been proposed since 2005 but rapidly developed in recent years.

Most of them focus on generalizing the idea of convolutional neural networks for general graph data structures (Niepert et al., 2016; Kipf & Welling, 2017; Hamilton et al., 2017b; Velickovic et al., 2018) or relational graph structures with multiple types of relations (Schlichtkrull et al., 2018) .

More recently, Xu et al. (2019) propose a graph isomorphism network (GIN) and show its discriminative power.

Others use the idea of recurrent neural networks (RNNs) which are originally proposed to deal with sequence data to work with graph data (Li et al., 2016; You et al., 2018) .

Interestingly, with external memory, sequence models can work well on complicated tasks such as language modeling (Sukhbaatar et al., 2015; Kumar et al., 2016) and shortest path finding on graphs (Graves et al., 2016) .

There is another branch of research called graph kernels (Vishwanathan et al., 2010; Shervashidze et al., 2011; Yanardag & Vishwanathan, 2015; Togninalli et al., 2019; Chen et al., 2019) which also convert graph isomorphism to a similarity learning problem.

However, they usually work on small graphs and do not focus on subgraph isomorphism identification or counting problems.

We begin by introducing the subgraph isomophism problems and then provide the general idea of our work by analyzing the time complexities of the problems.

Traditionally, the subgraph isomorphism problem is defined between two simple graphs or two directed simple graphs, which is an NP-complete problem.

We generalize the problem to a counting problem over directed heterogeneous multigraphs, whose decision problem is still NP-complete.

A graph or a pattern is defined as G = (V, E, X , Y) where V is the set of vertices.

E ??? V ?? V is the set of edges, X is a label function that maps a vertex to a vertex label, and Y is a label function that maps an edge to a set of edge labels.

We use an edge with a set of edge labels to represent multiedges with the same source and the same target for clarity.

That is, there are no two edges in a graph such that they have the same source, same target, and the same edge label.

To simplify the statement, we assume

In this paper, we discuss isomorphic mappings that preserve graph topology, vertex labels and edge labels, but not vertex ids.

More precisely, a pattern

Furthermore, G P being isomorphic to a graph G G is denoted as G P G G and the function f is named as an isomorphism.

The subgraph isomorphism counting problem is defined as to find the number of all different subgraph isomorphisms between a pattern graph G P and a graph G G .

Examples are shown in Figure 1 .

Intuitively, we need to compute O(P erm(|V G |, |V P |) ?? d |V P | ) to solve the subgraph isomorphism counting problem by enumeration, where P erm(n, k) = n! (n???k)! , |V G | is the number of graph nodes, |V P | is the number of pattern nodes, d is the maximum degree.

The first subgraph isomorphism algorithm, Ullmann's algorithm (Ullmann, 1976) , reduces the seaching time to

.

If the pattern and the graph are both small, the time is acceptable because both of two factors are not horrendously large.

However, since the computational cost grows exponentially, it is impossible to count as either the graph size or the pattern size increases.

If we use neural networks to learn distributed representations for V G and V P or E G and E P , we can reduce the complexity to O(

via source attention and self-attention.

Assuming that we can further learn a much higher level abstraction without loss of representation power for G P , then the computational cost can be further reduced to

However, the complexity of the latter framework is still not acceptable when querying over large graphs.

If we do not consider self-attention, the computational cost will be O(|V G |) or O(|E G |), but missing the self-attention will hurt the performance.

In this work, we hope to use attention mechanism and additional memory networks to further reduce the complexity compared with

while keeping the performance acceptable on the counting problem.

A graph (or a pattern) can be represented as a sequence of edges or a series of adjacent matrices and vertex features.

For sequence inputs we can use CNNs (Kim, 2014) , RNNs such as GRU (Cho et al., 2014) , or Transformer-XL (Dai et al., 2019) to extract high-level features.

While if the inputs are modeled as series of adjacent matrices and vertex features, we can use RGCN (Schlichtkrull et al., 2018) to learn vertex representations with message passing from neighborhoods.

After obtaining the pattern representation and the graph representation, we feed them into an interaction module to extract the correlated features from each side.

Then we feed the output context of the interaction module into a fully-connected layer to make predictions.

A general framework is shown in Figure 2 and the difference between sequence encoding and graph encoding is shown in Figure 3 .

In sequence models, the minimal element of a graph (or a pattern) is an edge.

By definition, at least three attributes are required to identify an edge e, which are the source vertex id u, the target vertex id v, and its edge label y ??? Y(e).

We further add two attributes of vertices' labels to form a 5-tuple (u, v, X (u), y, X (v)) to represent an edge e, where X (u) is the source vertex label and X (v) is the target vertex label.

A list of 5-tuple is referred as a code.

We follow the order defined in gSpan (Yan & Han, 2002) to compare pairs of code lexicographically; the detailed definition is given in Appendix A.

The minimum code is the code with the minimum lexicographic order with the same elements.

Finally, each graph can be represented by the corresponding minimum code, and vice versa.

Given that a graph is represented as a minimum code, or a list of 5-tuples, the next encoding step is to encode each 5-tuple into a vector.

Assuming that we know the max values of |V|, |X |, |Y| in a dataset in advance, we can encode each vertex id v, vertex label x, and edge label y into Bnary digits, where B is the base and each digit d ??? {0, 1, ?? ?? ?? , B ??? 1}.

It is easy to replace each digit with a one-hot vector so that each 5-tuple can be vectorized as a multi-hot vector which is the concatenation of one-hot vectors.

The length of the multi-hot vector of a 5-tuple is

Then we can easily calculate the graph dimension d g and the pattern dimension d p .

Furthermore, the minimum code can be encoded into a multi-hot matrix, G ??? R |E G |??dg for a graph G G or P ??? R |E P |??dp for a pattern G P according to this encoding method.

This encoding method can be extended when we have larger values of |V|, |X |, |Y|.

A larger value, e.g., |V|, only increases the length of one-hot vectors corresponding to its field.

Therefore, we can regard new digits as the same number of zeros in previous data.

As long as we process previous one-hot vectors carefully to keep these new dimensions from modifying the original distributed representations, we can also extend these multi-hot vectors without affecting previous models.

A simple but effective way is to initialize additional new weights related to new dimensions as zeros.

Given the encoding method in Section 4.1.1, we can simply embed graphs as multi-hot matrices.

Then we can use general strategies of sequence modeling to learn dependencies among edges in graphs.

Convolutional Neural Networks (CNNs) have been proved to be effective in sequence modeling (Kim, 2014) .

In our experiments, we apply multiple layers of the convolution operation to obtain a sequence of high-level features.

Recurrent Neural Networks (RNNs), such as GRU (Cho et al., 2014) , are widely used in many sequence modeling tasks.

Transformer-XL (TXL) (Dai et al., 2019 ) is a variant of the Transformer architecture (Vaswani et al., 2017) and enables learning long dependencies beyond a fixed length without disrupting temporal coherence.

Unlike the original autoregressive settings, in our model the Transformer-XL encoder works as a feature extractor, in which the attention mechanism has a full, unmasked scope over the whole sequence.

However, its computational cost grows quadratically with the size of inputs, so the tradeoff between performance and efficiency would be considered.

In graph models, each vertex has a feature vector and edges are used to pass information from its source to its sink.

GNNs do not need vertex ids and edge ids explicitly because the adjacency information is included in a adjacent matrix.

As explained in Section 4.1.1, we can vectorize vertex labels into multi-hot vectors as vertex features.

In a simple graph or a simple directed graph, the adjacent information can be stored in a sparse matrix to reduce the memory usage and improve the computation speed.

As for heterogeneous graphs, behaviors of edges should depend on edge labels.

RGCNs have relation-specific transformations so that each edge label and topological information are mixed into the message to the sink.

We follow this method and use basis-decomposition for parameter sharing (Schlichtkrull et al., 2018) .

Relational Graph Convolutional Networks (RGCNs) (Schlichtkrull et al., 2018) are developed specifically to handle multi-relational data in realistic knowledge bases.

Each relation corresponds to a transformation matrix to transform relation-specific information from a neighbor to the center vertex.

Two decomposition methods are proposed to address the rapid growth in the number of parameters with the number of relations: basis-decomposition and block-diagonal-decomposition.

We use the first method, which is equivalent to the MLPs in GIN (Xu et al., 2019) .

The original RGCN uses the mean aggregator, but Xu et al. (2019) find that the sum-based GNNs can capture graph structures better.

We implement both and named them as RGCN and RGCN-SUM respectively.

Figure 4: Illustration of dynamic intermedium attention memory network (DIAMNet).

?? 1 represents Eqs. (1) and (2), ?? 2 represents Eqs. (4) and (5), and two types of gates are Eqs. (3) and (6).

After obtaining a graph representation?? and a pattern representationP from a sequence model or a graph model where their column vectors are d-dimensional, we feed them as inputs of interaction layers to extract the correlated context between the pattern and the graph.

A naive idea is to use attention modules (Bahdanau et al., 2015) to model interactions between these two representations and interactions over the graph itself.

However, this method is not practical due to its complexity,

To address the problem of high computational cost in the attention mechanism, we propose the Dynamic Intermedium Attention Memory Network (DIAMNet), using an external memory as an intermedium to attend both the pattern and the graph in order.

To make sure that the memory has the knowledge of the pattern while attending the graph and vice-versa, this dynamic memory is designed as a gated recurrent network as shown in Figure 4 .

Assuming that the memory size is M and we have T recurrent steps, the time complexity is decreased into

, which means the method can be easily applied to large-scale graphs.

The external memory is divided into M blocks {m 1 , ..., m M }, where m j ??? R d .

At each time step t, {m j } is updated by the pattern and the graph in order via multi-head attention mechanism (Vaswani et al., 2017) .

Specifically, the update equations of our DIAMNet are given by:

Here M ultiHead is the attention method described in (Vaswani et al., 2017) , ?? represents the logistic sigmoid function, s j is the intermediate state of the j th block of memory that summarizes information from the pattern, and s for information from both the pattern and the graph.

z j and z j are two gates designed to control the updates on the states in the j th block.

U P , V P , U G , V G ??? R d??d are trainable parameters.

In this section, we report our major experimental results.

More results can be found in the Appendix.

In order to train and evaluate our neural models for the subgraph isomorphism counting problem, we need to generate enough graph-pattern data.

As there's no special constraint on the pattern, the pattern generator may produce any connected multigraph without identical edges, i.e., parallel edges with identical label.

In contrast, the ground truth number of subgraph isomorphisms must be tractable in our synthetic graph data.

Therefore, our graph generator first generates multiple disconnected components, possibly with some subgraph isomorphisms.

We use the idea of neighborhood equivalence class (NEC) in TurboISO (Han et al., 2013) to control the necessary conditions of a subgraph isomorphism in the graph generation process.

The detailed algorithms are shown in Appendix B. Then the generator merges these components into a larger graph and ensures that there is no more subgraph isomorphism generated in the merge process.

The subgraph isomorphism search can be done during these components subgraphs in parallel.

Using the pattern generator and the graph generator above, we can generate many patterns and graphs for neural models.

We are interested in follow research questions: whether sequence models and graph convolutional networks can perform well given limited data, whether their running time is acceptable, and whether memory can help models make better predictions even faced with a NPcomplete problem.

To evaluate different neural architectures and different prediction networks, we generate two datasets in different graph scales and the statistics are reported in Table 1 .

There are 187 unique patterns in whole pairs, where 75 patterns belong to the small dataset, 122 patterns belong to the large dataset.

Target data graphs are not required similar so they are generated randomly.

The generation details are reported in Appendix C.

Instead of directly feeding multi-hot encoding vectors into representation modules, we use two simple linear layers separately to transform graph multi-hot vectors and pattern multi-hot vectors to lower-dimensional, distributed ones.

To improve the efficiency, we also add a filtering layer to filter out irrelevant parts before all representation modules.

The details of this filter layer is shown in Section D.1.

In our experiments, we implemented five different representation models: (1) CNN is a 3-layer convolutional layers followed by max-pooling layers.

The convolutional kernels are 2,3,4 respectively and strides are 1.

The pooling kernels are 2,3,4 and strides are 1.

(2) RNN is a simple 3-layer GRU model.

(3) TXL is a 6-layer Transformer encoder with additional memory.

(4) RGCN is a 3-layer RGCN with the basis decomposition.

We follow the same setting in that ordinal paper to use meanpooling in the message propagation part.

(5) RGCN-SUM is a modification of RGCN to replace the mean-pooling with sum-pooling.

After getting a graph representation?? and a pattern representationP from the representation learning modules, we feed them into the following different types of interaction layers for comparison.

SumPool: A simple sum-pooling is applied for?? andP to obtain?? andp, and the model sends Concate(??,p,?? ???p,?? p) with the graph size and the pattern size information into the next fully connected (FC) layers.

MeanPool: Similar settings as SumPool, but to replace the pooling method with mean-pooling.

MaxPool: Similar settings as SumPool, but to replace the pooling method with max-pooling.

AttnPool: We want to use attention modules without much computational cost so the self-attention is not acceptable.

We simplify the attention by first applying a pooling mechanism for the pattern graph and then use the pooled vector to perform attention over the data graph rather than simply perform pooling over it.

Other settings are similar with pooling methods.

The detailed information is provided in Appendix D.2.

We only report results of mean-pooling based attention, because it is the best of the three variants.

We compare the performance and efficiency of our DIAMNet proposed in Section 4.3 with above interaction networks.

The initialization strategy we used is shown in Appendix D.3.

And we feed the whole memory with size information into the next FC layers.

For fair comparison, we set embedding dimensions, dimensions of all representation models, and the numbers of filters all 64.

The segment size and memory size in TXL are also 64 due to the computation complexity.

The length of memory is fixed to 4 and the number of recurrent steps is fixed to 3 in the DIAMNet for both small and large datasets.

We use the mean squared error (MSE) to train models and evaluate the validation set to choose best models.

The optimizer is Adam with learning rate 0.001.

L2 penalty is added and the coefficient is set as 0.001.

To avoid gradient explosion and overfitting, we add gradient clipping and dropout with a dropout rate 0.2.

We use Leaky ReLU as activation functions in all modules.

Due to the limited number of patterns, the representation module for patterns are easy to overfit.

Therefore, we use the same module with shared parameters to produce representation for both the pattern and the graph.

We also find that using curriculum learning (Bengio et al., 2009 ) can help models to converge better.

Hence, all models in Table 3 are fine-tuned based on the best models in small in the same settings.

Training and evaluating were finished on one single NVIDIA GTX 1080 Ti GPU under the PyTorch framework.

As we model this subgraph isomorphism counting problem as a regression problem, we use common metrics in regression tasks, including the root mean square error (RMSE) and the mean absolute error (MAE).

In this task, negative predictions are meaningless, so we only evaluate ReLU ( Y i ) as final prediction results.

Considering that about 75% of countings are 0's in our dataset, we also use evaluation metrics for the binary classification to analyze behaviors of different models.

We report F1 scores for both zero data (F1 zero ) and nonzero data (F1 nonzero ).

Two trivial baselines, Zero that always predicts 0 and Avg that always predicts the average counting of training data, are also used in comparison.

We first report results for small dataset in Table 2 and results for large dataset in Table 3 .

In addition to the trivial all-zero and average baselines and other neural network learning based baselines, we are also curious about to what extent our neural models can be faster than traditional searching algorithms.

Therefore, we also compare the running time.

Considering the graph generation strategy we used, we decide to compare with VF2 algorithm (Cordella et al., 2004) to avoid unnecessary interference from the similar searching strategy.

From the experiments, we can draw following observations and conclusions.

Comparison of different representation architectures.

As shown in Table 2 , in general, graph models outperform most of the sequence models but cost more time to do inference.

CNN is the worst model for the graph isomorphism counting problem.

The most possible reason is that the sequence encoding method is not suitable for CNN.

The code order does not consider the connectivity of adjacent vertices and relevant label information.

Hence, convolutional operations and pooling operations cannot extract useful local information but may introduce much noise.

From results of the large dataset, we can see that F1 nonzero =0.180 is even worse than others.

In fact, we find that CNN always predicts 0 for large graphs.

RNN and TXL are widely used in modeling sequences in many applications.

The two models with simple pooling can perform well.

We note that RNN with sum-pooling is better than TXL with memory.

RNN itself holds a memory but TXL also has much longer memory.

However, the memory in RNN can somehow memorize all information that Under review as a conference paper at ICLR 2020 has been seen previously but the memory of TXL is the representation of the previous segment.

In our experiments, the segment size is 64 so that TXL can not learn the global information at a time.

A part of the structure information misleads TXL, which is consistent with CNN.

A longer segment set for TXL may lead to better results, but it will require much more GPU memory and much longer time for training.

RGCN-SUM is much better than RGCN and other sequence models, which shows that sum aggregator is good at modeling vertex representation in this task.

The mean aggregator can model the distribution of neighbor but the distribution can also misguide models.

Effectiveness of the memory.

Table 2 shows the effectiveness of our dynamic attention memory network as the prediction layer.

It outperforms the other three pooling methods as well as the simple attention mechanism for all representation architectures.

Sum, Mean, and Attention pooling are all Figure 5: Model behaviors of three models in small dataset.

The x-axis is the example id and the y-axis is the count value.

We mark the ground truth value as orange + and the predictions as blue ??. We use two green dashed lines to separate patterns into three blocks based on numbers of vertices (3,4, and 8).

In each block, the examples are sorted based on the data graphs' sizes.

comparable with each other, because they all gather the global information of the pattern and graph representations.

Prediction layer based on max pooling, however, performs the worst, and even worse when the representation layer is CNN or Transformer-XL.

This observation indicates that every context of the pattern representation should be counted and we need a better way to compute the weights between each context.

The dynamic attention memory with global information of both the pattern and the graph achieves the best results in most of the cases.

One of the most interesting observations is that it can even help extract the context of pattern and graph while the representation layer (such as CNN) does not perform very well, which proves the power of our proposed method of DIAMNet.

Performance on larger graphs.

Table 3 shows our models can be applied to larger-scale graphs.

For the large dataset, we only choose the best pooling method for each of the baselines to report.

We can find most of the results are consistent to the small dataset, which means RGCN is the best representation method in our task and the dynamic memory is effective.

In terms of the running time, all learning based models are much faster than the traditional VF2 algorithm for subgraph isomorphism counting.

Model behaviors.

As shown in Figure 5 , we compare the model behaviors of the best model (RGCN+SUM) and the worst model (CNN), as well as the great improvement of CNN when memory is added.

We can find that CNN+SumPool tends to predict the count value below 400 and has the same behavior between three patterns.

This results may come from the fact that CNN can only extract local information of a sequence and Sum pooling is not a good way to aggregate it.

However, the memory can memorize local information to each memory cell so it can improve the representation power of CNN and can gain a better performance.

RGCN, on the other hand, can better represent the graph structure, so it achieves a better result, especially on the largest pattern (the third block of each figure) compared with CNN.

More results can be found in Appendix F.

In this paper, we study the challenging subgraph isomorphism counting problem.

With the help of deep graph representation learning, we are able to convert the NP-complete problem to a learning based problem.

Then we can use the learned model to predict the subgraph isomorphism counts in polynomial time.

Counting problem is more related to a global inference rather than only learning node or edge representations.

Therefore, we have developed a dynamic intermedium attention memory network to memorize local information and summarize for the global output.

We build two datasets to evaluate different representation learning models and global inference models.

Results show that learning based method is a promising direction for subgraph isomorphism detection and counting and memory networks indeed help the global inference.

We also performed detailed analysis of model behaviors for different pattern and graph sizes and labels.

Results show that there is much space to improve when the vertex label size is large.

Moreover, we have seen the potential real-world applications of subgraph isomorphism counting problems such as question answering and information retrieval.

It would be very interesting to see the domain adaptation power of our developed pretrained models on more real-world applications.

The lexicographic order is a linear order defined as follows:

If A = (a 0 , a 1 , ?? ?? ?? , a m ) and B = (b 0 , b 1 , ?? ?? ?? , b n ) are the codes, then A ??? B iff either of the following is true:

2. ???0 ??? k ??? m, a k = b k , and n ??? m.

In our setting,

)

iff one of the following is true:

B PATTERN GENERATOR AND GRAPH GENERATOR As proposed in Section 5.1, two generators are required to generate datasets.

The algorithm about the pattern generator is shown in Algorithm 1.

The algorithm first uniformly generates a directed tree.

Then it adds the remaining edges with random labels.

Vertex labels and edge labels are also uniformly generated but each label is required to appear at least once.

Algorithm 2 shows the process of graph generation.

Two hyperparameters control the density of subisomorphisms: (1) ?? ??? [0, 1] decides the probability of adding subisomorphisms rather than random edges; (2) ?? ??? N + is the parameter of Dirichlet distribution to sample sizes of components.

After generating several directed trees and satisfying the vertex number requirement, the algorithm starts to add remaining edges.

It can add edges in one component and try to add subgraph isomorphisms, or it can randomly add edges between two components or in one component.

The following merge subroutine aims to merge these components into a large graph.

Shuffling is also required to make datasets hard to be hacked.

The search of subisomorphisms in the whole graph is equivalent to the search in components respectively because edges between any two components do not satisfy the necessary conditions.

Algorithm 1 Pattern Generator.

Input: the number of vertices N v , the number of edges N e , the number of vertex labels L v , the number of edge labels L e .

1: P := GenerateDirectedTree(N v ) 2: AssignNodesLabels(P, L v ) 3: AddRandomEdges(P, P, null, N e ??? N v + 1) 4: AssignEdgesLabels(P, L e ) Output: the generated pattern P In Algorithm 1 and Algorithm 2, the function AddRandomEdges adds required edges from one component to the other without generating new subgraph isomorphisms.

The two component can also be the same one, which means to add in one component.

The NEC tree is utilized in TurboISO (Han et al., 2013) to explore the candidate region for further matching.

It takes O(|V p | 2 ) time but can significant reduce the searching space in the data graph.

It records the equivalence classes and necessary conditions of the pattern.

We make sure edges between two components dissatisfy necessary conditions in the NEC tree when adding random edges between them.

This data structure and this idea help us to generate more data and search subisomorphisms compared with random generation and traditional subgraph isomorphism searching.

We can generate as many examples as possible using two graph generators.

However, we limit the numbers of training, dev, and test examples whether learning based models can generalize to

Intuitively, not all vertices and edges in graphs can match certain subisomorphisms so we simply add a FilterNet to adjust graph encoding as follows:

where G is the graph representation in the pattern space, ?? is the sigmoid function, f i is the gate that decides to filter the j th vertex or the j th edge, W G ??? R dp??dg and W F ??? R 1??dp are trainable.

Thanks to the multi-hot encoding, We can simply use Eq. (7) to accumulate label information of patterns.

After this filter layer, only relevant parts of the graphs will be passed to the next representation layer.

The computation process of the AttnPool (Mean) is:

Combing with the pooling strategy to makeP as P ??? R 1??d , the time complexity is decreased to

We also implement three variants with sum-pooling, mean-pooling, and max-pooling respectively.

Results of AttnPool (Sum) and AttnPool (Max) are shown in the Appendix E.

There are many ways to initialize the memory {m

where s is the stride and k is the kernel size.

In Table 5 , we compared two additional pooling methods with MeanPool in Eq. (15).

Table 5 shows results of different representation models with different interaction networks in the small dataset.

AttnPool (Mean) and DIAMNet (MeanInit) usually perform better compared with other pooling methods.

As shown in Figure 7 , different interaction modules perform differently in different views.

We can find MaxPool always predicts higher counting values when the pattern is small and the graph is large, while AttnPool always predicts very small numbers except when the pattern vertex size is 8, and the graph vertex size is 64.

The same result appears when we use edge sizes as the x-axis.

This observation shows that AttnPool has difficulties predicting counting values when either of the pattern and the graph is small.

It shows that attention focuses more on the zero vector we added rather than the pattern pooling result.

Our DIAMNet, however, performs the best in all pattern/graph sizes.

When the bins are ordered by vertex label sizes or edge label sizes, the performance of all the three interaction modules among the distribution are similar.

When bins are ordered by vertex label sizes, we have the same discovery that AttnPool prefers to predict zeros when then patterns are small.

MaxPool fails when facing complex patterns with more vertex labels.

DIAMNet also performs not so good over these patterns.

As for edge labels, results look good for MaxPool and DIAMNet but AttnPool is not satisfactory.

As shown in Figure 8 , different representation modules perform differently in different views.

CNN performs badly when the graph size is large (shown in Figure 8a and 8d) and patterns become complicated (show in Figure 8g and 8j), which further indicates that CNN can only extract the local information and suffers from issues when global information is need in larger graphs.

RNN, on the other hand, performs worse when the graph are large, especially when patterns are small (show in Figure 8e ), which is consistent with its nature, intuitively.

On the contrary, RGCN-SUM with DIAMNet is not affected by the edge sizes because it directly learns vertex representations rather than edge representations.

<|TLDR|>

@highlight

In this paper, we study a new graph learning problem: learning to count subgraph isomorphisms.
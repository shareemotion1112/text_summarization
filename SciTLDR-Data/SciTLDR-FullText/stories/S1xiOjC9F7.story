This paper addresses the challenging problem of retrieval and matching of graph structured objects, and makes two key contributions.

First, we demonstrate how  Graph Neural Networks (GNN), which have emerged as an effective model for various supervised prediction problems defined on structured data, can be trained to produce embedding of graphs in vector spaces that enables efficient similarity reasoning.

Second, we propose a novel Graph Matching Network model that, given a pair of graphs as input, computes a similarity score between them by jointly reasoning on the pair through a new cross-graph attention-based matching mechanism.

We demonstrate the effectiveness of our models on different domains including the challenging problem of control-flow-graph based function similarity search that plays an important role in the detection of vulnerabilities in software systems.

The experimental analysis demonstrates that our models are not only able to exploit structure in the context of similarity learning but they can also outperform domain-specific baseline systems that have been carefully hand-engineered for these problems.

Graphs are natural representations for encoding relational structures that are encountered in many domains.

Expectedly, computations defined over graph structured data are employed in a wide variety of fields, from the analysis of molecules for computational biology and chemistry BID22 BID38 , to the analysis of knowledge graphs or graph structured parses for natural language understanding.

In the past few years graph neural networks (GNNs) have emerged as an effective class of models for learning representations of structured data and for solving various supervised prediction problems on graphs.

Such models are invariant to permutations of graph elements by design and compute graph node representations through a propagation process which iteratively aggregates local structural information (Scarselli et al., 2009; BID31 BID22 ).

These node representations are then used directly for node classification, or pooled into a graph vector for graph classification.

Problems beyond supervised classification or regression are relatively less well-studied for GNNs.

In this paper we study the problem of similarity learning for graph structured objects, which appears in many important real world applications, in particular similarity based retrieval in graph databases.

One motivating application is the computer security problem of binary function similarity search, where given a binary which may or may not contain code with known vulnerabilities, we wish to check whether any control-flow-graph in this binary is sufficiently similar to a database of known-vulnerable functions.

This helps identify vulnerable statically linked libraries in closed-source software, a recurring problem (CVE, 2010; BID16 for which no good solutions are currently available.

Figure 1 shows one example from this application, where the binary functions are represented as control flow graphs annotated with assembly instructions.

This similarity learning problem is very challenging as subtle differences can make two graphs be semantically very different, while graphs with different structures can still be similar.

A successful model for this problem should therefore (1) exploit the graph structures, and (2) be able to reason about the similarity of graphs both from the graph structures as well as from learned semantics.

Figure 1 : The binary function similarity learning problem.

Checking whether two graphs are similar requires reasoning about both the structure as well as the semantics of the graphs.

Here the left two control flow graphs correspond to the same function compiled with different compilers (and therefore similar), while the graph on the right corresponds to a different function.

In order to solve the graph similarity learning problem, we investigate the use of GNNs in this context, explore how they can be used to embed graphs into a vector space, and learn this embedding model to make similar graphs close in the vector space, and dissimilar graphs far apart.

One important property of this model is that, it maps each graph independently to an embedding vector, and then all the similarity computation happens in the vector space.

Therefore, the embeddings of graphs in a large database can be precomputed and indexed, which enables efficient retrieval with fast nearest neighbor search data structures like k-d trees BID4 or locality sensitive hashing BID23 .We further propose an extension to GNNs which we call Graph Matching Networks (GMNs) for similarity learning.

Instead of computing graph representations independently for each graph, the GMNs compute a similarity score through a cross-graph attention mechanism to associate nodes across graphs and identify differences.

By making the graph representation computation dependent on the pair, this matching model is more powerful than the embedding model, providing a nice accuracy-computation trade-off.

We evaluate the proposed models and baselines on three tasks: a synthetic graph edit-distance learning task which captures structural similarity only, and two real world tasks -binary function similarity search and mesh retrieval, which require reasoning about both the structural and semantic similarity.

On all tasks, the proposed approaches outperform established baselines and structure agnostic models; in more detailed ablation studies, we found that the Graph Matching Networks consistently outperform the graph embedding model and Siamese networks.

To summarize, the contributions of this paper are:(1) we demonstrate how GNNs can be used to produce graph embeddings for similarity learning; (2) we propose the new Graph Matching Networks that computes similarity through cross-graph attention-based matching; (3) empirically we show that the proposed graph similarity learning models achieve good performance across a range of applications, outperforming structure agnostic models and established hand-engineered baselines.

Graph Neural Networks and Graph Representation Learning The history of graph neural networks (GNNs) goes back to at least the early work by BID24 and Scarselli et al. (2009) , who proposed to use a propagation process to learn node representations.

These models have been further developed by incorporating modern deep learning components BID31 Veličković et al., 2017; BID10 .

A separate line of work focuses on generalizing convolutions to graphs BID10 BID9 .

Popular graph convolutional networks also compute node updates by aggregating information in local neighborhoods BID29 , making them the same family of models as GNNs.

GNNs have been successfully used in many domains BID29 Veličković et al., 2017; BID2 BID16 BID33 BID17 BID22 BID13 BID32 Wang et al., 2018a; BID3 .

Most of the previous work on GNNs focus on supervised prediction problems (with exceptions like BID13 BID32 Wang et al., 2018a) ).

The graph similarity learning problem we study in this paper and the new graph matching model can be good additions to this family of models.

Graph Similarity Search and Graph Kernels Graph similarity search has been studied extensively in database and data mining communities BID38 BID15 ).

The similarity is typically defined by either exact matches (full-graph or sub-graph isomorphism) BID5 Shasha et al., 2002; Yan et al., 2004; Srinivasa & Kumar, 2003) or some measure of structural similarity, e.g. in terms of graph edit distances (Willett et al., 1998; BID36 .

Most of the approaches proposed in this direction are not learning-based, and focus on efficiency.

Graph kernels are kernels on graphs designed to capture the graph similarity, and can be used in kernel methods for e.g. graph classification (Vishwanathan et al., 2010; Shervashidze et al., 2011) .

Popular graph kernels include those that measure the similarity between walks or paths on graphs BID7 BID27 Vishwanathan et al., 2010) , kernels based on limited-sized substructures BID25 Shervashidze et al., 2009 ) and kernels based on sub-tree structures (Shervashidze & Borgwardt, 2009; Shervashidze et al., 2011) .

Graph kernels are usually used in models that may have learned components, but the kernels themselves are handdesigned and motivated by graph theory.

They can typically be formulated as first computing the feature vectors for each graph (the kernel embedding), and then take inner product between these vectors to compute the kernel value.

One exception is BID39 where the co-occurrence of graph elements (substructures, walks, etc.) are learned, but the basic elements are still hand-designed.

Compared to these approaches, our graph neural network based similarity learning framework learns the similarity metric end-to-end.

Distance Metric Learning Learning a distance metric between data points is the key focus of the area of metric learning.

Most of the early work on metric learning assumes that the data already lies in a vector space, and only a linear metric matrix is learned to properly measure the distance in this space to group similar examples together and dissimilar examples to be far apart (Xing et al., 2003; Weinberger & Saul, 2009; BID14 .

More recently the ideas of distance metric learning and representation learning have been combined in applications like face verification, where deep convolutional neural networks are learned to map similar images to similar representation vectors BID12 BID26 Sun et al., 2014) .

In this paper, we focus on representation and similarity metric learning for graphs, and our graph matching model goes one step beyond the typical representation learning methods by modeling the cross-graph matchings.

Siamese Networks Siamese networks BID8 BID1 ) are a family of neural network models for visual similarity learning.

These models typically consist of two networks with shared parameters applied to two input images independently to compute representations, a small network is then used to fuse these representations and compute a similarity score.

They can be thought of as learning both the representations and the similarity metric.

Siamese networks have achieved great success in many visual recognition and verification tasks BID8 BID1 BID30 BID6 BID40 .

In the experiments we adapt Siamese networks to handle graphs, but found our graph matching networks to be more powerful as they do cross-graph computations and therefore fuse information from both graphs early in the computation process.

Independent of our work, recently (Shyam et al., 2017) proposed a cross-example attention model for visual similarity as an alternative to Siamese networks based on similar motivations and achieved good results.

Given two graphs DISPLAYFORM0 , we want a model that produces the similarity score s(G 1 , G 2 ) between them.

Each graph G = (V, E) is represented as sets of nodes V and edges E, optionally each node i ∈ V can be associated with a feature vector x i , and each edge (i, j) ∈ E associated with a feature vector x ij .

These features can represent, e.g. type of a node, direction of an edge, etc.

If a node or an edge does not have any associated features, we set the corresponding vector to a constant vector of 1s.

We propose two models for graph similarity learning: a model based on standard GNNs for learning graph embeddings, and the new and more powerful GMNs.

The two models are illustrated in FIG1 .

Graph embedding models embed each graph into a vector, and then use a similarity metric in that vector space to measure the similarity between graphs.

Our GNN embedding model comprises 3 parts: (1) an encoder, (2) propagation layers, and (3) an aggregator.

Encoder The encoder maps the node and edge features to initial node and edge vectors through separate MLPs: DISPLAYFORM0 Propagation Layers A propagation layer maps a set of node representations {h DISPLAYFORM1 } i∈V , as the following: DISPLAYFORM2 Here f message is typically an MLP on the concatenated inputs, and f node can be either an MLP or a recurrent neural network core, e.g. RNN, GRU or LSTM BID31 .

To aggregate the messages, we use a simple sum which may be alternatively replaced by other commutative operators such as mean, max or the attention-based weighted sum (Veličković et al., 2017) .

Through multiple layers of propagation, the representation for each node will accumulate information in its local neighborhood.

Aggregator After a certain number T rounds of propagations, an aggregator takes the set of node representations {h (T ) i } as input, and computes a graph level representation DISPLAYFORM3 which transforms node representations and then uses a weighted sum with gating vectors to aggregate across nodes.

The weighted sum can focus only on the important nodes, it is more powerful than a simple sum and also works significantly better empirically.

After the graph representations h G1 and h G2 are computed for the pair (G 1 , G 2 ), we compute the similarity between them using a similarity metric in the vector space, for example the Euclidean, cosine or Hamming similarities.

Note that without the propagation layers (or with 0 propagation steps), this model becomes an instance of the Deep Set BID41 or PointNet BID35 , which does computation on the individual nodes, and then pool the node representations into a representation for the whole graph.

Such a model, however, ignores the structure and only treats the data as a set of independent nodes.

Graph matching networks take a pair of graphs as input and compute a similarity score between them.

Compared to the embedding models, these matching models compute the similarity score jointly on the pair, rather than first independently mapping each graph to a vector.

Therefore these models are potentially stronger than the embedding models, at the cost of some extra computation efficiency.

We propose the following graph matching network, which changes the node update module in each propagation layer to take into account not only the aggregated messages on the edges for each graph as before, but also a cross-graph matching vector which measures how well a node in one graph can be matched to one or more nodes in the other: DISPLAYFORM0 Here f s is a standard vector space similarity between h G1 and h G2 .

f match is a function that communicates cross-graph information, which we propose to use an attention-based module: DISPLAYFORM1 s h is again a vector space similarity metric, like Euclidean or cosine similarity, a j→i are the attention weights, and j µ j→i intuitively measures the difference between h (t)i and its closest neighbor in the other graph.

Note that because of the normalization in a j→i , the function f match implicitly depends on the whole set of {h (t) j }, which we omitted in Eq. 8 for a cleaner notation.

Since attention weights are required for every pair of nodes across two graphs, this operation has a computation cost of O(|V 1 ||V 2 |), while for the GNN embedding model the cost for each round of propagation is O(|V | + |E|).

The extra power of the GMNs comes from utilizing the extra computation.

Note By construction, the attention module has a nice property that, when the two graphs can be perfectly matched, and when the attention weights are peaked at the exact match, we have j µ j→i = 0, which means the cross-graph communications will be reduced to zero vectors, and the two graphs will continue to compute identical representations in the next round of propagation.

On the other hand, the differences across graphs will be captured in the cross-graph matching vector j µ j→i , which will be amplified through the propagation process, making the matching model more sensitive to these differences.

Compared to the graph embedding model, the matching model has the ability to change the representation of the graphs based on the other graph it is compared against.

The model will adjust graph representations to make them become more different if they do not match.

The proposed graph similarity learning models can be trained on a set of example pairs or triplets.

Pairwise training requires us to have a dataset of pairs labeled as positive (similar) or negative (dissimilar), while triplet training only needs relative similarity, i.e. whether G 1 is closer to G 2 or G 3 .

We describe the losses on pairs and triplets we used below, which are then optimized with gradient descent based algorithms.

When using Euclidean similarity, we use the following margin-based pairwise loss: DISPLAYFORM0 where t ∈ {−1, 1} is the label for this pair, γ > 0 is a margin parameter, and d(G 1 , G 2 ) = h G1 − h G2 2 is the Euclidean distance.

This loss encourages d(G 1 , G 2 ) < 1 − γ when the pair is similar (t = 1), and d(G 1 , G 2 ) > 1 + γ when t = −1.

Given triplets where G 1 and G 2 are closer than G 1 and G 3 , we optimize the following margin-based triplet loss: DISPLAYFORM1 This loss encourages d(G 1 , G 2 ) to be smaller than d(G 1 , G 3 ) by at least a margin γ.

For applications where it is necessary to search through a large database of graphs with low latency, it is beneficial to have the graph representation vectors be binary, i.e. h G ∈ {−1, 1} H , so that efficient nearest neighbor search algorithms BID23 ) may be applied.

In such cases, we can minimize the Hamming distance of positive pairs and maximize it for negative pairs.

With this restriction the graph vectors can no longer freely occupy the whole Euclidean space, but we gain the efficiency for fast retrieval and indexing.

To achieve this we propose to pass the h G vectors through a tanh transformation, and optimize the following pair and triplet losses: DISPLAYFORM2 DISPLAYFORM3 where DISPLAYFORM4 is the approximate average Hamming similarity.

Both losses are bounded in [0, 1], and they push positive pairs to have Hamming similarity close to 1, and negative pairs to have similarity close to -1.

We found these losses to be a bit more stable than margin based losses for Hamming similarity.

In this section, we evaluate the graph similarity learning (GSL) framework and the graph embedding (GNNs) and graph matching networks (GMNs) on three tasks and compare these models with other competing methods.

Overall the empirical results demonstrate that the GMNs excel on graph similarity learning, consistently outperforming all other approaches.

Problem Background Graph edit distance between graphs G 1 and G 2 is defined as the minimum number of edit operations needed to transform G 1 to G 2 .

Typically the edit operations include add/remove/substitute nodes and edges.

Graph edit distance is naturally a measure of similarity between graphs and has many applications in graph similarity search BID15 BID42 BID21 .

However computing the graph edit distance is NP-hard in general BID42 , therefore approximations have to be used.

Through this experiment we show that the GSL models can learn structural similarity between graphs on very challenging problems.

Training Setup We generated training data by sampling random binomial graphs G 1 with n nodes and edge probability p BID18 , and then create positive example G 2 by randomly substituting k p edges from G 1 with new edges, and negative example G 3 by substituting k n edges from G 1 , where k p < k n 1 .

A model needs to predict a higher similarity score for positive pair (G 1 , G 2 ) than negative pair (G 1 , G 3 ).

Throughout the experiments we fixed the dimensionality of node vectors to 32, and the dimensionality of graph vectors to 128 without further tuning.

We also tried different number of propagation steps T from 1 to 5, and observed consistently better performance with increasing T .

The results reported in this section are all with T = 5 unless stated otherwise.

More details are included in Appendix B.1.Baseline We compare our models with the popular Weisfeiler Lehman (WL) kernel (Shervashidze et al., 2011) , which has been shown to be very competitive on graph classification tasks and the Weisfeiler Lehman algorithm behind this kernel is a strong method for checking graph isomorphism (edit distance of 0), a closely related task (Weisfeiler & Lehman, 1968; Shervashidze et al., 2011 ).Evaluation The performance of different models are evaluated using two metrics: (1) pair AUCthe area under the ROC curve for classifying pairs of graphs as similar or not on a fixed set of 1000 pairs and (2) triplet accuracy -the accuracy of correctly assigning higher similarity to the positive pair in a triplet than the negative pair on a fixed set of 1000 triplets.

Results We trained and evaluated the GSL models on graphs of a few specific distributions with different n, p, with k p = 1 and k n = 2 fixed.

The evaluation results are shown in Table 1 .

We can see that by learning on graphs of specific distributions, the GSL models are able to do better than generic baselines, and the GMNs consistently outperform the embedding model (GNNs).For the GMNs, we can visualize the cross-graph attention to gain further insight into how it is working. tested on graphs of 10 nodes.

The cross-graph attention weights are shown in green, with the scale of the weights shown as the transparency of the green edges.

We can see that the attention weights can align nodes well when the two graphs match, and tend to focus on nodes with higher degrees when they don't.

However the pattern is not as interpretable as in standard attention models.

More experiments on generalization capabilities of these models (train on small graphs, test on larger graphs, train on graphs with some k p , k n combinations, test on others) are included in Appendix B.1.

Problem Background Binary function similarity search is an important problem in computer security.

The need to analyze and search through binaries emerges when we do not have access to the source code, for example when dealing with commercial or embedded software or suspicious executables.

Combining a disassembler and a code analyzer, we can extract a control-flow graph (CFG) which contains all the information in a binary function in a structured format.

See Figure 1 and Appendix B.2 for a few example CFGs.

In a CFG, each node is a basic block of assembly instructions, and the edges between nodes represent the control flow, indicated by for example a jump or a return instruction used in branching, loops or function calls.

In this section, we target the vulnerability search problem, where a piece of binary known to have some vulnerabilities is used as the query, and we search through a library to find similar binaries that may have the same vulnerabilities.

2 Accurate identification of similar vulnerabilities enables security engineers to quickly narrow down the search space and apply patches.

In the past the binary function similarity search problem has been tackled with classical graph theoretical matching algorithms BID19 BID34 , and Xu et al. FORMULA1 and BID20 proposed to learn embeddings of CFGs and do similarity search in the embedding space.

Xu et al. (2017) in particular proposed an embedding method based on graph neural networks, starting from some hand selected feature vectors for each node.

Here we study further the performance of graph embedding and matching models, with pair and triplet training, different number of propagation steps, and learning node features from the assembly instructions.

Training Setup and Baseline We train and evaluate our model on data generated by compiling the popular open source video processing software ffmpeg using different compilers gcc and clang, and different compiler optimization levels, which results in 7940 functions and roughly 8 CFGs per function.

The average size of the CFGs is around 55 nodes per graph, with some larger graphs having up to a few thousand nodes (see Appendix B.2 for more detailed statistics).

Different compiler optimization levels result in CFGs of very different sizes for the same function.

We split the data and used 80% functions and the associated CFGs for training, 10% for validation and 10% for testing.

The models were trained to learn a similarity metric on CFGs such that the CFGs for the same function have high similarity, and low similarity otherwise.

Once trained, this similarity metric can be used to search through library of binaries and be invariant to compiler type and optimization levels.

We compare our graph embedding and matching models with Google's open source function similarity search tool BID16 , which has been used to successfully find vulnerabilities in binaries in the past.

This tool computes representations of CFGs through a hand-engineered graph hashing process which encodes the neighborhood structure of each node by hashing the degree sequence from a traversal of a 3-hop neighborhood, and also encodes the assembly instructions for each basic block by hashing the trigrams of assembly instruction types.

These features are then combined by using a SimHash-style BID11 algorithm with learned weights to form a 128-dimensional binary code.

An LSH-based search index is then used to perform approximate nearest neighbor search using hamming distance.

Following BID16 , we also map the CFGs to 128-dimensional binary vectors, and use the Hamming similarity formulation described in Section 3 for training.

We further studied two variants of the data, one that only uses the graph structure, and one that uses both the graph structure and the assembly instructions with learned node features.

When assembly instructions are available, we embed each instruction type into a vector, and then sum up all the embedding vectors for instructions in a basic block as the initial representation vector (the x i 's) for each node, these embeddings are learned jointly with the rest of the model.

Results FIG3 shows the performance of different models with different number of propagation steps and in different data settings.

We again evaluate the performance of these models on pair AUC and triplet accuracy on fixed sets of pairs and triplets from the test set.

It is clear from results that: (1) the performance of both the graph embedding and matching models consistently go up with more propagation steps, and in particular significantly outperforming the structure agnostic model special case which uses 0 propagation steps; (2) the graph embedding model is consistently better than the baselines with enough propagation steps; and (3) graph matching models outperforms the embedding models across all settings and propagation steps.

Additionally, we have tried the WL kernel on this task using only the graph structure, and it achieved 0.619 AUC and 24.5% triplet accuracy.

This is not surprising as the WL kernel is not designed for solving this task, while our models learn the features useful for the task of interest, and can achieve better performance than generic similarity metrics.

In this section, we carefully examine the effects of the design decisions we made in the GMN model and compare it against a few more alternatives.

In particular, we evaluate the popular Graph Convolutional Network (GCN) model by BID29 as an alternative to our GNN model, and Siamese versions of the GNN/GCN embedding models.

The GCN model replaces the message passing in Eq. 2 with graph convolutions, and the Siamese model predicts a distance value by concatenating two graph vectors and then pass through a 2 layer MLP.

The comparison with Siamese networks can in particular show the importance of the cross-graph attention early on in the similarity computation process, as Siamese networks fuse the representations for 2 graphs only at the very end.

We focus on the function similarity search task, and also conduct experiments on an extra COIL-DEL mesh graph dataset BID37 , which contains 100 classes of mesh graphs corresponding to 100 types of objects.

We treat graphs in the same class as similar, and used identical setup as the function similarity search task for training and evaluation.

Function Similarity Search COIL-DEL Table 2 : More results on the function similarity search task and the extra COIL-DEL dataset.

Table 2 summarizes the experiment results, which clearly show that: (1) the GNN embedding model is a competitive model (more powerful than the GCN model); (2) using Siamese network architecture to learn similarity on top of graph representations is better than using a prespecified similarity metric (Euclidean, Hamming etc.); (3) the GMNs outperform the Siamese models showing the importance of cross-graph information communication early in the computation process.

In this paper we studied the problem of graph similarity learning using graph neural networks.

Compared to standard prediction problems for graphs, similarity learning poses a unique set of challenges and potential benefits.

For example, the graph embedding models can be learned through a classification setting when we do have a set of classes in the dataset, but formulating it as a similarity learning problem can handle cases where we have a very large number of classes and only very few examples for each class.

The representations learned from the similarity learning setting can also easily generalize to data from classes unseen during training (zero-shot generalization).We proposed the new graph matching networks as a stronger alternative to the graph embedding models.

The added power for the graph matching models comes from the fact that they are not independently mapping each graph to an embedding, but rather doing comparisons at all levels across the pair of graphs, in addition to the embedding computation.

The model can then learn to properly allocate capacity toward the embedding part or the matching part.

The price to pay for this expressivity is the added computation cost in two aspects: (1) since each cross-graph matching step requires the computation of the full attention matrices, which requires at least O(|V 1 ||V 2 |) time, this may be expensive for large graphs; (2) the matching models operate on pairs, and cannot directly be used for indexing and searching through large graph databases.

Therefore it is best to use the graph matching networks when we (1) only care about the similarity between individual pairs, or (2) use them in a retrieval setting together with a faster filtering model like the graph embedding model or standard graph similarity search methods, to narrow down the search to a smaller candidate set, and then use the more expensive matching model to rerank the candidates to improve precision.

Developing neural models for graph similarity learning is an important research direction with many applications.

There are still many interesting challenges to resolve, for example to improve the efficiency of the matching models, study different matching architectures, adapt the GNN capacity to graphs of different sizes, and applying these models to new application domains.

We hope our work can spur further research in this direction.

In the propagation layers of the graph embedding and matching models, we used an MLP with one hidden layer as the f message module, with a ReLU nonlinearity on the hidden layer.

For node state vectors (the h (t) i vectors) of dimension D, the size of the hidden layer and the output is set to 2D.

We found it to be beneficial to initialize the weights of this f message module to be small, which helps stablizing training.

We used the standard Glorot initialization with an extra scaling factor of 0.1.

When not using this small scaling factor, at the begining of training the message vectors when summed up can have huge scales, which is bad for learning.

One extra thing to note about the propagation layers is that we can make all the propagation layers share the same set of parameters, which can be useful if this is a suitable inductive bias to have.

We tried different f node modules in both experiments, and found GRUs to generally work better than one-hidden layer MLPs, and all the results reported uses GRUs as f node , with the sum over edge messages j m j→i treated as the input to the GRU for the embedding model, and the concatenation of j m j→i and j µ j →i as the input to the GRU for the matching model.

In the aggregator module, we used a single linear layer for the node transformation MLP and the gating MLP gate in Eq.3.

The output of this linear layer has a dimensionality the same as the required dimensionality for the graph vectors.

σ(x) = 1 1+e −x is the logistic sigmoid function, and is the element-wise product.

After the weighted sum, another MLP with one hidden layers is used to further transform the graph vector.

The hidden layer has the same size as the output, with a ReLU nonlinearity.

For the matching model, the attention weights are computed as DISPLAYFORM0 We have tried the Euclidean similarity s h (h i , h j ) = − h i − h j 2 for s h , as well as the dot-product similarity s h (h i , h j ) = h i h j , and they perform similarly without significant difference.

We fixed the node state vector dimensionality to 32, and graph vector dimensionality to 128 throughout both the graph edit distance learning and binary function similarity search tasks.

We tuned this initially on the function similarity search task, which clearly performs better than smaller models.

Increasing the model size however leads to overfitting for that task.

We directly used the same setting for the edit distance learning task without further tuning.

Using larger models there should further improve model performance.

In this task the nodes and edges have no extra features associated with them, we therefore initialized the x i and x ij vectors as vectors of 1s, and the encoder MLP in Eq.1 is simply a linear layer for the nodes and an identity mapping for the edges.

We searched through the following hyperparameters: (1) triplet vs pair training; (2) number of propagation layers; (3) share parameters on different propagation layers or not.

Learning rate is fixed at 0.001 for all runs and we used the Adam optimizer BID28 .

Overall we found: (1) triplet and pair training performs similarly, with pair training slightly better, (2) using more propagation layers consistently helps, and increasing the number of propagation layers T beyond 5 may help even more, (3) sharing parameters is useful for performance more often than not.

Intuitively, the baseline WL kernel starts by labeling each node by its degree, and then iteratively updates a node's representation as the histogram of neighbor node patterns, which is effectively also a graph propagation process.

The kernel value is then computed as a dot product of graph representation vectors, which is the histogram of different node representations.

When using the kernel with T iterations of computation, a pair of graphs of size |V | can have as large as a 2|V |T Table 3 : Generalization performance on large graphs for the GSL models trained on small graphs with 20 ≤ n ≤ 50 and 0.2 ≤ p ≤ 0.5.

dimensional representation vector for each graph, and these sets of effective 'feature' types are different for different pairs of graphs as the node patterns can be very different.

This is an advantage for WL kernel over our models as we used a fixed sized graph vector regardless of the graph size.

We evaluate WL kernel for T up to 5 and report results for the best T on the evaluation set.

In addition to the experiments presented in the main paper, we have also tested the generalization capabilities of the proposed models, and we present the extra results in the following.

Train on small graphs, generalize to large graphs.

In this experiment, we trained the GSL models on graphs with n sampled uniformly from 20 to 50, and p sampled from range [0.2, 0.5] to cover more variability in graph sizes and edge density for better generalization, and we again fix k p = 1, k n = 2.

For evaluation, we tested the best embedding models and matching models on graphs with n = 100, 200 and p = 0.2, 0.5, with results shown in Table 3 .

We can see that for this task the GSL models trained on small graphs can generalize to larger graphs than they are trained on.

The performance falls off a bit on much larger graphs with much more nodes and edges.

This is also partially caused by the fact that we are using a fixed sized graph vector throughout the experiments , but the WL kernel on the other hand has much more effective 'features' to use for computing similarity.

On the other hand, as shown before, when trained on graphs from distributions we care about, the GSL models can adapt and perform much better.

Train on some k p , k n combinations, test on other combinations.

We have also tested the model trained on graphs with n ∈ [20, 50], p ∈ [0.2, 0.5], k p = 1, k n = 2, on graphs with different k p and k n combinations.

In particular, when evaluated on k p = 1, k n = 4, the models perform much better than on k p = 1, k n = 2, reaching 1.0 AUC and 100% triplet accuracy easily, as this is considerably simpler than the k p = 1, k n = 2 setting.

When evaluated on graphs with k p = 2, k n = 3, the performance is workse than k p = 1, k n = 2 as this is a harder setting.

In addition, we have also tried training on the more difficult setting k p = 2, k n = 3, and evaluate the models on graphs with k p = 1, k n = 2 and n ∈ [20, 50], p ∈ [0.2, 0.5].

The performance of the models on these graphs are actually be better than the models trained on this setting of k p = 1, k n = 2, which is surprising and clearly demonstrates the value of good training data.

However, in terms of generalizing to larger graphs models trained on k p = 2, k n = 3 does not have any significant advantages.

In this task the edges have no extra features so we initialize them to constant vectors of 1s, and the encoder MLP for the edges is again just an identity mapping.

When using the CFG graph structure only, the nodes are also initialized to constant vectors of 1s, and the encoder MLP is a linear layer.

In the case when using assembly instructions, we have a list of assembly code associated with each node.

We extracted the operator type (e.g. add, mov, etc.) from each instruction, and then embeds each operator into a vector, the initial node representation is a sum of all operator embeddings.

We searched through the following hyperparameters: (1) triplet or pair training, (2) learning rate in {10 −3 , 10 −4 }, (3) number of propagation layers; (4) share propagation layer parameters or not; (5) GRU vs one-layer MLP for the f node module.

Overall we found that (1) triplet training performs slightly better than pair training in this case; (2) both learning rates can work but the smaller learning rate is more stable; (3) increasing number of propagation layers generally helps; (4) using different propagation layer parameters perform better than using shared parameters; (5) GRUs are more stable than MLPs and performs overall better.

Figure 5: Example control flow graphs for the same binary function, compiled with different compilers (clang for the leftmost one, gcc for the others) and optimization levels.

Note that each node in the graphs also contains a set of assembly instructions which we also take into account when computing similarity using learned features.

In addition to the results reported in the main paper, we have also tried the same models on another dataset obtained by compiling the compression software unrar with different compilers and optimization levels.

Our graph similarity learning methods also perform very well on the unrar data, but this dataset is a lot smaller, with around 400 functions only, and overfitting is therefore a big problem for any learning based model, so the results on this dataset are not very reliable to draw any conclusions.

A few more control-flow graph examples are shown in Figure 5 .

The distribution of graph sizes in the training set is shown in FIG5 .

A few more attention visualizations are included in Figure 7 , Figure 8 and Figure 9 .

Here the graph matching model we used has shared parameters for all the propagation and matching layers and was trained with 5 propagation layers.

Therefore we can use a number T different from the number of propagation layers the model is being trained on to test the model's performance.

In both visualizations, we unrolled the propagation for up to 9 steps and the model still computes sensible attention maps even with T > 5.

Note that the attention maps do not converge to very peaked distributions.

This is partially due to the fact that we used the node state vectors both to carry information through the propagation process, as well as in the attention mechanism as is.

This makes it hard for the model to have very peaked attention as the scale of these node state vectors won't be very big.

A better solution is to compute separate key, query and value vectors for each node as done in the tensor2tensor self-attention formulation Vaswani et al. (2017) , which may further improve the performance of the matching model.

@highlight

We tackle the problem of similarity learning for structured objects with applications in particular in computer security, and propose a new model graph matching networks that excels on this task.

@highlight

Authors introduce a Graph Matching Network for retrival and matching of graph structured objects.

@highlight

The authors attack the problem of graph matching by proposing an extension of graph embedding networks

@highlight

The authors present two methods for learning a similarity score between pairs of graphs and show the beneficiality of introducing idesa from graph matching to graph neural networks.
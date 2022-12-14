Graph embedding techniques have been increasingly deployed in a multitude of different applications that involve learning on non-Euclidean data.

However, existing graph embedding models either fail to incorporate node attribute information during training or suffer from node attribute noise, which compromises the accuracy.

Moreover, very few of them scale to large graphs due to their high computational complexity and memory usage.

In this paper we propose GraphZoom, a multi-level framework for improving both accuracy and scalability of unsupervised graph embedding algorithms.

GraphZoom first performs graph fusion to generate a new graph that effectively encodes the topology of the original graph and the node attribute information.

This fused graph is then repeatedly coarsened into a much smaller graph by merging nodes with high spectral similarities.

GraphZoom allows any existing embedding methods to be applied to the coarsened graph, before it progressively refine the embeddings obtained at the coarsest level to increasingly finer graphs.

We have evaluated our approach on a number of popular graph datasets for both transductive and inductive tasks.

Our experiments show that GraphZoom increases the classification accuracy and significantly reduces the run time compared to state-of-the-art unsupervised embedding methods.

Recent years have seen a surge of interest in graph embedding, which aims to encode nodes, edges, or (sub)graphs into low dimensional vectors that maximally preserve graph structural information.

Graph embedding techniques have shown promising results for various applications such as vertex classification, link prediction, and community detection (Zhou et al., 2018) ; (Cai et al., 2018) ; (Goyal & Ferrara, 2018 ).

However, current graph embedding methods have several drawbacks.

On the one hand, random-walk based embedding algorithms, such as DeepWalk (Perozzi et al., 2014) and node2vec (Grover & Leskovec, 2016) , attempt to embed a graph based on its topology without incorporating node attribute information, which limits their embedding power.

Later, graph convolutional networks (GCN) are developed with the basic notion that node embeddings should be smooth over the graph (Kipf & Welling, 2016) .

While GCN leverages both topology and node attribute information for simplified graph convolution in each layer, it may suffer from high frequency noise in the initial node features, which compromises the embedding quality (Maehara, 2019) .

On the other hand, few embedding algorithms can scale well to large graphs with millions of nodes due to their high computation and storage cost (Zhang et al., 2018a) .

For example, graph neural networks (GNNs) such as GraphSAGE (Hamilton et al., 2017) collectively aggregate feature information from the neighborhood.

When stacking multiple GNN layers, the final embedding vector of a node involves the computation of a large number of intermediate embeddings from its neighbors.

This will not only drastically increase the number of computations among nodes but also lead to high memory usage for storing the intermediate results.

In literature, increasing the accuracy and improving the scalability of graph embedding methods are largely viewed as two orthogonal problems.

Hence most research efforts are devoted to addressing only one of the problems.

For instance, and Fu et al. (2019) proposed multi-level methods to obtain high-quality embeddings by training unsupervised models at every level; but their techniques do not improve scalability due to the additional training overhead.

Liang et al. (2018) developed a heuristic algorithm to coarsen the graph by merging nodes with similar local structures.

They use GCN to refine the embedding results on the coarsened graphs, which not only is timeconsuming to train but may also degrade accuracy when multiple GCN layers are stacked together.

More recently, Akbas & Aktas (2019) proposed a similar strategy to coarsen the graph, where certain properties of the graph structure are preserved.

However, this work lacks proper refinement methods to improve the embedding quality.

In this paper we propose GraphZoom, a multi-level spectral approach to enhancing the quality and scalability of unsupervised graph embedding methods.

Specifically, GraphZoom consists of four kernels: (1) graph fusion, (2) spectral graph coarsening, (3) graph embedding, and (4) embedding refinement.

More concretely, graph fusion first converts the node feature matrix into a feature graph and then fuses it with the original topology graph.

The fused graph provides richer information to the ensuing graph embedding step to achieve a higher accuracy.

Spectral graph coarsening produces a series of successively coarsened graphs by merging nodes based on their spectral similarities.

We show that our coarsening algorithm can efficiently and effectively retain the first few eigenvectors of the graph Laplacian matrix, which is critical for preserving the key graph structures.

During the graph embedding step, any of the existing unsupervised graph embedding techniques can be applied to obtain node embeddings for the graph at the coarsest level.

1 Embedding refinement is then employed to refine the embeddings back to the original graph by applying a proper graph filter to ensure embeddings are smoothed over the graph.

We validate the proposed GraphZoom framework on three transductive benchmarks: Cora, Citeseer and Pubmed citation networks as well as two inductive dataset: PPI and Reddit for vertex classification task.

We further test on friendster dataset which contains 8 million nodes and 400 million edges to show the scalability of GraphZoom.

Our experiments show that GraphZoom can improve the classification accuracy over all baseline embedding methods for both transductive and inductive tasks.

Our main technical contributions are summarized as follows:

??? GraphZoom generates high-quality embeddings.

We propose novel algorithms to encode graph structures and node attribute information in a fused graph and exploit graph filtering during refinement to remove high frequency noise.

This results in an increase of the embedding accuracy over the prior arts by up to 19.4%.

??? GraphZoom improves scalability.

Our approach can significantly reduce the embedding run time by effectively coarsening the graph without losing the key spectral properties.

Experiments show that GraphZoom can accelerate the entire embedding process by up to 40.8x while producing a similar or better accuracy than state-of-the-art techniques.

??? GraphZoom is highly composable.

Our framework is agnostic to underlying graph embedding techniques.

Any of the existing unsupervised embedding methods, either transductive or inductive, can be incorporated by GraphZoom in a plug-and-play manner.

GraphZoom draws inspiration from multi-level graph embedding and graph filtering to boost the performance and speed of unsupervised embedding methods.

Multi-level graph embedding.

Multi-level graph embedding attempts to coarsen the graph in a series of levels where graph embedding techniques can be applied on those coarsened graphs with decreasing size.

Lin et al. (2019) coarsen the graph into several levels and then perform embedding on the hierarchy of graphs from the coarsest to the original one.

Fu et al. (2019) adopt a similar idea by hierarchical sampling original graph into multi-level graphs whose embedding vectors are concatenated to obtain the final node embeddings of the original graph.

Both of these works only focus on improving embedding quality without improving the scalability.

Later, Zhang et al. (2018b) ; Akbas & Aktas (2019) attempt to improve graph embedding scalability by only embedding on the coarsest graph.

However, their approaches lack proper refinement methods to generate high-quality embeddings of the original graph.

Liang et al. (2018) propose MILE, which only trains the coarsest graph to obtain coarse embeddings, and leverages GCN as embeddings refinement method to improve embedding quality.

Nevertheless, MILE requires to train a GCN model which is very time consuming for large graphs and cannot support inductive embedding models due to the transductive property of GCN.

In contrast to the prior multi-level graph embedding techniques, GraphZoom is a simple yet theoretically motivated spectral approach to improving embedding quality as well as scalability of unsupervised graph embedding models.

Graph filtering.

Graph filters are direct analogs of classical filters in signal processing field, but intended for signals defined on graphs.

Shuman et al. (2013) defined graph filters in both vertex and spectral domains, and applies graph filter in image denoising and reconstruction tasks.

Recently, Maehara (2019) showed the fundamental link between graph embedding and filtering by proving that GCN model implicitly exploits graph filter to remove high frequency noise from the node feature matrix; a filter neural network (gfNN) is then proposed to derive a stronger graph filter to improve the embedding results.

further derived two generalized graph filters and apply them on graph embedding models to improve their embedding quality for various classification tasks.

3 GRAPHZOOM FRAMEWORK Figure 1 shows the proposed GraphZoom framework which consists of four key phases: Phase (1) is graph fusion, which constructs a weighted graph that fuses the information of both the graph topology and node attributes; In Phase (2), a spectral graph coarsening process is applied to form a hierarchy of coarsened fused graphs with decreasing size; In Phase (3), any of the prior graph embedding methods can be applied to the fused graph at the coarsest level; In Phase (4), the embedding vectors obtained at the coarsest level are mapped onto a finer graph using the mapping operators determined during the coarsening phase.

This is followed by a refinement (smoothing) procedure; by iteratively applying Phase (4) to increasingly finer graphs, the embedding vectors for the original graph can be eventually obtained.

In the rest of this section, we describe each of these four phases in more detail.

Graph w/ Node Embeddings Figure 1 : Overview of the GraphZoom framework.

Graph fusion aims to construct a weighted graph that has the same number of nodes as the original graph but potentially different set of edges (weights) that encapsulate the original graph topology as well as node attribute information.

Specifically, given an undirected graph G = (V, E) with N nodes, its adjacency matrix A topo ??? R N ??N and its node attribute (feature) matrix X ??? R N ??K , where K corresponds to the dimension of node attribute vector, graph fusion can be interpreted as a function f (??) that outputs a weighted graph G f usion = (V, E f usion ) represented by its adjacency matrix A f usion ??? R N ??N , namely, A f usion = f (A topo , X).

Graph fusion first converts the initial attribute matrix X into a weighted node attribute graph G f eat = (V, E f eat ) by generating a k-nearest-neighbor (kNN) graph based on the l 2 -norm distance between the attribute vectors of each node pair.

Note that a straightforward implementation requires comparing all possible node pairs and then selecting top-k nearest neighbors.

However, such a na??ve approach has a worst-case time complexity of O(N 2 ), which certainly does not scale to large graphs.

To allow constructing the attribute graph in linear time, we leverage our O(|E|) complexity spectral graph coarsening scheme described with details in Section 3.2.

More specifically, our approach starts with coarsening the original graph G to obtain a substantially reduced graph that has much fewer nodes.

Note that such a procedure is very similar to spectral graph clustering, which aims to group nodes into clusters of high conductance (Peng et al., 2015) .

Once such node clusters are formed through spectral coarsening, selecting the top-k nearest neighbors within each cluster can be accomplished in O(M 2 ), where M is the averaged node count within the same cluster.

Since we have roughly N/M clusters, the total run time for constructing the approximate kNN graph becomes O(M N ).

When a proper coarsening ratio (M N ) is chosen, say M = 50, the overall run time complexity will become almost linear.

For each edge in the attribute graph, we assign its weight w i,j according to the cosine similarity of two nodes' attribute vectors: w i,j = (X i,: ?? X j,: )/( X i,: X j,: ), where X i,: and X j,: are the attribute vectors of node i and j. Finally, we can construct the fused graph by combining the topological graph and the attribute graph: A f usion = A topo + ??A f eat , where ?? allows us to balance the graph topological and node attribute information in the fusion process.

The fused graph will enable the underlying graph embedding model to utilize both graph topological and node attribute information, and thus can be fed into any downstream graph embedding procedures to further improve embedding quality.

Graph coarsening via global spectral embedding.

To reduce the size of the original graph while preserving important spectral properties (e.g., the first few eigenvalues and eigenvectors of the graph Laplacian matrix 2 ), a straightforward way is to first embed the graph into a k-dimensional space using the first k eigenvectors of the graph Laplacian matrix, which is also known as the spectral graph embedding technique (Belkin & Niyogi, 2003; Peng et al., 2015) .

Next, the graph nodes that are close to each other in the low-dimensional embedding space can be aggregated to form the coarse-level nodes and subsequently the reduced graph.

However, it will be very costly to calculate the eigenvectors of the original graph Laplacian, especially for very large graphs.

Graph coarsening via local spectral embedding.

In this work, we leverage an efficient yet effective local spectral embedding scheme to identify node clusters based on emerging graph signal processing techniques (Shuman et al., 2013) .

There are obvious analogies between the traditional signal processing (Fourier analysis) and graph signal processing: (1) The signals at different time points in classical Fourier analysis correspond to the signals at different nodes in an undirected graph; (2) The more slowly oscillating functions in time domain correspond to the graph Laplacian eigenvectors associated with lower eigenvalues or the more slowly varying (smoother) components across the graph.

Instead of directly using the first few eigenvectors of the original graph Laplacian, we apply the simple smoothing (low-pass graph filtering) function to k random vectors to obtain smoothed vectors for k-dimensional graph embedding, which can be achieved in linear time.

Consider a random vector (graph signal) x that can be expressed with a linear combination of eigenvectors u of the graph Laplacian.

Low-pass graph filters can be adopted to quickly filter out the "high-frequency" components of the random graph signal or the eigenvectors corresponding to high eigenvalues of the graph Laplacian.

By applying the smoothing function on x, a smoothed vectorx can be obtained, which can be considered as a linear combination of the first few eigenvectors:

More specifically, we apply a few (e.g. five to ten) Gauss-Seidel iterations for solving the linear system of equations L G x (i) = 0 to a set of t initial random vectors T = (x (1) , . . .

, x (t) ) that are orthogonal to the all-one vector 1 satisfying 1 x (i) = 0, and L G is the Laplacian matrix of graph G or G f usion .

Based on the smoothed vectors in T , each node is embedded into a t-dimensional space such that nodes p and q are considered spectrally similar if their low-dimensional embedding vectors x p ??? R t and x q ??? R t are highly correlated.

Here the node distance is measured by the spectral node affinity a p,q for neighboring nodes p and q (Livne & Brandt, 2012; Chen & Safro, 2011) :

Once the node aggregation schemes are determined, the graph mapping operators on each level (

can be obtained and leveraged for constructing a series of spectrally-reduced

We emphasize that the aggregation scheme based on the above spectral node affinity calculations will have a (linear) complexity of O(|E f usion |) and thus allow preserving the spectral (global or structural) properties of the original graph in a highly efficient and effective way.

As suggested in (Zhao & Feng, 2019; Loukas, 2019) , a spectral sparsification procedure can be applied to effectively control densities of coarse level graphs.

In this work, a similarity-aware spectral sparsification tool "GRASS" (Feng, 2018) has been adopted for achieving a desired graph sparsity at the coarsest level.

Embedding the Coarsest Graph.

Once the coarsest graph G m is constructed, node embeddings E m on G m can be obtained by E m = l(G m ), where l(??) can be any unsupervised embedding methods.

Once the base node embedding results are available, we can easily project the node embeddings from graph G i+1 to the fine-grained graph G i with the corresponding projection operator H i i+1 :

Due to the property of the projection operator, embedding of the node in coarse-grained graph will be directly copied to the nodes of the same aggregation set in the fine-grained graph.

In this case, spectrally-similar nodes in the fine-grained graph will have the same embedding results if they are aggregated into a single node during the coarsening phase.

To further improve the quality of the mapped embeddings, we apply a local refinement process motivated by Tikhonov regularization to smooth the node embeddings over the graph by minimizing the following objective: min

where L i and E i are the normalized Laplacian matrix and mapped embedding matrix of the graph at the i-th coarsening level, respectively.

The refined embedding matrix??? i is obtained by solving Eq. (5), whose first term enforces the refined embeddings to agree with mapped embeddings while the second term employs Laplacian smoothing to smooth??? i over the graph.

By taking the derivative of the objective function in Eq. (5) and setting it to zero, we have:

where I is the identity matrix.

However, obtaining refined embeddings in this way is very time consuming since it involves matrix inversion whose time complexity is O(N 3 ).

Instead, we exploit a more efficient graph filter to smooth the embeddings.

Let the term (I + L) ???1 denoted by h(L), then its corresponding graph filter in spectral domain is h(??) = (1 + ??) ???1 .

To avoid the inversion term, we approximate h(??) by its first-order Taylor expansion, namely,h(??) = 1 ??? ??.

We then generaliz???

k , where k controls the power of graph filter.

After transformingh

where A is the adjacency matrix and D is the degree matrix.

It can be proved that adding a proper self-loop for every node in the graph can enableh k (L) to more effectively filter out high-frequency noise components (Maehara, 2019) (more details are available in Appendix G).

Thus, we modify the adjacency matrix as?? = A + ??I, where ?? is a small value to ensure every node has its own self-loop.

Finally, the low-pass graph filter can be utilized to smooth the mapped embedding matrix, as shown in (7).

We iteratively apply Eq. (7) to obtain the embeddings of the original graph (i.e., E 0 ).

Note that our refinement stage does not involve training and can be simply considered as several (sparse) matrix multiplications, which can be computed efficiently.

We have performed comparative evaluation of GraphZoom framework against several existing stateof-the-art unsupervised graph embedding techniques and multi-level embedding frameworks on five standard graph-based dataset (transductive as well as inductive).

In addition, we evaluate the scalability of GraphZoom on Friendster dataset that contains 8 million nodes and 400 million edges.

Finally, we further analyze GraphZoom kernels separately to show their effectiveness.

Datasets.

The statistics of datasets used in our experiments are demonstrated in Table 1 .

We use Cora, Citeseer, Pubmed, Friendster for transductive task and PPI, Reddit for inductive task.

We choose the same training and testing size used in Kipf & Welling (2016) ; Hamilton et al. (2017) .

Transductive baseline models.

Many existing graph embedding techniques are essentially transductive learning methods, which require all nodes in the graph be present during training, and their embedding models have to be retrained whenever a new node is added.

We compare GraphZoom with transductive models DeepWalk, node2vec, and Deep Graph Infomax (DGI) (Velikovi et al., and MILE (Liang et al., 2018) , which have shown improvement upon DeepWalk and node2vec in either embedding quality or scalability.

Inductive baseline models.

Inductive graph embedding models can be trained without seeing the whole graph structure and their trained models can be applied on new nodes added to graph.

To show GraphZoom can also enhance inductive learning, we compare it against GraphSAGE (Hamilton et al., 2017) using four different aggregation functions.

More details of datasets and baselines are available in Appendix A and B. We optimize hyperparameters of DeepWalk, node2vec, DGI (Velikovi et al., 2019) and GraphSAGE on original datasets as embedding baseline, and then we choose the same hyper-parameters to embed coarsened graph in HARP, MILE and our GraphZoom framework.

We run all the experiments on a machine running Linux with an Intel Xeon Gold 6242 CPU (32 cores, 2.40GHz) and 384 GB of RAM.

Tables 2 and 3 , respectively.

We report the mean classification accuracy for transductive task and micro-averaged F1 score for inductive task as well as CPU time after 10 runs for all the baselines and GraphZoom.

We measure the CPU time for graph embedding as the total run time of DeepWalk, node2vec, DGI, and GraphSAGE.

We use the sum of CPU time for graph coarsening, graph embedding, and embedding refinement as total run time of HARP and MILE.

Similarly, we sum up the CPU time for graph fusion, graph coarsening, graph embedding, and embedding refinement as total run time of GraphZoom.

We also perform fine-tuning on the hyper-parameters.

For both DeepWalk and node2vec, we use 10 walks with a walk length of 80, a window size of 10, and an embedding dimension of 128; we further set p = 1 and q = 0.5 in node2vec.

For DGI, we choose early stopping strategy with a learning rate of 0.001, an embedding dimension of 512.

For GraphSAGE, we train a two-layer model for one epoch, with a learning rate of 0.00001, an embedding dimension of 128, and a batch size of 256.

Comparing GraphZoom with baseline embedding methods.

We show the results of GraphZoom with coarsening level varying 1 to 3 for transductive learning and 1 to 2 for inductive learning.

Results with larger coarsening level are available in Figure 3 (blue curve) and the Appendix I. Our results demonstrate that GraphZoom is agnostic to underlying embedding methods and capable of boosting the accuracy and speed of state-of-the-art unsupervised embedding methods on various datasets.

More specifically, for transductive learning task, GraphZoom improves classification accuracy upon both DeepWalk and node2vec by a margin of 8.3%, 19.4%, and 10.4% on Cora, Citeseer, and Pubmed, respectively, while achieving up to 40.8x run-time reduction.

In regards to comparing with DGI, GraphZoom achieves comparable or better accuracy with speedup up to 11.2??.

Similarly, GraphZoom outperforms all the baselines by a margin of 3.4% and 3.3% on PPI and Reddit for inductive learning task, respectively, with speedup up to 7.6x.

Our results indicate that reducing graph size while properly retaining the key spectral properties of graph Laplacian and smoothing embeddings will not only boost the embedding speed but also lead to high embedding quality.

Comparing GraphZoom with multi-level frameworks.

As shown in Table 2 , HARP only slightly improves and sometimes even worsens the classification accuracy while significantly increasing the CPU time.

Although MILE improves both accuracy and CPU time compared to baseline embedding methods in some cases, the performance of MILE becomes worse with increasing coarsening levels (e.g., the classification accuracy of MILE drops from 0.708 to 0.618 on Pubmed dataset with node2vec as the embedding kernel).

GraphZoom achieves a better accuracy and speedup compared to MILE with the same coarsening level across all datasets.

Moreover, when increasing coarsening levels, namely, decreasing number of nodes on the coarsened graph, GraphZoom still produces comparable or even a better embedding accuracy with much shorter CPU times.

This further confirms GraphZoom can retain the key graph structure information to be utilized by underlying embedding models to generate high-quality node embeddings.

More results of GraphZoom on non-attributed graph for both node classification and link prediction tasks are available in Appendix J.

GraphZoom for large graph embedding.

To show GraphZoom can significantly improve performance and scalability of underlying embedding model on large graph, we test GraphZoom and MILE on Friendster dataset, which contains 8 million nodes and 400 million edges, using DeepWalk as the embedding kernel.

As shown in Figure 2 , GraphZoom drastically boosts the Micro-F1 score up to 47.6% compared to MILE and 49.9% compared to DeepWalk with speedup up to 119.8??.

When increasing coarsening level, GraphZoom achieves a higher speedup while the embedding accuracy decreases gracefully, which shows the key strength of GraphZoom: it can effectively coarsen large graph by merging many redundant nodes that are spectrally similar, which preserves the most important graph spectral (structural) properties key to underlying embedding model.

When applying basic embedding model on coarsest graph, it can learn more global information from spectral domain, leading to high-quality node embeddings.

On the contrary, heuristic graph coarsening algorithm used in MILE fails to preserve a meaningful coarsest graph, especially when coarsening graph by a large reduction ratio.

Comparisons of different kernel combinations in GraphZoom and MILE in classification accuracy on Cora, Citeseer, and Pubmed datasets -We choose DeepWalk (DW) as the embedding kernel.

GZoom F, GZoom C, GZoom R denote the fusion, coarsening, and refinement kernels proposed in GraphZoom, respectively; MILE C and MILE R denote the coarsening and refinement kernels in MILE, respectively; The blue curve is basically GraphZoom and the yellow one is MILE.

To study the effectiveness of our proposed GraphZoom kernels separately, we compare each of them against the corresponding kernel in MILE with other kernels fixed.

As shown in Figure 3 , when fixing coarsening kernel and comparing refinement kernel of GraphZoom with that of MILE (shown in purple curve and yellow curve), GraphZoom refinement kernel can improve embedding results upon MILE refinement kernel, especially when the coarsening level is large, which indicates that our proposed graph filter in refinement kernel can successfully filter out high frequency noise from graph to improve embedding quality.

Similarly, when comparing coarsening kernels in GraphZoom and MILE with refinement kernel fixed (shown in light blue curve and yellow curve), GraphZoom coarsening kernel can also improve embedding quality upon MILE coarsening kernel, which shows that our spectral graph coarsening algorithm can indeed retain key graph structure for underlying graph embedding models to exploit.

When combining GraphZoom coarsening kernel and refinement kernel (green curve), we can achieve better classification accuracy compared with the ones using any kernel in MILE (i.e., light blue curve, purple curve and yellow curve), which means that GraphZoom coarsening kernel and refinement kernel play different roles to boost embedding performance and their combination can further improve embedding result.

Moreover, when adding graph fusion kernel with the combination of GraphZoom coarsening and refinement kernels (blue curve, which is our GraphZoom framework), it improves classification accuracy by a large margin, which betokens that graph fusion can properly incorporate both graph topology and node attribute information and lifts the embedding quality of downstream embedding models.

Results of each kernel CPU time and speedup comparison are available in Appendix F and Appendix H.

In this work we propose GraphZoom, a multi-level framework to improve embedding quality and scalability of underlying unsupervised graph embedding techniques.

GraphZoom encodes graph structure and node attribute in a single graph and exploiting spectral coarsening and refinement methods to remove high frequency noise from the graph.

Experiments show that GraphZoom improves both classification accuracy and embedding speed on a number of popular datasets.

An interesting direction for future work is to derive a proper way to propagate node labels to the coarsest graph, which would allow GraphZoom to support supervised graph embedding models.

Transductive task.

We follow the experiments setup in Yang et al. (2016) for three standard citation network benchmark datasets: Cora, Citeseer, and Pubmed.

In all these three citation networks, nodes represent documents and edges correspond to citations.

Each node has a sparse bag-of-word feature vector and a class label.

We allow only 20 labels per class for training and 1, 000 labeled nodes for testing.

In addition, we further evaluate on Friendster dataset (Yang & Leskovec, 2015) , which contains 8 million nodes and 400 million edges, with 2.5% of the nodes used for training and 0.3% nodes for testing.

In Friendster, nodes represent users and a pair of nodes are linked if they are friends; each node has a class label but is not associated with a feature vector.

Inductive task.

We follow Hamilton et al. (2017) for setting up experiments on both proteinprotein interaction (PPI) and Reddit dataset.

PPI dataset consists of graphs corresponding to human tissues, where nodes are proteins and edges represent interaction effects between proteins.

Reddit dataset contains nodes corresponding to users' posts: two nodes are connected through an edge if the same users comment on both posts.

We use 60% nodes for training, 40% for testing on PPI and 65% for training and 35% for testing on Reddit.

DeepWalk first generates random walks based on graph structure.

Then, walks are treated as sentences in a language model and Skip-Gram model is exploited to obtain node embeddings.

node2vec is different from DeepWalk in terms of generating random walks by introducing the return parameter p and the in-out parameter q, which can combine DFS-like and BFS-like neighborhood exploration.

Deep Graph Infomax (DGI) is an unsupervised approach that generates node embeddings by maximizing mutual information between patch representations (local information) and corresponding high-level summaries (global information) of graphs .

GraphSAGE embeds nodes in an inductive way by learning an aggregation function that aggregates node features to obtain embeddings.

GraphSAGE supports four different aggregation functions: GraphSAGE-GCN, GraphSAGE-mean, GraphSAGE-LSTM and GraphSAGE-pool.

HARP coarsens the original graph into several levels and apply underlying embedding model to train the coarsened graph at each level sequentially to obtain the final embeddings on original graph.

Since the coarsening level is fixed in their implementation, we run HARP in our experiments without changing the coarsening level.

MILE is the state-of-the-art multi-level unsupervised graph embedding framework and similar to our GraphZoom framework since it also contains graph coarsening and embedding refinement kernels.

More specifically, MILE first uses its heuristic-based coarsening kernel to reduce the graph size and trains underlying unsupervised graph embedding model on coarsest graph.

Then, its refinement kernel employs Graph Convolutional Network (GCN) to refine embeddings on the original graph.

We compare GraphZoom with MILE on various datasets, including Friendster that contains 8 million nodes and 400 million edges (shown in Table 2 and Figure 2) .

Moreover, we further compare each kernel in GraphZoom and MILE in Figure 3 .

The details of graph size at different coarsening level on all six datasets are shown in Table 4 .

APPENDIX E SPECTRAL COARSENING Note that the mapping operator H i+1 i ??? {0, 1} |Vi+1|??|Vi| is a matrix containing only 0s and 1s.

It has following properties:

??? The row (column) index of H i+1 i corresponds to the node index in graph G i+1 (G i ).

???

It is a surjective mapping of the node set, where (H i+1 i ) p,q = 1 if node q in graph G i is aggregated to super-node p in graph G i+1 , and (H i+1 i ) p ,q = 0 for all nodes p ??? {v ??? V i+1 : v = p}.

??? It is a locality-preserving operator, where the coarsened version of G i induced by the nonzero entries of (H i+1 i ) p,: is connected for each p ??? V i+1 .

Algorithm 2: spectral coarsening algorithm Input: Adjacency matrix A i ??? R |Vi|??|Vi| Output: Adjacency matrix A i+1 ??? R |Vi+1|??|Vi+1| of the reduced graph G i+1 , mapping operator H i+1 i ??? R |Vi+1|??|Vi| 11 n = |V i |, n c = n; 12 [graph reduction ratio]

?? max = 1.8 , ?? = 0.9; 13 for each edge (p, q) ??? E i do 14 [spectral node affinity set] C ??? a p,q defined in Eq. 2 ; 15 end 16 for each node p ??? V i do 17 Figure 4 (note that the y axis is in logarithmic scale), the GraphZoom embedding kernel dominates the total CPU time, which can be more effectively reduced with a greater coarsening level L. All other kernels in GraphZoom are very efficient, which enable the GraphZoom framework to drastically reduce the total graph embedding time.

Figure 5a shows the original distribution of graph Laplacian eigenvalues which also can be interpreted as frequencies in graph spectral domain (smaller eigenvalue means lower frequency).

The proposed graph filter for embedding refinement (as shown in Figure 5e ) can be considered as a bandstop filter that passes all frequencies with the exception of those within the middle stop band that is greatly attenuated.

Therefore, the band-stop filter may not be very effective for removing highfrequency noises from the graph signals.

Fortunately, it has been shown that by adding self-loops to each node in the graph as follows?? = A+??I (shown in Figure 5b , 5c, 5d, where ?? = 0.5, 1.0, 2.0), the distribution of Laplacian eigenvalues can be squeezed to the left (towards zero) (Maehara, 2019) .

By properly choosing ?? such that large eigenvalues will mostly lie in the stop band (e.g., ?? = 1.0, 2.0 shown in Figure 5c and 5d), the graph filter will be able to effectively filtered out high-frequency components (corresponding to high eigenvalues) while retaining low-frequency components, which is similar to a low-pass graph filter as shown in Figure 5f .

It is worth noting that if ?? is too large, then most eigenvalues will be very close to zero, which makes the graph filter less effective for removing noises.

In this work, we choose ?? = 2.0 for all our experiments.

As shown in Figure 6 , the combination of GraphZoom coarsening and refinement kernels can always achieve the greatest speedups (green curves); adding GraphZoom fusion kernel (blue curves) will lower the speedups by a small margin but further boost the embedding quality, showing a clear tradeoff between embedding quality and runtime efficiency: to achieve the highest graph embedding quality, the graph fusion kernel should be included.

To further show that GraphZoom can work on non-attributed datasets, we evaluate it on PPI(Homo Sapiens) and Wiki datasets, following the same dataset configuration as used in Grover & Leskovec (2016) ; Liang et al. (2018) .

As shown in Table 5 , GraphZoom (without fusion kernel) improves

<|TLDR|>

@highlight

A multi-level spectral approach to improving the quality and scalability of unsupervised graph embedding.
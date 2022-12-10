Inductive and unsupervised graph learning is a critical technique for predictive or information retrieval tasks where label information is difficult to obtain.

It is also challenging to make graph learning inductive and unsupervised at the same time, as learning processes guided by reconstruction error based loss functions inevitably demand graph similarity evaluation that is usually computationally intractable.

In this paper, we propose a general framework SEED (Sampling, Encoding, and Embedding Distributions) for inductive and unsupervised representation learning on graph structured objects.

Instead of directly dealing with the computational challenges raised by graph similarity evaluation, given an input graph, the SEED framework samples a number of subgraphs whose reconstruction errors could be efficiently evaluated, encodes the subgraph samples into a collection of subgraph vectors, and employs the embedding of the subgraph vector distribution as the output vector representation for the input graph.

By theoretical analysis, we demonstrate the close connection between SEED and graph isomorphism.

Using public benchmark datasets, our empirical study suggests the proposed SEED framework is able to achieve up to 10% improvement, compared with competitive baseline methods.

Representation learning has been the core problem of machine learning tasks on graphs.

Given a graph structured object, the goal is to represent the input graph as a dense low-dimensional vector so that we are able to feed this vector into off-the-shelf machine learning or data management techniques for a wide spectrum of downstream tasks, such as classification (Niepert et al., 2016) , anomaly detection (Akoglu et al., 2015) , information retrieval (Li et al., 2019) , and many others (Santoro et al., 2017b; Nickel et al., 2015) .

In this paper, our work focuses on learning graph representations in an inductive and unsupervised manner.

As inductive methods provide high efficiency and generalization for making inference over unseen data, they are desired in critical applications.

For example, we could train a model that encodes graphs generated from computer program execution traces into vectors so that we can perform malware detection in a vector space.

During real-time inference, efficient encoding and the capability of processing unseen programs are expected for practical usage.

Meanwhile, for real-life applications where labels are expensive or difficult to obtain, such as anomaly detection (Zong et al., 2018) and information retrieval (Yan et al., 2005) , unsupervised methods could provide effective feature representations shared among different tasks.

Inductive and unsupervised graph learning is challenging, even compared with its transductive or supervised counterparts.

First, when inductive capability is required, it is inevitable to deal with the problem of node alignment such that we can discover common patterns across graphs.

Second, in the case of unsupervised learning, we have limited options to design objectives that guide learning processes.

To evaluate the quality of learned latent representations, reconstruction errors are commonly adopted.

When node alignment meets reconstruction error, we have to answer a basic question: Given two graphs G 1 and G 2 , are they identical or isomorphic (Chartrand, 1977) ?

To this end, it could be computationally intractable to compute reconstruction errors (e.g., using graph edit distance (Zeng et al., 2009) as the metric) in order to capture detailed structural information.

Given an input graph, its vector representation can be obtained by going through the components.

Previous deep graph learning techniques mainly focus on transductive (Perozzi et al., 2014) or supervised settings (Li et al., 2019) .

A few recent studies focus on autoencoding specific structures, such as directed acyclic graphs (Zhang et al., 2019) , trees or graphs that can be decomposed into trees (Jin et al., 2018) , and so on.

From the perspective of graph generation, You et al. (2018) propose to generate graphs of similar graph statistics (e.g., degree distribution), and Bojchevski et al.

(2018) provide a GAN based method to generate graphs of similar random walks.

In this paper, we propose a general framework SEED (Sampling, Encoding, and Embedding Distributions) for inductive and unsupervised representation learning on graph structured objects.

As shown in Figure 1 , SEED consists of three major components: subgraph sampling, subgraph encoding, and embedding subgraph distributions.

SEED takes arbitrary graphs as input, where nodes and edges could have rich features, or have no features at all.

By sequentially going through the three components, SEED outputs a vector representation for an input graph.

One can further feed such vector representations to off-the-shelf machine learning or data management tools for downstream learning or retrieval tasks.

Instead of directly addressing the computational challenge raised by evaluation of graph reconstruction errors, SEED decomposes the reconstruction problem into the following two sub-problems.

Q1: How to efficiently autoencode and compare structural data in an unsupervised fashion?

SEED focuses on a class of subgraphs whose encoding, decoding, and reconstruction errors can be evaluated in polynomial time.

In particular, we propose random walks with earliest visiting time (WEAVE) serving as the subgraph class, and utilize deep architectures to efficiently autoencode WEAVEs.

Note that reconstruction errors with respect to WEAVEs are evaluated in linear time.

Q2: How to measure the difference of two graphs in a tractable way?

As one subgraph only covers partial information of an input graph, SEED samples a number of subgraphs to enhance information coverage.

With each subgraph encoded as a vector, an input graph is represented by a collection of vectors.

If two graphs are similar, their subgraph distribution will also be similar.

Based on this intuition, we evaluate graph similarity by computing distribution distance between two collections of vectors.

By embedding distribution of subgraph representations, SEED outputs a vector representation for an input graph, where distance between two graphs' vector representations reflects the distance between their subgraph distributions.

Unlike existing message-passing based graph learning techniques whose expressive power is upper bounded by Weisfeiler-Lehman graph kernels (Xu et al., 2019; Shervashidze et al., 2011) , we show the direct relationship between SEED and graph isomorphism in Section 3.5.

We empirically evaluate the effectiveness of the SEED framework via classification and clustering tasks on public benchmark datasets.

We observe that graph representations generated by SEED are able to effectively capture structural information, and maintain stable performance even when the node attributes are not available.

Compared with competitive baseline methods, the proposed SEED framework could achieve up to 10% improvement in prediction accuracy.

In addition, SEED achieves high-quality representations when a reasonable number of small subgraph are sampled.

By adjusting sample size, we are able to make trade-off between effectiveness and efficiency.

Kernel methods.

Similarity evaluation is one of the key operations in graph learning.

Conventional graph kernels rely on handcrafted substructures or graph statistics to build vector representations for graphs (Borgwardt & Kriegel, 2005; Kashima et al., 2003; Vishwanathan et al., 2010; Horváth et al., 2004; Shervashidze & Borgwardt, 2009; Kriege et al., 2019) .

Although kernel methods are potentially unsupervised and inductive, it is difficult to make them handle rich node and edge attributes in many applications, because of the rigid definition of substructures.

Deep learning.

Deep graph representation learning suggests a promising direction where one can learn unified vector representations for graphs by jointly considering both structural and attribute information.

While most of existing works are either transductive (Perozzi et al., 2014; Kipf & Welling, 2016; Liu et al., 2018) or supervised settings (Scarselli et al., 2008; Battaglia et al., 2016; Defferrard et al., 2016; Duvenaud et al., 2015; Kearnes et al., 2016; Veličković et al., 2018; Santoro et al., 2017a; Xu et al., 2018; Hamilton et al., 2017) , a few recent studies focus on autoencoding specific structures, such as directed acyclic graphs (Zhang et al., 2019) , trees or graphs that can be decomposed into trees (Jin et al., 2018) , and so on.

In the case of graph generation, You et al. (2018) propose to generate graphs of similar graph statistics (e.g., degree distribution), and Bojchevski et al.

(2018) provide a method to generate graphs of similar random walks.

In addition, Li et al. (2019) propose a supervised method to learn graph similarity, and Xu et al. (2019) theoretically analyses the expressive power of existing message-passing based graph neural networks.

Unlike existing kernel or deep learning methods, our SEED framework is unsupervised with inductive capability, and naturally supports complex attributes on nodes and edges.

Moreover, it works for arbitrary graphs, and provides graph representations that simultaneously capture both structural and attribute information.

The core idea of SEED is to efficiently encode subgraphs as vectors so that we can utilize subgraph distribution distance to reflect graph similarity.

We first give an abstract overview on the SEED framework in Section 3.1, and then discuss concrete implementations for each component in Section 3.2, 3.3, and 3.4, respectively.

In Section 3.5, we share the theoretical insights in SEED.

For the ease of presentation, we focus on undirected graphs with rich node attributes in the following discussion.

With minor modification, our technique can also handle directed graphs with rich node and edge attributes.

SEED encodes an arbitrary graph into a vector by the following three major components, as shown in Figure 1 .

• Sampling.

A number of subgraphs are sampled from an input graph in this component.

The design goal of this component is to find a class of subgraphs that can be efficiently encoded and decoded so that we are able to evaluate their reconstruction errors in a tractable way.

• Encoding.

Each sampled subgraph is encoded into a vector in this component.

Intuitively, if a subgraph vector representation has good quality, we should be able to reconstruct the original subgraph well based on the vector representation.

Therefore, the design goal of this component is to find an autoencoding system that provides such encoding functionality.

• Embedding distribution.

A collection of subgraph vector representations are aggregated into one vector serving as the input graph's representation.

For two graphs, their distance in the output vector space approximates their subgraph distribution distance.

The design goal of this component is to find such a aggregation function that preserves a pre-defined distribution distance.

Although there could be many possible implementations for the above three components, we propose a competitive implementation in this paper, and discuss them in details in the rest of this section.

In this paper, we propose to sample a class of subgraphs called WEAVE (random Walk with EArliest Visit timE).

Let G be an input graph of a node set V (G) and an edge set E(G).

A WEAVE of length k is sampled from G as follows.

Figure 2: Expressive power comparison between WEAVEs and vanilla random walks: while blue and orange walks cannot be differentiated in terms of vanilla random walks, the difference under WEAVEs is outstanding.

• Initialization.

A starting node v (0) is randomly drawn from V (G) at timestamp 0, and its earliest visiting time is set to 0.

• Next-hop selection.

Without loss of generality, assume v (p) is the node visited at timestamp p (0 ≤ p < k).

We randomly draw a node v (p+1) from v (p) 's one-hop neighborhood as the node to be visited at timestamp p + 1.

If v (p+1) is a node that we have not visited before, its earliest visiting time is set to p + 1; otherwise, its earliest visiting is unchanged.

We hop to v (p+1) .

• Termination.

The sampling process ends when timestamp reaches k.

In practical computation, a WEAVE is denoted as a matrix

In particular,

t ] is a concatenation of two vectors, where

a includes attribute information for the node visited at timestamp p, and x (p) t contains its earliest visit time.

As earliest visit time is discrete, we use one-hot scheme to represent such information, where x Difference between WEAVEs and vanilla random walks.

The key distinction comes from the information of earliest visit time.

Vanilla random walks include coarser-granularity structural information, such as neighborhood density and neighborhood attribute distribution (Perozzi et al., 2014) .

As vanilla random walks have no memory on visit history, detailed structural information related to loops or circles is ignored.

While it is also efficient to encode and decode vanilla random walk, it is difficult to evaluate finer-granularity structural difference between graphs.

Unlike vanilla random walks, WEAVEs utilize earliest visit time to preserve loop information in sampled subgraphs.

As shown in Figure 2 , while we cannot tell the difference between walk w 1 and walk w 2 using vanilla random walk, the distinction is outstanding under WEAVEs.

Note that it is equally efficient to encode and decode WEAVEs, compared with vanilla random walks.

Given a set of sampled WEAVEs of length k {X 1 , X 2 , ..., X s }, the goal is to encode each sampled WEAVE into a dense low-dimensional vector.

As sampled WEAVEs share same length, their matrix representations also have identical shapes.

Given a WEAVE X, one could encode it by an autoencoder (Hinton & Salakhutdinov, 2006) as follows.

where z is the dense low-dimensional representation for the input WEAVE, f (·) is the encoding function implemented by an MLP with parameters θ e , and g(·) is the decoding function implemented by another MLP with parameters θ d .

The quality of z is evaluated through reconstruction errors as follows,

By conventional gradient descent based backpropagation (Kingma & Ba, 2014) , one could optimize θ e and θ d via minimizing reconstruction error L. After such an autoencoder is well trained, the latent representation z includes both node attribute information and finer-granularity structural information simultaneously.

Given s sampled WEAVEs of an input graph, the output of this component is s dense low-dimensional vectors {z 1 , z 2 , · · · , z s }.

Let G and H be two arbitrary graphs.

Suppose subgraph (e.g., WEAVE) distributions for G and H are P G and P H , respectively.

In this component, we are interested in evaluating the distance between P G and P H .

In this work, we investigate the feasibility of employing empirical estimate of the maximum mean discrepancy (MMD) (Gretton et al., 2012) to evaluate subgraph distribution distances, without assumptions on prior distributions, while there are multiple candidate metrics for distribution distance evaluation, such as KL-divergence (Kullback & Leibler, 1951) and Wasserstein distance (Arjovsky et al., 2017) .

We leave the detailed comparison among different choices of distance metrics in our future work.

Given s subgraphs sampled from G as {z 1 , · · · , z s } and s subgraphs sampled from H as {h 1 , · · · , h s }, we can estimate the distance between P G and P H under the MMD framework:

µ G andμ H are empirical kernel embeddings of P G and P H , respectively, and are defined as follows,

where φ(·) is the implicit feature mapping function with respect to the kernel function k(·, ·).

To this end,μ G andμ H are the output vector representation for G and H, respectively.

In terms of kernel selection, we find the following options are effective in practice.

Identity kernel.

Under this kernel, pairwise similarity evaluation is performed in original input space.

Its implementation is simple, but surprisingly effective in real-life datasets,

where output representations are obtained by average aggregation over subgraph representations.

Commonly adopted kernels.

For popular kernels (e.g., RBF kernel, inverse multi-quadratics kernel, and so on), it could be difficult to find and adopt their feature mapping functions.

While approximation methods could be developed for individual kernels (Ring & Eskofier, 2016) , we could train a deep neural network that approximates such feature mapping functions.

In particular,

whereφ(·; θ m ) is an MLP with parameters θ m , and D(·, ·) is the approximation to the empirical estimate of MMD.

Note thatμ G andμ H are output representations for G and H, respectively.

To train the functionφ(·; θ m ), we evaluate the approximation error by

where θ m is optimized by minimizing J(θ m ).

In this section, we sketch the theoretical connection between SEED and well-known graph isomorphism (Chartrand, 1977) , and show how walk length in WEAVE impacts the effectiveness in graph isomorphism tests.

The full proof of theorems and lemmas is detailed in Appendix.

To make the discussion self-contained, we define graph isomorphism and its variant with node attributes as follows.

Graph isomorphism with node attributes.

be two attributed graphs, where l 1 , l 2 are attribute mapping functions l 1 :

, and node attributes are denoted as d-dimensional vectors.

Then G and H are isomorphic with node attributes if there is a bijection f :

Identical distributions.

Two distributions P and Q are identical if and only if their 1st order Wasserstein distance (Rüschendorf, 1985) is W 1 (P, Q) = 0.

The following theory suggests the minimum walk length for WEAVEs, if every edge in a graph is expected to be visited.

) be a connected graph, then there exists a walk of length k which can visit all the edges of G, where k ≥ 2|E(G)| − 1.

Now, we are ready to present the connection between SEED and graph isomorphism.

) and H = (V (H), E(H)) be two connected graphs.

Suppose we can enumerate all possible WEAVEs from G and H with a fixed-length k ≥ 2 max{|E(G)|, |E(H)|}−1, where each WEAVE has a unique vector representation generated from a well-trained autoencoder.

The Wasserstein distance between G's and H's WEAVE distributions is 0 if and only if G and H are isomorphic.

The following theory shows the connection in the case of graphs with nodes attributes.

be two connected graphs with node attributes.

Suppose we can enumerate all possible WEAVEs on G and H with a fixed-length k ≥ 2 max{|E(G)|, |E(H)|}−1, where each WEAVE has a unique vector representation generated from a well-trained autoencoder.

The Wasserstein distance between G's and H's WEAVE distributions is 0 if and only if G and H are isomorphic with node attributes.

Note that similar results can be easily extended to the cases with both node and edge attributes, and the details can be found in Appendix E.

The theoretical results suggest the potential power of the SEED framework in capturing structural difference in graph data.

As shown above, in order to achieve the same expressive power of graph isomorphism, we need to sample a large number of WEAVEs with a long walk length so that all possible WEAVEs can be enumerated.

The resource demand is impractical.

However, in the empirical study in Section 4, we show that SEED can achieve state-of-the-art performance, when we sample a small number of WEAVEs with a reasonably short walk length.

We employ 7 public benchmark datasets to evaluate the effectiveness of SEED.

In this section, we mainly report the results for two representative datasets Deezer and MUTAG from online social network and chemistry domain.

The detailed descriptions for all the 7 datasets are presented in Appendix F.

• Deezer User-User Friendship Networks (Deezer) (Rozemberczki et al. (2018)) is a social network dataset collected from the music streaming service Deezer.

In this network, nodes are users, edges denote mutual friendships between users, and genre notations are extracted as node features.

In particular, to align this dataset with the SEED framework, for each node u, we generate its ego-graph which consists of the nodes and edges within its 3-hop neighborhood, and the egograph's label is assigned as node u's label (this user's nationality).

•

Three state-of-the-art representative techniques are implemented as baselines in the experiments.

• Graph Sample and Aggregate (GraphSAGE) (Hamilton et al. (2017)) is an inductive graph representation learning approach which can learn the structural information of the nodes.

We evaluate GraphSAGE in its unsupervised setting.

• Graph Matching Network (GMN) (Li et al. (2019)) utilizes graph neural networks to obtain graph representations for graph matching applications.

In particular, we employ its Graph Embedding Networks and deploy graph-based loss functions for unsupervised learning.

• Graph Isomorphism Network (GIN) (Xu et al. (2019)) provides an effective sum-based aggregator for graph representation learning.

We modify its objective for unsupervised learning.

In addition, we focus on downstream tasks, including classification and clustering.

Using the downstream tasks, we evaluate the quality of learned graph representations.

For classification tasks, a simple multi-layer fully connected neural network is built as a classifier, and the prediction accuracy (ACC) is used as the evaluation metric.

For clustering tasks, an effective conventional clustering approach, Normalized Cuts (NCut) (Jianbo Shi & Malik, 2000) , is used to cluster graph representations.

Prediction accuracy (ACC) and Normalized Mutual Information (NMI) (Wu et al., 2009 ) are used as its evaluation metrics.

More details of the baselines and downstream tasks are discussed in Appendix G.

In this section, we discuss the performance of SEED and its baselines in the downstream tasks with respect to Deezer and MUTAG.

The full evaluation results on the 7 datasets is detailed in Appendix G. In this set of experiments, SEED adopts identity kernel in the component of embedding distributions.

Table 1 : Evaluating graph representation quality by classification and clustering tasks

As shown in Table 1 , SEED consistently outperforms the baseline methods in both classification and clustering tasks with up to 0.18 absolute improvement in terms of classification accuracy.

For GIN and GMN, supervision information could be crucial in order to differentiate structural variations.

As GraphSAGE mainly focuses on aggregating feature information from neighbor nodes, it could be difficult for GraphSAGE to extract effective structural information from an unsupervised manner.

In the unsupervised setting, SEED is able to differentiate structural difference at finer granularity and capture rich attribute information, leading to high-quality graph representations with superior performance in downstream tasks.

Walk length and sample numbers are two meta-parameters in the SEED framework.

By adjusting these two meta-parameters, we can make trade-off between effectiveness and computational efficiency.

In the experiment, we empirically evaluate the impact of the two meta-parameters on the Table 3 : Representation quality with different walk lengths MUTAG dataset.

In Table 2 , each row denotes the performance with different sampling numbers (from 25 to 800) while the walk length is fixed to 10.

Moreover, we adjust the walk length from 5 to 25 while sampling number is fixed to 200 in Table 3 .

We can see that the performance of SEED in both classification and clustering tasks increases as there are more subgraphs sampled, especially for the changes from 25 to 200.

Meanwhile, we observe the increasing rates diminish dramatically when sampling number ranges from 200 to 800.

Similarly, the performance of SEED increase as the walk length grows from 5 to 20, and the performance starts to converge when the length goes beyond 20.

Red and blue colors indicate two labels.

We observe that the boundary becomes clearer when sample number or walk length increases.

In this paper, we propose a novel framework SEED (Sampling, Encoding, and Embedding distribution) framework for unsupervised and inductive graph learning.

Instead of directly dealing with the computational challenges raised by graph similarity evaluation, given an input graph, the SEED framework samples a number of subgraphs whose reconstruction errors could be efficiently evaluated, encodes the subgraph samples into a collection of subgraph vectors, and employs the embedding of the subgraph vector distribution as the output vector representation for the input graph.

By theoretical analysis, we demonstrate the close connection between SEED and graph isomorphism.

Our experimental results suggest the SEED framework is effective, and achieves state-of-the-art predictive performance on public benchmark datasets.

Proof.

We will use induction on |E(G)| to complete the proof.

Basic case: Let |E(G)| = 1, the only possible graph is a line graph of length 1.

For such a graph, the walk from one node to another can cover the only edge on the graph, which has length 1 ≥ 2 · 1 − 1.

Induction: Suppose that for all the connected graphs on less than m edges (i.e., |E(G)| ≤ m − 1), there exist a walk of length k which can visit all the edges if k ≥ 2|E(G)| − 1.

Then we will show for any connected graph with m edges, there also exists a walk which can cover all the edges on the graph with length k ≥ 2|E(G)| − 1.

Let G = (V (G), E(G)) be a connected graph with |E(G)| = m. Firstly, we assume G is not a tree, which means there exist a cycle on G. By removing an edge e = (v i , v j ) from the cycle, we can get a graph G on m − 1 edges which is still connected.

This is because any edge on a cycle is not bridge.

Then according to the induction hypothesis, there exists a walk w = v 1 v 2 . . .

v i . . .

v j . . .

v t of length k ≥ 2(m − 1) + 1 which can visit all the edges on G (The walk does not necessarily start from node 1, v 1 just represents the first node appears in this walk).

Next, we will go back to our graph G, as G is a subgraph of G, w is also a walk on G. By replacing the first appeared node v i on walk w with a walk v i v j v i , we can obtain a new walk

As w can cover all the edges on G and the edge e with length k = k + 2 ≥ 2(m − 1) − 1 + 2 = 2m − 1, which means it can cover all the edges on G with length k ≥ 2|E(G)| − 1.

Next, consider graph G which is a tree.

In this case, we can remove a leaf v j and its incident edge e = (v i , v j ) from G, then we can also obtain a connected graph G with |E(G )| = m − 1.

Similarly, according to the induction hypothesis, we can find a walk w = v 1 v 2 . . .

v i . . .

v t on G which can visit all the m − 1 edges of G of length k , where k ≥ 2(m − 1) − 1.

As G is a subgraph of G, any walk on G is also a walk on G including walk w .

Then we can also extend walk w on G by replacing the first appeared v i with a walk v i v j v i , which produce a new walk

w can visit all the edges of G as well as the edge e with length k = k + 2 ≥ 2(m − 1) − 1 + 2 = 2m − 1.

In other words, w can visit all the edges on G with length k ≥ 2|E(G)| − 1.

Now, we have verified our assumption works for all the connected graphs with m edges, hence we complete our proof. (To give an intuition for our proof of lemma 1, we provide an example of 5 edges in Figure 5 Figure 5 (b1) shows an example graph G which is a tree on 5 edges.

By removing the leaf v 4 and its incident edge (v 4 , v 3 ), we can get a tree G with 4 edges (Figure 5 (b2) ).

G has a walk w = v 1 v 2 v 3 v 5 which covers all the edges of G , as w is also a walk on G, by replacing v 3 with v 3 v 4 v 3 in w we can get a walk w = v 1 v 2 v 3 v 4 v 3 v 5 which can cover all the edges of G.

The following lemma is crucial for the proof of Theorem 1.

Lemma 2.

Suppose that w, w are two random walks on graph G and graph H respectively, if the representation of w and w are the same, i.e., r w = r w , the number of the distinct edges on w and w are the same, as well as the number of the distinct nodes on w and w .

Proof.

Let n 1 , n 2 be the number of distinct nodes on w, w respectively, let m 1 , m 2 be the number of distinct edges on w and w respectively.

First, let's prove n 1 = n 2 .

We will prove this by contradiction.

Assume n 1 = n 2 , without loss of generality, let n 1 > n 2 .

According to our encoding rule, the largest number appears in a representation vector is the number of the distinct nodes in the corresponding walk.

Hence, the largest element in vector r w is n 1 while the largest element in vector r w is n 2 .

Thus, r w = r w , which contradicts our assumption.

Therefore, we have n 1 = n 2 .

Next, we will show m 1 = m 2 .

We will also prove this point by contradiction.

Assume m 1 = m 2 , without loss of generality, let m 1 > m 2 .

As we have proved n 1 = n 2 , each edge on w and w will be encoded as a vector like [k 1 , k 2 ] , where k 1 , k 2 ∈ [n 1 ].

A walk consists of edges, hence the representation of a walk is formed by the representation of edges.

Since m 1 > m 2 , which means there exists at least two consecutive element [k 1 , k 2 ] in r w which will not appear in r w , thus r w = r w , which is a contradiction of our assumption.

As a result, we can prove m 1 = m 2 .

Proof.

We will first prove the sufficiency of the theorem, i.e., suppose graphs G = (V (G), E(G)) and H = (V (H), E(H)) are two isomorphic graphs, we will show that the WEAVE's distribution on G and H are the same.

Let A be the set of all the possible walks with length k on G, B be the set of all the possible walks with length k on H. Each element of A and B represents one unique walk on G and H respectively.

As we have assumed a WEAVE is a class of subgraphs, which means a WEAVE may corresponds to multiple unique walks in A or B. Consider a walk w = v 1 v 2 . . .

v i . . .

v t ∈ A (v i represent the ith node appears in the walk), for any edge e = (v i , v j ) on w i , as e ∈ E(G), according to the definition of isomorphism, there exists a mapping f :

If we map each node on w i to graph H, we can get a new walk

, besides, as the length of w i is also k, we have w i ∈ B. Hence, we can define a new mapping g : A → B, s.t.

Next, we will show that g is a bijective mapping.

Firstly, we will show that f is injective.

Suppose g(w 1 ) = g(w 2 ), we want to show w 1 = w 2 .

Assume w 1 = w 2 , there must exists one step i such that

is the ith step of g(w 2 ), thus the walk g(w 1 ) = g(w 2 ), which contradicts our assumption.

Therefore, the assumption is false, we have w 1 = w 2 .

Then we will show that g is surjective, i.e., for any w ∈ B, there exists a w ∈ A such that g(w) = w .

We will also prove this by contradiction, suppose there exists a walk w ∈ B such that we can't find any w ∈ A to make g(w) = w .

Let w = v 1 v 2 . . .

v t , according to the definition of isomorphism, for any edge

, where f −1 represents the inverse mapping of f .

Hence

as w is a walk on graph H with length k. Now consider g(w), based on the mapping rule of g, we need map each node on w via f , i.e.,

which is contradiction to our assumption.

Thus we have proved g is an injective mapping as well as a surgective mapping, then we can conclude that g is a bijective mapping.

Then we will show the WEAVEs' distribution of G and H are the same.

Since in our assumption, |E(G)| is limited, then |A| and |B| are limited, besides, according to our encoding rule, different walks may correspond to one specific WEAVE while each WEAVE corresponds a unique representation vector, thus the number of all the possible representation vectors is limited for both G and H. Thus, the representation vector's distributions P G for graph G and representation's distributions P H for graph H are both discrete distributions.

To compare the similarity of two discrete probability distributions, we can adopt the following equation to compute the Wasserstein distance and check if it is 0.

where W 1 (P, Q) is the Wasserstein distance of probability distribution P and Q, π(i, j) is the cost function and s(i, j) is a distance function, w qj and w pj are the probabilities of q j and p j respectively.

Since we have proved g : A → B is a bijection, besides, according to our encoding rule, g(w) and w will corresponds to the same WEAVE, hence they will share the same representation vector.

As a consequence, for each point (g i , w gi ) (g i corresponds to a representation vector, w gi represents the probability of g i ) in the distribution P G , we can find a point (h i , w hi ) in P H such that g i = h i , and w gi = w hi .

Then consider (11), for P G and P H , if we let π be a diagonal matrix with [w p1 , w p2 , . . .

, w pm ] on the diagonal and all the other elements be 0, we can make each element in the sum m i=1 n j=1 π(i, j)s(i, j) be 0, as this sum is supposed to be nonegative, its minimum is 0, hence W 1 (P G , P H ) = 0, which means for two isomorphic graphs G and H, their WEAVE's distributions P G and P H are the same.

Next we will prove the necessity of this theorem.

Suppose that the Wasserstein distance between the walk representation distributions P G and P H is 0, we will show that graph G and H are isomorphic.

Let the number of the nodes of graph G is n 1 , the number of the nodes of graph H is n 2 , let the number of the edges on graph G is m 1 , the number if the edges on graph H is m 2 .

Let k = 2 max{m 1 , m 2 } − 1.

Now, we will give a bijective mapping f : V (G) → v(H).

First, consider the walks on graph G, as k = 2 max{m 1 , m 2 } − 1 ≥ 2m 1 − 1, according to Lemma 1, there exists at least one walk of length k on graph G which can cover all the edges of G. Consider such a walk w G , let r G = [1, 2, 3, ..., t] be the representation vector (corresponds to a WEAVE) we obtained according to our encoding rule.

Now, we will use this representation to mark the nodes on graph G. Mark the first node in this walk as u 1 (corresponds to 1 in the representation), the second node as u 2 , the ith appearing node in w G is u i , continue this process untill we marked all the new appearing nodes in this walk.

Since w G can visit all the edges of graph G, all the nodes on this graph will definitely be marked, hence the last new appearing node will be marked as u n1 .

Now, let's consider the walks on graph H. As we have assumed that W 1 (P G , P H ) = 0, which means that for each point (g i , w gi ) on P G , we can find a point (h i , w hi ) in P H such that g i = h i , and w gi = w hi .

As a consequence, as r g is a point on P G , there must be a point r h on H such that r h = r g = [1, 2, 3, ..., t] .

Then choose any walk w h on H which produce r h , and apply the same method to mark the nodes in this walk in order as v 1 , v 2 , ..., v n1 .

Now we can define the mapping f , let f :

, which is exactly the mapping we are looking for.

Next, we just need show for each edge (u i , u j ) ∈ E(G), we have (f (u i ), f (u j )) ∈ E(H), and vice versa, then we can prove G and H are isomorphic.

The first direction is obviously true as w G covers all the edges on G, for any edge (u i , u j ) in w G , we have (f (u i ), f (u j )) = (v i , v j ) which belongs to w h , since w h is walk on H, we have (v i , v j ) ∈ E(H).

Then we will prove the reverse direction, i.e., for any

To prove this, we will first show that the number of edges of graph G and H are the same, i.e., m 1 = m 2 .

Suppose this is not true, without loss of generality, let m 1 > m 2 .

Since P G and P H are the results of random walks for infinite times.

Then there must exists some walks which can visit the additional edges on G, as a consequence, we can obtain some representation vector which will not appear Figure 6 : The walk representation distributions of graphs without attributes, with discrete attributes and with continuous attributes in P H , which contradicts our assumption.

Hence, we have m 1 = m 2 .

Besides, since we have r g = r h , according to Lemma 2, we can derive that the number of distinct edges on w g and w h are the same.

As w g covers all the edges on G, hence the number of distinct edges on w g is m 1 .

Therefore, the number of distinct edges on w h is also m 1 , which means w h also has visited all the edges on H. As for any edge (v i , v j ) on w h , we have (u i , u j ) on w h , in other words, we have

Hence we complete the proof.

Figure 6 shows the walk representation distributions for a 4 nodes ring with walk length k = 2 in three different cases: without node attributes, with discrete node attributes, and with continuous node attributes.

We can see the attributes will have an influence to the distributions, more specifically, the probability of each unique walk keeps the same no matter what the attributes are, however, the probability of each representation vector may vary as different unique walks may correspond to one representation vector, and the attributes may influence how many representation vectors there will be and how many unique walks correspond to a representation vector.

To clarify, in Figure 6 (a), the ring graph does not have nodes attributes, there exists 16 unique walks in total, among them walk ABD, BDC, DCA, CAB, DBA, CDB, ACD, BAC will all be encoded as r 1 = [1 2 3] , walk ABA, BAB, BDB, DBD, CDC, DCD, CAC, ACA will be encoded as r 2 = [1 2 1] .

Hence, for a graph in Figure 6 (a), we have P r(r 1 ) = 8 16 , P r(r 2 ) = 8 16 .

In Figure 6 (b), each node has a discrete attribute, i.e., red or green, there are still 16 unique walks in total.

However, in this case, there exits four different representation vectors, walk ABC, CBA, ADC, CDA will be encoded as r 1 = [1R 2G 3R] , where R represents Red while G represents Green; walk BCD, DCB, DAB, DCB correspond to r 2 = [1G 2R 3G] ; walk ABA, ADA, CDC, CBC correspond to r 3 = [1R 2G 3R] ; walk BAB, BCB, DCD, DAD correspond to r 3 = [1R 2G 3R] .

In this case, we have P r(r 1 ) = P r(r 2 ) = P r(r 3 ) = P r(r 4 ) = 4 16 .

In the last, let's consider the case when there exists continuous nodes attributes, for such a graph, the value of nodes attributes has infinite choices, hence, it is very likely that each node may have different attribute.

As a consequence, each unique walk will correspond to a unique representation vector.

In our example Figure 6 (c), there also exists 16 unique walks, each walk has a particular representation vector, hence, the probability of each representation vector is

Proof.

The proof for Theorem 2 is quite similar as the proof of Theorem 1, this is because the attributes just influence the representation vector form and how many unique walks correspond to a representation vector, however, the probability of each unique walk keeps same.

Hence, we can use a similar method to complete the proof.

Similarly, we will first prove the sufficiency.

Let G and H be two isomorphic graphs with attributes, we will prove that the walk representations distribution of G and H are the same.

Suppose that A and B are the sets of possible walks of length k on G and H respectively.

By applying the same analysis method as in the proof of Theorem 1, we can show that there exists a bijective mapping g : A → B such that for

where f :

)

∈ E(H) and for ∀v i ∈ V (G), the attribute of v i and f (v i ) are the same.

Hence, according to our encoding rule, w i and f (w i ) will be encoded as the same representation vector, which means for each point (r gi , P r(r gi )) in the representation distribution of G, we can find a point (r hi , P r(r hi )) in the distribution of H such that r gi = r hi , P r(r gi ) = P r(r hi ).

Thus, we can obtain the Wasserstein distance of distribution P G and the distribution P H is W 1 (P G , P H ) = 0 via a similar approach as in Theorem 1.

In other words, we have P G = P H .

In addition, the necessity proof of Theorem 2 is the same as Theorem 1.

If both the nodes and edges in a graph have attributes, the graph is an attributed graph denoted by G = (V, E, α, β), where α : V → L N and β : E → L E are nodes and edges labeling functions, L N , L E are sets of labels for nodes and edges.

In this case, the graph isomorphism are defined as:

G and H are isomorphic with node attributes as well as edge attributes if there is a bijection f :

Corollary 1.

Let G = (V (G), E(G)) and H = (V (H), E(H)) be two connected graphs with node attributes.

Suppose we can enumerate all possible WEAVEs on G and H with a fixed-length k ≥ 2 max{|E(G)|, |E(H)|}−1, where each WEAVE has a unique vector representation generated from a well-trained autoencoder.

The Wasserstein distance between G's and H's WEAVE distributions is 0 if and only if G and H are isomorphic with both node attributes and edge attributes.

Proof.

When both nodes and edges of a graph are given attributes, the representation vectors of random walks will be different.

However, just like the cases with only nodes attributes, the probability of each unique walk on the graph keeps same.

Hence, we can follow a similar analysis method as Theorem 2 to complete this proof.

We include the following 7 datasets in our experimental study.

• Deezer User-User Friendship Networks (Deezer) (Rozemberczki et al., 2018 ) is a social network dataset which is collected from the music streaming service Deezer.

It represents friendship network of users from three European countries (i.e., Romania, Croatia and Hungary).

There are three graphs which corresponds to the three countries.

Nodes represent the users and edges are the mutual friendships.

For the three graphs, the numbers of nodes are 41, 773, 54, 573, and 47, 538, respectively Borgwardt et al., 2005 ) is a bioinformatics dataset.

The proteins in the dataset are converted to graphs based on the sub-structures and physical connections of the proteins.

Specifically, nodes are secondary structure elements (SSEs), and edges represent the amino-acid sequence between the two neighbors.

PROTEINS has 3 discrete labels (i.e., helix, sheet, and turn).

There are 1, 113 graphs in total with 43, 471 edges.

• COLLAB (Leskovec et al., 2005 ) is a scientific collaboration dataset.

It belongs to a social connection network in general.

COLLAB is collected from 3 public collaboration datasets (i.e., High Energy Physics, Condensed Matter Physics, and Astro Physics).

The ego-networks are generated for individual researchers.

The label of each graph represents the field which this researcher belongs to.

There are 5, 000 graphs with 24, 574, 995 edges.

• IMDB-BINARY (Yanardag & Vishwanathan, 2015) is a movie collaboration dataset.

Each graph corresponds to an ego-network for each actor/actress, where nodes correspond to actors/actresses and an edge is drawn between two actors/actresses if they appear in the same movie.

Each graph is derived from a pre-specified genre of movies.

IMDB-BINARY has 1, 000 graphs associated with 19, 773 edges in total.

• IMDB-MULTI is multi-class version of IMDB-BINARY and contains a balanced set of egonetworks derived from Comedy, Romance, and Sci-Fi genres.

Specifically, there are 1, 500 graphs with 19, 502 edges in total.

• Graph Sample and Aggregate (GraphSAGE) (Hamilton et al., 2017) is an inductive graph representation learning approach in either supervised or unsupervised manner.

GraphSAGE explores node and structure information by sampling and aggregating features from the local neighborhood of each node.

A forward propagation algorithm is specifically designed to aggregates the information together.

We evaluate GraphSAGE in its unsupervised setting.

• Graph Matching Network (GMN) (Li et al., 2019) utilizes graph neural networks to obtain graph representations for graph matching applications.

A novel Graph Embedding Network is designed for better preserving node features and graph structures.

In particular, Graph Matching Network is proposed to directly obtain the similarity score of each pair of graphs.

In our implementation, we utilize the Graph Embedding Networks and deploy the graph-based loss function proposed in (Hamilton et al., 2017) for unsupervised learning fashion.

• Graph Isomorphism Network (GIN) (Xu et al., 2019) provides a simple yet effective neural network architecture for graph representation learning.

It deploys the sum aggregator to achieve more comprehensive representations.

The original GIN is a supervised learning method.

Thus, we follow the GraphSAGE approach, and modify its objective to fit an unsupervised setting.

Two downstream tasks, classification and clustering, are deployed to evaluate the quality of learned graph representations.

For classification task, a simple multi-layer fully connected neural network is built as a classifier.

Using 10-fold cross-validation, we report the average accuracy.

For clustering task, an effective conventional clustering approach, Normalized Cuts (NCut) (Jianbo Shi & Malik, 2000) , is used to cluster graph representations.

We consider two widely used metrics for clustering performance, including Accuracy (ACC) and Normalized Mutual Information (NMI) (Wu et al., 2009) .

ACC comes from classification with the best mapping, and NMI evaluates the mutual information across the ground truth and the recovered cluster labels based on a normalization operation.

Both ACC and NMI are positive measurements (i.e., the higher the metric is, the better the performance will be).

Table 4 (for NCI1 and PROTEINS) and in Table 5 (for COLLAB, IMDB-BINARY, and IMDB-MULTI).

We observe that SEED outperforms the baseline methods in 12 out of 15 cases, with up to 0.06 absolute improvement in classification and clustering accuracy.

In the rest 3 cases, SEED also achieves competitive performance.

Interestingly, for NCI and PROTEINS datasets, we see node features bring little improvement in the unsupervised setting.

One possible reason could be node feature information has high correlation with structural information in these cases.

H EMBEDDING DISTRIBUTION Identity kernels or commonly adopted kernels could be deployed in the component of embedding subgraph distributions.

In our implementation, we utilize a multi-layer deep neural network to approximate a feature mapping function, for kernels whose feature mapping function is difficult to obtain.

Figure 7 shows the t-SNE visualization of learned graph representations based on identity kernel and RBF kernel.

As shown in Table 6 , SEED variants with different kernels for distribution embedding could distinguish different classes with similar performance on the MUTAG dataset.

In this section, we investigate whether DeepSet (Zaheer et al., 2017) is an effective technique for distribution embedding.

In particular, we employ DeepSet to replace the multi-layer neural network for feature mapping function approximation, and similarity values generated by MMD serve Table 7 : Representation evaluation based on classification and clustering down-stream tasks as supervision signals to guide DeepSet training.

In our experiments, we compare the SEED implementation based on DeepSet with MMD (DeepSet in Table 7 ) with the SEED implementation based on the identity kernel (Identity Kernel in Table 7 ).

We also observe that the MMD does not have significant performance different.

In this section, we investigate the impact of node features and earliest visit time in WEAVE.

In Table 8 , Node feature means only node features in WEAVE are utilized for subgraph encoding (which is equivalent to vanilla random walks), earliest visit time means only earliest visit time information in WEAVE is used for subgraph encoding, and Node feature + earliest visit time means both information is employed.

We evaluate the impact on the MUTAG dataset.

As shown above, it is crucial to use both node feature and earliest visit time information in order to achieve the best performance.

Interestingly, on the MUTAG dataset, we observe that clustering could be easier if we only consider earliest visit time information.

On the MUTAG dataset, node features seem to be noisy for the clustering task.

As the clustering task is unsupervised, noisy node features could negatively impact its performance when both node features and earliest visit time information are considered.

In this section, we evaluate the impact of Nyström based kernel approximation (Williams & Seeger, 2001) to the component of embedding distributions.

First, we investigate the impact to the effectiveness in the downstream tasks.

In this set of experiment, we implement a baseline named SEED-Nyström, where the Nyström method is applied to approximate RBF kernel based MMD during training phases with 200 sampled WEAVEs.

In particular, top 30 eigenvalues and the corresponding eigenvectors are selected for the approximation.

As shown in Table 9 , across five datasets, SEED-Nyström achieves comparable performance, compared with the case where an identity kernel is adopted.

In addition, we evaluate the response time of exact RBF kernel based MMD and its Nyström approximation.

Table 9 : Representation evaluation based on classification and clustering down-stream tasks approximation.

As shown in Figure 8 , when we range the number of WEAVE samples from 100 to 2000, the Nyström approximation scales better than the exact MMD evaluation.

In summary, the Nyström method is a promising method that can further improve the scalability of the SEED framework in training phases, especially for the cases where a large number of WEAVE samples are required.

<|TLDR|>

@highlight

This paper proposed a novel framework for graph similarity learning in inductive and unsupervised scenario.
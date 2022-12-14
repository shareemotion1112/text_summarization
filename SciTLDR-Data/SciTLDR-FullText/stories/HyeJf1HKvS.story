This work presents a two-stage neural architecture for learning and refining structural correspondences between graphs.

First, we use localized node embeddings computed by a graph neural network to obtain an initial ranking of soft correspondences between nodes.

Secondly, we employ synchronous message passing networks to iteratively re-rank the soft correspondences to reach a matching consensus in local neighborhoods between graphs.

We show, theoretically and empirically, that our message passing scheme computes a well-founded measure of consensus for corresponding neighborhoods, which is then used to guide the iterative re-ranking process.

Our purely local and sparsity-aware architecture scales well to large, real-world inputs while still being able to recover global correspondences consistently.

We demonstrate the practical effectiveness of our method on real-world tasks from the fields of computer vision and entity alignment between knowledge graphs, on which we improve upon the current state-of-the-art.

Graph matching refers to the problem of establishing meaningful structural correspondences of nodes between two or more graphs by taking both node similarities and pairwise edge similarities into account (Wang et al., 2019b) .

Since graphs are natural representations for encoding relational data, the problem of graph matching lies at the heart of many real-world applications.

For example, comparing molecules in cheminformatics (Kriege et al., 2019b) , matching protein networks in bioinformatics (Sharan & Ideker, 2006; Singh et al., 2008) , linking user accounts in social network analysis (Zhang & Philip, 2015) , and tracking objects, matching 2D/3D shapes or recognizing actions in computer vision (Vento & Foggia, 2012) can be formulated as a graph matching problem.

The problem of graph matching has been heavily investigated in theory (Grohe et al., 2018) and practice (Conte et al., 2004) , usually by relating it to domain-agnostic distances such as the graph edit distance (Stauffer et al., 2017) and the maximum common subgraph problem (Bunke & Shearer, 1998) , or by formulating it as a quadratic assignment problem (Yan et al., 2016) .

Since all three approaches are NP-hard, solving them to optimality may not be tractable for large-scale, real-world instances.

Moreover, these purely combinatorial approaches do not adapt to the given data distribution and often do not consider continuous node embeddings which can provide crucial information about node semantics.

Recently, various neural architectures have been proposed to tackle the task of graph matching (Zanfir & Sminchisescu, 2018; Wang et al., 2019b; Zhang & Lee, 2019; Xu et al., 2019d; Derr et al., 2019; Heimann et al., 2018) or graph similarity (Bai et al., 2018; in a data-dependent fashion.

However, these approaches are either only capable of computing similarity scores between whole graphs (Bai et al., 2018; , rely on an inefficient global matching procedure (Zanfir & Sminchisescu, 2018; Wang et al., 2019b; Xu et al., 2019d; , or do not generalize to unseen graphs (Xu et al., 2019b; Derr et al., 2019; .

Moreover, they might be prone to match neighborhoods between graphs Typically, graph matching is formulated as an edge-preserving, quadratic assignment problem (Anstreicher, 2003; Gold & Rangarajan, 1996; Caetano et al., 2009; Cho et al., 2013) , i.e.,

subject to the one-to-one mapping constraints mentioned above.

This formulation is based on the intuition of finding correspondences based on neighborhood consensus (Rocco et al., 2018) , which shall prevent adjacent nodes in the source graph from being mapped to different regions in the target graph.

Formally, a neighborhood consensus is reached if for all node pairs (i, j) ??? V s ?? V t with S i,j = 1, it holds that for every node i ??? N 1 (i) there exists a node j ??? N 1 (j) such that S i ,j = 1.

In this work, we consider the problem of supervised and semi-supervised matching of graphs while employing the intuition of neighborhood consensus as an inductive bias into our model.

In the supervised setting, we are given pair-wise ground-truth correspondences for a set of graphs and want our model to generalize to unseen graph pairs.

In the semi-supervised setting, source and target graphs are fixed, and ground-truth correspondences are only given for a small subset of nodes.

However, we are allowed to make use of the complete graph structures.

In the following, we describe our proposed end-to-end, deep graph matching architecture in detail.

See Figure 1 for a high-level illustration.

The method consists of two stages: a local feature matching procedure followed by an iterative refinement strategy using synchronous message passing networks.

The aim of the feature matching step, see Section 3.1, is to compute initial correspondence scores based on the similarity of local node embeddings.

The second step is an iterative refinement strategy, see Sections 3.2 and 3.3, which aims to reach neighborhood consensus for correspondences using a differentiable validator for graph isomorphism.

Finally, in Section 3.4, we show how to scale our method to large, real-world inputs.

Stage I Stage II S ??, ?? ?? ??3 ( ??? ) Figure 1 : High-level illustration of our two-stage neighborhood consensus architecture.

Node features are first locally matched based on a graph neural network ?? ??1 , before their correspondence scores get iteratively refined based on neighborhood consensus.

Here, an injective node coloring of G s is transferred to G t via S, and distributed by ?? ??2 on both graphs.

Updates on S are performed by a neural network ?? ??3 based on pair-wise color differences.

We model our local feature matching procedure in close analogy to related approaches (Bai et al., 2018; Wang et al., 2019b; Zhang & Lee, 2019; Wang & Solomon, 2019) by computing similarities between nodes in the source graph G s and the target graph G t based on node embeddings.

That is, given latent node embeddings

|Vt|???? computed by a shared neural network ?? ??1 for source graph G s and target graph G t , respectively, we obtain initial soft correspondences as

Here, sinkhorn normalization is applied to obtain rectangular doubly-stochastic correspondence matrices that fulfill the constraints j???Vt S i,j = 1 ???i ??? V s and i???Vs S i,j ??? 1 ???j ??? V t (Sinkhorn & Knopp, 1967; Adams & Zemel, 2011; Cour et al., 2006) .

We interpret the i-th row vector S

i,: ??? [0, 1] |Vt| as a discrete distribution over potential correspondences in G t for each node i ??? V s .

We train ?? ??1 in a dicriminative, supervised fashion against ground truth correspondences ?? gt (??) by minimizing the negative log-likelihood of correct correspondence scores L (initial) = ??? i???Vs log(S

i,??gt(i) ).

We implement ?? ??1 as a Graph Neural Network (GNN) to obtain localized, permutation equivariant vectorial node representations Hamilton et al., 2017; Battaglia et al., 2018; Goyal & Ferrara, 2018) .

Formally, a GNN follows a neural message passing scheme (Gilmer et al., 2017) and updates its node features h (t???1) i in layer t by aggregating localized information via

where h (0) i = x i ??? X and {{. . .}} denotes a multiset.

The recent work in the fields of geometric deep learning and relational representation learning provides a large number of operators to choose from (Kipf & Welling, 2017; Gilmer et al., 2017; Veli??kovi?? et al., 2018; Schlichtkrull et al., 2018; Xu et al., 2019c) , which allows for precise control of the properties of extracted features.

Due to the purely local nature of the used node embeddings, our feature matching procedure is prone to finding false correspondences which are locally similar to the correct one.

Formally, those cases pose a violation of the neighborhood consensus criteria employed in Equation (1).

Since finding a global optimum is NP-hard, we aim to detect violations of the criteria in local neighborhoods and resolve them in an iterative fashion.

We utilize graph neural networks to detect these violations in a neighborhood consensus step and iteratively refine correspondences S (l) , l ??? {0, . . .

, L}, starting from S (0) .

Key to the proposed algorithm is the following observation: The soft correspondence matrix S ??? [0, 1] |Vs|??|Vt| is a map from the node function space

.

Therefore, we can use S to pass node functions x s ??? L(G s ), x t ??? L(G t ) along the soft correspondences by x t = S x s and x s = S x t (3) to obtain functions x t ??? L(G t ), x s ??? L(G s ) in the other domain, respectively.

Then, our consensus method works as follows: Using S (l) , we first map node indicator functions,

given as an injective node coloring V s ??? {0, 1} |Vs| in the form of an identity matrix I |Vs| , from G s to G t .

Then, we distribute this coloring in corresponding neighborhoods by performing synchronous message passing on both graphs via a shared graph neural network ?? ??2 , i.e.,

We can compare the results of both GNNs to recover a vector

which measures the neighborhood consensus between node pairs (i, j) ??? V s ?? V t .

This measure can be used to perform trainable updates of the correspondence scores

based on an MLP ?? ??3 .

The process can be applied L times to iteratively improve the consensus in

i,??gt(i) ) combines both the feature matching error and neighborhood consensus error.

This objective is fullydifferentiable and can hence be optimized in an end-to-end fashion using stochastic gradient descent.

Overall, the consensus stage distributes global node colorings to resolve ambiguities and false matchings made in the first stage of our architecture by only using purely local operators.

Since an initial matching is needed to test for neighborhood consensus, this task cannot be fulfilled by ?? ??1 alone, which stresses the importance of our two-stage approach.

The following two theorems show that d i,j is a good measure of how well local neighborhoods around i and j are matched by the soft correspondence between G s and G t .

The proofs can be found in Appendix B and C, respectively.

Theorem 1.

Let G s and G t be two isomorphic graphs and let ?? ??2 be a permutation equivariant GNN, i.e., P ?? ??2 (X, A) = ?? ??2 (P X, P AP ) for any permutation matrix P ??? {0, 1} |V|??|V| .

If S ??? {0, 1} |Vs|??|Vt| encodes an isomorphism between G s and

Theorem 2.

Let G s and G t be two graphs and let ?? ??2 be a permutation equivariant and T -layered GNN for which both AGGREGATE (t) and UPDATE (t) are injective for all t ??? {1, . . .

, T }.

Hence, a GNN ?? ??2 that satisfies both criteria in Theorem 1 and 2 provides equal node embeddings o Note that both requirements, permutation equivariance and injectivity, are easily fulfilled: (1) All common graph neural network architectures following the message passing scheme of Equation (2) are equivariant due to the use of permutation invariant neighborhood aggregators.

(2) Injectivity of graph neural networks is a heavily discussed topic in recent literature.

It can be fulfilled by using a GNN that is as powerful as the Weisfeiler & Lehman (1968) (WL) heuristic in distinguishing graph structures, e.g., by using sum aggregation in combination with MLPs on the multiset of neighboring node features, cf. (Xu et al., 2019c; Morris et al., 2019) .

Theoretically, we can relate our proposed approach to classical graph matching techniques that consider a doubly-stochastic relaxation of the problem defined in Equation (1), cf. (Lyzinski et al., 2016) and Appendix F for more details.

A seminal work following this method is the graduated assignment algorithm (Gold & Rangarajan, 1996) .

By starting from an initial feasible solution S (0) , a new solution S (l+1) is iteratively computed from S (l) by approximately solving a linear assignment problem according to

where Q denotes the gradient of Equation (1) at S (l) .

1 The softassign operator is implemented by applying sinkhorn normalization on rescaled inputs, where the scaling factor grows in every iteration to increasingly encourage integer solutions.

Our approach also resembles the approximation of the linear assignment problem via sinkhorn normalization.

Moreover, the gradient Q is closely related to our neighborhood consensus scheme for the particular simple, non-trainable GNN instantiation ??(X,

based on the similarity between O s and O t obtained from a fixed-function GNN ??, we choose to update correspondence scores via trainable neural networks ?? ??2 and ?? ??3 based on the difference between O s and O t .

This allows us to interpret our model as a deep parameterized generalization of the graduated assignment algorithm.

In addition, specifying node and edge attribute similarities in graph matching is often difficult and complicates its computation (Zhou & De la Torre, 2016; Zhang et al., 2019c) , whereas our approach naturally supports continuous node and edge features via established GNN models.

We experimentally verify the benefits of using trainable neural networks ?? ??2 instead of ??(X, A, E) = AX in Appendix D.

We apply a number of optimizations to our proposed algorithm to make it scale to large input domains.

See Algorithm 1 in Appendix A for the final optimized algorithm.

Sparse correspondences.

We propose to sparsify initial correspondences S (0) by filtering out low score correspondences before neighborhood consensus takes place.

That is, we sparsify S (0) by computing top k correspondences with the help of the KEOPS library (Charlier et al., 2019) without ever storing its dense version, reducing its required memory footprint from O(|V s ||V t |) to O(k|V s |).

In addition, the time complexity of the refinement phase is reduced from O(

where |E s | and |E t | denote the number of edges in G s and G t , respectively.

Note that sparsifying initial correspondences assumes that the feature matching procedure ranks the correct correspondence within the top k elements for each node i ??? V s .

Hence, also optimizing the initial feature matching loss L (initial) is crucial, and can be further accelerated by training only against sparsified correspondences with ground-truth entries

Replacing node indicators functions.

Although applying ?? ??2 on node indicator functions I |Vs| is computationally efficient, it requires a parameter complexity of O(|V s |).

Hence, we propose to replace node indicator functions I |Vs| with randomly drawn node functions R

s ??? R |Vs|??r with r |V s |, in iteration l. By sampling from a continuous distribution, node indicator functions are still guaranteed to be injective (DeGroot & Schervish, 2012) .

Note that Theorem 1 still holds because it does not impose any restrictions on the function space L(G s ).

Theorem 2 does not necessarily hold anymore, but we expect our refinement strategy to resolve any ambiguities by re-sampling R (l) s in every iteration l.

We verify this empirically in Section 4.1.

Softmax normalization.

The sinkhorn normalization fulfills the requirements of rectangular doubly-stochastic solutions.

However, it may eventually push correspondences to inconsistent integer solutions very early on from which the neighborhood consensus method cannot effectively recover.

Furthermore, it is inherently inefficient to compute and runs the risk of vanishing gradients ???S (l) /????? (l) (Zhang et al., 2019b) .

Here, we propose to relax this constraint by only applying row-wise softmax normalization on?? (l) , and expect our supervised refinement procedure to naturally resolve violations of i???Vs S i,j ??? 1 on its own by re-ranking false correspondences via neighborhood consensus.

Experimentally, we show that row-wise normalization is sufficient for our algorithm to converge to the correct solution, cf.

Section 4.1.

Number of refinement iterations.

Instead of holding L fixed, we propose to differ the number of refinement iterations

, for training and testing, respectively.

This does not only speed up training runtime, but it also encourages the refinement procedure to reach convergence with as few steps as necessary while we can run the refinement procedure until convergence during testing.

We show empirically that decreasing L (train) does not affect the convergence abilities of our neighborhood consensus procedure during testing, cf.

Section 4.1.

We verify our method on three different tasks.

We first show the benefits of our approach in an ablation study on synthetic graphs (Section 4.1), and apply it to the real-world tasks of supervised keypoint matching in natural images (Sections 4.2 and 4.3) and semi-supervised cross-lingual knowledge graph alignment (Section 4.4) afterwards.

All dataset statistics can be found in Appendix H.

Our method is implemented in PYTORCH (Paszke et al., 2017) using the PYTORCH GEOMETRIC ) and the KEOPS (Charlier et al., 2019) libraries.

Our implementation can process sparse mini-batches with parallel GPU acceleration and minimal memory footprint in all algorithm steps.

For all experiments, optimization is done via ADAM (Kingma & Ba, 2015) with a fixed learning rate of 10 ???3 .

We use similar architectures for ?? ??1 and ?? ??2 except that we omit dropout (Srivastava et al., 2014) in ?? ??2 .

For all experiments, we report Hits@k to evaluate and compare our model to previous lines of work, where Hits@k measures the proportion of correctly matched entities ranked in the top k.

In our first experiment, we evaluate our method on synthetic graphs where we aim to learn a matching for pairs of graphs in a supervised fashion.

Each pair of graphs consists of an undirected Erd??s & R??nyi (1959) graph G s with |V s | ??? {50, 100} nodes and edge probability p ??? {0.1, 0.2}, and a target graph G t which is constructed from G s by removing edges with probability p s without disconnecting any nodes (Heimann et al., 2018) .

Training and evaluation is done on 1 000 graphs each for different configurations p s ??? {0.0, 0.1, 0.2, 0.3, 0.4, 0.5}. In Appendix E, we perform additional experiments to also verify the robustness of our approach towards node addition or removal.

Architecture and parameters.

We implement the graph neural network operators ?? ??1 and ?? ??2 by stacking three layers (T = 3) of the GIN operator (Xu et al., 2019c)

due to its expressiveness in distinguishing raw graph structures.

The number of layers and hidden dimensionality of all MLPs is set to 2 and 32, respectively, and we apply ReLU activation (Glorot et al., 2011) after each of its layers.

Input features are initialized with one-hot encodings of node degrees.

We employ a Jumping Knowledge style concatenation et al., 2018) to compute final node representations h i .

We train and test our procedure with L (train) = 10 and L (test) = 20 refinement iterations, respectively.

show the matching accuracy Hits@1 for different choices of |V s | and p. We observe that the purely local matching approach via softmax(?? (0) ) starts decreasing in performance with the structural noise p s increasing.

This also holds when applying global sinkhorn normalization on?? (0) .

However, our proposed two-stage architecture can recover all correspondences, independent of the applied structural noise p s .

This applies to both variants discussed in the previous sections, i.e., our initial formulation sinkhorn(?? (L) ), and our optimized architecture using random node indicator sampling and row-wise normalization softmax(?? (L) ).

This highlights the overall benefits of applying matching consensus and justifies the usage of the enhancements made towards scalability in Section 3.4.

(refined) for varying number of iterations L (test) .

We observe that even when training to non-convergence, our procedure is still able to converge by increasing the number of iterations L (test) during testing.

Moreover, Figure 2 (d) shows the performance of our refinement strategy when operating on sparsified top k correspondences.

In contrast to its dense version, it cannot match all nodes correctly due to the poor initial feature matching quality.

However, it consistently converges to the perfect solution of Hits@1 ??? Hits@k in case the correct match is included in the initial top k ranking of correspondences.

Hence, with increasing k, we can recover most of the correct correspondences, making it an excellent option to scale our algorithm to large graphs, cf.

Section 4.4.

We perform experiments on the PASCALVOC (Everingham et al., 2010) with Berkeley annotations (Bourdev & Malik, 2009 ) and WILLOW-OBJECTCLASS (Cho et al., 2013) datasets which contain sets of image categories with labeled keypoint locations.

For PASCALVOC, we follow the experimental setups of Zanfir & Sminchisescu (2018) and Wang et al. (2019b) and use the training and test splits provided by Choy et al. (2016) .

We pre-filter the dataset to exclude difficult, occluded and truncated objects, and require examples to have at least one keypoint, resulting in 6 953 and 1 671 annotated images for training and testing, respectively.

The PASCALVOC dataset contains instances of varying scale, pose and illumination, and the number of keypoints ranges from 1 to 19.

In contrast, the WILLOW-OBJECTCLASS dataset contains at least 40 images with consistent orientations for each of its five categories, and each image consists of exactly 10 keypoints.

Following the experimental setup of peer methods (Cho et al., 2013; Wang et al., 2019b) , we pre-train our model on PASCALVOC and fine-tune it over 20 random splits with 20 per-class images used for training.

We construct graphs via the Delaunay triangulation of keypoints.

For fair comparison with Zanfir & Sminchisescu (2018) and Wang et al. (2019b) , input features of keypoints are given by the concatenated output of relu4 2 and relu5 1 of a pre-trained VGG16 (Simonyan & Zisserman, 2014) on IMAGENET (Deng et al., 2009 ).

Architecture and parameters.

We adopt SPLINECNN (Fey et al., 2018) as our graph neural network operator

whose trainable B-spline based kernel function ?? ?? (??) is conditioned on edge features e j,i between node-pairs.

To align our results with the related work, we evaluate both isotropic and anisotropic

99.62 ?? 0.28 73.47 ?? 3.32 77.47 ?? 4.92 77.10 ?? 3.25 88.04 ?? 1.38 L = 10 100.00 ?? 0.00 92.05 ?? 3.49 90.05 ?? 5.10 88.98 ?? 2.75 97.14 ?? 1.41 L = 20 100.00 ?? 0.00 92.05 ?? 3.24 90.28 ?? 4.67 88.97 ?? 3.49 97.14 ?? 1.83

98.47 ?? 0.61 49.28 ?? 4.31 64.95 ?? 3.52 66.17 ?? 4.08 78.08 ?? 2.61 L = 10 100.00 ?? 0.00 76.28 ?? 4.77 86.70 ?? 3.25 83.22 ?? 3.52 93.65 ?? 1.64 L = 20 100.00 ?? 0.00 76.57 ?? 5.28 89.00 ?? 3.88 84.78 ?? 2.73 95.29 ?? 2.22

99.96 ?? 0.06 91.90 ?? 2.30 91.28 ?? 4.89 86.58 ?? 2.99 98.25 ?? 0.71 L = 10 100.00 ?? 0.00 98.80 ?? 1.58 96.53 ?? 1.55 93.22 ?? 3.77 99.87 ?? 0.31 L = 20 100.00 ?? 0.00 99.40 ?? 0.80 95.53 ?? 2.93 93.00 ?? 2.71 99.39 ?? 0.70 edge features which are given as normalized relative distances and 2D Cartesian coordinates, respectively.

For SPLINECNN, we use a kernel size of 5 in each dimension, a hidden dimensionality of 256, and apply ReLU as our non-linearity function ??.

Our network architecture consists of two convolutional layers (T = 2), followed by dropout with probability 0.5, and a final linear layer.

During training, we form pairs between any two training examples of the same category, and evaluate our model by sampling a fixed number of test graph pairs belonging to the same category.

Results.

We follow the experimental setup of Wang et al. (2019b) and train our models using negative log-likelihood due to its superior performance in contrast to the displacement loss used in Zanfir & Sminchisescu (2018) .

We evaluate our complete architecture using isotropic and anisotropic GNNs for L ??? {0, 10, 20}, and include ablation results obtained from using ?? ??1 = MLP for the local node matching procedure.

Results of Hits@1 are shown in Table 1 and 2 for PASCALVOC and WILLOW-OBJECTCLASS, respectively.

We visualize qualitative results of our method in Appendix I.

We observe that our refinement strategy is able to significantly outperform competing methods as well as our non-refined baselines.

On the WILLOW-OBJECTCLASS dataset, our refinement stage at least reduces the error of the initial model (L = 0) by half across all categories.

The benefits of the second stage are even more crucial when starting from a weaker initial feature matching baseline (?? ??1 = MLP), with overall improvements of up to 14 percentage points on PASCALVOC.

However, good initial matchings do help our consensus stage to improve its performance further, as indicated by the usage of task-specific isotropic or anisotropic GNNs for ?? ??1 .

We also verify our approach by tackling the geometric feature matching problem, where we only make use of point coordinates and no additional visual features are available.

Here, we follow the experimental training setup of Zhang & Lee (2019) , and test the generalization capabilities of our model on the PASCALPF dataset (Ham et al., 2016) .

For training, we generate a synthetic set of graph pairs: We first randomly sample 30-60 source points uniformly from [???1, 1] 2 , and add Gaussian noise from N (0, 0.05

2 ) to these points to obtain the target points.

Furthermore, we add 0-20 outliers from [???1.5, 1.5] 2 to each point cloud.

Finally, we construct graphs by connecting each node with its k-nearest neighbors (k = 8).

We train our unmodified anisotropic keypoint architecture from Section 4.2 with input x i = 1 ??? R 1 ???i ??? V s ??? V t until it has seen 32 000 synthetic examples.

Results.

We evaluate our trained model on the PASCALPF dataset (Ham et al., 2016) which consists of 1 351 image pairs within 20 classes, with the number of keypoints ranging from 4 to 17.

Results of Hits@1 are shown in Table 3 .

Overall, our consensus architecture improves upon the state-of-the-art results of Zhang & Lee (2019) on almost all categories while our L = 0 baseline is weaker than the results reported in Zhang & Lee (2019) , showing the benefits of applying our consensus stage.

In addition, it shows that our method works also well even when not taking any visual information into account.

We evaluate our model on the DBP15K datasets (Sun et al., 2017) which link entities of the Chinese, Japanese and French knowledge graphs of DBPEDIA into the English version and vice versa.

Each dataset contains exactly 15 000 links between equivalent entities, and we split those links into training and testing following upon previous works.

For obtaining entity input features, we follow the experimental setup of Xu et al. (2019d) :

We retrieve monolingual FASTTEXT embeddings (Bojanowski et al., 2017) for each language separately, and align those into the same vector space afterwards (Lample et al., 2018) .

We use the sum of word embeddings as the final entity input representation (although more sophisticated approaches are just as conceivable).

Architecture and parameters.

Our graph neural network operator mostly matches the one proposed in Xu et al. (2019d) where the direction of edges is retained, but not their specific relation type:

We use ReLU followed by dropout with probability 0.5 as our non-linearity ??, and obtain final node representations via

i ].

We use a three-layer GNN (T = 3) both for obtaining initial similarities and for refining alignments with dimensionality 256 and 32, respectively.

Training is performed using negative log likelihood in a semi-supervised fashion: For each training node i in V s , we train L (initial) sparsely by using the corresponding ground-truth node in V t , the top k = 10 entries in S i,: and k randomly sampled entities in V t .

For the refinement phase, we update the sparse top k correspondence matrix L = 10 times.

For efficiency reasons, we train L (initial) and L

sequentially for 100 epochs each.

Results.

We report Hits@1 and Hits@10 to evaluate and compare our model to previous lines of work, see Table 4 .

In addition, we report results of a simple three-layer MLP which matches nodes purely based on initial word embeddings, and a variant of our model without the refinement of initial correspondences (L = 0).

Our approach improves upon the state-of-the-art on all categories with gains of up to 9.38 percentage points.

In addition, our refinement strategy consistently improves upon the Hits@1 of initial correspondences by a significant margin, while results of Hits@10 are shared due to the refinement operating only on sparsified top 10 initial correspondences.

Due to the scalability of our approach, we can easily apply a multitude of refinement iterations while still retaining large hidden feature dimensionalities.

Our experimental results demonstrate that the proposed approach effectively solves challenging realworld problems.

However, the expressive power of GNNs is closely related to the WL heuristic for graph isomorphism testing (Xu et al., 2019c; Morris et al., 2019) , whose power and limitations are well understood (Arvind et al., 2015) .

Our method generally inherits these limitations.

Hence, one possible limitation is that whenever two nodes are assigned the same color by WL, our approach may fail to converge to one of the possible solutions.

For example, there may exist two nodes i, j ??? V t with equal neighborhood sets N 1 (i) = N 1 (j).

One can easily see that the feature matching procedure generates equal initial correspondence distributions S :,j receive the same update, leading to non-convergence.

In theory, one might resolve these ambiguities by adding a small amount of noise to?? (0) .

However, the general amount of feature noise present in real-world datasets already ensures that this scenario is unlikely to occur.

Identifying correspondences between the nodes of two graphs has been studied in various domains and an extensive body of literature exists.

Closely related problems are summarized under the terms maximum common subgraph (Kriege et al., 2019b) , network alignment (Zhang, 2016) , graph edit distance and graph matching (Yan et al., 2016) .

We refer the reader to the Appendix F for a detailed discussion of the related work on these problems.

Recently, graph neural networks have become a focus of research leading to various proposed deep graph matching techniques (Wang et al., 2019b; Zhang & Lee, 2019; Xu et al., 2019d; Derr et al., 2019) .

In Appendix G, we present a detailed overview of the related work in this field while highlighting individual differences and similarities to our proposed graph matching consensus procedure.

We presented a two-stage neural architecture for learning node correspondences between graphs in a supervised or semi-supervised fashion.

Our approach is aimed towards reaching a neighborhood consensus between matchings, and can resolve violations of this criteria in an iterative fashion.

In addition, we proposed enhancements to let our algorithm scale to large input domains.

We evaluated our architecture on real-world datasets on which it consistently improved upon the state-of-the-art.

Our final optimized algorithm is given in Algorithm 1:

Input:

, hidden node dimensionality d, sparsity parameter k, number of consensus iterations L, number of random functions r Output:

Proof.

Since ?? ??2 is permutation equivariant, it holds for any node feature matrix X s ??? R |Vs|???? that ?? ??2 (S X s , S A s S) = S ?? ??2 (X s , A s ).

With X t = S X s and A t = S A s S, it follows that

Then, the T -layered GNN ?? ??2 maps both T -hop neighborhoods around nodes i ??? V s and j ??? V t to the same vectorial representation:

Because ?? ??2 is as powerful as the WL heuristic in distinguishing graph structures (Xu et al., 2019c; Morris et al., 2019) and is operating on injective node colorings I |V|s , it has the power to distinguish any graph structure from

j , where denotes the labeled graph isomorphism relation.

Hence, there exists an isomorphism P ??? {0, 1}

With I |Vs| being the identity matrix, it follows that

to its column-wise non-zero entries.

It follows that S N T (i),N T (j) = P is a permutation matrix describing an isomorphism.

Moreover, if d i,argmax Si,: = 0 for all i ??? V s , it directly follows that S is holding submatrices describing isomorphisms between any T -hop subgraphs around i ??? V s and argmax S i,: ??? V t .

Assume there exists nodes i, i ??? V s that map to the same node j = argmax S i,: = argmax S i ,:

which contradicts the injectivity requirements of AGGREGATE (t) and UPDATE (t) for all t ??? {1, . . .

, T }.

Hence, S must be itself a permutation matrix describing an isomorphism between G s and G t .

As stated in Section 3.3, our algorithm can be viewed as a generalization of the graduated assignment algorithm (Gold & Rangarajan, 1996) extending it by trainable parameters.

To evaluate the impact of a trainable refinement procedure, we replicated the experiments of Sections 4.2 and 4.4 by implementing ?? ??2 via a non-trainable, one-layer GNN instantiation ?? ??2 (X, A, E) = AX.

Tables 5 and 6 show that using trainable neural networks ?? ??2 consistently improves upon the results of using the fixed-function message passing scheme.

While it is difficult to encode meaningful similarities between node and edge features in a fixed-function pipeline, our approach is able to learn how to make use of those features to guide the refinement procedure further.

In addition, it allows us to choose from a variety of task-dependent GNN operators, e.g., for learning geometric/edge conditioned patterns or for fulfilling injectivity requirements.

The theoretical expressivity discussed in Section 5 could even be enhanced by making use of higher-order GNNs, which we leave for future work.

To experimentally validate the robustness of our approach towards node addition (or removal), we conducted additional synthetic experiments in a similar fashion to Xu et al. (2019b) .

We form graph-pairs by treating an Erd??s & R??nyi graph with |V s | ??? {50, 100} nodes and edge probability p ??? {0.1, 0.2} as our source graph G s .

The target graph G t is then constructed by first adding q% noisy nodes to the source graph, i.e., |V t | = (1 + q%)|V s |, and generating edges between these nodes and all other nodes based on the edge probability p afterwards.

We use the same network architecture and training procedure as described in Section 4.1.

As one can see, our consensus stage is extremely robust to the addition or removal of nodes while the first stage alone has major difficulties in finding the right matching.

This can be explained by the fact that unmatched nodes do not have any influence on the neighborhood consensus error since those nodes do not obtain a color from the functional map given by S. Our neural architecture is able to detect and gradually decrease any false positive influence of these nodes in the refinement stage.

Identifying correspondences between the nodes of two graphs is a problem arising in various domains and has been studied under different terms.

In graph theory, the combinatorial maximum common subgraph isomorphism problem is studied, which asks for the largest graph that is contained as subgraph in two given graphs.

The problem is NP-hard in general and remains so even in trees (Garey & Johnson, 1979) unless the common subgraph is required to be connected (Matula, 1978) .

Moreover, most variants of the problem are difficult to approximate with theoretical guarantees (Kann, 1992) .

We refer the reader to the survey by Kriege et al. (2019b) for a overview of the complexity results noting that exact polynomial-time algorithms are available for specific problem variants only that are most relevant in cheminformatics.

Fundamentally different techniques have been developed in bioinformatics and computer vision, where the problem is commonly referred to as network alignment or graph matching.

In these areas large networks without any specific structural properties are common and the studied techniques are non-exact.

In graph matching, for two graphs of order n with adjacency matrix A s and A t , respectively, typically the function

is to be minimized, where S ??? P with P the set of n ?? n permutation matrices and A 2 F = i,i ???V A 2 i,i denotes the squared Frobenius norm.

Since the first two terms of the right-hand side do not depend on S, minimizing Equation (12) is equivalent in terms of optimal solutions to the problem of Equation (1).

We briefly summarize important related work in graph matching and refer the reader to the recent survey by Yan et al. (2016) for a more detailed discussion.

There is a long line of research trying to minimize Equation (12)

n??n by a Frank-Wolfe type algorithm (Jaggi, 2013) and finally projecting the fractional solution to P (Gold & Rangarajan, 1996; Zaslavskiy et al., 2009; Leordeanu et al., 2009; Egozi et al., 2013; Zhou & De la Torre, 2016) .

However, the applicability of relaxation and projection is still poorly understood and only few theoretical results exist (Aflalo et al., 2015; Lyzinski et al., 2016) .

A classical result by Tinhofer (1991) states that the WL heuristic distinguishes two graphs G s and G t if and only if there is no fractional S such that the objective function in Equation (12) takes 0.

Kersting et al. (2014) showed how the Frank-Wolfe algorithm can be modified to obtain the WL partition.

Aflalo et al. (2015) proved that the standard relaxation yields a correct solution for a particular class of asymmetric graphs, which can be characterized by the spectral properties of their adjacency matrix.

Finally, Bento & Ioannidis (2018) studied various relaxations, their complexity and properties.

Other approaches to graph matching exist, e.g., based on spectral relaxations (Umeyama, 1988; Leordeanu & Hebert, 2005) or random walks (Gori et al., 2005) .

The problem of graph matching is closely related to the notoriously hard quadratic assignment problem (QAP) (Zhou & De la Torre, 2016) , which has been studied in operations research for decades.

Equation (1) can be directly interpreted as KoopmansBeckmann's QAP.

The more recent literature on graph matching typically considers a weighted version, where node and edge similarities are taken into account.

This leads to the formulation as Lawler's QAP, which involves an affinity matrix of size n 2 ?? n 2 and is computational demanding.

Zhou & De la Torre (2016) proposed to factorize the affinity matrix into smaller matrices and incorporated global geometric constraints.

Zhang et al. (2019c) studied kernelized graph matching, where the node and edge similarities are kernels, which allows to express the graph matching problem again as Koopmans-Beckmann's QAP in the associated Hilbert space.

Inspired by established methods for Maximum-A-Posteriori (MAP) inference in conditional random fields, Swoboda et al. (2017) studied several Lagrangean decompositions of the graph matching problem, which are solved by dual ascent algorithms also known as message passing.

Specific message passing schedules and update mechanisms leading to state-of-the-art performance in graph matching tasks have been identified experimentally.

Recently, functional representation for graph matching has been proposed as a generalizing concept with the additional goal to avoid the construction of the affinity matrix .

Graph edit distance.

A related concept studied in computer vision is the graph edit distance, which measures the minimum cost required to transform a graph into another graph by adding, deleting and substituting vertices and edges.

The idea has been proposed for pattern recognition tasks more than 30 years ago (Sanfeliu & Fu, 1983) .

However, its computation is NP-hard, since it generalizes the maximum common subgraph problem (Bunke, 1997) .

Moreover, it is also closely related to the quadratic assignment problem (Bougleux et al., 2017) .

Recently several elaborated exact algorithms for computing the graph edit distance have been proposed (Gouda & Hassaan, 2016; Lerouge et al., 2017; , but are still limited to small graphs.

Therefore, heuristics based on the assignment problem have been proposed (Riesen & Bunke, 2009 ) and are widely used in practice (Stauffer et al., 2017) .

The original approach requires cubic running time, which can be reduced to quadratic time using greedy strategies (Riesen et al., 2015a; , and even linear time for restricted cost functions (Kriege et al., 2019a) .

Network alignment.

The problem of network alignment typically is defined analogously to Equation (1), where in addition a similarity function between pairs of nodes is given.

Most algorithms follow a two step approach: First, an n ?? n node-to-node similarity matrix M is computed from the given similarity function and the topology of the two graphs.

Then, in the second step, an alignment is computed by solving the assignment problem for M .

Singh et al. (2008) proposed ISORANK, which is based on the adjacency matrix of the product graph K = A s ??? A t of G s and G t , where ??? denotes the Kronecker product.

The matrix M is obtained by applying PAGERANK (Page et al., 1999 ) using a normalized version of K as the GOOGLE matrix and the node similarities as the personalization vector.

Kollias et al. (2012) proposed an efficient approximation of ISORANK by decomposition techniques to avoid generating the product graph of quadratic size.

Zhang (2016) present an extension supporting vertex and edge similarities and propose its computation using nonexact techniques.

Klau (2009) proposed to solve network alignment by linearizing the quadratic optimization problem to obtain an integer linear program, which is then approached via Lagrangian relaxation.

Bayati et al. (2013) developed a message passing algorithm for sparse network alignment, where only a small number of matches between the vertices of the two graphs are allowed.

The techniques briefly summarized above aim to find an optimal correspondence according to a clearly defined objective function.

In practical applications, it is often difficult to specify node and edge similarity functions.

Recently, it has been proposed to learn such functions for a specific task, e.g., in form of a cost model for the graph edit distance (Cort??s et al., 2019) .

A more principled approach has been proposed by Caetano et al. (2009) where the goal is to learn correspondences.

The method presented in this work is related to different lines of research.

Deep graph matching procedures have been investigated from multiple perspectives, e.g., by utilizing local node feature matchings and cross-graph embeddings .

The idea of refining local feature matchings by enforcing neighborhood consistency has been relevant for several years for matching in images (Sattler et al., 2009) .

Furthermore, the functional maps framework aims to solve a similar problem for manifolds (Halimi et al., 2019) .

Deep graph matching.

Recently, the problem of graph matching has been heavily investigated in a deep fashion.

For example, Zanfir & Sminchisescu (2018) ; Wang et al. (2019b) ; Zhang & Lee (2019) develop supervised deep graph matching networks based on displacement and combinatorial objectives, respectively.

Zanfir & Sminchisescu (2018) model the graph matching affinity via a differentiable, but unlearnable spectral graph matching solver (Leordeanu & Hebert, 2005) .

In contrast, our matching procedure is fully-learnable.

Wang et al. (2019b) use node-wise features in combination with dense node-to-node cross-graph affinities, distribute them in a local fashion, and adopt sinkhorn normalization for the final task of linear assignment.

Zhang & Lee (2019) propose a compositional message passing algorithm that maps point coordinates into a high-dimensional space.

The final matching procedure is done by computing the pairwise inner product between point embeddings.

However, neither of these approaches can naturally resolve violations of inconsistent neighborhood assignments as we do in our work.

Xu et al. (2019b) tackles the problem of graph matching by relating it to the Gromov-Wasserstein discrepancy (Peyr?? et al., 2016) .

In addition, the optimal transport objective is enhanched by simultaneously learning node embeddings which shall account for the noise in both graphs.

In a follow-up work, Xu et al. (2019a) extend this concept to the tasks of multi-graph partioning and matching by learning a Gromov-Wasserstein barycenter.

Our approach also resembles the optimal transport between nodes, but works in a supervised fashion for sets of graphs and is therefore able to generalize to unseen graph instances.

In addition, the task of network alignment has been recently investigated from multiple perspectives.

Derr et al. (2019) leverage CYCLEGANs (Zhu et al., 2017) to align NODE2VEC embeddings (Grover & Leskovec, 2016) and find matchings based on the nearest neighbor in the embedding space.

design a deep graph model based on global and local network topology preservation as auxiliary tasks.

Heimann et al. (2018) utilize a fast, but purely local and greedy matching procedure based on local node embedding similarity.

Furthermore, Bai et al. (2019) use shared graph neural networks to approximate the graph edit distance between two graphs.

Here, a (non-differentiable) histogram of correspondence scores is used to fine-tune the output of the network.

In a follow-up work, Bai et al. (2018) proposed to order the correspondence matrix in a breadth-first-search fashion and to process it further with the help of traditional CNNs.

Both approaches only operate on local node embeddings, and are hence prone to match correspondences inconsistently.

Intra-and inter-graph message passing.

The concept of enhanching intra-graph node embeddings by inter-graph node embeddings has been already heavily investigated in practice Wang et al., 2019b; Xu et al., 2019d) .

and Wang et al. (2019b) enhance the GNN operator by not only aggregating information from local neighbors, but also from similar embeddings in the other graph by utilizing a cross-graph matching procedure.

Xu et al. (2019d) leverage alternating GNNs to propagate local features of one graph throughout the second graph.

Wang & Solomon (2019) tackle the problem of finding an unknown rigid motion between point clouds by relating it to a point cloud matching problem followed by a differentiable SVD module.

Intra-graph node embeddings are passed via a Transformer module before feature matching based on inner product similarity scores takes place.

However, neither of these approaches is designed to achieve a consistent matching, due to only operating on localized node embeddings which are alone not sufficient to resolve ambiguities in the matchings.

Nonetheless, we argue that these methods can be used to strengthen the initial feature matching procedure, making our approach orthogonal to improvements in this field.

Neighborhood consensus for image matching.

Methods to obtain consistency of correspondences in local neighborhoods have a rich history in computer vision, dating back several years (Sattler et al., 2009; Sivic & Zisserman, 2003; Schmid & Mohr, 1997) .

They are known for heavily improving results of local feature matching procedures while being computational efficient.

Recently, a deep neural network for neighborhood consensus using 4D convolution was proposed (Rocco et al., 2018) .

While it is related to our method, the 4D convolution can not be efficiently transferred to the graph domain directly, since it would lead to applying a GNN on the product graph with O(n 2 ) nodes and O(n 4 ) edges.

Our algorithm also infers errors for the (sparse) product graph but performs the necessary computations on the original graphs.

Functional maps.

The functional maps framework was proposed to provide a way to define continuous maps between function spaces on manifolds and is commonly applied to solve the task of

@highlight

We develop a deep graph matching architecture which refines initial correspondences in order to reach neighborhood consensus.

@highlight

A framework for answering graph matching questions consisting of local node embeddings with a message passing refinement step.

@highlight

A two-stage GNN-based architecture to establish correspondences between two graphs that performs well on real-world tasks of image matching and knowledge graph entity alignment.
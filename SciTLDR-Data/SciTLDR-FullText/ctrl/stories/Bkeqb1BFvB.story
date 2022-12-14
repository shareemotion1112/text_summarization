Graphs possess exotic features like variable size and absence of natural ordering of the nodes that make them difficult to analyze and compare.

To circumvent this problem and learn on graphs, graph feature representation is required.

Main difficulties with feature extraction lie in the trade-off between expressiveness, consistency and efficiency, i.e. the capacity to extract features that represent the structural information of the graph while being deformation-consistent and isomorphism-invariant.

While state-of-the-art methods enhance expressiveness with powerful graph neural-networks, we propose to leverage natural spectral properties of graphs to study a simple graph feature: the graph Laplacian spectrum (GLS).

We analyze the representational power of this object that satisfies both isomorphism-invariance, expressiveness and deformation-consistency.

In particular, we propose a theoretical analysis based on graph perturbation to understand what kind of comparison between graphs we do when comparing GLS.

To do so, we derive bounds for the distance between GLS that are related to the divergence to isomorphism, a standard computationally expensive graph divergence.

Finally, we experiment GLS as graph representation through consistency tests and classification tasks, and show that it is a strong graph feature representation baseline.

No matter where and at which scale we look, graphs are present.

Social networks, public transport, information networks, molecules, any structural dependency between elements of a global system is a graph.

An important task is to extract information from these graphs in order to understand whether they contain certain structural properties that can be represented and used in downstream machine learning tasks.

In general, graphs are difficult to use as input of standard algorithms because of their exotic features like variable size and absence of natural orientation.

Consequently, graph feature representation with equal dimensionality and dimension-wise alignment is required to learn on graphs.

Any embedding method is traditionally associated to a trade-off between preservation of structural information (expressiveness) and computation time (efficiency) (Cai et al., 2018) .

In the expressiveness, we particularly consider two key attributes of graph feature representation: consistency under deformation and invariance under isomorphism.

The first forces the embedding to discriminate two graphs consistently with their structural dissimilarity.

The second enables to have one representation for each graph, which can be a challenge since one graph has many possible orientations.

In this paper, we propose to analyze the importance of satisfying the introduced criteria through a known but unused, simple, expressive and efficient candidate graph feature representation: the graph Laplacian spectrum (GLS).

The Laplacian matrix of a graph is a well-known object in spectral learning (Belkin & Niyogi, 2002) for several reasons.

First, the Laplacian eigenvalues give many structural information like the presence of communities and partitions (Newman, 2013) , the regularity, the closed-walks enumeration, the diameter or the connectedness of the graph (Brouwer & Haemers, 2011) .

It is also interpretable in term of physics or mechanics (Bonald et al., 2018) .

It is backed by efficient and robust approximate eigen decomposition algorithms enabling to scale on large graphs and huge datasets (Halko et al., 2011) .

These properties give intuition that GLS can be an appropriate candidate for graph representation.

In this paper we go further and analyze additional interesting properties of the Laplacian spectrum through the following contributions: (1) we build a perturbation-based framework to analyze the representation capacity of the GLS, (2) we analyze Interpretation of the GLS The smallest non-zero eigenvalue of the Laplacian is the spectral gap, corresponding the difference between the two largest eigenvalues of the Laplacian.

It contains information about the connectivity of the graph.

High spectral gap means high connectivity.

For example, given a number of vertices in a connected graph, a minimum spectral gap indicates that the graph is a double kite (Marsden, 2013) .

The largest eigenvalue gives a lower bound of the maximal node degree of the graph.

The spectral gap can also be viewed as the difference in energy between the ground state and first excited state of a dynamical system (Cubitt et al., 2015) .

More generally each eigenvalue of the Laplacian corresponds to the energy level of a stable configuration of the nodes in the embedding space (Bonald et al., 2018) .

The lower the energy, the stabler the configuration.

In (Shuman et al., 2016) , the Laplacian eigenvalues correspond to frequencies associated to a Fourier decomposition of any signal living on the vertices of the graph.

Thus, the truncation of the Fourier decomposition acts as filter on the signal.

Characterizing a graph by the some eigenvalues of its Laplacian is thus comparable to characterizing a melody by some fundamental frequencies.

In summary, Laplacian spectrum contains many graph structural information.

Methods to get such information are generally computationally expensive.

In the light of these properties, we go further and analyze in the following sections the capacity of GLS to represent graph structure.

We consider two undirected and weighted graphs G 1 = (V 1 , E 1 , W 1 ) and G 2 = (V 2 , E 2 , W 2 ) with respective adjacency matrix W 1 and W 2 , degree matrix D 1 and D 2 .

These matrices are set with respect to an arbitrary indexing of the nodes.

Laplacian matrix L i of G i is defined as L i = D i ??? W i .

We aim at using the GLS to build fixed-dimensional representation that encodes structural information to compare any graphs G 1 and G 2 that are not aligned nor equally sized.

For the rest of the paper, and without loss of generality we postulate that |V 1 | ??? |V 2 |.

The rest of this section introduces the definitions, hypothesis and notations needed for our theoretical analysis of the GLS.

Definition 1.

Let G = (V, E, W ) a weighted graph with n nodes, with W ??? M n??n the n ?? n weighted adjacency matrices.

We define P ??? M n??n a symmetric matrix with

We define the two following perturbations applied on graph G:

??? Adding isolated nodes: W = W 0 n??m 0 m??n 0 m??m

??? Adding or removing edges:

We call edge-perturbation the addition or removal of edges, and node-perturbation the addition of nodes.

A complete perturbation is done by adding isolated nodes and perturbing the augmented graph with edge-perturbation.

We note that the withdrawal of a node is equivalent to removal of all edges around this nodes.

Moreover, if graph G is unweighted, i.e. with binary adjacency, then edge perturbations P ij ??? {???1, 0, 1}.

Remark 1.

If P = ???W + ?? T W ?? with ?? ??? P(n) then the perturbation is a permutation of the node indexing.

Such a perturbation is not interesting and edge perturbation due to node indexing has to be annihilated by a permutation matrix as in the following Definition 2:

Definition 2.

We say that G P * is a perturbed version of G if we have

i.e. such that P * is the sparsest possible i.e. does not include permutations.

Notations We denote P * the sparsest perturbation as defined in Definition 2.

We denote G the completion of G with isolated nodes.

If M is a matrix associated to G, we denote M the equivalent matrix for G. We denote ??(X) the eigenvalue of a square matrix X in ascending order, ?? i (X) the i th smallest eigenvalue.

Hypothesis 1.

Without loss of generality, we assume that G 2 is a perturbed version of G 1 , i.e. ???P * the sparsest |V 2 |-square perturbation matrix and ?? * ??? P(|V 2 |) a |V 2 |-square permutation matrix such that W 2 = ?? * T W 1 + P * ?? * .

P * is a |V 2 |-square block matrix, with top-left block P * 11 being a |V 1 |-square perturbation matrix for graph G 1 .

Bottom right block P * 22 is the (|V 2 | ??? |V 1 |)-square adjacency matrix of the additional nodes.

P * 12 is the |V 1 | ?? (|V 2 | ??? |V 1 |) adjacency matrix representing the links between graph G 1 and the additional nodes V 2 \ V 1 .

We have defined a notion of continuous deformation of graphs.

This deformation has a natural and simple interpretation: any graph G 2 is a perturbed version of graph G 1 , and the larger the perturbation the higher the structural dissimilarity between G 1 and G 2 .

The next section uses the previously presented mathematical framework to analyze the consistency of the Laplacian spectrum as graph representation and its natural link to graph isomorphism problem.

We place ourselves under the Hypothesis 1 saying that the difference between graphs G 1 and G 2 is characterized by the unknown deformation P * .

A good embedding of these graphs should be close when level of deformation is low, and far otherwise.

This level of deformation can be quantified by the global and node-wise entries of P * .

These features are by construction present in the Laplacian of P * , denoted L P * .

We use this idea to propose an analysis of the distance between to GLS.

Two graphs G 1 and G 2 are isomorphic if and only if ????? ??? P( Merris, 1994) , hence when they are structurally equivalent irrespective to the vertex ordering.

Several papers has proposed to use a notion of divergence to graph isomophism (DGI) to compare graphs (Grohe et al., 2018; Rameshkumar et al., 2013) .

The DGI between graphs G 1 and G 2 is generally the minimal Frobenius norm of the difference between L 1 and ?? ???1 L 2 ??. Considering this definition, the following Lemma links the graph-isomorphism problem and the Laplacian of the hypothetical perturbation P * and show that this divergence is the norm of L P * : Lemma 1.

Using the notations from Hypothesis 1, we have:

We remind that graph isomorphism is at best solved in quasipolynomial time (Babai, 2016) and can not be used in practice for large graphs and datasets.

The following Proposition show how the distance between GLS relaxes the isomorphism-based graph divergence.

Proposition 1.

Using Hypothesis 1 and Lemma 1:

The above result tells us that the higher the difference between GLS, the larger the hypothetical perturbation P * , i.e. the higher the structural dissimilarity.

We now study the implication of GLS closeness.

This problem tackles the notion of non-isomorphic L-cospectrality, i.e. the idea that two graphs can have equal eigenvalues while having different Laplacian matrix (Brouwer & Haemers, 2011) .

The following proposition gives a simple insight into the problem of spectral characterization in our perturbation-based framework:

This proposition shows that equal spectrum means equal graphs only when eigenvectors are also equal.

Otherwise, L-cospectrality for non-isomorphic graphs tells us that there exists families of graphs that are not fully determined by their spectrum.

These families are characterized by some structural properties such that two non-isomorphic graphs with equal Laplacian spectrum share these properties but not their adjacency (Van Dam & Haemers, 2003) .

In practice, this is not a problem.

First, almost all graphs are determined by their spectrum (Brouwer & Haemers, 2011) .

Second, equal GLS indicates the precious information that graphs share common statistical, mechanical, physical or chemical properties (see Section 2), no matter the adjacency matrix.

These physical properties are what we seek to represent when representing graphs for ML tasks.

Third, non-isomorphic L-cospectrality concerns equally sized graphs which is not likely with respect to all possible reallife graphs.

When the studied dataset contains specifically L-cospectral non-isomorphic graphs and when the task requires unique representation property, GLS is not appropriate and more sophisticated and powerful embedding methods taking for example eigenvectors into account (Verma & Zhang, 2017) should be studied and used.

Otherwise, i.e. in almost every situations, according to previously presented results, GLS characterizes the graph and is directly related to the hypothetical perturbation P * .

Nevertheless, we accordingly propose the Proposition 3 to better understand GLS proximity even when graphs are non-isomorphic cospectral.

Proposition 3.

The closer the GLS, the closer to unitary-similarity the Laplacian matrices.

We remind that two real n-square matrices A and B are unitary-similarity if there exists an orthogonal matrix O such that B = OAO T .

Similarity is an equivalence relation on the space of square-matrices.

Moreover, divergence to unitary-similarity is a relaxed version of the divergence to graph-isomorphism (Grohe et al., 2018) , where the permutation matrix space is replaced by a unitary matrix space.

Finally from Proposition 1 and 3 we can bound the distance between GLS as follows:

In this section, we have shown that structural similarity (divergence) between graphs can be reasonably approximated by the similarity (divergence) between their GLS.

Previous section showed the capacity of the distance between Laplacian spectrum to serve as proxy for graph similarity.

In practice, a fixed embedding dimension d must be chosen for all graphs in dataset D. According to previous analysis, the most obvious dimension is d = max G???D |V | and all graphs with less than d nodes may be pad with isolated nodes.

We note that padding with isolated nodes is equivalent than adding zeros in the GLS.

Nevertheless, in some datasets, some graphs can be significantly larger and the padding can become abusive.

We therefore propose for these graph to have d < max G???D |V |.

We simply truncate the GLS such that we keep only the highest d eigenvalues.

This method also enables to save computation time.

The problem with this method is that we may lose information for graphs with more than d nodes.

In practice, for large graphs, the contribution of the lowest eigenvalues to the distance between GLS as a proxy for graph divergence is negligible.

In particular, large graph have many sparse areas, such that many eigenvalues are very low, hence truncating the bottom part of the GLS may not be a problem.

We assess the impact of the truncation in the experimental section.

Though, we can also propose several ways to avoid this problem, like embedding the lowest eigenvalues with simple statistics, like moments or histograms.

In the experimental section, we do not use this trick.

All experiments can reproduced using the notebook given at: https://github.com/ researchsubmission/ICLR2020/.

As a first illustration of deformation-based results presented in Section 4, we propose to use ErdosRnyi random graphs (Erd??s & R??nyi, 1959) with parameter p = 0.05.

We focus on three simple experiments.

First, the distance between the Laplacian spectrum of a graph and a perturbed version of this graph is related to the number of perturbations.

We can find the experimental illustration in Figure 1 .

We see that the number of perturbations is roughly equally related to the distance between GLS features for edge addition and edge withdrawal, i.e. for graph sparsity decrease or increase.

A relation between graph sparsity and Laplacian eigenvalues can be seen for example through the Gershgorin circle theorem (Gershgorin, 1931) .

Second, we mentioned that when a graph is significantly bigger than other graphs of a dataset, w can use a truncated GLS (t-GLS).

This method both saves computation time thanks to iterative eigenvalues algorithms and avoid the addition of isolated nodes in all other graphs.

In Figure 3 , we show results of experiments showing that t-GLS is consistent with node addition.

As experimental setup, we take a reference graph with n nodes and compute its GLS.

Then we add a randomly connected node and compute the t-GLS of the new graph, by keeping only the n largest eigenvalues.

We repeat it 20 times.

We compute the L 2 -distance to reference GLS, for different levels of connectivity for the additional nodes.

We first observe that the t-GLS is consistent with node addition.

We also confirm our previous theoretical results by observing that the more connected the additional nodes, the higher the GLS divergence.

We evaluate spectral feature embedding with a classification task on molecular graphs and social network graphs.

Experimental setup for classification task is given in Appendix E.

We assume here that two structurally close graphs belong to the same class.

We challenge this assumption with the following experiments. (Hamilton et al., 2017) .

All deep learning methods are end-to-end graph classifers.

A description of these models is given in the related work, Section 6.

All values reported in Table 1 and Table 2 are taken from the above-mentioned papers.

On molecular graphs We use five datasets for the experiments: Mutag (MT), Enzymes (EZ), Proteins Full (PF), Dobson and Doig (DD) and National Cancer Institute (NCI1) (Kersting et al., 2016) .

All graphs are chemical components.

Nodes are atoms or molecules and edges represent checmical or electrostatic bindings.

We note that molecular graphs contain node attributes, that are used by some models presented in Table 1 .

We let the question of the relevance of comparing models with slightly different inputs to the discretion of the reader.

Description and statistics of molecular datasets are presented in Table 3 75.5 ?? 0.9 79.4 ?? 0.9 74.4 ?? 0.5 CapsGNN* 86.7 ?? 6.9 54.7 ?? 5.7 76.3 ?? 3.6 75.4 ?? 4.2 78.4 ?? 1.6 GIN-0* 89.4 ?? 5.6 -76.2 ?? 2.8 -82.7 ?? 1.7 GraphSAGE* 85.1 ?? 7.6 -75.9 ?? 3.2 -77.7 ?? 1.5 GLS + SVC 87.9 ?? 7.0 40.7 ?? 6.3 75.3 ?? 3.5 74.3 ?? 3.5 73.3 ?? 2.1 Table 1 : Accuracy (%) of classification with different graph representations, on molecular graphs.

SVC stands for support vector classifier.

Comparative models are divided into two groups: feature + SVC and end-to-end deep learning.

*Models using node attributes.

COLLAB.

All graphs are social networks.

The graphs of these datasets do not contain node attributes.

Therefore, we can more appropriately compare GLS + SVC to deep learning based classification.

Statistics about social networks datasets are presented in Table 4 , Appendix F.

74.0 ?? 3.4 51.9 ?? 3.8 --79.0 ?? 1.8 DGCNN 70.0 ?? 0.9 47.8 ?? 0.9 76.0 ?? 1.7 -73.8 ?? 0.5 CapsGNN 73.1 ?? 4.8 50.3 ?? 2.7 -52.9 ?? 1.5 79.6 ?? 0.9 GIN-0 75.1 ?? 5.1 52.3 ?? 2.8 92.4 ?? 2.5 57.5 ?? 1.5 80.2 ?? 1.9 GraphSAGE 72.3 ?? 5.3 50.9 ?? 2.2 ---GLS + SVC 73.2 ?? 4.2 48.5 ?? 2.5 87.4 ?? 3.4 52.0 ?? 1.8 78.5 ?? 1.1 Table 2 : Classification accuracy (%) of different deep learning based models plus ours over standard social networks datasets.

Graphs of these datasets does not have node features.

SVC stands for support vector classifier.

The classification results above illustrate the capacity of GLS to capture graph structural information, under the assumption that structurally close graphs belong to the same class.

The graph neural-networks are globally more expressive since they can leverage specific information for graph classification since is end-to-end.

In particular, they obtain strong results when there are node labels (see molecular experiments 5.2).

Nevertheless, GLS is a simple way to represent graphs in an unsupervised manner, with theoretical background, simplicity of implementation (eigendecomposition is accessible to anyone interested in any computer) and competitive downstream classification results.

On the reasonability of using truncated GLS We assess the impact of truncating the GLS.

Using truncated GLS (t-GLS) enables to (1) reduce the computational cost for large graphs and (2) reduce the dimensionality of the graph representation for all graphs.

Results are presented in Figure 3 for molecular datasets.

We see that truncating GLS is not highly impacting classification results.

Only ENZYME multiclass classification, which is a particularly difficult task (see experiments in Section 5.2), suffers from truncation.

Additional insight about the t-GLS is given in Appendix G.

We propose to divide graph feature representation into three categories: graph kernel methods, feature-based methods and deep learning.

Graph kernel methods Kernel methods create a high-dimensional feature representation of data.

The kernel trick (Shawe-Taylor et al., 2004) avoids to compute explicitly the coordinates in the feature space, only the inner product between all pairs of data image: it is an implicit embedding methods.

These methods are applied to graphs (Nikolentzos et al., 2018; .

It consists in performing pairwise comparisons between atomic substructures of the graphs until a good representative dictionary is found.

The embedding of a graph is then the number of occurrences of these substructures within it.

These substructures can be graphlets (Yanardag & Vishwanathan, 2015) , subtree patterns (Shervashidze et al., 2011 ), random walks (Vishwanathan et al., 2010 or paths (Borgwardt & Kriegel, 2005) .

The main difficulty lives in the choice of appropriate algorithm and kernel that accept graphs with variable size and capture useful feature for downstream task.

Moreover, kernel methods can be computationally expensive but techniques like the Nystrm algorithm (Williams & Seeger, 2001) allow to lower the number of comparison with a low rank approximation of the similarity matrix.

Feature-based methods Feature-based representation methods (Barnett et al., 2016) represent each graph as the concatenation of features.

Generally, the feature-based representation can offer a certain degree of interpretability and transparency.

The most basic ones are the number of nodes or edges, the histogram of node degrees.

These simple graph-level features offers by construction the sought isormorphism-invariance but suffer from low expressiveness.

More sophisticated algorithms consider features based on attributes of random walks on the graph (G??mez & Delvenne) while others are graphlet based (Kondor et al., 2009) . (Kondor & Borgwardt, 2008) explicitly built permutation-invariant features by mapping the adjacency matrix to a function on the symmetric group. (Verma & Zhang, 2017) proposed a family of graph spectral distances to build graph features.

Experimental work in (de Lara & Pineau, 2018) used normalized Laplacian spectrum with random forest for graph classification with promising results. (Wilson & Zhu, 2008) analyzes the cospectrality of different graph matrices and studies experimentally the representational power of their spectra.

These two last works are directly related to the current work.

Nevertheless, in both cases, the theoretical analysis is absent and comparative experiment with current benchmarks and methods is limited.

in this paper we propose a response to these concerns.

Deep learning based methods GNNs learn representation of nodes of a graph by leveraging together their attributes, information on neighboring nodes and the attributes of the connecting edges.

When graphs have no vertex features, the node degrees are used instead.

To create graph-level representation instead of node representation, node embeddings are pooled by a permutation invariant readout function like summation or more sophisticated information preserving ones Zhang et al., 2018) .

A condition of optimality for readout function is presented in (Xu et al., 2019) .

Recently, Xinyi & Chen (2018) levraged capsule networks (Sabour et al., 2017) , neural units designed to enable to better preserve information at pooling time.

Other popular evolution of GNNs formulate convolution-like operations on graphs.

Formulation in spectral domain (Bruna et al., 2013; Defferrard et al., 2016 ) is limited to the processing of different signals on a single graph structure, because they rely on the fixed spectrum of the Laplacian.

Conversly, formulation in spatial domain are not limited to one graph structure (Atwood & Towsley, 2016; Duvenaud et al., 2015; Niepert et al., 2016; Hamilton et al., 2017) and can infer information from unseen graph structures.

At the same time, alternative to GNN exist and are related to random walk embedding.

In , neural networks help to sample paths which preserve significant graph properties.

Other approaches transforms graphs into sequence of nodes embedding passed into a recurrent neural network (RNN) Pineau & de Lara, 2019) to get useful embedding.

These models do not inherently include isomorphism-invariance but greedy learn it by seeing the same graph numerous times with different node ordering and embedding.

These methods are powerful and globally obtain a high level of expressiveness (see experimental section 5.2).

In this paper, we analyzed the graph Laplacian spectrum (GLS) as whole graph representation.

In particular, we showed that comparing two GLS is a good proxy for the divergence between two graphs in term of structural information.

We coupled these results to the natural invariance to isomorphism, the simplicity of implementation, the computational efficiency offered by modern randomized algorithms and the rare occurrence of detrimental L-cospectral non-isomorphic graphs to propose the GLS as a strong baseline graph feature representation.

A PROOF OF LEMMA 1

Proof.

with L P * = diag(P * 1) ??? P * = D P * ??? P * and 1 the unit vector.

Therefore,

Moreover, from Weyl's eigenvalues inequalities and since eigenvalues are isomorphism invariant:

Hence:

Now let (??, x) be any eigen couple of a matrix M ??? M n??n .

We can always pick i ??? {1 . . .

n} and build x such that |x i | = 1 and |x j =i | < 1.

Hence:

Using previous results we get:

Proof.

We remind that the Forbenius norm is unitarily invariant thanks to the cyclic property of the trace.

For anyP ??? O(|V 2 |) we have:

We also have that

Hence:

Proof.

Denoting O(n) the n-orthogonal matrices group (orthogonal since real), we want to show that:

We denote

.

Since Q 1 is unitary and using property of Frobenius norm, we have, ???O ??? O(|V 2 |):

We know that Q 1 and Q 2 are orthogonal since they are respectively eigenvector matrices of symmetric matrix L 1 and L 2 .

We therefore have:

Hence,

with S(n) the permutation group of {1 . . .

n}.

For classification, we use standard 10-folds cross validation setup.

Each dataset is divided into 10 folds such that the class proportions are preserved in each fold for all datasets.

These folds are then used for cross-validation i.e, one fold serves as the testing set while the other ones compose the training set.

Results are averaged over all testing sets.

All figures gathered in the tables of results are build using this setup.

For the dimension d ??? 1, max G???D |V | , representing the number of eigenvalues we keep to build the truncated GLS, we chose the percentile 95 of the distribution of graph sizes in each dataset, i.e. we truncate the 5% smallest eigenvalues.

Considering weak truncation impact (see Section 5.2, when we have large datasets containing large graphs, like the two REDDIT datasets, we can truncate more severely to make the problem computationally more efficient.

In particular considering that GLS approached as a simple baseline more than a final graph representation for large scale usage.

We use the support vector classifier (SVC) from scikit-learn (Pedregosa et al., 2011) .

We impose Radial Basis Function as kernel, i.

2 ).

It is a similarity measure related to L 2 -norm between GLS.

Hence, our theoretical results remain consistent with our experiments.

Hyper parameters C and ?? are tuned among respectively {0.5, 1, 5} and {0.0001, 0.001, 0.01, 0.1, 0.5, 1, 5} for the molecular datasets, and {0.5, 1, 5, 25, 50} and {0.0001, 0.001, 0.01, 0.1} for the social network datasets.

In practice, using a global pool for all the datasets gives equivalent results, but hyperparameter inference becomes expensive with a too large grid, in particular in a 10-fold cross validation setup.

We use a nested hyperparameter search cross-validation for each of the 10 folds: in each 90% training fold we performe a 5-fold random search cross-validation before training.

We therefore avoid the problem of overfitting related to model selection that appear when using non-nested cross-validation (Cawley & Talbot, 2010) .

We additionally put our results with k-nearest neighbor algorithm with L 2 norm, in order to illustrate the notion of proximity we introduced: neighboring GLS is related to structurally close graphs (i.e. eventually graphs from the same class).

We use five molecular datasets and five social network datasets for the experiments (Kersting et al., 2016) .

Tables 3 and 4 gives statistics of the differents datasets.

All used datasets can be found at https://ls11-www.cs.tu-dortmund.de/staff/morris/ graphkerneldatasets (Kersting et al., 2016) .

Molecular graphs datasets are Mutag (MT), Enzymes (EZ), Proteins Full (PF), Dobson and Doig (DD) and National Cancer Institute (NCI1).

In MT, the graphs are either mutagenic and not mutagenic.

EZ graphs are tertiary structures of proteins from the 6 Enzyme Commission top level classes.

In DD, compounds are secondary structures of proteins that are enzyme or not.

PF is a subset of DD without the largest graphs.

In NCI1, graphs are anti-cancer or not.

The graphs of these datasets have node labels that can be leverages by graph neural networks.

Social networks datasets are IMDB-Binary (IMBD-B), IMDB-Multi (IMDB-M), REDDIT-Binary (REDDIT-B), REDDIT-5K-Multi (REDDIT-M) and COLLAB.

REDDIT-B and REDDIT-M contain graphs representing discussion thread, with edges between users (nodes) when one responded to the other's comment.

Classes are the subreddit topics from which thread have originated.

IMDB-B and IMDB-M contain networks of actors that appeared together within the same movie.

IMDB-B contains two classes for action or romance genres and IMDB-M three classes for comedy, romance and sci-fi.

COLLAB graphs represent scientific collaborations, with edge between two researchers meaning that they co-authored a paper.

Labels of the graphs correspond to subfields of Physics.

The graphs of these datasets have no node attributes and therefore enable fair comparison with deep learning methods.

Figure 4 illustrates the reasonability of using only the highest eigenvalues of the Laplacian spectrum as whole-graph feature representation.

We take the original and final graphs of the deformationconsistency test presented in Figure 3 .

We compute the L 2 distance between t-GLS with dimension d and divide it by d, for d varying from 1 to 15 .

The objective is to confirm that first eigenvalues Figure 4 : Illustration of the relative importance of the dimensionality of GLS-embedding, after the iterative addition of 20 new nodes with respectively 0, 1, 2 and 3 random connections with graph, for respectively synthetic a 80-nodes Erdos-Reyni graph (left) and a 28-nodes molecular graph from MUTAG dataset (right).

We see that the first largest eigenvalues of the Laplacian are the most important to discriminate a graph and its perturbed version.

are relatively more important to discriminate to structurally different graphs, which is the case.

We note that for the Erdos-Reyni case with few connected additional nodes, first eigenvalues are not as relatively important as for the other example.

In fact, adding nodes with stochastic connections is the construction process of Erdos-Reyni graphs.

Hence, discriminating augmented graph from the original one is difficult based only on the structural information.

<|TLDR|>

@highlight

We study theoretically the consistency the Laplacian spectrum and use it as whole-graph embeddding

@highlight

This paper forcuses on the laplacian spectrum of a graph as means to generate a representation to be used to compare graphs and classify them.

@highlight

This work proposed to use Graph Laplacian spectrum to learn graph representation.
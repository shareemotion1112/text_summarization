Graph Convolutional Networks (GCNs) have recently been shown to be quite successful in modeling graph-structured data.

However, the primary focus has been on handling simple undirected graphs.

Multi-relational graphs are a more general and prevalent form of graphs where each edge has a label and direction associated with it.

Most of the existing approaches to handle such graphs suffer from over-parameterization and are restricted to learning representations of nodes only.

In this paper, we propose CompGCN, a novel Graph Convolutional framework which jointly embeds both nodes and relations in a relational graph.

CompGCN leverages a variety of entity-relation composition operations from Knowledge Graph Embedding techniques and scales with the number of relations.

It also generalizes several of the existing multi-relational GCN methods.

We evaluate our proposed method on multiple tasks such as node classification, link prediction, and graph classification, and achieve demonstrably superior results.

We make the source code of CompGCN available to foster reproducible research.

Graphs are one of the most expressive data-structures which have been used to model a variety of problems.

Traditional neural network architectures like Convolutional Neural Networks (Krizhevsky et al., 2012) and Recurrent Neural Networks (Hochreiter & Schmidhuber, 1997) are constrained to handle only Euclidean data.

Recently, Graph Convolutional Networks (GCNs) (Bruna et al., 2013; Defferrard et al., 2016) have been proposed to address this shortcoming, and have been successfully applied to several domains such as social networks (Hamilton et al., 2017) , knowledge graphs (Schlichtkrull et al., 2017; Shang et al., 2019) , natural language processing (Marcheggiani & Titov, 2017; Vashishth et al., 2018a; b; , drug discovery (Ramsundar et al., 2019) , crystal property prediction (Sanyal et al., 2018) , and natural sciences (Fout et al., 2017) .

However, most of the existing research on GCNs (Kipf & Welling, 2016; Hamilton et al., 2017; Veličković et al., 2018) have focused on learning representations of nodes in simple undirected graphs.

A more general and pervasive class of graphs are multi-relational graphs 1 .

A notable example of such graphs is knowledge graphs.

Most of the existing GCN based approaches for handling relational graphs (Marcheggiani & Titov, 2017; Schlichtkrull et al., 2017) suffer from overparameterization and are limited to learning only node representations.

Hence, such methods are not directly applicable for tasks such as link prediction which require relation embedding vectors.

Initial attempts at learning representations for relations in graphs (Monti et al., 2018; Beck et al., 2018) have shown some performance gains on tasks like node classification and neural machine translation.

There has been extensive research on embedding Knowledge Graphs (KG) Wang et al., 2017) where representations of both nodes and relations are jointly learned.

These methods are restricted to learning embeddings using link prediction objective.

Even though GCNs can 1.

We propose COMPGCN, a novel framework for incorporating multi-relational information in Graph Convolutional Networks which leverages a variety of composition operations from knowledge graph embedding techniques to jointly embed both nodes and relations in a graph.

2.

We demonstrate that COMPGCN framework generalizes several existing multi-relational GCN methods (Proposition 4.1) and also scales with the increase in number of relations in the graph (Section 6.3).

3.

Through extensive experiments on tasks such as node classification, link prediction, and graph classification, we demonstrate the effectiveness of our proposed method.

The source code of COMPGCN and datasets used in the paper have been made available at http: //github.com/malllabiisc/CompGCN.

Graph Convolutional Networks: GCNs generalize Convolutional Neural Networks (CNNs) to non-Euclidean data.

GCNs were first introduced by Bruna et al. (2013) and later made scalable through efficient localized filters in the spectral domain (Defferrard et al., 2016) .

A first-order approximation of GCNs using Chebyshev polynomials has been proposed by Kipf & Welling (2016) .

Recently, several of its extensions have also been formulated (Hamilton et al., 2017; Veličković et al., 2018; Xu et al., 2019; .

Most of the existing GCN methods follow Message Passing Neural Networks (MPNN) framework (Gilmer et al., 2017) for node aggregation.

Our proposed method can be seen as an instantiation of the MPNN framework.

However, it is specialized for relational graphs.

GCNs for Multi-Relational Graph: An extension of GCNs for relational graphs is proposed by Marcheggiani & Titov (2017) .

However, they only consider direction-specific filters and ignore relations due to over-parameterization.

Schlichtkrull et al. (2017) address this shortcoming by proposing basis and block-diagonal decomposition of relation specific filters.

Weighted Graph Convolutional Network (Shang et al., 2019) utilizes learnable relational specific scalar weights during GCN aggregation.

While these methods show performance gains on node classification and link prediction, they are limited to embedding only the nodes of the graph.

Contemporary to our work, Ye et al. (2019) have also proposed an extension of GCNs for embedding both nodes and relations in multirelational graphs.

However, our proposed method is a more generic framework which can leverage any KG composition operator.

We compare against their method in Section 6.1.

Knowledge Graph Embedding: Knowledge graph (KG) embedding is a widely studied field Wang et al., 2017) with application in tasks like link prediction and question answering (Bordes et al., 2014) .

Most of KG embedding approaches define a score function and train node and relation embeddings such that valid triples are assigned a higher score than the invalid ones.

Based on the type of score function, KG embedding method are classified as translational (Bordes et al., 2013; Wang et al., 2014b) , semantic matching based (Yang et al., 2014; and neural network based (Socher et al., 2013; Dettmers et al., 2018; .

In our work, we evaluate the performance of COMPGCN on link prediction with methods of all three types.

In this section, we give a brief overview of Graph Convolutional Networks (GCNs) for undirected graphs and its extension to directed relational graphs.

GCN on Undirected Graphs: Given a graph G = (V, E, X ), where V denotes the set of vertices, E is the set of edges, and X ∈ R |V|×d0 represents d 0 -dimensional input features of each node.

The node representation obtained from a single GCN layer is defined as:

2 is the normalized adjacency matrix with added self-connections and D is defined as D ii = j (A + I) ij .

The model parameter is denoted by W ∈ R d0×d1 and f is some activation function.

The GCN representation H encodes the immediate neighborhood of each node in the graph.

For capturing multi-hop dependencies in the graph, several GCN layers can be stacked, one on the top of another as follows:

, where k denotes the number of layers,

is layer-specific parameter and H 0 = X .

GCN on Multi-Relational Graphs:

For a multi-relational graph G = (V, R, E, X ), where R denotes the set of relations, and each edge (u, v, r) represents that the relation r ∈ R exist from node u to v. The GCN formulation as devised by Marcheggiani & Titov (2017) is based on the assumption that information in a directed edge flows along both directions.

Hence, for each edge (u, v, r) ∈ E, an inverse edge (v, u, r −1 ) is included in G. The representations obtained after k layers of directed GCN is given by

Here, W k r denotes the relation specific parameters of the model.

However, the above formulation leads to over-parameterization with an increase in the number of relations and hence, Marcheggiani & Titov (2017) use direction-specific weight matrices.

Schlichtkrull et al. (2017)

In this section, we provide a detailed description of our proposed method, COMPGCN.

The overall architecture is shown in Figure 1 .

We represent a multi-relational graph by G = (V, R, E, X , Z) as defined in Section 3 where Z ∈ R |R|×d0 denotes the initial relation features.

Our model is motivated by the first-order approximation of GCNs using Chebyshev polynomials (Kipf & Welling, 2016) .

Following Marcheggiani & Titov (2017) , we also allow the information in a directed edge to flow along both directions.

Hence, we extend E and R with corresponding inverse edges and GCN Marcheggiani & Titov (2017) O(Kd 2 ) Weighted- GCN Shang et al. (2019) O(Kd 2 + K|R|) Relational- GCN Schlichtkrull et al. (2017) O(BKd 2 + BK|R|) relations, i.e.,

and R = R ∪ R inv ∪ { }, where R inv = {r −1 | r ∈ R} denotes the inverse relations and indicates the self loop.

Unlike most of the existing methods which embed only nodes in the graph, COMPGCN learns a

Representing relations as vectors alleviates the problem of over-parameterization while applying GCNs on relational graphs.

Further, it allows COMPGCN to exploit any available relation features (Z) as initial representations.

To incorporate relation embeddings into the GCN formulation, we leverage the entity-relation composition operations used in Knowledge Graph embedding approaches (Bordes et al., 2013; , which are of the form e o = φ(e s , e r ).

is a composition operator, s, r, and o denote subject, relation and object in the knowledge graph and e (·) ∈ R d denotes their corresponding embeddings.

In this paper, we restrict ourselves to non-parameterized operations like subtraction (Bordes et al., 2013) , multiplication (Yang et al., 2014) and circular-correlation .

However, COMPGCN can be extended to parameterized operations like Neural Tensor Networks (NTN) (Socher et al., 2013) and ConvE (Dettmers et al., 2018) .

We defer their analysis as future work.

As we show in Section 6, the choice of composition operation is important in deciding the quality of the learned embeddings.

Hence, superior composition operations for Knowledge Graphs developed in future can be adopted to improve COMPGCN's performance further.

The GCN update equation (Eq. 1) defined in Section 3 can be re-written as

where N (v) is a set of immediate neighbors of v for its outgoing edges.

Since this formulation suffers from over-parameterization, in COMPGCN we perform composition (φ) of a neighboring node u with respect to its relation r as defined above.

This allows our model to be relation aware while being linear (O(|R|d)) in the number of feature dimensions.

Moreover, for treating original, inverse, and self edges differently, we define separate filters for each of them.

The update equation of COMPGCN is given as:

where x u , z r denotes initial features for node u and relation r respectively, h v denotes the updated representation of node v, and W λ(r) ∈ R d1×d0 is a relation-type specific parameter.

In COMPGCN, we use direction specific weights, i.e., λ(r) = dir(r), given as:

Further, in COMPGCN, after the node embedding update defined in Eq. 2, the relation embeddings are also transformed as follows:

where W rel ∈ R d1×d0 is a learnable transformation matrix which projects all the relations to the same embedding space as nodes and allows them to be utilized in the next COMPGCN layer.

In Table 1 , we present a contrast between COMPGCN and other existing methods in terms of their features and parameter complexity.

Scaling with Increasing Number of Relations To ensure that COMPGCN scales with the increasing number of relations, we use a variant of the basis formulations proposed in Schlichtkrull et al. (2017) .

Instead of independently defining an embedding for each relation, they are expressed as a linear combination of a set of basis vectors.

Formally, let {v 1 , v 2 , ..., v B } be a set of learnable basis vectors.

Then, initial relation representation is given as:

Here, α br ∈ R is relation and basis specific learnable scalar weight.

On Comparison with Relational-GCN Note that this is different from the basis formulation in Schlichtkrull et al. (2017) , where a separate set of basis matrices is defined for each GCN layer.

In contrast, COMPGCN uses embedding vectors instead of matrices, and defines basis vectors only for the first layer.

The later layers share the relations through transformations according to Equation 4.

This makes our model more parameter efficient than Relational-GCN.

We can extend the formulation of Equation 2 to the case where we have k-stacked COMPGCN layers.

Let h k+1 v denote the representation of a node v obtained after k layers which is defined as

Similarly, let h k+1 r denote the representation of a relation r after k layers.

Then, Proof.

For Kipf-GCN, this can be trivially obtained by making weights (W λ(r) ) and composition function (φ) relation agnostic in Equation 5, i.e., W λ(r) = W and φ(h u , h r ) = h u .

Similar reductions can be obtained for other methods as shown in Table 2.

5 EXPERIMENTAL SETUP

In our experiments, we evaluate COMPGCN on the below-mentioned tasks. (Schlichtkrull et al., 2017) .248 -.417

.151 ----KBGAN (Cai & Wang, 2018) .

datasets.

The results of all the baseline methods are taken directly from the previous papers ('-' indicates missing values).

We find that COMPGCN outperforms all the existing methods on 4 out of 5 metrics on FB15k-237 and 3 out of 5 metrics on WN18RR.

Please refer to Section 6.1 for more details.

• Link Prediction is the task of inferring missing facts based on the known facts in Knowledge Graphs.

In our experiments, we utilize FB15k-237 (Toutanova & Chen, 2015) and WN18RR (Dettmers et al., 2018 ) datasets for evaluation.

Following Bordes et al. (2013) , we use filtered setting for evaluation and report Mean Reciprocal Rank (MRR), Mean Rank (MR) and Hits@N.

• Node Classification is the task of predicting the labels of nodes in a graph based on node features and their connections.

Similar to Schlichtkrull et al. (2017) , we evaluate COMPGCN on MUTAG (Node) and AM datasets.

• Graph Classification, where, given a set of graphs and their corresponding labels, the goal is to learn a representation for each graph which is fed to a classifier for prediction.

We evaluate on 2 bioinformatics dataset: MUTAG (Graph) and PTC (Yanardag & Vishwanathan, 2015) .

A summary statistics of the datasets used is provided in Appendix A.2

Across all tasks, we compare against the following GCN methods for relational graphs: (1) Relational-GCN (R-GCN) (Schlichtkrull et al., 2017) which uses relation-specific weight matrices that are defined as a linear combinations of a set of basis matrices.

(2) Directed-GCN (D-GCN) (Marcheggiani & Titov, 2017) has separate weight matrices for incoming edges, outgoing edges, and self-loops.

It also has relation-specific biases.

(3) Weighted-GCN (W-GCN) (Shang et al., 2019 ) assigns a learnable scalar weight to each relation and multiplies an incoming "message" by this weight.

Apart from this, we also compare with several task-specific baselines mentioned below.

Link prediction: For evaluating COMPGCN, we compare against several non-neural and neural baselines: TransE Bordes et al. (2013) , DistMult (Yang et al., 2014) , ComplEx (Trouillon et al., 2016) , R-GCN (Schlichtkrull et al., 2017) , KBGAN (Cai & Wang, 2018) , ConvE (Dettmers et al., 2018) , ConvKB (Nguyen et al., 2018) , SACN (Shang et al., 2019) , HypER (Balažević et al., 2019) , RotatE , ConvR (Jiang et al., 2019) , and VR-GCN (Ye et al., 2019) .

Node and Graph Classification: For node classification, following Schlichtkrull et al. (2017) , we compare with Feat (Paulheim & Fümkranz, 2012) , WL (Shervashidze et al., 2011) , and RDF2Vec .

Finally, for graph classification, we evaluate against PACHYSAN (Niepert et al., 2016) , Deep Graph CNN (DGCNN) (Zhang et al., 2018) , and Graph Isomorphism Network (GIN) (Xu et al., 2019) .

In this section, we attempt to answer the following questions.

Q1.

How does COMPGCN perform on link prediction compared to existing methods? (6.1) Q2.

What is the effect of using different GCN encoders and choice of the compositional operator in COMPGCN on link prediction performance? (6.1) Q3.

Does COMPGCN scale with the number of relations in the graph? (6.3) Q4.

How does COMPGCN perform on node and graph classification tasks? (6.4)

In this section, we evaluate the performance of COMPGCN and the baseline methods listed in Section 5.2 on link prediction task.

The results on FB15k-237 and WN18RR datasets are presented in Table 3 .

The scores of baseline methods are taken directly from the previous papers Cai & Wang, 2018; Shang et al., 2019; Balažević et al., 2019; Jiang et al., 2019; Ye et al., 2019) .

However, for ConvKB, we generate the results using the corrected evaluation code .

Overall, we find that COMPGCN outperforms all the existing methods in 4 out of 5 metrics on FB15k-237 and in 3 out of 5 metrics on WN18RR dataset.

We note that the best performing baseline RotatE uses rotation operation in complex domain.

The same operation can be utilized in a complex variant of our proposed method to improve its performance further.

We defer this as future work.

Next, we evaluate the effect of using different GCN methods as an encoder along with a representative score function (shown in Figure 2 ) from each category: TransE (translational), DistMult (semantic-based), and ConvE (neural network-based).

In our results, X + M (Y) denotes that method M is used for obtaining entity embeddings (and relation embeddings in the case of COMPGCN) with X as the score function as depicted in Figure 2 .

Y denotes the composition operator in the case of COMPGCN.

We evaluate COMPGCN on three non-parametric composition operators inspired from TransE (Bordes et al., 2013) , DistMult (Yang et al., 2014) , and HolE

• Subtraction (Sub): φ(e s , e r ) = e s − e r .

• Multiplication (Mult): φ(e s , e r ) = e s * e r .

• Circular-correlation (Corr): φ(e s , e r )=e s e r

The overall results are summarized in Table 4 .

Similar to Schlichtkrull et al. (2017) , we find that utilizing Graph Convolutional based method as encoder gives a substantial improvement in performance for most types of score functions.

We observe that although all the baseline GCN methods lead to some degradation with TransE score function, no such behavior is observed for COMPGCN.

On average, COMPGCN obtains around 6%, 4% and 3% relative increase in MRR with TransE, DistMult, and ConvE objective respectively compared to the best performing baseline.

The superior performance of COMPGCN can be attributed to the fact that it learns both entity and relation embeddings jointly thus providing more expressive power in learned representations.

Overall, we find that COMPGCN with ConvE (highlighted using · ) is the best performing method for link prediction.

Effect of composition Operator: The results on link prediction with different composition operators are presented in Table 4 .

We find that with DistMult score function, multiplication operator (Mult) gives the best performance while with ConvE, circular-correlation surpasses all other operators.

Overall, we observe that more complex operators like circular-correlation outperform or perform comparably to simpler operators such as subtraction.

In this section, we analyze the scalability of COMPGCN with varying numbers of relations and basis vectors.

For analysis with changing number of relations, we create multiple subsets of Schlichtkrull et al. (2017) and Xu et al. (2019) respectively.

Overall, we find that COMPGCN either outperforms or performs comparably compared to the existing methods.

Please refer to Section 6.4 for more details.

FB15k-237 dataset by retaining triples corresponding to top-m most frequent relations, where m = {10, 25, 50, 100, 237}. For all the experiments, we use our best performing model (ConvE + COMPGCN (Corr)).

Here, we analyze the performance of COMPGCN on changing the number of relation basis vectors (B) as defined in Section 4.

The results are summarized in Figure 3 .

We find that our model performance improves with the increasing number of basis vectors.

We note that with B = 100, the performance of the model becomes comparable to the case where all relations have their individual embeddings.

In Table 4 , we report the results for the best performing model across all score function with B set to 50.

We note that the parameter-efficient variant also gives a comparable performance and outperforms the baselines in all settings.

Effect of Number of Relations: Next, we report the relative performance of COMPGCN using 5 relation basis vectors (B = 5) against COMPGCN, which utilizes a separate vector for each relation in the dataset.

The results are presented in Figure 5 .

Overall, we find that across all different numbers of relations, COMPGCN, with a limited basis, gives comparable performance to the full model.

The results show that a parameter-efficient variant of COMPGCN scales with the increasing number of relations.

Comparison with R-GCN: Here, we perform a comparison of a parameter-efficient variant of COMPGCN (B = 5) against R-GCN on different number of relations.

The results are depicted in Figure 4 .

We observe that COMPGCN with limited parameters consistently outperforms R-GCN across all settings.

Thus, COMPGCN is parameter-efficient and more effective at encoding multirelational graphs than R-GCN.

In this section, we evaluate COMPGCN on node and graph classification tasks on datasets as described in Section 5.1.

The experimental results are presented in Table 5 .

For node classification task, we report accuracy on test split provided by , whereas for graph classification, following Yanardag & Vishwanathan (2015) and Xu et al. (2019) , we report the average and standard deviation of validation accuracies across the 10 folds cross-validation.

Overall, we find that COMPGCN outperforms all the baseline methods on node classification and gives a comparable performance on graph classification task.

This demonstrates the effectiveness of incorporating relations using COMPGCN over the existing GCN based models.

On node classification, compared to the best performing baseline, we obtain an average improvement of 3% across both datasets while on graph classification, we obtain an improvement of 3% on PTC dataset.

In this paper, we proposed COMPGCN, a novel Graph Convolutional based framework for multirelational graphs which leverages a variety of composition operators from Knowledge Graph embedding techniques to jointly embed nodes and relations in a graph.

Our method generalizes several existing multi-relational GCN methods.

Moreover, our method alleviates the problem of over-parameterization by sharing relation embeddings across layers and using basis decomposition. , based on the average number of tails per head and heads per tail, we divide the relations into four categories: one-to-one, one-to-many, many-to-one and many-to-many.

The results are summarized in Table 6 .

We observe that using GCN based encoders for obtaining entity and relation embeddings helps to improve performance on all types of relations.

In the case of one-to-one relations, COMPGCN gives an average improvement of around 10% on MRR compared to the best performing baseline (ConvE + W-GCN).

For one-to-many, many-to-one, and many-to-many the corresponding improvements are 10.5%, 7.5%, and 4%.

These results show that COMPGCN is effective at handling both simple and complex relations.

Table 6 : Results on link prediction by relation category on FB15k-237 dataset.

Following Wang et al. (2014a) , the relations are divided into four categories: one-to-one (1-1), one-to-many (1-N), manyto-one (N-1), and many-to-many (N-N).

We find that COMPGCN helps to improve performance on all types of relations compared to existing methods.

Please refer to Section A.1 for more details.

In this section, we provide the details of the different datasets used in the experiments.

For link prediction, we use the following two datasets:

• FB15k-237 (Toutanova & Chen, 2015) is a pruned version of FB15k (Bordes et al., 2013) dataset with inverse relations removed to prevent direct inference.

• WN18RR (Dettmers et al., 2018) , similar to FB15k-237, is a subset from WN18 (Bordes et al., 2013) dataset which is derived from WordNet (Miller, 1995) .

For node classification, similar to Schlichtkrull et al. (2017) , we evaluate on the following two datasets:

• MUTAG (Node) is a dataset from DL-Learner toolkit 3 .

It contains relationship between complex molecules and the task is to identify whether a molecule is carcinogenic or not.

• AM dataset contains relationship between different artifacts in Amsterdam Museum (de Boer et al., 2012) .

The goal is to predict the category of a given artifact based on its links and other attributes.

Finally, for graph classification, similar to Xu et al. (2019) , we evaluate on the following datasets:

• MUTAG (Graph) Debnath et al. (1991) is a bioinformatics dataset of 188 mutagenic aromatic and nitro compounds.

The graphs need to be categorized into two classes based on their mutagenic effect on a bacterium.

FB15k-237 WN18RR MUTAG (Node) AM MUTAG (Graph) PTC Table 7 : The details of the datasets used for node classification, link prediction, and graph classification tasks.

Please refer to Section 5.1 for more details.

• PTC Srinivasan et al. (1997) is a dataset consisting of 344 chemical compounds which indicate carcinogenicity of male and female rats.

The task is to label the graphs based on their carcinogenicity on rodents.

A summary statistics of all the datasets used is presented in Table 7 .

Here, we present the implementation details for each task used for evaluation in the paper.

For all the tasks, we used COMPGCN build on PyTorch geometric framework (Fey & Lenssen, 2019) .

Link Prediction:

For evaluation, 200-dimensional embeddings for node and relation embeddings are used.

For selecting the best model we perform a hyperparameter search using the validation data over the values listed in Table 8 .

For training link prediction models, we use the standard binary cross entropy loss with label smoothing Dettmers et al. (2018) .

Node Classification:

Following Schlichtkrull et al. (2017), we use 10% training data as validation for selecting the best model for both the datasets.

We restrict the number of hidden units to 32.

We use cross-entropy loss for training our model.

Graph Classification:

Similar to Yanardag & Vishwanathan (2015) ; Xu et al. (2019) , we report the mean and standard deviation of validation accuracies across the 10 folds cross-validation.

Crossentropy loss is used for training the entire model.

For obtaining the graph-level representation, we use simple averaging of embedding of all nodes as the readout function, i.e.,

where h v is the learned node representation for node v in the graph.

For all the experiments, training is done using Adam optimizer (Kingma & Ba, 2014) and Xavier initialization (Glorot & Bengio, 2010 ) is used for initializing parameters.

Number of GCN Layer (K) {1, 2, 3} Learning rate {0.001, 0.0001} Batch size {128, 256} Dropout {0.0, 0.1, 0.2, 0.3} Table 8 : Details of hyperparameters used for link prediction task.

Please refer to Section A.3 for more details.

<|TLDR|>

@highlight

A Composition-based Graph Convolutional framework for multi-relational graphs.

@highlight

The authors develop GCN on multi-relational graphs and propose CompGCN, which leverages insights from knowledge graph embeddings and learns node and relation representations to alleviate the problem of over-parameterization.

@highlight

This paper introduces a GCN framework for multi-relational graphs and generalizes several existing approaches to Knowledge Graph embedding into one framework.
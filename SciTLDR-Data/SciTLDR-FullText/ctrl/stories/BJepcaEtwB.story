We consider the task of few shot link prediction, where the goal is to predict missing edges across multiple graphs using only a small sample of known edges.

We show that current link prediction methods are generally ill-equipped to handle this task---as they cannot effectively transfer knowledge between graphs in a multi-graph setting and are unable to effectively learn from very sparse data.

To address this challenge, we introduce a new gradient-based meta learning framework, Meta-Graph, that leverages higher-order gradients along with a learned graph signature function that conditionally generates a graph neural network initialization.

Using a novel set of few shot link prediction benchmarks, we show that Meta-Graph enables not only fast adaptation but also better final convergence and can effectively learn using only a small sample of true edges.

Given a graph representing known relationships between a set of nodes, the goal of link prediction is to learn from the graph and infer novel or previously unknown relationships (Liben-Nowell & Kleinberg, 2003) .

For instance, in a social network we may use link prediction to power a friendship recommendation system (Aiello et al., 2012) , or in the case of biological network data we might use link prediction to infer possible relationships between drugs, proteins, and diseases (Zitnik & Leskovec, 2017) .

However, despite its popularity, previous work on link prediction generally focuses only on one particular problem setting: it generally assumes that link prediction is to be performed on a single large graph and that this graph is relatively complete, i.e., that at least 50% of the true edges are observed during training (e.g., see Grover & Leskovec, 2016; Kipf & Welling, 2016b; Liben-Nowell & Kleinberg, 2003; Lü & Zhou, 2011) .

In this work, we consider the more challenging setting of few shot link prediction, where the goal is to perform link prediction on multiple graphs that contain only a small fraction of their true, underlying edges.

This task is inspired by applications where we have access to multiple graphs from a single domain but where each of these individual graphs contains only a small fraction of the true, underlying edges.

For example, in the biological setting, high-throughput interactomics offers the possibility to estimate thousands of biological interaction networks from different tissues, cell types, and organisms (Barrios-Rodiles et al., 2005) ; however, these estimated relationships can be noisy and sparse, and we need learning algorithms that can leverage information across these multiple graphs in order to overcome this sparsity.

Similarly, in the e-commerce and social network settings, link prediction can often have a large impact in cases where we must quickly make predictions on sparsely-estimated graphs, such as when a service has been recently deployed to a new locale.

That is to say to link prediction for a new sparse graph can benefit from transferring knowledge from other, possibly more dense, graphs assuming there is exploitable shared structure.

We term this problem of link prediction from sparsely-estimated multi-graph data as few shot link prediction analogous to the popular few shot classification setting (Miller et al., 2000; Lake et al., 2011; Koch et al., 2015) .

The goal of few shot link prediction is to observe many examples of graphs from a particular domain and leverage this experience to enable fast adaptation and higher accuracy when predicting edges on a new, sparsely-estimated graph from the same domain-a task that can can also be viewed as a form of meta learning, or learning to learn (Bengio et al., 1990; 1992; Thrun & Pratt, 2012; Schmidhuber, 1987) in the context of link prediction.

This few shot link prediction setting is particularly challenging as current link prediction methods are generally ill-equipped to transfer knowledge between graphs in a multi-graph setting and are also unable to effectively learn from very sparse data.

Present work.

We introduce a new framework called Meta-Graph for few shot link prediction and also introduce a series of benchmarks for this task.

We adapt the classical gradient-based metalearning formulation for few shot classification (Miller et al., 2000; Lake et al., 2011; Koch et al., 2015) to the graph domain.

Specifically, we consider a distribution over graphs as the distribution over tasks from which a global set of parameters are learnt, and we deploy this strategy to train graph neural networks (GNNs) that are capable of few-shot link prediction.

To further bootstrap fast adaptation to new graphs we also introduce a graph signature function, which learns how to map the structure of an input graph to an effective initialization point for a GNN link prediction model.

We experimentally validate our approach on three link prediction benchmarks.

We find that our MetaGraph approach not only achieves fast adaptation but also converges to a better overall solution in many experimental settings, with an average improvement of 5.3% in AUC at convergence over non-meta learning baselines.

The basic set-up for few shot link prediction is as follows: We assume that we have a distribution p(G) over graphs, from which we can sample training graphs G i ∼ p(G), where each

is defined by a set of nodes V i , edges E i , and matrix of real-valued node attributes X ∈ R |Vi|×d .

When convenient, we will also equivalently represent a graph as

|Vi|×|Vi| is an adjacency matrix representation of the edges in E i .

We assume that each of these sampled graphs, G i , is a simple graph (i.e., contain a single type of relation and no self loops) and that every node v ∈ V i in the graph is associated with a real valued attribute vector x v ∈ R d from a common vector space.

We further assume that for each graph G i we have access to only a sparse subset of the true edges

In terms of distributional assumptions we assume that this p(G) is defined over a set of related graphs (e.g., graphs drawn from a common domain or application setting).

Our goal is to learn a global or meta link prediction model from a set of sampled training graphs G i ∼ p(G), i = 1...n, such that we can use this meta model to quickly learn an effective link prediction model on a newly sampled graph G * ∼ p(G).

More specifically, we wish to optimize a global set of parameters θ, as well as a graph signature function ψ(G i ), which can be used together to generate an effective parameter initialization, φ i , for a local link prediction model on graph G i .

Relationship to standard link prediction.

Few shot link prediction differs from standard link prediction in three important ways:

1.

Rather than learning from a single graph G, we are learning from multiple graphs {G 1 , ..., G n } sampled from a common distribution or domain.

2.

We presume access to only a very sparse sample of true edges.

Concretely, we focus on settings where at most 30% of the edges in E i are observed during training, i.e., where

By "true edges" we mean the full set of ground truth edges available in a particular dataset.

3.

We distinguish between the global parameters, which are used to encode knowledge about the underlying distribution of graphs, and the local parameters φ i , which are optimized to perform link prediction on a specific graph G i .

This distinction allows us to consider leveraging information from multiple graphs, while still allowing for individually-tuned link prediction models on each specific graph.

Relationship to traditional meta learning.

Traditional meta learning for few-shot classification, generally assumes a distribution p(T ) over classification tasks, with the goal of learning global parameters that can facilitate fast adaptation to a newly sampled task T i ∼ p(T ) with few examples.

We instead consider a distribution p(G) over graphs with the goal of performing link prediction on a newly sampled graph.

An important complication of this graph setting is that the individual predictions for each graph (i.e., the training edges) are not i.i.d..

Furthermore, for few shot link prediction we require training samples as a sparse subset of true edges that represents a small percentage of all edges in a graph.

Note that for very small percentages we effectively break all graph structure and recover the supervised setting for few shot classification and thus simplifying the problem.

We now outline our proposed approach, Meta-Graph, to the few shot link prediction problem.

We first describe how we define the local link prediction models, which are used to perform link prediction on each specific graph G i .

Next, we discuss our novel gradient-based meta learning approach to define a global model that can learn from multiple graphs to generate effective parameter initializations for the local models.

The key idea behind Meta-Graph is that we use gradient-based meta learning to optimize a shared parameter initialization θ for the local models, while also learning a parametric encoding of each graph G i that can be used to modulate this parameter initialization in a graph-specific way (Figure 1 ).

In principle, our framework can be combined with a wide variety of GNN-based link prediction approaches, but here we focus on variational graph autoencoders (VGAEs) (Kipf & Welling, 2016b) as our base link prediction framework.

Formally, given a graph G = (V, A, X), the VGAE learns an inference model, q φ , that defines a distribution over node embeddings q φ (Z|A, X), where each row z v ∈ R d of Z ∈ R |V|×d is a node embedding that can be used to score the likelihood of an edge existing between pairs of nodes.

The parameters of the inference model are shared across all the nodes in G, to define the approximate posterior

, where the parameters of the normal distribution are learned via GNNs:

and

The generative component of the VGAE is then defined as

i.e., the likelihood of an edge existing between two nodes, u and v, is proportional to the dot product of their node embeddings.

Given the above components, the inference GNNs can be trained to minimize the variational lower bound on the training data:

where a Gaussian prior is used for p(z).

We build upon VGAEs due to their strong performance on standard link prediction benchmarks (Kipf & Welling, 2016b) , as well as the fact that they have a well-defined probabilistic interpretation that generalizes many embedding-based approaches to link prediction (e.g., node2vec (Grover & Leskovec, 2016) ).

We describe the specific GNN implementations we deploy for the inference model in Section 3.3.

The key idea behind Meta-Graph is that we use gradient-based meta learning to optimize a shared parameter initialization θ for the inference models of a VGAE, while also learning a parametric encoding ψ(G i ) that modulates this parameter initialization in a graph-specific way.

Specifically, given a sampled training graph G i , we initialize the inference model q φi for a VGAE link prediction model using a combination of two learned components:

• A global initialization, θ, that is used to initialize all the parameters of the GNNs in the inference model.

The global parameters θ are optimized via second-order gradient descent to provide an effective initialization point for any graph sampled from the distribution p(G).

• A graph signature s Gi = ψ(G i ) that is used to modulate the parameters of inference model φ i based on the history of observed training graphs.

In particular, we assume that the inference model q φi for each graph G i can be conditioned on the graph signature.

That is, we augment the inference model to q φi (Z|A, X, s Gi ), where we also include the graph signature s Gi as a conditioning input.

We use a k-layer graph convolutional network (GCN) (Kipf & Welling, 2016a) , with sum pooling to compute the signature:

where GCN denotes a k-layer GCN (as defined in (Kipf & Welling, 2016a) ), MLP denotes a densely-connected neural network, and we are summing over the node embeddings z v output from the GCN.

As with the global parameters θ, the graph signature model ψ is optimized via second-order gradient descent.

The overall Meta-Graph architecture is detailed in Figure 1 and the core learning algorithm is summarized in the algorithm block below.

Result: Global parameters θ, Graph signature function ψ Initialize learning rates: α, Sample a mini-batch of graphs, G batch from p(G);

The basic idea behind the algorithm is that we (i) sample a batch of training graphs, (ii) initialize VGAE link prediction models for these training graphs using our global parameters and signature function, (iii) run K steps of gradient descent to optimize each of these VGAE models, and (iv) use second order gradient descent to update the global parameters and signature function based on a held-out validation set of edges.

As depicted in Fig 1, this corresponds to updating the GCN based encoder for the local link prediction parameters φ j and global parameters θ along with the graph signature function ψ using second order gradients.

Note that since we are running K steps of gradient descent within the inner loop of Algorithm 1, we are also "meta" optimizing for fast adaptation, as θ and ψ are being trained via second-order gradient descent to optimize the local model performance after K gradient updates, where generally K ∈ {0, 1, . . .

, 5}.

We consider several concrete instantiations of the Meta-Graph framework, which differ in terms of how the output of the graph signature function is used to modulate the parameters of the VGAE inference models.

For all the Meta-Graph variants, we build upon the standard GCN propagation rule (Kipf & Welling, 2016a) to construct the VGAE inference models.

In particular, we assume that all the inference GNNs (Equation 1) are defined by stacking K neural message passing layers of the form:

where h v ∈ R d denotes the embedding of node v at layer k of the model, N (v) = {u ∈ V : e u,v ∈ E} denotes the nodes in the graph neighborhood of v, and W (k) ∈ R d×d is a trainable weight matrix for layer k. The key difference between Equation 5 and the standard GCN propagation rule is that we add the modulation function m s G , which is used to modulate the message passing based on the graph signature s G = ψ(G).

We describe different variations of this modulation below.

In all cases, the intuition behind this modulation is that we want to compute a structural signature from the input graphs that can be used to condition the initialization of the local link prediction models.

Intuitively, we expect this graph signature to encode structural properties of sampled graphs G i ∼ p(G) in order to modulate the parameters of the local VGAE link prediction models and adapt it to the current graph. (2019), we experiment with basic feature-wise linear modulation (Strub et al., 2018) to define the modulation function m s G :

Here, we restrict the modulation terms β k and γ k output by the signature function to be in [−1, 1] by applying a tanh non-linearity after Equation 4.

GS-Gating.

Feature-wise linear modulation of the GCN parameters (Equation 6 ) is an intuitive and simple choice that provides flexible modulation while still being relatively constrained.

However, one drawback of the basic linear modulation is that it is "always on", and there may be instances where the modulation could actually be counter-productive to learning.

To allow the model to adaptively learn when to apply modulation, we extend the feature-wise linear modulation using a sigmoid gating term, ρ k (with [0, 1] entries), that gates in the influence of γ and β:

GS-Weights.

In the final variant of Meta-Graph, we extend the gating and modulation idea by separately aggregating graph neighborhood information with and without modulation and then merging these two signals via a convex combination:

where we use the basic linear modulation (Equation 6) to define m s β k ,γ k .

Note that a simplification of Meta-Graph, where the graph signature function is removed, can be viewed as an adaptation of model agnostic meta learning (MAML) (Finn et al., 2017) to the few shot link prediction setting.

As discussed in Section 2, there are important differences in the setup for few shot link prediction, compared to traditional few shot classification.

Nonetheless, the core idea of leveraging an inner and outer loop of training in Algorithm 1-as well as using second order gradients to optimize the global parameters-can be viewed as an adaptation of MAML to the graph setting, and we provide comparisons to this simplified MAML approach in the experiments below.

We formalize the key differences by depicting the graphical model of MAML as first depicted in (Grant et al., 2018) and contrasting it with the graphical model for Meta-Graph, in Figure 1 .

MAML when reinterpreted for a distribution over graphs, maximizes the likelihood over all edges in the distribution.

On the other hand, Meta-Graph when recast in a hierarchical Bayesian framework adds a graph signature function that influencesφ j to produce the modulated parameters φ j from N sampled edges.

This explicit influence of ψ is captured by the term p(φ j |ψ, φ j ) in Equation 7 below:

For computational tractability we take the likelihood of the modulated parameters as a point estimate -i.e., p(φ j |ψ,φ j ) = δ(ψ ·φ j ).

We design three novel benchmarks for the few-shot link prediction task.

All of these benchmarks contain a set of graphs drawn from a common domain.

In all settings, we use 80% of these graphs for training and 10% as validation graphs, where these training and validation graphs are used to optimize the global model parameters (for Meta-Graph) or pre-train weights (for various baseline approaches).

We then provide the remaining 10% of the graphs as test graphs, and our goal is to fine-tune or train a model on these test graphs to achieve high link prediction accuracy.

Note that in this few shot link prediction setting, there are train/val/test splits at both the level of graphs and edges: for every individual graph, we are optimizing a model using the training edges to predict the likelihood of the test edges, but we are also training on multiple graphs with the goal of facilitating fast adaptation to new graphs via the global model parameters.

Our goal is to use our benchmarks to investigate four key empirical questions:

Q1 How does the overall performance of Meta-Graph compare to various baselines, including (i) a simple adaptation of MAML (Finn et al., 2017 ) (i.e., an ablation of Meta-Graph where the graph signature function is removed), (ii), standard pre-training approaches where we pre-train the VGAE model on the training graphs before fine-tuning on the test graphs, and (iii) naive baselines that do not leverage multi-graph information (i.e., a basic VGAE without pre-training, the Adamic-Adar heuristic (Adamic & Adar, 2003) , and DeepWalk (Perozzi et al., 2014) )?

Q2 How well does Meta-Graph perform in terms of fast adaption?

Is Meta-Graph able to achieve strong performance after only a small number of gradient steps on the test graphs?

Q3 How necessary is the graph signature function for strong performance, and how do the different variants of the Meta-Graph signature function compare across the various benchmark settings?

Q4 What is learned by the graph signature function?

For example, do the learned graph signatures correlate with the structural properties of the input graphs, or are they more sensitive to node feature information?

Datasets.

Two of our benchmarks are derived from standard multi-graph datasets from proteinprotein interaction (PPI) networks (Zitnik & Leskovec, 2017) and 3D point cloud data (FirstMM-DB) (Neumann et al., 2013) .

These benchmarks are traditionally used for node and graph classification, respectively, but we adapt them for link prediction.

We also create a novel multi-graph dataset based upon the AMINER citation data (Tang et al., 2008) , where each node corresponds to a paper and links represent citations.

We construct individual graphs from AMINER data by sampling ego networks around nodes and create node features using embeddings of the paper abstracts (see Appendix for details).

We preprocess all graphs in each domain such that each graph contains a minimum of 100 nodes and up to a maximum of 20000 nodes.

For all datasets, we perform link prediction by training on a small subset (i.e., a percentage) of the edges and then attempting to predict the unseen edges (with 20% of the held-out edges used for validation).

Key dataset statistics are summarized in Table 1 .

Baseline details.

Several baselines correspond to modifications or ablations of Meta-Graph, including the straightforward adaptation of MAML (which we term MAML in the results), a finetune baseline where we pre-train a VGAE on the training graphs observed in a sequential order and finetune on the test graphs (termed Finetune).

We also consider a VGAE trained individually on each test graph (termed No Finetune).

For Meta-Graph and all of these baselines we employ Bayesian optimization with Thompson sampling (Kandasamy et al., 2018) to perform hyperparameter selection using the validation sets.

We use the recommended default hyperparameters for DeepWalk and Adamic-Adar baseline is hyperparameter-free.

Q1: Overall Performance.

Table 2 shows the link prediction AUC for Meta-Graph and the baseline models when trained to convergence using 10%, 20% or 30% of the graph edges.

In this setting, we adapt the link prediction models on the test graphs until learning converges, as determined by performance on the validation set of edges, and we report the average link prediction AUC over the test edges of the test graphs.

Overall, we find that Meta-Graph achieves the highest average AUC in all but one setting, with an average relative improvement of 4.8% in AUC compared to the MAML approach and an improvement of 5.3% compared to the Finetune baseline.

Notably, MetaGraph is able to maintain especially strong performance when using only 10% of the graph edges for training, highlighting how our framework can learn from very sparse samples of edges.

Interestingly, in the Ego-AMINER dataset, unlike PPI and FIRSTMM DB, we observe the relative difference in performance between Meta-Graph and MAML to increase with density of the training set.

We hypothesize that this is due to fickle nature of optimization with higher order gradients in MAML (Antoniou et al., 2018) which is somewhat alleviated in GS-gating due to the gating mechanism.

With respect to computational complexity we observe a slight overhead when comparing MetaGraph to MAML which can be reconciled by realizing that the graph signature function is not updated in the inner loop update but only in outer loop.

In the Appendix, we provide additional results when using larger sets of training edges, and, as expected, we find that the relative gains of Meta-Graph decrease as more and more training edges are available.

Q2: Fast Adaptation.

Table 3 setting we only compare to the MAML, Finetune, and No Finetune baselines, as fast adaption in this setting is not well defined for the DeepWalk and Adamic-Adar baselines.

In terms of fast adaptation, we again find that Meta-Graph is able to outperform all the baselines in all but one setting, with an average relative improvement of 9.4% compared to MAML and 8.0% compared to the Finetune baseline-highlighting that Meta-Graph can not only learn from sparse samples of edges but is also able to quickly learn on new data using only a small number of gradient steps.

Also, we observe poor performance for MAML in the Ego-AMINER dataset dataset which we hypothesize is due to extremely low learning rates -i.e.

1e − 7 needed for any learning, the addition of a graph signature alleviates this problem.

Figure 2 shows the learning curves for the various models on the PPI and FirstMM DB datasets, where we can see that Meta-Graph learns very quickly but can also begin to overfit after only a small number of gradient updates, making early stopping essential.

Q3: Choice of Meta-Graph Architecture.

We study the impact of the graph signature function and its variants GS-Gating and GS-Weights by performing an ablation study using the FirstMM DB dataset.

Figure 3 shows the performance of the different model variants and baselines considered as the training progresses.

In addition to models that utilize different signature functions we report a random baseline where parameters are initialized but never updated allowing us to assess the inherent power of the VGAE model for few-shot link prediction.

To better understand the utility of using a GCN based inference network we also report a VGAE model that uses a simple MLP on the node features and is trained analogously to Meta-Graph as a baseline.

As shown in Figure 3 many versions of the signature function start at a better initialization point or quickly achieve higher AUC scores in comparison to MAML and the other baselines, but simple modulation and GS-Gating are superior to GS-Weights after a few gradient steps.

Q4: What is learned by the graph signature?

To gain further insight into what knowledge is transferable among graphs we use the FirstMM DB and Ego-AMINER datasets to probe and compare the output of the signature function with various graph heuristics.

In particular, we treat the output of s G = ψ(G) as a vector and compute the cosine similarity between all pairs of graph in the training set (i.e., we compute the pairwise cosine similarites between graph signatures, s G ).

We similarly compute three pairwise graph statistics-namely, the cosine similarity between average node features in the graphs, the difference in number of nodes, and the difference in number of edges-and we compute the Pearson correlation between the pairwise graph signature similarities and these other pairwise statistics.

As shown in Table 4 we find strong positive correlation in terms of Pearson correlation coefficient between node features and the output of the signature function for both datasets, indicating that the graph signature function is highly sensitive to feature information.

This observation is not entirely surprising given that we use such sparse samples of edges-meaning that many structural graph properties are likely lost and making the meta-learning heavily reliant on node feature information.

We also observe moderate negative correlation with respect to the average Table 4 : Pearson scores between graph signature output and other graph statistics.

difference in nodes and edges between pairs of graphs for FirstMM DB dataset.

For Ego-AMINER we observe small positive correlation for difference in nodes and edges.

We now briefly highlight related work on link prediction, meta-learning, few-shot classification, and few-shot learning in knowledge graphs.

Link prediction considers the problem of predicting missing edges between two nodes in a graph that are likely to have an edge. (Liben-Nowell & Kleinberg, 2003) .

Common successful applications of link prediction include friend and content recommendations (Aiello et al., 2012) , shopping and movie recommendation (Huang et al., 2005) , knowledge graph completion (Nickel et al., 2015) and even important social causes such as identifying criminals based on past activities (Hasan et al., 2006) .

Historically, link prediction methods have utilized topological graph features such as common neighbors yielding strong baselines like Adamic/Adar measure (Adamic & Adar, 2003) , Jaccard Index among others.

Other approaches include Matrix Factorization (Menon & Elkan, 2011) and more recently deep learning and graph neural networks based approaches (Grover & Leskovec, 2016; Wang et al., 2015; Zhang & Chen, 2018 ) have risen to prominence.

A commonality among all the above approaches is that the link prediction problem is define over a single dense graph where the objective is to predict unknown/future links within the same graph.

Unlike these previous approaches, our approach considers link prediction tasks over multiple sparse graphs which are drawn from distribution over graphs akin to real world scenario such as protein-protein interaction graphs, 3D point cloud data and citation graphs in different communities.

In meta-learning or learning to learn (Bengio et al., 1990; 1992; Thrun & Pratt, 2012; Schmidhuber, 1987) , the objective is to learn from prior experiences to form inductive biases for fast adaptation to unseen tasks.

Meta-learning has been particularly effective in few-shot learning tasks with a few notable approaches broadly classified into metric based approaches (Vinyals et al., 2016; Snell et al., 2017; Koch et al., 2015) , augmented memory (Santoro et al., 2016; Kaiser et al., 2017; Mishra et al., 2017) and optimization based approaches (Finn et al., 2017; Lee & Choi, 2018) .

Recently, there are several works that lie at the intersection of meta-learning for few-shot classification and graph based learning.

In Latent Embedding Optimization, Rusu et al. (2018) learn a graph between tasks in embedding space while Liu et al. (2019) introduce a message propagation rule between prototypes of classes.

However, both these methods are restricted to the image domain and do not consider meta-learning over a distribution of graphs as done here.

Another related line of work considers the task of few-shot relation prediction in knowledge graphs.

Xiong et al. (2018) developed the first method for this task, which leverages a learned matching met-ric using both a learned embedding and one-hop graph structures.

More recently Chen et al. (2019) introduce Meta Relational Learning framework (MetaR) that seeks to transfer relation-specific meta information to new relation types in the knowledge graph.

A key distinction between few-shot relation setting and the one which we consider in this work is that we assume a distribution over graphs while in the knowledge graph setting there is only a single graph and the challenge is generalizing to new types of relations within this graph.

We introduce the problem of few-shot link prediction-where the goal is to learn from multiple graph datasets to perform link prediction using small samples of graph data-and we develop the Meta-Graph framework to address this task.

Our framework adapts gradient-based meta learning to optimize a shared parameter initialization for local link prediction models, while also learning a parametric encoding, or signature, of each graph, which can be used to modulate this parameter initialization in a graph-specific way.

Empirically, we observed substantial gains using Meta-Graph compared to strong baselines on three distinct few-shot link prediction benchmarks.

In terms of limitations and directions for future work, one key limitation is that our graph signature function is limited to modulating the local link prediction model through an encoding of the current graph, which does not explicitly capture the pairwise similarity between graphs in the dataset.

Extending Meta-Graph by learning a similarity metric or kernel between graphs-which could then be used to condition meta-learning-is a natural direction for future work.

Another interesting direction for future work is extending the Meta-Graph approach to multi-relational data, and exploiting similarities between relation types through a suitable Graph Signature function.

To construct the Ego-Aminer dataset we first create citation graphs from different fields of study.

We then select the top 100 graphs in terms number of nodes for further pre-processing.

Specifically, we take the 5-core of each graph ensuring that each node has a minimum of 5-edges.

We then construct ego networks by randomly sampling a node from the 5-core graph and taking its two hop neighborhood.

Finally, we remove graphs with fewer than 100 nodes and greater than 20000 nodes which leads to a total of 72 graphs as reported in Table 1 .

We list out complete results when using larger sets of training edges for PPI, FIRSTMM DB and Ego-Aminer datasets.

We show the results for two metrics i.e. Average AUC across all test graphs.

As expected, we find that the relative gains of Meta-Graph decrease as more and more training edges are available.

Table 6 : 5-gradient update AUC results for PPI for training edge splits

<|TLDR|>

@highlight

We apply gradient based meta-learning to the graph domain and introduce a new graph specific transfer function to further bootstrap the process.
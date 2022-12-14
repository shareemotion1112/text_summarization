Community detection in graphs is of central importance in graph mining, machine learning and network science.

Detecting overlapping communities is especially challenging, and remains an open problem.

Motivated by the success of graph-based  deep  learning  in  other  graph-related  tasks,  we  study  the  applicability  of this framework for overlapping community detection.

We propose a probabilistic model for overlapping community detection based on the graph neural network architecture.

Despite its simplicity, our model outperforms the existing approaches in the community recovery task by a large margin.

Moreover, due to the inductive formulation, the proposed model is able to perform out-of-sample community detection for nodes that were not present at training time

Graphs provide a natural way of representing complex real-world systems.

For understanding the structure and behavior of these systems, community detection methods are an essential tool.

Detecting communities allows us to analyze social networks BID9 , to detect fraud BID24 , to discover functional units of the brain BID8 , and to predict functions of proteins BID26 .

Over the last decades, this problem has attracted significant attention of the research community and numerous models and algorithms have been proposed BID32 .

In particular, it is a well known fact that communities in real graphs are in fact overlapping BID34 , thus, requiring the development of advanced models to capture this complex structure.

In this regard, the advent of deep learning methods for graph-structured data opens new possibilities for designing more accurate and more scalable algorithms.

Indeed, deep learning on graphs has already shown state-of-the-art results in s for various graph-related tasks such as semi-supervised node classification and link prediction BID2 .

Likewise, a few deep learning methods for community detection in graphs have been proposed BID36 BID4 .

However, they all have one drawback in common: they only focus on the special case of disjoint (non-overlapping) communities.

Handling overlapping communities, is a requirement not yet met by existing deep learning approaches to community detection.

In this paper we propose an end-to-end deep probabilistic model for overlapping community detection in graphs.

Our core idea lies in predicting the community affiliations using a graph neural network.

Despite its simplicity, our model achieves state-of-the art results in community recovery and significantly outperforms the existing approaches.

Moreover, our model is able to perform out-of-sample (inductive) community detection for nodes that were not seen at training time.

To summarize, our main contributions are:??? We propose the Deep Overlapping Community detection (DOC) model -a simple, yet effective deep learning model for overlapping community detection in graphs.

DOC is one of few methods is able to perform community detection both transductively and inductively.??? We introduce 5 new datasets for overlapping community detection, which can act as a benchmark to stimulate future work in this research area.??? We perform a thorough experimental evaluation of our model, and show its superior performance when comparing with established methods for overlapping community detection.

Assume that we are given an undirected unweighted graph G = (V, E), with N := |V| nodes and M := |E| edges, represented by a symmetric adjacency matrix A ??? {0, 1} N ??N .

Moreover, each node is associated with D real-valued attributes, that can be represented as a matrix X ??? R N ??D .

The goal of overlapping community detection is to assign nodes in the graph into C communities.

Such assignment can be represented as a non-negative community affiliation matrix F ??? R N ??C ???0 , where F uc denotes the strength of node u's membership in community c (with the notable special case of binary hard-assignment F ??? {0, 1} N ??C ).

There is no single universally definition of community in the literature.

However, most recent works tend to agree with the statement that a community is a group of nodes that have a higher probability to form edges with each other than with other nodes in the graph BID6 .One can broadly subdivide the existing methods for overlapping community detection into three categories: approaches based on non-negative matrix factorization (NMF), probabilistic inference, or heuristics.

Methods based on NMF are trying to recover the community affiliation matrix F by performing a low-rank decomposition of the adjacency matrix A or some other related matrix BID31 BID15 .

Probabilistic approaches, such as or BID37 , treat F as a latent variable in a generative model for the graph, p(A, F ).

This way the problem of community detection is cast as an instance of probabilistic inference.

Lastly, heuristic-based approaches usually define a goodness measure, like within-community edge density BID7 , and then directly optimize it.

All of these approaches can be very generally formulated as an optimization problem min DISPLAYFORM0 for an appropriate choice of the loss function L, be it Frobenius norm L( DISPLAYFORM1 Besides these traditional approaches, one can also view the problem of community detection through the lens of representation learning.

The community affiliation matrix F can be considered as an embedding of nodes into R C ???0 , with the aim of preserving the community structure.

Given the recent success of deep representation learning for graphs BID2 , a question arises: "Can the advances in deep representation learning for graphs be used to design better community detection algorithms?".A very simple idea is to first apply a node embedding approach to the graph, and then cluster the nodes in the embedding space using k-means to obtain communities (as done in, e.g., BID29 ).

However, such approach is only able to detect disjoint communities, which does not correspond to the structure of communities in real-world graphs BID34 .

Instead, we argue that an end-to-end deep learning architecture able to detect overlapping communities is preferable.

Traditional community detection methods treat F as a free variable, with respect to which optimization is performed FIG0 ).

This is similar to how embeddings are learned in methods like DeepWalk BID23 and node2vec (Grover & Leskovec, 2016) .

In contrast, recent works of BID13 Hamilton et al. (2017) ; BID1 have adopted the approach of defining the embeddings as a function of node attributes F := f ?? (X, A) and solving the optimization problem DISPLAYFORM2 where f ?? is defined by a neural network.

1 Such formulation allows to??? achieve better performance in downstream tasks like link prediction and node classification;??? naturally incorporate the attributes X without hand-crafting the generative model p(X, F );??? generate embeddings inductively for previously unseen nodes.

We propose to use this framework for overlapping community detection, and describe our model in the next section.

We let the community affiliations be produced by a three-layer graph convolutional neural network (GCN), as defined in BID14 .

DISPLAYFORM0 where?? =D DISPLAYFORM1 2 is the normalized adjacency matrix,?? = A + I, andD the corresponding degree matrix.

A ReLU nonlinearity is applied element-wise to the output layer to ensure nonnegativity of the community affiliation matrix F .

Any other graph neural network architecture can be used here -we choose GCN because of its simplicity and popularity.

Link function.

A good F explains well the community structure of the graph.

To model this formally, we adopt a probabilistic approach to community detection where we need to define the likelihood p(A|F ).

A standard assumption in probabilistic community detection is that the edges A uv are conditionally independent given the community memberships F .

Thus, once the F matrix is given, every pair of nodes (u, v) produces an interaction based on their community affiliations DISPLAYFORM2 .

For a probabilistic interpretation, this interaction is transformed into an edge probability by means of a link function g : R ???0 ??? [0, 1].

The edge probability is then given by DISPLAYFORM3 We consider two choices for the link function g: Bernoulli-Poisson link and sigmoid link.

Bernoulli-Poisson link, defined as ??(X uv ) = 1 ??? exp(???X uv ), is a common probabilistic model for overlapping community detection BID37 BID28 .

Note, that under the BP model a pair of nodes that have no communities in common (i.e. F u F T v = 0) have a zero probability of forming an edge.

This is an unrealistic assumption, which can be easily fixed by adding a small offset ?? > 0, that is ??(X uv ) = 1 ??? exp(???X uv ??? ??).

DISPLAYFORM4 ???1 , is the standard choice for binary classification problems.

It can also be used to convert the edge scores into probabilities in probabilistic models for graphs BID16 BID27 BID13 .

Since a non-negative F implies that the interactions between every pair of nodes X uv are at least 0, the edge probability under the sigmoid model is always above ??(0) = 0.5.

This can be fixed by introducing an offset: g(X uv ) = ??(X uv ??? b).

The offset b becomes an additional variable to optimize over, closedform expression for which is provided in BID25 .

However, while optimizing over b produces better likelihood scores, we have empirically observed that fixing it to zero leads to the same performance in community recovery (Section 4.3).

Thus, we set b = 0 in our experiments.

We consider both link functions, and denote the two variants of our model as DOC-BP and DOCSigmoid for Bernoulli-Poisson and sigmoid link functions respectively.

Maximizing the likelihood is equivalent to minimizing the negative log-likelihood, which corresponds to the well-known binary cross-entropy loss function.

Since real-world graphs are extremely sparse (only 10 ???2 ??? 10 ???5 of possible edges are present), we are dealing with an extremely imbalanced binary classification problem.

A standard way of dealing with this problem is by balancing the contribution from both classes, which corresponds to the following objective function (5) where P E and P N stand for uniform distributions over edges and non-edges respectively.

DISPLAYFORM0 Evaluating the gradient of the full loss requires O(N 2 ) operations (since we need to compute the expectation over N 2 possible edges/non-edges).

This is impractical even for moderately-sized graphs.

Instead, we optimize the objective using stochastic gradient descent.

That is, at every iteration we approximate ???L using S randomly sampled edges, as well as the same number of non-edges.

To summarize, we use stochastic gradient descent to optimize the objective DISPLAYFORM1 where the parameters ?? are the weights of the neural network, Datasets.

We perform all our experiments using the following real-world graph datasets.

Facebook BID18 ) is a collection of small (100-1000 nodes) ego-networks from the Facebook graph.

In our experiments we consider the 5 largest of these ego-networks (Facebook-0, Facebook-107, Facebook-1684 , Facebook-1912 .

DISPLAYFORM2 Larger graph datasets (1000+ nodes) with reliable ground-truth overlapping community information, and node attributes are not openly available, which hampers the evaluation of methods for overlapping community detection in attributed graphs.

For this reason we have collected and preprocessed 5 real-world datasets, that satisfy these criteria and can act as future benchmarks (we will provide the datasets for download after the blind-reviewing phase).

Coauthor-CS and CoauthorPhysics are subsets of the Microsoft Academic co-authorship graph, constructed based on the data from the KDD Cup 2016 2 .

Communities correspond to research areas in computer science and physics respectively.

Reddit-Technology and Reddit-Gaming represent user-user graphs from the content-sharing platform Reddit 3 .

Communities correspond to subreddits -topic-specific communities that users participate in.

Amazon is a segment of the Amazon co-purchase graph BID19 , where product categories represent the communities.

Details about how the datasets were constructed and exploratory analysis are provided in Appendix B.Model architecture.

We denote the model variant with the Bernoulli-Poisson link as DOC-BP, and the model variant with the sigmoid link as DOC-Sigmoid.

For all experiments we use a 3-layer GCN (Equation 3) as the basis for both models.

We use the same model configuration for all other experiments, unless otherwise specified.

More details about the model and the training procedure are provided in Appendix A. All reported results are averaged over 10 random initializations, unless otherwise specified.

As mentioned in Section 3, evaluation of the full loss (Equation 5 ) and its gradients is computationally prohibitive due to its O(N 2 ) scaling.

Instead, we propose to use a stochastic approximation, that only depends on the fixed batch size S. We perform the following experiment to ensure that our training procedure converges to the same result, as when using the full objective.

Experimental setup.

We train the the two variants of the model on the Facebook-1912 dataset, since it is small enough (N = 755) for full-batch training to be feasible.

We compare the full-batch training procedure with stochastic training for different choices of the batch size S. Starting with the same initialization, we measure the respective full losses (Equation 5) over the iterations.

Results.

FIG0 shows training curves for batch sizes S ??? {500, 1000, 2500, 4000, 5000}, as well as for full-batch training.

As we see, the stochastic training procedure is stable.

For all batch sizes the loss converges very closely to the value achieved by full-batch training.

The standard way of comparing overlapping community detection algorithms is by assessing how well they can recover communities in graphs, for which the ground truth community affiliations are known.

It may happen that the information used as "ground truth communities" does not correlate with the graph structure BID22 .

For the datasets considered in this paper, however, ground truth communities make sense both intuitively and quantitatively (see Appendix B for a more detailed discussion).

Therefore, good performance in this experiment is a good indicator of the utility of an algorithm.

Predicting community membership.

In order to compare the detected communities to the ground truth, we first need to convert continuous community affiliations F into binary community assignments.

We assign node u to community c if F uc is above a threshold ??.

We set ?? = 0.4 for DOC-BP and ?? = 0.2 for DOC-Sigmoid, as these are the values that achieve the best performance on the Facebook-1912 dataset.

Metrics.

We use overlapping normalized mutual information (NMI), as defined by BID20 , in order to quantify the agreement of the detected communities with the ground-truth data.

Baselines.

We compare our method against a number of established methods for overlapping community detection.

BigCLAM ) is a probabilistic model based on the Bernoulli-Poisson link that only considers the graph structure.

CESNA is an extension of BigCLAM, that additionally models the generative process for node attributes.

SNMF BID15 and CDE BID17 are non-negative matrix factorization approaches for overlapping community detection.

We also compared against the LDense algorithm from BID7 -a heuristic-based approach, that finds communities with maximum edge density and similar attributes.

However, since it achieved less than 1% NMI for 8 out of 10 datasets, we don't include the results for LDense into the table.

To ensure a fair comparison, all methods were given the true number of communities C. Other hyperparameters were set to their recommended values.

Detailed configurations of the baselines are provided in Appendix C.Results.

TAB1 shows how well different methods score in recovery of ground-truth communities.

DOC-BP achieves the best or the second best score for 9 out of 10 datasets.

DOC-Sigmoid achieves the best or the second best score 10 out of 10 times.

This demonstrates the potential of deep learning methods for overlapping community detection.

CESNA could not be run for the Amazon dataset, because it cannot handle continuous attributes.

In contrast, both DOC model variants can be used with any kind of attributes out of the box.

CDE was not able to process any of the graphs with N ??? 7K nodes within 24 hours.

On the other hand, both DOC-BP and DOC-Sigmoid converged in 30s-6min for all datasets except Amazon, where it took up to 20 minutes because of the dense attribute matrix.

As we just saw, the DOC-BP and DOC-Sigmoid models, both based on the GCN architecture, are able to achieve superior performance in community detection.

Intuitively, it makes sense to use a graph neural network (GNN) in our setting, since it allows to incorporate the attribute information and also produces similar community vectors, F u , for adjacent nodes.

Nevertheless, we should ask whether it's possible achieve comparable results with a simpler model.

To answer this question, we consider the following two baselines.

), we use a simple fully-connected neural network to generate F .

DISPLAYFORM0 This is indeed related to the model proposed by BID12 .

For this baseline, we use the same configuration (number and sizes of layers, training procedure, etc.) as for the GCN-based model.

Same as for GCN FORMULA9 ), we optimize the parameters of the MLP, DISPLAYFORM1 Free variable (FV): As an even more simple baseline, we consider treating the community affiliations F as a free variable in optimization.

DISPLAYFORM2 This is similar to standard community detection methods like BigCLAM.

Since this optimization problem is rather different from those of GCN FORMULA9 ) and MLP (Equation 8), we perform additional hyperparameter optimization for the FV model.

We consider different choices for the learning rate and two initialization strategies, while keeping other aspects of the training procedure as before (stochastic training, early stopping).

We pick the configuration that achieved the best average NMI score across all datasets.

Note that this gives a strong advantage to the FV model, since for GCN and MLP models the hyperparameters were fixed without the knowledge of the ground-truth communities.

Experimental setup.

We compare the NMI scores obtained by all three models, both for BernoulliPoisson and sigmoid link functions.

Results.

As shown in TAB2 , GNN-based models outperforms the simpler baselines in 16 out of 20 cases (Remember, that the free variable version even had the advantage of picking the hyperparmeters that lead to the highest NMI scores).

This highlights the fact that attribute information only is not enough for community detection, and incorporating the graph structure clearly helps to make better inferences.

So far, we have observed that the DOC model is able to recover communities with high precision.

What's even more interesting, since our model learns the mapping from node attributes to the producing community affiliations (Equation 3), it should also be possible to predict communities inductively for nodes that were not present at training time.

Experimental setup.

We hide a randomly selected fraction of nodes from each community, and train the DOC-BP and DOC-Sigmoid models on the remaining graph.

Once the parameters ?? are learned, we perform a forward pass of each model using the full adjacency and attribute matrix.

We then compute how well the communities were predicted for the nodes that were not present during training, using NMI as a metric.

We compare with the MLP model (Equation 7) as a baseline.

Results.

As can be seen in FIG1 , both DOC-BP and DOC-Sigmoid are able to infer communities inductively for previously unseen nodes with high accuracy (NMI ??? 40%), which is on the same level as for the transductive setting TAB2 .

On the other hand, MLP-BP and MLP-Sigmoid models both perform worse than the GCN-based ones, and significantly below their own scores for transductive community detection.

This highlights the fact that graph-based neural network architectures provide a significant advantage for community detection.

The problem of community detection in graphs is well-established in the research community, and methods such as stochastic block models BID0 and spectral methods BID30 have attracted a lot of attention.

Despite the popularity of these methods, they are only suitable for detecting non-overlapping communities (i.e. partitioning the network), which is not the setting usually encountered in real-world networks BID34 .

Methods for overlapping community detection have been proposed BID32 , but our understanding of their behavior is not as mature as for the non-overlapping methods.

As discussed in Section 2, methods for OCD can be broadly divided into methods based on nonnegative matrix factorization, probabilistic inference and heuristics.

These categories are not mutually exclusive, and often one method can be viewed as belonging to multiple categories.

For example, the factorization-based approaches that minimize the Frobenius norm A ??? F F DISPLAYFORM0 More generally, most NMF and probabilistic inference models are performing a non-linear low rank decomposition of the adjacency matrix, which can be connected to the generalized principle component analysis model BID5 .Deep learning for graph data can be broadly subdivided into two main categories: graph neural networks and node embeddings.

Graph neural networks BID14 Hamilton et al., 2017) are specialized neural network architectures that can operate on graph-structured data.

The goal of embedding approaches BID23 Grover & Leskovec, 2016; BID1 is to learn vector representations of nodes in a graph, that can later be used for other downstream machine learning tasks.

One can perform k-means clustering on the node embeddings (as done in, e.g., BID29 ) to cluster nodes into communities.

However, such approach is not able to capture the overlapping community structure present in real-world graphs.

Several works have devised deep learning methods for community detection in graphs.

BID36 and BID3 propose deep learning approaches that seek a low-rank decomposition of the modularity matrix BID21 .

This means both of these approaches are limited to finding disjoint communities, as opposed to our algorithm.

Also related to our model is the approach by BID12 , where they use a deep belief network to generate the community affiliation matrix.

However, their neural network architecture does not use the graph, which we have shown to be crucial in Section 4.4.

Lastly, BID4 designed a neural network architecture for supervised community detection.

Their model learns to detect communities by training on a labeled set with community information given.

This is very different from this paper, where we learn to detect communities in a fully unsupervised manner.

In this work we have proposed and studied two deep models for overlapping community detection: DOC-BP, based on the Bernoulli-Poisson link, and DOC-Sigmoid, that relies on the sigmoid link function.

The two variants of our model achieve state-of-the-art results and convincingly outperfom existing techniques in transductive and inductive community detection tasks.

Using stochastic training, both approaches are highly efficient and scale to large graphs.

Among the two proposed models, DOC-BP one average performs better than the DOC-Sigmoid variant.

We leave to future work to investigate the properties of communities detected by these two methods.

To summarize, the results obtained in our experimental evaluation provide strong evidence that deep learning for graphs deserves more attention as a framework for overlapping community detection.

Architecture.

We use a 3-layer graph convolutional neural network (Equation 3), with hidden sizes of 64, and the final (third) layer has size C (number of communities to be detected).

Dropout with 50% keep probability is applied at every layer.

We don't use any other forms of regularization, such as weight decay.

Training.

We train the model using Adam optimizer with default parameters.

The learning rate is set to 10 ???4 .

We use the following early stopping strategy: Before training, we set aside 1% of edges and the same number of non-edges.

Every 25 gradient steps we compute the loss (Equation 5) for the validation edges and non-edges.

We stop training if there was no improvement to the best validation loss for 20 ?? 25 = 500 iterations, or after 5000 epochs, whichever happens first.

Raw Amazon data is provided by BID19 at http://jmcauley.ucsd.edu/ data/amazon/links.html.??? Nodes: A node in the graph represents a product sold at Amazon.

To get a graph of manageable size, we restrict our attention to the products in 14 randomly-chosen subcategories of the "Video Games" supercategory.??? Edges: A pair of products (u, v) is connected by an edge if u is "also bought" with product v, or the other way around.

The "also bought" information is provided in the raw data.??? Communities: We treat the subcategories (e.g. "Xbox 360", "PlayStation 4") as community labels.

Every product can belong to multiple categories.??? Features: The authors of BID11 extracted visual features from product pictures using a deep CNN.

We use these visual features as attributes X in our experiments.

We use the dump of the Microsoft Academic Graph that was published for the KDD CUP 2016 competition (https://kddcup2016.azurewebsites.net/) to construct two co-authorship graphs -Coauthor-CS and Coauthor-Physics.??? Nodes: A node in the graph represents a researcher.??? Edges: A pair of researchers (u, v) is connected by an edge if u and v have co-authored one or more papers.??? Communities: For Computer Science (Coauthor-CS), we use venues as proxies for fields of study.

We pick top-5 venues for each subfield of CS according to Google Scholar (scholar.google.com).

An author u is assigned to field of study c if he published at least 5 papers in venues associated with this field of study.

For Physics (Coauthor-Physics), we use the Physcial Review A, B, C, D, E journals as indicators of fields of study (= communities).

An author u is assigned to field of study c if he published at least 5 papers in the respective Physical Review "?" journal.??? Features:

For author user u we construct a histogram over keywords that were assigned to their papers.

That is, the entry of the attribute matrix X ud = # of papers that author u has published that have keyword d.

For this graph we had to remove from our consideration the papers that had too many (??? 40) authors, since it led to very large fully-connected components in the resulting graph.

Reddit is an online content-sharing platform, where users share, rate, and comment on content on a wide range of topics.

The site consists of a number of smaller topic-oriented communi-ties, called subreddits.

We downloaded a dump of Reddit comments for February 2018 from http://files.pushshift.io/reddit/comments/. Using the list provided at https: //www.reddit.com/r/ListOfSubreddits/, we picked 48 gaming-related subreddits and 31 technology-based subreddits.

For each of these groups of subreddits we constructed a graph as following:??? Nodes: A node in the graph represents a user of Reddit, identified by their author id .???

Edges:

A pair of users (u, v) is connected by an edge if u and v have both commented on the same 3 or more posts.??? Communities: We treat subreddits as communities.

A user is assigned to a community c, if he commented on at least 5 posts posted in that community.??? Features: For every user u we construct a histogram of other subreddits (excluding the subreddits used as communities) that they commented in.

That is, the entry of the attribute matrix X ud = # of comments user u left on subreddit d.

Co-purchase graphs, co-authorship graphs and content-sharing platforms are classic examples of networks with overlapping community structure BID34 , so using these communities as ground truth is justifiable.

Additionally, we show that for all the five graphs considered, the probability of connection between a pair of nodes grows monotonically with the number of shared communities.

This further shows that our choice of communities makes sense.

??? We used the reference C++ implementations of BigCLAM and CESNA, that were provided by the authors (https://github.com/snap-stanford/snap).

Models were used with the default parameter settings for step size, backtracking line search constants, and balancing terms.

Since CESNA can only handle binary attributes, we binarize the original attributes (set the nonzero entries to 1) if they have a different type.??? We implemented SNMF ourselves using Python.

The F matrix is initialized by sampling from the Uniform[0, 1] distribution.

We run optimization until the improvement in the reconstruction loss goes below 10 ???4 per iteration, or for 300 epochs, whichever happens first.??? We use the Matlab implementation of CDE provided by the authors.

We set the hyperparameters to ?? = 1, ?? = 2, ?? = 5, as recommended in the paper, and run optimization for 20 iterations.??? We use the Python implementation of LDense provided by the authors (https:// research.cs.aalto.fi/dmg/software.shtml), and run the algorithm with the recommended parameter settings.

Same as with CESNA, since the methods only supports binary attributes, we binarize the original data if necessary.

@highlight

Detecting overlapping communities in graphs using graph neural networks
We present network embedding algorithms that capture information about a node from the local distribution over node attributes around it, as observed over random walks following an approach similar to Skip-gram.

Observations from neighborhoods of different sizes are either pooled (AE) or encoded distinctly in a multi-scale approach (MUSAE).

Capturing attribute-neighborhood relationships over multiple scales is useful for a diverse range of applications, including latent feature identification across disconnected networks with similar attributes.

We prove theoretically that matrices of node-feature pointwise mutual information are implicitly factorized by the embeddings.

Experiments show that our algorithms are robust, computationally efficient and outperform comparable models on social, web and citation network datasets.

Figure 1: Phenomena affecting and inspiring the design of the multi-scale attributed network embedding procedure.

In Figure 1a attributed nodes D and G have the same feature set and their nearest neighbours also exhibit equivalent sets of features, whereas features at higher order neighbourhoods differ.

Figure 1b shows that as the order of neighbourhoods considered (r) increases, the product of the adjacency matrix power and the feature matrix becomes less sparse.

This suggests that an implicit decomposition method would be computationally beneficial.

Our key contributions are:

1. to introduce the first Skip-gram style embedding algorithms that consider attribute distributions over local neighborhoods, both pooled (AE) and multi-scale (MUSAE), and their counterparts that attribute distinct features to each node (AE-EGO and MUSAE-EGO);

2. to theoretically prove that their embeddings approximately factorize PMI matrices based on the product of an adjacency matrix power and node-feature matrix;

3. to show that popular network embedding methods DeepWalk (Perozzi et al., 2014) and Walklets (Perozzi et al., 2017) are special cases of our AE and MUSAE;

4.

we show empirically that AE and MUSAE embeddings enable strong performance at regression, classification, and link prediction tasks for real-world networks (e.g. Wikipedia and Facebook), are computationally scalable and enable transfer learning between networks.

We provide reference implementations of AE and MUSAE, together with the datasets used for evaluation at https://github.com/iclr2020/MUSAE.

Efficient unsupervised learning of node embeddings for large networks has seen unprecedented development in recent years.

The current paradigm focuses on learning latent space representations of nodes such that those that share neighbors (Perozzi et al., 2014; Tang et al., 2015; Grover & Leskovec, 2016; Perozzi et al., 2017) , structural roles (Ribeiro et al., 2017; Ahmed et al., 2018) or attributes are located close together in the embedding space.

Our work falls under the last of these categories as our goal is to learn similar latent representations for nodes with similar sets of features in their neighborhoods, both on a pooled and multi-scale basis.

Neighborhood preserving node embedding procedures place nodes with common first, second and higher order neighbors within close proximity in the embedding space.

Recent works in the neighborhood preserving node embedding literature were inspired by the Skip-gram model (Mikolov et al., 2013a; b) , which generates word embeddings by implicitly factorizing a shifted pointwise mutual information (PMI) matrix (Levy & Goldberg, 2014) obtained from a text corpus.

This procedure inspired DeepWalk (Perozzi et al., 2014) , a method which generates truncated random walks over a graph to obtain a "corpus" from which the Skip-gram model generates neighborhood preserving node embeddings.

In doing so, DeepWalk implicitly factorizes a PMI matrix, which can be shown, based on the underlying first-order Markov process, to correspond to the mean of a set of normalized adjacency matrix powers up to a given order (Qiu et al., 2018) .

Such pooling of matrices can be suboptimal since neighbors over increasing path lengths (or scales) are treated equally or according to fixed weightings (Mikolov et al., 2013a; Grover & Leskovec, 2016) ; whereas it has been found that an optimal weighting may be task or dataset specific (Abu-El-Haija et al., 2018) .

In contrast, multi-scale node embedding methods such as LINE (Tang et al., 2015) , GraRep (Cao et al., 2015) and Walklets (Perozzi et al., 2017) separately learn lower-dimensional node embedding components from each adjacency matrix power and concatenate them to form the full node representation.

Such un-pooled representations, comprising distinct but less information at each scale, are found to give higher performance in a number of downstream settings, without increasing the overall number of free parameters (Perozzi et al., 2017) .

Attributed node embedding procedures refine ideas from neighborhood based node embeddings to also incorporate node attributes (equivalently, features or labels) (Yang et al., 2015; Liao et al., 2018; Huang et al., 2017; Yang et al., 2018; Yang & Yang, 2018) .

Similarities between both a node's neighborhood structure and features contribute to determining pairwise proximity in the node embedding space.

These models follow quite different strategies to obtain such representations.

The most elemental procedure, TADW (Yang et al., 2015) , decomposes a convex combination of normalized adjacency matrix powers into a matrix product that includes the feature matrix.

Several other models, such as SINE (Zhang et al., 2018) and ASNE (Liao et al., 2018) , implicitly factorize a matrix formed by concatenating the feature and adjacency matrices.

Other approaches such as TENE (Yang & Yang, 2018) , formulate the attributed node embedding task as a joint non-negative matrix factorization problem in which node representations obtained from sub-tasks are used to regularize one another.

AANE (Huang et al., 2017) uses a similar network structure based regularization approach, in which a node feature similarity matrix is decomposed using the alternating direction method of multipliers.

The method most similar to our own is BANE (Yang et al., 2018) , in which the product of a normalized adjacency matrix power and a feature matrix is explicitly factorized to obtain attributed node embeddings.

Many other methods exist, but do not consider the attributes of higher order neighborhoods (Yang et al., 2015; Liao et al., 2018; Huang et al., 2017; Zhang et al., 2018; Yang & Yang, 2018) .

The relationship between our pooled (AE) and multi-scale (MUSAE) attributed node embedding methods mirrors that between graph convolutional neural networks (GCNNs) and multi-scale GCNNs.

Widely used graph convolutional layers, such as GCN (Kipf & Welling, 2017) , GraphSage (Hamilton et al., 2017) , GAT (Veli??kovi?? et al., 2018) , APPNP (Klicpera et al., 2019) , SGCONV (Wu et al., 2019) and ClusterGCN (Chiang et al., 2019) , create latent node representations that pool node attributes from arbitrary order neighborhoods, which are then inseparable and unrecoverable.

In contrast, MixHop (Abu-El-Haija et al., 2019) learns latent features for each proximity.

We now define algorithms to learn node embeddings using the attributes of nearby nodes, that allows both node and attribute embeddings to be learned jointly.

The aim is to learn similar embeddings for nodes that occur in neighbourhoods of similar attributes; and similar embeddings for attributes that often occur in similar neighbourhoods of nodes.

Let G = (V, L) be an undirected graph of interest where V and L are the sets of vertices and edges (or links) respectively; and let F be the set of all possible node features (i.e. attributes).

We define F v ??? F as the subset of features belonging to each node v ??? V. An embedding of nodes is a mapping g : V ??? R d that assigns a d-dimensional representation g(v) (or simply g v ) to each node v and is fully described by a matrix G ??? R |V|??d .

Similarly, an embedding of the features (to the same latent space) is a mapping h : F ??? R d with embeddings denoted h(f ) (or simply h f ), and is fully described by a matrix H ??? R |F|??d .

The Attributed Embedding (AE) procedure is described by Algorithm 1.

We sample n nodes w 1 , from which to start attributed random walks on G, with probability proportional to their degree (Line 2).

From each starting node, a node sequence of length l is sampled over G (Line 3), where sampling follows a first order random walk.

For a given window size t, we iterate over each of the first l ??? t nodes of the sequence termed source nodes w j (Line 4).

For each source node, we consider the following t nodes as target nodes (Line 5).

For each target node w j+r , we add the tuple (w j , f ) to the corpus D for each target feature f ??? F wj+r (Lines 6 and 7).

We also consider features of the source node f ??? F wj , adding each (w j+r , f ) tuple to D (Lines 9 and 10).

Running Skip-gram on D with b negative samples (Line 15) generates the d-dimensional node and feature embeddings.

Algorithm 2: MUSAE sampling and training procedure 3.2 MULTI-SCALE ATTRIBUTED EMBEDDING The AE method (Algorithm 1) pools feature sets of neighborhoods at different proximities.

Inspired by the performance of (unattributed) multi-scale node embeddings, we adapt the AE algorithm to give multi-scale attributed node embeddings (MUSAE).

The embedding component of a node v ??? V for a specific proximity r ??? {1, ..., t} is given by a mapping g r : V ??? R d/t (assuming t divides d).

Similarly, the embedding component of feature f ??? F at proximity r is given by a mapping

Concatenating gives a d-dimensional embedding for each node and feature.

The Multi-Scale Attributed Embedding procedure is described by Algorithm 2.

We again sample n starting nodes w 1 with a probability proportional to node degree (Line 2) and, for each, sample a node sequence of length l over G (Line 3) according to either a first or second order random walk.

For a given window size t, we iterate over the first l ??? t (source) nodes w j of the sequence (Line 4) and for each source node we iterate through the t (target) nodes w j+r that follow (Line 5).

We again consider each target node feature f ??? F wj+r , but now add tuples (w j , f ) to a sub-corpus D r ??? (Lines 6 and 7).

We add tuples (w j+r , f ) to another sub-corpus D r Levy & Goldberg (2014) showed that the loss function of Skip-gram with negative sampling (SGNS) is minimized if the embedding matrices factorize a matrix of pointwise mutual information (PMI) of word co-occurrence statistics.

Specifically, for a word dictionary V with |V| = n, SGNS (with b negative samples) outputs two embedding matrices W , C ??? R d??n such that ???w, c ??? V:

, where #(w, c), #(w), #(c) denote counts of word-context pair (w, c), w and c over a corpus D; and word embeddings w w , c c ??? R d are columns of W and C corresponding to w and c respectively.

as empirical estimates of p(w), p(c) and p(w, c) respectively shows:

i.e. an approximate low-rank factorization of a shifted PMI matrix (low rank since typically d n).

Qiu et al. (2018) extended this result to node embedding models that apply SGNS to a "corpus" generated from random walks over the graph.

In the case of DeepWalk where random walks are first-order Markov, the joint probability distributions over nodes at different stages of a random walk can be expressed in closed form.

A closed form then follows for the factorized PMI matrix.

We show that AE and MUSAE implicitly perform analogous matrix factorizations.

Notation: A ??? R n??n denotes the adjacency matrix and D ??? R n??n the diagonal degree matrix of a graph G, i.e. D w,w = deg(w) = v A w,v .

We denote the volume of G by c = v,w A v,w .

We define the binary attribute matrix F ??? {0, 1} |V|??|F| by F w,f = 1 f ???Fw , ???w ??? V, f ??? F. For ease of notation, we let P = D ???1 A and E = diag(1 DF ), where diag indicates a diagonal matrix.

Assuming G is ergodic:

, w ??? V is the stationary distribution over nodes, i.e. c ???1 D = diag(p(w)); and c ???1 A is the stationary joint distribution over consecutive nodes p(w j , w j+1 ).

F w,f can be considered a Bernoulli parameter describing the probability p(f |w) of observing a feature f at a node w and so c ???1 DF describes the stationary joint distribution p(f, w j ) over nodes and features.

Accordingly, P is the matrix of conditional distributions p(w j+1 |w j ); and E is a diagonal matrix proportional to the probability of observing each feature at the stationary distribution p(f ) (note that p(f ) need not sum to 1, whereas p(w) necessarily must).

We know that the SGNS aspect of MUSAE (Algorithm 2, Line 17) is minimized when the learned embeddings g

Our aim is to express this factorization in terms of known properties of the graph G and its features.

Lemma 1.

The empirical statistics of node-feature pairs obtained from random walks give unbiased estimates of joint probabilities of observing feature f ??? F r steps (i) after; or (ii) before node v ??? V, as given by:

Proof.

See Appendix.

Lemma 2.

Empirical statistics of node-feature pairs obtained from random walks give unbiased estimates of joint probabilities of observing feature f ??? F r steps either side of node v ??? V, given by:

Marginalizing gives unbiased estimates of stationary probability distributions of nodes and features:

Theorem 1.

MUSAE embeddings approximately factorize the node-feature PMI matrix:

Proof.

Lemma 3.

The empirical statistics of node-feature pairs learned by the AE algorithm give unbiased estimates of mean joint probabilities over different path lengths as follows:

. .

, t} and so |D s | = t ???1 |D|.

Combining with Lemma 2, the result follows.

Theorem 2.

AE embeddings approximately factorize the pooled node-feature matrix:

Proof.

The proof is analogous to the proof of Theorem 1.

Remark 1.

DeepWalk is a corner case of AE with F = I |V| .

That is, DeepWalk is equivalent to AE if each node has a single unique feature.

Thus E = diag(1 DI) = D and, by Theorem 2, DeepWalk's embeddings factorize log c t ( Qiu et al. (2018) .

Remark 2.

Walklets is a corner case of MUSAE with F = I |V| .

Thus, for r = 1, . . .

, t, the embeddings of Walklets factorise log c

Remark 3.

Appending an identity matrix I to the feature matrices F of AE and MUSAE (denoted [F ; I]) adds a unique feature to each node.

The resulting algorithms, named AE-EGO and MUSAE-EGO, learn embeddings that, respectively, approximately factorize the node-feature PMI matrices:

and log c t (

Under the assumption of a constant number of features per source node and first-order attributed random walk sampling, the corpus generation has a runtime complexity of O(n l t x/y), where x = v???V |F v | the total number of features across all nodes (including repetition) and y = |V| the number of nodes.

Using negative sampling, the optimization runtime of a single asynchronous gradient descent epoch on AE and the joint optimization runtime of MUSAE embeddings is described by O(b d n l t x/y).

If one does p truncated walks from each source node, the corpus generation complexity is O(p y l t x) and the model optimization runtime is O(b d p y l t x).

Our later runtime experiments in Section 5 will underpin optimization runtime complexity discussed above.

Corpus generation has a memory complexity of O(n l t x/y) while the same when generating p truncated walks per node has a memory complexity of O(p y l t x).

Storing the parameters of an AE embedding has a memory complexity of O(y d) and MUSAE embeddings also use O(y d) memory.

In order to evaluate the quality of created representations we test the embeddings on supervised downstream tasks such as node classification, transfer learning across networks, regression, and link prediction.

Finally, we investigate how changes in the input size affect the runtime.

For doing so we utilize social networks and web graphs that we collected from Facebook, Github, Twitch and Wikipedia.

The data sources, collection procedures and the datasets themselves are described with great detail in Appendix B. In addition we tested our methods on citation networks widely used for model evaluation (Shchur et al., 2018) .

Across all experiments we use the same hyperparameter settings of our own model, competing unsupervised methods and graph neural networks -these are respectively listed in Appendices C, E and F.

We evaluate the node classification performance in two separate scenarios.

In the first we do k-shot learning by using the attributed embedding vectors with logistic regression to predict labels on the Facebook, Github and Twitch Portugal graphs.

In the second we test the predictive performance under a fixed size train-test split to compare against various embedding methods and competitive neural network architectures.

In this experiment we take k randomly selected samples per class, and use the attributed node embeddings to train a logistic regression model with l 2 regularization and predict the labels on the remaining vertices.

We repeated the above procedure with seeded splits 100 times to obtain robust comparable results (Shchur et al., 2018) .

From these we calculated the average of micro averaged F 1 scores to compare our own methods with other unsupervised node embedding procedures.

We varied k in order to show the efficacy of the methods -what are the gains when the training set size is increased.

These results are plotted in Figure 2 for Facebook, Github and Twitch Portugal networks.

Based on these plots it is evident that MUSAE and AE embeddings have little gains in terms of micro F 1 score when additional data points are added to the training set when k is larger than 12.

This implies that our method is data efficient.

Moreover, MUSAE-EGO and AE-EGO have a slight performance advantage, which means that including the nodes in the attributed random walks helps when a small amount of labeled data is available in the downstream task.

Figure 2 : Node classification k-shot learning performance as a function of training samples per class evaluated by average micro F 1 scores calculated from a 100 seeded train-test splits.

In this series of experiments we created a 100 seeded train test splits of nodes (80% train -20% test) and calculated weighted, micro and macro averaged F 1 scores on the test set to compare our methods to various embedding and graph neural network methods.

Across procedures the same random seeds were used to obtain the train-test split this way the performances are directly comparable.

We attached these results on the Facebook, Github and Twitch Portugal graphs as Table 6 of Appendix G. In each column red denotes the best performing unsupervised embedding model and blue corresponds to the strongest supervised neural model.

We also attached additional supporting results using the same experimental setting with the unsupervised methods on the Cora, Citeseer, and Pubmed graphs as Table 5 of Appendix G.

In terms of micro F 1 score our strongest method outperforms on the Facebook and GitHub networks the best unsupervised method by 1.01% and 0.47% respectively.

On the Twitch Portugal network the relative micro F 1 advantage of ASNE over our best method is 1.02%.

Supervised node embedding methods outperform our and other unsupervised methods on every dataset for most metrics.

In terms of micro F 1 this relative advantage over our best performing model variant is the largest with 4.67% on the Facebook network, and only 0.11% on Twitch Portugal.

One can make four general observations based on our results (i) multi-scale representations can help with the classification tasks compared to pooled ones; (ii) the addition of the nodes in the ego augmented models to the feature sets does not help the performance when a large amount of labeled training data is available; (iii) based on the standard errors supervised neural models do not necessarily have a significant advantage over unsupervised methods (see the results on the Github and Twitch datasets); (iv) attributed node embedding methods that only consider first-order neighbourhoods have a poor performance.

Neighbourhood based methods such as DeepWalk (Perozzi et al., 2014) are transductive and the function used to create the embedding cannot map nodes that are not connected to the original graph to the latent space.

However, vanilla MUSAE and AE are inductive and can easily map nodes to the embedding space if the attributes across the source and target graph are shared.

This also means that supervised models trained on the embedding of a source graph are transferable.

Importantly those attributed embedding methods such as AANE or ASNE that explicitly use the graph are unable to do this transfer.

The blue reference line denotes the test performance on the target dataset in a non transfer learning scenario (standard hyperparameter settings and split ratio).

The red reference line denotes the performance of random guesses.

Using the disjoint Twitch country level social networks (inter country edges are not present) we did a transfer learning experiment.

First, we learn an embedding function given the social network from a country with the standard parameter settings.

Second, we train regularized logistic regression on the embedding to predict whether the Twitch user streams explicit content.

Third, using the embedding function we map the target graph to the embedding space.

Fourth, we use the logistic model to predict the node labels on the target graph.

We evaluate the performance by the micro F 1 score based on 10 experimental repetitions.

These averages with standard error bars are plotted for the Twitch Germany, England and Spain datasets as target graphs on Figure 3 .

We added additional results with France, Portugal and Russia being the target country in Appendix H as Table 5 .

These results support that MUSAE and AE create features that are transferable across graphs that share vertex features.

For example, based on a comparison to non transfer-learning results we find that the transfer between the German and English user graphs is effective in terms of micro F 1 score.

Transfer from English users to German ones considerably improves performance, and the other way around there is a little gain.

We also see that the upstream and downstream models that we trained on graphs with more vertices transfer well while transfer to the small ones is generally poor -most of the times worse than random guessing.

There is no clear evidence that either MUSAE or AE gives better results on this specific problem.

We created embeddings of the Wikipedia webgraphs with all of our methods and the unsupervised baselines.

Using a 80% train -20% test split we predict the log of average traffic for each page using an elastic net model.

The hyperparameters of the downstream model are available in Appendix D. In Table 7 of Appendix I we report average test R 2 and standard error of the predictive performance over 100 seeded train-test splits.

Our key observation are: (i) that MUSAE outperforms all benchmark neighbourhood preserving and attributed node embedding methods, with the strongest MUSAE variant outperforming the best baseline between 2.05% and 10.03% (test R 2 ); (ii) that MUSAE significantly outperforms AE by between 2.49% and 21.64% (test R 2 ); and (iii) the benefit of using the vertices as features (ego augmented model) can improve the performance of embeddings, but appears to be dataset specific phenomenon.

The final series of experiments dedicated to the representation quality is about link prediction.

We carried out an attenuated graph embedding trial to predict the removed edges from the graph.

First, we randomly removed 50% of edges while the connectivity of the graph was not changed.

Second, an embedding is created from the attenuated graph.

Third, we calculate features for the removed edges and the same number of randomly selected pairs of nodes (negative candidates) with binary operators to create d-dimensional edge features.

We use the binary operators applied by Grover & Leskovec (2016) .

Specifically, we calculated the average, element-wise product, element-wise l 1 norm and the element-wise l 2 norm of vectors.

Finally, we created a 100 seeded 80% train -20% test splits and used logistic regression to predict whether an edge exists.

We compared to attributed and neighbourhood based embedding methods and average AUC scores are presented in Tables 8 and 9 of Appendix J. Our results show that Walklets (Perozzi et al., 2017) the multi-scale neighbourhood based embedding method materially outperforms every other method on most of the datasets and attributed embedding methods generally do poorly in terms of AUC compared to neighbourhood based ones.

In order to show the efficacy of our algorithms we run a series of experiments on synthetic graphs where we are able to manipulate the input size.

Specifically, we look at the effect of changing the number of vertices and features per vertex.

Our detailed experimental setup was as follows.

Each point in Figure 4 is the mean runtime obtained from 100 experimental runs on Erdos-Renyi graphs.

The base graph that we manipulated had 2 11 nodes, 2 3 edges and the same number of unique features per node uniformly selected from a feature set of 2 11 .

Our experimental settings were the same as the ones described in Appendix C except for the number of epochs.

We only did a single training epoch with asynchronous gradient descent on each graph.

We tested the runtime with 1, 2 and 4 cores and included a dashed line as the linear runtime reference in each subfigure.

We observe that doubling the average number of features per vertex doubles the runtime of AE and MUSAE.

Moreover, the number of cores used during the optimization does not decrease the runtime when the number of unique features per vertex compared to the cardinality of the feature set is large.

When we look at the change in the vertex set size we also see a linear behaviour.

Doubling the input size simply results in a doubled optimization runtime.

In addition, if one interpolates linearly from these results it comes that a network with 1 million nodes, 8 edges per node, 8 unique features per node can be embedded with MUSAE on commodity hardware in less than 5 hours.

This interpolation assumes that the standard parameter settings proposed in Appendix C and 4 cores were used for optimization.

We investigated attributed node embedding and proposes efficient pooled (AE) and multi-scale (MUSAE) attributed node embedding algorithms with linear runtime.

We proved that these algorithms implicitly factorize probability matrices of features appearing in the neighbourhood of nodes.

Two widely used neighbourhood preserving node embedding methods Perozzi et al. (2014; are in fact simplified cases of our models.

On several datasets (Wikipedia, Facebook, Github, and citation networks) we found that representations learned by our methods, in particular MUSAE, outperform neighbourhood based node embedding methods (Perozzi et al. (2014) ; Grover & Leskovec (2016) Our proposed embedding models are differentiated from other methods in that they encode feature information from higher order neighborhoods.

The most similar previous model BANE (Yang et al., 2018) encodes node attributes from higher order neighbourhoods but has non-linear runtime complexity and the product of adjacency matrix power and feature matrix is decomposed explicitly.

A PROOFS Lemma 1.

The empirical statistics of node-feature pairs obtained from random walks give unbiased estimates of joint probabilities of observing feature f ??? F r steps (i) after; or (ii) before node v ??? V, as given by:

Proof.

The proof is analogous to that given for Theorem 2.1 in Qiu et al. (2018) .

We show that the computed statistics correspond to sequences of random variables with finite expectation, bounded variance and covariances that tend to zero as the separation between variables within the sequence tends to infinity.

The Weak Law of Large Numbers (S.N.Bernstein) then guarantees that the sample mean converges to the expectation of the random variable.

We first consider the special case n = 1, i.e. we have a single sequence w 1 , ..., w l generated by a random walk (see Algorithm 1).

For a particular node-feature pair (w, f ), we let Y i , i ??? {1, ..., l ??? t}, be the indicator function for the event w i = w and f ??? F i+r .

Thus, we have:

the sample average of the Y i s. We also have:

for j > i + r. This allows us to compute the covariance:

where 1 is a vector of ones.

The difference term (indicated) tends to zero as j ??? i ??? ??? since then p(w j = w|w i+r ) tends to the stationary distribution p(w) =

, regardless of w i+r .

Thus, applying the Weak Law of Large Numbers, the sample average converges in probability to the expected value, i.e.:

A similar argument applies to

In both cases, the argument readily extends to the general setting where n > 1 with suitably defined indicator functions for each of the n random walks (see Qiu et al. (2018) ).

Lemma 2.

Empirical statistics of node-feature pairs obtained from random walks give unbiased estimates of joint probabilities of observing feature f ??? F r steps either side of node v ??? V, given by:

The final step follows by symmetry of A, indicating how the Lemma can be extended to directed graphs.

Our method was evaluated on a variety of social networks and web page-page graphs that we collected from openly available API services.

In Table 1 we described the graphs with widely used statistics with respect to size, diameter, and level of clustering.

We also included the average number of features per vertex and unique feature count in the last columns.

These datasets are available with the source code of MUSAE and AE at https://github.com/iclr2020/MUSAE.

This webgraph is a page-page graph of verified Facebook sites.

Nodes represent official Facebook pages while the links are mutual likes between sites.

Node features are extracted from the site descriptions that the page owners created to summarize the purpose of the site.

This graph was collected through the Facebook Graph API in November 2017 and restricted to pages from 4 categories which are defined by Facebook.

These categories are: politicians, governmental organizations, television shows and companies.

As one can see in Table 1 it is a highly clustered graph with a large diameter.

The task related to this dataset is multi-class node classification for the 4 site categories.

The largest graph used for evaluation is a social network of GitHub developers which we collected from the public API in June 2019.

Nodes are developers who have starred at least 10 repositories and edges are mutual follower relationships between them.

The vertex features are extracted based on the location, repositories starred, employer and e-mail address.

The task related to the graph is binary node classification -one has to predict whether the GitHub user is a web or a machine learning developer.

This target feature was derived from the job title of each user.

As the descriptive statistics show in Table 1 this is the largest graph that we use for evaluation with the highest sparsity.

The datasets that we use to perform node level regression are Wikipedia page-page networks collected on three specific topics: chameleons, crocodiles and squirrels.

In these networks nodes are articles from the English Wikipedia collected in December 2018, edges are mutual links that exist between pairs of sites.

Node features describe the presence of nouns appearing in the articles.

For each node we also have the average monthly traffic between October 2017 and November 2018.

In the regression tasks used for embedding evaluation the logarithm of average traffic is the target variable.

Table 1 shows that these networks are heterogeneous in terms of size, density, and clustering.

B.4 TWITCH DATASETS These datasets used for node classification and transfer learning are Twitch user-user networks of gamers who stream in a certain language.

Nodes are the users themselves and the links are mutual friendships between them.

Vertex features are extracted based on the games played and liked, location and streaming habits.

Datasets share the same set of node features, this makes transfer learning across networks possible.

These social networks were collected in May 2018.

The supervised task related to these networks is binary node classification -one has to predict whether a streamer uses explicit language.

In MUSAE and AE models we have a set of parameters that we use for model evaluation.

Our parameter settings listed in Table 2 are quite similar to the widely used general settings of random walk sampled implicit factorization machines (Perozzi et al., 2014; Grover & Leskovec, 2016; Ribeiro et al., 2017; Perozzi et al., 2017) .

Each of our models is augmented with a Doc2Vec (Mikolov et al., 2013a; b) embedding of node features -this is done such way that the overall dimension is still 128.

The downstream tasks uses logistic and elastic net regression from Scikit-learn (Pedregosa et al., 2011) for node level classification, regression and link prediction.

For the evaluation of every embedding model we use the standard settings of the library except for the regularization and norm mixing parameters.

These are described in Table 3 .

Our purpose was a fair evaluation compared to other node embedding procedures.

Because of this each we tried to use hyperparameter settings that give similar expressive power to the competing (Perozzi et al., 2014; Grover & Leskovec, 2016; Perozzi et al., 2017) and number of dimensions.

??? DeepWalk (Perozzi et al., 2014) : We used the hyperparameter settings described in Table  2 .

While the original DeepWalk model uses hierarchical softmax to speed up calculations we used a negative sampling based implementation.

This way DeepWalk can be seen as a special case of Node2Vec (Grover & Leskovec, 2016) when the second-order random walks are equivalent to the firs-order walks.

??? LINE 2 (Tang et al., 2015) : We created 64 dimensional embeddings based on first and second order proximity and concatenated these together for the downstream tasks.

Other hyperparameters are taken from the original work.

??? Node2Vec (Grover & Leskovec, 2016) :

Except for the in-out and return parameters that control the second-order random walk behavior we used the hyperparameter settings described in Table 2 .

These behavior control parameters were tuned with grid search from the {4, 2, 1, 0.5, 0.25} set using a train-validation split of 80% ??? 20% within the training set itself.

??? Walklets (Perozzi et al., 2017) : We used the hyperparameters described in Table 2 except for window size.

We set a window size of 4 with individual embedding sizes of 32.

This way the overall number of dimensions of the representation remained the same.

??? The attributed node embedding methods AANE, ASNE, BANE, TADW, TENE all use the hyperparameters described in the respective papers except for the dimension.

We parametrized these methods such way that each of the final embeddings used in the downstream tasks is 128 dimensional.

Each model was optimized with the Adam optimizer (Kingma & Ba, 2015) with the standard moving average parameters and the model implementations are sparsity aware modifications based on PyTorch Geometric (Fey & Lenssen, 2019) .

We needed these modifications in order to accommodate the large number of vertex features -see the last column in Table 1 .

Except for the GAT model (Veli??kovi?? et al., 2018) we used ReLU intermediate activation functions (Nair & Hinton, 2010) with a softmax unit in the final layer for classification.

The hyperparameters used for the training and regularization of the neural models are listed in Table 4 .

Except for the APPNP model each baseline uses information up to 2-hop neighbourhoods.

The model specific settings when we needed to deviate from the basic settings which are listed in Table  4 were as follows:

??? Classical GCN (Kipf & Welling, 2017) : We used the standard parameter settings described in this section.

??? GraphSAGE (Hamilton et al., 2017): We utilized a graph convolutional aggregator on the sampled neighbourhoods, samples of 40 nodes per source, and standard settings.

??? GAT (Veli??kovi?? et al., 2018) : The negative slope parameter of the leaky ReLU function was 0.2, we applied a single attention head, and used the standard hyperparameter settings.

??? MixHop (Abu-El-Haija et al., 2019): We took advantage of the 0 th , 1 st and 2 nd powers of the normalized adjacency matrix with 32 dimensional convolutional filters for creating the first hidden representations.

This was fed to a feed-forward layer to classify the nodes.

??? ClusterGCN (Chiang et al., 2019): Just as Chiang et al. (2019) did, we used the METIS procedure (Karypis & Kumar, 1998) .

We clustered the graphs into disjoint clusters, and the number of clusters was the same as the number of node classes (e.g. in case of the Facebook page-page network we created 4 clusters).

For training we used the earlier described setup.

??? APPNP (Klicpera et al., 2019) :

The top level feed-forward layer had 32 hidden neurons, the teleport probability was set as 0.2 and we used 20 steps for approximate personalized pagerank calculation.

??? SGCONV (Wu et al., 2019) : We used the 2 nd power of the normalized adjacency matrix for training the classifier.

I REGRESSION RESULTS ON WIKIPEDIA PAGE-PAGE GRAPHS

<|TLDR|>

@highlight

We develop efficient multi-scale approximate attributed network embedding procedures with provable properties.
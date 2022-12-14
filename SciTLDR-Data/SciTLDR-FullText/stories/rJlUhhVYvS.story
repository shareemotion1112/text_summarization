In recent years there has been a rapid increase in classification methods on graph structured data.

Both in graph kernels and graph neural networks, one of the implicit assumptions of successful state-of-the-art models was that incorporating graph isomorphism features into the architecture leads to better empirical performance.

However, as we discover in this work, commonly used data sets for graph classification have repeating instances which cause the problem of isomorphism bias, i.e. artificially increasing the accuracy of the models by memorizing target information from the training set.

This prevents fair competition of the algorithms and raises a question of the validity of the obtained results.

We analyze 54 data sets, previously extensively used for graph-related tasks, on the existence of isomorphism bias, give a set of recommendations to machine learning practitioners to properly set up their models, and open source new data sets for the future experiments.

Recently there has been an increasing interest in the development of machine learning models that operate on graph structured data.

Such models have found applications in chemoinformatics (Ralaivola et al. (2005) ; Rupp & Schneider (2010) ; Ferré et al. (2017) ) and bioinformatics (Borgwardt et al. (2005) ; Kundu et al. (2013) ), neuroscience (Sharaev et al. (2018) ; Jie et al. (2016) ; Wang et al. (2016) ), computer vision (Stumm et al. (2016) ) and system security (Li et al. (2016) ), natural language processing (Glavaš &Šnajder (2013) ), and others (Kriege et al. (2019) ; Nikolentzos et al. (2019) ).

One of the popular tasks that encompasses these applications is graph classification problem for which many graph kernels and graph neural networks have been developed.

One of the implicit assumptions that many practitioners adhere to is that models that can distinguish isomorphic instances from non-isomorphic ones possess higher expressiveness in classification problem and hence much efforts have been devoted to incorporate efficient graph isomorphism methods into the classification models.

As the problem of computing complete graph invariant is GI-hard (Gärtner et al. (2003) ), for which no known polynomial-time algorithm exists, other heuristics have been proposed as a proxy for deciding whether two graphs are isomorphic.

Indeed, from the early days topological descriptors such Wiener index (Wiener (1947a; b) ) attempted to find a single number that uniquely identifies a graph.

Later, graph kernels that model pairwise similarities between graphs utilized theoretical developments in graph isomorphism literature.

For example, graphlet kernel (Shervashidze et al. (2009) ) is based on the Kelly conjecture (see also Kelly (1957) ), anonymous walk kernel ) derives insights from the reconstruction properties of anonymous experiments (see also Micali & Allen Zhu (2016) ), and WL kernel (Shervashidze et al. (2011a) ) is based on an efficient graph isomorphism algorithm.

For sufficiently large k, kdimensional WL algorithm includes all combinatorial properties of a graph (Cai et al. (1992a) ), so one may hope its power is enough for the data set at hand.

Since only for k = Ω(n) WL algorithm is guaranteed to distinguish all graphs (for which the running time becomes exponential; see also Fürer (2017) ), in the general case WL algorithm can be used only as a strong baseline for graph isomorphism.

In similar fashion, graph neural networks exploit graph isomorphism algorithms and have been shown to be as powerful as k-dimensional WL algorithm (see for example Maron et al. (2019) ; Xu et al. (2018) ; ).

Experimental evaluation reveals that models based on the theoretical constructions with high combinatorial power such as WL algorithm performs better than the models without them such as Vertex histogram kernel (Vishwanathan et al. (2010) ) on a commonly used data sets.

This could add additional bias to results of comparison of classification algorithms since the models could simply apply a graph isomorphism method (or an efficient approximation) to determine a target label at the inference time.

However, purely judging on the accuracy of the algorithms in such cases would imply an unfair comparison between the methods as it does not measure correctly generalization ability of the models on the new test instances.

As we discover, indeed many of the data sets used in graph classification have isomorphic instances so much that in some of them the fraction of the unique non-repeating graphs is as low as 20% of the total size.

This challenges previous experimental results and requires understanding of how influential isomorphic instances on the final performance of the models.

Our contributions are:

• We analyze the quality of 54 graph data sets which are used ubiquitously in graph classification comparison.

Our findings suggest that in the most of the data sets there are isomorphic graphs and their proportion varies from as much as 100% to 0%.

Surprisingly, we also found that there are isomorphic instances that have different target labels suggesting they are not suitable for learning a classifier at all.

• We investigate the causes of isomorphic graphs and show that node and edge labels are important to identify isomorphic graphs.

Other causes include numerical attributes of nodes and edges as well as the sizes of the data set.

• We express an upper bound for the generalization gap through the Radamacher complexity of a classifier and the number of isomorphic graphs in a data set.

This bound presents theoretical evidence on how weightning of each graph in the training influences classification accuracy.

• We evaluate a classification model's performance on isomorphic instances and show that even strong models do not achieve optimal accuracy even if the instances have been seen at the training time.

Hence we show a model-agnostic way to artificially increase performance on several widely used data sets.

• We open-source new cleaned data sets that contain only non-isomorphic instances with no noisy target labels.

We give a set of recommendations regarding applying new models that work with graph structured data.

Measuring quality of data sets.

A similar issue of duplicates instances in commonly used data sets was recently discovered in computer vision domain.

Recht et al. (2019) ; Barz & Denzler (2019) ; Birodkar et al. (2019) discover that image data sets CIFAR and ImageNet contain at least 10% of the duplicate images in the test test invalidating previous performance and questioning generalization abilities of previous successful architectures.

In particular, evaluating the models in new test sets shows a drop of accuracy by as much as 15% (Recht et al., 2019) , which is explained by models' incapability to generalize to unseen slightly "harder" instances than in the original test sets.

In graph domain, a fresh look into understanding of expressiveness of graph kernels and the quality of data sets has been considered in Kriege et al. (2019) , where an extensive comparison of existing graph kernels is done and a few insights about models' behavior are suggested.

In contrast, we conduct a broader study of isomorphism metrics, revealing all isomorphism pairs in proposed 54 data sets, and propose new cleaned data.

Additionally we also consider graph neural network performance and argue that current data sets present isomorphism bias which can artificially boost evaluation metrics in a model-agnostic way.

Explaining performance of graph models.

Graph kernels (Kriege et al. (2019) ) and graph neural networks (Wu et al. (2019) ) are two competing paradigms for designing graph representations and solving graph classification and have significantly advanced empirical results due to more efficient algorithms, incorporating graph invariance into the models, and end-to-end training.

Several papers have tried to justify performance of different families of methods by studying different statistical properties.

For example, in Ying et al. (2019) by maximizing mutual information between explanation variables and predicted label distribution, the model is trained to return a small subgraph and the graph-specific attributes that are the most influential on the decision made by a GNN, which allows inspection of single-and multi-level predictions in an agnostic manner for GNNs.

In another work (Scarselli et al. (2018) ), the VC dimension of GNNs models has been shown to grow as O(p 4 N 2 ), where p is the number of network parameters and N is the number of nodes in a graph, which is comparable to RNN models.

Furthermore, stability and generalization properties of convolutional GNNs have been shown to depend on the largest eigenvalue of the graph filter and therefore are attained for properly normalized graph convolutions such as symmetric normalized graph Laplacian (Verma & Zhang (2019) ).

Finally, expressivity of graph kernels has been studied from statistical learning theory (Oneto et al. (2017b) ) and property testing (Kriege et al. (2018b) ), showing that graph kernels can capture certain graph properties such as planarity, girth, and chromatic number (Johansson et al. (2014) ).

Our approach is complementary to all of the above as we analyze if the data sets used in experiments have any effect on the final performance.

In this work we analyze 54 graph data sets from that are commonly used in graph classification task.

Examples of popular graph data sets are presented in Table 1 and statistics of all 54 data sets can be found in Table 5 , see Section A in the appendix.

All data sets represent a collection of graphs and accompanying categorical label for each graph in the data sets.

Some data sets also include node and/or edge labels that graph classification methods can use to improve the scoring.

Most of the data sets come either from biological domain or from social network domain.

Biological data sets such as MUTAG, ENZYMES, PROTEINS are graphs that represent small or large molecules, where edges of the graphs are chemical bonds or spatial proximity between different atoms.

Graph labels in these cases encode different properties like toxicity.

In social data sets such as IMDB-BINARY, REDDIT-MULTI-5K, COLLAB the nodes represent people and edges are relationships in movies, discussion threads, or citation network respectively.

Labels in these cases denote the type of interaction like the genre of the movie/thread or a research subfield.

For completeness we also include synthetic data sets SYNTHETIC ) that have continuous attributes and computer vision data sets MSRC ), where images are encoded as graphs.

The origin of all data sets can be found in the Table 5 .

Graph isomorphism problem asks if such function exists for given two graphs G 1 and G 2 .

We denote isomorphic graphs as G 1 ∼ = G 2 .

The problem has efficient algorithms in P for certain classes of graphs such as planar or bounded-degree graphs (Hopcroft & Wong (1974); Luks (1980) ), but in the general case admits only quasi-polynomial algorithm (Babai (2015) ).

In practice many GI solvers are based on individualization-refinement paradigm (Mckay & Piperno (2014) ), which for each graph iteratively updates a permutation of the nodes such that the resulted permutations of two graphs are identical if an only if they are isomorphic.

Importantly, while finding such canonical permutation of a graph is at least as hard as solving GI problem, state-of-the-art solvers tackle majority of pairs of graphs efficiently, only taking exponential time on the specific hard instances of graphs that possess highly symmetrical structures (Cai et al. (1992b) ).

To distinguish between different isomorphic graphs inside a data set we use the notion of graph orbits:

be a data set of graphs and target labels.

For a graph G i let a set o i = {G k } be a set of all isomorphic graphs in D to G i , including G i .

We call o i the orbit of graph G i in D. The cardinality of the orbit is called orbit size.

An orbit with size one is called trivial.

In a data set with no isomorphic graphs, the number of orbits equals to the number of graphs in a data set, N .

Hence, the more orbits in a data set, the "cleaner" it is.

Note however that the distribution of orbit sizes in two different data sets can vary even if they have the same number of orbits.

Therefore, we look at additional metrics that describe the data set.

• I, aggregated number of graphs that belong to an orbit of size greater than one, i.e. those graphs that isomorphic counterparts in a data set; • I, %, proportion of isomorphic graphs to the total data set size, i.e.

• IP, %, proportion of isomorphic pairs to the total number of graph pairs in a data set (

If we consider target labels of graphs in a data set

we can also measure agreement between the labels of two isomorphic data set.

If G 1 ∼ = G 2 and y 1 = y 2 , then we call graphs mismatched.

Note that if there is more than one target label in an orbit o, then all graphs in this orbit are mismatched.

To obtain isomorphic graphs, we run nauty algorithm (Mckay & Piperno, 2014) on all possible pairs of graphs in a data set.

We substantially reduce the number of calls between the graphs by verifying that a pair has the same number of nodes and edges before the call.

The metrics are presented in Table 2 for top-10 data sets and in Table 6 (see the appendix) for all data sets.

The graphs in Table 2 are sorted by the proportion of isomorphic graphs I%.

The results for the first Top-10 data sets are somewhat surprising: almost all graphs in the selected data sets have other isomorphic graphs.

If we look at all data sets in Table 6 , we see that the proportion of isomorphic graphs in the data sets varies from 100% to 0%.

However, more than 80% of the analyzed data sets have at least 10% of the graphs in a non-trivial orbit.

Another surprising observation is that the proportion of mismatched graphs is significant, ranging from 100% to 0%.

This clearly indicates that such graphs are not suitable for graph classification and require additional information to distinguish the models.

We analyze the reasons for this in the next section.

Also, the distribution of orbit sizes can vary significantly across the data sets.

In Figure 1 we plot a distribution of orbit sizes for several examples of data sets (and distributions for other data sets can be found in Appendix C).

For example, for IMDB-BINARY data set the number of orbits of small sizes, e.g. two or three, goes to 100, which indicate prevalence of pairs of isomorphic graphs that are non-isomorphic to the rest.

However, for Letter-med data set there are many orbits of sizes more than 100, while small orbits are not that common.

In this case, the graphs in this data set are equivalent to a lot of other graphs, which may have a substantial effect on the corresponding metrics.

While the orbit distribution changes from one data set to another, it is clear that in many situations there are isomorphic graphs that can affect training procedure by effectively increasing the weights for the corresponding graphs, change performance on the test by validating on the already seen instances, and by confusing the model by utilizing different target labels for topologically-equivalent graphs.

We analyze the reasons for it further.

Meta-information about graphs.

In addition to the topology of a graph, many data sets also include meta information about nodes and/or edges.

Out of 54 analyzed data sets there are 40 that additionally include node features and 25 that include edge features.

For example, in Synthetic data set all graphs are topologically identical but the nodes are endowed with normally distributed scalar attributes and in DHFR MD edges are attributed with distances and labeled according to a chemical bond type.

Alternatively, some graphs can have parallel edges which is equivalent to have a corresponding weight on the edges.

Thus some data sets include node/edge categorical features (labels) and numerical features (attributes), which leads to better distinction between the graphs and therefore their corresponding labels.

To see this, we rerun our previous analysis but now include the node labels, if any, when computing isomorphism between graphs.

Consider a tuple (G, l), where G is a graph and l : V (G) → {1, 2, . . .

, k} is a k-labeling of G. In this case of node label-preserving graph isomorphism from graph (G 1 , l 1 ) to graph (G 2 , l 2 ) we seek an isomorphism function φ :

Tables 3 and 7 (see the appendix) show the number of isomorphic graphs after considering node labels.

While for the first six data sets the proportion of isomorphic graphs has not changed much, it is clearly the case for the remaining data sets.

In particular, almost 90% of the analyzed data sets include less than 20% of isomorphic graphs.

Also, the number of mismatched graphs significantly decreases after considering node labels.

For example, for MUTAG data set the proportion of isomorphic graphs went down from 42.02% to 19.15% and the proportion of mismatched graphs from 6.91% to 0%.

Likewise, the orbit size distribution also changes significantly after considering node labels.

Figure 2 shows a changed distribution of orbits with and without considering node labels.

For majority of data sets large orbits vanish and the number of small orbits is substantially decreased in label-preserving graph isomorphism setting.

This indicates one of the reasons for presence of many isomorphic graphs in the data sets, which implies that including node/edge labels/attributes can be important for graph classification models.

Sizes of the data sets.

Another reason for having isomorphism in a data set is the sizes of graphs, which could be too small on average to lead to a diversity in a data set.

In general, the number of non-isomorphic graphs with n vertices and m edges can be computed using Polya enumeration theory and grows very fast.

For example, for a graph with 15 nodes and 15 edges, there are 2,632,420 non-isomorphic graphs.

Nevertheless, specifics of the origin of the data set may affect possible configurations that graphs have (e.g. structure of chemical compounds in COX2 MD or ego-networks for actors in IMDB-BINARY) and thus smaller graphs may tend to be close to isomorphic structures.

On the other hand, all five data sets with the average number of nodes greater than 100 have very low or zero proportion of isomorphic graphs.

Hence, the average size of the graphs directly impacts the possible structure of the data set and thus data sets with larger graphs tend to be more diverse.

We next analyze the consequences of the isomorphic graphs on classification methods.

We denote by

• F ⊆

Y X a class of binary classifiers with an input space X and an output space

• π a prior probability of a positive class, i.e. P = πP x|y=+1 + (1 − π)P x|y=−1 ,

We consider a zero-one loss function l(ŷ, y) = Iŷ =y .

Let us assume that classifiers from F can detect which graphs in D are isomorphic.

E.g. classifiers based on Weisfeiler-Lehman graph kernels (see Shervashidze et al. (2011a) ) are capable to do it for majority of graphs.

In such case the empirical risk

where J is an index set of non-isomorphic graphs, u j ≥ 1 is equal to the number of graphs in the initial sample D, isomorphic to the graph x j (we count x j as well).

Thus under such assumptions the graph classification problem with the training data set, containing isomorphic graphs, can be interpreted as a classification problem with a weighted loss.

Let us introduce general notations for this problem.

We define some (fixed) measurable weighting function u : (X × Y ) → (0, +∞).

Then the theoretical risk is equal to E P l(f (x), y) and the weighted empirical risk is equal to

We would like to derive an upper bound for the excess risk

we would like to quantify an upper bound for a generalization gap.

We optimize the weighted empirical risk when training a classifier and measure its accuracy using non-weighted theoretical risk.

There are some results about classification performance with a weighted loss.

E.g. in (Dupret & Koda, 2001 ) a bayesian framework for imbalanced classification with a weighted risk is proposed.

Scott (2012) investigated the calibration of asymmetric surrogate losses.

Natarajan et al. (2018) considered the case of cost-sensitive learning with noisy labels.

However, to the best of our knowledge, there is no studied upper bound for the excess risk with explicit dependence on the class imbalance and the weighting scheme that quantifies the influence on the overall classification performance.

We show this result next.

To derive explicit expressions we use some additional modeling asumption, namely, we consider u(x, y) = (1 + g + (w))I {y=+1} + (1 + g − (w)) I {y=−1} for some non-negative weighting functions g + (w) and g − (w) of the weight value w ≥ 0.

E.g. we can use g + (w) = w and g − (w) = 1/w.

Theorem 6.1.

With probability 1 − δ, δ > 0 for D ∼ P N the excess risk is upper bounded by

where R N (F) is a Rademacher complexity of the function class F.

Let us note that the Rademacher complexity of the function class F, defined by a graph kernel, was studied e.g. in Oneto et al. (2018) ; Oneto et al. (2017a) .

From equation 1 it follows that by tuning the weight parameter w we can make the upper bound tighter, namely collecting the terms with w in the RHS of equation 1 we solve

In case we set g + (w) = w and g − (w) = 1/w, the optimal weight w opt =

≈ 0 for N 1.

For such optimal w opt the RHS of equation 1 has the form

Thus we get theoretical evidence on how the weighting influences the classification accuracy: e.g. in the imbalanced case (when π ≈ 0 or π ≈ 1) selecting the weight optimally we reduce the generalization gap almost to zero for N 1; at the same time, not optimal weight can lead to overfitting.

As we already discussed, under some mild modeling assumptions the graph classification problem with isomorphic graphs in the training data set can be interpreted as the classification problem with a weighted loss.

Therefore the obtained estimate provides additional evidence on a negative effect of the isomorphic graphs when solving the graph classification problems: the presence of isomorphic graphs in the training data set could have the same negative effect as not optimal weight value for the classification with a weighted loss function.

To understand the impact of isomorphic graphs in the data set on the final metric we consider separately the results on two subparts of the data set.

Consider a graph classification model that is evaluated on normalized accuracy over a data set Y :

where acc(G) equals to one if the model predicts the label of G i correctly, and zero otherwise.

If |Y | = 0, then we consider acc(Y ) = 0.

We can see that the accuracy on the test data set can be written as the sum of two terms:

Equation 3 decomposes accuracy on the original data set as the weighted sum of two accuracies on the set of the new test instances Y new and a set of the instances Y iso already appeared in the train set and therefore available to the model.

We call the term acc(Y iso ) as isomorphism bias, which corresponds to the accuracy of the model on the isomorphic test instances.

As we will see next, the accuracy of the model on the new set Y new will be less if only if the model performs better on the isomorphic set Y iso .

The equation 4 gives a definite answer with the possible performance of the model on a new test set.

If the model performs well on isomorphic instances Y iso , then it will falsely increase performance on Y test in comparison to Y new .

Conversely, if the model performs poorly on the instances that appeared in the training set, then removing them from the test set and evaluating the model purely on Y new will demonstrate higher accuracy.

There are two reasons for the model to misclassify isomorphic instances Y iso : (i) the instances contain target labels that are different than those that it has seen, as we show in Table 2 the percentage of mismatched labels can be high in some data sets; or (ii) the model is not expressive enough to map the structure of the graphs to the target label correctly.

Crucially, while Y new tests generalization capabilities of the models, on Y iso the models can explicitly or implicitly memorize the right labels from the training.

We describe a model-agnostic way to guarantee increase of classification performance if |Y iso | = 0.

Note that there can be multiple isomorphic graphs

If for any G ∈ Y iso all target labels of the orbit of G are the same we call the set Y iso as homogeneous.

Consider a classification model M that maps each graph G to its label l(G).

We define a peering model M such that for each G ∈ Y iso it outputs the target label l( G).

Then the accuracy of the model M is at least as the accuracy of the original model M.

If Y iso is homogeneous, then the accuracy on Y test of a classification model M is at most as the accuracy of its peering modelM, i.e.:

Claim 7.2 establishes a way to increase performance only for homogeneous Y iso .

If there are noisy labels in the training set and hence the set is not homogeneous, the model cannot guarantee the right target label for these instances.

Nonetheless, one can select a heuristic such as majority vote among the training isomorphic instances to select a proper label at the testing time.

In experiments, we compare neural network model (NN) (Xu et al., 2018) with graph kernels, Weisfeiler-Lehman (WL) (Shervashidze et al., 2011b) and vertex histogram (V) (Sugiyama & Borgwardt, 2015 ).

For each model we consider two modifications: one for peering model on homogeneous Y iso (e.g. NN-PH) and one for peering model on all Y iso (e.g. NN-P).

We show accuracy on Y test and on Y iso (in brackets) in Table 4 .

Experimentation details can be found in Appendix G.

From Table 4 we can conclude that peering model on homogeneous data is always the top performer.

This is aligned with the result of Claim 7.2, which guarantees that acc(Y iso ) = 1, but it is an interesting observation if we compare it to the peering model on all isomorphic instances Y iso (-P models).

Moreover, the latter model often loses even to the original model, where no information from the train set is explicitly taken into the test set.

This can be explained by the noisy target labels in the orbits of isomorphic graphs, as can be seen both from the statistics for these datasets (Table 6 ) and accuracy measured just on isomorphic instances Y iso .

These results show that due to the presence of isomorphism bias performance of any classification model can be overestimated by as much as 5% of accuracy on these datasets and hence future comparison of classification models should be estimated on Y new instead.

These observations conforms with our theoretical findings and conclusions in Section 6.

In order to avoid measuring performance over the wrong test sets, we provide a set of recommendations that will guarantee measuring the right metrics for the models.

• We open-source new, "clean" data sets that do not include isomorphic instances that are in Table 8 .

To tackle this problem in the future, we propose to use clean versions of the data set for which isomorphism bias vanishes.

For each data set we consider the found graph orbits and keep only one graph from each orbit if and only if the graphs in the orbit have the same label.

If the orbit contains more than one label, a classification model can do little to predict a correct label at the inference time and hence we remove such orbit completely.

In this case, for a new data set Y iso = ∅ and hence it prevents the models to implicitly memorize the labels from the training set.

We consider the data set orbits that do not account for neither node nor edge labels because the remaining graphs are not isomorphic based purely on graph topology.

• Incorporating node and edge features into the models may be necessary to distinguish the graphs.

As we have seen, just using node labels can reduce the number of isomorphic graphs significantly and many data sets provide additional information to distinguish the models at full scope.

• Verification of the models on bigger graphs in general is more challenging due to the sheer number of non-isomorphic graphs.

For example, data sets related to REDDIT or DD include a number of big graphs for classification.

In this work we study isomorphism bias of the classification models in graph structured data that originates from substantial amount of isomorphic graphs in the data sets.

We analyzed 54 graph data sets and provide the reasons for it as well as a set of rules to avoid unfair comparison of the models.

We theoretically characterized the influence of isomorphism bias on the graph classification performance by providing an upper bound on the generalization gap.

We showed that in the current data sets any model can memorize the correct answers from the training set and we open-source new clean data sets where such problems do not appear.

A STATISTICS FOR ORIGINAL DATA SETS

Proof.

From the definition of the peering model we have:

Substituting these into the equation 3 we have:

G EXPERIMENTATION DETAILS NN model is from Xu et al. (2018) and evaluate it on the data sets from PyTorch-Geometric .

For each data set we perform 10-fold cross-validation such that each fold is evaluated on 10% of hold-out instances Y test .

For each fold we train the model for 350 epochs selecting the final model with the best performance on the validation set (20% from hold-out trained split) across all epochs.

Additionally we found that for small data set performance during the first epochs can be unstable on the validation set and thus we select our model only after the first 50 epochs.

The final model is evaluated on the test instances and corresponds to NN in the experiments.

Peering models NN-PH and NN-P are obtained from NN by replicating the target labels for homogeneous Y iso and non-homogeneous Y iso respectively.

Weisfeiler-Lehman and Vertex histogram kernels are taken from the code 2 of Sugiyama & Borgwardt (2015) .

We selected the height of subtree h = 5 for WL kernel.

We train an SVM model selecting C parameter from the range [0.001, 0.01, 0.1, 1, 10].

H NEW DATA SETS Proof.

Let us prove Theorem 6.1.

We denote by L = {(x, y) → L(f (x), y) : f ∈ F} a composite loss class.

For any L ∈ L we get that

Since any L ∈ L is bounded from above by 1 for the first term in equation 5 we obtain E P |(1 − u)L| ≤ E P g + (w)I {y=+1} + E P g − (w)I {y=−1} = g + (w)π + g − (w)(1 − π).

Thanks to McDiarmid'd concentration inequality Mohri et al. (2012) , applied to the function class L u = {uL : L ∈ L}, with probability 1 − δ, δ > 0 for D ∼ P N we get the upper bound on the excess risk sup L∈L (E P uL − E D uL) ≤ 2R N (L u ) + max[(1 + g + (w)), (1 + g − (w))] (log δ −1 )/(2N ) ≤ ≤ 2R N (L u ) + (2 + g + (w) + g − (w)) (log δ −1 )/(2N ).

Let us find a relation between R N (L u ) and R N (L).

Here we denote by z i a pair z i = (x i , y i ).

By the definition (see Mohri et al. (2012) ) the empirical Rademacher complexitŷ

Since we use the zero-one loss, then The Rademacher complexity

Using the fact that R N (L) = 1 2 R N (F), substituting inequalities 6, 7 and 8 into equation 5, we get that sup f ∈F E P l(f (x), y)−E D u(x, y)l(f (x), y) ≤ 3 (g + (w)π + g − (w)(1 − π)) + + R N (F) + (2 + g + (w) + g − (w)) (log δ −1 )/(2N ).

@highlight

Many graph classification data sets have duplicates, thus raising questions about generalization abilities and fair comparison of the models. 

@highlight

The authors discuss isomorphism bias in graph datasets, the overfitting effect in learning networks whenever graph isomorphism features are incorporated within the model, theoretically analogous to data leakage effects.
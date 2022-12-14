While many approaches to make neural networks more fathomable have been proposed, they are restricted to interrogating the network with input data.

Measures for characterizing and monitoring structural properties, however, have not been developed.

In this work, we propose neural persistence, a complexity measure for neural network architectures based on topological data analysis on weighted stratified graphs.

To demonstrate the usefulness of our approach, we show that neural persistence reflects best practices developed in the deep learning community such as dropout and batch normalization.

Moreover, we derive a neural persistence-based stopping criterion that shortens the training process while achieving comparable accuracies as early stopping based on validation loss.

The practical successes of deep learning in various fields such as image processing BID34 BID18 BID21 , biomedicine BID9 BID29 BID28 , and language translation BID2 BID41 still outpace our theoretical understanding.

While hyperparameter adjustment strategies exist BID3 , formal measures for assessing the generalization capabilities of deep neural networks have yet to be identified BID44 .

Previous approaches for improving theoretical and practical comprehension focus on interrogating networks with input data.

These methods include i) feature visualization of deep convolutional neural networks BID43 BID36 , ii) sensitivity and relevance analysis of features BID25 , iii) a descriptive analysis of the training process based on information theory BID39 BID33 BID32 BID1 , and iv) a statistical analysis of interactions of the learned weights BID40 .

Additionally, BID27 develop a measure of expressivity of a neural network and use it to explore the empirical success of batch normalization, as well as for the definition of a new regularization method.

They note that one key challenge remains, namely to provide meaningful insights while maintaining theoretical generality.

This paper presents a method for elucidating neural networks in light of both aspects.

We develop neural persistence, a novel measure for characterizing neural network structural complexity.

In doing so, we adopt a new perspective that integrates both network weights and connectivity while not relying on interrogating networks through input data.

Neural persistence builds on computational techniques from algebraic topology, specifically topological data analysis (TDA), which was already shown to be beneficial for feature extraction in deep learning BID19 and describing the complexity of GAN sample spaces BID23 .

More precisely, we rephrase deep networks with fully-connected layers into the language of algebraic topology and develop a measure for assessing the structural complexity of i) individual layers, and ii) the entire network.

In this work, we present the following contributions: -We introduce neural persistence, a novel measure for characterizing the structural complexity of neural networks that can be efficiently computed.

-We prove its theoretical properties, such as upper and lower bounds, thereby arriving at a normalization for comparing neural networks of varying sizes.

-We demonstrate the practical utility of neural persistence in two scenarios: i) it correctly captures the benefits of dropout and batch normalization during the training process, and ii) it can be easily used as a competitive early stopping criterion that does not require validation data.

Topological data analysis (TDA) recently emerged as a field that provides computational tools for analysing complex data within a rigorous mathematical framework that is based on algebraic topology.

This paper uses persistent homology, a theory that was developed to understand highdimensional manifolds BID15 , and has since been successfully employed in characterizing graphs BID35 BID31 , finding relevant features in unstructured data BID24 , and analysing image manifolds BID7 .

This section gives a brief summary of the key concepts; please refer to for an extensive introduction.

Simplicial homology The central object in algebraic topology is a simplicial complex K, i.e. a high-dimensional generalization of a graph, which is typically used to describe complex objects such as manifolds.

Various notions to describe the connectivity of K exist, one of them being simplicial homology.

Briefly put, simplicial homology uses matrix reduction algorithms BID26 to derive a set of groups, the homology groups, for a given simplicial complex K. Homology groups describe topological features-colloquially also referred to as holes-of a certain dimension d, such as connected components (d = 0), tunnels (d = 1), and voids (d = 2).

The information from the dth homology group is summarized in a simple complexity measure, the dth Betti number ?? d , which merely counts the number of d-dimensional features: a circle, for example, has Betti numbers (1, 1), i.e. one connected component and one tunnel, while a filled circle has Betti numbers (1, 0), i.e. one connected component but no tunnel.

In the context of analysing simple feedforward neural networks for two classes, BID4 calculated bounds of Betti numbers of the decision region belonging to the positive class, and were thus able to show the implications of different activation functions.

These ideas were extended by BID16 to obtain a measure of the topological complexity of decision boundaries.

Persistent homology For the analysis of real-world data sets, however, Betti numbers turn out to be of limited use because their representation is too coarse and unstable.

This prompted the development of persistent homology.

Given a simplicial complex K with an additional set of weights a 0 ??? a 1 ??? ?? ?? ?? ??? a m???1 ??? a m , which are commonly thought to represent the idea of a scale, it is possible to put K in a filtration, i.e. a nested sequence of simplicial complexes DISPLAYFORM0 This filtration is thought to represent the 'growth' of K as the scale is being changed.

During this growth process, topological features can be created (new vertices may be added, for example, which creates a new connected component) or destroyed (two connected components may merge into one).

Persistent homology tracks these changes and represents the creation and destruction of a feature as a point (a i , a j ) ??? R 2 for indices i ??? j with respect to the filtration.

The collection of all points corresponding to d-dimensional topological features is called the dth persistence diagram D d .

It can be seen as a collection of Betti numbers at multiple scales.

Given a point (x, y) ??? D d , the quantity pers(x, y) := |y ??? x| is referred to as its persistence.

Typically, high persistence is considered to correspond to features, while low persistence is considered to indicate noise BID15 .

This section details neural persistence, our novel measure for assessing the structural complexity of neural networks.

By exploiting both network structure and weight information through persistent homology, our measure captures network expressiveness and goes beyond mere connectivity properties.

Subsequently, we describe its calculation, provide theorems for theoretical and empirical DISPLAYFORM0 Figure 1: Illustrating the neural persistence calculation of a network with two layers (l 0 and l 1 ).Colours indicate connected components per layer.

The filtration process is depicted by colouring connected components that are created or merged when the respective weights are greater than or equal to the threshold w bounds, and show the existence of neural networks complexity regimes.

To summarize this section, FIG6 illustrates how our method treats a neural network.

Given a feedforward neural network with an arrangement of neurons and their connections E, let W refer to the set of weights.

Since W is typically changing during training, we require a function ?? : E ??? W that maps a specific edge to a weight.

Fixing an activation function, the connections form a stratified graph.

and (u, v) ??? E, we have j = i + 1.

Hence, edges are only permitted between adjacent vertex sets.

Given k ??? N, the kth layer of a stratified graph is the unique subgraph

This enables calculating the persistent homology of G and each G k , using the filtration induced by sorting all weights, which is common practice in topology-based network analysis BID8 BID20 where weights often represent closeness or node similarity.

However, our context requires a novel filtration because the weights arise from an incremental fitting procedure, namely the training, which could theoretically lead to unbounded values.

When analysing geometrical data with persistent homology, one typically selects a filtration based on the (Euclidean) distance between data points BID5 .

The filtration then connects points that are increasingly distant from each other, starting from points that are direct neighbours.

Our network filtration aims to mimic this behaviour in the context of fully-connected neural networks.

Our framework does not explicitly take activation functions into account; however, activation functions influence the evolution of weights during training.

Filtration Given the set of weights W for one training step, let w max := max w???W |w|.

Furthermore, let W ??? := {|w|/w max | w ??? W} be the set of transformed weights, indexed in non-ascending order, such that 1 = w DISPLAYFORM0 This permits us to define a filtration for the kth layer DISPLAYFORM1 ??? denotes the transformed weight of an edge.

We tailored this filtration towards the analysis of neural networks, for which large (absolute) weights indicate that certain neurons exert a larger influence over the final activation of a layer.

The strength of a connection is thus preserved by the filtration, and weaker weights with |w| ??? 0 remain close to 0.

Moreover, since w ??? ??? [0, 1] holds for the transformed weights, this filtration makes the network invariant to scaling, which simplifies the comparison of different networks.

Persistence diagrams Having set up the filtration, we can calculate persistent homology for every layer G k .

As the filtration contains at most 1-simplices (edges), we capture zero-dimensional topological information, i.e. how connected components are created and merged during the filtration.

These information are structurally equivalent to calculating a maximum spanning tree using the DISPLAYFORM2 ??? Establish filtration of kth layer 5: BID6 .

While it would theoretically be possible to include higher-dimensional information about each layer G k , for example in the form of cliques BID31 , we focus on zero-dimensional information in this paper, because of the following advantages: i) the resulting values are easily interpretable as they essentially describe the clustering of the network at multiple weight thresholds, ii) previous research BID30 BID19 indicates that zero-dimensional topological information is already capturing a large amount of information, and iii) persistent homology calculations are highly efficient in this regime (see below).

We thus calculate zero-dimensional persistent homology with this filtration.

The resulting persistence diagrams have a special structure: since our filtration solely sorts edges, all vertices are present at the beginning of the filtration, i.e. they are already part of G DISPLAYFORM3 k for each k. As a consequence, they are assigned a weight of 1, resulting in |V k ?? V k+1 | connected components.

Hence, entries in the corresponding persistence diagram D k are of the form (1, x), with x ??? W ??? , and will be situated below the diagonal, similar to superlevel set filtrations BID5 BID11 .

Using the p-norm of a persistence diagram, as introduced by BID12 , we obtain the following definition for neural persistence.

which (for p = 2) captures the Euclidean distance of points in D k to the diagonal.

The p-norm is known to be a stable summary BID12 of topological features in a persistence diagram.

For neural persistence to be a meaningful measure of structural complexity, it should increase as a neural network is learning.

We evaluate this and other properties in Section 4.Algorithm 1 provides pseudocode for the calculation process.

It is highly efficient: the filtration (line 4) amounts to sorting all n weights of a network, which has a computational complexity of O(n log n).

Calculating persistent homology of this filtration (line 5) can be realized using an algorithm based on union-find data structures BID15 .

This has a computational complexity of O (n ?? ?? (n)), where ??(??) refers to the extremely slow-growing inverse of the Ackermann function (Cormen et al., 2009, Chapter 22) .

We make our implementation and experiments available under https://github.com/BorgwardtLab/Neural-Persistence.

We elucidate properties about neural persistence to permit the comparison of networks with different architectures.

As a first step, we derive bounds for the neural persistence of a single layer G k .

DISPLAYFORM0 where |V k ?? V k+1 | denotes the cardinality of the vertex set, i.e. the number of neurons in the layer.

Proof.

We prove this constructively and show that the bounds can be realized.

For the lower bound, let G ??? k be a fully-connected layer with |V k | vertices and, given ?? ??? [0, 1], let ?? k (e) := ?? for every edge e. Since a vertex v is created before its incident edges, the filtration degenerates to a lexicographical ordering of vertices and edges, and all points in D k will be of the form (??, ??).

Thus, , while all other pairs will be of the form (b, a).

Consequently, we have DISPLAYFORM1 DISPLAYFORM2 so our upper bound can be realized.

To show that this term cannot be exceeded by NP(G) for any G, suppose we perturb the weight function ??(e) : DISPLAYFORM3 We can use the upper bound of Theorem 1 to normalize the neural persistence of a layer, making it possible to compare layers (and neural networks) that feature different architectures, i.e. a different number of neurons.

The normalized neural persistence of a layer permits us to extend the definition to an entire network.

While this is more complex than using a single filtration for a neural network, this permits us to side-step the problem of different layers having different scales.

Definition 4 (Mean normalized neural persistence).

Considering a network as a stratified graph G according to Definition 1, we sum the neural persistence values per layer to obtain the mean normalized neural persistence, i.e. NP( DISPLAYFORM0 While Theorem 1 gives a lower and upper bound in a general setting, it is possible to obtain empirical bounds when we consider the tuples that result from the computation of a persistence diagram.

Recall that our filtration ensures that the persistence diagram of a layer contains tuples of the form (1, w i ), with w i ??? [0, 1] being a transformed weight.

Exploiting this structure permits us to obtain bounds that could be used prior to calculating the actual neural persistence value in order to make the implementation more efficient.

Theorem 2.

Let G k be a layer of a neural network as in Theorem 1 with n vertices and m edges whose edge weights are sorted in non-descending order, i.e. DISPLAYFORM1 where DISPLAYFORM2 T are the vectors containing the n largest and n smallest weights, respectively.

Proof.

See Section A.2 in the appendix.

As an application of the two theorems, we briefly take a look at how neural persistence changes for different classes of simple neural networks.

To this end, we train a perceptron on the 'MNIST' data set.

Since our measure uses the weight matrix of a perceptron, we can compare its neural persistence with the neural persistence of random weight matrices, drawn from different distributions.

Moreover, we can compare trained networks with respect to their initial parameters.

FIG1 depicts the neural persistence values as well as the lower bounds according to Theorem 2 for different settings.

We can see that a network in which the optimizer diverges (due to improperly selected parameters) is similar to a random Gaussian matrix.

Trained networks, on the other hand, are clearly distinguished from all other networks.

Uniform matrices have a significantly lower neural persistence than Gaussian ones.

This is in line with the intuition that the latter type of networks induces functional sparsity because few neurons have large absolute weights.

For clarity, we refrain from showing the empirical upper bounds because most weight distributions are highly right-tailed; the bound will not be as tight as the lower bound.

These results are in line with a previous analysis BID35 of small weighted networks, in which persistent homology is seen to outperform traditional graph-theoretical complexity measures such as the clustering coefficient (see also Section A.1 in the appendix).

For deeper networks, additional experiments discuss the relation between validation accuracy and neural persistence (Section A.5), the impact of different data distributions, as well as the variability of neural persistence for architectures of varying depth (Section A.6).

This section demonstrates the utility and relevance of neural persistence for fully connected deep neural networks.

We examine how commonly used regularization techniques (batch normalization and dropout) affect neural persistence of trained networks.

Furthermore, we develop an early stopping criterion based on neural persistence and we compare it to the traditional criterion based on validation loss.

We used different architectures with ReLU activation functions across experiments.

The brackets denote the number of units per hidden layer.

In addition, the Adam optimizer with hyperparameters tuned via cross-validation was used unless noted otherwise.

Please refer to Table A .1 in the appendix for further details about the experiments.

We compare the mean normalized neural persistence (see Definition 4) of a two-layer (with an architecture of [650, 650] ) neural network to two models where batch normalization BID22 or dropout BID37 are applied.

FIG4 shows that the networks designed according to best practices yield higher normalized neural persistence values on the 'MNIST' data set in comparison to an unmodified network.

The effect of dropout on the mean normalized neural persistence is more pronounced and this trend is directly analogous to the observed accuracy on the test set.

These results are consistent with expectations if we consider dropout to be similar to ensemble learning BID17 .

As individual parts of the network are trained independently, a higher degree of per-layer redundancy is expected, resulting in a different structural complexity.

Overall, these results indicate that for a fixed architecture approaches targeted at increasing the neural persistence during the training process may be of particular interest.

Neural persistence can be used as an early stopping criterion that does not require a validation data set to prevent overfitting: if the mean normalized neural persistence does not increase by more than ??? min during a certain number of epochs g, the training process is stopped.

This procedure is called 'patience' and Algorithm 2 describes it in detail.

A similar variant of this algorithm, using validation loss instead of persistence, is the state-of-the-art for early stopping in training BID3 BID10 .

To evaluate the efficacy of our measure, we compare it against validation loss in an extensive set of scenarios.

More precisely, for a training process with at most G epochs, we define a G ?? G parameter grid consisting of the 'patience' parameter g and a burn-in rate b (both measured in epochs).

b defines the number of epochs after which an early stopping criterion starts monitoring, thereby preventing underfitting.

Subsequently, we set ??? min = 0 for all measures to remain comparable and scale-invariant, as non-zero values could implicitly favour one of them due to scaling.

For each data set, we perform 100 training runs of the same architecture, monitoring validation loss and mean normalized neural persistence every quarter epoch.

The early stopping behaviour of both measures is simulated for each combination of b and g and their performance over all runs is summarized in terms of median test accuracy and median stopping epoch; if a criterion is not triggered for one run, we report the test accuracy at the end of the training and the number of training epochs.

This results in a scatterplot, where each point (corresponding to a single parameter combination) shows the difference in epochs and the absolute difference in test accuracy (measured in percent).

The quadrants permit an intuitive explanation: Q 2 , for example, contains all configurations for which our measure stops earlier, while achieving a higher accuracy.

Since b and g are typically chosen to be small in an early stopping scenario, we use grey points to indicate uncommon configurations for which b or g is larger than half of the total number of epochs.

Furthermore, to summarize the performance of our measure, we calculate the barycentre of all configurations (green square).Figure 4a depicts the comparison with validation loss for the 'Fashion-MNIST' BID42 data set; please refer to Section A.3 in the appendix for more data sets.

Here, we observe that most common configurations are in Q 2 or in Q 3 , i.e our criterion stops earlier.

The barycentre is at (???0.53, ???0.08) , showing that out of 625 configurations, on average we stop half an epoch earlier than validation loss, while losing virtually no accuracy (0.08%).

FIG5 depicts detailed differences in accuracy and epoch for our measure when compared to validation loss; each cell in a heatmap corresponds to a single parameter configuration of b and g. In the heatmap of accuracy differences, blue, white, and red represent parameter combinations for which we obtain higher, equal, or lower accuracy, respectively, than with validation loss for the same parameters.

Similarly, in the Algorithm 2 Early stopping based on mean normalized neural persistence Require: Weighted neural network N , patience g, ??? min 1: P ??? 0, G ??? 0 ??? Initialize highest observed value and patience counter 2: procedure EARLYSTOPPING(N , g, ??? min ) ??? Callback that monitors training at every epoch 3: DISPLAYFORM0 if P ??? > P + ??? min then ??? Update mean normalized neural persistence and reset counter 5: FIG5 shows how often each measure is triggered.

Ideally, each measure should consist of a dark green triangle, as this would indicate that each configuration stops all the time.

For this data set, we observe that our method stops for more parameter combinations than validation loss, but not as frequently for all of them.

To ensure comparability across scenarios, we did not use the validation data as additional training data when stopping with neural persistence; we refer to Section A.7 for additional experiments in data scarcity scenarios.

We observe that our method stops earlier when overfitting can occur, and it stops later when longer training is beneficial.

DISPLAYFORM1

In this work, we presented neural persistence, a novel topological measure of the structural complexity of deep neural networks.

We showed that this measure captures topological information that pertains to deep learning performance.

Being rooted in a rich body of research, our measure is theoretically well-defined and, in contrast to previous work, generally applicable as well as computationally efficient.

We showed that our measure correctly identifies networks that employ best practices such as dropout and batch normalization.

Moreover, we developed an early stopping criterion that exhibits competitive performance while not relying on a separate validation data set.

Thus, by saving valuable data for training, we managed to boost accuracy, which can be crucial for enabling deep learning in regimes of smaller sample sizes.

Following Theorem 2, we also experimented with using the p-norm of all weights of the neural network as a proxy for neural persistence.

However, this did not yield an early stopping measure because it was never triggered, thereby suggesting that neural persistence captures salient information that would otherwise be hidden among all the weights of a network.

We extended our framework to convolutional neural networks (see Section A.4) by deriving a closed-form approximation, and observed that an early stopping criterion based on neural persistence for convolutional layers will require additional work.

Furthermore, we conjecture that assessing dissimilarities of networks by means of persistence diagrams (making use of higher-dimensional topological features), for example, will lead to further insights regarding their generalization and learning abilities.

Another interesting avenue for future research would concern the analysis of the 'function space' learned by a neural network.

On a more general level, neural persistence demonstrates the great potential of topological data analysis in machine learning.

Traditional complexity/structural measures from graph theory, such as the clustering coefficient, the average shortest path length, and global/local efficiency are already known to be insufficiently accurate to characterize different models of complex random networks BID35 .

Our experiments indicate that this holds true for (deep) neural networks, too.

As a brief example, we trained a perceptron on the MNIST data set with batch stochastic gradient descent (?? = 0.5), achieving a test accuracy of ??? 0.91.

Moreover, we intentionally 'sabotaged' the training by setting ?? = 1 ?? 10 ???5 such that SGD is unable to converge properly.

This leads to networks with accuracies ranging from 0.38-0.65.

A complexity measure should be capable of distinguishing both classes of networks.

However, as Figure A .1 (top) shows, this is not the case for the clustering coefficient.

Neural persistence (bottom), on the other hand, results in two regimes that can clearly be distinguished, with the trained networks having a significantly smaller variance.

Proof.

We may consider the filtration from Section 3.1 to be a subset selection problem with constraints, where we select n out of m weights.

The neural persistence NP(G k ) of a layer thus only depends on the selected weights that appear as tuples of the form (1, w i ) in D k .

Letting w denote the vector of selected weights arising from the persistence diagram calculation, we can rewrite neural persistence as NP(G k ) = ???1 ??? w??? p .

Furthermore, w satisfies ???w min ??? p ??? ??? w??? p ??? ???w max ??? p .

Since all transformed weights are non-negative in our filtration, it follows that (note the reversal of the two terms) DISPLAYFORM0 and the claim follows.

Due to space constraints and the large number of configurations that we investigated for our early stopping experiments, this section contains additional plots that follow the same schematic: the top row shows the differences in accuracy and epoch for our measure when compared to the commonlyused validation loss.

Each cell in the heatmap corresponds to a single configuration of b and g. In the heatmap of accuracy differences, blue represents parameter combinations for which we obtain a higher accuracy than validation loss for the same parameters; white indicates combinations for which we obtain the same accuracy, while red highlights combinations in which our accuracy decreases.

Similarly, in the heatmap of epoch differences, green represents parameter combinations for which we stop earlier than validation loss for the same parameter.

The scatterplots in Section 4.2 show an 'unrolled' version of this heat map, making it possible to count how many parameter combinations result in early stops while also increasing accuracy, for example.

The heatmaps, by contrast, make it possible to compare the behaviour of the two measures with respect to each parameter combination.

Finally, the bottom row of every plot shows how many times each measure was triggered for every parameter combination.

We consider a measure to be triggered if its stopping condition is satisfied prior to the last training epoch.

Due to the way the parameter grid is set up, no configuration above the diagonal can stop, because b + g would be larger than the total number of training epochs.

This permits us to compare the 'slopes' of cells for each measure.

Ideally, each measure should consist of a dark green triangle, as this would indicate that parameter configuration stops all the time.

MNIST Please refer to FIG1 3.

The colours in the difference matrix of the top row are slightly skewed because in a certain configuration, our measure loses 0.8% of accuracy when stopping.

However, there are many other configurations in which virtually no accuracy is lost and in which we are able to stop more than four epochs earlier.

The heatmaps in the bottom row again indicate that neural persistence is capable of stopping for more parameter combinations in general.

We do not trigger as often for some of them, though.

CIFAR-10 Please refer to Figure A. 4.

In general, we observe that this data set is more sensitive with respect to the parameters for early stopping.

While there are several configurations in which neural persistence stops with an increase of almost 10% in accuracy, there are also scenarios in which we cannot stop training earlier, or have to train longer (up to 15 epochs out of 80 epochs in total).

The second row of plots shows our measure triggers reliably for more configurations than validation loss.

Overall, the scatterplot of all scenarios (Figure A.5) shows that most practical configurations are again located in Q 2 and Q 3 .

While we may thus find certain configurations in which we reliably outperform validation loss as an early stopping criterion, we also want to point out that our measures behaves correctly for many practical configurations.

Points in Q 1 , where we train longer and achieve a higher accuracy, are characterized by a high patience g of approximately 40 epochs and a low burn-in rate b, or vice versa.

This is caused by the training for CIFAR-10, which does not reliably converge for FCNs.

Figure A .6 demonstrates this by showing loss curves and the mean normalized neural persistence curves of five runs over training (loss curves have been averaged over all runs; standard deviations are shown in grey; we show the first half of the training to highlight the behaviour for practical early stopping conditions).

For 'Fashion-MNIST', we observe that NP exhibits clear change points during the training process, which can be exploited for early stopping.

For 'CIFAR-10', we observe a rather incremental growth for some runs (with no clearlydefined maximum), making it harder to derive a generic early stopping criterion that does not depend on fine-tuned parameters.

Hence, we hypothesize that neural persistence cannot be used reliably in scenarios where the architecture is incapable of learning the data set.

In the future, we plan to experiment with deliberately selected 'bad' and 'good' architectures in order to evaluate to what extent our topological measure is capable of assessing their suitability for training, but this is beyond the scope of this paper.

IMDB Please refer to Figure A .7.

For this data set, we observe that most parameter configurations result in earlier stopping (up to two epochs earlier than validation loss), with accuracy increases of up to 0.10%.

This is also shown in the scatterplot A.8.

Only a single configuration, viz.

g = 1 and b = 0, results in a severe loss of accuracy; we removed it from the scatterplot for reasons of clarity, as its accuracy difference of ???21% would skew the display of the remaining configurations too much (this is also why the legends do not include this outlier).

In principle, the proposed filtration process could be applied to any bipartite graph.

Hence, we can directly apply our framework to convolutional layers, provided we represent them properly.

Specifically, for layer l we represent the convolution of its ith input feature map a DISPLAYFORM0 with the jth filter H j ??? R p??q as one bipartite graph G i,j parametrized by a sparse weight matrix DISPLAYFORM1 , which in each row contains the p ?? q unrolled values of H j on the diagonal, with h in ??? p zeros padded in between after each p values of vec(H j ).

This way, the flattened pre-activation can be described as vec(z DISPLAYFORM2 Since flattening does not change the topology of our bipartite graph, we compute the normalized neural persistence on this sparse weight matrix W (l) i,j as the unrolled analogue of the fully-connected network's weight matrix.

Averaging over all filters then gives a per-layer measure, similar to the way we derived mean normalized neural persistence in the main paper.

When studying the unrolled adjacency matrix W (l) i,j , it becomes clear that the edge filtration process can be approximated in a closed form.

Specifically, for m and n input and output neurons we initialize ?? = m + n connected components.

When using zero padding, the additional dummy input neurons have to included in m. For all ?? tuples in the persistence diagram the creation event c = 1.

Notably, each output neuron shares the same set of edge weights.

Due to this, the destruction events-except for a few special cases-simplify to a list of length ?? containing the largest filter values (each value is contained n times) in descending order until the list is filled.

This simplification of neural persistence of a convolution with one filter is shown as a closed expression in Equations 7-11, and our implementation is sketched in Algorithm 3.

We thus obtain DISPLAYFORM3 where we use DISPLAYFORM4 DISPLAYFORM5 DISPLAYFORM6 DISPLAYFORM7 where 1 0 := 0.

Following this notation, Equation 7 expresses neural persistence of the bipartite graph G i,j , with w denoting the vector of selected weights (i.e. the destruction events) when calculating the persistence diagram.

We use w to denote the flattened and sorted weight values (in descending order) of the convolutional filter H j , while w c represents the vector of all weights that are located in a corner of H j , whereas wc ,?? is the vector of all weights which do not originate from the corner of the filter while still belonging to the first (and thus largest) DISPLAYFORM8 weights in w, which we denote by w 1:???

?? n ??? .

For the subsequent experiments (see below), we use a simple CNN that employs 32 + 2048 filters.

Hence, by using the shortcut described above, we do not have to unroll 2080 weight matrices explicitly, thereby gaining both in memory efficiency and run time, as compared to the naive approach: on average, a naive exact computation based on unrolling required 8.77 s per convolutional filter and evaluation step, whereas the approximation only took about 0.000 38 s while showing very similar behaviour up to a constant offset.

For our experiments, we used an off-the-shelf 'LeNet-like' CNN model architecture (two convolutional layers each with max pooling and ReLU, 1 fully-connected and softmax) as described in BID0 .

We trained the model on 'Fashion-MNIST' and included this setup in the early stopping experiments (100 runs of 20 epochs).

In Figure A .9, we observe that stopping based on the neural persistence of a convolutional layer typically only incurs a considerable loss of accuracy: given a final test accuracy of 91.73??0.13, stopping with this naive extension of our measure reduces accuracy by up to 4%.

Furthermore, in contrast to early stopping on a fully-connected architecture, FIG1 , which shows the different 'regimes' of neural persistence for a perceptron network, we investigate a possible correlation of (high) neural persistence with (high) predictive accuracy.

For deeper networks, we find that neural persistence measures structural properties that arise from different parameters (such as training procedures or initializations), and no correlation can be observed.

DISPLAYFORM9 For our experiments, we constructed neural networks with a high neural persistence prior to training.

More precisely, following the theorems in this paper, we initialized most weights of each layer with very low values and reserved high values for very few weights.

This was achieved by sampling the weights from a beta distribution with ?? = 0.005 and ?? = 0.5.

Using this procedure, we are able to initialize [20, 20, 20] networks with NP ??? 0.90 ?? 0.003 compared to the same networks that have NP ??? 0.38 ?? 0.004 when initialized by Xavier initialization.

The mean validation accuracy of these untrained networks on the 'Fashion-MNIST' data set is 0.10 ?? 0.01 and 0.09 ?? 0.03, respectively.

Here, the approximated neural persistence calculation for the first convolutional layer was used.

However, we also ran few runs of the same experiment using the exact method which showed the same results.

Employing the second convolutional layer or both did not improve this result.

We also investigated the effect of depth on neural persistence.

We selected a fixed layer size (20 hidden units) and increased the number of hidden layers.

Figure A .11 (right) depicts the boxplots of mean NP for multiple architectures after 15 epochs of training on MNIST.

Adding layers initially increases the variability of NP by enabling the network to converge to different regimes (essentially, there are many more valid configurations in which a trained neural network might end up in).

However, this effect is reduced after a certain depth: networks with deeper architectures exhibit less variability in NP.

Labelled data is expensive in most domains of interest, which results in small data sets or low quality of the labels.

We investigate the following experimental set-ups: (1) Reducing the training data set size and (2) Permuting a fraction of the training labels.

We train a fully connected network ([500, 500, 200] architecture) on 'MNIST' and 'Fashion-MNIST'.

In the experiments, we compare the following measures for stopping the training: i) Stopping at the optimal test accuracy.

ii) Fixed stopping after the burn in period.

iii) Neural persistence patience criterion.

iv) Training loss patience criterion.

v) Validation loss patience criterion.

For a description of the patience criterion, see Algorithm 2.

All measures, except validation loss, include the validation datasets (20%) in the training process to simulate a larger data set when no cross-validation is required.

We report the accuracy on the non-reduced, non-permuted test sets.

The batch size is 32 training instances.

The stopping measures are evaluated every quarter epoch.

Figure A .12 shows the results averaged over 10 runs (the error is the standard deviation).

The difference between the top and the bottom panel is the data set and the patience parameters.

The x-axis depicts the fraction of the data set, which is warped for better accessibility.

In each panel, the left-hand side subplots depict the results of the reduced data set experiment where the right-hand side subplots depict the result of the permutation experiments.

The y-axis of the top subplot shows the accuracy on the non-reduced, non-permuted test set.

The y-axis of the bottom subplot shows when the stopping criterion was triggered.

We note the following observations, which hold for both panels: More, non-permuted data yields higher test accuracy.

Also, as expected, the optimal stopping gives the highest test accuracy.

The fixed early stopping results in inferior test accuracy when only a fraction of the data is available.

The neural persistence based stopping is triggered late when only a fraction of the data is available which results in a slightly better test accuracy compared to training and validation loss.

The training loss stopping achieves similar test accuracies compared to the persistence based stopping (for all regimes except the very small data set) with shorter training, on average.

We note that, it is generally not advisable to use training loss as a measure for stopping because the stability of this criterion also depends on the batch size.

When only a fraction of the data is available, the validation loss based stopping stops on average after the same number of training epochs as the training loss, which results in inferior test accuracy because the network has seen in total fewer training samples.

Most strikingly, validation loss based stopping is is triggered later (sometimes never) when most training and validation labels are randomly permuted which results in overfitting and poor test accuracy.

To conclude, the neural persistence based stopping achieves good performance without being affected by the batch size and noisy labels.

The authors also note that the result is consistent for multiple architectures and most patience parameters.

For increasing noise in the training labels (right-hand side), the stopping of NP remains stable, in contrast to the validation loss stopping, which leads to lower test accuracy after longer training at a high fraction of permuted labels.

The patience and burn in parameters are reported in quarter epochs.

We showed in the main text that neural persistence is capable of distinguishing between networks trained with/without batch normalization and/or dropout.

Figure A .13 additionally shows test set accuracies.

@highlight

We develop a new topological complexity measure for deep neural networks and demonstrate that it captures their salient properties.

@highlight

This paper proposes the notion of neural persistence, a topological measure to assign scores to fully-connected layers in a neural network.

@highlight

Paper proposes to analyze the complexity of a neural network using its zero-th persistent homology.
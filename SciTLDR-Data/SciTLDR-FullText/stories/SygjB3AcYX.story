The key challenge in semi-supervised learning is how to effectively leverage unlabeled data to improve learning performance.

The classical label propagation method, despite its popularity, has limited modeling capability in that it only exploits graph information for making predictions.

In this paper, we consider label propagation from a graph signal processing perspective and decompose it into three components: signal, filter, and classifier.

By extending the three components, we propose a simple generalized label propagation (GLP) framework for semi-supervised learning.

GLP naturally integrates graph and data feature information, and offers the flexibility of selecting appropriate filters and domain-specific classifiers for different applications.

Interestingly, GLP also provides new insight into the popular graph convolutional network and elucidates its working mechanisms.

Extensive experiments on three citation networks, one knowledge graph, and one image dataset demonstrate the efficiency and effectiveness of GLP.

The success of deep learning and neural networks comes at the cost of large amount of training data and long training time.

Semi-supervised learning BID37 BID8 ) is interesting and important as it can leverage ample available unlabeled data to aid supervised learning, thus greatly saving the cost, trouble, and time for human labeling.

Many researches have shown that when used properly, unlabeled data can significantly improve learning performance BID38 BID16 .

The key challenge for semi-supervised learning is how to effectively leverage the information of unlabeled data, such as graph structures and data features.

Label propagation BID39 BID36 BID2 is arguably the most popular method for graph-based semi-supervised learning.

As a simple and effective tool, it has been widely used in many scientific research fields and has found numerous industrial applications.

Given a non-oriented graph G = (V, W, X) with n = |V| vertices, a nonnegative symmetric affinity matrix W ∈ R n×n + encoding edge weights, and a feature matrix X ∈ R n×m which contains an mdimensional feature vector of each vertex.

For semi-supervised classification, only a small subset of vertices are labeled, and the goal is to predict the labels of other vertices.

Denote by Y ∈ {0, 1} n×l the labeling matrix 1 with l being the number of classes.

The objective of of label propagation (LP) is to find a prediction (embedding) matrix Z ∈ R n×l which agrees with Y while being smooth on the graph such that nearby vertices have similar embeddings: DISPLAYFORM0 where α is a balancing parameter, L = D − W is the graph Laplacian 2 and D is the degree matrix.

The term enforcing smoothness is called graph Laplacian regularization or Tikhonov regularization.

Solving the quadratic regularization framework gives the prediction of LP.As LP makes predictions only based on graph information (W ), its performance depends on whether the underlying graph structure can well represent the class information of data -vertices in the same 1 If the label of vertex vi is known, then Y (i, :) is a one-hot embedding of vi with yij = 1 if vi belongs to the j-th class and yij = 0 otherwise.

If the label of vertex vi is not given, then Y (i, :) is a vector of all zeros.2 Other variants such as the normalized Laplacian matrices are also applicable.cluster tend to have same labels.

For some applications such as social network analysis, data exhibits a natural graph structure.

For some other applications such as image or text classification, data may come in a vector form, and a graph is usually constructed using data features.

Nevertheless, in many cases, graphs only partially encode data information.

Take document classification in a citation network as an example, the citation links between documents form a graph which represents their citation relation, and each document is represented as a bag-of-words feature vector which describes its content.

To correctly classify a document, both the citation relations (W ) and the content information (X) need to be taken into account, as they contain different aspects of document information.

However, in this case, LP can only exploit the graph information to make predictions without using any of the feature information, thus resulting in poor performance.

To go beyond the limit of LP and jointly model graph and feature information, a common approach is to train a supervised learner to classify data features while regularizing the classifier using graph information.

Manifold regularization BID1 trains a support vector machine with a graph Laplacian regularizer.

Deep semi-supervised embedding BID32 and Planetoid BID34 ) train a neural network with an embedding-based regularizer.

The recently proposed graph convolutional neural networks BID16 ) adopts a different approach by integrating graph and feature information in each of its convolutional layer, which is coupled with a projection layer for classification.

In this paper, we extends the modeling capability of LP in the context of graph signal processing.

Casted in the spectral domain, LP can be interpreted as low-pass graph filtering BID10 BID11 .

In light of this, we decompose LP into three components: graph signal, graph filter, and classifier.

By naturally extending the three components, we propose a generalized label propagation (GLP) framework for semi-supervised learning.

In GLP, a low-pass graph filter is applied on vertex features to produce smooth features, which are then fed to a supervised learner for classification.

After filtering, the data features within each class are more similar and representative, making it possible to train a good classifier with few labeled examples.

GLP not only extends LP to incorporate vertex features in a simple way, but also offers the flexibility of designing appropriate graph filters and adopting domain-specific classifiers for different semisupervised applications.

The popular graph convolutional networks (GCN) BID16 is closely related to GLP.

In fact, GCN without internal ReLUs is a special case of GLP with a certain graph filter and a multilayer perceptron classifier.

When revisited under the GLP framework, it makes clear the working mechanisms of GCN including its design of convolutional filter and model parameter setting.

Extensive experiments on citation networks, knowledge graphs, and image datasets show substantial improvement of GLP over GCN and other baselines for semi-supervised classification, confirming the effectiveness of this simple and flexible framework.

The rest of the paper is organized as follows.

Section 2 interprets LP in the context of graph signal processing.

Section 3 presents the proposed GLP framework.

Section 4 revisits GCN under GLP.

Section 5 discusses the design of graph filters for GLP.

Section 6 presents experimental results.

Section 7 discusses related works.

Finally, section 8 concludes the paper.

In this section, we provide a spectral view of LP in the context of graph signal processing.

In graph signal processing BID24 , the eigenvectors and eigenvalues of the graph Laplacian play the role of Fourier basis and frequencies in parallel with classical harmonic analysis.

The graph Laplacian matrix can be eigen-decomposed as: L = ΦΛΦ −1 , where Λ = diag(λ 1 , · · · , λ n )

are the eigenvalues in an increasing order, i.e., 0 = λ 1 ≤ · · · ≤ λ n , and Φ = (φ 1 , · · · , φ n ) are the associated orthogonal eigenvectors.

Note that the row normalized graph Laplacian L r = D −1 L and the symmetrically normalized graph Laplacian L s = D A graph signal is a real-valued function f : V → R defined on the vertex set of a graph.

Denote by f = (f (v 1 ), · · · , f (v n )) a graph signal in a vector form.

Consider (φ i ) 1≤i≤n as basis functions.

Any graph signal f can be decomposed into a linear combination of the basis functions: DISPLAYFORM0 where c = (c 1 , · · · , c n ) and c i is the coefficient of φ i .

The magnitude of the coefficient |c i | represents the strength of the basis function φ i presented in the signal f .A graph filter is defined as a matrix G ∈ R n×n .

G is linear shift-invariant BID22 , if and only if there exists an function p(·) : R → R, satisfying G = Φp(Λ)Φ −1 , where DISPLAYFORM1 It is well known that the basis functions associated with lower frequencies (smaller eigenvalues) are smoother BID38 , as the smoothness of φ i can be measured by λ i : DISPLAYFORM2 This indicates that a smooth signal f should contain more low-frequency components than highfrequency components.

To produce a smooth signal, the graph filter G should be able to preserve the low-frequency components in f while filtering out the high-frequency components.

By Eq. (2), we havef DISPLAYFORM3 In the filtered signalf , the coefficient c i of the basis function φ i is scaled by p(λ i ).

To preserve the low-frequency components and remove the high-frequency components, p(λ i ) should amplify c i when λ i is small and suppress c i when λ i is large.

Simply put, p(·) should behave like a low-pass filter in classical harmonic analysis.

The prediction (embedding) matrix of LP can be obtained by taking the derivative of the unconstrained quadratic optimization problem in Eq.(1) and setting it to zero: DISPLAYFORM0 With the prediction matrix Z, each unlabeled vertex v i is usually classified by simply comparing the elements in Z(i, :).

In some methods, a normalization scheme may be applied on the columns of Z first before the comparison BID39 .Casted in the context of graph signal processing, LP can be decomposed into three components: signal, filter, and classifier.

By Eq. (5), the input signal matrix of LP is the labeling matrix Y , where it has l channels and each column Y (:, i) can be considered as a graph signal.

In Y (:, i), only the labeled vertices in class i have value 1 and others 0.The graph filter used in LP is DISPLAYFORM1 with frequency response function DISPLAYFORM2 Note that this also holds for the normalized graph Laplacians.

As shown in FIG2 , the frequency response function of LP is low-pass.

For any α > 0, p(λ i ) is near 1 when λ i is close to 0 and p(λ i ) decreases and approaches 0 as λ i increases.

Applying the filter on signal Y (:, i), it will produce a smooth signal Z(:, i) in which vertices of the same class have similar values and vertices in class i have larger values than others under the cluster assumption.

The balancing parameter α controls the degree of the graph Laplacian regularization.

When α increases, the filter becomes more low-pass ( FIG2 ) and will produce smoother embeddings.

Finally, LP applies a nonparametric classifier on the embeddings to classify the unlabeled vertices, i.e., the label of an unlabeled vertex v i is given by y i = arg max j Z(i, j).

We propose a generalized label propagation (GLP) framework by naturally generalizing the three components of LP for semi-supervised classification:• Signal: Use the feature matrix X instead of the labeling matrix Y as input signal.• Filter: The filter G can be any low-pass, linear, shift-invariant filter.• Classifier: The classifier can be any classifer trained on the embeddings of labeled vertices.

GLP consists of two steps.

First, a low-pass, linear, shift-invariant graph filter G is applied on the feature matrix X to obtain a smooth feature matrixX ∈ R n×m : DISPLAYFORM0 The next step is to train a supervised classifier (e.g., multilayer perceptron, convolutional neural networks, support vector machines, etc.) with the filtered features of labeled data, and then apply the classifier on the filtered features of unlabeled data to predict their labels.

GLP naturally combines graph and feature information in Eq. FORMULA9 , and allows taking advantage of a powerful supervised classifier.

The rationale behind GLP is to learn representative feature vectors of each class for easing the downstream classification task.

After filtered by G, vertices in the same class are expected to have more similar and representative features, which makes it much easier to train a good classifier with very few samples.

Consider an extreme case that each class is a connected component of the graph.

In this case, we can learn perfect features by an extremely low-pass filter G, whose spectrum p(·) is unit impulse function, i.e., p(0) = 1 and p(λ) = 0 if λ = 0.

We can compute G = Φp(Λ)Φ −1 in the spatial domain.

In particular, G ij = 1 l k if v i and v j are of the same class, otherwise G ij = 0, where l k is the number of labeled samples in class k. After filtered by G, vertices in the same class will have an identical feature vector which is its class mean.

Then any classifier that can correctly classify the labeled data will achieve 100% accuracy on the unlabeled data, and only one labeled example per class is needed to train the classifier.

In this section, we show that the graph convolutional networks (GCN) (Kipf & Welling, 2017) for semi-supervised classification can be interpreted under the GLP framework, which explains its implicit design features including the number of layers, the choice of the normalized graph Laplacian, and the renormalization trick on the convolutional filter.

Graph Convolutional Networks.

The GCN model contains three steps.

First, a renormalization trick is applied on the adjacency matrix W by adding an self-loop to each vertex, which results in a new adjacency matrixW = W + I with the degree matrixD = D + I. After that, symmetrically normalizeW and getW s =D DISPLAYFORM0 where H (t) is the matrix of activations in the t-th layer and DISPLAYFORM1 is the trainable weight matrix in layer t, and σ is the activation function, e.g., ReLU(·) = max(0, ·).

The graph convolution is defined by multiplying the input of each layer with the renormalized adjacency matrixW s from the left, i.e.,W s H (t) .

The convoluted features are then fed into a projection matrix Θ (t) .

Third, stack two layers up and apply a softmax function on the output features to produce a prediction matrix: DISPLAYFORM2 and train the model using the cross-entropy loss over the labeled instances.

The graph convolution in each layer of the GCN model actually performs feature smoothing with a low-pass filterW s = I −L s , whereL s is the symmetrically normalized graph Laplacian of the graph with extra self-loops.

Suppose thatL s can be eigen-decomposed asL s = ΦΛΦ −1 , then we have I −L s = Φ(I −Λ)Φ −1 .

The frequency response function of the filter is DISPLAYFORM0 Clearly, as shown in FIG2 , this function is linear and low-pass on the interval [0, 1], but not on [1, 2], as it amplifies the eigenvalues near 2.Interestingly, by removing the activation function ReLU in Eq. (9), we can see that GCN is a special case of GLP, where the input signal is X, the filter isW 2 s , and the classifier is a two-layer multi-layer perceptron (MLP).Why the Normalized Graph Laplacian.

Note that the eigenvalues of the normalized Laplacians L s and L r all fall into interval [0, 2] (Chung, 1997), while the unnormalized Laplacian L has eigenvalues in [0, +∞].

If using the unnormalized graph Laplacian, the response function in Eq. (10) will amplify eigenvalues in [2, +∞], which will introduce noise and undermine performance.

Why Two Convolutional Layers.

In Eq. (9), the GCN model stacks two convolutional layers.

Without the activation function, the feature matrix is actually filtered by I −L s twice, which is equivalent to be filtered by (I −L s ) 2 with response function (1 − λ) 2 .

As we can see from FIG2 , DISPLAYFORM1 2 is more low-pass than (1 − λ) by suppressing the eigenvalues in the mid-range of [0, 2] harder, which explains why GCNs with two convolutional layers perform better than those with only one.

Why the Renormalization Trick.

The effect of the renormalization trick is illustrated in FIG4 , where the response functions on the eigenvalues of L s andL s on the Cora citation network are plotted.

We can see that by adding a self-loop to each vertex, the range of eigenvalues shrink from [0, 2] to [0, 1.5], thus avoiding amplifying eigenvalues near 2 and reducing noise.

This explains why the renormalization trick works.

In this section, we discuss the design and computation of low-pass graph filters for GLP.Auto-Regressive.

The Auto-Regressive (AR) filter is the one used in LP: DISPLAYFORM0 Actually p ar is an auto-regressive filter of order one BID29 .

We have shown p ar is low-pass in section 2.2.

However, the computation of p ar involves matrix inversion, which is also computationally expensive with complexity O(n 3 ).

Fortunately, we can circumvent this problem by approximating p ar using its polynomial expansion: DISPLAYFORM1 We can then computeX = p ar (L)X iteratively with DISPLAYFORM2 and letX = 1 1+α X (k) .

Empirically, we find that k = 4α is enough to get a good approximation.

Hence, the computational complexity is reduced to O(nmα + N mα) (note that X is of size n × m), where N is the number of nonzero entries in L, and N n 2 when the graph is sparse.

Renormalization.

The renormalization (RNM) filter is an exponential function of the renormalized adjacency filter used in GCN: DISPLAYFORM3 We have shown in section 4.1 that although the response function p rnm (λ) = (1 − λ) k is not lowpass, the renormalization trick shrinks the range of eigenvalues ofL and makes p rnm resemble a low-pass filter.

The exponent parameter k controls the low-pass effect of p rnm .

When k = 0, p rnm is all-pass.

When k increases, p rnm becomes more low-pass.

Note that for a sparse graph, (I −L) is a sparse matrix.

Hence, the fastest way to computeX = p rnm (L)X is to left multiply X by (I −L) repeatedly for k times, which has the computational complexity O(N mk).Random Walk.

We also propose to design a random walk (RW) filter: DISPLAYFORM4 We call p rw the random walk filter because DISPLAYFORM5 is a stochastic matrix of a lazy random walk which at each step returns to the current state with probability 1 2 , and DISPLAYFORM6 is the k-step transition probability matrix.

Similarly, we can derive the response function of p rw as DISPLAYFORM7 Note that L r has the same eigenvalues with L s , with range [0, 2].

Unlike the RNM, p rw is a typical low-pass filter on [0, 2], as shown in FIG2 .

We can also see in FIG2 that the curves of (1−λ) 2 and (1 − 1 2 λ) 4 are very close, implying that to have the same level of low-pass effect, k in p rw should be set twice as large as in p rnm .

This may be explained by the fact that the two functions (1 − λ) DISPLAYFORM8 2k have the same derivative k at λ = 0.

On the computation side, RW has the same complexity O(N mk) as RNM.An important issue of filter design for GLP is how to control the strength of filters by setting parameters such as α and k. Intuitively, when labeled data is scarce, it would be desirable for the filtered features of each instance to be closer to its class mean and be more representative of its own class.

Hence, in this case, α and k should be set large to produce smoother features.

However, oversmoothing usually results in inaccurate class boundaries.

Therefore, when the amount of labeled data is reasonably large, α and k should be set relatively small to preserve feature diversity in order to learn more accurate class boundaries.

Datasets In this section, we test GLP on three citation networks -Cora, CiteSeer and PubMed BID23 , one knowledge graph -NELL BID5 , and one handwritten digit image dataset -MNIST BID18 .

Dataset discriptions are provided in Appendix A. Baselines On citation networks and NELL, we compare GLP against GCN BID16 , LP BID33 , multi-layer perceptron (MLP), Planetoid BID34 , DeepWalk BID21 , manifold regularization (ManiReg) BID1 , semi-supervised embedding (SemiEmb) BID32 , and iterative classification algorithm (ICA) BID23 .

On MNIST, we compare GLP against GCN, LP, MLP, and convolutional neural networks (CNN).Experimental Setup We test GLP with RNM, RW and AR filters (section 5) on all the datasets.

We use MLP as the classifier for GLP on citation networks and NELL, and use CNN as the classifier on MNIST.

Guided by our analysis in section 5, the filter parameters k and α should be set large with small label rate and set small with large label rate.

We use fixed parameters k = 10 for RNM, k = 20 for RW, and α = 20 for AR when label rate is less than or equal to 2%, and set them to 5, 10, 10 respectively otherwise.

We follow BID16 to set the parameters of MLP, including learning rate, dropout, and weight decay.

To make sure GLP works in practice and for more fair comparison with baselines, we do not use a validation set for classifier model selection as in BID16 , instead we select the classifier with the highest training accuracy in 200 steps.

Results of GLP and GCN on all the datasets are reported without using a validation set, except that on NELL, we also report the results with validation (on the right of "/").

More implementation details are provided in Appendix B due to space limitations.

Performance of GLP The results are summarized in TAB0 , where the top 3 classification accuracies are highlighted in bold.

Overall, GLP performs the best on all the datasets.

On citation networks, with 20 labels per class, GLP performs comparably with GCN and outperforms other baselines by a considerable margin.

With 4 labels per class, GLP significantly outperforms all baselines including GCN.

On NELL, GLP wtih RW and RNM filters consistently outperforms the best baseline Planetoid for each setting, and outperforms other baselines including GCN by a large margin.

Note that GLP achieves this performance without using any additional validation set.

The performance of GLP (and GCN) will be further boosted with validation, as shown on the right of "/".

On MNIST, GLP consistently outperforms all baselines for every setting.

The running times of GLP and some other baselines are also reported in TAB0 .

GLP runs much faster than GCN on most datasets, except for NELL, on which the running times of GLP with two filters are similar with GCN.

More discussions about running times are included in Appendix E.Results Analysis Compared with LP and DeepWalk which only use graph information, the large performance gains of GLP clearly comes from leveraging both graph and feature information.

Compared with purely supervised MLP and CNN which are trained on raw features, the performance gains of GLP come from the unsupervised feature filtering.

FIG5 visualizes the raw and filtered features (by RNM filter) of Cora projected by t-SNE (Van der BID30 .

The filtered features exhibit a much more compact cluster structure, thus making classification much easier.

In Appendix C, we show that feature filtering improves the accuracy of various classifiers significantly.

Compared with GCN and other baselines which use both graph and feature information, the performance gains of GLP come in two folds.

First, GLP allows using stronger filters to extract higher level data representations to improve performance when label rate is low, which can be easily achieved by increasing the filter parameters k and α, as shown in FIG5 .

But this cannot be easily achieved in CNN) 94.1 (5.1s) 95.3 (6.7s) 95.6 (8.9s) GLP (AR, CNN) 94.1 (7.2s) 95.5 (8.8s) 95.8 (11.1s) GCN.

As each convolutional layer of GCN is coupled with a projection layer, to increase smoothness one needs to stack many layers, and a deep GCN is difficult to train.

Second, GLP allows adopting domain-specific classifiers such as CNN to deal with vision tasks.

As shown in TAB2 , the performance of CNN trained on raw features of labeled data is very competitive and grows fast.

Due to space limitations, we include the stability analysis of GLP in Appendix D.

Many graph-based semi-supervised learning methods adopt a common assumption that nearby vertices are likely to have same labels.

One idea is to learn smooth low-dimensional embedding of data points by using Markov random walks BID25 , Laplacian eigenmaps BID0 , spectral kernels BID7 BID35 , and context-based methods BID21 .

Another idea hinges on graph partition, where the cuts should agree with the labeled data and be placed in low density regions BID3 BID39 BID13 BID4 .

Perhaps the most popular idea is to formulate a quadratic regularization framework to explicitly enforce the consistency with the labeled data and the cluster assumption, which is known as label propagation BID36 BID6 BID2 BID17 .To leverage more data information to improve predictions, a variety of methods proposed to jointly model data feature and graph information.

BID36 proposed to combine label propagation with external classifiers by attaching a "dongle" vertex to each unlabeled vertex.

Iterative classification algorithm BID23 iteratively classifies an unlabeled vertex using its neighbors' labels and features.

Manifold regularization BID1 , deep semi-supervised embedding BID32 , and Planetoid BID34 regularize a supervised classifier with a Laplacian regularizer or an embedding-based regularizer.

Graph convolutional networks BID16 combine graph and feature information in convolutional layers, which is actually doing Laplacian smoothing on data features .

Follow-up works include graph attention networks BID31 , attention-based graph neural network BID28 , and graph partition neural networks BID20 .The idea of feature smoothing has been widely used in computer graphics community for fairing 3D surface BID27 a; BID9 .

BID12 proposed manifold denoising which uses feature smoothing as a preprocessing step for running a label propagation algorithm, i.e, the denoised features are used to construct a better graph for LP.

This method is still "onedimensional", as it cannot use the preexisting graph information in data such as citation networks.

In contrast, the proposed GLP and the GCN frameworks are "two-dimensional".

In this paper, we have proposed a simple, flexible, and efficient framework GLP for semi-supervised learning, and demonstrated its effectiveness theoretically and empirically.

GLP offers new insights into existing methods and opens up possible avenues for new methods.

An important direction for future research is the design and selection of graph filters for GLP in different application scenarios.

Other directions include making GLP readily applicable to inductive problems, developing faster algorithms for GLP, and applying GLP to solve large-scale real-world problems.

We include dataset descriptions, experimental details, supplementary experiments, stability analysis, and running time analysis here.

Citation networks BID23 are networks that record documents' citation relationship.

In citation networks, vertices are documents and edges are citation links.

A pair of vertices are connected by an undirected edge if and only if one cites another.

Each vertex is associated with a feature vector, which encodes the document content.

In the three citation networks we tested on, CiteSeer, Cora and PubMed, feature vectors are 0/1 vectors that have the same length as the dictionary size and indicate whether a word appears in a document.

The statistics of datasets are summarized in TAB3 .Never Ending Language Learning (NELL) BID5 ) is a knowledge graph introduced by Carlson et al..

Yang et al. extracted an entity classification dataset from NELL, and converted the knowledge graph into a single relation graph.

For each relation type r, they created two new vertices r 1 and r 2 in the graph.

For each triplet (e 1 , r, e 2 ), they created two edges (e 1 , r 1 ) and (e 2 , r 2 ).

We follow BID16 to extend the features by assigning a unique one-hot representation for every relation vertex, resulting in a 61,278-dimensional sparse feature vector for each vertex.

Dataset statistics are also provided in TAB3 .MNIST contains 70,000 images of handwritten digits from 0 to 9 of size 28 × 28.

Each image is represented by a dense 784-dimensional vector where each dimension is a gray intensity pixel value.

A 5-NN graph is constructed based on the Euclidean distance between images.

If the i-th image is within the j-th image's 5 nearest neighbors or vice versa, then w ij = w ji = 1, otherwise w ij = w ji = 0.

We provide more experimental details here for the sake of reproduction.

Parameters We set k = 10 for RNM, k = 20 for RW, and α = 20 for AR, if label rate is less or equal than 2%; otherwise, we set them to 5, 10, 10 respectively.

Networks On citation networks, we follow BID16 to use a two-layer MLP with 16 hidden units for citation networks, 0.01 learning rate, 0.5 dropout rate, and 5 × 10 −4 L2 regularization.

On NELL, we also follow BID16 to use 64 hidden units, 10 −5 L2 regularization, 0.1 dropout rate and two layer-structure.

On MNIST, we use 256 hidden units, 0.01 learning rate, 0.5 dropout rate, and 5 × 10 −4 L2 regularization.

The CNN we use consists of six layers, whose structure is specified in TAB4 .

For CNN, we use 0.003 learning rate and 0.5 dropout.

All results of MNIST are averaged over 10 runs.

We train all networks using Adam BID14 .Baselines Results of some baselines including ManiReg, SemiEmb, DeepWalk, ICA, Planetoid are taken from BID16 , except for the 4-labels-per-class setting, for which we run BID34 .

All other results are reported by us.

To demonstrate the benefit of GLP, we compare training various supervised classifiers with raw and filtered features.

The classifiers include support vector machine (SVM), decision tree (DT), logistic regression (LR), and multilayer perceptron (MLP).

The results are summarized in TAB5 .

We can see that for all classifiers and on all datasets, there is a huge improvement in classification accuracy with the smooth features produced by the three filters we proposed.

This clearly demonstrates the advantage of filtered features over raw features.

In this experiment, we use 0.01 learning rate and 5 × 10 −4 L 2 regularization for LR.

For SVM, we use the RBF kernel with γ = 1/n and 1.0 L 2 regularization.

For DT, we use Gini impurity as quality measure.

We use the same parameters for MLP as described in Appendix B.

We test how the filter parameters k and α influence the performance of GLP.

Figs. 4 to 6 plot the classification accuracies of GLP with different k and α on three citation networks, with 4 labels per class.

Consistent with our analysis in section 5, the classification accuracy of GLP first increases and then decreases as k and α increases.

The results shows that GLP consistently outperforms GCN for a wide range of k and α.

@highlight

We extend the classical label propation methods to jointly model graph and feature information from a graph filtering perspective, and show connections to the graph convlutional networks.
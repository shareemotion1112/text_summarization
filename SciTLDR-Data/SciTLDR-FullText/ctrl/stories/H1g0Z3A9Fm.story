Community detection in graphs can be solved via spectral methods or posterior inference under certain probabilistic graphical models.

Focusing on random graph families such as the stochastic block model, recent research has unified both approaches and identified both statistical and computational detection thresholds in terms of the signal-to-noise ratio.

By recasting community detection as a node-wise classification problem on graphs, we can also study it from a learning perspective.

We present a novel family of Graph Neural Networks (GNNs) for solving community detection problems in a supervised learning setting.

We show that, in a data-driven manner and without access to the underlying generative models, they can match or even surpass the performance of the belief propagation algorithm on binary and multiclass stochastic block models, which is believed to reach the computational threshold in these cases.

In particular, we propose to augment GNNs with the non-backtracking operator defined on the line graph of edge adjacencies.

The GNNs are achieved good performance on real-world datasets.

In addition, we perform the first analysis of the optimization landscape of using (linear) GNNs to solve community detection problems, demonstrating that under certain simplifications and assumptions, the loss value at any local minimum is close to the loss value at the global minimum/minima.

Graph inference problems encompass a large class of tasks and domains, from posterior inference in probabilistic graphical models to community detection and ranking in generic networks, image segmentation or compressed sensing on non-Euclidean domains.

They are motivated both by practical applications, such as in the case of PageRank, and also by fundamental questions on the algorithmic hardness of solving such tasks.

From a data-driven perspective, these problems can be formulated in unsupervised, semi-supervised or supervised learning settings.

In the supervised case, one assumes a dataset of graphs with labels on their nodes, edges or the entire graphs, and attempts to perform node-wise, edge-wise and graph-wise classification by optimizing a loss over a certain parametric class, e.g. neural networks.

Graph Neural Networks (GNNs) are natural extensions of Convolutional Neural Networks to graph-structured data, and have emerged as a powerful class of algorithms to perform complex graph inference leveraging labeled data (Gori et al., 2005; BID3 (and references therein).

In essence, these neural networks learn cascaded linear combinations of intrinsic graph operators interleaved with node-wise (or edge-wise) activation functions.

Since they utilize intrinsic graph operators, they can be applied to varying input graphs, and they offer the same parameter sharing advantages as their CNN counterparts.

In this work, we focus on community detection problems, a wide class of node classification tasks that attempt to discover a clustered, segmented structure within a graph.

The algorithmic approaches to this problem include a rich class of spectral methods, which take advantage of the spectrum of certain operators defined on the graph, as well as approximate message-passing methods such as belief propagation (BP), which performs approximate posterior inference under predefined graphical models (Decelle et al., 2011) .

Focusing on the supervised setting, we study the ability of GNNs to approximate, generalize or even improve upon these class of algorithms.

Our motivation is two-fold.

On the one hand, this problem exhibits algorithmic hardness on some settings, opening up the possibility to discover more efficient algorithms than the current ones.

On the other hand, many practical scenarios fall beyond pre-specified probabilistic models, requiring data-driven solutions.

We propose modifications to the GNN architecture, which allow it to exploit edge adjacency information, by incorporating the non-backtracking operator of the graph.

This operator is defined over the edges of the graph and allows a directed flow of information even when the original graph is undirected.

It was introduced to community detection problems by Krzakala et al. (2013) , who propose a spectral method based on the non-backtracking operator.

We refer to the resulting GNN model as a Line Graph Neural Network (LGNN).

Focusing on important random graph families exhibiting community structure, such as the stochastic block model (SBM) and the geometric block model (GBM), we demonstrate improvements in the performance by our GNN and LGNN models compared to other methods, including BP, even in regimes within the so-called computational-to-statistical gap.

A perhaps surprising aspect is that these gains can be obtained even with linear LGNNs, which become parametric versions of power iteration algorithms.

We want to mention that besides community detection tasks, GNN and LGNN can be applied to other node-wise classification problems too.

The reason we are focusing on community detection problems is that this is a relatively well-studied setup, for which different algorithms have been proposed and where computational and statistical thresholds have been studied in several scenarios.

Moreover, synthetic datasets can be easily generated for community detection tasks.

Therefore, we think it is a nice setup for comparing different algorithms, besides its practical values.

The good performances of GNN and LGNN motivate our second main contribution: the analysis of the optimization landscape of simplified and linear GNN models when trained with planted solutions of a given graph distribution.

Under reparametrization, we provide an upper bound on the energy gap controlling the energy difference between local and global minima (or minimum).

With some assumptions on the spectral concentration of certain random matrices, this energy gap will shrink as the size of the input graphs increases, which would mean that the optimization landscape is benign on large enough graphs.

• We propose an extension of GNNs that operate on the line graph using the non-backtracking operator, which yields improvements on hard community detection regimes.• We show that on the stochastic block model we reach detection thresholds in a purely data-driven fashion, in the sense that our results improve upon belief propagation in hard SBM detection regimes, as well as in the geometric block model.• We perform the first analysis of the learning landscape of GNN models, showing that under certain simplifications and assumptions, they exhibit a form of "energy gap", where local mimima are confined in low-energy configurations.• We show that our model can perform well on community detection problems with real-world datasets.

We are interested in a specific class of node-classification tasks in which given an input graph G = (V, E), a labeling y : V → {1, . . .

, C} that encodes a partition of V into C communities is to be predicted at each node.

We assume that a training set {(G t , y t )} t≤T is given, with which we train a model that predictsŷ = Φ(G, θ) by minimizing DISPLAYFORM0 Since y encodes a partition of C groups, the specific label of each node is only important up to a global permutation of {1, . . .

, C}. Section 4.3 describes how to construct loss functions with such a property.

A permutation of the observed nodes translates into the same permutation applied to the labels, which justifies models Φ that are equivariant to permutations.

Also, we are interested in inferring properties of community detection algorithms that do not depend on the specific size of the graphs 1 .

We therefore require that the model Φ accepts graphs of variable size for the same set of parameters, similar to sequential RNN or spatial CNN models.

GNN was first proposed in Gori et al. (2005); Scarselli et al. (2009 ).

Bruna et al. (2013 generalize convolutional neural networks on general undirected graphs by using the graph Laplacian's eigenbasis.

This was the first time the Laplacian operator was used in a neural network architecture to perform classification on graph inputs.

Defferrard et al. (2016) consider a symmetric Laplacian generator to define a multiscale GNN architecture, demonstrated on classification tasks.

Similarly, Kipf & Welling (2016) use a similar generator as effective embedding mechanisms for graph signals and applies it to semi-supervised tasks.

This is the closest application of GNNs to our current contribution.

However, we highlight that semi-supervised learning requires bootstrapping the estimation with a subset of labeled nodes, and is mainly interested in generalization within a single, fixed graph.

In comparison, our setup considers community detection across a distribution of input graphs and assumes no initial labeling on the graphs in the test dataset except for the adjacency information.

There have been several extensions of GNNs by modifying their non-linear activation functions, parameter sharing strategies, and choice of graph operators (Li et al., 2015; Sukhbaatar et al., 2016; Duvenaud et al., 2015; Niepert et al., 2016) .

In particular, Gilmer et al. (2017) interpret the GNN architecture as learning an approximate message-passing algorithm, which extends the learning of hidden representations to graph edges in addition to graph nodes.

Recently, Velickovic et al. (2017) relate adjacency learning with attention mechanisms, and Vaswani et al. (2017) propose a similar architecture in the context of machine translation.

Another recent and related piece of work is by Kondor et al. (2018) , who propose a generalization of GNN that captures high-order node interactions through covariant tensor algebra.

Our approach to extend the expressive power of GNN using the line graph may be seen as an alternative to capture such high-order interactions.

Our energy landscape analysis is related to the recent paper by Shamir (2018), which establishes an energy bound on the local minima arising in the optimization of ResNets.

In our case, we exploit the properties of the community detection problem to produce an energy bound that depends on the concentration of certain random matrices, which one may hope for as the size of the input graphs increases.

Finally, Zhang (2016)'s work on data regularization for clustering and rank estimation is also motivated by the success of using Bethe-Hessian-like perturbations to improve spectral methods on sparse networks.

It finds good perturbations via matrix perturbations and also has successes on the stochastic block model.

Yang & Leskovec (2012a) curate benchmark datasets for community detection and quantify the quality of these datasets, while Yang & Leskovec (2012b) develop new algorithms for community detection by fitting to the networks the Affliation Graph Model (AGM), a generative model for graphs with overlapping communities.

This section introduces our GNN architectures that include the power graph adjacency (Section 4.1) and its extension to line graphs using the non-backtracking operator (Section 4.2), as well as the design of losses invariant to global label permutations (Section 4.3).

The Graph Neural Network (GNN), introduced in Scarselli et al. (2009) and later simplified in Li et al. (2015) ; Duvenaud et al. (2015) ; Sukhbaatar et al. (2016) , is a flexible neural network architecture based on local operators on a graph G = (V, E).

Given a state vector x ∈ R |V |×b on the vertices of Figure 1 .

Overview of the architecture of LGNN (Section 4.2).

Given a graph G, we construct its line graph L(G) with the non-backtracking operator FIG0 ).

In every layer, the states of all nodes in G and L(G) are updated according to (2).

The final states of nodes in G are used to predict node-wise labels, and the trainining is performed end-to-end using standard backpropagation with a label permutation invariant loss (Section 4.3).G, we consider intrinsic linear operators of the graph that act locally on x, which can be represented as |V |-by-|V | matrices.

For example, the adjacency matrix A is defined entry-wise by DISPLAYFORM0 , D is a diagonal matrix with D ii being the number of edges that the ith node has.

We can also define power graph adjacency matrices as A (j) = min(1, A 2 j ), which encodes 2 j -hop neighborhoods into a binary graph.

Finally, there is also the identity matrix, I. Given such a family of operators for each graph, DISPLAYFORM1 where DISPLAYFORM2 are trainable parameters and ρ(·) is a point-wise nonlinear activation function, chosen in this work to be the ReLU function, i.e. ρ(z) = max(0, z) for z ∈ R. Then we define DISPLAYFORM3 as the concatenation of z (k+1) and z (k+1) .

The layer thus includes linear "residual connections" (He et al., 2016) via z (k) , both to ease with the optimization when using large number of layers and to increase the expressivity of the model by enabling it to perform power iterations.

Since the spectral radius of the learned linear operators in (1) can grow as the optimization progresses, the cascade of GNN layers can become unstable to training.

In order to mitigate this effect, we consider spatial batch normalization (Ioffe & Szegedy, 2015) at each layer.

2 In our experiments, the initial states are set to be the degrees of the nodes, i.e., DISPLAYFORM4 satisfies the permutation equivariance property required for community detection: Given a permutation π among the nodes in the graph, Φ(G π , Πx (0) ) = ΠΦ(G, x (0) ), where Π is the permutation matrix associated with π.

Analogy between GNN and power iterations In our setup, spatial batch normalization not only prevents gradient blowup, but also performs the orthogonalisation relative to the constant vector, which reinforces the analogy with the spectral methods for community detection, some background of which is described in B.1.

In essence, in certain regimes, the eigenvector of A corresponding to its second largest eigenvalue and the eigenvector of the Laplacian matrix, L = D − A, corresponding to its second smallest eigenvalue (i.e. the Fiedler vector), are both correlated with the community structure of the graph.

Thus, spectral methods for community detection performs power iterations on these matrices to obtain the eigenvectors of interest and predicts the community structure based on them.

For example, to extract the Fiedler vector of a matrix M , whose eigenvector corresponding to the smallest eigenvalue is known to be v, one performing power iterations onM DISPLAYFORM5 If v is a constant vector, which is the case for L, then the normalization above is precisely performed within the spatial batch normalization step.

By incorporating a family of operators into the neural network framework, the GNN can not only approximate but also go beyond power iterations.

As explained in Section B.1, the Krylov subspace generated by the graph Laplacian (Defferrard et al., 2016) is not sufficient to operate well in the sparse regime, as opposed to the space generated by {I, D, A}. The expressive power of each layer is further increased by adding multiscale versions of A, although this benefit comes at the cost of computational efficiency, especially in the sparse regime.

The network depth is chosen to be of the order of the graph diameter, so that all nodes obtain information from the entire graph.

In sparse graphs with small diameter, this architecture offers excellent scalability and computational complexity.

Indeed, in many social networks diameters are constant (due to hubs), or log(|V |), as in the stochastic block model in the constant average degree regime (Riordan & Wormald, 2010) .

This results in a model with computational complexity on the order of |V | log(|V |), making it amenable to large-scale graphs.

For graphs with few cycles, posterior inference can be remarkably approximated by loopy belief propagation (Yedidia et al., 2003) .

As described in Section B.2, the message-passing rules are defined over the edge adjacency graph (see equation 57).

Although its second-order approximation around the critical point can be efficiently approximated with a power method over the original graph, a datadriven version of BP requires accounting for the non-backtracking structure of the message-passing.

In this section we describe an upgraded GNN model that exploits the non-backtracking structure.

and so |V L | = 2|E|.

The non-backtracking operator on the line graph is represented by a matrix B ∈ R 2|E|×2|E| defined as DISPLAYFORM0 This operator enables the directed propagation of information through on the line graph and was first proposed in the context of community detection on sparse graphs in Krzakala et al. (2013) .

The message-passing rules of BP can be expressed as a diffusion in the line graph L(G) using this non-backtracking operator, with specific choices of activation function that turn product of beliefs into sums.

A natural extension of the GNN architecture presented in Section 4.1 is thus to consider a second GNN defined on L(G), where B and D B = diag(B1) play the role of the adjacency and the degree matrices, respectively.

This effectively defines edge features that are updated according to the edge adjacency of G. Edge and node features communicate at each layer using the edge indicator matrices P m , P d ∈ {0, 1} |V |×2|E| , defined as P mi,(i→j) = 1, P dj,(i→j) = 1, P di,(i→j) = 1, P dj,(i→j) = −1 and 0 otherwise.

With the skip linear connections defined similarly, the resulting model becomes DISPLAYFORM1 where DISPLAYFORM2 and the trainable parameters are θ i , θ i , θ i ∈ R b k ×b k+1 and θ i ∈ R b k+1 ×b k+1 .

We call such a model a Line Graph Neural Network (LGNN).In our experiments, we set x (0) = deg(A) and y (0) = deg(B).

For graph families with constant average degree d (as |V | grows), the line graph has size 2|E| ∼ O(d|V |), and is therefore feasible from the computational point of view.

Furthermore, the construction of line graphs can be iterated to generate L(L(G)), L(L(L(G))), etc.

to yield a graph hierarchy, which could capture high-order interactions among nodes of G. Such an hierarchical construction is related to other recent efforts to generalize GNNs (Kondor et al., 2018) .Relationship between LGNN and edge feature learning approaches Several authors have proposed combining node and edge feature learning.

BID2 introduce edge features over directed and typed graphs, but does not discuss the undirected case.

Kearnes et al. FORMULA2 ; Gilmer et al. (2017) learn edge features on undirected graphs using f e = g(x(i), x(j)) for an edge e = (i, j), where g is commutative on its arguments.

Finally, Velickovic et al. FORMULA2 learns directed edge features on undirected graphs using stochastic matrices as adjacencies (which are either row or column-normalized).

However, we are not aware of works that consider the edge adjacency structure provided by the non-backtracking matrix on the line graph.

With non-backtracking matrix, our LGNN can be interpreted as learning directed edge features from an undirected graph.

Indeed, if each node i contains two distinct sets of features x s (i) and x r (i), the non-backtracking operator constructs edge features from node features while preserving orientation: For an edge e = (i, j), our model is equivalent to constructing oriented edge features f i→j = g(x s (i), x r (j)) and f j→i = g(x r (i), x s (j)) (where g is trainable and not necessarily commutative on its arguments) that are subsequently propagated through the graph.

Constructing such local oriented structure is shown to be important for improving performance in the next section.

For comparison, we also define a linear LGNN (LGNN-L) as the the LGNN that drops the nonlinear activation functions ρ in (2), and a symmetric LGNN (LGNN-S) as the LGNN whose line graph is defined on the undirected edges of the original graph: In LGNN-S, two edges of G are connected in the line graph if and only if they share one common node; also, F = {P }, with P ∈ R |V |×|E| defined as P i,(j→k) = 1 if i = j or k and 0 otherwise.

Let C = {1, . . .

, C} denote the set of all community labels, and consider first the case where communities do not overlap.

By applying the softmax function at the end, we interpret the cth dimension of the output of the models at node i as the conditional probability that the node belongs to community c, o i,c = p(y i = c |θ, G).

Let G = (V, E) be the input graph and let y i be the ground truth community label of node i. Since the community structure is defined up to global permutations of the labels, we can define a loss function with respect to a given graph instance as DISPLAYFORM0 where S C denotes the permutation group of C elements.

This is essentially taking the the cross entropy loss minimized over all possible permutations of C. In our experiments, we considered examples with small numbers of communities such as 2 and 5.

In general scenarios where C is much larger, the evaluation of the loss function (3) can be impractical due to the minimization over S C .

A possible solution is to randomly partition C labels intoC groups, and then to marginalize the model outputs DISPLAYFORM1 ,c ∈C, and and finally use (θ) = inf π∈SC − i∈V logō i,π(ȳi) as an approximate loss value, which only involves a permutation group of size (C!).Finally, if communities may overlap, we can enlarge C to include subsets of communities and define the permutation group accordingly.

For example, if there are two overlapping communities, we let C = {{1}, {2}, {1, 2}}, and only allow the permutation between 1 and 2 when computing the loss function as well as the overlap to be introduced in Section 6.

As described in the numerical experiments, we found that the GNN models without nonlinear activations already provide substantial gains relative to baseline (non-trainable) algorithms.

This section studies the optimization landscape of linear GNNs.

Despite defining a non-convex objective, we prove that the landscape is "benign" under certain further simplifications, in the sense that the local minima are confined in sublevel sets of low energy.

For simplicity, we consider only the binary c = 2 case where we replace the node-wise binary cross-entropy loss by the squared cosine distance 3 , assume a single feature map (b k = 1 for all k), and focus on the GNN described in Section 4.1 (although our analysis carries equally to describe the line graph version; see remarks below).

We also make the simplifying assumption to replace the layer-wise spatial batch normalization by a simpler projection onto the unit 2 ball (thus we do not remove the mean).

Without loss of generality, assume that the input graph G has size n, and denote by F = {A 1 , . . .

, A Q } the family of graph operators appearing in (1).

Each layer thus applies an arbitrary polynomial DISPLAYFORM0 q A q to the incoming node feature vector x (k) .

Given an input node vector w ∈ R n , the network output can thus be written aŝ DISPLAYFORM1 We highlight that this linear GNN setup is fundamentally different from the linear fully-connected neural networks (that is, neural networks with linear activation function), whose landscape has been analyzed in Kawaguchi (2016) .

First, the output of the GNN is on the unit sphere, which has a different geometry.

Next, since the operators in F depend on the input graph, they introduce fluctuations in the landscape.

In general, the operators in F are not commutative, but by considering the generalized Krylov subspace generated by powers of F, DISPLAYFORM2 Given the target y ∈ R n , the loss incurred by each pair (G, y) becomes 1 − | e,y | 2 e 2 , and therefore the population loss, when expressed in terms of β, equals DISPLAYFORM3 The landscape is thus specified by a pair of random matrices Y n , X n ∈ R M ×M .Assuming that EX n 0, we write the Cholesky decomposition of EX n as EX n = R n R T n , and define DISPLAYFORM4 denote the eigenvalues of K in nondecreasing order.

Then, the following theorem establishes that under appropriate assumptions, the concentration of relevant random matrices around their mean controls the energy gaps between local and global minima of L. DISPLAYFORM5 , and assume that all four quantities are finite.

Then if DISPLAYFORM6 , where ηn,µn,νn,δn = O(δ n ) for given η n , µ n , ν n as δ n → 0 and its formula is given in the appendix.

Corollary 5.2.

If (η n )

n∈N * , (µ n ) n∈N * , (ν n ) n∈N * are all bounded sequences, and lim n→∞ δ n = 0, DISPLAYFORM7 The main strategy of the proof is to consider the actual loss function L n as a perturbation of DISPLAYFORM8 , which has a landscape that is easier to analyze and 3 to account for the invariance up to global flip of label does not have poor local minima, since it is equivalent to a quadratic form defined over the sphere S M −1 .

Applying this theorem requires estimating spectral fluctuations of the pair X n , Y n , which in turn involve the spectrum of the C * algebras generated by the non-commutative family F. For example, for stochastic block models, it is an open problem how the bound behaves as a function of the parameters p and q. Another interesting question is to understand how the asymptotics of our landscape analysis relate to the hardness of estimation as a function of the signal-to-noise ratio.

Finally, another open question is to what extent our result could be extended to the non-linear residual GNN case, perhaps leveraging ideas from Shamir (2018).

We present experiments on community detection in synthetic datasets (Sections 6.1, 6.2 and Appendix C.1) as well as real-world datasets (Section 6.3).

In the synthetic experiments, the performance is measured by the overlap between predicted (ŷ) and true labels (y), which quantifies how much better than random guessing a predicted labeling is, given by DISPLAYFORM0 , where δ is the Kronecker delta function, and this quantity is maximized over global permutations within a graph of the set of labels.

In the real-world datasets, as the communies are overlapping and not balanced, the prediction accuracy is measured by 1 n u δ y(u),ŷ(u) , and the set of permutations to be maximized over is described in Section 4.3.

We used Adamax (Kingma & Ba, 2014) with learning rate 0.004 across all experiments.

All the neural network models have 30 layers and 8 features in the middle layers (i.e., b k = 8) for experiments in Sections 6.1 and 6.2, and 20 layers and 6 features for Section 6.3.

GNNs and LGNNs have J = 2 across the experiments except the ablation experiments in Section C.3.

The stochastic block model is a random graph model with planted community structure.

In its simplest form, the graph consists of |V | = n nodes, which are partitioned into C communities, that is, each node is assigned a label y ∈ {1, ..., C}. An edge connecting any two vertices u, v is drawn independently at random with probability p if y(v) = y(u), and with probability q otherwise.

In the binary case (i.e. C = 2), the sparse regime, where p, q 1/n, is well understood and provides an initial platform to compare the GNN and LGNN with provably optimal recovery algorithms (Appendix B).

We consider two learning scenarios.

In the first scenario, we choose different pairs of p and q, and train the models for each pair separately.

In particular, for each pair of (p i , q i ), we sample 6000 graphs under G ∼ SBM (n = 1000, p i , q i , C = 2) and then train the models for each i. In the second scenario, reported in Appendix C.2, we train a single set of parameters θ from a set of 6000 graphs sampled from a mixture of SBM with different pairs of (p i , q i ), and average degree.

Importantly, his setup shows that our models are not simply approximating known algorithms such as BP for particular SBM parameters, since the parameters vary in this dataset.

For the first scenario, we chose five different pairs of (p i , q i ) while fixing p i + q i , thereby corresponding to different signal-to-noise ratios (SNRs).

FIG2 reports the performance of our models on the binary SBM model for the different SNRs, compared with baseline methods including BP, spectral methods using the normalized Laplacian and the Bethe Hessian as well as Graph Attention Networks (GAT) 5 from Velickovic et al. (2017) .

We observe that both GNN and LGNN reach the performance of BP.

In addition, even the linear LGNN achieves a performance that is quite close to that of BP, in accordance to the spectral approximations of BP given by the Bethe Hessian (see supplementary), and significantly outperforms performing 30 power iterations on the Bethe Hessian or the normalized Laplacian, as was done in the spectral methods.

We also notice that our models outperform GAT in this task.

We ran experiments in the dissociative case (q > p), as well as with C = 3 communities and obtained similar results, which are not reported here.

Table 1 : Performance of different models on 5-community dissociative SBM graphs with n = 400, C = 5, p = 0, q = 18/n, corresponding to an average degree of 14.5.

The first row gives the average overlap across test graphs, and the second row gives the graph-wise standard deviation of the overlap.

In SBM with fewer than 4 communities, it is known that BP provably reaches the information-theoretic threshold BID0 Massoulié, 2014; Coja-Oghlan et al., 2016) .

The situation is different for k > 4, where it is conjectured that a gap emerges between the theoretical performance of MLE estimators and the performance of any polynomial-time estimation procedure (Decelle et al., 2011) .

In this context, one can use the GNN models to search the space of the generalizations of BP, and attempt to improve upon the detection performance of BP for scenarios where the SNR falls within the computational-to-statistical gap.

Table 1 presents results for the 5-community dissociative SBM, with n = 400, p = 0 and q = 18/n.

The SNR in this setup is above the information-theoretic threshold but below the asymptotic threshold above which BP is able to detect (Decelle et al., 2011) .

Note that since p = 0, this also amounts to a graph coloring problem.

We see that the GNN and LGNN models outperform BP in this experiment, indeed opening up the possibility to reduce the computation-information gap.

That said, our model may taking advantage of finite-size effects, which will vanish as n → ∞. The asymptotic study of these gains is left for future work.

In terms of average test accuracy, LGNN has the best performance.

In particular, it outperforms the symmetric version of LGNN, emphasizing the importance of the non-backtracking matrix used in LGNN.

Although equipped with the attention mechanism, GAT does not explicitly incorporate in itself the degree matrix, the power graph adjacency matrices or the line graph structure, and has inferior performance compared with the GNN and LGNN models.

Further ablation studies on GNN and LGNN are described in Section C.3.

We now compare the models on the SNAP datasets, whose domains range from social networks to hierarchical co-purchasing networks.

We obtain the training set as follows.

For each SNAP dataset, we start by focusing only on the 5000 top quality communities provided by the dataset.

We then identify edges (i, j) that cross at least two different communities.

For each of such edges, we consider pairs of communities C 1 , C 2 such that i / ∈ C 2 and j / ∈ C 1 , i ∈ C 1 , j ∈ C 2 , and extract the subset of nodes determined by C 1 ∪ C 2 together with the edges among them.

The resulting graph is connected since each community is connected.

Finally, we divide the dataset into training and testing sets by enforcing that no community belongs to both the training and the testing set.

In our experiment, due to computational limitations, we restrict our attention to the three smallest datasets in the SNAP collection (Youtube, DBLP and Amazon), and we restrict the largest community size to 200 nodes, which is a conservative bound.

We compare the performance of GNN and LGNN models with GAT as well as the CommunityAffiliation Graph Model (AGM), which is a generative model proposed in Yang & Leskovec (2012b) that captures the overlapping structure of real-world networks.

Community detection can be achieved by fitting AGM to a given network, which was shown to outperform some state-of-the-art algorithms.

TAB1 compares the performance, measured with a 3-class (C = {{1}, {2}, {1, 2}}) classification accuracy up to global permutation 1 ↔ 2.

GNN, LGNN, LGNN-S and GAT yield similar results and outperform AGMfit.

It further illustrates the benefits of data-driven models that strike the right balance between expressivity and structural design.

In this work, we have studied data-driven approaches to supervised community detection with graph neural networks.

Our models achieve comparable performance to BP in binary SBM for various SNRs, and outperform BP in the sparse regime of 5-class SBM that falls between the computationalto-statistical gap.

This is made possible by considering a family of graph operators including the power graph adjacency matrices, and importantly by introducing the line graph equipped with the non-backtracking matrix.

We also provided a theoretical analysis of the optimization landscapes of simplified linear GNN for community detection and showed the gap between the loss value at local and global minima are bounded by quantities related to the concentration of certain random matricies.

One word of caution is that our empirical results are inherently non-asymptotic.

Whereas models trained for given graph sizes can be used for inference on arbitrarily sized graphs (owing to the parameter sharing of GNNs), further work is needed in order to understand the generalization properties as |V | increases.

Nevertheless, we believe our work opens up interesting questions, namely better understanding how our results on the energy landscape depend upon specific signal-to-noise ratios, or whether the network parameters can be interpreted mathematically.

This could be useful in the study of computational-to-statistical gaps, where our model can be used to inquire about the form of computationally tractable approximations.

Another current limitation of our model is that it presumes a fixed number of communities to be detected.

Other directions of future research include the extension to the case where the number of communities is unknown and varied, or even increasing with |V |, as well as applications to ranking and edge-cut problems.

A PROOF OF THEOREM 5.1For simplicity and with an abuse of notation, in the remaining part we redefine L andL in the following way, to be the negative of their original definition in the main section: DISPLAYFORM0 .

Thus, minimizing the loss function (5) is equivalent to maximizing the function L n (β) redefined here.

We write the Cholesky decomposition of EX n as EX n = R n R T n , and define DISPLAYFORM1 n ) T , and ∆B n = B n − I n .

Given a symmetric matrix K ∈ R M ×M , we let λ 1 (K), λ 2 (K), ..., λ M (K) denote the eigenvalues of K in nondecreasing order.

Let us denote byβ g a global minimum of the mean-field lossL n .

Taking a step further, we can extend this bound to the following one (the difference is in the second term on the right hand side): DISPLAYFORM0 Proof of Lemma A.1.

We consider two separate cases: The first case is whenL n (β l ) ≥L n (β g ).

DISPLAYFORM1 The other case is whenL DISPLAYFORM2 Hence, to bound the "energy gap" |L n (β l ) − L n (β g )|, if suffices to bound the three terms on the right hand side of Lemma A.1 separately.

First, we consider the second term, DISPLAYFORM3 Thus, we apply a change-of-variable and try to bound DISPLAYFORM4 n , where R n is invertible, we know that λ 1 (∇ 2 S n (γ l )) ≤ 0, thanks to the following lemma: DISPLAYFORM5 Next, we relate the left hand side of the inequality above to cos(γ l ,γ g ), thereby obtaining an upper bound on [1 − cos 2 (γ l ,γ g )], which will then be used to bound |S n (γ l ) −S n (γ g )|.

DISPLAYFORM6 Thus, if we define DISPLAYFORM7 To bound λ 1 (∇ 2S n (γ)), we bound λ 1 (Q 1 ) and Q 2 as follows:SinceĀ n is symmetric, letγ 1 , . . .γ M be the orthonormal eigenvectors ofĀ n corresponding to nonincreasing eigenvalues l 1 , . . .

l M .

Note that the global minimum satisfiesγ g = ±γ 1 .

Write DISPLAYFORM8 Then, DISPLAYFORM9 To bound Q 2 : DISPLAYFORM10 Therefore, DISPLAYFORM11 Thus, DISPLAYFORM12 This yields the desired lemma.

Combining inequality 8 and Lemma A.3, we get DISPLAYFORM13 Thus, to bound the angle between γ l andγ g , we can aim to bound ∇S n (γ l ) and ∇ 2 S n (γ l ) − ∇ 2S n (γ l ) as functions of the quantities µ n , ν n and δ n .

Lemma A.4.

DISPLAYFORM14 Proof of Lemma A.4.

DISPLAYFORM15 Combining equations 17 and 18, we get DISPLAYFORM16 Then, by the generalized Hölder's inequality, DISPLAYFORM17 Hence, written in terms of the quantities µ n , ν n and δ n , we have DISPLAYFORM18 Lemma A.5.

With δ n = (E ∆B n 6 ) 1 6 , E|λ 1 (B n )| 6 ≤ 64 + 63δ 6 n Proof of Lemma A.5.

DISPLAYFORM19 Note that DISPLAYFORM20 and for k ∈ {1, 2, 3, 4, 5}, if X is a nonnegative random variable, DISPLAYFORM21 Therefore, E|λ 1 (B n )| 6 ≤ 64 + 63E ∆B n 6 .From now on, for simplicity, we introduce δ n = (64 + 63δ DISPLAYFORM22 Proof of Lemma A.6.

DISPLAYFORM23 where DISPLAYFORM24 DISPLAYFORM25 DISPLAYFORM26 H 4 , and we try to bound each term on the right hand side separately.

For the first term, there is DISPLAYFORM27 Applying generalized Hölder's inequality, we obtain DISPLAYFORM28 For the second term, there is DISPLAYFORM29 Hence, DISPLAYFORM30 Applying generalized Hölder's inequality, we obtain DISPLAYFORM31 For H 3 , note that DISPLAYFORM32 Hence, DISPLAYFORM33 Thus, DISPLAYFORM34 Applying generalized Hölder's inequality, we obtain DISPLAYFORM35 For the last term, DISPLAYFORM36 Thus, DISPLAYFORM37 Applying generalized Hölder's inequality, we obtain DISPLAYFORM38 Therefore, summing up the bounds above, we obtain DISPLAYFORM39 n (γ) ≤µ n ν n δ n (10 + 14ν n + 2δ n ν n + 16ν 2 n + 16δ n ν n + 8δ n ν 2 n + 8δ n ν n + 8δ n δ n ν) (44) Hence, combining inequality 15, Lemma A.4 and Lemma A.6, we get 1 − cos 2 (γ l ,γ g ) ≤η n [4µ n ν n δ n (1 + 3ν n δ n µ n ) + 1 2 µ n ν n δ n (10 + 14ν n + 2δ n ν n + 16ν 2 n + 16δ n ν n + 8δ n ν 2 n + 8δ n ν n + 8δ n δ n ν)] =µ n ν n δ n η n (9 + 19ν n + 5δ n ν n + 8ν 2 n + 8δ n ν n + 4δ n ν n 2 + 4δ n ν n + 4δ n δ n ν n )n +8δ n ν n +4δ n ν n 2 +4δ n ν n +4δ n δ n ν n .

Thus, DISPLAYFORM40 Following the notations in the proof of Lemma A.3, we write DISPLAYFORM41 Since Y n is positive semidefinite, EY n is also positive semidefinite, and henceĀ n = R DISPLAYFORM42 Next, we bound the first and the third term on the right hand side of the inequality in Lemma A.1.

Lemma A.7.

∀β, DISPLAYFORM43 Thus, we get the desired lemma by the generalized Hölder's inequality.

Combining inequality 46, inequality 48 and Lemma A.7, we get DISPLAYFORM44 Meanwhile, DISPLAYFORM45 Hence, DISPLAYFORM46 , or DISPLAYFORM47 Therefore, We consider graphs G = (V, E), modeling a system of N = |V | elements presumed to exhibit some form of community structure.

The adjacency matrix A associated with G is the N × N binary matrix such that A i,j = 1 when (i, j) ∈ E and 0 otherwise.

We assume for simplicity that the graphs are undirected, therefore having symmetric adjacency matrices.

The community structure is encoded in a discrete label vector s : V → {1, . . .

, C} that assigns a community label to each node, and the goal is to estimate s from observing the adjacency matrix.

DISPLAYFORM48 In the binary case, we can set s(i) = ±1 without loss of generality.

Furthermore, we assume that the communities are associative, which means two nodes from the same community are more likely to be connected than two nodes from the opposite communities.

The quantity DISPLAYFORM49 measures the cost associated with cutting the graph between the two communities encoded by s, and we wish to minimize it under appropriate constraints (Newman, 2006) .

Note that i,j A i,j = s T Ds, with D = diag(A1) (called the degree matrix), and so the cut cost can be expressed as a positive semidefinite quadratic form min DISPLAYFORM50 that we wish to minimize.

This shows a fundamental connection between the community structure and the spectrum of the graph Laplacian ∆ = D − A, which provides a powerful and stable relaxation of the discrete combinatorial optimization problem of estimating the community labels for each node.

The eigenvector of ∆ associated with the smallest eigenvalue is, trivially, 1, but its Fiedler vector (the eigenvector associated with the second smallest eigenvalue) reveals important community information of the graph under appropriate conditions (Newman, 2006) , and is associated with the graph conductance under certain normalization schemes (Spielman, 2015) .Given linear operator L(A) extracted from the graph (that we assume symmetric), we are thus interested in extracting eigenvectors at the edge of its spectrum.

A particularly simple algorithm is the power iteration method.

Indeed, the Fiedler vector of L(A) can be obtained by first extracting the leading eigenvector v ofÃ = L(A) I − L(A), and then iteratively compute DISPLAYFORM51 Unrolling power iterations and recasting the resulting model as a trainable neural network is akin to the LISTA sparse coding model, which unrolled iterative proximal splitting algorithms (Gregor & LeCun, 2010) .Despite the appeal of graph Laplacian spectral approaches, it is known that these methods fail in sparsely connected graphs (Krzakala et al., 2013) .

Indeed, in such scenarios, the eigenvectors of the graph Laplacian concentrate on nodes with dominant degrees, losing their correlation with the community structure.

In order to overcome this important limitation, people have resorted to ideas inspired from statistical physics, as explained next.

Graphs with labels on nodes and edges can be cast as a graphical model where the aim of clustering is to optimize label agreement.

This can be seen as a posterior inference task.

If we simply assume the graphical model is a Markov Random Field (MRF) with trivial compatibility functions for cliques greater than 2, the probability of a label configuration σ is given by DISPLAYFORM0 Generally, computing marginals of multivariate discrete distributions is exponentially hard.

For instance, in the case of P(σ i ) we are summing over |X| n−1 terms (where X is the state space of discrete variables).

But if the graph is a tree, we can factorize the MRF more efficiently to compute the marginals in linear time via a dynamic programming method called the sum-product algorithm, also known as belief propagation (BP).

An iteration of BP is given by DISPLAYFORM1 The beliefs (b i→j (σ i )) are interpreted as the marginal distributions of σ i .

Fixed points of BP can be used to recover marginals of the MRF above.

In the case of the tree, the correspondence is exact: DISPLAYFORM2 Certain sparse graphs, like SBM with constant degree, are locally similar to trees for such an approximation to be successful (Mossel et al., 2014) .

However, convergence is not guaranteed in graphs that are not trees.

Furthermore, in order to apply BP, we need a generative model and the correct parameters of the model.

If unknown, the parameters can be derived using expectation maximization, further adding complexity and instability to the method since it is possible to learn parameters for which BP does not converge.

The BP equations have a trivial fixed-point where every node takes equal probability in each group.

Linearizing the BP equation around this point is equivalent to spectral clustering using the nonbacktracking matrix (NB), a matrix defined on the directed edges of the graph that indicates whether two edges are adjacent and do not coincide.

Spectral clustering using NB gives significant improvements over spectral clustering with different versions of the Laplacian matrix L and the adjacency matrix A. High degree fluctuations drown out the signal of the informative eigenvalues in the case of A and L, whereas the eigenvalues of NB are confined to a disk in the complex plane except for the eigenvalues that correspond to the eigenvectors that are correlated with the community structure, which are therefore distinguishable from the rest.

NB matrices are still not optimal in that they are matrices on the edge set and also asymmetric, therefore unable to enjoy tools of numerical linear algebra for symmetric matrices.

Saade et al. (2014) showed that a spectral method can do as well as BP in the sparse SBM using the Bethe Hessian matrix defined by BH(r) := (r 2 − 1)I − rA + D, where r is a scalar parameter.

This is due to a one-to-one correspondence between the fixed points of BP and the stationary points of the Bethe free energy (corresponding Gibbs energy of the Bethe approximation) (Saade et al., 2014) .

The Bethe Hessian is a scaling of the Hessian of the Bethe free energy at an extrema corresponding to the trivial fixed point of BP.

Negative eigenvalues of BH(r) correspond to phase transitions in the Ising model where new clusters become identifiable.

The success of the spectral method using the Bethe Hessian gives a theoretical motivation for having a family of matrices including I, D and A in our GNN defined in Section 4, because in this way the GNN is capable of expressing the algorithm of performing power iteration on the Bethe Hessian.

While belief propagation requires a generative model, and the spectral method using the Bethe Hessian requires the selection of the parameter r, whose optimal value also depends on the underlying generative model, the GNN does not need a generative model and is able to learn and then make predictions in a data-driven fashion.

We briefly review the main properties needed in our analysis, and refer the interested reader to BID0 for an excellent recent review.

The stochastic block model (SBM) is a random graph model denoted by SBM (n, p, q, C).

Implicitly there is an F : V → {1, . . .

, C} associated with each SBM graph, which assigns community labels to each vertex.

One obtains a graph from this generative model by starting with n vertices and connecting any two vertices u, v independently at random with probability p if F (v) = F (u), and with probability q if F (v) = F (u).

We say the SBM is balanced if the communities are the same size.

LetF n : V → {1, C} be our predicted community labels for SBM (n, p, q, C).

We say that the F n 's give exact recovery on a sequence {SBM (n, p, q)} n if P(F n =F n ) → n 1, and give weak recovery or detection if ∃ > 0 such that P(|F n −F n | ≥ 1/k + ) → n 1 (i.eF n 's do better than random guessing).It is harder to tell communities apart if p is close to q (if p = q we just get an Erdős Renyi random graph, which has no communities).

In the two community case, It was shown that exact recovery is possible on SBM (n, p = a log n n , q = b log n n ) if and only if BID1 BID1 .

For exact recovery to be possible, p, q must grow at least O(log n) or else the sequence of graphs will not be connected, and thus the vertex labels will be underdetermined.

There is no information-computation gap in this regime, and so there exist polynomial time algorithms when recovery is possible BID0 Mossel et al., 2014) ).

In the sparser regime of constant degree, SBM (n, p = a n , q = b n ), detection is the best we could hope for.

The constant degree regime is also of most interest to us for real world applications, as most large datasets have bounded degree and are extremely sparse.

It is also a very challenging regime; spectral approaches using the Laplacian in its various (un)normalized forms or the adjacency matrix, as well as semidefinite programming (SDP) methods do not work well in this regime due to large fluctuations in the degree distribution that prevent eigenvectors from concentrating on the clusters BID0 .

Decelle et al. (2011) first proposed the BP algorithm on the SBM, which was proven to yield Bayesian optimal values in Coja-Oghlan et al. (2016) .

DISPLAYFORM0 In the constant degree regime with k balanced communities, the signal-to-noise ratio is defined as SN R = (a − b) 2 /(k(a + (k + 1)b)), and the Kesten-Stigum (KS) threshold is given by SN R = 1 BID0 .

When SN R > 1, detection can be solved in polynomial time by BP BID0 Decelle et al., 2011) .

For k = 2, it has been shown that when SN R < 1, detection is not solvable, and therefore SN R = 1 is both the computational and the information theoretic threshold BID0 .

For k > 4, it has been shown that for some SN R < 1, there exists non-polynomial time algorithms that are able to solve the detection problem BID0 .

Furthermore, it is conjectured that no polynomial time algorithm can solve detection when SN R < 1, in which case a gap would exist between the information theoretic threshold and the KS threshold BID0 .C FURTHER EXPERIMENTS C.1 GEOMETRIC BLOCK MODEL Table 3 : Overlap performance (in percentage) of GNN and LGNN on graphs generated by the Geometric Block Model compared with two spectral methods Model S = 1 S = 2 S = 4 Norm.

Laplacian 1 ± 0.5 1 ± 0.6 1 ± 1 Bethe Hessian 18 ± 1 38 ± 1 38 ± 2 GNN 20 ± 0.4 39 ± 0.5 39 ± 0.5 LGNN 22 ± 0.4 50 ± 0.5 76 ± 0.5The success of belief propagation on the SBM relies on its locally hyperbolic properties, which make it treelike with high probability.

This behavior is completely different if one considers random graphs with locally Euclidean geometry.

The Geometric Block Model (Sankararaman & Baccelli, 2018 ) is a random graph generated as follows.

We start by sampling n points x 1 , . . .

, x n i.i.d.

from a Gaussian mixture model given by means µ 1 , . . .

µ k ∈ R d at distances S apart and identity covariances.

The label of each sampled point corresponds to which Gaussian it belongs to.

We then draw an edge between two nodes i, j if x i − x j ≤ T / √ n. Due to the triangle inequality, the model contains a large number of short cycles, which affects the performance of loopy belief propagation.

This . left: k = 2.

We verify that BH(r) models cannot perform detection at both ends of the spectrum simultaneously.motivates other estimation algorithms based on motif-counting that require knowledge of the model likelihood function (Sankararaman & Baccelli, 2018) .

Table 3 shows the performance of GNN and LGNN on the binary GBM model, obtained with d = 2, n = 500, T = 5 √ 2 and varying S, as well as the performances of two spectral methods, using respectively the normalized Laplacian and the Bethe Hessian, which approximates BP around its stationary solution.

We note that LGNN model, thanks to its added flexibility and the multiscale nature of its generators, is able to significantly outperform both spectral methods as well as the baseline GNN.

We report here our experiments on the SBM mixture, generated with G ∼ SBM (n = 1000, p = kd − q, q ∼ Unif(0,d − d ), C = 2) , where the average degreed is either fixed constant or also randomized withd ∼ Unif(1, t).

FIG4 shows the overlap obtained by our model compared with several baselines.

Our GNN model is either competitive with BH or outperforms BH, which achieves the state of the art along with BP Saade et al. (2014) , despite not having any access to the underlying generative model (especially in cases where GNN was trained on a mixture of SBM and thus must be able to generalize the r parameter in BH).

They all outperform by a wide margin spectral clustering methods using the symmetric Laplacian and power method applied to BH I − BH using the same number of layers as our model.

Thus GNN's ability to predict labels goes beyond approximating spectral decomposition via learning the optimal r for BH(r).

The model architecture could allow it to learn a higher dimensional function of the optimal perturbation of the multiscale adjacency basis, as well as nonlinear power iterations, that amplify the informative signals in the spectrum.

Compared to f , each of h, i and k has one fewer operator in F, and j has two fewer.

We see that with the absence of A (2) , k has much worse performance than the other four, indicating the importance of the power graph adjacency matrices.

Interestingly, with the absence of I, i actually has better average accuracy than f .

One possibly explanation is that in SBM, each node has the same expected degree, and hence I may be not very far from D, which might make having both I and D in the family redundant to some extent.

Comparing GNN models a, b and c, we see it is not the case that having larger J will always lead to better performance.

Compared to f , GNN models c, d and e have similar numbers of parameters but all achieve worse average test accuracy, indicating that the line graph structure is essential for the good performance of LGNN in this experiment.

In addition, l also performs worse than f , indicating the significance of the non-backtracking line graph compared to the symmetric line graph.

<|TLDR|>

@highlight

We propose a novel graph neural network architecture based on the non-backtracking matrix defined over the edge adjacencies and demonstrate its effectiveness in community detection tasks on graphs.
Many irregular domains such as social networks, financial transactions, neuron connections, and natural language structures are represented as graphs.

In recent years, a variety of  graph neural networks (GNNs) have been successfully applied for representation learning and prediction on such graphs.

However, in many of the applications, the underlying graph changes over time and existing GNNs are inadequate for handling such dynamic graphs.

In this paper we propose a novel technique for learning embeddings of dynamic graphs based on a tensor algebra framework.

Our method extends the popular graph convolutional network (GCN) for learning representations of dynamic graphs using the recently proposed tensor M-product technique.

Theoretical results that establish the connection between the proposed tensor approach and spectral convolution of tensors are developed.

Numerical experiments on real datasets demonstrate the usefulness of the proposed method for an edge classification task on dynamic graphs.

Graphs are popular data structures used to effectively represent interactions and structural relationships between entities in structured data domains.

Inspired by the success of deep neural networks for learning representations in the image and language domains, recently, application of neural networks for graph representation learning has attracted much interest.

A number of graph neural network (GNN) architectures have been explored in the contemporary literature for a variety of graph related tasks and applications (Hamilton et al., 2017; Seo et al., 2018; Zhou et al., 2018; Wu et al., 2019) .

Methods based on graph convolution filters which extend convolutional neural networks (CNNs) to irregular graph domains are popular (Bruna et al., 2013; Defferrard et al., 2016; Kipf and Welling, 2016) .

Most of these GNN models operate on a given, static graph.

In many real-world applications, the underlining graph changes over time, and learning representations of such dynamic graphs is essential.

Examples include analyzing social networks (Berger-Wolf and Saia, 2006) , predicting collaboration in citation networks (Leskovec et al., 2005) , detecting fraud and crime in financial networks (Weber et al., 2018; Pareja et al., 2019) , traffic control (Zhao et al., 2019) , and understanding neuronal activities in the brain (De Vico Fallani et al., 2014) .

In such dynamic settings, the temporal interdependence in the graph connections and features also play a substantial role.

However, efficient GNN methods that handle time varying graphs and that capture the temporal correlations are lacking.

By dynamic graph, we mean a sequence of graphs (V, A (t) , X (t) ), t ??? {1, 2, . . .

, T }, with a fixed set V of N nodes, adjacency matrices A (t) ??? R N ??N , and graph feature matrices X (t) ??? R N ??F where X (t) n: ??? R F is the feature vector consisting of F features associated with node n at time t.

The graphs can be weighted, and directed or undirected.

They can also have additional properties like (time varying) node and edge classes, which would be stored in a separate structure.

Suppose we only observe the first T < T graphs in the sequence.

The goal of our method is to use these observations to predict some property of the remaining T ??? T graphs.

In this paper, we use it for edge classification.

Other potential applications are node classification and edge/link prediction.

In recent years, tensor constructs have been explored to effectively process high-dimensional data, in order to better leverage the multidimensional structure of such data (Kolda and Bader, 2009) .

Tensor based approaches have been shown to perform well in many image and video processing ap- plications Martin et al., 2013; Zhang et al., 2014; Zhang and Aeron, 2016; Lu et al., 2016; Newman et al., 2018) .

A number of tensor based neural networks have also been investigated to extract and learn multi-dimensional representations, e.g. methods based on tensor decomposition (Phan and Cichocki, 2010), tensor-trains (Novikov et al., 2015; Stoudenmire and Schwab, 2016) , and tensor factorized neural network (Chien and Bao, 2017) .

Recently, a new tensor framework called the tensor M-product framework (Braman, 2010; Kilmer and Martin, 2011; Kernfeld et al., 2015) was proposed that extends matrix based theory to high-dimensional architectures.

In this paper, we propose a novel tensor variant of the popular graph convolutional network (GCN) architecture (Kipf and Welling, 2016), which we call TensorGCN.

It captures correlation over time by leveraging the tensor M-product framework.

The flexibility and matrix mimeticability of the framework, help us adapt the GCN architecture to tensor space.

Figure 1 illustrates our method at a high level: First, the time varying adjacency matrices A (t) and feature matrices X (t) of the dynamic graph are aggregated into an adjacency tensor and a feature tensor, respectively.

These tensors are then fed into our TensorGCN, which computes an embedding that can be used for a variety of tasks, such as link prediction, and edge and node classification.

GCN architectures are motivated by graph convolution filtering, i.e., applying filters/functions to the graph Laplacian (in turn its eigenvalues) (Bruna et al., 2013) , and we establish a similar connection between TensorGCN and spectral filtering of tensors.

Experimental results on real datasets illustrate the performance of our method for the edge classification task on dynamic graphs.

Elements of our method can also be used as a preprocessing step for other dynamic graph methods.

The idea of using graph convolution based on the spectral graph theory for GNNs was first introduced by Bruna et al. (2013) .

Defferrard et al. (2016) then proposed Chebnet, where the spectral filter was approximated by Chebyshev polynomials in order to make it faster and localized.

Kipf and Welling (2016) presented the simplified GCN, a degree-one polynomial approximation of Chebnet, in order to speed up computation further and improve the performance.

There are many other works that deal with GNNs when the graph and features are fixed/static; see the review papers by Zhou et al. (2018) and Wu et al. (2019) and references therein.

These methods cannot be directly applied to the dynamic setting we consider.

Seo et al. (2018) devised the Graph Convolutional Recurrent Network for graphs with time varying features.

However, this method assumes that the edges are fixed over time, and is not applicable in our setting.

Wang et al. (2018) proposed a method called EdgeConv, which is a neural network (NN) approach that applies convolution operations on static graphs in a dynamic fashion.

Their approach is not applicable when the graph itself is dynamic.

Zhao et al. (2019) develop a temporal GCN method called T-GCN, which they apply for traffic prediction.

Their method assumes the graph remains fixed over time, and only the features vary.

The set of methods most relevant to our setting of learning embeddings of dynamic graphs use combinations of GNNs and recurrent architectures (RNN), to capture the graph structure and handle time dynamics, respectively.

The approach in Manessi et al. (2019) uses Long Short-Term Memory (LSTM), a recurrent network, in order to handle time variations along with GNNs.

They design architectures for semi-supervised node classification and for supervised graph classification.

Pareja et al. (2019) presented a variant of GCN called EvolveGCN, where Gated Recurrent Units (GRU) and LSTMs are coupled with a GCN to handle dynamic graphs.

This paper is currently the stateof-the-art.

However, their approach is based on a heuristic RNN/GRU mechanism, which is not theoretically viable, and does not harness a tensor algebraic framework to incorporate time varying information.

Newman et al. (2018) present a tensor NN which utilizes the M-product tensor framework.

Their approach can be applied to image and other high-dimensional data that lie on regular grids, and differs from ours since we consider data on dynamic graphs.

Here, we cover the necessary preliminaries on tensors and the M-product framework.

For a more general introduction to tensors, we refer the reader to the review paper by Kolda and Bader (2009) .

In this paper, a tensor is a three-dimensional array of real numbers denoted by boldface Euler script letters, e.g. X ??? R I??J??T .

Matrices are denoted by bold uppercase letters, e.g. X; vectors are denoted by bold lowercase letter, e.g. x; and scalars are denoted by lowercase letters, e.g. x. An element at position (i, j, t) in a tensor is denotes by subscripts, e.g. X ijt , with similar notation for elements of matrices and vectors.

A colon will denote all elements along that dimension; X i: denotes the ith row of the matrix X, and X ::k denotes the kth frontal slice of X. The vectors X ij: are called the tubes of X.

The framework we consider relies on a new definition of the product of two tensors, called the M-product (Braman, 2010; Kilmer and Martin, 2011; Kernfeld et al., 2015) .

A distinguishing feature of this framework is that the M-product of two three-dimensional tensors is also three-dimensional, which is not the case for e.g. tensor contractions (Bishop and Goldberg, 2012) .

It allows one to elegantly generalize many classical numerical methods from linear algebra, and has been applied e.g. in neural networks (Newman et al., 2018) , imaging Martin et al., 2013; Semerci et al., 2014) , facial recognition , and tensor completion and denoising (Zhang et al., 2014; Zhang and Aeron, 2016; Lu et al., 2016) .

Although the framework was originally developed for three-dimensional tensors, which is sufficient for our purposes, it has been extended to handle tensors of dimension greater than three (Martin et al., 2013) .

The following definitions 3.1-3.3 describe the M-product.

Definition 3.1 (M-transform).

Let M ??? R T ??T be a mixing matrix.

The M-transform of a tensor X ??? R I??J??T is denoted by X ?? 3 M ??? R I??J??T and defined elementwise as

We say that X ?? 3 M is in the transformed space.

may also be written in matrix form as

, where the unfold operation takes the tubes of X and stack them as columns into a T ?? IJ matrix, and fold(unfold(X)) = X. Appendix A provides illustrations of how the M-transform works.

Definition 3.2 (Facewise product).

Let X ??? R I??J??T and Y ??? R J??K??T be two tensors.

The

I??J??T and Y ??? R J??K??T be two tensors, and let M ??? R T ??T be an invertible matrix.

The M-product, denoted by X Y ??? R I??K??T , is defined as

In the original formulation of the M-product, M was chosen to be the Discrete Fourier Transform (DFT) matrix, which allows efficient computation using the Fast Fourier Transform (FFT) (Braman, 2010; Kilmer and Martin, 2011; .

The framework was later extended for arbitrary invertible M (e.g. discrete cosine and wavelet transforms) (Kernfeld et al., 2015) .

A benefit of the tensor M-product framework is that many standard matrix concepts can be generalized in a straightforward manner.

Definitions 3.4-3.7 extend the matrix concepts of diagonality, identity, transpose and orthogonality to tensors (Braman, 2010; .

Definition 3.5 (Identity tensor).

Let?? ??? R N ??N ??T be defined facewise as?? ::t = I, where I is the matrix identity.

The M-product identity tensor I ??? R N ??N ??T is then defined as

Definition 3.6 (Tensor transpose).

The transpose of a tensor X is defined as X def = Y ?? 3 M ???1 , where Y ::t = (X ?? 3 M) ::t for each t ??? {1, . . . , T }.

Definition 3.7 (Orthogonal tensor).

A tensor X ??? R N ??N ??T is said to be orthogonal if X X = X X = I.

Leveraging these concepts, a tensor eigendecomposition can now be defined (Braman, 2010; : Definition 3.8 (Tensor eigendecomposition).

Let X ??? R N ??N ??T be a tensor and assume that each frontal slice (X ?? 3 M) ::t is symmetric.

We can then eigendecompose these as (X ?? 3 M) ::t = Q ::tD::tQ ::t , whereQ ::t ??? R N ??N is orthogonal andD ::t ??? R N ??N is diagonal (see e.g. Theorem 8.1.1 in Golub and Van Loan (2013) ).

The tensor eigendecomposition of X is then defined as

Our approach is inspired by the first order GCN by Kipf and Welling (2016) for static graphs, owed to its simplicity and effectiveness.

For a graph with adjacency matrix A and feature matrix X, a GCN layer takes the form Y = ??(??XW), wher???

is the matrix identity, W is a matrix to be learned when training the NN, and ?? is an activation function, e.g., ReLU.

Our approach translates this to a tensor model by utilizing the M-product framework.

We first introduce a tensor activation function?? which operates in the transformed space.

Definition 4.1.

Let A ??? R I??J??T be a tensor and ?? an elementwise activation function.

We define the activation function?? as??(A)

We can now define our proposed dynamic graph embedding.

Let A ??? R N ??N ??T be a tensor with frontal slices A ::t =?? (t) , where?? (t) is the normalization of A (t) .

Moreover, let X ??? R N ??F ??T be a tensor with frontal slices X ::t = X (t) .

Finally, let W ??? R F ??F ??T be a weight tensor.

We define our dynamic graph embedding as Y = A X W ??? R N ??F ??T .

This computation can also be repeated in multiple layers.

For example, a 2-layer formulation would be of the form

One important consideration is how to choose the matrix M which defines the M-product.

For time-varying graphs, we choose M to be lower triangular and banded so that each frontal slice (A ?? 3 M) ::t is a linear combination of the adjacency matrices A ::max(1,t???b+1) , . . .

, A ::t , where we refer to b as the "bandwidth" of M. This choice ensures that each frontal slice (A ?? 3 M) ::t only contains information from current and past graphs that are close temporally.

Specifically, the entries of M are set to

otherwise, which implies that k M tk = 1 for each t. Another possibility is to treat M as a parameter matrix to be learned from the data.

In order to avoid over-parameterization and improve the performance, we choose the weight tensor W (at each layer), such that each of the frontal slices of W in the transformed domain remains the same, i.e., (W ?? 3 M) ::t = (W ?? 3 M) ::t ???t, t .

In other words, the parameters in each layer are shared and learned over all the training instances.

This reduces the number of parameters to be learned significantly.

An embedding Y ??? R N ??F ??T can now be used for various prediction tasks, like link prediction, and edge and node classification.

In Section 5, we apply our method for edge classification by using a model similar to that used by Pareja et al. (2019) : Given an edge between nodes m and n at time t, the predictive model is

where (Y ?? 3 M) m:t ??? R F and (Y ?? 3 M) n:t ??? R F are row vectors, U ??? R C??2F is a weight matrix, and C the number of classes.

Note that the embedding Y is first M-transformed before the matrix U is applied to the appropriate feature vectors.

This, combined with the fact that the tensor activation functions are applied elementwise in the transformed domain, allow us to avoid ever needing to apply the inverse M-transform.

This approach reduces the computational cost, and has been found to improve performance in the edge classification task.

Here, we present the results that establish the connection between the proposed TensorGCN and spectral convolution of tensors, in particular spectral filtering and approximation on dynamic graphs.

This is analogous to the graph convolution based on spectral graph theory in the GNNs by Bruna et al. (2013) , Defferrard et al. (2016) , and Kipf and Welling (2016) .

All proofs are provided in Appendix D.

Let L ??? R N ??N ??T be a form of tensor Laplacian defined as L def = I ??? A. Throughout the remainder of this subsection, we will assume that the adjacency matrices A (t) are symmetric.

Following the work by , three-dimensional tensors in R M ??N ??T can be viewed as operators on N ?? T matrices, with those matrices "twisted" into tensors in R N ??1??T .

With this in mind, we define a tensor variant of the graph Fourier transform.

Definition 4.4 (Tensor-tube M-product).

Let X ??? R I??J??T and ?? ??? R 1??1??T .

Analogously to the definition of the matrix-scalar product, we define X ?? via (X ??) ij:

Definition 4.5 (Tensor graph Fourier transform).

Let X ??? R N ??F ??T be a tensor.

We define a tensor graph Fourier transform F as F (X)

This is analogous to the definition of the matrix graph Fourier transform.

This defines a convolution like operation for tensors similar to spectral graph convolution (Shuman et al., 2013; Bruna et al., 2013) .

Each lateral slice X :j: is expressible in terms of the set {Q :n: } N n=1 as follows:

where each (Q X :j: ) n1: ??? R 1??1??T can be considered a tubal scalar.

In fact, the lateral slices Q :n: form a basis for the set R N ??1??T with product ; see Appendix D for further details.

Definition 4.6 (Tensor spectral graph filtering).

Given a signal X ??? R N ??1??T and a function g : R 1??1??T ??? R 1??1??T , we define the tensor spectral graph filtering of X with respect to g as

where

In order to avoid the computation of an eigendecomposition, Defferrard et al. (2016) use a polynomial to approximate the filter function.

We take a similar approach, and approximate g(D) with an M-product polynomial.

For this approximation to make sense, we impose additional structure on g. Assumption 4.7.

Assume that g :

where f is defined elementwise as

Proposition 4.8.

Suppose g satisfies Assumption 4.7.

For any ?? > 0, there exists an integer K and a set {??

where ?? is the tensor Frobenius norm, and where

As in the work of Defferrard et al. (2016) , a tensor polynomial approximation allows us to approximate X filt in (2) without computing the eigendecomposition of L:

All that is necessary is to compute tensor powers of L.

We can also define tensor polynomial analogs of the Chebyshev polynomials and do the approximation in (3) in terms of those instead of the tensor monomials D k .

This is not necessary for the purposes of this paper.

Instead, we note that if a degree-one approximation is used, the computation in (3) becomes

, which is analogous to the parameter choice made in the degree-one approximation by Kipf and Welling (2016), we get

If we let X contain F signals, i.e., X ??? R N ??F ??T , and apply F filters, (4) becomes

where ?? ??? R F ??F ??T .

This is precisely our embedding model, with ?? replaced by a learnable parameter tensor W.

Here, we present results for edge classification on four datasets 1 : The Bitcoin Alpha and OTC transaction datasets (Kumar et al., 2016) , the Reddit body hyperlink dataset (Kumar et al., 2018) , and a chess results dataset (Kunegis, 2013) .

The bitcoin datasets consist of transaction histories for users on two different platforms.

Each node is a user, and each directed edge indicates a transaction and is labeled with an integer between ???10 and 10 which indicates the senders trust for the receiver.

We convert these labels to two classes: positive (trustworthy) and negative (untrustworthy).

The Reddit dataset is build from hyperlinks from one subreddit to another.

Each node represents a subreddit, and each directed edge is an interaction which is labeled with ???1 for a hostile interaction or +1 for a friendly interaction.

We only consider those subreddits which have a total of 20 interactions or more.

In the chess dataset, each node is a player, and each directed edge represents a match with the source node being the white player and the target node being the black player.

Each edge is labeled ???1 for a black victory, 0 for a draw, and +1 for a white victory.

The data is temporally partitioned into T graphs, with each graph containing data from a particular time window.

Both T and the time window length can vary between datasets.

For each node-time pair (n, t) in these graphs, we compute the number of outgoing and incoming edges and use these two numbers as features.

The adjacency tensor A is then constructed as described in Section 4.

The T frontal slices of A are divided into S train training slices, S val validation slices, and S test testing slices, which come sequentially after each other; see Figure 2 and Table 2 .

Since the adjacency matrices corresponding to graphs are very sparse for these datasets, we apply the same technique as Pareja et al. (2019) and add the entries of each frontal slice A ::t to the following l ??? 1 frontal slices A ::t , . . .

, A ::(t+l???1) , where we refer to l as the "edge life." Note that this only affects A, and that the added edges are not treated as real edges in the classification problem.

The bitcoin and Reddit datasets are heavily skewed, with about 90% of edges labeled positively, and the remaining labeled negatively.

Since the negative instances are more interesting to identify (e.g. to prevent financial fraud or online hostility), we use the F1 score to evaluate the experiments on these datasets, treating the negative edges as the ones we want to identify.

The classes are more well-balanced in the chess dataset, so we use accuracy to evaluate those experiments.

We choose to use an embedding Y train = A ::(1:Strain) X ::(1:Strain) W for training.

When computing the embeddings for the validation and testing data, we still need S train frontal slices of A, which we get by using a sliding window of slices.

This is illustrated in Figure 2 , where the green, blue and red blocks show the frontal slices used when computing the embeddings for the training, validation and testing data, respectively.

The embeddings for the validation and testing data are Y val = A ::(Sval+1:Strain+Sval) X ::(Sval+1:Strain+Sval) W and Y test = A ::(Sval+Stest+1:T ) X ::(Sval+Stest+1:T ) W, respectively.

Preliminary experiments with 2-layer architectures did not show convincing improvements in performance.

We believe this is due to the fact that the datasets only have two features, and that a 1-layer architecture therefore is sufficient for extracting relevant information in the data.

For training, we use the cross entropy loss function: where f (m, n, t) ??? R C is a one-hot vector encoding the true class of the edge (m, n) at time t, and ?? ??? R C is a vector summing to 1 which contains the weight of each class.

Since the bitcoin and Reddit datasets are so skewed, we weigh the minority class more heavily in the loss function for those datasets, and treat ?? as a hyperparameter; see Appendix C for details.

The experiments are implemented in PyTorch with some preprocessing done in Matlab.

Our code is available at [url redacted for review].

In the experiments, we use an edge life of l = 10, a bandwidth b = 20, and F = 6 output features.

Since the graphs in the considered datasets are directed, we also investigate the impact of symmetrizing the adjacency matrices, where the symmetrized version of an adjacency matrix A is defined as A sym def = 1/2(A + A ).

We compare our method with three other methods.

The first one is a variant of the WD-GCN by Manessi et al. (2019) , which they specify in Equation (8a) of their paper.

For the LSTM layer in their description, we use 6 output features instead of N .

This is to avoid overfitting and make the method more comparable to ours which uses 6 output features.

For the final layer, we use the same prediction model as that used by Pareja et al. (2019) for edge classification.

The second method is a 1-layer variant of EvolveGCN-H by Pareja et al. (2019) .

The third method is a simple baseline which uses a 1-layer version of the GCN by Kipf and Welling (2016) .

It uses the same weight matrix W for all temporal graphs.

Both EvolveGCN-H and the baseline GCN use 6 output features as well.

Table 3 shows the results when the adjacency matrices have not been symmetrized.

In this case, our method outperforms the other methods on the two bitcoin datasets and the chess dataset, with WD-GCN performing best on the Reddit dataset.

Table 4 shows the results for when the adjacency matrices have been symmetrized.

Our method outperforms the other methods on the Bitcoin OTC dataset and the chess dataset, and performs similarly but slightly worse than the best performing methods on the Bitcoin Alpha and Reddit datasets.

Overall, it seems like symmetrizing the adjacency matrices leads to lower performance.

We have presented a novel approach for dynamic graph embedding which leverages the tensor Mproduct framework.

We used it for edge classification in experiments on four real datasets, where it performed competitively compared to state-of-the-art methods.

Future research directions include further developing the theoretical guarantees for the method, investigating optimal structure and learning of the transform matrix M, using the method for other prediction tasks, and investigating how to utilize deeper architectures for dynamic graph learning.

We provide some illustrations that show how the M-transform in Definition 3.1 works.

Recall that X ?? 3 M = fold(M unfold(X)).

The matrix X is first unfolded into a matrix, as illustrated in Figure 3 .

This unfolded tensor is then multiplied from the left by the matrix M, as illustrated in Figure 4 ; the figure also illustrates the banded lower triangular structure of M. Finally, the output matrix is folded back into a tensor.

The fold operation is defined to be the inverse of the unfold operation.

??? The Bitcoin Alpha dataset is available at https://snap.stanford.edu/data/soc-sign-bitcoin-alpha.html.

??? The Bitcoin OTC dataset is available at https://snap.stanford.edu/data/soc-sign-bitcoin-otc.html.

??? The Reddit dataset is available at https://snap.stanford.edu/data/soc-RedditHyperlinks.html.

Note that we use the dataset with hyperlinks in the body of the posts.

??? The chess dataset is available at http://konect.uni-koblenz.de/networks/chess.

When partitioning the data into T graphs, as described in Section 5, if there are multiple data points corresponding to an edge (m, n) for a given time step t, we only add that edge once to the corresponding graph and set the label equal to the sum of the labels of the different data points.

For example, if bitcoin user m makes three transactions to n during time step t with ratings 10, 2, ???1, then we add a single edge (m, n) to graph t with label 10 + 2 ??? 1 = 11.

For training, we run gradient descent with a learning rate of 0.01 and momentum of 0.9 for 10,000 iterations.

For each 100 iterations, we compute and store the performance of the model on the validation data.

As mentioned in Section 5, the weight vector ?? in the loss function (5) is treated as a hyperparameter in the bitcoin and Reddit experiments.

Since these datasets all have two edge classes, let ?? 0 and ?? 1 be the weights of the minority (negative) and majority (positive) classes, respectively.

Since these parameters add to 1, we have ?? 1 = 1 ??? ?? 0 .

For all methods, we repeat the bitcoin and Reddit experiments once for each ?? 0 ??? {0.75, 0.76, . . .

, 0.95}. For each model and dataset, we then find the best stored performance of the model on the validation data across all ?? 0 values.

We then treat the corresponding model as the trained model, and report its performance on the testing data in Tables 3 and 4 .

The results for the chess experiment are computed in the same way, but only for a single vector ?? = [1/3, 1/3, 1/3].

Throughout this section, ?? will denote the Frobenius norm (i.e., the square root of the sum of the elements squared) of a matrix or tensor, and ?? 2 will denote the matrix spectral norm.

We first provide a few further results that clarify the algebraic properties of the M-product.

Let R 1??1??T denote the set of 1 ?? 1 ?? T tensors.

Similarly, let R N ??1??T denote the set of N ?? 1 ?? T tensors.

Under the M-product framework, the set R 1??1??T play a role similar to that played by scalars in matrix algebra.

With this in mind, the set R N ??1??T can be seen as a length N vector consisting of tubal elements of length T .

Propositions D.1 and D.2 make this more precise.

Proposition D.1 (Proposition 4.2 in Kernfeld et al. (2015) ).

The set R 1??1??T with product , which is denoted by ( , R 1??1??T ), is a commutative ring with identity.

Proposition D.2 (Theorem 4.1 in Kernfeld et al. (2015) ).

The set R N ??1??T with product , which is denoted by ( , R N ??1??T ), is a free module over the ring ( , R 1??1??T ).

A free module is similar to a vector space.

Like a vector space, it has a basis.

Proposition D.3 shows that the lateral slices of Q in the tensor eigendecomposition form a basis for ( , R N ??1??T ), similarly to how the eigenvectors in a matrix eigendecomposition form a basis.

Proposition D.3.

The lateral slices Q :n: ??? R N ??1??T of Q in Definition 3.8 form a basis for ( , R N ??1??T ).

Proof.

Let X ??? R N ??1??T .

Note that Since each frontal face of Q ?? 3 M is an invertible matrix, this implies that each frontal face of S ?? 3 M is zero, and hence S = 0.

So the lateral slices of Q are also linearly independent in ( , R N ??1??T ).

@highlight

We propose a novel tensor based method for graph convolutional networks on dynamic graphs
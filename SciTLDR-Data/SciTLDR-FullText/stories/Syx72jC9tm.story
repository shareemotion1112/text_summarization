Invariant and equivariant networks have been successfully used for learning images, sets, point clouds, and graphs.

A basic challenge in developing such networks is finding the maximal collection of invariant and equivariant \emph{linear} layers.

Although this question is answered for the first three examples (for popular transformations, at-least), a full characterization of invariant and equivariant linear layers for graphs is not known.



In this paper we provide a characterization of all permutation invariant and equivariant linear layers for (hyper-)graph data, and show that their dimension, in case of edge-value graph data, is $2$ and $15$, respectively.

More generally, for graph data defined on $k$-tuples of nodes, the dimension is the $k$-th and $2k$-th Bell numbers.

Orthogonal bases for the layers are computed, including generalization to multi-graph data.

The constant number of basis elements and their characteristics allow successfully applying the networks to different size graphs.

From the theoretical point of view, our results generalize and unify recent advancement in equivariant deep learning.

In particular, we show that our model is capable of approximating any message passing neural network.



Applying these new linear layers in a simple deep neural network framework is shown to achieve comparable results to state-of-the-art and to have better expressivity than previous invariant and equivariant bases.

We consider the problem of graph learning, namely finding a functional relation between input graphs (more generally, hyper-graphs) G and corresponding targets T , e.g., labels.

As graphs are common data representations, this task received quite a bit of recent attention in the machine learning community BID2 ; BID13 ; ; BID38 .More specifically, a (hyper-)graph data point G = (V, A) consists of a set of n nodes V, and values A attached to its hyper-edges 1 .

These values are encoded in a tensor A. The order of the tensor A, or equivalently, the number of indices used to represent its elements, indicates the type of data it represents, as follows: First order tensor represents node-values where A i is the value of the i-th node; Second order tensor represents edge-values, where A ij is the value attached to the (i, j) edge; in general, k-th order tensor encodes hyper-edge-values, where A i1,...,i k represents the value of the hyper-edge represented by (i 1 , . . .

, i k ).

For example, it is customary to represent a graph using a binary adjacency matrix A, where A ij equals one if vertex i is connected to vertex j and zero otherwise.

We denote the set of order-k tensors by R n k .The task at hand is constructing a functional relation f (A ) ≈ T , where f is a neural network.

If T = t is a single output response then it is natural to ask that f is order invariant, namely it should produce the same output regardless of the node numbering used to encode A. For example, if we represent a graph using an adjacency matrix A = A ∈ R n×n , then for an arbitrary permutation matrix P and an arbitrary adjacency matrix A, the function f is order invariant if it satisfies f (P T AP ) = f (A).

If the targets T specify output response in a form of a tensor, T = T , then it is natural to ask that f is order equivariant, that is, f commutes with the renumbering of nodes operator acting on tensors.

Using the above adjacency matrix example, for every adjacency matrix A and Figure 1 : The full basis for equivariant linear layers for edge-value data A ∈ R n×n , for n = 5.

The purely linear 15 basis elements, B µ , are represented by matrices n 2 × n 2 , and the 2 bias basis elements (right), C λ , by matrices n × n, see equation 9.every permutation matrix P , the function f is equivariant if it satisfies f (P T AP ) = P T f (A)P .

To define invariance and equivariance for functions acting on general tensors A ∈ R n k we use the reordering operator: P A is defined to be the tensor that results from renumbering the nodes V according to the permutation defined by P .

Invariance now reads as f (P A) = f (A); while equivariance means f (P A) = P f (A).

Note that the latter equivariance definition also holds for functions between different order tensors, f : R n k → R n l .Following the standard paradigm of neural-networks where a network f is defined by alternating compositions of linear layers and non-linear activations, we set as a goal to characterize all linear invariant and equivariant layers.

The case of node-value input A = a ∈ R n was treated in the pioneering works of BID39 ; BID26 .

These works characterize all linear permutation invariant and equivariant operators acting on node-value (i.e., first order) tensors, R n .

In particular it it shown that the linear space of invariant linear operators L : R n → R is of dimension one, containing essentially only the sum operator, L(a) = α1T a. The space of equivariant linear operators L : DISPLAYFORM0 The general equivariant tensor case was partially treated in where the authors make the observation that the set of standard tensor operators: product, element-wise product, summation, and contraction are all equivariant, and due to linearity the same applies to their linear combinations.

However, these do not exhaust nor provide a full and complete basis for all possible tensor equivariant linear layers.

In this paper we provide a full characterization of permutation invariant and equivariant linear layers for general tensor input and output data.

We show that the space of invariant linear layers L : R n k → R is of dimension b(k), where b(k) is the k-th Bell number.

The k-th Bell number is the number of possible partitions of a set of size k; see inset for the case k = 3.

Furthermore, the space of equivariant linear layers DISPLAYFORM1 Remarkably, this dimension is independent of the size n of the node set V. This allows applying the same network on graphs of different sizes.

For both types of layers we provide a general formula for an orthogonal basis that can be readily used to build linear invariant or equivariant layers with maximal expressive power.

Going back to the example of a graph represented by an adjacency matrix A ∈ R n×n we have k = 2 and the linear invariant layers L : Figure 1 shows visualization of the basis to the linear equivariant layers acting on edge-value data such as adjacency matrices.

DISPLAYFORM2 In BID12 the authors provide an impressive generalization of the case of node-value data to several node sets, V 1 , V 2 , . . .

, V m of sizes n 1 , n 2 , . . . , n m .

Their goal is to learn interactions across sets.

That is, an input data point is a tensor A ∈ R n1×n2×···×nm that assigns a value to each element in the cartesian product V 1 × V 2 × · · · × V m .

Renumbering the nodes in each node set using permutation matrices P 1 , . . .

, P m (resp.) results in a new tensor we denote by P 1:m A. Order invariance means f (P 1:m A) = f (A) and order equivariance is f (P 1:m A) = P 1:m f (A).

BID12 introduce bases for linear invariant and equivariant layers.

Although the layers in BID12 satisfy the order invariance and equivariance, they do not exhaust all possible such layers in case some node sets coincide.

For example, if V 1 = V 2 they have 4 independent learnable parameters where our model has the maximal number of 15 parameters.

Our analysis allows generalizing the multi-node set case to arbitrary tensor data over V 1 × V 2 × · · · × V m .

Namely, for data points in the form of a tensor A ∈ R n k 1 1 ×n k 2 2 ×···×n km m .

The tensor A attaches a value to every element of the Cartesian product DISPLAYFORM3 2 , that is, k 1 -tuple from V 1 , k 2 -tuple from V 2 and so forth.

We show that the linear space of invariant linear layers DISPLAYFORM4 , while the equivariant linear layers L : DISPLAYFORM5 We also provide orthogonal bases for these spaces.

Note that, for clarity, the discussion above disregards biases and features; we detail these in the paper.

In appendix C we show that our model is capable of approximating any message-passing neural network as defined in BID9 which encapsulate several popular graph learning models.

One immediate corollary is that the universal approximation power of our model is not lower than message passing neural nets.

In the experimental part of the paper we concentrated on possibly the most popular instantiation of graph learning, namely that of a single node set and edge-value data, e.g., with adjacency matrices.

We created simple networks by composing our invariant or equivariant linear layers in standard ways and tested the networks in learning invariant and equivariant graph functions: (i) We compared identical networks with our basis and the basis of BID12 and showed we can learn graph functions like trace, diagonal, and maximal singular vector.

The basis in BID12 , tailored to the multi-set setting, cannot learn these functions demonstrating it is not maximal in the graph-learning (i.e., multi-set with repetitions) scenario.

We also demonstrate our representation allows extrapolation: learning on one size graphs and testing on another size; (ii) We also tested our networks on a collection of graph learning datasets, achieving results that are comparable to the state-of-the-art in 3 social network datasets.

Our work builds on two main sub-fields of deep learning: group invariant or equivariant networks, and deep learning on graphs.

Here we briefly review the relevant works.

Invariance and equivariance in deep learning.

In many learning tasks the functions that we want to learn are invariant or equivariant to certain symmetries of the input object description.

Maybe the first example is the celebrated translation invariance of Convolutional Neural Networks (CNNs) BID20 BID19 ; in this case, the image label is invariant to a translation of the input image.

In recent years this idea was generalized to other types of symmetries such as rotational symmetries BID3 b; BID35 .

BID3 introduced Group Equivariant Neural Networks that use a generalization of the convolution operator to groups of rotations and reflections; BID35 ; also considered rotational symmetries but in the case of 3D shapes and spherical functions.

showed that any equivariant layer is equivalent to a certain parameter sharing scheme.

If we adopt this point of view, our work reveals the structure of the parameter sharing in the case of graphs and hyper-graphs.

In another work, show that a neural network layer is equivariant to the action of some compact group iff it implements a generalized form of the convolution operator.

BID37 suggested certain group invariant/equivariant models and proved their universality.

To the best of our knowledge these models were not implemented.

Learning of graphs.

Learning of graphs is of huge interest in machine learning and we restrict our attention to recent advancements in deep learning on graphs.

BID10 ; BID28 introduced Graph Neural Networks (GNN): GNNs hold a state (a real valued vector) for each node in the graph, and propagate these states according to the graph structure and learned parametric functions.

This idea was further developed in BID22 that use gated recurrent units.

Following the success of CNNs, numerous works suggested ways to define convolution operator on graphs.

One promising approach is to define convolution by imitating its spectral properties using the Laplacian operator to define generalized Fourier basis on graphs BID2 .

Multiple follow-up works BID13 BID6 BID16 BID21 suggest more efficient and spatially localized filters.

The main drawback of spectral approaches is that the generalized Fourier basis is graph-dependent and applying the same network to different graphs can be challenging.

Another popular way to generalize the convolution operator to graphs is learning stationary functions that operate on neighbors of each node and update its current state BID1 BID7 BID11 BID25 BID32 BID31 .

This idea generalizes the locality and weight sharing properties of the standard convolution operators on regular grids.

As shown in the important work of BID9 , most of the the above mentioned methods (including the spectral methods) can be seen as instances of the general class of Message Passing Neural Networks.

In this section we characterize the collection of linear invariant and equivariant layers.

We start with the case of a single node set V of size n and edge-value data, that is order 2 tensors A = A ∈ R n×n .

As a typical example imagine, as above, an adjacency matrix of a graph.

We set a bit of notation.

Given a matrix X ∈ R a×b we denote vec(X) ∈ R ab×1 its column stack, and by brackets the inverse action of reshaping to a square matrix, namely [vec(X)] = X. Let p denote an arbitrary permutation and P its corresponding permutation matrix.

Let L ∈ R 1×n 2 denote the matrix representing a general linear operator L : R n×n → R in the standard basis, then L is order invariant iff Lvec(P T AP ) = Lvec(A).

Using the property of the Kronecker product that vec(XAY ) = Y T ⊗ Xvec(A), we get the equivalent equality DISPLAYFORM0 .

Since the latter equality should hold for every A we get (after transposing both sides of the equation) that order invariant L is equivalent to the equation DISPLAYFORM1 For equivariant layers we consider a general linear operator L : R n×n → R n×n and its cor- DISPLAYFORM2 Using the above property of the Kronecker product again we get DISPLAYFORM3 Noting that P T ⊗ P T is an n 2 × n 2 permutation matrix and its inverse is P ⊗ P we get to the equivalent equality P ⊗ P LP T ⊗ P T vec(A) = Lvec(A).

As before, since this holds for every A and using the properties of the Kronecker product we get that L is order equivariant iff for all permutation matrices P DISPLAYFORM4 From equations 1 and 2 we see that finding invariant and equivariant linear layers for the order-2 tensor data over one node set requires finding fixed points of the permutation matrix group represented by Kronecker powers P ⊗ P ⊗ · · · ⊗ P of permutation matrices P .

As we show next, this is also the general case for order-k tensor data A ∈ R n k over one node set, V. That is, DISPLAYFORM5 for every permutation matrix P , where DISPLAYFORM6 k is the matrix of an invariant operator; and in equation 4, L ∈ R n k ×n k is the matrix of an equivariant operator.

We call equations 3,4 the fixed-point equations.

To see this, let us add a bit of notation first.

Let p denote the permutation corresponding to the permutation matrix P .

We let P A denote the tensor that results from expressing the tensor A after renumbering the nodes in V according to permutation P .

Explicitly, the (p(i 1 ), p(i 2 ), . . . , p(i k ))-th entry of P A equals the (i 1 , i 2 , . . .

, i k )-th entry of A. The matrix that corresponds to the operator P in the standard tensor basis e (i1) ⊗ · · · ⊗ e (i k ) is the Kronecker power P T ⊗k = (P T ) ⊗k .

Note that vec(A) is exactly the coordinate vector of the tensor A in this standard basis and therefore we have vec(P A) = P T ⊗k vec(A).

We now show: DISPLAYFORM7

Proof.

Similarly to the argument from the order-2 case, let L ∈ R 1×n k denote the matrix corresponding to a general linear operator L : R n k → R. Order invariance means DISPLAYFORM0 Using the matrix P T ⊗k we have equivalently LP T ⊗k vec(A) = Lvec(A) which is in turn equivalent to P ⊗k vec(L) = vec(L) for all permutation matrices P .

For order equivariance, let L ∈ R n k ×n k denote the matrix of a general linear operator L : DISPLAYFORM1 Similarly to above this is equivalent to LP T ⊗k vec(A) = P T ⊗k Lvec(A) which in turn leads to P ⊗k LP T ⊗k = L, and using the Kronecker product properties we get P ⊗2k vec(L) = vec(L).

We have reduced the problem of finding all invariant and equivariant linear operators L to finding all solutions L of equations 3 and 4.

Although the fixed point equations consist of an exponential number of equations with only a polynomial number of unknowns they actually possess a solution space of constant dimension (i.e., independent of n).To find the solution of P ⊗ vec(X) = vec(X), where X ∈ R n , note that P ⊗ vec(X) = vec(Q X), where Q = P T .

As above, the tensor Q X is the tensor resulted from renumbering the nodes in V using permutation Q. Equivalently, the fixed-point equations we need to solve can be formulated as Q X = X, ∀Q permutation matricesThe permutation group is acting on tensors X ∈ R n with the action X → Q X. We are looking for fixed points under this action.

To that end, let us define an equivalence relation in the index space of tensors R n , namely in [n] , where with a slight abuse of notation (we use light brackets) we set DISPLAYFORM0 The equality pattern equivalence relation partitions the index set [n] into equivalence classes, the collection of which is denoted [n] / ∼ .

Each equivalence class can be represented by a unique partition of the set [ ] where each set in the partition indicates maximal set of identical values.

Let us exemplify.

For = 2 we have two equivalence classes γ 1 = {{1} , {2}} and γ 2 = {{1, 2}}; γ 1 represents all multi-indices (i, j) where i = j, while γ 2 represents all multi-indices (i, j) where i = j. For = 4, there are 15 equivalence classes DISPLAYFORM1 For each equivalence class γ ∈ [n] / ∼ we define an order-tensor B γ ∈ R n by setting Proof.

Let us first show that: X is a solution to equation 7 iff X is constant on equivalence classes of the equality pattern relation, ∼. Since permutation q : [n] →

[n] is a bijection the equality patterns of a = (i 1 , i 2 , . . .

, i ) ∈ [n] and q(a) = (q(i 1 ), q(i 2 ), . . . , q(i )) ∈ [n] are identical, i.e., a ∼ q(a).

Taking the a ∈ [n] entry of both sides of equation 7 gives X q(a) = X a .

Now, if X is constant on equivalence classes then in particular it will have the same value at a and q(a) for all a ∈ [n] and permutations q.

Therefore X is a solution to equation 7.

For the only if part, consider a tensor X for which there exist multi-indices a ∼ b (with identical equality patterns) and X a = X b then X is not a solution to equation 7.

Indeed, since a ∼ b one can find a permutation q so that b = q(a) and using the equation above, X b = X q(a) = X a which leads to a contradiction.

To finish the proof note that any tensor X, constant on equivalence classes, can be written as a linear combination of B γ , which are merely indicators of the equivalence class.

Furthermore, the collection B γ have pairwise disjoint supports and therefore are an orthogonal basis.

Combining propositions 1 and 2 we get the characterization of invariant and equivariant linear layers acting on general k-order tensor data over a single node set V: DISPLAYFORM2 DISPLAYFORM3 Biases Theorem 1 deals with purely linear layers, that is without bias, i.e., without constant part.

Nevertheless extending the previous analysis to constant layers is straight-forward.

First, any constant layer R n k → R is also invariant so all constant invariant layers are represented by constants c ∈ R. For equivariant layers L : R n k → R n k we note that equivariance means C = L(P A) = P L(A) = P C. Representing this equation in matrix form we get P T ⊗k vec(C) = vec(C).

This shows that constant equivariant layers on one node set acting on general k-order tensors are also characterized by the fixed-point equations, and in fact have the same form and dimensionality as invariant layers on k-order tensors, see equation 3.

Specifically, their basis is B λ , λ ∈ [n] k / ∼ .

For example, for k = 2, the biases are shown on the right in figure 1.Features.

It is pretty common that input tensors have vector values (i.e., features) attached to each hyper-edge (k-tuple of nodes) in V, that is A ∈ R n k ×d .

Now linear invariant R n k ×d → R 1×d or equivariant R n k ×d → R n k ×d layers can be formulated using a slight generalization of the previous analysis.

The operator P A is defined to act only on the nodal indices, i.e., i 1 , . . .

, i k (the first k indices).

Explicitly, the (p(i 1 ), p(i 2 ), . . . , p(i k ), i k+1 )-th entry of P A equals the (i 1 , i 2 , . . .

, i k , i k+1 )-th entry of A.Invariance is now formulated exactly as before, equation 5, namely Lvec(P A) = Lvec(A).

The matrix that corresponds to P acting on R n k ×d in the standard basis is P T ⊗k ⊗ I d and therefore DISPLAYFORM4 Since this is true for all A we have (P DISPLAYFORM5 , using the properties of the Kronecker product.

Equivariance is written as in equation 6, [Lvec(P A)] = P [Lvec(A)].

In matrix form, the equivariance equation becomes DISPLAYFORM6 , since this is true for all A and using the properties of the Kronecker product again we get to DISPLAYFORM7 The basis (with biases) to the solution space of these fixed-point equations is defined as follows.

We use a, Note that these basis elements are similar to the ones in equation 8 with the difference that we have different basis tensor for each pair of input j and output j feature channels.

DISPLAYFORM8 An invariant (equation 10a)/ equivariant (equation 10b) linear layer L including the biases can be written as follows for input A ∈ R n k ×d : DISPLAYFORM9 where the learnable parameters are w ∈ R b(k)×d×d and b ∈ R d for a single linear invariant layer R n k ×d → R d ; and it is w ∈ R b(2k)×d×d and b ∈ R b(k)×d for a single linear equivariant layer DISPLAYFORM10 The natural generalization of theorem 1 to include bias and features is therefore: DISPLAYFORM11 with basis elements defined in equation 9; equation 10a (10b) show the general form of such layers.

Since, by similar arguments to proposition 2, the purely linear parts B and biases C in equation 9 are independent solutions to the relevant fixed-point equations, theorem 2 will be proved if their number equals the dimension of the solution space of these fixed-point equations, namely dd b(k) for purely linear part and d for bias in the invariant case, and dd b(2k) for purely linear and d b(k) for bias in the equivariant case.

This can be shown by repeating the arguments of the proof of proposition 2 slightly adapted to this case, or by a combinatorial identity we show in Appendix B .For example, figure 1 depicts the 15 basis elements for linear equivariant layers R n×n → R n×n taking as input edge-value (order-2) tensor data A ∈ R n×n and outputting the same dimension tensor.

The basis for the purely linear part are shown as n 2 × n 2 matrices while the bias part as n × n matrices (far right); the size of the node set is |V| = n = 5.Mixed order equivariant layers.

Another useful generalization of order equivariant linear layers is to linear layers between different order tensor layers, that is, L : R n k → R n l , where l = k. For example, one can think of a layer mapping an adjacency matrix to per-node features.

For simplicity we will discuss the purely linear scalar-valued case, however generalization to include bias and/or general feature vectors can be done as discussed above.

Consider the matrix L ∈ R n l ×n k representing the linear layer L, using the renumbering operator, P , order equivariance is equivalent to [Lvec(P A)] = P [Lvec(A)].

Note that while this equation looks identical to equation 6 it is nevertheless different in the sense that the P operator in the l.h.s.

of this equation acts on k-order tensors while the one on the r.h.s.

acts on l-order tensor.

Still, we can transform this equation to a matrix equation as before by remembering that P T ⊗k is the matrix representation of the renumbering operator P acting on k-tensors in the standard basis.

Therefore, repeating the arguments in proof of proposition 1, equivariance is equivalent to P ⊗(k+l) vec(L) = vec(L), for all permutation matrices P .

This equation is solved as in section 3.1.

The corresponding bases to such equivariant layers are computed as in equation 9b, with the only difference that now DISPLAYFORM12

Implementation details.

We implemented our method in Tensorflow BID0 .

The equivariant linear basis was implemented efficiently using basic row/column/diagonal summation operators, see appendix A for details.

The networks we used are composition of 1 − 4 equivariant linear layers with ReLU activation between them for the equivariant function setting.

For invariant function setting we further added a max over the invariant basis and 1 − 3 fully-connected layers with ReLU activations.

Synthetic datasets.

We tested our method on several synthetic equivariant and invariant graph functions that highlight the differences in expressivity between our linear basis and the basis of BID12 .

Given an input matrix data A ∈ R n×n we considered: (i) projection onto the symmetric matrices 1 2 (A+A T ); (ii) diagonal extraction diag(diag(A)) (keeps only the diagonal and plugs zeros elsewhere); (iii) computing the maximal right singular vector arg max v 2 =1 Av 2 ; and (iv) computing the trace tr(A).

Tasks (i)-(iii) are equivariant while task (iv) is invariant.

We created accordingly 4 datasets with 10K train and 1K test examples of 40×40 matrices; for tasks (i), (ii), (iv) we used i.i.d.

random matrices with uniform distribution in [0, 10]; we used mean-squared error (MSE) as loss; for task (iii) we random matrices with uniform distribution of singular values in [0, 0.5] and spectral gap ≥ 0.5; due to sign ambiguity in this task we used cosine loss of the form l(x, y) = 1 − x/ x , y/ y 2 .

We trained networks with 1, 2, and 3 hidden layers with 8 feature channels each and a single fullyconnected layer.

Both our models as well as BID12 use the same architecture but with different bases for the linear layers.

TAB1 logs the best mean-square error of each method over a set of hyper-parameters.

We add the MSE for the trivial mean predictor.

This experiment emphasizes simple cases in which the additional parameters in our model, with respect to BID12 , are needed.

We note that BID12 target a different scenario where the permutations acting on the rows and columns of the input matrix are not necessarily the same.

The assumption taken in this paper, namely, that the same permutation acts on both rows and columns, gives rise to additional parameters that are associated with the diagonal and with the transpose of the matrix (for a complete list of layers for the k = 2 case see appendix A).

In case of an input matrix that represents graphs, these parameters can be understood as parameters that control self-edges or node features, and incoming/outgoing edges in a different way.

TAB2 shows the result of applying the learned equivariant networks from the above experiment to graphs (matrices) of unseen sizes of n = 30 and n = 50.

Note, that although the network was trained on a fixed size, the network provides plausible generalization to different size graphs.

We note that the generalization of the invariant task of computing the trace did not generalize well to unseen sizes and probably requires training on different sizes as was done in the datasets below.

Graph classification.

We tested our method on standard benchmarks of graph classification.

We use 8 different real world datasets from the benchmark of BID36 : five of these datasets originate from bioinformatics while the other three come from social networks.

In all datasets the adjacency matrix of each graph is used as input and a categorial label is assigned as output.

In the bioinformatics datasets node labels are also provided as inputs.

These node labels can be used in our framework by placing their 1-hot representations on the diagonal of the input.

TAB3 specifies the results for our method compared to state-of-the-art deep and non-deep graph learning methods.

We follow the evaluation protocol including the 10-fold splits of BID40 .

For each dataset we selected learning and decay rates on one random fold.

In all experiments we used a fixed simple architecture of 3 layers with (16, 32, 256) features accordingly.

The last equivariant layer is followed by an invariant max layer according to the invariant basis.

We then add two fully-connected hidden layers with (512, 256) features.

We compared our results to seven deep learning methods: DGCNN BID40 , PSCN BID25 , DCNN BID1 , ECC BID31 , DGK BID36 , DiffPool BID38 and CCN .

We also compare our results to four popular graph kernel methods: Graphlet Kernel (GK) BID29 ,Random Walk Kernel (RW) BID34 , Propagation Kernel (PK) BID24 , and Weisfeiler-lehman kernels (WL) BID30 and two recent feature-based methods: Family of Graph Spectral Distance (FGSD) BID33 and Anonymous Walk Embeddings (AWE) BID15 .

Our method achieved results comparable to the state-of-the-art on the three social networks datasets, and slightly worse results than state-of-the-art on the biological datasets.

Lastly, we provide a generalization of our framework to data that is given on tuples of nodes from a collection of node sets V 1 , V 2 , . . .

, V m of sizes n 1 , n 2 , . . . , n m (resp.), namely A ∈ , where for simplicity we do not discuss features that can be readily added as discussed in section 3.

Note that the case of k i = l i = 1 for all i = 1, . . .

, m is treated in BID12 .

The reordering operator now is built out of permutation matrices P i ∈ R ni×ni (p i denotes the permutation), i = 1, . . .

, m, denoted P 1:m , and defined as follows: the (p 1 (a 1 ), p 2 (a 2 ), . . .

, p m (a m ))-th entry of the tensor P 1:m A, where DISPLAYFORM0 ki is defined to be the (a 1 , a 2 , . . . , a m )-th entry of the tensor A. Rewriting the invariant and equivariant equations, i.e., equation 5, 6, in matrix format, similarly to before, we get the fixed-point equa- (11) where DISPLAYFORM1 The number of these tensors is m i=1 b(i) for invariant layers and m i=1 b(k i + l i ) for equivariant layers.

Since these are all linear independent (pairwise disjoint support of non-zero entries) we need to show that their number equal the dimension of the solution of the relevant fixed-point equations above.

This can be done again by similar arguments to the proof of proposition 2 or as shown in appendix B. To summarize: DISPLAYFORM2 Orthogonal bases for these layers are listed in equation 11.

This research was supported in part by the European Research Council (ERC Consolidator Grant, "LiftMatch" 771136) and the Israel Science Foundation (Grant No. 1830/17).We normalize each operation to have unit max operator norm.

We note that in case the input matrix is symmetric, our basis reduces to 11 elements in the first layer.

If we further assume the matrix has zero diagonal we get a 6 element basis in the first layer.

In both cases our model is more expressive than the 4 element basis of BID12 and as the output of the first layer (or other inner states) need not be symmetric nor have zero diagonal the deeper layers can potentially make good use of the full 15 element basis.

We prove a useful combinatorial fact as a corollary of proposition 2.

This fact will be used later to easily compute the dimensions of more general spaces of invariant and equivariant linear layers.

We use the fact that if V is a representation of a finite group G then DISPLAYFORM0 is a projection onto V G = {v ∈ V | gv = v, ∀g ∈ G}, the subspace of fixed points in V under the action of G, and consequently that tr(φ) = dim(V G ) (see BID8 for simple proofs).

Proposition 3.

The following formula holds: DISPLAYFORM1 where Π n is the matrix permutation group of dimensions n × n.

Proof.

In our case, the vector space is the space of order-k tensors and the group acting on it is the matrix group G = P ⊗k | P ∈ Π m .

dim(V G ) = tr(φ) = 1 |G| g∈G tr(g) = 1 n!

P ∈Πn tr(P ⊗k ) = 1 n!

P ∈Πn tr(P ) k ,where we used the multiplicative law of the trace with respect to Kronecker product.

Now we use proposition 2 noting that in this case V G is the solution space of the fixed-point equations.

Therefore, dim(V G ) = b(k) and the proof is finished.

Recall that for a permutation matrix P , tr(P ) = | {i ∈ [n] s.t.

P fixes e i } |.

Using this, we can interpret the equation in proposition 3 as the k-th moment of a random variable counting the number of fixed points of a permutation, with uniform distribution over the permutation group.

Proposition 3 proves that the k-th moment of this random variable is the k-th Bell number.

We can now use proposition 3 to calculate the dimensions of two linear layer spaces: (i) Equivariant layers acting on order-k tensors with features (as in 3); and (ii) multi-node sets (as in section 5).Theorem 2.

The space of invariant (equivariant) linear layers R n k ,d → R d (R n k ×d → R n k ×d ) is of dimension dd b(k) + d (for equivariant: dd b(2k) + d b(k)) with basis elements defined in equation 9; equations 10a (10b) show the general form of such layers.

Proof.

We prove the dimension formulas for the invariant case.

The equivariant case is proved similarly.

The solution space for the fixed point equations is the set V G for the matrix group G = P ⊗k ⊗

I d ⊗

I d | P ∈ Π n .

Using the projection formula 12 we get that the dimension of the solution subspace, which is the space of invariant linear layers, can be computed as follows: DISPLAYFORM2 5.

The last step is to apply an MLP to the last d + d feature channels of the diagonal of Z 4 .After this last step we have Z The errors i depend on the approximation error of the MLP to the relevant function, the previous errors i−1 (for i > 1), and uniform bounds as-well as uniform continuity of the approximated functions.

Corollary 1.

Our model can represent any message passing network to an arbitrary precision on compact sets.

In other words, in terms of universality our model is at-least as powerful as any message passing neural network (MPNN) that falls into the framework of BID9 .

@highlight

The paper provides a full characterization of permutation invariant and equivariant linear layers for graph data.
Convolution acts as a local feature extractor in convolutional neural networks (CNNs).

However, the convolution operation is not applicable when the input data is supported on an irregular graph such as with social networks, citation networks, or knowledge graphs.

This paper proposes the topology adaptive graph convolutional network (TAGCN), a novel graph convolutional network that generalizes CNN architectures to graph-structured data and provides a systematic way to design a set of fixed-size learnable filters to perform convolutions on graphs.

The topologies of these filters are adaptive to the topology of the graph when they scan the graph to perform convolution, replacing the square filter for the grid-structured data in traditional CNNs.

The outputs are the weighted sum of these filters’ outputs, extraction of both vertex features and strength of correlation between vertices.

It can be used with both directed and undirected graphs.

The proposed TAGCN not only inherits the properties of convolutions in CNN for grid-structured data, but it is also consistent with convolution as defined in graph signal processing.

Further, as no approximation to the convolution is needed, TAGCN exhibits better performance than existing graph-convolution-approximation methods on a number of data sets.

As only the polynomials of degree two of the adjacency matrix are used, TAGCN is also computationally simpler than other recent methods.

Convolutional neural network (CNN) architectures exhibit state-of-the-art performance on a variety of learning tasks dealing with 1D, 2D, and 3D grid-structured data such as acoustic signals, images, and videos, in which convolution serves as a feature extractor BID12 .

However, the (usual) convolution operation is not applicable when applying CNN to data that is supported on an arbitrary graph rather than on a regular grid structure, since the number of neighbors of each vertex on the graph varies, and it is difficult to design a fixed-size filter scanning over the graph-structured data for feature extraction.

Recently, there has been an increasing interest in graph CNNs (Bruna et al., 2014; BID3 BID10 BID13 , attempting to generalize deep learning methods to graph-structured data, specifically focusing on the design of graph CNN .

In this paper, we propose the topology adaptive graph convolutional network (TAGCN), a unified convolutional neural network to learn nonlinear representations for the graph-structured data.

It slides a set of fixed-size learnable filters on the graph simultaneously, and the output is the weighted sum of these filters' outputs, which extract both vertex features and strength of correlation between vertices.

Each filter is adaptive to the topology of the local region on the graph where it is applied.

TAGCN unifies filtering in both the spectrum and vertex domains; and applies to both directed and undirected graphs.

In general, the existing graph CNNs can be grouped into two types: spectral domain techniques and vertex domain techniques.

In Bruna et al. (2014) , CNNs have been generalized to graph-structured data, where convolution is achieved by a pointwise product in the spectrum domain according to the convolution theorem.

Later, BID3 and BID13 proposed spectrum filtering based methods that utilize Chebyshev polynomials and Cayley polynomials, respectively.

The assumption of symmetric adjacency matrix in these spectrum based methods restrict the application to undirected graphs.

BID10 simplified this spectrum method and obtained a filter in the vertex domain, which achieves state-of-the-art performance.

Other researchers BID0 worked on designing feature propagation models in the vertex domain for graph CNNs.

BID22 ; BID2 ; Grover & Leskovec (2016) ; BID5 study transforming graph-structured data to embedding vectors for learning problems.

Nevertheless, it still remains open how to extend CNNs from grid-structured data to arbitrary graph-structured data with local feature extraction capability.

This paper proposes a modification to the graph convolution step in CNNs that is particularly relevant for graph structured data.

Our proposed TAGCN is graph-based convolution and draws on techniques from graph signal processing.

We define rigorously the graph convolution operation on the vertex domain as multiplication by polynomials of the graph adjacency matrix, which is consistent with the notion of convolution in graph signal processing.

In graph signal processing, polynomials of the adjacency matrix are graph filters, extending to graph based data from the usual concept of filters in traditional time or image based signal processing.

Thus, comparing ours with existing work on graph CNNs, our paper provides a solid theoretical foundation for our proposed convolution step instead of an ad-hoc approach to convolution in CNNs for graph structured data.

Further, our method avoids computing the spectrum of the graph Laplacian as in Bruna et al. (2014) , or approximating the spectrum using high degree Chebyshev polynomials of the graph Laplacian matrix (in BID3 , it is suggested that one needs a 25 th degree Chebyshev polynomial to provide a good approximation to the graph Laplacian spectrum) or using high degree Cayley polynomials of the graph Laplacian matrix (in BID13 , 12 th degree Cayley polynomials are needed).

We also clarify that the GCN method in BID10 is a first order approximation of the Chebyshev polynomials approximation in BID3 , which is very different from our method.

Our method has a much lower computational complexity than the complexity of the methods proposed in Bruna et al. (2014) ; BID3 ; BID13 , since our method only uses polynomials of the adjacency matrix with maximum degree 2 as shown in our experiments.

Finally, the method that we propose exhibits better performance than existing methods.

Our contributions are summarized follows:• We propose a general K-localized filter for graph convolution in the vertex domain to extract local features on a set of size-1 up to size-K receptive fields.

The topologies of these filters are adaptive to the topology of the graph as they scan the graph to perform convolution.

It replaces the fixed square filters in traditional CNNs for the input gridstructured data volumes in traditional CNNs.

Thus, our convolution definition that we use in the convolution step for the vertex domain is consistent with convolution in traditional CNNs.• TAGCN is based on the graph signal processing and it is consistent with the convolution in graph signal processing.

It applies to both directed and undirected graphs.

Moreover, it has a much lower computational complexity compared with recent methods since it only needs polynomials of the adjacency matrix with maximum degree 2 compared with the 25 th and 12 th degree Laplacian matrix polynomials in BID3 and BID13 .•

As no approximation to the convolution is needed in TAGCN, it achieves better performance compared with existing methods.

We contrast TAGCN with recently proposed graph CNN including both spectrum filtering methods (Bruna et al., 2014; BID3 and vertex domain propagation methods BID10 BID0 , evaluating their performances on three commonly used data sets for graph vertices classification.

Our experimental tests show that TAGCN outperforms consistently all other approaches for each of these data sets.

We use boldface uppercase and lowercase letters to represent matrices and vectors, respectively.

The information and their relationship on a graph G can be represented by G = (V, E,Ā), where V is the set of vertices, E is the set of edges, andĀ is the weighted adjacency matrix of the graph; the graph can be weighted or unweighted, directed or undirected.

We assume there is no isolated vertex in G. If G is a directed weighted graph, the weightĀ n,m is on the directed edge from vertex m to n. The entryĀ n,m reveals the dependency between node n and m and can take arbitrary real or complex values.

The graph convolution is general and can be adapted to graph CNNs for particular tasks.

In this paper, we focus on the vertex semisupervised learning problem, where we have access to limited labeled vertices, and the task is to classify the remaining unlabeled vertices.

Without loss of generality, we demonstrate graph convolution on the -th hidden layer.

The results apply to any other hidden layers.

Suppose on the -th hidden layer, the input feature map for each vertex of the graph has C features.

We collect the -th hidden layer input data on all vertices for the c-th feature by the vector x DISPLAYFORM0 where DISPLAYFORM1 f is a learnable bias, and 1 N is the N dimension vector of all ones.

We design DISPLAYFORM2 c is a meaningful convolution on a graph with arbitrary topology.

In the recent theory on graph signal processing BID16 , the graph shift is defined as a local operation that replaces a graph signal at a graph vertex by a linear weighted combination of the values of the graph signal at the neighboring vertices: DISPLAYFORM3 The graph shiftĀ extends the time shift in traditional signal processing to graph-structured data.

Following BID16 , a graph filter G ( ) c,f is shift-invariant, i.e., the shiftĀ and the filter G DISPLAYFORM4 In (2), the g DISPLAYFORM5 3 We adopt the normalized adjacency matrix to guarantee that all the eigenvalues of A are inside the unit circle, and therefore G ( ) c,f is computationally stable.

The next subsection shows we will adopt 1 × C , 2 × C , . . .

, and K × C filters sliding on the graph-structured data.

This fact coincides with GoogLeNet BID21 , in which a set of filters with different sizes are used in each convolutional layer.

Following the CNN architecture, an additional nonlinear operation, e.g, rectified linear unit (ReLU) is used after every graph convolution operation: DISPLAYFORM6 where σ(·) denotes the ReLU activation function applied to the vertex values.

In the following subsection, we demonstrate that convolution operator shown in (2) exhibits the nice properties of a square filter in traditional CNNs and generalize the convolution theorem from classical signal processing to signal processing on graphs BID16 .

In this section, we would like to understand the proposed convolution as a feature extraction operator in traditional CNN rather than as propagating labeled data on the graph.

Taking this point of view helps us to profit from the design knowledge/experience from traditional CNN and apply it to grid structured data.

Our definition of weight of a path and the following filter size for graph convolution in this section make it possible to design a graph CNN architecture similar to GoogLeNet BID21 , in which a set of filters with different sizes are used in each convolutional layer.

In fact, we found that a combination of size 1 and size 2 filters gives the best performance in all three data sets studied, which is a polynomial with maximum order 2.In traditional CNN, a K × K × C filter scans over the input grid-structured data for feature extraction.

For image classification problems, the value K varies for different CNN architectures and tasks to achieve better performance.

For example, in VGG-Verydeep-16 CNN model BID20 , only 3 × 3 × C filters are used; in ImageNet CNN model BID11 , 11 × 11 × C filters are adopted; and in GoogLeNet BID21 , rather than using the same size filter in each convolutional layer, different size filters, for example, 1 × 1 × C , 3 × 3 × C and 5 × 5 × C filters, are concatenated in each convolution layer.

Similarly, we propose a general K-localized filter for graph CNN.For a graph-structured data, we cannot use a square filter window since the graph topology is no longer a grid.

In the following, we demonstrate that the convolution operation G DISPLAYFORM0 k is equivalent to using a set of filters with filter size from 1 up to K. Each k-size filter, which is used for local feature extraction on the graph, is k-localized in the vertex domain.

Define a path of length m on a graph G as a sequence DISPLAYFORM1 Here one path may visit the same vertex or cross the same edge multiple times.

The following adjacency matrix A is one such example: DISPLAYFORM2 DISPLAYFORM3 Since A is asymmetric, it represents a directed graph, given in FIG5 .

In this example, there are 6 different length 3-paths on the graph from vertex 2 to vertex 1, namely, (2, 1, 4, 1), (2, 1, 2, 1), (2, 1, 5, 1), (2, 3, 2, 1), (2, 4, 2, 1), and (2, 4, 5, 1).We further define the weight of a path to be the product of the edge weights along the path, i.e., DISPLAYFORM4 For example, the weight of the path (2, 1, 4, 1) is 1 × 1 × 2 = 2.

Then, the (i, j)th entry of A k in (2), denoted by ω(p k j,i ), can be interpreted as the sum of the weights of all the length-k paths from j to i, which is DISPLAYFORM5 In the above example, it can be easily verified that A 3 1,2 = 18 by summing up the weights of all the above six paths from vertex 2 to vertex 1 with length 3.

Then, the ith component of DISPLAYFORM6 c is the weighted sum of the input features of each vertex x ( ) c that are length-k paths away to vertex i. Here, k is defined as the filter size.

The output feature map is a vector with each component given by the size-k filter sliding on the graph following a fixed order of the vertex indices.

The output at the i-th component can be written explicitly as DISPLAYFORM7 This weighted sum is similar to the dot product for convolution for a grid-structured data in traditional CNNs.

Finnaly, the output feature map is a weighted sum of convolution results from filters with different sizes, which is DISPLAYFORM8 The above equation shows that each neuron in the graph convolutional layer is connected only to a local region (local vertices and edges) in the vertex domain of the input data volume, which is adaptive to the graph topology.

The strength of correlation is explicitly utilized in ω(p k j,i ).

FIG3 summarizes the graph convolution operation at each neuron.

We refer to this method as topology adaptive graph convolutional network (TAGCN).In FIG5 , we show TAGCN with an example of 2-size filter sliding from vertex 1 (figure on the lefthand-side) to vertex 2 (figure on the right-hand-side).

The filter is first placed at vertex 1.

Since paths (1, 2, 1) (5, 4, 1) and so on (paths with red glow) are all 2-length paths to vertex 1, they are covered by this 2-size filter.

Since paths (2, 3) and (7, 3) are not on any 2-length path to vertex 1, they are not covered by this filter.

Further, when this 2-size filter moves to vertex 2, paths (1, 5), (4, 5) and (6, 5) are no longer covered, but paths (2, 3) and (7, 3) are first time covered and contribute to the convolution with output at vertex 2.

Though the convolution operator defined in (2) is defined on the vertex domain, it can also be understood as a filter in the spectrum domain, and it is consistent with the definition of convolution in graph signal processing.

We provide detailed discussion in the appendix.

Equation FORMULA30 shows that TAGCN is equivalent to performing graph convolution either on the vertex domain or on the spectrum domain.

We next show that it is preferred to implement TAGCN in the vertex domain.

To obtain F, F −1 and J, the eigendecomposition for a diagonalizable matrix is in general O(N 3 ).

If the graph is directed, A may only be block diagonalizable, needing the Jordan decomposition, which is sensitive to round-off errors.

Additional matrix and vector multiplications may be needed to project the filtered out results from the spectrum domain back to the vertex domain.

In contrast, filtering directly in the vertex domain that involves powers of the graph adjacency matrix with N vertices and M edges can be computed in O(M N ) by performing a breadth-first search starting from each vertex to determine the distances to all other vertices BID8 .

For a sparse graph, which is often the case in practical problems, M N .

Since the motivation for performing graph convolution is to detect local features for graph-structured data, the optimal K usually is very small.

The experiments section shows that a filter size K = 2 leads to the best performance is selected for the data sets and specific tasks tested.

In summary, using polynomial filters in the vertex domain rather than in the spectrum domain saves significantly computational costs.

In this section, we show the connections and differences between the proposed TAGCN and existing methods.

In general, there are two types of graph convolution operators for the CNN architecture.

One defines the convolution in the spectrum domain, whose output feature map is the multiplication of the inverse Fourier transform matrix with the filtered results in the spectrum domain (Bruna et al., 2014; BID3 BID13 .

By doing further approximations based on this spectrum domain operator, a simplified convolution was obtained in Kipf & Welling BID20 .

The other defines convolution by a feature propagation model in the vertex domain such as MoNet in and the diffusion CNN (DCNN) in BID0 .

We investigate in detail each alternative.

In Bruna et al. (2014) ; BID3 ; BID13 , the convolution operation was defined using the convolution theorem and filtering operation in the spectrum domain by computing the eigendecomposition of the normalized Laplacian matrix of the graph.

The Laplacian matrix L is defined as L = D − A with the further assumption that A is symmetric to guarantee that L is positive semi-definite.

The convolution defined by the multiplication in the spectrum domain is approximated by BID3 by DISPLAYFORM0 where DISPLAYFORM1 is the kth order matrix Chebyshev polynomial BID18 where DISPLAYFORM2 with the initial values defined as DISPLAYFORM3 We refer later to this method as ChebNet for performance comparison.

Note that the Laplacian matrix can be seen as a differentiator operator.

The assumption of symmetric A restricts the application to undirected graphs.

Note that in BID3 , Laplacian matrix polynomials with maximum order K = 25 is needed to approximate the convolution operation on the left-hand side in (4), which imposes the computational burden.

While TAGCN only needs an adjacency matrix polynomials with maximum order 2 to achieve better performance as shown in the experiment part.

In BID10 , a graph convolutional network (GCN) was obtained by a first order approximation of (4).

In particular, let K = 1 and make the further assumptions that λ max = 2 and θ 0 = θ 1 = θ.

Then a simpler convolution operator that does not depend on the spectrum knowledge is obtained as DISPLAYFORM4 2 is a matrix with eigenvalues in [0, 2] .

A renormalization trick is adopted here by letting A = A + I and D i,i = j A i,j .

Finally, the convolutional operator is approximated by DISPLAYFORM5 where DISPLAYFORM6 It is interesting to observe that this method though obtained by simplifying the spectrum method has a better performance than the spectrum method BID3 .

The reason may be because the simplified form is equivalent to propagating vertex features on the graph, which can be seen as a special case of our TAGCN method, though there are other important differences.

Though our TAGCN is able to leverage information at a farther distance, it is not a simple extension of GCN Kipf & Welling BID20 .

First, the graph convolution in GCN is defined as a first order Chebyshev polynomial of the graph Laplacian matrix, which is an approximation to the graph convolution defined in the spectrum domain in BID3 .

In contrast, our graph convolution is rigorously defined as multiplication by polynomials of the graph adjacency matrix; this is not an approximation, rather, it simply is filtering with graph filters as defined and as being consistent with graph signal processing.

Next, we show the difference between our work and the GCN method in Kipf & Welling BID20 when using 2nd order (K=2, 2 steps away from the central node) Chebyshev polynomials of Laplacian matrix.

In the GCN paper Kipf & Welling BID20 , it has been shown that DISPLAYFORM7 2 by the definition of Chebyshev polynomial.

Then, extending GCN to the second order Chebyshev polynomials (two steps away from a central node) can be obtained from the original definition in T. Kipfs GCN (Kipf & Welling, 2017, eqn (5) DISPLAYFORM8 2 − I, which is different from our definition as in (2).

Thus, it is evident that our method is not a simple extension of GCN.

We apply graph convolution as proposed from basic principles in the graph signal processing, with no approximations involved, while both GCN in BID10 and BID3 BID13 are based on approximating the convolution defined in the spectrum domain.

In our approach, the degree of freedom is the design of the graph filter-its degree and its coefficients.

Ours is a principled approach and provides a generic methodology.

The performance gains we obtain are the result of capturing the underlying graph structure with no approximation in the convolution operation.

Simonovsky & Komodakis (2017) proposed the edge convolution network (ECC) to extend the convolution operator from regular grids to arbitrary graphs.

The convolution operator is defined similarly to (6) as DISPLAYFORM9 with Θ ( ) j,i is the weight matrix that needs to be learned.

A mixture model network (MoNet) was proposed in , with convolution defined as DISPLAYFORM10 where κ f is a Gaussian kernel with DISPLAYFORM11 f (u − µ f ) and g f is the weight coefficient for each Gaussian kernel κ f .

It is further assumed that Σ f is a 2 × 2 diagonal matrix.

GCN, ECC, and MoNet all design a propagation model on the graph; their differences are on the weightings used by each model.

Table 1 : Summary of number of weights need to be learned for the -th layer.

FORMULA0 proposed a diffusion CNN (DCNN) method that considers a diffusion process over the graph.

The transition probability of a random walk on a graph is given by P = D −1 A, which is equivalent to the normalized adjacency matrix.

DISPLAYFORM12 DISPLAYFORM13 By comparing the above methods with TAGCN in (3), it can be concluded that GCN, ECC and MoNet can be seen as a special case of TAGCN because in (3) the item with k = 1 can be seen as an information propagation term.

However, the strength of the correlation between vertices that are taken into account in ω(p k j,i ) in (3) can not be utilized in the GCN, ECC and MoNet methods.

Further, the TAGCN method is a systematic way to design a set of fixed size filters that is adaptive to the input graph topology when performing convolution on the graph.

Compared with existing spectrum methods (Bruna et al., 2014; BID3 BID13 , TAGCN satisfies the convolution theorem as shown in the previous subsection and implemented in the vertex domain, which avoids performing costly and practically numerical unstable eigendecompositions.

We further compare the number of weights that need to be learned in each hidden layer for these different methods in Table 1 .

As we show later, K = 2 is selected in our experiments using cross validation.

However, for ChebNet in BID3 , it is suggested that one needs a 25 th degree Chebyshev polynomial to provide a good approximation to the graph Laplacian spectrum.

Thus we have a moderate number of weights to be learned.

In the following, we show that our method achieves the best performance for each of those commonly used graph-structured data sets.

The proposed TAGCN is general and can be fit to the general graph CNN architectures for different tasks.

In the experiments, we focus on the vertex semisupervised learning problem, where we have access to only a few labeled vertices, and the task is to classify the remaining unlabeled vertices.

To compare the performance of TAGCN with that of existing methods, we extensively evaluate TAGCN on three graph-structured datasets, including the Cora, Citesser and Pubmed datasets.

The datasets split and experiments setting closely follow the standard criteria in BID22 .

TAGCN is evaluated on three data sets coming from citation networks: Citeseer, Cora, and Pubmed.

We closely follow the data set split and experimental setup in BID22 .

Each data set consists of a certain classes of documents, yet only a few documents are labeled.

The task is to classify the documents in the test set with these limited number of labels.

In each data set, the vertices are the documents and the edges are the citation links.

Each document is represented by sparse bagof-words feature vectors, and the citation links between documents are provided.

Detailed statistics of these three data sets are summarized in TAB0 .

It shows the number of nodes and edges that corresponding to documents and citation links.

Classes for the graph vertices shows the number of document classes in each data set.

Also, the number of features at each vertex is given.

Label rate denotes the number of labeled documents that are used for training divided by the total number of documents in each data set.

We construct a graph for each data set with nodes representing documents and undirected edges 4 linking two papers if there is a citation relationship.

We obtain the adjacency matrixĀ with 0 and 1 components and further obtain the normalized matrix A.In the following experiments, we design a TAGCN with two hidden layers (obtained from cross validation) for the semi-supervised node classification.

In each hidden layer, the proposed TAGCN is applied for convolution, followed by a ReLU activation.

16 hidden units (filters) are designed for each hidden layer, and dropout is applied after each hidden layer.

The softmax activation function is applied to the output of the second hidden layer for the final classification.

For ablation study, we evaluate the performance of TAGCN with different kernel size from 1 to 4.

To investigate the performance for different number of parameters, we also design a TAGCN with 8 filters for each hidden layer and compare its classification accuracy with all the baselines and TAGCN with 16 filters.

We train our model using Adam BID9 with a learning rate of 0.01 and early stopping with a window size of 20.

Hyperparameters of the networks (filter size, dropout rate, and number of hidden layers) are selected by cross validation.

To make a fair comparison, we closely follow the same split of training, validation, and testing sets as in BID10 ; BID22 , i.e., 500 labeled examples for hyperparameters (filter size, dropout rate, and number of hidden layers) optimization and cross-entropy error is used for classification accuracy evaluation.

The performance results of the proposed TAGCN method are an average over 100 runs with random initializations BID6 .

We compare the classification accuracy with other recently proposed graph CNN methods as well as a graph embedding method known as Planetoid BID22 .

The quantitative results are summarized in Table 3 .

Reported numbers denote classification accuracy in percent.

Results for Planetoid, GCN, and ChebNet are taken from BID10 , and results for DCNN and MoNet are taken from .

All the experiments for different methods are based on the same data statistics shown in TAB0 .

The datasets split and experiments settings closely follow the standard criteria in BID22 ; BID10 .

Table 3 shows that our method For ablation study, we further compare the performance of different filter sizes from K = 1 to K = 4 in TAB1 .

It shows that the performances for filter size K = 2 are always better than that for other filter sizes.

The value K = 1 gives the worst classification accuracy.

This further validates that a local feature extraction is more important than just propagating features among neighbors on the graph.

In TAB1 , we also compare the performance of different number of filters, which reflects different number of network parameters.

Note, we also choose filer size K = 2 and filter number F = 8 that results in the same number of network parameters as that in GCN, MoNet, ECC and DCNN according to Table 1 .

It shows that the classification accuracy using 8 filters is comparable with that using 16 filters in each hidden layer for TAGCN.

Moreover, TAGCN with 8 filters can still achieve higher accuracy than GCN, MoNet, ECC and DCNN methods.

This proves that, even with a similar number of parameters or architecture, our method still exhibits superior performance than GCN.As we explained in Section 3, TAGCN in our paper is not simply extending GCN BID10 to k-th order.

Nevertheless, we implement A 2 and compare its performance with ours.

For the data sets Pubmed, Cora, and Citeseer, the classification accuracies are 79.1(81.1), 81.7(82.5) and 70.8(70.9), where the numbers in parentheses are the results obtained with our method.

Our method still achieves a noticeable performance advantage over A 2 for the Pubmed and Cora data; in particular, we note the significant performance gain with the Pubmed database that has the largest number of nodes among these three data sets.

We have defined a novel graph convolutional network that rearchitects the CNN architecture for graph-structured data.

The proposed method, known as TAGCN, is adaptive to the graph topology as the filter scans the graph.

Further, TAGCN inherits properties of the convolutional layer in classical CNN, i.e., local feature extraction and weight sharing.

It can further extract the strength of correlation between vertices in the filtering region.

On the other hand, by the convolution theorem, TAGCN that implements in the vertex domain offers implement in the spectrum domain unifying graph CNN in both the spectrum domain and the vertex domain.

TAGCN is consistent with convolution in graph signal processing.

These nice properties lead to a noticeable performance advantage in classification accuracy on different graph-structured datasets for semi-supervised graph vertex classification problems with low computational complexity.

In classical signal processing BID15 , the convolution in the time domain is equivalent to multiplication in the spectrum domain.

This relationship is known as the convolution theorem.

BID16 showed that the graph filtering defined in the vertex domain satisfies the generalized convolution theorem naturally and can also interpret spectrum filtering for both directed and undirected graphs.

Recent work (Bruna et al., 2014; BID3 used the convolution theorem for undirected graph-structured data and designed a spectrum graph filtering.

The expression on the left-hand-side of the above equation represents the filtering/convolution on the vertex domain.

Matrix F defines the graph Fourier transform BID16 BID17 , and Fx ( ) c is the input feature spectrum map, which is a linear mapping from the input feature on the vertex domain to the spectrum domain.

The polynomial K k=0 g ( ) c,f,k J k is the spectrum of the graph filter.

Relation (7), which is equation (27) in BID16 generalizes the classical convolution theorem to graph-structured data: convolution/filtering on the vertex domain becomes multiplication in the spectrum domain.

When the graph is in the 1D cyclic form, as shown in FIG6 , the corresponding adjacency matrix is of the form where DFT is the discrete Fourier transform matrix.

The convolution operator defined in (2) is consistent with that in classical signal processing.

@highlight

Low computational complexity graph CNN (without approximation) with better classification accuracy

@highlight

Proposes a new CNN approach to graph classification using a filter based on outgoing walks of increasing length to incorporate information from more distant vertices in one propagation step.

@highlight

Proposal for a new neural network architecture for semi-supervised graph classification, building upon graph polynomial filters and utilizing them on successive neural network layers with ReLU activation functions.

@highlight

The paper introduces Topology Adaptive GCN to generalize convolutional networks to graph-structured data
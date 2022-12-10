We present graph wavelet neural network (GWNN), a novel graph convolutional neural network (CNN), leveraging graph wavelet transform to address the shortcomings of previous spectral graph CNN methods that depend on graph Fourier transform.

Different from graph Fourier transform, graph wavelet transform can be obtained via a fast algorithm without requiring matrix eigendecomposition with high computational cost.

Moreover, graph wavelets are sparse and localized in vertex domain, offering high efficiency and good interpretability for graph convolution.

The proposed GWNN significantly outperforms previous spectral graph CNNs in the task of graph-based semi-supervised classification on three benchmark datasets: Cora, Citeseer and Pubmed.

Convolutional neural networks (CNNs) BID15 have been successfully used in many machine learning problems, such as image classification BID10 and speech recognition BID11 , where there is an underlying Euclidean structure.

The success of CNNs lies in their ability to leverage the statistical properties of Euclidean data, e.g., translation invariance.

However, in many research areas, data are naturally located in a non-Euclidean space, with graph or network being one typical case.

The non-Euclidean nature of graph is the main obstacle or challenge when we attempt to generalize CNNs to graph.

For example, convolution is not well defined in graph, due to that the size of neighborhood for each node varies dramatically .Existing methods attempting to generalize CNNs to graph data fall into two categories, spatial methods and spectral methods, according to the way that convolution is defined.

Spatial methods define convolution directly on the vertex domain, following the practice of the conventional CNN.

For each vertex, convolution is defined as a weighted average function over all vertices located in its neighborhood, with the weighting function characterizing the influence exerting to the target vertex by its neighbors .

The main challenge is to define a convolution operator that can handle neighborhood with different sizes and maintain the weight sharing property of CNN.

Although spatial methods gain some initial success and offer us a flexible framework to generalize CNNs to graph, it is still elusive to determine appropriate neighborhood.

Spectral methods define convolution via graph Fourier transform and convolution theorem.

Spectral methods leverage graph Fourier transform to convert signals defined in vertex domain into spectral domain, e.g., the space spanned by the eigenvectors of the graph Laplacian matrix, and then filter is defined in spectral domain, maintaining the weight sharing property of CNN.

As the pioneering work of spectral methods, spectral CNN BID3 exploited graph data with the graph Fourier transform to implement convolution operator using convolution theorem.

Some subsequent works make spectral methods spectrum-free BID4 BID14 BID12 , achieving locality in spatial domain and avoiding high computational cost of the eigendecomposition of Laplacian matrix.

In this paper, we present graph wavelet neural network to implement efficient convolution on graph data.

We take graph wavelets instead of the eigenvectors of graph Laplacian as a set of bases, and define the convolution operator via wavelet transform and convolution theorem.

Graph wavelet neural network distinguishes itself from spectral CNN by its three desirable properties: (1) Graph wavelets can be obtained via a fast algorithm without requiring the eigendecomposition of Laplacian matrix, and thus is efficient; (2) Graph wavelets are sparse, while eigenvectors of Laplacian matrix are dense.

As a result, graph wavelet transform is much more efficient than graph Fourier transform; (3) Graph wavelets are localized in vertex domain, reflecting the information diffusion centered at each node BID27 .

This property eases the understanding of graph convolution defined by graph wavelets.

We develop an efficient implementation of the proposed graph wavelet neural network.

Convolution in conventional CNN learns an individual convolution kernel for each pair of input feature and output feature, causing a huge number of parameters especially when the number of features is high.

We detach the feature transformation from convolution and learn a sole convolution kernel among all features, substantially reducing the number of parameters.

Finally, we validate the effectiveness of the proposed graph wavelet neural network by applying it to graph-based semi-supervised classification.

Experimental results demonstrate that our method consistently outperforms previous spectral CNNs on three benchmark datasets, i.e., Cora, Citeseer, and Pubmed.2 OUR METHOD 2.1 PRELIMINARY Let G = {V, E, A} be an undirected graph, where V is the set of nodes with |V| = n, E is the set of edges, and A is adjacency matrix with A i,j = A j,i to define the connection between node i and node j.

The graph Laplacian matrix L is defined as L = D −A where D is a diagonal degree matrix with D i,i = j A i,j , and the normalized Laplacian matrix is L = I n − D −1/2 AD −1/2 where I n is the identity matrix.

Since L is a real symmetric matrix, it has a complete set of orthonormal eigenvectors U = (u 1 , u 2 , ..., u n ), known as Laplacian eigenvectors.

These eigenvectors have associated real, non-negative eigenvalues {λ l } n l=1 , identified as the frequencies of graph.

Eigenvectors associated with smaller eigenvalues carry slow varying signals, indicating that connected nodes share similar values.

In contrast, eigenvectors associated with larger eigenvalues carry faster varying signals across connected nodes.

Taking the eigenvectors of normalized Laplacian matrix as a set of bases, graph Fourier transform of a signal x ∈ R n on graph G is defined asx = U x, and the inverse graph Fourier transform is x = Ux BID24 .

Graph Fourier transform, according to convolution theorem, offers us a way to define the graph convolution operator, denoted as * G .

Denoting with y the convolution kernel, * G is defined as DISPLAYFORM0 where is the element-wise Hadamard product.

Replacing the vector U y by a diagonal matrix g θ , then Hadamard product can be written in the form of matrix multiplication.

Filtering the signal x by the filter g θ , we can write Equation (1) as U g θ U x.

However, there are some limitations when using Fourier transform to implement graph convolution:(1) Eigendecomposition of Laplacian matrix to obtain Fourier basis U is of high computational cost with O(n 3 ); (2) Graph Fourier transform is inefficient, since it involves the multiplication between a dense matrix U and the signal x; (3) Graph convolution defined through Fourier transform is not localized in vertex domain, i.e., the influence to the signal on one node is not localized in its neighborhood.

To address these limitations, ChebyNet BID4 restricts convolution kernel g θ to a polynomial expansion DISPLAYFORM1 where K is a hyper-parameter to determine the range of node neighborhoods via the shortest path distance, θ ∈ R K is a vector of polynomial coefficients, and Λ =diag({λ l } n l=1 ).

However, such a polynomial approximation limits the flexibility to define appropriate convolution on graph, i.e., with a smaller K, it's hard to approximate the diagonal matrix g θ with n free parameters.

While with a larger K, locality is no longer guaranteed.

Different from ChebyNet, we address the aforementioned three limitations through replacing graph Fourier transform with graph wavelet transform.

Similar to graph Fourier transform, graph wavelet transform projects graph signal from vertex domain into spectral domain.

Graph wavelet transform employs a set of wavelets as bases, defined as ψ s = (ψ s1 , ψ s2 , ..., ψ sn ), where each wavelet ψ si corresponds to a signal on graph diffused away from node i and s is a scaling parameter.

Mathematically, ψ si can be written as DISPLAYFORM0 where U is Laplacian eigenvectors, G s =diag g(sλ 1 ), ..., g(sλ n ) is a scaling matrix and g(sλ i ) = e λis .Using graph wavelets as bases, graph wavelet transform of a signal x on graph is defined asx = ψ −1 s x and the inverse graph wavelet transform is x = ψ sx .

Note that ψ −1 s can be obtained by simply replacing the g(sλ i ) in ψ s with g(−sλ i ) corresponding to a heat kernel BID6 .

Replacing the graph Fourier transform in Equation FORMULA0 with graph wavelet transform, we obtain the graph convolution as DISPLAYFORM1 Compared to graph Fourier transform, graph wavelet transform has the following benefits when being used to define graph convolution: 2.

High spareness: the matrix ψ s and ψ −1 s are both sparse for real world networks, given that these networks are usually sparse.

Therefore, graph wavelet transform is much more computationally efficient than graph Fourier transform.

For example, in the Cora dataset, more than 97% elements in ψ −1 s are zero while only less than 1% elements in U are zero TAB3 .

3.

Localized convolution: each wavelet corresponds to a signal on graph diffused away from a centered node, highly localized in vertex domain.

As a result, the graph convolution defined in Equation FORMULA3 is localized in vertex domain.

We show the localization property of graph convolution in Appendix A. It is the localization property that explains why graph wavelet transform outperforms Fourier transform in defining graph convolution and the associated tasks like graph-based semisupervised learning.

4.

Flexible neighborhood: graph wavelets are more flexible to adjust node's neighborhoods.

Different from previous methods which constrain neighborhoods by the discrete shortest path distance, our method leverages a continuous manner, i.e., varying the scaling parameter s. A small value of s generally corresponds to a smaller neighborhood.

FIG0 shows two wavelet bases at different scale on an example network, depicted using GSP toolbox BID22 .

Replacing Fourier transform with wavelet transform, graph wavelet neural network (GWNN) is a multi-layer convolutional neural network.

The structure of the m-th layer is DISPLAYFORM0 where ψ s is wavelet bases, ψ −1 s is the graph wavelet transform matrix at scale s which projects signal in vertex domain into spectral domain, X m [:,i] with dimensions n × 1 is the i-th column of X m , F m i,j is a diagonal filter matrix learned in spectral domain, and h is a non-linear activation function.

This layer transforms an input tensor X m with dimensions n × p into an output tensor X m+1 with dimensions n × q.

In this paper, we consider a two-layer GWNN for semi-supervised node classification on graph.

The formulation of our model is DISPLAYFORM1 second layer : DISPLAYFORM2 where c is the number of classes in node classification, Z of dimensions n × c is the prediction result.

The loss function is the cross-entropy error over all labeled examples: DISPLAYFORM3 where y L is the labeled node set, Y li = 1 if the label of node l is i, and Y li = 0 otherwise.

The weights F are trained using gradient descent.

In Equation FORMULA4 , the parameter complexity of each layer is O(n × p × q), where n is the number of nodes, p is the number of features of each vertex in current layer, and q is the number of features of each vertex in next layer.

Conventional CNN methods learn convolution kernel for each pair of input feature and output feature.

This results in a huge number of parameters and generally requires huge training data for parameter learning.

This is prohibited for graph-based semi-supervised learning.

To combat this issue, we detach the feature transformation from graph convolution.

Each layer in GWNN is divided into two components: feature transformation and graph convolution.

Spectially, we have feature transformation : DISPLAYFORM0 graph convolution : DISPLAYFORM1 where W ∈ R p×q is the parameter matrix for feature transformation, X m with dimensions n × q is the feature matrix after feature transformation, F m is the diagonal matrix for graph convolution kernel, and h is a non-linear activation function.

After detaching feature transformation from graph convolution, the parameter complexity is reduced from O(n × p × q) to O(n + p × q).

The reduction of parameters is particularly valuable fro graphbased semi-supervised learning where labels are quite limited.

Graph convolutional neural networks on graphs.

The success of CNNs when dealing with images, videos, and speeches motivates researchers to design graph convolutional neural network on graphs.

The key of generalizing CNNs to graphs is defining convolution operator on graphs.

Existing methods are classified into two categories, i.e., spectral methods and spatial methods.

Spectral methods define convolution via convolution theorem.

Spectral CNN BID3 is the first attempt at implementing CNNs on graphs, leveraging graph Fourier transform and defining convolution kernel in spectral domain.

BID1 developed a local spectral CNN approach based on the graph Windowed Fourier Transform.

BID4 introduced a Chebyshev polynomial parametrization for spectral filter, offering us a fast localized spectral filtering method.

BID14 provided a simplified version of ChebyNet, gaining success in graph-based semi-supervised learning task.

BID12 represented images as signals on graph and learned their transformation invariant representations.

They used Chebyshev approximations to implement graph convolution, avoiding matrix eigendecomposition.

BID16 used rational functions instead of polynomials and created anisotropic spectral filters on manifolds.

Spatial methods define convolution as a weighted average function over neighborhood of target vertex.

GraphSAGE takes one-hop neighbors as neighborhoods and defines the weighting function as various aggregators over neighborhood BID8 .

Graph attention network (GAT) proposes to learn the weighting function via self-attention mechanism BID28 .

MoNet offers us a general framework for design spatial methods, taking convolution as the weighted average of multiple weighting functions defined over neighborhood .

Some works devote to making graph convolutional networks more powerful.

BID20 alternated convolutions on vertices and edges, generalizing GAT and leading to better performance.

GraphsGAN BID5 generalizes GANs to graph, and generates fake samples in low-density areas between subgraphs to improve the performance on graph-based semi-supervised learning.

Graph wavelets.

Sweldens (1998) presented a lifting scheme, a simple construction of wavelets that can be adapted to graphs without learning process.

BID9 proposed a method to construct wavelet transform on graphs.

Moreover, they designed an efficient way to bypass the eigendecomposition of the Laplacian and approximated wavelets with Chebyshev polynomials.

BID27 leveraged graph wavelets for multi-scale community mining by modulating a scaling parameter.

Owing to the property of describing information diffusion, BID6 learned structural node embeddings via wavelets.

All these works prove that graph wavelets are not only local and sparse but also valuable for signal processiong on graph.

To evaluate the proposed GWNN, we apply GWNN on semi-supervised node classification, and conduct experiments on three benchmark datasets, namely, Cora, Citeseer and Pubmed BID23 .

In the three citation network datasets, nodes represent documents and edges are citation links.

Details of these datasets are demonstrated in TAB0 .

Here, the label rate denotes the proportion of labeled nodes used for training.

Following the experimental setup of GCN (Kipf & Welling, 2017), we fetch 20 labeled nodes per class in each dataset to train the model.

We compare with several traditional semi-supervised learning methods, including label propagation (LP) BID31 , semi-supervised embedding (SemiEmb) BID29 , manifold regularization (ManiReg) BID0 , graph embeddings (DeepWalk) BID21 , iterative classification algorithm (ICA) BID17 and Planetoid BID30 .Furthermore, along with the development of deep learning on graph, graph convolutional networks are proved to be effective in semi-supervised learning.

Since our method is a spectral method based on convolution theorem, we compare it with the Spectral CNN BID3 .

ChebyNet BID4 and GCN (Kipf & Welling, 2017) , two variants of the Spectral CNN, are also included as our baselines.

Considering spatial methods, we take MoNet as our baseline, which also depends on Laplacian matrix.

We train a two-layer graph wavelet neural network with 16 hidden units, and prediction accuracy is evaluated on a test set of 1000 labeled samples.

The partition of datasets is the same as GCN BID14 with an additional validation set of 500 labeled samples to determine hyper-parameters.

Weights are initialized following BID7 .

We adopt the Adam optimizer (Kingma & Ba, 2014) for parameter optimization with an initial learning rate lr = 0.01.

For computational efficiency, we set the elements of ψ s and ψ −1 s smaller than a threshold t to 0.

We find the optimal hyper-parameters s and t through grid search, and the detailed discussion about the two hyperparameters is introduced in Appendix B. For Cora, s = 1.0 and t = 1e − 4.

For Citeseer, s = 0.7 and t = 1e − 5.

For Pubmed, s = 0.5 and t = 1e − 7.

To avoid overfitting, dropout BID25 is applied.

Meanwhile, we terminate the training if the validation loss does not decrease for 100 consecutive epochs.

Since the number of parameters for the undetached version of GWNN is O(n × p × q), we can hardly implement this version in the case of networks with a large number n of nodes and a huge number p of input features.

Here, we validate the effectiveness of detaching feature transformation form convolution on ChebyNet (introduced in Section 2.2), whose parameter complexity is O(K × p × q).

For ChebyNet of detaching feature transformation from graph convolution, the number of parameters is reduced to O(K + p × q).

TAB1 shows the performance and the number of parameters on three datasets.

Here, the reported performance is the optimal performance varying the order K = 2, 3, 4.

As demonstrated in TAB1 , with fewer parameters, we improve the accuracy on Pubmed by a large margin.

This is due to that the label rate of Pubmed is only 0.003.

By detaching feature transformation from convolution, the parameter complexity is significantly reduced, alleviating overfitting in semi-supervised learning and thus remarkably improving prediction accuracy.

On Citeseer, there is a little drop on the accuracy.

One possible explanation is that reducing the number of parameters may restrict the modeling capacity to some degree.

We now validate the effectiveness of GWNN with detaching technique on node classification.

Experimental results are reported in TAB2 .

GWNN improves the classification accuracy on all the three datasets.

In particular, replacing Fourier transform with wavelet transform, the proposed GWNN is comfortably ahead of Spectral CNN, achieving 10% improvement on Cora and Citeseer, and 5% improvement on Pubmed.

The large improvement could be explained from two perspectives: (1) Convolution in Spectral CNN is non-local in vertex domain, and thus the range of feature diffusion is not restricted to neighboring nodes; (2) The scaling parameter s of wavelet transform is flexible to adjust the diffusion range to suit different applications and different networks.

GWNN consistently outperforms ChebyNet, since it has enough degree of freedom to learn the convolution kernel, while ChebyNet is a kind of approximation with limited degree of freedom.

Furthermore, our GWNN also performs better than GCN and MoNet, reflecting that it is promising to design appropriate bases for spectral methods to achieve good performance.

Besides the improvement on prediction accuracy, wavelet transform with localized and sparse transform matrix holds sparsity in both spatial domain and spectral domain.

Here, we take Cora as an example to illustrate the sparsity of graph wavelet transform.

The sparsity of transform matrix.

There are 2,708 nodes in Cora.

Thus, the wavelet transform matrix ψ −1 s and the Fourier transform matrix U both belong to R 2,708×2,708 .

The first two rows in TAB3 demonstrate that ψ −1 s is much sparser than U .

Sparse wavelets not only accelerate the computation, but also well capture the neighboring topology centered at each node.

The sparsity of projected signal.

As mentioned above, each node in Cora represents a document and has a sparse bag-of-words feature.

The input feature X ∈ R n×p is a binary matrix, and X [i,j] = 1 when the i-th document contains the j-th word in the bag of words, it equals 0 otherwise.

Here, X [:,j] denotes the j-th column of X, and each column represents the feature vector of a word.

Considering a specific signal X [:,984] , we project the spatial signal into spectral domain, and get its projected vector.

Here, p = ψ −1 s X [:,984] denotes the projected vector via wavelet transform, q = U X [:,984] denotes the projected vector via Fourier transform, and p, q ∈ R 2,708 .

The last row in TAB3 lists the numbers of non-zero elements in p and q. As shown in TAB3 , with wavelet transform, the projected signal is much sparser.

Compare with graph convolution network using Fourier transform, GWNN provides good interpretability.

Here, we show the interpretability with specific examples in Cora.

Each feature, i.e. word in the bag of words, has a projected vector, and each element in this vector is associated with a spectral wavelet basis.

Here, each basis is centered at a node, corresponding to a document.

The value can be regarded as the relation between the word and the document.

Thus, each value in p can be interpreted as the relation between W ord 984 and a document.

In order to elaborate the interpretability of wavelet transform, we analyze the projected values of different feature as following.

Considering two features W ord 984 and W ord 1177 , we select the top-10 active bases, which have the 10 largest projected values of each feature.

As illustrated in Figure 2 , for clarity, we magnify the local structure of corresponding nodes and marked them with bold rims.

The central network in each subgraph denotes the dataset Cora, each node represents a document, and 7 different colors represent 7 classes.

These nodes are clustered by OpenOrd BID18 based on the adjacency matrix.

Figure 2a shows the top-10 active bases of W ord 984 .

In Cora, this word only appears 8 times, and all the documents containing W ord 984 belong to the class " Case-Based ".

Consistently, all top-10 nodes activated by W ord 984 are concentrated and belong to the class " Case-Based ".

And, the frequencies of W ord 1177 appearing in different classes are similar, indicating that W ord 1177 is a universal word.

In concordance with our expectation, the top-10 active bases of W ord 1177 are discrete and belong to different classes in Figure 2b .

DISPLAYFORM0 Figure 2: Top-10 active bases of two words in Cora.

The central network of each subgraph represents the dataset Cora, which is split into 7 classes.

Each node represents a document, and its color indicates its label.

The nodes that represent the top-10 active bases are marked with bold rims.

(a) W ord 984 only appears in documents of the class " Case-Based " in Cora.

Consistently, all its 10 active bases also belong to the class " Case-Based ".

(b) The frequencies of W ord 1177 appearing in different classes are similar in Cora.

As expected, the top-10 active bases of W ord 1177 also belong to different classes.

Owing to the properties of graph wavelets, which describe the neighboring topology centered at each node, the projected values of wavelet transform can be explained as the correlation between features and nodes.

These properties provide an interpretable domain transformation and ease the understanding of graph convolution.

Replacing graph Fourier transform with graph wavelet transform, we proposed GWNN.

Graph wavelet transform has three desirable properties: (1) Graph wavelets are local and sparse; (2) Graph wavelet transform is computationally efficient; (3) Convolution is localized in vertex domain.

These advantages make the whole learning process interpretable and efficient.

Moreover, to reduce the number of parameters and the dependence on huge training data, we detached the feature transformation from convolution.

This practice makes GWNN applicable to large graphs, with remarkable performance improvement on graph-based semi-supervised learning.

We use a diagonal matrix Θ to represent the learned kernel transformed by wavelets ψ −1 s y, and replace the Hadamard product with matrix muplication.

Then Equation FORMULA3 is: DISPLAYFORM0 We set ψ s = (ψ s1 , ψ s2 , ..., ψ sn ), ψ DISPLAYFORM1 ).

Equation FORMULA0 becomes : DISPLAYFORM2 As proved Since each M k is local, for any convolution kernel Θ, ψ s Θψ −1 s is local, and it means that convolution is localized in vertex domain.

By replacing Θ with an identity matrix in Equation FORMULA0 , we get x * G y = n k=1 M k x. We define H = n k=1 M k , and Figure 4 shows H [1,:] in different scaling, i.e., correlation between the first node and other nodes during convolution.

The locality of H suggests that graph convolution is localized in vertex domain.

Moreover, as the scaling parameter s becomes larger, the range of feature diffusion becomes larger.

GWNN leverages graph wavelets to implement graph convolution, where s is used to modulate the range of neighborhoods.

From Figure 5 , as s becomes larger starting from 0, the range of neighboring nodes becomes large, resulting the increase of accuracy on Cora.

However when s becomes too large, some irrelevant nodes are included, leading to decreasing of accuracy.

The hyperparameter t only used for computational efficiency, has any slight influence on its performance.

For experiments on specific dataset, s and t are choosen via grid search using validation.

Generally, a appropriate s is in the range of [0.5, 1], which can not only capture the graph structure but also guarantee the locality of convolution, and t is less insensive to dataset.

We show the parameter complexity of node classification in TAB4 .

The high parameter complexity O(n * p * q) of Spectral CNN makes it difficult to generalize to real world networks.

ChebyNet approximates the convolution kernel via polynomial function of the diagonal matrix of Laplacian eigenvalues, reducing parameter complexity to O(K * p * q) with K being the order of polynomial function.

GCN simplifies ChebyNet via setting K=1.

We detach feature transformation from graph convolution to implement GWNN and Spectral CNN in our experiments, which can reduce parameter to O(n + p * q).

In Cora and Citeseer, with smaller parameter complexity, GWNN achieves better performance than ChebyNet, reflecting that it is promising to implement convolution via graph wavelet transform.

As Pubmed has a large number of nodes, the parameter complexity of GWNN is larger than ChebyNet.

As future work, it is an interesting attempt to select wavelets associated with a subset of nodes, further reducing parameter complexity with potential loss of performance.

With the stable recurrence relation T k (y) = 2yT k−1 (y) − T k−2 (y), we can generate the Chebyshev polynomials T k (y).

Here T 0 = 1 and T 1 = y. For y sampled between -1 and 1, the trigonometric expression T k (y) = cos(k arccos (y) cos(kθ)g(s(a(cos(θ) + 1)))dθ.

we truncate the Chebyshev expansion to m terms and achieve Polynomial approximation.

Here we give the example of the ψ −1 s and g(sx) = e −sx , the graph signal is f ∈ R n .

Then we can give the fast approximation wavelets by ψ

The sparsity of the graph wavelets depends on the sparsity of the Laplacian matrix and the hyperparameter s, We show the sparsity of spectral transform matrix and Laplacian matrix in TAB5 .

The sparsity of Laplacian matrix is sparser than graph wavelets, and this property limits our method, i.e., the higher time complexity than some methods depending on Laplacian matrix and identity matrix, e.g., GCN.

Specifically, both our method and GCN aim to improve Spectral CNN via designing localized graph convolution.

GCN, as a simplified version of ChebyNet, leverages Laplacian matrix as weighted matrix and expresses the spectral graph convolution in spatial domain, acting as spatial-like method .

However, our method resorts to using graph wavelets as a new set of bases, directly designing localized spectral graph convolution.

GWNN offers a localized graph convolution via replacing graph Fourier transform with graph wavelet transform, finding good spectral basis with localization property and good interpretability.

This distinguishes GWNN from ChebyNet and GCN, which express the graph convolution defined via graph Fourier transform in vertex domain.

@highlight

We present graph wavelet neural network (GWNN), a novel graph convolutional neural network (CNN), leveraging graph wavelet transform to address the shortcoming of previous spectral graph CNN methods that depend on graph Fourier transform.
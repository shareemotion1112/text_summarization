The rise of graph-structured data such as social networks, regulatory networks, citation graphs, and functional brain networks, in combination with resounding success of deep learning in various applications, has brought the interest in generalizing deep learning models to non-Euclidean domains.

In this paper, we introduce a new spectral domain convolutional architecture for deep learning on graphs.

The core ingredient of our model is a new class of parametric rational complex functions (Cayley polynomials) allowing to efficiently compute spectral filters on graphs that specialize on frequency bands of interest.

Our model generates rich spectral filters that are localized in space, scales linearly with the size of the input data for sparsely-connected graphs, and can handle different constructions of Laplacian operators.

Extensive experimental results show the superior performance of our approach on spectral image classification, community detection, vertex classification and matrix completion tasks.

In many domains, one has to deal with large-scale data with underlying non-Euclidean structure.

Prominent examples of such data are social networks, genetic regulatory networks, functional networks of the brain, and 3D shapes represented as discrete manifolds.

The recent success of deep neural networks and, in particular, convolutional neural networks (CNNs) BID19 have raised the interest in geometric deep learning techniques trying to extend these models to data residing on graphs and manifolds.

Geometric deep learning approaches have been successfully applied to computer graphics and vision ; BID3 a) ; BID24 , brain imaging BID18 , and drug design BID10 problems, to mention a few.

For a comprehensive presentation of methods and applications of deep learning on graphs and manifolds, we refer the reader to the review paper BID4 .Related work.

The earliest neural network formulation on graphs was proposed by BID11 and BID27 , combining random walks with recurrent neural networks (their paper has recently enjoyed renewed interest in BID20 ; BID30 ).

The first CNN-type architecture on graphs was proposed by BID5 .

One of the key challenges of extending CNNs to graphs is the lack of vector-space structure and shift-invariance making the classical notion of convolution elusive.

Bruna et al. formulated convolution-like operations in the spectral domain, using the graph Laplacian eigenbasis as an analogy of the Fourier transform BID29 ).

BID13 used smooth parametric spectral filters in order to achieve localization in the spatial domain and keep the number of filter parameters independent of the input size.

BID8 proposed an efficient filtering scheme using recurrent Chebyshev polynomials applied on the Laplacian operator.

BID17 simplified this architecture using filters operating on 1-hop neighborhoods of the graph.

BID0 proposed a Diffusion CNN architecture based on random walks on graphs.

BID24 (and later, Hechtlinger et al. (2017) ) proposed a spatial-domain generalization of CNNs to graphs using local patch operators represented as Gaussian mixture models, showing a significant advantage of such models in generalizing across different graphs.

In BID25 , spectral graph CNNs were extended to multiple graphs and applied to matrix completion and recommender system problems.

Main contribution.

In this paper, we construct graph CNNs employing an efficient spectral filtering scheme based on Cayley polynomials that enjoys similar advantages of the Chebyshev filters BID8 ) such as localization and linear complexity.

The main advantage of our filters over BID8 is their ability to detect narrow frequency bands of importance during training, and to specialize on them while being well-localized on the graph.

We demonstrate experimentally that this affords our method greater flexibility, making it perform better on a broad range of graph learning problems.

Notation.

We use a, a, and A to denote scalars, vectors, and matrices, respectively.z denotes the conjugate of a complex number, Re{z} its real part, and i is the imaginary unit.

diag(a 1 , . . . , a n ) denotes an n×n diagonal matrix with diagonal elements a 1 , . . .

, a n .

Diag(A) = diag(a 11 , . . . , a nn ) denotes an n × n diagonal matrix obtained by setting to zero the off-diagonal elements of A. Off(A) = A − Diag(A) denotes the matrix containing only the off-diagonal elements of A. I is the identity matrix and A • B denotes the Hadamard (element-wise) product of matrices A and B. Proofs are given in the appendix.

Spectral graph theory.

Let G = ({1, . . .

, n}, E, W) be an undirected weighted graph, represented by a symmetric adjacency matrix W = (w ij ).

We define DISPLAYFORM0 We denote by N k,m the k-hop neighborhood of vertex m, containing vertices that are at most k edges away from m. The unnormalized graph Laplacian is an n × n symmetric positive-semidefinite matrix DISPLAYFORM1 In the following, we use the generic notation ∆ to refer to some Laplacian.

Since both normalized and unnormalized Laplacian are symmetric and positive semi-definite matrices, they admit an eigendecomposition ∆ = ΦΛΦ , where Φ = (φ 1 , . . .

φ n ) are the orthonormal eigenvectors and Λ = diag(λ 1 , . . .

, λ n ) is the diagonal matrix of corresponding non-negative eigenvalues (spectrum) 0 = λ 1 ≤ λ 2 ≤ . . .

≤ λ n .

The eigenvectors play the role of Fourier atoms in classical harmonic analysis and the eigenvalues can be interpreted as (the square of) frequencies.

Given a signal f = (f 1 , . . . , f n ) on the vertices of graph G, its graph Fourier transform is given byf = Φ f .

Given two signals f , g on the graph, their spectral convolution can be defined as the element-wise product of the Fourier transforms, f g = Φ (Φ g)•(Φ f ) = Φ diag(ĝ 1 , . . .

,ĝ n )f , which corresponds to the property referred to as the Convolution Theorem in the Euclidean case.

Spectral CNNs.

BID5 used the spectral definition of convolution to generalize CNNs on graphs, with a spectral convolutional layer of the form DISPLAYFORM2 (Here the n × p and n × q matrices DISPLAYFORM3 represent respectively the p-and q-dimensional input and output signals on the vertices of the graph, (ĝ l,l ,1 , . . .

,ĝ l,l ,k ) is a k × k diagonal matrix of spectral multipliers representing a learnable filter in the frequency domain, and ξ is a nonlinearity (e.g., ReLU) applied on the vertex-wise function values.

Pooling is performed by means of graph coarsening, which, given a graph with n vertices, produces a graph with n < n vertices and transfers signals from the vertices of the fine graph to those of the coarse one.

This framework has several major drawbacks.

First, the spectral filter coefficients are basis dependent, and consequently, a spectral CNN model learned on one graph cannot be transferred to another graph.

Second, the computation of the forward and inverse graph Fourier transforms incur expensive O(n 2 ) multiplication by the matrices Φ, Φ , as there is no FFT-like algorithms on general graphs.

Third, there is no guarantee that the filters represented in the spectral domain are localized in the spatial domain (locality property simulates local reception fields, BID7 ); assuming k = O(n) Laplacian eigenvectors are used, a spectral convolutional layer requires O(pqk) = O(n) parameters to train.

DISPLAYFORM4 To address the latter issues, BID13 argued that smooth spectral filter coefficients result in spatially-localized filters (an argument similar to vanishing moments).

The filter coefficients are represented asĝ i = g(λ i ), where g(λ) is a smooth transfer function of frequency λ.

Applying such filter to signal f can be expressed as DISPLAYFORM5 where applying a function to a matrix is understood in the operator functional calculus sense (applying the function to the matrix eigenvalues).

BID13 used parametric functions of the form g(λ) = r j=1 α j β j (λ), where β 1 (λ), . . .

, β r (λ) are some fixed interpolation kernels such as splines, and α = (α 1 , . . .

, α r ) are the interpolation coefficients used as the optimization variables during the network training.

In matrix notation, the filter is expressed as Gf = Φdiag(Bα)Φ f , where B = (b ij ) = (β j (λ i )) is a k × r matrix.

Such a construction results in filters with r = O(1) parameters, independent of the input size.

However, the authors explicitly computed the Laplacian eigenvectors Φ, resulting in high complexity.

ChebNet.

BID8 used polynomial filters represented in the Chebyshev basis DISPLAYFORM6 applied to rescaled frequencyλ ∈ [−1, 1]; here, α is the (r + 1)-dimensional vector of polynomial coefficients parametrizing the filter and optimized for during the training, and Such an approach has several important advantages.

First, since g α (∆) = r j=0 α j T j (∆) contains only matrix powers, additions, and multiplications by scalar, it can be computed avoiding the explicit expensive O(n 3 ) computation of the Laplacian eigenvectors.

Furthermore, due to the recursive definition of the Chebyshev polynomials, the computation of the filter g α (∆)f entails applying the Laplacian r times, resulting in O(rn) operations assuming that the Laplacian is a sparse matrix with O(1) non-zero elements in each row (a valid hypothesis for most real-world graphs that are sparsely connected).

Second, the number of parameters is O(1) as r is independent of the graph size n. Third, since the Laplacian is a local operator affecting only 1-hop neighbors of a vertex and a polynomial of degree r of the Laplacian affects only r-hops, the resulting filters have guaranteed spatial localization.

DISPLAYFORM7 A key disadvantage of Chebyshev filters is the fact that using polynomials makes it hard to produce narrow-band filters, as such filters require very high order r, and produce unwanted non-local filters.

This deficiency is especially pronounced when the Laplacian has clusters of eigenvalues concentrated around a few frequencies with large spectral gap ( FIG4 , middle right).

Such a behavior is characteristic of graphs with community structures, which is very common in many real-world graphs, for instance, social networks.

To overcome this major drawback, we need a new class of filters, that are both localized in space, and are able to specialize in narrow bands in frequency.

A key construction of this paper is a family of complex filters that enjoy the advantages of Chebyshev filters while avoiding some of their drawbacks.

A Cayley polynomial of order r is a real-valued function with complex coefficients, DISPLAYFORM0 where c = (c 0 , . . .

, c r ) is a vector of one real coefficient and r complex coefficients and h > 0 is the spectral zoom parameter, that will be discussed later.

A Cayley filter G is a spectral filter defined on real signals f by where the parameters c and h are optimized for during training.

Similarly to the Chebyshev filters, Cayley filters involve basic matrix operations such as powers, additions, multiplications by scalars, and also inversions.

This implies that application of the filter Gf can be performed without explicit expensive eigendecomposition of the Laplacian operator.

In the following, we show that Cayley filters are analytically well behaved; in particular, any smooth spectral filter can be represented as a Cayley polynomial, and low-order filters are localized in the spatial domain.

We also discuss numerical implementation and compare Cayley and Chebyshev filters.

DISPLAYFORM1 Analytic properties.

Cayley filters are best understood through the Cayley transform, from which their name derives.

Denote by e iR = {e iθ : θ ∈ R} the unit complex circle.

The Cayley transform DISPLAYFORM2 x+i is a smooth bijection between R and e iR \ {1}. The complex matrix C(h∆) DISPLAYFORM3 −1 obtained by applying the Cayley transform to the scaled Laplacian h∆ has its spectrum in e iR and is thus unitary.

Since DISPLAYFORM4 .

Therefore, using 2Re{z} = z + z, any Cayley filter (4) can be written as a conjugateeven Laurent polynomial w.r.t.

C(h∆), DISPLAYFORM5 Since the spectrum of C(h∆) is in e iR , the operator C j (h∆) can be thought of as a multiplication by a pure harmonic in the frequency domain e iR for any integer power j, DISPLAYFORM6 A Cayley filter can be thus seen as a multiplication by a finite Fourier expansions in the frequency domain e iR .

Since (5) is conjugate-even, it is a (real-valued) trigonometric polynomial.

Note that any spectral filter can be formulated as a Cayley filter.

Indeed, spectral filters g(∆) are specified by the finite sequence of values g(λ 1 ), . . .

, g(λ n ), which can be interpolated by a trigonometric polynomial.

Moreover, since trigonometric polynomials are smooth, we expect low order Cayley filters to be well localized in some sense on the graph, as discussed later.

Finally, in definition (4) we use complex coefficients.

If c j ∈ R then FORMULA14 is an even cosine polynomial, and if c j ∈ iR then (5) is an odd sine polynomial.

Since the spectrum of h∆ is in R + , it is mapped to the lower half-circle by C, on which both cosine and sine polynomials are complete and can represent any spectral filter.

However, it is beneficial to use general complex coefficients, since complex Fourier expansions are overcomplete in the lower half-circle, thus describing a larger variety of spectral filters of the same order without increasing the computational complexity of the filter.

Spectral zoom.

To understand the essential role of the parameter h in the Cayley filter, consider C(h∆).

Multiplying ∆ by h dilates its spectrum, and applying C on the result maps the non-negative spectrum to the complex half-circle.

The greater h is, the more the spectrum of h∆ is spread apart in R + , resulting in better spacing of the smaller eigenvalues of C(h∆).

On the other hand, the smaller h is, the further away the high frequencies of h∆ are from ∞, the better spread apart are the high frequencies of C(h∆) in e iR (see FIG1 ).

Tuning the parameter h allows thus to 'zoom' in to different parts of the spectrum, resulting in filters specialized in different frequency bands.

Numerical properties.

The numerical core of the Cayley filter is the computation of C j (h∆)f for j = 1, . . .

, r, performed in a sequential manner.

Let y 0 , . . .

, y r denote the solutions of the following linear recursive system, DISPLAYFORM7 Note that sequentially approximating y j in (6) using the approximation of y j−1 in the rhs is stable, since C(h∆) is unitary and thus has condition number 1.Equations FORMULA16 can be solved with matrix inversion exactly, but it costs O(n 3 ).

An alternative is to use the Jacobi method, 1 which provides approximate solutionsỹ j ≈ y j .

Let J = −(Diag(h∆ + iI)) −1 Off(h∆ + iI) be the Jacobi iteration matrix associated with equation (6).

For the unnormalized Laplacian, J = (hD + iI) −1 hW. Jacobi iterations for approximating (6) for a given j have the form DISPLAYFORM8 initialized withỹ DISPLAYFORM9 .

The application of the approximate Cayley filter is given by Gf = r j=0 c jỹj ≈ Gf , and takes O(rKn) operations under the previous assumption of a sparse Laplacian.

The method can be improved by normalizing ỹ j 2 = f 2 .Next, we give an error bound for the approximate filter.

For the unnormalized Laplacian, let DISPLAYFORM10 < 1.

For the normalized Laplacian, we assume that (h∆ n + iI) is dominant diagonal, which gives κ = J ∞ < 1.

Proposition 1.

Under the above assumptions, Proposition 1 is pessimistic in the general case, while requires strong assumptions in the regular case.

We find that in most real life situations the behavior is closer to the regular case.

It also follows from Proposition 1 that smaller values of the spectral zoom h result in faster convergence, giving this parameter an additional numerical role of accelerating convergence.

DISPLAYFORM11 Complexity.

In practice, an accurate inversion of (h∆ + iI) is not required, since the approximate inverse is combined with learned coefficients, which "compensate", as necessary, for the inversion inaccuracy.

In a CayleyNet for a fixed graph, we fix the number of Jacobi iterations.

Since the convergence rate depends on κ, that depends on the graph, different graphs may need different numbers of iterations.

The convergence rate also depends on h. Since there is a trade-off between the spectral zoom amount h, and the accuracy of the approximate inversion, and since h is a learnable parameter, the training finds the right balance between the spectral zoom amount and the inversion accuracy.

We study the computational complexity of our method, as the number of edges n of the graph tends to infinity.

For every constant of a graph, e.g d, κ, we add the subscript n, indicating the number of edges of the graph.

For the unnormalized Laplacian, we assume that d n and h n are bounded, which gives κ n < a < 1 for some a independent of n. For the normalized Laplacian, we assume that κ n < a < 1.

By Theorem 1, fixing the number of Jacobi iterations K and the order of the filter r, independently of n, keeps the Jacobi error controlled.

As a result, the number of parameters is O(1), and for a Laplacian modeled as a sparse matrix, applying a Cayley filter on a signal takes O(n) operations.

Localization.

Unlike Chebyshev filters that have the small r-hop support, Cayley filters are rational functions supported on the whole graph.

However, it is still true that Cayley filters are well localized on the graph.

Let G be a Cayley filter and δ m denote a delta-function on the graph, defined as one at vertex m and zero elsewhere.

We show that Gδ m decays fast, in the following sense: Definition 2 (Exponential decay on graphs).

Let f be a signal on the vertices of graph G, 1 ≤ p ≤ ∞, and 0 < < 1.

Denote by S ⊆ {1, . . .

, n} a subset of the vertices and by S c its complement.

We say that the L p -mass of f is supported in S up to if f | S c p ≤ f p , where f | S c = (f l ) l∈S c is the restriction of f to S c .

We say that f has (graph) exponential decay about vertex m, if there exists some γ ∈ (0, 1) and c > 0 such that for any k, the L p -mass of f is supported in N k,m up to cγ k .

Here, N k,m is the k-hop neighborhood of m. 1 We remind that the Jacobi method for solving Ax = b consists in decomposing A = Diag(A) + Off(A) and obtaining the solution iteratively as Remark 3.

Note that Definition 2 is analogous to classical exponential decay on Euclidean space: DISPLAYFORM12 DISPLAYFORM13 Theorem 4.

Let G be a Cayley filter of order r.

Then, Gδ m has exponential decay about m in L 2 , with constants c = 2M1 Gδm 2 and γ = κ 1/r (where M and κ are from Proposition 1).

Chebyshev as a special case of Cayley.

For a regular graph with D = dI, using Jacobi inversion based on zero iterations, we get that any Cayley filter of order r is a polynomial of ∆ in the monomial base h∆−i hd+i j .

In this situation, a Chebyshev filter, which is a real valued polynomial of ∆, is a special case of a Cayley filter.

Spectral zoom and stability.

Generally, both Chebyshev polynomials and trigonometric polynomials give stable approximations, optimal for smooth functions.

However, this crude statement is oversimplified.

One of the drawbacks in Chebyshev filters is the fact that the spectrum of ∆ is always mapped to [−1, 1] in a linear manner, making it hard to specialize in small frequency bands.

In Cayley filters, this problem is mitigated with the help of the spectral zoom parameter h. As an example, consider the community detection problem discussed in the next section.

A graph with strong communities has a cluster of small eigenvalues near zero.

Ideal filters g(∆) for extracting the community information should be able to focus on this band of frequencies.

Approximating such filters with Cayley polynomials, we zoom in to the band of interest by choosing the right h, and then project g onto the space of trigonometric polynomials of order r, getting a good and stable approximation ( FIG4 , bottom right).

However, if we project g onto the space of Chebyshev polynomials of order r, the interesting part of g concentrated on a small band is smoothed out and lost ( FIG4 , middle right).

Thus, projections are not the right way to approximate such filters, and the stability of orthogonal polynomials cannot be invoked.

When approximating g on the small band using polynomials, the approximation will be unstable away from this band; small perturbations in g will result in big perturbations in the Chebyshev filter away from the band.

For this reason, we say that Cayley filters are more stable than Chebyshev filters.

Regularity.

We found that in practice, low-order Cayley filters are able to model both very concentrated impulse-like filters, and wider Gabor-like filters.

Cayley filters are able to achieve a wider range of filter supports with less coefficients than Chebyshev filters FIG3 ), making the Cayley class more regular than Chebyshev.

Complexity.

Under the assumption of sparse Laplacians, both Cayley and Chebyshev filters incur linear complexity O(n).

Besides, the new filters are equally simple to implement as Chebyshev filters; as seen in Eq.7, they boil down to simple sparse matrix-vector multiplications providing a GPU friendly implementation.

Experimental settings.

We test the proposed CayleyNets reproducing the experiments of Defferrard et al. FORMULA7 ; BID17 ; BID24 a) and using ChebNet BID8 ) as our main baseline method.

All the methods were implemented in TensorFlow of M. BID21 .

The experiments were executed on a machine with a 3.5GHz Intel Core i7 CPU, 64GB of RAM, and NVIDIA Titan X GPU with 12GB of RAM.

SGD+Momentum and Adam BID16 ) optimization methods were used to train the models in MNIST and the rest of the experiments, respectively.

Training and testing were always done on disjoint sets.

Community detection.

We start with an experiment on a synthetic graph consisting of 15 communities with strong connectivity within each community and sparse connectivity across communities FIG4 .

Though rather simple, such a dataset allows to study the behavior of different algorithms in controlled settings.

On this graph, we generate noisy step signals, defined as f i = 1 + σ i if i belongs to the community, and f i = σ i otherwise, where DISPLAYFORM0 The goal is to classify each such signal according to the community it belongs to.

The neural network architecture used for this task consisted of a spectral convolutional layer (based on Chebyshev or Cayley filters) with 32 output features, a mean pooling layer, and a softmax classifier for producing the final classification into one of the 15 classes.

The classification accuracy is shown in FIG4 (right, top) along with examples of learned filters (right, bottom).

We observe that CayleyNet significantly outperforms ChebNet for smaller filter orders, with an improvement as large as 80%.Studying the filter responses, we note that due to the capability to learn the spectral zoom parameter, CayleyNet allows to generate band-pass filters in the low-frequency band that discriminate well the communities FIG4 .

Complexity.

We experimentally validated the computational complexity of our model applying filters of different order r to synthetic 15-community graphs of different size n using exact matrix inversion and approximation with different number of Jacobi iterations FIG5 center and right, Figure 6 in the appendix).

All times have been computed running 30 times the considered models and averaging the final results.

As expected, approximate inversion guarantees O(n) complexity.

We further conclude that typically very few Jacobi iterations are required FIG5 , left shows that our model with just one Jacobi iteration outperforms ChebNet for low-order filters on the community detection problem).MNIST.

Following BID8 ; BID24 , for a toy example, we approached the classical MNIST digits classification as a learning problem on graphs.

Each pixel of an image is a vertex of a graph (regular grid with 8-neighbor connectivity), and pixel color is a signal on the graph.

We used a graph CNN architecture with two spectral convolutional layers based on Chebyshev and Cayley filters (producing 32 and 64 output features, respectively), interleaved with pooling layers performing 4-times graph coarsening using the Graclus algorithm (Dhillon et al. FORMULA7 finally a fully-connected layer (this architecture replicates the classical LeNet5, BID19 , architecture, which is shown for comparison).

MNIST classification results are reported in TAB0 .

CayleyNet (11 Jacobi iterations) achieves the same (near perfect) accuracy as ChebNet with filters of lower order (r = 12 vs 25).Examples of filters learned by ChebNet and CayleyNet are shown in FIG3 .

0.1776 +/-0.06079 sec and 0.0268 +/-0.00841 sec are respectively required by CayleyNet and ChebNet for analyzing a batch of 100 images at test time.

FORMULA7 ) 86.64 % 47K ChebNet BID8 ) 87.07 % 46K CayleyNet 88.09 % 46KCitation network.

Next, we address the problem of vertex classification on graphs using the popular CORA citation graph, BID28 .

Each of the 2708 vertices of the CORA graph represents a scientific paper, and an undirected unweighted edge represents a citation (5429 edges in total).

For each vertex, a 1433-dimensional binary feature vector representing the content of the paper is given.

The task is to classify each vertex into one of the 7 groundtruth classes.

We split the graph into training (1,708 vertices), validation (500 vertices) and test (500 vertices) sets, for simulating the labeled and unlabeled information.

We train ChebNet and CayleyNet with the architecture presented in BID17 ; BID24 (two spectral convolutional layers with 16 and 7 outputs), DCNN BID0 ) with 2 diffusion layer (10 hidden features and 2 diffusion hops) and GCN BID17 ) with 3 convolutional layer (32 and 16 hidden features).

Figure 5 compares ChebNets and CayleyNets, in a number of different settings.

Since ChebNets require Laplacians with spectra bounded in [−1, 1], we consider both the normalized Laplacian (the two left figures), and the scaled unnormalized Laplacian (2∆/λ max − I), where ∆ is the unnormalized Laplacian and λ max is its largest eigenvalue (the two right figures).

For fair comparison, we fix the order of the filters (top figures), and fix the overall number of network parameters (bottom figures).

In the bottom figure, the Cayley filters are restricted to even cosine polynomials by considering only real filter coefficients.

TAB1 shows a comparison of the performance obtained with different methods (all architectures roughly present the same amount of parameters).

The best CayleyNets consistently outperform the best competitors.

Recommender system.

In our final experiment, we applied CayleyNet to recommendation system, formulated as matrix completion problem on user and item graphs, BID24 .

The task is, given a sparsely sampled matrix of scores assigned by users (columns) to items (rows), to fill in the missing scores.

The similarities between users and items are given in the form of column and row graphs, respectively.

BID24 approached this problem as learning with a Recurrent Graph CNN (RGCNN) architecture, using an extension of ChebNets to matrices defined on multiple graphs in order to extract spatial features from the score matrix; these features are then fed into an Figure 5 : ChebNet (blue) and CayleyNet (orange) test accuracies obtained on the CORA dataset for different polynomial orders.

Polynomials with complex coefficients (top) and real coefficients (bottom) have been exploited with CayleyNet in the two analysis.

Orders 1 to 6 have been used in both comparisons.

The best CayleyNet consistently outperform the best ChebNet requiring at the same time less parameters (CayleyNet with order r and complex coefficients requires a number of parameters equal to ChebNet with order 2r).RNN producing a sequential estimation of the missing scores.

Here, we repeated verbatim their experiment on the MovieLens dataset BID23 ), replacing Chebyshev filters with Cayley filters.

We used separable RGCNN architecture with two CayleyNets of order r = 4 employing 15 Jacobi iterations.

The results are reported in TAB3 .

To present a complete comparison we further extended the experiments reported in BID24 by training sRGCNN with ChebNets of order 8, this provides an architecture with same number of parameters as the exploited CayleyNet (23K coefficients).

Our version of sRGCNN outperforms all the competing methods, including the previous result with Chebyshev filters reported in BID24 .

sRGCNNs with Chebyshev polynomials of order 4 and 8 respectively require 0.0698 +/-0.00275 sec and 0.0877 +/-0.00362 sec at test time, sRGCNN with Cayley polynomials of order 4 and 15 jacobi iterations requires 0.165 +/-0.00332 sec. BID14 ; BID31 ) 1.653 GMC BID15 0.996 GRALS BID26 ) 0.945 sRGCNN Cheby,r=4 BID24 0.929 sRGCNN Cheby,r=8 BID24 0.925 sRGCNN Cayley 0.922

In this paper, we introduced a new efficient spectral graph CNN architecture that scales linearly with the dimension of the input data.

Our architecture is based on a new class of complex rational Cayley filters that are localized in space, can represent any smooth spectral transfer function, and are highly regular.

The key property of our model is its ability to specialize in narrow frequency bands with a small number of filter parameters, while still preserving locality in the spatial domain.

We validated these theoretical properties experimentally, demonstrating the superior performance of our model in a broad range of graph learning problems.

First note the following classical result for the approximation of Ax = b using the Jacobi method: if the initial condition is DISPLAYFORM0 In our case, note that if we start with initial conditionỹ (0) j = 0, the next iteration givesỹ (0) j = b j , which is the initial condition from our construction.

Therefore, since we are approximating y j = C(h∆)ỹ j−1 byỹ j =ỹ DISPLAYFORM1 Define the approximation error in C(h∆) j f by DISPLAYFORM2 By the triangle inequality, by the fact that C j (h∆) is unitary, and by (8) DISPLAYFORM3 where the last inequality is due to DISPLAYFORM4 Now, using standard norm bounds, in the general case we have DISPLAYFORM5 The solution of this recurrent sequence is DISPLAYFORM6 If we use the version of the algorithm, in which eachỹ j is normalized, we get by (9) e j ≤ e j−1 + √ nκ K+1 .

The solution of this recurrent sequence is DISPLAYFORM7 We denote in this case M j = j √ nIn case the graph is regular, we have D = dI. In the non-normalized Laplacian case, DISPLAYFORM8 The spectral radius of ∆ is bounded by 2d.

This can be shown as follows.

a value λ is not an eigenvalue of ∆ (namely it is a regular value) if and only if (∆ − λI) is invertible.

Moreover, the matrix (∆ − λI) is strictly dominant diagonal for any |λ| > 2d.

By Levy-Desplanques theorem, any strictly dominant diagonal matrix is invertible, which means that all of the eigenvalues of ∆ are less than 2d in their absolute value.

As a result, the spectral radius of (dI − ∆) is realized on the smallest eigenvalue of ∆, namely it is |d − 0| = d. This means that the specral radius of J is DISPLAYFORM9 .

As a DISPLAYFORM10 = κ.

We can now continue from (9) to get e j ≤ e j−1 + J K+1 2(1 + e j−1 ) = e j−1 + κ K+1 (1 + e j−1 ).As before, we get e j ≤ jκ K+1 + O(κ 2K+2 ), and e j ≤ jκ K+1 if eachỹ j is normalized.

We denote in this case M j = j.

In the case of the normalized Laplacian of a regular graph, the spectral radius of ∆ n is bounded by 2, and the diagonal entries are all 1.

Equation (10) in this case reads J = h h+i (I − ∆ n ), and J has spectral radius h √ h 2 +1.

Thus J 2 = h √ h 2 +1= κ and we continue as before to get e j ≤ jκ K+1 and M j = j.

In all cases, we have by the triangle inequality Figure 6: Test (above) and training (below) times with corresponding ratios as function of filter order r and graph size n on our community detection dataset.

@highlight

A spectral graph convolutional neural network with spectral zoom properties.
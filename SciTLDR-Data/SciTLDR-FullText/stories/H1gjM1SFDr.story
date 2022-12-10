High-dimensional data often lie in or close to low-dimensional subspaces.

Sparse subspace clustering methods with sparsity induced by L0-norm, such as L0-Sparse Subspace Clustering (L0-SSC), are demonstrated to be more effective than its L1 counterpart such as Sparse Subspace Clustering (SSC).

However, these L0-norm based subspace clustering methods are restricted to clean data that lie exactly in subspaces.

Real data often suffer from noise and they may lie close to subspaces.

We propose noisy L0-SSC to handle noisy data so as to improve the robustness.

We show that the optimal solution to the optimization problem of noisy L0-SSC achieves subspace detection property (SDP), a key element with which data from different subspaces are separated, under deterministic and randomized models.

Our results provide theoretical guarantee on the correctness of noisy L0-SSC in terms of SDP on noisy data.

We further propose Noisy-DR-L0-SSC which provably recovers the subspaces on dimensionality reduced data.

Noisy-DR-L0-SSC first projects the data onto a lower dimensional space by linear transformation, then performs noisy L0-SSC on the dimensionality reduced data so as to improve the efficiency.

The experimental results demonstrate the effectiveness of noisy L0-SSC and Noisy-DR-L0-SSC.

High-dimensional data often lie in or close to low-dimensional subspaces.

Sparse subspace clustering methods with sparsity induced by 0 -norm, such as 0 -Sparse Subspace Clustering ( 0 -SSC) Yang et al. (2016) , are demonstrated to be more effective than its 1 counterpart such as Sparse Subspace Clustering (SSC) Elhamifar & Vidal (2013) .

However, these 0 -norm based subspace clustering methods are restricted to clean data that lie exactly in subspaces.

Real data often suffer from noise and they may lie close to subspaces.

We propose noisy 0 -SSC to handle noisy data so as to improve the robustness.

We show that the optimal solution to the optimization problem of noisy 0 -SSC achieves subspace detection property (SDP), a key element with which data from different subspaces are separated, under deterministic and randomized models.

Our results provide theoretical guarantee on the correctness of noisy 0 -SSC in terms of SDP on noisy data.

We further propose Noisy-DR-0 -SSC which provably recovers the subspaces on dimensionality reduced data.

Noisy-DR-0 -SSC first projects the data onto a lower dimensional space by linear transformation, then performs noisy 0 -SSC on the dimensionality reduced data so as to improve the efficiency.

The experimental results demonstrate the effectiveness of noisy 0 -SSC and Noisy-DR-0 -SSC.

Clustering is an important unsupervised learning procedure for analyzing a broad class of scientific data in biology, medicine, psychology and chemistry.

On the other hand, high-dimensional data, such as facial images and gene expression data, often lie in low-dimensional subspaces in many cases, and clustering in accordance to the underlying subspace structure is particularly important.

For example, the well-known Principal Component Analysis (PCA) works perfectly if the data are distributed around a single subspace.

The subspace learning literature develops more general methods that recover multiple subspaces in the original data, and subspace clustering algorithms Vidal (2011) aim to partition the data such that data belonging to the same subspace are identified as one cluster.

Among various subspace clustering algorithms, the ones that employ sparsity prior, such as Sparse Subspace Clustering (SSC) Elhamifar & Vidal (2013) and 0 -Sparse Subspace Clustering ( 0 -SSC) Yang et al. (2016) , have been proven to be effective in separating the data in accordance with the subspaces that the data lie in under certain assumptions.

Sparse subspace clustering methods construct the sparse similarity matrix by sparse representation of the data.

Subspace detection property (SDP) defined in Section 4.1 ensures that the similarity between data from different subspaces vanishes in the sparse similarity matrix, and applying spectral clustering Ng et al. (2001) on such sparse similarity matrix leads to compelling clustering performance.

Elhamifar and Vidal Elhamifar & Vidal (2013) prove that when the subspaces are independent or disjoint, SDP can be satisfied by solving the canonical sparse linear representation problem using data as the dictionary, under certain conditions on the rank, or singular value of the data matrix and the principle angle between the subspaces.

SSC has been successfully applied to a novel deep neural network architecture, leading to the first deep sparse subspace clustering method Peng et al. (2016) .

Under the independence assumption on the subspaces, low rank representation Liu et al. (2010; is also proposed to recover the subspace structures.

Relaxing the assumptions on the subspaces to allowing overlapping subspaces, the Greedy Subspace Clustering Park et al. (2014) and the LowRank Sparse Subspace Clustering achieve subspace detection property with high probability.

The geometric analysis in Soltanolkotabi & Cands (2012) shows the theoretical results on subspace recovery by SSC.

In the following text, we use the term SSC or 1 -SSC exchangeably to indicate the Sparse Subspace Clustering method in Elhamifar & Vidal (2013) .

Real data often suffer from noise.

Noisy SSC proposed in handles noisy data that lie close to disjoint or overlapping subspaces.

While 0 -SSC Yang et al. (2016) has guaranteed clustering correctness via subspace detection property under much milder assumptions than previous subspace clustering methods including SSC, it assumes that the observed data lie in exactly in the subspaces and does not handle noisy data.

In this paper, we present noisy 0 -SSC, which enhances 0 -SSC by theoretical guarantee on the correctness of clustering on noisy data.

It should be emphasized that while 0 -SSC on clean data Yang et al. (2016) empirically adopts a form of optimization problem robust to noise, it lacks theoretical analysis on the correctness of 0 -SSC on noisy data.

In this paper, the correctness of noisy 0 -SSC on noisy data in terms of the subspace detection property is established.

Our analysis is under both deterministic model and randomized models, which is also the model employed in the geometric analysis of SSC Soltanolkotabi & Cands (2012) .

Our randomized analysis demonstrates potential advantage of noisy 0 -SSC over its 1 counterpart as more general assumption on data distribution can be adopted.

Moreover, we present Noisy Dimensionality Reduced 0 -Sparse Subspace Clustering (Noisy-DR-0 -SSC), an efficient version of noisy 0 -SSC which also enjoys robustness to noise.

Noisy-DR-0 -SSC first projects the data onto a lower dimensional space by random projection, then performs noisy 0 -SSC on the dimensionality reduced data.

Noisy-DR-0 -SSC provably recovers the underlying subspace structure in the original data from the dimensionality reduced data under deterministic model.

Experimental results demonstrate the effectiveness of both noisy 0 -SSC and Noisy-DR-0 -SSC.

We use bold letters for matrices and vectors, and regular lower letter for scalars throughout this paper.

The bold letter with superscript indicates the corresponding column of a matrix, e.g. A i is the i-th column of matrix A, and the bold letter with subscript indicates the corresponding element of a matrix or vector.

· F and · p denote the Frobenius norm and the vector p -norm or the matrix p-norm, and diag(·) indicates the diagonal elements of a matrix.

H T ⊆ R d indicates the subspace spanned by the columns of T, and A I denotes a submatrix of A whose columns correspond to the nonzero elements of I (or with indices in I without confusion).

σ t (·) denotes the t-th largest singular value of a matrix, and σ min (·) indicates the smallest singular value of a matrix.

supp(·) is the support of a vector, P S is an operator indicating projection onto the subspace S .

We hereby introduce the notations for subspace clustering on noisy data considered in this paper.

The uncorrupted data matrix is denoted by Y = [y 1 , . . .

, y n ] ∈ R d×n , where d is the dimensionality and n is the size of the data.

The uncorrupted data Y lie in a union of

is the additive noise.

x i = y i + n i is the noisy data point that is corrupted by the noise n i .

n k = n, and denote the corresponding columns in X by X (k) .

The data X are normalized such that each column has unit 2 -norm in our deterministic analysis.

We consider deterministic noise model where the noise Z is fixed and max n i ≤ δ.

Note that our analysis can be extended to a random noise model which is common and also considered by noisy SSC Wang & Xu (2013) , and the random noise model assumes that columns of Z are sampled i.i.d. and max n i ≤ δ with high probability.

Note that such random noise model does not require spherical symmetric noise as that in .

0 -SSC Yang et al. (2016) proposes to solve the following 0 sparse representation problem

and it proves that the subspace detection property defined in Definition 1 is satisfied with the globally optimal solution to (1).

We resort to solve the 0 regularized sparse approximation problem below to handle noisy data for 0 -SSC, which is the optimization problem of noisy 0 -SSC:

The definition of subspace detection property for noisy 0 -SSC and noiseless 0 -SSC, i.e. 0 -SSC on noiseless data, is defined in Definition 1 below.

Definition 1. (Subspace detection property for noisy and noiseless 0 -SSC) Let Z * be the optimal solution to (2).

The subspaces {S k } K k=1 and the data X satisfy subspace detection property for noisy 0 -SSC if Z i is a nonzero vector, and nonzero elements of Z i correspond to the columns of X from the same subspace as y i for all 1 ≤ i ≤ n.

Similarly, in the noiseless setting where X = Y , let Z * be the optimal solution to (1).

The subspaces {S k } K k=1 and the data X satisfy the subspace detection property for noiseless 0 -SSC if Z i is a nonzero vector, and nonzero elements of Z i correspond to the columns of X that from the same subspace as y i for all 1 ≤ i ≤ n.

We say that subspace detection property holds for x i if nonzero elements of Z * i correspond to the data that lie in the same subspace as y i , for either noisy 0 -SSC or noiseless 0 -SSC.

Similar to Soltanolkotabi & Cands (2012) , we introduce the deterministic, semi-random and fullyrandom models for the analysis of noisy 0 -SSC.

• Deterministic Model: the subspaces and the data in each subspace are fixed.

• Semi-Random Model: the subspaces are fixed but the data are independent and identically distributed in each of the subspaces.

• Fully-Random Model: both the subspaces and the data of each subspace are independent and identically distributed.

The data in the above definitions refer to clean data without noise.

We refer to semi-random model and fully-random model as randomized models in this paper.

The theoretical results on the subspace detection property for noisy 0 -SSC are presented in this section under deterministic model and randomized models.

All the data Y are normalized to have unit norm for illustration purpose, so they lie on the surface of the sphere.

S 1 and S 2 are two subspaces in the three-dimensional ambient space.

The subspace spanned by y i ∈ S 1 and y j ∈ S 2 is an external subspace, and the intersection of this external subspace and S 1 is a dashed line y i OA.

We introduce the definition of general position and external subspace before our analysis on noisy 0 -SSC.

The assumption of general condition is rather mild.

In fact, if the data points in X (k) are independently distributed according to any continuous distribution, then they almost surely in general position.

Let the distance between a point x ∈ R d and a subspace S ⊆ R d be defined as d(x, S) = inf y∈S x− y 2 , the definition of external subspaces is presented as follows.

Figure 1 illustrates an example of external subspace.

spanned by a set of linear independent points {y ij } L j=1 ⊆ Y is defined to be an external subspace of y if

.

The point y is said to be away from its external subspaces if

are the set of all external subspaces of y of dimension no greater than d for y, i.e.

}.

All the data points in Y (k) are said to be away from the external subspaces if each of them is away from the its associated external spaces.

Remark 1. (Subspace detection property holds for noiseless 0 -SSC under the deterministic model) It can be verified that the following statement is true.

Under the deterministic model, suppose data is noiseless,

If all the data points in Y (k) are away from the external subspaces for any 1 ≤ k ≤ K, then the subspace detection property for 0 -SSC holds with the optimal solution Z * to (1).

To present our theoretical results of the correctness of noisy 0 -SSC, we also need the definitions of the minimum restricted eigenvalue and the subspace separation margin, which are defined as follows.

In the following analysis, we employ β to denote the sparse code of datum x i so that a simpler notation other than Z i is dedicated to our analysis.

Definition 4.

The minimum restricted eigenvalue of the uncorrupted data is defined as

for r ≥ 1.

In addition, the normalized minimum restricted eigenvalue of the uncorrupted data is defined byσ

We have the following perturbation bound for the distance between a data point and the subspaces spanned by noisy and noiseless data, which is useful to establish the conditions when the subspace detection property holds for noisy 0 -SSC.

Lemma 1.

Let β ∈ R n and Y β has full column rank.

Suppose δ <σ Y ,r where r = β 0 , then X β is a full column rank matrix, and

The optimization problem of noisy 0 -SSC (2) is separable.

For each 1 ≤ i ≤ n, the optimization problem with respect to the sparse code of i-th data point is

Lemma 2 shows that the optimal solution to the noisy 0 -SSC problem (6) is also that to a 0 -minimization problem with tolerance to noise.

Lemma 2.

Let nonzero vector β * be the optimal solution to the noisy 0 -SSC problem (6) for point

then β * is the optimal solution to the following sparse approximation problem with the uncorrupted data as the dictionary:

where c *

Define B(x i , c 0 ) = {x : x − x i ≤ c 0 } be the ball centered at x i with radius c 0 .

If B(x i , c 0 ) is away from the corresponding confusion area, i.e. all the external subspaces in H yi,d k , then subspace detection property holds with the solution to a proper sparse approximation problem where x i is approximated by the uncorrupted data, as shown in the following Lemma.

Lemma 3.

Suppose Y is in general position and

Then the subspace detection property holds for x i with the optimal solution to the following sparse approximation problem, denoted by β * , i.e. nonzero elements of β * correspond to the columns of X from the same subspace as y i .

Now we use the above results to present the main result on the correctness of noisy 0 -SSC.

Theorem 1. (Subspace detection property holds for noisy 0 -SSC) Let nonzero vector β * be the optimal solution to the noisy 0 -SSC problem (6) for point x i with β * 0 = r * > 1, and c *

Then the subspace detection property holds for x i with β * .

Here τ 0 , τ 1 ,σ * Y and σ * X are defined in Lemma 2.

Remark 2.

When δ = 0 and there is no noise in the data X, the conditions for the correctness of noisy 0 -SSC in Theorem 1 almost reduce to that for noiseless 0 -SSC.

To see this, the conditions are reduced to B(y i , c * ) ∩ H = ∅, which are exactly the conditions required by noiseless 0 -SSC, namely data are away from the external subspaces by choosing λ → 0 and it follows that c * = 0.

While Theorem 1 establishes geometric conditions under which the subspace detection property holds for noisy 0 -SSC, it can be seen that these conditions are often coupled with the optimal solution β * to the noisy 0 -SSC problem (6).

In the following theorem, the correctness of noisy 0 -SSC is guaranteed in terms of λ, the weight for the 0 regularization term in (6), and the geometric conditions independent of the optimal solution to (6).

Let M i > 0 be the minimum distance between y i ∈ S k and its external subspaces when y i is away from its external subspaces, i.e.

The following two quantities related to the spectrum of clean and noisy data, µ r and σ X,r , are defined as follows with r > 1 for the analysis in Theorem 2.

Theorem 2. (Subspace detection property holds for noisy 0 -SSC under deterministic model, with conditions in terms of λ) Let nonzero vector β * be the optimal solution to the noisy 0 -SSC problem (6) for point x i with β * 0 = r * , n k ≥ d k + 1 for every 1 ≤ k ≤ K, and there exists 1 < r 0 ≤ d such that 1 < r * ≤ r 0 .

Suppose Y is in general position, y i ∈ S k for some 1 ≤ k ≤ K, δ < min 1≤r<r0σY ,r , and

and

Then if

where λ 0 max{λ 1 , λ 2 } and λ1 inf{0 < λ < 1 :

the subspace detection property holds for x i with β * .

Here M i , µ r0 and σ X,r0 are defined in (9), (10) and (11) respectively.

Remark 3.

The two conditions (12) and (13) (12) and (13) hold, λ 1 and λ 2 can always be chosen in accordance with (15) and (16).

Remark 4.

It can be observed from condition (14) that noisy 0 -SSC encourages sparse solution by a relatively large λ so as to guarantee the subspace detection property.

This theoretical finding is consistent with the empirical study shown in the experimental results.

In this subsection, the correctness of noisy 0 -SSC is analyzed when the clean data in each subspace are distributed at random.

We assume that the data in subspace S (k) are i.i.d.

isotropic samples on sphere of radius √ d k centered at the origin according to some continuous distribution, for

In addition, for each 1 ≤ k ≤ K, we assume that the following condition holds:

(a) There exists a constant M ≥ 1 such that for any t > 0, any y ∈ Y (k) , and any vector v with unit 2 -norm,

Intuitively, condition (a) requires that the projection of any data point onto arbitrary unit vector is bounded from both sides with relatively large probability.

This condition is also required in Yaskov (2014) to derive lower bound for the least singular value of a random matrix with independent isotropic columns.

In order to meet the conditions in Theorem 2 so as to guarantee the subspace detection property under randomized models, the following lemma is presented and it provides the geometric concentration inequality for the distance between a point y ∈ Y (k) and any of its external subspaces.

It renders a lower bound for M i , namely the minimum distance between y i ∈ S k and its external subspaces.

Lemma 4.

Under randomized models, given 1 ≤ k ≤ K and y ∈ Y (k) , suppose H ∈ H yi,d k is any external subspace of y. Then for any t > 0,

We then have the following results regarding to the subspace detection property of noisy 0 -SSC under randomized models.

0 -SSC under randomized models, with conditions in terms of λ) Under randomized models, let nonzero vector β * be the optimal solution to the noisy 0 -SSC problem (6) for point x i with β * 0 = r * , n k ≥ d k + 1 for every 1 ≤ k ≤ K, and there exists 1 < r 0 ≤ d such that 1 < r * ≤ r 0 .

Suppose the data in each subspace are i.i.d.

isotropic samples according to some continuous distribution that satisfies condition (a).

.

For t > 0 such that 1 − 2t

and

where λ 0 max{λ 1 , λ 2 } and λ 1 inf{0 < λ < 1 :

Then with probability at least

2 ), the subspace detection property holds for x i with β * .

Remark 5.

Note that there is no assumption on the distribution of subspaces in Theorem 3, so it is not required that the subspaces should have uniform distribution, an is required in the geometric analysis of 1 -SSC Soltanolkotabi & Cands (2012) and its noisy version .

In addition, while Soltanolkotabi & Cands (2012) ; 4 NOISY 0 -SSC ON DIMENSIONALITY REDUCED DATA: NOISY-DR-0 -SSC Albeit the theoretical guarantee and compelling empirical performance of noisy 0 -SSC to be shown in the experimental results, the computational cost of noisy 0 -SSC is high with the high dimensionality of the data.

In this section, we propose Noisy Dimensionality Reduced 0 -SSC (Noisy-DR-0 -SSC) which performs noisy 0 -SSC on dimensionality reduced data.

The theoretical guarantee on the correctness of Noisy-DR-0 -SSC under deterministic model as well as its empirical performance are presented.

Noisy-DR-0 -SSC performs subspace clustering by the following two steps: 1) obtain the dimension reduced dataX = PX with a linear transformation P ∈ R p×d (p < d).

2) perform noisy 0 -SSC on the compressed dataX:

If p < d, Noisy-DR-0 -SSC operates on the compressed dataX rather than on the original data, so that the efficiency is improved.

High-dimensional data often exhibits low-dimensional structures, which often leads to low-rankness of the data matrix.

Intuitively, if the data is low rank, then it could be safe to perform noisy 0 -SSC on its dimensionality reduced version by the linear projection P, and it is expected that P can preserve the information of the subspaces contained in the original data as much as possible, while effectively removing uninformative dimensions.

To this end, we propose to choose P as a random projection induced by randomized low-rank approximation of the data.

The key idea is to obtain an approximate low-rank decomposition of the data.

Using the random projection induced by such low-rank approximation as the linear transformation P, the clustering correctness hold for Noisy-DR-0 -SSC with a high probability.

Randomized algorithms are efficient and they have been extensively studied in the computer science and numerical linear algebra literature.

They have been employed to accelerate various numerical matrix computation and matrix optimization problems, including random projection for matrix decomposition Formally, a random matrix T ∈ R n×p is generated such that each element T ij is sampled independently according to the Gaussian distribution N (0, 1).

QR decomposition is then performed on XT to obtain the basis of its column space, namely XT = QR where Q ∈ R d×p is an orthogonal matrix of rank p and R ∈ R p×p is an upper triangle matrix.

The columns of Q form the orthogonal basis for the sample matrix XT.

An approximation of X is then obtained by projecting X onto the column space of XT: QQ X = QW =X where W = Q X ∈ R p×n .

In this manner, a randomized low-rank decomposition of X is achieved as follows:

We present probabilistic result on the correctness of Noisy-DR-0 -SSC using the random projection induced by randomized low-rank decomposition of the data X, namely P = Q , in Theorem 4.

In the sequel,x = Px for any x ∈ R n .

To guarantee the subspace detection property on the dimensionality-reduced dataX, it is crucial to make sure that the conditions, such as (12) and (13) in Theorem 2, still hold after the linear transformation.

We denote byβ * the optimal solution to (25).

We also define the following quantities in the analysis of the subspace detection property, which correspond to M i ,σ Y ,r , σ X,r and µ r used in the analysis on the original data:

where Hỹ

is all the external subspaces ofỹ i with dimension no greater thand k in the transformed space by P.σỸ

Theorem 4. (Subspace detection property holds for Noisy-DR-0 -SSC under deterministic model) Let nonzero vector β * be the optimal solution to the noisy 0 -SSC problem (6) for point x i with β * 0 = r * , n k ≥ d k + 1 for every 1 ≤ k ≤ K, and there exists 1 < r 0 ≤ d such that 1 < r * ≤ r 0 .

Suppose Y is in general position, δ < min 1≤r<r0σY ,r , andM i,δ M i − δ.

Suppose the following conditions hold:

for all y i ∈ S k and 1 ≤ k ≤ K.

then with probability at least 1 − 6e −p , the subspace detection property holds forx i withβ * .

Herẽ M i ,μ r andσX ,r0 are defined in (27), (30) and (29) respectively.

We employ Proximal Gradient Descent (PGD) to optimize the objective function of noisy 0 -SSC and Noisy-DR-0 -SSC.

For example, in the k-th iteration of PGD for problem (6), the variable β is updated according to

where g(β) x i − Xβ 2 2 , T θ is an element-wise hard thresholding operator:

It is proved in Yang & Yu (2019) that the sequence {β (k) } generated by PGD converges to a critical point of (6), denoted byβ.

Let β * be the optimal solution to (6).

Theorem 5 in Yang & Yu (2019) to problem (6) shows that the β * −β 2 is bounded.

Theorem 5 establishes the conditions under whichβ is also the optimal solution to (6).

The following theorem demonstrates thatβ = β * if λ is two-side bounded andβ min = min t:βt =0 |β t | is sufficiently large.

Theorem 5. (Conditions that the sub-optimal solution by PGD is also globally optimal) If

and µ

thenβ = β * .

We demonstrate the performance of noisy 0 -SSC and Noisy-DR-0 -SSC, with comparison to other competing clustering methods including K-means (KM), Spectral Clustering (SC), noisy SSC, Sparse Manifold Clustering and Embedding (SMCE) Elhamifar & Vidal (2011) and SSC-OMP Dyer et al. (2013) .

With the coefficient matrix Z obtained by the optimization of noisy 0 -SSC or Noisy-DR-0 -SSC, a sparse similarity matrix is built by W =

, and spectral clustering is performed on W to obtain the clustering results.

Two measures are used to evaluate the performance of different clustering methods, i.e. the Accuracy (AC) and the Normalized Mutual Information (NMI) Zheng et al. (2004) .

We use randomized rank-p decomposition of the data matrix in Noisy-DR-0 -SSC with p = min{d,n} 10 .

It can be observed that noisy 0 -SSC and Noisy-DR-0 -SSC always achieve better performance than other methods in Table 1 , including the noisy SSC on dimensionality reduced data (Noisy DR-SSC) Wang et al. (2015) .

Throughout all the experiments we find that the best clustering accuracy is achieved whenever λ is chosen by 0.5 < λ < 0.95, justifying our theoretical finding claimed in Remark 4 and (39) in Theorem 5.

More experimental results on the CMU Multi-PIE data are shown in Table 2 .

For all the methods that involve random projection, we conduct the experiments for 30 times and report the average performance.

Note that the cluster accuracy of SSC-OMP on the extended Yale-B data set is reported according to You et al. (2016) .

The time complexity of running PGD for noisy 0 -SSC and Noisy-DR-0 -SSC are O(T nd) and O(T pd) respectively, where T is the maximum iteration number.

The actual running time of both algorithms confirms such time complexity, and we observe that Noisy-DR-0 -SSC is always more than 8.7 times faster than noisy 0 -SSC with the same number of iterations.

We present provable noisy 0 -SSC that recovers subspaces from noisy data through 0 -induced sparsity in a robust manner, with the theoretical guarantee on its correctness in terms of subspace detection property under both deterministic and randomized models.

Experimental results shows the superior performance of noisy 0 -SSC.

We also propose Noisy-DR-0 -SSC which performs noisy 0 -SSC on dimensionality reduced data and still provably recovers the subspaces in the original data.

Experiment results demonstrate the effectiveness of both noisy 0 -SSC and Noisy-DR-0 -SSC.

β = 0.

Perform the above analysis for all 1 ≤ i ≤ n, we can prove that the subspace detection property holds for all 1 ≤ i ≤ n.

The following proposition is used for proving Lemma 1.

Lemma B. (Perturbation of distance to subspaces) Let A, B ∈ R m×n are two matrices and rank(A) = r, rank(B) = s. Also, E = A − B and E 2 ≤ C, where · 2 indicates the spectral norm.

Then for any point x ∈ R m , the difference of the distance of x to the column space of A and B, i.e. |d(x,

Proof.

Note that the projection of x onto the subspace H A is AA + x where A + is the Moore-Penrose pseudo-inverse of the matrix A, so d(x, H A ) equals to the distance between x and its projection, namely

It follows that

According to the perturbation bound on the orthogonal projection in Chen et al. (2016); Stewart (1977) ,

Since EA

, combining (42) and (43)

So that (5) is proved.

Proof of Lemma 1.

We have yi = xi − ni, and σmin(

By Weyl Weyl (1912)

It follows that σmin(X β ) ≥ σ Y ,r − √ rδ > 0 and X β has full column rank.

Also, X β − Y β 2 ≤ X β − Y β F ≤ √ rδ.

According to Lemma B,

A.3 PROOF OF LEMMA 2

Proof.

xi − Xβ * 2 2 + λ β * 0 ≤ xi − X0 2 2 + λ 0 0 = 1 ⇒ c * = xi − Xβ * 2 < 1.

We first prove that β * is the optimal solution to the sparse approximation problem min β β 0 s.t.

xi − Xβ 2 ≤ c * , βi = 0.

To see this, suppose there is a vector β such that xi − Xβ 2 ≤ c * and β 0 < β that y − xi 2 ≤ c0 since c0 ≥ d(xi, S k ).

Also, d k points in Y (k) can linearly represent y since Y (k) is in general position, and it follows that β and V A have orthonormal columns with U A U A = V A V A = I. Then QA = U QA ΣV QA is the singular value decomposition of QA with U QA = QU A and V QA = V A .

This is because the columns of U QA are orthonormal since the columns Q are orthonormal: U QA U QA = U A Q QU A = I, and Σ is a diagonal matrix with nonnegative diagonal elements.

It follows that σmin(QA) = σmin(A) for any A ∈ R p×q .

For a point xi = yi + ni, after projection via P, we have the projected noiseñi = Pni.

Because

the magnitude of the noise in the projected data is also bounded by δ.

Also,

Let β ∈ R n ,Ỹ β = PY β with β 0 = r. Then σmin(QỸ β ) = σmin(Ỹ β )).

Since

Therefore, it follows from (59) that if

thenỸ is also in general position.

In addition, since λ ≥ 1 r 0

, we have λ β * 0 ≤ L(0) ≤ 1, and it follows that β * 0 ≤ 1 λ ≤ r0.

Based on (59) we have |σỸ ,r −σ Y ,r | ≤ Cp,p 0 + 2δ

it follows that δ < min 1≤r<r 0σỸ ,r because δ < min 1≤r<r 0σ Y ,r − Cp,p 0 − 2δ √ r0.

Again, for β ∈ R n with β 0 = r ≤ r0, we have |σmin(X β ) − σmin(X β )| = |σmin(QX β ) − σmin(X β )|

It can be verified that |σX ,r − σ X,r | ≤ Cp,p 0

Combining (63) and Lemma D, noting that σ X,r 0 − Cp,p 0 , since

we haveM i,δ

where yi ∈ S k .

Based on (61) and (63), we haveμ

@highlight

We propose Noisy-DR-L0-SSC (Noisy Dimension Reduction L0-Sparse Subspace Clustering) to efficiently partition noisy data in accordance to their underlying subspace structure.
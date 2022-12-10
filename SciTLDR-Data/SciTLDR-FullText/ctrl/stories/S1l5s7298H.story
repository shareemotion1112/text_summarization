We introduce a “learning-based” algorithm for the low-rank decomposition problem: given an $n \times d$ matrix $A$, and a parameter $k$, compute a rank-$k$ matrix $A'$ that minimizes the approximation loss $||A-

A'||_F$. The algorithm uses a training set of input matrices in order to optimize its performance.

Specifically, some of the most efficient approximate algorithms for computing low-rank approximations proceed by computing a projection $SA$, where $S$ is a sparse random $m \times n$ “sketching matrix”, and then performing the singular value decomposition of $SA$. We show how to replace the random matrix $S$ with a “learned” matrix of the same sparsity to reduce the error.



Our experiments show that, for multiple types of data sets, a learned sketch matrix can substantially reduce the approximation loss compared to a random matrix $S$, sometimes by one order of magnitude.

We also study mixed matrices where only some of the rows are trained and the remaining ones are random, and show that matrices still offer improved performance while retaining worst-case guarantees.

The success of modern machine learning made it applicable to problems that lie outside of the scope of "classic AI".

In particular, there has been a growing interest in using machine learning to improve the performance of "standard" algorithms, by fine-tuning their behavior to adapt to the properties of the input distribution, see e.g., [1] [2] [3] [4] [5] [6] [7] [8] [9] [10] [11] [12] [13] .

This "learning-based" approach to algorithm design has attracted a considerable attention over the last few years, due to its potential to significantly improve the efficiency of some of the most widely used algorithmic tasks.

Many applications involve processing streams of data (video, data logs, customer activity etc) by executing the same algorithm on an hourly, daily or weekly basis.

These data sets are typically not "random" or "worst-case"; instead, they come from some distribution which does not change rapidly from execution to execution.

This makes it possible to design better algorithms tailored to the specific data distribution, trained on past instances of the problem.

The method has been particularly successful in the context of compressed sensing.

In the latter framework, the goal is to recover an approximation to an n-dimensional vector x, given its "linear measurement" of the form Sx, where S is an m × n matrix.

Theoretical results [14, 15] show that, if the matrix S is selected at random, it is possible to recover the k largest coefficients of x with high probability using a matrix S with m = O(k log n) rows.

This guarantee is general and applies to arbitrary vectors x. However, if vectors x are selected from some natural distribution (e.g., they represent images), recent works [8, 9, 11] show that one can use samples from that distribution to compute matrices S that improve over a completely random matrix in terms of the recovery error.

Compressed sensing is an example of a broader class of problems which can be solved using random projections.

Another well-studied problem of this type is low-rank decomposition: given an n × d matrix A, and a parameter k, compute a rank-k matrix

Low-rank approximation is one of the most widely used tools in massive data analysis, machine learning and statistics, and has been a subject of many algorithmic studies.

In particular, multiple algorithms developed over the last decade use the "sketching" approach, see e.g., [16] [17] [18] [19] [20] [21] [22] [23] [24] .

Its idea is to use efficiently computable random projections (a.k.a., "sketches") to reduce the problem size before performing low-rank decomposition, which makes the computation more space and time efficient.

For example, [16, 19] show that if S is a random matrix of size m × n chosen from an appropriate distribution, for m depending on , then one can recover a rank-k matrix A such that

by performing an SVD on SA ∈ R m×d followed by some post-processing.

Typically the sketch length m is small, so the matrix SA can be stored using little space (in the context of streaming algorithms) or efficiently communicated (in the context of distributed algorithms).

Furthermore, the SVD of SA can be computed efficiently, especially after another round of sketching, reducing the overall computation time.

See the survey [25] for an overview of these developments.

In light of the aforementioned work on learning-based compressive sensing, it is natural to ask whether similar improvements in performance could be obtained for other sketch-based algorithms, notably for low-rank decompositions.

In particular, reducing the sketch length m while preserving its accuracy would make sketch-based algorithms more efficient.

Alternatively, one could make sketches more accurate for the same values of m. This is the problem we address in this paper.

Our Results.

Our main finding is that learned sketch matrices can indeed yield (much) more accurate low-rank decompositions than purely random matrices.

We focus our study on a streaming algorithm for low-rank decomposition due to [16, 19] , described in more detail in Section 2.

Specifically, suppose we have a training set of matrices Tr = {A 1 , . . .

, A N } sampled from some distribution D. Based on this training set, we compute a matrix S * that (locally) minimizes the empirical loss

where SCW(S * , A i ) denotes the output of the aforementioned Sarlos-Clarkson-Woodruff streaming low-rank decomposition algorithm on matrix A i using the sketch matrix S * .

Once the the sketch matrix S * is computed, it can be used instead of a random sketch matrix in all future executions of the SCW algorithm.

We demonstrate empirically that, for multiple types of data sets, an optimized sketch matrix S * can substantially reduce the approximation loss compared to a random matrix S, sometimes by one order of magnitude (see Figure 1) .

Equivalently, the optimized sketch matrix can achieve the same approximation loss for lower values of m.

A possible disadvantage of learned sketch matrices is that an algorithm that uses them no longer offers worst-case guarantees.

As a result, if such an algorithm is applied to an input matrix that does not conform to the training distribution, the results might be worse than if random matrices were used.

To alleviate this issue, we also study mixed sketch matrices, where (say) half of the rows are trained and the other half are random.

We observe that if such matrices are used in conjunction with the SCW algorithm, its results are no worse than if only the random part of the matrix was used 2 .

Thus, the resulting algorithm inherits the worst-case performance guarantees of the random part of the sketching matrix.

At the same time, we show that mixed matrices still substantially reduce the approximation loss compared to random ones, in some cases nearly matching the performance of "pure" learned matrices with the same number of rows.

Thus, mixed random matrices offer "the best of both worlds": improved performance for matrices from the training distribution, and worst-case guarantees otherwise.

Notation.

Consider a distribution D on matrices A ∈ R n×d .

We define the training set as {A 1 , · · · , A N } sampled from D. For matrix A, its singular value decomposition (SVD) can be written as A = U ΣV such that both U and V have orthonormal columns and Σ = diag{λ 1 , · · · , λ d } is a diagonal matrix with nonnegative entries.

In many applications it is quicker and more economical to Algorithm 1 Rank-k approximation of a matrix A using a sketch matrix S (from Section 4.1.1 of

How sketching works.

We start by describing the SCW algorithm (Algorithm 1) for low-rank approximation.

The algorithm computes the SVD(SA) := U ΣV , and compute the best rank-k approximation of AV .

Finally it outputs [AV ] k V as a rank-k approximation of A. Note that if m is much smaller than d and n, the space bound of this algorithm is significantly better than when computing a rank-k approximation for A in the naïve way.

Thus, minimizing m automatically reduces the space usage of the algorithm.

Sketching matrix.

We use matrix S that is sparse.

Specifically, each column of S has exactly one non-zero entry, which is either +1 or −1.

This means that the fraction of non-zero entries in S is 1/m.

Therefore, one can use a vector to represent S, which is very memory efficient.

It is worth noting, however, after multiplying S with other matrices, the resulting matrix is in general not sparse.

In this section, we describe our learning-based algorithm for computing a data dependent sketch S. The main idea is to use backpropagation algorithm to compute the stochastic gradient of S with respect to the rank-k approximation loss in Equation 1, where the initial value of S is the same random sparse matrix used in SCW.

Once we have the stochastic gradient, we can run stochastic gradient descent (SGD) algorithm to optimize S, in order to improve the loss.

Our algorithm maintains the sparse structure of S, and only optimizes the values of the n non-zero entries (initially +1 or −1).

However, the standard SVD implementation (step 2 in Algorithm 1 ) is not differentiable, which means we cannot get the gradient in the straightforward way.

To make SVD implementation differentiable, we use the fact that the SVD procedure can be represented as m individual top singular value decompositions (see e.g. [26] ), and that every top singular value decomposition can be computed using the power method.

The full description is deferred to the long version of this paper.

Due to the extremely long computational chain, it is infeasible to write down the explicit form of loss function or the gradients.

However, just like how modern deep neural networks compute their gradients, we used the autograd feature in PyTorch to numerically compute the gradient with respect to the sketching matrix S.

We emphasize again that our method is only optimizing S for the training phase.

After S is fully trained, we still call Algorithm 1 for low rank approximation, which has exactly the same running time as the SCW algorithm, but with better performance.

The main question considered in this paper is whether, for natural matrix datasets, optimizing the sketch matrix S can improve the performance of the sketching algorithm for the low-rank decomposition problem.

To answer this question, we implemented and compared the following methods for computing S ∈ R m×n .

• Sparse Random.

Sketching matrices are generated at random as in [20] .

Specifically, we select a random hash function h :

, and for all i ∈ [n], S h[i],i is selected to be either +1 or −1 with equal probability.

All other entries in S are set to 0.

• Dense Random.

All entries in the sketching matrices are sampled from Gaussian distribution.

• Learned.

Using the sparse random matrix as the initialization, we optimize the sketching matrix using the training set, and return the optimized matrix.

• Mixed (J).

We first generate two sparse random matrices S 1 , S 2 ∈ R m 2 ×n (assuming m is even), and define S to be their combination.

We then optimize S using the training set, but only S 1 will be updated, while S 2 is fixed.

Therefore, S is a mixture of learned matrix and random matrix, and the first matrix is trained jointly with the second one.

• Mixed (S).

We first compute a learned matrix S 1 ∈ R m 2 ×n using the training set, and then append another sparse random matrix S 2 to get S ∈ R m×n .

Therefore, S is a mixture of learned matrix and random matrix, but the learned matrix is trained separately.

Datasets.

We used a variety of datasets to test the performance of our methods:

• Videos 3 : Logo, Friends, Eagle.

We downloaded three high resolution videos from Youtube, including logo video, Friends TV show, and eagle nest cam.

From each video, we collect 500 frames of size 1920 × 1080 × 3 pixels, and use 400 (100) matrices as the training (test) set.

For each frame, we resize it as a 5760 × 1080 matrix.

• Hyper.

We use matrices from HS-SOD, a dataset for hyperspectral images from natural scenes [27] .

Each matrix has 1024 × 768 pixels, and we use 400 (100) matrices as the training (test) set.

• Tech.

We use matrices from TechTC-300, a dataset for text categorization [28] .

Each matrix has 835, 422 rows, but on average only 25, 389 of the rows contain non-zero entries.

On average each matrix has 195 columns.

We use 200 (95) matrices as the training (test) set.

To evaluate the quality of a sketching matrix S, it suffices to evaluate the output of Algorithm 1 using the sketching matrix S on different input matrices A. For a collection of matrices Te, we define the error of the sketch S as Err(Te, S)

where the second term denotes the optimal approximation loss on Te.

In our datasets, some of the matrices have much larger singular values than the others.

To avoid imbalance in the dataset, we normalize the matrices so that their top singular values are all equal.

We first test all methods on different datasets, with various combination of k, m. See Figure 1 for the results when k = 10, m = 20.

As we can see, for video datasets, learned sketching matrices can get 20× better test error than the sparse random or dense random sketching matrices.

For other datasets, learned sketching matrices are still more than 2× better.

We also include the test error results in Table  1 for the case when k = 20, 30.

In Table 2 , we investigate the performance of the mixed sketching matrices by comparing them with random and learned sketching matrices.

In all scenarios, mixed sketching matrices yield much better results than random sketching matrices, and sometimes the results are comparable to those of learned sketching matrices.

This means, in most cases it suffices to train one half of the sketching matrix to obtain good empirical results, and at the same time, we can use the remaining random half of the sketch matrix to obtain worst-case guarantees.

<|TLDR|>

@highlight

Learning-based algorithms can improve upon the performance of classical algorithms for the low-rank approximation problem while retaining the worst-case guarantee.
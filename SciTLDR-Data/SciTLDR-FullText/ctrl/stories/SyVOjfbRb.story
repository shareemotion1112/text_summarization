Stochastic Gradient Descent or SGD is the most popular optimization algorithm for large-scale problems.

SGD estimates the gradient by uniform sampling with sample size one.

There have been several other works that suggest faster epoch wise convergence by using weighted non-uniform sampling for better gradient estimates.

Unfortunately, the per-iteration cost of maintaining this adaptive distribution for gradient estimation is more than calculating the full gradient.

As a result, the false impression of faster convergence in iterations leads to slower convergence in time, which we call a chicken-and-egg loop.

In this paper, we break this barrier by providing the first demonstration of a sampling scheme, which leads to superior gradient estimation, while keeping the sampling cost per iteration similar to that of the uniform sampling.

Such an algorithm is possible due to the sampling view of Locality Sensitive Hashing (LSH), which came to light recently.

As a consequence of superior and fast estimation, we reduce the running time of all existing gradient descent algorithms.

We demonstrate the benefits of our proposal on both SGD and AdaGrad.

Stochastic gradient descent or commonly known as SGD is the most popular choice of optimization algorithm in large-scale setting for its computational efficiency.

A typical interest in machine learning is to minimize the average loss function f over the training data, with respect to the parameters ✓, i.e., the objective function of interest is DISPLAYFORM0 Throughout the paper, our training data D = {x i , y i } N i=1 will have N instances with d dimensional features x i 2 R d and labels y i .

The labels can be continuous real valued for regression problems.

For classification problem, they will take value in a discrete set, i.e., y i 2 {1, 2, · · · , K}. Typically, the function f is a convex function.

The least squares f (x i , ✓) = (✓ · x i y i ) 2 , used in regression setting is a classical example of f .

SGD BID1 samples an instance x j uniformly from N instances, and performs the gradient descent update: DISPLAYFORM1 where ⌘ t is the step size at the t th iteration.

The gradient rf (x j , ✓ t 1 ) is only evaluated on x j , using the current ✓ t 1 .

DISPLAYFORM2 rf (x i , ✓ t 1 ).

Thus, a uniformly sampled gradient rf (x j , ✓ t 1 ) is an unbiased estimator of the full gradient, i.e., DISPLAYFORM3 This is the key reason why, despite only using one sample, SGD still converges to the local minima, analogously to full gradient descent, provided ⌘ t is chosen properly BID13 BID1 .However, it is known that the convergence rate of SGD is slower than that of the full gradient descent BID14 .

Nevertheless, the cost of computing the full gradient requires O(N ) evaluations of rf compared to just O(1) evaluation in SGD.

Thus, with the cost of one epoch of full gradient descent, SGD can perform O(N ) epochs, which overcompensates the slow convergence.

Therefore, despite slow convergence rates, SGD is almost always the chosen algorithm in large-scale settings as the calculation of the full gradient in every epoch is prohibitively slow.

Further improving SGD is still an active area of research.

Any such improvement will directly speed up almost all the state-of-the-art algorithms in machine learning.

The slower convergence of SGD is expected due to the poor estimation of the gradient (the average) by only sampling a single instance uniformly.

Clearly, the variance of the one sample estimator is high.

As a result, there have been several efforts in finding sampling strategies for better estimation of the gradients BID19 BID12 Zhao & Zhang, 2015; BID0 .

The key idea behind these methods is to replace the uniform distribution with a weighted distribution which leads tp a lower variance.

However, with all adaptive sampling methods for SGD, whenever the parameters and the gradients change, the weighted distribution has to change.

Unfortunately, as argued in BID8 , all of these methods suffer from what we call the chicken-and-egg loop -adaptive sampling improves stochastic estimation but maintaining the required adaptive distribution will cost O(N ) per iteration, which is also the cost of computing the full gradient exactly.

To the best of our knowledge, there does not exist any generic sampling scheme for adaptive gradient estimation, where the cost of maintaining and updating the distribution, per iteration, is O(1) which is comparable to SGD.

Our work provides first such sampling scheme utilizing the recent advances in sampling and unbiased estimation using Locality Sensitive Hashing BID17 ).

For non-uniform sampling, we can sample each x i with an associated weight w i .

These w i 's can be tuned to minimize the variance.

It was first shown in BID0 , that sampling x i with probability in proportion to the L 2 norm of the gradient, i.e. ||rf (x i , ✓ t 1 )|| 2 , leads to the optimal distribution that minimizes the variance.

However, sampling x i with probability in proportion to w i = ||rf (x i , ✓ t 1 )|| 2 , requires first computing all the w i 's, which change in every iteration because ✓ t 1 gets updated.

Therefore, maintaining the values of w i 's is even costlier than computing the full gradient.

BID8 proposed to mitigate this overhead partially by exploiting additional side information such as the cluster structure of the data.

Prior to the realization of optimal variance distribution, BID19 and BID12 proposed to sample a training instance with a probability proportional to the Lipschitz constant of the function f (x i , ✓ t 1 ) or rf (x i , ✓ t 1 ) respectively.

Again, as argued, in BID8 , the cost of maintaining the distribution is prohibitive.

It is worth mentioning that before these works, a similar idea was used in designing importance sampling-based low-rank matrix approximation algorithms.

The resulting sampling methods, known as leverage score sampling, are again proportional to the squared Euclidean norms of rows and columns of the underlying matrix BID6 .The Chicken-and-Egg Loop: In summary, to speed up the convergence of stochastic gradient descent, we need non-uniform sampling for better estimates (low variance) of the full gradient.

Any interesting non-uniform sampling is dependent on the data and the parameter ✓ t which changes in every iteration.

Thus, maintaining the non-uniform distribution for estimation requires O(N ) computations to calculate the weights w i , which is the same cost computing it exactly.

It is not even clear that there exists any sweet and adaptive distribution which breaks this computational chickenand-egg loop.

We provide the first affirmative answer by giving an unusual distribution which is derived from probabilistic indexing based on locality sensitive hashing.

Our Contributions: In this work, we propose a novel LSH-based samplers, that breaks the aforementioned chicken-and-egg loop.

Our algorithm, which we call LSD (LSH Sampled Stochastic gradient Descent), are generated via hash lookups which have O(1) cost.

Moreover, the probability of selecting x i is provably adaptive.

Therefore, the current gradient estimates have lower variance, compared to a single sample SGD, while the computational complexity of sampling is constant and of the order of SGD sampling cost.

Furthermore, we demonstrate that LSD can be utilized to speed up any existing gradient-based optimization algorithm such as AdaGrad BID7 .As a direct consequence, we obtain a generic and efficient gradient descent algorithm which converges significantly faster than SGD, both in terms of epochs as well as running time.

It should be noted that rapid epoch wise convergence alone does not imply computational efficiency.

For instances, Newtons method converges faster, epoch wise, than any first-order gradient descent, but it is prohibitively slow in practice.

The wall clock time or the amount of floating point operations performend to reach convergence should be the metric of consideration for useful conclusions.

Accuracy Vs Running Time: It is rare to see any fair (same computational setting) empirical comparisons of SGD with existing adaptive SGD schemes, which compare the improvement in accuracy with respect to running time on the same computational platform.

Almost all methods compare accuracy with the number of epochs, which is unfair to SGD which can complete O(N ) epochs at the computational cost (or running time) of one epoch for adaptive sampling schemes.

We first describe a recent advancement in the theory of sampling and estimation using locality sensitive hashing (LSH) BID9 which will be heavily used in our proposal.

Before we get into details of sampling, let us revise the two-decade-old theory of LSH.

Locality-Sensitive Hashing (LSH) BID9 ) is a popular, sub-linear time algorithm for approximate nearest-neighbor search.

The high-level idea is to place similar items into the same bucket of a hash table with high probability.

An LSH hash function maps an input data vector to an integer key DISPLAYFORM0 A collision occurs when the hash values for two data vectors are equal: h(x) = h(y).

The collision probability of most LSH hash functions is generally a monotonic function of the similarity DISPLAYFORM1 where M is a monotonically increasing function.

Essentially, similar items are more likely to collide with each other under the same hash fingerprint.

The algorithm uses two parameters, (K, L).

We construct L independent hash tables from the collection C.

Each hash table has a meta-hash function H that is formed by concatenating K random independent hash functions from F. Given a query, we collect one bucket from each hash table and return the union of L buckets.

Intuitively, the meta-hash function makes the buckets sparse and reduces the number of false positives, because only valid nearest-neighbor items are likely to match all K hash values for a given query.

The union of the L buckets decreases the number of false negatives by increasing the number of potential buckets that could hold valid nearest-neighbor items.

The candidate generation algorithm works in two phases [See BID17 for details]:1.

Pre-processing Phase: We construct L hash tables from the data by storing all elements x 2 C. We only store pointers to the vector in the hash tables because storing whole data vectors is very memory inefficient.2.

Query Phase: Given a query Q; we will search for its nearest-neighbors.

We report the union from all of the buckets collected from the L hash tables.

Note, we do not scan all the elements in C, we only probe L different buckets, one bucket for each hash table.

After generating the set of potential candidates, the nearest-neighbor is computed by comparing the distance between each item in the candidate set and the query.

An item returned as candidate from a (K, L)-parameterized LSH algorithm (section 3.2) is sampled with probability 1 (1 p K ) L , where p is the collision probability of LSH function.

The LSH family defines the precise form of p used to build the hash tables.

This sampling view of LSH was first utilized to perform adaptive sparsification of deep networks in near-constant time, leading to efficient backpropagation algorithm BID16 .A year later, BID17 ) demonstrated the first theory of using these samples for unbiased estimation of partition functions in log-linear models.

More specifically, the authors showed that since we know the precise probability of sampled elements 1 (1 p K ) L , we could design provably unbiased estimators using importance sampling type idea.

This was the first demonstration that random sampling could be beaten with roughly the same computational cost as vanilla sampling.

BID11 used the same approach for unbiased estimation of anomaly scoring function. (Charikar & Siminelakis) rigorously formalized these notions and showed provable improvements in sample complexity of kernel density estimation problems.

Recently, used the sampling in a very different context of connected component estimation for unique entity counts.

Recent advances in maximum inner product search (MIPS) using asymmetric locality sensitive hashing has made it possible to sample large inner products.

For this paper, it is safe to assume that given a collection C of vectors and query vector Q, using (K, L)-parameterized LSH algorithm with MIPS hashing BID15 , we get a candidate set S that every element x i 2 C is sampled with probability p i  1, where p i is a monotonically increasing function of Q · x i .

Thus, we can pay a one-time linear cost of preprocessing C into hash tables, and any further adaptive sampling for query Q only requires few hash lookups.

We can also compute the probability of getting x.

Before getting into our main algorithm where we use the above sampling process for estimation, we would like to cover some of its properties.

To begin with, the sampling scheme is not a valid distribution, i.e., P xi2C p i 6 = 1.

In addition, given a query, the probability of sampling x i is not independent of the probability of sampling x j (i 6 = j).

However, we can still use it for unbiased estimation.

Details of such sampling are included in BID17 .

In fact, the form of sampling probability p i is quite unusual.

DISPLAYFORM0 is the collision probability.3 THE LSD ALGORITHM

Our algorithm leverages the efficient estimations using locality sensitive hashing, which usually beats random sampling estimators while keeping the sampling cost near-constant.

We first provide the intuition of our proposal, and the analysis will follow.

Consider least squares regression with loss function DISPLAYFORM0 2 , where ✓ t is the parameter in the t th iteration.

The gradient is just like a partition function.

If we simply follow the procedures in BID17 , we can easily show a generic unbiased estimator via adaptive sampling.

However, better sampling alternatives are possible.

Observing that the gradient, with respect to ✓ t concerning x i , is given by 2(y DISPLAYFORM1 , the L 2 norm of the gradient can therefore be written as an absolute value of inner product.

according to BID0 , the L 2 norm of the gradient is also the optimal sampling weight w DISPLAYFORM2 where h✓ t , 1i is a vector concatenation of ✓ with 1.

If the data is normalized then we should sample x i in proportion to w i ⇤ = h✓ t , 1i · hx i , y i i , i.e. large magnitude inner products should be sampled with higher probability.

As argued, such sampling process is expensive because w ⇤ i changes with ✓ t .

We address this issue by designing a sampling process that does not exactly sample with probability w ⇤ i but instead samples from a different weighted distribution which is a monotonic function of w ⇤ i .

Specifically, we sample from w lsh i = f (w ⇤ i ), where f is some monotonic function.

Before we describe the efficient sampling process, we first argue that a monotonic sampling is a good choice for gradient estimation.

DISPLAYFORM3 DISPLAYFORM4 preprocessed training data vectors x lsh , y lsh and then put hx i lsh , y i lsh i into LSH Data structure.

Get x 0 train , y 0 train from preprocessed data t = 0 while NotConverged do xFor any monotonic function f , the weighted distribution w lsh i = f (w ⇤ i ) is still adaptive and changes with ✓ t .

Also, due to monotonicity, if the optimal sampling prefers x i over x j i.e. w ⇤ i w ⇤ j , then monotonic sampling will also have same preference, i.e., w lsh i w lsh j .

The key insight is that there are two quantities in the inner product (equation 4), h✓ t , 1i and hx i , y i i. With successive iteration, h✓ t , 1i changes while hx i , y i i is fixed.

Thus, it is possible to preprocess hx i , y i i into hash tables (one time cost) and query with h✓ t , 1i for efficient and adaptive sampling.

With every iteration, only the query changes to h✓ t+1 , 1i, but the hash tables remains the same.

Few hash lookups are sufficient to sample x i for gradient estimation adaptively.

Therefore, we only pay one-time preprocessing cost of building hash tables and few hash lookups, typically just one, in every iteration to get a sample for estimation.

There are few more technical subtleties due to the absolute value of inner product h✓ t , 1i·hx i , y i i , rather than the inner product itself.

However, the square of the absolute value of the inner product DISPLAYFORM5 , can also be written as an inner product as it is a quadratic kernel, and T is the corresponding feature expansion transformation.

Again square is monotonic function, and therefore, our sampling is still monotonic as composition of monotonic functions is monotonic.

Thus, technically we hash T (hx i , y i i) to create hash tables and the query at t th step is T (h✓ t , 1i).

Once an x i is sampled via LSH sampling (Algorithm 2), we can precisely compute the probability of its sampling, i.e., p i (See section 2).

It is not difficult to show that our estimation of full gradient is unbiased (Section 3.3).

We first describe the detailed step of our gradient estimator in Algorithm 1.

We also provide the sampling algorithm 2 with detail.

Assume that we have access to the right LSH function h, and DISPLAYFORM0 its collision probability expression cp(x, y) = P r(h(x) = h(y)).

For linear regression, we can use signed random projections, simhash BID3 , or MIPS hashing.

With normalized data, simhash collision probability is cp(x, y) = 1 DISPLAYFORM1 , which is monotonic in the inner product.

Furthermore, we centered the data we need to store in the LSH hash table to make the simhash query more efficient.

The computational cost of SGD sampling is merely a single random number generator.

The cost of gradient update (equation 2) is one inner product, which is d multiplications.

If we want to design an adaptive sampling procedure that beats SGD, the sampling cost cannot be significantly larger than d multiplications.

The cost of LSD sampling (Algorithm 2) is K ⇥ l hash computations followed by l + 1 random number generator, (1 extra for sampling from the bucket).

Since the scheme works for any K, we can always choose K small enough so that empty buckets are rare (see BID17 ).

In all of our experiments, K = 5 for which l is almost always 1.

Thus, we require K hash computations and only two random number generations.

If we use very sparse random projections, then K hash computations only require a constant ⌧ d multiplications.

For example, in all our experiments we only need d 30 multiplication, in expectation, to get all the hashes using sparse projections.

Therefore, our sampling cost is significantly less than d multiplication which is the cost of gradient update.

Using fast hash computation is critical for our method to work in practice.

It might be tempting to use approximate near-neighbor search with query ✓ t to find x i .

Nearneighbor search has been used in past BID5 to speed up coordinate descent.

However, near-neighbor queries are expensive due to candidate generation and filtering.

It is still sub-linear in N (and not constant).

Thus, even if we see epoch wise faster convergence, iterations with a nearneighbor query would be orders of magnitude slower than a single SGD iteration.

Moreover, the sampling probability of x cannot be calculated for near-neighbor search which would cause bias in the gradient estimates.

It is important to note that although LSH is heavily used for near-neighbor search, in our case, we use it as a sampler.

For efficient near neighbor search, K and L grow with N BID9 .

In contrast, the sampling works for any K and l 1 as small as one leading to only approximately 1.5 times the cost of SGD iteration (see section 4).

Efficient unbiased estimation is the key difference that makes sampling practical while near-neighbor query prohibitive.

It is unlikely that a nearneighbor query would beat SGD in time, while sampling would.

In this section, we first prove that our estimator of the gradient is unbiased with lower variance than SGD for most real datasets.

Define S as the bucket that contains the sample x from in Algorithm 2.

For simplicity we denote the query as ✓ t and p i = 1 (1 cp(x i , ✓ t ) K ) l as the probability of finding x i in bucket S. Theorem 1.

The following expression is an unbiased estimator of the full gradient DISPLAYFORM0 DISPLAYFORM1 Theorem 2.

The Trace of the covariance of our estimator: DISPLAYFORM2 The trace of the covariance of LSD is the total variance of the descent direction.

The variance can be minimized when the sampling probability of x i is proportional to the L 2 -norm of the gradient we mentioned in Section 1.1.

The intuition of the advantage of LSD estimator comes from sampling x i under a distribution monotonic to the optimal one.

We first make a simple comparison of the variance of LSD with that of SGD theoretically and then in Section 4 and we would further empirically show the drastic superiority of LSD over SGD.

Lemma 1.

The Trace of the covariance of LSD's estimator is smaller than that of SGD's estimator if SGD would perform well if the data is uniformly distributed but it is unlikely in practice.

Recall that the collision probability p i = 1 (1 p K ) l mentioned in Section 2.2.

Note that l here according to Algorithm 2 is the number of tables that have been utlized by the sampling process.

In most practical cases and also in our experiment, K and l are relatively small.

L should be large to ensure enough randomness but it does not show up in the sampling time (See Alg.

2).

LSD can be efficient and achieve a much smaller variance than SGD by setting small values of K and l. It is not difficult to see that if several terms in the summation satisfy |S| piN  1, then the variance of our estimator is better than random sampling.

If the data is clustered nicely, i.e. a random pair has low similarity,by tuning K, we can achieve the above inequality of |S|, p i and N. See Spring & Shrivastava (2017); Charikar & Siminelakis for more details on when LSH sampling is better than random sampling.

DISPLAYFORM3 DISPLAYFORM4

We examine the effectiveness of our algorithm on three large regression dataset, in the area of musical chronometry, clinical computed tomography, and WiFi-signal localization, respectively.

The dataset descriptions and our experiment results are as follows:YearPredictionMSD: BID10 The dataset contains 515,345 instances subset of the Million Song Dataset with dimension 90.

We respect the original train/test split, first 463,715 examples for training and the remaining 51,630 examples for testing, to avoid the 'producer effect' by making sure no song from a given artist ends up in both the train and test set.

Slice: BID10 The data was retrieved from a set of 53,500 CT images from 74 different patients.

It contains 385 features.

We use 42,800 instances as training set and the rest 10,700 instances as the testing set.

As argued before, LSD samples with probability monotonic to L 2 norm of the gradients while SGD samples uniformly.

It matches with the results shown in the plots that LSD queries points with larger gradient than SGD does.

Subplots (d)(e)(f) show the comparison of the cosine similarity between gradient estimated by LSD and the true gradient and the cosine similarity between gradient estimated by SGD and the true gradient.

Note that the variance of both norm and cosine similarity reduce when we average over more samples.

UJIIndoorLoc: (Torres-Sospedra et al., 2014) The database covers three buildings of Universitat Jaume I with 4 or more floors and almost 110,000 m 2 .

It is a collection of 21,048 indoor location information with 529 attributes containing the WiFi fingerprint, the coordinates where it was taken, and other useful information.

We equally split the total instances for training and testing.

All datasets were preprocessed as described in Section 3.2.

Note that for all the experiments, the choice of the gradient decent algorithm was the same.

For both SGD and LSD, the only difference in the gradient algorithm was the gradient estimator.

For SGD, a random sampling estimator was used, while for LSD, the estimator used the adaptive estimator.

We used fixed values K = 5 and L = 100 for all the datasets.

l is the number of hash tables that have been searched before landing in a non-empty bucket in a query.

In our experiments l is almost always as low as 1.

L only affects preprocessing but not sampling.

Our hash function was simhash (or signed random projections) and we used sparse random projections with sparsity 1 30 for speed.

We know that epoch wise convergence is not a true indicator of speed as it hides per epoch computation.

Our main focus is convergence with running time, which is a better indicator of computational efficiency.

To the best of our knowledge, there is no other adaptive estimation baseline, where the cost of sampling per iteration is less than linear O(N ).

Since our primary focus would be on wall clock speedup, no O(N ) estimation method would be able to outperform O(1) SGD (and LSD) estimates on the same platform.

From section 3.2.2, even methods requiring a near-neighbor query would be too costly (orders of magnitude) to outperform SGD from computational perspective.

In the first experiment, we compare vanilla SGD with LSD, i.e., we use simple SGD with fixed learning rate.

This basic experiment aims to demonstrate the performance of pure LSD and SGD without involving other factors like L 1 /L 2 regularizations on linear regression task.

In such a way, we can quantify the superiority of LSD.

FIG0 shows the decrease in the squared loss error with epochs.

Blue lines represent SGD and red lines represent LSD.

It is obvious that LSD converges much faster than SGD in both training and testing loss comparisons.

This is not surprising with the claims in Section 3.2.1 and theoretical proof in Section 3.3.

Since LSD uses slightly more computations per epoch than SGD does, it is hard to defend if LSD gains enough benefits simply from the epoch wise comparisons.

We therefore also show the decrease in error with wall clock time also in figure 1.

Wall clock time is the actual quantification of speedups.

Again, on every single dataset, LSD shows faster time-wise convergence as well.

(Plots for Testing loss are in the supplementary material.)

4.2 LSD+ADAGRAD VS.

SGD+ADAGRAD As argued in section 1.1, our LSD algorithm is complimentary to any gradient-based optimization algorithm.

We repeated the first experiment but using AdaGrad BID7 instead of plain SGD.

Again, other settings are fixed for both algorithms but the only change in the competing algorithm is the gradient estimates per epoch.

In this section, as a sanity check, we first verify weather LSD samples data point with probability monotonic to L 2 norm of the gradient mentioned in section 3.1.

In order to do that, we freeze the optimization at an intermediate iteration and use the ✓ at that moment to sample data points with LSD as well as SGD to compute gradient L 2 norm separately.

The upper three plots in FIG2 show the comparison of the sampled gradient norm of LSD and SGD.

X-axis represents the number of samples that we averaged in the above process.

It is obvious that LSD sampled points have larger gradient norm than SGD ones consistently across all three datasets.

In addition, we also do a sanity check that if empirically, the chosen sample from LSD get better estimation of the true gradient direction than that of SGD.

Again, we freeze the program at an intermediate iteration like the experiments above.

Then we compute the angular similarity of full gradient (average over the training data) direction with both LSD and SGD gradient direction, where, DISPLAYFORM0 .

From the bottom three plots in FIG2 , we can see that in average, LSD estimated gradient has smaller angle (more aligned) to true gradient than SGD estimated gradient.

The variance of both norm and cosine similarity reduce when we average them over more samples as shown in plots.

In this paper, we proposed a novel LSH-based sampler with a reduction to the gradient estimation variance.

We achieved it by sampling with probability proportional to the L 2 norm of the instances gradients leading to an optimal distribution that minimizes the variance of estimation.

More remarkably, LSD is as computationally efficient as SGD but achieves faster convergence not only epoch wise but also time wise.

Peilin Zhao and Tong Zhang.

Stochastic optimization with importance sampling for regularized loss minimization.

In Proceedings of the 32nd International Conference on Machine Learning (ICML-15), pp.

1-9, 2015.A EPOCH PLOTS AND PROOFS Theorem 3.

Let S be the bucket that sample x m is chosen from in Algorithm 2.

Let p m be the sampling probability associated with sample x m .

Suppose we query a sample with ✓ t .

Then we have an unbiased estimator of the full gradient: DISPLAYFORM0 Proof.

DISPLAYFORM1 Theorem 4.

The Trace of the covariance of our estimator is: DISPLAYFORM2 Proof.

Lemma 2.

The Trace of the covariance of LSD's estimator is smaller than that of SGD's estimator if DISPLAYFORM0 Proof.

The trace of covariance of regular SGD is

By FORMULA0 and FORMULA0 , one can easily see that T r(⌃(Est)) < T r(⌃(Est 0 )) when (16) satisfies.

<|TLDR|>

@highlight

We improve the running of all existing gradient descent algorithms.

@highlight

Authors propose sampling stochastic gradients from a monotonic function proportional to gradient magnitudes by using LSH. 

@highlight

Considers SGD over an objective of the form of a sum over examples of a quadratic loss.
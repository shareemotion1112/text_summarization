Online learning has attracted great attention due to the increasing demand for systems that have the ability of learning and evolving.

When the data to be processed is also high dimensional and dimension reduction is necessary for visualization or prediction enhancement, online dimension reduction will play an essential role.

The purpose of this paper is to propose new online learning approaches for supervised dimension reduction.

Our first algorithm is motivated by adapting the sliced inverse regression (SIR), a pioneer and effective algorithm for supervised dimension reduction, and making it implementable in an incremental manner.

The new algorithm, called incremental sliced inverse regression (ISIR), is able to update the subspace of significant factors with intrinsic lower dimensionality fast and efficiently when new observations come in.

We also refine the algorithm by using an overlapping technique  and develop an incremental overlapping sliced inverse regression (IOSIR) algorithm.

We verify the effectiveness and efficiency of both algorithms by simulations and real data applications.

Dimension reduction aims to explore low dimensional representation for high dimensional data.

It helps to promote our understanding of the data structure through visualization and enhance the predictive performance of machine learning algorithms by preventing the "curse of dimensionality".

Therefore, as high dimensional data become ubiquitous in modern sciences, dimension reduction methods are playing more and more important roles in data analysis.

Dimension reduction algorithms can be either unsupervised or supervised.

Principle component analysis (PCA) might be the most popular unsupervised dimension reduction method.

Other unsupervised dimension reduction methods include the kernel PCA, multidimensional scaling, and manifold learning based methods such as isometric mapping and local linear embedding.

Unlike unsupervised dimension reduction, supervised dimension reduction involves a response variable.

It finds the intrinsic lower-dimensional representations that are relevant to the prediction of the response values.

Supervised dimension reduction methods can date back to the well known linear discriminant analysis (LDA) while its blossom occurred in the last twenty years.

Many approaches have been proposed and successfully applied in various scientific domains; see BID21 ; BID9 ; BID22 ; BID35 ; BID31 ; BID14 ; BID24 ; BID33 ; BID34 ; BID10 and the references therein.

We are in a big data era and facing the challenges of big data processing, thanks to the fast development of modern information technology.

Among others, two primary challenges are the big volume and fast velocity of the data.

When a data set is too big to be stored in a single machine or when the data arrives in real time and information update is needed frequently, analysis of the data in an online manner is necessary and efficient.

If the data is simultaneously big and high dimensional, it becomes necessary to develop online learning approaches for dimension reduction.

As PCA and LDA are the most wildly used dimension reduction techniques, a bunch of PCA-based and LDA-based online dimension reduction algorithms has been proposed.

Incremental PCA have been described in BID17 BID18 ; BID32 ; BID43 ; BID29 .

Incremental LDA have been developed in BID27 ; BID42 ; BID20 ; BID7 .

Other strategies like QR decomposition or SVD have also been used in BID6 ; BID36 ; BID28 .In this paper, our purpose is to propose a new online learning approach for supervised dimension reduction.

Our motivation is to implement the sliced inverse regression (SIR) in an incremental manner.

SIR was proposed in BID21 and has become one of the most efficient supervised dimension reduction method.

SIR and its refined versions have been found successful in many scientific areas such as bioinformatics, hyperspectral image analysis, and physics; see BID8 BID4 ; BID3 ; BID12 ; BID15 ; BID19 ; BID0 ; BID23 BID11 ; BID41 .

SIR can be implemented by solving an generalized eigen-decomposition problem, Γβ = λΣβ, where Γ is a matrix depending on the response variable (whose definition is described in the next section) and Σ is the covariance matrix.

To make it implementable in an online manner we rewrite it as standard eigendecomposition problem Σ − 1 2 ΓΣ − 1 2 η = λη where η = Σ 1 2 β and adopt the ideas from incremental PCA.

We need to overcome two main challenges in this process.

First, how do we transform the data so that they are appropriate for the transformed PCA problem?

Note that simply normalizing the data does not work.

Second, online update of Σ − 1 2 , if not impossible, seems very difficult.

The first contribution of this paper is to overcome these difficulties and design a workable incremental SIR method.

Our second contribution will be to refine the method by an overlapping technique and design an incremental overlapping SIR algorithm.

The rest of this paper is arranged as follows.

We review SIR algorithm in Section 2 and the incremental PCA algorithm in Section 3.

We propose the incremental SIR algorithm in Section 4 and refine it in Section 5.

Simulations are done in Section 6.

We close with discussions in Section 7.

The goal of supervised dimension reduction is to find an intrinsic lower-dimensional subspace that contains all the information to predict the response variable.

Assume a multivariate predictor x = (x 1 , x 2 , . . . , x p ) ∈ R p and a scalar response y are linked by a semi-parametric regression model DISPLAYFORM0 where β k ∈ R p is a p × 1 vector and is the error term independent of x. It implies DISPLAYFORM1 where |= denotes "statistical independence" and B = (β 1 , β 2 , . . .

, β K ) is a p × K matrix.

The column space of B is called the effective dimension reduction (EDR) space and each β i is an EDR direction.

Note B x contains all information for the prediction of y. The purpose of supervised dimension reduction is to learn the EDR directions from data.

Unlike the classical regression problem which regresses y against x, sliced inverse regression considers regressing x against y. With the semi-parametric model (1) and the assumption that x follows an elliptical contour distribution (e.g., normal distribution), it was proved in BID21 DISPLAYFORM2 Consequently, all or part of the EDR directions can be recovered by solving a generalized eigenvalue decomposition problem: DISPLAYFORM3 where DISPLAYFORM4 is the covariance matrix of inverse regression curve E[x|y], Σ is the covariance matrix of x. Each eigenvector associated with a non-zero eigenvalue is an EDR direction.

DISPLAYFORM5 , SIR algorithm can be implemented as follows: DISPLAYFORM6 x i and the sample covariance matrix DISPLAYFORM7 2) Bin the observations into H slices according to y values.

For each slice s h , h = 1, . . .

, H, compute the sample probability p h = n h n and the sample slice mean DISPLAYFORM8 3) Solve the generalized eigen-decomposition problem DISPLAYFORM9 The EDR directions are estimated by the top K eigenvectors β k , k = 1, 2, . . .

, K.This algorithm is not very sensitive to the choice of parameter H provided it is sufficiently larger than K while not greater than n 2 .

Root-n consistency is usually promised.

It is suggested samples are evenly distributed into the H slices for the best performance.

PCA looks for directions along which the data have the largest variances.

It is implemented by solving an eigen-decomposition problem DISPLAYFORM0 The principal components are the eigenvectors corresponding to largest eigenvalues.

Throughout this paper, we assume all eigenvalues are arranged in a descending order, i.e., DISPLAYFORM1 Suppose that we need to retain the top K principal components.

Denote DISPLAYFORM2 In incremental PCA, after receiving a new coming observation x 0 , we need to update the reduced eigen-system to a new one DISPLAYFORM3 A "=" is generally impossible unless λ K+1 = . . .

= λ p = 0 in (4).The idea of updating the system in Hall et al. FORMULA0 is as follows.

Compute a residual vector DISPLAYFORM4 wherex is the mean of all observations (including x 0 ).

It defines the component of x 0 that is perpendicular with the subspace defined by U K .

If x 0 lies exactly within the current eigenspace, then the residual vector is zero and there is no need to update the system.

Otherwise, we normalize v to obtainv = v v .

We may reasonably assume each column vector of U K is a linear combination of column vectors of U K andv.

(Note this is exactly true if λ K+1 = . . .

= λ p = 0.)

This allows us to write DISPLAYFORM5 where R is a (K + 1) × (K + 1) rotation matrix and u K+1 is an approximation of the (K + 1)th eigenvector of Σ .

So we have DISPLAYFORM6 which is equivalent to DISPLAYFORM7 This is an eigen-decomposition problem of dimensionality K + 1 p.

It solves the rotation matrix R and allows us to update principal components to U K , given by the first DISPLAYFORM8 If we need to increase the number of principal components, we can just update the system to K = K + 1 and DISPLAYFORM9 .

This incremental PCA algorithm was shown convergent to a stable solution when the sample size increases BID17 .

Our idea to develop the incremental sliced inverse regression (ISIR) is motivated by reformulating SIR problem to a PCA problem.

To this end, we define η = Σ 1 2 β, called the standardized EDR direction, and rewrite the generalized eigen-decomposition problem (3) as an eigen-decomposition problem Σ DISPLAYFORM0 Note that Σ .

To apply the ideas from IPCA to this transformed PCA problem, however, is not as direct as it looks like.

We face two main challenges.

First, when a new observation (x 0 , y 0 ) is received, we need to transform it to an observation for the standardized inverse regression curve.

This is different from simply standardizing the data.

Second, conceptually, we need to update Σ − 1 2 in an online manner in order to standardize the data.

This does not seem feasible.

In the following, we will describe in detail how we address these challenges and make the ISIR implementable.

Suppose we have n observations in hand with well defined sample slice probabilities p h and means ( m h ,ȳ h ) for h = 1, . . .

, H, and the eigenvectors B = [ β 1 , . . . , β K ] of the generalized eigendecomposition problem Γ = λ Σβ.

Then with Λ K = diag(λ 1 , . . . , λ K ), we have DISPLAYFORM1 When we have a new observation (x 0 , y 0 ), we first locate which slice it belongs to according to the distances from y 0 to sample slice mean valuesȳ h of the response variable.

Let us suppose the distance from y 0 toȳ k is the smallest.

So we place the new observation into the slice k and update sample slice probabilities by p h = n p h n+1 for h = k and p k = np k +1n+1 .

Let n k = np k be the number of observations in slice k before receiving the new observation.

For slice mean values we update DISPLAYFORM2 for slice k only.

We can regard z 0 = Σ and normalize it tov = v v when v is not zero.

To update the eigen-decomposition system to where Λ K+1 = diag(Λ K , λ K+1 ) and λ K+1 is the (K + 1)th eigenvalue.

Multiplying both sides by [Ξ,v] , we obtain DISPLAYFORM3 DISPLAYFORM4 Note, however, since Σ cannot be easily updated, we have to avoid using it.

To overcome this challenge, we notice that DISPLAYFORM5 and the well known Sherman-Morisson formula allows us to update the inverse covariance matrix DISPLAYFORM6 If we store Σ −1and update it incrementally, we can approximate the quantities in (9) as follows: DISPLAYFORM7 .

Finally notice that the new EDR space B = Σ Note that we avoided updating the inverse square root of the covariance matrix by using the approx- DISPLAYFORM0 2 .

This approximation can be very accurate when n is large enough because both converge to Σ − 1 2 .

Therefore, we may expect the convergence of ISIR as a corollary of the convergence of IPCA.

However, when n is small, the approximation may be less accurate and result in larger difference between EDR spaces estimated by ISIR and SIR.

So we recommend that ISIR be used with a warm start, that is, using SIR first on a small amount of data before using ISIR.In terms of memory, the primary requirement is the storage of Σ (x 0 −x) and matrix addition and thus has a complexity of O(p 2 ).

Since we need to store M and update it sequentially, it is not efficient to store and update Γ for either memory or computation consideration.

Instead, we use the fact Γ = M P M where P = diag( p 1 , . . .

, p H ) and write DISPLAYFORM1 Notice that DISPLAYFORM2

In BID40 an overlapping technique was introduced to SIR algorithm and shown effectively improving the accuracy of EDR space estimation.

It is motivated by placing each observation in two or more adjacent slices to reduce the deviations of the sample slice means m h from the EDR subspace.

This is equivalent to using each observation two or more times.

In this section, we adopt the overlapping technique to ISIR algorithm above to develop an incremental overlapping sliced inverse regression (IOSIR) algorithm and wish it refines ISIR.To apply the overlapping idea, we use each observation twice.

So when we have n observations, we duplicate them and assume we have N = 2n observations.

When a new observation (x 0 , y 0 ) is received, we duplicate it and assume we receive two identical observations.

Based on the y 0 value we place the first copy into the slice s k ifȳ k is closest to y 0 and run ISIR update as described in Section 4.

Note that ifȳ 1 < y 0 <ȳ H , then y 0 must fall into the interval DISPLAYFORM0 So we place the second copy of the new observation into the slice s k , an adjacent slice to s k , and run ISIR algorithm again.

If y 0 ≤ȳ 1 or y 0 >ȳ H , the second copy will be still placed into s k to guarantee all observations are weighted equally.

As OSIR has superior performance over SIR, we expect IOSIR will perform better than ISIR by a price of double calculation time.

We remark that SIR and ISIR can be used for both regression problems and classification problems.

But since the concept of "adjacent slice" cannot be defined for categorical values (as is the case in classification problems), IOSIR can only be used for regression problems where the response variable is numeric.

In this section, we will verify the effectiveness of ISIR and IOSIR with simulations on artificial and real-world data.

Comparisons will be made between them and SIR.

In the simulations with artificial data, since we know the true model, we measure the performance by the accuracy of the estimated EDR space.

We adopt the trace correlation r(K) = trace(P B P B )/K used in BID13 as the criterion, where P B and P B are the projection operators onto the true EDR space B and the estimated EDR space B, respectively.

We consider the following model from BID21 DISPLAYFORM0 where x = [x 1 , x 2 , . . .

, x p ] follows a multivariate normal distribution, follows standard normal distribution and is independent of x. It has K = 2 effective dimensions with β 1 = (1, 0, 0, . . . , 0) and β 2 = (0, 1, 0 . . .

, 0) .

We conduct the simulation in p = 10 dimensional space and select the number of slices as H = 10.

We give the algorithm a warm start with the initial guess of the EDR space obtained by applying SIR algorithm to a small data set of 40 observations.

Then a total of 400 new observations will be fed to update the EDR space one by one.

SIR, ISIR, and IOSIR are applied when each observation was fed in and we calculate their trace correlation and cumulative computation time.

We repeat this process 100 times.

The mean trace correlations of all three methods are reported in FIG7 (a) and the mean cumulative time is in FIG7 .

We see that ISIR performs quite similar to SIR.

IOSIR slightly outperforms ISIR and SIR.

ISIR is much faster than SIR.

IOSIR gains higher accuracy by sacrificing on computation time.

This verifies the convergence and efficiency of ISIR and IOSIR.

We validate the reliability of ISIR on two real data sets: the Concrete Compressive Strength and Cpusmall (available on https://www.csie.ntu.edu.tw/˜cjlin/libsvmtools/ datasets/regression.html).

There have been many proposed algorithms to increase the prediction accuracy on these data sets BID37 BID38 BID39 Öztaş et al., 2006; BID16 .

We do not intend to outperform those methods.

Our goal is to compare the performance of supervised dimension reduction algorithms and verify the effectiveness of our incremental methods.

The Concrete Compressive Strength data has p = 8 predictors and 1030 samples.

We use H = 10 and K = 3 to run SIR, ISIR, and IOSIR.

We select 50 observations to warm start ISIR and IOSIR algorithms, then 700 observations are fed sequentially.

The left 280 observations are used as test data.

After each new observation is received we estimate the EDR space, project the available training set to the estimated EDR space, build a regression model using the k-nearest neighbor method, and compute the MSE on the test data set.

This process is repeated 100 times and the average MSE was reported in FIG8 (a).

For the Cpusmall data, which has p = 12 predictors and 8192 samples, we do the experiment with H = 10, K = 3, 50 observations to warm start ISIR and IOSIR, 2000 observations for sequential training, and 6142 observations for testing.

The average MSE was plotted in FIG8 (b).

The results indicate both ISIR and IOSIR are as effective as SIR.

We proposed two online learning approaches for supervised dimension reduction, namely, ISIR and IOSIR.

They are motivated by standardizing the data and reformulate the SIR algorithm to a PCA problem.

However, data standardization is only used to motivate the algorithm while not explicitly calculated in the algorithms.

We proposed to use Sherman Morrison formula to online update Σ −1 and some approximated calculations to circumvent explicit data standardization.

This novel idea played a key role in our algorithm design.

Both algorithms are shown effective and efficient.

While IOSIR does not apply to classification problems, it is usually superior over ISIR in regression problems.

We remark that the purpose of ISIR and IOSIR is to keep the dimension reduction accuracy in the situation that a batch learning is not suitable.

This is especially the case for streaming data where information update and system involving is necessary whenever new data becomes available.

When the whole data set is given and one only needs the EDR space from batch learning, ISIR or IOSIR is not necessarily more efficient than SIR because their complexity to run over the whole sample path is O(p 2 N ), comparable to the complexity O(p 3 + p 2 N ) of SIR.There are two open problems worth further investigation.

First, the need to store and use Σ −1 during the updating process is the main bottleneck for ISIR and IOSIR when the dimensionality of the data is ultrahigh.

Second, for SIR and other batch dimension reduction methods, many methods have been proposed to determine the intrinsic dimension K; see e.g. BID21 ; Schott (1994); BID5 ; BID1 ; BID2 ; Nkiet (2008) .

They depend on all p eigenvalues of the generalized eigen-decomposition problem and are impractical for incremental learning.

We do not have obvious solutions to these problems at this moment and would like to leave them for future research.

<|TLDR|>

@highlight

We proposed two new approaches,  the incremental sliced inverse regression and incremental overlapping sliced inverse regression, to implement supervised dimension reduction in an online learning manner.

@highlight

Studies sufficient dimension reduction problem and proposes an incremental sliced inverse regression algorithm.

@highlight

This paper proposes an online learning algorithm for supervised dimension reduction, called incremental sliced inverse regression
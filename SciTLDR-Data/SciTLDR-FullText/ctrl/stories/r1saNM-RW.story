Support Vector Machines (SVMs) are one of the most popular algorithms for classification and regression analysis.

Despite their popularity, even efficient implementations have proven to be computationally expensive to train at a large-scale, especially in streaming settings.

In this paper, we propose a novel coreset construction algorithm for efficiently generating compact representations of massive data sets to speed up SVM training.

A coreset is a weighted subset of the original data points such that SVMs trained on the coreset are provably competitive with those trained on the original (massive) data set.

We provide both lower and upper bounds on the number of samples required to obtain accurate approximations to the SVM problem as a function of the complexity of the input data.

Our analysis also establishes sufficient conditions on the existence of sufficiently compact and representative coresets for the SVM problem.

We empirically evaluate the practical effectiveness of our algorithm against synthetic and real-world data sets.

Popular machine learning algorithms are computationally expensive, or worse yet, intractable to train on Big Data.

The notion of using coresets BID6 BID3 BID2 , small weighted subsets of the input points that provably approximate the original data set, has shown promise in accelerating machine learning algorithms such as kmeans clustering BID6 , mixture model training , and logistic regression BID12 .Coreset constructions were originally introduced in the context of computational geometry BID1 and subsequently generalized for applications to other problems BID14 BID6 .

Coresets provide a compact representation of the structure of static and streaming data, with provable approximation guarantees with respect to specific algorithms.

For instance, a data set consisting of K clusters would yield a coreset of size K, with each cluster represented by one coreset point.

Even if the data has no structure (e.g., uniformly distributed), coresets will correctly down sample the data to within prescribed error bounds.

For domains where the data has structure, the coreset representation has the potential to greatly and effectively reduce the time required to manually label data for training and the computation time for training, while at the same time providing a mechanism of supporting machine learning systems for applications with streaming data.

Coresets are constructed by approximating the relative importance of each data point in the original data set to define a sampling distribution and sampling sufficiently many points in accordance with this distribution.

This construction scheme suggests that beyond providing a means of conducting provably fast and accurate inference, coresets also serve as efficient representations of the full data set and may be used to automate laborious representation tasks, such as automatically generating semantic video representations or detecting outliers in data BID17 .The representative power and provable guarantees provided by coresets also motivate their use in training of one of the most popular algorithms for classification and regression analysis: Support Vector Machines (SVMs).

Despite their popularity, SVMs are computationally expensive to train on massive data sets, which has proven to be computationally problematic with the rising availability of Big Data.

In this paper, we present a novel coreset construction algorithm for efficient, large-scale Support Vector Machine training.1.

A practical coreset construction algorithm for accelerating SVM training based on an efficient importance evaluation scheme for approximating the importance of each point.2.

An analysis proving lower bounds on the number of samples required by any coreset construction algorithm to approximately represent the data.3.

An analysis proving the efficiency and theoretical guarantees of our algorithm and characterizing the family of data sets for which applications of coresets are most suited.4.

Evaluations against synthetic and real-world data sets that demonstrate the practical effectiveness of our algorithm for large-scale SVM training.

Training a canonical Support Vector Machine (SVM) requires O(n 3 ) time and O(n 2 ) space where n is the number of training points BID23 .

Work by BID23 introduced Core Vector Machines (CVMs) that reformulated the SVM problem as the Minimum Enclosing Ball (MEB) problem and used existing coreset methods for MEB to compress the data.

The authors proposed a method that generates a (1 + ε) 2 approximation to the two-class L2-SVM in O(n/ε 2 + 1/ε 4 ) time, when certain assumptions about the kernel used are satisfied.

However, CVM's accuracy and convergence properties have been noted to be at times inferior to the performance of existing SVM implementations BID16 .

Similar geometric approaches including extensions to the MEB formulation, those based on convex hulls, and extreme points, among others, were also investigated by BID21 Since the SVM problem is inherently a quadratic optimization problem, prior work has investigated approximations to the quadratic programming problem using the Frank-Wolfe algorithm or Gilbert's algorithm BID4 .

Another line of research has been in reducing the problem of polytope distance to solve the SVM problem BID8 ).

The authors establish lower and upper bounds for the polytope distance problem and use Gilbert's algorithm to train an SVM in linear time.

A variety of prior approaches were based on randomized algorithms with the property that they generated accurate approximations with high probability.

Most notable are the works of BID5 BID11 .

BID11 used a primal-dual approach combined with Stochastic Gradient Descent (SGD) in order to train linear SVMs in sub-linear time.

They proposed the SVM-SIMBA approach and proved that it generates an ε-approximate solution with probability at least 1/2 to the SVM problem that uses hinge loss as the objective function.

The key idea in their method is to access single features of the training vectors rather than the entire vectors themselves.

Their method is nondeterministic and returns the correct ε-approximation with probability greater than a constant probability, similar to the probabilistic guarantees of coresets.

BID5 present sub-linear-time (in the size of the input) approximation algorithms for some optimization problems such as training linear classifiers (e.g., perceptron) and finding MEB.

They introduce a technique that is originally applied to the perceptron algorithm, but extend it to the related problems of MEB and SVM in the hard margin or L2-SVM formulations.

BID22 introduce Pegasos, a stochastic sub-gradient algorithm for approximately solving the SVM optimization problem, that runs in O(dnC/ε) time for a linear kernel, where C is the SVM regularization parameter and d is the dimensionality of the input data points.

These works offer probabilistic guarantees, similar to those provided by coresets, and have been noted to perform well empirically; however, unlike coresets, SGD-based approaches cannot be trivially extended to streaming settings since each new arriving data point in the stream results in a change of the gradient.

BID13 presents an alternative approach to training SVMs in linear time based on the cutting plane method that hinges on an alternative formulation of the SVM optimization problem.

He shows that the Cutting-Plane algorithm can be leveraged to train SVMs in O(sn) time for classification and O(sn log n) time for ordinal regression where s is the average number of non-zero features.

BID10 constructs coresets to approximate the maximum margin separation, i.e., a hyperplane that separates all of the input data with margin larger than (1 − ε)ρ * , where ρ * is the best achievable margin.

We assume that we are given a set of weighted training points P = {(x i , y i )} n i=1 with the corresponding weight function u : P → R ≥0 , such that for every i ∈ [n], x i ∈ R d , y i ∈ {−1, 1}, and u(p i ) corresponds to the weight of point p i .

For simplicity, we assume that the bias term is embedded into the feature space by definingx i = (x i , 1) for each point andw = (w, 1) for each query.

Thus, we henceforth assume that we are dealing with d + 1 dimensional points and refer tow andx i as just w and x i respectively.

Under this setting, the hinge loss of a point p i = (x i , y i ) with respect to the separating hyperplane w is defined as h( DISPLAYFORM0 For any subset of points, P ⊆ P, define H(P , w) = p∈P u(p)h(p, w) as the sum of the hinge losses and u(P ) = p∈P u(p) as the sum of the weights of points in set P .

To clearly depict the contribution of each point to the objective value of the SVM problem, we present the SVM objective function, f (P, w), as the sum of per-point objective function evaluations, which we formally define below.

DISPLAYFORM1 is the corresponding hinge loss, and C ∈ [0, 1] is the regularization parameter.

Definition 2 (Soft-margin SVM Problem).

Given a set of d + 1 dimensional weighted points P with weight function u : P → R ≥0 the primal of the SVM problem is expressed by the following quadratic program DISPLAYFORM2 where f is the evaluation of the weighted point set P with weight function u : P → R ≥0 , DISPLAYFORM3 When the set of points P and the corresponding weight function u are clear from context, we will henceforth denote f ((P, u), w) by f (P, w) for notational convenience.

Coresets can be seen as a compact representation of the full data set that approximate the SVM cost function (2) uniformly over all queries w ∈ Q. Thus, rather than introducing an entirely new algorithm for solving the SVM problem, our approach is to reduce the runtime of standard SVM algorithms by compressing the size of the input points from n to a compact set whose size is sublinear (ideally, polylogarithmic) in n.

Definition 3 (ε-coreset).

Let ε ∈ (0, 1/2) and P ⊂ R d+1 × {−1, 1} be a set of n weighted points with weight function u : P → R ≥0 .

The weighted subset (S, v), where S ⊂ P with corresponding weight function v : S → R ≥0 is an ε-coreset if for any query w ∈ Q, (S, v) satisfies the coresetproperty DISPLAYFORM0 Our overarching goal is to efficiently construct an ε-coreset, (S, v), such that the size of S is sufficient small in comparison to the original number of points n.

Our coreset construction scheme is based on the unified framework of BID14 BID6 and is shown as Alg.

1.

The crux of our algorithm lies in generating the importance sampling distribution via efficiently computable upper bounds (proved in Sec. 5) on the importance of each point (Lines 1-6).

Sufficiently many points are then sampled from this distribution and each point is given a weight that is inversely proportional to its sample probability (Lines 7-9).

The number of points required to generate an ε-coreset with probability at least 1 − δ is a function of the desired accuracy ε, failure probability δ, and complexity of the data set (t from Theorem 9).

Under mild assumptions on the problem at hand (see Sec. 5.2), the required sample size is polylogarithmic in n.

Intuitively, our algorithm can be seen as an importance sampling procedure that first generates a judicious sampling distribution based on the structure of the input points and samples sufficiently many points from the original data set.

The resulting weighted set of points, (S, v), serves as an unbiased estimator for f (P, w) for any query w ∈ Q, i.e., E[f ((S, v), w)] = f (P, w).

Although sampling points uniformly with appropriate weights can also generate such an unbiased estimator, it turns out that the variance of this estimation is minimized if the points are sampled according to the distribution defined by the ratio between each point's sensitivity and the sum of sensitivities, i.e., γ(p i )/t on Line 9 BID2 .

Coresets are intended to provide efficient and provable approximations to the optimal SVM solution, however, the very first line of our algorithm entails computing the optimal solution to the SVM problem.

This seemingly eerie phenomenon is explained by the merge-and-reduce technique BID9 ) that ensures that our coreset algorithm is only run against small partitions of the original data set BID9 BID3 .

The merge-and-reduce approach leverages the fact that coresets are composable and reduces the coreset construction problem for a (large) set of n points into the problem of computing coresets for n 2|S| points, where 2|S| is the minimum size of input set that can be reduced to half using Alg.

1 BID3 .

Assuming that the sufficient conditions for obtaining polylogarithmic size coresets implied by Theorem 9 hold, the overall time required for coreset construction is nearly linear in n, O ε,δ (d 3 n) 1 .

This follows from the fact that 2|S| = O δ,ε (d) by Theorem 9, that the Interior Point Method runs in time O(|S| 3 ) =Õ δ,ε (d 3 ) for an input set of size 2|S|, and that the merge-and-reduce tree has height at most log n , meaning that an accuracy parameter of ε = ε/ log n has to be used in the intermediate coreset constructions to account for the compounded error over all levels of the tree BID3 .

We briefly remark on a straightforward extension that can be made to our algorithm to accelerate performance and applicability.

In particular, the computation of the optimal solution to the SVM problem in line 1 can be replaced by an efficient gradient-based method, such as Pegasos (ShalevShwartz et al., 2011) , to compute an approximately ξ optimal solution in O (dnC/ξ) time, which is particularly suited to scenarios with C small.

We give this result as Lemma 11, an extension of Lemma 7.

We also note that based on our analytical results (Lemmas 7 and 11), any SVM solver, either exact or approximate, can be used in Line 1 as a replacement for the Interior Point Method.

In this section, we prove upper and lower bounds on the sensitivity of a point in terms of the complexity of the given data set.

Our main result is Theorem 9, which establishes sufficient conditions 1 O ε,δ notation suppresses ε, δ and polylog(n) factors.

Algorithm 1: CORESET(P, u, ε, δ)

A set of training points P ⊆ R d+1 × {−1, 1} containing n points, a weight function u : P → R ≥0 , an error parameter ε ∈ (0, 1), and failure probability δ ∈ (0, 1).

An ε-coreset (S, v) with probability at least 1 − δ.// Compute the optimal solution using an Interior Point Method.

1 w * ← InteriorPointMethod(P, u, C) DISPLAYFORM0 for each y i ∈ {−1, 1}; // Compute an upper bound for the sensitivity of each point according to Eqn.(5).

DISPLAYFORM1 for the existence of small coresets depending on the properties of the data.

Our theoretical results also highlight the influence of the regularization parameter, C, in the size of the coreset.

Definition 4 (Sensitivity BID3 ).

The sensitivity of an arbitrary point p ∈ P, p = (x, y) is defined as DISPLAYFORM2 where u : P → R ≥0 is the weight function as before.

We first prove the existence of a hard point set for which the sum of sensitivities is approximately Ω(nC), ignoring d factors, which suggests that if the regularization parameter is too large, then the required number of samples for property (3) to hold is Ω(n).

Lemma 5.

There exists a set of n points P such that the sensitivity of each point p i is bounded below by Ω The same hard point set from Lemma 5 can be used to also prove a bound that is nearly exponential in the dimension, d. Corollary 6.

There exists a set of n points P such that the total sensitivity is bounded below by DISPLAYFORM0 We next prove upper bounds on the sensitivity of each data point with respect to the complexity of the input data.

Despite the non-existence results established above, our upper bounds shed light into the class of data sets for which coresets of sufficiently small size exist, and thus have potential to significantly speed up SVM training.

For any arbitrary point p = (x i , y i ) ∈ P, let P yi ⊂ P denote the set of points of label y i , let P c yi = P \ P yi be its complement, and let w * denote the optimal solution to the SVM problem (2).

We assume that the points are normalized to have a Euclidean norm of at most one, i.e., ∀(x, y) ∈ P x 1:d 2 ≤ 1, where x 1:d refers to original input point, without the bias embedding.

Lemma 7.

The sensitivity of any point p i ∈ P is bounded above by DISPLAYFORM0 where p ∆ =p yi − p i and K yi = u(P c yi )/ (2u(P) · u(P yi )).Let P + = P 1 ⊂ P and P − = P \ P 1 denote the set of points with positive and negative labels respectively.

Letp + andp − denote the weighted mean of the positive and labeled points respectively, and for any p i ∈ P + let p DISPLAYFORM1 The sum of sensitivities over all points P is bounded by DISPLAYFORM2 where f (P, w * ) is the optimal value of the SVM problem, and Var(P + ) and Var(P − ) denote the total deviation of positive and negative labeled points from their corresponding label-specific mean DISPLAYFORM3 Theorem 9.

Given any ε ∈ (0, 1/2), δ ∈ (0, 1) and a weighted data set P with corresponding weight function u, with probability greater than 1 − δ, Algorithm 1 generates an ε-coreset, i.e., a weighted set (S, v), of size S ∈ Ω t ε 2 d log t + log(1/δ) in O(n 3 ) time, where t is the upper bound on the sum of sensitivities from Lemma 8, DISPLAYFORM4 For any subset T ⊆ P, let w * T denote the optimal separating hyperplane with respect to the set of points in T .

The following corollary immediately follows from Theorem 9 and implies that training an SVM on an ε-coreset, (S, v) , to obtain w * S yields a solution that is provably competitive with the optimal solution on the full data-set, w * P = w * .Corollary 10.

Given any ε ∈ (0, 1/2), δ ∈ (0, 1) and a weighted data set (P, u), the weighted set of points (S, v) generated by Alg.

1 satisfies f ((P, u), w * S ) ≤ (1 + 4ε)f ((P, u), w * P ), with probability greater than 1 − δ.

Sufficient Conditions Theorem 9 immediately implies that, for reasonable ε and δ, coresets of polylogarithmic (in n) size can be obtained if d = O(polylog(n)), which is usually the case in our target applications, and if DISPLAYFORM5 For example, a value of C ≤ log n n for the regularization parameter C satisfies the sufficient condition for all data sets with normalized points.

Interpretation of Bounds Our approximation of the sensitivity of a point p i ∈ P, i.e., its relative importance, is a function of the following highly intuitive variables.1.

Relative weight with respect to the weights of points of the same label (u(p i )/u(P yi )): the sensitivity increases as this ratio increases.

2.

Distance to the label-specific mean point p yi − p i 2 : points that are considered outliers with respect to the label-specific cluster are assigned higher importance.

3.

Distance to the optimal hyperplane ( w * , p ∆ ): importance increases as distance of the difference vector p ∆ =p yi − p i to the optimal hyperplane increases.

Note that the sum of sensitivities, which dictates how many samples are necessary to obtain an ε-coreset with probability at least 1 − δ and in a sense measures the difficulty of the problem, increases monotonically with the sum of distances of the points from their label-specific means.

We conclude our analysis with an extension of Lemma 7 to the case where only an approximately optimal solution to the SVM problem is available.

Lemma 11.

Consider the case where only a ξ-approximate solutionŵ is available such that f (P,ŵ) ≤ f (P, w * ) + ξ, for ξ ∈ (0, f (P, w * )/2).

Then, the sensitivity of any arbitrary point p i ∈ P is bounded above by DISPLAYFORM0

We evaluate the performance of our coreset construction algorithm against synthetic and real-world, publicly available data sets BID15 .

We compare the effectiveness of our method to uniform subsampling on a wide variety of data sets and also to Pegasos, one of the most popular Stochastic-Gradient Descent based algorithm for SVM training BID22 .

For each data set of size N , we selected a set of M = 10 subsample sizes S 1 , . . .

, S M ⊂ [N ] and ran each coreset construction algorithm to construct and evaluate the accuracy subsamples sizes S 1 , . . .

, S M .

The results were averaged across 100 trials for each subsample size.

Our results of relative error and sampling variance are shown as FIG2 .

The computation time required for each sample size and approach can be found in the Appendix (Fig. 4) .

Our experiments were implemented in Python and performed on a 3.2GHz i7-6900K (8 cores total) machine with 16GB RAM.

We considered the following data sets for evaluation.• Pathological -1, 000 points in two dimensional space describing two clusters distant from each other of different labels, as well as two points of different labels which are close to each other.

We note that uniform sampling performs particularly poorly against this data set due to the presence of outliers.• Synthetic & Synthetic100K-The Synthetic and Synthetic100K are datasets with 6, 000, 100, 000 points, each consisting of 3 and 4 dimensions respectively.

The datasets describe two blocks of mirrored nested rings of points, each of different labels such that Gaussian noise has been added to them.• HTRU 2 -17, 898 radio emissions of Pulsar (rare type of Neutron star) each consisting of 9 features.• CreditCard 3 -30, 000 client entries each consisting of 24 features that include education, age, and gender among other factors.• Skin 4 -245, 057 random samples of B,G,R from face images consisting of 4 dimensions.

Evaluation We computed the relative error of the sampling-based algorithms with respect to the cost of the optimal solution to the SVM problem, f (P, w * P ) and the approximate cost generated by the subsample, f ((S, v), w * S ).

We have also evaluated against Pegasos, running Pegasos the amount of time needed to construct the coreset and comparing the resulted error, applying 128 repetitions as presented at FIG3 .

Furthermore, we have run our coreset constructing under streaming setting, where subsamples are used as leaf size and half of the leaf's size is then used to set the subsample for our sampling approach.

In addition, we also compared our coreset construction's related error to CVM's related error with respect to the cost of the optimal solution to the SVM problem, as function of subsample sizes.

Finally, we have evaluated the variance of the estimators for the sampling-based approaches and observed that the variances of the estimates generated by our coreset were lower than those of uniform subsampling.

We presented an efficient coreset construction algorithm for generating compact representations of the input data points that provide provably accurate inference.

We presented both lower and upper bounds on the number of samples required to obtain accurate approximations to the SVM problem as a function of input data complexity and established sufficient conditions for the existence of compact representations.

Our experimental results demonstrate the effectiveness of our approach in speeding up SVM training when compared to uniform sub-samplingThe method presented in this paper is also applicable to streaming settings, using the merge-andreduce technique from coresets literature BID3 .We conjecture that our coreset construction method can be extended to significantly speed up SVM training for nonlinear kernels as well as other popular machine learning algorithms, such as deep learning.

8 APPENDIX Figure 3 : The estimator variance of query evaluations.

We note that due to the use of a judicious sampling distribution based on the points' sensitivities, the variance of our coreset estimator is lower than that of uniform sampling for all data sets.

Proof.

Following BID25 we define the set of n points P ⊆ R d+1 × {−1, 1}, each point p ∈ P with weight u(p) = 1, such that for each i ∈ [n], among the first d entries of p i , exactly d/2 entries are equivalent to γ: DISPLAYFORM0 where R is the normalization factor (typically R = 1), the remaining d/2 entries among the first d are set to 0, and p i(d+1) = y i as before.

For each i ∈ [n], define the set of non-zero entries of p i as the set DISPLAYFORM1 Now, for bounding the sensitivity of point p i , consider the normal to the margin w i with entries defined as DISPLAYFORM2 Note that for R = 1, contributed by other points j ∈ [n], j = i note that B i \ B j = ∅, thus: DISPLAYFORM3 DISPLAYFORM4 which implies that h(p j , w i ) = 0.

Thus, it follows that H( DISPLAYFORM5 Putting it all together, we have for the sensitivity of any arbitrary i ∈ [n]: Moreover, we have for the sum of sensitivities that DISPLAYFORM6 DISPLAYFORM7 Proof.

Consider the set of points P from the proof of Lemma 5 and note that DISPLAYFORM8

Proof.

Consider any arbitrary point p i ∈ P and let p = x i y i for brevity when the point p i = (x i , y i ) is clear from the context.

We proceed to bound s(p i )/u(p i ) by first leveraging the Lipschitz 8.4 PROOF OF LEMMA 8Proof.

Let P + = P 1 ⊂ P and P − = P \P 1 denote the set of points with positive and negative labels respectively and let K + and K − denote the corresponding constants defined by (8).

Letp + andp − denote the weighted mean of the positive and labeled points respectively, and for any p i ∈ P + let p + ∆i =p + − p i and p − ∆i =p − − p i .

Since the sensitivity can be decomposed into sum over the two disjoint sets, i.e., S(P) = p∈P s(p) = p∈P1 s(p) + p∈P− s(p) = S(P 1 ) + S(P − ), we consider first bounding S(P 1 ).

Invoking Lemma 7 yields S(P 1 ) ≤ 1 + .Using the same argument as above, an analogous bound holds for S(P − ), thus we have DISPLAYFORM0 f (P, w * ) = t.

Proof.

By Lemma 8 and Theorem 5.5 of BID3 we have that the coreset constructed by our algorithm is an ε-coreset with probability at least 1 − δ if |S| ≥ Ω t ε 2 d log t + log(1/δ) ,where we used the fact that the VC dimension of a separating hyperplane in the case of a linear kernel is bounded dim(F) ≤ d + 1 = O(d) BID24 .

Moreover, note that the computation time of our algorithm is dominated by computing the optimal solution of the SVM problem using interior-point Method which takes O(d 3 L) = O(n 3 ) time BID20 , where L is the bit length of the input data.

Proof.

By theorem 9, (S, v) is an ε-coreset for (P, u) with probability at least 1 − δ, thus we have f ((P, u), w * P ) ≤ f ((P, u), w * S ) ≤ f ((S, v), w * S ) 1 − ε ≤ (1 + ε)f ((P, u), w * P ) 1 − ε ≤ (1 + 4ε)f ((P, u), w

<|TLDR|>

@highlight

We present an algorithm for speeding up SVM training on massive data sets by constructing compact representations that provide efficient and provably approximate inference.

@highlight

Studies the approach of coreset for SVM and aims at sampling a small set of weighted points such that the loss function over the points provably approximates that over the whole dataset

@highlight

The paper suggests an importance sampling based Coreset construction to represent large training data for SVMs
In this paper, we propose a nonlinear unsupervised metric learning framework to boost of the performance of clustering algorithms.

Under our framework, nonlinear distance metric learning and manifold embedding are integrated and conducted simultaneously to increase the natural separations among data samples.

The metric learning component is implemented through feature space transformations, regulated by a nonlinear deformable model called Coherent Point Drifting (CPD).

Driven by CPD, data points can get to a higher level of linear separability, which is subsequently picked up by the manifold embedding component to generate well-separable sample projections for clustering.

Experimental results on synthetic and benchmark datasets show the effectiveness of our proposed approach over the state-of-the-art solutions in unsupervised metric learning.

Cluster analysis has broad applications in various disciplines.

Grouping data samples into categories with similar features is an efficient way to summarize the data for further processing.

In measuring the similarities among data samples, the Euclidean distance is the most common choice in clustering algorithms.

Under Euclidean distance, feature components are assigned with the same weight, which essentially assumes all features are equally important across the entire data space.

In practice, such setup is often not optimal.

Learning a customized metric function from the data samples can usually boost the performance of various machine learning algorithms BID1 .

While metric learning has been extensively researched under supervised BID19 BID18 BID17 BID14 and semi-supervised settings BID15 BID3 BID23 BID13 , unsupervised metric learning (UML) remains a challenge, in part due to the absence of ground-truth label information to define a learning optimality.

In this paper, we focus on the problem of UML for clustering.

As the goal of clustering is to capture the natural separations among data samples, one common practice in the existing UML solutions is to increase the data separability and make the separations more identifiable for the ensuing clustering algorithm.

Such separability gain can be achieved by projecting data samples onto a carefully chosen low-dimensional manifold, where geometric relationships, such as the pairwise distances, are preserved.

The projections can be carried out linearly, as through the Principle Component Analysis, or nonlinearly, as via manifold learning solutions.

Under the dimension-reduced space, clustering algorithms, such as K-means, can then be applied.

Recent years have seen the developments of UML solutions exploring different setups for the lowdimensional manifolds.

FME ) relies on the learning of an optimum linear regression function to specify the target low-dimensional space.

BID0 model local sample densities of the data to estimate a new metric space, and use the learned metric as the basis to construct graphs for manifold learning.

Application-specific manifolds, such as Grassmann space BID6 and Wasserstein geometry BID16 , have also been studied.

When utilized as a separate preprocessing step, dimensionality reduction UML solutions are commonly designed without considering the ensuing clustering algorithm and therefore cannot be fine-tuned accordingly.

AML takes a different approach, performing clustering and distance metric learning simultaneously.

The joint learning under AML is formulated as a trace maximization problem, and numerically solved through an EM-like iterative procedure, where each iteration consists of a data projection step, followed by a clustering step via kernel K-means.

The projection is parameterized by an orthogonal, dimension-reducing matrix.

A kernelized extension of AML was proposed in BID2 .

As the projection models are built on linear transformations, their capabilities to deal with complex nonlinear structures are limited.

UML solutions performing under the original input space have also been proposed.

SSO BID7 learns a global similarity metric through a diffusion procedure that propagates smooth metrics through the data space.

CPCM BID4 relies on the ratio of within cluster variance over the total data variance to obtain a linear transformation, aiming to improved data separability.

As the original spaces are usually high-dimensional, UML solutions in this category tend to suffer from the local minima problem.

In light of the aforementioned limitations and drawbacks, we propose a new nonlinear UML framework in this paper.

Our solution integrates nonlinear feature transformation and manifold embedding together to improve the data separability for K-means clustering.

Our model can be regarded as a fully nonlinear generalization of AML, in which the transformation model is upgraded to a geometric model called Coherent Point Drifting (CPD) BID8 .

Data points are driven by CPD to reach a higher level of linear separability, which will be subsequently picked up by the manifold embedding component to generate well-separable sample projections.

At the end, K-means is applied on the transformed, dimension-reduced embeddings to produce label predictions.

The choice of CPD is with the consideration of its capability of generating high-order yet smooth transformations.

The main contributions of this paper include the following.• Our proposed fully nonlinear UML solution enhances data separability through the combination of CPD-driven deformation and spectral embeddings.• To the best of our knowledge, this is the first work that utilizes dense, spatial varying deformations in unsupervised metric learning.• The CPD optimization has a closed-form solution, therefore can be efficiently computed.• Our model outperforms state-of-the-art UML methods on six benchmark databases, indicating promising performance in many real-world applications.

The rest of this paper is organized as follows.

Section 2 describes our proposed method in detail.

It includes the description of CPD model, formulation of our CPD based UML, optimization strategy and the approach to kernelize our model.

Experimental results are presented in Section 3 to validate our solutions with both synthetic and real-world datasets.

Section 4 concludes this paper.

Many machine learning algorithms have certain assumption regarding the distribution of the data to be processed.

K-means always produces clustering boundaries of hyperplanes, working best for the data set made of linearly separable groups.

For data sets that are not linearly separable, even they are otherwise well-separable, K-means will fail to deliver.

Nonlinearly displacing the data samples to make them linearly separable would provide a remedy, and learning such a transformation is the goal of our design.

The application of such a smooth nonlinear transformation throughout feature space (either input space or kernel space) would change pairwise distances among samples, which is equivalent to assigning spatially varying metrics in different areas of the data space.

In our framework, the CPD model is chosen to perform the transformation.

Originally designed for regulating points matching, CPD moves the points U towards the target V by estimating an optimal continuous velocity function v(x) : DISPLAYFORM0 where n is the number of samples in the dataset, and d is the data dimension.

L represents a linear differentiation operator, and λ is the regularization parameter.

The regularization term in CPD is a Gaussian low-pass filter.

The optimal solution v(x) to matching U and V can be written in the matrix format as BID9 : DISPLAYFORM1 where Ψ (size d × n) is the weight matrix for the Gaussian kernel functions, g( DISPLAYFORM2 .

σ is the width of the Gaussian filter, which controls the smoothness level of the deformation field.

K-means clustering aims to partition the samples into K groups S = {S 1 , S 2 , ..., S K }, through the minimization of the following objective function: DISPLAYFORM0 S c is the set of data samples in the c-th cluster.

n c is the number of data instances in cluster S c , and µ c is the mean of S c .Allowing samples to be moved, we intend to learn a spatial transformation to improve the performance of K-means clustering by making groups more linearly separable, as well as by harnessing the updated distance measure under the transformed feature space.

Let x i be the initial location of an instance.

Through the motion in Eqn.

FORMULA1 , x i will be moved to a new position x DISPLAYFORM1 With Eqn.

FORMULA5 , Eqn.

(2) can be reformulated as: DISPLAYFORM2 Now DISPLAYFORM3 c is the mean vector of the instances in cluster S 1 c .

Our proposed CPD based unsupervised metric learning (CPD-UML) is designed to learn a spatial transformation Ψ and a clustering S 1 at the same time.

Eqn.

(4) can be reformulated into a matrix format through the following steps.

First, put the input dataset into a d-by-n data matrix.

Second, define a Gaussian kernel function matrix for the CPD deformation as: DISPLAYFORM4 The size of G is n-by-n.

Third, let p be a vector of dimension n c -by-1 with all elements equal to one, then the mean of the data instances within a cluster S 1 c can be written as µ BID21 .

With these three formulations, and let E be a permutation matrix, Eqn.

(4) can be rewritten as: DISPLAYFORM5 DISPLAYFORM6 where X 1 is the transformed data matrix.

Since ||A|| 2 F = trace(A T A), Eqn.

FORMULA10 can be written in the form of the trace operation: DISPLAYFORM7 As trace(AB) = trace(BA), and p T p = n c , the J in Eqn.

FORMULA11 can be further reformulated as: DISPLAYFORM8 Similar to BID21 ), we define a n-by-k orthonormal matrix Y as the cluster indicator matrix: DISPLAYFORM9 With X 1 = X + ΨG(X, X) and the cluster indicator matrix in Eqn.

FORMULA13 , Eqn.

FORMULA12 can be written into the following: DISPLAYFORM10 To reduce overfitting, we add the squared Frobenius norm λ||Ψ|| 2 F = λtrace(Ψ T Ψ), to penalize any non-smoothness in the estimated transformations.

λ is a regularization parameter.

Finally, our nonlinear CPD-UML solution is formulated as a trace minimization problem, parameterized by Y and Ψ: DISPLAYFORM11

To search for the optimal solutions of Y and Ψ, an EM-like iterative minimization framework is adopted to update Y and Ψ alternatingly.

The transformation matrix Ψ is initialized with all 0 elements, and the cluster indicator is initialized with a K-means clustering result of the input data samples.

Optimization for Y With Ψ fixed, Eqn.

FORMULA1 reduces to a trace maximization problem: DISPLAYFORM0 Since Y is an orthonormal matrix: Y T Y = I K , the spectral relaxation technique BID21 ) can be adopted to compute the optimal Y .

The solution is based on Ky Fan matrix inequalities below:Theorem. (Ky Fan) If A be a symmetric matrix with eigenvalues {λ 1 ≥ λ 2 ≥ ...

≥ λ n }.

Let the corresponding eigenvectors be DISPLAYFORM1 where the optimal Y * is given by DISPLAYFORM2 This spectral relaxation solution can be regarded as a manifold learning method that projects data samples from the original d-dimensional space to a new K-dimensional space.

In our case, the A matrix in Ky Fan Theorem takes the form of X T X. In implementation, we first compute the K largest eigenvectors of X T X, and then apply the traditional K-means method, under the induced K-dimensional space, to compute the cluster assignments.

Optimization for Ψ With the Y generated from Eqn. (12), Eqn. (11) becomes a trace minimization problem w.r.t.

Ψ: DISPLAYFORM3 Through a careful investigation of the gradient and Hessian matrix of Eqn.

FORMULA1 , we found the J could be proved to a smooth convex function, with its Hessian w.r.t.

Ψ being positive definite (PD) everywhere.

Therefore, the only stationary point of J, where the gradient is evaluated to 0, locates the global minimum, and provides the optimal Ψ * .

The convexity proof is given as follows.

Convexity proof of J w.r.t.

Ψ: Firstly, we update J in Eqn. (13), through several straightforward derivation steps (the details are given in Appendix A), to an equivalent form: DISPLAYFORM4 The gradient of J w.r.t.

Ψ can then be computed as: DISPLAYFORM5 To facilitate the convexity proof, we rewrite this gradient equation as: DISPLAYFORM6 N is a matrix of size d × n. M is a symmetric matrix of size n × n, which can be proved positive definite, based on the theorem in (Horn & Johnson, 2012):Theorem.

"Suppose that A ∈ M m,n and B ∈ M n,m with m ≤ n. Then BA has the same eigenvalues as AB, counting multiplicity, together with an additional n − m eigenvalues equal to 0."We know Y T * Y = I K , whose eigenvalues are all 1s.

Then, according to this Theorem, the eigenvalues of Y Y T are 1s (multiplicity is K), and 0 (multiplicity is n − K).

In the matrix M of Eqn.

FORMULA1 DISPLAYFORM7 T is a positive semidefinite matrix as it is symmetric and its eigenvalues are either 0 or 1.

G is also positive definite because it is a kernel (Gram) matrix with the Gaussian kernel.

With G being symmetric PD and λ setting to be a positive number in our algorithm, the matrix M is guaranteed to be a PD matrix.

Expanding the gradient formulated in Eqn. (16) to individual elements of Ψ, it can be further written as: DISPLAYFORM8 With Eqn.

FORMULA1 , Eqn.

FORMULA1 can be resized into a vector of size d × n.

Then, the Hessian matrix of J w.r.t.

Ψ can be calculated as below: DISPLAYFORM9 DISPLAYFORM10 It is clear that H is a symmetric matrix with size (d * n) × (d * n).

The diagonal of H is composed by d repeating M matrices.

Let z be any non-zero column vector with size (d * n) × 1.

To prove H is a PD matrix, we want to show that z T H z is always positive.

To this end, we rewrite z as [ z 1 , z 2 , ..., z d ], where z i is the sub-column of z with size n × 1.

Then z T H z can be computed as: DISPLAYFORM11 As M has been proved to be a PD matrix, each item in Eqn.

FORMULA1 is positive.

Therefore, the summation z T H z is also positive.

Since z is an arbitrary non-zero column vector, this shows H is PD.

With the Hessian matrix H being PD everywhere, the objective function J is convex w.r.t.

Ψ.

As a result, the stationary point of J makes the unique global minimum solution Ψ * .

Let Eqn. (15) equal to 0, we get DISPLAYFORM12 The matrix M on the left is proved PD, thus invertible.

The optimal solution of Ψ is given as: DISPLAYFORM13

Based on the description above, our proposed CPD-UML algorithm can be summarized as the pseudo-code below:Algorithm

So far, we developed and applied our proposed CPD-UML under input feature spaces.

However, it can be further kernelized to improve the clustering performance for more complicated data.

A kernel principal component analysis (KPCA) based framework ) is utilized in our work.

After the input data instances are projected into kernel spaces introduced by KPCA, CPD-UML can be applied under the kernel spaces to learn both deformation field and clustering result, in the same manner as it is conducted under the original input spaces.

We performed experiments on a synthetic dataset and six benchmark datasets.

Comparisons are made with state-of-the-art unsupervised metric learning solutions.

The two-moon synthetic dataset 1 was tested in the first set of experiments.

It consists of two classes with 100 examples in each class. (see FIG1 ).

All the samples were treated as unlabeled samples in the experiments.

Both linear and kernel versions of our CPD-UML were tested.

Linear version CPD-UML In this experiment, our CPD-UML was applied in deforming the data samples to achieve better separability under the input space.

The effectiveness of our approach is demonstrated by comparing with the base algorithm K-means.

The clustering results of K-means and CPD-UML are shown in FIG1 (a) and 1 (b) respectively.

The sample labels are distinguished using blue and red colors.

The clustering results are shown using the decision boundary.

It is obvious that K-means cannot cluster the two-moon data well due to the data's non-separability under the input space.

Our CPD-UML, on the contrary, achieves a 99% clustering accuracy by making the data samples linearly separable via space transformations.

The deformation field of FIG1 in the input space is shown in FIG1 and (d).

It is evident that our nonlinear metric learning model can deform feature spaces in a sophisticated yet smooth way to improve the data separability.

Kernel version CPD-UML In this set of experiments, various RBF kernels were applied on the two-moon dataset to simulate linearly non-separable cases under kernel spaces.

The clustering results of kernel K-means with different RBF kernels (width = 4, 8, 16, 32) are shown in FIG2 (a) -2 (d).

Colors and decision boundaries stand for the same meaning as those in FIG1 .

Obviously, the performance of kernel K-means was getting worse with sub-optimal kernels, as in 2 (b), 2 (c) and 2 (d).

Searching for an optimal RBF kernel requires cross-validation among many candidates, which could result in a large number of iterations.

This procedure can be greatly eased by our kernel CPD-UML.

The CPD transformation under kernel spaces provides a supplementary force to the kernelization to further improve the data separability, the same as it performs under the input space.

FIG2 (f) -2 (h) demonstrate the effectiveness of our CPD-UML.

Same RBF kernels as in FIG2 (b) -2 (d) were used, but better clustering results were obtained.

The ability to work with sub-optimal kernels should also be regarded as a computational advantage of our model.

Experimental Setup In this section, we employ six benchmark datasets to evaluate the performance of our CPD-UML.

They are five UCI datasets 2 : Breast, Diabetes, Cars, Dermatology, E. Coli and the USPS_20 handwritten data.

Their basic information is summarized in Appendix B.Both linear and kernel versions of our proposed approach were tested.

For linear version, K-means method was used as the baseline for comparison.

In addition, three unsupervised metric learning solutions, AML , RPCA-OM BID12 and FME were utilized as the competing solutions.

For kernel version, the baseline algorithm is kernel K-means.

NAML BID2 , the kernel version of AML is adopted.

Since RPCA-OM and FME do not have their kernel version, the same kernelization strategy in 2.4 was applied to kernelize these two solutions.

RBF kernels were applied for all kernel solutions.

Each dataset was partitioned into seen and unseen data randomly.

Optimal cluster centers and parameters are determined by the seen data.

Clustering performance is evaluated via the unseen data, which are labeled directly based on their distances away from the cluster centers.

Similar setups have been used in BID11 BID6 .

In the experiments, we performed 3-fold cross validation, in which two folds were used as seen data and one fold as unseen data.

In the competing solutions, the hyper-parameters were searched within the same range as in their publications.

In our proposed approach, the regularization parameter λ and smooth parameter σ were searched from {10 0 ∼ 10 10 } and {2 0 ∼ 2 10 }, respectively.

The RBF kernel width for all kernel methods is chosen from {2 −5 ∼ 2 10 }.

Since the performance of tested methods depends on the initialization clusters, the clustering result of K-means was applied as the initialization clusters for all the competing solutions in each run.

The performance of each algorithm was calculated over 20 runs.

Results We measured the performance using the ground truth provided in all six benchmark datasets.

Three standard performance metrics were calculated: accuracy, normalized mutual information and purity.

To better compare the tested methods in statistic, we conducted a Student's t-test with a p-value 0.05 between each pair of solutions for each dataset.

The solutions were ranked using a scoring schema from BID17 .

Compared with other methods, an algorithm scores 1 if it performs significantly better than one opponent in statistic; 0.5 if there is no significant difference, and 0 if it is worse.

Tables 1, 2 and 3 summarize the clustering performance and ranking scores.

The best performance is identified in Boldface for each dataset.

It is evident that our CPD-UML outperforms other competing solutions in all three standard measurements with significant margins.

Highest ranking scores in the performance tables are all achieved by our kernel version approach.

In addition, significant improvements have been obtained by our proposed approach compared with the baseline algorithm K-means and kernel K-means.

It is also noteworthy that, the linear CPD-UML achieved comparable results with the other competing methods using RBF kernels, which further demonstrates the effectiveness of our nonlinear feature space transformation.

The proposed CPD-UML model learns a nonlinear metric and the clusters for the given data simultaneously.

The nonlinear metric is achieved by a globally smooth nonlinear transformation, which improves the separability of given data during clustering.

CPD is used as the transformation model because of its capability in deforming feature space in sophisticated yet smooth manner.

Evaluations on synthetic and benchmark datasets demonstrate the effectiveness of our approach.

Applying the proposed approach to other computer vision and machine learning problems are in the direction of our future research.

<|TLDR|>

@highlight

 a nonlinear unsupervised metric learning framework to boost the performance of clustering algorithms.
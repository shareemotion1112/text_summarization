We consider the problem of improving kernel approximation via feature maps.

These maps arise as Monte Carlo approximation to integral representations of kernel functions and scale up kernel methods for larger datasets.

We propose to use more efficient numerical integration technique to obtain better estimates of the integrals compared to the state-of-the-art methods.

Our approach allows to use information about the integrand to enhance approximation and facilitates fast computations.

We derive the convergence behavior and conduct an extensive empirical study that supports our hypothesis.

Kernel methods proved to be an efficient technique in numerous real-world problems.

The core idea of kernel methods is the kernel trick -compute an inner product in a high-dimensional (or even infinite-dimensional) feature space by means of a kernel function k:k(x, y) = ψ(x), ψ(y) ,where ψ : X → F is a non-linear feature map transporting elements of input space X into a feature space F. It is a common knowledge that kernel methods incur space and time complexity infeasible to be used with large-scale datasets directly.

For example, kernel regression has O(N 3 + N d 2 ) training time, O(N 2 ) memory, O(N d) prediction time complexity for N data points in original d-dimensional space X .

One of the most successful techniques to handle this problem BID18 ) introduces a low-dimensional randomized approximation to feature maps: DISPLAYFORM0 This is essentially carried out by using Monte-Carlo sampling to approximate scalar product in (1).

A randomized D-dimensional mappingΨ(·) applied to the original data input allows employing standard linear methods, i.e. reverting the kernel trick.

In doing so one reduces the complexity to that of linear methods, e.g. D-dimensional approximation admits O(N D 2 ) training time, O(N D) memory and O(N ) prediction time.

It is well known that as D → ∞, the inner product in (2) converges to exact kernel k(x, y).

Recent research BID22 ; BID9 ; BID4 ) aims to improve the convergence of approximation so that a smaller D can be used to obtain the same quality of approximation.

This paper considers kernels that allow the following integral representation k(x, y) = E q(w) g xy (w) ≈ E p(w) f xy (w) = I(f xy ), p(w) = 1 (2π) d/2 e − w 2 2 ,where q(w) is a density associated with a kernel, e.g. the popular Gaussian kernel has q(w) = p(w), so the exact equality holds with g xy (w) = f xy (w) = φ(w x) φ(w y), where φ(·) = [cos(·), sin(·)] .The class of kernels admitting the form in (3) covers shift-invariant kernels (e.g. radial basis function (RBF) kernels) and Pointwise Nonlinear Gaussian (PNG) kernels.

They are widely used in practice and have interesting connections with neural networks BID3 BID21 ).The main challenge for the construction of low-dimensional feature maps is the approximation of the expectation in (3) which is d-dimensional integral with Gaussian weight.

While standard MonteCarlo rule is easy to implement, there are better quadrature rules for such kind of integrals.

For example, BID22 apply quasi-Monte Carlo (QMC) rules and obtain better quality kernel matrix approximations compared to random Fourier features of BID18 .Unlike other research studies we refrain from using simple Monte Carlo estimate of the integral, instead, we propose to use specific quadrature rules.

We now list our contributions:1.

We propose to use advanced quadrature rules to improve kernel approximation accuracy.

We also provide an analytical estimate of the error for the used quadrature rules.

2.

We note that for kernels with specific integrand f xy (w) in (3) one can improve on its properties.

For example, for kernels with even function f xy (w) we derive the reduced quadrature rule which gives twice smaller embedded dimension D with the same accuracy.

This applies, for example, to any RBF kernel.

3.

We use structured orthogonal matrices (so-called butterfly matrices) when designing quadrature rule that allow fast matrix by vector multiplications.

As a result, we speed up the approximation of the kernel function and reduce memory requirements.

4.

We demonstrate our approach on a set of regression and classification problems.

Empirical results show that the proposed approach has a better quality of approximation of kernel function as well as better quality of classification and regression when using different kernels.

We start with rewriting the expectation in equation FORMULA2 as integral of f xy with respect to p(w): DISPLAYFORM0 Integration can be performed by means of quadrature rules.

The rules usually take a form of interpolating function that is easy to integrate.

Given such a rule, one may sample points from the domain of integration and calculate the value of the rule at these points.

Then, the sample average of the rule values would yield the approximation of the integral.

We use the average of sampled quadrature rules developed by BID12 to yield unbiased estimates of I(f xy ).

A change of coordinates is the first step to facilitate stochastic sphericalradial rules.

Now, let w = rz, with z z = 1, so that w w = r 2 for r ∈ [0, ∞], leaving us with DISPLAYFORM1 where DISPLAYFORM2 As mentioned earlier we are going to use a combination of radial R and spherical S rules.

We now describe the logic behind the used quadratures.

Stochastic radial rules.

Stochastic radial rule R(h) of degree 2l + 1 has the form of weighted symmetric sums: DISPLAYFORM3 where h is an integrand in infinite range integral DISPLAYFORM4 To get an unbiased estimate for T (h), points ρ i are sampled from specific distributions which depend on the degree of the rule.

Weights w i are derived so that R has a polynomial degree 2l + 1, i.e. is exact for integrands h(r) = r p with p = 0, 1, . . .

, 2l + 1.

For radial rules of degree three R 3 the point ρ 0 = 0, while ρ 1 ∼ χ(d + 2) follows Chi-distribution with d + 2 degrees of freedom.

Higher degrees require samples from more complex distributions which are hard to sample from.

Stochastic spherical rules.

Spherical rule S(s) approximates an integral of a function s(z) over the surface of unit d-sphere U d and takes the following form: DISPLAYFORM5 where z j are points on U d , i.e. z z = 1.

If we set weight w j = |U d | 2(d+1) and sum function s values at original and reflected vertices v j of randomly rotated d-simplex V, we will end up with a degree three rule: DISPLAYFORM6 where v j is the j'th vertex of d-simplex V with vertices on U d and Q is a random d × d orthogonal matrix.

We justify the choice of the degree in Appendix A.Since the value of the integral is approximated as the sample average, the key to unbiased estimate is proper randomization.

In this case, randomization is attained through the matrix Q. It is crucial to generate uniformly random orthogonal matrices to achieve an unbiased estimate for spherical surface integrals.

We consider various designs of such matrices further in Section 3.Stochastic spherical-radial rules.

Meanwhile, combining foregoing rules results in stochastic spherical-radial rule of degree three: DISPLAYFORM7 which we finally apply to the approximation of (4) by averaging the samples of SR

Q,ρ : DISPLAYFORM0 where n is the number of sampled SR rules.

Speaking in terms of approximate feature maps, the new feature dimension D in case of quadrature based approximation equals 2n(d + 1) as we sample n rules and evaluate each of them at 2(d + 1) points.

Surprisingly, empirical results (see Section 5) show that even a small number of rule samples n provides accurate approximations.

Properties of the integrand.

We also note here that for specific functions f xy (w) we can derive better versions of SR rule by taking on advantage of the knowledge about the integrand.

For example, the Gaussian kernel has f xy (w) = cos(w (x − y)).

Note that f is even, so we can discard an excessive term in the summation in (5), since f (w) = f (−w), i.e SR 3,3 rule reduces to DISPLAYFORM1 Variance of the error.

We contribute the variance estimation for the stochastic spherical-radial rules when applied to kernel function.

To the best of our knowledge, it has not been done before.

In case of kernel functions the integrand f xy can be represented as DISPLAYFORM2 where z 1 , z 2 are scalar values.

Using this representation of kernel function and its Taylor expansion we can obtain the following proposition (see Appendix B for detailed derivation of the result): Proposition 2.1.

The quadrature rule (6) is an unbiased estimate of integral of any integrable function f .

If function f can be represented in the form (8), i.e. f (w) = g(z 1 , z 2 ), z 1 = w x, z 2 = w y for some x, y ∈ R d , all 4-th order partial derivatives of g are bounded and D = 2n(d+1) is the number of generated features, then DISPLAYFORM3 and L = max{ x , y }.Constants M 1 , M 2 in the proposition are upper bounds on the derivatives of function g and don't depend on the data set, while L plays the role of the scale of inputs.

The proposition implies that the error of approximation is proportional to L -the less the scale, the better the accuracy (see Figure 1 ).

However, scaling input vectors is equivalent to changing the parameters of the kernel function.

For example, decreasing the norm of input variables for RBF kernel is equivalent to increasing the kernel width σ.

Therefore, the wide RBF kernels are approximated better than the narrow ones.

This result also gives us the rate of convergence O( 1 /nd) for the quadrature rule.

Figure 1: Relative error of approximation of kernel matrix with 95% confidence interval depending on the scaling factor, Gaussian kernel was used.

In this experiment for each scaling factor α we construct approximate kernel matrix K and the exact kernel matrix K using scaled input vectors x = x/α.

To plot the confidence interval we run each experiment 10 times each time generating new weights.

Experiment was conducted for 3 different input dimensions: d = 10, 100, 500.The quadrature rule (5) grants us some freedom in the choice of random orthogonal matrix Q. The next section discusses such matrices and suggests butterfly matrices for fast matrix by vector multiplication as the SR 3,3 rule implementation involves multiplication of the matrix QV by the data vector x.

Previously described stochastic spherical-radial rules require a random orthogonal matrix Q (see equation FORMULA10 ).

If Q follows Haar distribution on the set of all matrices in the orthogonal group O(d) in dimension d, then the averages of spherical rules S 3 Qi (s) provide unbiased degree three estimates for integrals over unit sphere.

Essentially, Haar distribution means that all orthogonal matrices in the group are equiprobable, i.e. uniformly random.

Methods for sampling such matrices vary in their complexity of generation and multiplication.

Techniques based on QR decomposition (Mezzadri (2006) ) have complexity cubic in d, and the resulting matrix does not allow fast matrix by vector multiplications.

Another set of methods is based on a sequence of reflectors BID20 ) or rotators BID0 ).

The complexity is better (quadratic in d), however the resulting matrix is unstructured and, thus, implicates no fast matrix by vector multiplication.

In BID5 random orthogonal matrices are considered.

They are constructed as a product of random diagonal matrices and Hadamard matrices and therefore enable fast matrix by vector products.

Unfortunately, they are not guaranteed to follow the Haar distribution.

To satisfy both our requirements, i.e low computational/space complexity and generation of Haar distributed orthogonal matrices, we propose to use so-called butterfly matrices.

Butterfly matrices.

The method from BID11 generates Haar distributed random orthogonal matrix B. As it happens to be a product of butterfly structured factors, a matrix of this type conveniently possesses the property of fast multiplication.

For d = 4 an example of butterfly orthogonal matrix is DISPLAYFORM0 is defined recursively as follows DISPLAYFORM1 is the same as B (d) with indexes i shifted by d, e.g. DISPLAYFORM2 has log d factors and each factor requires O(d) operations.

Another advantage is space complexity: DISPLAYFORM3 One can easily define butterfly matrix B (d) for the cases when d is not a power of two (see Appendix C.1 for details).

The randomization is based on the sampling of angles θ and we discuss it in Appendix C.2.

The method that uses butterfly orthogonal matrices is denoted B in the experiments section.

This section gives examples on how quadrature rules can be applied to a number of kernels.

Radial basis function (RBF) kernels are popular kernels widely used in kernel methods.

Gaussian kernel is a widely exploited RBF kernel and has the following form: DISPLAYFORM0 In this case the integral representation has φ(w x) = [cos(w x), sin(w x)] .

Since f xy (0) = 1, SR 3,3 rule for Gaussian kernel has the form (σ appears due to scaling): DISPLAYFORM1

Arc-cosine kernels were originally introduced by BID3 upon studying the connections between deep learning and kernel methods.

The integral representation of the b th -order arc-cosine kernel is DISPLAYFORM0 is the Heaviside function and p is the density of the standard Gaussian distribution.

Such kernels can be seen as an inner product between the representation produced by infinitely wide single layer neural network with random Gaussian weights.

They have closed form expression in terms of the angle θ = cos −1x y x y between x and y. DISPLAYFORM1 DISPLAYFORM2 Let φ 0 (w x) = Θ(w x) and φ 1 (w x) = max(0, w x), then we can rewrite the integral representation as follows: DISPLAYFORM3 .

For arccosine kernel of order 0 the value of the function φ 0 (0) = Θ(0) = 0.5 results in DISPLAYFORM4 In the case of arc-cosine kernel of order 1, the value of φ 1 (0) is 0 and the SR 3,3 rule reduces to DISPLAYFORM5

The explicit mapping can be written as follows: DISPLAYFORM0 so that W ∈ R 2n(d+1)×d , where only Q j ∈ R d×d and ρ j (j = 1, . . .

, n) are generated randomly.

For example, in case of the Gaussian kernel the mapping can be rewritten as DISPLAYFORM1

We extensively study the proposed method on several established benchmarking datasets: Powerplant, LETTER, USPS, MNIST, CIFAR100 BID15 ), LEUKEMIA BID14 ).

In Section 5.2 we show kernel approximation error across different kernels and number of features.

We also report the quality of SVM models with approximate kernels on the same data sets in Section 5.3.

The compared methods are described below.

We present a comparison of our method with estimators based on simple Monte Carlo 3 .

The Monte Carlo approach has a variety of ways to generate samples: unstructured Gaussian BID18 ), structured Gaussian BID9 ), random orthogonal matrices (ROM) BID5 ).1 It may be the case when sampling ρ that 1 − d ρ 2 < 0, simple solution is just to resample ρ to satisfy the non-negativity of the expression.

2 We do not use reflected points for Gaussian kernel as noted earlier, so DISPLAYFORM0 We also study quasi-Monte Carlo BID22 ) performance.

See Appendix D for details.

Quadrature rules.

Our main method that uses stochastic spherical-radial rules with Q = B 4 (butterfly matrix) is denoted by B. As mentioned earlier we also include a variant of our algorithm that uses an orthogonal matrix Q based on a sequence of random reflectors (we denote it as H).

To measure kernel approximation quality we use relative error in Frobenius norm DISPLAYFORM0 , where K andK denote exact kernel matrix and its approximation.

We run experiments for the kernel approximation on a random subset of a dataset (see Appendix D for details).

Approximation was constructed for different number of SR samples n = D 2(d+1) , where d is an original feature space dimensionality and D is the new one.

For the Gaussian kernel we set hyperparameter γ = 1 2σ 2 to the same value for all the approximants, while arc-cosine kernels have no hyperparameters.

We run experiments for each [kernel, dataset, n] tuple and plot 95% confidence interval around the mean value line.

FIG1 show results for kernel approximation error on LETTER, MNIST, CIFAR100 and LEUKEMIA datasets.

2 score using embeddings with three kernels (columns: arc-cosine 0, arccosine 1, Gaussian) on three datasets (rows: Powerplant, LETTER, USPS).

Higher is better.

The x-axis represents the factor to which we extend the original feature space, n = D 2(d+1) , where d is the dimensionality of the original feature space, D is the dimensionality of the new feature space.

We drop one of our methods H here since its kernel approximation almost coincides with B.We observe that for the most of the datasets and kernels the methods we propose in the paper (B, H) show better results than the baselines.

They do coincide almost everywhere, which is expected, as the B method is only different from H in the choice of the matrix Q to facilitate speed up.

We report accuracy and R 2 scores for the classification and regression tasks on the same data sets (see FIG2 .

We examine the performance with the same setting (the number of runs for each [kernel, dataset, n] tuple) as in experiments for kernel approximation error, except now we map the whole dataset.

We use Support Vector Machines to obtain predictions.

We also drop one of our methods H here since its kernel approximation almost coincides with B.Kernel approximation error does not fully define the final prediction accuracy -the best performing kernel matrix approximant not necessarily yields the best accuracy or R 2 score.

However, the empirical results illustrate that our method delivers comparable and often the best quality on the final tasks.

We also note that in many cases our method provides greater performance using less number of features n, e.g. LETTER and Powerplant datasets with arc-cosine kernel of the first order.

The most popular methods for scaling up kernel methods are based on a low-rank approximation of the kernel using either data-dependent or independent basis functions.

The first one includes Nyström method BID7 ), greedy basis selection techniques BID19 ), incomplete Cholesky decomposition BID10 ).The construction of basis functions in these techniques utilizes the given training set making them more attractive for some problems compared to Random Fourier Features approach.

In general, data-dependent approaches perform better than data-independent approaches when there is a gap in the eigen-spectrum of the kernel matrix.

The rigorous study of generalization performance of both approaches can be found in (Yang et al. (2012) ).In data-independent techniques, the kernel function is approximated directly.

Most of the methods (including the proposed approach) that follow this idea are based on Random Fourier Features BID18 ).

They require so-called weight matrix that can be generated in a number of ways.

BID16 form the weight matrix as a product of structured matrices.

It enables fast computation of matrix-vector products and speeds up generation of random features.

Another work BID9 ) orthogonalizes the features by means of orthogonal weight matrix.

This leads to less correlated and more informative features increasing the quality of approximation.

They support this result both analytically and empirically.

The authors also introduce matrices with some special structure for fast computations.

BID5 propose a generalization of the ideas from BID16 ) and BID9 ), delivering an analytical estimate for the mean squared error (MSE) of approximation.

All these works use simple Monte Carlo sampling.

However, the convergence can be improved by changing Monte Carlo sampling to Quasi-Monte Carlo sampling.

Following this idea BID22 apply quasi-Monte Carlo to Random Fourier Features.

In (Yu et al. (2015) ) the authors make attempt to improve quality of the approximation of Random Fourier Features by optimizing sequences conditioning on a given dataset.

Among the recent papers there are works that, similar to our approach, use the numerical integration methods to approximate kernels.

While BID1 carefully inspects the connection between random features and quadratures, they did not provide any practically useful explicit mappings for kernels.

Leveraging the connection BID6 propose several methods with Gaussian quadratures, among them three schemes are data-independent and one is data-dependent.

The authors do not compare them with the approaches for random feature generation other than random Fourier features.

The data-dependent scheme optimizes the weights for the quadrature points to yield better performance.

In this work we proposed to apply advanced integration rule that allowed us to achieve higher quality of kernel approximation.

Our derivation of the variance of the error implies the dependence of the error on the scale of data, which in case of Gaussian kernel can be interpreted as width of the kernel.

However, as we have seen earlier, accuracy on the final task has no direct dependence on the approximation quality, so we can only speculate whether better approximated wide kernels deliver better accuracy compared to the poorer approximated narrow ones.

It is interesting to explore this connection in the future work.

To speed up the computations we employed butterfly orthogonal matrices yielding the computational complexity O(d log d).

Although the procedure we used to generate butterfly matrices claims to produce uniformly random orthogonal matrices, we found that it is not always so.

However, the comparison of the method H (uses properly distributed orthogonal matrices) with method B (sometimes fails to do so) did not reveal any differences.

We also leave it for the future investigation.

Our experimental study confirms that for many kernels on the most datasets the proposed approach delivers better kernel approximation.

Additionally, the empirical results showed that the quality of the final task (classification/regression) is also higher than the state-of-the-art baselines.

The connection between the final score and the kernel approximation error is to be explored as well.

The degree of the rules.

We now discuss the choice of the degree for the SR rule.

BID13 show that higher degree rules, being more computationally expensive, often bring in only marginal improvement in performance.

For these reasons in our experiments we use the rule of degree three SR 3,3 , i.e. a combination of radial rule R 3 and spherical rule S 3 .

The function f xy in quadrature rule FORMULA12 can be considered as a function of two variables, i.e. f xy = φ(w x)φ(w y) = g(z 1 , z 2 ), where z 1 = w x, z 2 = w y.

In the quadrature rule ρ ∼ χ(d + 2) and Q is a random orthogonal matrix.

Therefore, random variables w i = Qv i are uniformly distributed on a unit-sphere.

Now, let's write down 4-th order Taylor expansion with Lagrange remainder of function g(ρw i x, ρw i y) + g(−ρw i x, −ρw i y) around 0 (odd terms cancel out) DISPLAYFORM0 1 is between 0 and (ρw i x), i 2 is between 0 and (ρw i y), i 3 is between 0 and (−ρw i x) and i 4 is between 0 and (−ρw i y).

Plugging this expression into (5) we obtain DISPLAYFORM1 Q is a random orthogonal matrix uniformly distributed on a set of orthogonal matrices O(n).

From uniformity of orthogonal matrix Q it follows that vector w j is uniform on a unit n-sphere.

Also note that A i and B j are independent if i = j, B i and B j are independent if i = j, however, A i and A j are dependent as they have common random variable ρ.

Therefore, DISPLAYFORM2 Let us calculate the variance of the estimate.

DISPLAYFORM3 Distribution of random variable ρ is χ(d + 2), therefore DISPLAYFORM4 , d > 2.1.

Now, let us calculate the variance VS i from the first term of equation FORMULA34 DISPLAYFORM5 where DISPLAYFORM6 To calculate expectations of the form E(w x) k we will use expression for integral of monomial over unit sphere Baker (1997) DISPLAYFORM7 DISPLAYFORM8 For example, let us show how to calculate E(w x)4 : DISPLAYFORM9 = thanks to symmetry all terms, for which at least one index doesn't coincide with other indices, are equal to 0.

= E DISPLAYFORM10 Using the same technique for (10) we obtain DISPLAYFORM11 For the variance of A i we have DISPLAYFORM12 where M 2 = max j=0,1,2 DISPLAYFORM13 Let's estimate covariance Cov(A i , B i ): DISPLAYFORM14 The first term of the right hand side: DISPLAYFORM15 The second term EA i EB i (again for x ≥ y ) DISPLAYFORM16 Combining the derived inequalities we obtain DISPLAYFORM17 2.

Now, let's examine the expectation of the second term in (9): Substituting FORMULA0 and FORMULA0 into FORMULA34 we obtain DISPLAYFORM18 DISPLAYFORM19 And finally For the cases when d is not a power of two, the resulting B has deficient columns with zeros ( FIG3 , right), which introduces a bias to the integral estimate.

To correct for this bias one may apply additional randomization by using a product BP, where P ∈ {0, 1} d×d is a permutation matrix.

DISPLAYFORM20 Even better, use a product of several BP's: B = (BP) 1 (BP) 2 . . . (BP) t .

We set t = 3 in the experiments.

The key to uniformly random orthogonal butterfly matrix B is the sequence of d−1 angles θ i .

To get B (d) Haar distributed, we follow BID8 algorithm that first computes a uniform random point u from U d .

It then calculates the angles by taking the ratios of the appropriate u coordinates θ i = ui ui+1 , followed by computing cosines and sines of the θ's.

Here we discuss the datasets that did not appear in the main body of the paper.

TAB2 displays the settings for the experiments across the datasets.

Figure 5 shows the results for the kernel approxi- BID9 ).

It also has larger constant factor hidden under O notation and higher complexity.

For QMC the weight matrix M is generated as a transformation of quasi-random sequences.

We run our experiments with Halton sequences in compliance with the previous work (see Figure 5 with QMC method included).Although, for the arc-cosine kernels, our methods are the best performing estimators, for the Gaussian kernel the error is not always the lowest one and depends on the dataset, e.g. on the USPS dataset the lowest is Monte Carlo with ROM.

However, for the most of the datasets we demonstrate superiority of our approach with this kernel.

We also notice that the dataset with a small amount of features, Powerplant, enjoys Halton and Orthogonal Random Features best, while ROM's convergence stagnates at some point.

This could be due the small input feature space with d = 4 and we leave it for the future investigation.

We also included subsampled dense grid method from BID6 into our comparison as it is the only data-independent approach from the paper that is shown to work well.

We reimplemented code for the paper to the best of our knowledge since it is not open sourced.

We run comparison on all datasets with Gaussian kernel.

The FIG6 illustrates the performance on the LETTER dataset across different expansions.

We can see almost coinciding performance of the method (denoted GQ) with the baseline RFF (denoted G).

For other datasets the figures are very similar to the case of LETTER, with RFF and GQ methods showing nearly matching relative error of kernel approximation.

It should also be noted that explicit mappings produced by Gaussian quadratures do not possess any convenient structure and, thus, cannot boast any better computational complexity.

We have also run the time measurement on our somewhat unoptimized implementation of the proposed method.

Indeed, FIG8 demonstrates that the method scales as theoretically predicted with larger dimensions thanks to the structured nature of the mapping.

Thanks to structured explicit mapping of the proposed method B, it is favorable for higher dimensional data.

<|TLDR|>

@highlight

Quadrature rules for kernel approximation.

@highlight

The paper proposes improving the kernel approximation of random features by using quadrature rules like stochastic spherical-radial rules.

@highlight

The authors propose a novel version of the random feature map approach to approximately solve large-scale kernel problems.

@highlight

This paper shows that techniques due to Genz & Monahan (1998) can be used to achieve low kernel approximation error under the framework of random fourier feature, a new way to apply quadrature rules to improve kernel approximation.
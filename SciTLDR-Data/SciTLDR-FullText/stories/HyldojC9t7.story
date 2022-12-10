We present a new methodology that constructs a family of \emph{positive definite kernels} from any given dissimilarity measure on structured inputs whose elements are either real-valued time series or discrete structures such as strings, histograms, and graphs.

Our approach, which we call D2KE (from Distance to Kernel and Embedding), draws from the literature of Random Features.

However, instead of deriving random feature maps from a user-defined kernel to approximate kernel machines, we build a kernel from a random feature map, that we specify given the distance measure.

We further propose use of a finite number of random objects to produce a random feature embedding of each instance.

We provide a theoretical analysis showing that D2KE enjoys better generalizability than universal Nearest-Neighbor estimates.

On one hand, D2KE subsumes the widely-used \emph{representative-set method} as a special case, and relates to the well-known \emph{distance substitution kernel} in a limiting case.

On the other hand, D2KE generalizes existing \emph{Random Features methods} applicable only to vector input representations to complex structured inputs of variable sizes.

We conduct classification experiments over such disparate domains as time series, strings, and histograms (for texts and images), for which our proposed framework compares favorably to existing distance-based learning methods in terms of both testing accuracy and computational time.

In many problem domains, it is easier to specify a reasonable dissimilarity (or similarity) function between instances than to construct a feature representation.

This is particularly the case with structured inputs whose elements are either real-valued time series or discrete structures such as strings, histograms, and graphs, where it is typically less than clear how to construct the representation of entire structured inputs with potentially widely varying sizes, even when given a good feature representation of each individual component.

Moreover, even for complex structured inputs, there are many well-developed dissimilarity measures, such as the Dynamic Time Warping measure between time series, Edit Distance between strings, Hausdorff distance between sets, and Wasserstein distance between distributions.

However, standard machine learning methods are designed for vector representations, and classically there has been far less work on distance-based methods for either classification or regression on structured inputs.

The most common distance-based method is Nearest-Neighbor Estimation (NNE), which predicts the outcome for an instance using an average of its nearest neighbors in the input space, with nearness measured by the given dissimilarity measure.

Estimation from nearest neighbors, however, is unreliable, specifically having high variance when the neighbors are far apart, which is typically the case when the intrinsic dimension implied by the distance is large.

To address this issue, a line of research has focused on developing global distance-based (or similaritybased) machine learning methods BID38 BID16 BID1 BID12 , in large part by drawing upon connections to kernel methods BID43 or directly learning with similarity functions BID1 BID12 BID2 BID29 ; we refer the reader in particular to the survey in BID7 .

Among these, the most direct approach treats the data similarity matrix (or transformed dissimilarity matrix) as a kernel Gram matrix, and then uses standard kernel-based methods such as Support Vector Machines (SVM) or kernel ridge regression with this Gram matrix.

A key caveat with this approach however is that most similarity (or dissimilarity) measures do not provide a positive-definite (PD) kernel, so that the empirical risk minimization problem is not well-defined, and moreover becomes non-convex BID33 BID28 .A line of work has therefore focused on estimating a positive-definite (PD) Gram matrix that merely approximates the similarity matrix.

This could be achieved for instance by clipping, or flipping, or shifting eigenvalues of the similarity matrix BID36 , or explicitly learning a PD approximation of the similarity matrix BID6 BID8 .

Such modifications of the similarity matrix however often leads to a loss of information; moreover, the enforced PD property is typically guaranteed to hold only on the training data, resulting in an inconsistency between the set of testing and training samples BID7 1.Another common approach is to select a subset of training samples as a held-out representative set, and use distances or similarities to structured inputs in the set as the feature function BID20 BID36 ).

As we will show, with proper scaling, this approach can be interpreted as a special instance of our framework.

Furthermore, our framework provides a more general and richer family of kernels, many of which significantly outperform the representative-set method in a variety of application domains.

To address the aforementioned issues, in this paper, we propose a novel general framework that constructs a family of PD kernels from a dissimilarity measure on structured inputs.

Our approach, which we call D2KE (from Distance to Kernel and Embedding), draws from the literature of Random Features BID39 , but instead of deriving feature maps from an existing kernel for approximating kernel machines, we build novel kernels from a random feature map specifically designed for a given distance measure.

The kernel satisfies the property that functions in the corresponding Reproducing Kernel Hilbert Space (RKHS) are Lipschitz-continuous w.r.t.

the given distance measure.

We also provide a tractable estimator for a function from this RKHS which enjoys much better generalization properties than nearest-neighbor estimation.

Our framework produces a feature embedding and consequently a vector representation of each instance that can be employed by any classification and regression models.

In classification experiments in such disparate domains as strings, time series, and histograms (for texts and images), our proposed framework compares favorably to existing distance-based learning methods in terms of both testing accuracy and computational time, especially when the number of data samples is large and/or the size of structured inputs is large.

We highlight our main contributions as follows:• From the perspective of distance kernel learning, we propose for the first time a methodology that constructs a family of PD kernels via Random Features from a given distance measure for structured inputs, and provide theoretical and empirical justifications for this framework.• From the perspective of Random Features (RF) methods, we generalize existing Random Features methods applied only to vector input representations to complex structured inputs of variable sizes.

To the best of our knowledge, this is the first time that a generic RF method has been used to accelerate kernel machines on structured inputs across a broad range of domains such as time-series, strings, and the histograms.

Distance-Based Kernel Learning.

Existing approaches either require strict conditions on the distance function (e.g. that the distance be isometric to the square of the Euclidean distance) BID22 BID42 , or construct empirical PD Gram matrices that do not necessarily generalize to the test samples BID36 BID38 BID34 BID16 .

BID22 and BID42 provide conditions under which one can obtain a PD kernel through simple transformations of the distance measure, but which are not satisfied for many commonly used dissimilarity measures such as Dynamic Time Warping, Hausdorff distance, and Earth Mover's distance (Haasdonk & Bahlmann, 1A generalization error bound was provided for the similarity-as-kernel approach in BID7 , but only for a positive-definite similarity function. ).

Equivalently, one could also find a Euclidean embedding (also known as dissimilarity representation) approximating the dissimilarity matrix as in Multidimensional Scaling BID36 BID38 BID34 BID16 2.

Differently, BID29 presented a theoretical foundation for an SVM solver in Krein spaces and directly evaluated a solution that uses the original (indefinite) similarity measure.

There are also some specific approaches dedicated to building a PD kernel on some structured inputs such as text and time-series BID11 BID13 , that modify a distance function over sequences to a kernel by replacing the minimization over possible alignments into a summation over all possible alignments.

This type of kernel, however, results in a diagonal-dominance problem, where the diagonal entries of the kernel Gram matrix are orders of magnitude larger than the off-diagonal entries, due to the summation over a huge number of alignments with a sample itself.

Interest in approximating non-linear kernel machines using randomized feature maps has surged in recent years due to a significant reduction in training and testing times for kernel based learning algorithms BID14 .

There are numerous explicit nonlinear random feature maps that have been constructed for various types of kernels, including Gaussian and Laplacian Kernels BID39 BID48 , intersection kernels BID30 ), additive kernels BID47 , dot product kernels BID25 BID37 , and semigroup kernels BID31 .

Among them, the Random Fourier Features (RFF) method, which approximates a Gaussian Kernel function by means of multiplying the input with a Gaussian random matrix, and its fruitful variants have been extensively studied both theoretically and empirically BID46 BID17 BID41 BID0 BID10 .

To accelerate the RFF on input data matrix with high dimensions, a number of methods have been proposed to leverage structured matrices to allow faster matrix computation and less memory consumption BID27 BID23 BID9 .However, all the aforementioned RF methods merely consider inputs with vector representations, and compute the RF by a linear transformation that is either a matrix multiplication or an inner product under Euclidean distance metric.

In contrast, D2KE takes structured inputs of potentially different sizes and computes the RF with a structured distance metric (typically with dynamic programming or optimal transportation).

Another important difference between D2KE and existing RF methods lies in the fact that existing RF work assumes a user-defined kernel and then derives a randomfeature map, while D2KE constructs a new PD kernel through a random feature map and makes it computationally feasible via RF.

The table 1 lists the differences between D2KE and existing RF methods.

A very recent piece of work BID49 has developed a kernel and a specific algorithm for computing embeddings of single-variable real-valued time-series.

However, despite promising results, this method cannot be applied on discrete structured inputs such as strings, histograms, and graphs.

In contrast, we have an unified framework for various structured inputs beyond the limits of BID49 and provide a general theoretical analysis w.r.t KNN and other generic distance-based kernel methods.

We consider the estimation of a target function f : X → R from a collection of samples {( DISPLAYFORM0 , where x i ∈ X is the structured input object, and y i ∈ Y is the output observation associated with the target function f (x i ).

For instance, in a regression problem, y i ∼ f (x i ) + ω i ∈ R for some random noise ω i , and in binary classification, we have y i ∈ {0, 1} with P(y i = 1|x i ) = f (x i ).

We are given a dissimilarity measure d : X × X → R between input objects instead of a feature representation of x.2A proof of the equivalence between PD of similarity matrix and Euclidean of dissimilarity matrix can be found in BID4 .Note that the size structured inputs x i may vary widely, e.g. strings with variable lengths or graphs with different sizes.

For some of the analyses, we require the dissimilarity measure to be a metric as follows.

DISPLAYFORM1 An ideal feature representation for the learning task is (i) compact and (ii) such that the target function f (x) is a simple (e.g. linear) function of the resulting representation.

Similarly, an ideal dissimilarity measure d(x 1 , x 2 ) for learning a target function f (x) should satisfy certain properties.

On one hand, a small dissimilarity d(x 1 , x 2 ) between two objects should imply small difference in the function DISPLAYFORM2 On the other hand, we want a small expected distance among samples, so that the data lies in a compact space of small intrinsic dimension.

We next build up some definitions to formalize these properties.

Assumption 2 (Lipschitz Continuity).

For any DISPLAYFORM3 We would prefer the target function to have a small Lipschitz-continuity constant L with respect to the dissimilarity measure d(., .).

Such Lipschitz-continuity alone however might not suffice.

For example, one can simply set d(x 1 , x 2 ) = ∞ for any x 1 x 2 to satisfy Eq. equation 1.

We thus need the following quantity that measures the size of the space implied by a given dissimilarity measure.

DISPLAYFORM4

Assuming the input domain X is compact, the covering number N(δ; X, d) measures its size w.r.t.

the distance measure d. We show how the two quantities defined above affect the estimation error of a Nearest-Neighbor Estimator.

DISPLAYFORM0 We extend the standard analysis of the estimation error of k-nearest-neighbor from finite-dimensional vector spaces to any structured input space X, with an associated distance measure d, and a finite covering number N(δ; X, d), by defining the effective dimension as follows.

Assumption 3 (Effective Dimension).

Let the effective dimension p X,d > 0 be the minimum p satisfying DISPLAYFORM1 Here we provide an example of effective dimension in case of measuring the space of Multiset.

A multiset is a set that allows duplicate elements.

Consider two multisets DISPLAYFORM0 be a ground distance that measures the distance between two elements u i , v j ∈ V in a set.

The (modified) Hausdorff Distance BID15 DISPLAYFORM1 Let N(δ; V, ∆) be the covering number of V under the ground distance ∆. Let X denote the set of all sets of size bounded by L. By constructing a covering of X containing any set of size less or equal than L with its elements taken from the covering of V, we have N(δ; DISPLAYFORM2 Equipped with the concept of effective dimension, we can obtain the following bound on the estimation error of the k-Nearest-Neighbor estimate of f (x).Theorem 1.

Let V ar(y| f (x)) ≤ σ 2 , andf n be the k-Nearest Neighbor estimate of the target function f constructed from a training set of size n. Denote p := p X,d .

We have DISPLAYFORM3 for some constant c > 0.

For σ > 0, minimizing RHS w.r.t.

the parameter k, we have DISPLAYFORM4 Proof.

The proof is almost the same to a standard analysis of k-NN's estimation error in, for example, BID21 , with the space partition number replaced by the covering number, and dimension replaced by the effective dimension in Assumption 3.When p X,d is reasonably large, the estimation error of k-NN decreases quite slowly with n. Thus, for the estimation error to be bounded by , requires the number of samples to scale exponentially in p X,d .

In the following sections, we develop an estimatorf based on a RKHS derived from the distance measure, with a considerably better sample complexity for problems with higher effective dimension.

We aim to address the long-standing problem of how to convert a distance measure into a positivedefinite kernel.

Here we introduce a simple but effective approach D2KE that constructs a family of positive-definite kernels from a given distance measure.

Given an structured input domain X and a distance measure d(., .), we construct a family of kernels as DISPLAYFORM0 where ω ∈ Ω is a random structured object whose elements could be real-valued time-series, strings, and histograms, p(ω) is a distribution over Ω, and φ ω (x) is a feature map derived from the distance of x to all random objects ω ∈ Ω.

The kernel is parameterized by both p(ω) and γ.

Relationship to Distance Substitution Kernel.

An insightful interpretation of the kernel in Equation (4) can be obtained by expressing the kernel in Equation FORMULA12 as DISPLAYFORM1 where the soft minimum function, parameterized by p(ω) and γ, is defined as DISPLAYFORM2 Therefore, the kernel k(x, y) can be interpreted as a soft version of the distance substitution kernel BID22 , where instead of substituting d(x, y) into the exponent, it substitutes a soft version of the form DISPLAYFORM3 Note when γ → ∞, the value of Equation FORMULA15 is determined by min ω ∈Ω d(x, ω) + d(ω, y), which equals d(x, y) if X ⊆ Ω, since it cannot be smaller than d(x, y) by the triangle inequality.

In other words, when X ⊆ Ω, DISPLAYFORM4 On the other hand, unlike the distance-substituion kernel, our kernel in Equation FORMULA13 is always PD by construction.

1: Draw R samples from p(ω) to get {ω j } R j=1 .

2: Set the R-dimensional feature embedding aŝ DISPLAYFORM0 3: Solve the following problem for some µ > 0: DISPLAYFORM1 Random Feature Approximation.

The reader might have noticed that the kernel in Equation FORMULA12 cannot be evaluated analytically in general.

However, this does not prohibit its use in practice, so long as we can approximate it via Random Features (RF) BID39 , which in our case is particularly natural as the kernel itself is defined via a random feature map.

Thus, our kernel with the RF approximation can not only be used in small problems but also in large-scale settings with a large number of samples, where standard kernel methods with O(n 2 ) complexity are no longer efficient enough and approximation methods, such as Random Features, must be employed BID39 .

Given the RF approximation, one can then directly learn a target function as a linear function of the RF feature map, by minimizing a domain-specific empirical risk.

It is worth noting that a recent work BID45 ) that learns to select a set of random features by solving an optimization problem in an supervised setting is orthogonal to our D2KE approach and could be extended to develop a supervised D2KE method.

We outline this overall RF based empirical risk minimization for our class of D2KE kernels in Algorithm 1.

It is worth pointing out that in line 2 of Algorithm 1 the random feature embeddings are computed by a structured distance measure between the original structured inputs and the generated random structured inputs, followed by the application of the exponent function parameterized by γ.

This is in contrast with traditional RF methods that translate the input data matrix into the embedding matrix via a matrix multiplication with random Gaussian matrix followed by a non-linearity.

We will provide a detailed analysis of our estimator in Algorithm 1 in Section 5, and contrast its statistical performance to that of K-nearest-neighbor.

Relationship to Representative-Set Method.

A naive choice of p(ω) relates our approach to the representative-set method (RSM): setting Ω = X, with p(ω) = p(x).

This gives us a kernel Equation (4) that depends on the data distribution.

One can then obtain a Random-Feature approximation to the kernel in Equation (4) by holding out a part of the training data {x j } R j=1 as samples from p(ω), and creating an R-dimensional feature embedding of the form: DISPLAYFORM2 as in Algorithm 1.

This is equivalent to a 1/ √ R-scaled version of the embedding function in the representative-set method (or similarity-as-features method) BID20 BID36 BID38 BID34 BID7 BID16 , where one computes each sample's similarity to a set of representatives as its feature representation.

However, here by interpreting Equation (8) as a random-feature approximation to the kernel in Equation (4), we obtain a much nicer generalization error bound even in the case R → ∞. This is in contrast to the analysis of RSM in BID7 , where one has to keep the size of the representative set small (of the order O(n)) in order to have reasonable generalization performance.

The choice of p(ω) plays an important role in our kernel.

Surprisingly, we found that many "close to uniform" choices of p(ω) in a variety of domains give better performance than for instance the choice of the data distribution p(ω) = p(x) (as in the representative-set method).

Here are some examples from our experiments: i) In the time-series domain with dissimilarity computed via Dynamic Time Warping (DTW), a distribution p(ω) corresponding to random time series of length uniform in ∈ [2, 10], and with Gaussian-distributed elements, yields much better performance than the Representative-Set Method (RSM); ii) In string classification, with edit distance, a distribution p(ω) corresponding to random strings with elements uniformly drawn from the alphabet Σ yields much better performance than RSM; iii) When classifying sets of vectors with the Hausdorff distance in Equation (2), a distribution p(ω) corresponding to random sets of size uniform in ∈ [3, 15] with elements drawn uniformly from a unit sphere yields significantly better performance than RSM.We conjecture two potential reasons for the better performance of the chosen distributions p(ω) in these cases, though a formal theoretical treatment is an interesting subject we defer to future work.

Firstly, as p(ω) is synthetic, one can generate unlimited number of random features, which results in a much better approximation to the exact kernel in Equation (4).

In contrast, RSM requires held-out samples from the data, which could be quite limited for a small data set.

Second, in some cases, even with a small or similar number of random features to RSM, the performance of the selected distribution still leads to significantly better results.

For those cases we conjecture that the selected p(ω) generates objects that capture semantic information more relevant to the estimation of f (x), when coupled with our feature map under the dissimilarity measure d(x, ω).

In this section, we analyze the proposed framework from the perspectives of error decomposition.

Let H be the RKHS corresponding to the kernel in Equation (4).

Let DISPLAYFORM0 be the population risk minimizer subject to the RKHS norm constraint f H ≤ C. And let DISPLAYFORM1 be the corresponding empirical risk minimizer.

In addition, letf R be the estimated function from our random feature approximation (Algorithm 1).

Then denote the population and empirical risks as L( f ) andL( f ) respectively.

We have the following risk decomposition DISPLAYFORM2 In the following, we will discuss the three terms from the rightmost to the leftmost.

Function Approximation Error.

The RKHS implied by the kernel in Equation FORMULA12 is DISPLAYFORM3 which is a smaller function space than the space of Lipschitz-continuous function w.r.t.

the distance d(x 1 , x 2 ).

As we show, any function f ∈ H is Lipschitz-continous w.r.t.

the distance d(., .).

where L f = γC.We refer readers to the detailed proof in Appendix A.1.

While any f in the RKHS is Lipschitzcontinuous w.r.t.

the given distance d(., .), we are interested in imposing additional smoothness via the RKHS norm constraint f H ≤ C, and by the kernel parameter γ.

The hope is that the best function f C within this class approximates the true function f well in terms of the approximation error L( f C ) − L( f ).

The stronger assumption made by the RKHS gives us a qualitatively better estimation error, as discussed below.

Estimation Error.

Define D λ as DISPLAYFORM0 is the eigenvalues of the kernel in Equation (5) and λ is a tuning parameter.

It holds that for any λ ≥ D λ /n, with probability at least 1 − δ, L(f n ) − L( f C ) ≤ c(log 1 δ ) 2 C 2 λ for some universal constant c BID50 .

Here we would like to set λ as small as possible (as a function of n).

By using the following kernel-independent bound: D λ ≤ 1/λ, we have λ = 1/ √ n and thus a bound on the estimation error DISPLAYFORM1 The estimation error is quite standard for a RKHS estimator.

It has a much better dependency w.r.t.

n (i.e. n −1/2 ) compared to that of k-nearest-neighbor method (i.e. n −2/(2+p X, d ) ) especially for higher effective dimension.

A more careful analysis might lead to tighter bound on D λ and also a better rate w.r.t.

n. However, the analysis of D λ for our kernel in Equation FORMULA12 is much more difficult than that of typical cases as we do not have an analytic form of the kernel.

Random Feature Approximation.

DenoteL(.) as the empirical risk function.

The error from RF approximation DISPLAYFORM2 where the first and third terms can be bounded via the same estimation error bound in Equation FORMULA3 , as bothf R andf n have RKHS norm bounded by C. Therefore, in the following, we focus only on the second term of empirical risk.

We start by analyzing the approximation error of the kernel DISPLAYFORM3 , we have uniform convergence of the form P max DISPLAYFORM4 , where p X,d is the effective dimension of X under metric d(., .).

In other words, to guarantee |∆ R (x 1 , x 2 )| ≤ with probability at least 1 − δ, it suffices to have DISPLAYFORM5 We refer readers to the detailed proof in Appendix A.2.

Proposition 2 gives an approximation error in terms of kernel evaluation.

To get a bound on the empirical riskL(f R ) −L(f n ), consider the optimal solution of the empirical risk minimization.

By the Representer theorem we havê DISPLAYFORM6 .

Therefore, we have the following corollary.

Corollary 1.

To guaranteeL(f R ) −L(f n ) ≤ , with probability 1 − δ, it suffices to have DISPLAYFORM7 where M is the Lipschitz-continuous constant of the loss function (.

, y), and A is a bound on α 1 /n.

We refer readers to the detailed proof in Appendix A.3.

For most of loss functions, A and M are typically small constants.

Therefore, Corollary 1 states that it suffices to have number of Random Features proportional to the effective dimension O(p X,d / 2 ) to achieve an approximation error.

Combining the three error terms, we can show that the proposed framework can achieve -suboptimal performance.

Claim 1.

Letf R be the estimated function from our random feature approximation based ERM estimator in Algorithm 1, and let f * denote the desired target function.

Suppose further that for some absolute constants c 1 , c 2 > 0 (up to some logarithmic factor of 1/ and 1/δ):1.

The target function f * lies close to the population risk minimizer f C lying in the RKHS spanned by the D2KE kernel: DISPLAYFORM8

We then have that: L(f R ) − L( f * ) ≤ with probability 1 − δ.

We evaluate the proposed method in four different domains involving time-series, strings, texts, and images.

First, we discuss the dissimilarity measures and data characteristics for each set of experiments.

Then we introduce comparison among different distance-based methods and report corresponding results.

Distance Measures.

We have chosen three well-known dissimilarity measures: 1) Dynamic Time Warping (DTW), for time-series BID3 ; 2) Edit Distance (Levenshtein distance), for strings BID32 ); 3) Earth Mover's distance BID40 for measuring the semantic distance between two Bags of Words (using pretrained word vectors), for representing documents.

4) (Modified) Hausdorff distance BID24 BID15 for measuring the semantic closeness of two Bags of Visual Words (using SIFT vectors), for representing images.

Note that Bag of (Visual) Words in 3) and 4) can also be regarded as a histogram.

Since most distance measures are computationally demanding, having quadratic complexity, we adapted or implemented C-MEX programs for them; other codes were written in Matlab.

Datasets.

For each domain, we selected 4 datasets for our experiments.

For time-series data, all are multivariate time-series and the length of each time-series varies from 2 to 205 observations; three are from the UCI Machine Learning repository BID18 , the other is generated from the IQ (In-phase and Quadrature components) samples from a wireless line-of-sight communication system from GMU.

For string data, the size of alphabet is between 4 and 8 and the length of each string ranges from 34 to 198; two of them are from the UCI Machine Learning repository and the other two from the LibSVM Data Collection BID5 .

For text data, all are chosen partially overlapped with these in BID26 .

The length of each document varies from 9.9 to 117.

For image data, all of datasets were derived from Kaggle; we computed a set of SIFTdescriptors to represent each image and the size of SIFT feature vectors of each image varies from 1 to 914.

We divided each dataset into 70/30 train and test subsets (if there was no predefined train/test split).

Properties of these datasets are summarized in TAB5 in Appendix B. Baselines.

We compare D2KE against 5 state-of-the-art baselines, including 1) KNN: a simple yet universal method to apply any distance measure to classification tasks; 2) DSK_RBF BID22 : distance substitution kernels, a general framework for kernel construction by substituting a problem specific distance measure in ordinary kernel functions.

We use a Gaussian RBF kernel; 3) DSK_ND BID22 : another class of distance substitution kernels with negative distance; 4) KSVM BID29 : learning directly from the similarity (indefinite) matrix followed in the original Krein Space; 5) RSM BID36 : building an embedding by computing distances from randomly selected representative samples.

Among these baselines, KNN, DSK_RBF, DSK_ND, and KSVM have quadratic complexity O(N 2 L 2 ) in both the number of data samples and the length of the sequences, while RSM has computational complexity O(N RL 2 ), linear in the number of data samples but still quadratic in the length of the sequence.

These compare to our method, D2KE, which has complexity O(N RL), linear in both the number of data samples and the length of the sequence.

For each method, we search for the best parameters on the training set by performing 10-fold cross validation.

For our new method D2KE, since we generate random samples from the distribution, we can use as many as needed to achieve performance close to an exact kernel.

We report the best number in the range R = [4, 4096] (typically the larger R is, the better the accuracy).

We employ a linear SVM implemented using LIBLINEAR (Fan et al., 2008) for all embedding-based methods (RSM and D2KE) and use LIBSVM BID5 for precomputed dissimilairty kernels (DSK_RBF, DSK_ND, and KSVM).

More details of experimental setup are provided in Appendix B. TAB1 , 4, and 5, D2KE can consistently outperform or match the baseline methods in terms of classification accuracy while requiring far less computation time.

There are several observations worth making here.

First, D2KE performs much better than KNN, supporting our claim that D2KE can be a strong alternative to KNN across applications.

Second, compared to the two distance substitution kernels DSK_RBF and DSK_ND and the KSVM method operating directly on indefinite similarity matrix, our method can achieve much better performance, suggesting that a representation induced from a truly PD kernel makes significantly better use of the data than indefinite kernels.

Among all methods, RSM is closest to our method in terms of practical construction of the feature matrix.

However, the random objects (time-series, strings, or sets) sampled by D2KE perform significantly better, as we discussed in section 4.

More detailed discussions of the experimental results for each domain are given in Appendix C.

In this work, we have proposed a general framework for deriving a positive-definite kernel and a feature embedding function from a given dissimilarity measure between input objects.

The framework is especially useful for structured input domains such as sequences, time-series, and sets, where many well-established dissimilarity measures have been developed.

Our framework subsumes at least two existing approaches as special or limiting cases, and opens up what we believe will be a useful new direction for creating embeddings of structured objects based on distance to random objects.

A promising direction for extension is to develop such distance-based embeddings within a deep architecture to support use of structured inputs in an end-to-end learning system.

DISPLAYFORM0 Proof.

Note the function g(t) = exp(−γt) is Lipschitz-continuous with Lipschitz constant γ.

Therefore, DISPLAYFORM1 Proof.

Our goal is to bound the magnitude of DISPLAYFORM2 Hoefding's inequality, we have DISPLAYFORM3 a given input pair (x 1 , x 2 ).

To get a unim bound that holds ∀(x 1 , x 2 ) ∈ X × X, we find an -covering E of X w.r.t.

d(., .) of size N( , X, d).

Applying union bound over the -covering E for x 1 and x 2 , we have P max DISPLAYFORM4 Then by the definition of E we have |d( DISPLAYFORM5 Together with the fact that exp(−γt) is Lipschitz-continuous with parameter γ for t ≥ 0, we have DISPLAYFORM6 for γ chosen to be ≤ 1.

This gives us DISPLAYFORM7 Combining equation 13 and equation 14, we have P max DISPLAYFORM8 Choosing = t/6γ yields the result.

A.3 P C 1Proof.

First of all, we have DISPLAYFORM9 by the optimality of {α j } n j=1 w.r.t.

the objective using the approximate kernel.

Then we havê DISPLAYFORM10 where A is a bound on α 1 /n.

Therefore to guaranteê DISPLAYFORM11 Then applying Theorem 2 leads to the result.

B G E S General Setup.

For each method, we search for the best parameters on the training set by performing 10-fold cross validation.

Following BID22 , we use an exact RBF kernel for DSK_RBF while choosing squared distance for DSK_ND.

We use the Matlab implementation provided by BID29 to run experiments for KSVM.

Similarly, we adopted a simple method -random selection -to obtain R = [4, 512] data samples as the representative set for RSM BID36 ).

For our new method D2KE, since we generate random samples from the distribution, we can use as many as needed to achieve performance close to an exact kernel.

We report the best number in the range R = [4, 4096] (typically the larger R is, the better the accuracy).

We employ a linear SVM implemented using LIBLINEAR (Fan et al., 2008) for all embedding-based methods (RSM, and D2KE) and use LIBSVM BID5 for precomputed dissimilairty kernels (DSK_RBF, DSK_ND, and KSVM).All datasets are collected from popular public websites for Machine Learning and Data Science research, including the UCI Machine Learning repository BID18 , the LibSVM Data Collection BID5 , and the Kaggle Datasets, except one time-series dataset IQ that is shared from researchers from George Mason University.

TAB5 lists the detailed properties of the datasets from four different domains.

All computations were carried out on a DELL dual-socket system with Intel Xeon processors at 2.93GHz for a total of 16 cores and 250 GB of memory, running the SUSE Linux operating system.

To accelerate the computation of all methods, we used multithreading with 12 threads total for various distance computations in all experiments.

C D E R T -S , S , IC.1 R -

For time-series data, we employed the most successful distance measure -DTW -for all methods.

For all datasets, a Gaussian distribution was found to be applicable, parameterized by its bandwidth σ.

The best values for σ and for the length of random time series were searched in the ranges [1e-3 1e3] and [2 50], respectively.

TAB1 , D2KE can consistently outperform or match all other baselines in terms of classification accuracy while requiring far less computation time for multivariate time-series.

The first interesting observation is that our method performs substantially better than KNN, often by a large margin, i.e., D2KE achieves 26.62% higher performance than KNN on IQ_radio.

This is because KNN is sensitive to the data noise common in real-world applications like IQ_radio, and has notoriously poor performance for high-dimensional data sets like Auslan.

Moreover, compared to the two distance substitution kernels DSK_RBF and DSK_ND, and KSVM operating directly on indefinite similarity matrix, our method can achieve much better performance, suggesting that a representation induced from a truly p.d.

kernel makes significantly better use of the data than indefinite kernels.

Among all methods, RSM is closest to our method in terms of practical construction of the feature matrix.

However, the random time series sampled by D2KE performs significantly better, as we discussed in section 4.

First, RSM simply chooses a subset of the original data points and computes the distances between the whole dataset and this representative set; this may suffer significantly from noise or redundant information in the time-series.

In contrast, our method samples a short random sequence that could both denoise and find the patterns in the data.

Second, the number of data points that can be sampled is limited by the total size of the data while the number of possible random sequences drawn from the distribution is unlimited, making the feature space much more abundant.

Third, RSM may incur significant computational cost for long time-series, due to its quadratic complexity in length.

Setup.

For string data, there are various well-known edit distances.

Here, we choose Levenshtein distance as our distance measure since it can capture global alignments of the underlying strings.

We first compute the alphabet from the original data and then uniformly sample characters from this alphabet to generate random strings.

We search for the best parameters for γ in the range [1e-5 1], and for the length of random strings in the range [2 50], respectively.

Results.

As shown in TAB2 , D2KE consistently performs better than or similarly to other distancebased baselines.

Unlike the previous experiments where DTW is not a distance metric, Levenshtein distance is indeed a distance metric; this helps improve the performance of our baselines.

However, D2KE still offers a clear advantage over baseline.

It is interesting to note that the performance of DSK_RBF is quite close to our method's, which may be due to DKS_RBF with Levenshtein distance producing a c.p.d.

kernel which can essentially be converted into a p.d.

kernel.

Notice that on relatively large datasets, our method, D2KE, can achieve better performance, and often with far less computation than other baselines with quadratic complexity in both number and length of data samples.

For instance, on mnist-str8 D2KE obtains higher accuracy with an order of magnitude less runtime compared to DSK_RBF and DSK_ND, and two orders of magnitude less than KSVM, due to higher computational costs both for kernel matrix construction and for eigendecomposition.

Setup.

For text data, following BID26 we use the earth mover's distance as our distance measure between two documents, since this distance has recently demonstrated a strong performance when combining with KNN for document classifications.

We first compute the Bag of Words for each document and represent each document as a histogram of word vectors, where google pretrained word vectors with dimension size 300 is used.

We generate random documents consisting of each random word vectors uniformly sampled from the unit sphere of the embedding vector space R 300 .

We search for the best parameters for γ in the range [1e-2 1e1], and for length of random document in the range [3 21].Results.

As shown in TAB3 , D2KE outperforms other baselines on all four datasets.

First of all, all distance based kernel methods perform better than KNN, illustrating the effectiveness of SVM over KNN on text data.

Interestingly, D2KE also performs significantly better than other baselines by a notiably margin, in large part because document classification mainly associates with "topic" learning where our random documents of short length may fit this task particularly well.

For the datasets with large number of documents and longer length of document, D2KE achieves about one order of magnitude speedup compared with other exact kernel/similarity methods, thanks to the use of random features in D2KE.

Setup.

For image data, following BID36 BID22 we use the modified Hausdorff distance (MHD) BID15 as our distance measure between images, since this distance has shown excellent performance in the literature BID44 BID19 .

We first applied the open-source OpenCV library to generate a sequence of SIFT-descriptors with dimension 128, then MHD to compute the distance between sets of SIFTdescriptors.

We generate random images of each SIFT-descriptor uniformly sampled from the unit sphere of the embedding vector space R 128 .

We search for the best parameters for γ in the range [1e-3 1e1], and for length of random SIFT-descriptor sequence in the range [3 15].Results.

As shown in TAB4 , D2KE performance outperforms or matches other baselines in all cases.

First, D2KE performs best in three cases while DSK_RBF is the best on dataset decor.

This may be because the underlying SIFT features are not good enough and thus random features is not effective to find the good patterns quickly in images.

Nevertheless, the quadratic complexity of DSK_RBF, DSK_ND, and KSVM in terms of both the number of images and the length of SIFT descriptor sequences makes it hard to scale to large data.

Interestingly, D2KE still performs much better than KNN and RSM, which again supports our claim that D2KE can be a strong alternative to KNN and RSM across applications.

@highlight

From Distance to Kernel and Embedding via Random Features For Structured Inputs
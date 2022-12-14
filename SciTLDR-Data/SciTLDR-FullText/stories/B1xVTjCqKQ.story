In this paper, we focus on two challenges which offset the promise of sparse signal representation, sensing, and recovery.

First, real-world signals can seldom be described as perfectly sparse vectors in a known basis, and traditionally used random measurement schemes are seldom optimal for sensing them.

Second, existing signal recovery algorithms are usually not fast enough to make them applicable to real-time problems.

In this paper, we address these two challenges by presenting a novel framework based on deep learning.

For the first challenge, we cast the problem of finding informative measurements by using a maximum likelihood (ML) formulation and show how we can build a data-driven dimensionality reduction protocol for sensing signals using convolutional architectures.

For the second challenge, we discuss and analyze a novel parallelization scheme and show it significantly speeds-up the signal recovery process.

We demonstrate the significant improvement our method obtains over competing methods through a series of experiments.

High-dimensional inverse problems and low-dimensional embeddings play a key role in a wide range of applications in machine learning and signal processing.

In inverse problems, the goal is to recover a signal X ∈ R N from a set of measurements Y = Φ(X) ∈ R M , where Φ is a linear or non-linear sensing operator.

A special case of this problem is compressive sensing (CS) which is a technique for efficiently acquiring and reconstructing a sparse signal BID12 BID6 BID1 .

In CS Φ ∈ R M ×N (M N ) is typically chosen to be a random matrix resulting in a random low-dimensional embedding of signals.

In addition, X is assumed to be sparse in some basis Γ, i.e., X = ΓS, where S 0 = K N .While sparse signal representation and recovery have made significant real-world impact in various fields over the past decade (Siemens, 2017) , arguably their promise has not been fully realized.

The reasons for this can be boiled down to two major challenges: First, real-world signals are only approximately sparse and hence, random/universal sensing matrices are sub-optimal measurement operators.

Second, many existing recovery algorithms, while provably statistically optimal, are slow to converge.

In this paper, we propose a new framework that simultaneously takes on both these challenges.

To tackle the first challenge, we formulate the learning of the dimensionality reduction (i.e., signal sensing operator) as a likelihood maximization problem; this problem is related to the Infomax principle BID24 asymptotically.

We then show that the simultaneous learning of dimensionality reduction and reconstruction function using this formulation gives a lower-bound of the objective functions that needs to be optimized in learning the dimensionality reduction.

This is similar in spirit to what Vincent et al. show for denoising autoencoders in the non-asymptotic setting BID38 .

Furthermore, we show that our framework can learn dimensionality reductions that preserve specific geometric properties.

As an example, we demonstrate how we can construct a data-driven near-isometric low-dimensional embedding that outperforms competing embedding algorithms like NuMax BID18 .

Towards tackling the second challenge, we introduce a parallelization (i.e., rearrangement) scheme that significantly speeds up the signal sensing and recovery process.

We show that our framework can outperform state-of-the-art signal recovery methods such as DAMP BID26 and LDAMP BID25 both in terms of inference performance and computational efficiency.

We now present a brief overview of prior work on embedding and signal recovery.

Beyond random matrices, there are other frameworks developed for deterministic construction of linear (or nonlinear) near-isometric embeddings BID18 BID16 BID0 BID35 BID39 BID5 BID37 BID32 .

However, these approaches are either computationally expensive, not generalizable to outof-sample data points, or perform poorly in terms of isometry.

Our framework for low-dimensional embedding shows outstanding performance on all these aspects with real datasets.

Algorithms for recovering signals from undersampled measurements can be categorized based on how they exploit prior knowledge of a signal distribution.

They could use hand-designed priors BID7 BID13 BID9 BID29 , combine hand-designed algorithms with data-driven priors BID25 BID3 BID20 BID8 BID17 , or take a purely data-driven approach BID28 BID22 BID41 .

As one moves from hand-designed approaches to data-driven approaches, models lose simplicity and generalizability while becoming more complex and more specifically tailored for a particular class of signals of interest.

Our framework for sensing and recovering sparse signals can be considered as a variant of a convolutional autoencoder where the encoder is linear and the decoder is nonlinear and specifically designed for CS application.

In addition, both encoder and decoder contain rearrangement layers which significantly speed up the signal sensing and recovery process, as we discuss later.

Convolutional autoencoder has been previously used for image compression ; however, our work is mainly focused on the CS application rather than image compression.

In CS, measurements are abstract and linear whereas in the image compression application measurements are a compressed version of the original image and are nonlinear.

Authors in have used bicubic interpolation for upscaling images; however, our framework uses a data-driven approach for upscaling measurements.

Finally, unlike the image compression application, when we deploy our framework for CS and during the test phase, we do not have high-resolution images beforehand.

In addition to image compression, there have been previous works BID34 BID22 to jointly learn the signal sensing and reconstruction algorithm in CS using convolutional networks.

However, the problem with these works is that they divide images into small blocks and recover each block separately.

This blocky reconstruction approach is unrealistic in applications such as medical imaging (e.g. MRI) where the measurement operator is a Fourier matrix and hence we cannot have blocky reconstruction.

Since both papers are designed for block-based recovery whereas our method senses/recovers images without subdivision, we have not compared against them.

Note that our method could be easily modified to learn near-optimal frequency bands for medical imaging applications.

In addition, BID34 and BID22 use an extra denoiser (e.g. BM3D, DCN) for denoising the final reconstruction while our framework does not use any extra denoiser and yet outperforms state-of-the-art results as we show later.

Beside using convolutional autoencoders, authors in BID40 have introduced the sparse recovery autoencoder (SRA).

In SRA, the encoder is a fully-connected layer while in this work, the encoder has a convolutional structure and is basically a circulant matrix.

For large-scale problems, learning a fully-connected layer (as in the SRA encoder) is significantly more challenging than learning convolutional layers (as in our encoder).

In SRA, the decoder is a T -step projected subgradient.

However, in this work, the decoder is several convolutional layers plus a rearranging layer.

It should also be noted that the optimization in SRA is solely over the measurement matrix and T (which is the number of layers in the decoder) scalar values.

However, here, the optimization is performed over convolution weights and biases that we have across different layers of our network.

In this section, we describe our framework for sparse signal representation and recovery and demonstrate how we can learn (near-)optimal projections and speed up signal recovery using parallelization along with convolutional layers.

We call our framework by DeepSSRR, which stands for Deep Sparse Signal Representation and Recovery.

DeepSSRR consists of two parts: A linear dimensionality reduction Φ : R N → R M for taking undersampled measurements and a nonlinear inverse mapping f Λ (.) : R M → R N for recovering signals from their undersampled measurements.

We learn both Φ and f Λ (.) from training data.

DeepSSRR FIG0 ) is based primarily on deep convolutional networks (DCN) as this gives us two advantages: (a) sparse connectivity of neurons, and (b) having shared weights which increases learning speed compared to fully-connected networks.

Therefore, we impose a convolutional network architecture on both Φ and f Λ (.) while learning them.

Please note that we assume that measurements are linear; however, it is easy to extend DeepSSRR to adopt nonlinear measurements, i.e., allowing for Φ(.) to be nonlinear by adding nonlinear units to convolutional layers.

Given that the intervening layers are linear, one might argue that one convolutional layer (i.e., a single circulant matrix) is enough since we can merge kernel matrices into a single matrix.

However, we consider a multi-layer architecture for learning Φ for two reasons.

First, computationally it is cheaper to have separate and smaller kernels and second, it makes the implementation suitable for adding the aforementioned nonlinearities.

We previously mentioned that in order to speed up the sensing and recovery process, we add a parallelization scheme in learning both Φ and f Λ (.) that we describe in the following.

Our original sensing model was Y = ΦX where X ∈ R N and Y ∈ R M .

Assume that the undersampling ratio, i.e., M N is equal to 1 r .

The left vector-matrix multiplication in FIG1 (a) denotes a convolution of zero-padded input signal with size N = rM = r(M + q − 1), filter size rq, stride (i.e., filter shift at every step) of size r, and output size of M .

If we denote the input signal by X (in) and output by X (out) and filter by W we can write DISPLAYFORM0 If we concatenate the sub-filters and sub-signals denoted in orange in the left vector-matrix multiplication of FIG1 (a), we derive a new vector-matrix multiplication shown on the right side of FIG1 (a).

There the input size is M = (M + q − 1), filter size is q, stride size is 1, and output size is M .

Equation FORMULA0 states that the left convolution in FIG1 (a) can be written as the summation of r separate and parallel convolutions shown on the right side.

Much like in the sensing part (i.e., learning Φ), as shown in FIG1 (b), a large strided deconvolution can be chopped into several parallel smaller deconvolutions for the recovery part (i.e., learning f Λ (.)).

Because of these parallelizations, the computational complexity of calculating the outputs of layers in DeepSSRR is O(M ) which is much less than the one for typical iterative and unrolled algorithms O(M N ) (e.g. DAMP and LDAMP BID26 ) or previous recovery algorithms based on deep learning O(N ) (e.g. DeepInverse ).

DISPLAYFORM1

Input: Training Dataset D, Number of Epochs n epochs , Network Parameters Ω e Output: A near-isometric embedding Φ : R N → R M for i = 1 to n epochs do -generate a randomly permuted training set → P(D) for every batch B j ∈ P(D)

do -compute embedding Φ(X) for every x ∈ B j -compute the loss function corresponding to B j as the maximum deviation from isometry DISPLAYFORM0 As DeepSSRR architecture is shown in FIG0 , For learning Φ, we first divide the input signal (of size N ) into r (r = N M ) sub-signals (of size M ) such that all the congruent entries (modulo r) are in the same sub-signal.

Then we run parallel convolutions on r sub-signals and stack the outputs (of size M ), deriving a tensor of length M and depth r. Through several convolutional layers, we turn this tensor into a vector of size M which is the measurements vector Y and this completes construction of Φ. Similarly and for learning f Λ (.), through several convolutional layers, we turn vector Y into a tensor of length M and depth r. We then unstack channels similar to the sub-pixel layer architecture BID33 and derive the final reconstruction X = f Λ (Y ) = f Λ (ΦX).

We use MSE as a loss function and ADAM BID21 to learn the convolution kernels and biases.

where M N .

Therefore, an important question is how does one design Φ?

Conventional CS is based on random projections of a signal which means that Φ is a random matrix in conventional CS.

However, since signals are usually structured, random projections are not optimal for successfully recovering the corresponding signals.

In many applications (e.g. medical imaging), we know a lot about the signals we are acquiring.

Hence, given a large-scale dataset of the same type of signals of interest, we can learn (near-)optimal measurement matrices.

As in the usual CS paradigm, if we assume that the measurement matrix Φ is fixed, each DISPLAYFORM0 } consists of pairs of signals and their corresponding measurements.

Accordingly, we define the optimal measurement operator Φ as the one which maximizes the probability of training data given the undersampled projections, Φ = arg max DISPLAYFORM1 According to the law of large numbers, notice that we can write DISPLAYFORM2 where in (a) I denotes the mutual information, and the equality follows since the Shannon entropy H(X) is constant for every Φ. According to (2), in the asymptotic setting, the measurement matrix which maximizes the probability of training data given its measurements, maximizes the mutual information between the input signal and undersampled measurements as well.

Equation FORMULA4 is the same as infomax principle first introduced in BID24 .Now, suppose that we have a function f (.) : DISPLAYFORM3 and reconstructs input signals as DISPLAYFORM4 We define the best reconstruction as the one which generates training data with the highest probability.

In other words, we define DISPLAYFORM5 Therefore, in the asymptotic setting and similar to (2) we can write DISPLAYFORM6 = arg max DISPLAYFORM7 In practice and since we do not know the true underlying probability distribution of P(X| X), we maximize a parametric distribution q(X| X) instead.

In this case, in the asymptotic setting we can write DISPLAYFORM8 = arg max DISPLAYFORM9 Therefore, since Kullback-Leibler divergence is bounded above zero we have DISPLAYFORM10 [log(P(X|Y = ΦX; Λ))], meaning that learning a parametric distribution for reconstructing X from Y is equivalent to maximizing a lower-bound of true conditional entropy and accordingly, mutual information between the input signal X and undersampled measurements Y .

Hence, although we are not maximizing the mutual information between X and Y , we are maximizing a lower-bound of it through learning Φ and Λ. If we assume X = X + , where and has an isotropic Gaussian distribution, then, since q(X| X = x) = N ( x, λI), the above maximization may be performed by minimizing the mean squared error (MSE).

DeepSSRR is mainly designed for jointly sensing and recovering sparse signals for CS applications.

However, we can specifically train the sensing part of DeepSSRR (without using the recovery part) for several important dimensionality reduction tasks.

The sensing part of DeepSSRR (i.e., the encoder or matrix Φ) is a linear low-dimensional embedding that we can apply it to learn a mapping from a subset of R N to R M (M < N ) that is a near-isometry, i.e., a mapping that nearly preserves all inter-point distances.

This problem has a range of applications, from approximate nearest neighbor search to the design of sensing matrices for CS.

Recall that, for a set Q ⊂ R N and > 0, the (linear or nonlinear) mapping Φ : Q → R M is an -isometry w.r.t the 2 -norm if for every x and x in Q we have DISPLAYFORM11 Algorithm 1 demonstrates the use of the low-dimensional embedding matrix Φ of DeepSSRR to construct a near-isometric embedding.

We achieve this by penalizing the maximum deviation from isometry in several batches of data that are created by permuting the original training data in every training epoch.

In Section 3 we will show how our proposed algorithm works compared to competing methods.

We now illustrate the performance of DeepSSRR against competing methods in several problems.

We first study the quality of the linear embeddings produced by DeepSSRR and its comparison with two other linear algorithms -NuMax BID18 and random Gaussian projections.

To show the price of linearity, we also pit these against the nonlinear version of DeepSSRR and a DCN (eight nonlinear convolutional layers + a max-pooling layer).

We use the grayscale version of CIFAR-10 dataset (50,000 training + 10,000 test 32 × 32 images).

We train DeepSSRR and DCN according to Algorithm 1 by using filters of size 5 × 5.

For DeepSSRR, depending on the size of the embedding we use five to seven layers to learn Φ in Algorithm 1.Figure 3(a) shows the size of embedding M as a function of the isometry constant for different methods.

For the random Gaussian projections we have considered 100 trials and the horizontal error bars represent the deviation from average value.

As we can see, the nonlinear version of DeepSSRR low-dimensional embedding outperforms almost all the other methods by achieving a given isometry constant with fewer measurements.

The only exception is when > 0.6 (i.e., a regime where we are not demanding a good isometry), where the DCN outperforms the nonlinear version of DeepSSRR; though, with more number of parameters.

A convolutional layer is equivalent to the product of a circulant matrix and the vectorized input.

The number of nonzero elements in a circulant matrix depends on the size of the convolution filter.

As the number of such layers grows, so does the number of nonzero elements in the final embedding matrix.

There are lower bounds BID30 on the number of nonzero elements in a matrix to ensure it is near-isometric.

TAB0 shows the isometry constant value of DeepSSRR's low-dimensional embedding with different number of layers and different filter sizes.

As we can see, gets smaller as the final embedding matrix has more nonzero elements (more layers, larger filters).Approximate Nearest Neighbors.

Finding the closest k points to a given query datapoint is challenging for high-dimensional datasets.

One solution is to create a near-isometric embedding that maps datapoints from R N to R M (M < N ) and solving the approximate nearest neighbors (ANN) problem in the embedded space.

FIG2 compares the performance of different methods in the ANN problem.

It shows the fraction of k-nearest neighbors that are retained when embedding datapoints in a low-dimensional space.

We have considered two separate embedding problems: First M = 65 for random embedding and NuMax and M = 64 for DCN and DeepSSRR's low-dimensional embedding.

Second, M = 289 for random embedding and NuMax and M = 256 for DCN and DeepSSRR's lowdimensional embedding.

Since the size of the embedding for DCN and DeepSSRR's low-dimensional embedding is smaller in both settings, they have a more challenging task to find the nearest neighbors.

As shown in FIG2 (b) DeepSSRR's low-dimensional embedding outperforms other approaches.

We divide the discussion of this section into two parts.

In the first part, we study the performance of DeepSSRR in the sparse signal recovery problem.

The discussion of this part along with experimental results showing the effect of learning a sparse representation and parallelization on different criteria (e.g. phase transition, recovery accuracy and speed) are provided in Appendix A. In the second part that we provide in the following, we study the performance of DeepSSRR for the compressive image recovery problem.

Compressive Image Recovery.

In this part, we study the compressive image recovery problem by comparing DeepSSRR with two state-of-the-art algorithms DAMP BID26 and LDAMP BID25 .

Both DAMP and LDAMP use random Gaussian Φ while DeepSSRR learns a Φ. Here we run DAMP for 10 iterations and use a BM3D denoiser at every iteration.

We also run LDAMP for 10 layers and use a 20-layer DCN in every layer as a denoiser.

For DeepSSRR, we use 7 layers to learn the Φ and 7 layers to learn the f Λ (.).

DeepSSRR is trained with an initial learning rate of 0.001 that is changed to 0.0001 when the validation error stops decreasing.

For training, we have used batches of 128 images of size 64 × 64 from ImageNet BID31 .

Our training and validation sets include 10,000 and 500 images, respectively.

On the other hand, DeepSSRR uses only 7 convolutional layers to recover the Man image which is significantly smaller compared to LDAMP's number of layers.

Iterative recovery algorithms and their unrolled versions such as DAMP and LDAMP typically involve a matrix vector multiplication in every iteration or layer, and hence their computational complexity is O(M N ).

In DeepSSRR, the length of feature maps in every convolutional layer is equal to the size of embedding M .

Therefore, computing the output of typical middle layers will cost O(M ) that is significantly cheaper than the one for iterative or unrolled methods such as DAMP and LDAMP.Effect of the Number of Layers.

Our experiments indicate that having more number of layers does not necessarily result in a better signal recovery performance.

This phenomenon is also observed in BID11 for the image super-resolution problem.

The reason for this problem is the increased non-convexity and non-smoothness of loss function as we add more layers.

One way to mitigate this problem is to add skip connections between layers.

As shown in , skip connections smooth the loss surface of deep networks and make the optimization problem simpler.

In this paper we introduced DeepSSRR, a framework that can learn both near-optimal sensing schemes, and fast signal recovery procedures.

Our findings set the stage for several directions for future exploration including the incorporation of adversarial training and its comparison with other methods BID2 BID14 BID10 ).

Furthermore, a major question arising from our work is quantifying the generalizability of a DeepSSRR-learned model based on the richness of training data.

We leave the exploration of this for future research.

In this section we study the problem of sparse signal recovery by comparing DeepSSRR to another DCN called DeepInverse and to the LASSO BID36 1 -solver implemented using the coordinate descent algorithm of BID15 .

We assume that the optimal regularization parameter of the LASSO is given by an oracle in order to obtain its best possible performance.

Also, both training and test sets are wavelet-sparsified versions of 1D signals of size N = 512 extracted from rows of CIFAR-10 images and contain 100,000 and 20,000 signals, respectively.

While DeepSSRR learns how to take undersampled measurements of data through its low-dimensional embedding Φ, DeepInverse uses random undersampling (i.e., a random Φ).

DeepSSRR in this section has 3 layers for learning Φ and 3 layers for learning f Λ (.) with filter size 25 × 1 while DeepInverse has five layers for learning the inverse mapping with filter size 125 × 1.Figure 5(a) shows the 1 phase transition plot BID13 ).

This plot associates each grid point to an ordered pair (δ, ρ) ∈ [0, 1] 2 , where δ = M N denotes the undersampling ratio and ρ = K M denotes the normalized sparsity level.

Each grid point (δ, ρ) represents the probability of an algorithm's success in signal recovery for that particular problem configuration.

As the name suggests, there is a sharp phase transition between values of (δ, ρ) where recovery fails with high probability to when it succeeds with high probability.

In FIG4 (a), the blue curve is the 1 phase transition curve.

The circular points denote the problem instances on which we study the performance of DeepInverse and the LASSO.

The square points denote the problem instances on which we have trained and tested DeepSSRR.

By design, all these problem instances are on the "failure" side of the 1 phase transition.

For DeepSSRR (square points), we have made recovery problems harder by reducing δ and increasing ρ.

The arrows between the square points and circular points in FIG4 (a) denote correspondence between problem instances in DeepSSRR and DeepInverse.

TAB1 shows the average normalized MSE (NMSE) for the test signals.

While DeepSSRR recovers the same signals from fewer measurements, it outperforms DeepInverse and the LASSO.

DeepSSRR outperforms DeepInverse while having significantly fewer number of parameters (less than 70,000 vs. approximately 200,000 parameters).

This is mainly due to the fact that DeepSSRR learns Φ instead of using a random Φ as is the case in DeepInverse and conventional CS.

While training and test sets are the same, the configuration for DeepInverse (and LASSO) is (δ, ρ) = (0.7, 0.72) and for DeepSSRR is (δ, ρ) = (0.5, 1.003) which means we have given DeepSSRR a more challenging problem.

As shown in FIG4 , due to the extra parallelization scheme (i.e., rearrangement layer) convergence is significantly faster for DeepSSRR compared to DeepInverse.

DeepSSRR outperforms the LASSO after only 4 training epochs while DeepInverse takes 138 epochs.

This fast convergence has two major reasons: First, DeepSSRR has fewer number of parameters to learn.

Second, DeepSSRR learns adaptive measurements (i.e., low-dimensional embedding) instead of using random measurements (i.e., random embedding).

≤ 0.01 , where X (j) is the j-th sample,X (j) is the recovered signal from measurements of j-th sample, and I(.) is the indicator function.

We denote empirical successful recovery probability by P δ = 1 q q j=1 ϕ δ,j .

In FIG4 , our test samples are k-sparse where k = 34, and we have considered three different configurations: M = 64, 128, 256 that correspond to above, on, and below the 1 phase transition, respectively.

As we can see in FIG4 , DeepSSRR significantly outperforms LASSO when the problem configuration lies above (failure phase) or on the 1 phase transition and LASSO slightly outperforms when the problem configuration lies below the 1 phase transition (success phase).

For a setting below the 1 phase transition, we expect 1 minimization to behave the same as 0 minimization.

However, DeepSSRR should learn a transformation for transforming measurements back to the original signals.

Furthermore, FIG4 shows the price we pay for using a linear low-dimensional embedding ΦX instead of a nonlinear one Φ(X).

The main message of FIG4 (c) is that by using DeepSSRR we can have a significantly better phase transition compared to 1 -minimization.

In this section we study another example of the compressive image recovery problem.

The settings we have used in here is exactly the same as Section 3.2.

FIG8 shows the reconstruction of the mandrill image ( M N = 0.25).

FIG8 shows the reconstruction of whole face and FIG8 shows the reconstruction of nose and cheeks.

As we can see, although LDAMP slightly outperforms our method in FIG8 , our method does a significantly better job in recovering the texture of nose and cheeks in FIG8 .

Not only our method outperforms LDAMP by 0.9 dB, but also it has a better visual quality and fewer artifacts (e.g. less over-smoothing).

In this section we compare the running time of different algorithms.

We consider the reconstruction of a 512 × 512 image with an undersampling ratio of M N = 0.25.

Table 3 shows the comparison between different algorithms.

We should note that authors in BID25 have used coded diffraction pattern in DAMP and LDAMP which simplifies the computational complexity of vector-matrix multiplications in DAMP and LDAMP to O(N log(N )) instead of O(M N ).

In addition, we should note that LDAMP uses filters of size 3 × 3 in its convolutional layers while we use filters of size 5 × 5 in the convolutional layers of our architecture.

Table 3 shows that our method is almost 4 times faster than the LDAMP method.

@highlight

We use deep learning techniques to solve the sparse signal representation and recovery problem.
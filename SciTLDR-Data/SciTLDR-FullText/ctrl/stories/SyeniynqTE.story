The linear transformations in converged deep networks show fast eigenvalue decay.

The distribution of eigenvalues looks like a Heavy-tail distribution, where the vast majority of eigenvalues is small, but not actually zero, and only a few spikes of large eigenvalues exist.

We use a stochastic approximator to generate histograms of eigenvalues.

This allows us to investigate layers with hundreds of thousands of dimensions.

We show how the distributions change over the course of image net training, converging to a similar heavy-tail spectrum across all intermediate layers.

The study of generalization in deep networks has shifted its focus from the skeleton structure of neural networks BID1 to the properties of the linear operators of the network layers BID3 BID4 BID12 .

Measures like matrix norms, including standard Frobenius, other p-Norms or spectral norm) or stable rank BID2 are important components of theoretical bounds on generalization.

All of these measures rely on the singular values of the linear maps A, or equivalently on the eigenvalues of the operator times its transpose AA T .

In order to visually inspect the eigenvalue spectrum of a matrix, it is useful to compute a histogram.

Histograms allows us to roughly estimate the distribution of eigenvalues and detect properties like decay behavior or the largest eigenvalues.

An example of such a histogram is FIG0 , that shows the eigenvalues of a Convolution-Layer in a squeeze_net network that maps to a feature map of dimension 256 × 13 × 13 = 43, 264.

It shows many interesting, but not wellunderstood characteristic properties of fully trained networks: A Heavy-tail eigenvalue distribution with the vast majority of eigenvalues being near zero, though none are actually zero, and only few spikes of large eigenvalues.

Martin and Mahoney show this phenomenon in the linear layer of large pre-trained models BID10 , we also show it in the convolution layers and follow its evolution over the course of optimization.

Computing singular values exactly is a costly operation, particularly because the dimension in stateof-the art convolutional neural networks often exceeds 100,000.

Fortunately when we are interested only in a histogram, we do not need to know the exact eigenvalues, but only the number of eigenvalues that fall into the bins of the histogram.

Based on the decay property exhibited in deep networks, we propose an approach for estimating histograms of eigenvalues in deep networks based on two techniques: For estimating the few high eigenvalues, called spikes, and particularly the largest eigenvalue, we use ARPACK, a truncated eigenvalue decomposition method that does not require the matrix explicitly, but accesses it only via matrix-vector products.

For estimating the remainder, called bulk, we use a method based on matrix Chebyshev approximations and Hutchinson's trace estimator BID11 .

Like ARPACK, it only accesses the matrix via matrix-vector products.

In FIG0 , we have colored the bins we computed exactly in red, the approximated are blue.

We denote the number of eigenvalues of a symmetric linear operator A : DISPLAYFORM0

The highest-dimensional linear operators in deep networks used in production environments are probably convolution layers.

These layers transform feature maps with number of raw-features in the input-and output feature often exceeding 100k.

This is feasible because the convolution operator is not implemented as matrix-vector multiplication with dense weight matrices, but specialized and highly-optimized convolution routines are used.

Really every reputable deep learning software framework provides these routines.

We can use these same routines when we estimate the eigenvalues of the linear maps of network layers.

We first make sure that the network layer does not add a bias term BID0 .

Now let H be the linear map of a neural network layer.

The forward-pass of that layer computes Hx efficiently, whereas the backward-pass computes H T y with backward-flowing gradient information y.

We are interested in the eigenvalues of HH T , hence to compute HH T y, we first pass y through the backward pass of H, and pass the resulting gradient through the forward pass to obtain the resulting vector.

To estimate the spikes and particularly the largest eigenvalue, we use the implicitly restarted Lanczos method as implemented in the ARPACK software package.

It computes a truncated eigenvalue decomposition for implicit matrices that are accessed via matrix-vector products BID9 .

We specify a number of spikes T > 0 and compute the first T eigenvalues with ARPACK.

From the largest eigenvalue, we derive the equidistant histogram-binning over the range [0, λ 1 ].

We use a technique for stochastically estimating eigenvalue counts proposed by Napoli et al. BID11 .

It requires that all eigenvalues fall into the range [−1, 1].

Hence, we first transform the matrix via A → (2λ −1 1 A − I) since we already know λ 1 from the ARPACK-based spike estimator.

We define the indicator function δ l,u (λ) that is 1 iff.

l ≤ λ < u and notice that we can write the number of eigenvalues in [l, u) as DISPLAYFORM0 We can approximate δ l,u with Chebyshev polynomials of a fixed degree DISPLAYFORM1 is the kth Chebyshev basis and b k ∈ R its corresponding coefficient.

These coefficients are DISPLAYFORM2 Compute coefficients b k for k = 0, ..., K according to BID0 .

DISPLAYFORM3 known for the indicator function BID11 : DISPLAYFORM4 Now we can rewrite the count as the trace of a polynomial matrix function 2 applied to our matrix of interest, as it holds that trf DISPLAYFORM5 where Φ k (A) is the kth Chebyshev base for matrix functions.

This quantity in turn can be approximated using stochastic trace estimators, aka Hutchinson estimators BID7 .

It holds that trA = E x x T Ax where each component of x is drawn independently from a zero-mean distribution like standard normal or Rademacher.

This expression lends itself to a simple sampling algorithm, where we draw S independent x 1 , ..., x S and estimate DISPLAYFORM6 We do not have to explicitly compute Φ k (A), as only the product Φ k (A)x i is required.

Since Chebyshev polynomials by construction follow the recursion Φ k+1 (A) = 2AΦ k (A) − Φ k−1 (A), we derive Algorithm 1 to estimate the count.

Our experiments are based on a pyTorch implementation of the proposed histogram estimator.

We train a squeeze_net architecure BID8 on imagenet data.

After 30 epochs of training, we reduce the learning rate to 10%, and repeat this after another 30 epochs.

We train using plain stochastic gradient descent with mini-batches of size 128 and compute histograms for all convolution layers before the first and after every epoch.

For the histogram computation, we use a budget of 1000 for the exact computation of eigenvalues and approximate the remainder using the stochastic estimator.

We present some histograms in FIG2 for the first and last convolution layers BID2 .

The histograms of the other layers show similar behavior as the last layer, for instance FIG0 shows an intermediate layer after the first epoch of training quite similar to FIG2 , but with less extreme decay.

BID1 A real-valued function f (x) : R → R has a corresponding matrix function f (A) : R m×m → R m×m and the eigenvalues of f (A) are f (λ1), ..., f (λm).

For polynomials, we get this matrix function by replacing scalar multiplications with matrix multiplications and scalar additions with matrix additions.

For other classes of functions and a comprehensive introduction to matrix functions see Highham's book BID6 .

BID2 Addional and animated histograms are available at https://whadup.github.io/Resultate/, however note that the website is not sufficiently anonymized for double-blind reviewing.

Proceed with caution.

Like Martin and Mahoney BID10 we identify different phases in the spectograms.

Right after initialization, the matrix behaves almost as random matrix theory suggests given the element-wise independent random initialization with Gaussian random variables BID13 .

This can be observed in FIG2 .

However, we note that on the first layer Fig. 3a , there are some unexpected bumps in the histogram.

We conjecture that this may be due to padding in the convolutions.

As optimization commences, we start to see heavytail behavior.

Already after one epoch of traning, the largest eigenvalues have seperated from the bulk, while the majority of eigenvalues remains in the same order of magnitude as before training.

This can be seen for the first and large convolution layer in FIG2 .

The bumps in the first layer smooth out a little.

Over the course of the first 30 epochs, the largest eigenvalues grow steadily as the tail of the spectrum grows further.

Then as soon as the learning rate is reduced to 10%, the operator norms of the linear maps start to decrease as depicted in FIG1 .

Considering the importance of the operator norm in known generalization bounds for feed-forward networks, this suggests that some sort of regularization is happening.

The bumps in the first layer smooth out further, but remain visible.

The last layer for the most part keeps its shape in the last 60 epochs, beside the reduction of the norm we notice that the largest bar decreases in size from 4157 to 2853 and that the difference seems to move to the other bars in the blue portion of the histogram.

Understanding the structure in the linear transformations might be an important aspect of understanding generalization in deep networks.

To this end we have presented a stochastic approach that allows us to estimate the eigenvalue spectrum of these transformations.

We show how the spectrum evolves during imagenet training using convolutional networks, more specifically squeeze_net networks.

In the future we want to apply similar approaches to estimating the covariance structure of the intermediate feature representations and investigate the relations between covariance matrices and parameter matrices.

Since the estimator we use is differentiable BID5 BID0 , it may be interesting to investigate its usefulness for regularization.

<|TLDR|>

@highlight

We investigate the eigenvalues of the linear layers in deep networks and show that the distributions develop heavy-tail behavior during training.
Limited angle CT reconstruction is an under-determined linear inverse problem that requires appropriate regularization techniques to be solved.

In this work we study how pre-trained generative adversarial networks (GANs) can be used to clean noisy, highly artifact laden reconstructions from conventional techniques, by effectively projecting onto the inferred image manifold.

In particular, we use a robust version of the popularly used GAN prior for inverse problems, based on a recent technique called corruption mimicking, that significantly improves the reconstruction quality.

The proposed approach operates in the image space directly, as a result of which it does not need to be trained or require access to the measurement model, is scanner agnostic, and can work over a wide range of sensing scenarios.

Computed Tomography (CT) reconstruction is the process of recovering the structure and density of objects from a series of x-ray projections, called sinograms.

While traditional full-view CT is relatively easier to solve, the problem becomes under-determined in two crucial scenarios often encountered in practice -(a) few-view: when the number of available x-ray projections is very small, and (b) limited-angle: when the total angular range is less than 180 degrees, as a result of which most of the object of interest is invisible to the scanner.

These scenarios arise in applications which require the control of x-ray dosage to human subjects, limiting the cost by using fewer sensors, or handling structural limitations that restrict how an object can be scanned.

When such constraints are not extreme, suitable regularization schemes can help produce artifact-free reconstructions.

While the design of such regularization schemes are typically driven by priors from the application domain, they are found to be insufficient in practice under both few-view and limited-angle settings.

In the recent years, there is a surge in research interest to utilize deep learning approaches for challenging inverse problems, including CT reconstruction [1, 2, 3] .

These networks implicitly learn to model the manifold of CT images, hence resulting in higher fidelity reconstruction, when compared to traditional methods such as Filtered Backprojection (FBP), or Regularized Least Squares (RLS), for the same number of measurements.

While these continue to open new opportunities in CT reconstruction, they rely of directly inferring mappings between sinograms and the corresponding CT images, in lieu of regularized optimization strategies.

However, the statistics of sinogram data can vary significantly across different scanner types, thus rendering reconstruction networks trained on one scanner ineffective for others.

Furthermore, in practice, the access to the sinogram data for a scanner could be restricted in the first place.

This naturally calls for entirely image-domain methods that do not require access to the underlying measurements.

In this work, we focus on the limited-angle scenario, which is known to be very challenging due to missing information.

Instead of requiring sinograms or scanner-specific representations, we pursue an alternate solution that is able to directly work in the image domain, with no pairwise (sinogram-image) training necessary.

To this end, we advocate the use of generative adversarial networks (GANs) [4] as image manifold priors.

GANs have emerged as a powerful, unsupervised technique to parameterize high dimensional image distributions, allowing us to sample from these spaces to produce very realistic looking images.

We train the GAN to capture the space of all possible reconstructions using a training set of clean CT images.

Next, we use an initial seed reconstruction using an existing technique such as Filtered Back Projection (FBP) or Regularized Least Squares (RLS) and 'clean' it by projecting it onto the image manifold, which we refer to as the GAN prior following [6] .

Since the final reconstruction is always forced to be from the manifold, it is expected to be artifact-free.

More specifically, this process involves sampling from the latent space of the GAN, in order to find an image that resembles the seed image.

Though this has been conventionally carried out using projected gradient descent (PGD) [5, 6 ], as we demonstrate in our results, this approach performs poorly when the initial estimate is too noisy or has too many artifacts, which is common under extremely limited angle scenarios.

Instead, our approach utilizes a recently proposed technique referred to as corruption mimicking, used in the design of MimicGAN [7] , that achieves robustness to the noisy seed reconstruction through the use of a randomly initialized shallow convolutional neural network (CNN), in addition to PGD.

By modeling the initial guess of this network as a random corruption for the unknown clean image, the process of corruption mimicking alternates between estimating the unknown corruption and finding the clean solution, and this alternating optimization is repeated until convergence, in terms of effectively matching the observed noisy data.

The resulting algorithm is test time only, and can operate in an artifact-agnostic manner, i.e. it can clean images that arise from a large class of distortions like those obtained from various limited-angle reconstructions.

Furthermore, it reduces to the well-known PGD style of projection, when the CNN is replaced by an identity function.

We restrict our study to parallel beam and fan beam types of scanners, that produces a CT reconstruction in a 2D slice-by-slice manner.

The CT reconstruction problem, like most other inverse problems, can be written as: X * = arg min X A(X) − y + R(X), where X ∈ R d×d is the image to be reconstructed, y ∈ R v×d is the projection, referred to as a "sinogram", and A is the x-ray projection operator of the particular CT scanner.

Here, the number of available x-ray projections is given by v, and the number of detector columns is given by d. Note, A(X) can be written as a matrix multiplication, but the matrix tends to be a sparse, very large matrix.

Here, for simplicity we denote it as an operator acting on X. Typically, a regularization function in the form of R(X) is used to further reduce the space of possible solutions.

In order to get a complete faithful reconstruction of X, the object must be scanned from a full 180

• .

When the viewing angle is much lesser than << 180

• , most existing methods return an X * that is extremely corrupted by noise and missing edges, with little or no information of the original structure present.

While several kinds of regularization functions have been used (for e.g. total variation and its variants), in this paper we advocate the use of R(X) such that it forces X to be from a known image manifold.

We achieve this by using generative adversarial networks (GANs) [4] , which have emerged as a powerful way to represent image manifolds.

In particular at test time, given a sinogram y, the problem can be formulated as

A(G(z)) − y , where G is a pre-trained generator,

and finally, X * = G(z * ) and can be solved using stochastic gradient descent.

This has been referred to as a GAN prior [6] or a manifold prior [3] for inverse imaging.

However, solving the equation of the form in (1) is not always possible since one may not have access to the measurement model A. A more accessible (yet different) form that does not require A to be known is given by:

In this work we obtain the initial estimate X RLS using regularized least squares (RLS) approach.

As a result of (2), the quality of the final estimate largely depends on the quality of the initial reconstruction.

Particularly, if the estimate is very noisy or poor, as is the case for limited angle CT, the optimization in (2) can easily fail, especially when the loss is not robust to the type of corruption noise or distortion.

In scenarios of interest in this paper, even a powerful regularizer such as the GAN prior can fail due to a poor initial estimate.

In order to avoid this, we propose to use a recently proposed modification of the GAN prior, that performs better even with heavily distorted images.

The process called corruption mimicking was proposed in [7] , was designed to improve the quality of projection onto the manifold under a variety of corruptions.

Corruption Mimicking and the Robust GAN Prior: Let us suppose X RLS = f (X * ), where f is an unknown distortion or corruption function, and X * is the unknown global optima to (2) .

Corruption Mimicking is the process of estimating both X * and f simultaneously, using a shallow neural network to approximate f with a few examples.

As a result, we now modify (2) as follows:

Equation (3) is solved using alternating optimization, where we first solve for the optimalf * conditioned on the current estimate X * , and repeat the process until convergence.

Since we constrain f to be shallow, even as few as 100 samples are sufficient.

In our setting,f contains 2 convolutional layers with ReLU activations, followed by a masking layer (pixel-wise multiplication).

Finally, we also include a shortcut connection at the end to encourage it to learn identity [7] .

The GAN prior now becomes a special case of the Robust GAN prior, whenf = I, the identity function.

An appealing property of this technique is that it is corruption-agnostic i.e., the same system can be reused to obtain accurate CT reconstructions across a wide variety of limited-angle settings.

We test the effectiveness of the robust GAN prior by performing CT reconstruction of the MNIST [8] and Fashion-MNIST [9] datasets.

We first project these datasets into their projection space (sinograms), using a forward projection operation, to simulate the CT-scan process.

While we consider a parallel beam scanner in these experiments, the methods and reported observations are applicable to other scanner types, since the proposed method operates directly in the image space.

Next, we recover the images using the regularized least squares algorithm (RLS), which is commonly adopted in CT reconstruction.

We emulate the limited-angle scenario by providing only a partial sinogram to RLS.

We provide the resulting reconstruction as the input to the proposed algorithm.

Experimental Settings:

On both datasets, we train a standard DCGAN [10] to generate images using the 60K training 28 × 28 images.

We run all our reconstruction experiments on a subset of the 10K validation set.

Corruption-mimicking requires choosing 4 main hyperparameters [7] : T 1 = 15, T 2 = 15, γ s = 1e − 2, γ g = 8e − 2 that control the number of iterations in the alternating optimization and learning rates (see section 2 for details), these are kept fixed on both datasets, across all viewing angle settings.

We observed the performance to be robust across a wide range of settings for these hyper-parameters.

Finally, we compare the performance of the robust GAN prior against the standard GAN prior, without corruption-mimicking.

In both cases, we run the latent space optimization for a total of ∼ 2500 iterations, which typically only takes about 10 seconds on a P100 NVIDIA GPU.

In figures 1, 2, we show qualitative and quantitative results obtained for both the MNIST and Fashion-MNIST datasets respectively.

In both cases, we demonstrate significant improvements in recovering the true reconstruction compared to the vanilla GAN prior.

It should be noted that a performance boost of nearly 4-5 dB on MNIST and 0.5-1dB on Fashion-MNIST are achieved with no additional information or data, but due to the inclusion of the robust GAN prior.

Additionally, PSNR and SSIM tend to be uncorrelated with perceptual metrics in many cases, as perceptually poor reconstructions can be deceptively close in PSNR or SSIM.

A potential fix in GAN-based reconstruction approaches is to compute error in the discriminator feature space as a proxy for perceptual quality.

[8] : Given the RLS reconstruction, we improve them by projecting onto the image manifold using corruption mimicking [7] .

In all cases, we show the improvement obtained by using the robust GAN prior over a standard GAN projection.

<|TLDR|>

@highlight

We show that robust GAN priors work better than GAN priors for limited angle CT reconstruction which is a highly under-determined inverse problem.
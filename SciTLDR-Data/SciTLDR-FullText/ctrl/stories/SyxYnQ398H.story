Recent advances have illustrated that it is often possible to learn to solve linear inverse problems in imaging using training data that can outperform more traditional regularized least squares solutions.

Along these lines, we present some extensions of the Neumann network, a recently introduced end-to-end learned architecture inspired by a truncated Neumann series expansion of the solution map to a regularized least squares problem.

Here we summarize the Neumann network approach, and show that it has a form compatible with the optimal reconstruction function for a given inverse problem.

We also investigate an extension of the Neumann network that incorporates a more sample efficient patch-based regularization approach.

We consider solving linear inverse problems in imaging in which a p-pixel image, β ∈ R p (in vectorized form), is observed via m noisy linear projections as y = Xβ + , where X ∈ R m×p and ∈ R m is a noise vector.

The problem of estimating β from y is referred to as image reconstruction, and a typical estimate is given by solving the regualarized least squares problem β = arg min

where r(·) is a regularizer.

Classical image reconstruction methods specify a choice of regularizer to promote piecewise smoothness of the reconstruction, sparsity in some dictionary or basis, or other geometric properties.

However, an emerging body of research explores the idea that training data can be used to learn to solve inverse problems using neural networks.

At a high level, existing learning-based approaches to solving inverse problems can be categorized as either decoupled or end-to-end.

Decoupled approaches first learn a representation of the data that is independent of the forward model X, followed by a reconstruction phase that uses X explicitly.

Existing methods in this vein include using training images to learn a low-dimensional image manifold captured by the range of a generative adversarial network (GAN) and constraining the estimate β to lie on this manifold [1] , or learning a denoising autoencoder that can be treated as a regularization step (i.e., proximal operator) within an iterative reconstruction scheme [2, 3] .

End-to-end approaches incorporate the forward model X directly into the network architecture during both training and testing, and are optimized for a specific X or class of X's.

Many end-to-end approaches are based on "unrolling" finitely many iterations of an optimization algorithm for solving (1) , where instances of the regularizer (or its gradient or proximal operator) are replaced by a neural network to be trained; see [4, 5, 6, 7, 8] among others.

The advantage of a decoupled approach is that the learned representation can be used for a wide variety of inverse problems without having to retrain.

However, this flexibility comes with a high price in terms of sample complexity.

Learning a generative model or a denoising autoencoder fundamentally amounts to estimating a probability distribution and its support over the space of images; let us denote this distribution as P (β).

On the other hand, if X is known at training time, then we only need to learn the conditional distribution P (β|Xβ), which can require far fewer samples to estimate [9] .

To make this idea more precise, consider the problem of finding the MSE optimal reconstruction function in the noiseless setting:

Then ρ * is characterized as follows.

Proposition 1.

Let X ∈ R m×p , m ≤ p, be full rank, and let X ⊥ ∈ R p−m×p be a matrix whose rows form an orthonormal basis for the nullspace of X. Then the MSE-optimal reconstruction function ρ * in (2) is given by

where X + is the pseudoinverse of X and

We omit the proof of Proposition 1 for brevity, but the technique is similar to those used in [10] to derive the expressions for the MSE optimal autoencoder for a given data distribution.

This proposition shows that the optimal reconstruction function only requires estimating a conditional expectation of the component of the image in the nullspace of the linear forward model, or implicitly a conditional probability density rather than the full probability density over the space of all images.

Therefore, in settings where training data is limited, end-to-end approaches are expected to outperform decoupled approaches due to their lower sample complexity.

It also implies the end-to-end networks should have a structure compatible with (2) if they are to well-approximate the MSE optimal reconstruction function.

The focus of this work is on the recently-proposed Neumann network architecture as an end-to-end approach for learning to solve inverse problems [11] .

Here we summarize the Neumann network architecture, and give it a new interpretation in light of Proposition 1.

We also introduce an extension of Neumann networks to the case of a patch-based regularization strategy, which further improves the sample complexity of the approach.

Neumann networks are motivated by the regularized least squares optimization problem (1) in the special case where the regularizer r is quadratic.

In particular, assume r(β) = 1 2 β Rβ so that ∇r(β) = Rβ for some matrix R ∈ R p×p .

A necessary condition for β to be a minimizer of (1) in this case is (X X + R)β = X y (4) If the matrix on the left-hand side of (4) is invertible, the solution is given by

To approximate the matrix inverse in (5), the authors of [11] use a Neumann series expansion for the inverse of a linear operator [12] , given by A −1 = η ∞ k=0 (I − ηA) k , which converges provided I − ηA < 1.

Applying this series expansion to the matrix inverse appearing in (5), we have β = ∞ j=0 (I − ηX X − ηR) j (ηX y).

Truncating this series to B + 1 terms, and replacing multiplication by the matrix R with a general learnable mapping R : R p → R p , motivates an estimator β of the form

The estimator above becomes trainable by letting R = R θ be a neural network depending on a vector of parameters θ ∈ R q to be learned from training data, along with the scale parameter η.

Neumann Net Figure 1 : Neumann network architecture.

Unlike other networks based on unrolling of iterative optimization algorithms, the series structure of Neumann networks lead naturally to additional "skip connections" (highlighted in red) that route the output of each dashed block to directly to the output layer.

Any estimator β(y) = β(y; θ, η) specified (6) with trainable network R = R θ is called a Neumann network in [11] .

Figure 1 shows a block diagram which graphically illustrates a Neumann network.

The main architectural difference with Neumann networks over related unrolling approaches is the presence of additional "skip connections" that arise naturally due to the series structure.

Empirical evidence in [11] suggests these additional skip connections may improve the optimization landscape relative to other architectures, and make Neumann networks easier to train.

Efficiently finding a solution to the linear system (4) using an iterative method can be challenging when the matrix X X + R is ill-conditioned.

This suggests that the Neumann network, which is derived from a Neumann series expansion of the system in (4), may benefit from preconditioning.

Starting from (4), for any λ > 0 we have (X X + λI)β + (R − λI)β = X y. Applying T λ := (X X + λI) −1 to both sides and rearranging terms gives (I − λT λ +R)β = T λ X y.

Following the same steps used to derive the Neumann network gives the modified estimator

which is called a preconditioned Neumann network in [11] .

HereR =R θ is a trainable mapping depending on parameters θ.

The Neumann network estimators in (6) and (7) can be interpreted as approximating the MSE optimal reconstruction function in Proposition 1.

To see this, observe that the pseudo-inverse X + y = (X X) −1 X y is given by the Neumann series

The preconditioned Neumann network estimator β(y) has the form

where β R (y) collects all terms that depend on R. The preconditioned Neumann network more directly approximates ρ * (y) since the initial iterateβ (0) = T λ X y = (X X + λI) −1 X y already well-approximates X + y provided λ > 0 is small.

Here we present an extension to the Neumann network which incorporates a learned patchwise regularizer.

For large images, learning an accurate regularizer may require more samples than are practical to gather due to cost or time constraints, leading to inaccurate reconstructions or overfitting.

However, empirical evidence suggests there is considerable low-rank and other subspace structure shared among small patches of natural images [13] .

Redundancy and subspace structure across image patches permits learning parameters of statistical models for image patches using training data, like Gaussian mixture models with low-rank covariance structure [14, 15] .

We propose leveraging the highly structured nature of image patches in the learned component of the Neumann network.

Specifically, the patchwise learned regularizer first divides the input image into overlapping patches, subtracting the mean from each patch (a standard preprocessing technique in patch-based methods [16] ), and passing each mean-subtracted patch through the learned component (e.g., neural network).

The original patch means are added to the regularizer outputs, which are recombined.

Figure 2 compares the presented learning-based methods at different training set sizes.

Methods that do not incorporate the forward model, like ResAuto and CSGM, appear not to perform well in the low-sample regime.

We also demonstrate that patchwise regularization enables reconstruction of large images with very small training sets.

In this experiment, the training set consists only of a single clean image, taken from the SpaceNet dataset [17] .Test PSNR is 31.90 ± 1.42 dB for the 8x8 patchwise regularized NN, and 18.34 ± 1.31 for the full-image regularized NN across a test set of size 64.

Fig. 3 contains some sample reconstructions of an image from the test set.

This work explores the Neumann network architecture to solve linear inverse problems, which can be interpreted as an approximation of the MSE optimal reconstruction according to our Proposition 1.

The Neumann network architecture also permits a learned patchwise regularizer, which learns the low-dimensional conditional distributions over image patches instead of the whole image.

The Neumann network is empirically competitive with other state-of-the-art methods for inverse problems in imaging, and we demonstrate the ability to learn to regularize from a single training pair.

<|TLDR|>

@highlight

Neumann networks are an end-to-end, sample-efficient learning approach to solving linear inverse problems in imaging that are compatible with the MSE optimal approach and admit an extension to patch-based learning.
Basis pursuit is a compressed sensing optimization in which the l1-norm is minimized subject to model error constraints.

Here we use a deep neural network prior instead of l1-regularization.

Using known noise statistics, we jointly learn the prior and reconstruct images without access to ground-truth data.

During training, we use alternating minimization across an unrolled iterative network and jointly solve for the neural network weights and training set image reconstructions.

At inference, we fix the weights and pass the measurements through the network.

We compare reconstruction performance between unsupervised and supervised (i.e. with ground-truth) methods.

We hypothesize this technique could be used to learn reconstruction when ground-truth data are unavailable, such as in high-resolution dynamic MRI.

Deep learning in tandem with model-based iterative optimization [2] - [6] , i.e. model-based deep learning, has shown great promise at solving imaging-based inverse problems beyond the capabilities of compressed sensing [7] .

These networks typically require hundreds to thousands of examples for training, consisting of pairs of corrupted measurements and the desired ground-truth image.

The reconstruction is then trained in an end-to-end fashion, in which data are reconstructed with the network and compared to the ground-truth result.

in many cases, collecting a large set of fully sampled data for training is expensive, impractical, or impossible.

In this work, we present an approach to model-based deep learning without access to ground-truth data [8] - [10] .

We take advantage of (known) noise statistics for each training example and formulate the problem as an extension of basis pursuit denoising [11] with a deep convolutional neural network (CNN) prior in place of image sparsity.

During training, we jointly solve for the CNN weights and the reconstructed training set images.

At inference time, we fix the weights and pass the measured data through the network.

As proof of principle, we apply the technique to undersampled, multi-channel magnetic resonance imaging (MRI).

We compare our Deep Basis Pursuit (DBP) formulation with and without supervised learning, as well as to MoDL [6] , a recently proposed unrolled modelbased network that uses ground-truth data for training.

We show that in the unsupervised setting, we are able to approach the image reconstruction quality of supervised learning, thus opening the door to applications where collecting fully sampled data is not possible.

We focus on the discretized linear signal model under additive white Gaussian noise:

where x ??? C N is the vectorized unknown image, A ??? C M ??N is the discretized forward model describing the imaging system, y ??? C M is a vector of the acquired measurements, and v ??? N c 0, ?? 2 I is a complex-valued Gaussian noise vector.

We are interested in the ill-posed regime, where M < N .

To make the inverse problem well-posed, x is commonly solved through a regularized least-squares:

where Q(x) is a suitable regularization term, and ?? > 0 is the strength of the regularization.

An alternative, equivalent formulation that directly accounts for the model error due to noise is the constrained problem:

where = ?? ??? M is the square-root of the expected noise power in the measurements.

When an 1 -norm is used for regularization, this is known as basis pursuit denoising [11] , and provides an intuitive formulation as it finds the best (sparsist) representation given a noise error constraint.

CNNs have recently been used to solve imaging inverse problems, relying on the network architecture and training data to learn the inverse mapping.

When a large corpus of training data is available, it is possible to learn the inverse mapping directly from under-sampled measurements, typically by first transforming the measurements to the image domain either through the adjoint operation A * y or through a conventional reconstruction.

Except for the initial transformation, these models do not take advantage of knowledge of the imaging system in the network architecture.

Thus, they require substantial training data and are prone to overfitting and CNN artifacts [12] .

More recently, network architectures that combine both CNN blocks and data consistency blocks incorporating knowledge of the forward model have grown in popularity, as they allow for robustness against CNN artifacts and training with limited data [5] , [6] .

These architectures are inspired by conventional first-order iterative algorithms intended to solve the unconstrained problem (2) , and typically alternate between data consistency and manifold projection.

To facilitate training with backpropagation, the iterative algorithms are unrolled for a finite number of steps and optimized in an end-to-end manner.

As the network is differentiable, gradient updates can be computed through the application of the forward operator with auto-differentiation.

For a particular network architecture, we can view the image reconstruction as a feed-forward network

where F w is a deep network parameterized by weights w that operates on the measurements and optionally incorporates knowledge of the forward model.

Given a training set of inputs {y

and corresponding ground-truth images {x

, the network weights can be trained in a traditional end-to-end fashion by minimizing the average training loss as measured by the loss function L:

For inference, the weights are fixed and new measurements are reconstructed through a forward pass of (4).

Inspired by other model-based deep learning architectures [2] - [6] , we propose a new unrolled network based on basis pursuit denoising, which we call Deep Basis Pursuit (DBP).

We assume the noise statistics of the measurements are known and we use them to selfregularize the solution.

In turn, we propose to train in an unsupervised fashion in the measurement domain, taking advantage of explicit control of the error between the measurements and the output of the network.

We first describe the DBP model, and then discuss training the in an unsupervised fashion without ground-truth data.

We combine the data consistency constraint of basis pursuit denoising (3) with the 2 -norm incorporating a CNN auto-encoder.

The DBP optimization is given by arg min

where N w (x) ??? x ??? R w (x) is a CNN parameterized by weights w that aims to estimate noise and aliasing [4] , [6] .

In other words, R w (x) represents a denoised version of x. In this way, we seek to find the "cleanest" representation of x while allowing for the expected data inconsistency due to noise.

To approximately solve (6), we consider an alternating minimization [6] , repeated N 1 times:

x k = arg min

Subproblem (7) is a forward pass through the CNN.

Subproblem (8) is convex and can solved with ADMM [13] .

We introduce the slack variable z = Ax and the dual variable u, and apply the following update steps, repeated N 2 times:

where ?? > 0 is the ADMM penalty parameter and L2Proj(z, ) is the projection of z onto the 2 -ball of radius .

The update steps are amenable to matrix-free optimization, as the forward and adjoint calculations can be represented as computationally efficient operators.

In particular, subproblem (9) can be approximately solved with N 3 iterations of the Conjugate Gradient Method.

Altogether, we can view DBP as an unrolled optimization alternating between CNN layers and data consistency layers, as shown in Fig. 1 .

At each layer, the same CNN is used, though it is possible in general to relax this requirement [4] .

For a fixed CNN R w , the DBP model is a special case of (4):xw ??? Fw(y; A, ), wher??? w = (w, ??) are the network parameters, and the network uses measurements together with knowledge of the system and noise power to return an estimate of the image.

and ground-truth

training data are available, the network weights can be trained in a traditional end-to-end fashion according to (5) .

When ground-truth data are not available, we consider a loss functionL imposed in the measurement domain:

The measurement loss can be a useful surrogate for the true loss, as the measurements contain (noisy) information about the ground-truth image [9] , [14] .

Thus, we may hope to learn about the image statistics given a large-enough training set that includes a diversity of measurements.

We consider the application to under-sampled, multichannel MRI.

The MRI reconstruction task is well-suited to DBP, as the noise statistics are Gaussian and can be measured during a short pre-scan.

We first describe the multi-channel MRI forward operator and general sampling strategy.

Then we discuss the experimental setup, including the dataset and implementation details.

In multi-channel MRI, the signal is measured by an array of receive coils distributed around an object, each with a spatially-varying sensitivity profile.

In the measurement model, the image is linearly mixed with each coil sensitivity profile, Fourier transformed, and sampled.

We can describe the measurement model as A = (P F S 1 ) ?? ?? ?? (P F S C ) ??? C M ??N , where C is the number of receive coils, S c ??? C N ??N is a diagonal operator containing the spatial sensitivity profile of the c th coil along the diagonal, F is the Fourier transform operator, and P ??? {0, 1} M C ??N is a diagonal operator that selects the sampled frequencies.

Data: We used the "Stanford Fully Sampled 3D FSE Knees" dataset from mridata.org, containing 3D Cartesian proton-density knee scans of 20 healthy volunteers.

Each 3D volume consisted of 320 slices with matrix size 320??256 and was scanned with an 8-channel receive coil array.

Although each slice is fully sampled, in practice the "ground-truth" data itself has noise.

To aid in experimental comparison, "noise-free" ground-truth data were created by averaging the data from seven adjacent slices.

For each slice, the spatial sensitivity profiles of each coil were estimated using ESPIRiT [15] , a self-calibrated parallel imaging method.

Ground-truth images were reconstructed by solving (2) using the fully sampled data without regularization.

Each slice was then passed through the forward model and retrospectively under-sampled using a different variable-density Poisson-disc sampling pattern [7] , [16] with a 16??16 calibration region and acceleration factor R ??? 12.

Slices from the first 16 volunteers were used for training, discarding the first and last 20 edge slices of each volume (4,384 slices).

Similarly, slices from the next two volunteers were used for validation (548 slices), and slices from the last two volunteers were used for testing (548 slices).

We added complex-valued Gaussian noise with standard deviation ?? = 0.01 to the noise-free, averaged data.

Implementation: For all experiments we used a Euclidean norm loss function for training.

When training with ground-truth (supervised), the loss was applied in the image domain.

For unsupervised training, the loss was applied in the measurement (Fourier) domain.

We used a U-Net architecture [17] for the CNN autoencoder, with separate input channels for the real and imaginary components.

The U-Net consisted of three encoding layers with ReLU activation functions and 64, 128, and 256 channels, respectively, followed by three similar decoding layers.

A final convolutional layer with no activation function mapped the decoder back to two channels.

All convolutions used a 3 ?? 3 kernel size.

For comparison, MoDL [6] was also implemented using the same unrolled parameters and CNN architecture.

All networks were implemented in PyTorch.

Evaluation: DBP was separately trained with and without ground-truth data.

We also trained MoDL with ground-truth data.

In addition, we also evaluated parallel imaging and compressed sensing (PICS) [7] using BART [16] , with 1 -Wavelet regularization parameter optimized over the validation set.

Normalized root mean-squared error (NRMSE) was used to compare reconstructions.

V. RESULTS Fig. 2 shows the mean NRMSE on the training set for each epoch.

In addition to a performance gap between supervised and unsupervised learning, unsupervised DBP has noisier updates, likely because the loss function in the measurement domain is a noisy surrogate to the NRMSE.

Fig. 3 shows the NRMSE across the validation set for different numbers of unrolls during inference.

Even though the networks were trained with 5 unrolls, best performance is seen for different number of unrolls (6, 10 and 12 for MoDL, unsupervised DBP, and supervised DBP, respectively).

Compared to MoDL, the DBP formulation behaves more stably as the number of unrolls increases, which may be due to the hard data consistency constraint.

At the optimal number of unrolls, unsupervised DBP outperforms PICS.

Fig. 4 .

Box plot of test set NRMSE for supervised and unsupervised DBP at two different unrolls -the first matching unrolls at training, and the second chosen to minimize validation set mean NRMSE.

Also shown is PICS NRMSE for optimized regularization on validation set.

Fig. 5 shows some of the intermediate output stages for the supervised and unsupervised DBP networks, indicating that similar structure is learned in both CNNS; however, the supervised DBP appears to better amplify and denoise features in the image.

The magnitude reconstructions and error maps of a representative slice from the test set are shown in Fig. 6 .

Supervised DBP achieves the lowest error, followed by unsupervised DBP, PICS, and MoDL.

Small details in edge sharpness are retained with DBP, but reduced with MoDL and PICS.

There are strong connections to iterative optimization and unrolled deep networks [10] , [18] , [19] .

Jointly optimizing over the images and weights could be viewed a non-linear extension to dictionary learning.

Nonetheless, there is a cost in reconstruction error when moving to unsupervised learning, highlighting the importance of a large training data set to offset the missing ground-truth information.

The choice of measurement loss function and data SNR may also greatly impact the quality.

Fortunately, in many practical settings there is an abundance of undersampled or corrupted measurement data available for training.

In conclusion, the combination of basis pursuit denoising and deep learning can take advantage of undersampled data and provide a means to learn model-based deep learning reconstructions without access to groundtruth images.

@highlight

We present an unsupervised deep learning reconstruction for imaging inverse problems that combines neural networks with model-based constraints.
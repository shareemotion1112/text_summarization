For numerous domains, including for instance earth observation, medical imaging, astrophysics,..., available image and signal datasets often irregular space-time sampling patterns and large missing data rates.

These sampling properties is a critical issue to apply state-of-the-art learning-based (e.g., auto-encoders, CNNs,...) to fully benefit from the available large-scale observations and reach breakthroughs in the reconstruction and identification of processes of interest.

In this paper, we address the end-to-end learning of representations of signals, images and image sequences from irregularly-sampled data, {\em i.e.} when the training data involved missing data.

From an analogy to Bayesian formulation, we consider energy-based representations.

Two energy forms are investigated: one derived from auto-encoders and one relating to Gibbs energies.

The learning stage of these energy-based representations (or priors) involve a joint interpolation issue, which resorts to solving an energy minimization problem under observation constraints.

Using a neural-network-based implementation of the considered energy forms, we can state an end-to-end learning scheme from irregularly-sampled data.

We demonstrate the relevance of the proposed representations for different case-studies: namely, multivariate time series, 2{\sc } images and image sequences.

In numerous application domains, the available observation datasets do not involve gap-free and regularly-gridded signals or images.

The irregular-sampling may result both from the characteristics of the sensors and sampling strategy, e.g. considered orbits and swaths in spacebone earth observation and astrophysics, sampling schemes in medical imaging, as well as environmental conditions which may affect the sensor, e.g. atmospheric conditions and clouds for earth observation.

A rich literature exists on interpolation for irregularly-sampled signals and images (also referred to as inpainting in image processing (4)).

A classic framework states the interpolation issue as the miminisation of an energy, which may be interpreted in a Bayesian framework.

A variety of energy forms, including Markovian priors (12) , patch-based priors (20) , gradient norms in variational and/or PDE-based formulations (4), Gaussian priors () as well as dynamical priors in fluid dynamics (3) .

The later relates to optimal interpolation and kriging (8) , which is among the state-of-the-art and operational schemes in geoscience (10) .

Optimal schemes classically involve the inference of the considered covariance-based priors from irregularly-sampled data.

This may however be at the expense of Gaussianity and linearity assumptions, which do not often apply for real signals and images.

For the other types of energy forms, their parameterization are generally set a priori and not learnt from the data.

Regarding more particularly data-driven and learning-based approaches, most previous works (2; 11; 20) have addressed the learning of interpolation schemes under the assumption that a representative gap-free dataset is available.

This gap-free dataset may be the image itself (9; 20; 18) .

For numerous application domains, as mentionned above, this assumption cannot be fulfilled.

Regarding recent advances in learning-based schemes, a variety of deep learning models, e.g. (7; 16; 24; 23) , have been proposed.

Most of these works focus on learning an interpolator.

One may however expect to learn not only an interpolator but also some representation of considered data, which may be of interest for other applications.

In this respect, RBM models (Restricted Boltzmann

In this section, we formally introduce the considered issue, namely the end-to-end learning of representations and interpolators from irregularly-sampled data.

Within a classic Bayesian or energybased framework, interpolation issues may be stated as a minimization issue

where X is the considered signal, image or image series (referred to hereafter as the hidden state), Y the observation data, only available on a subdomain Ω of the entire domain D, and U θ () the considered energy prior parameterized by θ.

As briefly introduced above, a variety of energy priors have been proposed in the literature, e.g. (4; 20; 5).

We assume we are provided with a series of irregularly-sampled observations, that is to say a set

is only defined on subdomain Ω (i) .

Assuming that all X (i) share some underlying energy representation U θ (), we may define the following operator

such that I(

Here, we aim to learn the parameters θ() of the energy U θ () from the available observation dataset {Y (i) , Ω (i) } i .

Assuming operator I is known, this learning issue can be stated as the minimization of reconstruction error for the observed data

where .

2 Ω refers to the L2 norm evaluated on subdomain.

Learning energy U θ () from observation dataset {Y (i) , Ω (i) } i clearly involves a joint interpolation issue solved by operator I.

Given this general formulation, the end-to-end learning issue comes to solve minimization (3) according to some given parameterization of energy U θ ().

In (3), interpolation operator I is clearly critical.

In Section 3, we investigate a neural-network implementation of this general framework, which embeds a neural-network formulations both for energy U θ () and interpolation operator I.

In this section, we detail the proposed neural-network-based implementation of the end-to-end formulation introduced in the previous section.

We first present the considered paramaterizations for energy U θ () in (3) (Section 3.1).

We derive associated NN-based interpolation operators I (Section 3.2) and describe our overall NN architectures for the end-to-end learning of representations and interpolators from irregularly-sampled datasets (Section 3.3).

We first investigate NN-based energy representations based on auto-encoders (15) .

Let us denote by φ E and φ D the encoding and decoding operators of an auto-encoder (AE), which may comprise both dense auto-encoders (AEs), convolutional AEs as well as recurrent AEs when dealing with time-related processes.

The key feature of AEs is that the encoding operator φ E maps the state X into a low-dimensional space.

Auto-encoders are naturally associated with the following energy

Minimizing (1) according to this energy amounts to retrieving the hidden state whose lowdimensional representation in the encoding space matches the observed data in the original decoded space.

Here, parameters θ refer to the parameters of the encoder φ E and decoder φ D , respectively θ E and θ D .

The mapping to lower-dimensional space may be regarded as a potential loss in the representation potential of the representation.

Gibbs models provide an appealing framework for an alternative energy-based representation, with no such dimensionality reduction constraint.

Gibbs models introduced in statistical physics have also been widely explored in computer vision and pattern recognition (13) from the 80s.

Gibbs models relate to the decomposition of U θ as a sum of potentials U θ (X) = c∈C V c (X c ) where C is a set of cliques, i.e. a set of interacting sites (typically, local neighbors), and V c the potential on clique c. In statistical physics, this formulation states the global energy of the system as the sum of local energies (the potential over locally-interacting sites).

Here, we focus on the following parameterization of the potential function

with N s the set of neighbors of site s for the entire domain D and ψ a potential function.

Lowenergy state for this energy refers to state X which operator ψ provides a good prediction at any site s knowing the state in the neighborhood N s of s. This type of Gibbs energy relates to Gaussian Markov random fields, where the conditional likelihood at one site given its neighborhood follows a Gaussian distribution.

We implement this type of Gibbs energy using the following NN-based parameterization of operator ψ: ψ(X) = ψ 2 (ψ 1 (X)) (6) It involves the composition of a space and/or time convolutional operator ψ 1 and a coordinate-wise operator ψ 2 .

The convolutional kernel for operator ψ 1 is such that the coefficients for the center of convolutional window are set to zero.

This property fulfills the constraint that X(s) is not involved in the computation of ψ (X Ns ) at site s. As an example, for a univariate image, ψ 1 can be set as a convolutional layer with N F filters with kernels of size 3x3x1, such that for each kernel K f K f (1, 1, 0) = 0 (the same applies to biases).

In such a case, operator ψ 2 would be a convolution layer with one filter with a kernel of size 1x1xN F .

Both ψ 1 and ψ 2 can also involve non-linear activations.

Without loss of generality, given this parameterization for operator ψ, we may rewrite energy U θ as U θ (X) = X − ψ (X) 2 where ψ (X)) at site s is given by ψ (X Ns ).

Overall, we may use the following common formulation for the two types of energy-based representation

They differ in the parameterization chosen for operator ψ.

Besides the NN-based energy formulation, the general formulation stated in (3) involves the definition of interpolation operator I, which refers to minimization (1).

We here derive NN-based interpolation architectures from the considered NN-based energy parameterization.

Given parameterization (7), a simple fixed-point algorithm may be considered to solve for (3) .

This algorithm at the basis of DINEOF algorithm and XXX for matrix completion under subspace constraints (2; 14) involves the following iterative update

Interestingly, the algorithm is parameter-free and can be readily implemented in a NN architecture given the number of iterations to be considered.

Given some initialisation, one may typically consider an iterative gradient-based descent which applies at each iteration k

with J U θ the gradient of energy U θ w.r.t.

state X, λ the gradient step and Ω the missing data area.

Automatic differentiation tool embedded in neural network frameworks may provide the numerical computation for gradient J U θ given the NN-based parameterization for energy U θ .

This proved numerically too expensive and was not further investigated in our experiments.

Given the considered form for energy U θ , its gradient w.r.t.

X decomposes as a product

and X − ψ (X) may be regarded as a suboptimal gradient descent.

Hence, rather than considering the true Jacobian J ψ for operator ψ, we may consider an approximation through a trainable CNN G() such that the gradient descent becomes

where

andG is a CNN to be learnt jointly to ψ during the learning stage.

Interestingly, this gradient descent embeds the fixed-point algorithm whenG is the identity.

Let us denote respectively by I F P and I G the fixed-point and gradient-based NN-based interpolators, which implement N I iterations of the proposed interpolation updates.

Below, I N N will denote both I F P and I G .

Whereas I F P is parameter-free, I G involves the parameterization of operator G. We typically consider a CNN with ReLu activations with increasing numbers of filter through layers up to the final layer which applies a linear convolutional with a number of filters given by the dimension of the state.

Figure 1 : Sketch of the considered end-to-end architecture: we depict the considered N I -block architecture which implements a N I -step interpolation algorithm described in Section 3.2.

Operator ψ is defined through energy representation (7) and G refers to the NN-based approximation of the gradient-based update for minimization (1) .

This architecture uses as input a mask Ω corresponding to the missing-data-free domain and an initial gap-filling X (0) for state X. We typically initially fill missing data with zeros for centered and normalized states.

Given the parameterizations for energy U θ and the associated NN-based interpolators presented previously, we design an end-to-end learning for energy representation U θ and associated interpolator I N N , which uses as inputs an observed sample Y (i) and the associated missing-data-free domain Ω (i) .

Using a normalization preprocessing step, we initially fill missing data with zeros to provide an initial interpolated state to the architecture.

We provide a sketch of the architecture in Fig.1 .

Regarding implementation details, beyond the design of the architectures, which may be applicationdependent for operators ψ andG (see Section 4), we consider an implementation under keras using tensorflow as backend.

Regarding the training strategy, we use adam optimizer.

We iteratively increase the number of blocks N I (number of gradient steps) to avoid the training to diverge.

Similarly, we decrease the learning rate accross iterations, typically from 1e-3 to 1e-6.

In our experiments, we typically consider from 5 to 15 blocks.

All the experiments were run under workstations with a single GPU (Nvidia GTX 1080 and GTX 1080 Ti).

In this section, we report numerical experiments on different datasets to evaluate and demonstrate the proposed scheme.

We consider three different case-studies: an image dataset, namely MNIST; a multivariate time-series through an application to Lorenz-63 dynamics (17) and an image sequence dataset through an application to ocean remote sensing data with real missing data patterns.

In all experiments, we refer to the AE-based framework, respectively as FP(d)-ConvAE and G(d)-ConvAE using the fixed-point or gradient-based interpolator where the value of d refers to the number of interpolation steps.

Similarly, we refer to the Gibbs-based frameworks respectively as FP(d)-GENN and G(d)-GENN.

We evaluate the proposed framework on MNIST datasets for which we simulate missing data patterns.

The dataset comprises 60000 28x28 grayscale images.

For this dataset, we only evaluate the AE-based setting.

We consider the following convolutional AE architecture with a 20-dimensional encoding space:

• Encoder operator φ E : Conv2D(20)+ ReLU + AvPooling + Conv2D(40) + ReLU + AveragePooling + Dense(80) + ReLU + Dense(20);

• Decoder operator φ E : Conv2DTranspose(40) + ResNet(2), ResNet: Conv2D(40)+ReLU+Conv2D (20) We generate random missing data patterns composed of N S squares of size W S xW S , the center of the square is randomly sampled uniformly over the image grid.

As illustrated in Fig.3 , we consider four missing data patterns: N S = 20 and W S = 5, N S = 30 and W S = 5, N S = 3 and W S = 9, N S = 6 and W S = 9.

As performance measure, we evaluate an interpolation score (I-score), a global reconstruction score (R-score) for the interpolated images and an auto-encoding (AE-score) score of the trained auto-encoder applied to gap-free data, in terms of explained variance.

We also evaluate a classification score (C-score), in terms of mean accurcay, using the 20-dimensional encoding space as feature space for classification with a 3-layer MLP.

We report all performance measures for both the test dataset in Tab.1 for MNIST dataset.

For benchmarking purposes, we also report the performance of DINEOF framework, which uses a 20-dimensional PCA trained on the gap-free dataset, the auto-encoder architecture trained on gap-free dataset as well as the considered convolutional auto-encoder trained using an initial zero-filling for missing data areas and a training loss computed only of observed data areas.

The later can be regarded as a FP(1)-ConvAE architecture using a single block in Fig.1 .

Overall, these results illustrate that representations trained from gap-free data may not apply when considering significant missing data rates as illustrated by relatively poor performance of PCA-based and AE schemes, when trained from gap-free data.

Similarly, training an AE representations using as input a zero-filling strategy lowers the auto-encoding power when applied to gap-free data.

Overall, the proposed scheme guarantees a good representation in terms of AE score with an additional gain in terms of interpolation performance, typically between ≈ 15% and 30% depending of the missing data patterns, the gain being greater when considering larger missing data areas.

Table 1 : Performance of AE schemes in presence of missing data for Fashion MNIST dataset: for a given convolutional AE architecture (see main text for details), a PCA and ConvAE models trained on gap-free data with a 15-iteration projection-based interpolation (resp., DINEOF and ConvAE), a zero-filling stratefy with the same ConvAE architecture (Zero-ConvAE) and the fixedpoint and gradient-based versions of the proposed scheme.

For each experiment, we evaluate four measures: the reconstruction performance for the known image areas (R-score), the interpolation performance for the missing data areas (I-score), the reconstruction performance of the trained AE when applied to gap-free images (AE-score), the classification score of a MLP classifier trained in the trained latent space for training images involving missing data.

We present an application to the Lorenz-63 dynamics (17) , which involve a 3-dimensional state governed by the following ordinary differential equation:

Under parameterization σ = 10, ρ = 28 and β = 8/3 considered here, Lorenz-63 dynamics are chaotic dynamics, which make then challening in our context.

They can be regarded as a reducedorder model of turbulence dynamics.

We simulate Lorenz-63 time series of 200 time steps using a Runge-Kutta-4 ODE solver with an integration step of 0.01 from an initial condition in the attractor.

For a given experiment, we first subsample the simulated series to a given time step dt and then generate using a uniform random samplong a missing data mask accounting for 75% of the data.

Overall, training and test time series are formed by subsequences of 200 time steps.

We report experiments with the GE-NN setting.

The AE-based framework showed lower performance and is not included here.

The considered GE-NN architecture is as follows: a 1D convolution layer with 120 filters with a kernel width of 3, zero-weight-constraints for the center of the convolution kernel and a Relu activation, a 1D convolution layer with 6 filters a kernel width of 1 and a Relu activation, a residual network with 4 residual units using 6 filters with a kernel width of 1 and a linear activation.

The last layer is a convolutional layer with 3 filters, a kernel width of 1 and a linear activation.

Figure 2 : Example of missing data interpolation for Lorenz-63 dynamics: from left to right, the time series of each of the three components of Lorenz-63 states for dt = 0.02 and a 75% missing data rate.

We depict the irregularly-sampled observed data (black dots), the true state (green,-), the interpolated states using DINEOF (blue, -) and the interpolated states using the proposed approach (G-NN-FP-OI) (red, -).

Visually, the interpolated sequence using our approach can hardly be distinguished from the true states.

For benchmarking purposes, we report the interpolation performance issued from an ensemble Kalman smoother (EnKS) (10) knowing the true model, regarded as a lower-bound of the interpolation performance.

The parameter setting of the EnKS is as follows: 200 members, noise-free dynamical model and spherical observation covariannce to 0.1 · I. We also compare the proposed approach to DINEOF (2; 21).

Here, the learning of the PCA decomposition used in the DINEOF scheme relies on gap-free data.

Fig.2 illustrates this comparison for one sequence of 200 time steps with dt = 0.02.

In this example, one can hardly distinguish the interpolated sequence using the proposed approach (FP(15)-GE-NN).

By contrast, DINEOF scheme cannot retrieve some of the largest deviations.

We report in Appendix Tab.3 the performance of the different interpolation schemes.

The proposed approach clearly outperforms DINEOF by about one order of magnitude for the experiments with a time-step dt = 0.02.

The interpolation error for observed states (first line in Tab.3) also stresses the improved prior issued from the proposed Gibbs-like energy setting.

For chaotic dynamics, global PCA representation seems poorly adapted where local representations as embedded by the considered Gibbs energy setting appear more appealing.

The third case-study addresses satellite-derived Sea Surface Temperature (SST) image time series.

Due to their sensitivity to the cloud cover, such SST datasets issued from infrared sensors may involve large missing data rates (typically, between 70% and 90%, Fig.? ?

for an illustration).

For evaluation purposes, we build a groundtruthed dataset from high-resolution numerical simulations, namely NATL60 data (1), using real cloud masks from METOP AVHRR sensor (19 For this case-study, we consider the following four architectures for the AEs and the GE-NNs:

• ConvAE 1 : the first convolutional auto-encoder involves the following encoder architecture:

five consecutive blocks with a Conv2D layer, a ReLu layer and a 2x2 average pooling layer, the first one with 20 filters the following four ones with 40 filters, and a final linear convolutional layer with 20 filters.

The output of the encoder is 4x16x20.

The decoder involves a Conv2DTranspose layer with ReLu activation for an initial 16x16 upsampling stage a Conv2DTranspose layer with ReLu activation for an additional 2x2 upsampling stage, a Conv2D layer with 16 filters and a last Conv2D layer with 5 filters.

All Conv2D layers use 3x3 kernels.

Overall, this model involves ≈ 400,000 parameters.

• ConvAE 2 : we consider a more complex auto-encoder with an architecture similar to ConvAE 1 where the number of filters is doubled (e.g., The output of the encoder is a 4x16x40 tensor).

Overall, this model involves ≈ 900,000 parameters.

• GE-NN 1,2 : we consider two GE-NN architectures.

They share the same global architecture with an initial 4x4 average pooling, a Conv2D layer with ReLu activation with a zeroweight constraint on the center of the convolution window, a 1x1 Conv2D layer with N filters, a ResNet with a bilinear residual unit, composed of an initial mapping to an initial 32x128x(5*N) space with a Conv2D+ReLu layer, a linear 1x1 Conv2D+ReLu layer with N filters and a final 4x4 Conv2DTranspose layer with a linear activation for an upsampling to the input shape.

GE-NN 1 and GE-NN 2 differ in the convolutional parameters of the first Conv2D layers and in the number of residual units.

GE-NN 1 involves 5x5 kernels, N = 20 and 3 residual units for a total of ≈ 30,000 parameters.

For GE-NN 2 , we consider 11x11 kernels, N = 100 and 10 residual units for a total of ≈ 570,000 parameters.

These different parameterizations were selected so that ConvAE 1 and GE-NN 2 involve a modeling complexity in the same range.

We may point out that the considered GE-NN architecture are not applied to the finest resolution but to downscaled grids by a factor of 4.

The application of GENNs to the finest resolution showed poor performance.

This is regarded as an illustration of the requirement for considering a scale-selection problem when applying a given prior.

The upscaling involves the combination of a Conv2DTranspose layer with 11 filters, a Conv2D layer with a ReLu activation with 22 filters and a linear Conv2D layer with 11 filters.

Similarly to MNIST dataset, we report the performance of the different models in terms of interpolation score (I-score), reconstruction score (R-score) and auto-encoding score (AE-score) both for the training and test dataset.

We compare the performance of the four models using the fixed-point and gradient-based interpolation.

Overall, we can draw conlusions similar to MNIST case-study.

Representations trained from gap-free data lead to poor performance and the proposed scheme reaches the best performance (gain over 50% in terms of explained variance for the interpolation and reconstruction score).

Here, models trained with a zero-filling strategy show good interpolation and reconstruction performance, but very poor AE score, stressing that cannot apply beyond the considered interpolation task.

When comparing GE-NN and AE settings, GE-NNs show slightly better performance with a much lower complexity (e.g., 30,000 parameters for GE-NN 1 vs. 400,000 parameters for ConvAE 1 ).

Regarding the comparison between the fixed-point and gradient-based interpolation strategies, the later reaches slightly better interpolation and reconstruction score.

We may point out the significant gain w.r.t.

OI, which is the current operational tool for ocean remote sensing data.

We illustrate these results in Appendix (Fig.6) , which further stresses the gain w.r.t.

OI for the reconstruction of finer-scale structures.

Table 2 : Performance on SST dataset: We evaluate for each model interpolation, reconstruction and auto-encoding scores, resp.

I-score, R-score and AE-score, in terms of percentage of explained variance resp.

for the interpolation of missing data areas, the reconstruction of the whole image with missing data and the reconstruction of gap-free images.

For each model, we evaluate these score for the training data (first row) and the test dataset (second row in brackets).

We consider four different auto-encoder models, namely 20 and 80-dimensional PCAs and ConvAE 1,2 models, and two GE-NN models, GE-NN 1,2 , combined with three interpolation strategies: the classic zero-filling strategy (Zero) and proposed iterative fixed-point (FP) and gradient-based (G) schemes, the figure in brackets denoting the number of iterations.

For instance, FP(10)-GE-NN 1 refers to GE-NN 1 with a 10-step fixed-point interpolation scheme.

The PCAs are trained from gap-free data.

We also consider an Optimal Interpolation (OI) with a space-time Gaussian covariance with empiricallytuned parameters.

We refer the reader to the main text for the detailed parameterization of the considered models.

In this paper, we have addressed the learning of energy-based representations of signals and images from observation datasets involving missing data (with possibly very large missing data rates).

Using the proposed architectures, we can jointly learn relevant representations of signals and images while jointly providing the associated interpolation schemes.

Our experiments stress that learning representations from gap-free data may lead to representations poorly adapted to the analysis of data with large missing data areas.

We have also introduced a Gibbs priors embedded in a neural network architecture.

Relying on local characteristics rather than global ones as in AE schemes, these priors involve a much lower complexity.

Our experiments support their relevance for addressing inverse problems in signal and image analysis.

Future work may further explore multi-scale extensions of the proposed schemes along with couplings between global and local energy representations and hybrid minimization schemes combining both gradient-based and fixed-point strategies in the considered end-to-end formulation.

A.1 SUPPLEMENTARY FOR MNIST DATASET We illustrate below both the considered masking patterns as well as reconstruction examples for the proposed framework applied to MNIST dataset.

The first row depicts the reference image, the second row the missing data mask and the third one the interpolated image.

The first two panels illustrate interpolation results for training data and last two for test data.

We depict grayscale mnist images using false colors to highmight differences.

We report below a Table which details the interpolation performance of the proposed GE-NN representation applied to Lorenz-63 time series in comparison with a PCA-based scheme and a lowerbound provided by the interpolation assuming the ODE model is known.

A.3 SUPPLEMENTARY FOR SST DATASET We report below reconstruction examples for the application of the proposed GE-NN approach to SST time series with real missing data masks, which involve very large missing data rates (typically above 80%).

The consistency between the interpolation results and the reconstruction of the gap-free image from the learnt energy-based representation further stresses the ability of the proposed approach to extract a generic representation from irregularly-sampled data.

These reulsts (12) is known, and a DINEOF scheme (21) .

We report interpolation results for a 75% missing data rate with uniform random sampling for three different sampling time steps, dt = 0.01, dt = 0.02 and dt = 0.04.

We report the mean square error of the interpolation for the observed data (first row) and masked ones (second row).

also emphasize a much greater ability of the proposed learning-based scheme to reconstruct finescale structures, which can hardly be reconstructed by an OI scheme with a Gaussian space-time covariance model.

We may recall that the later is the stae-of-the-art approach for the processing of satellite-derived earth observation data (8).

Interpolation examples for SST data used during training: first row, reference SST images corresponding to the center of the considered 11-day time window; second row, associated SST observations with missing data, third row, interpolation issued from FP(15)-GE-NN 2 model; third row, reconstruction of the gap-free image series issued from FP(15)-GE-NN 2 model; interpolation issued from an optimal interpolation scheme using a Gaussian covariance model with empirically tuned parameters.

Figure 6 : Interpolation examples for SST data never seen during training: first row, reference SST images corresponding to the center of the considered 11-day time window; second row, associated SST observations with missing data, third row, interpolation issued from FP(15)-GE-NN 2 model; third row, reconstruction of the gap-free image series issued from FP(15)-GE-NN 2 model; interpolation issued from an optimal interpolation scheme using a Gaussian covariance model with empirically tuned parameters.

<|TLDR|>

@highlight

We address the end-to-end learning of energy-based representations for signal and image observation dataset with irregular sampling patterns.
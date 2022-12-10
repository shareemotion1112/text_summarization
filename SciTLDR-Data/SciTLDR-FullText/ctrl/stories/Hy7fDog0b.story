Generative models provide a way to model structure in complex distributions and have been shown to be useful for many tasks of practical interest.

However, current techniques for training generative models require access to fully-observed samples.

In many settings, it is expensive or even impossible to obtain fully-observed samples, but economical to obtain partial, noisy observations.

We consider the task of learning an implicit generative model given only lossy measurements of samples from the distribution of interest.

We show that the true underlying distribution can be provably recovered even in the presence of per-sample information loss for a class of measurement models.

Based on this, we propose a new method of training Generative Adversarial Networks (GANs) which we call AmbientGAN.

On three benchmark datasets, and for various measurement models, we demonstrate substantial qualitative and quantitative improvements.

Generative models trained with our method can obtain $2$-$4$x higher inception scores than the baselines.

The output of the generator is passed through a simulated random measurement function f Θ .

The discriminator must decide if a measurement is real or generated.

models of the data structure.

Recent work has shown that generative models can be particularly effective for easier sensing BID4 ; BID20 ]-but if sensing is expensive in the first place, how can we collect enough data to train a generative model to start with?This work solves this chicken-and-egg problem by training a generative model directly from noisy or incomplete samples.

We show that our observations can be even projections or more general measurements of different types and the unknown distribution is still provably recoverable.

A critical assumption for our framework and theory to work is that the measurement process is known and satisfies certain technical conditions.

We present several measurement processes for which it is possible to learn a generative model from a dataset of measured samples, both in theory and in practice.

Our approach uses a new way of training GANs, which we call AmbientGAN.

The idea is simple: rather than distinguish a real image from a generated image as in a traditional GAN, our discriminator must distinguish a real measurement from a simulated measurement of a generated image; see FIG0 .

We empirically demonstrate the effectiveness of our approach on three datasets and a variety of measurement models.

Our method is able to construct good generative models from extremely noisy observations and even from low dimensional projections with drastic per-sample information loss.

We show this qualitatively by exhibiting samples with good visual quality, and quantitatively by comparing inception scores BID28 ] to baseline methods.

Theoretical results.

We first consider measurements that are noisy, blurred versions of the desired images.

That is, we consider convolving the original image with a Gaussian kernel and adding independent Gaussian noise to each pixel (our actual theorem applies to more general kernels and noise distributions).

Because of the noise, this process is not invertible for a single image.

However, we show that the distribution of measured images uniquely determines the distribution of original images.

This implies that a pure Nash equilibrium for the GAN game must find a generative model that matches the true distribution.

We show similar results for a dropout measurement model, where each pixel is set to zero with some probability p, and a random projection measurement model, where we observe the inner product of the image with a random Gaussian vector.

Empirical results.

Our empirical work also considers measurement models for which we do not have provable guarantees.

We present results on some of our models now and defer the full exploration to Section 8.In FIG1 , we consider the celebA dataset of celebrity faces BID19 ] under randomly placed occlusions, where a randomly placed square containing 1/4 of the pixels is set to zero.

It is hard to inpaint individual images, so cleaning up the data by inpainting and then learning a GAN on the result yields significant artifacts.

By incorporating the measurement process into the GAN training, we can produce much better samples.

In FIG2 we consider learning from noisy, blurred version of images from the celebA dataset.

Each image is convolved with a Gaussian kernel and then IID Gaussian noise is added to each pixel.

Learning a GAN on images denoised by Wiener deconvolution leads to poor sample quality while our models are able to produce cleaner samples.

In FIG2 , we consider learning a generative model on the 2D images in the MNIST handwritten digit dataset BID17 ] from pairs of 1D projections.

That is, measurements consist of picking two random lines and projecting the image onto each line, so the observed value along the line is the sum of all pixels that project to that point.

We consider two variants: in the first, the choice of line is forgotten, while in the second the measurement includes the choice of line.

We find for both variants that AmbientGAN recovers a lot of the underlying structure, although the first variant cannot identify the distribution up to rotation or reflection.

There are two distinct approaches to constructing neural network based implicit generative models; autoregressive BID15 BID25 a) ], and adversarial BID11 ].

Some combination approaches have also been successful BID21 ].The adversarial framework has been shown to be extremely powerful in modeling complex data distributions such as images BID27 ; ; BID3 ], video BID18 ; BID31 ], and 3D models BID0 ; BID32 .

A learned generative model can be useful for many applications.

A string of papers BID4 BID14 ; BID33 ] explore the utility of generative priors to solve ill-posed inverse problems.

BID29 ] demonstrate that synthetic data can be made more realistic using GANs.

BID14 ] and [Zhu et al. (2017) ] show how to translate images from one domain to another using GANs.

The idea of operating generators and discriminators on different spaces has been proposed before.

BID22 ] explores an interesting connection of training stability with low dimensional projections of samples.

They show that training a generator against an array of discriminators, each operating on a different low-dimensional projection of the data can improve stability.

Our work is also closely related to BID10 ] where the authors create 3D object shapes from a dataset of 2D projections.

We note that their setup is a special case of the AmbientGAN framework where the measurement process creates 2D projections using weighted sums of voxel occupancies.

Throughout, we use superscript 'r' to denote real or true distribution, superscript 'g' for the generated distributions, 'x' for the underlying space and 'y' for measurements.

Let p r x be a real underlying distribution over R n .

We observe lossy measurements performed on samples from p r x .

If we let m be the size of each observed measurement, then, each measurement is an output of some measurement function f θ : R n → R m , parameterized by θ.

We allow the measurement function to be stochastic by letting the parameters of the measurement functions have a distribution p θ .

With this notation, for a given x and θ, the measurements are given by y = f θ (x).

We assume that it is easy to sample Θ ∼ p θ and to compute f θ (x) for any x and θ.

The distributions p r x and p θ naturally induce a distribution over the measurements y which we shall denote by p r y .

In other words, if X ∼ p r x and DISPLAYFORM0 Our task is the following: there is some unknown distribution p r x and a known distribution p θ .

We are given a set of IID realizations {y 1 , y 2 , . . .

, y s } from the distribution p r y .

Using these, our goal is to create an implicit generative model of p r x , i.e., a stochastic procedure that can sample from p r x .

Our main idea is to combine the measurement process with the adversarial training framework, as shown in FIG0 .

Just like in the standard GAN setting, let Z ∈ R k , Z ∼ p z be a random latent vector for a distribution p z that is easy to sample from, such as IID Gaussian or IID uniform.

Let DISPLAYFORM1 , and let p g x be the distribution of X g .

Thus, our goal is to learn a generator G such that p g x is close to p r x .

However, unlike the standard GAN setting, we do not have access to the desired objects (X ∼ p r x ).

Instead, we only have a dataset of measurements (samples from Y ∼ p r y ).

Our main idea is to simulate random measurements on the generated objects X g , and use the discriminator to distinguish real measurements from fake measurements.

Thus, we sample a random measurement function f Θ by sampling Θ ∼ p θ and apply it on X g to obtain DISPLAYFORM2 We set up the discriminator to predict if a given y is a sample from the real measurement distribution p r y as opposed to the generated measurement distribution p g y .

Thus, the discriminator is a function D : R m → R.We let q(·) be the quality function that is used to define the objective, based on the discriminator output.

For vanilla GAN, q(x) = log(x) and for Wasserstein GAN ], q(x) = x. Accordingly, the AmbientGAN objective is the following: DISPLAYFORM3 We additionally require f θ to be differentiable with respect to its inputs for all θ.

We implement G and D as feedforward neural networks.

With these assumptions, our model is end-to-end differentiable and can be trained using an approach similar to the standard gradient-based GAN training procedure.

In each iteration, we sample Z ∼ p z , Θ ∼ p θ , and Y r ∼ UNIF{y 1 , y 2 , . . .

, y s } to use them to compute stochastic gradients of the objective with respect to parameters in G and D by backpropagation.

We alternate between updates to parameters of D and updates to parameters of G.We note that our approach is compatible with and complementary to the various improvements proposed to the GAN objective, network architectures, and the training procedures.

Additionally, we can easily incorporate additional information, such as per sample labels, in our framework through conditional versions of the generator and discriminator.

This is exemplified in our experiments, where we use unconditional and conditional versions of DCGAN BID27 ], unconditional Wasserstein GAN with gradient penalty BID12 ], and an Auxiliary Classifier Wasserstein GAN BID23 ] with gradient penalty.

Now, we describe the measurement models that we use for our theoretical and empirical results.

We primarily focus on 2D images and thus our measurement models are tailored to this setting.

The AmbientGAN learning framework, however, is more general and can be used for other data formats and other measurement models as well.

For the rest of this section, we assume that input to the measurement function (x) is a 2D image.

We consider the following measurement models:Block-Pixels: Each pixel is independently set to zero with probability p. Convolve+Noise: Let k be a convolution kernel and let Θ ∼ p θ be the distribution of noise.

Then the measurements are given by f Θ (x) = k * x + Θ, where * is the convolution operator.

Block-Patch: A randomly chosen k × k patch is set to zero.

Keep-Patch: All pixels outside a randomly chosen k × k patch are set to zero.

Extract-Patch: A random k ×k patch is extracted.

Note that unlike the previous measurement function, the information about the location of the patch is lost.

Pad-Rotate-Project: We pad the image on all four sides by zeros.

Then we rotate the image by a random angle (θ) about its center.

The padding is done to make sure that the original pixels stay within the boundary.

Finally, for each channel in the image, we sum the pixels along the vertical axis to get one measurement vector.

PadRotate-Project-θ: This is the same as the previous measurement function, except that along with the projection values, the chosen angle is also included in the measurements.

Gaussian-Projection: We project onto a random Gaussian vector which is included in the measurements.

So, Θ ∼ N (0, I n ), and f Θ (x) = (Θ, Θ, x ).

We show that we can provably recover the true underlying distribution p r x for certain measurement models.

Our broad approach is to show that there is a unique distribution p r x consistent with the observed measurement distribution p r y , i.e., the mapping of distributions of samples p r x to distribution of measurements p r y is invertible even though the map from an individual image x to its measurements f θ (x) is not.

If this holds, then the following lemma immediately gives a consistency guarantee with the AmbientGAN training procedure.

Lemma 5.1.

As in Section 3, let p r x be the data distribution, p θ be the distribution over parameters of the measurement functions and p r y be the induced measurement distribution.

Further, assume that for the given p θ , there is a unique probability distribution p r x that induces the given measurement distribution p r y .

Then, for the vanilla GAN model BID11 DISPLAYFORM0 All proofs including this one are deferred to Appendix A. Note that the previous lemma makes a non-trivial assumption of uniqueness of the true underlying distribution given the measurement distribution.

The next few theorems show that this assumption is satisfied under Gaussian-Projection, Convolve+Noise and Block-Pixels measurement models, thus showing that that we can recover the true underlying distribution with the AmbientGAN framework.

We remark that the required conditions in the preceding theorem are easily satisfied for the common setting of Gaussian blurring kernel with additive Gaussian noise.

The same guarantee can be generalized for any continuous and invertible function instead of a convolution.

We omit the details.

Our next theorem makes an assumption of a finite discrete set of pixel values.

This assumption holds in most practical scenarios since images are represented with a finite number of discrete values per channel.

In this setting, in addition to a consistency guarantee, we also give a sample complexity result for approximately learning the distributions in the AmbientGAN framework.

Theorem 5.4.

Assume that each image pixel takes values in a finite set P. Thus x ∈ P n ⊂ R n .

Assume 0 ∈ P , and consider the Block-Pixels measurement model (Section 4) with p being the probability of blocking a pixel.

If p < 1, then there is a unique distribution p r x that can induce the measurement distribution p r y .

Further, for any > 0, δ ∈ (0, 1], given a dataset of DISPLAYFORM1 IID measurement samples from p r y , if the discriminator D is optimal, then with probability ≥ 1 − δ over the dataset, any optimal generator G must satisfy DISPLAYFORM2

We used three datasets for our experiments.

MNIST is a dataset of 28 × 28 images of handwritten digits BID17 ].

CelebA is a dataset of face images of celebrities BID19 ].

We use an aligned and cropped version where each image is 64 × 64 RGB.

The CIFAR-10 dataset consists of 32 × 32 RGB images from 10 different classes BID16 DISPLAYFORM0 We briefly describe the generative models we used for our experiments.

More details on architectures and hyperparameters can be found in the appendix.

For the MNIST dataset, we use two GAN models.

The first model is a conditional DCGAN which follows the architecture in [Radford et al. (2015)] 1 , while the second model is an unconditional Wasserstein GAN with gradient penalty (WGANGP) which follows the architecture in BID12 2 .

For the celebA dataset, we use an unconditional DCGAN and follow the architecture in BID27 3 .

For the CIFAR-10 dataset, we use an Auxiliary Classifier Wasserstein GAN with gradient penalty (ACWGANGP) which follows the residual architecture in BID12 4 .For measurements with 2D outputs, i.e. Block-Pixels, Block-Patch, Keep-Patch, Extract-Patch, and Convolve+Noise (see Section 4), we use the same discriminator architectures as in the original work.

For 1D projections, i.e. Pad-Rotate-Project, Pad-Rotate-Project-θ, we use fully connected discriminators.

The architecture of the fully connected discriminator used for the MNIST dataset was 25-25-1 and for the celebA dataset was 100-100-1.

Now, we describe some baseline approaches that we implemented to evaluate the relative performance of the AmbientGAN framework.

Recall that we have a dataset of IID samples {y 1 , y 2 , . . .

y s } from the measurement distribution p r y and our goal is to create an implicit generative model for p r x .

A crude baseline is to ignore that any measurement happened at all.

In other words, for cases where the measurements lie in the same space as the full-samples (for example Convolve+Noise) we can learn a generative model directly on the measurements and test how well it approximates the true distribution p r x .

We call this the "ignore" baseline.

A stronger baseline is based on the following observation: If the measurement functions f θ were invertible, and we observed θ i for each measurement y i in our dataset, we could just invert the functions to obtain full-samples DISPLAYFORM0 θi (y i ).

Then we could directly learn a generative model using these full-samples.

Notice that both assumptions are violated in the AmbientGAN setting.

First, we may not observe θ i and second, the functions may not be invertible.

Indeed all the measurement models in Section 4 violate one of the assumptions.

However, we can try to approximate an inverse function and use the inverted samples to train a generative model.

Thus, given a measurement y i = f θi (x i ), we try to "unmeasure" it and obtain x i , an estimate of x i .

We then learn a generative model with the estimated inverse samples and test how well it approximates p For the measurement models described in Section 4, we now describe the methods we used to obtain approximate inverse functions: (a) For the Block-Pixels measurements, a simple approximate inverse function is to just blur the image so that zero pixels are filled in from the surrounding.

We also implemented a more sophisticated approach to fill in the pixels by using total variation inpainting.

(b) For Convolve+Noise measurements with a Gaussian kernel and additive Gaussian Noise, we approximate the inverse by a Wiener deconvolution.

(c) For Block-Patch measurements, we use the Navier Stokes based inpainting method BID2 ] to fill in the zero pixels.

For other measurement models, it is unclear how to obtain an approximate inverse function.

For the Keep-Patch measurement model, no pixels outside a box are known and thus inpainting methods are not suitable.

Inverting Extract-Patch measurements is even harder since the information about the position of the patch is also lost.

For the Pad-Rotate-Project-θ measurements, a conventional technique is to sample many angles, and use techniques for inverting the Radon transform BID7 ].

However, since we observe only a few projections at a time, these methods aren't readily applicable.

Inverting Pad-Rotate-Project measurements is even harder since it lacks information about θ.

So, on this subset of experiments, we report only the results with the AmbientGAN models.

We present some samples generated by the baselines and our models.

For each experiment, we show the samples from the dataset of measurements (Y r ) available for training, samples generated by the baselines (when applicable) and the samples generated by our models (X g ).

We show samples only for a selected value of parameter settings.

More results are provided in the appendix.

All results on MNIST are deferred to the appendix.

Block-Pixels: FIG5 shows results on celebA with DCGAN and FIG7 on CIFAR-10 with ACW-GANGP.

We see that the samples are heavily degraded in our measurement process (left image).

Thus, it is challenging for baselines to invert the measurements process, and correspondingly, they do not produce good samples (middle image).

Our models are able to produce images with good visual quality (right image).Convolve+Noise: We use a Gaussian kernel and IID Gaussian noise.

FIG2 shows results on celebA with DCGAN.

We see that the measurements are drowned in noise (left image) and the baselines Block-Patch, Keep-Patch: FIG1 shows the results for Block-Patch and FIG6 for Keep-Patch measurements on celebA with DCGAN.

On both measurement distributions, our models are able to create coherent faces (right image) by observing only parts of one image at a time.1D projections: Pad-Rotate-Project and Pad-Rotate-Project-θ measurement models exhibit drastic signal degradation; most of the information in a sample is lost during the measurements process.

For our experiments, we use two measurements at a time.

FIG2 shows the results on MNIST with DCGAN.

While the first model is able to learn only up to rotation and reflection (left image), we note that generated digits have similar orientations and chirality within each class without any explicit incentive.

We hypothesize that the model prefers this mode because it is easier to learn with consistent orientation per class.

The second measurement model contains the rotation angle and thus produces upright digits (right image).

While in both cases, the generated images are of lesser visual quality, our method demonstrates that we can produce images of digits given only 1D projections.

Failure case: In FIG6 , we show the samples obtained from our model trained on celebA dataset with Pad-Rotate-Project-θ measurements with a DCGAN.

We see that the model has learned a very crude outline of a face, but lacks details.

This highlights the difficulty in learning complex distributions with just 1D projections and a need for better understanding of distribution recovery under projection measurement model as well as better methods for training GANs.

We report inception scores BID28 ] to quantify the quality of the generative models learned in the AmbientGAN framework.

For the CIFAR-10 dataset, we use the Inception model BID30 ] trained on the ImageNet dataset BID8 1 .

For computing a similar score on MNIST, we trained a classification model with two conv+pool layers followed by two fully connected layers 2 .

The final test set accuracy of this model was 99.2%.

For Block-Pixels measurements on MNIST, we trained several models with our approach and the baselines, each with a different probability p of blocking pixels.

For each model, after convergence, we computed the inception score using the network described above.

A plot of the inception scores as a function of p is shown in Fig. 7 (left).

We note that at p = 0, i.e. if no pixels are blocked, our model is equivalent to a conventional GAN.

As we increase p, the baseline models quickly start to perform poorly, while the AmbientGAN models continue to perform relatively well.

For the Convolve+Noise measurements with a Gaussian kernel of radius 1 pixel, and additive Gaussian noise with zero mean and standard deviation σ, we trained several models on MNIST by varying the value of σ.

A plot of the inception score as a function of σ is shown in Fig. 7 (right) .

We see that for small variance of additive noise, Wiener deconvolution and the "ignore" baseline perform quite well.

However, as we start to increase the noise levels, these baselines quickly deteriorate in performance, while the AmbientGAN models maintain a high inception score.

For 1D projection measurements, we report the inception scores for the samples produced by the AmbientGAN models trained with two projection measurements at a time.

The Pad-Rotate-Project model produces digits at various orientations and thus does quite poorly, achieving an inception score of just 4.18.

The model with Pad-Rotate-Project-θ measurements produces well-aligned digits and achieves an inception score of 8.12.

For comparison, the vanilla GAN model trained with fullyobserved samples achieves an inception score of 8.99.

Thus, the second model comes quite close to the performance of the fully-observed case while being trained only on 1D projections.

In Fig. 8 (left) , we show a plot of inception score vs the probability of blocking pixels p in the BlockPixels measurement model on CIFAR-10.

We note that the total variation inpainting method is quite slow and the performance on MNIST was about the same as unmeasure-blur baseline.

So, we do not run inpainting baselines on the CIFAR-10 dataset.

From the plots, we see a trend similar to the plot obtained with MNIST (Fig. 7, left) , showing the superiority of our approach over baselines.

We show the inception score as a function of training iteration in Fig. 8 (right) .Generative models are powerful tools, but constructing a generative model requires a large, highquality dataset of the distribution of interest.

We show how to relax this requirement, by learning a distribution from a dataset that only contains incomplete, noisy measurements of the distribution.

We hope that this will allow for the construction of new generative models of distributions for which no high-quality dataset exists.

Lemma.

As in Section 3, let p r x be the data distribution, p θ be the distribution over parameters of the measurement functions and p r y be the induced measurement distribution.

Further, assume that for the given p θ , there is a unique probability distribution p r x that induces the given measurement distribution p r y .

Then, for the vanilla GAN model BID11 ], if the Discriminator D is optimal, so that DISPLAYFORM0 Proof.

From the same argument as in Theorem 1 in BID11 ], it follows that p g y = p r y .

Then, since there is a unique probability distribution p Proof.

We note that Since Θ ∼ N (0, I n ), all possible directions for projections are covered.

Further, since the measurement model includes the projection vector Θ as a part of the measurements, in order to match the measurement distribution, the underlying distribution p r x must be such that all 1D marginals are matched.

Thus, by Cramer-Wold theorem BID6 ], any sequence of random vectors that match the 1D marginals must converge in distribution to the true underlying distribution.

Thus, in particular, there is a unique probability distribution p r x that can match all 1D marginals obtained with the Gaussian projection measurements.

Theorem.

Let F(·) denote the Fourier transform and let supp(·) be the support of a function.

Consider the Convolve+Noise measurement model (Section 4) with the convolution kernel k and additive noise distribution p θ .

If supp(F(k)) c = φ and supp(F(p θ )) c = φ, then there is a unique distribution p r x that can induce the measurement distribution p r y .

DISPLAYFORM0 With a slight abuse of notation, we will denote the probability density functions (pdf) also by p subscripted with the variable name.

Then we have DISPLAYFORM1 where the penultimate step follows since by assumption, F(k) is nowhere 0.

In the last step, F −1 is the inverse Fourier transform.

Thus, there is a bijective map between X and Z. Since the Fourier and the inverse Fourier are continuous transformations, this map is also continuous.

So, we can write Z = h(X), where h is a bijective, differentiable function.

So, the pdfs of X and Z are related as DISPLAYFORM2 where J h (x) is the Jacobian of h evaluated atx.

Now, note that since Y is a sum of two random variables, its pdf is a convolution of the individual probability density functions.

So we have: DISPLAYFORM3 Taking the Fourier transform on both sides, we have DISPLAYFORM4 where the penultimate step follows since by assumption, F(p θ ) is nowhere 0.Combining the two results, we have a reverse map from the measurement distribution p y to a sample distribution p x .

Thus, the reverse map uniquely determines the true underlying distribution p x , concluding the proof.

We first state a slightly different version of Theorem 1 from BID11 ] for the discrete setting.

We shall use [n] to denote the set {1, 2, . . .

n}, and use I(·) to denote the indicator function.

Lemma 10.1.

Consider a dataset of measurement samples {y 1 , y 2 , . . .

y s }, where each y i ∈ [t].

We define the empirical version of the vanilla GAN objective as DISPLAYFORM0 For j ∈ [t], letp r y (j) = I(y i = j)/s be the empirical distribution of samples.

Then the optimal discriminator for the empirical objective is such that DISPLAYFORM1 Additionally, if we fix the discriminator to be optimal, then any optimal generator must satisfy p g y =p r y .Proof.

The Empirical Risk Minimization (ERM) version of the loss is equivalent to the taking expectation of the data dependent term with respect to the empirical distribution.

Replacing the real data distribution with the empirical version in the proof of Theorem 1 from BID11 ], we obtain the result.

Now we give a proof of Theorem 5.4.Theorem.

Assume that each image pixel takes values in a finite set P. Thus x ∈ P n ⊂ R n .

Assume 0 ∈ P , and consider the Block-Pixels measurement model (Section 4) with p being the probability of blocking a pixel.

If p < 1, then there is a unique distribution p Proof.

We first consider a more general case and apply that to the Block-Pixels model.

Consider a discrete distribution p x over [t].

We apply random measurement functions to samples from p x to obtain measurements.

Assume that each measurement also belongs to the same set, i.e. [t] .

Let A ∈ R t×t be the transition matrix so that A ij is the probability (under the randomness in measurement functions) that measurement i was produced by sample j.

Then the distribution over measurements p y can be written in terms of p x and A as: DISPLAYFORM2 Thus, if the matrix A is invertible, we can guarantee that the distribution p x is recoverable from p y .Assuming A is invertible, we now turn to the sample complexity.

Let λ be the minimum of magnitude of eigenvalues of A. Since A is invertible, λ > 0.

Let the dataset of measurements be {y 1 , y 2 , . . .

y s }.

For j ∈ [t] and for k ∈ [s], Let Y j k = I(y k = j).

Then for any > 0, we have DISPLAYFORM3 where we used union bound and Chernoff inequalities.

Setting this to δ, we get s = t 2 2λ 2 2 log 2t δ .From Lemma 10.1, we know that the optimal generator must satisfy p .

Thus, we obtain that with probability ≥ 1 − δ, DISPLAYFORM4 = .Now we turn to the specific case of Block-Pixels measurement.

We proceed by dividing the set of all possible |P | n images into n + 1 classes.

The i-th class has those images that have exactly i pixels with zero value.

We sort the images according to their class number (arbitrary ordering within the class) and consider the transition matrix A. Note that given an image from class i it must have j ≥

i zero pixels after the measurement.

Also, no image in class i can produce another image in the same class after measurements.

Thus, the transition matrix is lower triangular.

Since each pixel is blocked independently with probability p and since there are n pixels, the event that no pixels are blocked occurs with probability FIG0 n .

Thus, every image has at least (1 − p) n chance of being unaffected by the measurements.

Any unaffected image maps to itself and thus forms diagonal entries in the transition matrix.

So, we observe that the diagonal entries of the transition matrix are strictly positive and their minimum value is (1 − p) n .For a triangular matrix, the diagonal entries are precisely the eigenvalues and hence we have proved that A is invertible and the smallest eigenvalue is (1 − p) n .

Combined with the result above, by setting λ = (1 − p) n , and t = |P | n , we conclude the proof.

The DCGAN model on MNIST follows the architecture in BID27 ].

The noise input to the generator (Z) has 100 dimensions where each coordinate is sampled IID Uniform on [−1, 1].

The generator uses two linear layers followed by two deconvolutional layers.

The labels are concatenated with the inputs of each layer.

The discriminator uses two convolutional layers followed by two linear layers.

As with the generator, the labels are concatenated with the inputs of each layer.

Batch-norm is used in both generator and the discriminator.

The WGANGP model on MNIST follows the architecture in BID12 ].

The generator takes in a latent vector of 128 dimensions where each coordinate is sampled IID Uniform on [−1, 1].

The generator then applies one linear and three deconvolutional layers.

The discriminator uses three convolutional layers followed by one linear layer.

Batch-norm is not used.

The unconditional DCGAN model on celebA follows the architecture in BID27 ].

The latent vector has 100 dimensions where each coordinate is Uniform on [−1, 1].

The generator applies one linear layer followed by four deconvolutional layers.

The discriminator uses four convolutional layers followed by a linear layer.

Batch-norm is used in both generator and the discriminator.

The ACWGANGP model on CIFAR-10 follows the residual architecture in BID12 ].

The latent vector has 128 dimensions where each coordinate is sampled from IID standard Gaussian distribution.

The generator has a linear layer followed by three residual blocks.

Each residual block consists of two repetitions of the following three operations: conditional batch normalization followed by a nonlinearity followed by an upconvolution layer.

The residual blocks are followed by another conditional batch normalization, a final convolution, and a final tanh non-linearity.

The discriminator consists of one residual block with two convolutional layers followed by three residual blocks, and a final linear layer.

Here, we present some more results for various measurement models.

So far, in our analysis and experiments, we assumed that the parametric form of the measurement function and the distribution of those parameters is exactly known.

This was then used for simulating the stochastic measurement process.

Here, we consider the case where the parameter distribution is only approximately known.

In this case, one would like the training process to be robust, i.e. the quality of the learned generator to be close to the case where the parameter distribution is exactly known.

Through the following experiment, we empirically demonstrate that the AmbientGAN approach is robust to systematic mismatches in the parameter distribution of the measurement function.

Consider the Block-Pixels measurement model (Section 4).

We use the MNIST dataset.

Pixels are blocked with probability p * = 0.5 to obtain a dataset of measurements.

For several values of blocking probability p for the measurement function applied to the output of the generator, we train AmbientGAN models with this dataset.

After training, we compute the inception score of the learned generators and plot it as a function of p in FIG0 .

We note that the plot peaks at p = p * = 0.5 and gradually drops on both sides.

This suggests that our method is somewhat robust to parameter distribution mismatch.

We provide further evidence that the generator learned through AmbientGAN approach captures the data distribution well.

Generative models have been shown to improve sensing over sparsitybased approaches BID4 ].

We attempt to use the GAN learned using our procedure for compressed sensing.

We trained an AmbientGAN with Block-Pixels measurement model (Section 4) on MNIST with p = 0.5.

Using the learned generator, we followed the rest of the procedure in BID4 ] using their code 3 .

FIG0 (right) shows a plot of reconstruction error vs the number of measurements, comparing Lasso with AmbienGAN.

Thus, we observe a similar reduction in the number of measurements while using AmbientGAN trained with corrupted samples instead of a regular GAN trained with fully observed samples.

<|TLDR|>

@highlight

How to learn GANs from noisy, distorted, partial observations
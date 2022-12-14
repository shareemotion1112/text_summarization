Assessing distance betweeen the true and the sample distribution is a key component of many state of the art generative models, such as Wasserstein Autoencoder (WAE).

Inspired by prior work on Sliced-Wasserstein Autoencoders (SWAE) and kernel smoothing we construct a new generative model – Cramer-Wold AutoEncoder (CWAE).

CWAE cost function, based on introduced Cramer-Wold distance between samples, has a simple closed-form in the case of normal prior.

As a consequence, while simplifying the optimization procedure (no need of sampling necessary to evaluate the distance function in the training loop), CWAE performance matches quantitatively and qualitatively that of WAE-MMD (WAE using maximum mean discrepancy based distance function) and often improves upon SWAE.

One of the crucial aspects in construction of generative models is devising effective method for computing and minimizing distance between the true and the model distribution.

Originally in Variational Autencoder (VAE) BID10 this computation was carried out using variational methods.

An important improvement was brought by the introduction of Wasserstein metric BID14 and the construction of WAE-GAN and WAE-MMD models, which relax the need for variational methods.

WAE-GAN requires a separate optimization problem to be solved to approximate the used divergence measure, while in WAE-MMD the discriminator has the closed-form obtained from a characteristic kernel, i.e. one that is injective on distributions BID12 .

A recent contribution to this trend of simplifying the construction of generative models is Sliced-Wasserstein Autoencoder (SWAE, BID11 ), where a significantly simpler AutoEncoder based model based on Wasserstein distance is proposed.

The main innovation of SWAE was the introduction of the sliced-Wasserstein distance -a fast to estimate metric for comparing two distributions, based on the mean Wasserstein distance of one-dimensional projections.

However, even in SWAE there is no close analytic formula that would enable computing the distance of the sample from the standard normal distribution.

Consequently in SWAE two types of sampling are needed: (i) sampling from the prior distribution and (ii) sampling over one-dimensional projections.

Our main contribution is introduction of the CramerWold distance between distributions, which has a closed-form for the distance of a sample from standard multivariate normal distribution.

Its important feature is that it is given by a characteristic kernel which has a closed-form given by equation 7 for the product of radial Gaussians 1 .

We use it to construct an AutoEncoder based generative model, called Cramer-Wold AutoEncoder (CWAE), in which the cost function, for a normal prior distribution, has a closed analytic formula.

Thus

Motivated by the prevalent use of normal distribution as prior in modern generative models, we investigate whether it is possible to simplify optimization of such models.

As the first step towards this, in this section we introduce Cramer-Wold distance, which has a simple analytical formula for computing normality of a high-dimensional sample.

On a high level our approach uses the traditional L 2 distance of kernel-based density estimation, computed across multiple single-dimensional projections of the true data and the output distribution of the model.

We base our construction on the following two popular tricks of the trade:Sliced-based decomposition of a distribution: Following the footsteps of BID11 ; BID5 , the first idea is to leverage the Cramer-Wold Theorem BID3 and Radon Transform BID4 to reduce computing distance between two distributions to one dimensional calculations.

For v in the unit sphere S D ⊂ R D , the projection of the set X ⊂ R D onto the space spanned by v is given by v T X and the projection of N (m, αI) is N (v T m, α).

Cramer-Wold theorem states that two multivariate distributions can be uniquely identified by their all one-dimensional projections.

For example, to obtain the key component of SWAE model, i.e. the sliced-Wasserstein distance between two samples X, Y ∈ R D , we compute the mean Wasserstein distance between all one-dimensional projections: DISPLAYFORM0 where S D denotes the unit sphere in R D and σ D is the normalized surface measure on S D .

This approach is effective since the one-dimensional Wasserstein distance between samples has the closed form, and therefore to estimate (1) one has to sample only over the projections.

Smoothing distributions: Using the sliced-based decomposition requires us to define distance between two sets of samples, in a single dimensional space.

To this end we will use a trick-of-trade applied commonly in statistics in order to compare samples or distributions which is to first smoothen (sample) distribution with a Gaussian kernel.

For the sample R = (r i ) i=1..n ⊂ R by its smoothing with Gaussian kernel N (0, γ) we understand DISPLAYFORM1 where by N (m, S) we denote the one-dimensional normal density with mean m and variance S. This produces a distribution with regular density, and is commonly used in kernel density estimation.

If R comes from the normal distribution with standard deviation close to one, the asymptotically optimal choice of γ is given by the Silverman's rule of thumb γ = ( 4 3n ) 2/5 , see BID13 .

For continuous density f , its smoothing sm γ (f ) is given by the convolution with N (0, γ), and in the special case of Gaussians we have sm γ (N (m, S)) = N (m, S + γ).

While in general kernel density estimations works well only in low-dimensional spaces, this fits the bill for us, as we will only compute distances on single dimensional projections of the data.

Cramer-Wold distance.

We are now ready to introduce the Cramer-Wold distance.

In a nutshell, we propose to compute the squared distance between two samples by considering the mean squared L 2 distance between their smoothed projections over all single dimensional subspaces.

By the squared L 2 distance between functions f, g : R → R we refer to f − g 2 2 = |f (x) − g(x)| 2 dx.

A key feature of this distance is that it permits a closed-form in the case of normal distribution.

More precisely, the following algorithm fully defines the Cramer-Wold distance between two samples DISPLAYFORM2 D (for illustration of Steps 1 and 2 see FIG0 ): DISPLAYFORM3 .

compute the squared L 2 distance of the densities sm γ (v T X) and sm γ (v T X): DISPLAYFORM4 .

to obtain squared Cramer-Wold distance average (integrate) the above formula over all possible v ∈ S D .

The key theoretical outcome of this paper is that the result of the computation of the Cramer-Wold distance from the previous section can be simplified to a closed form solution.

Consequently, to compute the distance of two samples there is no need of finding the optimal transport like in WAE or the necessity to sample over the projections like in SWAE.

For the case of simplicity we provide in this section the formulas for the distance between two samples and the distance of a sample from the standard normal density.

The general definition of Cramer-Wold metric is presented in Appendix, Section A. DISPLAYFORM0 We formally define the squared Cramer-Wold distance by the formula DISPLAYFORM1 where DISPLAYFORM2 ; −s) and 1 F 1 is the Kummer's confluent hypergeometric function (see, e.g., BID2 ).

Moreover, φ D (s) has the following asymptotic formula valid for D ≥ 20: DISPLAYFORM3 To prove the Theorem 3.1 we will need the following crucial technical proposition.

Proposition 3.1.

Let z ∈ R D and γ > 0 be given.

Then DISPLAYFORM4 Proof.

By applying orthonormal change of coordinates without loss of generality we may assume that z = (z 1 , 0, . . . , 0), and then DISPLAYFORM5 Making use of the formula for slice integration of functions on spheres BID1 , Corollary A.6)

we get: DISPLAYFORM6 where V K denotes the surface volume of a sphere S K ⊂ R K .

Applying the above equality for the function f (v 1 , . . .

, v D ) = N (z 1 v 1 , γ)(0) and s = z 2 1 /(2γ) = z 2 /(2γ) we consequently get that the LHS of (4) simplifies to DISPLAYFORM7 which completes the proof since DISPLAYFORM8 Proof of Theorem 3.1.

Directly from the definition of smoothing we obtain that DISPLAYFORM9 Now applying the one-dimensional formula the the L 2 -scalar product of two Gaussians: DISPLAYFORM10 and the equality f − g DISPLAYFORM11 , we simplify the squared-L 2 norm in the integral of RHS of (5) to DISPLAYFORM12 Applying directly Proposition 3.1 we obtain formula (2).

Proof of the formula for the asymptotics of the function φ D is provided in the Appendix.

Thus to estimate the distance of a given sample X to some prior distribution f , one can follow the common approach and take the distance between X and a sample from f .

As the main theoretical result of the paper we view the following theorem, which says that in the case of standard Gaussian multivariate prior, we can completely reduce the need for sampling (we omit the proof since it is similar to that of Theorem 3.1).

DISPLAYFORM13 One can easily obtain the general formula for the distance between mixtures of radial distributions.

This follows from the fact that the Cramer-Wold distance is given by a scalar product ·, · cw which has a closed-form for the product of two radial Gaussians: DISPLAYFORM14 The above formula means that Cramer-Wold distance is defined by Cramer-Wold kernel, for more details see Appendix, Section A.

This section is devoted to the construction of CWAE.

Since we base our construction on the AutoEncoder, to establish notation let us formalize it here.

AutoEncoder.

Let X = (x i ) i=1..n ⊂ R N be a given data set.

The basic aim of AE is to transport the data to a typically, but not necessarily, less dimensional latent space Z = R D with reconstruction error as small as possible.

Thus, we search for an encoder E : R n → Z and decoder D : Z → R n functions, which minimize the reconstruction error on the data set X: DISPLAYFORM0 AutoEncoder based generative model.

CWAE, similarly to WAE, is a classical AutoEncoder model with modified cost function which forces the model to be generative, i.e. ensures that the data transported to the latent space comes from the (typically Gaussian) prior f .

This statement is formalized by the following important remark, see also BID14 .

Remark 4.1.

Let X be an N -dimensional random vector from which our data set was drawn, and let Y be a random vector with a density f on latent Z.Suppose that we have constructed functions E : R N → Z and D : Z → R N (representing the encoder and the decoder) such that DISPLAYFORM1 2.

random vector EX has the distribution f .Then by the point 1 we obtain that D(EX) = X, and therefore DY has the same distribution as DISPLAYFORM2 This means that to produce samples from X we can instead produce samples from Y and map them by the decoder D.Since an estimator of the image of the random vector X is given by its sample X, we conclude that a generative model is correct if it has small reconstruction error and resembles the prior distribution in the latent.

Thus, to construct a generative AutoEncoder model (with Gaussian prior), we add to its cost function a measure of distance of a given sample from normal distribution.

Once the crucial ingredient of CWAE is ready, we can describe its cost function.

To ensure that the data transported to the latent space Z are distributed according to the standard normal density, we add to the cost function logarithm 4 of the Cramer-Wold distance from standard multivariate normal density d DISPLAYFORM0 Since the use of special functions involved in the formula for Cramer-Wold distance might be cumbersome, we apply in all experiments (except for the illustrative 2D case) the asymptotic form (12) of function φ D : DISPLAYFORM1 where γ n = ( 4 3n ) 2/5 is chosen by the Silverman's rule of thumb BID13 .Comparison with WAE and SWAE models.

Finally, let us briefly recapitulate differences between the introduced CWAE, WAE variants of BID14 and SWAE BID11 .

In contrast to WAE-MMD and SWAE, CWAE model does not require sampling from normal distribution (as in WAE-MMD) or over slices (as in SWAE) to evaluate its cost function, and in this sense uses a closed formula cost function.

In contrast to WAE-GAN, our objective does not require a separately trained neural network to approximate the optimal transport function, thus avoiding pitfalls of adversarial training.

In this paper we are interested in WAE-MMD and SWAE models, which do not use parameterized distance functions, e.g. trained adversarially like in WAE-GAN.

However, in future work we plan to introduce an adversarial version of CWAE and compare it with WAE-GAN.

3 We recall that for function (or in particular random vector) X : Ω → R D by image(X) we denote the set consisting of all possible values X can attain, i.e. {X(ω) : ω ∈ Ω}. 4 We take the logarithm of the Cramer-Wold distance to improve balance between the two terms in the objective function.

In this section we empirically validate the proposed CWAE model on standard benchmarks for generative models: CelebA, Cifar-10 and MNIST.

We will compare CWAE model with WAE-MMD BID14 and SWAE BID11 .

As we will see, our results match those of WAE-MMD, and in some cases improve upon SWAE, while using a simpler to optimize cost function (see the previous section for a more detailed discussion).

The rest of this section is structured as follows.

In Section 5.2 we report results on standard qualitative tests, as well as a visual investigations of the latent space.

In Section 5.3 we will turn our attention to quantitative tests using Fréchet Inception Distance and other metrics.

In the experiment we have used two basic architecture types.

Experiments on MNIST were performed using a feedforward network for both encoder and decoder, and a 20 neuron latent layer, all with ReLU activations.

In case of CIFAR-10 and CelebA data sets we used convolution-deconvolution architectures.

Please refer to Appendix E for full details.

The quality of a generative model is typically evaluated by examining samples or interpolations.

We present such a comparison between CWAE with WAE-MMD in FIG1 .

We follow the same procedure as in BID14 .

In particular, we use the same base neural architecture for both CWAE and WAE-MMD.

We consider for each model (i) interpolation between two random examples from the test set (leftmost in FIG1 ), (ii) reconstruction of a random example from the test set (middle column in FIG1 ), and finally a sample reconstructed from a random point sampled from the prior distribution (right column in FIG1 .

The experiment shows that there are no perceptual differences between CWAE and WAE-MMD generative distribution.

In the next experiment we qualitatively assess normality of the latent space.

This will allow us to ensure that CWAE does not compromise on the normality of its latent distribution, which recall is part of the cost function for all the models except AE.

We compare CWAE 5 with AE, VAE, WAE and SWAE on the MNIST data with using 2-dimensional latent space and a two dimensional Gaussian prior distribution.

Results are reported in Figure 3 .

As is readily visible, the latent distribution of CWAE is as close, or perhaps even closer, to the normal distribution than that of the other models.

Furthermore, the AutoEncoder presented in the second figure is noticeably different from a Gaussian distribution, which is to be expected because it does not optimize for normality in contrast to the other models.

To summarize, both in terms of perceptual quality and satisfying normality objective, CWAE matches WAE-MMD.

The next section will provide more quantitative studies.

In order to quantitatively compare CWAE with other models, in the first experiment we follow the common methodology and use the Fréchet Inception Distance (FID) introduced by BID8 .

Further, we evaluate the sharpness of generated samples using the Laplace filter following BID14 .

Results for CWAE and WAE are summarized in Tab.

1.

In agreement with the qualitative studies, we observe FID and sharpness scores of CWAE to be similar to WAE-MMD.

Figure 3: Latent distribution of CWAE is close to the normal distribution.

Each subfigure presents points sampled from two-dimensional latent spaces, AE, VAE, WAE, SWAE, and CWAE (left to right).

All trained on the MNIST data set.

Next, by comparing training time between CWAE and other models, we found that for batch-sizes up to 1024, which covers the range of batch-sizes used typically for training autoencoders, CWAE is faster (in terms of time spent per batch) than other models.

More precisely, CWAE is approximately 2× faster up to 256 batch-size.

Details are relegated to the Appendix D.Finally, motivated by Remark 4.1 we propose a novel method for quantitative assessment of the models based on their comparison to standard normal distribution in the latent.

To achieve this we have decided to use one of the most popular statistical normality tests, i.e. Mardia tests BID7 ).

Mardia's normality tests are based on verifying whether the skewness b 1,D (·) and kurtosis DISPLAYFORM0 are close to that of standard normal density.

The expected Mardia's skewness and kurtosis for standard multivariate normal distribution is 0 and D(D + 2), respectively.

To enable easier comparison in experiments we consider also the value of the normalized Mardia's kurtosis given by b 2,D (X) − D(D + 2), which equals zero for the standard normal density.

Results are presented in Figure 4 and TAB1 .

In Figure 4 we report for CelebA data set the value of FID score, Mardia's skewness and kurtosis during learning process of AE, VAE, WAE, SWAE and CWAE (measured on the validation data set).

WAE, SWAE and CWAE models obtain the best reconstruction error, comparable to AE.

VAE model exhibits a sightly worse reconstruction error, but values of kurtosis and skewness indicating their output is closer to normal distribution.

As expected, the output of AE is far from normal distribution; its kurtosis and skewness grow during learning.

This arguably less standard evaluation, which we hope will find adoption in the community, serves as yet another evidence that CWAE has strong generative capabilities which at least match performance of WAE-MMD.

Moreover we observe that VAE model's output distribution is closest to the normal distribution, at the expense of the reconstruction error, which is reflected by the blurred reconstructions typically associated with VAE model.

Moreover, motivated by the above approach based on normality tests 6 we have verified how the Cramer-Wold metric works as a Gaussian goodness of fit, however, the results were not satisfactory.

The tests based on Cramer-Wold metric were, in general, in the middle of compared tests (Mardia, Henze-Zirkler and Royston tests).On the whole, WAE-MMD and CWAE achieve, practically speaking, the same level of performance in terms of FID score, sharpness, and our newly introduced normality test.

Additionally, CWAE fares better in many of these metrics than SWAE.

In the paper we have presented a new autoencoder based generative model CWAE, which matches results of WAE-MMD, while using a cost function given by a simple closed analytic formula.

We hope this result will encourage future work in developing simpler to optimize analogs of strong neural models.

Crucial in the construction of CWAE is the use of the developed Cramer-Wold metric between samples and distributions, which can be effectively computed for Gaussian mixtures.

As a consequence we obtain a reliable measure of the divergence from normality.

Future work could explore use of the Cramer-Wold distance in other settings, in particular in adversarial models.

In this section we first formally define the Cramer-Wold metric, and later show that it is given by a characteristic kernel which has closed-form for spherical Gaussians.

For more information on the kernels, and in general kernel embedding of distributions we refer the reader to BID12 .Let us first introduce the general definition of the cw-metric.

To do so we generalize the notion of smoothing for arbitrary measures µ by the formula: DISPLAYFORM0 where * denotes the convolution operator for two measures, and we identify the normal density N (0, γI) with the measure it introduces.

It is well-known that the resulting measure has the density given by x → N (x, γI)(y)dµ(y).

sm γ (N (0, αI)) = N (0, (α + γ)I)).

Moreover, by applying the characteristic function one obtains that if the smoothing of two measures coincide, then the measures also coincide: DISPLAYFORM0 We also need to define the transport of the density by the projection x → v T x, where v is chosen from the unit sphere S D .

The definition is formulated so that if X is a random vector with density f , then f v is the density of the random vector DISPLAYFORM1 where d D−1 denotes the D − 1-dimensional Lebesgue measure.

In general, if µ is a measure on R D , then µ v is the measure defined on R by the formula DISPLAYFORM2 Since, if a random vector X has the density N (a, γI), then the random variable X v has the density N (v T a, α), we directly conclude that DISPLAYFORM3 It is also worth noticing, that due to the fact that the projection of a Gaussian is a Gaussian, the smoothing and projection operators commute, i.e.: DISPLAYFORM4 Given fixed γ > 0, the two above notions allow us to formally define the cw-distance of two measures µ and ν by the formula DISPLAYFORM5 First observe that this implies that cw-distance is given by the kernel function DISPLAYFORM6 Let us now prove that the function d cw defined by equation 10 is a metric (which, in the kernel function literature means that the kernel is characteristic).

Theorem A.1.

The function d cw is a metric.

Proof.

Since d cw comes from a scalar product, we only need to show that if a distance of two measures is zero, the measures coincide.

So let µ, ν be given measures such that d cw (µ, ν) = 0.

This implies that DISPLAYFORM7 By equation 9 this implies that µ v = ν v .

Since this holds for all v ∈ S D , by the Cramer-Wold Theorem we obtain that µ = ν.

Thus we can summarize the above by saying that the Cramer-Wold kernel is a characteristic kernel which has the closed-form the scalar product of two radial Gaussians given by equation 7: DISPLAYFORM8 Remark A.1.

Observe, that except for the Gaussian kernel it is the only kernel which has the closed form for the spherical Gaussians, which as we discuss in the next section is important, as the RBF (Gaussian) kernels cannot be successfully applied in AutoEncoder based generative models.

The reason is that the derivative of Gaussian decreases to fast, and therefore it does not enforce the proper learning of the model, see also the comments in BID14 , Section 4, WAE-GAN and WAE-MMD specifics).

In this section we are going to compare CWAE model to WAE-MMD.

In particular we show that CWAE can be seen as the intersection of the sliced-approach together with MMD-based models.

Since both WAE and CWAE use kernels to discriminate between sample and normal density, to compare the models we first describe the WAE model.

WAE cost function for a given characteristic kernel k and sample X = ( DISPLAYFORM0 .n is a sample from the standard normal density N (0, I), and d 2 k (X, Y ) denotes the kernel-based distance between the probability distributions representing X and Y , that is 1 n i δ xi and 1 n i δ yi , where δ z denotes the atom Dirac measure at z ∈ R D .

The inverse multiquadratic kernel k is chosen as default DISPLAYFORM1 where in experiments in BID14 ) a value C = 2Dσ 2 was used, where σ is the hyper-parameter denoting the size of the normal density.

Thus the model has hyper-parameters λ and σ, which were chosen to be λ = 10, σ 2 = 1 in MNIST, λ = 100, σ 2 = 2 in CelebA. Observe that the hyper-parameters do not depend on the sample size and that, in general, the WAE-MMD model hyper-parameters have to be chosen by hand.

Now let us describe the CWAE model.

CWAE cost function for a sample X = ( DISPLAYFORM2 where distance between the sample and standard normal distribution is taken with respect to the Cramer-Wold kernel with a regularizing hyperparameter γ given by the Silverman's rule of thumb (the motivation for such a choice of hyper-parameters is explained in Section 2).Thus, we have the following differences:• Due to the properties of Cramer-Wold kernel, in the distance we are able to substitute the sample estimation of d • CWAE, as compared to WAE, has no hyper-parameters:1.

In our preliminary experiments we have observed that in many situations (like in the case of log-likelihood), taking the logarithm of the nonnegative factors of the cost function, which we aim to minimize to zero, improves the learning process.

Motivated by this, instead of taking the additional weighting hyper-parameter λ (as in WAE-MMD), whose aim is to balance the MSE and divergence terms, we take the logarithm of the divergence.

Automatically (independently of dimension) balance those terms in the learning process.

2.

The choice of regularization hyper-parameter is given by the Silverman's rule of thumb, and depends on the sample size (contrary to WAE-MMD, where the hyper-parameters are chosen by hand, and in general do not depend on the sample size).Summarizing, in CWAE model, contrary to WAE-MMD, we do not have to choose hyper-parameters.

Moreover, since we do not have the noise in the learning process given by the random choice of the sample Y from N (0, I), the learning should be more stable.

As a consequence, see Figure 7 , CWAE in generally learns faster then WAE-MMD, and has smaller standard deviation of the cost-function during the learning process.

In this section we consider the estimation of values of the function DISPLAYFORM0 ; −s) for s ≥ 0, which is crucial in the formulation for the Cramer-Wold distance.

First we will provide its approximate asymptotic formula valid for dimensions D ≥ 20, and then we shall consider the special case of D = 2 (see FIG3 To do so, let us first recall (Abramowitz & Stegun, 1964, Chapter 13 ) that the Kummer's confluent hypergeometric function 1 F 1 (denoted also by M ) has the following integral representation DISPLAYFORM1 valid for a, b > 0 such that b > a. Since we consider that latent is at least of dimension D ≥ 2, it follows that DISPLAYFORM2 By making a substitution u = x 2 , du = 2xdx, we consequently get DISPLAYFORM3 DISPLAYFORM4 Proof.

By (11) we have to estimate asymptotics of DISPLAYFORM5 Since for large D, for all x ∈ [−1, 1] we have DISPLAYFORM6 we get DISPLAYFORM7 To simplify the above we apply the formula (1) from BID15 : DISPLAYFORM8 with α, β fixed so that α + β = 1 (so only the error term of order O(|z| −2 ) remains), and get DISPLAYFORM9 Summarizing, DISPLAYFORM10 In general one can obtain the iterative direct formulas for function φ D with the use of erf and modified Bessel functions of the first kind I 0 and I 1 , but for large D they are of little numerical value.

We consider here only the special case D = 2 since it is used in the paper for illustrative reasons in the latent for the MNIST data set.

Since we have the equality (Gradshteyn & Ryzhik, 2015, (8.406.3) and (9.215.3)): DISPLAYFORM11 to practically implement φ 2 we apply the approximation of I 0 from (Abramowitz & Stegun, 1964, page 378) given in the following remark.

Remark C.1.

Let s ≥ 0 be arbitrary and let t = s/7.5.

Then DISPLAYFORM12 +.02635537t −6 −.01647633t −7 +.00392377t −8 ) for s ≥ 7.5.

FIG4 gives comparison of mean learning time for different most frequently used batch-sizes.

Time spent on processing a batch is actually smaller for CWAE for a practical range of batch-sizes [32, 512] .

For batch-sizes larger than 1024, CWAE is slower due to its quadratic complexity with respect to the batch-size.

However, we note that batch-sizes larger even than 512 are relatively rarely used in practice for training autoencoders.

CelebA (with images centered and cropped to 64×64 with 3 color layers) a convolution-deconvolution network:

encoder four convolution layers with 4 × 4 filters and 2 × 2 strides (consecutively 32, 32, 64, and 64 output channels), all ReLU activations, two dense layers (1024 and 256 ReLU neurons) latent 64-dimensional, decoder first two dense layers (256 and 1024 ReLU neurons), three transposed-convolution layers with 4 × 4 filters with 2 × 2 strides (consecutively 64, 32, 32 channels) with ReLU activation, transposed-convolution 4 × 4 with 2 × 2 stride, 3 channels, and sigmoid activation.

CIFAR-10 dataset (32× images with three color layers): a convolution-deconvolution network encoder four convolution layers with 2 × 2 filters, the second one with 2 × 2 strides, other non-strided (3, 32, 32, and 32 channels) with ReLU activation, 128 ReLU neurons dense layer, latent with 64 neurons, decoder two dense ReLU layers with 128 and 8192 neurons, two transposed-convolution layers with 2 × 2 filters (32 and 32 channels) and ReLU activation, a transposed convolution layer with 3 × 3 filter and 2 × 2 strides (32 channels) and ReLU activation, a transposed convolution layer with 2 × 2 filter (3 channels) and sigmoid activation.

The last layer returns the reconstructed image.

The results for all above architectures are given in TAB1 .

All networks were trained with the Adam optimizer BID9 .

The hyperparameters used were learning rate = 0.001, β1 = 0.9, β2 = 0.999, = 1e − 8.

MNIST models were trained for 500 epochs, both CIFAR-10 and CelebA for 200.Additionally, to have a direct comparison to WAE-MMD model on CelebA, an identical architecture was used as that in BID14 utilized for the WAE-MMD model (WAE-GAN architecture is, naturally, different):encoder four convolution layers with 5 × 5 filters, each layer followed by a batch normalization (consecutively 128, 256, 512, and 1024 channels) and ReLU activation, latent 64-dimensional, decoder dense 1024 neuron layer, three transposed-convolution layers with 5 × 5 filters, and each layer followed by a batch normalization with ReLU activation (consecutively 512, 256, and 128 channels), transposed-convolution layer with 5 × 5 filter and 3 channels, clipped output value.

The results for this architecture for CWAE compared to VAE and WAE-MMD models are given in TAB0 .Similarly to BID14 , models were trained using Adam with for 55 epochs, with the same optimizer parameters.

Figure 9 : Results of VAE, WAE-MMD, SWAE, and CWAE models trained on CelebA dataset using the WAE architecture from BID14 .

In "test reconstructions" odd rows correspond to the real test points.

<|TLDR|>

@highlight

Inspired by prior work on Sliced-Wasserstein Autoencoders (SWAE) and kernel smoothing we construct a new generative model – Cramer-Wold AutoEncoder (CWAE).

@highlight

This paper proposes a WAE variant based on a new statistical distance between the encoded data distribution and the latent prior distribution

@highlight

Introduces a variation on the Wasserstein AudoEncoders which is a novel regularized auto-encoder architecture that proposes a specific choice of the divergence penalty

@highlight

This paper proposes the Cramer-Wold autoencoder, which uses the Cramer-Wold distance between two distributions based on the Cramer-Wold Theorem.
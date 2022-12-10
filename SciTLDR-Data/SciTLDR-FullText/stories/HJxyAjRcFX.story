Recent advances in conditional image generation tasks, such as image-to-image translation and image inpainting, are largely accounted to the success of conditional GAN models, which are often optimized by the joint use of the GAN loss with the reconstruction loss.

However, we reveal that this training recipe shared by almost all existing methods causes one critical side effect: lack of diversity in output samples.

In order to accomplish both training stability and multimodal output generation, we propose novel training schemes with a new set of losses named moment reconstruction losses that simply replace the reconstruction loss.

We show that our approach is applicable to any conditional generation tasks by performing thorough experiments on image-to-image translation, super-resolution and image inpainting using Cityscapes and CelebA dataset.

Quantitative evaluations also confirm that our methods achieve a great diversity in outputs while retaining or even improving the visual fidelity of generated samples.

Recently, active research has led to a huge progress on conditional image generation, whose typical tasks include image-to-image translation BID11 ), image inpainting BID31 ), super-resolution BID18 ) and video prediction BID27 ).

At the core of such advances is the success of conditional GANs BID28 ), which improve GANs by allowing the generator to take an additional code or condition to control the modes of the data being generated.

However, training GANs, including conditional GANs, is highly unstable and easy to collapse BID8 ).

To mitigate such instability, almost all previous models in conditional image generation exploit the reconstruction loss such as 1 / 2 loss in addition to the GAN loss.

Indeed, using these two types of losses is synergetic in that the GAN loss complements the weakness of the reconstruction loss that output samples are blurry and lack high-frequency structure, while the reconstruction loss offers the training stability required for convergence.

In spite of its success, we argue that it causes one critical side effect; the reconstruction loss aggravates the mode collapse, one of notorious problems of GANs.

In conditional generation tasks, which are to intrinsically learn one-to-many mappings, the model is expected to generate diverse outputs from a single conditional input, depending on some stochastic variables (e.g. many realistic street scene images for a single segmentation map BID11 ).

Nevertheless, such stochastic input rarely generates any diversity in the output, and surprisingly many previous methods omit a random noise source in their models.

Most papers rarely mention the necessity of random noise, and a few others report that the model completely ignores the noise even if it is fed into the model.

For example, BID11 state that the generator simply learns to ignore the noise, and even dropout fails to incur meaningful output variation.

The objective of this paper is to propose a new set of losses named moment reconstruction losses that can replace the reconstruction loss with losing neither the visual fidelity nor diversity in output samples.

The core idea is to use maximum likelihood estimation (MLE) loss (e.g. 1 / 2 loss) to predict conditional statistics of the real data distribution instead of applying it directly to the generator as done in most existing algorithms.

Then, we assist GAN training by enforcing the generated distribution to match its statistics to the statistics of the real distribution.

In summary, our major contributions are three-fold.

First, we show that there is a significant mismatch between the GAN loss and the reconstruction loss, thereby the model cannot achieve the optimality w.r.t.

both losses.

Second, we propose two novel loss functions that enable the model to accomplish both training stability and multimodal output generation.

Our methods simply replace the reconstruction loss, and thus are applicable to any conditional generation tasks.

Finally, we show the effectiveness and generality of our methods through extensive experiments on three generation tasks, including image-to-image translation, super-resolution and image inpainting, where our methods outperform recent strong baselines in terms of realism and diversity.

Conditional Generation Tasks.

Since the advent of GANs BID8 ) and conditional GANs BID28 ), there has been a large body of work in conditional generation tasks.

A non-exhaustive list includes image translation BID11 BID49 BID25 BID41 BID47 , super-resolution BID18 BID42 BID26 BID43 BID5 BID34 , image inpainting BID31 BID10 BID30 BID29 BID33 BID45 and video prediction BID27 BID13 BID37 BID50 BID1 BID38 .However, existing models have one common limitation: lack of stochasticity for diverse output.

Despite the fact that the tasks are to be one-to-many mapping, they ignore random noise input which is necessary to generate diverse samples from a single conditional input.

A number of works such as BID27 BID11 BID42 try injecting random noise into their models but discover that the models simply discard it and instead learn a deterministic mapping.

Multimodality Enhancing Models.

It is not fully understood yet why conditional GAN models fail to learn the full multimodality of data distribution.

Recently, there has been a series of attempts to incorporate stochasticity in conditional generation as follows.(1) Conditional VAE-GAN.

VAE-GAN BID17 ) is a hybrid model that combines the decoder in VAE BID16 with the generator in GAN BID8 .

Its conditional variants have been also proposed such as CVAE-GAN BID0 , Bicycle-GAN and SAVP BID19 .

These models harness the strengths of the two models, output fidelity by GANs and diversity by VAEs, to produce a wide range of realistic images.

Intuitively, the VAE structure drives the generator to exploit latent variables to represent the multimodality of the conditional distribution.(2) Disentangled representation.

and BID20 propose to learn disentangled representation for multimodal unsupervised image-to-image translation.

These models split the embedding space into a domain-invariant space for sharing information across domains and a domain-specific space for capturing styles and attributes.

These models encode an input into the domain-invariant embedding and sample domain-specific embedding from some prior distribution.

The two embeddings are fed into the decoder of the target domain, for which the model can generate diverse samples.

Conditional VAE-GANs and disentanglement-based methods both leverage the latent variable to prevent the model from discarding the multimodality of output samples.

On the other hand, we present a simpler and orthogonal direction to achieve multimodal conditional generation by introducing novel loss functions that can replace the reconstruction loss.

We briefly review the objective of conditional GANs in section 3.1 and discuss why the two loss terms cause the loss of modality in the sample distribution of the generator in section 3.2.

Conditional GANs aims at learning to generate samples that are indistinguishable from real data for a given input.

The objective of conditional GANs usually consists of two terms, the GAN loss L GAN and the reconstruction loss L Rec .

DISPLAYFORM0 Another popular loss term is the perceptual loss BID14 BID4 BID18 .

While the reconstruction loss encodes the pixel-level distance, the perceptual loss is defined as the distance between the features encoded by neural networks.

Since they share the same form (e.g. 1 / 2 loss), we consider the perceptual loss as a branch of the reconstruction loss.

The loss L GAN is defined to minimize some distance measure (e.g. JS-divergence) between the true and generated data distribution conditioned on input x. The training scheme is often formulated as the following minimax game between the discriminator D and the generator G. DISPLAYFORM1 where each data point is a pair (x, y), and G generates outputs given an input x and a random noise z. Note that D also observes x, which is crucial for the performance BID11 .The most common reconstruction losses in conditional GAN literature are the 1 BID11 and 2 loss BID31 BID27 .

Both losses can be formulated as follows with p = 1, 2, respectively.

DISPLAYFORM2 These two losses naturally stem from the maximum likelihood estimations (MLEs) of the parameters of Laplace and Gaussian distribution, respectively.

The likelihood of dataset (X, Y) assuming each distribution is defined as follows.

DISPLAYFORM3 where N is the size of the dataset, and f is the model parameterized by θ.

The central and dispersion measure for Gaussian are the mean and variance σ 2 , and the correspondences for Laplace are the median and mean absolute deviation (MAD) b. Therefore, using 2 loss leads the model output f θ (x) to become an estimate of the conditional average of y given x, while using 1 loss is equivalent to estimating the conditional median of y given x BID2 .

Note that the model f θ is trained to predict the mean (or median) of the data distribution, not to generate samples from the distribution.

We argue that the joint use of the reconstruction loss with the GAN loss can be problematic, because it may worsen the mode collapse.

Below, we discuss this argument both mathematically and empirically.

One problem of the 2 loss is that it forces the model to predict only the mean of p(y|x), while pushing the conditional variance to zero.

According to BID12 , for any symmetric loss function L s and an estimatorŷ for y, the loss is decomposed into one irreducible term Var(y) and two reducible terms, SE(ŷ, y) and VE(ŷ, y), where SE refers to systematic effect, the change in error caused by the bias of the output, while VE refers to variance effect, the change in error caused by the variance of the output.

DISPLAYFORM0 where S is an operator that is defined to be Notice that the total loss is minimized whenŷ = Sŷ = Sy, reducing both SE and VE to 0.

For 2 loss, Sy and Sŷ are the expectations of y andŷ.

Furthermore, SE and VE correspond to the squared bias and the variance, respectively.

DISPLAYFORM1 DISPLAYFORM2 DISPLAYFORM3 The decomposition above demonstrates that the minimization of 2 loss is equivalent to the minimization of the bias and the variance of prediction.

In the context of conditional generation tasks, this implies that 2 loss minimizes the conditional variance of output.

Figure 1 shows some examples of the Pix2Pix model BID11 ) that is applied to translate segmentation labels to realistic photos in Cityscapes dataset BID6 .

We slightly modify the model so that it takes additional noise input.

We use 1 loss as the reconstruction loss as done in the original paper.

We train four models that use different combinations of loss terms, and generate four samples with different noise input.

As shown in Figure 1 (c), the model trained with only the GAN loss fails to generate realistic images, since the signal from the discriminator is too unstable to learn the translation task.

In Figure 1 (d), the model with only 1 loss is more stable but produces results far from being realistic.

The combination of the two losses in Figure 1 (e) helps not only to reliably train the model but also to generate visually appealing images; yet, its results lack variation.

This phenomenon is also reported in BID27 BID11 , although the cause is unknown.

BID31 and BID10 even state that better results are obtained without noise in the models.

Finally, Figure 1 (f) shows that our new objective enables the model to generate not only visually appealing but also diverse output samples.

We propose novel alternatives for the reconstruction loss that are applicable to virtually any conditional generation tasks.

Trained with our new loss terms, the model can accomplish both the stability of training and multimodal generation as already seen in Figure 1 (f).

FIG0 illustrates architectural comparison between conventional conditional GANs and our models with the new loss terms.

In conventional conditional GANs, MLE losses are applied to the generator's objective to make sure that it generates output samples well matched to their ground-truth.

On the other hand, K to calculate the sample moments, i.e. mean and variance.

Then, we compute the moment reconstruction loss, which is basically an MLE loss between the sample moments and y. (c) The second model proxy MR-GAN is a more stable variant of MR-GAN.

The predictor P , a twin network of G, is trained with an MLE loss to estimate the conditional mean and variances of y given x.

Then, we match the sample meanμ to the predictedμ and the sample varianceσ 2 to the predictedσ 2 .our key idea is to apply MLE losses to make sure that the statistics, such as mean or variance, of the conditional distribution p(y|x) are similar between the generator's sample distribution and the actual data distribution.

In section 4.1, we extend the 1 and 2 loss to estimate all the parameters of Laplace and Gaussian distribution, respectively.

In section 4.2-4.3, we propose two novel losses for conditional GANs.

Finally, in section 4.4, we show that the reconstruction loss conflicts with the GAN loss while our losses do not.

The 2 loss encourages the model to perform the MLE of the conditional mean of y given x while the variance σ 2 , the other parameter of Gaussian, is assumed to be fixed.

If we allow the model to estimate the conditional variance as well, the MLE loss corresponds to DISPLAYFORM0 where the model f θ now estimates bothμ andσ 2 for x. Estimating the conditional variance along with the mean can be interpreted as estimating the heteroscedastic aleatoric uncertainty in BID15 where the variance is the measure of aleatoric uncertainty.

For the Laplace distribution, we can derive the similar MLE loss as DISPLAYFORM1 wherem is the predicted median andb is the predicted MAD BID3 .

In practice, it is more numerically stable to predict the logarithm of variance or MAD BID15 .In the following, we will describe our methods mainly with the 2 loss under Gaussian assumption.

It is straightforward to obtain the Laplace version of our methods for the 1 loss by simply replacing the mean and variance with the median and MAD.

Our first loss function is named Moment Reconstruction (MR) loss, and we call a conditional GAN with the loss as MR-GAN.

As depicted in FIG0 (b), the overall architecture of MR-GAN follows that of a conditional GAN, but there are two important updates.

First, the generator produces K different samplesỹ 1:K for each input x by varying noise input z. Second, the MLE loss is applied to the sample moments (i.e. mean and variance) in contrast to the reconstruction loss which is used directly on the samples.

Since the generator is supposed to approximate the real distribution, we can regard that the moments of the generated distribution estimate the moments of the real distribution.

We approximate the moments of the generated distribution with the sample momentsμ andσ 2 , which are defined as DISPLAYFORM0 The MR loss is defined by pluggingμ andσ 2 in Eq.(9) intoμ andσ 2 in Eq. FORMULA8 .

The final loss of MR-GAN is the weighted sum of the GAN loss and the MR loss: DISPLAYFORM1 As a simpler variant, we can use onlyμ and ignoreσ 2 .

We denote this simpler one as MR 1 (i.e. using mean only) and the original one as MR 2 (i.e. using both mean and variance) according to the number of moments used.

DISPLAYFORM2 In addition, we can easily derive the Laplace versions of the MR loss that use median and MAD.

One possible drawback of the MR loss is that the training can be unstable at the early phase, since the irreducible noise in y directly contributes to the total loss.

For more stable training, we propose a variant called Proxy Moment Reconstruction (proxy MR) loss and a conditional GAN with the loss named proxy MR-GAN whose overall training scheme is depicted in FIG0 (c).

The key difference is the presence of predictor P , which is a clone of the generator with some minor differences: (i) no noise source as input and (ii) prediction of both mean and variance as output.

The predictor is trained prior to the generator by the MLE loss in Eq. FORMULA8 with ground-truth y to predict conditional mean and variance, i.e.μ andσ 2 .

When training the generator, we utilize the predicted statistics of real distribution to guide the outputs of the generator.

Specifically, we match the predicted mean/variance and the mean/variance of the generator's distribution, which is computed by the sample meanμ and varianceσ 2 fromỹ 1:K .

Then, we define the proxy MR loss L pMR as the sum of squared errors between predicted statistics and sample statistics: DISPLAYFORM0 One possible variant is to match only the first moment µ.

As with the MR loss, we denote the original method proxy MR 2 and the variant proxy MR 1 where the number indicates the number of matched moments.

Deriving the Laplace version of the proxy MR loss is also straightforward.

The detailed algorithms of all eight variants are presented in appendix A.Compared to proxy MR-GAN, MR-GAN allows the generator to access real data y directly; thus there is no bias caused by the predictor.

On the other hand, the use of predictor in proxy MR-GAN provides less variance in target values and leads more stable training especially when the batch or sample size is small.

Another important aspect worth comparison is overfitting, which should be carefully considered when using MLE with finite training data.

In proxy MR-GAN, we can choose the predictor with the smallest validation loss to avoid overfitting, and freeze it while training the generator.

Therefore, the generator trained with the proxy MR loss suffers less from overfitting, compared to that trained with the MR loss that directly observes training data.

To sum up, the two methods have their own pros and cons.

We will empirically compare the behaviors of these two approaches in section 5.2.

We here show the mismatch between the GAN loss and the 2 loss in terms of the set of optimal generators.

We then discuss why our approach does not suffer from the loss of diversity in the output samples.

Refer to appendix E for the mismatch of 1 loss.

The sets of optimal generators for the GAN loss and the 2 loss, denoted by G and R respectively, can be formulated as follows: DISPLAYFORM0 Recall that 2 loss is minimized when both bias and variance are zero, which implies that R is a subset of V where G has no conditional variance: DISPLAYFORM1 In conditional generation tasks, however, the conditional variance Var(y|x) is assumed to be nonzero (i.e. diverse output y for a given input x).

Thereby we conclude G∩V = ∅, which reduces to G∩ R = ∅. It means that any generator G cannot be optimal for both GAN and 2 loss simultaneously.

Therefore, it is hard to anticipate what solution is attained by training and how the model behaves when the two losses are combined.

One thing for sure is that the reconstruction loss alone is designed to provide a sufficient condition for mode collapse as discussed in section 3.2.Now we discuss why our approach does not suffer from loss of diversity in the output samples unlike the reconstruction loss in terms of the set of optimal generators.

The sets of optimal mappings for our loss terms are formulated as follows: DISPLAYFORM2 where M 1 corresponds to MR 1 and proxy MR 1 while M 2 corresponds to MR 2 and proxy MR 2 .

It is straightforward to show that DISPLAYFORM3 is satisfied at optimum, the conditional expectations and variations of both sides should be the same too: DISPLAYFORM4 To summarize, there is no generator that is both optimal for the GAN loss and the 2 loss since G ∩ R = ∅. Moreover, as the 2 loss pushes the conditional variance to zero, the final solution is likely to lose multimodality.

On the other hand, the optimal generator w.r.t.

the GAN loss is also optimal w.r.t.

our loss terms since DISPLAYFORM5 This proof may not fully demonstrate why our approach does not give up multimodality, which could be an interesting future work in line with the mode collapsing issue of GANs.

Nonetheless, we can at least assert that our loss functions do not suffer from the same side effect as the reconstruction loss does.

Another remark is that this proof is confined to conditional GAN models that have no use of latent variables.

It may not be applicable to the models that explicitly encode latent variables such as BID0 , , BID19 , , and BID20 , since the latent variables can model meaningful information about the target diversity.

In order to show the generality of our methods, we apply them to three conditional generation tasks: image-to-image translation, super-resolution and image inpainting, for each of which we select Pix2Pix, SRGAN BID18 and GLCIC BID10 as base models, respectively.

We use Maps BID11 and Cityscapes dataset BID6 for image translation and CelebA dataset BID23 for the other tasks.

We minimally modify the base models to include random noise input and train them with MR and proxy MR objectives.

We do not use any other loss terms such as perceptual loss, and train the models from scratch.

We present the details of training and implementation and more thorough results in the appendix.

In every task, our methods successfully generate diverse images as presented in Figure 3 .

From qualitative aspects, one of the most noticeable differences between MR loss and proxy MR loss lies in the training stability.

We find that proxy MR loss works as stably as the reconstruction loss in all three tasks.

On the other hand, we cannot find any working configuration of MR loss in the image Figure 3 : Comparison between the results of our proxy MR 2 -GAN and the state-of-the-art methods on image-to-image translation, super-resolution and image inpainting tasks.

In every task, our model generates diverse images of high quality, while existing methods with the reconstruction loss do not.inpainting task.

Also, the MR loss is more sensitive to the number of samples K generated for each input.

In SRGAN experiments, for example, both methods converge reliably with K = 24, while the MR loss often diverges at K = 12.

Although the MR loss is simpler and can be trained in an end-to-end manner, its applicability is rather limited compared to the proxy MR loss.

We provide generated samples for each configuration in the appendix.

Following , we quantitatively measure diversity and realism of generated images.

We evaluate our methods on Pix2Pix-Cityscapes, SRGAN-CelebA and GLCIC-CelebA tasks.

For each (method, task) pair, we generate 20 images from each of 300 different inputs using the trained model.

As a result, the test sample set size is 6,000 in total.

For diversity, we measure the average LPIPS score BID48 .

Among 20 generated images per input, we randomly choose 10 pairs of images and compute conditional LPIPS values.

We then average the scores over the test set.

For realism, we conduct a human evaluation experiment from 33 participants.

We present a real or fake image one at a time and ask participants to tag whether it is real or fake.

The images are presented for 1 second for SRGAN/GLCIC and 0.5 second for Pix2Pix, as done in .

We calculate the accuracy of identifying fake images with averaged F-measure F and use 2(1 − F ) as the realism score.

The score is assigned to 1.0 when all samples are completely indistinguishable from real images and 0 when the evaluators make no misclassification.

There are eight configurations of our methods in total, depending on the MLE (Gaussian and Laplace), the number of statistics (one and two) and the loss type (MR and proxy MR).

We test all variants for the Pix2Pix task and only Gaussian proxy MR 2 loss for the others.

We compare with base models in every task and additionally BicycleGAN for the Pix2Pix task.

We use the official implementation by the authors with minimal modification.

BicycleGAN has not been applied to image inpainting and super-resolution tasks, for which we do not report its result.

Table 1 summarizes the results.

In terms of both realism and diversity, our methods achieve competitive performance compared to the base models, sometimes even better.

For instance, in Pix2Pix-Cityscapes, three of our methods, Gaussian MR 1 , Gaussian proxy MR 1 , and Gaussian proxy MR 2 , significantly outperform BicycleGAN and Pix2Pix+noise in both measures.

Without exception, the diversity scores of our methods are far greater than those of the baselines while maintaining competitive realism scores.

These results confirm that our methods generate a broad spectrum of high-quality images from a single input.

Interestingly, MR 1 and proxy MR 1 loss generate comparable or even better outputs compared to MR 2 and proxy MR 2 loss.

That is, matching means could be enough to guide GAN training in many tasks.

It implies that adding more statistics (i.e. adding more guidance signal to the model) may be helpful in some cases, but generally may not improve the performance.

Moreover, additional errors may arise from predicting more statistics, which could degrade the GAN training.

Table 1 : Quantitative evaluation on three (method, dataset) pairs.

Realism is measured by 2(1 − F ) where F is the averaged F-measure of identifying fake by human evaluators.

Diversity is scored by the average of conditional LPIPS values.

In both metrics, the higher the better.

In all three tasks, our methods generate highly diverse images with competitive or even better realism.

In this work, we pointed out that there is a mismatch between the GAN loss and the conventional reconstruction losses.

As alternatives, we proposed a set of novel loss functions named MR loss and proxy MR loss that enable conditional GAN models to accomplish both stability of training and multimodal generation.

Empirically, we showed that our loss functions were successfully integrated with multiple state-of-the-art models for image translation, super-resolution and image inpainting tasks, for which our method generated realistic image samples of high visual fidelity and variability on Cityscapes and CelebA dataset.

There are numerous possible directions beyond this work.

First, there are other conditional generation tasks that we did not cover, such as text-to-image synthesis, text-to-speech synthesis and video prediction, for which our methods can be directly applied to generate diverse, high-quality samples.

Second, in terms of statistics matching, our methods can be extended to explore other higher order statistics or covariance.

Third, using the statistics of high-level features may capture additional correlations that cannot be represented with pixel-level statistics.

We elaborate on the algorithms of all eight variants of our methods in detail from Algorithm 5 to Algorithm 4.

The presented algorithms assume a single input per update, although we use mini-batch training in practice.

Also, we use non-saturating GAN loss, − log D(x,ỹ) BID7 .For Laplace MLEs, the statistics that we compute are median and MAD.

Unlike mean, however, the gradient of median is defined only in terms of the single median sample.

Therefore, a naive implementation would only calculate the gradient for the median sample, which is not effective for training.

Therefore, we use a special trick to distribute the gradients to every sample.

In proxy MR loss, we first calculate the difference between the predicted median and the sample median, and then add it to samplesỹ 1:K to set the target values t 1:K .

We consider t 1:K as constants so that the gradient is not calculated for the target values (this is equivalent to Tensor.detach() in PyTorch and tf.stop gradient() in TensorFlow).

Finally, we calculate the loss between the target values and samples, not the medians.

We use the similar trick for the MR loss.

Algorithm 1 Generator update in Gaussian MR 1 -GAN Require: Generator G, discriminator D, MR loss coefficient λ MR Require: input x, ground truth y, the number of samples K 1: for i = 1 to K do 2: DISPLAYFORM0 Algorithm 2 Generator update in Gaussian MR 2 -GAN Require: Generator G, discriminator D, MR loss coefficient λ MR Require: input x, ground truth y, the number of samples K 1: for i = 1 to K do 2: DISPLAYFORM1 Algorithm 3 Generator update in Laplace MR 1 -GAN Require: Generator G, discriminator D, MR loss coefficient λ MR Require: input x, ground truth y, the number of samples K 1: for i = 1 to K do 2: DISPLAYFORM2 Algorithm 4 Generator update in Laplace MR 2 -GAN Require: Generator G, discriminator D, MR loss coefficient λ MR Require: input x, ground truth y, the number of samples K 1: for i = 1 to K do 2: DISPLAYFORM3 Algorithm 5 Generator update in Gaussian proxy MR 1 -GAN Require: Generator G, discriminator D, pre-trained predictor P , proxy MR loss coefficient λ pMR Require: input x, ground truth y, the number of samples K 1: for i = 1 to K do 2: DISPLAYFORM4 Algorithm 6 Generator update in Gaussian proxy MR 2 -GAN Require: Generator G, discriminator D, pre-trained predictor P , proxy MR loss coefficient λ pMR Require: input x, ground truth y, the number of samples K 1: for i = 1 to K do 2: DISPLAYFORM5 Algorithm 7 Generator update in Laplace proxy MR 1 -GAN Require: Generator G, discriminator D, pre-trained predictor P , proxy MR loss coefficient λ pMR Require: input x, ground truth y, the number of samples K 1: for i = 1 to K do 2: DISPLAYFORM6 Algorithm 8 Generator update in Laplace proxy MR 2 -GAN Require: Generator G, discriminator D, pre-trained predictor P , proxy MR loss coefficient λ pMR Require: input x, ground truth y, the number of samples K 1: for i = 1 to K do 2: DISPLAYFORM7 We use PyTorch for the implementation of our methods.

In every experiment, we use AMSGrad optimizer BID32 with LR = 10 −4 , β 1 = 0.5, β 2 = 0.999.

We use the weight decay of a rate 10 −4 and the gradient clipping by a value 0.5.

In case of proxy MR-GAN, we train the predictor until it is overfitted, and use the checkpoint with the lowest validation loss.

The weight of GAN loss is fixed to 1 in all cases.

Convergence speed.

Our methods need more training steps (about 1.5×) to generate high-quality images compared to those with the reconstruction loss.

This is an expectable behavior because our methods train the model to generate a much wider range of outputs.

Training stability.

The proxy MR loss is similar to the reconstruction loss in terms of training stability.

Our methods work stably with a large range of hyperparameter λ.

For example, the coefficient of the proxy MR loss can be set across several orders of magnitude (from tens to thousands) with similar results.

However, as noted, the MR loss is unstable compared to the proxy MR loss.

Our Pix2Pix variant is based on the U-net generator from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix.• Noise input: We concatenate Gaussian noise tensors of size H × W × 32 at the 1 × 1, 2 × 2, 4 × 4 feature map of the decoder.

Each element in the noise tensors are independently sampled from N (0, 1).• Input normalization: We normalize the inputs so that each channel has a zero-mean and a unit variance.• Batch sizes: We use 16 for the discriminator and the predictor and 8 for the generator.

When training the generator, we generate 10 samples for each input, therefore its total batch size is 80.• Loss weights: We set λ MR = λ pMR = 10.

For the baseline, we use 1 loss as the reconstruction loss and set λ 1 = 100.• Update ratio: We update generator once per every discriminator update.

Our SRGAN variant is based on the PyTorch implementation of SRGAN from https://github.com/zijundeng/SRGAN.• Noise input: We concatenate Gaussian noise tensor of size H × W × 16 at each input of the residual blocks of the generator, except for the first and last convolution layers.

Each element in the noise tensors are independently sampled from N (0, 1).• Input normalization: We make 16 × 16 × 3 input images' pixel values lie between -1 and 1.

We do not further normalize them with their mean and standard deviation.• Batch sizes: We use 32 for the discriminator and the predictor and 8 for the generator.

When training the generator, we generate 24 samples for each input, and thus its total batch size is 192.• Loss weights: We set λ MR = 20 and λ pMR = 2400.

For the baseline, we use 2 loss as the reconstruction loss and set λ 2 = 1000.• Update ratio: We update generator five times per every discriminator update.

We built our own PyTorch implementation of the GLCIC model.• Noise input: We concatenate Gaussian noise tensor of size H × W × 32 at each input of the first and second dilated convolution layers.

We also inject the noise to the convolution layer before the first dilated convolution layer.

Each element in the noise tensors are independently sampled from N (0, 1).• Input resizing and masking: We use square-cropped CelebA images and resize them to 128 × 128.

For masking, we randomly generate a hole of size between 48 and 64 and fill it with the average pixel value of the entire training dataset.• Input normalization: We make 128 × 128 × 3 input images' pixel values lie between 0 and 1.

We do not further normalize them with their mean and standard deviation.• Batch sizes: We use 16 for the discriminator and the predictor and 8 for the generator.

When training the generator, we generate 12 samples for each input, therefore its total batch size is 96.• Loss weights: For GLCIC, we tested Gaussian proxy MR 2 loss and MR lossess.

We successfully trained GLCIC with the Gaussian proxy MR 2 loss using λ pMR = 1000.

However, we could not find any working setting for the MR loss.

For the baseline model, we use 2 loss for the reconstruction loss and set λ 2 = 100.• Update ratio: We update generator three times per every discriminator update.

Our MR and proxy MR losses have preventive effect on the mode collapse.

FIG1 shows toy experiments of unconditional generation on synthetic 2D data which is hard to learn with GANs due to the mode collapse.

We train a simple 3-layer MLP with different objectives.

When trained only with GAN loss, the model captures only one mode as shown in figure 4b.

Adding 2 loss cannot fix this issue either as in FIG1 .

In contrast, all four of our methods ( FIG1 ) prevent the mode collapse and successfully capture all eight modes.

Notice that even the simpler variants MR 1 and proxy MR 1 losses effectively keep the model from the mode collapse.

Intuitively, if the generated samples are biased toward a single mode, their statistics, e.g. mean or variance, deviate from real statistics.

Our methods penalize such deviations, thereby reducing the mode collapse significantly.

Although we restrict the scope of this paper to conditional generation tasks, these toy experiments show that our methods have a potential to mitigate the mode collapse and stabilize training even for unconditional generation tasks.

D GENERATED SAMPLES The first rows of following images are composed of input, ground-truth, predicted mean, sample mean, predicted variance, and sample variance.

The other rows are generated samples.

To begin with, 1 loss is decomposed as follows: DISPLAYFORM0 To minimize 1 loss, we need the gradient w.r.t.

G(x, z) to be zero for all x and z that p(x) > 0 and p(z) > 0.

Note that this is a sufficient condition for minimum since the 1 loss is convex. .

If every real data belongs to the interval of conditional median, then the generator can be optimal for both GAN loss and 1 loss.

For instance, assume that there are only two discrete values of y possible for any given x, say −1 and 1, with probability 0.5 for each.

Then the interval of median becomes [−1, 1], thus any G(x, z) in the interval [−1, 1] minimizes the 1 loss to 1.

If the generated distribution is identical to the real distribution, i.e. generating −1 and 1 with the probability of 0.5, the generator is optimal w.r.t.

both GAN loss and 1 loss.

However, we note that such cases hardly occur.

In order for such cases to happen, for any x, every y with p(y|x) > 0 should be the conditional median, which is unlikely to happen in natural data such as images.

Therefore, the set of optimal generators for 1 loss is highly likely to be disjoint with the optimal set for GAN loss.

We present some more experiments on different combinations of loss functions.

The following results are obtained in the same manner as section D.2 and section D.3.

FIG8 shows the results when we train the Pix2Pix model only with our loss term (without the GAN loss).

The samples of MR 1 and proxy MR 1 losses are almost similar to those with the reconstruction loss only.

Since there is no reason to generate diverse images, the variances of the samples are near zero while the samples are hardly distinguishable from the output of the predictor.

On the other hand, MR 2 and proxy MR 2 losses do generate diverse samples, although the variation styles are different from one another.

Intriguingly, MR 2 loss incurs high-frequency variation patterns as the sample variance of MR 2 loss is much closer than proxy MR 2 loss to the predicted variance.

In the case of proxy MR 2 loss, where the attraction for diversity is relatively mild, the samples show low-frequency variation.

Another experiment that we carry out is about the joint use of all loss terms: GAN loss, our losses and reconstruction loss.

Specifically, we use the following objective with varying λ Rec from 0 to FIG0 shows the results of the experiments.

As λ Rec increases, the sample variance reduces.

This confirms again that the reconstruction is the major cause of loss of variability.

However, we find one interesting phenomenon regarding the quality of the samples.

The sample quality deteriorates up to a certain value of λ Rec , but gets back to normal as λ Rec further increases.

It implies that either proxy MR loss or the reconstruction loss can find some high-quality local optima, but the joint use of them is not desirable.

Figure 12: Joint use of GAN, proxy MR 2 , and reconstruction losses.

Each image is presented in the same manner as FIG8 .

We fix the coefficients of GAN and proxy MR 2 losses and test multiple values for the coefficient of reconstruction loss.

@highlight

We prove that the mode collapse in conditional GANs is largely attributed to a mismatch between reconstruction loss and GAN loss and introduce a set of novel loss functions as alternatives for reconstruction loss.

@highlight

The paper proposes a modification to the traditional conditional GAN objective in order to promote diverse, multimodal generation of images. 

@highlight

This paper proposes an alternative to L1/L2 errors that are used to augment adversarial losses when training conditional GANs.
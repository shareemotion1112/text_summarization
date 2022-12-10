We propose a simple yet highly effective method that addresses the mode-collapse problem in the Conditional Generative  Adversarial  Network (cGAN).

Although conditional distributions are multi-modal (i.e., having many modes) in practice, most cGAN approaches tend to learn an overly simplified distribution where an input is always mapped to a single output regardless of variations in latent code.

To address such issue, we propose to explicitly regularize the generator to produce diverse outputs depending on latent codes.

The proposed regularization is simple, general, and can be easily integrated into most conditional GAN objectives.

Additionally, explicit regularization on generator allows our method to control a balance between visual quality and diversity.

We demonstrate the effectiveness of our method on three conditional generation tasks: image-to-image translation, image inpainting, and future video prediction.

We show that simple addition of our regularization to existing models leads to surprisingly diverse generations, substantially outperforming the previous approaches for multi-modal conditional generation specifically designed in each individual task.

The objective of conditional generative models is learning a mapping function from input to output distributions.

Since many conditional distributions are inherently ambiguous (e.g. predicting the future of a video from past observations), the ideal generative model should be able to learn a multi-modal mapping from inputs to outputs.

Recently, Conditional Generative Adversarial Networks (cGAN) have been successfully applied to a wide-range of conditional generation tasks, such as image-to-image translation (Isola et al., 2017; Wang et al., 2018; Zhu et al., 2017a) , image inpainting (Pathak et al., 2016; Iizuka et al., 2017 ), text-to-image synthesis (Huang et al., 2017; Hong et al., 2018) , video generation (Villegas et al., 2017) , etc..

In conditional GAN, the generator learns a deterministic mapping from input to output distributions, where the multi-modal nature of the mapping is handled by sampling random latent codes from a prior distribution.

However, it has been widely observed that conditional GANs are often suffered from the mode collapse problem (Salimans et al., 2016; , where only small subsets of output distribution are represented by the generator.

The problem is especially prevalent for highdimensional input and output, such as images and videos, since the model is likely to observe only one example of input and output pair during training.

To resolve such issue, there has been recent attempts to learn multi-modal mapping in conditional generative models (Zhu et al., 2017b; Huang et al., 2018) .

However, they are focused on specific conditional generation tasks (e.g. image-toimage translation) and require specific network architectures and objective functions that sometimes are not easy to incorporate into the existing conditional GANs.

In this work, we introduce a simple method to regularize the generator in conditional GAN to resolve the mode-collapse problem.

Our method is motivated from an observation that the mode-collapse happens when the generator maps a large portion of latent codes to similar outputs.

To avoid this, we propose to encourage the generator to produce different outputs depending on the latent code, so as to learn a one-to-one mapping from the latent codes to outputs instead of many-to-one.

Despite the simplicity, we show that the proposed method is widely applicable to various cGAN architectures and tasks, and outperforms more complicated methods proposed to achieve multi-modal conditional generation for specific tasks.

Additionally, we show that we can control a balance between visual quality and diversity of generator outputs with the proposed formulation.

We demonstrate the effectiveness of the proposed method in three representative conditional generation tasks, where most existing cGAN approaches produces deterministic outputs: Image-to-image translation, image inpainting and video prediction.

We show that simple addition of the proposed regularization to the existing cGAN models effectively induces stochasticity from the generator outputs.

Resolving the mode-collapse problem in GAN is an important research problem, and has been extensively studied in the standard GAN settings (Metz et al., 2017; Gulrajani et al., 2017; Salimans et al., 2016; Miyato et al., 2018) .

These approaches include unrolling the generator gradient update steps (Metz et al., 2017) , incorporating the minibatch statistics into the discriminator (Salimans et al., 2016) , employing the improved divergence measure to smooth the loss landscape of the discriminator (Gulrajani et al., 2017; Miyato et al., 2018) , etc.. Although these approaches have been successful in modeling unconditional data distribution to some extent, recent studies have reported that it is still not sufficient to resolve a mode-collapse problem in many conditional generative tasks, especially for high-dimensional input and output.

Recently, some approaches have been proposed to address the mode-collapse issue in conditional GAN.

Zhu et al. (2017b) proposed a hybrid model of conditional GAN and Variational Autoencoder (VAE) for multi-modal image-to-image translation task.

The main idea is designing the generator to be invertible by employing an additional encoder network that predicts the latent code from the generated image.

The similar idea has been applied to unsupervised image-to-image translation (Huang et al., 2018) and stochastic video generation (Lee et al., 2018) but with non-trivial task-specific modifications.

However, these approaches are designed to achieve multi-modal generation in each specific task, and there has been no unified solution that addresses the mode-collapse problem for general conditional GANs.

Recently, (Odena et al., 2018) proposed a method that regularizes the generator by clamping the generator Jacobian within a certain range.

Our method shares the similar motivation with (Odena et al., 2018) but employs a different objective function that simply maximizes the norm of the generator gradient with an optional upper-bound, which we found that works much more stable over a wide range of tasks with less number of hyper-parameters.

Consider a problem of learning a conditional mapping function G : X → Y, which generates an output y ∈ Y conditioned on the input x ∈ X .

Our goal is to learn a multi-modal mapping G : X ×Z → Y, such that an input x can be mapped to multiple and diverse outputs in Y depending on the latent factors encoded in z ∈ Z. To learn such multi-modal mapping G, we consider a conditional Generative Adversarial Network (cGAN), which learns both conditional generator G and discriminator D by optimizing the following adversarial objective: DISPLAYFORM0 Although conditional GAN has been proved to work well for many conditional generation tasks, it has been also reported that optimization of Eq. (1) often suffers from the mode-collapse problem, which in extreme cases leads the generator to learn a deterministic mapping from x to y and ignore any stochasticity induced by z. To address such issue, previous approaches encouraged the generator to learn an invertible mapping from latent code to output by E(G(x, z)) = z (Zhu et al., 2017b; Huang et al., 2018) .

However, incorporating an extra encoding network E into the existing conditional GANs requires non-trivial modification of network architecture and introduce the new training challenges, which limits its applicability to various models and tasks.

We introduce a simple yet effective regularization on the generator that directly penalizes its modecollapsing behavior.

Specifically, we add the following maximization objective to the generator: DISPLAYFORM1 where · indicates a norm and τ is a bound for ensuring numerical stability.

The intuition behind the proposed regularization is very simple: when the generator collapses into a single mode and produces deterministic outputs based only on the conditioning variable x, Eq. (2) approaches its minimum since G(x, z 1 ) ≈ G(x, z 2 ) for all z 1 , z 2 ∼ N (0, 1).

By regularizing generator to maximize Eq. (2), we force the generator to produce diverse outputs depending on latent code z.

Our full objective function can be written as: DISPLAYFORM2 where λ controls an importance of the regularization, thus, the degree of stochasticity in G. If G has bounded outputs through a non-linear output function (e.g. sigmoid), we remove the margin from Eq. (2) in practice and control its importance only with λ.

In this case, adding our regularization introduces only one additional hyper-parameter.

The proposed regularization is simple, general, and can be easily integrated into most existing conditional GAN objectives.

In the experiment, we show that our method can be applied to various models under different objective functions, network architectures, and tasks.

In addition, our regularization allows an explicit control over a degree of diversity via hyper-parameter λ.

We show that different types of diversity emerge with different λ.

Finally, the proposed regularization can be extended to incorporate different distance metrics to measure the diversity of samples.

We show this extension using distance in feature space and for sequence data.

Connection to Generator Gradient.

We show in Appendix A that the proposed regularization in Eq. (2) corresponds to a lower-bound of averaged gradient norm of G over [z 1 , z 2 ] as: DISPLAYFORM0 where γ(t) = tz 2 + (1 − t)z 1 is a straight line connecting z 1 and z 2 .

It implies that optimizing our regularization (LHS of Eq. (4)) will increase the gradient norm of the generator ∇ z G .It has been known that the GAN suffers from a gradient vanishing issue since the gradient of optimal discriminator vanishes almost everywhere ∇D ≈ 0 except near the true data points.

To avoid this issue, many previous works had been dedicated to smoothing out the loss landscape of D so as to relax the vanishing gradient problem Miyato et al., 2018; Gulrajani et al., 2017; Kurach et al., 2018) .

Instead of smoothing ∇D by regularizing discriminator, we increase ∇ z G to encourage G(x, z) to be more spread over the output space from the fixed z j ∼ p(z), so as to capture more meaningful gradient from D.Optimization Perspective.

We provide another perspective to understand how the proposed method addresses the mode-collapse problem.

For notational simplicity, here we omit the conditioning variable from the generator and focus on a mapping of latent code to output G θ : Z → Y.Let a mode M denotes a set of data points in an output space Y, where all elements of the mode have very small differences that are perceptually indistinguishable.

We consider that the mode-collapse happens if the generator maps a large portion of latent codes to the mode M.Under this definition, we are interested in a situation where the generator output G θ (z 1 ) for a certain latent code z 1 moves closer to a mode M by a distance of via a single gradient update.

Then we show in Appendix B that such gradient update at z 1 will also move the generator outputs of neighbors in a neighborhood N r (z 1 ) to the same mode M. In addition, the size of neighborhood N r (z 1 ) can be arbitrarily large but is bounded by an open ball of a radius DISPLAYFORM1 , where θ t and θ t+1 denote the generator parameters before and after the gradient update, respectively.

, a single gradient update can cause the generator outputs for a large amount of latent codes to be collapsed into a mode M. We propose to shrink the size of such neighborhood by constraining DISPLAYFORM0 above some threshold τ > 0, therefore prevent the generator placing a large probability mass around a mode M. BicycleGAN (Zhu et al., 2017b) .

We establish an interesting connection of our regularization with Zhu et al. (2017b) .

Recall that the objective of BicycleGAN is encouraging an invertibility of a generator by minimizing z −E(G(z)) 1 .

By taking derivative with respect to z, it implies that optimal E will satisfy I = ∇ G E(G(z))∇ z G(z).

Because an ideal encoder E should be robust against spurious perturbations from inputs, we can naturally assume that the gradient norm of E should not be very large.

Therefore, to maintain invertibility, we expect the gradient of G should not be zero, i.e. ∇ z G(z) > τ for some τ > 0, which prevents a gradient of the generator being vanishing.

It is related to our idea that penalizes a vanishing gradient of the generator.

Contrary to BicycleGAN, however, our method explicitly optimizes a generator gradient to have a reasonably high norm.

It also allows us to control a degree of diversity with a hyper-parameter λ.

In this section, we demonstrate the effectiveness of the proposed regularization in three representative conditional generation tasks that most existing methods suffer from mode-collapse: imageto-image translation, image inpainting and future frame prediction.

In each task, we choose an appropriate cGAN baseline from the previous literature, which produces realistic but deterministic outputs, and apply our method by simply adding our regularization to their objective function.

We denote our method as DSGAN (Diversity-Sensitive GAN).

Note that both cGAN and DSGAN use the exactly the same networks.

Throughout the experiments, we use the following objective: DISPLAYFORM0 where L rec (G) is a regression (or reconstruction) loss to ensure similarity between a predictionŷ and ground-truth y, which is chosen differently by each baseline method.

Unless otherwise stated, DISPLAYFORM1 We provide additional video results at anonymous website: https://sites.google.com/view/iclr19-dsgan/.

In this section, we consider a task of image-to-image translation.

Given a set of training data (x, y) ∈ (X , Y), the objective of the task is learning a mapping G that transforms an image in domain X to another image in domain Y (e.g. sketch to photo image).As a baseline cGAN model, we employ the generator and discriminator architectures from BicycleGAN (Zhu et al., 2017b) for a fair comparison.

We evaluate the results on three datasets: label→image (Radim Tyleček, 2013), edge→photo (Zhu et al., 2016; Yu & Grauman, 2014) , map→image (Isola et al., 2017) .

For evaluation, we measure both the quality and the diversity of generation using two metrics from the previous literature.

We employed Learned Perceptual Image Path Similarity (LPIPS) (Zhang et al., 2018) to measure the diversity of samples, which computes the distance between generated samples using features extracted from the pretrained CNN.

Higher LPIPS score indicates more perceptual differences in generated images.

In addition, we use Fréchet Inception Distance (FID) (Heusel et al., 2017) to measure the distance between training and generated distributions using the features extracted by the inception network (Szegedy et al., 2015) .

The lower FID indicates that the two distributions are more similar.

To measure realism of the generated images, we also present human evaluation results using Amazon Mechanical Turk (AMT).

Detailed evaluation protocols are described in Appendix D.1.

To analyze the impact of our regularization on learning a multi-modal mapping, we first conduct an ablation study by varying the weights (λ) for our regularization.

We choose label→image dataset for this experiment, and summarize the results in FIG0 .

From the figure, it is clearly observed that the baseline cGAN (λ = 0) experiences a severe mode-collapse and produces deterministic outputs.

By adding our regularization (λ > 0), we observe that the diversity emerges from the generator outputs.

Increasing the λ increases LPIPS scores and lower the FID, which means that the generator learns a more diverse mapping from input to output, and the generated distribution is getting closer to the actual distribution.

If we impose too strong constraints on diversity with high λ, the diversity keeps increasing, but generator outputs become less realistic and deviate from the actual distribution as shown in high FID (i.e. we got FID= 191 and LPIPS=0.20 for λ = 20).

It shows that there is a natural trade-off between realism and diversity, and our method can control a balance between them by controlling λ.

Comparison with BicycleGAN (Zhu et al., 2017b) .

Next, we conduct comparison experiments with BicycleGAN (Zhu et al., 2017b) , which is proposed to achieve multi-modal conditional generation in image-to-image translation.

In this experiment, we fix λ = 8 for our method across all datasets and compare it against BicycleGAN with its optimal settings.

TAB1 summarizes the results.

Compared to the cGAN baseline, both our method and BicycleGAN are effective to learn multi-modal output distributions as shwon in higher LPIPS scores.

Compared to BicycleGAN, our method still generates much diverse outputs and distributions that are generally more closer to actual ones as shown in lower FID score.

In human evaluation on perceptual realism, we found that there is no clear winning method over others.

It indicates that outputs from all three methods are in similar visual quality.

Note that applying BicycleGAN to baseline cGAN requires non-trivial modifications in network architecture and obejctive function, while the proposed regularization can be simply integrated into the objective function without any modifications.

FIG1 illustrates generation results by our method.

See Appendix D.1.3 for qualitative comparisons to BicycleGAN and cGAN.We also conducted an experiment by varying a length of latent code z. Extension to High-Resolution Image Synthesis.

The proposed regularization is agnostic to the choice of network architecture and loss, therefore can be easily applicable to various methods.

To demonstrate this idea, we apply our regularization to the network of pix2pixHD (Wang et al., 2018) , which synthesizes a photo-realistic image of 1024 × 512 resolution from a segmentation label.

In addition to the network architectures, Wang et al. FORMULA0 incorporates a feature matching loss based on the discriminator as a reconstruction loss in Eq. (5).

Therefore, this experiment also demonstrates that our regularization is compatible with other choices of L rec .

TAB4 shows the comparison results on Cityscape dataset (Cordts et al., 2016) .

In addition to FID and LPIPS scores, we compute the segmentation accuracy to measure the visual quality of the generated images.

We compare the pixel-wise accuracy between input segmentation label and the predicted one from the generated image using DeepLab V3. (Chen et al., 2018) .

Since applying BicycleGAN to this baseline requires non-trivial modifications, we compared against the original BicycleGAN.

As shown in the table, applying our method to the baseline effectively increases the output diversity with a cost of slight degradation in quality.

Compared to BicycleGAN, our method generates much more visually plausible images.

FIG2 illustrates the qualitative comparison.

In this section, we demonstrate an application of our regularization to image inpainting task.

The objective of this task is learning a generator G : X → Y that takes an image with missing regions x ∈ X and generates a complete image y ∈ Y by inferring the missing regions.

For this task, we employ generator and discriminator networks from Iizuka et al. FORMULA0 as a baseline cGAN model with minor modification (See Appendix for more details).

To create a data for inpainting, we take 256 × 256 images of centered faces from the celebA dataset (Liu et al., 2015) and remove center pixels of size 128 × 128 which contains most parts of the face.

Similar to the image-to-image task, we employ FID and LPIPS to measure the generation performance.

Please refer Appendix D.2 for more details about the network architecture and implementation details.

In this experiment, we also test with an extension of our regularization using a different sample distance metric.

Instead of computing sample distances directly from the generator output as in Eq. FORMULA1 , we use the encoder features that capture more semantically meaningful distance between samples.

Similar to feature matching loss (Wang et al., 2018) , we use the features from a discriminator to compute our regularization as follow: DISPLAYFORM0 where D l indicates a feature extracted from l th layer of the discriminator D. We denote our methods based on Eq. (2) and Eq. (6) as DSGAN RGB and DSGAN FM , respectively.

Since there is no prior work on stochastic image inpainting to our best knowledge, we present comparisons of cGAN baseline along with our variants.

Analysis on Regularization.

We conduct both quantitative and qualitative comparisons of our methods and summarize the results in TAB5 and FIG8 , respectively.

As we observed in the previous section, adding our regularization induces multi-modal outputs from the baseline cGAN.

See FIG8 for qualitative impact of λ.

Interestingly, we can see that sample variations in DSGAN RGB tend to be in a low-level (e.g. global skin-color).

We believe that sample difference in color may not be appropriate for faces, since human reacts more sensitively to the changes in semantic features (e.g. facial landmarks) than just color.

Employing perceptual distance metric in our regularization leads to semantically more meaningful variations, such as expressions, identity, etc..

FIG8 : Qualitative comparisons of our variants.

We present one example for baseline as it produces deterministic outputs.

Analysis on Latent Space.

To further understand if our regularization encourages z to encode meaningful features, we conduct qualitative analysis on z. We employ DSGAN FM for this experiments.

We generate multiple samples across various input conditions while fixing the latent codes z. FIG4 illustrates the results.

We observe that our method generates outputs which are realistic and diverse depending on z. More interestingly, the generator outputs given the same z exhibit similar attributes (e.g. gaze direction, smile) but also context-specific characteristics that match the input condition (e.g. skin color, hairs).

It shows that our method guides the generator to learn meaningful latent factors in z, which are disentangled from the input context to some extent.

Given an input image with missing region (first row), we generate multiple faces by sampling different z (second-fifth rows).

Each row is generated from the same z, and exhibits similar face attributes.

In this section, we apply our method to a conditional sequence generation task.

Specifically, we consider a task of anticipating T future frames {x K+1 , x K+2 , ..., x K+T } ∈ Y conditioned on K previous frames {x 1 , x 2 , ..., x K } ∈ X .

Since both the input and output of the generator are sequences in this task, we simply modify our regularization by Figure 6: Stochastic video prediction results.

Given two input frames, we present three random samples generated by each method.

Compared to the baseline that produces deterministic outputs and SAVP that has limited diversity in KTH, our method generates diverse futures in both datasets.

TAB9 : Comparisons of cGAN, SAVP, and DSGAN (ours).

Diversity: pixel-wise distance among the predicted videos.

Sim max : largest cosine similarity between the predicted video and the ground truth.

Dist min : closest pixel-wise distance between the predicted video and the ground truth.

DISPLAYFORM0 where x 1:t represents a set of frames from time step 1 to t.

We compare our method against SAVP (Lee et al., 2018), which also addresses the multi-modal video prediction task.

Similar to Zhu et al. (2017b) , it employs a hybrid model of conditional GAN and VAE, but using the recurrent generator designed specifically for future frame prediction.

We take only GAN component (generator and discriminator networks) from SAVP as a baseline cGAN model and apply our regularization with λ = 50 to induce stochasticity.

We use |z| = 8 for all compared methods.

See Appendix D.3.2 for more details about the network architecture.

We conduct experiments on two datasets from the previous literature: the BAIR action-free robot pushing dataset (Ebert et al., 2017) and the KTH human actions dataset (Schuldt et al., 2004) .

To measure both the diversity and the quality, we generate 100 random samples of 28 future frames for each test video and compute the (a) Diversity (pixel-wise distance among the predicted videos) and the (b) Dist min (minimum pixel-wise distance between the predicted videos and the ground truth).

Also, for a better understanding of quality, we additionally measured the (c) Sim max (largest cosine similarity metric between the predicted video and the ground truth on VGGNet (Simonyan & Zisserman, 2015) feature space).

An ideal stochastic video prediction model may have higher Diversity, while having lower Dist min with higher Sim max so that a model still can predict similar to the ground truth as a candidate.

More details about evaluation metric are described in Appendix D.3.3.We present both quantitative and qualitative comparison results in TAB9 and FIG8 , respectively.

As illustrated in the results, both our method and SAVP can predict diverse futures compared to the baseline cGAN that produces deterministic outputs.

As shown in TAB9 , our method generates more diverse and realistic outputs than SAVP with much less number of parameters and simpler training procedures.

Interestingly, as shown in KTH results, SAVP still suffers from a mode-collapse problem when the training videos have limited diversity, whereas our method generally works well in both cases.

It shows that our method generalizes much better to various videos despite its simplicity.

In this paper, we investigate a way to resolve a mode-collapsing in conditional GAN by regularizing generator.

The proposed regularization is simple, general, and can be easily integrated into existing conditional GANs with broad classes of loss function, network architecture, and data modality.

We apply our regularization for three conditional generation tasks and show that simple addition of our regularization to existing cGAN objective effectively induces the diversity.

We believe that achieving an appropriate balance between realism and diversity by learning λ and τ such that the learned distribution matches an actual data distribution would be an interesting future work.

In this section, we provide a derivation of our regularization term from a true gradient norm of the generator.

Given arbitrary latent samples z 1 , z 2 , from gradient theorem we have DISPLAYFORM0 where γ is a straight line connecting z 1 and z 2 , where γ(0) = z 1 and γ(1) = z 2 .Apply expectation on both sides of FORMULA11 with respect to z 1 , z 2 from standard Gaussian distribution gives Eqn.

4: DISPLAYFORM1 B COLLAPSING TO A MODE AS A GROUP For notational simplicity, we omit the conditioning variable from the generator and focus on a mapping of latent code to output G θ : Z → Y where Y is the image space.

Definition B.1.

A mode M is a subset of Y satisfying max y∈M y − y * < α for some image y * and α > 0.

Let z 1 be a sample in latent space, we say z 1 is attracted to a mode M by from a gradient step if y * − G θt+1 (z 1 ) + < y * − G θt (z 1 ) , where y * ∈ M is an image in a mode, θ t and θ t+1 are the generator parameters before and after the gradient updates respectively.

In other words, we define modes as sets consisting of images that are close to some real images, and we consider a situation where the generator output G θt (z 1 ) at certain z 1 is attracted to a mode M by a single gradient update.

With Definition B.1, we are now ready to state and prove the following proposition.

Proposition B.1.

Suppose z 1 is attracted to the mode M by , then there exists a neighborhood N r (z 1 ) of z 1 such that z 2 is attracted to M by /2, for all z 2 ∈ N r (z 1 ).

The size of N r (z 1 ) can be arbitrarily large but is bounded by an open ball of radius r where DISPLAYFORM2 Proof.

Consider the following expansion.

DISPLAYFORM3 C ABLATION STUDY

Since the proposed regularization is not only limited to conditional GAN, we further analyze its impact on unconditional GAN.

To this end, we adopt synthetic datasets from (Srivastava et al., 2017; Metz et al., 2017) , a mixture of eight 2D Gaussian distributions arranged in a ring.

For unconditional generator and discriminator, we adopt the vanilla GAN implementation from (Srivastava et al., 2017) , and train the model with and without our regularization (λ = 0.1).

We follow the same evaluation protocols used in (Srivastava et al., 2017) .

It generates 2,500 samples by the generator, and counts a sample as high-quality if it is within three standard deviations of the nearest mode.

Then the performance is reported by 1) counting the number of modes containing at least one high-quality sample and 2) computing the portion of high-quality samples from all generated ones.

We summarize the qualitative and quantitative results (10-run average) in FIG8 and TAB9 , respectively.

As illustrated in FIG8 , we observe that vanilla GAN experiences a severe mode collapse, which puts a significant probability mass around a single output mode.

Contrary to results reported in (Srivastava et al., 2017) , we observed that the mode captured by the generator is still not close enough to the actual mode, resulted in 0 high-quality samples as shown in TAB9 .

On the other hand, applying our regularization effectively alleviates the mode-collapse problem by encouraging the generator to efficiently explore the data space, enabling the generator to capture much more modes compared to vanilla GAN setting.

Step 0kStep 4kStep 8kStep 12kStep 16kStep 20k

Vanilla GAN Ground truth ( = 0.02)Ground truth ( = 0.05)Step 0kStep FORMULA11 Step 16kStep 24kStep 32kStep FORMULA0 ).

In principle, our regularization can help the generator to cope with vanishing gradient from the discriminator to some extent, as it spreads out the generator landscape thus increases the chance to capture useful gradient signals around true data points.

To verify its impact, we simulate the vanishing gradient problem by training the baseline cGAN until it converges and retraining the model with our regularization while initializing the discriminator with the pre-trained weights.

Empirically we observed that the pre-trained discriminator can distinguish the real data and generated samples from the randomly initialized generator almost perfectly, and the generator experiences a severe vanishing gradient problem at the beginning of the training.

We use the image-to-image translation model on label→image dataset for this experiment.

In the experiment, we found that the generator converges to the FID and LPIPS scores of 52.32 and 0.16, respectively, which are close to the ones we achieved with the balanced discriminator (FID: 57.20, LPIPS: 0.18).

We observed that our regularization loss goes down very quickly in the early stage of training, which helps the generator to efficiently explorer its output space when the discriminator gradients are vanishing.

Together with the reconstruction loss, we found that it helps the generator to capture useful learning signals from the discriminator and learn both realistic and diverse modes in the conditional distribution.

This section provides additional experiment details and results that could not be accommodated in the main paper due to space restriction.

We are going to release the code and datasets upon the acceptance of the paper.

Here we provide a detailed descriptions for evaluation metrics and evaluation protocols.

Learned Perceptual Image Patch Similarity, LPIPS (Zhang et al., 2018) .

LPIPS score measures the diversity of the generated samples using the L1 distance of features extracted from pretrained AlexNet (Krizhevsky et al., 2012) .

We generate 20 samples for each validation image, and compute the average of pairwise distances between all samples generated from the same input.

Then we report the average of LPIPS scores over all validation images.

Fréchet Inception Distance, FID (Heusel et al., 2017) .

For each method, we compute FID score on the validation dataset.

For each input from the validation dataset, we sample 20 randomly generated output.

We take the generated images as a generated dataset and compute the FID score between the generated dataset and training dataset.

If the size of an image is different between the training dataset and generated dataset, we resize training images to the size of generated images.

We use the features from the final average pooling layer of the InceptionV3 (Szegedy et al., 2015) network to compute Fréchet Distance.

To compare the perceptual quality of generations among different methods, we conduct human evaluation via AMT.

We conduct sideby-side comparisons between our method and a competitor (i.e. baseline cGAN and BicycleGAN).

Specifically, we present two sets of images generated by each compared method given the same input condition, and ask turkers to choose the set that is visually more plausible and matches the input condition.

Each set has 6 randomly sampled images.

We collect answers over 100 examples for each dataset, where each question is answered by 5 unique turkers.

Qualitative Comparison.

We present the qualitative comparisons of various methods presented in TAB1 in the main paper.

FIG6 TAB4 .

We observe that the generation results from BicycleGAN suffers from low-visual quality, while our method is able to generate fine details of the objects and scene by exploiting the network for high-resolution image synthesis.

Input and GT Generated Images TAB4 .Analysis on Latent Space.

To better understand the latent space learned with the proposed regularization, we generate images in Cityscape dataset by interpolating two randomly sampled latent vectors by spherical linear interpolation (White, 2016) .

FIG8 illustrates the interpolation results.

As shown in the figure, the intermediate generation results are all reasonable and exhibit smooth transitions, which implies that the learned latent space has a smooth manifold.

Figure D: Interpolation in latent space of DSGAN on Cityscapes dataset.

We also present the comparison of interpolation results between DSGAN and BicycleGAN on maps → images dataset.

As shown in FIG8 , DSGAN generates meaningful and diverse predictions on ambiguous regions (e.g. forest on a map) and has a smooth transition from one latent code to another.

On contrary, the BicyceGAN does not show meaningful changes within the interpolations and sometimes has a sudden changes on its output (e.g. last generated image).

We also observe similar patterns across many examples in this dataset.

It shows an example that DSGAN learns better latent space than BicycleGAN.

In this section, we provide details of image inpainting experiment.

Network Architecture.

We employ the generator and discriminator networks from Iizuka et al. (2017) as baseline conditional GAN.

Our generator takes 256 × 256 image with the masked region x as an input and produces 256 × 256 prediction of the missing regionŷ as an output.

Then we combine the predicted image with the input by y = (1 − M ) x + M ŷ as an output of the network, where M is a binary mask indicating the missing region.

Then the combined output y is passed as an input to the discriminator.

We apply two modifications to the baseline model to achieve better generation quality.

First, compared to the original model that employs the Mean Squared Error (MSE) as a reconstruction loss L rec (G) in Eq. FORMULA7 , we apply the feature matching loss based on the discriminator (Wang et al., 2018) .

Second, compared to the original model that employs two discriminators applied independently to the inpainted region and entire image, we employ only one discriminator on the inpainted region but using patchGAN-style discriminator (Li & Wand, 2016) .

Please note that these modifications are to achieve better image quality but irrelevant to our regularization.

Analysis on Regularization.

First, we conduct qualitative analysis on how the proposed regularization controls a diversity of the generator outputs.

To this end, we train the model (DSGAN FM ) by varying the weights for our regularization, and present the results in FIG8 .

As already observed in Section 5.1, imposing stronger constraints on the generator by our regularization indeed increases the diversity in the generator outputs.

With small weights (e.g. λ = 2), we observe limited visual differences among samples, such as subtle changes in facial expressions or makeup.

By increasing λ (e.g. λ = 5), we can see that more meaningful diversity emerges such as hair-style, age, and even identity while maintaining the visual quality and alignment to input condition.

It shows more intuitively how our regularization can effectively help the model to discover more meaningful modes in the output space.

Ground truth Input Ground truth FIG8 :

Image inpainting results with different λ.

We observe more diversity emerges from the genrator outputs as we increase the weights for our regularization.

Analysis on Latent Space.

We further conduct a qualitative analysis on the learned latent space.

To verify that the model learns a continuous conditional distribution with our regularization, we conduct the interpolation experiment similar to the previous section.

Specifically, we sample two random latent codes from the prior distribution and generate images by linearly interpolating the latent code between two samples.

FIG8 illustrates the results.

As it shows, the generator outputs exhibit a smooth transition between two samples, while most intermediate samples also look realistic.

Figure G: Interpolation results on image inpainting task.

For each row, we sample the two latent codes (leftmost and rightmost images), and generate the images from the interpolated latent codes from one latent code to another.

In this section, we provide more details on network architecture, datasets and evaluation metrics on the video prediction task.

We measure the effectiveness of our method based on two real-world datasets: the BAIR action-free robot pushing dataset (Ebert et al., 2017) and the KTH human actions dataset (Schuldt et al., 2004) .

For both of the dataset, we provide two frames as the condition and train the model to predict 10 future frames (k = 2, T = 10 in Eq. FORMULA10 ).

In testing time, we run each model to predict 28 frames (k = 2, T = 28).

Following Lee et al. FORMULA0 , we used 64 × 64 frames for both datasets.

The details of data pre-processing are described in below.

BAIR Action-Free (Ebert et al., 2017) .

This dataset contains randomly moving robot arms on a table with a static background.

This dataset contains the diverse movement of a robot arm with a diverse set of objects.

We downloaded the pre-processed data provided by the authors (Lee et al., 2018) and used it directly for our experiment.

KTH (Schuldt et al., 2004) .

Each video in this dataset contains a single person in a static background performing one of six activities: walking, jogging, running, boxing, hand waving, and hand clapping.

We download the pre-processed videos from Villegas et al. FORMULA0 ; Denton & Fergus (2018) , which contains the frames with reasonable motions.

Following Jang et al. FORMULA0 , we added a diversity to videos by randomly skipping frames in a range of [1, 3] .

We compare our method against SAVP (Lee et al., 2018) which is proposed to achieve stochastic video prediction.

SAVP addresses a mode-collapse problem using the hybrid model of conditional GAN and VAE.

For a fair comparison, we construct our baseline cGAN by taking GAN component from SAVP (the generator and discriminator networks).

In below, we provide more details of the generator and discriminator architectures used in our baseline cGAN model.

The generator is based on the encoder-decoder network with convolutional LSTM (Xingjian et al., 2015) .

At each step, it takes a frame together with a latent code as inputs and produces the next frame as an output.

Contrary to the original SAVP that takes a latent code at each step to encode framewise stochasticity, we modified the generator to take one latent code per sequence that encodes the global dynamics of a video.

Then the discriminator takes the entire video as an input and produces a prediction on real or fake through 3D convolution operations.

We provide more details about the evaluation metrics used in our experiment.

For each test video, we generate 100 random samples with a length of 28 frames and evaluate the performance based on the following metrics:• Diversity: To measure the degree of diversity of the generated samples, we computed the frame-wise distance between each pair of the generated videos based on Mean Squared Error (MSE).

Then we reported the average distance over all pairs as a result.• Dist min : Following Lee et al. FORMULA0 , we evaluate the quality of generations by measuring the distance of the closest sample among the all generated ones to the ground-truth.

Specifically, for each test video, we computed the minimum distance between the generated samples and the ground-truth based on MSE and reported the average of the distances over the entire test videos.• Sim max : As another measure for the generation quality, we compute the similarity of the closest sample to the ground-truth similar to Dist min but using the cosine similarity of features extracted from VGGNet (Simonyan & Zisserman, 2015) .

We report the average of the computed similarity for entire test videos.

We present more video prediction results on both BAIR and KTH datasets in FIG8 , which corresponds to FIG8 in the main paper.

As discussed in the main paper, the baseline cGAN produces realistic but deterministic outputs, whereas both SAVP and our method generate far more diverse future predictions.

SAVP fails to generate the diverse outputs in KTH datasets, mainly because the dataset contains many examples with small motions.

On the contrary, our method generates diverse outputs in both datasets, since our regularization directly penalizes the mode-collapsing behavior and force the model to discover various modes.

Interestingly, we found that our model sometimes generates actions different from the input video when the motion in input frames are ambiguous (e.g. hand-clapping to hand-waving in the highlighted example).

It shows that our method can generate diverse and meaningful futures.

Both baseline cGAN and SAVP exhibit some noises in the predicted videos due to the failures in separating the moving foreground object from the background clutters (red arrow).

Compared to this, our method tends to generate more clear predictions on both foreground and background.

Interestingly, SAVP sometimes fail to predict interaction between objects (magenta arrows).

For instance, the objects on a table stay in the same position even after pushed by the robot arm.

On the other hand, our method is able to capture such interactions more precisely (blue arrows).

<|TLDR|>

@highlight

We propose a simple and general approach that avoids a mode collapse problem in various conditional GANs.

@highlight

The paper proposes a regularization term for the conditional GAN objective in order to promote diverse multimodal generation and prevent mode collapse.

@highlight

The paper proposes a method for generating diverse outputs for various conditional GAN frameworks including image-to-image translation, image-inpainting, and video prediction, which can be applied to various conditional synthesis frameworks for various tasks. 
Conditional generative adversarial networks (cGAN) have led to large improvements in the task of conditional image generation, which lies at the heart of computer vision.

The major focus so far has been on performance improvement, while there has been little effort in making cGAN more robust to noise.

The regression (of the generator) might lead to arbitrarily large errors in the output, which makes cGAN unreliable for real-world applications.

In this work, we introduce a novel conditional GAN model, called RoCGAN, which leverages structure in the target space of the model to address the issue.

Our model augments the generator with an unsupervised pathway, which promotes the outputs of the generator to span the target manifold even in the presence of intense noise.

We prove that RoCGAN share similar theoretical properties as GAN and experimentally verify that our model outperforms existing state-of-the-art cGAN architectures by a large margin in a variety of domains including images from natural scenes and faces.

Image-to-image translation and more generally conditional image generation lie at the heart of computer vision.

Conditional Generative Adversarial Networks (cGAN) (Mirza & Osindero, 2014) have become a dominant approach in the field, e.g. in dense 1 regression (Isola et al., 2017; Pathak et al., 2016; Ledig et al., 2017; BID1 Liu et al., 2017; Miyato & Koyama, 2018; Yu et al., 2018; Tulyakov et al., 2018) .

They accept a source signal as input, e.g. prior information in the form of an image or text, and map it to the target signal (image).

The mapping of cGAN does not constrain the output to the target manifold, thus the output can be arbitrarily off the target manifold (Vidal et al., 2017) .

This is a critical problem both for academic and commercial applications.

To utilize cGAN or similar methods as a production technology, we need to study their generalization even in the face of intense noise.

Similarly to regression, classification also suffers from sensitivity to noise and lack of output constraints.

One notable line of research consists in complementing supervision with unsupervised learning modules.

The unsupervised module forms a new pathway that is trained with the same, or different data samples.

The unsupervised pathway enables the network to explore the structure that is not present in the labelled training set, while implicitly constraining the output.

The addition of the unsupervised module is only required during the training stage and results in no additional computational cost during inference.

Rasmus et al. (2015) and Zhang et al. (2016) modified the original bottom-up (encoder) network to include top-down (decoder) modules during training.

However, in dense regression both bottom-up and top-down modules exist by default, and such methods are thus not trivial to extend to regression tasks.

Motivated by the combination of supervised and unsupervised pathways, we propose a novel conditional GAN which includes implicit constraints in the latent subspaces.

We coin this new model 'Robust Conditional GAN' (RoCGAN).

In the original cGAN the generator accepts a source signal and maps it to the target domain.

In our work, we (implicitly) constrain the decoder to generate samples that span only the target manifold.

We replace the original generator, i.e. encoder-decoder, with a two pathway module (see FIG0 ).

The first pathway, similarly to the cGAN generator, performs regression while the second is an autoencoder in the target domain (unsupervised pathway).

The two pathways share a similar network structure, i.e. each one includes an encoder-decoder network.

The weights of the two decoders are shared which promotes the latent representations of the two pathways to be semantically similar.

Intuitively, this can be thought of as constraining the output of our dense regression to span the target subspace.

The unsupervised pathway enables the utilization of all the samples in the target domain even in the absence of a corresponding input sample.

During inference, the unsupervised pathway is no longer required, therefore the testing complexity remains the same as in cGAN.

(a) The source signal is embedded into a low-dimensional, latent subspace, which is then mapped to the target subspace.

The lack of constraints might result in outcomes that are arbitrarily off the target manifold.

(b) On the other hand, in RoCGAN, steps 1b and 2b learn an autoencoder in the target manifold and by sharing the weights of the decoder, we restrict the output of the regression (step 2a).

All figures in this work are best viewed in color.

In the following sections, we introduce our novel RoCGAN and study their (theoretical) properties.

We prove that RoCGAN share similar theoretical properties with the original GAN, i.e. convergence and optimal discriminator.

An experiment with synthetic data is designed to visualize the target subspaces and assess our intuition.

We experimentally scrutinize the sensitivity of the hyper-parameters and evaluate our model in the face of intense noise.

Moreover, thorough experimentation with both images from natural scenes and human faces is conducted in two different tasks.

We compare our model with both the state-of-the-art cGAN and the recent method of Rick Chang et al. (2017) .

The experimental results demonstrate that RoCGAN outperform the baseline by a large margin in all cases.

Our contributions can be summarized as following:• We introduce RoCGAN that leverages structure in the target space.

The goal is to promote robustness in dense regression tasks.•

We scrutinize the model performance under (extreme) noise and adversarial perturbations.

To the authors' knowledge, this robustness analysis has not been studied previously for dense regression.• We conduct a thorough experimental analysis for two different tasks.

We outline how RoCGAN can be used in a semi-supervised learning task, how it performs with lateral connections from encoder to decoder.

Notation: Given a set of N samples, s (n) denotes the n th conditional label, e.g. a prior image; y (n) denotes the respective target image.

Unless explicitly mentioned otherwise || · || will declare an 1 norm.

The symbols L * define loss terms, while λ * denote regularization hyper-parameters optimized on the validation set.

Conditional image generation is a popular task in computer vision, dominated by approaches similar to cGAN.

Apart from cGAN, the method by Isola et al. (2017) , widely known as 'pix2pix', is the main alternative.

Pix2pix includes three modifications over the baseline cGAN: i) lateral skip connections between the encoder and the decoder network are added in the generator, ii) the discriminator accepts pairs of source/gt and source/model output images, iii) additional content loss terms are added.

The authors demonstrate how those performance related modifications can lead to an improved visual outcome.

Despite the improved performance, the problem with the additional guarantees remains the same.

That is we do not have any direct supervision in the process, since both the latent subspace and the projection are learned; the only supervision is provided by the ground-truth (gt) signal in the generator's output.

Adding regularization terms in the loss function can impose stronger supervision, thus restricting the output.

The most frequent regularization term is feature matching, e.g. perceptual loss (Ledig et al., 2017; Johnson et al., 2016) , or embeddings for faces (Schroff et al., 2015) .

Feature matching minimizes the distance between the projection of generated and ground-truth signals.

However, the pre-defined feature space is restrictive.

The method introduced by Salimans et al. (2016) performs feature matching in the discriminator; the motivation lies in matching the low-dimensional distributions created by the discriminator layers.

Matching the discriminator's features has demonstrated empirical success.

However, this does not affect the generator and its latent subspaces directly.

A new line of research that correlates with our goals is that of adversarial attacks BID6 Yuan et al., 2017; Samangouei et al., 2018) .

It is observed that perturbing input samples with a small amount of noise, often imperceptible to the human eye, can lead to severe classification errors.

There are several techniques to 'defend' against such attacks.

A recent example is the Fortified networks of Lamb et al. (2018) which uses Denoising Autoencoders (Vincent et al., 2008) to ensure that the input samples do not fall off the target manifold.

Kumar et al. (2017) estimate the tangent space to the target manifold and use that to insert invariances to the discriminator for classification purposes.

Even though RoCGAN share similarities with those methods, the scope is different since a) the output of our method is high-dimensional 2 and b) adversarial examples are not extended to dense regression 3 .Except for the study of adversarial attacks, combining supervised and unsupervised learning has been used for enhancing the classification performance.

In the Ladder network, Rasmus et al. (2015) modify a typical bottom-up network for classification by adding a decoder and lateral connections between the encoder and the decoder.

During training they utilize the augmented network as two pathways: i) labelled input samples are fed to the initial bottom-up module, ii) input samples are corrupted with noise and fed to the encoder-decoder with the lateral connections.

The latter pathway is an autoencoder; the idea is that it can strengthen the resilience of the network to samples outside the input manifold, while it improves the classification performance.

Our core goal consists in constraining the model's output.

Aside from deep learning approaches, such constraints in manifolds were typically tackled with component analysis.

Canonical correlation analysis (Hotelling, 1936) has been extensively used for finding common subspaces that maximally correlate the data (Panagakis et al., 2016) .

The recent work of Murdock et al. (2018) combines the expressiveness of neural networks with the theoretical guarantees of classic component analysis.

In this section, we elucidate our proposed RoCGAN.

To make the paper self-contained we first review the original conditional GAN model (sec. 3.1), before introducing RoCGAN (sec. 3.2).

Sequentially, we pose the modifications required in case of shortcut connections from the encoder to the decoder (sec. 3.3).

In sec. 3.4 we assess the intuition behind our model with synthetic data.

The core idea in RoCGAN is to leverage structure in the output space of the model.

We achieve that by replacing the single pathway in the generator with two pathways.

In the appendix, we study the theoretical properties of our method and prove that RoCGAN share the same properties as the original GAN .

GAN consist of a generator and a discriminator module commonly optimized with alternating gradient descent methods.

The generator samples z from a prior distribution p z , e.g. uniform, and tries to model the target distribution p d ; the discriminator D tries to distinguish between the samples generated from the model and the target (ground-truth) distributions.

Conditional GAN (cGAN) (Mirza & Osindero, 2014) extend the formulation by providing the generator with additional labels.

In cGAN the generator G typically takes the form of an encoder-decoder network, where the encoder projects the label into a low-dimensional latent subspace and the decoder performs the opposite mapping, i.e. from low-dimensional to high-dimensional subspace.

If we denote s the conditioning label and y a sample from the target distribution, the adversarial loss is expressed as: DISPLAYFORM0 by solving the following min-max problem: DISPLAYFORM1 where w G , w D denote the generator's and the discriminator's parameters respectively.

To simplify the notation, we drop the dependencies on the parameters and the noise z in the rest of the paper.

The works of Salimans et al. (2016) and Isola et al. (2017) demonstrate that auxiliary loss terms, i.e. feature matching and content loss, improve the final outcome, hence we consider those as part of the vanilla cGAN.

The feature matching loss (Salimans et al., 2016) is: DISPLAYFORM2 where π() extracts the features from the penultimate layer of the discriminator.

The final loss function for the cGAN is the following: DISPLAYFORM3 where λ c , λ π are hyper-parameters to balance the loss terms.

Just like cGAN, RoCGAN consist of a generator and a discriminator.

The generator of RoCGAN includes two pathways instead of the single pathway of the original cGAN.

The first pathway, referred as reg pathway henceforth, performs a similar regression as its counterpart in cGAN; it accepts a sample from the source domain and maps it to the target domain.

We introduce an additional unsupervised pathway, named AE pathway.

AE pathway works as an autoencoder in the target domain.

Both pathways consist of similar encoder-decoder networks 4 .

By sharing the weights of their decoders, we promote the regression outputs to span the target manifold and not induce arbitrarily large errors.

A schematic of the generator is illustrated in FIG1 .

The discriminator can remain the same as the cGAN: it accepts the reg pathway's output along with the corresponding target sample as input.

To simplify the notation below, the superscript 'AE' abbreviates modules of the AE pathway and 'G' modules of the reg pathway.

We denote G(s DISPLAYFORM0 ) the output of the reg pathway and DISPLAYFORM1 ) the output of the AE pathway.

The unsupervised module (autoencoder in the target domain) contributes the following loss term: DISPLAYFORM2 where f d denotes a divergence metric (in this work an 1 loss).Despite sharing the weights of the decoders, we cannot ensure that the latent representations of the two pathways span the same space.

To further reduce the distance of the two representations in the latent space, we introduce the latent loss term L lat .

This term minimizes the distance between the encoders' outputs, i.e. the two representations are spatially close (in the subspace spanned by the encoders).

The latent loss term is: DISPLAYFORM3 The final loss function of RoCGAN combines the loss terms of the original cGAN (eq. 3) with the additional two terms for the AE pathway: DISPLAYFORM4 As a future step we intend to replace the latent loss term L lat with a kernel-based method (Gretton et al., 2007) or a learnable metric for matching the distributions (Ma et al., 2018) .

The RoCGAN model of sec. 3.2 describes a family of networks and not a predefined set of layers.

A special case of RoCGAN emerges when skip connections are included in the generator.

In the next few paragraphs, we study the modification required, i.e. an additional loss term.

Skip connections are frequently used as they enable deeper layers to capture more abstract representations without the need of memorizing all the information.

Nevertheless, the effects of the skip connections in the representation space have not been thoroughly studied.

The lower-level representations are propagated directly to the decoder through the shortcut, which makes it harder to train the longer path (Rasmus et al., 2015) , i.e. the network excluding the skip connections.

This challenge can be implicitly tackled by maximizing the variance captured by the longer path representations.

To that end, we add a loss term that penalizes the correlations in the representations (of a layer) and thus implicitly encourage the representations to capture diverse and useful information.

We implement the decov loss introduced by BID2 : DISPLAYFORM0 where diag() computes the diagonal elements of a matrix and C is the covariance matrix of the layer's representations.

The loss is minimized when the covariance matrix is diagonal, i.e. it imposes a cost to minimize the covariance of hidden units without restricting the diagonal elements that include the variance of the hidden representations.

A similar loss is explored by Valpola (2015) , where the decorrelation loss is applied in every layer.

Their loss term has stronger constraints: i) it favors an identity covariance matrix but also ii) penalizes the smaller eigenvalues of the covariance more.

We have not explored this alternative loss term, as the decov loss worked in our case without the additional assumptions of the Valpola (2015).

We design an experiment on synthetic data to explore the differences between the original generator and our novel two pathway generator.

Specifically, we design a network where each encoder/decoder consists of two fully connected layers; each layer followed by a RELU.

We optimize the generators only, to avoid adding extra learned parameters.

The inputs/outputs of this network span a low-dimensional space, which depends on two independent variables x, y ∈ [−1, 1].

We've experimented with several arbitrary functions in the input and output vectors and they perform in a similar way.

We showcase here the case with input vector [x, y, e 2x ] and output vector [x + 2y + 4, e x + 1, x + y + 3, x + 2].

The reg pathway accepts the three inputs, projects it into a two-dimensional space and the decoder maps it to the target four-dimensional space.

We train the baseline and the autoencoder modules separately and use their pre-trained weights to initialize the two pathway network.

The loss function of the two pathway network consists of the L lat (eq. 5) and 2 content losses in the two pathways.

The networks are trained either till convergence or till 100, 000 iterations (batch size 128) are completed.

During testing, 6, 400 new points are sampled and the overlaid results are depicted in FIG2 ; the individual figures for each output can be found in the appendix.

The 1 errors for the two cases are: 9, 843 for the baseline and 1, 520 for the two pathway generator.

We notice that the two pathway generator approximates the target manifold better with the same number of parameters during inference.

Implementation details: To provide a fair comparison to previous cGAN works, our implementation is largely based on the conventions of Isola et al. FORMULA0 ; Salimans et al. (2016); Zhu et al. (2017) .

A 'layer' refers to a block of three units: a convolutional unit with a 4 × 4 kernel size, followed by Leaky RELU and batch normalization (Ioffe & Szegedy, 2015) .

To obtain RoCGAN, we augment a vanilla cGAN model as follows: i) we duplicate the encoder/decoder; ii) we share the decoder's weights in Each plot corresponds to the respective manifolds in the output vector; the first and third depend on both x, y (xyz plot), while the rest on x (xz plot).

The green color visualizes the target manifold, the red the baseline and the blue ours.

Even though the two models include the same parameters during inference, the baseline does not approximate the target manifold as well as our method.the two pathways; iii) we add the additional loss terms.

The values of the additional hyper-parameters are λ l = 25, λ ae = 100 and λ decov = 1; the common hyper-parameters with the vanilla cGAN, e.g. λ c , λ π , remain the same.

The decov loss is applied in the output of the encoder, which in our experimentation did minimize the correlations in the longer path.

The rest hyper-parameters remain the same as in the baseline.

We conduct a number of auxiliary experiments in the appendix.

Specifically, an ablation study on the significance and the sensitivity of the hyper-parameters is conducted; additional architectures are implemented, while we evaluate our model under more intense noise.

In addition, we extend the concept of adversarial examples in regression and verify that our model is more resilient to them than the baseline.

The results demonstrate that our model accepts a range of hyper-parameter values, while it is robust to additional sources of noise.

We experiment with two categories of images with significant applications: images from i) natural scenes and ii) faces.

In the natural scenes case, we constrain the number of training images to few thousand since frequently that is the scale of the labelled examples available.

The network used in the experiments below, dumped '4layer', consists of four layers in the decoder, while the decoder followed by four layers in the decoder.

Two inverse tasks, i.e. denoising and sparse inpainting, are selected for our quantitative evaluation.

During training, the images are corrupted, for the two tasks, in the following way: for denoising 25% of the pixels in each channel are uniformly dropped; for sparse inpainting 50% of the pixels are converted to black.

During testing, we evaluate the methods in two settings: i) similar corruption as they were trained, ii) more intense corruption, i.e. we drop 35% of the pixels in the denoising case and 75% of the pixels in the sparse inpainting case.

The widely used image quality loss (SSIM) (Wang et al., 2004 ) is used as a quantitative metric.

We train and test our method against the i) baseline cGAN, ii) the recent strong-performing OneNet (Rick Chang et al., 2017) .

OneNet uses an ADMM learned prior, i.e. it projects the corrupted prior images into the subspace of natural images to guide the ADMM solver.

In addition, we train an Adversarial Autoencoder (AAE) (Makhzani et al., 2015) as an established method capable of learning compressed representations.

Each module of the AAE shares the same architecture as its cGAN counterpart, while the AAE is trained with images in the target space.

During testing, we provide the ground-truth images as input and use the reconstruction for the evaluation.

In our experimental setting, AAE can be thought of as an upper performance limit of RoCGAN/cGAN for a given capacity (number of parameters).

We train the '4layer' baseline/RoCGAN with images from natural scenes, both indoors and outdoors.

The 4, 900 samples of the VOC 2007 Challenge BID4 form the training set, while the 10, 000 samples of tiny ImageNet BID3 ) consist the testing set.

The quantitative evaluation with SSIM is presented in Tab.

1.

OneNet (Rick Chang et al., 2017) does not perform as well as the baseline or our model.

From our experimentation this can be attributed to the projection to the manifold of natural images that is not trivial, however it is more resilient to additional noise than the baseline.

In both inverse tasks RoCGAN improve the baseline cGAN results by a margin of 0.05 (10 − 13% relative improvement).

When we apply additional corruption in the testing images, RoCGAN are more robust with a considerable improvement over the baseline.

This DISPLAYFORM0 Figure 4: Qualitative results (best viewed in color).

The first row depicts the target image, the second row the corrupted one (used as input to the methods).

The third row depicts the output of the baseline cGAN, while the outcome of our method is illustrated in the fourth row.

There are different evaluations visualized for faces: (a) denoising, (b) denoising with additional noise at test time, (c) sparse inpainting, (d) sparse inpainting with 75% black pixels.

For natural scenes the columns (e) and (f) denote the denoising and sparse inpainting results respectively.can be attributed to the implicit constraints of the AE pathway, i.e. the decoder is more resilient to approximating the target manifold samples.`````````M Table 1 : Quantitative results in the '4layer' network in both faces and natural scenes cases.

For both 'objects' we compute the SSIM.

In both denoising and sparse inpainting, the leftmost evaluation is the one with corruptions similar to the training, while the one on the right consists of samples with additional corruptions, e.g. in denoising 35% of the pixels are dropped.

In this experiment we utilize the MS-Celeb (Guo et al., 2016) as the training set (3, 4 million samples), and the whole Celeb-A (Liu et al., 2015) as the testing set (202, 500 samples).

The large datasets enable us to validate our model extensively in a wide range of faces.

We use the whole training set to train the two compared methods (Baseline-4layer and Ours-4layer) and the 4-layer AAE.

The results of the quantitative evaluation exist in table 1.

Our method outperforms both the baseline and OneNet by a significant margin; the difference increases when evaluated with more intense corruptions.

The reason that the sparse inpainting task appears to have a smaller improvement remains elusive; in the different architectures in the appendix our model has similar performance in the two tasks.

We include the AAE as an upper limit of the representation capacity of the architecture.

The AAE result specifies that with the given architecture the performance can be up to 0.866.

We introduce the Robust Conditional GAN (RoCGAN) model, a new conditional GAN capable of leveraging unsupervised data to learn better latent representations, even in the face of large amount of noise.

RoCGAN's generator is composed of two pathways.

The first pathway (reg pathway), performs the regression from the source to the target domain.

The new, added pathway (AE pathway) is an autoencoder in the target domain.

By adding weight sharing between the two decoders, we implicitly constrain the reg pathway to output images that span the target manifold.

In this following sections (of the appendix) we include additional insights, a theoretical analysis along with additional experiments.

The sections are organized as following:• In sec. B we validate our intuition for the RoCGAN constraints through the linear equivalent.• A theoretical analysis is provided in sec. C.• We implement different networks in sec. D to assess whether the performance gain can be attributed to a single architecture.• An ablation study is conducted in sec. E comparing the hyper-parameter sensitivity and the robustness in the face of extreme noise.

The FIG3 , 7, 8 include all the outputs of the synthetic experiment of the main paper.

As a reminder, the output vector is [x + 2y + 4, e x + 1, x + y + 3, x + 2] with x, y ∈ [−1, 1].

The exact nature and convergence properties of deep networks remain elusive (Vidal et al., 2017), however we can study the linear equivalent of deep methods to build on our intuition.

To that end, we explore the linear equivalent of our method.

Since the discriminator in RoCGAN can remain the same as in the baseline cGAN, we focus in the generators.

To perform the analysis on the linear equivalent, we simply drop the piecewise non-linear units in the generators.

The linear autoencoder (AE) has a similar structure; W (AE) l denote the respective parameters for the AE.

We denote with X the input signal, with Y the target signal andŶ the AE output,Ỹ the regression output.

Then:Ŷ DISPLAYFORM0 is the reconstruction of the autoencoder and DISPLAYFORM1 is the regression of the generator (reg pathway).

We define the auxiliary DISPLAYFORM2 and DISPLAYFORM3 .

Then Eq. 8 and 9 can be written as: DISPLAYFORM4 The AE approximates under mild condition robustly the target manifold of the data BID0 .

If we now define U D,(G) = U D , then the output of the generatorỸ spans the subspace of U D .Given that U D,(G) = U D , we constrain the output of the generator to lie in the subspaces learned with the AE.To illustrate how a projection to a target subspace can contribute to constraining the image, the following visual example is designed.

We learn a PCA model using one hundred thousand images from MS-Celeb; we do not apply any pose normalization or alignment (out of the paper's scope).

We maintain 90% of the variance.

In FIG7 we sample a random image from Celeb-A and downscale it 5 ; we use bi-linear interpolation to upscale it to the original dimensions.

We project and reconstruct both the original and the upscaled versions; note that the output images are similar.

This similarity illustrates how the linear projection forces both images to span the same subspace.

In the next few paragraphs, we prove that RoCGAN share the properties of the original GAN .

We derive the optimal discriminator and then compute the optimal value of L adv (G, D).

Proposition 1.

If we fix the generator G (reg pathway), the optimal discriminator is: DISPLAYFORM0 where p g is the model (generator) distribution.

Proof.

Since the generator is fixed, the goal of the discriminator is to maximize the L adv where: DISPLAYFORM1 To maximize the L adv , we need to optimize the integrand above.

We note that with respect to D the integrand has the form f (y) = a · log(y) + b · log(1 − y).

The function f for a, b ∈ (0, 1) as in our case, obtains a global maximum in a a+b , so: DISPLAYFORM2 with DISPLAYFORM3 thus L adv obtains the maximum with D * .Proposition 2.

Given the optimal discriminator D * the global minimum of L adv is reached if and only if p g = p d , i.e. when the model (generator) distribution matches the data distribution.

Proof.

From proposition 1, we have found the optimal discriminator as D * , i.e. the arg max D L adv .

If we replace the optimal value we obtain: DISPLAYFORM4 We add and subtract log(2) from both terms, which after few math operations provides: DISPLAYFORM5 where in the last row KL symbolizes the Kullback-Leibler divergence.

The latter one can be rewritten more conveniently with the help of the Jensen-Shannon (JSD) divergence as DISPLAYFORM6 The Jensen-Shannon divergence is non-negative and obtains the zero value only if FORMULA6 and has a global minimum (under the constraint that the discriminator is optimal) when p d = p g .

DISPLAYFORM7

In this section, we describe additional experimental results and details.

In addition to the SSIM metric, we use the 1 loss to measure the loss in the experiments of the main paper.

The results in table 2 confirm that RoCGAN outperform both compared methods.

The larger difference in the cases of more intense noise demonstrates that our model is indeed robust to additional cases not trained on.

Additional visualizations are provided in FIG0 .`````````M Table 2 : Quantitative results in the '4layer' network in both faces and natural scenes cases.

In this table, the 1 loss is reported.

In each task, the leftmost evaluation is the one with corruptions similar to the training, while the one on the right consists of samples with additional corruptions, e.g. in denoising 35% of the pixels are dropped.

Unless otherwise mentioned, the experiments in the following paragraphs are conducted in the face case, while the evaluation metrics remain the same as in the main paper, i.e. the noise during training/testing and the SSIM evaluation metric.

Table 4 : Additional quantitative results (SSIM, see main paper) for the following protocols: i) '5layer' network, ii) 50 thousand training images, iii) skip connections.

DISPLAYFORM0

To delineate further the performance of our model in different settings, we conduct an experiment with Imagenet (Russakovsky et al., 2015) , a large dataset for natural images.

We utilize the training set of Imagenet which consists of 1, 2 million images and its testset that includes 98 thousand images.

The experimental results are depicted in table 3.

The outcomes corroborate the experimental results of the main paper, as RoCGAN outperforms the cGAN in both tasks.

We note that the AAE works as the upper limit of the methods and denotes the representation power that the given encoder-decoder can reach.

To assess whether RoCGAN's improvement is network-specific, we implement different architectures including more layers.

The goal of this work is not to find the best performing architecture, thus we do not employ an exhaustive search in all proposed cGAN models.

Our goal is to propose an alternative model to the baseline cGAN and evaluate how this works in different networks.

We implement three additional networks which we coin '5layer', '6layer' and '4layer-skip'.

Those include five, six layers in the encoder/decoder respectively, while the '4layer-skip' includes a lateral connection from the output of the third encoding layer to the input of the second decoding layer.

The first two increase the capacity of the network, while the '4layer-skip' implements the modification for the skip case in the '4layer' network 6 .We evaluate these three networks as in the '4layer' network (main paper); the results are added in table 4.

Notice that both '5layer' and '6layer' networks improve their counterpart in the '4layer' case, however the '6layer' networks do not improve their '5layer' counterpart.

This can be partly attributed to the increased difficulty of training deeper networks without additional regularization techniques (He et al., 2016) .

In addition, we emphasize that the denoising and the sparse inpainting results cannot be directly compared, since they correspond to different a) types and b) amount of corruption in all evaluations.

Nevertheless, the improvement in the sparse inpainting with additional noise is impressive, given that the hyper-parameters are optimized for the denoising case (see sec E).The most critical observation is that in all cases our model consistently outperforms the baseline.

The FIG0 : Qualitative results; best viewed in color.

The first row depicts the ground-truth image, the second row the corrupted one (input to methods), the third the output of the baseline cGAN, the fourth illustrates the outcome of our method.

The four first columns are based on the protocol of '4layer' network, while the four rightmost columns on the protocol '4layer-50k'.

There are different evaluations visualized for faces: (a), (e) Denoising, (b), (f) denoising with augmented noise at test time, (c), (g) sparse inpainting, (d), (h) sparse inpainting with 75% black pixels.

DISPLAYFORM0 difference is increasing under additional noise during inference time with up to 15% performance improvement observed in the sparse inpainting case.

A side-benefit of our new model is the ability to utilize unsupervised data to learn the AE pathway.

Collecting unlabelled data in the target domain is frequently easier than finding pairs of corresponding samples in the two domains.

To that end, we test whether RoCGAN support such semi-supervised learning.

We randomly pick 50, 000 labelled images while we use the rest three million as unlabelled.

The 'label' in our case is the corrupted images.

The baseline model is trained with the labelled 50, 000 samples.

RoCGAN model is trained with 50, 000 images in the reg pathway, while the AE pathway with all the available (unlabelled) samples.

Table.

5 includes the quantitative results of the semi-supervised case.

As expected the performance in most experiments drops from the full training case, however we observe that the performance in RoCGAN decreases significantly less than cGAN ('baseline-4layer-50k').

In other words, RoCGAN can benefit greatly from additional examples in the target domain.

We hypothesize that this enables the AE pathway to learn a more accurate representation, which is reflected to the final RoCGAN outcome.

In our experimental setting every input image should be mapped (close) to its target image.

To assess the domain-specific performance for faces, we utilize the cosine distance distribution plot (CDDP).One of the core features in images of faces is the identity of the person.

We utilize the well-studied recognition embeddings (Schroff et al., 2015) to evaluate the similarity of the target image with the outputs of compared methods.

The ground-truth identities in our case are not available in the embeddings' space; we consider instead the target image's embedding as the ground-truth for each DISPLAYFORM0 Baseline-4layer-50k 0.788 0.747 0.798 0.617 Ours-4layer-50k 0.829 0.813 0.813 0.681 Table 5 : Quantitative results for the semi-supervised training of RoCGAN (sec. D.3).

The difference of the two models is increased (in comparison to the fully supervised case).

RoCGAN utilize the additional unsupervised data to improve the mapping between the domains even with less corresponding pairs.

DISPLAYFORM1 The plot is constructed as follows: For each pair of output and corresponding target image, we compute the cosine distance of their embeddings; the cumulative distribution of those distances is plotted.

Mathematically the distance of the n th pair is formulated as: DISPLAYFORM2 where o (n) denotes the output of each method, y (n) the respective target image and Φ is the function computing the embedding.

A perfect reconstruction per comparison, e.g. F(y (n) , y (n) ), would yield a plot of a Dirac delta around one; a narrow distribution centered at one denotes proximity to the target images' embeddings.

The plot with the CDDP is visualized in FIG0 for the '4layer' case as detailed in the main paper.

The results illustrate that AAE has embeddings that are closer to the target embeddings as expected; from the compared methods the RoCGAN outperform the cGAN in the proximity to the target embeddings.

All the images utilized in this work are resized to 64 × 64 × 3.

In the case of natural scenes, instead of rescaling the images during the training stage, we crop random patches in every iteration from the image.

We utilize the ADAM optimizer with a learning rate of 2 · 10 −5 for all our experiments.

The batch size is 128 for images of faces and 64 for the natural scenes.

In table 6 the details about the layer structure for the '4layer' generator are provided; the other networks include similar architecture as depicted in tables 7, 8.

The discriminator retains the same structure in all the experiments in this work (see table 9 ).

In the following paragraphs we conduct an ablation study to assess RoCGAN in different cases, i.e. effect of hyper-parameters, loss terms, additional noise.

Unless mentioned otherwise, the architecture used is the '4layer' network.

The experiments are in face denoising with the similarity metric (SSIM) and the setup similar to the main paper comparisons.

Table 9 : Details of the discriminator.

The discriminator structure remains the same throughout all the experiments in this work.

Our model introduces three new loss terms, i.e. L lat , L AE and L decov (in the case with skip) with respect to the baseline cGAN.

Understandably, those introduce three new hyper-parameters, which need to be validated.

The validation and selection of the hyper-parameters was done in a withheld set of images.

In the following paragraphs, we design an experiment where we scrutinize one hyperparameter every time, while we keep the rest in their selected value.

During our experimentation, we observed that the optimal values of these three hyper-parameters might differ per case/network, however in this manuscript the hyper-parameters remain the same throughout our experimentation.(a) (b) FIG0 : The layer schematics of the generators in case of (a) the '4layer-skip' case, (b) the '5layer' case.

The search space for each term is decided from its theoretical properties and our intuition.

For instance, the λ ae would have a value similar to the λ c 7 .

In a similar manner, the latent loss encourages the two streams' latent representations to be similar, however the final evaluation is performed in the pixel space, hence we assume that a value smaller than λ c is appropriate.

In table 10, we assess different values for the λ l .

The results demonstrate that values larger than 10 the results are similar, which dictates that our model resilient to the precise selection of the latent loss hyper-parameter.

Even though the best results are obtained with λ ae = 250, we select λ ae = 100 for our experiments.

The difference for the two choices is marginal, thus we choose the value 100 since resonates with our intuition (λ ae = λ c ).The third term of λ decov is scrutinized in the table 12.

In our experimentation, the λ decov has a different effect per experiment; based on the results of our validation we choose λ decov = 1 for our experiments.

In conclusion, the additional hyper-parameters introduced by our model can accept a range of values without affecting significantly the results.

Table 12 : Validation of λ decov values (hyper-parameter choices) in the '4layer-skip' network.

The network is more sensitive to the value of the λ decov than the λ l and λ ae .

To study further the significance of the four loss terms, we experiment with setting λ * = 0 alternatingly.

Apart from the '4layer' network, we implement the '4layer-skip' to assess the λ decov = 0 case.

The '4layer-skip' includes the same layers as the '4layer', however it includes a lateral connection from the encoder to the decoder.

The experimental results in table 13 confirm our prior intuition that the latent loss (L lat ) is the most crucial for our model in the no-skip case, but not as significant in the skip case.

In the skip case, the reconstruction losses in both pathways are significant.

Table 13 : Quantitative results (SSIM) for setting λ * = 0 alternatingly (sec. E.2).

In each column, we set the respective hyper-parameter to zero while keeping the rest fixed.

DISPLAYFORM0

To evaluate whether RoCGAN/cGAN are resilient to noise, we experiment with additional noise.

We include a baseline cGAN to the comparison to study whether their performance changes similarly.

Both networks are trained with the 25% noise (denosing task).We evaluate the performance in two cases: i) additional noise of the same type, ii) additional noise of different type.

For this experiment, we abbreviate noise as x/y where x depicts the amount of noise in denoising task (i.e. x% of the pixels in each channel are dropped with a uniform probability) and y the sparse inpainting task (i.e. y% black pixels).

In both cases, we evaluate the performance by incrementally increasing the amount of noise.

Specifically, both networks are tested in 25/0, 35/0, 50/0 for noise of the same type and 25/10, 25/20 and 25/25 for different type of noise.

We note the networks have not been trained on any of the testing noises other than the 25/0 case.

To illustrate the difference of performance between the two models, we accumulate the SSIM values of each case and divide them in 20 bins 8 .

In FIG0 , the histograms of each case are plotted.

We note that RoCGAN is much more resilient to increased or even unseen noise.

Qualitative results of the FIG0 : Qualitative figure illustrating the different noise levels (sec. E.3).

The first row depicts different target samples, while every three-row block, depicts the corrupted image, the baseline output and our output.

The blocks top down correspond to the 25%, 35%, 50% noise (25/0, 35/0 and 50/0).

The images in the first blocks are closer to the respective target images; as we increase the noise the baseline results deteriorate faster than RoCGAN outputs.

The readers can zoom-in to further notice the difference in the quality of the outputs.

difference are offered in FIG0 .

We consider that this improvement in the robustness in the face of additional noise is in its own a considerable improvement to the original cGAN.

Apart from testing in the face of additional noise, we explore the adversarial attacks.

Recent works BID6 Yuan et al., 2017; Samangouei et al., 2018) explore the robustness of (deep) classifiers.

Adversarial attacks modify the input image (of the network) so that the network misclassifies the image.

To the authors' knowledge, there has not been much investigation of adversarial attacks in the context of image-to-image translation or any other regression task.

However, if we consider adversarial examples as a perturbation of the original input, we explore whether this has any effect in the methods.

FIG0 .

The first row depicts different target samples, while every three-row block, depicts the corrupted image, the baseline output and our output.

The blocks top down correspond to the 25/10, 25/20, 25/25 cases (different type of testing noise).

The last block contains the most challenging noise in this work, i.e. both increased noise and of different type than the training noise.

Nevertheless, our model generates a more realistic image in comparison to the baseline.

Neither cGAN nor RoCGAN are designed to be robust in adversarial examples, however in this section we explore how adversarial examples can affect them.

We consider the FGSM method of BID6 as one of the first and simplest methods for generating adversarial examples.

In our case, we modify each source signal s as: DISPLAYFORM0 where η is the perturbation.

That is defined as: DISPLAYFORM1 with a hyper-parameter, y the target signal and L the loss.

In our case, we select the 1 loss as the loss between the target and the generated images.

We set = 0.01 following the original paper.

The evaluation is added in

@highlight

We introduce a new type of conditional GAN, which aims to leverage structure in the target space of the generator. We augment the generator with a new, unsupervised pathway to learn the target structure. 
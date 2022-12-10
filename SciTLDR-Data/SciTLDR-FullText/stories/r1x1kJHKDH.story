Unsupervised representation learning holds the promise of exploiting large amount of available unlabeled data to learn general representations.

A promising technique for unsupervised learning is the framework of Variational Auto-encoders (VAEs).

However, unsupervised representations learned by VAEs are significantly outperformed by those learned by supervising for recognition.

Our hypothesis is that to learn useful representations for recognition the model needs to be encouraged to learn about repeating and consistent patterns in data.

Drawing inspiration from the mid-level representation discovery work, we propose PatchVAE, that reasons about images at patch level.

Our key contribution is a bottleneck formulation in a VAE framework that encourages mid-level style representations.

Our experiments demonstrate that representations learned by our method perform much better on the recognition tasks compared to those learned by vanilla VAEs.

Due to the availability of large labeled visual datasets, supervised learning has become the dominant paradigm for visual recognition.

That is, to learn about any new concept, the modus operandi is to collect thousands of labeled examples for that concept and train a powerful classifier, such as a deep neural network.

This is necessary because the current generation of models based on deep neural networks require large amounts of labeled data (Sun et al., 2017) .

This is in stark contrast to the insights that we have from developmental psychology on how infants develop perception and cognition without any explicit supervision (Smith & Gasser, 2005) .

Moreover, the supervised learning paradigm is ill-suited for applications, such as health care and robotics, where annotated data is hard to obtain either due to privacy concerns or high cost of expert human annotators.

In such cases, learning from very few labeled images or discovering underlying natural patterns in large amounts of unlabeled data can have a large number of potential applications.

Discovering such patterns from unlabeled data is the standard setup of unsupervised learning.

Over the past few years, the field of unsupervised learning in computer vision has followed two seemingly different tracks with different goals: generative modeling and self-supervised learning.

The goal of generative modeling is to learn the probability distribution from which data was generated, given some training data.

A learned model can draw samples from the same distribution or evaluate the likelihoods of new data.

Generative models are also useful for learning compact representation of images.

However, we argue that these representations are not as useful for visual recognition.

This is not surprising since the task of reconstructing images does not require the bottleneck representation to sort out meaningful data useful for recognition and discard the rest; on the contrary, it encourages preserving as much information as possible for reconstruction.

In comparison, the goal in selfsupervised learning is to learn representations that are useful for recognition.

The standard paradigm is to establish proxy tasks that don't require human-supervision but can provide signals useful for recognition.

Due to the mismatch in goals of unsupervised learning for visual recognition and the representations learned from generative modeling, self-supervised learning is a more popular way of learning representations from unlabeled data.

However, fundamental limitation of this self-supervised paradigm is that we need to define a proxy-task that can mimic the desired recognition.

It is not always possible to establish such a task, nor are these tasks generalizable across recognition tasks.

In this paper, we take the first steps towards enabling the unsupervised generative modeling approach of VAEs to learn representations useful for recognition.

Our key hypothesis is that for a representation to be useful, it should capture just the interesting parts of the images, as opposed to everything in the images.

What constitutes an interesting image part has been defined and studied in earlier works that pre-date the end-to-end trained deep network methods Doersch et al., 2012; Juneja et al., 2013) .

Taking inspiration from these works, we propose a novel representation that only encodes such few parts of an image that are repetitive across the dataset, i.e., the patches that occur often in images.

By avoiding reconstruction of the entire image our method can focus on regions that are repeating and consistent across many images.

In an encoder-decoder based generative model, we constrain the encoder architecture to learn such repetitive parts -both in terms of representations for appearance of these parts (or patches in an image) and where these parts occur.

We formulate this using variational auto-encoder (β-VAEs) (Kingma & Welling, 2013; Matthey et al., 2017) , where we impose novel structure on the latent representations.

We use discrete latents to model part presence or absence and continuous latents to model their appearance.

We present this approach, PatchVAE, in Section 3 and demonstrate that it learns representations that are much better for recognition as compared to those learned by the standard β-VAEs (Kingma & Welling, 2013; Matthey et al., 2017) .

In addition, we propose in Section 3.4 that losses that favor foreground, which is more likely to contain repetitive patterns, result in representations that are much better at recognition.

In Section 4, we present results on CIFAR100 (Krizhevsky et al., 2009) , MIT Indoor Scene Recognition (Quattoni & Torralba, 2009) , Places (Zhou et al., 2017) , and ImageNet (Deng et al., 2009 ) datasets.

Our contributions are as follows:

• We propose a novel patch-based bottleneck in the VAE framework that learns representations that can encode repetitive parts across images.

• We demonstrate that our method, PatchVAE, learns unsupervised representations that are better suited for recognition in comparison to traditional VAEs.

• We show that losses that favor foreground are better for unsupervised learning of representations for recognition.

• We perform extensive ablation analysis to understand the importance of different aspects of the proposed PatchVAE architecture.

Due to its potential impact, unsupervised learning (particularly for deep networks) is one of the most researched topics in visual recognition over the past few years.

Generative models such as VAEs (Kingma & Welling, 2013; Matthey et al., 2017; Kingma et al., 2016; Gregor et al., 2015) , PixelRNN (van den Oord et al., 2016) , PixelCNN (Gulrajani et al., 2016; Salimans et al., 2017) , and their variants have proven effective when it comes to learning compressed representation of images while being able to faithfully reconstruct them as well as draw samples from the data distribution.

GANs (Goodfellow et al., 2014; Radford et al., 2015; Zhu et al., 2017; Arjovsky et al., 2017) on the other hand, while don't model the probability density explicitly, can still produce high quality image samples from noise.

There has been work combining VAEs and GANs to be able to simultaneously learn image data distribution while being able to generate high quality samples from it (Khan et al., 2018; Larsen et al., 2015) .

Convolution sparse coding (Affara et al., 2018) is an alternative approach for reconstruction or image in-painting problems.

Our work complements existing generative frameworks in that we provide a structured approach for VAEs that can learn beyond low-level representations.

We show the effectiveness of the representations learned by our model by using them for standard visual recognition tasks.

There has been a lot of work in interpreting or disentangling representations learned using generative models such as VAEs (Matthey et al., 2017; Fraccaro et al., 2017; Kim & Mnih, 2018) .

However, there is little evidence of effectiveness of disentangled representations in visual recognition tasks.

In our work, we focus on incorporating inductive biases in these generative models (e.g., VAEs) such that they can learn representations better suited for visual recognition tasks.

A related, but orthogonal, line of work is self-supervised learning where a proxy task is designed to learn representation useful for recognition.

These proxy tasks vary from simple tasks like arranging patches in an image in the correct spatial order (Doersch et al., 2014; and arranging frames from a video in correct temporal order (Wang & Gupta, 2015; Pathak et al., 2017) , to more involved tasks like in-painting (Pathak et al., 2016) and context prediction (Noroozi & Favaro, 2016; Wang et al., 2017) .

We follow the best practices from this line of work for evaluating the learned representations.

Our encoder network computes a set of feature maps f using φ(x).

This is followed by 2 independent single layer networks -bottom network generates part visibility parameters Q V .

We combine Q V with output of top network to generate part appearance parameters Q A .

We sample zvis and zapp to constructẑ as described in Section 3.2 which is input to the decoder network.

We also visualize the corresponding priors for latents zapp and zvis in the dashed gray boxes.

Our work builds upon VAE framework proposed by Kingma & Welling (2013) .

We briefly review relevant aspects of the VAE framework and then present our approach.

Standard VAE framework assumes a generative model for data where first a latent z is sampled from a prior p(z) and then the data is generated from a conditional distribution G(x|z).

A variational approximation Q(z|x) to the true intractable posterior is introduced and the model is learned by minimizing the following negative variational lower bound (ELBO).

Q(z|x) is often referred to as an encoder as it can be viewed as mapping data to the the latent space, while G(x|z) is referred to as a decoder (or generator) that can be viewed as mapping latents to the data space.

Both Q and G are commonly paramterized as neural networks.

Fig. 1a shows the commonly used VAE architecture.

If the conditional G(x|z) takes a gaussian form, negative log likelihood in the first term of RHS of Eq. 1 becomes mean squared error between generator output x = G(x|z) and input data x. In the second term, prior p(z) is assumed to be a multi-variate normal distribution with zero-mean and diagonal covariance N (0, I) and the loss simplifies to

When G and Q are differentiable, entire model can be trained with SGD using reparameterization trick (Kingma & Welling, 2013) .

Matthey et al. (2017) propose an extension for learning disentangled representation by incorporating a weight factor β for the KL Divergence term yielding

VAE framework aims to learn a generative model for the images where the latents z represent the corresponding low dimensional generating factors.

The latents z can therefore be treated as image representations that capture the necessary details about images.

However, we postulate that representations produced by the standard VAE framework are not ideal for recognition as they are learned to capture all details, rather than capturing 'interesting' aspects of the data and dropping the rest.

This is not surprising since there formulation does not encourage learning semantic information.

For learning semantic representations, in the absence of any relevant supervision (as is available in self-supervised approaches), inductive biases have to be introduced.

Therefore, taking inspiration from works on unsupervised mid-level pattern discovery Doersch et al., 2012; Juneja et al., 2013) , we propose a formulation that encourages the encoder to only encode such few parts of an image that are repetitive across the dataset, i.e., the patches that occur often in images.

Since the VAE framework provides a principled way of learning a mapping from image to latent space, we consider it ideal for our proposed extension.

We chose β-VAEs for their simplicity and widespread use.

In Section 3.2, we describe our approach in detail and in Section 3.4 propose a modification in the reconstruction error computation to bias the error term towards foreground high-energy regions (similar to the biased initial sampling of patterns in ).

Given an image x, let f = φ(x) be a deterministic mapping that produces a 3D representation f of size h × w × d e , with a total of L = h × w locations (grid-cells).

We aim to encourage the encoder network to only encode parts of an image that correspond to highly repetitive patches.

For example, a random patch of noise is unlikely to occur frequently, whereas patterns like faces, wheels, windows, etc.

repeat across multiple images.

In order capture this intuition, we force the representation f to be useful for predicting frequently occurring parts in an image, and use just these predicted parts to reconstruct the image.

We achieve this by transforming f toẑ which encodes a set of parts at a small subset of L locations on the grid cells.

We refer toẑ as "patch latent codes" for an image.

Next we describe how we re-tool the β-VAE framework to learn these local latent codes.

We first describe our setup for a single part and follow it up with a generalization to multiple parts (Section 3.3).

Image Encoding.

Given the image representation f = φ(x), we would like to learn part representations at each grid location l (where l ∈ {1, . . .

, L}).

A part is parameterized by its appearance z app and its visibility z l vis (i.e., presence or absence of the part at grid location l).

We use two networks,

vis | f ) of the part parameters z app and z l vis respectively.

Since the mapping f = φ(x) is deterministic, we can re-write these distribu-

.

Therefore, given an image x the encoder networks estimate the posterior

Note that f is a deterministic feature map, whereas z app and z l vis are stochastic.

Image Decoding.

We utilize a generator or decoder network G, that given z vis and z app , reconstructs the image.

First, we sample a part appearanceẑ app (d p dimensional, continuous) and then sample part visibilitiesẑ l vis (L dimensional, binary) one for each location l from the posteriorŝ

Next, we construct a 3D representationẑ by placingẑ app at every location l where the part is present (i.e.,ẑ l vis = 1).

This can be implemented by a broadcasted product ofẑ app andẑ l vis .

We refer toẑ as patch latent code.

Again note that f is deterministic andẑ is stochastic.

Finally, a deconvolutional network takesẑ as input and generates an imagex.

This image generation process can be written aŝ

Since all latent variables (z l vis for all l and z app ) are independent of each other, they can be stacked as

This enables us to use a simplified the notation (refer to (4) and (5)):

Note that despite the additional structure, our model still resembles the setup of variational autoencoders.

The primary difference arises from: (1) use of discrete latents for part visibility, (2) patchbased bottleneck imposing additional structure on latents, and (4) feature assembly for generator.

Training.

We use the training setup of β-VAE and use the maximization of variational lower bound to train the encoder and decoder jointly (described in Section 3.1).

The posterior Q A , which captures the appearance of a part, is assumed to be a zero-mean Normal distribution with diagonal covariance N (0, I).

The posterior Q V , which captures the presence or absence a part, is assumed to be a Bernoulli distribution Bern z prior vis with prior z prior vis .

Therefore, the ELBO for our approach can written as (refer to (3)):

where, the D KL term can be expanded as:

Implementation details.

As discussed in Section 3.1, the first and second terms of the RHS of (8) can be trained using L2 reconstruction loss and reparameterization trick (Kingma & Welling, 2013) .

In addition, we also need to compute KL Divergence loss for part visibility.

Learning discrete probability distribution is a challenging task since there is no gradient defined to backpropagate reconstruction loss through the stochastic layer at decoder even when using the reparameterization trick.

Therefore, we use the relaxed-bernoulli approximation (Maddison et al., 2016; Agustsson et al., 2017) for training part visibility distributions z

, where (h, w) are spatial dimensions and d e is the number of channels.

Therefore, the number of locations

vis > 1, the part occurs at multiple locations in an image.

Since all these locations correspond to same part, their appearance should be the same.

To incorporate this, we take the weighted average of the part appearance feature at each location, weighted by the probability that the part is present.

Since we use the probability values for averaging the result is deterministic.

This operation is encapsulated by the Q A encoder (refer to Figure 1b) .

During image generation, we sampleẑ app once and replicate it at each location whereẑ l vis = 1.

During training, this forces the model to: (1) only predictẑ l vis = 1 where similar looking parts occur, and (2) learn a common representation for the part that occurs at these locations.

Note that z app can be modeled as a mixture of distributions (e.g., mixture of gaussians) to capture complicated appearances.

However, in this work we assume that the convolutional neural network based encoders are powerful enough to map variable appearance of semantic concepts to similar feature representations.

Therefore, we restrict ourselves to a single gaussian distribution.

Next we extend the framework described above to use multiple parts.

To use N parts, we use N × 2 encoder networks Q

vis parameterize the i th part.

Again, this can be implemented efficiently as 2 networks by concatenating the outputs together.

The image generator samplesẑ

vis from the outputs of these encoder networks and constructsẑ (i) .

We obtain the final patch latent codeẑ by concatenating allẑ

For this multiple part case, (6) can be written as:

Similarly, (8) and (9) can be written as:

The training details and assumptions of posteriors follow the previous section.

Figure 2: Concepts captured by parts: We visualize a few representative examples for several parts to qualitatively demonstrate the visual concepts captured by parts.

For each part, we crop image patches centered on the part location where it is predicted to be present.

Selected patches are sorted by part visibility probability as score.

We have manually selected a diverse set from the top 50 occurrences from the training images.

As visible, a single part may capture diverse set of concepts that are similar in shape or texture or occur in similar context, but belong to different categories.

We show which categories the patches come from.

The L2 reconstruction loss used for training β-VAEs (and other reconstruction based approaches) gives equal importance to each region of an image.

This might be reasonable for tasks like image compression and image de-noising.

However, for the purposes of learning semantic representations, not all regions are equally important.

For example, "sky" and "walls" occupy large portions of an image, whereas concepts like "windows," "wheels,", "faces" are comparatively smaller, but arguably more important.

To incorporate this intuition, we use a simple and intuitive strategy to weigh the regions in an image in proportion to the gradient energy in the region.

More concretely, we compute laplacian of an image to get the intensity of gradients per-pixel and average the gradient magnitudes in 8 × 8 local patches.

The weight multiplier for the reconstruction loss of each 8 × 8 patch in the image is proportional to the average magnitude of the patch.

All weights are normalized to sum to one.

We refer to this as weighted loss (L w ).

Note that this is similar to the gradient-energy biased sampling of mid-level patches used in ; Doersch et al. (2012) .

In Appendix 6.1, we show examples of weight masks for some of the images.

In addition, we also consider an adversarial training strategy from GANs to train VAEs as proposed by Larsen et al. (2015) , where the discriminator network from GAN implicitly learns to compare images and gives a more abstract reconstruction error for the VAE.

We refer to this variant by using 'GAN' suffix in experiments.

In Section 4, we demonstrate that the proposed weighted loss (L w ) is complementary to the discriminator loss from adversarial training, and these losses result in better recognition capabilities for both β-VAE and PatchVAE.

Datasets.

We evaluate our proposed model on CIFAR100 (Krizhevsky et al., 2009) , MIT Indoor Scene Recognition (Quattoni & Torralba, 2009) , and Places (Zhou et al., 2017) datasets.

Details of these datasets can be found in Appendix 6.2.

Learning paradigm.

In order to evaluate the utility of features learned for recognition, we setup the learning paradigm as follows: we will first train the model in an unsupervised manner on all the images other than test set images.

After that, we discard the generator network and use only part of the encoder network φ(x) to train a supervised model on the classification task of the respective dataset.

We study different training strategies for the classification stage as discussed later.

Training details.

In all experiments, we use the following architectures.

For CIFAR100, Indoor67, and Place205, φ(x) has a conv layer followed by two residual blocks (He et al., 2016) .

For ImageNet, φ(x) is a ResNet18 model (a conv layer followed by four residual blocks).

For all datasets, Q A and Q V have a single conv layer each.

For classification, we start from φ(x), and add a fully-connected layer with 512 hidden units and a final fully-connected layer as classifier.

More details can be found in Appendix 6.2 and 6.3.

During the unsupervised learning part of training, all methods are trained for 90 epochs for CIFAR100 and Indoor67, 2 epochs for Places205, and 30 epochs for ImageNet dataset.

All methods use ADAM optimizer for training, with initial learning rate of 1 × 10 −4 and a minibatch size of 128.

For relaxed bernoulli in Q V , we start with the temperature of 1.0 with an annealing rate of 3 × 10 −5 (details in (Agustsson et al., 2017) ).

For training the classifier, all methods use stochastic gradient descent (SGD) with momentum with a minibatch size of 128.

Initial learning rate is 1 × 10 −2 and we reduce it by a factor of 10 every 30 epochs.

All experiments are trained for 90 epochs for CIFAR100 and Indoor67, 5 epochs for Places205, and 30 epochs for ImageNet datasets.

Table 1 : Classification results on CIFAR100, Indoor67, and Places205.

We initialize the classification model with the representations φ(x) learned from unsupervised learning task.

The model φ(x) comprises of a conv layer followed by two residual blocks (each having 2 conv layers).

First column (called 'Conv1') corresponds to Top-1 classification accuracy with pre-trained model with the first conv layer frozen, second and third columns correspond to results with 3 conv layers and 5 conv layers frozen respectively.

Details in Section 4.1.

Baselines.

We use the β-VAE model (Section 3.1) as our primary baseline.

In addition, we use weighted loss and discriminator loss resulting in the β-VAE-* family of baselines.

We also compare against a BiGAN model from .

We use similar backbone architectures for encoder/decoder (and discriminator if present) across all methods, and tried to keep the number of parameters in different approaches comparable to the best of our ability.

Exact architecture details can be found in Appendix 6.3.

In Table 1 , we report the top-1 classification results on CIFAR100, Indoor67, and Places205 datasets for all methods with different training strategies for classification.

First, we keep all the pre-trained weights in φ(x) from the unsupervised task frozen and only train the two newly added conv layers in the classification network (reported under column 'Conv5').

We notice that our method (with different losses) generally outperforms the β-VAE counterpart by a healthy margin.

This shows that the representations learned by PatchVAE framework are better for recognition compared to β-VAEs.

Moreover, better reconstruction losses ('GAN' and L w ) generally improve both β-VAE and PatchVAE, and are complementary to each other.

Next, we fine-tune the last residual block along with the two conv layers ('Conv3' column).

We observe that PatchVAE performs better than VAE under all settings except the for CIFAR100 with just L2 loss.

However, when using better reconstruction losses, the performance of PatchVAE improves over β-VAE.

Similarly, we fine-tune all but the first conv layer and report the results in 'Conv1' column.

Again, we notice similar trends, where our method generally performs better than β-VAE on Indoor67 and Places205 dataset, but β-VAE performs better CIFAR100 by a small margin.

When compared to BiGAN, PatchVAE representations are better on all datasets ('Conv5') by a huge margin.

However, when fine-tuning the pre-trained weights, BiGAN performs better on two out of four datasets.

We also report results using pre-trained weights in φ(x) using supervised ImageNet ImageNet Results.

Finally, we report results on the large-scale ImageNet benchmark in Table 2 .

For these experiments, we use ResNet18 (He et al., 2016) architecture for all methods.

All weights are first learned using the unsupervised tasks.

Then, we fine-tune the last two residual blocks and train the two newly added conv layers in the classification network (therefore, first conv layer and the following two residual blocks are frozen).

We notice that PatchVAE framework outperforms β-VAE under all settings, and the proposed weighted loss helps both approaches.

Finally, the last row in Table 2 reports classification results of same architecture randomly initialized and trained end-to-end on ImageNet using supervised training for comparison.

We study the impact of various hyper-parameters used in our experiments.

For the purpose of this evaluation, we follow a similar approach as in the 'Conv5' column of Table 1 and all hyperparameters from the previous section.

We use CIFAR100 and Indoor67 datasets for ablation analysis.

Maximum number of patches.

Maximum number of parts N used in our framework.

Depending on the dataset, higher value of N can provide wider pool of patches to pick from.

However, it can also make the unsupervised learning task harder, since in a minibatch of images, we might not get too many repeat patches.

Table 3 (left) shows the effect of N on CIFAR100 and Indoor67 datasets.

We observe that while increasing number of patches improves the discriminative power in case of CIFAR100, it has little or negative effect in case of Indoor67.

A possible reason for this decline in performance for Indoor67 can be smaller size of the dataset (i.e., fewer images to learn).

Number of hidden units for a patch appearanceẑ app .

Next, we study the impact of the number of channels in the appearance featureẑ app for each patch (d p ).

This parameter reflects the capacity of individual patch's latent representation.

While this parameter impacts the reconstruction quality of images.

We observed that it has little or no effect on the classification performance of the base features.

Results are summarized in Table 3 (right) for both CIFAR100 and Indoor67 datasets.

Prior probability for patch visibility z prior vis .

In all our experiments, prior probability for a patch is fixed to 1/N , i.e., inverse of maximum number of patches.

The intuition is to encourage each location on visibility maps to fire for at most one patch.

Increasing this patch visibility prior will allow all patches to fire at the same location.

While this would make the reconstruction task easier, it will become harder for individual patches to capture anything meaningful.

Table 4 shows the deterioration of classification performance on increasing z prior vis .

Patch visibility loss weight β vis .

The weight for patch visibility KL Divergence has to be chosen carefully.

If β vis is too low, more patches can fire at same location and this harms the the learning capability of patches; and if β vis is too high, decoder will not receive any patches to reconstruct from and both reconstruction and classification will suffer.

Table 5 summarizes the impact of varying β vis .

We presented a patch-based bottleneck in a VAE framework that encourages learning useful representations for recognition.

Our method, PatchVAE, constrains the encoder architecture to only learn patches that are repetitive and consistent in images as opposed to learning everything, and therefore results in representations that perform much better for recognition tasks compared to vanilla VAEs.

We also demonstrate that losses that favor high-energy foreground regions of an image are better for unsupervised learning of representations for recognition.

6 APPENDIX 6.1 VISUALIZATION OF WEIGHTED LOSS Figure 3 shows an illustration of the reconstruction loss L w proposed in Section 3.4.

Notice that in first column, guitar has more weight that rest of the image.

Similarly in second, fourth and sixth columns that train, painting, and people are respectively weighed more heavily by L w than rest of the image; thus favoring capturing the foreground regions.

Image Laplacian

Figure 3: Masks used for weighted reconstruction loss Lw.

First row contains images randomly samples from MIT Indoor datatset.

Second and third rows have the corresponding image laplacians and final reconstruction weight masks respectively.

In the last row, we take the product of first and third row to highlight which parts of image are getting more attention while reconstruction.

6.2 DATASET AND TRAINING DETAILS CIFAR100 consists of 60000 32 × 32 color images in 100 classes, with 600 images per class.

There are 50000 training images and 10000 test images.

Indoor dataset contains 67 categories, and a total of 15620 images ('Indoor67').

Train and test subsets consist of 80 and 20 images per class respectively.

Places dataset has 2.5 millions of images with 205 categories ('Places205').

Finally, we report results on the large-scale ImageNet (Deng et al., 2009 ) dataset, which has ∼1.28M training and 50k validation images spanning 1000 categories.

The generator network has two deconv layers with batchnorm (Ioffe & Szegedy, 2015) and a final deconv layer with tanh activation.

When training with 'GAN' loss, the additional discriminator has four conv layers, two of with have batchnorm.

In this section, we share the exact architectures used in various experiments.

As discussed in Section 4, we evaluated our proposed model on CIFAR100, Indoor67, and Places205 datasets.

We resize and center-crop the images such that input image size for CIFAR100 datasets is 32 × 32 × 3 while for Indoor67 and Places205 datasets input image size is 64 × 64 × 3.

PatchVAE can treat images of various input sizes in exactly same way allowing us to keep the architecture same for different datasets.

In case of VAE and BiGAN however, we have to go through a fixed size bottleneck layer and hence architectures need to be a little different for different input image sizes.

Wherever possible, we have tried to keep the number of parameters in different architectures comparable.

Tables 6 and 7 show the architectures for encoders used in different models.

In the unsupervised learning task, encoder comprises of a fixed neural network backbone φ(x), that given an image of size h × w × 3 generated feature maps of size

This backbone architecture is common to different models discussed in the paper and consists of a single conv layer followed by 2 residual blocks.

We refer to this φ(x) as Resnet-9 and it is described as Conv1-5 layers in Table 10 .

Rest of the encoder architecture varies depending on the model in consideration and is described in the tables below.

Tables 8 and 9 show the architectures for decoders used in different models.

We use a pyramid like network for decoder where feature map size is doubled in consecutive layers, while number of channels is halved.

Final non-linearity used in each decoder is tanh.

@highlight

A patch-based bottleneck formulation in a VAE framework that learns unsupervised representations better suited for visual recognition.
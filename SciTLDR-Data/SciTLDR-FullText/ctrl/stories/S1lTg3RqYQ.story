Image-to-image translation has recently received significant attention due to advances in deep learning.

Most works focus on learning either a one-to-one mapping in an unsupervised way or a many-to-many mapping in a supervised way.

However, a more practical setting is many-to-many mapping in an unsupervised way, which is harder due to the lack of supervision and the complex inner- and cross-domain variations.

To alleviate these issues, we propose the Exemplar Guided & Semantically Consistent Image-to-image Translation (EGSC-IT) network which conditions the translation process on an exemplar image in the target domain.

We assume that an image comprises of a content component which is shared across domains, and a style component specific to each domain.

Under the guidance of an exemplar from the target domain we apply Adaptive Instance Normalization to the shared content component, which allows us to transfer the style information of the target domain to the source domain.

To avoid semantic inconsistencies during translation that naturally appear due to the large inner- and cross-domain variations, we introduce the concept of feature masks that provide coarse semantic guidance without requiring the use of any semantic labels.

Experimental results on various datasets show that EGSC-IT does not only translate the source image to diverse instances in the target domain, but also preserves the semantic consistency during the process.

Image-to-image (I2I) translation refers to the task of mapping an image from a source domain to a target domain, e.g. semantic maps to real images, gray-scale to color images, low-resolution to high-resolution images, and so on.

The recent advances in deep learning have greatly improved the quality of I2I translation methods for a number of applications, including super-resolution BID3 , colorization BID33 , inpainting BID26 , attribute transfer BID18 , style transfer BID4 , and domain adaptation BID8 BID22 .

Most of these works BID11 BID30 BID35 have been very successful in these cross-domain I2I translation tasks because they rely on large datasets of paired training data as supervision.

However, for many tasks it is not easy, or even possible, to obtain such paired data that show how an image in the source domain should be translated to an image in the target domain, e.g. in cross-city street view translation or male-female face translation.

For this Figure 2 : The x A to x AB translation procedure of our EGSC-IT framework.

1) Source domain image x A is fed into an encoder E A to compute a shared latent code z A and is further decoded to a common high-level content representation c A .

2) Meanwhile, x A is also fed into a sub-network to compute feature masks m A .3) The target domain exemplar image x B is fed to a sub-network to compute affine parameters ?? B and ?? B for AdaIN .

4) The content representation c A is transferred to the target domain using m A , ?? B , ?? B , and is further decoded to an image x AB by target domain generator G B .unsupervised setting, BID34 proposed to use a cycle-consistency loss, which assumes that a mapping from domain A to B, followed by its reverse operation approximately yields an identity function, that is, F (G(x A )) ??? x A .

BID22 further proposed a shared-latent space constraint, which assumes that a pair of corresponding images (x A , x B ) from domains A and B respectively can be mapped to the same representation z in a shared latent space Z. Note that, all the aforementioned methods assume that there is a deterministic one-to-one mapping between the two domains, i.e. each image in A is translated to only a single image in B. By doing so, they fail to capture the multimodal nature of the image distribution within the target domain, e.g. different color and style of shoes in sketch-to-image translation and different seasons in synthetic-to-real street view translation.

In this work, we propose Exemplar Guided & Semantically Consistent I2I Translation (EGSC-IT) to explicitly address this issue.

As shown in concurrent works BID6 BID18 , we assume that an image is composed of two disentangled representations.

In our case, first a domain-shared representation that models the content in the image, and second a domain-specific representation that contains the style information.

However, for a multimodal domain with complex inner-variations, as the ones we target in this paper, e.g. street views of day-and-night or different seasons, it is difficult to have a single static representation which covers all variations in that domain.

Moreover, it is unclear which style (time-of-day/season) to pick during the image translation process.

To handle such multimodal I2I translations, some approaches BID0 BID6 BID18 incorporate noise vectors as additional inputs to the generator, but as shown in BID11 BID35 this could lead to mode collapsing issues.

Instead, we propose to condition the image translation process on an arbitrary image in the target domain, i.e. an exemplar.

By doing so, EGSC-IT does not only enable multimodal (i.e. many-to-many) image translations, but also allows for explicit control over the translation process, since by using different exemplars as guidance we are able to translate an input image into images of different styles within the target domain -see FIG0 .To instantiate this idea, we adopt the weight sharing architecture proposed in UNIT BID22 , but instead of having a single latent space shared by both domains, we propose to decompose the latent space into two components according to the two disentangled representations presented above.

That is, a domain-shared component that focuses on the image content, and a domain-specific component that captures the style information associated with the exemplar.

In our particular case, the domain-shared content component contains semantic information, such as the objects' category, shape and spatial layout, while the domain-specific style component contains the style information, such as the color and texture, to be translated from a target domain exemplar to an image in the source domain.

To realize this translation, we apply adaptive instance normalization (AdaIN) BID9 to the shared content component of the source domain image using the AdaIN parameters computed from the target domain exemplar.

However, directly applying AdaIN to the feature maps of the shared content component would mix up all objects and scenes in the image, Published as a conference paper at ICLR 2019 making the image translation prone to failure when an image contains diverse objects and scenes.

To tackle this problem, existing works BID5 BID8 BID20 BID24 use semantic labels as an additional form of supervision.

However, ground-truth semantic labels are not easy to obtain for most tasks as they require labor-intensive annotations.

Instead, to maintain the semantic consistency during image translation without using any semantic labels we propose to compute feature masks.

One can think of feature masks as attention modules that approximately decouple different semantic categories in an unsupervised way under the guidance of perceptual losses and adversarial loss.

In particular, one feature mask corresponding to a certain semantic category is applied to one feature map of the shared content component, and consequently the AdaIN for that channel is only required to capture and model the style difference for that category, e.g. sky's style in two domains.

To the best of our knowledge, this is the first line of work that addresses the semantic consistency issue under this setting.

See Fig. 2 for an overview of EGSC-IT.Our contribution is three-fold.

i) We propose a novel approach for the I2I translation task, which enables multimodal (i.e. many-to-many) mappings and allows for explicit style control over the translation process.

ii) We introduce the concept of feature masks for the unsupervised, multimodal I2I translation task, which provides coarse semantic guidance without using any semantic labels.

iii) Evaluation on different datasets show that our method is robust to mode collapse and can generate results with semantic consistency, conditioned on a given exemplar image.

I2I translation.

I2I translation is used to learn a mapping from one image (i.e. source domain) to another (i.e. target domain).

Recently, with the advent of generative models BID7 BID15 , there have been a lot of works on this topic.

BID11 proposed pix2pix to learn the mapping from input images to output images using a U-Net neural network in an adversarial way.

BID30 extended the method to pix2pixHD, to turn semantic label maps into high-resolution photo-realistic images.

BID35 extended pix2pix to BicycleGAN, which can model multimodal distributions and produce both diverse and realistic results.

All these methods, however, require paired training data as supervision which may be difficult or even impossible to collect in many scenarios, such as synthetic-to-real street view translation or face-to-cartoon translation BID28 .Recently, several unsupervised methods have been proposed to learn the mappings between two image collections without paired training data.

Note that, this is an ill-posed problem since there are infinitely many mappings existing between two unpaired image domains.

To address this ill-posed problem, different constraints have been added to the network to regularize the learning process BID13 BID22 BID28 BID32 BID34 .

One popular constraint is cycle-consistency, which enforces the network to learn deterministic mappings for various applications.

Going one step further, BID22 proposed a shared-latent space constraint which encourages a pair of images from different domains to be mapped to the same representation in the latent space.

In a similar vein, BID28 proposed to enforce a feature-level constraint with a latent embedding reconstruction loss.

However, we argue that these constraints are not well suited for complex domains with large inner-domain variations, as also mentioned in BID0 BID6 BID18 BID21 .

Unlike these methods, to address this problem we propose to add a target domain exemplar as guidance during image translation through AdaIN BID9 .

As explained in the previous section, the AdaIN technique is utilized to transfer the style component from the target domain exemplar to the shared content component of the source domain image.

This allows multimodal (i.e. many-to-many) translations and can produce images of desired styles with explicit control over the translation process.

Concurrent to our work, MUNIT , also proposed to use AdaIN to transfer style information from the target domain to the source domain.

Unlike MUNIT, before applying AdaIN to the shared content component we compute feature masks to decouple different semantic categories and preserve the semantic consistency during the translation process.

In particular, by applying feature masks to the feature maps of the shared content component, each channel can specialize and model the style difference only for a single semantic category, which is crucial when handling domains with complex scenes.

Style transfer.

Style transfer aims at transferring the style information from an exemplar image to a content image, while preserving the content information.

The seminal work by BID4 BID34 , UNIT BID22 , Augmented CycleGAN BID0 , CDD-IT (Gonzalez-Garcia et al., 2018), DRIT BID18 , MUNIT , EGSC-IT (Ours).

Disentangle & InfoFusion

Perceptual loss DISPLAYFORM0 proposed to transfer style information by matching the feature correlations, i.e. Gram matrices, in the convolutional layers of a deep neural network (DNN) following an iterative optimization process.

In order to improve the speed and flexibility, several feed-forward neural networks have been proposed.

BID9 proposed a simple but effective method, called AdaIN, which aligns the mean and variance of the content image features with those of the style image features.

BID19 proposed the whitening and coloring transform (WCT) algorithm, which directly matches the features' covariance in the content image to those in the given style image.

However, due to the lack of semantic consistency during translation, these stylization methods usually generate non-photorealistic images, suffering from the "spills over" problem BID24 .

To address this, semantic label maps are used as additional supervision to help style transfer between corresponding semantic regions BID5 BID20 BID24 .

Unlike these works, we propose to compute feature masks to approximately model such semantic information without using any semantic labels that are very hard to collect.

TAB0 summarizes the features of the most related works.

As can be seen, our method using the combination of AdaIN and feature masks under the guidance of perceptual loss is, to the best of our knowledge, the first to achieve multimodal I2I translations in the unsupervised setting with high semantic consistency, without requiring any ground-truth semantic labels.

Our goal is to learn a many-to-many mapping between two domains in an unsupervised way, which is guided by the style of an exemplar while retaining the semantic consistency at the same time.

For example, a synthetic street view image can be translated to either a day-time or night-time realistic scene, depending on the exemplar.

To realize this, similarly to concurrent works BID6 BID18 we assume that an image can be decomposed into two disentangled components.

In our case, that is, one modeling the shared content between domains, i.e. domain-shared content component, and another modeling the style information specific to exemplars in the target domain, i.e. domain-specific style component.

In what follows, we present our EGSC-IT framework, the architecture of its networks, and the learning procedure.

For simplicity, we present EGSC-IT in the A???B direction -see Fig. 2 .

Each image domain (i.e. source and target) is modeled by a VAE- GAN Larsen et al. (2016) , which includes an encoder E A , a generator G A , and a discriminator D A .

For the B???A direction, the translation process as well as the notation are analogous.

Weight sharing for domain-shared content.

To learn the content component of an image pair that is shared across source and target domains we employ the weight sharing strategy proposed in UNIT BID22 .

The latter assumes that the two domains, A and B, share a common latent space, and any image pair from the two domains, x A and x B , can be mapped to the same latent representation in this shared-latent space z. They achieve this by simply sharing the weights of the last layer in E A and E B as well as the first layer in G A and G B .

For more details about the weight-sharing strategy we refer the reader to the original UNIT paper.

Exemplar-based AdaIN for domain-specific style.

The shared content component contains semantic information, such as the objects' category, shape and spatial layout, but no style information, e.g. their color and texture.

Inspired by BID9 , who showed that AdaIN's affine parameters have a big influence on the output image's style, we propose to apply AdaIN to the shared content component before the decoding stage.

In particular, the exemplar from the target domain is fed to another network (see Fig. 2 , blue line) to compute a set of feature maps f B , which are expected to contain the style information of the target domain.

As in BID9 , means and variances are calculated for each channel of f B and used as AdaIN's affine parameters, DISPLAYFORM0 DISPLAYFORM1 DISPLAYFORM2 where ??(??) and ??(??) respectively denote a function to compute the mean and variance across spatial dimensions.

The shared content component is first normalized by these affine parameters, as shown in Eq. 2, and then decoded to a target-domain image using the target domain generator.

Since different affine parameters normalize the feature statistics in different ways, by using different exemplar images in the target domain as input we can translate an image in the source domain to different sub-styles in the target domain.

Therefore, EGSC-IT does not only allow for multimodal I2I translations, but at the same time enables the user to have explicit style control over the translation process.

Feature masks for semantic consistency.

Directly applying AdaIN to the shared content component does not give satisfying results.

The reason is that one channel in the shared content component is likely to contain information from multiple objects and scenes.

The difference of these objects and scenes between the two domains is not always uniform, due to the large inner-and cross-domain variations.

As such, applying AdaIN over a feature map with complex semantics is prone to mix styles of different objects and scenes together, hence failing to give semantically-consistent translations.

To tackle this problem, existing works use semantic labels as an additional form of supervision.

However, ground-truth semantic labels are not easy to obtain for most tasks as they require laborintensive annotations.

Instead, we propose to compute feature masks (see Fig. 2 , red line) to make an approximate estimation of semantic categories without using any ground-truth semantic labels.

The feature masks m A , which can be regarded as attention modules, are computed by applying a nonlinear activation function and a threshold to feature maps f A , i.e. DISPLAYFORM3 where ?? is a threshold and ?? is the sigmoid function.

Feature masks contain substantial semantic information, which can be used to retain the semantic consistency during translation, e.g.

The overall framework can be divided into several sub-networks 1 .

1) Two Encoders, E A and E B .

Each one consists of several strided convolutional layers and several residual blocks to compute the shared content component.

2) A feature mask network and an AdaIN network, F A and F B for A ??? B translation (vise versa for B ??? A) have the same architecture as the Encoder above except Published as a conference paper at ICLR 2019 for the weight-sharing layers.

3) Two Generators, G A and G B , are almost symmetric to the Encoders except that the up-sampling is done by transposed convolutional layers.

4) Two Discriminators, D A and D B , are fully-convolutional networks containing a stack of convolutional layers.

5) A VGG sub-network BID29 , V GG, that contains the first few layers (up to relu5_1) of a pre-trained VGG-19 BID29 , which is used to calculate perceptual losses.

Note that, although we use UNIT as our baseline framework to build upon, this is not a hard restriction.

In theory, UNIT can be replaced with any baseline framework with similar functionality.

The learning procedure of EGSC-IT contains VAEs, GANs, cycle-consistency and perceptual losses.

To make the training more stable, we first pre-train the feature mask network and AdaIN network for each domain separately within a VAE-GAN architecture, and use the encoder part as fixed feature extractors, i.e. F A and F B , for the remaining training.

The overall loss is shown in Eq. 3, DISPLAYFORM0 where the VAEs, GANs and cycle-consistency losses are identical to the ones used in BID22 .

The perceptual loss consists of the content loss captured by V GG19 feature maps ?? containing localized spatial information, and the style loss captured by the Gram matrix containing non-localized style information similar to BID4 BID12 , is as follows, DISPLAYFORM1 where ?? c and ?? s are the weights for content and style losses, which depend on the dataset domain variations and tasks.

The content loss L cA (E A , G A ) and style loss L sA (E A , G A ) are defined as, DISPLAYFORM2 We use the first convolutional layer of the five blocks in V GG19 to extract the feature maps.

L cB (E B , G B ) and L sB (E B , G B ) are defined likewise.

For the content losses L cA and L cB , a linear weighting scheme is adopted to help the network focus more on the high-level semantic information.

In both content and style losses we use the L1 distance, which in our experiments outperforms L2.

Now that we have introduced all losses, we can explain how these losses help to achieve I2I translation, multimodal translation, and semantic consistency.

I2I translation: L VAE , L GAN and L CC help to maintain the shared latent space by relating the two different domains and finding the optimal translation between the two in an unsupervised way.

Multimodal translation: L S and L GAN help to encourage x AB to look not only like the main mode of variation in domain B, but also like an exemplar from domain B, x B , since the domain space is actually supported by each data sample.

Semantic consistency: L C encourages the network to utilize the feature mask information for semantic consistency, without relying on hard correspondences between semantic labels as existing works do.

We evaluate EGSC-IT's translation ability, i.e. how well it generates domain-realistic-looking and semantically consistent images, both qualitatively and quantitatively on three tasks with progressively increasing visual complexity: 1) single-digit translation; 2) multi-digit translation; 3) street view translation.

We first perform an ablation study on various components of EGSC-IT on the single-digit translation task.

Then, we present results on more challenging translation tasks, and evaluate EGSC-IT quantitatively on the semantic segmentation task.

In supplementary material, we apply EGSC-IT to the face gender translation task and perform the ablation study on the street-view translation task.

Single-digit translation.

We set up a controlled experiment on the MNIST-Single dataset, which is created based on the handwritten digits dataset MNIST BID17 .

The MNIST-Single dataset consists of two different domains as shown in FIG4 .

For domain A of both training/test sets, the foreground and background are randomly set to black or white but different from each other.

For domain B of training set, the foreground and background for digits from 0 to 4 are randomly assigned a color from red, green, blue , and the foreground and background for digits from 5 to 9 are fixed to red and green, respectively.

For domain B of testing set, the foreground and background of all digits are randomly assigned a color from red, green, blue .

Such data imbalance is designed on purpose to test the translation diversity and generalization ability.

In particular, for diversity, we want to check whether a method would suffer from the mode collapse issue and translate the images to the dominant mode, i.e. (red, green), while for generalization, we want to check whether the model can be applied to new styles in the target domain that never appear in the training set, e.g. translate number 6 from black foreground and white background to blue foreground and red background.

We first analyze the importance of three main components of EGSC-IT, i.e. feature masks, AdaIN, and perceptual loss, on the MNIST-Single dataset.

As shown in FIG4 , EGSC-IT can successfully transfer the source image into the style of the exemplar image.

Ablating the feature mask from EGSC-IT, leads to incorrect foreground and background shape, indicating that feature masks can indeed provide semantic information to transfer the corresponding local regions.

Without AdaIN, the network suffers from the mode collapse issue in A???B translation, i.e. all samples are transferred to the dominant mode with red foreground and green background, indicating that the exemplar's style information can help the network to learn many-to-many mappings and avoid the mode collapse issue.

Without perceptual losses L P , colors of foreground and background are incorrect, which shows that perceptual losses can encourage the network to learn semantic knowledge, in this case foreground and background, without ground-truth semantic labels.

As for other I2I translation methods, CycleGAN BID34 and UNIT BID22 can only do deterministic image translations and suffer from mode collapse issue, such as white ??? green and black ??? red for CycleGAN in FIG4 .

MUNIT can successfully transfer the style of the exemplar image, but the foreground and background are mixed and the digit's shape is not kept well.

These qualitative observations are in accordance with the quantitative results in Tab.

2, where our full EGSC-IT obtains higher SSIM scores than all other alternatives.

In addition, we compare with other style transfer methods, Neural ST (Gatys et al., 2016) , AdaIN BID9 , and WCT BID19 .

In each case, we resize the input image to 512??512 resolution and choose the best performing hyper-parameters.

Note how style transfer methods can transfer the style successfully but fail to keep semantic consistency.

Quantitative results for style transfer methods are in supplementary material.

To verify EGSC-IT's ability to match the target domain distributions of real data and translated results, we visualize them using t-SNE embeddings BID25 in Fig. 5 .

The t-SNE embeddings are calculated from the translated images with PCA dimension reducing.

Our method can match the distributions well, while others either collapse to few modes or mismatch the distributions.

Multi-digit translation.

The MNIST-Multiple dataset is another controlled experiment designed to mimic the complexity in real-world scenarios.

It is used to test whether the network understands the semantics, i.e. digits, in an image and translates each digit accordingly.

Each image in MNISTMultiple contains all ten digits, which are randomly placed in 4??4 grids.

Two domains are designed: in domain A, the foreground and background are randomly set to black or white, but different from each other; in domain B, the background is randomly assigned to either black or white and each foreground digit is assigned to a certain color, but with a little saturation and lightness perturbation.

Our goal is to encourage the network to understand the semantic information, i.e. the different digits and backgrounds, when translate an image from domain A to domain B. That is, a successfully translated image should have the content of domain A, the digit class, and the style of domain B, the digit and background colors respectively.

This experiment is quite challenging, but we observe that our model can still obtain good results without the need for ground-truth semantic labels or paired data.

For example, in Figure 6 top row the digits 1,2,3,4,6 can be successfully translated given the criteria described above.

As seen in Fig. 6 , MUNIT can not translate the foreground color with semantic consistency, and the colors look more "fake".

Street view translation.

We carry out a synthetic ??? real experiment for street view translation between GTA5 BID27 and Berkeley Deep Drive (BDD) BID31 datasets.

The street view datasets are more complex than the digit ones (different illumination/weather conditions, complex environments).

As shown in Fig. 7 , our method can successfully translate the images from the source to the target domain according to the exemplar's style.

For small variations, e.g. day???day (first row), MUNIT can keep up, however for large variations, e.g. day???night and vice versa (second row), which is exactly the problem we examine in this paper, only EGSC-IT can successfully translate details like the proper sky color and illumination condition w.r.t.

the exemplar.

Similar to FCN-score used by BID11 , we also use the semantic segmentation performance to quantitatively evaluate the image translation DISPLAYFORM0

Published as a conference paper at ICLR 2019 quality.

We first translate images in GTA5 dataset to an arbitrary image in BDD dataset.

We only generate images of size 256??512 due to the limitation on GPU memory.

Then, we train a single-scale Deeplab model on the translated images and test it on BDD test set.

The mean Intersection over Union (mIoU) scores in Tab.

3 show that training with our translated synthetic images can improve the segmentation results, which indicates that our method can indeed reasonably translate the source GTA5 image to the target domain style with semantic consistency and reduce the domain difference successfully.

Since our method does not use any semantic segmentation labels nor paired data, there are some artifacts in the results for some hard cases.

For example, as to the street view translation, day???night and night???day (e.g. Fig. 7 bottom row) are more challenging than day???day (e.g. Fig. 7 top row) .

As a result, it is sometimes hard for our model to understand the semantics in such cases.

In the future, it would be interesting to extend our method to the semi-supervised setting in order to benefit from the presence of some fully-labeled data.

We introduced the EGSC-IT framework to learn a multimodal mapping across domains in an unsupervised way.

Under the guidance of an exemplar from the target domain, we showed how to combine AdaIN with feature masks in order to transfer the style of the exemplar to the source image, while retaining semantic consistency at the same time.

Numerous quantitative and qualitative results demonstrate the effectiveness of our method in this particular setting.

Face gender translation.

The Large-scale CelebFaces Attributes (CelebA) dataset BID23 is a largescale face attributes dataset with more than 200K celebrity images.

We divide the aligned face images into male and female domains, containing 84,434 and 118,165 images respectively.

We perform face gender translation on this dataset to show how the proposed method can be generalized to tasks with attributes as styles.

From Fig. 8 , we observe that EGSC-IT can translate the face gender successfully, and most importantly transfer the style of hair, skin and background according to the given exemplar image, unlike MUNIT.

In addition, we also provide the male???female face translation results matrix Fig. 9 .

We observe that the output image's content is consistent with the source image's and its style is consistent with the target image's.

Such observation can reflect how the latent space changes given different input images, i.e. the the content latent is only related to source image since the style information is combined in the decoder part through feature mask and AdaIN techniques.

Letter-digit translation.

To further evaluate the generalization ability of our method, we use EMNIST BID2 for three translation tasks: 1) black-white letter ??? colored digit; 2) black-white digit ??? colored letter; 3) black-white letter ??? colored letter.

EMNIST dataset, a set of handwritten character digits, has the same image format and dataset structure with MNIST dataset.

We apply the same process to EMNIST letter images as that for MNIST single-digit images in the main paper.

As shown in 10, our model trained with only single-digit data can successfully generalize to the letter data.

Single-digit translation.

For the quantitative comparison, we also report the SSIM socre of style transfer methods in Tab.

4.

In addition, the larger size version of the single-digit translation t-SNE embedding visualization as shown in FIG0 .

Multi-digit translation.

The setting of this experiment was presented in the main paper.

Here, we provide more details on the results.

As seen in FIG0 , both CycleGAN and UNIT can not translate the foreground and background color accordingly, and the colors in CycleGAN look more "fake".

This is due to the fact that CycleGAN and UNIT only learn a one-to-one mapping.

These observations are consistent with the SSIM score in Tab.

5, where both CycleGAN and UNIT have much lower SSIM scores.

Differently, MUNIT can not translate the foreground color with semantic consistency, and the colors look more "fake".

These observations are consistent with the visualization of t-SNE embeddings.

As to the SSIM score, MUNIT seems comparable to ours although visually it is performing worse.

The probable reason is that MUNIT can mostly translate the background successfully which occupies the majority of the image.

Street view translation.

We also provide a larger size version of the results for GTA5 ??? BDD translation as shown in Fig. 14.

In addition, we also provide the ablation study results in FIG0 .

We observe that: 1) removing feature mask will lead to color mismatches or inaccuracies (e.g. FIG0 (a) 1st row 3rd col); 2) removing AdaIN will reduce the model to unimodality (e.g. all images are translated to a sunny day with blue sky, see FIG0 (a) 4th col) since the output image's style is not guided by the exemplar image; 3) removing perceptual loss will lead to incorrect style (e.g. FIG0 (b) 5th col) and the color will spread even given the feature mask since there is no perceptual feedback during training (e.g. FIG0 (a) 5th col).

Published as a conference paper at ICLR 2019

The network architecture and training parameters are listed in Tab.

6.

We set the number of down-sampling and up-sampling convolutional layers n1 = 1 in single-digit translation and n1 = 3 in other translation experiments.

Following UNIT BID22 , the number of residual blocks in Encoder and Generator is set to n2 = 4 with one sharing layer, and the number of convolutional layers in discriminator is set to n3 = 5.

The threshold parameter ?? is used to adjust how much the feature mask affects the information flow.

Setting ?? = 0, i.e. using the feature maps (paper Fig. 2 top branch) as feature mask directly, leads to useful information being dropped out and artifacts in the results.

Setting ?? = 1, i.e. not using feature mask at all, leads to results without semantic consistency (see paper FIG4 ).

After experimenting with different values, we fixed it to 0.5.

We use the Adam BID14 optimizer with ??1 = 0.5 and ??2 = 0.999.

The learning rate is polynomially decayed with a power of 0.9, as mentioned in .

In order to keep training stable, we update encoder and generator 5 times, and discriminator 1 time in each iteration.

The loss weights in LUNIT are following BID22 , and ??c, ??s are chosen according to the dataset variations and tasks.

For data augmentation, we do left-right flip and random crop.

In addition, we set a low ??c for face gender translation, since we need to change the shape and add/remove hair in this translation task.

Published as a conference paper at ICLR 2019

<|TLDR|>

@highlight

We propose the Exemplar Guided & Semantically Consistent Image-to-image Translation (EGSC-IT) network which conditions the translation process on an exemplar image in the target domain.

@highlight

Discusses a core failing and need for I2I translation models.

@highlight

The paper explores the idea that an image has two components and applies an attention model where the feature masks that steer the translation process do not require semantic labels
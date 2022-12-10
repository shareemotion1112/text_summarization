We propose a novel method for incorporating conditional information into a generative adversarial network (GAN) for structured prediction tasks.

This method is based on fusing features from the generated and conditional information in feature space and allows the discriminator to better capture higher-order statistics from the data.

This method also increases the strength of the signals passed through the network where the real or generated data and the conditional data agree.

The proposed method is conceptually simpler than the joint convolutional neural network - conditional Markov random field (CNN-CRF) models and enforces higher-order consistency without being limited to a very specific class of high-order potentials.

Experimental results demonstrate that this method leads to improvement on a variety of different structured prediction tasks including image synthesis, semantic segmentation, and depth estimation.

Convolutional neural networks (CNNs) have demonstrated groundbreaking results on a variety of different learning tasks.

However, on tasks where high dimensional structure in the data needs to be preserved, per-pixel regression losses typically result in unstructured outputs since they do not take into consideration non-local dependencies in the data.

Structured prediction frameworks such as graphical models and joint CNN-graphical model-based architectures e.g. CNN-CRFs have been used for imposing spatial contiguity using non-local information BID13 BID2 BID24 .

The motivation to use CNN-CRF models stems from their ability to capture some structured information from second order statistics using the pairwise part.

However, statistical interactions beyond the second-order are tedious to incorporate and render the models complicated BID0 BID12 ).

Other approaches have used task-specific perceptual losses to solve this problem BID10 .Generative models provide another way to represent the structure and spacial contiguity in large high-dimensional datasets with complex dependencies.

Implicit generative models specify a stochastic procedure to produce outputs from a probability distribution.

Such models are appealing because they do not demand parametrization of the probability distribution they are trying to model.

Recently, there has been great interest in CNN-based implicit generative models using autoregressive BID4 and adversarial training frameworks BID16 .Generative adversarial networks (GANs) BID6 can be seen as a two player minimax game where the first player, the generator, is tasked with transforming a random input to a specific distribution such that the second player, the discriminator, can not distinguish between the true and synthesized distributions.

The most distinct feature of adversarial networks is the discriminator that assesses the discrepancy between the current and target distributions.

The discriminator acts as a progressively precise critic of an increasingly accurate generator.

Despite their structured prediction capabilities, such a training paradigm is often unstable and can suffer from mode collapse.

However, recent work on spectral normalization (SN) and gradient penalty has significantly increased training stability BID7 .

Conditional GANs (cGANs) BID19 incorporate conditional image information in the discriminator and have been widely used for class conditioned image generation .

To that effect, unlike in standard GANs, a discriminator for cGANs discriminates between DISPLAYFORM0 Adversarial loss (a) Concatenated Image Conditioning x y Adversarial loss DISPLAYFORM1 Discriminator models for image conditioning.

We propose fusing the features of the input and the ground truth or generated image rather than concatenating.the generated distribution and the target distribution on pairs of samples y and conditional information x.

For class conditioning, several unique strategies have been presented to incorporate class information in the discriminator BID23 BID22 .

However, a cGAN can also be conditioned by structured data such as an image.

Such conditioning is much more useful for structured prediction problems.

Since the discriminator in image conditioned-GANs has access to large portions of the image the adversarial loss can be interpreted as a learned loss that incorporates higher order statistics, essentially eliminating the need to manually design higher order loss functions.

Consequently, this variation of cGANs has extensively been used for image-to-image translation tasks .

However, the best way of incorporating image conditional information into a GAN is not always clear and methods of feeding generated and conditional images to the discriminator tend to use a naive concatenation approach.

In this work we address this gap by proposing a discriminator architecture specifically designed for image conditioning.

Such a discriminator can contribute to the promise of generalization GANs bring to structured prediction problems whereby a singular and simplistic setup can be used for capturing higher order non-local structural information from higher dimensional data without complicated modeling of energy functions.

Contributions.

We propose an approach to incorporating conditional information into a cGAN using a fusion architecture (Fig. 1b) .

In particular, we make the following key contributions:1.

We propose a novel discriminator architecture optimized for incorporating conditional information in cGANs for structured prediction tasks.

The method is designed to incorporate conditional information in feature space and thereby allows the discriminator to enforce higher-order consistency in the model.

At the same time, this method is conceptually simpler than alternative structured prediction methods such as CNN-CRFs where higher-order potentials have to be manually incorporated in the loss function.2.

We demonstrate the effectiveness of this method on a variety of structured prediction tasks including semantic segmentation, depth estimation, and generating real images from semantic masks.

Our empirical study demonstrates that using a fusion discriminator is more effective in preserving high-order statistics and structural information in the data.2 RELATED WORK 2.1 CNN-CRF MODELS Models for structured prediction have been extensively studied in computer vision.

In the past these models often entailed the construction of hand-engineered features.

In 2015, BID15 demonstrated that a fully convolutional approach to semantic segmentation could yield state-ofthe-art results at that time with no need for hand-engineering features.

BID1 showed that post-processing the results of a CNN with a conditional Markov random field led to significant improvements.

Subsequent work by many authors have refined this approach by incorporating the CRF as a layer within a deep network and thereby enabling the parameters of both models to be learnt simultaneously BID11 .

Many researchers have used this approach for other structured prediction problems, including image-to-image translation and depth estimation BID14 .In most cases CNN-CRF models only incorporate unary and pairwise potentials.

Recent work by BID0 has investigated incorporating higher-order potentials into CNN-based models for semantic segmentation, and has found that while it is possible to learn the parameters of these potentials, they can be tedious to incorporate and render the model quite complex.

There is a need for developing methods that can incorporate higher-order statistical information with out manual modeling of higher order potentials.

Adversarial Training.

Generative adversarial networks were introduced in BID6 .

A GAN consists of a pair of models (G, D), where G attempts to model the distribution of the source domain and D attempts to evaluate the divergence between the generative distribution q and the true distribution p. GANs are trained by training the discriminator and the generator in turn, iteratively refining both the quality of the generated data and the discriminator's ability to distinguish between p and q. The result is that D and G compete to reach a Nash equilibrium that can be expressed by the training procedure.

While GAN training is often unstable and prone to issues such as mode collapse, recent developments such as spectral normalization and gradient penalty have increased GAN training stability BID7 .

Furthermore, GANs have the advantage of being able to access the joint configuration of many variables, thus enabling a GAN to enforce higher-order consistency that is difficult to enforce via other methods BID16 .Conditional GANs.

A conditional GAN (cGAN) is a GAN designed to incorporate conditional information BID19 .

cGANs have shown promise for several tasks such as class conditional image synthesis and image-to-image translation BID19 .

There are several advantages to using the cGAN model for structured prediction, including the simplicity of the framework.

Image conditioned cGANs can be seen as a structured prediction problem tasked with learning a new representation given an input image while making use of non-local dependencies.

However, the method by which the conditional information should be incorporated into the model is often unmotivated.

Usually, the conditional data is concatenated to some layers in the discriminator (often the input layers).

A notable exception to this methodology is the projection cGAN, where for data is either known or assumed to follow certain simple distributions and a hard mathematical rule for incorporating conditional data can be derived from the underlying probabilistic graphical model .

As mentioned in , the method is more likely to produce good results if the data follows one of the prescribed distributions.

For structured prediction tasks where the GAN framework has to be conditioned by an image, this is often not the case.

In the following section we introduce the fusion discriminator and explain the motivation behind it.

As mentioned, the most significant part of cGANs for structured prediction is the discriminator.

The discriminator has continuous access to pairs of the generated data or real data y and the conditional information (i.e. the image)

x. The cGAN discriminator can then be defined as, DISPLAYFORM0 , where A is the activation function, and f is a function of x and y and θ represents the parameters of f .

Let p and q designate the true and the generated distributions.

The adversarial loss for the discriminator can then be defined as DISPLAYFORM1 Here, A represents the sigmoid function, D represents the conditional discriminator, and G represents the generator.

By design, this frameworks allows the discriminator to significantly effect the generator BID6 .

The most common approach currently in use to incorporate conditional image information into a GAN is to concatenate the conditional image information to the input of the discriminator at some layer, often the first .

Other approaches for conditional information fusion are limited to class conditional fusion where conditional information is often a one-hot vector rather than higher dimensional structured data.

Since the discriminator classifies pairs of input and output images, concatenating high-dimensional data may not exploit some inherent dependencies in the structure of the data.

Fusing the input and output information in an intuitive way such as to preserve the dependencies is instrumental in designing an adversarial framework with high structural capacity.

We propose the use of a fusion discriminator architecture with two branches.

The branches of this discriminator are convolutional neural networks, say φ(x) and ψ(y), that extract features from both the conditional data and the generated or real data respectively.

The features extracted from the conditional data are then fused with the features from the real or generated data at various stages FIG0 .

The proposed discriminator architecture is similar to the encoder-part of the FuseNet architecture, which has been used to incorporate depth information from RGB-D images for semantic segmentation BID8 .

In FIG0 , we illustrate a four layer and a VGG16-style fusion discriminator, in which both branches are similar in depth and structure to the VGG16 model BID26 .

The key ingredient of the fusion discriminator architecture is the fusion block, which combines the feature maps of x and y. The fusion layer (red, FIG0 ) is implemented as element-wise summation and is always inserted after a convolution → spectral normalization → ReLU instance.

By making use of this fusion layer the discontinuities in the features maps of x and y are added into the y branch in order to enhance the overall feature maps.

This preserves representation from both x and y. For structured prediction tasks x and y often have features that complement each other; for instance, in tasks like depth estimation, semantic segmentation, and image synthesis x and y all have complimentary features.

Theoretical Motivation.

When the data is passed through two networks with identical architectures and the activations at corresponding layers are added, the effect in general is to pass forward through the combined network (the upper branch in FIG0 ) a stronger signal than would be passed forward by applying an activation to concatenated data.

To see this, suppose the k th feature map in the l th layer is denoted by hk .

Let the weights and biases for this feature and layer be denoted W DISPLAYFORM0 T and b DISPLAYFORM1 Further, let h = x T y T T , where x and y represent the learned features from the conditional and real or generated data respectively.

Assuming a ReLU activation function, DISPLAYFORM2 Based on the inequality in Eq. 4 we demonstrate that the fusion of the activations in ψ(x) and φ(y) produces a stronger signal than the activation on concatenated inputs.

Indeed, strengthening some of the activations is not by any means a guarantee of improved performance in general.

However, using the fusion discriminator not only increases the neuron-wise activation values but also preserves activations at different neuron locations.

In the context of conditional GANs the fusing operation results in the strongest signals being passed through the discriminator specifically at those places where the model finds useful information simultaneously in both the conditional data and the real or generated data.

In addition, the model preserves a signal, albeit a weaker one, when either x or y contain useful information both for the learning of higher-level features and for the discriminator's eventual classification of the data.

Empirical Motivation.

We use gradient-weighted Class Activation Mapping (Grad-CAM) BID25 which uses the class-specific gradient information going into the final convolutional layer of a trained CNN to produce a coarse localization map of the important regions in the image.

We visualized the outputs of a fusion and concatenated discriminator for several different tasks to observe the structure and strength of the signal being passed forward.

We observed that the fusion discriminator architecture always had a visually strong signal at important features for the given task.

Representative images from classifying x and y pairs as 'real' for two different structured prediction tasks are shown in FIG1 .

This provides visual evidence that a fusion discriminator preserves more structural information from the input and output image pairs and classifies overlapping patches based on that information.

Indeed, this is not evidence that a stronger signal will lead to a more accurate classification, but it is a heuristic justification that more representative features from x and y will be used to make the determination.

In order to evaluate the effectiveness of the proposed fusion discriminator we conducted three sets of experiments on structured prediction problems: 1) generating real images from semantic masks (Cityscapes); 2) semantic segmentation (Cityscapes); 3) depth estimation (NYU v2) .

For all three tasks we used a U-Net based generator.

We applied spectral normalization to all weights of the

In order to demonstrate the structure preserving abilities of our discriminator we use the proposed setup in the image-to-image translation setting.

We focus on the application of generating realistic images from semantic labels.

This application has recently been studied for generating realistic synthetic data for self driving cars BID28 BID3 .

Unlike recent approaches where the objective is to generate increasingly realistic high definition (HD) images, the purpose of this experiment is to explore if a generic fusion discriminator can outperform a concatenated discriminator when using a simple generator.

We used 2,975 training images from the Cityscapes dataset BID5 and re-scaled them to 256 × 256 for computational efficiency.

The provided Cityscapes test set with 500 images was used for testing.

Our ablation study focused on changing the discriminator between a standard 4-layer concatenation discriminator used in seminal image-to-image translation work , a combination of this 4-layer discriminator with spectral normalization (SN) , a VGG-16 concatenation discriminator and the proposed 4-layer and VGG-16 fusion discriminators.

Since standard GAN evaluation metrics such as inception score and FID can not directly be applied to image-to-image translation tasks we use an evaluation technique previously used for such image synthesis .

To quantitatively evaluate and comparatively analyze the effectiveness of our proposed discriminator architecture we perform semantic segmentation on synthesized images and compare the similarity between the predicted segments and the input.

The intuition behind this kind of experimentation is that if the generated images corresponds to the input label map an existing semantic segmentation model such as a PSPNet BID29 should be able to predict the input segmentation mask.

Similar experimentation has been suggested in and .

Table 1 reports segmentation both pixel-wise accuracy and overall intersection-over-union (IoU), the proposed fusion discriminator outperforms the concatenated discriminator by a large margin.

Our result is closer to the theoretical upper bound achieved by real images.

This confirms that the fusion discriminator contributes to preserving more Figure 5 : A comparative analysis of concatenation, projection and fusion discriminators on three different structured prediction tasks, i.e., image synthesis, semantic segmentation, and depth estimation.

Table 1 : PSPNet-based semantic segmentation IoU and accuracy scores using generated images from different discriminators.

Our results outperform concatenation-based methods by a large margin and is close to the accuracy and IoU on actual images (GT/Oracle).

Mean IoU Pixel Accuracy 4-Layer Concat.

structure in the output image.

The fusion discriminator could be used with high definition images, however, such analysis is beyond the scope of the current study.

Representative images for this task are shown in FIG2 .

Fig. 5 shows a comparative analysis of the concatenation, projection and fusion discriminators in an ablation study upto 550k iterations.

The projection discriminator was modified image conditioning according to the explanation given in for the super-resolution task.

Semantic segmentation is vital for visual scene understanding and is often formulated as a dense labeling problem where the objective is to predict the category label for each individual pixel.

Semantic segmentation is a classical structured prediction problem and CNNs with pixel-wise loss often fail to make accurate predictions BID16 .

Much better results have been achieved by incorporating higher order statistics in the image using CRFs as a post-processing step or jointly training them with CNNs BID2 .

It has been shown that incorporating higher order potentials continues to improve semantic segmentation improvement, making this an ideal task for evaluating the structured prediction capabilities of GANs and their enhancement using our proposed discriminator.

Here, we empirically validate that the adversarial framework with the fusion discriminator can preserve more spacial context in comparison to CNN-CRF setups.

We demonstrate that our proposed fusion discriminator is equipped with the ability to preserve higher order details.

For comparative analysis we compare with relatively shallow and deep architectures for both concatenation and fusion discriminators.

We also conduct an ablation study to analyze the effect of spectral normalization.

The generator for all semantic segmentation experiments was a U-Net.

For the experiment without spectral normalization, we trained each model for 950k iterations, which was sufficient for

Depth estimation is another structured prediction task that has been extensively studied because of its wide spread applications in computer vision.

As with semantic segmentation, both per-pixel losses and non-local losses such as CNN-CRFs have been widely used for depth estimation.

Stateof-the art with depth estimation has been achieved using a hierarchical chain of non-local losses.

We argue that it is possible to incorporate higher order information using a simple adversarial loss with a fusion discriminator.

In order to validate our claims we conducted a series of experiments with different discriminators, similar to the series of experiments conducted for semantic segmentation.

We used the Eigen testtrain split for the NYU v2 ? dataset containing 1449 images for training and 464 images for testing.

We observed that as with image synthesis and semantic segmentation the fusion discriminator outperforms concatenation-based methods and pairwise CNN-CRF methods every time.

Structured prediction problems can be posed as image conditioned GAN problems.

The discriminator plays a crucial role in incorporating non-local information in adversarial training setups for structured prediction problems.

Image conditioned GANs usually feed concatenated input and output pairs to the discriminator.

In this research, we proposed a model for the discriminator of cGANs that involves fusing features from both the input and the output image in feature space.

This method provides the discriminator a hierarchy of features at different scales from the conditional data, and thereby allows the discriminator to capture higher-order statistics from the data.

We qualitatively demonstrate and empirically validate that this simple modification can significantly improve the general adversarial framework for structured prediction tasks.

The results presented in this paper strongly suggest that the mechanism of feeding paired information into the discriminator in image conditioned GAN problems is of paramount importance.

The generator G tries to minimize the loss expressed by equation 5 while the discriminator D tries to maximize it.

In addition, we impose an L1 reconstruction loss: DISPLAYFORM0 leading to the objective, DISPLAYFORM1 6.2 GENERATOR ARCHITECTURE We adapt our network architectures from those explained in .

Let CSRk denote a Convolution-Spectral Norm -ReLU layer with k filters.

Let CSRDk donate a similar layer with dropout with a rate of 0.5.

All convolutions chosen are 4 × 4 spatial filters applied with a stride 2, and in decoders they are up-sampled by 2.

All networks were trained from scratch and weights were initialized from a Gaussian distribution of mean 0 and standard deviation of 0.02.

All images were cropped and rescaled to 256 × 256, were up sampled to 268 × 286 and then randomly cropped back to 256 × 256 to incorporate random jitter in the model.

Decoder: CSRD512→CSRD1024→CSRD1024→CSR1024→CSR1024→CSR512→CSR256→CSR128The last layer in the decoder is followed by a convolution to map the number of output channels (3 in the case of image synthesis and semantic labels and 1 in the case of depth estimation).

This is followed by a Tanh function.

Leaky ReLUs were used throughout the encoder with a slope of 0.2, regular ReLUs were used in the decoder.

Skip connections are placed between each layer l in the encoder and layer ln in the decoder assuming l is the maximum number of layers.

The skip connections concatenate activations from the l th layer to layer (l − n) th later.

<|TLDR|>

@highlight

We propose a novel way to incorporate conditional image information into the discriminator of GANs using feature fusion that can be used for structured prediction tasks.
We propose a fully-convolutional conditional generative model, the latent transformation neural network (LTNN), capable of view synthesis using a light-weight neural network suited for real-time applications.

In contrast to existing conditional generative models which incorporate conditioning information via concatenation, we introduce a dedicated network component, the conditional transformation unit (CTU), designed to learn the latent space transformations corresponding to specified target views.

In addition, a consistency loss term is defined to guide the network toward learning the desired latent space mappings, a task-divided decoder is constructed to refine the quality of generated views, and an adaptive discriminator is introduced to improve the adversarial training process.

The generality of the proposed methodology is demonstrated on a collection of three diverse tasks: multi-view reconstruction on real hand depth images, view synthesis of real and synthetic faces, and the rotation of rigid objects.

The proposed model is shown to exceed state-of-the-art results in each category while simultaneously achieving a reduction in the computational demand required for inference by 30% on average.

Generative models have been shown to provide effective frameworks for representing complex, structured datasets and generating realistic samples from underlying data distributions BID8 .

This concept has also been extended to form conditional models capable of sampling from conditional distributions in order to allow certain properties of the generated data to be controlled or selected BID20 .

These generative models are designed to sample from broad classes of the data distribution, however, and are not suitable for inference tasks which require identity preservation of the input data.

Models have also been proposed which incorporate encoding components to overcome this by learning to map input data to an associated latent space representation within a generative framework BID18 .

The resulting inference models allow for the defining structure/features of inputs to be preserved while specified target properties are adjusted through conditioning BID34 .

Conventional conditional models have largely relied on rather simple methods, such as concatenation, for implementing this conditioning process; however, BID21 have shown that utilizing the conditioning information in a less trivial, more methodical manner has the potential to significantly improve the performance of conditional generative models.

In this work, we provide a general framework for effectively performing inference with conditional generative models by strategically controlling the interaction between conditioning information and latent representations within a generative inference model.

In this framework, a conditional transformation unit (CTU), Φ, is introduced to provide a means for navigating the underlying manifold structure of the latent space.

The CTU is realized in the form of a collection of convolutional layers which are designed to approximate the latent space operators defined by mapping encoded inputs to the encoded representations of specified targets (see FIG7 ).

This is enforced by introducing a consistency loss term to guide the CTU mappings during training.

In addition, a conditional discriminator unit (CDU), Ψ, also realized as a collection of convolutional layers, is included in the network's discriminator.

This CDU is designed to improve the network's ability to identify and eliminate transformation specific artifacts in the network's predictions.

The network has also been equipped with RGB balance parameters consisting of three values {θ R , θ G , θ B } designed to give the network the ability to quickly adjust the global color balance of FIG7 : The conditional transformation unit Φ constructs a collection of mappings {Φ k } in the latent space which produce high-level attribute changes to the decoded outputs.

Conditioning information is used to select the appropriate convolutional weights ω k for the specified transformation; the encoding l x of the original input image x is transformed to l y k = Φ k (l x ) = conv(l x , ω k ) and provides an approximation to the encoding l y k of the attribute-modified target image y k .the images it produces to better align with that of the true data distribution.

In this way, the network is easily able to remove unnatural hues and focus on estimating local pixel values by adjusting the three RGB parameters rather than correcting each pixel individually.

In addition, we introduce a novel estimation strategy for efficiently learning shape and color properties simultaneously; a task-divided decoder is designed to produce a coarse pixel-value map along with a refinement map in order to split the network's overall task into distinct, dedicated network components.

1.

We introduce the conditional transformation unit, with a family of modular filter weights, to learn high-level mappings within a low-dimensional latent space.

In addition, we present a consistency loss term which is used to guide the transformations learned during training.2.

We propose a novel framework for color inference which separates the generative process into three distinct network components dedicated to learning i) coarse pixel value estimates, ii) pixel refinement scaling factors, and iii) the global RGB color balance of the dataset.3.

We introduce the conditional discriminator unit designed to improve adversarial training by identifying and eliminating transformation-specific artifacts present in generated images.

Each contribution proposed above has been shown to provide a significant improvement to the network's overall performance through a series of ablation studies.

The resulting latent transformation neural network (LTNN) is placed through a series of comparative studies on a diverse range of experiments where it is seen to outperform existing state-of-the-art models for (i) simultaneous multi-view reconstruction of real hand depth images in real-time, (ii) view synthesis and attribute modification of real and synthetic faces, and (iii) the synthesis of rotated views of rigid objects.

Moreover, the CTU conditioning framework allows for additional conditioning information, or target views, to be added to the training procedure ad infinitum without any increase to the network's inference speed.

BID4 has proposed a supervised, conditional generative model trained to generate images of chairs, tables, and cars with specified attributes which are controlled by transformation and view parameters passed to the network.

The range of objects which can be synthesized using the framework is strictly limited to the pre-defined models used for training; the network can generate different views of these models, but cannot generalize to unseen objects to perform inference tasks.

Conditional generative models have been widely used for geometric prediction BID23 BID32 .

These models are reliant on additional data, such as depth information or mesh models, to perform their target tasks, however, and cannot be trained using images alone.

Other works have introduced a clamping strategy to enforce a specific organizational structure in the latent space BID27 BID17 ; these networks require extremely detailed labels for supervision, such as the graphics code parameters used to create each example, and are therefore very difficult to implement for more general tasks (e.g. training with real images).

Zhou et al. (2016) have proposed the appearance flow network (AFN) designed specifically for the prediction of rotated viewpoints of objects from images.

This framework also relies on geometric concepts unique to rotation and is not generalizable to other inference tasks.

The conditional variational autoencoder (CVAE) incorporates conditioning information into the standard variational autoencoder (VAE) framework BID16 and is capable of synthesizing specified attribute changes in an identity preserving manner BID29 BID34 .

CVAE-GAN BID0 further adds adversarial training to the CVAE framework in order to improve the quality of generated predictions.

Zhang et al. (2017) have introduced the conditional adversarial autoencoder (CAAE) designed to model age progression/regression in human faces.

This is achieved by concatenating conditioning information (i.e. age) with the input's latent representation before proceeding to the decoding process.

The framework also includes an adaptive discriminator with conditional information passed using a resize/concatenate procedure.

BID13 have proposed Pix2Pix as a general-purpose image-to-image translation network capable of synthesizing views from a single image.

The IterGAN model introduced by BID6 is also designed to synthesize novel views from a single image, with a specific emphasis on the synthesis of rotated views of objects in small, iterative steps.

To the best of our knowledge, all existing conditional generative models designed for inference use fixed hidden layers and concatenate conditioning information directly with latent representations; in contrast to these existing methods, the proposed model incorporates conditioning information by defining dedicated, transformation-specific convolutional layers at the latent level.

This conditioning framework allows the network to synthesize multiple transformed views from a single input, while retaining a fully-convolutional structure which avoids the dense connections used in existing inference-based conditional models.

Most significantly, the proposed LTNN framework is shown to outperform state-of-the-art models in a diverse range of view synthesis tasks, while requiring substantially less FLOPs for inference than other conditional generative models (see Tables 1 & 2) .

In this section, we introduce the methods used to define the proposed LTNN model.

We first give a brief overview of the LTNN network structure.

We then detail how conditional transformation unit mappings are defined and trained to operate on the latent space, followed by a description of the conditional discriminator unit implementation and the network loss function used to guide the training process.

Lastly, we describe the task-division framework used for the decoding process.

The basic workflow of the proposed model is as follows:1.

Encode the input image x to a latent representation l x = Encode(x).2.

Use conditioning information k to select conditional, convolutional filter weights ω k .3.

Map the latent representation l x to l y k = Φ k (l x ) = conv(l x , ω k ), an approximation of the encoded latent representation l y k of the specified target image y k .4.

Decode l y k to obtain a coarse pixel value map and a refinement map.5.

Scale the channels of the pixel value map by the RGB balance parameters and take the Hadamard product with the refinement map to obtain the final prediction y k .6.

Pass real images y k as well as generated images y k to the discriminator, and use the conditioning information to select the discriminator's conditional filter weights ω k .7.

Compute loss and update weights using ADAM optimization and backpropagation.

A detailed overview of the proposed network structure is provided in Section A.1 of the appendix.

Provide: Labeled dataset x, {y k } k∈T with target transformations indexed by a fixed set T , encoder weights θ E , decoder weights θ D , RGB balance parameters {θ R , θ G , θ B }, conditional transformation unit weights {ω k } k∈T , discriminator D with standard weights θ D and conditionally selected weights {ω k } k∈T , and loss function hyperparameters γ, ρ, λ, κ corresponding to the smoothness, reconstruction, adversarial, and consistency loss terms, respectively.

The specific loss function components are defined in detail in Equations 1 -5 in Section 3.2.1: procedure TRAIN( ) 2:x , {y k } k∈T = get train batch() # Sample input and targets from training set 3: DISPLAYFORM0 for k in T do 5: DISPLAYFORM1 DISPLAYFORM2 DISPLAYFORM3 # Assemble final network prediction for target 9: 10:# Update encoder, decoder, RGB, and CTU weights 11: DISPLAYFORM4 # Update discriminator and CDU weights 19: DISPLAYFORM5

Generative models have frequently been designed to explicitly disentangle the latent space in order to enable high-level attribute modification through linear, latent space interpolation.

This linear latent structure is imposed by design decisions, however, and may not be the most natural way for a network to internalize features of the data distribution.

Several approaches have been proposed which include nonlinear layers for processing conditioning information at the latent space level.

In these conventional conditional generative frameworks, conditioning information is introduced by combining features extracted from the input with features extracted from the conditioning information (often using dense connection layers); these features are typically combined using standard vector concatenation, although some have opted to use channel concatenation (Zhang et al., 2017; BID0 .

Six of these conventional conditional network designs are illustrated in FIG0 along with the proposed LTNN network design for incorporating conditioning information.

Rather than directly concatenating conditioning information, we propose using a conditional transformation unit (CTU), consisting of a collection of distinct convolutional mappings in the network's latent space; conditioning information is then used to select which collection of weights, i.e. which CTU mapping, should be used in the convolutional layer to perform a specified transformation.

For view point estimation, there is an independent CTU per viewpoint.

Each CTU mapping maintains its own collection of convolutional filter weights and uses Swish activations BID26 .

The filter weights and Swish parameters of each CTU mapping are selectively updated by controlling the gradient flow based on the conditioning information provided.

The CTU mappings are trained to transform the encoded, latent space representation of the network's input in a manner which produces high-level view or attribute changes upon decoding.

This is accomplished by introducing a consistency term into the loss function which is minimized precisely when the CTU mappings behave as depicted in FIG7 .

In this way, different angles of view, light directions, and deformations, for example, can be synthesized from a single input image.

The discriminator used in the adversarial training process is also passed conditioning information which specifies the transformation which the model has attempted to make.

The conditional discriminator unit (CDU), consisting of convolutional layers with modular weights similar to the CTU, is trained to specifically identify unrealistic artifacts which are being produced by the corresponding conditional transformation unit mappings.

For view point estimation, there is an independent CDU per viewpoint.

The incorporation of this context-aware discriminator structure has significantly boosted the performance of the network (see Table 4 in the appendix).The proposed model uses the adversarial loss as the primary loss component.

The discriminator, D, is trained using the adversarial loss term L D adv defined below in Equation 1.

Additional loss terms corresponding to structural reconstruction, smoothness BID14 , and a notion of consistency, are also used for training the encoder/decoder: DISPLAYFORM0 DISPLAYFORM1 where y k is the modified target image corresponding to an input x, ω k are the weights of the CDU mapping corresponding to the k th transformation, Φ k is the CTU mapping for the k th transformation, y k = Decode Φ k Encode[x] is the network prediction, and τ i,j is the two-dimensional, discrete shift operator.

The final loss function for the encoder and decoder components is given by: DISPLAYFORM2 with hyperparameters typically selected so that λ, ρ γ, κ.

The consistency loss is designed to guide the CTU mappings toward approximations of the latent space mappings which connect the latent representations of input images and target images as depicted in FIG7 .

In particular, the consistency term enforces the condition that the transformed encoding, DISPLAYFORM3 , approximates the encoding of the k th target image, l y k = Encode[y k ], during the training process.

The decoding process has been divided into three tasks: estimating the refinement map, pixel-values, and RGB color balance of the dataset.

We have found this decoupled framework for estimation helps the network converge to better minima to produce sharp, realistic outputs.

The decoding process begins with a series of convolutional layers followed by bilinear interpolation to upsample the low resolution latent information.

The last component of the decoder's upsampling process consists of two distinct transpose convolutional layers used for task separation; one layer is allocated for predicting the refinement map while the other is trained to predict pixel-values.

The refinement map layer incorporates a sigmoidal activation function which outputs scaling factors intended to refine the coarse pixel value estimations.

RGB balance parameters, consisting of three trainable variables, are used as weights for balancing the color channels of the pixel value map.

The Hadamard product of the refinement map and the RGB-rescaled value map serves as the network's final output: DISPLAYFORM0 In this way, the network has the capacity to mask values which lie outside of the target object (i.e. by setting refinement map values to zero) which allows the value map to focus on the object itself during the training process.

Experimental results show that the refinement maps learn to produce masks which closely resemble the target objects' shapes and have sharp drop-offs along the boundaries.

BID33 .

The input depth-map hand pose image is shown to the far left, followed by the network predictions for 9 synthesized view points.

The views synthesized using LTNN are seen to be sharper and also yield higher accuracy for pose estimation (see FIG4 ).

To show the generality of our method, we have conducted a series of diverse experiments: (i) hand pose estimation using a synthetic training set and real NYU hand depth image data BID33 for testing, (ii) synthesis of rotated views of rigid objects using the real ALOI dataset BID7 and synthetic 3D chair dataset BID1 , (iii) synthesis of rotated views using a real face dataset BID5 , and (iv) the modification of a diverse range of attributes on a synthetic face dataset (IEE, 2009) .

For each experiment, we have trained the models using 80% of the datasets.

Since ground truth target depth images were not available for the real hand dataset, an indirect metric has been used to quantitatively evaluate the model as described in Section 4.1.

Ground truth data was available for all other experiments, and models were evaluated directly using the L 1 mean pixel-wise error and the structural similarity index measure (SSIM) BID23 BID19 (the masked pixel-wise error L M 1 BID6 ) was used in place of the L 1 error for the ALOI experiment).

More details regarding the precise training configurations and the creation of the synthetic datasets can be found in the appendix.

To evaluate the proposed framework with existing works, two comparison groups have been formed: conditional inference models (CVAE-GAN, CVAE, and CAAE) with comparable encoder/decoder structures for comparison on experiments with non-rigid objects, and view synthesis models (MV3D BID32 , IterGAN, Pix2Pix, AFN (Zhou et al., 2016) , and TVSN BID23 ) for comparison on experiments with rigid objects.

Additional experiments have been performed to compare the proposed CTU conditioning method with other conventional concatenation methods (see FIG0 ; results are shown in FIG3 .

Qualitative results and comparisons for each experiment are provided in the appendix.

Hand pose experiment: Since ground truth predictions for the real NYU hand dataset were not available, the LTNN model has been trained using a synthetic dataset generated using 3D mesh hand models.

The NYU dataset does, however, provide ground truth coordinates for the input hand pose; using this we were able to indirectly evaluate the performance of the model by assessing the accuracy of a hand pose estimation method using the network's multi-view predictions as input.

More specifically, the LTNN model was trained to generate 9 different views which were then fed into the pose estimation network from BID3 (also trained using the synthetic dataset).A comparison of the quantitative hand pose estimation results is provided in FIG3 where the proposed LTNN framework is seen to provide a substantial improvement over existing methods; qualitative results are also available in FIG1 .

With regard to real-time applications, the proposed model runs at 114 fps without batching and at 1975 fps when applied to a mini-batch of size 128 (using a single TITAN Xp GPU and an Intel i7-6850K CPU).

The stereo face database BID5 , consisting of images of 100 individuals from 10 different viewpoints, was used for experiments with real faces; these faces were segmented using the method of BID22 and then cropped and centered to form the final dataset.

The LTNN model was trained to synthesize images of input faces corresponding to three consecutive horizontal rotations.

As shown in FIG4 , our method significantly outperforms the CVAE-GAN, CAAE, and IterGAN models in both the L 1 and SSIM metrics; qualitative results are also available in FIG5 and Section A.6 of the appendix.

Real object experiment: The ALOI dataset BID7 , consisting of images of 1000 real objects viewed from 72 rotated angles (covering one full 360• rotation), has been used for experiments on real objects.

As shown in Table 1 and in Figure 8 , our method outperforms other state-of-the-art methods with respect to the L 1 metric and achieves comparable SSIM metric scores.

Of note is the fact that the LTNN framework is capable of effectively performing the specified rigid-object transformations using only a single image as input, whereas most state-of-the-art view synthesis methods require additional information which is not practical to obtain for real datasets.

For example, MV3D requires depth information and TVSN requires 3D models to render visibility maps for training which is not available in the ALOI dataset.

We have tested our model's ability to perform 360• view estimation on the chairs and compared the results with the other state-of-the-art methods.

The proposed model outperforms existing models specifically designed for the task of multi-view prediction and require the least FLOPs for inference compared with all other methods (see Table 2 ).

Table 2 : Results for 3D chair 360• view synthesis.

The proposed method uses significantly less parameters during inference, requires the least FLOPs, and yields the fastest inference times.

FLOP calculations correspond to inference for a single image with resolution 256×256×3.

To evaluate the proposed framework's performance on a more diverse range of attribute modification tasks, a synthetic face dataset and five conditional generative models with comparable encoder/decoder structures to the LTNN model have been selected for comparison.

These models have been trained to synthesize discrete changes in elevation, azimuth, light direction, and age from a single grayscale image; results are shown in Table 3 and ablation results are available in Table 4 .

Near continuous attribute modification is also possible within the proposed framework, and distinct CTU mappings can be composed with one another to synthesize multiple modifications simultaneously; more details and related figures are provided in sections A.7.4 and A.7.5 of the appendix.

Table 3 : Results for simultaneous colorization and attribute modification on synthetic face dataset.

DISPLAYFORM0

In this work, we have introduced an effective, general framework for incorporating conditioning information into inference-based generative models.

We have proposed a modular approach to incorporating conditioning information using CTUs and a consistency loss term, defined an efficient task-divided decoder setup for deconstructing the data generation process into managable subtasks, and shown that a context-aware discriminator can be used to improve the performance of the adversarial training process.

The performance of this framework has been assessed on a diverse range of tasks and shown to outperform state-of-the-art methods.

At the bottle-neck between the encoder and decoder, a conditional transformation unit (CTU) is applied to map the 2×2 latent features directly to the transformed 2×2 latent features on the right.

This CTU is implemented as a convolutional layer with filter weights selected based on the conditioning information provided to the network.

The noise vector z ∈ R 4 from normal distribution N (0, 1) is concatenated to the transformed 2×2 features and passed to the decoder for the face attributes task only.

The 32×32 features near the end of the decoder component are processed by two independent convolution transpose layers: one corresponding to the value estimation map and the other corresponding to the refinement map.

The channels of the value estimation map are rescaled by the RGB balance parameters, and the Hadamard product is taken with the refinement map to produce the final network output.

For the ALOI data experiment, we have followed the IterGAN Galama & Mensink (2018) encoder and decoder structure, and for the stereo face dataset BID5 experiment, we have added an additional Block v1 layer in the encoder and decoder to utilize the full 128×128×3 resolution images.

The encoder incorporates two main block layers, as defined in Figure A. 2, which are designed to provide efficient feature extraction; these blocks follow a similar design to that proposed by , but include dense connections between blocks, as introduced by BID10 .

We normalize the output of each network layer using the batch normalization method as described in BID12 .

For the decoder, we have opted for a minimalist design, inspired by the work of BID24 .

Standard convolutional layers with 3 × 3 filters and same padding are used through the penultimate decoding layer, and transpose convolutional layers with 5 × 5 filters and same padding are used to produce the value-estimation and refinement maps.

All parameters have been initialized using the variance scaling initialization method described in BID9 .Our method has been implemented and developed using the TensorFlow framework.

The models have been trained using stochastic gradient descent (SGD) and the ADAM optimizer BID15 with initial parameters: learning rate = 0.005, β 1 = 0.9, and β 2 = 0.999 (as defined in the TensorFlow API r1.6 documentation for tf.train.AdamOptimizer).

, along with loss function hyperparameters: λ = 0.8, ρ = 0.2, γ = 0.0002, and κ = 0.00005 (as introduced in FORMULA8 ).

The discriminator is updated once every two encoder/decoder updates, and one-sided label smoothing BID28 has been used to improve stability of the discriminator training procedure.

All datasets have also been normalized to the interval [0, 1] for training.

Once the total number of output channels, N out , is specified, the remaining N out − N in output channels are allocated to the non-identity filters (where N in denotes the number of input channels).

For the Block v1 layer at the start of the proposed LTNN model, for example, the input is a single grayscale image with N in = 1 channel and the specified number of output channels is N out = 32.

One of the 32 channels is accounted for by the identity component, and the remaining 31 channels are the three non-identity filters.

When the remaining channel count is not divisible by 3 we allocate the remainder of the output channels to the single 3 × 3 convolutional layer.

Swish activation functions are used for each filter, however the filters with multiple convolutional layers (i.e. the right two filters in the Block v1 diagram) do not use activation functions for the intermediate 3 × 3 convolutional layers (i.e. those after the 1 × 1 layers and before the final 3 × 3 layers).

A.2.1 DATASET A kinematic hand model with 33 degrees of freedom has been used to generate 200,000 distinct hand poses with nine depth images from different viewpoints for each pose.

We sampled hand pose uniformly from each of the 18 joint angle parameters, covering a full range of hand articulations.

The nine viewpoints are centered around a designated input view and correspond to 30• changes in the spherical coordinates of the viewer (i.e. 30• up, 30• right, 30• up and 30• right, etc.).

Testing was performed on the MSRA and NYU BID33 hand datasets.

We follow the training procedure proposed by BID3 for the hand pose estimation network; after training, the LTNN multi-view predictions are fed into the network and the accuracy of the predicted angles is used to assess how well these network predictions approximate the unknown, true views.

As seen in Figure A .3, the optimal results are obtained when all 9 synthesized views points are fed into the pose estimation network. (2018) , we have used images of resolution 256 × 256 × 3 from the ALOI database for training and testing.

While the LTNN model is capable of making simultaneous predictions of multiple viewpoints, as illustrated in the hand and chair experiments, the Pix2Pix BID13 and IterGAN BID6 networks are designed to produce a single synthesized view.

To make fair comparisons between these existing networks and the proposed LTNN model, each model has been trained only to produce a single 30• rotated view of the ALOI objects.

In particular, only two CTU mappings were trained: one corresponding to the identity, and one corresponding to the rotated view.

Figure A .8:

Experiment on unseen objects from the ALOI real object dataset.

First row of is ground truth, second row is ours, third row is ours without task-division, fourth row is IterGAN, and bottom row is Pix2Pix.

As shown from the figure, our methods are sharper and realistic than other methods in the majority of generated views.

Chairs from the ShapeNet BID2 ) 3D model repository have been rotated horizontally 20• 17 times and vertically 10 • 3 times; 6742 chairs have been selected following the data creation methodology from BID23 .

A.6 REAL FACE EXPERIMENT A.6.1 DATASET The original dataset has background and 100 identification and 5 different views from 2 distinct cameras, which results 1000 face images in total.

Since the dataset is not huge, we would like to reduce background noise and perform face segmentation with BID22 and manually filtered badly segmented faces from the background with the segmentation method and create 300 × 300 × 3 face images.

For training, we resize the original images into 128 × 128 × 3 resolution.

Each face has been rendered at four distinct age ranges, and four different lighting directions have been used.

The orientation of faces is allowed to vary in elevation from −20• to 29• by increments of 7• and in azimuth from 10 • to 150• by increments of 20• .

To demonstrate the model's colorization capabilities, the input images have been converted to gray-scale using the luminosity method.

Table 4 : Ablation/comparison results using identical encoder, decoder, and training procedure.

DISPLAYFORM0 Task-division: An overview of the task-division decoding procedure applied to the synthetic face dataset is provided in Figure A .13.

As noted in Section 3.3, the refinement maps tend to learn to produce masks which closely resemble the target objects' shapes and have sharp drop-offs along the objects' boundaries.

In addition to masking extraneous pixels, these refinement maps have been shown to apply local color balancing by, for example, filtering out the green and blue channels near lips when applied to human faces (i.e. the refinement maps for the green and blue channels show darker regions near the lips, thus allowing for the red channel to be expressed more distinctly).The use of a task-divided decoder can also be seen to remove artifacts in the generated images in Figure A. 13; e.g. removal of the blurred eyebrow (light), elimination of excess hair near the side of ear (azimuth), and reduction of the reddish vertical stripe on the forehead (age).

As noted in Section 4.3, near-continuous attribute modification can be performed by piecewise-linear interpolation in the latent space.

For example, we can train 9 CTU mappings {Φ k } 8 k=0 corresponding to discrete, incremental 7• changes in elevation {θ k }.

In this setting, the network predictions for an elevation change of θ 0 = 0• and θ 1 = 7• are given by Decode[Φ 0 (l x )] and Decode[Φ 1 (l x )], respectively (where l x denotes the encoding of the input image).

To predict an elevation change of 3.5• , we can perform linear interpolation in the latent space between the representations Φ 0 (l x ) and Φ 1 (l x ); that is, we may take our network prediction for the intermediate change of 3.5• to be: y = Decode[ l y ] where l y = 0.5 · Φ 0 (l x ) + 0.5 · Φ 1 (l x )Likewise, to approximate a change of 10.5• in elevation we may take Decode[ l y ], where l y = 0.5 · Φ 1 (l x ) + 0.5 · Φ 2 (l x ), as the network prediction.

More generally, we can interpolate between the latent CTU map representations to predict a change θ via: DISPLAYFORM0 with k ∈ {0, . . .

, 7} and λ ∈ [0, 1] chosen so that θ = λ · θ k + (1 − λ) · θ k+1 .

Accordingly, the proposed framework naturally allows for continuous attribute changes to be approximated by using this piecewise-linear latent space interpolation procedure.

Figure A .19: Near continuous attribute modification is attainable using piecewise-linear interpolation in the latent space.

Provided a gray-scale image (corresponding to the faces on the far left), modified images corresponding to changes in light direction (first), age (second), azimuth (third), and elevation (fourth) are produced with 17 degrees of variation.

These attribute modified images have been produced using 9 CTU mappings, corresponding to varying degrees of modification, and linearly interpolating between the discrete transformation encodings in the latent space.

Additional qualitative results for near-continuous attribute modification can be found in Section A.7.6.

@highlight

We introduce an effective, general framework for incorporating conditioning information into inference-based generative models.
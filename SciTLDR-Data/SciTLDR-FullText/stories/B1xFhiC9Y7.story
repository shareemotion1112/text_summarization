Predicting structured outputs such as semantic segmentation relies on expensive per-pixel annotations to learn strong supervised models like convolutional neural networks.

However, these models trained on one data domain may not generalize well to other domains unequipped with annotations for model finetuning.

To avoid the labor-intensive process of annotation, we develop a domain adaptation method to adapt the source data to the unlabeled target domain.

To this end, we propose to learn discriminative feature representations of patches based on label histograms in the source domain, through the construction of a disentangled space.

With such representations as guidance, we then use an adversarial learning scheme to push the feature representations in target patches to the closer distributions in source ones.

In addition, we show that our framework can integrate a global alignment process with the proposed patch-level alignment and achieve state-of-the-art performance on semantic segmentation.

Extensive ablation studies and experiments are conducted on numerous benchmark datasets with various settings, such as synthetic-to-real and cross-city scenarios.

Recent deep learning-based methods have made significant progress on vision tasks, such as object recognition BID17 and semantic segmentation BID19 , relying on large-scale annotations to supervise the learning process.

However, for a test domain different from the annotated training data, learned models usually do not generalize well.

In such cases, domain adaptation methods have been developed to close the gap between a source domain with annotations and a target domain without labels.

Along this line of research, numerous methods have been developed for image classification BID29 BID8 , but despite recent works on domain adaptation for pixel-level prediction tasks such as semantic segmentation BID14 , there still remains significant room for improvement.

Yet domain adaptation is a crucial need for pixel-level predictions, as the cost to annotate ground truth is prohibitively expensive.

For instance, road-scene images in different cities may have various appearance distributions, while conditions even within the same city may vary significantly over time or weather.

Existing state-of-the-art methods use feature-level BID14 or output space adaptation BID31 to align the distributions between the source and target domains using adversarial learning BID11 BID37 .

These approaches usually exploit the global distribution alignment, such as spatial layout, but such global statistics may already differ significantly between two domains due to differences in camera pose or field of view.

Figure 1 illustrates one example, where two images share a similar layout, but the corresponding grids do not match well.

Such misalignment may introduce an incorrect bias during adaptation.

Instead, we consider to match patches that are more likely to be shared across domains regardless of where they are located.

One way to utilize patch-level information is to align their distributions through adversarial learning.

However, this is not straightforward since patches may have high variation among each other and there is no guidance for the model to know which patch distributions are close.

Motivated by recent advances in learning disentangled representations BID18 BID24 , we adopt a similar approach by considering label histograms of patches as a factor and learn discriminative Figure 1 : Illustration of the proposed patch-level alignment against the global alignment that considers the spatial relationship between grids.

We first learn discriminative representations for source patches (solid symbols) and push a target representation (unfilled symbol) close to the distribution of source ones, regardless of where these patches are located in the image.representations for patches to relax the high-variation problem among them.

Then, we use the learned representations as a bridge to better align patches between source and target domains.

Specifically, we utilize two adversarial modules to align both the global and patch-level distributions between two domains, where the global one is based on the output space adaptation BID31 , and the patch-based one is achieved through the proposed alignment by learning discriminative representations.

To guide the learning process, we first use the pixel-level annotations provided in the source domain and extract the label histogram as a patch-level representation.

We then apply K-means clustering to group extracted patch representations into K clusters, whose cluster assignments are then used as the ground truth to train a classifier shared across two domains for transferring a learned discriminative representation of patches from the source to the target domain.

Ideally, given the patches in the target domain, they would be classified into one of K categories.

However, since there is a domain gap, we further use an adversarial loss to push the feature representations of target patches close to the distribution of the source patches in this clustered space (see Figure 1 ).

Note that our representation learning can be viewed as a kind of disentanglement guided by the label histogram, but is different from existing methods that use pre-defined factors such as object pose BID18 .In experiments, we follow the domain adaptation setting in BID14 and perform pixellevel road-scene image segmentation.

We conduct experiments under various settings, including the synthetic-to-real, i.e., GTA5 BID27 )/SYNTHIA BID28 to Cityscapes BID5 ) and cross-city, i.e., Cityscapes to Oxford RobotCar BID23 scenarios.

In addition, we provide extensive ablation studies to validate each component in the proposed framework.

By combining global and patch-level alignments, we show that our approach performs favorably against state-of-the-art methods in terms of accuracy and visual quality.

We note that the proposed framework is general and could be applicable to other forms of structured outputs such as depth, which will be studied in our future work.

The contributions of this work are as follows.

First, we propose a domain adaptation framework for structured output prediction by utilizing global and patch-level adversarial learning modules.

Second, we develop a method to learn discriminative representations guided by the label histogram of patches via clustering and show that these representations help the patch-level alignment.

Third, we demonstrate that the proposed adaptation method performs favorably against various baselines and state-of-the-art methods on semantic segmentation.

Within the context of this work, we discuss the domain adaptation methods, including image classification and pixel-level prediction tasks.

In addition, algorithms that are relevant to learning disentangled representations are discussed in this section.

Domain Adaptation.

Domain adaptation approaches have been developed for the image classification task via aligning the feature distributions between the source and target domains.

Conventional methods use hand-crafted features BID10 BID7 to minimize the discrep-ancy across domains, while recent algorithms utilize deep architectures BID8 BID32 to learn domain-invariant features.

One common practice is to adopt the adversarial learning scheme BID9 and minimize the Maximum Mean Discrepancy BID20 .

A number of variants have been developed via designing different classifiers BID21 and loss functions BID33 .

In addition, other recent work aims to enhance feature representations by pixel-level transfer BID1 and domain separation BID0 .Compared to the image classification task, domain adaptation for structured pixel-level predictions has not been widely studied.

BID14 first introduce to tackle the domain adaptation problem on semantic segmentation for road-scene images, e.g., synthetic-to-real images.

Similar to the image classification case, they propose to use adversarial networks and align global feature representations across two domains.

In addition, a category-specific prior is extracted from the source domain and is transferred to the target distribution as a constraint.

However, these priors, e.g., object size and class distribution, may be already inconsistent between two domains.

Instead of designing such constraints, the CDA method BID36 applies the SVM classifier to capture label distributions on superpixels as the property to train the adapted model on the target domain.

Similarly, as proposed in BID4 , a class-wise domain adversarial alignment is performed by assigning pseudo labels to the target data.

Moreover, an object prior is extracted from Google Street View to help alignment for static objects.

The above-mentioned domain adaptation methods on structured output all use a global distribution alignment and some class-specific priors to match statistics between two domains.

However, such class-level alignment does not preserve the structured information like the patches.

In contrast, we propose to learn discriminative representations for patches and use these learned representations to help patch-level alignment.

Moreover, our framework does not require additional priors/annotations and the entire network can be trained in an end-to-end fashion.

Compared to the recently proposed output space adaptation method BID31 ) that also enables end-to-end training, our algorithm focuses on learning patch-level representations that aid the alignment process.

Learning Disentangled Representation.

Learning a latent disentangled space has led to a better understanding for numerous tasks such as facial recognition BID26 , image generation BID3 BID24 , and view synthesis BID18 BID35 .

These approaches use pre-defined factors to learn interpretable representations of the image.

BID18 propose to learn graphic codes that are disentangled with respect to various image transformations, e.g., pose and lighting, for rendering 3D images.

Similarly, BID35 synthesize 3D objects from a single image via an encoder-decoder architecture that learns latent representations based on the rotation factor.

Recently, AC-GAN BID24 ) develops a generative adversarial network (GAN) with an auxiliary classifier conditioned on the given factors such as image labels and attributes.

Although these methods present promising results on using the specified factors and learning a disentangled space to help the target task, they focus on handling the data in a single domain.

Motivated by this line of research, we propose to learn discriminative representations for patches to help the domain adaptation task.

To this end, we take advantages of the available label distributions and naturally utilize them as a disentangled factor, in which our framework does not require to pre-define any factors like conventional methods.

In this section, we describe our proposed domain adaptation framework for predicting structured outputs, our adversarial learning scheme to align distributions across domains, and the use of discriminative representations for patches to help the alignment.

Given the source and target images I s , I t ∈ R H×W ×3 and the source labels Y s , our goal is to align the predicted output distribution O t of the target data with the source distribution O s .

As shown in FIG0 (a), we use a loss function for supervised learning on the source data to predict the structured output, and an adversarial loss is adopted to align the global distribution.

Based on this baseline model, we further incorporate a classification loss in a clustered space to learn patch-level discriminative representations F s from the source output distribution O s , shown in FIG0 (b).

For target data, we employ another adversarial loss to align the patch-level distributions between F s and F t , where the goal is to push F t to be closer to the distribution of F s .Objective Function.

As described in FIG0 (b), we formulate the adaptation task as composed of the following loss functions: DISPLAYFORM0 where L s and L d are supervised loss functions for learning the structured prediction and the discriminative representation on source data, respectively, while Γ denotes the clustering process on the ground truth label distribution.

To align the target distribution, we utilize global and patch-level adversarial loss functions, which are denoted as L g adv and L l adv , respectively.

Here, λ's are the weights for different loss functions.

The following sections describe details of the baseline model and the proposed framework.

Figure 3 shows the main components and loss functions of our method.

We first adopt a baseline model that consists of a supervised cross-entropy loss L s and an output space adaptation module using L g adv for global alignment as shown in FIG0 (a).

The loss L s can be optimized by a fully-convolutional network G that predicts the structured output with the loss summed over the spatial map indexed with h, w and the number of categories C: DISPLAYFORM0 where O s = G(I s ) ∈ (0, 1) is the predicted output distribution through the softmax function and is up-sampled to the size of the input image.

Here, we will use the same h and w as the index for all the formulations.

For the adversarial loss L g adv , we follow the practice of GAN training by optimizing G and a discriminator D g that performs the binary classification to distinguish whether the output prediction is from the source image or the target one.

DISPLAYFORM1 Then we optimize the following min-max problem for G and D g , with inputs to the functions dropped for simplicity: min DISPLAYFORM2 3.3 PATCH-LEVEL ALIGNMENT WITH DISCRIMINATIVE REPRESENTATIONS Figure 1 shows that we may find transferable structured output representations shared across source and target images from smaller patches rather than from the entire image or larger grids.

Based on this observation, we propose to perform a patch-level domain alignment.

Specifically, rather than naively aligning the distributions of all patches between two domains, we first perform clustering Figure 3 : The proposed network architecture that consists of a generator G and a categorization module H for learning discriminative patch representations.

In the clustered space, solid symbols denote source representations and unfilled ones are target representations pulled to the source distribution.on patches from the source-domain examples using ground truth segmentation labels to construct a set of prototypical patch patterns.

Then, we let patches from the target domain adapt to this disentangled (clustered) space of source patch representations by guiding them to select the closest cluster regardless of the spatial location via adversarial objective.

In the following, we describe details of the proposed patch-level alignment.

Learning Discriminative Representations.

In order to learn a disentangled space, class labels BID30 or pre-defined factors BID24 are usually provided as supervisory signals.

However, it is non-trivial to assign some sort of class membership to individual patches of an image.

One may apply unsupervised clustering of image patches using pixel representations, but it is unclear whether the constructed clustering would separate patches in a semantically meaningful way.

In this work, we take advantage of already available per-pixel annotations in the source domain to construct semantically disentangled space of patch representations.

To achieve this, we use label histograms for patches as the disentangled factor.

We first randomly sample patches from source images, use a 2-by-2 grid on patches to extract spatial label histograms, and concatenate them into a vector, where each histogram is a 2 · 2 · C dimensional vector.

Second, we apply K-means clustering on these histograms, whereby the label for any patch can be assigned as the cluster center with the closest distance on the histogram.

To incorporate this clustered space during training the network G on source data, we add a classification module H after the predicted output O s , to simulate the procedure of constructing the label histogram and learn a discriminative representation.

We denote the learned representation as F s = H(G(I s )) ∈ (0, 1) U ×V ×K through the softmax function, where K is the number of clusters.

Here, each data point on the spatial map F s corresponds to a patch of the input image, and we obtain the group label Γ(Y s ) for each patch accordingly.

Then the learning process to construct the clustered space can be formulated as a cross-entropy loss: DISPLAYFORM3 Patch-level Adversarial Alignment.

The ensuing task is to align the representations of target patches to the clustered space constructed in the source domain.

To this end, we utilize another adversarial loss between F s and F t , where F t is generated in the same way as described above.

Our goal is to align patches regardless of where they are located in the image, that is, without the spatial and neighborhood supports.

Thus, we reshape F by concatenating the K-dimensional vectors along the spatial map, which results in U · V independent data points.

We note that a similar effect can be achieved by using a convolution layer with a proper stride and kernel size.

We denote this reshaped data asF and formulate the adversarial objective: DISPLAYFORM4 where D l is the discriminator to classify whether the feature representationF is from the source or the target domain.

Finally, we integrate (5) and (6) into the min-max problem in (4): DISPLAYFORM5 3.4 NETWORK OPTIMIZATION Following the standard procedure for training a GAN BID11 , we alternate the optimization between three steps: 1) update the discriminator D g , 2) update the discriminator D l , and 3) update the network G and H while fixing the discriminators.

Update the Discriminator D g .

We train the discriminator D g to distinguish between the source output distribution (labeled as 1) and the target distribution (labeled as 0).

The maximization problem on D g in FORMULA6 is equivalent to minimizing the binary cross-entropy loss: DISPLAYFORM6 Update the Discriminator D l .

Similarly, we train the discriminator D l to classify whether the feature representationF is from the source or the target domain: DISPLAYFORM7 Update the Network G and H. The goal of this step is to push the target distribution closer to the source distribution using the optimized D g and D l , while maintaining good performance on the main tasks using G and H. As a result, the minimization problem in FORMULA6 is the combination of two supervised loss functions, namely, FORMULA1 and FORMULA4 , with two adversarial loss functions, where the adversarial ones can be expressed as binary cross-entropy loss functions that assign the source label to the target distribution: DISPLAYFORM8 We note that updating H would also update G through back-propagation, and thus the feature representations are enhanced in G. In addition, we only require G during the testing phase, so that runtime is unaffected compared to the baseline approach.

Discriminator.

For the discriminator D g using a spatial map O as the input, we adopt an architecture similar to but use fully-convolutional layers.

It contains 5 convolution layers with kernel size 4 × 4, stride 2 and channel numbers {64, 128, 256, 512, 1}. In addition, a leaky ReLU activation BID22 ) is added after each convolution layer, except the last layer.

For the discriminator D l , input data is a K-dimensional vector and we utilize 3 fully-connected layers similar to BID33 , with leaky ReLU activation and channel numbers {256, 512, 1}. Generator.

The generator consists of the network G with a categorization module H. For a fair comparison, we follow the framework used in BID31 ) that adopts DeepLab-v2 BID2 with the ResNet-101 architecture BID13 pre-trained on ImageNet BID6 ) as our baseline network G. To add the module H on the output prediction O, we first use an adaptive average pooling layer to generate a spatial map, where each data point on the map has a desired receptive field corresponding to the size of extracted patches.

Then this pooled map is fed into two convolution layers and a feature map F is produced with the channel number K. Figure 3 illustrates the main components of the proposed architecture.

Implementation Details.

We implement the proposed framework using the PyTorch toolbox on a single Titan X GPU with 12 GB memory.

To train the discriminators, we use the Adam optimizer BID16 with initial learning rate of 10 −4 and momentums set as 0.9 and 0.99.

For learning the generator, we use the Stochastic Gradient Descent (SGD) solver where the momentum is 0.9, the weight decay is 5 × 10 −4 and the initial learning rate is 2.5 × 10 −4 .

For all Table 1 : Ablation study on GTA5-to-Cityscapes using the ResNet-101 network.

We also show the corresponding loss functions used in each setting.

the networks, we decrease the learning rates using the polynomial decay with a power of 0.9, as described in BID2 .

During training, we use λ d = 0.01, λ g adv = λ l adv = 0.0005 and K = 50 for all the experiments.

Note that we first train the model only using the loss L s for 10K iterations to avoid initially noisy predictions and then train the network using all the loss functions for 100K iterations.

More details of the hyper-parameters such as image and patch sizes are provided in the appendix.

DISPLAYFORM0

We evaluate the proposed framework for domain adaptation on semantic segmentation.

We first conduct an extensive ablation study to validate each component in the algorithm on the GTA5-toCityscapes (synthetic-to-real) scenario.

Second, we show that our method performs favorably against state-of-the-art approaches on numerous benchmark datasets and settings.

We evaluate our domain adaptation method on semantic segmentation under various settings, including synthetic-to-real and cross-city scenarios.

First, we adapt the synthetic GTA5 BID27 dataset to the Cityscapes BID5 dataset that contains real road-scene images.

Similarly, we use the SYNTHIA BID28 dataset with a larger domain gap compared to Cityscapes images.

For the above experiments, we follow BID14 to split the training and test sets.

To overcome the realistic case when two domains are in different cities under various weather conditions, we adapt Cityscapes with sunny images to the Oxford RobotCar BID23 dataset that contains rainy scenes.

We manually select 10 sequences in the Oxford RobotCar dataset annotated with the rainy tag, in which we randomly split them into 7 sequences for training and 3 for testing.

We sequentially sample 895 images as training images and annotate 271 images with per-pixel semantic segmentation ground truth as the test set for evaluation.

The annotated ground truth will be made publicly available.

For all the experiments, intersection-over-union (IoU) ratio is used as the metric to evaluate different methods.

In Table 1 , we conduct an ablation study on the GTA5-to-Cityscapes scenario to understand the impact of different loss functions and design choices in the proposed framework.

Loss Functions.

In the first row of Table 1 , we show different steps of the proposed method, including disentanglement, global alignment, and patch-level alignment.

Interestingly, we find that adding disentanglement without any alignments (L s + L d ) also improves the performance (from 36.6% to 38.8%), which demonstrates that the learned feature representation enhances the discrimination and generalization ability.

Finally, as shown in the last result of the second row, our method that combines both the global and patch-level alignments achieve the highest IoU as 43.2%.Impact on L d and L l adv .

In the first two results of the second row, we conduct experiments to validate the effectiveness of our patch-level alignment.

We show that both losses, L d and L l adv , are necessary to assist this alignment process.

Removing either of them will result in performance loss, i.e., 1.9% and 1.5% lower than our final result.

The reason behind this is that, L d is to construct a clustered space so that L l adv can then effectively perform patch-level alignment in this space.

Without ReshapedF .

In the module H that transforms the output distribution to the clustered space, the features are reshaped as independent data pointsF to remove the spatial relationship and are then used as the input to the discriminator D l .

To validate the usefulness, we show that without the reshaping process, the performance drops 2.4% in IoU. This result matches our assumption that patches with similar representations should be aligned regardless of their locations.

Visualization of Feature Representations.

In FIG1 , we show the t-SNE visualization BID34 of the patch-level features in the clustered space of our method and compare with the one without patch-level adaptation.

The result shows that with adaptation in the clustered space, the features are embedded into groups and the source/target representations overlap to each other well.

Example patch visualizations are provided in the appendix.

In this section, we compare the proposed method with state-of-the-art algorithms under various scenarios, including synthetic-to-real and cross-city cases.

Synthetic-to-real Case.

We first present experimental results for adapting GTA5 to Cityscapes in TAB1 .

The methods in the upper group adopt the VGG-16 architecture as the base network and we show that our approach performs favorably against state-of-the-art adaptations via feature BID14 BID36 , pixel-level BID15 , and output space BID31 alignments.

In the bottom group, we further utilize the stronger ResNet-101 base network and compare our result with BID31 under two settings, i.e., feature and output space adaptations.

We show that the proposed method improves the IoU with a gain of 1.8% and achieves the best IoU on 14 out of the 19 categories.

In TAB2 , we show results for adapting SYNTHIA to Cityscapes and similar improvements are observed comparing with state-of-the-art methods.

In addition, we shows visual comparisons in Figure 5 and more results are presented in the appendix.

Cross-city Case.

Adapting between real images across different cities and conditions is an important scenario for practical applications.

We choose a challenge case where the weather condition is different (i.e., sunny v.s rainy) in two cities by adapting Cityscapes to Oxford RobotCar.

The proposed Target Image Ground Truth Before Adaptation Global Alignment Ours Figure 5 : Example results for GTA5-to-Cityscapes.

Our method often generates the segmentation with more details (e.g., sidewalk and pole) while producing less noisy regions.

BID31 , we run the authors' released code and obtain a mean IoU of 63.6%, which is 1.4% lower than the proposed method.

Further results and comparisons are provided in the appendix.

In this paper, we present a domain adaptation method for structured output via a general framework that combines global and patch-level alignments.

The global alignment is achieved by the output space adaptation, while the patch-level one is performed via learning discriminative representations of patches across domains.

To learn such patch-level representations, we propose to construct a clustered space of the source patches and adopt an adversarial learning scheme to push the target patch distributions closer to the source ones.

We conduct extensive ablation study and experiments to validate the effectiveness of the proposed method under numerous challenges on semantic segmentation, including synthetic-to-real and cross-city scenarios, and show that our approach performs favorably against existing algorithms.

To train the model in an end-to-end manner, we randomly sample one image from each of the source and target domain (i.e., batch size as 1) in a training iteration.

Then we follow the optimization strategy as described in Section 3.4 of the paper.

TAB3 shows the image and patch sizes during training and testing.

Note that, the aspect ratio of the image is always maintained (i.e., no cropping) and then the image is down-sampled to the size as in the table.

BID12 can be used as a loss in our model to push the target feature representation F t to one of the source clusters.

To add this regularization, we replace the adversarial loss on the patch level with an entropy loss as in BID21 (u,v,k) , H is the information entropy function, σ is the softmax function, and τ is the temperature of the softmax.

The model with adding this entropy regularization achieves the IoU as 41.9%, that is lower than the proposed patchlevel adversarial alignment as 43.2%.

The reason is that, different from the entropy minimization approach that does not use the source distribution as the guidance, our model learns discriminative representations for the target patches by pushing them closer to the source distribution in the clustered space guided by the label histogram.

DISPLAYFORM0

In FIG3 , we show example patches from the source and target domains corresponding to the t-SNE visualization.

For each group in the clustered space via t-SNE, we show that source and target patches share high similarity between each other, which demonstrates the effectiveness of the proposed patch-level alignment.

In TAB4 , we present the complete result for adapting Cityscapes (sunny condition) to Oxford RobotCar (rainy scene).

We compare the proposed method with the model without adaptation and the output space adaptation approach BID31 .

More qualitative results are provided in FIG4 and 8.

We provide more visual comparisons for GTA5-to-Cityscapes and SYNTHIA-to-Cityscapes scenarios from Figure 9 to Figure 11 .

In each row, we present the results of the model without adaptation, output space adaptation BID31 , and the proposed method.

We show that our approach often yields better segmentation outputs with more details and produces less noisy regions.

We sequentially show images in a video and their adapted segmentations generated by our method.

Target Image Ground Truth Before Adaptation Global Alignment Ours Figure 9 : Example results of adapted segmentation for the GTA5-to-Cityscapes setting.

For each target image, we show results before adaptation, output space adaptation BID31 , and the proposed method.

Target Image Ground Truth Before Adaptation Global Alignment Ours Figure 10 : Example results of adapted segmentation for the GTA5-to-Cityscapes setting.

For each target image, we show results before adaptation, output space adaptation BID31 , and the proposed method.

Target Image Ground Truth Before Adaptation Global Alignment Ours Figure 11 : Example results of adapted segmentation for the SYNTHIA-to-Cityscapes setting.

For each target image, we show results before adaptation, output space adaptation BID31 , and the proposed method.

@highlight

A domain adaptation method for structured output via learning patch-level discriminative feature representations
In contrast to fully connected networks, Convolutional Neural Networks (CNNs) achieve efficiency by learning weights associated with local filters with a finite spatial extent.

An implication of this is that a filter may know what it is looking at, but not where it is positioned in the image.

Information concerning absolute position is inherently useful, and it is reasonable to assume that deep CNNs may implicitly learn to encode this information if there is a means to do so.

In this paper, we test this hypothesis revealing the surprising degree of absolute position information that is encoded in commonly used neural networks.

A comprehensive set of experiments show the validity of this hypothesis and shed light on how and where this information is represented while offering clues to where positional information is derived from in deep CNNs.

Convolutional Neural Networks (CNNs) have achieved state-of-the-art results in many computer vision tasks, e.g. object classification (Simonyan & Zisserman, 2014; and detection (Redmon et al., 2015; , face recognition (Taigman et al., 2014) , semantic segmentation (Long et al., 2015; Chen et al., 2018; Noh et al., 2015) and saliency detection (Cornia et al., 2018; Li et al., 2014) .

However, CNNs have faced some criticism in the context of deep learning for the lack of interpretability (Lipton, 2016) .

The classic CNN model is considered to be spatially-agnostic and therefore capsule (Sabour et al., 2017) or recurrent networks (Visin et al., 2015) have been utilized to model relative spatial relationships within learned feature layers.

It is unclear if CNNs capture any absolute spatial information which is important in position-dependent tasks (e.g. semantic segmentation and salient object detection).

As shown in Fig. 1 , the regions determined to be most salient (Jia & Bruce, 2018) tend to be near the center of an image.

While detecting saliency on a cropped version of the images, the most salient region shifts even though the visual features have not been changed.

This is somewhat surprising, given the limited spatial extent of CNN filters through which the image is interpreted.

In this paper, we examine the role of absolute position information by performing a series of randomization tests with the hypothesis that CNNs might indeed learn to encode position information as a cue for decision making.

Our experiments reveal that position information is implicitly learned from the commonly used padding operation (zero-padding).

Zero-padding is widely used for keeping the same dimensionality when applying convolution.

However, its hidden effect in representation learning has been long omitted.

This work helps to better understand the nature of the learned features in CNNs and highlights an important observation and fruitful direction for future investigation.

Previous works try to visualize learned feature maps to demystify how CNNs work.

A simple idea is to compute losses and pass these backwards to the input space to generate a pattern image that can maximize the activation of a given unit (Hinton et al., 2006; Erhan et al., 2009) .

However, it is very difficult to model such relationships when the number of layers grows.

Recent work (Zeiler & Fergus, 2013 ) presents a non-parametric method for visualization.

A deconvolutional network (Zeiler et al., 2011 ) is leveraged to map learned features back to the input space and their results reveal what types of patterns a feature map actually learns.

Another work (Selvaraju et al., 2016) proposes to combine pixel-level gradients with weighted class activation mapping to locate the region which maximizes class-specific activation.

As an alternative to visualization strategies, an empirical study (Zhang et al., 2016) has shown that a simple network can achieve zero training Cropping results in a shift in position rightward of features relative to the centre.

It is notable that this has a significant impact on output and decision of regions deemed salient despite no explicit position encoding and a modest change to position in the input.

loss on noisy labels.

We share the similar idea of applying a randomization test to study the CNN learned features.

However, our work differs from existing approaches in that these techniques only present interesting visualizations or understanding, but fail to shed any light on spatial relationships encoded by a CNN model.

In summary, CNNs have emerged as a way of dealing with the prohibitive number of weights that would come with a fully connected end-to-end network.

A trade-off resulting from this is that kernels and their learned weights only have visibility of a small subset of the image.

This would seem to imply solutions where networks rely more on cues such as texture and color rather than shape (Baker et al., 2018) .

Nevertheless, position information provides a powerful cue for where objects might appear in an image (e.g. birds in the sky).

It is conceivable that networks might rely sufficiently on such cues that they implicitly encode spatial position along with the features they represent.

It is our hypothesis that deep neural networks succeed in part by learning both what and where things are.

This paper tests this hypothesis, and provides convincing evidence that CNNs do indeed rely on and learn information about spatial positioning in the image to a much greater extent than one might expect.

CNNs naturally try to extract fine-level high spatial-frequency details (e.g. edges, texture, lines) in the early convolutional stages while at the deepest layers of encoding the network produces the richest possible category specific features representation Simonyan & Zisserman (2014) ; ; Badrinarayanan et al. (2017) .

In this paper, we propose a hypothesis that position information is implicitly encoded within the extracted feature maps and plays an important role in classifying, detecting or segmenting objects from a visual scene.

We therefore aim to prove this hypothesis by predicting position information from different CNN archetypes in an end-to-end manner.

In the following sections, we first introduce the problem definition followed by a brief discussion of our proposed position encoding network.

Problem Formulation:

Given an input image I m ??? R h??w??3 , our goal is to predict a gradient-like position information maskf p ??? R h??w where each pixel value defines the absolute coordinates of an pixel from left???right or top???bottom.

We generate gradient-like masks G pos ??? R h??w (Sec. 2.2) for supervision in our experiments, with weights of the base CNN archetypes being fixed.

Our Position Encoding Network (PosENet) (See Fig. 2 ) consists of two key components: a feedforward convolutional encoder network f enc and a simple position encoding module, denoted as f pem .

The encoder network extracts features at different levels of abstraction, from shallower to deeper layers.

The position encoding module takes multi-scale features from the encoder network as input and predicts the absolute position information at the end.

Encoder: We use ResNet and VGG based architectures to build encoder networks (f enc ) by removing the average pooling layer and the layer that assigns categories.

As shown in Fig. 2 , the encoder module consists of five feature extractor blocks denoted by (f where W a denotes weights that are frozen.

* denotes the convolution operation.

Note that in probing the encoding network, only the position encoding module f pem is trained to focus on extracting position information while the encoder network is forced to maintain their existing weights.

Once we have the same spatial dimension for multi-scale features, we concatenate them together followed by a sequence of k ?? k convolution operations.

In our experiments, we vary the value of k between {1, 3, 5, 7} and most experiments are carried out with a single convolutional layer in the position encoding module f pem .

The key operations can be summarized as follows:

where W c pos is the trainable weights attached with the transformation function T pos .

The main objective of the encoding module is to validate whether position information is implicitly learned when trained on categorical labels.

Additionally, the position encoding module models the relationship between hidden position information and the gradient like ground-truth mask.

The output is expected to be random if there is no position information encoded in the features maps and vice versa (ignoring any guidance from image content).

To validate the existence of position information in a network, we implement a randomization test by assigning a normalized gradient-like 1 position map as ground-truth shown in Fig. 3 .

We first generate gradient-like masks in Horizontal (H) and vertical (V) directions.

Similarly, we apply a Gaussian filter to design another type of ground-truth map, Gaussian distribution (G) .

The key motivation of generating these three patterns is to validate if the model can learn absolute position on one or two axes.

Additionally, We also create two types of repeated patterns, horizontal and vertical stripes, (HS, VS).

Regardless of the direction, the position information in the multi-level features is likely to be modelled through a transformation by the encoding module f pem .

Our design of gradient ground-truth can be considered as a type of random label because there is no correlation between the input image and the ground-truth with respect to position.

Since the extraction of position information is independent of the content of images, we can choose any image datasets.

Meanwhile, we also build synthetic images, e.g. black, white and Gaussian noise to validate our hypothesis.

As we implicitly aim to encode the position information from a pretrained network, we freeze the encoder network f enc in all of our experiments.

Our position encoding module f pem generates the position mapf p of interest.

During training, for a given input image I m ??? R h??w??3 and associated ground-truth position map G h pos , we apply the supervisory signal onf p by upsampling it to the size of G h pos .

Then, we define a pixel-wise mean squared error loss to measure the difference between predicted and ground-truth position maps.

The overall objective function of our network can be written as:

where x ??? IR n and y ??? IR n (n denotes the spatial resolution) are the vectorized predicted position and ground-truth map respectively.

x i and y i refer to a pixel off p and G (2014) dataset.

The synthetic images (white, black and Gaussian noise) are also used as described in Section 2.2.

Note that we follow the common setting used in saliency detection just to make sure that there is no overlap between the training and test sets.

However, any images can be used in our experiments given that the position information is relatively content independent.

Evaluation Metrics:

As position encoding measurement is a new direction, there is no universal metric.

We use two different natural choices for metrics (Spearmen Correlation (SPC) and Mean Absoute Error (MAE)) to measure the position encoding performance.

The SPC is defined as the Spearman's correlation between the ground-truth and the predicted position map.

For ease of interpretation, we keep the SPC score within range [-1 1] .

MAE is the average pixel-wise difference between the predicted position map and the ground-truth gradient position map.

We initialize the architecture with a network pretrained for the ImageNet classification task.

The new layers in the position encoding branch are initialized with xavier initialization Glorot & Bengio (2010) .

We train the networks using stochastic gradient descent for 15 epochs with momentum of 0.9, and weight decay of 1e???4.

We resize each image to a fixed size of 224??224 during training and inference.

Since the spatial extent of multi-level features are different, we align all the feature maps to a size of 28 ?? 28.

We report experimental results for the following baselines that are described as follows: VGG indicates PosENet is based on the features extracted from the VGG16 model.

Similarly, ResNet represents the combination of ResNet-152 and PosENet.

PosENet alone denotes only the PosENet model is applied to learn position information directly from the input image.

H, V, G, HS and VS represent the five different ground-truth patterns, horizontal and vertical gradients, 2D Gaussian distribution, horizontal and vertical stripes respectively.

Position Information in Pretrained Models: We first conduct experiments to validate the existence of position information encoded in a pretrained model.

Following the same protocol, we train the VGG and ResNet based networks on each type of the ground-truth and report the experimental results in Table 1 .

We also report results when we only train PosENet without using any pretrained model to justify that the position information is not driven from prior knowledge of objects.

Our experiments do not focus on achieving higher performance on the metrics but instead validate how much position information a CNN model encodes or how easily PosENet can extract this information.

Note that, we only use one convolutional layer with a kernel size of 3 ?? 3 without any padding in the PosENet for this experiment.

As shown in PosENet can extract position information consistent with the ground-truth position map only when coupled with a deep encoder network.

As mentioned prior, the generated ground-truth map can be considered as a type of randomization test given that the correlation with input has been ignored Zhang et al. (2016) .

Nevertheless, the high performance on the test sets across different groundtruth patterns reveals that the model is not blindly overfitting to the noise and instead is extracting true position information.

However, we observe low performance on the repeated patterns (HS and VS) compared to other patterns due to the model complexity and specifically the lack of correlation between ground-truth and absolute position (last two rows of Table 1 ).

The H pattern can be seen as one quarter of a sine wave whereas the striped patterns (HS and VS) can be considered as repeated periods of a sine wave which requires a deeper comprehension.

The qualitative results for several architectures across different patterns are shown in Fig. 4 .

We can see the correlation between the predicted and the ground-truth position maps corresponding to H, G and HS patterns, which further reveals the existence of position information in these networks.

The quantitative and qualitative results strongly validate our hypothesis that position information is implicitly encoded in every architecture without any explicit supervision towards this objective.

Moreover, PosENet alone shows no capacity to output a gradient map based on the synthetic data.

We further explore the effect of image semantics in Sec. 4.1.

It is interesting to note the performance gap among different architectures specifically the ResNet based models achieve higher performance than the VGG16 based models.

The reason behind this could be the use of different convolutional kernels in the architecture or the degree of prior knowledge of the semantic content.

We show an ablation study in the next experiment for further investigation.

For the rest of this paper, we only focus on the natural images, PASCAL-S dataset, and three representative patterns, H, G and HS.

In this section, we conduct ablation studies to examine the role of the proposed position encoding network by highlighting two key design choices.

(1) the role of varying kernel size in the position encoding module and (2) stack length of convolutional layers we add to extract position information from the multi-level features.

Table 1 show the existence of position information learned from an object classification task.

In this experiment, we change the design of PosENet to examine if it is possible to extract hidden position information more accurately.

The PosENet used in the prior experiment (Table 1) has only one convolutional layer with a kernel size of 3 ?? 3.

Here, we apply a stack of convolutional layers of varying length to the PosENet and report the experimental results in Table 2 (a).

Even though the stack size is varied, we aim to retain a relatively simple PosENet to only allow efficient readout of positional information.

As shown in Table 2 , we keep the kernel size fixed at 3 ?? 3 while stacking multiple layers.

Applying more layers in the PosENet can improve the readout of position information for all the networks.

One reason could be that stacking multiple convolutional filters allows the network to have a larger effective receptive field, for example two 3 ?? 3 convolution layers are spatially equal to one 5 ?? 5 convolution layer Simonyan & Zisserman (2014) .

An alternative possibility is that positional information may be represented in a manner that requires more than first order inference (e.g. a linear readout).

Our previous experiments reveal that the position information is encoded in a pretrained CNN model.

It is also interesting to see whether position information is equally distributed across the layers.

In this experiment, we train PosENet on each of the extracted features, f separately using VGG16 to examine which layer encodes more position information.

Similar to Sec. 3.3, we only apply one 3 ?? 3 kernel in F pem to obtain the position map.

As shown in Table 3 , the VGG based PosENet with top f 5 pos features achieves higher performance compared to the bottom f 1 pos features.

This may partially a result of more feature maps being extracted from deeper as opposed to shallower layers, 512 vs 64 respectively.

However, it is likely indicative of stronger encoding of the positional information in the deepest layers of the network where this information is shared by high-level semantics.

We further investigate this effect for VGG16 where the top two layers (f

We believe that the padding near the border delivers position information to learn.

Zero-padding is widely used in convolutional layers to maintain the same spatial dimensions for the input and output, with a number of zeros added at the beginning and at the end of both axes, horizontal and vertical.

To validate this, we remove all the padding mechanisms implemented within VGG16 but still initialize the model with the ImageNet pretrained weights.

Note that we perform this experiment only using VGG based PosENet since removing padding on ResNet models will lead to inconsistent sizes of Table 4 : Quantitative comparison subject to padding in the convolution layers used in PosENet and VGG (w/o and with zero padding) on natural images.

skip connections.

We first test the effect of zero-padding used in VGG, no padding used in PosENet.

As we can see from Table 4 , the VGG16 model without zero-padding achieves much lower performance than the default setting(padding=1) on the natural images.

Similarly, we introduce position information to the PosENet by applying zero-padding.

PosENet with padding=1 (concatenating one zero around the frame) achieves higher performance than the original (padding=0).

When we set padding=2, the role of position information is more obvious.

This also validates our experiment in Section 3.3, that shows PosENet is unable to extract noticeable position information because no padding was applied, and the information is encoded from a pretrained CNN model.

This is why we did not apply zero-padding in PosENet in our previous experiments.

Moreover, we aim to explore how much position information is encoded in the pretrained model instead of directly combining with the PosENet.

Fig. 6 illustrates the impact of zero-padding on encoding position information subject to padding using a Gaussian pattern.

Recall that the position information is considered to be content independent but our results in Table 1 show that semantics within an image may affect the position map.

To visualize the impact of semantics, we compute the content loss heat map using the following equation: As shown in Figure 7 , the heatmaps of PosENet have larger content loss around the corners.

While the loss maps of VGG and ResNet correlate more with the semantic content.

Especially for ResNet, the deeper understanding of semantic content leads to a stronger interference in generating a smooth gradient.

The highest losses are from the face, person, cat, airplane and vase respectively (from left to right).

This visualization can be an alternative method to show which regions a model focuses on, especially in the case of ResNet.

Saliency Detection: We further validate our findings in the position-dependent tasks (semantic segmentation and salient object detection (SOD)).

First, we train the VGG network with and without zero-padding from scratch to validate if the position information delivered by zero-padding is critical for detecting salient regions.

For these experiments, we use the publicly available MSRA dataset Cheng et al. (2015) as our SOD training set and evaluate on three other datasets (ECSSD, PASCAL-S, and DUT-OMRON).

From Table 5 (a), we can see that VGG without padding achieves much worse results on both of the metrics (F-measure and MAE) which further validates our findings that zero-padding is the key source of position information.

Semantic Segmentation: We also validate the impact of zero-padding on the semantic segmentation task.

We train the VGG16 network with and without zero padding on the training set of PASCAL VOC 2012 dataset and evaluate on the validation set.

Similar to SOD, the model with zero padding significantly outperforms the model with no padding.

We believe that CNN models pretrained on these two tasks can learn more position information than classification task.

To validate this hypothesis, we take the VGG model pretrained on ImageNet as our baseline.

Meanwhile, we train two VGG models on the tasks of semantic segmentation and saliency detection from scratch, denoted as VGG-SS and VGG-SOD respectively.

Then we finetune these three VGG models following the protocol used in Section 3.3.

From Table 6 , we can see that the VGG-SS and VGG-SOD models outperform VGG by a large margin.

These experiments further reveal that the zero-padding strategy plays an important role in a position-dependent task, an observation that has been long-ignored in neural network solutions to vision problems.

Table 6 : Comparison of VGG models pretrained for classification, SOD, and semantic segmentation.

In this paper we explore the hypothesis that absolute position information is implicitly encoded in convolutional neural networks.

Experiments reveal that positional information is available to a strong degree.

More detailed experiments show that larger receptive fields or non-linear readout of positional information further augments the readout of absolute position, which is already very strong from a trivial single layer 3 ?? 3 PosENet.

Experiments also reveal that this recovery is possible when no semantic cues are present and interference from semantic information suggests joint encoding of what (semantic features) and where (absolute position).

Results point to zero padding and borders as an anchor from which spatial information is derived and eventually propagated over the whole image as spatial abstraction occurs.

These results demonstrate a fundamental property of CNNs that was unknown to date, and for which much further exploration is warranted.

@highlight

Our work shows positional information has been implicitly encoded in a network. This information is important for detecting position-dependent features, e.g. semantic and saliency.
In this paper, we introduce a method to compress intermediate feature maps of deep neural networks (DNNs) to decrease memory storage and bandwidth requirements during inference.

Unlike previous works, the proposed method is based on converting fixed-point activations into vectors over the smallest GF(2) finite field followed by nonlinear dimensionality reduction (NDR) layers embedded into a DNN.

Such an end-to-end learned representation finds more compact feature maps by exploiting quantization redundancies within the fixed-point activations along the channel or spatial dimensions.

We apply the proposed network architecture to the tasks of ImageNet classification and PASCAL VOC object detection.

Compared to prior approaches, the conducted experiments show a factor of 2 decrease in memory requirements with minor degradation in accuracy while adding only bitwise computations.

Recent achievements of deep neural networks (DNNs) make them an attractive choice in many computer vision applications including image classification BID6 and object detection BID9 .

The memory and computations required for DNNs can be excessive for low-power deployments.

In this paper, we explore the task of minimizing the memory footprint of DNN feature maps during inference and, more specifically, finding a network architecture that uses minimal storage without introducing a considerable amount of additional computations or on-the-fly heuristic encoding-decoding schemes.

In general, the task of feature map compression is tightly connected to an input sparsity.

The input sparsity can determine several different usage scenarios.

This may lead to substantial decrease in memory requirements and overall inference complexity.

First, a pen sketches are spatially sparse and can be processed efficiently by recently introduced submanifold sparse CNNs BID4 .

Second, surveillance cameras with mostly static input contain temporal sparsity that can be addressed by Sigma-Delta networks BID15 .

A more general scenario presumes a dense input e.g. video frames from a high-resolution camera mounted on a moving autonomous car.

In this work, we address the latter scenario and concentrate on feature map compression in order to minimize memory footprint and bandwidth during DNN inference which might be prohibitive for high-resolution cameras.

We propose a method to convert intermediate fixed-point feature map activations into vectors over the smallest finite field called the Galois field of two elements (GF(2)) or, simply, binary vectors followed by compression convolutional layers using a nonlinear dimensionality reduction (NDR) technique embedded into DNN architecture.

The compressed feature maps can then be projected back to a higher cardinality representation over a fixed-point (integer) field using decompression convolutional layers.

Using a layer fusion method, only the compressed feature maps need to be kept for inference while adding only computationally inexpensive bitwise operations.

Compression and decompression layers over GF(2) can be repeated within the proposed network architecture and trained in an end-to-end fashion.

In brief, the proposed method resembles autoencoder-type BID7 structures embedded into a base network that work over GF(2).

Binary conversion and compression-decompression layers are implemented in the Caffe BID12 framework and publicly available 1 .The rest of the paper is organized as follows.

Section 2 reviews related work.

Section 3 gives notation for convolutional layers, describes conventional fusion and NDR methods, and explains the proposed method including details about network training and the derived architecture.

Section 4 presents experimental results on ImageNet classification and PASCAL VOC object detection using SSD BID13 , memory requirements, and obtained compression rates.

Feature Map Compression using Quantization.

Unlike a weight compression, surprisingly few papers consider feature map compression.

This can most likely be explained by the fact that feature maps have to be compressed for every network input as opposed to offline weight compression.

Previous feature map compression methods are primarily developed around the idea of representation approximation using a certain quantization scheme: fixed-point quantization BID2 BID5 , binary quantization BID10 BID17 BID20 BID19 , and power-of-two quantization BID14 .

The base floatingpoint network is converted to the approximate quantized representation and, then, the quantized network is retrained to restore accuracy.

Such methods are inherently limited in finding more compact representations since the base architecture remains unchanged.

For example, the dynamic fixed-point scheme typically requires around 8-bits of resolution to achieve baseline accuracy for state-of-the-art network architectures.

At the same time, binary networks experience significant accuracy drops for large-scale datasets or compact (not over-parametrized) network architectures.

Instead, our method can be considered in a narrow sense as a learned quantization using binary representation.

Embedded NDR Layers.

Another interesting approach is implicitly proposed by BID11 .

Although the authors emphasized weight compression rather than feature map compression, they introduced NDR-type layers into network architecture that allowed to decrease not only the number of weights but also feature map sizes by a factor of 8, if one keeps only the outputs of socalled "squeeze" layers.

The latter is possible because such network architecture does not introduce any additional convolution recomputations since "squeeze" layer computations with a 1×1 kernel can be fused with the preceding "expand" layers.

Our work goes beyond this approach by extending NDR-type layers to work over GF(2) to find a more compact feature map representation.

Hardware Accelerator Architectures.

BID8 estimated that off-chip DRAM access requires approximately 100× more power than local on-chip cache access.

Therefore, currently proposed DNN accelerator architectures propose various schemes to decrease memory footprint and bandwidth.

One obvious solution is to keep only a subset of intermediate feature maps at the expense of recomputing convolutions BID0 .

The presented fusion approach seems to be oversimplified but effective due to high memory access cost.

Our approach is complementary to this work but proposes to keep only compressed feature maps with minimum additional computations.

Another recent work BID16 exploits weight and feature map sparsity using a more efficient encoding for zeros.

While this approach targets similar goals, it requires having high sparsity, which is often unavailable in the first and the largest feature maps.

In addition, a special control and encoding-decoding logic decrease the benefits of this approach.

In our work, compressed feature maps are stored in a dense form without the need of special control and enconding-decoding logic.

The input feature map of lth convolutional layer in commonly used DNNs can be represented by a tensor X l−1 ∈ RĆ ×H×W , whereĆ, H and W are the number of input channels, the height and the width, respectively.

The input X l−1 is convolved with a weight tensor W l ∈ R C×Ć×H f ×W f , where C is the number of output channels, H f and W f are the height and the width of filter kernel, respectively.

A bias vector b ∈ R C is added to the result of convolution operation.

Once all C channels are computed, an element-wise nonlinear function is applied to the result of the convolution operations.

Then, the cth channel of the output tensor X l ∈ R C×H×W can be computed as DISPLAYFORM0 where * denotes convolution operation and g() is some nonlinear function.

In this paper, we assume g() is the most commonly used rectified linear unit (ReLU) defined as g(x) = max (0, x) such that all activations are non-negative.

We formally describe previously proposed methods briefly reviewed in Section 2 using the unified model illustrated in FIG0 .

To simplify notation, biases are not shown.

Consider a network built using multiple convolutional layers and processed according to (1).

Similar to BID0 , calculation of N sequential layers can be fused together without storing intermediate feature maps DISPLAYFORM0 For example, fusion can be done in a channel-wise fashion using memory buffers which are much smaller than the whole feature map.

Then, feature map X l ∈ R can be quantized intoX l ∈ Q using a nonlinear quantization function q() where Q is a finite field over integers.

The quantization step may introduce a drop in accuracy due to imperfect approximation.

The network can be further finetuned to restore some of the original accuracy BID2 BID5 .

The network architecture is not changed after quantization and feature maps can be compressed only up to a certain suboptimal bitwidth resolution.

The next step implicitly introduced by Iandola et al. FORMULA0 is to perform NDR using an additional convolutional layer.

A mappingX DISPLAYFORM1 ×H×W can be performed using projection weights P l ∈ RC ×C×H f ×W f , where the output channel dimensionC < C. Then, only compressed feature mapŶ l needs to be stored in the memory buffer.

During the inverse steps, the compressed feature map can be projected back onto the higher-dimensional tensorX l+1 ∈ Q using weights R l ∈ R C×C×H f ×W f and, lastly, converted back to X l+1 ∈ R using an inverse quantization function q −1 ().

In the case of a fully quantized network, the inverse quantization can be omitted.

In practice, the number of bits for the feature map quantization step depends on the dataset, network architecture and desired accuracy.

For example, over-parameterized architecture like AlexNet may require only 1 or 2 bits for small-scale datasets (CIFAR-10, MNIST, SVHN), but experience significant accuracy drops for large-scale datasets like ImageNet.

In particular, the modified AlexNet (with the first and last layers kept in full-precision) top-1 accuracy is degraded by 12.4% and 6.8% for 1-bit XNOR-Net BID17 and 2-bit DoReFa-Net BID20 , respectively.

At the same time, efficient network architectures e.g. BID11 using NDR layers require 6-8 bits for the fixed-point quantization scheme on ImageNet and fail to work with lower precision activations.

In this paper, we follow the path to select an efficient base network architecture and then introduce additional compression layers to obtain smaller feature maps as opposed to initially selecting an over-parametrized network architecture for quantization.

Consider a scalar x from X l ∈ R. A conventional feature map quantization step can be represented as a scalar-to-scalar mapping or a nonlinear functionx = q(x) such that DISPLAYFORM0 wherex is the quantized scalar, Q is the GF(2 B ) finite field for fixed-point representation and B is the number of bits.

We can introduce a newx representation by a linear binarization function b() defined bŷ DISPLAYFORM1 where ⊗ is a bitwise AND operation, DISPLAYFORM2 An inverse linear function b −1 () can be written as DISPLAYFORM3 Equations FORMULA4 - FORMULA6 show that a scalar over a higher cardinality finite field can be linearly converted to and from a vector over a finite field with two elements.

Based on these derivations, we propose a feature map compression method shown in Figure 2 .

Similar to BID2 , we quantize activations to obtainX l and, then, apply transformation (3).

The resulting feature map can be represented asX DISPLAYFORM4 B×C×H×W .

For implementation convenience, a new bit dimension can be concatenated along channel dimension resulting in the feature mapX l ∈ B BC×H×W .

Next, a single convolutional layer using weights P l or a sequence of layers with P l i weights can be applied to obtain a compressed representation over GF(2).

Using the fusion technique, only the compressed feature mapsỸ l ∈ B need to be stored in memory during inference.

Non-compressed feature maps can be processed using small buffers e.g. in a sequential channel-wise fashion.

Lastly, the inverse function b −1 () from (4) using convolutional layers R l i and inverse of quantization q −1 () undo the compression and quantization steps.

Figure 2: Scheme of the proposed method.

The graph model shown in FIG1 explains details about the inference (forward pass) and backpropagation (backward pass) phases of the newly introduced functions.

The inference phase represents (3)-(4) as explained above.

Clearly, the backpropagation phase may seem not obvious at a first glance.

One difficulty related to the quantized network is that quantization function itself is not differentiable.

But many studies e.g. BID2 show that a mini-batch-averaged floating-point gradient practically −1 () can be represented as gates that make hard decisions similar to ReLU.

The gradient of b −1 () can then be calculated using results of BID1 aŝ DISPLAYFORM0 Lastly, the gradient of b() is just a scaled sum of the gradient vector calculated bỹ DISPLAYFORM1 where x 0 is a gradient scaling factor that represents the number of nonzero elements inx .

Practically, the scaling factor can be calculated based on statistical information only once and used as a static hyperparameter for gradient normalization.

Since the purpose of the network is to learn and keep only the smallestỸ l , the choice of P l and R l initialization is important.

Therefore, we can initialize these weight tensors by an identity function that maps the non-compressed feature map to a truncated compressed feature map and vice versa.

That provides a good starting point for training.

At the same time, other initializations are possible e.g. noise sampled from some distribution studied by BID1 can be added as well.

To highlight benefits of the proposed approach, we select a base network architecture with the compressed feature maps according to Section 3.2.

The selected architecture is based on a quantized SqueezeNet network.

It consists of a sequence of "fire" modules where each module contains two concatenated "expand" layers and a "squeeze" layer illustrated in FIG2 .

The "squeeze" layers perform NDR over the field of real or, in case of the quantized model, integer numbers.

Specifically, the size of concatenated "expand 1×1" and "expand 3×3" layers is compressed by a factor of 8 along channel dimension by "squeeze 1×1" layer.

Activations of only the former one can be stored during inference according to the fusion method.

According to the analysis presented in BID5 , activations quantized to 8-bit integers do not experience significant accuracy drop.

The given base architecture is extended by the proposed method.

The quantized "squeeze" layer feature map is converted to its binary representation following Figure 2 .

Then, the additional compression rate is defined by selecting parameters of P l i .

In the simplest case, only a single NDR layer can be introduced with the weights P l .

In general, a number of NDR layers can be added with 1×1, 3×3 and other kernels with or without pooling at the expense of increased computational cost.

For example, 1×1 kernels allow to learn optimal quantization and to compensate redundancies along channel dimension only.

But 3×3 kernels can address spatial redundancies and, while being implemented with stride 2 using convolutional-deconvolutional layers, decrease feature map size along spatial dimensions.

We implemented the new binarization layers from Section 3 as well as quantization layers using modified BID5 code in the Caffe BID12 framework.

The latter code is modified to accurately support binary quantization during inference and training.

SqueezeNet v1.1 is selected as a base floating-point network architecture, and its pretrained weights were downloaded from the publicly available source 2 .

In the experiments, we compress the "fire2/squeeze" and "fire3/squeeze" layers which are the largest of the "squeeze" layers due to high spatial dimensions.

The input to the network has a resolution of 227×227, and the weights are all floating-point.

The quantized and compressed models are retrained for 100,000 iterations with a mini-batch size of 1024 on the ImageNet BID18 (ILSVRC2012) training dataset, and SGD solver with a step-policy learning rate starting from 1e-3 divided by 10 every 20,000 iterations.

Although this large mini-batch size was used by the original model, it helps the quantized and compressed models to estimate gradients as well.

The compressed models were derived and retrained iteratively from the 8-bit quantized model.

TAB0 reports top-1 and top-5 inference accuracies of 50,000 images from ImageNet validation dataset.

The fourth column in TAB0 represents speed in terms of frames per second (FPS) of network forward pass with mini-batch size of 1 using NVIDIA Tesla P100 GPU.

All quantized computations are emulated on GPU using fp32.

Hence, it shows a measure of emulation speed for additional layers rather than speed of the optimized layers.

According to TAB0 , the quantized models after retraining experience -0.2%, 0.6% and 3.5% top-1 accuracy drops for 8-bit, 6-bit and 4-bit quantization, respectively.

For comparison, the quantized models without retraining are shown in parentheses.

The proposed compression method using 1 × 1 kernels allows us to restore corresponding top-1 accuracy by 1.0% and 2.4% for 6-bit and 4-bit versions at the expense of a small increase in the number of weights and bitwise convolutions.

Moreover, we evaluated a model with a convolutional layer followed by a deconvolutional layer both with a 3 × 3 stride 2 kernel at the expense of a 47% increase in weight size for 6-bit activations.

That allowed us to decrease feature maps in spatial dimensions by exploiting local spatial quantization redundancies.

Then, the feature map size is further reduced by a factor of 4, while top-1 accuracy dropped by 4.3% and 4.6% for 8-bit and 6-bit activations, respectively.

A comprehensive comparison with the state-of-the-art networks is given in Appendix A.

Accuracy experiments.

We evaluate object detection using Pascal VOC BID3 dataset which is a more realistic application for autonomous cars where the high-resolution cameras emphasize feature map compression benefits.

The VOC2007 test dataset contains 4,952 images and a training dataset of 16,551 images is a union of VOC2007 and VOC2012.

We adopted SSD512 model BID13 for the proposed architecture.

SqueezeNet pretrained on ImageNet is used as a feature extractor instead of the original VGG-16 network.

This reduces number of parameters and overall inference time by a factor of 4 and 3, respectively.

The original VOC images are rescaled to 512×512 resolution.

As with ImageNet experiments, we generated several models for comparisons: a base floating-point model, quantized models, and compressed models.

We apply quantization and compression to the "fire2/squeeze" and "fire3/squeeze" layers which represent, if the fusion technique is applied, more than 80% of total feature map memory due to their large spatial dimensions.

Typically, spatial dimensions decrease quadratically because of max pooling layers compared to linear growth in the depth dimension.

The compressed models are derived from the 8-bit quantized model, and both are retrained for 10,000 mini-batch-256 iterations using SGD solver with a step-policy learning rate starting from 1e-3 divided by 10 every 2,500 iterations.

TAB1 presents mean average precision (mAP) results for the original VGG-16 and SqueezeNetbased models as well as size of the weights, feature maps to compress and the speed metric with Section 4.1 assumptions.

The 8-bit quantized model with retraining drops accuracy by less than 0.04%, while 6-bit, 4-bit and 2-bit models decrease accuracy by 0.5%, 2.2% and 12.3%, respectively.

For reference, mAPs for the quantized models without retraining are shown in parentheses.

Using the proposed compression-decompression layers with a 1×1 kernel, mAP for the 6-bit model is increased by 0.5% and mAP for the 4-bit is decreased by 0.5%.

We conclude that compression along channel dimension is not beneficial for SSD unlike ImageNet classification either due to low quantization redundancy in that dimension or the choice of hyper-parameters e.g. mini-batch size.

Then, we evaluate the models with spatial-dimension compression which is intuitively appealing for high-resolution images.

Empirically, we found that a 2×2 kernel with stride 2 performs better than a corresponding 3×3 kernel while requiring less parameters and computations.

According to TAB1 , an 8-bit model with 2×2 kernel and downsampling-upsampling layers achieves 1% higher mAP than a model with 3×3 kernel and only 3.7% lower than the base floating-point model.

Memory Requirements.

TAB2 summarizes memory footprint benefits for the evaluated SSD models.

Similar to the previous section, we consider only the largest feature maps that represent more than 80% of total activation memory.

Assuming that the input frame is stored separately, the fusion technique allows to compress feature maps by a factor of 19.

Note that no additional recomputations are needed.

Second, conventional 8-bit and 4-bit fixed-point models decrease the size of feature maps by a factor of 4 and 8, respectively.

Third, the proposed model with 2×2 stride 2 kernel gains another factor of 2 compression compared to 4-bit fixed-point model with only 1.5% degradation in mAP.

This result is similar to ImageNet experiments which showed relatively limited compression gain along channel dimension only.

At the same time, learned quantization along combined channel and spatial dimensions pushes further compression gain.

In total, the memory footprint for this feature extractor is reduced by two orders of magnitude.

We introduced a method to decrease memory storage and bandwidth requirements for DNNs.

Complementary to conventional approaches that use fused layer computation and quantization, we presented an end-to-end method for learning feature map representations over GF(2) within DNNs.

Such a binary representation allowed us to compress network feature maps in a higher-dimensional space using autoencoder-inspired layers embedded into a DNN along channel and spatial dimensions.

These compression-decompression layers can be implemented using conventional convolutional layers with bitwise operations.

To be more precise, the proposed representation traded cardinality of the finite field with the dimensionality of the vector space which makes possible to learn features at the binary level.

The evaluated compression strategy for inference can be adopted for GPUs, CPUs or custom accelerators.

Alternatively, existing binary networks can be extended to achieve higher accuracy for emerging applications such as object detection and others.

We compare recently reported ImageNet results for networks that compress feature maps as well as several configurations of the proposed approach.

Most of the works use the over-parametrized AlexNet architecture while ours is based on SqueezeNet.

TAB3 shows accuracy results for base networks as well as their quantized versions.

Binary XNOR-Net BID17 estimates based on AlexNet as well as ResNet-18.

DoReFa-Net BID20 ) is more flexible and can adjust the number of bits for weights and activations.

Since its accuracy is limited by the number of activation bits, we present three cases with 1-bit, 2-bit, and 4-bit activations.

The most recent work BID19 solves the problem of binarizing the last layer weights, but weights of the first layer are full-precision.

Overall, AlexNet-based low-precision networks achieve 43.6%, 49.8%, 53.0% top-1 accuracy for 1-bit, 2-bit and 4-bit activations, respectively.

Around 70% of the memory footprint is defined by the first two layers of AlexNet.

The fusion technique is difficult in such architectures due to large kernel sizes (11×11 and 5×5 for AlexNet) which can cause extra recomputations.

Thus, activations require 95.4KB, 190.0KB and 381.7KB of memory for 1-bit, 2-bit and 4-bit models, respectively.

The NIN-based network from BID19 with 2-bit activations achieves 51.4% top-1 accuracy, but its activation memory is larger than AlexNet due to late pooling layers.

BID17 whether the activations were binarized or not for the first and the last layer.

3 Weights are not binarized for the first layer.

4 Number of bits for the compressed "fire2,3/squeeze" layers and, in parentheses, for quantized only layers.

5 For comparison, accuracy in parentheses represents result for the corresponding model in TAB0 The SqueezeNet-based models in TAB3 are finetuned from the corresponding models in TAB0 for 40,000 iterations with a mini-batch size of 1024, and SGD solver with a step-policy learning rate starting from 1e-4 divided by 10 every 10,000 iterations.

The model with fusion and 8-bit quantized weights and activations, while having an accuracy similar to floating-point model, outperforms the state-of-the-art networks in terms of weight and activation memory.

The proposed four models from TAB0 further decrease activation memory by adding compression-decompression layers to "fire2,3/squeeze" modules.

This step allowed to shrink memory from 189.9KB to 165.4KB, 140.9KB, 116.4KB and 110.3KB depending on the compression configuration.

More compression is possible, if apply the proposed approach to other "squeeze" layers.

<|TLDR|>

@highlight

Feature map compression method that converts quantized activations into binary vectors followed by nonlinear dimensionality reduction layers embedded into a DNN
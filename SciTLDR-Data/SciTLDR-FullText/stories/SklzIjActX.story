High throughput and low latency inference of deep neural networks are critical for the deployment of deep learning applications.

This paper presents a general technique toward 8-bit low precision inference of convolutional neural networks, including 1) channel-wise scale factors of weights, especially for depthwise convolution, 2) Winograd convolution, and 3) topology-wise 8-bit support.

We experiment the techniques on top of a widely-used deep learning framework.

The 8-bit optimized model is automatically generated with a calibration process from FP32 model without the need of fine-tuning or retraining.

We perform a systematical and comprehensive study on 18 widely-used convolutional neural networks and demonstrate the effectiveness of 8-bit low precision inference across a wide range of applications and use cases, including image classification, object detection, image segmentation, and super resolution.

We show that the inference throughput and latency are improved by 1.6X and 1.5X respectively with minimal within 0.6%1to no loss in accuracy from FP32 baseline.

We believe the methodology can provide the guidance and reference design of 8-bit low precision inference for other frameworks.

All the code and models will be publicly available soon.

While convolutional neural networks (CNN) shows state-of-the-art (SOTA) accuracy for wide range of computation vision tasks, it still faces challenges during industrial deployment due to its high computational complexity of inference.

Low precision is one of the key techniques being actively studied recently to conquer the problem BID29 BID8 ; BID20 ; BID18 ; BID17 .

With hardware acceleration support, low precision inference can compute more operations per second, reduce the memory access pressure and better utilize the cache, and deliver higher throughput and lower latency.

Convolution is the primary operation in CNN models and it is a common practice to enable 8-bit low precision (INT8) inference for convolution in deep learning frameworks (e.g., TensorFlow, MXNet, and TensorRT).

To make it work, convolution utilizes INT8 computation, which requires two scale factors for activation and weight, respectively.

It is workable for standard convolution with single group and two groups BID13 .

However, it does not work well for convolution with large groups, especially for depthwise convolution BID0 .

In addition to direct convolution, it is worthwhile to explore INT8 Winograd convolution BID14 for better performance, which is absent in previous research 2 .

Although recent work have demonstrated INT8 inference with minimal accuracy loss across various models BID29 BID4 ; ; BID11 , INT8 inference is limited due to more complex topology primarily introduced by sum operation in residual block and concatenation operation in inception block BID0 .

Existing solutions need to convert the convolution output from INT8 to FP32, and apply the sum or concatenation operation on FP32.

The sacrifice of memory bandwidth and frequent data conversion lead to considerable performance overhead and therefore limit the real deployment.

Moreover, there is no systematical study of INT8 inference on various use cases, including image classification BID13 ; BID25 ; BID0 ; ), object detection Ren et al. (2015 ; BID1 ; BID15 , image segmentation BID16 ; BID15 , etc.

In this paper, we present a general technique towards efficient INT8 inference of CNN models.

We experiment the technique on top of a widely-used deep learning framework.

To the best of our knowledge, our work is the first attempt to address the above problems.

We summarize our contributions below:1.

We provide a systematical approach to channel-wise quantization of convolution, which is essential to keep the accuracy for depthwise convolution.

Top1 accuracy of INT8 inference on MobileNet-V1 and MobileNet-V2 is improved by 1.98% and 70.6%, respectively.

2.

We explore the approach of INT8 Winograd convolution and present the calibration details that cannot be trivially derived from direct convolution.

Our experiment on VGG-16 shows Top1 and Top5 accuracy loss with INT8 Winograd convolution is minimal within 0.30% and 0.25% from FP32 baseline, reducing from 5.31% and 3.38%, respectively.

3.

We add the support of sum in residual block, concatenation in inception block, and convolution for classification.

We also fuse the memory-bound operation convolution with a rectified linear unit (ReLU) BID19 and fold the parameters of batch normalization BID10 into convolution kernels.

With topology-wise INT8 support, inference speed is greatly improved by data conversion reduction and memory saving.

4.

To our knowledge, this is the first time such a systematic study is applied to and empirical result is reported on many CNN use cases and models.

We develop a calibration tool that automatically generates optimized INT8 model from FP32 model without the need of fine-tuning or retraining for easy and repeatable deployment.

We perform a comprehensive study on 18 widely-used CNN models and demonstrate the effectiveness of INT8 inference across a wide range of applications, including image classification, object detection, image segmentation, and super resolution.

The inference throughput and latency are improved by 1.6X and 1.5X respectively, while the accuracy loss is minimal within 0.6% to no loss from FP32 baseline.

We believe our methodology is general for CNN models and can provide the guide and reference on other frameworks.

All the code and models will be publicly available soon.

The rest of the paper is organized as follows, Section 2 discusses related work on low-precision inference in deep learning.

Section 3 describes INT8 inference quantization approach and recipe for CNN models.

Section 4 includes experimental results, comprehensive study, and related discussion.

Finally, Section 5 concludes the summary with results and future directions.

Computer vision tasks win considerable attentions in deep learning field in recent years.

Although CNN models provide SOTA accuracy for various computer vision tasks, it still faces challenges during industrial deployment due to its high computational complexity of inference.

NVidia have demonstrated minimal accuracy loss of INT8 inference on several CNN models for image classification (e.g., GoogleNet, AlexNet In additional to existing inference tools and frameworks from industry, many researchers have experimented low-precision inference with customized low-bit for activation and weights in deep learning tasks.

INT8 activations and weights have been proposed in BID29 , while biases and first layer input are kept with FP32 for the task of speech recognition on CPUs.

CNN approximation has been presented BID4 to perform automatic network quantization and scoring, using different bit-widths for number representation, to find a good balance between compression rate and network accuracy.

Baidu researchers 3 have successfully used 8-bits of fixed precision with 1 sign bit, 4-bits for the integer part and 3-bits for the fractional part.

Various quantization techniques have been discussed in BID26 , showing minimal to no loss at reduced precision while keeping FP32 for the first and last layers.

Deep compression with pruning, quantization, and Huffman coding has been worked out to reduce the storage requirement of neural networks significantly without affecting the accuracy, thus making easy for deployment on edge device BID5 .

Moreover, we focus on the efficient inference on commodity servers while others might require special hardware support like FPGA.

Of course, some of our insights like calibrating INT8 Winograd can complement others' work as well.

In this section, we first formulate quantization and de-quantization mathematically and then present the general recipe of INT8 inference.

We define a quantization function Q : DISPLAYFORM0 to turn an n-dimensional rational tensor r into an n-dimensional integer tensor z with the scale factor q and bit-precision p.

Here n could be of arbitrary dimensionality.

The function Round is a rounding function approximating a rational tensor with an integer tensor.

DISPLAYFORM1 We also define a de-quantization function D : Z n × R → R n that approximates the rational tensor r with its quantized form z in Equation 2.

DISPLAYFORM2 We then define + and × arithmetics on (z, q) in Equation 3.

Here we assume + and × have already been defined for tensor r and z, e.g., when they are matrices.

DISPLAYFORM3 In practice, we perform sampling for each activation, weight and bias tensor on the given dataset to get a maximum absolute value max from each tensor and set the scale factor of the tensor as DISPLAYFORM4 max where p is the precision of quantization.

p = 8 is used for all non-negative activation tensors which are mostly true for popular CNN models after batch normalization operations are folded with convolution and ReLU with zero negative slope is fused into convolution BID32 .

For potentially negative input tensors such as the one for first convolution, the operation falls back to FP32 since the hardware-accelerated INT8 convolution only supports non-negative activations as input (more details refer to BID22 ).

p = 7 is used for weight tensors.

Then most activations and weights can be stored with INT8.

We employ round-half-to-even as the Round function for best statistical accuracy.

We present the general INT8 recipe for CNN models, including depthwise convolution, Winograd convolution, and topology-wise more INT8 support.

As a common practice, INT8 convolution uses a single scale factor for each tensor, i.e. one for activation and one for weight respectively.

It is workable for standard convolution with single group (e.g., VGG-16, GoogleNet-V1, and ResNet-50) and two groups (e.g., AlexNet).

However, it does not perform well for convolution with large groups, especially for depthwise convolution (e.g., MobileNet-V1 Howard et al. FORMULA1 , MobileNet-V2 Sandler et al. (2018) ).

Different than standard convolution, depthwise convolution applies a single filter per each input channel.

As a result, a single tensor-wise scale factor for weight is not capable to represent the dynamic data range of each channel effectively.

FIG0 indicates the distribution of the first 10 filters per output channel for standard convolution (a) and depthwise convolution (b).

As the partial filter distribution is representative, we omit the demonstration of entire weight tensor distribution.

Based on the above findings, we propose channel-wise scale factors for weight tensor, similar to BID12 .

Each scale factor represents the dynamic data range per each filter.

The resulting scale factors are q activation × q weighti , where q activation is the scale factor of activation and q weighti is the scale factor of the i th filter.

With channel-wise scaling factors, Top1 accuracy of INT8 inference on MobileNet-V1 and MobileNet-V2 is improved by 1.98% and 70.6%, respectively.

Winograd is a fast algorithm for convolution and it has been widely-used in FP32 training and inference BID14 .

However, the study of INT8 Winograd convolution is not publicly available.

Considering the attractive performance gains, it is worthwhile to explore INT8 Winograd convolution.

We select standard algorithm F(2, 3) for discussion, which can leverage INT8 computation benefit from integer-based input transformation matrix.

To make INT8 Winograd convolution work, the key component is to take the scale factor for activation and weight after transformation.

DISPLAYFORM0 Equation FORMULA5 shows the formula to compute the scale factor after transformation, where B and B T are transformation matrices defined in BID14 .

Before and after transformation, we have the activation tensor for x b and x a , the scale factor for q x b (for direction convolution by default) and q xa , the maximum absolute value for max x b and max xa , respectively.

Similarly, we can compute the scale factor of weight before and after transformation.

The scale factor of activation and weight after transformation is set for INT8 Winograd convolution finally.

We experiment the idea on VGG-16, a classical model for Winograd convolution.

With the scale factor q xa , Top1 and Top5 accuracy loss is minimal within 0.30% and 0.25% from FP32 baseline, while with the scale factor q x b , the accuracy loss is significant with 5.31% and 3.38%, respectively.

Note that our approach is general and can be applied to other algorithms besides standard algorithm F(2, 3).

We extend INT8 computation to other computation types besides convolution and also apply constant folding and computation fusion to consecutive computations so that almost all input and output activation tensors use INT8 while accumulators use INT32 or FP32 for best accuracy.

In this section, we discuss these topology-wise INT8 opportunities.

We also discuss topology patterns in which output tensors should be kept in FP32 for good accuracy.

Pooling.

Both max pooling and average pooling are computed directly with INT8.

The scale factors of the input and output tensors are same.

We use INT32 accumulator for average pooling to avoid arithmetic overflow.

BID0 .

FIG1 demonstrates the inception block that concatenates convolution output per filter.

Our study shows that the dynamic ranges of the input tensors are quite close.

So we set the scale factor of INT8 output tensor to the smallest scale factor of INT8 input tensors.

Batch Normalization Folding.

Computing INT8 batch normalization without losing accuracy is challenging.

Fortunately, in most recent CNN models, batch normalization is usually added after convolution.

Since the computation is essentially an affine transformation during inference, it can be folded into the convolution kernel as in Equation 5.

Both the new convolution weight w and bias b are affine transformation of the original weight w and bias b. As defined in BID10 , µ and σ 2 are the learned mini-batch mean and variance respectively, and γ and β are the scale and shift terms.

Fusing Convolution and Element-wise Post-Operations.

For the best arithmetic accuracy and efficient execution, convolution output elements are first accumulated in FP32 and then fused with the element-wise post-operations immediately after it before being quantized back to INT8.

The post-operations and quantization can be efficiently computed in registers.

Examples of these postoperations are ReLU , Sum, Sum ReLU and Sum BatchN orm ReLU .

The latter three are common patterns of residual networks .

Figure 3 illustrates a residual block from ResNet-50 and the sum operation (a) is fused into res2a branch2c (b).

Then, res2a branch2c accepts two inputs res2a branch1 and res2a branch2b, and perform the sum operation.

With the general recipe of INT8 inference, we experiment the techniques and develop the calibration tool on top of a widely-used deep learning framework .

We next discuss the experimental configurations and perform a systematical study on 18 classical CNN models.

We demonstrate the effectiveness of INT8 inference across a wide range of applications and use cases.

We develop the calibration tool that automatically generates the optimized INT8 model from FP32 model without the need of fine-tuning or retraining.

The calibration process has two inputs, CNN model with pre-trained FP32 weights and calibration dataset.

Besides, the tool provides the additional items to facilitate the calibration process:Iteration number.

It allows user to define the iteration number for sampling on activation.

Scale factor mode.

It allows user to define scale factor mode single or multiple (channel-wise).Calibration strategy.

It allows users to define the calibration algorithm (Direct or KL) to compute the scale factor by DISPLAYFORM0 max , where p is the quantization precision.

Direct selects the maximum absolute value of the tensor as max directly, while KL computes max in terms of the entropy loss of quantization following the work in TensorRT.Accuracy tuning.

It allows users to define the accuracy loss tolerance on INT8 model.

Calibration process makes some operations fall back to FP32 to meet the accuracy goal.

We select totally 18 CNN models in our experiments in TAB1 .

Basically, we have three rules for model selection: 1) it is classical and representative; 2) it comes from various use cases; and 3) it is publicly available with pre-trained weight or is easy to train with existing hyper-parameters.

DISPLAYFORM0 Topology column shows the selected CNN model.

On ResNet-50, we use two versions, default one from and variant one from FaceBook (with FB) BID3 .

Use case column shows the model category, IC (image classification), OD (object detection), IS (image segmentation), and SR (super resolution).

Weight column shows whether the pre-trained weight is publicly available.

With respect to calibration dataset, we use ImageNet-1k BID23 for image classification, PASCAL VOC Everingham et al. (2015) for object detection and image segmentation, and internal gaming images for super resolution.

We perform calibration on training dataset with sampling iteration from 1, 2, 5, 10, 20, to 30, scale factor mode single or multiple, and different algorithm Direct or KL.

The total calibration cost is affordable since it takes seconds to minutes to complete each calibration.

We measure the accuracy on validation dataset independently from calibration dataset.

TAB2 shows the best accuracy of CNN models under INT8 inference.

Note that we use standard metrics to measure the accuracy, Top1 and Top5 for image classification, mAP (mean Average Precision) for object detection, mean accuracy and IoU (Intersection of Union) for image segmentation, and SSIM (Structural SIMilarity) and PSNR (Peak Signal-to-Noise Ratio) for super resolution.

Our experiments demonstrate the effectiveness across a wide range of use cases, keeping the accuracy loss from FP32 baseline, within 0.6% for Top1 and 0.3% for Top5 on image classification, 0.5% for mAP on object detection, 0.2% for mean IoU on image segmentation, and 0.1% for PSNR on super resolution.

Moreover, INT8 inference recipe also works well for models ResNet-50/101/152 with sparsity removal BID22 .On the other hand, we evaluate the errors of 50k images from ImageNet validation set for FP32 and INT8 inference and find that there is no obvious bias at image class based on empirical analysis on incorrectly-predicted images.

With further analysis on typical images, we figure out that it is more difficult for INT8 model to distinguish the objects with small differences.

As an example, INT8 model can recognize the dog (ILSVRC2012 val 00046423) correctly, but fails to figure out the accurate breed.

Moreover, we find that the information loss from FP32 to INT8 model may lead to potential misclassification (e.g., ILSVRC2012 val 00031193).

We also compute the entropy of Softmax output for both FP32 and INT8 model.

The results show the probability is average for INT8 model, which indicates the entropy increases and Top1 classification capability decreases.

On performance side, we measure the performance of INT8 inference and speedup over FP32 using dummy data, as shown in TAB2 .

We can see that the throughput and latency are improved by 1.6X and 1.5X in average and 2.0X and 2.1X as maximum, respectively.

Please note that the convolution improvement on INT8 over FP32 is 1.3X based on HW instructions support BID22 and therefore the latency improvement might be smaller for those non-computationintensive topologies (e.g., MobileNetV2).

To align the model with best accuracy, the above performance in TAB2 does not include INT8 Winograd convolution.

We expect to deliver similar performance improvement of Winograd on INT8 as FP32 BID14 during our development.

Different from previous work BID29 BID26 , we also experiment the first convolution using INT8 than FP32, which shows reasonable accuracy within 1% loss.

Our experimental results also demonstrate the impact of calibration process on accuracy with different sampling iteration, different calibration algorithm, or different scale factor mode.

We summarize our findings: (1) Channel-wise scaling factors can always deliver better accuracy than single scale factor, especially for depthwise convolution; (2) Direct algorithm is more effective in most cases than KL, while KL algorithm can deliver better accuracy than FP32 baseline in some cases; and (3) More sampling iterations show more stable dynamic data rage and therefore better accuracy.

How to select the optimal calibration strategy is an interesting topic as one of our future directions.

In this paper, we propose the general recipe of INT8 inference and experiment the techniques on a widely-used deep learning framework.

We develop an automatic calibration tool for optimized INT8 model generation and demonstrate the effectiveness on 18 CNN models across a wide range of use cases.

The inference throughput and latency are improved by 1.6X and 1.5X respectively, while the accuracy loss is minimal within 0.6% to no loss from FP32 baseline.

We believe our methodology is general for CNN models and can provide the guide and reference on other frameworks.

@highlight

We present a general technique toward 8-bit low precision inference of convolutional neural networks. 

@highlight

This paper designs a system to automatically quantize the CNN pretrained models
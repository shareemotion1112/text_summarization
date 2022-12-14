A successful application of convolutional architectures is to increase the resolution of single low-resolution images -- a image restoration task called super-resolution (SR).

Naturally, SR is of value to resource constrained devices like mobile phones, electronic photograph frames and televisions to enhance image quality.

However, SR demands perhaps the most extreme amounts of memory and compute operations of any mainstream vision task known today, preventing SR from being deployed to devices that require them.

In this paper, we perform a early systematic study of system resource efficiency for SR, within the context of a variety of architectural and low-precision approaches originally developed for discriminative neural networks.

We present a rich set of insights, representative SR architectures, and efficiency trade-offs; for example, the prioritization of ways to compress models to reach a specific memory and computation target and techniques to compact SR models so that they are suitable for DSPs and FPGAs.

As a result of doing so, we manage to achieve better and comparable performance with previous models in the existing literature, highlighting the practicality of using existing efficiency techniques in SR tasks.

Collectively, we believe these results provides the foundation for further research into the little explored area of resource efficiency for SR.

Rapid progress has been made in the development of convolutional networks BID10 that are capable of taking a low-resolution image and producing an image with a significant increase in resolution.

This image restoration task is referred to as super-resolution (SR) and has many potential applications in devices with limited memory and compute capacity.

The fundamental problem however is that the state-of-the-art networks consist of thousands of layers and are some of the most resource intensive networks currently known.

Furthermore, due to the spatial dimensions of feature maps needed to maintain or up-scale the input, the number of operations are counted in the billions as opposed to millions in models for discriminative tasks.

As a result, there is a need for a general systematic approach to improve the efficiency of SR models.

The challenge of the system resource requirements for deep learning models for tasks other than SR have been carefully studied in previous works BID62 BID40 BID48 , achieving massive gains in size and compute with little to no loss in performance.

These reductions are achieved with a wide variety of methods being developed grounded in primarily architecture-level changes and techniques grounded in the use of low precision and quantized model parameters.

However, how these efficiency methods behave when applied within SR have not yet been studied in significant depth, with very few results published in the literature.

Extrapolating from prior results for other tasks is problematic given that predominantly existing studies are applied to discriminative tasks with substantially different architectures and operations.

Due to the up-sampling structure of SR models, these efficiency methods may therefore produce potentially stronger side-effects to image distortion.

In this paper, we detail a systematic study that seeks to bridge current understanding in SR and known approaches for scaling down the consumption of system resources by deep models.

By examining the impact on image distortion quality when performing various efficiency techniques, we provide the following new insights:??? The effectiveness of low rank tensor decomposition and other convolution approximations, which are comparable and successful in discriminative tasks, can vary considerably in SR.(See section 4.1).??? Unlike image discriminative networks, SR networks suffer from a worse trade-off between efficiency and performance as more layers are compressed. (See section 4.2)??? The practicality of adopting compression techniques for other tasks to SR as our best models are better or comparable to existing literature.

For instance, our best model achieves significantly better performance and 6x less compute than MemNet BID51 and VDSR BID27 .

Additionally, it also performs better and is 4.1x-5.8x smaller than SRMDNF BID61 . (See section 4.3)??? Successful quantization techniques used in image discriminative tasks are equally successful in SR. (See section 5)

We focus on using neural networks for SR as they have shown to achieve superior performance against previous traditional approaches ( BID53 BID29 BID6 ).

An SR image can either be evaluated using standard image distortion metrics, such as PSNR, SSIM BID57 and IFC BID49 , or using perception metrics, such as BID39 , NIQE BID45 , and BRISQUE BID44 .

BID5 provided theoretical backups on the trade-off between image distortion and perception.

Distortion SR: In the distortion line of work, models favour pixel-to-pixel comparisons and are usually trained on either the L1 or L2 (MSE) loss.

These models have been known to produce more visually pleasing outcomes on structural images BID4 than perceptual SR models.

BID10 first proposed using convolutional networks for SR, leading to a surge in using neural networks for SR.

These networks differ in their building blocks for feature extraction and up-sampling.

For instance, BID11 proposed a faster convolutional network by taking the down-sampled low-resolution image as an input.

Other variations include using more layers BID27 , recursive layers BID26 BID50 , memory blocks BID51 BID0 , DenseNet BID20 blocks BID54 , residual blocks BID34 , and multipleimage degradations BID61 .

Additionally, more recent models use attention BID1 mechanisms , back-projection BID17 , and other non-conventional non-linear layers .Perceptual SR: Perceptual SR models, on the other hand, are better at reconstructing unstructured details with high perceptual quality BID4 .

These models usually adopt popular models for image distortion and train them using a variety of different loss functions, such as the perceptual loss BID25 , contextual loss BID43 , adversarial loss BID15 , and the Gram loss BID14 .

For instance, adopted EUSR and BID34 ; BID42 adopted SRResNet BID34 by making slight architecture changes and replacing the objective.

Although these perceptual models are able to generate more visually pleasing results on certain images, they do not seem to work well as inputs for image classification BID24 .Efficient SR: As models in both tracks are resource-intensive, the recent PIRM 2018 Challenge for mobile BID22 ) presented a range of high efficiency models that were designed to run faster and perform better than SRCNN BID10 .

These models are complementary to our work and can follow our best practices to achieve greater efficiency gains.

A work closely related to our work is done by BID0 who systemically investigate the impact of using grouped convolutions.

Due to the massive design space caused by the variability of training and evaluating these models, we focus on the trade-offs between performance measured by the image distortion metrics and efficiency and leave the rest as future work.

The key step in our work is to build understanding towards building resource-efficient architectures for super-resolution.

While there is a lot of understanding of how these efficiency-saving techniques work in classification problems, there is a lack of experimental studies and systematic approaches to understand their practicality in super-resolution.

To our knowledge, this is the first systematic study of wide range efficiency methods on super-resolution.

We measure performances using PSNR and SSIM BID57 and measure efficiency of memory and compute using the number of parameters and the number of multiply-add operations (Mult-Adds), both of which dictate which platform these models can run on.

However, these metrics alone do not reflect the trade-off between performance and efficiency.

Therefore, we introduce two new metrics that measures the number of Giga Mult-Adds saved and the number of parameters saved for every 0.01dB PSNR loss in the test sets: Set5 BID2 , Set14 BID59 , B100 BID41 , and Urban100 BID21 .

These metrics are calculated by taking the difference between the compressed model and the uncompressed model.

All Mult-Adds are calculated by upscaling to a 720p image.

We decide to use RCAN Zhang et al. (2018) as our baseline model as it proves to be the state-ofthe-art and has the best performance in the image distortion metrics at the time of writing.

We take its simplest building block and build a shallower network and use that as a basis for exploring the use of a variety of techniques.

Implementation Details: We train our models in section 4 and section 5.1 in the same manner as that of EDSR Lim et al. (2017) .

In particular, we use 48??48 RGB patches of LR images from the DIV2K dataset BID52 .

We augment the training data with random horizontal flips and 90 degree rotations and pre-process them by subtracting the mean RGB value of the DIV2K dataset.

Our model is trained using the ADAM Kingma & Ba (2014) optimizer with hyper-parameters ?? 1 = 0.9, ?? 2 = 0.999, and = 10 ???8 .

The mini-batch size is 16, learning rate begins with 1e ??? 4 and is halved at 200 epochs, and the model is trained for 300 epochs using L1 loss.

We train x2 models from scratch and use them as pre-trained models to train x3 and x4 models for faster convergence.

Lastly, for ternary quantization in section 5.2, we further train the model with quantization enabled in each forward pass for 40 epochs, starting at a learning rate of 5e ??? 5, and then fix the quantized ternary weights and further train for another 10 epochs at a learning rate of 2.5e ??? 5.

We begin our evaluation by conducting a series of experiments: (i) we explore the effects of applying different resource-efficient architectures to our baseline model (section 4.1), (ii) we consider two best techniques and present trade-off solutions while applying them to different parts of our baseline model (section 4.2), (iii) and lastly, we compare our best results with previous SR architectures (section 4.3).

Motivation: Resource-efficient architectures use various low rank tensor decomposition and other convolutional approximation techniques, which is agnostic and is not specifically designed for any particular task, to build fast and accurate image discriminative models.

We first develop an initial understanding of the trade-off solution by replacing and modifying 3x3 convolution layer blocks in the baseline model.

We explore the use of known techniques such as the bottleneck design, separable/grouped convolutions, and channel shuffling.

We take the feature extraction unit from resourceefficient architectures and remove all batch normalisation layers as they were previously shown to reduce performance and increase GPU memory usage .

For our first set of experiments, we replace all 3x3 convolution layers in the residual groups of our baseline model.

bl: Our baseline model from RCAN .

We reduce the number of residual groups (RG) from 10 to 2 and the number of residual channel attention block (RCAB) in each RG from 20 to 5.

We use a feature map size of 64.

Making the network shallower and small in parameters allow us to clearly understand each architectural changes as opposed to having a deep network which may cause other effects and interplay.

blrn(r): We adopt the residual bottleneck design from ResNet with a reduction factor of r. Specifically, a 1x1 convolution is used to compress information among channels by a reduction factor, resulting in a cheaper 3x3 convolution.

Another 1x1 convolution is then used to recover the dimension of the output channel and a skip connection is used to pass on information that may have been lost.

blrxn(r,g): We replace the 3x3 convolution in blrn to a 3x3 grouped convolution, forming a block that is similar to that of ResNeXt BID58 with an additional group size of g. Computation cost is further reduced by the use of grouped convolutions BID32 .blm1: In order to further improve efficiency of the 3x3 grouped convolution, we can maximise the group size, forming a convolution that is known as depthwise convolution.

Following this idea, we adopt the MobileNet v1 unit which uses depthwise separable convolutions, each consist of a 3x3 depthwise convolution followed by a 1x1 convolution, also known as a pointwise convolution.

We can further approximate the 3x3 depthwise convolution by using a 1x3 and a 3x1 depthwise convolution, a technique that is used in EffNet BID13 .

We adopt the unit from EffNet by removing the pooling layers.bls1(r, g): We group both 3x3 and 1x1 convolutions and added channel shuffling in order to improve the information flow among channels.

In order to test the effects of channel shuffling, we adopt the ShuffleNet v1 BID62 unit.blclc(g1, g2): Channel shuffle is also used in Clcnet BID60 to further improve efficiency of blm1.

In order to maximise efficiency from our adoption of ClcNet units, we follow the group size guidelines recommended by the authors for both the group sizes of the 3x3 (g1) and 1x1 (g2) grouped convolution.

bls2: Apart from using grouped convolutions, BID40 proposed splitting the flow into two, which is termed as channel splitting, and performing convolution on only half of the input channels in each unit at each pass.

Channel shuffle is then used to enable information flow between both branches.blm2(e): Inverted residuals can be used to enable skip connections directly on the bottleneck layers.

Therefore, we adopt the MobileNet v2 BID48 ) unit in our experiments Results: Our results in TAB0 show that techniques that result in a better trade-off between memory and performance will have a better trade-off between compute and performance.

1 .

Overall, the use of bottlenecks alone (blrn) result in the best trade-offs followed by the use of separable/grouped convolutions.

Reducing the number of features to accommodate inverted bottlenecks (blm2) severely impact the performance and thus we omit the results from the table.

We speculate that doing so would result in insufficient features at the up-sampling layer to fully capture the up-sampled image representation.

Thus, we use the same number of feature maps as our bottleneck.

Although the use of inverted 1 Results for 3x and 4x upscaling show similar performance and efficiency trade-offs residuals in our experiments seem worse off, it may perform better on models that use a larger feature size or multiple smaller up-sampling layers.

Lastly, the use of 1x1 grouped convolution or channel splitting with channel shuffling further reduces the evaluation metric.

Although doing so can drastically reduce size, the trade-off does not seem to justify its advantages.

Therefore, we recommend using bottlenecks for building resourceefficient SR architectures.

If the budget for memory and efficiency is tight, we recommend the use of depthwise separable convolutions instead.

In image discriminative tasks, the proposed architecture changes are comparable in terms of efficiency and accuracy trade-offs.

In our work, we show that the sole use of low rank tensor decomposition (bottleneck architectures) provide the best trade-offs, followed by the use of separable/grouped convolutions and the use of both channel splitting and shuffling.

Motivation: BID3 and BID30 have shown that it is possible in image classification to maintain a similar or slight drop in performance by decomposing tensors of known models.

However, our models suffer a significant drop in performance.

TAB0 ).

Therefore, in order to further understand the extent of their applicability in SR, we apply the top two best techniques, which are bottleneck reduction (blrn) and depthwise separable convolutions (blm1), on various different parts of our baseline model.

Our preliminary experiments with applying some of these techniques on the first and last convolution layer led to worse trade-offs.

Therefore, we apply our techniques between them.

We replace the sub-pixel convolution upsampling layer to the enhanced upscaling module (EUM) as proposed by to allow the use of skip connections.

Using EUM leads to an increase in performance at a slight cost of both memory and compute.

Thus, in order to maintain the memory cost, we use recursion, forming the enhanced recursive upscaling module (ERUM) shown in figure 1.

The number of ERUMs is the same as the scaling factor and each ERUM recurses twice or thrice for x2, x4 or x3 scales respectively.

Experiments that use ERUMs for up-sampling are indicated with a postfix -e.

We calculate our trade-off metrics based on our baseline model with ERUM as its up-sampling layer instead bl-e.

We modify all 3x3 convolution layers as such: TAB1 reinforce our findings in section 4.1 that the adoption of bottleneck reduction alone leads to the best trade-offs, followed by the use of group convolutions.

Therefore, we recommend taking gradual steps to compress the model.

For instance, we suggest gradually changing convolutions to use bottleneck reduction, avoiding the up-sampling, first, and last convolutions until a budget is reached.

If further compression is needed, we suggest changes to the up-sampling layer or the use of group convolutions.

We take our derived best models based on different budgets from our first two experiments (See section 4.1 & 4.2) and compare them with the existing literature, which is shown in TAB2 .

For fair comparisons, we omit models that are way bigger by several magnitudes as their performances are much better.

Likewise, we exclude models that are way smaller as their performance are much worse.

Regardless, our techniques can be applied to any model for further trade-offs between performance and efficiency.

Although our main objective is not to beat previous models but to understand and recommend techniques that can be applied to any existing model, we manage to derive models that are better or comparable to other models in the literature.

For instance, in terms of size and evaluation metric, our best model (blrn-e[rb] ) outperforms all models that have a count of 1,500K parameters and below.

By comparing compute and evaluation, our best model performs better and has roughly x6 less operations than MemNet BID51 .

It is also comparable with the CARN model in the number of operations, trading a slightly worse performance with a 2.5x size reduction.

Overall, our best model is better than earlier models such as VDSR BID27 and later models such as SRMDNF BID61 ) for 3x and 4x scales.

Our second and third best models also outperform earlier models in performance with huge savings in the number of operations for 3x and 4x scales.

Our results show that these techniques which are designed for image discriminative tasks can be effective in SR.

Visual comparisons for some of these models can be found in the appendix.

In our next set of experiments, we examine the viability of quantization and the use of extreme low-precision (ternary/binary) as mechanisms to reduce system resource for SR.

Motivation:

With the success of low precision on neural networks on classification problems, we aim to show initial understanding of applying 8-bits integer quantization on our baseline model as described in section 4.1.

Moving from 32-bits to 8-bits will result in a 4x reduction in memory and allow support for low-power embedded devices.

Approach: We train the model in full precision and apply the quantization scheme in TensorflowLite for integer-only arithmetic BID23 and retrain for an additional 5 epochs with the a learning rate of 5e ??? 5.Results: Our results show that applying quantization lead to a slight evaluation loss in 2x scaling and a slight improvement in 4x scaling.

Our results are similar to that of classification BID23 .

Furthermore the results show that deep neural networks are robust to noise and perturbations caused by quantization.

Therefore, we strongly recommend quantization especially on hardware that can further utilise its benefits.

Motivation:

The success of using binarized BID9 BID47 and ternarized neural networks BID35 BID55 to approximate the full-precision convolutions in image discriminative tasks motivates us to experiment the effectiveness of these techniques in SR.Approach: We adapt the baseline SR architecture used in prior experiments in section 4.1 but modify it structurally by replacing every convolution layer with sum-product convolution layers proposed in StrassenNets BID55 .

These sum-product convolution layers represent a sum-product network (SPN) that is used to approximate a matrix multiplication.

Specifically, each convolution layer is replaced with a convolution layer that outputs r feature maps, followed by a element-wise multiplication and a transpose convolution layer.

As both the convolution layers hold ternary weights, the number of multiply operations required is determined by the number of element-wise multiplication which is controlled by r. Besides outlining the trade-off of tuning r, we aggressively use group convolutions.

Results: Similar to section 5.1, the results in TAB4 are similar to that of image discriminative tasks.

Specifically, the higher the width of the hidden layer of the SPN, r, the better the performance at a cost of additional multiplications and additions.

When r = 6c out, we achieve an evaluation score that is close to the uncompressed model for 2x scales and suffer a slight drop for 3x and 4x scales.

Any further attempts to increase r do not improve evaluation metric.

As proposed by BID55 , we use group convolutions to reduce the number of additions.

We take a step further and experiment with a wide range of groups as well.

We found that the reduced number of additions do not justify the evaluation drop; the use of a lower r is better than the use of groups.

Additionally, since multipliers are more costly and take up more area on chip than adders, we suggest lowering r instead of using grouped convolutions.

Through an extensive set of experiments, we show that some of the previous efficiency techniques that are successful in image discriminative tasks can be successfully applied to SR.

Although these techniques are comparable in the former tasks, we highlight their varying effectiveness in SR and derive a list of best practices to construct or reduce any model that are designed to reduce image distortion:??? The sole use of low rank tensor decomposition (bottleneck design) results in the best tradeoffs between performance and efficiency.

If further compression of memory and/or compute is needed, separable/grouped convolution is recommended.

If efficiency on conventional hardware is the topmost priority, we recommend reducing the number of layers or adopting the use of both channel splitting and shuffling BID40 .???

The fewer resource-efficient architecture changes applied, the better the trade-off.

Therefore, we recommend a mixture of convolution and resource-efficient units unless further compression is needed.??? Avoid architecture changes on the first and last convolution layers.??? We strongly recommend using any form of quantization if the hardware supports it.

@highlight

We build an understanding of resource-efficient techniques on Super-Resolution

@highlight

The paper proposes a detailed empirical evaluation of the trade-offs achieved by various convolutional neural networks on the super resolution problem.

@highlight

This paper proposed to improve the system resource efficiency for super resolution networks.
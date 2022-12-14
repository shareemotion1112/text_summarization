We present a simple and general method to train a single neural network executable at different widths (number of channels in a layer), permitting instant and adaptive accuracy-efficiency trade-offs at runtime.

Instead of training individual networks with different width configurations, we train a shared network with switchable batch normalization.

At runtime, the network can adjust its width on the fly according to on-device benchmarks and resource constraints, rather than downloading and offloading different models.

Our trained networks, named slimmable neural networks, achieve similar (and in many cases better) ImageNet classification accuracy than individually trained models of MobileNet v1, MobileNet v2, ShuffleNet and ResNet-50 at different widths respectively.

We also demonstrate better performance of slimmable models compared with individual ones across a wide range of applications including COCO bounding-box object detection, instance segmentation and person keypoint detection without tuning hyper-parameters.

Lastly we visualize and discuss the learned features of slimmable networks.

Code and models are available at: https://github.com/JiahuiYu/slimmable_networks

Recently deep neural networks are prevailing in applications on mobile phones, augmented reality devices and autonomous cars.

Many of these applications require a short response time.

Towards this goal, manually designed lightweight networks BID51 BID50 are proposed with low computational complexities and small memory footprints.

Automated neural architecture search methods ) also integrate on-device latency into search objectives by running models on a specific phone.

However, at runtime these networks are not re-configurable to adapt across different devices given a same response time budget.

For example, there were over 24,000 unique Android devices in 2015 2 .

These devices have drastically different runtimes for the same neural network BID19 , as shown in TAB0 .

In practice, given the same response time constraint, high-end phones can achieve higher accuracy by running larger models, while low-end phones have to sacrifice accuracy to reduce latency.

Although a global hyper-parameter, width multiplier, is provided in lightweight networks BID51 BID50 to trade off between latency and accuracy, it is inflexible and has many constraints.

First, models with different width multipliers need to be trained, benchmarked and deployed individually.

A big offline table needs to be maintained to document the allocation of different models to different devices, according to time and energy budget.

Second, even on a same device, the computational budget varies (for example, excessive consumption of background apps reduces the available computing capacity), and the energy budget varies (for example, a mobile phone may be in low-power or power-saving mode).

Third, when switching to a larger or smaller model, the cost of time and data for downloading and offloading models is not negligible.

Recently dynamic neural networks are introduced to allow selective inference paths.

BID29 introduce controller modules whose outputs control whether to execute other modules.

It has low theoretical computational complexity but is nontrivial to optimize and deploy on mobiles since dynamic conditions prohibit layer fusing and memory optimization.

adapt early-exits into networks and connect them with dense connectivity.

and propose to selectively choose the blocks in a deep residual network to execute during inference.

Nevertheless, in contrast to width (number of channels), reducing depth cannot reduce memory footprint in inference, which is commonly constrained on mobiles.

The question remains: Given budgets of resources, how to instantly, adaptively and efficiently trade off between accuracy and latency for neural networks at runtime?

In this work we introduce slimmable neural networks, a new class of networks executable at different widths, as a general solution to trade off between accuracy and latency on the fly.

FIG0 shows an example of a slimmable network that can switch between four model variants with different numbers of active channels.

The parameters of all model variants are shared and the active channels in different layers can be adjusted.

For brevity, we denote a model variant in a slimmable network as a switch, the number of active channels in a switch as its width.

0.25?? represents that the width in all layers are scaled by 0.25 of the full model.

In contrast to other solutions above, slimmable networks have several advantages: (1) For different conditions, a single model is trained, benchmarked and deployed.

(2) A near-optimal trade-off can be achieved by running the model on a target device and adjusting active channels accordingly.

(3) The solution is generally applicable to (normal, group, depthwise-separable, dilated) convolutions, fully-connected layers, pooling layers and many other building blocks of neural networks.

It is also generally applicable to different tasks including classification, detection, identification, image restoration and more.

(4) In practice, it is straightforward to deploy on mobiles with existing runtime libraries.

After switching to a new configuration, the slimmable network becomes a normal network to run without additional runtime and memory cost.

However, neural networks naturally run as a whole and usually the number of channels cannot be adjusted dynamically.

Empirically training neural networks with multiple switches has an extremely low testing accuracy around 0.1% for 1000-class ImageNet classification.

We conjecture it is mainly due to the problem that accumulating different number of channels results in different feature mean and variance.

This discrepancy of feature mean and variance across different switches leads to inaccurate statistics of shared Batch Normalization layers BID20 , an important training stabilizer.

To this end, we propose a simple and effective approach, switchable batch normalization, that privatizes batch normalization for different switches of a slimmable network.

The variables of moving averaged means and variances can independently accumulate feature statistics of each switch.

Moreover, Batch Normalization usually comes with two additional learnable scale and bias parameter to ensure same representation space BID20 .

These two parameters may able to act as conditional parameters for different switches, since the computation graph of a slimmable network depends on the width configuration.

It is noteworthy that the scale and bias can be merged into variables of moving mean and variance after training, thus by default we also use independent scale and bias as they come for free.

Importantly, batch normalization layers usually have negligible size (less than 1%) in a model.

BID10 BID53 BID22 and BID16 implanted earlyexiting prediction branches to reduce the average execution depth.

The computation graph of these methods are conditioned on network input, and lower theoretical computational complexity can be achieved.

Conditional Normalization.

Many real-world problems require conditional input.

Feature-wise transformation BID7 ) is a prevalent approach to integrate different sources of information, where conditional scales and biases are applied across the network.

It is commonly implemented in the form of conditional normalization layers, such as batch normalization or layer normalization BID2 .

Conditional normalization is widely used in tasks including style transfer BID6 BID25 BID18 BID26 , image recognition BID24 BID45 and many others BID34 a) .

To train slimmable neural networks, we begin with a naive approach, where we directly train a shared neural network with different width configurations.

The training framework is similar to the one of our final approach, as shown in Algorithm 1.

The training is stable, however, the network obtains extremely low top-1 testing accuracy around 0.1% on 1000-class ImageNet classification.

Error curves of the naive approach are shown in FIG2 .

We conjecture the major problem in the naive approach is that: for a single channel in a layer, different numbers of input channels in previous layer result in different means and variances of the aggregated feature, which are then rolling averaged to a shared batch normalization layer.

The inconsistency leads to inaccurate batch normalization statistics in a layer-by-layer propagating manner.

Note that these batch normalization statistics (moving averaged means and variances) are only used during testing, in training the means and variances of the current mini-batch are used.

We then investigate incremental training approach (a.k.a.

progressive training) BID39 .

We experiment with Mobilenet v2 on ImageNet classification task.

We first train a base model A (MobileNet v2 0.35??).

We fix it and add extra parameters B to make it an extended model A+B (MobileNet v2 0.5??).

The extra parameters are fine-tuned along with the fixed parameters of A on the training data.

Although the approach is stable in both training and testing, the top-1 accuracy only increases from 60.3% of A to 61.0% of A+B. In contrast, individually trained MobileNet v2 0.5?? achieves 65.4% accuracy on the ImageNet validation set.

The major reason for this accuracy degradation is that when expanding base model A to the next level A+B, new connections, not only from B to B, but also from B to A and from A to B, are added in the computation graph.

The incremental training prohibits joint adaptation of weights A and B, significantly deteriorating the overall performance.

Motivated by the investigations above, we present a simple and highly effective approach, named Switchable Batch Normalization (S-BN), that employs independent batch normalization BID20 for different switches in a slimmable network.

Batch normalization (BN) was originally proposed to reduce internal covariate shift by normalizing the feature: DISPLAYFORM0 + ??, where y is the input to be normalized and y is the output, ??, ?? are learnable scale and bias, ??, ?? 2 are mean and variance of current mini-batch during training.

During testing, moving averaged statistics of means and variances across all training images are used instead.

BN enables faster and stabler training of deep neural networks BID20 BID35 , also it can encode conditional information to feature representations BID34 BID24 .To train slimmable networks, S-BN privatizes all batch normalization layers for each switch in a slimmable network.

Compared with the naive training approach, it solves the problem of feature aggregation inconsistency between different switches by independently normalizing the feature mean and variance during testing.

The scale and bias in S-BN may be able to encode conditional information of width configuration of current switch (the scale and bias can be merged into variables of moving mean and variance after training, thus by default we also use independent scale and bias as they come for free).

Moreover, in contrast to incremental training, with S-BN we can jointly train all switches at different widths, therefore all weights are jointly updated to achieve a better performance.

A representative training and validation error curve with S-BN is shown in FIG2 .S-BN also has two important advantages.

First, the number of extra parameters is negligible.

TAB2 enumerates the number and percentage of parameters in batch normalization layers (after training, ??, ??, ??, ?? are merged into two parameters).

In most cases, batch normalization layers only have less than 1% of the model size.

Second, the runtime overhead is also negligible for deployment.

In practice, batch normalization layers are typically fused into convolution layers for efficient inference.

For slimmable networks, the re-fusing of batch normalization can be done on the fly at runtime since its time cost is negligible.

After switching to a new configuration, the slimmable network becomes a normal network to run without additional runtime and memory cost.

Get next mini-batch of data x and label y.

Clear gradients of weights, optimizer.zero grad().

for width in switchable width list do 7:Switch the batch normalization parameters of current width on network M .

Execute sub-network at current width,?? = M (x).

Compute loss, loss = criterion(??, y).

Compute gradients, loss.backward().

end for 12:Update weights, optimizer.step().

13: end for

In this section, we first evaluate slimmable networks on ImageNet BID5 classification.

Further we demonstrate the performance of a slimmable network with more switches.

Finally we apply slimmable networks to a number of different applications.

We experiment with the ImageNet BID5 ) classification dataset with 1000 classes.

It is comprised of around 1.28M training images and 50K validation images.

We first investigate slimmable neural networks on three state-of-the-art lightweight networks, MobileNet v1 , MobileNet v2 , ShuffleNet , and one representative large model ResNet-50 .To make a fair comparison, we follow the training settings (for example, learning rate scheduling, weight initialization, weight decay, data augmentation, input image resolution, mini-batch size, training iterations, optimizer) in corresponding papers respectively BID50 BID51 BID52 .

One exception is that for MobileNet v1 and MobileNet v2, we use stochastic gradient descent (SGD) as the optimizer instead of the RMSPropOptimizer BID50 .

For ResNet-50 , we train for 100 epochs, and decrease the learning rate by 10?? at 30, 60 and 90 epochs.

We evaluate the top-1 classication error on the center 224 ?? 224 crop of images in the validation set.

More implementation details are included in Appendix A.We first show training and validation error curves in FIG2 .

The results of naive training approach are also reported as comparisons.

Although both our approach and the naive approach are stable in training, the testing error of naive approach is extremely high.

With switchable batch normalization, the error rates of different switches are stable and the rank of error rates is also preserved consistently across all training epochs.

Next we show in TAB3 the top-1 classification error for both individual networks and slimmable networks given same width configurations.

We use S-to indicate slimmable models.

The error rates for individual models are from corresponding papers except those denoted with ??? .

The runtime FLOPs (number of Multiply-Adds) for each model are also reported as a reference.

TAB3 shows that slimmable networks achieve similar performance compared to those that are individually trained.

Intuitively compressing different networks into a shared network poses extra optimization constraints to each network, a slimmable network is expected to have lower performance than individually trained ones.

However, our experiments show that joint training of different switches indeed improves the performance in many cases, especially for slim switches (for example, MobileNet v1 0.25?? is improved by 3.3%).

We conjecture that the improvements may come from implicit model distilling BID14 BID36 where the large model transfers its knowledge to small model by weight sharing and joint training.

Our proposed approach for slimmable neural networks is generally applicable to the above representative network architectures.

It is noteworthy that we experiment with both residual and nonresidual networks (MobileNet v1).

The training of slimmable models can be applied to convolutions, depthwise-separable convolutions BID4 , group convolutions BID44 , pooling layers, fully-connectted layers, residual connections, feature concatenations and many other building blocks of deep neural networks.

The more switches available in a slimmable network, the more choices one have for trade-offs between accuracy and latency.

We thus investigate how the number of switches potentially impact accuracy.

In Table 4 , we train a 8-switch slimmable MobileNet v1 and compare it with 4-switch and individually trained ones.

The results show that a slimmable network with more switches have similar performance, demonstrating the scalability of our proposed approach.

Finally, we apply slimmable networks on tasks of bounding-box object detection, instance segmentation and keypoints detection based on detection frameworks MMDetection and Detectron .

Following the settings of R-50-FPN-1?? BID28 , pre-trained ResNet-50 models at different widths are fine-tuned and evaluated.

The lateral convolution layers in feature pyramid network BID28 are same for different pre-trained backbone networks.

For individual models, we train ResNet-50 with different width multipliers on ImageNet and fine-tune them on each task individually.

For slimmable models, we first train on ImageNet using Algorithm 1.

Following , the moving averaged means and variances of switchable batch normalization are also fixed after training.

Then we fine-tune the slimmable models on each task using Algorithm 1.

The detection head and lateral convolution layers in feature pyramid network BID28 are shared across different switches in a slimmable network.

In this way, each switch in a slimmable network is with exactly same network architecture and FLOPs with its individual baseline.

More details of implementation are included in Appendix B.

We train all models on COCO 2017 train set and report Average Precision (AP) on COCO 2017 validation set in TAB6 .

In general, slimmable neural networks perform better than individually trained ones, especially for slim network architectures.

The gain of performance is presumably due to implicit model distillation BID14 BID36 and richer supervision signals.

Note that the white color in RGB is [255, 255, 255] , yellow in RGB is [255, 255, 0] .Visualization of Top-activated Images.

Our primary interest lies in understanding the role that the same channel played in different switches in a slimmable network.

We employ a simple visualization approach BID8 to visualize the images with highest activation values on a specific channel.

FIG4 shows the top-activated images of the same channel in different switches.

Images with green outlines are correctly classified by the corresponding model, while images with red outlines are mis-classified.

Interestingly the results show that for different switches, the major role of same channel (channel 3 9 in S-MobileNet v1) transits from recognizing white color (RGB value [255, 255, 255] ) to yellow color (RGB value [255, 255, 0] ) when the network width increases.

It indicates that the same channel in slimmable network may play similar roles (in this case to recognize colors of RGB value [255, 255, * ]) but have slight variations in different switches (the one in quarter-sized model focuses more on white color while the one in full model on yellow color).

FIG3 .

The results show that for shallow layers, the mean, variance, scale and bias are very close, while in deep layers they are diverse.

The value discrepancy is increased layer by layer in our observation, which also indicates that the learned features of a same channel in different switches have slight variations of semantics.

We introduced slimmable networks that permit instant and adaptive accuracy-efficiency trade-offs at runtime.

Switchable batch normalization is proposed to facilitate robust training of slimmable networks.

Compared with individually trained models with same width configurations, slimmable networks have similar or better performances on tasks of classification, object detection, instance segmentation and keypoints detection.

The proposed slimmable networks and slimmable training could be further applied to unsupervised learning and reinforcement learning, and may help to related fields such as network pruning and model distillation.end, we mainly conduct COCO experiments based on another detection framework: MMDetection , which has hyper-parameter settings with same pytorch-style ResNet-50.

With same hyper-parameter settings (i.e., RCNN R50 FPN 1 ??), we fine-tune both individual ResNet-50 models and slimmable ResNet-50 on tasks of object detection and instance segmentation.

Our reproduced results on ResNet-50 1.0?? is consistent with official models in MMDetection ).

For keypoint detection task, we conduct experiment on Detectron framework by modifying caffe-style ResNet-50 to pytorch-style and training on 4 GPUs without other modification of hyper-parameters.

We have released code (training and testing) and pretrained models on both ImageNet classification task and COCO detection tasks.

In our work, private parameters ??, ??, ??, ?? 2 of BN are introduced in Switchable Batch Normalization for each sub-network to independently normalize feature y = ?? y????? ??? ?? 2 + + ??, where y is input and y is output, ??, ?? are learnable scale and bias, ??, ?? 2 are moving averaged statistics for testing.

In switchable batch normalization, the private ??, ?? come for free because after training, they can be merged as y = ?? y + ?? , ?? = ?? ??? ?? 2 + , ?? = ?? ??? ?? ??. Nevertheless, we present ablation study on how these conditional parameters affect overall performance.

The results are shown in TAB8 . (-0.2) 35.9 (-0.7) 30.9 (-0.4) 28.8 (-0.3)

@highlight

We present a simple and general method to train a single neural network executable at different widths (number of channels in a layer), permitting instant and adaptive accuracy-efficiency trade-offs at runtime.

@highlight

The paper proposes an idea of combining different size models together into one shared net, greatly improving performance for detection

@highlight

This paper trains a single network executable at different widths.
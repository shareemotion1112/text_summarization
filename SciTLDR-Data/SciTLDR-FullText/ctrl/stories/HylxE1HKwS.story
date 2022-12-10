We address the challenging problem of efficient deep learning model deployment, where the goal is to design neural network architectures that can fit different hardware platform constraints.

Most of the traditional approaches either manually design or use Neural Architecture Search (NAS) to find a specialized neural network and train it from scratch for each case, which is computationally expensive and unscalable.

Our key idea is to decouple model training from architecture search to save the cost.

To this end, we propose to train a once-for-all network (OFA) that supports diverse architectural settings (depth, width, kernel size, and resolution).

Given a deployment scenario, we can then quickly get a specialized sub-network by selecting from the OFA network without additional training.

To prevent interference between many sub-networks during training, we also propose a novel progressive shrinking algorithm, which can train a surprisingly large number of sub-networks ($> 10^{19}$) simultaneously.

Extensive experiments on various hardware platforms (CPU, GPU, mCPU, mGPU, FPGA accelerator) show that OFA consistently outperforms SOTA NAS methods (up to 4.0% ImageNet top1 accuracy improvement over MobileNetV3) while reducing orders of magnitude GPU hours and $CO_2$ emission.

In particular, OFA achieves a new SOTA 80.0% ImageNet top1 accuracy under the mobile setting ($<$600M FLOPs).

Code and pre-trained models are released at https://github.com/mit-han-lab/once-for-all.

Deep Neural Networks (DNNs) deliver state-of-the-art accuracy in many machine learning applications.

However, the explosive growth in model size and computation cost gives rise to new challenges on how to efficiently deploy these deep learning models on diverse hardware platforms, since they have to meet different hardware efficiency constraints (e.g., latency, energy).

For instance, one mobile application on App Store has to support a diverse range of hardware devices, from a high-end Samsung Note10 with a dedicated neural network accelerator to a 5-year-old Samsung S5 with a much slower processor.

With different hardware resources (e.g., on-chip memory size, #arithmetic units), the optimal neural network architecture varies significantly.

Even running on the same hardware, under different battery conditions or workloads, the best model architecture also differs a lot.

Given different hardware platforms and efficiency constraints (defined as deployment scenarios), researchers either design compact models specialized for mobile (Howard et al., 2017; Sandler et al., 2018; or accelerate the existing models by compression (He et al., 2018) for efficient deployment.

However, designing specialized DNNs for every scenario is engineer-expensive and computationally expensive, either with human-based methods or NAS.

Since such methods need to repeat the network design process and retrain the designed network from scratch for each case.

Their total cost grows linearly as the number of deployment scenarios increases, which will result in excessive energy consumption and CO 2 emission (Strubell et al., 2019) .

It makes them unable to handle the vast amount of hardware devices (23.14 billion IoT devices till 2018 1 ) and highly dynamic deployment environments (different battery conditions, different latency requirements, etc.).

This paper introduces a new solution to tackle this challenge -designing a once-for-all network that can be directly deployed under diverse architectural configurations, amortizing the training cost.

The Figure 1: Left: a single once-for-all network is trained to support versatile architectural configurations including depth, width, kernel size, and resolution.

Given a deployment scenario, a specialized subnetwork is directly selected from the once-for-all network without training.

Middle: this approach reduces the cost of specialized deep learning deployment from O(N) to O(1).

Right: once-for-all network followed by model selection can derive many accuracy-latency trade-offs by training only once, compared to conventional methods that require repeated training.

inference is performed by selecting only part of the once-for-all network.

It flexibly supports different depths, widths, kernel sizes, and resolutions without retraining.

A simple example of Once for All (OFA) is illustrated in Figure 1 (left).

Specifically, we decouple the model training stage and the model specialization stage.

In the model training stage, we focus on improving the accuracy of all sub-networks that are derived by selecting different parts of the once-for-all network.

In the model specialization stage, we sample a subset of sub-networks to train an accuracy predictor and latency predictors.

Given the target hardware and constraint, a predictor-guided architecture search (Liu et al., 2018a ) is conducted to get a specialized sub-network, and the cost is negligible.

As such, we reduce the total cost of specialized neural network design from O(N) to O(1) (Figure 1 middle).

However, training the once-for-all network is a non-trivial task, since it requires joint optimization of the weights to maintain the accuracy of a large number of sub-networks (more than 10 19 in our experiments).

It is computationally prohibitive to enumerate all sub-networks to get the exact gradient in each update step, while randomly sampling a few sub-networks in each step will lead to significant accuracy drops.

The challenge is that different sub-networks are interfering with each other, making the training process of the whole once-for-all network inefficient.

To address this challenge, we propose a progressive shrinking algorithm for training the once-for-all network.

Instead of directly optimizing the once-for-all network from scratch, we propose to first train the largest neural network with maximum depth, width, and kernel size, then progressively fine-tune the once-for-all network to support smaller sub-networks that share weights with the larger ones.

As such, it provides better initialization by selecting the most important weights of larger sub-networks, and the opportunity to distill smaller sub-networks, which greatly improves the training efficiency.

We extensively evaluated the effectiveness of OFA on ImageNet with many hardware platforms (CPU, GPU, mCPU, mGPU, FPGA accelerator) and efficiency constraints.

Under all deployment scenarios, OFA consistently improves the ImageNet top1 accuracy by a significant margin compared to SOTA hardware-aware NAS methods while saving the GPU hours, dollars, and CO 2 emission by orders of magnitude.

On the ImageNet mobile setting, OFA achieves a new SOTA 80.0% top1 accuracy with 595M FLOPs.

To the best of our knowledge, this is the first time that the SOTA ImageNet top1 accuracy reaches 80% under the mobile setting.

Efficient Deep Learning.

Many efficient neural network architectures are proposed to improve the hardware efficiency, such as SqueezeNet (Iandola et al., 2016) , MobileNets (Howard et al., 2017; Sandler et al., 2018) , ShuffleNets (Ma et al., 2018; , etc.

Orthogonal to architecting efficient neural networks, model compression ) is another very effective technique for efficient deep learning, including network pruning that removes redundant units (Han et al., 2015) or redundant channels (He et al., 2018; , and quantization that reduces the bit width for the weights and activations Courbariaux et al., 2015; .

Neural Architecture Search.

Neural architecture search (NAS) focuses on automating the architecture design process (Zoph & Le, 2017; Zoph et al., 2018; Real et al., 2019; Cai et al., 2018a; .

Early NAS methods (Zoph et al., 2018; Real et al., 2019; Cai et al., 2018b) search for high-accuracy architectures without taking hardware efficiency into consideration.

Therefore, the produced architectures (e.g., NASNet, AmoebaNet) are not efficient for inference.

Recent hardwareaware NAS methods Cai et al., 2019; Wu et al., 2019) directly incorporate the hardware feedback into architecture search.

As a result, they are able to improve inference efficiency.

However, given new inference hardware platforms, these methods need to repeat the architecture search process and retrain the model, leading to prohibitive GPU hours, dollars and CO 2 emission.

They are not scalable to a large number of deployment scenarios.

The individually trained models do not share any weight, leading to a large total model size and high downloading bandwidth.

Dynamic Neural Networks.

To improve the efficiency of a given neural network, some work explored skipping part of the model based on the input image.

For example, ; Liu & Deng (2018) ; learn a controller or gating modules to adaptively drop layers; Huang et al. (2018) introduce early-exit branches in the computation graph; Lin et al. (2017) adaptively prune channels based on the input feature map; Kuen et al. (2018) introduce stochastic downsampling point to reduce the feature map size adaptively.

Recently, Slimmable Nets (Yu et al., 2019; Yu & Huang, 2019b) propose to train a model to support multiple width multipliers (e.g., 4 different global width multipliers), building upon existing human-designed neural networks (e.g., MobileNetV2 0.35, 0.5, 0.75, 1.0).

Such methods can adaptively fit different efficiency constraints at runtime, however, still inherit a pre-designed neural network (e.g., MobileNet-v2), which limits the degree of flexibility (e.g., only width multiplier can adapt) and the ability in handling new deployment scenarios where the pre-designed neural network is not optimal.

In this work, in contrast, we enable a much more diverse architecture space (depth, width, kernel size, and resolution) and a significantly larger number of architectural settings (10 19 v.s. 4 (Yu et al., 2019) ).

Thanks to the diversity and the large design space, we can derive new specialized neural networks for many different deployment scenarios rather than working on top of an existing neural network that limits the optimization headroom.

However, it is more challenging to train the network to achieve this flexibility, which motivates us to design the progressive shrinking algorithm to tackle this challenge.

Assuming the weights of the once-for-all network as W o and the architectural configurations as {arch i }, we then can formalize the problem as

where C(W o , arch i ) denotes a selection scheme that selects part of the model from the once-for-all network W o to form a sub-network with architectural configuration arch i .

The overall training objective is to optimize W o to make each supported sub-network maintain the same level of accuracy as independently training a network with the same architectural configuration.

Our once-for-all network provides one model but supports many sub-networks of different sizes, covering four important dimensions of the convolutional neural networks (CNNs) architectures, i.e., depth, width, kernel size, and resolution.

Following the common practice of many CNN models (He et al., 2016; Sandler et al., 2018; , we divide a CNN model into a sequence of units with gradually reduced feature map size and increased channel numbers.

Each unit consists of a sequence of layers where only the first layer has stride 2 if the feature map size decreases (Sandler et al., 2018) .

All the other layers in the units have stride 1.

progressively shrink the width We allow each unit to use arbitrary numbers of layers (denoted as elastic depth); For each layer, we allow to use arbitrary numbers of channels (denoted as elastic width) and arbitrary kernel sizes (denoted as elastic kernel size).

In addition, we also allow the CNN model to take arbitrary input image sizes (denoted as elastic resolution).

For example, in our experiments, the input image size ranges from 128 to 224 with a stride 4; the depth of each unit is chosen from {2, 3, 4}; the width expansion ratio in each layer is chosen from {3, 4, 6}; the kernel size is chosen from {3, 5, 7}. Therefore, with 5 units, we have roughly

19 different neural network architectures and each of them can be used under 25 different input resolutions.

Since all of these sub-networks share the same weights (i.e., W o ) (Cheung et al., 2019) , we only require 7.7M parameters to store all of them.

Without sharing, the total model size will be prohibitive.

Naïve Approach.

Training the once-for-all network can be cast as a multi-objective problem, where each objective corresponds to one sub-network.

From this perspective, a naïve training approach is to directly optimize the once-for-all network from scratch using the exact gradient of the overall objective, which is derived by enumerating all sub-networks in each update step, as shown in Eq. (1).

The cost of this approach is linear to the number of sub-networks.

Therefore, it is only applicable to scenarios where a limited number of sub-networks are supported (Yu et al., 2019) , while in our case, it is computationally prohibitive to adopt this approach.

Another naïve training approach is to sample a few sub-networks in each update step rather than enumerate all of them, which does not have the issue of prohibitive cost.

However, with such a large number of sub-networks that share weights thus interfere with each other, we find it suffers from significant accuracy drop.

In the following section, we introduce a solution to address this challenge by adding a progressive shrinking training order to the training process.

Correspondingly, we refer to the naïve training approach as random order.

Progressive Shrinking.

The once-for-all network comprises many sub-networks of different sizes where small sub-networks are nested in large sub-networks.

To prevent interference between the sub-networks, we propose to enforce a training order from large sub-networks to small sub-networks in a progressive manner.

We name this training order as progressive shrinking (PS).

An example of the training process with PS is provided in Figure 2 , where we start with training the largest neural network with the maximum kernel size (i.e., 7), depth (i.e., 4), and width (i.e., 6).

Next, we progressively fine-tune the network to support smaller sub-networks by gradually adding them into the sampling space (larger sub-networks may also be sampled).

Specifically, after training the largest network, we first support elastic kernel size which can choose from {3, 5, 7} at each layer, while the depth and width remain the maximum values.

Then, we support elastic depth and elastic width sequentially, as is shown in Figure 2 .

The resolution is elastic throughout the whole training process, which is implemented by sampling different image sizes for each batch of training data.

We also use the knowledge distillation technique after training the largest neural network (Hinton et al., 2015; Ashok et al., 2018; Yu & Huang, 2019b) .

It combines two loss terms using both the soft labels given by the largest neural network and the real labels.

In this example, we progressively support 4, 3, and 2 channel settings.

We perform channel sorting and pick the most important channels (with large L1 norm) to initialize the smaller channel settings.

The important channels' weights are shared.

Compared to the naïve approach, PS prevents small sub-networks from interfering large sub-networks, since large sub-networks are already well-trained when the once-for-all network is fine-tuned to support small sub-networks.

Additionally, during fine-tuning, the model is optimized in the local space around the well-trained large sub-networks by using a small learning rate and revisiting (i.e., sampling) well-trained large sub-networks.

Regarding the small sub-networks, they share the weights with the large ones.

Therefore, PS allows initializing small sub-networks with the most important weights of well-trained large sub-networks, which expedites the training process.

We describe the details of the PS training flow as follows:

• Elastic Kernel Size (Figure 3 left) .

We let the center of a 7x7 convolution kernel also serve as a 5x5 kernel, the center of which can also be a 3x3 kernel.

Therefore, the kernel size becomes elastic.

The challenge is that the centering sub-kernels (e.g., 3x3 and 5x5) are shared and need to play multiple roles (independent kernel and part of a large kernel).

The weights of centered sub-kernels may need to have different distribution or magnitude as different roles.

Forcing them to be the same degrades the performance of some sub-networks.

Therefore, we introduce kernel transformation matrices when sharing the kernel weights.

We use separate kernel transformation matrices for different layers.

Within each layer, the kernel transformation matrices are shared among different channels.

As such, we only need 25 × 25 + 9 × 9 = 706 extra parameters to store the kernel transformation matrices in each layer, which is negligible.

• Elastic Depth (Figure 3 right) .

To derive a sub-network that has D layers in a unit that originally has N layers, we keep the first D layers and skip the last N − D layers, rather than keeping any D layers as done in current NAS methods (Cai et al., 2019; Wu et al., 2019) .

As such, one depth setting only corresponds to one combination of layers.

In the end, the weights of the first D layers are shared between large and small models.

• Elastic Width (Figure 4) .

Width means the number of channels.

We give each layer the flexibility to choose different channel expansion ratios.

Following the progressive shrinking scheme, we first train a full-width model.

Then we introduce a channel sorting operation to support partial widths.

It reorganizes the channels according to their importance, which is calculated based on the L1 norm of a channel's weight.

Larger L1 norm means more important.

For example, when shrinking from a 4-channel-layer to a 3-channel-layer, we select the largest 3 channels; whose weights are shared with the 4-channel-layer ( Figure 4 left and middle).

Thereby, smaller sub-networks are initialized with the most important channels on the once-for-all network which is already well trained.

This channel sorting operation preserves the accuracy of larger sub-networks.

Table 1 : ImageNet top1 accuracy (%) performances of sub-networks under resolution 224 × 224.

"(D = d, W = w, K = k)" denotes a sub-network with d layers in each unit, and each layer has an width expansion ratio w and kernel size k. "Mbv3-L" denotes "MobileNetV3-Large"

Having trained a once-for-all network, the next stage is to derive the specialized sub-network for a given deployment scenario.

The goal is to search for a neural network that satisfies the efficiency (e.g., latency, energy) constraints on the target hardware while optimizing the accuracy.

Since OFA decouples model training from architecture search, we do not need any training cost in this stage.

To eliminate the repeated search cost, we use the predictor-guided architecture search process (Liu et al., 2018b) to find specialized sub-networks.

Specifically, we randomly sample 16K sub-networks with different architectures and input image sizes, then measure their accuracy on 10K validation images sampled from the original training set.

These [architecture, accuracy] pairs are used to train an accuracy predictor to predict the accuracy of a model given its architecture and input image size 2 .

We also build a latency lookup table (Cai et al., 2019) on each target hardware platform to predict the latency.

Given the target hardware and latency constraint, we conduct an evolutionary search (Real et al., 2019) based on the predictors to get a specialized sub-network.

Since the cost of searching with predictors is negligible, we only need 40 GPU hours to collect the data pairs, and the cost stays constant regardless of #deployment scenarios.

In this section, we first apply the progressive shrinking algorithm to train the once-for-all network on ImageNet (Deng et al., 2009 ).

Then we demonstrate the effectiveness of our trained once-for-all network on various hardware platforms (Samsung S7 Edge, Note8, Note10, Google Pixel1, Pixel2, LG G8, NVIDIA 1080Ti, V100 GPUs, Jetson TX2, Intel Xeon CPU, Xilinx ZU9EG, and ZU3EG FPGAs) with different latency constraints.

Training Details.

We use the same architecture space as MobileNetV3 (Howard et al., 2019) .

For training the full network, we use the standard SGD optimizer with Nesterov momentum 0.9 and weight decay 3e −5 .

The initial learning rate is 2.6, and we use the cosine schedule (Loshchilov & Hutter, 2016) for learning rate decay.

The full network is trained for 180 epochs with batch size 2048 on 32 GPUs.

Then we follow the schedule described in Figure 2 to further fine-tune the full network 3 .

The whole training process takes around 1,200 GPU hours on V100 GPUs.

This is a one-time training cost that can be amortized by many deployment scenarios.

Results.

Table 1 reports the top1 accuracy of sub-networks derived from the once-for-all networks that are trained with our progressive shrinking (PS) algorithm and random order respectively.

Due to space limits, we take 8 sub-networks for comparison, and each of them is denoted as "(D = d, W = w, K = k)".

It represents a sub-network that has d layers for all units while the expansion ratio and kernel size are set to w and k for all layers.

Compared to random order, PS can improve the ImageNet accuracy of sub-networks by a significant margin under all architectural settings.

Specifically, without architecture optimization, PS can achieve 74.8% top1 accuracy using 226M FLOPs under the architecture setting (D=4, W=3, K=3), which is on par with MobileNetV3-Large.

In contrast, random order only achieves 71.5% which is 3.3% lower.

We apply our trained once-for-all network to get specialized sub-networks for different hardware platforms.

For the GPU platforms, the latency is measured with batch size 64 on NVIDIA 1080Ti and V100 with Pytorch 1.0+cuDNN.

The CPU latency is measured with batch size 1 on Intel Xeon E5-2690 v4+MKL-DNN 4 .

To measure mobile latency, we use TF-Lite with batch size 1.

On Jetson TX2, we use a batch size of 16.

On Xilinx ZU9EG and ZU3EG FPGAs, we use DNNDK with batch size 1.

Comparison with NAS on Mobile.

Table 2 reports the comparison between OFA and state-of-theart hardware-aware NAS methods on the mobile platform (Pixel1).

OFA is much more efficient than NAS when handling multiple deployment scenarios since the cost of OFA is constant while others are linear to the number of deployment scenarios (N ).

With N = 40, the training time of OFA is 16× faster than ProxylessNAS, 19× faster than FBNet, and 1,300× faster than MnasNet.

Without retraining, OFA achieves 76.0% top1 accuracy on ImageNet, which is 0.8% higher than MobileNetV3-Large while maintaining similar mobile latency.

We can further improve the top1 accuracy to 76.4% by fine-tuning the specialized sub-network for 25 epochs and to 76.9% by finetuning for 75 epochs.

Besides, we also observe that OFA with PS can achieve 3.6% better accuracy than without PS, showing the effectiveness of PS.

Results under Different Efficiency Constraints.

Figure 5 summarizes the results of OFA under different FLOPs and Pixel1 latency constraints.

OFA achieves 79.1% ImageNet top1 accuracy with 389M FLOPs, being 2.8% more accurate than EfficientNet-B0 that has similar FLOPs.

With 595M FLOPs, OFA reaches a new SOTA 80.0% ImageNet top1 accuracy under the mobile setting (<600M FLOPs), which is 0.2% higher than EfficientNet-B2 while using 1.68× fewer FLOPs.

More importantly, OFA runs much faster than EfficientNets on hardware.

Specifically, with 143ms Pixel1 latency, OFA achieves 80.1% ImageNet top1 accuracy, being 0.3% more accurate and 2.6× faster than EfficientNet-B2.

Figure 6 reports detailed comparisons between OFA and MobileNetV3 on 6 mobile devices.

Remarkably, OFA can produce the entire trade-off curves with many points over a wide range of latency constraints.

It is impossible for previous NAS methods Cai et al., 2019) due to the prohibitive training cost.

Results on Diverse Hardware Platforms.

Besides the mobile platforms, we extensively studied the effectiveness of OFA on 6 additional hardware platforms (Figure 7 ) using the ProxylessNAS architecture space (Cai et al., 2019) .

OFA consistently improves the trade-off between accuracy and latency by a significant margin, especially on GPUs which have more parallelism.

With similar latency as MobileNetV2 0.35, "OFA #25" improves the ImageNet top1 accuracy from MobileNetV2's 60.3% to 72.6% (+12.3% improvement) on the 1080Ti GPU.

It reveals the insight that using the same model for different deployment scenarios with only the width multiplier modified has a limited impact on efficiency improvement: the accuracy drops quickly as the latency constraint gets tighter.

On Xilinx ZU9EG and ZU3EG FPGAs, our quantized OFA models still outperform MobileNetV2 under all latency constraints.

We profiled the FPGA and drew its roofline model (see Appendix B): OFA models improved the arithmetic intensity by up to 48%/43% and GOPS/s by 70%/92% compared with the MobileNetV2/MnasNet family.

OFA designs specialized models that better fit the hardware.

We proposed Once for All (OFA), a new methodology that decouples model training from architecture search for efficient deep learning deployment under a large number of deployment scenarios.

Unlike previous approaches that design and train a neural network for each deployment scenario, we designed a once-for-all network that supports different architectural configurations, including elastic depth, width, kernel size, and resolution.

It greatly reduces the training cost (GPU hours, energy Specialized deployment results on CPU, GPU, mGPU, and FPGA accelerator.

Specialized models by OFA consistently achieve significantly higher ImageNet accuracy with similar latency than non-specialized neural networks.

More remarkably, specializing for a new hardware platform does not add training cost using OFA.

consumption, and CO 2 emission) compared to conventional methods.

To prevent sub-networks of different sizes from interference, we proposed a progressive shrinking algorithm that enables a large number of sub-network to achieve the same level of accuracy compared to training them independently.

Experiments on a diverse range of hardware platforms and efficiency constraints demonstrated the effectiveness of our approach.

We use a three-layer feedforward neural network that has 400 hidden units in each layer as the accuracy predictor.

Given a model, we encode each layer in the neural network into a one-hot vector based on its kernel size and expand ratio, and we assign zero vectors to layers that are skipped.

Besides, we have an additional one-hot vector that represents the input image size.

We concatenate these vectors into a large vector that represents the whole neural network architecture and input image size, which is then fed to the three-layer feedforward neural network to get the predicted accuracy.

In our experiments, this simple accuracy prediction model can provide very accurate predictions.

At convergence, the root-mean-square error (RMSE) between predicted accuracy and estimated accuracy on the test set is only 0.21%.

Figure 8 shows the relationship between the RMSE of the accuracy prediction model and the final results (i.e., the accuracy of selected sub-networks).

We can find that lower RMSE typically leads to better final results.

Once for All #25

We analyze our specialized models on FPGA using DNNDK 5 .

The profiling results are summarized in Figure 9 , while the roofline models (Williams et al., 2009 ) on ZU9EG and ZU3EG are illustrated in Figure 10 .

Unlike high-end FPGAs that have DRAM modules which are directly connected to the programmable logic (PL), ZU9EG and ZU3EG share the same DRAM module with the processing system (includes the ARM CPU).

Thus memory bandwidth is an expensive resource.

Correspondingly, our specialized models reduce costly memory accesses by increasing arithmetic intensity (i.e., the number of arithmetic operations for each byte of memory access), which results in higher GOPS/s.

Finally, we have two stages of fine-tuning to incorporate elastic width.

We sample four sub-networks and aggregate their gradients in each update step.

The first stage (i.e., W ∈ [6, 4]) takes 25 epochs with an initial learning rate of 0.08 while the second stage (i.e., W ∈ [6, 4, 3]) takes 125 epochs with an initial learning rate of 0.24.

<|TLDR|>

@highlight

We introduce techniques to train a single once-for-all network that fits many hardware platforms.

@highlight

Method results in a network from which one can extract sub-networks for various resouce constraints (latency, memory) which perform well without a need for retraining.

@highlight

This paper tries to tackle the problem of searching best architectures for specialized resource constraint deployment scenarios with a prediction based NAS method.
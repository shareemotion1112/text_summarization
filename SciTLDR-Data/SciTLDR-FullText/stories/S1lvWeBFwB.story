We present Random Partition Relaxation (RPR), a method for strong quantization of the parameters of convolutional neural networks to binary (+1/-1) and ternary (+1/0/-1) values.

Starting from a pretrained model, we first quantize the weights and then relax random partitions of them to their continuous values for retraining before quantizing them again and switching to another weight partition for further adaptation.

We empirically evaluate the performance of RPR with ResNet-18, ResNet-50 and GoogLeNet on the ImageNet classification task for binary and ternary weight networks.

We show accuracies beyond the state-of-the-art for binary- and ternary-weight GoogLeNet and competitive performance for ResNet-18 and ResNet-50 using a SGD-based training method that can easily be integrated into existing frameworks.

Deep neural networks (DNNs) have become the preferred approach for many computer vision, audio analysis and general signal processing tasks.

However, they are also known for their associated high computation workload and large model size.

These are great hurdles to their wide-spread adoption due to the consequential cost, which is often prohibitive for low-power, mobile and alwayson applications.

This concern has driven a lot of research into various DNN topologies and their basic building blocks in order to reduce the required compute cost at a small accuracy penalty.

Furthermore, efforts have been made towards compressing the models from often hundreds of megabytes to a size that is suitable for over-the-air updates and does not negatively impact the user experience by taking up lots of storage on consumer devices and long loading times.

Recent research into specialized hardware accelerators has shown that improvements by 10-100× in energy efficiency over optimized software are achievable (Sze et al., 2017) .

These accelerators can be integrated into a system-on-chip like those used in smartphones and highly integrated devices for the internet-of-things market.

These devices still spend most energy on I/O for streaming data in and out of the hardware unit repeatedly as only a limited number of weights can be stored in working memory-and if the weights fit on chip, local memories and the costly multiplications start dominating the energy cost.

This allows devices such as (Andri et al., 2018) Quantizing neural networks is crucial to allow more weights to be stored in on-chip working memory or to be loaded more efficiently from external memory, thereby reducing the number of repeated memory accesses to load and store partial results.

Complex network compression schemes cannot be applied at this point as decompression is often a lengthy process requiring a lot of energy by itself.

Furthermore, by strongly quantizing the network's parameters, the multiplications in the convolution and linear layers can be simplified, replaced with lightweight bit-shift operations, or even completely eliminated in case of binary and ternary weight networks (BWNs, TWNs) (Zhou et al., 2017) .

Extreme network quantization has started with BinaryConnect (Courbariaux et al., 2015) proposing deterministic or stochastic rounding during the forward pass and updating the underlying continuous-valued parameters based on the so-obtained gradients which would naturally be zero almost everywhere.

Then, XNOR-net (Rastegari et al., 2016) successfully trained both binary neural networks (BNNs), where the weight and the activations are binarized, as well as BWNs, with a clear jump in accuracy over BinaryConnect by means of dynamic (input-dependent) normalization and for the first time reporting results for a deeper and more modern ResNet topology.

Shortly after, (Li et al., 2016) presented ternary weight networks (TWNs), where they introduced learning the quantization thresholds while keeping the quantization levels fixed and showing a massive improvement over previous work and a top-1 accuracy drop of only 3.6% on ImageNet, making TWNs a viable approach for practical inference.

Thereafter, (Zhu et al., 2017) introduced trained ternary quantization (TTQ), relaxing the constraint of the weights being scaled values of {−1, 0, 1} to {α 1 , 0, α 2 }.

A method called incremental network quantization (INQ) was developed in (Zhou et al., 2017) , making clear improvements by neither working with inaccurate gradients or stochastic forward passes.

Instead, the network parameters were quantized step-by-step, allowing the remaining parameters to adapt to the already quantized weights.

This further improved the accuracy for TWNs and fully matched the accuracy of the baseline networks with 5 bit and above.

Last year, (Leng et al., 2018) presented a different approach to training quantized neural networks by relying on the alternating direction method of multipliers (ADMM) more commonly used in chemical process engineering.

They reformulated the optimization problem for quantized neural networks with the object function being a sum of two separable objectives and a linear constraint.

ADMM alternatingly optimizes each of these objectives and their dual to enforce the linear constraint.

In the context of quantized DNNs, the separable objectives are the optimization of the loss function and the enforcement of the quantization constraint, which results in projecting the continuous values to their closest quantization levels.

While ADMM achieves state-of-the-art results to this day, it requires optimization using the extragradient method, thereby becoming incompatible with standard DNN toolkits and hindering widespread adoption.

A few months ago, quantization networks (QNs) was introduced in (Yang et al., 2019) .

They pursue a very different approach, annealing a smoothed multi-step function the hard steps quantization function while using L2-norm gradient clipping to handle numerical instabilities during training.

They follow the approach of TTQ and learn the values of the quantization levels.

In this section, we describe the intuition behind RPR, its key components and their implementation.

When training DNNs, we optimize the network's parameters w ∈ R d to minimize a non-convex function f ,

This has been widely and successfully approached with stochastic gradient descent-based methods for DNNs in the hope of finding a good local optimum close to the global one of this non-convex function.

As we further constrain this optimization problem by restricting a subset of the parameters to take value in a finite set of quantization levels L, we end up with a mixed-integer non-linear program (MINLP):

where w q are the quantized (e.g., filter weights) and w c the continuous parameters (e.g., biases, batch norm factors) of the network.

Common sets of quantization levels L are symmetric uniform with or without zero ({0} ∪ {±i} i or {±i} i ) and symmetric exponential ({0} ∪ {±2 i } i ) due to their hardware suitability (multiplications can be implemented as bit-shifts).

Less common but also used are trained symmetric or arbitrary quantization levels ({±α i } i or {α i } i ).

Typically, the weights of the convolutional and linear layers are quantized except for the first and last layers in the network, since quantizing these has been shown to have a much stronger impact on the final accuracy than that of the other layers .

As in most networks the convolutional and linear layers are followed by batch normalization layers, any linear scaling of the quantization levels has no impact on the optimization problem.

Mixed-integer non-linear programs such as (2) are NP-hard and practical optimization algorithms trying to solve it are only approximate.

Most previous works approach this problem by means of annealing a smoothed multi-step function applied to underlying non-quantized weights (and clipping the gradients) or by quantizing the weights in the SGD's forward pass and introducing proxy gradients in the backward pass (e.g., the straight-through estimator (STE)) to allow the optimization to progress despite the gradients being zero almost everywhere.

Recently, (Leng et al., 2018) proposed to use the alternating direction method of multipliers (ADMM) to address this optimization problem with promising results.

However, their method requires a non-standard gradient descent optimizer, thus preventing simple integration into commonly used deep learning toolkits and thereby wide-spread adoption.

For RPR, we propose to approach the MINLP through alternating optimization.

Starting from continuous values for the parameters in W q , we randomly partition W q into W

This allows the relaxed parameters to co-adapt to the constrained/quantized ones.

This step is repeated, alternating between optimizing other randomly relaxed partitions of the quantized parameters (cf.

Figure 1 .

As the accuracy converges, FF is increased until it reaches 1, at which point all the constrained parameters are quantized.

The non-linear program (3) can be optimized using standard SGD or its derivatives like Adam, RMSprop, . . . .

We have experimentally found performing gradient descent on (3) for one full epoch before advancing to the next random partition of W q to converge faster than other configurations.

Note that w constr q is always constructed from the underlying continuous representation of w q .

We also initialize w relaxed q to the corresponding continuous-valued representation as well, thus providing a warm-start for optimizing (3) using gradient descent.

Starting with the standard initialization method for the corresponding network has worked well for training VGG-style networks on CIFAR-10 and ResNet-18 on ImageNet.

We experimentally observed that smaller freezing fractions FF can be used for faster convergence at the expense of less reliable convergence to a good local optimum.

However, a network can be quantized much faster and tends to reach a better local optimum when starting from a pre-trained network.

When convolution and linear layers are followed by a batch normalization layer, their weights become scale-invariant as the variance and mean are immediately normalized, hence we can define our quantization levels over the range [−1, 1] without adding any restrictions.

However, the continuous-valued parameters of a pretrained model might not be scaled suitably.

We thus re-scale each filter of each layer i to minimize the 2 distance between the continuous-valued and the quantized parameters, i.e.

Practically, we implemented (4) using a brute force search over 1000 points spread uniformly over [0, max i |w i |] before locally fine-tuning the best result using the downhill simplex method.

The time for this optimization is negligible relative to the overall compute time and in the range of a few minutes for all the weights to be quantized within ResNet-50.

We conducted experiments on ImageNet with ResNet-18, ResNet-50, and GoogLeNet in order to show the performance of RPR by training them as binary weight and ternary weight networks.

We refrain from reporting results on CIFAR-10 and with AlexNet on Imagenet as these networks are known to be overparametrized and thus rely on additional regularization techniques not to overfitthis is an irrelevant scenario for resource-efficient deployment of DNNs as a smaller DNN would be selected anyway.

Following common practice, we do not quantize the first and last layers of the network.

If not stated otherwise, we start from the corresponding pretrained model available through the torchvision v0.4.0 library.

The preprocessing and data augmentation methods used in related work vary wildly and from simple image rescaling and croping with horizontal flips and mean/variance normalization to methods with randomized rescaling, cropping to different aspect ratios, and brightness/contrast/saturation/lighting variations.

Consistent with literature, we have found that a quite minimal preprocessing by rescaling the image such that the shorter edge has 256 pixels followed by random crops of 224×224 pixels and random horizontal flips showed best results.

During testing, the same resizing and a 224×224 center crop were applied.

We observed simpler preprocessing methods working better: this is expected as the original networks' capacities are reduced by the strong quantization, and training the network to correctly classify images sampled from a richer distribution of distortions than that of the original data takes away some of the capacity of the network.

We trained the networks using the Adam optimizer with initial learning rates identical to the full-precision baseline models (10 −3 for all models).

During an initial training phase we use a freezing fraction FF = 0.9 until stabilization of the validation metric.

We proceed with FF = 0.95, 0.975, 0.9875, 1.0.

Each different FF was kept for 15 epochs, always starting with the initial learning rate and reducing it by 10× after 10 epochs at the specific FF.

After reaching FF = 1.0, the learning rate is kept for 10 cycles each at 1×, 0.1×, and 0.01× the initial learning rate.

An example of a freezing fraction and learning rate schedule is shown in Figure 2 .

In practice, quantizing a network with RPR requires a number of training epochs similar to training the full-precision model.

This is shown for the quantization of GoogLeNet to ternary weights in Figure 2 .

The quantization with FF = 0.9 requires 37 epochs followed by 45 epochs of iteratively increasing FF before a final phase of optimizing only the continuous parameters for 30 additional epochs.

We provide an overview of our results and a comparison to related work in Table 1 .

For ResNet-18, our method shows similar accuracy to the ADMM-based method, clearly outperforming other methods such as the XNOR-net BWN, TWN, and INQ.

As discussed before, the ADMM algorithm requires an optimization procedure that is not a simple variation of SGD and has thus not yet found widespread adoption.

A higher accuracy than RPR is achieved by TTQ with an enlarged network (2.25× as many parameters) and by Quantization Networks.

Both methods however, introduce trained quantization levels with dire consequences for hardware implementations: either as many multipliers as in full-

@highlight

State-of-the-art training method for binary and ternary weight networks based on alternating optimization of randomly relaxed weight partitions

@highlight

The paper proposes a new training scheme of optimizing a ternary neural network.

@highlight

Authors propose RPR, a way to randomly partition and quantize weights and train the remaining parameters followed by relaxation in alternate cycles to train quantized models.
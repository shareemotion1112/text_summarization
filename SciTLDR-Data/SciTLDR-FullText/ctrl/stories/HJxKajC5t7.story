We present a method to train self-binarizing neural networks, that is, networks that evolve their weights and activations during training to become binary.

To obtain similar binary networks, existing methods rely on the sign activation function.

This function, however, has no gradients for non-zero values, which makes standard backpropagation impossible.

To circumvent the difficulty of training a network relying on the sign activation function, these methods alternate between floating-point and binary representations of the network during training, which is sub-optimal and inefficient.

We approach the binarization task by training on a unique representation involving a smooth activation function, which is iteratively sharpened during training until it becomes a binary representation equivalent to the sign activation function.

Additionally, we introduce a new technique to perform binary batch normalization that simplifies the conventional batch normalization by transforming it into a simple comparison operation.

This is unlike existing methods, which are forced to the retain the conventional floating-point-based batch normalization.

Our binary networks, apart from displaying advantages of lower memory and computation as compared to conventional floating-point and binary networks, also show higher classification accuracy than existing state-of-the-art methods on multiple benchmark datasets.

Deep learning has brought about remarkable advancements to the state-of-the-art in several fields including computer vision and natural language processing.

In particular, convolutional neural networks (CNN's) have shown state-of-the-art performance in several tasks such as object recognition with AlexNet BID19 , VGG BID29 , ResNet and detection with R-CNN BID10 BID9 BID26 .

However, to achieve real-time performance these networks are dependent on specialized hardware like GPU's because they are computation and memory demanding.

For example, AlexNet takes up 250Mb for its 60M parameters while VGG requires 528Mb for its 138M parameters.

While the performance of deep networks has been gradually improving over the last few years, their computational speed has been steadily decreasing BID32 .

Notwithstanding this, interest has grown significantly in the deployment of CNN's in virtual reality headsets (Oculus, GearVR), augmented reality gear (HoloLens, Epson Moverio), and other wearable, mobile, and embedded devices.

While such devices typically have very restricted power and memory capacites, they demand low latency and real-time performance to be able to provide a good user experience.

Not surprisingly, there is considerable interest in making deep learning models computationally efficient to better suit such devices BID24 BID4 BID40 .Several methods of compression, quantization, and dimensionality reduction have been introduced to lower memory and computation requirements.

These methods produce near state-of-the-art results, either with fewer parameters or with lower precision parameters, which is possible thanks to the redundancies in deep networks BID3 .In this paper we focus on the solution involving binarization of weights and activations, which is the most extreme form of quantization.

Binarized neural networks achieve high memory and computational efficiency while keeping performance comparable to their floating point counterparts.

BID5 have shown that binary networks allow the replacement of multiplication and additions by simple bit-wise operations, which are both time and power efficient.

The challenge in training a binary neural network is to convert all its parameters from the continuous domain to a binary representation, typically done using the sign activation function.

However, since the gradient of sign is zero for all nonzero inputs, it makes standard back-propagation impossible.

Existing state-of-the-art methods for network binarization BID5 BID25 alternate between a binarized forward pass and a floating point backward pass to circumvent this problem.

In their case, the gradients for the sign activation are approximated during the backward pass, thus introducing inaccuracies in training.

Furthermore, batch normalization BID16 is necessary in binary networks to avoid exploding feature map values due to the large scale of the weights.

However, during inference, using batch normalization introduces intermediary floating point representations.

This means, despite binarizing weights and activations, these networks can not be used on chips that do not support floating-point computations.

In our method, the scaled hyperbolic tangent function tanh is used to bound the values in the range [−1, 1].

The network starts with floating point values for weights and activations, and progressively evolves into a binary network as the scaling factor is increased.

Firstly, this means that we do not have to toggle between the binary and floating point weight representations.

Secondly, we have a continuously differentiable function that allows backpropagation passes.

As another important contribution, we reduce the standard batch normalization operation during the inference stage to a simple comparison.

This modification is not only very efficient and can be accomplished using fixedpoint operations, it is also an order of magnitude faster than the floating-point counterpart.

More clearly, while existing binarization methods perform, at each layer, the steps of binary convolutions, floating-point batch normalization, and sign activation, we only need to perform binary convolutions followed by our comparison-based batch normalization, which serves as the sign activation at the same time (see Fig. 1 ).

convolutions, batch norm, sign activation convolutions, combined comparison batch norm and sign activation Figure 1 : Schematic comparison of a typical binary neural network generated by existing methods (top half) with that of our method (bottom half).

Existing methods are unable to avoid the 32-bit floating point batch norm, while we simplify it to a comparison operation, which simultaneously serves as the activation layer.

We validate the performance of our self-binarizing networks by comparing them to those of existing binarization methods.

We choose the standard bechmarks of CIFAR-10, CIFAR-100 BID18 ) as well as ImageNet BID27 and popular network architectures such as VGG and AlexNet.

We demonstrate higher accuracies despite using less memory and fewer computations.

To the best of our knowledge, our proposed networks are the only ones that are free of any floating point computations and can therefore be deployed on low-precision integrated chips or micro-controllers.

In what follows, in Sec. 2 we describe previous work related to reducing the network complexity and where we are placed among them.

In Sec. 3 we explain how the scaled tanh function can be used for progressive binarization.

In Sec. 4 we explain our binarization method for weights and activations and explain how to simplify batch normalization at inference time.

In Sec. 5 we compare our technique to existing state-of-the-art binary networks on standard benchmarks and demonstrate the performance gains from our proposed batch normalization simplification.

Sec. 6 concludes the paper.

Making deep networks memory and computation efficient has been approached in various ways.

In this section we cover some of the relevant literature and explain how our work is positioned with respect to the state-of-the-art.

The interested reader may refer to BID4 , BID24 , and Zhu (2018) for a wider coverage.

Since most of the computation in deep networks is due to convolutions, it is logical to focus on reducing the computational burden due to them.

Howard et al. FORMULA0 ; BID34 BID8 employ variations of convolutional layers by taking advantage of the separability of the kernels either by convolving with a group of input channels or by splitting the kernels across the dimensions.

In addition to depth-wise convolutions, MobileNetV2 BID28 uses inverted residual structures to learn how to combine the inputs in a residual block.

In general, these methods try to design a model that computes convolutions in an efficient manner.

This is different from this work because it focuses on redesigning the structure of the network, while ours tries to reduce the memory requirements by using lower precision parameters.

However, our method can be applied to these networks as well.

Reducing the number of weights likewise reduces the computational and memory burden.

Due to the redundancy in deep neural networks BID3 , there are some weights that contribute more to the output than others.

The process of removing the less contributing weights is called pruning.

In BID20 , the contribution of weights is measured by the effect on the training error when this parameter is zeroed.

In Deep Compression BID11 , the weights with lowest absolute value are pruned and the remaining are quantized and compressed using Huffman coding.

In other work, Fisher Information BID31 ) is used to measure the amount of information the output carries about each parameter, which allows pruning.

While these approaches often operate on already trained networks and fine-tune them after compression, our method trains a network to a binary state from scratch without removing any parameters or feature maps.

This allows us to retain the original structure of the network, while still leaving the potential for further compression after or during binarization using pruning techniques.

Another way to reduce memory consumption and potentially improve computational efficiency is the quantization of weights.

Quantized neural networks BID36 occupy less memory while retaining similar performance as their full precision counterparts.

DoReFaNet BID36 proposes to train low bitwidth networks with low bitwidth gradients using stochastic quantization.

Similarly, devise a weight partitioning technique to incrementally quantize the network at each step.

The degree of quantization can vary between techniques.

BID7 , BID37 , and BID22 quantize weights to three levels, i.e., two bits only.

These quantizations, while severe, still allow for accurate inference.

However, to improve computational efficency, specialized hardware is needed to take advantage of the underlying ternary operations.

An extreme form of such quantization is binarization, which requires only one bit to represent.

Expectation BackPropagation (EBP) paved the way for training neural networks for precision limitedhardware BID30 .

BinaryConnect BID5 extended the idea to train neural networks with binary weights.

The authors later propose BinaryNet to binarize activations as well.

Similarly, XNORNet BID25 extends BinaryNet by adding a scaling factor to the parameters of every layer while keeping the weights and activations binary.

ABCNet approximates full precision parameters as a linear combination of binary weights and uses multiple binary activations to compensate for the information loss arising from quantization.

BID13 use Hessian approximations to minimize loss with respect to the binary weights during training.

The focus of our work is the binarization of weights and activations of a network.

In previous binarization methods, the binarization process is non-differentiable leading to approximations during the training that can affect the final accuracy.

In contrast, we use a differentiable function to pro-gressively self-binarize the network and improve its accuracy.

Additionally, we differ from these techniques as we introduce a comparison-based binary batch normalization that eliminates all floating point operations at inference time.

In typical binary network training, during the forward pass, the floating-point weights and activations are quantized to binary values {−1, 1}, through a piece-wise constant function, most commonly the sign function: DISPLAYFORM0 This non-linear activation leads to strong artifacts during the forward pass, and does not generate gradients for backpropagation.

The derivatives of the binarized weights are therefore approximately computed using a Straight Through Estimator (STE) BID1 .

STE creates non-zero derivative approximations for functions that either have a zero derivative everywhere or are nondifferentiable.

Typically, the derivative of sign is estimated by using the following STE: DISPLAYFORM1 In the backward propagation step, the gradients are computed on the binarized weights using the STE and the corresponding floating-point representations are updated.

Since both forward and backward functions are different, the training is ill-defined.

The lack of an accurate derivative for the weights and activations creates a mismatch between the quantized and floating-point values and influences learning accuracy.

This kind of problems has been studied previously, and continuation methods have been proposed to simplify its solution BID0 .

To do so, the original complex and non-smooth optimization problem is transformed by smoothing the original function and then gradually decreasing the smoothness during training, building a sequence of sub-optimization problems converging to the original one.

For example, BID2 apply these methods on the last layer of a neural network to predict hashes from images.

Following this philosophy, we introduce a much simpler and efficient training method that allows the network to self-binarize.

We pass all our weights and activations through the hyperbolic tangent function tanh whose slope can be varied using a scale parameter ν > 0.

As seen in FIG0 , when ν is large enough the tanh(νx) converges to the sign(x) function: DISPLAYFORM2 Throughout the training process, the weights and activations use floating-point values.

Starting from a value of ν = 1, as the scale factor is progressively incremented, the weights and activations are forced to attain binary values {−1, 1}. During the training and while ν is still small, the derivative of tanh exists for every value of x and is expressed as DISPLAYFORM3 where sech is the hyperbolic secant.

Using the scaled tanh, we can build a continuously differentiable network which progressively approaches a binary state, leading to a more principled approach to obtain a binarized network.

In this section, we formally describe our self-binarizing approach.

We first explain how weights and activations are binarized.

Then, we propose a more efficient, comparison-based batch-normalization method that is more suitable when working in binary settings.

As stated above, we cannot use binary weights at training time as it would make gradient computation infeasible.

Instead, we use a set of constrained floating-point weights W at each layer .

Unlike traditional networks, these weights are not learnable parameters of our model, but depend on learnable parameters.

For each layer of the network, we define a set of learnable, unconstrained parameters P and use the scaled tanh to compute the weights as DISPLAYFORM0 where ν e is the scale factor at epoch e, taken from a sequence 1 = ν 0 < ν 1 < . . .

< ν M → ∞ of increasingly larger values.

During training, parameters P are updated to minimize a loss function using a standard gradient-based optimization procedure, such as stochastic gradient descent.

The scaled tanh transforms the parameters P to obtain weights W that are bounded in the range [−1, 1] and become closer to binary values as the training proceeds.

At the end of the training, weights W are very close to exact binary values.

At this point, we obtain the final binary weights B by taking the sign of the parameters, DISPLAYFORM1 At inference time, we drop all learnable parameters P and constrained weights W , and use only binary weights B .

Just as for weights, we cannot use binary activations either at training time.

We follow the idea as above to address activation self-binarization as well.

During training, we use the scaled tanh as the activation function of the network.

For a given layer , the activation function transforms the output O of the layer to lead to the activation DISPLAYFORM0 The activations are constrained to lie within the range [−1, 1], and eventually become binary at the end of the training procedure.

At inference time we make the network completely binary by substituting the scaled tanh by the sign operator as the binary activation function.

Batch Normalization (BN) introduced by BID16 accelerates the training of a general deep network.

During training, the BN layers compute the running mean µ r and standard deviation σ r of the feature maps that pass through them, as well as two parameters, β and γ, that define an affine transformation.

Later, at inference time, BN layers normalize and transform the input I to obtain an output O as given by DISPLAYFORM0 For binary networks BID5 BID25 in particular, BN becomes essential in order to avoid exploding activation values.

However, using BN brings in the limitation of relying on floating-point computations.

Apart from affecting computation and memory requirements, the floating-point BN effectively eliminates the possibility of using the network on low-precision hardware.

A useful observation of BN is that, in a networks with binary activations, the output of the BN layers is always fed into a sign function.

This means only the sign of the normalized output is kept, while its magnitude is not relevant for the subsequent layers.

We can leverage this property to simplify the BN operation.

The sign of the output O of a BN layer can be reformulated as: DISPLAYFORM1 with DISPLAYFORM2 While T is a floating-point value, in practice we represent it as a fixed-point 8-bit integer.

This sacrifices some amount of numerical precision, but we observed no negative effect on the performance of our models.

We refer to our simplified batch normalization as Binary Batch Normalization (BinaryBN).Note that the derivation of Eq. FORMULA8 does not hold when γ = 0.

This is handled as a special case that simply evaluates β > 0.

It must be emphasized that BinaryBN is not an approximate method; it computes the exact value of the sign of the output of the standard BN.During training we use the conventional BN layers.

At inference time, we replace them with the BinaryBN without any loss in prediction accuracy.

Our trained models can thus bypass the floatingpoint batch normalization with an efficient alternative that requires mere comparison operations.

In this section, we first compare our self-binarization method against other known techniques.

Later on, we discuss and demonstrate the efficiency gained using our proposed BinaryBN instead of the typical BN layer.

We compare our self-binarizing networks with other state-of-the-art methods that use binary networks.

BID5 present BinaryConnect (BC), a method that only binarizes the weights.

Binary Neural Networks (BNN) improves BC by also binarizing the activations.

Similarly, BID25 present two variants of their method: Binary Weight Networks (BWN), which only binarizes the weights, and XNORnet (XNOR), which binarizes both weights and activations.

For a fair comparison and to prove the generality of our method, we use the original implementations of BC, BNN, BWN, and XNOR, and apply our self-binarizing technique to them.

Specifically, we substituted the weights from the original implementations by a pair of parameters P and constrained weights W given by Eq. (5).

Also, we substituted the activation functions by the scaled tanh as described in Eq. FORMULA6 .

At inference time we used the binary weights obtained with Eq. (6) and the sign operator as the activation function.

Additionally, we replace the batch normalization layers by our BinaryBN layer for the cases where the activations are binarized.

We evaluate the methods on three common benchmark datasets: CIFAR-10, CIFAR-100 BID18 ) and ILSVRC12 ImageNet BID27 .

For CIFAR-10 and CIFAR-100, we use a VGG-16-like network with data augmentation as proposed in BID21 : 4 pixels are padded on each side, and a 32x32 patch is randomly cropped from the padded image or its horizontal flip.

During testing, only a single view of the original 32x32 image is evaluated.

The model is trained with a mini-batch size of 256 for 100 epochs.

The ILSVRC12 ImageNet dataset is used to train an AlexNet-like network without drop-out or local response normalization layers.

We use the data augmentation strategy from .

At inference time, only the center crop is used from the validation set.

The model is trained with a mini-batch size of 64 and a total of 50 epochs.

In all our experiments we increase ν from 1 to 1000 during training in an exponential manner.

The final ν = 1000 is large enough to make weights and activations almost binary in practice.

We optimize all models using Adam BID17 with an exponentially decaying learning rate starting at 10 −3 .

TAB0 shows the results of the experiments.

Our self-binarizing approach achieves the highest accuracies for CIFAR-10 and ImageNet.

Our method is only slightly outperformed by BC for CIFAR-100, but still gives better Top-5 accuracy for the same model.

For both weights and activation binarization, our method obtains the best results across all datasets and architectures.

What is remarkable is that the improved performance comes despite eliminating floating-point computations and using drastically fewer bits than the previous methods, as can be seen in the columns B w , B a and B BN of TAB0 .

For CIFAR-10 and CIFAR-100, BWN outperforms the full precision models likely because the binarization serves as a regularizer BID5 .

With all other computations being common between our self-binarizing networks and other networks with binary weights and activations, any difference in computational efficiency has to arise from the use of a different batch normalization scheme.

We therefore compare our proposed BinaryBN layer to the conventional BN layer as well as to the Shift-based Batch Normalization (SBN) proposed by .

SBN proposes to round both γ and σ r to their nearest power of 2 and replace multiplications and divisions by left and right shifts, respectively.

SBN follows the same formula as BN, and is used both at training and inference time, so that the network is trained with the rounded parameters.

Table.

2 summarizes the requirements of memory and computational time of these three types of batch normalization layers.

We assume the standard case where a binary convolution is followed by a batch normalization layer and then a binary activation function.

For storing BN parameters, we need four 32-bit vectors of length c, amounting to 32×4c = 128c bits, c being the number of channels in this layer.

For SBN, we need two 32-bit vectors and two 8-bit vectors of length c, resulting in 80c bits.

For BinaryBN, we need an 8-bit vector of size c to store the T value of each channel, and a binary vector for the sign of γ, totaling 9c bits.

We experimentally assessed the time and memory requirements of the three batch normalization techniques.

We run batches of increasing sizes through BN, SBN and BinaryBN layers with randomly generated values for µ r , σ r , β, and γ, and measure time and memory consumption.

FIG1 shows the results.

Overall, BinaryBN is nearly one order of magnitude less memory consuming and faster than BN and SBN.

We present a novel method to binarize a deep network that is principled, simple, and results in binarization of weights and activations.

Instead of relying on the sign function, we use the tanh function with a controllable slope.

This simplifies the training process without breaking the flow of derivatives in the back-propagation phase as compared to that of existing methods that have to toggle between floating-point and binary representations.

In addition to this, we replace the conventional batch normalization, which forces existing binarization methods to use floating point computations, by a simpler comparison operation that is directly adapted to networks with binary activations.

Our simplified batch normalization is not only computationally trivial, it is also extremely memoryefficient.

Our trained binary networks outperform those of existing binarization schemes on standard benchmarks despite using lesser memory and computation.

<|TLDR|>

@highlight

A method to binarize both weights and activations of a deep neural network that is efficient in computation and memory usage and performs better than the state-of-the-art.
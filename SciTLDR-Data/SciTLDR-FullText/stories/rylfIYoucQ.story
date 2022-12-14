Deep neural networks require extensive computing resources, and can not be efficiently applied to embedded devices such as mobile phones, which seriously limits their applicability.

To address this problem, we propose a novel encoding scheme by using {-1,+1} to decompose quantized neural networks (QNNs) into multi-branch binary networks, which can be efficiently implemented by bitwise operations (xnor and bitcount) to achieve model compression, computational acceleration and resource saving.

Our method can achieve at most ~59 speedup and ~32 memory saving over its full-precision counterparts.

Therefore, users can easily achieve different encoding precisions arbitrarily according to their requirements and hardware resources.

Our mechanism is very suitable for the use of FPGA and ASIC in terms of data storage and computation, which provides a feasible idea for smart chips.

We validate the effectiveness of our method on both large-scale image classification (e.g., ImageNet) and object detection tasks.

tational acceleration, many solutions have been proposed, such as network sparse and pruning 23 [7, 22, 12, 25 , 21], low-rank approximation [5, 11, 23] , architecture design BID2 10, 8, BID1 16] , 24 model quantization BID0 13, 3, 9, 18, 14] , and so on. [17, 6, 28] constrain their weights to {−1, +1} 25 or {−1, 0, 1} and achieve limited acceleration by using simple accumulation instead of complicated 26 multiplication-accumulations.

In particular, [2, 18, 4, BID10 24] quantize activation values and weights BID10 to bits and use bitwise logic operations to achieve extreme acceleration ratio in inference process but 28 they are suffering from significant performance degradation.

However, most models are proposed 29 for fixed precision, and can not extend to other precision models.

They easily fall into local optimal 30 solutions and face slow convergence speed in training process.

In order to bridge the gap between 31 low-bit and full-precision and be applied to many cases, we propose a novel encoding scheme of 32 using {−1, +1} to easily decompose trained QNNs into multi-branch binary networks.

Therefore, 33 the inference process can be efficiently implemented by bitwise operations to achieve model com-34 pression, computational acceleration and resource saving.

As the basic computation in most neural network layers, matrix multiplication costs lots of resources 37 and also is the most time consuming operation.

Modern computers store and process data in bina-38 ry format, thus non-negative integers can be directly encoded by {0, 1}. We propose a novel de- DISPLAYFORM0 All of the above operations consist of N multiplications and (N −1) additions.

Based on the above 43 encoding scheme, the vector x can be encoded to binary form using M bits, i.e., DISPLAYFORM1 Then we convert the right-hand side of (2) into the following form: DISPLAYFORM2 where DISPLAYFORM3 .

In such an encoding scheme, the number of represented states is not greater than 2 M .

In addition,

we encode another vector w with K-bit numbers in the same way.

Therefore, the dot product of the 48 two vectors can be computed as follows: DISPLAYFORM0 From the above formulas, the dot product is decomposed into M ×K sub-operations, in which each 50 element is 0 or 1.

Because of the restriction of encoding and without using the sign bit, the above 51 representation can only be used to encode non-negative integers.

However, it's impossible to limit 52 the weights and the values of the activation functions to non-negative integers.

In order to encode 53 both positive and negative integers, we propose a novel encoding scheme, which uses {-1, +1} as 54 the basic elements rather than {0, 1}. Then we can use multiple bitwise operations (i.e., xnor and 55 bitcount) to effectively achieve the above vector multiplications.

Our operation mechanism can be 56 suitable for all vector/matrix multiplications.

Besides fully connected layers, our mechanism is also 57 suitable for convolution and deconvolution layers in deep neural networks.

ing of input data.

Therefore, the dot product can be computed by the formula (6).

Without other 63 judgment and mapping calculation, we use trigonometric functions as the basic encoding functions.

In the end, we use the sign function to hard divide to -1 or +1.

The mathematical expression can be 65 formulated as follows:

In this section, we use the same network architecture described in [17, 2] for CIFAR-10 and choose DISPLAYFORM0

ResNet-18 as the basic network for ImageNet.

It is very hard to train on large-scale training sets (e.g.,

ImageNet), and thus parameter initialization is particularly important.

In particular, the well-trained 72 full-precision model parameters activated by ReLU can be directly used as initialization parameters 73 for our 8-bit quantized network.

After fine-tuning dozens of epochs, 8-bit quantized networks can be 74 well-trained.

Similarly, we use the 8-bit model parameters as the initialization parameters to train 7-75 bit quantized networks, and so on.

We use the loss computed by quantized parameters to update full 76 precision parameters described as the straight-through estimator [26] .

TAB2

@highlight

A novel encoding scheme of using {-1, +1} to decompose QNNs into multi-branch binary networks, in which we used bitwise operations (xnor and bitcount) to achieve model compression, computational acceleration and resource saving. 
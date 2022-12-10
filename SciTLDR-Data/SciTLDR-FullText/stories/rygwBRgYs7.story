Typical recent neural network designs are primarily convolutional layers, but the tricks enabling structured efficient linear layers (SELLs) have not yet been adapted to the convolutional setting.

We present a method to express the weight tensor in a convolutional layer using diagonal matrices, discrete cosine transforms (DCTs) and permutations that can be optimised using standard stochastic gradient methods.

A network composed of such structured efficient convolutional layers (SECL) outperforms existing low-rank networks and demonstrates competitive computational efficiency.

Deep neural networks have evolved, and no longer contain the gigantic linear layers seen in Si- network's parameters and multiply-add (Mult-Add) operations are used in these convolutions.

We 13 present a method to reduce both, while keeping the network structure the same.14 SELLs provide a framework to approximate linear layers that we adapt to make convolutions more 15 efficient.

A convolution can be viewed as a matrix multiplication between the extracted patches from 16 the input tensor and the weights of the filters passed over the image.

In this work, we show that this 17 matrix multiplication can be replaced with a structured efficient transformation based on the ACDC 18 layer (Moczulski et al., 2016 DISPLAYFORM0 In which, D are diagonal matrices, P are permutations, S are sparse matrices and B are bases, such If we define a function to map the patches over which a convolution passes on an input tensor to 47 rows of a matrix as F, we can express convolution using a kernel matrix W as DISPLAYFORM1

This is the common algorithm known as im2col-gemm BID5 .

From this, we define a

Structured Efficient Convolutional Layer (SECL) with the following parameterisation for W: DISPLAYFORM0 Where A and D are diagonal matrices, C and C −1 are the forward and inverse DCTs and P is a 51 riffle shuffle.

with a deck of cards BID12 .

As described in Section 4, this was found to work as well as a 56 fixed random permutation and can be evaluated much faster BID36 .

Substituting this parameterisation into convolutional layers presents a problem: most kernel matrices 58 are not square, with one exception.

Kernel matrices in pointwise convolutions are square when the 59 number of input channels matches the output.

To increase the number of channels, we repeat the 60 input along the channel dimension.

As channels commonly increase in integer steps, this allows us to 61 implement almost all pointwise convolutions.

Given a pointwise convolution, we can now implement a convolution with any kernel size by preceding 63 the pointwise with a grouped convolution.

This is known as a depthwise separable convolution and 64 has been demonstrated as a substitute for convolution BID6 .

This can also be implemented 65 using an ACDC parameterisation for each filter, but there is not much benefit, as shown in Section 4.

The motivation underlying ACDC layers comes from their complex equivalent; using Fourier trans- DISPLAYFORM0 Where D 2i−1 is a diagonal and R 2i is composed of FDF −1 .

However, machine learning systems typically operate using real numbers, leading to the decision to 4 Experiments

We demonstrate the effectiveness of this compression strategy in Section 4.1 in experiments on

The results are shown in

@highlight

It's possible to substitute the weight matrix in a convolutional layer to train it as a structured efficient layer; performing as well as low-rank decomposition.

@highlight

This work applies previous Structured Efficient Linear Layers to conv layers and proposes Structured Efficient Convolutional Layers as substitution of original conv layers.
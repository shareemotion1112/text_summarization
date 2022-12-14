The state-of-the-art (SOTA) for mixed precision training is dominated by variants of low precision floating point operations, and in particular, FP16 accumulating into FP32 Micikevicius et al. (2017).

On the other hand, while a lot of research has also happened in the domain of low and mixed-precision Integer training, these works either present results for non-SOTA networks (for instance only AlexNet for ImageNet-1K), or relatively small datasets (like CIFAR-10).

In this work, we train state-of-the-art visual understanding neural networks on the ImageNet-1K dataset, with Integer operations on General Purpose (GP) hardware.

In particular, we focus on Integer Fused-Multiply-and-Accumulate (FMA) operations which take two pairs of INT16 operands and accumulate results into an INT32 output.

We propose a shared exponent representation of tensors and develop a Dynamic Fixed Point (DFP) scheme suitable for common neural network operations.

The nuances of developing an efficient integer convolution kernel is examined, including methods to handle overflow of the INT32 accumulator.

We implement CNN training for ResNet-50, GoogLeNet-v1, VGG-16 and AlexNet; and these networks achieve or exceed SOTA accuracy within the same number of iterations as their FP32 counterparts without any change in hyper-parameters and with a 1.8X improvement in end-to-end training throughput.

To the best of our knowledge these results represent the first INT16 training results on GP hardware for ImageNet-1K dataset using SOTA CNNs and achieve highest reported accuracy using half precision

While single precision floating point (FP32) representation has been the mainstay for deep learning training, half-precision and sub-half-precision arithmetic has recently captured interest of the academic and industrial research community.

Primarily this interest stems from the ability to attain potentially upto 2X or more speedup of training as compared to FP32, when using half-precision fused-multiply and accumulate operations.

For instance NVIDIA Volta NVIDIA (2017) provides 8X more half-precision Flops as compared to FP32.Unlike single precision floating point, which is a unanimous choice for 32b training, half-precision training can either use half-precision floating point (FP16), or integers (INT16).

These two options offer varying degrees of precision and range; with INT16 having higher precision but lower dynamic range as compared to FP16.

This also leads to residues between half-precision representation and single precision to be fundamentally different -with integer representations contributing lower residual errors for larger (and possibly more important) elements of a tensor.

Beyond this first order distinction in data types, there are multiple algorithmic and semantic differences (for example FP16 multiply-and-accumulate operation accumulating into FP32 results) for each of these data types.

Hence, when discussing half-precision training, the whole gamut of tensor representation, semantics of multiply-and-accumulate operation, down-conversion scheme (if the accumulation is to a higher precision), scaling and normalization techniques, and overflow management methods must be considered in totality to achieve SOTA accuracy.

Indeed, unless the right combination of the aforesaid vectors are selected, half precision training is likely to fail.

Conversely, drawing conclusions on the efficacy of a method by not selecting all vectors properly can lead to inaccurate conclusions.

In this work we describe a mixed-precision training setup which uses:??? INT16 tensors with shared tensor-wide exponent, with a potential to extend to sub-tensor wide exponents.??? An instruction which multiplies two INT16 numbers and stores the output into a INT32 accumulator.??? A down-convert scheme based on the maximum value of the output tensor in the current iteration using multiple rounding methods like nearest, stochastic, and biased rounding.??? An overflow management scheme which accumulates partial INT32 results into FP32, along with trading off input precision with length of accumulate chain to gain performance.

The compute for neural network training is dominated by GEMM-like, convolution, or dot-product operations.

These are amenable to speedup via specialized low-precision instructions for fusedmultiply-and-accumulate (FMA), like AVX512_4VNNI 1 .

However, this does not necessarily mean using half-precision representation for all tensors, or using only half-precision operations.

In fact, performance speedups by migrating the compute intensive operations in both forward and back prorogation (FPROP, BPROP and WTGRAD) is often close to the maximum achievable speedup obtained by replacing all operations (for instance SGD) in half-precision.

In cases where it is not, performance degradation typically happens due to limitations of memory bandwidth, and other architectural reasons.

Hence on a balanced general purpose machine, a mixed-precision strategy of keeping precision critical operations (like SGD and some normalizations) in single precision and compute intensive operations in half precision can be employed.

The proposed integer-16 based mixed-precision training follows this template.

Using the aforesaid method, we train multiple visual understanding CNNs and achieve Top-1 accuracies BID16 on the ImageNet-1K dataset BID2 which match or exceed single precision results.

These results are obtained without changing any hyper-parameters, and in as many iterations as the baseline FP32 training.

We achieve 75.77% Top-1 accuracy for ResNet-50 which, to the best of our knowledge, significantly exceeds any result published for halfprecision training, for example ; .

Further, we also demonstrate our methodology achieves state-of-the-art accuracy (comparable to FP32 baseline) with int16 training on GoogLeNet-v1, VGG-16 and AlexNet networks.

To the best of our knowledge, these are first such results using int16 training.

The rest of the paper is organized as follows: Section 2 discusses the literature pertaining to various aspects of half-precision training.

The dynamic fixed point format for representing half-precision tensors is described in Section 3.

Dynamic fixed point kernels and neural network training operations are described in Section 4, and experimental results are presented in Section 5.

Finally, we conclude this work in Section 6.

Using reduced precision for Deep learning has been an active topic of research.

As a result there are a number of different reduced precision data representations, the more standard floating-point based ; Dettmers (2015) and custom fixed point schemes BID19 BID0 ; BID5 ; BID8 ; BID10 .The recently published mixed precision training work from uses 16-bit floating point storage for activations, weights and gradients.

The forward, back propagation computation uses FP16 computation with results accumulating into FP32 and a master-copy of the full precision (FP32) weights are retained for the update operation.

They demonstrate a broad variety of deep learning training applications involving deep networks and larger data-sets (ILSVRC-class problems) with minimal loss compared to baseline FP32 results.

Further, this shows that FP16/FP32 mixed precision requires loss scaling to achieve near-SOTA accuracy.

This ensures back-propagated gradient values are shifted into FP16 representable range and the small magnitude (negative exponent) values, which are critical for accuracy are captured.

Such scaling is inherent with fixed point representations, making it more amenable and accurate for deep learning training.

Custom fixed point representations offer more flexibility -in terms of both increased precision and dynamic range.

This allows for better mapping of the representation to the underlying application, thus making it more robust and accurate than floating-point based schemes.

BID19 have shown that the dynamically scaled fixed point representation proposed by BID20 can be very effective for convolution neural networks -demonstrating upto to 4?? improvement over an aggressively tuned floating point implementation on general purpose CPU hardware.

BID5 have done a comprehensive study on the effect of low precision fixed point computation for deep learning and have successfully trained smaller networks using 16-bit fixed point on specialized hardware.

With further reduced bit-widths, such fixed point data representations are more attractive -offering increased capacity for precision with larger mantissa bits and dynamically scaled shared exponents.

There have been several publications with <16-bit precision and almost all of them use such custom fixed point schemes.

BID0 use a dynamical fixed point format (DFXP), with low precision multiplications with upto 12-bit operations.

Building on this BID1 proposed training with only binary weights while all other tensors and operations are in full precision.

BID7 further extended this to use binary activations as well, but with gradients and weights still retained in full precision.

BID8 proposed training with activations and weights quantized up to 6-bits and gradients in full precision.

BID15 use binary representation for all components including gradients.

However, all of the aforementioned use smaller benchmark model/data-sets and results in a non-trivial drop in accuracy with larger ImageNet data-set BID2 and classification task BID16 .

BID10 have shown that a fixed point numerical format designed for deep neural networks (Flexpoint), out-performs FP16 and achieves numerical parity with FP32 across a wide set of applications.

However, this is designed specifically for specialized hardware and the published results are with software emulation.

Here we propose a more general dynamic fixed point representation and associated compute primitives, which can leverage general purpose hardware using the integercompute pipeline.

Further we provide actual accuracy and performance for training large networks for the ILSVRC classification task, measured on available hardware.

Dynamic Fixed Point (DFP) tensors are represented by a combination of an integer tensor I and an exponent E s , shared across all the integer elements.

For the sake of convenience, the DFP tensor can be denoted as DFP-P = I, E s , where P represents the number of bits used by the integer elements in I (ex: DFP-16 contains 16-bit integers).

FIG0 illustrates the differences in data representation between IEEE-754 standard format float, half-float and DFP-16 data format.

DFP-16 data type offers a trade-off between float and half-float in terms of precision and dynamic range.

When compared to full-precision floats, DFP-16 can achieve higher compute density and can carry higher effective precision compared to half-floats because of larger 15-bit mantissa (compared to 11-bits for half-floats).

Further, the effective dynamic range of DFP format can be increased by extending the data type to use Blocked-DFP representation.

Blocked-DFP uses fine-grained quantization to assign multiple exponents per tensor with smaller blocks of integers sharing a common exponent.

BID12 have demonstrated effectiveness of fine-grained quantization for low-precision inference tasks.

In this work, we use a single shared exponent for each tensor.

The integers are stored in 2's complement representation and the shared exponent is an 8-bit signed integer.

We use standard commodity integer hardware to perform arithmetic operations on DFP tensors.

This implies that the exponent handling and precision management of DFP is done in the software, which is covered in more detail in Section 4.3.

To facilitate end-to-end mixed-precision training using DFP, we have created primitives to perform arithmetic operations on DFP tensors and data conversions between DFP and float.

When converting floating point tensors into to DFP data type, the shared exponent is derived from the exponent of absolute maximum value of the floating point tensor.

If F is the floating point tensor, the exponent of the absolute maximum value is expressed as follows.

DISPLAYFORM0 The value of the shared exponent E s is a function of E f max and the number of bits P used by the output integer tensor I. DISPLAYFORM1 The relationship of the resulting DFP tensor I, E s with the input floating point tensor F is expressed by Eq.3.???i n ??? I, f n = i n ?? 2 Es , wheref n ??? F (3) Extending this basic formulation Eq.3, we can define a set of common DFP primitives required for neural network training.??? Multiplying two DFP-16 tensors produces 32-bit I tensor with a new shared exponent expressed as follows.

DISPLAYFORM2 ??? Adding two DFP-16 tensors results in a 32-bit I tensor and a new shared exponent.

DISPLAYFORM3 Note that when a Fused Multiply and Add operation is performed, all products have the same shared exponent: DISPLAYFORM4 s , and hence the sum of such products also has the same shared exponent.??? Down-Conversion scales DFP-32 output of a layer to DFP-16 to be passed as input to the next layer.

The 32-bit I tensor right-shifted R s bits to fit into 16-bit tensor.

The R s value and the new shared exponent are expressed as follows.

DISPLAYFORM5 In Eqn.6, A is accumulator bit-width, LZC( ) returns the leading zero bit-count.

Neural network training is an iterative process over mini-batches of data points, with four main operations on a given mini-batch: forward propagation (FPROP), back-propagation (BPROP), weight gradient computation (WTGRAD), and the solver (typically stochastic gradient descent, or ADAM).In a CNN, the three steps of forward-propagation, back-propagation, and weight-gradient computation are often the compute intensive steps, and consist of GEMM-like (General Matrix Multiply) convolution operations which dominate the compute, and additional element-wise operations like normalization, non-linear (ReLU) and element-wise addition.

In this work we propse a method to use INT16 operations, for implementing kernels for the convolutions and GEMM.

There kernels are stitched with the rest of the operations in neural network training via Dynamic Fixed Point to floating point conversions described earlier in Section 3.

In this section, we first describe the overall method for using dynamic fixed point in neural network training, and then explain the optimized kernel for convolutions.

The mixed precision training scheme used in this work is described in FIG2 .

The core compute kernels in this scheme are the FP, BP, and WU convolution functions which take two DFP-16 tensors as input and produces a FP32 tensor as output.

For example FP accepts two DFP-16 tensors, a q , and w q (activations and weights for layer-l), and produces a FP32 output tensorThe FP and BP operations are followed by quantization steps (Q a , Q e ) which convert the FP32 tensors to DFP-16 tensors (?? l q , e l q ) for operations in the next layer.

The WU step is followed by the Stochastic Gradient Descent (SGD) step, which takes the FP32 tensor for weight-gradients (???w) and a FP32 copy of the weights (W l ) as inputs, and produces an updated weight tensor as output.

We follow the now established practice of keeping a FP32 copy of weights as well as a low precision (DFP-16) copy of weights.

Therefore SGD or other solvers are FP32 operations.

In case a batch-norm layer is used, the DFP-16 tensors are loaded into registers and then the data is up-converted to FP32 to prevent overflows during stats computation.

In this section we delve into efficient implementations of core compute kernels written using Integer FMA instruction sequence; in particular the AVX512_4VNNI instruction (described in Algorithm 1).

This instruction takes a memory pointer as the first input and four vector registers as the second input and performs 8 multiply-add operations per output (16 Integer-OPs).

For each 32b lane, the instruction takes two pairs of 16-bit Integers, performs a multiply followed by a horizontal add.

The FPROP convolution kernel is written using AVX512_4VNNI instruction in Algorithm 2.

The data layout of the weights captures the 2-way horizontal accumulation operation in AVX512_4VNNI.

Here the last dimension moves along consecutive input-feature maps.

Hence the dimensions of activations is: N, C/16, H, W, 16, and that of weights is C/16, K/16, KH, KW, 8c, 16k, 2c (where C and K are input and output feature maps, H, W are input feature map height and width, and KH, KW are kernel height/width).

Note that while we briefly touch upon data layout and blocking of the core kernel loops in Algorithm 2, detailed analysis of performance is not the objective of this work.

These details are explored only to highlight different functional components of the kernel.

Multiplication of two INT16 numbers can result in a 30-bit outcome, and hence an accumulate chain of 3 products of INT16 multiplicative pairs can cause an overflow of the INT32 accumulator.

In neural network training, accumulate chains can exceed a million in length (for example in the WTGRAD kernel).One way to prevent overflows is to convert an INT32 intermediate output into FP32 before accumulation as described in lines 26-31 in Algorithm 2.

Here we first convert the INT32 result to FP32 using the VCVTINTFP32 instruction, followed by a scale and accumulate into the final FP32 result using the VFP32MADD instruction.

The scale used is 2 (Einp+Ewt) (equation 3), which is broadcast and stored in the vscale vector register.

The instruction sequence in lines 26-31 can be applied after every AVX512_4VNNI instruction to prevent almost all overflows.

However the overheads would be significant and hurt performance.

Hence we pick the strategy of partial accumulations into INT32 for short accumulate chains, and subsequently converting the results into FP32.Performance Impact: As outlined in Algorithm2 for performance we block additionally over the input feature maps (ICBLK) and use optimal register blocking (RB_SIZE).

The difference between an ideal instruction sequence (with no overflow management) and Algorithm 2 is essentially the additional VCVTINTFP32 instruction (line 28).

In the loop in lines 8-31, we have (ICBLK/16)*KH*KW*2*RB AVX512_4VNNI instructions, and RB*4 + (ICBLK/16)*KH*KW*4 non-AVX512_4VNNI instructions, and RB VCVTINTFP32 instructions.

The instruction overhead from overflow management therefore varies between <1% in most cases, to at most 3%.The length of the accumulate chain (via sizing the input feature map blocking factor ICBLK in line 7) is selected to optimize instruction overheads and cache/instruction reuse.

In this work we strive to keep the accumulate chain to more than 200 (which is empirically shown to be close to optimal).

Often this accumulate chain also overflows, which we circumvent by shifting inputs.

In this work, we shift both the inputs by 1-bit for all convolutions in all experiments.

Hence effectively we have a DFP15 representation of all DFP tensors.

It is notable that this shift value is largely dependent on this

We trained several CNNs for the ImageNet-1K classification task using mixed precision DFP16: AlexNet BID11 , VGG-16 Simonyan & Zisserman (2014) , GoogLeNet-v1 Szegedy et al. (2015) , ResNet-50 He et al. (2016) .

We use exactly the same batch-size and hyper-parameter configuration for both the baseline FP32 and DFP16 training runs TAB1 1).

In both cases, the models are trained from scratch using synchronous SGD on multiple nodes.

In our experiments the first convolution layer (C1) and the fully connected layers are in FP32 (constituting about 5 ??? 10% of compute for modern CNNs).

TAB1 1 shows ImageNet-1K classification accuracies, training with DFP16 achieve SOTA accuracy for all four models and in several cases even better than the baseline full precision result.

To the best of our knowledge, top-1 accuracy of 75.77% and top-5 accuracy of 92.84% for ResNet-50 with mixed precision DFP16 -is highest achieved accuracy on the ImageNet-1K classification task with any form of reduced precision training.

It can be seen from Figure.

3 that DFP16, closely tracks the full precision training.

For some models like GoogLeNet-v1 and AlexNet, we observe the initially DFP16 training lags the baseline, however this gaps is closed with subsequent epochs especially after the learning rate changes.

Further, we observe that compared to baseline run -with DFP16 the validation/test loss tracks much closer to the training loss.

We believe this is the effect of the additional noise introduced from reduced precision computation/storage, which is results in better generalization with reduced training-testing gap and better accuracies.

For the convolution kernels going from FP32 to DFP16, the 3 ?? 3 kernels are 1.8?? faster and the 1??1 kernels are 1.4?? faster; resulting in overall 1.5?? speedup.

The baseline kernels include memory prefetch optimization, which when applied to DFP kernels should improve the performance by an additional 20%.

The batchnorm computation is 2?? faster with DFP16, the speed up here is primarily due to 50% bandwidth saving due to smaller memory footprint.

In addition, the ReLU and EltWise layers are fused with batchnorm ( FIG4 ) to avoid additional memory passes over the activation tensor.

This fusion technique is orthogonal to mixed precision DFP16 training and can also be applied to baseline FP32 version as well, however its more relevant mixed precision DFP16 training due to faster compute.

Furthermore, such memory bandwidth optimizations are becoming more critical with the growing disparity between compute capabilities and memory bandwidth with advent of specialized compute accelerators.

With the above optimizations, we achieve an overall training throughput of 276 images/sec and 1.8X speed up over FP32 for ResNet-50.

Additionally, we have improved SGD computation by 3?? over the standard implementation in Intel-Caffe, pushing the training throughput to 317 images/sec, shown as the framework overhead reduction in Figure.

4.

When exlpoiting similar tuning knobs, such as fusion and improved SGD, in case the of the baseline FP32 version its performance increases to 194 images/sec.

Even in this case Mixed Precision DFP16 can yield a high speedup of 1.6X with respect to time-to-train.

We demonstrate industry-first reduced precision INT-based training result on large networks/data-sets.

Showing on-par or better than FP32 baseline accuracies and potentially 2?? savings in computation, communication and storage.

Further, we propose a general dynamic fixed point representation scheme, with associated compute primitives and algorithm for the shared exponent management.

This DFP solution can be used with general purpose hardware, leveraging the integer compute pipeline.

We demonstrate this with implementation of CNN training for ResNet-50, GoogLeNet-v1, VGG-16 and AlexNet; training these networks with mixed precision DFP16 for the ImageNet-1K classification task.

While this work focuses on visual understanding CNNs, in future we plan to demonstrate the efficacy of this method for other types of networks like RNNs, LSTMs, GANs and extend this to wider set of applications.

@highlight

Mixed precision training pipeline using 16-bit integers on general purpose HW;  SOTA accuracy for ImageNet-class CNNs; Best reported accuracy for ImageNet-1K classification task with any reduced precision training;

@highlight

This paper shows that a careful implementation of mixed-precision dynamic fixed point computation can achieve state of the art accuracy using a reduced precision deep learning model with a 16 bit integer representation

@highlight

Proposes a "dynamic fixed point" scheme that shares the exponent part for a tensor and develops procedures to do NN computing with this format and demonstrates this for limited precision training.
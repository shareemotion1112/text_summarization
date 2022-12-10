Researches on deep neural networks with discrete parameters and their deployment in embedded systems have been active and promising topics.

Although previous works have successfully reduced precision in inference, transferring both training and inference processes to low-bitwidth integers has not been demonstrated simultaneously.

In this work, we develop a new method termed as ``"WAGE" to discretize both training and inference, where weights (W), activations (A), gradients (G) and errors (E) among layers are shifted and linearly constrained to low-bitwidth integers.

To perform pure discrete dataflow for fixed-point devices, we further replace batch normalization by a constant scaling layer and simplify other components that are arduous for integer implementation.

Improved accuracies can be obtained on multiple datasets, which indicates that WAGE somehow acts as a type of regularization.

Empirically, we demonstrate the potential to deploy training in hardware systems such as integer-based deep learning accelerators and neuromorphic chips with comparable accuracy and higher energy efficiency, which is crucial to future AI applications in variable scenarios with transfer and continual learning demands.

Recently deep neural networks (DNNs) are being widely used for numerous AI applications BID11 BID21 .

Depending on the massive tunable parameters, DNNs are considered to have powerful multi-level feature extraction and representation abilities.

However, training DNNs needs energy-intensive devices such as GPU and CPU with high precision (float32) processing units and abundant memory, which has greatly challenged their extensive applications for portable devices.

In addition, a state-of-art network often has far more weights and effective capacity to shatter all training samples , leading to overfitting easily.

As a result, there is much interest in reducing the size of network during inference BID8 BID17 BID14 , as well as dedicated hardware for commercial solutions BID10 BID20 .

Due to the accumulation in stochastic gradient descent (SGD) optimization, the precision demand for training is usually higher than inference BID8 .

Therefore, most of the existing techniques only focus on the deployment of a well-trained compressed network, while still keeping high precision and computational complexity during training.

In this work, we address this problem as how to process both training and inference with low-bitwidth integers, which is essential for implementing DNNs in dedicated hardware.

To this end, two fundamental issues are addressed for discretely training DNNs: i) how to quantize all the operands and operations, and ii) how many bits or states are needed for SGD computation and accumulation.

With respect to the issues, we propose a framework termed as "WAGE" that constrains weights (W), activations (A), gradients (G) and errors (E) among all layers to low-bitwidth integers in both training and inference.

Firstly, for operands, linear mapping and orientation-preserved shifting are applied to achieve ternary weights, 8-bit integers for activations and gradients accumulation.

Secondly, for operations, batch normalization BID9 is replaced by a constant scaling factor.

Other techniques for fine-tuning such as SGD optimizer with momentum and L2 regularization are simplified or abandoned with little performance degradation.

Considering the overall bidirectional propagation, we completely streamline inference into accumulate-compare cycles and training into low-bitwidth multiply-accumulate (MAC) cycles with alignment operations, respectively.

We heuristically explore the bitwidth requirements of integers for error computation and gradient accumulation, which have rarely been discussed in previous works.

Experiments indicate that it is the relative values (orientations) rather than absolute values (orders of magnitude) in error that guides previous layers to converge.

Moreover, small values have negligible effects on previous orientations though propagated layer by layer, which can be partially discarded in quantization.

We leverage these phenomena and use an orientation-preserved shifting operation to constrain errors.

As for the gradient accumulation, though weights are quantized to ternary values in inference, a relatively higher bitwidth is indispensable to store and accumulate gradient updates.

The proposed framework is evaluated on MNIST, CIFAR10, SVHN, ImageNet datasets.

Comparing to those who only discretize weights and activations at inference time, it has comparable accuracy and can further alleviate overfitting, indicating some type of regularization.

WAGE produces pure bidirectional low-precision integer dataflow for DNNs, which can be applied for training and inference in dedicated hardware neatly.

We publish the code on GitHub 1 .

We mainly focus on reducing precision of operands and operations in both training and inference.

Orthogonal and complementary techniques for reducing complexity like network compression, pruning BID4 BID27 and compact architectures BID7 are impressively efficient but outside the scope this paper.

Weight and activation BID2 ; BID8 propose methods to train DNNs with binary weights (BC) and activations (BNN) successively.

They add noises to weights and activations as a form of regularization but real-valued gradients are accumulated in real-valued variables, suggesting that high precision accumulation is likely required for SGD optimization.

XNOR-Net BID17 ) has a filter-wise scaling factor for weights to improve the performance.

Convolutions in XNOR-Net can be implemented efficiently using XNOR logical units and bit-count operations.

However, these floating-point factors are calculated simultaneously during training, which generally aggravates the training effort.

In TWN BID14 and TTQ BID29 two symmetric thresholds are introduced to constrain the weights to be ternary-valued: {+1, 0, −1}. They claimed a tradeoff between model complexity and expressive ability.

Gradient computation and accumulation DoReFa-Net BID28 ) quantizes gradients to low-bitwidth floating-point numbers with discrete states in the backward pass.

TernGrad quantizes gradient updates to ternary values to reduce the overhead of gradient synchronization in distributed training.

Nevertheless, weights in DoReFa-Net and TernGrad are stored and updated with float32 during training like previous works.

Besides, the quantization of batch normalization and its derivative is ignored.

Thus, the overall computation graph for the training process is still presented with float32 and more complex with external quantization.

Generally, it is difficult to apply DoReFa-Net training in an integer-based hardware directly, but it shows potential for exploring high-dimensional discrete spaces with discrete gradient descent directions.

The main idea of WAGE quantization is to constrain four operands to low-bitwidth integers: weight W and activation a in inference, error e and gradient g in backpropagation training, see Figure 1 .

We extend the original definition of errors to multi-layer: error e is the gradient of activation a for the perspective of each convolution or fully-connected layer, while gradient g particularly refers to the gradient accumulation of weight W .

Considering the i-th layer of a feed-forward network, wehave: DISPLAYFORM0 where L is the loss function.

We separate these two terms that are mixed up in most existing schemes.

The gradient of weight g and the gradient of activation e flow to different paths in each layer, which is a fork both in inference and in backward training and generally acts as node of MAC operations.

For the forward propagation in the i-th layer, assuming that weights are stored and accumulated with k G -bit integers, then numerous works strive for a better quantization function Q W (·) that maps higher precision weights to their k W -bit reflections, for example, [−0.9, 0.1, 0.7] to [−1, 0, 1].

Although weights are accumulated with high precision like float32, the deployment of the reflections in dedicated hardware are much more memory efficient after training.

Activations are quantized with function Q A (·) to k A bits to align the increased bitwidth caused by MACs.

Weights and activations are discretized to even binary values in previous works, then MACs degrade into logical and bit-count operations that are extremely efficient BID17 .For the backward propagation in the i-th layer, the gradients of activations and weights are calculated by the derivatives of MACs that are generally considered to be in 16-bit floating-point precision at least.

As illustrated in Figure 1 , the MACs between k A -bit inputs and k W -bit weights will increase the bitwidth of outputs to [k A + k W − 1] in signed integer representation, and the similar broadening happens to errors e as well.

In consideration of training with only low-bitwidth integers, we propose additional functions Q E (·) and Q G (·) to constrain bitwidth of e and g to k E bits and k G bits, respectively.

In general, where there is a MAC operation, there are quantization operators named DISPLAYFORM1 and Q E (·) in inference and backpropagation.

DISPLAYFORM2 Forward the -th layer Backward the -th layer DISPLAYFORM3 : operand : operation : dataflow DISPLAYFORM4 [ ]: bit width DISPLAYFORM5 , Q E (·) added in WAGE computation dataflow to reduce precision, bitwidth of signed integers are below or on the right of arrows, activations are included in MAC for concision.

In WAGE quantization, we adopt a linear mapping with k-bit integers for simplicity, where continuous and unbounded values are discretized with uniform distance σ: DISPLAYFORM0 Then the basic quantization function that converts a floating-point number x to its k-bitwidth signed integer representation can be formulated as: DISPLAYFORM1 where round approximates continuous values to their nearest discrete states.

Clip is the saturation function that clips unbounded values to [−1 + σ, 1 − σ], where the negative maximum value −1 is removed to maintain symmetry.

For example, Q(x, 2) quantizes {−1, 0.2, 0.6} to {−0.5, 0, 0.5}. Equation 3 is merely used for simulation in floating-point hardware like GPU, whereas in a fixedpoint device, quantization and saturation is satisfied automatically.

Before applying linear mapping in some operands (e.g., error), we introduce an additional monolithic scaling factor for shifting values distribution to an appropriate order of magnitude, otherwise values will be all saturated or cleared by Equation 3.

The scaling factor is calculated by Shif t function and then divided in later steps: DISPLAYFORM2 (4) Finally, we propose stochastic rounding to substitute small and real-valued updates for gradient accumulation in training.

Section 3.3.4 will detail the implementation of operator Q G (·), where high bitwidth gradients are constrained to k G -bit integers stochastically by a 16-bit random number generator.

Figure 2 summarizes quantization methods used in WAGE.

Stochastic Rounding DISPLAYFORM0

Figure 2: Quantization methods used in WAGE.

The notation P , x, · and · denotes probability, vector, f loor and ceil, respectively.

Shif t(·) refers to Equation 4 with a certain argument.

In previous works, weights are binarized directly by sgn function or ternarized by threshold parameters calculated during training.

However, BNN fails to converge without batch normalization because weight values ±1 are rather big for a typical DNN.

Batch normalization not only efficiently avoids the problem of exploding and vanishing gradients, but also alleviates the demand for proper initialization.

However, normalizing outputs for each layer and computing their gradients are quite complex without floating point unit (FPU).

Besides, the moving averages of batch outputs occupy external memory.

BNN shows a shift-based variation of batch normalization but it is hard to transform all of the elements to the fixed-point representations.

As a result, weights should be cautiously initialized in this work where batch normalization is simplified to a constant scaling layer.

A modified initialization method based on MSRA BID5 can be formulated as: DISPLAYFORM0 where n in is the layer fan-in number, and the original limit 6/n in in MSRA is calculated to keep same variance between inputs and outputs of the same layer theoretically.

The additional limit L min is a minimum value that the uniform distribution U should reach, and β is a constant greater than 1 to create overlaps between minimum step size σ and maximum value L. In case of k W -bit linear mapping, if weights W are quantized directly with original limits, we will get all-zero tensors when bitwidth k W is small enough, e.g., 4, or fan-in n in is wide enough, where initialized weights may never reach the minimum step σ presented by fixed-point integers.

So L min ensures that weights can go beyond σ and quantized to non-zero values after Q W (·) when initialized randomly.

DISPLAYFORM1 The modified initialization in Equation 5 will amplify weights holistically and guarantee their proper distribution, then W is quantized directly with Equation 3: DISPLAYFORM2 It should be noted that the variance of weights is scaled compared to the original limit, which will cause exploding of network's outputs.

To alleviate the amplification effect, XNOR-Net proposed a filter-wise scaling factor calculated continuously with full precision.

In consideration of integer implementation, we introduce a layer-wise shift-based scaling factor α to attenuate the amplification effect: DISPLAYFORM3 where α is a pre-defined constant for each layer determined by the network structure.

The modified initialization and attenuation factor α together approximates floating-point weights to their integer representations, except that α takes effect after activations to maintain precision of weights presented by k W -bit integers.

As stated above, the bitwidth of operands increases after MACs.

Then a typical CNN is usually followed with pooling, normalization and activation.

Average pooling is avoided because mean operations will increase precision demand.

Besides, we hypothesize that batch outputs of each hidden layer approximately have zero-mean, then batch normalization degenerates into to a scaling layer where trainable and batch-calculated scaling parameters are replaced by α mentioned in Equation FORMULA13 .

If activations are presented in k A bits, the overall quantization of activations can be formulated as: DISPLAYFORM0

Errors e are calculated layer by layer using the chain rule during training.

Although the computation graph of backpropagation is similar to the inference, the inputs are the gradients of L, which are relatively small compared to actual inputs for networks.

More importantly, the errors are unbounded and might have significantly larger ranges than that of activations, e.g., [10 −9 , 10 −4 ].

DoReFa-Net first applies an affine transform on e to map them into [−1, 1], and then inverts the transform after quantization.

Thus, the quantized e are still presented as float32 numbers with discrete states and mostly small values.

However, experiments uncover that it is the orientations rather than orders of magnitude in errors that guides previous layers to converge, then the inverse transformation after quantization in DoReFa-Net is no longer needed.

The orientation-only preservation prompts us to propagate errors with integer thoroughly, where error distribution is firstly scaled into [− √ 2, + √ 2] by dividing a shift factor as shown in Figure 2 and then quantized by Q(e, k E ): DISPLAYFORM0 where max{|e|} extracts the layer-wise maximum absolute value among all elements in error e, multi-channel for convolution and multi-sample for batch training.

The quantization of error discards large proportion of values smaller than σ, we will discuss the influence on accuracy later.

Since we only preserve relative values of error after shifting, the gradient updates g derived from MACs between backward errors e and forward activations a are shifted consequently.

We first rescale gradients g with another scaling factor and then bring in shift-based learning rate η: DISPLAYFORM0 where η is an integer power of 2.

The shifted gradients g s represent for minimum step numbers and directions for updating weights.

If weights are stored with k G -bit numbers, the minimum step of modification will be ±1 for integers and ±σ(k G ) for floating-point values, respectively.

The implement of learning rate η here is quite different from that in a vanilla DNN based on float32.

In WAGE, there only remain directions for weights to change and the step sizes are integer multiples of minimum step σ.

Shifted gradients g s may get greater than 1 if η is 2 or bigger to accelerate training at the beginning, or smaller than 0.5 during latter half of training when learning rate decay is usually applied.

As illustrated in Figure 2 , to substitute accumulation of small gradients in latter case, we separate g s into integer parts and decimal parts, then use a 16-bit random number generator to constrain high bitwidth g s to k G -bit integers stochastically: DISPLAYFORM1 where Bernoulli BID28 stochastically samples decimal parts to either 0 or 1.

With proper setting of k G , quantization of gradients will restrict the minimum step size, which may avoid local minimum and overfitting.

Furthermore, the gradients will be ternary values when η is not greater than 1, which reduces communication costs for distributed training .

At last, weights W might exceed the range [−1 + σ, 1 − σ] presented by k G -bit integers after updating with discrete increments ∆W .

So Clip function is indispensable to saturate and make sure there are only 2 k G −1 − 1 states for weights accumulation.

In case of the t-th iteration, we have: DISPLAYFORM2

From the above, we have illustrated our quantization methods for weights, activations, gradients and errors.

See Algorithm 1 for the detailed computation graph.

There remain some issues to specify in an overall training process with only integers.

Gradient descent optimizer like Momentum, RMSProp and Adam contains at least one copy of gradient updates ∆W or their moving average, doubling memory consumption for weights during training, which is partially equivalent to use bigger k G .

Since the weight updates ∆W are quantized to integer multiple of σ and scaled by η, we adopt pure mini-batch SGD without any form of momentum or adaptive learning rate to show the potential of reducing storage demands.

Although L2 regularization works quite well for many large-scale DNNs where overfitting occurs commonly, WAGE removes small values in Equation 3 and introduces randomness in Equation 11, acting as certain types of regularization and can get comparable accuracy in later experiments.

Thus, we remain L2 weight decay and dropout as supplementary regularization methods.

The Softmax layer and cross-entropy criterion are widely adopted in classification tasks but the calculation of e x can hardly be applied in low-bitwidth linear mapping occasions.

For tasks with small number of categories, we avoid Softmax layer and apply mean-square-error criterion but omit mean operation to form a sum-square-error (SSE) criterion since shifted errors will get the same values in Equation 9.

In this section, we set W-A-G-E bits to 2-8-8-8 as default for all layers in a CNN or MLP.

The bitwidth k W is 2 for ternary weights, which implies that there are no multiplications during inference.

Constant parameter β is 1.5 to make equal probabilities for ternary weights when initialized randomly.

Activations and errors should be of the same bitwidth since computation graph of backpropagation is similar to inference and might be applied in the same partition of hardware or memristor array BID19 .

Although XNOR-Net achieves 1-bit activations, reducing errors to 4 or less bits dramatically degenerates accuracies in our tests, so the bitwidth k A and k E are increased to 8 simultaneously.

Weights are stored with 8-bit integers during training and ternarized by two constant symmetrical thresholds during inference.

We first build the computation graph for a vanilla network, then insert quantization nodes in forward propagation and override gradients in backward propagation for each layer on Tensorflow BID0 .

Our method is evaluated on MNIST, SVHN, CIFAR10 and ILSVRC12 BID18 and TAB0 shows the comparison results.

MNIST: A variation of LeNet-5 (LeCun et al., 1998 ) with 32C5-MP2-64C5-MP2-512FC-10SSE is adopted.

The input grayscale images are regarded as activations and quantized by Equation 8 where α equals to 1.

The learning rate η in WAGE remains as 1 for the whole 100 epochs.

We report average accuracy of 10 runs on the test set.

We use a VGG-like network BID22 with 2×(128C3)-MP2-2×(256C3)-MP2-2×(512C3)-MP2-1024FC-10SSE.

For CIFAR10 dataset, we follow the data augmentation in BID13 for training: 4 pixels are padded on each side, and a 32×32 patch is randomly cropped from the padded image or its horizontal flip.

For testing, only single view of the original 32×32 image is evaluated.

The model is trained with mini-batch size of 128 and totally 300 epochs.

Learning rate η is set to 8 and divided by 8 at epoch 200 and epoch 250.

The original images are scaled and biased to the range of [−1, +1] for 8-bit integer activation representation.

As for SVHN dataset, we leave out randomly flip augmentation and reduce training epochs to 40 since it is a rather big dataset.

The error rate is evaluated in the same way as MNIST.ImageNet: WAGE framework is evaluated on ILSVRC12 dataset with AlexNet BID11 model but removes dropout and local response normalization layers.

Images are firstly resized to 256×256 then randomly cropped to 224×224 and horizontally flipped, followed by bias subtraction as CIFAR10.

For testing, the single center crop in validation set is evaluated.

Since ImageNet task is much difficult than CIFAR10 and has 1000 categories, it is hard to converge when applying SSE or hinge loss criterion in WAGE, so we add Softmax and remove quantizations in the last layer for fear of severe accuracy drop BID24 .

The model is trained with mini-batch size of 256 and totally 70 epochs.

Learning rate η is set to 4 and divided by 8 at epoch 60 and epoch 65.

We further compare WAGE variations and a vanilla CNN on CIFAR10.

The vanilla CNN has the same VGG-like architecture described above except that none quantization of any operand or operation is applied.

We add batch normalization in each layer and Softmax for the last layer, replace SSE with cross-entropy criterion, and then use a L2 weight decay of 1e-4 and momentum of 0.9 for training.

The learning rate is set to 0.1 and divided by 10 at epoch 200 and epoch 250.

For variations of WAGE, pattern 28ff has no quantization nodes in backpropagation.

Although the 28ff pattern has the same optimizer and learning rate annealing method as the vanilla pattern, we find that weight updates are decreased by the rescale factor α in Equation 7.

Therefore, the learning rate for 28ff is amplified and tuned, which reduces the error rate by 3%.

Figure 3 shows the training curves of three counterparts.

It can be seen that the 2888 pattern has comparable convergence rate to the vanilla CNN, better accuracy than those who only discretize weights and activations in inference time, though slightly more volatile.

The discretization of backpropagation somehow acts as another type of regularization and have significant error rate drop when decreasing learning rate η.

Figure 3: Training curves of WAGE variations and a vanilla CNN on CIFAR10.

The bitwidth k E is set to 8 as default in previous experiments.

To further explore a proper bitwidth and its truncated boundary, we firstly export errors from vanilla CNN for CIFAR10 after 100 training epochs.

The histogram of errors in the last convolution layer among 128 mini-batch data is shown in FIG2 .

It is obvious that errors approximately obey logarithmic normal distribution where values are relatively small and have significantly large range.

When quantized with k E -bit integers, a proper window function should be chosen to truncate the distribution while retaining the approximate orientations for backpropagation.

For more details about the layerwise histograms of all W, A, G, E operands, see Figure 5 .Firstly, the upper (right) boundary is immobilized to the maximum absolute value among all elements in errors as described in Equation 9.

Then the left boundary will be based on the bitwidth k E .

We conduct a series of experiments for k E ranging from 4 to 15.

The boxplot in FIG2 indicates that 4-8 bits of errors represented by integers are enough for CIFAR10 classification task.

Bitwidth 8 is chosen as default to match the 8-bit image color levels and most operands in the micro control unit (MCU).

The histogram of errors in the same layer of WAGE-2888 shows that after being shifted and quantized layer by layer, the distribution of errors reshapes and mostly aggregates into truncated window.

Thus, most information for orientations is retained.

Besides, the smaller values in errors have negligible effects on previous orientations though accumulated layer by layer, which are partially discarded in quantization.

Since the width of the window has been optimized, we left-shift the window with factor γ to explore its horizontal position.

The right boundary can be formulated as max{|e|}/γ.

TAB2 shows the effect of shifting errors: although large values are in the minority, they play critical roles for backpropagation training while the majority with small values actually act as noises.

The bitwidth k G is set to 8 as default in previous experiments.

Although weights are propagated with ternary values in inference and achieve 16× compression rate than float32 weights, they are saved and accumulated in a relatively higher bitwidth (8 bits) for backpropagation training.

Therefore, the overall compression rate is only 4×.

The inconsistent bitwidth between weight updates k G and their effects in inference k W provides indispensable buffer space.

Otherwise, there might be too many weights changing their ternary values in each iteration, making training very slow and unstable.

To further explore a proper bitwidth for gradients, we use WAGE 2-8-8-8 in CIFAR10 as baseline and range k G from 2 to 12, the learning rate η is divided by 2 every time the k G decreases 1 bit to keep approximately equal weights accumulation in large number of iterations.

Results from TAB3 show the effect of k G and indicate the similar bitwidth requirement as previous experiments for k E .

54.22 51.57 28.22 18.01 11.48 7.61 6.78 6.63 6.43 6.55 6.57 For ImageNet implementation, we conduct six patterns to show bitwidth requirements: 2888 from TAB0 , 288C for more accurate errors (12 bits), 28C8 for larger buffer space, 28f8 for none quantization of gradients, 28ff for errors and gradients in float32 as unlimited case and its BN counterpart.

The accuracy of original AlexNet reproduction is reported as baseline.

Learning rate η is set to 64 and divided by 8 in 28C8 pattern, 0.01 and divided by 10 in 28f8, 28ff counterparts and vanilla AlexNet.

We observe overfitting when increasing k G thus add L2 weight decay of 1e-4, 1e-4 and 5e-4 for 28f8, 28ff and 28ff-BN patterns, respectively.

In table 4, the comparison between pattern 28C8 and 288C reveals that it might be more important to make more buffer space k G for gradient accumulation than to keep high-resolution orientation k E .

Besides, when it comes to ImageNet dataset, the gradient accumulation, i.e., the bit width of gradients (k G ) and batch normalization become more important since samples in training set are so variant.

To avoid external memory consumption of full-precision weights during training, BID3 achieved 1-bit weights representation in both training and inference.

They use a much larger minibatch size of 1000 and float32 backpropagation dataflow to accumulate more precise weight updates, equally compensating the buffer space in WAGE provided by external bits of k G .

However, large batch size will dramatically increase total training time, counteracting the speed benefits brought by integer arithmetic units.

Besides, intermediate variables like feature maps often consume much more memory than weights and linearly correlated with mini-batch size.

Therefore, we apply bigger k G for better convergence rate, accuracy and lower memory usage.

TAB5 ), but also halve the memory accesses costs and memory size requirements during training, which will greatly benefit mobile devices with on-site learning capability.

There are some points not involved in this work but yet to be improved or solved in future algorithm developments and hardware deployment.

MAC Operation: WAGE framework is mainly tested with 2-8-8-8 bitwidth configuration, which means that though there are no multiplications during inference with ternary weights, MACs are still needed to calculate g in training.

Possible solution is 2-2-8-8 pattern if we do not consider the matching of bitwidths between a and e. However, ternary a will dramatically slow down convergence and hurt accuracy since Q(x, 2) has two relatively high thresholds and clear most outputs of each layer at the beginning of training, this phenomenon is also observed in our BNN replication.

The linear mapping with uniform distance is adopted in WAGE for its simplicity.

However, non-linear quantization method like logarithmic representation BID16 BID27 ) might be more efficient because the weights and activations in a trained network naturally have logarithmic normal distributions as shown in FIG2 .

Besides, values in logarithmic representation have much larger range with fewer bits than fixed-point representation and are naturally encoded in digital hardware.

It is promising to training DNNs with integers encoded with logarithmic representation.

Normalization: Normalization layers like Softmax and batch normalization are avoided or removed in some WAGE demonstrations.

We think normalizations are essential for end-to-end multi-channel perception where sensors with different modalities have different input distributions, as well as cross-model features encoding and cognition where information from different branches gather to form higher-level representations.

Therefore, a better way to quantize normalization is of great interest in further studies.

6 CONCLUSION WAGE empowers pure low-bitwidth integer dataflow in DNNs for both training and inference.

We introduce a new initialization method and a layer-wise constant scaling factor to replace batch normalization, which is a pain spot for network quantization.

Many other components for training are also considered or simplified by alternative solutions.

In addition, the bitwidth requirements for error computation and gradient accumulation are explored.

Experiments reveal that we can quantize relative values of gradients, as well as discard the majority of small values and their orders of magnitude in backpropagation.

Although the accumulation for weights updates are indispensable for stable convergence and final accuracy, there still remain works for compression and memory consumption can be further reduced in training.

WAGE achieves state-of-art accuracies on multiple datasets with 2-8-8-8 bitwidth configuration.

It is promising for incremental works via fine-tuning, more efficient mapping, quantization of batch normalization, etc.

Overall, we introduce a framework without floating-point representation and demonstrate the potential to implement both discrete training and inference on integer-based lightweight ASIC or FPGA with on-site learning capability.

We assume that network structures are defined and initialized with Equation 5.

The annotations after pseudo code are potential corresponding operations for implementation in a fixed-point dataflow.

Algorithm 1 Training an I-layer net with WAGE method on floating-point-based or integer-based device.

Weights, activations, gradients and errors are quantized according to Equations 6 -12.

Require: a mini-batch of inputs and targets (a 0 q , a * ) which are quantized to k A -bit integers, shiftbased α for each layer, learning rate scheduler η, previous weight W saved in k G bits.

Ensure: updated weights W t+11.

Figure 5: Layerwise histograms of a trained VGG-like network with bitwidth configuration: 2-8-8-8 and learning rate η equals to 8.

The Y-axis represents for probability in W-plots and G-plots, and logarithmic probability in A-plots and E-plots, respectively.

In A-plots histograms are one-layer ahead so the first figure shows the quantized input data.

<|TLDR|>

@highlight

We apply training and inference with only low-bitwidth integers in DNNs

@highlight

A method called WAGE which quantizes all operands and operators in a neural network to reduce the number of bits for representation in a network.

@highlight

The authors propose discretized weights, activations, gradients, and errors at both training and testing time on neural networks
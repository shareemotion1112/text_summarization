The tremendous memory and computational complexity of Convolutional Neural Networks (CNNs) prevents the inference deployment on resource-constrained systems.

As a result, recent research focused on CNN optimization techniques, in particular quantization, which allows weights and activations of layers to be represented with just a few bits while achieving impressive prediction performance.

However, aggressive quantization techniques still fail to achieve full-precision prediction performance on state-of-the-art CNN architectures on large-scale classification tasks.

In this work we propose a method for weight and activation quantization that is scalable in terms of quantization levels (n-ary representations) and easy to compute while maintaining the performance close to full-precision CNNs.

Our weight quantization scheme is based on trainable scaling factors and a nested-means clustering strategy which is robust to weight updates and therefore exhibits good convergence properties.

The flexibility of nested-means clustering enables exploration of various n-ary weight representations with the potential of high parameter compression.

For activations, we propose a linear quantization strategy that takes the statistical properties of batch normalization into account.

We demonstrate the effectiveness of our approach using state-of-the-art models on ImageNet.

The increasing computational complexity and memory requirements of Convolutional Neural Networks (CNNs) have motivated recent research efforts in efficient representation and processing of CNNs.

Several optimization and inference approaches have been proposed with the objective of model compression and inference acceleration.

The primary aim of model compression is to enable on-device storage (e.g., mobile phones and other resource-constrained devices) and to leverage on-chip memory in order to reduce energy consumption BID13 , latency and bandwidth of parameter accesses.

Inference acceleration can be achieved by lowering the precision of computations in terms of resolution, removing connections within networks (pruning), and specialized software/hardware architectures.

Quantized Neural Networks are optimization techniques where weights and/or activations of a neural network are transformed from 32-bit floating point into a lower resolution; for aggressive quantization techniques down to binary BID7 BID22 or ternary BID16 BID31 representations.

Although prior quantization techniques achieve fullprecision accuracy on highly over-parameterized architectures (e.g., AlexNet, VGG) or toy tasks (e.g., MNIST, SVHN, CIFAR-10), there is still an unacceptable gap between extremely low bitwidth representations and state-of-the-art architectures for real-world tasks.

Furthermore, aggressive quantization approaches are usually designed for specific representations (e.g., binary or ternary), and are not scalable in the sense that they do not allow for more quantization levels (i.e., weight values) if required.

Thus, accuracy degradation cannot be compensated without changes in the baseline architecture which includes deepening or widening the neural network, and/or using full-precision weights in the input and output layers, respectively.

In this work, we address the issue of accuracy degradation by introducing a scalable non-uniform quantization method for weights that is based on trainable scaling factors in combination with a nested-means clustering approach.

In particular, nested-means splits the weight distribution iteratively into several quantization intervals until a pre-defined discretization level is reached.

Subsequently, all weights within a certain quantization interval are assigned the same weight (i.e., the scaling factor).

Nested-means clustering tends to assign small weights to larger quantization intervals while less frequent larger weights are assigned to smaller quantization intervals.

This improves classification performance which is in line with recent observations that larger weights carry more information than smaller weights BID9 .

We evaluate our approach on state-of-theart CNN architectures in terms of computational requirements and prediction accuracy using the ImageNet classification task.

The paper is structured as follows.

In Sec. 2 and Sec. 3 related work and background on inference acceleration is discussed.

Weight and activation quantization is presented in Sec. 4 and Sec. 5, respectively.

Experimental results for ImageNet are shown in Sec. 6.

Sec. 7 concludes the paper.

CNN quantization is an active research area with various approaches and objectives.

In this section, we briefly review the most promising strategies from different categories.

We distinguish between two orthogonal qualitative dimensions: (i) Approaches that only quantize the weights and approaches that quantize both the weights and the activations, and (ii) scalable approaches that allow for different bit-widths and non-scalable approaches that are designed for specific weight representations.

Non-scalable weight-only quantization: BID7 introduced Binary Connect (BC) in which they constrain weights to either -1 or 1.

BID22 proposed Binary Weight Networks (BWNs), which improves BC by introducing channel-wise scaling factors.

BID16 introduced Ternary Weight Networks (TWNs), in order to account for the accuracy degradation of BWNs.

BID31 proposed Trained Ternary Quantization (TTQ), extending TWNs by non-uniform and trainable scaling factors.

Scalable weight-only quantization: BID9 proposed Deep Compression, a method that leverages pruning, weight sharing and Huffman Coding for model compression.

BID29 proposed Incremental Network Quantization (INQ) to quantize pre-trained full-precision DNNs to zeros and powers of two.

This is accomplished by iteratively partitioning the weights into two sets, one of which is quantized while the other is retrained to compensate for accuracy degradation.

Non-scalable quantization for weights and activations: Binarized Neural Networks BID6 , XNOR-Net BID22 , Bi-Real Net BID19 and ABCNet quantize both weights and activations to either -1 or 1.

BID3 proposed low-precision activations using Half-Wave Gaussian Quantization (HWGQ), and relies on BWN for weight binarization.

Based on symmetric binary and ternary weights as proposed by BID31 , BID8 introduced Symmetric Quantization (SYQ) by pixel-wise scaling factors and fixed-point activation quantization.

Scalable quantization for weights and activations: DoReFa-Net BID30 , PACT BID5 and PACT-SAWB BID4 allows weights and activation to be variable configurable.

BID1 proposed uniform noise injection for non-uniform quantization (UNIQ) of both weights and activations.

use multi-bit quantization.

BID28 proposed an adaptively learnable quantizer (LQ-Nets) and achieves state-of-the-art accuracy as of today.

In this work, we focus on a scalable quantization of weights and activations, with the main objective of maintaining accuracy while exploiting model compression and reduced computational complexity for inference acceleration.

Our approach differs from prior work as we introduce a novel quantization strategy that is easy to compute, maintains state-of-the-art prediction accuracy, and has significantly less impact on training time.

In particular, our approach enables various non-uniform n-ary weight representations with a high amount of sparsity.

A quantization function maps the filter weights and/or the inputs to the convolution within a quantization interval to a quantization level α i .

A quantizer is uniform if α i+1 − α i = c, ∀i, where c is a constant quantization step.

If both filter weights and input data are quantized uniformely using the same number of quantization levels, then the scalar products in the QNN forward convolution can be realized by reduced-precision integer/fixed-point BID15 or half-precision floatingpoint BID21 ) operations.

Recent research BID3 shows, however, that input data requires more quantization levels than the filter weights in order to achieve single-precision floating point accuracy.

Few-bit integer scalar products with different amounts of quantization levels can be implemented using the bitserial/bit-plane approach BID6 which is done using bitwise XNOR or AND followed by popcount operations.

Non-uniform quantization usually achieves better classification accuracy than uniform quantization.

Furthermore, a certain amount of sparsity (percentage of zero elements) in the filter weights potentially reduces the number of operations without affecting the prediction performance BID9 .

In order to account for these representations, proposes the Efficient Inference Engine, a specialized hardware CNN inference accelerator that leverages non-uniform weights, uniform activations, and sparsity.

Another approach are reduce-and-scale architectures that can be efficiently realized in software on general-purpose hardware BID25 .

The basic idea of reduce-and-scale inference is to sum all equally weighted inputs and to multiply the result with the respective weight.

This approach efficiently leverages sparsity in weights and requires only one multiplication per quantization level and per output feature, shifting the computational bottleneck to additions.

Furthermore, weight quantization can be non-uniform, and quantization levels can be tailored for weights and activations, respectively.

Here, we focus on accelerating the inference based on the reduce-and-scale approach.

For weight quantization, we employ the common strategy of maintaining full-precision weights for training that are quantized during forward propagation.

The gradient of the full-precision weights is approximated by backpropagating through quantization functions using the straight-through gradient estimator (STE) BID2 .

The gradient is subsequently used to update the fullprecision weights.

For inference, the full-precision weights are discarded and only the quantized weights are used for model compression and inference acceleration.

In this section we describe the quantization strategy that is used at forward propagation.

The classification accuracy of aggressive quantization techniques heavily relies on the usage of scaling factors as model capacity is improved significantly.

For binary weights, BID22 propose one uniform scaling factor α l k per layer l and output feature map k, which is calculated as the mean of absolute floating-point weights, i.e., α DISPLAYFORM0 k denotes the set of weights connected to output feature map k. For ternary weights, BID31 propose two trainable non-uniform scaling factors α l + and α l − per layer that are determined by gradient descent.

This adjusts the scaling factors so as to minimize the given loss function while at the same time increasing the model capacity due to non-uniform scaling.

Therefore, we adopt trainable scaling factors in our method.

Let δ DISPLAYFORM1 +Kp be a set of interval thresholds that partition the real numbers into intervals DISPLAYFORM2 (1) If the zero weight should be explicitly modeled, we define an interval DISPLAYFORM3 , we assign a trainable scaling factor α l i that is used to quantize the weights as DISPLAYFORM4 .

During training, we update the scaling factors α l i using gradients computed as DISPLAYFORM5 where E denotes the loss function and w l q denotes the quantized weights.

In case the zero weight is modeled, we have a fixed scaling factor α l 0 = 0 that is not updated during gradient descent.

Finding good interval thresholds δ is essential for prediction performance and will be discussed in Sec. 4.3.

Allowing the weights to be non-uniform enables various explorations of weight representations.

Since weight distributions tend to be symmetric around zero (see FIG1 ), good quantized weight representations also exhibit a symmetry around zero.

Candidates for such representations are summarized in TAB0 .

DISPLAYFORM0 Binary and ternary representations gained a lot of interest lately due to their high compression ratios and, in view of their low expressiveness, relatively good prediction performance.

On large-scale classification tasks, however, only ternary weights demonstrate a prediction accuracy similar to fullprecision weights.

Furthermore, for ternary representations weight pruning is possible with little impact on prediction accuracy which additionally reduces the computational complexity of inference.

In this work, we also explore other n-ary weight representations that facilitate compression levels similar to ternary weights while significantly improving model capacity.

For instance, quaternary weights can also be encoded with only two bits, but introduce either an additional positive (quaternary+) or negative (quaternary-) scaling factor, respectively.

Quinary weights extend ternary weights by one positive and one negative value, but are still encoded with only two bits in a sparse format.

In a sparse format, only the indices of non-zero weight entries are stored such that two bits are sufficient to represent the non-zero values.

For an optimal approximation, weight clustering is required to partition a set of weights that are later represented by a single discrete value per cluster (cf.

Sec. 4.1).

The clustering can be implemented either statically (once before training) or dynamically (repeatedly during training) by calculating thresholds δ which represent the boundaries of the respective cluster.

The static approach has the advantage of allowing iterative clustering algorithms to be applied (e.g., k-means clustering or Lloyd's algorithm BID20 ) that are able to find an optimal solution for the cluster assignment.

However, as quantization lowers the resolution and therefore changes in the weight distribution, quantization requires re-training to compensate for this loss of information.

As a consequence, the optimal solution found by an iterative algorithm will become non-optimal during the following re-training process.

Lowering the learning rate for re-training can diminish heavy changes in the weight distribution, at the cost of longer time to converge and the risk to get stuck at plateau regions, which is especially critical for trainable scaling factors (Eq. 2).

Applying an iterative clustering approach repeatedly during training is practically infeasible, since it causes a dramatic increase in training time.

A practical useful clustering solution is to calculate cluster thresholds during training based on the maximum absolute value of the full-precision weights per layer BID31 DISPLAYFORM0 where t is a hyperparameter.

This approach is beneficial because it defines cluster thresholds which are influenced by large weights that were shown to play a more important role than smaller weights BID10 .

Furthermore, training time is virtually unaffected by this rather simple calculation.

However, having an additional hyperparameter t i for each scaling factor α i renders the mandatory hyperparameter tuning infeasible.

Furthermore, the sensitivity to the maximum value results in aggressive threshold changes caused by weight updates, possibly even preventing the network from converging.

In order to overcome these issues, we propose a symmetric nested-means clustering algorithm for assigning full-precision weights to a set of quantization clusters.

Since weight distributions are typically symmetric around zero, we divide the weights into a positive and a negative cluster (I l +1and I l −1 ).

These clusters are then divided at their arithmetic means δ l +i and δ l −i into two subclustersfor each cluster we obtain an inner cluster and an outer cluster containing the tail of the distribution.

The subclusters containing the tail of the distribution (I l +i+1 and I l −i−1 ) are repeatedly divided at their arithmetic means until the targeted number of quantization intervals is reached.

More formally, nested-means clustering iteratively computes Nested-means clustering naturally defines the cluster thresholds in a way that the cluster intervals become smaller for larger weights.

Although this might seem counter-intuitive since most weights are close to zero and the sum of quantization errors could be made smaller by using a higher quantization resolution around zero, this is actually beneficial as large weights were shown to play a more important role than smaller weights BID9 .

At the same time the mean is less sensitive to weight updates than the maximum absolute weight value BID31 , allowing for better convergence.

Last, nested-means clustering is hyperparameter-free and requires only one arithmetic mean per cluster, which is computationally efficient.

DISPLAYFORM1 DISPLAYFORM2

The computational heavy lifting of the reduce-and-scale inference is reducing equally weighted inputs to a convolution layer (as discussed in Sec. 3).

Lowering the bit width of the inputs enables better usage of data level parallelism (performing the same operation on multiple inputs simultaneously) and results in less operations and memory accesses.

We use a linear quantization scheme for activations that allows the required low bit-width additions to be performed using integer arithmetic on commodity devices.

The rectified linear unit (ReLU) activation function f (x) = max(0, x) is a commonly used nonlinear activation function due its computational efficiency and its ability to alleviate the vanishing gradient problem.

However, the ReLU function produces unbounded outputs which potentially require a high dynamic range and are therefore difficult to quantize.

A common solution for this is to clip the outputs in the interval (0, γ]: DISPLAYFORM0 When selecting the clipping parameter γ, a trade-off needs to be made: On the one side, small values of γ produce gradient mismatches due to different forward and backward approximations in the clipped interval (γ, ∞).

On the other side, large values of γ result in a large interval (0, γ] that needs to be quantized.

This is problematic if only a few bits are used as quantization errors might become large.

In order to define an appropriate clipping interval, we use the observation that pre-activations tend to have a Gaussian distribution BID14 .

In a Gaussian distribution, most values lie within a rather small range and there are only a few outliers that yield a high absolute range.

For instance, 99.7% of the values lie within three standard deviations σ of the mean µ (Smirnov & Dunin-Barkovskiȋ, 1963).

We find this empirical rule to be a good approximation to filter out outliers and define the clipping interval as γ = µ + 3σ.

This approach approximates the ReLU function well but suffers from the drawback that µ and σ need to be repeatedly calculated during training.

In recent years, batch normalization became a standard tool to accelerate convergence of state-of-the-art CNNs BID14 .

Batch normalization transforms individual pre-activations to approximately have zero mean and unit variance across all data samples.

BID3 experimentally showed that the pre-activation distribution after batch normalization are all close to a Gaussian with zero mean and unit variance.

Therefore, we propose to select a fixed clipping parameter γ = 3 as it results in a small quantization interval (0, γ] while also keeping the number of clipped activations x > γ small.

We applied the proposed quantization approach on ResNet BID12 and a variant of the Inception network BID14 , trained on the ImageNet classification task (Russakovsky et al., 2015) .

We use TensorFlow BID0 ) and the Tensorpack library .

For the ResNet network, the learning rate starts from 0.1 and is divided by 10 each 30 × 100 iterations, a weight decay of 0.0001, and a momentum of 0.9 is used.

For the Inception network, we schedule learning rates following the configuration of , a weight decay of 0.0001, and a momentum of 0.9.

We use eight GPUs for training and a batch size of 64 per GPU.

The quantized networks leverage initialization with pre-trained full-precision parameters.

We quantize all convolutional layers and fully-connected layers except the input and output layers, respectively, to avoid accuracy degradation BID30 BID31 BID8 BID28 .

A detailed comparison to reported results of various related methods is summarized in Appendix B.

We evaluate our nested-means weight quantization on ternary, quaternary-and quinary representations TAB0 as these representations have good compression/acceleration potential while achieving the best accuracy.

TAB1 reports the validation accuracy of ResNet-18 and Inception-BN on the ImageNet task.

The training time increases with increasing quantization levels because of the additional computations of interval thresholds (Eq. 3).

While the impact on training time is negligible for the Inception- BN model, the ResNet model shows an increase of up to 2.0x.

This is caused by a better GPU utilization of ResNet which makes the overall training time more sensitive to additional operations.

In this section, we evaluate the activation quantization using ResNet-18 on ImageNet.

We use the clipped ReLU (Eq. 5) and set γ = 3 (as discussed in Sec. 5) for quantized activations, and we use the ReLU without clipping and quantization for 32-bit activations.

TAB2 reports the validation accuracy and the increase in training time for several activation bit-widths.

The training time is only influenced by the weight quantization, whereas the influence of activation quantization is neglibible.

The efficiency of the activation quantization is also shown in FIG2 .

As can be seen, the learning curves of ternary weights and 32-bit and 4-bit activations are roughly identical which highlights the robustness of the proposed quantization approach.

Only 2-bit activations result in a slight accuracy degradation, but also this case shows a stable learning behavior.

In order to show the effectiveness of the nested-mean clustering, we experimentally show the impact of the key components.

First, we show the importance of the threshold robustness to weight changes and the ability to represent general n-ary weights with a comparison to BID31 .

Then we reason about the effectiveness of nested-means clustering by comparing it to several combinations of a quantile clustering approach that is described below.

Impact of threshold robustness and configurable quantization levels: As described in Section 4.3, defining cluster thresholds that are robust against updates in the underlying weight distribution is vital for convergence and the prediction performance.

The approach of BID31 defines the thresholds based on the absolute maximum of the underlying weight distribution which causes aggressive threshold changes during weight updates.

Furthermore, the configurability of quantization levels of our approach allows n-ary weight representations.

For instance, the quaternary representations requires the same amount of bits as ternary representations (thus, the same compression ratio) but achieves a significantly higher accuracy.

TAB3 summarizes the accuracy differences of both Impact of the nested clustering: We validate the effectiveness of the nested-mean clustering by comparing it to quantile clustering.

Assuming that the weights are approximately Gaussian distributed, we compute the cluster thresholds so that each cluster approximately contains a prespecified amount of weights.

Let Φ −1 (p) be the quantile function of the standard Gaussian distribution with zero mean and unit variance.

Given a vector p = (p 1 , . . .

, p L ) of pre-specified cluster probabilities that sum to one, the thresholds are computed as δ summarizes the accuracy using quinary weights of both nested-means clustering and quantile clustering for several cluster sizes p.

We start with equal cluster sizes and incrementally increase the cluster size of smaller weights while at the same time decreasing the cluster size of larger weights.

The accuracy improves if we assign larger clusters to small weights and smaller clusters to large weights which validates our hypothesis.

We want to emphasize that quantile-clustering assumes a Gaussian weight distribution which nested-means clustering does not.

We summarize the parameter requirements and compression ratios, including the first and last layer of the ResNet18 model, in TAB5 .

The computational efficiency is shown on the example of a typical ResNet layer in TAB6 .

As described in Sec. 3, we target reduce-and-scale inference which lowers most operations to reduced-precision additions and requires one full-precision multiplication per quantization level and per output feature.

Please note, that we add another 7 bits to the bit width of activations for the addition operations in order to prevent overflows (we cannot rely on the normalization ability of floating-point arithmetic on integer hardware).

Our approach removes

almost the complete multiplication workload due to the extremely low amount of quantization levels.

The sparsity in the filter weights and the reduced bit-width of activations improves the addition workload by a factor of 4.5x to 9.1x.

We have presented a novel approach for compressing CNNs through quantization and connection pruning, which reduces the resolution of weights and activations and is scalable in terms of the number of quantization levels.

As a result, the computational complexity and memory requirements of DNNs are substantially reduced, and an execution on resource-constrained devices is more feasible.

We introduced a nested-means clustering algorithm for weight quantization that finds suitable interval thresholds that are subsequently used to assign each weight to a trainable scaling factor.

Our approach exhibits both a low computational complexity and robustness to weight updates, which makes it an attractive alternative to other clustering methods.

Furthermore, the proposed quantization method is flexible as it allows for various numbers of quantization levels, enabling high compression rates while achieving prediction accuracies close to single-precision floating-point weights.

For instance, we utilize this flexibility to add an extra quantization level to ternary weights (quaternary weights), resulting in an improvement in prediction accuracy while keeping the bit width at two.

For activation quantization, we developed an approximation based on statistical attributes that have been observed when batch normalization is employed.

Experiments using state-of-the-art DNN architectures on real-world tasks, including ResNet-18 and ImageNet, show the effectiveness of our approach.

<|TLDR|>

@highlight

We propose a quantization scheme for weights and activations of deep neural networks. This reduces the memory footprint substantially and accelerates inference.

@highlight

CNN model compression aand inference acceleration using quantization.
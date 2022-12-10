Recent work has shown that performing inference with fast, very-low-bitwidth (e.g., 1 to 2 bits) representations of values in models can yield surprisingly accurate results.

However, although 2-bit approximated networks have been shown to be quite accurate, 1 bit approximations, which are twice as fast, have restrictively low accuracy.

We propose a method to train models whose weights are a mixture of bitwidths, that allows us to more finely tune the accuracy/speed trade-off.

We present the “middle-out” criterion for determining the bitwidth for each value, and show how to integrate it into training models with a desired mixture of bitwidths.

We evaluate several architectures and binarization techniques on the ImageNet dataset.

We show that our heterogeneous bitwidth approximation achieves superlinear scaling of accuracy with bitwidth.

Using an average of only 1.4 bits, we are able to outperform state-of-the-art 2-bit architectures.

With Convolutional Neural Nets (CNNs) now outperforming humans in vision classification tasks BID11 , it is clear that CNNs will be a mainstay of AI applications.

However, CNNs are known to be computationally demanding, and are most comfortably run on GPUs.

For execution in mobile and embedded settings, or when a given CNN is evaluated many times, using a GPU may be too costly.

The search for inexpensive variants of CNNs has yielded techniques such as hashing BID0 , vector quantization BID4 , and pruning BID5 .

One particularly promising track is binarization BID1 , which replaces 32-bit floating point values with single bits, either +1 or -1, and (optionally) replaces floating point multiplies with packed bitwise popcount-xnors .

Binarization can reduce the size of models by up to 32×, and reduce the number of operations executed by up to 64×.Binarized CNNs are faster and smaller, but also less accurate.

Much research has therefore focused on reducing the accuracy gap between binary models and their floating point counterparts.

The typical approach is to add bits to the activations and weights of a network, giving a better approximation of the true values.

However, the cost of extra bits is quite high.

Using n bits to approximate just the weights increases the computation and memory required by a factor of n compared to 1-bit binarization.

Further using n bits to approximate activations as well requires n 2 times the resources as one bit.

There is thus a strong motivation to use as few bits as possible while still achieving acceptable accuracy.

However, today's binary approximations are locked to use the same number of bits for all approximated values, and the gap in accuracy between bits can be substantial.

For example, recent work concludes 1-bit accuracy is unsatisfactory while 2-bit accuracy is quite high BID12 (also see TAB0 ).In order to bridge the gap between integer bits, we introduce Heterogeneous Bitwidth Neural Networks (HBNNs), which use a mix of integer bitwidths to allow values to have effectively (i.e., on average) fractional bitwidths.

The freedom to select from multiple bitwidths allows HBNNs to approximate each value better than fixed-bitwidth schemes, giving them disproportionate accuracy gains for the number of effective bits used.

For instance, Alexnet trained with an average of 1.4 bits has comparable (actually, slightly higher) accuracy to training with a fixed two bits TAB0 .Our main contributions are:(1) We propose HBNNs as a way to break the integer-bitwidth barrier in binarized networks.(2) We study several techniques for distributing the bitwidths in a HBNN, and introduce the middle-out bitwidth selection algorithm, which uses the full representational power of heterogeneous bitwidths to learn good bitwidth distributions.

(3) We perform a comprehensive study of heterogeneous binarization on the ImageNet dataset using an AlexNet architecture.

We evaluate many fractional bitwidths and compare to state of the art results.

HBNNs typically yield the smallest and fastest networks at each accuracy.

Further, we show that it is usually possible to equal, or improve upon, 2-bitbinarized networks with an average of 1.4 bits.

(4) We show that heterogeneous binarization is applicable to MobileNet BID6 , demonstrating that its benefits apply even to modern, optimized architectures.

In this section we discuss existing techniques for binarization.

When training a binary network, all techniques including ours maintain weights in floating point format.

During forward propagation, the weights (and activations, if both weights and activations are to be binarized) are passed through a binarization function B, which projects incoming values to a small, discrete set.

In backwards propagation, a custom gradient,which updates the floating point weights, is applied for the binarization layer,.

After training is complete, the binarization function is applied one last time to the floating point weights to create a true binary (or more generally, small, discrete) set of weights, which is used for inference from then on.

Binarization was first introduced by BID1 .

In this initial investigation, dubbed BinaryConnect, 32-bit tensors T were converted to 1-bit variants T B using the stochastic equation DISPLAYFORM0 +1 with probability p = σ(T ), -1 with probability 1 − pwhere σ is the hard sigmoid function defined by σ(x) = max(0, min(1,2 )).

For the custom gradient function, BinaryConnect simply used DISPLAYFORM1 Although BinaryConnect showed excellent results on relatively simple datasets such as CIFAR-10 and MNIST, it performed poorly on ImageNet, achieving only an accuracy of 27.9%.

later improved this model by simplifying the binarization by simply taking T B = sign(T ) and adding a gradient for this operation, namely the straight-through estimator: DISPLAYFORM2 The authors showed that the straight-through estimator further improved accuracy on small datasets.

However, they did not attempt to train a model ImageNet in this work.

BID10 made a slight modification to the simple pure single bit representation that showed improved results.

Now taking a binarized approximation as DISPLAYFORM3 This additional scalar term allows binarized values to better fit the distribution of the incoming floating-point values, giving a higher fidelity approximation for very little extra computation.

The addition of scalars and the straight-through estimator gradient allowed the authors to achieve an accuracy on ImageNet, 44.2% Top-1, a significant improvement over previous work.

and BID13 found that increasing the number of bits used to quantize the activations of the network gave a considerable boost to the accuracy, achieving similar Top-1 accuracy of 51.03% and 50.7% respectively.

The precise binarization function varied, but the typical approaches include linearly placing the quantization points between 0 and 1, clamping values below a threshold distance from zero to zero BID9 , and computing higher bits by measuring the residual error from lower bits BID12 .

All n-bit binarization schemes require similar amounts of computation at inference time, and have similar accuracy (see TAB0 ).

In this work, we extend the residual error binarization function BID12 for binarizing to multiple (n) bits: DISPLAYFORM4 where T is the input tensor, E n is the residual error up to bit n, T B n is a tensor representing the n th bit of the approximation, and µ n is a scaling factor for the n th bit.

Note that the calculation of bit n is a recursive operation that relies on the values of all bits less than n. Residual error binarization has each additional bit take a step from the value of the previous bit.

FIG0 illustrates the process of binarizing a single value to 3 bits.

Since every binarized value is derived by taking n steps, where each step goes left or right, residual error binarization approximates inputs using one of 2 n values.

To date, there remains a considerable gap between the performance of 1-bit and 2-bit networks (compare rows 7 and 9 of TAB0 ).

The highest full (i.e., where both weights and activations are quantized) single-bit performer on AlexNet, Xnor-Net, remains roughly 7 percentage points less accurate (top 1) than the 2-bit variant, which is itself about 5.5 points less accurate than the 32-bit variant (row 16).

When only weights are binarized, very recent results BID3 similarly find that binarizing to 2 bits can yield nearly full accuracy (row 2), while the 1-bit equivalent lags by 4 points (row 1).

The flip side to using 2 bits for binarization is that the resulting models require double the number of operations as the 1-bit variants at inference time.

These observations naturally lead to the question, explored below, of whether it is possible to attain accuracies closer to those of 2-bit models while running at speeds closer to those of 1-bit variants.

Of course, it is also fundamentally interesting to understand whether it is possible to match the accuracy of higher bitwidth models with those that have lower (on average) bitwidth.

In this section, we discuss how to extend residual error binarization to allow heterogeneous (effectively fractional) bitwidths.

We develop several different methods for distributing the bits of a heterogeneous approximation.

We point out the inherent representational benefits of heterogeneous binarization.

Finally, we discuss how HBNNs could be implemented efficiently to benefit from increased speed and compression.

We modify Equation 4 , which binarizes to n bits, to instead binarize to a mixture of bitwidths by changing the third line as follows: DISPLAYFORM0 Note that the only addition is tensor M , which is the same shape as T , and specifies the number of bits M j that the j th entry of T should be binarized to.

In each round n of the binarization recurrence, we now only consider values that are not finished binarizing, i.e, which have M j ≥ n. Unlike homogeneous binarization, therefore, heterogeneous binarization generates binarized values by taking up to, not necessarily exactly, n steps.

Thus, the number of distinct values representable is n i=1 2 n = 2 n+1 − 2, which is roughly double that of the homogeneous case.

In the homogeneous case, on average, each step improves the accuracy of the approximation, but there may be certain individual values that would benefit from not taking a step, in FIG0 for example, it is possible that (µ 1 − µ 2 ) approximates the target value better than (µ 1 − µ 2 + µ 3 ).

If values that benefit from not taking a step can be targeted and assigned fewer bits, the overall approximation accuracy will improve despite there being a lower average bitwidth.

The question of how to distribute bits in a heterogeneous binary tensor to achieve high representational power is equivalent to asking how M should be generated.

When computing M , our goal is to take a set of constraints indicating what fraction of T should be binarized to each bitwidth, perhaps 70% to 1 bit and 30% to 2 bits for example, and choose those values which benefit most (or are hurt least) by not taking additional steps.

Algorithm 1 shows how we compute M .

Input: A tensor T of size N and a list P of tuples containing a bitwidth and the percentage of T that should be binarized to that bitwidth.

P is sorted by bitwidth, smallest first.

Output: A bit map M that can be used in Equation 5 to heterogeneously binarize T .

1: R = T Initialize R, which contains values that have not yet been assigned a bitwidth 2: DISPLAYFORM0 b is a bitwidth and p b is the percentage of T to binarize to width b.

S = select(R) Sort indices of remaining values by suitability for b-bit binarization.5: DISPLAYFORM0 Do not consider these indices in next step.7:x += p b N 8: end forThe algorithm simply steps through each bitwidth b (line 3), and for the corresponding fraction p b of values to be binarized to b bits, selects (lines 4 and 5) the "most suitable" p b N values of T to be binarized to b bits.

Once values are binarized, they are not considered in future steps (line 6).

We propose several simple methods as candidates for the select function: Top-Down (TD), Middle-Out (MO), Bottom-Up (BU), and Random (R) selection.

The first three techniques depend on the input data.

They pick the largest, closest-to-mean or smallest values.

The last technique is oblivious to incoming data, and assigns a fixed uniform pattern of bitwidths.

DISPLAYFORM1 The intuition for Middle-Out derives from FIG0 , where we see that when a step is taken from the previous bit, that previous bit falls in the middle of the two new values.

Thus, the entries of T that most benefit from not taking a step are those that are close to the center (or "middle") of the remaining data.

This suggests that fixing values near the middle of the distribution to a low bitwidth will yield the best results.

Our results show that MO is much better than the other techniques.

The typical appeal of binary networks is that they reduce model size and the number of computations needed.

Model size is reduced by replacing 32-bit-float weights with a small number of bits and packing those bits into 64-bit integers.

Computation reduction becomes possible when both inputs and weights are binarized .

This allows floating point multiplications to be replaced by popcount-xnor operations (which is the bit equivalent of a multiply accumulate).

A single popcount-xnor on packed 64-bit inputs does the work of 64 multiply accumulates.

However, because heterogeneous bitwidth tensors are essentially sparse, they can not be efficiently packed into integers.

Both packing and performing xnor-popcounts on a heterogeneous tensor would require an additional tensor like M that indicates the bitwidth of each value.

However, packing is only needed because CPUs and GPUs are designed to operate on groups of bits simultaneously.

In custom hardware such as an ASIC or FPGA, each bit operation can be efficiently performed individually.

Because the distribution of heterogeneous weight bits will be fixed at inference time (activations would be binarized homogeneously), fixed gates can be allocated depending on the bitwidth of individual values.

This addresses the challenge of sparsity and allows a heterogeneous bitwidth FPGA implementation to have fewer total gates and a lower power consumption than a fixed bitwidth implementation.

To evaluate HBNNs we wished to answer the following four questions:(1) How does accuracy scale with an uninformed bit distribution?

In this section we address each of these questions.

AlexNet with batch-normalization (AlexNet-BN) is the standard model used in binarization work due to its longevity and the general acceptance that improvements made to accuracy transfer well to more modern architectures.

Batch normalization layers are applied to the output of each convolution block, but the model is otherwise identical to the original AlexNet model proposed by BID8 .

Besides it's benefits in improving convergence, batch-normalization is especially important for binary networks because of the need to equally distribute values around zero.

We additionally insert binarization functions within the convolutional layers of the network when binarizing weights and at the input of convolutional layers when binarizing inputs.

We keep a floating point copy of the weights that is updated during back-propagation, and binarized during forward propagation as is standard for binary network training.

We use the straight-through estimator for gradients.

When binarizing the weights of the network's output layer, we add a single parameter scaling layer that helps reduce the numerically large outputs of a binary layer to a size more amenable to softmax, as suggested by BID12 .

We train all models using an SGD solver with learning rate 0.01, momentum 0.9, and weight decay 1e-4 and randomly initialized weights for 90 epochs, using the PyTorch framework.

Here we conduct two simple experiments to measure the ability of various binarization schemes to approximate a floating point tensor.

As a baseline, we test a "poor man's" approach to HBNNs, where we fix up front the number of bits each kernel is allowed, require all values in a kernel to have its associated bitwidth, and then train as with conventional, homogeneous binarization.

We consider 10 mixes of 1, 2 and 3-bit kernels so as to sweep average bitwidths between 1 and 2.

We trained as described in Section 4.1.

For this experiment, we used the CIFAR-10 dataset with a deliberately hobbled (4-layer fully conventional) model with a maximum accuracy of roughly 78% as the baseline 32-bit variant.

We chose CIFAR-10 to allow quick experimentation.

We chose not to use a large model for CIFAR-10, because for large models it is known that even 1-bit models have 32-bit-level accuracy .

FIG5 shows the results.

Essentially, accuracy increases roughly linearly with average bitwidth.

Although such linear scaling of accuracy with bitwidth is itself potentially useful (since it allows finer grain tuning on FPGAs), we are hoping for even better scaling with the "data-aware" bitwidth selection techniques of Equation 5.

To compare the approximating capability of the selection methods of Equation 6, we generate a large tensor of normally distributed random values and apply top-down, bottom-up, middle-out, and random binarization with a variety of bit mixes to binarize it.

We compare the normalized Euclidean distance of the approximated tensors with the input as a measure of how well the input is approximated.

We additionally binarize the tensor using homogeneous bitwidth approximation to gauge the representational power of previous works.

The results, shown in FIG2 , show that middle-out selection vastly outperforms other methods.

In fact, it is clear that 1.4 bits distributed with the middle-out algorithm approximates the input roughly Unbinarized (our implementation) 16 Alexnet BID8 full precision / full precision 56.5% 80.1%as well as standard 2-bit binarization.

Also interesting to note is that the actual bit mix changes performance rather than just the average.

A mix of 40%/50%/10% 1-bit/2-bit/3-bit provides quite a bit better than a mix of 50%/30%/20% 1-bit/2-bit/3-bit despite both having an average of 1.7 bits.

This difference suggests that some bit mixtures may be better suited to approximating a distribution than others.

To measure the transfer of the representational benefits of Middle-Out selection to a real network training scenario, we apply each of the selection techniques discussed to the AlexNet-BN model described in Section 4.3 and train with the same hyper-parameters as in Section 4.1.

We binarize the weights of each model to an average of 1.4 bits using 70% 1 bit values, 20% 2 bit values, and 10% 3 bit values.

For random distribution, we sample from a uniform bitwidth distribution before training and then fix those values.

Unlike in CIFAR-10, bits are randomly distributed within kernels as well.

The results are shown in FIG2 .

Quite clearly, Middle-Out selection outperforms other selection techniques by a wide margin, and is in fact roughly the same accuracy as using a full two bits.

Interestingly, the accuracy achieved with Bottom-Up selection falls on the linear projection between 1 and 2 bits.

Random and Top-Down distribution perform below the linear.

Thus, Middle-Out selection seems to be the only technique that allows us to achieve a favorable trade-off between accuracy and bitwidth and for this reason is the technique we focus on in the rest of our experiments.

Recently, BID3 were able to binarize the weights of an AlexNet-BN model to 2 bits and achieve nearly full precision accuracy (row 2 of TAB0 ).

We consider this to be the state of the art in weight binarization since the model achieves excellent accuracy despite all layer weights being binarized, including the first and last layers which have traditionally been difficult to approximate.

We perform a sweep of AlexNet-BN models binarized with fractional bitwidths using middle-out selection with the goal of achieving comparable accuracy using fewer than two bits.

The results of this sweep are shown in FIG5 .

We were able to achieve nearly identical top-1 accuracy to the best full 2 bit results (55.3%) with an average of only 1.4 bits (55.2%).

As we had hoped, we also found that the accuracy scales in a super-linear manner with respect to bitwidth when using middle-out compression.

Specifically, the model accuracy increases extremely quickly from 1 bit to 1.3 bits before slowly approaching the full precision accuracy.

We explored many different mixes of bits that gave the same average bitwidth, but found that they gave nearly identical results, suggesting that when training from scratch the composition of a bit mix does not matter nearly so much as the average number of bits.

Our 1-bit performance is notably worse, perhaps because we did not incorporate the improvements to training binary nets suggested by BID3 .

Adding stochastic layer binarization may have boosted our low-bitwidth results and allowed us to achieve near full precision accuracy with an even lower bitwidth.

To confirm that heterogeneous binarization can transfer to state of the art networks, we apply 1.4 bit binarization with 70% 1 bit, 20% 2 bit, and 10% 3 bit values to MobileNet (Howard et al., 2017) , a state of the art architecture that achieves 68.8% top-1 accuracy.

To do this, we binarize the weights of all the depthwise convolutional layers (i.e., 13 of 14 convolutional layers) of the architecture to 1.4 bits using middle-out selection and train with the same hyper-parameters as AlexNet.

Our HBNN reached a top-1 accuracy of 65.1%.

In order to realize the speed-up benefits of binarization (on CPU or FPGA) in practice, it is necessary to binarize both inputs the weights, which allows floating point multiplies to be replaced with packed bitwise logical operations.

The number of operations in a binary network is reduced by a factor of 64 mn where m is the number of bits used to binarize inputs and n is the number of bits to binarize weights.

Thus, there is significant motivation to keep the bitwidth of both inputs and weights as low as possible without losing too much accuracy.

When binarizing inputs, the first and last layers are typically not binarized as the effects on the accuracy are much larger than other layers.

We perform another sweep on AlexNet-BN with all layers but the first and last fully binarized and compare the accuracy of HBNNs to several recent results.

Row 7 of TAB0 is the top previously reported accuracy (44.2%) for single bit input and weight binarization, while row 9 (51%) is the top accuracy for 2-bit inputs and 1-bit weights.

TAB0 (rows 12 to 15) reports a selection of results from this search.

Using 1.4 bits to binarize inputs and weights (mn = 1.4 × 1.4 = 1.96) gives a very high accuracy (53.2% top-1) while having the same number of total operations mn as a network, such as the one from row 7, binarized with 2 bit activations and 1 bit weights.

We have similarly good results when leaving the input binarization bitwidth an integer.

Using 1 bit inputs and 1.4 bit weights, we reach 49.4% top-1 accuracy which is a large improvement over BID10 at a small cost.

We found that using more than 1.4 average bits had very little impact on the overall accuracy.

Binarizing inputs to 1.4 bits and weights to 1 bit (row 14) similarly outperforms (row 7, mentioned above); however, the accuracy improvement margin is smaller.

In this paper, we present Heterogeneous Bitwidth Neural Networks (HBNNs), a new type of binary network that is not restricted to integer bitwidths.

Allowing effectively fractional bitwidths in networks gives a vastly improved ability to tune the trade-offs between accuracy, compression, and speed that come with binarization.

We show a simple method of distributing bits across a tensor lead to a linear relationship between accuracy and number of bits, but using a more informed method allows higher accuracy with fewer bits.

We introduce middle-out bit selection as the top performing technique for determining where to place bits in a heterogeneous bitwidth tensor and find that Middle-Out enables a heterogeneous representation to be more powerful than a homogeneous one.

On the ImageNet dataset with AlexNet and MobileNet models, we perform extensive experiments to validate the effectiveness of HBNNs compared to the state of the art and full precision accuracy.

The results of these experiments are highly compelling, with HBNNs matching or outperforming competing binarization techniques while using fewer average bits.

The use of HBNNs enables applications which require higher compression and speeds offered by a low bitwidth but also need the accuracy of a high bitwidth.

As future work, we will investigate modifying the bit selection method to make heterogeneous bit tensors more amenable for CPU computation as well as develop a HBNN FPGA implementation which can showcase both the speed and accuracy benefits of heterogeneous binarization.

<|TLDR|>

@highlight

We introduce fractional bitwidth approximation and show it has significant advantages.

@highlight

Suggests a method for varying the degree of quantization in a neural network during the forward propagation phase

@highlight

Maintaining the accuracy of 2bits netword while using less than 2bits weights
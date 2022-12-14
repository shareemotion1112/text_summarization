Operating deep neural networks on devices with limited resources requires the reduction of their memory footprints and computational requirements.

In this paper we introduce a training method, called look-up table quantization (LUT-Q), which learns a dictionary and assigns each weight to one of the dictionary's values.

We show that this method is very flexible and that many other techniques can be seen as special cases of LUT-Q. For example, we can constrain the dictionary trained with LUT-Q to generate networks with pruned weight matrices or restrict the dictionary to powers-of-two to avoid the need for multiplications.

In order to obtain fully multiplier-less networks, we also introduce a multiplier-less version of batch normalization.

Extensive experiments on image recognition and object detection tasks show that LUT-Q consistently achieves better performance than other methods with the same quantization bitwidth.

In this paper, we propose a training method for reducing the size and the number of operations of a deep neural network (DNN) that we call look-up table quantization (LUT-Q).

As depicted in Fig. 1 , LUT-Q trains a network that represents the weights W ∈ R O×I of one layer by a dictionary d ∈ R K and assignments A ∈ [1, . . .

, K] O×I such that Q oi = d Aoi , i.e., elements of Q are restricted to the K dictionary values in d. To learn the assignment matrix A and dictionary d, we iteratively update them after each minibatch.

Our LUT-Q algorithm, run for each mini-batch, is summarized in TAB1 LUT-Q has the advantage to be very flexible.

By simple modifications of the dictionary d or the assignment matrix A, it can implement many weight compression schemes from the literature.

For example, we can constrain the assignment matrix and the dictionary in order to generate a network with pruned weight matrices.

Alternatively, we can constrain the dictionary to contain only the values {−1, 1} and obtain a Binary Connect Network BID3 , or to {−1, 0, 1} resulting in a Ternary Weight Network BID12 .

Furthermore, with LUT-Q we can also achieve Multiplier-less networks by either choosing a dictionary d whose elements d k are of the form d k ∈ {±2 b k } for all k = 1, . . .

, K with b k ∈ Z, or by rounding the output of the k-means algorithm to powers-of-two.

In this way we can learn networks whose weights are powers-of-two and can, hence, be implemented without multipliers.

The memory used for the parameters is dominated by the weights in affine/convolution layers.

Using LUT-Q, instead of storing W, the dictionary d and the assignment matrix A are stored.

Hence, for an affine/convolution layer with N parameters, we reduce the memory usage in bits from N B float to just KB float + N ⌈log 2 K⌉, where B float is the number of bits used to store one weight.

Furthermore, using LUT-Q we also achieve a reduction in the number of computations: for example, affine layers trained using LUT-Q need to compute just K multiplications at inference time, instead of I multiplications for a standard affine layer with I input nodes.

DISPLAYFORM0

For the description of our results we use the following naming convention: Quasi multiplier-less networks avoid multiplications in all affine/convolution layers, but they are not completely multiplierless since they contain multiplications in standard batch normalization (BN) layers.

For example, the networks described in BID23 are quasi multiplier-less.

Fully multiplier-less networks avoid all multiplications at all as they use our multiplier-less BN (see appendix A).

Finally, we call all other networks unconstrained.

We conducted extensive experiments with LUT-Q and multiplier-less networks on the CIFAR-10 image classification task BID10 , on the ImageNet ILSVRC-2012 task BID20 and on the Pascal VOC object detection task BID4 .

All experiments are carried out with the Sony Neural Network Library 4 .For CIFAR-10, we first use the full precision 32-bit ResNet-20 as reference (7.4% error rate).

Quasi multiplier-less networks using LUT-Q achieve 7.6% and 8.0% error rate for 4-bit and 2-bit quantization respectively.

Fully multiplier-less networks with LUT-Q achieve 8.1% and 9.0% error rates, respectively.

LUT-Q can also be used to prune and quantize networks simultaneously.

Fig. 2 shows the error rate increase between the baseline full precision ResNet-20 and the pruned and quantized network.

Using LUT-Q we can prune the network up to 70% and quantize it to 2-bit without significant loss in accuracy.

For Imagenet, we used ResNet-18, ResNet-34 and ResNet-50 BID7 as reference networks.

We report their validation error in TAB2 .

In TAB2 , we compare LUT-Q against the published results using the INQ approach BID23 , which also trains networks with power-of-two weights.

We also compare with the baseline reported BID14 which correspond the best results from the literature for each weight and quantization configuration.

Note that we cannot directly compare the results of this appentrice method BID14 itself because they do not quantize the first and last layer of the ResNets.

We observe that LUT-Q always achieves better performance than other methods with the same weight and activation bitwidth except for ResNet-18 with 2-bit weight and 8-bit activation quantization.

Remarkably, ResNet-50 with 2-bit weights and 8-bit activations achieves 26.9% error rate which is only 1.0% worse than the baseline.

The memory footprint for parameters and activations of this network is only 7.4MB compared to 97.5MB for the full precision network.

Furthermore, the number of multiplications is reduced by two orders of magnitude and most of them can be replaced by bit-shifts.

Finally, we evaluated LUT-Q on the Pascal VOC BID4 object detection task.

We use our implementation of YOLOv2 BID18 as baseline.

This network has a memory footprint of 200MB and achieves a mean average precision (mAP) of 72% on Pascal VOC.

We were able to reduce the total memory footprint by a factor of 20 while maintaining the mAP above 70% by carrying out several modifications: replacing the feature extraction network with traditional residual networks BID7 , replacing the convolution layers by factorized convolutions BID4 , and finally applying LUT-Q in order to quantize the weights of the network to 8-bit.

Using LUT-Q with 4-bit quantization we are able to further reduce the total memory footprint down to just 1.72MB and still achieve a mAP of about 64%.

DISPLAYFORM0 Step 2: Compute current cost and gradients DISPLAYFORM1 // Step 3: Update full precision weights (here: SGD) DISPLAYFORM2 end for // Step 4: Update weight tying by M k-means DISPLAYFORM3 end for end for end for

Different compression methods were proposed in the past in order to reduce the memory footprint and the computational requirements of DNNs: pruning BID5 BID11 , quantization BID1 BID6 BID22 , teacherstudent network training BID8 BID14 BID17 BID19 are some examples.

In general, we can classify the methods for quantization of the parameters of a neural network into three types:• Soft weight sharing: These methods train the full precision weights such that they form clusters and therefore can be more efficiently quantized BID0 BID1 BID13 BID16 BID22 ].•

Fixed quantization: These methods choose a dictionary of values beforehand to which the weights are quantized.

Afterwards, they learn the assignments of each weight to the dictionary entries.

Examples are Binary Neural Networks BID3 , Ternary Weight Networks BID12 and also BID14 BID15 .•

Trained quantization: These methods learn a dictionary of values to which weights are quantized during training.

However, the assignment of each weight to a dictionary entry is fixed BID6 .Our LUT-Q approach takes the best of the latter two methods: For each layer, we jointly update both dictionary and weight assignments during training.

This approach to compression is similar to Deep Compression BID6 in the way that we learn a dictionary and assign each weight in a layer to one of the dictionary's values using the k-means algorithm, but we update iteratively both assignments and dictionary at each mini-batch iteration.

We have presented look-up table quantization, a novel approach for the reduction of size and computations of deep neural networks.

After each minibatch update, the quantization values and assignments are updated by a clustering step.

We show that the LUT-Q approach can be efficiently used for pruning weight matrices and training multiplier-less networks as well.

We also introduce a new form of batch normalization that avoids the need for multiplications during inference.

As argued in this paper, if weights are quantized to very low bitwidth, the activations may dominate the memory footprint of the network during inference.

Therefore, we perform our experiments with activations quantized uniformly to 8-bit.

We believe that a non-uniform activation quantization, where the quantization values are learned parameters, will help quantize activations to lower precision.

This is one of the promising directions for continuing this work.

Recently, several papers have shown the benefits of training quantized networks using a distillation strategy BID8 BID14 .

Distillation is compatible with our training approach and we are planning to investigate LUT-Q training together with distillation.

From BID9 we know that the traditional batch normalization (BN) at inference time for the oth output is DISPLAYFORM0 where x and y are the input and output vectors to the BN layer, γ and β are parameters learned during training, E [x] and VAR [x] are the running mean and variance of the input samples, and ǫ is a small constant to avoid numerical problems.

During inference, γ, β, E [x] and VAR [x] are constant and, therefore, the BN function (1) can be written as DISPLAYFORM1 where we use the scale DISPLAYFORM2 In order to obtain a multiplier-less BN, we require a to be a vector of powers-of-two during inference.

This can be achieved by quantizing γ toγ.

The quantizedγ is learned with the same idea as for WT: During the forward pass, we use traditional BN with the quantizedγ =â/ VAR[x] + ǫ whereâ is obtained from a by using the power-of-two quantization.

Then, in the backward pass, we update the full precision γ.

Please note that the computations during training time are not multiplier-less but γ is only learned such that we obtain a multiplier-less BN during inference time.

This is different to BID2 which proposed a shift-based batch normalization using a different scheme that avoids all multiplications in the batch normalization operation by rounding multiplicands to powers-of-two in each forward pass.

Their focus is on speeding up training by avoiding multiplications during training time, while our novel multiplier-less batch normalization approach avoids multiplications during inference.

<|TLDR|>

@highlight

In this paper we introduce a training method, called look-up table quantization (LUT-Q), which learns a dictionary and assigns each weight to one of the dictionary's values
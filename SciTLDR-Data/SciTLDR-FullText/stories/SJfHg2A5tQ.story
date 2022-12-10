Deep neural networks (DNN) are widely used in many applications.

However, their deployment on edge devices has been difficult because they are resource hungry.

Binary neural networks (BNN) help to alleviate the prohibitive resource requirements of DNN, where both activations and weights are limited to 1-bit.

We propose an improved binary training method (BNN+), by introducing a regularization function that encourages training weights around binary values.

In addition to this, to enhance model performance we add trainable scaling factors to our regularization functions.

Furthermore, we use an improved approximation of the derivative of the sign activation function in the backward computation.

These additions are based on linear operations that are easily implementable into the binary training framework.

We show experimental results on CIFAR-10 obtaining an accuracy of 86.5%, on AlexNet and 91.3% with VGG network.

On ImageNet, our method also outperforms the traditional BNN method and XNOR-net, using AlexNet by a margin of 4% and 2% top-1 accuracy respectively.

Deep neural networks (DNNs) have demonstrated success for many supervised learning tasks ranging from voice recognition to object detection BID26 BID11 .

The focus has been on increasing accuracy, in particular for image tasks, where deep convolutional neural networks (CNNs) are widely used.

However, their increasing complexity poses a new challenge, and has become an impediment to widespread deployment in many applications; specifically when trying to deploy such models to resource constrained and lower-power devices.

A typical DNN architecture contains tens to thousands of layers, resulting in millions of parameters.

As an example, AlexNet BID16 requires 200MB of memory, VGGNet BID26 requires 500MB memory.

Large model sizes are further exacerbated by their computational cost, requiring GPU implementation to allow real-time inference.

Such requirements evidently cannot be accustomed by edge devices as they have limited memory, computation power, and battery.

This motivated the community to investigate methods for compressing and reducing computation cost of DNNs.

To make DNNs compatible with the resource constraints of low power devices, there have been several approaches developed, such as network pruning BID17 , architecture design BID25 , and quantization BID0 BID4 .

In particular, weight compression using quantization can achieve very large savings in memory, where binary (1-bit), and ternary (2-bit) approaches have been shown to obtain competitive accuracy BID10 BID31 BID29 .

Using such schemes reduces model sizes by 8x to 32x depending on the bit resolution used for computations.

In addition to this, the speed by quantizing the activation layers.

In this way, both the weights and activations are quantized so that one can replace the expensive dot products and activation function evaluations with bitwise operations.

This reduction in bit-width benefits hardware accelerators such as FPGAs and neural network chips.

An issue with using low-bit DNNs is the drastic drop in accuracy compared to its full precision counterpart, and this is made even more severe upon quantizing the activations.

This problem is largely due to noise and lack of precision in the training objective of the networks during back-propagation BID19 .

Although, quantizing the weights and activations have been attracting large interests thanks to their computational benefits, closing the gap in accuracy between the full precision and the quantized version remains a challenge.

Indeed, quantizing weights cause drastic information loss and make neural networks harder to train due to a large number of sign fluctuations in the weights.

Therefore, how to control the stability of this training procedure is of high importance.

In theory, it is infeasible to back-propagate in a quantized setting as the weights and activations employed are discontinuous and discrete.

Instead, heuristics and approximations are proposed to match the forward and backward passes.

Often weights at different layers of DNNs follow a certain structure.

How to quantize the weights locally, and maintaining a global structure to minimize a common cost function is important BID18 .Our contribution consists of three ideas that can be easily implemented in the binary training framework presented by BID10 to improve convergence and generalization accuracy of binary networks.

First, the activation function is modified to better approximate the sign function in the backward pass, second we propose two regularization functions that encourage training weights around binary values, and lastly a scaling factor is introduced in both the regularization term as well as network building blocks to mitigate accuracy drop due to hard binarization.

Our method is evaluated on CIFAR-10 and ImageNet datasets and compared to other binary methods.

We show accuracy gains to traditional binary training.

We focus on challenges present in training binary networks.

The training procedure emulates binary operations by restricting the weights and activations to single-bit so that computations of neural networks can be implemented using arithmetic logic units (ALU) using XNOR and popcount operations.

More specifically, XNOR and popcount instructions are readily available on most CPU and GPU processing units.

Custom hardware would have to be implemented to take advantage of operations with higher bits such as 2 to 4 bits.

The goal of this binary training is to reduce the model size and gain inference speedups without performance degradation.

Primary work done by BID0 (BinaryConnect) trains deep neural networks with binary weights {−1, +1}. They propose to quantize real values using the sign function.

The propagated gradient applies update to weights |w| ≤ 1.

Once the weights are outside of this region they are no longer updated, this is done by clipping weights between {−1, +1}. In that work, they did not consider binarizing the activation functions.

BNN BID10 ) is the first purely binary network quantizing both the weights and activations.

They achieve comparable accuracy to their prior work on BinaryConnect, and achieve significantly close performance to full-precision, by using large and deep networks.

Although, they performed poorly on large datasets like ImageNet BID24 .

The resulting network presented in their work obtains 32× compression rate and approximately 7× increase in inference speed.

To alleviate the accuracy drop of BNN on larger datasets, BID23 proposed XNORNet, where they strike a trade-off between compression and accuracy through the use of scaling factors for both weights and activation functions.

In their work, they show performance gains compared to BNN on ImageNet classification.

The scaling factors for both the weights and activations are computed dynamically, which slows down training performance.

Further, they introduced an additional complexity in implementing the convolution operations on the hardware, slightly reducing compression rate and speed up gains.

DoReFa-Net BID30 further improves XNOR-Net by approximating the activations with more bits.

The proposed rounding mechanism allows for low bit back-propagation as well.

Although they perform multi-bit quantization, their model still suffers from large accuracy drop upon quantizing the last layer.

Later in ABC-Net, BID29 propose several strategies, adjusting the learning rate for larger datasets.

They show BNN achieve similar accuracy as XNOR-Net without the scaling overhead by adding a regularizer term which allows binary networks to generalize better.

They also suggest a modified BNN, where they adopted the strategy of increasing the number of filters, to compensate for accuracy loss similar to wide reduced-precision networks BID21 .

More recently, developed a second-order approximation to the sign activation function for a more accurate backward update.

In addition to this, they pre-train the network in which they want to binarize in full precision using the hard tangent hyperbolic (htanh) activation, see FIG0 .

They use the pre-trained network weights as an initialization for the binary network to obtain state of the art performance.

Training a binary neural network faces two major challenges: on weights, and on activation functions.

As both weights and activations are binary, the traditional continuous optimization methods such as SGD cannot be directly applied.

Instead, a continuous approximation is used for the sign activation during the backward pass.

Further, the gradient of the loss with respect to the weights are small.

So as training progresses weight sign remains unchanged.

These are both addressed in our proposed method.

In this section, we present our approach to training 1-bit CNNs in detail.

We quickly revisit quantization through binary training as first presented by BID0 .

In BID10 , the weights are quantized by using the sign function which is +1 if w > 0 and −1 otherwise.

In the forward pass, the real-valued weights are binarized to w b , and the resulting loss is computed using binary weights throughout the network.

For hidden units, the sign function non-linearity is used to obtain binary activations.

Prior to binarizing, the real weights are stored in a temporary variable w.

The variables w are stored because one cannot back-propagate through the sign operation as its gradient is zero everywhere, and hence disturbs learning.

To alleviate this problem the authors suggest using a straight through estimator BID7 for the gradient of the sign function.

This method is a heuristic way of approximating the gradient of a neuron, DISPLAYFORM0 where L is the loss function and 1(.) is the indicator function.

The gradients in the backward pass are then applied to weights that are within [−1, +1].

The training process is summarized in Figure 1 .

As weights undergo gradient updates, they are eventually pushed out of the center region and instead make two modes, one at −1 and another at +1.

This progression is also shown in Figure 1 .

DISPLAYFORM1 Figure 1: Binary training, where arrows indicate operands flowing into operation or block.

Reproduced from Guo (2018) (left).

A convolutional layer depicting weight histogram progression during the popular binary training.

The initial weight distribution is a standard Gaussian (right).

Our first modification is on closing the discrepancy between the forward pass and backward pass.

Originally, the sign derivative is approximated using the htanh(x) activation, as in FIG0 .

Instead, we modify the Swish-like activation BID22 BID1 BID6 , where it has shown to outperform other activations on various tasks.

The modifications are performed by taking its derivative and centering it around 0 DISPLAYFORM0 where σ(z) is the sigmoid function and the scale β > 0 controls how fast the activation function asymptotes to −1 and +1.

The β parameter can be learned by the network or be hand-tuned as a hyperparameter.

As opposed to the Swish function, where it is unbounded on the right side, the modification makes it bounded and a valid approximator of the sign function.

As a result, we call this activation SignSwish, and its gradient is DISPLAYFORM1 which is a closer approximation function compared to the htanh activation.

Comparisons are made in FIG0 .

BID10 noted that the STE fails to learn weights near the borders of −1 and +1.

As depicted in FIG0 , our proposed SignSwish activation alleviates this, as it remains differentiable near −1 and +1 allowing weights to change signs during training if necessary.

Note that the derivative d dx SS β (x) is zero at two points, controlled by β.

Indeed, it is simple to show that the derivative is zero for x ≈ ±2.4/β.

By adjusting this parameter beta, it is possible to adjust the location at which the gradients start saturating.

In contrast to the STE estimators, where it is fixed.

Thus, the larger β is, the closer the approximation is to the derivative of the sign function.

DISPLAYFORM2

In general, a regularization term is added to a model to prevent over-fitting and to obtain robust generalization.

The two most commonly used regularization terms are L 1 and L 2 norms.

If one were to embed these regularization functions in binary training, it would encourage the weights to be near zero; though this does not align with the objective of a binary network.

Instead, it is important to define a function that encourages the weights around −1 and +1.

Further, in BID23 they present a scale to enhance the performance of binary networks.

This scale is computed dynamically during training, using the statistics of the weights.

To make the regularization term more general we introduce scaling factors α, resulting in a symmetric regularization function with two minimums, one at −α and another at +α.

As these scales are introduced in the regularization function and are embedded into the layers of the network they can be learned using backpropagation.

The Manhattan regularization function is defined as DISPLAYFORM0 whereas the Euclidean version is defined as DISPLAYFORM1 where α > 0 is the scaling factor.

As depicted in Figure 3 , in the case of α = 1 the weights are penalized at varying degrees upon moving away from the objective quantization values, in this case, {−1, +1}.The proposed regularizing terms are inline with the wisdom of the regularization function R(w) = (1 − w 2 )1 {|w|≤1} as introduced in BID29 .

A primary difference are in introducing a trainable scaling factor, and formulating it such that the gradients capture appropriate sign updates to the weights.

Further, the regularization introduced in BID29 does not penalize weights that are outside of [−1, +1].

One can re-define their function to include a scaling factor as R(w) = (α − w 2 )1 {|w|≤α} .

In Figure 3 , we depict the different regularization terms to help with intuition.−3 −2 −1 1 2 3

−3 −2 −1 1 2 3

Figure 3: R 1 (w) (left) and R 2 (w) (right) regularization functions for α = 0.5 (solid line) and α = 1 (dashed line).

The scaling factor α is trainable, as a result the regularization functions can adapt accordingly.

Combining both the regularization and activation ideas, we modify the training procedure by replacing the sign backward approximation with that of the derivative of SS β activation (2).

During training, the real weights are no longer clipped as in BNN training, as the network can back-propagate through the SS β activation and update the weights correspondingly.

Additional scales are introduced to the network, which multiplies into the weights of the layers.

The regularization terms introduced are then added to the total loss function, DISPLAYFORM0 where L(W, b) is the cost function, W and b are the sets of all weights and biases in the network, W l is the set weights at layer l and α l is the corresponding scaling factor.

Here, R(.) is the regularization function (4) or (5).

Further, λ controls the effect of the regularization term.

To introduce meaningful scales, they are added to the basic blocks composing a typical convolutional neural network.

For example, for convolutions, the scale is multiplied into the quantized weights prior to the convolution operation.

Similarly, in a linear layer, the scales are multiplied into the quantized weights prior to the dot product operation.

This is made more clear in the training algorithm 1.The scale α is a single scalar per layer, or as proposed in BID23 is a scalar for each filter in a convolutional layer.

For example, given a CNN block with weight dimensionality (C in , C out , H, W ), where C in is the number of input channels, C out is the number of output channels, and H, W , the height and width of the filter respectively, then the scale parameter would be a vector of dimension C out , that factors into each filter.

As the scales are learned jointly with the network through backpropagation, it is important to initialize them appropriately.

In the case of the Manhattan penalizing term (4), given a scale factor α and weight filter then the objective is to minimize DISPLAYFORM1 The minimum of the above is obtained when DISPLAYFORM2 Similarly, in the case of the Euclidean penalty (5) the minimum is obtained when α * = mean(|W |) (9) The scales are initialized with the corresponding optimal values after weights have been initialized first.

One may notice the similarity of these optimal values with that defined by BID23 , whereas in their case the optimal value for the weight filters and activations better matches the R 2 (w) goal.

A difference is on how these approximations are computed, in our case they are updated on the backward pass, as opposed to computing the values dynamically.

The final resulting BNN+ training method is defined in Algorithm 1.

In the following section, we present our experimental results and important training details.

Algorithm 1 BNN+ training.

L is the unregularized loss function.λ and R 1 are the regularization terms we introduced.

SS β is the SignSwish function we introduced and (SS β ) is its derivative.

N is the number of layers.• indicates element-wise multiplication.

BatchNorm() specifies how to batchnormalize the activation and BackBatchNorm() how to back-propagate through the normalization.

ADAM() specifies how to update the parameters when their gradients are known.

Require: a minibatch of inputs and targets (x 0 , x * ), previous weights W , previous weights' scaling factors α, and previous BatchNorm parameters θ.

Ensure: updated weights W t+1 , updated weights' scaling factors α t+1 and updated BatchNorm parameters θ t+1 .

{1.

Forward propagation:} s 0 ← x 0 W 0 {We do not quantize the first layer.} DISPLAYFORM3 {We use our modified straight-through estimator to back-propagate through sign: DISPLAYFORM4

We evaluate our proposed method with the accuracy performance of training using BNN+ scheme versus other proposed binary networks, BID10 ; BID23 ; BID29 .

We run our method on CIFAR-10 and ImageNet datasets and show accuracy gains.

They are discussed in their respective sections below.

The CIFAR-10 data BID15 ) consists of 50,000 train images and a test set of 10,000.

For pre-processing the images are padded by 4 pixels on each side and a random crop is taken.

We train both, AlexNet BID16 , and VGG BID26 using the ADAM (Kingma & Ba, 2014) optimizer.

The architecture used for VGG is conv(256) → conv(256) → conv(512) → conv(512) → conv(1024) → conv(1024) → fc(1024) → fc FORMULA2 where conv(·) is a convolutional layer, and fc(·) is a fully connected layer.

The standard 3 × 3 filters are used in each layer.

We also add a batch normalization layer BID12 prior to activation.

For AlexNet, the architecture from BID14 is used, and batch normalization layers are added prior to activations.

We use a batch size of 256 for training.

Many learning rates were experimented with such as 0.1, 0.03, 0.001, etc, and the initial learning rate for AlexNet was set to 10 −3 , and 3 × 10 −3 for VGG.

The learning rates are correspondingly reduced by a factor 10 every 10 epoch for 50 epochs.

We set the regularization parameter λ to 10 −6 , and use the regularization term as defined in (4).

In these experiments weights are initialized using BID2 initialization.

Further, the scales are introduced for each convolution filter, and are initialized by sorting the absolute values of the weights for each filter and choosing the 75 th percentile value.

The results are summarized in TAB0 .

The ILSVRC-2012 dataset consists of ∼ 1.2M training images, and 1000 classes.

For pre-processing the dataset we follow the typical augmentation: the images are resized to 256 × 256, then are randomly cropped to 224 × 224 and the data is normalized using the mean and standard deviation statistics of the train inputs; no additional augmentation is done.

At inference time, the images are first scaled to 256 × 256, center cropped to 224 × 224 and then normalized.

We evaluate the performance of our training method on two architectures AlexNet and Resnet-18 .

Following previous work, we used batch-normalization before each activation function.

Additionally, we keep the first and last layers to be in full precision, as we lose 2-3% accuracy otherwise.

This approach is followed by other binary methods that we compare to BID10 BID23 BID29 .

The results are summarized in TAB1 .

In all the experiments involving R 1 regularization we set the λ to 10 −7 and R 2 regularization to 10 −6 .

Also, in every network, the scales are introduced per filter in convolutional layers, and per column in fully connected layers.

The weights are initialized using a pre-trained model with htan activation function as done in .

Then the learning rate for AlexNet is set to 2.33 × 10 − 3 and multiplied by 0.1 at the 12 th , 18 th epoch for a total of 25 epochs trained.

For the 18-layer ResNet the learning rate is started from 0.01 and multiplied by 0.1 at 10 th , 20 th , 30 th epoch.

We proposed two regularization terms (4) and (5) and an activation term (2) with a trainable parameter β.

We run several experiments to better understand the effect of the different modifications to the training method, especially using different regularization and asymptote parameters β.

The parameter β is trainable and would add one equation through back-propagation.

However, we fixed β throughout our experiments to explicit values.

The results are summarized in TAB1 .Through our experiments, we found that adding regularizing term with heavy penalization degrades the networks ability to converge, as the term would result in total loss be largely due to the regu- larizing term and not the target cross entropy loss.

Similarly, the regularizing term was set to small values in BID29 .

As a result, we set λ with a reasonably small value 10 −5 − 10 −7 , so that the scales move slowly as the weights gradually converge to stable values.

Some preliminary experimentation was to gradually increase the regularization with respect to batch iterations updates done in training, though this approach requires careful tuning and was not pursued further.

From TAB1 , and referring to networks without regularization, we see the benefit of using SwishSign approximation versus the STE.

This was also noted in , where their second approximation provided better results.

There is not much difference between using R 1 versus R 2 towards model generalization although since the loss metric used was the cross-entropy loss, the order of R 1 better matches the loss metric.

Lastly, it seems moderate values of β is better than small or large values.

Intuitively, this happens because for small values of β, the gradient approximation is not good enough and as β increases the gradients become too large, hence small noise could cause large fluctuations in the sign of the weights.

We did not compare our network with that of as they introduce a shortcut connection that proves to help even the full precision network.

As a final remark, we note that the learning rate is of great importance and properly tuning this is required to achieve convergence.

Table 3 summarizes the best results of the ablation study and compares with BinaryNet, XNOR-Net, and ABC-Net.

Table 3 : Comparison of top-1 and top-5 accuracies of our method BNN+ with BinaryNet, XNORNet and ABC-Net on ImageNet, summarized from TAB1 .

The results of BNN, XNOR, & ABC-Net are reported from the corresponding papers BID23 BID10 BID29 .

Results for ABC-NET on AlexNet were not available, and so is not reported.

To summarize we propose three incremental ideas that help binary training: i) adding a regularizer to the objective function of the network, ii) trainable scale factors that are embedded in the regularizing term and iii) an improved approximation to the derivative of the sign activation function.

We obtain competitive results by training AlexNet and Resnet-18 on the ImageNet dataset.

For future work, we plan on extending these to efficient models such as CondenseNet BID9 , MobileNets BID8 , MnasNet BID28 and on object recognition tasks.

@highlight

The paper presents an improved training mechanism for obtaining binary networks with smaller accuracy drop that helps close the gap with it's full precision counterpart
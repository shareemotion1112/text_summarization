Making deep convolutional neural networks more accurate typically comes at the cost of increased computational and memory resources.

In this paper, we reduce this cost by exploiting the fact that the importance of features computed by convolutional layers is highly input-dependent, and propose feature boosting and suppression (FBS), a new method to predictively amplify salient convolutional channels and skip unimportant ones at run-time.

FBS introduces small auxiliary connections to existing convolutional layers.

In contrast to channel pruning methods which permanently remove channels, it preserves the full network structures and accelerates convolution by dynamically skipping unimportant input and output channels.

FBS-augmented networks are trained with conventional stochastic gradient descent, making it readily available for many state-of-the-art CNNs.

We compare FBS to a range of existing channel pruning and dynamic execution schemes and demonstrate large improvements on ImageNet classification.

Experiments show that FBS can respectively provide 5× and 2× savings in compute on VGG-16 and ResNet-18, both with less than 0.6% top-5 accuracy loss.

State-of-the-art vision and image-based tasks such as image classification BID19 BID33 BID11 , object detection BID31 and segmentation BID26 are all built upon deep convolutional neural networks (CNNs).

While CNN architectures have evolved to become more efficient, the general trend has been to use larger models with greater memory utilization, bandwidth and compute requirements to achieve higher accuracy.

The formidable amount of computational resources used by CNNs present a great challenge in the deployment of CNNs in both cost-sensitive cloud services and low-powered edge computing applications.

One common approach to reduce the memory, bandwidth and compute costs is to prune over-parameterized CNNs.

If performed in a coarse-grain manner this approach is known as channel pruning BID37 BID25 BID35 .

Channel pruning evaluates channel saliency measures and removes all input and output connections from unimportant channels-generating a smaller dense model.

A saliency-based pruning method, however, has threefold disadvantages.

Firstly, by removing channels, the capabilities of CNNs are permanently lost, and the resulting CNN may never regain its accuracy for difficult inputs for which the removed channels were responsible.

Secondly, despite the fact that channel pruning may drastically shrink model size, without careful design, computational resources cannot be effectively reduced in a CNN without a detrimental impact on its accuracy.

Finally, the saliency of a neuron is not static, which can be illustrated by the feature visualization in FIG0 .

Here, a CNN is shown a set of input images, certain channel neurons in a convolutional output may get highly excited, whereas another set of images elicit little response from the same channels.

This is in line with our understanding of CNNs that neurons in a convolutional layer specialize in recognizing distinct features, and the relative importance of a neuron depends heavily on the inputs.

The above shortcomings prompt the question: why should we prune by static importance, if the importance is highly input-dependent?

Surely, a more promising alternative is to prune dynamically depending on the current input.

A dynamic channel pruning strategy allows the network to learn to prioritize certain convolutional channels and ignore irrelevant ones.

Instead of simply reducing model size at the cost of accuracy with pruning, we can accelerate convolution by selectively computing only a subset of channels predicted to be important at run-time, while considering the sparse input from the preceding convolution layer.

In effect, the amount of cached activations and the number of read, write and arithmetic operations used by a well-designed dynamic model can be almost identical to an equivalently sparse statically pruned one.

In addition to saving computational resources, a dynamic model preserves all neurons of the full model, which minimizes the impact on task accuracy.

In this paper, we propose feature boosting and suppression (FBS) to dynamically amplify and suppress output channels computed by the convolutional layer.

Intuitively, we can imagine that the flow of information of each output channel can be amplified or restricted under the control of a "valve".

This allows salient information to flow freely while we stop all information from unimportant channels and skip their computation.

Unlike pruning statically, the valves use features from the previous layer to predict the saliency of output channels.

With conventional stochastic gradient descent (SGD) methods, the predictor can learn to adapt itself by observing the input and output features of the convolution operation.

FBS introduces tiny auxiliary connections to existing convolutional layers.

The minimal overhead added to the existing model is thus negligible when compared to the potential speed up provided by the dynamic sparsity.

Existing dynamic computation strategies in CNNs BID28 BID2 produce on/off pruning decisions or execution path selections.

Training them thus often resorts to reinforcement learning, which in practice is often computationally expensive.

Even though FBS similarly use non-differentiable functions, contrary to these methods, the unified losses are still wellminimized with conventional SGD.We apply FBS to a custom CIFAR-10 ( BID20 classifier and popular CNN models such as VGG-16 BID33 and ResNet-18 (He et al., 2016) trained on the ImageNet dataset BID3 .

Empirical results show that under the same speed-ups, FBS can produce models with validation accuracies surpassing all other channel pruning and dynamic conditional execution methods examined in the paper.

BID11 , the outputs from certain channel neurons may vary drastically.

The top rows in (a) and (b) are found respectively to greatly excite neurons in channels 114 and 181 of layer block 3b/conv2, whereas the bottom images elicit little activation from the same channel neurons.

The number below each image indicate the maximum values observed in the channel before adding the shortcut and activation.

Finally, (c) shows the distribution of maximum activations observed in the first 20 channels.

Since BID21 introduced optimal brain damage, the idea of creating more compact and efficient CNNs by removing connections or neurons has received significant attention.

Early literature on pruning deep CNNs zero out individual weight parameters BID9 BID7 .

This results in highly irregular sparse connections, which were notoriously difficult for GPUs to exploit.

This has prompted custom accelerator solutions that exploit sparse weights BID29 BID8 .

Although supporting both sparse and dense convolutions efficiently normally involves some compromises in terms of efficiency or performance.

Alternatively, recent work has thus increasingly focused on introducing structured sparsity BID35 BID37 BID1 BID39 , which can be exploited by GPUs and allows custom accelerators to focus solely on efficient dense operations.

BID35 added group Lasso on channel weights to the model's training loss function.

This has the effect of reducing the magnitude of channel weights to diminish during training, and remove connections from zeroed-out channels.

To facilitate this process, Alvarez & Salzmann (2016) additionally used proximal gradient descent, while and BID12 proposed to prune channels by thresholds, i.e. they set unimportant channels to zero, and fine-tune the resulting CNN.

The objective to induce sparsity in groups of weights may present difficulties for gradient-based methods, given the large number of weights that need to be optimized.

A common approach to overcome this is to solve or learn BID25 BID37 channel saliencies to drive the sparsification of CNNs.

solved an optimization problem which limits the number of active convolutional channels while minimizing the reconstruction error on the convolutional output.

BID25 used Lasso regularization on channel saliencies to induce sparsity and prune channels with a global threshold.

BID37 learned to sparsify CNNs with an iterative shrinkage/thresholding algorithm applied to the scaling factors in batch normalization.

There are methods BID27 BID40 ) that use greedy algorithms for channel selection.

and BID14 adopted reinforcement learning to train agents to produce channel pruning decisions.

PerforatedCNNs, proposed by BID6 , use predefined masks that are model-agnostic to skip the output pixels in convolutional layers.

In a pruned model produced by structured sparsity methods, the capabilities of the pruned neurons and connections are permanently lost.

Therefore, many propose to use dynamic networks as an alternative to structured sparsity.

During inference, a dynamic network can use the input data to choose parts of the network to evaluate.

Convolutional layers are usually spatially sparse, i.e. their activation outputs may contain only small patches of salient regions.

A number of recent publications exploit this for acceleration.

BID4 introduced low-cost collaborative layers which induce spatial sparsity in cheap convolutions, so that the main expensive ones can use the same sparsity information.

BID5 proposed spatially adaptive computation time for residual networks BID11 , which learns the number of residual blocks required to compute a certain spatial location.

BID0 presented dynamic capacity networks, which use the gradient of a coarse output's entropy to select salient locations in the input image for refinement.

BID30 assumed the availability of a priori spatial sparsity in the input image, and accelerated the convolutional layer by computing non-sparse regions.

There are dynamic networks that make binary decisions or multiple choices for the inference paths taken.

BlockDrop, proposed by , trains a policy network to skip blocks in residual networks.

BID24 proposed conditional branches in deep neural networks (DNNs), and used Q-learning to train the branching policies.

BID28 designed a DNN with layers containing multiple modules, and decided which module to use with a recurrent neural network (RNN).

learned an RNN to adaptively prune channels in convolutional layers.

The on/off decisions commonly used in these networks cannot be represented by differentiable functions, hence the gradients are not well-defined.

Consequently, the dynamic networks above train their policy functions by reinforcement learning.

There exist, however, methods that workaround such limitations.

BID32 introduced sparsely-gated mixture-of-experts and used a noisy ranking on the backpropagate-able gating networks to select the expensive experts to evaluate.

BID2 trained differentiable policy functions to implement early exits in a DNN.

BID15 learned binary policies that decide whether partial or all input channels are used for convolution, but approximate the gradients of the non-differentiable policy functions with continuous ones.

We start with a high-level illustration FIG1 ) of how FBS accelerates a convolutional layer with batch normalization (BN).

The auxiliary components (in red) predict the importance of each output channel based on the input features, and amplify the output features accordingly.

Moreover, certain output channels are predicted to be entirely suppressed (or zero-valued as represented by ), such output sparsity information can advise the convolution operation to skip the computation of these channels, as indicated by the dashed arrow.

It is notable that the expensive convolution can be doubly accelerated by skipping the inactive channels from both the input features and the predicted output channel saliencies.

The rest of this section provides detailed explanation of the components in FIG1 .subsample channel saliency predictor multiple winners take all wta wta convolution normalize bias) DISPLAYFORM0

For simplicity, we consider a deep sequential batch-normalized BID18 DISPLAYFORM0 which comprise of C l channels of features with height H l and width W l .

The l th layer is thus defined as: DISPLAYFORM1 Here, additions (+) and multiplications (·) are element-wise, (z) + = max (z, 0) denotes the ReLU activation, γ l , β l ∈ R C l are trainable parameters, norm (z) normalizes each channel of features z across the population of z, with µ z , σ 2 z ∈ R C l respectively containing the population mean and variance of each channel, and a small prevents division by zero: DISPLAYFORM2 Additionally, conv l (x l−1 , θ l ) computes the convolution of input features x l−1 using the weight tensor θ l ∈ R DISPLAYFORM3 functions, as a CNN spends the majority of its inference time in them, using k 2 C l−1 C l H l W l multiply-accumulate operations (MACs) for the l th layer.

Consider the following generalization of a layer with dynamic execution: DISPLAYFORM0 where f and π respectively use weight parameters θ and φ and may have additional inputs, and compute tensors of the same output shape, denoted by F and G. retrieves the c th feature image.

We can further sparsify and accelerate the layer by adding, for instance, a Lasso on π to the total loss, where E x [z] is the expectation of z over x: DISPLAYFORM1 Despite the simplicity of this formulation, it is however very tricky to designf properly.

Under the right conditions, we can arbitrarily minimize the Lasso while maintaining the same output from the layer by scaling parameters.

For example, in low-cost collaborative layers BID4 , f and π are simply convolutions (with or without ReLU activation) that respectively have weights θ and φ.

Since f and π are homogeneous functions, one can always halve φ and double θ to decrease (4) while the network output remains the same.

In other words, the optimal network must have φ ∞ → 0, which is infeasible in finiteprecision arithmetic.

For the above reasons, BID4 observed that the additional loss in (4) always degrades the CNN's task performance.

BID37 pointed out that gradient-based training algorithms are highly inefficient in exploring such reparameterization patterns, and channel pruning methods may experience similar difficulties.

BID32 avoided this limitation by finishing π with a softmax normalization, but (4) can no longer be used as the softmax renders the 1 -norm, which now evaluates to 1, useless.

In addition, similar to sigmoid, softmax (without the cross entropy) is easily saturated, and thus may equally suffer from vanishing gradients.

Many instead design π to produce on/off decisions and train them with reinforcement learning as discussed in Section 2.

Instead of imposing sparsity on features or convolutional weight parameters (e.g. BID35 ; Alvarez & Salzmann (2016); ; BID12 ), recent channel pruning methods BID25 BID37 induce sparsity on the BN scaling factors γ l .

Inspired by them, FBS similarly generates a channel-wise importance measure.

Yet contrary to them, instead of using the constant BN scaling factors γ l , we predict channel importance and dynamically amplify or suppress channels with a parametric function π(x l−1 ) dependent on the output from the previous layer x l−1 .

Here, we propose to replace the layer definition f l (x l−1 ) for each of l ∈ [1, L] withf l (x l−1 ) which employs dynamic channel pruning: DISPLAYFORM0 where a low-overhead policy π l (x l−1 ) evaluates the pruning decisions for the computationally demanding conv (x l−1 , θ l ): DISPLAYFORM1 Here, wta k (z) is a k-winners-take-all function, i.e. it returns a tensor identical to z, except that we zero out entries in z that are smaller than the k largest entries in absolute magnitude.

In other words, wta dC l (g l (x l−1 )) provides a pruning strategy that computes only dC l most salient channels predicted by g l (x l−1 ), and suppresses the remaining channels with zeros.

In Section 3.4, we provide a detailed explanation of how we design a cheap g l (x l−1 ) that learns to predict channel saliencies.

It is notable that our strategy prunes C l − dC l least salient output channels from l th layer, where the density d ∈ ]0, 1] can be varied to sweep the trade-off relationship between performance and accuracy.

Moreover, pruned channels contain all-zero values.

This allows the subsequent (l + 1) th layer to trivially make use of input-side sparsity, since all-zero features can be safely skipped even for zero-padded layers.

Because all convolutions can exploit both input-and output-side sparsity, the speed-up gained from pruning is quadratic with respect to the pruning ratio.

For instance, dynamically pruning half of the channels in all layers gives rise to a dynamic CNN that uses approximately 1 4 of the original MACs.

Theoretically, FBS does not introduce the reparameterization discussed in Section 3.2.

By batch normalizing the convolution output, the convolution kernel θ l is invariant to scaling.

Computationally, it is more efficient to train.

Many alternative methods use nondifferentiable π functions that produce on/off decisions.

In general, DNNs with these policy functions are incompatible with SGD, and resort to reinforcement learning for training.

In contrast, (6) allows end-to-end training, as wta is a piecewise differentiable and continuous function like ReLU.

BID34 suggested that in general, a network is easier and faster to train for complex tasks and less prone to catastrophic forgetting, if it uses functions such as wta that promote local competition between many subnetworks.

This section explains the design of the channel saliency predictor g l (x l−1 ).

To avoid significant computational cost in g l , we subsample x l−1 by reducing the spatial dimensions of each channel to a scalar using the following function ss : R C×H×W → R C : DISPLAYFORM0 where s x DISPLAYFORM1 reduces the c th channel of z to a scalar using, for instance, the 1 -norm DISPLAYFORM2 l−1 .

The results in Section 4 use the 1 -norm by default, which is equivalent to global average pooling for the ReLU activated x l−1 .

We then design g l (x l−1 ) to predict channel saliencies with a fully connected layer following the subsampled activations ss (x l−1 ), where φ l ∈ R C l ×C l−1 is the weight tensor of the layer: DISPLAYFORM3 We generally initialize ρ l with 1 and apply BID10 's initialization to φ l .

Similar to how BID25 and BID37 induced sparsity in the BN scaling factors, we regularize all layers with the Lasso on DISPLAYFORM4

We ran extensive experiments on CIFAR-10 ( BID20 and the ImageNet ILSVRC2012 BID3 , two popular image classification datasets.

We use MCifarNet BID38 , a custom 8-layer CNN for CIFAR-10 (see Appendix A for its structure), using only 1.3 M parameters with 91.37% and 99.67% top-1 and top-5 accuracies respectively.

M-CifarNet is much smaller than a VGG-16 on CIFAR-10 (Liu et al., 2017), which uses 20 M parameters and only 2.29% more accurate.

Because of its compactness, our CNN is more challenging to accelerate.

By faithfully reimplementing Network Slimming (NS) BID25 , we closely compare FBS with NS under various speedup constraints.

For ILSVRC2012, we augment two popular CNN variants, ResNet-18 BID11 and VGG-16 BID33 , and provide detailed accuracy/MACs trade-off comparison against recent structured pruning and dynamic execution methods.

Our method begins by first replacing all convolutional layer computations with FORMULA7 , and initializing the new convolutional kernels with previous parameters.

Initially, we do not suppress any channel computations by using density d = 1 in (6) and fine-tune the resulting network.

For fair comparison against NS, we then follow BID25 by iteratively decrementing the overall density d of the network by 10% in each step, and thus gradually using fewer channels to sweep the accuracy/performance trade-off.

The difference is that NS prunes channels by ranking globally, while FBS prunes around 1 − d of each layer.

By respectively applying NS and FBS to our CIFAR-10 classifier and incrementally increasing sparsity, we produce the trade-off relationships between number of operations (measured in MACs) and the classification accuracy as shown in FIG3 .

FBS clearly surpasses NS in its ability to retain the task accuracy under an increasingly stringent computational budget.

Besides comparing FBS against NS, we are interested in combining both methods, which demonstrates the effectiveness of FBS if the model is already less redundant, i.e. it cannot be pruned further using NS without degrading the accuracy by more than 1%.

The composite method (NS+FBS) is shown to successfully regain most of the lost accuracy due to NS, producing a trade-off curve closely matching FBS.

It is notable that under the same 90.50% accuracy constraints, FBS, NS+FBS, and NS respectively achieve 3.93×, 3.22×, and 1.19× speed-up ratios.

Conversely for a 2× speed-up target, they respectively produce models with accuracies not lower than 91.55%, 90.90% and 87.54%.

FIG3 demonstrates that our FBS can effectively learn to amplify and suppress channels when dealing with different input images.

The 8 heat maps respectively represent the channel skipping probabilities of the 8 convolutional layers.

The brightness of the pixel at location (x, y) denotes the probability of skipping the x th channel when looking at an image of the y th category.

The heat maps verify our belief that the auxiliary network learned to predict which channels specialize to which features, as channels may have drastically distinct probabilites of being used for images of different categories.

The model here is a M-CifarNet using FBS with d = 0.5, which has a top-1 accuracy of 90.59% (top-5 99.65%).

Moreover, channels in the heat maps are sorted so the channels that are on average least frequently evaluated are placed on the left, and channels shaded in stripes are never evaluated.

The network in FIG3 is not only approximately 4× faster than the original, by removing the unused channels, we also reduce the number of weights by 2.37×.

This reveals that FBS naturally subsumes channel pruning strategies such as NS, as we can simply prune away channels that are skipped regardless of the input.

It is notable that even though we specified a universal density d, FBS learned to adjust its dynamicity across all layers, and prune different ratios of channels from the convolutional layers.

By applying FBS and NS respectively to ResNet-18, we saw that the ILSVRC2012 validation accuracy of FBS consistently outperforms NS under different speed-up constraints (see Appendix B for the implementation details and trade-off curves).

For instance, at d = 0.7, it utilizes only 1.12 G MACs (1.62× fewer) to achieve a top-1 error rate of 31.54%, while NS requires 1.51 G MACs (1.21× fewer) for a similar error rate of 31.70%.

When compared across recent dynamic execution methods examined in TAB2 , FBS demonstrates simultaneously the highest possible speed-up and the lowest error rates.

It is notable that the baseline accuracies for FBS refer to a network that has been augmented with the auxiliary layers featuring FBS but suppress no channels, i.e. d = 1.

We found that this method brings immediate accuracy improvements, an increase of 1.73% in top-1 and 0.46% in top-5 accuracies, to the baseline network, which is in line with our observation on M-CifarNet.

In TAB3 , we compare different structured pruning and dynamic execution methods to FBS for VGG-16 (see Appendix B for the setup).

At a speed-up of 3.01×, FBS shows a minimal increase of 0.44% and 0.04% in top-1 and top-5 errors respectively.

At 5.23× speed-up, it only degrades the top-1 error by 1.08% and the top-5 by 0.59%.Not only does FBS use much fewer MACs, it also demonstrates significant reductions in bandwidth and memory requirements.

In Table 3 , we observe a large reduction in the number of memory accesses in single image inference as we simply do not access suppressed weights and activations.

Because these memory operations are often costly DRAM accesses, minimizing them leads to power-savings.

Table 3 further reveals that in diverse application scenarios such as low-end and cloud environments, the peak memory usages by the optimized models are much smaller than the originals, which in general improves cache utilization.

Filter Pruning , reproduced by ) -8.6 14.6 Perforated CNNs BID6 3.7 5.5 -Network Slimming BID25 , our implementation) 1.37 3.26 5.18 Runtime Neural Pruning 2.32 3.23 3.58 Channel Pruning 0.0 1.0 1.7 AutoML for Model Compression BID14 --1.4 ThiNet-Conv BID27 0.37 --Feature Boosting and Suppression (FBS) 0.04 0.52 0.59 Table 3 : Comparisons of the memory accesses and peak memory usage of the ILSVRC2012 classifiers with FBS respectively under 3× and 2× inference speed-ups.

The Weights and Activations columns respectively show the total amount of weight and activation accesses required by all convolutions for a single image inference.

The Peak Memory Usage columns show the peak memory usages with different batch sizes.

In summary, we proposed feature boosting and suppression that helps CNNs to achieve significant reductions in the compute required while maintaining high accuracies.

FBS fully preserves the capabilities of CNNs and predictively boosts important channels to help the accelerated models retain high accuracies.

We demonstrated that FBS achieves around 2× and 5× savings in computation respectively on ResNet-18 and VGG-16 within 0.6% loss of top-5 accuracy.

Under the same performance constraints, the accuracy gained by FBS surpasses all recent structured pruning and dynamic execution methods examined in this paper.

In addition, it can serve as an off-the-shelf technique for accelerating many popular CNN networks and the fine-tuning process is unified in the traditional SGD which requires no algorithmic changes in training.

Finally, the implementation of FBS and the optimized networks are fully open source and released to the public 1 .

For the CIFAR-10 classification task, we use M-CifarNet, a custom designed CNN, with less than 1.30 M parameters and takes 174 M MACs to perform inference for a 32-by-32 RGB image.

The architecture is illustrated in Table 4 , where all convolutional layers use 3 × 3 kernels, the Shape column shows the shapes of each layer's features, and pool7 is a global average pooling layer.

We trained M-CifarNet (see Appendix A) with a 0.01 learning rate and a 256 batch size.

We reduced the learning rate by a factor of 10× for every 100 epochs.

To compare FBS against NS fairly, every model with a new target MACs budget were consecutively initialized with the previous model, and trained for a maximum of 300 epochs, which is enough for all models to converge to the best obtainable accuracies.

For NS, we follow BID25 and start training with an 1 -norm sparsity regularization weighted by 10 −5 on the BN scaling factors.

We then prune at 150 epochs and fine-tune the resulting network without the sparsity regularization.

We additionally employed image augmentation procedures from BID19 to preprocess each training example.

Each CIFAR-10 example was randomly horizontal flipped and slightly perturbed in the brightness, saturation and hue.

Table 4 additionally provides further comparisons of layer-wise compute costs between FBS, NS, and the composition of the two methods (NS+FBS).

It is notable that the FBS column has two different output channel counts, where the former is the number of computed channels for each inference, and the latter is the number of channels remaining in the layer after removing the unused channels.

Table 4 : The network structure of M-CifarNet for CIFAR-10 classification.

In addition, we provide a detailed per-layer MACs comparison between FBS, NS, and the composition of them (NS+FBS).

We minimize the models generated by the three methods while maintaining a classification accuracy of at least 90.5%.

FIG4 shows how the skipping probabilites heat maps of the convolutional layer conv4 evolve as we fine-tune FBS-augmented M-CifarNet.

The network was trained for 12 epochs, and we saved the model at every epoch.

The heat maps are generated with the saved models in sequence, where we apply the same reordering to all heat map channels with the sorted result from the first epoch.

It can be observed that as we train the network, the channel skipping probabilites become more pronounced.

ILSVRC2012 classifiers, i.e. ResNet-18 and VGG-16, were trained with a procedure similar to Appendix A. The difference was that they were trained for a maximum of 35 epochs, the learning rate was decayed for every 20 epochs, and NS models were all pruned at 15 epochs.

For image preprocessing, we additionally cropped and stretched/squeezed images randomly following BID19 .Since VGG-16 is computationally intensive with over 15 G MACs, We first applied NS on VGG-16 to reduce the computational and memory requirements, and ease the training of the FBS-augmented variant.

We assigned a 1% budget in top-5 accuracy degradation and compressed the network using NS, which gave us a smaller VGG-16 with 20% of all channels pruned.

The resulting network is a lot less redundant, which almost halves the compute requirements, with only 7.90 G MACs remaining.

We then apply FBS to the well-compressed network.

Residual networks BID11 , such as ResNet-18, adopt sequential structure of residual blocks: DISPLAYFORM0 , where x b is the output of the b th block, K is either an identity function or a downsampling convolution, and F consists of a sequence of convolutions.

For residual networks, we directly apply FBS to all convolutional layers, with a difference in the way we handle the feature summation.

Because the (b + 1) th block receives as input the sum of the two features with sparse channels K (x b−1 ) and F (x b−1 ), a certain channel of this sum is treated as sparse when the same channels in both features are simultaneously sparse.

<|TLDR|>

@highlight

We make convolutional layers run faster by dynamically boosting and suppressing channels in feature computation.

@highlight

A feature boosting and suppression method for dynamic channel pruning that predicts the importance of each channel and then uses an affine function to amplify/suppress channel importance.

@highlight

Proposal for a channel pruning method for dynamically selecting channels during testing.
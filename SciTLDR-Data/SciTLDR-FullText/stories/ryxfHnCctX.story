To reduce memory footprint and run-time latency, techniques such as neural net-work pruning and binarization have been explored separately.

However, it is un-clear how to combine the best of the two worlds to get extremely small and efficient models.

In this paper, we, for the first time, define the filter-level pruning problem for binary neural networks, which cannot be solved by simply migrating existing structural pruning methods for full-precision models.

A novel learning-based approach is proposed to prune filters in our main/subsidiary network frame-work, where the main network is responsible for learning representative features to optimize the prediction performance, and the subsidiary component works as a filter selector on the main network.

To avoid gradient mismatch when training the subsidiary component, we propose a layer-wise and bottom-up scheme.

We also provide the theoretical and experimental comparison between our learning-based and greedy rule-based methods.

Finally, we empirically demonstrate the effectiveness of our approach applied on several binary models,  including binarizedNIN, VGG-11, and ResNet-18, on various image classification datasets.

For bi-nary ResNet-18 on ImageNet, we use 78.6% filters but can achieve slightly better test error 49.87% (50.02%-0.15%) than the original model

Deep neural networks (DNN), especially deep convolution neural networks (DCNN), have made remarkable strides during the last decade.

From the first ImageNet Challenge winner network, AlexNet, to the more recent state-of-the-art, ResNet, we observe that DNNs are growing substantially deeper and more complex.

These modern deep neural networks have millions of weights, rendering them both memory intensive and computationally expensive.

To reduce computational cost, the research into network acceleration and compression emerges as an active field.

A family of popular compression methods are the DNN pruning algorithms, which are not only efficient in both memory and speed, but also enjoy relatively simple procedure and intuition.

This line of research is motivated by the theoretical analysis and empirical discovery that redundancy does exist in both human brains and several deep models BID7 BID8 .

According to the objects to prune, we can categorize existing research according to the level of the object, such as connection (weights)-level pruning, unit/channel/filter-level pruning, and layer-level pruning BID28 .

Connection-level pruning is the most widely studied approach, which produces sparse networks whose weights are stored as sparse tensors.

Although both the footprint memory and the I/O consumption are reduced BID12 , Such methods are often not helpful towards the goal of computation acceleration unless specifically-designed hardware is leveraged.

This is because the dimensions of the weight tensor remain unchanged, though many entries are zeroed-out.

As a wellknown fact, the MAC operations on random structured sparse matrices are generally not too much faster than the dense ones of the same dimension.

In contrast, structural pruning techniques BID28 , such as unit/channel/filter-level pruning, are more hardware friendly, since they aim to produce tensors of reduced dimensions or having specific structures.

Using these techniques, it is possible to achieve both computation acceleration and memory compression on general hardware and is common for deep learning frameworks.

We consider the structural network pruning problem for a specific family of neural networks -binary neural networks.

A binary neural network is a compressed network of a general deep neural network through the quantization strategy.

Convolution operations in DCNN 1 inherently involve matrix multiplication and accumulation (MAC).

MAC operations become much more energy efficient if we use low-precision (1 bit or more) fixed-point number to approximate weights and activation functions (i.e., to quantify neurons) BID3 .

To the extreme extent, the MAC operation can even be degenerated to Boolean operations, if both weights and activation are binarized.

Such binary networks have been reported to achieve ∼58x computation saving and ∼32x memory saving in practice.

However, the binarization operation often introduces noises into DNNs , thus the representation capacity of DNNs will be impacted significantly, especially if we also binarize the activation function.

Consequently, binary neural networks inevitably require larger model size (more parameters) to compensate for the loss of representation capacity.

Although Boolean operation in binary neural networks is already quite cheap, even smaller models are still highly desired for low-power embedded systems, like smart-phones and wearable devices in virtual reality applications.

Even though quantization (e.g., binarization) has significantly reduced the redundancy of each weight/neuron representation, our experiment shows that there is still heavy redundancy in binary neural networks, in terms of network topology.

In fact, quantization and pruning are orthogonal strategies to compress neural networks: Quantization reduces the precision of parameters such as weights and activations, while pruning trims the connections in neural networks so as to attain the tightest network topology.

However, previous studies on network pruning are all designed for full-precision models and cannot be directly applied for binary neural networks whose both weights and activations are 1-bit numbers.

For example, it no longer makes any sense to prune filters by comparing the magnitude or L 1 norm of binary weights, and it is nonsensical to minimize the distance between two binary output tensors.

We, for the first time, define the problem of simplifying binary neural networks and try to learn extremely efficient deep learning models by combining pruning and quantization strategies.

Our experimental results demonstrate that filters in binary neural networks are redundant and learning-based pruning filter selection is constantly better than those existing rule-based greedy pruning criteria (like by weight magnitude or L 1 norm).We propose a learning-based method to simplify binary neural network with a main-subsidiary framework, where the main network is responsible for learning representative features to optimize the prediction performance, whereas the subsidiary component works as a filter selector on the main network to optimize the efficiency.

The contributions of this paper are summarized as follows:• We propose a learning-based structural pruning method for binary neural networks to significantly reduce the number of filters/channels but still preserve the prediction performance on large-scale problems like the ImageNet Challenge.• We show that our non-greedy learning-based method is superior to the classical rule-based methods in selecting which objects to prune.

We design a main-subsidiary framework to iteratively learn and prune feature maps.

Limitations of the rule-based methods and advantages of the learning-based methods are demonstrated by theoretical and experimental results.

In addition, we also provide a mathematical analysis for L 1 -norm based methods.• To avoid gradient mismatch of the subsidiary component, we train this network in a layerwise and bottom-up scheme.

Experimentally, the iterative training scheme helps the main network to adopt the pruning of previous layers and find a better local optimal point.2 RELATED WORK 2.1 PRUNING Deep Neural Network pruning has been explored in many different ways for a long time.

BID13 proposed Optimal Brain Surgeon (OBS) to measure the weight importance using the second-order derivative information of loss function by Taylor expansion.

BID9 further adapts OBS for deep neural networks and has reduced the retraining time.

Deep Compression BID12 prunes connections based on weight magnitude and achieved great compression ratio.

The idea of dynamic masks BID10 is also used for pruning.

Other approaches used Bayesian methods and exploited the diversity of neurons to remove weights BID23 BID22 .

However, these methods focus on pruning independent connection without considering group information.

Even though they harvest sparse connections, it is still hard to attain the desired speedup on hardware.

To address the issues in connection-level pruning, researchers proposed to increase the groupsparsity by applying sparse constraints to the channels, filters, and even layers BID28 BID0 BID25 BID1 .

used LASSO constraints and reconstruction loss to guide network channel selection.

introduced L 1 -Norm rank to prune filters, which reduces redundancy and preserves the relatively important filters using a greedy policy.

BID21 leverages a scaling factor from batch normalization to prune channels.

To encourage the scaling factor to be sparse, a regularization term is added to the loss function.

On one hand, methods mentioned above are all designed for full-precision models and cannot be trivially transferred to binary networks.

For example, to avoid introducing any non-Boolean operations, batch normalization in binary neural networks (like XNOR-Net) typically doesn't have scaling (γ) and shifting (β) parameters BID3 .

Since all weights and activation only have two possible values {1, −1}, it is also invalid to apply classical tricks such as ranking filters by their L 1 -Norms, adding a LASSO constraint, or minimizing the reconstruction error between two binary vectors.

On the other hand, greedy policies that ignore the correlations between filters cannot preserve all important filters.

Recent work shows that full precision computation is not necessary for the training and inference of DNNs BID11 .

Weights quantization is thus widely investigated, e.g., to explore 16-bit BID11 and 8-bit (Dettmers, 2015) fixed-point numbers.

To achieve higher compression and acceleration ratio, extremely low-bit models like binary weights BID5 BID18 and ternary weights BID30 BID29 BID27 have been studied, which can remove all the multiplication operations during computation.

Weight quantization has relatively milder gradient mismatch issue as analyzed in Section 3.1.2, and lots of methods can achieve comparable accuracy with full-precision counterparts on even large-scale tasks.

However, the ultimate goal for quantization networks is to replace all MAC operations by Boolean operations, which naturally desires that both activation and weights are quantized, even binarized.

The activation function of quantized network has the form of a step function, which is discontinuous and non-differentiable.

Gradient cannot flow through a quantized activation function during backpropagation.

The straight-through estimator (STE) is widely adopted to circumvents this problem, approximating the gradient of step function as 1 in a certain range BID16 BID2 .

BID4 proposed the Half-wave Gaussian Quantization (HWGQ) to further reduce the mismatch between the forward quantized activation function and the backward ReLU BID24 .

Binary Neural Networks (BNN) proposed in BID6 and BID3 use only 1 bit for both activation functions and weights, which ends up with an extremely small and faster network.

BNNs inherit the drawback of acceleration via quantization strategy and their accuracy also need to be further improved. .

Because both weights and activations are binary, we remove the subscripts of F b and W b for clarity.

The goal of pruning is to remove certain filters W i n,:,:,: , n ∈ Ω, where Ω is the indices of pruned filters.

If a filter is removed, the corresponding output feature map of this layer (which is also the input feature map of next layer) will be removed, too.

Furthermore, the input channels of all filters in the next layer would become unnecessary.

If all filters in one layer can be removed, the filter-level pruning will upgrade to layerlevel pruning naturally.

The goal of our method is to remove as many filters as possible for binary neural networks which are already compact and have inferior numerical properties, thus this task is more challenging compared with pruning a full-precision model.

We borrow the ideas from binary network optimization to simplify binary networks.

While it sounds tautological, note that the optimization techniques were originally invented to solve the quantization problem, but we will show that it can be crafted to solve the pruning problem for binary networks.

A new binary network, called subsidiary component, acts as learnable masks to screen out redundant features in the main network, which is the network to complete classification tasks.

Each update of the subsidiary component can be viewed as the exploration in the mask search space.

We try to find a (local) optimal mask in that space with the help of the subsidiary component.

The process of training subsidiary and main networks is as follows:

For layer i, the weights of subsidiary component M i ∈ R Ni+1×Ni×Ki+1×Ki+1 are initialized by the uniform distribution: DISPLAYFORM0 In practice, σ is chosen to be less than 10 −5 .

To achieve the goal of pruning filters, all elements whose first index is the same share the same value.

DISPLAYFORM1 is an output tensor from the subsidiary component.

In the first stage, we use the Iden(·) function (identity transforma- DISPLAYFORM2 We apply the filter mask O i to screen main network's weights W i , DISPLAYFORM3 , where ⊗ is element-wise product.

Ŵ i denotes the weights of the main network after transformation, which is used to be convolved with the input feature maps, F i , to produce the output feature maps F i+1 .

Then, weights of the main network, W j , j ∈ [1, I], are set to be trainable while weights of the subsidiary component, M j , j ∈ [1, I], are fixed.

Because subsidiary weights are fixed and initialized to be near-zero, it will not function in the Feature Learning stage, thuŝ DISPLAYFORM4 The whole main binary neural network will be trained from scratch.

Training Subsidiary Component within a Single Layer i: After training the whole main network from scratch, we use a binary operator to select features in a layer-wise manner.

In opposite to the previous Feature Learning stage, the weights of all layers W j , j ∈ [1, I] of the main network and the weights except layer i of the subsidiary component M j , j ∈ [1, I]/[i] are set to be fixed, while the subsidiary component's weights at the current layer M i are trainable when selecting features for Layer i. The transformation function for the filter mask O i is changed from Iden(·) to Bin(·) (sign transformation + linear affine), DISPLAYFORM0 2 By doing this, we project the float-point M i to binarized numbers ranging from 0 to 1.

Elements in O i which are equal to 0 indicate that the corresponding filters are removed and the elements of value 1 imply to keep this filter.

Since Bin(·) is not differentiable, we use the following function instead of the sign function in back propagation when training the subsidiary component M i BID16 BID2 , DISPLAYFORM1 Apart from the transformation, we also need to add regularization terms to prevent all O i from degenerating to zero, which is a trivial solution.

So the loss function of training Layer i in the subsidiary component is, arg min DISPLAYFORM2 where L cross entropy is the loss on data and L distill is the distillation loss defined in (7).Finally, we fix the layers M j , j ∈ [1, I] in the subsidiary component and layers before i in the main network (i.e., W j , j ∈ [1, i − 1]), and retrain the main layers after Layer i (i.e., W j , j ∈ [i, I]).Bottom-up Layer-wise Training for Multiple Layers: We showed how to train a layer in the subsidiary component above.

To alleviate the gradient mismatch and keep away from the trivial solution during Features Selection, next, we propose a layer-wise and bottom-up training scheme for the subsidiary component: Layers closer to the input in the subsidiary component will be trained with priority.

As Layer i is under training, all previous layers (which should have already been trained) will be fixed and subsequent layers will constantly be the initial near-zero value during training.

There are three advantages of this training scheme.

First, as in (1) , we use STE as in BID16 BID2 to approximate the gradient of the sign function.

By chain rule, for each activation node j in Layer i, we would like to compute an "error term" δ i j = ∂L ∂a i j which measures how much that node is responsible for any errors in the output.

For binary neural networks, activation is also binarized by a sign function which need STE for back-propagation.

The "Error term" for binary neural networks is given by, DISPLAYFORM3 where (3) and (5) can be obtained by the chain rule, and (4) and (6) are estimated from STE, which will introduce gradient mismatch into back-propagation.

We refer (6) as weight gradient mismatch issue and (4) as activation gradient mismatch issue.

They are two open problems in the optimization of binary neural networks, both caused by the quantization transform functions like Sign(·).

Starting from bottom layers, we can train and fix layers who are harder to train as early as possible for the subsidiary component.

In addition, because of the retraining part in Features Selection, bottom-up training scheme allows bottom layers to be fixed earlier, as well.

In practice, this scheme results in more stable training curves and can find a better local optimal point.

Second, the bottom-up layer-wise training scheme helps the main network to better accommodate the feature distribution shift caused by the pruning of previous layers.

As mentioned before, the main difference in the motivation between our pruning method and rule-based methods is that we have more learnable parameters to fit the data by focusing on the final network output.

With the bottom-up and layer-wise scheme, even if the output of Layer i changes, subsequent layers in the main network can accommodate this change by modifying their features.

Lastly and most importantly, we achieve higher pruning ratio by this scheme.

According to our experiments, a straight-forward global training scheme leads to limited pruning ratio.

Some layers are pruned excessively and hence damaged the accuracy, while some layers are barely pruned, which hurts the pruning ratio.

The layer-wise scheme would enforce all layer to be out of the comfort zone and allow balancing between accuracy and pruning ratio.

The pipeline of our method is as follows:1.

Initialize weights of subsidiary component M j , j ∈ [1, I] with near-zero σ's.2.

Set M j , j ∈ [1, I] to be fixed, and train the whole main network from scratch.3.

Train starting from the first binary kernel.

Each layer is the same as in the algorithm shown below:• Change the activation function for M i from Iden(·) to Bin(·).

And all other parameters apart from M i are fixed.

Train subsidiary component according to (2).• Fix the subsidiary layers M j , j ∈ [1, I] and main layers before i-th layer W j , j ∈ [1, i − 1], and retrain main layers after i-th layer W j , j ∈ [i, I].

Though pruning network filters is not an explicit transfer learning task, the aim is to guide the thin network to learn more similar output distributions with the original network.

The model is supposed to learn a soft distribution but not a hard one as proposed in previous traditional classifier networks.

Hence, we add a distillation loss to guide the training subsidiary component to be more stable, as shown in FIG2 .

DISPLAYFORM0 We set p to be the original binary neural network distribution.

Because the distribution is fixed, the H(p) is a constant and can be removed from L distill .

It means that the distillation loss can be written as DISPLAYFORM1 ) where z i and t i represent the final output of the pruned and original networks before the softmax layer.

T is a temperature parameter for the distillation loss defined in BID17 .

We set T as 1 in practice.

M is the number of classes.

Previous methods use rules to rank the importance of each filter and then remove the top k least important filters.

The rules can be weight magnitude, e.g., measured by the L 1 norm, or some other well-designed criteria.

Studies in this line share the same motivation that individual filters have their own importance indication, and filters with less importance can be removed relatively safely.

This assumption ignores interactions among filters.

As mentioned before, rule-based pruning algorithms use a greedy way to prune filters, i.e., they assume that individual filters behave independently and their own importance (or function) for representation learning.

We give a theoretical analysis in Section 3.3 about this point.

In fact, pruning filters independently may cause problems when filter are strongly correlated.

For example, if two filters have learned the same features (or concepts), these two filters may be pruned out together by rule-based methods, because their rankings are very close.

Clearly, pruning one of them is a better choice.

However, almost all these criteria are based on value statistics and are completely unsuitable for the binary scenario with only two discrete values.

One possible pruning method is to exhaustively search the optimal pruning set, but this is NP-Hard and prohibitive for modern DNNs that have thousands of filters.

Our method uses the subsidiary component to "search" the optimal solution.

Our soft "search" strategy is gradient-based and batch-based compared to exhaustive search, and it is much more efficient.

If our main network is full-precision, the L 1 -Norm based pruning technique would be strongly relevant to our method, except that we target at optimizing the final output of the network, whereas the L 1 -Norm based method greedily controls the perturbation of the feature map in the next layer.

Suppose that W = [w 1 ; . . . ; w n ] is the original filter blocked by rows, W = [w 1 ; . . . ; w n ] is the pruned filter, and x is the input feature map.

Let ∆w i ≡ w i − w i .

Then, the L 1 -Norm approach minimizes the upper bound of the following problem: max x ∞ <τ W x − W x .

To see this, note DISPLAYFORM0 To minimize i ∆w 1 by zeroing-out a single row w i , obviously, the solution is to select the one with the smallest L 1 -Norm.

However, note that this strategy cannot be trivially applied for binary networks, because the L 1 -Norm for any filter that is a {−1, +1} tensor of the same shape is always identical.

Previous work (He et al.) uses the LASSO regression to minimize the reconstruction error of each layer: DISPLAYFORM1 β 0 ≤ C .

Solving this L 0 minimization problem is NPhard, so the L 0 regularization is usually relaxed to L 1 .

In the binary/quantization scenario, activations only have two/several values and the least reconstruction error is not applicable.

Instead of minimizing the reconstruction error of a layer, our method pays attention on the final network output with the help of the learnable subsidiary component.

We directly optimize the discrete variables of masks (a.k.a subsidiary component) without the relaxation.

To evaluate our method, we conduct several pruning experiments for VGG-11, Net-InNet (NIN), and ResNet-18 on CIFAR-10 and ImageNet.

Since our goal is to simplify binary neural networks, whose activation and weights are both 1-bit, all main models and training settings in our experiments inherit from XNOR-Net BID26 ).

Since we are, to the best of our knowledge, the first work to define filter-level pruning for binary neural networks, we proposed a rule-based method by ourselves as the baseline.

Instead of ranking filters according to the L 1 -Norm , we use the magnitude of each filter's scaling factor (MSF) as our pruning criterion.

Inspired by , we test both the "prune once and retrain" scheme 2 and the "prune and retrain iteratively" scheme 3 .

Figure 3: Learning curve for subsidiary component.

We train the subsidiary component with different learning rate.

These curves are smoothed for the directly seeing the trend of the learning Subsidiary Component.

All the dotted lines represent the learning curve of the large learning rate 10 −3 , the normal lines represent the learning curves of the small learning rate 10 −4 .As pointed out in BID26 we set weights of the first layer and last layer as full-precision, which also means that we only do pruning for the intermediate binary layers.

We measure effectiveness of pruning methods in terms of PFR, the ratio of the number of pruned filters to original filter number, and error rate before and after retraining.

For error ratio, smaller is better.

For PFR, larger is better.

For CIFAR-10, when training the main network, learning rate starts from 10 −4 , and learningrate-decay is equal to 0.1 for every 20 epochs.

Learning rate is fixed with 10 −3 when training the subsidiary component.

For ImageNet, we set a constant learning rate of 10 −3 for the subsidiary component and main work.

For fair comparison, we control PFR for each layer of these methods to be the same to observe the final Retrain-Error.

In FIG3 , MSF-Layerwise refers to the "prune once and retrain" scheme, and the MSF-Cascade refers the "prune and retrain iteratively" scheme.

The first three figures of experiments were done on the CIFAR-10 dataset.

The last figure refers to results on Imagenet.

4.1 NIN AND VGG-11 ON CIFAR-10 NIN is a fully convolutional network, using two 1 × 1 convolution layers instead of fully connected layer, and has quite compact architecture.

VGG-11 is a high-capacity network for classification.

VGG-11 on CIFAR-10 consists of 8 convolutional layers(including 7 binary layers) and 1 fully connected layers.

Batch normalization is used between every binary convolution and activation layer, which makes the training process more stable and converge with high performance.

For both MSF-Layerwise and MSF-Cascade, with the same PCR, the performance is worse than us.

With 30% ∼ 40% of pruning filter ratio, the pruned network error rate only increased 1% ∼ 2%.

An interesting phenomenon is observed when training subsidiary components for different models.

We try different learning rates in our experiments and observe it impacts final convergent point a lot as shown in Figure 3 .

The relatively smaller learning rate (10 −4 ) will converge with lower accuracy and higher pruning number; however, the larger learning rate (10 −3 ) leads to the opposite result.

One possible explanation is that the solution space of the high-dimensional manifold for binary neural networks is more discrete compared to full-precision networks, so it is difficult for a subsidiary component to jump out of a locally optimal point to a better one.

Moreover, in the binary scenario, larger learning rate will increase the frequency of value changing for weights.

Our motivation is to use a learnable subsidiary components to approximate exhaustive search, so using a larger learning rate will enable the subsidiary component to "search" more aggressively.

A large learning rate may be unsuitable for normal binary neural networks like the main network in this paper, but it is preferred by the subsidiary component.

As mentioned in section 3.1.1, we use the uniform distribution to initialize the mask.

According to the expectation of the uniform distribution, E(SP ) = 0.5, where SP is the ratio of the number of positive elements in subsidiary weights to size of weights.

However, since we use Sign(·), different SP may impact the result a lot.

We conduct six experiments on different models across different layers and show that initialization with 0.4, 0.6, 1.0 SP will all converge to the same state.

However, when SP is 0.2, final performance will be very poor.

A possible reason is that the number of filters thrown out by the initialization is too large, and due to the existence of the regularization term, the network's self-adjustment ability is limited and cannot converge to a good state.

Hence we recommend the SP to be intialized to greater than 0.4.

Compared with NIN and VGG-11, ResNet has identity connections within residual block and much more layers.

As the depth of network increases, the capacity of network also increases, which then leads to more redundancy.

From experimental results, we find that when the identification mapping network has a downsampling layer, the overall sensitivity of the residual block will increase.

Overall result for ResNet on CIFAR-10 is shown in table (1), and statistics for each layer can be found in Appendix.

We further verify our method with ResNet-18 on ImageNet.

α can be set from 10 −7 to 10 −9 depending on the expected PFR, the accuracy and pruning ratio are balanced before retraining.

After 20 epoches retraining for each layer, the final PFR is 21.4%, with the retrained error has decreased from 50.02% to 49.87%.

Using STE, weights gradient mismatch will be introduced here.

Using STE, activation gradient mismatch will be introduced here.

Figure 5: Gradient flow of binary neural networks during back-propagation.

Rectangles represent the weight tensor and ellipses represent functional operation.

In this paper, we use binary operation as a special quantization function.

MAC is short for multiplication and accumulate operations, or the equivalent substitution like XNOR BID3 in BNN.

For fair comparison, we control PFR for each layer of these methods to be the same to observe the final Retrain-Error.

In TAB1 , MSF-Layerwise refers to the "prune once and retrain" scheme, and the MSF-Cascade refers the "prune and retrain iteratively" scheme.

The first three groups of experiments were done on the CIFAR-10 dataset.

The last group refers to results on Imagenet.

@highlight

we define the filter-level pruning problem for binary neural networks for the first time and propose method to solve it.
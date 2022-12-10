Skip connections are increasingly utilized by deep neural networks to improve accuracy and cost-efficiency.

In particular, the recent DenseNet is efficient in computation and parameters, and achieves state-of-the-art predictions by directly connecting each feature layer to all previous ones.

However, DenseNet's extreme connectivity pattern may hinder its scalability to high depths, and in applications like fully convolutional networks, full DenseNet connections are prohibitively expensive.

This work first experimentally shows that one key advantage of skip connections is to have short distances among feature layers during backpropagation.

Specifically, using a fixed number of skip connections, the connection patterns with shorter backpropagation distance among layers have more accurate predictions.

Following this insight, we propose a connection template, Log-DenseNet, which, in comparison to DenseNet,  only slightly increases the backpropagation distances among layers from 1 to  ($1 + \log_2 L$), but uses only $L\log_2 L$ total connections instead of $O(L^2)$. Hence, \logdenses are easier to scale than DenseNets, and no longer require careful GPU memory management.

We demonstrate the effectiveness of our design principle by showing better performance than DenseNets on tabula rasa semantic segmentation, and competitive results on visual recognition.

Deep neural networks have been improving performance for many machine learning tasks, scaling from networks like AlexNet BID17 to increasingly more complex and expensive networks, like VGG BID30 , ResNet BID8 and Inception BID5 .

Continued hardware and software advances will enable us to build deeper neural networks, which have higher representation power than shallower ones.

However, the payoff from increasing the depth of the networks only holds in practice if the networks can be trained effectively.

It has been shown that naïvely scaling up the depth of networks actually decreases the performance BID8 , partially because of vanishing/exploding gradients in very deep networks.

Furthermore, in certain tasks such as semantic segmentation, it is common to take a pre-trained network and fine-tune, because training from scratch is difficult in terms of both computational cost and reaching good solutions.

Overcoming the vanishing gradient problem and being able to train from scratch are two active areas of research.

Recent works attempt to overcome these training difficulties in deeper networks by introducing skip, or shortcut, connections BID25 BID7 BID31 BID8 BID19 so the gradient reaches earlier layers and compositions of features at varying depth can be combined for better performance.

In particular, DenseNet is the extreme example of this, concatenating all previous layers to form the input of each layer, i.e., connecting each layer to all previous ones.

However, this incurs an O(L 2 ) run-time complexity for a depth L network, and may hinder the scaling of networks.

Specifically, in fully convolutional networks (FCNs), where the final feature maps have high resolution so that full DenseNet connections are prohibitively expensive, BID14 propose to cut most of connections from the mid-depth.

To combat the scaling issue, propose to halve the total channel size a number of times.

Futhermore, cut 40% of the channels in DenseNets while maintaining the accuracy, suggesting that much of the O(L 2 ) computation is redundant.

Therefore, it is both necessary and natural to consider a more efficient design principle for placing shortcut connections in deep neural networks.1In this work, we address the scaling issue of skip connections by answering the question: if we can only afford the computation of a limited number of skip connections and we believe the network needs to have at least a certain depth, where should the skip connections be placed?

We design experiments to show that with the same number of skip connections at each layer, the networks can have drastically different performance based on where the skip connections are.

In particular, we summarize this result as the following design principle, which we formalize in Sec. 3.2: given a fixed number of shortcut connections to each feature layer, we should choose these shortcut connections to minimize the distance among layers during backpropagation.

Following this principle, we design a network template, Log-DenseNet.

In comparison to DenseNets at depth L, Log-DenseNets cost only L log L, instead of O(L 2 ) run-time complexity.

Furthermore, Log-DenseNets only slightly increase the short distances among layers during backpropagation from 1 to 1 + log L. Hence, Log-DenseNets can scale to deeper and wider networks, even without custom GPU memory managements that DenseNets require.

In particular, we show that Log-DenseNets outperform DenseNets on tabula rasa semantic segmentation on CamVid BID2 , while using only half of the parameters, and similar computation.

Log-DenseNets also achieve comparable performance to DenseNet with the same computations on visual recognition data-sets, including ILSVRC2012 BID29 .

In short, our contributions are as follows:• We experimentally support the design principle that with a fixed number of skip connections per layer, we should place them to minimize the distance among layers during backpropagation.• The proposed Log-DenseNets achieve small 1 + log 2 L between-layer distances using few connections (L log 2 L), and hence, are scalable for deep networks and applications like FCNs.• The proposed network outperforms DenseNet on CamVid for tabula rasa semantic segmentation, and achieves comparable performance on ILSVRC2012 for recognition.

Skip connections.

The most popular approach to creating shortcuts is to directly add features from different layers together, with or without weights.

Residual and Highway Networks BID8 BID31 propose to sum the new feature map at each depth with the ones from skip connections, so that new features can be understood as fitting residual features of the earlier ones.

FractalNet BID19 explicitly constructs shortcut networks recursively and averages the outputs from the shortcuts.

Such structures prevent deep networks from degrading from the shallow shortcuts via "teacher-student" effects.

BID11 implicitly constructs skip connections by allowing entire layers to be dropout during training.

DualPathNet BID4 combines the insights of DenseNet and ResNet BID8 , and utilizes both concatenation and summation of previous features.

Run-time Complexity and Memory of DenseNets.

DenseNet emphasizes skip connections by directly connecting each layer to all previous layers.

However, this quadratic complexity may prevent DenseNet from scale to deep and wide models.

In order to scale, DenseNet applies block compression, which halves the number of channels in the concatenation of previous layers.

DenseNet also opts not to double the output channel size of conv layers after downsampling, which divides the computational cost of each skip connection.

These design choices enable DenseNets to be deep for image classification where final layers have low resolutions.

However, final layers in FCNs for semantic segmentation have higher resolution than in classification.

Hence, to fit models in the limited GPU memory, FC-DenseNets BID14 have to cut most of their skip connections from mid-depth layers.

Furthermore, a naïve implementation of DenseNet requires O(L 2 ) memory, because the inputs of the L convolutions are individually stored.

Though there exist O(L) implementations via memory sharing among layers BID23 , they require custom GPU memory management, which is not supported in many existing packages.

Hence, one may have to use custom implementations and recompile packages for memory efficient Densenets, e.g., it costs a thousand lines of C++ on Caffe BID22 .

Our work recognizes the contributions of DenseNet's architecture to utilize skip connections, and advocates for the efficient use of compositional skip connections to shorten the distances among feature layers during backpropagation.

Our design principle can especially help applications like FC-DenseNet BID14 where the network is desired to be at least a certain depth, but only a limited number of shortcut connections can be formed.

2Network Compression.

A wide array of works have proposed methods to compress networks by reducing redundancy and computational costs.

BID6 BID15 BID13 decompose the computation of convolutions at spatial and channel levels to reduce convolution complexity.

BID9 BID0 BID28 propose to train networks with smaller costs to mimic expensive ones.

uses L1 regularization to cut 40% of channels in DenseNet without losing accuracy.

These methods, however, cannot help in applications that cannot fit the complex networks in GPUs in the first place.

This work, instead of cutting connections arbitrarily or post-design, advocates a network design principle to place skip connections intelligently to minimize between-layer distances.3 FROM DENSENET TO LOG-DENSENET 3.1 PRELIMINARY ON DENSENETS Formally, we call the feature layers in a feed-forward convolutional network as x 0 , x 1 , ..., x L , where x 0 is the result of the initial convolution on the input image.

Each x i is a transformation f i with parameter θ i and takes input from a subset of x 0 , ..., x i−1 .

E.g., a traditional feed-forward network has x i = f i (x i−1 ; θ i ), and the recent DenseNet proposes to form each feature layer x i using all previous features layers, i.e., DISPLAYFORM0 where concat(•) concatenates all features in its input collection along the feature channel dimension.

Each f i is a bottleneck structure , i.e., BN-ReLU-1x1conv-BN-ReLU-3x3conv, where the final conv produces the growth rate g number of channels, and the bottleneck 1x1 conv produces 4g channels of features.

DenseNet also organizes layers into n block number of blocks.

Between two contiguous blocks, there is a block compression using a 1x1conv-BN-ReLU, followed by an average pooling, to downsample previous features for deeper and coarser layers.

In practice, n block is small in visual recognition architectures BID8 BID5 .The direct connections among layers in DenseNet are believed to introduce implicit deep supervision BID20 in intermediate layers, and reduce the vanishing/exploding gradient problem by enabling direct influence between any two feature layers.

Inspired by this belief, we propose a design principle to organize the skip connections: with a fixed connection budget, we should minimize the connection distance among layers.

To formalize our design principle, we consider each x i as a node in a graph, and the directed edge DISPLAYFORM0 is then the length of the shortest path from x i to x j on the graph.

Then we define the maximum backpropagation distance (MBD) as the maximum BD among all pairs i >

j.

Then DenseNet has a MBD of 1, if we disregard transition layers, but at the cost of O(L 2 ) connections.

We next propose short connection patterns for when the connection budget is O(L log L).

In comparison to DenseNet, The proposed Log-DenseNet increases the MBD to 1 + log 2 L while using only O(L log L) connections.

Since the current practical networks have less than 1000 depths, the proposed method has a single-digit MBD.3.3 LOG-DENSENET For simplicity, we let log(•) denote log 2 (•).

In a proposed Log-Dense Network, each layer i takes direct input from at most log(i) + 1 number of previous layers, and these input layers are exponentially apart from depth i with base 2, i.e., DISPLAYFORM1 where • is the nearest integer function and • is the floor function.

For example, the input features for layer i are layer i − 1, i − 2, i − 4, ....

We define the input index set at layer i to be {i − 2 k : k = 0, ..., log(i) }.

We illustrate the connection in FIG0 .

Since the complexity of layer i is log(i)+1, the overall complexity of a Log-DenseNet is DISPLAYFORM2 , which is significantly smaller than the quadratic complexity, Θ(L 2 ), of a DenseNet.

Log-DenseNet V1: independent transition.

Following , we organize layers into blocks.

Layers in the same block have the same resolution; the feature map side is halved after each block.

In between two consecutive blocks, a transition layer will shrink all previous layers so that future layers can use them in Eq 2.

We define a pooling transition as a 1x1 conv followed by a 2x2 average pooling, where the output channel size of the conv is the same as the input one.

We refer to x i after t number of pooling transition as DISPLAYFORM3 i exists}, and compute x (t+1) i .

We abuse the notation x i when it is used as an input of a feature layer to mean the appropriate x (t) i so that the output and input resolutions match.

Unlike DenseNet, we independently process each early layer instead of using a pooling transition on the concatenated early features, because the latter option results in O(L 2 ) complexity per transition layer, if at least O(L) layers are to be processed.

Since Log-DenseNet costs O(L) computation for each transition, the total transition cost is O(L log L) as long as we have O(log L) transitions.

Log-DenseNet V2: block compression.

Unfortunately, many neural network packages, such as TensorFlow, cannot compute the O(L) 1x1 conv for transition efficiently: in practice, this O(L) operation costs about the same wall-clock time as the O(L 2 )-cost 1x1 conv on the concatenation of the O(L) layers.

To speed up transition and to further reduce MBD, we propose a block compression for Log-DenseNet similar to the block compression in DenseNet .

At each transition, the newly finished block of feature layers are concatenated and compressed into g log L channels using 1x1 conv.

The other previous compressed features are concatenated, followed by a 1x1 conv that keep the number of channels unchanged.

These two blocks of compressed features then go through 2x2 average pooling to downsample, and are then concatenated together.

FIG0 illustrates how the compressed features are used when n block = 3, where x 0 , the initial conv layer of channel size 2g, is considered the initial compressed block.

The total connections and run-time complexity are still O(L log L), at any depth the total channel from the compressed feature is at most (n block − 1)g log L + 2g, and we assume n block ≤ 4 is a constant.

Furthermore, these transitions cost O(L log L) connections and computation in total, since compressing of the latest block costs O(L log L) and transforming the older blocks costs O(log 2 L).

DISPLAYFORM4 in LogDenseNet only increases the MBD among layers to 1 + log L.

This result is summarized as follows.

Proposition 3.1.

For any two feature layers x i = x j in Log-DenseNet that has n block number of blocks, the maximum backpropagation distance between x i and x j is at most log |j − i| + n block .This proposition argues that if we ignore pooling layers, or in the case of Log-DenseNet V1, consider the transition layers as part of each feature layer, then any two layers x i and x j are only log |j −i|+1 away from each other during backpropagation, so that layers can still easily affect each other to fit the training signals.

Sec. 4.1 experimentally shows that with the same amount the connections, the connection strategy with smaller MBD leads to better accuracy.

We defer the proof to the appendix.

In comparison to Log-DenseNet V1, V2 reduces the BD between any two layers from different blocks to be at most n block , where the shortest paths go through the compressed blocks.

Deep supervision.

Since we cut the majority of the connections in DenseNet when forming LogDenseNet, we found that having additional training signals at the intermediate layers using deep DISPLAYFORM5 DISPLAYFORM6 L log L .

LD clearly outperforms N and E thanks to its low MBD.

Log-DenseNet V2 (LD2) outperforms the others, since it has about n block /2 times total connections as the others.

LD2 has MBD 1 + log L n block .

supervision BID20 for the early layers helps the convergence of the network, even though the original DenseNet does not see performance impact from deep supervision.

For simplicity, we place the auxiliary predictions at the end of each block.

Let x i be a feature layer at the end of a block.

Then the auxiliary prediction at x i takes as input x i along with x i 's input features.

Following BID10 , we put half of the total weighting in the final prediction and spread the other half evenly.

After convergence, we take one extra epoch of training optimizing only the final prediction.

We found this results in the lower validation error rate than always optimizing the final loss alone.

For visual recognition, we experiment on CIFAR10, CIFAR100 BID16 ), SVHN BID26 , and ILSVRC2012 BID29 .

1 We follow BID8 for the training procedure and parameter choices.

Specifically, we optimize using stochastic gradient descent with a moment of 0.9 and a batch size of 64 on CIFAR and SVHN.

The learning rate starts at 0.1 and is divided by 10 after 1/2 and 3/4 of the total iterations are done.

We train 250 epochs on CIFAR, 60 on SVHN, and 90 on ILSVRC.

For CIFAR and SVHN, we specify a network by a pair (n, g), where n is the number of dense layers in each of the three dense blocks, and g, the growth rate, is the number of channels in each new layer.

This section verifies that short MBD is an important design principle by comparing the proposed Log-DenseNet V1 against two other intuitive connection strategies that also connects each layer i to 1 + log(i) previous layers.

The first strategy, called NEAREST connects layer i to its previous log(i) depths, i.e., x i = f i (concat({x i−k : k = 1, ..., log b (i) }) ; θ i ).

The second strategy, called EVENLY-SPACED connects layer i to log(i) previous depths that are evenly spaced; i.e., x i = f i (concat({x i−1−kδ : δ = i log(i) and k = 0, 1, 2, ... and kδ ≤ i − 1}) ; θ i ).

Both methods above are intuitive.

However, each of them has a MBD that is on the order of O( L log(L) ), which is much higher than the O(log(L)) MBD of the proposed Log-DenseNet V1.

We experiment with networks whose (n, g) are in {12, 32, 52}×{16, 24, 32}, and show in Table 1 that Log-DenseNet almost always outperforms the other two strategies.

Furthermore, the average relative increase of top-1 error rate using NEAREST and EVENLY-SPACED from using Log-DenseNet is 12.2% and 8.5%, which is significant: for instance, (52,32) achieves 23.10% error rate using EVENLY-SPACED, which is about 10% relatively worse than the 20.58% from (52,32) using Log-DenseNet, but (52,16) using Log-DenseNet already has 23.45% error rate using a quarter of the computation of (52,32).We also showcase the advantage of small MBD when each layer x i is connects to ≈ i 2 number of previous layers.

With these O(L 2 ) total connections, NEAREST has a MBD of log L, because Table 2 : Performance of connection patterns with O(L 2 ) total connections.

NEAREST (N), EVENLY-SPACED (E), and NearestHalfAndLog (N+LD) connect layer i to about i/2 previous layers.

DenseNet without block compression (D) connects i to all previous i − 1 layers, and is thus about twice as expensive as the other three options.

We highlight that N+LD greatly improves over N, because the few log L additional connections greatly reduced the MBD.

The MBD of N, E, N+LD, and D are log L, 2, 2, and 1.

we can halve i (assuming i > j) until j > i/2 so that i and j are directly connected.

EVENLY-SPACED has a MBD of 2, because each x i takes input from every other previous layer.

Table 2 shows that EVENLY-SPACED significantly outperform NEAREST on CIFAR10 and CIFAR100, validating the importance of MBD.

We also show that NEAREST can be greatly improved with just a few additional shortcuts to reduce the MBD.

In the NEAREST scheme, x i already connects to x i−1 , x i−2 , ..., x i/2 .

We then also connects x i to x i/4 , x i/8 , x i/16 , ....

We call this scheme NearestHalfAndLog, and it has a MBD of 2, because any j < i is either directly connected to i, if j > i/2, or j is connected to some i/ i/2 k for some k, which is connected to i directly.

FIG0 illustrates the connections of this scheme.

We observe in Table 2 that with this few log i −1 additional connections to the existing i/2 ones, we drastically reduce the error rates to the level of EVENLY-SPACED, which has the same MBD of 2.

These comparisons support our design principle: with the same number of connections at each depth i, the connection patterns with low MBD outperform the ones with high MBD.

DISPLAYFORM0

Semantic segmentation assigns every pixel of input images with a label class, and it is an important step for understanding image scenes for robotics such as autonomous driving.

The state-ofthe-art training procedure BID32 BID3 typically requires training a fullyconvolutional network (FCN) BID25 and starting with a recognition network that is trained on large data-sets such as ILSVRC or COCO, because training FCNs from scratch is prone to overfitting and is difficult to converge.

BID14 shows that DenseNets are promising for enabling FCNs to be trained from scratch.

In fact, fully convolutional DenseNets (FC-DenseNets) are shown to be able to achieve the state-of-the-art predictions training from scratch without additional data on CamVid BID2 and GATech (Raza et al., 2013) .

However, the drawbacks of DenseNet are already manifested in applications on even relatively small images (360x480 resolution from CamVid).

In particular, to fit FC-DenseNet into memory and to run it in reasonable speed, BID14 proposes to cut many mid-connections: during upsampling, each layer is only directly connected to layers in its current block and its immediately previous block.

Such connection strategy is similar to the NEAREST strategy in Sec. 4.1, which has already been shown to be less effective than the proposed Log-DenseNet in classification tasks.

We now experimentally show that fully-convolutional Log-DenseNet (FC-Log-DenseNet) outperforms FC-DenseNet.

6Figure 2: Each row: input image, ground truth labeling, and any scene parsing results at 1/4, 1/2, 3/4 and the final layer.

Noting that the first half of the network downsamples feature maps, and the second half upsamples, we have the lowest resolution of predictions at 1/2, so that its prediction appear blurred.

FC-Log-DenseNet 103.

Following BID14 , we form FC-Log-DenseNet V1-103 with 11 Log-DenseNet V1 blocks, where the number of feature layers in the blocks are 4, 5, 7, 10, 12, 15, 12, 10, 7, 5, 4 .

After each of the first five blocks, there is a transition that transforms and downsamples previous layers independently.

After each of the next five blocks, there is a transition that applies a transposed convolution to upsample each previous layer.

Both down and up sampling are only done when needed, so that if a layer is not used directly in the future, no transition is applied to it.

Each feature layer takes input using the Log-DenseNet connection strategy.

Since Log-DenseNet connections are sparse to early layers, which contain important high resolution features for high resolution semantic segmentation, we add feature layer x 4 , which is the last layer of the first block, to the input set of all subsequent layers.

This adds only one extra connection for each layer after the first block, so the overall complexity remains roughly the same.

We do not form any other skip connections, since Log-DenseNet already provides sparse connections to past layers.

We do not form FC networks using Log-DenseNet V2, because there are 11 blocks, so that V2 would multiply the final block cost by about 10.

This is significant, because the final block already costs about half of the total FLOPS.

We breakdown the FLOPS by blocks in the appendix Fig. 5b .Training details.

Our training procedure and parameters follow from those of FC-DenseNet BID14 , except that we set the growth rate to 24 instead of 16, in order to have around the same computational cost as FC-DenseNet.

We defer the details to the appendix.

However, we also found auxiliary predictions at the end of each dense block reduce overfitting and produce interesting progression of the predictions, as shown in Fig. 2 .

Specifically, these auxiliary predictions produces semantic segmentation at the scale of their features using 1x1 conv layers.

The inputs of the predictions and the weighting of the losses are the same as in classification, as specified in Sec. 3.3.Performance analysis.

We note that the final two blocks of FC-DenseNet and FC-Log-DenseNet cost half of their total computation.

This is because the final blocks have fine resolutions, which also make the full DenseNet connection in the final two blocks prohibitively expensive.

This is also why FC-DenseNets BID14 have to forgo all the mid-depth the shortcut connections in its upsampling blocks.

Table 3 lists the Intersection-over-Union ratios (IoUs) of the scene parsing results.

FC-Log-DenseNet achieves 67.3% mean IoUs, which is slightly higher than the 66.9% of FC-DenseNet.

Among the 11 classes, FC-Log-DenseNet performs similarly to FC-DenseNet.

Hence FC-Log-DenseNet achieves the same level of performance as FC-DenseNet with 50% fewer parameters and similar computations in FLOPS.

This supports our hypothesis that we should minimize MBD when we have can only have a limited number of skip connections.

FC-Log-DenseNet can potentially be improved if we reuse the shortcut connections in the final block to reduce the number of upsamplings.

This section studies the trade-off between computational cost and the accuracy of networks on visual recognition.

In particular, we address the question of whether sparser networks like Log-DenseNet perform better than DenseNet using the same computation.

DenseNets can be very deep for image classification, because they have low resolution in the final block.

In particular, a skip connection to 7 the final block costs 1/64 of one to the first block.

FIG1 illustrates the error rates on CIFAR100 of Log-DenseNet V1 and V2 and DenseNet.

The Log-DenseNet variants have g = 32, and n = 12, 22, 32, ..., 82.

DenseNets have g = 32, and n = 12, 22, 32, 42.

Log-DenseNet V2 has around the same performance as DenseNet on CIFAR100.

This is partially explained by the fact that most pairs of x i , x j in Log-DenseNet V2 are cross-block, so that they have the same MBD as in Densenets thanks to the compressed early blocks.

The within block distance is bounded by the logarithm of the block size, which is smaller than 7 here.

Log-DenseNet V1 has similar error rates as the other two, but is slightly worse, an expected result, because unlike V2, backpropagation distances between a pair x i , x j in V1 is always log |i − j|, so on average V1 has a higher MBD than V2 does.

The performance gap between Log-DenseNet V1 and DenseNet also gradually widens with the depth of the network, possibly because the MBD of Log-DenseNet has a logarithmic growth.

We observe similar effects on CIFAR10 and SVHN, whose performance versus computational cost plots are deferred to the appendix.

These comparisons suggest that to reach the same accuracy, the sparse Log-DenseNet costs about the same computation as the DenseNet, but is capable of scaling to much higher depths.

We also note that using naïve implementations, and a fixed batch size of 16 per GPU, DenseNets (52, 24) already have difficulties fitting in the 11GB RAM, but Log-DenseNet can fit models with n > 100 with the same g. We defer the plots for number of parameters versus error rates to the appendix as they look almost the same as plots for FLOPS versus error rates.

On the more challenging ILSVRC2012 BID29 , we observe that Log-DenseNet V2 can achieve comparable error rates to DenseNet.

Specifically, Log-DenseNet V2 is more computationally efficient than ResNet BID8 ) that do not use bottlenecks (ResNet18 and ResNet34): Log-DenseNet V2 can achieve lower prediction errors with the same computational cost.

However, Log-DenseNet V2 is not as computationally efficient as ResNet with bottlenecks (ResNet 50 and ResNet101), or DenseNet.

This implies there may be a trade-off between the shortcut connection density and the computation efficiency.

For problems where shallow networks with dense connections can learn good predictors, there may be no need to scale to very deep networks with sparse connections.

However, the proposed Log-DenseNet provides a reasonable trade-off between accuracy and scalability for tasks that require deep networks, as in Sec. 4.2.

We show that short backpropagation distances are important for networks that have shortcut connections: if each layer has a fixed number of shortcut inputs, they should be placed to minimize MBD.

Based on this principle, we design Log-DenseNet, which uses O(L log L) total shortcut connections on a depth-L network to achieve 1 + log L MBD.

We show that Log-DenseNets improve the performance and scalability of tabula rasa fully convolutional DenseNets on CamVid.

Log-DenseNets also achieve competitive results in visual recognition data-sets, offering a trade-off between accuracy and network depth.

Our work provides insights for future network designs, especially those that cannot afford full dense shortcut connections and need high depths, like FCNs.

8 smallest interval in the recursion tree such that i, j ∈ [s, t].

Then we can continue the path to x j by following the recursion calls whose input segments include j until j is in a key location set.

The longest path is then the depth of the recursion tree plus one initial jump, i.e., 2 + log log L. Figure 5a shows the average number of input layers for each feature layer in LogLog-DenseNet.

Without augmentations, lglg_conn on average has 3 to 4 connections per layer.

With augmentations using Log-DenseNet, we desire each layer to have four inputs if possible.

On average, this increases the number of inputs by 1 to 1.5 for L ∈ (10, 2000).

We follow BID14 to optimize the network using 224x224 random cropped images with RMSprop.

The learning rate is 1e-3 with a decay rate 0.995 for 700 epochs.

We then fine-tune on full images with a learning rate of 5e-4 with the same decay for 300 epochs.

The batch size is set to 6 during training and 2 during fine-tuning.

We train on two GTX 1080 GPUs.

We use no preprocessing of the data, except left-right random flipping.

Following BID1 , we use the median class weighting to balance the weights of classes, i.e., the weight of each class C is the median of the class probabilities divided by the over the probability of C.C.2 COMPUTATIONAL EFFICIENCY ON CIFAR10 AND SVHN Fig. 6a and Fig. 6b illustrate the trade-off between computation and accuracy of Log-DenseNet and DenseNets on CIFAR10 and SVHN.

Log-DenseNets V2 and DenseNets have similar performances on these data-sets: on CIFAR10, the error rate difference at each budget is less than 0.2% out of 3.6% total error; on SVHN, the error rate difference is less than 0.05% out of 1.5%.

Hence, in both cases, the error rates between Log-DenseNet V2 and DenseNets are around 5%.

13 Figure 8: Performance of LogLog-DenseNet (red) with different hub multiplier (1 and 3).

Larger hubs allow more information to be passed by the hub layers, so the predictions are more accurate.

This section experiments with LogLog-DenseNet and show that there are more that just MBD that affects the performance of networks.

Ideally, since LogLog-DenseNet have very small MBD, its performance should be very close to DenseNet, if MBD is the sole decider of the performance of networks.

However, we observe in Fig. 8a that LogLog-DenseNet is not only much worse than Log-DenseNet and DenseNet in terms accuracy at each given computational cost (in FLOPS), it is also widening the performance gap to the extent that the test error rate actually increases with the depth of the network.

This suggests there are more factors at play than just MBD, and in deep LogLog-DenseNet, these factors inhibit the networks from converging well.

One key difference between LogLog-DenseNet's connection pattern to Log-DenseNet's is that the layers are not symmetric, in the sense that layers have drastically different shortcut connection inputs.

In particular, while the average input connections per layer is five (as shown in Fig. 5a ), some nodes, such as the nodes that are multiples of L 1 2 , have very large in-degrees and out-degrees (i.e., the number of input and output connections).

These nodes are given the same number of channels as any other nodes, which means there must be some information loss passing through such "hub" layers, which we define as layers that are densely connected on the depth zero of lglg_conn call.

Hence a natural remedy is to increase the channel size of the hub nodes.

In fact, Fig. 8b shows that by giving the hub layers three times as many channels, we greatly improve the performance of LogLog-DenseNet to the level of Log-DenseNet.

This experiment also suggests that the layers in networks with shortcut connections should ensure that high degree layers have enough capacity (channels) to support the amount of information passing.

We show additional semantic segmentation results in Figure 9 .

We also note in Figure 5b how the computation is distributed through the 11 blocks in FC-DenseNets and FC-Log-DenseNets.

In particular, more than half of the computation is from the final two blocks because the final blocks have high resolutions, making them exponentially more expensive than layers in the mid depths and final layers of image classification networks.

@highlight

We show shortcut connections should be placed in patterns that minimize between-layer distances during backpropagation, and design networks that achieve log L distances using L log(L) connections.
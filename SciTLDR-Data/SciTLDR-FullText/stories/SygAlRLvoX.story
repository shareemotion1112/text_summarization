We propose a new anytime neural network which allows partial evaluation by subnetworks with different widths as well as depths.

Compared to conventional anytime networks only with the depth controllability, the increased architectural diversity leads to higher resource utilization and consequent performance improvement under various and dynamic resource budgets.

We highlight architectural features to make our scheme feasible as well as efficient, and show its effectiveness in image classification tasks.

When we deploy deep neural network models on resource-constrained mobile devices or autonomous vehicles with a strict real-time latency requirement, it is essential to develop a model which makes the best of available resources.

Although many network compaction techniques including distillation, pruning and quantization have been proposed BID7 BID12 BID4 BID3 BID9 BID6 BID13 , this goal is still challenging because (1) the resource availabilities are continuously changing over time while these resources are being shared with other program instances BID1 , and (2) multiple resources with different characteristics (e.g. computational capacity, memory usage) should be considered together.

Anytime machine learning algorithms have addressed the first issue, how to get the optimal performance under dynamic resource budgets, by allowing us to activate only a part of the model with graceful output quality degradation BID14 BID2 .

Most anytime algorithms based on deep neural networks appear in the form of early termination of forward processing according to the current resource budget or the difficulty of the given task BID8 BID0 BID10 .

In other words, the conventional anytime networks are trained to embed many potential sub-networks with different effective depths so that the best one can be chosen according to the current budget.

In this work, we propose a new type of the anytime neural network, doubly nested network, to solve the other issue, more efficient utilization of multiple heterogeneous resources.

The proposed network can be sliced along the width as well as the depth to generate more diverse sub-networks than the conventional anytime networks allowing the sub-network extraction only along the depth-wise direction.

As depicted in FIG0 , the increased degree of freedom enables us to get higher resource utilization in the devices constrained by dynamically changing resource budget with multiple criteria.

Causal convolution It is straightforward to form a sub-network along the depth by appending a separate output generation stage to the final convolution layer of the sub-network.

Since one specific layer's output does not depend on the following (upper) layers' outputs, the jointly trained subnetwork does not suffer from the performance degradation even after the extraction.

However, this approach does not work along the width because of the interdependence between two nodes at different horizontal locations (e.g. different channels).

To address this issue, we propose a channelcausal convolution where i-th channel group in one layer is calculated only with activation values from the channel groups from the first to i-th channel group in the previous layer as shown in the right of FIG1 .

The circle indicates the feature map while the square indicates the classifier.

Color refers each channel.

Our network based on the causal convolution allows us to extract the sub-network easily along any directions by making both horizontal and vertical data flow unidirectionally.

Output generation stage sharing fully-connected layers Rather than having a separate fullyconnected (FC) layer for one sub-network to generate the final output (e.g. a predicted class given image input), our network is designed to have the FC layers each of which takes only a part of activations from the preceding convolution layers and produce the final output for one sub-network by averaging multiple FC layers' outputs as depicted in FIG1 .

Sharing the FC layers between the sub-networks at the same depth helps us to have similar computational and memory costs of the FC layers in the depth-controllable anytime network BID8 even with much more possible output locations.

We can obtain a loss function for each sub-network: DISPLAYFORM0 where L and C refer to the number of possible vertical and horizontal partitions and N is the number of classes.

y i is a target label of class i andŷ DISPLAYFORM1

Experimental setup We evaluated the proposed method on the CIFAR-10 and the SVHN datasets.

Similarly to the ResNet-32 model BID5 , our full network architecture consists of one convolution layer fed by external input, the following 15 residual blocks and fully-connected layers for the final output generation.

The network has 16 possible output locations along the depth from the first convolution layer and all residual blocks, and 22 locations along the width.

Thus, we can extract 16×22 sub-networks with different widths and depths from the base network.

Resource usages of the sub-networks As the selected sub-network gets deeper or wider, all computational and memory requirements such as the number of MAC (multiply-accumulate) operations, the number of parameters and the size of the largest feature map increase.

However, their rates of the increase are different from each other as shown in FIG3 .

This means that our scheme can benefit from the larger diversity of resource usage compared to the conventional anytime methods.

Comparison with other methods One of the key advantages of the proposed architecture is nontrivial nesting of sub-networks along the width direction.

FIG4 shows that our scheme outperforms two straightforward vertical slicing schemes (Brute-force slicing, Fine-tuning) that can generate sub-networks with different widths to a large extent without significant performance degradation compared to the upper bound (Full training).

We revealed that resource-constrained devices could benefit from the architectural diversity enriched by our anytime prediction scheme.

Our future works include adding adaptive conditioning BID11 which modulates intermediate activations or weight parameters depending on the current sub-network configuration to improve the performance only with a small increase of conditioning parameters.

@highlight

We propose a new anytime neural network which allows partial evaluation by subnetworks with different widths as well as depths.
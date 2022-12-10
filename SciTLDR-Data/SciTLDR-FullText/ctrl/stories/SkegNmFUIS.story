There are many differences between convolutional networks and the ventral visual streams of primates.

For example, standard convolutional networks lack recurrent and lateral connections, cell dynamics, etc.

However, their feedforward architectures are somewhat similar to the ventral stream, and warrant a more detailed comparison.

A recent study found that the feedforward architecture of the visual cortex could be closely approximated as a convolutional network, but the resulting architecture differed from widely used deep networks in several ways.

The same study also found, somewhat surprisingly, that training the ventral stream of this network for object recognition resulted in poor performance.

This paper examines the performance of this network in more detail.

In particular, I made a number of changes to the ventral-stream-based architecture, to make it more like a DenseNet, and tested performance at each step.

I chose DenseNet because it has a high BrainScore, and because it has some cortex-like architectural features such as large in-degrees and long skip connections.

Most of the changes (which made the cortex-like network more like DenseNet) improved performance.

Further work is needed to better understand these results.

One possibility is that details of the ventral-stream architecture may be ill-suited to feedforward computation, simple processing units, and/or backpropagation, which could suggest differences between the way high-performance deep networks and the brain approach core object recognition.

For most cortical areas, these are layers L2/3, L4, L5, and L6.

For V1, these are L2/3(blob), L2/3(interblob), L4B, L4Cα, L4Cβ.

For LGN, they are parvo, magno, and koniocellular divisions.

[2] reported validation accuracy of 79% on CIFAR-10, for a similar ventral-stream sub-network that 49 omitted connections with FLNe<0.15, and was trained for 50 epochs with the Adam update algorithm.

In the present study, networks were trained for 300 epochs, using SGD with momentum 0.9, starting 51 with learning rate 0.1 and reducing it by 10x every 100 epochs.

This resulted in validation accuracy 52 of 84.59%.

A standard DenseNet (github.com/kuangliu/pytorch-cifar) was trained using the same 53 procedure, resulting in validation accuracy of 95.36%.

To understand the basis of this performance gap, I created hybrid networks, with features of both the 55 ventral-stream network (VSN) and DenseNet.

The VSN has a wide range of kernel sizes, optimized 56 to fill realistic receptive field sizes.

In the first hybrid (H1), all kernel sizes of the VSN were set 57 to 3x3.

The VSN also has a wide range of sparsity, with some connections consisting mostly of 58 zeros.

In the second hybrid network (H2), in addition to using 3x3 kernels, I eliminated pixel-wise 59 sparsity, and limited channel-wise sparsity so that at least half of the input channels were used in 60 each connection.

Thirdly (H3), I replaced each layer with a two-layer bottleneck module, specifically 61 a 1x1-kernel layer followed by 3x3 layer with four times fewer channels.

The number of channels Most of these modifications improved performance, when applied cumulatively to make the ventral-

stream increasingly similar to a DenseNet (Table 2, [5] .

In the monkey inferotemporal cortex data, mean selectivity is 3.5, and mean sparseness is 12.51.

The ventral-stream model has much higher means, and DenseNet has much lower means.

C. Representational dissimilarity, using a subset of images from [6] .

The plotted values are percentiles of one minus the Pearson correlations between responses to different stimuli.

Monkey cell data shows relatively low values (high similarity) throughout the lower-right quarter of this matrix (spanning non-animal natural and artificial images) [6] , but neither of the deep networks does.

efficiently, or the connection pattern of the ventral stream may be better suited to extracting certain 135 feature combinations in unsupervised learning than to communicating gradients through many layers.

<|TLDR|>

@highlight

An approximation of primate ventral stream as a convolutional network performs poorly on object recognition, and multiple architectural features contribute to this. 
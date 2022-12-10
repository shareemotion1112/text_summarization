Recurrent convolution (RC) shares the same convolutional kernels and unrolls them multiple times, which is originally proposed to model time-space signals.

We suggest that RC can be viewed as a model compression strategy for deep convolutional neural networks.

RC reduces the redundancy across layers and is complementary to most existing model compression approaches.

However, the performance of an RC network can't match the performance of its corresponding standard one, i.e. with the same depth but independent convolutional kernels.

This reduces the value of RC for model compression.

In this paper, we propose a simple variant which improves RC networks: The batch normalization layers of an RC module are learned independently (not shared) for different unrolling steps.

We provide insights on why this works.

Experiments on CIFAR show that unrolling a convolutional layer several steps can improve the performance, thus indirectly plays a role in model compression.

Deep convolution neural networks (DCNNs) have achieved ground-breaking results on a broad range of fields, such as computer vision BID0 and natural language processing BID1 .

Unfortunately, DCNNs are both computation intensive and memory intensive for industrial applications.

Many approaches have been proposed recently to obtain more compact DCNNs while keep their performance as much as possible.

Conceptually, those approaches fall in two categories: 1) Reduce the computational cost or memory usage of big DCNNs by weights pruning, quantization and sharing BID2 BID3 BID4 BID5 ; 2) Improve the performance of small DCNNs by knowledge distillation BID6 or other techniques.

In this paper, we explore a potential compression strategy which is complementary to most of the existing approaches: training a recurrent convolutional (RC) neural network.

As the name suggests, the same convolutional kernels are unrolled multiple times on the computational graph.

This is a weights sharing mechanism applied to the whole layer.

Suppose there is a network with n RC layers each of which unrolls k times, then we say its depth is nk.

If the performance of this network can match the performance of a standard DCNN with nk layers (Suppose other conditions are the same), we could say we compress the standard one with factor k.

Then we can further compress the obtained RC network by applying other existing approaches, such as weight quantization.

The key intuition is that RC can reduce the redundancy across layers by sharing weights of the whole layer.

While most existing approaches work at a layer-wise manner and only remove part of a layer.

This is why we say RC is a complementary strategy.

However, we find the performance of RC networks can't match the performance of DCNNs with the same depth.

This significantly reduces the value of RC for model compression.

In this paper, we aim to improve the performance of RC in a simple way.

Specifically, we learn the batch normalization layers (BN) BID7 independently at each unrolling step.

We describe our insights in the next section.

Experiments on CIFAR dataset demonstrate that such a simple variant works well.

We also compare RC networks with their corresponding standard ones.

The idea of RC is not new.

Many works have used it to model time-space signals BID8 or to obtain a larger receptive field BID9 .

However, to our knowledge, none of them view RC as a potential model compression strategy and none of them compare RC networks with their corresponding standard ones strictly.

The intention of this paper is to show that RC is a considerable or at least a heuristic solution for model compression.

In this paper, we focus on image classification via DCNNs.

Thus there are no sequential inputs for RC layers.

The state equation of RC is DISPLAYFORM0 where f is a set of differentiable operators including convolutions and f is parameterized by w. Here, we set f to the so-called res-block used in ResNet BID0 .

Each res-block is a composition of convolutional layers, BN layers and Relu activations BID10 , as shown in figure 1(a).

h i and h i+1 are the input and output of a res-block at ith unrolling step respectively.

When we directly unroll each res-block several steps during training, the performance is not satisfactory.

We believe that it is caused by the shared BN layers.

A BN layer captures the moving mean and variance of its input, then scales and shifts the normalized input by its learned parameters.

However, there are no reasons to expect the statistics of the inputs over different unrolling steps are the same.

When we share the BN layers across unrolling steps, we merge those statistics over unrolling steps into a single mean and variance vector.

This would hurt the performance of RC networks.

Thus, we apply independent BN layers at each unrolling step.

Then the number of BN layers is proportional to the number of unrolling steps.

Since we focus on image classification (no sequential inputs), we can set the total unrolling steps of each res-block to a fixed number.

Using independent BN layers may also improve the representation power of RC networks.

Because the parameters of BN are different across unrolling steps, the overall mapping function at each unrolling step is no longer the same.

We can regard the parameters of BN layers as the additional sequential inputs to an RC module.

Then the state equation of RC can be reformalized as follows: DISPLAYFORM1 where w c is the shared convolution kernels and b i+1 is the parameters of (i + 1)th group of BN layers.

This is the most general form of the state equation of recurrent neural networks although there is no explicit sequential input.

Moreover, using independent BN layers is cheap on both computation and memory point of views, compared with the convolutional layers.

Now we turn to some practical considerations.

Due to its recurrent nature, the input size and output size of each RC block should be the same.

We change the size of feature map outside RC blocks.

Specifically, we reduce the spatial resolution via average pooling.

We reduce the spatial resolution and increase the channels simultaneously via the invertible downsampling operation described in BID11 , which consists in reorganizing the initial spatial channels into the 4 spatially decimated copies obtainable by 2 × 2 spatial sub-sampling.

To avoid gradient explosion and make the training more stable, we use gradient clip when training RC networks.

We do three sets of experiments on CIFAR-10 and CIFAR-100 BID12 datasets: 1) Study the effects of BN layer usage; 2) Compare RC networks with the standard ones; 3) Compare RC networks with different unrolling steps.

Both CIFAR-10 and CIFAR-100 have 50K training samples and 10K test samples, each of which is a 32 × 32 color image.

The former has 10 classes of images while the later has 100 classes of images.

All neural network models are implemented with Pytorch BID13 .

We train each model 3 times and average its test errors.

See B for detailed experimental settings.

We choose the res-block in 1(a) as the basic unrolling cell and we set the number of cells of each RC network to 2.

Denote RC-i as the network whose cells unroll i steps.

We train RC-1, RC-2 and RC-4 in this paper.

For fair comparisons with the standard DCNNs, we slightly modify ResNet-18 used in BID0 such that its computational graph is exactly the same as the one of unrolled RC-4, except the weight sharing mechanism.

See A for detailed network architecture.

As shown in FIG2 , the moving means of the first BN layer of RC-4 vary across unrolling steps.

See C.2 for more results.

Further, as shown in table 1, using independent BN layers improves the accuracy of RC-4 by a large margin.

Using shared BN layers has even lower accuracy than without using BN layers.

Those results support the independent learning of BN layers and our discussions in section 2.As shown in table 2, the accuracy of RC networks is improved when their cells are unrolled more steps.

But when the standard ResNet and the unrolled RC network have the same depth, the former is still better (the gap is not large).

It is worth noting that RC-4 whose original depth is 6 has better performance than Resnet-10.

We also show the number of parameters of each model trained on CIFAR-10 without further compression.

As shown in FIG3 , the convergence speed of RC networks and the standard ones looks similar.

In another word, RC networks are easy to train.

We suggest recurrent convolution is a considerable strategy for model compression.

RC reduces the redundancy across layers (which is ignored by most of the compression methods).

We can train an RC network and then further compress it via existing approaches.

We also suggest it is significantly better to learn independent BN parameters at each unrolling step when training RC networks.

Experiments on CIFAR dataset demonstrate that unrolling the same convolutional layer several steps can improve the accuracy of the whole network, thus indirectly plays a role in model compression.

We believe that the performance of RC could be further improved.

Denote Block(i, k) as the kth unrolling step of an RC res-block with i channels.

Denote InvP ool FORMULA1 as the invertible downsampling operation described in BID11 .

Then the architecture of RC-4 is: DISPLAYFORM0 For RC-1, AvgP ool(2) is applied before the first unrolling step.

For RC-2, AvgP ool(2) is applied before the second unrolling step.

Moreover, we slightly modify the standard ResNet such that its computational graph is exactly the same as the one of unrolled RC network.

For fair comparisons, most of the hyper-parameters keep the same for all experiments.

We train all of the networks via SGD with moment 0.9 and weight decay 1e-4.

Batchsize is set to 128.

Initial learning rate is set to 0.1.

We reduce the learning rate by factor 10 at 60th epoch and 90th epoch.

The networks whose depth are larger than 10 are trained with 160 epochs.

While the networks whose depth are smaller or equal to 10 are trained with 120 epochs.

See table 2 for the depth of networks.

For all experiments, we clip the gradients to [−0.1, 0.1].

We find gradient clip makes the training process more stable and improves the performance for RC networks.

For standard networks, gradient clip has nearly no effect on their performance.

Other hyper-parameters are set to Pytorch's default settings.

Training loss and test errors on CIFAR-10 are shown in FIG3 .

The convergence speed of RC networks and standard networks is similar.

Step 1Step 2Step 3Step 4 (c) γ Step 1Step 2Step 3Step 4 (d) β

The inference process of a BN layer is formulated as: DISPLAYFORM0 where µ is the moving mean and σ 2 is the moving variance.

γ and β are the learned parameters by SGD.

We show those variables of both the first BN layer and the second BN layer of RC-4 trained on CIFAR-10 in FIG6 and 4 respectively.

Step 1Step 2Step 3Step 4 (a) µ Step 1Step 2Step 3Step 4 (b) σ Step 1Step 2Step 3Step 4(d) β

@highlight

Recurrent convolution for model compression and a trick for training it, that is learning independent BN layres over steps.

@highlight

The author modifies the recurrent convolution neural network (RCNN) with independent batch normalization, with the experimental results on RCNN compatible with the ResNet neural network architecture when it contains the same number of layers.
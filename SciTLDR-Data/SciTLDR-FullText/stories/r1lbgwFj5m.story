Pruning is a popular technique for compressing a neural network: a large pre-trained network is fine-tuned while connections are successively removed.

However, the value of pruning has largely evaded scrutiny.

In this extended abstract, we examine residual networks obtained through Fisher-pruning and make two interesting observations.

First, when time-constrained, it is better to train a simple, smaller network from scratch than prune a large network.

Second, it is the architectures obtained through the pruning process  --- not the learnt weights --- that prove valuable.

Such architectures are powerful when trained from scratch.

Furthermore, these architectures are easy to approximate without any further pruning: we can prune once and obtain a family of new, scalable network architectures for different memory requirements.

Deep neural networks excel at a multitude of tasks BID11 , but are typically cumbersome, and difficult to deploy on embedded devices.

This can be rectified by compressing the network; specifically, reducing the number of parameters it uses in order to reduce its runtime memory.

This also reduces the number of operations that have to be performed; making the network faster.

A popular means of doing this is through pruning.

One trains a large network and fine-tunes it while removing connections in succession.

This is an expensive procedure that often takes longer than simply training a smaller network from scratch.

Does a pruned-and-tuned network actually outperform a simpler, smaller counterpart?

If not, is there any benefit to pruning?In this work, we show that for a given parameter budget, pruned-and-tuned networks are consistently beaten by networks with simpler structures (e.g. a linear rescaling of channel widths) trained from scratch.

This indicates that there is little value in the weights learnt through pruning.

However, when the architectures obtained through pruning are trained from scratch (i.e. when all weights are reinitialised at random and the network is trained anew) they surpass their fine-tuned equivalents and the simpler networks.

Moreover, these architectures are easy to approximate; we can look at which connections remain after pruning and derive a family of copycat architectures that when trained from scratch display similar performance.

This gives us a new set of compact, powerful architectures.

Typically, a network is pruned by either setting unimportant weights to zero, resulting in a network with large, sparse weight matrices BID4 BID10 BID2 BID3 BID15 BID19 , or by severing channel connections to produce a smaller, dense network BID14 BID16 BID5 .

In this work we use Fisher-pruning BID16 , a powerful channel pruning method, as recent work BID17 has empirically demonstrated that on embedded devices this technique produces small, dense networks that both run efficiently, and perform well.

These networks are therefore applicable to real-time applications where efficiency is of the essence.

Concurrent to this work, BID13 have demonstrated the benefits of training pruned models from scratch for a variety of other pruning techniques.

They postulate that pruning may be seen as a form of architecture search.

This is contrary to the work of BID1 who hypothesise that the pruning process finds well-initialised weights.

Very recently, BID12 proposed a single-shot pruning scheme to produce a reduced architecture that is then trained from scratch; the benchmarks of this work and our discovered architectures are given in Section 3.2.

Here, we perform a set of experiments to answer our two questions: (i) Does a pruned network outperform a smaller simpler network? (ii) Is the pruned architecture itself useful?

We evaluate the performance of our networks by their classification error rate on the test set of CIFAR-10 ( BID9 .

Implementation details are given at the end of the section.

For our base network, we use WRN-40-2 -a WideResNet BID20 with 40 layers, and channel width multiplier 2.

We chose this network as it features modular blocks and skip connections, and is therefore representative of a large number of commonly used networks.

It is reasonably compact and doesn't have a large, redundant fully connected layer.

It has 2.2M parameters in total, the bulk of which lie in 18 residual blocks, each containing two convolutional layers.

Let us denote each block as B(N i , N m , N o ) for variable N i , N m , N o -the input has N i channels and the first convolutional layer has a N m channel output; this goes through the second layer which outputs DISPLAYFORM0 The network is first trained from scratch, and is then Fisher-pruned BID16 .

We prune the channels of the activations between the convolutions in each block; this has the effect of introducing a bottleneck (changing N m ), reducing the number of parameters used in each convolution.

We alternate between fine-tuning the network and removing the channel that has the smallest estimated effect on the loss, as in BID16 .

Before each channel is removed, the test error and parameter count for the network is recorded.

The resulting trajectory of Test Error versus Number of Parameters is represented by the red curve in FIG0 .We compare this process to training smaller, simpler networks from scratch.

We train WRN-40-k networks -the yellow curve in FIG0 -treating k as an architectural hyperparameter which we can vary to control the total number of parameters.

We also train WRN-40-2 networks where we apply a scalar multiplier z to the middle channel dimension in each block -B (N i , zN o , N o ) giving the green curve in FIG0 .

We similarly vary z to control the total number of parameters.

In both cases, we vary the relevant architectural hyperparameter (k or z) to produce networks that have between 200K and 2M parameters, in increments of roughly 100K.

We can see that these simple networks trained from scratch consistently outperform the pruned-and-tuned networks.

This difference is markedly more pronounced the smaller the networks get.

Curves for networks trained from scratch show the average error across 3 runs.

The red curve corresponds to a WRN-40-2 undergoing Fisher-pruning and fine-tuning.

The yellow curve represents WRN-40-k networks trained from scratch for various k whereas the green curve represents bottlenecked WRN-40-2 networks (also trained from scratch).

Notice that the green and yellow curve are almost always below the red: it is preferable to train smaller architectures from scratch rather than prune.

If we take Fisher-pruned architectures obtained along the red curve and train them from scratch we get the blue curve.

Finally, if we profile the channel structure of a Fisher-pruned architecture, linearly scale it and train from scratch, we get the pink curve which closely follows the blue curve.

Note that these blue and pink networks tend to outperform all others.

The channel profile for a Fisher-pruned network after 500 channels have been pruned.

The left bar chart shows the percentage of remaining channels at the bottleneck in each residual block and the right bar chart shows the actual number.

We use this profile to produce copycat networks.

It may seem thus far that pruning is a pointless endeavor.

However, when we take the architectures produced along the pruning trajectory (at the same increments as above) and train them from scratch we obtain the blue curve in FIG0 .

We can see that these networks outperform both (i) the fine-tuned versions of the same architectures and (ii) the simpler networks for most parameter budgets.

This supports the notion that the true value of pruning is as a form of architecture search.

FIG1 shows how many channels remain at the bottleneck of each block (the various N m values) in the pruned-and-tuned WRN-40-2 after 500 channels have been Fisher-pruned (note that this profile is consistent with those obtained after further pruning).

Notice that channels in later layers tend to be pruned; they are more expendable than those in earlier layers.

It is intriguing that channels are rarely pruned in blocks 7 and 13; these blocks both contain a strided convolution that reduces the spatial resolution of the image representation.

It is imperative that information capacity is retained for this process.

These observations are consistent with those made in previous works BID18 BID7 BID8 .To assess the merits of this particular structure, we train copycat Fisher architectures -architectures designed to mimic those obtained through Fisher-pruning.

Specifically, we train a range of architectures from scratch where N m in each block is proportional to that found in the Fisher-pruned architecture -block j can be represented by B(N i , ??N mj , N o ): N mj is a block-specific value corresponding to the height of each bar in FIG1 (right), and ?? scales the whole architecture.

We vary ?? to produce networks with different parameter counts as before, which are then trained from scratch.

These are represented by the pink curve in FIG0 .

For most parameter budgets these networks perform similarly to those found through Fisher-pruning; this means we can simply prune a network once to find a powerful, scalable network architecture.

Furthermore, the resulting networks are competitive.

We compare them to the WideResNets produced by the pruning method of BID12 : one of which gets 5.85% error on CIFAR-10 and has 858K parameters, and another gets 6.63% error with 548K parameters.

Comparably, the copycat WideResNets in this work achieve (on average) 5.66% with 800K parameters, and 6.35% error with 500K parameters.

Implementation Details To train a network from scratch, we use SGD with momentum to minimise the cross-entropy loss for 200 epochs using mini-batches of size 128.

Images were augmented using horizontal flips and random crops.

The initial learning rate is 0.1 and is decayed by a factor of 0.2 every 60 epochs.

We use weight decay of 5 ?? 10 ???4 and momentum of 0.9.

For Fisher-pruning we fine-tune the learnt network with the lowest learning rate reached during training (8 ?? 10 ???4 ) and momentum and weight decay as above.

We measure the approximate effect on the change in loss BID16 for each candidate channel in the network over 100 update steps and then remove the channel with the lowest value.

We experimented with several different values for the FLOP penalty hyperparameter but found this made little difference.

We therefore set it to zero.

Pruning is a very expensive procedure, and should be avoided when one is time-constrained.

We show that under such constraints it is preferable to train a smaller, simpler network from scratch.

Our work supports the view that pruning should be seen as a form of architecture search; it is the resulting structure -not the learnt weights -that is important.

Pruned architectures trained from scratch perform well, and are easily emulated, as demonstrated by our copycat networks.

These are a step towards a more efficient architecture for residual networks.

Future work could entail expanding this analysis to other network types, datasets, and pruning schema.

It would also be possible to use distillation techniques BID21 between our pruned architectures and the original architecture to further boost performance .

@highlight

Training small networks beats pruning, but pruning finds good small networks to train that are easy to copy.
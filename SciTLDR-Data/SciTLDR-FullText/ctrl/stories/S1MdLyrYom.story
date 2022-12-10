High performance of deep learning models typically comes at cost of considerable model size and computation time.

These factors limit applicability for deployment on memory and battery constraint devices such as mobile phones or embedded systems.

In this work we propose a novel pruning technique that eliminates entire filters and neurons according to their relative L1-norm as compared to the rest of the network, yielding more compression and decreased redundancy in the parameters.

The resulting network is non-sparse, however, much more compact and requires no special infrastructure for its deployment.

We prove the viability of our method by achieving 97.4%, 47.8% and 53% compression of LeNet-5, ResNet-56 and ResNet-110 respectively, exceeding state-of-the-art compression results reported on ResNet without losing any performance compared to the baseline.

Our approach does not only exhibit good performance, but is also easy to implement on many architectures.

While deep learning models have become the method of choice for a multitude of applications, their training requires a large number of parameters and extensive computational costs (energy, memory footprint, inference time).

This limits their deployment on storage and battery constraint devices, such as mobile phones and embedded systems.

To compress deep learning models without loss in accuracy, previous work proposed pruning weights by optimizing network's complexity using second order derivative information BID1 BID4 .

While second order derivative introduces a high computational overhead, BID7 BID9 explored low rank approximations to reduce the size of the weight tensors.

Another line of work BID3 BID14 , proposed to prune individual layer weights with the lowest absolute value (nonstructural sparsification of layer weights).

BID2 followed the same strategy while incorporating quantization and Huffman coding to further boost compression.

While the aforementioned methods considered every layer independently, BID12 proposed to prune the network weights in a class-blind manner, e.g. individual layer weights are pruned according to their magnitude as compared to all weights in the network.

Noteworthy, all approaches that prune weights non-structurally, generally result in high sparsity models that require dedicated hardware and software.

Structured pruning alleviates this by removing whole filters or neurons, producing a non-sparse compressed model.

In this regard, BID11 proposed channel-wise pruning according to the L1-norm of the corresponding filter.

BID15 learned a compact model based on learning structured sparsity of different parameters.

A data-free algorithm was implemented to remove redundant neurons iteratively on fully connected layers in BID13 .

In BID6 , connections leading to weak activations were pruned.

Finally, BID16 pruned neurons by measuring their importance with respect to the penultimate layer.

Generally, in structured pruning, each layer is pruned separately, which requires calculation of layer importance before training.

This work features two key components: a) Blindness: all layers are considered simultaneously; blind pruning was first introduced by BID12 to prune individual weights; b) Structured Pruning: removal of entire filters instead of individual weights.

To the best of our knowledge, we are the first to use these two components together to prune filters based on their relative L1-norm compared to the sum of all filters' L1-norms across the network, instead of pruning filters according to their L1-norm within the layer BID11 , inducing a global importance score for each filter.

The contribution of this paper is two-fold: i) Proposing a structured class-blind pruning technique to compress the network by removing whole filters and neurons, which results in a compact non-sparse network with the same baseline performance.

ii) Introducing a visualization of global filter importance to devise the pruning percentage of each layer.

As a result, the proposed approach achieves higher compression gains with higher accuracy compared to the state-of-the-art results reported on ResNet-56 and ResNet-110 on the CIFAR10 dataset BID8 .

Consider a network with a convolutional (conv) layer and a fully connected (fc) layer.

We denote each filter F ilter i , where i ∈ [1, F ], and F is the total number of filters in the conv layer.

Each filter is a 3D kernel space consisting of channels, where each channel contains 2D kernel weights.

For the fc layer, we denote W m , a 1-D feature space containing all the weights connected to certain neuron N euron m , with m ∈ [1, N ] and N denoting the number of neurons.

It should be noted that We do not prune the classification layer.

Each pruning iteration in our algorithm is structured as follows:Algorithm 1 Pruning procedure 1: for i ← 1 to F do loop over filters of a conv layer 2: DISPLAYFORM0 calculate L1-norm of all channels' kernel weights 3: if norm_conv ( i) < threshold then 12: prune(F ilter i ) remove filter if its normalized norm is less than threshold 13: for m ← 1 to N do 14: if norm_f c(m) < threshold then 15: prune(N euron m ) remove neuron if its normalized norm is less than threshold Importance calculation.

Although pre-calculation of filters or layers' sensitivity to be pruned is not needed in our method, it can be visualized as part of the pruning criteria.

In our algorithm, blindness implies constructing a hidden importance score, which corresponds to the relative normalized L1-norm.

For instance, the relevant importance for a certain filter in a conv layer w.r.t.

all other filters in all layers is the ratio between the filter's normalized norm and the sum of all filters' normalized norms across the network.

DISPLAYFORM1 Normalization.

As each layer's filters have different number of kernel weights, we normalize filters' L1-norms by dividing each over the number of kernel weights corresponding to the filter (Line 3 and 6 as indicated in Algorithm 1).

Alternatively without normalization, filters with a higher number of kernel weights would have higher probabilities of higher L1-norms, hence lower probability to get pruned.

Retraining process.

Pruning without further adaption, results in performance loss.

Therefore, in order to regain base performance, it is necessary for the model to be retrained.

To this end, we apply an iterative pruning schedule that alternates between pruning and retraining.

This is conducted until a maximum compression is reached without losing the base accuracy.

BID3 0.77 92.00 84.00 Srinivas et al. BID14 0.81 95.84 91.68 Han et al. BID2 0.74 97.45 -.

Table 2 : Results on LeNet-5.

Error% percentage for different percentage of parameters pruned (Par.%); "E.Par%" is the effective pruning percentage after adding the extra indices' storage for non-structured pruning as studied by BID0 3 ExperimentIn order to assess the efficacy of the proposed method, we evaluate the performance of our technique on a set of different networks: first, LeNet-5 on MNIST BID10 ; second, ResNet-56 and ResNet-110 ( BID5 ) on CIFAR-10 BID8 .

We use identical training settings as BID5 , after pruning we retrain with learning rate of 0.05.For ResNet, when a filter is pruned, the corresponding batch-normalization weight and bias applied on that filter are pruned accordingly.

After all pruning iterations are finished, a new model with the remaining number of parameters is created.

We report compression results on the existing benchmark BID11 BID16 .

As shown in Table 1 , we outperform the state-of-the-art compression results reported by BID16 on both ResNet-56 and ResNet-110 with a lower classification error even compared to the baseline.

In Table 2 , while using one-shot pruning, the influence of our method's different components; structured pruning and blindness, is analyzed by removing a component each test, resulting in: i) Non-Structured -pruning applied on weights separately.

ii) Non-Blind -every layer is pruned individually.

Then, the effect of the pruning strategy on the method with all its components is analyzed by comparing: i) Ours-Oneshot -using one-shot pruning and ii) Ours -using iterative pruning.

By comparing the previous versions that are using one-shot pruning, our method has less number of parameters compared to the other versions; ("Non-Structured" and "Non-Blind").Finally, applying pruning iteratively is superior to one-shot pruning.

We also show that our method performs better than previously mentioned non-structured weight pruning techniques BID3 BID14 .

Proposed structured class-blind pruning offers comparable performance as BID2 , without requiring dedicated hardware and software to realize compression.

We presented a novel structured pruning method to compress neural networks without losing accuracy.

By pruning layers simultaneously instead of looking at each layer individually, our method combines all filters and output features of all layers and prunes them according to a global threshold.

We have surpassed state-of-the-art compression results reported on ResNet-56 and ResNet-110 on CIFAR-10 BID16 , compressing more than 47% and 53% respectively.

Also, we showed that only 11K parameters are sufficient to exceed the baseline performance on LeNet-5, compressing more than 97%.

To realize the advantages of our method, no customized hardware or libraries are needed.

It is worth to say that due to removing whole filters and neurons, the pruning percentage reflects the effective model compression percentage.

For the future work, we are dedicated to proving the applicability of our method on several different architectures and datasets.

Hence, we plan to experiment on VGG-16, ResNet on ImageNet and/or other comparable architectures.

<|TLDR|>

@highlight

We propose a novel structured class-blind pruning technique to produce highly compressed neural networks.
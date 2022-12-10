Designing of search space is a critical problem for neural architecture search (NAS) algorithms.

We propose a fine-grained search space comprised of atomic blocks, a minimal search unit much smaller than the ones used in recent NAS algorithms.

This search space facilitates direct selection of channel numbers and kernel sizes in convolutions.

In addition, we propose a resource-aware architecture search algorithm which dynamically selects atomic blocks during training.

The algorithm is further accelerated by a dynamic network shrinkage technique.

Instead of a  search-and-retrain two-stage paradigm, our method can simultaneously search and train the target architecture in an end-to-end manner.

Our method achieves state-of-the-art performance under several FLOPS configurations on ImageNet with a negligible searching cost.

We open our entire codebase at: https://github.com/meijieru/AtomNAS.

Human-designed neural networks are already surpassed by machine-designed ones.

Neural Architecture Search (NAS) has become the mainstream approach to discover efficient and powerful network structures (Zoph & Le (2017) ; Pham et al. (2018) ; ; ).

Although the tedious searching process is conducted by machines, humans still involve extensively in the design of the NAS algorithms.

Designing of search spaces is critical for NAS algorithms and different choices have been explored.

Cai et al. (2019) and Wu et al. (2019) utilize supernets with multiple choices in each layer to accommodate a sampled network on the GPU.

Chen et al. (2019b) progressively grow the depth of the supernet and remove unnecessary blocks during the search.

Tan & Le (2019a) propose to search the scaling factor of image resolution, channel multiplier and layer numbers in scenarios with different computation budgets.

Stamoulis et al. (2019a) propose to use different kernel sizes in each layer of the supernet and reuse the weights of larger kernels for small kernels.

; Tan & Le (2019b) adopts Inverted Residuals with Linear Bottlenecks (MobileNetV2 block) (Sandler et al., 2018) , a building block with light-weighted depth-wise convolutions for highly efficient networks in mobile scenarios.

However, the proposed search spaces generally have only a small set of choices for each block.

DARTS and related methods Chen et al., 2019b; use around 10 different operations between two network nodes. ; Cai et al. (2019) ; Wu et al. (2019) ; Stamoulis et al. (2019a) search the expansion ratios in the MobileNetV2 block but still limit them to a few discrete values.

We argue that more fine-grained search space is essential to find optimal neural architectures.

Specifically, the searched building block in a supernet should be as small as possible to generate the most diversified model structures.

We revisit the architectures of state-of-the-art networks ; Tan & Le (2019b) ; He et al. (2016) ) and find a commonly used building block: convolution -channel-wise operation -convolution.

We reinterpret such structure as an ensemble of computationally independent blocks, which we call atomic blocks.

This new formulation enables a much larger and more fine-grained search space.

Starting from a supernet which is built upon atomic blocks, the search for exact channel numbers and various operations can be achieved by selecting a subset of the atomic blocks.

For the efficient exploration of the new search space, we propose a NAS algorithm named AtomNAS to conduct architecture search and network training simultaneously.

Specifically, an importance factor is introduced to each atomic block.

A penalty term proportional to the computation cost of the atomic block is enforced on the network.

By jointly learning the importance factors along with the weights of the network, AtomNAS selects the atomic blocks which contribute to the model capacity with relatively small computation cost.

Training on large supernets is computationally demanding.

We observe that the scaling factors of many atomic blocks permanently vanish at the early stage of model training.

We propose a dynamic network shrinkage technique which removes the ineffective atomic blocks on the fly and greatly reduce the computation cost of AtomNAS.

In our experiment, our method achieves 75.9% top-1 accuracy on ImageNet dataset around 360M FLOPs, which is 0.9% higher than state-of-the-art model (Stamoulis et al., 2019a) .

By further incorporating additional modules, our method achieves 77.6% top-1 accuracy.

It outperforms MixNet by 0.6% using 363M FLOPs, which is a new state-of-the-art under the mobile scenario.

In summary, the major contributions of our work are:

1.

We propose a fine-grained search space which includes the exact number of channels and mixed operations (e.g., combination of different convolution kernels).

2.

We propose an efficient end-to-end NAS algorithm named AtomNAS which can simultaneously search the network architecture and train the final model.

No finetuning is needed after AtomNAS finishes.

3.

With the proposed search space and AtomNAS, we achieve state-of-the-art performance on ImageNet dataset under mobile setting.

Recently, there is a growing interest in automated neural architecture design.

Reinforce learning based NAS methods (Zoph & Le, 2017; Tan & Le, 2019b; a) are usually computational intensive, thus hampering its usage with limited computational budget.

To accelerate the search procedure, ENAS (Pham et al., 2018) represents the search space using a directed acyclic graph and aims to search the optimal subgraph within the large supergraph.

A training strategy of parameter sharing among subgraphs is proposed to significantly increase the searching efficiency.

The similar idea of optimizing optimal subgraphs within a supergraph is also adopted by ; Wu et al. (2019) ; Guo et al. (2019); Cai et al. (2019) .

A prominent disadvantage of the above methods is their coarse search spaces only include limited categories of properties, e.g. kernel size, expansion ratio, the number of layer, etc.

Because of the restriction of search space, it is difficult to learn optimal architectures under computational resource constraints.

On the contrary, our method proposes the fine-grained search space to enable searching more flexible network architectures under various resource constraints.

Assuming that many parameters in the network are unnecessary, network pruning methods start from a computation-intensive model, identify the unimportant connections and remove them to get a compact and efficient network.

Early method (Han et al., 2016) simultaneously learns the important connections and weights.

However, non-regularly removing connections in these works makes it hard to achieve theoretical speedup ratio on realistic hardwares due to extra overhead in caching and indexing.

To tackle this problem, structured network pruning methods (He et al., 2017b; Liu et al., 2017; Luo et al., 2017; Ye et al., 2018; Gordon et al., 2018) are proposed to prune structured show that in structured network pruning, the learned weights are unimportant.

This suggests structured network pruning is actually a neural architecture search focusing on channel numbers.

Our method jointly searches the channel numbers and a mix of operations, which is a much larger search space.

We formulate our neural architecture search method in a fine-grained search space with the atomic block used as the basic search unit.

An atomic block is comprised of two convolutions connected by a channel-wise operation.

By stacking atomic blocks, we obtain larger building blocks (e.g. residual block and MobileNetV2 block proposed in a variety of state-of-the-art models including ResNet, MobileNet V2/V3 (He et al., 2016; Sandler et al., 2018) .

In Section 3.1, We first show larger network building blocks (e.g. MobileNetV2 block) can be represented by an ensembles of atomic blocks.

Based on this view, we propose a fine-grained search space using atomic blocks.

In Section 3.2, we propose a resource-aware atomic block selection method for end-to-end architecture search.

Finally, we propose a dynamic network shrinkage technique in Section 3.3, which greatly reduces the search cost.

Under the typical block-wise NAS paradigm Tan & Le, 2019b) , the search space of each block in a neural network is represented as the Cartesian product C = i=1 P i , where each P i is the set of all choices of the i-th configuration such as kernel size, number of channels and type of operation.

For example, C = {conv, depth-wise conv, dilated conv} × {3, 5} × {24, 32, 64, 128} represents a search space of three types of convolutions by two kernel sizes and four options of channel number.

A block in the resulting model can only pick one convolution type from the three and one output channel number from the four values.

This paradigm greatly limits the search space due to the few choices of each configuration.

Here we present a more fine-grained search space by decomposing the network into smaller and more basic building blocks.

We denote f c ,c (X) as a convolution operator, where X is the input tensor and c, c are the input and output channel numbers respectively.

A wide range of manually-designed and NAS architectures share a structure that joins two convolutions by a channel-wise operation:

where g is a channel-wise operator.

For example, in VGG (Simonyan & Zisserman, 2015) and a Residual Block (He et al., 2016) , f 0 and f 1 are convolutions and g is one of Maxpool, ReLU and BN-ReLU; in a MobileNetV2 block (Sandler et al., 2018) , f 0 and f 1 are point-wise convolutions and g is depth-wise convolution with BN-ReLU in the MobileNetV2 block.

Eq. (1) can be reformulated as follows:

where f

] is the operator of the i-th channel of g, and {f

are obtained by splitting the kernel tensor of f 1 along the the input channel dimension.

Each term in the summation can be seen as a computationally independent block, which is called atomic block.

Fig. (1) demonstrate this reformulation.

By determining whether to keep each atomic block in the final model individually, the search of channel number c is enabled through channel selection, which greatly enlarges the search space.

This formulation also naturally includes the selection of operators.

To gain a better understanding, we first generalize Eq. (2) as:

Note the array indices i are moved to subscripts.

In this formulation, we can use different types of operators for f 0i , f 1i and g i ; in other words, f 0 , f 1 and g can each be a combination of different operators and each atomic block can use different operators such as convolution with different kernel sizes.

Formally, the search space is formulated as a supernet which is built based on the structure in Eq. (1); such structure satisfies Eq. (3) and thus can be represented by atomic blocks; each of f 0 , f 1 and g is a combination of operators.

The new search space includes some state-of-the-art network architectures.

For example, by allowing g to be a combination of convolutions with different kernel sizes, the MixConv block in MixNet (Tan & Le, 2019b ) becomes a special case in our search space.

In addition, our search space facilitates discarding any number of channels in g, resulting in a more fine-grained channel configuration.

In comparison, the channels numbers are determined heuristically in Tan & Le (2019b) .

In this work, we adopt a differentiable neural architecture search paradigm where the model structure is discovered in a full pass of model training.

With the supernet defined above, the final model can be produced by discarding part of the atomic blocks during training.

Following DARTS ), we introduce a scaling factor α to scale the output of each atomic block in the supernet.

Eq. (3) then becomes

Here, each α i is tied with an atomic block comprised of three operators f c ,1

1i ,g i and f

1,c 0i .

The scaling factors are learned jointly with the network weights.

Once the training finishes, the atomic blocks with factors smaller than a threshold are discarded.

We still need to address two issues related to the factor α.

First, where should we put them in the supernet?

The scaling parameters in the BN layers can be directly used as such scaling factors ( Liu et al. (2017) ).

In most cases, g contains at least one BN layer and we use the scaling parameters of the last BN layer in g as α.

If g has no BN layers, which is rare, we can place α anywhere between f 0 and f 1 , as long as we apply regularization terms to the weights of f 0 and f 1 (e.g., weight decays) in order to prevent weights in f 0 and f 1 from getting too large and canceling the effect of α.

The second issue is how to avoid performance deterioration after discarding some of the atomic blocks.

For example, DARTS discards operations with small scale factors after iterative training of model parameters and scale factors.

Since the scale factors of the discarded operations are not small enough, the performance of the network will be affected which needs re-training to adjust the weights again.

In order to maintain the performance of the supernet after dropping some atomics blocks, the scaling factors α of those atomic blocks should be sufficiently small.

Inspired by the channel pruning work in Liu et al. (2017) , we add L1 norm penalty loss on α, which effectively Initialize the supernet and the exponential moving average; while epoch ≤ max epoch do Update network weights and scaling factors α by minimizing the loss function L ; Update theα by Eq. (7); if Total FLOPs of dead blocks ≥ ∆ then Remove dead blocks from the supernet; end Recalculate BN's statistics by forwarding some training examples; Validate the performance of the current supernet; end Algorithm 1: Dynamic network shrinkage pushes many scaling factors to near-zero values.

At the end of learning, atomic blocks with α close to zero are removed from the supernet.

Note that since the BN scales change more dramatically during training due to the regularization term, the running statistics of BNs might be inaccurate and needs to be calculated again using the training set.

With the added regularization term, the training loss is

where λ is the coefficient of L1 penalty term, S is the index set of all atomic blocks, and E is the conventional training loss (e.g. cross-entropy loss combined with the weight decay term).

|α i | is weighted by coefficient c i which is proportional to the computation cost of i-th atomic block, i.e. c i .

By using computation costs aware regularization, we encourage the model to learn network structures that strike a good balance between accuracy and efficiency.

In this paper, we use FLOPs as the criteria of computation cost.

Other metrics such as latency and energy consumption can be used similarly.

As a result, the whole loss function L trades off between accuracy and FLOPs.

Usually, the supernet is much larger than the final search result.

We observe that many atomic blocks become "dead" starting from the early stage of the search, i.e., their scaling factors α are close to zero till the end of the search.

To utilize computational resources more efficiently and speed up the search process, we propose a dynamic network shrinkage algorithm which cuts down the network architecture by removing atomic blocks once they are deemed "dead".

We adopt a conservative strategy to decide whether an atomic block is "dead": for scaling factors α, we maintain its momentumα which is updated aŝ

where α t is the scaling factors at t-th iteration and β is the decay term.

An atomic block is considered "dead" if bothα and α t are smaller than a threshold, which is set to 1e-3 throughout experiments.

Once the total FLOPs of "dead" blocks reach a predefined threshold, we remove those blocks from the supernet.

As discussed above, we recalculate BN's running statistics before deploying the network.

The whole training process is presented in Algorithm 1.

We show the FLOPs of a sample network during the search process in Fig. 2 .

We start from a supernet with 1521M FLOPs and dynamically discard "dead" atomic blocks to reduce search cost.

The overall search and train cost only increases by 17.2% compared to that of training the searched model from scratch.

We first describe the implementation details in Section 4.1 and then compare AtomNAS with previous state-of-the-art methods under various FLOPs constraints in Section 4.2.

Finally, we provide more analysis about AtomNAS in Section 4.3.

The picture on the left of Fig. 3 illustrates a search block in the supernet.

Within this search block, f 0 is a 1 × 1 pointwise convolutions that expands the input channel number from C to 3 × 6C; g is a mix of three depth-wise convolutions with kernel sizes of 3 × 3, 5 × 5 and 7 × 7, and f 1 is another 1×1 pointwise convolutions that projects the channel number to the output channel number.

Similar to Sandler et al. (2018) , if the output dimension stays the same as the input dimension, we use a skip connection to add the input to the output.

In total, there are 3 × 6C atomic blocks in the search block.

The overall architecture of the supernet is shown in the table on the right of Fig. 3 .

The supernet has 21 search blocks.

We use the same training configuration (e.g., RMSProp optimizer, EMA on weights and exponential learning rate decay) as Tan , 2018) .

We find that using this configuration is sufficient for our method to achieve good performance.

Our results are shown in Table 1 and Table 3 .

When training the supernet, we use a total batch size of 2048 on 32 Tesla V100 GPUs and train for 350 epochs.

For our dynamic network shrinkage algorithm, we set the momentum factor β in Eq. (7) to 0.9999.

At the beginning of the training, all of the weights are randomly initialized.

To avoid removing atomic blocks with high penalties (i.e., FLOPs) prematurely, the weight of the penalty term in Eq. (5) is increased from 0 to the target λ by a linear scheduler during the first 25 epochs.

By setting the weight of the L1 penalty term λ to be 1.8×10 −4 , 1.2×10 −4 and 1.0×10 −4 respectively, we obtain networks with three different sizes: AtomNAS-A, AtomNAS-B, and AtomNAS-C. They have the similar FLOPs as previous state-of-the-art networks under 400M: MixNet-S (Tan & Le, 2019b) , MixNet-M (Tan & Le, 2019b) and SinglePath (Stamoulis et al., 2019a).

We apply AtomNAS to search high performance light-weight model on ImageNet 2012 classification task (Deng et al., 2009) .

Table 1 compares our methods with previous state-of-the-art models, either manually designed or searched.

With models directly produced by AtomNAS, our method achieves the new state-of-the-art under all FLOPs constraints.

Especially, AtomNAS-C achieves 75.9% top-1 accuracy with only 360M FLOPs, and surpasses all other models, including models like PDARTS and DenseNAS which have much higher FLOPs.

Techniques like Swish activation function (Ramachandran et al., 2018) and Squeeze-and-Excitation (SE) module (Hu et al., 2018) † means methods use extra techniques like Swish activation and Squeeze-and-Excitation module.

fair comparison with methods that use these techniques, we directly modify the searched network by replacing all ReLU activation with Swish and add SE module with ratio 0.5 to every block and then retrain the network from scratch.

Note that unlike other methods, we do not search the configuration of Swish and SE, and therefore the performance might not be optimal.

Extra data augmentations such as MixUp and AutoAugment are still not used.

We train the models from scratch with a total batch size of 4096 on 32 Tesla V100 GPUs for 250 epochs.

Simply adding these techniques improves the results further.

AtomNAS-A+ achieves 76.3% top-1 accuracy with 260M FLOPs, which outperforms many heavier models including MnasNet-A2.

It performs as well as Efficient-B0 (Tan & Le, 2019a ) by using 130M less FLOPs and without extra data augmentations.

It also outperforms the previous state-of-the-art MixNet-S by 0.5%.

In addition, AtomNAS-C+ improves the top-1 accuracy on ImageNet to 77.6%, surpassing previous state-of-the-art MixNet-M by 0.6% and becomes the overall best performing model under 400M FLOPs.

Fig. 4 visualizes the top-1 accuracy on ImageNet for different models.

It's clear that our fine-grained search space and the end-to-end resource-aware search method boost the performance significantly.

† denotes methods using extra network modules such as Swish activation and Squeeze-and-Excitation module.

‡ denotes using extra data augmentation such as MixUp and AutoAugment.

* denotes models searched and trained simultaneously.

Parameters FLOPs Top-1(%) Top-5(%)

MobileNetV1 (Howard et al., 2017) 4.2M 575M 70.6 89.5 MobileNetV2 (Sandler et al., 2018) 3.4M 300M 72.0 91.0 MobileNetV2 (our impl.)

3.4M 301M 73.6 91.5 MobileNetV2 (1.4) 6.9M 585M 74.7 92.5 ShuffleNetV2 (Ma et al., 2018) 3.5M 299M 72.6 -ShuffleNetV2 2× 7.4M 591M 74.9 -FBNet-A (Wu et al., 2019) 4.3M 249M 73.0 -FBNet-C 5.5M 375M 74.9 -Proxyless (mobile) (Cai et al., 2019) 4.1M 320M 74.6 92.2 SinglePath (Stamoulis et al., 2019a) 4.4M 334M 75.0 92.2 NASNet-A (Zoph & Le, 2017) 5.3M 564M 74.0 91.6 DARTS (second order) 4.9M 595M 73.1 -PDARTS (cifar 10) (Chen et al., 2019b) 4.9M 557M 75.6 92.6 DenseNAS-A (Fang et al., 2019) 7.9M 501M 75.9 92.6 FairNAS-A (Chu et al., 2019b)

4 3x3  32x112x112  16x112x112  24x56x56  24x56x56  24x56x56  24x56x56  40x28x28  40x28x28  40x28x28  40x28x28  80x14x14  80x14x14  80x14x14  80x14x14  96x14x14  96x14x14  96x14x14  96x14x14  192x7x7  192x7x7  192x7x7  192x7x7 Pooling FC 320x7x7 3 5 7

Figure 5: The architecture of AtomNAS-C. Blue, orange, cyan blocks denote atomic blocks with kernel size 3, 5 and 7 respectively; the heights of these blocks are proportional to their expand ratios.

We plot the structure of the searched architecture AtomNAS-C in Fig. 5 , from which we see more flexibility of channel number selection, not only among different operators within each block, but also across the network.

In Fig. 6a , we visualize the ratio between atomic blocks with different kernel sizes in all 21 search blocks.

First, we notice that all search blocks have convolutions of all three kernel sizes, showing that AtomNAS learns the importance of using multiple kernel sizes in network architecture.

Another observation is that AtomNAS tends to keep more atomic blocks at the later stage of the network.

This is because in earlier stage, convolutions of the same kernel size costs more FLOPs; AtomNAS is aware of this (thanks to its resource-aware regularization) and try to keep as less as possible computationally costly atomic blocks.

To demonstrate the effectiveness of the resource-aware regularization in Section 3.2, we compare it with a baseline without FLOPs-related coefficients c i , which is widely used in network pruning (Liu et al., 2017; He et al., 2017b) .

Table 2 shows the results.

First, by using the same L1 penalty coefficient λ = 1.0 × 10 −4 , the baseline achieves a network with similar performance but using much more FLOPs; then by increasing λ to 1.5 × 10 −4 , the baseline obtain a network which has similar FLOPs but inferior performance (i.e., about 1.0% lower).

In Fig. 6b we visualized the ratio of different types of atomic blocks of the baseline network obtained by λ = 1.5×10 −4 .

The baseline network keeps more atomic blocks in the earlier blocks, which have higher computation cost due to higher input resolution.

On the contrary, AtomNAS is aware of the resource constraint, thus keeping more atomic blocks in the later blocks and achieving much better performance.

As the BN's running statistics might be inaccurate as explained in Section 3.2 and Section 3.3, we re-calculate the running statistics of BN before inference, by forwarding 131k randomly sampled training images through the network.

Table 3 shows the impact of the BN recalibration.

The top-1 accuracies of AtomNAS-A, AtomNAS-B, and AtomNAS-C on ImageNet improve by 1.4%, 1.7%, and 1.2% respectively, which clearly shows the benefit of BN recalibration.

Our dynamic network shrinkage algorithm speedups the search and train process significantly.

For AtomNAS-C, the total time for search-and-training is 25.5 hours.

For reference, training the final architecture from scratch takes 22 hours.

Note that as the supernet shrinks, both the GPU memory consumption and forward-backward time are significantly reduced.

Thus it's possible to dynamically change the batch size once having sufficient GPU memory, which would further speed up the whole procedure.

In this paper, we revisit the common structure, i.e., two convolutions joined by a channel-wise operation, and reformulate it as an ensemble of atomic blocks.

This perspective enables a much larger and more fine-grained search space.

For efficiently exploring the huge fine-grained search space, we propose an end-to-end algorithm named AtomNAS, which conducts architecture search and network training jointly.

The searched networks achieve significantly better accuracy than previous state-of-the-art methods while using small extra cost.

Table 4 : Comparision with baseline backbones on COCO object detection and instance segmentation.

Cls denotes the ImageNet top-1 accuracy; detect-mAP and seg-mAP denotes mean average precision for detection and instance segmentation on COCO dataset.

The detection results of baseline models are from Stamoulis et al. (2019b) .

SinglePath+ (Stamoulis et al., 2019b) In this section, we assess the performance of AtomNAS models as feature extractors for object detection and instance segmentation on COCO dataset (Lin et al., 2014) .

We first pretrain AtomNAS models (without Swish activation function (Ramachandran et al., 2018) and Squeeze-and-Excitation (SE) module (Hu et al., 2018) ) on ImageNet, use them as drop-in replacements for the backbone in the Mask-RCNN model (He et al., 2017a) by building the detection head on top of the last feature map, and finetune the model on COCO dataset.

We use the open-source code MMDetection (Chen et al., 2019a) .

All the models are trained on COCO train2017 with batch size 16 and evaluated on COCO val2017.

Following the schedule used in the open-source implementation of TPU-trained Mask-RCNN , the learning rate starts at 0.02 and decreases by a scale of 10 at 15-th and 20th epoch respectively.

The models are trained for 23 epochs in total.

Table 4 compares the results with other baseline backbone models.

The detection results of baseline models are from Stamoulis et al. (2019b) .

We can see that all three AtomNAS models outperform the baselines on object detection task.

The results demonstrate that our models have better transferability than the baselines, which may due to mixed operations, a.k.a multi-scale are here, are more important to object detection and instance segmentation.

https://github.com/tensorflow/tpu/tree/master/models/official/mask_ rcnn

<|TLDR|>

@highlight

A new state-of-the-art on Imagenet for mobile setting
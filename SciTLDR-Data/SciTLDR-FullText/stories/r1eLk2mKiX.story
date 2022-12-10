Network pruning is widely used for reducing the heavy computational cost of deep models.

A typical pruning algorithm is a three-stage pipeline, i.e., training (a large model), pruning and fine-tuning.

In this work, we make a rather surprising observation: fine-tuning a pruned model only gives comparable or even worse performance than training that model with randomly initialized weights.

Our results have several implications: 1) training a large, over-parameterized model is not necessary to obtain an efficient final model, 2) learned "important" weights of the large model are not necessarily useful for the small pruned model, 3) the pruned architecture itself, rather than a set of inherited weights, is what leads to the efficiency benefit in the final model, which suggests that some pruning algorithms could be seen as performing network architecture search.

Network pruning is a commonly used approach for obtaining an efficient neural network.

A typical procedure of network pruning consists of three stages: 1) train a large, over-parameterized model, 2) prune the unimportant weights according to a certain criterion, and 3) fine-tune the pruned model to regain accuracy.

Generally, there are two common beliefs behind this pruning procedure.

First, it is believed that training a large network first is important [1] and the three-stage pipeline can outperform directly training the small model from scratch.

Second, both the architectures and the weights of the pruned model are believed to be essential for the final efficient model, which is why most existing pruning methods choose to fine-tune the pruned model instead of training it from scratch.

Also because of this, how to select the set of important weights is a very active research topic [1, 2, 3, 4].Predefined: prune x% channels in each layer Automatic: prune a%, b%, c%, d% channels in each layer A 4-layer model Figure 1 : Difference between predefined and non-predefined (automatically discovered) target architectures.

The sparsity x is user-specified, while a, b, c, d are determined by the pruning algorithm.

In this work, we show that both of the beliefs mentioned above are not necessarily true.

We make a surprising observation that directly training the target pruned model from random initialization can achieve the same or better performance as the model obtained from the three-stage pipeline.

This means that, for pruning methods with a predefined target architecture (Figure 1 ), starting with a large model is not necessary and one could instead directly train the target model from scratch.

For pruning algorithms with automatically discovered target architectures, what brings the efficiency benefit is the obtained architecture, instead of the inherited weights.

Our results advocate a rethinking of existing network pruning algorithms: the preserved "important" weights from the large model are not necessary for obtaining an efficient final model; instead, the value of automatic network pruning methods may lie in identifying efficient architectures and performing implicit architecture search.

Target Pruned Architectures.

We divide pruning methods by whether the target pruned architecture is determined by either human or the pruning algorithm (see Figure 1 ).

An example of designing predefined target architecture is specifying how many channels to prune in each layer.

In contrast, when the target architecture is automatically discovered, the pruning algorithm determines how many channels to prune in each layer, by comparing the importance of channels across layers.

Training from scratch.

Because the pruned model requires less computation, it may be unfair to train the pruned model for the same number of epochs as the large model.

In our experiments, we use Scratch-E to denote training the small pruned models for the same epochs, and Scratch-B to denote training for the same amount of computation budget (measured by FLOPs).

We use standard training hyper-parameters and data-augmentation schemes.

The optimization method used is SGD with Nesterov momentum.

In this section we present our experimental results comparing training pruned models from scratch with fine-tuning.

For method with predefined architectures, we evaluate L1-norm based channel pruning method [3] .

For method with automatically discovered target architectures, we use Network Slimming [5] .

The models, datasets and the number of epochs for fine-tuning are the same as those in the original paper.

More results for other pruning methods and transfer learning can be found in Appendix B.

is one of the earliest work on channel pruning for convolutional networks.

In each layer, a certain percentage of channels with smaller L 1 -norm of its filter weights will be pruned.

Table 1 shows our results.

The Pruned Model column shows the list of predefined target models.

We observe that in each row, scratch-trained models achieve at least the same level of accuracy as fine-tuned models, with Scratch-B slightly higher than Scratch-E in most cases.

On ImageNet, both Scratch-B models are better than the fine-tuned ones.

Network Slimming [5] imposes L 1 -sparsity regularization on channel-wise scaling factors from Batch Normalization layers [6] during training, and prunes channels with lower scaling factors afterward.

Since the channel scaling factors are compared across layers, this method produces automatically discovered target architectures.

As shown in TAB2 , for all networks, the small models trained from scratch can reach the same accuracy as the fine-tuned models, where Scratch-B consistently outperforms the fine-tuned models.

Morever, when we extend the standard training of large model, the above observation still holds.

In this section, we demonstrate that the value of automatic network pruning methods actually lies in searching efficient architectures.

We use Network Slimming [5] as an example automatic method.

Parameter Efficiency of Pruned Architectures.

In FIG0 that uniformly prunes the same percentage of channels in all layer.

All architectures are trained from random initialization for the same number of epochs without sparsity regularization.

We see that the architectures obtained by Network Slimming are more parameter efficient, as they could achieve the same level of accuracy using 5× fewer parameters than uniformly pruning architectures.

Generalizable Design Principles from Pruned Architectures.

Given that the automatically discovered architectures tend to be parameter efficient, here we show a generalizable principle on designing better architectures.

We show two design strategies: 1) "Guided Pruning": use the average number of channels in each layer stage (layers with the same feature map size) from pruned architectures to construct a new set of architectures.

2) "Transferred Guided Pruning": use these patterns from a different architecture on a different dataset.

FIG0 (right) shows our results.

Here for "Transferred Guided Pruning", we use the patterns obtained by a pruned VGG-16 on CIFAR-10, to design the network for VGG-19 on CIFAR-100.

We observe that both "Guided Pruning" (green) and "Transferred Guided Pruning" (brown) can both perform on par architectures directly pruned on the VGG-19 and CIFAR-100 (blue), and are significantly better than uniform pruning (red).

This means we could distill generalizable design principles from pruned architectures.

In practice, for methods with predefined target architectures, training from scratch is more computationally efficient and saves us from implementing the pruning procedure and tuning the additional hyper-parameters.

Still, pruning methods are useful when a pretrained large model is already given, in this case fine-tuning is much faster.

Also, obtaining multiple models of different sizes can be done quickly by pruning from a large model by different ratios.

In summary, our experiments have shown that training the small pruned model from scratch can almost always achieve comparable or higher level of accuracy than the model obtained from the typical "training, pruning and fine-tuning" procedure.

This changed our previous belief that over-parameterization is necessary for obtaining a final efficient model, and the understanding on effectiveness of inheriting weights that are considered important by the pruning criteria.

We further demonstrated the value of automatic pruning algorithms could be regarded as finding efficient architectures and providing architecture design guidelines.

[1] Jian-Hao Luo, Jianxin Wu, and Weiyao Lin.

Thinet: A filter level pruning method for deep neural network compression.

In ICCV, 2017.[2]

Song Han, Jeff Pool, John Tran, and William Dally.

Learning both weights and connections for efficient neural network.

In NIPS, 2015.[3] Hao Li, Asim Kadav, Igor Durdanovic, Hanan Samet, and Hans Peter Graf.

Pruning filters for efficient convnets.

In ICLR, 2017.[4]

Yihui He, Xiangyu Zhang, and Jian Sun.

Channel pruning for accelerating very deep neural networks.

In ICCV, 2017.[5]

Zhuang Liu, Jianguo Li, Zhiqiang Shen, Gao Huang, Shoumeng Yan, and Changshui Zhang.

Learning efficient convolutional networks through network slimming.

In ICCV, 2017.[6] Sergey Ioffe and Christian Szegedy.

Batch normalization: Accelerating deep network training by reducing internal covariate shift.

arXiv preprint arXiv:1502.03167, 2015.[7] Alex Krizhevsky.

Learning multiple layers of features from tiny images.

Technical report, 2009.[8]

Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei.

Imagenet: A large-scale hierarchical image database.

In CVPR, 2009.[9] Karen Simonyan and Andrew Zisserman.

Very deep convolutional networks for large-scale image recognition.

ICLR, 2015.[10] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun.

Deep residual learning for image recognition.

In CVPR, 2016.[11] Gao Huang, Zhuang Liu, Laurens van der Maaten, and Kilian Q Weinberger.

Densely connected convolutional networks.

In CVPR, 2017.[12] Zehao Huang and Naiyan Wang.

Data-driven sparse structure selection for deep neural networks.

ECCV, 2018.[ [16] Saining Xie, Ross Girshick, Piotr Dollár, Zhuowen Tu, and Kaiming He.

Aggregated residual transformations for deep neural networks.

In CVPR, 2017.[17] Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun.

Faster r-cnn: Towards real-time object detection with region proposal networks.

In NIPS, 2015.[18] Zhiqiang Shen, Zhuang Liu, Jianguo Li, Yu-Gang Jiang, Yurong Chen, and Xiangyang Xue.

Dsod: Learning deeply supervised object detectors from scratch.

In ICCV, 2017.[19] Jianwei Yang, Jiasen Lu, Dhruv Batra, and Devi Parikh.

A faster pytorch implementation of faster r-cnn.

https://github.com/jwyang/faster-rcnn.pytorch, 2017.[20] Barret Zoph and Quoc V Le.

Neural architecture search with reinforcement learning.

ICLR, 2017.

Here we provide additional details on our experiment setups.

Implementation.

In order to keep our setup as close to the original paper as possible, we use the following protocols: 1) If a previous pruning method's training setup is publicly available, e.g.[5] and [12], we adopt the original implementation; 2) Otherwise, for simpler pruning methods, e.g., [2, 3], we re-implement the three-stage pruning procedure and achieve similar results to the original paper; 3) For the remaining two methods [1, 4], the pruned models are publicly available but without the training setup, thus we choose to re-train both large and small target models from scratch.

Interestingly, the accuracy of our re-trained large model is higher than what is reported in the original paper 3 .

In this case, to accommodate the effects of different frameworks and training setups, we report the relative accuracy drop from the unpruned large model.

The results of two pruning methods are already shown in the extended abstract, here in appendix we provide results of the remaining four pruning methods.

We also include an experiment on transfer learning from image classification to object detection.

ThiNet [1] greedily prunes the channel that has the smallest effect on the next layer's activation values.

As shown in TAB7 , for VGG-16 and ResNet-50, both Scratch-E and Scratch-B can almost always achieve better performance than the fine-tuned model, often by a significant margin.

The only exception is Scratch-E for VGG-Tiny, where the model is pruned very aggressively from VGG-16 (FLOPs reduced by 15×), and as a result, drastically reducing the training budget for Scratch-E. The training budget of Scratch-B for this model is also 7 times smaller than the original large model, yet it can achieve the same level of accuracy as the fine-tuned model.

TAB8 .

Again, in terms of relative accuracy drop from the large models, scratch-trained models are better than the fine-tuned models.

In summary, for pruning methods with predefined target architectures, training the small models for the same number of epochs as the large model (Scratch-E), is often enough to achieve the same accuracy as models output by the three-stage pipeline.

Combined with the fact that the target architecture is predefined, in practice one would prefer to train the small model from scratch directly.

Moreover, when provided with the same amount of computation budget (measured by FLOPs) as the large model, scratch-trained models can even lead to better performance than the fine-tuned models.

Sparse Structure Selection [12] also imposes sparsity regularization on the scaling factors during training to prune structures, and can be seen as a generalization of Network Slimming.

Other than channels, pruning can be on residual blocks in ResNet or groups in ResNeXt [16] .

We examine residual blocks pruning, where ResNet-50 are pruned to be ResNet-41, ResNet-32 and ResNet-26.

TAB13 shows our results.

On average Scratch-E outperforms pruned models, and for all models Scratch-B is better than both.

Table 5 : Results (accuracy) for residual block pruning using Sparse Structure Selection [12] .

In the original paper no fine-tuning is required so there is a "Pruned" column instead of "Fine-tuned" as before.

Non-structured Weight Pruning [2] prunes individual weights that have small magnitudes.

This pruning granularity leaves the weight matrices sparse, hence it is commonly referred to as nonstructured weight pruning.

Because all the network architectures we evaluated are fully-convolutional (except for the last fully-connected layer), for simplicity, we only prune weights in convolution layers here.

Before training the pruned sparse model from scratch, we re-scale the standard deviation of the Gaussian distribution for weight initialization, based on how many non-zero weights remain in this layer.

This is to keep a constant scale of backward gradient signal [15] .

As shown in TAB11 , on CIFAR datasets, Scratch-E sometimes falls short of the fine-tuned results, but Scratch-B is able to perform at least on par with the latter.

On ImageNet, we note that sometimes even Scratch-B is slightly worse than fine-tuned result.

This is the only case where Scratch-B does not achieve comparable accuracy in our attempts.

We hypothesize this could be due to the task complexity of ImageNet and the fine pruning granularity.

Effects of Sparsity Regularization.

Some methods [5, 12] use sparsity regularization during the training of large, over-parameterized models, to smooth the following pruning process, and in finetuning, no such sparsity is imposed.

In all of our experiments presented above, no sparsity is used for training the pruned models from scratch.

Here we analyze the effects of using sparsity regularization (or not) for Network Slimming [5] .

TAB13 shows the results when all training procedures are with sparsity regularization.

We can see that Scratch-B are still able to be on par with the fine-tuned models.

TAB14 shows the results when we do not use sparsity regularization in any training procedures, including large model training.

We can see that when no sparsity is induced, Scratch-B can also consistently achieve comparable results with the fine-tuned models.

We have shown that the small pruned model can be trained from scratch to match the accuracy of the fine-tuned model in classification tasks.

To see whether this phenomenon would also hold for transfer learning to other vision tasks, we evaluate the L 1 -norm based pruning method [3] on the PASCAL VOC object detection task, using Faster-RCNN [17] .Object detection frameworks usually require transferring model weights pre-trained on ImageNet classification, and one can perform pruning either before or after the weight transfer.

More specifically, the former could be described as "train on classification, prune on classification, fine-tune on classification, transfer to detection", while the latter is "train on classification, transfer to detection, prune on detection, fine-tune on detection".

We call these two approaches Prune-C (classification) and Prune-D (detection) respectively, and report the results in TAB15 shows our result, and we can see that the model trained from scratch can surpass the performance of fine-tuned models under the transfer setting.

Another interesting observation from TAB15 is that Prune-C is able to outperform Prune-D, which is surprising since if our goal task is detection, directly pruning away weights that are considered unimportant for detection should presumably be better than pruning on the pre-trained classification models.

We hypothesize that this might be because pruning early in the classification stage makes the final model less prone to being trapped in a bad local minimum caused by inheriting weights from the large model.

This is in line with our observation that Scratch-E/B, which trains the small models from scratch starting even earlier at the classification stage, can achieve further performance improvement.

Here we provide results that complements Section 4 for non-structured weight pruning [2] .

The networks and datasets used are the same as those used in Section 4.

We also discuss the relation with conventional architecture search methods at the end.

Figure 3(left) shows our results on parameter efficiency of architectures obtained by non-structured pruning.

Here "Uniform Sparsifying" means uniformly sparsifying individual weights in the network at a fixed probability.

It can be seen that pruned architectures are more parameter-efficient than uniform sparsified architectures.

Figure 3(right) shows our results for architectures obtained by other different design strategies.

For "Guided Sparsifying" and "Transferred Guided Sparsifying", we use the average sparsity patterns of 3 × 3 kernel in each layer stage to design new structures.

Similar to Network Slimming, both "Guided Sparsifying" (green) and "Transferred Guided Sparsifying" (brown) are significantly more parameter-efficient than uniform sparsifying (red).Comparison with Traditional Architecture Search Methods.

Conventional techniques for network architecture search include reinforcement learning [20, BID0 and evolutionary algorithms BID1 BID2 .In each iteration, a randomly initialized network is trained and evaluated to guide the search, and the search process usually requires thousands of iterations to find the goal architecture.

In contrast, using network pruning as architecture search only requires a one-pass training, however the search space is restricted to the set of all "sub-networks" inside a large network, whereas traditional methods can search for more variations, e.g., activation functions or different layer orders.

Recently, BID3 uses a similar pruning technique to Network Slimming [5] to automate the design of network architectures; BID4 prune channels using reinforcement learning and automatically compresses the architecture.

On the other hand, in the network architecture search literature, sharing/inheriting trained parameters BID5 BID6 has become a popular approach to accelerate the convergence and reduce the training budget during searching, though it would be interesting to investigate whether training from scratch would sometimes yield better results as we observed in network pruning.

We can see that these two areas, namely network pruning and architecture search, share many common traits and start to borrow wisdom from each other.

We show the weight distribution of unpruned large models, fine-tuned pruned models and scratchtrained pruned models, for two pruning methods: Network Slimming [5] and non-structured weight level pruning [2] .

We choose DenseNet-40 and CIFAR-10 for visualization and compare the weight distribution of unpruned models, fine-tuned models and scratch-trained models.

For Network Slimming, the prune ratio is chosen to be 60%.

For non-structured weight level pruning, the prune ratio is chosen to be 80%.

FIG5 shows our result.

We can see that the weight distribution of fine-tuned models and scratch-trained pruned models are different from the unpruned large models -the weights that are close to zero are much fewer.

This seems to imply that there are less redundant structures in the found pruned architecture, and support the effect of architecture search for automatic pruning methods.

@highlight

In network pruning, fine-tuning a pruned model only gives comparable or worse performance than training it from scratch. This advocate a rethinking of existing pruning algorithms.
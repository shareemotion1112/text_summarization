Parameter pruning is a promising approach for CNN compression and acceleration by eliminating redundant model parameters with tolerable performance loss.

Despite its effectiveness, existing regularization-based parameter pruning methods usually drive weights towards zero with large and constant regularization factors, which neglects the fact that the expressiveness of CNNs is fragile and needs a more gentle way of regularization for the networks to adapt during pruning.

To solve this problem, we propose a new regularization-based pruning method (named IncReg) to incrementally assign different regularization factors to different weight groups based on their relative importance, whose effectiveness is proved on popular CNNs compared with state-of-the-art methods.

Recently, deep Convolutional Neural Networks (CNNs) have made a remarkable success in computer vision tasks by leveraging large-scale networks learning from big amount of data.

However, CNNs usually lead to massive computation and storage consumption, thus hindering their deployment on mobile and embedded devices.

To solve this problem, many research works focus on compressing the scale of CNNs.

Parameter pruning is a promising approach for CNN compression and acceleration, which aims at eliminating redundant model parameters at tolerable performance loss.

To avoid hardware-unfriendly irregular sparsity, structured pruning is proposed for CNN acceleration BID0 BID22 .

In the im2col implementation BID1 BID3 of convolution, weight tensors are expanded into matrices, so there are generally two kinds of structured sparsity, i.e. row sparsity (or filter-wise sparsity) and column sparsity (or shape-wise sparsity) BID24 BID23 .There are mainly two categories of structured pruning.

One is importance-based methods, which prune weights in groups based on some established importance criteria BID17 BID19 BID23 .

The other is regularization-based methods, which add group regularization terms to learn structured sparsity BID24 BID16 BID8 .

Existing group regularization approaches mainly focus on the regularization form (e.g. Group LASSO BID26 ) to learn structured sparsity, while ignoring the influence of regularization factor.

In particular, they tend to use a large and constant regularization factor for all weight groups in the network BID24 BID16 , which has two problems.

Firstly, this 'one-size-fit-all' regularization scheme has a hidden assumption that all weights in different groups are equally important, which however does not hold true, since weights with larger magnitude tend to be more important than those with smaller magnitude.

Secondly, few works have noticed that the expressiveness of CNNs is so fragile BID25 during pruning that it cannot withstand a large penalty term from beginning, especially for large pruning ratios and compact networks (like ResNet BID7 ).

AFP BID6 was proposed to solve the first problem, while ignored the second one.

In this paper, we propose a new regularization-based method named IncReg to incrementally learn structured sparsity.

Given a conv kernel, modeled by a 4-D tensor DISPLAYFORM0 , where DISPLAYFORM1 and W (l) are the dimension of the lth (1 ??? l ??? L) weight tensor along the axis of filter, channel, height and width, respectively, our proposed objective function for regularization can be formulated DISPLAYFORM2 ), where W denotes the collection of all weights in the CNN; L(W) is the loss function for prediction; R(W) is non-structured regularization on every weight, i.e. weight decay in this paper; R(W (l) g ) is the structured sparsity regularization term on group g of layer l and G (l) is the number of weight groups in layer l. In BID16 BID24 , the authors used the same ?? g for all groups and adopted Group LASSO BID26 for R(W DISPLAYFORM3 ).

In this work, since we emphasize the key problem of group regularization lies in the regularization factor rather than the regularization form, we use the most common regularization form weight decay, for R(W (l) g ), but we vary the regularization factors ?? g for different weight groups and at different iterations.

Our method prunes all the conv layers simultaneously and independently.

For simplicity, we omit the layer notation l for following description.

All the ?? g 's are initialized to zero.

At each iteration, ?? g is increased by ?? new g = max(?? g + ????? g , 0).

Like AFP BID6 , we agree that unimportant weights should be punished more, so we propose a decreasing piece-wise linear punishment function (Eqn.1, FIG0 ) to determine ????? g .

Note that the regularization increment is negative (i.e. reward actually) when ranking is above the threshold ranking RG, since above-the-threshold means these weights are expected to stay in the end.

Regularization on these important weights is not only unnecessary but also very harmful via our experiment confirmation.

DISPLAYFORM4 where R is the pre-assigned pruning ratio for a layer, G is the number of weight groups, A is a hyper-parameter in our method to describe the maximum penalty increment (set to half of the original weight decay in default), r is the ranking obtained by sorting in ascending order based on a proposed importance criterion, which is essentially an averaged ranking over time, defined as 1 N N n=1 r n , where r n is the ranking by L 1 -norm at nth iteration, N is the number of passed iterations.

This averaging is adopted as smoothing for a more stable pruning process.

As training proceeds, the regularization factors of different weight groups increase gradually, which will push the weights towards zero little by little.

When the magnitude of a weight group is lower than some threshold (10 ???6 ), the weights are permanently removed from the network, thus leading to increased structured sparsity.

When the sparsity of a layer reaches its pre-assigned pruning ratio R, that layer automatically stops structured regularization.

Finally, when all conv layers reach their pre-assigned pruning ratios, pruning is over, followed by a retraining process to regain accuracy.

We firstly compare our proposed IncReg with other two group regularization methods, i.e. SSL BID24 and AFP BID6 , with ConvNet BID14 on CIFAR-10 BID13 , where both row sparsity and column sparsity are explored.

Caffe BID12 is used for all of our experiments.

Experimental results are shown in Tab.1.

We can see that IncReg consistently achieves higher speedups and accuracies than the other two constant regularization schemes.

Notably, even though AFP achieves similar performance as our method under relatively small speedup (about 4.5??), when the speedup ratio is large (about 8?? ??? 10??), our method outperforms AFP by a large margin.

We argue that this is because the incremental way of regularization gives the network more time to adapt during pruning, which is especially important in face of large pruning ratios.

Table 1 : Comparison of our method with SSL BID24 and AFP BID6 with ConvNet on CIFAR-10.

We further evaluate our method with VGG-16 BID21 (13 conv layers) and ResNet-50 BID7 (53 conv layers) on ImageNet BID5 .We download the open-sourced caffemodel as our pre-trained model, whose single-view top-5 accuracy on ImageNet validation dataset is 89.6% (VGG-16) and 91.2% (ResNet-50).

For VGG-16, following SPP BID23 , the proportion of remaining ratios of low layers (conv1_x to conv3_x), middle layers (conv4_x) and high layers (conv5_x) are set to 1 : 1.

We propose a new structured pruning method based on an incremental way of regularization, which helps CNNs to transfer their expressiveness to the rest parts during pruning by increasing the regularization factors of unimportant weight groups little by little.

Our method is proved to be comparably effective on popular CNNs compared with state-of-the-art methods, especially in face of large pruning ratios and compact networks.

<|TLDR|>

@highlight

 we propose a new regularization-based pruning method (named IncReg) to incrementally assign different regularization factors to different weight groups based on their relative importance.

@highlight

This paper proposes a regularization-based pruning method to incrementally assign different regularization factors to different weight groups based on their relative importance.
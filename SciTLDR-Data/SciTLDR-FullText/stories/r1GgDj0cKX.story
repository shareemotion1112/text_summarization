This paper proposes a Pruning in Training (PiT) framework of learning to reduce the parameter size of networks.

Different from existing works, our PiT framework employs the sparse penalties to train networks and thus help rank the importance of weights and filters.

Our PiT algorithms can directly prune the network without any fine-tuning.

The pruned networks can still achieve comparable performance to the original networks.

In particular, we introduce the (Group) Lasso-type Penalty (L-P /GL-P), and (Group) Split LBI Penalty (S-P / GS-P) to regularize the networks, and a pruning strategy proposed  is used in help prune the network.

We conduct the extensive experiments on MNIST, Cifar-10, and miniImageNet.

The results validate the efficacy of our proposed methods.

Remarkably, on MNIST dataset, our PiT framework can save 17.5% parameter size of LeNet-5, which achieves the 98.47% recognition accuracy.

The expressive power of Deep Convolutional Neural Networks (DNNs) comes from the millions of parameters, which are optimized by various algorithms such as Stochastic Gradient Descent (SGD), and Adam BID18 .

However, one has to strike a trade-off between the representation capability and computational cost, caused by the plenty of parameters in the real world applications, e.g., robotics, self-driving cars, and augmented reality.

Pruning significant number of parameters would be essential to reduce the computational complexity and thus facilitate a timely and efficient fashion on a resource-limited platform, e.g. devices of Internet of Things (IoT).

In addition, it has long been conjectured that the state-of-the-art DNNs may be too complicated for most specific tasks; and we may have the free lunch of "reducing 2× connections without losing accuracy and without retraining" BID7 .To compress DNNs, recent efforts had been made on learning the DNNs of small size.

They either reduce the number and size of weights of parameters of original networks, and fine-tune the pruned networks BID0 ; BID32 , or distill the knowledge of large model , or directly learning the compact and lightweight small DNNs, such as ShuffleNet BID24 , MobileNet Howard et al. (2017) , and SqueezeNet BID13 .

Note that, (1) to efficiently learn the compressed DNNs, previous works had to introduce additional computational cost in fine-tuning, or training the updated networks; (2) it is not practical nor desirable to learn the tailored, or bespoke networks for any applications, beyond computer vision tasks.

To this end, the center idea of this paper is to propose a Pruning in Training (PiT) framework that enables pruning networks in the training process.

Particularly, the sparsity regularizers, including lasso-type, and split LBI penalties are applied to train the networks.

Such regularizers not only encourage the sparsity of DNNs, i.e., fewer (sparse) connections with non-zero values, but also can accelerate the speed of DNNs convergence.

Furthermore, in the learning process, we can iteratively compute the regularization path of layer-wise parameters of DNNs.

The parameters can be ranked by the regularization path in a descending order, as BID3 .

The parameters in the high rank are in the high priority of not being pruned.

More importantly, our PiT can learn the sparse structures of DNNs, and utilize the functionality of filters and connection weights (in fully connected layers).

In the optimal cases, the weights (or filters) of each layer should be learned fully orthogonal to each other and thus formulate an orthogonal basis.

The orthogonal constraint may be only enforced as the initialization (e.g., SVD Jia (2017) and BID26 ), or via the other regularization tricks, such as dropout preventing co-adaption BID27 , or batch normalization reducing the internal covariate shift of hidden layers BID14 .

Therefore, our PiT can help uncover redundant information in a network by compressing less important filters and weights, and facilitate pruning out more interpretable networks.

The deeper and wider deep CNN architectures can enable the superior performance on various tasks, and yet cause the prohibitively expensive computation cost.

To efficiently train the networks, the regularization is usually applied to the weight parameters (Sec. 2.1).

It is also essential to prune networks to reduce the size of networks (Sec. 2.2)

Due to large number of parameters, the deep networks require large amount of memory and computational resources, and are inclined to overfit the training data.

To alleviate this problem, it is essential to regularize the networks in training stage; such as dropout BID27 preventing the co-adaptation, and adding L 2 or L 1 regularization to weights.

In particular, the L 1 regularization enforces the sparsity on the weights and results in a compact, memory-efficient network with slightly sacrificing the prediction performance BID2 .

Further, group sparsity regularization BID34 can also been applied to deep networks with desirable properties.

Alvarez et al. Alvarez & Salzmann (2016) utilized a group sparsity regularizer to automatically decide the optimal number of neuron groups.

The structured sparsity BID30 ; BID33 has also been investigated to exert good data locality and group sparsity.

Different from these works, the (Group) Split LBI penalty is for the first time, introduced to regularize the networks.

This regularization term can not only enforce the structured sparsity, but also can efficiently compute the solution paths of each variable.

Compressing the networks involves the pruning and compressing the weights and filters of DNNs.

The common strategies include (1) matrix decomposition methods BID15 ; BID29 by decomposing the weight matrix of DNNs as a low-rank product of two smaller matrices; (2) low-precision weights methods BID11 ; BID40 by learning to store low-precision weights of DNNs; and (3) pruning methods BID7 ; Li et al. (2017) directly removing weights of connections, or neurons.

Our framework is one of pruning methods.

Previous pruning works, iteratively prune the weights or neurons, and fine-tune the network BID7 ; BID5 .

Remarkably, network regularization is of significant important in pruning methods.

The sparse properties of features maps and/or weights of DNNs exerted by network regularization, are utilized in Wen et al. (2016b) ; BID19 .

BID23 adopt the statistics information from next layer to guide and save the importance of filters of the current layer.

BID25 employed Taylor expansion to approximate the change of cost function which can be further utilized as the criterion in pruning network parameters.

A LASSO-based channel selection strategy is investigated in BID9 .

Abbasi-Asl et al. BID0 defined a filter importance index of greedy pruning the network.

Comparing with all the methods, our framework is different in two points: (1) Criterion of importance of weights and filters.

We rank the importance of weights and filters by their solution paths computed by sparse regularizers, rather than designing the elaborated metrics as previous works BID0 ; BID32 .

Specifically, our algorithm is a process of solving the discrete partial differential equations; and our framework can result in the solution paths of optimizing the weights and filters, whose importance are ranked, according to the selected order in the path, as Fu et al. (2016b) .

(2) Pruning in training: once DNNs are trained, we simply prune out less important weights/filter by a threshold.

In this section, the Residual Network (ResNet) structureHe et al. FORMULA1 is employed to elaborate our framework.

Our algorithms can be used in the other DNNs, e.g. Lenet-5.

We adopt the notations of ResNet strucutre, in which the output of the ith block O i can be represented as: DISPLAYFORM0 where x the input of the first layer of the ith block, {W } i and W i respresent the filter weights in the ith block and the shortcut weight matrix, respectively.

The function F(·) represent the multiple convolutional layers.

Denote the weight matrix of the first convolutional layer as W conv1 and that of the fully connected layer as W f c .

Suppose there are I blocks, then we denote all the parameters of the network as Θ := {W conv1 , {W } 1 , ..., {W } I , W 1 , ..., W I , W f c } and Θ −W := Θ\W for W ∈ Θ. Our key objective is to train a sparse DNN of less parameters, and yet comparable performance to the non-sparse DNN.

The training function of DNN is defined as, DISPLAYFORM1 where P (·) is the penalty function of parameters Θ. If we use (X , y) as the sample set of the dataset; then in classification task, the loss function is the cross-entropy function as DISPLAYFORM2 where N and K are the number of samples and classes and p n,k (Θ) denotes the probability of the nth sample belongs to class k. Generally, we can use Stochastic Gradient Descent (SGD) algorithm to update Θ; and the algorithm is summarized in algorithm 1.Algorithm 1 SGD for ResNet 1: Input: Learning rate η, X and y 2: Initialize: DISPLAYFORM3

One direct intuition is to adopt the sparsity regularization on the parameters, or those of the one layer of the network, such as BID22 ; Wen et al. (2016b) .

To reduce the number of connection weights, one can consider different types of regularization, including (1) Lasso-type penalty (L 1 ), (2) Group-Lasso-type penalty BID34 ; (3) An iterative regularization path with structural sparsity (e.g., elastic net BID42 , and Split LBI Huang et al. (2016) ): here we employ the Split LBI which learns the structural sparsity via variable splitting and Linearized Bregman Iteration (LBI), due to the computational efficiency of the LBI, and model selection consistency, Lasso-type penalty can be directly implemented on the fully connection layer i as, DISPLAYFORM0 Under review as a conference paper at ICLR 2019Group-Lasso-type penalty BID34 aims at regularizing the groups of parameters Θ, and W (g) is a group of partial weights in Θ, DISPLAYFORM1 where DISPLAYFORM2 , and W (g) is the number of weights in W (g) ; G is the total number of groups.

This Split LBI algorithm BID12 introduces an augmented variable Γ which is enforced sparsity and kept close to W , by variable splitting term DISPLAYFORM0 .

Then the objective function turns to: DISPLAYFORM1 To enforce the sparsity of Γ, we here implement the LBI algorithm on the W , and the algorithm can be summarized in algorithm 2, where DISPLAYFORM2 The 5th-8th lines are Split LBI algorithm, which returns a regularization path of {Θ DISPLAYFORM3 It starts from the null model with Γ 0 = 0, and tends to select more and more variables as the algorithm evolves, until over-fitted.

At each step, the sparse estimator W k is the projection of W k onto the subset of the support set of Γ k .

The remainder of the projection is affected by weak signals with small magnitude and mostly the ones mainly affected by random noise.

Particularly, we highlight several points,•

The κ is the damping factor, which enjoys the low bias with larger value, however, at the sacrifice of high computational cost.

The α is the step size.

In BID12 , it has been proved that the α is the inverse scale with κ and should be small enough to ensure the statistical property.

In our scenario, we set it to 0.01/κ.• The t k = kα is the regularization parameter, which plays the similar role with λ in Lasso.

It's the trade-off between underfiting and overfiting, which can be determined via the loss/accuracy on the validation dataset.• The ν controls the difference between W and W .

In BID12 , it has been proved that larger value of ν can enjoy better model selection consistency, however may suffer from the larger parameter estimation error.

In ; BID39 , it has been proved that as long as ν 0, the dense estimator W can enjoy better prediction error by leveraging weak signals.

We will discuss it in the next subsection.• Each component of the closed form solution W ∈ R p1×p2 in equation 5 can be simplified as, DISPLAYFORM4

The pruning algorithm is inspired by the Fu et al. (2016b) .

Particularly, it has been pointed out in BID39 that the dense estimator can be orthogonally decomposed into three parts: strong signals which correspond to non-zero elements in W , weak signals and random noise.

Due to the ability to leverage additional weak signals as long as ν is large enough, it has been proved theoretically and experimentally that, the dense estimator outperforms the sparse estimator in prediction.

Algorithm 2 SGD for ResNet with Split LBI 1: Input: Learning rate η, ν > 0, step size of LBI α, damping factor κ > 0, X and y 2: Initialize: DISPLAYFORM0 This inspires us to sequentially consider all available solutions for all sparse variables along the Regularization Path (RP) by gradually decreasing the values of regularization coefficients.

Specifically, we can order the parameter set Θ according to the magnitude values of weights W .

Following this order, we identify the top r% of weights in Θ r .

The complementary set Θ 1−r = Θ\Θ r can be pruned.

Compared to the pruning methods in Han et al. (2015a) , we can prune the weights in the training process and do not need to fine-tune the weights.

Furthermore, one can easily extend algorithm 2 to prune at L (L > 1) layers.

We take the Split LBI as an example; the other two methods can also be directly applied to multiple layers.

The corresponding algorithm is described in algorithm 3.

Algorithm 3 SGD for ResNet with Split LBI on multiple layers 1: Input: Learning rate η, ν > 0, step size of LBI α, damping factor κ > 0, X and y 2: Initialize: DISPLAYFORM0 10: DISPLAYFORM1 We conduct the experiments on three datasets, namely, MNIST, CIFAR10, and MiniImageNet.

We use the standard supervised training and testing splits on all datasets, except MiniImageNet, whose setting is splitted by ourselves, and will be released.

The classification accuracy is reported on each dataset.

Competitors.

We compare three methods of pruning networks.

(1) Plain: we train a plain network and use the L 2 − penalty P (W ) = W 2 .

For all layers, we set the coefficient λ as 5e − 4 in Eq (1).

We prune the trained network by ranking the weights and filters, in term of their magnitude values in the descending order.

This pruning strategy can be taken as a simplified version of our pruning algorithm in Sec. 3.4.

(2) Rand.

We randomly remove the weights or filters in the networks.

This is a naive baseline.

(3) Ridge-Penalty (R-P) BID7 to rank the weights and filters by L 2 regularization path.

For that particular layer that we want to do the pruning, the coefficient λ would be finally set as 1e − 3.We also compare two variants of our PiT framework.

(4) Lasso-type penalty or Group-Lasso-type penalty (L-P / GL-P): the L-P is used to prune the weights of fully connected layers, and we employ the GL-P to directly remove the filters of convolutional layers.

(5) Split LBI or Group Split LBI penalty(S-P / GS-P): the split BLI penalty is utilized to prune the weights.

Accordingly, we have the Group Split LBI penalty by regularizing the groups of filter parameters as BID34 .

Note that all the results are trained for one time; and we do not have fine-tuning step after the pruning.

The handwritten digits MNIST dataset is widely used to experimentally evaluate the machine learning methods.

We use the standard supervised split and LeNet-5 LeCun et al. (1998) which is composed of 3 convolutional layers and 2 fully connected layers.

All the models are trained and get converged in 50 epochs.

Note that each experiment is repeated for five times, and the averaged results are reported.

In the experiments, we consider saving the portion of 100%, 50%, 25%, 12.5%, 6.25%, 3.13%, and 1.57% of original parameters on each layer.

Please refer to the Appendix for more detailed results.

Pruning each layer.

The results are shown in Tab.

1.

We employ our PiT algorithms to prune each individual layers of LeNet-5, while we keep the parameters of the other layers unchanged.

We have the following observations:(1) On two fully connected layers (fc.f6 and fc.f7), both the L-P and S-P of our PiT framework work very well.

For example, on the fc.f7 layer, our S-P only has 1.57% of the parameters on these layers.

Surprisingly, our performance is only 0.03% lower than that of the original network.

In contrast, we compare the pruning results with the baseline: Plain, Rand, and R-P. There is significant performance dropping with the more parameters pruned.

This shows the efficacy of our PiT framework.(2) On the convolutional layer (conv.c5), our L-P and S-P layers also achieve remarkable results.

Note that the conv.c5 layer has 48k out of 60k number of parameters in Lenet-5.

We show that our S-P saves 12.5% of total parameters of this layer (i.e., 42k number of parameters have been removed on this layer) and the results get only dropped by 0.3%.

This demonstrates that our PiT framework indeed can save the relatively important weights and filters, and effectively do the network pruning.(3) The conv.c3 layer is another convolutional layer in LeNet-5.

We found that this layer is very important to maintain a good performance of overall network.

Nevertheless, the results of our pruning L-P and S-P are still better than the other baselines.

Pruning two layers.

Totally, the LeNet-5 has 60k parameters, while the conv.c5 and fc.f6 have 48k and 10k number of parameters respectively.

That means these two layers have the most number of Table 2 : Pruning two layers in LeNet-5 on the MNIST dataset.

Each column indicates the percentage of parameter saved on these two layers.

Com-Rat, is short for the compression ratio of the total network, i.e., the ratio of saved parameters divided the total number of parameters of LeNet-5.

Table 3 : Pruning each block in ResNet-18 on Cifar-10 dataset.

Note that each block has two CNN layers.parameters.

In this case, we utilize our PiT algorithms to prune both fc.f6 + fc.f7, and conv.c5+fc.f6 layers.

The results are reported in Tab.

2.

We can show that our PiT framework can still efficiently compress the network while preserve significant performance.

The best compressed model.

When we prune the conv.c5 and fc.f6 layers, our model can achieve the best and efficient performance.

With only 17.60% parameter size of original LeNet-5, our model can beat the performance as high as 98.47%.

Remarkably, our PiT framework has not done any fine-tuning and re-training the pruned network by any other dataset.

This suggests that our PiT can indeed uncover the important weights and filters.

Our best models will be downloaded online.

Table 4 : Pruning multiple blocks in ResNet-18 on Cifar-10 dataset. (Chance-level = 10%).

ComRat, is short for compression ratio of the total network, as in Tab.

2.

The CIFAR-10 dataset consists of 60,000 images of size 32 × 32 in 10 classes, with 6000 images per class on average.

There are 50,000 training images and 10,000 test images.

We use the standard supervised split; and ResNet-18 is employed as the classification network.

All the models are trained and get converged in 40 epochs.

Note that each experiment is repeated for five times, and the averaged results are reported.

We still show the results which have 100%, 50%, 25%, 12.5%, 6.25%, 3.13%, and 1.57% parameter size of original networks on each layer.

Pruning one Residual Block.

The results are shown in Tab.

3.

In this table, we apply our PiT algorithm on one residual block while the other layers are unchanged.

We draw several conclusions,(1) Our PiT framework (i.e., GS-P and GL-P) can efficiently train and prune the network.

From Block #3.0 -Block #4.1, surprisingly the pruned network with 1.57% of original parameter size of ResNet-18, can also achieve almost the same recognition accuracies as the non-pruned ResNet-18.

From Block #1.0 -Block #2.1, the smallest pruned ratio of PiT can still hit significant high performance if compared with the other competitors.

This reflects the efficacy of our pruning algorithm.

In particular, in the training process, our PiT framework is optimized to learn and select the important weights or filters; and our PiT can thus conduct a direct dimension reduction of these parameters.(2) By the increased ratio of pruned parameters, the R-P method can also have better performance than Rand, and Plain methods.

This shows that our pruning algorithm also works in the general cases.

However, the R-P is not enforcing the sparse constraints in learning the weight parameters of network.

Thus it has inferior performance to two PiT methods.

Pruning multiple blocks.

The ResNet-18 totally has around 10.95M parameters.

Block #4.1, #4.0, #3.1, #3.0 have 4.7M , 3.5M , 1.2M , and 0.88M parameters.

These blocks have the most number of parameters; and we prune these multiple blocks.

The results are shown in Tab.

4. Note that even only 1.57% parameter size of those layers are saved, our PiT algorithms (GL-P, and GS-P) can still remain remarkable high recognition accuracy.

Again it shows the efficacy of our PiT framework.

TAB6 : Top 5 accuracy on miniImagenet by pruning ResNet-18, the fully connected layer, Block#4.0 and #4.1 layers.

The miniImageNet dataset is a subset of ImageNet and is composed of 60,000 images in 100 categories.

In each category, we take 500 images as training set and other 100 as testing set.

We also use the ResNet-18 structure on miniImageNet.

All the models are trained and get converged in 50 epochs.

Note that each experiment is repeated for five times, and the averaged results are reported.

In term of the analysis in Sec. 4.2, we prune the fully connected layer, Block #4.0, and #4.1.

The results are shown in Tab.

5.When no parameters are pruned, the R-P can achieve better results than our PiT algorithms.

These results make sense, since the ridge penalty does not enforce the sparsity to the network 1 .

However, with the increased ratio of parameters pruned, the performance of R-P gets degraded dramatically.

In contrast, the results of our methods in PiT framework get decreased very slow.

For example, when only 50% are saved in all the layers, the Top-5 accuracies are reduced by only 0% and 1.1% for L-P / GL-P, and S-P / GS-P respectively.

Remarkably, if we only save 1.57% of original parameters on those layers, the S-P / GS-P can still is as high as 65.17, which is only 13.78% performance dropped.

Again, note that all the methods have not done any fine-tuning step, and only been trained in one round.

That means our S-P / GS-P can indeed select the most expressive weights or filters, and thus reduce the size of networks.

As the experiments shown in these three datasets, our PiT indeed can learn to prune networks without fine-tuning.

We give some further discussion and highlight the potential future works, 1.

In all our experiments, our L-P / GL-P, and S-P / GS-P are applied to, at most, four layers in one network.

Theoretically, our PiT algorithms should be able to be directly applied to any layers of DNNs, since PiT only adds some sparse penalties in the loss functions.

However, in practice, we found that the network training algorithm, i.e., SGD in Alg.

3, is unstable, if we apply the sparse penalties more than four layers.

It will take much more time and training epochs to get the networks converged.

2.

Essentially, our PiT presents a feature selection algorithm, which can dynamically learn the importance of weights and filters in the learning process; mostly importantly, we donot need any fine-tuning step, which, we believe, will destroy values and properties of selected weights and filters.

Therefore, it would be very interesting to analyze the statistical properties of selected features in each layer.

3.

Theoretically, we can not guarantee the orthogonality of weights and filters in the trained model.

Empirically, we adapt some strategies.

For example, the weights and filters of each layer can be orthogonally initialized; and we apply the common regularization tricks, e.g., dropout, and batch normalization.

These can help decorrelate the learned parameters of the same layers.

Practically, our PiT framework works well in selecting the important parameters and prune the networks as shown in the experiments.

We also visualize the correlation between removed and none removed filters in the Appendix.

4.

It is a conjecture that the capacity of DNNs may be too large to learn a small dataset;and it is essential to do network pruning.

However, it is also an open question as how to numerically measure the capacity of DNNs and the complexity of one dataset.

@highlight

we propose an algorithm of learning to prune network by enforcing structure sparsity penalties

@highlight

This paper introduces an approach to pruning while training a network using lasso and split LBI penalties
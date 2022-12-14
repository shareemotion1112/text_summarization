We introduce a new deep convolutional neural network, CrescendoNet, by stacking simple building blocks without residual connections.

Each Crescendo block contains independent convolution paths with increased depths.

The numbers of convolution layers and parameters are only increased linearly in Crescendo blocks.

In experiments, CrescendoNet with only 15 layers outperforms almost all networks without residual connections on benchmark datasets, CIFAR10, CIFAR100, and SVHN.

Given sufficient amount of data as in SVHN dataset, CrescendoNet with 15 layers and 4.1M parameters can match the performance of DenseNet-BC with 250 layers and 15.3M parameters.

CrescendoNet provides a new way to construct high performance deep convolutional neural networks without residual connections.

Moreover, through investigating the behavior and performance of subnetworks in CrescendoNet, we note that the high performance of CrescendoNet may come from its implicit ensemble behavior, which differs from the FractalNet that is also a deep convolutional neural network without residual connections.

Furthermore, the independence between paths in CrescendoNet allows us to introduce a new path-wise training procedure, which can reduce the memory needed for training.

Deep convolutional neural networks (CNNs) have significantly improved the performance of image classification BID3 BID25 .

However, training a CNN also becomes increasingly difficult with the network deepening.

One of important research efforts to overcome this difficulty is to develop new neural network architectures BID6 BID14 .Recently, residual network BID3 and its variants BID8 have used residual connections among layers to train very deep CNN.

The residual connections promote the feature reuse, help the gradient flow, and reduce the need for massive parameters.

The ResNet BID3 and DenseNet BID8 achieved state-of-the-art accuracy on benchmark datasets.

Alternatively, FractalNet BID14 expanded the convolutional layers in a fractal form to generate deep CNNs.

Without residual connections BID3 and manual deep supervision BID15 , FractalNet achieved high performance on image classification based on network structural design only.

Many studies tried to understand reasons behind the representation view of deep CNNs.

BID27 showed that residual network can be seen as an ensemble of relatively shallow effective paths.

However, BID2 argued that ensembles of shallow networks cannot explain the experimental results of lesioning, layer dropout, and layer reshuffling on ResNet.

They proposed that residual connections have led to unrolled iterative estimation in ResNet.

Meanwhile, BID14 speculated that the high performance of FractalNet was due to the unrolled iterative estimation of features of the longest path using features of shorter paths.

Although unrolled iterative estimation model can explain many experimental results, it is unclear how it helps improve the classification performance of ResNet and FractalNet.

On the other hand, the ensemble model can explain the performance improvement easily.

In this work, we propose CrescendoNet, a new deep convolutional neural network with ensemble behavior.

Same as other deep CNNs, CrescendoNet is created by stacking simple building blocks, called Crescendo blocks FIG0 ).

Each Crescendo block comprises a set of independent feed-forward paths with increased number of convolution and batch-norm layers (Ioffe & Szegedy, 2015a).

We only use the identical size, 3 ?? 3, for all convolutional filters in the entire network.

Despite its simplicity, CrescendoNet shows competitive performance on benchmark CIFAR10, CI-FAR100, and SVHN datasets.

Similar to FractalNet, CrescendoNet does not include residual connections.

The high performance of CrescendoNet also comes completely from its network structural design.

Unlike the FractalNet, in which the numbers of convolutional layers and associated parameters are increased exponentially, the numbers of convolutional layers and parameters in Crescendo blocks are increased linearly.

CrescendoNet shows clear ensemble behavior (Section 3.4).

In CrescendoNet, although the longer paths have better performances than those of shorter paths, the combination of different length paths have even better performance.

A set of paths generally outperform its subsets.

This is different from FractalNet, in which the longest path alone achieves the similar performance as the entire network does, far better than other paths do.

Furthermore, the independence between paths in CrescendoNet allows us to introduce a new pathwise training procedure, in which paths in each building block are trained independently and sequentially.

The path-wise procedure can reduce the memory needed for training.

Especially, we can reduce the amortized memory used for training CrescendoNet to about one fourth.

We summarize our contribution as follows:??? We propose the Crescendo block with linearly increased convolutional and batch-norm layers.

The CrescendoNet generated by stacking Crescendo blocks further demonstrates that the high performance of deep CNNs can be achieved without explicit residual learning.??? Through our analysis and experiments, we discovered an emergent behavior which is significantly different from which of FractalNet.

The entire CrescendoNet outperforms any subset of it can provide an insight of improving the model performance by increasing the number of paths by a pattern.??? We introduce a path-wise training approach for CrescendoNet, which can lower the memory requirements without significant loss of accuracy given sufficient data.

Crescendo Block The Crescendo block is built by two layers, the convolution layer with the activation function and the following batch normalization layer BID10 .

The convolutional layers have the identical size, 3 ?? 3.

The Conv-Activation-BatchNorm unit f 1 , defined in the Eq.1 is the base branch of the Crescendo block.

We use ReLU as the activation function to avoid the problem of vanishing gradients BID17 .

DISPLAYFORM0 The variable z denotes the input feature maps.

We use two hyper-parameters, the scale S and the interval I to define the structure of the Crescendo block H S .

The interval I specifies the depth difference between every two adjacent branches and the scale S sets the number of branches per block.

The structure of the n th branch is defined by the following equation: DISPLAYFORM1 where the superscript nI is the number of recursion time of the function f 1 .

The structure of Crescendo block H S can be obtained below: DISPLAYFORM2 where ??? denotes an element-wise averaging operation.

Note that the feature maps from each path are averaged element-wise, leaving the width of the channel unchanged.

A Crescendo block with S = 4 and I = 1 is shown in FIG0 .The structure of Crescendo block is designed for exploiting more feature expressiveness.

The different depths of parallel paths lead to different receptive fields and therefore generate features in different abstract levels.

In addition, such an incremental and parallel form explicitly supports the ensemble effects, which shows excellent characteristics for efficient training and anytime classification.

We will explain and demonstrate this in the following sections.

CrescendoNet Architecture The main body of CrescendoNet is composed of stacked Crescendo blocks with max-pooling layers between adjacent blocks ( FIG0 ).

Following the main body, like most deep CNNs, we use two fully connected layers and a soft-max layer as the classifier.

In all experiments, the two fully connected layers have 384 hidden units and 192 hidden units respectively.

The overall structure of CrescendoNet is simple and we only need to tune the Crescendo block to modify the entire network.

To reduce the memory consumption during training CrescendoNet, we propose a path-wise training procedure, leveraging the independent multi-path structure of our model.

We denote stacked Conv-BatchNorm layers in one Crescendo block as one path.

We train each path individually, from the shortest to the longest repetitively.

When we are training one path, we freeze the parameters of other paths.

In other words, these frozen layers only provide learned features to support the training.

FIG1 illustrates the procedure of path-wise training within a CrescendoNet block containing four paths.

There are two advantages of path-wise training.

First, path-wise training procedure significantly reduces the memory requirements for convolutional layers, which constitutes the major memory cost for training CNNs.

For example, the higher bound of the memory required for computation and storage of gradients using momentum stochastic gradient descent algorithms can be reduced to about 40% for a Crescendo block with 4 paths where interval = 1.

Second, path-wise training works well with various optimizers and regularizations.

Even dropout and drop-path can be applied to the model during the training.

Dropout and drop-connect BID28 , which randomly set a selected subset of activations or weights to zero respectively, are effective regularization techniques for deep neural networks.

Their variant, drop-path BID14 , shows further performance improvement by dropping paths when training FractalNet.

We use both dropout and drop-path for regularizing the Crescendo block.

We drop the branches in each block with a predefined probability.

For example, given drop-path rate, p = 0.3, the expectation of the number of dropped branches is 1.2 for a Crescendo block with four branches.

For the fully connected layers, we use L2 norm of their weights as an additional term to the loss.

We evaluate our models with three benchmark datasets: CIFAR10, CIFAR100 , and Street View House Numbers (SVHN) BID19 .

CIFAR10 and CIFAR100 each have 50,000 training images and 10,000 test images, belonging to 10 and 100 classes respectively.

All the images are in RGB format with the size of 32 ?? 32-pixel.

SVHN are color images, with the same size of 32 ?? 32-pixel, containing 604,388 and 26,032 images for training and testing respectively.

Note that these digits are cropped from a series of numbers.

Thus, there may be more than one digit in an image, but only the one in the center is used as the label.

For data augmentation, we use a widely adopted scheme BID16 BID14 BID6 b; BID24 BID21 BID3 .

We first pad images with 4 zero pixels on each side, then crop padded images to 32 ?? 32-pixel randomly and horizontally flipping with a 50% probability.

We preprocess each image in all three datasets by subtracting off the mean and dividing the variance of the pixels.

We use Mini-batch gradient descent to train all our models.

We implement our models using TensorFlow distributed computation framework BID0 and run them on NVidia P100 GPU.

We also optimize our models by adaptive momentum estimation (Adam) optimization BID11 and Nesterov Momentum optimization BID18 respectively.

For Adam optimization, we set the learning rate hyper-parameter to 0.001 and let Adam adaptively tune the learning rate during the training.

We choose the momentum decay hyper-parameter ?? 1 = 0.9 and ?? 2 = 0.999.

And we set the smoothing term = 10 ???8 .

This configuration is the default setting for the AdamOptimizer class in TensorFlow.

For Nesterov Momentum optimization, we set the hyper-parameter momentum = 0.9.

We decay the learning rate from 0.1 to 0.01 after 512 epochs for CIFAR and from 0.05 to 0.005, then to 0.0005, after 42 epochs and 63 epochs respectively for SVHN.

We use truncated normal distribution for parameter initialization.

The standard deviation of hyper-parameters is 0.05 for convolutional weights and 0.04 for fully connected layer weights.

For all datasets, we use the batch size of 128 on each training replica.

For the whole net training, we run 700 epochs on CIFAR and 70 epochs on SVHN.

For the path-wise training, we run 1400 epochs on CIFAR and 100 epochs on SVHN.Using a CrescendoNet model with three blocks each contains four branches as illustrated in FIG0 , we investigate the following preliminary aspects: the model performance under different block widths, the ensemble effect, and the path-wise training performance.

We study the Crescendo block with three different width configurations: equal width globally, equal width within the block, and increasing width.

All the three configurations have the same fully connected layers.

For the first one, we set the number of feature maps to 128 for all the convolutional layers.

For the second, the numbers of feature maps are (128, 256, 512) for convolutional layers in each block.

For the last, we gradually increase the feature maps for each branch in three blocks to (128, 256, 512) correspondingly.

For example, the number of feature maps for the second and fourth branches in the second block is (192, 256) and (160, 192, 224, 256) .

The exact number of maps for each layer is defined by the following equation: DISPLAYFORM0 where n maps denotes the number of feature maps for a layer, n inmaps and n outmaps are number of input and output maps respectively, n layers is the number of layers in the block, and i layer is the index of the layer in the branch, starting from 1.To inspect the ensemble behavior of CrescendoNet, we compare the performance of models with and without drop-path technique and subnets composed by different combinations of branches in each block.

For the simplicity, we denote the branch combination as a set P containing the index of the branch.

For example, P = {1, 3} means the blocks in the subnet only contains the first and third branches.

The same notation is used in TAB0 and FIG2 .

Table 1 gives a comparison among CrescendoNet and other representative models on CIFAR and SVHN benchmark datasets.

For five datasets, CrescendoNet with only 15 layers outperforms almost all networks without residual connections, plus original ResNet and ResNet with Stochastic Depth.

For CIFAR10 and CIFAR100 without data augmentation, CrescendoNet also performs better than all the given models except DenseNet with bottleneck layers and compression (DenseNet-BC) with 250 layers.

However, CrescendoNet's error rate 1.76% matches the 1.74% error rate of given DenseNet-BC, on SVHN dataset which has plentiful data for each class.

Comparing with FractalNet, another outstanding model without residual connection, CrescendoNet has a simpler structure, fewer parameters, but higher accuracies.

The lower rows in Table 1 compare the performance of our model given different configuration.

In three different widths, the performance simultaneously grows with the number of feature maps.

In other words, there is no over-fitting when we increase the capacity of CrescendoNet in an appropriate scope.

Thus, CrescendoNet demonstrates a potential to further improve its performance by scaling up.

In addition, the drop-path technique shows its benefits to our models on all the datasets, just as it does to FractalNet.

Another interesting result from Table 1 is the performance comparison between Adam and Nesterov Momentum optimization methods.

Comparing with Nesterov Momentum method, Adam performs similarly on CIFAR10 and SVHN, but worse on CIFAR100.

Note that there are roughly 60000, 5000, and 500 training images for each class in SVHN, CIFAR10, and CIFAR100 respectively.

This implies that Adam may be a better option for training CrescendoNet when the training data is abundant, due to the convenience of its adaptive learning rate scheduling.

The last row of Table 1 gives the result from path-wise training.

Training the model with less memory requirement can be achieved at the cost of some performance degradation.

However, Pathwise trained CrescendoNet still outperform many of networks without residual connections on given datasets. (128, 256, 512) .

The results show the ensemble behavior of our model.

Specifically, the more paths contained in the network, the better the Table 1 : Whole net classification error (%) on CIFAR10/CIFAR100/SVHN.

We highlight the top three accuracies in each column with the bold font.

The three numbers in the parentheses denote the number of output feature maps of each block.

The plus sign (+) denotes the data augmentation.

The sign (-W) means that the feature maps of layers in each branch increase as explained in the model configuration section.

The compared models include: Network in Network BID24 , ALL-CNN (Springenberg et al., 2014) , Deeply Supervised Net BID15 , Highway Network BID24 , FractalNet BID14 , ResNet BID3 , ResNet with Stochastic Depth BID7 , Wide ResNet BID30 , and DenseNet BID6 performance.

And the whole net outperforms any single path network with a large margin.

For example, the whole net and the net based on the longest path show the inference error rate of 6.90% and 10.69% respectively, for CIFAR10 without data augmentation.

This implicit ensemble behavior differentiates CrescendoNet from FractalNet, which shows a student-teacher effect.

Specifically, the longest path in FractalNet can achieve a similar or even lower error rate compared to the whole net.

To investigate the dynamic behavior of subnets, we test the error rate changes of subnets during the training.

We use Adam to train the CrescendoNet with the structure shown in FIG0 on CIFAR10 for 450 epochs.

FIG2 illustrates the behavior of different path combinations during the training.

It shows that the inference accuracy of the whole net grows simultaneously with all the subnets, which demonstrates the ensemble effect.

Second, for any single path network, the performance grows with the depth.

This behavior of the anytime classifier is also shown by FractalNet.

In other words, we could use the short path network to give a rough but quick inference, then use more paths to gradually increase the accuracy.

This may be useful for time-critical applications, like integrated recognition system for autonomous driving tasks.

Conventional deep CNNs, such as AlexNet VGG-19 (Simonyan & BID20 , directly stacked the convolutional layers.

However, the vanishing gradient problem makes it difficult to train and tune very deep CNN of conventional structures.

Recently, stacking small convolutional blocks has become an important method to build deep CNNs.

Introducing new building blocks becomes the key to improve the performance of deep CNN.

BID16 first introduced the NetworkInNetwork module which is a micro neural network using a multiple layer perceptron (MLP) for local modeling.

Then, they piled the micro neural networks into a deep macro neural network.

BID25 introduced a new building block called Inception, based on which they built GoogLeNet.

Each Inception block has four branches of shallow CNNs, building by convolutional kernels with size 1 ?? 1, 3 ?? 3, 5 ?? 5, and max-pooling with kernel size 3 ?? 3.

Such a multiple-branch scheme is used to extract diversified features while reducing the need for tuning the convolutional sizes.

The main body of GoogLeNet has 9 Inception blocks stacked each other.

Stacking multiplebranch blocks can create an exponential combination of feed-forward paths.

Such a structure com-bined with the dropout technique can show an implicit ensemble effect BID27 BID22 .

GoogLeNet was further improved with new blocks to more powerful models, such as Xception BID1 and Inception-v4 BID26 .

To improve the scalability of GoogLeNet, BID26 used convolution factorization and label-smoothing regularization in Inception-v4.

In addition, BID1 explicitly defined a depth-wise separable convolution module replacing Inception module.

Recently, BID14 introduced FractalNet built by stacked Fractal blocks, which are the combination of identical convolutional layers in a fractal expansion fashion.

FractalNet showed that it is possible to train very deep neural network through the network architecture design.

FractalNet implicitly also achieved deep supervision and student-teacher learning by the fractal architecture.

However, the fractal expansion form increases the number of convolution layers and associated parameters exponentially.

For example, the original FractalNet model with 21 layers has 38.6 million parameters, while a ResNet of depth 1001 with similar accuracy has only 10.2 million parameters BID6 .

Thus, the exponential expansion reduced the scalability of FractalNet.

Another successful idea in network architecture design is the use of skip-connections BID3 b; BID6 BID30 BID29 .

ResNet BID3 used the identity mapping to short connect stacked convolutional layers, which allows the data to pass from a layer to its subsequent layers.

With the identity mapping, it is possible to train a 1000-layer convolutional neural network.

BID6 recently proposed DenseNet with extremely residual connections.

They connected each layer in the Dense block to every subsequent layer.

DenseNet achieved the best performance on benchmark datasets so far.

On the other hand, Highway networks BID23 used skip-connections to adaptively infuse the input and output of traditional stacked neural network layers.

Highway networks have helped to achieve high performance in language modeling and translation.

CNN has shown excellent performance on image recognition tasks.

However, it is still challenging to tune, modify, and design an CNN.

We propose CrescendoNet, which has a simple convolutional neural network architecture without residual connections BID3 .

Crescendo block uses convolutional layers with same size 3 ?? 3 and joins feature maps from each branch by the averaging operation.

The number of convolutional layers grows linearly in CrescendoNet while exponentially in FractalNet BID14 .

This leads to a significant reduction of computational complexity.

Even with much fewer layers and a simpler structure, CrescendoNet matches the performance of the original and most of the variants of ResNet on CIFAR10 and CIFAR100 classification tasks.

Like FractalNet BID14 , we use dropout and drop-path as regularization mechanisms, which can train CrescendoNet to be an anytime classifier, namely, CrescendoNet can perform inference with any combination of the branches according to the latency requirements.

Our experiments also demonstrated that CrescendoNet synergized well with Adam optimization, especially when the training data is sufficient.

In other words, we can avoid scheduling the learning rate which is usually performed empirically for training existing CNN architectures.

CrescendoNet shows a different behavior from FractalNet in experiments on CIFAR10/100 and SVHN.

In FractalNet BID14 , the longest path alone achieves the similar performance as the entire network, far better than other paths, which shows the student-teacher effect.

The whole FractalNet except the longest path acts as a scaffold for the training and becomes dispensable later.

On the other hand, CrescendoNet shows that the whole network significantly outperforms any set of it.

This fact sheds the light on exploring the mechanism which can improve the performance of deep CNNs by increasing the number of paths.

@highlight

We introduce CrescendoNet, a deep CNN architecture by stacking simple building blocks without residual connections.
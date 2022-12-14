In recent years, three-dimensional convolutional neural network (3D CNN) are intensively applied in the video analysis and action recognition and receives good performance.

However, 3D CNN leads to massive computation and storage consumption, which hinders its deployment on mobile and embedded devices.

In this paper, we propose a three-dimensional regularization-based pruning method to assign different regularization parameters to different weight groups based on their importance to the network.

Our experiments show that the proposed method outperforms other popular methods in this area.

In recent years, convolutional neural network (CNN) has developed rapidly and has achieved remarkable success in computer vision tasks such as identification, classification and segmentation.

However, due to the lack of motion modeling, this image-based end-to-end feature can not directly apply to videos.

In BID0 BID1 , the authors use three-dimensional convolutional networks (3D CNN) to identify human actions in videos.

Tran et al. proposed a 3D CNN for action recognition which contains 1.75 million parameters BID2 .

The development of 3D CNN also brings challenges because of its higher dimensions.

This leads to massive computing and storage consumption, which hinders its deployment on mobile and embedded devices.

In order to reduce the computation cost, researchers propose methods to compress CNN models, including knowledge distillation BID3 , parameter quantization BID4 BID5 , matrix decomposition BID6 and parameter pruning BID7 .

However, all of the above methods are based on two-dimensional convolution.

In this paper, we expand the idea of BID8 to 3D CNN acceleration.

The main idea is to add group regularization items to the objective function and prune weight groups gradually, where the regularization parameters for different weight groups are differently assigned according to some importance criteria.

For a three-dimensional convolutional neural network with L layers, the weights of the DISPLAYFORM0 l and D l are the dimensions along the axes of filter, channel, spatial height, spatial width and spatial depth.

The proposed objective function for structured sparsity regularization is defined by Eqn.

BID0 .

Here L(W) is the loss on data; R(W) is the non-structured regularization (L 2 norm in this paper).

R g is the structured sparsity regularization on each layer.

In BID9 BID10 , the authors used the same ?? g for all groups and adopted Group LASSO for R g .

Recently Wang et al. BID8 use the squared L 1 norm for R g and vary the regularization parameters ?? g for different groups.

We build on top of that approach but extend it from two dimensions to three dimensions.

DISPLAYFORM1 The structure learned is determined by the way of splitting groups of W (l)g .There are normally filer-wise, channel-wise, shape-wise, and depth-wise structured sparsity with different ways of grouping BID9 .

Pruning of different weight groups for 3D CNN is shown in FIG0 .In BID8 , Wang et al. theoretically proved that by increasing the regularization parameter ?? g , the magnitude of weights tends to be minimized.

The more ?? g increases, the more magnitude of weights are compressed to zero.

Therefore, we can assign different ?? g for the weight groups based on their importance to the network.

Here, we use the L 1 norm as a criterion of importance.

Our goal is to prune RN g weight groups in the network, where R is the pruning ratio to each layer and N g is total number of weight groups in the layer.

In other words, we need to prune RN g weight groups which ranks lower in the network.

We sort the weight groups in ascending order of the L 1 norms.

In order to remove the oscillation of ranks during one training iteration, we averaged the rank through training iterations to obtain the average rank r avg in N training iterations: r avg = 1 N N n=1 r n .

The final average rank r is obtained by sorting r avg of different weight groups in ascending order, making its range from 0 to N g ??? 1.

The update of ?? g is determined by the following formula: ?? DISPLAYFORM2 Here ????? g is the function of average rank r, we follow the formula proposed by Wang BID8 as follows: DISPLAYFORM3 Here A is a hyperparameter which controls the speed of convergence.

According to Eqn.(2), we can see that ????? g is zero when r = RN g because we need to increase the regularization parameters of the weight groups whose ranks are below RN g to further decrease their L 1 norms; and for those with greater L 1 norms and rank above RN g , we need to decrease their regularization parameters to further increase their L 1 norms.

Thus, we can ensure that exactly RN g weight groups are pruned at the final stage of the algorithm.

When we obtain ?? (new) g , the weights can be updated through back-propagation deduced from Eqn.

BID0 .

Further details can be found in BID8 .

Our experiments are carried out by Caffe BID11 .

We set the weight decay factor ?? to be the same as the baseline and set hyper-parameter A to half of ??.

We only compress the weights in convolutional layers and leave the fully connected layers unchanged because we focus on network acceleration.

The pruning ratios of the convolutional layers are set to the same for convenience.

The methods used for comparison are Taylor Pruning (TP) BID12 and Filter Pruning (FP) BID13 .

For all experiments, the ratio of speedup is calculated by GFLOPS reduction.

We apply the proposed method to C3D BID2 , which is composed of 8 convolution layers, 5 max-pooling layers, and 2 fully connected layers.

We download the open Caffe model as our pre-trained model, whose accuracy on UCF101 dataset is 79.94%.

UCF101 contains 101 types of actions and a total of 13320 videos with a resolution of 320 ?? 240.

All videos are decoded into image files with 25 fps rate.

Frames are resized into 128 ?? 171 and randomly cropped to 112 ?? 112.

Then frames are split into non-overlapped 16-frame clips which are then used as input to the networks.

The results are shown in TAB0 .

With different speedup ratios, our approach is always better than TP and FP.

We further demonstrate our method on 3D-ResNet18 BID2 , which has 17 convolution layers and 1 fully-connected layer.

The network is initially trained on the Sport-1M database.

We download the model and then fine-tune it by UCF101 for 30000 iterations, obtaining an accuracy of 72.50%.

The video preprocessing method is the same as above.

The training settings are similar to that of C3D.Experimental results are shown in TAB1 .

Our approach only suffers 0.91% increased error while achieving 2?? acceleration, obtaining better results than TP and FP.

FIG2 shows the loss during the pruning process for different methods.

As the number of iterations increases, the losses of TP and FP change dramatically, while the loss of our method remains at a lower level consistently.

This is probably because the proposed method imposes gradual regularization, making the network changes little-by-little in the parameter space, while both the TP and FP direct prune less important weights once for all.

In this paper, we implement the regularization based method for 3D CNN acceleration.

By assigning different regularization parameters to different weight groups according to the importance estimation, we gradually prune weight groups in the network.

The proposed method achieves better performance than other two popular methods in this area.

@highlight

In this paper, we propose a three-dimensional regularization-based pruning method to accelerate the 3D-CNN.
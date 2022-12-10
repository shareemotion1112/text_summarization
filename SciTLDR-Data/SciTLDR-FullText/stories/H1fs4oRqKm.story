Unsupervised monocular depth estimation has made great progress after deep learning is involved.

Training with binocular stereo images is considered as a good option as the data can be easily obtained.

However, the depth or disparity prediction results show poor performance for the object boundaries.

The main reason is related to the handling of occlusion areas during the training.

In this paper, we propose a novel method to overcome this issue.

Exploiting disparity maps property, we generate an occlusion mask to block the back-propagation of the occlusion areas during image warping.

We also design new networks with flipped stereo images to induce the networks to learn occluded boundaries.

It shows that our method achieves clearer boundaries and better evaluation results on KITTI driving dataset and Virtual KITTI dataset.

Monocular depth estimation becomes an active research topic as deep learning is applied in various computer vision tasks.

It has many applications, from navigation through to scene understanding.

A single traditional camera can be a cheaper alternative to the expensive LIDAR sensor for automotive cars if accurate estimation can be achieved.

Meanwhile, single camera simplifies the design of depth estimation solution which can be adopted quite widely at a low cost.

One straight-forward way to train deep depth estimation models is to use ground truth depth images as the supervision signals BID1 .

However, supervised deep learning method is eager for massive data with ground truth.

Collecting large datasets with ground truth depth in varied real scenarios is challenge and expensive.

Instead, training using stereo images without depth label is an alternative option.

BID7 proposed a method to exploit the left-right consistency of stereo images to tackle the monocular depth estimation, which achieved quite promising results.

However, the depth predicted by their method has blurred boundaries.

The issue is mainly due to the occlusions during the image warping.

Though it can be alleviated in some extent with proper post processing, the fundamental problem is not well addressed.

In this paper, we propose a new method to overcome the blurred boundaries when using stereo pairs to train the monocular depth model.

An example is illustrated in FIG0 .

During the image warping, we generate an occlusion mask using the disparity map to block the inappropriate back-propagation gradients for occlusion areas.

However, the mask only cannot guarantee clear boundaries as there is no constrain for the masked areas.

Then we design new networks to fully exploit the information of stereo images.

With flipped stereo pairs, the network is induced to learn clear boundaries for occlusion areas.

Our method provides a solution to the fundamental learning difficulty of occluded areas introduced by image warping in depth estimation.

Empirical evaluation on KITTI driving dataset BID6 ) and Virtual KITTI dataset BID4 ) demonstrates the effectiveness of our approach.

Moreover, we find the depth label of KITTI 2015 is usually very sparse near the object boundaries, which is not very sensitive to evaluate the clearness of boundaries.

Large amounts of multi-view based approaches have been proposed such as stereo matching BID22 ), difference view point BID3 ) or temporal sequence BID18 ).

Here we briefly review work based on single view depth estimation which is usually harder since reasoning depth from monocular colored image is an ill-posed problem.

The most intuition way is treating monocular depth estimation problem as a supervised problem by taking RGB images as inputs and Lidar depth points as ground truth.

BID21 proposed a method known as Make3d which breaks the image into homogeneous patches.

For each small homogeneous patch, Saxena et al used a Markov Random Field (MRF) to infer a set of plane parameters that capture both the 3D location and 3D orientation.

However, this approach has a hard time capturing thin structures since the predictions are made locally.

BID1 first exploited CNN in a coarse to fine manner.

BID13 BID12 proposed a network jointly explore the capacity of deep CNN and continuous conditional random field (CRF).

BID10 incorporated semantic segmentations in to single-view depth estimation task since they are closely tied to the property of perspective geometry.

BID11 proposed a residual network with up-sampling module using fully convolutional architecture.

Also, reverse Huber loss was introduced.

However, large amount of high-quality labelled data is needed, which is hard to require in practice.

To overcome the lack of high quality labelled data, several semi-supervised and fully unsupervised methods have been proposed.

BID2 first proposed a view synthesis based method called DeepStereo which generates new view image from nearby image.

BID24 FIG2 proposed a method which generates the right image through probability distribution over all the possible disparities for each pixel.

BID5 first proposed a warp based method by aligning reconstructed image with the ground truth left image as described in 3.1.

However, their loss is not fully differentiable.

BID7 improve this methods by introducing a novel loss.

BID19 extended the network into two separate channel with 6 or 12 losses which improves the result.

Based on Garg et al, BID9 proposed a semi-supervised methods that exploited both the sparse Lidar points as supervision and stereo pairs as unsupervion signals.

The semisupervised method was further improved by BID15 , they decoupled the monocular depth prediction problem into two procedure, a view synthesis procedure followed by stereo matching.

Recently, several work using only monocular temporal sequence comes out which enables more training data such as video sequence on YouTube.

BID26 proposed a network that predicts depth and camera pose separately.

Using the predicted depth and camera pose, relative temporal image can be reconstructed by image warping with which final loss can be constructed.

BID16 performed a novel 3D loss that enforces consistency of the estimated 3D point clouds and ego-motion across consecutive frames and combined it with 2D photometric loss.

proposed a differentiable implementation of Direct Visual Odometry (DVO) and a novel depth normalization strategy.

However, all the temporal sequence based training meet the same problem of object motion.

This problem can be alleviated by including stereo pairs during training known as trinocular training BID25 BID17 BID8 ).

However, all these warp based methods have the difficulty of learning occluded area which would infect the result.

Godard et al achieves state of art result of unsupervised monocular depth estimation with only stereo images.

Follow Godard et al, we proposed monocular depth estimation network with novel mask methods which can be trained end-to-end and without ground-truth label.

Our method is superior to Godard et al in result quality with clearer boundaries, especially on dense evaluation dataset such as virtual-KITTI BID4 ).

In general, our goal is to learn a network that can predict a pixel-wise dense depth map from single colored image( DISPLAYFORM0 However, all supervised methods have a hard time with acquiring large dense labelled data in real scenarios.

Thus, several unsupervised methods have been proposed to overcome the obstacle.

Among these methods, training by image reconstruction using rectified stereo pairs became more and more popular currently due to its high accuracy and easy accessibility of training data.

BID5 , Left-right consistency network proposed by Godard et al, Our refined network without shared parameters and our refined network with shared parameters First proposed by BID5 , the monodepth estimation network takes left image (I l ) as input and outputs the disparity aligned with the right image (d r ) which can be used to sample from the left image (I l ) to reconstruct the right image (Ĩ r ) during training.

Thus, image reconstruction loss can be constructed between the reconstructed right image(Ĩ r ) and original right image(I r ).

When testing, only one colored image are required, the predicted disparity can be converted to depth simply using depth = b * f /disparity, where b and f are given baseline and camera focal length respectively.

It is worth to mention that disparities (d l and d r ) are a scalar per pixel as the images are rectified.

The network was further improved by BID7 by introducing left-right consistency loss and refined encoder-decoder network.

Given the input left image, the network predicts both the left and right disparities simultaneously which enables constructing the left-right consistency.

This consistency restriction leads to more accurate result and less artifacts.

Also, fully differentiable backward bilinear sampling was used to reconstruct the left image which makes the model easier to optimize.

With better network architecture and better loss, Godard et al achieved the state of art result of unsupervised monodepth estimation only with rectified stereo pairs, and even outperforming supervised methods.

Their network architectures are shown in FIG1 .However, there still are unsatisfactory artifacts at the occlusion boundaries showing blurred ramps on the left side of image and of the occluders.

Even with the post-processing step which weighted sums the flipped disparity of the flipped input image and the disparity of the input image, the blurred ramps are still visible near the objects, especially for those near the cameras with higher disparities as illustrated in FIG0 .

Though common backward warping using bilinear sampler is fully differentiable which enables us to train the model end to end, some undesirable duplicates and artifacts are introduced during the warping process because of occlusions according to BID14 .

In FIG2 , we use SYTHIA dataset BID20 ), a synthesized virtual driving dataset to illustrate.

Though ground truth disparities are used for warping, there still exists obvious duplicates and even a huge black region on the left of the reconstructed image(Ĩ l ) because of the occlusion.

If those inevitable artifacts and duplicates are back propagated during the training process, unwanted high losses will be introduced forcing the network learn to blur in those regions, as the blurriness(disparity ramps) in the occluded regions will make the reconstructed image show stretched patterns (Fig 7) which are more similar to original ground truth image compared to duplicates and large black regions FIG2 ). .

We used a warping mask which is generated automatically from disparities to block the back-propagation of those artifacts.

The final output is shown in (e), where the white regions are masked.

In order to block the back propagation process of those warping induced artifacts, we designed an algorithm which can mask the occlusion region automatically.

In backward bilinear sampling process, a disparity map was used to sample from the source image.

The intuition here is that, if any pixel in the source image has never been sampled during the bilinear sampling process, this pixel should only be visible in the source image and should be masked when reconstructing this source image later on.

Thus, the mask methods takes, say, left disparity as input while generating the mask of reconstructed right image and vice versa.

The pseudo code is shown in Algorithm 1 and the mask schematic diagram is shown in Fig 4.

However, the mask method alone cannot guarantee the clearness since no constrain is added in masked region.

Specifically, though the masks block the back propagation of duplicates and artifacts induced by warping, they also block the further learning process of those regions.

Once the disparities start to blur, hardly can we correct the network back to clearness.

To solve this problem, we refined the network architecture and introduced a flip-over training scheme.

Though the mask blocks the process of further learning of the blurred regions, we find that the blurred side are definite which can be exploited to reactive the learning process.

For example, when the network is only trained on left images and takes corresponding right images as ground truth, the disparity ramps (blurred regions) will only appear on the left side of the occluders.

So, if we randomly flipped the input images horizontally, the disparity ramps will still appear on the left side.

When flipped the output disparities back for warping, the blurred regions will appear on the right side where no mask is added.

Examples are shown in the last column of Fig 7.

Thus, those blurred regions will make distortions when reconstructing images, and back propagate despite the 3 for x,y in grid(h,w) do DISPLAYFORM0 13 end 14 end 15 return mask l , mask r mask.

Also, because the flip-over scheme is performed randomly, any blurriness on the definite side will receive punishment on average which restrict the prediction to be clear.

It is worth to mention that flip-over scheme will not affect masking process and we still use masks to block the back propagation of duplicates.

The flip-over schematic diagram is shown in Fig. 4 .However, the predicted disparities of the right branch won't make any sense if we take the flipped left image as input and are totally mismatched with the ground truths as is shown in Fig 4.

As a result, we delete the predicting branch of the right disparity and add another encoder-decoder network which takes right images as input and predicts right disparities as is shown in FIG1 This doubled encoder-decoder network enables us to preform left-right consistency loss at the cost of doubling the training parameters.

However, it won't slow down the test speed since only one branch of encoder-decoder network is used when testing.

We also tried another network architecture with shared the encoder-decoder network which achieves comparable result as non-shared network while halves the training time.

More details can be found in 6.1

We use similar training loss as BID7 .

The losses are performed on four different scale and are finally summed as the total loss.

C = 4 s=1 C s .

For each scale, three different losses including appearance match loss(C ap ), disparity smoothness loss(C ds ) and LR consistency loss(C lr ) are performs as follows.

DISPLAYFORM0 Intuitively, appearance matching loss measures photometric error between the reconstructed image(Ĩ l ij )) and the ground truth image(I l ij ), which is defined as the weighted sum of L1 and SSIM shown as follows, DISPLAYFORM1 Disparity smoothness loss aims to force the smoothness of predicted disparities through Figure 4 : Schematic diagram.

The orange squares refer to an object such as pedestrians or vehicles.

The blue and black squares refer to blurred disparity region and masked region respectively.

Left: Naive masking process.

Warping mask(black) is overlap with blurred region(blue).

As a result, masks will block the learning of those region.

Right: Flip-over training scheme.

We flipped the left images(f (I l )) as input and flipped the output disparities(f (d(f (I l )))) back to reconstruct the image(Ĩ l ).

Different from the diagram shown in left, the blurred region will switch to the opposite side to avoid the mask.

As a result, losses will be introduced leading to clear boundaries.

We can also observed that the right branch (flipped predicted right disparity of flipped left image(f (d(I r )))) is totally mismatch with the ground truth right image, so we delete this branch and add another encoder-decoder branch as shown in FIG1 .

DISPLAYFORM2 Finally, L1 penalty is added to constrict the left-right consistency, which further reduce the artifacts, DISPLAYFORM3

Our network is based on BID7 and is implemented in Tensorflow (Abadi et al.) .

The network contains encoder network based on VGG16 or Resnet50 and decoder network with 7 upconvolutional layers, 7 merging layers and 4 disparity prediction layers.

With 7 skip connections, the network can better handles features at different scales.

Different from Godard et al, we modified the channel number of disparity prediction layers to predict only one disparity instead of two.

Also, we tuned the default hyper parameters α and α ds .

The rest are the same with Godard et al with α ap = 1, α lr = 1.

The learning rate λ remains 10 −4 for the first 30 epoch and halves every 10 epoch when trained for 50 epoch.

We also tried batch norm but might lead to unstable result.

Data augmentation is performed on the fly similar as Godard et al including flipping and color shifting.

We trained our model on rectified stereo image pairs in KITTI and Cityscapes dataset and evaluated mainly on KITTI split BID6 ), Eigen split BID1 ) and virtual-KITTI datasets BID4 ).

Also, we find the problem of KITTI 2015 evaluation such as sparsity and man-made defects which infects our evaluation result.

When testing on dense datasets like virtual-KITTI, the superiority becomes more obvious.

For comparison, we use the same split as BID7 high quality disparity image issued by official KITTI dataset.

It is worth mentioned that though these disparity images are of better quality than projected velodyne laser point and with CAD models inserted in cars, most part of ground truth are extremely sparse, especially in terms of occluded regions.

As the red boxes in the FIG3 shown, there is some blank space on the left side of image and occluded regions which is not involved in the evaluation.

Thus, those blank space just covers the shortcomings the disparity ramps on the occluded regions.

As a result, our result shows less superiority over Godard et al on KITTI stereo 2015 dataset.

As shown in TAB2 , we use the metrics from BID1 and D1-all metrics from KITTI.

Our model achieves comparable result with Godard et al when both trained on KITTI dataset only for 50 epoch, and superior result when both trained longer for 100 epoch.

BID6 ).

K means the model is only trained on KITTI dataset, while CS + K means the model is trained on Cityscapes dataset and then finetune on KITTI dataset.

Also, pp means post processing step, and more details about post processing can be found in 6.3.

For a fair comparison, we train the network proposed by Godard et al and our non-shared network both to 50 and 100 epoch, and find that our network has larger improvement and better performance than BID7 when trained longer to 100 epoch.

We guess that our network requires longer time to converge.

We choose different hyperparameters from Godard et al for our network.

Though it is unfair to evaluation on sparse KITTI dataset, our result still outperforms that of Godard et al.

Similar to Godard et al, we use the test split of 697 images proposed by BID1 .

Each image contains 3D points captured by velodyne laser which was used to generate ground truth depth.

We keep the same 22600 stereo pairs for training.

According to Godard et al, all evaluation results are done under the Garg crop BID5 ) except for Eigen results for a fair comparison.

We also present the uncropped result which our models' superiority become more obvious, because it will crop the disparity ramps on the left side for evaluation which boosting Godard et al's result.

Also, ground truths are captured by velodyne Lidar thus rather sparse which would reduce our superiority over Godard et al. The evaluation result is in TAB4 4.3 VIRTUAL-KITTITo prevent sparsity induced inaccurate result, we evaluate models on the Virtual-KITTI dataset BID6 ).

Virtual-KITTI dataset contains 50 monocular videos generated from five different virtual worlds in urban settings under different weather conditions with corresponding pixel-level ground truth depth.

With the same resolution, scenes and camera parameters as KITTI 2015, Virtual-KITTI dataset can be implement naively when testing.

Since KITTI 2015 dataset we used for training does not cover those weathers, only 2126 labbeled images without weather condi-Method Supervised Dataset Abs Rel Sq Rel RMSE RMSE log δ < 1.25 δ < 1.25 2 δ < 1.25 3 BID1 BID1 .

K is KITTI and CS is cityscapes dataset for training.

We use the evaluation result provided in BID7 .

For a fair comparison, we use the crop the same as BID5 except for BID1 and apply the same hyper-parameters as Godard et al on our model.

Besides, we set the maximun evaluation depth to 50 meters(cap 50m) in the second row which is the same as BID5 , while others remain 80 meters.

Also, we compared uncropped result with Godard et al, on which the superiority become more obvious.tion are used for evaluation.

The evaluation result is shown in TAB5 : Evaluation result on virtual-KITTI.

Once our ground truth become dense, our model out performance other models for its sharp and clear boundaries on predicted depth map.

Even we crop the black edges of Godard et al (the left 10% of the whole disparity map) in the second row, our model is still superior to BID7 (the state of art unsupervised monocular depth estimation using left and right information only).

We evaluate on 3 different methods: naive, crop the black edge, both crop the edge and set the maximun evaluation depth to 50 meters

In this work, we present an occlusion mask and filp-over training scheme to enable effective learning of object boundaries when using image warping.

With our new network, our model achieves state of art result using only stereo images.

Moreover, as warping based image reconstruction is commonly used in depth estimation problem, our method provides a solution to the fundamental difficulty of occluded areas introduced by image warping.

In the future, our method can be incorporated with more accurate network trained on trinocular data (temporal Stereo sequence) such as BID25 , BID17 and BID8 , which would further boost the accuracy.6 SUPPLEMENTARY MATERIALS

The shared weight network is similar to the non-shared weight network except for the shared weight.

The architecture is shown in 2.

Fortunately, the shared weight model naturally handled the problem.

Because the blurriness always occur in the occluded region, say, disparity ramps (burriness) in left disparity maps will only appear on the left side of the occluders, the network cannot distinguish which side the input image belongs to and which side to blur when trained on the left and right image simultaneously with shared weights.

Thus, any wrong blurriness (blurriness on the wrong side, say, right blurriness in the left disparity maps) will lead to punishment in raising reconstructed loss since no mask was adding on the other side (wrong side, say, right side).

As a result, the network achieves comparable result as non-shared network while halves the training time.

However, shared method is slightly inferior to non-shared network but still out performance Godard FIG0 , Godard et al used non-linear weight.

Because our model do not predict black edge, so we simply average them as our post-processing.

Here we use the network by BID7 to illustrate how flip-over scheme could solve the problem of back-propagate, so the predicted disparities are rather blurred.

In Fig 7 , from left to right: left disparity(d l ), reconstructed left image(Ĩ l ), corresponding mask(mask l ), ground truth(I l ), flipped disparity of flipped input imagef (d(f (I l ))).

To minimize the reconstruction error, the network predicts blurred result forcing the network to sample from the background, which leads to more similar stretched patterns instead of high loss duplicates as shown in FIG2 .

However, the blurred regions are aligned with the mask, which freezes the fine-tuning process of blurred regions.

To solve the problem, we randomly flip the colored image as input(Input = f (I l )), then flipped the output back as final disparities for warping(f (d(f (I l )))).

As shown in the last column, the blurred regions are no longer aligned with the masks which enables further learning process.

As the training process goes on, the boundaries will become clearer as shown in FIG0 .

We also use schematic diagram to illustrate in Fig. 4 Figure 7: Example of blurred results and flip-over scheme.

@highlight

This paper propose a mask method which solves the previous blurred results of unsupervised monocular depth estimation caused by occlusion
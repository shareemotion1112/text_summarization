This paper introduces a network architecture to solve the structure-from-motion (SfM) problem via feature-metric bundle adjustment (BA), which explicitly enforces multi-view geometry constraints in the form of feature-metric error.

The whole pipeline is differentiable, so that the network can learn suitable features that make the BA problem more tractable.

Furthermore, this work introduces a novel depth parameterization to recover dense per-pixel depth.

The network first generates several basis depth maps according to the input image, and optimizes the final depth as a linear combination of these basis depth maps via feature-metric BA.

The basis depth maps generator is also learned via end-to-end training.

The whole system nicely combines domain knowledge (i.e. hard-coded multi-view geometry constraints) and deep learning (i.e. feature learning and basis depth maps learning) to address the challenging dense SfM problem.

Experiments on large scale real data prove the success of the proposed method.

The Structure-from-Motion (SfM) problem has been extensively studied in the past a few decades.

Almost all conventional SfM algorithms BID46 BID39 BID16 BID13 jointly optimize scene structures and camera motion via the Bundle-Adjustment (BA) algorithm BID43 BID1 , which minimizes the geometric BID46 BID39 or photometric BID17 BID13 error through the Levenberg-Marquardt (LM) algorithm BID35 .

Some recent works BID44 attempt to solve SfM using deep learning techniques, but most of them do not enforce the geometric constraints between 3D structures and camera motion in their networks.

For example, in the recent work DeMoN BID44 , the scene depths and the camera motion are estimated by two individual sub-network branches.

This paper formulates BA as a differentiable layer, the BA-Layer, to bridge the gap between classic methods and recent deep learning based approaches.

To this end, we learn a feed-forward multilayer perceptron (MLP) to predict the damping factor in the LM algorithm, which makes all involved computation differentiable.

Furthermore, unlike conventional BA that minimizes geometric or photometric error, our BA-layer minimizes the distance between aligned CNN feature maps.

Our novel feature-metric BA takes CNN features of multiple images as inputs and optimizes for the scene structures and camera motion.

This feature-metric BA is desirable, because it has been observed by BID17 that the geometric BA does not exploit all image information, while the photometric BA is sensitive to moving objects, exposure or white balance changes, etc.

Most importantly, our BA-Layer can back-propagate loss from scene structures and camera motion to learn appropriate features that are most suitable for structure-from-motion and bundle adjustment.

In this way, our network hard-codes the multi-view geometry constraints in the BA-Layer and learns suitable feature representations from training data.

We strive to estimate a dense per-pixel depth, because dense depth is critical for many tasks such as object detection and robot navigation.

A major challenge in solving dense per-pixel depth is to find a compact parameterization.

Direct per-pixel depth is computational expensive, which makes the network training intractable.

So we train a network to generate a set of basis depth maps for an arbitrary input image and represent the result depth map as a linear combination of these basis 2 RELATED WORK Monocular Depth Estimation Networks Estimating depth from a monocular image is an ill-posed problem because an infinite number of possible scenes may have produced the same image.

Before the raise of deep learning based methods, some works predict depth from a single image based on MRF BID37 BID36 , semantic segmentation BID29 , or manually designed features BID27 .

BID15 propose a multi-scale approach for depth prediction with two CNNs, where a coarse-scale network first predicts the scene depth at the global level and then a fine-scale network will refine the local regions.

This approach was extended in BID14 to handle semantic segmentation and surface normal estimation as well.

Recently, BID30 propose to use ResNet BID24 based structure to predict depth, and BID47 construct multi-scale CRFs for depth prediction.

In comparison, we exploit monocular image depth estimation network for depth parameterization, which only produces a set of basis depth maps and the final result will be further improved through optimization.

Structure-from-Motion Networks Recently, some works exploit CNNs to resolve the SfM problem.

BID22 solve the camera motion by a network from a pair of images with known depth.

employ two CNNs for depth and camera motion estimation respectively, where both CNNs are trained jointly by minimizing the photometric loss in an unsupervised manner.

implement the direct method BID40 as a differentiable component to compute camera motion after scene depth is estimated by the method in .

In BID44 , the scene depth and the camera motion are predicted from optical flow features, which help to make it generalizing better to unseen data.

However, the scene depth and the camera motion are solved by two separate network branches, multi-view geometry constraints between depth and motion are not enforced.

Recently, propose to solve nonlinear least squares in two-view SfM using a LSTM-RNN BID26 as the optimizer.

Our method belongs to this category.

Unlike all previous works, we propose the BA-Layer to simultaneously predict the scene depth and the camera motion from CNN features, which explicitly enforces multi-view geometry constraints.

The hard-coded multi-view geometry constraints enable our method to reconstruct more than two images, while most deep learning methods can only handle two images.

Furthermore, we propose to minimize a feature-metric error instead of the photometric error in to enhance robustness.

Before introducing our BA-Net architecture, we revisit the classic BA to have a better understanding about where the difficulties are and why feature-metric BA and feature learning are desirable.

We only introduce the most relevant content and refer the readers to BID43 and BID1 for a comprehensive introduction.

Given images I = {I i |i = 1 · · · N i }, the geometric BA BID43 BID1 jointly optimizes camera poses T = {T i |i = 1 · · · N i } and 3D scene point coordinates P = {p j |j = 1 · · · N j } by minimizing the re-projection error: DISPLAYFORM0 where the geometric distance e g i,j (X ) = π(T i , p j ) − q i,j measures the difference between a projected scene point and its corresponding feature point.

The function π projects scene points to image space, q i,j = [x i,j , y i,j , 1] is the normalized homogeneous pixel coordinate, and DISPLAYFORM1 contains all the points' and the cameras' parameters.

The general strategy to minimize Equation (1) is the Levenberg-Marquardt (LM) BID35 BID32 algorithm.

At each iteration, the LM algorithm solves for an optimal update ∆X * to the solution by minimizing: DISPLAYFORM2 Here, DISPLAYFORM3 Ni,Nj (X )], and J(X ) is the Jacobian matrix of E(X ) respect to X , D(X ) is a non-negative diagonal matrix, typically the square root of the diagonal of the approximated Hessian J(X ) J(X ).

The non-negative value λ controls the regularization strength.

The special structure of J(X ) J(X ) motivates the use of Schur-Complement BID6 .This geometric BA with re-projection error is the golden standard for structure-from-motion in the last two decades, but with two main drawbacks:• Only image information conforming to the respective feature types, typically image corners, blobs, or line segments, is utilized.• Features have to be matched to each other, which often result in a lot of outliers.

Outlier rejection like RANSAC is necessary, which still cannot guarantee correct result.

These two difficulties motivate the recent development of direct methods BID17 BID13 which propose the photometric BA algorithm to eliminate feature matching and directly minimizes the photometric error (pixel intensity difference) of aligned pixels.

The photometric error is defined as: DISPLAYFORM4 where d j ∈ D = {d j |j = 1 · · · N j } is the depth of a pixel q j at the image I 1 , and d j · q j upgrade the pixel q j to its 3D coordinate.

Thus, the optimization parameter is DISPLAYFORM5 .

The direct methods have the advantages of using all pixels with sufficient gradient magnitude.

They have demonstrated superior performance, especially at less textured scenes.

However, these methods also have some drawbacks:• They are sensitive to initialization as demonstrated in BID33 and BID41 because the photometric error increases the non-convexity BID16 ).•

They are sensitive to camera exposure and white balance changes.

An automatic photometric calibration is required BID16 ).•

They are more sensitive to outliers such as moving objects.

To deal with the above challenges, we propose a feature-metric BA algorithm which estimates the same scene depth and camera motion parameters X as in photometric BA, but minimizes the feature-metric difference of aligned pixels: DISPLAYFORM0 where BID49 as the backbone network, a Basis Depth Maps Generator that generates a set of basis depth maps, a Feature Pyramid Constructor that constructs multi-scale feature maps, and a BA-Layer that optimizes both the depth map and the camera poses through a novel differentiable LM algorithm.

DISPLAYFORM1 We learn features suitable for SfM via back-propagation, instead of using pre-trained CNN features for image classification BID10 .

Therefore, it is crucial to design a differentiable optimization layer, our BA-Layer, to solve the optimization problem, so that the loss information can be back-propagated.

The BA-Layer predicts the camera poses T and the dense depth map D during forward pass and back-propagates the loss from T and D to the feature pyramids F for training.

As illustrated in FIG0 , our BA-Net receives multiple images and then feed them to the backbone DRN-54.

We use DRN-54 BID49 because it replaces max-pooling with convolution layers and generates smoother feature maps, which is desirable for BA optimization.

Note the original DRN is memory inefficient due to the high resolution feature maps after dilation convolutions.

We replace the dilation convolution with ordinary convolution with strides to address this issue.

After DRN-54, a feature pyramid is then constructed for each input image, which are the inputs for the BA-Layer.

At the same time, the basis depth maps generator generates multiple basis depth maps for the image I 1 , and the final depth map is represented as a linear combination of these basis depth maps.

Finally, the BA-Layer optimizes for the camera poses and the dense depth map jointly by minimizing the feature-metric error defined in Equation FORMULA6 , which makes the whole pipeline end-to-end trainable.

The feature pyramid learns suitable features for the BA-Layer.

Similar to the feature pyramid networks (FPN) for object detection BID31 , we exploit the inherent multi-scale hierarchy of deep convolutional networks to construct feature pyramids.

A top-down architecture with lateral connections is applied to propagate richer context information from coarser scales to finer scales.

Thus, our feature-metric BA will have a larger convergence radius.

As shown in Figure 2 (a), we construct a feature pyramid from the backbone DRN-54.

We denote the last residual blocks of conv1, conv2, conv3, conv4 in DRN-54 as {C 1 , C 2 , C 3 , C 4 }, with strides {1, 2, 4, 8} respectively.

We upsample a feature map C k+1 by a factor of 2 with bilinear interpolation and concatenate the upsampled feature map with C k in the next level.

This procedure is iterated until the finest level.

Finally, we apply a 3 × 3 convolution on the concatenated feature maps to reduce its dimensionality to 128 to balance the expressiveness and computational complexity, which leads to the final feature pyramid DISPLAYFORM0 We visualize some typical channels from the raw image I (i.e. the RGB channels), the pre-trained DRN-54 C 3 and our learned F 3 in Figure 2 (b).

It is evident that, after training with our BA-Layer, the feature pyramid becomes smoother and each channel correspondences to different regions in the image.

Note that our feature pyramids have higher resolution than FPN to facilitate precise alignment.

To have a better intuition about how much the BA optimization benefits from our learned features, we visualize different distances in Figure 3 .

We evaluate the distance between a pixel marked by a yellow cross in the top image in Figure 3 (a) and all pixels in a neighbourhood of its corresponding point in the bottom image of Figure 3 (a).

The distances evaluated from raw RGB values, pretrained feature C 3 , and our learned feature F 3 are visualized in (b), (c), and (d) respectively.

All distances are normalized to [0, 1] and visualized as heat maps.

The x-axis and y-axis are the offsets to the ground-truth corresponding point.

The RGB distance in (b) (i.e. e p in Equation FORMULA4 ) has no clear global minimum, which makes the photometric BA sensitive to initialization BID17 .

The distance measured by the pretrained feature C 3 has both global and local minimums.

Finally, the distance measured by our learned feature F 3 has a clear global minimum and smooth basin, which is helpful in gradient based optimization such as the LM algorithm.

After building feature pyramids for all images, we optimize camera poses and a dense depth map by minimizing the feature-metric error in Equation (4).

Following the conventional Bundle Adjustment principle, we optimize Equation (4) using the Levenberg-Marquardt (LM) algorithm.

However, the original LM algorithm is non-differentiable because of two difficulties:• The iterative computation terminates when a specified convergence threshold is reached.

This if-else based termination strategy makes the output solution X non-differentiable with respect to the input F (Domke, 2012).• In each iteration, it updates the damping factor λ based on the current value of the objective function.

It raises λ if a step fails to reduce the objective; otherwise it reduces λ.

This if-else decision also makes X non-differentiable with respect to F.When the solution X is non-differentiable with respect to F, feature learning by back-propagation becomes impossible.

The first difficulty has been studied in Domke (2012) and the author proposes to fix the number of iterations, which is refered as 'incomplete optimization'.

Besides making the optimization differentiable, this 'incomplete optimization' technique also reduces memory consumption because the number of iterations is usually fixed at a small value.

The second difficulty has never been studied.

Previous works mainly focus on gradient descent (Domke, 2012) or quadratic minimization BID3 BID38 .

In this section, we propose a simple yet effective approach to soften the if-else decision and yields a differentiable LM algorithm.

We send the current objective value to a MLP network to predict λ.

This technique not only makes the optimization differentiable, but also learns to predict a better damping factor λ, which helps the optimization to reach a better solution within limited iterations.

To start with, we illustrate a single iteration of the LM optimization as a diagram in Figure 4 by interpreting intermediate variables as network nodes.

During the forward pass, we compute the solution update ∆X from feature pyramids F and current solution X as the following steps:• We compute the feature-metric error • We then compute the Jacobian matrix J(X ), the Hessian matrix J(X ) J(X ) and its diagonal matrix D(X ); • To predict the damping factor λ, we use global average pooling to aggregate the aboslute value of E(X ) over all pixels for each feature channel, and get a 128D feature vector.

We then send it to a MLP sub-network to predict λ; • Finally, the update ∆X to the current solution is computed as a standard LM step: DISPLAYFORM0 DISPLAYFORM1 In this way, we can consider λ as an intermediate variable and denote each LM step as a function g about features pyramids F and the solution X from the previous iteration.

In other words, ∆X = g(X ; F).

Therefore, the solution after the k-th iteration is: DISPLAYFORM2 Here, • denotes parameters updating, which is addition for depth and SE(3) exponential mapping for camera poses.

Equation (6) is differentiable with respect to the feature pyramids F, which makes back-propagation possible through the whole pipeline for feature learning.

The MLP that predicts λ is also shown in Figure 4 .

We stack four fully-connected layers to predict λ from the input 128D vector.

We use ReLU as the activation function to guarantee λ is non-negative.

Following the photometric BA BID17 , we solve our feature-metric BA using a coarse-to-fine strategy with feature map warping at each iteration.

We apply the differentiable LM algorithm for 5 iterations at each pyramid level, leading to 15 iterations in total.

All the camera poses are initialized with identity rotation and zero translation, and the initialization of depth map will be introduced in Section 4.4.

Parameterizing a dense depth map by a per-pixel depth value is impractical under our formulation.

Firstly, it introduces too many parameters for optimization.

For example, an image of 320 × 240 pixels results in 76.8k parameters.

Secondly, in the beginning of training, many pixels will become invisible in the other views because of the poorly predicted depth or motion.

So little information can be back-propagated to improve the network, which makes training difficult.

To deal with these problems, we use the convolutional network for monocular image depth estimation as a compact parameterization, rather than using it as an initialization as in BID42 and BID48 .

We use a standard encoder-decoder architecture for monocular depth learning as in BID30 .

We use DRN-54 as the encoder to share the same backbone features with our feature pyramids.

For the decoder, we modify the last convolutional feature maps of BID30 to 128 channels and use these feature maps as the basis depth maps for optimization.

The final depth map is generated as the linear combination of these basis depth maps, which is: DISPLAYFORM0 Here, D is the h · w depth map that contains depth values for all pixels, B is a 128 × h · w matrix, representing 128 basis depth maps generated from network, w is the linear combination weights of these basis depth maps.

The w will be optimized in our BA-Layer.

The ReLU activation function guarantees the final depth is non-negative.

Once B is generated from the network, we fix B and use w as a compact depth parameterization in BA optimization, and the feature-metric distance becomes: DISPLAYFORM1 where B[j] is the j-th column of B, and ReLU(w B [j] ) is the corresponding depth of q j .

To further speedup convergence, we learn the initial weight w 0 as a 1D convolution filter for an arbitrary image, i.e. D 0 = ReLU(w 0 B).

The B of various images are visualized in the appendix.

The BA-Net learns the feature pyramid, the damping factor predictor, and the basis depth maps generator in a supervised manner.

We apply the following commonly used loss for training, though more sophisticated ones might be designed.

The camera rotation loss is the distance between rotation quaternion vectors L rotation = q − q * .

Similarly, translation loss is the Euclidean distance between prediction and groundtruth in metric scale, L translation = t − t * .

For each dense depth map we applies the berHu Loss BID51 as in BID30 .We initialize the back-bone network from DRN-54 BID49 , and the other components are trained with ADAM (Kingma & Ba, 2015) from scratch with initial learning rate 0.001, and the learning rate is divided by two when we observe plateaus from the Tensorboard interface.

ScanNet ScanNet BID11 ) is a large-scale indoor dataset with 1,513 sequences in 706 different scenes.

Camera poses and depth maps are not perfect, because they are estimated via BundleFusion BID12 .

The metric scale is known in all data from ScanNet, because the data are recorded with a depth camera which returns absolute depth values.

To sample image pairs for training, we apply a simple filtering process.

We first filter out pairs with a large photo-consistency error, to avoid image pairs with large pose or depth error.

We also filter out image pairs, if less than 50% of the pixels from one image are visible in the other image.

In addition, we also discard a pair if their roundness score BID4 ) is less than 0.001, which avoids pairs with too narrow baselines.

We split the whole dataset into the training and the testing sets.

The training set contains the first 1,413 sequences and the testing set contains the rest 100 sequences.

We sample 547,991 training pairs and 2,000 testing pairs from the training and testing sequences respectively.

KITTI KITTI BID20 ) is a widely used benchmark dataset collected by car-mounted cameras and a LIDAR sensor on streets.

It contains 61 scenes belonging to the "city", "residential", or "road" categories.

BID15 select 28 scenes for testing and 28 scenes from the remaining for training.

We use the same data split, to make a fair comparison with previous methods.

Since ground truth pose is unavailable from the raw KITTI dataset, we compute camera poses by LibVISO2 BID19 and take them as ground truth after discarding poses with large errors.

ScanNet To evaluate the results' quality, we use the depth error metrics suggested in BID14 , where RMSE (linear, log, and log, scale inv.) measure the RMSE of the raw, the logarithmical, and aligned logarithmical depth values, while the other two metrics measure the mean of the ratios that divide the absolute and square error by groundtruth depth..

The errors in camera Table 1 : Quantitative comparisons with DeMoN and classic BA.

The superindex * denotes that the model is trained on the trainning set described in BID44 .

poses are measured by the rotation error (the angle between the ground truth and the estimated camera rotations), the translation direction error (the angle between the ground truth and estimated camera translation directions) and the absolute position error (the distance between the ground truth and the estimated camera translation vectors).In Table 1 , we compare our method with DeMoN BID44 and the conventional photometric and geometric BA.

Note that we cannot get DeMoN trained on the ScanNet.

For fair comparison, we train our network on the same training data as DeMoN and test both networks on our testing data 1 .

We also show the results of our network trained on ScanNet.

Our BA-Net consistently performs better than DeMoN no matter which training data is used.

Since DeMoN does not recover the absolute scale, we align its depth map with the groundtruth to recover its metric scale for evaluation.

We further compare with conventional geometric BID34 and photometric BID17 BA.

Again, our method produces better results.

The geometric BA works poorly here, because feature matching is difficult in indoor scenes.

Even the RANSAC process cannot get rid of all outliers.

While for photometirc BA, the highly non-convex objective function is difficult to optimize as described in Section 3.KITTI We use the same metrics as the comparisons on ScanNet for depth evaluation.

To evaluate the camera poses, we follow to use the Absolute Trajectory Error (ATE), which measures the Euclidean differences between two trajectories BID40 , on the 9th and 10th sequences from the KITTI odometry data.

In this experiment, we create short sequences of 5 frames by first computing 5 two-view reconstructions from our BA-Net and then align the two-view reconstructions in the coordinate system anchored at the first frame.

minimize the photometric error.

Ours Wang et al. (2018) BID21 BID15 Table 2 : Quantitative comparisons on KITTI with supervised BID15 and unsupervised BID21 methods.

Table 2 summarizes our results on KITTI.

Our method outperforms the supervised methods BID15 as well as recent unsupervised methods BID21 .

Our method also achieves more accurate camera trajectories than and .

We believe this is due to our feature-metric BA with features learned specifically for SfM problem, which makes the objective function closer to convex and easier to optimize as discussed in Section 4.2.

In comparison, and minimize the photometric error.

More comparison with DeMoN, ablation studies, and multi-view SfM (up to 5 views) are reported in the appendix due to page limit.

This paper presents the BA-Net, a network that explicitly enforces multi-view geometry constraints in terms of feature-metric error.

It optimizes scene depths and camera motion jointly via feature-metric bundle adjustment.

The whole pipeline is differentiable and thus end-to-end trainable, such that the features are learned from data to facilitate structure-from-motion.

The dense depth is parameterized as a linear combination of several basis depth maps generated from the network.

Our BA-Net nicely combines domain knowledge (hard-coded multi-view geometry constraint) with deep learning (learned feature representation and basis depth maps generator).

It outperforms conventional BA and recent deep learning based methods.

DISPLAYFORM0

(a) Architecture of the DRN-54 backbone FIG4 illustrates the detailed network architectures for the backbone DRN-54 and the depth basis generator.

The architecture of the feature pyramid has been provided in Figure 2(a) .

We modify the dilated convolution of the original DRN-54 to convolution with strides and discard the conv7 and conv8 as shown in FIG4 (a).

C 1 to C 6 are layers with {1,2,4,8,16,32} strides and {16,32,256,512,1024,2048} channels, where C 1 and C 2 are basic convolution layers, while C 3 to C 6 are standerd bottleneck blocks as in ResNet BID24 .

Figure 5(b) visualizes our depth basis generator which adopts the up-projection structure proposed in BID30 .

The depth basis generator is a stander decoder that takes the output of C 6 as input and stacks five up-projection blocks to generate 128 basis depth maps, and each of the basis depth maps is half the resolution of the input image.

The up-projection block is shown on the right of FIG4 (b) which upsample the input by 2× and then apply convolutions with projection connection.

Evaluation Time To evaluate the running time of our method, we use the Tensorflow profiler tool to retrieve the time in ms for all network nodes and then summarize the results corresponding to each component in our pipeline.

As shown in TAB5 , our method takes 95.21 ms to reconstruct two 320 × 240 images, which is slightly faster than DeMoN that takes 110 ms for two 256 × 192 images.

The current computation bottleneck is the BA-Layer which contains a large amount of matrix operations and can be further speeded up by direct CUDA implementation.

Since we explicitly hard-code the multi-view geometry constraints in the BA-Layer, it is possible to share the backbone DRN-54 with other high-level vision tasks, such as semantic segmentation and object detection, to maximize reuse of network structures and minimize extra computation cost.

TAB6 , the pre-trained features (i.e. w/o Feature Learning) produce larger error.

This proves the discussion in Section 4.2.Bundle Adjustment Optimization vs SE(3) Pose Estimation Our BA-Layer optimizes depth and camera poses jointly.

We compare it to the SE(3) camera pose estimation with fixed depth map (e.g. the initialized depth D 0 in Section 4.4), and similar strategy is adopted in .

To make a fair comparison, we also use our learned feature pyramids for the SE(3) camera pose estimation.

As shown in TAB6 , without BA optimization (i.e. w/o Joint Optimization), both the depth maps and camera poses are worse, because the errors in the depth estimation will degrades the camera pose estimation.

Differentiable Levenberg-Marquardt vs Gauss-Newton To make the whole pipeline end-to-end trainable, we makes the Levenberg-Marquardt algorithm differentiable by learning the damping factor from the network.

We first compare our method against vanilla Gauss-Newton without damping factor λ (i.e. λ = 0).

Since the objective function of feature-metric BA is non-convex, the Hessian matrix J(X ) J(X ) might not be positive definite, which makes the matrix inversion by Cholesky decomposition fail.

To deal with this problem, we use QR decomposition instead for training with Gauss-Newton.

As shown in TAB6 , the Gauss-Newton algorithm (i.e. w/o λ) generates much larger error, because the BA optimization is non-convex and the Gauss-Newton algorithm has no guaranteed convergence unless the initial solution is sufficiently close to the optimal BID35 .

This comparison reveals that, similar to conventional BA, our differnetiable Levenberg-Marquardt algorithm is superior than the Gauss-Newton algorithm for feature-metric BA.

Predicted vs Constant λ Another way to make the Levenberg-Marquardt algorithm differentiable is to fix the λ during the iterations.

We compare with this strategy.

As shown in FIG6 (a), increasing λ makes the both rotation and translation error decreases, until λ = 0.5, and then increases.

The reason is that a small λ makes the algorithm close to the Gauss-Newton algorithm, which has convergence issues.

A large λ leads to a small update at each iteration, which makes it difficult to reach a good solution within limited iterations.

While in FIG6 (b), increasing λ always makes depth errors decrease, probably because a larger λ leads to a small update and makes the final depth close to the initialed depth, which is better than the optimized one with small constant λ.

Using constant λ value consistently generates worse results than using a predicted λ from the MLP network, because there is no optimal λ for all data and it should be adapted to different data and different iterations.

We draw the errors of our method in FIG6 (a) and FIG6 (b) as the flat dash lines for a reference.

APPENDIX C: EVALUATION ON DEMON DATASET Table 5 summarizes our results on the DeMoN dataset.

For a comparison, we also cite the results from DeMoN BID44 and the most recent work LS-Net .

We further cite the results from some conventional approaches as reported in DeMoN, indicated as Oracle, SIFT, FF, and Matlab respectively.

Here, Oracle uses ground truth camera poses to solve the multi-view stereo by SGM BID25 , while SIFT, FF, and Matlab further use sparse features, optical flow, and KLT tracking respectively for feature correspondence to solve camera poses by the 8-pt algorithm BID23 Table 5 : Quantitative comparisons on the DeMoN dataset.

Our method consistently outperforms DeMoN BID44 at both camera motion and scene depth, except on the 'Scenes11' data, because we enforce multi-view geometry constraint in the BA-Layer.

Our results are poorer on the 'Scene11' dataset, because the images there are synthesized with random objects from the ShapeNet BID8 without physically correct scale.

This setting is inconsistent with real data and makes it harder for our method to learn the basis depth map generator.

When compared with LS-Net , our method achieves similar accuracy on camera poses but better scene depth.

It proves our feature-metric BA with learned feature is superior than the photometric BA in the LS-Net.

Our method can be easily extended to reconstruct multiple images.

We evaluate our method in the multi-view setting on the ScanNet BID11 dataset.

To sample multi-view images for training, we randomly select two-view image pairs that shares a common image to construct N -view sequences.

Due to the limited GPU memory (12G), we limit N to 5.As shown in the Table 6 , the accuracy is consistently improved when more views are included, which demonstrates the strength of the multi-view geometry constraints.

Instead, most existing deep learning approaches can only handle two views at a time, which is sub-optimal as known in structure-from-motion literature.

Table 6 : Quantitative comparisons on multi-view reconstruction on ScanNet.

We compare our method with CodeSLAM which adopts similar idea for depth parameterization.

But the difference is that CodeSLAM learns the conditioned depth auto-encoder separately and uses the depth codes in a standalone photometric BA component, while our method learns the feature pyramid and basis depth maps generator through feature-metric BA end-to-end.

Since there is no public code for CodeSLAM, we directly cite the results from their paper.

2 To get the trajectory on the EuroC MH02 sequence of our method, we select one frame every four frames and concatenate the reconstructed groups that contains every five selected frames.

Then we use the same evaluation metrics as in CodeSLAM, which measures the translation errors correspond to different traveled distances.

As shown in FIG7 , our method outperforms CodeSLAM.

Our median error is less than the half of CodeSLAM's error, i.e. CodeSLAM exhibits an error of roughly 1 m for a traveled distance of 9 m, while our method's error is about 0.4 m. This comparison demonstrates the superiority of end-to-end learning with feature pyramid and feature-metric BA over learning depth parameterization only.

In FIG8 , we visualize four typical basis depth maps as heat maps for each of the four images.

An interesting observation is that one basis depth map has higher responses on close objects while another oppositely has higher responses to the far background.

Some other basis depth maps have smoothly varying responses and correspond to the layouts of scenes.

This observation reveals that our learned basis depth maps have captured the latent structures of scenes.

Finally, we show some qualitative comparison with the previous methods.

Figure 9 shows the recovered depth map by our method and DeMoN BID44 on the ScanNet data.

As we can see from the regions highlighted with a red circle, our method recovers more shape details.

This is consistent with the quantitative results in Table 1 .

FIG0 shows the recovered depth maps by our method, , and BID21 respectively.

Similarly, we observe more shape details in our results, as reflected in the quantitative results in Table 2 .

FORMULA0 and BID21 .

<|TLDR|>

@highlight

This paper introduces a network architecture to solve the structure-from-motion (SfM) problem via feature bundle adjustment (BA)
Multiview stereo aims to reconstruct scene depth from images acquired by a camera under arbitrary motion.

Recent methods address this problem through deep learning, which can utilize semantic cues to deal with challenges such as textureless and reflective regions.

In this paper, we present a convolutional neural network called DPSNet (Deep Plane Sweep Network) whose design is inspired by best practices of traditional geometry-based approaches.

Rather than directly estimating depth and/or optical flow correspondence from image pairs as done in many previous deep learning methods, DPSNet takes a plane sweep approach that involves building a cost volume from deep features using the plane sweep algorithm, regularizing the cost volume via a context-aware cost aggregation, and regressing the depth map from the cost volume.

The cost volume is constructed using a differentiable warping process that allows for end-to-end training of the network.

Through the effective incorporation of conventional multiview stereo concepts within a deep learning framework, DPSNet achieves state-of-the-art reconstruction results on a variety of challenging datasets.

Various image understanding tasks, such as semantic segmentation BID3 and human pose/action recognition BID29 BID33 , have been shown to benefit from 3D scene information.

A common approach to reconstructing 3D geometry is by multiview stereo, which infers depth based on point correspondences among a set of unstructured images BID10 ; .

To solve for these correspondences, conventional techniques employ photometric consistency constraints on local image patches.

Such photo-consistency constraints, though effective in many instances, can be unreliable in scenes containing textureless and reflective regions.

Recently, convolutional neural networks (CNNs) have demonstrated some capacity to address this issue by leveraging semantic information inferred from the scene.

The most promising of these methods employ a traditional stereo matching pipeline, which involves computation of matching cost volumes, cost aggregation, and disparity estimation BID5 ; BID19 ; BID14 ; BID0 .

Some are designed for binocular stereo BID31 ; BID19 ; BID0 and cannot readily be extended to multiple views.

The CNN-based techniques for multiview processing BID5 ; BID14 both follow the plane-sweep approach, but require plane-sweep volumes as input to their networks.

As a result, they are not end-to-end systems that can be trained from input images to disparity maps.

In this paper, we present Deep Plane Sweep Network (DPSNet), an end-to-end CNN framework for robust multiview stereo.

In contrast to previous methods that employ the plane-sweep approach BID14 ; BID5 , DPSNet fully models the plane-sweep process, including construction of plane-sweep cost volumes, within the network.

This is made possible through the use of a differentiable warping module inspired by spatial transformer networks BID17 to build the cost volumes.

With the proposed network, plane-sweep stereo can be learned in an end-to-end fashion.

Additionally, we introduce a cost aggregation module based on local cost-volume filtering BID26 for context-aware refinement of each cost slice.

Through this cost-volume regularization, the effects of unreliable matches scattered within the cost volume are reduced considerably.

With this end-to-end network for plane-sweep stereo and the proposed cost aggregation, we obtain state-of-the-art results over several standard datasets.

Ablation studies indicate that each of these technical contributions leads to appreciable improvements in reconstruction accuracy.

CNN-based depth estimation has been studied for stereo matching, depth from single images, and multiview stereo.

Recent work in these areas are briefly reviewed in the following.

Stereo matching Methods for stereo matching address the particular case of depth estimation where the input is a pair of rectified images captured by a stereo rig.

Various network structures have been introduced for this problem.

BID38 present a Siamese network structure to compute matching costs based on the similarity of two image patches.

The estimated initial depth is then refined by traditional cost aggregation and refinement as post-processing.

BID24 directly stack several convolution and deconvolution layers upon the matching costs and train the network to minimize the distance between the estimates and ground truth.

BID1 propose a CNN that estimates initial disparity and then refines it using both prior and posterior feature consistency in an end-to-end manner.

BID19 leverage geometric knowledge in building a cost volume from deep feature representations.

It also enables learning of contextual information in a 3D volume and regresses disparity in an end-to-end manner.

BID0 introduce a pyramid pooling module for incorporating global contextual information into image features and a stacked hourglass 3D CNN to extend the regional support of contextual information.

Depth from single images Similar to these stereo matching approaches, single-image methods extract CNN features to infer scene depths and perform refinements to increase depth accuracy.

The first of these methods was introduced by BID4 , which demonstrated that CNN features could be utilized for depth inference.

Later, BID22 combined a superpixel-based conditional random field (CRF) to a CNN to improve the quality of depth estimates from single images.

To facilitate training, recent studies ; BID23 ; BID36 present an end-to-end learning pipeline that utilizes the task of view synthesis as supervision for single-view depth and camera pose estimation.

These systems consist of a depth network and a pose estimation network which simultaneously train on sequential images with a loss computed from images warped to nearby views using the estimated depth.

View synthesis has similarly been used as supervision by warping between stereo image pairs BID7 ; BID8 .

In contrast to these single-image works which employ warping as a component of view synthesis for self-supervised learning, our network computes warps with respect to multiple depth planes to produce plane-sweep cost volumes both for training and at test time.

The cost volumes undergo further processing in the form of cost aggregation and regularization to improve the robustness of depth estimates.

Multi-view stereo In multi-view stereo, depth is inferred from multiple input images acquired from arbitrary viewpoints.

To solve this problem, some methods recover camera motion between the unstructured images but are designed to handle only two views BID31 ; .

The DeMoN system BID31 consists of encoder-decoder networks for optical flow, depth/motion estimation, and depth refinement.

By alternating between estimating optical flow and depth/motion, the network is forced to use both images in estimating depth, rather than resorting to single-image inference.

perform monocular visual odometry in an unsupervised manner.

In the training step, the use of stereo images with extrinsic parameters allows 3D depth estimation to be estimated with metric scale.

Among networks that can handle an arbitrary number of views, camera parameters are assumed to be known or estimated by conventional geometric methods.

BID18 introduce an endto-end learning framework based on a viewpoint-dependent voxel representation which implicitly encodes images and camera parameters.

The voxel representation restricts the scene resolution that can be processed in practice due to limitations in GPU memory.

BID16 formulate a geometric relationship between optical flow and depth to refine the estimated scene geometry, but is designed for image sequences with a very small baseline, i.e., an image burst from a handheld camera.

BID14 compute a set of plane-sweep volumes using calibrated pose data as input for the network, which then predicts an initial depth feature using an encoder-decoder network.

In the depth prediction step, they concatenate a reference image feature to the decoder input as an intra-feature aggregation, and cost volumes from each of the input images are aggregated by max-pooling to gather information for the multiview matching.

Its estimated depth map is refined using a conventional CRF.

By contrast, our proposed DPSNet is developed to be trained end-to-end from input images to the depth map.

Moreover, it leverages conventional multiview stereo concepts by incorporating context-aware cost aggregation.

Finally, we would like to refer the reader to the concurrent work by BID35 that also adopts differential warping to construct a multi-scale cost volume, then refined an initial depth map guided by a reference image feature.

Our work is independent of this concurrent effort.

Moreover, we make distinct contributions: (1) We focus on dense depth estimation for a reference image in an end-to-end learning manner, different from BID35 which reconstructs the full 3D of objects.

(2) Our cost volume is constructed by concatenating input feature maps, which enables inference of accurate depth maps even with only two-view matching.

(3) Our work refines every cost slice by applying context features of a reference image, which is beneficial for alleviating coarsely scattered unreliable matches such as for large textureless regions.

Our Deep Plane Sweep Network (DPSNet) is inspired by traditional multiview stereo practices for dense depth estimation and consists of four parts: feature extraction, cost volume generation, cost aggregation and depth map regression.

The overall framework is shown in FIG1 .

We first pass a reference image and target images through seven convolutional layers (3 × 3 filters except for the first layer, which has a 7 × 7 filter) to encode them, and extract hierarchical contextual information from these images using a spatial pyramid pooling (SPP) module BID12 with four fixed-size average pooling blocks (16 × 16, 8 × 8, 4 × 4, 2 × 2).

The multi-scale features extracted by SPP have been shown to be effective in many visual perception tasks such as visual recognition BID12 , scene parsing BID39 and stereo matching BID14 .

After upsampling the hierarchical contextual information to the same size as the original feature map, we concatenate all the feature maps and pass them through 2D convolutional layers.

This process yields 32-channel feature representations for all the input images, which are next used in building cost volumes.

We propose to generate cost volumes for the multiview images by adopting traditional plane sweep stereo Collins ( images for each pixel.

In a similar manner to traditional plane sweep stereo, we construct a cost volume from an input image pair.

To reduce the effects of image noise, multiple images can be utilized by averaging cost volumes for other pairs.

For this cost volume generation network, we first set the number of virtual planes perpendicular to the z-axis of the reference viewpoint [0, 0, 1] and uniformly sample them in the inverse-depth space as follows: DISPLAYFORM0 where L is the total number of depth labels and d min is the minimum scene depth as specified by the user.

Then, we warp all the paired features F i , (i = 1, .., N ), where i is an index of viewpoints and N is the total number of input views, into the coordinates of the reference feature (of size W idth × Height × CHannel) using pre-computed intrinsics K and extrinsic parameters consisting of a rotation matrix R i and a translation matrix t i of the i th camera: DISPLAYFORM1 where u,ũ l are the homogeneous coordinates of a pixel in the reference view and the projected coordinates onto the paired view, respectively.

F il (u) denotes the warped features of the paired image through the l th virtual plane.

Unlike the traditional plane sweeping method which utilizes a distance metric, we use a concatenation of features in learning a representation and carry this through to the cost volume as proposed in BID19 .

We obtain a 4D volume (W ×H ×2CH ×L) by concatenating the reference image features and the warped image features for all of the depth labels.

In Eq. (2), we assume that all images are captured by the same camera, but it can be directly extended to images with different intrinsics.

For the warping process, we use a spatial transformer network BID17 for all hypothesis planes, which does not require any learnable parameters.

In TAB4 , we find that concatenating features improves performance over the absolute difference of the features.

Given the 4D volume 1 , our DPSNet learns a cost volume generation of size W × H × L by using a series of 3D convolutions on the concatenated features.

All of the convolutional layers consist of 3 × 3 × 3 filters and residual blocks.

In the training step, we only use one paired image (while the other is the reference image) to obtain the cost volume.

In the testing step, we can use any number of paired images (N ≥ 1) by averaging all of the cost volumes.

The key idea of cost aggregation BID26 is to regularize the noisy cost volume through edge-preserving filtering BID11 volume filtering, we introduce a context-aware cost aggregation method in our end-to-end learning process.

The context network takes each slice of the cost volume and the reference image features extracted from the previous step, and then outputs the refined cost slice.

We run the same process for all the cost slices.

The final cost volume is then obtained by adding the initial and residual volumes as shown in FIG2 .Here, we use dilated convolutions in the context network for cost aggregation to better exploit contextual information ; BID37 .

The context network consists of seven convolutional layers with 3 × 3 filters, where each layer has a different receptive field (1, 2, 4, 8, 16, 1, and 1).

We jointly learn all the parameters, including those of the context network.

All cost slices are processed with shared weights of the context network.

Then, we upsample the cost volume, whose size is equal to the feature size, to the original size of the images via bilinear interpolation.

We find that this leads to moderate performance improvement as shown in TAB4 .

We regress continuous depth values using the method proposed in BID19 .

The probability of each label l is calculated from the predicted cost c l via the softmax operation σ(·).

The predicted labell is computed as the sum of each label l weighted by its probability.

With the predicted label, the depth is calculated from the number of labels L and minimum scene depth d min as follows:d DISPLAYFORM0 We set L and d min to 64 and 0.5, respectively.

Let θ be the set of all the learnable parameters in our network, which includes feature extraction, cost volume generation and cost aggregation (plane sweep and depth regression have no learnable parameters).

Letd,d denote the predicted depth from the initial and refined cost volumes, respectively, and let d gt be the corresponding supervision signal.

The training loss is then formulated as DISPLAYFORM0 where | · | H denotes the Huber norm, referred to as SmoothL1Loss in PyTorch.

The weight value λ for depth from the initial cost volume is set to 0.7.

In the training procedure, we use image sequences, ground-truth depth maps for reference images, and the provided camera poses from public datasets, namely SUN3D, RGBD, and Scenes11 2 .

We train our model from scratch for 1200K iterations in total.

All models were trained end-to-end with the ADAM optimizer (β 1 = 0.9, β 2 = 0.999).

We use a batch size of 16 and set the learning rate to 2e−4 for all iterations.

The training is performed with a customized version of PyTorch on four NVIDIA 1080Ti GPUs, which usually takes four days.

A forward pass of the proposed network takes about 0.5 seconds for 2-view matching and an additional 0.25 seconds for every new frame matched (640 × 480 image resolution).

In our evaluations, we use common quantitative measures of depth quality: absolute relative error (Abs Rel), absolute relative inverse error (Abs R-Inv), absolute difference error (Abs diff), square relative error (Sq Rel), root mean square error and its log scale (RMSE and RMSE log) and inlier ratios (δ < 1.25 i where i ∈ {1, 2, 3}).

All are standard metrics used in a public benchmark suite 3 .For our comparisons, we choose state-of-the-art methods for traditional geometry-based multiview stereo (COLMAP) , depth from unstructured two-view stereo (DeMoN) BID31 and CNN-based multiview stereo (DeepMVS) BID14 .

We estimate the depth maps from two unstructured views using the test sets in MVS, SUN3D, RGBD and Scenes11, as done for DeMoN 4 .

The results are reported in Table 1 .

Our DPSNet provides the best performance on nearly all of the measures.

Of particular note, DPSNet accurately recovers scene depth in homogeneous regions as well as along object boundaries as exhibited in FIG3 .

DeMoN generally produces good depth estimates but often fails to reconstruct scene details such as the keyboard (third row) and fine structures (first, second and fourth rows).

By contrast, DPSNet estimates accurate depth maps at those regions because the differential feature warping penalizes inaccurate reconstructions, playing a role similar to the left-right consistency check that has been used in stereo matching BID7 .

The first and third rows of FIG3 exhibit problems of COLMAP and DeepMVS in handling textureless regions.

DPSNet instead produces accurate results, courtesy of the cost aggregation network.

For a more balanced comparison, we adopt measures used in BID14 as additional evaluation criteria: (1) completeness, which is the percentage of pixels whose errors are below a certain threshold.

(2) geometry error, taking the L1 distance between the estimated disparity and the ground truth.

(3) photometry error, which is the L1 distance between the reference image and warped image using the estimated disparity map.

The results for COLMAP, DeMoN and DeepMVS are directly reported from BID14 in TAB1 .

In this experiment, we use the ETH3D dataset on which all methods are not trained.

Following BID35 , we take 5 images with 1152 × 864 resolution and set 192 depth labels based on ground-truth depth to obtain optimal results for MVSNet.

For the DPSNet results, we use 4 views with 810 × 540 resolution and set 64 labels whose range is determined by the minimum depth values of the ground truth.

In TAB1 , our DPSNet shows the best performance overall among the all the comparison methods, except for filtered COLMAP.

Although filtered COLMAP achieves the best performance, its completeness is only 71% and its unfiltered version shows a significant performance drop in all error metrics.

On the other hand, our DPSNet with 100% completeness shows promising results on all measures.

We note that our DPSNet has a different purpose compared to COLMAP and MVSNet.

COLMAP and MVSNet are designed for full 3D reconstruction with an effective outlier rejection process, while DPSNet aims to estimate a dense depth map for a reference view.

An extensive ablation study was conducted to examine the effects of different components on DPSNet performance.

We summarize the results in TAB4 .Cost Volume Generation In TAB4 (a) and (e), we compare the use of cost volumes generated using the traditional absolute difference BID2 and using the concatenation of features from the reference image and warped image.

The absolute difference is widely used for depth label selection via a winner-take-all strategy.

However, we observe that feature concatenation provides better performance in our network than the absolute difference.

A possible reason is that the CNN may learn to extract 3D scene information from the tensor of stacked features.

The tensor is fed into the CNN to produce an effective feature for depth estimation, which is then passed through our cost aggregation network for the initial depth refinement.

Cost Aggregation For our cost aggregation sub-network, we compare DPSNet with and without it in TAB4 (e) and (b), respectively.

It is shown that including the proposed cost aggregation leads to significant performance improvements.

Examples of depth map refinement with the cost aggregation are displayed in FIG5 .Our cost aggregation is also compared to using a stacked hourglass to aggregate feature information along the depth dimension as well as the spatial dimensions as done recently for stereo matching BID0 .

Although the stacked hourglass is shown in For further analysis of cost aggregation, we display slices of 3D cost volumes after the softmax operation (in Eq. FORMULA2 ) that span depth labels and the rows of the images.

The cost slices in Figure 6 (c), (d) show that our feature-guided cost aggregation regularizes noisy cost slices while preserving edges well.

The cleaner cost profiles that ensue from the cost aggregation lead to clearer and edge-preserving depth regression results.

As mentioned in a recent study BID13 , a cost profile that gives confident estimates should have a single, distinct minimum (or maximum), while an ambiguous profile has multiple local minima or multiple adjacent labels with similar costs, making it hard to exactly localize the global minimum.

Based on two quantitative confidence measures BID13 on cost volumes in TAB5 , the proposed aggregation improves the reliability of the correct match corresponding to the minimum cost.

Depth Label Sampling In the plane sweep procedure, depth labels can be sampled in either the depth domain or the inverse depth domain, which provides denser sampling in areas closer to a camera.

TAB4 (d) and (e) show that uniform depth label sampling in the inverse depth domain produces more accurate depth maps in general.

We examine the performance of DPSNet with respect to the number of input images.

As displayed in FIG7 , a greater number of images yields better results, since cost volume noise is reduced through averaging over more images, and more viewpoints help to provide features from areas unseen in other views.

FIG7 shows that adding input views aids in distinguishing object boundaries.

Note that the performance improvement plateaus when seven or more images are used.

Rectified Stereo Pair CNNs-based stereo matching methods have similarity to DPSNet, but differ from it in that correspondences are obtained by shifting learned features in BID24 ; BID19 BID30 .

The purpose of this study is to show readers that not only descriptor shift but also plane sweeping can be applied to rectified stereo matching.

We apply DPSNet on the KITTI dataset, which provides rectified stereo pairs with a specific baseline.

As shown in FIG8 , although DPSNet is not designed to work on rectified stereo images, it produces reasonable results.

In particular, DPSNet fine-tuned on the KITTI dataset in TAB7 achieves performance similar to BID24 in terms of D1-all score, with 4.34% for all pixels and 4.05% for non-occluded pixels in the KITTI benchmark.

We expect that the depth accuracy would improve if we were to adopt rectified stereo pair-specific strategies, such as the feature consistency check in BID1 .

We developed a multiview stereo network whose design is inspired by best practices of traditional non-learning-based techniques.

The plane sweep algorithm is formulated as an end-to-end network via a differentiable construction of plane sweep cost volumes and by solving for depth as a multilabel classification problem.

Moreover, we propose a context-aware cost aggregation method that leads to improved depth regression without any post-processing.

With this incorporation of traditional multiview stereo schemes into a deep learning framework, state-of-the-art reconstruction results are achieved on a variety of datasets.

Directions exist for improving DPSNet.

One is to integrate semantic instance segmentation into the cost aggregation, similar to the segment-based cost aggregation method of BID25 .

Another direction is to improve depth prediction by employing viewpoint selection in constructing cost volumes BID6 , rather than by simply averaging the estimated cost volumes as currently done in DPSNet.

Lastly, the proposed network requires pre-calibrated intrinsic and extrinsic parameters for reconstruction.

Lifting this restriction by additionally estimating camera poses in an end-to-end learning framework is an important future challenge.

@highlight

A convolution neural network for multi-view stereo matching whose design is inspired by best practices of traditional geometry-based approaches
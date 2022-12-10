Stacked hourglass network has become an important model for Human pose estimation.

The estimation of human body posture depends on the global information of the keypoints type and the local information of the keypoints location.

The consistent processing of inputs and constraints makes it difficult to form differentiated and determined collaboration mechanisms for each stacked hourglass network.

In this paper, we propose a Multi-Scale Stacked Hourglass (MSSH) network to high-light the differentiation capabilities of each Hourglass network for human pose estimation.

The pre-processing network forms feature maps of different scales,and dispatch them to various locations of the stack hourglass network, where the small-scale features reach the front of stacked hourglass network, and large-scale features reach the rear of stacked hourglass network.

And a new loss function is proposed for multi-scale stacked hourglass network.

Different keypoints have different weight coefficients of loss function at different scales, and the keypoints weight coefficients are dynamically adjusted from the top-level hourglass network to the bottom-level hourglass network.

Experimental results show that the pro-posed method is competitive with respect to the comparison algorithm on MPII and LSP datasets.

Human pose estimation need locate the body keypoints (head, shoulder, elbow, wrist, knee, ankle, etc.) from the input image, and it is basic method for some advanced vision task BID4 , such as human motion recognition, human-computer interaction, and human reidentification et al. We focus on single-person pose estimation problems in a single RGB image.

Due to the high flexibility of the human body and limbs, diverse viewpoint, camera projection transformation and occlusion, it still is a difficult task to accurately determine body keypoints from a single image.

In recent years, the deep convolutional neural network (DCNN) has made significant progress in the human pose estimation; especially the stacked hourglass network BID7 has achieved good results and has attracted much attention.

The human pose estimation involves two kinds of information: the type and location of body keypoints.

The type of body keypoints needs to be determined in a larger receptive field, and the location of body keypoints needs to be based on the specific pixel position, which are respectively equivalent to global information and local information.

The hourglass network uses a convolution layer and a deconvolution layer to form an hourglass structure, and establish crossover channels between convolution and deconvolution layers on different scales.

Using the hourglass network for human pose estimation, the hourglass structure extracts global information through information compression, and the crossover channels compensates for local information loss in information compression.

The stacked hourglass network continuously improves the human pose estimation by enhancing the context constraints among body keypoints through the stacked structure.

The stacked hourglass network theoretically increases the stacked depth to expand the receptive field and form a stronger context constraints among body keypoints.

However, in practical applications, simply increasing the stacked depth is difficult to effectively improve the accuracy of human pose estimation.

The main reason is that the consistent processing of inputs and constraints makes it difficult to form differentiated and determined collaboration mechanisms for each stacked hourglass network, to make up the information loss caused by the functional consistency of hourglass networks.

Inspired by multi-scale information fusion from the single hourglass network, we propose a Multi-Scale Stacked Hourglass (MSSH) network.

A pre-processing network consisting primarily of residual networks is designed to generate multi-scale features, and features of each scale are sent to different stacked hourglass networks.

Each hourglass network has different inputs: the output of the pre-level hourglass network, and the received feature.

Small-scale feature input makes the hourglass network tend to focus on global information, such as the type and context constraints of body keypoints, and large-scale feature input makes the hourglass network more likely to focus on local information, such as the location of body keypoints.

The input of entire stacking hourglass network has changed from small-scale feature to large-scale feature, and it is a top-down route of body keypoints estimation.

The iterative relationship between the loss functions of different scales of the MSSH network is established, so that the pre-level detection results affect the weight of the next keypoint loss function, and the optimization process of network model training is controlled by adaptive weighted loss function.

The iterative relationship of the adjacent network loss function is established on the MSSH network, so that the weight of the loss function on pre-level hourglass network affects the weight of the loss function on the current hourglass network.

The optimization process of the model training is controlled by the adaptive weighted loss function.

The main contributions of this paper can be summarized as follows:• In a multi-scale stacked hourglass network, the pre-processing network generates features of different scales and dispatchs them to every hourglass network, where the small-scale features reaches the front of stacked hourglass network and the large-scale features reaches the rear of stacked hourglass network.

From global information to local information, each hourglass network can form a differentiated function, which is conducive to the formation of collaborative processing.• A new loss function of the MSSH network is proposed.

The weighting coefficient of the loss function in hourglass network is defined.

The weight coefficient and the loss function of the pre-level hourglass network are used to adjust the weight coefficient of the current hourglass network, and the convergence process of the model training is optimized by the adaptive weighted loss function.

The remainder of this paper is organized as follows.

Section 2 briefly reviews recent work on human pose estimation.

Section 3 details the structure of the MSSH network.

Section 4 describes the new loss function for MSSH network networks.

Section 5 describes the implementation details and experimental results.

Section 6 summarizes our paper.

DeepPose BID11 firstly introduce CNN to solve pose estimation problem, which proposes a cascade of CNN to deal with pose estimation.

Joint train approach BID10 attempts to predict the keypoints of heatmaps of using CNN and graphical models.

Most late work shows good performance by using a deep convolutional neural network to generate heatmaps of keypoints.

The Convolutional Pose Machines BID12 uses a sequential convolutional architecture to express context relationships and uses multiple scales to process input feature maps.

The hourglass network BID7 reuses bottom-up and top-down strategies and stacks up several hourglass modules to inference the keypoints of the human body.

On the basis of the hourglass network, In order to obtain better human body pose estimation results, researchers are more inclined to design more complex networks from multi-stage processing, the multi-scale feature and loss function.

For multi-stage processing, Multi-Context Attention network BID2 uses the stacked hourglass network to generate attention maps with different resolution, and use CRF to enhancement the association of adjacent regions in the attention map.

Pyramid structure increases the receptive field of the network through the complication of the building block, which enhance the deep convolutional neural networks using multiscale subsampling to learn the features of different resolutions.

For multi-scale feature, cascaded pyramid network BID1 suggests that GlobalNet predicts the keypoints on multi-scale feature map in the first-stage and makes RefineNet predict the online hard keypoints in the second phase.

The multi-scale structure-aware network BID5 improves the detection result using multi-scale supervision and regression by matching features across all scales in building block.

For loss function, deep consensus voting BID6 adds voting constraints to the loss function.

Recurrent human pose estimation BID0 uses multiple regression networks to generate multiple loss functions to optimize network.

A novel bone-based part representation Tang et al. FORMULA0 is proposed to avoid potentially large state spaces for higher-level parts through multi-scale loss function.

The proposed MSSH network draws on the concept of multi-scale, uses pre-processing networks to form feature maps of different scales, and assigns them to different locations in the stacked network, which is inspired by chain prediction BID3 .

The small size features reach the front of stacked hourglass network, and the large-scale features reach the rear of stacked hourglass network.

A new loss function is designed to dynamically adjust the keypoints weight coefficients from the top layer to the bottom layer, and it pay more attention to hard keypoints in multi-scale stacked hourglass networks.

In this paper, we propose a Multi-Scale Stacked Hourglass (MSSH) network to promote functional collaboration and relieve misleading caused by information loss for human pose estimation.

An overall framework is illustrated in FIG1 .

We adopt the stacked hourglass network as the basic structure of the MSSH network to process features across all scales and capture the various context relationships associated with the body.

The pre-processing network generates feature maps of different scales, and dispatch them to each hourglass network, where the small-scale features reach the front of the MSSH network and the large-scale features reach the rear of the MSSH network.

The input to each hourglass network consists of two parts, one is the dispatched feature of pre-processing network and another is the output of the pre-level hourglass network.

In addition, to further enhance the performance of our network, we propose the inception-resnet as illustrated in FIG3 to replace the original residual network as a building block for the hourglass network.

We first briefly review the structure of stacked hourglass network.

Then a detailed introduction of our pre-processing network and network structure enhancement is presented.

The hourglass network is motivated by capturing information contained in the images at different scales.

First, the convolution and pooling process are performed, and multiple downsampling operations are performed to obtain some features with lower resolution, thereby reducing computational complexity.

In order to increase the resolution of the image features, multiple upsampling is performed.

The upsampling operation increases the resolution of the image and is more capable of predicting the exact position of the object.

Through such a process, the network structure can obtain more context information by increasing the operation of the receptive field compared to other networks.

With intermediate supervision at the end of the hourglass network, the the type and location information of the body keypoints are integrated into the output feature maps.

The stacked hour- glass network serially connect multiple hourglass networks.

So the subsampling and upsampling are repeated for several times to construct the stacked hourglass network.

This means that the entire network has multiple bottom-up and top-down processes to capture information at different scales.

And the information of the body keypoints are continuously enhanced as the stack number increases.

Stacked hourglass network can fine-tune keypoints gradually.

The goal of the multi-scale pre-processing network is to generate different-scale feature maps.

The multi-scale pre-processing network help the stacked hourglass networks to form differentiate the determined collaboration mechanisms, and avoid information loss caused by the functional consistency of hourglass networks.

As shown in FIG1 , the pre-processing network is construct as a feature preprocessing module before stacked hourglass network.

The pre-processing network generates different scales feature maps.

The smallest scale feature map is sent to the first-level hourglass network, and the largest scale feature map is sent to the last-level hourglass network, and other features from the small to the large are successively dispatch to the hourglass network from front to rear.

To generate these multi-scale feature maps, it consists of multiple branches with different depths to form feature maps as shown in FIG2 .

The convolution layers on each branch are used to extract the features, and the max pooling layers are used to change the resolution of the input as well as to expanding the receptive field.

To enhance the performance of the hourglass network, the inception-resnet is used as the basic building block in each hourglass network.

As shown in FIG3 (a), inception-resnet-A consists of convolutional layers, batch norm layers and Relu units, with channel-wise concatenation and pixelwise additions.

The concatenation of two branches maintains different level of information, but the concatenated features across different channels need to be transformed and normalized by the subsequent convolutional layers.

The benefit of the convolutional layers with 1*1 kernels is that the input and output have the same resolution while the depth of channels can be flexible.

In addition, inception-resnet-A increases the receptive field of the unit structure by adding a small number of convolution layers, effectively learning the context relationships and improving the implicit space model without the gradient disappearing.

Compared with inception-resnet-A, inception-resnet-B uses a subsampling layer and upsampling layer to deepen the building block and further extract different levels of information on the feature maps, as shown in FIG3 (b).

The consistent processing of constraints makes it difficult to form differentiated collaboration mechanisms for each stacked hourglass network.

Motivated by recurrent human pose estimation that uses multiple regression networks to generate multiple loss functions, new loss function with adaptive weight coefficients is designed to pay more attention to hard keypoints, where different keypoints have different weight coefficients of loss function at different scales, and the key weight coefficients are dynamically adjusted from the small scale to the large scale.

If the loss function value of a keypoint in the pre-level hourglass network is large, the weight of loss functionin is increased corresponding keypoint on the current hourglass network.

If the loss function value of a keypoint in the pre-level hourglass network is small, the weight of loss functionin is decreased corresponding keypoint on the current hourglass network.

By dynamically adjusting the weight coefficient, the network gradually increases the focus on the hard keypoints.

In this paper, the method of BID10 is used to generate a 2-D Gaussian heatmap centered on the position of keypoints.

The 2-D Gaussian heatmap of the ith keypoint at the jth level is generated as DISPLAYFORM0 Where a is the amplitude of the Gaussian funciton, which is set to be +1, if the landmark is nonoccluded or set to be -1, if the landmark is occluded.

µ represents mean and σ represents standard deviation of the Gaussian function.

The MSE loss function is used on the heatmap of each hourglass network to obtain the loss function of the ith keypoint at the jth level, which is expressed as DISPLAYFORM1 Where I j,i (m, n) represents the predicted heatmap of the ith keypoint at the jth level, I j,i (m, n) represents the ground truth heatmap of the ith keypoint at the jth level.

According to the structure of MSSH network, it is unreasonable to add the same weight to the loss function of the keypoints directly at all scales, because the standard deviation of the heatmap on the small scale is large, and the standard deviation of the heatmap on the large scale is small, so we need to weight on the basis of the loss function.

The weighted loss function is expressed as DISPLAYFORM2 Where w j,i represents the weight coefficient of ith keypoint at jth level.

First, initialize the weight coefficient to ] , where N represents the total number of keypoints, and according to formula 3, the loss function of the keypoints at the first level is calculated.

DISPLAYFORM0 Based on the adaboost regression algorithm, the greater the error in this iteration, the greater the weight given by the classifier in the next iteration.

We use the following formula to assign the weight coefficients of the loss function for each keypoint at the (j+1)th level.

DISPLAYFORM1 Where the layer coefficient α j is calculated as DISPLAYFORM2 Where Z j in formula 4 is the normalization factor and is expressed as DISPLAYFORM3 It can be seen from formula 4 : if the detection effect of the ith keypoint is smaller, Loss j,i is smaller, then the weight w j+1,i is bigger, and the corresponding learner will focus on this keypoint.

Otherwise, the weight of the loss function is bigger, and w j+1,i is smaller and learner will reduce its interference with keypoints that worse for detection.

The overall calculation process of the weight coefficient is as follows: DISPLAYFORM4

Our overall structure uses a multi-scale stacked hourglass network with adaptive weight coefficients of loss function.

In the experimental section, the database, criteria and implementation details are first introduced.

Then, the influence of adaptive adjustment of weight coefficient on the convergence of the model training, the impact of pre-processing network on performance, and the comparison of hourglass network structure are discussed in detail.

At last, quantitative assessments are performed on baseline datasets, and their performance is analyzed and discussed 5.1 EXPERIMENTAL SETUP 5.1.1 DATASETS MPII dataset is a state of the art benchmark for evaluation of articulated human pose estimation.

The dataset includes around 25K images containing over 40K people with annotated body joints.

The images were systematically collected using an established taxonomy of every day human activities.

Overall the dataset covers 410 human activities and each image is provided with an activity label.

Each image was extracted from a YouTube video and provided with preceding and following unannotated frames.

In addition, for the test set we obtained richer annotations including body part occlusions and 3D torso and head orientations.

LSP dataset contains 2000 pose annotated images of mostly sports people gathered from Flickr using the tags shown above.

The images have been scaled such that the most prominent person is roughly 150 pixels in length.

Each image has been annotated with 14 joint locations.

Left and right joints are consistently labelled from a person-centric viewpoint.

Attributions and Flickr URLs for the original images can be found in the JPEG comment field of each image file.

PCP is a widely-used criterion for human pose estimation, which evaluates the localization accuracy of body parts.

It requires the estimated part end points must be within half of the part length from the ground truth part end points.

Some early work requires only the average of the endpoints of a part to be correct, rather than both endpoints.

Moreover, the early PCP implementation selects the best matched output without penalizing false positives.

PCK is similar to the Percentage of Detected Joints (PDJ) criterion which is to measure the detection rate of body joints, where a joint is considered to be detected if the distance between the detected joint and the true joint is less than a fraction of the torso diameter.

The only difference is that the torso diameter is replaced with the maximum side length of the external rectangle of ground truth body joints.

For full body images with extreme pose (especially when the torso becomes very small), the PCK may be more suitable to evaluate the accuracy of body part localization.

PCK can be calculated by equation 4.

DISPLAYFORM0 Where y i represents the Ground-Truth position of the ith keypoint andỹ i represents the predicted position of the ith keypoint.

y lhip denotes the position of the Ground-Truth of the keypoint of the shin, and y rsho denotes the position of the Ground-Truth of the keypoint of the shoulder.

The value of γ is between 0 ∼ 1.PCKh is the modified PCK measure that uses the matching threshold as 50% of the head segment length.

The input image is 256*256 cropped from a resized image according to the annotated body position and scale.

We randomly rotate and flip the images, perform random rescaling and color jittering to make the model more robust to scale and illumination changes.

Training data are augmented by scaling, rotation, flipping, and adding color noise.

All the models are trained using pyTorch.

We use RMSProp to optimize the network on a 1080 GPU with a batchsize of 4 for 220 epochs.

The learning rate is initialized as 2.5*10 −4 and is dropped by 150 at the 175th and the 200th epoch.

The Mean-Squared Error (MSE loss) was used in the experiment to compare predicted scoremaps with Ground-Truth scoremaps consisting of 2D Gaussians centered around the human joint position.

In this subsection, we validate the effectiveness of our network from various aspects: the adaptive adjustment of weight coefficient, the performance of pre-processing network and the comparison of hourglass network structure.

Since the testing annotations for MPII are not available to the public, the train is on a subset of training images and the evaluation is on a held-out validation set of around 3000 samples.

According to the experimental results, the most appropriate method is used to build our network.

We first trained the hourglass network as a baseline model with a PCKh score of 88.78% on the validation set.

Through the pre-processing network, dispatch the feature maps of different resolutions (32, 64, 128) into each hourglass network, and use the adaptive weighting function to adjust the proportional weight of each keypoint loss function to optimize the progressive relationship between the loss functions of different scales.

As the input size of the images increases, more location details of human keypoints are fed into the network resulting in a large performance improvement.

Additionally, the adaptive weighting function works better when the input size of the images is enlarged in 8 stages.

The experimental results show that the PCKh score of the model trained by the new method reaches 89.25%, which is 0.47% better than the original structure.

In order to enhance the hourglass network for more information, we propose unit structures inception-resnet-A, and inception-resnet-B compared with the residual network as the baseline building block.

The structure of inception-resnet-A and inception-resnet-B has been elaborated in Section 3.3 .

In this section we mainly evaluate the effects of three structures on the pose estimation.

Under the same conditions, the PCkh scores of inception-resnet-A, inception-resnet-B and residual network are respectively 89.91%, 82.95% and 89.19%.

By comparing inception-resnet-A with the baseline building block, it is found that in the absence of a gradient disappearing, the addition of a small number of convolution layers increases the receptive field of the unit structure, effectively learns the context relationships, and improves the implicit space model.

Comparing inception-resnet-A with inception-resnet-B, it is found that the use of subsampling in the cell structure destroys the structural consistency of the key feature map.

Therefore, inception-resnet-C does not apply to the estimation of human pose.

We need to design a pre-processing network to provide multi-scale feature maps for MSSH network.

Because the quality of the feature map output by the pre-processing network directly affects the subsequent detection results, the design of the pre-processing network is crucial.

Borrowed the idea of the FPN network and gradually reduced the feature map through the inception-restnet building block.

As a comparative test, we used the addition of horizontal connections as a criterion to study the optimal pre-processing network that fits MSSH network and two networks is shown in FIG2 and 5 .

By testing on the MPII verification set, the results of the structure with horizontal connection and the structure without horizontal connection are 88.66% and 89.91% respectively.

Therefore, the feature map output by the pre-processing network needs to possess more local information instead of more advanced semantic information.

As shown in FIG1 , in stacked hourglass network, the input of the hourglass network stack with the dispatched feature of the pre-processing network and the output of the pre-level hourglass network, so each hourglass networks is able to access new information.

Therefore, the way of information fusion of these two parts is particularly important.

In the benchmark hourglass network, we used pixel-wise add as the way of information fusion.

Using concatenation as a comparison test to combine features generated from two pipelines, which is similar to inception models.

Results show that pixel-wise addition has the better performance with an accuracy improvement of 0.66%, which the pixel-wise add method is 89.91% and the concatenate method is 89.36%.

Therefore, we ended up using pixel-wise add.

We use PCKh@0.5 on the MPII test set, use PCK@0.2 and PCP@0.5 on the LSP dataset.

The comparisons of our method and state-of-the-art methods are shown in the Tabel 1,2,3.

Specifically, on the MPII test set, our method achieves 0.6% and 0.8% improvements on elbow and ankle, where ankle is considered as one of the most challenging parts to be detected.

In this work, we have proposed to dispatch the multi-scale feature maps from pre-processing network to each stacked hourglass network, which can be potentially used to aid other deep neural network in training tasks.

With adaptive weight loss function, it increases the weight coefficient value of the hard keypoints and optimize the convergence performance, which can be applied to similar multi-stage training loss function for optimization convergence.

The effectiveness of the proposed structure and loss function is evaluated on two widely used benchmarks.

Later, we hope to explore the extended test of the method under the condition of complex loss function constraints.

The figure below shows the training curve for the hourglass network and the MSSH network.

As can be seen from the figure, the MSSH network is easier to converge at the beginning and has a higher accuracy in the final stage.

FIG7 shows the detection results on the MPII validation set and the LSP dataset when the body joints are not twisted and the keypoints are not occluded.

FIG8 shows the detection results on the MPII validation set and the LSP dataset when the body joints are severely twisted and he keypoints are occluded.

FIG9 shows the detection results on the MPII validation set when the human body is occluded or the body joints are twisted.

<|TLDR|>

@highlight

Differentiated inputs cause functional differentiation of the network, and the interaction of loss functions between networks can affect the optimization process.

@highlight

A modification to the original hourglass network for single pose estimation that yields improvements over the original baseline.

@highlight

Authors extend a stacked hourglass network with inception-resnet-A modules and propose a multi-scale approach for human pose estimation in still RGB images.
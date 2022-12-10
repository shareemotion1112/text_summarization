This paper introduces the task of semantic instance completion: from an incomplete RGB-D scan of a scene, we aim to detect the individual object instances comprising the scene and infer their complete object geometry.

This enables a semantically meaningful decomposition of a scanned scene into individual, complete 3D objects, including hidden and unobserved object parts.

This will open up new possibilities for interactions with object in a scene, for instance for virtual or robotic agents.

To address this task, we propose 3D-SIC, a new data-driven approach that jointly detects object instances and predicts their completed geometry.

The core idea of 3D-SIC is a novel end-to-end 3D neural network architecture that leverages joint color and geometry feature learning.

The fully-convolutional nature of our 3D network enables efficient inference of semantic instance completion for 3D scans at scale of large indoor environments in a single forward pass.

In a series evaluation, we evaluate on both real and synthetic scan benchmark data, where we outperform state-of-the-art approaches by over 15 in mAP@0.5 on ScanNet, and over 18 in mAP@0.5 on SUNCG.

Understanding 3D environments is fundamental to many tasks spanning computer vision, graphics, and robotics.

In particular, in order to effectively navigate, and moreover interact with an environment, an understanding of the geometry of a scene and the objects it comprises is essential.

This is in contrast to the partial nature of reconstructed RGB-D scans; e.g., due to sensor occlusions.

For instance, for a robot exploring an environment, it needs to infer instance-level object segmentation and complete object geometry in order to perform tasks like grasping, or estimate spatial arrangements of individual objects.

Additionally, for content creation or mixed reality applications, captured scenes must be decomposable into their complete object components, in order to enable applications such as scene editing or virtual-real object interactions; i.e., it might be insufficient to predict object instance masks only for observed regions.

Thus, we aim to address this task of predicting object detection as well as instance-level completion for an input partial 3D scan of a scene; we refer to this task as semantic instance completion.

Previous approaches have considered semantic scene segmentation jointly with scan completion , but lack the notion of individual objects.

In contrast, our approach focuses on the instance level, as knowledge of instances is essential towards enabling interaction with the objects in an environment.

In addition, the task of semantic instance completion is not only important towards enabling objectlevel understanding and interaction with 3D environments, but we also show that the prediction of complete object geometry informs the task of semantic instance segmentation.

Thus, in order to address the task of semantic instance completion, we propose to consider instance detection and object completion in an end-to-end, fully differentiable fashion.

From an input RGB-D scan of a scene, our new 3D semantic instance completion network first regresses bounding boxes for objects in the scene, and then performs object classification followed by a prediction of complete object geometry.

Our approach leverages a unified backbone from which instance detection and object completion are predicted, enabling information to flow from completion to detection.

We incorporate features from both color image and 3D geometry of a scanned scene, as well as a fully-convolutional design in order to effectively predict the complete object decomposition of varying-sized scenes.

In summary, we present a fully-convolutional, end-to-end 3D CNN formulation to predict 3D instance completion that outperforms state-of-the-art, decoupled approaches to semantic instance completion by 15.8 in mAP@0.5 on real-world scan data, and 18.5 in mAP@0.5 on synthetic data:

• We introduce the task of semantic instance completion for 3D scans;

• we propose a novel, end-to-end 3D convolutional network which predicts 3D semantic instance completion as object bounding boxes, class labels, and complete object geometry, • and we show that semantic instance completion task can benefit semantic instance segmentation performance.

Object Detection and Instance Segmentation Recent advances in convolutional neural networks have now begun to drive impressive progress in object detection and instance segmentation for 2D images (Girshick, 2015; Ren et al., 2015; Liu et al., 2016; Redmon et al., 2016; Lin et al., 2017; Lin et al., 2018) .

Combined with the increasing availability of synthetic and real-world 3D data (Dai et al., 2017a; , we are now seeing more advances in object detection (Song & Xiao, 2014; 2015; Qi et al., 2017) and segmentation for 3D.

Recently, several approaches have been introduced to perform object detection and instance segmentation, applicable to single or multi-frame RGB-D input.

Wang et al. (2018) introduced SGPN to operate on point clouds by clustering semantic segmentation predictions.

Yi et al. (2018) leverages an object proposal-based approach to predict instance segmentation for a point cloud.

Simultaneously, Hou et al. (2019) presented an approach leveraging joint color-geometry feature learning for instance segmentation on volumetric 3D data.

Our approach also leverages an anchor-based object proposal mechanism for detection, but we leverage object completion to predict instance completion, as well as improve instance segmentation performance.

Scan completion of 3D shapes has been a long-studied problem in geometry processing, particularly for cleaning up broken mesh models.

In this context, traditional methods have largely focused on filling small holes by locally fitting geometric primitives, or through continuous energy minimization (Sorkine & Cohen-Or, 2004; Nealen et al., 2006; Zhao et al., 2007) .

Surface reconstruction approaches on point cloud inputs (Kazhdan et al., 2006; Kazhdan & Hoppe, 2013) can also be applied in this fashion to locally optimize for missing surfaces.

Other shape completion approaches leverage priors such as symmetry and structural priors (Thrun & Wegbreit, 2005; Mitra et al., 2006; Pauly et al., 2008; Sipiran et al., 2014; Speciale et al., 2016 ), or CAD model retrieval (Nan et al., 2012; Shao et al., 2012; Kim et al., 2012; Li et al., 2015; Shi et al., 2016) to predict the scan completion.

Recently, methods leveraging generative deep learning have been developed to predict the complete geometry of 3D shapes (Wu et al., 2015; Dai et al., 2017b; Han et al., 2017; Häne et al., 2017) .

extended beyond shapes to predicting the voxel occupancy for a single depth frame.

Recently, presented a first approach for data-driven scan completion of full 3D scenes, leveraging a fully-convolutional, autoregressive approach.

Both and show that inferring the complete scan geometry can improve 3D semantic segmentation.

With our approach for 3D semantic instance completion, this task not only enables new applications requiring instance-based knowledge of a scene (e.g., virtual or robotic interactions with objects in a scene), but we also show that instance segmentation can benefit from instance completion.

Our network takes as input an RGB-D scan, and learns to join together features from both the color images as well as the 3D geometry to inform the semantic instance completion.

The architecture is shown in Fig. 2 .

The input 3D scan is encoded as a truncated signed distance field (TSDF) in a volumetric grid.

To combine this with color information from the RGB images, we first extract 2D features using 2D convolutional layers on the RGB images, which are then back-projected into a 3D volumetric grid, and subsequently merged with geometric features extracted from the geometry.

The joint features are then fed into an encoder-decoder backbone, which leverages a series of 3D residual blocks to learn the representation for the task of semantic instance completion.

Objects are detected through anchor proposal and bounding box regression; these predicted object boxes are then used to crop and extract features from the backbone encoder to predict the object class label as well as the complete object geometry for each detected object as per-voxel occupancies.

We adopt in total five losses to supervise the learning process illustrated in Fig. 2 .

Detection contains three losses: (1) objectness using binary cross entropy to indicate that there is an object, (2) box location using a Huber loss to regress the 3D bounding box locations, and (3) classification of the class label loss using cross entropy.

Following detection, the completion head contains two losses: per-instance completion loss using binary cross entropy to predict per-voxel occupancies, and a proxy completion loss using binary cross entropy to classify the surface voxels belonging to all objects in the scene.

Our method operates on a unified backbone for detection followed by instance completion, enabling object completion to inform the object detection process; this results in effective 3D detection as well as instance completion.

Its fully-convolutional nature enables us to train on cropped chunks of 3D scans but test on a whole scene in a single forward pass, resulting in an efficient decomposition of a scan into a set of complete objects.

From an RGB-D scan input, our network operates on the scan's reconstructed geometry, encoded as a TSDF in a volumetric grid, as well as the color images.

To jointly learn from both color and geometry, color features are first extracted in 2D with a 2D semantic segmentation network Paszke et al. (2016) , and then back-projected into 3D to be combined with the TSDF features, similar to ; Hou et al. (2019) .

This enables complementary semantic features to be learned from both data modalities.

These features are then input to the backbone of our network, which is structured in an encoder-decoder style.

The encoder-decoder backbone is composed of a series of five 3D residual blocks, which generates five volumetric feature maps F = {f i |i = 1 . . .

5}. The encoder results in a reduction of spatial dimension by a factor of 4, and symmetric decoder results in an expansion of spatial dimension by a factor of 4.

Skip connections link spatially-corresponding encoder and decoder features.

For a more detailed description of the network architecture, we refer to the appendix.

As raw color data is often of much higher resolution than 3D geometry, to effectively learn from both color and geometry features, we leverage color information by back-projecting 2D CNN features learned from RGB images to 3D, similar to ; Hou et al. (2019) .

For each (2019)).

These joint features are used for object detection (as 3D bounding boxes and class labels) followed by per-instance geometric completion, for the task of semantic instance completion.

voxel location v i = (x, y, z) in the 3D volumetric grid, we find its pixel location p i = (x, y) in 2D views by camera intrinsic and extrinsic matrices.

We assign the voxel feature at location v i with the learned 2D CNN feature vector at p i .

To handle multiple image observations of the same voxel v i , we apply element-wise view pooling; this also allows our approach to handle a varying number of input images.

Note that this back-projection is differentiable, allowing our model to be trained end-to-end and benefit from both RGB and geometric signal.

For object detection, we predict the bounding box of each detected object as well as the class label.

To inform the detection, features are extracted from feature maps F 2 and F 3 of the backbone encoder.

We define two set of anchors on these two features maps, A s = {a i |i = 1 . . .

N s } and A b = {a i |i = 1 . . .

N b } representing 'small' and 'large' anchors for the earlier F 2 and later F 3 , respectively, so that the larger anchors are associated with the feature map of larger receptive field.

These anchors A s ∪ A b are selected through a k-means clustering of the ground truth 3D bounding boxes.

For our experiments, we use N s + N b = 9.

From these N s + N b clusters, A b are those with any axis > 1.125m, and the rest are in A s .

The two features maps F 2 and F 3 are then processed by a 3D region proposal to regress the 3D object bounding boxes.

The 3D region proposal first employs a 1 × 1 × 1 convolution layer to output objectness scores for each potential anchor, producing an objectness feature map with 2(N s + N b ) channels for the positive and negative objectness probabilities.

Another 1 × 1 × 1 convolution layer is used to predict the 3D bounding box locations as 6-dimensional offsets from the anchors; we then apply a non-maximum suppression based on the objectness scores.

We use a Huber loss on the log ratios of the offsets to the anchor sizes to regress the final bounding box predictions:

where µ is the box center point and φ is the box width.

The final bounding box loss is then:

otherwise.

Using these predicted object bounding boxes, we then predict the object class labels using features cropped from the bounding box locations from F 2 and F 3 .

We use a 3D region of interest pooling layer to unify the sizes of the cropped feature maps to a spatial dimension of 4 × 4 × 4 to be input to an object classification MLP.

For each object, we infer its complete geometry by predicting per-voxel occupancies.

Here, we crop features from feature map F 5 of the backbone, which has a feature map resolution matching the input spatial resolution, using the predicted object bounding box.

These features are processed through a series of five 3D convolutions which maintain the spatial resolution of their input.

The complete geometry is then predicted as voxel occupancy using a binary cross entropy loss.

We predict N classes potential object completions for each class category, and select the final prediction based on the predicted object class.

We define ground truth bounding boxes b i and masks m i as γ = {(b i , m i )|i = 1 . . .

N b }.

Further, we define predicted bounding boxesb i along with predicted masksm i asγ = {(b i ,m i )|i = 1 . .

.N b }.

During training, we only train on predicted bounding boxes that overlap with the ground truth bounding boxes:

We can then define the instance completion loss for each associated pair in Ω:

We further introduce a global geometric completion loss on entire scene level that serves as an intermediate proxy.

To this end, we use feature map F 5 as input to a binary cross entropy loss whose target is the composition of all complete object instances of the scene:

Our intuition is to obtain a strong gradient during training by adding this additional constraint to each voxel in the last feature map F 5 .

We find that this global geometric completion loss further helps the final instance completion performance; see Sec 6.

5 NETWORK TRAINING

The input 3D scans are represented as truncated signed distance fields (TSDFs) encoded in volumetric grids.

The TSDFs are generated through volumetric fusion (Curless & Levoy, 1996) during the 3D reconstruction process.

For all our experiments, we used a voxel size of ≈ 4.7cm and truncation of 3 voxels.

We also input the color images of the RGB-D scan, which we project to the 3D grid using their camera poses.

We train our model on both synthetic and real scans, computing 9 anchors through k-means clustering; for real-world ScanNet (Dai et al., 2017a) data, this results in 4 small anchors and 5 large anchors, and for synthetic SUNCG data, this results in 3 small anchors and 6 large anchors.

At test time, we leverage the fully-convolutional design to input the full scan of a scene along with its color images.

During training, we use random 96 × 48 × 96 crops (4.5 × 2.25 × 4.5 meters) of the scanned scenes, along with a greedy selection of ≤ 5 images covering the most object geometry in the crop.

Only objects with 50% of their complete geometry inside the crop are considered.

We train our model jointly, end-to-end from scratch.

We use an SGD optimizer with batch size 64 for object proposals and 16 for object classification, and all positive bounding box predictions (> 0.5 IoU with ground truth box) for object completion.

We use a learning rate of 0.005, which is decayed by a factor of 0.1 every 100k steps.

We train our model for 200k steps (≈ 60 hours) to convergence, on a single Nvidia GTX 1080Ti.

Additionally, we augment the data for training the object completion using ground truth bounding boxes and classification in addition to predicted object detection.

We evaluate our approach on semantic instance completion performance on synthetic scans of SUNCG scenes as well as on real-world ScanNet (Dai et al., 2017a) (Avetisyan et al., 2019) targets at mAP@0.5.

Our end-to-end formulation achieves significantly better performance than alternative, decoupled approaches that first use state-of-the-art scan completion and then instance segmentation (Hou et al., 2019 ) method or first instance segmentation (Hou et al., 2019) and then shape completion (Dai et al., 2017b ).

where we obtain ground truth object locations and geometry from CAD models aligned to ScanNet provided by (Avetisyan et al., 2019) .

To evaluate semantic instance completion, we use a mean average precision metric on the complete masks (at IoU 0.5).

Qualitative results are shown in Figs. 3 and 4.

Comparison to state-of-the-art approaches for semantic instance completion.

Tables 1 and 3 evaluate our method against alternatives for the task of semantic instance completion on our real and synthetic scans, respectively, with qualitative comparisons on ScanNet (Dai et al., 2017a) shown in Fig. 3 .

We compare to state-of-the-art 3D instance segmentation and scan completion approaches used sequentially; that is, first applying a 3D instance segmentation approach followed by a shape completion method on the predicted instance segmentation, as well as first applying a scene completion approach to the input partial scan, followed by a 3D instance segmentation method.

For 3D instance segmentation, we evaluate 3D-SIS (Hou et al., 2019) , which achieves state-of-the-art performance on a dense volumetric grid representation (the representation we use), and for scan completion we evaluate the 3D-EPN (Dai et al., 2017b) shape completion approach and ScanComplete scene completion approach.

Our end-to-end approach for semantic instance completion results in significantly improved performance due to information flow from instance completion to object detection.

Note that the ScanComplete model applied on ScanNet data is trained on synthetic data, due to the lack of complete ground truth scene data for real-world scans.

We can also evaluate our semantic instance completion predictions on the task of semantic instance segmentation by taking the intersection between the predicted complete mask and the input partial scan geometry to be the predicted instance segmentation mask.

In Tables 2 and 4 , we evaluate our method on 3D semantic instance segmentation with and without predicting instance completion, on ScanNet (Dai et al., 2017a) and SUNCG scans, as well as against a state-of-the-art 3D volumetric instance segmentation approach 3D-SIS (Hou et al., 2019) .

Here, we find that predicting instance completion significantly benefits instance segmentation performance.

What is the effect of a global completion proxy?

In Tables 1 and 3 , we demonstrate the impact of the geometric completion proxy loss; here, we see that this loss improves the semantic instance completion performance on both real and synthetic data.

In Tables 2 and 4 , we can see that it also improves semantic instance segmentation performance.

Can color input help?

We evaluate our approach with and without the color input stream; on both real and synthetic scans, the color input notably improves semantic instance completion performance, as shown in Tables 1 and 3.

display table bathtub trashbin sofa chair cabinet bookshelf avg 3D-SIS (Hou et al., 2019) 19 Table 2 : 3D Semantic Instance Segmentation on ScanNet (Dai et al., 2017a ) scans at mAP@0.5.

We evaluate our instance completion predictions on the task of semantic instance segmentation.

Here, predicting instance completion notably increases performance from only predicting instance segmentation (no completion).

scans at mAP@0.5.

Our semantic instance completion approach achieves significantly better performance than alternative approaches with decoupled state-of-the-art scan completion (SC) followed by instance segmentation (IS) (Hou et al., 2019) , as well as instance segmentation followed by shape completion (Dai et al., 2017b) .

We additionally evaluate our approach without color input (no color) and without a completion proxy loss on the network backbone (no proxy).

Our approach shows significant potential in the task of semantic instance completion, but several important limitations still remain.

First, we output a binary mask for the complete object geometry, which can limit the amount of detail represented by the completion; other 3D representations such as distance fields or sparse 3D representations (Graham & van der Maaten, 2017) could potentially resolve greater geometric detail.

Our approach also uses axis-aligned bounding boxes for object detection; it would be helpful to additionally predict the object orientation.

We also do not Table 4 : 3D Semantic Instance Segmentation on synthetic SUNCG scans at mAP@0.5.

We compare to 3D-SIS (Hou et al., 2019) , a state-of-the-art approach for 3D semantic instance segmentation on volumetric input, and additionally evaluate our approach without completion (no cmpl), without color input (no color), and without a completion proxy loss on the network backbone (no proxy).

Predicting instance completion notably benefits instance segmentation.

Figure 4: Qualitative results on SUNCG dataset (left: full scans, right: close-ups).

We sample RGB-D images to reconstruct incomplete 3D scans from random camera trajectories inside SUNCG scenes.

Note that different colors denote distinct object instances in the visualization.

consider object movement over time, which contains significant opportunities for semantic instance completion in the context of dynamic environments.

In this paper, we introduced the new task of semantic instance completion along with 3D-SIC, a new 3D CNN-based approach for this task, which jointly detects objects and predicts their complete geometry.

Our proposed 3D CNN learns from both color and geometry features to detect and classify objects, then predict the voxel occupancy for the complete geometry of the object in end-to-end fashion, which can be run on a full 3D scan in a single forward pass.

On both real and synthetic scan data, we significantly outperform alternative approaches for semantic instance completion.

We believe that our approach makes an important step towards higher-level scene understanding and helps to enable object-based interactions and understanding of scenes, which we hope will open up new research avenue.

Table 6 : Anchor sizes (in voxels) used for SUNCG region proposal.

Sizes are given in voxel units, with voxel resolution of ≈ 4.69cm Table 10 details the layers used in our backbone.

3D-RPN, classification head, and mask completion head are described in Table 11 .

Additionally, we leverage the residual blocks in our backbone, which is listed in Table 9 .

Note that both the backbone and mask completion head are fully-convolutional.

For the classification head, we use several fully-connected layers; however, we leverage 3D RoIpooling on its input, we can run our method on large 3D scans of varying sizes in a single forward pass.

We additionally list the anchors used for the region proposal for our model trained on our ScanNetbased semantic instance completion benchmark (Avetisyan et al., 2019; Dai et al., 2017a) and SUNCG datasets in Tables 5 and 6 , respectively.

Anchors for each dataset are determined through k-means clustering of ground truth bounding boxes.

The anchor sizes are given in voxels, where our voxel size is ≈ 4.69cm.

In this section, we present the inference timing with and without color projection in Table 7 and 8.

Note that our color projection layer currently projects the color signal into 3D space sequentially, and can be further optimized using CUDA, so that it can project the color features back to 3D space in parallel.

A scan typically contains several hundreds of images; hence, this optimization could significantly further improve inference time.

physical size (m) 4.7 x 7.7 7.9 x 9.6 10.7

@highlight

From an incomplete RGB-D scan of a scene, we aim to detect the individual object instances comprising the scene and infer their complete object geometry.

@highlight

Proposes an end-to-end 3D CNN structure which combines color features and 3D features to predict the missing 3D structure of a scene from RGB-D scans.

@highlight

The authors propose a novel end-to-end 3D convolutional network which predicts 3D semantic instance completion as object bounding boxes, class labels and complete object geometry.
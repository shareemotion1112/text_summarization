Discovering objects and their attributes is of great importance for autonomous agents to effectively operate in human environments.

This task is particularly challenging due to the ubiquitousness of objects and all their nuances in perceptual and semantic detail.

In this paper we present an unsupervised approach for learning disentangled representations of objects entirely from unlabeled monocular videos.

These continuous representations are not biased by or limited by a discrete set of labels determined by human labelers.

The proposed representation is trained with a metric learning loss, where objects with homogeneous features are pushed together, while those with heterogeneous features are pulled apart.

We show these unsupervised embeddings allow to discover object attributes and can enable robots to self-supervise in previously unseen environments.

We quantitatively evaluate performance on a large-scale synthetic dataset with 12k object models, as well as on a real dataset collected by a robot and show that our unsupervised object understanding generalizes to previously unseen objects.

Specifically, we demonstrate the effectiveness of our approach on robotic manipulation tasks, such as pointing at and grasping of objects.

An interesting and perhaps surprising finding in this approach is that given a limited set of objects, object correspondences will naturally emerge when using metric learning without requiring explicit positive pairs.

Figure 1: Object-Contrastive Networks (OCN): by attracting embedding nearest neighbors and repulsing others using metric learning, continuous object representations naturally emerge.

In a video collected by a robot looking at a table from different viewpoints, objects are extracted from random pairs of frames.

Given two lists of objects, each object is attracted to its closest neighbor while being pushed against all other objects.

Noisy repulsion may occur when the same object across viewpoint is not matched against itself.

However the learning still converges towards disentangled and semantically meaningful object representations which can be useful in autonomous robotics applications.

The ability to autonomously train to recognize and differentiate previously unseen objects as well as infer general properties and attributes is an important skill for robotic agents.

Increased autonomy leads to robustness, one of the main challenges real-world robotics faces.

It also renders scaling up data collection practical.

Additionally, removing human supervision from the loop has the potential to enable learning richer and less biased continuous representations than ones supervised by a limited set of discrete labels.

Unbiased representations can prove useful in unknown future environments different from the ones seen during supervision, a typical challenge for robotics.

In this work we present an unsupervised method that learns representations that disentangle perceptual and semantic object attributes such as class, function, and color.

We automatically acquire training data by capturing videos with a real robot; a robot base moves around a table to capture objects in various arrangements.

Assuming a pre-existing objectness detector, we extract objects from random frames within a same scene containing the same objects, and let the metric learning system decide how to assign positive and negative pairs of embeddings.

Representations that generalize across objects naturally emerge despite not being given groundtruth matches.

Unlike previous methods, we abstain from employing additional self-supervisory training signals such as tracking or depth.

The only inputs to the system are monocular videos.

This simplifies data collection and allows our embedding to integrate into existing end-to-end learning pipelines.

We demonstrate that a trained Object-Contrastive Network (OCN) embedding allows us to reliably identify object instances based on their visual features such as color and shape.

Moreover, we show that objects are also organized along their semantic or functional properties.

For example, a cup might not only be associated with other cups, but also with other containers like bowls or vases.

The key contributions of this work are: (1) an unsupervised algorithm for learning representations of objects (naturally encoding attributes like class, color, texture and function) which generalize to previously unseen objects; (2) showing monocular videos are sufficient to contrast similar and dissimilar objects pairs naturally without requiring explicit correspondences; (3) demonstrating the autonomy of the system, using a robot from data collection to tasks such as pointing and grasping similar objects to ones presented to it.

Object discovery from visual media.

Identifying objects and their attributes has a long history in computer vision and robotics BID42 .

Traditionally, approaches focus on identifying regions in unlabeled images to locate and identify objects BID40 BID36 BID1 BID8 BID18 .

Discovering objects based on the notion of 'objectness' instead of specific categories enables more principled strategies for object recognition BID43 BID35 .

Several methods address the challenge to discover, track, and segment objects in videos based on supervised BID46 or unsupervised BID20 BID38 BID10 techniques.

The spatio-temporal signal present in videos can also help to reveal additional cues that allow to identify objects BID47 BID15 .

In the context of robotics, methods also focus on exploiting depth to discover objects and their properties BID25 BID17 .Many recent approaches exploit the effectiveness of convolutional deep neural networks to detect objects BID34 BID24 BID23 and to even provide pixel-precise segmentations BID12 .

While the detection efficiency of these methods is unparalleled, they rely on supervised training procedures and therefore require large amounts of labeled data.

Self-supervised methods for the discovery of object attributes mostly focus on learning representations by identifying features in multi-view imagery BID4 BID21 and videos BID47 , or by stabilizing the training signal through domain randomization BID5 BID50 .Some methods not only operate on RGB images but also employ additional signals, such as depth BID7 BID32 or egomotion BID0 to self-supervise the learning process.

It has been recognized, that contrasting observations from multiple views can provide a view-invariant training signal allowing to even differentiate subtle cues as relevant features that can be leveraged for instance categorization and imitation learning tasks BID39 .Unsupervised representation learning.

Unlike supervised learning techniques, unsupervised methods focus on learning representations directly from data to enable image retrieval BID30 , transfer learning BID51 , image denoising BID45 , and other tasks BID6 BID19 .

Using data from multiple modalities, such as imagery of multiple views BID39 , sound BID27 BID2 , or other sensory inputs BID3 , along with the often inherent spatio-temporal coherence BID5 BID33 , can facilitate the unsupervised learning of representations and embeddings.

For example, BID49 explore multiple architectures to compare image patches and BID29 exploit temporal coherence to learn object-centric features.

BID9 rely of spatial proximity of detected objects to determine attraction in metric learning, OCN operates similarly but does not require spatial proximity for positive matches, it does however take advantage of the likely presence of a same object in any pair of frames within a video.

BID52 also take a similar unsupervised metric learning approach for tracking specific faces, using tracking trajectories and heuristics for matching trajectories and obtain richer positive matches.

While our approach is simpler in that it does not require tracking or 3D matching, it could be augmented with extra matching signals.

In robotics and other real-world scenarios where agents are often only able obtain sparse signals from their environment, self-learned embeddings can serve as an efficient representation to optimize learning objectives.

BID28 introduce a curiosity-driven approach to obtain a reward signal from visual inputs; other methods use similar strategies to enable grasping BID31 and manipulation tasks BID39 , or to be pose and background agnostic BID13 .

BID26 jointly uses 3D synthetic and real data to learn a representation to detect objects and estimate their pose, even for cluttered configurations.

BID14 learn semantic classes of objects in videos by integrating clustering into a convolutional neural network.

We propose an unsupervised approach to the problem of object understanding for multiple reasons: (1) make data collection simple and scalable, (2) increase autonomy in robotics by continuously learning about new objects without assistance, (3) discover continuous representations that are richer and more subtle than the discrete set of attributes that humans might provide as supervision which may not match future new environments.

All these objectives require a method that can learn about objects and differentiate them without supervision.

To bootstrap our learning signal we leverage two assumptions: (1) we are provided with a general objectness model so that we can attend to individual objects in a scene, (2) during an observation sequence the same objects will be present in most frames (this can later be relaxed by using an approximate estimation of ego-motion).

Given a video sequence around a scene containing multiple objects, we randomly select two frames I and I in the sequence and detect the objects present in each image.

Let us assume N and M objects are detected in image I andÎ, respectively.

Each of the n-th and m-th cropped object images are embedded in a low dimensional space, organized by a metric learning objective.

Unlike traditional methods which rely on human-provided similarity labels to drive metric learning, we use a selfsupervised approach to mine synthetic synthetic similarity labels.

To detect objects, we use Faster-RCNN (Ren et al., 2015) trained on the COCO object detection dataset BID22 .

Faster-RCNN detects objects in two stages: first generate class-agnostic bounding box proposals all objects present in an image ( Fig. 1) , second associate detected objects with class labels.

We use OCN to discover object attributes, and only rely on the first objectness stage of Faster-R-CNN to detect object candidates.

Examples of detected objects are illustrated in Fig. 1 .

We denote a cropped object image by x ∈ X and compute its embedding via a convolutional neural network f (x) ∶ X → K. Note that for simplicity we may omit x from f (x) while f inherits all superscripts and subscripts.

Let us consider two pairs of images I andÎ that are taken at random from the same contiguous observation sequence.

Let us also assume there are n and m objects detected in I andÎ respectively.

We denote the n-th and m-th objects in the images I andÎ as x I n and xÎ m , respectively.

We compute the distance matrix DISPLAYFORM0 For every embedded anchor f I n , n ∈ 1..N , we select a positive embedding fÎ m with minimum distance as positive: fÎ n+ = argmin(D n,m ).

Given a batch of (anchor, positive) pairs {( DISPLAYFORM1 , the n-pair loss is defined as follows BID41 : DISPLAYFORM2 The loss learns embeddings that identify ground truth anchor-positive pairs from all other anchornegative pairs in the same batch.

It is formulated as a sum of softmax multi-class cross-entropy losses over a batch, encouraging the inner product of each anchor-positive pair (f i , f + i ) to be larger than all anchor-negative pairs ( DISPLAYFORM3 The final OCN training objective over an observation sequence is the sum of npairs losses over all pairs of individual frames: DISPLAYFORM4 OCN takes a standard ResNet50 architecture until layer global pool and initializes it with ImageNet pre-trained weights.

We then add three additional ResNet convolutional layers and a fully connected layer to produce the final embedding.

The network is trained with the n-pairs metric learning loss as discussed in Sec. 3.2.

Our architecture is depicted in Fig. 1 class (12) color (8) has_buttons (2) has_flat_surface (2) has_legs (2) has_lid (2) has_wheels (2) is_container (2) is_device (2) Figure 2: Models and baselines: for comparison purposes all models evaluated in Sec. 5 share the same architecture of a standard ResNet50 model followed by additional layers.

While the architectures are shared, the weights are not across models.

While the unsupervised model (left) does not require supervision labels, the 'softmax' baseline as well as the supervised evaluations (right) use attributes labels provided with each object.

We evaluate the quality of the embeddings with two types of classifiers: linear and nearest neighbor.

By using multiple views of the same scene and by attending to individual objects, our architecture allows us to differentiate subtle variations of object attributes.

Observing the same object across different views facilitates learning invariance to scene-specific properties, such as scale, occlusion, lighting, and background, as each frame exhibits variations of these factors.

The network solves the metric loss by representing object-centric attributes, such as shape, function, texture, or color, as these are consistent for (anchor, positive)-pairs, and dissimilar for (anchor, negative)-pairs.

One might expect that this approach may only work if it is given a good enough initialization so that matching the same object across multiple frames is more likely than random chance.

While ImageNet pretraining certainly helps convergence as shown in Table 1 , it is not a requirement to learn meaningful representations as shown in Sec. 8.

When all weights are random and no labels are provided, what can drive the network to consistently converge to meaningful embeddings?

We estimate that the co-occurrence of the following hypotheses drives this convergence: (1) objects often remains visually similar to themselves across multiple viewpoints, (2) limiting the possible object matches within a scene increases the likelihood of a positive match, (3) the low-dimensionality of the embedding space forces the model to generalize by sharing abstract features across objects, (4) the smoothness of embeddings learned with metric learning facilitates convergence when supervision signals are weak, and (5) occasional true-positive matches (even by chance) yield more coherent gradients than false-positive matches which produce inconsistent gradients and dissipate as noise, leading over time to an acceleration of consistent gradients and stronger initial supervision signal.

To evaluate the effectiveness of OCN embeddings we generated two datasets of real and synthetic objects.

For the (unlabeled) real data we arrange objects in table-top configurations and capture frames from continuous camera trajectories.

The (labeled) synthetic data is generated from renderings of 3D objects in a similar configuration.

Details about the datasets are reported in TAB5 .

To generate diverse object configurations we use 12 categories (airplane, car, chair, cup, bottle, bowl, guitars, keyboard, lamp, monitor, radio, vase) from ModelNet BID48 .

The selected categories cover around 8k models of the 12k models available in the entire dataset.

ModelNet provides the object models in a 80-20 split for training and testing.

We further split the testing data into models for test and validation, resulting in a 80-10-10 split for training, validation, and test.

For validation purposes, we manually assign each model labels describing the semantic and functional properties of the object, including the labels 'class', 'has lid', 'has wheels', 'has buttons', 'has flat surface', 'has legs', 'is container', 'is sittable', 'is device'.

Fig. 9 shows an example scene.

We randomly define the number of objects (up to 20) in a scene and select half of the objects from two randomly selected categories.

The other half is selected from the remaining object categories.

We further randomly define the positions of the objects and vary their sizes, both so that they do not intersect.

Additionally, each object is assigned one of eight predefined colors.

We use this setup to generate 100K scenes for training, and 50K scenes for each, validation and testing.

For each scene we generate a number (n = 10) of views and select random combination of two views for detecting objects.

In total we produce 400K views (200 pairs) for training and 50K views (25K pairs) for each, validation and testing.

Our real object data set consists of 187 unique object instances spread across six categories including 'balls', 'bottles & cans', 'bowls', 'cups & mugs', 'glasses', and 'plates'.

TAB6 provides details about the number of objects in each category and how they are split between training, test, and validation.

Note that we distinguish between cups & mugs and glasses categories based on whether it contains a handle.

FIG1 provides a snapshot of our entire object dataset.

We automated the real world data collection through using a mobile robot equipped with an HD camera (Fig. 8) .

At each run, we place about 10 objects on the table and then trigger the capturing process by having the robot rotate around the table by 90 degrees (see Fig. 8 ).

In average 130 images are captured at each run.

We select random pairs of frames for each trajectory during training of the OCN.

We performed 345, 109, and 122 runs of data collection for training, test, and validation dataset, respectively.

In total 43084 images were captured for OCN training and 15061 and 16385 were used for test and validation, respectively.

An OCN is trained based on two views of the same synthetic or real scene.

We randomly pick two frames of a camera trajectory around the scene to ensure the same objects are present; the frames are selected based on their time stamps so that they are as far apart as possible.

We set the npairs regularization to λ = 0.002.

The distance matrix D n,m (Sec. 3.2) is constructed based on the individually detected objects for each of the two frames.

The object detector was not specifically trained on any of our datasets.

Furthermore, we only used scenes where at least 5 objects were detected in each frame.

Operating on less objects results in a more noisy training signal as the n-pairs loss cannot create enough meaningful (anchor, negative)-pairs for contrasting them with the (anchor, positive)-pair.

As the number of detected objects per view varies, we reciprocally use both frames to find anchors and their corresponding positives as discussed in Sec. 3.2.

Across our experiments, the OCN training converged after 600k-1.2M iterations.

To evaluate the effectiveness of an OCN embedding as representation for object attribute disentanglement, we performed experiments on a large-scale synthetic dataset and two robotic tasks of pointing and grasping in a real-world environment.

Moreover, the experiments are designed in a way to directly showcase the usefulness of OCN on real robotics applications.

One way to evaluate the quality of unsupervised embeddings is to train attribute classifiers on top of the embedding using labeled data.

Note however this may not entirely reflect the quality of an embedding because it is only measuring a discrete and small number of attributes while an embedding may capture more continuous and larger number of abstract concepts.

We consider two types of classifiers to be applied on top of existing embeddings in this experiment: linear and nearest-neighbor classifiers.

The linear classifier consists of a single linear layer going from embedding space to the 1-hot encoding of the target label for each attribute.

It is trained with a range of learning rates and the best model is retained for each attribute.

The nearest-neighbor classifier consists of embedding an entire 'training' set, and for each embedding of the evaluation set, assigning to it the labels of the nearest sample from the training set.

Nearestneighbor classification is not a perfect approach because it does not necessarily measure generalization as linear classification does and results may vary significantly depending on how many nearest neighbors are available.

It is also less subject to data imbalances.

We still report this metric to get a sense of its performance because in an unsupervised inference context, the models might be used in a nearest-neighbor fashion (e.g. as in Sec. 5.3).Baselines: we compare multiple baselines in TAB7 .

The 'Softmax' baseline refers to the model described in Fig. 2, i .e.

the exact same architecture as for OCN except that the model is trained with a supervised cross-entropy/softmax loss.

The 'ResNet50' baseline refers to using the unmodified outputs of the ResNet50 model BID11 (He et al., ) (2048 as embeddings and training a nearest-neighbor classifier as defined above.

We consider 'Softmax' and 'ResNet50' baselines as the lower and upper error-bounds for standard approaches to a classification task.

The 'OCN supervised' baseline refers to the exact same OCN training described in Fig. 2 , except that the positive matches are provided rather than discovered automatically. '

OCN supervised' represents the metric learning upper bound for classification.

Finally we indicate as a reference the error rates for random classification.

Results: we quantitatively evaluate our unsupervised models against supervised baselines on the labeled synthetic datasets (train and test) introduced in Sec. 4.

Note that there is no overlap in object instances between the training and the evaluation set.

The first take-away is that unsupervised performance closely follows its supervised baseline when trained with metric learning.

As expected the cross-entropy/softmax approach performs best and establishes the error lower bound while the ResNet50 baseline are upper-bound results.

Note that the dataset is heavily imbalanced for the Table 1 : Attributes classification errors: using attribute labels, we train either a linear or nearest-neighbor classifier on top of existing fixed embeddings.

The supervised OCN is trained using labeled positive matches, while the unsupervised one decides on positive matches on its own.

All models here are initialized and frozen with ImageNet-pretrained weights for the ResNet50 part of the architecture (see Fig. 2 Figure 4 : An OCN embedding organizes objects along their visual and semantic features.

For example, a red bowl as query object is associated with other similarly colored objects and other containers.

The leftmost object (black border) is the query object and its nearest neighbors are listed in descending order.

The top row shows renderings of our synthetic dataset, while the bottom row shows real objects.binary attributes reported in TAB7 and require balancing for linear classification.

In Fig. 4 and Sec. 9, 11, we show qualitative results of nearest neighbor objects discovered by OCN.

An OCN embedding can be used to match instances of the same object across multiple views and over time.

This is illustrated in Fig. 5 , where objects of one view (anchors) are matched against the objects of another view.

We can find the nearest neighbors (positives) in the scene through the OCN embedding space as well as the closest matching objects with descending similarity (negatives).

We report the quality of finding corresponding objects in TAB3 and differentiate between attribute errors, that indicate a mismatch of specific attributes (e.g. a blue cup is associated with a red cup), and object matching errors, which measure when objects are not of the same instance.

An OCN embedding significantly improves detecting object instances across multiple views.

Objects of View 1

Anchors Positives Negatives Distances Figure 5 : View-to-view object correspondences: the first column shows all objects detected in one frame (anchors).

Each object is associated to the objects found in the other view, objects in the second column are the nearest neighbors.

The third column shows the distances of all objects, all other objects are shown from left to right in descending order according to their distances to the anchor.

Pointing: We evaluate performance of OCN on a pointing robotic task (Fig. 6) .

The robot has to point to an object that it deems most similar to the object directly in front of him on the small table.

The objects on the big table are randomly selected from each of the six object categories TAB6 .

We consider two sets of these target objects.

The quantitative experiment in TAB4 uses three query objects per category and is ran three times for each combination of query and target objects (3 × 2 × 18 = 108 experiments performed).

The full set of experiments for one of the three runs is illustrated in FIG5 .

TAB4 quantifies OCN performance of this experiment.

We report on errors related to 'class' and 'container' attributes (note that the other ten attributes described in Sec. 4.1 are not relevant to the real object data set).

While the trained OCN model is performing well on the most categories, it has particularly some difficulty on the object classes 'cups & mugs' and 'glasses'.

These categories are generally mistaken with the category 'bowls'.

As a result the network performs much better in the attribute 'container' since all the three categories 'bowls', 'bottles & cans', and 'glasses' refer to the same attribute.

Grasping: We qualitatively evaluate the OCN performance on a grasping task in an environment unseen during training.

First, a person holds and shows an object to the robot, then the robot picks up the most similar object from a set of objects on a table (see Fig. 7 ).

In this experiment, we focus on evaluating OCN with objects that have either similar shape or color attribute.

Using OCN the robot can successfully identify and grasp the object that has the closest color and shape attributes to the query object.

Note training data did not contain objects held by hand.

Figure 6 : The robot experiment of pointing to the best match to a query object (placed in front of the robot on the small table).

The closest match is selected from two sets of target object list, which are placed on the long table behind the query object.

The first and the second row respectively correspond to the experiment for the first and second target object lists.

Each column also illustrates the query objects for each object category.

Image snapshots with green frame correspond to cases where both the 'class' and 'container' attributes are matched correctly.

Image snapshots with blue frame refer to the cases where only 'container' attribute is matched correctly.

Images with red frames indicates neither of attributes are matched.

We introduced a novel unsupervised representation learning algorithm that allows us to differentiate object attributes, such as color, shape, and function.

An OCN embedding is learned by contrasting the features of objects captured from two frames of single view camera trajectories of table-top indoor environments.

We specifically attend to individual objects by detecting object bounding boxes and leverage a metric learning loss to disentangle subtle variations of object attributes.

The resulting embedding space allows to organize objects along multiple dimensions and serves as representation for robotic learning.

We show that an OCN embedding can be used on real robotic tasks such as Figure 7 : Robot experiment of grasping the object that is closest to the query object (held by hand).

Images on the left are captured by the robot camera, and the images on the right are the video frames from a third person view camera.

The leftmost object (black border) is the query object and its nearest neighbors are listed in descending order.

The top row and the bottom row show the robot successfully identifies and grasps the object with similar color and shape attribute respectively.grasping and pointing, where it is important to differentiate visual and semantic attributes of individual object instances.

Finally, we show that an OCN can be trained efficiently from RGB videos that are automatically obtained from a real robotic agent.

Figure 8: Consecutive frames captured with our robotic setup.

At each run we randomly select 10 objects and place them on the table.

Then a robot moves around the table and take snapshots of the table at different angles.

We collect in average 80-120 images per scene.

We select pairs of two frames of the captured trajectory and train the OCN on the detected objects.

Figure 9: Synthetic data: two frames of a synthetically generated scene of table-top objects (a) and a subset of the detected objects (c).

To validate our method against a supervised baseline, we additionally render color masks (b) that allow us to identify objects across the views and to associate them with their semantic attributes after object detection.

Note that objects have the same color id across different views.

The color id's allow us to supervise the OCN during training.

We find in TAB7 that models that are not pretrained with ImageNet supervision perform worse but still yield reasonable results.

This indicates that the approach does not rely on a good initialization to bootstrap itself without labels.

Even more surprisingly, when freezing the weights of the ResNet50 base of the model to its random initialization, results degrade but still remain far below chance as well as below the 'ResNet50 embeddings' baseline.

Obtaining reasonable results with random weights has already been observed in prior work such as BID16 BID37 and BID44 .

Figure 10 : A result showing the organization of real bowls based on OCN embeddings.

The query object (black border, top left) was taken from the validation all others from the training data.

As the same object is used in multiple scenes the same object is shown multiple times.

Figure 11 : A result showing the organization of real bowls based on OCN embeddings.

The query object (black border, top left) was taken from the validation all others from the training data.

As the same object is used in multiple scenes the same object is shown multiple times.

The robot experiment of pointing to the best match to a query object (placed in front of the robot on the small table).

The closest match is selected from two sets of target object list, which are placed on the long table behind the query object.

The first and the last three rows respectively correspond to the experiment for the first and second target object lists.

Each column also illustrates the query objects for each object category.

Image snapshots with green frame correspond to cases where both the 'class' and 'container' attributes are matched correctly.

Image snapshots with blue frame refer to the cases where only 'container' attribute is matched correctly.

Images with red frames indicates neither of attributes are matched.

Figure 16: An OCN embedding organizes objects along their visual and semantic features.

For example, a red bowl as query object is associated with other similarly colored objects and other containers.

The leftmost object (black border) is the query object and its nearest neighbors are listed in descending order.

The top row shows renderings of our synthetic dataset, while the bottom row shows real objects.

For real objects we removed the same instance manually.

<|TLDR|>

@highlight

An unsupervised approach for learning disentangled representations of objects entirely from unlabeled monocular videos.

@highlight

Designs a feature representation from video sequences captured from a scene from different view points.

@highlight

Proposal for an unsupervised representation learning method for visual inputs that incorporates a metric learning approach pulling nearest neighbor pairs of image patches close in embedding space while pushing apart other pairs.

@highlight

This paper explores self-supervised learning of object representations, with the main idea to encourage objects with similar features to get further ‘attracted’ to each other.
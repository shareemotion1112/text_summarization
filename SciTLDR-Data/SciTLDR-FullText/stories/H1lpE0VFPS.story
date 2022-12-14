In this paper, we consider the problem of detecting object under occlusion.

Most object detectors formulate bounding box regression as a unimodal task (i.e., regressing a single set of bounding box coordinates independently).

However, we observe that the bounding box borders of an occluded object can have multiple plausible configurations.

Also, the occluded bounding box borders have correlations with visible ones.

Motivated by these two observations, we propose a deep multivariate mixture of Gaussians model for bounding box regression under occlusion.

The mixture components potentially learn different configurations of an occluded part, and the covariances between variates help to learn the relationship between the occluded parts and the visible ones.

Quantitatively, our model improves the AP of the baselines by 3.9% and 1.2% on CrowdHuman and MS-COCO respectively with almost no computational or memory overhead.

Qualitatively, our model enjoys explainability since we can interpret the resulting bounding boxes via the covariance matrices and the mixture components.

Figure 1: We observe that an occluded bounding box usually exhibits multiple modes in most detection datasets, no matter whether the ground truth annotation is visible box or full box: (a) visible bounding box annotation (b) full object bounding box labeled by different annotators (c) visible bounding box annotated accurately (d) visible bounding box annotated inaccurately

Object detectors based on deep convolutional neural networks (CNNs) are the backbone of many real-world applications like self-driving cars (Huval et al., 2015) , robotics grasping (Calandra et al., 2018) and video surveillance (Joshi & Thakore, 2012) .

Most object detectors learn to detect an object in two folds (Ren et al., 2015) : (1) categorization of the candidate bounding box (2) regress each coordinate of the candidate box towards the ground truth one independently.

Currently, there are two styles of bounding box annotation among the large-scale object detection datasets: (1) visible box that only contains visible parts (e.g., MS-COCO (Lin et al., 2014) and PASCAL VOC (Everingham et al., 2010) ) (2) full box that contains both visible and occluded parts (e.g., CrowdHuman (Shao et al., 2018) and VehicleOcclusion (Wang et al., 2017) ).

For full box annotation, regressing a single set of bounding box coordinates works well for fully visible objects, since it is a unimodal problem.

However, when an object is occluded, we observe that its occluded parts can have several plausible configurations (e.g., Figure 1 (b)), which is a multimodal problem.

Even for visible box annotation, an object sometimes still exhibits multiple modes due to inaccurate labeling (e.g., Figure 1 (c) vs. (d)).

We argue that an object detector robust to occlusion should learn a multimodal distribution with the capability of proposing more than one plausible hypothesis for the configuration of an occluded part.

Besides, we also observe that the bounding box coordinates have correlations by nature.

Take Figure 1 (c) as an example, by knowing the position of the car's roof, we can easily infer the location of the left border even without looking at it.

Therefore, an object detector robust to occlusion also needs to be capable of inferring the correlations between the occluded bounding box borders and the visible ones.

Motivated by these two observations, we propose a deep multivariate mixture of Gaussians model for object detection under occlusion.

Concretely, instead of regressing a single set of bounding box coordinates, our model regresses several sets of coordinates, which are the means of the Gaussians.

Moreover, we learn a covariance matrix for the coordinates of each Gaussian mixture component.

These components are summed together as the prediction for the distribution of plausible bounding box configurations.

At inference time, we choose the expectation of our model's distribution as the final predicted bounding box.

To demonstrate the generalizability of our proposed model, we conduct experiments on four datasets: CrowdHuman, MS-COCO, VehicleOcclusion, and PASCAL VOC 2007.

Quantitatively, our model improves the AP (Average Precision of the baselines by 3.9% and 1.2% on CrowdHuman and MS-COCO respectively (Table 1 and Table 2 ).

Qualitatively, our model enjoys explainability since the resulting bounding boxes can be interpreted using the covariance matrices and the Gaussian mixture components ( Figure 5 and Figure 4 ).

More importantly, our model is almost computation and memory free, since predicting the mixture components only requires a fully-connected layer, and we can discard the covariance matrices at inference time (Table 5) .

Object Detection: Deep convolutional neural networks were first introduced to object detection in R-CNN (Girshick et al., 2014) and Fast R-CNN (Girshick, 2015) .

Currently, there are mainly two types of object detectors: one-stage object detectors and two-stage object detectors.

One-stage detectors like YOLO (Redmon et al., 2016) , SSD (Liu et al., 2016) and RetinaNet are fast in general.

Two-stage detectors (Ren et al., 2015; Zhu et al., 2018; Singh et al., 2018) are accurate however sacrificing speed.

In this paper, although we conduct experiments based on the Faster R-CNN heads of Faster R-CNN and Mask R-CNN, our method is not limited to two-stage detectors.

Object Detection Under Occlusion: Occlusion-aware R-CNN (Zhang et al., 2018b) proposes to divide pedestrian detection into five parts and predict the visibility scores respectively, which are integrated with the prior structure information of the human body into the network to handle occlusion.

Zhang et al. (2018a) proposes an attention network with self or external guidance.

These methods are specifically designed for pedestrian detection task.

By contrast, our method is designed for general object detection.

Deep Voting (Zhang et al., 2018c) proposes to utilize spatial information between visual cues and semantic parts and also learn visual cues from the context outside an object.

However, detecting semantic parts needs manual labels, which our approach does not require.

Besides, our approach does not introduce additional computation during the inference (Table 5 ).

Amodal instance segmentation (Li & Malik, 2016) considers the task of predicting the region encompassing both visible and occluded parts of an object.

The authors propose to add synthetic occlusion to visible objects and retain their original masks, then employ a CNN to learn on the generated composite images, which resembles the VehicleOcclusion in our experiments.

He et al. (2019) proposes bounding box regression with uncertainty, which is a degradation case of our model (Gaussian).

Datasets for Detection under Occlusion: Currently, there are three categories of annotation for an occluded object: (1) visible bounding box that contains the visible parts (2) full box that contains both visible and occluded parts of an object annotated by human (3) full box by synthesizing occluders on a visible object.

MS-COCO, PASCAL VOC, ImageNet (Deng et al., 2009) Figure 2 : Faster R-CNN head architecture for our approach: We extended the existing Faster R-CNN head to predict the parameters of multivariate mixture of Gaussian ??, ?? and ?? Cityscapes (Cordts et al., 2016) fall into the first category.

CrowdHuman and Semantic Amodal Segmentation dataset require the annotators to label the invisible parts.

VehicleOcclusion instead synthesizes the occluders for visible objects.

We conduct experiments on MS-COCO, PASCAL VOC 2007, CrowdHuman, and VehicleOcclusion, covering all these categories.

We observe that when an object is partially occluded, the occluded bounding box border can usually be inferred to some extent by other visible parts of the object (e.g., it is easy to infer the left border of the car given the car roof position in Figure 1 (c)).

Besides, the occluded bounding box exhibits multiple modes.

For example, the left arm of the teddy bear could have several possible configurations in Figure 1 (b) .

Motivated by these two observations, we propose to estimate the bounding box coordinates as a probability distribution during bounding box regression instead of a set of deterministic coordinates.

Specifically, we propose to estimate a multivariate mixture of Gaussians distribution with a deep network.

Multivariate Gaussian helps the case where bounding box borders have correlations, and a mixture of Gaussians helps the case where an occluded bounding box border exhibits multiple modes.

Formally, we predict the distribution p ?? (x|I) given the feature maps I of a region of interest (RoI).

The distribution is parameterized by ??, which is a neural network (e.g., Faster R-CNN head, Figure 2 ).

The distribution has

T , which is the most probable bounding box coordinates relative to the RoI, estimated by the component:

?? is the covariance matrix, which is a symmetric semi-positive definite matrix in general.

To be able to compute the inverse ?? ???1 , we constrain the covariance matrix to be a symmetric positive definite matrix.

In this case, the precision matrix ?? ???1 is also a symmetric positive definite matrix.

During training, the model estimates the precision matrix ?? ???1 instead of the covariance matrix ??, so that we do not need to compute the inverse every time during training which we also find more stable in our experiments.

To ensure the properties of the precision matrix ?? ???1 , we parameterize it using the Cholesky decomposition:

where U is an upper triangular matrix with strictly positive diagonal entries, such that Cholesky decomposition is guaranteed to be unique.

We parameterize the mixture weights ?? i using Softmax, so that they range from 0 to 1 and sum to 1:

z i , u ii and ?? i are outputs produced by a fully-connected layer on top of the final fully-connected layer fc7 on the Faster R-CNN head.

Take Faster R-CNN with RPN as an example, Figure 2 shows the architecture of our model.

Since we only modify a small part of the architecture, our approach might also be applied to other object detectors than Faster R-CNN, like one-stage object detectors YOLO and RetinaNet.

Learning: Our model parameterizes the distribution over bounding boxes using a neural network which depends on RoI features.

During training, we estimate the parameters ?? with maximum likelihood estimation on a given dataset {I , ?? * | = 1, 2, ..., N }, where ?? * represents the ground truth coordinates for RoI feature maps I and N is the number of observations:

In practice, N is the number of samples in a mini-batch.

We use momentum stochastic gradient descent (SGD) to minimize the localization loss L loc and the classification loss L cls :

Note that we use different parameters ?? for different classes in practice.

For simplicity, the formulation above only considers the regression problem for a single class.

Inference: During testing, we use the expectation of our mixture module as prediction:

Notice that the covariance matrix ?? i is not involved in inference.

In practice, we discard the neurons that produce the covariance matrix to speed up inference.

In our experiments (Table 5) , our model has almost the same inference latency and memory consumption as the baseline network.

Multivariate Gaussian: When the number of mixture components K = 1, our model degrades into a multivariate Gaussian model.

And the localization loss can be rewritten as follow (for simplicity, we only illustrate the loss for a single sample ):

where 2 ln 2?? is a constant which can be ignored during training.

where (U i ) jj is the jth diagonal element of the matrix U i .

Multimodality is helpful under occlusion because an occluded object usually has multiple modes.

Gaussian: When the number of mixture components K = 1 and the covariance is constrained to be a diagonal matrix, it becomes a simple Gaussian model where different variables are independent:

We argue that this simple model helps detection in most cases.

Here (U ) jj behaves like a balancing term.

When the bounding box regression is inaccurate (large (?? * j ??? ?? j ) 2 /2), the variance 1/(U ) 2 jj tends to be larger.

Therefore smaller gradient will be provided to bounding box regression (U )

2 /2 in this case, which might help training the network (Table 1 and Table 2 ).

If bounding box regression is perfect, U tend to infinity (i.e., the variance should be close 0).

However, regression is not that accurate in practice, U will be punished for being too large.

Euclidean Loss:

When all the diagonal elements (U ) jj are one (u jj = 0), our model degenerates to the standard euclidean loss:

We initialize the weights of ?? i , z i and u ii layers (Figure 2 ) using random Gaussian initialization with standard deviations 0.0001 and biases 0, ???1 and 0 respectively.

So that at the start of training, bounding box coordinate ?? i is at an unbiased position, U i is an identity matrix and ?? i treats each mixture component equally.

Our model can be trained end-to-end.

Unless specified, we follow settings in Detectron and those original papers.

To demonstrate the generalizability of our method, we conduct experiments on four datasets: CrowdHuman (Shao et al., 2018 ) is a large, rich-annotated and highly diverse dataset for better evaluation of detectors in crowd scenarios.

Its training and validation sets contain a total of 470k human instances, and around 22.6 persons every image under various kinds of occlusions.

The annotations for occluded bounding boxes are full boxes (Figure 1 (b) ) instead of visible boxes (Figure 1 (a) ).

The experiments are in Table 1 .

VehicleOcclusion is a synthetic dataset designed for object detection under occlusion (Wang et al., 2017) .

Same as above, the annotations are full boxes.

The occlusion annotations are more accurate since the occluders (occluding objects) are randomly placed on the annotated visible object.

It contains six types of vehicles and occluded instances at various difficulty levels.

Specifically, it consists of four occlusion levels: No occlusion (0%), L1 (20% ??? 40%), L2 (40% ??? 60%), L3 (60% ??? 80%).

The percentages are computed by pixels.

At level L1, L2 and L3, there are two, three, and four occluders placed on the object, respectively (Table 4) . (Lin et al., 2014 ) is a large-scale object detection dataset containing 80 object categories, 330k images (> 200k labeled) and 1.5 million object instances.

Compared with the two datasets above, MS-COCO has fewer occlusion cases.

For example, the IoU (intersection over union) between overlapped human bounding boxes in MS-COCO are less than 0.7 (Shao et al., 2018) .

We use train2017 for training and val2017 for testing (Table 2) .

Different from above, the annotations are visible boxes.

PASCAL VOC 2007 has 9,963 images and 20 classes in total, containing 24,640 annotated objects (Everingham et al.) .

Similar with MS-COCO, this dataset has less occlusion cases than the first two datasets.

We use voc_2007_train and voc_2007_val for training and voc_2007_test for testing (Table 3 ).

The annotations are visible boxes.

Number of Mixture Components: Shown in Figure 3 , we test our mixture of Gaussians model by varying the number of mixture components.

The baseline is ResNet-50 FPN Faster R-CNN (He et al., 2016; Lin et al., 2017) on CrowdHuman.

As the number of components increases from 1, 4 to 8, we observe consistent performance improvement.

The mixture of eight Gaussians model (Eq. 8) outperforms Gaussian model (Eq. 9) by 1% AP.

However, the performance goes down when there are more than 16 components.

This might be because the objects in the dataset might not have as many as 16 modes when occluded.

Besides, the more components we have, the higher the chance of over-fitting.

Unless specified, we use eight components for the mixture of Gaussians model.

Mixture of Gaussian vs. Multivariate Gaussian: Shown in Table 1 and 2, we compare the degradation cases of our complete model (Eq. 1): Gaussian (Eq. 9), mixture of Gaussians (Eq. 8) and multivariate Gaussian (Eq. 7) on CrowdHuman and MS-COCO.

For CrowdHuman, we use ResNet-50 FPN Faster R-CNN as the baseline.

For MS-COCO, we use ResNet-50 FPN Mask R-CNN.

On CrowdHuman which has a lot of crowded scenes, our model greatly improves the baseline.

Gaussian improves the baseline by 1.2% AP.

A mixture of eight Gaussians improves 2.3% AP, and multivariate Gaussians improves 2.9% AP.

The complete model improves the performance by 3.9% AP.

The improvements indicate all these assumptions are helpful under heavy occlusion.

Gaussian helps training the regression network by learning to decrease the gradients for high variance cases.

Multivariate Gaussian helps to learn the correlations between an occluded border and the visible borders.

Mixture of Gaussians helps to learn a multimodal model for the occluded cases which have multiple modes.

Soft-NMS (Bodla et al., 2017) modifies classification scoring, while our approach improves localization.

Though it achieves comparable performance (1.7% AP improvement), it can be applied together with our method.

With soft-NMS, the AP of mixture of 8 Gaussian, multivariate Gaussian and the complete model further improves 1.7%, 1.5% and 1.5% respectively.

On MS-COCO, the bounding box annotations are visible boxes instead of full boxes used in CrowdHuman.

Gaussian still works here which improves the baseline by 0.4% AP, since there are variances in the dataset caused by inaccurate annotation (e.g., Figure 1 (d) ).

Gaussian helps to reduce the gradients for these ambiguous cases.

A mixture of eight Gaussians improves 0.6% AP, and multivariate Gaussians improves 0.7% AP.

The complete model improves the performance by 1.2% AP.

The improvements are noticeable, however less significant than on CrowdHuman.

On the one hand, there are fewer occluded instances in MS-COCO, multimodality and covariances might not as helpful as in CrowdHuman.

On the other hand, predicting full boxes require guessing the invisible parts where multimodality and covariances are more useful.

We further conduct experiments on PASCAL VOC 2007, shown in Table 3 .

VGG-CNN-M-1024 Faster R-CNN (Simonyan & Zisserman, 2014 ) is the baseline.

Similar to MS-COCO, the bounding box annotations are visible boxes instead of full boxes used in CrowdHuman.

We observe that Gaussian improve the mAP (mean Average Precision) by 1.5%.

The complete model improves the mAP by 2.0%.

Multimodality and multivariate Gaussian do not substantially improve the performance.

These observations coincide with the observations on MS-COCO.

Comparison with State-of-the-art: Shown in Table 4 , we compare multivariate mixture of eight Gaussians model to DeepVoting Zhang et al. (2018c) on VehicleOcclusion.

Similar to CrowdHuman, the bounding box annotations are full boxes.

The baseline is VGG-16 Faster R-CNN.

Our multivariate mixture of eight Gaussians model outperforms DeepVoting by a large margin at different occlusion levels.

Without occlusion, our model also helps to learn a better detector, coinciding the experiments above.

We argue that our model considers multiple modes of an object and the correlations between each border of a bounding box, which helps detection under occlusion.

Model Size and Inference Speed: We measure the inference speed of our models using ResNet-50 FPN Mask R-CNN with a TITAN Xp, CUDA 10.1 and cuDNN 7.5.0 on MS-COCO val2017.

Shown in Table 5 , Gaussian (Eq. 9) and multivariate Gaussian (Eq. 7) neither slow down the inference nor increase the number of parameters, since we can discard the covariance ?? at inference time (Section 3.1).

The complete model, multivariate mixture of eight Gaussians (Eq. 1), only increases 2M parameters and sacrifices 0.9 FPS on GPU.

Our models outperform the baselines by large margins (Table 1 , 2 and 4), while requires almost no additional computation and memory.

Note that we measure the inference latency on MS-COCO where there are 80 classes, such that the number of parameters for ?? is 1024 ?? 80 ?? K (1024 is the number of output channels of fc7, Figure 2 ).

On CrowdHuman where there is only one class (human), the number of parameters for ?? is only 1024 ?? K, which will consume even fewer computation and memory resources.

Figure 4 shows the visualization of our mixture of Gaussian prediction results on CrowdHuman.

When the object is not occluded, our model usually only exhibits a single mode.

In Figure 4 (a), the predictions of the mixture components for the athlete are almost the same.

When the object is occluded, the occluded bounding box border usually exhibits multiple modes.

For example, the left arm of the man can have several reasonable poses in Figure 4 (b).

Figure 5 shows the visualization of our multivariate Gaussian prediction results on CrowdHuman.

When the object is not occluded, like in Figure 5 (a), most terms in the covariance matrix are usually almost zeros.

When a border of the object is occluded, like in Figure 5 (b) , the variance term for that border tends to be very high.

Sometimes our model learns the covariance between bounding box borders.

For example, in Figure 5 (c), x 1 and x 2 has a positive correlation, which suggests if the left border moves right, the right border might also move right.

When the object is heavily occluded, most of its variance terms are usually very high, shown in Figure 5 (d).

We propose a multivariate mixture of Gaussians model for object detection under occlusion.

Quantitatively, it demonstrates consistent improvements over the baselines among MS-COCO, PASCAL VOC 2007, CrowdHuman, and VehicleOcclusion.

Qualitatively, our model enjoys explainability as the detection results can be diagnosed via the covariance matrices and the mixture components.

@highlight

a deep multivariate mixture of Gaussians model for bounding box regression under occlusion
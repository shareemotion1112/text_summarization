Offset regression is a standard method for spatial localization in many vision tasks, including human pose estimation, object detection, and instance segmentation.

However,  if high localization accuracy is crucial for a task, convolutional neural networks will offset regression usually struggle to deliver.

This can be attributed to the locality of the convolution operation, exacerbated by variance in scale, clutter, and viewpoint.

An even more fundamental issue is the multi-modality of real-world images.

As a consequence, they cannot be approximated adequately using a single mode model.

Instead, we propose to use mixture density networks (MDN) for offset regression, allowing the model to manage various modes efficiently and learning to predict full conditional density of the outputs given the input.

On 2D human pose estimation in the wild, which requires accurate localisation of body keypoints, we show that this yields significant improvement in localization accuracy.

In particular, our experiments reveal viewpoint variation as  the dominant  multi-modal factor.

Further, by carefully initializing MDN parameters, we do not face any instabilities in training, which is known to be a big obstacle for widespread deployment of MDN.

The method can be readily applied to any task with a spatial regression component.

Our findings  highlight the multi-modal nature of real-world vision, and the significance of explicitly accounting for viewpoint variation, at least when spatial localization is concerned.

Training deep neural networks is a non-trivial task in many ways.

Properly initializing the weights, carefully tuning the learning rate, normalization of weights or targets, or using the right activation function can all be vital for getting a network to converge at all.

From another perspective, it is crucial to carefully formulate the prediction task and loss on top of a rich representation to efficiently leverage all the features learned.

For example, combining representations at various network depths has been shown to be important to deal with objects at different scales Newell et al. (2016) ; Lin et al. (2017) ; Liu et al. (2016) .

For some issues, it is relatively straightforward to come up with a network architecture or loss formulation to address them -see e.g. techniques used for multi-scale training and inference.

In other cases it is not easy to manually devise a solution.

For example, offset regression is extensively used in human pose estimation and instance segmentation, but it lacks high spatial precision.

Fundamental limitations imposed by the convolution operation and downsampling in networks, as well as various other factors contribute to this -think of scale variation, variation in appearance, clutter, occlusion, and viewpoint.

When analyzing a standard convolutional neural network (CNN) with offset regression, it seems the network knows roughly where a spatial target is located and moves towards it, but cannot get precise enough.

How can we make them more accurate?

That's the question we address in this paper, in the context of human pose estimation.

Mixture density models offer a versatile framework to tackle such challenging, multi-modal settings.

They allow for the data to speak for itself, revealing the most important modes and disentangling them.

To the best of our knowledge, mixture density models have not been successfully integrated in 2D human pose estimation to date.

In fact, our work has only become possible thanks to recent work of Zhou et al. (2019a) proposing an offset based method to do dense human pose estimation, object detection, depth estimation, and orientation estimation in a single forward pass.

Essentially, in a dense fashion they classify some central region of an instance to decide if it belongs to a particular category, and then from that central location regress offsets to spatial points of interest belonging to the instance.

In human pose estimation this would be keypoints; in instance segmentation it could be extreme points; and in tracking moving objects in a video this could be used to localize an object in a future frame Zhou et al. (2019b) ; Neven et al. (2019) ; Novotny et al. (2018) ; Cui et al. (2019) .

This eliminates the need for a two stage top-down model or for an ad hoc post processing step in bottom-up models.

The former would make it very slow to integrate a density estimation method, while for the latter it is unclear how to do so -if possible at all.

In particular, we propose to use mixture density networks (MDN) to help a network disentangle the underlying modes that, when taken together, force it to converge to an average regression of a target.

We conduct experiments on the MS COCO human pose estimation task Lin et al. (2014) , because its metric is very sensitive to spatial localization: if the ground truth labels are displaced by just a few pixels, the scores already drop significantly, as shown in top three rows of Table 4 .

This makes the dataset suitable for analyzing how well different models perform on high precision localization.

Any application demanding high precision localization can benefit from our approach.

For example, spotting extremely small broken elements on an electronic board or identifying surface defects on a steel sheet using computer vision are among such applications.

In summary, our contributions are as follows:

??? We propose a new solution for offset regression problems in 2D using MDNs.

To the best of our knowledge this is the first work to propose a full conditional density estimation model for 2D human pose estimation on a large unconstrained dataset.

The method is general and we expect it to yield significant gains in any spatial dense prediction task.

??? We show that using MDN we can have a deeper understanding of what modes actually make a dataset challenging.

Here we observe that viewpoint is the most challenging mode that forces a single mode model to settle down for a sub-optimal solution.

Multi-person human pose estimation solutions usually work either top-down or bottom-up.

In the top-down approach, a detector finds person instances to be processed by a single person pose estimator Newell et al. (2016); .

When region-based detectors Girshick et al. (2014) are deployed, top-down methods are robust to scale variation.

But they are slower compared to bottom-up models.

In the bottom-up approach, all keypoints are localized by means of heatmaps Cao et al. (2017) , and for each keypoint an embedding is learned in order to later group them into different instances.

Offset based geometric Cao et al. (2018; ; Papandreou et al. (2018) and associative Newell et al. (2017) embeddings are the most successful models.

However, they lead to inferior accuracy and need an ad hoc post-processing step for grouping.

To overcome these limitations, recently Zhou et al. (2019a) proposed a solution that classifies each spatial location as corresponding to (the center of) a person instance or not and at the same location generates offsets for each keypoint.

This method is very fast and eliminates the need for a detector or post-processing to group keypoints.

In spirit, it is similar to YOLO and SSD models developed for object detection Redmon et al. (2016) ; Liu et al. (2016) .

However, offset regression does not deliver high spatial precision and the authors still rely on heatmaps to further refine the predictions.

Overcoming this lack of accuracy is the main motivation for this work.

As for the superiority of having multiple choice solutions for vision tasks, Guzman-Rivera et al.

(2012); Lee et al. (2015; ; Rupprecht et al. (2017) have shown that having multiple prediction heads while enforcing them to have diverse predictions, works better than a single head or an ensemble of models.

However, they depend on an oracle to choose the best prediction for a given input.

The underlying motivation is that the system later will be used by another application that can assess and choose the right head for an input.

Clearly this is a big obstacle in making such models practical.

And, of course, these methods do not have a mechanism to learn conditional density of outputs for a given input.

This is a key feature of mixture models.

Mixture density networks Bishop (1994) have attracted a lot of attention in the very recent years.

In particular, it has been applied to 3D human pose estimation Li & Lee (2019) , and 3D hand pose estimation Ye & Kim (2018) .

Both works are applied to relatively controlled environments.

In 2D human pose estimation, Rupprecht et al. (2017)

We first review the mixture density networks and then show how we adapt it for offset regression.

Mixture models are theoretically very powerful tools to estimate the density of any distribution McLachlan & Basford (1988) .

They recover different modes that contribute to the generation of a dataset, and are straightforward to interpret.

Mixture density networks (MDN) Bishop (1994) is a technique that enables us to use neural networks to estimate the parameters of a mixture density model.

MDNs estimate the probability density of a target conditioned on the input.

This is a key technique to avoid converging to an average target value given an input.

For example, if a 1D distribution consists of two Gaussians with two different means, trying to estimate its density using a single Gaussian will result in a mean squashed in between the two actual means, and will fail to estimate any of them.

This effect is well illustrated in Figure 1 of the original paper by Bishop Bishop (1994).

In a regression task, given a dataset containing a set of input vectors as {x 0 . . .

x n } and the associated target vectors {t 0 . . .

t n }, MDN will fit the weigths of a neural network such that it maximizes the likelihood of the training data.

The key formulation then is the representation of the probability density of the target values conditioned on the input, as shown in equation 1:

Here M is a hyper-parameter and denotes the number of components constituting the mixture model.

?? m (x i ) is called mixing coefficient and indicates the probability of component m being responsible for generation of the sample x i .

?? m denotes the probability density function of component m for t i |x i .

The conditional density function is not restricted to be Gaussian, but that is the most common choice and works well in practice.

It is given in equation 2:

In equation 2, c is the dimension of the target vector, ?? m is the component mean and ?? m is the common variance for the elements of the target.

The variance term does not have to be shared between dimensions of target space, and can be replaced with a diagonal or full co-variance matrix if necessary Bishop (1994).

Given an image with an unspecified number of people in uncontrolled poses, the goal of human pose estimation is to localize a predefined set of keypoints for each person and have them grouped together per person.

We approach this problem using a mixed bottom-up and top-down formulation very recently proposed in Zhou et al. (2019a) .

In this formulation, unlike the top-down methods there is no need to use an object detector to localize the person instance first.

And unlike bottom-up methods, the grouping is not left as a post-processing step.

Rather, at a given spatial location, the model predicts if it is the central pixel of a person, and at the same location, for each keypoint it generates an offsets vector to the keypoint location.

This formulation takes the best of both approaches: it is fast like a bottom-up method, and postprocessing free as in a top-down model.

At least equally important is the fact that it enables applying many advanced techniques in an end-to-end manner.

As a case in point, in this paper it allows us to perform density estimation for human pose in a dense fashion.

That is, in a single forward pass through the network, we estimate the parameters of a density estimation model.

In particular, we use the mixture density model to learn the probability density of poses conditioned on an input image.

Formally, Zhou et al. (2019a) start from an input RGB image I of size H * W * 3, and a CNN that receives I and generates an output with height H , width W , and C channels.

If we indicate the downsampling factor of the network with D, then we have H = D * H , and similarly for width.

We refer to the set of output pixels as P .

Given the input, the network generates a dense 2D classification map C to determine instance centers, i.e. C p indicates the probability of location p ??? P corresponding to the center of a person instance.

Simultaneously, at p , the network predicts

, where K is the number of keypoints that should be localized (17 in the COCO dataset).

Once the network classifies p as a person's central pixel, the location of each keypoint is directly given by the offset vectors O.

In the literature, it is common to train for offset regression O using L 1 loss Papandreou et al. (2018) ; Kreiss et al. (2019) ; Cao et al. (2017) ; Zhou et al. (2019a) .

However, spatial regression is a multimodal task and having a single set of outputs will lead to a sub-optimal prediction, in particular when precise localization is important.

With this in mind, we use mixture density networks to model the offset regression task.

In this case, ?? m from equation 2 would be used to represent offsets predicted by different components.

Then the density of the ground truth offset vectors G conditioned on image I is given by equation 3, where the density ?? m for each component is given by equation 4.

Here O m (I) is the input dependent network output function for component m that generates offsets and

p,y ] indicates the ground truth offsets.

?? m (I) is the standard deviation of the component m in two dimensions, X and Y. It is shared by all keypoints of an instance.

However, in order to account for scale differences of keypoints, in equation 4 for each keypoint we divided ?? m (I) by its scale factor provided in COCO dataset.

In this framework, the keypoints are independent within each component, but the full model does not assume such independence.

Given the conditional probability density of the ground truth in equation 3, we can define the loss using the negative log likelihood formulation and minimize it using stochastic gradient descent.

The loss for MDN is given in equation 5.

Here N is the number of samples in the dataset.

Practically, this loss term replaces the popular L 1 loss.

Please note that MDN is implemented in a dense fashion, that density estimation is done independently at each spatial location p ??? P .

A schematic overview of the model is shown in Figure 1 .

We do not modify the other loss terms used in Zhou et al. (2019a) .

This includes a binary classification loss L C , a keypoint heatmap loss L HM (used for refinement), a small offset regression loss to compensate for lost spatial precision due to downsampling for both center and keypoints L C of f and L KP of f , and a loss for instance size regression L wh .

The total loss is given in equation 6:

Once the network is trained, at each spatial location, C will determine if that is the center of a person (the bounding box center is used for training).

Each MDN component at that location will generate Figure 1: Schematic overview of our proposed solution using mixture density networks.

offsets conditioned on the input.

To obtain the final offset vectors, we can use either the mixture of the components or the component with the highest probability.

We do experiments with both and using the maximum component leads to slightly better results.

Once we visually investigate what modes the components have learned, they seem to have very small overlap.

Hence, it is not surprising that both approaches have similar performance Bishop (1994) .

Clevert et al. (2015) , but modify it such that minimum values is 10.

We did experiment with smaller and larger values for minimum, but did not observe any significant difference.

To avoid numerical issues, we have implemented the log likelihood using the LogSumExp function.

Our implementation is on top of the code base published by Zhou et al. (2019a) , and we use their model as the base in our comparisons.

The network architecture is based on a version of stacked hourglass Newell et al. (2016) presented in Law & Deng (2018) .

We refer to this architecture as LargeHG.

To analyse effect of model capacity, we also conduct experiments with smaller variants.

The SmallHG architecture is obtained by replacing the residual layers with convolutional layers, and XSmallHG is obtained by further removing one layer from each hourglass level.

Unless stated otherwise, all models are trained for 50 epochs (1X schedule) using batch size 12 and ADAM optimizer Kingma & Ba (2014) with learning rate 2.5e-4.

Only for visualization and comparison to state-of-the-art we use a version of our model trained for 150 epochs (3X).

Except for comparison with the state-of-the-art, we have re-trained the base model to assure fair comparison.

To analyse effect of number of components, we train on XSmallHG and SmallHG architectures with up to 5 components, and on LargeHG architecture with up to 3 components.

Table 1 shows the evaluation results for various models on the coco-val.

The table also shows the evaluation for MDN models when we ignore the predictions by particular components.

We report predictions with and without using the extra heatmap based refinement deployed in Zhou et al. (2019a) .

This refinement is a post-processing step, which tries to remedy the lack of enough precision in offset regression by pushing the detected keypoints towards the nearest detection from the keypoint heatmaps.

It is clear that MDN leads to a significant improvement.

Interestingly, only two modes will be retrieved, no matter how many components we train and how big the network is.

Having more than two components results in slightly better recall, but it will not improve precision.

Only when the network capacity is very low more than two component seems to have significant contribution Visualizing prediction by various models, makes it clear that one of the modes focuses on frontal view instances, and the other one on the instance with backward view.

Figure 2 shows sample visualisation from M DN 3 model trained with 3X schedule.

We further evaluate the M DN 2 trained on the LargeHG on various subsets of the COCO validation split by ignoring predictions by each of components or forcing all predictions to be made by a particular component.

The detailed evaluations are presented in table 2.

The results show that the components correlate well with face viability, confirming the conclusion we make by visualising predictions.

It is worth noting that although we use annotation of nose as indicator of face visibility, it is noisy, as in some case the person is view from side such that nose is just barely visible and the side view is very close to back view (like the first image in the third row of Figure 2 ).

But, even this noisy split is enough to show that two modes are chose based on viewpoint.

Table 1 ).

The full coco validation split Visible Keypoints

All keypoints that are occluded and annotated are ignored Occluded Keypoints

All keypoints that are visible and annotated are ignored Visible Face

Instances with at least 5 annotated keypoints where nose is visible and annotated

Instances with at least 5 annotated keypoints where nose is occluded or not annotated Table 3 : Statistics for coco-val subsets and MDN max component.

For face visibility, instances with more than 5 annotated keypoints (in parentheses for minimum of 10) are used.

For components, predictions with score at least .5 are considered (in parentheses for minimum of .7).

Occluded Keypoints Visible Keypoints Occluded Face Visible Face comp1 comp1 comp1 (back view) comp2 comp2 comp2 (front view) Table 3 compares portion of the dataset each subset comprises against portion of predictions made by each component of M DN 2 .

Obviously, the component statistics correlates well with the face visibility, which in fact is an indicator of viewpoint in 2D.

Majority of instances in the dataset are in frontal view, and similarly the front view component makes majority of the prediction.

Related to our results, Belagiannis & Zisserman (2017) have shown that excluding occluded keypoints from training (by treating them as background) leads to improved performance.

More recently, Ye & Kim (2018) achieves more accurate 3D habd pose estimation by proposing a model that directly predicts occlusion of a keypoint in order to use it for selecting a downstream model.

And, here we illustrate that occlusion caused by viewpoint imposes more challenge to spatial regression models, than other possible factors, like variation in pose itself.

It is common to train offset regression targets with L1 loss Zhou et al. (2019a); Papandreou et al. (2018); Law & Deng (2018) .

In contrast, the single component version of our model is technically equal to L2 loss normalized by the instance scale learned via MDN variance terms.

This is in fact equal to directly optimizing the MS COCO OKS scoring metric for human pose estimation.

Comparing the performance of the two losses in Table 1 , we see normalized L2 yields superior results.

That is, for any capacity, M DN 1 outperforms the base model which is trained using L1.

For a deeper insight on what body parts gain the most from the MDN, we do fine grained evaluation for various keypoint subsets.

In doing so, we modify the COCO evaluation script such that it only considers set of keypoints we are interested in.

Table 4 shows the results.

For the facial keypoints where the metric is the most sensitive, the improvement is higher.

nevertheless, the highest improvement comes for the wrists, which have the highest freedom to move.

On the other hand, for torso keypoints (shoulders and hips) which are the most rigid, there is almost no different in comparison to base model.

Given that MDN revels two modes, we build a hierarchical model by doing a binary classification and using it to choose from two separate full MDN models.

The goal is to see, if binary classification .

Therefore, a two component MDN that learns full conditional probability density and assumes dependence between target dimensions delivers higher performance.

For comparison to the state-of-the-art in human pose estimation, we train M DN 1 and M DN 3 for 150 epochs using the LaregHG architecture, and test it on COCO test-dev split.

The results are presentetd in Table 5 .

Using MDN significantly improves the offset regression accuracy (row 6 vs row 10 of the table).

When refined, both models achieve similar performance.

In contrast to all other state-of-the-art models, MDNs performance drops if we deploy the ad-hoc left-right flip augmentation at inference time.

This is a direct consequence of using a multi-modal prediction model which learns to deal with viewpoint.

It is important to note that left-right flip is used widely for increasing accuracy at test time for object detection and segmentation tasks as well.

Therefore, we expect our method to improve performance for those tasks as well.

M DN 1 with refinement gives slightly lower accuracy than the base model.

Our investigation attributes this discrepancy to a difference in the training batch size.

The official base model is trained with batch size 24, but we train all models with batch size 12, due to limited resources.

Under the same training setting, M DN 1 will outperform the base model, as shown in Table 1 .

We have shown mixture density models significantly improve spatial offset regression accuracy.

Further, we have demonstrate that MDNs can be deployed on real world data for conditional density estimation without facing mode collapse.

Analyzing the ground truth data and revealed modes, we have observe that in fact MDN picks up on a mode, that significantly contributes to achieving higher accuracy and it can not be incorporated in a single mode model.

In the case of human pose estimation, it is surprising that viewpoint is the dominant factor, and not the pose variation.

This stresses the fact that real world data is multi-modal, but not necessarily in the way we expect.

Without a principled approach like MDNs, it is difficult to determine the most dominant factors in a data distribution.

A stark difference between our work and others who have used mixture models is the training data.

Most of the works reporting mode collapse rely on small and controlled datasets for training.

But here we show that when there is a large and diverse dataset, just by careful initialization of parameters, MDNs can be trained without any major instability issues.

We have made it clear that one can actually use a fully standalone multi-hypothesis model in a real-world scenario without the need to rely on an oracle or postponing model selection to a downstream task.

We think there is potential to learn more finer modes from the dataset, maybe on the pose variance, but this needs further research.

Specially, it will be very helpful if the role of training data diversity could be analysed theoretically.

At the same time, the sparsity of revealed modes also reminds us of the sparsity of latent representations in generative models Xu et al. (2019) .

We attribute this to the fact that deep models, even without advanced special prediction mechanism, are powerful enough to deliver fairly high quality results on the current datasets.

Perhaps, a much needed future direction is applying density estimation models to fundamentally more challenging tasks like the very recent large vocabulary instance segmentation task Gupta et al. (2019) .

@highlight

We use mixture density networks to do full conditional density estimation for spatial offset regression and apply it to the human pose estimation task.
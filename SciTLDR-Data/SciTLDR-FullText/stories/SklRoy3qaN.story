We introduce a systematic framework for quantifying the robustness of classifiers to naturally occurring perturbations of images found in videos.

As part of this framework, we construct ImageNet-Vid-Robust, a human-expert--reviewed dataset of 22,668 images grouped into 1,145 sets of perceptually similar images derived from frames in the ImageNet Video Object Detection dataset.

We evaluate a diverse array of classifiers trained on ImageNet, including models trained for robustness, and show a median classification accuracy drop of 16\%.

Additionally, we evaluate the Faster R-CNN and R-FCN models for detection, and show that natural perturbations induce both classification as well as localization errors, leading to a median drop in detection mAP of 14 points.

Our analysis shows that natural perturbations in the real world are heavily problematic for current CNNs, posing a significant challenge to their deployment in safety-critical environments that require reliable, low-latency predictions.

Despite their strong performance on various computer vision benchmarks, convolutional neural networks (CNNs) still have many troubling failure modes.

At one extreme,`padversarial examples can cause large drops in accuracy for state of the art models with visually imperceptible changes to the input image BID4 .

But since carefully crafted`pperturbations are unlikely to occur naturally in the real world, they usually do not pose a problem outside a fully adversarial context.

To study more realistic failure modes, researchers have investigated benign image perturbations such as rotations & translations, colorspace changes, and various image corruptions [7, 8, 4] .

However, it is still unclear whether these perturbations reflect the robustness challenges commonly arising in real data since the perturbations also rely on synthetic image modifications.

Recent work has therefore turned to videos as a source of naturally occurring perturbations of images [6, BID0 .

In contrast to other failure modes, the perturbed images are taken from existing image data without further modifications that make the task more difficult.

As a result, robustness to such perturbations directly corresponds to performance improvements on real data.

However, it is currently unclear to what extent such video perturbations pose a significant robustness challenge.

Azulay and Weiss BID0 only provide anecdotal evidence from a small number of videos.

While [6] work with a larger video dataset to obtain accuracy estimates, they only observe a small drop in accuracy of around 2.7% on videoperturbed images, suggesting that small perturbations in videos may not actually reduce the accuracy of current CNNs significantly.

We address this question by conducting a thorough evaluation of robustness to natural perturbations arising in videos.

As a cornerstone of our investigation, we introduce ImageNet-Vid-Robust, a carefully curated subset of ImageNet-Vid [12] .

In contrast to earlier work, all images in ImageNet-Vid-Robust were screened by a set of expert labelers to ensure a high annotation quality and to minimize selection biases that arise when filtering with CNNs.

Overall, ImageNet-Vid-Robust contains 22,668 images grouped into 1,145 sets of temporally adjacent and visually similar images of a total of 30 classes.

We then utilize ImageNet-Vid-Robust to measure the accuracy of current CNNs to small, naturally occurring perturbations.

Our testbed contains over 40 different model types, varying both architecture and training methodology (adversarial training, data augmentation, etc).

We find that natural perturbations from ImageNet-Vid-Robust induce a median 16% accuracy drop for classification tasks and a median 14% drop in mAP for detection tasks.

Even for the best-performing model, we observe an accuracy drop of 14% -significantly larger than the 2.7% drop in [6] over the same time horizon in the video.

Our results show that robustness to natural perturbations in videos is indeed a significant challenge for current CNNs.

As these models are increasingly deployed in safety-critical environments that require both high accuracy and low latency (e.g., autonomous vehicles), ensuring reliable predictions on every frame of a video is an important direction for future work.

The ImageNet-Vid-Robust dataset is sourced from videos contained in the ImageNet-Vid dataset [12], we provide more details about ImageNet-Vid in the supplementary.

Next, we describe how we extracted neighboring sets of naturally perturbed frames from ImageNet-Vid to create ImageNet-Vid-Robust.

A straightforward approach is to select a set of anchor frames and use nearby frames in the video with the assumption that such frames contain only small perturbations from the anchor frame.

However, as FIG0 in the supplementary illustrates, this assumption is frequently broken, especially in the presence of fast camera or object motion.

Instead, we collect a preliminary dataset of natural perturbations and then we manually review each of the frame sets.

For each video, we first randomly sample an anchor frame and then take k = 10 frames before and after the anchor frame as candidate perturbation images.

This results in a dataset containing 1 anchor frame each from 1,314 videos, with approximately 20 candidate perturbation frames each BID0 .Next, we curate the dataset with the help of four expert human annotators.

The goal of the curation step is to ensure that each anchor frame and nearby frame is correctly labeled with the same ground truth class and that the anchor frame and the nearby frame are visually similar.

For each pair of anchor and candidate perturbation frame, an expert human annotator labels (1) whether the pair is correctly labeled in the dataset, (2) whether the pair is similar.

Asking human annotators to label whether a pair of frames is similar can be highly subjective.

We took several steps to mitigate this issue and ensure high annotation quality.

First, we trained reviewers to mark frames as dissimilar if the scene undergoes any of the following transformations: (1) significant motion, (2) significant background change, or (3) significant blur change, and additionally asked reviewers to mark each of the dissimilar frames with one of these transformations, or "other".

Second, as presenting videos or groups of frames to reviewers could cause them to miss potentially large changes due to the well-studied phenomenon of change blindness [9], we present only a single pair of frames at a time to reviewers.

Finally, to increase consistency in annotation, human annotators proceed using two rounds of review.

In the first round, all annotators were given identical labeling instructions, and then individually reviewed 6500 images pairs.

We instructed annotators to err on the side of marking a pair of images as dissimilar if a BID0 Note that some anchor frames may have less than 20 candidate frames if the anchor frame is near the start or end of the video.

distinctive feature of the object is only visible in one of the two frames (such as the face of a dog).

If an annotator was unsure about a pair he or she could mark the pair as "don't know".For the second round of review, all annotators jointly reviewed all frames marked as dissimilar, "don't know" or "incorrect".

A frame was only considered similar if a strict majority of the annotators marked the pair of as "similar".After the reviewing was complete, we discarded all anchor frames and candidate perturbations that annotators marked as dissimilar or incorrectly labeled.

Our final dataset contains 1,145 anchor frames with a minimum of 1, maximum of 20 and median of 20 similar frames.

Given the dataset above, we would like to measure a model's robustness to natural perturbations.

In particular, let A = {a 1 , ..., a n } be the set of valid anchor frames in our dataset.

Let Y = {y 1 , ..., y n } be the set of labels for A. We let N k (a i ) be the set of frames marked as similar to anchor frame a i .

In our setting N k is a subset of the 2k temporally adjacent frames (plus/minus k frames from anchor).The pm-k analogues of the standard metrics for detection and classification evaluate only on the worst-case frame in the set of N k .

We formally define the pm-k analogues for the standard metrics for classification and detection (acc pmk and mAP pmk ) in the supplementary.

We evaluate a testbed of 50 classification models and 3 state of the art detection models on ImageNet-Vid-Robust.

We first discuss the various types of classification models evaluated with pm-k classification metric.

We then study the per-class accuracies to study whether our perturbations exploits a few "hard" classes or affects performance uniformly across classes.

Second we use the bounding box annotations inherited from ImageNet-VID to study the effect of detection models evaluated on ImageNet-Vid-Robust using the pm-k metric.

We then analyze the errors made on the detection adversarial examples to isolate the effects of localization errors vs classification errors.

In FIG1 , we plot acc orig versus acc pmk for all classification models in our test bed and find that there is a surprisingly linear relationship between acc orig and acc pmk across all 48 models in our test bed.

We note the similarity of this plot to FIG0 in BID12 .1578 out 22668 frames in ImageNet-Vid-Robust have multiple correct classification labels, due to multiple objects in the frame.

To handle this in a classification set- Each data point corresponds to one model in our testbed (shown with 95% Clopper-Pearson confidence intervals).

Each "perturbed" frame was taken from a neighborhood of a maximum 10 adjacent frames to the original frame in a 30 FPS video.

This allows the scene to change for roughly 0.3s.

All frames were reviewed by humans to confirm visual similarity to the original frames.ting, we opt for the most conservative approach: we count a prediction as correct if the model predicts any of the classes for a frame.

We note that this is a problem that plagues many classification datasets, where objects of multiple classes can be in an image BID12 but there is only one true label.

We considered 5 models types of increasing levels of supervision.

We present our full table of classification accuracies in the supplementary material, and results for representative models from each model type in Table 1 .ILSVRC Trained As mentioned in ??, leveraging the WordNet hierarchy enables evaluating models available from [2] trained on the 1000 class ILSVRC challenge on images in ImageNet-Vid-Robust directly.

We exploit this to evaluate a wide array of model architectures against our natural perturbations.

We note that this test set is a substantial distribution shift from the original ILSVRC validation set that these models are tuned for.

Thus we will expect the benign accuracy acc orig to be lower than the comparable accuracy on the ILSVRC validation set.

However the quantity of interest for this experiment is the difference between the original and perturbed accuracies accuracies acc origacc pmk , which should be less sensitive to an absolute drop in acc orig .ILSVRC Trained with Noisy Augmentation One hypothesis for the accuracy drop is that subtle artifacts and corruptions introduced by video compression schemes could introduce a large accuracy drop when evaluated on these corrupted frames.

The worst-case nature of the pm-k metric could be biasing evaluation towards these corrupt frames.

One model for these corruptions are the perturbations introduced in [7] .

To test this hypothesis we evaluate models augmented with a subset of the perturbations (Gaussian noise Gaussian blur, shot noise, contrast change, impulse noise, JPEG compression) found in [7] .

We found that this augmentation scheme did little to help robustness against our perturbations.

ILSVRC Trained for L 2 /L 1 Robustness We evaluate the best performing robust model against the very strong L 2 /L 1 attacks [14].

We find that this model does have a slightly smaller performance drop than both ILSVRC and ILSVRC trained with noise augmentation but the difference is well within the error bars induced by the small size of our evaluations set.

We also note that this robust model gets significantly lower original and perturbed accuracy than examples from either of the model types above.

ILSVRC Trained + Finetuned on ImageNet-VID To adapt to the 30 class problem and the different domain of videos we fine tune several network architectures on the training set in ImageNet VID.

We start with a base learning rate of 1e 4 and train with the SGD optimizer until the validation accuracy plateaus.

We trained using cross entropy loss using the largest object in the scene as the label during training, as we found this performed better than training using a multi-label loss function.

After training for 10 epochs we evaluate on ImageNet-Vid-Robust.

These models do improve in absolute accuracy over their ILSVRC pretrained counterparts (12% for a ResNet50).

However, this improvement in absolute accuracy does not significantly decrease the accuracy drop induced by natural perturbations.

Finally, we analyze whether additional supervision, in the form of bounding box annotations, improves robustness.

To this end, we train the Faster R-CNN detection model [11] with a ResNet 50 backbone on ImageNet Vid.

Following standard practice, the detection backbone is pre-trained on ILSVRC.

To evaluate this detector for classification, we assign the score for each label for an image as the score of the most confident bounding box for that label.

We find that this transformation reduces accuracy compared to the

To analyze the generalizability of natural perturbations to other tasks, we next analyze their impact on the object localization and detection tasks.

We report results for two related tasks: object localization and detection.

Object detection is the standard computer vision task of correctly classifying an object and regressing the coordinates of a tight bounding box containing the object.

"

Object localization", meanwhile, refers to the only the subtask of regressing to the bounding box, without attempting to correctly classify the object.

This is an important problem from a practical perspective (for example, the size and location of an obstacle may be more important for navigation than the category), as well as from an analytical perspective, as it allows analyzing mistakes orthogonal to classification errors.

For example, it may be the case that natural perturbations cause misclassification errors frequently, as it may be natural to mistake a cat for a fox, but cause few localization errors.

We present our results using the popular Faster R-CNN [11] and R-FCN [3, 13] architectures for object detection and localization in TAB2 .

We first note the significant drop in mAP of 12 15% for object detection due to perturbed frames for both the Faster R-CNN and R-FCN architectures.

Next, we show that localization is indeed easier than detection, as the mAP increases significantly (e.g., from 61.8 to 75.5 for Faster R-CNN with ResNet 50 backbone).

Perhaps surprisingly, however, switching to the localization task does not improve the delta between original and perturbed frames, indicating that natural perturbations induce both classification and localization errors.

An advantage of using the ImageNet-Vid dataset as the source of our dataset is that all 30 object Anchor frame Discarded frame Anchor frame Anchor frame Discarded frame Discarded frame FIG0 : Temporally adjacent frames may not be visually similar.

We visualize three randomly sampled frame pairs where the nearby frame was marked during human review as "dissimilar" to the anchor frame and discarded from our dataset.

Classification accuracy is defined as: DISPLAYFORM0 Submitted to 33rd Conference on Neural Information Processing Systems (NeurIPS 2019).

Do not distribute.

as: DISPLAYFORM1 Which simply corresponds to picking the worst frame from the each N k (a i ) set before computing 20 misclassification accuracy.

Detection The standard metric for detection is mean average precision of the predictions at a fixed ).

We define the pm-k analog of mAP by replacing each anchor frame in the dataset with a nearby 30 frame that minimizes the per-image average precision.

Note that as the category-specific average 31 precision is undefined for categories not present in an image, we minimize the average precision 32 across categories for each frame rather than the mAP.

We then define the pm-k mAP as follows, with 33 a slight abuse of notation to denote y b as the label for frame b: DISPLAYFORM0

In FIG1 , we plot the relationship between perturbed accuracy and and perturbation distance (i.e the 36 k in the pm-k metric described in Section 3).

We note that the entire x-axis in FIG1 corresponds 37 to a temporal distance of 0s to 0.3s between the original and perturbed frames.

We study the effect of our peturbations on the 30 classes found in ImageNet-Vid-Robust to 40 determine whethre our performance drop was concentrated in a few "hard" classes.

"artificial" nature of L p attacks, recent work has proposed more realistic modifications to images.

Engstrom et.

al. BID4 study an adversary that performs minor rotations and translations of the input,

Hosseni et.

al. BID12 1e-5 for all models.

We additionally detail hyperparameters for detection models in

@highlight

We introduce a systematic framework for quantifying the robustness of classifiers to naturally occurring perturbations of images found in videos.
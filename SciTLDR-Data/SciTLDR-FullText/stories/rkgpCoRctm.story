The ability to detect when an input sample was not drawn from the training distribution is an important  desirable property of deep neural networks.

In this paper, we show that a simple ensembling of first and second order deep feature statistics can be exploited to effectively differentiate in-distribution and out-of-distribution samples.

Specifically, we observe that  the mean and standard deviation within feature maps  differs greatly between in-distribution and out-of-distribution samples.

Based on this observation, we propose a simple and  efficient plug-and-play detection procedure that does not require re-training, pre-processing or changes to the model.

The proposed method outperforms the state-of-the-art by a large margin in all standard benchmarking tasks, while being much simpler to implement and execute.

Notably, our method improves the true negative rate from 39.6% to 95.3% when 95% of in-distribution (CIFAR-100) are correctly detected using a DenseNet and the out-of-distribution dataset is TinyImageNet resize.

The source code of our method will be made publicly available.

In the past few years, deep neural networks (DNNs) BID6 have settled as the state-of-the art-techniques in many difficult tasks in a plurality of domains, such as image classification BID18 , speech recognition BID9 , and machine translation BID2 BID28 .

This recent progress has been mainly due to their high accuracy and good generalization ability when dealing with realworld data.

Unfortunately, DNNs are also highly confident when tested against unseen samples, even if the samples are vastly different from the ones employed during training BID12 .

Moreover, several works have shown that such deep networks are easily fooled by minor perturbations to the input BID5 BID27 .

Obtaining a calibrated confidence score from a deep neural network is a problem under continuous investigation BID12 ) and a major thread in artificial intelligence (AI) safety BID1 .

In fact, knowing when the model is wrong or inaccurate has a direct impact in many production systems, such as self-driving cars, authentication and disease identification BID0 BID7 , to name a few.

BID8 showed that despite producing significantly low classification errors, DNNs confidence scores are not faithful estimates of the true certainty.

Their experiments confirmed that depth, width, weight decay, and batch normalization are the main reasons for overconfident scores.

Moreover, they demonstrated that a simple and yet powerful method of temperature scaling in the softmax scores is an effective way to improve calibrate a DNNs confidence.

While calibrating the classifier's output to represent a faithful likelihood from the training (in-distribution) data has effective solutions, the problem of detecting whether or not the samples are generated from a known distribution (out-of-distribution), is still an open problem BID12 .One straightforward approach to calibrate the classifier's confidence in order to detect samples whose distribution differs from the training samples distribution is to train a secondary classifier that digests both in-distribution (ID) and out-of-distribution (OOD) data so that anomalies are scored differently from ID samples, as performed in BID13 .

Re-training a network, however, can be computationally intensive and may even be intractable, since the number of OOD samples is virtually infinite.

Other solutions rely on training both classification and generative neural net- Table 1 : Summary comparison of the characteristics of recent related methods.

Test complexity refers to the required number of passes over the network.

Training data is the number of samples for which the methods were calibrated against (with all standing for the whole training set).

AUROC is the area under receiver characteristic curve (detailed in Section 4).

Performance shown is for DenseNet trained on CIFAR-100 and using TinyImageNet (resized) as OOD dataset.

Input pre-proc.

Training data AUROC BID12 works for OOD using a multi-task loss BID20 , or using unsupervised fully convolutional networks as done by BID26 to detect OOD in video samples.

All these methods, however, have a major drawback: they require re-training a modified model using a different loss function possibly with additional parameters, which increases the computational burden, and demands access to the entire original (and probably huge) training data.

In this work, we propose a novel OOD sample detection method that explores low-level statistics from feature layers.

The statistics are obtained directly from the batch normalization layers BID16 , requiring no extra computations during training time, no changes to the network architecture and loss functions, nor preprocessing of the input image.

During test time, the proposed method extracts statistics from selected layers and combines them into an OOD-ness score via a linear classifier.

Throughout this paper, we observe that the mean and standard deviation of a given channel in a layer differ greatly between ID and OOD samples, which naturally motivates their use as features to be employed by an OOD detector.

By selecting the BN layers of a network, we are able to normalize the features according to the learned batch statistics.

The effectiveness of the proposed method is validated in two state-of-the-art DNN architectures (DenseNet and ResNet) BID15 BID10 BID25 BID32 that are trained for image classification tasks in popular datasets.

The proposed approach achieves state-of-the-art performance, surpassing all competitors by a large margin in all tested scenarios, while being much more efficient.

Notably, our method only requires one forward pass while BID22 ; BID21 ; BID29 require two forward and one backward passes.

The rest of the paper is organized as follows.

Section 2 describes prior work on OOD samples detection.

Section 3 introduces the proposed method, whereas Section 4 details all experiments and compares the results with state-of-the-art methods.

Finally, we draw our conclusions in Section 5.

In this section, we describe recent prior work on OOD detection methods.

A summary of all methods described can be seen in Table 1 .Hendrycks & Gimpel (2017) proposed a baseline method based on the posterior distribution (i.e. softmax scores).

They showed that well-trained models tend to produce higher scores for ID samples than for OOD ones.

Hence their method comprises of applying a threshold on the softmaxnormalized output of a classifier.

If the largest score is below the threshold, then the sample is considered OOD.

et al. (2018) continued the aforementioned line of work and proposed the Out-of-Distribution detector for Neural networks (ODIN), which includes a temperature scaling T ∈ R + * to the softmax classifier as in BID8 .

The authors in ODIN argued that a good manipulation of T eases the separation between in-and out-of-distribution samples.

Allied to that, they also incorporated small perturbations to the input (inspired by BID5 ) whose goal is to increase the softmax score of the predicted class.

BID22 found that this kind of perturbation has a stronger effect on ID samples than on OOD ones, increasing the separation between ID and OOD samples.

ODIN outperforms the baseline method BID12 ) by a fair margin; however, it is three times slower due to the two forward and one backward passes needed to preprocess the input, while BID12 only requires one forward pass.

BID29 describes a novel loss function, called margin entropy loss, over the softmax output that attempts to increase the distance between ID and OOD samples.

During training, they partition the training data itself into ID and OOD by assigning samples labeled as certain classes as OOD and use the different partitions to train an ensemble of classifiers that are then used to detect OOD samples during test time.

They also use the input pre-processing step proposed in BID22 , including temperature scaling.

BID21 is the most recent work on OOD detection that we have knowledge of.

They show that the posterior distribution defined by a generative classifier (under Gaussian discriminant analysis) is equivalent to that of the softmax classifier, and the generative classifier eases the separation between in-and out-of-distribution samples.

The confidence score is defined using the Mahalanobis distance between the sample and the closest class-conditional Gaussian distribution.

They argue that abnormal samples can be better characterized in the DNN feature space rather than the output space of softmax-based posterior distribution as done in previous work (e.g., ODIN).

Samples are pre-processed similarly as done in ODIN, but the confidence score is increased instead of the softmax one.

To further improve the performance, they also consider intermediate generative classifiers for all layers in the network.

The final OOD sample detector is computed as an ensemble of confidence scores, chosen by training a logistic regression on validation samples.

This method also shows remarkable results for detection of adversarial attacks and for incremental learning.

An OOD detector should incorporate information from the training data in a natural manner, without being directly influenced by the loss function, which is intrinsically related to the task which could be well-defined for ID samples but be meaningless for most OOD samples.

Moreover, if the OOD method is more dependent on the training distribution, it should be able to be applied to a wide variety networks, and not be designed specifically for a given architecture.

DISPLAYFORM0 Linear decision function DISPLAYFORM1 stat.

stat.

DISPLAYFORM2 OOD-ness score Figure 1 : An illustration of the complete proposed model.

At each BN layer, we extract the input, normalize it using the running statistics, and compute the first and second order features.

The outputs are fed to a linear decision function to predict if the input sample is out-of-distribution or not.

Our method is based on a very simple observation.

Input samples with different distributions generate statistically distinct feature spaces in a DNN.

In other words, the deep features of an ID sample are distinct from those of an OOD one.

Moreover, when using a normalization scheme, such as BN, the features are normalized by the statistics of the ID batch during training, possibly leading to feature statistics that are more similar to the batch statistics, as depicted in Figure 2 .The main problem then becomes how to summarize the feature space distribution for ID samples in a way that best discriminates between ID and OOD samples.

In this work we show that using the first and second order statistics within each feature map performs remarkably well for this task.

is formalized in Section 3.1.

Finally, the linear classifier used to combine the statistics from the different layers is described in Section 3.2.

In the previous section, we motivated that characterizing the feature-space distributions might lead to a robust OOD detector.

As a first approach, one could model these distributions using a nonparametric method to estimate the distribution of the features for each channel, which requires the computation of sufficient statistics using the training data or using a parametric method to fit the distribution BID3 , which are both computationally intensive.

Here, we propose to use only the mean and standard deviation computed along the spatial dimension for each channel to summarize the per-channel distribution.

As it will be shown, these two statistics are sufficient to distinguish between ID and OOD.

Moreover, because the mean and standard deviation of each channel are normalized by the running mean and variance computed during training by the BN layers BID16 , these statistics can be naturally combined within each layer to produce effective features for OOD detection.

We describe such a procedure in what follows.

Given the l-th BN layer with input tensor X ∈ R C×W ×H , the output BN DISPLAYFORM0 where DISPLAYFORM1 c are the per channel per layer learned scaling and shifting parameters, > 0 is a small constant value for numerical stability, µ DISPLAYFORM2 2 ∈ R + are the mean and variance estimated through a moving average using the batch statistics during training, and DISPLAYFORM3 is the normalized feature tensor.

It is worth noting that the statistics are calculated independently for each channel c at each layer l.

In this paper, we conjecture that low-order statistics computed from either X DISPLAYFORM4 c,i,j can be used to discriminate between ID and OOD samples.

Given the unnormalized input X DISPLAYFORM5 and the features for the normalized tensor Z (l) c,i,j are defined as DISPLAYFORM6 i.e., the normalized mean feature m

Intra-layers aggregation.

The features derived from low-order statistics (equation 3) can be readily used to train a predictor for ID/OOD discrimination.

Of course, if they were produced for every feature map in the network, this would result in a feature vector of very high dimension, typically tens of thousands.

Instead, we propose to combine these features within each BN layer, so that in the end we obtain one pair of features per layer: average mean and average variance.

Taking advantage of the fact that features are approximately normalized by BN's running statistics, we propose to simply average them for all channels within a layer.

Thus, each layer yields the following features, for the normalized case:m DISPLAYFORM0 where C l is the number of channels in layer l. Note that doing this aggregation amounts to computing the mean and standard deviation of all normalized features at the given layer.

Using averages of the low-order statistics could lead to issues in deeper layers, where activations are in general concentrated over fewer number of channels.

In this case, the mean of the statistics over channels might not be an appropriate data reduction function.

Nevertheless, as we show in the experiments section, this did not impact the performance of the proposed method, but more investigation is warranted.

Inter-layers ensemble and final classification.

Using all the features in equation 4, i.e., f = ( DISPLAYFORM1 we fit a simple logistic regression model h(f ; θ) with parameters θ ∈ R 2L+1 .

The parameters of the linear model are learned using a separate validation set formed with ID samples (positive examples) and OOD samples (negative examples).

In this section, we evaluate the effectiveness of the proposed method in state-of-the-art deep neural architectures, such as DenseNet BID15 and Wide ResNet BID32 , on several computer vision benchmark datasets: CIFAR BID17 ), TinyImageNet, a subset of ImageNet BID4 , LSUN BID31 , and iSUN BID30 .

We also use Gaussian noise and uniform noise as synthetic datasets.

This evaluation protocol is the de facto standard in recent OOD detection literature BID12 BID22 BID29 BID21 .

All experiments were performed on four models trained from scratch (each one initialized with a different random seed) for each architecture, to account for variance in the model parameters.

The code to reproduce all results is publicly available 1 .

Datasets:

For backbone training, we use CIFAR-10 and CIFAR-100 datasets which have 10 and 100 classes respectively, both containing 50,000 images in the training set and 10,000 images in the test set.

At test time, the test images from CIFAR-10 (CIFAR-100) are considered as ID (positive) samples.

For OOD (negative) datasets, we test with datasets containing natural images, such as TinyImageNet resize and crop, LSUN resize and crop, and iSUN, as well as synthetic datasets, such as Gaussian/uniform noise, which is the same dataset setup as in BID22 ; BID12 .

This is summarized in TAB1 .

For all datasets, we did the validation/test set split following the procedure in BID22 .

Just for reproducibility, 1000 samples from the test set are separated in a validation set used for fitting the logistic regressor and hyper-parameter tuning.

The remaining samples (unseen for both backbone model and OOD detector) are used for testing.

Backbone training: Following Liang et al. FORMULA1 , we adopt the DenseNet BID15 and Wide ResNet BID32 architectures as our benchmark networks.

For DenseNet, we use depth L = 100, growth rate k = 12, and zero dropout rate (DenseNet-BC-100-12).

For Wide ResNet, we also follow BID22 , with L = 28 and widen factor of 10 (WRN-28-10).

All hyperparameters are identical to their original papers.

All networks were trained using stochastic gradient descent with Nesterov momentum BID25 and an initial learning rate of 0.1.

We train the DenseNet-BC-100-12 for 300 epochs, with batch size 64, momentum 0.9, weight decay of 10 −4 and decay the learning rate by a factor of 10 after epochs 150 and 225.

We train the WRN-28-10 for 200 epochs, with batch size 128, momentum 0.9, weight decay 0.0005, and decay the learning rate by a factor of 10 after epochs 60, 120, and 160.

TAB2 shows each network error rate over 4 independent runs each one initialized with a different random seed.

Logistic regression: The logistic regressor is trained considering only the validation partitions for ID (positive examples) and OOD (negative examples) datasets (see TAB1 ).

Using both mean and standard deviation as input (from equation 4), we have 50 features for WRN-28-10 models, and 198 features for DenseNet-BC-100-12 models.

The training was performed using 5-fold cross-validation with the 2 minimization and the regularization factor being chosen as the best one (according to the 5-folds) among 10 values linearly spaced in the range 10 −4 and 10 4 .Evaluation metrics: To evaluate the proposed method, we use the following metrics:1.

True negative rate (TNR) at 95% true positive rate (TPR).

Let TP, TN, FP, and FN be the true positive, true negative, false positive, and false negative, respectively.

The TNR is defined as TN/(TN+FP) whereas TPR is defined as TP/(TP+FN).

2.

Area under the receiver operating characteristic curve (AUROC).

AUROC is the area under the FPR=1-TNR against TPR curve.

We applied t-SNE (L. van der Maaten, 2008) to visualize our high-dimensional feature space in order to see the similarities between ID/OOD samples.

For this, we used one of the WRN-28-10 models trained with CIFAR-10 as ID dataset.

We fitted the t-SNE using the ID and all OOD validation samples together using both mean and standard deviation features, and the result is shown in FIG3 using a perplexity of 30.

It is clear from the visualization that the proposed features are concentrated around well-defined clusters.

Both synthetic OOD datasets have clear distinct behavior from the natural images ones, and it is straightforward to differentiate them.

TinyImageNet (c), LSUN (c) are similar and have some intersection with TinyImageNet (r), LSUN (r).

Interestingly, the clustering seems to reflect the dataset generation method (resizing or cropping).

Most importantly, one can see that the OOD samples are in different clusters than the ID (CIFAR-10) ones, which indicates that this feature choice is adequate for separating them.

In this section, following other methods in the literature, we adjust the linear classifier for each ID/OOD pair.

That is, for each pair of ID/OOD datasets, a different OOD classifier is trained using their respective validation samples.

We performed a comparison between the proposed method and four recent methods described in Section 2 BID12 BID21 BID22 BID29 .

Since BID21 did not test using Wide ResNet models and the same datasets as in BID22 ; BID29 , here we only show the intersection between them: DenseNet BC 100-12 model, using CIFAR-10 (CIFAR-100) as in-distribution and TinyImageNet (resize) and LSUN (resize) as OOD distribution 2 .

Extended results can be found in the appendix.

For both BID12 and BID22 results, we reimplemented the method using their reference implementation 3,4 .

For ODIN BID22 , we employed the same procedure as described by the authors to tune the methods parameters.

A detailed description of the procedure can be found in Appendix B.1.

For BID21 and BID29 , we use the values presented on their papers TAB1 , respectively).The results are compiled in TAB3 .

Notably, our method outperforms the baseline and ODIN methods by a large margin, and yields better results than BID21 and BID29 in all tested cases without requiring any preprocessing, or changing the backbone model.

In fact, when setting the OOD-ness threshold to obtain 95% TPR, our method is able to correctly detect all OOD samples from the test partition.

We argue that the pairwise fitting scenario presented in the previous section is a limited performance measurement.

In fact, many practical applications have OOD samples that do not come from a single distribution mode, and it might be infeasible to collect data from the many different modes of the OOD distribution (in general, some modes are unknown during training).

A good OOD detector should be able to correctly identify samples from OOD distibutions for which its parameters have not been adjusted to.

With this in mind, we propose a different, harder task in which an OOD detector is fitted to one, or a few, OOD datasets and then it is tested on all OOD datasets available.

We note that this is not a standard practice in previous works, like BID22 .

We begin by evaluating the generalization ability of our detector in some preliminary experiments, which motivate our decisions in the choice of OOD datasets for fitting the model and feature selection.

Selecting Features: We evaluate the individual impact of each of the proposed features (i.e., layers average mean and standard deviation) by comparing the performance of the classifier with different features as inputs.

In this experiment, we fit the linear classifier to a specific OOD validation dataset and test on all of the OOD test datasets available.

Our performance metric is the TNR @ 95% TPR averaged over all the tested datasets.

TAB4 presents the performance of the classifier for the WRN-28-10, averaged of the four available models, with CIFAR-100 as ID dataset.

The classifier fitted using only standard deviation features still achieved very good performance generalizing very well to unknown OOD datasets.

Since we are interested in designing an OOD detector that is able to differentiate ID samples from any OOD sample, all our results from now on are presented using the averaged standard deviations per layer as the only features used to train the linear decision function.

Selecting OOD Dataset: To understand to what extent a classifier trained considering one OOD dataset is able to generalize and detect samples from other OOD datasets, we trained the logistic regression considering as positive examples 1000 samples from the ID dataset (CIFAR-100) and as negative examples 1000 samples of a given OOD dataset, using the WRN-28-10 backbone model.

As motivated in the previous section, we use the averages of normalized standard deviation features (equation 4).

The obtained logistic regressor was then evaluated on the remaining OOD test datasets (unseen by both backbone training and logistic regressor).

This procedure was then repeated for each possible OOD dataset, and the results are summarized in Table 6 .

We see that all classifier fitted using only natural images are capable to generalize well over all other OOD sets, while this is not entirely true when fitting on random noise datasets.

Also, fitting to all OODs validation sets (penultimate row), we can achieve even higher TNR scores over all test sets.

Using no OOD Dataset: To further evaluate the effectiveness of the method, we also tested the extreme case where no OOD samples are available for training.

To do this, we used an unsupervised algorithm (one-class SVM BID23 with RBF kernel), and we only fitted to ID samples (i.e., no OOD samples are seen in the training step).

The unsupervised results are summarized in the last row of Table 6 .

As one can see, even the unsupervised method shows reasonable performance; showing again that in the proposed feature space the ID/OOD samples have different behavior.

This corroborates the assumption that these features are a good indicator of OOD-ness.

Table 6 : Generalization to unseen OOD sets using CIFAR-100 as ID dataset and the WRN-28-10 backbone model.

Performance of the OOD detector when the logistic regression is fit using 1000 samples of a given OOD dataset and then evaluated with respect to other OOD test datasets using only "std" as features.

Results are TNR @ 95% TPR formatted as "mean (std)".

TinyImageNet ( Comparison with ODIN BID22 : We compare the generalization capabilities of our method with the state-of-the-art technique ODIN BID22 , in this new harder task.

We fit both OOD classifier to maximize their detection performance on TinyImageNet (c) and Gaussian validation sets (i.e., 2000 OOD samples), and evaluate on all OOD test datasets.

For our model, we use standard deviation features as inputs.

For ODIN, we tune its hyperparameters using the grid search described in B.1.

The results, presented in Table 7 , show that our method outperforms ODIN by a large margin, indicating better generalization to samples from unseen OOD datasets.

We evaluated how much it helps to use the batch statistics computed by BN.

As shown in Table 8 , normalizing the latent space using the BN statistics before computing the features has clear advantages.

We study if our method can correctly detect OOD samples even when a small number of samples is available.

Figure 4 shows the TNR @ 95% TPR for WRN 28-10 trained on CIFAR-10 (CIFAR-100), where only 27, 45, 75, 150, 300, 700, 1.5k, 3k (all) test images (ID + TinyImageNet crop + Gaussian, equally divided) are used to fit 25 coefficients of our logistic regressor.

Using our method, only 27 images (from each ID and OOD), are necessary to achieve an average of 87.6% of TNR @ 95% TPR.

Figure 4: Averaged TNR @ 95% TPR over the OOD datasets using only a few samples to fit the logistic regressor (for WRN-28-10).

The logistic regressor was fitted using TinyImageNet (c) and Gaussian validation sets using onlys (l) as features.

This experiment shows, in accordance to our results, that CIFAR-100 is more difficult to differentiate from other OOD datasets than CIFAR-10.

Table 7 : Comparison between ODIN and our proposed OOD detector for several setups using image classification networks.

All detector parameters (and ODIN's hyperparameters) were tuned for TinyImageNet (c) and Gaussian validation sets.

The results are formatted as "mean (std)".

Deep neural networks trained to maximize classification performance in a given dataset are extremely adapted to said dataset.

The statistics of activations throughout the network for samples from the training distribution (in-distribution) are remarkably stable.

However, when a sample from a different distribution (out-of-distribution) is given to the network, its activation statistics depart greatly from those of in-distribution samples.

Based on this observation, we propose a very simple yet efficient method to detect out-of-distribution samples.

Our method is based on computing averages of low-order statistics at the batch normalization layers of the network, and then use them as features in a linear classifier.

This procedure is much simpler and efficient than current stateof-the-art methods, and outperforms them by a large margin in the traditional ID/OOD fitting task (as proposed in previous works).

We evaluated all methods in the challenging task of fitting on a single OOD dataset and testing on samples from other (unseen) datasets.

In this harder scenario, our method generalizes well to unseen OOD datasets, outperforming ODIN by an even larger margin.

Moreover, we show some preliminary results that even in the extreme case where no OOD samples are used for the training (unsupervised) we get reasonable performance.

tuning is carried out on each pair of in-and out-of-distribution samples, which is the same procedure presented in BID22 and BID21 .

For reproducibility, a grid search is employed considering T ∈ {1, 1000} and with 21 linearly spaced values between 0 and 0.004 plus [0.00005, 0.0005, 0.0011].

We also train a Wide ResNet with L = 40 and widen factor of 4 (WRN-40-4) using the same training setup in Section 4, and the results for both training on CIFAR-10 and CIFAR-100 is depicted in TAB6 .

To compare all methods described in Section 2 to our proposed method, we also use the following additional metrics:1.

Area under precision-recall curve (AUPR).

The precision is evaluated as TP/(TP+FP) and recall in this case is the TPR.

The AUPR-in (AUPR-out) is defined when in-(out-of)-distribution samples are considered as the positive ones.

We also tested using the Street View House Number (SVHN) dataset BID24 as ID and DenseNet BC 100-12 as backbone model.

The pre-trained model is from BID21 , which has a test error rate of 3.58%.

The results are displayed in Table 10 .

Table 10 : Generalization to unseen OOD sets using SVHN as ID dataset and the DenseNet BC 100-12 backbone model.

Performance of the OOD detector when the logistic regression is fit using 1000 samples of a given OOD dataset and then evaluated with respect to other OOD test datasets using only "std" as features.

Results are TNR @ 95% TPR.

(a) CIFAR-10 DISPLAYFORM0 Figure 7: TNR @ 95% TPR obtained when aggregating information from multiple layers.

The leftmost bin corresponds to the deepest (last) layer, and the rightmost bin to the first BN layer in the network using a WRN-28-10 trained on CIFAR-10/CIFAR-100, the logistic regressor was fitted on TinyImageNet (c) + Gaussian validation sets and the results are an average over all OOD test datasets.

We also tested our proposed model on different TPR levels, and the results are depicted in TAB1 .

@highlight

Detecting out-of-distribution samples by using low-order feature statistics without requiring any change in underlying DNN.

@highlight

Presents an algorithm to detect out-of-distribution samples by using the running estimate of mean and variance within BatchNorm layers to construct feature representations later fed into a linear classifier.

@highlight

An approach for detecting out-of-distribution samples in which the authors propose to use logistic regression over simple statistics of each batch normalization layer of CNN.

@highlight

The paper suggests using Z-scores for comparing ID and OOD samples to evaluate what deep nets are trying to do.
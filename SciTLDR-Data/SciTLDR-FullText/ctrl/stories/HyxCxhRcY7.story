It is important to detect anomalous inputs when deploying machine learning systems.

The use of larger and more complex inputs in deep learning magnifies the difficulty of distinguishing between anomalous and in-distribution examples.

At the same time, diverse image and text data are available in enormous quantities.

We propose leveraging these data to improve deep anomaly detection by training anomaly detectors against an auxiliary dataset of outliers, an approach we call Outlier Exposure (OE).

This enables anomaly detectors to generalize and detect unseen anomalies.

In extensive experiments on natural language processing and small- and large-scale vision tasks, we find that Outlier Exposure significantly improves detection performance.

We also observe that cutting-edge generative models trained on CIFAR-10 may assign higher likelihoods to SVHN images than to CIFAR-10 images; we use OE to mitigate this issue.

We also analyze the flexibility and robustness of Outlier Exposure, and identify characteristics of the auxiliary dataset that improve performance.

Machine Learning systems in deployment often encounter data that is unlike the model's training data.

This can occur in discovering novel astronomical phenomena, finding unknown diseases, or detecting sensor failure.

In these situations, models that can detect anomalies (Liu et al., 2018; Emmott et al., 2013) are capable of correctly flagging unusual examples for human intervention, or carefully proceeding with a more conservative fallback policy.

Behind many machine learning systems are deep learning models (Krizhevsky et al., 2012) which can provide high performance in a variety of applications, so long as the data seen at test time is similar to the training data.

However, when there is a distribution mismatch, deep neural network classifiers tend to give high confidence predictions on anomalous test examples (Nguyen et al., 2015) .

This can invalidate the use of prediction probabilities as calibrated confidence estimates (Guo et al., 2017) , and makes detecting anomalous examples doubly important.

Several previous works seek to address these problems by giving deep neural network classifiers a means of assigning anomaly scores to inputs.

These scores can then be used for detecting outof-distribution (OOD) examples (Hendrycks & Gimpel, 2017; Lee et al., 2018; Liu et al., 2018) .

These approaches have been demonstrated to work surprisingly well for complex input spaces, such as images, text, and speech.

Moreover, they do not require modeling the full data distribution, but instead can use heuristics for detecting unmodeled phenomena.

Several of these methods detect unmodeled phenomena by using representations from only in-distribution data.

In this paper, we investigate a complementary method where we train models to detect unmodeled data by learning cues for whether an input is unmodeled.

While it is difficult to model the full data distribution, we can learn effective heuristics for detecting out-of-distribution inputs by exposing the model to OOD examples, thus learning a more conservative concept of the inliers and enabling the detection of novel forms of anomalies.

We propose leveraging diverse, realistic datasets for this purpose, with a method we call Outlier Exposure (OE).

OE provides a simple and effective way to consistently improve existing methods for OOD detection.

Through numerous experiments, we extensively evaluate the broad applicability of Outlier Exposure.

For multiclass neural networks, we provide thorough results on Computer Vision and Natural Language Processing tasks which show that Outlier Exposure can help anomaly detectors generalize to and perform well on unseen distributions of outliers, even on large-scale images.

We also demonstrate that Outlier Exposure provides gains over several existing approaches to out-of-distribution detection.

Our results also show the flexibility of Outlier Exposure, as we can train various models with different sources of outlier distributions.

Additionally, we establish that Outlier Exposure can make density estimates of OOD samples significantly more useful for OOD detection.

Finally, we demonstrate that Outlier Exposure improves the calibration of neural network classifiers in the realistic setting where a fraction of the data is OOD.

Our code is made publicly available at https://github.com/hendrycks/outlier-exposure.

Out-of-Distribution Detection with Deep Networks.

Hendrycks & Gimpel (2017) demonstrate that a deep, pre-trained classifier has a lower maximum softmax probability on anomalous examples than in-distribution examples, so a classifier can conveniently double as a consistently useful outof-distribution detector.

Building on this work, DeVries & Taylor (2018) attach an auxiliary branch onto a pre-trained classifier and derive a new OOD score from this branch.

Liang et al. (2018) present a method which can improve performance of OOD detectors that use a softmax distribution.

In particular, they make the maximum softmax probability more discriminative between anomalies and in-distribution examples by pre-processing input data with adversarial perturbations (Goodfellow et al., 2015) .

Unlike in our work, their parameters are tailored to each source of anomalies.

Lee et al. (2018) train a classifier concurrently with a GAN (Radford et al., 2016; Goodfellow et al., 2014) , and the classifier is trained to have lower confidence on GAN samples.

For each testing distribution of anomalies, they tune the classifier and GAN using samples from that out-distribution, as discussed in Appendix B of their work.

Unlike Liang et al. (2018) ; Lee et al. (2018) , in this work we train our method without tuning parameters to fit specific types of anomaly test distributions, so our results are not directly comparable with their results.

Many other works (de Vries et al., 2016; Subramanya et al., 2017; Malinin & Gales, 2018; Bevandic et al., 2018 ) also encourage the model to have lower confidence on anomalous examples.

Recently, Liu et al. (2018) provide theoretical guarantees for detecting out-of-distribution examples under the assumption that a suitably powerful anomaly detector is available.

Utilizing Auxiliary Datasets.

Outlier Exposure uses an auxiliary dataset entirely disjoint from test-time data in order to teach the network better representations for anomaly detection.

Goodfellow et al. (2015) train on adversarial examples to increased robustness.

Salakhutdinov et al. (2011) pre-train unsupervised deep models on a database of web images for stronger features.

Radford et al. (2017) train an unsupervised network on a corpus of Amazon reviews for a month in order to obtain quality sentiment representations.

Zeiler & Fergus (2014) find that pre-training a network on the large ImageNet database (Russakovsky et al., 2015) endows the network with general representations that are useful in many fine-tuning applications.

Chen & Gupta (2015) ; Mahajan et al. (2018) show that representations learned from images scraped from the nigh unlimited source of search engines and photo-sharing websites improve object detection performance.

We consider the task of deciding whether or not a sample is from a learned distribution called D in .

Samples from D in are called "in-distribution," and otherwise are said to be "out-of-distribution" (OOD) or samples from D out .

In real applications, it may be difficult to know the distribution of outliers one will encounter in advance.

Thus, we consider the realistic setting where D out is unknown.

Given a parametrized OOD detector and an Outlier Exposure (OE) dataset D OE out , disjoint from D test out , we train the model to discover signals and learn heuristics to detect whether a query is sampled from D in or D OE out .

We find that these heuristics generalize to unseen distributions D out .

Deep parametrized anomaly detectors typically leverage learned representations from an auxiliary task, such as classification or density estimation.

Given a model f and the original learning objective L, we can thus formalize Outlier Exposure as minimizing the objective DISPLAYFORM0 over the parameters of f .

In cases where labeled data is not available, then y can be ignored.

Outlier Exposure can be applied with many types of data and original tasks.

Hence, the specific formulation of L OE is a design choice, and depends on the task at hand and the OOD detector used.

For example, when using the maximum softmax probability baseline detector (Hendrycks & Gimpel, 2017) , we set L OE to the cross-entropy from f (x ) to the uniform distribution (Lee et al., 2018) .

When the original objective L is density estimation and labels are not available, we set L OE to a margin ranking loss on the log probabilities f (x ) and f (x).

We evaluate OOD detectors with and without OE on a wide range of datasets.

Each evaluation consists of an in-distribution dataset D in used to train an initial model, a dataset of anomalous examples D OE out , and a baseline detector to which we apply OE.

We describe the datasets in Section 4.2.

The OOD detectors and L OE losses are described on a case-by-case basis.

In the first experiment, we show that OE can help detectors generalize to new text and image anomalies.

This is all accomplished without assuming access to the test distribution during training or tuning, unlike much previous work.

In the confidence branch experiment, we show that OE is flexible and complements a binary anomaly detector.

Then we demonstrate that using synthetic outliers does not work as well as using real and diverse data; previously it was assumed that we need synthetic data or carefully selected close-to-distribution data, but real and diverse data is enough.

We conclude with experiments in density estimation.

In these experiments we find that a cutting-edge density estimator unexpectedly assigns higher density to out-of-distribution samples than in-distribution samples, and we ameliorate this surprising behavior with Outlier Exposure.

We evaluate out-of-distribution detection methods on their ability to detect OOD points.

For this purpose, we treat the OOD examples as the positive class, and we evaluate three metrics: area under the receiver operating characteristic curve (AUROC), area under the precision-recall curve (AUPR), and the false positive rate at N % true positive rate (FPRN ).

The AUROC and AUPR are holistic metrics that summarize the performance of a detection method across multiple thresholds.

The AUROC can be thought of as the probability that an anomalous example is given a higher OOD score than a in-distribution example (Davis & Goadrich, 2006 Whereas the previous two metrics represent the detection performance across various thresholds, the FPRN metric represents performance at one strict threshold.

By observing performance at a strict threshold, we can make clear comparisons among strong detectors.

The FPRN metric (Liu et al., 2018; Kumar et al., 2016; Balntas et al., 2016) is the probability that an in-distribution example (negative) raises a false alarm when N % of anomalous examples (positive) are detected, so a lower FPRN is better.

Capturing nearly all anomalies with few false alarms can be of high practical value.

In what follows, we use Outlier Exposure to enhance the performance of existing OOD detection techniques with multiclass classification as the original task.

Throughout the following experiments, we let x ∈ X be a classifier's input and y ∈ Y = {1, 2, . . .

, k} be a class.

We also represent the classifier with the function f : X → R k , such that for any x, 1 T f (x) = 1 and f (x) 0.Maximum Softmax Probability (MSP).

Consider the maximum softmax probability baseline (Hendrycks & Gimpel, 2017) which gives an input x the OOD score − max c f c (x).

Out-ofdistribution samples are drawn from various unseen distributions (Appendix A).

For each task, we test with approximately twice the number of D test out distributions compared to most other papers, and we also test on NLP tasks.

The quality of the OOD example scores are judged with the metrics described in Section 4.1.

For this multiclass setting, we perform Outlier Exposure by fine-tuning a pre-trained classifier f so that its posterior is more uniform on D OE out samples.

Specifically, the finetuning objective is DISPLAYFORM0 , where H is the cross entropy and U is the uniform distribution over k classes.

When there is class imbalance, we could encourage f (x) to match (P (y = 1), . . .

, P (y = k)); yet for the datasets we consider, matching U works well enough.

Also, note that training from scratch with OE can result in even better performance than fine-tuning (Appendix C).

This approach works on different architectures as well (Appendix D).

TAB14 [log b(x)] to the network's original optimization objective.

In TAB5 , the baseline values are derived from the maximum softmax probabilities produced by the classifier trained with DeVries & Taylor (2018)'s publicly available training code.

The confidence branch improves over this MSP detector, and after OE, the confidence branch detects anomalies more effectively.

TAB7 shows the large gains from using OE with a real and diverse dataset over using synthetic samples from a GAN.

DISPLAYFORM1 DISPLAYFORM2

In-Distribution Density estimators learn a probability density function over the data distribution D in .

Anomalous examples should have low probability density, as they are scarce in D in by definition (Nalisnick et al., 2019) .

Consequently, density estimates are another means by which to score anomalies (Zong et al., 2018) .

We show the ability of OE to improve density estimates on low-probability, outlying data.

PixelCNN++.

Autoregressive neural density estimators provide a way to parametrize the probability density of image data.

Although sampling from these architectures is slow, they allow for evaluating the probability density with a single forward pass through a CNN, making them promising candidates for OOD detection.

We use PixelCNN++ (Salimans et al., 2017) as a baseline OOD detector, and we train it on CIFAR-10.

The OOD score of example x is the bits per pixel (BPP), defined as nll(x)/num_pixels, where nll is the negative log-likelihood.

With this loss we fine-tune for 2 epochs using OE, which we find is sufficient for the training loss to converge.

Here OE is implemented with a margin loss over the log-likelihood difference between in-distribution and anomalous examples, so that the loss for a sample x in from D in and point x out from D OE out is max{0, num_pixels + nll(x in ) − nll(x out )}.Results are shown in greatly simplify the task of OOD detection.

Accordingly, the OOD detection task is to provide a score for 70-or 150-token sequences in the unseen D test out datasets.

We train word-level models for 300 epochs, and character-level models for 50 epochs.

We then fine-tune using OE on WikiText-2 for 5 epochs.

For the character-level language model, we create a character-level version of WikiText-2 by converting words to lowercase and leaving out characters which do not appear in PTB.

OOD detection results for the word-level and character-level language models are shown in

Extensions to Multilabel Classifiers and the Reject Option.

Outlier Exposure can work in more classification regimes than just those considered above.

For example, a multilabel classifier trained on CIFAR-10 obtains an 88.8% mean AUROC when using the maximum prediction probability as the OOD score.

By training with OE to decrease the classifier's output probabilities on OOD samples, the mean AUROC increases to 97.1%.

This is slightly less than the AUROC for a multiclass model tuned with OE.

An alternative OOD detection formulation is to give classifiers a "reject class" (Bartlett & Wegkamp, 2008) .

Outlier Exposure is also flexible enough to improve performance in this setting, but we find that even with OE, classifiers with the reject option or multilabel outputs are not as competitive as OOD detectors with multiclass outputs.

In addition to size and realism, we found diversity of D OE out to be an important factor.

Concretely, a CIFAR-100 classifier with CIFAR-10 as D OE out hardly improves over the baseline.

A CIFAR-10 classifier exposed to ten CIFAR-100 outlier classes corresponds to an average AUPR of 78.5%.

Exposed to 30 such classes, the classifier's average AUPR becomes 85.1%.

Next, 50 classes corresponds to 85.3%, and from thereon additional CIFAR-100 classes barely improve performance.

This suggests that dataset diversity is important, not just size.

In fact, experiments in this paper often used around 1% of the images in the 80 Million Tiny Images dataset since we only briefly fine-tuned the models.

We also found that using only 50,000 examples from this dataset led to a negligible degradation in detection performance.

Additionally, D OE Improves Calibration.

When using classifiers for prediction, it is important that confidence estimates given for the predictions do not misrepresent empirical performance.

A calibrated classifier gives confidence probabilities that match the empirical frequency of correctness.

That is, if a calibrated model predicts an event with 30% probability, then 30% of the time the event transpires.

Existing confidence calibration approaches consider the standard setting where data at test-time is always drawn from D in .

We extend this setting to include examples from D test out at test-time since systems should provide calibrated probabilities on both in-and out-of-distribution samples.

The classifier should have low-confidence predictions on these OOD examples, since they do not have a class.

Building on the temperature tuning method of Guo et al. (2017) , we demonstrate that OE can improve calibration performance in this realistic setting.

Summary results are shown in FIG3 .

Detailed results and a description of the metrics are in Appendix G.

In this paper, we proposed Outlier Exposure, a simple technique that enhances many current OOD detectors across various settings.

It uses out-of-distribution samples to teach a network heuristics to detect new, unmodeled, out-of-distribution examples.

We showed that this method is broadly applicable in vision and natural language settings, even for large-scale image tasks.

OE can improve model calibration and several previous anomaly detection techniques.

Further, OE can teach density estimation models to assign more plausible densities to out-of-distribution samples.

Finally, Outlier Exposure is computationally inexpensive, and it can be applied with low overhead to existing systems.

In summary, Outlier Exposure is an effective and complementary approach for enhancing out-of-distribution detection systems.

Expanded mutliclass out-of-distribution detection results are in TAB14 Table 8 : NLP OOD example detection for the maximum softmax probability (MSP) baseline detector and the MSP detector after fine-tuning with Outlier Exposure (OE).

All results are percentages and the result of 10 runs.

Values are rounded so that 99.95% rounds to 100%.Anomalous Data.

For each in-distribution dataset D in , we comprehensively evaluate OOD detectors on artificial and real anomalous distributions D test out following Hendrycks & Gimpel (2017) .

For each learned distribution D in , the number of test distributions that we compare against is approximately double that of most previous works.

Gaussian anomalies have each dimension i.i.d.

sampled from an isotropic Gaussian distribution.

Rademacher anomalies are images where each dimension is −1 or 1 with equal probability, so each dimension is sampled from a symmetric Rademacher distribution.

Bernoulli images have each pixel sampled from a Bernoulli distribution if the input range is [0, 1].

Blobs data consist in algorithmically generated amorphous shapes with definite edges.

Icons-50 is a dataset of icons and emojis (Hendrycks & Dietterich, 2019) ; icons from the "Number" class are removed.

Textures is a dataset of describable textural images (Cimpoi et al., 2014) .

Places365 consists in images for scene recognition rather than object recognition (Zhou et al., 2017) .

LSUN is another scene understanding dataset with fewer classes than Places365 (Yu et al., 2015) .

ImageNet anomalous examples are taken from the 800 ImageNet-1K classes disjoint from Tiny ImageNet's 200 classes, and when possible each image is cropped with bounding box information as in Tiny ImageNet.

For the Places365 experiment, ImageNet is ImageNet-1K with all 1000 classes.

With CIFAR-10 as D in , we use also CIFAR-100 as D test out and vice versa; recall that the CIFAR-10 and CIFAR-100 classes do not overlap.

Chars74K is a dataset of photographed characters in various styles; digits and letters such as "O" and "l" were removed since they can look like numbers.

Places69 has images from 69 scene categories not found in the Places365 dataset.

SNLI is a dataset of predicates and hypotheses for natural language inference.

We use the hypotheses for D OE out .

IMDB is a sentiment classification dataset of movie reviews, with similar statistics to those of SST.

Multi30K is a dataset of English-German image descriptions, of which we use the English descriptions.

WMT16 is the English portion of the test set from WMT16.

Yelp is a dataset of restaurant reviews.

English Web Treebank (EWT) consists of five individual datasets: Answers (A), Email (E), Newsgroups (N), Reviews (R), and Weblog (W).

Each contains examples from the indicated domain.

Validation Data.

For each experiment, we create a set of validation distributions D

Elsewhere we show results for pre-trained networks that are fine-tuned with OE.

However, a network trained from scratch which simultaneously trains with OE tends to give superior results.

For example, a CIFAR-10 Wide ResNet trained normally obtains a classification error rate of 5.16% and an FPR95 of 34.94%.

Fine-tuned, this network has an error rate of 5.27% and an FPR95 of 9.50%.

Yet if we instead train the network from scratch and expose it to outliers as it trains, then the error rate is 4.26% and the FPR95 is 6.15%.

This architecture corresponds to a 9.50% RMS calibration error with OE fine-tuning, but by training with OE from scratch the RMS calibration error is 6.15%.

Compared to fine-tuning, training a network in tandem with OE tends to produce a network with a better error rate, calibration, and OOD detection performance.

The reason why we use OE for fine-tuning is because training from scratch requires more time and sometimes more GPU memory than fine-tuning.

Outlier Exposure also improves vision OOD detection performance for more than just Wide ResNets.

Table 9 shows that Outlier Exposure also improves vision OOD detection performance for "All Convolutional Networks" (Salimans & Kingma, 2016 While − max c f c (x) tends to be a discriminative OOD score for example x, models with OE can do better by using −H(U; f (x)) instead.

This alternative accounts for classes with small probability mass rather than just the class with most mass.

Additionally, the model with OE is trained to give anomalous examples a uniform posterior not just a lower MSP.

This simple change roundly aids performance as shown in TAB16 : Comparison between the maximum softmax probability (MSP) and H(U; p) OOD scoring methods on a network fine-tuned with OE.

Results are percentages and an average of 10 runs.

For example, CIFAR-10 results are averaged over "Gaussian," "Rademacher," . . .

, or "CIFAR-100" measurements.

Detailed OOD detection results with language modeling datasets are shown in TAB18 .

DISPLAYFORM0

Models integrated into a decision making process should indicate when they are trustworthy, and such models should not have inordinate confidence in their predictions.

In an effort to combat a false sense of certainty from overconfident models, we aim to calibrate model confidence.

A model is calibrated if its predicted probabilities match empirical frequencies.

Thus if a calibrated model predicts an event with 30% probability, then 30% of the time the event transpires.

Prior research (Guo et al., 2017; Nguyen & O'Connor, 2015; Kuleshov & Liang, 2015) considers calibrating systems where test-time queries are samples from D in , but systems also encounter samples from D test out and should also ascribe low confidence to these samples.

Hence, we use OE to control the confidence on these samples.

In order to evaluate a multiclass classifier's calibration, we present three metrics.

First we establish context.

For input example X ∈ X , let Y ∈ Y = {1, 2, . . .

, k} be the ground truth class.

Let Y be the model's class prediction, and let C be the corresponding model confidence or prediction probability.

Denote the set of prediction-label pairs made by the model with S = {( y 1 , c 1 ), ( y 2 , c 2 ), . . . , ( y n , c n )}.

DISPLAYFORM0 Along similar lines, the MAD Calibration Error-which is an improper scoring rule due to its use of absolute differences rather than squared differences-is estimated with DISPLAYFORM1 Soft F1 Score.

If a classifier makes only a few mistakes, then most examples should have high confidence.

But if the classifier gives all predictions high confidence, including its mistakes, then the previous metrics will indicate that the model is calibrated on the vast majority of instances, despite having systematic miscalibration.

The Soft F1 score (Pastor-Pellicer et al., 2013; Hendrycks & Gimpel, 2017) is suited for measuring the calibration of a system where there is an acute imbalance between mistaken and correct decisions.

Since we treat mistakes a positive examples, we can write the model's confidence that the examples are anomalous with c a = (1 − c 1 , 1 − c 2 , . . .

, 1 − c n ).

To indicate that an example is positive (mistaken), we use the vector m ∈ {0, 1} n such that m i = 1(y i = y i ) for 1 ≤ i ≤ n.

Then the Soft F1 score is TAB3 : Calibration results for the temperature tuned baseline and temperature tuning + OE.

There are many ways to estimate a classifier's confidence.

One way is to bind a logistic regression branch onto the network, so that confidence values are in [0, 1] .

Other confidence estimates use the model's logits l ∈ R k , such as the estimate σ(max i l i ) ∈ [0, 1], where σ is the logistic sigmoid.

Another common confidence estimate is max i exp (l i )/ k j=1 exp (l j ) .

A modification of this estimate is our baseline.

Softmax Temperature Tuning.

Guo et al. (2017) show that good calibration can be obtained by including a tuned temperature parameter into the softmax: p(y = i | x) = exp(l i /T )/ k j=1 exp(l j /T ).

We tune T to maximize log likelihood on a validation set after the network has been trained on the training set.

Results.

In this calibration experiment, the baseline is confidence estimation with softmax temperature tuning.

Therefore, we train SVHN, CIFAR-10, CIFAR-100, and Tiny ImageNet classifiers with 5000, 5000, 5000, and 10000 training examples held out, respectively.

A copy of this classifier is fine-tuned with Outlier Exposure.

Then we determine the optimal temperatures of the original and OE-fine-tuned classifiers on the held-out examples.

To measure calibration, we take equally many examples from a given in-distribution dataset D Out-of-distribution points are understood to be incorrectly classified since their label is not in the model's output space, so calibrated models should assign these out-of-distribution points low confidence.

Results are in TAB3 .

Outlier Exposure noticeably improves model calibration.

While temperature tuning improves calibration, the confidence estimate p(y = i | x) cannot be less than 1/k, k the number of classes.

For an out-of-distribution example like Gaussian Noise, a good model should have no confidence in its prediction over k classes.

One possibility is to add a reject option, or a (k + 1)st class, which we cover in Section 5.

A simpler option we found is to perform an affine transformation of p(y = i | x) ∈ [1/k, 1] with the formula ( p(y = i | x) − 1/k)/(1 − 1/k) ∈ [0, 1].

This simple transformation makes it possible for a network to express no confidence on an out-of-distribution input and improves calibration performance.

As TAB5 shows, this simple 0-1 posterior rescaling technique consistently improves calibration, and the model fine-tuned with OE using temperature tuning and posterior rescaling achieved large calibration improvements.

In FIG8 , we show additional PR and ROC Curves using the Tiny ImageNet dataset and various anomalous distributions.

<|TLDR|>

@highlight

OE teaches anomaly detectors to learn heuristics for detecting unseen anomalies; experiments are in classification, density estimation, and calibration in NLP and vision settings; we do not tune on test distribution samples, unlike previous work
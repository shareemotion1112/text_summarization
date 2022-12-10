Intuitively, unfamiliarity should lead to lack of confidence.

In reality, current algorithms often make highly confident yet wrong predictions when faced with unexpected test samples from an unknown distribution different from training.

Unlike domain adaptation methods, we cannot gather an "unexpected dataset" prior to test, and unlike novelty detection methods, a best-effort original task prediction is still expected.

We compare a number of methods from related fields such as calibration and epistemic uncertainty modeling, as well as two proposed methods that reduce overconfident errors of samples from an unknown novel distribution without drastically increasing evaluation time: (1) G-distillation, training an ensemble of classifiers and then distill into a single model using both labeled and unlabeled examples, or (2) NCR, reducing prediction confidence based on its novelty detection score.

Experimentally, we investigate the overconfidence problem and evaluate our solution by creating "familiar" and "novel" test splits, where "familiar" are identically distributed with training and "novel" are not.

We discover that calibrating using temperature scaling on familiar data is the best single-model method for improving novel confidence, followed by our proposed methods.

In addition, some methods' NLL performance are roughly equivalent to a regularly trained model with certain degree of smoothing.

Calibrating can also reduce confident errors, for example, in gender recognition by 95% on demographic groups different from the training data.

In machine learning and computer vision, the i.i.d.

assumption, that training and test sets are sampled from the same distribution (henceforth "familiar" distribution), is so prevalent as to be left unwritten.

In experiments, it is easy to satisfy the i.i.d.

condition by randomly sampling training and test data from a single pool, such as photos of employees' faces.

But in real-life applications, test samples are often sampled differently (e.g., faces of internet users) and may not be well-represented, if at all, by the training samples.

Prior work BID24 has shown networks to be unreliable when tested on semantically unrelated input (e.g. feeding CIFAR into MNIST-trained networks), but users would not expect useful predictions on these data.

However, we find this issue extends to semantically related input as well, such as gender classifiers applied to faces of older or younger people than those seen during training, which is a more common occurrence in practice and more problematic from a user's perspective.

We demonstrate that, counter to the intuition that unfamiliarity should lead to lack of confidence, current algorithms (deep networks) are more likely to make highly confident wrong predictions when faced with such "novel" samples, both for real-world image datasets (Figure 1 ; see caption) and for toy datasets FIG1 ; see subcaption 2(a) and Appendix A).

The reason is simple: the classification function, such as a deep network, is undefined or loosely regulated for areas of the feature space that are unobserved in training, so the learner may extrapolate wildly without penalty.

Confident errors on novel samples can be catastrophic.

Whether one would ride in a self-driving car with a 99.9% accurate vision system, probably depends on how well-behaved the car is on the 0.1% mistakes.

When a trained model labeled a person as a gorilla BID52 , the public trust in that system was reduced.

When a driving vision system confidently mistook a tractor trailer BID50 , a person died.

Scholars that study the impact of AI on society consider differently distributed samples to be a major risk BID47 : "This is one form of epistemic Novel herptile, model: 99.4% bird (ours: 76.5%) Novel fish, model: 99.0% bird (ours: 92.0%)Figure 1: Deep networks can often make highly confident mistakes when samples are drawn from outside the distribution observed during training.

Shown are example images that have ages, breeds, or species that are not observed during training and are misclassified by a deep network model with high confidence.

Using our methods (shown here is G-distillation, to distill an ensemble of networks on both training and unsupervised examples) makes fewer confident errors for novel examples, increasing the reliability of prediction confidence overall.uncertainty that is quite relevant to safety because training on a dataset from a different distribution can cause much harm."

Our trust in a system requires its ability to avoid confident errors.

Unfortunately, these novel samples may differ from training in both expected and unexpected ways.

This means gathering a set of "unexpected" training samples, though required by covariate shift and domain adaptation methods, becomes unviable BID47 .

One may use novelty detection methods to identify novel samples at test time (to report an error or seek human guidance), but this may be insufficient.

These "outliers" (e.g. underrepresented faces that users upload) are perfectly normal samples in the eyes of a user and they would reasonably expect a prediction.

Also, human guidance may not be fast enough or affordable.

In this paper, our aim is to reduce confident errors for predictions on samples from a different (often non-overlapping) but unknown distribution (henceforth "novel" distribution).

In contrast with most recent work, we focus on the confidence of the prediction rather than only the most likely prediction (accuracy) or the confidence ordering (average precision or ROC).

In addition to reviewing the effectiveness of established methods, we propose and evaluate two ideas that are straightforward extensions to ensemble and rejection methods.

One is that multiple learners, with different initializations or subsamples of training data, may make different predictions on novel data (see BID24 ).

Hence, ensembles of classifiers tend to be better behaved.

But ensembles are slow to evaluate.

If we distill BID18 an ensemble into a single model using the training data, the distilled classifier will have the original problem of being undefined for novel areas of the feature space.

Fortunately, it is often possible to acquire many unlabeled examples, such as faces from a celebrity dataset.

By distilling the ensemble on both the training set and on unsupervised examples, we can produce a single model that outperforms, in terms of prediction confidence, single and standard distilled models on both identically and differently distributed samples.

Another idea is that if the training set does not provide enough information for the unseen data, therefore it may be desired to simply avoid confident predictions.

We can lower their confidence according to the output of an off-the-shelf novelty detector.

This means reducing confident errors by reducing confidence, regardless of correctness.

It may improve novel sample prediction quality, but in turn degrade performance on familiar samples.

Although this idea sounds like a natural extension to novelty detection, we are unaware of any implementation or analysis in the literature.

Experimentally, we investigate the confidence problem and perform an extensive study by creating "familiar" and "novel" test splits, where "familiar" are identically distributed with training and (a) Demonstration of deep networks' generalization behavior with a 2-dimensional toy dataset "square".

Column 1: we design a binary ground truth for classification on a 2-dimensional feature space.

Column 2: for training, we only provide samples on the left and lower portions (the "familiar" distribution), and reserve the upper-right only for testing (the "novel" distribution).

Column 3-7: we show negative log likelihood (NLL) predicted for each point in the feature space.

A small NLL indicates correct prediction, while a very large NLL indicates a confident error.

Column 3, 4: multiple runs of the network have similar performances on familiar regions but vary substantially in novel regions where the training data imposes little or no regulation, due to optimization randomness.

Column 5: an ensemble of 10 such networks can smooth the predictions and reduce confident errors at the sacrifice of test efficiency.

Column 6: distilling the ensemble using the training samples results in the same irregularities as single models.

Column 7: one of our proposals is to distill the ensemble into a single model using both the labeled training data and unsupervised data from a "general" distribution.

"novel" are not.

For example, in cat vs. dog classification, the novel examples are from breeds not seen during training, or in gender classification, the novel examples are people that are older or younger than those seen during training.

Our evaluation focuses on negative log likelihood (NLL) and the fraction of highly confident predictions that are misclassified ("E99").

They measure both prediction accuracy and how often confident errors occur.

To summarize, our contributions are:• Draw attention to a counter-intuitive yet important problem of highly confident wrong predictions when samples are drawn from a unknown distribution that is different than training.• Evaluate and compare the effectiveness of methods in related fields, including two proposed methods to reduce such overconfident errors.• Propose an experimental methodology to study the problem by explicitly creating familiar and novel test splits and measuring performance with NLL and E99.

To the best of our knowledge, there is no prior work with exactly the same goal and focus as this paper, but we are very similar to several lines of work.

To avoid confusion, we compile TAB0 to show our differences from these prior work for your reading convenience.

Epistemic uncertainty.

The most related works model and reduce epistemic uncertainty (model uncertainty from incomplete knowledge) by estimating and evaluating a set or distribution of models that agree with the training data.

The intuition is that on familiar test data, performance is boosted Bayesian approaches BID0 BID1 BID17 estimate a distribution of network parameters and produce a Bayesian estimate for likelihood.

These methods are usually very computationally intensive BID24 , limiting their practical application.

BID9 propose MC-dropout as a discrete approximation of Bayesian networks, adopting the dropout technique for a Monte-Carlo sample of likelihood estimates.

They further propose to jointly estimate aleatoric and epistemic uncertainties BID22 to increase the performance and quality of uncertainty.

BID24 propose to instead use an ensemble to emulate a Bayesian network and achieve better performance.

These works reduce uncertainty and improve generalization under i.i.d.

assumptions.

Our work differs in its focus of improving confidence estimates for novel samples (differently distributed from training), and our method is much more efficient at test time than MC-dropout or ensembles.

BID13 reduces the confidence towards a prior on unseen data, which is similar to our proposed NCR method.

We note that they perform regression on a low-dimensional feature space generalizing into future data, which is a very different problem from our image classification generalizing into unexpected samples.

They use perturbed original dataset features and the prior as the labels for reducing confidence.

This can be hard to generalize to an 224x224x3 image space.

Domain adaptation BID36 ) aims at training on a source domain and improving performance on a slightly different target domain, either through unsupervised data or a small amount of supervised data in the target domain.

In our settings, it is unviable to sample a representative "unexpected dataset" prior to test time, which we consider unknowable in advance.

Consequently, the unsupervised samples we use are not from the familiar or novel distribution.

Domain generalization BID31 BID43 ) is more closely related to our work, aiming to build models that generalize well on a previously unspecified domain, whose distribution can be different from all training domains.

Unlike in domain adaptation, the target domain in this case is unknown prior to testing.

These models generally build a domaininvariant feature space BID31 or a domain-invariant model BID43 , or factor models into domain-invariant and domain-specific parts .Also related are attribute-based approaches, such as BID6 , who build an attribute classifier that generalizes into novel object categories, similar to our experiment set-up.

They select features that are discriminative for an attribute within individual classes to build invariance across object classes.

These models all require multiple training domains to learn invariant representations, with the intent to improve robustness to variations in the target domain.

In contrast, our method is concerned only with novel data and does not require multiple training domains.

Some methods refuse to predict by estimating whether the system is likely to fail and outputting a signal requesting external intervention when necessary.

Meta-recognition BID42 estimates the performance based on the model output.

Rejection options BID8 BID10 or failure detection BID53 BID49 estimate the risk of misclassification to determine whether to refuse prediction, usually by looking at how close samples are to decision boundaries.

Unlike our work, these works are not concerned with the quality of confidence estimates and do not analyze their rejection ability for samples from novel distributions.

Outlier detection BID4 BID25 BID28 , novelty detection, and one-class classification BID46 BID23 ) determine whether a test sample comes from the same distribution as the training data.

The downside is that these methods would provide no confidence estimate for rejected samples, even though the models can still provide informative estimates (e.g. in Section 5 gender classification, baseline accuracy on novel data is 85% despite bad NLL performance).

One of our methods naturally extends these methods by using their output to improve confidence estimates.

Generalization.

Various techniques have been proposed in deep learning literature to minimize the generalization gap, and popular ones include data augmentation, dropout BID44 , batch normalization BID21 , and weight decay.

BID19 propose better hyperparameter selection strategies for better generalization.

Bagging BID2 and other model averaging techniques are also used prior to deep learning.

These methods focus on reducing generalization gap between training and validation.

They do not address issues with unexpected novel samples and can be used in conjunction with our method.

Theoretical analyses for off-training-set error BID41 BID38 and empirical analysis of generalization for test samples BID33 are also available, but these methods measure, rather than reduce, generalization errors.

Calibration methods (e.g. BID12 ) aim to improve confidence estimates, but since the confidence estimates are learned from familiar samples (i.i.d. with training), risk is not reduced for novel samples.

However, we experimentally find that BID12 performs surprisingly well on unseen novel data, which the method is not optimized for.

Distilling BID18 can be used on unsupervised samples.

BID37 obtain soft labels on transformed unlabeled data and use them to distill for unsupervised learning.

BID27 extend models to new tasks without retaining old task data, using the new-task examples as unsupervised examples for the old tasks with a distillation loss.

Distillation has also been used to reduce sensitivity to adversarial examples that are similar to training examples BID34 .

Our work differs from all of these in the focus on producing accurate confidences for samples from novel distributions that may differ significantly from those observed in training.

One-shot learning (e.g. BID48 ) and zero-shot learning (e.g. BID51 ) aim to build a classifier through one sample or only metadata of the class.

They focus on building a new class, while we focus on generalizing existing classes to novel samples within.

The goal of this paper is to improve confidence estimate of deep models on unseen data.

We focus on a classification setting, although our framework could be naturally extended to tasks similar to classification, such as VQA and semantic segmentation.

We assume the probability of label given data P (y|x) is the same where familiar and novel distributions overlap, but unlike covariate shift, we assume no knowledge of the novel distribution other than what is already in the familiar distribution.

Notations.

Denote by (x F , y F ) ∼ F the familiar data distribution, and F tr the training set, F ts the test set drawn from F. Further denote by (x N , y N ) ∼ N a novel data distribution, which satisfies P F (y|x) = P N (y|x) where the input distributions overlap.

Denote by N ts the test set drawn from N .

The inputs x F and x N may occupy different portions of the entire feature space, with little to no overlap.

Later in this section, we introduce and describe an unsupervised distribution x G ∼ G with training set G tr .In our problem setting, the goal is to improve performance and quality of the estimation for P N (y|x).

Only F tr (and unsupervised G tr ) are provided in training, while F ts and N ts are used in test time.

No training sample from N is ever used, and N should not have been seen by the model, even during pre-training.

Distillation of an ensemble.

We base our first method on the original distillation from an ensemble BID18 , which we briefly summarize.

First, train an ensemble f Ens (·) on F tr , which Table 2 : Illustration of the data usage of G-distillation on familiar data F, general data G, and some novel data N that we assume no knowledge of.

DISPLAYFORM0 consists of several networks such as ResNet18 BID16 .

Then, for each (x F , y F ) ∈ F tr , obtain a soft label y F , where y (c) DISPLAYFORM1 is the probability estimate for class c given by the ensemble.

Finally, train a new single-model network f θ by taking its probability prediction y F and applying the distillation loss L dis ( y F ,ŷ F ).

This is simply a cross-entropy loss between the distribution estimates, with a temperature T applied to the logits (see BID18 ) to put more focus on relative probability mass.

To further improve distillation performance, a classification loss L cls (cross-entropy is used) over F is added as an auxiliary.

The final loss becomes: DISPLAYFORM2 As can be seen, distillation is still a method focused on the familiar distribution F, and we have shown that a distilled network is not necessarily well-behaved on N .Method 1: G-distillation of an ensemble with extra unsupervised data.

To improve distillation on novel data, a natural idea would be having the distilled model mimic the ensemble on some kind of novel data.

Denote by x G ∼ G an unlabeled general data distribution which ideally encompasses familiar F and any specific novel N .

Here "encompass" means that data from F and N can appear in G: ∀x, DISPLAYFORM3 We draw a training set G tr from such general distribution G.A distribution that has all related images is nearly impossible to sample in practice.

Hence, we cannot rely on G tr ∼ G to encompass all possible novel N .

We need to pick a dataset that is sufficiently diverse, sufficiently complex, and sufficiently relevant to the task, so our method can use it to generalize.

Often, such a set can be obtained through online searches or mining examples from datasets with different annotations.

For example, for making a gender classifier robust to additional age groups, we could sample G tr from the CelebA BID45 dataset (note that CelebA does not have age information and we do not need it; for the sake of the experiment we also discard gender information from CelebA).

For making a cat-dog classifier robust to novel breeds, we could sample G tr from ImageNet (Russakovsky et al., 2015) .

Note these G tr 's do not encompass N , but only provide a guideline for generalization.

Similar to distillation, we train an ensemble f Ens and obtain the soft labels y F .

In addition, we also obtain soft labels y G by evaluating f Ens on x G ∈ G tr .

Then we add the samples G tr into the training set, and train using the combined data: DISPLAYFORM4 Note that G is unsupervised, so we cannot apply L cls on its samples.

For test time, we simply evaluate the probability estimation y F against the ground truth in F ts , and y N against those in N ts , respectively.

Table 2 demonstrates our training and evaluation procedure.

Method 2: Novelty Confidence Reduction (NCR) with a novelty detector.

This method is more straightforward.

We make use of a recent off-the-shelf, model-free novelty detector ODIN BID28 .

We run both the single model and the ODIN procedure on the input x to get the probability estimateŷ and the detection score s 0 (x) (where a smaller s 0 means a more novel input).

BID15 LFW+, age 0-17 & 60+ -CelebA BID45 ImageNet superclasses ILSVRC12, some species* ILSVRC12, other species* Per superclass: train: 1k, test: 400 COCO BID29 Cat-dog binary recognition Pets, some breeds* BID35 Pets, other breeds* -ILSVRC12 (Russakovsky et al., 2015) VOC-COCO recognition VOC, 20 classes BID5 COCO, but ignore non-VOC classes -Places365-standard BID54 However, s 0 may not be calibrated and may lie in an arbitrary interval specific to the detector, e.g. [0.5, 0.51].

We normalize it to [0, 1] with a piecewise-linear function s(x) = max 0, min DISPLAYFORM5 , 1 where s ·% are the ·% percentiles of the detection scores for all training samples.

s(x) closer to 1 means the confidence should be reduced more.

At test time, we linearly interpolate betweenŷ and the prior y 0 : DISPLAYFORM6 where y (c) 0 = P F (y = c) is the prior of class c on the familiar F tr , and λ s = 0.15 a hyperparameter.

Efficiency comparison.

An ensemble BID24 requires M forward passes where M is the number of ensemble members.

MC-dropout BID9 requires ∼ 50× forward passes for the Monte Carlo samples.

G-distillation needs only one forward pass, and NCR only needs to evaluate an extra novelty detector (in particular, ODIN needs a forwardbackward pass).

Therefore at test time they are much faster.

However, G-distillation pays a price at training time by training both an ensemble and a distilled model.

We refer readers to Appendix B for implementation details.

Datasets.

To demonstrate effectiveness in appropriately handling unexpected novel data, and reduce confident failures thereof, we perform extensive analysis on four classification datasets mimicking different scenarios.

We set up the F, N , and G distributions, and make sure that F and N are completely non-overlapping, unless otherwise noted.

TAB1 illustrates our datasets:• Gender recognition, mimicking a dataset with obvious selective bias.• ImageNet animal superclass (mammals, birds, herptiles, and fishes) recognition, mimick-ing an animal classifier being tested on unseen species within these superclasses.

(*) We determine the original classes belonging to each superclass, sort the classes by their indices, and use the first half of the class list as familiar and the second half novel.

This makes sure F and N do not overlap by definition.• Cat-dog recognition.

Similar in spirit as the above.

(*) The first 20 dog breeds and 9 cat breeds are deemed familiar, and the rest are novel.• VOC-COCO recognition (determining presence of object categories).

Mimics a model trained on a lab dataset being applied on more complex real world images with unexpected input.

tvmonitor is mapped to tv and not laptop.

Note that VOC-COCO is quite different from the others where F and N do not overlap, because VOC and COCO images can be very similar.

Pre-training.

When designing our experiments, we should make sure N is still novel even considering information gained from the pre-training dataset.

Although using ImageNet (Russakovsky et al., 2015) pre-trained models may improve performance, we note that the dataset almost always contains classes that appear in our "unseen" novel N choices; therefore pre-training on ImageNet would render the novel N not actually novel.

Instead, we opt to use Places365-standard BID54 as the pre-training dataset.

We argue that our method would generalize to ImageNet pre-trained models when N classes are disjoint from ImageNet.

Validation splits.

Since we need to report negative log likelihood on both F ts and N ts , datasets with undisclosed test set cannot be directly used.

For ImageNet animals, VOC, and COCO, we report performance on the validation set, while splitting the training set for hyperparameter tuning.

For LFW+ and Oxford-IIIT Pets, we split the training set for validation and report on the test set.

For LFW+, we use the first three folds as training, and the last two folds as test since the dataset is smaller after the familiar-novel split.

For the general G tr ∼ G dataset we proportionally sample from the corresponding training set, or a training split of it during hyperparameter tuning.

Compared methods.

We compare to normal single models ("single"), and standard distillation using F tr images ("distilling") as baselines.

We compare to using an ensemble ("ensemble"), which is less efficient in test time.

In addition, we compare to BID22 where aleatoric and epistemic uncertainties are modeled to increase performance ("uncertainty").

Since ResNet18 does not use dropout, for this method we insert a dropout of p = 0.2 after the penultimate layer.

For a fairer comparison, we also include an experiment using DenseNet161 as the base network, which is designed to use dropout.

We also compare to BID12 ) ("T -scaling") where only familiar samples are considered to calibrate the confidences, as well as an ensemble with members with the same calibration ("calib.

ens.").In our experiments, we discover that some methods outperform on novel samples and underperform on familiar, and one may suspect that the trade off is simply an effect of smoothing the probability regardless of the sample being novel or familiar, or the prediction being correct or wrong.

To analyze this, we also report the performance of further raising or decreasing the temperature of the prediction logits of each method by τ :ŷ DISPLAYFORM0 whereŷ (c) is the original estimate for class c. We use τ ∈ [0.5, 5], and plot the trace for how the familiar and novel NLL change.

Note that this is equivalent to using BID12 with a number of different temperature choices, but the trace can be optimistic (and seen as an oracle), since in practice the choice of τ has to be decided before evaluating on test.

Performance measure.

Note that we want to measure how well the model can generalize, and how unlikely the model is to produce confident errors.

As discussed in Section 1, we mainly use the negative log likelihood (NLL), a long-standing performance metric BID32 BID7 , as a measure of the quality of our prediction confidence.

If a prediction is correct, we prefer it to be confident, leading to a lower NLL.

And for incorrect predictions, we prefer it to be unconfident, which means the likelihood of the correct class should in turn be higher, which also leads to a lower NLL.

The metric gives more penalty towards confident errors, suiting our needs.

In summary, the model needs to learn from only familiar labeled data (plus unsupervised data) and produce the correct confidence (or lack of confidence) for novel data to improve the NLL.During evaluation, we clip the softmax probability estimates given by all models to [0.001, 0.999] in order to prevent a small number of extremely confident wrong predictions from dominating the NLL.

We clip these estimates also considering overconfidence beyond this level is risky with little benefit in general.

The processed confidences still sum up to 1 for binary classifiers, but may be off by a small amount for multiclass scenarios.

We also report a confident error rate ("E99") where we show the error rate for samples with probability estimate of any class larger than 0.99.

Ideally this value should be within [0, 0.01].

Further, we also report accuracy and mean Average Precision (mAP) and test the hypothesis that we perform comparably on these metrics.

Due to the relatively high variance of NLL on N ts , we run our experiments 10 times to ensure statistical significance with a p-test, but we run the ensemble method only once (variance estimated using ensemble member performance variance).

Our 10 runs of the distillation methods use the same ensemble run.

We first compare G-distill to the baselines on the four scenarios.

Tables in Figures 3 and 4 Figure 3: Our performance on familiar F and novel data N , for tasks with no overlap between the two.

Tables: in terms of novel NLL, calibrated methods perform the best, while our methods outperform the rest of the single-model methods.

Methods perform similarly on familiar NLL, while ours and calibration are more reliable with high-confidence estimates (E99), indicating a better generalization.

We perform on the same level on accuracy or mAP as other methods except for Gdistill in (c).

Graphs: Trade off familiar and novel NLL by smoothing the probability estimates with a higher or lower temperature.

Crosses indicate performance without smoothing (τ = 1).

Bottomleft is better.

Our methods do not outperform single models considering smoothing trade-off.

Figure 4 : Our performance on VOC-COCO recognition, where familiar F and novel data N have a strong overlap.

Table: for NLL, we outperform other single-model based methods on novel data but underperform on familiar.

Calibration methods do not make much difference.

Graph: considering trade-off between familiar and novel NLL, G-distillation performs similarly to distillation, while NCR underperforms.

See Figure 3 for details.

Note that this figure is zoomed in more.that among these experiments, those in Figure 3 have novel distribution N completely disjoint from familiar F, while Figure 4 does not.

Single models and the ensemble: single models perform much worse on novel NLL than familiar.

Their familiar E99 is properly small, but novel E99 is far beyond 0.01.

This confirms that confident errors in novel data is indeed a problem in our real image datasets.

Ensembles not only improve performance on familiar data, but also are much better behaved on novel data.

Figure 4 , properly calibrated T -scaling with single models perform as well as, or better than, all methods using a single model at test time.

An ensemble of calibrated members is nearly equivalent to a ensemble with further smoothed predictions, and it performs best in terms of novel NLL (at the expense of familiar NLL, and test efficiency).

It is very surprising that even when calibrated using familiar data only, single models are able to outperform all methods except ensembles of calibrated members in Figure 3 .

These indicate that proper calibration is a key factor when reducing confident errors even on novel data.

In Figure 4 , calibration does not make much difference (single models are well-calibrated already).

Our proposed methods: G-distill and sometimes NCR perform reasonably well in familiar-novel balance (except on Figure 3(c) ).

In Figure 4 (VOC-COCO), distilling and G-distill outperforms on novel NLL, but note that the difference is much smaller compared to other experiments.

With smoothing trade-off: Looking at predictions with different levels of smoothing, we can tell that uncertainty, distillation, G-distill, and NCR are all equivalent or inferior to a smoothed version of a single model.

Considering smoothing with τ is equivalent to tuning T -scaling on test data, this means that (1) these methods may outperform single models in accuracy or mAP, and have bettercalibrated NLL than single models, but are ultimately inferior to better calibrated single models in confidence quality; and (2) even if these methods are calibrated, their confidence may not outperform a calibrated single model trained regularly.

We note that in practice, one cannot choose the smoothing τ using a test novel dataset, so the smoothing results are only for analysis purposes.

Figure 3 , our error rate among novel confident predictions is far below compared non-calibrated methods and much closer to familiar E99.

For Figure 4 , we also have a slightly lower novel E99.

These indicate a better calibration with confident predictions, especially with a disjoint N .

However, the E99 for both NCR and the calibrated methods (T -scaling and ensemble with calibrated members) is usually far lower than 0.01, suggesting often under-confident predictions.

Other metrics: for accuracy or mAP, our methods (especially NCR) remain competitive compared to methods other than the ensemble, in Figures 3(a) and 3(b) .

However, they only perform similarly to distillation in Figure 3 (c) and slightly underperform the others in Figure 4 .

Calibration with T -scaling and smoothing with temperature do not change accuracy or mAP.Novelty detector effectiveness: one interesting phenomenon is that the novelty detector is not very effective in distinguishing familiar and novel samples (AUC for each dataset: LFW+ 0.562, ImageNet 0.623, Pets 0.643, VOC-COCO 0.665), but are quite effective in separating out wrongly classified samples (AUC: LFW+ 0.905, ImageNet 0.828, Pets 0.939, VOC-COCO 0.541).

We hypothesize that the novelty detector can fail to detect some data in our novel split that are too similar to the familiar.

However, this does not affect the prediction performance much since the classifier model are less prone to confident failures on these samples.

Miscellaneous.

In Appendix C, we evaluate the impact of some factors on our experiments, namely the choice of ensemble type, and the base network size.

It is also possible to train an ensemble of Gdistillation models to boost the performance, at the sacrifice of test time performance.

We find that it improves the foreign NLL beyond the original ensemble, while still underperforming calibration.

We also tried to combine G-distillation and NCR, by performing novelty detection on the G-distilled network and further reduce novelty confidence.

However, the results show that only when both methods show advantage against baselines, the combined method can outperform both components.

Otherwise the combined method may underperform baselines.

In this paper, we draw attention to the importance of minimizing harm from confident errors in unexpected novel samples different from training.

We propose an experiment methodology to explicitly study generalization issues with unseen novel data, and compare methods from several related fields.

We propose two simple methods that use an ensemble and distillation to better regularize network behavior outside the training distribution, or reduce confidence on detected novel samples, and consequently reduce confident errors.

For future work, it can be beneficial to investigate the ability to handle adversarial examples using this framework, and improve calibration with unexpected novel samples taken into account.

To better visualize the problem of confident errors when using single neural networks, we demonstrate the generalization behavior of networks on 2-dimensional feature spaces.

We construct toy datasets with input x ∈ R 2 and define labels y ∈ {0, 1}. For each dataset we split the feature space into familiar region F and novel region N , train models on F and evaluate on a densely-sampled meshgrid divided into F and N regions.

See FIG1 , column 1 for an illustration of the ground truth and familiar set of our "square" dataset.

Intuitively, the "square" dataset seems to be easy, and the network may perform well on the novel data, while for harder datasets the network might either perform badly or produce a low-confidence estimation.

FIG1 show the results.

In the "square" dataset, the networks do not generalize to the novel corner, but rather draw an irregular curve to hallucinate something smoother.

As a consequence, the networks become confidently wrong on a significant portion of the dataset.

In multiple runs of the optimization, this behavior is consistent, but the exact regions affected are different.

The ensemble gives a much smoother result and (appropriately) less confident estimates on the novel region, and the area of confident errors is largely reduced.

Figure 2(b) further shows the negative log likelihood (NLL) performance of each method.

While the performance for familiar samples is similar across methods, the NLL for novel samples improves dramatically when using an ensemble.

This indicates that the ensemble has a more reliable confidence estimate on novel data.

We further test standard distillation BID18 and our method (see Section 3 for notations and definitions) against these baselines.

We note that this experiment can only be seen as an upper bound of what our method could achieve, because when we sample our unsupervised set from the general distribution in this experiment, we use F and N combined for simplicity.

In our later experiments on real images, the novel distribution is considered unknown.

Figure 2(b) shows that standard distillation performs not much better than single methods, consistent with our hypothesis that distillation using the familiar samples still results in irregularities on novel data.

On the other hand, our method performs much closer to the ensemble.

This positive result encourages us to further test our method on datasets with real images.

FIG3 illustrates the irregularities of single models with more toy experiments, and shows the performance of all methods involved.

Although the toy datasets can be very simple ("xor" and "circle") or very complex ("checkerboard"), in all cases, the same observations in FIG1 applies.

The single models are more likely to have erratic behaviors outside the familiar region, the ensemble behaves more regularly, and our method can emulate the ensemble in terms of the robustness in novel regions.

For toy experiments.

We take 1200 samples for both train and validation, while our test set is simply a densely-sampled meshgrid divided into familiar and novel test sets.

We use a 3-hiddenlayer network, both layers with 1024 hidden units and Glorot initialization similar to popular deep networks, to avoid bad local minima when layer widths are too small BID3 .

Batchnorm BID21 and then dropout BID44 are applied after ReLU.

We use the same hyperparameter tuning, initialization, and training procedures as described in Section 3 implementation details.

For image experiments.

We use the following experimental settings, unless otherwise explained.

We set λ dis = 2 3 and λ cls = 1 3 .

The original paper used 1 and 0.5, but we scale them to better compare with non-distillation methods.

Temperature T = 2.

We sample G tr to be roughly 1 4 the size of F tr .

For the network, we mostly use ResNet18 BID16 ), but we also perform an experiment on DenseNet161 BID20 to check that our conclusions are not architecturedependent.

To enable training on small datasets, we use networks pre-trained on very large datasets as a starting point for both ensemble members and the distilled network.

We use a simple ensemble of 10 members.

We also compare to a bagging BID2 ensemble scheme which resamples the dataset.

This is more popular prior to deep networks, but we empirically find that bagging undermines the log likelihood on both familiar data and novel data.

We initialize the final layer of our pre-trained network using Glorot initialization BID11 .

We optimize using stochastic gradient descent with a momentum of 0.9.

For data augmentation, we use a random crop and mirroring similar to Inception .

At test time we evaluate on the center crop of the image.

We lower the learning rate to 10% when validation performance plateaus and run an additional 1/3 the number of past epochs.

We perform hyper-parameter tuning for e.g. the learning rate, the number of epochs, and T in BID12 using a manual search on a validation split of the training data, but use these hyperparameters on networks trained on both training and validation splits of the training data.

Note that N is unknown while training, so we should use the F ts performance (accuracy, mAP) as the only tuning criteria.

However, λ s for our second method NCR needs to be tuned according to some novelty dataset, which we assume unavailable.

This makes a fair hyper-parameter tuning hard to design.

We further split the Pets dataset familiar F tr into validation-familiar and validation-novel splits, again by assigning some breeds as familiar and others as novel.

We tune the hyperparameter on this validation set with a grid search, and use the result λ s = 0.15 on all experiments.

We also manually tune the choice of percentiles (95% and 5%) this way.

Ensemble type.

We use simple ensembles since they are reported to perform better than a bagging ensemble BID24 on the familiar dataset.

We investigate whether bagging will benefit novel data in turn.

We compare bagging to a simple ensemble and the distillation methods with them on the animal superclass task in Figure 6 .The simple ensemble indeed has better familiar performance, but without smoothing, bagging has better novel performance.

Considering smoothing trade-off, there is a loss in both novel and familiar NLL when using a bagging ensemble.

Network structure.

To evaluate the robustness of our experiments to the base network, we use a DenseNet161 trained on Places365-standard (provided by BID54 ) as the base model, and perform our experiment on the gender recognition task in FIG4 .

Our observations from using ResNet18 hold.

Our methods perform better than single models, but are outperformed by proper calibrated models.

However, G-distill now underperforms T -scaling without further smoothing.

Using an ensemble of G-distilled networks, at the sacrifice of test computational efficiency.

Specifically, we train a bagging ensemble using standard training procedures, obtain ensemble soft labels y F and y G like before, but train 10 G-distilled networks using the same set of soft labels (without resampling for convenience).

At test time, the outputs from these networks are averaged to get the probability estimate.

This scheme has the same drawback as the ensemble -reduced test efficiency.

We name this scheme "G-distill ×10".

For completeness, we also compare to an ensemble of the standard distillation, "distilling ×10", which may already behave well on novel data due to model averaging.

As shown in Figure 8 , we find that for G-distill ×10, the foreign NLL improves beyond the original ensemble, while the familiar NLL still falls slightly behind.

Considering smoothing trade-off for all methods, G-distill ×10 still fall behind an ensemble.

Our E99 confident error rates on both familiar and novel data are usually similar to or lower than the ensemble and distilling ×10.

Our accuracy or mAP is on the same level as the ensemble, except for Figure 8(c) .

These indicate that without τ smoothing, the G-distill ensembles are better behaved on novel samples than the original ensembles, although the former tend to be less confident with familiar data.

(e) Animal recognition with ImageNet, using DenseNet161Figure 8: Using an ensemble of G-distilled models to further boost the performance.

Although we do obtain a better novel NLL compared to the ensemble, we usually lag behind the ensemble considering the smoothing trade-off.

@highlight

Deep networks are more likely to be confidently wrong when testing on unexpected data. We propose an experimental methodology to study the problem, and two methods to reduce confident errors on unknown input distributions.

@highlight

Proposes two ideas for reducing overconfident wrong predictions: "G-distillation" of am ensemble with extra unsupervised data and Novelty Confidence Reduction using novelty detector

@highlight

The authors propose two methods for estimating classification confidence on novel unseen data distributions. The first idea is to use ensemble methods as the base approach to help identify uncertain cases and then use distillation methods to reduce the ensemble into a single model mimicking behavior of the ensemble. The second idea is to use a novelty detector classifier and weight the network output by the novelty score.
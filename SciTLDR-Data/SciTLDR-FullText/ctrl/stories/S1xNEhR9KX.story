Neural networks are vulnerable to small adversarial perturbations.

Existing literature largely focused on understanding and mitigating the vulnerability of learned models.

In this paper, we demonstrate an intriguing phenomenon about the most popular robust training method in the literature, adversarial training: Adversarial robustness, unlike clean accuracy, is sensitive to the input data distribution.

Even a semantics-preserving transformations on the input data distribution can cause a significantly different robustness for the adversarial trained model that is both trained and evaluated on the new distribution.

Our discovery of such sensitivity on data distribution is based on a study which disentangles the behaviors of clean accuracy and robust accuracy of the Bayes classifier.

Empirical investigations further confirm our finding.

We construct semantically-identical variants for MNIST and CIFAR10 respectively, and show that standardly trained models achieve comparable clean accuracies on them, but adversarially trained models achieve significantly different robustness accuracies.

This counter-intuitive phenomenon indicates that input data distribution alone can affect the adversarial robustness of trained neural networks, not necessarily the tasks themselves.

Lastly, we discuss the practical implications on evaluating adversarial robustness, and make initial attempts to understand this complex phenomenon.

Neural networks have been demonstrated to be vulnerable to adversarial examples BID22 BID3 .

Since the first discovery of adversarial examples, great progress has been made in constructing stronger adversarial attacks BID12 BID18 BID17 BID6 .

In contrast, defenses fell behind in the arms race BID5 BID1 .

Recently a line of works have been focusing on understanding the difficulty in achieving adversarial robustness from the perspective of data distribution.

In particular, BID24 demonstrated the inevitable tradeoff between robustness and clean accuracy in some particular examples.

BID20 showed that the sample complexity of "learning to be robust" learning could be significantly higher than that of "learning to be accurate".In this paper, we contribute to this growing literature from a new angle, by studying the relationship between adversarial robustness and the input data distribution.

We focus on the adversarial training method, arguably the most popular defense method so far due to its simplicity, effectiveness and scalability BID12 BID13 BID15 BID17 BID8 .

Our main contribution is the finding that adversarial robustness is highly sensitive to the input data distribution:A semantically-lossless shift on the data distribution could result in a drastically different robustness for adversarially trained models.

Note that this is different from the transferability of a fixed model that is trained on one data distribution but tested on another distribution.

Even retraining the model on the new data distribution may give us a completely different adversarial robustness on the same new distribution.

This is also in sharp contrast to the clean accuracy of standard training, which, as we show in later sections, is insensitive to such shifts.

To our best knowledge, our paper is the first work in the literature that demonstrates such sensitivity.

Our investigation is motivated by the empirical observations on the MNIST dataset and the CIFAR10 dataset.

In particular, while comparable SOTA clean accuracies (the difference is less than 3%) are achieved by MNIST and CIFAR10 BID10 , CIFAR10 suffers from much lower achievable robustness than MNIST in practice.

1 Results of this paper consist of two parts.

First in theory, we start with analyzing the difference between the regular Bayes error and the robust error, and show that the regular Bayes error is invariant to invertible transformations of the data distribution, but the robust error is not.

We further prove that if the input data is uniformly distributed, then the perfect decision boundary cannot be robust.

However, we also manage to find a robust model for the binarized MNIST dataset (semantically almost identical to MNIST, later described in Section 3).

The certification method by BID26 guarantees that this model achieves at most 3% robust error.

Such a sharp contrast suggests the important role of the data distribution in adversarial robustness, and leads to our second contribution on the empirical side: we design a series of augmented MNIST and CIFAR10 datasets to demonstrate the sensitivity of adversarial robustness to the input data distribution.

Our finding of such sensitivity raises the question of how to properly evaluate adversarial robustness.

In particular, the sensitivity of adversarial robustness suggests that certain datasets may not be sufficiently representative when benchmarking different robust learning algorithms.

It also raises serious concerns about the deployment of believed-to-be-robust training algorithm in a real product.

In a standard development procedure, various models (for example different network architectures) would be prototyped and measured on the existing data.

However, the sensitivity of adversarial robustness makes the truthfulness of the performance estimations questionable, as one would expect future data to be slightly shifted.

We illustrate the practical implications in Section 4 with two practical examples: 1) the robust accuracy of PGD trained model is sensitive to gamma values of gamma-corrected CIFAR10 images.

This indicates that image datasets collected under different light conditions may have different robustness properties; 2) both as a "harder" version of MNIST, the fashion-MNIST BID27 and edge-fashion-MNIST (an edge detection variant described in Section 4.2) exhibit completely different robustness characteristics.

This demonstrates that different datasets may give completely different evaluations for the same algorithm.

Finally, our finding opens up a new angle and provides novel insights to the adversarial vulnerability problem, complementing several recent works on the issue of data distributions' influences on robustness.

BID24 hypothesize that there is an intrinsic tradeoff between clean accuracy and adversarial robustness.

Our studies complement this result, showing that there are different levels of tradeoffs depending on the characteristics of input data distribution, under the same learning settings (training algorithm, model and training set size).

BID20 show that different data distributions could have drastically different properties of adversarially robust generalization, theoretically on Bernoulli vs mixtures of Gaussians, and empirically on standard benchmark datasets.

From the sensitivity perspective, we demonstrate that being from completely different distributions (e.g. binary vs Gaussian or MNIST vs CIFAR10) may not be the essential reason for having large robustness difference.

Gradual semantics-preserving transformations of data distribution can also cause large changes to datasets' achievable robustness.

We make initial attempts in Section 5 to further understand this sensitivity.

We investigated perturbable volume and inter-class distance as the natural causes of the sensitivity; model capacity and sample complexity as the natural remedies.

However, the complexity of the problem has so far defied our efforts to give a definitive answer.

We specifically consider the image classification problem where the input data is inside a high dimensional unit cube.

We denote the data distribution as a joint distribution P(x, y), where DISPLAYFORM0 is the number of pixels, and y ∈ {1, 2, . . .

, k} is the discrete label.

We assume the support of x is the whole pixel space [0, 1] d .

When x is a random noise (or human perceptually unclassifiable image), one can think of P(y | x) being closed to uniform distribution on labels.

In the standard setting, the samples (x i , y i ) can be interpreted as x i is independently sampled from the marginal distribution P(x), and then y i is sampled from P(x | x i ).

In this paper, we discuss P(x)'s influences on adversarial robustness, given a fixed P(y|x).In our experiments, we only discuss the whitebox robustness, as it represents the "intrinsic" robustness.

We use models learned by adversarially augmented training BID17 (PGD training) , which has the SOTA whitebox robustness.

We consider bounded ∞ attack as the attack for evaluating robustness for 2 reasons: 1) PGD training can defend against ∞ relatively well, while for other attacks, how to train a robust model is still an open question; 2) in the image domain ∞ attack is the mostly widely researched attack.

Let H denote the universal set of all the measurable functions.

Given a joint distribution P(x, y) on the space X × Y, we define the Bayes error R * = inf h∈H E P(x,y) L(y; h(x)) = R * (P(x, y)), where L is the objective function.

In other words, Bayes error is the error of the best possible classifier we can have, h * , without restriction on the function space of classifiers.

We further define (adversarial) robust error RR(h) = E P(x,y) max δ ∞< L(y; h(x + δ)) = RR(P(x, y)).

We denote RR * = RR(h * ) to be the robust error achieved by the Bayes classifier h * .

For simplicity, we assume our algorithm can always learn h * , which reduces clean accuracy to be (1 − Bayes error), and robust accuracy of the Bayes classifier to be (1 − RR * ).

As mentioned in the introduction, although the SOTA clean accuracies are similar for MNIST and CIFAR10, the robust accuracy on CIFAR10 is much more difficult to achieve, which indicates the different behaviors of the clean accuracy and robust accuracy.

The first result in this section is to further confirm this indication in a simple setting, where the clean accuracy remains the same but the robust accuracy completely changes under a distribution shift.

Based on results from the concentration of measure literature, we further show that under uniform distribution, no algorithm can achieve good robustness, as long as they have high clean accuracy.

On the other hand, we examine the performance of a verifiable defense method on binarized MNIST (pixels values rounded to 0 and 1), and the result suggests the exact opposite: provable adversarial robustness on a MNISTlike dataset is achievable.

Such contrast thus suggests the important role of the data distribution in achieving adversarial robustness.

One immediate result is that Bayes error remains the same under any distribution shift induced by an injective map T : X → X .

To see that, simply note that T −1 exists and h * • T −1 gives the same Bayes error for the shifted distribution.

However, such invariance property does not hold for the robust error of the Bayes classifier.

Furthermore, the following two examples show that Bayes error can have completely different behavior from its robust error.

Although both examples have 0 Bayes error, they have completely different robust errors.

Example 1.

Assume x is uniformly distributed in [0, 1] d and y = 1, for all x with x e 1 > 1/2 and y = 0, for x e 1 ≤ 1/2, where e 1 is the one-hot vector.

We use the 0-1 loss here.

Note that the Bayes error decision boundaries are given by the following hyperplane: DISPLAYFORM0 under the budget δ ∞ < .

In this case, the robust error is tolerable and relatively robust measured by the fraction of points that are successfully attacked, 2 .Moreover, consider an injective map T which maps {x : x e 1 > 1/2} to {x : DISPLAYFORM1 The Bayes error on the new distribution remains 0, as T is invertible.

In contrast, the robust error is much worse.

In fact, DISPLAYFORM2 Remark 2.1.

Note that here the robust error of the Bayes classifier will grow to 1 as the dimensionality increases, for a fixed budget .

Example 1 shows that good clean accuracy does not necessary lead to good robust accuracy.

In contrast, we will show in this section that achieving a good robust accuracy is impossible given uniformly distributed data, as long as we ask for good clean accuracies.

Our tool are classical results from the concentration of measure BID16 .

DISPLAYFORM0 DISPLAYFORM1 For any B ⊂ B d , with P(B) ≥ 1/2, DISPLAYFORM2 where δ 2 ( ) = 1 − 1 − 2 4 and Φ is the standard normal cumulative distribution function.

Based on Theorem 2.1 we can now show that under some circumstances, no algorithm that achieves can perfect clean accuracy can also achieve a good robust accuracy.

Example 2 (Vulnerability Guarantee).

Consider the joint distribution P(x, y), where the input data x is uniformly distributed on [0, 1] d and label y has 10 classes.

Further assume the marginal distribution of y is also uniform 3 .

Theorem 2.1 implies that under 2 adversarial attack with = 0.5, at least 94 % of the samples are ether wrongly classified or can be successfully attacked for a classifier with perfect clean accuracy.

DISPLAYFORM3 On the one hand, Theorem 2.1 and Example 2 suggest that the uniform distribution on [0, 1] d enjoys more robustness than the uniform distribution on B d , and it is not affected by the high dimensionality.

This may partially explain why MNIST is more adversarially robust than CIFAR10, as the distribution of x in CIFAR10 is "closer" to DISPLAYFORM4 On the other hand, while not completely sharp, they also suggest the intrinsic difficulty in achieving good robust accuracy.

Note that one limit of Theorem 2.1 and Example 2 is the uniform distribution assumption, which is surely not true for natural images.

Indeed, although rigorously developed, Theorem 2.1 and Example 2 do not explain certain empirical observations.

Following Wong and Kolter (2018), we train a provably 4 robust model on a binarized MNIST dataset (bMNIST) 5 .

Our experiments shows that the learned model achieves 3.00% provably robust error on bMNIST test data, while maintaining 97.65% clean accuracy.

Details of this experiment in described in Appendix B.2.The above MNIST experiment and Example 2 suggest the essential role of the data distribution in achieving good robust and clean accuracies.

While it is hard to completely answer the question what geometric properties differentiate the concentration rates between the ball/cube in high dimension and the distribution of bMNIST, we remark that one obvious difference is the distance distributions in both spaces.

Could the distance distributions explain the differences in clean and robust accuracies?

Note that the same method can only achieve 37.70% robust error on original MNIST data, and even higher error on CIFAR10, which further supports this hypothesis.

In the rest of this paper, we further investigate the dependence of robust accuracy on the distribution of real data.

Section 2.2 clearly suggests that the data distribution plays an essential role in the achievable robust accuracy.

In this section we carefully design a series of datasets and experiments to further study its influence.

One important property of our new datasets is that they have different P(x)'s while keep P(y|x) reasonably fixed, thus these datasets are only different in a "semantic-lossless" shift.

Our experiments reveal an unexpected phenomenon that while standard learning methods manage to achieve stable clean accuracies across different data distributions under "semantic-lossless" shifts, however, adversarial training, arguably the most popular method to achieve robust models, loses this 3 but their joint distribution is not necessary uniform.

4 "Provably" means that the robust accuracy of the model can be rigorously proved.

5 It is created by rounding all pixel values to 0 or 1 from the original MNIST desirable property, in that its robust accuracy becomes unstable even under a "semantic-lossless" shift on the data distribution.

We emphasize that different from preprocessing steps or transfer learning, here we treat the shifted data distribution as a new underlying distribution.

We both train the models and test the robust accuracies on the same new distribution.

We now explain how the new datasets are generated under "semantic-lossless" shifts.

In general, MNIST has a more binary distribution of pixels, while CIFAR10 has a more continuous spectrum of pixel values, as shown in Figure 1a and 1b.

To bridge the gap between these two datasets that have completely different robust accuracies, we propose two operations to modify their distribution on x: smoothing and saturation, as described below.

We apply different levels of "smoothing" on MNIST to create more CIFAR-like datasets, and different levels of "saturation" on CIFAR10 to create more "binary" ones.

Note that we would like to maintain the semantic information of the original data, which means that such operations should be semantics-lossless and not arbitrarily wide.

Smoothing is applied on MNIST images, to make images "less binary".

Given an image x i , its smoothed versionx i (s) is generated by first applying average filter of kernel size s to x i to generate an intermediate smooth image, and then take pixel-wise maximum between x i and the intermediate smooth image.

Our MNIST variants include the binarized MNIST and smoothed MNIST with different kernel sizes.

As shown in Figure 1c , all MNIST variants still maintain the semantic information in MNIST, which indicates that P(y |x (s) ) should be similar to P(y | x).

It is thus reasonable to assume that y i is approximately sampled from P(y |x (s) ), and as such we assign y i as the label ofx (s) .

Note that all the data points in the binarized MNIST are on the corners of the unit cube.

For the smoothed versions, pixels on the digit boundaries are pushed off the corner of the unit cube.

Saturation of the image x is denoted by x (p) , and the procedure is defined as below: DISPLAYFORM0 where all the operations are pixel-wise and each element of x is that it pushes x to the corners of the data domain where the pixel values are either 0 or 1 when p ≥ 2, and pull the data to the center of 0.5 when p ≤ 2.

When p = 2 it does not change the image, and when p = ∞ it becomes binarization.

In this section we use the smoothing and saturation operations to manipulate the data distributions of MNIST and CIFAR10, and show empirical results on how data distributions affects robust accuracies of neural networks trained on them.

Since we are only concerned with the intrinsic robustness of neural networks models, we do not consider methods like preprocessing that tries to remove perturbations or randomizing inputs.

We perform standard neural network training on clean data to measure the difficulty of the classification task, and projected gradient descent (PGD) based adversarial training BID17 to measure the difficulty to achieve robustness.

By default, we use LeNet5 on all the MNIST variants, and use wide residual networks BID29 with widen factor 4 for all the CIFAR10 variants.

Unless otherwise specified, PGD training on MNIST variants and CIFAR10 variants all follows the settings in BID17 .

Details of network structures and training hyperparameters can be found in Appendix B.We evaluate the classification performance using the test accuracy of standardly trained models on clean unperturbed examples, and the robustness using the robust accuracy of PGD trained model, which is the accuracy on adversarially perturbed examples.

Although not directly indicating robustness, we report the clean accuracy on PGD trained models to indicate the tradeoff between being accurate and robust.

To understand whether low robust accuracy is due to low clean accuracy or vulnerability of model, we also report robustness w.r.t.

predictions, where the attack is used to perturb against the model's clean prediction, instead of the true label.

We use ∞ untargeted PGD attacks BID17 as our adversary, since it is the strongest attack in general based on our BID17 .

We use the PGD attack implementation from the AdverTorch toolbox BID7 .3.3 SENSITIVITY OF ROBUST ACCURACY TO DATA TRANSFORMATIONS Results on MNIST variants are presented in FIG4 6 .

The clean accuracy of standard training is very stable across different MNIST variants.

This indicates that their classification tasks have similar difficulties, if the training has no robust considerations.

When performing PGD adversarial training, clean accuracy drops only slightly.

However, both robust accuracy and robustness w.r.t.

predictions drop significantly.

This indicates that as smooth level goes up, it is significantly harder to achieve robustness.

Note that for binarized MNIST with adversarial training, the clean accuracy and the robust accuracy are almost the same.

Indicating that getting high robust accuracy on binarized MNIST does not conflict with achieving high clean accuracy.

This result conforms with results of provably robust model having high robustness on binarized MNIST described in Section 2.CIFAR10 result tell a similar story, as reported in FIG4 6 .

For standard training, the clean accuracy maintains almost at the original level until saturation level 16, despite that it is already perceptually very saturated.

In contrast, PGD training has a different trend.

Before level 16, the robust accuracy significantly increases from 43.2% until 79.7%, while the clean test accuracy drops only in a comparatively small range, from 85.4% to 80.0%.

After level 16, PGD training has almost the same clean accuracy and robust accuracy.

However, robustness w.r.t.

predictions still keeps increasing, which again indicates the instability of the robustness.

On the other hand, if the saturation level is smaller than 2, we get worse robust accuracy after PGD training, e.g. at saturation level 1 the robust accuracy is 33.0%.

Simultaneously, the clean accuracy maintains almost the same.

Note that after saturation level 64 the standard training accuracies starts to drop significantly.

This is likely due to that high degree of saturation has caused "information loss" of the images.

Models trained on highly saturated CIFAR10 are quite robust and the gap between robust accuracy and robustness w.r.t.

predictions is due to lower clean accuracy.

In contrast, In MNIST variants, the robustness w.r.t.

predictions is always almost the same as robust accuracy, indicating that drops in robust accuracy is due to adversarial vulnerability.

From these results, we can conclude that robust accuracy under PGD training is much more sensitive than clean accuracy under standard training to the differences in input data distribution.

More importantly, a semantically-lossless shift on the data transformation, while not introducing any unexpected risk for the clean accuracy of standard training, can lead to large variations in robust accuracy.

Such previously unnoticed sensitivity raised serious concerns in practice, as discussed in the next section.

Given adversarial robustness' sensitivity to input distribution, we further demonstrate two practical implications: 1) Robust accuracy could be sensitive to image acquisition condition and preprocessing.

This leads to unreliable benchmarks in practice; 2) When introducing new dataset for benchmarking adversarial robustness, we need to carefully choose datasets with the right characteristics.4.1 ROBUST ACCURACY IS SENSITIVE TO GAMMA CORRECTION The natural images are acquired under different lighting conditions, with different cameras and different camera settings.

They are usually preprocessed in different ways.

All these factors could lead to mild shifts on the input distribution.

Therefore, we might get very different performance measures when performing adversarial training on images taken under different conditions.

In this section, we demonstrate this phenomenon on variants of CIFAR10 images under different gamma mappings.

These variants are then used to represent image dataset acquired under different conditions.

Gamma mapping is a simple element-wise operation that takes the original image x, and output the gamma mapped imagex (γ) by performingx (γ) = x γ .

Gamma mapping is commonly used to adjust the exposure of an images.

We refer the readers to Szeliski (2010) on more details about gamma mappings.

FIG2 shows variants of the same image processed with different gamma values.

Lower gamma value leads to brighter images and higher gamma values gives darker images, since pixel values range from 0 to 1.

Despite the changes in brightness, the semantic information is preserved.

We perform the same experiments as in the saturated CIFAR10 variants experiment in Section 3.

The results are displayed in FIG2 .

Accuracies on clean data almost remain the same across different gamma values.

However, under PGD training, both accuracy and robust accuracy varies largely following different gamma values.

These results should raise practitioners' attention on how to interpret robustness benchmark "values".

For the same adversarial training setting, the robustness measure might change drastically between image datasets with different "exposures".

In other words, if a training algorithm achieves good robustness on one image dataset, it doesn't necessarily achieve similar robustness on another semantically-identical but slightly varied datasets.

Therefore, the actual robustness could either be significantly underestimated or overestimated.

This raises the questions on whether we are evaluating image classifier robustness in a reliable way, and how we choose benchmark settings that can match the real robustness requirements in practice.

This is an important open question and we defer it to future research.

As discussed, evaluating robustness on a suitable dataset is important.

Here we use fashion-MNIST (fMNIST) BID27 and edge-fashion-MNIST (efMNIST) as examples to analyze characteristics of "harder" datasets.

The edge-fashion MNIST is generated by running Canny edge detector BID4 with σ = 1 on the fashion MNIST images.

FIG2 shows examples of fMNIST and efMNIST.

We performed the same standard training and PGD training experiments on both fMNIST and efMNIST as we did on MNIST.

FIG2 shows the results.

We can see that fMNIST exhibit similar behavior to CIFAR10, where the test accuracy is significantly affected by PGD training and the gap between robust accuracy and accuracy is large.

On the other hand, efMNIST is closer to the binarized MNIST: the accuracy is affected very little by PGD training, along with an insignificant difference between robust accuracy and accuracy.

Both fMNIST and efMNIST can be seen as a "harder" MNIST, but they are harder in different ways.

One one hand, since efMNIST results from the edge detection run on fMNIST, it contains less information.

It is therefore harder to achieve higher accuracy on efMNIST than on fMNIST, where richer semantics is accessible.

However, fMNIST's richer semantics makes it better resembles natural images' pixel value distribution, which could lead to increased difficulty in achieving adversarial robustness.

efMNIST, on the other hand, can be viewed as a set of "more complex binary symbols" compared to MNIST or binarized MNIST.

It is harder to classify these more complex symbols.

However, it is easy to achieve high robustness due to the binary pixel value distribution.

To sum up, when introducing new dataset for adversarial robustness, we should not only look for a "harder" one, but we also need to consider whether the dataset is "harder in the right way".

In this section, we make initial attempts to understand the sensitivity of adversarial robustness.

We use CIFAR10 variants as the running example, but these analyses apply to MNIST variants as well.

Saturation pushes pixel values towards 0 or 1, i.e. towards the corner of unit cube, which naturally suggests two potential factors for the change in robustness.

1) the "perturbable volume" decreases; 2) distances between data examples increases.

Intuitively, both could be related to the increasd robustness.

We analyze them and show that although they are correlated with robustness change, none of them can fully explain the observed phenomena.

We then further examine the possibility of increasing robust accuracy on less robust datasets by having larger models and more data.

5.1 ON THE INFLUENCE OF PERTURBABLE VOLUME Saturation moves the pixel values towards 0 and 1, therefore pushing the data points to the corners of the unit cube input domain.

This makes the valid perturbation space to be smaller, since the space of perturbation is the intersection between the -∞ ball and the input domain.

Due to high dimensionality, the volume of "perturbable region" changes drastically across different saturation levels.

For example, the average log perturbable volume 7 of original CIFAR10 images are -12354, and the average log perturbable volume of ∞-saturated CIFAR10 is -15342, which means that the perturbable volume differs by a factor of 2 2990 = 2 (−12352−(−15342)) .

If the differences in perturbable volume is a key factor on the robustness' sensitivity, then by allowing the attack to go beyond the domain boundary 8 , the robust accuracies across different saturation levels should behave similarly again, or at least significantly differ from the case of box constrained attacks.

We performed PGD attack allowing the perturbation to be outside of the data domain boundary, and compare the robust accuracy to what we get for normal PGD attack within domain boundary.

We found that the expected difference is not observed, which serves as evidence that differences in perturbable volume are not causing the differences in robustness on the tested MNIST and CIFAR10 variants.

When saturation pushes data points towards data domain boundaries, the distances between data points increase too.

Therefore, the margin, the distance from data point to the decision boundary, could also increase.

We use the "inter-class distance" as an approximation.

Inter-class distance 9 characterizes the distances between each class to rest of classes in each dataset.

Intuitively, if the distances between classes are larger, then it should be easier to achieve robustness.

We also observed (in Appendix D.2.1 FIG5 ) that inter-class distances are positively correlated with robust accuracy.

However, we also find counter examples where datasets having the same inter-class distance exhibit different robust accuracies.

Specifically, We construct scaled variants of original MNIST and binarized MNIST, such that their inter-class distances are the same as smooth-3, smooth-4, smooth-5 MNIST.

The scaling operation is defined asx (α) = α(x − 0.5) + 0.5, where α is the scaling coefficient.

When α < 1.

each dimension of x is pushed towards the center with the same rate.

TAB1 shows the results.

We can see that although having the same interclass distances, the smoothed MNIST is still less robust than the their correspondents of scaled binarized MNIST and original MNIST.

This indicates the complexity of the problem, such that a simple measure like inter-class distance cannot fully characterize robustness property of datasets, at least on the variants of MNIST.

* for the given data distribution.

In the case RR * is not yet achieved, there is a nonexhaustive list that we can improve upon: 1) use better training/learning algorithms; 2) increase the model capacity; 3) train on more data.

Finding a better learning algorithm is beyond the scope of this paper.

Here we inspect 2) and 3) to see if it is possible to improve robustness by having larger model and more data.

For model capacity, we use differently sized LeNet5 by multiplying the number of channels at each layer with different widen factors.

These factors include 0.125, 0.25, 0.5, 1, 2, 4.

On CIFAR10 variants, we use WideResNet with widen factors 0.25, 1 and 4.

For sample complexity, we follow the practice in Section 3 except that we use a weight decay value of 0.002 to prevent overfitting.

For both MNIST and CIFAR10, we test on 1000, 3000, 9000, 27000 and entire training set.

Both model capacity and sample complexity results are shown in FIG3 .For MNIST, both training and test accuracies of clean training are invariant to model sizes, even we only use a model with widen factor 0.125.

In slight contrast, both the training and test accuracy of PGD training increase as the model capacity increases, but it plateaus after widen factor 1 at an almost 100% accuracy.

For robust accuracy, training robust accuracy kept increasing as model gets larger until the value is close to 100%.

However, test robust accuracy stops increasing after widen factor 1, additional model capacity leads to larger (robust) generalization gap.

When we vary the size of training set, the model can always fit the training set well to almost 100% clean training accuracy under standard training.

The clean test accuracy grows as the training set size get larger.

Training set size has more significant impact on robust accuracies of PGD trained models.

For most MNIST variants except for binarized MNIST, training robust accuracy gradually drops, and test robust accuracy gradually increases as the training set size increases.

This shows that when training set size is small, PGD training overfits to the training set.

As training set gets larger, the generalization gap becomes smaller.

Both training and test robust accuracies plateau after training set size reaches 27000.

Indicating that increasing the training set size might not help in this setting.

In conclusion, for MNIST variants, increasing training set size and model capacity does not seem to help beyond a certain point.

Therefore, it is not obvious on how to improve robustness on MNIST variants with higher smoothing levels.

CIFAR10 variants exhibit similar trends in general.

One notable difference is that for PGD training, the training robust accuracy does not plateau as model size increases.

However the test robust accu- racy plateaus after widen factor 1.

Also when training set size increases, the training robust accuracy drops and test robust accuracy increases with no plateau present.

These together suggest that having more training data and training a larger model could potentially improve the robust accuracies on CI-FAR10 variants.

One interesting phenomenon is that binarized MNIST and ∞-saturated CIFAR10 has different sample complexity property, despite both being "cornered" datasets.

This indicates that the although binarization can largely influence robustness, it does not decide every aspect of it, such as sample complexity.

This complex interaction between the classification task and input data distribution is still to be understood further.

In this paper we provided theoretical analyses to show the significance of input data distribution in adversarial robustness, which further motivated our systematic experiments on MNIST and CI-FAR10 variants.

We discovered that, counter-intuitively, robustness of adversarial trained models are sensitive to semantically-preserving transformations on data.

We demonstrated the practical implications of our finding that the existence of such sensitivity questions the reliability in evaluating robust learning algorithms on particular datasets.

Finally, we made initial attempts to understand this sensitivity.

DISPLAYFORM0 Then we apply Markov's inequality, for all real number t > 0: DISPLAYFORM1 Finally, we observe that the longest (in terms of 2 norm) such ∞ attacks vector to HP 2 are parallel to the normal vector 1 to HP 2 .

They have 2 distance √ d. The set these attacks cover is characterized by {x DISPLAYFORM2 Let t = 2 d, we have: DISPLAYFORM3 In the case of zero-one loss, RR DISPLAYFORM4 A.2 PROOF FOR THEOREM 2.1Proof. (First Inequality for Cube) The proof here follows that of BID16 , but we track of the tight constants so as to give tighter adversarial robustness calculations.

Let Φ be one dimensional standard normal cumulative distribution function and let µ d denote d dimensional Gaussian measures.

Consider the map T : DISPLAYFORM5 T pushes forward µ d defined on R d into a probability measure P on (0, 1) d : DISPLAYFORM6 for A ⊂ (0, 1) d .

Next we have the following Gaussian isoperimetric/concentration inequality BID16 : DISPLAYFORM7 Now for A ⊂ (0, 1) d , we have: DISPLAYFORM8 where the first inequality follows from that T has Lipschitz constant DISPLAYFORM9 , and thus T −1 has Lipschitz constant √ 2π; and the second one follows from Gaussian isoperimetric inequality.

DISPLAYFORM10 Additionally, the inequality Φ(x) ≥ 1 − e x 2 2 implies the last inequality in the theorem.

We first define the notion of modulus of convexity for a normed space, in this case 2 : DISPLAYFORM0 The important property about δ 2 ( ) is that there is a constant C such that: DISPLAYFORM1 By elementary algebraic calculuation, We can take C = 2− √ 3 3 .

By Equation (2.25) in BID16 , DISPLAYFORM2 The LeNet5 (widen factor 1) is composed of 32-channel conv filter + ReLU + size 2 max pooling + 64-channel conv filter + ReLU + size 2 max pooling + fc layer with 1024 units + ReLU + fc layer with 10 output classes.

We do not preprocess MNIST images before feeding into the model.

For training LeNet5 on MNIST variants, we use the Adam optimizer with an initial learning rate of 0.0001 and train for 100000 steps with batch size 50.

We use the WideResNet-28-4 as described in BID29 for our experiments, where 28 is the depth and 4 is the widen factor.

We use "per image standardization" 10 to preprocess CIFAR10 images, following BID17 .For training WideResNet on CIFAR10 variants, we use stochastic gradient descent with momentum 0.9 and weight decay 0.0002.

We train 80000 steps in total with batch size 128.

The learning rate is set to 0.1 at step 0, 0.01 at step 40000, and 0.001 at step 60000.We performed manual hyperparameter search for our initial experiment and do not observe improvements over the above settings.

Therefore we used these settings throughout the all the experiments in the paper unless otherwise indicated.

For the linear programming based provably robust model BID26 (LP-robust model).

We trained a ConvNet identical to the one in the original paper.

It has 2 convolutional layers, with 16 and 32 channels, each with a stride of 2; and 2 fully connected layers, the first one maps the flattened convolution features to hidden dimension 100, the second maps to 10 logit units.

We use ReLUs as the nonlinear activation and there is no max pooling in the network.

We train for 100 epochs with batch size 50.

The first 50 epochs are warm start epochs where epsilon increases from 0.01 to 0.3 linearly.

We use Adam optimizer BID14 ) with a constant learning rate of 0.001.

We listed exact numbers of experiments involved in the main body in TAB2 , 3, 4 and 5.

One natural hypothesis about the reason of achieving better robustness could be that it is the effect of the boundaries.

Indeed, if the data distribution is closer to the data domain boundary, the valid perturbation space, the -∞ ball may be restricted since it will intersect with the boundary.

We then test the correlation between "how close the data distribution is to the boundary" and its achievable robustness, by examining the volume of the allowed perturbed box across different datasets.

The intersection of the data domain, unit cube [0, 1] d , with the allowed perturbation space, -∞ ball DISPLAYFORM0 are the indexes over input dimensions.

The size of the available perturbation space at x and is defined by the volume of this hyperrectangle: DISPLAYFORM1 In high dimensional space, when is fixed, this volume varies greatly based on the location of x. For example, if x is on one of the corners of the unit cube, Vol(x corner , ) = d .

If each dimension of x is at least away from all the data boundaries, then the volume of the hyperrectangle is Vol(x inside , ) = (2 ) d .

Therefore there can be 2 d times difference of perturbable space between different data points.

As shown in the average log perturbable volumes TAB6 , we can see that different variations of datasets has significantly different perturbable volumes, with the same trend with previously described.

It is notable that for the original CIFAR10 datasets has log volume -12354, which is very close to the -12270.

The different of 84 bits indicates on average, the perturbation space is 2 84 smaller than the full -∞ ball if there is no intersection with the data domain boundary.

Volume differences between different saturation or smooth level can be interpreted in the similar way.

Note that for CIFAR10 images with large saturation, although they appear similar to human, they actually have very large differences in terms of perturbable volumes.

If the perturbable volume hypothesis holds, then we should observe significantly lower accuracy under PGD attack if we allow perturbation outside of data domain boundary.

Since this greatly increases the perturbable volume.

We measure the accuracy under PGD attack with and without considering data domain boundary for both MNIST and CIFAR10 variants.

The results are shown in TAB7 .

"With considering boundary" corresponds to regular PGD attacks.

We can see that allowing PGD to perturb out of bound do not reduce accuracy under attack.

This means that PGD is not able to use the significantly larger additional volumes even for binarized MNIST or highly saturated CIFAR10, whose data points are on or very close to the corner.

In some cases, allowing perturbation outside of domain boundary makes the attack slightly less effective.

This might be due to that data domain boundary constrained the perturbation to be in an "easier" region.

This might seem surprising considering the huge difference in perturbable volumes, these results conform with empirical results in previous research BID12 BID25 that adversarial examples appears in certain directions instead of being distributed in small pockets across space.

Therefore, the perturbable volume hypothesis is rejected.

Note that we choose 2 distance for inter-class distance, instead of using the ∞ which measures the robustness.

This is because ∞ -distance between data examples is essentially the max over the per pixel differences, which is always very close to 1.

Therefore the ∞ -distance between data examples is not really representative / distinguishable.

FIG5 shows the inter-class distances (averaged over all classes) calculated on MNIST and CI-FAR10 variants.

The binarized MNIST has a significantly larger inter-class distance.

As smoothing kernel size increases, the distance also decrease slightly.

On CIFAR10 variants, as the saturation level gets higher, the inter-class distance increases monotonically.

We also directly plot inter-class distance vs robust accuracy on MNIST and CIFAR10 variants.

In general, inter-class distance shows a strong positive correlation with robust accuracy under these transformations.

With one exception that original MNIST has smaller inter-class distance, but is sightly more robust than smooth-2 MNIST.

This, together with the counter examples we gave in TAB1 , suggests that inter-class distance cannot fully explain the robust variation across different dataset variants.

We attempt to understand the relation between the inter-class distance of a dataset and its achievable robustness in this section.

We first illustrate our intuition in a synthetic experiment, where a ReLU network is trained to perfectly separate 2 concentric spheres BID11 , as shown in Figure 6 .

Here the inter-class distance is the width of the ring between two spheres.

In such example, adversarial training is actually closely related to the inter-class distance of the data.

In fact, in the simple setting where the classifier is linear, it has been shown in BID28 that adversarial training, as a particular form of robust optimization, is equivalent to maximizing the classification margins.

Following this intuition, one can easily see that the effect of adversarial training is to push two spheres close to each other, and requires the network to perfectly separate the new spheres with much smaller inter-class.

Intuitively, when the inter-class distance is large, i.e. the gap between two spheres are large, a reasonable model should be able to achieve good standard accuracy.

We have also observed such phenomenon on original MNIST and saturated CIFAR10 (say level 16).

As the inter-class distance gets smaller, although the model capacity could still be enough for the standard training, it may no longer be enough for adversarial training, upon which we would observe that although the test accuracies stay similar, accuracies under adversarial attack significantly would drop.

We have also seen similar behavior on smooth MNIST data and smaller level of saturated CIFAR10 data.

Finally, when the inter-class distance is so small such that even a high clean test accuracy may be difficult to achieve.

Considering robust accuracy as the clean accuracy with a smaller gap between the spheres, the next theorem provides a theoretical guarantee in relating together the difficulty of attaining good accuracy under attack and the model capacity BID2 , verifying our intuition above.

Note that one way to measure the capacity of a ReLU network is by counting the number of its induced piece-wise linear region, which is closely related to the number of facets of its decision boundary.

Theorem D.1.

Let d(K, L) between symmetric convex bodies K and L denote the least positive d for which there is a linear imageL of L such thatL ⊂ K ⊂ dL. Let K be a (symmetric) polytope in R n with d(K, B n 2 ) = d. Then K has at least e n/(2d 2 ) facets.

On the other hand, for each n, there is a polytope with 4n facets whose distance from the ball is at most 2.

Figure 6 : Illustration of the relationship between the inter-class distance and the required model capacity.

Left: when distance is small, a small capacity polytope classifier could separate original data; middle: when distance is small, the small capacity polytope classifier is not able to separate data points "robustly", but a more complex nonlinear classifier could; right:when distance is large, the small capacity polytope classifier can separate data points "robustly".The above analysis is partially supported by our experiments on model capacity in Section 5.3.

However, as we've shown in Section 5.2, the nature of the problem is complex and more conclusive statements requires further research.

<|TLDR|>

@highlight

Robustness performance of PGD trained models are sensitive to semantics-preserving transformation of image datasets, which implies the trickiness of evaluation of robust learning algorithms in practice.

@highlight

Paper clarifies the difference between clean and robust accuracy and shows that changing the marginal distribution of the input data P(x) while preserving its semantic P(y|x) affects the robustness of the model.

@highlight

This paper investigates the origin of the lack of robustness of classifiers to perturbations of adversarial inputs under l-inf bounded perturbations.
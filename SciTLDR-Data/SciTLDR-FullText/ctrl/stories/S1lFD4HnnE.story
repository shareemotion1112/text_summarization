Neural networks are vulnerable to small adversarial perturbations.

While existing literature largely focused on the vulnerability of learned models, we demonstrate an intriguing phenomenon that adversarial robustness, unlike clean accuracy, is sensitive to the input data distribution.

Even a semantics-preserving transformations on the input data distribution can cause a significantly different robustness for the adversarially trained model that is both trained and evaluated on the new distribution.

We show this by constructing semantically- identical variants for MNIST and CIFAR10 respectively, and show that standardly trained models achieve similar clean accuracies on them, but adversarially trained models achieve significantly different robustness accuracies.

This counter-intuitive phenomenon indicates that input data distribution alone can affect the adversarial robustness of trained neural networks, not necessarily the tasks themselves.

Lastly, we discuss the practical implications on evaluating adversarial robustness, and make initial attempts to understand this complex phenomenon.

We study the relationship between adversarial robustness and the input data distribution.

We focus on the adversarial training method [3] , arguably the most popular defense method so far due to its simplicity, effectiveness and scalability.

Our main contribution is the finding that adversarial robustness is highly sensitive to the input data distribution:A semantically-lossless shift on the data distribution could result in a drastically different robustness for adversarially trained models.

Note that this is different from the transferability of a fixed model that is trained on one data distribution but tested on another distribution.

Even retraining the model on the new data distribution may give us a completely different adversarial robustness on the same new distribution.

This is also in sharp contrast to the clean accuracy of standard training, which, as we show in later sections, is insensitive to such shifts.

To our best knowledge, our paper is the first work in the literature that demonstrates such sensitivity.

Such sensitivity raises the question of how to properly evaluate adversarial robustness.

In particular, the sensitivity of adversarial robustness suggests that certain datasets may not be sufficiently representative when benchmarking different robust learning algorithms.

It also raises serious concerns about the deployment of believed-to-be-robust training algorithm in a real product.

In a standard development procedure, various models would be prototyped and measured on the existing data.

However, the sensitivity of adversarial robustness makes the truthfulness of the performance estimations questionable, as one would expect future data to be slightly shifted.

We illustrate the practical implications in Section 3: the robust accuracy of PGD trained model is sensitive to gamma values of gamma-corrected CIFAR10 images.

This indicates that image datasets collected under different lighting conditions may have different robustness properties.

Finally, our finding opens up a new angle and provides novel insights to the adversarial vulnerability problem, complementing several recent works on the issue of data distributions' influences on robustness.

[6] hypothesizes that there is an intrinsic tradeoff between clean accuracy and adversarial robustness.

Our studies complement this result, showing that there are different levels of tradeoffs depending on the characteristics of input data distribution, under the same learning settings (training algorithm, model and training set size).

[4] shows that different data distributions could have drastically different properties of adversarially robust generalization, theoretically on Bernoulli vs mixtures of Gaussians, and empirically on standard benchmark datasets.

From the sensitivity perspective, we demonstrate that being from completely different distributions (e.g. binary vs Gaussian or MNIST vs CIFAR10) may not be the essential reason for having large robust-ness difference.

Gradual semantics-preserving transformations of data distribution can also cause large changes to datasets' achievable robustness.

In this section we carefully design a series of datasets and experiments to further study its influence.

One important property of our new datasets is that they have different input data distributions P(x)'s while keeping the true classification P(y|x) reasonably fixed, thus these datasets are only different in a "semantic-lossless" shift.

Our experiments reveal an unexpected phenomenon that while standard learning methods manage to achieve stable clean accuracies across different data distributions under "semantic-lossless" shifts, however, adversarial training, arguably the most popular method to achieve robust models, loses this desirable property, in that its robust accuracy becomes unstable even under a "semantic-lossless" shift on the data distribution.

We emphasize that different from preprocessing steps or transfer learning, here we treat the shifted data distribution as a new underlying distribution.

We both train the models and test the robust accuracies on the same new distribution.

In general, MNIST has a more binary distribution of pixels, while CIFAR10 has a more continuous spectrum of pixel values.

We apply different levels of "smoothing" on MNIST to create more CIFAR-like datasets, and different levels of "saturation" on CIFAR10 to create more "binary" ones, as shown in FIG0 and 1b.

Note that we would like to maintain the semantic information of the original data, which means that such operations should be semantics-lossless.

Smoothing is applied on MNIST images, to make images "less binary".

Given an image x i , its smoothed versionx i (s) is generated by first applying average filter of kernel size s to x i to generate an intermediate smooth image, and then take pixel-wise maximum between x i and the intermediate smooth image.

Saturation of the image x is denoted by x (p) , and the procedure is defined as: DISPLAYFORM0

We use the smoothing and saturation to manipulate the data distributions of MNIST and CIFAR10, and show empirical results on how data distributions affects robust accuracies of neural networks trained on them.

To measure the difficulty of the classification task, we perform standard neural network training and test accuracies on clean data.

To measure the difficulty to achieve robustness, we perform ??? projected gradient descent (PGD) based adversarial training [3] and test robust accuracies on adversarially perturbed data.

To understand whether low robust accuracy is due to low clean accuracy or vulnerability of model, we also report robustness w.r.t.

predictions, where the attack is used to perturb against the model's clean prediction, instead of the true label.

We use LeNet5 on all the MNIST variants, and use wide residual networks [8] with widen factor 4 and depth 28 for all the CIFAR10 variants.

Unless otherwise specified, PGD training on MNIST variants and CIFAR10 variants all follows the settings in [3] .

PGD attacks on MNIST variants run with = 0.3, step size of 0.01 and 40 iterations, and runs with = 8/255, step size of 2/255 and 10 iterations on CIFAR10 variants , same as in [3] .

Results on MNIST variants are presented in FIG0 .

The clean accuracy of standard training is very stable across different MNIST variants.

This indicates that their classification tasks have similar difficulties, if the training has no robust considerations.

When performing PGD adversarial training, clean accuracy drops only slightly.

However, both robust accuracy and robustness w.r.t.

predictions drop significantly.

This indicates that as smooth level goes up, it is significantly harder to achieve robustness.

Note that for binarized MNIST with adversarial training, the clean accuracy and the robust accuracy are almost the same.

Indicating that getting high robust accuracy on binarized MNIST does not conflict with achieving high clean accuracy.

CIFAR10 result tell a similar story, as reported in FIG0 .

For standard training, the clean accuracy maintains almost at the original level until saturation level 16, despite that it is already perceptually very saturated.

In contrast, PGD training has a different trend.

Before level 16, the robust accuracy significantly increases from 43.2% until 79.7%, while the clean test accuracy drops only in a comparatively small range, from 85.4% to 80.0%.

After level 16, PGD training has almost the same clean accuracy and robust accuracy.

However, robustness w.r.t.

predictions still keeps increasing, which again indicates the instability of the robustness.

On the other hand, if the saturation level is smaller than 2, we get worse robust accuracy after PGD training, e.g. at saturation level 1 the robust ac- curacy is 33.0%.

Simultaneously, the clean accuracy maintains almost the same.

Note that after saturation level 64 the standard training accuracies starts to drop significantly.

This is likely due to that high degree of saturation has caused "information loss".

Models trained on highly saturated CIFAR10 are quite robust and the gap between robust accuracy and robustness w.r.t.

predictions is due to lower clean accuracy.

In contrast, In MNIST variants, the robustness w.r.t.

predictions is always almost the same as robust accuracy, indicating that drops in robust accuracy is due to adversarial vulnerability.

From these results, we can conclude that robust accuracy under PGD training is much more sensitive than clean accuracy under standard training to the differences in input data distribution.

More importantly, a semantically-lossless shift on the data transformation, while not introducing any unexpected risk for the clean accuracy of standard training, can lead to large variations in robust accuracy.

Such previously unnoticed sensitivity raised serious concerns in practice, as discussed in the next section.

The natural images are acquired under different lighting conditions, with different cameras and different camera settings.

They are usually preprocessed in different ways.

All these factors could lead to mild shifts on the input distribution.

Therefore, we might get very different performance measures when performing adversarial training on images taken under different conditions.

In this section, we demonstrate this phenomenon on variants of CIFAR10 images under different gamma mappings.

These variants are then used to represent image dataset acquired under different conditions.

Gamma mapping is a simple element-wise operation that takes the original image x, and output the gamma mapped imagex (??) by performingx (??) = x ?? .

Gamma mapping is commonly used to adjust the exposure of an images.

We refer the readers to [5] on more details about gamma mappings.

FIG0 shows variants of the same image processed with different gamma values.

Lower gamma value leads to brighter images and higher gamma values gives darker images, since pixel values range from 0 to 1.

Despite the changes in brightness, the semantic information is preserved.

We perform the same experiments as in the saturated CIFAR10 variants experiment in Section 2, with results displayed in FIG0 .

Clean accuracies almost remain the same across different gamma values.

However, under PGD training, both accuracy and robust accuracy varies largely under different gamma values.

These results should raise practitioners' attention on how to interpret robustness benchmark "values".

For the same adversarial training setting, the robustness measure might change drastically between image datasets with different "exposures".

In other words, if a training algorithm achieves good robustness on one image dataset, it doesn't necessarily achieve similar robustness on another semantically-identical but slightly varied datasets.

Therefore, the actual robustness could be underestimated or overestimated significantly.

This raises the questions on whether we are evaluating image classifier robustness in a reliable way, and how we choose benchmark settings that can match the real robustness requirements in practice.

We defer this important open question to future research.

Saturation moves the pixel values towards 0 and 1, therefore pushing the data points to the corners of the unit cube input domain.

This makes the valid perturbation space to be smaller, since the space of perturbation is the intersection between the -??? ball and the input domain.

Due to high dimensionality, the volume of "perturbable region" changes drastically across different saturation levels.

For example, the average log perturbable volume 1 of original CIFAR10 images are -12354, and the average log perturbable volume of ???-saturated CIFAR10 is -15342, which means that the perturbable volume differs by a factor of 2 2990 = 2 (???12352???(???15342)) .

If the differences in perturbable volume is a key factor on the robustness' sensitivity, then by allowing the attack to go beyond the domain boundary 2 , the robust accuracies across different saturation levels should behave similarly again, or at least significantly differ from the case of box constrained attacks.

We performed PGD attack allowing the perturbation to be outside of the data domain boundary, and compare the robust accuracy to what we get for normal PGD attack within domain boundary.

We found that the expected difference is not observed, in TAB1 , which serves as evidence that differences in perturbable volume are not causing the differences in robustness on the tested MNIST and CIFAR10 variants.1 Definition of "log perturbable volume" and other detailed analysis of perturbable volume are given in Appendix C.1.2 So we have a controlled and constant perturbable volume across all cases, where the volume is that of the -??? ball

When saturation pushes data points towards data domain boundaries, the distances between data points increase too.

Therefore, the margin, the distance from data point to the decision boundary, could also increase.

We use the "inter-class distance" as an approximation.

Inter-class distance 3 characterizes the distances between each class to rest of classes in each dataset.

Intuitively, if the distances between classes are larger, then it should be easier to achieve robustness.

We also observed (in Appendix C.2.1 Figure 2 ) that inter-class distances are positively correlated with robust accuracy.

However, we also find counter examples where datasets having the same inter-class distance exhibit different robust accuracies.

Specifically, We construct scaled variants of original MNIST and binarized MNIST, such that their inter-class distances are the same as smooth-3, smooth-4, smooth-5 MNIST.

The scaling operation is defined asx (??) = ??(x ??? 0.5) + 0.5, where ?? is the scaling coefficient.

When ?? < 1.

each dimension of x is pushed towards the center with the same rate.

TAB2 shows the results.

We can see that although having the same interclass distances, the smoothed MNIST is still less robust than the their correspondents of scaled binarized MNIST and original MNIST.

This indicates the complexity of the problem, such that a simple measure like inter-class distance cannot fully characterize robustness property of datasets, at least on the variants of MNIST.

The LeNet5 (widen factor 1) is composed of 32-channel conv filter + ReLU + size 2 max pooling + 64-channel conv filter + ReLU + size 2 max pooling + fc layer with 1024 units + ReLU + fc layer with 10 output classes.

We do not preprocess MNIST images before feeding into the model.

For training LeNet5 on MNIST variants, we use the Adam optimizer with an initial learning rate of 0.0001 and train for 100000 steps with batch size 50.We use the WideResNet-28-4 as described in [8] for our experiments, where 28 is the depth and 4 is the widen factor.

We use "per image standardization" 4 to preprocess CIFAR10 images, following [3] .For training WideResNet on CIFAR10 variants, we use stochastic gradient descent with momentum 0.9 and weight decay 0.0002.

We train 80000 steps in total with batch size 128.

The learning rate is set to 0.1 at step 0, 0.01 at step 40000, and 0.001 at step 60000.We performed manual hyperparameter search for our initial experiment and do not observe improvements over the above settings.

Therefore we used these settings throughout the all the experiments in the paper unless otherwise indicated.

We listed exact numbers of experiments involved in the main body in TAB4 , 4, 5 and 6.

One natural hypothesis about the reason of achieving better robustness could be that it is the effect of the boundaries.

Indeed, if the data distribution is closer to the data domain boundary, the valid perturbation space, the -??? ball may be restricted since it will intersect with the boundary.

We then test the correlation between "how close the data distribution is to the boundary" and its achievable robustness, by examining the volume of the allowed perturbed box across different datasets.

The intersection of the data domain, unit cube [0, 1] d , with the allowed perturbation space, -??? ball DISPLAYFORM0 d , where i = 1, ?? ?? ?? , d are the indexes over input dimensions.

The size of the available perturbation space at x and is defined by the volume of this hyperrectangle: DISPLAYFORM1 In high dimensional space, when is fixed, this volume varies greatly based on the location of x. For example, if x is on one of the corners of the unit cube, Vol(x corner , ) = d .

If each dimension of x is at least away from all the data boundaries, then the volume of the hyperrectangle is Vol(x inside , ) = (2 ) d .

Therefore there can be 2 d times difference of perturbable space between different data points.

As shown in the average log perturbable volumes TAB9 , we can see that different variations of datasets has significantly different perturbable volumes, with the same trend with previously described.

It is notable that for the original CI-FAR10 datasets has log volume -12354, which is very close to the -12270.

The different of 84 bits indicates on average, the perturbation space is 2 84 smaller than the full -??? ball if there is no intersection with the data domain boundary.

Volume differences between different saturation or smooth level can be interpreted in the similar way.

Note that for CIFAR10 images with large saturation, although they appear similar to human, they actually have very large differences in terms of perturbable volumes.

If the perturbable volume hypothesis holds, then we should observe significantly lower accuracy under PGD TAB1 .

"With considering boundary" corresponds to regular PGD attacks.

We can see that allowing PGD to perturb out of bound do not reduce accuracy under attack.

This means that PGD is not able to use the significantly larger additional volumes even for binarized MNIST or highly saturated CIFAR10, whose data points are on or very close to the corner.

In some cases, allowing perturbation outside of domain boundary makes the attack slightly less effective.

This might be due to that data domain boundary constrained the perturbation to be in an "easier" region.

This might seem surprising considering the huge difference in perturbable volumes, these results conform with empirical results in previous research [2, 7] that adversarial examples appears in certain directions instead of being distributed in small pockets across space.

Therefore, the perturbable volume hypothesis is rejected.

We calculate the inter-class distance as follows.

Let D = {x i } denote the set of all the input data points, D c = {x i |y i = c} denote the set of all the data points in class c, and D ??c = {x i |y i = c} denote all the data points not in class c. DISPLAYFORM0 Note that we choose 2 distance for inter-class distance, instead of using the ??? which measures the robustness.

This is because ??? -distance between data examples is essentially the max over the per pixel differences, which is always very close to 1.

Therefore the ??? -distance between data examples is not really representative / distinguishable.

Figure 2 shows the inter-class distances (averaged over all classes) calculated on MNIST and CIFAR10 variants.

The binarized MNIST has a significantly larger inter-class distance.

As smoothing kernel size increases, the distance also decrease slightly.

On CI-FAR10 variants, as the saturation level gets higher, the inter-class distance increases monotonically.

We also directly plot inter-class distance vs robust accuracy on MNIST and CIFAR10 variants.

In general, inter-class distance shows a strong positive correlation with robust accuracy under these transformations.

With one exception that original MNIST has smaller inter-class distance, but is sightly more robust than smooth-2 MNIST.

This, together with the counter examples we gave in TAB2 , suggests that inter-class distance cannot fully explain the robust variation across different dataset variants.

<|TLDR|>

@highlight

Robustness performance of PGD trained models are sensitive to semantics-preserving transformation of image datasets, which implies the trickiness of evaluation of robust learning algorithms in practice.
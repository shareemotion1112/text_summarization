Deploying machine learning systems in the real world requires both high accuracy on clean data and robustness to naturally occurring corruptions.

While architectural advances have led to improved accuracy, building robust models remains challenging, involving major changes in training procedure and datasets.

Prior work has argued that there is an inherent trade-off between robustness and accuracy, as exemplified by standard data augmentation techniques such as Cutout, which improves clean accuracy but not robustness, and additive Gaussian noise, which improves robustness but hurts accuracy.

We introduce Patch Gaussian, a simple augmentation scheme that adds noise to randomly selected patches in an input image.

Models trained with Patch Gaussian achieve state of the art on the CIFAR-10 and ImageNet Common Corruptions benchmarks while also maintaining accuracy on clean data.

We find that this augmentation leads to reduced sensitivity to high frequency noise (similar to Gaussian) while retaining the ability to take advantage of relevant high frequency information in the image (similar to Cutout).

We show it can be used in conjunction with other regularization methods and data augmentation policies such as AutoAugment.

Finally, we find that the idea of restricting perturbations to patches can also be useful in the context of adversarial learning, yielding models without the loss in accuracy that is found with unconstrained adversarial training.

Patch Gaussian augmentation overcomes the accuracy/robustness tradeoff observed in other augmentation strategies.

Larger σ of Patch Gaussian (→) improves mean corruption error (mCE) and maintains clean accuracy, whereas larger σ of Gaussian (→) and patch size of Cutout (→) hurt accuracy or robustness.

More robust and accurate models are down and to the right.

Modern deep neural networks can achieve impressive performance at classifying images in curated datasets (Karpathy, 2011; Krizhevsky et al., 2012; Tan & Le, 2019 ).

Yet, they lack robustness to various forms of distribution shift that typically occur in real-world settings.

For example, neural networks are sensitive to small translations and changes in scale (Azulay & Weiss, 2018) , blurring and additive noise (Dodge & Karam, 2017) , small objects placed in images (Rosenfeld et al., 2018) , and even different images from a distribution similar to the training set (Recht et al., 2019; .

For models to be useful in the real world, they need to be both accurate on a high-quality held-out set of images, which we refer to as "clean accuracy," and robust on corrupted images, which we refer to as "robustness."

Most of the literature in machine learning has focused on architectural changes (Simonyan & Zisserman, 2015; Szegedy et al., 2015; He et al., 2016; Szegedy et al., 2017; Han et al., 2017; Hu et al., 2017; Liu et al., 2018) to improve clean accuracy but interest has recently shifted toward robustness as well.

Research in neural network robustness has tried to quantify the problem by establishing benchmarks that directly measure it (Hendrycks & Dietterich, 2018; Gu et al., 2019) and comparing the performance of humans and neural networks (Geirhos et al., 2018b; Elsayed et al., 2018) .

Others have tried to understand robustness by highlighting systemic failure modes of current methods.

For instance, networks exhibit excessive invariance to visual features (Jacobsen et al., 2018) , texture bias (Geirhos et al., 2018a) , sensitivity to worst-case (adversarial) perturbations (Goodfellow et al., 2014) , and a propensity to rely on non-robust, but highly predictive features for classification (Doersch et al., 2015; Ilyas et al., 2019) .

Of particular relevance, Ford et al. (2019) has established connections between popular notions of adversarial robustness and some measures of distribution shift considered here.

Another line of work has attempted to increase model robustness performance, either by projecting out superficial statistics (Wang et al., 2019) , via architectural improvements (Cubuk et al., 2017) , pretraining schemes (Hendrycks et al., 2019) , or with the use of data augmentations.

Data augmentation increases the size and diversity of the training set, and provides a simple way to learn invariances that are challenging to encode architecturally (Cubuk et al., 2017) .

Recent work in this area includes learning better transformations (DeVries & Taylor, 2017; Zhang et al., 2017; Zhong et al., 2017) , inferring combinations of them (Cubuk et al., 2018) , unsupervised methods (Xie et al., 2019) , theory of data augmentation (Dao et al., 2018) , and applications for one-shot learning (Asano et al., 2019) .

Despite these advances, individual data augmentation methods that improve robustness do so at the expense of reduced clean accuracy.

Further, achieving robustness on par with the human visual system is thought to require major changes in training procedures and datasets: the current state of the art in robustness benchmarks involves creating a custom dataset with styled-transferred images before training (Geirhos et al., 2018a) , and still incurs a significant drop in clean accuracy.

The ubiquity of reported robustness/accuracy trade-offs in the literature have even led to the hypothesis that these trade-offs may be inevitable (Tsipras et al., 2018) .

Because of this, many recent works focus on improving either one or the other (Madry et al., 2017; Geirhos et al., 2018a) .

In this work we propose a simple data augmentation method that overcomes this trade-off, achieving improved robustness while maintaining clean accuracy.

Our contributions are as follows:

• We characterize a trade-off between robustness and accuracy in standard data augmentations Cutout and Gaussian (Section 2.1).

• We describe a simple data augmentation method (which we term Patch Gaussian) that allows us to interpolate between the two augmentations above (Section 3.1).

Despite its simplicity, Patch Gaussian achieves a new state of the art in the Common Corruptions robustness benchmark (Hendrycks & Dietterich, 2018) , while maintaining clean accuracy, indicating current methods have not reached this fundamental trade-off (Section 4.1).

• We demonstrate that Patch Gaussian can be combined with other regularization strategies (Section 4.2) and data augmentation policies (Section 4.3).

• We perform a frequency-based analysis (Yin et al., 2019) of models trained with Patch Gaussian and find that they can better leverage high-frequency information in lower layers, while not being too sensitive to them at later ones (Section 5.1).

• We show a similar method can be used in adversarial training, suggesting under-explored questions about training distributions' effect on out-of-distribution robustness (Section 5.2).

We start by considering two data augmentations: Cutout (DeVries & Taylor, 2017) and Gaussian (Grandvalet & Canu, 1997) .

The former sets a random patch of the input image to a constant (mean pixel in the dataset) in order to improve clean accuracy.

The latter adds independent Gaussian noise to each pixel of the input image, which directly increases robustness to Gaussian noise.

We compare the effectiveness of Gaussian and Cutout data augmentation for accuracy and robustness by measuring the performance of models trained with each on clean as well as corrupted data.

Here, robustness is defined as average accuracy of the model, when tested on data corrupted by various σ (0.1, 0.2, 0.3, 0.5, 0.8, 1.0) of Gaussian noise, relative to the clean accuracy:

Relative Gaussian Robustness = E σ (Accuracy on Data Corrupted by σ) − Clean Accuracy Fig. 2 highlights an apparent trade-off in using these methods.

In accordance to previous work (DeVries & Taylor, 2017) , Cutout improves accuracy on clean test data.

Despite this, we find it does not lead to increased robustness.

Conversely, training with higher σ of Gaussian can lead to increased robustness to Gaussian noise, but also leads to decreased accuracy on clean data.

Therefore, any robustness gains are offset by poor overall performance: a model with a perfect Relative Robustness of 0, but whose clean accuracy dropped to 50% will be wrong half the time, even on clean data.

The y-axis is the change in accuracy when tested on data corrupted with Gaussian noise at various σ (average corrupted accuracy minus clean accuracy).

The diamond indicates augmentation hyper-parameters selected by the method in Section 3.2.

At first glance, these results seem to reinforce the findings of previous work (Tsipras et al., 2018) , indicating that robustness comes at the cost of generalization, which would offset any benefits of improved robustness.

In the following sections, we will explore whether there exists augmentation strategies that do not exhibit this limitation.

Each of the two methods seen so far achieves one half of our stated goal: either improving robustness or slightly improving/maintaining clean test accuracy, but never both.

To explore whether this observed trade-off is fundamental, we introduce Patch Gaussian, a technique that combines the noise robustness of Gaussian with the slightly improved clean accuracy of Cutout.

Our method is intentionally simple but, as we'll see, it's powerful enough to overcome the limitations described and beats complex training schemes designed to provide robustness.

Patch Gaussian works by adding a W × W Figure 3 : Patch Gaussian is the addition of Gaussian noise to pixels in a square patch.

It allows us to interpolate between Gaussian and Cutout, approaching Gaussian with increasing patch size and Cutout with increasing σ.

patch of Gaussian noise to the image ( Figure 3) .

As with Cutout, the center of the patch is sampled to be within the image.

By varying the size of this patch and the maximum standard deviation of noise sampled σ max , we can interpolate between Gaussian (which applies additive Gaussian noise to the whole image) and an approximation of Cutout (which removes all information inside the patch).

See Fig. 9 for more examples.

Our goal is to learn models that achieve both good clean accuracy and improved robustness to corruptions.

Prior work has optimized for one or the other but, as noted before, to meaningfully improve robustness to other distributions, a method can't incur a significant drop in clean accuracy.

Therefore, when selecting hyper-parameters, we focus on identifying the models that are most robust while still achieving a minimum accuracy (Z) on the clean test data.

Values of Z are selected to incur negligible decrease in clean accuracy.

As such, they vary per dataset and model, and can be found in the Appendix (Table 5 ).

If no model has clean accuracy ≥ Z, we report the model with highest clean accuracy, unless otherwise specified.

We find that patch sizes around 25 on CIFAR (≤250 on ImageNet, i.e.: uniformly sampled with maximum value 250) with σ ≤ 1.0 generally perform the best.

A complete list of selected hyperparameters for all augmentations can be found in Table 5 .

We are interested in out-of-distribution robustness, and report performance of selected models on Common Corruption (Hendrycks & Dietterich, 2018) .

However, when selecting hyper-parameters, we use Relative Gaussian Robustness as a stand-in for "robustness. " Ford et al. (2019) indicates that this metric is correlated with performance on Common Corruptions, so selected models should be generally robust beyond Gaussian corruptions.

By picking models based on robustness to Gaussian noise, we ensure that our selection process does not overfit to the Common Corruptions benchmark.

Models trained with Patch Gaussian overcome the observed trade-off and gain robustness to Gaussian noise while maintaining clean accuracy (Fig. 1) .

Because Gaussian robustness is only used for hyper-parameter selection, we omit these results, but refer the curious reader to Appendix Fig. 7 .

Instead, we report how this Gaussian robustness translates into better Common Corruption robustness, which is in line with reports of the correlation between the two (Ford et al., 2019) .

In doing so, we establish a new state of the art in the Common Corruptions benchmark (Section 4.1), despite the simplicity of our method when compared with the previous best (Geirhos et al., 2018a) .

We then show that Patch Gaussian can be used in complement to other common regularization strategies (Section 4.2) and data augmentation policies (Cubuk et al., 2018 ) (Section 4.3).

In this section, we look at how our augmentations impact robustness to corruptions beyond Gaussian noise.

Rather than focusing on adversarial examples that are worst-case bounded perturbations, we focus on a more general set of corruptions (Gilmer et al., 2018 ) that models are likely to encounter in real-world settings: the Common Corruptions benchmark (Hendrycks & Dietterich, 2018) .

This benchmark, also referred to as CIFAR-C and ImageNet-C, is composed of images transformed with 15 corruptions, at 5 severities each.

Each is designed to model transformations commonly found in real-world settings, such as brightness, different weather conditions, and different kinds of noise.

Table 1 shows that Patch Gaussian achieves state of the art on both of these benchmarks in terms of mean Corruption Error (mCE).

A "Corruption Error" is a model's average error over 5 severities of a given corruption, normalized by the same average of a baseline model.

However, ImageNet-C was released in compressed JPEG format (ECMA International, 2009 ), which alters the corruptions applied to the raw pixels.

Therefore, we report results on the benchmark as-released ("Original mCE") 1 as well as a version of 12 corruptions without the extra compression ("mCE").

Additionally, because Patch Gaussian is a noise-based augmentation, we wanted to verify whether its gains on this benchmark were solely due to improved performance on noise-based corruptions (Gaussian Noise, Shot Noise, and Impulse Noise).

To do this, we also measure the models' average performance on all other corruptions, reported as "Original mCE (-noise)", and "mCE (-noise)".

The models used to normalize Corruption Errors are the "Baselines" trained with only flip and crop data augmentation.

The one exception is Original mCE ImageNet, where we use the AlexNet baseline to be directly comparable with previous work (Hendrycks & Dietterich, 2018; Geirhos et al., 2018a) .

On CIFAR, we compare with an adversarially-trained model (Madry et al., 2017) .

On ImageNet, we compare with a model trained with Random Erasing (Zhong et al., 2017) , as well as a shape-biased model "SIN+IN ftIN" (Geirhos et al., 2018a) .

Finally, previous work (Yin et al., 2019) has found that augmentation diversity is a key component of robustness gains.

To confirm that Patch Gaussian's gains aren't simply a result of using multiple augmentations, we also report results on training with Cutout and Gaussian applied in sequence ("Cutout & Gaussian" and "Gaussian & Cutout"), as well as to 50% of batches ("Gaussian or Cutout").

We observe that Patch Gaussian outperforms all models, even on corruptions like fog where Gaussian hurts performance (Ford et al., 2019) .

Scores for each corruption can be found in the Table 1 : Patch Gaussian achieves state of the art in the CIFAR-C (left) and ImageNet-C (right) robustness benchmarks while maintaining clean test accuracy.

"

Original mCE" refers to the jpegcompressed benchmark, as used in Geirhos et al. (2018a) ; Hendrycks & Dietterich (2018) .

"mCE" is a version of it without the extra jpeg compression.

Note that Patch Gaussian improves robustness even in corruptions that aren't noise-based.

*

Cutout 16 is presented for direct comparison with DeVries & Taylor (2017); Gastaldi (2017) .

For Resnet-200, we also present Gaussian at a higher σ to highlight the accuracy-robustness trade-off.

Augmentation hyper-parameters were selected based on the method in Section 3.2 and can be found in Appendix.

See text for details.

These results are surprising: achieving robustness on par with the human visual system is thought to require major changes in training procedures and datasets.

Training shape-biased models (Geirhos et al., 2018a) involves creating a custom dataset of style-transferred images, which is a computationallyexpensive operation.

Even with these, the most robust model reported SIN+IN displays a significant drop in clean accuracy.

Because of this, our main comparison is with SIN+IN ftIN, which is fine-tuned on ImageNet.

A comparison with SIN+IN can be found in Appendix Table 8 .

In sum, despite its simplicity, Patch Gaussian achieves a substantial decrease in mCE relative to other models, indicating that current methods have not reached the theoretical trade-off (Tsipras et al., 2018) , and that complex training schemes (Geirhos et al., 2018a) are not needed for robustness.

Since Patch Gaussian has a regularization effect on the models trained above, we compare it with other regularization methods: larger weight decay, label smoothing, and dropblock (Table 2) .

We find that while label smoothing improves clean accuracy, it weakens the robustness in all corruption metrics we have considered.

This agrees with the theoretical prediction from Cubuk et al. (2017), which argued that increasing the confidence of models would improve robustness, whereas label smoothing reduces the confidence of predictions.

We find that increasing the weight decay from the default value used in all models does not improve clean accuracy or robustness.

Here, we focus on analyzing the interaction of different regularization methods with Patch Gaussian.

Previous work indicates that improvements on the clean accuracy appear after training with Dropblock for 270 epochs (Ghiasi et al., 2018 ), but we did not find that training for 270 epochs changed our analysis.

Thus, we present models trained at 90 epochs for direct comparison with other results.

Due to the shorter training time, Dropblock does not improve clean accuracy, yet it does make the model more robust (relative to baseline) according to all corruption metrics we consider.

We find that using label smoothing in addition to Patch Gaussian has a mixed effect, it improves clean accuracy while slightly improving robustness metrics except for the Original mCE.

Combining Dropblock with Patch Gaussian reduces the clean accuracy relative to the Patch Gaussian-only model, as Dropblock seems to be a strong regularizer when used for 90 epochs.

However, using Dropblock and Patch Gaussian together leads to the best robustness performance.

These results indicate that Patch Gaussian can be used in conjunction with existing regularization strategies.

Knowing that Patch Gaussian can be combined with other regularizers, it's natural to ask whether it can also be combined with other data augmentation policies.

Previous work has found that varied augmentation policies have a large positive impact on model robustness (Yin et al., 2019) .

In this section, we verify that Patch Gaussian can be added to these policies for further gains.

Because AutoAugment leads to state of the art accuracies, we are interested in seeing how far it can be combined with Patch Gaussian to improve results.

Therefore, and unlike previous experiments, models are trained for 180 epochs to yield best results possible.

In an attempt to understand Patch Gaussian's performance, we perform a frequency-based analysis of models trained with various augmentations using the method introduced in Yin et al. (2019) .

First, we perturb each image in the dataset with noise sampled at each orientation and frequency in Fourier space.

Then, we measure changes in the network activations and test error when evaluated with these Fourier-noise-corrupted images: we measure the change in 2 norm of the tensor directly after the first convolution, as well as the absolute test error.

This procedure yields a heatmap, which indicates model sensitivity to different frequency and orientation perturbations in the Fourier domain.

Each image in Fig 4 shows first layer (or test error) sensitivity as a function of frequency and orientation of the sampled noise, with the middle of the image containing the lowest frequencies, and the edges of the image containing highest frequencies.

For CIFAR-10 models, we present this analysis for the entire Fourier domain, with noise sampled with norm 4.

For ImageNet, we focus our analysis on lower frequencies that are more visually salient add noise with norm 15.7.

Note that for Cutout and Gaussian, we chose larger patch sizes and σs than those selected with the method in Section 3.2 in order to highlight the effect of these augmentations on sensitivity.

Heatmaps of other models can be found in the Appendix (Figure 11 ).

We confirm findings by Yin et al. (2019) that Gaussian encourages the model to learn a low-pass filter of the inputs.

Models trained with this augmentation, then, have low test error sensitivity at high frequencies, which could help robustness.

However, valuable high-frequency information (Brendel & Bethge, 2019 ) is being thrown out at low layers, which could explain the lower test accuracy.

We further find that Cutout encourages the use of high-frequency information, which could help explain its improved generalization performance.

Yet, it does not encourage lower test error sensitivity, which explains why it doesn't improve robustness either.

Patch Gaussian, on the other hand, seems to allow high-frequency information through at lower layers, but still encourages relatively lower test error sensitivity at high frequencies.

Indeed, when we measure accuracy on images filtered with a high-pass filter, we see that Patch Gaussian models can maintain accuracy in a similar way to the baseline and to Cutout, where Gaussian fails to.

See Figure 4 for full results.

Cutout encourages the use of high frequencies in earlier layers, but its test error remains too sensitive to them.

Gaussian learns low-pass filtering of features, which increases robustness at later layers, but makes lower layers too invariant to highfrequency information (thus hurting accuracy).

Patch Gaussian allows high frequencies to be used in lower layers, and its test error remains relatively robust to them.

This can also be seen by the presence of high-frequency kernels in the first layer filters of the models (or lack thereof, in the case of Gaussian). (right) Indeed, Patch Gaussian models match the performance of Cutout and Baseline when presented with only the high frequency information of images, whereas Gaussian fails to effectively utilize this information (see Appendix Fig. 12 for experiment details).

This pattern of reduced sensitivity of predictions to high frequencies in the input occurs across all augmentation magnitudes, but here we use larger patch sizes and σ of noise to highlight the differences in models indicated by *.

See text for details.

Understanding the impact of data distributions and noise on representations has been well-studied in neuroscience (Barlow et al., 1961; Simoncelli & Olshausen, 2001; Karklin & Simoncelli, 2011) .

The data augmentations that we propose here alter the distribution of inputs that the network sees, and thus are expected to alter the kinds of representations that are learned.

Prior work on efficient coding (Karklin & Simoncelli, 2011) and autoencoders (Poole et al., 2014) has shown how filter properties change with noise in the unsupervised setting, resulting in lower-frequency filters with Gaussian, as we observe in Fig. 4 .

Consistent with prior work on natural image statistics (Torralba & Oliva, 2003) , we find that networks are least sensitive to low frequency noise where spectral density is largest.

Performance drops at higher frequencies when the amount of noise we add grows relative to typical spectral density observed at these frequencies.

In future work, we hope to better understand the relationship between naturally occurring properties of images and sensitivity, and investigate whether training with more naturalistic noise can yield similar gains in corruption robustness.

Our results indicate that patching a transformation can prevent overfitting to that particular transformation and maintain clean accuracy.

To further confirm this, we train a model with adversarial training applied only to a patch of the training input.

Adversarial training is a method of achieving robustness to worst-case perturbations.

Models trained in this setting notoriously exhibit decreased clean accuracy, so it is a good candidate to verify whether our robustness gains come from patching.

We train our models with PGD, in a setting equivalent to Madry et al. (2017) .

For Patch PGD, the adversarial perturbation is calculated on the whole image for all steps, and patched after the fact.

We also tried calculating PGD on a patch only and found similar results.

We select hyper-parameters based on PGD performance on validation set, while maintaining accuracy above 90%.

However, in this section we are not interested in improving adversarial robustness performance, but on seeing its effect on robustness to Common Corruptions, to evaluate out-of-distribution (o.o.d.) robustness.

We leave an analysis of the effect of patching on adversarial robustness to future work.

Indeed, Table 4 shows that training with Patch PGD obtains similar PGD accuracy to training with PGD, but maintains most of the clean accuracy of the baseline model.

Surprisingly, Patch PGD also improves robustness to unseen Common Corruptions, when compared to the baseline without adversarial training, indicating that patching is a generally powerful tool.

This also suggests there are unexplored questions regarding the training distribution and how that translates into i.i.d and o.o.d generalization.

We hope to explore these in future work.

In this work, we introduced a simple data augmentation operation, Patch Gaussian, which improves robustness to common corruptions without incurring a drop in clean accuracy.

For models that are large relative to the dataset size (like ResNet-200 on ImageNet and all models on CIFAR-10)

, Patch Gaussian improves clean accuracy and robustness concurrently.

We showed that Patch Gaussian achieves this by interpolating between two standard data augmentation operations Cutout and Gaussian.

Finally, we analyzed the sensitivity to noise in different frequencies of models trained with Cutout and Gaussian, and showed that Patch Gaussian combines their strengths without inheriting their weaknesses.

Our method is much simpler than previous state of the art, and can be used in conjunction with other regularization and data augmentation strategies, indicating it is generally useful.

We end by showing that applying perturbations in patches can be a powerful method to vary training distributions in the adversarial setting.

Our results indicate current methods have not reached a fundamental robustness/accuracy trade-off, and that future work is needed to understand the effect of training distributions in o.o.d.

robustness.

Fig .

5 shows the accuracy/robustness trade-off of models trained with various hyper-parameters of Cutout and Gaussian.

Fig. 6 shows clean accuracy change of models trained with various hyper-parameters of Patch Gaussian.

Fig. 7 shows how Patch Gaussian can overcome the observed trade-off and gain Gaussian robustness in various models and datasets.

We run our experiments on CIFAR-10 ( Krizhevsky & Hinton, 2009 ) and ImageNet (Deng et al., 2009 ) datasets.

On CIFAR-10, we use the Wide-ResNet-28-10 model (Zagoruyko & Komodakis, 2016) , as well as the Shake-shake-112 model (Gastaldi, 2017) , trained for 200 epochs and 600 epochs respectively.

The Wide-ResNet model uses a initial learning rate of 0.1 with a cosine decay schedule.

Weight decay is set to be 5 × 10 −4 and batch size is 128.

We train all models, including the baseline, with standard data augmentation of horizontal flips and pad-and-crop.

Our code uses the same hyper parameters as Cubuk et al. (2018) On ImageNet, we use the ResNet-50 and Resnet-200 models (He et al., 2016) , trained for 90 epochs.

We use a weight decay rate of 1 × 10 −4 , global batch size of 512 and learning rate of 0.2.

The learning rate is decayed by 10 at epochs 30, 60, and 80.

We use standard data augmentation of horizontal flips and crops.

All CIFAR-10 and ImageNet experiments use the listed hyper-parameters above, unless specified otherwise.

To apply Gaussian, we uniformly sample a standard deviation σ from 0 up to some maximum value σ max , and add i.i.d.

noise sampled from N (0, σ 2 ) to each pixel.

To apply Cutout, we use a fixed patch size W , and randomly set a square region with size W × W to the constant mean of each RGB channel in the dataset.

As in DeVries & Taylor (2017) , the patch location is randomly sampled and can lie outside of the 32 × 32 CIFAR-10 (or 224 × 224 ImageNet) image but its center is constrained to lie within it.

Patch sizes and σ max are selected based on the method in Section 3.2.

Table 5 : Augmentation hyper-parameters selected with the method in Section 3.2 for each model/dataset.

*Indicates manually-chosen stronger hyper-parameters, used to highlight the effect of the augmentation on the models.

"≤" indicates that the value is uniformly sampled up to the given maximum value.

Since Patch Gaussian can be combined with both regularization strategies as well as data augmentation policies, we want to see if it is generally useful beyond classification tasks.

We train a RetinaNet detector (Lin et al., 2017) with ResNet-50 backbone (He et al., 2016) on the COCO dataset (Lin et al., 2014) .

Images for both baseline and Patch Gaussian models are horizontally flipped half of the time, after being resized to 640 × 640.

We train both models for 150 epochs using a learning rate of 0.08 and a weight decay of 1 × 10 −4 .

The focal loss parameters are set to be α = 0.25 and γ = 1.5.

Despite being designed for classification, Patch Gaussian improves detection performance according to all metrics when tested on the clean COCO validation set (Table 9 ).

On the primary COCO metric mean average precision (mAP), the model trained with Patch Gaussian achieves a 1% higher accuracy over the baseline, whereas the model trained with Gaussian suffers a 2.9% loss.

Next, we evaluate these models on the validation set corrupted by i.i.d.

Gaussian noise, with σ = 0.25.

We find that model trained with Gaussian and Patch Gaussian achieve the highest mAP of 26.1% on the corrupted data, whereas the baseline achieves 11.6%.

It is interesting to note that Patch Gaussian model achieves a better result on the harder metrics of small object detection and stricter intersection over union (IoU) thresholds, whereas the Gaussian model achieves a better result on the easier tasks of large object detection and less strict IOU threshold metric.

Overall, as was observed for the classification tasks, training object detection models with Patch Gaussian leads to significantly more robust models without sacrificing clean accuracy.

FOURIER ANALYSIS Fig. 10 shows a fourier analysis of selected models reported.

Fig. 11 shows complete filters for ResNet-50 models.

Fig. 12 shows high-pass filters used in high-pass experiment in Fig. 4 .

Figure 4 for details.

We again note the presence of filters of high fourier frequency in models trained with Cutout* and Patch Gaussian.

We also note that Gaussian* exhibits high variance filters.

We posit these have not been trained and have little importance, given the low sensitivity of this model to high frequencies.

Future work will investigate the importance of filters on sensitivity.

Ford et al. (2019) reports that PGD training helps with corruption robustness.

However, they fail to report mCE values for their models.

We find that, indeed, PGD helps with some corruptions, and when all corruption severities' errors are averaged, it mostly maintains performance (23.8% error, compared to baseline error of 23.51%).

However, as table 4 shows, when we properly calculate mCE by normalizing with a baseline model, PGD displays much worse robustness, while Patch PGD improves performance.

Figure 13 shows the frequency-based analysis (Yin et al., 2019) for models with different hyperparameters of Patch Gaussian.

First, for hyper-parameters W =16, σ=1.0 (center), the reader will note that these are very similar to the frequency sensitivity reported in Figure 4 .

The main difference being that the smaller patch size (16 vs 25 in Figure 4 ) makes the model slightly more sensitive to high frequencies.

This makes sense since smaller patch size moves the model further away from a Gaussian-trained one.

When we make the scale smaller (W =16, σ=0.3, left), less information is corrupted in the patch, which moves the model farther from the one trained with Cutout (and therefore closer to a Gaussiantrained one).

This can be seen in the increased invariance to high frequencies at the first layer, which is reflected in invariance at test error as well.

If we, instead, make the scale larger(W =16, σ=2.0, right), we move the model closer to the one trained with Cutout.

Notice the higher intensity red in the first layer plot, indicating higher sensitivity to high-frequency features.

We also see this sensitivity reflected in the test error, which matches the behavior for Cutout-trained models.

Figure 13 : Frequency-based analysis (Yin et al., 2019) for models with different hyper-parameters of Patch Gaussian.

@highlight

Simple augmentation method overcomes robustness/accuracy trade-off observed in literature and opens questions about the effect of training distribution on out-of-distribution generalization.
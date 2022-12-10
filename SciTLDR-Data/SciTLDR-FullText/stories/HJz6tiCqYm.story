In this paper we establish rigorous benchmarks for image classifier robustness.

Our first benchmark, ImageNet-C, standardizes and expands the corruption robustness topic, while showing which classifiers are preferable in safety-critical applications.

Then we propose a new dataset called ImageNet-P which enables researchers to benchmark a classifier's robustness to common perturbations.

Unlike recent robustness research, this benchmark evaluates performance on common corruptions and perturbations not worst-case adversarial perturbations.

We find that there are negligible changes in relative corruption robustness from AlexNet classifiers to ResNet classifiers.

Afterward we discover ways to enhance corruption and perturbation robustness.

We even find that a bypassed adversarial defense provides substantial common perturbation robustness.

Together our benchmarks may aid future work toward networks that robustly generalize.

The human vision system is robust in ways that existing computer vision systems are not BID50 BID1 .

Unlike current deep learning classifiers BID36 BID21 BID60 , the human vision system is not fooled by small changes in query images.

Humans are also not confused by many forms of corruption such as snow, blur, pixelation, and novel combinations of these.

Humans can even deal with abstract changes in structure and style.

Achieving these kinds of robustness is an important goal for computer vision and machine learning.

It is also essential for creating deep learning systems that can be deployed in safety-critical applications.

Most work on robustness in deep learning methods for vision has focused on the important challenges of robustness to adversarial examples BID54 BID5 , unknown unknowns BID25 BID23 BID41 , and model or data poisoning BID53 .

In contrast, we develop and validate datasets for two other forms of robustness.

Specifically, we introduce the IMAGETNET-C dataset for input corruption robustness and the IMAGENET-P dataset for input perturbation robustness.

To create IMAGENET-C, we introduce a set of 75 common visual corruptions and apply them to the ImageNet object recognition challenge BID7 .

We hope that this will serve as a general dataset for benchmarking robustness to image corruptions and prevent methodological problems such as moving goal posts and result cherry picking.

We evaluate the performance of current deep learning systems and show that there is wide room for improvement on IMAGENET-C. We also introduce a total of three methods and architectures that improve corruption robustness without losing accuracy.

To create IMAGENET-P, we introduce a set of perturbed or subtly differing ImageNet images.

Using metrics we propose, we measure the stability of the network's predictions on these perturbed images.

Although these perturbations are not chosen by an adversary, currently existing networks exhibit surprising instability on common perturbations.

Then we then demonstrate that approaches which enhance corruption robustness can also improve perturbation robustness.

For example, some recent architectures can greatly improve both types of robustness.

More, we show that the Adversarial Logit Pairing ∞ adversarial example defense can yield substantial robustness gains on diverse and common perturbations.

By defining and benchmarking perturbation and corruption robustness, we facilitate research that can be overcome by future networks which do not rely on spurious correlations or cues inessential to the object's class.

Adversarial Examples.

An adversarial image is a clean image perturbed by a small distortion carefully crafted to confuse a classifier.

These deceptive distortions can occasionally fool black-box classifiers BID38 .

Algorithms have been developed that search for the smallest additive distortions in RGB space that are sufficient to confuse a classifier .

Thus adversarial distortions serve as type of worst-case analysis for network robustness.

Its popularity has often led "adversarial robustness" to become interchangeable with "robustness" in the literature BID2 .

In the literature, new defenses BID42 BID47 BID44 BID22 often quickly succumb to new attacks BID12 BID5 , with some exceptions for perturbations on small images BID52 .

For some simple datasets, the existence of any classification error ensures the existence of adversarial perturbations of size O(d −1/2 ), d the input dimensionality BID18 .

For some simple models, adversarial robustness requires an increase in the training set size that is polynomial in d .

BID17 suggest modifying the problem of adversarial robustness itself for increased real-world applicability.

Robustness in Speech.

Speech recognition research emphasizes robustness to common corruptions rather than worst-case, adversarial corruptions BID39 BID45 .

Common acoustic corruptions (e.g., street noise, background chatter, wind) receive greater focus than adversarial audio, because common corruptions are ever-present and unsolved.

There are several popular datasets containing noisy test audio BID27 BID26 .

Robustness in noisy environments requires robust architectures, and some research finds convolutional networks more robust than fully connected networks BID0 .

Additional robustness has been achieved through pre-processing techniques such as standardizing the statistics of the input BID40 BID58 BID20 BID35 .

Studies.

Several studies demonstrate the fragility of convolutional networks on simple corruptions.

For example, BID28 apply impulse noise to break Google's Cloud Vision API.

Using Gaussian noise and blur, BID9 demonstrate the superior robustness of human vision to convolutional networks, even after networks are fine-tuned on Gaussian noise or blur.

BID15 compare networks to humans on noisy and elastically deformed images.

They find that fine-tuning on specific corruptions does not generalize and that classification error patterns underlying network and human predictions are not similar.

BID56 ; propose different corrupted datasets for object and traffic sign recognition.

Robustness Enhancements.

In an effort to reduce classifier fragility, BID59 finetune on blurred images.

They find it is not enough to fine-tune on one type of blur to generalize to other blurs.

Furthermore, fine-tuning on several blurs can marginally decrease performance.

BID61 also find that fine-tuning on noisy images can cause underfitting, so they encourage the noisy image softmax distribution to match the clean image softmax.

BID8 address underfitting via a mixture of corruption-specific experts assuming corruptions are known beforehand.

We now define corruption and perturbation robustness and distinguish them from adversarial perturbation robustness.

To begin, we consider a classifier f : X → Y trained on samples from distribution D, a set of corruption functions C, and a set of perturbation functions E. We let P C (c), P E (ε) approximate the real-world frequency of these corruptions and perturbations.

Most classifiers are judged by their accuracy on test queries drawn from D, i.e., P (x,y)∼D (f (x) = y).

Yet in a vast range of cases the classifier is tasked with classifying low-quality or corrupted inputs.

In view of this, we suggest also computing the classifier's corruption robustness E c∼C [P (x,y)∼D (f (c(x) = y))].

This contrasts with a popular notion of adversarial robustness, often formulated min δ p <b P (x,y)∼D (f (x + δ) = y), b a small budget.

Thus, corruption robustness measures the classifier's average-case performance on corruptions C, while adversarial robustness measures the worst-case performance on small, additive, classifier-tailored perturbations.

Average-case performance on small, general, classifier-agnostic perturbations motivates us to define perturbation robustness, namely E ε∼E [P (x,y)∼D (f (ε(x)) = f (x))].

Consequently, in measuring perturbation robustness, we track the classifier's prediction stability, reliability, or consistency in the face of minor input changes.

Now in order to approximate C, E and these robustness measures, we designed a set of corruptions and perturbations which are frequently encountered in natural images.

We will refer to these as "common" corruptions and perturbations.

These common corruptions and perturbations are available in the form of IMAGENET-C and IMAGENET-P.4 THE IMAGENET-C AND IMAGENET-P ROBUSTNESS BENCHMARKS 4.1 THE DATA OF IMAGENET-C AND IMAGENET-P IMAGENET-C Design.

The IMAGENET-C benchmark consists of 15 diverse corruption types applied to validation images of ImageNet.

The corruptions are drawn from four main categoriesnoise, blur, weather, and digital-as shown in Figure 1 .

Research that improves performance on this benchmark should indicate general robustness gains, as the corruptions are diverse and numerous.

Each corruption type has five levels of severity since corruptions can manifest themselves at varying intensities.

Appendix A gives an example of the five different severity levels for impulse noise.

Real-world corruptions also have variation even at a fixed intensity.

To simulate these, we introduce variation for each corruption when possible.

For example, each fog cloud is unique to each image.

These algorithmically generated corruptions are applied to the ImageNet BID7 ) validation images to produce our corruption robustness dataset IMAGENET-C. The dataset can be downloaded or re-created by visiting https://github.com/hendrycks/robustness.

IMAGENET-C images are saved as lightly compressed JPEGs; this implies an image corrupted by Gaussian noise is also slightly corrupted by JPEG compression.

Our benchmark tests networks with IMAGENET-C images, but networks should not be trained on these images.

Networks should be trained on datasets such as ImageNet and not be trained on IMAGENET-C corruptions.

To enable further experimentation, we designed an extra corruption type for each corruption category (Appendix B), and we provide CIFAR-10-C, TINY IMAGENET-C, IMAGENET 64 × 64-C, and Inception-sized editions.

Overall, the IMAGENET-C dataset consists of 75 corruptions, all applied to ImageNet validation images for testing a pre-existing network.

IMAGENET-P Design.

The second benchmark that we propose tests the classifier's perturbation robustness.

Models lacking in perturbation robustness produce erratic predictions which undermines user trust.

When perturbations have a high propensity to change the model's response, then perturbations could also misdirect or destabilize iterative image optimization procedures appearing in style transfer BID14 , decision explanations BID13 , feature visualization BID46 , and so on.

Like IMAGENET-C, IMAGENET-P consists of noise, blur, weather, and digital distortions.

Also as before, the dataset has validation perturbations; has difficulty levels; has CIFAR-10, Tiny ImageNet, ImageNet 64 × 64, standard, and Inception-sized editions; and has been designed for benchmarking not training networks.

IMAGENET-P departs from IMAGENET-C by having perturbation sequences generated from each ImageNet validation image; examples are in FIG0 .

Each sequence contains more than 30 frames, so we counteract an increase in dataset size and evaluation time by using only 10 common perturbations.

Common Perturbations.

Appearing more subtly than the corruption from IMAGENET-C, the Gaussian noise perturbation sequence begins with the clean ImageNet image.

The following frames in the sequence consist in the same image but with minute Gaussian noise perturbations applied.

This sequence design is similar for the shot noise perturbation sequence.

However the remaining perturbation sequences have temporality, so that each frame of the sequence is a perturbation of the previous frame.

Since each perturbation is small, repeated application of a perturbation does not bring the image far out-of-distribution.

For example, an IMAGENET-P translation perturbation sequence shows a clean ImageNet image sliding from right to left one pixel at a time; with each perturbation of the pixel locations, the resulting frame is still of high quality.

The perturbation sequences with temporality are created with motion blur, zoom blur, snow, brightness, translate, rotate, tilt (viewpoint variation through minor 3D rotations), and scale perturbations.

IMAGENET-C Metrics.

Common corruptions such as Gaussian noise can be benign or destructive depending on their severity.

In order to comprehensively evaluate a classifier's robustness to a given type of corruption, we score the classifier's performance across five corruption severity levels and aggregate these scores.

The first evaluation step is to take a trained classifier f, which has not been trained on IMAGENET-C, and compute the clean dataset top-1 error rate.

Denote this error rate E f clean .

The second step is to test the classifier on each corruption type c at each level of severity s (1 ≤ s ≤ 5).

This top-1 error is written E f s,c .

Before we aggregate the classifier's performance across severities and corruption types, we will make error rates more comparable since different corruptions pose different levels of difficulty.

For example, fog corruptions often obscure an object's class more than brightness corruptions.

We adjust for the varying difficulties by dividing by AlexNet's errors, but any baseline will do (even a baseline with 100% error rates, corresponding to an average of CEs).

This standardized aggregate performance measure is the Corruption Error, computed with the formula DISPLAYFORM0 .

This results in the mean CE or mCE for short.

We now introduce a more nuanced corruption robustness measure.

Consider a classifier that withstands most corruptions, so that the gap between the mCE and the clean data error is minuscule.

Contrast this with a classifier with a low clean error rate which has its error rate spike in the presence of corruptions; this corresponds to a large gap between the mCE and clean data error.

It is possible that the former classifier has a larger mCE than the latter, despite the former degrading more gracefully in the presence of corruptions.

The amount that the classifier declines on corrupted inputs is given by the formula Relative CE DISPLAYFORM0 .

Averaging these 15 Relative Corruption Errors results in the Relative mCE.

This measures the relative robustness or the performance degradation when encountering corruptions.

IMAGENET-P Metrics.

A straightforward approach to estimate E ε∼E [P (x,y)∼D (f (ε(x)) = f (x))] falls into place when using IMAGENET-P perturbation sequences.

Let us denote m perturbation sequences with S = x DISPLAYFORM1 where each sequence is made with perturbation p.

The "Flip Probability" of network f : X → {1, 2, . . . , 1000} on perturbation sequences S is DISPLAYFORM2 For noise perturbation sequences, which are not temporally related, DISPLAYFORM3 1 .

We can recast the FP formula for noise sequences as FP DISPLAYFORM4 .

As was done with the Corruption Error formula, we now standardize the Flip Probability by the sequence's difficulty for increased commensurability.

We have, then, the "Flip Rate" FR DISPLAYFORM5 Averaging the Flip Rate across all perturbations yields the mean Flip Rate or mFR.

We do not define a "relative mFR" since we did not find any natural formulation, nor do we directly use predicted class probabilities due to differences in model calibration BID19 .When the top-5 predictions are relevant, perturbations should not cause the list of top-5 predictions to shuffle chaotically, nor should classes sporadically vanish from the list.

We penalize top-5 inconsistency of this kind with a different measure.

Let the ranked predictions of network f on x be the permutation τ (x) ∈ S 1000 .

Concretely, if "Toucan" has the label 97 in the output space and "Pelican" has the label 145, and if f on x predicts "Toucan" and "Pelican" to be the most and second-most likely classes, respectively, then τ (x)(97) = 1 and τ (x)(144) = 2.

These permutations contain the top-5 predictions, so we use permutations to compare top-5 lists.

To do this, we define DISPLAYFORM6 .

If the top-5 predictions represented within τ (x) and τ (x ) are identical, then d(τ (x), τ (x )) = 0.

More examples of d on several permutations are in Appendix C. Comparing the top-5 predictions across entire perturbation sequences results in the unstandardized Top-5 Distance uT5D DISPLAYFORM7 For noise perturbation sequences, we have uT5D DISPLAYFORM8 Once the uT5D is standardized, we have the Top-5 Distance T5D counts of the scenery before them.

Hence, we propose the following protocol.

The image recognition network should be trained on the ImageNet training set and on whatever other training sets the investigator wishes to include.

Researchers should clearly state whether they trained on these corruptions or perturbations; however, this training strategy is discouraged (see Section 2).

We allow training with other distortions (e.g., uniform noise) and standard data augmentation (i.e., cropping, mirroring), even though cropping overlaps with translations.

Then the resulting trained model should be evaluated on IMAGENET-C or IMAGENET-P using the above metrics.

Optionally, researchers can test with the separate set of validation corruptions and perturbations we provide for IMAGENET-C and IMAGENET-P.

How robust are current methods, and has progress in computer vision been achieved at the expense of robustness?

As seen in Figure 3 , as architectures improve, so too does the mean Corruption Error (mCE).

By this measure, architectures have become progressively more successful at generalizing to corrupted distributions.

Note that models with similar clean error rates have fairly similar CEs, and in TAB3 there are no large shifts in a corruption type's CE.

Consequently, it would seem that architectures have slowly and consistently improved their representations over time.

However, it appears that corruption robustness improvements are mostly explained by accuracy improvements.

Recall that the Relative mCE tracks a classifier's accuracy decline in the presence of corruptions.

Figure 3 shows that the Relative mCEs of many subsequent models are worse than that of AlexNet BID36 .

Full results are in Appendix D. In consequence, from AlexNet to ResNet, corruption robustness in itself has barely changed.

Thus our "superhuman" classifiers are decidedly subhuman.

On perturbed inputs, current classifiers are unexpectedly bad.

For example, a ResNet-18 on Scale perturbation sequences have a 15.6% probability of flipping its top-1 prediction between adjacent frames (i.e., FP ResNet-18 Scale = 15.6%); the uT5DResNet-18 Scale is 3.6.

More results are in Appendix E. Clearly perturbations need not be adversarial to fool current classifiers.

What is also surprising is that while VGGNets are worse than ResNets at generalizing to corrupted examples, on perturbed examples they can be just as robust or even more robust.

Likewise, Batch Normalization made VGG-19 less robust to perturbations but more robust to corruptions.

Yet this is not to suggest that there is a fundamental trade-off between corruption and perturbation robustness.

In fact, both corruption and perturbation robustness can improve together, as we shall see later.

Be aware that Appendix F contains many informative failures in robustness enhancement.

Those experiments underscore the necessity in testing on a a diverse test set, the difficulty in cleansing corruptions from image, and the futility in expecting robustness gains from some "simpler" models.

Histogram Equalization.

Histogram equalization successfully standardizes speech data for robust speech recognition BID58 BID20 .

For images, we find that preprocessing with Contrast Limited Adaptive Histogram Equalization BID48 ) is quite effective.

Unlike our image denoising attempt (Appendix F), CLAHE reduces the effect of some corruptions while not worsening performance on most others, thereby improving the mCE.

We demonstrate CLAHE's net improvement by taking a pre-trained ResNet-50 and fine-tuning the whole model for five epochs on images processed with CLAHE.

The ResNet-50 has a 23.87% error rate, but ResNet-50 with CLAHE has an error rate of 23.55%.

On nearly all corruptions, CLAHE slightly decreases the Corruption Error.

The ResNet-50 without CLAHE preprocessing has an mCE of 76.7%, while with CLAHE the ResNet-50's mCE decreases to 74.5%.Multiscale Networks.

Multiscale architectures achieve greater corruption robustness by propagating features across scales at each layer rather than slowly gaining a global representation of the input as in typical convolutional neural networks.

Some multiscale architectures are called Multigrid Networks BID34 .

Multigrid networks each have a pyramid of grids in each layer which enables the subsequent layer to operate across scales.

Along similar lines, Multi-Scale Dense Networks (MSDNets) BID31 use information across scales.

MSDNets bind network layers with DenseNet-like BID30 ) skip connections.

These two different multiscale networks both enhance corruption robustness, but they do not provide any noticeable benefit in perturbation robustness.

Now before comparing mCE values, we first note the Multigrid network has a 24.6% top-1 error rate, as does the MSDNet, while the ResNet-50 has a 23.9% top-1 error rate.

On noisy inputs, Multigrid networks noticeably surpass ResNets and MSDNets, as shown in Figure 5 .

Since multiscale architectures have high-level representations processed in tandem with fine details, the architectures appear better equipped to suppress otherwise distracting pixel noise.

When all corruptions are evaluated, ResNet-50 has an mCE of 76.7%, the MSDNet has an mCE of 73.6%, and the Multigrid network has an mCE of 73.3%.Feature Aggregating and Larger Networks.

Some recent models enhance the ResNet architecture by increasing what is called feature aggregation.

Of these, DenseNets and ResNeXts BID60 are most prominent.

Each purports to have stronger representations than ResNets, and the evidence is largely a hard-won ImageNet error-rate downtick.

Interestingly, the IMAGENET-C mCE clearly indicates that DenseNets and ResNeXts have superior representations.

Accordingly, a switch from a ResNet-50 (23.9% top-1 error) to a DenseNet-121 (25.6% error) decreases the mCE from 76.7% to 73.4% (and the relative mCE from 105.0% to 92.8%).

More starkly, switching from a ResNet-50 to a ResNeXt-50 (22.9% top-1) drops the mCE from 76.7% to 68.2% (relative mCE decreases from 105.0% to 88.6%).

Corruption robustness results are summarized in Figure 5 .

This shows that corruption robustness may be a better way to measure future progress in representation learning than the clean dataset top-1 error rate.

Some of the greatest and simplest robustness gains sometimes emerge from making recent models more monolithic.

Apparently more representations, more redundancy, and more capacity allow these massive models to operate more stably on corrupted inputs.

We saw earlier that making models smaller does the opposite.

Swapping a DenseNet-121 (25.6% top-1) with the larger DenseNet-161 (22.9% top-1) decreases the mCE from 73.4% to 66.4% (and the relative mCE from 92.8% to 84.6%).

In a similar fashion, a ResNeXt-50 (22.9% top-1) is less robust than the a giant ResNeXt-101 (21.0% top-1).

The mCEs are 68.2% and 62.2% respectively (and the relative mCEs are 88.6% and 80.1% respectively).

Both model size and feature aggregation results are summarized in Figure 6 .

Consequently, future models with even more depth, width, and feature aggregation may attain further corruption robustness.

Feature aggregation and their larger counterparts similarly improve perturbation robustness.

While a ResNet-50 has a 58.0% mFR and a 78.3% mT5D, a DenseNet-121 obtains a 56.4% mFR and 76.8% mT5D, and a ResNeXt-50 does even better with a 52.4% mFR and a 74.2% mT5D.

Reflecting the corruption robustness findings further, the larger DenseNet-161 has a 46.9% mFR and 69.5% mT5D, while the ResNeXt-101 has a 43.2% mFR and 65.9% mT5D.

Thus in two senses feature aggregating networks and their larger versions markedly enhance robustness.

Stylized ImageNet.

BID16 propose a novel data augmentation scheme where ImageNet images are stylized with style transfer.

The intent is that classifiers trained on stylized images will rely less on textural cues for classification.

When a ResNet-50 is trained on typical ImageNet images and stylized ImageNet images, the resulting model has an mCE of 69.3%, down from 76.7%.Adversarial Logit Pairing.

ALP is an adversarial example defense for large-scale image classifiers BID33 .

Like nearly all other adversarial defenses, ALP was bypassed and has unclear value as an adversarial defense going forward BID11 ), yet this is not a decisive reason dismiss it.

ALP provides significant perturbation robustness even though it does not provide much adversarial perturbation robustness against all adversaries.

Although ALP was designed to increase robustness to small gradient perturbations, it markedly improves robustness to all sorts of noise, blur, weather, and digital IMAGENET-P perturbations-methods generalizing this well is a rarity.

In point of fact, a publicly available Tiny ImageNet ResNet-50 model fine-tuned with ALP has a 41% and 40% relative decrease in the mFP and mT5D on TINY IMAGENET-P, respectively.

ALP's success in enhancing common perturbation robustness and its modest utility for adversarial perturbation robustness highlights that the interplay between these problems should be better understood.

In this paper, we introduced what are to our knowledge the first comprehensive benchmarks for corruption and perturbation robustness.

This was made possible by introducing two new datasets, IMAGENET-C and IMAGENET-P. The first of which showed that many years of architectural advancements corresponded to minuscule changes in relative corruption robustness.

Therefore benchmarking and improving robustness deserves attention, especially as top-1 clean ImageNet accuracy nears its ceiling.

We also saw that classifiers exhibit unexpected instability on simple perturbations.

Thereafter we found that methods such as histogram equalization, multiscale architectures, and larger featureaggregating models improve corruption robustness.

These larger models also improve perturbation robustness.

However, we found that even greater perturbation robustness can come from an adversarial defense designed for adversarial ∞ perturbations, indicating a surprising interaction between adversarial and common perturbation robustness.

In this work, we found several methods to increase robustness, introduced novel experiments and metrics, and created new datasets for the rigorous study of model robustness, a pressing necessity as models are unleashed into safety-critical real-world settings.

Clean Severity = 1 Severity = 2 Severity = 3 Severity = 4 Severity = 5Figure 7: Impulse noise modestly to markedly corrupts a frog, showing our benchmark's varying severities.

In Figure 7 , we show the Impulse noise corruption type in five different severities.

Clearly, IMAGENET-C corruptions can range from negligible to pulverizing.

Because of this range, the benchmark comprehensively assesses each corruption type.

Speckle Noise Gaussian Blur Spatter Saturate Directly fitting the types of IMAGENET-C corruptions should be avoided, as it would cause researchers to overestimate a model's robustness.

Therefore, it is incumbent on us to simplify model validation.

This is why we provide an additional form of corruption for each of the four general types.

These are available for download at https://github.com/hendrycks/robustness.

There is one corruption type for each noise, blur, weather, and digital category in the validation set.

The first corruption type is speckle noise, an additive noise where the noise added to a pixel tends to be larger if the original pixel intensity is larger.

Gaussian blur is a low-pass filter where a blurred pixel is a result of a weighted average of its neighbors, and farther pixels have decreasing weight in this average.

Spatter can occlude a lens in the form of rain or mud.

Finally, saturate is common in edited images where images are made more or less colorful.

See FIG3 for instances of each corruption type.

For some readers, the following function may be opaque, DISPLAYFORM0 where σ = (τ (x)) −1 τ (x ) and the empty sum is understood to be zero.

A high-level view of d is that it computes the deviation between the top-5 predictions of two prediction lists.

For simplicity we find the deviation between the identity and σ rather than τ (x) and τ (x ).

In consequence we can consider d Also, d ((2, 3, 4, 5, 6 , . . .

, 1)) = 5.

Distinctly, d ((1, 2, 3, 5, 6, 4, 7, 8, . . .)) = 2.

As a final example, d ((5, 4, 3, 2, 1, 6, 7, 8, 9 , . . .)) = 12.It may be that we want perturbation robustness for all predictions, including classes with lesser relevance.

In such cases, it is still common that the displacement of the top prediction matters more than the displacement of, say, the 500th ranked class.

For this there are many possibilities, such as the measure d (σ) = 1000 i=1 w i |w i − w σ(i) | such that w i = 1/i.

This uses a Zipfian assumption about the rankings of the classes: the first class is n times as relevant as the nth class.

Other possibilities involve using logarithms rather than hyperbolic functions as in the discounted cumulative gain BID37 .

One could also use the class probabilities provided by the model (should they exist).

However such a measure could make it difficult to compare models since some models tend to be more uncalibrated than others BID19 .As progress is made on this task, researchers may be interested in perturbations which are more likely to cause unstable predictions.

To accomplish that, researchers can simply compare a frame with the frame two frames ahead rather than just one frame ahead.

We provide concrete code of this slight change in the metric at https://github.com/hendrycks/robustness.

For nontemporal perturbation sequences, i.e., noise sequences, we provide sequences where the noise perturbation is larger.

IMAGENET-C corruption relative robustness results are in BID32 .

IMAGENET-P mFR values are in TAB7 , and mT5D values are in

Stability Training.

Stability training is a technique to improve the robustness of deep networks BID61 .

The method's creators found that training on images corrupted with noise can lead to underfitting, so they instead propose minimizing the cross-entropy from the noisy image's softmax distribution to the softmax of the clean image.

The authors evaluated performance on images with subtle differences and suggested that the method provides additional robustness to JPEG corruptions.

We fine-tune a ResNet-50 with stability training for five epochs.

For training with noisy images, we corrupt images with uniform noise, where the maximum and minimum of the uniform noise is tuned over {0.01, 0.05, 0.1}, and the stability weight is tuned over {0.01, 0.05, 0.1}. Across all noise strengths and stability weight combinations, the models with stability training tested on IMAGENET-C have a larger mCEs than the baseline ResNet-50's mCE.

Even on unseen noise corruptions, stability training does not increase robustness.

However, the perturbation robustness slightly improves.

The best model according to the IMAGENET-P validation set has an mFR of 57%, while the original ResNet's mFR is 58%.

An upshot of this failure is that benchmarking robustness-enhancing techniques requires a diverse test set.

Image Denoising.

An approach orthogonal to modifying model representations is to improve the inputs using image restoration techniques.

Although general image restoration techniques are not yet mature, denoising restoration techniques are not.

We thus attempt restore an image with the denoising technique called non-local means BID3 .

The amount of denoising applied is determined by the noise estimation technique of BID10 .

Therefore clean images receive should nearly no modifications from the restoration method, while noisy images should undergo considerable restoration.

We found that denoising increased the mCE from 76.7% to 82.1%.

A plausible account is that the non-local means algorithm striped the images of their subtle details even when images lacked noise, despite having the non-local means algorithm governed by the noise estimate.

Therefore, the gains in noise robustness were wiped away by subtle blurs to images with other types of corruptions, showing that targeted image restoration can prove harmful for robustness.10-Crop Classification.

Viewing an object at several different locations may give way to a more stable prediction.

Having this intuition in mind, we perform 10-crop classification.

10-crop classification is executed by cropping all four corners and cropping the center of an image.

These crops and their horizontal mirrors are processed through a network to produce 10 predicted class probability distributions.

We average these distributions to compute the final prediction.

Of course, a prediction informed by 10-crops rather than a single central crop is more accurate.

Ideally, this revised prediction should be more robust too.

However, the gains in mCE do not outpace the gains in accuracy on a ResNet-50.

In all, 10-crop classification is a computationally expensive option which contributes to classification accuracy but not noticeably to robustness.

Smaller Models.

All else equal, "simpler" models often generalize better, and "simplicity" frequently translates to model size.

Accordingly, smaller models may be more robust.

We test this hypothesis with CondenseNets BID29 .

A CondenseNet attains its small size via sparse convolutions and pruned filter weights.

An off-the-shelf CondenseNet (C = G = 4) obtains a 26.3% error rate and a 80.8% mCE.

On the whole, this CondenseNet is slightly less robust than larger models of similar accuracy.

Even more pruning and sparsification yields a CondenseNet (C = G = 8) with both deteriorated performance (28.9% error rate) and robustness (84.6% mCE).

Here again robustness is worse than larger model robustness.

Though models fashioned for mobile devices are smaller and in some sense simpler, this does not improve robustness.

Another goal for machine learning is to learn the fundamental structure of categories.

Broad categories, such as "bird," have many subtypes, such as "cardinal" or "bluejay."

Humans can observe previously unseen bird species yet still know that they are birds.

A test of learned fundamental structure beyond superficial features is subtype robustness.

In subtype robustness we test generalization to unseen subtypes which share share essential characteristics of a broader type.

We repurpose the ImageNet-22K dataset for a closer investigation into subtype robustness.

Subtype Robustness.

A natural image dataset with a hierarchical taxonomy and numerous types and subtypes is ImageNet-22K, an ImageNet-1K superset.

In this subtype robustness experiment, we manually select 25 broad types from ImageNet-22K, listed in the next paragraph.

Each broad type has many subtypes.

We call a subtype "seen" if and only if it is in ImageNet-1K and a subtype of one of the 25 broad types.

The subtype is "unseen" if and only if it is a subtype of the 25 broad types and is from ImageNet-22K but not ImageNet-1K.

In this experiment, the correct classification decision for an image of a subtype is the broad type label.

We take pre-trained ImageNet-1K classifiers which have not trained on unseen subtypes.

Next we fine-tune the last layer of these pre-trained ImageNet-1K classifiers on seen subtypes so that they predict one of 25 broad types.

Then, we test the accuracy on images of seen subtypes and on images of unseen subtypes.

Accuracy on unseen subtypes is our measure of subtype robustness.

Seen and unseen accuracies are shown in FIG5 , while the ImageNet-1K classification accuracy before fine-tuning is on the horizontal axis.

Despite only having 25 classes and having trained on millions of images, these classifiers demonstrate a subtype robustness performance gap that should be far less pronounced.

We also observe that the architectures proposed so far hardly deviate from the trendline.

@highlight

We propose ImageNet-C to measure classifier corruption robustness and ImageNet-P to measure perturbation robustness
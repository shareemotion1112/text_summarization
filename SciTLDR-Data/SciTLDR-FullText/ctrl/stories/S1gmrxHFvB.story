Modern deep neural networks can achieve high accuracy when the training distribution and test distribution are identically distributed, but this assumption is frequently violated in practice.

When the train and test distributions are mismatched, accuracy can plummet.

Currently there are few techniques that improve robustness to unforeseen data shifts encountered during deployment.

In this work, we propose a technique to improve the robustness and uncertainty estimates of image classifiers.

We propose AugMix, a data processing technique that is simple to implement, adds limited computational overhead, and helps models withstand unforeseen corruptions.

AugMix significantly improves robustness and uncertainty measures on challenging image classification benchmarks, closing the gap between previous methods and the best possible performance in some cases by more than half.

Current machine learning models depend on the ability of training data to faithfully represent the data encountered during deployment.

In practice, data distributions evolve (Lipton et al., 2018) , models encounter new scenarios (Hendrycks & Gimpel, 2017) , and data curation procedures may capture only a narrow slice of the underlying data distribution (Torralba & Efros, 2011) .

Mismatches between the train and test data are commonplace, yet the study of this problem is not.

As it stands, models do not robustly generalize across shifts in the data distribution.

If models could identify when they are likely to be mistaken, or estimate uncertainty accurately, then the impact of such fragility might be ameliorated.

Unfortunately, modern models already produce overconfident predictions when the training examples are independent and identically distributed to the test distribution.

This overconfidence and miscalibration is greatly exacerbated by mismatched training and testing distributions.

Small corruptions to the data distribution are enough to subvert existing classifiers, and techniques to improve corruption robustness remain few in number.

Hendrycks & Dietterich (2019) show that classification error of modern models rises from 22% on the usual ImageNet test set to 64% on ImageNet-C, a test set consisting of various corruptions applied to ImageNet test images.

Even methods which aim to explicitly quantify uncertainty, such as probabilistic and Bayesian neural networks, struggle under data shift, as recently demonstrated by Ovadia et al. (2019) .

Improving performance in this setting has been difficult.

One reason is that training against corruptions only encourages networks to memorize the specific corruptions seen during training and leaves models unable to generalize to new corruptions (Vasiljevic et al., 2016; Geirhos et al., 2018) .

Further, networks trained on translation augmentations remain highly sensitive to images shifted by a single pixel (Gu et al., 2019; Hendrycks & Dietterich, 2019) .

Others have proposed aggressive data augmentation schemes (Cubuk et al., 2018) , though at the cost of a computational increase.

demonstrates that many techniques may improve clean accuracy at the cost of robustness while many techniques which improve robustness harm uncertainty, and contrariwise.

In all, existing techniques have considerable trade-offs.

In this work, we propose a technique to improve both the robustness and uncertainty estimates of classifiers under data shift.

We propose AUGMIX, a method which simultaneously achieves new state-of-the-art results for robustness and uncertainty estimation while maintaining or improving accuracy on standard benchmark datasets.

AUGMIX utilizes stochasticity and diverse augmentations, a Jensen-Shannon Divergence consistency loss, and a formulation to mix multiple augmented images to achieve state-of-the-art performance.

On CIFAR-10 and CIFAR-100, our method roughly halves the corruption robustness error of standard training procedures from 28.4% to 12.4% and 54.3% to 37.8% error, respectively.

On ImageNet, AUGMIX also achieves state-of-the-art corruption robustness and decreases perturbation instability from 57.2% to 37.4%.

Code is available at https://github.com/google-research/augmix.

Figure 2: Example ImageNet-C corruptions.

These corruptions are encountered only at test time and not during training.

Robustness under Data Shift.

Geirhos et al. (2018) show that training against distortions can often fail to generalize to unseen distortions, as networks have a tendency to memorize properties of the specific training distortion.

Vasiljevic et al. (2016) show training with various blur augmentations can fail to generalize to unseen blurs or blurs with different parameter settings.

Hendrycks & Dietterich (2019) propose measuring generalization to unseen corruptions and provide benchmarks for doing so.

Kang et al. (2019) construct an adversarial version of the aforementioned benchmark.

Gilmer et al. (2018) ; Gilmer & Hendrycks (2019) argue that robustness to data shift is a pressing problem which greatly affects the reliability of real-world machine learning systems.

Calibration under Data Shift.

Guo et al. (2017) ; Nguyen & O'Connor (2015) propose metrics for determining the calibration of machine learning models.

Lakshminarayanan et al. (2017) find that simply ensembling classifier predictions improves prediction calibration.

Hendrycks et al. (2019a) show that pre-training can also improve calibration.

Ovadia et al. (2019) demonstrate that model calibration substantially deteriorates under data shift.

Data Augmentation.

Data augmentation can greatly improve generalization performance.

For image data, random left-right flipping and cropping are commonly used He et al. (2015) .

Random occlusion techniques such as Cutout can also improve accuracy on clean data (Devries & Taylor, 2017; Zhong et al., 2017) .

Rather than occluding a portion of an image, CutMix replaces a portion of an image with a portion of a different image .

Mixup also uses information from two images.

Rather than implanting one portion of an image inside another, Mixup produces an elementwise convex combination of two images (Zhang et al., 2017; Tokozume et al., Figure 3 : A cascade of successive compositions can produce images which drift far from the original image, and lead to unrealistic images.

However, this divergence can be balanced by controlling the number of steps.

To increase variety, we generate multiple augmented images and mix them.

2018).

Guo et al. (2019) show that Mixup can be improved with an adaptive mixing policy, so as to prevent manifold intrusion.

Separate from these approaches are learned augmentation methods such as AutoAugment (Cubuk et al., 2018) , where a group of augmentations is tuned to optimize performance on a downstream task.

Patch Gaussian augments data with Gaussian noise applied to a randomly chosen portion of an image .

A popular way to make networks robust to p adversarial examples is with adversarial training (Madry et al., 2018) , which we use in this paper.

However, this tends to increase training time by an order of magnitude and substantially degrades accuracy on non-adversarial images (Raghunathan et al., 2019) .

AUGMIX is a data augmentation technique which improves model robustness and uncertainty estimates, and slots in easily to existing training pipelines.

At a high level, AugMix is characterized by its utilization of simple augmentation operations in concert with a consistency loss.

These augmentation operations are sampled stochastically and layered to produce a high diversity of augmented images.

We then enforce a consistent embedding by the classifier across diverse augmentations of the same input image through the use of Jensen-Shannon divergence as a consistency loss.

Mixing augmentations allows us to generate diverse transformations, which are important for inducing robustness, as a common failure mode of deep models in the arena of corruption robustness is the memorization of fixed augmentations (Vasiljevic et al., 2016; Geirhos et al., 2018) .

Previous methods have attempted to increase diversity by directly composing augmentation primitives in a chain, but this can cause the image to quickly degrade and drift off the data manifold, as depicted in Figure 3 .

Such image degradation can be mitigated and the augmentation diversity can be maintained by mixing together the results of several augmentation chains in convex combinations.

A concrete account of the algorithm is given in the pseudocode below.

x aug += w i · chain(x orig ) Addition is elementwise Sample weight m ∼ Beta(α, α)

Interpolate with rule x augmix = mx orig + (1 − m)x aug 13: return x augmix 14: end function 15: x augmix1 = AugmentAndMix(x orig )

x augmix1 is stochastically generated 16: Augmentations.

Our method consists of mixing the results from augmentation chains or compositions of augmentation operations.

We use operations from AutoAugment.

Each operation is visualized in Appendix C. Crucially, we exclude operations which overlap with ImageNet-C corruptions.

In particular, we remove the contrast, color, brightness, sharpness, and Cutout operations so that our set of operations and the ImageNet-C corruptions are disjoint.

In turn, we do not use any image noising nor image blurring operations so that ImageNet-C corruptions are encountered only at test time.

Operations such as rotate can be realized with varying severities, like 2

• or −15

• .

For operations with varying severities, we uniformly sample the severity upon each application.

Next, we randomly sample k augmentation chains, where k = 3 by default.

Each augmentation chain is constructed by composing from one to three randomly selected augmentation operations.

Mixing.

The resulting images from these augmentation chains are combined by mixing.

While we considered mixing by alpha compositing, we chose to use elementwise convex combinations for simplicity.

The k-dimensional vector of convex coefficients is randomly sampled from a Dirichlet(α, . . .

, α) distribution.

Once these images are mixed, we use a "skip connection" to combine the result of the augmentation chain and the original image through a second random convex combination sampled from a Beta(α, α) distribution.

The final image incorporates several sources of randomness from the choice of operations, the severity of these operations, the lengths of the augmentation chains, and the mixing weights.

Jensen-Shannon Divergence Consistency Loss.

We couple with this augmentation scheme a loss that enforces smoother neural network responses.

Since the semantic content of an image is approximately preserved with AUGMIX, we should like the model to embed x orig , x augmix1 , x augmix2 similarly.

Toward this end, we minimize the Jensen-Shannon divergence among the posterior distributions of the original sample x orig and its augmented variants.

That is, for p orig =p(y | x orig ), p augmix1 =p(y | x augmix1 ), p augmix2 =p(y|x augmix2 ), we replace the original loss L with the loss

To interpret this loss, imagine a sample from one of the three distributions p orig , p augmix1 , p augmix2 .

The Jensen-Shannon divergence can be understood to measure the average information that the sample reveals about the identity of the distribution from which it was sampled.

This loss can be computed by first obtaining M = (p orig + p augmix1 + p augmix2 )/3 and then computing

Unlike an arbitrary KL Divergence between p orig and p augmix , the Jensen-Shannon divergence is upper bounded, in this case by the logarithm of the number of classes.

Note that we could instead compute JS(p orig ; p augmix1 ), though this does not perform as well.

The gain of training with JS(p orig ; p augmix1 ; p augmix2 ; p augmix3 ) is marginal.

The Jensen-Shannon Consistency Loss impels to model to be stable, consistent, and insensitive across to a diverse range of inputs (Zheng et al., 2016; Kannan et al., 2018; .

Ablations are in Section 4.3 and Appendix A.

Datasets.

The two CIFAR (Krizhevsky & Hinton, 2009 ) datasets contain small 32 × 32 × 3 color natural images, both with 50,000 training images and 10,000 testing images.

CIFAR-10 has 10 categories, and CIFAR-100 has 100.

The ImageNet (Deng et al., 2009 ) dataset contains 1,000 classes of approximately 1.2 million large-scale color images.

In order to measure a model's resilience to data shift, we evaluate on the CIFAR-10-C, CIFAR-100-C, and ImageNet-C datasets (Hendrycks & Dietterich, 2019) .

These datasets are constructed by corrupting the original CIFAR and ImageNet test sets.

For each dataset, there are a total of 15 noise, blur, weather, and digital corruption types, each appearing at 5 severity levels or intensities.

Since these datasets are used to measure network behavior under data shift, we take care not to introduce these 15 corruptions into the training procedure.

The CIFAR-10-P, CIFAR-100-P, and ImageNet-P datasets also modify the original CIFAR and ImageNet datasets.

These datasets contain smaller perturbations than CIFAR-C and are used to measure the classifier's prediction stability.

Each example in these datasets is a video.

For instance, a video with the brightness perturbation shows an image getting progressively brighter over time.

We should like the network not to give inconsistent or volatile predictions between frames of the video as the brightness increases.

Thus these datasets enable the measurement of the "jaggedness" (Azulay & Weiss, 2018 ) of a network's prediction stream.

Metrics.

The Clean Error is the usual classification error on the clean or uncorrupted test data.

In our experiments, corrupted test data appears at five different intensities or severity levels 1 ≤ s ≤ 5.

For a given corruption c, the error rate at corruption severity s is E c,s .

We can compute the average error across these severities to create the unnormalized corruption error uCE c = 5 s=1 E c,s .

On CIFAR-10-C and CIFAR-100-C we average these values over all 15 corruptions.

Meanwhile, on ImageNet we follow the convention of normalizing the corruption error by the corruption error of AlexNet (Krizhevsky et al., 2012) .

We compute CE c =

.

The average of the 15 corruption errors CE Gaussian Noise , CE Shot Noise , . . .

, CE Pixelate , CE JPEG gives us the Mean Corruption Error (mCE).

Perturbation robustness is not measured by accuracy but whether video frame predictions match.

Consequently we compute what is called the flip probability.

Concretely, for videos such as those with steadily increasing brightness, we determine the probability that two adjacent frames, or two frames with slightly different brightness levels, have "flipped" or mismatched predictions.

There are 10 different perturbation types, and the mean across these is the mean Flip Probability (mFP).

As with ImageNet-C, we can normalize by AlexNet's flip probabilities and obtain the mean Flip Rate (mFR).

In order to assess a model's uncertainty estimates, we measure its miscalibration.

Classifiers capable of reliably forecasting their accuracy are considered "calibrated." For instance, a calibrated classifier should be correct 70% of the time on examples to which it assigns 70% confidence.

Let the classifier's confidence that its predictionŶ is correct be written C.

Then the idealized RMS Calibration Error is

, which is the squared difference between the accuracy at a given confidence level and actual the confidence level.

In Appendix E, we show how to empirically estimate this quantity and calculate the Brier Score.

Training Setup.

In the following experiments we show that AUGMIX endows robustness to various architectures including an All Convolutional Network (Springenberg et al., 2014; Salimans & Kingma, 2016) , a DenseNet-BC (k = 12, d = 100) (Huang et al., 2017) , a 40-2 Wide ResNet (Zagoruyko & Komodakis, 2016) , and a ResNeXt-29 (32 × 4) (Xie et al., 2016) .

All networks use an initial learning rate of 0.1 which decays following a cosine learning rate (Loshchilov & Hutter, 2016) .

All input images are pre-processed with standard random left-right flipping and cropping prior to any augmentations.

We do not change AUGMIX parameters across CIFAR-10 and CIFAR-100 experiments for consistency.

The All Convolutional Network and Wide ResNet train for 100 epochs, and the DenseNet and ResNeXt require 200 epochs for convergence.

We optimize with stochastic gradient descent using Nesterov momentum.

Table 1 : Average classification error as percentages.

Across several architectures, AUGMIX obtains CIFAR-10-C and CIFAR-100-C corruption robustness that exceeds the previous state of the art.

Results.

Simply mixing random augmentations and using the Jensen-Shannon loss substantially improves robustness and uncertainty estimates.

Compared to the "Standard" data augmentation baseline ResNeXt on CIFAR-10-C, AUGMIX achieves 16.6% lower absolute corruption error as shown in Figure 5 .

In addition to surpassing numerous other data augmentation techniques, Table 1 demonstrates that these gains directly transfer across architectures and on CIFAR-100-C with zero additional tuning.

Crucially, the robustness gains do not only exist when measured in aggregate.

Figure 12 shows that AUGMIX improves corruption robustness across every individual corruption and severity level.

Our method additionally achieves the lowest mFP on CIFAR-10-P across three different models all while maintaining accuracy on clean CIFAR-10, as shown in Figure 6 (left) and Table 6 .

Finally, we demonstrate that AUGMIX improves the RMS calibration error on CIFAR-10 and CIFAR-10-C, as shown in Figure 6 (right) and Table 5 .

Expanded CIFAR-10-P and calibration results are in Appendix D, and Fourier Sensitivity analysis is in Appendix B.

Baselines.

To demonstrate the utility of AUGMIX on ImageNet, we compare to many techniques designed for large-scale images.

While techniques such as Cutout (Devries & Taylor, 2017) have not been demonstrated to help on the ImageNet scale, and while few have had success training adversarially robust models on ImageNet (Engstrom et al., 2018) , other techniques such as Stylized ImageNet have been demonstrated to help on ImageNet-C. Patch Uniform ) is similar to Cutout except that randomly chosen regions of the image are injected with uniform noise; the original paper uses Gaussian noise, but that appears in the ImageNet-C test set so we use uniform noise.

We tune Patch Uniform over 30 hyperparameter settings.

Next, AutoAugment (Cubuk et al., 2018) searches over data augmentation policies to find a high-performing data augmentation policy.

We denote AutoAugment results with AutoAugment* since we remove augmentation operations that overlap with ImageNet-C corruptions, as with AUGMIX.

We also test with Random AutoAugment*, an augmentation scheme where each image has a randomly sampled augmentation policy using AutoAugment* operations.

In contrast to AutoAugment, Random AutoAugment* and AUGMIX require far less computation and provide more augmentation variety, which can offset their lack of optimization.

Note that Random AutoAugment* is different from RandAugment introduced recently by Cubuk et al. (2019): RandAugment uses AutoAugment operations and optimizes a single distortion magnitude hyperparameter for all operations, while Random AutoAugment* randomly samples magnitudes for each operation and uses the same operations as AUGMIX.

MaxBlur Pooling (Zhang, 2019 ) is a recently proposed architectural modification which smooths the results of pooling.

Now, Stylized ImageNet (SIN) is a technique where models are trained with the original ImageNet images and also ImageNet images with style transfer applied.

Whereas the original Stylized ImageNet technique pretrains on ImageNet-C and performs style transfer with a content loss coefficient of 0 and a style loss coefficient of 1, we find that using 0.5 content and style loss coefficients decreases the mCE by 0.6%.

Later, we show that SIN and AUGMIX can be combined.

All models are trained from scratch, except MaxBlur Pooling models which has trained models available.

Training Setup.

Methods are trained with ResNet-50 and we follow the standard training scheme of Goyal et al. (2017) , in which we linearly scale the learning rate with the batch size, and use a learning rate warm-up for the first 5 epochs, and AutoAugment and AUGMIX train for 180 epochs.

All input images are first pre-processed with standard random cropping horizontal mirroring.

Table 2 : Clean Error, Corruption Error (CE), and mCE values for various methods on ImageNet-C. The mCE value is computed by averaging across all 15 CE values.

AUGMIX reduces corruption error while improving clean accuracy, and it can be combined with SIN for greater corruption robustness.

Results.

Our method achieves 68.4% mCE as shown in Table 2 , down from the baseline 80.6% mCE.

Additionally, we note that AUGMIX allows straightforward stacking with other methods such as SIN to achieve an even lower corruption error of 64.1% mCE.

Other techniques such as AutoAugment* require much tuning, while ours does not.

Across increasing severities of corruptions, our method also produces much more calibrated predictions measured by both the Brier Score and RMS Calibration Error as shown in Figure 7 .

As shown in Table 3 , AUGMIX also achieves a state-of-the art result on ImageNet-P at with an mFR of 37.4%, down from 57.2%.

We demonstrate that scaling up AUGMIX from CIFAR to ImageNet also leads to state-of-the-art results in robustness and uncertainty estimation.

We locate the utility of AUGMIX in three factors: training set diversity, our Jensen-Shannon divergence consistency loss, and mixing.

Improving training set diversity via increased variety of augmentations can greatly improve robustness.

For instance, augmenting each example with a Table 3 : ImageNet-P results.

The mean flipping rate is the average of the flipping rates across all 10 perturbation types.

AUGMIX improves perturbation stability by approximately 20%.

Figure 7 : Uncertainty results on ImageNet-C. Observe that under severe data shifts, the RMS calibration error with ensembles and AUGMIX is remarkably steady.

Even though classification error increases, calibration is roughly preserved.

Severity zero denotes clean data.

randomly sampled augmentation chain decreases the error rate of Wide ResNet on CIFAR-10-C from 26.9% to 17.0% Table 4 .

Adding in the Jensen-Shannon divergence consistency loss drops error rate further to 14.7%.

Mixing random augmentations without the Jenson-Shannon divergence loss gives us an error rate of 13.1%.

Finally, re-introducing the Jensen-Shannon divergence gives us AUGMIX with an error rate of 11.2%.

Note that adding even more mixing is not necessarily beneficial.

For instance, applying AUGMIX on top of Mixup increases the error rate to 13.3%, possibly due to an increased chance of manifold intrusion (Guo et al., 2019) .

Hence AUGMIX's careful combination of variety, consistency loss, and mixing explain its performance.

CIFAR-10-C Error Rate CIFAR-100-C Error Rate Table 4 : Ablating components of AUGMIX on CIFAR-10-C and CIFAR-100-C. Variety through randomness, the Jensen-Shannon divergence (JSD) loss, and augmentation mixing confer robustness.

AUGMIX is a data processing technique which mixes randomly generated augmentations and uses a Jensen-Shannon loss to enforce consistency.

Our simple-to-implement technique obtains state-of-the-art performance on CIFAR-10/100-C, ImageNet-C, CIFAR-10/100-P, and ImageNet-P. AUGMIX models achieve state-of-the-art calibration and can maintain calibration even as the distribution shifts.

We hope that AUGMIX will enable more reliable models, a necessity for models deployed in safety-critical environments.

In this section we demonstrate that AUGMIX's hyperparameters are not highly sensitive, so that AUGMIX performs reliably without careful tuning.

For this set of experiments, the baseline AUGMIX model trains for 90 epochs, has a mixing coefficient of α = 0.5, has 3 examples per Jensen-Shannon Divergence (1 clean image, 2 augmented images), has a chain depth stochastically varying from 1 to 3, and has k = 3 augmentation chains.

Figure 8 shows that the performance of various AUGMIX models with different hyperparameters.

Under these hyperparameter changes, the mCE does not change substantially.

A commonly mentioned hypothesis (Gilmer & Hendrycks, 2019) for the lack of robustness of deep neural networks is that they readily latch onto spurious high-frequency correlations that exist in the data.

In order to better understand the reliance of models to such correlations, we measure model sensitivity to additive noise at differing frequencies.

We create a 32 × 32 sensitivity heatmap.

That is, we add a total of 32 × 32 Fourier basis vectors to the CIFAR-10 test set, one at a time, and record the resulting error rate after adding each Fourier basis vector.

Each point in the heatmap shows the error rate on the CIFAR-10 test set after it has been perturbed by a single Fourier basis vector.

Points corresponding to low frequency vectors are shown in the center of the heatmap, whereas high frequency vectors are farther from the center.

For further details on Fourier sensitivity analysis, we refer the reader to Section 2 of .

In Figure 9 we observe that the baseline model is robust to low frequency perturbations but severely lacks robustness to high frequency perturbations, where error rates exceed 80%.

The model trained with Cutout shows a similar lack of robustness.

In contrast, the model trained with AUGMIX maintains robustness to low frequency perturbations, and on the mid and high frequencies AUGMIX is conspicuously more robust.

The augmentation operations we use for AUGMIX are shown in Figure 10 .

We do not use augmentations such as contrast, color, brightness, sharpness, and Cutout as they may overlap with ImageNet-C test set corruptions.

We should note that augmentation choice requires additional care.

Guo et al. (2019) show that blithely applying augmentations can potentially cause augmented images to take different classes.

Figure 11 shows how histogram color swapping augmentation may change a bird's class, leading to a manifold intrusion.

Manifold Intrusion from Color Augmentation Figure 11 : An illustration of manifold intrusion (Guo et al., 2019) , where histogram color augmentation can change the image's class.

We include various additional results for CIFAR-10, CIFAR-10-C and CIFAR-10-P below.

Figure 12 reports accuracy for each corruption, Table 5 reports calibration results for various architectures and  Table 6 reports clean error and mFR.

We refer to Section 4.1 for details about the architecture and training setup.

Table 6 : CIFAR-10 Clean Error and CIFAR-10-P mean Flip Probability.

All values are percentages.

While adversarial training performs well on CIFAR-10-P, it induces a substantial drop in accuracy (increase in error) on clean CIFAR-10 where AUGMIX does not.

Due to the finite size of empirical test sets, the RMS Calibration Error must be estimated by partitioning all n test set examples into b contiguous bins {B 1 , B 2 , . . .

, B b } ordered by prediction confidence.

In this work we use bins which contain 100 predictions, so that we adaptively partition confidence scores on the interval [0, 1] (Nguyen & O'Connor, 2015; Hendrycks et al., 2019b) .

Other works partition the interval [0, 1] with 15 bins of uniform length (Guo et al., 2017) .

With these b bins, we estimate the RMS Calibration Error empirically with the formula

This is separate from classification error because a random classifier with an approximately uniform posterior distribution is approximately calibrated.

Also note that adding the "refinement" E C [(P(Y = Y |C = c)(1 − (P(Y =Ŷ |C = c))] to the square of the RMS Calibration Error gives us the Brier Score (Nguyen & O'Connor, 2015) .

<|TLDR|>

@highlight

We obtain state-of-the-art on robustness to data shifts, and we maintain calibration under data shift even though even when accuracy drops
This paper investigates strategies that defend against adversarial-example attacks on image-classification systems by transforming the inputs before feeding them to the system.

Specifically, we study applying image transformations such as bit-depth reduction, JPEG compression, total variance minimization, and image quilting before feeding the image to a convolutional network classifier.

Our experiments on ImageNet show that total variance minimization and image quilting are very effective defenses in practice, in particular, when the network is trained on transformed images.

The strength of those defenses lies in their non-differentiable nature and their inherent randomness, which makes it difficult for an adversary to circumvent the defenses.

Our best defense eliminates 60% of strong gray-box and 90% of strong black-box attacks by a variety of major attack methods.

As the use of machine intelligence increases in security-sensitive applications BID2 BID0 , robustness has become a critical feature to guarantee the reliability of deployed machine-learning systems.

Unfortunately, recent research has demonstrated that existing models are not robust to small, adversarially designed perturbations of the input BID1 BID31 BID14 BID20 BID6 .

Adversarially perturbed examples have been deployed to attack image classification services BID22 , speech recognition systems BID6 , and robot vision BID25 .

The existence of these adversarial examples has motivated proposals for approaches that increase the robustness of learning systems to such examples BID28 BID20 BID7 .The robustness of machine learning models to adversarial examples depends both on the properties of the model (i.e., Lipschitzness) and on the nature of the problem considered, e.g., on the input dimensionality and the Bayes error of the problem BID11 .

Consequently, defenses that aim to increase robustness against adversarial examples fall in one of two main categories.

The first category comprises model-specific strategies that enforce model properties such as invariance and smoothness via the learning algorithm or regularization scheme BID30 BID20 BID7 , potentially exploiting knowledge about the adversary's attack strategy BID14 .

The second category of defenses are model-agnostic: they try to remove adversarial perturbations from the input.

For example, in the context of image classification, adversarial perturbations can be partly removed via JPEG compression BID9 or image re-scaling BID23 .

Hitherto, none of these defenses has been shown to be very effective.

Specifically, model-agnostic defenses appear too simple to sufficiently remove adversarial perturbations from input images.

By contrast, model-specific defenses make strong assumptions about the nature of the adversary (e.g., on the norm that the adversary minimizes or on the number of iterations it uses to generate the perturbation).

Consequently, they do not satisfy BID18 principle: the adversary can alter its attack to circumvent such model-specific defenses.

In this paper, we focus on increasing the effectiveness of model-agnostic defense strategies by developing approaches that (1) remove the adversarial perturbations from input images, (2) maintain sufficient information in input images to correctly classify them, and (3) are still effective in settings in which the adversary has information on the defense strategy being used.

We explore transformations based on image cropping and rescaling BID15 , bit-depth reduction ), JPEG compression (Dziugaite et al., 2016 , total variance minimization BID29 , and image quilting BID10 .

We show that these defenses can be surprisingly effective against existing attacks, in particular, when the convolutional network is trained on images that are transformed in a similar way.

The image transformations are good at countering the (iterative) fast gradient sign method BID20 ), Deepfool (Moosavi-Dezfooli et al., 2016 , and the BID5 attack, even in gray-box settings in which the model architecture and parameters are public.

Our strongest defenses are based on total variation minimization and image quilting: these defenses are non-differentiable and inherently random, which makes it difficult for an adversary to get around them.

Our best defenses eliminate 60% of gray-box attacks and 90% of black-box attacks by four major attack methods that perturb pixel values by 8% on average.

We study defenses against non-targeted adversarial examples for image-recognition systems.

Let DISPLAYFORM0 H×W ×C be the image space.

Given an image classifier h(·) and a source image x ∈ X , a non-targeted 1 adversarial example of x is a perturbed image x ∈ X such that h(x) = h(x ) and d(x, x ) ≤ ρ for some dissimilarity function d(·, ·) and ρ ≥ 0.

Ideally, d(·, ·) measures the perceptual difference between x and x but, in practice, the Euclidean distance d(x, x ) = x−x 2 or the Chebyshev distance d(x, x ) = x − x ∞ is most commonly used.

Given a set of N images {x 1 , . . .

, x N } and a target classifier h(·), an adversarial attack aims to generate {x 1 , . . .

, x N } such that each x n is an adversarial example for x n .

The success rate of an attack is measured by the proportion of predictions that was altered by an attack: DISPLAYFORM1 The success rate is generally measured as a function of the magnitude of the perturbations performed by the attack, using the normalized L 2 -dissimilarity: DISPLAYFORM2 A strong adversarial attack has a high success rate whilst its normalized L 2 -dissimilarity is low.

In most practical settings, an adversary does not have direct access to the model h(·) and has to do a black-box attack.

However, prior work has shown successful attacks by transferring adversarial examples generated for a separately-trained model to an unknown target model BID22 .

Therefore, we investigate both the black-box and a more difficult gray-box attack setting: in our gray-box setting, the adversary has access to the model architecture and the model parameters, but is unaware of the defense strategy that is being used.

A defense is an approach that aims make the prediction on an adversarial example h(x ) equal to the prediction on the corresponding clean example h(x).

In this study, we focus on imagetransformation defenses g(x) that perform prediction via h(g(x )).

Ideally, g(·) is a complex, nondifferentiable, and potentially stochastic function: this makes it difficult for an adversary to attack the prediction model h(g(x)) even when the adversary knows both h(·) and g(·).

One of the first successful attack methods is the fast gradient sign method (FGSM; BID14 ).

Let (·, ·) be the differentiable loss function that was used to train the classifier h(·), e.g., the cross-entropy loss.

The FGSM adversarial example corresponding to a source input x and true label y is: DISPLAYFORM0 for some > 0 that governs the perturbation magnitude.

A stronger variant of this attack, called iterative FGSM (I-FGSM; BID21 ), iteratively applies the FGSM update: Alternative attacks aim to minimize the Euclidean distance between the input and the adversarial example instead.

For instance, assuming h(·) is a binary classifier, DeepFool (Moosavi-Dezfooli et al., 2016) projects x onto a linearization of the decision boundary defined by h(·) for M iterations: DISPLAYFORM1 DISPLAYFORM2 where x (0) and x are defined as in I-FGSM.

The multi-class variant of DeepFool performs the projection onto the nearest class boundaries.

The linearization performed in DeepFool is particularly well suited for ReLU-networks, as these represent piecewise linear class boundaries.

Carlini-Wagner's L 2 attack (CW-L2; BID5 ) is an optimization-based attack that combines a differentiable surrogate for the model's classification accuracy with an L 2 -penalty term.

Let Z(x) be the operation that computes the logit vector (i.e., the output before the softmax layer) for an input x, and Z(x) k be the logit value corresponding to class k. The untargeted variant of CW-L2 finds a solution to the unconstrained optimization problem min DISPLAYFORM3 where κ denotes a margin parameter, and where the parameter λ f trades off the perturbation norm and the hinge loss of predicting a different class.

We perform the minimization over x using the Adam optimizer BID19 for 100 iterations with an initial learning rate of 0.001.All of the aforementioned attacks enforce that x ∈ X by clipping values between 0 and 1.

FIG0 shows adversarial images produced by all four attacks at five normalized L 2 -dissimilarity levels.

Adversarial attacks alter particular statistics of the input image in order to change the model prediction.

Indeed, adversarial perturbations x−x have a particular structure, as illustrated by FIG0 .

We design and experiment with image transformations that alter the structure of these perturbations, and investigate whether the alterations undo the effects of the adversarial attack.

We investigate five image transformations: (1) image cropping and rescaling, (2) bit-depth reduction, (3) JPEG compression, (4) total variance minimization, and (5) image quilting.

Figure 2: Illustration of total variance minimization and image quilting applied to an original and an adversarial image (produced using I-FGSM with = 0.03, corresponding to a normalized L 2 -dissimilarity of 0.075).

From left to right, the columns correspond to: (1) no transformation, (2) total variance minimization, and (3) image quilting.

From top to bottom, rows correspond to: (1) the original image, (2) the corresponding adversarial image produced by I-FGSM, and (3) the absolute difference between the two images above.

Difference images were multiplied by a constant scaling factor to increase visibility.

We first introduce three simple image transformations: image cropping-rescaling BID15 , bit-depth reduction , and JPEG compression and decompression BID9 .

Image croppingrescaling has the effect of altering the spatial positioning of the adversarial perturbation, which is important in making attacks successful.

Following BID16 , we crop and rescale images at training time as part of the data augmentation.

At test time, we average predictions over random image crops.

Bitdepth reduction ) perform a simple type of quantization that can removes small (adversarial) variations in pixel values from an image; we reduce images to 3 bits in our experiments.

JPEG compression (Dziugaite et al., 2016) removes small perturbations in a similar way; we perform compression at quality level 75 (out of 100).

An alternative way of removing adversarial perturbations is via a compressed sensing approach that combines pixel dropout with total variation minimization BID29 .

This approach randomly selects a small set of pixels, and reconstructs the "simplest" image that is consistent with the selected pixels.

The reconstructed image does not contain the adversarial perturbations because these perturbations tend to be small and localized.

Specifically, we first select a random set of pixels by sampling a Bernoulli random variable X(i, j, k) for each pixel location (i, j, k); we maintain a pixel when X(i, j, k) = 1.

Next, we use total variation minimization to constructs an image z that is similar to the (perturbed) input image x for the selected set of pixels, whilst also being "simple" in terms of total variation by solving: DISPLAYFORM0 Herein, denotes element-wise multiplication, and TV p (z) represents the L p -total variation of z: DISPLAYFORM1 The total variation (TV) measures the amount of fine-scale variation in the image z, as a result of which TV minimization encourages removal of small (adversarial) perturbations in the image.

The objective function (6) is convex in z, which makes solving for z straightforward.

In our implementation, we set p = 2 and employ a special-purpose solver based on the split Bregman method BID13 ) to perform total variance minimization efficiently.

The effectiveness of TV minimization is illustrated by the images in the middle column of Figure 2 : in particular, note that the adversarial perturbations that were present in the background for the nontransformed image (see bottom-left image) have nearly completely disappeared in the TV-minimized adversarial image (bottom-center image).

As expected, TV minimization also changes image structure in non-homogeneous regions of the image, but as these perturbations were not adversarially designed we expect the negative effect of these changes to be limited.

Image quilting BID10 ) is a non-parametric technique that synthesizes images by piecing together small patches that are taken from a database of image patches.

The algorithm places appropriate patches in the database for a predefined set of grid points, and computes minimum graph cuts BID3 ) in all overlapping boundary regions to remove edge artifacts.

Image quilting can be used to remove adversarial perturbations by constructing a patch database that only contains patches from "clean" images (without adversarial perturbations); the patches used to create the synthesized image are selected by finding the K nearest neighbors (in pixel space) of the corresponding patch from the adversarial image in the patch database, and picking one of these neighbors uniformly at random.

The motivation for this defense is that the resulting image only consists of pixels that were not modified by the adversary -the database of real patches is unlikely to contain the structures that appear in adversarial images.

The right-most column of Figure 2 illustrates the effect of image quilting on adversarial images.

Whilst interpretation of these images is more complicated due to the quantization errors that image quilting introduces, it is interesting to note that the absolute differences between quilted original and the quilted adversarial image appear to be smaller in non-homogeneous regions of the image.

This suggests that TV minimization and image quilting lead to inherently different defenses.

We performed five experiments to test the efficacy of our defenses.

The experiment in Section 5.2 considers gray-box attacks: it applies the defenses on adversarial images before using them as input into a convolutional network trained to classify "clean" images.

In this setting, the adversary has access to the model architecture and parameters but is unaware of the defense strategy.

The experiment in Section 5.3 focuses on a black-box setting: it replaces the convolutional network by networks that were trained on images with a particular input-transformation.

The experiment in Section 5.4 combines our defenses with ensembling and model transfer.

The experiment in Section 5.5 investigates to what extent networks trained on image-transformations can be attacked in a gray-box setting.

The experiment in Section 5.6 compares our defenses with prior work.

The setup of our gray-box and black-box experiments is illustrated in FIG1 .

Code to reproduce our results is available at https://github.com/facebookresearch/adversarial_image_defenses.

We performed experiments on the ImageNet image classification dataset.

The dataset comprises 1.2 million training images and 50, 000 test images that correspond to one of 1, 000 classes.

Our adversarial images are produced by attacking a ResNet-50 model BID16 .

We evaluate our defense strategies against the four adversarial attacks presented in Section 3.

We measure the strength of an adversary in terms of its normalized L 2 -dissimilarity and report classification accu- racies as a function of the normalized L 2 -dissimilarity.

To produce adversarial images like those in FIG0 , we set the normalized L 2 -dissimilarity for each of the attacks as follows:• FGSM.

Increasing the step size increases the normalized L 2 -dissimilarity.• I-FGSM.

We fix M = 10, and increase to increase the normalized L 2 -dissimilarity.• DeepFool.

We fix M = 5, and increase to increase the normalized L 2 -dissimilarity.• CW-L2.

We fix κ = 0 and λ f = 10, and multiply the resulting perturbation by an appropriately chosen ≥ 1 to alter the normalized L 2 -dissimilarity.

We fixed the hyperparameters of our defenses in all experiments: specifically, we set pixel dropout probability p = 0.5 and the regularization parameter of the total variation minimizer λ TV = 0.03.

We use a quilting patch size of 5×5 and a database of 1, 000, 000 patches that were randomly selected from the ImageNet training set.

We use the nearest neighbor patch (i.e., K = 1) for experiments in Sections 5.2 and 5.3, and randomly select a patch from one of K = 10 nearest neighbors in all other experiments.

In the cropping defense, we sample 30 crops of size 90×90 from the 224×224 input image, rescale the crops to 224×224, and average the model predictions over all crops.

FIG2 shows the top-1 accuracy of a ResNet-50 tested on transformed adversarial images as a function of the adversary strength for each of the four attacks.

Each plot shows results for five different transformations we apply to the images at test time (viz., image cropping-rescaling, bitdepth reduction, JPEG compression, total variation minimization, and image quilting).

The dotted line shows the classification error of the ResNet-50 model on images that are not adversarially perturbed, i.e., it gives an upper bound on the accuracy that defenses can achieve.

In line with the results reported in the literature, the four adversaries successfully attack the ResNet-50 model in nearly all cases (FGSM has a slightly lower favorable attack rate of 80−90%) when the input images are not transformed.

The results also show that the proposed image transformations are capable of partly eliminating the effect of the attacks.

In particular, ensembling 30 predictions over different, random image crops is very efficient: these predictions are correct for 40−60% of the images (note that 76% is the highest accuracy that one can expect to achieve).

This result suggests that adversarial examples are susceptible to changes in the location and scale of the adversarial perturbations.

While not as effective, image transformations based on total variation minimization and image quilting also successfully defend against adversarial examples from all four attacks: applying these transformations allows us to classify 30−40% of the images correctly.

This result suggests that total variation minimization and image quilting can successfully remove part of the perturbations from adversarial images.

In particular, the accuracy of the image-quilting defense hardly deteriorates as the strength of the adversary increases.

However, the quilting transformation does severely impact the model's accuracy on non-adversarial images.

The high relative performance of image cropping-rescaling in 5.2 may be partly explained by the fact that the convolutional network was trained on randomly cropped-rescaled images 2 , but not on any of the other transformations.

This implies that independent of whether an image is adversarial or not, the network is more robust to image cropping-rescaling than it is to those transformations.

The results in FIG2 suggest that this negatively affects the effectiveness of these defenses, even if the defenses are successful in removing the adversarial perturbation.

To investigate this, we trained ResNet-50 models on transformed ImageNet training images.

We adopt the standard data augmentation from BID16 , but apply bit-depth reduction, JPEG compression, TV minimization, or image quilting on the resized image crop before feeding it to the network.

We measure the classification accuracy of the resulting networks on the same adversarial images as before.

Note that this implies that we assume a black-box setting in this experiment.

Table 1 : Top-1 classification accuracy of ensemble and model transfer defenses (columns) against four black-box attacks (rows).

The four networks we use to classify images are ResNet-50 (RN50), ResNet-101 (RN101), DenseNet-169 (DN169), and Inception-v4 (Iv4).

Adversarial images are generated by running attacks against the ResNet-50 model, aiming for an average normalized L 2 -dissimilarity of 0.06.

Higher is better.

The best defense against each attack is typeset in boldface.

We present the results of these experiments in FIG3 .

Training convolutional networks on images that are transformed in the same way as at test time, indeed, dramatically improves the effectiveness of all transformation defenses.

In our experiments, the image-quilting defense is particularly effective against strong attacks: it successfully defends against 80−90% of all four attacks, even when the normalized L 2 -dissimilarity of the attack approaches 0.08.

We evaluate the efficacy of (1) ensembling different defenses and (2) "transferring" attacks to different network architectures (in a black-box setting).

Specifically, we measured the accuracy of four networks using ensembles of defenses on adversarial images generated to attack a ResNet-50; the four networks we consider are ResNet-50, ResNet-101, DenseNet-169 BID17 , and Inception-v4 BID33 .

To ensemble the image quilting and TVM defenses, we average the image-quilting prediction (using a weight of 0.5) with model predictions for 10 different TVM reconstructions (with a weight of 0.05 each), re-sampling the pixels used to measure the reconstruction error each time.

To combine cropping with other transformations, we first apply those transformations and average predictions over 10 random crops from the transformed images.

The results of our ensembling experiments are presented in Table 1 .

The results show that gains of 1 − 2% in classification accuracy can be achieved by ensembling different defenses, whereas transferring attacks to different convolutional network architectures can lead to an improvement of 2−3%.

Inception-v4 performs best in our experiments, but this may be partly due to that network having a higher accuracy even in non-adversarial settings.

Our best black-box defense achieves an accuracy of about 71% against all four defenses: the attacks deteriorate the accuracy of our best classifier (which combines cropping, TVM, image quilting, and model transfer) by at most 6%.

The previous experiments demonstrated the effectiveness of image transformations against adversarial images, in particular, when convolutional networks are re-trained to be robust to those image transformations.

In this experiment, we investigate to what extent the resulting networks can be attacked in a gray-box setting in which the adversary has access to those networks (but does not have access to the input transformations applied at test time).

We use the four attack methods to generate novel adversarial images against the transformation-robust networks trained in 5.3, and measure the accuracy of the networks on these novel adversarial images in FIG4 .The results show that bit-depth reduction and JPEG compression are weak defenses in such a graybox setting.

Whilst their relative ordering varies between attack methods, image cropping and rescaling, total variation minimization, and image quilting are fairly robust defenses in the white-box setting.

Specifically, networks using these defenses classify up to 50% of adversarial images correctly.

In our final set of experiments, we compare our defenses with the state-of-the-art ensemble adversarial training approach proposed by BID34 .

Ensemble adversarial training fits the parameters of a convolutional network on adversarial examples that were generated to attack an ensemble of pre-trained models.

These adversarial examples are very diverse, which makes the convolutional network being trained robust to a variety of adversarial perturbation.

In our experiments, we used the model released by BID34 : an Inception-Resnet-v2 BID32 trained on adversarial examples generated by FGSM against Inception-Resnet-v2 and Inception-v3 models.

We compare the model to our ResNet-50 models with image cropping, total variance minimization, and image quilting defenses.

We note that there are two small differences in terms of the assumptions that ensemble adversarial training makes and the assumptions our defenses make: (1) in contrast to ensemble adversarial training, our defenses assume that part of the defense strategy (viz., the input transformation) is unknown to the adversary, and (2) in contrast to ensemble adversarial training, our defenses assume no prior knowledge of the attacks being used.

The former difference is advantageous to our defenses, whereas the latter difference gives our defenses a disadvantage compared to ensemble adversarial training.

Table 2 compares the classification accuracies of the defense strategief on adversarial examples with a normalized L 2 -dissimilarity of 0.06.

The results show that ensemble adversarial training works better on FGSM attacks (which it uses at training time), but is outperformed by each of the transformation-based defenses all other attacks.

Input transformations particularly outperform ensemble adversarial training against the iterative attacks: our defense are are 18−24× more robust than ensemble adversarial training against DeepFool attacks.

Combining cropping, TVM, and quilting increases the accuracy of our defenses against DeepFool gray-box attacks to 51.51% (compared to 1.84% for ensemble adversarial training).

The results from this study suggest there exists a range of image transformations that have the potential to remove adversarial perturbations while preserving the visual content of the image: one merely has to train the convolutional network on images that were transformed in the same way.

A critical property that governs which image transformations are most effective in practice is whether Table 2 : Top-1 classification accuracy on images perturbed using attacks against ResNet-50 models trained on input-transformed images, and an Inception-v4 model trained using ensemble adversarial.

Adversarial images are generated by running attacks against the models, aiming for an average normalized L 2 -dissimilarity of 0.06.

The best defense against each attack is typeset in boldface.an adversary can incorporate the transformation in its attack.

For instance, median filtering likely is a weak remedy because one can backpropagate through the median filter, which is sufficient to perform any of the attacks described in Section 3.

A strong input-transformation defense should, therefore, be non-differentiable and randomized, a strategy has been previously shown to be effective BID35 b) .

Two of our top defenses possess both properties:1.

Both total variation minimization and image quilting are difficult to differentiate through.

Specifically, total variation minimization involves solving a complex minimization of a function that is inherently random.

Image quilting involves a discrete variable that selects the patch from the database, which is a non-differentiable operation, and the graph-cut optimization complicates the use of differentiable approximations BID24 .2.

Both total variation minimization and image quilting give rise to randomized defenses.

Total variation minimization randomly selects the pixels it uses to measure reconstruction error on when creating the denoised image.

Image quilting randomly selects one of the K nearest neighbors uniformly at random.

The inherent randomness of our defenses makes it difficult to attack the model: it implies the adversary has to find a perturbation that alters the prediction for the entire distribution of images that could be used as input, which is harder than perturbing a single image BID27 .Our results with gray-box attacks suggest that randomness is particularly important in developing strong defenses.

Therefore, we surmise that total variation minimization, image quilting, and related methods BID8 are stronger defenses than deterministic denoising procedures such as bit-depth reduction, JPEG compression, or non-local means BID4 .

Defenses based on total variation minimization and image quilting also have an advantage over adversarial-training approaches BID20 : an adversarially trained network is differentiable, which implies that it can be attacked using the methods in Section 3.

An additional disadvantage of adversarial training is that it focuses on a particular attack; by contrast, transformation-based defenses generalize well across attack methods because they are model-agnostic.

While our study focuses exclusively on image classification, we expect similar defenses to be useful in other domains for which successful attacks have been developed, such as semantic segmentation and speech recognition BID6 BID38 .

In speech recognition, for example, total variance minimization can be used to remove perturbations from waveforms, and one could develop "spectrogram quilting" techniques that reconstruct a spectrogram by concatenating "spectrogram patches" along the temporal dimension.

We leave such extensions to future work.

In future work, we also intend to study combinations of our input-transformation defenses with ensemble adversarial training BID34 , and we intend to investigate new attack methods that are specifically designed to circumvent our input-transformation defenses.

<|TLDR|>

@highlight

We apply a model-agnostic defense strategy against adversarial examples and achieve 60% white-box accuracy and 90% black-box accuracy against major attack algorithms.
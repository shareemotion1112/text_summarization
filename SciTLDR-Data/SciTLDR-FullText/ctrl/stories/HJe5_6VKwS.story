Adversarial perturbations cause a shift in the salient features of an image, which may result in a misclassification.

We demonstrate that gradient-based saliency approaches are unable to capture this shift, and develop a new defense which detects adversarial examples based on learnt saliency models instead.

We study two approaches: a CNN trained to distinguish between natural and adversarial images using the saliency masks produced by our learnt saliency model, and a CNN trained on the salient pixels themselves as its input.

On MNIST, CIFAR-10 and ASSIRA, our defenses are able to detect various adversarial attacks, including strong attacks such as C&W and DeepFool, contrary to gradient-based saliency and detectors which rely on the input image.

The latter are unable to detect adversarial images when the L_2- and L_infinity- norms of the perturbations are too small.

Lastly, we find that the salient pixel based detector improves on saliency map based detectors as it is more robust to white-box attacks.

Adversarial examples highlight a crucial difference between human vision and computer image processing.

Often computers fail to understand the relevant characteristics of an image for classification (Ribeiro et al., 2016) or fail to generalize locally, i.e., misclassify examples close to the training data (Szegedy et al., 2013) .

Attacks exploit this property by altering pixels the classifier heavily relies on -pixels which are irrelevant to humans for object recognition.

As a consequence, adversarial perturbations fool classifiers while the correct class remains clear to humans.

Saliency maps identify the pixels an image classifier uses for its prediction; as such, they can be used as a tool to understand why a classifier is fooled.

Building on this concept, researchers have shown qualitatively that adversarial perturbations cause a shift in the saliency of classifiers (Fong & Vedaldi, 2017; Gu & Tresp, 2019) .

Figure 1 shows examples of a natural image and corresponding adversarial images, each above their respective saliency maps.

The saliency maps corresponding to adversarial images show perceptible differences to that of the original image, even though adversarial images themselves often seem unperturbed.

For the original image, the saliency map shows that the classifier focuses on the four (and a couple of random pixels on the left).

We observe that for the adversarial images, the classifier starts focusing more on irrelevant aspects of the left side of the image.

There is ample research into different techniques for finding saliency maps (see e.g. Zeiler & Fergus, 2014; Springenberg et al., 2014; Bach et al., 2015; Ribeiro et al., 2016; Shrikumar et al., 2017; Selvaraju et al., 2017; Zintgraf et al., 2017; Fong & Vedaldi, 2017) .

However, not all saliency maps are equally informative (Fong & Vedaldi, 2017) .

For example, the Jacobian 1 can be used to determine the saliency of a pixel in the classification of the image (Papernot et al., 2016b; Zhang et al., 2018) .

As the Jacobian is often used to generate adversarial examples, intuitively, we expect that it can be used effectively to detect adversarial perturbations.

Zhang et al. (2018) propose a defense to this effect: they determine whether an input is adversarial, given the Jacobian-based The top is the input image and the bottom shows the corresponding saliency map.

In the second row, lighter colours correspond to higher saliency (black corresponds to a saliency of 0, the lowest possible value).

The classifier predicts (from left to right) the images as: 4, 9, 9 , 8, 9, 9.

Note the stark difference between the saliency masks of the original image and those of the adversarial examples.

saliency map concatenated with the image.

However, as shown qualitatively by Gu & Tresp (2019) , gradients are not always able to capture differences between adversarial images and natural images (for an example see Figures 7 and 8 in Appendix D).

2 Here we inspect the proposed Jacobian-based approach and show that only the concatenated input affects the technique's performance in detecting adversarial examples, with the Jacobian having no effect.

While gradients may not be informative for detection, saliency should be an effective tool for detecting adversarial images.

In our analysis, we use more powerful model-based saliency techniques and show that the magnitude of the shift of the saliency map due to adversarial perturbations often exceeds the L 2 distance between the saliency maps of different natural images.

Building on this result, we consider two different possible effects adversarial perturbations might have on the classifier: 1.

They might cause the classifier to focus on the wrong pixel locations 2.

They might change the pixel values of salient pixels Based on these hypotheses, we employ two CNN classifier architectures to detect adversarial images.

Claim (1) can be captured by shifts in saliency maps, as previously considered by Fong & Vedaldi (2017) .

In this work, we extend on their analysis 3 by proving the defensive capability of our model-based saliency against difficult black-box attacks, such as C&W and DeepFool 4 , as well as white-box adversarial attacks.

By considering claim (2), we demonstrate that incorporating pixel values improves the performance of the classifier when shifts in saliency maps do not suffice to capture adversarial perturbations.

We also show that our salient-pixel based defense generalizes well (detecting stronger attacks when trained on weaker attacks) and is more robust than the saliency map defense against white-box attacks.

Lastly, we demonstrate that saliency can be used to detect adversarial examples generated by small perturbations, contrary to other defenses, which exhibit threshold behavior: i.e., when the adversarial perturbation is too small, other defenses (specifically Gong et al., 2017; Zhang et al., 2018) are unable to detect the adversarial images.

Saliency maps and adversarial perturbations have similar mathematical formulations and derivations.

Both are computed by investigating the relation between the values of pixels and the classification score.

Adversarial examples are found by deriving the minimal perturbations required to change the classification of an image.

Saliency is computed by finding the pixels used by the model 2 Similarly, Fong & Vedaldi (2017) show that gradient-based heat maps are less effective than other saliency methods in detecting adversarial perturbations generated using BIM (Kurakin et al., 2016) .

3 Their main contribution is that saliency maps generated by different techniques are not equally effective in capturing changes due to adversarial perturbations (produced using BIM (Kurakin et al., 2016) .

4 These attacks generate smaller L2 perturbations, making them more difficult to detect.

The perturbation size used by Fong & Vedaldi (2017) can likely still be detected by a simple classifier that trains on images.

to determine the class of an object (Simonyan et al., 2013) .

Saliency maps can be found by considering the smallest part of an image that is sufficient for a correct classification, known as the smallest sufficient region (SSR), or whose removal is sufficient for an incorrect classification, known as the smallest destroying region (SDR) (Dabkowski & Gal, 2017; Fong & Vedaldi, 2017) .

Observe that the latter definition of saliency is very close to that of adversarial examples.

Mathematically, both saliency maps and adversarial perturbations can be derived in a similar fashion.

Consider adversarial examples.

The general formulation of an adversarial attack can be summarized as follows:

where x is the natural image, r is the adversarial perturbation, y is the correct class, and y is an incorrect class.

Due to the non-linearity of NNs, solving the above problem requires non-linear optimization.

Therefore, in practice several different approaches to solving the above formulation have been implemented.

For example, Goodfellow et al. (2014) set r = ??sign(

??x ).

Similarly, saliency can be computed using the forward derivative

??x (Papernot et al., 2016b; Zhang et al., 2018) .

Previous research has already started investigating the relation between saliency and adversarial examples.

This includes:

Using saliency to attack Researchers have devised adversarial attacks that use saliency (Papernot et al., 2016b; Yu et al., 2018) .

The key idea is to use saliency to determine the pixel that is most sensitive to perturbation iteratively.

The main benefit is that fewer pixels are perturbed -often perturbing as few as 4% of the pixels suffices to change the classification of the image (Papernot et al., 2016b) .

Using saliency to defend Fong & Vedaldi (2017) introduce a method that detects adversarial perturbations by using heat-map visualizations of the predicted class.

However, in their analysis, they only use BIM (Kurakin et al., 2016) , which is easily detected.

Further, Zhang et al. (2018) hypothesize that there is a mismatch between the saliency of a classification model and the adversarial example.

They propose a defense against adversarial attacks by training a classifier on images concatenated with their saliency map, which is computed by calculating the Jacobian of the classifier with respect to the image x, i.e., s x = ??? x f (x).

Zhang et al. (2018) find that their method obtains a high accuracy (often near 100%) when detecting adversarial images generated by FGSM, MIM, and C&W attacks on MNIST, CIFAR-10, and 10-ImageNet.

However, Gu & Tresp (2019) contradict these results, and demonstrate that the gradients show imperceptible differences due to adversarial perturbations (see Figures 7 and 8 in Appendix D).

Adversarial robustness and interpretability of models Fong & Vedaldi (2017) and Gu & Tresp (2019) 5 show that saliency maps can be used to explain adversary classifications.

Both highlight an important trend: not all techniques used to compute saliency maps show shifts in saliency maps due to adversarial perturbations.

Further, Tsipras et al. (2018) shows that more robust models have more interpretable saliency masks.

Etmann et al. (2019) quantify the relation by investigating the alignment between the saliency map and the input image.

In this section, we explain how we construct and evaluate our saliency-based adversarial example detectors.

We train a convolutional neural network image classifier, which we target with black-box attacks; the architectures are summarized in Appendix A. We use cross-entropy loss and optimize the parameters using Adam (Kingma & Ba, 2014) Goodfellow et al., 2014; Kurakin et al., 2016; Dong et al., 2018; Carlini & Wagner, 2017b; Papernot et al., 2016b; Moosavi-Dezfooli et al., 2016, respectively) .

We use the implementation as provided in cleverhans (Papernot et al., 2016a).

The hyper-parameters are summarized in Appendix B.

To generate saliency masks, we adapt the method used by Dabkowski & Gal (2017) .

Our reason is twofold: the technique computes high-quality saliency masks at a low computational cost.

Dabkowski & Gal (2017) employ a U-Net with a novel loss function that targets SDR, SSR, mask sparsity, and mask smoothness.

We adapt the original loss function to omit the total variational term, as mask smoothness is not required in our analysis.

) denote the generated map.

First, the map average AV (f s ) is used to ensure that the area of the map is small.

Second, log(f c (??(x, f s ))) is included to ensure that the salient pixels suffice to identify the correct class.

Finally, f c (??(x, 1 ??? f s )) is included to ensure that the classifier can no longer recognize the class if the saliency map is removed.

Therefore, our saliency loss function is:

where f c is the softmax probability of the class c, ??(x, f s ) applies mask f s to image x, and ?? i ??? 0 are hyper-parameters.

We adapt the PyTorch implementation provided by Dabkowski & Gal (2017) 6 and train the saliency model on standard, non-adversarial images only.

For evaluation, we use the same saliency model for both natural and adversarial images.

When generating the saliency maps for our images, we use the predicted classification for feature selection to prevent an information leak (which would occur if we use the true label).

Our hypothesis is that if an image is adversarial, the classifier likely focuses on the wrong aspects or the pixels on which it focuses are misleading (due to the perturbed color or intensity) when classifying an image as adversarial.

We consider two different cases by building classifiers for (1) saliency maps and (2) salient pixels.

For both classifiers, we use the same architecture (and hyperparameters) as for the black-box image classifiers (as summarized in Appendix A).

We build a detector based on the saliency maps of images as follows.

First, we train a classifier and generate adversarial images for every natural image in the training dataset.

Then we generate the saliency maps for the clean data {f s (X)} and adversarial images {f s (X adv )}.

We build a binary detector for the saliency maps, which predicts whether the corresponding image is adversarial or natural.

We abbreviate this defense as SMD (Saliency Map Defense).

We do not concatenate the saliency maps to the input image.

We construct a second classifier for the salient pixels.

We follow the same steps as outlined in the previous section, aside from the final step.

We define the salient pixels as f s (x) ?? x, where x is the image, f s (x) is the saliency map corresponding to x and ?? denotes the element-wise product.

We abbreviate this defense as SPD (Salient Pixel Defense).

Similarly to SMD, we do not concatenate the saliency maps to the input image.

To benchmark our results, we consider two baselines.

First, we train a baseline classifier that classifies input as adversarial or natural based on the images alone.

This allows us to evaluate the added benefit of using saliency maps.

This method was implemented by Gong et al. (2017) .

We abbreviate this defense as ID (Image Defense).

Second, we compare our defense method with the saliency-based defense of Zhang et al. (2018) (see Section 2).

We abbreviate this defense as JSD, for Jacobian-based Saliency map Defense.

In our implementation, we adapt the method of Zhang et al. (2018); we find that if we use f s (x) = ??? x f (x) as the saliency map it leads to underflow, resulting in a zero matrix.

Therefore, instead we take the derivative with respect to the logits, i.e. f s (x) = ??? x z(x).

JSD is mathematically related to the other defenses.

First, it is more general compared to ID: the filters of JSD can learn to ignore the Jacobian-based saliency, in which case the two methods are equivalent.

Further, JSD is similar to SMD, as the filters can learn to ignore the image input.

In this case, the only difference between JSD and SMD is that they use different techniques to derive saliency.

However, JSD differs from SPD, as CNN filters cannot multiply one channel by another.

We follow the evaluation protocol of Zhang et al. (2018) and train each defense to detect adversarial images generated by a specific attack, thereby generating six different detection models (one for each black-box attack).

To generate the training data, we generate one adversarial example for every clean image.

The training data becomes [X, X adv ], where X denotes the clean data and X adv denotes the adversarial data, and the labels are [1 n , 0 n ], 1 n and 0 n are one-and zero-vectors of length n, respectively.

We use the same training procedure and models, as summarized in Appendix A, and report the accuracy of the classifiers on the test dataset.

We compare the performance of the models on MNIST, CIFAR-10 and ASSIRA (see Burges & Cortes, 1998; Krizhevsky et al., 2009; Elson et al., 2007, respectively) .

In addition to the two frequently used benchmarks, we consider the ASSIRA cats and dogs dataset 7 as it contains highquality images but is less computationally expensive than ImageNet.

8 Further details on the datasets can be found in Appendix A.

Many defenses hold up against black-box attacks but often are unable to defend against white-box attacks (Carlini & Wagner, 2017a) .

For this reason, we generate white-box attacks tailored to the defense strategy.

Our white-box attacks are iterative gradient-based attacks, which target both the classifier and the defense.

Inspired by FGSM, we can target the classifier f as

and the defense d as

where Clip clips the pixels to the pre-defined maximum range.

Using the above idea, we iterate between Equations 3 and 4 to generate the white-box attack for ID (the defense based on image classification).

We propose similar white-box attacks for the other defenses, as shown in Appendix C. We limit the number of iterations T to 5, as we find it to be adequate to generate a sufficiently strong attack and further increasing T does not improve the performance.

Our method is similar to that of Metzen et al. (2017) .

They propose finding adversarial examples as:

where in our case ?? = 0.5.

The key difference is that we iterate between Equations 3 and 4, rather than applying 3 and 4 simultaneously.

We find that this is more effective at targeting the defense, which is more difficult to fool than the original classifier.

We start by assessing the shift in saliency maps generated by adversarial perturbations and then present the efficacy of the detector against different adversarial attacks.

Details, such as attack success rate, can be found in Appendix B.

We start by quantifying the shift in saliency maps due to adversarial perturbations; we compute the L 2 distance between saliency maps of a natural image and its corresponding adversarial image.

As a baseline, we compare these values with the L 2 distance between two different natural images.

These statistics are summarized in Table 5 .

For CIFAR-10 and ASSIRA, the L 2 -norm between the saliency maps of a natural image and its corresponding adversarial image is comparable to or larger than the L 2 distance between two different natural images.

Using a Mann-Whitney U-test, we prove quantitatively that the shift is significant for most adversarial attacks on CIFAR-10 and ASSIRA images.

This suggests that our saliency-based method is an effective way of capturing adversarial perturbations.

Table 1 : L 2 distance between (1) saliency maps of different images (row labelled Different Images) and (2) the saliency maps of natural images and the adversarial image (generated by the type of attack specified in the row).

The entries correspond to MNIST/CIFAR-10/ASSIRA.

The p-value is derived using the Mann Whitney U-test, where we test whether the sample of L 2 distances between a natural and adversarial image is from the same distribution as different images.

We use a nonparametric test to avoid assuming normality of the data.

Figure 2 summarizes the performance of the defense models trained on a single adversarial attack on different adversarial attacks; the values and standard deviations can found in Appendix G. The overall performance of the model-based saliency defense suggests that saliency can be used to determine whether an image is adversarial.

Salient Pixel Defense outperforms Saliency Map Defense Overall, SPD (shown in blue) outperforms the other defenses, suggesting that the salient pixels provide useful information to the detector.

Further, our defense generalizes well: even when trained on a weaker attack, SPD is able to detect stronger attacks.

Both baseline methods, ID and JSD, only generalize well when trained on a stronger attack.

When trained on a weaker attack, they are not able to detect stronger adversarial attacks.

Worse generalization on JSMA We observe a drop in performance of the models when detecting JSMA, likely because JSMA is an L 0 -norm attack, which generates a different type of adversarial examples.

This may suggest that defenses trained on a specific norm, only generalize well to other attacks generated by a norm that produces similar perturbations.

FGSM, BIM, and MIM are L ??? -norm attacks, and C &W and DF are L 2 -norm attacks.

Both generate perturbations that are spread out over the entire image, contrary to L 0 norm attacks, which changes a few pixels using larger perturbations.

Threshold Behavior Both ID and JSD exhibit threshold behavior: they are unable to detect adversarial examples if the perturbation size is below a given threshold.

For example, see the performance of both defenses on the ASSIRA dataset.

There is a strong correlation between detection accuracy and perturbation size, as measured by L ??? and L 2 (see Table 2 ).

ID is able to detect all adversarial images for which the perturbation size is either L 2 > 0.027 or L ??? > 0.50, such as FGSM and JSMA.

10 However, the perturbations are much smaller for DF and CW, making these attacks harder to detect.

The threshold appears to occur around L 2 = 0.025, as ID can sometimes detect the FGSM perturbations, generated with this size.

This observation is in line with the results of Gong et al. (2017) , who find that ID is highly efficient at detecting adversarial images with perturbations of ?? ??? 0.03 but unable to detect adversarial perturbations generated using ?? = 0.01 (using FGSM for images scaled between 0 and 1), obtaining an accuracy of 50.0% in the latter case.

11 9 We observe that JSD performs similarly, although sometimes worse, compared to ID.

Theoretically, the parameter space of ID is a subset of the parameters of JSD.

The additional input (the Jacobian) makes the model more difficult to train.

Therefore, the difference in results can be attributed to training: the model is more difficult to train due to the increased number of parameters and does not learn to ignore the additional input.

10 FGSM is known to generate large perturbations.

The perturbations for JSMA are relatively large as the attack minimizes the L0 norm, thereby perturbing as few pixels as possible, but by a large amount.

11 Our perturbations for FGSM are larger than 0.01 to ensure that FGSM is sufficiently strong (see Appendix B for a summary of the attack success rates).

Table 3 summarizes the performance of different defenses against our white-box attack.

Our whitebox methods are highly effective in fooling the classifier as well as the defenses for MNIST and ASSIRA, as shown by the before adversarial training results.

The white-box attack is unable to fool the detector for CIFAR-10 successfully.

Next, we perform adversarial training: we iteratively train the detectors against the white-box attack and allow the white-box attack access to the new defense.

The white-box attack no longer successfully defeats SPD, which becomes more robust against the attack, whereas SMD is not able to become robust against the white-box attack.

In our analysis, we ascertain that the saliency maps of adversarial images differ from those of natural images.

Further, we show that salient pixel based defenses perform better than a saliency map defense.

When trained on a single black-box attack, our method is able to detect adversarial perturbations generated by different and stronger attacks.

We show that gradients are unable to capture shifts in saliency due to adversarial perturbations and present an alternative adversarial defense using learnt saliency models that is effective against both black-box and white-box attacks.

Building on the work of Gong et al. (2017) , we further establish the notion of threshold behavior, showing that the trend depends on the L 2 and L ??? -norms of the perturbations and therefore also prevails when using other methods (JSD) and across different attacks.

Future work could further investigate the performance of the defense in different applications.

For example, as our method runs in real-time, it could be used to detect adversarial perturbations in video to counter recent attacks Jiang et al., 2019) .

A ARCHITECTURES, HYPER-PARAMETERS AND DATA Figure 3 : ASSIRA, CIFAR-10, and MNIST image classifier architecture and hyper-parameters.

The first entry corresponds to the first layer, and the table proceeds chronologically until the last layer.

Parameters f, k, p, s and n represent the number of filters, kernel size, pooling size, stride, number of filters, respectively.

If stride is omitted, it is set to 1.

All classifiers have a final softmax activation.

We apply drop-out before every dense layer.

Using a validation set, we experimented with different drop-out rates between 0.3 and 0.7 and found that the rate ?? = 0.6 was optimal.

We use a ReLu activation for the penultimates layers and a softmax activation for the final layer.

We train the model for 10 epochs on batches of size 50.

We compare the performance of the models on MNIST, CIFAR-10 and ASSIRA (see Burges & Cortes, 1998; Krizhevsky et al., 2009; Elson et al., 2007, respectively) .

For MNIST and CIFAR-10, we use the standard train and test splits, and for ASSIRA, we use 3, 000 images.

We use 10% of the training data for the validation set, and re-train on the full training dataset once hyper-parameters were selected.

Further experimentation of ID and JSD architecture We further experiment with the architectures of ID and JSD to determine whether the observed performance was the result of the architecture.

In particular, we considered the adjustments as summarized in Table A ; however, we found that the changes did not improve performance.

In this section, we present the black-box adversarial attack hyper-parameters (see Figure 4) , the success rates of the different adversarial attacks (see Table 4 ) and an example of an adversarial image generated by the various black-box attacks (see Figure 5 ).

et al., 2016a) are used.

?? is the maximum perturbation allowed and ?? i is the maximum perturbation allowed in an iteration.

We use different hyperparameters for the MNIST and CIFAR-10 to ensure the attack is sufficiently strong.

MNIST and CIFAR-10 Figure 5 : Example of an Adversarial Image for the MNIST dataset.

From top to bottom: the top row is the set of images; the bottom row shows the size of the noise added.

Gray indicates no change, whereas white indicates that the image has been made lighter, and black indicates that the image has been made darker.

As MNIST images are gray-scale low-resolution images, the adversarial perturbations are perceptible to the human eye.

Nevertheless, the correct classification of the image is still clearly 4.

However, the classifier predicts (from left to right) the images as: 4, 9, 9 , 8, 9, 9.

Further, we observe that the perturbations of FGSM, BIM, and MIM are more visible than those of C & W and DF.

Algorithm 1 White-box attack for JSD 1: x adv ??? x 2: for t = 0 : T do 3:

for j = 1 : n do

if x adv does not fool the classifier then 5:

x adv ??? x adv + ??sign(???f (x adv , y)) Algorithm 1 provides the white-box attack for JSD.

As mentioned in Section 2, JSD concatenates the image with its saliency map (computed as the Jacobian) and uses this as an input to the classifier.

Algorithms 2 and 3 provide the white-box attacks for our defenses: SPD and SMD.

The function f s corresponds to generating the saliency map using the method introduced by Dabkowski & Gal (2017) .

Their method returns a two-dimensional saliency map.

However, as the image is three dimensions, we expand the last dimension and stack the map to match the number of channels (n c ) of the image.

In doing so, we assume that the saliency is constant along depth.

Algorithm 2 White-box attack for SPD 1: x adv ??? x 2: for j = 1 : n do 3:

for t = 1 : T do

if x adv does not fool the classifier then 5:

end if 7:

if sp(x adv ) does not fool the detector then 10:

if n c > 1, repeat r along the last dimension until it matches n c 12:

x adv ??? Clip(x adv + r)

if s adv does not fool the detector then 9:

r ??? ??sign(???d(s adv , y)) 10:

if n c > 1, repeat r along the last dimension until it matches n c 11: The first row shows the natural and adversarial images, and the second row shows their respective saliency maps.

There are no perceptible differences between the saliency map of the original image and adversarial images generated using MIM, C&W, and DF.

Further, we observe that for FGSM and JSMA, the gradients are all zero-valued.

This is a second drawback of using gradients-they are unstable and generate uninformative saliency maps due to underflow.

E L 2 DISTANCES BETWEEN SALIENCY MAPS CORRESPONDING TO ADVERSARIAL IMAGES GENERATED BY DIFFERENT ATTACKS F SINGLE BLACK-BOX ADVERSARIAL ATTACK DETECTOR Table 6 summarizes the accuracies of the different defenses when training a single detector against a combination of different types of black-box attacks.

All methods perform relatively similarly as when trained against a single defense, obtaining slightly worse performances than when trained against a specific adversarial attack.

This is useful in practice when it is unclear which adversarial attack is used.

<|TLDR|>

@highlight

We show that gradients are unable to capture shifts in saliency due to adversarial perturbations and present an alternative adversarial defense using learnt saliency models that is effective against both black-box and white-box attacks.
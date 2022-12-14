The Deep Image Prior (DIP, Ulyanov et al., 2017) is a fascinating recent approach for recovering images which appear natural, yet is not fully understood.

This work aims at shedding some further light on this approach by investigating the properties of the early outputs of the DIP.

First, we show that these early iterations demonstrate invariance to adversarial perturbations by classifying progressive DIP outputs and using a novel saliency map approach.

Next we explore using DIP as a defence against adversaries, showing good potential.

Finally, we examine the adversarial invariancy of the early DIP outputs, and hypothesize that these outputs may remove non-robust image features.

By comparing classification confidence values we show some evidence confirming this hypothesis.

1 Introduction Ulyanov et al. (2017) surprisingly showed that just the structure of a convolutional neural network is capable of capturing a good portion of images' statistics.

They demonstrated that starting with an untrained network, and then training to guide the output towards a specific target image for image restoration tasks such as denoising, super-resolution, and in-painting achieved performance which is comparable to state-of-the-art approaches.

Their approach, termed the Deep Image Prior (DIP), shows that the architecture of a network can act as a powerful prior.

The same network has been found to have excellent performance as a natural image prior (Ulyanov et al., 2017) .

The ability to detect natural images poses great significance in recent years, especially with the increasing security concerns raised over natural-looking images that are not correctly classified, called adversarial examples (Szegedy et al., 2013) .

These adversarial examples can be thought of as incremental, non-natural perturbations.

As such, using the Deep Image Prior as a recovery method can indicate its ability to work as a natural denoiser, a hypothesis that will initially be tested.

Furthermore, we use the Deep Image Prior to develop an adversarial defence, thereby investigating its potential.

Then, we investigate the early iterations of the network by producing saliency maps of the Deep Image Prior outputs (DIP outputs).

Saliency maps show the pixels which are most salient (relevant) in reaching a target classification.

We hope to show that the salient pixels gather to display more clear, distinct, and robust features of the images.

Recently, Ilyas et al. (2019) showed that adversarial examples are a result of the existence of nonrobust features in the images, which are highly predictive, yet incomprehensible to humans.

The successful performance of the Deep Image Prior in recovering the original classes from adversarial examples (Ilyas et al., 2017) , raises the argument that the Deep Image Prior produces images that have 'dropped' their non-robust features and are left with the robust image features.

To test this theory we directly use the dataset from Ilyas et al. (2019) consisting of robust and non-robust image features, and passing these through the Deep Image Prior.

By comparing the DIP outputs of the robust and non-robust image datasets, we hope to see evidence towards the ability of the Deep Image Prior to select robust images.

33rd Conference on Neural Information Processing Systems (NeurIPS 2019), Vancouver, Canada.

As a prerequisite we first show the ability of the Deep Image Prior to recover from adversarial perturbations.

For this investigation, three methods for generating adversarial examples will be considered: the Fast Gradient-Sign Method (FGSM) (Goodfellow et al., 2015) , the Basic Iterative method (BI), and the Least-Likely Class Iterative method (LLCI) (Kurakin et al., 2016) .

Adversarial examples were generated for 20 images using the three methods for various adversarial strengths.

The DIP output was collected and classified every 100 iterations and the values of the true class confidence were obtained.

The classifier used was ResNet18 (He et al., 2015a) .

The results are shown in Figure 1 .

More accurate classifiers were also briefly considered, such as Inception-v3 , but no qualitative differences were found.

It is apparent that the DIP outputs in the earlier iterations allow the classifier to recover the true class, as evident from the peaks in the confidence values.

Then, the network output converges back to the adversary observed through the decrease in confidence values.

The number of iterations at which peak confidence occurs appears to be different among adversaries and also among adversarial strengths.

Nevertheless the theme is similar; exhibiting peaks at the earlier DIP iterations, showing the ability of the network to recover from adversarial perturbations.

We have shown that the DIP can be an interesting and effective tool for recovering from adversarial perturbations.

However, details about the iterative process and the transformation of the input into the various DIP outputs are still unknown.

To test the nature of these outputs we introduce a novel saliency map approach, termed MIG-SG.

This method is a variation of the integrated gradients approach (Sundararajan et al., 2017) while using SmoothGrad (Smilkov et al., 2017) .

More information about this method can be found in the Appendix.

Figure 2 shows, on a step-by-step basis, how the salient features of the image change for progressive DIP outputs.

While the image is very blurry after 300-400 iterations, the saliency map already shows that all the key features of the knife have already been identified.

This is confirmed by the confidence of the DIP output, which increased to > 0:5 after just 200 iterations.

On the contrary, observing the outputs at 2000 and 4000 iterations shows that salient features have become more focused on the blade of the knife.

Previously, the salient features focused on the handle and the butt of the blade as observed from the bottom row of images in Figure 2 .

Furthermore, it is no longer clear what the salient features represent, a fact also illustrated in the decreasing value of the true class confidence.

Overall, the salient maps "lost" various salient features as the DIP output was converging towards the adversarial example.

Overall, with the clearest saliency maps observed at low iteration numbers (300-400), we observe evidence that the early DIP outputs are invariant towards adversarial effects.

To mount a defence using the Deep Image Prior we aim to transform the input of the classifier to a state where the adversarial perturbations are not detectable.

The classifier used for this investigation was ResNet18 (He et al., 2015a) .

By using a simple classifier to make this defence, we are able to evaluate the potential of the Deep Image Prior to recover the original class from adversarial perturbations individually.

Our results are compared against the defence from Xie et al. (2017) that uses randomisation to defend against adversarial perturbations, and which also uses a similar evaluation method.

Understandably, using the Deep Image Prior decreases the accuracy of the network on clean images.

From Table 1 , there is a noticeable decrease in top-1 accuracy, especially when using fewer DIP iterations.

As the number of iterations is increased, the top-1 accuracy increases with it, at a loss of computational speed.

Since the computational costs are already very high, the defence was not tested for larger iteration numbers, as that would make it slow and impractical.

The results of using the Deep Image Prior on adversarial examples are shown in Table 2 and display a very competitive performance with the reference defence method, having a higher accuracy across all three adversaries used in that comparison.

The average accuracy is highest after 1000 iterations, however, this is not best for all the adversaries as observed from the FGSM adversary with = 10.

Overall, we see a decreased ability to correctly classify images, combined with an increased ability to defend against adversaries.

This result, is similar to the one described by Ilyas et al. (2019) , where the classifier trained on the robust image dataset, highlighting the ability of the Deep Image Prior to select these robust features.

5 Using the robust image dataset

For this test, We used the pre-generated robust and non-robust image datasets from Ilyas et al. (2019) .

The architecture used for the Deep Image Prior had to be altered since CIFAR-10 images were used.

Details can be found in the Appendix.

The outputs were evaluated through the classification confidence of the original class of the image.

Both figures in 3 show the difference between the classification of robust and non-robust datasets to an external classifier, where robust images hold more information about the true class compared to the non-robust images.

Regarding the response at the earlier iteration numbers, it is very subtle, yet we see some evidence to support our hypothesis.

The non-robust image datasets show a trough before converging to their final classification confidence, while the robust image datasets shows a peak in confidence, indicating that the the convergence of the network towards the robust images was faster than the convergence on the non-robust ones.

6 Discussion on architecture & Conclusions Ulyanov et al. (2017) showed that the DIP achieved remarkable results, apparently due to the prior encoded in the network architecture.

To test this, we evaluated the sensitivity of DIP to changes in network architecture.

Surprisingly, we found that while some sensitivity exists, it is not high, with various architecture changes showing little to no changes in performance.

However, some changes showed great influence on the performance of the network.

In particular, the network was found to fail when no skip connections were used, or when a very shallow network was used.

Nevertheless, no evidence was found that can describe this sensitivity as a "resonance", as stated in the original paper.

See Appendix for details.

We observed the network's ability to successfully recover from adversarial perturbations, caused by the resistance of the early DIP outputs to adversarial perturbations.

This was further observed from looking at appropriate saliency maps, where we introduced a new method.

Consequently, the network was found to create a promising adversarial defence.

Lastly, we provided evidence for the ability of the Deep Image Prior to select robust image features over non-robust features in its early iterations, as defined by Ilyas et al. (2019) .

As the name suggests, this method performs integration between a baseline and our image, numerically, by calculating the forward derivative of the network.

SmoothGrad calculates this forward derivative, by performing the differentiation on a noisy version of the input, and averaging the derivative over multiple samples (Smilkov et al., 2017) .

As a result, combining the two methods appears to yield significantly improved saliency maps.

Since we are performing integration, solely taking the absolute value of the result of the grad function, failed to produce results.

However, a small modification was made to the algorithm in an attempt to stop the method from failing.

By also taking the absolute value of the final result, the method produced very promising results.

Using the absolute values of the gradients for coloured images, enables negative gradients to also be contribute to the salient features of the image.

Mathematically our saliency method can be expressed as:

where x is the input image, x0 is the baseline image and is only used for comparative purposes (Sundararajan et al., 2017) .

Additionally, m is the number of integration steps and N is the number of samples used in the computation of SmoothGrad (Smilkov et al., 2017) .

Lastly, Ft(x) returns the classification result of class t before the cost function is calculated.

Common saliency maps have been generated for a panda image, shown in Figure 4 .

The MIG-SG saliency map is observed in Figure 4e and while it can definitely appear as a scary image, it provides very interesting information about the panda.

This saliency map, instead of picking up all the panda in the image, has instead focused on its characteristic features, the eyes and the mouth.

This makes it a very useful tool to visually interpret images.

For the Deep Image Prior, the original architecture was used (Ulyanov et al., 2017) .

The number of iterations was left at a low value, as the results of this work suggested that the DIP output is less sensitive to adversarial perturbations at earlier iterations.

Three iteration numbers were tested: 500, 750 and 1000.

The tests were conducted on a dataset of images from 200 randomly selected classes from the ImageNet database.

From this dataset, 500 images correctly classified using the ResNet18 classifier were then chosen to test the performance of our defence.

The diagram of the defence is shown in Figure 5 .

Two architectures were considered, the first is the one used in the original paper of the Deep Image Prior (Ulyanov et al., 2017) but with the encoder depth changed from 5 to 3, to allow for the decreased dimensionality of the CIFAR-10 images.

The second architecture, uses only 16 feature maps in each layer, compared with the original number which was 128, while also the encoder depth was kept at 3.

Architecture-1 and architecture-2 can be found in Tables 3 and 4 respectively.

<|TLDR|>

@highlight

We investigate properties of the recently introduced Deep Image Prior (Ulyanov et al, 2017)
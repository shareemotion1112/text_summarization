The checkerboard phenomenon is one of the well-known visual artifacts in the computer vision field.

The origins and solutions of checkerboard artifacts in the pixel space have been studied for a long time, but their effects on the gradient space have rarely been investigated.

In this paper, we revisit the checkerboard artifacts in the gradient space which turn out to be the weak point of a network architecture.

We explore image-agnostic property of gradient checkerboard artifacts and propose a simple yet effective defense method by utilizing the artifacts.

We introduce our defense module, dubbed Artificial Checkerboard Enhancer (ACE), which induces adversarial attacks on designated pixels.

This enables the model to deflect attacks by shifting only a single pixel in the image with a remarkable defense rate.

We provide extensive experiments to support the effectiveness of our work for various attack scenarios using state-of-the-art attack methods.

Furthermore, we show that ACE is even applicable to large-scale datasets including ImageNet dataset and can be easily transferred to various pretrained networks.

The checkerboard phenomenon is one of the well-known artifacts that arise in various applications such as image super resolution, generation and segmentation BID29 BID27 .

In general, the checkerboard artifacts indicate an uneven pattern on the output of deep neural networks (DNNs) that occurs during the feed-forward step.

BID21 have investigated in-depth the origin of the phenomenon that the artifacts come from the uneven overlap of the deconvolution operations (i.e., transposed convolution, ) on pixels.

Its solutions have been suggested in various studies BID35 BID0 .Interestingly, a possible relationship between the checkerboard artifacts and the robustness of neural networks has been noted by BID21 but not has it been seriously investigated.

Moreover, while the previous works BID16 BID20 BID21 BID8 have concentrated on the artifacts in the pixel space, studies have been rare on the artifacts in the gradient space that occur during the backward pass of the convolution operation.

To show that the gradient checkerboard artifacts phenomenon is crucial for investigating the network robustness and is indeed a weak point of a neural network, we focus on analyzing its effects on the gradient space in terms of adversarial attack and defense.

By explicitly visualizing the gradients, we demonstrate that the phenomenon is inherent in many contemporary network architectures such as ResNet BID12 , which use strided convolutions with uneven overlap.

It turns out that the gradient checkerboard artifacts substantially influence the shape of the loss surface, and the effect is image-agnostic.

Based on the analysis, we propose an Artificial Checkerboard Enhancer module, dubbed ACE.

This module further boosts or creates the checkerboard artifacts in the target network and manipulates the gradients to be caged in the designated area.

Because ACE guides the attacks to the intended environment, the defender can easily dodge the attack by shifting a single pixel (Figure 1 ) with a negligible accuracy loss.

Moreover, we demonstrate that our module is scalable to large-scale datasets such as ImageNet BID6 ) and also transferable to other models in a plug and play fashion without additional fine-tuning of pretrained networks.

Therefore, our module is highly practical in general scenarios in that we can easily plug any pretrained ACE module into the target architecture.

Base Network (e.g. ResNet, VGG) ACE Module

The first layer of the ACE module has kernel size 1 and stride 2 Gradient Checkerboard Artifact (GCA)

Generate adversarial image from the model's enhanced gradient checkerboard artifact Adv Image GCA

Induce attack on GCA Zero-pad a single row/column Remove opposite row/column Adv Image Input ImageFigure 1: Defense procedure using the proposed Artificial Checkerboard Enhancer (ACE) module.

ACE shapes the gradient into a checkerboard pattern, thus attracting adversarial attacks to the checkerboard artifacts.

Since the defender is aware of the guided location in advance, adversarial attacks can be easily deflected during inference by padding the image with a single row/column and discarding the opposite row/column.

Our contributions are summarized as three-fold:Analysis of gradient checkerboard artifacts.

We investigate the gradient checkerboard artifacts in depth which are inherent in many of the contemporary network architectures with the uneven overlap of convolutions.

To the best of our knowledge, this is the first attempt to analyze the artifacts of the gradient space in terms of network robustness.

We empirically show that the gradient checkerboard artifacts incur vulnerability to the network.

Artificial Checkerboard Enhancer (ACE).

We introduce ACE module that strengthens gradient artifacts and induces adversarial attacks to our intended spaces.

After guiding the attacks to the pre-specified area using ACE, we deflect adversarial attacks by one-pixel padding.

Our extensive experimental results support that our proposed defense mechanism using ACE module successfully defends various adversarial attacks on CIFAR-10 (Krizhevsky, 2009) and ImageNet BID6 datasets.

Scalability.

We show that ACE is readily transferable to any pretrained model without fine-tuning, which makes the module scalable to a large-scale dataset.

To the best of our knowledge, this is the first defense method that attempts and succeeds to defend the attacks with the projected gradient descent algorithm (PGD) BID17 BID1 on ImageNet dataset.

Adversarial attacks can be conducted in various ways depending on how much the adversary, also known as the threat model, has access to the target model.

If the attacker can acquire the gradients of the target, gradient-based attacks can be performed, which are usually very effective because iterative optimization becomes possible BID10 BID14 MoosaviDezfooli et al., 2016; BID22 BID5 .

Score-based attacks can be valid when the adversary can use the logits or the predicted probabilities for generating adversarial examples BID32 .

If generated adversarial examples are not from the target model, we call this transfer-based attack .

Recently, a new type of attack, called a decisionbased attack, has been introduced where the adversary only has knowledge about the final decision of the model (e.g., top-1 class label) BID3 .According to , defense methods can be largely categorized into gradient masking and adversarial training.

Gradient masking methods usually make adversaries difficult to compute exact gradients and make it challenging to fool the target.

Gradient obfuscation, which is a recently introduced term of gradient masking by BID2 , includes specific gradient categories such as stochastic gradients, shattered gradients and vanishing/exploding gradients.

Works related to our defense method can be considered as input transformations which focus on the input image BID36 BID25 BID4 BID37 BID11 .On the other hand, adversarial training has been known to make models robust against adversarial perturbation BID17 BID19 BID33 , but there remains an issue that the robustness comes with the cost of accuracy BID34 BID31 .

Moreover, according to BID28 , restricting on l ∞ bounded perturbations during adversarial training has limited robustness to attacks with different distortion metrics.

We would like to define the following terms here and will use them without further explanation throughout this paper.

Gradient Overlap (Ω(x i )) represents the number of parameters associated with a single pixel in the input x. For more explanation on its calculation, see Appendix C. We define the set of pixels whose gradient overlap is in the top p fraction as G(p) (e.g., G(1.0) represents the entire pixel set).

Gradient Checkerboard Artifacts (GCA) is a phenomenon which shows checkerboard patterns in gradients.

The existence of GCA has been introduced in BID21 , although not has it been thoroughly examined.

GCA occurs when a model uses a convolution operation of kernel size k that is not divisible by its stride s.

We first introduce a simple experiment that motivated the design of our proposed ACE module.

We conduct an experiment to visualize the attack success rate of a given image set using a single pixel perturbation attack.

For each pixel of an input image, we perturb the pixel to white (i.e., a pixel with RGB value of (255, 255, 255) ).

Next, we use our toy model based on LeNet (LeCun et al., 1998) (see Appendix A for the detailed architecture) and ResNet-18 BID12 to measure the attack success rate on the test images in CIFAR-10 dataset.

Note that the attack success rate P attack in Figure 2 is defined as the average number of successful attacks per pixel on the entire set of test images.

As we can see in Figure 2a and Figure 2c , checkerboard patterns in the attack success rate are clearly observable.

This pattern can be considered as image-agnostic because it is the result of the average over the entire set of the test images of CIFAR-10 dataset.

Then a natural question arises: What is the cause of this image-agnostic phenomenon?

We speculate that the uneven gradient overlap is the cause of this phenomenon, which is directly associated with the number of parameters that are connected to a single pixel.

As depicted in Figure 2b and Figure 2d , we can observe checkerboard patterns in the gradient overlap.

In fact, this uneven overlap turns out to be substantially susceptible to the adversarial attacks.

We will provide the supporting results on this in the following sections.(a) Pattack of Toy model DISPLAYFORM0 Figure 2: Illustration of the attack success rate P attack and the gradient overlap Ω(x i ) of toy model and ResNet-18.

The illustrated gradient overlap of ResNet-18 comes from the features after the fifth convolutional layer.

Attack success rate (a) and (c) are computed by perturbing each pixel of an image to white (i.e., (255, 255, 255) ), over the entire set of test images in CIFAR-10 dataset.

Note that higher probability at a pixel denotes a higher success rate when it is attacked.

We can observe patterns in P attack aligned to our gradient overlap on (b) toy model and (d) ResNet-18.

Table 1 : Top-1 test accuracy (%) after performing various adversarial attacks on every pixel G(p = 1.0), its subset G(p = 0.3), and their differences (i.e., diff) on CIFAR-10 dataset.

The toy model (see Appendix A) and ResNet-18 achieved 81.4% and 94.6% top-1 test accuracy, respectively.

Note that all the diffs are close to zero.

ResNet-18 Attack Methods DISPLAYFORM0 OnePixel BID32 56.6 58.4 1.7 57.2 59.5 2.4 JSMA BID22 0.2 0.4 0.2 3.2 9.8 6.6 DeepFool BID18 18.5 18.6 0.1 7.2 11.5 4.3 CW BID5 0.0 0.0 0.0 0.0 0.0 0.0 PGD BID17 0.0 1.6 1.6 0.0 0.0 0.0

To show that the pixels with high gradient overlaps are indeed a weak point of network, we generate adversarial examples on G(p).

We evaluate top-1 accuracy of our toy model (the model defined as in the previous subsection) and ResNet-18 on CIFAR-10 dataset after performing five adversarial attacks BID32 BID22 BID18 BID5 BID17 for p ∈ {1.0, 0.3} (Table 1) .

Interestingly, constraining the domain of the attacks to G(0.3) barely decreases the success rate compared to the attacks on G(1.0).We can observe that the pixels with the high gradient overlaps are more susceptible (i.e., likely to be in a vulnerable domain) to the adversarial attacks.

Considering all the observations, we leverage the vulnerable domain of the pixels for adversarial defense.

If we can intentionally impose the GCA onto a model input and let GCA occupy the vulnerable domain, we can fully induce the attacks on it so that the induced attacks can be dodged easily by a single padding operation.

In this section, we propose the Artificial Checkerboard Enhancer (ACE) module, which artificially enhances the checkerboard pattern in the input gradients so that it induces the vulnerable domain to have the identical pattern.

Figure 3a illustrates our proposed ACE module, which is based on a convolutional autoencoder.

The encoder consists of convolutional layers where the first layer's k is not divisible by s (k ≡ 0 mod s), for example, when k = 1 and s = 2.

In order to preserve the information of the input x, we add an identity skip connection that bypasses the input of ACE module to the output.

The hyperparameter λ is to control the magnitude of checkerboard artifacts in the input gradients.

We plug our ACE module in front of a base convolutional network to enhance the checkerboard artifacts on its input gradients.

By increasing λ, we can artificially increase the gradient checkerboard artifacts of the network.

Figure 3b and 3c show the heatmaps of the input gradients of ResNet-18 when λ = 10 and λ = 0 (i.e., without ACE module), respectively.

The heatmap is generated by a channel-wise absolute sum of input gradients.

Note that the checkerboard pattern is clearly observed in Figure 3b .By changing the value of λ, we report the top-1 test accuracy and the proportion of the pixels having checkerboard artifacts (C) in the top-30% gradient overlaps G(0.3) ( TAB0 ).

More precisely, we denote C as the GCA imposed by ACE module, which is identical to the set of pixels that are connected to its first convolutional layer of k = 1 and s = 2.

In TAB0 , we can observe that 1) there is only a small decrease in the accuracy even with a large λ and 2) the pixels with the large gradient overlaps gradually coincide with the GCA as the λ increases.

Furthermore, according to the results in Table 1 , the existing adversarial attacks tend to be induced on the pixels with the high gradient overlaps.

Therefore, we can conjecture that our ACE module which builds a high gradient overlap with a significant checkerboard pattern could cage the adversarial attacks into the checkerboard artifacts, and this will be empirically proven in Section 5.

We now study the effects of λ.

To this end, we first visualize the classified labels with respect to the magnitude of the perturbation on pixels in checkerboard artifacts C and pixels in non-checkerboard artifacts X\C. Let x be the input image andê C = DISPLAYFORM0 where M C denotes a mask (i.e., value of i-th element of M C equals to one if x i ∈ C and zero otherwise) and denotes the element-wise multiplication.

We defineê X\C in a similar way.

We plot the classified label map of x +īê X\C +jê C by varyingī andj.

For the experiment, we first train ACE module as an autoencoder using ImageNet BID6 ) datasets and plug ACE module into pretrained ResNet-152 as described in the following experiment section.

Next, we plot classified labels for a sample image from ImageNet by varyingī andj from −100 to 100 by interval 1 (i.e., we test 40k perturbed images per each image) in FIG1 .

The figure signifies the classified label map with respect to the perturbation on X\C and C. Without using our ACE module (when λ = 0), the perturbation through non-artifact pixels and artifact pixels similarly affect the label map.

However, when λ > 0, we can observe that the artifact pixels are susceptible to change their labels with only a small perturbation while non-artifact pixels are robust to the same perturbation.

Note that the asymmetry between the artifact pixels and non-artifacts becomes more clear as λ increases.

Here, we propose a novel defense method using ACE module.

First, we plug ACE module into the input of a given network which enhances gradient checkerboard artifacts.

Next, we let the adversary generate adversarial images using the network with ACE module.

Because ACE module is likely to expose the pixels in the vulnerable domain, which is empirically shown in Appendix I, we may consider the ACE module as an inducer to the adversarial perturbations generated by the adversary into the checkerboard.

Interestingly, because the pixels in the vulnerable domain are going to be aligned to the repeated checkerboard pattern, by shifting a single pixel of the adversarial sample, we can move perturbations into the non-vulnerable domain (i.e., non-artifact pixels).

The proposed defense mechanism is similar to the defense method introduced in BID36 .

However, thanks to our ACE module, the vulnerable pixels are induced to checkerboard artifacts so that only onepixel padding is enough to avoid several adversarial attacks aiming the pixels in the vulnerable domain.

We also report the defense results regarding the diverse padding-sizes in Appendix D. It is worthwhile to note that large λ induces significant gradient checkerboard artifacts hence leads to more robust defense.

The detailed results are reported in Section 5.

For thorough evaluation, we evaluate our proposed defense method in the following three attack scenarios, which are vanilla, transfer, and adaptive attacks.

First, in the vanilla attack scenario, the adversary has access to the target model but not our proposed defense method (i.e., single pixel padding defense).

We remark that the vanilla attack scenario is similar to the scenario used by BID36 .

Second, in the transfer attack scenario, the adversary generates adversarial perturbations from a source model, which is different from the target model.

Finally, in the adaptive attack scenario, the adversary knows every aspect of the model and the defense method so that it can directly exploit our defense.

For our experiments, Expectation Over Transformation (EOT) BID2 ) is used for the adaptive attack scenario.

For the evaluation of our defense method, we use the following five attack methods as our adversary.

OnePixel BID32 , JSMA BID22 , DeepFool (Moosavi-Dezfooli et al., 2016), CW BID5 and PGD BID17 1 .

In addition, we conduct experiments on CIFAR-10 (Krizhevsky, 2009) and ImageNet BID6 ) datasets for the attack scenarios.

For the models evaluated in CIFAR-10 dataset, we train the models with the two layered ACE module from scratch.

Note that the first convolutional layer in ACE has k = 1 and s = 2, and the following deconvolutional layer has k = 3 and s = 2 so that it enhances gradient checkerboard artifacts.

We would like to recall that the top-1 accuracy of VGG-11 BID30 and ResNet-18 BID12 with ACE module in respect of different λ are reported in TAB0 .

Meanwhile, for a large-scale dataset such as ImageNet, training an entire network is very expensive.

Hence, we train the ACE module as autoencoder with UNet architecture BID26 and plug ACE into the input of a pretrained network without any additional training procedure.

In order to retain the scale of the input image of the pretrained network, we slightly modify the ACE module by constraining λ ∈ [0, 1] and multiplying (1 − λ) to the identity skip connection.

In this way, our ACE module becomes capable of handling large-scale datasets in a plug and play fashion on any model.

We now introduce two evaluation metrics named attack survival rate and defense success rate which denote that top-1 accuracy after attack divided by the original top-1 accuracy of the model Table 3 : Attack survival rate (%) and defense success rate (%) (larger is better) by one-pixel padding defense on CIFAR-10 dataset with varying λ.

Note that λ = 0 is the equivalent setting to single padding pixel experiments in BID36 Table 4 : Attack survival rate (%) and defense success rate (%) (larger is better) by one-pixel padding defense on ImageNet dataset with varying λ.

ResNet-18-ACE 0.0 0.0 10.2 23.9 32.1 93.5 98.3 VGG-11-ACE 0.8 1.7 1.9 9.7 85.9 79.5 98.8

and top-1 accuracy after defending attack divided by the original accuracy, respectively.

We note that all experimental results reported in this section are reproduced by ourselves 2 .Vanilla attack scenario.

We evaluate our defense method in both CIFAR-10 and ImageNet dataset.

For CIFAR-10 experiments, we train an entire network including ACE.

For ImageNet experiments, we only train ACE module as conventional autoencoder by minimizing the mean squared error without training an entire network.

Table 3 shows that our proposed method defends various attack methods successfully on CIFAR-10 dataset.

We remark that by choosing a large λ (e.g., λ = 100), the top-1 accuracy after defense on performed attacks is barely dropped.

In table 4, we report same experiments conducted on ImageNet dataset.

We use the pretrained models of VGG-19 and ResNet-152 3 repository whose top-1 test accuracy on ImageNet dataset is 72.38% and 78.31%, respectively.

We abbreviate the results of OnePixel, JSMA and DeepFool due to the infeasibly high cost of time and memory limit for those algorithms to craft numerous adversarial images in ImageNet dataset.

Comparison with other defense methods BID25 BID36 BID4 for CIFAR-10 and ImageNet datasets are reported in Appendix E. To investigate the effectiveness of λ regards to defense success rate, we evaluate PGD attack for λ ∈ {0, 1, 2, 5, 10, 20, 100} and report the defense success rates in Table 5 .

The result shows that : Defense success rate (%) on CIFAR-10 test dataset after one-pixel padding defense on transfer attacks from the source model to the target model via JSMA, CW and PGD.

The number followed by name of network denotes the intensity of λ, e.g., VGG-11-100 denotes VGG-11 + ACE with λ = 100.

Note that adversarial examples generated by different λ are not transferable to other models.when λ increases, the defense success rate improves as well.

To the best of our knowledge, this is the first work that defends PGD up to 98%.Transfer attack scenario.

It has been demonstrated that conventional deep networks are vulnerable to transfer attacks proposed by BID23 .

To show that our method is robust to transfer attacks, we conduct transfer attacks for VGG-11 and ResNet-18 by choosing λ ∈ {0, 10, 100} on CIFAR-10 dataset.

We report the defense success rate after one-pixel padding defense of transfer attacks by JSMA, CW and PGD in FIG2 .

More results including OnePixel and DeepFool transfer attack experiments are reported in Appendix G. The results show that generated adversarial samples are not transferable to other models with different λ.

Adaptive attack scenario.

A successful defense method should defend l 0 , l 2 , and even l ∞ bounded adversaries and also show robust results on the adaptive white-box setting.

We report our defense results combined with adversarial training against Expectation Over Transformation (EOT) BID2 ) of PGD attack in Appendix H. From the results, it turns out that our method is complementary with robust training defense methods (e.g., adversarial training).

Therefore, if we combine our method with the existing robust training defense methods together, we can secure promising results on the vanilla scenario and even perform well on the adaptive scenario.

In this paper, we have investigated the gradient checkerboard artifacts (GCA) which turned out to be a potential threat to the network robustness.

Based on our observations, we proposed a novel Artificial Checkerboard Enhancer (ACE) which attracts the attack to the pre-specified domain.

ACE is a module that can be plugged in front of any pretrained model with a negligible performance drop.

Exploiting its favorable characteristics, we provide a simple yet effective defense method that is even scalable to a large-scale dataset such as ImageNet.

Our extensive experiments show that the proposed method can deflect various attack scenarios with remarkable defense rates compared with several existing adversarial defense methods.

We present our toy model used in Section 3.

The model consists of three convolutional layers followed by ReLUs and two fully-connected layers (Table 6 ).

Table 6 : Architecture detail of our toy model used on CIFAR-10 dataset.

Input image x ∈ R 32×32×3 3×3, stride=2 conv16 ReLU 3×3, stride=2 conv32 ReLU 3×3, stride=1 conv64 ReLU dense 1600 → 512 dense 512 → 10

To visualize the existence of checkerboard artifacts in the gradients, here we present the average heatmaps over the test images.

Let ∇x l i,j,k denote the feature gradient in layer l where i, j and k are the indices of xy-axis and the channel, respectively.

In each layer, the heatmap h l is gathered by a channel-wise absolute sum of the feature gradients.

More specifically, h We can observe checkerboard patterns before strided convolutional layers (layer 5, 9, 13) and they propagate to their front layers (layer 4, 8).

To count the gradient overlaps per pixel (Ω(x i )), for simplicity, we only consider convolutional and fully-connected layers.

We set each and every network parameter to 1 to evenly distribute the gradients from the loss layer to the previous layers.

Therefore, the gradients of the modified network are aggregated in a certain location (or pixel) if and only if there is a linear overlap from the later layers at the backward pass.

In TAB2 , we report the defense success rate varying the padding-size from one to five.

We can observe that the proposed defense mechanism almost prevents accuracy drop with the padding-sizes of one, three and five (odd numbers).

The result supports that ACE induces adversarial perturbations into checkerboard artifacts, which could be avoided by padding an image with one pixel.

BID25 BID36 BID4 ) by following their papers.

For fair comparison, we follow the suggested settings in their papers, and the results are presented in TAB3 .

Specifically, for BID25 , with R-CAM implemented, the number of deflection is set to 100 with window-size of 10 and sigma for denoiser of 0.04, respectively.

For BID36 , the scale factor is set to 0.9.

Finally, for BID4 , the number of the ensemble is set to 1000 and the radius of the region is set to 0.02.

In this section, we plot the classified label map after pixel perturbation on artifact pixels and nonartifact pixels as described in Section 4.2.

We follow the same setting of Section 4.2.

The results are reported in FIG5 .

Adaptive attack case A successful defense method should be able to defend various conditions including l 0 , l 2 and l ∞ -bounded adversaries as well as an adaptive white-box setting where the adversary knows our defense method in every aspect.

Under the adaptive white-box setting, we conducted experiments in Table 10 .

In order to avoid direct exploitation of our padding direction, we shift our images in the random direction around the known safe points near our checkerboard artifacts.

By combining PGD adversarial training BID17 for robustness on l ∞ bounded attacks to our method, we can defend the corresponding adaptive attack for stochastic methods known as Expectation Over Transformation (EOT, BID1 ).

This method was used to break BID36 in BID2 .

Although we have some loss in Top-1 accuracy when λ is high, we have advantages in that we can defend vanilla attack cases at the same time.

Table 10 : Top-1 accuracy (%) after EOT attack on our defense method together with adversarial training on CIFAR-10 dataset.

All images were padded by one-pixel to X-axis.

Conducted attacks are written in the format of PGD-norm-iterations.

ACE module only shows training accuracy loss due to high λ.

<|TLDR|>

@highlight

We propose a novel aritificial checkerboard enhancer (ACE) module which guides attacks to a pre-specified pixel space and successfully defends it with a simple padding operation.
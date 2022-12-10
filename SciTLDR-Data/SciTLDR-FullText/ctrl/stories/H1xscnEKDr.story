We study the problem of defending deep neural network approaches for image classification from physically realizable attacks.

First, we demonstrate that the two most scalable and effective methods for learning robust models, adversarial training with PGD attacks and randomized smoothing, exhibit very limited effectiveness against three of the highest profile physical attacks.

Next, we propose a new abstract adversarial model, rectangular occlusion attacks, in which an adversary places a small adversarially crafted rectangle in an image, and develop two approaches for efficiently computing the resulting adversarial examples.

Finally, we demonstrate that adversarial training using our new attack yields image classification models that exhibit high robustness against the physically realizable attacks we study, offering the first effective generic defense against such attacks.

State-of-the-art effectiveness of deep neural networks has made it the technique of choice in a variety of fields, including computer vision (He et al., 2016) , natural language processing (Sutskever et al., 2014) , and speech recognition (Hinton et al., 2012) .

However, there have been a myriad of demonstrations showing that deep neural networks can be easily fooled by carefully perturbing pixels in an image through what have become known as adversarial example attacks (Szegedy et al., 2014; Goodfellow et al., 2015; Carlini & Wagner, 2017b; Vorobeychik & Kantarcioglu, 2018) .

In response, a large literature has emerged on defending deep neural networks against adversarial examples, typically either proposing techniques for learning more robust neural network models (Wong & Kolter, 2018; Wong et al., 2018; Raghunathan et al., 2018b; Cohen et al., 2019; Madry et al., 2018) , or by detecting adversarial inputs (Metzen et al., 2017; Xu et al., 2018) .

Particularly concerning, however, have been a number of demonstrations that implement adversarial perturbations directly in physical objects that are subsequently captured by a camera, and then fed through the deep neural network classifier (Boloor et al., 2019; Eykholt et al., 2018; Athalye et al., 2018b; Brown et al., 2018) .

Among the most significant of such physical attacks on deep neural networks are three that we specifically consider here: 1) the attack which fools face recognition by using adversarially designed eyeglass frames (Sharif et al., 2016) , 2) the attack which fools stop sign classification by adding adversarially crafted stickers (Eykholt et al., 2018) , and 3) the universal adversarial patch attack, which causes targeted misclassification of any object with the adversarially designed sticker (patch) (Brown et al., 2018) .

Oddly, while considerable attention has been devoted to defending against adversarial perturbation attacks in the digital space, there are no effective methods specifically to defend against such physical attacks.

Our first contribution is an empirical evaluation of the effectiveness of conventional approaches to robust ML against two physically realizable attacks: the eyeglass frame attack on face recognition (Sharif et al., 2016) and the sticker attack on stop signs (Eykholt et al., 2018) .

Specifically, we study the performance on adversarial training and randomized smoothing against these attacks, and show that both have limited effectiveness in this context (quite ineffective in some settings, and somewhat more effective, but still not highly robust, in others), despite showing moderate effectiveness against l ∞ and l 2 attacks, respectively.

Our second contribution is a novel abstract attack model which more directly captures the nature of common physically realizable attacks than the conventional l p -based models.

Specifically, we consider a simple class of rectangular occlusion attacks in which the attacker places a rectangular sticker onto an image, with both the location and the content of the sticker adversarially chosen.

We develop several algorithms for computing such adversarial occlusions, and use adversarial training to obtain neural network models that are robust to these.

We then experimentally demonstrate that our proposed approach is significantly more robust against physical attacks on deep neural networks than adversarial training and randomized smoothing methods that leverage l p -based attack models.

Related Work While many approaches for defending deep learning in vision applications have been proposed, robust learning methods have been particularly promising, since alternatives are often defeated soon after being proposed (Madry et al., 2018; Raghunathan et al., 2018a; Wong & Kolter, 2018; Vorobeychik & Kantarcioglu, 2018) .

The standard solution approach for this problem is an adaptation of Stochastic Gradient Descent (SGD) where gradients are either with respect to the loss at the optimal adversarial perturbation for each i (or approximation thereof, such as using heuristic local search (Goodfellow et al., 2015; Madry et al., 2018) or a convex over-approximation (Raghunathan et al., 2018b; Wang et al., 2018) ), or with respect to the dual of the convex relaxation of the attacker maximization problem (Raghunathan et al., 2018a; Wong & Kolter, 2018; Wong et al., 2018) .

Despite these advances, adversarial training a la Madry et al. (2018) remains the most practically effective method for hardening neural networks against adversarial examples with l ∞ -norm perturbation constraints.

Recently, randomized smoothing emerged as another class of techniques for obtaining robustness (Lecuyer et al., 2019; Cohen et al., 2019) , with the strongest results in the context of l 2 -norm attacks.

In addition to training neural networks that are robust by construction, a number of methods study the problem of detecting adversarial examples (Metzen et al., 2017; Xu et al., 2018) , with mixed results (Carlini & Wagner, 2017a) .

Of particular interest is recent work on detecting physical adversarial examples (Chou et al., 2018) .

However, detection is inherently weaker than robustness, which is our goal, as even perfect detection does not resolve the question of how to make decisions on adversarial examples.

Finally, our work is in the spirit of other recent efforts that characterize robustness of neural networks to physically realistic perturbations, such as translations, rotations, blurring, and contrast (Engstrom et al., 2019; Hendrycks & Dietterich, 2019) .

Adversarial examples involve modifications of input images that are either invisible to humans, or unsuspicious, and that cause systematic misclassification by state-of-the-art neural networks (Szegedy et al., 2014; Goodfellow et al., 2015; Vorobeychik & Kantarcioglu, 2018) .

Commonly, approaches for generating adversarial examples aim to solve an optimization problem of the following form:

arg max

where x is the original input image, δ is the adversarial perturbation, L(·) is the adversary's utility function (for example, the adversary may wish to maximize the cross-entropy loss), and · p is some l p norm.

While a host of such digital attacks have been proposed, two have come to be viewed as state of the art: the attack developed by Carlini & Wagner (2017b) , and the projected gradient descent attack (PGD) by Madry et al. (2018) .

While most of the work to date has been on attacks which modify the digital image directly, we focus on a class of physical attacks which entail modifying the actual object being photographed in order to fool the neural network that subsequently takes its digital representation as input.

The attacks we will focus on will have three characteristics:

1.

The attack can be implemented in the physical space (e.g., modifying the stop sign); 2.

the attack has low suspiciousness; this is operationalized by modifying only a small part of the object, with the modification similar to common "noise" that obtains in the real world; for example, stickers on a stop sign would appear to most people as vandalism, but covering the stop sign with a printed poster would look highly suspicious; and 3.

the attack causes misclassification by state-of-the-art deep neural network.

Since our ultimate purpose is defense, we will not concern ourselves with the issue of actually implementing the physical attacks.

Instead, we will consider the digital representation of these attacks, ignoring other important issues, such as robustness to many viewpoints and printability.

For example, in the case where the attack involves posting stickers on a stop sign, we will only be concerned with simulating such stickers on digital images of stop signs.

For this reason, we refer to such attacks physically realizable attacks, to allude to the fact that it is possible to realize them in practice.

It is evident that physically realizable attacks represent a somewhat stronger adversarial model than their actual implementation in the physical space.

Henceforth, for simplicity, we will use the terms physical attacks and physically realizable attacks interchangeably.

We consider three physically realizable attacks.

The first is the attack on face recognition by Sharif et al. (2016) , in which the attacker adds adversarial noise inside printed eyeglass frames that can subsequently be put on to fool the deep neural network (Figure 1a ).

The second attack posts adversarially crafted stickers on a stop sign to cause it to be misclassified as another road sign, such as the speed limit sign (Figure 1b ) (Eykholt et al., 2018) .

The third, adversarial patch, attack designs a patch (a sticker) with adversarial noise that can be placed onto an arbitrary object, causing that object to be misclassified by a deep neural network (Brown et al., 2018) .

While numerous approaches have been proposed for making deep learning robust, many are heuristic and have soon after been defeated by more sophisticated attacks (Carlini & Wagner, 2017b; He et al., 2017; Carlini & Wagner, 2017a; Athalye et al., 2018a) .

Consequently, we focus on principled approaches for defense that have not been broken.

These fall broadly into two categories: robust learning and randomized smoothing.

We focus on a state-of-the-art representative from each class.

Robust Learning The goal of robust learning is to minimize a robust loss, defined as follows:

where D denotes the training data set.

In itself this is a highly intractable problem.

Several techniques have been developed to obtain approximate solutions.

Among the most effective in practice is the adversarial training approach by Madry et al. (2018) , who use the PGD attack as an approximation to the inner optimization problem, and then take gradient descent steps with respect to the associated adversarial inputs.

In addition, we consider a modified version of this approach termed curriculum adversarial training (Cai et al., 2018) .

Our implementation of this approach proceeds as follows: first, apply adversarial training for a small , then increase and repeat adversarial training, and so on, increasing until we reach the desired level of adversarial noise we wish to be robust to.

The second class of techniques we consider works by adding noise to inputs at both training and prediction time.

The key idea is to construct a smoothed classifier g(·) from a base classifier f (·) by perturbing the input x with isotropic Gaussian noise with variance σ.

The prediction is then made by choosing a class with the highest probability measure with respect to the induced distribution of f (·) decisions:

To achieve provably robust classification in this manner one typically trains the classifier f (·) by adding Gaussian noise to inputs at training time (Lecuyer et al., 2019; Cohen et al., 2019) .

Most of the approaches for endowing deep learning with adversarial robustness focus on adversarial models in which the attacker introduces l p -bounded adversarial perturbations over the entire input.

Earlier we described two representative approaches in this vein: adversarial training, commonly focused on robustness against l ∞ attacks, and randomized smoothing, which is most effective against l 2 attacks (although certification bounds can be extended to other l p norms as well).

We call these methods conventional robust ML.

In this section, we ask the following question:

Are conventional robust ML methods robust against physically realizable attacks?

This is similar to the question was asked in the context of malware classifier evasion by Tong et al. (2019) , who found that l p -based robust ML methods can indeed be successful in achieving robustness against realizable evasion attacks.

Ours is the first investigation of this issue in computer vision applications and for deep neural networks, where attacks involve adversarial masking of objects.

We study this issue experimentally by considering two state-of-the-art approaches for robust ML: adversarial training a-la-Madry et al. (2018), along with its curriculum learning variation (Cai et al., 2018) , and randomized smoothing, using the implementation by Cohen et al. (2019) .

These approaches are applied to defend against two physically realizable attacks described in Section 2.1: an attack on face recognition which adds adversarial eyeglass frames to faces (Sharif et al., 2016) , and an attack on stop sign classification which adds adversarial stickers to a stop sign to cause misclassification (Eykholt et al., 2018) .

We consider several variations of adversarial training, as a function of the l ∞ bound, , imposed on the adversary.

Just as Madry et al. (2018) , adversarial instances in adversarial training were generated using PGD.

We consider attacks with ∈ {4, 8} (adversarial training failed to make progress when we used = 16).

For curriculum adversarial training, we first performed adversarial training with = 4, then doubled to 8 and repeated adversarial training with the model robust to = 4, then doubled again, and so on.

In the end, we learned models for ∈ {4, 8, 16, 32}. For all versions of adversarial training, we consider 7 and 50 iterations of the PGD attack.

We used the learning rate of /4 for the former and 1 for the latter.

In all cases, pixels are in 0 ∼ 255 range and retraining was performed for 30 epochs using the ADAM optimizer.

For randomized smoothing, we consider noise levels σ ∈ {0.25, 0.5, 1} as in Cohen et al. (2019) , and take 1000 Monte Carlo samples at test time.

We applied white-box dodging (untargeted) attacks on the face recognition systems (FRS) from Sharif et al. (2016) .

We used both the VGGFace data and transferred VGGFace CNN model for the face recognition task, subselecting 10 individuals, with 300-500 face images for each.

Further details about the dataset, CNN architecture, and training procedure are in Appendix A. For the attack, we used identical frames as in Sharif et al. (2016) occupying 6.5% of the pixels.

Just as Sharif et al. (2016) , we compute attacks (that is, adversarial perturbations inside the eyeglass frame area) by using the learning rate 20 as well as momentum value 0.4, and vary the number of attack iterations between 0 (no attack) and 300. (2016) eyeglass frame attack.

First, it is clear that none of the variations of adversarial training are particularly effective once the number of physical attack iterations is above 20.

The best performance in terms of adversarial robustness is achieved by adversarial training with = 8, for approaches using either 7 or 50 PGD iterations (the difference between these appears negligible).

However, non-adversarial accuracy for these models is below 70%, a ∼20% drop in accuracy compared to the original model.

Moreover, adversarial accuracy is under 40% for sufficiently strong physical attacks.

Curriculum adversarial training generally achieves significantly higher non-adversarial accuracy, but is far less robust, even when trained with PGD attacks that use = 32.

Figure 2 (right) shows the performance of randomized smoothing when faced with the eyeglass frames attack.

It is readily apparent that randomized smoothing is ineffective at deflecting this physical attack: even as we vary the amount of noise we add, accuracy after attacks is below 20% even for relatively weak attacks, and often drops to nearly 0 for sufficiently strong attacks.

Following Eykholt et al. (2018), we use the LISA traffic sign dataset for our experiments, and 40 stop signs from this dataset as our test data and perform untargeted attacks (this is in contrast to the original work, which is focused on targeted attacks).

For the detailed description of the data and the CNN used for traffic sign prediction, see Appendix A. We apply the same settings as in the original attacks and use ADAM optimizer with the same parameters.

Since we observed few differences in performance between running PGD for 7 vs. 50 iterations, adversarial training methods in this section all use 7 iterations of PGD.

Again, we begin by considering adversarial training ( Figure 3 , left and middle).

In this case, both the original and curriculum versions of adversarial training with PGD are ineffective when = 32 (error rates on clean data are above 90%); these are consequently omitted from the plots.

Curriculum adversarial training with = 16 has the best performance on adversarial data, and works well on clean data.

Surprisingly, most variants of adversarial training perform at best marginally better than the original model against the stop sign attack.

Even the best variant has relatively poor performance, with robust accuracy under 50% for stronger attacks.

Figure 3 (right) presents the results for randomized smoothing.

In this set of experiments, we found that randomized smoothing performs inconsistently.

To address this, we used 5 random seeds to repeat the experiments, and use the resulting mean values in the final results.

Here, the best variant uses σ = 0.25, and, unlike experiments with the eyeglass frame attack, significantly outperforms adversarial training, reaching accuracy slightly above 60% even for the stronger attacks.

Neverthe-less, even randomized smoothing results in significant degradation of effectiveness on adversarial instances (nearly 40%, compared to clean data).

There are two possible reasons why conventional robust ML perform poorly against physical attacks: 1) adversarial models involving l p -bounded perturbations are too hard to enable effective robust learning, and 2) the conventional attack model is too much of a mismatch for realistic physical attacks.

In Appendix B, we present evidence supporting the latter.

Specifically, we find that conventional robust ML models exhibit much higher robustness when faced with the l p -bounded attacks they are trained to be robust to.

As we observed in Section 3, conventional models for making deep learning robust to attack can perform quite poorly when confronted with physically realizable attacks.

In other words, the evidence strongly suggests that the conventional models of attacks in which attackers can make l p -bounded perturbations to input images are not particularly useful if one is concerned with the main physical threats that are likely to be faced in practice.

However, given the diversity of possible physical attacks one may perpetrate, is it even possible to have a meaningful approach for ensuring robustness against a broad range of physical attacks?

For example, the two attacks we considered so far couldn't be more dissimilar: in one, we engineer eyeglass frames; in another, stickers on a stop sign.

We observe that the key common element in these attacks, and many other physical attacks we may expect to encounter, is that they involve the introduction of adversarial occlusions to a part of the input.

The common constraint faced in such attacks is to avoid being suspicious, which effectively limits the size of the adversarial occlusion, but not necessarily its shape or location.

Next, we introduce a simple abstract model of occlusion attacks, and then discuss how such attacks can be computed and how we can make classifiers robust to them.

We propose the following simple abstract model of adversarial occlusions of input images.

The attacker introduces a fixed-dimension rectangle.

This rectangle can be placed by the adversary anywhere in the image, and the attacker can furthermore introduce l ∞ noise inside the rectangle with an exogenously specified high bound (for example, = 255, which effectively allows addition of arbitrary adversarial noise).

This model bears some similarity to l 0 attacks, but the rectangle imposes a contiguity constraint, which reflects common physical limitations.

The model is clearly abstract: in practice, for example, adversarial occlusions need not be rectangular or have fixed dimensions (for example, the eyeglass frame attack is clearly not rectangular), but at the same time cannot usually be arbitrarily superimposed on an image, as they are implemented in the physical environment.

Nevertheless, the model reflects some of the most important aspects common to many physical attacks, such as stickers placed on an adversarially chosen portion of the object we wish to identify.

We call our attack model a rectangular occlusion attack (ROA).

An important feature of this attack is that it is untargeted: since our ultimate goal is to defend against physical attacks whatever their target, considering untargeted attacks obviates the need to have precise knowledge about the attacker's goals.

For illustrations of the ROA attack, see Appendix C.

The computation of ROA attacks involves 1) identifying a region to place the rectangle in the image, and 2) generating fine-grained adversarial perturbations restricted to this region.

The former task can be done by an exhaustive search: consider all possible locations for the upper left-hand corner of the rectangle, compute adversarial noise inside the rectangle using PGD for each of these, and choose the worst-case attack (i.e., the attack which maximizes loss computed on the resulting image).

However, this approach would be quite slow, since we need to perform PGD inside the rectangle for every possible position.

Our approach, consequently, decouples these two tasks.

Specifically, we first perform an exhaustive search using a grey rectangle to find a position for it that maximizes loss, and then fix the position and apply PGD inside the rectangle.

An important limitation of the exhaustive search approach for ROA location is that it necessitates computations of the loss function for every possible location, which itself requires full forward propagation each time.

Thus, the search itself is still relatively slow.

To speed the process up further, we use the gradient of the input image to identify candidate locations.

Specifically, we select a subset of C locations for the sticker with the highest magnitude of the gradient, and only exhaustively search among these C locations.

C is exogenously specified to be small relative to the number of pixels in the image, which significantly limits the number of loss function evaluations.

Full details of our algorithms for computing ROA are provided in Appendix D.

Once we are able to compute the ROA attack, we apply the standard adversarial training approach for defense.

We term the resulting classifiers robust to our abstract adversarial occlusion attacks Defense against Occlusion Attacks (DOA), and propose these as an alternative to conventional robust ML for defending against physical attacks.

As we will see presently, this defense against ROA is quite adequate for our purposes.

We now evaluate the effectiveness of DOA-that is, adversarial training using the ROA threat model we introduced-against physically realizable attacks (see Appendix G for some examples that defeat conventional methods but not DOA).

Recall that we consider only digital representations of the corresponding physical attacks.

Consequently, we can view our results in this section as a lower bound on robustness to actual physical attacks, which have to deal with additional practical constraints, such as being robust to multiple viewpoints.

In addition to the two physical attacks we previously considered, we also evaluate DOA against the adversarial patch attack, implemented on both face recognition and traffic sign data.

We consider two rectangle dimensions resulting in comparable area: 100 × 50 and 70 × 70, both in pixels.

Thus, the rectangles occupy approximately 10% of the 224 × 224 face images.

We used {30, 50} iterations of PGD with = 255/2 to generate adversarial noise inside the rectangle, and with learning rate α = {8, 4} correspondingly.

For the gradient version of ROA, we choose C = 30.

DOA adversarial training is performed for 5 epochs with a learning rate of 0.0001.

Figure 4: Performance of DOA (using the 100 × 50 rectangle) against the eyeglass frame attack in comparison with conventional methods.

Left: comparison between DOA, adversarial training, and randomized smoothing (using the most robust variants of these).

Middle/Right: Comparing DOA performance for different rectangle dimensions and numbers of PGD iterations inside the rectangle.

Middle: using exhaustive search for ROA; right: using the gradient-based heuristic for ROA.

Figure 4 (left) presents the results comparing the effectiveness of DOA against the eyeglass frame attack on face recognition to adversarial training and randomized smoothing (we took the most robust variants of both of these).

We can see that DOA yields significantly more robust classifiers for this domain.

The gradient-based heuristic does come at some cost, with performance slightly worse than when we use exhaustive search, but this performance drop is relatively small, and the result is still far better than conventional robust ML approaches.

Figure 4 (middle and right) compares the performance of DOA between two rectangle variants with different dimensions.

The key observation is that as long as we use enough iterations of PGD inside the rectangle, changing its dimensions (keeping the area roughly constant) appears to have minimal impact.

We now repeat the evaluation with the traffic sign data and the stop sign attack.

In this case, we used 10 × 5 and 7 × 7 rectangles covering ∼5 % of the 32 × 32 images.

We set C = 10 for the gradientbased ROA.

Implementation of DOA is otherwise identical as in the face recognition experiments above.

We present our results using the square rectangle, which in this case was significantly more effective; the results for the 10 × 5 rectangle DOA attacks are in Appendix F. Figure 5 (left) compares the effectiveness of DOA against the stop sign attack on traffic sign data with the best variants of adversarial training and randomized smoothing.

Our results here are for 30 iterations of PGD; in Appendix F, we study the impact of varying the number of PGD iterations.

We can observe that DOA is again significantly more robust, with robust accuracy over 90% for the exhaustive search variant, and ∼85% for the gradient-based variant, even for stronger attacks.

Moreover, DOA remains 100% effective at classifying stop signs on clean data, and exhibits ∼95% accuracy on the full traffic sign classification task.

Finally, we evaluate DOA against the adversarial patch attacks.

In these attacks, an adversarial patch (e.g., sticker) is designed to be placed on an object with the goal of inducing a target prediction.

We study this in both face recognition and traffic sign classification tasks.

Here, we present the results for face recognition; further detailed results on both datasets are provided in Appendix F.

As we can see from Figure 5 (right), adversarial patch attacks are quite effective once the attack region (fraction of the image) is 10% or higher, with adversarial training and randomized smoothing both performing rather poorly.

In contrast, DOA remains highly robust even when the adversarial patch covers 20% of the image.

As we have shown, conventional methods for making deep learning approaches for image classification robust to physically realizable attacks tend to be relatively ineffective.

In contrast, a new threat model we proposed, rectangular occlusion attacks (ROA), coupled with adversarial training, achieves high robustness against several prominent examples of physical attacks.

While we explored a number of variations of ROA attacks as a means to achieve robustness against physical attacks, numerous questions remain.

For example, can we develop effective methods to certify robustness against ROA, and are the resulting approaches as effective in practice as our method based on a combination of heuristically computed attacks and adversarial training?

Are there other types of occlusions that are more effective?

Answers to these and related questions may prove a promising path towards practical robustness of deep learning when deployed for downstream applications of computer vision such as autonomous driving and face recognition. (Parkhi et al., 2015 ) is a benchmark for face recognition, containing 2622 subjusts with 2.6 million images in total.

We chose ten subjects: A. J. Buckley, A. R. Rahman, Aamir Khan, Aaron Staton, Aaron Tveit, Aaron Yoo, Abbie Cornish, Abel Ferrara, Abigail Breslin, and Abigail Spencer, and subselected face images pertaining only to these individuals.

Since approximately half of the images cannot be downloaded, our final dataset contains 300-500 images for each subject.

We used the standard corp-and-resize method to process the data to be 224 × 224 pixels, and split the dataset into training, validation, and test according to a 7:2:1 ratio for each subject.

In total, the data set has 3178 images in the training set, 922 images in the validation set, and 470 images in the test set.

We use the VGGFace convolutional neural network (Parkhi et al., 2015) model, a variant of the VGG16 model containing 5 convolutional layer blocks and 3 fully connected layers.

We make use of standard transfer learning as we only classify 10 subjects, keeping the convolutional layers as same as VGGFace structure, 3 but changing the fully connected layer to be 1024 →

1024 →10 instead of 4096 → 4096 →2622.

Specifically, in our Pytorch implementation, we convert the images from RGB to BGR channel orders and subtract the mean value [129.1863, 104.7624, 93 .5940] in order to use the pretrained weights from VGG-Face on convolutional layers.

We set the batch size to be 64 and use Pytorch built-in Adam Optimizer with an initial learning rate of 10 −4 and default parameters in Pytorch.

4 We drop the learning rate by 0.1 every 10 epochs.

Additionally, we used validation set accuracy to keep track of model performance and choose a model in case of overfitting.

After 30 epochs of training, the model successfully obtains 98.94 % on test data.

To be consistent with (Eykholt et al., 2018), we select the subset of LISA which contains 47 different U.S. traffic signs (Møgelmose et al., 2012) .

To alleviate the problem of imbalance and extremely blurry data, we picked 16 best quality signs with 3509 training and 1148 validation data points.

From the validation data, we obtain the test data that includes only 40 stop signs to evaluate performance with respect to the stop sign attack, as done by Eykholt et al. (2018) .

In the main body of the paper, we present results only on this test data to evaluate robustness to stop sign attacks.

In the appendix below, we also include performance on the full validation set without adversarial manipulation.

All the data was processed by standard crop-and-resize to 32 × 32 pixels.

We use the LISA-CNN architecture defined in (Eykholt et al., 2018) , and construct a convolutional neural network containing three convolutional layers and one fully connected layer.

We use the Adam Optimizer with initial learning rate of 10 −1 and default parameters 4 , dropping the learning rate by 0.1 every 10 epochs.

We set the batch size to be 128.

After 30 epochs, we achieve the 98.69 % accuracy on the validation set, and 100% accuracy in identifying the stop signs in our test data.

AND l 2 ATTACKS

In this appendix, we show that adversarial training and randomized smoothing degrade more gracefully when faced with attacks that they are designed for.

In particular, we consider here variants of projected gradient descent (PGD) for both the l ∞ and l 2 attacks Madry et al. (2018) .

In particular, the form of PGD for the l ∞ attack is

where Proj is a projection operator which clips the result to be feasible, x t the adversarial example in iteration t, α the learning rate, and L(·) the loss function.

In the case of an l 2 attack, PGD becomes

where the projection operator normalizes the perturbation δ = x t+1 − x t to have δ 2 ≤ if it doesn't already Kolter & Madry (2019) .

The experiments were done on the face recognition and traffic sign datasets, but unlike physical attacks on stop signs, we now consider adversarial perturbations to all sign images.

We begin with our results on the face recognition dataset.

Tables 1 and 2 present results for (curriculum) adversarial training for varying of the l ∞ attacks, separately for training and evaluation.

As we can see, curriculum adversarial training with = 16 is generally the most robust, and remains reasonably effective for relatively large perturbations.

However, we do observe a clear tradeoff between accuracy on non-adversarial data and robustness, as one would expect.

Table 3 presents the results of using randomized smoothing on face recognition data, when facing the l 2 attacks.

Again, we observe a high level of robustness and, in most cases, relatively limited drop in performance, with σ = 0.5 perhaps striking the best balance.

Tables 4 and 5 present evaluation on traffic sign data for curriculum adversarial training against the l ∞ attack for varying .

As with face recognition data, we can observe that the approaches tend to be relatively robust, and effective on non-adversarial data for adversarial training methods using < 32.

The results of randomized smoothing on traffic sign data are given in Table 6 .

Since images are smaller here than in VGGFace, lower values of for the l 2 attacks are meaningful, and for ≤ 1 we generally see robust performance on randomized smoothing, with σ = 0.5 providing a good balance between non-adversarial accuracy and robustness, just as before.

Our basic algorithm for computing rectangular occlusion attacks (ROA) proceeds through the following two steps:

1.

Iterate through possible positions for the rectangle's upper left-hand corner point in the image.

Find the position for a grey rectangle (RGB value =[127.5, 127.5, 127.5] ) in the image that maximizes loss.

2.

Generate high-l ∞ noise inside the rectangle at the position computed in step 1.

Algorithm 1 presents the full algorithm for identifying the ROA position, which amounts to exhaustive search through the image pixel region.

This algorithm has several parameters.

First, we assume that images are squares with dimensions N 2 .

Second, we introduce a stride parameter S. The purpose of this parameter is to make location computation faster by only considering every other Sth pixel during the search (in other words, we skip S pixels each time).

For our implementation of ROA attacks, we choose the stride parameter S = 5 for face recognition and S = 2 for traffic sign classification.

Once we've found the place for the rectangle, our next step is to introduce adversarial noise inside it.

For this, we use the l ∞ version of the PGD attack, restricting perturbations to the rectangle.

We used {7, 20, 30, 50} iterations of PGD to generate adversarial noise inside the rectangle, and with learning rate α = {32, 16, 8, 4} correspondingly.

Grad.

plot Grad.

searching Exh.

searching

Physically realizable attacks that we study have a common feature: first, they specify a mask, which is typically precomputed, and subsequently introduce adversarial noise inside the mask area.

Let M denote the mask matrix constraining the area of the perturbation δ; M has the same dimensions as the input image and contains 0s where no perturbation is allowed, and 1s in the area which can be perturbed.

The physically realizable attacks we consider then solve an optimization problem of the following form: arg max

Next, we describe the details of the three physical attacks we consider in the main paper.

Following Sharif et al. (2016), we first initialized the eyeglass frame with 5 different colors, and chose the best starting color by calculating the cross-entropy loss.

For each update step, we divided the gradient value by its maximum value before multiplying by the learning rate which is 20.

Then we only kept the gradient value of eyeglass frame area.

Finally, we clipped and rounded the pixel value to keep it in the valid range.

Following Eykholt et al. (2018), we initialized the stickers on the stop signs with random noise.

For each update step, we used the Adam optimizer with 0.1 learning rate and with default parameters.

Just as for other attacks, adversarial perturbations were restricted to the mask area exogenously specified; in our case, we used the same mask as Eykholt et al. (2018)-a collection of small rectangles.

We used gradient ascent to maximize the log probability of the targeted class P [y target |x], as in the original paper (Brown et al., 2018) .

When implementing the adversarial patch, we used a square patch rather than the circular patch in the original paper; we don't anticipate this choice to be practically consequential.

We randomly chose the position and direction of the patch, used the learning rate of 5, and fixed the number of attack iterations to 100 for each image.

We varied the attack region (mask) R ∈ {0%, 5%, 10%, 15%, 20%, 25%}.

For the face recognition dataset, we used 27 images (9 classes (without targeted class) × 3 images in each class) to design the patch, and then ran the attack over 20 epochs.

For the smaller traffic sign dataset, we used 15 images (15 classes (without targeted class) × 1 image in each class) to design the patch, and then ran the attack over 5 epochs.

Note that when evaluating the adversarial patch, we used the validation set without the targeted class images.

Figure 11 : Examples of the eyeglass attack on face recognition.

From left to right: 1) the original input image, 2) image with adversarial eyeglass frames, 3) face predicted by a model generated through adversarial training, 4) face predicted by a model generated through randomized smoothing, 5) face predicted (correctly) by a model generated through DOA.

Each row is a separate example.

Figure 12 : Examples of the stop sign attack.

From left to right: 1) the original input image, 2) image with adversarial eyeglass frames, 3) face predicted by a model generated through adversarial training, 4) face predicted by a model generated through randomized smoothing, 5) face predicted (correctly) by a model generated through DOA.

Each row is a separate example.

H EFFECTIVENESS OF DOA METHODS AGAINST l ∞ ATTACKS For completeness, this section includes evaluation of DOA in the context of l ∞ -bounded attacks implemented using PGD, though these are outside the scope of our threat model.

Table 23 presents results of several variants of DOA in the context of PGD attacks in the context of face recognition, while Table 24 considers these in traffic sign classification.

The results are quite consistent with intuition: DOA is largely unhelpful against these attacks.

The reason is that DOA fundamentally assumes that the attacker only modifies a relatively small proportion (∼5%) of the scene (and the resulting image), as otherwise the physical attack would be highly suspicious.

l ∞ bounded attacks, on the other hand, modify all pixels.

To further illustrate the ability of DOA to generalize, we evaluate its effectiveness in the context of three additional occlusion patterns: a union of triangles and circle, a single larger triangle, and a heart pattern.

As the results in Figures 13 and 14 suggest, DOA is able to generalize successfully to a variety of physical attack patterns.

It is particularly noteworthy that the larger patterns (large triangle-middle of the figure, and large heart-right of the figure) are actually quite suspicious (particularly the heart pattern), as they occupy a significant fraction of the image (the heart mask, for example, accounts for 8% of the face).

Mask2 Mask3 Abbie Cornish Abbie Cornish Abbie Cornish

<|TLDR|>

@highlight

Defending Against Physically Realizable Attacks on Image Classification
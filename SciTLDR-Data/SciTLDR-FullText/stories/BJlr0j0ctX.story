Adversarial training is one of the strongest defenses against adversarial attacks, but it requires adversarial examples to be generated for every mini-batch during optimization.

The expense of producing these examples during training often precludes adversarial training from use on complex image datasets.

In this study, we explore the mechanisms by which adversarial training improves classifier robustness, and show that these mechanisms can be effectively mimicked using simple regularization methods, including label smoothing and logit squeezing.

Remarkably, using these simple regularization methods in combination with Gaussian noise injection, we are able to achieve strong adversarial robustness -- often exceeding that of adversarial training -- using no adversarial examples.

Deep Neural Networks (DNNs) have enjoyed great success in many areas of computer vision, such as classification BID7 , object detection BID4 , and face recognition BID11 .

However, the existence of adversarial examples has raised concerns about the security of computer vision systems BID16 BID1 .

For example, an attacker may cause a system to mistake a stop sign for another object BID3 or mistake one person for another BID14 .

To address security concerns for high-stakes applications, researchers are searching for ways to make models more robust to attacks.

Many defenses have been proposed to combat adversarial examples.

Approaches such as feature squeezing, denoising, and encoding BID19 BID13 BID15 BID10 have had some success at pre-processing images to remove adversarial perturbations.

Other approaches focus on hardening neural classifiers to reduce adversarial susceptibility.

This includes specialized non-linearities BID20 , modified training processes BID12 , and gradient obfuscation BID0 .Despite all of these innovations, adversarial training BID5 , one of the earliest defenses, still remains among the most effective and popular strategies.

In its simplest form, adversarial training minimizes a loss function that measures performance of the model on both clean and adversarial data as follows DISPLAYFORM0 where L is a standard (cross entropy) loss function, (x i , y i ) is an input image/label pair, ?? contains the classifier's trainable parameters, ?? is a hyper-parameter, and x i,adv is an adversarial example for image x. BID9 pose adversarial training as a game between two players that similarly requires computing adversarial examples on each iteration.

A key drawback to adversarial training methods is their computational cost; after every mini-batch of training data is formed, a batch of adversarial examples must be produced.

To train a network that resists strong attacks, one needs to train with the strongest adversarial examples possible.

For example, networks hardened against the inexpensive Fast Gradient Sign Method (FGSM, Goodfellow et al. (2014) ) can be broken by a simple two-stage attack BID17 .

Current state-of-theart adversarial training results on MNIST and CIFAR-10 use expensive iterative adversaries BID9 , such as the Projected Gradient Descent (PGD) method, or the closely related Basic Iterative Method (BIM) BID8 .

Adversarial training using strong attacks may be 10-100 times more time consuming than standard training methods.

This prohibitive cost makes it difficult to scale adversarial training to larger datasets and higher resolutions.

In this study, we show that it is possible to achieve strong robustness -comparable to or greater than the robustness of adversarial training with a strong iterative attack -using fast optimization without adversarial examples.

We achieve this using standard regularization methods, such as label smoothing BID18 and the more recently proposed logit squeezing BID6 .

While it has been known for some time that these tricks can improve the robustness of models, we observe that an aggressive application of these inexpensive tricks, combined with random Gaussian noise, are enough to match or even surpass the performance of adversarial training on some datasets.

For example, using only label smoothing and augmentation with random Gaussian noise, we produce a CIFAR-10 classifier that achieves over 73% accuracy against black-box iterative attacks, compared to 64% for a state-of-the-art adversarially trained classifier BID9 .

In the white-box case, classifiers trained with logit squeezing and label smoothing get ??? 50% accuracy on iterative attacks in comparison to ??? 47% for adversarial training.

Regularized networks without adversarial training are also more robust against non-iterative attacks, and more accurate on non-adversarial examples.

Our goal is not just to demonstrate these defenses, but also to dig deep into what adversarial training does, and how it compares to less expensive regularization-based defenses.

We begin by dissecting adversarial training, and examining ways in which it achieves robustness.

We then discuss label smoothing and logit squeezing regularizers, and how their effects compare to those of adversarial training.

We then turn our attention to random Gaussian data augmentation, and explore the importance of this technique for adversarial robustness.

Finally, we combine the regularization methods with random Gaussian augmentation, and experimentally compare the robustness achievable using these simple methods to that achievable using adversarial training.

Adversarial training injects adversarial examples into the training data as SGD runs.

During training, adversarial perturbations are applied to each training image to decrease the logit corresponding to its correct class.

The network must learn to produce logit representations that preserve the correct labeling even when faced with such an attack.

At first glance, it seems that adversarial training might work by producing a large "logit gap," i.e., by producing a logit for the true class that is much larger than the logit of other classes.

Surprisingly, adversarial training has the opposite effect -we will see below that it decreases the logit gap.

To better understand what adversarial training does, and how we can replicate it, we now break down the different strategies for achieving robustness.

This section presents a simple metric for adversarial robustness that will help us understand adversarial training.

Consider an image x, and its logit representation z (i.e. pre-softmax activation) produced by a neural network.

Let z y denote the logit corresponding to the correct class y. If we add a small perturbation ?? to x, then the corresponding change in logits is approximately ?? T ??? x z y under a linear approximation, where ??? x z y is the gradient of z y with respect to x.

Under a linearity assumption, we can calculate the step-size L needed to move an example from class y to another class??.

A classifier is susceptible to adversarial perturbation ?? if the perturbed logit of the true class is smaller than the perturbed logit of any other class: DISPLAYFORM0 Assuming a one-step ??? attack such as FGSM, the perturbation ?? can be expressed as DISPLAYFORM1 where L is the ??? -norm of the perturbation.

Using this choice of ??, Equation 2 becomes Table 1 : Experimental and predicted accuracy of classifiers for MNIST and CIFAR-10.

The predicted accuracy is the percentage of images for which < L .

The empirical accuracy is the percent of images that survive a perturbation of size .

Attacks on both the cross-entropy (X-ent) and logits as in BID2 (CW) are presented.

DISPLAYFORM2 where ?? 1 denotes the 1 -norm of a vector.

Therefore the smallest ??? -norm of the perturbation required is the ratio of "logit gap" to "gradient gap", i.e., DISPLAYFORM3 Equation FORMULA4 measures robustness by predicting the smallest perturbation size needed to switch the class of an image.

While the formula for L makes linearity assumptions, the approximation L fairly accurately predicts the robustness of classifiers of the CIFAR-10 dataset (where perturbations are small and linearity assumptions cause little distortion).

It is also a good ballpark approximation on MNIST, even after adversarial training (see Table 1 ).Maximal robustness occurs when L is as large as possible.

From equation 4, we observe 3 different strategies for hardening a classifier:??? Increase the logit gap: Maximize the numerator of equation 4 by producing a classifier with relatively large z y .???

Squash the adversarial gradients: Train a classifier that has small adversarial gradients ??? x z?? for any class??.

In this case a large perturbation is needed to significantly change z??.??? Maximize gradient coherence: Produce adversarial gradients ??? x z?? that are highly correlated with the gradient for the correct class ??? x z y .

This will shrink the denominator of equation 4, and produce robustness even if adversarial gradients are large.

In this case, one cannot decrease z y without also decreasing z??, and so large perturbations are needed to change the class label.

The most obvious strategy for achieving robustness is to increase the numerator in equation 4 while fixing the denominator.

Remarkably, our experimental investigation reveals that adversarial training does not rely on this strategy at all, but rather it decreases the logit gap and gradient gap simultaneously.

This can be observed in FIG0 , which shows distributions of logit gaps for naturally and adversarially trained models on MNIST.

Note that the cross entropy loss actually limits adversarial training from increasing logit gaps.

The accuracy of the classifier goes down in the presence of adversarial examples, and so the cross entropy loss is minimized by smaller logit gaps that reflect the lower level of certainty in the adversarial training environment.

Adversarial training succeeds by minimizing the denominator in Equation 4; it simultaneously squeezes the logits and crushes the adversarial gradients.

FIG0 shows that the adversarial gradients shrink dramatically more than the logit gaps, and so the net effect is an increase in robustness.

If we closely examine the phenomenon of shrinking the logit gaps, we find that this shrink is due in part to an overall shrink in the size of the logits themselves, (i.e., |z i | for any class i).

To see this, we plot histograms of the logits when classifiers are adversarially trained with strong adversaries 1 , weak adversaries 2 , and with no adversarial examples.

FIG1 shows that adversarial training does indeed squash the logits, although not enough to fully explain the decrease in |z y ??? z??| in FIG0 .

3 We have seen that adversarial training works by squashing adversarial gradients and slightly increasing gradient coherence.

But the fact that it cannot do this without decreasing the logit gap leads us to suspect that these quantities are inherently linked.

This leads us to ask an important question: If we directly decrease the logit gap, or the logits themselves, using an explicit regularization term, will this have the desired effect of crushing the adversarial gradients?

There are two approaches to replicating the effect on the logits produced by adversarial training.

The first is to replicate the decrease in logit gap seen in FIG0 .

This can be achieved by label smoothing.

A second approach to replicating adversarial training is to just directly crush all logit values and mimic the behavior in FIG1 .

This approach is known as "logit squeezing," and works by adding a regularization term to the training objective that explicitly penalizes large logits.

Label smoothing Label smoothing converts "one-hot" label vectors into "one-warm" vectors that represents a low-confidence classification.

Because large logit gaps produce high-confidence classifications, label-smoothed training data forces the classifier to produce small logit gaps.

Label smoothing is a commonly used trick to prevent over-fitting on general classification problems, and it was first observed to boost adversarial robustness by BID18 , where it was used as an inexpensive replacement for the network distillation defense BID12 .

A one-hot label vector y hot is smoothed using the formula DISPLAYFORM0 where ?? ??? [0, 1] is the smoothing parameter, and N c is the number of classes.

If we pick ?? = 0 we get a hard decision vector with no smoothing, while ?? = 1 creates an ambiguous decision by assigning equal certainty to all classes.

Logit squeezing A second approach to replicating adversarial training is to just directly crush all logit values and mimic the behavior in FIG1 .

This approach is known as "logit squeezing," and works by adding a regularization term to the training objective that explicitly penalizes large logits.

BID6 were the first to introduce logit-squeezing as an alternative to a "logit pairing" defense.

Logit squeezing relies on the loss function DISPLAYFORM1 where ?? is the squeezing parameter (i.e., coefficient for the logit-squeezing term) and ||.|| F is the Frobenius norm of the logits for the mini-batch.

Can such simple regularizers really replicate adversarial training?

Our experimental results suggest that simple regularizers can hurt adversarial robustness, which agrees with the findings in BID20 .

However, these strategies become highly effective when combined with a simple trick from the adversarial training literature -data augmentation with Gaussian noise.

Adding Gaussian noise to images during training (i.e, Gaussian noise augmentation) can be used to improve the adversarial robustness of classifiers BID6 BID20 .

However, the effect of Gaussian noise is not well understood BID6 .

Here, we take a closer look at the behavior of Gaussian augmentation through systematic experimental investigations, and see that its effects are more complex than one might think.

Label smoothing and logit squeezing become shockingly effective at hardening networks when they are combined with Gaussian noise augmentation.

From the robustness plots in FIG2 , we can see that training with Gaussian noise alone produces a noticeable change in robustness, which seems to be mostly attributable to a widening of the logit gap and slight decrease in the gradient gap ( ??? x z y ??? ??? x z?? 1 ).

The small increase in robustness from random Gaussian augmentation was also reported by BID6 .

We also see that label smoothing alone causes a very slight drop in robustness; the shrink in the gradient gap is completely offset by a collapse in the logit gap.

Surprisingly, Gaussian noise and label smoothing have a powerful synergistic effect.

When used together they cause a dramatic drop in the gradient gap, leading to a surge in robustness.

A similar effect happens in the case of logit squeezing, and results are shown in Appendix B (Figure 8 ).

Regularization methods have the potential to squash or align the adversarial gradients, but these properties are only imposed during training on images from the manifold that the "true" data lies on.

At test time, the classifier sees adversarial images that do not "look like" training data because they lie off of, but adjacent to, the image manifold.

By training the classifier on images with random perturbations, we teach the classifier to enforce the desired properties for input images that lie off the manifold.

The generalization property of Gaussian augmentation seems to be independent from, and sometimes conflicting with, the synergistic properties discuss above.

In our experiments below, we find that smaller noise levels lead to a stronger synergistic effect, and yield larger L and better robustness to FGSM attacks.

However, larger noise levels enable the regularization properties to generalize further off the manifold, resulting in better robustness to iterative attacks or attacks that escape the flattened region by adding an initial random perturbation.

See the results on MNIST in Table 2 and the results on CIFAR-10 in TAB3 for various values of ?? (standard deviation of Gaussian noise).

For more comprehensive experiments on the different parameters that contribute to the regularizers see Table 6 for MNIST and Tables 7 & 8

Label smoothing (i.e. reducing the variance of the logits) is helpful because it causes the gradient gap to decrease.

The decreased gradient gap may be due to smaller element-wise gradient amplitudes, the alignments of the adversarial gradients, or both.

To investigate the causes, we plot the 1 norm of the gradients of the logits with respect to the input image 4 and the cosine of the angle between the gradients FIG3 .

We see that in label smoothing (with Gaussian augmentation), both the gradient magnitude decreases and the gradients get more aligned.

Larger smoothing parameters ?? cause the gradient to be both smaller and more aligned.

When logit squeezing is used with Gaussian augmentation, the magnitudes of the gradients decrease.

The distribution of the cosines between gradients widens, but does not increase like it did for label smoothing.

These effects are very similar to the behavior of adversarial training in FIG0 .

Interestingly, in the case of logit squeezing with Gaussian noise, unlike label smoothing, the numerator of Equation 4 increases as well.

This increase in the logit gap disappears once we take away Gaussian augmentation (See Appendix B Figure 8 ).

Simultaneously increasing the numerator and decreasing the denominator of Equation 4 potentially gives a slight advantage to logit squeezing.

There are multiple factors that can affect the robustness of the MNIST classifier 5 .

While regularizers do not yield more robustness than adversarial training for MNIST, the results are promising given that these relatively high values of robustness come at a cheap cost in comparison to adversarial training.

In Table 2 we notice that as we increase the number of training iterations k, we get more robust models for both logit squeezing and label smoothing 6 .

We get more robust models when we use larger smoothing (??) and squeezing (??) parameters, and when Gaussian augmentation is used with standard deviation ?? that is greater than the desired (the maximum perturbation size).

Table 2 : Accuracy of different MNIST classifiers against PGD and FGSM attacks on X-ent and CW losses under the white-box and black-box threat models.

Attacks have maximum ??? perturbation = 0.3.

The iterative white-box attacks have an initial random step.

The naturally trained model is used for generating black-box attacks.

We use CW loss for the black-box attack.

We trained Wide-Resnet CIFAR-10 classifiers (depth=28 and k=10 ) using aggressive values for the smoothing and squeezing parameters on the CIFAR10 data set.

Similar to BID9 , we use the standard data-augmentation techniques and weight-decay.

We compare our results to those of BID9 .

Note that the adversarially trained model from BID9 has been trained for 80,000 iterations on adversaries which are generated using a 7-step PGD.

Keeping in mind that each step requires a forward and backward pass, the running time of training for 80,000 iterations on 7-step PDG examples is equivalent to 640,000 iterations of training with label smoothing or logit squeezing.

A short version of our results on white-box attacks are summarized in TAB3 .

The results of some of our black-box experiments are summarized in Table 4 7 .

While logit squeezing seems to outperform label smoothing in the white-box setting, label smoothing is slightly better under the black-box threat.

We see that aggressive logit squeezing with squeezing parameter ?? = 10 and ?? = 20 results in a model that is more robust than the adversarially trained model from BID9 BID9 63.39%* 64.38%* 63.39%* 64.38%* 67.00% 67.25% Table 4 : Black-box attacks on CIFAR-10 models.

Attacks are ??? with = 8.

Similar to BID9 , We build 7-step PGD attacks and FGSM attacks for a public adversarially trained model.

*Values taken from the original paper by BID9 .

Some defenses work by obfuscating gradients and masking the gradients.

BID0 suggest these models can be identified by performing "sanity checks" such as attacking them with Figure 5 : The cross-entropy landscape of the first eight images of the validation set for the model trained for 160k iterations with hyper-parameters ?? = 10 and ?? = 30.

To plot the loss landscape we take two random directions r 1 and r 2 (i.e. r 1 , r 2 ??? Rademacher(0.5) ).

We plot the cross-entropy (i.e. xent) loss at different points x = x i + 1 ?? r 1 + 2 ?? r 2 .

Where x i is the clean image and ???10 ??? 1 , 2 ??? 10.unbounded strong adversaries (i.e. unbounded with many iterations).

By attacking our robust models using these unbounded attacks, we verify that the unbounded adversary can degrade the accuracy to 0.00% which implies that the adversarial example generation optimization attack (PGD) is working properly.

Also, it is known that models which do not completely break the PGD attack (such as us) can possibly lead to a false sense of security by creating a loss landscape that prevents an -bounded but weak adversary from finding strong adversarial examples.

This can be done by convoluting the loss landscape such that many local-optimal points exist.

This false sense of security can be identified by increasing the strength of the adversary using methods such as increasing the number of steps and random restarts of the PGD attack.

We run the stronger PGD attacks on a sample model from TAB3 with hyper-parameters k = 160k, ?? = 10, and ?? = 30.

We notice that performing 9 random restarts for the PGD attack on the cross-entropy loss only drops the accuracy slightly to 49.86%.

Increasing the number of PGD steps to 1000 decreases the accuracy slightly more to 40.94% 8 .

While for such strong iterative white-box attacks our sample model is less robust than the adversarially trained model, there are other areas where this hardened model is superior to the adversarially trained model: Very high robustness in the black box setting (roughly 3% higher than that for adversarial training according to Table 4 ) and against white-box non-iterative (or less iterative) attacks (roughly 15%); high test accuracy on clean data (roughly 3%); and, very fast training time compared to adversarial training.

To further verify that our robust model is not falsely creating a sense of robustness by "breaking" the optimization problem that generates adversarial examples by either masking the gradients or making the loss landscape convoluted, we visualize the loss landscape of our sample model from TAB3 .

We plot the classification (e.g., cross-entropy) loss for points surrounding the first eight validation images that belong to the subspace spanned by two random directions 9 in Figure 5 .

It seems that the loss landscape has not become convoluted.

This observation is backed up by the fact that increasing the number of PGD attack steps does not substantially affect accuracy.

To check the performance of our proposed regularizers on more complicated datasets with more number of classes, we perform aggressive logit squeezing on the CIFAR-100 dataset which contains 100 categories.

We use the same architecture and settings used for training the CIFAR-10 classifiers.

The white-box performance of two hardened models with logit squeezing and a PGD adversarially trained model for the same architecture are summarized in Table 5 Table 5 : White-box iterative attacks on CIFAR-100 models.

We use ??? attacks with = 8.

For brevity, we only report the results for attacking the cross-entropy loss.

We attack the models with adversaries having different strengths by varying the number of PGD steps.

Similar to CIFAR-10, aggressive logit squeezing can result in models as robust as those that are adversarially trained at a fraction of the cost.

The logit-squeezed model that is trained for only 80k iterations achieves high classification accuracy for natural/clean examples.

It is also more robust against white-box PGD attacks compared to the adversarially trained model that requires much more training time.

The logit-squeezed model with k = 160k improves the robustness and clean accuracy even further and still trains faster than the adversarially trained model (28.8 hours vs 34.3 hours).

We studied the robustness of adversarial training, label smoothing, and logit squeezing through a linear approximation L that relates the magnitude of adversarial perturbations to the logit gap and the difference between the adversarial directions for different labels.

Using this simple model, we observe how adversarial training achieves robustness and try to imitate this robustness using label smoothing and logit squeezing.

The resulting methods perform well on MNIST, and can get results on CIFAR-10 and CIFAR-100 that can excel over adversarial training in both robustness and accuracy on clean examples.

By demonstrating the effectiveness of these simple regularization methods, we hope this work can help make robust training easier and more accessible to practitioners.

Similarly to what we observed about adversarial training on MNIST, adversarial training on CIFAR-10 works by greatly shrinking the adversarial gradients and also shrinking the logit gaps.

The shrink in the gradients is much more dramatic than the shrink in the logit gap, and overwhelms the decrease in the numerator of Equation 4.

See FIG4 .

As shown empirically in Table 6 , and analytically using the linear approximation in Equation 4 evaluated in Figure 8 , logit squeezing worsens robustness when Gaussian augmentation is not used.

However, when fused with Gaussian augmentation, logit squeezing achieves good levels of robustness.

This addition of Gaussian augmentation has three observed effects: the gradients get squashed, the logit gap increases, and the gradients get slightly more aligned.

The increase in the logit gaps increases the numerator in Equation 4.

This gives a slight edge to logit squeezing in comparison to label smoothing, that mostly works by decreasing the denominator in Equation 4.

The results for all of our experiments on MNIST are summarized in Table 6 .

As can be seen, Gaussian random augmentation by itself (?? = ?? = 0) is effective in increasing robustness on black-box attacks.

It does not, however, significantly increase robustness on white-box attacks.

Models that are only trained with either logit squeezing or label smoothing without random Gaussian augmentation (?? = 0), can be fooled by an adversary that has knowledge of the model parameters (white-box) with accuracy 100%.

In the black-box setting, they are also not robust.

Increasing the magnitude of the noise (??) generally increases robustness but degrades the performance on the clean examples.

Keeping the number of iterations k constant, for extremely large ?? Table 6 : Accuracy of different models trained on MNIST with a 40 step PGD attack on the crossentropy (X-ent) loss and the Carlini-Wagner (CW) loss under the white-box and black-box threat models.

Attacks are ??? attacks with a maximum perturbation of = 0.3.

The iterative white-box attacks have an initial random step.

The naturally trained model was used for generating the attacks for the black-box threat model.

We use the CW loss for the FGSM attack in the blackbox case.

k is the number of training iterations.

Here we take a deeper look at reguarlized training results for CIFAR-10.

The conclusions that can be drawn in this case are parallel with those of MNIST discussed in Appendix C. It is worth noting that while the results of logit squeezing outperform those from label smoothing in the white-box setting, training with large squeezing coefficient ?? often fails and results in low accuracy on test data.

This breakdown of training rarely happens for label smoothing (even for very large smoothing parameters ??).

BID9 100.00% 87.25% 45.84% 46.90% 56.22% 55.57% Table 7 : White-box attacks on the CIFAR-10 models.

All attacks are ??? attacks with = 8.

For the 20-step PGD, similar to BID9 , we use an initial random perturbation.

While it seems that logit squeezing, label smoothing, and adversarial training have a lot in common when we look at quantities affecting the linear approximation L , we wonder whether they still look similar with respect to other metrics.

Here, we look at the sum of the activations in the logit layer for every logit FIG0 ) and the sum of activations for every neuron of the penultimate layer ( FIG0 ).

The penultimate activations are often seen as the "feature representation" that the neural network learns.

By summing over the absolute value of the activations of all test examples for every neuron in the penultimate layer, we can identify how many of the neurons are effectively inactive.

When we perform natural training, all neurons become active for at least some images.

After adversarial training, this is no longer the case.

Adversarial training is causing the effective dimensionality of the deep feature representation layer to decrease.

One can interpret this as adversarial training learning to ignore features that the adversary can exploit ( 400 out of the 1024 neurons of the penultimate layer are deactivated).

Shockingly, both label smoothing and logit squeezing do the samethey deactivate roughly 400 neurons from the deep feature representation layer.

BID9 63.39%* 64.38%* 63.39%* 64.38%* 67.00% 67.25% Table 8 : Black-box attacks on the CIFAR-10 models.

All attacks are ??? attacks with = 8.

Similar to BID9 , We build 7-step PGD attacks and FGSM attacks for the public adversarial trained model of MadryLab.

We then use the built attacks for attacking the different models.

*: Since we do not have the Madry model, we cannot evaluate it under the PGD attack with and without random initialization and therefore we use the same value that is reported by them for both.

As a sanity check, and to verify that the robustness of our models are not due to degrading the functionality of PGD attacks, here we verify that our models indeed have zero accuracy when the adversary is allowed to make huge perturbations.

In Table 9 by performing an unbounded PGD attack on a sample logit-squeezed model for the CIFAR-10 (?? = 10, ?? = 30, and k = 160k), we verify that our attack is not benefiting from obfuscating the gradients.

Table 9 : The effect of unbounded on the accuracy.

The decline in the accuracy as a sanity check shows that the sample model is at least not completely breaking the PGD attack and is not obfuscating the gradients.

We attack our sample hardened CIFAR-10 model (?? = 10, Gaussian augmentation ?? = 30 model, and k = 160k iterations), by performing many random restarts.

Random restarts can potentially increase the strength of the PGD attack that has a random initialization.

As shown in Table 10 , increasing the number of random restarts does not significantly degrade the robustness.

It should be noted that attacking with more iterations and more random restarts hurts adversarially trained models as well (see the leaderboard in Madry's Cifar10 Github Repository).

Table 10 : The effect of the number of random restarts while generating the adversarial examples on the accuracy of a model trained with the logit squeezing.

It shows that the accuracy plateaus at 9 random restarts.

Another way to increase the strength of an adversary is by increasing the number of PGD steps for the attack.

We notice that increasing the number of steps does not greatly affect the robustness of our sample logit-squeezed model (See Table 11 ).

Table 11 : The effect of the number of steps of the white-box PGD attack on the CW loss (worst case based on TAB3 ) for the model trained in 160k steps with logit squeezing parameters ?? = 10 and ?? = 30 on CIFAR-10 dataset.

The model remains resistant against ??? attacks with = 8 Figure 9 : The cross-entropy landscape of the first eight images of the validation set for the model trained for 160k iteration and hyper-parameters ?? = 10 and ?? = 30.

To plot the loss landscape we take walks in one random direction r 1 ??? Rademacher(0.5) and the adversarial direction a 2 = sign(??? x xent) where xent is the cross-entropy loss.

We plot the cross-entropy (i.e. xent) loss at different points x = x i + 1 ?? r 1 + 2 ?? a 2 .

Where x i is the clean image and ???10 ??? 1 , 2 ??? 10.

As it can be seen moving along the adversarial direction changes the loss value a lot and moving along the random direction does not make any significant major changes.

Similar to Figure 5 , we plot the classification loss landscape surrounding the input images for the first eight images of the validation set.

Unlike in Figure 5 which we changed the clean image along two random directions, in Figure 9 , we wander around the clean image by moving in the adversarial direction and one random direction.

From Figure 9 we observe that the true direction that the classification loss changes is along the adversarial direction which illustrates that the logit-squeezed model is not masking the gradients.

2) with random noise, and logit squeezing (?? = 0.5) with random noise.

In all cases, the noise is Gaussian with ?? = 0.5.

Interestingly, the combination of Gaussian noise and label smoothing, similar to the combination of Gaussian noise and logit squeezing, deactivates roughly 400 neurons.

This is similar to adversarial training.

In some sense it seems that all three methods are causing the "effective" dimensionality of the deep feature representation layer to shrink.

FIG0 : For MNIST, we plot the cumulative sum of activation magnitudes for all neurons in logit layer of a network produced by natural training, adversarial training, natural training with random noise, label smoothing (LS = 0.2) with random noise, and logit squeezing (?? = 0.5) with random noise.

In all cases, the noise is Gaussian with ?? = 0.5.

@highlight

Achieving strong adversarial robustness comparable to adversarial training without training on adversarial examples
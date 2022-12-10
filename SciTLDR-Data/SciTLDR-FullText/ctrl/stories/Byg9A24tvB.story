Previous work shows that adversarially robust generalization requires larger sample complexity, and the same dataset, e.g., CIFAR-10, which enables good standard accuracy may not suffice to train robust models.

Since collecting new training data could be costly, we focus on better utilizing the given data by inducing the regions with high sample density in the feature space, which could lead to locally sufficient samples for robust learning.

We first formally show that the softmax cross-entropy (SCE) loss and its variants convey inappropriate supervisory signals, which encourage the learned feature points to spread over the space sparsely in training.

This inspires us to propose the Max-Mahalanobis center (MMC) loss to explicitly induce dense feature regions in order to benefit robustness.

Namely, the MMC loss encourages the model to concentrate on learning ordered and compact representations, which gather around the preset optimal centers for different classes.

We empirically demonstrate that applying the MMC loss can significantly improve robustness even under strong adaptive attacks, while keeping state-of-the-art accuracy on clean inputs with little extra computation compared to the SCE loss.

The deep neural networks (DNNs) trained by the softmax cross-entropy (SCE) loss have achieved state-of-the-art performance on various tasks (Goodfellow et al., 2016) .

However, in terms of robustness, the SCE loss is not sufficient to lead to satisfactory performance of the trained models.

It has been widely recognized that the DNNs trained by the SCE loss are vulnerable to adversarial attacks (Carlini & Wagner, 2017a; Goodfellow et al., 2015; Kurakin et al., 2017; Papernot et al., 2016) , where human imperceptible perturbations can be crafted to fool a high-performance network.

To improve adversarial robustness of classifiers, various kinds of defenses have been proposed, but many of them are quickly shown to be ineffective to the adaptive attacks, which are adapted to the specific details of the proposed defenses .

Besides, the methods on verification and training provably robust networks have been proposed (Dvijotham et al., 2018a; b; Hein & Andriushchenko, 2017; .

While these methods are exciting, the verification process is often slow and not scalable.

Among the previously proposed defenses, the adversarial training (AT) methods can achieve state-of-the-art robustness under different adversarial settings Zhang et al., 2019b) .

These methods either directly impose the AT mechanism on the SCE loss or add additional regularizers.

Although the AT methods are relatively strong, they could sacrifice accuracy on clean inputs and are computationally expensive (Xie et al., 2019) .

Due to the computational obstruction, many recent efforts have been devoted to proposing faster verification methods Xiao et al., 2019) and accelerating AT procedures (Shafahi et al., 2019; Zhang et al., 2019a) .

However, the problem still remains.

show that the sample complexity of robust learning can be significantly larger than that of standard learning.

Given the difficulty of training robust classifiers in practice, they further postulate that the difficulty could stem from the insufficiency of training samples in the commonly used datasets, e.g., CIFAR-10 (Krizhevsky & Hinton, 2009) .

Recent work intends to solve this problem by utilizing extra unlabeled data (Carmon et al., 2019; Stanforth et al., 2019) , while we focus on the complementary strategy to exploit the labeled data in hand more efficiently.

Note that although the samples in the input space are unchangeable, we could instead manipulate the local sample distribution, i.e., sample density in the feature space via appropriate training objectives.

Intuitively, by inducing high-density feature regions, there would be locally sufficient samples to train robust classifiers and return reliable predictions .

!

""# ∈ %&, %& + ∆% (low sample density) !

""# ∈ %

*, %* + ∆% (high sample density) + , * !.#/ ∈ %&, %& + ∆% (medium sample density) !

.#/ ∈ %*, %* + ∆% (medium sample density)

Learned features of training data with label 0

Prefixed feature center of label 0 in ℒ223 Contours of the objective loss (45 > 47, ∆4 is a small value)

Moving directions of learned features during training Similar to our attempt to induce high-density regions in the feature space, previous work has been proposed to improve intra-class compactness.

Contrastive loss (Sun et al., 2014) and triplet loss (Schroff et al., 2015) are two classical objectives for this purpose, but the training iterations will dramatically grow to construct image pairs or triplets, which results in slow convergence and instability.

The center loss (Wen et al., 2016) avoids the pair-wise or triplet-wise computation by minimizing the squared distance between the features and the corresponding class centers.

However, since the class centers are updated w.r.t.

the learned features during training, the center loss has to be jointly used with the SCE loss to seek for a trade-off between inter-class dispersion and intra-class compactness.

Therefore, the center loss cannot concentrate on inducing strong intra-class compactness to construct high-density regions and consequently could not lead to reliable robustness, as shown in our experiments.

In this paper, we first formally analyze the sample density distribution induced by the SCE loss and its other variants Wan et al., 2018) in Sec. 3.2, which demonstrates that these previously proposed objectives convey unexpected supervisory signals on the training points, which make the learned features tend to spread over the space sparsely.

This undesirable behavior mainly roots from applying the softmax function in training, which makes the loss function only depend on the relative relation among logits and cannot directly supervise on the learned representations.

We further propose a novel training objective which can explicitly induce high-density regions in the feature space and learn more structured representations.

To achieve this, we propose the MaxMahalanobis center (MMC) loss (detailed in Eq. (8)) as the substitute of the SCE loss.

Specifically, in the MMC loss, we first preset untrainable class centers with optimal inter-class dispersion in the feature space according to , then we encourage the features to gather around the centers by minimizing the squared distance similar with the center loss.

The MMC loss can explicitly control the inter-class dispersion by a single hyperparameter, and further concentrate on improving intra-class compactness in the training procedure to induce high-density regions, as intuitively shown in Fig. 1 .

Behind the simple formula, the MMC loss elegantly combines the favorable merits of the previous methods, which leads to a considerable improvement on the adversarial robustness.

In experiments, we follow the suggestion by that we test under different threat models and attacks, including the adaptive attacks on MNIST, CIFAR-10, and CIFAR-100 (Krizhevsky & Hinton, 2009; LeCun et al., 1998 ).

The results demonstrate that our method can lead to reliable robustness of the trained models with little extra computation, while maintaining high clean accuracy with faster convergence rates compared to the SCE loss and its variants.

When combined with the existing defense mechanisms, e.g., the AT methods , the trained models can be further enhanced under the attacks different from the one used to craft adversarial examples for training.

This section first provides the notations, then introduces the adversarial attacks and threat models.

In this paper, we use the lowercases to denote variables and the uppercases to denote mappings.

Let L be the number of classes, we define the softmax function

, where [L] := {1, · · · , L} and h is termed as logit.

A deep neural network (DNN) learns a non-linear mapping from the input x ∈ R p to the feature z = Z(x) ∈ R d .

One common training objective for DNNs is the softmax cross-entropy (SCE) loss:

for a single input-label pair (x, y), where 1 y is the one-hot encoding of y and the logarithm is defined as element-wise.

Here W and b are the weight matrix and bias vector of the SCE loss, respectively.

Previous work has shown that adversarial examples can be easily crafted to fool DNNs (Biggio et al., 2013; Nguyen et al., 2015; Szegedy et al., 2014) .

A large amount of attacking methods on generating adversarial examples have been introduced in recent years (Carlini & Wagner, 2017a; Chen et al., 2017; Goodfellow et al., 2015; Ilyas et al., 2018; Kurakin et al., 2017; Papernot et al., 2016; Uesato et al., 2018) .

Given the space limit, we try to perform a comprehensive evaluation by considering five different threat models and choosing representative attacks for each threat model following :

White-box l ∞ distortion attack: We apply the projected gradient descent (PGD) method, which is efficient and widely studied in previous work (Pang et al., 2019) .

White-box l 2 distortion attack: We apply the C&W (Carlini & Wagner, 2017a) method, which has a binary search mechanism on its parameters to find the minimal l 2 distortion for a successful attack.

Black-box transfer-based attack: We use the momentum iterative method (MIM) that is effective on boosting adversarial transferability .

Black-box gradient-free attack: We choose SPSA (Uesato et al., 2018) since it has broken many previously proposed defenses.

It can still perform well even when the loss is difficult to optimize.

General-purpose attack: We also evaluate the general robustness of models when adding Gaussian noise (Gilmer et al., 2019) or random rotation on the input images.

Furthermore, to exclude the false robustness caused by, e.g., gradient mask , we modify the above attacking methods to be adaptive attacks (Carlini & Wagner, 2017b; Herley & Van Oorschot, 2017) when evaluating on the robustness of our method.

The adaptive attacks are much more powerful than the non-adaptive ones, as detailed in Sec. 4.2.

Various theoretical explanations have been developed for adversarial examples Ilyas et al., 2019; Papernot et al., 2018) .

In particular, show that training robust classifiers requires significantly larger sample complexity compared to that of training standard ones, and they further postulate that the difficulty of training robust classifiers stems from, at least partly, the insufficiency of training samples in the common datasets.

Recent efforts propose alternatives to benefit training with extra unlabeled data (Carmon et al., 2019; Stanforth et al., 2019) , while we explore the complementary way to better use the labeled training samples for robust learning.

Although a given sample is fixed in the input space, we can instead manipulate the local sample distribution, i.e., sample density in the feature space, via designing appropriate training objectives.

Intuitively, by inducing high-density regions in the feature space, it can be expected to have locally sufficient samples to train robust models that are able to return reliable predictions.

In this section, we first formally define the notion of sample density in the feature space.

Then we provide theoretical analyses of the sample density induced by the SCE loss and its variants.

Finally, we propose our new Max-Mahalanobis center (MMC) loss and demonstrate its superiority compared to previous losses.

Given a training dataset D with N input-label pairs, and the feature mapping Z trained by the objective L(Z(x), y) on this dataset, we define the sample density nearby the feature point z = Z(x) following the similar definition in physics (Jackson, 1999) as

Here Vol(·) denotes the volume of the input set, ∆B is a small neighbourhood containing the feature point z, and ∆N = |Z(D)

∩ ∆B| is the number of training points in ∆B, where Z(D) is the set of all mapped features for the inputs in D. Note that the mapped feature z is still of the label y.

In the training procedure, the feature distribution is directly induced by the training loss L, where minimizing the loss value is the only supervisory signal for the feature points to move (Goodfellow et al., 2016) .

This means that the sample density varies mainly along the orthogonal direction w.r.t.

the loss contours, while the density along a certain contour could be approximately considered as the same.

For example, in the right panel of Fig. 1 , the sample density induced by our MMC loss (detailed in Sec. 3.3) changes mainly along the radial direction, i.e., the directions of red arrows, where the loss contours are dashed concentric circles.

Therefore, supposing L(z, y) = C, we choose

where ∆C > 0 is a small value.

Then Vol(∆B) is the volume between the loss contours of C and C + ∆C for label y in the feature space.

Generalized SCE loss.

To better understand how the SCE loss and its variants Wan et al., 2018) affect the sample density of features, we first generalize the definition in Eq. (1) as:

where the logit h = H(z) ∈ R L is a general transformation of the feature z, for example, h = W z +b in the SCE loss.

We call this family of losses as the generalized SCE (g-SCE) loss.

Wan et al. (2018) propose the large-margin Gaussian Mixture (L-GM) loss, where

under the assumption that the learned features z distribute as a mixture of Gaussian.

Here µ i and Σ i are extra trainable means and covariance matrices respectively, m is the margin, and δ i,y is the indicator function.

propose the Max-Mahalanobis linear discriminant analysis (MMLDA) loss, where

under the similar mixture of Gaussian assumption, but the main difference is that µ * i are not trainable, but calculated before training with optimal inter-class dispersion.

These two losses both fall into the family of the g-SCE loss with quadratic logits:

where B i are the bias variables.

Besides, note that for the SCE loss, there is

According to Eq. (4), the SCE loss can also be regraded as a special case of the g-SCE loss with quadratic logits, where

2 and Σ i = I are identity matrices.

Therefore, later when we refer to the g-SCE loss, we assume that the logits are quadratic as in Eq. (4) by default.

The contours of the g-SCE loss.

To provide a formal representation of the sample density induced by the g-SCE loss, we first derive the formula of the contours, i.e., the closed-form solution of L g-SCE (Z(x), y) = C in the space of z, where C ∈ (0, +∞) is a given constant.

Let C e = exp(C) ∈ (1, +∞), from Eq. (3), we can represent the contours as the solution of

The function in Eq. (5) does not provide an intuitive closed-form solution for the contours, since the existence of the term log l =y exp(h l ) .

However, note that this term belongs to the family of Log-Sum-Exp (LSE) function, which is a smooth approximation to the maximum function (Nesterov, 2005; Nielsen & Sun, 2016) .

Therefore, we can locally approximate the function in Eq. (5) with

whereỹ = arg max l =y h l .

In the following text, we apply colored characters with tilde likeỹ to better visually distinguish them.

According to Eq. (6), we can define L y,ỹ (z) = log[exp(hỹ − h y ) + 1] as the local approximation of the g-SCE loss nearby the feature point z, and substitute the neighborhood

For simplicity, we assume scaled identity covariance matrix in Eq. (4), i.e., Σ i = σ i I, where σ i > 0 are scalars.

Through simple derivations (detailed in Appendix A.1), we show that if σ y = σỹ, the solution of L y,ỹ (z) = C is a (d − 1)-dimensional hypersphere with the center M y,ỹ = (σ y −σỹ) −1 (σ y µ y −σỹµỹ); otherwise if σ y = σỹ, the hypersphere-shape contour will degenerate to a hyperplane.

Figure 2: Intuitive illustration on the inherent limitations of the g-SCE loss.

Reasonably learned features for a classification task should distribute in clusters, so it is counter-intuitive that the feature points tend to move to infinity to pursue lower loss values when applying the g-SCE loss.

In contrast, MMC induces models to learn more structured and orderly features.

The induced sample density.

Since the approximation in Eq. (6) depends on the specific y andỹ, we define the training subset

includes the data with the true label of class k, while the highest prediction returned by the classifier is classk among other classes.

Then we can derive the approximated sample density in the feature space induced by the g-SCE loss, as stated in the following theorem:

, and σ k = σk, then the sample density nearby the feature point z based on the approximation in Eq. (6) is

, and

where for the input-label pair in

Limitations of the g-SCE loss.

Based on Theorem 1 and the approximation in Eq. (6), let

* will act as a tight lower bound for C, i.e., the solution set of C < C * is empty.

This will make the training procedure tend to avoid this case since the loss C cannot be further minimized to zero, which will introduce unnecessary biases on the returned predictions.

On the other hand, if σ k < σk, C could be minimized to zero.

However, when C → 0, the sample density will also tend to zero since there is B k,k + log(Ce−1) σ k −σk → ∞, which means the feature point will be encouraged to go further and further from the hypersphere center M k,k only to make the loss value C be lower, as intuitively illustrated in Fig. 2(a) .

This counter-intuitive behavior mainly roots from applying the softmax function in training.

Namely, the softmax normalization makes the loss value only depend on the relative relation among logits.

This causes indirect and unexpected supervisory signals on the learned features, such that the points with low loss values tend to spread over the space sparsely.

Fortunately, in practice, the feature point will not really move to infinity, since the existence of batch normalization layers (Ioffe & Szegedy, 2015) , and the squared radius from the center M k,k increases as O(| log C|) when minimizing the loss C. These theoretical conclusions are consistent with the empirical observations on the two-dimensional features in previous work (cf.

Fig. 1 in Wan et al. (2018) ).

Another limitation of the g-SCE loss is that the sample density is proportional to N k,k , which is on average N/L 2 .

For example, there are around 1.3 million training data in ImageNet (Deng et al., 2009 ), but with a large number of classes L = 1, 000, there are averagely less than two samples in each D k,k .

These limitations inspire us to design the new training loss as in Sec 3.3.

Remark 1.

If σ k = σk (e.g., as in the SCE loss), the features with loss values in [C, C + ∆C] will be encouraged to locate between two hyperplane contours without further supervision, and consequently there will not be explicit supervision on the sample density as shown in the left panel of Fig. 1 .

Remark 2.

Except for the g-SCE loss, Wen et al. (2016) propose the center loss in order to improve the intra-class compactness of learned features, formulated as L Center (Z(x), y) = 1 2 z − µ y 2 2 .

Here the center µ y is updated based on a mini-batch of learned features with label y in each training iteration.

The center loss has to be jointly used with the SCE loss as L SCE + λL Center , since simply supervise the DNNs with the center loss term will cause the learned features and centers to degrade to zeros (Wen et al., 2016) .

This makes it difficult to derive a closed-form formula for the induced sample density.

Besides, the center loss method cannot concentrate on improving intra-class compactness, since it has to seek for a trade-off between inter-class dispersion and intra-class compactness.

Inspired by the above analyses, we propose the Max-Mahalanobis center (MMC) loss to explicitly learn more structured representations and induce high-density regions in the feature space.

The MMC loss is defined in a regression form without the softmax function as

Here L] are the centers of the Max-Mahalanobis distribution (MMD) .

The MMD is a mixture of Gaussian distribution with identity covariance matrix and preset centers µ * , where µ * l 2 = C MM for any l ∈ [L], and C MM is a hyperparameter.

These MMD centers are invariable during training, which are crafted according to the criterion: µ * = arg min µ max i =j µ i , µ j .

Intuitively, this criterion is to maximize the minimal angle between any two centers, which can provide optimal inter-class dispersion as shown in .

In Appendix B.1, we provide the generation algorithm for µ * in MMC.

We derive the sample density induced by the MMC loss in the feature space, as stated in Theorem 2.

Similar to the previously introduced notations, here we define the subset D k = {(x, y) ∈ D|y = k} and

and L MMC (z, y) = C, the sample density nearby the feature point z is

where for the input-label pair in D k , there is L MMC ∼ p k (c).

According to Theorem 2, there are several attractive merits of the MMC loss, as described below.

Inducing higher sample density.

Compared to Theorem 1, the sample density induced by MMC is proportional to N k rather than N k,k , where N k is on average N/L. It facilitates producing higher sample density.

Furthermore, when the loss value C is minimized to zero, the sample density will exponentially increase according to Eq. (9), as illustrated in Fig. 2(b) .

The right panel of Fig. 1 also provides an intuitive insight on this property of the MMC loss: since the loss value C is proportional to the squared distance from the preset center µ * y , the feature points with lower loss values are certain to locate in a smaller volume around the center.

Consequently, the feature points of the same class are encouraged to gather around the corresponding center, such that for each sample, there will be locally enough data in its neighborhood for robust learning .

The MMC loss value also becomes a reliable metric of the uncertainty on returned predictions.

Better exploiting model capacity.

Behind the simple formula, the MMC loss can explicitly monitor inter-class dispersion by the hyperparameter C MM , while enabling the network to concentrate on minimizing intra-class compactness in training.

Instead of repeatedly searching for an internal tradeoff in training as the center loss, the monotonicity of the supervisory signals induced by MMC can better exploit model capacity and also lead to faster convergence, as empirically shown in Fig. 3(a) .

Avoiding the degradation problem.

The MMC loss can naturally avoid the degradation problem encountered in Wen et al. (2016) when the center loss is not jointly used with the SCE loss, since the preset centers µ * for MMC are untrainable.

In the test phase, the network trained by MMC can still return a normalized prediction with the softmax function.

More details about the empirical superiorities of the MMC loss over other previous losses are demonstrated in Sec. 4.

Remark 3.

In Appendix B.2, we discuss on why the squared-error form in Eq. (8) is preferred compared to, e.g., the absolute form or the Huber form in the adversarial setting.

We further introduce flexible variants of the MMC loss in Appendix B.3, which can better adapt to various tasks.

. (10) Note that there is Σ i = 1 2 I in Eq. (4) for the MMLDA loss, similar with the SCE loss.

Thus the MMLDA method cannot explicitly supervise on the sample density and induce high-density regions in the feature space, as analyzed in Sec. 3.2.

Compared to the MMLDA method, the MMC loss introduces extra supervision on intra-class compactness, which facilitates better robustness.

In this section, we empirically demonstrate several attractive merits of applying the MMC loss.

We experiment on the widely used MNIST, CIFAR-10, and CIFAR-100 datasets (Krizhevsky & Hinton, 2009; LeCun et al., 1998) .

The main baselines for MMC is SCE (He et al., 2016) , Center loss (Wen et al., 2016) , MMLDA , and L-GM (Wan et al., 2018) .

The network architecture applied is ResNet-32 with five core layer blocks (He et al., 2016) .

Here we use MMC-10 to indicate the MMC loss with C MM = 10, where C MM is assigned based on the cross-validation results in .

The hyperparameters for the center loss, L-GM loss and the MMLDA method all follow the settings in the original papers Wan et al., 2018; Wen et al., 2016) .

The pixel values are scaled to the interval [0, 1].

For each training loss with or without the AT mechanism, we apply the momentum SGD (Qian, 1999) optimizer with the initial learning rate of 0.01, and train for 40 epochs on MNIST, 200 epochs on CIFAR-10 and CIFAR-100.

The learning rate decays with a factor of 0.1 at 100 and 150 epochs, respectively.

When applying the AT mechanism , the adversarial examples for training are crafted by 10-steps targeted or untargeted PGD with = 8/255.

In Fig. 3(a) , we provide the curves of the test error rate w.r.t.

the training time.

Note that the MMC loss induces faster convergence rate and requires little extra computation compared to the SCE loss and its variants, while keeping comparable performance on the clean images.

In comparison, implementing the AT mechanism is computationally expensive in training and will sacrifice the accuracy on the clean images.

As stated in , only applying the existing attacks with default hyperparameters is not sufficient to claim reliable robustness.

Thus, we apply the adaptive versions of existing attacks when evading the networks trained by the MMC loss (detailed in Appendix B.4).

For instance, the non-adaptive objectives for PGD are variants of the SCE loss , while the adaptive objectives are −L MMC (z, y) and L MMC (z, y t ) in the untargeted and targeted modes for PGD, respectively.

Here y t is the target label.

To verify that the adaptive attacks are more effective than the non-adaptive ones, we modify the network architecture with a two-dimensional feature layer and visualize the PGD attacking procedure in Fig. 3(b) .

The two panels separately correspond to two randomly selected clean inputs indicated by black stars.

The ten colored clusters in each panel consist of the features of all the 10,000 test samples in MNIST, where each color corresponds to one class.

We can see that the adaptive attacks are indeed much more efficient than the non-adaptive one.

We first investigate the white-box l ∞ distortion setting using the PGD attack, and report the results in Table 1 .

According to , we evaluate under different combinations of the attacking parameters: the perturbation , iteration steps, and the attack mode, i.e., targeted or untargeted.

Following the setting in , we choose the perturbation = 8/255 and 16/255, with the step size be 2/255.

We have also run PGD-100 and PGD-200 attacks, and find that the accuracy converges compared to PGD-50.

In each PGD experiment, we ran several times with different random restarts to guarantee the reliability of the reported results.

Ablation study.

To investigate the effect on robustness induced by high sample density in MMC, we substitute uniformly sampled center set Duan et al., 2019) , i.e., µ r = {µ r l } l∈ [L] for the MM center set µ * , and name the resulted method as "MMC-10 (rand)" as shown in Table 1 .

There is also µ r l 2 = C MM , but µ r is no longer the solution of the min-max problem in Sec. 3.3.

From the results in Table 1 , we can see that higher sample density alone in "MMC-10 (rand)" can already lead to much better robustness than other baseline methods even under the adaptive attacks, while using the optimal center set µ * as in "MMC-10" can further improve performance.

When combining with the AT mechanism, the trained models have better performance under the attacks different from the one used to craft adversarial examples for training, e.g, PGD un 50 with = 16/255.

Then we investigate the white-box l 2 distortion setting.

We apply the C&W attack, where it has a binary search mechanism to find the minimal distortion to successfully mislead the classifier under the untargeted mode, or lead the classifier to predict the target label in the targeted mode.

Following the suggestion in Carlini & Wagner (2017a) , we set the binary search steps to be 9 with the initial constant c = 0.01.

The iteration steps for each value of c are set to be 1,000 with the learning rate of 0.005.

In the Part I of Table 2 , we report the minimal distortions found by the C&W attack.

As expected, it requires much larger distortions to successfully evade the networks trained by MMC.

As suggested in , providing evidence of being robust against the black-box attacks is critical to claim reliable robustness.

We first perform the transfer-based attacks using PGD and MIM.

Since the targeted attacks usually have poor transferability , we only focus on the untargeted mode in this case, and the results are shown in Fig. 4 .

We further perform the gradient-free attacks using the SPSA method and report the results in the Part II of Table 2 .

To perform numerical approximations on gradients in SPSA, we set the batch size to be 128, the learning rate is 0.01, and the step size of the finite difference is δ = 0.01, as suggested by Uesato et al. (2018) .

We also evaluate under stronger SPSA attacks with batch size to be 4096 and 8192 in Table 3 , where the = 8/255.

With larger batch sizes, we can find that the accuracy under the black-box SPSA attacks converges to it under the white-box PGD attacks.

These results indicate that training with the MMC loss also leads to robustness under the black-box attacks, which verifies that our method can induce reliable robustness, rather than the false one caused by, e.g., gradient mask .

To show that our method is generally robust, we further test under the general-purpose attacks .

We apply the Gaussian noise Gilmer et al., 2019) and rotation transformation , which are not included in the data augmentation for training.

The results are given in the Part III of Table 2 .

Note that the AT methods are less robust to simple transformations like rotation, as also observed in previous work .

In comparison, the models trained by the MMC loss are still robust to these easy-to-apply attacks.

In Table 4 and Table 5 , we provide the results on CIFAR-100 under the white-box PGD and C&W attacks, and the black-box gradient-free SPSA attack.

The hyperparameter setting for each attack is the same as it on CIFAR-10.

Compared to previous defense strategies which also evaluate on CIFAR-100 (Pang et al., 2019; Mustafa et al., 2019) , MMC can improve robustness more significantly, while keeping better performance on the clean inputs.

Compared to the results on CIFAR-10, the averaged distortion of C&W on CIFAR-100 is larger for a successful targeted attack and is much smaller for a successful untargeted attack.

This is because when only the number of classes increases, e.g., from 10 to 100, it is easier to achieve a coarse untargeted attack, but harder to make a subtle targeted attack.

Note that in Table 5 , we also train on the ResNet-110 model with eighteen core block layers except for the ResNet-32 model.

The results show that MMC can further benefit from deep network architectures and better exploit model capacity to improve robustness.

Similar properties are also observed in previous work when applying the AT methods .

In contrast, as shown in Table 5 , the models trained by SCE are comparably sensitive to adversarial perturbations for different architectures, which demonstrates that SCE cannot take full advantage of the model capacity to improve robustness.

This verifies that MMC provides effective robustness promoting mechanism like the AT methods, with much less computational cost.

In this paper, we formally demonstrate that applying the softmax function in training could potentially lead to unexpected supervisory signals.

To solve this problem, we propose the MMC loss to learn more structured representations and induce high-density regions in the feature space.

In our experiments, we empirically demonstrate several favorable merits of our method: (i) Lead to reliable robustness even under strong adaptive attacks in different threat models; (ii) Keep high performance on clean inputs comparable TO SCE; (iii) Introduce little extra computation compared to the SCE loss; (iv) Compatible with the existing defense mechanisms, e.g., the AT methods.

Our analyses in this paper also provide useful insights for future work on designing new objectives beyond the SCE framework.

In this section, we provide the proof of the theorems proposed in the paper.

According to the definition of sample density

we separately calculate ∆N and Vol(∆B).

Since

Now we calculate Vol(∆B) by approximating it with Vol(∆B y,ỹ ).

We first derive the solution of L y,ỹ = C. For simplicity, we assume scaled identity covariance matrix, i.e., Σ i = σ i I, where

Note that each value of c corresponds to a specific contour, where M i,j and B i,j can be regraded as constant w.r.t.

c.

When B i,j < (σ i − σ j ) −1 c, the solution set becomes empty.

Specially, if σ i = σ j = σ, the hypersphere-shape contour will degenerate to a hyperplane:

For example, for the SCE loss, the solution of the contour is z (W i − W j ) = b j − b i + c. For more general Σ i , the conclusions are similar, e.g., the solution in Eq. (12) will become a hyperellipse.

Now it easy to show that the solution of L y,ỹ = C when y = k,ỹ =k is the hypersphere:

According to the formula of the hypersphere surface area (Loskot & Beaulieu, 2007) , the volume of ∆B y,ỹ is Vol(∆B y,ỹ ) = 2π

where Γ(·) is the gamma function.

Finally we can approximate the sample density as

A.2 PROOF OF THEOREM 2

Similar to the proof of Theorem 1, there is

Unlike for the g-SCE, we can exactly calculate Vol(∆B) for the MMC loss.

Note that the solution of L MMC = C is the hypersphere: According to the formula of the hypersphere surface area (Loskot & Beaulieu, 2007) , we have

where Γ(·) is the gamma function.

Finally we can obtain the sample density as

B TECHNICAL DETAILS

In this section, we provide more technical details we applied in our paper.

Most of our experiments are conducted on the NVIDIA DGX-1 server with eight Tesla P100 GPUs.

We give the generation algorithm for crafting the Max-Mahalanobis Centers in Algorithm 1, proposed by .

Note that there are two minor differences from the originally proposed algorithm.

First is that in they use C = µ i 2 2 , while we use C MM = µ i 2 .

Second is that we denote the feature z ∈ R d , while they denote z ∈ R p .

The Max-Mahalanobis centers generated in the low-dimensional cases are quite intuitive and comprehensible as shown in Fig. 5 .

For examples, when L = 2, the Max-Mahalanobis centers are the two vertexes of a line segment; when L = 3, they are the three vertexes of an equilateral triangle; when L = 4, they are the four vertexes of a regular tetrahedron.

according to the tree structure.

Specifically, we first assign a virtual center (i.e., the origin) to the root node.

For any child node n c in the tree, we denote its parent node as n p , and the number of its brother nodes as L c .

We locally generate a set of MM centers as

, where s is the depth of the child node n c , C s is a constant with smaller values for larger s.

Then we assign the virtual center to each child node of n p from µ np + µ (s,Lc) , i.e., a shifted set of crafted MM centers, where µ np is the virtual center assigned to n p .

If the child node n c is a leaf node, i.e., it correspond to a class label l, then there is µ Ada = L MMC (z, y t )−L MMC (z, y), where y t is the targeted label,ỹ is generally the highest predicted label except for y as defined in Sec. 3.2.

These objectives refer to previous work by Carlini & Wagner (2017a; b) .

Specifically, the adaptive objectives L Ada are used in the C&W attacks.

In Fig. 6 , we demonstrate the attacking mechanisms induced by different adaptive adversarial objectives.

Note that we only focus on the gradients and ignore the specific method which implements the attack.

Different adaptive objectives are preferred under different adversarial goals.

For examples, when decreasing the confidence of the true label is the goal, L un,1

Ada is the optimal choice; in order to mislead the classifier to predict an untrue label or the target label, L There are many previous work in the face recognition area that focus on angular margin-based softmax (AMS) losses (Liu et al., 2016; Liang et al., 2017; Wang et al., 2018; Deng et al., 2019) .

They mainly exploit three basic operations: weight normalization (WN), feature normalization (FN), and angular margin (AN).

It has been empirically shown that WN can benefit the cases with unbalanced data (Guo & Zhang, 2017) ; FN can encourage the models to focus more on hard examples ; AN can induce larger inter-class margins and lead to better generalization in different facial tasks (Wang et al., 2018; Deng et al., 2019) .

However, there are two critical differences between our MMC loss and these AMS losses:

Difference one: The inter-class margin

• The AMS losses induce the inter-class margins mainly by encouraging the intra-class compactness, while the weights are not explicitly forced to have large margins (Qi & Zhang, 2018 ).

• The MMC loss simultaneously fixes the class centers to be optimally dispersed and encourages the intra-class distribution to be compact.

Note that both of the two mechanisms can induce inter-class margins, which can finally lead to larger inter-class margins compared to the AMS losses.

Difference two: The normalization

• The AMS losses use both WN and FN to exploit the angular metric, which makes the normalized features distribute on hyperspheres.

The good properties of the AMS losses are at the cost of abandoning the radial degree of freedom, which may reduce the capability of models.

• In the MMC loss, there is only WN on the class centers, i.e., µ * y = C MM , and we leave the degree of freedom in the radial direction for the features to keep model capacity.

However, note that the MMC loss z − µ * y 2 2 ≥ ( z 2 − C MM ) 2 is a natural penalty term on the feature norm, which encourage z 2 to not be far from C MM .

This prevents models from increasing feature norms for easy examples and ignoring hard examples, just similar to the effect caused by FN but more flexible.

<|TLDR|>

@highlight

Applying the softmax function in training leads to indirect and unexpected supervision on features. We propose a new training objective to explicitly induce dense feature regions for locally sufficient samples to benefit adversarial robustness.
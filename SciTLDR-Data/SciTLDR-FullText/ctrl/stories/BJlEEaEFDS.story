Batch Normalization (BatchNorm) has shown to be effective for improving and accelerating the training of deep neural networks.

However, recently it has been shown that it is also vulnerable to adversarial perturbations.

In this work, we aim to investigate the cause of adversarial vulnerability of the BatchNorm.

We hypothesize that the use of different normalization statistics during training and inference (mini-batch statistics for training and moving average of these values at inference) is the main cause of this adversarial vulnerability in the BatchNorm layer.

We empirically proved this by experiments on various neural network architectures and datasets.

Furthermore, we introduce Robust Normalization (RobustNorm) and experimentally show that it is not only resilient to adversarial perturbation but also inherit the benefits of BatchNorm.

In spite of their impressive performance on challenging tasks in computer vision such as image classification and semantic segmentation, deep neural networks (DNNs) are shown to be highly vulnerable to adversarial examples, i.e. carefully crafted samples which look similar to natural images but designed to mislead a trained neural network model (Goodfellow et al., 2014; Nguyen et al., 2015; Carlini & Wagner, 2017) .

Designing defense mechanisms against these adversarial perturbations has been subjected to much research recently (Xie et al., 2019; Madry et al., 2017; Tramèr et al., 2017; Papernot et al., 2016) .

Meanwhile, Batch Normalization (BatchNorm or BN) (Ioffe & Szegedy, 2015) has successfully proliferated throughout all areas of deep learning as it enables stable training, higher learning rates, faster convergence, and higher generalization accuracy.

Initially, the effectiveness of the BatchNorm has been attributed to its ability to eliminate the internal covariate shift (ICS), the tendency of the distribution of activations to drift during training.

However, later on, alternative reasons including avoiding exploding activations, smooth loss landscape, reducing the sensitivity to initialization, etc.

have also been proposed as the basis of BatchNorm's success (Santurkar et al., 2018; Bjorck et al., 2018; Luo et al., 2018) .

While there exist a plethora of reasons for the adversarial vulnerability of deep neural networks (Jacobsen et al., 2018; Simon-Gabriel et al., 2018) , a recent study by Galloway et al. (2019) showed that BatchNorm is one of the reasons for this vulnerability.

Specifically, they empirically showed that removing the BatchNorm layer enhances robustness against adversarial perturbations.

However, removal of BatchNorm also means a sacrifice of benefits such as the use of higher learning rates, faster convergence, and significant improvement in the clean test set accuracy among many others.

In this paper, we propose a new perspective regarding the adversarial vulnerability of the BatchNorm layer.

Specifically, we probe why BatchNorm layer causes the adversarial vulnerability.

We hypothesize that the use of different normalization statistics during training and inference phase (mini-batch statistics for training and moving average of these statistics also called tracking, at inference time) is the cause of this adversarial vulnerability of the BatchNorm layer.

Our experiments show that by removing this part of the BatchNorm, the robustness of the network increases by 20%.

Similarly, robustness can further be enhanced by up to 30% after adversarial training.

However, by removing the tracking part, the test accuracy on the clean images drops significantly ( though better than without normalization).

To circumvent this issue, we propose Robust Normalization (RobustNorm or RN).

Our experiments demonstrate that RobustNorm not only significantly improve the test performance of adversarially-trained DNNs but is also able to achieve the comparable test accuracy to that of BatchNorm on unperturbed datasets.

We perform numerical experiments over standard datasets and DNN architectures.

In almost all of our experiments, we obtain a better adversarial robustness performance on perturbed examples for training with natural as well as adversarial training.

We consider a standard classification task for data, having underlying distribution denoted as D, over the pair of examples x ∈ R n and corresponding true labels y ∈ {1, 2, ..., k} where k represents different labels.

We denote deep neural network (DNN) as a function F θ (x), where θ denotes trainable parameters.

θ is learned by minimizing a loss function L(x, y) with training data x, y. The output of the DNN is a feature representation f ∈ R d , that we give input to a classifier C : R n → {1, 2, ..., k}. The objective of the adversary is to add the additive perturbation δ ∈ R n under the constrain that the generated adversarial sample x adv = x + δ that looks visually similar to the true image x, and for which the corresponding labels are not same i.e. C(x) = C(x adv ).

In this work, we have added the perturbation via following well-known adversarial attack approaches.

Fast Gradient Sign Method:

Given an input image x along with its corresponding true label y, FGSM Goodfellow et al. (2014) aims to generate the adversarial image x adv as,

where is the perturbation budget that is chosen to be sufficiently small.

We use two of its variants: Gradient (Grad) where graidents are used and Gradient sign (GradSign) which is similar to 1.

Basic Iterative Method (BIM): BIM (Kurakin et al., 2016 ) is a straight forward extension of FGSM, that applies it multiple times with a smaller step size.

Specifically,

where x 0 adv is the clean image and N denotes iteration number.

Carlini-Wagner attack (CW): CW is an effective optimization-based attack model introduced by Carlini & Wagner (2017) .

It works by definining an auxilary variable ϑ and minimizes the following objective functions min

where 1 2 (tanh(ϑ) + 1) − x is the perturbation δ, c is a scalar constant, and f(.) is defined as:

Here, is to control the adversarial sample's confidence and Z x adv are the logits values for class k.

Projected Gradient Descent (PGD): PGD perturbs the true image x for total number of N steps with smaller step sizes.

After each step of perturbation, PGD projects the adversarial example back onto the -ball of normal image x , if it goes beyond the -ball.

Specifically,

where Π is the projection operator, α is step size, and x N denotes adversarial example at the N -th step.

We have used ∞ norm as a distance measure.

Gaussian Noise For comparison purposes, we also have used Gaussian noise with 0 mean and 0.25 variance.

It has been shown that empirical risk minimization using only clean images for training can decrease the robustness performance of DNNs.

A standard approach to achieve the adversarial robustness in DNNs is adversarial training which involves fitting a classifier C on adversarially-perturbed samples along with clean images.

We have used PGD based adversarial training which has shown to be effective against many firstorder adversaries (Madry et al., 2017) unlike other methods which overfit for on a single attack.

We have used two network architectures, Resnet He et al. (2016) with 20,38 and 50 layers and VGG Simonyan & Zisserman (2014) with 11 and 16 layers.

We have used CIFAR10 and 100 datasets (Krizhevsky et al., 2009 ) for all the evaluations.

We have used term natural training for training with clean images while adversarial training is done with PGD based method formulated by Madry et al. (2017) .

We have always used a learning rate of 0.1 except for no normalization scenarios where convergence is not possible with higher learning rates.

In such cases, we have used a learning rate of 0.001.

We decrease the learning rate 10 times after 120 epochs and trained all the networks for 164 epochs.

For robustness evaluations, we have used =0.03/1 for most of the experiments and used 20 epochs for all the iterative attacks.

We also have tested the model on different noise levels ranging from 0.003/1 to 0.9/1.

In this section, we briefly explain the working principle of BatchNorm layer.

Broadly speaking, the BatchNorm layer has been introduced to circumvent the issue of internal covariate shift (ICS).

Consider a minibatch B of size M , containing samples x m for m = 1, 2, ..., M .

BatchNorm normalizes the mini-batch during training, by calculating the mean µ β and variance σ 2 β as follows:

Activation normalization is then performed as,

To further compensate for the possible loss of representational ability of network, BatchNorm also learns per-channel linear transformation as:

for trainable parameters γ and β that represent scale and shift, respectively.

These parameters are learnt using the same procedure, such as stochastic gradient descent, as other weights in the network.

During training, the model usually maintains the moving averages of mini-batch means and variances(a.k.a.

tracking), and during inference, uses these tracked statistics in place of the mini-batch statistics.

Formally, tracking (moving average) of mean and variance, for scalar τ are given as,

(9) For inference, we can write,

Recently, Galloway et al. (2019) empirically showed that the accelerated training properties and occasionally higher clean test accuracy of employing BatchNorm in network come at the cost of low robustness to adversarial perturbations.

While removing the BatchNorm layer may be helpful for robustness, it also means the loss of desirable properties of BatchNorm like very high learning rate, faster convergence, boost in test accuracy, etc.

Therefore, it is pertinent to devise a normalization method that is not only robust to the adversarial perturbations but also inherits the desirable properties of the BatchNorm layer.

In this section, we aim to investigate the reasons behind the adversarial vulnerability of the BatchNorm layer on the following two grounds;

• We note that during training, mini-batch statistics are used to normalize activations as shown in Equations 6 and 7.

Moving average of these statistics are also calculated during the training that is called tracking (shown in Equation 9).

The tracked mean and variance are used in the inference step (Equation 10).

In this way, different values for mean and variance are used during training and inference.

We show this in Figure 1 where it is clear that batch statistics at the start of the training are very different from tracked values that are used at the inference.

• Our second observation is based on the recent work of (Ding et al., 2019; Jacobsen et al., 2019) .

These works shed light on the link of the distributional shift in input data and robustness.

Specifically, Ding et al. (2019) showed that adversarial robustness is highly sensitive to change in the input data distribution and prove that even a semantically-lossless shift on the data distribution could result in drastically different robustness for adversarially trained models.

representations being used at training and inference time which causes drift in input distributions of these layers.

Therefore, the tracking part is the main culprit behind the adversarial vulnerability of the BatchNorm layer.

To prove our hypothesis, we have done extensive experiments.

For each experiment, we train a neural network model with three different normalization layers: BatchNorm, BatchNorm without tracking, and no normalization.

To prove the generality of our argument, we have used various architectures, depths, and datasets as written in section 2.1.

We train these networks on clean images as well as with based adversarial training procedure.

For adversarial training, we use PGD attack for perturbation.

We choose PGD due to its ability to generalize well for other adversarial attacks.

Table 1 shows our results on Resnet20.

For detailed experimental results on various architectures, depths and dataset with different attacks see Table 4 , 5 in appendix.

The results clearly show that while the elimination of BatchNorm and training at a very small learning rate can help increase robustness, it also reduces clean data accuracy (with BatchNorm).

More importantly, this proves our hypothesis that by removing tracking, we can increase the robustness of a neural network significantly.

By using BatchNorm without tracking, we also keep many benefits of BatchNorm.

Unfortunately, by eliminating the tracking part of BatchNorm, clean accuracy of a network also reduces as compared to clean accuracy with BatchNorm.

We tackle this issue in the next section.

Although alleviation of ICS was claimed reason for the success of BatchNorm, recently, Bjorck et al. (2018) have shown that BatchNorm works because it avoids activation explosion by repeatedly correcting all activations.

For this reason, it is possible to train networks with large learning rates, as activations cannot grow uncontrollably and convergence becomes easier.

On a different side, recent work on robustness has shown a connection between the removal of outliers in activations and robustness (Xie et al., 2019; Etmann et al., 2019) .

Based on these observations, we use min-max rescaling that is often employed in preprocessing.

This is also useful in the elimination of outliers since it rescales the data to a specific scale, Minmax normalization is defined as; Table 3 : Comparison of clean accuracy of BatchNorm with RobustNorm for both adversarial and natural training scenarios.

RobustNorm's accuracy is better than BatchNorm when tracking is not used while its accuracy is same when tracking is used.

However, we experimentally found this layer to be less effective in terms of convergence.

Considering the importance of mean (Salimans & Kingma, 2016) , we modify this to;

We empirically observe the effectiveness of Equation 12 over Equation 11 but the overall performance was still inadequate.

During debugging, we found that Equation 12 suppress activations much stronger than BatchNorm.

This can also be seen from Popoviciu's inequality (Popoviciu, 1935) ,

Following Popoviciu's inequality, we introduce the hyperparameter 0 < p < 1, that reduces the denominator in Equation 12,

We experimentally found that p = 0.2 value generalizes well for many networks as well as datasets.

We call this normalization Robust Normalization (RobustNorm or RN) due to its robustness properties.

We do not use tracking for the RobustNorm.

But for comparison purposes, we keep running average of both mean and denominator and use this running average during inference and call this normalization RobustNorm with tracking.

Table 3 shows the accuracy of RobustNorm on Resnet20 for clean as well as adversarial training with both CIFAR10 and CIFAR100 datasets with 95% confidence interval calculated over 5 random restarts.

These results show a better clean accuracy of RobustNorm in both natural and adversarial training scenarios.

Apart from this, RobustNorm with tracking also shows better performance compared to BatchNorm with tracking.

For adversarial robustness, we have shown Figure 2 with different attacks on CIFAR100 dataset.

From the Table 3 and Figure 2 , it is clear that RobustNorm keeps its clean accuracy while being more robust.

For more results on Resnet38, Resnet50, VGG11, and VGG16 with CIFRAR10 and CIFAR100 datasets and both natural and adversarial training and many attack methods, please have a look at Table 4 and 5.

Figure3 shows the evolution of validation loss and accuracy for PGD based adversarial training with a confidence interval of 95% on Resnet20 architecture and CIFAR100 dataset.

From Figure 8 in the appendix, it can also be seen that the evolution of training loss and accuracy is normal.

But validation loss and accuracy for normalizations with tracking is much different for different random restarts.

This can probably be explained based on flat and sharp minima attained by different normalizations as can be seen in loss landscape in Figure 9 in the appendix.

For further discussion on the loss landscape, please see section B in appendix.

Figure 4 .

As increases, the robustness of neural network decreases but the robustness of neural network with RobustNorm is much higher than BatchNorm while also higher than BatchNorm w/o tracking.

To see the effect of an increase in adversarial noise on CIFAR100 dataset, see Figure 6 in the appendix.

In the previous sections, we have empirically shown the wickedness of tracking in BatchNorm.

But there is more to the story.

One benefit of tracking that makes it a necessary evil in BatchNorm is its ability to have It is also important to note that RobustNorm's results are restored by increasing batch size by a small number.

Similarly, tracking is also less harmful for RobustNorm and RobustNorm with tracking is still more robust while having all the benefits of BatchNorm.

Addition of maliciously crafted noise in normal inputs, also called adversarial examples has proven to be deceptive for neural networks.

While there are many reasons for this phenomena, recent work has shown BatchNorm to be a cause of this vulnerability as well.

In this paper, we have investigated the reasons behind this issue and found that tracking part of BatchNorm causes this adversarial vulnerability.

Then, we showed that by eliminating it, we can increase the robustness of a neural network.

Afterward, based on the intuitions from the work done for the understanding of BatchNorm, we proposed RobustNorm which has much higher robustness than BatchNorm for both natural as well as adversarial training scenarios.

In the end, we have shown how tracking can be a necessary evil and argued that it requires further careful investigation.

In this section, we provide more detailed results for our experiments for both CIFAR10 and CIFAR100 datasets.

In this section, we put results of increasing adversarial noise on CIFAR100 dataset.

A.3 ANOTHER ASPECT OF ROBUSTNORM

As we have discussed, ICS hypothesis has been negated by a few recent studies.

One of these studies (Santurkar et al., 2018) suggested that based on the results, " it might be valuable to perform a principled exploration of the design space of normalization schemes as it can lead to better performance."

In this way, we can see RobustNorm with tracking as a new normalization scheme which is based on alternative explanations yet having performance equal to BatchNorm which, in a way, weakens ICS hypothesis.

See Figure 7 for a comparison of accuracies over different models for CIFAR10 and CIFAR100 datasets.

In this section, we discuss possible reasons for the high variation of validation loss for a network with normalization with tracking.

As shown in Figure 8 , training loss of all the norms converges with similar Figure 8 : Training and validation loss and accuracy evolution for adversarial training.

Interestingly, BatchNorm's training loss decreases normally, its validation loss has lot more uncertainty and either remaining same or start increasing.

This way, BatchNorm is overfitting.

A similar trend is also shown by RobustNorm when tracking is used thought it vanishes when remove tracking and decrease in loss becomes normal and less uncertain.

fashion for all the random restarts but validation loss has a lot of variation over these restarts.

In other words, the value of training loss is similar among many restarts while the same loss values change drastically for validation.

To further understand it, we draw loss landscape of these networks using formulation given by Li et al. (2018b) in Figure 9 .

From these plots, we observe an interesting behaviour: networks having normalization without tracking(i.e.

better robustness) tend to have sharp minima as can be seen in figures 9c, 9d, 9e, 9f while their counterparts have more flat loss landscape i.e. figures 9a, 9b, 9g, 9h.

There is a long history of debate on generalization ability of sharp vs flat minima Hochreiter & Schmidhuber (1997) ; Keskar et al. (2016) ; Dinh et al. (2017) .

We think more work in this direction can lead to a better understanding of how BatchNorm causes this vulnerability.

In this section, we discuss some less interesting experiments done to understand the role of power in RobustNorm.

In this section, we show the effect of changing hyperparameter p for clean as well as robust accuracy.

From figure 4 and 11, it can be seen that both robustness to many attacks as well as accuracy changing with the power.

So it can be concluded that by tuning hyperparameters, we can get better results.

<|TLDR|>

@highlight

Investigation of how BatchNorm causes adversarial vulnerability and how to avoid it. 

@highlight

This paper addresses vulnerability to adversarial perturbations in BatchNorm, and proposes an alternative called RobustNorm, using min-max rescaling instead of normalization.

@highlight

This paper investigates the reason behind the vulnerability of BatchNorm and proposes Robust Normalization, a normalization method that achieves significantly better results under a variety of attack methods.
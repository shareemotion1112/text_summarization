By injecting adversarial examples into training data, adversarial training is promising for improving the robustness of deep learning models.

However, most existing adversarial training approaches are based on a specific type of adversarial attack.

It may not provide sufficiently representative samples from the adversarial domain, leading to a weak generalization ability on adversarial examples from other attacks.

Moreover, during the adversarial training, adversarial perturbations on inputs are usually crafted by fast single-step adversaries so as to scale to large datasets.

This work is mainly focused on the adversarial training yet efficient FGSM adversary.

In this scenario, it is difficult to train a model with great generalization due to the lack of representative adversarial samples, aka the samples are unable to accurately reflect the adversarial domain.

To alleviate this problem, we propose a novel Adversarial Training with Domain Adaptation (ATDA) method.

Our intuition is to regard the adversarial training on FGSM adversary as a domain adaption task with limited number of target domain samples.

The main idea is to learn a representation that is semantically meaningful and domain invariant on the clean domain as well as the adversarial domain.

Empirical evaluations on Fashion-MNIST, SVHN, CIFAR-10 and CIFAR-100 demonstrate that ATDA can greatly improve the generalization of adversarial training and the smoothness of the learned models, and outperforms state-of-the-art methods on standard benchmark datasets.

To show the transfer ability of our method, we also extend ATDA to the adversarial training on iterative attacks such as PGD-Adversial Training (PAT) and the defense performance is improved considerably.

Deep learning techniques have shown impressive performance on image classification and many other computer vision tasks.

However, recent works have revealed that deep learning models are often vulnerable to adversarial examples BID14 , which are maliciously designed to deceive the target model by generating carefully crafted adversarial perturbations on original clean inputs.

Moreover, adversarial examples can transfer across models to mislead other models with a high probability BID9 .

How to effectively defense against adversarial attacks is crucial for security-critical computer vision systems, such as autonomous driving.

As a promising approach, adversarial training defends from adversarial perturbations by training a target classifier with adversarial examples.

Researchers have found BID7 BID11 that adversarial training could increase the robustness of neural networks.

However, adversarial training often obtains adversarial examples by taking a specific attack technique (e.g., FGSM) into consideration, so the defense targeted such attack and the trained model exhibits weak generalization ability on adversarial examples from other adversaries BID7 .

BID22 showed that the robustness of adversarial training can be easily circumvented by the attack that combines with random perturbation from other models.

Accordingly, for most existing adversarial training methods, there is a risk of overfitting to adversarial examples crafted on the original model with the specific attack.

In this paper, we propose a novel adversarial training method that is able to improve the generalization of adversarial training.

From the perspective of domain adaptation (DA) BID20 , there is a big domain gap between the distribution of clean examples and the distribution of adversarial examples in the high-level representation space, even though adversarial perturbations are imperceptible to humans. showed that adversarial perturbations are progressively amplified along the layer hierarchy of neural networks, which maximizes the distance between the original and adversarial subspace representations.

In addition, adversarial training simply injects adversarial examples from a specific attack into the training set, but there is still a large sample space for adversarial examples.

Accordingly, training with the classification loss on such a training set will probably lead to overfitting on the adversarial examples from the specific attack.

Even though BID24 showed that adversarial training with iterative noisy attacks has stronger robustness than the adversarial training with single-step attacks, iterative attacks have a large computational cost and there is no theoretical analysis to justify that the adversarial examples sampled in such way could be sufficiently representative for the adversarial domain.

Our contributions are focused on how to improve the generalization of adversarial training on the simple yet scalable attacks, such as FGSM (Goodfellow et al.) .

The key idea of our approach is to formulate the learning procedure as a domain adaptation problem with limited number of target domain samples, where target domain denotes adversarial domain.

Specifically, we introduce unsupervised as well as supervised domain adaptation into adversarial training to minimize the gap and increase the similarity between the distributions of clean examples and adversarial examples.

In this way, the learned models generalize well on adversarial examples from different ∞ bounded attacks.

We evaluate our ATDA method on standard benchmark datasets.

Empirical results show that despite a small decay of accuracy on clean data, ATDA significantly improves the generalization ability of adversarial training and has the transfer ability to extend to adversarial training on PGD BID11 .

In this section, we introduce some notations and provides a brief overview of the current advanced attack methods, as well as the defense methods based on adversarial training.

Denote the clean data domain and the adversarial data domain by D and A respectively, we consider a classifier based on a neural network f (x) : DISPLAYFORM0 outputs the probability distribution for an input x ∈ [0, 1] d , and k denotes the number of classes in the classification task.

Let ϕ be the mapping at the logits layer (the last neural layer before the final softmax function), so that f (x) = sof tmax(ϕ(x)).

Let be the magnitude of the perturbation.

Let x adv be the adversarial image computed by perturbing the original image x. The cost function of image classification is denoted as J(x, y).

We define the logits as the logits layer representation, and define the logit space as the semantic space of the logits layer representation.

We divide attacks into two types: white-box attacks have the complete knowledge of the target model and can fully access the model; black-box attacks have limited knowledge of the target classifier (e.g.,its architecture) but can not access the model weights.

We consider four attack methods to generate adversarial examples.

For all attacks, the components of adversarial examples are clipped in [0, 1].Fast Gradient Sign Method (FGSM).

Goodfellow et al. introduced FGSM to generate adversarial examples by applying perturbations in the direction of the gradient.

DISPLAYFORM0 As compared with other attack methods, FGSM is a simple, yet fast and efficient adversary.

Accordingly, FGSM is particularly amenable to adversarial training.

Projected Gradient Descent (PGD).

The Projected Gradient Descent (PGD) adversary was introduced by BID11 without random start, which is a stronger iterative variant of FGSM.

This method applies FGSM iteratively for k times with a budget α instead of a single step.

DISPLAYFORM1 Here clip(·, a, b) function forces its input to reside in the range of [a, b] .

PGD usually yields a higher success rate than FGSM does in the white-box setting but shows weaker capability in the black-box setting.

.

BID22 proposed R+FGSM against adversarially trained models by applying a small random perturbation of step size α before applying FGSM.

DISPLAYFORM0 Momentum Iterative Method (MIM).

MIM ) is a modification of the iterative FGSM and it won the first place of NIPS 2017 Adversarial Attacks Competition.

Its basic idea is to utilize the gradients of the previous t steps with a decay factor µ to update the gradient at step t + 1 before applying FGSM with a budget α.

DISPLAYFORM1

An intuitive technique to defend a deep model against adversarial examples is adversarial training, which injects adversarial examples into the training data during the training process.

First, Goodfellow et al. proposed to increase the robustness by feeding the model with both original and adversarial examples generated by FGSM and by learning with the modified objective function.

BID7 scaled the adversarial training to ImageNet BID16 and showed better results by replacing half the clean example at each batch with the corresponding adversarial examples.

Meanwhile, BID7 discovered the label leaking effect and suggested not to use the FGSM defined with respect to the true label y true .

However, their approach has weak robustness to the RAND+FGSM adversary.

BID22 proposed an ensemble adversarial training to improve robustness on black-box attacks by injecting adversarial examples transferred from a number of fixed pre-trained models into the training data.

DISPLAYFORM0 For adversarial training, another approach is to train only with adversarial examples.

BID13 proposed a specialization of the method (Goodfellow et al.) that learned only with the objective function of adversarial examples.

BID11 demonstrated successful defenses based on adversarial training with the noisy PGD, which randomly initialize an adversarial example within the allowed norm ball before running iterative attack.

However, this technique is difficult to scale to large-scale neural networks BID6 as the iterative attack increases the training time by a factor that is roughly equal to the number of iterative steps.

BID24 developed a robust training method by linear programming that minimized the loss for the worst case within the perturbation ball around each clean data point.

However, their approach achieved high test error on clean data and it is still challenging to scale to deep or wide neural networks.

As described above, though adversarial training is promising, it is difficult to select a representative adversary to train on and most existing methods are weak in generalization for various adversaries, as the region of the adversarial examples for each clean data is large and contiguous BID21 BID19 .

Furthermore, generating a representative set of adversarial examples for large-scale datasets is computationally expensive.

In this work, instead of focusing on a better sampling strategy to obtain representative adversarial data from the adversarial domain, we are especially concerned with the problem of how to train with clean data and adversarial examples from the efficient FGSM, so that the adversarially trained model is strong in generalization for different adversaries and has a low computational cost during the training.

We propose an Adversarial Training with Domain Adaptation (ATDA) method to defense adversarial attacks and expect the learned models generalize well for various adversarial examples.

Our motivation is to treat the adversarial training on FGSM as a domain adaptation task with limited number of target domain samples, where the target domain denotes adversarial domain.

We combine standard adversarial training with the domain adaptor, which minimizes the domain gap between clean examples and adversarial examples.

In this way, our adversarially trained model is effective on adversarial examples crafted by FGSM but also shows great generalization on other adversaries.

It's known that there is a huge shift in the distributions of clean data and adversarial data in the high-level representation space.

Assume that in the logit space, data from either the clean domain or the adversarial domain follow a multivariate normal distribution, i.e., DISPLAYFORM0 Our goal is to learn the logits representation that minimizes the shift by aligning the covariance matrices and the mean vectors of the clean distribution and the adversarial distribution.

To implement the CORrelation ALignment (CORAL), we define a covariance distance between the clean data and the adversarial data as follows.

DISPLAYFORM1 where C ϕ(D) and C ϕ(A) are the covariance matrices of the clean data and the adversarial data in the logit space respectively, and · 1 denotes the L 1 norm of a matrix.

Note that L CORAL (D, A) is slightly different from the CORAL loss proposed by BID17 .Similarly, we use the standard distribution distance metric, Maximum Mean Discrepancy (MMD) BID1 , to minimize the distance of the mean vectors of the clean data and the adversarial data.

DISPLAYFORM2 The loss function for Unsupervised Domain Adaptation (UDA) can be calculated as follows.

DISPLAYFORM3

Even though the unsupervised domain adaptation achieves perfect confusion alignment, there is no guarantee that samples of the same label from clean domain and adversarial domain would map nearby in the logit space.

To effectively utilize the labeled data in the adversarial domain, we introduce a supervised domain adaptation (SDA) by proposing a new loss function, denoted as margin loss, to minimize the intra-class variations and maximize the inter-class variations on samples of different domains.

The SDA loss is shown in Eq. FORMULA10 .

DISPLAYFORM0 Here sof tplus denotes a function ln(1 + exp(·)); c ytrue ∈ R k denotes the center of y true class in the logit space; C = { c j | j = 1, 2, ..., k} is a set consisting of the logits center for each class, which will be updated as the logits changed.

Similar to the center loss BID23 , we update center c j for each class j: DISPLAYFORM1 where 1 condition = 1 if the condition is true, otherwise 1 condition = 0; α denotes the learning rate of the centers.

During the training process, the logits center for each class can integrate the logits representation from both the clean domain and the adversarial domain.

For adversarial training, iterative attacks are fairly expensive to compute and single-step attacks are fast to compute.

Accordingly, we use a variant of FGSM attack BID7 ) that avoids the label leaking effect to generate a new adversarial example x adv i for each clean example x i .

DISPLAYFORM0 where y target denotes the predicted class arg max{ϕ(x i )} of the model.

However, in this case, the sampled adversarial examples are aggressive but not sufficiently representative due to the fact that the sampled adversarial examples always lie at the boundary of the ∞ ball of radius (see FIG1 ) and the adversarial examples within the boundary are ignored.

For adversarial training, if we train a deep neural network only on the clean data and the adversarial data from the FGSM attack, the adversarially trained model will overfit on these two kinds of data and exhibits weak generalization ability on the adversarial examples sampled from other attacks.

From a different perspective, such problem can be viewed as a domain adaptation problem with limited number of labeled target domain samples, as only some special data point can be sampled in the adversarial domain by FGSM adversary.

Consequently, it is natural to combine the adversarial training with domain adaptation to improve the generalization ability on adversarial data.

We generate new adversarial examples by the variant of FGSM attack shown in Eq. FORMULA1 , then we use the following loss function to meet the criteria of domain adaptation while training a strong classifier.

Compute the loss by Eq. (12) and update parameters of network f by back propagation; 10: until the training converges.

DISPLAYFORM1

In this section, we evaluate our ATDA method on various benchmark datasets to demonstrate the robustness and contrast its performance against other competing methods under different white-box and black-box attacks with bounded ∞ norm.

Code for these experiments is available at https: //github.com/JHL-HUST/ATDA.

Datasets.

We consider four popular datasets, namely Fashion-MNIST BID26 , SVHN BID12 , CIFAR-10 and CIFAR-100 BID5 Baselines.

To evaluate the generalization power on adversarial examples in both the white-box and black-box settings, we report the clean test accuracy, the defense accuracy on FGSM, PGD, R+FGSM and MIM in the non-targeted way.

The common settings for these attacks are shown in Table 5 of the Appendix.

We compare our ATDA method with normal training as well as several state-of-the-art adversarial training methods:• Normal Training (NT).

Training with cross-entropy loss on the clean training data.• Standard Adversarial Training (SAT) (Goodfellow et al.) .

Training with the cross-entropy on the clean training data and the adversarial examples from the FGSM variant with perturbation to avoid label leaking.• Ensemble Adversarial Training (EAT) BID22 .

Training with cross-entropy on the clean training data and the adversarial examples crafted from the currently trained model and the static pre-trained models by the FGSM variant with the perturbation to avoid label leaking.• Provably Robust Training (PRT) BID24 .

Training with cross-entropy loss on the worst case in the ∞ ball of radius around each clean training data point.

It could be seen as training with a complicated method of sampling in the ∞ ball of radius .

Evaluation Setup.

For each benchmark dataset, we train a normal model and various adversarial models with perturbation on a main model with ConvNet architecture, and evaluate them on various attacks bounded by .

Moreover, for Ensemble Adversarial Training (EAT), we use two different models as the static pre-trained models.

For black-box attacks, we test trained models on the adversarial examples transferred from a model held out during the training.

All experiments are implemented on a single Titan X GPU.

For all experiments, we set the hyper-parameter λ in Eq. (12) to 1/3 and the hyper-parameter α in Eq. (10) to 0.1.

For more details about neural network architectures and training hyper-parameters, see Appendix A. We tune the networks to make sure they work, not to post concentrates on optimizing these settings.

We evaluate the defense performance of our ATDA method from the perspective of classification accuracy on various datasets, and compare with the baselines.

Evaluation on Fashion-MNIST.

The accuracy results on Fashion-MNIST are reported in TAB1 .

NT yields the best performance on the clean data, but generalizes poorly on adversarial examples.

SAT and EAT overfit on the clean data and the adversarial data from FGSM.

PRT achieves lower error against various adversaries, but higher error on the clean data.

ATDA achieves stronger robustness against different ∞ bounded adversaries as compared to SAT (adversarial training on FGSM).Evaluation on SVHN.

The classification accuracy on SVHN are summarized in TAB1 .

PRT seems to degrade the performance on the clean testing data and exhibits weak robustness on various attacks.

As compared to SAT, ATDA achieves stronger generalization ability on adversarial examples from various attacks and higher accuracy on the white-box adversaries, at the same time it only loses a negligible performance on clean data.

Evaluation on CIFAR-10.

Compared with Fashion-MNIST and SVHN, CIFAR-10 is a more difficult dataset for classification.

As PRT is challenging and expensive to scale to large neural networks due to its complexity, the results of PRT are not reported.

The accuracy results on CIFAR-10 are summarized in TAB1 .

ATDA outperforms all the competing methods on most adversaries, despite a slightly lower performance on clean data.

Evaluation on CIFAR-100.

The CIFAR-100 dataset contains 100 image classes, with 600 images per class.

Our goal here is not to achieve state-of-the-art performance on CIFAR-100, but to compare the generalization ability of different training methods on a comparatively large dataset.

The results on CIFAR-100 are summarized in TAB1 .

Compared to SAT, ATDA achieves better generalization on various adversarial examples and it does not degrade the performance on clean data.

In conclusion, the accuracy results provide empirical evidence that ATDA has great generalization ability on different adversaries as compared to SAT and outperforms other competing methods.

To further investigate the defence performance of the proposed method, we compute two other metrics: the local loss sensitivity to perturbations and the shift of adversarial data distribution with respect to the clean data distribution.

Local Loss Sensitivity.

One method to quantify smoothness and generalization to perturbations for models is the local loss sensitivity BID0 .

It is calculated in the clean testing data as follows.

The lower the value is, the smoother the loss function is.

DISPLAYFORM0 The results of the local loss sensitivity for the aforementioned learned models are summarized in TAB2 .

The results suggest that adversarial training methods do increase the smoothness of the model as compared with the normal training and ATDA performs the best.

Distribution Discrepancy.

To quantify the dissimilarity of the distributions between the clean data and the adversarial data, we compare our learned logits embeddings with the logits embeddings of the competing methods on Fashion-MNIST.

We use t-SNE BID10 for the comparison on the training data, testing data and adversarial testing data from the white-box FGSM or PGD.

The comparisons are illustrated in FIG3 and we report the detailed MMD distances across domains in TAB3 .

Compared with NT, SAT and EAT actually increase the MMD distance across domains of the clean data and the adversarial data.

In contrast, PRT and ATDA can learn domain invariance between the clean domain and the adversarial domain.

Furthermore, our learned logits representation achieves the best performance on domain invariance.

FIG4 .

For each model, we report the average accuracy rates over all white-box attacks and all black-box attacks, respectively.

The results illustrate that, by aligning the covariance matrix and mean vector of the clean and adversarial examples, UDA plays a key role in improving the generalization of SAT on various attacks.

In general, the aware of margin loss on SDA can also improve the defense quality on standard adversarial training, but the effectiveness is not very stable over all datasets.

By combining UDA and SDA together with SAT, our final algorithm ATDA can exhibits stable improvements on the standard adversarial training.

In general, the performance of ATDA is slightly better than SAT+UDA.

We report the average accuracy rates over all white-box attacks and all black-box attacks, respectively.

ATDA can simply be extended to adversarial training on other adversaries.

We now consider to extend the ATDA method to PGD-Adversarial Training (PAT) BID11 : adversarial training on the noisy PGD with perturbation .

By combining adversarial training on the noisy PGD with domain adaptation, we implement an extension of ATDA for PAT, called PATDA.

For the noisy PGD, we set the iterated step k as 10 and the budget α as /4 according to BID11 .As shown in TAB4 , we evaluate the defense performance of PAT and PATDA on various datasets.

On Fashion-MNIST, we observe that PATDA fails to increase robustness to most adversaries as compared to PAT.

On SVHN, PAT and PATDA fail to converge properly.

The results are not surprising, as training with the hard and sufficient adversarial examples (from the noisy PGD) requires the neural networks with more parameters.

On CIFAR-10 and CIFAR-100, PATDA achieves stronger robustness to various attacks than PAT.

In general, PATDA exhibits stronger robustness to various adversaries as compared to PAT.

The results indicate that domain adaptation can be applied flexibly to adversarial training on other adversaries to improve the defense performance.

In this study, we regard the adversarial training as a domain adaptation task with limited number of target labeled data.

By combining adversarial training on FGSM adversary with unsupervised and supervised domain adaptation, the generalization ability on adversarial examples from various attacks and the smoothness on the learned models can be highly improved for robust defense.

In addition, ATDA can easily be extended to adversarial training on iterative attacks (e.g., PGD) to improve the defense performance.

The experimental results on several benchmark datasets suggest that the proposed ATDA and its extension PATDA achieve significantly better generalization results as compared with current competing adversarial training methods.

This work is supported by National Natural Science Foundation (61772219).

In the appendix, we show all details of the common settings, neural network architectures and training hyper-parameters for the experiments.

A.1 HYPER-PARAMETERS FOR ADVERSARIES.For each dataset, the details about the hyper-parameters of various adversaries are shown in Table 5 , where denotes the magnitude of adversarial perturbations.

Fashion-MNIST.

In the training phase, we use Adam optimizer with a learning rate of 0.001 and set the batch size to 64.

For Fashion-MNIST, the neural network architectures for the main model, the static pre-trained models and the model held out during training are depicted in TAB6 .

For all adversarial training methods, the magnitude of perturbations is 0.1 in ∞ norm.

In the training phase, we use Adam optimizer with a learning rate of 0.001 and set the batch size to 32 and use the same architectures as in Fashion-MNIST.

For all adversarial training methods, the magnitude of perturbations is 0.02 in ∞ norm.

FORMULA1 FC FORMULA1 FC FORMULA1 CIFAR-10.

In the training phase, we use the same training settings as in SVHN.

we use Adam optimizer with a learning rate of 0.001 and set the batch size to 32.

In order to enhance the expressive power of deep neural networks, we use Exponential Linear Unit (ELU) BID2 as the activation function and introduce Group Normalization BID25 into the architectures.

The neural network architectures for CIFAR-10 are shown in Table 7 .

For all adversarial training methods, the magnitude of perturbations is 4/255 in ∞ norm.

CIFAR-100.

We use the same training settings as in CIFAR-10.

For CIFAR-100, the neural network architectures for the main model, the static pre-trained models and the model held out during training are shown in Table 8 .

For all adversarial training methods, the magnitude of perturbations is 4/255 in ∞ norm.

@highlight

We propose a novel adversarial training with domain adaptation method that significantly improves the generalization ability on adversarial examples from different attacks.
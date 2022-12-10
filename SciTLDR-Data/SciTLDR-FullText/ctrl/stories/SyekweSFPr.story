Previous work on adversarially robust neural networks requires large training sets and computationally expensive training procedures.

On the other hand, few-shot learning methods are highly vulnerable to adversarial examples.

The goal of our work is to produce networks which both perform well at few-shot tasks and are simultaneously robust to adversarial examples.

We adapt adversarial training for meta-learning, we adapt robust architectural features to small networks for meta-learning, we test pre-processing defenses as an alternative to adversarial training for meta-learning, and we investigate the advantages of robust meta-learning over robust transfer-learning for few-shot tasks.

This work provides a thorough analysis of adversarially robust methods in the context of meta-learning, and we lay the foundation for future work on defenses for few-shot tasks.

For safety-critical applications like facial recognition, traffic sign detection, and copyright control, adversarial attacks pose an actionable threat (Zhao et al., 2018; Eykholt et al., 2017; .

Conventional adversarial training and pre-processing defenses aim to produce networks that resist attack (Madry et al., 2017; Zhang et al., 2019; Samangouei et al., 2018) , but such defenses rely heavily on the availability of large training data sets.

In applications that require few-shot learning, such as face recognition from few images, recognition of a video source from a single clip, or recognition of a new object from few example photos, the conventional robust training pipeline breaks down.

When data is scarce or new classes arise frequently, neural networks must adapt quickly (Duan et al., 2017; Kaiser et al., 2017; Pfister et al., 2014; Vartak et al., 2017) .

In these situations, metalearning methods conduct few-shot learning by creating networks that learn quickly from little data and with computationally cheap fine-tuning.

While state-of-the-art meta-learning methods perform well on benchmark few-shot classification tasks, these naturally trained neural networks are highly vulnerable to adversarial examples.

In fact, even adversarially trained feature extractors fail to resist attacks in the few-shot setting (see Section 4.1).

We propose a new approach, called adversarial querying, in which the network is exposed to adversarial attacks during the query step of meta-learning.

This algorithm-agnostic method produces a feature extractor that is robust, even without adversarial training during fine-tuning.

In the few-shot setting, we show that adversarial querying outperforms other robustness techniques by a wide margin in terms of both clean accuracy and adversarial robustness (see Table 1 ).

We solve the following minimax problem:

where S and (x, y) are data sampled from the training distribution, A is a fine-tuning algorithm for the model parameters, θ, and is a p-norm bound for the attacker.

In Section 4, we further motivate adversarial querying and exhibit a wide range of experiments.

To motivate the necessity for adversarial querying, we test methods, such as adversarial fine-tuning and pre-processing defenses, which if successful, would eliminate the need for expensive adversarial training routines.

We find that these methods are far less effective than adversarial querying.

A nat A adv AT transfer learning (R2-D2 backbone) 39.13% 25.33% Naturally Trained R2-D2 73.01% 0.00% ADML 47.75% 18.49% AQ R2-D2 (ours) 57.87% 31.52% Table 1 : Adversarially trained transfer learning, R2-D2 (Bertinetto et al., 2018) , ADML (Yin et al., 2018) , and our adversarially queried (AQ) R2-D2 model on 5-shot Mini-ImageNet.

Natural accuracy is denoted A nat , and robust accuracy, A adv , is computed with a 20-step PGD attack as in Madry et al. (2017) with = 8/255.

A description of our training regime can be found in Appendix A.5.

Before the emergence of meta-learning, a number of approaches existed to cope with few-shot problems.

One simple approach is transfer learning, in which pre-trained feature extractors are trained on large data sets and fine-tuned on new tasks (Bengio, 2012) .

Metric learning methods avoid overfitting to the small number of training examples in new classes by instead performing classification using nearest-neighbors in feature space with a feature extractor that is trained on a large corpus of data and not re-trained when classes are added (Snell et al., 2017; Gidaris & Komodakis, 2018; Mensink et al., 2012) .

Metric learning methods are computationally efficient when adding new classes at inference, since the feature extractor is not re-trained.

Meta-learning algorithms create a "base" model that quickly adapts to new tasks by fine-tuning.

This model is created using a set of training tasks {T i } that can be sampled from a task distribution.

Each task comes with support data, T s i , and query data, T q i .

Support data is used for fine-tuning, and query data is used to measure the performance of the resulting network.

In practice, each task is taken to be a classification problem involving only a small subset of classes in a large many-class data set.

The number of examples per class in the support set is called the shot, so that fine-tuning on five support examples per class is 5-shot learning.

An iteration of training begins by sampling tasks {T i } from the task distribution.

In the "inner loop" of training, the base model is fine-tuned on the support data from the sampled tasks.

In the "outer loop" of training, the fine-tuned network is used to make predictions on the query data, and the base model parameters are updated to improve the accuracy of the resulting fine-tuned model.

The outer loop requires backpropagation through the fine-tuning steps.

A formal treatment of the prototypical meta-learning routine can be found in Algorithm 1.

Algorithm 1: The meta-learning framework Require: Base model, F θ , fine-tuning algorithm, A, learning rate, γ, and distribution over tasks, p(T ).

Initialize θ, the weights of F ; while not done do Sample batch of tasks,

, where

Fine-tune model on T i (inner loop).

New network parameters are written

Note that the fine-tuned parameters, θ i = A(θ, T s i ), in Algorithm 1, are a function of the base model's parameters so that the gradient computation in the outer loop may backpropagate through A. For validation after training, the base model is fine-tuned on the support set of hold-out tasks, and accuracy on the query set is reported.

In this work, we report performance on Omniglot, MiniImageNet, and CIFAR-FS (Lake et al., 2015; Vinyals et al., 2016; Bertinetto et al., 2018) .

We focus on four meta-learning algorithms: MAML, R2-D2, MetaOptNet, and ProtoNet (Finn et al., 2017; Bertinetto et al., 2018; Lee et al., 2019; Snell et al., 2017) .

During fine-tuning, MAML uses SGD to update all parameters, minimizing cross-entropy loss.

Since unrolling SGD steps into a deep computation graph is expensive, first-order variants have been developed to avoid computing second-order derivatives.

We use the original MAML formulation.

R2-D2 and MetaOptNet, on the other hand, only update the final linear layer during fine-tuning, leaving the "backbone network" that extracts these features frozen at test time.

R2-D2 replaces SGD with a closed-form differentiable solver for regularized ridge regression, while MetaOptNet achieves its best performance when replacing SGD with a solver for SVM.

Because the objective of these linear problems is convex, differentiable convex optimizers can be efficiently deployed to find optima, and differentiate these optima with respect to the backbone parameters at train time.

ProtoNet takes an approach inspired by metric learning.

It constructs class prototypes as centroids in feature space for each task.

These centroids are then used to classify the query set in the outer loop of training.

Because each class prototype is a simple geometric average of feature representations, it is easy to differentiate through the fine-tuning step.

Following standard practices, we assess the robustness of models by attacking them with ∞ -bounded perturbations.

We craft adversarial perturbations using the projected gradient descent attack (PGD) since it has proven to be one of the most effective algorithms both for attacking as well as for adversarial training (Madry et al., 2017) .

A detailed description of the PGD attack can be found in Algorithm 2.

We consider perturbations with ∞ bound 8/255 and a step size of 2/255 as described by Madry et al. (2017) .

Adversarial training is the industry standard for creating robust models that maintain good cleanlabel performance (Madry et al., 2017) .

This method involves replacing clean examples with adversarial examples during the training routine.

A simple way to harden models to attack is adversarial training, which solves the minimax problem

where L θ (x + δ, y) is the loss function of a network with parameters θ, x is an input image with label y, and δ is an adversarial perturbation.

Adversarial training finds network parameters which maintain low loss (and correct class labels) even when adversarial perturbations are added.

Algorithm 2: PGD Attack Require: network, F θ , input data, (x, y), number of steps, n, step size, γ, and attack bound, .

Initialize δ ∈ B (x) randomly;

Several authors have tried to learn robust models in the data scarce regime.

The authors of study robustness properties of transfer learning.

They find that retraining earlier layers of the network during fine-tuning impairs the robustness of the network, while only retraining later layers can largely preserve robustness.

ADML is the first attempt at achieving robustness through meta-learning.

ADML is a MAML variant, specifically designed for robustness, which employs adversarial training (Yin et al., 2018) .

However, this method for robustness is only compatible with MAML, an outdated meta-learning algorithm.

Moreover, ADML is computationally expensive, and the authors only test their method against a weak attacker.

We implement ADML and test it against a strong attacker.

We show that our method simultaneously achieves higher robustness and higher natural accuracy.

In this section, we benchmark the robustness of existing meta-learning methods.

Similarly to classically trained classifiers, we expect that few-shot learners are highly vulnerable to attack when adversarial defenses are not employed.

We test prominent meta-learning algorithms against a 20-step PGD attack as in Madry et al. (2017) .

Table 2 contains natural and robust accuracy on the MiniImageNet and CIFAR-FS 5-shot tasks (Vinyals et al., 2016; Bertinetto et al., 2018) .

Experiments in the 1-shot setting can be found in Appendix A.1.

A We find that these algorithms are completely unable to resist the attack.

Interestingly, MetaOptNet uses SVM for fine-tuning, which is endowed with a wide margins property.

The failure of even SVM to express robustness during testing suggests that using robust fine-tuning methods (at test time) on naturally trained meta-learners is insufficient for robust performance.

To further examine this, we consider MAML, which updates the entire network during fine-tuning.

We use a naturally trained MAML model and perform adversarial training during fine-tuning (see Table 3 ).

Adversarial training is performed with 7-PGD as in (Madry et al., 2017) .

If adversarial fine-tuning yielded robust classification, then we could avoid expensive adversarial training variants during meta-learning.

Table 3 : MAML models on Mini-ImageNet and Omniglot.

A nat and A adv are natural and robust test accuracy, respectively, where robust accuracy is computed with respect to a 20-step PGD attack.

A nat(AT ) and A adv(AT ) are natural and robust test accuracy with 7-PGD fine-tuning.

While clean trained MAML models with adversarial fine-tuning are slightly more robust than their naturally fine-tuned counterparts, they achieve almost no robustness on Mini-ImageNet.

Omniglot is an easier data set, and the performance of adversarially fine-tuned MAML on the 5-shot version is below a reasonable tolerance for robustness.

We conclude from these experiments that naturally trained meta-learners are vulnerable to adversarial examples, and robustness techniques specifically for few-shot learning are required.

We now introduce adversarial querying (AQ), an adversarial training algorithm for meta-learning.

Let A(θ, S) denote a fine-tuning algorithm.

Then, A is a map from support data set, S, and network parameters, θ, to parameters for the fine-tuned network.

Then, we seek to solve the following minimax problem (Equation 1 revisited):

where S and (x, y) are support and query data, respectively, sampled from the training distribution, and is a p-norm bound for the attacker.

Thus, the objective is to find a central parameter vector which, when fine-tuned on support data, minimizes the expected query loss against an attacker.

We approach this minimax problem with an alternating algorithm consisting of the following steps:

1.

Sample support and query data 2.

Fine-tune on the support data (inner loop)

3.

Perturb query data to maximize loss 4.

Minimize query loss, backpropagating through the fine-tuning algorithm (outer loop)

A formal treatment of this method is presented in Algorithm 3.

We test adversarial querying across multiple data sets and meta-learning protocols.

It is important to note that adversarial querying is algorithm-agnostic.

We test this method on the ProtoNet, R2-D2, and MetaOptNet algorithms on CIFAR-FS and Mini-ImageNet (see Table 4 and Table 5 ).

Algorithm 3: Adversarial Querying Require: Base model, F θ , fine-tuning algorithm, A, learning rate, γ, and distribution over tasks, p(T ).

Initialize θ, the weights of F ; while not done do Sample batch of tasks,

, where In our tests, R2-D2 outperforms MetaOptNet in robust accuracy despite having a less powerful backbone architecture.

In Section 4.5, we dissect the effects of backbone architecture and classification head on robustness of meta-learned models.

In Section 4.7, we verify that adversarial querying generates networks robust to a wide array of strong attacks.

We observe above that few-shot learning methods with a non-robust feature extractor break under attack.

But what if we use a robust feature extractor?

In the following section, we consider both transfer learning and meta-learning with a robust feature extractor.

In order to compare robust transfer learning and meta-learning, we train the backbone networks from meta-learning algorithms on all training data simultaneously in the fashion of standard adversarial training using 7-PGD (not meta-learning).

We then fine-tune using the head from a meta-learning algorithm on top of the transferred feature extractor.

We compare the performance of these feature extractors to that of those trained using adversarially queried meta-learning algorithms with the same backbones and heads.

This experiment provides a direct comparison of feature extractors produced by robust transfer learning and robust meta-learning (see Table 6 ).

In the adversarial querying procedure detailed in Algorithm 3, we only attack query data.

Consider that the loss value on query data represents performance on testing data after fine-tuning on the support data.

Thus, low loss on perturbed query data represents robust accuracy on testing data after fine-tuning.

Simply put, minimizing loss on adversarial query data moves the parameter vector towards a network with high robust test accuracy.

It follows that attacking only support data is not an option for achieving robust meta-learners.

Attacking support data but not query data can be seen as maximizing clean test accuracy when fine-tuned in a robust manner, but since we want to maximize robust test accuracy, this would be inappropriate.

One question remains: should we attack both query and support data?

One reason to perturb only query data is computational efficiency.

The bulk of computation in adversarial training is spent computing adversarial attacks since perturbing each batch requires an iterative algorithm.

Perturbing support data doubles the number of adversarial attacks computed during training.

Thus, only attacking query data significantly accelerates training.

Another reason to avoid attacking support data is that some meta-learning fine-tuning algorithms based on metric learning, such as ProtoNet, do not involve loss minimization during fine-tuning, so it is not clear what loss the attacker would maximize during the inner loop.

But if attacking support data during the inner loop of training were to significantly improve robust performance, we would like to know.

We now compare adversarial querying to a variant in which support data is also perturbed during training.

We use MAML to conduct this comparison on the Omniglot and Mini-ImageNet data sets.

Additional experiments on 1-shot tasks can be found in Appendix A.3.

In these experiments, we find that adversarially attacking the support data during the inner loop of meta-learning does not improve performance over adversarial querying.

Furthermore, networks trained in this fashion require adversarial fine-tuning during test time or else they suffer a massive loss in robust test accuracy.

Following these results and the significant reasons to avoid attacking support data, we subsequently only attack query data.

Table 8 : Performance on 5-shot Mini-ImageNet.

Robust accuracy, A adv , is computed with respect to a 20-step PGD attack.

A nat(AT ) and A adv(AT ) are natural and robust test accuracy with 7-PGD training during fine-tuning.

Top robust accuracy with and without adversarial fine-tuning is bolded.

Adversarial querying can also be used to construct meta-learning analogues for other variants of adversarial training.

We explore this by using the TRADES loss function in the querying step of AQ (Zhang et al., 2019) .

We refer to this method as meta-TRADES.

While meta-TRADES can marginally outperform our initial adversarial querying method in robust accuracy with a careful hyperparameter choice, λ, we find that networks trained with meta-TRADES severely sacrifice natural accuracy (see Table 9 ).

Additional experiments on 1-shot tasks can be found in Appendix A.4.

Table 9 : 5-shot Mini-ImagNet (MI) and CIFAR-FS (FS) results comparing meta-TRADES to adversarial querying.

A nat and A adv are natural and robust test accuracy, respectively, where robust accuracy is computed with respect to a 20-step PGD attack.

High performing meta-learning models, like MetaOptNet and R2-D2, fix their feature extractor and only update their last linear layer during fine-tuning.

In the setting of transfer learning, robustness is a feature of early convolutional layers, and re-training these early layers leads to a significant drop in robust test accuracy .

We verify that re-training only the last layer leads to improved natural and robust accuracy in adversarially queried meta-learners by training a MAML model but only updating the final fully-connected layer during fine-tuning including during the inner loop of meta-learning.

We find that the AQ model trained by fine-tuning only the last layer during the inner loop decisively outperforms the standard AQ MAML algorithm in both natural and robust accuracy (see Table 10 ).

Table 10 : Adversarially queried MAML compared with a MAML variant where only the last layer is re-trained during fine-tuning on 5-shot Mini-ImageNet.

A nat and A adv are natural and robust test accuracy, respectively, where robust accuracy is computed with respect to a 20-step PGD attack.

A nat(AT ) and A adv(AT ) are natural and robust test accuracy, respectively with 7-PGD training during fine-tuning.

Layers are fine-tuned for 10 steps with a learning rate of 0.01.

PERFORMANCE.

The naturally trained R2-D2 algorithm performs worse than MetaOptNet in natural accuracy, but previous research has found that performance discrepancies between meta-learning algorithms might be an artifact of different backbone networks .

We confirm that MetaOptNet with the R2-D2 backbone performs similarly to R2-D2 in the natural meta-learning setting (see Table 11 ).

However, we find that the performance discrepancy in the adversarial setting is not explained by differences in backbone architecture.

In our adversarial querying experiments, we see that MetaOptNet is less robust than R2-D2.

This discrepancy remains when we train MetaOptNet with the R2-D2 backbone (see Table 12 ).

We conclude that MetaOptNet's backbone is not responsible for its inferior robustness.

These experiments suggest that ridge regression may be a more effective fine-tuning technique than SVM for robust performance.

ProtoNet with R2-D2 backbone also performs worse than the other two adversarially queried models with the same backbone architecture.

Table 12 : Robust test accuracy of adversarially queried R2-D2, MetaOptNet, and the MetaOptNet and heads with R2-D2 backbone on Mini-ImageNet (MI) and CIFAR-FS (FS) data sets.

Robust accuracy is computed with respect to a 20-step PGD attack.

In addition to adversarial training, architectural features have been used to enhance robustness (Xie et al., 2019) .

Feature denoising blocks pair classical denoising operations with learned 1 × 1 convolutions to reduce the feature noise in feature maps at various stages of a network, and thus reduce the success of adversarial attacks.

Massive architectures with these blocks have achieved state-of-theart robustness against targeted adversarial attacks on ImageNet.

However, when deployed on small networks for meta-learning, we find that denoising blocks do not improve robustness.

We deploy denoising blocks identical to those in Xie et al. (2019) after various layers of the R2-D2 network.

The best results for the denoising experiments are achieved by adding a denoising block after the fourth layer in the R2-D2 embedding network (see Table 13 ).

73.02% 0.00% R2-D2 AQ 57.87% 31.52% R2-D2 AQ Denoising 57.68% 31.14% Table 13 : 5-shot MiniImageNet results for our highest performing R2-D2 with feature denoising blocks.

A nat and A adv are natural and robust test accuracy, respectively, where robust accuracy is computed with respect to a 20-step PGD attack.

Top robust accuracy is bolded.

We test our method by exposing our adversarially queried R2-D2 model to a variety of powerful adversarial attacks.

We implement the momentum iterated fast gradient sign method (MI-FGSM), DeepFool, and 20-step PGD with 20 random restarts (Dong et al., 2018; Moosavi-Dezfooli et al., 2016; Madry et al., 2017) .

Our adversarially queried model is indeed nearly as robust against the strongest ∞ bounded attacker as it is against the 20-step PGD attack with a single random start we tested against previously.

Note that DeepFool is not ∞ bounded and thus the perturbed images are outside of the robustness radius enforced during adversarial querying.

Additional experiments on CIFAR-FS can be found in Appendix A.6.

A DF A M I A 20−P GD R2-D2 7.91% 0.01% 0.0% R2-D2 AQ (ours) 14.45% 31.87% 30.31% R2-D2 Transfer 0.42% 24.01% 19.75% Table 14 : 5-shot MiniImageNet results against DeepFool (DF) (2 iteration) ∞ attack, MI-FGSM (MI) ( = 8/255) attack, and PGD attack with 20 random restarts (20-PGD).

We compare R2-D2 trained with adversarial-querying (AQ) to the adversarially trained transfer learning R2-D2 as in section 4.1.

Recent works have proposed pre-proccessing defenses for sanitizing adversarial examples before feeding them into a naturally trained classifier.

If successful, these methods would avoid the expensive adversarial querying procedure during training.

While this approach has found success in the mainstream literature, we find that it is ineffective in the few-shot regime.

In DefenseGAN, a GAN trained on natural images is used to sanitize an adversarial example by replacing (possible corrupted) test images with the nearest image in the output range of the GAN (Samangouei et al., 2018) .

Unfortunately, GANs are not expressive enough to preserve the integrity of testing images on complex data sets involving high-resolution natural images, and recent attacks have critically compromised the performance of this defense (Ilyas et al., 2017; Athalye et al., 2018) .

We found the expressiveness of the generator architecture used in the original DefenseGAN setup to be insufficient for even CIFAR-FS, so we substitute a stronger ProGAN generator to model the CIFAR-100 classes (Karras et al., 2017) .

Another pre-processing method, the superresolution defense, first denoises data with sparse wavelet filters and then performs superresolution (Mustafa et al., 2019) .

This defense is also motivated by the principle of projecting adversarial examples onto the natural image manifold.

We test the superresolution defense using the same wavelet filtering and superresolution network (SRResNet) used by Mustafa et al. (2019) and first introduced by Ledig et al. (2017) .

We train the SRResNet, in a similar fashion to the generator for DefenseGAN, on the entire CIFAR-100 data set before applying the superresolution defense.

We find that these methods are not well suited to the few-shot domain, in which the generative model or superresolution network may not be able to train on the little data available.

Morever, even after training the generator on all CIFAR-100 classes, we find that DefenseGAN with a naturally trained R2-D2 meta-learner performs significantly worse in both natural and robust accuracy than an adversarially queried meta-learner of the same architecture.

Similarly, the superresolution defense achieves little robustness.

The results of these experiments can be found in

Naturally trained networks for few-shot image classification are vulnerable to adversarial attacks, and existing robust transfer learning methods do not perform well on few-shot tasks.

Even, when adversarially fine-tuned, naturally trained networks suffer from adversarial vulnerability.

We thus identify the need for few-shot methods for adversarial robustness.

In particular, we study robustness in the context of meta-learning.

We develop an algorithm-agnostic method, called adversarial querying, for hardening meta-learning models.

We find that meta-learning models are most robust when the feature extractor is fixed and only the last layer is retrained during fine-tuning.

We further identify that choice of classification head significantly impacts robustness.

We believe that this paper is a starting point for developing adversarially robust methods for few-shot applications.

We train ProtoNet, R2-D2, and MetaOptNet models for 60 epochs with SGD.

We use a learning rate of 0.1, momentum (Nesterov) of 0.9, and a weight decay term of 5(10 −4 ) for the parameters of both the head and the embedding.

We decrease the learning rate to 0.06 after epoch 20, 0.012 after epoch 40, and 0.0024 after epoch 50.

MAML is trained for 60000 epochs with meta learning rate of 0.001 and fine-tuning learning rate of 0.01.

Fine-tuning is performed for 10 steps per task.

A DF A M I A 20−P GD R2-D2 0.00% 0.39% 0.01% R2-D2 AQ (ours)

14.45% 53.46% 46.57% R2-D2 AT (Transfer Learning) 1.41% 38.28% 33.17% Table 23 : 5-shot CIFAR-FS results against DeepFool (DF) (2 iteration) ∞ attack, MI-FGSM (MI) ( = 8/255) attack, and PGD attack with 20 random restarts (20-PGD).

We compare R2-D2 trained with adversarial-querying (AQ) to the transfer learning R2-D2 as in section 4.1.

A ResN et R2-D2 0.00% R2-D2 AQ (ours) 59.68% R2-D2 AT (Transfer Learning) 42.02% Table 24 : 5-shot CIFAR-FS results against black-box transfer attacks crafted on an adversarially trained (transfer learning) ResNet-12 model using 7-PGD.

We then test R2-D2 trained with adversarial-querying (AQ) and the transfer learning R2-D2 model on these crafted perturbations.

<|TLDR|>

@highlight

We develop meta-learning methods for adversarially robust few-shot learning.

@highlight

This paper presents a method that enhances the robustness of few-shot learning by introducing adversarial query data attack in the inner-task fine-tuning phase of a meta-learning algorithm.

@highlight

The authors of this paper propose a novel approach for training a robust few-shot model. 
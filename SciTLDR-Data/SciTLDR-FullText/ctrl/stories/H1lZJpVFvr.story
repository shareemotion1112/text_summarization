Adversarial training has been demonstrated as one of the most effective methods for training robust models to defend against adversarial examples.

However, adversarially trained models often lack adversarially robust generalization on unseen testing data.

Recent works show that adversarially trained models are more biased towards global structure features.

Instead, in this work, we would like to investigate the relationship between the generalization of adversarial training and the robust local features, as the robust local features generalize well for unseen shape variation.

To learn the robust local features, we develop a Random Block Shuffle (RBS) transformation to break up the global structure features on normal adversarial examples.

We continue to propose a new approach called Robust Local Features for Adversarial Training (RLFAT), which first learns the robust local features by adversarial training on the RBS-transformed adversarial examples, and then transfers the robust local features into the training of normal adversarial examples.

To demonstrate the generality of our argument, we implement RLFAT in currently state-of-the-art adversarial training frameworks.

Extensive experiments on STL-10, CIFAR-10 and CIFAR-100 show that RLFAT significantly improves both the adversarially robust generalization and the standard generalization of adversarial training.

Additionally, we demonstrate that our models capture more local features of the object on the images, aligning better with human perception.

Deep learning has achieved a remarkable performance breakthrough on various challenging benchmarks in machine learning fields, such as image classification (Krizhevsky et al., 2012) and speech recognition .

However, recent studies (Szegedy et al., 2014; Goodfellow et al., 2015) have revealed that deep neural network models are strikingly susceptible to adversarial examples, in which small perturbations around the input are sufficient to mislead the predictions of the target model.

Moreover, such perturbations are almost imperceptible to humans and often transfer across diverse models to achieve black-box attacks (Papernot et al., 2017; Liu et al., 2017; Lin et al., 2020) .

Though the emergence of adversarial examples has received significant attention and led to various defend approaches for developing robust models Dhillon et al., 2018; Wang & Yu, 2019; Zhang et al., 2019a) , many proposed defense methods provide few benefits for the true robustness but mask the gradients on which most attacks rely (Carlini & Wagner, 2017a; Athalye et al., 2018; Uesato et al., 2018; Li et al., 2019) .

Currently, one of the best techniques to defend against adversarial attacks (Athalye et al., 2018; Li et al., 2019 ) is adversarial training Zhang et al., 2019a) , which improves the adversarial robustness by injecting adversarial examples into the training data.

Among substantial works of adversarial training, there still remains a big robust generalization gap between the training data and the testing data Zhang et al., 2019b; Ding et al., 2019; Zhai et al., 2019) .

The robustness of adversarial training fails to generalize on unseen testing data.

Recent works (Geirhos et al., 2019; Zhang & Zhu, 2019) further show that adversarially trained models capture more on global structure features but normally trained models are more biased towards local features.

In intuition, global structure features tend to be robust against adversarial perturbations but hard to generalize for unseen shape variations, instead, local features generalize well for unseen shape variations but are hard to generalize on adversarial perturbation.

It naturally raises an intriguing question for adversarial training:

For adversarial training, is it possible to learn the robust local features , which have better adversarially robust generalization and better standard generalization?

To address this question, we investigate the relationship between the generalization of adversarial training and the robust local features, and advocate for learning robust local features for adversarial training.

Our main contributions are as follows:

• To our knowledge, this is the first work that sheds light on the relationship between adversarial training and robust local features.

Specifically, we develop a Random Block Shuffle (RBS) transformation to study such relationship by breaking up the global structure features on normal adversarial examples.

• We propose a novel method called Robust Local Features for Adversarial Training (RLFAT), which first learns the robust local features, and then transfers the information of robust local features into the training on normal adversarial examples.

• To demonstrate the generality of our argument, we implement RLFAT in two currently stateof-the-art adversarial training frameworks, PGD Adversarial Training (PGDAT) and TRADES (Zhang et al., 2019a) .

Empirical results show consistent and substantial improvements for both adversarial robustness and standard accuracy on several standard datasets.

Moreover, the salience maps of our models on images tend to align better with human perception.

In this section, we introduce some notations and provide a brief description on current advanced methods for adversarial attacks and adversarial training.

Let F (x) be a probabilistic classifier based on a neural network with the logits function f (x) and the probability distribution p F (·|x).

Let L(F ; x, y) be the cross entropy loss for image classification.

The goal of the adversaries is to find an adversarial example x ∈ B p (x) := {x : x − x p ≤ } in the p norm bounded perturbations, where denotes the magnitude of the perturbations.

In this paper, we focus on p = ∞ to align with previous works.

Projected Gradient Descent.

Projected Gradient Descent (PGD) ) is a stronger iterative variant of Fast Gradient Sign Method (FGSM) (Goodfellow et al., 2015) , which iteratively solves the optimization problem max x : x −x ∞ < L (F ; x , y) with a step size α:

where U denotes the uniform distribution, and Π B ∞ (x) indicates the projection of the set B ∞ (x).

Carlini-Wagner attack.

Carlini-Wagner attack (CW) (2017b) is a sophisticated method to directly solve for the adversarial example x adv by using an auxiliary variable w:

The objective function to optimize the auxiliary variable w is defined as:

where

The constant k controls the confidence gap between the adversarial class and the true class.

N attack.

N attack (Li et al., 2019 ) is a derivative-free black-box adversarial attack and it breaks many of the defense methods based on gradient masking.

The basic idea is to learn a probability density distribution over a small region centered around the clean input, such that a sample drawn from this distribution is likely to be an adversarial example.

Despite a wide range of defense methods, Athalye et al. (2018) and Li et al. (2019) have broken most previous defense methods ( Dhillon et al., 2018; Buckman et al., 2018; Wang & Yu, 2019; Zhang et al., 2019a) , and revealed that adversarial training remains one of the best defense method.

The basic idea of adversarial training is to solve the min-max optimization problem, as shown in Eq. (4): min

Here we introduce two currently state-of-the-art adversarial training frameworks.

PGD adversarial training.

PGD Adversarial Training (PGDAT) leverages the PGD attack to generate adversarial examples, and trains only with the adversarial examples.

The objective function is formalized as follows:

where x PGD is obtained via the PGD attack on the cross entropy L(F ; x, y).

Zhang et al. (2019a) propose TRADES to specifically maximize the trade-off of adversarial training between adversarial robustness and standard accuracy by optimizing the following regularized surrogate loss:

where

and λ is a hyper-parameter to control the trade-off between adversarial robustness and standard accuracy.

Unlike adversarially trained models, normally trained models are more biased towards the local features but vulnerable to adversarial examples (Geirhos et al., 2019) .

It indicates that, in contrast to global structural features, local features seems be more well-generalized but less robust against adversarial perturbation.

Based on the basic observation, in this work, we focus on the learning of robust local features on adversarial training, and propose a novel form of adversarial training called RLFAT that learns the robust local features and transfers the robust local features into the training of normal adversarial examples.

In this way, our adversarially trained models not only yield strong robustness against adversarial examples but also show great generalization on unseen testing data.

It's known that adversarial training tends to capture global structure features so as to increase invariance against adversarial perturbations (Zhang & Zhu, 2019; Ilyas et al., 2019) .

To advocate for the learning of robust local features during adversarial training, we propose a simple and straightforward image transformation called Random Block Shuffle (RBS) to break up the global structure features of the images, at the same time retaining the local features.

Specifically, for an input image, we randomly split the target image into k blocks horizontally and randomly shuffle the blocks, and then we perform the same split-shuffle operation vertically on the resulting image.

As illustrated in Figure 1 , RBS transformation can destroy the global structure features of the images to some extent and retain the local features of the images.

Then we apply the RBS transformation on adversarial training.

Different from normal adversarial training, we use the RBS-transformed adversarial examples rather than normal adversarial examples as the adversarial information to encourage the models to learn robust local features.

Note that we only use the RBS transformation as a tool to learn the robust local features during adversarial training and will not use RBS transformation in the inference phase.

we refer to the form of adversarial training as RBS Adversarial Training (RBSAT).

To demonstrate the generality of our argument, we consider two currently state-of-the-art adversarial training frameworks, PGD Adversarial Training (PGDAT) and TRADES (Zhang et al., 2019a) , to demonstrate the effectiveness of the robust local features.

We use the following loss function as the alternative to the objective function of PGDAT:

where RBS(·) denotes the RBS transformation; x PGD is obtained via the PGD attack on the cross entropy L(F ; x, y).

Similarly, we use the following loss function as the alternative to the objective function of TRADES:

where

Since the type of input images in the training phase and the inference phase is different (RBS transformed images for training, versus original images for inference), we consider to transfer the knowledge of the robust local features learned by RBSAT to the normal adversarial examples.

Specifically, we present a knowledge transfer scheme, called Robust Local Feature Transfer (RLFT).

The goal of RLFT is to learn the representation that minimizes the feature shift between the normal adversarial examples and the RBS-transformed adversarial examples.

In particular, we apply RLFT on the logit layer for high-level feature alignment.

Formally, the objective functions of robust local feature transfer for PGDAT and TRADES are formalized as follows, respectively:

where f (·) denotes the mapping of the logit layer, and · 2 2 denotes the squared Euclidean norm.

Since the quality of robust local feature transfer depends on the quality of the robust local features learned by RBSAT, we integrate RBSAT and RLFT into an end-to-end training framework, which we refer to as RLFAT (Robust Local Features for Adversarial Training).

The general training process of RLFAT is summarized in Algorithm 1.

Note that the computational cost of RBS transformation (line 7) is negligible in the total computational cost.

Algorithm 1 Robust Local Features for Adversarial Training (RLFAT).

1: Randomly initialize network F (x); 2: Number of iterations t ← 0; 3: repeat 4:

Read a minibatch of data {x 1 , ..., x m } from the training set;

6:

Generate the normal adversarial examples {x Calculate the overall loss following Eq. (10).

Update the parameters of network F through back propagation; 10: until the training converges.

We implement RLFAT in two currently state-of-the-art adversarial training frameworks, PGDAT and TRADES, and have new objective functions to learn the robust and well-generalized feature representations, which we call RLFAT P and RLFAT T :

where η is a hyper-parameter to balance the two terms.

In this section, to validate the effectiveness of RLFAT, we empirically evaluate our two implementations, denoted as RLFAT P and RLFAT T , and show that our models make significant improvement on both robust accuracy and standard accuracy on standard benchmark datasets, which provides strong support for our main hypothesis.

Codes are available online 1 .

Baselines.

Since most previous defense methods provide few benefit in true adversarially robustness (Athalye et al., 2018; Li et al., 2019) , we compare the proposed methods with state-of-theart adversarial training defenses, PGD Adversarial Training (PGDAT) and TRADES (Zhang et al., 2019a) .

Adversarial setting.

We consider two attack settings with the bounded ∞ norm: the white-box attack setting and the black-box attack setting.

For the white-box attack setting, we consider existing strongest white-box attacks: Projected Gradient Descent (PGD) ) and CarliniWagner attack (CW) (Carlini & Wagner, 2017b) .

For the black-box attack setting, we perform the powerful black-box attack, N attack (Li et al., 2019) , on a sample of 1,500 test inputs as it is timeconsuming.

Datasets.

We compare the proposed methods with the baselines on widely used benchmark datasets, namely CIFAR-10 and CIFAR-100 (Krizhevsky & Hinton, 2009 Hyper-parameters.

To avoid posting much concentrate on optimizing the hyper-parameters, for all datasets, we set the hyper-parameter λ in TRADES as 6, set the hyper-parameter η in RLFAT P as 0.5, and set the hyper-parameter η in RLFAT T as 1.

For the training jobs of all our models, we set the hyper-parameters k of the RBS transformation as 2.

More details about the hyper-parameters are provided in Appendix A.

We first validate our main hypothesis: for adversarial training, is it possible to learn the robust local features that have better adversarially robust generalization and better standard generalization?

In Table 1 , we compare the accuracy of RLFAT P and RLFAT T with the competing baselines on three standard datasets.

The proposed methods lead to consistent and significant improvements on adversarial robustness as well as standard accuracy over the baseline models on all datasets.

With the robust local features, RLFAT T achieves better adversarially robust generalization and better standard generalization than TRADES.

RLFAT P also works similarly, showing a significant improvement on the robustness against all attacks and standard accuracy than PGDAT.

The results demonstrate that, the robust local features can significantly improve both the adversarially robust generalization and the standard generalization over the state-of-the-art adversarial training frameworks, and strongly support our hypothesis.

That is, for adversarial training, it is possible to learn the robust local features, which have better robust and standard generalization.

Motivation.

Ding et al. (2019) and Zhang et al. (2019b) found that the effectiveness of adversarial training is highly sensitive to the "semantic-loss" shift of the test data distribution, such as gamma mapping.

To further investigate the performance of the proposed methods, we quantify the smoothness of the models under the distribution shifts of brightness perturbation and gamma mapping.

Loss sensitivity on brightness perturbation.

To quantify the smoothness of models on the shift of the brightness perturbation, we propose to estimate the Lipschitz continuity constant F by using the gradients of the loss function with respect to the brightness perturbation of the testing data.

We adjust the brightness factor of images in the HSV (hue, saturation, value) color space, which we refer to as x b = V(x, α), where α denotes the magnitude of the brightness adjustment.

The lower the value of b F (α) is, the smoother the loss function of the model is:

Loss sensitivity on gamma mapping.

Gamma mapping (Szeliski, 2011 ) is a nonlinear elementwise operation used to adjust the exposure of images by applyingx (γ) = x γ on the original image x. Similarly, we approximate the loss sensitivity under gamma mapping, by using the gradients of the loss function with respect to the gamma mapping of the testing data.

A smaller value indicates a smoother loss function.

Sensitivity analysis.

The results for the loss sensitivity of the adversarially trained models under brightness perturbation are reported in Table 2a , where we adopt various magnitude of brightness adjustment on each testing data.

In Table 2b , we report the loss sensitivity of adversarially trained models under various gamma mappings.

We observe that RLFAT T provides the smoothest model under the distribution shifts on all the three datasets.

The results suggest that, as compared to PGDAT and TRADES, both RLFAT P and RLFAT T show lower gradients of the models on different data distributions, which we can directly attribute to the robust local features.

To further gain insights on the performance obtained by the robust local features, we perform ablation studies to dissect the impact of various components (robust local feature learning and robust local feature transfer).

As shown in Figure 2 , we conduct additional experiments for the ablation studies of RLFAT P and RLFAT T on STL-10, CIFAR-10 and CIFAR-100, where we report the standard accuracy over the clean data and the average robust accuracy over all the attacks for each model.

We first analyze that as compared to adversarial training on normal adversarial examples, whether adversarial training on RBS-transformed adversarial examples produces better generalization and more robust features.

As shown in Figure 2 , we observe that Robust Local Features Learning (RLFL) exhibits stable improvements on both standard accuracy and robust accuracy for RLFAT P and RLFAT T , providing strong support for our hypothesis.

Does robust local feature transfer help?

We further add Robust Local Feature Transfer (RLFT), the second term in Eq. (10), to get the overall loss of RLFAT.

The robust accuracy further increases on all datasets for RLFAT P and RLFAT T .

The standard accuracy further increases also, except for RLFAT P on CIFAR-100, but it is still clearly higher than the baseline model PGDAT.

It indicates that transferring the robust local features into the training of normal adversarial examples does help promote the standard accuracy and robust accuracy in most cases.

We would like to investigate the features of the input images that the models are mostly focused on.

Following the work of Zhang & Zhu (2019) , we generate the salience maps using SmoothGrad (Smilkov et al., 2017) on STL-10 dataset.

The key idea of SmoothGrad is to average the gradients of class activation with respect to noisy copies of an input image.

As illustrated in Figure 3 , all the adversarially trained models basically capture the global structure features of the object on the images.

As compared to PGDAT and TRADES, both RLFAT P and RLFAT T capture more local feature information of the object, aligning better with human perception.

Note that the images are correctly classified by all these models.

For more visualization results, see Appendix B.

Differs to existing adversarially trained models that are more biased towards the global structure features of the images, in this work, we hypothesize that robust local features can improve the generalization of adversarial training.

To validate this hypothesis, we propose a new stream of adversarial training approach called Robust Local Features for Adversarial Training (RLFAT) and implement it in currently state-of-the-art adversarial training frameworks, PGDAT and TRADES.

We provide strong empirical support for our hypothesis and show that the proposed methods based on RLFAT not only yield better standard generalization but also promote the adversarially robust generalization.

Furthermore, we show that the salience maps of our models on images tend to align better with human perception, uncovering certain unexpected benefit of the robust local features for adversarial training.

Our findings open a new avenue for improving adversarial training, whereas there are still a lot to explore along this avenue.

First, is it possible to explicitly disentangle the robust local features from the perspective of feature disentanglement?

What is the best way to leverage the robust local features?

Second, from a methodological standpoint, the discovered relationship may also serve as an inspiration for new adversarial defenses, where not only the robust local features but also the global information is taken into account, as the global information is useful for some tasks.

These questions are worth investigation in future work, and we hope that our observations on the benefit of robust local features will inspire more future development.

Here we show the details of the training hyper-parameters and the attack hyper-parameters for the experiments.

Training Hyper-parameters.

For all training jobs, we use the Adam optimizer with a learning rate of 0.001 and a batch size of 32.

For CIFAR-10 and CIFAR-100, we run 79,800 steps for training.

For STL-10, we run 29,700 steps for training.

For STL-10 and CIFAR-100, the adversarial examples are generated with step size 0.0075, 7 iterations, and = 0.03.

For CIFAR-10, the adversarial examples are generated with step size 0.0075, 10 iterations, and = 0.03.

Attack Hyper-parameters.

For the PGD attack, we use the same attack parameters as those of the training process.

For the CW attack, we use PGD to minimize its loss function with a high confidence parameter (k = 50) following the work of .

For the N attack, we set the maximum number of optimization iterations to T = 200, b = 300 for the sample size, the variance of the isotropic Gaussian σ 2 = 0.01, and the learning rate η = 0.008.

We provide more salience maps of the adversarially trained models on sampled images in Figure 4 .

Original PGDAT TRADES RLFATP RLFATT Original PGDAT TRADES RLFATP RLFATT Figure 4 : More Salience maps of the four models.

For each group of images, we have the original image, and the salience maps of the four models sequentially.

<|TLDR|>

@highlight

We propose a new stream of adversarial training approach called Robust Local Features for Adversarial Training (RLFAT) that significantly improves both the adversarially robust generalization and the standard generalization.
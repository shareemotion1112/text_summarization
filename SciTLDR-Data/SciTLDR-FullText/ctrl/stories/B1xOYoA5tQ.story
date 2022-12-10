Deep models are state-of-the-art for many computer vision tasks including image classification and object detection.

However, it has been shown that deep models are vulnerable to adversarial examples.

We highlight how one-hot encoding directly contributes to this vulnerability and propose breaking away from this widely-used, but highly-vulnerable mapping.

We demonstrate that by leveraging a different output encoding, multi-way encoding, we can make models more robust.

Our approach makes it more difficult for adversaries to find useful gradients for generating adversarial attacks.

We present state-of-the-art robustness results for black-box, white-box attacks, and achieve higher clean accuracy on four benchmark datasets: MNIST, CIFAR-10, CIFAR-100, and SVHN when combined with adversarial training.

The strength of our approach is also presented in the form of an attack for model watermarking, raising challenges in detecting stolen models.

Deep learning models are vulnerable to adversarial examples BID19 ].

Evidence shows that adversarial examples are transferable BID14 ; BID11 ].

This weakness can be exploited even if the adversary does not know the target model under attack, posing severe concerns about the security of the models.

This is because an adversary can use a substitute model for generating adversarial examples for the target model, also known as black-box attacks.

Black-box attacks such as BID4 rely on perturbing input by adding an amount dependent upon the gradient of the loss function with respect to the input of a substitute model.

An example adversarial attack is x adv = x + sign(∇ x Loss(f (x)), where f (x) is the model used to generate the attack.

This added "noise" can fool a model although it may not be visually evident to a human.

The assumption of such gradient-based approaches is that the gradients with respect to the input, of the substitute and target models, are correlated.

Our key observation is that the setup of conventional deep classification frameworks aids in the correlation of such gradients.

Typically, a cross-entropy loss, a soft-max layer, and a one-hot vector encoding for a target label are used when training deep models.

These conventions make a model more vulnerable to black-box attacks.

This setting constrains the encoding length, and the number of possible non-zero gradient directions at the encoding layer.

This makes it easier for an adversary to pick a harmful gradient direction and perform an attack.

We aim to increase the adversarial robustness of deep models.

Our multi-way encoding representation relaxes the one-hot encoding to a real number encoding, and embeds the encoding in a space that has dimension higher than the number of classes.

These encoding methods lead to an increased number of possible gradient directions, as illustrated in Figure 1 .

This makes it more difficult for an adversary to pick a harmful direction that would cause a misclassification of a correctly classified point, generating a targeted or untargeted attack.

Untargeted attacks aim to misclassify a point, while targeted attacks aim to misclassify a point to a specific target class.

Multi-way encoding also helps improve a model's robustness in cases where the adversary has full knowledge of the target model under attack: a white-box attack.

The benefits of multi-way encoding are demonstrated in experiments with four benchmark datasets: MNIST, CIFAR-10, CIFAR-100, and SVHN.We also demonstrate the strength of our approach by introducing an attack for the recent model watermarking algorithm of BID24 , which deliberately trains a model to misclassify (a) (b) (c) Figure 1 : Demonstration of the benefit of relaxing and increasing the encoding dimensionality, for a binary classification problem at the final encoding layer.

C i is the codebook encoding for class i, axis s i represents the output activation of neuron i in the output encoding layer, where i = 1, . . .

, l and l is the encoding dimensionality.

The depicted points are correctly classified points of the green and blue classes.

The arrows depict the possible non-zero perturbation directions sign( ∂Loss ∂si ).

(a) 2D 1of K softmax-crossentropy setup: Only two non-zero gradient directions exist for a 1of K encoding.

Of these two directions, only one is an adversarial direction, depicted in red.

(b) 2D multi-way encoding: Four non-zero perturbation directions exist.

The fraction of directions that now move a point to the adversarial class (red) drops.

(c) 3D multi-way encoding: A higher dimensional encoding results in a significantly lower fraction of gradient perturbations whose direction would move an input from the green ground-truth class to the blue class, or vice versa.

certain watermarked images.

We interpret such watermarked images as adversarial examples.

We demonstrate that the multi-way encoding reduces the transferability of the watermarked images, making it more challenging to detect stolen models.

We summarize our contributions as follows:1.

We show that the traditional 1of K mapping is a source of vulnerability to adversarial gradients.

2.

We propose a novel solution using multi-way encoding to alleviate the vulnerability caused by the 1of K mapping.

3.

We empirically show that the proposed approach improves model robustness against both black-box and white-box attacks.

4.

We also show how to apply our encoding framework in attacking the recently proposed model watermarking scheme of BID24 .

A wide range of work on adversarial attacks and defenses is presented in BID0 .

We review recent attacks and defenses that are closely related to our work and present how alternate output encoding schemes have been utilized in deep classification models.

Attacks.

Adversarial examples are crafted images for fooling a classifier with small perturbations.

Recently, many different types of attacks have been proposed to craft adversarial examples.

We focus on gradient-based attacks such as BID4 ; BID9 ; BID1 ] which deploy the gradient of the loss with respect to the input.

BID4 propose the Fast Gradient Sign Method (FGSM) which generates adversarial images by adding the sign of the input gradients scaled by , where the restricts ∞ of the perturbation.

BID9 propose the Basic Iterative Method (BIM), which is an iterative version of FGSM and is also called Projected Gradient Descent (PGD).

BID12 show that PGD with randomly chosen starting points within allowed perturbation can make an attack stronger.

Defenses.

Most of the state-of-the-art adversarial defenses rely on gradient masking ] by designing a defense that makes it more difficult for an adversary to find useful gradients to generate adversarial examples.

However, BID1 show that works including BID2 ; BID5 BID12 ; BID7 ], are robust to BPDA attack.

These methods are most similar to our approach because they do not rely on obfuscated gradients.

However, BID12 and BID7 use the conventional one-hot (1of K) encoding for both source and target models, while we propose a higher dimensional multiway encoding that obstructs the adversarial gradient search.

Output encoding.

There have been attempts to use alternate output encodings, also known as target encodings, for image classification in deep models.

For example, BID22 and BID16 use an output encoding that is based on Error-Correcting Output Codes (ECOC), for increased performance and faster convergence, but not for adversarial defense.

In contrast, we use an alternate output encoding scheme, multi-way encoding, to make models more robust to adversarial attacks.

In this section we will explain our approach using the following notation: g(x) is the target model to be attacked, and f (x) is the substitute model used to generate a black-box attack for g(x).

In the case of a white-box attack, f (x) is g(x).

Canonical state-of-the-art attacks like FGSM and PGD are gradient-based methods.

Such approaches perturb an input x by an amount dependent upon sign(∇ x Loss(f (x))).

An adversarial example x adv is generated as follows: DISPLAYFORM0 where is the strength of the attack.

Therefore x adv would be a translated version of x, in a vicinity further away from that of the ground-truth class, and thus becomes more likely to be misclassified, resulting in a successful adversarial attack.

If the attack is a targeted one, x could be deliberately moved towards some other specific target class.

This is conventionally accomplished by using the adversarial class as the ground truth when back-propagating the loss, and subtracting the perturbation from the original input.

The assumption being made in such approaches is: DISPLAYFORM1 We now present the most widely used setup for state-of-the-art deep classification networks.

Let the output activation of neuron i in the final encoding (fully-connected) layer be s i , where i = 1, 2, . . .

, l and l is the encoding length.

Then, the softmax prediction y i of s i , and the cross-entropy loss are: DISPLAYFORM2 , and DISPLAYFORM3 respectively, where k is the number of classes.

The partial derivative of the loss with respect to the pre-softmax logit output is: DISPLAYFORM4 The multi-way encoding we propose in this work is Random Orthogonal (RO) output vector encoding generated via Gram-Schmidt orthogonalization.

Starting with a random matrix A = [a 1 |a 2 | . . .

|a n ] ∈ R k×l , the first, second, and k th orthogonal vectors are computed as follows: DISPLAYFORM5 For a classification problem of k classes, we create a codebook C RO ∈ R k×l , where C i = βe i is a length l encoding for class i, and i ∈ 1, . . .

, k, and β is a scaling hyper-parameter dependent upon l. A study on the selection of the length l is presented in the experiments section.

By breaking-away from the 1of K encoding, softmax and cross-entropy become ill-suited for the model architecture and training.

Instead, we use the loss between the output of the encoding-layer and the RO ground-truth vector, Loss(f (x), t RO ), where f (x) ∈ R l .

In our multi-way encoding setup, s and f (x) become equivalent.

Classification is performed using arg min i Loss(f (x), t i RO ).

We use Mean Squared Error (MSE) Loss.

Figure 1 illustrates how using a multi-way and longer encoding results in an increased number of possible gradient directions, reducing the probability of an adversary selecting a harmful direction that would cause misclassification.

For simplicity we consider a binary classifier.

Axis s i in each graph represents the output activation of neuron i in the output encoding layer, where i = 1, . . .

, l. The depicted points are correctly classified points for the green and blue classes.

The arrows depict the sign of non-zero gradients ∂Loss ∂si .

(a) Using a 1of K encoding and a softmax-cross entropy classifier, there are only two directions for a point to move, a direct consequence of 1of K encoding together with Eqn.

4.

Of these two directions, only one is an adversarial direction, depicted in red.

(b) Using 2-dimensional multi-way encoding, we get four possible non-zero gradient directions.

The fraction of directions that now move a correctly classified point to the adversarial class is reduced.

(c) Using a higher dimension multi-way encoding results in a less constrained gradient space compared to that of 1of K encoding.

In the case of attacks formulated following Eqn.

1, this results in 2 l possible gradient directions, rather than l in the case of 1of K encoding.

The fraction of gradients whose direction would move an input from the green ground-truth class to the blue class, or vice versa, decreases significantly.

In addition, multi-way encoding provides additional robustness by increasing the gradients' dimensionality.

We also combine multi-way encoding with adversarial training for added robustness.

We use the following formulation to solve the canonical min-max problem BID12 , BID7 ] against PGD attacks: DISPLAYFORM6 wherep data is the underlying training data distribution, (x, y) are the training points, and λ determines a weight of the loss on clean data together with the adversarial examples at train time.

We conduct experiments on four commonly-used benchmark datasets: MNIST, CIFAR-10, CIFAR-100, and SVHN.

MNIST BID10 ] is a dataset of handwritten digits.

It has a training set of 60K examples, and a test set of 10K examples.

CIFAR-10 [ BID8 ] is a canonical benchmark for image classification and retrieval, with 60K images from 10 classes.

The training set consists of 50K images, and the test set consists of 10K images.

CIFAR-100 BID8 ] is similar to CIFAR-10 in format, but has 100 classes containing 600 images each.

Each class has 500 training images and 100 testing images.

SVHN BID13 ] is an image dataset for recognizing street view house numbers obtained from Google Street View images.

The training set consists of 73K images, and the test set consists of 26K images.

In this work we define a black-box attack as one where the adversary knows the architecture but not the weights, and not the output encoding used.

This allows us to test the efficacy of our proposed encoding when the adversary assumes the conventional 1of K encoding.

We define a white-box attack as one where the adversary knows full information about our model, including the encoding.

.., 3000) of the output encoding layer on the classification accuracy (%) of a model that uses RO multi-way encoding for the MNIST dataset on (1) data perturbed using an FGSM black-box attack with = 0.2 by a model that uses 1of K encoding, and (2) clean data.

As the dimension increases, accuracy increases up to a certain point; We use 2000 for the length of our multi-way encoding layer. .

We conclude: a) g(x) is more vulnerable to attacks when f (x) uses the same encoding, hence the lower reported accuracy.

b) Even when the source and target models are the same and use the same encoding (*), i.e. white-box attacks, RO encoding leads to better accuracy compared to 1of K. c) In brackets is the Pearson correlation coefficient of the gradients of g(x) and f (x) with respect to the input x. Gradients are less correlated when the source and target models use different encodings.

In addition, if the same encoding is used in the source and target models, RO results in a lower correlation compared to 1of K. DISPLAYFORM0

In this section we analyze the case where neither the target nor substitute model undergoes adversarial training.

In all experiments we use RO encoding as the multi-way encoding with dimension 2000 determined by Table 1 and β = 1000.

We first analyze using our multi-way encoding scheme in-depth using the MNIST dataset (4.1.1).

We then present results of comprehensive experiments on white-box and black-box attacks, targeted and untargeted, on the four benchmark datasets (4.1.2).

We conduct experiments to examine how multi-way output encodings can increase adversarial robustness.

We compare models trained on 1of K encodings (A 1of K and C 1of K ) with models having the same architecture but trained on Random Orthogonal output encodings (A RO and C RO ).

Models A and C are LeNet-like CNNs and inherit their names from BID20 .

We use their architecture with dropout before fully-connected layers.

We trained models A and C on MNIST with the momentum optimizer and an initial learning rate of 0.01, momentum = 0.5.

We generated adversarial examples using FGSM with an attack strength = 0.2.

All models achieve ∼99% on the clean test set.

It should be noted that substitute and target models are trained on clean data and do not undergo any form of adversarial training.

Table 2 presents the classification accuracy (%) of target models under attack from various substitute models.

Columns represent the substitute models used to generate adversarial examples and rows represent the target models to be tested on the adversarial examples.

The diagonal represents whitebox attacks, i.e. generating attacks from the target model, and others represent black-box attacks.

Every cell in this table generates attacks from a substitute model f (x) for a target model g(x).It is evident from the results of Table 2 that g(x) is more vulnerable to attacks when f (x) uses the same encoding, hence the lower reported accuracy.

This suggests that a model can be far more robust if the output encoding is hidden from an adversary.

It is also evident from the results of this experiment in Table 2 that even when the source and target models are the same, denoted by (*), i.e. white-box attacks, and use the same encoding, RO Figure 2 : Black-box attacks of varying strength epsilon using 1of K and RO encodings for MNIST.On the left, the substitute model is C 1of K , therefore the attacks generated by this model will have a stronger negative effect on a model trained using 1of K, and a less negative effect on a model that uses a different output encoding, RO.

An analogous argument goes for the plot on the right.encoding leads to better accuracy, and therefore robustness to attack, compared to 1of K encoding.

We present further ablation studies in Appendix A.Finally, Table 2 also reports the Pearson correlation coefficient of sign(∇ x Loss(f (x))) and sign(∇ x Loss(g(x))) used to perturb an input image x to create an adversarial example x adv as shown in Eqn.

1.

These gradients are significantly less correlated when the source and target models use different encodings.

In addition, if the same encoding is used in the source and target models, RO results in a lower correlation compared to 1of K. We report correlation coefficients for all convolutional layers in Appendix B.Figure 2 presents black-box FGSM attacks of varying strengths for 1of K and RO encodings.

On the left is a 1of K substitute model used to generate attacks for a model originally trained using a 1of K encoding (green), and a model originally trained using a RO encoding (blue).

On the right is a RO substitute model used to generate attacks for a model originally trained using a 1of K encoding (green), and a model originally trained using a RO encoding (blue).

This confirms that using a different encoding for the source and target models makes the target model more robust to adversarial attacks; Maintaining a higher accuracy even as the strength of the attack increases.

We now demonstrate how using multi-way encoding helps increase robustness in black-box attacks compared to 1of K encoding for both targeted and untargeted attacks on the four benchmark datasets.

Targeted attacks are attacks where an adversary would like to misclassify an example to a specific incorrect class.

Targeted attacks use the sign of the gradients of the loss on the target class and subtract the perturbation from the original input.

We use PGD attacks with a random start, and follow the PGD parameter configuration of BID12 , BID7 , and BID2 .

Black-box attacks are generated from a substitute model independently trained using a 1of K encoding.

For MNIST and Cifar-10, we follow the experimental settings in BID12 ; for MNIST we use LeNet, for CIFAR-10 we use a ResNet BID6 ] of BID12 .

For Cifar-100 and SVHN we use a WideResNet BID23 ] of depth 28 and 16, respectively, with a width factor 4 and a dropout of 0.3 following BID2 ].

We use the optimizer used by BID12 and BID2 .The result of this experiment is presented in Table 3 .

In the first column we present the average classification accuracy over all classes for untargeted attacks, and find that models using RO encoding are consistently more resilient to black-box attacks compared to models using 1of K encoding.

In the second column we present the average targeted attack success rate over all classes.

RO consistently results in a significantly lower attack success rate compared to 1of K for all four benchmark datasets.

Table 3 : RO (target model) consistently results in a significantly higher classification accuracy for untargeted attacks, and a significantly lower attack success rate compared to 1of K for all four benchmark datasets.

The numbers reported in this table are the average classification and attack success rate over all classes of each dataset.

We note that the clean accuracy for MNIST, CIFAR-10, CIFAR-100, and SVHN is, 99.1, 94.3, 74.5, 96.2, respectively (±0.1 for RO or 1of K).

In this section we analyze the case where target models undergo adversarial training.

This is when adversarial examples are injected in the training data of the target model, making it more difficult for a substitute model to attack.

We compare against state-of-the-art methods, which also use adversarial training.

All black-box attacks in this section are generated from an independently trained copy of BID12 (substitute model).

For adversarial training, we use a mix of clean and adversarial examples for MNIST, CIFAR-10, and CIFAR-100, and adversarial examples only for SVHN following the experimental setup used by BID12 and BID2 .We compare against state-of-the-art defense methods BID12 and BID7 .

Both approaches use a LeNet for MNIST.

BID12 presents results for Cifar-10 on a WideResNet BID6 ), we implement the approach of BID7 on the same architecture and compare both against our approach.

We implement BID12 and BID7 on WideResNet BID23 ] following BID2 and compare against our approach for CIFAR-100 and SVHN.

TAB5 presents the results of combining our multi-way encoding formulation with adversarial training.

We obtain state-of-the-art robustness for white-box and black-box attacks, while at the same time increasing the accuracy on the clean dataset for all four benchmark datasets.

(*) indicates our replication of BID7 using the experimental setting of BID12 on MNIST, also used by ours, that uses only 90% of the training set.5 APPLICATION: ATTACKING MODEL WATERMARKING BID24 introduced an algorithm to detect whether a model is stolen or not.

They do so by adding a watermark to sample images of specific classes and deliberately training the model to misclassify these examples to other specific classes.

This work has demonstrated to be robust even when the model is fine-tuned on a different training set.

We introduce an attack for this algorithm using our multi-way encoding, making it more challenging to detect whether a model is stolen or not.

We do this by fine-tuning the stolen model using multiway encoding, rather than the encoding used in pre-training the model.

We interpret the watermarked image used to deliberately cause a misclassification as an adversarial example.

When the encoding of the substitute and target models is different, adversarial examples become less transferable.

We follow the same CIFAR-10 experimental setup for detecting a stolen model as BID24 : We split the test set into two halves.

The first half is used to fine-tune pre-trained networks, and the second half is used to evaluate new models.

When we fine-tune the 1of K model, we reinitialize the last layer.

When we fine-tune the RO model we replace the output encoding layer with our 2000-dimension fully-connected layer, drop the softmax, and freeze convolutional weights.

Table 5 : Our attack is capable of fooling the watermarking detection algorithm.

Fine-tuning a stolen model using RO encoding remarkably reduces the watermarking detection accuracy, and makes it comparable to the accuracy of models trained from scratch and do not use the stolen model.

The accuracy of fine-tuned models benefits significantly from the pre-trained weights of the stolen model.

We present results on the CIFAR-10 dataset in Table 5 .

When the fine-tuning was performed using the 1of K encoding (also used in pre-training the model), watermarking detection is 87.8%, and when the fine-tuning was performed using the multi-way RO encoding the watermarking detection is only 12.9%.

The watermark detection rate of the model fine-tuned using RO is significantly lower than that fine-tuned using 1of K encoding, and is more comparable to models that are trained from scratch and do not use the stolen model (6.1% and 10.0%).

The accuracy of the fine-tuned models benefits significantly from the pre-trained weights of the stolen model.

We perform ablation studies to further investigate the effectiveness of our RO encoding.

We train the model used in Table 2 with two different combinations of encodings and loss functions.

A.1.1 RO sof tmaxWe evaluate a network that uses RO encoding, a softmax layer, and cross-entropy loss.

We compute the probability of i th class as follows: DISPLAYFORM0 n j=1 exp(s e j ) where s is the normalized final layer representation, e i is the RO encoding vector (ground-truth vector) from the codebook, and n is the number of classes.

We also evaluate a network that uses mean-squared error (MSE) loss with the 1of K encoding.

We generate FGSM attacks with = 0.2 from substitute models A 1of K and C 1of K on MNIST to evaluate the models of Section A.1.1 and Section A.1.2.

We also measure a correlation coefficient of the sign of the input gradients between target and substitute models as explained in Section 4.1.1.

TAB7 demonstrate that RO, among the different target models, achieves the highest accuracy and the lowest input gradient correlation with the substitute model.

In order to measure proper correlations, we average gradients of convolutional layers over channels similar to the way used to generate a gradient-based saliency map BID17 .

Otherwise, the order of convolutional filters affects the correlations and makes it hard to measure proper correlations between models.

In this sense, the correlations at FC1 (before the last layer) may not give meaningful information since neurons in the FC layer do not have a strict ordering.

In Table 8 and 9, we find that the correlations of Conv1 and Conv2 between 1ofK models are much higher than those of RO models.

In addition, even though RO models used the same output encoding, they are not highly correlated.

TAB10 shows that the correlations between RO and 1ofK are also low.

<|TLDR|>

@highlight

We demonstrate that by leveraging a multi-way output encoding, rather than the widely used one-hot encoding, we can make deep models more robust to adversarial attacks.

@highlight

This paper proposes replacing the final cross-entropy layer trained on one-hot labels in classifiers by encoding each label as a high-dimensional vector and training the classifier to minimize L2 distance from the encoding of the correct class.

@highlight

Authors propose new method against adversarial attacks that shows significant amount of gains compared to baselines
Existing neural networks are vulnerable to "adversarial examples"---created by adding maliciously designed small perturbations in inputs to induce a misclassification by the networks.

The most investigated defense strategy is adversarial training which augments training data with adversarial examples.

However, applying single-step adversaries in adversarial training does not support the robustness of the networks, instead, they will even make the networks to be overfitted.

In contrast to the single-step, multi-step training results in the state-of-the-art performance on MNIST and CIFAR10, yet it needs a massive amount of time.

Therefore, we propose a method, Stochastic Quantized Activation (SQA) that solves overfitting problems in single-step adversarial training and fastly achieves the robustness comparable to the multi-step.

SQA attenuates the adversarial effects by providing random selectivity to activation functions and allows the network to learn robustness with only single-step training.

Throughout the experiment, our method demonstrates the state-of-the-art robustness against one of the strongest white-box attacks as PGD training, but with much less computational cost.

Finally, we visualize the learning process of the network with SQA to handle strong adversaries, which is different from existing methods.

As Convolutional Neural Networks (CNNs) stand out as a solution to many real world computer vision tasks BID17 BID0 BID18 BID20 , achieving a certain level of robustness has become indispensable for security-sensitive systems, such as autonomous driving, robot vision, and identity authentication.

However, recent studies BID26 BID10 have shown that the existing CNNs are vulnerable to small perturbations of the input that are intentionally or adversarially designed to fool the system.

The adversarial attack is a serious problem since these maliciously designed attacks have shown effective in physical world scenarios, where inputs are obtained from signals of cameras and other sensors BID15 BID8 .

Another disconcerting feature about adversarial examples is their transferability across different models BID26 BID23 BID21 ) that enables black-box attacks.

In other words, adversarial examples can be designed from a different model without having the information about the target network.

The most studied defense strategy against adversarial attacks is adversarial training BID10 BID16 BID28 BID22 , which increases robustness by augmenting training data with adversarial examples.

Since adversarial training requires the model to train adversarial examples in addition to training data, the model consumes extra time to learn features of the examples via fine-tuning.

Even though the model is trained on more examples, it still might be defenseless to new examples generated by different attack due to the overfitting problem.

Recently, BID22 have found that adversarial training on examples created via gradient descent with random restarts, Projected Gradient Descent (PGD) training, results in a universally and partially unbreakable model on MNIST and CIFAR-10.

This method shows the state-of-the-art performance on MNIST and CIFAR-10 to the best of our knowledge, but the examples are created iteratively and the time increases proportionally to the number of steps.

For instance, in our CIFAR-10 training, FGSM training on ResNet18 took less than 2 hours for 30 epochs; however, PGD training took about 30 hours for the same epochs.

Thus, it is essential to find the universal method that is resistant against all of the attacks, with less computational cost.

Since high dimensional representations of the neural networks give extreme complexity to the boundary of trained manifolds BID27 BID7 , we start from the idea that is to reduce degrees of freedom available to the adversary.

In this sense, we propose a Stochastic Quantized Activation (SQA) that provides stochastic randomness to the output of an original activation and reduces the opportunity for the attacker to make adversaries.

The best advantage of SQA is that SQA with fast adversarial training, training with only FGSM examples, allows the model to have robustness comparable to PGD training with less computational cost.

In particular, although SQA is one of the obfuscated gradients defined by BID1 , iterative optimization-based methods does not successfully circumvent our defense.

Besides, SQA can be combined with any deep learning models with a few lines of code but guarantees a certain level of robustness against adversarial attacks.

In this paper, we first explain existing methods for adversarial attacks and defenses we refer in Section 2.

We separate the existing defense strategies into two categories and analyze the strengths and weaknesses.

In Section 3, we introduce the procedure of SQA, with an algorithm described in 1.

In Section 4, we show our experimental results on MNIST and CIFAR-10 and compare with existing defense systems.

Lastly, we visualize the penultimate layer of our networks and compare how SQA with fast adversarial training, learns differently from the existing methods.

Section 5 concludes the work and contributions of this paper are as follows:• We propose a Stochastic Quantized Activation (SQA) which achieves a significant level of robustness combined with FGSM training, comparable to state-of-the-art PGD adversarial training with much less computational cost.• Due to the efficiency and the flexibility of the proposed method, it can be fastly and widely applied to any existing deep neural networks and combine with other types of defense strategies.• We analytically demonstrate how SQA makes the model robust against adversaries in highlevel and low-level by using t-SNE, and plotting activation maps.

In this section, we investigate the existing methods of adversarial attacks and defenses that appear in the following subsections.

First, we define the adversarial examples with the notations formally used in this paper.

Let x denote input and y denote the prediction of the input from the DNN classifier f , y = f (x).

Then, an adversarial example is crafted by adding a malicious noise η into the original input x, causing a different prediction from the true label, y * .

The formal representation is as follows, where x is an adversarial example and is the noise level.

DISPLAYFORM0

Fast Gradient Sign Method (FGSM) is a fast single-step method to create adversarial examples proposed by BID10 .

The authors suggest the adversarial examples are crafted because of the effects of the linear summation in DNNs, and the algorithm is as follows.

DISPLAYFORM0 Here J(f (x), y * ) is the loss between the output prediction f (x) and the true label y * .

However, calculating the loss based on the difference between predictions and true labels makes the label leaking effect BID16 , so one simple way to prevent it is to put the prediction y instead of y * .

The intuition behind of the Equation 2 is that increasing loss J by perturbing the input x adding the gradient of loss, which makes the prediction get out of the extrema.

Projected Gradient Descent (PGD) is one of the strongest known white box attacks BID22 .

It is a multi-step variant of FGSM, which means that it finds the adversarial perturbation η n by using the same equation from FGSM, but iteratively.

What makes this attack stronger is that it finds the adversary from starts with random -uniform perturbation clipped in the range of the normalized pixel values, [0,1].

DISPLAYFORM1 is strong optimization-based iterative attack proposed by BID4 .

It uses Adam BID14 to optimize over the adversarial perturbation η n using an auxiliary variable ω n and solves the equation below.

DISPLAYFORM2 The function f (·) is defined as DISPLAYFORM3 and we can determine the confidence with which the misclassification occurs by adjusting κ.

Adversarial training increases robustness by augmenting training data in relation to adversarial examples.

Previous studies BID10 BID16 BID28 have shown that adversarially training models improve the classification accuracy when presenting them with adversarial examples.

However, the intrinsic problem of this method is the high cost associated with additionally generating adversarial examples and patching them into a training batch.

For this reason, practical adversarial training on a large scale dataset such as ImageNet uses fast-generated adversarial examples using FGSM only for training data.

However, BID22 have shown that FGSM adversaries don't increase robustness especially for large since the network overfits to these adversarial examples.

They instead, suggest to train the network with a multi-step FGSM k , PGD adversaries, and it shows the state-of-the-art performance on MNIST and CIFAR-10.Obfuscated Gradients make the network hard to generate adversaries by not having useful gradients.

Recently, BID1 defined three types of obfuscated gradients: Shattered Gradients, Stochastic Gradients, and Exploding & Vanishing Gradients.

BID6 BID3 BID25 BID31 have considered one of these gradients, but BID1 make the attacks which successfully circumvent the defense by making 0% accuracy on 6 out of 7 defenses at ICLR2018.

SQA can be considered as both shattered gradients and stochastic gradients.

However, we found that our method does not overfit to the adversarial examples and shows robustness against the different type of attacks including the one used to break obfuscated gradients.

The next section explains the details of our method.

In this section, we introduce the concept of SQA starting from a typical low-bit representation in DNNs as prerequisites BID5 .

Then, we show the procedure of our quantization stochasticity.

The difference between typical low-bit DNNs BID12 BID5 BID13 and our proposed method is that we only consider the quantization of activations except weight vectors.

We found that this does not significantly slow down the training with PyTorch BID24 but maintains full-precision weight representation, which enables easier convergence than BNNs without additional training strategies.

BinaryConnect constraints the weights to either +1 or -1 during propagations BID5 .

Two types of binarization, deterministic and stochastic, are introduced.

They are respectively described by the following equations.

DISPLAYFORM0 w b = +1 with probability p = σ(w), −1 with probability 1 − p.where DISPLAYFORM1 BNNs are originally designed to reduce the significant amount of memory consumption and costs taken by propagating in full-precision networks.

Recently, however, BID9 shows another benefit of low-precision neural networks, which improves robustness against some adversarial attacks.

Thus, we propose SQA, a stochastic activation function giving the quantized threshold effects into vanilla CNNs, which is described in Algorithm 1.

The algorithm can be divided into three steps.• Min-Max normalization with scaling DISPLAYFORM2 Let h i be a latent space, the output from a ith convolutional layer after ReLU activation.

We first perform min-max normalization, making h i ranging from 0 to 1.

Then we scale the h i ranging from 0 to λ by multiplying a scale factor λ, which determines the level of quantization from binary to quaternary in our experiment.

In the next step, we stochastically quantize the scaled g i as g i presented in the below equation.

DISPLAYFORM3 This makes g i converge into the closest or second closest integers, either g i or g i + 1 with a probability of each, 1 -(g i − g i ) and g i − g i .

For instance, if we let g i = 1.7, then the probability of g i = 1 is 0.3 and g i = 2 is 0.7.

The final step is rescaling g i into the range of original output ReLU activation h i .

To rescale the value within the original range, g i is first divided by λ, and inverse min-max normalization is applied as presented in Algorithm 1.Since it is impossible to find exact derivatives with respect of discretized activations, an alternative is to approximate it by a straight through estimator BID2 .

The concept of a straight through estimator is fixing the incoming gradients to a threshold function equal to its outgoing gradients, ignoring the derivative of the threshold function itself.

This is the reason why we rescale g i to the original range of h i .

In other words, we do not want to consider the scale factors multiplied in the activation function when we use a straight through estimator.

In this experiment, we show the feasibility of our approach with several different settings on MNIST and CIFAR-10 using PyTorch BID24 .

We use Adversarial Box BID30

For MNIST, we use a baseline model as a Vanilla CNN consisting of three convolutional layers with two fully-connected layers on the top.

Since there is a correlation between robustness and model capacity BID22 , we use two networks with different channel sizes and increase channels by a factor of 2.

This result in networks with each (16, 32, 64) and (64, 128, 256) filters and they are denoted as SMALLandLARGE in TAB0 .

We apply SQA on the first and second layers with each λ = 1 and 2.

We use Stochastic Gradient Descent (SGD) with learning rate of 0.1, momentum of 0.9, and weight decay of 5e-4.

We adjust the learning rate decreasing by 0.1 after every 30 steps within total 100 epochs.

For CIFAR-10, we use ResNet model BID11 as a baseline.

We adpot 34 and 101 layers of ResNets denoted as RES34 and RES101 in Table 2 .

An interesting property found on training CIFAR-10 dataset is that quantization with stochasticity shows much higher accuracy rather than deterministic quantization.

It seems reasonable since stochasticity is able to provide higher capacity to learn complex RGB images.

We apply SQA on the output from the first layer of ResNet and its bottleneck module with each λ = 1 and 2.

The same hyper-parameters are applied to the MNIST training except with total epochs of 350 and decreasing learning rate by 0.1 after every 150 steps.

Throughout the experiments, different l ∞ intensity levels are applied to the attacks.

For MNIST, = 0.2 and 0.3 are used for FGSM and C&W attacks to give strong adversarial perturbations.

We choose 40 steps for C&W attacks.

Also, we set = 0.2, a step size of 0.01 and 40 steps for PGD attacks.

For CIFAR-10, = 4, 8 are considered for the adversarial attacks.

We choose 30 steps for C&W attacks.

For PGD attacks we fix 7 steps and the step size as 2 with random perturbation 8.

Note that the values for MNIST are in the scale of (0,1) and (0,255) for CIFAR-10.Step sizes for the attacks are chosen to be consistent with BID22 .

Since quantizing the weights or activation lowers the accuracy on clean images BID5 BID12 , it is important to find where to put SQA modules in networks.

Thus, we investigate the nth layer-wise quantization applying the deterministic quantization from the first layer of CNN to the third.

The result is shown in FIG0 .

It is clear that applying quantization on the earlier steps gives higher robustness.

This observation is another proof for the argument from BID19 that a small perturbation in an image is amplified to a large perturbation in a higher-level representation so that quantizing the activations in lower-level representation gives more robustness.

We further, empirically found that giving binary quantization on the first layer and ternary quantization on the second layer provides less degradation for accuracy and a fair amount of robustness.

Table 2 : Performance comparison for CIFAR10 between full-precision and SQA SQA v.s. Full-Precision We explore the robustness of SQA against three types of adversarial attacks and the result is shown in TAB0 .

The networks are all trained with fast single-step adversaries and we could find two known, but interesting properties from the experiments.

First, FGSM training the full-precision networks, denoted as SMALL f ull , LARGE f ull , makes themselves overfit to the adversaries.

They show depressed accuracy on especially, PGD attacks, nearly close to 0.

However, SQA models does not overfit to the adversaries.

Even though SQA models show lower performance on FGSM attacks, they exhibit remarkably high accuracy on the other adversarial examples that have not seen before.

The second interesting fact is that the correlation between robustness and model capacity.

BID22 have shown that increasing model capacity helps to train the network against strong adversaries successfully.

Our experiment also confirms this phenomenon.

The performance of LARGE SQA is stronger than SMALL SQA against FGSM attacks and more than ten times robust against PGD attacks.

This result shows that the model capacity not only increases robustness against the adversaries that have been learned but also prevent overfitting to them.

SQA v.s. Full-Precision We performed experiments on CIFAR-10 to show the effectiveness of SQA on the RGB image dataset.

We tried the same types of white-box attacks as in MNIST experiments, and the result is shown in Table 2 .

Instead of training Vanilla networks, we adopt ResNet BID11 since the Vanilla networks are hard to learn useful features on CIFAR-10.

Two different ResNets are used for comparing robustness regarding the model capacity, and we found the same phenomena as in MNIST experiments.

In other words, SQA module helps to get out of overfitting to the FGSM adversaries, and the larger capacity provides, the higher robustness against different types of attacks.

We compare our module, SQA, with recently proposed defenses including the state-of-the-art, BID22 .

We also include SAP, PixelDefend, and Thermometer BID6 BID25 BID3 since they use stochastic gradients or shattered gradients that are one of the obfuscated gradients, where our method belongs to.

TAB2 shows the performance 1 comparison against PGD and C&W attacks for l ∞ ( = 8).

Note that the architectures from the defenses on TAB2 are all different and it is impossible to exactly compare the robustness.

We denote the architectures as RESN k W,C , where W stands for Wider ResNets, N is depth, C is the channel size of the first layer, and k is the widen factor.

As BID1 claimed, our method is more robust against gradient-based PGD rather than optimization-based C&W, pushing the state-of-the-art accuracy to 52% against PGD attacks.

Also, it shows a fair amount of accuracy against C&W attacks comparable to Adv.

Training.

This result shows a dramatic impact in a sense that other methods based on obfuscated gradients almost fail to defend against these strong adversaries.

Model PGD C&W =8 SAP BID6 RES20 -0 PixelDefend BID25 RES62 9 -Thermometer BID3 Table 4 : Average training time (sec) for one iteration on ResNet34 for CIFAR-10

In this subsection, we explore the time complexity of adversarial training both single-steps and multi-steps.

Let τ as the time taken by forward and backward propagation in neural networks, κ is number of steps to find adversaries, and υ is for other processing times including data loading, weight update and etc.

Then, we can define the time complexity of adversarial training as follows, DISPLAYFORM0 Then when we consider α as processing time for SQA module and compare SQA + FGSM training with PGD training, DISPLAYFORM1 As we can see in Table 4 , SQA + FGSM training is almost 18 times faster than PGD training where κ is 100.

In this subsection, we analyze the penultimate layers of the network trained with our method comparing with two full-precision networks: with no defense and with FGSM training.

We use C&W attacks to make adversaries with the parameters described in Section 4.1.

We use two different ways to visualize the penultimate layers in high level and low level by using t-SNE (van der Maaten & Hinton, 2008) and plotting activation maps with both clean images and adversarial examples.

Firstly, FIG1 shows t-SNE results from the penultimate layer of our network and a point in t-SNE is represented as an image.

We select four classes to clearly show how the networks learn and what happens when adversarial noise is added.

Here, we demonstrate that the full-precision network trained with FGSM does not correctly classify the classes against the adversarial attacks, as depicted in (B).

However, only (C) which is our method shows that the clusters are less broken compared to the other methods.

Furthermore, in light of the fact that the robust classifier requires a more complicated decision boundary BID22 , our model seems to have the complicated one by learning adversarial examples.

Secondly, we closely look into the penultimate layer in a low level by plotting the each of the activations.

In this time, a point of an activation map stands for the mean value of the activations across about a thousand images per classes.

We found that the yellow spots which are the highest values stay in the same location under the adversarial attack, as depicted in (C), FIG2 .

In other words, our method shows stable activation frequencies against the adversarial attacks, but training full-precision models with FGSM adversaries does not help to increase robustness, as shown in (B).

In this paper, we have found that SQA, a stochastic quantization in an activation function, make existing neural networks prevent overfitting to FGSM training.

It provides stochastic randomness in quantization to learn a robust decision boundary against adversarial attacks with FGSM training.

Our method not only shows dramatic improvements against one of the strongest white-box attacks, comparable to state-of-the-art PGD training but also significantly reduces the computational cost.

Throughout visualizing the penultimate layers of our network, we demonstrate that the network learns strong adversaries without overfitting.

We expect that SQA could be fastly and widely applied to other defense strategies because of its efficiency and flexibility.

In the future work, we plan to experiment on large scale image datasets.

<|TLDR|>

@highlight

This paper proposes Stochastic Quantized Activation that solves overfitting problems in FGSM adversarial training and fastly achieves the robustness comparable to multi-step training.

@highlight

The paper proposes a model to improve adversarial training by introducing random perturbations in the activations of one of the hidden layers
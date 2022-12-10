Neural network quantization is becoming an industry standard to efficiently deploy deep learning models on hardware platforms, such as CPU, GPU, TPU, and FPGAs.

However, we observe that the conventional quantization approaches are vulnerable to adversarial attacks.

This paper aims to raise people's awareness about the security of the quantized models, and we designed a novel quantization methodology to jointly optimize the efficiency and robustness of deep learning models.

We first conduct an empirical study to show that vanilla quantization suffers more from adversarial attacks.

We observe that the inferior robustness comes from the error amplification effect, where the quantization operation further enlarges the distance caused by amplified noise.

Then we propose a novel Defensive Quantization (DQ) method by controlling the Lipschitz constant of the network during quantization, such that the magnitude of the adversarial noise remains non-expansive during inference.

Extensive experiments on CIFAR-10 and SVHN datasets demonstrate that our new quantization method can defend neural networks against adversarial examples, and even achieves superior robustness than their full-precision counterparts, while maintaining the same hardware efficiency as vanilla quantization approaches.

As a by-product, DQ can also improve the accuracy of quantized models without adversarial attack.

Neural network quantization BID10 BID34 BID12 ) is a widely used technique to reduce the computation and memory costs of neural networks, facilitating efficient deployment.

It has become an industry standard for deep learning hardware.

However, we find that the widely used vanilla quantization approaches suffer from unexpected issues -the quantized model is more vulnerable to adversarial attacks ( FIG1 ).

Adversarial attack is consist of subtle perturbations on the input images that causes the deep learning models to give incorrect labels BID28 .

Such perturbations are hardly detectable by human eyes but can easily fool neural networks.

Since quantized neural networks are widely deployed in many safety-critical scenarios, e.g., autonomous driving BID1 , the potential security risks cannot be neglected.

The efficiency and latency in such applications are also important, so we need to jointly optimize them.

The fact that quantization leads to inferior adversarial robustness is counter intuitive, as small perturbations should be denoised with low-bit representations.

Recent work BID31 ) also demonstrates that quantization on input image space , i.e. color bit depth reduction, is quite effective to defend adversarial examples.

A natural question then rises, why the quantization operator is yet effective when applied to intermediate DNN layers ?

We analyze that such issue is caused by the error amplification effect of adversarial perturbation BID15 -although the magnitude of perturbation on the image is small, it is amplified significantly when passing through deep neural network (see FIG4 .

The deeper the layers are, the more significant such side effect is.

Such amplification pushes values into a different quantization bucket, which is undesirable.

We conducted empirical experiments to analyze how quantization influences the activation error between clean and adversarial samples FIG4 ): when the magnitude of the noise is small, activation quantization is capable of reducing the errors by eliminating small perturbations; However, when the magnitude of perturbation is larger than certain threshold, quantization instead amplify the errors, which causes the quantized model to make mistakes.

We argue that this is the main reason causing the inferior robustness of the quantized models.

In this paper, we propose Defensive Quantization (DQ) that not only fixes the robustness issue of quantized models, but also turns activation quantization into a defense method that further boosts adversarial robustness.

We are inspired by the success of image quantization in improving robustness.

Intuitively, it will be possible to defend the attacks with quantization operations if we can keep the magnitude of the perturbation small.

However, due to the error amplification effect of gradient based adversarial samples, it is non-trivial to keep the noise at a small scale during inference.

Recent works BID5 BID21 have attempted to make the network non-expansive by controlling the Lipschitz constant of the network to be smaller than 1, which has smaller variation change in its output than its input.

In such case, the input noise will not propagate through the intermediate layers and impact the output, but attenuated.

Our method is built on the theory.

Defensive quantization not only quantizes feature maps into low-bit representations, but also controls the Lipschitz constant of the network, such that the noise is kept within a small magnitude for all hidden layers.

In such case, we keep the noise small, as in the left zone of FIG4 , quantization can reduce the perturbation error.

The produced model with our method enjoys better security and efficiency at the same time.

Experiments show that Defensive Quantization (DQ) offers three unique advantages.

First, DQ provides an effective way to boost the robustness of deep learning models while maintains the efficiency.

Second, DQ is a generic building block of adversarial defense, which can be combined with other adversarial defense techniques to advance state-of-the-art robustness.

Third, our method makes quantization itself easier thanks to the constrained dynamic range.

Neural network quantization BID10 BID23 BID33 BID6 BID34 ) are widely adopted to enable efficient inference.

By quantizing the network into low-bit representation, the inference of network requires less computation and less memory, while still achieves little accuracy degradation on clean images.

However, we find that the quantized models suffer from severe security issues -they are more vulnerable against adversarial attacks compared to full-precision models, even when they have the same clean image accuracy.

Adversarial perturbation is applied to the input image, thus it is most related to activation quantization BID7 .

We carry out the rest of the paper using ReLU6 based activation quantization BID24 , as it is computationally efficient and is widely adopted by modern frameworks like TensorFlow BID0 .

As illustrated in Figure 2 , a quantized convolutional network is composed of several quantized convolution block, each containing a serial of conv + BN + ReLU6 + linear quantize operators.

As the quantization operator has DISPLAYFORM0 Lipschitz Regularization 1 Figure 2 .

Defensive quantization with Lipschitz regularization.0 gradient almost everywhere, we followed common practice to use a STE BID4 function y = x + stop gradient[Quantize(x) − x] for gradient computation, which also eliminates the obfuscated gradient problem BID2 .Our work bridges two domains: model quantization and adversarial defense.

Previous work BID8 claims binary quantized networks can improve the robustness against some attacks.

However, the improvement is not substantial and they used randomized quantization, which is not practical for real deployment (need extra random number generators in hardware).

It also causes one of the obfuscated gradient situations BID2 : stochastic gradients, leading to a false sense of security.

BID22 tries to use quantization as an effective defense method.

However, they employed Tanh-based quantization, which is not hardware friendly on fixed-point units due to the large overhead accessing look-up table.

Even worse, according to our re-implementation, their method actually leads to severe gradient masking problem BID19 during adversarial training, due to the nature of Tanh function (see A.1 for detail).

As a result, the actual robustness of this work under black-box attack has no improvement over full-precision model and is even worse.

Therefore, there is no previous work that are conducted under real inference setting to study the quantized robustness for both black-box and white-box.

Our work aim to raise people's awareness about the security of the actual deployed models.

Given an image X, an adversarial attack method tries to find a small perturbation ∆ with constraint ||∆|| ≤ , such that the neural network gives different outputs for X and X adv X + ∆. Here is a scalar to constrain the norm of the noise (e.g., = 8 is commonly used when we represent colors from 0-255), so that the perturbation is hardly visible to human.

For this paper we choose to study attacks defined under || · || ∞ , where each element of the image can vary at most to form an adversary.

We introduce several attack and defense methods used in our work in the following sections.

Random Perturbation (Random) Random perturbation attack adds a uniform sampled noise within [− , ] to the image, The method has no prior knowledge of the data and the network, thus is considered as the weakest attack method.

Fast Gradient Sign Method (FGSM) & R+FGSM Goodfellow et al. proposed a fast method to calculate the adversarial noise by following the direction of the loss gradient ∇ X L(X, y), where L(X, y) is the loss function for training (e.g. cross entropy loss).

The adversarial samples are computed as: DISPLAYFORM0 As FGSM is an one-step gradient-based method, it can suffer from sharp curvature near the data points, leading a false direction of ascent.

Therefore, BID29 proposes to prepend FGSM by a random step to escape the non-smooth vicinity.

The new method is called R+FGSM, defined as follows, for parameters and 1 (where 1 < ): DISPLAYFORM1 In our paper, we set 1 = /2 following BID29 .

BID14 suggests a simple yet much stronger variant of FGSM by applying it multiple times iteratively with a small step size α.

The method is called BIM, defined as: DISPLAYFORM2 DISPLAYFORM3 where clip X means clipping the result image to be within the -ball of X. In BID16 , the BIM is prepended by a random start as in R+FGSM method.

The resulting attack is called PGD, which proves to be a general first-order attack.

In our experiments we used PGD for comprehensive experiments as it proves to be one of the strongest attack.

Unlike BID16 that uses a fixed and α, we follow BID14 ; BID26 to use α = 1 and number of iterations of min( + 4, 1.25 ) , so that we can test the model's robustness under different strength of attacks.

Current defense methods either preprocess the adversarial samples to denoise the perturbation BID31 BID26 BID15 or making the network itself robust BID30 BID20 BID16 BID14 BID29 .

Here we introduced several defense methods related to our experiments.

BID31 proposes to detect adversarial images by squeezing the input image.

Specifically, the image is processed with color depth bit reduction (5 bits for our experiments) and smoothed by a 2 × 2 median filter.

If the low resolution image is classified differently as the original image, then this image is detected as adversarial.

Adversarial Training Adversarial training BID16 BID14 BID29 ) is currently the strongest method for defense.

By augmenting the training set with adversarial samples, the network learns to classify adversarial samples correctly.

As adversarial FGSM can easily lead to gradient masking effect BID19 , we study adversarial R+FGSM as in BID29 .

We also experimented with PGD training BID16 .Experiments show that above defense methods can be combined with our DQ method to further improve the robustness.

The robustness has been tested under the aforementioned attack methods.

Conventional neural network quantization is more vulnerable to adversarial attacks.

We experimented with VGG-16 BID25 ) and a Wide ResNet BID32 of depth 28 and width 10 on CIFAR-10 (Krizhevsky & Hinton, 2009) dataset.

We followed the training protocol as in BID32 .

Adversarial samples are generated with a FGSM (Goodfellow et al.) attacker ( = 8) on the entire test set.

As in FIG1 , the clean image accuracy doesn't significantly drop until the model is quantized to 4 bits ( FIG1 ).

However, under adversarial attack, even with 5-bit quantization, the accuracy drastically decreased by 25.3% and 9.2% respectively.

Although the full precision model's accuracy has dropped, the quantized model's

Small increase of perturbed range.

The large accumulated error pushes the activation to a different quantization bucket

Activations stay within the same quantization bucket.

1 Figure 4 .

The error amplification effect prevents activation quantization from defending adversarial attacks.accuracy dropped much harder, showing that the conventional quantization method is not robust.

Clean image accuracy used to be the sole figure of merit to evaluate a quantized model.

We show that even when the quantized model has no loss of performance on clean images, it can be much more easily fooled compared to full-precision ones, thus raising security concerns.

Input image quantization, i.e., color bit depth reduction is an effective defense method BID31 .

Counter intuitively, it does not work when applied to hidden layers, and even make the robustness worse.

To understand the reason, we studied the effect of quantization w.r.t.

different perturbation strength.

We first randomly sample 128 images X from the test set of CIFAR-10, and generate corresponding adversarial samples X adv .

The samples are then fed to the trained Wide ResNet model.

To mimic different strength of activation perturbation, we vary the from 1 to 8.

We inspected the activation after the first convolutional layer f 1 (Conv + BN + ReLU6), denoted as A 1 = f 1 (X) and A DISPLAYFORM0 To measure the influence of perturbation, we define a normalized distance between clean and perturbed activation as: DISPLAYFORM1 We compare D(A, A adv ) and D(Quantize(A), Quantize(A adv )), where Quantize indicates uniform quantization with 3 bits.

The results are shown in FIG4 .

We can see that only when is small, quantization helps to reduce the distance by removing small magnitude perturbations.

The distance will be enlarged when is larger than 3.The above experiment explains the inferior robustness of the quantized model.

We argue that such issue arises from the error amplification effect BID15 , where the relative perturbed distance will be amplified when the adversarial samples are fed through the network.

As illustrated in Figure 4 , the perturbation applied to the input image has very small magnitude compared to the image itself (±8 versus 0 − 255), corresponding to the left zone of FIG4 (desired), where quantization helps to denoise the perturbation.

Nevertheless, the difference in activation is amplified as the inference carries on.

If the perturbation after amplification is large enough, the situation corresponds to the right zone (actual) of FIG4 , where quantization further increases the normalized distance.

Such phenomenon is also observed in the quantized VGG-16.

We plot the normalized distance of each convolutional layer's input in FIG4 .

The fewer bits in the quantized model, the more severe the amplification effect.

Given the robustness limitation of conventional quantization technique, we propose Defensive Quantization (DQ) to defend the adversarial examples for quantized models.

DQ suppresses the noise amplification effect, keeping the magnitude of the noise small, so that we can arrive at the left zone FIG4 where quantization helps robustness instead of making it worse.

We control the neural network's Lipschitz constant BID28 BID3 BID5 to suppress network's amplification effect.

Lipschitz constant describes: when input changes, how much does the output change correspondingly.

For a function f : X → Y , if it satisfies DISPLAYFORM0 for a real-valued k ≥ 0 and some metrics D X and D Y , then we call f Lipschitz continuous and k is the known as the Lipschitz constant of f .

If we consider a network f with clean inputs x 1 = X and corresponding adversarial inputs x 2 = X adv , the error amplification effect can be controlled if we have a small Lipschitz constant k (in optimal situation we can have k ≤ 1).

In such case, the error introduced by adversarial perturbation will not be amplified, but reduced.

Specifically, we consider a feed-forward network composed of a serial of functions: DISPLAYFORM1 where φ i can be a linear layer, convolutional layer, pooling, activation functions, etc.

Denote the Lipschitz constant of a function f as Lip(f ), then for the above network we have DISPLAYFORM2 Lip(φ i )As the Lipschitz constant of the network is the product of its individual layers' Lipschitz constants, Lip(f ) can grow exponentially if Lip(φ i ) > 1.

This is the common case for normal network training BID5 , and thus the perturbation will be amplified for such a network.

Therefore, to keep the Lipschitz constant of the whole network small, we need to keep the Lipschitz constant of each layer Lip(φ i ) ≤ 1.

We call a network with Lip(φ i ) ≤ 1, ∀i = 1, ..., L a non-expansive network.

We describe a regularization term to keep the Lipschitz constant small.

Let us first consider linear layers with weight W ∈ R cout×cin under || · || 2 norm.

The Lipschitz constant is by definition the spectral norm of W : ρ(W), i.e., the maximum singular value of W. Computing the singular values of each weight matrix is not computationally feasible during training.

Luckily, if we can keep the weight matrix row orthogonal, the singular values are by nature equal to 1, which meets our non-expansive requirements.

Therefore we transform the problem of keeping ρ(W) ≤ 1 into keeping W T W ≈ I, where I is the identity matrix.

Naturally, we introduce a regularization term ||W T W − I||, where I is the identity matrix.

Following BID5 , for convolutional layers with weight W ∈ R cout×cin×k×k , we can view it as a two-dimension matrix of shape W ∈ R cout×(cinkk) and apply the same regularization.

The final optimization objective is: DISPLAYFORM3 where L CE is the original cross entropy loss and W denotes all the weight matrices of the neural network.

β is the weighting to adjust the relative importance.

The above discussion is based on simple feed forward networks.

For ResNets in our experiments, we also follow BID5 to modify the aggregation layer as a convex combination of their inputs, where the 2 coefficients are updated using specific projection algorithm (see BID5 for details).Our Defensive Quantization is illustrated in Figure 2 .

The key part is the regularization term, which suppress the noise amplification effect by regularizing the Lipschitz constant.

As a result, the perturbation at each layer is kept within a certain range, the adversarial noise won't propagate.

Our method not only fixes the drop of robustness induced by quantization, but also takes quantization as a defense method to further increase the robustness.

Therefore it's named Defensive Quantization.

Our experiments demonstrate the following advantages of Defensive Quantization.

First, DQ can retain the robustness of a model when quantized with low-bit.

Second, DQ is a general and effective defense method under various scenarios, thus can be combined with other defensive techniques to further advance state-of-the-art robustness.

Third, as a by-product, DQ can also improve the accuracy of training quantized models on clean images without attacks, since it limits the dynamic range.

We conduct experiments with Wide ResNet BID32 of 28 × 10 on the CIFAR-10 dataset BID13 ) using ReLU6 based activation quantization, with number of bits ranging from 1 to 5.

All the models are trained following BID32 with momentum SGD for 200 epochs.

The adversarial samples are generated using FGSM attacker with = 8.

The results are presented in TAB3 .

For vanilla models, though the adversarial robustness increases with the number of bits, i.e., the models closer to full-precision one has better robustness, the best quantized model still has inferior robustness by −9.1%.

While with our Defensive Quantization, the quantized models have better robustness than full-precision counterparts.

The robustness is better when the number of bits are small, since it can de-noise larger adversarial perturbations.

We also find that the robustness is generally increasing as β gets larger, since the regularization of Lipschitz constant itself keeps the noise smaller at later layers.

At the same time, the quantized models consistently achieve better robustness.

The robustness of quantized model also increases with β.

We conduct a detailed analysis of the effect of β in Section B. The conclusion is: (1) conventional quantized models are less robust.

FORMULA2 Lipschitz regularization makes the model robust.

(3) Lipschitz regularization + quantization makes model even more robust.

As shown in BID2 , many of the defense methods actually lead to obfuscated gradient, providing a false sense of security.

Therefore it is important to check the model's robustness under black-box attack.

We separately trained a substitute VGG-16 model on the same dataset to generate adversarial samples, as it was proved to have the best transferability BID27 .

The results are presented in FIG5 .

Trends of white-box and black-box attack are consistent.

Vanilla quantization leads to inferior black-box robustness, while with our method can further improve the models' robustness.

As the robustness gain is consistent for both white-box and black-box setting, our method does not suffer from gradient masking.

In this section, we show that we can combine Defensive Quantization with other defense techniques to achieve state-of-the-art robustness.

Setup: We conducted experiments on the Street View House Number dataset (SVHN) BID17 and CIFAR-10 dataset BID13 .

Since adversarial training is time consuming, we only use the official training set for experiments.

CIFAR-10 is another widely used dataset containing 50,000 training samples and 10,000 testing samples of size 32 × 32.

For both datasets, we divide the pixel values by 255 as a pre-processing step.

Following BID2 BID5 BID16 , we used Wide ResNet BID32 ) models in our experiments as it is considered as the standard model on the dataset.

We used depth 28 and widening factor 10 for CIFAR-10, and depth 16 and widening factor 4 for SVHN.

We followed the training protocol in BID32 that uses a SGD optimizer with momentum=0.9.

For CIFAR-10, the model is trained for 200 epochs with initial learning rate 0.1, decayed by a factor of 0.2 at 60, 120 and 160 epochs.

For SVHN dataset, the model is trained for 160 epochs, with initial learning rate 0.01, decayed by 0.1 at 80 and 120 epochs.

For DQ, we used bit=1 and β = 2e-3 as it offers the best robustness (see Section B).We combine DQ with other defense methods to further boost the robustness.

For Feature Squeezing BID31 , we used 5 bit for image color reduction, followed by a 2 × 2 median filter.

As adversarial FGSM training leads to gradient masking issue BID29 ) (see A.2 for our experiment), we used the variant adversarial R+FGSM training.

To avoid over-fitting into certain , we randomly sample using ∼ N (0, δ), = clip [0,2δ] (abs( )).

Specifically we used δ = 8 to cover from 0-16.

During test time, the is set to a fixed value (2/8/16).

We also conducted adversarial PGD training.

Following BID14 BID26 , during training we sample random as in R+FGSM setting, and generate adversarial samples using step size 1 and number of iterations min( + 4, 1.25 ) .

The results are presented in TAB0 , where (B) indicates black-box attack with a seperately trained VGG-16 model.

The bold number indicates the best result in its column.

We observe that for all normal training, feature squeezing and adversarial training settings, our DQ method can further improve the model's robustness.

Among all the defenses, adversarial training provides the best performance against various attacks, epecially adversarial R+FGSM training.

While white box PGD attack is generally the strongest attack in our experiments.

Our DQ method also consistently improves the black-box robustness and there is no sign of gradient masking.

Thus DQ proves to be an effective defense for various white-box and black-box attacks.

As a by-product of our method, Defensive Quantization can even improve the accuracy of quantized models on clean images without attack, making it a beneficial drop-in substitute for normal quantization procedures.

Due to conventional quantization method's amplification effect, the distribution of activation can step over the truncation boundary (0-6 for ReLU6, 0-1 for ReLU1), which makes the optimization difficult.

DQ explicitly add a regularization to shrink the dynamic range of activation, so that it is fitted within the truncation range.

To demonstrate our hypothesis, we experimented with ResNet and CIFAR-10.

We quantized the activation with 4-bit (because NVIDIA recently introduced INT4 in Turing architecture) using ReLU6 and ReLU1 respectively.

Vanilla quantization and DQ training are conducted for comparison.

As shown in TAB6 , with vanilla quantization, ReLU1 model has around 1% worse accuracy than ReLU6 model, although they are mathematically equal if we

In this work, we aim to raise people's awareness about the security of the quantized neural networks, which is widely deployed in GPU/TPU/FPGAs, and pave a possible direction to bridge two important areas in deep learning: efficiency and robustness.

We connect these two domains by designing a novel Defensive Quantization (DQ) module to defend adversarial attacks while maintain the efficiency.

Experimental results on two datasets validate that the new quantization method can make the deep learning models be safely deployed on mobile devices.

BID16 .

To make a comparison, we also trained a full precision model and a ReLU6 quantized model following same setting.

All the quantized models use bit=2.

We tested the trained model using PGD attack by BID16 under both white-box and black-box setting.

We use a VGG-16 model separately trained with PGD adversarial training as the black-box attacker.

The results are provided in TAB8 .

Our white-box result is consistent with BID22 , where Tanh based quantizaion with PGD training gives much higher white-box accuracy compared to the full precision model.

However, we can see that the black-box robustness decreases.

Worse still, the black-box attack successful rate is even lower than white-box, which is abnormal since black-box attack is generally weaker than white-box.

This phenomenon indicates severe gradient masking problem BID19 BID2 , which gives a false sense of security.

As a comparison, we also trained a ReLU6 quantized model.

With ReLU6 quantization, there is no sign of gradient masking, nor improvement in robustness, indicating that gradient masking problem majorly comes from Tanh activation function.

Actually, ReLU6 quantized model has slightly worse robustness.

Therefore we conclude that simple quantization cannot help to improve robustness.

Instead, it leads to inferior robustness.

Here we demonstrate that FGSM adversarial training leads to significant gradient masking problem, while R+FGSM fixes such issues.

We trained a Wide ResNet using adversarial FGSM and adversarial R+FGSM respectively.

Then the model is tested using FGSM with = 8 under white-box and black-box setting (VGG-16 as substitute).

The results are shown in Table Table 6 .

White-box and black-box robustness of FGSM/R+FGSM adversarially trained Wide ResNet on CIFAR-10.

We can clearly see the same gradient masking effect.

For adversarial FGSM training, it gives much higher white box robustness than R+FGSM adversarial training, while the black-box robustness is much worse.

To avoid gradient masking, we thus use R+FGSM adversarial training instead.

B HYPER-PARAMETERS: β STUDY Figure 6 .

Clean and adversarial accuracy of 4-bit quantized Wide ResNet w.r.t.

different β.

As observed in Section 5.1, the adversarial robustness of the model generally increases as β gets larger.

Here we aim to find what is the optimal β for our experiment.

We took 4-bit quantized Wide ResNet for example.

As shown in Figure 6 , the adversarial accuracy first gets larger as β increases, and slowly goes down afterwards, reaching peak performance at β = 0.002.

The clean accuracy is more stable.

At the first stage, as regularization gets stronger, the effect of noise amplify is suppressed more.

While for the second stage, the training suffers from a too strong regularization.

Therefore, we used a β = 0.002 for our experiments unless specified.

We visualize some of the clean and adversarial samples from CIFAR-10 test set, and the corresponding prediction for full precision model (FP), vanilla 4-bit quantized model (VQ) and our 4-bit Defensive Quantization model (DQ), as 4-bit quantization is now now supported by Tensor Cores with NVIDIA's Turing Architecture (NVIDIA).

Predicted class and probability is provided.

Compared to FP and DQ, VQ has worse robustness by misclassifying more adversarial samples (sample 1, 2, 4, 6).

Our DQ model enjoys better robustness than FP model in two aspects: 1. our DQ model succeeds to defend some of the attacks when FP failed (sample 5, 7); 2.

our DQ model has a better confidence for true label compared to FP model when both succeeds to defend (sample 1, 4, 6).

Even when all models fail to defend, our DQ model has the lowest confidence for the misclassified class (sample 3, 8).GT: bird

Clean Adversarial Figure 7 .

Visualization of adversarial samples and the corresponding predictions (label and probability).

FP: full-precision model; VQ: vanilla 4-bit quantized model; DQ: our 4-bit Defensive Quantization model

@highlight

We designed a novel quantization methodology to jointly optimize the efficiency and robustness of deep learning models.

@highlight

Proposes a regularization scheme to protect quantized neural networks from adversarial attacks using a Lipschitz constant filitering of the inner layers' inpout-output.
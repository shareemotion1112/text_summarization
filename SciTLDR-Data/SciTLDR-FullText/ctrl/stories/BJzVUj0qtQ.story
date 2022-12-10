Deep neural networks are vulnerable to adversarial examples, which can mislead classifiers by adding imperceptible perturbations.

An intriguing property of adversarial examples is their good transferability, making black-box attacks feasible in real-world applications.

Due to the threat of adversarial attacks, many methods have been proposed to improve the robustness, and several state-of-the-art defenses are shown to be robust against transferable adversarial examples.

In this paper, we identify the attention shift phenomenon, which may hinder the transferability of adversarial examples to the defense models.

It indicates that the defenses rely on different discriminative regions to make predictions compared with normally trained models.

Therefore, we propose an attention-invariant attack method to generate more transferable adversarial examples.

Extensive experiments on the ImageNet dataset validate the effectiveness of the proposed method.

Our best attack fools eight state-of-the-art defenses at an 82% success rate on average based only on the transferability, demonstrating the insecurity of the defense techniques.

Recent progress in machine learning and deep neural networks has led to substantial improvements in various pattern recognition tasks such as image understanding BID21 BID9 , speech recognition BID7 , and machine translation .

However, deep neural networks are highly vulnerable to adversarial examples BID2 BID24 BID6 .

They are maliciously generated by adding small perturbations to legitimate examples, but make deep neural networks produce unreasonable predictions.

The existence of adversarial examples, even in the physical world BID11 BID5 BID1 , has raised concerns in security-sensitive applications, e.g., self-driving cars, healthcare and finance.

Attacking deep neural networks has drawn an increasing attention since the generated adversarial examples can serve as a surrogate to evaluate the robustness of different models BID3 and help to improve the robustness BID6 BID16 .

Several methods have been proposed to generate adversarial examples with the knowledge of the gradient information of a given model, such as fast gradient sign method BID6 , basic iterative method BID11 , and BID3 's method, which are known as white-box attacks.

Moreover, it is shown that adversarial examples have cross-model transferability BID15 , i.e., the adversarial examples crafted for one model can fool a different model with a high probability.

The transferability of adversarial examples enables practical black-box attacks to real-world applications and induces serious security issues.

The threat of adversarial examples has motivated extensive research on building robust models or techniques to defend against adversarial attacks.

These include training with adversarial examples BID6 BID12 BID27 BID16 , image denoising/transformation BID29 BID8 , leveraging generative models to move adversarial examples towards data manifold BID20 , and theoretically-certified defenses BID19 BID28 .

Although the non-certified defenses have demonstrated robustness against common attacks, they do so by causing obfuscated gradients, which can be easily circumvented by new attacks BID0 .

However, some of the defenses BID27 BID29 ; BID8 Figure 1: Demonstration of the attention shift phenomenon of the defense models compared with normally trained models.

We adopt class activation mapping (Zhou et al., 2016) to visualize the attentive regions of three normally trained models-Inception v3 BID25 , Inception ResNet v2 BID26 , ResNet 152 BID9 and four defense models BID27 BID29 BID8 .

These defense models focus their attention on slightly different regions compared with normally trained models, which may affect the transferability of adversarial examples.

BID8 claim to be resistant to transferable adversarial examples, making black-box attacks difficult to evade these defenses.

In this paper, we identify attention shift, that the defenses make predictions based on slightly different discriminative regions compared with normally trained models, as a phenomenon which may hinder the transferability of adversarial examples to the defense models.

For example, we show the attention maps of several normally trained models and defense models in Fig. 1 , to represent the discriminative regions for their predictions.

It is apparent that the normally trained models have similar attention maps while the defenses induce shifting attention maps.

The attention shift of the defenses is caused by either training under different data distributions BID27 or transforming the inputs before classification BID29 BID8 .

Therefore, the transferability of adversarial examples is largely reduced to the defenses since the structure information hidden in adversarial perturbations may be easily overlooked if a model focuses its attention on different regions.

To mitigate the effect of attention shift and evade the defenses by transferable adversarial examples, we propose an attention-invariant attack method.

In particular, we generate an adversarial example for an ensemble of examples composed of an legitimate one and its shifted versions.

Therefore the resultant adversarial example is less sensitive to the attentive region of the white-box model being attacked and may have a bigger chance to fool another black-box model with a defense mechanism based on attention shift.

We further show that this method can be simply implemented by convolving the gradient with a pre-defined kernel under a mild assumption.

The proposed method can be integrated into any gradient-based attack methods such as fast gradient sign method and basic iterative method.

Extensive experiments demonstrate that the proposed attention-invariant attack method helps to improve the success rates of black-box attacks against the defense models by a large margin.

Our best attack reaches an average success rate of 82% to evade eight state-of-the-art defenses based only on the transferability, thus demonstrating the insecurity of the current defenses.

Adversarial Examples.

Deep neural networks are shown to be vulnerable to adversarial examples first in the visual domain BID24 .

Then several methods are proposed to generate adversarial examples for the purpose of high success rates and minimal size of perturbations BID6 BID11 BID3 .

They also exist in the physical world BID11 BID5 BID1 .

Although adversarial examples are recently crafted for many domains, we focus on image classification tasks in this paper.

Black-box Attacks.

Black-box adversaries have no access to the architecture or parameters of the target model, which are under a more challenging threat model to perform attacks.

The transferability of adversarial examples provides an opportunity to attack a black-box model BID15 .Several methods BID30 have been proposed to improve the transferability, which enable powerful black-box attacks.

Besides the transfer-based black-box attacks, there is another line of works that perform attacks based on adaptive queries.

For example, BID18 use queries to distill the knowledge of the target model and train a surrogate model.

It therefore turns the black-box attacks to the white-box attacks.

Recent methods use queries to estimate the gradient or the decision boundary of the black-box model (??) to generate adversarial examples.

However, these methods usually require tremendous number of queries, which may be impractical in real-world applications.

In this paper, we resort to transferable adversarial examples for black-box attacks.

Defend against Adversarial Attacks.

A large variety of methods have been proposed to increase the robustness of deep learning models.

Besides directly making the models produce correct predictions for adversarial examples, some methods attempt to detect them instead BID17 ?) .

However most of the non-certified defenses demonstrate the robustness by causing obfuscated gradients, which are successfully circumvented by new developed attacks BID0 .

Although these defenses are not robust in the white-box setting, some of them BID27 BID29 BID8 empirically show the resistance against transferable adversarial examples in the black-box setting.

In this paper, we focus on generating more transferable adversarial examples against these defenses.

In this section, we provide the detailed description of our algorithm.

Let x real denote a real example and y denote the corresponding ground-truth label.

Given a classifier f (x) : X → Y that outputs a label as the prediction for an input, we want to generate an adversarial example x adv which is visually indistinguishable from x real but fools the classifier, i.e., f (x adv ) = y. 1 In most cases, the L p norm of the adversarial perturbation is required to be smaller than a threshold as ||x adv − x real || p ≤ .

In this paper, we use the L ∞ norm as the measurement.

For adversarial example generation, the objective is to maximize the loss function J(x adv , y) of the classifier, where J is often the cross-entropy loss.

So the constrained optimization problem can be written as arg max DISPLAYFORM0 To solve this optimization problem, the gradient of the loss function with respect to the input needs to be calculated, termed as white-box attacks.

However in some cases, we cannot get access to the gradient of the classifier, where we need to perform attacks in the black-box manner.

We resort to transferable adversarial examples which are generated for a different white-box classifier but have high transferability for black-box attacks.

Several methods have been proposed to solve the optimization problem in Eq. (1).

We give a brief introduction of them in this section.

et al., 2015) generates an adversarial example x adv by linearizing the loss function in the input space and performing one-step update as DISPLAYFORM0 DISPLAYFORM1 where ∇ x J is the gradient of the loss function with respect to x. sign(·) is the sign function to make the perturbation meet the L ∞ norm bound.

FGSM can generate more transferable adversarial examples but is usually not effective enough for attacking white-box models BID12 .Basic Iterative Method (BIM) BID11 extends FGSM by iteratively applying gradient updates multiple times with a small step size α, which can be expressed as DISPLAYFORM2 To restrict the generated adversarial examples within the -ball of x real , we can clip x adv t after each update or set α = /T with T being the number of iterations.

It has been shown that BIM induces much more powerful white-box attacks than FGSM at the cost of worse transferability BID12 .

proposes to improve the transferability of adversarial examples by integrating a momentum term into the iterative attack method.

The update procedure is

where g t gathers the gradient information up to the t-th iteration with a decay factor µ.Diverse Inputs Iterative Fast Gradient Sign Method BID30 applies random transformations to the inputs and feeds the transformed images into the classifier for gradient calculation.

The image transformation includes random resizing and padding with a given probability.

This method can be combined with the momentum-based method to further improve the transferability.

DISPLAYFORM0 where the loss function J could be different from the cross-entropy loss.

This method aims to find adversarial examples with minimal size of perturbations, to measure the robustness of different models.

It also lacks the efficacy for black-box attacks like BIM.

Although many attack methods BID30 can generate adversarial examples with very high transferability across normally trained models, they are less effective to attack defense models in the black-box manner.

Some of the defenses BID27 BID29 BID8 are shown to be quite robust against black-box attacks.

So we want to answer that: Are these defenses really free from transferable adversarial examples?We identify the attention shift phenomenon which may inhibit the transferability of adversarial examples to the defenses.

The attention shift refers to that the discriminative regions used by the defenses to identify object categories are slightly different from those used by normally trained models, as shown in Fig. 1 .

The adversarial examples generated for one model can be hardly transferred to another model with attention shift since that the structure information in adversarial perturbations may be easily destroyed if the model focuses its attention on different regions.

To reduce the effect of attention shift, we propose an attention-invariant attack method.

In particular, rather than optimizing the objective function at a single point as Eq. FORMULA0 , the proposed method uses a set of shifted images to optimize an adversarial example as arg max DISPLAYFORM0 where T ij (x) is a transformation operation that shifts image x by i and j pixels along the twodimensions respectively, i.e., each pixel (a, b) of the transformed image is T ij (x) a,b = x a−i,b−j , and w ij is the weight for the loss J(T ij (x adv ), y).

We set i, j ∈ {−k, ..., 0, ..., k} with k being the maximal number of pixels to shift.

With this method, the generated adversarial perturbations are less sensitive to the attentive regions of the white-box model, which may be transferred to another model with a higher success rate.

However, we need to calculate the gradients for (2k + 1) 2 images, which introduces much more computations.

Sampling a small number of shifted images for gradient calculation is a feasible way BID1 .

But we show that we can perform attacks by calculating the gradient for only one image under a mild assumption.

Convolutional neural networks are known to have the shift-invariant property BID13 , that an object in the input can be recognized in spite of its position.

Pooling layers contribute resilience to slight transformation of the input.

Therefore, we make an assumption that the shifted image T ij (x) is almost the same as x as inputs to the models, as well as their gradients DISPLAYFORM1 Based on this assumption, we calculate the gradient of the loss defined in Eq. (6) at a pointx as DISPLAYFORM2 Given Eq. FORMULA8 , we do not need to calculate the gradients for (2k + 1) 2 images.

Instead, we only need to get the gradient for the unchanged imagex and then average all the shifted gradients.

This procedure is equivalent to convolving the gradient with a kernel composed of all the weights w ij as i,j DISPLAYFORM3 where W is the kernel matrix of size (2k + 1) × (2k + 1) with W i,j = w −i−j .

In this paper, we generate the kernel W from a two-dimensional Gaussian function because: 1) the images with bigger shifts have relatively lower weights to make the adversarial perturbation fool the model at the unshifted image effectively; 2) by using a Gaussian function, this procedure is known as Gaussian blur, which is widely used in image processing.

Note that we only illustrate how to calculate the gradient of the loss function defined in Eq. (6), but do not specify the update algorithm for generating adversarial examples.

This indicates that our method can be integrated into any gradient-based attack methods including FGSM, BIM, MI-FGSM, etc.

Specifically, in each step we calculate the gradient ∇ x J(x adv t , y) at the current solution x adv t , then convolve the gradient with the pre-defined kernel W , and finally get the new solution x adv t+1 following the update rule in different attack methods (In FGSM, there is only one step of update).

In this section, we present the experimental results to demonstrate the effectiveness of the proposed method on improving the transferability of adversarial examples to the defense models.

We use an ImageNet-compatible dataset 2 comprised of 1000 images to conduct experiments.

This dataset was used in the NIPS 2017 adversarial competition.

We include eight defense models which are shown to be robust agsinst black-box attacks on the ImageNet dataset.

These are• Inc-v3 ens3 , Inc-v3 ens4 , IncRes-v2 ens BID27 ; • high-level representation guided denoiser (HGD, rank-1 submission in the NIPS 2017 defense competition) ); • input transformation through random resizing and padding (R&P, rank-2 submission in the NIPS 2017 defense competition) BID29 ); • input transformation through JPEG compression or total variance minimization (TVM) BID8 ); • rank-3 submission 3 in the NIPS 2017 defense competition (NIPS-r3).To attack these defenses based on the transferability, we also include four normally trained modelsInception v3 (Inc-v3) BID25 , Inception v4 (Inc-v4), Inception ResNet v2 (IncResv2) BID26 , and ResNet v2-152 (Res-v2-152) BID10 , as the white-box models to generate adversarial examples.

DISPLAYFORM0 The adversarial examples generated for Inc-v3 using FGSM and A-FGSM.

In our experiments, we integrate our method into the fast gradient sign method (FGSM) BID6 , momentum iterative fast gradient sign method (MI-FGSM) and diverse input iterative fast gradient sign method with momentum (DIM) BID30 .

We do not include the basic iterative method and Carlini & Wagner (2017)'s method since that they are not good at generating transferable adversarial examples .

We denote the attacks combined with our attention-invariant method as A-FGSM, A-MI-FGSM and A-DIM respectively.

For the settings of hyper-parameters, we set the maximum perturbation to be = 16 among all experiments with pixel value in [0, 255] .

For the iterative attack methods, we set the number of iteration as 10 and the step size as α = 1.6.

For MI-FGSM and A-MI-FGSM, we adopt the default dacay factor µ = 1.0.

For DIM and A-DIM, the transformation probability is set to 0.7.

Please note that the settings for each attack method and its attention-invariant version are the same, because our method is not concerned with the specific attack precedure.

We first perform adversarial attacks for Inc-v3, Inc-v4, IncRes-v2 and Res-v2-152 respectively using FGSM, MI-FGSM, DIM and their extensions by combining with the proposed attention-invariant attack method as A-FGSM, A-MI-FGSM and A-DIM.

We then use the generated adversarial examples to attack the eight defense models we consider based only on the transferability.

We report the success rates of black-box attacks in TAB1 , where the success rates are the misclassification rates of the corresponding defense models with adversarial images as inputs.

In the attention-invariant based attacks, we set the size of the kernel matrix W as 15 × 15 across all experiments, and we will study the effect of kernel size in Section 4.4.From the tables, we observe that the success rates against the defenses are improved by a large margin when using the proposed method regardless of the attack algorithms or the white-box models being attacked.

In general, the attention-invariant based attacks consistently outperform the baseline attacks by 5% ∼ 30%.

In particular, when using A-DIM, the combination of our method and DIM, to attack the IncRes-v2 model, the resultant adversarial examples have about 60% success rates against the defenses (as shown in TAB3 ).

It demonstrates the vulnerability of the current defenses against black-box attacks.

The results also validate the effectiveness of the proposed method.

Although we only compare the results of our attack method with baseline methods against the defense models, our attacks remain the success rates of baseline attacks in the white-box setting and the black-box setting against normally trained models, which will be shown in the Appendix.

We show several adversarial images generated for the Inc-v3 model by FGSM and A-FGSM in Fig. 2 .

It can be seen that by using A-FGSM, in which the gradients are convolved by a kernel W before applying to the raw images, the adversarial perturbations are much smoother than those generated by FGSM.

The smooth effect also exists in other attention-invariant based attacks.

In this section, we further present the results when adversarial examples are generated for an ensemble of models.

BID15 have shown that attacking multiple models at the same time can improve the transferability of the generated adversarial examples.

It is due to that if an example remains adversarial for multiple models, it is more likely to transfer to another black-box model.

We adopt the ensemble method proposed by , which fuses the logit activations of different models.

We attack the ensemble of with equal ensemble weights using FGSM, A-FGSM, MI-FGSM, A-FGSM, DIM and A-DIM respectively.

We also set the kernel size in the attention-invariant based attacks as 15 × 15.In TAB4 , we show the results of black-box attacks against the eight defenses.

The proposed method also improves the success rates across all experiments over the baseline attacks.

It should by noted that the adversarial examples generated by A-DIM can fool the state-of-the-art defenses at an 82% success rate on average based on the transferability.

And the adversarial examples are generated for normally trained models unaware of the defense strategies.

The results in the paper demonstrate that the current defenses are far from real security, and cannot be deployed in real-world applications.

The size of the kernel W plays a key role for improving the success rates of black-box attacks.

If the kernel size equals to 1 × 1, the attention-invariant based attacks degenerate to their vanilla versions.

Therefore, we conduct an ablation study to examine the effect of different kernel sizes.

We attack the Inc-v3 model by A-FGSM, A-MI-FGSM and A-DIM with the kernel length ranging from 1 to 21 with a granularity 2.

In Fig. 3 , we show the success rates against five defense models- IncRes-v2 ens , HGD, R&P, TVM and NIPS-r3.

The success rate continues increasing at first, and turns to remain stable after the kernel size exceeds 15 × 15.We also show the adversarial images generated for the Inc-v3 model by A-FGSM with different kernel sizes in Fig. 4 .

Due to the smooth effect given by the kernel, we can see that the adversarial perturbations are smoother when using a bigger kernel.

In this paper, we propose an attention-invariant attack method to mitigate the attention shift phenomenon and generate more transferable adversarial examples against the defense models.

Our method optimizes an adversarial image by using a set of shifted images.

Based on an assumption, our method is simply implemented by convolving the gradient with a pre-defined kernel, and can be integrated into any gradient-based attack methods.

We conduct experiments to validate the effectiveness of the proposed method.

Our best attack A-DIM, the combination of the proposed attentioninvariant method and diverse input iterative method BID30 , can fool eight state-of-the-art defenses at an 82% success rate on average, where the adversarial examples are generated against four normally trained models.

The results identify the vulnerability of the current defenses, which raises security issues for the development of more robust deep learning models.

We further show the results of the proposed attention-invariant attack method for white-box attacks and black-box attacks against normally trained models.

We adopt the same settings for attacks.

We also generate adversarial examples for Inc-v3, Inc-v4, IncRes-v2 and Res-v2-152 respectively using FGSM, A-FGSM, MI-FGSM, A-MI-FGSM, DIM and A-DIM.

For the attention-invariant based attacks, we set the kernel size as 7 × 7 since that the normally trained models have similar attentions.

We then use these adversarial examples to attack six normally trained models-Inc-v3, .

The results are shown in TAB5 , Table 6 and Table 7 .

The attention-invariant based attacks get better results in most cases than the baseline attacks.

<|TLDR|>

@highlight

We propose an attention-invariant attack method to generate more transferable adversarial examples for black-box attacks, which can fool state-of-the-art defenses with a high success rate.

@highlight

The paper proposes a new way of overcoming state of the art defences against adversarial attacks on CNN.

@highlight

This paper suggests that "attention shift" is a key property behind failure of adversarial attacks to transfer and propose an attention-invariant attack method
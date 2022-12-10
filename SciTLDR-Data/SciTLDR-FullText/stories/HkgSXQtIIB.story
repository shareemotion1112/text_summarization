Neural networks trained with backpropagation, the standard algorithm of deep learning which uses weight transport, are easily fooled by existing gradient-based adversarial attacks.

This class of attacks are based on certain small perturbations of the inputs to make networks misclassify them.

We show that less biologically implausible deep neural networks trained with feedback alignment, which do not use weight transport, can be harder to fool, providing actual robustness.

Tested on MNIST, deep neural networks trained without weight transport (1) have an adversarial accuracy of 98% compared to 0.03% for neural networks trained with backpropagation and (2) generate non-transferable adversarial examples.

However, this gap decreases on CIFAR-10 but is still significant particularly for small perturbation magnitude less than 1 ⁄ 2.

Deep neural networks trained with backpropagation (BP) are not robust against certain hardly perceptible perturbation, known as adversarial examples, which are found by slightly altering the network input and nudging it along the gradient of the network's loss function [1] .

The feedback-path synaptic weights of these networks use the transpose of the forward-path synaptic weights to run error propagation.

This problem is commonly named the weight transport problem.

Here we consider more biologically plausible neural networks introduced by Lillicrap et al. [2] to run error propagation using feedbackpath weights that are not the transpose of the forward-path ones i.e. without weight transport.

This mechanism was called feedback alignment (FA).

The introduction of a separate feedback path in [2] in the form of random fixed synaptic weights makes the feedback gradients a rough approximation of those computed by backpropagation.

Since gradient-based adversarial attacks are very sensitive to the quality of gradients to perturb the input and fool the neural network, we suspect that the gradients computed without weight transport cannot be accurate enough to design successful gradient-based attacks.

Here we compare the robustness of neural networks trained with either BP or FA on three well-known gradient-based attacks, namely the fast gradient sign method (FGSM) [3] , the basic iterative method (BIM) and the momentum iterative fast gradient sign method (MI-FGSM) [4] .

To the best of our knowledge, no prior adversarial attacks have been applied for deep neural networks without weight transport.

A typical neural network classifier, trained with the backpropagation algorithm, computes in the feedback path the error signals δ and the weight update ∆W according to the error-backpropagation equations:

where y l is the output signal of layer l, φ is the derivative of the activation function φ and η W is a learning-rate factor.

For neuroscientists, the weight update in equation 1 is a biologically implausible computation: the backward error δ requires W T , the transposed synapses of the forward path, as the feedback-path synapses.

However, the synapses in the forward and feedback paths are physically distinct in the brain and we do not know any biological mechanism to keep the feedback-path synapses equal to the transpose of the forward-path ones [5, 6] .

To solve this modeling difficulty, Lillicrap et al. [2] made the forward and feedback paths physically distinct by fixing the feedback-path synapses to different matrices B that are randomly fixed (not learned) during the training phase.

This solution, called feedback alignment, enables deep neural networks to compute the error signals δ without weight transport problem by the rule

In the rest of this paper, we add the superscript "bp" and "fa" in the notation of any term computed respectively with backpropagation and feedback alignment to avoid any confusion.

We call a "BP network" a neural network trained with backpropagation and "FA network" a neural network trained with feedback alignment.

Authors in [7] showed recently that the angles between the gradients ∆W f a and ∆W bp stay > 80°for most layers of ResNet-18 and ResNet-50 architectures.

This means that feedback alignment provides inaccurate gradients ∆W f a that are mostly not aligned with the true gradients ∆W bp .

Since gradient-based adversarial attacks rely on the true gradient to maximize the network loss function, can less accurate gradients computed by feedback alignment provide less-effective adversarial attacks ?

3 Gradient-based adversarial attacks

The objective of gradient-based attacks is to find gradient updates to the input with the smallest perturbation possible.

We compare the robustness of neural networks trained with either feedback alignment or backpropagation using three techniques mentioned in the recent literature.

Goodfellow et al. proposed an attack called Fast Gradient Sign Method to generate adversarial examples x [3] by perturbing the input x with one step gradient update along the direction of the sign of gradient, which can be summarized by

where is the magnitude of the perturbation, J is the loss function and y * is the label of x. This perturbation can be computed through transposed forward-path synaptic weights like in backpropagation or through random synaptic weights like in feedback alignment.

While the Fast Gradient Sign method computes a one step gradient update for each input x, Kurakin et al. extended it to the Basic Iterative Method [8] .

It runs the gradient update for multiple iterations using small step size and clips pixel values to avoid large changes on each pixel in the beginning of each iteration as follows

where α is the step size and Clip X (·) denotes the clipping function ensuring that each pixel x i,j is in the interval [x ij -, x ij + ].

This method is also called the Projected Gradient Descent Method because it "projects" the perturbation onto its feasible set using the clip function.

This method is a natural extension to the Fast Gradient Sign Method by introducing momentum to generate adversarial examples iteratively [4] .

At each iteration t, the gradient g t is computed by the rule

All the experiments were performed on neural networks with the LeNet architecture [9] with the cross-entropy loss function.

We vary the perturbation magnitude from 0 to 1 with a step size of 0.1.

All adversarial examples were generated using the number of iterations n = 10 for BIM and MI-FGSM attacks and µ = 0.8 for the MI-FGSM attack.

The results of the accuracy as function of the perturbation magnitude on MNIST are given in Figure 1a .

We find that when performing the three gradient-based adversarial attacks (FGSM, BIM and MI-FGSM) on a FA neural network, the accuracy does not decrease and stays around 97%.

This suggests that MNIST adversarial examples generated by FA gradients cannot fool FA neural networks for ∈ [0,1] unlike BP neural networks whose accuracy drastically decreases to 0% as the perturbation increases.

In the legend, we denote by "BP → F A" the generation of adversarial examples using BP to fool the FA network, and "F A → BP " the generation of adversarial examples using FA to fool the BP network Additionally, we investigate the transferability of the adversarial examples generated with either BP or FA networks using each one of the three studied attacks.

As shown in Figure 1b , we find that the generated adversarial examples by the FA network don't fool the BP network.

This means that these adversarial examples are not transferable.

The mutual conclusion is not true since adversarial examples generated by the BP network can fool the FA network.

Results on CIFAR-10 about the robustness of FA and BP networks to the three gradient-based adversarial attacks can be found in Figure  2a .

Unlike the results on MNIST, the accuracy of FA networks as function of the perturbation magnitude does decrease but still with a lower rate than the accuracy of BP networks.

For the transferability of adversarial examples, we still find that the generated adversarial examples by the BP network do fool the FA network.

However, unlike the non-transferability of adversarial examples from the FA network to the BP network on MNIST, the BP network is significantly fooled as long as the perturbation magnitude increases.

We perform an empirical evaluation investigating both the robustness of deep neural networks without weight transport and the transferability of adversarial examples generated with gradient-based attacks.

The results on MNIST clearly show that (1) FA networks are robust to adversarial examples generated with FA and (2) the adversarial examples generated by FA are not transferable to BP networks.

On the other hand, we find that these two conclusions are not true on CIFAR-10 even if FA networks showed a significant robustness to Figure 1b , we denote by "BP → F A" the generation of adversarial examples using BP to fool the FA network, and "F A → BP " the generation of adversarial examples using FA to fool the BP network gradient-based attacks.

Therefore, one should consider performing more exhaustive analysis on more complex datasets to understand the impact of the approximated gradients provided by feedback alignment on the adversarial accuracy of biologically plausible neural networks attacked with gradient-based methods.

@highlight

Less biologically implausible deep neural networks trained without weight transport can be harder to fool.
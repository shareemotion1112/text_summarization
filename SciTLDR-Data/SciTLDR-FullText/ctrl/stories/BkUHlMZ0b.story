The robustness of neural networks to adversarial examples has received great attention due to security implications.

Despite various attack approaches to crafting visually imperceptible adversarial examples, little has been developed towards a comprehensive measure of robustness.

In this paper, we provide theoretical justification for converting robustness analysis into a local Lipschitz constant estimation problem, and propose to use the Extreme Value Theory for efficient evaluation.

Our analysis yields a novel robustness metric called CLEVER, which is short for Cross Lipschitz Extreme Value for nEtwork Robustness.

The proposed CLEVER score is attack-agnostic and is computationally feasible for large neural networks.

Experimental results on various networks, including ResNet, Inception-v3 and MobileNet, show that (i) CLEVER is aligned with the robustness indication measured by the $\ell_2$ and $\ell_\infty$ norms of adversarial examples from powerful attacks, and (ii) defended networks using defensive distillation or bounded ReLU indeed give better CLEVER scores.

To the best of our knowledge, CLEVER is the first attack-independent robustness metric that can be applied to any neural network classifiers.

Recent studies have highlighted the lack of robustness in state-of-the-art neural network models, e.g., a visually imperceptible adversarial image can be easily crafted to mislead a well-trained network BID28 BID9 BID3 .

Even worse, researchers have identified that these adversarial examples are not only valid in the digital space but also plausible in the physical world BID17 BID8 .

The vulnerability to adversarial examples calls into question safety-critical applications and services deployed by neural networks, including autonomous driving systems and malware detection protocols, among others.

In the literature, studying adversarial examples of neural networks has twofold purposes: (i) security implications: devising effective attack algorithms for crafting adversarial examples, and (ii) robustness analysis: evaluating the intrinsic model robustness to adversarial perturbations to normal examples.

Although in principle the means of tackling these two problems are expected to be independent, that is, the evaluation of a neural network's intrinsic robustness should be agnostic to attack methods, and vice versa, existing approaches extensively use different attack results as a measure of robustness of a target neural network.

Specifically, given a set of normal examples, the attack success rate and distortion of the corresponding adversarial examples crafted from a particular attack algorithm are treated as robustness metrics.

Consequently, the network robustness is entangled with the attack algorithms used for evaluation and the analysis is limited by the attack capabilities.

More importantly, the dependency between robustness evaluation and attack approaches can cause biased analysis.

For example, adversarial training is a commonly used technique for improving the robustness of a neural network, accomplished by generating adversarial examples and retraining the network with corrected labels.

However, while such an adversarially trained network is made robust to attacks used to craft adversarial examples for training, it can still be vulnerable to unseen attacks.

Motivated by the evaluation criterion for assessing the quality of text and image generation that is completely independent of the underlying generative processes, such as the BLEU score for texts BID25 and the INCEPTION score for images BID27 , we aim to propose a comprehensive and attack-agnostic robustness metric for neural networks.

Stemming from a perturbation analysis of an arbitrary neural network classifier, we derive a universal lower bound on the minimal distortion required to craft an adversarial example from an original one, where the lower bound applies to any attack algorithm and any p norm for p ≥ 1.

We show that this lower bound associates with the maximum norm of the local gradients with respect to the original example, and therefore robustness evaluation becomes a local Lipschitz constant estimation problem.

To efficiently and reliably estimate the local Lipschitz constant, we propose to use extreme value theory BID6 for robustness evaluation.

In this context, the extreme value corresponds to the local Lipschitz constant of our interest, which can be inferred by a set of independently and identically sampled local gradients.

With the aid of extreme value theory, we propose a robustness metric called CLEVER, which is short for Cross Lipschitz Extreme Value for nEtwork Robustness.

We note that CLEVER is an attack-independent robustness metric that applies to any neural network classifier.

In contrast, the robustness metric proposed in BID11 , albeit attack-agnostic, only applies to a neural network classifier with one hidden layer.

We highlight the main contributions of this paper as follows:• We propose a novel robustness metric called CLEVER, which is short for Cross Lipschitz Extreme Value for nEtwork Robustness.

To the best of our knowledge, CLEVER is the first robustness metric that is attack-independent and can be applied to any arbitrary neural network classifier and scales to large networks for ImageNet.• The proposed CLEVER score is well supported by our theoretical analysis on formal robustness guarantees and the use of extreme value theory.

Our robustness analysis extends the results in BID11 from continuously differentiable functions to a special class of non-differentiable functions -neural+ networks with ReLU activations.• We corroborate the effectiveness of CLEVER by conducting experiments on state-of-theart models for ImageNet, including ResNet BID10 , Inception-v3 BID29 and MobileNet (Howard et al., 2017) .

We also use CLEVER to investigate defended networks against adversarial examples, including the use of defensive distillation BID23 and bounded ReLU BID34 .

Experimental results show that our CLEVER score well aligns with the attack-specific robustness indicated by the 2 and ∞ distortions of adversarial examples.

One of the most popular formulations found in literature for crafting adversarial examples to mislead a neural network is to formulate it as a minimization problem, where the variable δ ∈ R d to be optimized refers to the perturbation to the original example, and the objective function takes into account unsuccessful adversarial perturbations as well as a specific norm on δ for assuring similarity.

For instance, the success of adversarial examples can be evaluated by their cross-entropy loss BID28 BID9 or model prediction BID2 .

The norm constraint on δ can be implemented in a clipping manner BID18 or treated as a penalty function BID2 .

The p norm of δ, defined as δ p = ( DISPLAYFORM0 for any p ≥ 1, is often used for crafting adversarial examples.

In particular, when p = ∞, δ ∞ = max i∈{1,

...,d} |δ i | measures the maximal variation among all dimensions in δ.

When p = 2, δ 2 becomes the Euclidean norm of δ.

When p = 1, δ 1 = p i=1 |δ i | measures the total variation of δ.

The state-of-the-art attack methods for ∞ , 2 and 1 norms are the iterative fast gradient sign method (I-FGSM) BID9 BID18 , Carlini and Wagner's attack (CW attack) BID2 , and elastic-net attacks to deep neural networks (EAD) BID4 , respectively.

These attacks fall into the category of white-box attacks since the network model is assumed to be transparent to an attacker.

Adversarial examples can also be crafted from a black-box network model using an ensemble approach BID20 , training a substitute model , or employing zeroth-order optimization based attacks BID5 .

Since the discovery of vulnerability to adversarial examples BID28 , various defense methods have been proposed to improve the robustness of neural networks.

The rationale for defense is to make a neural network more resilient to adversarial perturbations, while ensuring the resulting defended model still attains similar test accuracy as the original undefended network.

Papernot et al. proposed defensive distillation BID23 , which uses the distillation technique BID12 and a modified softmax function at the final layer to retrain the network parameters with the prediction probabilities (i.e., soft labels) from the original network.

BID34 showed that by changing the ReLU function to a bounded ReLU function, a neural network can be made more resilient.

Another popular defense approach is adversarial training, which generates and augments adversarial examples with the original training data during the network training stage.

On MNIST, the adversarially trained model proposed by BID21 can successfully defend a majority of adversarial examples at the price of increased network capacity.

Model ensemble has also been discussed to increase the robustness to adversarial examples BID30 BID19 .

In addition, detection methods such as feature squeezing BID33 and example reforming BID22 can also be used to identify adversarial examples.

However, the CW attack is shown to be able to bypass 10 different detection methods BID1 .

In this paper, we focus on evaluating the intrinsic robustness of a neural network model to adversarial examples.

The effect of detection methods is beyond our scope.

BID28 compute global Lipschitz constant for each layer and use their product to explain the robustness issue in neural networks, but the global Lipschitz constant often gives a very loose bound.

BID11 gave a robustness lower bound using a local Lipschitz continuous condition and derived a closed-form bound for a multi-layer perceptron (MLP) with a single hidden layer and softplus activation.

Nevertheless, a closed-form bound is hard to derive for a neural network with more than one hidden layer.

BID31 utilized terminologies from topology to study robustness.

However, no robustness bounds or estimates were provided for neural networks.

On the other hand, works done by BID7 ; BID15 b) ; BID14 focus on formally verifying the viability of certain properties in neural networks for any possible input, and transform this formal verification problem into satisfiability modulo theory (SMT) and large-scale linear programming (LP) problems.

These SMT or LP based approaches have high computational complexity and are only plausible for very small networks.

Intuitively, we can use the distortion of adversarial examples found by a certain attack algorithm as a robustness metric.

For example, BID0 proposed a linear programming (LP) formulation to find adversarial examples and use the distortions as the robustness metric.

They observe that the LP formulation can find adversarial examples with smaller distortions than other gradient-based attacks like L-BFGS BID28 .

However, the distortion found by these algorithms is an upper bound of the true minimum distortion and depends on specific attack algorithms.

These methods differ from our proposed robustness measure CLEVER, because CLEVER is an estimation of the lower bound of the minimum distortion and is independent of attack algorithms.

Additionally, unlike LP-based approaches which are impractical for large networks, CLEVER is computationally feasible for large networks like Inception-v3.

The concept of minimum distortion and upper/lower bound will be formally defined in Section 3.

In this section, we provide formal robustness guarantees of a classifier in Theorem 3.2.

Our robustness guarantees are general since they only require a mild assumption on Lipschitz continuity of the classification function.

For differentiable classification functions, our results are consistent with the main theorem in BID11 but are obtained by a much simpler and more BID11 ) is in fact a special case of our analysis.

We start our analysis by defining the notion of adversarial examples, minimum p distortions, and lower/upper bounds.

All the notations are summarized in TAB0 .

Definition 3.1 (perturbed example and adversarial example).

Let DISPLAYFORM0 DISPLAYFORM1 An adversarial example is a perturbed example x a that changes c(x 0 ).

A successful untargeted attack is to find a x a such that c(x a ) = c(x 0 ) while a successful targeted attack is to find a x a such that c(x a ) = t given a target class t = c(x 0 ).

Definition 3.2 (minimum adversarial distortion ∆ p,min ).

Given an input vector x 0 of a classifier f , the minimum p adversarial distortion of x 0 , denoted as ∆ p,min , is defined as the smallest ∆ p over all adversarial examples of x 0 .

Definition 3.3 (lower bound of ∆ p,min ).

Suppose ∆ p,min is the minimum adversarial distortion of DISPLAYFORM2 , is defined such that any perturbed examples of x 0 with δ p ≤ β L are not adversarial examples.

Definition 3.4 (upper bound of ∆ p,min ).

Suppose ∆ p,min is the minimum adversarial distortion of x 0 .

An upper bound of ∆ p,min , denoted by β U where β U ≥ ∆ p,min , is defined such that there exists an adversarial example of x 0 with δ p ≥ β U .The lower and upper bounds are instance-specific because they depend on the input x 0 .

While β U can be easily given by finding an adversarial example of x 0 using any attack method, β L is not easy to find.

β L guarantees that the classifier is robust to any perturbations with δ p ≤ β L , certifying the robustness of the classifier.

Below we show how to derive a formal robustness guarantee of a classifier with Lipschitz continuity assumption.

Specifically, our analysis obtains a lower bound of DISPLAYFORM3 Lemma 3.1 (Lipschitz continuity and its relationship with gradient norm (Paulavičius &Žilinskas, 2006) ).

Let S ⊂ R d be a convex bounded closed set and let h(x) : S → R be a continuously differentiable function on an open set containing S. Then, h(x) is a Lipschitz function with Lipschitz constant L q if the following inequality holds for any x, y ∈ S: DISPLAYFORM4 where DISPLAYFORM5 ) is the gradient of h(x), and DISPLAYFORM6 Given Lemma 3.1, we then provide a formal guarantee to the lower bound β L .

Theorem 3.2 (Formal guarantee on lower bound β L for untargeted attack).

Let x 0 ∈ R d and f : R d → R K be a multi-class classifier with continuously differentiable components f i and let c = argmax 1≤i≤K f i (x 0 ) be the class which f predicts for x 0 .

For all δ ∈ R d with DISPLAYFORM7 argmax 1≤i≤K f i (x 0 + δ) = c holds with DISPLAYFORM8 is a lower bound of minimum distortion.

The intuitions behind Theorem 3.2 is shown in FIG0 with an one-dimensional example.

The function value g(x) = f c (x) − f j (x) near point x 0 is inside a double cone formed by two lines passing (x 0 , g(x 0 )) and with slopes equal to ±L q , where L q is the (local) Lipschitz constant of g(x) near x 0 .

In other words, the function value of g(x) around x 0 , i.e. g(x 0 + δ) can be bounded by g(x 0 ), δ and the Lipschitz constant L q .

When g(x 0 + δ) is decreased to 0, an adversarial example is found and the minimal change of δ is DISPLAYFORM9 Lq .

The complete proof is deferred to Appendix A. DISPLAYFORM10 is the Lipschitz constant of the function involving cross terms: f c (x) − f j (x), hence we also call it cross Lipschitz constant following BID11 .To distinguish our analysis from BID11 , we show in Corollary 3.2.1 that we can obtain the same result in BID11 by Theorem 3.2.

In fact, the analysis in BID11 ) is a special case of our analysis because the authors implicitly assume Lipschitz continuity on f i (x) when requiring f i (x) to be continuously differentiable.

They use local Lipschitz constant (L q,x0 ) instead of global Lipschitz constant (L q ) to obtain a tighter bound in the adversarial perturbation δ.

DISPLAYFORM11 .

By Theorem 3.2, we obtain the bound in BID11 ): DISPLAYFORM12 An important use case of Theorem 3.2 and Corollary 3.2.1 is the bound for targeted attack: Corollary 3.2.2 (Formal guarantee on β L for targeted attack).

Assume the same notation as in Theorem 3.2 and Corollary 3.2.1.

For a specified target class j, we have δ p ≤ min DISPLAYFORM13 In addition, we further extend Theorem 3.2 to a special case of non-differentiable functions -neural networks with ReLU activations.

In this case the Lipchitz constant used in Lemma 3.1 can be replaced by the maximum norm of directional derivative, and our analysis above will go through.

Lemma 3.3 (Formal guarantee on β L for ReLU networks).3 Let h(·) be a l-layer ReLU neural network with W i as the weights for layer i.

We ignore bias terms as they don't contribute to gradient.

DISPLAYFORM14 is the one-sided directional direvative, then Theorem 3.2, Corollary 3.2.1 and Corollary 3.2.2 still hold.

In this section, we provide an algorithm to compute the robustness metric CLEVER with the aid of extreme value theory, where CLEVER can be viewed as an efficient estimator of the lower bound β L and is the first attack-agnostic score that applies to any neural network classifiers.

Recall in Section 3 2 proof deferred to Appendix B 3 proof deferred to Appendix C we show that the lower bound of network robustness is associated with g(x 0 ) and its cross Lipschitz constant L j q,x0 , where g( DISPLAYFORM0 is readily available at the output of a classifier and L j q,x0 is defined as max x∈Bp(x0,R) ∇g(x) q .

Although ∇g(x) can be calculated easily via back propagation, computing L j q,x0 is more involved because it requires to obtain the maximum value of ∇g(x) q in a ball.

Exhaustive search on low dimensional x in B p (x 0 , R) seems already infeasible, not to mention the image classifiers with large feature dimensions of our interest.

For instance, the feature dimension d = 784, 3072, 150528 for MNIST, CIFAR and ImageNet respectively.

One approach to compute L j q,x0 is through sampling a set of points x (i) in a ball B p (x 0 , R) around x 0 and taking the maximum value of ∇g(x (i) ) q .

However, a significant amount of samples might be needed to obtain a good estimate of max ∇g(x) q and it is unknown how good the estimate is compared to the true maximum.

Fortunately, Extreme Value Theory ensures that the maximum value of random variables can only follow one of the three extreme value distributions, which is useful to estimate max ∇g(x) q with only a tractable number of samples.

It is worth noting that although BID32 also applied extreme value theory to estimate the Lipschitz constant.

However, there are two main differences between their work and this paper.

First of all, the sampling methodology is entirely different.

BID32 calculates the slopes between pairs of sample points whereas we directly take samples on the norm of gradient as in Lemma 3.1.

Secondly, the functions considered in BID32 are only one-dimensional as opposed to the high-dimensional classification functions considered in this paper.

For comparison, we show in our experiment that the approach in BID32 , denoted as SLOPE in Table 3 and FIG3 , perform poorly for high-dimensional classifiers such as deep neural networks.4.1 ESTIMATE L j q,x0 VIA EXTREME VALUE THEORY When sampling a point x uniformly in B p (x 0 , R), ∇g(x) q can be viewed as a random variable characterized by a cumulative distribution function (CDF).

For the purpose of illustration, we derived the CDF for a 2-layer neural network in Theorem D.1.

4 For any neural networks, suppose we have n samples { ∇g(x (i) ) q }, and denote them as a sequence of independent and identically distributed (iid) random variables Y 1 , Y 2 , · · · , Y n , each with CDF F Y (y).

The CDF of max{Y 1 , · · · , Y n }, denoted as F n Y (y), is called the limit distribution of F Y (y).

Fisher-TippettGnedenko theorem says that F n Y (y), if exists, can only be one of the three family of extreme value distributions -the Gumbel class, the Fréchet class and the reverse Weibull class.

Theorem 4.1 (Fisher-Tippett-Gnedenko Theorem).

If there exists a sequence of pairs of real numbers (a n , b n ) such that a n > 0 and lim n→∞ F n Y (a n y + b n ) = G(y), where G is a non-degenerate distribution function, then G belongs to either the Gumbel class (Type I), the Fréchet class (Type II) or the Reverse Weibull class (Type III) with their CDFs as follows: DISPLAYFORM1 Fréchet class (Type II): DISPLAYFORM2 if y ≥ a W , where a W ∈ R, b W > 0 and c W > 0 are the location, scale and shape parameters, respectively.

Theorem 4.1 implies that the maximum values of the samples follow one of the three families of distributions.

If g(x) has a bounded Lipschitz constant, ∇g(x (i) ) q is also bounded, thus its limit distribution must have a finite right end-point.

We are particularly interested in the reverse Weibull class, as its CDF has a finite right end-point (denoted as a W ).

The right end-point reveals the upper limit of the distribution, known as the extreme value.

The extreme value is exactly the unknown local cross Lipschitz constant L j q,x0 we would like to estimate in this paper.

To estimate L j q,x0 , we first generate N s samples of x (i) over a fixed ball B p (x 0 , R) uniformly and independently in each batch with a total of N b batches.

We then compute ∇g(x (i) ) q and store the maximum values of each batch in set S. Next, with samples in S, we perform a maximum likelihood estimation of reverse Weibull distribution parameters, and the location estimateâ W is used as an estimate of L j q,x0 .

Given an instance x 0 , its classifier f (x 0 ) and a target class j, a targeted CLEVER score of the classifier's robustness can be computed via g(x 0 ) and L j q,x0 .

Similarly, untargeted CLEVER scores can be computed.

With the proposed procedure of estimating L j q,x0 described in Section 4.1, we summarize the flow of computing CLEVER score for both targeted attacks and un-targeted attacks in Algorithm 1 and 2, respectively.

Algorithm 1: CLEVER-t, compute CLEVER score for targeted attack Input: a K-class classifier f (x), data example x 0 with predicted class c, target class j, batch size N b , number of samples per batch N s , perturbation norm p, maximum perturbation R Result: CLEVER Score µ ∈ R + for target class DISPLAYFORM0 Algorithm 2: CLEVER-u, compute CLEVER score for un-targeted attack Input:

Same as Algorithm 1, but without a target class j Result: CLEVER score ν ∈ R + for un-targeted attack DISPLAYFORM1

We conduct experiments on CIFAR-10 (CIFAR for short), MNIST, and ImageNet data sets.

For the former two smaller datasets CIFAR and MNIST, we evaluate CLEVER scores on four relatively small networks: a single hidden layer MLP with softplus activation (with the same number of hidden units as in BID11 ), a 7-layer AlexNet-like CNN (with the same structure as in BID2 ), and the 7-layer CNN with defensive distillation BID23 ) (DD) and bounded ReLU BID34 ) (BReLU) defense techniques employed.

For ImageNet data set, we use three popular deep network architectures: a 50-layer Residual Network BID10 ) (ResNet-50), Inception-v3 BID29 and MobileNet (Howard et al., 2017) .

They were chosen for the following reasons: (i) they all yield (close to) state-of-theart performance among equal-sized networks; and (ii) their architectures are significantly different with unique building blocks, i.e., residual block in ResNet, inception module in Inception net, and depthwise separable convolution in MobileNet.

Therefore, their diversity in network architectures is appropriate to test our robustness metric.

For MobileNet, we set the width multiplier to 1.0, achieving a 70.6% accuracy on ImageNet.

We used public pretrained weights for all ImageNet models 5 .In all our experiments, we set the sampling parameters N b = 500, N s = 1024 and R = 5.

For targeted attacks, we use 500 test-set images for CIFAR and MNIST and use 100 test-set images for ImageNet; for each image, we evaluate its targeted CLEVER score for three targets: a random target class, a least likely class (the class with lowest probability when predicting the original example), and the top-2 class (the class with largest probability except for the true class, which is usually the easiest target to attack).

We also conduct untargeted attacks on MNIST and CIFAR for 100 test-set images, and evaluate their untargeted CLEVER scores.

Our experiment code is publicly available 6 .

Figure 3 .

If the p-value is greater than 0.05, the null hypothesis cannot be rejected, meaning that the underlying data samples fit a reverse Weibull distribution well.

Figure 3 shows that all numbers are close to 100%, validating the use of reverse Weibull distribution as an underlying distribution of gradient norm samples empirically.

Therefore, the fitted location parameter of reverse Weibull distribution (i.e., the extreme value),â W , can be used as a good estimation of local cross Lipschitz constant to calculate the CLEVER score.

The exact numbers are shown in TAB5 in Appendix E. All numbers for each model are close to 100%, indicating S fits reverse Weibull distributions well.

We apply the state-of-the-art white-box attack methods, iterative fast gradient sign method (I-FGSM) BID9 BID18 and Carlini and Wagner's attack (CW) BID2 , to find adversarial examples for 11 networks, including 4 networks trained on CIFAR, 4 networks trained on MNIST, and 3 networks trained on ImageNet.

For CW attack, we run 1000 iterations for ImageNet and CIFAR, and 2000 iterations for MNIST, as MNIST has shown to be more difficult to attack BID4 .

Attack learning rate is individually tuned for each model: 0.001 for Inception-v3 and ResNet-50, 0.0005 for MobileNet and 0.01 for all other networks.

For I-FGSM, we run 50 iterations and choose the optimal ∈ {0.01, 0.025, 0.05, 0.1, 0.3, 0.5, 0.8, 1.0} to achieve the smallest ∞ distortion for each individual image.

For defensively distilled (DD) networks, 50 iterations of I-FGSM are not sufficient; we use 250 iterations for CIFAR-DD and 500 iterations for MNIST-DD to achieve a 100% success rate.

For the problem to be non-trivial, images that are classified incorrectly are skipped.

We report 100% attack success rates for all the networks, and thus the average distortion of adversarial examples can indicate the attack-specific robustness of each network.

For comparison, we compute the CLEVER scores for the same set of images and attack targets.

To the best of our knowledge, CLEVER is the first attack-independent robustness score that is capable of handling the large networks studied in this paper, so we directly compare it with the attack-induced distortion metrics in our study.

We evaluate the effectiveness of our CLEVER score by comparing the upper bound β U (found by attacks) and CLEVER score, where CLEVER serves as an estimated lower bound, β L .

Table 3 compares the average 2 and ∞ distortions of adversarial examples found by targeted CW and I-FGSM attacks and the corresponding average targeted CLEVER scores for 2 and ∞ norms, and FIG3 visualizes the results for ∞ norm.

Similarly, Table 2 compares untargeted CW and I-FGSM attacks with untargeted CLEVER scores.

As expected, CLEVER is smaller than the distortions of adversarial images in most cases.

More importantly, since CLEVER is independent of attack algorithms, the reported CLEVER scores can roughly indicate the distortion of the best possible attack in terms of a specific p distortion.

The average 2 distortion found by CW attack is close to the 2 CLEVER score, indicating CW is a strong 2 attack.

In addition, when a defense mechanism (Defensive Distillation or Bounded ReLU) is used, the corresponding CLEVER scores are consistently increased (except for CIFAR-BReLU), indicating that the network is indeed made more resilient to adversarial perturbations.

For CIFAR-BReLU, both CLEVER scores and p norm of adversarial examples found by CW attack decrease, implying that bound ReLU is an ineffective defense for CIFAR.

CLEVER scores can be seen as a security checkpoint for unseen attacks.

For example, if there is a substantial gap in distortion between the CLEVER score and the considered attack algorithms, it may suggest the existence of a more effective attack that can close the gap.

Since CLEVER score is derived from an estimation of the robustness lower bound, we further verify the viability of CLEVER per each example, i.e., whether it is usually smaller than the upper bound found by attacks.

TAB4 shows the percentage of inaccurate estimations where the CLEVER score is larger than the distortion of adversarial examples found by CW and I-FGSM attacks in three ImageNet networks.

We found that CLEVER score provides an accurate estimation for most of the examples.

For MobileNet and Resnet-50, our CLEVER score is a strict lower bound of these two attacks for more than 96% of tested examples.

For Inception-v3, the condition of strict lower bound Table 2 : Comparison between the average untargeted CLEVER score and distortion found by CW and I-FGSM untargeted attacks.

DD and BReLU represent Defensive Distillation and Bounded ReLU defending methods applied to the baseline CNN network.

Table 3 : Comparison of the average targeted CLEVER scores with average ∞ and 2 distortions found by CW, I-FSGM attacks, and the average scores calculated by using the algorithm in BID32 (denoted as SLOPE) to estimate Lipschitz constant.

DD and BReLU denote Defensive Distillation and Bounded ReLU defending methods applied to the CNN network.

We did not include SLOPE in ImageNet networks because it has been shown to be ineffective even for smaller networks.(a) avergage ∞ distortion of CW and I-FGSM targeted attacks, and CLEVER and SLOPE estimation.

Some very large SLOPE estimates (in parentheses) exceeding the maximum possible ∞ distortion are reported as 1.

Random Target Top-2 Target CW I-FGSM CLEVER SLOPE CW I-FGSM CLEVER BID32 .

SLOPE significantly exceeds the distortions found by attacks, thus it is an inappropriate estimation of lower bound β L .is worse (still more than 75%), but we found that in these cases the attack distortion only differs from our CLEVER score by a fairly small amount.

In Figure 5 we show the empirical CDF of the gap between CLEVER score and the 2 norm of adversarial distortion generated by CW attack for the same set of images in TAB4 .

In Figure 6 , we plot the 2 distortion and CLEVER scores for each DISPLAYFORM0 0% 0% 0% 2% 0% 0% 0% 0% 0% 0% 0% Resnet-50 4% 0% 0% 0% 2% 0% 0% 0% 1% 0% 0% 0% Inception-v3 25% 0% 0% 0% 23% 0% 0% 0% 15% 0% 0% 0% (a individual image.

A positive gap indicates that CLEVER (estimated lower bound) is indeed less than the upper bound found by CW attack.

Most images have a small positive gap, which signifies the near-optimality of CW attack in terms of 2 distortion, as CLEVER suffices for an estimated capacity of the best possible attack.

In Figure 7 , we vary the number of samples (N b = 50, 100, 250, 500) and compute the 2 CLEVER scores for three large ImageNet models, Inception-v3, ResNet-50 and MobileNet.

We observe that 50 or 100 samples are usually sufficient to obtain a reasonably accurate robustness estimation despite using a smaller number of samples.

On a single GTX 1080 Ti GPU, the cost of 1 sample (with N s = 1024) is measured as 2.9 s for MobileNet, 5.0 s for ResNet-50 and 8.9 s for Inception-v3, thus the computational cost of CLEVER is feasible for state-of-the-art large-scale deep neural networks.

Additional figures for MNIST and CIFAR datasets are given in Appendix E.

In this paper, we propose the CLEVER score, a novel and generic metric to evaluate the robustness of a target neural network classifier to adversarial examples.

Compared to the existing robustness evaluation approaches, our metric has the following advantages: (i) attack-agnostic; (ii) applicable to any neural network classifier; (iii) comes with strong theoretical guarantees; and (iv) is computationally feasible for large neural networks.

Our extensive experiments show that the CLEVER score well matches the practical robustness indication of a wide range of natural and defended networks.

A PROOF OF THEOREM 3.2Proof.

According to Lemma 3.1, the assumption that g( DISPLAYFORM0 Let x = x 0 + δ and y = x 0 in (4), we get DISPLAYFORM1 When g(x 0 + δ) = 0, an adversarial example is found.

As indicated by (5), DISPLAYFORM2 no adversarial examples can be found: DISPLAYFORM3 Finally, to achieve argmax 1≤i≤K f i (x 0 + δ) = c, we take the minimum of the bound on δ p in (A) over j = c. I.e. if DISPLAYFORM4 , the classifier decision can never be changed and the attack will never succeed.

B PROOF OF COROLLARY 3.2.1Proof.

By Lemma 3.1 and let g = f c − f j , we get L j q,x0 = max y∈Bp(x0,R) ∇g(y) q = max y∈Bp(x0,R) ∇f j (y) − ∇f c (y) q , which then gives the bound in Theorem 2.1 of BID11 .

DISPLAYFORM5 Proof.

For any x, y, let d = y−x y−x p be the unit vector pointing from x to y and r = y − x p .

Define uni-variate function u(z) = h(x + zd), then u(0) = h(x) and u(r) = h(y) and observe that D + h(x + zd; d) and D + h(x + zd; −d) are the right-hand and left-hand derivatives of u(z), we have DISPLAYFORM6 For ReLU network, there can be at most finite number of points in z ∈ (0, r) such that g (z) does not exist.

This can be shown because each discontinuous z is caused by some ReLU activation, and there are only finite combinations.

Let 0 = z 0 < z 1 < · · · < z k−1 < z k = 1 be those points.

Then, using the fundamental theorem of calculus on each interval separately, there existsz i ∈ (z i , z i−1 ) for each i such that Proof.

The j th output of a one-hidden-layer neural network can be written as DISPLAYFORM7 DISPLAYFORM8 where σ(z) = max(z, 0) is ReLU activation function, W and V are the weight matrices of the first and second layer respectively, and w r is the r th row of W .

Thus, we can compute g(x) and ∇g(x) q below: .

The red dash line encloses the ball B 2 (x 0 , R 1 ) and the blue dash line encloses a larger ball B 2 (x 0 , R 2 ).

If we draw samples uniformly within the balls, the probability of ∇g(x) 2 = y is proportional to the intersected volumes of the ball and the regions with ∇g(x) 2 = y. DISPLAYFORM9 As illustrated in FIG7 , the hyperplanes w r x + b r = 0, r ∈ {1, . . .

, U } divide the d dimensional spaces R d into different regions, with the interior of each region satisfying a different set of inequality constraints, e.g. w r+ x + b r+ > 0 and w r− x + b r− < 0.

Given x, we can identify which region it belongs to by checking the sign of w r x + b r for each r. Notice that the gradient norm is the same for all the points in the same region, i.e. for any x 1 , x 2 satisfying I(w r x 1 + b r ) = I(w r x 2 + b r ) ∀r, we have ∇g(x 1 ) q = ∇g(x 2 ) q .

Since there can be at most M = d i=0 U i different regions for a d-dimensional space with U hyperplanes, ∇g(x) q can take at most M different values.

Therefore, if we perform uniform sampling in a ball B p (x 0 , R) centered at x 0 with radius R and denote ∇g(x) q as a random variable Y , the probability distribution of Y is discrete and its CDF is piece-wise constant with at most M pieces.

Without loss of generality, assume there are M 0 ≤ M distinct values for Y and denote them as m (1) , m (2) , . . .

, m (M0) in an increasing order, the CDF of Y , denoted as F Y (y), is the following: E.1 PERCENTAGE OF EXAMPLES HAVING P VALUE > 0.05 TAB5 shows the percentage of examples where the null hypothesis cannot be rejected by K-S test, indicating that the maximum gradient norm samples fit reverse Weibull distribution well.

Figure 3 .

FIG10 shows the 2 CLEVER score with different number of samples (N b = 50, 100, 250, 500) for MNIST and CIFAR models.

For most models except MNIST-BReLU, reducing the number of samples only change CLEVER scores very slightly.

For MNIST-BReLU, increasing the number of samples improves the estimated lower bound, suggesting that a larger number of samples is preferred.

In practice, we can start with a relatively small N b = a, and also try 2a, 4a, · · · samples to see if CLEVER scores change significantly.

If CLEVER scores stay roughly the same despite increasing N b , we can conclude that using N b = a is sufficient.

DISPLAYFORM10

<|TLDR|>

@highlight

We propose the first attack-independent robustness metric, a.k.a CLEVER, that can be applied to any neural network classifier.
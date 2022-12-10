Deep neural networks provide state-of-the-art performance for many applications of interest.

Unfortunately they are known to be vulnerable to adversarial examples, formed by applying small but malicious perturbations to the original inputs.

Moreover, the perturbations can transfer across models: adversarial examples generated for a specific model will often mislead other unseen models.

Consequently  the adversary can leverage it to attack against the deployed black-box systems.

In this work, we demonstrate that the adversarial perturbation can be decomposed into two components: model-specific and data-dependent one, and it is the latter that mainly contributes to the transferability.

Motivated by this understanding, we propose to craft adversarial examples by utilizing the noise reduced gradient (NRG) which approximates the data-dependent component.

Experiments on various classification models trained on ImageNet demonstrates that the new approach enhances the transferability dramatically.

We also find that low-capacity models have more powerful attack capability than high-capacity counterparts, under the condition that they have comparable test performance.

These insights give rise to a principled manner to construct adversarial examples with high success rates and could potentially provide us guidance for designing effective defense approaches against black-box attacks.

With the resurgence of neural networks, more and more large neural network models are applied in real-world applications, such as speech recognition, computer vision, etc.

While these models have exhibited good performance, recent works BID15 ; BID5 show that an adversary is always able to fool the model into producing incorrect outputs by manipulating the inputs maliciously.

The corresponding manipulated samples are called adversarial examples.

However, how to understand this phenomenon BID5 ; BID17 ) and how to defend against adversarial examples effectively BID6 ; BID16 ; BID3 ) are still open questions.

Meanwhile it is found that adversarial examples can transfer across different models, i.e., the adversarial examples generated from one model can also fool another model with a high probability.

We refer to such property as transferability, which can be leveraged to attack black-box systems BID13 ; BID8 ).The phenomenon of adversarial vulnerability was first introduced and studied in BID15 .

The authors modeled the adversarial example generation as an optimization problem solved by box-constraint L-BFGS, and also attributed the presence of adversarial examples to the strong nonlinearity of deep neural networks.

BID5 argued instead that the primary cause of the adversarial instability is the linear nature and the high dimensionality, and the view yielded the fast gradient sign method (FGSM) .

Similarly based on an iterative linearization of the classifier, BID11 proposed the DeepFool method.

In BID6 ; BID16 , it was shown that the iterative gradient sign method provides stronger white-box attacks but does not work well for black-box attacks.

BID8 analyzed the transferability of adversarial examples in detail and proposed ensemble-based approaches for effective black-box attacks.

In BID3 it was demonstrated that high-confidence adversarial examples that are strongly misclassified by the original model have stronger transferability.

In addition to crafting adversarial examples for attacks, there exist lots of works on devising more effective defense.

BID12 proposed the defensive distillation.

BID5 introduced the adversarial training method, which was examined on ImageNet by BID6 and BID16 .

BID9 utilized image transformation, such as rotation, translation, and scaling, etc, to alleviate the harm of the adversarial perturbation.

Instead of making the classifier itself more robust, several works BID7 ; BID4 ) attempted to detect the adversarial examples, followed by certain manual processing.

Unfortunately, all of them can be easily broken by designing stronger and more robust adversarial examples BID3 ; BID0 ).In this work, we give an explanation for the transferability of adversarial examples and use the insight to enhance black-box attacks.

Our key observation is that adversarial perturbation can be decomposed into two components: model-specific and data-dependent one.

The model-specific component comes from the model architecture and random initialization, which is noisy and represents the behavior off the data manifold.

In contrast, the data-dependent component is smooth and approximates the ground truth on the data manifold.

We argue that it is the data-dependent part that mainly contributes to the transferability of adversarial perturbations across different models.

Based on this view, we propose to construct adversarial examples by employing the data-dependent component of gradient instead of the gradient itself.

Since this component is estimated via noise reduction strategy, we call it noise-reduced gradient (NRG) method.

Benchmark on the ImageNet validation set demonstrates that the proposed noise reduced gradient used in conjunction with other known methods could dramatically increase the success rate of black-box attacks.

to perform black-box attacks over ImageNet validation set.

We also explore the dependence of success rate of black-box attacks on model-specific factors, such as model capacity and accuracy.

We demonstrate that models with higher accuracy and lower capacity show stronger capability to attack unseen models.

Moreover this phenomenon can be explained by our understanding of transferability, and may provide us some guidances to attack unseen models.

We use f : R d → R K to denote the model function, which is obtained via minimizing the empirical risk over training set.

For simplicity we omit the dependence on the trainable model parameter θ, since it is fixed in this paper.

For many applications of interest, we always have d 1 and K = o(1).

According to the local linear analysis in BID5 , the high dimensionality makes f (x) inevitably vulnerable to the adversarial perturbation.

That is, for each x, there exists a small perturbation η that is nearly imperceptible to human eyes, such that DISPLAYFORM0 We call η adversarial perturbation and correspondingly x adv := x + η adversarial example.

In this work, we mainly study the adversarial examples in the context of deep neural networks, though they also exist in other models, for example, support vector machine (SVM) and decision tree, etc BID13 ).We call the attack (1) a non-targeted attack because the adversary has no control of the output other than requiring x to be misclassified by the model.

In contrast, a targeted attack aims at fooling the model into producing a wrong label specified by the adversary.

I.e.f (x + η) = y target .In contrast, we call the attack (1) a non-targeted attack.

In the black-box attack setting, the adversary has no knowledge of the target model (e.g. architecture and parameters) and is not allowed to query the model.

That is, the target model is a pure black-box.

However the adversary can construct adversarial examples on a local model (also called the source model) that is trained on the same or similar dataset with the target model.

Then it deploys those adversarial examples to fool the target model.

This is typically referred to as a black-box attack, as opposed to the white-box attack whose target is the source model itself.

In general, crafting adversarial perturbation can be modeled as following optimization problem, DISPLAYFORM0 where J is a loss function measuring the discrepancy between the prediction and ground truth; · is the metric to quantify the magnitude of the perturbation.

For image data, there is also an implicit constraint: x adv ∈ [0, 255] d , with d being the number of pixels.

In practice, the commonly choice of J is the cross entropy.

BID2 introduced a loss function that directly manipulates the output logit instead of probability.

This loss function is also adopted by many works.

As for the measurement of distortion, The best metric should be human eyes, which is unfortunately difficult to quantify.

In practice, ∞ and 2 norms are commonly used.

Ensemble-based approaches To improve the strength of adversarial examples, instead of using a single model, BID8 suggest using a large ensemble consisting of f 1 , f 2 , · · · , f Q as our source models.

Although there exist several ensemble strategies, similar as BID8 , we only consider the most commonly used method, averaging the predicted probability of each model.

The corresponding objective can be written as DISPLAYFORM1 where w i are the ensemble weights with w i = 1.Objectives FORMULA1 and FORMULA2 are for non-targeted attacks, and the targeted counterpart can be derived similarly.

There are various optimizers to solve problem (2) and (3).

In this paper, we mainly use the normalized-gradient based optimizer.

Fast Gradient Based Method This method BID5 ) attempts to solve (2) by performing only one step iteration, DISPLAYFORM0 where g(x) is a normalized gradient vector.

For ∞ -attack , DISPLAYFORM1 similarly for q -attack, DISPLAYFORM2 Both of them are called fast gradient based method (FGBM) and are empirically shown to be fast.

Also they have very good transferability BID6 ; BID16 ) though not optimal.

So it is worth considering this simple yet effective optimizer.

Iterative Gradient Method This method BID10 ; BID6 ) performs the projected normalized-gradient ascent DISPLAYFORM3 for k steps, where x 0 is the original clean image; clip x,ε (·) is the projection operator to enforce DISPLAYFORM4 and α is the step size.

In analogous to the fast gradient based method, the normalized gradient g(x) is chosen as g ∞ (x) for ∞ -attack, called iterative gradient sign method (IGSM), and g q (x) for qattack.

The fast gradient based method (4) can be viewed as a special case of (7) when α = ε, k = 1.

There are few articles trying to understand why adversarial examples can transfer between models, though it is extremely important for performing black-box attacks and building successful defenses.

To the best of our knowledge, the only two works are BID8 ; BID17 ), which suggested that the transferability comes from the similarity between the decision boundaries of the source and target models, especially in the direction of transferable adversarial examples.

BID17 also claimed that transferable adversarial examples span a contiguous subspace.

To investigate the transfer phenomenon, we ask a further question: what similarity between the different models A and B that enables transferability of adversarial examples across them?

Since the model A and B have a high performance on the same dataset, they must have learned a similar function on the data manifold.

However, the behaviour of the models off the data manifold can be different.

This is determined by the architectures of the models and random initializations, both of which are data-independent factors.

This clearly hints us to decompose the perturbation into two factors on and off the data manifold.

We referred to them as data-dependent and model-specific components.

We hypothesize that the component on the data manifold mainly contributes to the transferability from A to B, since this component captures the data-dependent information shared between models.

The model-specific one contributes little to the transferability due to its different behaviours off the data manifold for different models.

DISPLAYFORM0 , and the same color indicates the same predicted label.

More details can be found in Section 6.3.

A as the adversarial perturbation crafted from model A, we illustrate this explanation in FIG0 .

In the left panel, the decision boundaries of two models are similar in the inter-class area.

As can be observed, ∇f A can mislead both model A and B.

Then we decompose the perturbation into two parts, a data-dependent component ∇f A and a model-specific one ∇f A ⊥ , respectively.

Since ∇f A is almost aligned to the inter-class deviation (red line), i.e. on the data manifold, it can attack model B easily.

However, The model-specific ∇f A ⊥ contributes little to the transfer from A to B, though it can successfully fool model A with a very small distortion.

In the right panel, we plot the decision boundaries of resnet34 (model A) and densenet121 (model B) for ImageNet.

The horizontal axis represents the direction of data-dependent component ∇f This understanding suggests that to increase success rates of black-box adversarial attacks, we should enhance the data-dependent component.

We hence propose the NRG method which achieves that by reducing the model-specific noise, as elaborated in the following.

Noise-reduced gradient (NRG) The model-specific component ∇f ⊥ inherits from the random initialization, so it must be very noisy, i.e. high-frequency (the same observation is also systematically investigated in BID1 ); while ∇f is smooth, i.e. low-frequency, since it encodes the knowledge learned from training data.

Therefore, local average can be applied to remove the noisy model-specific information, yielding an approximation of the data-dependent component, DISPLAYFORM0 where the m is the number of samples chosen for averaging.

We call (8) noise-reduced gradient (NRG), which captures more data-dependent information than ordinary gradient ∇f .

To justify the effectiveness of this method, similar to BID14 , we visualize NRG for various m in FIG2 .

As shown, larger m leads to a smoother and more data-dependent gradient, especially for m = 100 the NRG can capture the semantic information of the obelisk.

The noisy model-specific information of ∇f could mislead the optimizer into the solutions that are overfitting to the specific source model, thus generalizing poorly to the other model.

Therefore we propose to perform attacks by using ∇f in Eq.(8), instead of ∇f , which can drive the optimizer towards the solutions that are more data-dependent.

Noise-reduced Iterative Sign Gradient Method (nr-IGSM) The iterative gradient sign method mounted by NRG can be written as, DISPLAYFORM1 The special case k = 1, α = ε is called noise-reduced fast gradient sign method (nr-FGSM) accordingly.

For q -attack, the noise-reduced version is similar, replacing the second equation of (9) by DISPLAYFORM2 (10) For any general optimizer, the corresponding noise-reduced counterpart can be derived similarly.

To justify and analyze the effectiveness of NRG for enhancing the transferability, we use the start-ofthe-art classification models trained on ImageNet dataset.

We elaborate the details in the following.

Dataset We use the ImageNet ILSVRC2012 validation set that contains 50, 000 samples.

For each attack experiment, we randomly select 5, 000 images that can be correctly recognized by all the models, since it is meaningless to construct adversarial perturbations for the images that target models cannot classify correctly.

And for the targeted attack experiments, each image is specified by a random wrong label.

Models We use the pre-trained models provided by PyTorch including resnet18, resnet34, resnet50, resnet101, resnet152, vgg11 bn, vgg13 bn, vgg16 bn, vgg19 bn, densenet121, densenet161, densenet169, densenet201, alexnet, squeezenet1 1.

The Top-1 and Top-5 accuracies of them can be found on website 1 .

To increas the reliability of experiments, all the models have been used, however for a specific experiment we only choose several of them to save computational time.

Criteria Given a set of adversarial examples, {(x DISPLAYFORM0 If F is the model used to generate adversarial examples, then the rate indicates the the white-box attack performance.

For targeted attacks, each image x adv is associated with a pre-specified label y target = y true .

Then we evaluate the performance of the targeted attack by the following Top-1 success rate, DISPLAYFORM1 The corresponding Top-5 rates can be computed in a similar way.

Attacks Throughout this paper the cross entropy 2 is chosen as our loss function J. We measure the distortion by two distances: ∞ norm and scaled 2 norm, i.e. root mean square deviation (RMSD) DISPLAYFORM2 , where d is the dimensionality of inputs.

As for optimizers, both FGSM and IGSM are considered.

In this section we demonstrate the effectiveness of our noise-reduced gradient technique by combining it with several commonly-used methods.

Fast gradient based method We first examine the combination of noise reduced gradient and fast gradient based methods.

The success rates of FGSM and nr-FGSM are summarized in TAB0 (results of FGM 2 -attacks can be found in Appendix C).

We observe that, for any pair of blackbox attack, nr-FGSM performs better than original FGSM consistently and dramatically.

Even the white-box attacks (the diagonal cells) also have improvements.

This result implies that the direction of noise-reduced gradient is indeed more effective than the vanilla gradient for enhancing the transferability.

Iterative gradient sign method One may argue that the above comparison is unfair, since nr-FGSM consumes more computational cost than FGSM, determined by m: number of perturbed inputs used for local average.

Here we examine IGSM and nr-IGSM under the same number of gradient calculations.

Table 2 presents the results.

Except for the attacks from alexnet, as we expect, adversarial examples generated by nr-IGSM indeed transfer much more easily than those generated by IGSM.

This indicates that the noise-reduced gradient (NRG) does guide the optimizer to explore the more data-dependent solutions.

Some observations By comparing TAB0 , we find that large models are more robust to adversarial transfer than small models, for example the resnet152.

This phenomenon has also been extensively investigated by BID10 .

It also shows that the transfer among the Table 2 .

This implies that the model-specific component also contributes to the transfer across models with similar architectures.

Additionally in most cases, IGSM generates stronger adversarial examples than FGSM except the attacks against alexnet.

This contradicts the claims in BID6 and BID16 that adversarial examples generated by FGSM transfer more easily than the ones of IGSM.

However our observation is consistent with the conclusions of BID2 : the higher confidence (smaller cross entropy) adversarial examples have in the source model, the more likely they will transfer to the target model.

We conjecture that this is due to the inappropriate choice of hyperparameters, for instance α = 1, k = min(ε + 4, 1.24ε) in BID6 are too small, and the solution has not fit the source model enough (i.e. underfitting).

When treated as a target model to be attacked, the alexnet is significantly different from those source models in terms of both architecture and test accuracy.

And therefore, the multiple iterations cause the IGSM to overfit more than FGSM, producing a lower fooling rate.

These phenomena clearly indicate that we should not trust the objective in Eq. (2) completely, which might cause the solution to overfit to the source model-specific information and leads to poor transferability.

Our noise reduced gradient technique regularizes the optimizer by removing the modelspecific information from the original gradients, and consequently, it can converge to a more datadependent solution that has much better cross-model generalization capability.

In this part, we apply NRG method into ensemble-based approaches described in Eq. (3).

Due to the high computational cost of model ensembles, we select 1, 000 images, instead of 5, 000, as our evaluation set.

For non-targeted attacks, both FGSM, IGSM and their noise reduced versions are tested.

The Top-1 success rates of IGSM attacks are nearly saturated, so we report the corresponding Top-5 rates in Table 3 (a) to demonstrate the improvements of our methods more clearly.

The results of FGSM and nr-FGSM attacks can be found in Appendix C.For targeted attacks, to generate an adversarial example predicted by unseen target models as a specific label, is much harder than generating non-targeted examples.

BID8 demonstrated that single-model based approaches are very ineffective in generating targeted adversarial examples.

That is why we did not consider targeted attacks in Section 5.1.Different from the non-targeted attacks, we find that targeted adversarial examples are sensitive to the step size α used in the optimization procedures (6) and (9).

After trying lots of α, we find that a large step size is necessary for generating strong targeted adversarial examples.

Readers can refer to Appendix A for more detailed analysis on this issue, though we cannot fully understand it.

Therefore we use a much larger step size compared to the non-target attacks.

The Top-5 success rates are reported in Table 3(b) .By comparing success rates of normal methods and our proposed NRG methods in Table 3 for both targeted and non-targeted attacks, we observed that NRG methods outperform the corresponding normal methods by a remarkable large margin in this scenario.

Table 3 : Top-5 success rates of ensemble-based approaches.

The cell (S, T ) indicates the attack performances from the ensemble S (row) against the target model T (column).

For each cell: the left is the rate of normal method, in contrast the right is the one of the noise-reduced counterpart.

The corresponding Top-1 success rates can be found in Appendix C (Table 7 and Table 8 ).

In this part, we explore the sensitivity of hyper parameters m, σ when applying our NRG methods for black-box attacks.

We take nr-FGSM approach as a testbed over the selected evaluation set described in Section 4.

Four attacks are considered here, and the results are shown in FIG6 .

It is not surprising that larger m leads to higher fooling rate for any distortion level ε due to the better estimation of the data-dependent direction of the gradient.

We find there is an optimal value of σ inducing the best performance.

Overly large σ will introduce a large bias in (8).

Extremely small σ is unable to remove the noisy model-specific information effectively, since noisy components of gradients of different perturbed inputs are still strongly correlated for small σ.

Moreover the optimal σ varies for different source models, and in this experiment it is about 15 for resnet18, compared to 20 for densenet161.

In this part, we preliminarily explore the robustness of adversarial perturbations to image transformations, such as rotation, scaling, blurring, etc.

This property is particularly important in practice, since it directly affects whether adversarial examples can survive in the physical world BID6 ; BID0 ; BID9 ).

To quantify the influence of transformations, we use the notion of destruction rate defined in BID6 , DISPLAYFORM0 where N is the number of images used to estimate the destruction rate, T (·) is an arbitrary image transformation.

The function c(x) indicates whether x is classified correctly: DISPLAYFORM1 And thus, the above rate R describes the fraction of adversarial images that are no longer misclassified after the transformation T (·).Densenet121 and resnet34 are chosen as our source and target model, respectively; and four image transformations are considered: rotation, Gaussian noise, Gaussian blur and JPEG compression.

FIG7 displays the results, which show that adversarial examples generated by our proposed NRG methods are much more robust than those generated by vanilla methods.

FIG15 of Appendix C ).

In this section we study the decision boundaries of different models to help us understand why NRG-based methods perform better.

Resnet34 is chosen as the source model and nine target models are considered, including resnet18, resnet50, resnet101, resnet152, vgg11 bn, vgg16 bn, vgg19 bn, densenet121, alexnet.

The ∇f is estimated by (8) with m = 1000, σ = 15.

Each point (u, v) in this 2-D plane corresponds to the image perturbed by u and v along sign ∇f and sign (∇f ⊥ ), respectively, i.e. clip x + u sign ∇f + v sign (∇f ⊥ ) , 0, 255where x represents the original image.

We randomly select one image that can be recognized by all the models examined.

The FIG8 shows the decision boundaries.

We also tried a variety of other source models and images, all the plots are similar.

From the aspect of changing the predicted label, the direction of sign ∇f is as sensitive as the direction of sign (∇f ⊥ ) for the source model resnet34.

However, except alexnet all the other target models are much more sensitive along sign ∇f than sign (∇f ⊥ ).

This is consistent with the argument in Section 3.1.

Removing ∇f ⊥ from gradients can also be thought as penalizing the optimizer along the model-specific direction, thus avoiding converging to a source model-overfitting solution that transfers poorly to the other target models.

Moreover, we find that, along the sign ∇f , the minimal distance u to produce adversarial transfer varies for different models.

The distances for complex models are significantly larger than those of small models, for instance, the comparison between resnet152 and resnent50.

This provides us a geometric understanding of why big models are more robust than small models, as observed in Section 5.1.

In earlier experiments TAB0 ), we can observe that adversarial examples crafted from alexnet generalize worst across models, for example nr-FGSM attack of alexnet → resnet152 only achieves 19.3 percent.

However, attacks from densenet121 consistently perform well for any target model, for example 84.3% of nr-IGSM adversarial examples can transfer to vgg19 bn, whose architecture is completely different from densenet121.

This observation indicates that different models can exhibit different performances in attacking the same target model.

Now we attempt to find the principle behind this important phenomenon, which can guide us to choose a better local model to generate adversarial examples for attacking the remote black-box system.

We select vgg19 bn and resnet152 as our target model, and use a variety of models to perform both FGSM and IGSM attacks against them.

The results are summarized in FIG11 .

The horizontal axis is the Top-1 test error, while the vertical axis is the number of model parameters that roughly quantifies the model capacity.

We can see that the models with powerful attack capability concentrate in the bottom left corner, while fooling rates are very low for those models with large test error and number of parameters 3 .

We can obtain an important observation that the smaller test error and lower capacity a model has, the stronger its attack capability is.

Here we attempt to explain this phenomenon from our understanding of transferability in Section 3.1.

A smaller test error indicates a lower bias for approximating the ground truth along the data manifold.

On the other hand, a less complex model might lead to a smaller model-specific component ∇f ⊥ , facilitating the data-dependent factor dominate.

In a nutshell, the model with small ∇f ⊥ and large ∇f can provide strong adversarial examples that transfer more easily.

This is consistent with our arguments presented previously.

Number of paramters (million) Top-1 Error (%) Here, the models of vgg-style have been removed, since the contribution from architecture similarities is not in our consideration.

The distortion is chosen as ε = 15.

The plots of attacking resnet152 are similar and can be found in Appendix C (Figure 9 ).

In this paper, we have verified that an adversarial perturbation can be decomposed into two components: model-specific and data-dependent ones.

And it is the latter that mainly contributes to the transferability of adversarial examples.

Based on this understanding, we proposed the noise-reduced gradient (NRG) based methods to craft adversarial examples, which are much more effective than previous methods.

We also show that the models with lower capacity and higher test accuracy are endowed with stronger capability for black-box attacks.

In the future, we will consider combining NRG-based methods with adversarial training to defend against black-box attacks.

The component contributing to the transferability is data-dependent, which is intrinsically low-dimensional, so we hypothesize that black-box attacks can be defensible.

On the contrary, the white-box attack origins from the extremely high-dimensional ambient space, thus its defense is much more difficult.

Another interesting thread of future research is to learn stable features beneficial for transfer learning by incorporating our NRG strategy, since the reduction of model-specific noise can lead to more accurate information on the data manifold.

When using IGSM to perform targeted black-box attacks, there are two hyper parameters including number of iteration k and step size α.

Here we explore their influence to the quality of adversarial examples generated.

The success rates are valuated are calculated on 1, 000 images randomly selected according to description of Section 4.

resnet152 and vgg16 bn are chosen as target models.

The performance are evaluated by the average Top-5 success rate over the three ensembles used in Table 3 (b).

FIG13 shows that for the optimal step size α is very large, for instance in this experiment it is about 15 compared to the allowed distortion ε = 20.

Both too large and too small step size will yield harm to the attack performances.

It interesting noting that with small step size α = 5, the large number of iteration provides worser performance than small number of iteration.

One possible explain is that more iterations lead optimizers to converge to more overfit solution.

In contrast, a large step size can prevent it and encourage the optimizer to explore more model-independent area, thus more iteration is better.

10/255 15/255 20/255Step size: α Step size: α

To further confirm the influence of model redundancy on the attack capability, we conduct an additional experiment on MNIST dataset.

We use the fully networks of D layers with width of each layer being 500, e.g. the architecture of model with D = 2 is 784 − 500 − 500 − 10.

Models of depth 1, 3, 5, 9 are considered in this experiment.

The Top-1 success rates of cross-model attacks are reported in TAB5 .The results of TAB5 demonstrate the low-capacity model has much stronger attack capability than large-capacity.

This is consistent with our observation in Section 6.4.

Top-1 Error (%) Figure 9: Top-1 success rates of FGSM and IGSM (k = 20, α = 5) attacks against resnet152 for various models.

The annotated value is the percentage of adversarial examples that can transfer to the resnet152.

Here, the models of resnet-style have been removed, since the contribution from architecture similarities is not in our consideration.

The distortion is chosen as ε = 15.

<|TLDR|>

@highlight

We propose a new method for enhancing the transferability of adversarial examples by using the noise-reduced gradient.

@highlight

This paper postulates that an adversarial perturbation consists of a model-specific and data-specific component, and that amplification of the latter is best suited for adversarial attacks.

@highlight

This paper focuses on enhancing the transferability of adversarial examples from one model to another model.
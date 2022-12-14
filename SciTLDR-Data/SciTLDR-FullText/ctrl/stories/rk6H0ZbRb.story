It is becoming increasingly clear that many machine learning classifiers are vulnerable to adversarial examples.

In attempting to explain the origin of adversarial examples, previous studies have typically focused on the fact that neural networks operate on high dimensional data, they overfit, or they are too linear.

Here we show that distributions of logit differences have a universal functional form.

This functional form is independent of architecture, dataset, and training protocol; nor does it change during training.

This leads to adversarial error having a universal scaling, as a power-law, with respect to the size of the adversarial perturbation.

We show that this universality holds for a broad range of datasets (MNIST, CIFAR10, ImageNet, and random data), models (including state-of-the-art deep networks, linear models, adversarially trained networks, and networks trained on randomly shuffled labels), and attacks (FGSM, step l.l., PGD).

Motivated by these results, we study the effects of reducing prediction entropy on adversarial robustness.

Finally, we study the effect of network architectures on adversarial sensitivity.

To do this, we use neural architecture search with reinforcement learning to find adversarially robust architectures on CIFAR10.

Our resulting architecture is more robust to white \emph{and} black box attacks compared to previous attempts.

An intriguing aspect of deep learning models in computer vision is that while they can classify images with high accuracy, they fail catastrophically when those same images are perturbed slightly in an adversarial fashion BID17 BID1 .

The prevalence of adversarial examples presents challenges to our understanding of how deep networks generalize and pose security risks in real world applications BID11 BID5 .

Several techniques have been proposed to defend against adversarial examples.

Adversarial training BID1 augments the training data with adversarial examples.

It has been shown that using stronger adversarial attacks in adversarial training can increase the robustness to stronger attacks, but at the cost of a decrease in clean accuracy (i.e. accuracy on samples that have not been adversarially perturbed) BID8 .

Defensive distillation BID12 , feature squeezing BID22 , and Parseval training BID0 have also been shown to make models more robust against adversarial attacks.

The goal of this work is to study the common properties of adversarial examples.

We calculate the adversarial error, defined as the difference between clean accuracy and adversarial accuracy at a given size of adversarial perturbation ( ).

Surprisingly, adversarial error has a similar dependence on small values of for all network models and datasets we studied, including linear, fully-connected, simple convolutional networks, Inception v3 BID19 , Inception-ResNet v2, Inception v4 BID20 , ResNet v1, ResNet v2 BID2 , NasNet-A BID24 BID25 , adversarially trained Inception v3 BID6 and Inception-ResNet v2 BID21 , and networks trained on randomly shuffled labels of MNIST.

Adversarial error due to the Fast Gradient Sign Method (FGSM), its L2-norm variant, and Projected Gradient Descent (PGD) attack grows as a power-law like A B with B between 0.9 and 1.3.

By contrast, we find that adversarial error caused by one-step least likely class method (step l.l.) also scales as a power-law where B is between 1.8 and 2.5 for small .

This observed universality points to a mysterious commonality between these models and datasets, despite the different number of channels, pixels, and classes present.

Adversarial error caused by FGSM on the training set of randomly shuffled labels of MNIST (LeCun & Cortes) also has the power-law form where B = 1.2, which implies that the universality is not a result of the specific content of these datasets nor the ability of the model to generalize.

To discover the mechanism behind this universality we show how, at small , the success of an adversarial attack depends on the input-logit Jacobian of the model and on the logits of the network.

We demonstrate that the susceptibility of a model to FGSM and PGD attacks is in large part dictated by the cumulative distribution of the difference between the most likely logit and the second most likely logit.

We observe that this cumulative distribution has a universal form among all datasets and models studied, including randomly produced data.

Together, we believe these results provide a compelling story regarding the susceptibility of machine learning models to adversarial examples at small .We show that training with single-step adversarial examples offers protection against large attacks (between 0.2 and 32), but does not help appreciably at defending against small attacks (below 0.2).

At = 0.2, all ImageNet models we studied incur 10 to 25% adversarial error, and surprisingly, vanilla NASNet-A (best clean accuracy in our study) has a lower adversarial error than adversarially trained Inception-ResNet v2 or Inception v3 BID6 (Fig. 1(a) ).

In light of these results, we explore a different avenue to adversarial robustness through architecture selection.

We perform neural architecture search (NAS) using reinforcement learning BID24 BID25 .

These techniques allow us to find several architectures that are especially robust to adversarial perturbations.

In addition, by analyzing the adversarial robustness of the tens-of-thousands of architectures constructed by NAS, we gain insights into the relationship between size of a model, its clean accuracy, and its adversarial robustness.

In summary, the key contributions of our work are:??? We study the functional form of adversarial error and logit differences across several models and datasets, which turn out to be universal.

We analytically derive the commonality in the power-law tails of the logit differences, and show how it leads to the commonality in the form of adversarial error.??? We observe that although the qualitative form of logit differences and adversarial error is universal, it can be quantitatively improved with entropy regularization and better network architectures.??? We study the dependence of adversarial robustness on the network architecture via NAS.We show that while adversarial accuracy is strongly correlated with clean accuracy, it is only weakly correlated with model size.

Our work leads to architectures that are more robust to white-box and black-box attacks on CIFAR10 BID4 ) than previous studies.

FGSM computes adversarial examples as: DISPLAYFORM0 where x is the clean image, y is the correct label for that image, x adv is the adversarial image, is the size of the adversarial perturbation, and L(x, y) is the loss function.

values are specified in range [0, 255] .

We only study white-box attacks in this section.

We begin with a preliminary examination of the architectural dependence of adversarial robustness.

To that end, in Fig. 1 (a) we plot the test set adversarial error due to an FGSM attack as a function of for several models on ImageNet BID15 .

We note that for < 0.2, the adversarial error follows a power law form with an exponent between 0.9 and 1.1 for all models studied.

Even adversarially trained models BID6 , while adversarially much more robust for larger values of , follow a similar form and reach as large as 20% adversarial error at smaller .In light of the surprising commonality in adversarial error at small-, we investigate whether there is any way to get a different form for the adversarial error.

To do this, we evaluate the adversarial error due to step l.l.

attack, which is computed as: DISPLAYFORM1 where y l.l. is least likely class predicted by the network on clean image x BID6 .

The adversarial error also follows a power law, however with a larger exponent.

The exponents range from 1.8 to 2.2 for ImageNet models ( Fig. 1(b) ), and 1.8 to 2.5 for models trained on MNIST ( FIG0 ) and CIFAR10.

Thus, we see that attack protocol can change the exponent of the observed power-law.

Figure 1: Test set adversarial error as a function of for models trained on ImageNet due to FGSM and step l.l.

attack in (a) and (b), respectively.

adv.

tr. denotes models that are adversarially trained BID6 .

In (b), we also show two of the power law fits with straight lines.

To test the limits of the universality observed in Fig. 1 , we perform a number of more extensive tests.

First, we investigate the effect of architecture by stochastically sampling thousands of different neural networks and train them on MNIST.

We then measure their adversarial error due to FGSM on the test set.

The architectures we sample are either fully-connected networks with 1-4 hidden layers and 30-2000 hidden nodes in each layer, or simple convolutional networks with dropout rates between 0-0.5.

The adversarial error of representative linear, fully-connected, and convolutional networks are shown in FIG0 .

As above, these models all have the same form of adversarial error with a powerlaw dependence on with exponents between 0.9 and 1.2.See Fig. 9 in the Appendix for a plot with all of the generated networks.

We perform the same analysis on a 32-layer ResNet trained on CIFAR10 BID2 , which achieves a 92.6% clean accuracy on the test set.

The result is shown in FIG6 , where the adversarial error follows a power law with an exponent of 0.99 up to an of 1.Next, we probe the relationship between generalization and adversarial robustness following a similar approach to .

In particular, we train a fully connected network on MNIST with shuffled labels until it reaches perfect accuracy on the training set.

The adversarial error on the training set is shown in FIG0 (b).

Once again we see that the adversarial error follows an almost identical power-law form at small with an exponent of 1.2.Finally, we further investigate the dependence of adversarial robustness on attack protocol.

We plot in FIG0 (c) the adversarial error on MNIST with an L ??? -normalized FGSM attack, an L 2 -normalized FGSM attack, and a 20-step projected gradient descent (PGD) attack BID8 .

We see that despite the anomalous exponent observed for step-l.l.

attacks, the other attack methods display the same universality with exponents of 1.1, 1.2, and 1.3 for L2-norm, FGSM, and PGD attacks, respectively.

step-l.l. attack on MNIST has an exponent of 2.3.

We now offer a theoretical explanation for the observed universal behavior of adversarial error.

The breadth of these observations shows that adversarial error for small adversarial perturbations does not depend on the specifics of the neural network, which implies that we can understand the small regime by making simplifying approximations.

We begin by considering the linear response of a neural network to adversarial perturbations.

Another approach to adversarial examples that considers the linear response of the network can be found in BID10 .

The effect of margins on adversarial robustness has been brought up in ?

.We will study an L 2 -variant of the FGSM attack.

Here, the adversarial perturbation is given by, DISPLAYFORM0 instead of the more commonly used ??? variant.

As shown above, the form and exponent of adversarial error is qualitatively insensitive to this choice (see FIG0 (c)).

We will now attempt to compute the minimum , that we call?? (x), required before the class assigned to an input, x, changes.

Assuming the network was able to perfectly classify clean images, the adversarial error rate will then be P (?? < ).

While perfect classification will not be achieved in practice, the insensitivity of the form of adversarial error to clean accuracy demonstrated above for many systems suggests that this approximation is sound.

Notationally, we will refer to the output of the network as?? i (x) and the corresponding logits as h i (x).

The class prediction of the network will then be argmax(??(x)) = argmax(h(x)).

For simplicity we will choose an ordering of the logits such that DISPLAYFORM1 We can then enumerate a set of logit-differences, DISPLAYFORM2 If an adversarial perturbation is to successfully cause the network to make an erroneous prediction, then it must be true that DISPLAYFORM3 We calculate the response of the logits to adversarial perturbation.

We consider the linearized response of the network and find that in the limit of small (see Appendix 6.2.1), DISPLAYFORM4 where J ij = ???h j /???x i is the input-logit Jacobian of the network and ?? i = ???L/???h i is the error of the outputs of the network.

For notational convenience we will define ??(x) = J T J??/||J??|| 2 .

In this linear model we therefore predict that the logit-differences will scale as follows, DISPLAYFORM5 Recall that the adversarial perturbation will successfully cause the network to just barely misclassify an input precisely when ??? 1j (x adv ) = 0 for at least one j.

We can predict per-class -thresholds beyond which the network will misclassify a given point, DISPLAYFORM6 Together this allows us to compute a linear approximation to?? given by?? linear (x) = min j ( j (x)).We can confirm that the change in the logits for small changes in the inputs is well-described by this linear model.

This is shown in FIG1 (a) where we see the logits for a single example upon perturbation over a range of .

In particular, for small we see an excellent agreement between the linear approximation and the true logit dynamics.

We also see that the where the first and second logit cross is well-approximated by the linear prediction.

In FIG1 (b) we plot?? linear (x) against?? (x) evaluated on every MNIST example in the test set.

The white dashed line is the lin?? (x) =?? linear (x).

We see that when?? (x) is small the?? linear (x) concentrate increasingly around th?? (x).

Together these results show that the linear response predictions are valid for small adversarial perturbations.

While the linear model outlined above gives excellent agreement in the ??? 0 limit, the ?? i (x) are themselves complicated objects (being functions of the Jacobian).

This makes the analytic evaluation of Eq. (6) challenging.

We therefore introduce a "mean-field" approximation to Eq. (6) by replacing ?? i (x) by its average over the dataset, ?? i (x) .

Similar independence approximations have previously been successful in analyzing the expressivity and trainability of neural networks BID16 BID14 .

Finally, we observe that the vast majority of the time (for example, more than 95% of successful FGSM attacks for < 50), it is ??? 12 (x adv ) that goes to zero before any of the other ??? 1j .

We therefore assume that this will be the dominant failure mode for neural networks and write down a mean-field estimate for?? , DISPLAYFORM7 We show in FIG1 (c) that this approximation continues to be strongly correlated with?? .

Together these results suggest that the adversarial error rate for perturbations of size will be DISPLAYFORM8 where we have defined?? = ( ?? 2 ??? ?? 1 ) to be a network-specific rescaling of .

We therefore expect the adversarial error rate at small to be dictated by the cumulative distribution of ??? 12 for attacks that effectively target the second most likely class (e.g. FGSM, PGD ...etc.).

With the results from the preceding section in hand, we now investigate the distribution of logit differences, P (??? 1j ).

Since we are particularly interested in the small regime, we seek to compute P (??? 1j ) for small ??? 1j .

To make progress we will again make a mean field approximation and assume that each of the logits are i.i.d.

with arbitrary distribution.

With this approximation we find that for small ??? 1j (see Appendix 6.2.2), DISPLAYFORM0 where C is a network specific constant.

An interesting consequence of this result is that the difference that we are particularly interested in, P (??? 12 ), scales as O(1) as ??? 12 ??? 0.

This implies that, generically, we expect the most likely logit and second most likely logit to have a finite probability of being arbitrarily close together.

We interpret this as an inherent uncertainty in the predictions of neural networks.

While it is not obvious that the assumption of a factorial logit distribution is valid here, we will see that Eq. (9) captures the universal features of the distribution at small values of the logit difference.

Indeed, from the previous section we see that the form of Eq. (9) implies that the adversarial error rate should scale as follows DISPLAYFORM1 as was broadly observed in the previous section.

To further test whether or not the mean field approximation is valid, we evaluate ??? 1j for a number of different neural network architectures and datasets.

We find that on all datasets and models studied, the ??? 1j distributions have power-law tails.

As predicted, ??? 12 has a power-law tail with an exponent of about 0, and ??? 1j for j > 2 have power-law tails with positive exponents increasing with j. We note, however, that the powers are typically not integral for large j. It seems likely that this breakdown is the result of correlations between the logits.

In FIG2 , we compare distribution of ??? 1j with j = 2, 3, 4 for ImageNet and logits that are independently sampled from a uniform distribution for 5 million samples with 10 classes.

In FIG3 we see similar results for MNIST.

Together these results verify our predictions over a vast set of networks and datasets.

The empirical results above show that adversarial error has a power-law form for well-studied and random datasets, simple full connected networks as well as complicated state-of-the-art models.

It follows that the prevalence and commonality of adversarial examples is not due to the depth of the model (for example, see the linear model in FIG0 ), or the high-dimensionality of the datasets.

Rather, they are due to the fact that lots of examples have small ??? 12 values.

This makes it easy to find examples to fool the model at test-time.

It is interesting that the distribution of ??? 1j , at small ??? 1j , for trained models is essentially identical to that of i.i.d.

random logits (especially for j = 2).

This suggests that while our training procedures are good at modifying the largest logit in a way that leads to good clean accuracies, these procedures do not induce strong enough correlations between the logits to disrupt the essential scaling uncovered above.

This problem is reminiscent of the problem distillation BID3 attempts to solve, by incorporating information about ratios of incorrect classes.

This might be one of the reasons defensive distillation improves adversarial robustness.

It would be interesting to study the distributions of logit differences during training of distillation networks.

Given the large density of small ??? 12 values, we study an entropy penalty regularizer to make models more robust.

Our proposed loss function can be written as: loss = old loss ??? ?? n i=1 p i log p i .

where ?? is a hyperparameter, n is the number of classes, and p i are the outputs of the neural network.

This regularization term has been used by BID9 for semi-supervised learning tasks.

It aims to increase the confidence of the network on each sample, which is the opposite of previous regularization attempts that penalized confidence to increase generalization accuracy BID13 BID18 .

By penalizing the entropy of the softmax outputs, we aim to increase the logit differences.

In Fig. 5 , we show that entropy regularization with a ?? = 4.5 increases the adversarial robustness both for regularly trained networks and step l.l.

adversarially trained networks, and both for permutation invariant and regular MNIST.

We note that the same qualitative results hold for other values of ?? we tried.

Despite the increase in adversarial accuracy, the permutation invariant MNIST model has 0.8% lower clean accuracy when trained with the entropy penalty.

In FIG7 , we show that a wide ResNet trained with the entropy regularizer has improved robustness with no loss in clean accuracy.

Figure 5: Step l.l.

attack adversarial accuracy as a function of for CNN and permutation invariant MNIST in (a) and (b), respectively.

Regular training (purple), entropy regularization (red), adversarial training (green), and adversarial training with entropy regularization (blue) have been implemented.

Adversarial training was done using the step l.l.

method.

In (c), we show the PGD attack adversarial accuracy on permutation invariant MNIST trained with and without step l.l.

adversarial training.

We investigate whether the increased adversarial robustness is due to increased logit differences.

In FIG3 , we plot the distribution of ??? 1j for j up to 4, for two networks trained on permutation invariant MNIST, with and without entropy regularization.

As expected, margins are shifted to larger values and density of samples with small ??? 1j are reduced.

The tails still follow a power-law form with the same exponents, however there are fewer samples with small margins compared to a regularly trained network.

Although entropy regularization made our networks white-box attacks, it did not lead to a significant improvement against black-box attacks.

For this reason, we focus on the influence of network architectures on adversarial sensitivity below.

Several recent papers observe that larger networks are more robust against adversarial examples, regardless if they are adversarially trained or not BID6 BID8 .

However, it is not clear if network architectures play an important role in adversarial robustness.

Are larger models more robust because they have more trainable parameters, or simply because they have higher clean accuracy?

Is it possible to find more robust network architectures that do not necessarily have more parameters?

We run several experiments to answer these questions, as well as to find an adversarially more robust model on CIFAR10.

We perform neural architecture search (NAS) with reinforcement learning.

Our search space and procedure are almost exactly the same as in BID25 .

One difference is that we restrict the search space so that the normal cell must be the same as the reduction cell.

This reduces the complexity of the search space as now we have only half as many predictions.

Finally, we increase the number of prediction steps from 5 to 7 to slightly gain back the complexity that was lost when we restricted the normal cell to be equal to the reduction cell.

We carry out two experiments:??? Experiment 1: NAS where child models are trained with clean and step l.l.

adversarial examples and the reward is computed on the validation set with FGSM adversarial accuracy at = 8.??? Experiment 2: NAS where child models are trained with clean and PGD adversarial examples and the reward is computed on the validation set with FGSM adversarial accuracy at = 8.In both experiments, child models are trained for 10 epochs on a training set of 25 thousand samples.

Child models are trained on mini-batches where half of the samples are adversarially perturbed, following the procedure in BID6 .

At the end of each experiment, we pick the child model with the highest FGSM adversarial accuracy at = 8 on the validation set of 5000 samples, and scale up the number of filters.

We train the enlarged models for 100 epochs on the full training set of 45000 samples for 12 different hyperparameter sets, and pick the one with the highest adversarial accuracy on the validation set.

Finally, we report below the performance of these models on a held-out test set of 10 thousand samples.

To provide a comparison with the results of our two experiments, we also run a vanilla NAS where the reward is clean validation accuracy.

We will refer to the best architecture from vanilla NAS as NAS Baseline.

When trained using the setup above only on clean examples, NAS Baseline reaches a test set accuracy of 95.3%.

We present the results of Experiment 1 in FIG4 .

Here the green curve is the adversarial accuracy NAS Baseline.

The blue curve is the adversarial accuracy of a network architecture that was found by Experiment 1.

Both of these architectures are trained with the same adversarial training procedure.

We try the same sets of hyperparameters and report here the models with best adversarial accuracy at = 8 on the validation set.

Adversarial training reduced the clean accuracy by 0.2%.

Adversarially trained models both have clean accuracy of 95.1% on the test set, whereas the model that was trained without adversarial training reached 95.3% accuracy.

We next use PGD adversarial examples in the training of child models, to find architectures that are more robust to any adversarial attack within an ball BID8 .

Following the training procedure by BID8 , we use 7 steps of size 2, for a total = 8.

We present the results of Experiment 2 in FIG4 .

As was the case in Experiment 1, the architecture found by adversarial NAS leads to a more robust model.

At = 8, the architecture from Experiment 2 Table 1 : Performance of our best architecture from Experiment 2 at = 8.

Black-box attacks are sourced from a copy of the network independently initialized and trained.reaches a 17% higher adversarial accuracy on PGD examples.

We compare our results to the results by BID8 .

BID8 trained only on PGD examples, whereas half of our minibatches are clean examples.

Despite this, we match their accuracy on white-box PGD attacks.

Against other white-and black-box attacks our model is more robust, and our clean accuracy is 5.9% higher.

We also note that NAS Baseline model has 4.9 million trainable parameters, whereas the model from Experiments 1 and 2 have 2.3 million and 3.5 million parameters, respectively.

NAS found an adversarially more robust architecture with many fewer parameters.

Best architecture from Experiment 2 and NAS Baseline are presented in Appendix FIG3 .

Finally, we study the performance statistics of child models during NAS.

In FIG5 , we report the results for 9360 child models that were trained during Experiment 1.

As explained above, these models are only trained for 10 epochs.

In FIG5 , we see that the correlation between adversarialy accuracy and the number of trainable parameters of the model is not very strong.

On the other hand, adversarial accuracy is strongly correlated with clean accuracy FIG5 ).

We hypothesize that this is the reason both BID8 and BID6 found that making networks larger increased adversarial robustness, because it also increased the clean accuracy.

This implies that commonly used architectures, like Inception v3 and ResNet, benefit from having more parameters.

This however was not the case for most child models during NAS.

On the other hand, having a high clean accuracy is not sufficient for adversarial robustness.

As seen in FIG5 , there is a large variance in the adversarial accuracy of models with good clean accuracy.

The range of adversarial accuracies in the histogram of models with larger than 85% clean accuracy is 22% and the standard deviation is 2.6%.

For this reason, our experiments led to more robust architectures than NAS Baseline.

In this paper we studied common properties of adversarial examples across different models and datasets.

We theoretically derived a universality in logit differences and adversarial error of machine learning models.

We showed that architecture plays an important role in adversarial robustness, which correlates strongly with clean accuracy.

such that t ?? = 1 if ?? = ?? for some ?? and t ?? = 0 otherwise.

We assume our network gets the answer correct so that h ?? > h ?? for all ?? = ??.

Then we apply the adversarial perturbation, DISPLAYFORM0 Note that we can write DISPLAYFORM1 Where we associate J ???? = ???h ?? /???x ?? with the input-to-logit Jacobian linking the inputs to the logits and ?? = ???L/???h ?? the error of the outputs of the network.

We can compute the change to the logits of the network due to this perturbation.

We find, DISPLAYFORM2 DISPLAYFORM3 DISPLAYFORM4 where we have plugged in for eq. (11).

Expressing the above equation in terms of the Jacobian, it follows that we can write the effect of the adversarial perturbation on the logits by, DISPLAYFORM5 as postulated.

To make progress we will again make a mean field approximation and assume that each of the logits are i.i.d.

with arbitrary distribution P (h).

We denote the cumulative distribution F (h).

While it is not obvious that the factorial approximation is valid here, we will see that the resulting distribution of P (??? 1j ) shares many qualitative similarities with the distribution observed in real networks.

We first change variables from the logits to a sorted version of the logits, r i .

The ranked logits are defined such that r 1 = max({h i }), r 2 = max({h i }\{r 1 }), ?? ?? ?? .

Our first result is to compute the resulting joint distribution between r 1 and r j , P j (r 1 , r j ) = A(N, j)F N ???j (r j ) [F (r 1 ) ??? F (r j )]

j???2 P (r j )P (r 1 )where A(N, j) = N (N ??? 1) N ???2 j???2 is a combinatorial factor.

Eq. (18) has a simple interpretation.

F N ???j (r j ) is the probability that there are N ??? j variables less than r j ; [F (r 1 ) ??? F (r j )]

j???2 is the probability that j ??? 2 variables are between r j and r 1 ; P (r j )P (r 1 ) is the probability that there is one variable equal to each of r 1 and r j .

The combinatorial factor can be understood since there are N ways of selecting r 1 , N ??? 1 ways of selecting r j , and N ???2 j???2 ways of choosing j ??? 2 variables out of the remaining N ??? 2 to be between r j and r 1 .In terms of eq. FORMULA0 we can compute the distribution over ??? 1j to be given by, P (??? 1j ) = drP j (r + ??? 1j , r)= A(N, j) drF N ???j (r) [F (r + ??? 1j ) ??? F (r)]

j???2 P (r)P (r + ??? 1j ).We can analyze this equation for small ??? 1j .

Expanding to lowest order in ??? 1j , P (??? 1j ) ??? A(N, j) drF N ???j (r) [F (r) + ??? 1j P (r) ??? F (r)] j???2 P (r) P (r) + ??? 1j dP (r) dr ( Since the term in the integral does not depend on ??? 1j the result follows with, DISPLAYFORM6 6.2.3 ARCHITECTURES FIG3 : Left: Best architecture from Experiment 1.

Right: Architecture of NAS Baseline.

We note that the architecture from Experiment 1 is "longer" and "narrower" than previous architectures found by NAS for higher clean accuracy BID24 BID25 .

<|TLDR|>

@highlight

Adversarial error has similar power-law form for all datasets and models studied, and architecture matters.
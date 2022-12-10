Adversarial training is by far the most successful strategy for improving robustness of neural networks to adversarial attacks.

Despite its success as a defense mechanism, adversarial training fails to generalize well to unperturbed test set.

We hypothesize that this poor generalization is a consequence of adversarial training with uniform perturbation radius around every training sample.

Samples close to decision boundary can be morphed into a different class under a small perturbation budget, and enforcing large margins around these samples produce poor decision boundaries that generalize poorly.

Motivated by this hypothesis, we propose instance adaptive adversarial training -- a technique that enforces sample-specific perturbation margins around every training sample.

We show that using our approach, test accuracy on unperturbed samples improve with a marginal drop in robustness.

Extensive experiments on CIFAR-10, CIFAR-100 and Imagenet datasets demonstrate the effectiveness of our proposed approach.

A key challenge when deploying neural networks in safety-critical applications is their poor stability to input perturbations.

Extremely tiny perturbations to network inputs may be imperceptible to the human eye, and yet cause major changes to outputs.

One of the most effective and widely used methods for hardening networks to small perturbations is "adversarial training" (Madry et al., 2018) , in which a network is trained using adversarially perturbed samples with a fixed perturbation size.

By doing so, adversarial training typically tries to enforce that the output of a neural network remains nearly constant within an p ball of every training input.

Despite its ability to increase robustness, adversarial training suffers from poor accuracy on clean (natural) test inputs.

The drop in clean accuracy can be as high as 10% on CIFAR-10, and 15% on Imagenet (Madry et al., 2018; Xie et al., 2019) , making robust models undesirable in some industrial settings.

The consistently poor performance of robust models on clean data has lead to the line of thought that there may be a fundamental trade-off between robustness and accuracy (Zhang et al., 2019; Tsipras et al., 2019) , and recent theoretical results characterized this tradeoff (Fawzi et al., 2018; Shafahi et al., 2018; Mahloujifar et al., 2019) .

In this work, we aim to understand and optimize the tradeoff between robustness and clean accuracy.

More concretely, our objective is to improve the clean accuracy of adversarial training for a chosen level of adversarial robustness.

Our method is inspired by the observation that the constraints enforced by adversarial training are infeasible; for commonly used values of , it is not possible to achieve label consistency within an -ball of each input image because the balls around images of different classes overlap.

This is illustrated on the left of Figure 1 , which shows that the -ball around a "bird" (from the CIFAR-10 training set) contains images of class "deer" (that do not appear in the training set).

If adversarial training were successful at enforcing label stability in an = 8 ball around the "bird" training image, doing so would come at the unavoidable cost of misclassifying the nearby "deer" images that come along at test time.

At the same time, when training images lie far from the decision boundary (eg., the deer image on the right in Fig 1) , it is possible to enforce stability with large with no compromise in clean accuracy.

When adversarial training on CIFAR-10, we see that = 8 is too large for some images, causing accuracy loss, while being unnecessarily small for others, leading to sub-optimal robustness.

Figure 1: Overview of instance adaptive adversarial training.

Samples close to the decision boundary (bird on the left) have nearby samples from a different class (deer) within a small L p ball, making the constraints imposed by PGD-8 / PGD-16 adversarial training infeasible.

Samples far from the decision boundary (deer on the right) can withstand large perturbations well beyond = 8.

Our adaptive adversarial training correctly assigns the perturbation radius (shown in dotted line) so that samples within each L p ball maintain the same class.

The above observation naturally motivates adversarial training with instance adaptive perturbation radii that are customized to each training image.

By choosing larger robustness radii at locations where class manifolds are far apart, and smaller radii at locations where class manifolds are close together, we get high adversarial robustness where possible while minimizing the clean accuracy loss that comes from enforcing overly-stringent constraints on images that lie near class boundaries.

As a result, instance adaptive training significantly improves the tradeoff between accuracy and robustness, breaking through the pareto frontier achieved by standard adversarial training.

Additionally, we show that the learned instance-specific perturbation radii are interpretable; samples with small radii are often ambiguous and have nearby images of another class, while images with large radii have unambiguous class labels that are difficult to manipulate.

Parallel to our work, we found that Ding et al. (2018) uses adaptive margins in a max-margin framework for adversarial training.

Their work focuses on improving the adversarial robustness, which differs from our goal of understanding and improving the robustness-accuracy tradeoff.

Moreover, our algorithm for choosing adaptive margins significantly differs from that of Ding et al. (2018) .

Adversarial attacks are data items containing small perturbations that cause misclassification in neural network classifiers (Szegedy et al., 2014) .

Popular methods for crafting attacks include the fast gradient sign method (FGSM) (Goodfellow et al., 2015) which is a one-step gradient attack, projected gradient descent (PGD) (Madry et al., 2018) which is a multi-step extension of FGSM, the C/W attack (Carlini & Wagner, 2017 ), DeepFool (Moosavi-Dezfooli et al., 2016 , and many more.

All these methods use the gradient of the loss function with respect to inputs to construct additive perturbations with a norm-constraint.

Alternative attack metrics include spatial transformer attacks (Xiao et al., 2018) , attacks based on Wasserstein distance in pixel space (Wong et al., 2019) , etc.

Defending against adversarial attacks is a crucial problem in machine learning.

Many early defenses (Buckman et al., 2018; Samangouei et al., 2018; Dhillon et al., 2018) , were broken by strong attacks.

Fortunately, adversarially training is one defense strategy that remains fairly resistant to most existing attacks.

denote the set of training samples in the input dataset.

In this paper, we focus on classification problems, hence, y i ∈ {1, 2, . . .

N c }, where N c denotes the number of classes.

Let f θ (x) : R c×m×n → R Nc denote a neural network model parameterized by θ.

Classifiers are often trained by minimizing the cross entropy loss given by

whereỹ i is the one-hot vector corresponding to the label y i .

In adversarial training, instead of optimizing the neural network over the clean training set, we use the adversarially perturbed training set.

Mathematically, this can be written as the following min-max problem

This problem is solved by an alternating stochastic method that takes minimization steps for θ, followed by maximization steps that approximately solve the inner problem using k steps of PGD.

For more details, refer to Madry et al. (2018) .

end if 8:

S + = {i|f (x i ) is correctly classified as y i }

S − = {i|f (x i ) is incorrectly classified as y i } 11:

12: end for

To remedy the shortcomings of uniform perturbation radius in adversarial training (Section 1), we propose Instance Adaptive Adversarial Training (IAAT), which solves the following optimization:

Like vanilla adversarial training, we solve this by sampling mini-batches of images {x i }, crafting adversarial perturbations {δ i } of size at most { i }, and then updating the network model using the perturbed images.

The proposed algorithm is distinctive in that it uses a different i for each image x i .

Ideally, we would choose each i to be as large as possible without finding images of a different class within the i -ball around x i .

Since we have no a-priori knowledge of what this radius is, we use a simple heuristic to update i after each epoch.

After crafting a perturbation for x i , we check if the perturbed image was a successful adversarial example.

If PGD succeeded in finding an image with a different class label, then i is too big, so we replace i ← i − γ.

If PGD failed, then we set i ← i + γ.

Since the network is randomly initialized at the start of training, random predictions are made, and this causes { i } to shrink rapidly.

For this reason, we begin with a warmup period of a few (usually 10 epochs for CIFAR-10/100) epochs where adversarial training is performed using uniform for every sample.

After the warmup period ends, we perform instance adaptive adversarial training.

A detailed training algorithm is provided in Alg.

1.

Set i = 2 8: else 9:

To evaluate the robustness and generalization of our models, we report the following metrics: (1) test accuracy of unperturbed (natural) test samples, (2) adversarial accuracy of white-box PGD attacks, (3) adversarial accuracy of transfer attacks and (4) accuracy of test samples under common image corruptions (Hendrycks & Dietterich, 2019) .

Following the protocol introduced in Hendrycks & Dietterich (2019), we do not train our models on any image corruptions.

On CIFAR-10 and CIFAR-100 datasets, we perform experiments on Resnet-18 and WideRenset-32-10 models following (Madry et al., 2018; Zhang et al., 2019) .

All models are trained on PGD-10 attacks i.e., 10 steps of PGD iterations are used for crafting adversarial attacks during training.

In the whitebox setting, models are evaluated on: (1) PGD-10 attacks with 5 random restarts, (2) PGD-100 attacks with 5 random restarts, and (3) PGD-1000 attacks with 2 random restarts.

For transfer attacks, an independent copy of the model is trained using the same training algorithm and hyper-parameter settings, and PGD-1000 adversarial attacks with 2 random restarts are crafted on the surrogate model.

For image corruptions, following (Hendrycks & Dietterich, 2019) , we report average classification accuracy on 19 image corruptions.

Beating the robustness-accuracy tradeoff:

In adversarial training, the perturbation radius is a hyper-parameter.

Training models with varying produces a robustness-accuracy tradeoff curvemodels with small training achieve better natural accuracy and poor adversarial robustness, while models trained on large have improved robustness and poor natural accuracy.

To generate this tradeoff, we perform adversarial training with in the range {1, 2, . . .

8}. Instance adaptive adversarial training is then compared with respect to this tradeoff curve in Fig. 3a , 3b.

Two versions of IAAT are reported -with and without a warmup phase.

In both versions, we clearly achieve an improvement over the accuracy-robustness tradeoff.

Use of the warmup phase helps retain robustness with a drop in natural accuracy compared to its no-warmup counterpart.

Clean accuracy improves for a fixed level of robustness: On CIFAR-10, as shown in Table.

1, we observe that our instance adaptive adversarial training algorithm achieves similar adversarial robustness as the adversarial training baseline.

However, the accuracy on clean test samples increases by 4.06% for Resnet-18 and 4.49% for WideResnet-32-10.

We also observe that the adaptive training algorithm improves robustness to unseen image corruptions.

This points to an improvement in overall generalization ability of the network.

On CIFAR-100 (Table.

2), the performance gain in natural test accuracy further increases -8.79% for Resnet-18, and 9.22% for Wideresnet-32-10.

The adversarial robustness drop is marginal.

Maintaining performance over a range of test : Next, we plot adversarial robustness over a sweep of values used to craft attacks at test time.

Fig. 4a, 4b shows an adversarial training baseline with = 8 performs well at high regimes and poorly at low regimes.

On the other hand, adversarial training with = 2 has a reverse effect, performing well at low and poorly at high regimes.

Our instance adaptive training algorithm maintains good performance over all regimes, achieving slightly less performance than the = 2 model for small test , and dominating all models for larger test .

Interpretability of : We find that the values of i chosen by our adaptive algorithm correlate well with our own human concept of class ambiguity.

Figure 2 (and Figure 6 in Appendix B) shows that a sampling of images that receive small i contains many ambiguous images, and these images are perturbed into a (visually) different class using = 16.

In contrast, images that receive a large i have a visually definite class, and are not substantially altered by an = 16 perturbation.

Robustness to other attacks: While our instance adaptive algorithm is trained on PGD attacks, we are interested to see if the trained model improves robustness on other adversarial attacks.

As shown in Table.

3, IAAT achieves similar level of robustness as adversarial training on other gradient-based attacks, while improving the natural accuracy.

Xie et al. (2019) , we attack Imagenet models using random targeted attacks instead of untargeted attacks as done in previous experiments.

During training, adversarial attacks are generated using 30 steps of PGD.

As a baseline, we use adversarial training with a fixed of 16/255.

This is the setting used in Xie et al. (2019) .

Adversarial training on Imagenet is computationally intensive.

To make training practical, we use distributed training with synchronized SGD on 64/128 GPUs.

More implementation details can be found in Appendix E.

At test time, we evaluate the models on clean test samples and on whitebox adversarial attacks with = {4, 8, 12, 16}. PGD-1000 attacks are used.

Additionally, we also report normalized mean corruption error (mCE), an evaluation metric introduced in Hendrycks & Dietterich (2019) to test the robustness of neural networks to image corruptions.

This metric reports mean classification error of different image corruptions averaged over varying levels of degradation.

Note that while accuracies are reported for natural and adversarial robustness, mCE reports classification errors, so lower numbers are better.

Our experimental results are reported in Table.

4.

We observe a huge drop in natural accuracy for adversarial training (25%, 22% and 20% drop for Resnet-50, 101 and 152 respectively).

Adaptive adversarial training significantly improves the natural accuracy -we obtain a consistent performance gain of 10+% on all three models over the adversarial training baseline.

On whitebox attacks, IAAT outperforms the adversarial training baseline on low regimes, however a drop of 13% is observed at high 's ( = 16).

On the corruption dataset, our model consistently outperforms adversarial training.

Recall from Section 3 that during warmup, adversarial training is performed with uniform normbound constraints.

Once the warmup phase ends, we switch to instance adaptive training.

From Table 5 and 6, we observe that when warmup is used, adversarial robustness improves with a small drop in natural accuracy, with more improvements observed in CIFAR-100.

However, as shown in Fig. 3a and 3b, both these settings improve the accuracy-robustness tradeoff.

We are interested in estimating instance-specific perturbation radius i such that predictions are consistent within the chosen i -ball.

To obtain an exact estimate of such an i , we can perform a line search as follows: Given a discretization η and a maximum perturbation radius max , generate PGD attacks with radii {iη}

.

Choose the desired i as the maximum iη for which the prediction remains consistent as that of the ground-truth label.

We compare the performance of exact line search with that of IAAT in Table 7 .

We observe that exact line search marginally improves compared to IAAT.

However, exact line search is computationally expensive as it requires performing max /η additional PGD computations, whereas IAAT requires only 2.

In this work, we focus on improving the robustness-accuracy tradeoff in adversarial training.

We first show that realizable robustness is a sample-specific attribute: samples close to the decision boundary can only achieve robustness within a small ball, as they contain samples from a different class beyond this radius.

On the other hand samples far from the decision boundary can be robust on a relatively large perturbation radius.

Motivated by this observation, we develop instance adaptive adversarial training, in which label consistency constraints are imposed within sample-specific perturbation radii, which are in-turn estimated.

Our proposed algorithm has empirically been shown to improve the robustness-accuracy tradeoff in CIFAR-10, CIFAR-100 and Imagenet datasets.

A recent paper that addresses the problem of improving natural accuracy in adversarial training is mixup adversarial training (Lamb et al., 2019) , where adversarially trained models are optimized using mixup loss instead of the standard cross-entropy loss.

In this paper, natural accuracy was shown to improve with no drop in adversarial robustness.

However, the robustness experiments were not evaluated on strong attacks (experiments were reported only on PGD-20).

We compare our implementation of mixup adversarial training with IAAT on stronger attacks in Table.

8.

We observe that while natural accuracy improves for mixup, drop in adversarial accuracy is much higher than IAAT.

A visualization of samples from CIFAR-10 dataset with the corresponding value assigned by IAAT is shown in Figure.

5.

We observe that samples for which low 's are assigned are visually confusing (eg., top row of Figure.

5), while samples with high distinctively belong to one class.

In addition, we also show more visualizations of samples near decision boundary which contain samples from a different class within a fixed ∞ ball in Figure.

6.

The infeasibility of label consistency constraints within the commonly used perturbation radius of ∞ = 8 is apparent in this visualization.

Our algorithm effectively chooses an appropriate that retains label information within the chosen radius.

Next, we visualize the evolution of over epochs in adaptive adversarial training.

A plot showing the average growth, along with the progress of 3 randomly picked samples are shown in Fig. 7a and 7b.

We observe that average converges to around 11, which is higher than the default setting of = 8 used in adversarial training.

Also, each sample has a different profile -for some, increases well beyond the commonly use radius of = 8, while for others, it converges below it.

In addition, a plot showing the histogram of 's at different snapshots of training is shown in Fig. 8 .

We observe an increase in spread of the histogram as the training progresses.

Testing against a strong adversary is crucial to assess the true robustness of a model.

A popular practice in adversarial robustness community is to attack models using PGD with many attack iterations (Xie et al., 2019) .

So, we test our instance adaptive adversarially trained models on a sweep of PGD iterations for a fixed level.

Following (Xie et al., 2019) , we perform the sweep upto 2000 attack steps fixing = 16.

The resulting plot is shown in Figure.

9.

For all three Resnet models, we observe a saturation in adversarial robustness beyond 500 attack iterations.

As shown in Alg.

2, IAAT algorithm has two hyper-parameters -smoothing constant β and discretization γ.

In this section, we perform a sensitivity analysis of natural and robust accuracies by varying these hyper-parameters.

Results are reported in Table.

9.

We observe that the algorithm is not too sensistive to the choice of hyper-parameters.

But the best performance is obtained for γ = 1.9 and β = 0.1.

On CIFAR-10 and CIFAR-100 datasets, our implementation follows the standard adversarial training setting used in Madry et al. (2018) .

During training, adversarial examples are generated using PGD-10 attacks, which are then used to update the model.

All hyperparameters we used are tabulated in Table.

10.

For Imagenet implementation, we mimic the setting used in Xie et al. (2019) .

During training, adversaries are generated with PGD-30 attacks.

This is computationally expensive as every training update is followed by 30 backprop iterations to generate the adversarial attack.

To make training feasible, we perform distributed training using synchronized SGD updates on 64 / 128 GPUs.

We follow the training recipe introduced in Goyal et al. (2017) for large batch training.

Also, during training, adversarial attacks are generated with FP-16 precision.

However, in test phase, we use FP-32.

We further use two more tricks to speed-up instance adaptive adversarial training: (1) A weaker attacker(PGD-10) is used in the algorithm for selecting (Alg.

2).

(2) After i is selected per Alg.

2, we clip it with a lower-bound i.e., i ← max( i , lb ).

lb = 4 was used in our experiments.

Hyperparameters used in our experiments are reported in Table 11 .

All our models were trained on PyTorch.

Resnet-50 model was trained on 64 Nvidia V100 GPUs, while Resnet-101 and Resnet-152 models were trained on 128 GPUs.

Time taken for instance adaptive adversarial training for all models is reported in Table.

12.

<|TLDR|>

@highlight

Instance adaptive adversarial training for improving robustness-accuracy tradeoff
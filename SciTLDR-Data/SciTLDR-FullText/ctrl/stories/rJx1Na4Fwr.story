Adversarial training is one of the most popular ways to learn robust models but is usually attack-dependent and time costly.

In this paper, we propose the MACER algorithm, which learns robust models without using adversarial training but performs better than all existing provable l2-defenses.

Recent work shows that randomized smoothing can be used to provide certified l2 radius to smoothed classifiers, and our algorithm trains provably robust smoothed classifiers via MAximizing the CErtified Radius (MACER).

The attack-free characteristic makes MACER faster to train and easier to optimize.

In our experiments, we show that our method can be applied to modern deep neural networks on a wide range of datasets, including Cifar-10, ImageNet, MNIST, and SVHN.

For all tasks, MACER spends less training time than state-of-the-art adversarial training algorithms, and the learned models achieve larger average certified radius.

Modern neural network classifiers are able to achieve very high accuracy on image classification tasks but are sensitive to small, adversarially chosen perturbations to the inputs (Szegedy et al., 2013; Biggio et al., 2013) .

Given an image x that is correctly classified by a neural network, a malicious attacker may find a small adversarial perturbation δ such that the perturbed image x + δ, though visually indistinguishable from the original image, is assigned to a wrong class with high confidence by the network.

Such vulnerability creates security concerns in many real-world applications.

Researchers have proposed a variety of defense methods to improve the robustness of neural networks.

Most of the existing defenses are based on adversarial training (Szegedy et al., 2013; Madry et al., 2017; Goodfellow et al., 2015; Huang et al., 2015; Athalye et al., 2018) .

During training, these methods first learn on-the-fly adversarial examples of the inputs with multiple attack iterations and then update model parameters using these perturbed samples together with the original labels.

However, such approaches depend on a particular (class of) attack method.

It cannot be formally guaranteed whether the resulting model is also robust against other attacks.

Moreover, attack iterations are usually quite expensive.

As a result, adversarial training runs very slowly.

Another line of algorithms trains robust models by maximizing the certified radius provided by robust certification methods (Weng et al., 2018; Gowal et al., 2018; Zhang et al., 2019c) .

Using linear or convex relaxations of fully connected ReLU networks, a robust certification method computes a "safe radius" r for a classifier at a given input such that at any point within the neighboring radius-r ball of the input, the classifier is guaranteed to have unchanged predictions.

However, the certification methods are usually computationally expensive and can only handle shallow neural networks with ReLU activations, so these training algorithms have troubles in scaling to modern networks.

In this work, we propose an attack-free and scalable method to train robust deep neural networks.

We mainly leverage the recent randomized smoothing technique (Cohen et al., 2019) .

A randomized smoothed classifier g for an arbitrary classifier f is defined as g(x) = E η f (x + η), in which η ∼ N (0, σ 2 I).

While Cohen et al. (2019) derived how to analytically compute the certified radius of the randomly smoothed classifier g, they did not show how to maximize that radius to make the classifier g robust.

Salman et al. (2019) proposed SmoothAdv to improve the robustness of g, but it still relies on the expensive attack iterations.

Instead of adversarial training, we propose to learn robust models by directly taking the certified radius into the objective.

We outline a few challenging desiderata any practical instantiation of this idea would however have to satisfy, and provide approaches to address each of these in turn.

A discussion of these desiderata, as well as a detailed implementation of our approach is provided in Section 4.

And as we show both theoretically and empirically, our method is numerically stable and accounts for both classification accuracy and robustness.

Our contributions are summarized as follows:

• We propose an attack-free and scalable robust training algorithm by MAximizing the CErtified Radius (MACER).

MACER has the following advantages compared to previous works: -Different from adversarial training, we train robust models by directly maximizing the certified radius without specifying any attack strategies, and the learned model can achieve provable robustness against any possible attack in the certified region.

Additionally, by avoiding time-consuming attack iterations, our proposed algorithm runs much faster than adversarial training.

-Different from other methods that maximize the certified radius but are not scalable to deep neural networks, our method can be applied to architectures of any size.

This makes our algorithm more practical in real scenarios.

• We empirically evaluate our proposed method through extensive experiments on Cifar-10, ImageNet, MNIST, and SVHN.

On all tasks, MACER achieves better performance than state-of-the-art algorithms.

MACER is also exceptionally fast.

For example, on ImageNet, MACER uses 39% less training time than adversarial training but still performs better.

Neural networks trained by standard SGD are not robust -a small and human imperceptible perturbation can easily change the prediction of a network.

In the white-box setting, methods have been proposed to construct adversarial examples with small ∞ or 2 perturbations (Goodfellow et al., 2015; Madry et al., 2017; Carlini & Wagner, 2016; Moosavi-Dezfooli et al., 2015) .

Furthermore, even in the black-box setting where the adversary does not have access to the model structure and parameters, adversarial examples can be found by either transfer attack (Papernot et al., 2016) or optimization-based approaches (Chen et al., 2017; Rauber et al., 2017; Cheng et al., 2019) .

It is thus important to study how to improve the robustness of neural networks against adversarial examples.

Adversarial training So far, adversarial training has been the most successful robust training method according to many recent studies.

Adversarial training was first proposed in Szegedy et al. (2013) and Goodfellow et al. (2015) , where they showed that adding adversarial examples to the training set can improve the robustness against such attacks.

More recently, Madry et al. (2017) showed that adversarial training can be formulated as a min-max optimization problem and demonstrated that adversarial training with PGD attack can lead to very robust models empirically.

Zhang et al. (2019b) further proposed to decompose robust error as the sum of natural error and boundary error to achieve better performance.

Although models obtained by adversarial training empirically achieve good performance, they do not have certified error guarantees.

Despite the popularity of PGD-based adversarial training, one major issue is that its speed is too slow.

Some recent papers propose methods to accelerate adversarial training.

For example, Freem (Shafahi et al., 2019) replays an adversarial example several times in one iteration, YOPO-m-n (Zhang et al., 2019a) restricts back propagation in PGD within the first layer, and Qin et al. (2019) estimates the adversary with local linearization.

Robustness certification and provable defense Many defense algorithms proposed in the past few years were claimed to be effective, but Athalye et al. (2018) showed that most of them are based on "gradient masking" and can be bypassed by more carefully designed attacks.

It is thus important to study how to measure the provable robustness of a network.

A robustness certification algorithm takes a classifier f and an input point x as inputs, and outputs a "safe radius" r such that for any δ subject to δ ≤ r, f (x) = f (x + δ).

Several algorithms have been proposed recently, including the convex polytope technique , abstract interpretation methods (Singh et al., 2018; and the recursive propagation algrithms (Weng et al., 2018; .

These methods can provide attack-agnostic robust error lower bounds.

Moreover, to achieve networks with nontrivial certified robust error, one can train a network by minimizing the certified robust error computed by the above-mentioned methods, and several algorithms have been proposed in the past year Gowal et al., 2018; Zhang et al., 2019c; .

Unfortunately, they can only be applied to shallow networks with limited activation and run very slowly.

More recently, researchers found a new class of certification methods called randomized smoothing.

The idea of randomization has been used for defense in several previous works (Xie et al., 2017; Liu et al., 2018) but without any certification.

Later on, Lecuyer et al. (2018) first showed that if a Gaussian random noise is added to the input or any intermediate layer.

A certified guarantee on small 2 perturbation can be computed via differential privacy.

Li et al. (2018) and Cohen et al. (2019) then provided improved ways to compute the 2 certified robust error for Gaussian smoothed models.

In this paper, we propose a new algorithm to train on these 2 certified error bounds to significantly reduce the certified error and achieve better provable adversarial robustness.

It is thus quite natural to improve model robustness via maximizing the robust radius.

Unfortunately, computing the robust radius (1) of a classifier induced by a deep neural network is very difficult.

Weng et al. (2018) showed that computing the l 1 robust radius of a deep neural network is NP-hard.

Although there is no result for the l 2 radius yet, it is very likely that computing the l 2 robust radius is also NP-hard.

Certified radius Many previous works proposed certification methods that seek to derive a tight lower bound of R(f θ ; x, y) for neural networks (see Section 2 for related work).

We call this lower bound certified radius and denote it by CR(f θ ; x, y).

The certified radius satisfies 0 ≤ CR(f θ ; x, y) ≤ R(f θ ; x, y) for any f θ , x, y.

The certified radius leads to a guaranteed upper bound of the 0/1 robust classification error, which is called 0/1 certified robust error.

The 0/1 certified robust error of classifier f θ on sample (x, y) is defined as l

e. a sample is counted as correct only if the certified radius reaches .

The expectation of certified robust error over (x, y) ∼ p data serves as a performance metric of the provable robustness:

Recall that CR(f θ ; x, y) is a lower bound of the true robust radius, which immediately implies that L 0/1

Therefore, a small 0/1 certified robust error leads to a small 0/1 robust classification error.

Randomized smoothing In this work, we use the recent randomized smoothing technique (Cohen et al., 2019) , which is scalable to any architectures, to obtain the certified radius of smoothed deep neural networks.

The key part of randomized smoothing is to use the smoothed version of f θ , which is denoted by g θ , to make predictions.

The formulation of g θ is defined as follows.

Definition 1.

For an arbitrary classifier f θ ∈ F and σ > 0, the smoothed classifier g θ of f θ is defined as g θ (x) = arg max

In short, the smoothed classifier g θ (x) returns the label most likely to be returned by f θ when its input is sampled from a Gaussian distribution N (x, σ 2 I) centered at x. Cohen et al. (2019) proves the following theorem, which provides an analytic form of certified radius: Theorem 1. (Cohen et al., 2019) Let f θ ∈ F, and η ∼ N (0, σ 2 I).

Let the smoothed classifier g θ be defined as in (6).

Let the ground truth of an input x be y. If g θ classifies x correctly, i.e.

Then g θ is provably robust at x, with the certified radius given by

where Φ is the c.d.f.

of the standard Gaussian distribution.

As we can see from Theorem 1, the value of the certified radius can be estimated by repeatedly sampling Gaussian noises.

More importantly, it can be computed for any deep neural networks.

This motivates us to design a training method to maximize the certified radius and learn robust models.

To minimize the 0/1 robust classification error in (3) or the 0/1 certified robust error in (5), many previous works (Zhang et al., 2019b; Zhai et al., 2019) proposed to first decompose the error.

Note that a classifier g θ has a positive 0/1 certified robust error on sample (x, y) if and only if exactly one of the following two cases happens:

• g θ (x) = y, i.e. the classifier misclassifies x.

• g θ (x) = y, but CR(g θ ; x, y) < , i.e. the classifier is correct but not robust enough.

Thus, the 0/1 certified robust error can be decomposed as the sum of two error terms: a 0/1 classification error and a 0/1 robustness error:

Minimizing the 0-1 error directly is intractable.

A classic method is to minimize a surrogate loss instead.

The surrogate loss for the 0/1 classification error is called classification loss and denoted by l C (g θ ; x, y).

The surrogate loss for the 0/1 robustness error is called robustness loss and denoted by l R (g θ ; x, y).

Our final objective function is

We would like our loss functions l C (g θ ; x, y) and l R (g θ ; x, y) to satisfy some favorable conditions.

These conditions are summarized below as (C1) -(C3):

• (C1) (Surrogate condition): Surrogate loss should be an upper bound of the original error function, i.e. l C (g θ ; x, y) and l R (g θ ; x, y) should be upper bounds of 1 {g θ (x) =y} and 1 {g θ (x)=y,CR(g θ ;x,y)< } , respectively.

• (C2) (Differentiablity): l C (g θ ; x, y) and l R (g θ ; x, y) should be (sub-)differentiable with respect to θ.

• (C3) (Numerical stability): The computation of l C (g θ ; x, y) and l R (g θ ; x, y) and their (sub-)gradients with respect to θ should be numerically stable.

The surrogate condition (C1) ensures that l(g θ ; x, y) itself meets the surrogate condition, i.e.

Conditions (C2) and (C3) ensure that (10) can be stably minimized with first order methods.

We next discuss choices of the surrogate losses that ensure we satisfy condition (C1).

The classification surrogate loss is relatively easy to design.

There are many widely used loss functions from which we can choose, and in this work we choose the cross-entropy loss as the classification loss:

For the robustness surrogate loss, we choose the hinge loss on the certified radius:

where˜ > 0 and λ ≥ 1 .

We use the hinge loss because not only does it satisfy the surrogate condition, but also it is numerically stable, which we will discuss in Section 4.4.

The classification surrogate loss in (12) is differentiable with respect to θ, but the differentiability of the robustness surrogate loss in (13) requires differentiability of CR(g θ ; x, y).

In this section we will show that the randomized smoothing certified radius in (8) does not meet condition (C2), and accordingly, we will introduce soft randomized smoothing to solve this problem.

Whether the certified radius (8) is sub-differentiable with respect to θ boils down to the differentiablity of E η 1 {f θ (x+η)=y} .

Theoretically, the expectation is indeed differentiable.

However, from a practical point of view, the expectation needs to be estimated by Monte Carlo sampling

Gaussian noise and k is the number of samples.

This estimation, which is a sum of indicator functions, is not differentiable.

Hence, condition (C2) is still not met from the algorithmic perspective.

To tackle this problem, we leverage soft randomized smoothing (Soft-RS).

In contrast to the original version of randomized smoothing (Hard-RS), Soft-RS is applied to a neural network z θ (x) whose last layer is softmax.

The soft smoothed classifierg θ is defined as follows.

Definition 2.

For a neural network z θ : X → P(K) whose last layer is softmax and σ > 0, the soft smoothed classifierg θ of z θ is defined as

Using Lemma 2 in Salman et al. (2019), we prove the following theorem in Appendix A: Theorem 2.

Let the ground truth of an input x be y. Ifg θ classifies x correctly, i.e.

Theng θ is provably robust at x, with the certified radius given by

where Φ is the c.d.f.

of the standard Gaussian distribution.

We notice that in Salman et al. (2019) (see its Appendix B), a similar technique was introduced to overcome the non-differentiability in creating adversarial examples to a smoothed classifier.

Different from their work, our method uses Soft-RS to obtain a certified radius that is differentiable in practice.

The certified radius given by soft randomized smoothing meets condition (C2) in the algorithmic design.

Even if we use Monte Carlo sampling to estimate the expectation, (16) is still sub-differentiable with respect to θ as long as z θ is sub-differentiable with respect to θ.

Connection between Soft-RS and Hard-RS We highlight two main properties of Soft-RS.

Firstly, it is a differentiable approximation of the original Hard-RS.

To see this, note that when

a.e.

− − → 1 {y=arg max c u c θ (x)} , sog θ converges to g θ almost everywhere.

Consequently, the Soft-RS certified radius (16) converges to the Hard-RS certified radius (8) almost everywhere as β goes to infinity.

Secondly, Soft-RS itself provides an alternative way to get a provable robustness guarantee.

In Appendix A, we will provide Soft-RS certification procedures that certifỹ g θ with the Hoeffding bound or the empirical Bernstein bound.

In this section, we will address the numerical stability condition (C3).

While Soft-RS does provide us with a differentiable certified radius (16) which we could maximize with first-order optimization methods, directly optimizing (16) suffers from exploding gradients.

The problem stems from the inverse cumulative density function Φ −1 (x), whose derivative is huge when x is close to 0 or 1.

Fortunately, by minimizing the robustness loss (13) instead, we can maximize the robust radius free from exploding gradients.

The hinge loss restricts that samples with non-zero robustness loss must satisfy 0 < CR(g θ ; x, y) < +˜ , which is equivalent to 0 < ξ θ (x, y) < γ where

.

Under this restriction, the derivative of Φ −1 is always bounded as shown in the following proposition.

The proof can be found in Appendix B.

, γ} with respect to p 1 and p 2 is bounded.

We are now ready to present the complete MACER algorithm.

Expectations over Gaussian samples are approximated with Monte Carlo sampling.

Let

).

During training we minimize E (x,y)∼p data l(g θ ; x, y).

Detailed implementation is described in Algorithm 1.

To simplify the implementation, we choose γ to be a hyperparameter instead of˜ .

The inverse temperature of softmax β is also a hyperparameter.

For each

Compute the empirical expectations:

Update θ with one step of any first-order optimization method to minimize

10: end for

Compare to adversarial training Adversarial training defines the problem as a mini-max game and solves it by optimizing the inner loop (attack generation) and the outer loop (model update) iteratively.

In our method, we only have a single loop (model update).

As a result, our proposed algorithm can run much faster than adversarial training because it does not require additional back propagations to generate adversarial examples.

Compare to previous work The overall objective function of our method, a linear combination of a classification loss and a robustness loss, is similar to those of adversarial logit pairing (ALP) (Kannan et al., 2018) and TRADES (Zhang et al., 2019b) .

In MACER, the λ in the objective function (17) can also be viewed as a trade-off factor between accuracy and robustness.

However, the robustness term of MACER does not depend on a particular adversarial example x , which makes it substantially different from ALP and TRADES.

In this section, we empirically evaluate our proposed MACER algorithm on a wide range of tasks.

We also study the influence of different hyperparameters in MACER on the final model performance.

To fairly compare with previous works, we follow Cohen et al. (2019) and Salman et al. (2019) to use LeNet for MNIST, ResNet-110 for Cifar-10 and SVHN, and ResNet-50 for ImageNet.

MACER Training For Cifar-10, MNIST and SVHN, we train the models for 440 epochs using our proposed algorithm.

The learning rate is initialized to be 0.01, and is decayed by 0.1 at the 200 th /400 th epoch.

For all the models, we use k = 16, γ = 8.0 and β = 16.0.

The value of λ trades off the accuracy and robustness and we find that different λ leads to different robust accuracy when the model is injected by different levels (σ) of noise.

We find setting λ = 12.0 for σ = 0.25 and λ = 4.0 for σ = 0.50 works best.

For ImageNet, we train the models for 120 epochs.

The initial learning rate is set to be 0.1 and is decayed by 0.1 at the 30 th /60 th /90 th epoch.

For all models on ImageNet, we use k = 2, γ = 8.0 and β = 16.0.

More details can be found in Appendix C.

Baselines We compare the performance of MACER with two previous works.

The first work (Cohen et al., 2019) trains smoothed networks by simply minimizing cross-entropy loss.

The second one (Salman et al., 2019) uses adversarial training on smoothed networks to improve the robustness.

For both baselines, we use checkpoints provided by the authors and report their original numbers whenever available.

In addition, we run Cohen et al. (2019) 's method on all tasks as it is a speical case of MACER by setting k = 1 and λ = 0.

Certification Following previous works, we report the approximated certified test set accuracy, which is the fraction of the test set that can be certified to be robust at radius r. However, the approximated certified test set accuracy is a function of the radius r.

It is hard to compare two models unless one is uniformly better than the other for all r. Hence, we also use the average certified radius (ACR) as a metric: for each test data (x, y) and model g, we can estimate the certified radius CR(g; x, y).

The average certified radius is defined as 1 |Stest| (x,y)∈Stest CR(g; x, y) where S test is the test set.

To estimate the certified radius for data points, we use the source code provided by Cohen et al. (2019) .

We report the results on Cifar-10 and ImageNet in the main body of the paper.

Results on MNIST and SVHN can be found in Appendix C.2.

Performance The performance of different models on Cifar-10 are reported in Table 1 , and in Figure 1 we display the radius-accuracy curves.

Note that the area under a radius-accuracy curve is equal to the ACR of the model.

First, the plots show that our proposed method consistently achieves significantly higher approximated certified test set accuracy than Cohen et al. (2019) .

This shows that robust training via maximizing the certified radius is more effective than simply minimizing the cross entropy classification loss.

Second, the performance of our model is different from that of Salman et al. The gain of our model is relatively smaller when σ = 1.0.

This is because σ = 1.0 is a very large noise level (Cohen et al., 2019) and both models perform poorly.

The ImageNet results are displayed in Table 2 and Figure 2 , and the observation is similar.

All experimental results show that our proposed algorithm is more effective than previous ones.

Training speed Since MACER does not require adversarial attack during training, it runs much faster to learn a robust model.

Empirically, we compare MACER with Salman et al. (2019) on the average training time per epoch and the total training hours, and list the statistics in Table 3 .

For a fair comparison, we use the codes 34 provided by the original authors and run all algorithms on the same machine.

For Cifar-10 we use one NVIDIA P100 GPU and for ImageNet we use four NVIDIA P100 GPUs.

According to our experiments, on ImageNet, MACER achieves ACR=0.544 in 117.90 hours.

On the contrary, Salman et al. (2019) only achieves ACR=0.528 but uses 193.10 hours, which clearly shows that our method is much more efficient.

One might question whether the higher performance of MACER comes from the fact that we train for more epochs than previous methods.

In Section C.3 we also run MACER for 150 epochs and compare it with the models in Table 3 .

The results show that when run for only 150 epochs, MACER still achieves a performance comparable with SmoothAdv, and is 4 times faster at the same time.

In this section, we carefully examine the effect of different hyperparameters in MACER.

All experiments are run on Cifar-10 with σ = 0.25 or 0.50.

The results for σ = 0.25 are shown in Figure 3 .

All details can be found in Appendix C.4.

Effect of k We sample k Gaussian samples for each input to estimate the expectation in (16).

We can see from Figure 3 (a) that using more Gaussian samples usually leads to better performance.

For example, the radius-accuracy curve of k = 16 is uniformly above that of k = 1.

Effect of λ The radius-accuracy curves in Figure 3 (b) demonstrate the trade-off effect of λ.

From the figure, we can see that as λ increases, the clean accuracy drops while the certified accuracy at large radii increases.

Effect of γ γ is defined as the hyperparameter in the hinge loss.

From Figure 3 (c) we can see that when γ is small, the approximated certified test set accuracy at large radii is small since γ "truncates" the large radii.

As γ increases, the robust accuracy improves.

It appears that γ also acts as a trade-off between accuracy and robustness, but the effect is not as significant as the effect of λ.

Effect of β Similar to Salman et al. (2019)'s finding (see its Appendix B), we also observe that using a larger β produces better results.

While Salman et al. (2019) pointed out that a large β may make training unstable, we find that if we only apply a large β to the robustness loss, we can maintain training stability and achieve a larger average certified radius as well.

In this work we propose MACER, an attack-free and scalable robust training method via directly maximizing the certified radius of a smoothed classifier.

We discuss the desiderata such an algorithm would have to satisfy, and provide an approach to each of them.

According to our extensive experiments, MACER performs better than previous provable l 2 -defenses and trains faster.

Our strong empirical results suggest that adversarial training is not a must for robust training, and defense based on certification is a promising direction for future research.

Moreover, several recent papers (Carmon et al., 2019; Zhai et al., 2019; suggest that using unlabeled data helps improve adversarially robust generalization.

We will also extend MACER to the semisupervised setting.

In this section we provide theoretical analysis and certification procedures for Soft-RS.

Our proof is based on the following lemma:

Proof of Theorem 2.

Let y

Because z c θ :

Meanwhile, z B ≤ 1 − z A , so we can take z B = 1 − z A , and

It reduces to find a confidence lower bound of z A .

Here we provide two bounds:

Hoeffding Bound The random variable z

.

By Hoeffding's inequality we have

Hence, a 1 − α confidence lower bound z A of z A is

where S 2 is the sample variance of X 1 , · · · , X k , i.e.

Consequently, a 1 − α confidence lower bound z A of z A is

The full certification procedure with the above two bounds is described in Algorithm 2.

Algorithm 2 Soft randomized smoothing certification 1: # Certify the robustness ofg around an input x with Hoeffding bound 2: function CERTIFYHOEFFDING(z, σ 2 , x, n 0 , n, α)

if z A > A ← SAMPLEUNDERNOISE(z, x, n, σ 2 )

23:

for j = 1 to num do 24:

Sample noise η j ∼ N (0, σ 2 I)

Compute: z j = z(x + η j ) Figure 4 .

The results show that Hard-RS consistently gives a larger lower bound of robust radius than Soft-RS.

We also observe that there is a gap between Soft-RS and Hard-RS when β → ∞, which implies that the empirical Bernstein bound, though tighter than the Hoeffding bound, is still looser than the Clopper-Pearson bound.

Proof of Proposition 1.

We only need to consider the case when

is a strictly increasing function of p, p * is unique, and

Since p 1 is the largest value and p 1 + p 2 + ...

is continuous in any closed interval of (0, 1), the derivative of Φ −1 (p 1 ) − Φ −1 (p 2 ) with respect to p 1 is bounded.

Similarly, p 2 is the largest among p 2 , ...

, and the derivative of Φ −1 (p 1 ) − Φ −1 (p 2 ) with respect to p 2 is bounded.

In this section we list all compared models in the main body of this paper.

Cifar-10 models are listed in Table 4 , and ImageNet models are listed in Table 5 .

1.00 Salman-1.00 2-sample 10-step SmoothAdv P GD with = 2.00 MACER-1.00 MACER with k = 16, dynamic λ 5 , β = 16.0 and γ = 8.0

The results are reported in Table 6 .

For all σ, we use k = 16, λ = 16.0, γ = 8.0 and β = 16.0.

Table 7 .

We use k = 16, λ = 12.0, γ = 8.0 and β = 16.0.

Table 8 we report the performance and training time of MACER on Cifar-10 when it is only run for 150 epochs, and compare with SmoothAdv (Salman et al., 2019) and MACER (440 epochs).

The learning rate is decayed by 0.1 at epochs 60 and 120.

All other hyperparameters are kept the same as in Table 4 .

Table 9 for detailed experimental settings.

Results are reported in Tables 10-13.

<|TLDR|>

@highlight

We propose MACER: a provable defense algorithm that trains robust models by maximizing the certified radius. It does not use adversarial training but performs better than all existing provable l2-defenses.
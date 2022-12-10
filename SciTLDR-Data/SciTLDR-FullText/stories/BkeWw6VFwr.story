It is well-known that  classifiers are vulnerable to adversarial perturbations.

To defend against adversarial perturbations, various certified robustness results have been derived.

However, existing certified robustnesses are limited to top-1 predictions.

In many real-world applications, top-$k$ predictions are more relevant.

In this work, we aim to derive certified robustness for top-$k$ predictions.

In particular, our certified robustness is based on randomized smoothing, which turns any classifier to a new classifier via adding noise to an input example.

We adopt randomized smoothing because it is scalable to large-scale neural networks and applicable to any classifier.

We derive a tight robustness in $\ell_2$ norm for top-$k$ predictions  when using randomized smoothing with Gaussian noise.

We find that generalizing the certified robustness  from top-1 to top-$k$ predictions faces significant technical challenges.

We also empirically evaluate our method on CIFAR10 and ImageNet.

For example, our method can obtain an ImageNet classifier with a certified top-5 accuracy of 62.8\% when the $\ell_2$-norms of the adversarial perturbations are less than 0.5 (=127/255).

Our code is publicly available at: \url{https://github.com/jjy1994/Certify_Topk}.

Classifiers are vulnerable to adversarial perturbations (Szegedy et al., 2014; Goodfellow et al., 2015; Carlini & Wagner, 2017b; Jia & Gong, 2018) .

Specifically, given an example x and a classifier f , an attacker can carefully craft a perturbation δ such that f makes predictions for x + δ as the attacker desires.

Various empirical defenses (e.g., Goodfellow et al. (2015) ; Svoboda et al. (2019) ; Buckman et al. (2018) ; Ma et al. (2018) ; Guo et al. (2018) ; Dhillon et al. (2018) ; Xie et al. (2018) ; Song et al. (2018) ) have been proposed to defend against adversarial perturbations.

However, these empirical defenses were often soon broken by adaptive adversaries (Carlini & Wagner, 2017a; .

As a response, certified robustness (e.g., Wong & Kolter (2018) ; Raghunathan et al. (2018a) ; Liu et al. (2018) ; Lecuyer et al. (2019) ; Cohen et al. (2019) ) against adversarial perturbations has been developed.

In particular, a robust classifier verifiably predicts the same top-1 label for data points in a certain region around any example x.

In many applications such as recommender systems, web search, and image classification cloud service (Clarifai; Google Cloud Vision), top-k predictions are more relevant.

In particular, given an example, a set of k most likely labels are predicted for the example.

However, existing certified robustness results are limited to top-1 predictions, leaving top-k robustness unexplored.

To bridge this gap, we study certified robustness for top-k predictions in this work.

Our certified top-k robustness leverages randomized smoothing (Cao & Gong, 2017; Cohen et al., 2019) , which turns any base classifier f to be a robust classifier via adding random noise to an example.

For instance, Cao & Gong (2017) is the first to propose randomized smoothing with uniform noise as an empirical defense.

We consider random Gaussian noise because of its certified robustness guarantee (Cohen et al., 2019) .

Specifically, we denote by p i the probability that the base classifier f predicts label i for the Gaussian random variable N (x, σ 2 I).

The smoothed classifier g k (x) predicts the k labels with the largest probabilities p i 's for the example x.

We adopt randomized smoothing because it is scalable to large-scale neural networks and applicable to any base classifier.

Our major theoretical result is a tight certified robustness bound for top-k predictions when using randomized smoothing with Gaussian noise.

Specifically, given an example x, a label l is verifiably among the top-k labels predicted by the smoothed classifier g k (x + δ) when the 2 -norm of the adversarial perturbation δ is less than a threshold (called certified radius).

The certified radius for top-1 predictions derived by Cohen et al. (2019) is a special case of our certified radius when k = 1.

As our results and proofs show, generalizing certified robustness from top-1 to top-k predictions faces significant new challenges and requires new techniques.

Our certified radius is the unique solution to an equation, which depends on σ, p l , and the k largest probabilities p i 's (excluding p l ).

However, computing our certified radius in practice faces two challenges: 1) it is hard to exactly compute the probability p l and the k largest probabilities p i 's, and 2) the equation about the certified radius does not have an analytical solution.

To address the first challenge, we estimate simultaneous confidence intervals of the label probabilities via the Clopper-Pearson method and Bonferroni correction in statistics.

To address the second challenge, we propose an algorithm to solve the equation to obtain a lower bound of the certified radius, where the lower bound can be tuned to be arbitrarily close to the true certified radius.

We evaluate our method on CIFAR10 (Krizhevsky & Hinton, 2009) and ImageNet (Deng et al., 2009) datasets.

For instance, on ImageNet, our method respectively achieves approximate certified top-1, top-3, and top-5 accuracies as 46.6%, 57.8%, and 62.8% when the 2 -norms of the adversarial perturbations are less than 0.5 (127/255) and σ = 0.5.

Our contributions are summarized as follows:

• Theory.

We derive the first certified radius for top-k predictions.

Moreover, we prove our certified radius is tight for randomized smoothing with Gaussian noise.

• Algorithm.

We develop algorithms to estimate our certified radius in practice.

• Evaluation.

We empirically evaluate our method on CIFAR10 and ImageNet.

Suppose we have a base classifier f , which maps an example x ∈ R d to one of c candidate labels {1, 2, · · · , c}. f can be any classifier.

Randomized smoothing (Cohen et al., 2019) adds an isotropic Gaussian noise N (0, σ 2 I) to an example x. We denote p i as the probability that the base classifier f predicts label i when adding a random isotropic Gaussian noise to the example x, i.e., p i = Pr(f (x + ) = i), where ∼ N (0, σ 2 I).

The smoothed classifier g k (x) returns the set of k labels with the largest probabilities p i 's when taking an example x as input.

Our goal is to derive a certified radius R l such that we have l ∈ g k (x + δ) for all ||δ|| 2 < R l .

Our main theoretical results are summarized in the following two theorems.

Theorem 1 (Certified Radius for Top-k Predictions).

Suppose we are given an example x, an arbitrary base classifier f , ∼ N (0, σ 2 I), a smoothed classifier g, an arbitrary label l ∈ {1, 2, · · · , c}, and p l , p 1 , · · · , p l−1 , p l+1 , · · · , p c ∈ [0, 1] that satisfy the following conditions: Pr(f (x + ) = l) ≥ p l and Pr(f (x + ) = i) ≤ p i , ∀i = l,

where p and p indicate lower and upper bounds of p, respectively.

Let

where ties are broken uniformly at random.

Moreover, we denote by S t = {b 1 , b 2 , · · · , b t } the set of t labels with the smallest probability upper bounds in the k largest ones and by p St = t j=1 p bj the sum of the t probability upper bounds, where t = 1, 2, · · · , k. Then, we have:

where R l is the unique solution to the following equation:

where Φ and Φ −1 are the cumulative distribution function and its inverse of the standard Gaussian distribution, respectively.

Proof.

See Appendix A.

Algorithm 1: PREDICT Input: f , k, σ, x, n, and α.

Output: ABSTAIN or predicted top-k labels.

1 T = ∅ 2 counts = SAMPLEUNDERNOISE(f, σ, x, n) 3 c 1 , c 2 , · · · , c k+1 = top-{k + 1} indices in counts (ties are broken uniformly at random)

return ABSTAIN 10 return T Theorem 2 (Tightness of the Certified Radius).

Assuming we have p l + k j=1 p bj ≤ 1 and p l + i=1,··· ,l−1,l+1,··· ,c p i ≥ 1.

Then, for any perturbation ||δ|| 2 > R l , there exists a base classifier f * consistent with (1) but we have l / ∈ g k (x + δ).

Proof.

We show a proof sketch here.

Our detailed proof is in Appendix B. In our proof, we first show that, via mathematical induction and the intermediate value theorem, we can construct

Then, we construct a base classifier f * that predicts label i for an example if and only if the example is in the region C i , where i ∈ {1, 2, · · · , c}. As p l + k j=1 p bj ≤ 1 and p l + i=1,··· ,l−1,l+1,··· ,c p i ≥ 1, f * is well-defined.

Moreover, f * satisfies the conditions in (1).

Finally, we show that if ||δ|| 2 > R l , then we have Pr(f

We have several observations about our theorems.

• Our certified radius is applicable to any base classifier f .

• According to Equation 3, our certified radius R l depends on σ, p l , and the k largest probability upper bounds {p b k , p b k−1 , · · · , p b1 } excluding p l .

When the lower bound p l and the upper bounds {p b k , p b k−1 , · · · , p b1 } are tighter, the certified radius R l is larger.

When R l < 0, the label l is not among the top-k labels predicted by the smoothed classifier even if no perturbation is added, i.e., l / ∈ g k (x).

• When using randomized smoothing with Gaussian noise and no further assumptions are made on the base classifier, it is impossible to certify a 2 radius for top-k predictions that is larger than R l .

, where p b1 is an upper bound of the largest label probability excluding p l .

The certified radius derived by Cohen et al. (2019) for top-1 predictions (i.e., their Equation 3) is a special case of our certified radius with k = 1, l = A, and b 1 = B.

It is challenging to compute the top-k labels g k (x) predicted by the smoothed classifier, because it is challenging to compute the probabilities p i 's exactly.

To address the challenge, we resort to a Monte Carlo method that predicts the top-k labels with a probabilistic guarantee.

In particular, we leverage the hypothesis testing result from a recent work (Hung et al., 2019) .

Algorithm 1 shows our PREDICT function to estimate the top-k labels predicted by the smoothed classifier.

The function SAMPLEUNDERNOISE(f, σ, x, n) first randomly samples n noise 1 , 2 , · · · , n from the Gaussian distribution N (0, σ 2 I), uses the base classifier f to predict the label of x + j for each j ∈ {1, 2, · · · , n}, and returns the frequency of each label, i.e., counts[i] = n j=1 I(f (x + j ) = i) for i ∈ {1, 2, · · · , c}. The function BINOMPVALUE performs the hypothesis testing to calibrate the abstention threshold such that we can bound with probability α of returning an incorrect set of top-k labels.

Formally, we have the following proposition: Proposition 1.

With probability at least 1−α over the randomness in PREDICT, if PREDICT returns a set T (i.e., does not ABSTAIN), then we have g k (x) = T .

Proof.

See Appendix C.

Given a base classifier f , an example x, a label l, and the standard deviation σ of the Gaussian noise, we aim to compute the certified radius R l .

According to our Equation 3, our R l relies on a lower bound of p l , i.e., p l , and the upper bound of p St , i.e., p St , which are related to f , x, and σ.

We first discuss two Monte Carlo methods to estimate p l and p St with probabilistic guarantees.

However, given p l and p St , it is still challenging to exactly solve R l as the Equation 3 does not have an analytical solution.

To address the challenge, we design an algorithm to obtain a lower bound of R l via solving Equation 3 through binary search.

Our lower bound can be tuned to be arbitrarily close to R l .

Our approach has two steps.

The first step is to estimate p l and p i for i = l. The second step is to estimate p St using p i for i =

l.

Estimating p l and p i for i = l: The probabilities p 1 , p 2 , · · · , p c can be viewed as a multinomial distribution over the labels {1, 2, · · · , c}. If we sample a Gaussian noise uniformly at random, then the label f (x + ) can be viewed as a sample from the multinomial distribution.

Therefore, estimating p l and p i for i = l is essentially a one-sided simultaneous confidence interval estimation problem.

In particular, we aim to estimate these bounds with a confidence level at least 1 − α.

In statistics, Goodman (1965) ; Sison & Glaz (1995) are well-known methods for simultaneous confidence interval estimations.

However, these methods are insufficient for our problem.

Specifically, Goodman's method is based on Chi-square test, which requires the expected count for each label to be no less than 5.

We found that this is usually not satisfied, e.g., ImageNet has 1,000 labels, some of which have close-to-zero probabilities and do not have more than 5 counts even if we sample a large number of Gaussian noise.

Sison & Glaz's method guarantees a confidence level of approximately 1 − α, which means that the confidence level could be (slightly) smaller than 1 − α.

However, we aim to achieve a confidence level of at least 1 − α.

To address these challenges, we discuss two confidence interval estimation methods as follows: 1) BinoCP.

This method estimates p l using the standard one-sided Clopper-Pearson method and treats p i as p i = 1 − p l for each i = l. Specifically, we sample n random noise from N (0, σI 2 ), i.e., 1 , 2 , · · · , n .

We denote the count for the label l as n l = n j=1 I(f (x + j ) = l).

n l follows a binomial distribution with parameters n and p l , i.e., n l ∼ Bin(n, p l ).

Therefore, according to the Clopper-Pearson method, we have:

where 1 − α is the confidence level and B(α; u, v) is the αth quantile of the Beta distribution with shape parameters u and v. We note that the Clopper-Pearson method was also adopted by Cohen et al. (2019) to estimate label probability for their certified radius of top-1 predictions.

2) SimuEM.

The above method estimates p i as 1 − p l , which may be conservative.

A conservative estimation makes the certified radius smaller than what it should be.

Therefore, we introduce SimuEM to directly estimate p i together with p l .

We let n i = n j=1 I(f (x + j ) = i) for each i ∈ {1, 2, · · · , c}. Each n i follows a binomial distribution with parameters n and p i .

We first use Algorithm 2: CERTIFY Input: f , k, σ, x, l, n, µ, and α.

return ABSTAIN the Clopper-Pearson method to estimate a one-sided confidence interval for each label i, and then we obtain simultaneous confidence intervals by leveraging the Bonferroni correction.

Specifically, if we can obtain a confidence interval with confidence level at least 1 − α c for each label i, then Bonferroni correction tells us that the overall confidence level for the simultaneous confidence intervals is at least 1 − α, i.e., we have confidence level at least 1 − α that all confidence intervals hold at the same time.

Formally, we have the following bounds by applying the Clopper-Pearson method with confidence level 1 − α c to each label:

Estimating p St : One natural method is to estimate p St = t j=1 p bj .

However, this bound may be loose.

For example, when using BinoCP to estimate the probability bounds, we have p St = t · (1 − p l ), which may be bigger than 1.

To address the challenge, we derive another bound for p St from another perspective.

Specifically, we have p St ≤ i =l p i ≤ 1 − p l .

Therefore, we can use 1 − p l as an upper bound of p St , i.e., p St = 1 − p l .

Finally, we combine the above two estimations by taking the minimal one, i.e., p St = min(

It is challenging to compute the certified radius R l exactly because Equation 3 does not have an analytical solution.

To address the challenge, we design a method to estimate a lower bound of R l that can be tuned to be arbitrarily close to R l .

Specifically, we first approximately solve the following equation for each t ∈ {1, 2, · · · , k}:

We note that it is still difficult to obtain an analytical solution to Equation 7 when t > 1.

However, we notice that the left-hand side has the following properties: 1) it decreases as R t l increases; 2) when R t l → −∞, it is greater than 0; 3) when R t l → ∞, it is smaller than 0.

Therefore, there exists a unique solution R t l to Equation 7.

Moreover, we leverage binary search to find a lower bound R l t that can be arbitrarily close to the exact solution R t l .

In particular, we run the binary search until the left-hand side of Equation 7 is non-negative and the width of the search interval is less than a parameter µ > 0.

Formally, we have:

Figure 1: Impact of k on the certified top-k accuracy.

After obtaining R l t , we let R l = max

and Equation 8, we have the following guarantee:

Algorithm 2 shows our algorithm to estimate the certified radius for a given example x and a label l. The function SAMPLEUNDERNOISE is the same as in Algorithm 1.

Functions BINOCP and SIMUEM return the estimated probability bound for each label.

Function BINARYSEARCH performs binary search to solve the Equation 7 and returns a solution satisfying Equation 8.

Formally, our algorithm has the following guarantee:

Proposition 2.

With probability at least 1 − α over the randomness in CERTIFY, if CERTIFY returns a radius R l (i.e., does not ABSTAIN), then we have l ∈ g k (x + δ), ∀||δ|| 2 < R l .

Proof.

See Appendix D.

We conduct experiments on the standard CIFAR10 (Krizhevsky & Hinton, 2009) and ImageNet (Deng et al., 2009 ) datasets to evaluate our method.

We use the publicly available pre-trained models from Cohen et al. (2019) .

Specifically, the architectures of the base classifiers are ResNet-110 and ResNet-50 for CIFAR10 and ImageNet, respectively.

Parameter setting: We study the impact of k, the confidence level 1 − α, the noise level σ, the number of samples n, and the confidence interval estimation methods on the certified radius.

Unless otherwise mentioned, we use the following default parameters: k = 3, α = 0.001, σ = 0.5, n = 100, 000, and µ = 10 −5 .

Moreover, we use SimuEM to estimate bounds of label probabilities.

When studying the impact of one parameter on the certified radius, we fix the other parameters to their default values.

Approximate certified top-k accuracy:

For each testing example x whose true label is l, we compute the certified radius R l using the CERTIFY algorithm.

Then, we compute the certified topk accuracy at a radius r as the fraction of testing examples whose certified radius are at least r. Note that our computed certified top-k accuracy is an approximate certified top-k accuracy instead of the true certified top-k accuracy.

However, we can obtain a lower bound of the true certified top-k accuracy based on the approximate certified top-k accuracy.

Appendix E shows the details.

Moreover, the gap between the lower bound of the true certified top-k accuracy and the approximate top-k accuracy is negligible when α is small.

For convenience, we simply use the term certified top-k accuracy in the paper.

Figure 1 shows the certified top-k accuracy as the radius r increases for different k. Naturally, the certified top-k accuracy increases as k increases.

On CIFAR10, we respectively achieve certified top-1, top-2, and top-3 accuracies as 45.2%, 58.8%, and 67.2% when the 2 -norm of the adversarial perturbation is less than 0.5 (127/255).

On ImageNet, we respectively achieve certified top-1, top-3, and top-5 accuracies as 46.6%, 57.8%, and 62.8% when the 2 -norm of the adversarial perturbation is less than 0.5.

On CIFAR10, the gaps between the certified top-k accuracy for different k are smaller than those between the top-k accuracy under no attacks, and they become smaller as the radius increases.

On ImageNet, the gaps between the certified top-k accuracy for different k remain similar to those between the top-k accuracy under no attacks as the radius increases.

Figure 2 shows the influence of the confidence level.

We observe that confidence level has a small influence on the certified top-k accuracy as the different curves almost overlap.

The reason is that the estimated confidence intervals of the probabilities shrink slowly as the confidence level increases.

Figure 3 shows the influence of σ.

We observe that σ controls a trade-off between normal accuracy under no attacks and robustness.

Specifically, when σ is smaller, the accuracy under no attacks (i.e., the accuracy when radius is 0) is larger, but the certified top-k accuracy drops more quickly as the radius increases.

Figure 4 compares BinoCP with SimuEM.

The results show that SimuEM is better when the certified radius is small, while BinoCP is better when the certified radius is large.

We found the reason is that when the certified radius is large, p l is relatively large, and thus 1 − p l already provides a good estimation for p i , where i = l.

Numerous defenses have been proposed against adversarial perturbations in the past several years.

These defenses either show robustness against existing attacks empirically, or prove the robustness against arbitrary bounded-perturbations (known as certified defenses).

The community has proposed many empirical defenses.

The most effective empirical defense is adversarial training (Goodfellow et al., 2015; Kurakin et al., 2017; Tramèr et al., 2018; Madry et al., 2018) .

However, adversarial training does not have certified robustness guarantees.

Other examples of empirical defenses include defensive distillation (Papernot et al., 2016) , MagNet (Meng & Chen, 2017) , PixelDefend (Song et al., 2017) , Feature squeezing (Xu et al., 2018) , and many others (Liu et al., 2019; Svoboda et al., 2019; Schott et al., 2019; Buckman et al., 2018; Ma et al., 2018; Guo et al., 2018; Dhillon et al., 2018; Xie et al., 2018; Song et al., 2018; Samangouei et al., 2018; Na et al., 2018; Metzen et al., 2017) .

However, many of these defenses were soon broken by adaptive attacks (Carlini & Wagner, 2017a; Uesato et al., 2018; .

To end the arms race between defenders and adversaries, researchers have developed certified defenses against adversarial perturbations.

Specifically, in a certifiably robust classifier, the predicted top-1 label is verifiably constant within a certain region (e.g., 2 -norm ball) around an input example, which provides a lower bound of the adversarial perturbation.

Such certified defenses include satisfiability modulo theories based methods (Katz et al., 2017; Carlini et al., 2017; Ehlers, 2017; Huang et al., 2017) , mixed integer linear programming based methods (Cheng et al., 2017; Lomuscio & Maganti, 2017; Dutta et al., 2017; Fischetti & Jo, 2018; Bunel et al., 2018) , abstract interpretation based methods (Gehr et al., 2018; Tjeng et al., 2018) , and global (or local) Lipschitz constant based methods (Cisse et al., 2017; Gouk et al., 2018; Tsuzuku et al., 2018; Anil et al., 2019; Wong & Kolter, 2018; Wang et al., 2018a; b; Raghunathan et al., 2018a; b; Wong et al., 2018; Dvijotham et al., 2018a; b; Croce et al., 2018; Gehr et al., 2018; Mirman et al., 2018; Singh et al., 2018; Gowal et al., 2018; Weng et al., 2018; Zhang et al., 2018) .

However, these methods are not scalable to large neural networks and/or make assumptions on the architectures of the neural networks.

For example, these defenses are not scalable/applicable to the complex neural networks for ImageNet.

Randomized smoothing was first proposed as an empirical defense (Cao & Gong, 2017; Liu et al., 2018) (2019) employed adversarial training to improve the performance of randomized smoothing.

Unlike the other certified defenses, randomized smoothing is scalable to large neural networks and applicable to arbitrary classifiers.

Our work derives the first certified robustness guarantee of randomized smoothing for top-k predictions.

Moreover, we show that our robustness guarantee is tight for randomized smoothing with Gaussian noise.

Adversarial perturbation poses a fundamental security threat to classifiers.

Existing certified defenses focus on top-1 predictions, leaving top-k predictions untouched.

In this work, we derive the first certified radius under 2 -norm for top-k predictions.

Our results are based on randomized smoothing.

Moreover, we prove that our certified radius is tight for randomized smoothing with Gaussian noise.

In order to compute the certified radius in practice, we further propose simultaneous confidence interval estimation methods as well as design an algorithm to estimate a lower bound of the certified radius.

Interesting directions for future work include 1) deriving a tight certified radius under other norms such as 1 and ∞ , 2) studying which noise gives the tightest certified radius for randomized smoothing, and 3) studying certified robustness for top-k ranking.

A PROOF OF THEOREM 1

Given an example x, we define the following two random variables:

where ∼ N (0, σ 2 I).

The random variables X and Y represent random samples obtained by adding isotropic Gaussian noise to the example x and its perturbed version x + δ, respectively.

Moreover, we have the following lemma from Cohen et al. (2019) .

Lemma 2.

Given an example x, a number q ∈ [0, 1], and regions A and B defined as follows:

)

Then, we have the following equations:

Proof.

Please refer to Cohen et al. (2019) .

Based on Lemma 1 and 2, we derive the following lemma: Lemma 3.

Suppose we have an arbitrary base classifier f , an example x, a set of labels which are denoted as S, two probabilities p S and p S that satisfy p S ≤ p S = Pr(f (X) ∈ S) ≤ p S , and regions A S and B S defined as follows:

Proof.

We know that Pr(X ∈ A S ) = p S based on Lemma 2.

Combined with the condition that p S ≤ Pr(f (X) ∈ S), we obtain the first inequality in (20) .

Similarly, we can obtain the second inequality in (20).

We define M (z) = I(f (z) ∈ S).

Based on the first inequality in (20) and Lemma 1, we have the following:

which is the first inequality in (21).

The second inequality in (21) can be obtained similarly.

Next, we restate Theorem 1 and show our proof.

Theorem 1 (Certified Radius for Top-k Predictions).

Suppose we are given an example x, an arbitrary base classifier f , ∼ N (0, σ 2 I), a smoothed classifier g, an arbitrary label l ∈ {1, 2, · · · , c}, and p l , p 1 , · · · , p l−1 , p l+1 , · · · , p c ∈ [0, 1] that satisfy the following conditions:

where p and p indicate lower and upper bounds of p, respectively.

Let

where ties are broken uniformly at random.

Moreover, we denote by S t = {b 1 , b 2 , · · · , b t } the set of t labels with the smallest probability upper bounds in the k largest ones and by p St = t j=1 p bj the sum of the t probability upper bounds, where t = 1, 2, · · · , k. Then, we have:

where R l is the unique solution to the following equation:

where Φ and Φ −1 are the cumulative distribution function and its inverse of the standard Gaussian distribution, respectively.

Proof.

Roughly speaking, our idea is to make the probability that the base classifier f predicts l when taking Y as input larger than the smallest one among the probabilities that f predicts for a set of arbitrary k labels selected from all labels except l. For simplicity, we let Γ = {1, 2, · · · , c} \ {l}, i.e., all labels except l. We denote by Γ k a set of k labels in Γ. We aim to find a certified radius R l such that we have max Γ k ⊆Γ min i∈Γ k Pr(f (Y) = i) < Pr(f (Y) = l), which guarantees l ∈ g k (x + δ).

We first upper bound the minimal probability min i∈Γ k Pr(f (Y) = i) for a given Γ k , and then we upper bound the maximum value of the minimal probability among all possible Γ k ⊆ Γ. Finally, we obtain the certified radius R l via letting the upper bound of the maximum value smaller than Pr(f (Y) = l).

Bounding min i∈Γ k Pr(f (Y) = i) for a given Γ k : We use S to denote a non-empty subset of Γ k and use |S| to denote its size.

We define p S = i∈S p i , which is the sum of the upper bounds of the probabilities for the labels in S. Moreover, we define the following region associated with the set S:

We have Pr(f (Y) ∈ S) ≤ Pr(Y ∈ B S ) by applying Lemma 3 to the set S. In addition, we have

.

Therefore, we have:

Moreover, we have:

where we have the first inequality because S is a subset of Γ k and we have the second inequality because the smallest value in a set is no larger than the average value of the set.

Equation 27 holds for any S ⊆ Γ k .

Therefore, by taking all possible sets S into consideration, we have the following:

where S t is the set of t labels in Γ k whose probability upper bounds are the smallest, where ties are broken uniformly at random.

We have Equation 30 from Equation 29 because Pr(Y ∈ B S ) decreases as p S decreases.

Since Pr(Y ∈ B St ) increases as p St increases, Equation 30 reaches its maximum value when Γ k = {b 1 , b 2 , · · · , b k }, i.e., when Γ k is the set of k labels in Γ with the largest probability upper bounds.

Formally, we have:

where

Obtaining R l : According to Lemma 3, we have the following for S = {l}:

Recall that our goal is to make Pr(f (Y) = l) > max Γ k ⊆Γ min i∈Γ k Pr(f (Y) = i).

It suffices to let:

According to Lemma 2, we have Pr(

.

Therefore, we have the following constraint on δ:

Since the left-hand side of the above inequality 1) decreases as ||δ|| 2 increases, 2) is larger than 0 when ||δ|| 2 → −∞, and 3) is smaller than 0 when ||δ|| 2 → ∞, we have the constraint ||δ|| 2 < R l , where R l is the unique solution to the following equation:

B PROOF OF THEOREM 2

Following the terminology we used in proving Theorem 1, we define a region A {l} as follows:

According to Lemma 2, we have Pr(X ∈ A {l} ) = p l .

We first show the following lemma, which is the key to prove our Theorem 2.

Lemma 4.

Assuming we have p l + k j=1 p bj ≤ 1.

For any perturbation δ 2 > R l , there exists k disjoint regions C bj ⊆ R d \ A {l} , j ∈ {1, 2, · · · , k} that satisfy the following:

where the random variables X and Y are defined in Equation 10 and 11, respectively; and {b 1 , b 2 , · · · , b k } and S t are defined in Theorem 1.

Proof.

Our proof is based on mathematical induction and the intermediate value theorem.

For convenience, we defer the proof to Appendix B.1.

Next, we restate Theorem 2 and show our proof.

Theorem 2 (Tightness of the Certified Radius).

Assuming we have p l + k j=1 p bj ≤ 1 and p l + i=1,··· ,l−1,l+1,··· ,c p i ≥ 1.

Then, for any perturbation ||δ|| 2 > R l , there exists a base classifier f * consistent with (1) but we have l / ∈ g k (x + δ).

Proof.

Our idea is to construct a base classifier such that l is not among the top-k labels predicted by the smoothed classifier for any perturbed example x + δ when ||δ|| 2 > R l .

First, according to Lemma 4, we know there exists k disjoint regions C bj ⊆ R d \ A {l} , j ∈ {1, 2, · · · , k} that satisfy Equation 37 and 38.

Moreover, we divide the remaining region R d \ (A {l} ∪ k j=1 C bj ) into c−k −1 regions, which we denote as C b k+1 , C b k+2 , · · · , C bc−1 and satisfy Pr(X ∈ C bj ) ≤ p bj for j ∈ {k + 1, k + 2, · · · , c − 1}. Note that b 1 , b 2 , · · · , b c−1 is some permutation of {1, 2, · · · , c} \ {l}. We can divide the remaining region into such c−k −1 regions because p l + i=1,··· ,l−1,l+1,··· ,c p i ≥ 1.

Then, based on these regions, we construct the following base classifier:

Based on the definition of f * , we have the following:

Pr(f

Therefore, f * satisfies the conditions in (1).

Next, we show that l is not among the top-k labels predicted by the smoothed classifier for any perturbed example x + δ when ||δ|| 2 > R l .

Specifically, we have:

where j = 1, 2, · · · , k. Since we have found k labels whose probabilities are larger than the probability of the label l, we have l / ∈ g k (x + δ) when δ 2 > R l .

We first define some key notations and lemmas that will be used in our proof.

Definition 1 (C(q 1 , q 2 ), C (q 1 , q 2 ), r x (q 1 , q 2 ), r y (q 1 , q 2 )).

Given two values q 1 and q 2 that satisfy 0 ≤ q 1 < q 2 ≤ 1, we define the following region:

According to Lemma 2, we have:

where the Gaussian random variable X is defined in Equation 10 .

Moreover, assuming we have pairs of (q

We define the following region:

C (q 1 , q 2 ) is the remaining region of C(q 1 , q 2 ) excluding C(q

where the random variables X and Y are defined in Equation 10 and 11, respectively.

Next, we show a key property of our defined functions r x (q Proof.

We consider three scenarios.

We denote h x and h y as the probability densities for the random variables X and Y, respectively.

Then, we have h x (z) = (

and h y (z) = (

).

Therefore, the ratio of the probability density of Y and the probability density of X at a given point z is as follows:

Next, we compare the ratio for the points in different regions and have the following:

The Equation 57 from 56 is based on Equation 55 and the fact that δ

Combining the Equation 56, 57, and 60, we have the following:

Taking an integral on both sides of the Equation 61 in the region C (q

Similarly, we have:

Based on Equation 62, 63, and the condition that r x (q

, we have the following:

Therefore, we have the following equation:

Similar to Scenario I, we know that there exists u such that:

Similar to Scenario I, we have the following based on Equation 66:

Therefore, we have the following:

Next, we list the well-known Intermediate Value Theorem and show several other properties of our defined functions r x (q

Roughly speaking, the Intermediate Value Theorem tells us that if a continuous function has values no larger and no smaller (or no smaller and no larger) than v at the two end points of an interval, respectively, then the function takes value v at some point in the interval.

Lemma 7.

Given two probabilities q x , q y , if we have:

Then, there exists q 1 , q 2 ∈ [q 1 , q 2 ] such that:

Furthermore, if we have:

then there exists q 1 , q 2 ∈ [q 1 , q 2 ] such that:

.

Therefore, according to Lemma 6, there exists q 1 ∈ [q 1 , q 2 ] such that:

Similarly, we can prove that there exists q 2 ∈ [q 1 , q 2 ] such that r x (q 1 , q 2 ) = q x .

For any q e 2 ∈ [q 2 , q 2 ], we define H(x) = r x (x, q e 2 ).

Then, we know H(q 1 ) = r x (q 1 , q e 2 ) ≥ r x (q 1 , q 2 ) = q x since q e 2 ≥ q 2 .

Moreover, we have H(q e 2 ) = 0 ≤ q x .

Therefore, we have (H(q 1 ) − q x ) · (H(q e 2 ) − q x ) ≤ 0.

According to Lemma 6,we know there exists q e 1 ∈ [q 1 , q e 2 ] such that r x (q e 1 , q e 2 ) = q x for arbitrary q e 2 ∈ [q 2 , q 2 ].

We define G(x) = r y (q e 1 , x) where x ∈ [q 2 , q 2 ], and q e 1 are a value such that r x (q e 1 , x) = q x for a given x. When x = q 2 , we can let q e 1 = q 1 since r x (q 1 , q 2 ) = q x , and when x = q 1 , we can let q e 1 = q 2 since r x (q 2 , q 2 ) = q x .

Based on Equation 73 and Lemma 6, we know that there exists x ∈ [q 2 , q 2 ] such that G(x) = q y .

Therefore, there exists q 1 and q 2 such that:

Lemma 8.

Assuming we have q

1 , then we have the following:

2 ) since no region is excluded.

Therefore, we have r y (q

Proof.

By applying Lemma 5.

We further generalize Lemma 5 to two regions.

Specifically, we have the following lemma: Lemma 10.

Assuming we have a region C w ⊆ C(q

and r x (q 1 , q 2 ) ≤ Pr(X ∈ C w ), then we have:

Proof.

We let q 1 = max(q 1 , q

we can obtain the conclusion by applying Lemma 5 on C (q 1 , q 2 ) ∪ C w .

Next, we restate Lemma 4 and show our proof.

, 2, · · · , k} that satisfy the following:

where the random variables X and Y are defined in Equation 10 and 11, respectively; and {b 1 , b 2 , · · · , b k } and S t are defined in Theorem 1.

Proof.

Our proof leverages Mathematical Induction, which contains two steps.

In the first step, we show that the statement holds initially.

In the second step, we show that if the statement is true for the mth iteration, then it also holds for the (m+1)th iteration.

Without loss of generality, we assume τ = argmin

.

Therefore, we have the following:

Recall the definition of B Sτ and we have the following:

where p Sτ = j∈Sτ p j .

We can split B S k into two parts: B Sτ and B S k \ B Sτ .

We will show that ∀j ∈ [1, τ ], we can find disjoint C bj ⊆ B Sτ whose union is B Sτ such that:

For the other part, we will show that ∀j ∈ [τ + 1, k], we can find disjoint C bj ⊆ B S k \ B Sτ whose union is B S k \ B Sτ such that:

We first show that ∀j ∈ [1, τ ], we can find C bj ⊆ B Sτ that satisfy Equation 85 and 86.

Since our proof leverages Mathematical Induction, we iteratively construct each C bj , ∀j ∈ [1, τ ].

Specifically, we first show that we can find C bτ ⊆ B Sτ that satisfies the requirements.

Then, assuming we can find C bτ , · · · , C bτ−m+1 , we show that we can find C bτ−m ⊆ B Sτ \ (∪ τ j=τ −m+1 C bj ).

We will leverage Lemma 7 to prove the existence for each C bj .

Next, we show the two steps.

Step I: We show that we can find C bτ ⊆ B Sτ that satisfies Equation 85 and 86.

We let q 1 = 0 and q 2 = p Sτ , and we define the following region:

We have:

which can be directly obtained as C (q 1 , q 2 ) = B Sτ .

As we have

Moreover, we have the following:

The equality in the middle is from Lemma 8, the left inequality is because q 2 = p bτ ≥ p b1 , and the right inequality is from Equation 83.

Furthermore, we have the following:

We

Thus, there exists (q

based on Lemma 7.

Then, we have the following based on the definition of r x , r y :

Finally, we let C bτ = C (q 1 , q 2 ) ∩ C(q τ 1 , q τ 2 ), which meets our goal.

Step II:

Assuming we can find {(q

as well as the following:

We denote e = τ − m.

We show we can find C be such that we have:

We let q 1 = 0, q 2 = p Sτ and denote

We have the following:

The Equation 123 from 122 is because C(q 1 , q 2 ) = B Sτ and the Equation 107.

We have p be ≤ r x (q 1 , q 2 ).

Therefore, based on Lemma 7, there exist q 1 , q 2 such that:

We have:

Equation 128

In particular, we consider two scenarios.

and r x (q 1 , q 2 ) = p be ≤ Pr(X ∈ C w ) = p bt , we have the following based on Lemma 10:

Scenario 2).

q 1 < min τ j=τ −m+1 q j 1 .

We have the following:

Furthermore, we have:

Moreover, we have r x (q 1 , q 1 ) = q 1 − q 1 from Lemma 8.

The above two should be equal.

Thus, we have q 1 = e−1 j=1 p bj = p Sτ−m−1 since e = τ − m. we have:

We obtain Equation 140 from Equation 139 based on Lemma 8.

Therefore, we have the following in both scenarios:

Based on Lemma 7, there exist q .

Then, we have the following based on the definition of r x , r y :

We let C e = C (q 1 , q 2 ) ∩ C(q e 1 , q e 2 ).

From the definition of C (q 1 , q 2 ), we have ∀j ∈ [e + 1, τ ], C (q 1 , q 2 ) ∩ C bj = ∅. Thus, we have ∀j ∈ [e + 1, τ ], C be ∩ C bj = ∅ since C be ⊆ C (q 1 , q 2 ).

Therefore, we reach our goal by Mathematical Induction, i.e., for ∀j ∈ [1, τ ], we have:

We can also verify that ∪ τ j=1 C bj = B Sτ .

Next, we show our proof based on Mathematical Induction for the other part, i.e., B S k \ B Sτ .

Our construction process is similar to the above first part but has subtle differences.

Step I:

Let q 1 = τ j=1 p bj and q 2 = k j=1 p bj .

We define:

Then, we have:

The Equation 150 is based on the fact that C (q 1 , q 2 ) = C(q 1 , q 2 ) and Definition 1, and we obtain Equation 154 from 155 based on Equation 83.

We have p b k ≤ r x (q 1 , q 2 ).

Therefore, based on Lemma 7, we know that there exists

We consider two scenarios.

In this scenario, we consider r y (q 1 , q 2 ) > q 2 ) .

Then, we have:

Therefore, we have the following:

Scenario 2).

In this scenario, we consider r y (q 1 , q 2 ) ≤ Pr(Y∈B Sτ ) τ

.

We have the following:

We

Therefore, from Lemma 7, we know that there exist

We also have the following:

Similarly, we let Step II: We show that if we can find (q

Then, we can find (q

)

such that:

For simplicity, we denote e = k − m, we let q 1 = τ j=1 p bj and q 2 = k j=1 p bj , and we define:

Then, we have:

Published as a conference paper at ICLR 2020

We have p be ≤ r x (q 1 , q 2 ).

Therefore, based on Lemma 7, we know that there exists q 1 , q 2 such that:

Similarly, we consider two scenarios:

Scenario 1).

In this scenario, we consider that the following holds:

We let q e 1 = q 1 , q e 2 = q 2 , i.e., C be = C(q 1 , q 2 ) ∩ C (q 1 , q 2 ).

Then, we have:

We note that we have q 1 ≤ min k j=e+1 q i 1 in this scenario.

Otherwise, Equation 186 will not hold based on Lemma 10.

We give a short proof.

Note that in this case, we have q w 2 < q 2 because q w 2 = q 2 and q 1 > q w 1 cannot hold at the same time as long as r y (q 1 , q 2 ) > 0.

Thus, we have Pr(Y ∈ C w ) =

, we have q w 2 = q 2 .

As we have q 1 > q w 1 , q 2 > q w 2 and r x (q 1 , q 2 ) = p be ≤ Pr(X ∈ C w ) = p bw .

We have the following based on Lemma 10:

Since Equation

Therefore, we have r x (q 1 , q 1 ) = q 1 − q 1 from Definition 1.

Moreover, we have the following:

The above two should be equal.

Therefore, we have q 1 = e−1 j=τ +1 p bj + q 1 = e−1 j=1 p bj .

Recall that we let C be = C (q 1 , q 2 ) ∩ C(q 1 , q 2 ).

Thus, we have:

=Pr(Y ∈ C (q 1 , q 2 ) \ C be ) (196) =Pr(Y ∈ C(q 1 , q 1 )) (197) =Pr(Y ∈ C(0, q 1 )) − Pr(Y ∈ C(0, q 1 )) (198) =Pr(Y ∈ B Se−1 ) − Pr(Y ∈ B Sτ ) (199)

Scenario 2).

In this scenario, we consider that the following holds:

Note that we have:

≥r y (q 1 , q 2 ) · 1 r x (q 1 , q 2 )/r x (q 1 , q 2 ) (203)

We obtain

We let C be = C (q 1 , q 2 ) ∩ C(q e 1 , q e 2 ).

We also have the following:

=Pr(Y ∈ C (q 1 , q 2 ) \ C be ) (209) =r y (q 1 , q 2 ) − r y (q

Similar to

Step I, we still hold the conclusion that if Pr(Y ∈ C be ) > Pr(Y∈B Sτ ) τ , we have q e 2 = q 2 .

Then, we can apply Mathematical Induction to reach the conclusion.

Also, we can verify ∪ k j=τ +1 C bj = B S k \ B Sτ .

The function SAMPLEUNDERNOISE(f, k, σ, x, n, α) works as follows: we first draw n random noise from N (0, σ 2 I), i.e., 1 , 2 , · · · , n .

Then, we compute the values: ∀i ∈ [1, c], c i = n j=1 I(f (x + j ) = i).

The function BINOMPVALUE(n ct , n ct + n ct+1 , p) returns the result of p-value of the two-sided hypothesis test for n ct ∼ Bin(n ct + n ct+1 , p).

Proposition 1: With probability at least 1−α over the randomness in PREDICT, if PREDICT returns a set T (i.e., does not ABSTAIN), then we have g k (x) = T .

Proof.

We aim to compute the probability that PREDICT returns a set which not equals to g k (x), which happens if and only if g k (x) = T and PREDICT doesn't abstain.

Specifically, we have:

Pr(PREDICT returns a set = g k (x)) (213) =Pr(g k (x) = T, PREDICT doesn't abstain) (214) =Pr(g k (x) = T ) · Pr(PREDICT doesn't abstain|g k (x) = T ) (215) ≤Pr(PREDICT doesn't abstain|g k (x) = T )

Theorem 1 in Hung et al. (2019) shows the above conditional probability is as follows:

Pr(PREDICT doesn't abstain|g k (x) = T ) ≤ α

Therefore, we reach the conclusion.

Proposition 2: With probability at least 1−α over the randomness in CERTIFY, if CERTIFY returns a radius R l (i.e., does not ABSTAIN), then we have l ∈ g k (x + δ), ∀ δ 2 < R l .

Proof.

From the definition of BINOCP and SIMUEM, we know the probability that the following inequalities simultaneously hold is at least 1 − α over the sampling of counts:

Then, with the returned bounds, we can invoke Theorem 1 to obtain the robustness guarantee if the calculated radius is larger than 0.

Note that otherwise CERTIFY abstains.

We show how to derive a lower bound of the certified top-k accuracy based on the approximate certified top-k accuracy.

The process is similar to that Cohen et al. (2019) used to derive a lower bound of the certified top-1 accuracy based on the approximate certified top-1 accuracy.

Specifically, we have the following lemma from Cohen et al. (2019) .

Lemma 11.

Let z i be a binary variable and Y i be a Bernoulli random variable.

Suppose if z i = 1, then Pr(Y i = 1) ≤ α.

Then, for any ρ > 0, with probability at least 1 − ρ, we have the following:

Proof.

Please refer to Cohen et al. (2019) .

Assuming we have a test dataset D test = {(x 1 , y 1 ), (x 2 , y 2 ), · · · , (x m , y m )} as well as a radius r. We define the following indicate value:

a i = I(y i ∈ g k (x i + δ)), ∀||δ|| 2 < r

Then, the certified top-k accuracy of the smoothed classifier g at radius r can be computed as 1 m m i=1 a i .

For each sample x i , we run the CERTIFY function with 1 − α confidence level and we use a random variable Y i to denote that the function CERTIFY returns a radius bigger than r. From Proposition 2, we know:

The approximate certified top-k accuracy of the smoothed classifier at radius r is 1 m m i=1 Y i .

Then, we can use Lemma 11 to obtain a lower bound of 1 m m i=1 a i .

Specifically, for any ρ > 0, with probability at least 1 − ρ over the randomness of CERTIFY, we have:

We can see that the difference between the certified top-k accuracy and the approximate certified top-k accuracy is negligible when α is small.

@highlight

We study the certified robustness for top-k predictions via randomized smoothing under Gaussian noise and derive a tight robustness bound in L_2 norm.

@highlight

This paper extends work on deducing a certified radius using randomized smoothing, and shows the radius at which a smoothed classifier under Gaussian perturbations is certified for the top k predictions.

@highlight

This paper builds upon the random smoothing technique for top-1 prediction, and aims to provide certification on top-k predictions.
The point estimates of ReLU classification networks, arguably the most widely used neural network architecture, have recently been shown to have arbitrarily high confidence far away from the training data.

This architecture is thus not robust, e.g., against out-of-distribution data.

Approximate Bayesian posteriors on the weight space have been empirically demonstrated to improve predictive uncertainty in deep learning.

The theoretical analysis of such Bayesian approximations is limited, including for ReLU classification networks.

We present an analysis of approximate Gaussian posterior distributions on the weights of ReLU networks.

We show that even a simplistic (thus cheap), non-Bayesian Gaussian distribution fixes the asymptotic overconfidence issue.

Furthermore, when a Bayesian method, even if a simple one, is employed to obtain the Gaussian, the confidence becomes better calibrated.

This theoretical result motivates a range of Laplace approximations along a fidelity-cost trade-off.

We validate these findings empirically via experiments using common deep ReLU networks.

As neural networks have been successfully applied in ever more domains, including safety-critical ones, the robustness of their predictions and the calibration of their predictive uncertainty have moved into focus, subsumed under the notion of AI safety (Amodei et al., 2016) .

A principal goal of uncertainty calibration is that learning machines (and neural networks in particular) should assign low confidence to test cases not explained well by the training data or prior information (Gal, 2016) .

The most obvious such instance are test points that lie "far away" from the training data.

Many methods to achieve this goal have been proposed, both Bayesian (Gal & Ghahramani, 2016; Blundell et al., 2015; Louizos & Welling, 2017) and non-Bayesian (Lakshminarayanan et al., 2017; Liang et al., 2018; Hein et al., 2019) .

ReLU networks are currently among the most widely used neural architectures.

This class comprises any network that can be written as a composition of linear layers (including fully-connected, convolutional, and residual layers) and a ReLU activation function.

But while ReLU networks often achieve high accuracy, the uncertainty of their predictions has been shown to be miscalibrated (Guo et al., 2017) .

Indeed, Hein et al. (2019) demonstrated that ReLU networks are always overconfident "far away from the data": scaling a training point x (a vector in a Euclidean input space) with a scalar δ yields predictions of arbitrarily high confidence in the limit δ → ∞. This means ReLU networks are susceptible to adversarial or out-of-distribution (OOD) examples.

Bayesian methods have long been known empirically to improve predictive uncertainty calibration.

MacKay (1992) demonstrated empirically that the predictive uncertainty of Bayesian neural networks will naturally be high in regions not covered by training data.

Results like this raise the hope that the overconfidence problem of ReLU networks, too, might be mitigated by the use of Bayesian methods.

This paper offers a theoretical analysis of the binary classification case of ReLU networks with logistic output layer.

We show that equipping such networks with virtually any Gaussian probability distribution (i.e. regardless of whether it is motivated in a Bayesian fashion or not) mitigates the aforementioned theoretical problem, so that predictive confidence far away from the training data approaches a known constant, bounded away from one, whose value is controlled by the covariance (cf.

Figure 1) .

At the same time, this treatment does not change the decision boundary of the trained network, so it has no negative effect on the predictive performance.

Figure 1: Binary classification on a toy dataset using a MAP estimate (a) and various Gaussian approximations over the weights, sorted by their complexity of inverting the precision matrix.

These approximations are carried out only at the last layer of the network and d denotes the number of hidden units at that layer.

The shade of color represents the confidence of the prediction (darker shade means higher confidence).

The decision boundary is in thick black.

Even an arbitrary (i.e. nonBayesian) isotropic (b) or diagonal (c) covariance makes the confidence bounded away from one.

Using the data in a more Bayesian fashion (d) calibrates the uncertainty further, in particular in regions close to the data.

A central aspect of our result is that asymptotic overconfidence can be mitigated with an essentially arbitrary Gaussian distribution on the weight space, including one of simple diagonal or even scalar covariance, and one whose covariance need not even depend on the training data.

Achieving calibration at finite distances from the training data requires increasing levels of fidelity towards full Bayesian inference, for which our results also give some quantification.

Our results thus answer a question about "how Bayesian" one needs to be to achieve certain levels of calibration.

This is valuable because even approximate Bayesian treatments of deep learning, such as through Laplace approximations, can have high computational cost.

We empirically validate our results through a simple Laplace approximation to only the last layer of deep ReLU architectures, and find that this cheap procedure is already competitive to recently proposed non-Bayesian methods specifically constructed to overcome the overconfidence problem of ReLU networks.

We also show that this cheap Bayesian approach yields good performance in the multi-class classification setting, indicating that our analysis may carry over to this case.

Section 2 begins with a rigorous problem statement and assumptions, then develops the main theoretical results.

We discuss related work in Section 3, while empirical results are in Section 4.

Definitions We call a function f : R n → R piecewise affine if there exists a finite set of polytopes {Q r } R r=1 , referred to as linear regions of f , such that ∪ R r=1 Q r = R n and f | Qr is an affine function for every Q r .

ReLU networks are networks that result in piecewise affine classifier functions (Arora et al., 2018) which include networks with fully-connected, convolutional, and residual layers where just ReLU or leaky-ReLU are used as activation functions and max or average pooling are used as a convolution layer.

be a dataset, where the targets t i ∈ {0, 1} or t i ∈ {1, . . .

, k} for the binary and multi-class case, respectively.

We define the logistic (sigmoid) function as σ(z) := 1/(1 + exp(−z)) for z ∈ R and the softmax function as softmax(z, i) := exp(z i )/ j exp(z j ) for z ∈ R k .

Given a linear classifier, 1 we will consider probability distributions p(w|D) or p(W|D) over the weight vector and matrix, respectively.

We call these distributions posterior if they arose from Bayes' theorem or an approximation thereof.

The predictive distribution (also called the marginalized prediction) is

for the binary and multi-class cases, respectively.

For Euclidean spaces we use the standard inner product and norm.

Finally, λ i (·), λ max (·), and λ min (·) return the ith, maximum, and minimum eigenvalue (which are assumed to exist) of their matrix argument, respectively.

Problem statement The following theorem from Hein et al. (2019) shows that ReLU networks exhibit arbitrarily high confidence far away from the training data: If a training point x ∈ R n is scaled by a sufficiently large scalar δ > 0, the input δx attains arbitrarily high confidence.

Q r and f (x) = V r x + a r be the piecewise affine representation of the output of a ReLU network on Q r .

Suppose that V r does not contain identical rows for all r = 1, . . .

, R, then for almost any x ∈ R n and > 0 there exists an δ > 0 and a class i ∈ {1, . . .

, K} such that it holds softmax(f (δx), i) ≥ 1 − .

Moreover, lim δ→∞ softmax(f (δx), i) = 1.

For binary classification tasks, it is standard to treat neural networks as probabilistic models of the conditional distribution p(y|x, w).

Standard deep training involves assigning a maximum a posteriori (MAP) value w MAP to the weights.

Doing so ignores potential uncertainty on w. We will show that this lack of uncertainty is the primary cause of the overconfidence discussed in Hein et al. (2019) .

Unfortunately, there is generally no analytic solution for eq. (1).

But for the logistic link function, good approximations exist when the distribution over the weights is Gaussian p(w|D) = N (w; µ, Σ) with mean µ and covariance Σ. One such approximation (MacKay, 1992) is constructed by scaling the input of the probit function 2 Φ by a constant λ = π/8 .

Using this approximation and the Gaussian assumption, if we let a := w T x, we get

where the last step uses the approximation Φ( π/8 x) ≈ σ(x) a second time, with

In the case of µ = w MAP , eq. (3) can be seen as the "softened" version of the MAP prediction of the classifier, using the covariance of the Gaussian.

The principal aspect of interest of in this paper will be not so much any philosophical point about Bayesian inference, but that the approximate probabilistic Gaussian formalism as outlined in eqs. (1) and (3) introduces the second set of parameters in the form of Σ. We will find that at least asymptotic overconfidence problems can be fixed by setting Σ to virtually any sensible value, regardless of whether they are motivated in a Bayesian fashion or not.

As a first notable property of this approximation, we show below that, in contrast to some other methods for uncertainty quantification (e.g. Monte Carlo dropout (Gal & Ghahramani, 2016) ) it preserves the decision boundary induced by the MAP estimate.

Moreover, this property still holds even if we use any feature map φ and define the linear classifier on the image of this map instead.

The implication is important in practice, as this gives a guarantee that if we apply this approximation to the last layer of any MAP pre-trained neural networks, then the classification accuracy of the marginalized prediction is exactly the same as the MAP classification accuracy. .

The confidence of the marginalized prediction of a linear classifier is the highest in the direction of the lowest curvature, as described by Σ. Then we can obtain a Gaussian approximation p(w|D) ≈ N (w|µ, Σ) of the posterior by setting µ = w MAP and

, the inverse Hessian of the negative log-posterior.

In our binary classification case, p(y|x, w) is assumed to be Bernoulli(σ(w T x)) while p(w) is assumed to be N (w|0, σ 2 0 I), leading to the standard 2 -regularized binary cross-entropy loss.

As our central theoretical contribution, we show that, far away from the training points, z(x) goes to a quantity that only depends on the mean and covariance of the Gaussian over the weights.

This result implies that we can make p(y = 1|x, D) closer to one-half far away from the training points if we can make z(x) closer to zero by controlling the Gaussian.

Proposition 2.3 below shows this in the case of linear classifiers (also cf.

Figure 2 ), while Theorem 2.4 shows that the analysis actually also holds in the case of ReLU networks.

Proposition 2.3.

Let f : R n → R be a binary linear classifier defined by f (x) := w T x and p(w|D) := N (w|µ, Σ) be the distribution over w. Then for any x ∈ R n ,

Furthermore, if x ∈ R n then as δ > 0 goes to infinity

Proof.

See Proposition A.3 in Appendix A.

Recall from the definition, φ is a piecewise affine function.

Thus, we can write the input space as R n = ∪ R r=1 Q r and for every Q r , the restriction φ| Qr :

is an affine function φ| Qr (x) := V r x + a r for some V r ∈ R d×n and a r ∈ R d .

Note that if i, j ∈ {1, . . .

, M } with i = j then in general V i = V j and a i = a j .

Using this definition, we can also show a similar result to Proposition 2.3, in the case when x is replaced by any feature vector in the image of φ.

ReLU network and let p(w|D) := N (w|µ, Σ) be the distribution over w. Then for any x ∈ R n ,

where V ∈ R d×n and a ∈ R d are some matrix and vector that depend on x. Furthermore, as δ > 0 goes to infinity

Proof.

See Theorem A.5 in Appendix A.

Given a target upper bound on the logit and confidence values of a ReLU network, we can concretely pick the covariance Σ that respects the asymptotic bound of Theorem 2.4.

Corollary 2.5 (Σ from a desired upper confidence bound on ReLU networks).

Let f • φ, with φ :

and N (w|µ, Σ) be the distribution over w where the mean µ is fixed and Σ is any SPD matrix.

Then:

(i) For any > 0 there exists Σ such that for any x ∈ R n far away from the training data, we have that |z • φ(x)| ≤ .

(ii) For any 0.5 < p < 1 there exists Σ such that for any x ∈ R n far away from the training data, we have that σ(|z

Proof.

See Corollary A.6 in Appendix A.

Proposition 2.3 and Theorem 2.4 imply that the confidence of a binary linear classifier with ReLU features can be bound closer to one-half by increasing the minimum eigenvalue of the posterior covariance.

In this section, we will move towards the Bayesian setting (i.e. using an explicit prior and likelihood, not just an imposed probability measure on the weights).

Specifically, we will present a way to control the posterior through the prior in a Laplace approximation.

Concretely, the following proposition and its immediate corollary point out that the eigenvalues of the posterior covariance can be increased (bringing |z(δx)| closer to zero) by increasing the prior variance.

.

Let p(w|D) := N (w|µ, Σ) be the posterior over w, obtained via a Laplace approximation with prior N (w|0, σ 2 0 I).

Suppose H is the Hessian w.r.t.

w at µ of the negative log-likelihood of the model.

Then

(ii) For each i = 1, . . .

, d, the ith eigenvalue λ i (Σ) of Σ is a non-decreasing function of σ Proof.

See Proposition A.7 in Appendix A.

We get an immediate corollary from Proposition 2.6 that relates its results to Theorem 2.4.

.

Let p(w|D) := N (w|µ, Σ) be the posterior over w, obtained via a Laplace approximation with prior N (w|0, σ

is a non-increasing function of σ 2 0 with limits

where H is as defined in (i) of Proposition 2.6.

Proof.

See Corollary A.8 in Appendix A.

Lastly, the following corollary formalizes the intuition that the marginalized prediction with the inverse empirical features covariance C −1 as Σ will naturally have high uncertainty far away from the training data.

Furthermore, this property can also be observed for Laplace approximation if the spectral properties of the Hessian ((i) of Proposition 2.6) are not too different to those of C.

(ii) Σ is obtained via a Laplace approximation w.r.t.

a prior N (w|0, σ 2 0 I) with σ 2 0 → ∞ and suppose H defined in (i) of Proposition 2.6 is invertible and the ordering of its eigenvalues is the same as that of C, while the eigenvectors are the same as those of C, then on any level set of µ T φ(x), the confidence decreases faster in the direction where the training data are sparser in the feature space R d .

Proof.

See Corollary A.9 in Appendix A.

Similar statements for multi-class classifiers are not as straight-forward due to the lack of a good closed-form approximation of the integral of softmax under a Gaussian measure.

However, as can be seen in Appendix C, at least the application of the above analysis can easily be generalized to the multi-class case.

In fact, in the experiments (Section 4), we mainly use multi-class classifiers and show empirically that they are effective in mitigating issues that arise from the overconfidence problem.

The overconfidence problem of deep neural networks, and thus ReLU networks, has long been known in the deep learning community (Nguyen et al., 2015) .

However, only recently this issue was demonstrated formally (Hein et al., 2019) .

Many methods have been proposed to combat or at least detect this issue.

Post-hoc heuristics based on temperature or Platt scaling (Guo et al., 2017; Liang et al., 2018) are unable to detect inputs with arbitrarily high confidence far away from the training data (Hein et al., 2019) .

Hein et al. (2019) proposed enhanced training objectives based on robust optimization to mitigate this issue.

Bayesian methods have long been thought to mitigate the overconfidence problem on any neural network (MacKay, 1992) .

Empirical evidence supporting this intuition has also been presented (Liu et al., 2019; Wu et al., 2019, etc.) .

Our results complement these with a theoretical justification for the ReLU-logistic case.

But while our work is theoretical in nature, we believe its application has practical value since it shows that a full Bayesian (expensive) treatment is not necessary if one is only worried about overconfidence.

Indeed, fully Bayesian neural networks are often intractable and crude approximations have to be used, resulting in undesirable results (Foong et al., 2019) .

In this section we validate our theoretical results by applying a Laplace approximation only to the last layer of various widely used ReLU networks and call this method last-layer Laplace approximation (LLLA).

We refer the reader to Appendix C for details.

Note that since LLLA is the simplest Laplace approximation that we can apply to deep networks, our results should also hold for more general Laplace methods, e.g. Kronecker-factored Laplace (KFLA) (Ritter et al., 2018) , where not only the linear classifier's posterior but also the posterior of the feature map is approximated.

Note however, these fully-Bayesian methods are significantly more expensive and require a significant amount of implementation effort.

We will present our empirical results on (i) a 2D toy classification task and (ii) out-of-distribution (OOD) data detection experiments.

For the OOD experiment, we find the optimal prior variance σ 2 0 via a heuristic that follows directly from Corollary 2.7.

Concretely, we pick the largest positive integer that makes the drop on the mean maximum confidence (MMC) of the in-distribution dataset to be within around 0.03 of the MAP's MMC.

Thus, we only set this once without seeing any of the OOD datasets.

The dataset is constructed by sampling the input points from k Gaussians.

The corresponding targets indicate from which Gaussian the point was sampled.

We use a 5-layer ReLU network with 100 hidden units at each layer as the feature map φ.

The classifier, along with this feature map is trained jointly.

We show the results for the binary and multi-class (k = 4) case in Figure 3 .

As we can see, the MAP predictions have high confidence (low entropy) everywhere except at the region close to the decision boundary.

The widely used MC-dropout does not remedy this issue.

While ACET remedies the overconfidence issue, it is expensive and in general does not preserve the decision boundary.

In contrast, LLLA yields better calibrated predictions: high confidence close to the training points and high uncertainty otherwise, while maintaining the MAP's decision boundary.

We furthermore show the zoomed-out version of LLLA prediction we have presented in Figure 1d , along with the contour of the denominator of z (eq. (4)) in Figure 4 .

We see that the covariance acts as a "moderator" for the MAP predictions: As a test point moves away from the training data, the denominator of z becomes larger and the marginalized prediction goes to a constant close to one-half.

Table 1 , LLLA yields competitive performance compared to both CEDA and ACET.

We have shown that even an extremely approximate and virtually non-Bayesian probabilistic Gaussian treatment mitigates the most extreme aspects of overconfidence in ReLU networks.

Our analytical results bound the confidence of the Bayesian prediction of linear classifiers and ReLU networks far away from the training data away from one.

This motivates a spectrum of approximations, from ad-hoc isotropic to "full Bayesian" Laplace approximations.

In the Laplace approximation case, the bound asymptotically converges to a constant whose value can be controlled via the prior.

We validated our results experimentally by constructing a simple Laplace method that can still capture the properties we have shown, specifically by only approximating the last-layer's posterior distribution.

In contrast to other approximations, this method is cheap and simple to implement, yet already yields competitive performance compared to the more expensive, recently proposed non-Bayesian method for combating the overconfidence problem.

While more elaborate Laplace approximations can improve fidelity the further, our results provide virtually any ReLU network with a simple and computationally lightweight way to mitigate overconfidence.

1/2 = 0.

Notice, the denominator of the l.h.s.

is positive.

Thus, it follows that µ f must be 0, implying that σ(µ f ) = 0.5.

Lemma A.2.

Let x ∈ R n be a vector and A ∈ R n×n be an SPD matrix.

If λ min (A) is the minimum eigenvalue of A, then x T Ax ≥ λ min x 2 .

Proof.

Since A is SPD, it admits an eigendecomposition A = QΛQ T and Λ = Λ 1 2 Λ 1 2 makes sense.

Therefore, by keeping in mind that Q T x is a vector in R n , we have

where the last equality is obtained as Q T x 2 = x T Q T Qx and noting that Q is an orthogonal matrix.

Proposition A.3.

Let f : R n → R be a binary linear classifier defined by f (x) := w T x and p(w|D) := N (w|µ, Σ) be the distribution over w. Then for any x ∈ R n ,

Furthermore, if x ∈ R n then as δ > 0 goes to infinity

Proof.

The first result follows directly from Lemma A.2 and by noting that the denominator of eq. (4) is positive since Σ is symmetric positive-definite (SPD) by definition.

For the second result, let x ∈ R n be arbitrary.

By computation and again since the denominator of eq. (4) is positive, we have

We would like to inspect the asymptotic behavior of z(δx) with respect to δ.

First, for the sake of completeness, we can compute that lim δ→0 |z(δx)| = 0.

This reflects the case when δx goes to the decision boundary.

Now, for the case when δ → ∞, we can see that

since 1/δ 2 → 0 as δ → ∞. Therefore, using Lemma A.2 and Cauchy-Schwarz inequality, we have

thus the proof is complete.

Under review as a conference paper at ICLR 2020

Lemma A.4 (Hein et al. (2019)).

Let {Q i } R l=1 be the set of linear regions associated to the ReLU network φ : R n → R n .

For any x ∈ R n there exists α ∈ R with α > 0 and t ∈ {1, . . .

, R} such that δx ∈ Q t for all β ≥ α.

Furthermore, the restriction of φ to Q t can be written as an affine function.

Theorem A.5.

Let f : R d → R be a binary linear classifier defined by f • φ(x) := w T φ(x) where φ : R n → R d is a ReLU network and let p(w|D) := N (w|µ, Σ) be the distribution over w. Then for any x ∈ R n ,

where V ∈ R d×n and a ∈ R d are some matrix and vector that depend on x. Furthermore, as δ > 0 goes to infinity such that x ∈ Q and φ| Q (x) := Vx + a. Applying eq. (4) to φ| Q (x) and following the proof of Proposition 2.3 yield

thus the first result is obtained. , such that for any δ ≥ α, we have that δx ∈ R and the restriction φ| R can be written as Ux + c. Therefore, for any such δ,

Now, notice that as δ → ∞, 1/δ 2 and 1/δ goes to zero.

So, in the limit, we have that

Again, following the proof of Proposition 2.3 (i.e. using Cauchy-Schwarz and Lemma A.2), we can upper-bound this limit with

which concludes the proof.

Corollary A.6 (λ min (Σ) from a desired upper confidence bound on ReLU networks).

Let f • φ, with φ : R n → R d and f : R d → R, be a ReLU network defined by f • φ(x) := w T φ(x) and N (w|µ, Σ) be the distribution over w where the mean µ is fixed and Σ is any SPD matrix.

Then: (i) For any > 0 there exists Σ such that for any x ∈ R n far away from the training data, we have that |z • φ(x)| ≤ .

(ii) For any 0.5 < p < 1 there exists Σ such that for any x ∈ R n far away from the training data, we have that σ(|z • φ(x)|) ≤ p.

Proof.

We begin with (i).

Let > 0 and δ = 8 π µ 2 .

Pick any Σ SPD with λ min (Σ) = δ.

Then, by eq. (12) of Theorem 2.4 and our choice of λ min (Σ), for any z ∈ R n , asymptotically we have that

which is the desired result.

For (ii), let 0.5 < p < 1 be arbitrary.

Observe that the inverse logistic function is given by σ −1 (x) := log x/(1 − x) for 0 < x < 1 and it is positive for 0.5 < x < 1.

Therefore by setting in (i) with

2 and verify that for any x ∈ R n this gives |z(x)| ≤ σ −1 (p).

Thus, for any x ∈ R n far away from the training data, since σ is monotonic, we have that

and the proof is complete.

.

Let p(w|D) := N (w|µ, Σ) be the posterior over w, obtained via a Laplace approximation with prior N (w|0, σ 2 0 I).

Suppose H is the Hessian w.r.t.

w at µ of the negative log-likelihood of the model.

Then

(ii) For each i = 1, . . .

, d, the ith eigenvalue λ i (Σ) of Σ is a non-decreasing function of σ Proof.

The negative log-likelihood of Bernoulli distribution is given by

Now, observing that σ (x) = σ(x)(1 − σ(x)) for all x ∈ R, we can compute

T .

T , since t ∈ {0, 1} by assumption.

By considering all x, t ∈ D, we get (i).

For (ii), first we assume that all Hessians mentioned below are w.r.t.

w. We note that the assumption on the prior implies − log p(w) = 1/2 w T (1/σ 2 0 I)w + const, which has Hessian 1/σ 2 0 I. Thus, the Hessian of the negative log posterior − log p(w|D) = −

log p(w) − log x,t∈D p(y|x, w) is 1/σ 2 0 I + H. This implies that the posterior covariance Σ of the Laplace approximation is given by

Therefore, the ith eigenvalue of Σ for any i = 1, . . .

, n is

.

For all i = 1, . . .

, n, the derivative of λ i (Σ) w.r.t.

σ T for some saturating h : R → R, which has lim x→∞ h(x) = l. Let also φ : R n → R d defined as φ(x) := g(Vx + a) for some V ∈ R d×n and a ∈ R d be a feature map.

Suppose p(w|D) := N (w|µ, Σ) is the distribution over w. Then for any x ∈ D, as δ > 0 goes to infinity

Proof.

By definition,

By definition of g, lim δ→∞ g(δVx + a) = (l, . . .

, l) T =: l, which implies

The theoretical results in the main text essentially tell us that if we have a Gaussian approximate posterior that comes from a Laplace approximation, then using eq. (1) (and eq. (2)) when making a prediction can remedy the overconfidence problem on any ReLU network.

In this section we describe a simple Laplace method that can still capture the properties that we have presented in Section 2.

Concretely, we apply the Laplace approximation only to the linear last layer of ReLU networks, that have been trained via MAP estimation.

For the sake of clarity, we omit the bias in the following and revisit the case where the bias is included at the end of this section.

For the binary classification case, let g : R n → R be a MAP-trained deep ReLU neural network with a linear last-layer.

We can decompose g into a feature map φ : R n → R d and a linear classifier f :

.

Based on Proposition 2.6, we can simply perform a Laplace approximation to get the posterior of the weight of the linear classifier f , i.e. p(w|D) = N (w|w MAP , H −1 ) where H is the Hessian of the negative log-posterior w.r.t.

w at w MAP .

This Hessian could be obtained via automatic differentiation or via the explicit formula stated in (i) of Proposition 2.6.

We emphasize that we only deal with the weight at the last layer of g, i.e. the weight of f , and not the weight of the whole network, thus the inversion of H is rarely a problem.

For instance, large models such as DenseNet-201 (Huang et al., 2017) and ResNet-152 (He et al., 2016a) In the case of multi-class classification, we now have f :

.

We obtain the posterior over a random matrix W ∈ R k×d in the form N (vec(W)|vec(W MAP ), Σ) for some Σ ∈ R dk×dk SPD.

The procedure is still similar to the one described above, since the exact Hessian of the linear multi-class classifier can still be easily and efficiently obtained via automatic differentiation.

Note that in this case we need to invert a dk × dk matrix, which, depending on the size of k, can be quite large.

For a more efficient procedure, we can make a further approximation to the posterior in the multiclass case by assuming the posterior is a matrix Gaussian distribution.

We can use the Kroneckerfactored Laplace approximation (KFLA) Ritter et al. (2018) , but only for the last layer of the network.

That is, we find the Kronecker factorization of the Hessian

Then by definition of a matrix Gaussian (Gupta & Nagar, 1999), we immediately 4 Based on the implementations available in the TorchVision package.

5 For example, the ImageNet dataset has k = 1000.

6 In practice, we take the running average of the Kronecker factors of the Hessian over the mini-batches.

obtain the posterior MN (W|W MAP , U, V).

The distribution of the latent functions is Gaussian, since f := Wφ(x) and p(W|D) = MN (W|W MAP , U, V) imply

where the last equality follows since (φ(x) T Vφ(x)) is a scalar.

We then have the following integral

which can be approximated via MC-integration.

While one can always assume that the bias trick is already used, i.e. it is absorbed in the weight matrix/vector, in practice when dealing with pre-trained networks, one does not have such liberty.

In this case, one can simply assume that the bias b or b is independent of the weight w or W, respectively in the two-and multi-class cases.

By using the same Laplace approximation procedure, one can easily get p(b|D) :

.

This implies w T φ(x)+b =: f and Wφ(x) + b =: f are also Gaussians given by .

Similarly, in the case when the Kronecker-factored approximation is used, we have

Because of the construction above, which is simply done by applying Laplace approximation on the last layer of a ReLU network, we call this method last layer Laplace approximation or LLLA for short.

We present the pseudocodes of LLLA in Algorithms 1 and 2.

Algorithm 1 LLLA with exact Hessian for binary classification.

We train all networks we use in Table 1 for 100 epochs with batch size of 128.

The initial learning rates are 0.001 and 0.1 for MNIST and CIFAR-10 experiments, respectively, and we divide them by 10 at epoch 50, 75, and 95.

We use ADAM and SGD with 0.9 momentum, respectively.

Standard data augmentations, i.e. random crop and standardization are also used for training the network on CIFAR-10.

Meanwhile, for LLLA, we use the Kronecker-factored Hessian.

Algorithm 2 LLLA with Kronecker-factored Hessian for multi-class classification.

A pre-trained network f • φ with W MAP as the weight of f , (averaged) cross-entropy loss L, training set D train , test set D test , mini-batch size m, number of samples s, running average weighting ρ, and prior precision τ 0 = 1/σ 2 0 .

Predictions P containing p(y = i|x, D train ) ∀x ∈ D test ∀i ∈ {1, . . .

, k}.

We further compare the OOD detection performance of LLLA to the temperature scaling method.

To find the optimal temperature, we follow the method of Guo et al. (2017) .

In particular, we use the implementation provided by https://github.com/JonathanWenger/pycalib.

<|TLDR|>

@highlight

We argue theoretically that by simply assuming the weights of a ReLU network to be Gaussian distributed (without even a Bayesian formalism) could fix this issue; for a more calibrated uncertainty, a simple Bayesian method could already be sufficient.
The ADAM optimizer is exceedingly popular in the deep learning community.

Often it works very well, sometimes it doesn’t.

Why?

We interpret ADAM as a combination of two aspects: for each weight, the update direction is determined by the sign of the stochastic gradient, whereas the update magnitude is solely determined by an estimate of its relative variance.

We  disentangle these two aspects and analyze them in isolation, shedding light on ADAM ’s inner workings.

Transferring the "variance adaptation” to momentum- SGD gives rise to a novel method, completing the practitioner’s toolbox for problems where ADAM fails.

Many prominent machine learning models pose empirical risk minimization problems of the form DISPLAYFORM0 where θ ∈ R d is a vector of parameters, {x 1 , . . .

, x M } a training set, and (θ; x) is a loss quantifying the performance of parameter vector θ on example x. Computing the exact gradient in each step of an iterative optimization algorithm becomes inefficient for large M .

Instead, we construct a minibatch B ⊂ {1, . . .

, M } of |B| M data points sampled uniformly and independently from the training set and compute an approximate stochastic gradient DISPLAYFORM1 which is an unbiased estimate, E[g(θ)] = ∇L(θ).

We will denote by σ(θ) The basic stochastic optimizer is stochastic gradient descent (SGD, Robbins & Monro, 1951) and its momentum variants (Polyak, 1964; Nesterov, 1983) .

A number of methods, widely-used in the deep learning community, choose per-element update magnitudes based on the history of stochastic gradient observations.

Among these are ADAGRAD (Duchi et al., 2011) , RMSPROP (Tieleman & Hinton, 2012) , ADADELTA (Zeiler, 2012) and ADAM (Kingma & Ba, 2015) .

We start out from a reinterpretation of the widely-used ADAM optimizer.

Some of the considerations naturally extend to ADAM's close relatives RMSPROP and ADADELTA, but we restrict our attention to ADAM to keep the presentation concise.

ADAM .

In this example, the θ 2 -coordinate has much higher relative variance (η 2 2 = 2.25) than the θ 1 -coordinate (η 2 1 = 0.25) and is thus shortened.

This reduces the variance of the update direction at the expense of biasing it away from the true gradient in expectation.tic gradients and their element-wise square 2 , m t = β 1mt−1 + (1 − β 1 )g t , m t = (1 − β DISPLAYFORM0 DISPLAYFORM1 Here, m t and v t are "bias-corrected" versions of the exponential moving averages to obtain convex combinations of past observed (squared) gradients.

ADAM then updates DISPLAYFORM2 with a small constant ε > 0 guaranteeing numerical stability of this division.

Ignoring ε and assuming |m t,i | > 0 for the moment, we can rewrite the update direction as DISPLAYFORM3 Since m t and v t approximate the first and second moment of the stochastic gradient g t , respectively, v t − m This perspective naturally suggests two alternative methods by incorporating one of the aspects but not the other (see TAB0 ).

Taking the sign of the stochastic gradient (or momentum term) without any further modification gives rise to "Stochastic Sign Descent" (SSD).

On the other hand, "Stochastic Variance-Adapted Gradient" (SVAG) applies element-wise variance adaptation factors directly on the stochastic gradient (or momentum term) instead of on its sign.

We proceed as follows:

In Section 2, we investigate the sign aspect.

In the simplified setting of stochastic quadratic problems, we derive conditions under which the element-wise sign of a stochastic gradient can be a better update direction than the stochastic gradient itself.

Section 3 discusses the variance adaptation.

We present a principled derivation of "optimal" element-wise variance adaptation factors for a stochastic gradient as well as its sign.

Subsequently, we incorporate momentum and briefly discuss the practical estimation of stochastic gradient variance.

Section 4 presents some experimental results.

The idea of using the sign of the gradient as the principal source of the optimizer update has already received some attention in the literature.

The RPROP algorithm (Riedmiller & Braun, 1993) ignores the magnitude of the gradient and dynamically adapts the per-element magnitude of the update based on observed sign changes.

With the goal of reducing communication cost in distributed training of neural networks, Seide et al. (2014) empirically investigate the use of the sign of stochastic gradients.

Regarding the variance adaptation, Schaul et al. (2013) derive element-wise step sizes for stochastic gradient descent that have (among other factors) a dependency on the stochastic gradient variance.

We briefly establish a fact that will be used throughout the paper.

The sign of a stochastic gradient s(θ) = sign(g(θ)) estimates the sign of the true gradient.

Its distribution (and thus the quality of this estimate) is fully characterized by the success probabilities DISPLAYFORM0 These depend on the distribution of the stochastic gradient.

If we assume g(θ) to be Gaussian-which is strongly supported by a Central Limit Theorem argument on Eq. (2)-we have DISPLAYFORM1 see §B.2 in the supplements.

Furthermore, it is E[s(θ) i ] = (2ρ i − 1) sign(∇L(θ) i ).

Can it make sense to ignore the gradient magnitude?

We provide some intuition under which circumstances the element-wise sign of a stochastic gradient is a better update direction than the stochastic gradient itself.

This question is difficult to tackle in general, which is why we restrict the problem class to the simple, yet insightful, case of stochastic quadratic problems, where we can investigate the effects of curvature properties and its interaction with stochastic noise.

Model Problem (Stochastic Quadratic Problem (QP)).

Consider the loss function (θ, x) = 0.5 (θ − x) T Q(θ − x) with a symmetric positive definite matrix Q ∈ R d and "data" coming from DISPLAYFORM0 with ∇L(θ) = Q(θ − x * ).

Stochastic gradients are given by g(θ) = Q(θ − x) ∼ N (x * , ν 2 I).

We want to compare update directions on stochastic QPs in terms of their expected decrease in function value from a single update step.

If we update from θ to θ + αz, we have DISPLAYFORM0 For this comparison of update directions, we allow for the optimal step size that minimizes Eq. (10), which is easily found to be α * = −∇L(θ) DISPLAYFORM1 and yields an expected improvement of DISPLAYFORM2 We find the following expressions/bounds for the improvement of SGD and SSD: DISPLAYFORM3 where the λ i ∈ R + are the eigenvalues of Q with orthonormal eigenvectors DISPLAYFORM4 .

Derivations can be found in §B.1 of the supplements.

Comparing these expressions, we make two observations.

Firstly, I(s) has a dependency on i,j |q ij |.

This quantity relates to the eigenvalues, as well as the orientation of the eigenbasis of Q. By writing Q in its eigendecomposition one finds that DISPLAYFORM5 .

If the eigenvectors are perfectly axis-aligned (diagonal Q), their 1-norms are v i 1 = v i 2 = 1.

It is intuitive that this is the best case for the intrinsically axis-aligned sign update.

In general, the 1-norm is only bounded by DISPLAYFORM6 suggesting that the sign update will have difficulties with arbitrarily oriented eigenbases.

We can alternatively express this matter in terms of "diagonal dominance".

Assuming Q has a percentage c ∈ [0, 1] of its "mass" on the diagonal, i.e., i |q ii | ≥ c i,j |q ij |, we can write DISPLAYFORM7 Becker & LeCun (1988) empirically investigated the diagonal dominance of Hessians in optimization problems arising from neural networks and found relatively high percentages of mass on the diagonals of c = 0.1 up to c = 0.6 for the problems they investigated.

Secondly, I(g) contains the constant offset ν DISPLAYFORM8 i in the denominator, which can become hugely obstructive for ill-conditioned and noisy problems.

In I(s), on the other hand, there is no such interaction between the magnitude of the noise and the eigenspectrum; the noise only manifests in the element-wise success probabilities ρ i , its effect in the denominator is bounded.

A recent paper (Chaudhari et al., 2016) investigated the eigenspectrum in deep learning problems and found it to be very ill-conditioned with the majority of eigenvalues close to zero and a few very large ones.

In summary, we can expect the sign update to be beneficial for noisy, ill-conditioned problems with "diagonally dominant" Hessians.

There is some (weak) empirical evidence that these conditions might be fulfilled in deep learning problems.

Steps Axis-alignedFigure 2: Performance of SGD and SSD on 100-dimensional stochastic quadratic problems.

Rows correspond to different QPs: the eigenspectrum is shown and each is used with a randomly rotated and an axis-aligned eigenbasis.

Columns correspond to different noise levels.

Horizontal axis is number of steps; vertical axis is log function value and is shared per row for comparability.

We verify the above findings on artificially generated stochastic QPs, where all relevant quantities are known analytically and controllable.

We control the eigenspectrum by specifying a diagonal matrix Λ of eigenvalues: (1) a mildly-conditioned problem with eigenvalues drawn uniformly from [0.1, 1.1] and (2) an ill-conditioned problem with a structured eigenspectrum similar to the one reported for neural networks by Chaudhari et al. (2016) by uniformly drawing 90% of the eigenvalues from [0, 1] and 10% from [30, 60] .

Q is then generated by (1) Q = Λ to produce an axis-aligned problem and (2) Q = RΛR T with a rotation matrix R drawn uniformly at random (see Diaconis & Shahshahani, 1987) .

This makes four different matrices, which we consider at noise levels ν ∈ {0, 0.1, 4.0}. We compare SGD and SSD, both with the optimal step size as derived from Eq. (10), which can be computed exactly in this setting.

FIG0 shows the results, which confirm the theoretical findings.

On the well-conditioned, noisefree problem, gradient descent vastly outperforms the sign-based method.

Surprisingly, adding even a little noise almost evens out the difference in performance.

The orientation of the eigenbasis had little effect on the performance of SSD in the well-conditioned case.

On the ill-conditioned problem, the methods work roughly equally well when the eigenbasis is randomly rotated.

As predicted, SSD benefits drastically from an axis-aligned eigenbasis (last row), where it clearly outperforms SGD.

DISPLAYFORM0 and min DISPLAYFORM1 where DISPLAYFORM2 In the sign case, γ i is proportional to the success probability with γ i = 1 if we are certain about the sign (ρ i = 1) and γ i = 0 if we have no information about the sign at all (ρ i = .5).

Applying Eq. (15) top = g, the optimal variance adaptation factors for the sign of a stochastic gradient are found to be γ i = 2ρ i − 1, where DISPLAYFORM0 Recall from Eq. FORMULA7 that, under the Gaussian assumption, the success probabilities of the sign of a stochastic gradient DISPLAYFORM1 .

ADAM uses the variance adaptation factors (1 + η DISPLAYFORM2 , which turns out to be a close approximation of erf[( DISPLAYFORM3 , as shown in Figure 5 in the supplements.

Hence, ADAM can be regarded as an approximate realization of this optimal variance adaptation scheme.

We experimented with both variants and found them to have identical effects.

The small difference between them can be regarded as insignificant when η itself is subject to approximation error.

We thus stick to (1 + η DISPLAYFORM4 for accordance with ADAM and to avoid the (more costly) error function.

Applying Eq. (14) top = g, the optimal variance adaptation factors for SGD are found to be DISPLAYFORM0 This term is known from Schaul et al. (2013) , where it appears together with diagonal curvature estimates in element-wise step sizes for SGD.

We refer to this method (without curvature estimates) as "Stochastic Variance-Adapted Gradient" (SVAG).

A momentum variant will be derived below.

Intriguingly, variance adaptation of this form guarantees convergence without manually decreasing the global step size.

We recover the O(1/t) rate of SGD for smooth, strongly convex functions.

We emphasize that this result considers an "idealized" version of SVAG with exact η 2 i .

It is a motivation for this form of variance adaptation, not a statement about the performance with estimated variances.

Theorem 1.

Let f be µ-strongly convex and L-smooth.

Assume we update θ t+1 = θ t − α(γ t g t ), where g t is a stochastic gradient with DISPLAYFORM1 where f * is the minimum value of f . (Proof in §B.4)

In practice, the relative variance is of course not known and must be estimated.

As noted in the introduction, ADAM obtains an estimate of the stochastic gradient variance from moving averages, σ DISPLAYFORM0 The underlying assumption is that the function does not change drastically over the "effective time horizon" of the moving average, such that the recent gradients can approximately be considered to be iid draws from the stochastic gradient distribution.

An estimate of the relative variance can then be obtained by DISPLAYFORM1 Unlike ADAM we do not use different moving average constants for m t and v t .

The constant for the moving average should define a time horizon over which the gradients can approximately be considered to come from the same distribution.

From this perspective, it is hardly justifiable to use different horizons for the gradient and its square.

Furthermore, we found individual moving average constants for m t and v t to have only minor effect on the performance of our methods.

An alternative variance estimate can be computed locally "within" a single mini-batch.

A more detailed discussion of both estimators can be found in §C of the supplements.

We have experimented with both estimators and found them to work equally well for our purpose of variance adaptation.

We thus stick to moving average-based estimates for the main paper.

Appendix D provides details and experimental results for the mini-batch variant.

When we add momentum-i.e., we want to update in the direction r t or sign(r t ) with a momentum term r t = µr t−1 + g t = t s=0 µ s g t−s -the variance adaptation factors should be determined by the relative variance of r t , according to Lemma 1.

It is DISPLAYFORM0 Replacing DISPLAYFORM1 t−s we could compute these quantities.

However, this would require two additional moving averages and can thus be discarded as impractical.

Fortunately, we can motivate an approximation that does not require any additional memory requirements (see §C): DISPLAYFORM2 Note that the correction factor κ(µ, t) does not appear in ADAM, which updates in the direction sign(m t ) = sign(r t ) but performs variance adaptation based on (v t − m

We compare momentum-SGD (M-SGD) and ADAM to two new methods: First, we consider M-SSD: stochastic sign descent using a momentum term.

The second method is M-SVAG, i.e., SGD with momentum and variance adaptation of the form DISPLAYFORM0 , where the relative variance of the momentum term is estimated from moving averages according to Eq. (19).

These four methods are the four possible recombinations of the sign aspect and the variance adaptation aspect of ADAM, as laid out in TAB0 Compute stochastic gradient g = g(θ)4:Update moving averagesm ← µm DISPLAYFORM1 Bias DISPLAYFORM2 Compute relative variance estimate η 2 = κ(µ, t) DISPLAYFORM3 Eq. FORMULA0 7:Compute variance adaptation factors γ = (1 + η 2 ) −1

Update θ ← θ − α(γ m) 9: end for We do not use an ε-parameter as in ADAM.

In the (rare) case that m i = 0 for coordinate i, the division by zero in line 6 is caught and the update magnitude will be set to zero in line 8.

We tested all methods on three problems: a simple fully-connected neural network on the MNIST data set (LeCun et al., 1998) , as well as convolutional neural networks (CNNs) on the CIFAR-10 and CIFAR-100 data sets (Krizhevsky, 2009).

On CIFAR-10, we used a simple CNN with three convolutional layers, interspersed with max-pooling, and three fully-connected layers.

On CIFAR-100 we used the AllCNN architecture of Springenberg et al. FORMULA0 with a total of nine convolutional layers.

A complete description of all network architectures has been moved to §A. While MNIST and CIFAR-10 are trained with a constant global step size (α), we used a fixed decreasing schedule for CIFAR-100, dividing by 10 after 40k and 50k steps (adopted from Springenberg et al., 2014) .

We used a batch size of 128 on MNIST and 256 on the two CIFAR data sets.

Step sizes (initial step sizes in the case of CIFAR-100) were tuned for each method individually by first finding the maximal stable step size by trial and error, then searching downwards over two orders of magnitude (details in §A).

We selected the one that yielded maximal overall test accuracy within the fixed number of training steps.

Experiments with the best step size have been replicated ten times with different random seeds and all performance indicators are reported as mean plus/minus one standard deviation.

Results are shown in FIG6 .

On MNIST, ADAM clearly outperforms M-SGD.

Interestingly, there is only a very small difference in performance between the two sign-based methods, M-SSD and ADAM.

Apparently, the advantage of ADAM over M-SGD on this problem is primarily due to the sign aspect.

Going from M-SGD to M-SVAG, gives a considerable boost in performance, but M-SVAG is still outperformed by the two sign-based methods.

On CIFAR-10, the sign-based methods again have superior performance.

Neither M-SSD nor M-SGD can benefit significantly from adding variance adaptation.

Finally, the situation is reversed on CIFAR-100, where M-SGD outperforms ADAM.

It attains lower minimal loss values (both training and test) and converges faster.

This is also reflected in the test accuracies, where M-SGD beats ADAM by almost 10 percentage points.

Furthermore, ADAM is much less stable with significantly larger variance in performance.

On this problem, variance adaptation has a small but significant positive effect for the sign-based methods as well as for M-SGD.

When going from M-SGD to M-SVAG we gain some speed in the initial phase.

The difference is later evened out by the manual learning rate decrease (which was necessary, for all methods, to train this architecture to satisfying performance).

We have argued that ADAM combines two aspects: taking signs and variance adaptation.

Our separate analysis of both aspects provides some insight into the inner workings of this method.

Taking the sign can be beneficial, but does not need to be.

Our theoretical analysis suggests that it depends on the interplay of stochasticity, the conditioning of the problem, and its "axis-alignment".

Our experiments confirm that sign-based methods work well on some, but not all problems.

Variance adaptation can be applied to any stochastic update direction.

In our experiments it was beneficial in all cases, but its effect can sometimes be minuscule.

M-SVAG, a variance-adapted variant of momentum-SGD, is a useful addition to the practitioner's toolbox for problems where sign-based methods like ADAM fail.

Its memory and computation cost are identical to ADAM and it has two hyper-parameters, the momentum constant µ and the global step size α.

Our TensorFlow (Abadi et al., 2015) implementation of this method will be made available upon publication.

MNIST We train a simple fully-connected neural network with three hidden layers of 1000, 500 and 100 units with ReLU activation.

The output layer has 10 units with softmax activation.

We use the cross-entropy loss function and apply L 2 -regularization on all weights, but not the biases.

We use a batch size of 128.

The global learning rate α stays constant.

The CIFAR-10 data set consists of 32×32px RGB images with one of ten categorical labels.

We train a convolutional neural network (CNN) with three convolutional layers (64 filters of size 5×5, 96 filters of size 3×3, and 128 filters of size 3×3) interspersed with max-pooling over 3×3 areas with stride 2.

Two fully-connected layers with 512 and 256 units follow.

We use ReLU activation function for all layers.

The output layer has 10 units for the 10 classes of CIFAR-10 with softmax activation.

We use the cross-entropy loss function and apply L 2 -regularization on all weights, but not the biases.

During training we perform some standard data augmentation operations (random cropping of sub-images, left-right mirroring, color distortion) on the input images.

We use a batch size of 256.

The global learning rate α stays constant.

We use the AllCNN architecture of Springenberg et al. (2014) .

It consists of seven convolutional layers, some of them with stride, and no pooling layers.

The fully-connected layers are replaced with two layers of 1×1 convolutions with global spatial averaging in the end.

ReLU activation function is used in all layers.

Details can be found in the original paper.

We use the cross-entropy loss function and apply L 2 -regularization on all weights, but not the biases.

We used the same data augmentation operations as for CIFAR-10 and a batch size of 256.

The global learning rate α is decreased by a factor of 10 after 40k and 50k steps.

Learning rates for each optimizer have been tuned by first finding the maximal stable learning rate by trial and error and then searching downwards over two orders of magnitude with learning rates 6 · 10 m , 3 · 10 m , and 1 · 10 m for order of magnitude m. We evaluated loss and accuracy on the full test set at a constant interval and selected the best-performing learning rate for each method in terms of maximally reached test accuracy.

Using the best learning rate, we replicated the experiment ten times with different random seeds.

We derive the expressions for I(s) and I(g) in Eq. (12).

We drop the fixed θ from the notation for readability.

For SGD, we have DISPLAYFORM0 , which is a general fact for quadratic forms of random variables.

For the stochastic QP the gradient covariance is DISPLAYFORM1 i .

Plugging everything into Eq. (11) yields DISPLAYFORM2 For stochastic sign descent, we have E[s i ] = (2ρ i − 1) sign(∇L i ) and thus Figure 4: Probability density functions (pdf) of three Gaussian distributions, all with µ = 1, but different variances σ 2 = 0.5 (left), σ 2 = 1.0 (middle), σ 2 = 4.0 (right).

The shaded area under the curve corresponds to the probability that a sample from the distribution has the opposite sign than its mean.

For the Gaussian distribution, this probability is uniquely determined by the fraction σ/|µ|, as shown in Lemma 2.

DISPLAYFORM3 Plugging everything into Eq. (11) yields DISPLAYFORM4

We have stated in the main text that the sign of a stochastic gradient, s(θ) = sign(g(θ)), has success probabilities DISPLAYFORM0 under the assumption that g ∼ N (∇L, Σ).

The following Lemma formally proves this statement and Figure 4 provides a pictorial illustration.

DISPLAYFORM1 Proof.

The cumulative density function (cdf) of DISPLAYFORM2 ) is the cdf of the standard normal distribution.

If µ < 0, then DISPLAYFORM3 If µ > 0, then DISPLAYFORM4 where the last step used the anti-symmetry of the error function.

Variance adaptation factor DISPLAYFORM5 Figure 5: Variance adaptation factors as functions of the relative standard deviation DISPLAYFORM6 is the optimal variance adaptation factor for SGD (Eq. 16).

The optimal factor for the sign of a stochastic gradient is erf(( √ 2η) −1 ) under the Gaussian assumption (Eq. 15).

It is closely approximated by DISPLAYFORM7 , which is the factor implicitly employed by ADAM (Eq. 6).

Proof of Lemma 1.

DISPLAYFORM0 Setting the derivative w.r.t.

γ i to zero, we find the optimal choice DISPLAYFORM1 Using DISPLAYFORM2 and easily find the optimal choice DISPLAYFORM3 by setting the derivative to zero.

See Figure 5 for a plot of the variance adaptation factors considered in this paper.

We proof the convergence results for idealized variance-adapted stochastic gradient descent.

We have to clarify an aspect that we have glossed over in the main text.

A stochastic optimizer generates a discrete stochastic process {θ t } t∈N0 .

We denote as E t [·] = E[·|θ 0 , . . .

, θ t ] the conditional expectation given a realization of that process up to time step t. DISPLAYFORM0 Proof of Theorem 1.

Using the Lipschitz continuity of ∇f , we can bound DISPLAYFORM1 Hence, DISPLAYFORM2 Plugging in the definition DISPLAYFORM3 and simplifying, we get DISPLAYFORM4 Using Jensen's inequality DISPLAYFORM5 Due to strong convexity, we have ∇f t 2 ≥ 2µ(f t − f * ) and can further bound DISPLAYFORM6 Inserting this in (33) and subtracting f * , we get DISPLAYFORM7 and, consequently, by total expectation DISPLAYFORM8 which we rewrite, using the shorthand DISPLAYFORM9 LG 2 .To conclude the proof, we will show that this implies e t ∈ O( 1 t ).

Without loss of generality, we assume e t+1 > 0 and get DISPLAYFORM10 where the second step is due to the simple fact that (1−x) −1 ≥ (1+x) for any x ∈ [0, 1).

Summing this inequality over t = 0, . . .

, T − 1 yields e −1T ≥ e −1 0 + T c and, thus, DISPLAYFORM11 which shows that e t ∈ O( 1 t ).4 Jensen's inequality says that i ciφ(xi) ≥ φ( i cixi) for a convex function φ and convex coefficients ci ≥ 0, i ci = 1.

Here, we apply it to the convex function φ(x) = 1/x, x > 0, and coefficients DISPLAYFORM12 g t , we anyways make the assumption that all gradients in the effective time horizon of the moving average have the same mean and variance.

We thus further approximate by replacing m t−s with m t and get DISPLAYFORM13 DISPLAYFORM14 The two scalar factors lead to the correction term κ(µ, t) in Eq. (19).When estimating the gradient variance from the mini-batch (Eq. 44), we can obtain an unbiased estimate of var[r t ] in Eq. (18) vias DISPLAYFORM15 whereŝ t is given by Eq. (44).

Based on the considerations in Section 3, we examined three more variance-adapted methods.

The first is a variation of M-SVAG which estimates stochastic gradient variances locally within the minibatch, as explained in §C.2.

Pseudo-code can be found in Alg.

5.

Furthermore, we tested a variant of ADAM that applies the correction factor from Eq. (19) to the estimate of the relative variance of the momentum term.

We refer to this method as ADAM*. Two variants of ADAM* with the two variance estimates can be found in Algorithms 4 and 5.Algorithm FORMULA2 Compute stochastic gradient g = g(θ)4:Update moving averages m ← µm + (1 − µ)g, v ← µv + (1 − µ)g Update θ ← θ − α(γ sign(m)) 9: end for This is ADAM (β1 = β2 = µ, ε = 0), expect for the correction factor κ(µ, t) for the relative variance.

Update θ ← θ − α(γ sign(m)) 8: end for

We evaluated the variants on the two CIFAR test problems.

FIG11 shows a comparison of the two ADAM* variants with the original ADAM.

FIG12 compares the mini-batch variant of M-SVAG to the one with exponential moving averages.

<|TLDR|>

@highlight

Analyzing the popular Adam optimizer

@highlight

The paper trys to improve Adam based on variance adaption with momentum by proposing two algorithms

@highlight

This paper analyzes the scale-invariance and the particular shape of the learning rate used in Adam, arguing that Adam's update is a combination of a sign-update and a variance-based learning rate.

@highlight

The paper splits ADAM algorithm into two components: stochastic direction in sign of gradient and adaptive stepwise with relative variance, and two algorithms are proposed to test each of them.
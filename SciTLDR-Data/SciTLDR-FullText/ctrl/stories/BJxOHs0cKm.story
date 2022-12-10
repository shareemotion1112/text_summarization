While it has not yet been proven, empirical evidence suggests that model generalization is related to local properties of the optima which can be described via the Hessian.

We connect model generalization with the local property of a solution under the PAC-Bayes paradigm.

In particular, we prove that model generalization ability is related to the Hessian, the higher-order "smoothness" terms characterized by the Lipschitz constant of the Hessian, and the scales of the parameters.

Guided by the proof, we propose a metric to score the generalization capability of the model, as well as an algorithm that optimizes the perturbed model accordingly.

Deep models have proven to work well in applications such as computer vision BID18 BID8 BID14 , speech recognition , and natural language processing BID35 BID6 BID25 .

Many deep models have millions of parameters, which is more than the number of training samples, but the models still generalize well BID11 .On the other hand, classical learning theory suggests the model generalization capability is closely related to the "complexity" of the hypothesis space, usually measured in terms of number of parameters, Rademacher complexity or VC-dimension.

This seems to be a contradiction to the empirical observations that over-parameterized models generalize well on the test data 1 .

Indeed, even if the hypothesis space is complex, the final solution learned from a given training set may still be simple.

This suggests the generalization capability of the model is also related to the property of the solution.

BID15 and BID1 empirically observe that the generalization ability of a model is related to the spectrum of the Hessian matrix ∇ 2 L(w * ) evaluated at the solution, and large eigenvalues of the ∇ 2 L(w * ) often leads to poor model generalization.

Also, BID15 , BID1 and BID31 introduce several different metrics to measure the "sharpness" of the solution, and demonstrate the connection between the sharpness metric and the generalization empirically.

BID2 later points out that most of the Hessian-based sharpness measures are problematic and cannot be applied directly to explain generalization.

In particular, they show that the geometry of the parameters in RELU-MLP can be modified drastically by re-parameterization.

Another line of work originates from Bayesian analysis.

Mackay (1995) first introduced Taylor expansion to approximate the (log) posterior, and considered the second-order term, characterized by the Hessian of the loss function, as a way of evaluating the model simplicity, or "Occam factor".

Recently BID34 use this factor to penalize sharp minima, and determine the optimal batch size.

BID4 connect the PAC-Bayes bound and the Bayesian marginal likelihood when the loss is (bounded) negative log-likelihood, which leads to an alternative perspective on Occam's razor.

BID19 , and more recently, BID7 BID28 BID29 use PAC-Bayes bound to analyze the generalization behavior of the deep models.

Since the PAC-Bayes bound holds uniformly for all "posteriors", it also holds for some particular "posterior", for example, the solution parameter perturbed with noise.

This provides a natural The sharp minimum, even though it approximates the true label better, has some complex structures in its predicted labels, while the flat minimum seems to produce a simpler classification boundary.

way to incorporate the local property of the solution into the generalization analysis.

In particular, BID28 suggests to use the difference between the perturbed loss and the empirical loss as the sharpness metric.

BID3 tries to optimize the PAC-Bayes bound instead for a better model generalization.

Still some fundamental questions remain unanswered.

In particular we are interested in the following question:How is model generalization related to local "smoothness" of a solution?In this paper we try to answer the question from the PAC-Bayes perspective.

Under mild assumptions on the Hessian of the loss function, we prove the generalization error of the model is related to this Hessian, the Lipschitz constant of the Hessian, the scales of the parameters, as well as the number of training samples.

The analysis also gives rise to a new metric for generalization.

Based on this, we can approximately select an optimal perturbation level to aid generalization which interestingly turns out to be related to Hessian as well.

Inspired by this observation, we propose a perturbation based algorithm that makes use of the estimation of the Hessian to improve model generalization.

We consider the supervised learning in PAC-Bayes scenario BID24 BID22 BID23 BID20 ).

Suppose we have a labeled data set S = DISPLAYFORM0 The PAC-Bayes paradigm assumes probability measures over the function class F : X → Y.

In particular, it assumes a "posterior" distribution D f as well as a "prior" distribution π f over the function class F.

We are interested in minimizing the expected loss, in terms of both the random draw of samples as well as the random draw of functions: DISPLAYFORM1 Correspondingly, the empirical loss in the PAC-Bayes paradigm is the expected loss over the draw of functions from the posterior: DISPLAYFORM2 PAC-Bayes theory suggests the gap between the expected loss and the empirical loss is bounded by a term that is related to the KL divergence between D f and π f (McAllester, 1999) BID20 .

In particular, if the function f is parameterized as f (w) with w ∈ W, when D w is perturbed around any w, we have the following PAC-Bayes bound BID32 ) BID33 BID28 BID29 : DISPLAYFORM3 , and π be any fixed distribution over the parameters W. For any δ > 0 and η > 0, with probability at least 1 − δ over the draw of n samples, for any w and any random perturbation u, DISPLAYFORM4 One may further optimize η to get a bound that scales approximately as BID33 .

3 A nice property of the perturbation bound FORMULA4 is it connects the generalization with the local properties around the solution w through some perturbation u around w. In particular, supposeL(w * ) is a local optimum, when the perturbation level of u is small, E u [L(w * + u)] tends to be small, but KL(w * + u||π) may be large since the posterior is too "focused" on a small neighboring area around w * , and vice versa.

As a consequence, we may need to search for an "optimal" perturbation level for u so that the bound is minimized.

DISPLAYFORM5

While some researchers have already discovered empirically the generalization ability of the models is related to the second order information around the local optima, to the best of our knowledge there is no work on how to connect the Hessian matrix ∇ 2L (w) with the model generalization rigorously.

In this section we introduce the local smoothness assumption, as well as our main theorem.

It may be unrealistic to assume global smoothness properties for the deep models.

Usually the assumptions only hold in a small local neighborhood N eigh(w * ) around a reference point w * .

In this paper we define the neighborhood set as DISPLAYFORM0 + is the "radius" of the i-th coordinate.

In our draft we focus on a particular type of radius κ i (w * ) = γ|w * i | + , but our argument holds for other types of radius, too.

In order to get a control of the deviation of the optimal solution we need to assume in N eigh γ, (w * ), the empirical loss functionL in (1) is Hessian Lipschitz, which is defined as: Definition 1 (Hessian Lipschitz).

A twice differentiable function f (·) is ρ-Hessian Lipschitz if: DISPLAYFORM1 where · is the operator norm.

The Hessian Lipschitz condition has been used in the numeric optimization community to model the smoothness of the second-order gradients BID27 BID0 BID13 .

In the rest of the draft we always assume the following: FORMULA2 is convex, and ρ-Hessian Lipschitz.

DISPLAYFORM2 For the uniform perturbation, the following theorem holds: Theorem 2.

Suppose the loss function l(f, x, y) ∈ [0, 1], and model weights are bounded |w i | + κ i (w) ≤ τ i ∀i.

With probability at least 1 − δ over the draw of n samples, for anyw ∈ R m such that assumption 1 holds DISPLAYFORM3 Theorem 2 says if we choose the perturbation levels carefully, the expected loss of a uniformly perturbed model is controlled.

The bound is related to the diagonal element of Hessian (logarithmic), the Lipschitz constant ρ of the Hessian (logarithmic), the neighborhood scales characterized by κ (logarithmic), the number of parameters m, and the number of samples n. Also roughly the perturbation level is inversely related to ∇ 2 i,iL , suggesting the model be perturbed more along the coordinates that are "flat".

4 Similar argument can be made on the truncated Gaussian perturbation, which is presented in Appendix B. In the next section we walk through some intuitions of our arguments.

Suppose the empirical loss functionL(w) satisfies the local Hessian Lipschitz condition, then by Lemma 1 in BID27 , the perturbation of the function around a fixed point can be bounded by terms up to the third-order, DISPLAYFORM0 For perturbations with zero expectation, i.e., E[u] = 0, the linear term in (5), E u [∇L(w) T u] = 0.

Because the perturbation u i for different parameters are independent, the second order term can also be simplified, since FORMULA10 and assumption 1, it is straightforward to see the bound below holds with probability at least DISPLAYFORM1 DISPLAYFORM2 That is, the "posterior" distribution of the model parameters are uniform distribution, and the distribution supports vary for different parameters.

We also assume the perturbed parameters are bounded, i.e., |w i | + κ i (w) ≤ τ i ∀i.5 If we choose the prior π to be u i ∼ U (−τ i , τ i ), and then KL(w + u||π) = i log(τ i /σ i ).The third order term in (6) is bounded by DISPLAYFORM3 Unfortunately the bound in theorem 2 does not explain the over-parameterization phenomenon since when m n the right hand side explodes.

5 One may also assume the same τ for all parameters for a simpler argument.

The proof procedure goes through in a similar way.

DISPLAYFORM4 Solve for σ that minimizes the right hand side, and we have the following lemma: Lemma 3.

Suppose the loss function l(f, x, y) ∈ [0, 1], and model weights are bounded |w i | + κ i (w) ≤ τ i ∀i.

Given any δ > 0 and η > 0, with probability at least 1 − δ over the draw of n samples, for any w * ∈ R m such that assumption 1 holds, DISPLAYFORM5 where DISPLAYFORM6 .

uniformly perturbed random variables, and DISPLAYFORM7 In our experiment, we simply treat η as a hyper-parameter.

Other other hand, one may further build a weighted grid over η and optimize for the best η BID33 .

That leads to Theorem 2.

Details of the proof are presented in the Appendix C and D.5 ON THE RE-PARAMETERIZATION OF RELU-MLP BID2 points out the spectrum of ∇ 2L itself is not enough to determine the generalization power.

In particular, for a multi-layer perceptron with RELU as the activation function, one may re-parameterize the model and scale the Hessian spectrum arbitrarily without affecting the model prediction and generalization when cross entropy (negative log likelihood) is used as the loss and w * is the "true" parameter of the sample distribution.

In general our bound does not assume the loss to be the cross entropy.

Also we do not assume the model is RELU-MLP.

As a result we would not expect our bound stays exactly the same during the re-parameterization.

On the other hand, the optimal perturbation levels in our bound scales inversely when the parameters scale, so the bound only changes approximately with a speed of logarithmic factor.

According to Lemma (3), if we use the optimal σ * on the right hand side of the bound, ∇ 2L (w), ρ, and w * are all behind the logarithmic function.

As a consequence, for RELU-MLP, if we do the re-parameterization trick, the change of the bound is small.

In the next two sections we introduce some heuristic-based approximations enlightened by the bound, as well as some interesting empirical observations.

6 AN APPROXIMATE GENERALIZATION METRIC AssumingL(w) is locally convex around w * , so that ∇ 2 i,iL (w * ) ≥ 0 for all i.

If we look at Lemma 3, for fixed m and n, the only relevant term is i log τi σ * i .

Replacing the optimal σ * , and using |w i | + κ i (w) to approximate τ i , we come up with PAC-Bayes based Generalization metric, called pacGen, DISPLAYFORM8 .6 Even though we assume the local convexity in our metric, in application we may calculate the metric on every points.

When ∇ A self-explained toy example is displayed in FIG0 .

To calculate the metric on real-world data we need to estimate the diagonal elements of the Hessian ∇ 2L as well as the Lipschitz constant ρ of the Hessian.

For efficiency concern we follow Adam (Kingma & Ba, 2014) and approximate ∇ To estimate ρ, we first estimate the Hessian of a randomly perturbed model ∇

(w + u), and then DISPLAYFORM0 .

For the neighborhood radius κ we use γ = 0.1 and = 0.1 for all the experiments in this section.

We used the same model without dropout from the PyTorch example 7 .

Fixing the learning rate as 0.1, we vary the batch size for training.

The gap between the test loss and the training loss, and the metric Ψ κ (L, w * ) are plotted in Figure 2 .

We had the same observation as in BID15 ) that as the batch size grows, the gap between the test loss and the training loss tends to get larger.

Our proposed metric Ψ κ (L, w * ) also shows the exact same trend.

Note we do not use LR annealing heuristics as in BID5 which enables large batch training.

Similarly we also carry out experiment by fixing the training batch size as 256, and varying the learning rate.

Figure 4 shows generalization gap and Ψ κ (L, w * ) as a function of epochs.

It is observed that as the learning rate decreases, the gap between the test loss and the training loss increases.

And the proposed metric Ψ κ (L, w * ) shows similar trend compared to the actual generalization gap.

Similar trends can be observed if we run the same model on CIFAR-10 (Krizhevsky, 2009) as shown in Figure 3 and Figure 5 .

Adding noise to the model for better generalization has proven successful both empirically and theoretically BID38 BID10 BID12 BID3 BID30 .

Instead of only minimizing the empirical loss, (Langford & Caruana, The right hand side of (2) has E u [L(w + u)].

This suggests rather than minimizing the empirical lossL(w), we should optimize the perturbed empirical loss E u [L(w + u)] instead for a better model generalization power.

We introduce a systematic way to perturb the model weights based on the PAC-Bayes bound.

Again we use the same exponential smoothing technique as in Adam (Kingma & Ba, 2014) to estimate the Hessian ∇

.

The details of the algorithm is presented in Algorithm 1, where we treat η as a hyper-parameter.

Even though in theoretical analysis E u [∇L · u] = 0, in applications, ∇L ·

u won't be zero especially when we only implement 1 trial of perturbation.

On the other hand, if the gradient ∇L is close to zero, then the first order term can be ignored.

As a consequence, in Algorithm 1 we only perturb the parameters that have small gradients whose absolute value is below β 2 .

For efficiency issues we used a per-parameter ρ i capturing the variation of the diagonal element of Hessian.

Also we decrease the perturbation level with a log factor as the epoch increases.

We compare the perturbed algorithm against the original optimization method on CIFAR-10, CIFAR-100 BID17 , and Tiny ImageNet 8 .

The results are shown in FIG6 .

We use the Wide-ResNet BID36 as the prediction model.

9 The depth of the chosen model is 58, and the widen-factor is set as 3.

The dropout layers are turned off.

For CIFAR-10 and CIFAR-100, we use Adam with a learning rate of 10 −4 , and the batch size is 128.

For the perturbation parameters we use η = 0.01, γ = 10, and =1e-5.

For Tiny ImageNet, we use SGD with learning rate 10 −2 , and the batch size is 200.

For the perturbed SGD we set η = 100, γ = 1,

Require: η, γ = 0.1, β 1 = 0.999, β 2 = 0.1, =1e-5.

1: Initialization: DISPLAYFORM0 for minibatch in one epoch do

for all i do

if t > 0 then 6: DISPLAYFORM0 g t+1 ← ∇ wLt (w t + u t ) (get stochastic gradients w.r.t.

perturbed loss) 11: DISPLAYFORM1 w t+1 ← OPT(w t ) (update w using off-the-shell algorithms) 13: ImageNet.

For CIFAR, Adam is used as the optimizer, and the learning rate is set as 10 −4 .

For the Tiny ImageNet, SGD is used as the optimizer, and the learning rate is set as 10 −2 .

The dropout method in the comparison uses 0.1 as the dropout rate.

Details can be found in Appendix G. and =1e-5.

Also we use the validation set as the test set for the Tiny ImageNet.

We observe the effect with perturbation appears similar to regularization.

With the perturbation, the accuracy on the training set tends to decrease, but the test on the validation set increases.

The perturbedOPT also works better than dropout possibly due to the fact that the it puts different levels of perturbation on different parameters according to the local smoothness structures, while only one dropout rate is set for the all the parameters across the model for the dropout method.

DISPLAYFORM2

We connect the smoothness of the solution with the model generalization in the PAC-Bayes framework.

We prove that the generalization power of a model is related to the Hessian and the smoothness of the solution, the scales of the parameters, as well as the number of training samples.

In particular, we prove that the best perturbation level scales roughly as the inverse of the square root of the Hessian, which mostly cancels out scaling effect in the re-parameterization suggested by BID2 .

To the best of our knowledge, this is the first work that integrate Hessian in the model generalization bound rigorously.

It also roughly explains the effect of re-parameterization over the generalization.

Based on our generalization bound, we propose a new metric to test the model generalization and a new perturbation algorithm that adjusts the perturbation levels according to the Hessian.

Finally, we empirically demonstrate the effect of our algorithm is similar to a regularizer in its ability to attain better performance on unseen data.

This section discusses the details of the toy example shown in FIG0 .

We construct a small 2-dimensional sample set from a mixture of 3 Gaussians, and then binarize the labels by thresholding them from the median value.

The sample distribution is shown in FIG0 .

For the model we use a 5-layer MLP with sigmoid as the activation and cross entropy as the loss.

There are no bias terms in the linear layers, and the weights are shared.

For the shared 2-by-2 linear coefficient matrix, we treat two entries as constants and optimize the other 2 entries.

In this way the whole model has only two free parameters w 1 and w 2 .

The model is trained using 100 samples.

Fixing the samples, we plot the loss function with respect to the model variablesL(w 1 , w 2 ), as shown in FIG0 .

Many local optima are observed even in this simple two-dimensional toy example.

In particular: a sharp one, marked by the vertical green line, and a flat one, marked by the vertical red line.

The colors on the loss surface display the values of the generalization metric scores (pacGen) defined in Section 6.

Smaller metric value indicates better generalization power.

As displayed in the figure, the metric score around the global optimum, indicated by the vertical green bar, is high, suggesting possible poor generalization capability as compared to the local optimum indicated by the red bar.

We also plot a plane on the bottom of the figure.

The color projected on the bottom plane indicates an approximated generalization bound, which considers both the loss and the generalization metric.10 The local optimum indicated by the red bar, though has a slightly higher loss, has a similar overall bound compared to the "sharp" global optimum.

On the other hand, fixing the parameter w 1 and w 2 , we may also plot the labels predicted by the model given the samples.

Here we plot the prediction from both the sharp minimum FIG0 ) and the flat minimum FIG0 .

The sharp minimum, even though it approximates the true label better, has some complex structures in its predicted labels, while the flat minimum seems to produce a simpler classification boundary.

Because the Gaussian distribution is not bounded but the inequality (5) requires bounded perturbation, we first truncate the distribution.

The procedure of truncation is similar to the proof in BID29 and BID24 .

DISPLAYFORM0 Now let's look at the event DISPLAYFORM1 , by union bound P(E) ≥ 1/2.

Here erf −1 is the inverse Gaussian error function defined as erf(x) = Suppose the coefficients are bounded such that i w 2 i ≤ τ , where τ is a constant.

Choose the prior π as N (0, τ I), and we have DISPLAYFORM2 10 the bound was approximated with η = 39 using inequality (8) Notice that after the truncation the variance only becomes smaller, so the bound of (6) for the truncated Gaussian becomes DISPLAYFORM3 Again whenL(w) is convex around w * such that ∇

(w * ) ≥ 0, solve for the best σ i and we get the following lemma: Lemma 4.

Suppose the loss function l(f, x, y) ∈ [0, 1], and model weights are bounded i w 2 i ≤ τ .

For any δ > 0 and η, with probability at least 1 − δ over the draw of n samples, for any w * ∈ R m such that assumption 1 holds, DISPLAYFORM0 DISPLAYFORM1 .

random variables distributed as truncated Gaussian, DISPLAYFORM2 and σ * 2 i is the i-th diagonal element in Σ * .Again we have an extra term η, which may be further optimized over a grid to get a tighter bound.

In our algorithm we treat η as a hyper-parameter instead.

C PROOF OF LEMMA 3Proof.

We rewrite the inequality (7) below DISPLAYFORM3 The terms related to σ i on the right hand side of (17) are DISPLAYFORM4 Since the assumption is DISPLAYFORM5 Solving for σ that minimizes the right hand side of FORMULA2 , and we have DISPLAYFORM6 The term DISPLAYFORM7 i on the right hand side of FORMULA14 is monotonically increasing w.r.t.

σ 2 , so DISPLAYFORM8 Combine the inequality (20), and the equation FORMULA2 with FORMULA2 , and we complete the proof.

Proof.

Combining (4) and FORMULA14 , we get DISPLAYFORM0 The following proof is similar to the proof of Theorem 6 in BID33 .

Note the η in Lemma (3) cannot depend on the data.

In order to optimize η we need to build a grid of the form DISPLAYFORM1 For a given value of i log τǐ σi , we pick η j , such that j = 1 2 log i log F A LEMMA ABOUT EIGENVALUES OF HESSIAN AND GENERALIZATION By extrema of the Rayleigh quotient, the quadratic term on the right hand side of inequality FORMULA10 is further bounded by DISPLAYFORM2 This is consistent with the empirical observations of BID15 that the generalization ability of the model is related to the eigenvalues of ∇ 2L (w).

The inequality (23) still holds even if the perturbations u i and u j are correlated.

We add another lemma about correlated perturbations below.

Lemma 5.

Suppose the loss function l(f, x, y) ∈ [0, 1].

Let π be any distribution on the parameters that is independent from the data.

Given δ > 0 η > 0, with probability at least 1 − δ over the draw of n samples, for any local optimal w * such that ∇L(w * ) = 0,L(w) satisfies the local ρ-Hessian Lipschitz condition in N eigh κ (w * ), and any random perturbation u, s.t.

, DISPLAYFORM3 Proof.

The proof of the Lemma 5 is straightforward.

Since ∇L(w * ) = 0, the first order term is zero at the local optimal point even if E[u] = 0.

By extrema of the Rayleigh quotient, the quadratic term on the right hand side of inequality FORMULA10 is further bounded by DISPLAYFORM4 Due to the linearity of the expected value, DISPLAYFORM5 which does not assume independence among the perturbations u i and u j for i = j.

This section contains several figures comparing dropout and the proposed perturbation algorithm.

Dropout can be viewed as multiplicative perturbation using Bernoulli distribution.

It has already been widely used in almost every deep models.

For comparison we present results using the exact same wide resnet architectures except the dropout layers are turned on or off.

We report the accuracy with dropout rate of 0.0, 0.1, 0.3, and 0.5 on CIFAR-10 and CIFAR-100.

For Tiny ImageNet we report the result with dropout rate being 0.0, 0.1, and 0.3.

Again for the pertOPT algorithm all the dropout layers are turned off.

The depth of the chosen wide resnet model BID36 is 58, and the widenfactor is set as 3.

For CIFAR-10 and CIFAR-100, we use Adam with a learning rate of 10 −4 , and the batch size is 128.

For the perturbation parameters we use η = 0.01, γ = 10, and =1e-5.

For Tiny ImageNet, we use SGD with learning rate 10 −2 , and the batch size is 200.

For the perturbed SGD we set η = 100, γ = 1, and =1e-5.

Also we use the validation set as the test set for the Tiny ImageNet.

Figure FORMULA14 , (8) , and (9) show the accuracy versus epochs for training and validation in CIFAR-10, CIFAR-100, and Tiny ImageNet respectively.

It is pretty clear that with added dropout the validation/test accuracy got boosted compared to the original method.

For CIFAR-10, dropout rate 0.3 seems to work best compared to all the other dropout configurations.

For CIFAR-100 and Tiny ImageNet, dropout 0.1 seems to work better.

This may be due to the fact that CIFAR-10 has less training samples so more regularization is needed to prevent overfit.

Although both perturbedOPT and dropout can be viewed as certain kind of regularization, in all experiments the perturbed algorithm shows better performance on the validation/test data sets compared to the dropout methods.

One possible explanation is maybe the perturbed algorithm puts different levels of perturbation on different parameters according to the local smoothness structures, while only one dropout rate is set for all the parameters across the model.

<|TLDR|>

@highlight

a theory connecting Hessian of the solution and the generalization power of the model
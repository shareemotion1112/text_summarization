While adversarial training can improve robust accuracy (against an adversary), it sometimes hurts standard accuracy (when there is no adversary).

Previous work has studied this tradeoff between standard and robust accuracy, but only in the setting where no predictor performs well on both objectives in the infinite data limit.

In this paper, we show that even when the optimal predictor with infinite data performs well on both objectives, a tradeoff can still manifest itself with finite data.

Furthermore, since our construction is based on a convex learning problem, we rule out optimization concerns, thus laying bare a fundamental tension between robustness and generalization.

Finally, we show that robust self-training mostly eliminates this tradeoff by leveraging unlabeled data.

Neural networks trained using standard training have very low accuracies on perturbed inputs commonly referred to as adversarial examples BID11 .

Even though adversarial training BID3 BID5 can be effective at improving the accuracy on such examples (robust accuracy), these modified training methods decrease accuracy on natural unperturbed inputs (standard accuracy) BID5 BID18 .

Table 1 shows the discrepancy between standard and adversarial training on CIFAR-10.

While adversarial training improves robust accuracy from 3.5% to 45.8%, standard accuracy drops from 95.2% to 87.3%.One explanation for a tradeoff is that the standard and robust objectives are fundamentally at conflict.

Along these lines, Tsipras et al. BID13 and Zhang et al. BID18 construct learning problems where the perturbations can change the output of the Bayes estimator.

Thus no predictor can achieve both optimal standard accuracy and robust accuracy even in the infinite data limit.

However, we typically consider perturbations (such as imperceptible ∞ perturbations) which do not change the output of the Bayes estimator, so that a predictor with both optimal standard and high robust accuracy exists.

Another explanation could be that the hypothesis class is not rich enough to contain predictors that have optimal standard and high robust accuracy, even if they exist BID8 .

However, Table 1 shows that adversarial training achieves 100% standard and robust accuracy on the training set, suggesting that the hypothesis class is expressive enough in practice.

Having ruled out a conflict in the objectives and expressivity issues, Table 1 suggests that the tradeoff stems from the worse generalization of adversarial training either due to (i) the statistical properties of the robust objective or (ii) the dynamics of optimizing the robust objective on neural networks.

In an attempt to disentangle optimization and statistics, we ask does the tradeoff indeed disappear if we rule out optimization issues?

After all, from a statistical perspective, the robust objective adds information (constraints on the outputs of perturbations) which should intuitively aid generalization, similar to Lasso regression which enforces sparsity BID12 .Contributions.

We answer the above question negatively by constructing a learning problem with a convex loss where adversarial training hurts generalization even when the optimal predictor has both optimal standard and robust accuracy.

Convexity rules out optimization issues, revealing a fundamental statistical explanation for why adversarial training requires more samples to obtain high standard accuracy.

Furthermore, we show that we can eliminate the tradeoff in our constructed problem using the recently-proposed robust self-training BID14 BID0 BID7 BID17 on additional unlabeled data.

In an attempt to understand how predictive this example is of practice, we subsample CIFAR-10 and visualize trends in the performance of standard and adversarially trained models with varying training sample sizes.

We observe that the gap between the accuracies of standard and adversarial training decreases with larger sample size, mirroring the trends observed in our constructed problem.

Recent results from BID0 show that, similarly to our constructed setting, robust self-training also helps to mitigate the trade-off in CIFAR-10.Standard vs. robust generalization.

Recent work BID10 BID15 BID4 BID6 has focused on the sample complexity of learning a predictor that has high robust accuracy (robust generalization), a different objective.

In contrast, we study the finite sample behavior of adversarially trained predictors on the standard learning objective (standard generalization), and show that adversarial training as a particular training procedure could require more samples to attain high standard accuracy.

We construct a learning problem with the following properties.

First, fitting the majority of the distribution is statistically easy-it can be done with a simple predictor.

Second, perturbations of these majority points are low in probability and require complex predictors to be fit.

These two ingredients cause standard estimators to perform better than their adversarially trained robust counterparts with a few samples.

Standard training only fits the training points, which can be done with a simple estimator that generalizes well; adversarial training encourages fitting perturbations of the training points making the estimator complex and generalize poorly.

We consider mapping x ∈ X ⊂ R to y ∈ R where (x, y) is a sample from the joint distribution P and conditional densities exist.

We denote by P x the marginal distribution on X .

We generate the data as y = f (x) + σv i where DISPLAYFORM0 ∼ N (0, 1) and f : X → R. For an example (x, y), we measure robustness of a predictor with respect to an invariance set B(x) that contains the set of inputs on which the predictor is expected to match the target y.

The central premise of this work is that the optimal predictor is robust.

In our construction, we let f be robust by enforcing the invariance property (see Appendix A) DISPLAYFORM1 Given training data consisting of n i.i.d.

samples (x i ,y i ) ∼ P, our goal is to learn a predictor f ∈ F.

We assume that the hypothesis class F contains f and consider the squared loss.

Standard training simply minimizes the empirical risk over the training points.

Robust training seeks to enforce invariance to perturbations of training points by penalizing the worst-case loss over the invariance set B(x i ) with respect to target y i .

We consider regularized estimation and obtain the following standard and robust (adversarially trained) estimators for sample size n: DISPLAYFORM2 DISPLAYFORM3 We construct a P and f such that both estimators above converge to f , but such that the error of the robust estimator f rob n is larger than that off std n for small sample size n.

In our construction, we consider linear predictors as "simple" predictors that generalize well and staircase predictors as "complex" predictors that generalize poorly FIG1 ).

Input distribution.

In order to satisfy the property that a simple predictor fits most of the distribution, we define f to be linear on the set X line ⊆

X , where DISPLAYFORM0 for parameters δ ∈ [0, 1] and a positive integer s. Any predictor that fits points in X line will have low (but not optimal) standard error when δ is small.

Perturbations.

We now define the perturbations such that that fitting perturbations of the majority of the distribution requires complex predictors.

We can obtain a staircase by flattening out the region around the points in X line locally FIG1 ).

This motivates our construction where we treat points in X line as anchor points and the set X c line as local perturbations of these points: x ± for x ∈ X line .

This is a simpler version of the commonly studied ∞ perturbations in computer vision.

For a point that is not an anchor point, we define B(x) as the invariance set of the closest anchor point x .

Formally, for some ∈ (0, DISPLAYFORM1 for some parameter m. Setting the slope as m = 1 makes f resemble a staircase.

Such an f satisfies the invariance property (1) that ensures that the optimal predictor for standard error is also robust.

Note that f (x) = mx (a simple linear function) when restricted to x in X line .

Note also that the invariance sets B(x) are disjoint.

This is in contrast to the example in BID18 , where any invariant function is also globally constant.

Our construction allows a non-trivial robust and accurate estimator.

We generate the output by adding Gaussian noise to the optimal predictor f , i.e., y = f (x)+σv i where v i DISPLAYFORM2 ∼ N (0,1).

An illustration of our convex problem with slope m = 1, with size of the circles proportional to probability under the data distribution.

The dashed blue line shows a simple linear predictor that has low test error but not robust to perturbations to nearby low-probability points, while the solid orange line shows the complex optimal predictor f that is both robust and accurate.

(b): With small sample size (n = 40), any robust predictor that fits the sets B(x) is forced to be a staircase that generalizes poorly.

(c): With large sample size (n = 25000), the training set contains all the points from Xline and the robust predictor is close to f by enforcing the right invariances.

The standard predictor also has low error, but higher than the robust predictor.

(d): An illustration of our convex problem when the slope m = 0.

The optimal predictor f that is robust is a simple linear function.

This setting sees no tradeoff for any sample size.

We empirically validate the intuition that the staircase problem is sensitive to robust training by simulating training with various sample sizes and comparing the test MSE of the standard and robust estimators FORMULA2 and (3) .

We report final test errors here; trends in generalization gap (difference between train and test error) are nearly identical.

See Appendix D for more details.

FIG3 shows the difference in test errors of the two estimators.

For each sample size n, we compare the standard and robust estimators by performing a grid search over regularization parameters λ that individually minimize the test MSE of each estimator.

With few samples, most training samples are from X line and standard training learns a simple linear predictor that fits all of X line .

On the other hand, robust estimators fit the low probability perturbations XAnother common approach to encoding invariances is data augmentation, where perturbations are sampled from B(x) and added to the dataset.

Data augmentation is less demanding than adversarial training which minimizes loss on the worst-case point within the invariance set.

We find that for our staircase example, an estimator trained even with the less demanding data augmentation sees a similar tradeoff with small training sets, due to increased complexity of the augmented estimator.

Section 2.3 shows that the gap between the standard errors of robust and standard estimators decreases as training sample size increases.

Moreover, if we obtained training points spanning X line , then the robust estimator (staircase) would also generalize well and have lower error than the standard estimator.

Thus, a natural strategy to eliminate the tradeoff is to sample more training points.

In fact, we do not need additional labels for the points on X line -a standard trained estimator fits points on X line with just a few labels, and can be used to generate labels on additional unlabeled points.

Recent works have proposed robust self-training (RST) to leverage unlabeled data for robustness BID9 BID0 BID14 BID7 BID17 .

RST is a robust variant of the popular self-training algorithm for semi-supervised learning BID9 , which uses a standard estimator trained on a few labels to generate psuedo-labels for unlabeled data as described above.

See Appendix C for details on RST.For the staircase problem (m = 1), RST mostly eliminates the tradeoff and achieves similar test error to standard training (while also being robust, see Appendix C.2) as shown in FIG3 .

In our staircase problem from Section 2, robust estimators perform worse on the standard objective because these predictors are more complex, thereby generalizing poorly.

Does this also explain the drop in standard accuracy we see for adversarially trained models on real datasets like CIFAR-10? .

Difference between test errors (robust -standard) as a function of the # of training samples n.

For each n, we choose the best regularization parameter λ for each of robust and standard training and plot the difference.

Positive numbers show that the robust estimator has higher MSE than the standard estimator.

(a) For the staircase problem with slope m = 1, we see that for small n, test loss of the robust estimator is larger.

As n increases, the gap closes, and eventually the robust estimator has smaller MSE.

(b) On subsampling CIFAR-10, we see that the gap between test errors (%) of standard and adversarially trained models decreases as the number of samples increases, just like the staircase construction in (a).

Extrapolating, the gap should close as we have more samples.

(c) Robust self-training (RST), using 1000 additional unlabeled points, achieves comparable test MSE to standard training (with the same amount of labeled data) and mostly eliminates the tradeoff seen in robust training.

The shaded regions represent 1 STD.We subsample CIFAR-10 by various amounts to study the effect of sample size on the standard test errors of standard and robust models.

To train a robust model, we use the adversarial training procedure from BID5 against ∞ perturbations of varying sizes (see FIG3 .

The gap in the errors of the standard and adversarially trained models decreases as sample size increases, mirroring the trends in the staircase problem.

Extrapolating the trends, more training data should eliminate the tradeoff in CIFAR-10.

Similarly to the staircase example, BID0 showed that robust self-training with additional unlabeled data improves robust accuracy and standard accuracy in CIFAR-10.

See Appendix C for more details.

One of the key ingredients that causes the tradeoff in the staircase problem is the complexity of robust predictors.

If we change our construction such that robust predictors are also simple, we see that adversarial training instead offers a regularization benefit.

When m = 0, the optimal predictor (which is robust) is linear FIG1 ).

We find that adversarial training has lower standard error by enforcing invariance on B(x) making the robust estimator less sensitive to target noise FIG6 ).Similarly, on MNIST , the adversarially trained model has lower test error than standard trained model.

As we increase the sample size, both standard and adversarially trained models converge to obtain same small test error.

We remark that our observation on MNIST is contrary to that reported in BID13 , due to a different initialization that led to better optimization (see Appendix Section D.2).

In this work, we shed some light on the counter-intuitive phenomenon where enforcing invariance respected by the optimal function could actually degrade performance.

Being invariant could require complex predictors and consequently more samples to generalize well.

Our experiments support that the tradeoff between robustness and accuracy observed in practice is indeed due to insufficient samples and additional unlabeled data is sufficient to mitigate this tradeoff.

We show that the invariance condition (restated, FORMULA7 ) is a sufficient condition for the minimizers of the standard and robust objectives under P in the infinite data limit to be the same.

DISPLAYFORM0 for all x ∈ X .Recall that y = f (x) + σv i where v i DISPLAYFORM1 if f is in the hypothesis class F, then f minimizes the standard objective for the square loss.

If bothf std n (2) andf rob n (3) converge to the same Bayes optimal f as n → ∞, we say that the two estimatorsf std n andf rob n are consistent.

In this section, we show that the invariance condition (7) implies consistency off rob n andf std n .Intuitively, from (7), since f is invariant for all x in B(x), the maximum over B(x) in the robust objective is achieved by the unperturbed input x (and also achieved by any other element of B(x)).

Hence the standard and robust loss of f are equal.

For any other predictor, the robust loss upper bounds the standard loss, which in turn is an upper bound on the standard loss of f (since f is Bayes optimal).

Therefore f also obtains optimal robust loss andf std n and f rob n are consistent and converge to f with infinite data.

Formally, let be the square loss function, and the population loss be E (x,y)∼P [ (f (x),y)].

In this section, all expectations are taken over the joint distribution P. Theorem 1. (Regression) Consider the minimizer of the standard population squared loss, f DISPLAYFORM2 2 .

Assuming (7) holds, we have that for any f , E[maxx ∈B(x) (f (x), y)] ≥ E[maxx ∈B(x) (f * (x),y)], such that f * is also optimal for the robust population squared loss.

Proof.

Note that the optimal standard model is the Bayes estimator, such that f DISPLAYFORM3 where the first equality follows because f * is the Bayes estimator and the second equality is from BID6 .

Noting that for y) ], the theorem statement follows.

DISPLAYFORM4 For the classification case, consistency requires label invariance, which is that argmax y p(y | x) = argmax y p(y |x) ∀x ∈ B(x), BID7 such that the adversary cannot change the label that achieves the maximum but can perturb the distribution.

The optimal standard classifier here is the Bayes optimal classifier f c = argmax y p(y | x).

Assuming that f c = argmax y p(y | x) is in F, then consistency follows by essentially the same argument as in the regression case.

Proof.

Replacing f with f c and (f (x), y) with the zero-one loss 1{argmax j f (x) j = y} in the proof of Theorem 1 gives the result.

In our staircase problem, from (1), we assume that the target y is generated as follows: y = f (x)+σv i where v i DISPLAYFORM5 ∼ N (0,1), we see that the points within an invariance sets B(x) have the same target distribution (target distribution invariance).

DISPLAYFORM6 for all x ∈ X .The target invariance condition above implies consistency in both the regression and classification case.

Distribution of X .

We focus on a 1-dimensional regression case.

Let s be the total number of "stairs" in the staircase problem.

Let s 0 ≤ s be the number of stairs that have a large weight in the data distribution.

Define δ ∈ [0,1] to be the probability of sampling a perturbation point, i.e. x ∈ X c line , which we will choose to be close to zero.

The size of the perturbations is ∈ [0, 1 2 ), which is bounded by 1 2 so that x± = x, for any x ∈ X line .

The standard deviation of the noise in the targets is σ > 0.

Finally, m ∈ [0,1] is a parameter controlling the slope of the points in X line .Let w ∈ ∆ s be a distribution over X line where ∆ s is the probability simplex of dimension s.

We define the data distribution with the following generative process for one sample x. First, sample a point i from X line according to the categorical distribution described by w, such that i ∼ Categorical(w).

Second, sample x by perturbing i with probability δ such that DISPLAYFORM0 Note that this is just a formalization of the distribution described in Section 2.

The sampled x is in X line with probability 1 − δ and X c line with probability δ, where we choose δ to be small.

In addition, in order to exaggerate the difference between robust and standard estimators for small sample sizes, we set w such that the first s 0 stairs have the majority of probability mass.

To achieve this, we set the unnormalized probabilities of w asŵ j = 1/s 0 j < s 0 0.01 j ≥ s 0 and define w by normalizing w =ŵ/ jŵ j .

For our examples, we fix s 0 = 5.

In general, even though we can increase s to create versions of our example with more stairs, s 0 is fixed to highlight the bad extrapolation behavior of the robust estimator.

2 ), where x rounds x to the nearest integer.

The invariance sets are B(x) = { x − , x , x + }.

We define the distribution such that for any x, all points in B(x) have the same mean target value m x .

See FIG1 for an illustration.

Note that B(x) is defined such that ((9)) holds, since for any x 1 ,x 2 ∈ B(x), x 1 = x 2 and thus p(y | x 1 ) = p(y | x 2 ).

The conditional distributions are defined since p(x) > 0 for anyx ∈ B(x).

Our hypothesis class is the family of cubic B-splines as defined in BID2 .

Cubic B-splines are piecewise cubic functions, where the endpoints of each cubic function are called the knots.

In our example, we fix the knots to be τ = [− ,0, ,...,(s−1)− ,s−1,(s−1)+ ], which places a knot on every point on the support of X .

This ensures that the family is expressive enough to include f , which is any function in F which satisfies f (x) = m x for all x in X .

Cubic B-splines can be viewed as a kernel method with kernel feature map Φ : X → R 3s+2 , where s is the number of stairs in the example.

For some regularization parameter λ ≥ 0 we optimize with the penalized smoothing spline loss function over parameters θ, DISPLAYFORM0 where Ω i,j = Φ (t) i Φ (t) j dt measures smoothness in terms of the second derivative.

With respect to the regularized objectives BID1 and (3) , the norm regularizer is f 2 = θ T Ωθ.

We implement the optimization of the standard and robust objectives using the basis described in BID2 .

The regularization penalty matrix Ω computes second-order finite differences of the parameters θ.

Suppose we have n samples of training inputs X = {x 1 ,...,x n } and targets y = {y 1 ,...,y n } drawn from P .

The standard spline objective solves the linear system DISPLAYFORM1 where the i-th row of Φ(X) ∈ R n×(3s+2) is Φ(x i ).

The standard estimator is thenf std n (x) = Φ(x) Tθ std .

We solve the robust objective directly as a pointwise maximum of squared losses over the invariance sets (which is still convex) using CVXPY BID1 .

To construct an example where robustness hurts generalization, the main parameters needed are that the slope m is large and that the probability δ of drawing samples from perturbation points X c line is small.

When slope m is large, the complexity of the true function increases such that good generalization requires more samples.

A small δ ensures that a low-norm linear solution has low test error.

This example is insensitive to whether there is label noise, meaning that σ = 0 is sufficient to observe that robustness hurts generalization.

If m ≈ 0, then the complexity of the true function is low and we observe that robustness helps generalization.

In contrast, this example relies on the fact that there is label noise (σ > 0) so that the noise-cancelling effect of robust training improves generalization.

In the absence of noise, robustness neither hurts nor helps generalization since both the robust and standard estimators converge to the true function (f * (x) = 0) with only one sample.

We show plots for a variety of quantities against number of samples n.

For each n, we pick the best regularization parameter λ with respect to standard test MSE individually for robust and standard training.

in the m = 1 (robustness hurts) and m = 0 (robustness helps) cases, with all the same parameters as before.

In both cases, the test MSE and generalization gap (difference between training MSE and test MSE) are almost identical due to robust and standard training having similar training errors.

In the m = 1 case where robustness hurts FIG10 ), robust training finds higher norm estimators for all sample sizes.

With enough samples, standard training begins to increase the norm of its solution as it starts to converge to the true function (which is complex) and the robust train MSE starts to drop accordingly.

In the m = 0 case where robustness helps (Figure 7 ), the optimal predictor is the line f (x) = 0, which has 0 norm.

The robust estimator has consistently low norm.

With small sample size, the standard estimator has low norm but has high test MSE.

This happens when the standard estimator is close to linear (has low norm), but the estimator has the wrong slope, causing high test MSE.

However, in the infinite data limit, both standard and robust estimators converge to the optimal solution.

We describe the robust self-training procedure, which performs robust training on a dataset augmented with unlabeled data.

The targets for the unlabeled data are generated from a standard estimator trained on the labeled training data.

Since the standard estimator has good standard generalization, the generated targets for the unlabeled data have low error on expectation.

Robust training on the augmented dataset seeks to improve both the standard and robust test error of robust training (over just the labeled training data).

Intuitively, robust self-training achieves these gains by mimicking the standard estimator on more of the data distribution (by using unlabeled data) while also optimizing the robust objective.

In robust self-training, we are given n samples of training inputs X = {x 1 ,...,x n } and targets y = {y 1 ,...,y n } drawn from P .

Suppose that we have additional m unlabeled samples X u drawn from P x .

Robust self-training uses the following steps for a given regularization λ:1.

Compute the standard estimatorf std n (2) on the labeled data (X, y) with regularization parameter λ.2.

Generate pseudo-targets y u =f std n (X u ) by evaluating the standard estimator obtained above on the unlabeled data X u .

3.

Construct an augmented dataset X aug = X ∪ X u , y aug = y ∪ y u .4.

Return a robust estimatorf rob n (3) with the augmented dataset (X aug , y aug ) as training data.

We present relevant results from the recent work of BID0 on robust self-training applied on CIFAR-10 augmented with unlabeled data in TAB2 .

The procedure employed in BID0 is identical to the procedure describe above, using a modified version of adversarial training (TRADES) BID18 as the robust estimator.

In Section 2.4, we show that if we have access to additional unlabeled samples from the data distribution, robust self-training (RST) can mitigate the tradeoff in standard error between robust and standard estimators.

It is important that we do not sacrifice robustness in order to have better standard error.

FIG7 shows that in the case where robustness hurts generalization in our convex construction (m = 1), RST improves over robust training not only in standard test error BID5 .

The models are trained for 200 epochs using minibatched gradient descent with momentum, such that 100% standard training accuracy is achieved for both standard and adversarial models in all cases and > 98% adversarial training accuracy is achieved by adversarially trained models in most cases.

We did not include reuslts for subsampling factors greater than 50, since the test accuracies are very low (20-50%).

However, we note that for very small sample sizes (subsampling factor 500), the robust estimator can have slightly better test accuracy than the standard estimator.

While this behavior is not captured by our example, we focus on capturing the observation that standard and robust test errors converge with more samples.

The MNIST dataset consists of 60000 labeled examples of digits.

We sub-sample the dataset by factors of {1, 2, 5, 8, 10, 20, 40, 50, 80, 200, 500} and report results for a small 3-layer CNN averaged over 2 trials for each sub-sample factor.

All models are trained for 200 epochs and achieve 100% standard training accuracy in all cases.

The adversarial models achieve > 99% adversarial training accuracy in all cases.

We train the adversarial models under the ∞ attack model with PGD adversarial training and = 0.3.

For computing the max in each training step, we use 40 steps of PGD, with step size 0.01 (the parameters used in BID5 ).

We use the Adam optimizer.

The final robust test accuracy when training with the full training set was 91%.Initialization and trade-off for MNIST .

We note here that the tradeoff for adversarial training reported in BID13 is because the adversarially trained model hasn't converged (even after a large number of epochs).

Using the Xavier initialization, we get faster convergence with adversarial training and see no drop in clean accuracy at the same level of robust accuracy.

Interestingly, standard training is not affected by initialization, while adversarial training is dramatically affected.

Number of labeled samples Figure 7 .

Plots as number of samples varies for the case where robustness helps (m = 0).

For each n, we pick the best regularization parameter λ with respect to standard test MSE individually for robust and standard training.

(a),(b) The robust estimator has lower test MSE, and the gap shrinks with more samples.

Note that the trend in test MSE is almost identical to generalization gap.

(c) The robust estimator has consistent norm throughout due to the noise-cancelling behavior of optimizing the robust objective.

While the standard estimator has low norm for small samples, it has high test MSE due to finding a low norm (close to linear) solution with the wrong slope.

@highlight

Even if there is no tradeoff in the infinite data limit, adversarial training can have worse standard accuracy even in a convex problem.
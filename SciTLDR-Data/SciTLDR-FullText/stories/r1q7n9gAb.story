We show that gradient descent on an unregularized logistic regression problem, for almost all separable datasets, converges to the same direction as the max-margin solution.

The result generalizes also to other monotone decreasing loss functions with an infimum at infinity, and we also discuss a multi-class generalizations to the cross entropy loss.

Furthermore, we show this convergence is very slow, and only logarithmic in the convergence of the loss itself.

This can help explain the benefit of continuing to optimize the logistic or cross-entropy loss even after the training error is zero and the training loss is extremely small, and, as we show, even if the validation loss increases.

Our methodology can also aid in understanding implicit regularization in more complex models and with other optimization methods.

It is becoming increasingly clear that implicit biases introduced by the optimization algorithm play a crucial role in deep learning and in the generalization ability of the learned models BID7 BID1 BID15 BID5 Wilson et al., 2017) .

In particular, minimizing the training error, without any explicit regularization, over models with more parameters and more capacity then the number of training examples, often yields good generalization, despite the empirical optimization problem being highly underdetermined.

That is, there are many global minima of the training objective, most of which will not generalize well, but the optimization algorithm (e.g. gradient descent) biases us toward a particular minimum that does generalize well.

Unfortunately, we still do not have a good understanding of the biases introduced by different optimization algorithms in different situations.

We do have a decent understanding of the implicit regularization introduced by early stopping of stochastic methods or, at an extreme, of one-pass (no repetition) stochastic optimization.

However, as discussed above, in deep learning we often benefit from implicit bias even when optimizing the (unregularized) training error to convergence, using stochastic or batch methods.

For loss functions with attainable, finite, minimizers, such as the squared loss, we have some understanding of this: In particular, when minimizing an underdetermined least squares problem using gradient descent starting from the origin, we know we will converge to the minimum Euclidean norm solution.

But the logistic loss, and its generalization the cross-entropy loss which is often used in deep learning, do not admit a finite minimizer on separable problems.

Instead, to drive the loss toward zero and thus minimize it, the predictor must diverge toward infinity.

Do we still benefit from implicit regularization when minimizing the logistic loss on separable data?

Clearly the norm of the predictor itself is not minimized, since it grows to infinity.

However, for prediction, only the direction of the predictor, i.e. the normalized w(t)/ w(t) , is important.

How does w(t)/ w(t) behave as t → ∞ when we minimize the logistic (or similar) loss using gradient descent on separable data, i.e., when it is possible to get zero misclassification error and thus drive the loss to zero?In this paper, we show that even without any explicit regularization, for all most all datasets (except a zero measure set), when minimizing linearly separable logistic regression problems using gradient descent, we have that w(t)/ w(t) converges to the L 2 maximum margin separator, i.e. to the solution of the hard margin SVM.

This happens even though the norm w , nor the margin constraint, are in no way part of the objective nor explicitly introduced into optimization.

More generally, we show the same behavior for generalized linear problems with any smooth, monotone strictly decreasing, lower bounded loss with an exponential tail.

Furthermore, we characterize the rate of this convergence, and show that it is rather slow, with the distance to the max-margin predictor decreasing only as O(1/ log(t)).

This explains why the predictor continues to improve even when the training loss is already extremely small.

We emphasize and demonstrate that this bias is specific to gradient descent, and changing the optimization algorithm, e.g. using adaptive learning rate methods such as ADAM BID6 , changes this implicit bias.

Consider a dataset {x n , y n } N n=1 , with binary labels y n ∈ {−1, 1}.

We analyze learning by minimizing an empirical loss of the form DISPLAYFORM0 y n w x n .(2.1)where w ∈ R d is the weight vector.

A bias term could be added in the usual way, extending x n by an additional '1' component.

To simplify notation, we assume that ∀n : y n = 1 -this is true without loss of generality, since we can always re-define y n x n as x n .We are particularly interested in problems that are linearly separable, and the loss is smooth monotone strictly decreasing and non-negative: Assumption 1.

The dataset is strictly linearly separable: ∃w * such that ∀n : w * x n > 0 .

1 , (so ∀u : (u) > 0, (u) < 0 and lim u→∞ (u) = lim u→∞ (u) = 0) and a β-smooth function, i.e. its derivative is β-Lipshitz.

Many common loss functions, including the logistic, exp-loss, probit and sigmoidal losses, follow Assumption 2.

Assumption 2 also straightforwardly implies that L (w) is a βσ 2 max (X )-smooth function, where the columns of X are all samples, and σ max (X ) is the maximal singular value of X.Under these conditions, the infimum of the optimization problem is zero, but it is not attained at any finite w. Furthermore, no finite critical point w exist.

We consider minimizing eq. 2.1 using Gradient Descent (GD) with a fixed learning rate η, i.e., with steps of the form: DISPLAYFORM0 We do not require convexity.

Under Assumptions 1 and 2, gradient descent converges to the global minimum (i.e. to zero loss) even without it: Lemma 1.

Let w (t) be the iterates of gradient descent (eq. 2.2) with η < 2β DISPLAYFORM1 max (X ) and any starting point w(0).

Under Assumptions 1 and 2, we have: (1) lim t→∞ L (w (t)) = 0, (2) lim t→∞ w (t) = ∞, and (3) ∀n : lim t→∞ w (t) x n = ∞.Proof.

Since the data is strictly linearly separable, ∃w * which linearly separates the data, and therefore DISPLAYFORM2 For any finite w, this sum cannot be equal to zero, as a sum of negative terms, since ∀n : w * x n > 0 and ∀u : (u) < 0.

Therefore, there are no finite critical points w, for which ∇L (w) = 0.

But gradient descent on a smooth loss with an appropriate stepsize is always guaranteed to converge to a critical point: ∇L (w (t)) → 0 (see, e.g. Lemma 5 in Appendix A.4, slightly adapted from BID1 , Theorem 2).

This necessarily implies that w (t) → ∞ while ∀n : w (t) x n > 0 for large enough t-since only then w (t) x n → 0.

Therefore, L (w) → 0, so GD converges to the global minimum.

The main question we ask is: can we characterize the direction in which w(t) diverges?

That is, does the limit lim t→∞ w (t) / w (t) always exists, and if so, what is it?In order to analyze this limit, we will need to make a further assumption on the tail of the loss function:Definition 2.

A function f (u) has a "tight exponential tail", if there exist positive constants c, a, µ + , µ − , u + and u − such that DISPLAYFORM3 Assumption 3.

The negative loss derivative − (u) has a tight exponential tail (Definition 2).For example, the exponential loss (u) = e −u and the commonly used logistic loss (u) = log (1 + e −u ) both follow this assumption with a = c = 1.

We will assume a = c = 1 -without loss of generality, since these constants can be always absorbed by re-scaling x n and η.

We are now ready to state our main result:Theorem 3.

For almost all datasets (i.e., except for a measure zero) which are strictly linearly separable (Assumption 1) and given a β-smooth decreasing loss function (Assumption 2) with an exponential tail (Assumption 3), gradient descent (as in eq. 2.2) with stepsize η < 2β DISPLAYFORM4 max (X ) and any starting point w(0) will behave as: DISPLAYFORM5 where the residual ρ (t) is bounded and so DISPLAYFORM6 whereŵ is the L 2 max margin vector (the solution to the hard margin SVM): DISPLAYFORM7 Since the theorem holds for almost all datasets, in particular, it holds with probability 1 if {x n } N n=1are sampled from an absolutely continuous distribution.

Proof Sketch We first understand intuitively why an exponential tail of the loss entail asymptotic convergence to the max margin vector: Assume for simplicity that (u) = e −u exactly, and examine the asymptotic regime of gradient descent in which ∀n : w (t) x n → ∞, as is guaranteed by Lemma 1.

If w (t) / w (t) converges to some limit w ∞ , then we can write w (t) = g (t) w ∞ + ρ (t) such that g (t) → ∞, ∀n :x n w ∞ > 0, and lim t→∞ ρ (t) /g (t) = 0.

The gradient can then be written as: DISPLAYFORM8 (2.5) As g(t) → ∞ and the exponents become more negative, only those samples with the largest (i.e., least negative) exponents will contribute to the gradient.

These are precisely the samples with the smallest margin argmin n w ∞ x n , aka the "support vectors".

The negative gradient (eq. 2.5) would then asymptotically become a non-negative linear combination of support vectors.

The limit w ∞ will then be dominated by these gradients, since any initial conditions become negligible as w (t) → ∞ (from Lemma 1).

Therefore, w ∞ will also be non-negative linear combination of support vectors, and so will its scalingŵ = w ∞ / min n w ∞ x n .

We therefore have: DISPLAYFORM9 These are precisely the KKT condition for the SVM problem (eq. 2.4) and we can conclude thatŵ is indeed its solution and w ∞ is thus proportional to it.

To prove Theorem 3 rigorously, we need to show that w (t) / w (t) has a limit, that g (t) = log (t) and to bound the effect of various residual errors, such as gradients of non-support vectors and the fact that the loss is only approximately exponential.

To do so, we substitute eq. 2.3 into the gradient descent dynamics (eq. 2.2), with w ∞ =ŵ being the max margin vector and g(t) = log t.

We then show that the increment in the norm of ρ (t) is bounded by C 1 t −ν for some C 1 > 0 and ν > 1, which is a converging series.

This happens because the increment in the max margin term, w [log (t + 1) − log (t)] ≈ŵt −1 , cancels out the dominant t −1 term in the gradient −∇L (w (t)) (eq. 2.5 with g (t) = log (t) and w ∞ x n = 1).

A complete proof can be found in Appendix A.More refined analysis: characterizing the residual We can furthermore characterize the asymptotic behavior of ρ (t).

To do so, we need to refer to the KKT conditions (eq. 2.6) of the SVM problem (eq. 2.4) and the associated support vectors S = argmin nŵ x n .

The following refinement of Theorem 3 is also proved in Appendix A: Theorem 4.

Under the conditions and notation of Theorem 3, if, in addition the support vectors span the data (i.e. rank (X S ) = rank (X) where the columns of X are all samples and of X S are the support vectors), then lim t→∞ ρ (t) =w, wherew is unique, given w (0), and a solution to DISPLAYFORM10 Note these equations are well-defined for almost all datasets, since (see Lemma 8 in Appendix F) then there are at most d support vectors, α n are unique and ∀n ∈ S : α n = 0.

The solution in eq. 2.3 implies that w (t) / w (t) converges to the normalized max margin vectorŵ/ ŵ .

Moreover, this convergence is very slow-logarithmic in the number of iterations.

Specifically, in Appendix B we show that Theorem 3 implies the following tight rates of convergence:The normalized weight vector converges to normalized max margin vector in L 2 norm DISPLAYFORM0 and in angle 2) and the margin converges as DISPLAYFORM1 DISPLAYFORM2 this slow convergence is in sharp contrast to the convergence of the (training) loss: DISPLAYFORM3 A simple construction (also in Appendix B) shows that the rates in the above equations are tight.

Thus, the convergence of w(t) to the max-marginŵ can be logarithmic in the loss itself, and we might need to wait until the loss is exponentially small in order to be close to the max-margin solution.

This can help explain why continuing to optimize the training loss, even after the training error is zero and the training loss is extremely small, still improves generalization performance-our results suggests that the margin could still be improving significantly in this regime.

The dataset (positive and negatives samples (y = ±1) are respectively denoted by + and • ), max margin separating hyperplane (black line), and the asymptotic solution of GD (dashed blue).

For both GD and GD with momentum (GDMO), we show: (B) The norm of w (t), normalized so it would equal to 1 at the last iteration, to facilitate comparison.

As expected (eq. 2.3), the norm increases logarithmically; (C) the training loss.

As expected, it decreases as t −1 (eq. 3.4); and (D&E) the angle and margin gap of w (t) fromŵ (eqs. 3.2 and 3.3).

As expected, these are logarithmically decreasing to zero.

Implementation details: The dataset includes four support vectors: x 1 = (0.5, 1.5) , x 2 = (1.5, 0.5) with y 1 = y 2 = 1, and x 3 = −x 1 , x 4 = −x 2 with y 3 = y 4 = −1 (the L 2 normalized max margin vector is thenŵ = (1, 1) / √ 2 with margin equal to √ 2 ), and 12 other random datapoints (6 from each class), that are not on the margin.

We used a learning rate η = 1/σ max (X), where σ max (X) is the maximal singular value of X, momentum γ = 0.9 for GDMO, and initialized at the origin.

A numerical illustration of the convergence is depicted in FIG0 .

As predicted by the theory, the norm w(t) grows logarithmically (note the semi-log scaling), and w(t) converges to the max-margin separator, but only logarithmically, while the loss itself decreases very rapidly (note the log-log scaling).An important practical consequence of our theory, is that although the margin of w(t) keeps improving, and so we can expect the population (or test) misclassification error of w(t) to improve for many datasets, the same cannot be said about the expected population loss (or test loss)!

At the limit, the direction of w(t) will converge toward the max margin predictorŵ.

Althoughŵ has zero training error, it will not generally have zero misclassification error on the population, or on a test or a validation set.

Since the norm of w(t) will increase, if we use the logistic loss or any other convex loss, the loss incurred on those misclassified points will also increase.

More formally, consider the logistic loss (u) = log(1+e −u ) and define also the hinge-at-zero loss h(u) = max(0, −u).

Sinceŵ classifies all training points correctly, we have that on the training set N n=1 h(ŵ x n ) = 0.

However, on the population we would expect some errors and so E[h(ŵ x)] > 0.

Since w(t) ≈ŵ log t and (αu) → αh(u) as α → ∞, we have: DISPLAYFORM4 That is, the population loss increases logarithmically while the margin and the population misclassification error improve.

Roughly speaking, the improvement in misclassification does not out-weight the increase in the loss of those points still misclassified.

The increase in the test loss is practically important because the loss on a validation set is frequently used to monitor progress and decide on stopping.

Similar to the population loss, the validation loss L val (w (t)) = x∈V w (t) x calculated on an independent validation set V, will increase logarithmically with t (since we would not expect zero validation error), which might cause us to think we are over-fitting or otherwise encourage us to stop the optimization.

But this increase does not actually represent the model getting worse, merely w(t) getting larger, and in fact the model might be getting better (with larger margin and possibly smaller error rate).

We discuss several possible extensions of our results.

So far, we have discussed the problem of binary classification.

For multi-class problems commonly encountered, we frequently learn a predictor w k for each class, and use the cross-entropy loss with a softmax output, which is a generalization of the logistic loss.

What do the linear predictors w k (t) converge to if we minimize the cross-entropy loss by gradient descent on the predictors?

In Appendix C we analyze this problem for separable data, and show that again, the predictors diverge to infinity and the loss converges to zero.

Furthermore, we show that, generically, the loss converges to a logistic loss for transformed data, for which our theorems hold.

This strongly suggests that gradient descent converges to a scaling of the K-class SVM solution: DISPLAYFORM0 We believe this can also be established rigorously and for generic exponential tailed multi-class loss.

In this paper we examined the implicit bias of gradient descent.

Different optimization algorithms exhibit different biases, and understanding these biases and how they differ is crucial to understanding and constructing learning methods attuned to the inductive biases we expect.

Can we characterize the implicit bias and convergence rate in other optimization methods?In FIG0 we see that adding momentum does not qualitatively affects the bias induced by gradient descent.

In FIG4 in Appendix E we also repeat the experiment using stochastic gradient descent, and observe a similar bias.

This is consistent with the fact that momentum, acceleration and stochasticity do not change the bias when using gradient descent to optimize an under determined least squares problems.

It would be beneficial, though, to rigorously understand how much we can generalize our result to gradient descent variants, and how the convergence rates might change in these cases.

Employing adaptive methods, such as AdaGrad BID0 and ADAM BID6 , does significantly affect the bias.

In FIG1 we show the predictors obtained by ADAM and by gradient descent on a simple data set.

Both methods converge to zero training error solutions.

But although gradient descent converges to the L 2 max margin predictor, as predicted by our theory, ADAM does not.

The implicit bias of adaptive method has been a recent topic of interest, with BID3 and Wilson et al. (2017) suggesting they lead to worse generalization.

Figure 3: Training of a convolutional neural network on CIFAR10 using stochastic gradient descent with constant learning rate and momentum, softmax output and a cross entropy loss, where we achieve 8.3% final validation error.

We observe that, approximately: (1) The training loss decays as a t −1 , (2) the L 2 norm of last weight layer increases logarithmically, (3) after a while, the validation loss starts to increase, and (4) in contrast, the validation (classification) error slowly improves.

Wilson et al. discuss the limit of AdaGrad on lest square problems, but fall short of providing an actual characterization of the limit.

This is not surprising, as the limit of AdaGrad on least square problems is fragile and depends on the choice of stepsize and other parameters, and thus complicated to characterize.

We expect our methodology could be used to precisely characterize the implicit bias of such methods on logistic regression problems.

The asymptotic nature of the analysis is appealing here, as it is insensitive to the initial point, initial conditioning matrix, and large initial steps.

More broadly, it would be interesting to study the behavior of mirror descent and natural gradient descent, and relate the bias they induce to the potential function or divergence underlying them.

A reasonable conjecture, which we have not yet investigated, is that for any potential function Ψ(w), these methods converge to the maximum Ψ-margin solution arg min w Ψ(w)s.t.∀n : w x n ≥ 1.

Since mirror descent can be viewed as regularizing progress using Ψ(w), it is worth noting the results of BID11 : they considered the regularization path w λ = arg min L(w) + λ w p p for similar loss function as we do, and showed that lim λ→0 w λ / w λ p is proportional to the maximum L p margin solution.

Rosset et al. do not consider the effect of the optimization algorithm, and instead add explicit regularization-here we are specifically interested in the bias implied by the algorithm not by adding (even infinitesimal) explicit regularization.

Our analysis also covers the exp-loss used in boosting, as its tail is similar to that of the logistic loss.

However, boosting is a coordinate descent procedure, and not a gradient descent procedure.

Indeed, the coordinate descent interpretation of AdaBoost shows that coordinate descent on the exp-loss for a linearly separable problem is related to finding the maximum L 1 margin solution BID12 BID10 BID13 .

In this paper, we only consider linear prediction.

Naturally, it is desirable to generalize our results also to non-linear models and especially multi-layer neural networks.

Even without a formal extension and description of the precise bias, our results already shed light on how minimizing the cross-entropy loss with gradient descent can have a margin maximizing effect, how the margin might improve only logarithmically slow, and why it might continue improving even as the validation loss increases.

These effects are demonstrated in FIG2 and Table 1 which portray typical training of a convolutional neural network using unregularized gradient descent 2 .

As can be seen, the norm of the weight increases, but the validation error continues decreasing, albeit very slowly (as predicted by the theory), even after the training error is zero and the training loss is extremely small.

We can now understand how even though the loss is already extremely small, some sort of margin might be gradually improving as we continue optimizing.

We can also observe how the validation loss increases despite the validation error decreasing, as discussed in Section 3.As an initial advance toward tackling deep network, we can point out that for two special cases, our results may be directly applied to multi-layered networks.

First, our results may be applied exactly, 8.9% Table 1 : Sample values from various epochs in the experiment depicted in FIG2 .as we show in Appendix D, if only a single weight layer is being optimized, and furthermore, after a sufficient number of iterations, the activation units stop switching and the training error goes to zero.

Second, our results may also be applied directly to the last weight layer if the last hidden layer becomes fixed and linearly separable after a certain number of iterations.

This can become true, either approximately, if the input to the last hidden layer is normalized (e.g., using batch norm), or exactly, if the last hidden layer is quantized (Hubara et al., 2016) .

With multi-layered neural networks in mind, BID2 recently embarked on a study of the implicit bias of under-determined matrix factorization problems, where we minimize the squared loss of linear observation of a matrix by gradient descent on its factorization.

Since a matrix factorization can be viewed as a two-layer network with linear activations, this is perhaps the simplest deep model one can study in full, and can thus provide insight and direction to studying more complex neural networks.

Gunasekar et al. conjectured, and provided theoretical and empirical evidence, that gradient descent on the factorization for an under-determined problem converges to the minimum nuclear norm solution, but only if the initialization is infinitesimally close to zero and the step-sizes are infinitesimally small.

With finite step-sizes or finite initialization, Gunasekar et al. could not characterize the bias.

It would be interesting to study the same problem with a logistic loss instead of squared loss.

Beyond the practical relevance of the logistic loss, taking our approach has the advantage that because of its asymptotic nature, it does not depend on the initialization and step-size.

It thus might prove easier to analyze logistic regression on a matrix factorization instead of the least square problem, providing significant insight into the implicit biases of gradient descent on non-convex multi-layered optimization.

We characterized the implicit bias induced by gradient descent when minimizing smooth monotone loss functions with an exponential tail.

This is the type of loss commonly being minimized in deep learning.

We can now rigorously understand:1.

How gradient descent, without early stopping, induces implicit L 2 regularization and converges to the maximum L 2 margin solution, when minimizing the logistic loss, or exploss, or any other monotone decreasing loss with appropriate tail.

In particular, the non-tail part does not affect the bias and so the logistic loss and the exp-loss, although very different on non-separable problems, behave the same for separable problems.

The bias is also independent of the step-size used (as long as it is small enough to ensure convergence) and (unlike for least square problem) is also independent on the initialization.

2.

This convergence is very slow.

This explains why it is worthwhile continuing to optimize long after we have zero training error, and even when the loss itself is already extremely small.

3.

We should not rely on slow decrease of the training loss, or on no decrease of the validation loss, to decide when to stop.

We might improve the validation, and test, errors even when the validation loss increases and even when the decrease in the training loss is tiny.

Perhaps that gradient descent leads to a max L 2 margin solution is not a big surprise to those for whom the connection between L 2 regularization and gradient descent is natural.

Nevertheless, we are not familiar with any prior study or mention of this fact, let alone a rigorous analysis and study of how this bias is exact and independent of the initial point and the step-size.

Furthermore, we also analyze the rate at which this happens, leading to the novel observations discussed above.

Perhaps even more importantly, we hope that our analysis can open the door to further analysis of different optimization methods or in different models, including deep networks, where implicit regularization is not well understood even for least square problems, or where we do not have such a natural guess as for gradient descent on linear problems.

Analyzing gradient descent on logistic/cross-entropy loss is not only arguably more relevant than the least square loss, but might also be technically easier.

In the following proofs, for any solution w (t), we define r (t) = w (t) −ŵ log t −w, whereŵ andw follow the conditions of Theorems 3 and 4, that isŵ is the L 2 is the max margin vector, which satisfies eq. 2.4: DISPLAYFORM0 andw is a vector which satisfies eq. 2.7: DISPLAYFORM1 where we recall that we denoted X S ∈ R d×|S| as the matrix whose columns are the support vectors, a subset S ⊂ {1, . . .

, N } of the columns of X = [x 1 , . . . , DISPLAYFORM2 In Lemma 8 (Appendix F) we prove that for almost every dataset α is uniquely defined, there are no more then d support vectors and α n = 0, ∀n ∈ S. Therefore, eq. A.1 is well-defined in those cases.

If the support vectors do not span the data, then the solutionw to eq. A.1 might not be unique.

In this case, we can use any such solution in the proof.

We furthermore denote θ = min DISPLAYFORM3 and by C i , i ,t i (i ∈ N) various positive constants which are independent of t. Lastly, we define P 1 ∈ R d×d as the orthogonal projection matrix 3 to the subspace spanned by the support vectors (the columns of X S ), and P 2 = I − P 1 as the complementary projection (to the left nullspace of X S ).

In this section we first examine the special case that (u) = e −u and take the continuous time limit of gradient descent: η → 0 , soẇ (t) = −∇L (w (t)) .

The proof in this case is rather short and self-contained (i.e., does not rely on any previous results), and so it helps to clarify the main ideas of the general (more complicated) proof which we will give in the next sections.

Recall we defined r (t) = w (t) − log (t)ŵ −w . (A.3) Our goal is to show that r (t) is bounded, and therefore ρ (t) = r (t) +w is bounded.

Eq. A.3 implies thatṙ where in the last equality we used eq. A.3 and decomposed the sum over support vectors S and non-support vectors.

We examine both bracketed terms.

DISPLAYFORM0 Recall thatŵ x n = 1 for n ∈ S, and that we defined (in eq. A.1)w so that n∈S exp −w x n x n =ŵ.

Thus, the first bracketed term in eq. A.5 can be written as 1 t n∈S exp −w x n − x n r (t) x n r (t) − 1 t n∈S exp −w x n x n = 1 t n∈S exp −w x n exp −x n r (t) − 1 x n r (t) ≤ 0 (A.6) since z (e −z − 1) ≤ 0.

Furthermore, since exp (−z) z ≤ 1 and θ = argmin n / ∈S x nŵ (eq. A.2), the second bracketed term in eq. A.5 can be upper bounded by DISPLAYFORM1 Substituting eq. A.6 and A.7 into eq. A.5 and integrating, we obtain, that ∃C, C such that DISPLAYFORM2 since θ > 1 (eq. A.2).

Thus, we showed that r(t) is bounded, which completes the proof for the special case.

Next, we give the proof for the general case (discrete time, and exponentially-tailed functions).

Though it is based on a similar analysis as in the special case we examined in the previous section, it is somewhat more involved since we have to bound additional terms.

First, we state two auxilary Lemmata, which are proven below in appendix sections A.4 and A.5: Lemma 5.

Let L (w) be a β-smooth non-negative objective.

If η < 2β −1 , then, for any w(0), with the GD sequence w (t + 1) = w (t) − η∇L (w(t)) (A.8)we have that ∞ u=0 ∇L (w (u)) 2 < ∞ and therefore lim t→∞ ∇L (w (t)) 2 = 0.Lemma 6.

We have DISPLAYFORM0 Additionally, ∀ 1 > 0 , ∃C 2 , t 2 , such that ∀t > t 2 , if (A.10) then the following improved bound holds DISPLAYFORM1 DISPLAYFORM2 Our goal is to show that r (t) is bounded, and therefore ρ (t) = r (t) +w is bounded.

To show this, we will upper bound the following equation DISPLAYFORM3 First, we note that first term in this equation can be upper-bounded by DISPLAYFORM4 = w (t + 1) −ŵ log (t + 1) −w − w (t) +ŵ log (t) +w DISPLAYFORM5 where in (1) we used eq. 2.3, in (2) we used eq. 2.2, and in (3) we used ∀x > 0 : x ≥ log (1 + x) > 0, and also thatŵ DISPLAYFORM6 sinceŵ x n ≥ 1 (from the definition ofŵ) and (u) ≤ 0.Also, from Lemma 5 we know that DISPLAYFORM7 Substituting eq. A.15 into eq. A.13, and recalling that a t −ν power series converges for any ν > 1, we can find C 0 such that .16) Note that this equation also implies that ∀ 0 DISPLAYFORM8 DISPLAYFORM9 Next, we would like to bound the second term in eq. A.12.

From eq. A.9 in Lemma 6, we can find t 1 , C 1 such that ∀t > t 1 : DISPLAYFORM10 Thus, by combining eqs. A.18 and A.16 into eq. A.12, we find DISPLAYFORM11 which is a bounded, since θ > 1 (eq. A.2).

Therefore, r (t) is bounded.

All that remains now is to show that r (t) → 0 if rank (X S ) = rank (X), and thatw is unique given w (0).

To do so, this proof will continue where the proof of Theorem 3 stopped, using notations and equations from that proof.

Since r (t) has a bounded norm, its two orthogonal components r (t) = P 1 r (t) + P 2 r (t) also have bounded norms (recall that P 1 , P 2 were defined in the beginning of appendix section A).

From eq. 2.2, ∇L (w) is spanned by the columns of X. If rank (X S ) = rank (X), then it is also spanned by the columns of X S , and so P 2 ∇L (w) = 0.

Therefore, P 2 r (t) is not updated during GD, and remains constant.

Sincew in eq. 2.3 is also bounded, we can absorb this constant P 2 r (t) intow without affecting eq. 2.7 (since ∀n ∈ S : x n P 2 r (t) = 0).

Thus, without loss of generality, we can assume that r (t) = P 1 r (t).

∃C 2 , t 2 : ∀t > t 2 : (r (t + 1) − r (t)) r (t) ≤ −C 2 t −1 < 0 .Combining this with eqs. A.12 and A.16, implies that ∃t 3 > max [t 2 , t 0 ] such that ∀t > t 3 such that r (t) > 1 , we have that r (t + 1) 2 is a decreasing function since then DISPLAYFORM0 Additionally, this result also implies that we cannot have r (t) > 1 ∀t > t 3 , since then we arrive to the contradiction.

DISPLAYFORM1 Therefore, ∃t 4 > t 3 such that r (t 4 ) ≤ 1 .

Recall also that r (t) is a decreasing function whenever r (t) ≥ 1 (eq. A.19).

Also, recall that t 4 > t 0 , so from eq. A.17, we have that ∀t > t 4 ,| r (t + 1) − r (t) | < 0 .

Combining these three facts we conclude that ∀t > t 4 : r (t) ≤ 1 + 0 .

Since this reasoning holds ∀ 1 , 0 , this implies that r (t) → 0.Lastly, we note that since P 2 r (t) is not updated during GD, we have that P 2 (w − w (0)) = 0.

This setsw uniquely, together with eq. 2.7.A.4 PROOF OF LEMMA 5Lemma 5.

Let L (w) be a β-smooth non-negative objective.

If η < 2β −1 , then, for any w(0), with the GD sequence w (t + 1) = w (t) − η∇L (w(t)) (A.8)we have that DISPLAYFORM2 This proof is a slightly modified version of the proof of Theorem 2 in BID1 .

Recall a well-known property of β-smooth functions: DISPLAYFORM3 The right hand side is upper bounded by a finite constant, since L (w (0)) < ∞ and 0 ≤ L (w (t + 1)).

This implies DISPLAYFORM4 and therefore ∇L (w (t)) 2 → 0.A.5 PROOF OF LEMMA 6Recall that we defined r (t) = w (t) −ŵ log t −w, withŵ andw follow the conditions of the Theorems 3 and 4, i.e.ŵ is the L 2 max margin vector and (eq. 2.4), and eq. 2.7 holds ∀n ∈ S : η exp −x nw = α n .Lemma 6.

We have DISPLAYFORM5 then the following improved bound holds DISPLAYFORM6 From Lemma 1, ∀n : lim t→∞ w (t) x n = ∞. In addition, from assumption 3 the negative loss derivative − (u) has an exponential tail e −u (recall we assume a = c = 1 without loss of generality).

Combining both facts, we have positive constants µ − , µ + , t − and t + such that ∀n DISPLAYFORM7 Next, we examine the expression we wish to bound, recalling that r (t) = w (t) −ŵ log t −w: DISPLAYFORM8 where in last line we used eqs. 2.6 and 2.7 to obtain DISPLAYFORM9 We examine the three terms in eq. A.23.

The first term can be upper bounded bŷ DISPLAYFORM10 where in (1) we used that P 2ŵ = P 2 X S α = 0 from eq. 2.6, and in (2) we used thatŵ r (t) = o (t), sinceŵ DISPLAYFORM11 where in the last line we used that ∇L (w (t)) = o (1), from Lemma 5.Next, we upper bound the second term in eq. A.23, ∀t > t + : DISPLAYFORM12 ≤ η n / ∈S: x n r(t)≥0 DISPLAYFORM13 ≤ η n / ∈S: x n r(t)≥0 DISPLAYFORM14 ≤ η n / ∈S: x n r(t)≥0 (A.25) where in (1) we used eq. A.21, in (2) we used w (t) =ŵ log t +w + r (t), in (3) we used xe −x ≤ 1 and x n r (t) ≥ 0, in (4) we used θ > 1, from eq. A.2 and in (5) we defined t + = max t + , exp min nw x n .

DISPLAYFORM15 Lastly, we will aim to bound the sum in the third term in eq. A.23 DISPLAYFORM16 We examine each term k in this sum, and divide into two cases, depending on the sign of x k r (t).First, if x k r (t) ≥ 0, then term k in eq. A.26 can be upper bounded ∀t > t + , using eq. A.21, by DISPLAYFORM17 (A.27) We further divide into cases:1.

If x k r(t) ≤ C 0 t −0.5µ+ , then we can upper bound eq. A.27 with DISPLAYFORM18 2.

If x k r(t) > C 0 t −0.5µ+ , then we can find t + > t + to upper bound eq. A.27 ∀t > t + : DISPLAYFORM19 ≤ ηt .29) where in (1) we used the fact that e −x ≤ 1 − x + x 2 for x ≥ 0 and in (2) we defined t + so that the previous expression is negative -this is possible since t −0.5µ+ decreases slower then t −µ+ .

DISPLAYFORM20 3.

If x k r(t) ≥ 2 , then we define t + > t + such that t + > exp min nw x n e 0.5 2 − 1 −1/µ+ , and therefore ∀t > t + , we have 1 + t −µ+ exp −µ +w x n e −

2 < e −0.5 2 .This implies that ∀t > t + we can upper bound eq. A.27 by DISPLAYFORM21 Second, if x k r(t) < 0, we again further divide into cases:1.

If x k r(t) ≤ C 0 t −0.5µ− , then, since − w (t) x n > 0, we can upper bound term k in eq. A.26 with DISPLAYFORM22 2.

If x k r (t) > C 0 t −0.5µ− , then, using eq. A.22 we upper bound term k in eq. A.26 with DISPLAYFORM23 Next, we will show that ∃t − > t − such that the last expression is strictly negative ∀t > t − .Let M > 1 be some arbitrary constant.

Then, since t .34) which is lower bounded by DISPLAYFORM24 DISPLAYFORM25 In this case last line is strictly larger then 1 for sufficiently large t.

Therefore, after we substitute eqs. A.33 and A.34 into A.32, we find that ∃t − > t M > t − such that ∀t > t − , term k in eq. A.26 is strictly negative .35) 3.

If x k r(t) ≥ 2 , which is a special case of the previous case ( x k r (t) > C 0 t −0.5µ− ) then ∀t > t − , either eq. A.33 or A.34 holds.

Furthermore, in this case, ∃t − > t − and M > 1 such that ∀t > t − eq. A.34 can be lower bounded by DISPLAYFORM26 DISPLAYFORM27 Substituting this, together with eq. A.33, into eq. A.32, we can find C 0 > 0 such we can upper bound term k in eq. A.26 with DISPLAYFORM28 To conclude, we choose t 0 = max t + , t − :1.

If P 1 r (t) ≥ 1 (as in Eq. A.10), we have that A.37) where in (1) we used P 1 x n = x n ∀n ∈ S, in (2) we denoted by σ min (X S ), the minimal non-zero singular value of X S and used eq. A.10.

Therefore, for some k, DISPLAYFORM29 DISPLAYFORM30 .

In this case, we denote C 0 as the minimum between C 0 (eq. A.36) and η exp − max nw x n 1 − e −0.5 2 2 (eq. A.30).

Then we find that eq. A.26 can be upper bounded by −C 0 t −1 + o t −1 , ∀t > t 0 , given eq. A.10.

Substituting this result, together with eqs. A.24 and A.25 into eq. A.23, we obtain ∀t > t 0 (r (t + 1) − r (t)) r (t) ≤ −C 0 t −1 + o t −1 .This implies that ∃C 2 < C 0 and ∃t 2 > t 0 such that eq. A.11 holds.

This implies also that eq. A.9 holds for P 1 r (t) ≥ 1 .

2.

Otherwise, if P 1 r (t) < 1 , we find that ∀t > t 0 , each term in eq. A.26 can be upper bounded by either zero (eqs. A.29 and A.35), or terms proportional to t −1−1.5µ+ (eq. A.28) or t −1−0.5µ− , (eq. A.31).

Combining this together with eqs. A.24, A.25 into eq. A.23 we obtain (for some positive constants C 3 , C 4 , C 5 , and C 6 ) DISPLAYFORM31 Therefore, ∃t 1 > t 0 and C 1 such that eq. A.9 holds.

From Theorem 3, we can write w (t) =ŵ log t + ρ (t), where ρ (t) has bounded norm.

Calculation of normalized weight vector (eq. 3.1): DISPLAYFORM0 where to obtain eq. B.1 we used DISPLAYFORM1 , and in the last line we used the fact that ρ (t) has a bounded norm.

We use eq. B.1 to calculate the angle (eq. 3.2): DISPLAYFORM2 Published as a conference paper at ICLR 2018Calculation of margin (eq. 3.3): DISPLAYFORM3 where in eq. B.2 we used eq. A.2.Calculation of the training loss (eq. 3.4): DISPLAYFORM4 Next, we give an example demonstrating the bounds above are strict.

Consider optimization with and exponential loss (u) = e −u , and a single data point x = (1, 0).

In this caseŵ = (1, 0) and ŵ = 1.

We take the limit η → 0, and obtain the continuous time version of GD: w 1 (t) = exp (−w (t)) ;ẇ 2 (t) = 0.We can analytically integrate these equations to obtain w 1 (t) = log (t + exp (w 1 (0))) ; w 2 (t) = w 2 (0) .Using this example with w 2 (0) > 0, it is easy to see that the above upper bounds are strict.

Lastly, recall that V is a set of indices for validation set samples.

We calculate of the validation loss for logistic loss, if the error of the L 2 max margin vector has some classification errors on the validation, i.e., ∃k ∈ V :ŵ x k < 0: DISPLAYFORM5

We examine multiclass classification.

In the case the labels are the class index y n ∈ {1, . . .

, K} and we have a weight matrix W ∈ R K×d with w k being the k-th row of W.Furthermore, we define w = vec W , a basis vector e k ∈ R K so that(e k ) i = δ ki , and the matrix A k ∈ R dK×d so that A k = e k ⊗ I d , where ⊗ is the Kronecker product and I d is the d-dimension identity matrix.

Note that A k w = w k .Consider the cross entropy loss with softmax output DISPLAYFORM0 Using our notation, this loss can be re-written as DISPLAYFORM1 If, again, make the assumption that the data is strictly linearly separable, i.e., in our notation DISPLAYFORM2 then the expression DISPLAYFORM3 .

is strictly negative for any finite w. However, from Lemma 5, in gradient descent with learning rate η > 2β −1 , we have that ∇L (w (t)) → 0.

This implies that: w (t) → ∞, and ∀k = y n , ∃r : w (t) (A r − A k ) x n → ∞, which implies ∀k = y n , max k w (t) (A k − A yn ) x n → −∞. Examining the loss (eq. C.1) we find that L (w (t)) → 0 in this case.

Thus, we arrive to an equivalent Lemma to Lemma 1, for this case:Lemma 7.

Let w (t) be the iterates of gradient descent (eq. 2.2) with η < 2β −1 , for crossentropy loss operating on a softmax output, under the assumption of strict linear separability (Assumption 4), then: (1) lim t→∞ L (w (t)) = 0, (2) lim t→∞ w (t) = ∞, and (3) ∀n, k = y n : DISPLAYFORM4 Therefore, since DISPLAYFORM5 where in the last line we assumed for simplicity that w (t) (A k − A yn ) x n has a unique minimum in k, since then the other exponential terms inside the log become negligible.

If DISPLAYFORM6 has a limit k n , then we definex n = (A yn − A kn ) x n , so eq. C.2 is transformed to the standard logistic regression loss N n=1 log 1 + exp −w (t) x n , to which our Theorems directly apply.

Therefore, w (t) / w (t) →ŵ wherê w = argmin w w 2 s.t.

∀n : w x n ≥ 1Recalling that A k w = w k , we can re-write this as arg min w1,...,w K K k=1 w k 2 s.t.

∀n, ∀k = y n : w yn x n ≥ w k x n + 1

We examine a deep neural network (DNN) with m = 1, . . .

, L layers, piecewise linear activation functions f l and loss function following assumption 3, parameterized by weights matrices W l .

Since f l are piecewise linear, we can write for almost every u: f l (u) = ∇f l (u) u (an element-wise product).

Given an input sample x n , for each layer l the input u n,l and output v n,l are calculated sequentially in a "forward propagation" u n,l =W l v n,l−1 ; v n,l = f l (u n,l ) (D.1)initialized by v n,0 = x n .

Then, given the DNN output u n,L and target y n ∈ {−1, 1} the loss (y n u n,L ) can be calculated.

During training, the gradients of the loss are calculated using the chain rule in a "back-propagation" ∇f l (u n,m ) W m x n = δ n,l W l v n,l−1 .Denotingx n,l = y n δ n,l ⊗ v n,l−1 and w l = vec W l we obtain w l (t + 1) − w l (t) = −η N n=1 w lxn,l x n,l .We got the same update as in eq. 2.2.

Thus, ifx n,l does not change between iterations and becomes linearly separable so the training error can go to zero, we can apply Theorem 3.

This can happen if we only optimize W l , and the activation units stop crossing their thresholds, after a sufficient number of iterations.

Lemma 8.

For almost all datasets there is a unique α which satisfies the KKT conditions (eq. 2.6):

α n x n ∀n α n ≥ 0 andŵ x n = 1 OR α n = 0 andŵ x n > 1 Furthermore, in this solution α n = 0 ifŵ x n = 1, i.e., x n is a support vector (n ∈ S), and there are at most d such support vectors.

Proof.

For almost every set X, no more than d points x n can be on the same hyperplane.

Therefore, since all support vectors must lie on the same hyperplane, there can be at most d support vectors, for almost every X.Given the set of support vectors, S, the KKT conditions of eq. 2.6 entail that α n = 0 if n / ∈ S and 1 = X Sŵ = X S X S α S , (F.1)where we denoted α S as α restricted to the support vector components.

For almost every set X, since d ≥ |S|, X S X S ∈ R |S|×|S| is invertible.

Therefore, α S has the unique solution DISPLAYFORM0 This implies that ∀n ∈ S, α n is equal to a rational function in the components of X S , i.e., α n = p n (X S ) /q n (X S ), where p n and q n are polynomials in the components of X S .

Therefore, if α n = 0, then p n (X S ) = 0, so the components of X S must be at a root of the polynomial p n .

The roots of the polynomial p n have measure zero, unless ∀X S : p n (X S ) = 0.

However, p n cannot be identically equal to zero, since, for example, if X S = I |S|×|S| , 0 |S|×(d−|S|) , then X S X S = I |S|×|S| , and so in this case ∀n ∈ S, α n = 1 = 0, from eq. F.2.Therefore, for a given S, the event that "eq. F.1 has a solution with a zero component" has a zero measure.

Moreover, the union of these events, for all possible S, also has zero measure, as a finite union of zero measures sets (there are only finitely many possible sets S ⊂ {1, . . .

, N } ).

This implies that, for almost all datasets X, α n = 0 only if n / ∈ S. Furthermore, for almost all datasets the solution α is unique: for each dataset, S is uniquely deteremined, and given S , the solution eq. F.1 is uniquely given by eq. F.2.

@highlight

The normalized solution of gradient descent on logistic regression (or a similarly decaying loss) slowly converges to the L2 max margin solution on separable data.

@highlight

The paper offers a formal proof that gradient descent on the logistic loss converges very slowly to the hard SVM solution in the case where the data are linearly separable. 

@highlight

This paper focuses on characterising the behaviour of log loss minimisation on linearly separable data, and shows that log-loss, minimised with gradient descent, leads to convergence to the max-margin solution.
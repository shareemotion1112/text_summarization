In this paper, we study the implicit regularization of the gradient descent algorithm in homogeneous neural networks, including fully-connected and convolutional neural networks with ReLU or LeakyReLU activations.

In particular, we study the gradient descent or gradient flow (i.e., gradient descent with infinitesimal step size) optimizing the logistic loss or cross-entropy loss of any homogeneous model (possibly non-smooth), and show that if the training loss decreases below a certain threshold, then we can define a smoothed version of the normalized margin which increases over time.

We also formulate a natural constrained optimization problem related to margin maximization, and prove that both the normalized margin and its smoothed version converge to the objective value at a KKT point of the optimization problem.

Our results generalize the previous results for logistic regression with one-layer or multi-layer linear networks, and provide more quantitative convergence results with weaker assumptions than previous results for homogeneous smooth neural networks.

We conduct several experiments to justify our theoretical finding on MNIST and CIFAR-10 datasets.

Finally, as margin is closely related to robustness, we discuss potential benefits of training longer for improving the robustness of the model.

A major open question in deep learning is why gradient descent or its variants, are biased towards solutions with good generalization performance on the test set.

To achieve a better understanding, previous works have studied the implicit bias of gradient descent in different settings.

One simple but insightful setting is linear logistic regression on linearly separable data.

In this setting, the model is parameterized by a weight vector w, and the class prediction for any data point x is determined by the sign of w x. Therefore, only the direction w/ w 2 is important for making prediction.

Soudry et al. (2018a; b) ; Ji and Telgarsky (2018) ; Nacson et al. (2018) investigated this problem and proved that the direction of w converges to the direction that maximizes the L 2 -margin while the norm of w diverges to +∞, if we train w with (stochastic) gradient descent on logistic loss.

Interestingly, this convergent direction is the same as that of any regularization path: any sequence of weight vectors {w t } such that every w t is a global minimum of the L 2 -regularized loss L(w) + with λ t → 0 (Rosset et al., 2004) .

Indeed, the trajectory of gradient descent is also pointwise close to a regularization path (Suggala et al., 2018) .

The aforementioned linear logistic regression can be viewed as a single-layer neural network.

A natural and important question is to what extent gradient descent has similiar implicit bias for modern deep neural networks.

For theoretical analysis, a natural candidate is to consider homogeneous neural networks.

Here a neural network Φ is said to be (positively) homogeneous if there is a number L > 0 (called the order) such that the network output Φ(θ; x), where θ stands for the parameter and x stands for the input, satisfies the following:

∀c > 0 : Φ(cθ; x) = c L Φ(θ; x) for all θ and x.

(1) It is important to note that many neural networks are homogeneous (Neyshabur et al., 2015; Du et al., 2018) .

For example, deep fully-connected neural networks or deep CNNs with ReLU or LeakyReLU activations can be made homogeneous if we remove all the bias terms, and the order L is exactly equal to the number of layers.

In (Wei et al., 2018) , it is shown that the regularization path does converge to the max-margin direction for homogeneous neural networks with cross-entropy or logistic loss.

This result suggests that gradient descent or gradient flow may also converges to the max-margin direction by assuming homogeneity, and this is indeed true for some sub-classes of homogeneous neural networks.

For gradient flow, this convergent direction is proven for linear fully-connected networks (Ji and Telgarsky, 2019a) .

For gradient descent on linear fully-connected and convolutional networks, (Gunasekar et al., 2018b ) formulate a constrained optimization problem related to margin maximization and prove that gradient descent converges to the direction of a KKT point or even the max-margin direction, under various assumptions including the convergence of loss and gradient directions.

In an independent work, (Nacson et al., 2019a) generalize the result in (Gunasekar et al., 2018b) to smooth homogeneous models (we will discuss this work in more details in Section 2).

In this paper, we identify a minimal set of assumptions for proving our theoretical results for homogeneous neural networks on classification tasks.

Besides homogeneity, we make two additional assumptions:

1.

Exponential-type Loss Function.

We require the loss function to have certain exponential tail (see Appendix A for the details).

This assumption is not restrictive as it includes the most popular classfication losses: exponential loss, logistic loss and cross-entropy loss.

2. Separability.

The neural network can separate the training data during training (i.e., the neural network can achieve 100% training accuracy) 1 .

While the first assumption is natural, the second requires some explanation.

In fact, we assume that at some time t 0 , the training loss is smaller than a threshold, and the threshold here is chosen to be so small that the training accuracy is guaranteed to be 100% (e.g., for the logistic loss and cross-entropy loss, the threshold can be set to ln 2).

Empirically, state-of-the-art CNNs for image classification can even fit randomly labeled data easily (Zhang et al., 2017) .

Recent theoretical work on over-parameterized neural networks (Allen-Zhu et al., 2019; Zou et al., 2018) show that gradient descent can fit the training data if the width is large enough.

Furthermore, in order to study the margin, ensuring the training data can be separated is inevitable; otherwise, there is no positive margin between the data and decision boundary.

Our Contribution.

Similar to linear models, for homogeneous models, only the direction of parameter θ is important for making predictions, and one can see that the margin γ(θ) scales linearly with θ L 2 , when fixing the direction of θ.

To compare margins among θ in different directions, it makes sense to study the normalized margin,γ(θ) := γ(θ)/ θ L 2 .

In this paper, we focus on the training dynamics of the network after t 0 (recall that t 0 is a time that the training loss is less than the threshold).

Our theoretical results can answer the following questions regarding the normalized margin.

First, how does the normalized margin change during training?

The answer may seem complicated since one can easily come up with examples in whichγ increases or decreases in a short time interval.

However, we can show that the overall trend of the normalized margin is to increase in the following sense: there exists a smoothed version of the normalized margin, denoted asγ,such that (1) |γ −γ| → 0 as t → ∞; and (2)γ is non-decreasing for t > t 0 .

Second, how large is the normalized margin at convergence?

To answer this question, we formulate a natural constrained optimization problem which aims to directly maximize the margin.

We show that every limit point of {θ(t)/ θ(t) 2 : t > 0} is along the direction of a KKT point of the maxmargin problem.

This indicates that gradient descent/gradient flow performs margin maximization implicitly in deep homogeneous networks.

This result can be seen as a significant generalization of previous works (Soudry et al., 2018a; b; Ji and Telgarsky, 2019a; Gunasekar et al., 2018b ) from linear classifiers to homogeneous classifiers.

As by-products of the above results, we derive tight asymptotic convergence/growth rates of the loss and weights.

It is shown in (Soudry et al., 2018a; b; Ji and Telgarsky, 2018 ) that the loss decreases Figure 1: (a) Training CNNs with and without bias on MNIST, using SGD with learning rate 0.01.

The training loss (left) decreases over time, and the normalized margin (right) keeps increasing after the model is fitted, but the growth rate is slow (≈ 1.8 × 10 −4 after 10000 epochs).

(b) Training CNNs with and without bias on MNIST, using SGD with the loss-based learning rate scheduler.

The training loss (left) decreases exponentially over time (< 10 −800 after 9000 epochs), and the normalized margin (right) increases rapidly after the model is fitted (≈ 1.2 × 10 −3 after 10000 epochs, 10× larger than that of SGD with learning rate 0.01).

Experimental details are in Appendix J.

at the rate of O(1/t), the weight norm grows as O(log t) for linear logistic regression.

In this work, we generalize the result by showing that the loss decreases at the rate of O(1/(t(log t) 2−2/L )) and the weight norm grows as O((log t) 1/L ) for homogeneous neural networks with exponential loss, logistic loss, or cross-entropy loss.

Experiments.

The main practical implication of our theoretical result is that training longer can enlarge the normalized margin.

To justify this claim empiricaly, we train CNNs on MNIST and CIFAR-10 with SGD (see Section J.1).

Results on MNIST are presented in Figure 1 .

For constant step size, we can see that the normalized margin keeps increasing, but the growth rate is rather slow (because the gradient gets smaller and smaller).

Inspired by our convergence results for gradient descent, we use a learning rate scheduling method which enlarges the learning rate according to the current training loss, then the training loss decreases exponentially faster and the normalized margin increases significantly faster as well.

For feedforward neural networks with ReLU activation, the normalized margin on a training sample is closely related to the L 2 -robustness (the L 2 -distance from the training sample to the decision boundary).

Indeed, the former divided by a Lipschitz constant is a lower bound for the latter.

For example, the normalized margin is a lower bound for the L 2 -robustness on fully-connected networks with ReLU activation (see, e.g., Theorem 4 in (Sokolic et al., 2017) ).

This fact suggests that training longer may have potential benefits on improving the robustness of the model.

In our experiments, we observe noticeable improvements of L 2 -robustness on both training and test sets (see Section J.2).

Implicit Bias in Training Linear Classifiers.

For linear logistic regression on linearly separable data, Soudry et al. (2018a; b) showed that full-batch gradient descent converges in the direction of the max L 2 -margin solution of the corresponding hard-margin Support Vector Machine (SVM).

Subsequent works extended this result in several ways: Nacson et al. (2018) extended the results to the case of stochastic gradient descent; Gunasekar et al. (2018a) considered other optimization methods; Nacson et al. (2019b) considered other loss functions including those with poly-exponential tails; Ji and Telgarsky (2018) characterized the convergence of weight direction without assuming separability; Ji and Telgarsky (2019b) proved a tighter convergence rate for the weight direction.

Those results on linear logistic regression have been generalized to deep linear networks.

Ji and Telgarsky (2019a) showed that the product of weights in a deep linear network with strictly decreasing loss converges in the direction of the max L 2 -margin solution.

Gunasekar et al. (2018b) showed more general results for gradient descent on linear fully-connected and convolutional networks with exponential loss, under various assumptions on the convergence of the loss and gradient direction.

Non-smooth Analysis.

For a locally Lipschitz function f : X → R, the Clarke's subdifferential (Clarke, 1975; Clarke et al., 2008; Davis et al., 2019 ) at x ∈ X is the convex set ∂

• f (x) := conv {lim k→∞ ∇f (x k ) : x k → x, f is differentiable at x k }.

For brevity, we say that a function z : I → R d on the interval I is an arc if z is absolutely continuous for any compact sub-interval of I. For an arc z, z (t) (or dz dt (t)) stands for the derivative at t if it exists.

Following the terminology in (Davis et al., 2019) , we say that a locally Lipschitz function f :

Binary Classification.

Let Φ be a neural network, assumed to be parameterized by θ.

The output of Φ on an input x ∈ R dx is a real number Φ(θ; x), and the sign of Φ(θ; x) stands for the classification result.

A dataset is denoted by D = {(x n , y n ) : n ∈ [N ]}, where x n ∈ R dx stands for a data input and y n ∈ {±1} stands for the corresponding label.

For a loss function : R → R, we define the training loss of Φ on the dataset D to be L(θ) := N n=1 (y n Φ(θ; x n )).

Gradient Descent.

We consider the process of training this neural network Φ with either gradient descent or gradient flow.

For gradient descent, we assume the training loss L(θ) is C 2 -smooth and describe the gradient descnet process as θ(t + 1) = θ(t) − η(t)∇L(θ(t)), where η(t) is the learning rate at time t and ∇L(θ(t)) is the gradient of L at θ(t).

Gradient Flow.

For gradient flow, we do not assume the differentibility but only some regularity assumptions including locally Lipschitz.

Gradient flow can be seen as gradient descent with infinitesimal step size.

In this model, θ changes continuously with time, and the trajectory of parameter θ during training is an arc θ : [0, +∞) → R d , t → θ(t) that satisfies the differential inclusion

is actually a C 1 -smooth function, the above differential inclusion reduces to dθ(t) dt = −∇L(θ(t)) for all t ≥ 0, which corresponds to the gradient flow with differential in the usual sense.

In this section, we first state our results for gradient flow and gradient descent on homogeneous models with exponential loss (q) := e −q for simplicity of presentation.

Due to space limit, we defer the more general results which hold for a large family of loss functions (including logistic loss and cross-entropy loss) to Appendix A and F.

Gradient Flow.

For gradient flow, we assume the following: (A1). (Regularity).

For any fixed x, Φ( · ; x) is locally Lipschitz and admits a chain rule; (A2). (Homogeneity).

There exists L > 0 such that ∀α > 0 :

(A4). (Separability).

There exists a time t 0 such that L(θ(t 0 )) < 1.

(A1) is a technical assumption about the regularity of the network output.

As shown in (Davis et al., 2019) , the output of almost every neural network admits a chain rule (as long as the neural network is composed by definable pieces in an o-minimal structure, e.g., ReLU, sigmoid, LeakyReLU).

(A2) assumes the homogeneity, the main property we assume in this work. (A3), (A4) correspond to the two conditions introduced in Section 1.

The exponential loss in (A3) is main focus of this section, and more general results are in Appendix A and F. (A4) is a separability assumption: the condition L(θ(t 0 )) < 1 ensures that (y n Φ(θ(t 0 ); x n )) < 1 for all n ∈ [N ], and thus y n Φ(θ(t 0 ); x n ) > 0, meaning that Φ classifies every x n correctly.

Gradient Descent.

For gradient descent, we assume (A2), (A3), (A4) similarly as for gradient flow, and the following two assumptions (S1) and (S5).

(S5). (Learning rate condition, Informal).

η(t) = η 0 for a sufficiently small constant η 0 .

In fact, η(t) is even allowed to be as large as

.

See Appendix E.1 for the details.

(S5) is natural since deep neural networks are usually trained with constant learning rates. (S1) ensures the smoothness of Φ, which is often assumed in the optimization literature in order to analyze gradient descent.

While (S1) does not hold for neural networks with ReLU, it does hold for neural networks with smooth homogeneous activation such as the quadratic activation φ(x) := x 2 (Li et al., 2018b; Du and Lee, 2018) or powers of ReLU φ(x) := ReLU(x) α for α > 2 (Zhong et al., 2017; Klusowski and Barron, 2018; Li et al., 2019) .

The margin for a single data point (x n , y n ) is defined to be q n (θ) := y n Φ(θ; x n ), and the margin for the entire dataset is defined to be q min (θ) := min n∈[N ] q n (θ).

By homogenity, the margin q min (θ) scales linearly with θ L 2 for any fixed direction since q min (cθ) = c L q min (θ).

So we consider the normalized margin defined as below:

We say f is an -additive approximation for the normalized margin ifγ − ≤ f ≤γ, and cmultiplicative approximation if cγ ≤ f ≤γ.

Gradient Flow.

Our first result is on the overall trend of the normalized marginγ(θ(t)).

For both gradient flow and gradient descent, we identify a smoothed version of the normalized margin, and show that it is non-decreasing during training.

More specifically, we have the following theorem for gradient flow.

2 )-additive approximation functionγ(θ) for the normalized margin such that the following statements are true for gradient flow:

3.

L(θ(t))

→ 0 and θ(t) 2 → ∞ as t → +∞; therefore, |γ(θ(t)) −γ(θ(t))| → 0.

More concretely, the functionγ(θ) in Theorem 4.1 is defined as

Note that the only difference betweenγ(θ) andγ(θ) is that q min (θ) inγ(θ) is replaced by log

, where LSE(a 1 , . . .

, a N ) = log(exp(a 1 ) + · · · + exp(a N )) is the LogSumExp function.

This is indeed a very natural idea, and previous works on linear models (e.g., (Telgarsky, 2013; Nacson et al., 2019b) ) also approximate q min with LogSumExp in the analysis of margin.

It is easy to see

e an ≤ N e amax holds for a max = max{a 1 , . . .

, a N }, so a max ≤ LSE(a 1 , . . .

, a N ) ≤ a max + log N ; combining this with the definition ofγ(θ) gives

Gradient Descent.

For gradient descent, Theorem 4.1 holds similarly with a slightly different functionγ(θ) that approximatesγ(θ) multiplicatively rather than additively.

Theorem 4.2 (Corollary of Theorem E.2).

Under assumptions (S1), (A2) -(A4), (S5), there exists an (1 − O(1/(log 1 L )))-multiplicative approximation functionγ(θ) for the normalized margin such that the following statements are true for gradient descent:

1.

For all t > t 0 ,γ(θ(t + 1)) ≥γ(θ(t)); 2.

For all t > t 0 , eitherγ(θ(t + 1)) >γ(θ(t)) or

3.

L(θ(t))

→ 0 and θ(t) 2 → ∞ as t → +∞; therefore, |γ(θ(t)) −γ(θ(t))| → 0.

Due to the discreteness of gradient descent, the explicit formula forγ(θ) is somewhat technical, and we refer the readers to Appendix E for full details.

Convergence Rates.

It is shown in Theorem 4.1, 4.2 that L(θ(t)) → 0 and θ(t) 2 → ∞. In fact, with a more refined analysis, we can prove tight loss convergence and weight growth rates using the monotonicity of normalized margins.

Theorem 4.3 (Corollary of Theorem A.10 and E.5).

For gradient flow under assumptions (A1) -(A4) or gradient descent under assumptions (S1), (A2) -(A4), (S5), we have the following tight bounds for training loss and weight norm:

where T = t for gradient flow and T = t−1 τ =t0 η(τ ) for gradient descent.

For gradient flow,γ is upper-bounded byγ ≤γ ≤ sup{q n (θ) : θ 2 = 1}. Combining this with Theorem 4.1 and the monotone convergence theorem, it is not hard to see that lim t→+∞γ (θ(t)) and lim t→+∞γ (θ(t)) exist and equal to the same value.

Using a similar argument, we can draw the same conclusion for gradient descent.

To understand the implicit regularization effect, a natural question arises: what optimality property does the limit of normalized margin have?

To this end, we identify a natural constrained optimization problem related to margin maximization, and prove that θ(t) directionally converges to its KKT points, as shown below.

We note that we can extend this result to the finite time case, and show that gradient flow or gradient descent passes through an approximate KKT point after a certain amount of time.

See Theorem A.9 in Appendix A and Theorem E.4 in Appendix E for the details.

We will briefly review the definition of KKT points and approximate KKT points for a constraint optimization problem in Appendix C.1.

Theorem 4.4 (Corollary of Theorem A.8 and E.3).

For gradient flow under assumptions (A1) -(A4) or gradient descent under assumptions (S1), (A2) -(A4), (S5), any limit pointθ of

: t ≥ 0 is along the direction of a KKT point of the following constrained optimization problem (P):

That is, for any limit pointθ, there exists a scaling factor α > 0 such that αθ satisfies Karush-KuhnTucker (KKT) conditions of (P).

Minimizing (P) over its feasible region is equivalent to maximizing the normalized margin over all possible directions.

The proof is as follows.

Note that we only need to consider all feasible points θ with q min (θ) > 0.

For a fixed θ, αθ is a feasible point of (P) iff α ≥ q min (θ) −1/L .

Thus, the minimum objective value over all feasible points of (P) in the direction of θ is

−2/L .

Taking minimum over all possible directions, we can conclude that if the maximum normalized margin isγ * , then the minimum objective of (P) is

It can be proved that (P) satisfies the Mangasarian-Fromovitz Constraint Qualification (MFCQ) (See Lemma C.7).

Thus, KKT conditions are first-order necessary conditions for global optimality.

For linear models, KKT conditions are also sufficient for ensuring global optimality; however, for deep homogenuous networks, q n (θ) can be highly non-convex.

Indeed, as gradient descent is a first-order optimization method, if we do not make further assumptions on q n (θ), then it is easy to construct examples that gradient descent does not lead to a normalized margin that is globally optimal.

Thus, proving the convergence to KKT points is perhaps the best we can hope for in our setting, and it is an interesting future work to prove stronger convergence results with further natural assumptions.

Moreover, we can prove the following corollary, which characterizes the optimality of the normalized margin using SVM with Neural Tangent Kernel (NTK, introduced in (Jacot et al., 2018)) defined at limit points.

The proof is deferred to Appendix C.6.

Corollary 4.5 (Corollary of Theorem 4.4).

Assume (S1).

Then for gradient flow under assumptions (A2) -(A4) or gradient descent under assumptions (A2) -(A4), (S5), any limit pointθ of {θ(t)/ θ(t) 2 : t ≥ 0} is along the max-margin direction for the hard-margin SVM with kernel

.

That is, for some α > 0, αθ is the optimal solution for the following constrained optimization problem:

If we assume (A1) instead of (S1) for gradient flow, then there exists a mapping h(x) ∈ ∂ • Φ x (θ) such that the same conclusion holds for Kθ(x, x ) = h(x), h(x ) .

The above results can be extended to other settings.

Here we discuss them in the context of gradient flow for simplicity, but it is not hard to generalize them to gradient descent.

Other Binary Classification Loss.

The results on exponential loss can be generalized to a much broader class of binary classification loss.

The class includes the logistic loss which is one of the most popular loss functions, (q) = log(1 + e −q ).

The function class also includes other losses with exponential tail, e.g., (q) = e −q 3 , (q) = log(1 + e −q

3 ).

For all those loss functions, we can use its inverse function −1 to define the smoothed normalized margin as follows

Theorem 4.1 and 4.4 continue to hold for gradient flow.

See Appendix A for the details.

Cross-entropy Loss.

In multi-class classification, we can define q n to be the difference between the classification score for the true label and the maximum score for the other labels, then the margin

can be similarly defined as before.

In Appendix F, we define the smoothed normalized margin for cross-entropy loss to be the same as that for logistic loss (See Remark A.4), and we show that Theorem 4.1 and Theorem 4.4 still hold (but with a slightly different definition of (P)) for gradient flow.

Multi-homogeneous Models.

Some neural networks indeed possess a stronger property than homogeneity, which we call multi-homogeneity.

For example, the output of a CNN (without bias terms) is 1-homogeneous with respect to the weights of each layer.

In general, we say that a neural network Φ(θ; x) with θ = (w 1 , . . .

, One can easily see that that (k 1 , . . .

, k m )-homogeneity implies L-homogeneity, where L = m i=1 k i , so our previous analysis for homogeneous models still applies to multi-homogeneous models.

But it would be better to define the normalized margin for multi-homogeneous model as

In this case, the smoothed approximation ofγ for general binary classification loss (under some conditions) can be similarly defined:

It can be shown thatγ is also non-decreasing during training when the loss is small enough (Appendix G).

In the case of cross-entropy loss, we can still defineγ by (5) while (·) is set to the logistic loss in the formula.

In this section, we present a proof sketch in the case of gradient flow on homogeneous model with exponential loss to illustrate our proof ideas.

Due to space limit, the proof for the main theorems on gradient flow and gradient descent in Section 4 are deferred to Appendix A and E respectively.

For convenience, we introduce a few more notations for a L-homogeneous neural network Φ(θ; x).

∈ S d−1 to be the length and direction of θ.

For both gradient descent and gradient flow, θ is a function of time t. For convenience, we also view the functions of θ, including L(θ), q n (θ), q min (θ), as functions of t. So we can write L(t) := L(θ(t)), q n (t) := q n (θ(t)), q min (t) := q min (θ(t)).

Lemma 5.1 below is the key lemma in our proof.

It decomposes the growth of the smoothed normalized margin into the ratio of two quantities related to the radial and tangential velocity components of θ respectively.

We will give a proof sketch for this later in this section.

We believe that this lemma is of independent interest.

Lemma 5.1 (Corollary of Lemma B.1).

For a.e.

t > t 0 ,

Using Lemma 5.1, the first two claims in Theorem 4.1 can be directly proved.

For the third claim, we make use of the monotonicity of the margin to lower bound the gradient, and then show L → 0 and ρ → +∞. Recall thatγ is an O(ρ −L )-additive approximation forγ.

So this proves the third claim.

We defer the detailed proof to Appendix B.

To show Theorem 4.4, we first change the time measure to log ρ, i.e., now we see t as a function of log ρ.

So the second inequality in Lemma 5.1 can be rewritten as

Integrating on both sides and noting thatγ is upper-bounded, we know that there must be many instant log ρ such that dθ d log ρ 2 is small.

By analyzing the landscape of training loss, we show that these points are "approximate" KKT points.

Then we show that every convergent sub-sequence of {θ(t) : t ≥ 0} can be modified to be a sequence of "approximate" KKT points which converges to the same limit.

Then we conclude the proof by applying a theorem from (Dutta et al., 2013) to show that the limit of this convergent sequence of "approximate" KKT points is a KKT point.

We defer the detailed proof to Appendix C. Now we give a proof sketch for Lemma 5.1, in which we derive the formula ofγ step by step.

In the proof, we obtain several clean close form formulas for several relevant quantities, by using the chain rule and Euler's theorem for homogenuous functions extensively.

Proof Sketch of Lemma 5.1.

For ease of presentation, we ignore the regularity issues of taking derivatives in this proof sketch.

We start from the equation

which follows from the chain rule (see also Lemma H.3).

Then we note that dθ dt can be decomposed into two parts: the radial component v :=θθ

where the last equality is due to ∂ • q n , θ = Lq n by homogeneity of q n .

This equation is sometimes called Euler's theorem for homogeneous functions (see Theorem B.2).

For differentiable functions, it can be easily proved by taking the derivative over c on both sides of q n (cθ) = c L q n (θ) and letting c = 1.

With (6), we can lower bound

where the last inequality uses the fact that e −qmin ≤ L. (7) also implies that

dt on the leftmost and rightmost sides, we have

, where the LHS is exactly d dt logγ.

In this paper, we analyze the dynamics of gradient flow/descent of homogeneous neural networks under a minimal set of assumptions.

The main technical contribution of our work is to prove rigorously that for gradient flow/descent, the normalized margin is increasing and converges to a KKT point of a natural max-margin problem.

Our results leads to some natural further questions:

• Can we generalize our results for gradient descent on smooth neural networks to nonsmooth ones?

In the smooth case, we can lower bound the decrement of training loss by the gradient norm squared, multiplied by a factor related to learning rate.

However, in the non-smooth case, no such inequality is known in the optimization literature, and it is unclear what kind of natural assumption can make it holds.

• Can we make more structural assumptions on the neural network to prove stronger results?

In this work, we use a minimal set of assumptions to show that the convergent direction of parameters is a KKT point.

A potential research direction is to identify more key properties of modern neural networks and show that the normalized margin at convergence is locally or globally optimal (in terms of optimizing (P)).

• Can we extend our results to neural networks with bias terms?

In our experiments, the normalized margin of the CNN with bias also increases during training despite that its output is non-homogeneous.

It is very interesting (and technically challenging) to provide a rigorous proof for this fact.

In this section, we state our results for a broad class of binary classification loss.

A major consequence of this generalization is that the logistic loss, one of the most popular loss functions, (q) = log(1 + e −q ) is included.

The function class also includes other losses with exponential tail, e.g., (q) = e −q 3 , (q) = log(1 + e −q

3 ).

We first focus on gradient flow.

We assume (A1), (A2) as we do for exponential loss.

For (A3), (A4), we replace them with two weaker assumptions (B3), (B4).

All the assumptions are listed below:

(A1). (Regularity).

For any fixed x, Φ( · ; x) is locally Lipschitz and admits a chain rule;

(A1) and (A2) remain unchanged. (B3) is satisfied by exponential loss (q) = e −q (with f (q) = q) and logistic loss (q) = log(1 + e −q ) (with f (q) = − log log(1 + e −q )). (B4) are essentially the same as (A4) but (B4) uses a threshold value that depends on the loss function.

Assuming (B3), it is easy to see that (B4) ensures the separability of data since (q n ) < e −f (b f ) implies q n > b f ≥ 0.

For logistic loss, we can set b f = 0 (see Remark A.2).

So the corresponding threshold value in (B4) is (0) = log 2.

Now we discuss each of the assumptions in (B3). (B3.1) is a natural assumption on smoothness. (B3.2) requires (·) to be monotone decreasing, which is also natural since (·) is used for binary classification.

The rest of two assumptions in (B3) characterize the properties of (q) when q is large enough. (B3.3) is an assumption that appears naturally from the proof.

For exponential loss, f (q)q = q is always non-decreasing, so we can set b f = 0.

In (B3.4), the inverse function g is defined.

It is guaranteed by (B3.1) and (B3.2) that g always exists and g is also C 1 -smooth.

Though (B3.4) looks very complicated, it essentially says that f (Θ(q)) = Θ(f (q)), g (Θ(q)) = Θ(g (q)) as q → ∞. (B3.4) is indeed a technical assumption that enables us to asymptotically compare the loss or the length of gradient at different data points.

It is possible to base our results on weaker assumptions than (B3.4), but we use (B3.4) for simplicity since it has already been satisfied by many loss functions such as the aforementioned examples.

We summarize the corresponding f, g and b f for exponential loss and logistic loss below: Remark A.1.

Exponential loss (q) = e −q satisfies (B3) with

Remark A.2.

Logistic loss (q) = log(1 + e −q ) satisfies (B3) with

The proof for Remark A.1 is trivial.

For Remark A.2, we give a proof below.

Proof for Remark A.2.

By simple calculations, the formulas for

Thus, f (q)q is a strictly increasing function on R. As b f is required to be non-negative, we set b f = 0.

For proving that f (q)q → +∞ and (B4), we only need to notice that f (q) ∼ e −q 1·e −q = 1 and g (x) = 1/f (g(x)) ∼ 1.

For a loss function (·) satisfying (B3), it is easy to see from (B3.2) that its inverse function −1 (·) must exist.

For this kind of loss functions, we define the smoothed normalized margin as follows: Definition A.3.

For a loss function (·) satisfying (B3), the smoothed normalized marginγ(θ) of θ is defined asγ

where −1 (·) is the inverse function of (·) and ρ := θ 2 .

Remark A.4.

For logistic loss (q) = log(

, which is the same as (3).

Now we give some insights on how wellγ(θ) approximatesγ(θ) using a similar argument as in Section 4.2.

Using the LogSumExp function, the smoothed normalized marginγ(θ) can also be written asγ

LSE is a (log N )-additive approximation for max.

So we can roughly approximateγ(θ) bỹ

Note that (B3.3) is crucial to make the above approximation reasonable.

Similar to exponential loss, we can show the following lemma asserting thatγ is a good approximation ofγ.

Lemma A.5.

Assuming (B3) 3 , we have the following properties about the margin: qmin) .

Combining (a) and the monotonicity of g(·), we further have

3).

Also note that there exists a constant B 0 such thatγ(θ m ) ≤ B 0 for all m sinceγ is continuous on the unit sphere S d−1 .

So we have

where the first inequality follows since ξ m ≤ f (q min (θ m )).

Together with (8), we have |γ(

Remark A.6.

For exponential loss, we have already shown in Section 4.2 thatγ(θ) is an O(ρ −L )-additive approximation forγ(θ).

For logistic loss, it follows easily from g (q) = Θ(1) and (b)

Now we state our main theorems.

For the monotonicity of the normalized margin, we have the following theorem.

The proof is provided in Appendix B.

Theorem A.7.

Under assumptions (A1), (A2), (B3) 4 , (B4), the following statements are true for gradient flow:

For the normalized margin at convergence, we have two theorems, one for infinite-time limiting case, and the other being a finite-time quantitative result.

Their proofs can be found in Appendix C. As in the exponential loss case, we define the constrained optimization problem (P) as follows:

First, we show the directional convergence of θ(t) to a KKT point of (P).

Theorem A.8.

Consider gradient flow under assumptions (A1), (A2), (B3), (B4).

For every limit pointθ of θ (t) :

Second, we show that after finite time, gradient flow can pass through an approximate KKT point.

Theorem A.9.

Consider gradient flow under assumptions (A1), (A2), (B3), (B4).

For any , δ > 0, there exists r := Θ(log δ −1 ) and ∆ :

For the definitions for KKT points and approximate KKT points, we refer the readers to Appendix C.1 for more details.

With a refined analysis, we can also provide tight rates for loss convergence and weight growth.

The proof is given in Appendix D.

Theorem A.10.

Under assumptions (A1), (A2), (B3), (B4), we have the following tight rates for loss convergence and weight growth:

Applying Theorem A.10 to exponential loss and logistic loss, in which g(x) = Θ(x), we have the following corollary:

In this section, we consider gradient flow and prove Theorem A.7.

We assume (A1), (A2), (B3), (B4) as mentioned in Appendix A.

We follow the notations in Section 5 to define ρ := θ 2 andθ := θ θ 2 ∈ S d−1 , and sometimes we view the functions of θ as functions of t.

To prove the first two propositions, we generalize our key lemma (Lemma 5.1) to general loss.

Lemma B.1.

Forγ defined in Definition A.3, the following holds for all t > t 0 ,

Before proving Lemma B.1, we review two important properties of homogeneous functions.

Note that these two properties are usually shown for smooth functions.

By considering Clarke's subdifferential, we can generalize it to locally Lipschitz functions that admit chain rules:

Proof.

Let D be the set of points x such that F is differentiable at x. According to the definition of Clarke's subdifferential, for proving (a), it is sufficient to show that

Fix x k ∈ D. Let U be a neighborhood of x k .

By definition of homogeneity, for any h ∈ R d and any y ∈ U \ {x k },

Taking limits y → x k on both sides, we know that the LHS converges to 0 iff the RHS converges to 0.

Then by definition of differetiability and gradient, F is differentiable at αx k iff it is differentiable at x k , and ∇F (αx k ) = α k−1 h iff ∇F (x k ) = h. This proves (10).

To prove (b), we fix

Taking derivative with respect to α on both sides (for differentiable points), we have

holds for a.e.

α > 0.

Pick an arbitrary α > 0 making (11) hold.

Then by (a), (11) is equivalent to

Applying Theorem B.2 to homogeneous neural networks, we have the following corollary: Corollary B.3.

Under the assumptions (A1) and (A2), for any θ ∈ R d and x ∈ R dx ,

where Φ x (θ) = Φ(θ; x) is the network output for a fixed input x.

Corollary B.3 can be used to derive an exact formula for the weight growth during training.

Theorem B.4.

For a.e.

t ≥ 0,

Proof.

The proof idea is to use Corollary B.3 and chain rules (See Appendix H for chain rules in Clarke's sense).

Applying the chain rule on t → ρ 2 = θ 2 2 yields 1 2

By Corollary B.3, θ, h n = Lq n , and thus

For convenience, we define ν(t) := N n=1 e −f (qn) f (q n )q n for all t ≥ 0.

Then Theorem B.4 can be rephrased as

Combining this with the definitions of ν(t) and L gives

Proof for Lemma B.1.

Note that

ρ 2 by Theorem B.4.

Then it simply follows from Lemma B.5 that d dt log ρ > 0 for a.e.

t > t 0 .

For the second inequality, we first prove

exists and is always positive for all t ≥ t 0 , which proves the existence of logγ.

By the chain rule and Lemma B.5, we have

On the one hand,

for a.e.

t > 0 by Lemma H.3; on the other hand, Lν(t) = θ,

by Theorem B.4.

Combining these together yields

By the chain rule,

To prove the third proposition, we prove the following lemma to show that L → 0 by giving an upper bound for L. Since L can never be 0 for bounded ρ, L → 0 directly implies ρ → +∞. For showing |γ −γ| → 0, we only need to apply (c) in Lemma A.5, which shows this when L → 0.

Lemma B.6.

For all t > t 0 ,

Therefore, L(t) → 0 and ρ(t) → +∞ as t → ∞.

Proof for Lemma B.6.

By Lemma H.3 and Theorem B.4,

Using Lemma B.5 to lower bound ν and replacing ρ with g(log

L ) 2 L, where the last inequality uses the monotonicity ofγ.

So the following holds for a.e.

Integrating on both sides from t 0 to t, we can conclude that

Note that 1/L is non-decreasing.

If 1/L does not grow to +∞, then neither does G(1/L).

But the RHS grows to +∞, which leads to a contradiction.

So L → 0.

To make L → 0, q min must converge to +∞.

So ρ → +∞.

In this section, we analyze the convergent direction of θ and prove Theorem A.8 and A.9, assuming (A1), (A2), (B3), (B4) as mentioned in Section A.

We follow the notations in Section 5 to define ρ := θ 2 andθ := θ θ 2 ∈ S d−1 , and sometimes we view the functions of θ as functions of t.

We first review the definition of Karush-Kuhn-Tucker (KKT) conditions for non-smooth optimization problems following from (Dutta et al., 2013) .

Consider the following optimization problem (P) for x ∈ R d :

where f, g 1 , . . . , g n : R d → R are locally Lipschitz functions.

We say that x ∈ R d is a feasible point of (P) if x satisfies g n (x) ≤ 0 for all n ∈ [N ].

Definition C.1 (KKT Point).

A feasible point x of (P) is a KKT point if x satisfies KKT conditions: there exists λ 1 , . . .

, λ N ≥ 0 such that

It is important to note that a global minimum of (P) may not be a KKT point, but under some regularity assumptions, the KKT conditions become a necessary condition for global optimality.

The regularity condition we shall use in this paper is the non-smooth version of Mangasarian-Fromovitz Constraint Qualification (MFCQ) (see, e.g., the constraint qualification (C.Q.5) in (Giorgi et al., 2004) ): Definition C.2 (MFCQ).

For a feasible point x of (P), (P) is said to satisfy MFCQ at x if there exists v ∈ R d such that for all n ∈ [N ] with g n (x) = 0,

Following from (Dutta et al., 2013), we define an approximate version of KKT point, as shown below.

Note that this definition is essentially the modified -KKT point defined in their paper, but these two definitions differ in the following two ways: (1) First, in their paper, the subdifferential is allowed to be evaluated in a neighborhood of x, so our definition is slightly stronger; (2) Second, their paper fixes δ = 2 , but in our definition we make them independent.

As shown in (Dutta et al., 2013) , ( , δ)-KKT point is an approximate version of KKT point in the sense that a series of ( , δ)-KKT points can converge to a KKT point.

We restate their theorem in our setting: Theorem C.4 (Corollary of Theorem 3.6 in (Dutta et al., 2013) ).

Let {x k ∈ R d : k ∈ N} be a sequence of feasible points of (P), { k > 0 : k ∈ N} and {δ k > 0 : k ∈ N} be two sequences.

x k is an ( k , δ k )-KKT point for every k, and k → 0, δ k → 0.

If x k → x as k → +∞ and MFCQ holds at x, then x is a KKT point of (P).

Recall that for a homogeneous neural network, the optimization problem (P) is defined as follows:

Using the terminologies and notations in Appendix C.1, the objective and constraints are f (x) = 1 2 x 2 2 and g n (x) = 1 − q n (x).

The KKT points and approximate KKT points for (P) are defined as follows: Definition C.5 (KKT Point of (P)).

A feasible point θ of (P) is a KKT point if there exist λ 1 , . . .

, λ N ≥ 0 such that

2.

∀n ∈ [N ] : λ n (q n (θ) − 1) = 0.

Definition C.6 (Approximate KKT Point of (P)).

A feasible point θ of (P) is an ( , δ)-KKT point of (P) if there exists λ 1 , . . .

, λ N ≥ 0 such that

By the homogeneity of q n , it is easy to see that (P) satisfies MFCQ, and thus KKT conditions are first-order necessary condition for global optimality.

Lemma C.7.

(P) satisfies MFCQ at every feasible point θ.

Proof.

Take v := θ.

For all n ∈ [N ] satisfying q n = 1, by homogeneity of q n ,

to be the cosine of the angle between θ and dθ dt .

Here β(t) is only defined for a.e.

t > 0.

Since q n is locally Lipschitz, it can be shown that q n is (globally) Lipschitz on the compact set S d−1 , which is the unit sphere in R d .

Define

For showing Theorem A.8 and Theorem A.9, we first prove Lemma C.8.

In light of this lemma, if we aim to show that θ is along the direction of an approximate KKT point, we only need to show β → 1 (which makes → 0) and L → 0 (which makes δ → 0).

Lemma C.8.

Let C 1 , C 2 be two constants defined as

Proof.

Let h(t) := dθ dt (t) for a.e.

t > 0.

By the chain rule, there exist h 1 , . . .

, h N such that

Thenθ can be shown to be an ( , δ)-KKT point by the monotonicityγ(t) ≥γ(t 0 ) for t > t 0 .

Proof of (12).

From our construction,

where the last equality is by Lemma A.5.

Proof for (13).

According to our construction,

Note that h 2 ≥ h,θ = Lν/ρ.

By Lemma B.5 and Lemma D.1, we have

where the last inequality uses f (γρ L ) = log 1 L and L ≥ e −f (qmin) .

Combining these gives

If q n > q min , then by the mean value theorem there exists ξ n ∈ (q min , q n ) such that

where the second inequality uses q −2/L min ρ 2 ≤γ −2/L by Lemma A.5 and the fact that the function x → e −x x on (0, +∞) attains the maximum value e at x = 1.

By Theorem A.7, we have already known that L → 0.

So it remains to bound β(t).

For this, we first prove the following lemma to bound the integral of β(t).

Lemma C.9.

For all t 2 > t 1 ≥ t 0 ,

.

By the chain rule,

where the last equality follows from the definition of β.

Combining 14 and 15, we have

Integrating on both sides from t 1 to t 2 proves the lemma.

A direct corollary of Lemma C.9 is the upper bound for the minimum β 2 − 1 within a time interval:

Corollary C.10.

For all t 2 > t 1 ≥ t 0 , then there exists t * ∈ (t 1 , t 2 ) such that

Under review as a conference paper at ICLR 2020

Proof.

Denote the RHS as C. Assume to the contrary that β(τ ) −2 − 1 > C for a.e.

τ ∈ (t 1 , t 2 ).

By Lemma B.1, log ρ(τ ) > 0 for a.e.

τ ∈ (t 1 , t 2 ).

Then by Lemma C.9, we have

, which leads to a contradiction.

In the rest of this section, we present both asymptotic and non-asymptotic analyses for the directional convergence by using Corollary C.10 to bound β(t).

We first prove an auxiliary lemma which gives an upper bound for the change ofθ.

Lemma C.11.

For a.e.

t > t 0 , dθ dt

Proof.

Observe that

.

It is sufficient to bound

.

By the chain rule, there exists h 1 , . . .

,

Note that every summand is positive.

By Lemma A.5, q n is lower-bounded by q n ≥ q min ≥ g(log 1 L ), so we can replace q n with g(log 1 L ) in the above inequality.

Combining with the fact that

So we have

To prove Theorem A.8, we consider each limit pointθ/q min (θ) 1/L , and construct a series of approximate KKT points converging to it.

Thenθ/q min (θ) 1/L can be shown to be a KKT point by Theorem C.4.

The following lemma ensures that such construction exists.

Lemma C.12.

For every limit pointθ of θ (t) : t ≥ 0 , there exists a sequence of {t m : m ∈ N} such that t m ↑ +∞,θ(t m ) →θ, and β(t m ) → 1.

Proof.

Let { m > 0 : m ∈ N} be an arbitrary sequence with m → 0.

Now we construct {t m } by induction.

Suppose t 1 < t 2 < · · · < t m−1 have already been constructed.

Sinceθ is a limit point andγ(t) ↑γ ∞ (recall thatγ ∞ := lim t→+∞γ (t)), there exists s m > t m−1 such that

Let s m > s m be a time such that log ρ(s m ) = log ρ(s m ) + m .

According to Theorem A.7, log ρ → +∞, so s m must exist.

We construct t m ∈ (s m , s m ) to be a time that β(t m ) −2 − 1 ≤ Now we show that this construction meets our requirement.

It follows from β(t m ) −2 − 1 ≤ 2 m that β(t m ) ≥ 1/ 1 + 2 m → 1.

By Lemma C.11, we also know that

This completes the proof.

Proof of Theorem A.8.

Letθ :=θ/q min (θ) 1/L for short.

Let {t m : m ∈ N} be the sequence constructed as in Lemma C.12.

For each t m , define (t m ) and δ(t m ) as in Lemma C.8.

Then we know that θ(t m )/q min (t m ) 1/L is an ( (t m ), δ(t m ))-KKT point and θ(t m )/q min (t m ) 1/L →θ, (t m ) → 0, δ(t m ) → 0.

By Lemma C.7, (P) satisfies MFCQ.

Applying Theorem C.4 proves the theorem.

Proof of Theorem A.9.

.

Without loss of generality, we assume < √ 6 2 C 1 and δ < C 2 /f (b f ).

Let t 1 be the time such that log ρ(

= Θ(log δ −1 ) and t 2 be the time such that

.

By Corollary C.10, there exists t * ∈ (t 1 , t 2 ) such

1 .

Now we argue thatθ(t * ) is an ( , δ)-KKT point.

By Lemma C.8, we only need to show

For the first inequality, by assumption <

C.6 PROOF FOR COROLLARY 4.5

By the homogeneity of q n , we can characterize KKT points using kernel SVM.

Lemma C.13.

If θ * is KKT point of (P), then there exists h n ∈ ∂ • Φ xn (θ * ) for n ∈ [N ] such that 1 L θ * is an optimal solution for the following constrained optimization problem (Q):

Proof.

It is easy to see that (Q) is a convex optimization problem.

For θ = 2 L θ * , from Theorem B.2, we can see that y n θ, h n = 2q n (θ * ) ≥ 2 > 1, which implies Slater's condition.

Thus, we only need to show that 1 L θ * satisfies KKT conditions for (Q).

By the KKT conditions for (P), we can construct Proof.

By Theorem A.8, every limit pointθ is along the direction of a KKT point of (P).

Combining this with Lemma C.13, we know that every limit pointθ is also along the max-margin direction of (Q).

For smooth models, h n in (Q) is exactly the gradient ∇Φ xn (θ).

So, (Q) is the optimization problem for SVM with kernel Kθ(x, x ) = ∇Φ x (θ), ∇Φ x (θ) .

For non-smooth models, we can construct an arbitrary function h(x) ∈ ∂ • Φ x (θ) that ensures h(x n ) = h n .

Then, (Q) is the optimization problem for SVM with kernel Kθ(x, x ) = h(x), h(x ) .

In this section, we give proof for Theorem A.10, which gives tight bounds for loss convergence and weight growth under Assumption (A1), (A2), (B3), (B4).

Before proving Theorem A.10, we show some consequences of (B3.4).

Lemma D.1.

For f (·) and g(·), we have

Thus, g(x) = Θ(xg (x)), f (y) = Θ(yf (y)).

Proof.

To prove Item 1, it is sufficient to show that

To prove Item 2, we only need to notice that Item 1 implies yf (y) =

Recall that (B3.4) directly implies that f (Θ(x)) = Θ(f (x)) and g (Θ(x)) = Θ(g (x)).

Combining this with Lemma D.1, we have the following corollary:

Also, note that Lemma D.1 essentially shows that (log f (x)) = Θ(1/x) and (log g(x)) = Θ(1/x).

So log f (x) = Θ(log x) and log g(x) = Θ(log x), which means that f and g grow at most polynomially.

Corollary D.3.

f (x) = x Θ(1) and g(x) = x Θ(1) .

We follow the notations in Section 5 to define ρ := θ 2 andθ := θ θ 2 ∈ S d−1 , and sometimes we view the functions of θ as functions of t. And we use the notations B 0 , B 1 from Appendix C.3.

The key idea to prove Theorem A.10 is to utilize Lemma B.6, in which L(t) is bounded from above by

.

So upper bounding L(t) reduces to lower bounding G −1 .

In the following lemma, we obtain tight asymptotic bounds for G(·) and G −1 (·):

Lemma D.4.

For function G(·) defined in Lemma B.6 and its inverse function G −1 (·), we have the following bounds:

Proof.

We first prove the bounds for G(x), and then prove the bounds for G −1 (y).

On the other hand, for x ≥ exp(2b g ), we have

Bounding for G −1 (y).

Let x = G −1 (y) for y ≥ 0.

G(x) always has a finite value whenever x is finite.

So x → +∞ when y → +∞. According to the first part of the proof, we know that y = Θ g(log x) 2/L (log x) 2 x .

Taking logarithm on both sides and using Corollary D.3, we have log y = Θ(log x).

By Corollary D.2, g(log y) = g(Θ(log x)) = Θ(g(log x)).

Therefore,

For other bounds, we derive them as follows.

We first show that g(log

.

With this equivalence, we derive an upper bound for the gradient at each time t in terms of L, and take an integration to bound L(t) from below.

Now we have both lower and upper bounds for L(t).

Plugging these two bounds to g(log

gives the lower and upper bounds for ρ(t).

Proof for Theorem A.10.

We first prove the upper bound for L. Then we derive lower and upper bounds for ρ in terms of L, and use these bounds to give a lower bound for L. Finally, we plug in the tight bounds for L to obtain the lower and upper bounds for ρ in terms of t.

Upper Bounding L.

By Lemma B.6, we have

g(log t) 2/L t , which completes the proof.

.

Therefore, we have the following relationship between ρ L and g(log 1 L ):

Lower Bounding L. Let h 1 , . . .

, h N be a set of vectors such that h n ∈ ∂qn ∂θ and

By (17) and

Combining these two bounds together, it follows from Corollary D.2 that

By definition of G(·), this implies that there exists a constant c such that

for any L that is small enough.

We can complete our proof by applying Lemma D.4.

Bounding ρ in Terms of t. By (17) and the tight bound for (Θ(log t)) ).

Using Corollary D.2, we can conclude that ρ L = Θ(g(log t)).

In this section, we discretize our proof to prove similar results for gradient descent on smooth homogeneous models with exponential loss.

As usual, the update rule of gradient descent is defined as θ(t + 1) = θ(t) − η(t)∇L(t) (18) Here η(t) is the learning rate, and ∇L(t) := ∇L(θ(t)) is the gradient of L at θ(t).

The main difficulty for discretizing our previous analysis comes from the fact that the original version of the smoothed normalized marginγ(θ) := ρ −L log 1 L becomes less smooth when ρ → +∞. Thus, if we take a Taylor expansion forγ(θ(t + 1)) from the point θ(t), although one can show that the first-order term is positive as in the gradient flow analysis, the second-order term is unlikely to be bounded during gradient descent with a constant step size.

To get a smoothed version of the normalized margin that is monotone increasing, we need to define another one that is even smoother thanγ.

Technically, recall that dL dt = − ∇L 2 2 does not hold exactly for gradient descent.

However, if the smoothness can be bounded by s(t), then it is well-known that

By analyzing the landscape of L, one can easily find that the smoothness is bounded locally by O(L · polylog( 1 L )).

Thus, if we set η(t) to a constant or set it appropriately according to the loss, then this discretization error becomes negligible.

Using this insight, we define the new smoothed normalized marginγ in a way that it increases slightly slower thanγ during training to cancel the effect of discretization error.

As stated in Section 4.1, we assume (A2), (A3), (A4) similarly as for gradient flow, and two additional assumptions (S1) and (S5).

(A4). (Separability).

There exists a time t 0 such that L(θ(t 0 )) < 1.

Here H(L) is a function of the current training loss.

The explicit formula of H(L) is given below:

where C η is a constant, and κ(x), µ(x) are two non-decreasing functions.

For constant learning rate η(t) = η 0 , (S5) is satisfied when η 0 if sufficiently small.

Roughly speaking, C η κ(x) is an upper bound for the smoothness of L in a neighborhood of θ when x = L(θ).

And we set the learning rate η(t) to be the inverse of the smoothness multiplied by a factor µ(x) = o(1).

In our analysis, µ(x) can be any non-decreasing function that maps (0, L(t 0 )] to (0, 1/2] and makes the integral 1/2 0 µ(x)dx exist.

But for simplicity, we define µ(x) as

The value of C η will be specified later.

The definition of

where κ max := e (2−2/L)(ln(2−2/L)−1) .

The specific meaning of C η , κ(x) and µ(x) will become clear in our analysis.

Now we define the smoothed normalized margins.

As usual, we defineγ(θ) := log 1 L ρ L .

At the same time, we also defineγ

Here φ : (0, L(t 0 )]

→ (0, +∞) is constructed as follows.

Construct the first-order derivative of φ(x) as

.

And then we set φ(x) to be φ(x) = log log 1

It can be verified that φ(x) is well-defined and φ (x) is indeed the first-order derivative of φ(x).

Moreover, we have the following relationship amongγ,γ andγ.

Lemma E.1.γ(θ) is well-defined for L(θ) ≤ L(t 0 ) and has the following properties:

Proof.

First we verify thatγ is well-defined.

To see this, we only need to verify that

exists for all x ∈ (0, L(t 0 )], then it is trivial to see that φ (w) is indeed the derivative of φ(w) by

Note that I(x) exists for all x ∈ (0, L(t 0 )] as long as I(x) exists for a small enough x > 0.

By definition, it is easy to verify that r(w) := 1+2(1+λ(w))µ(w) w log 1 w is decreasing when w is small enough.

Thus, for a small enough w > 0, we have

So we have the following for small enough x:

This proves the existence of I(x).

.

By Lemma A.5,γ(θ) ≤γ(θ), so we only need to prove thatγ(θ) <γ(θ).

To see this, note that for all w ≤ L(t 0 ), r(w) >

To prove (b), we combine (19) and (20), then for small enough L(θ m ), we havê

Now we specify the value of C η .

By (S1) and (S2), we can define B 0 , B 1 , B 2 as follows:

Then we set

Under review as a conference paper at ICLR 2020 E.3 THEOREMS Now we state our main theorems for the monotonicity of the normalized margin and the convergence to KKT points.

We will prove Theorem E.2 in Appendix E.4, and prove Theorem E.3 and E.4 in Appendix E.5.

Theorem E.2.

Under assumptions (S1), (A2) -(A4), (S5), the following are true for gradient descent:

1.

For all t ≥ t 0 ,γ(t + 1) ≥γ(t);

2.

For all t ≥ t 0 , eitherγ(t + 1) >γ(t) orθ(t + 1) =θ(t);

3. L(t) → 0 and ρ(t) → ∞ as t → +∞; therefore, |γ(t) −γ(t)| → 0.

Theorem E.3.

Consider gradient flow under assumptions (S1), (A2) -(A4), (S5).

For every limit pointθ of θ (t) : t ≥ 0 ,θ/q min (θ) 1/L is a KKT point of (P).

Theorem E.4.

Consider gradient descent under assumptions (S1), (A2) -(A4), (S5).

For any , δ > 0, there exists r := Θ(log δ −1 ) and ∆ := Θ( −2 ) such that θ/q min (θ) 1/L is an ( , δ)-KKT point at some time t * satisfying log ρ(t * ) ∈ (r, r + ∆).

With a refined analysis, we can also derive tight rates for loss convergence and weight growth.

We defer the proof to Appendix E.6.

Theorem E.5.

Under assumptions (S1), (A2) -(A4), (S5), we have the following tight rates for training loss and weight norm:

where T = t−1 τ =t0 η(τ ).

We define ν(t) := N n=1 e −qn(t) q n (t) as we do for gradient flow.

Then we can get a closed form for θ(t), −∇L(t) easily from Corollary B.3.

Also, we can get a lower bound for ν(t) using Lemma B.5 for exponential loss directly.

For proving the first two propositions in Theorem E.2, we only need to prove Lemma E.7. (P1) gives a lower bound forγ. (P2) gives both lower and upper bounds for the weight growth using ν(t). (P3) gives a lower bound for the decrement of training loss.

Finally, (P4) shows the monotonicity ofγ, and it is trivial to deduce the first two propositions in Theorem E.2 from (P4).

Lemma E.7.

For all t = t 0 , t 0 + 1, . . .

, we interpolate between θ(t) and θ(t + 1) by defining θ(t + α) = θ(t) − αη(t)∇L(t) for α ∈ (0, 1).

Then for all integer t ≥ t 0 , ν(t) > 0, and the following holds for all α ∈ [0, 1]:

To prove Lemma E.7, we only need to prove the following lemma and then use an induction:

Proof for Lemma E.7.

We prove this lemma by induction.

For t = t 0 , α = 0, ν(t) > 0 by (S4) and Corollary E.6. (P2), (P3), (P4) hold trivially since logγ(t + α) = logγ(t), L(t + α) = L(t) and logγ(t + α) = logγ(t).

By Lemma E.1, (P1) also holds trivially.

Now we fix an integer T ≥ t 0 and assume that (P1), (P2), (P3), (P4) hold for any t + α ≤ T (where t ≥ t 0 is an integer and α ∈ [0, 1]).

By (P3), L(t) ≤ L(t 0 ) < 1, so ν(t) > 0.

We only need to show that (P1), (P2), (P3), (P4) hold for t = T and α ∈ [0, 1].

Let A := inf{α ∈ [0, 1] : α = 1 or (P1) does not hold for (T, α)}. If A = 0, then (P1) holds for (T, A) since (P1) holds for (T − 1, 1); if A > 0, we can also know that (P1) holds for (T, A) by Lemma E.8.

Suppose that A < 1.

Then by the continuity ofγ(T + α) (with respect to α), we know that there exists A > A such thatγ(T + α) >γ(t 0 ) for all α ∈ [A, A ], which contradicts to the definition of A. Therefore, A = 1.

Using Lemma E.8 again, we can conclude that (P1), (P2), (P3), (P4) hold for t = T and α ∈ [0, 1].

Now we turn to prove Lemma E.8.

Then by Corollary E.6, we have ν(t) > 0.

Applying (P2) on (t, α) ∈ {t 0 , . . . , T − 1} × 1, we can get ρ(t) ≥ ρ(t 0 ).

Fix t = T .

By (P2) with α ∈ [0, A) and the continuity ofγ, we haveγ(t + A) ≥γ(t 0 ).

Thus,

We call this proposition as (P1').

By Corollary E.6, we have

where the last equality uses the definition of λ and the inequality

Proof for (P3). (P3) holds trivially for α = 0 or ∇L(t) = 0.

So now we assume that α = 0 and ∇L(t) = 0.

By the update rule (18) and Taylor expansion, there exists ξ ∈ (0, α) such that

Under review as a conference paper at ICLR 2020

By the chain rule, we have ∇ 2 L(t + ξ) = N n=1 e −qn(t+ξ) (∇q n (t + ξ)∇q n (t + ξ) − ∇ 2 q n (t + ξ)), and so

.

Combining these together, we have

Thus we have

Now we only need to show that

) by the monotonicity of κ, and thus

, and thus Proof for (P4).

We define v(t) :=θ(t)θ(t) (−∇L(t)) and u(t) := I −θ(t)θ(t) (−∇L(t)) similarly as in the analysis for gradient flow.

For v(t), we have

By Corollary E.6 and (P2), we further have

From the definition φ, it is easy to see that

Then by convexity of φ and ψ, we have

And by definition ofγ, this can be re-written as

Proof for (P1).

By (P4), logγ(t + α) ≥ logγ(t) ≥ logγ(t 0 ).

Note that φ(x) ≥ log log 1 x .

So we haveγ (t + α) >γ(t + α) ≥γ(t 0 ), which completes the proof.

For showing the third proposition in Theorem E.2, we use (P1) to give a lower bound for ∇L(t) 2 , and use (P3) to show the speed of loss decreasing.

Then it can be seen that L(t) → 0 and ρ(t) → +∞. By Lemma E.1, we then have |γ −γ| → 0.

Therefore, L(t) → 0 and ρ(t) → +∞ as t → ∞.

Proof.

For any integer t ≥ t 0 , µ(L(t)) ≤ 1 2 and ∇L(t) 2 ≥ v(t) 2 .

Combining these with (P3), we have

.

Thus we have

It is easy to see that

Note that L is non-decreasing.

If L does not decreases to 0, then neither does E(L).

But the RHS grows to +∞, which leads to a contradiction.

So L → 0.

To make L → 0, q min must converge to +∞.

So ρ → +∞.

The proofs for Theorem E.3 and E.4 are similar as those for Theorem A.8 and A.9 in Appendix C.

Define β(t) := 1 ∇L(t) 2 θ , −∇L(t) as we do in Appendix C. It is easy to see that Lemma C.8 still holds if we replaceγ(t 0 ) withγ(t 0 ).

So we only need to show L → 0 and β → 1 for proving convergence to KKT points.

L → 0 can be followed from Theorem E.2.

Similar as the proof for Lemma C.9, it follows from Lemma E.7 and (15) that for all t 2 > t 1 ≥ t 0 ,

Now we prove Theorem E.4.

Proof for Theorem E.4.

We make the following changes in the proof for Theorem A.9.

First, we replaceγ(t 0 ) withγ(t 0 ), sinceγ(t) (t ≥ t 0 )

is lower bounded byγ(t 0 ) rather thanγ(t 0 ).

Second, when choosing t 1 and t 2 , we make log ρ(t 1 ) and log ρ(t 2 ) equal to the chosen values approximately with an additive error o(1), rather than make them equal exactly.

This is possible because it can be shown from (P2) in Lemma E.7 that the following holds:

Dividing ρ(t) 2 on the leftmost and rightmost sides, we have

which implies that log ρ(t + 1) − log ρ(t) = o(1).

Therefore, for any R, we can always find the minimum time t such that log ρ(t) ≥ R, and it holds for sure that log ρ(t)−R → 0 as R → +∞.

For proving Theorem E.3, we also need the following lemma as a variant of Lemma C.11.

Lemma E.10.

For all t ≥ t 0 ,

Proof.

According to the update rule, we have

.

γ(t)ρ(t) .

So we can bound the first term as

where the last inequality uses the inequality a−b a ≤ log(a/b).

Using this inequality again, we can bound the second term by

Combining these together gives θ (t + 1) −θ(t)

Now we are ready to prove Theorem E.3.

Proof for Theorem E.3.

As discussed above, we only need to show a variant of Lemma C.12 for gradient descent: for every limit pointθ of θ (t) : t ≥ 0 , there exists a sequence of {t m : m ∈ N} such that t m ↑ +∞,θ(t m ) →θ, and β(t m ) → 1.

We only need to change the choices of s m , s m , t m in the proof for Lemma C.12.

We choose s m > t m−1 to be a time such that

Then we let s m > s m be the minimum time such that log ρ(s m ) ≥ log ρ(s m ) + m .

According to Theorem E.2, s m and s m must exist.

Finally, we construct t m ∈ {s m , . . .

, s m − 1} to be a time that

, where the existence can be shown by (22).

To see that this construction meets our requirement, note that β(

where the last inequality is by Lemma E.10.

E.6 PROOF FOR THEOREM E.5

Proof.

By a similar analysis as Lemma D.4, we have

We can also bound the inverse function E −1 (y) by Θ 1 y(log y) 2−2/L .

With these, we can use a similar analysis as Theorem A.10 to prove Theorem E.5.

First, using a similar proof as for (17), we have ρ

.

With a similar analysis as for (P3) in Lemma E.8, we have the following bound for L(τ + 1) − L(τ ):

Using the fact that µ ≤ 1/2, we have

Using a similar proof as

for Lemma E.9, we can show that E(L(t)) ≤ O(T ).

Combining this with Lemma E.9, we have

In this section, we generalize our results to multi-class classification with cross-entropy loss.

This part of analysis is inspired by Theorem 1 in (Zhang et al., 2019) , which gives a lower bound for the gradient in terms of the loss L.

Since now a neural network has multiple outputs, we need to redefine our notations.

Let C be the number of classes.

The output of a neural network Φ is a vector Φ(θ; x) ∈ R C .

We use Φ j (θ; x) ∈ R to denote the j-th output of Φ on the input x ∈ R dx .

A dataset is denoted by D = {x n , y n } N n=1 = {(x n , y n ) : n ∈ [N ]}, where x n ∈ R dx is a data input and y n ∈ [C] is the corresponding label.

The loss function of Φ on the dataset D is defined as

− log e −Φy n (θ;xn)

C j=1 e −Φj (θ;xn)

.

The margin for a single data point (x n , y n ) is defined to be q n (θ) := Φ yn (θ; x n ) − max j =yn {Φ j (θ; x n )}, and the margin for the entire dataset is defined to be q min (θ) = min n∈[N ] q n (θ).

We define the normalized margin to beγ(θ) := q min (θ) = q min (θ)/ρ L , where

Let (q) := log(1+e −q ) be the logistic loss.

Recall that (q) satisfies (B3).

Let f (q) = − log (q) = − log log(1 + e −q ).

Let g be the inverse function of f .

So g(q) = − log(e e −q − 1).

The cross-entropy loss can be rewritten in other ways.

Let

We assume the following in this section: (M4). (Separability).

There exists a time t 0 such that L(t 0 ) < log 2.

If L < log 2, then j =yn e −snj < 1 for all n ∈ [N ], and thus

.

So (M4) ensures the separability of training data.

Definition F.1.

For cross-entropy loss, the smoothed normalized marginγ(θ) of θ is defined as

where −1 (·) is the inverse function of the logistic loss (·).

Proof.

Define ν(t) by the following formula:

Using a similar argument as in Theorem B.4, it can be proved that 1 2 dρ 2 dt = Lν(t) for a.e.

t > 0.

It can be shown that Lemma B.5, which asserts that

L, still holds for this new definition of ν(t).

By definition, s nj ≥ q n .

Also note that e −qn ≥ e −qn .

So s nj ≥ q n ≥q n .

Then we have qn) .

Then using Lemma B.5 for logistic loss can conclude that

The rest of the proof for this lemma is exactly as same as that for Lemma 5.1.

In this section, we extend our results to multi-homogeneous models.

Let Φ(w 1 , . . .

,

The smoothed normalized margin defined in (5) can be rewritten as follows: Definition G.1.

For a multi-homogeneous model with loss function (·) satisfying (B3), the smoothed normalized marginγ(θ) of θ is defined as

We only prove the generalized version of Lemma 5.1 here.

The other proofs are almost the same.

Proof.

Note that

by Theorem B.4.

It simply follows from Lemma B.5 that d dt log ρ > 0 for a.e.

t > t 0 .

And it is easy to see that logγ = log g(log 1 L )

/ρ L exists for all t ≥ t 0 .

By the chain rule and Lemma B.5, we have

On the one hand,

for a.e.

t > 0 by Lemma H.3; on the other hand,

by Theorem B.4.

Combining these together yields

For cross-entropy loss, we can combine the proofs in Appendix F to show that Lemma G.2 holds if we use the following definition of the smoothed normalized margin:

Definition G.3.

For a multi-homogeneous model with cross-entropy, the smoothed normalized marginγ(θ) of θ is defined asγ

where −1 (·) is the inverse function of the logistic loss (·).

The only place we need to change in the proof for Lemma G.2 is that instead of using Lemma B.5, we need to prove

L in a similar way as in Lemma F.2.

The other parts of the proof are exactly the same as before.

In this section, we provide some background on the chain rule for non-differentiable functions.

The ordinary chain rule for differentiable functions is a very useful formula for computing derivatives in calculus.

However, for non-differentiable functions, it is difficult to find a natural definition of subdifferential so that the chain rule equation holds exactly.

To solve this issue, Clarke proposed Clarke's subdifferential (Clarke, 1975; 1990; Clarke et al., 2008) for locally Lipschitz functions, for which the chain rule holds as an inclusion rather than an equation:

Theorem H.1 (Theorem 2.3.9 and 2.3.10 of (Clarke, 1990) ).

Let z 1 , . . .

, z n : R d → R and f : R n → R be locally Lipschitz functions.

Let (f • z)(x) = f (z 1 (x), . . .

, z n (x)) be the composition of f and z. Then,

For analyzing gradient flow, the chain rule is crucial.

For a differentiable loss function L(θ), we can see from the chain rule that the function value keeps decreasing along the gradient flow

But for locally Lipschitz functions which could be non-differentiable, (24) may not hold in general since Theorem H.1 only holds for an inclusion.

Following (Davis et al., 2019; Drusvyatskiy et al., 2015) , we consider the functions that admit a chain rule for any arc.

holds for a.e.

t > 0.

It is shown in (Davis et al., 2019; Drusvyatskiy et al., 2015) that a generalized version of (24)

holds for a.e.

t > 0.

We can see that C 1 -smooth functions admit chain rules.

As shown in (Davis et al., 2019) , if a locally Lipschitz function is subdifferentiablly regular or Whitney C 1 -stratifiable, then it admits a chain rule.

The latter one includes a large family of functions, e.g., semi-algebraic functions, semianalytic functions, and definable functions in an o-minimal structure (Coste, 2002; van den Dries and Miller, 1996) .

It is worth noting that the class of functions that admits chain rules is closed under composition.

This is indeed a simple corollary of Theorem H.1.

Theorem H.4.

Let z 1 , . . .

, z n : R d → R and f : R n → R be locally Lipschitz functions and assume all of them admit chain rules.

Let (f • z)(x) = f (z 1 (x), . . .

, z n (x)) be the composition of f and z. Then f • z also admits a chain rule.

Proof.

We can see that f • z is locally Lipschitz.

Let

is also an arc.

For any closed sub-interval I, z(x(I)) must be contained in a compact set U .

Then it can be shown that the locally Lipschitz continuous function z is (globally) Lipschitz continuous on U .

By the fact that the composition of a Lipschitz continuous and an absolutely continuous function is absolutely continuous, z • x is absolutely continuous on I, and thus it is an arc.

Since f and z admit chain rules on arcs z • x and x respectively, the following holds for a.e.

t > 0,

Combining these we obtain that for a.e.

t > 0,

for all α ∈ ∂ • f (z(x(t))) and for all h i ∈ ∂ • z i (x(t)).

The RHS can be rewritten as

can be written as a convex combination of a finite set of points in the form of

In this section, we give an example to illustrate that gradient flow does not necessarily converge in direction, even for C ∞ -smooth homogeneous models.

It is known that gradient flow (or gradient descent) may not converge to any point even when optimizing an C ∞ function (Curry, 1944; Zoutendijk, 1976; Palis and De Melo, 2012; Absil et al., 2005) .

One famous counterexample is the "Mexican Hat" function described in (Absil et al., 2005) : However, the Maxican Hat function is not homogeneous, and Absil et al. (2005) did not consider the directional convergence, either.

To make it homogeneous, we introduce an extra variable z, and normalize the parameter before evaluate f .

In particular, we fix L > 0 and define

We can show the following theorem.

Theorem I.1.

Consider gradient flow on L(θ) = N n=1 e −qn(θ) , where q n (θ) = h(θ) for all n ∈ [N ].

Suppose the polar representation of (u, v) is (r cos ϕ, r sin ϕ).

If 0 < r < 1 and ϕ = 1 1−r 2 holds at time t = 0, then

does not converge to any point, and the limit points of

Proof.

Define ψ = ϕ − 1 1−r 2 .

Our proof consists of two parts, following from the idea in (Absil et al., 2005) .

First, we show that dψ dt = 0 as long as ψ = 0.

Then we can infer that ψ = 0 for all t ≥ 0.

Next, we show that r → 1 as t → +∞. Using ψ = 0, we know that the polar angle ϕ → +∞ as t → +∞. Therefore, (u, v) circles around {(u, v) : u 2 + v 2 = 1}, and thus it does not converge.

Proof for dψ dt = 0.

For convenience, we use w to denote z/ρ.

By simple calculation, we have the following formulas for partial derivatives: For gradient flow, we have

By writing down the movement of (u, v) in the polar coordinate system, we have

For ψ = 0, the partial derivatives of f with respect to r and ϕ can be evaluated as follows:

It is easy to see that r ≤ 1 from the normalization of θ in the definition.

According to Theorem 4.4, we know that (ū,v) is a stationary point ofγ(u(t), v(t)) = 1 − f (u(t), v(t)).

Ifr = 0, then f (ū,v) > f (u(0), v(0)), which contradicts to the monotonicity ofγ(t) =γ(t) = 1 − f (u(t), v(t)).

1−r 2 = 0, which again leads to a contradiction.

Therefore,r = 1, and thus r → 1.

To validate our theoretical results, we conduct several experiments.

We mainly focus on MNIST dataset.

We trained two models with Tensorflow.

The first one (called the CNN with bias) is a standard 4-layer CNN with exactly the same architecture as that used in MNIST Adversarial Examples Challenge 5 .

The layers of this model can be described as conv-32 with filter size 5×5, max-pool, conv-64 with filter size 3 × 3, max-pool, fc-1024, fc-10 in order.

Notice that this model has bias terms in each layer, and thus does not satisfy homogeneity.

To make its outputs homogeneous to its parameters, we also trained this model after removing all the bias terms except those in the first layer (the modified model is called the CNN without bias).

Note that keeping the bias terms in the first layer prevents the model to be homogeneous in the input data while retains the homogeneity in parameters.

We initialize all layer weights by He normal initializer (He et al., 2015) and all bias terms by zero.

In training the models, we use SGD with batch size 100 without momentum.

We normalize all the images to [0, 1] 32×32 by dividing 255 for each pixel.

In the first part of our experiments, we evaluate the normalized margin every few epochs to see how it changes over time.

From now on, we view the bias term in the first layer as a part of the weight in the first layer for convenience.

Observe that the CNN without bias is multi-homogeneous in layer weights (see (4) in Section 4.4).

So for the CNN without bias, we define the normalized margin γ as the margin divided by the product of the L 2 -norm of all layer weights.

Here we compute the L 2 -norm of a layer weight parameter after flattening it into a one-dimensional vector.

For the CNN with bias, we still compute the smoothed normalized margin in this way.

When computing the L 2 -norm of every layer weight, we simply ignore the bias terms if they are not in the first layer.

For completeness, we include the plots for the normalized margin using the original definition (2) in Figure 3 SGD with Constant Learning Rate.

We first train the CNNs using SGD with constant learning rate 0.01.

After about 100 epochs, both CNNs have fitted the training set.

After that, we can see that the normalized margins of both CNNs increase.

However, the growth rate of the normalized margin is rather slow.

The results are shown in Figure 1 in Section 1.

We also tried other learning rates other than 0.01, and similar phenomena can be observed.

SGD with Loss-based Learning Rate.

Indeed, we can speed up the training by using a proper scheduling of learning rates for SGD.

We propose a heuristic learning rate scheduling method, called the loss-based learning rate scheduling.

The basic idea is to find the maximum possible learning rate at each epoch based on the current training loss (in a similar way as the line search method).

See Appendix K.1 for the details.

As shown in Figure 1 , SGD with loss-based learning rate scheduling decreases the training loss exponentially faster than SGD with constant learning rate.

Also, a rapid growth of normalized margin is observed for both CNNs.

Note that with this scheduling the training loss can be as small as 10 −800 , which may lead to numerical issues.

To address such issues, we applied some re-parameterization tricks and numerical tricks in our implementation.

See Appendix K.2 for the details.

Experiments on CIFAR-10.

To verify whether the normalized margin is increasing in practice, we also conduct experiments on CIFAR-10.

We use a modified version of VGGNet-16.

The layers of this model can be described as conv-64 ×2, max-pool, conv-128 ×2, max-pool, conv-256 ×3, max-pool, conv-512 ×3, max-pool, conv-512 ×3, max-pool, fc-10 in order, where each conv has filter size 3 × 3.

We train two networks: one is exactly the same as the VGGNet we described, and the other one is the VGGNet without any bias terms except those in the first layer (similar as in the experiments on MNIST).

The experiment results are shown in Figure 5 and 6.

We can see that the normalize margin is increasing over time.

Test Accuracy.

Previous works on margin-based generalization bounds (Neyshabur et al., 2018; Bartlett et al., 2017; Golowich et al., 2018; Li et al., 2018a; Wei et al., 2018; Banburski et al., 2019) usually suggest that a larger margin implies a better generalization bound.

To see whether the generalization error also gets smaller in practice, we plot train and test accuracy for both MNIST and CIFAR-10.

As shown in Figure 7 , the test accuracy changes only slightly after training with lossbased learning rate scheduling for 10000 epochs, although the normalized margin does increase a lot.

We leave it as a future work to study this interesting gap between generalization bound and generalization error.

Training and test accuracy during training VGGNet without bias on CIFAR-10, using SGD with the loss-based learning rate scheduler.

Every number is averaged over 3 runs.

Recently, robustness of deep learning has received considerable attention (Szegedy et al., 2013; Biggio et al., 2013; Athalye et al., 2018) , since most state-of-the-arts deep neural networks are found to be very vulnerable against small but adversarial perturbations of the input points.

In our experiments, we found that enlarging the normalized margin can improve the robustness.

In particular, by simply training the neural network for a longer time with our loss-based learning rate, we observe noticeable improvements of L 2 -robustness on both the training set and test set.

We first elaborate the relationship between the normalized margin and the robustness from a theoretical perspective.

For a data point z = (x, y), we can define the robustness (with respect to some norm · ) of a neural network Φ for z to be R θ (z) := inf

where X is the data domain (which is [0, 1] 32×32 for MNIST).

It is well-known that the normalized margin is a lower bound of L 2 -robustness for fully-connected networks (See, e.g., Theorem 4 in (Sokolic et al., 2017) ).

Indeed, a general relationship between those two quantities can be easily shown.

Note that a data point z is correctly classified iff the margin for z, denoted as q θ (z), is larger than 0.

For homogeneous models, the margin q θ (z) and the normalized margin qθ(z) for x have the same sign.

If qθ(·) : R dx → R is β-Lipschitz (with respect to some norm · ), then it is easy to see that R θ (z) ≥ qθ(z)/β.

This suggests that improving the normalize margin on the training set can improve the robustness on the training set.

Therefore, our theoretical analysis suggests that training longer can improve the robustness of the model on the training set.

This observation does match with our experiment results.

In the experiments, we measure the L 2 -robustness of the CNN without bias for the first time its loss decreases below 10 −10 , 10 −15 , 10 −20 , 10 −120 (labelled as model-1 to model-4 respectively).

We also measure the L 2 -robustness for the final model after training for 10000 epochs (labelled as model-5), whose training loss is about 10 −882 .

The normalized margin of each model is monotone increasing with respect to the number of epochs, as shown in Table 1 .

Table 1 for the statistics of each model).

Figures on the first row show the robust accuracy on the training set, and figures on the second row show that on the test set.

On every row, the left figure and the right figure plot the same curves but they are in different scales.

From model-1 to model-4, noticeable robust accuracy improvements can be observed.

The improvement of model-5 upon model-4 is marginal or nonexistent for some , but the improvement upon model-1 is always significant.

We use the standard method for evaluating L 2 -robustness in (Carlini and Wagner, 2017) and the source code from the authors with default hyperparameters 6 .

We plot the robust accuracy (the percentage of data with robustness > ) for the training set in the figures on the first row of Figure 8 .

It can be seen from the figures that for small (e.g., < 0.3), the relative order of robust accuracy is just the order of model-1 to model-5.

For relatively large (e.g., > 0.3), the improvement of model-5 upon model-2 to model-4 becomes marginal or nonexistent in certain intervals of , but model-1 to model-4 still have an increasing order of robust accuracy and the improvement of model-5 upon model-1 is always significant.

This shows that training longer can help to improve the L 2 -robust accuracy on the training set.

We also evaluate the robustness on the test set, in which a misclassified test sample is considered to have robustness 0, and plot the robust accuracy in the figures on the second row of Figure 8 .

It can be seen from the figures that for small (e.g., < 0.2), the curves of the robust accuracy of model-1 to model-5 are almost indistinguishable.

However, for relatively large (e.g., > 0.2), again, model-1 to model-4 have an increasing order of robust accuracy and the improvement of model-5 upon model-1 is always significant.

This shows that training longer can also help to improve the L 2 -robust accuracy on the test set.

We tried various different settings of hyperparameters for the evaluation method (including different learning rates, different binary search steps, etc.) and we observed that the shapes and relative positions of the curves in Figure 8 are stable across different hyperparameter settings.

In this section, we provide additional details of our experiments.

The intuition of the loss-based learning rate scheduling is as follows.

If the training loss is α-smooth, then optimization theory suggests that we should set the learning rate to roughly 1/α.

For a homogeneous model with cross-entropy loss, if the training accuracy is 100% at θ, then a simple calculation can show that the smoothness (the L 2 -norm of the Hessian matrix) at θ is O(L·poly(ρ)), whereL is the average training loss and poly(ρ) is some polynomial.

Motivated by this fact, we parameterize the learning rate η(t) at epoch t as

whereL(t − 1) is the average training loss at epoch t − 1, and α(t) is a relative learning rate to be tuned (Similar parameterization has been considiered in (Nacson et al., 2019b) for linear model).

The loss-based learning rate scheduling is indeed a variant of line search.

In particular, we initialize α(0) by some value, and do the following at each epoch t:

Step 1.

Initially α(t) ← α(t − 1); LetL(t − 1) be the training loss at the end of the last epoch;

Step 2.

Run SGD through the whole training set with learning rate η(t) := α(t)/L(t − 1);

Step 3.

Evaluate the training lossL(t) on the whole training set;

Step 4.

IfL(t) <L(t − 1), α(t) ← α(t) · r u and end this epoch; otherwise, α(t) ← α(t)/r d and go to

Step 2.

In all our experiments, we set α(0) := 0.1, r u := 2 1/5 ≈ 1.149, r d := 2 1/10 ≈ 1.072.

This specific choice of those hyperparameters is not important; other choices can only affact the computational efficiency, but not the overall tendency of normalized margin.

Since we are dealing with extremely small loss (as small as 10 −800 ), the current Tensorflow implementation would run into numerical issues.

To address the issues, we work as follows.

LetL B (θ) be the (average) training loss within a batch B ⊆

[N ].

We use the notations C, s nj ,q n , q n from Appendix F. We only need to show how to perform forward and backward passes forL B (θ).

is in the range of float64.

R B (θ) can be thought of a relative training loss with respect to F. Instead of evaluating the training lossL B (θ) directly, we turn to evaluate this relative training loss in a numerically stable way:

Step 1.

Perform forward pass to compute the values of s nj with float32, and convert them into float64;

Step 2.

Let Q := 30.

If q n (θ) > Q for all n ∈ B, then we compute This algorithm can be explained as follows.

Step 1 is numerically stable because we observe from the experiments that the layer weights and layer outputs grow slowly.

Now we consider Step 2.

If q n (θ) ≤ Q for some n ∈ [B], thenL B (θ) = Ω(e −Q ) is in the range of float64, so we can compute R B (θ) by (27) directly except that we need to use a numerical stable implementation of log(1 + x).

For q n (θ) > Q, arithmetic underflow can occur.

By Taylor expansion of log(1 + x), we know that when x is small enough log(1 + x) ≈ x in the sense that the relative error

for q n (θ) > Q, and only introduce a relative error of O(Ce −Q ) (recall that C is the number of classes).

Using a numerical stable implementation of LSE, we can computeq n easily.

Then the RHS of (28) can be rewritten as e −(qn(θ)+ F ) .

Note that computing e −(qn(θ)+ F ) does not have underflow or overflow problems if F is a good approximation for logL B (θ).

Backward Pass.

To perform backward pass, we build a computation graph in Tensorflow for the above forward pass for the relative training loss and use the automatic differentiation.

We parameterize the learning rate as η =η · e F .

Then it is easy to see that taking a step of gradient descent for L B (θ) with learning rate η is equivalent to taking a step for R B (θ) withη.

Thus, as long asη can fit into float64, we can perform gradient descent on R B (θ) to ensure numerical stability.

The Choice of F. The only question remains is how to choose F. In our experiments, we set F(t) := logL(t − 1) to be the training loss at the end of the last epoch, since the training loss cannot change a lot within one single epoch.

For this, we need to maintain logL(t) during training.

This can be done as follows: after evaluating the relative training loss R(t) on the whole training set, we can obtain logL(t) by adding F(t) and log R(t) together.

It is worth noting that with this choice of F,η(t) = α(t) in the loss-based learning rate scheduling.

As shown in the right figure of Figure 4 , α(t) is always between 10 −9 and 10 0 , which ensures the numerical stability of backward pass.

<|TLDR|>

@highlight

We study the implicit bias of gradient descent and prove under a minimal set of assumptions that the parameter direction of homogeneous models converges to KKT points of a natural margin maximization problem.
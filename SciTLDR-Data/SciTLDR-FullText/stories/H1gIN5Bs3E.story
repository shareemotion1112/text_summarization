When considering simultaneously a finite number of tasks, multi-output learning enables one to account for the similarities of the tasks via appropriate regularizers.

We propose a generalization of the classical setting to a continuum of tasks by using vector-valued RKHSs.

Several fundamental problems in machine learning and statistics can be phrased as the minimization of a loss function described by a hyperparameter.

The hyperparameter might capture numerous aspects of the problem: (i) the tolerance w. r. t. outliers as the -insensitivity in Support Vector Regression (Vapnik et al., 1997) , (ii) importance of smoothness or sparsity such as the weight of the l 2 -norm in Tikhonov regularization (Tikhonov & Arsenin, 1977) , l 1 -norm in LASSO (Tibshirani, 1996) , or more general structured-sparsity inducing norms BID3 , (iii) Density Level-Set Estimation (DLSE), see for example one-class support vector machines One-Class Support Vector Machine (OCSVM, Schölkopf et al., 2000) , (iv) confidence as exemplified by Quantile Regression (QR, Koenker & Bassett Jr, 1978) , or (v) importance of different decisions as implemented by Cost-Sensitive Classification (CSC, Zadrozny & Elkan, 2001) .

In various cases including QR, CSC or DLSE, one is interested in solving the parameterized task for several hyperparameter values.

Multi-Task Learning (Evgeniou & Pontil, 2004 ) provides a principled way of benefiting from the relationship between similar tasks while preserving local properties of the algorithms: ν-property in DLSE (Glazer et al., 2013) or quantile property in QR (Takeuchi et al., 2006) .A natural extension from the traditional multi-task setting is to provide a prediction tool being able to deal with any value of the hyperparameter.

In their seminal work, (Takeuchi et al., 2013) extended multi-task learning by considering an infinite number of parametrized tasks in a framework called Parametric Task Learning (PTL) .

Assuming that the loss is piecewise affine in the hyperparameter, the authors are able to get the whole solution path through parametric programming, relying on techniques developed by Hastie et al. (2004) .In this paper 1 , we relax the affine model assumption on the tasks as well as the piecewise-linear assumption on the loss, and take a different angle.

We propose Infinite Task Learning (ITL) within the framework of functionvalued function learning to handle a continuum number of parameterized tasks using Vector-Valued Reproducing Kernel Hilbert Space (vv-RKHS, Pedrick, 1957) .

After introducing a few notations, we gradually define our goal by moving from single parameterized tasks to ITL through multi-output learning.

A supervised parametrized task is defined as follows.

Let (X, Y ) ∈ X × Y be a random variable with joint distribution P X,Y which is assumed to be fixed but unknown; we also assume that Y ⊂ R. We have access to n independent identically distributed (i. i. d.) observations called training samples: S:=((x i , y i )) n i=1 ∼ P ⊗n X,Y .

Let Θ be the domain of hyperparameters, and v θ : Y × Y → R be a loss function associated to θ ∈ Θ. Let H ⊂ F (X ; Y) denote our hypothesis class; throughout the paper H is assumed to be a Hilbert space with inner product ·, · H .

For a given θ, the goal is to estimate the minimizer of the expected risk DISPLAYFORM0 over H, using the training sample S. This task can be addressed by solving the regularized empirical risk minimization problem DISPLAYFORM1 where R θ S (h):= 1 n n i=1 v θ (y i , h(x i )) is the empirical risk and Ω : H → R is a regularizer.

Below we give two examples.

Quantile Regression:

In this setting θ ∈ (0, 1).

For a given hyperparameter θ, in Quantile Regression the goal is to predict the θ-quantile of the real-valued output conditional distribution P Y |X .

The task can be tackled using the pinball loss (Koenker & Bassett Jr, 1978) defined in Eq. (3).

DISPLAYFORM2 Density Level-Set Estimation: Examples of parameterized tasks can also be found in the unsupervised setting.

For instance in outlier detection, the goal is to separate outliers from inliers.

A classical technique to tackle this task is OCSVM (Schölkopf et al., 2000) .

OCSVM has a free parameter θ ∈ (0, 1], which can be proven to be an upper bound on the fraction of outliers.

This unsupervised learning problem can be empirically described by the minimization of a regularized empirical risk R S θ (h, t) + Ω(h), solved jointly over h ∈ H and t ∈ R with DISPLAYFORM3 In the aforementioned problems, one is rarely interested in the choice of a single hyperparameter value (θ) and associated risk R S θ , but rather in the joint solution of multiple tasks.

The naive approach of solving the different tasks independently can easily lead to inconsistencies.

A principled way of solving many parameterized tasks has been cast as a MTL problem (Evgeniou et al., 2005) which takes into account the similarities between tasks and helps providing consistent solutions.

For example it is possible to encode the similarities of the different tasks in MTL through an explicit constraint function (Ciliberto et al., 2017 ).

In the current work, the similarity between tasks is designed in an implicit way through the loss function and the use of a kernel on the hyperparameters.

Moreover, in contrast to MTL, in our case the input space and the training samples are the same for each task; a task is specified by a value of the hyperparameter.

This setting is sometimes refered to as multi-output learning (Álvarez et al., 2012) .Formally, assume that we have p tasks described by parameters (θ j ) p j=1 .

The idea of multi-task learning is to minimize the sum of the local loss functions R S θj , i. e.arg min DISPLAYFORM4 where the individual tasks are modelled by the real-valued h j functions, the overall R p -valued model is the vectorvalued function h: x → (h 1 (x), . . .

, h p (x)), and Ω is a regularization term encoding similarities between tasks.

Such approaches have been developed in (Sangnier et al., 2016) for QR and in (Glazer et al., 2013) for DLSE.Learning a continuum of tasks:

In the following, we propose a novel framework called Infinite Task Learning in which we learn a function-valued function h ∈ F (X ; F (Θ; Y)).

Our goal is to be able to handle new tasks after the learning phase and thus, not to be limited to given predefined values of the hyperparameter.

Regarding this goal, our framework generalizes the Parametric Task Learning approach introduced by Takeuchi et al. (2013) , by allowing a wider class of models and relaxing the hypothesis of piece-wise linearity of the loss function.

Moreover a nice byproduct of this vv-RKHS based approach is that one can benefit from the functional point of view, design new regularizers and impose various constraints on the whole continuum of tasks, e. g.,• The continuity of the θ → h(x)(θ) function is a natural desirable property: for a given input x, the predictions on similar tasks should also be similar.• Another example is to impose a shape constraint in QR: the conditional quantile should be increasing w. r. t. the hyperparameter θ.

This requirement can be imposed through the functional view of the problem.• In DLSE, to get nested level sets, one would want that for all x ∈ X , the decision function θ → 1 R+ (h(x)(θ) − t(θ)) changes its sign only once.

To keep the presentation simple, in the sequel we are going to focus on ITL in the supervised setting; unsupervised tasks can be handled similarly.

Assume that h belongs to some space H ⊆ F (X ; F (Θ; Y)) and introduce an integrated loss function DISPLAYFORM5 where the local loss v: Θ × Y × Y → R denotes v θ seen as a function of three variables including the hyperparameter and µ is a probability measure on Θ which encodes the importance of the prediction at different hyperparameter values.

Without prior information and for compact Θ, one may consider µ to be uniform.

The true risk reads then DISPLAYFORM6 Intuitively, minimizing the expectation of the integral over θ in a rich enough space corresponds to searching for a pointwise minimizer x → h * (x)(θ) of the parametrized tasks introduced in Eq. (1) with, for instance, the implicit space constraint that θ → h * (x)(θ) is a continuous function for each input x.

We show in Proposition S.4.1 that this is precisely the case in QR.Interestingly, the empirical counterpart of the true risk minimization can now be considered with a much richer family of penalty terms: DISPLAYFORM7 Here, Ω(h) can be a weighted sum of various penalties as seen in Section 3.

Many different models (H) could be applied to solve this problem.

In our work we consider Reproducing Kernel Hilbert Spaces as they offer a simple and principled way to define regularizers by the appropriate choice of kernels and exhibit a significant flexibility.

This section is dedicated to solving the ITL problem defined in Eq. (6).

We first focus on the objective ( V ), then detail the applied vv-RKHS model family with various penalty examples, followed by representer theorems which give rise to computational tractability.

In practice solving Eq. (6) can be rather challenging due to the integral over θ.

One might consider different numerical integration techniques to handle this issue.

We focus here on Quasi Monte Carlo (QMC) methods as they allow (i) efficient optimization over vv-RKHSs which we will use for modelling H (Proposition 3.1), and (ii) enable us to derive generalization guarantees (Proposition 3.3).

Indeed, let DISPLAYFORM0 be the QMC approximation of Eq. (4).

Let w j = m −1 F −1 (θ j ), and (θ j ) m j=1 be a sequence with values in [0, 1] d such as the Sobol or Halton sequence where µ is assumed to be absolutely continuous w. r. t. the Lebesgue measure and F is the associated cdf.

Using this notation and the training samples S = ((x i , y i ))

n i=1 , the empirical risk takes the form DISPLAYFORM1 and the problem to solve is DISPLAYFORM2 Hypothesis class (H): Recall that H ⊆ F (X ; F (Θ; Y)), in other words h(x) is a Θ → Y function for all x ∈ X .

In this work we assume that the Θ → Y mapping can be described by an RKHS H kΘ associated to a k Θ : Θ × Θ → R scalar-valued kernel defined on the hyperparameters.

Let k X : X × X → R be a scalar-valued kernel on the input space.

The x → (hyperparameter → output) relation, i. e. h: X → H kΘ is then modelled by the Vector-Valued Reproducing Kernel Hilbert Spa- DISPLAYFORM3 where the operator-valued kernel K is defined as K(x, z) = k X (x, z)I, and I = I H k Θ is the identity operator on H kΘ .This so-called decomposable Operator-Valued Kernel has several benefits and gives rise to a function space with a well-known structure.

One can consider elements h ∈ H K as mappings from X to H kΘ , and also as functions from (X × Θ) to R. It is indeed known that there is an isometry between H K and H k X ⊗ H kΘ , the RKHS associated to the product kernel k X ⊗ k Θ .

The equivalence between these views allows a great flexibility and enables one to follow a functional point of view (to analyse statistical aspects) or to leverage the tensor product point of view (to design new kind of penalization schemes).

Below we detail various regularizers before focusing on the representer theorems.• Ridge Penalty: For QR, a natural regularization is the squared vv-RKHS norm DISPLAYFORM4 This choice is amenable to excess risk analysis (see Proposition 3.3).

It can be also seen as the counterpart of the classical (multi-task regularization term introduced by Sangnier et al. FORMULA0 , compatible with an infinite number of tasks.

·

2 H K acts by constraining the solution to a ball of a finite radius within the vv-RKHS, whose shape is controlled by both k X and k Θ .• L 2,1 -penalty: For DLSE, it is more adequate to apply an L 2,1 -RKHS mixed regularizer: DISPLAYFORM5 which is an example of a Θ-integrated penalty.

This Ω choice allows the preservation of the θ-property (see Fig. S. 3), i. e. that the proportion of the outliers is θ.• Shape Constraints: Taking the example of QR it is advantageous to ensure the monotonicity of the estimated quantile function Let ∂ Θ h denotes the derivative of h(x)(θ) with respect to θ.

Then one should solve arg min DISPLAYFORM6 However, the functional constraint prevents a tractable optimization scheme.

To mitigate this bottleneck, we penalize if the derivative of h w. r. t. θ is negative: DISPLAYFORM7 When P:=P X this penalization can rely on the same anchors and weights as the ones used to approximate the integrated loss function: DISPLAYFORM8 Thus, one can modify the overall regularizer in QR to be DISPLAYFORM9 Representer Theorems:

Apart from the flexibility of regularizer design, the other advantage of applying vv-RKHS as hypothesis class is that it gives rise to finite-dimensional representation of the ITL solution under mild conditions.

Proposition 3.1 (Representer).

Assume that for ∀θ ∈ Θ, v θ is a proper lower semicontinuous convex function with respect to its second argument.

Then DISPLAYFORM10 with Ω(h) defined as in Eq. FORMULA0 , has a unique solution h * , and DISPLAYFORM11 For DLSE, we similarly get a representer theorem with the following modelling choice.

Let DISPLAYFORM12 2 Then, learning a continuum of level sets boils down to the minimization problem arg min DISPLAYFORM13 where DISPLAYFORM14 Remarks:• Relation to Joint Quantile Regression (JQR): In Infinite Quantile Regression (∞-QR), by choosing k Θ to be the (JQR) framework as a special case of our approach.

In contrast to the JQR, however, in ∞-QR one can predict the quantile value at any θ ∈ (0, 1), even outside the (θ j ) m j=1 used for learning.

DISPLAYFORM15 • Relation to q-OCSVM: In DLSE, by choosing k Θ (θ, θ ) = 1 (for all θ , θ ∈ Θ) to be the constant kernel, DISPLAYFORM16 δ θj , our approach specializes to q-OCSVM (Glazer et al., 2013) .•

Relation to Kadri et al. (2016) : Note that Operator-Valued Kernels for functional outputs have also been used in (Kadri et al., 2016) , under the form of integral operators acting on L 2 spaces.

Both kernels give rise to the same space of functions; the benefit of our approach being to provide an exact finite representation of the solution (see Proposition 3.1).• Efficiency of the decomposable kernel: this kernel choice allows to rewrite the expansions in Propositions 3.1 and 3.2 as a Kronecker products and the complexity of the prediction of n points for m quantile becomes DISPLAYFORM17 Excess Risk Bounds: Below we provide a generalization error analysis to the solution of Eq. FORMULA10 for QR (with Ridge regularization and without shape constraints) by stability argument BID4 , extending the work of BID2 to Infinite-Task Learning.

The proposition (finite sample bounds are given in Corollary S.5.6) instantiates the guarantee for the QMC scheme.

Proposition 3.3 (Generalization).

Let h * ∈ H K be the solution of Eq. (9) for the QR problem with QMC approximation.

Under mild conditions on the kernels k X , k Θ and P X,Y , stated in the supplement, one has DISPLAYFORM18 (n, m) Trade-off: The proposition reveals the interplay between the two approximations, n (the number of training samples) and m (the number of locations taken in the integral approximation), and allows to identify the regime in λ = λ(n, m) driving the excess risk to zero.

Indeed by choosing m = √ n and discarding logarithmic factors for simplicity, λ n −1 is sufficient.

The mild assumptions imposed are: boundedness on both kernels and the random variable Y , as well as some smoothness of the kernels.

Numerical Experiments: The efficiency of the ITL scheme for QR has been tested on several benchmarks; the results are summarized in Table S .1 for 20 real datasets from the UCI repository.

An additional experiment concerning the non-crossing property on a synthetic dataset can be found in Fig. S Let us recall the expression of the pinball loss: DISPLAYFORM19 Proposition S.4.1.

Let X, Y be two random variables (r. v.s) respectively taking values in X and R, and q: X → F([0, 1], R) the associated conditional quantile function.

Let µ be a positive measure on [0, 1] such that DISPLAYFORM20 where R is the risk defined in Eq. (5).Proof.

The proof is based on the one given in (Li et al., 2007) for a single quantile.

Let f ∈ F (X ; F ([0, 1]; R)), θ ∈ (0, 1) and (x, y) ∈ X × R. Let also DISPLAYFORM21

Then, notice that DISPLAYFORM0 and since q is the true quantile function, DISPLAYFORM1 Moreover, (t − s) is negative when q(x)(θ) ≤ y ≤ h(x)(θ), positive when h(x)(θ) ≤ y ≤ q(x)(θ) and 0 otherwise, thus the quantity (t − s)(y − h(x)(θ)) is always positive.

As a consequence, DISPLAYFORM2

There are several ways to solve the non-smooth optimization problems associated to the QR, DLSE and CSC tasks.

One could proceed for example by duality-as it was done in JQR Sangnier et al. (2016)-, or apply sub-gradient descent techniques (which often converge quite slowly).

In order to allow unified treatment and efficient solution in our experiments we used the L-BFGS-B (Zhu et al., 1997) optimization scheme which is widely popular in large-scale learning, with non-smooth extensions (Skajaa, 2010; Keskar & Wächter, 2017) .

The technique requires only evaluation of objective function along with its gradient, which can be computed automatically using reverse mode automatic differentiation (as in BID0 ).

To benefit from from the available fast smooth implementations (Jones et al., 2001; Fei et al., 2014) , we applied an infimal convolution on the non-differentiable terms of the objective.

Under the assumption that m = O( √ n) (see Proposition 3.3), the complexity per L-BFGS-B iteration is O(n 2 √ n).The efficiency of the non-crossing penalty is illustrated in Fig. S .2 on a synthetic sine wave dataset where n = 40 and m = 20 points have been generated.

Many crossings are visible on the right plot, while they are almost not noticible on the left plot, using the non-crossing penalty.

Concerning our real-world examples (20 UCI datasets), to study the efficiency of the proposed scheme in quantile regression the following experimental protocol was applied.

Each dataset was splitted randomly into a training set (70%) and a test set (30%).

We optimized the hyperparameters by minimizing a 5-folds cross validation with a Bayesian optimizer 3 .

Once the hyperparameters were obtained, a new regressor was learned on the whole training set using the optimized hyperparameters.

We report the value of the pinball loss and the crossing loss on the test set for three methods: our technique is called ∞-QR, we refer to Sangnier et al. (2016) 's approach as JQR, and independent learning (abbreviated as IND-QR) represents a further baseline.

We repeated 20 simulations (different random training-test splits); the results are also compared using a Mann-WhitneyWilcoxon test.

A summary is provided in Table S.1.Notice that while JQR is tailored to predict finite many quantiles, our ∞-QR method estimates the whole quantile function hence solves a more challenging task.

Despite the more difficult problem solved, as Table S .1 suggest that the performance in terms of pinball loss of ∞-QR is comparable to that of the state-of-the-art JQR on all the twenty studied benchmarks, except for the 'crabs' and 'cpus' datasets (p.-val.

< 0.25%).

In addition, when considering the non-crossing penalty one can observe that ∞-QR outperforms the IND-QR baseline on eleven datasets (p.-val.

< 0.25%) and JQR on two datasets.

This illustrates the efficiency of the constraint based on the continuum scheme.

The analysis of the generalization error will be performed using the notion of uniform stability introduced in BID4 .

For a derivation of generalization bounds in vv-RKHS, we refer to (Kadri et al., 2016) .

In their framework, the goal is to minimize a risk which can be expressed as DISPLAYFORM0 where S = ((x 1 , y 1 ) , . . .

, (x n , y n )) are i. i. d. inputs and λ > 0.

We almost recover their setting by using losses defined as DISPLAYFORM1 where V is a loss associated to some local cost defined in Eq. (7).

Then, they study the stability of the algorithm which, given a dataset S, returns DISPLAYFORM2 There is a slight difference between their setting and ours, since they use losses defined for some y in the output space of the vv-RKHS, but this difference has no impact on the validity of the proofs in our case.

The use of their theorem requires some assumption that are listed below.

We recall the shape of the OVK we use : DISPLAYFORM3 , where k X and k Θ are both bounded scalar-valued kernels, in other words there exist (κ X , κ Θ ) ∈ R 2 such that sup DISPLAYFORM4 Remark 1.

Assumptions 1, 2 are satisfied for our choice of kernel.

Assumption 3.

The application (y, h, x) → (y, h, x) is σ-admissible, i. e. convex with respect to f and Lipschitz continuous with respect to f (x), with σ as its Lipschitz constant.

Assumption 4.

∃ξ ≥ 0 such that ∀(x, y) ∈ X × Y and ∀S training set, (y, h * S , x) ≤ ξ.

Definition S.5.1.

Let S = ((x i , y i )) n i=1 be the training data.

We call S i the training data S i = ((x 1 , y 1 DISPLAYFORM5 Definition S.5.2.

A learning algorithm mapping a dataset S to a function h * S is said to be β-uniformly stable with respect to the loss function if ∀n ≥ 1, ∀1 ≤ i ≤ n, ∀S training set, DISPLAYFORM6 Proposition S.5.1.

BID4 Let S → h * S be a learning algorithm with uniform stability β with respect to a loss satisfying Assumption 4.

Then ∀n ≥ 1, ∀δ ∈ (0, 1), with probability at least 1 − δ on the drawing of the samples, it holds that DISPLAYFORM7 Proposition S. Quantile Regression: We recall that in this setting, v(θ, y, h(x)(θ)) = max (θ(y − h(x)(θ)), (1 − θ)(y − h(x)(θ))) and the loss is DISPLAYFORM8 Moreover, we will assume that |Y | is bounded by B ∈ R as a r. v..

We will therefore verify the hypothesis for y ∈ [−B, B] and not y ∈ R.Lemma S.5.3.

In the case of the QR, the loss is σ-admissible with σ = 2κ Θ .Proof.

Let h 1 , h 2 ∈ H K and θ ∈ [0, 1].

∀x, y ∈ X × R, it holds that DISPLAYFORM9 where s = 1 y≤h1(x)(θ) and t = 1 y≤h2(x)(θ) .

We consider all possible cases for t and s : DISPLAYFORM10 | because of the conditions on t, s. DISPLAYFORM11 By summing this expression over the (θ j ) m j=1 , we get that DISPLAYFORM12 and is σ-admissible with σ = 2κ Θ .Lemma S.5.4.

Let S = ((x 1 , y 1 ), . . .

, (x n , y n )) be a training set and λ > 0.

Then ∀x, θ ∈ X × (0, 1), it holds that DISPLAYFORM13 Proof.

Since h * S is the output of our algorithm and 0 ∈ H K , it holds that DISPLAYFORM14 Lemma S.5.5.

Assumption 4 is satisfied for ξ = 2 B + κ X κ Θ B λ .Proof.

Let S = ((x 1 , y 1 ), . . .

, (x n , y n )) be a training set and h * S be the output of our algorithm.

DISPLAYFORM15 Corollary S.5.6.

The QR learning algorithm defined in Eq. (9) is such that ∀n ≥ 1, ∀δ ∈ (0, 1), with probability at least 1 − δ on the drawing of the samples, it holds that DISPLAYFORM16 Proof.

This is a direct consequence of Proposition S.5.2, Proposition S.5.1, Lemma S.5.3 and Lemma S.5.5.Definition S.5.3 (Hardy-Krause variation).

Let Π be the set of subdivisions of the interval Θ = [0, 1].

A subdivision will be denoted σ = (θ 1 , θ 2 , . . .

, θ p ) and f : Θ → R be a function.

We call Hardy-Krause variation of the function f the quantity sup DISPLAYFORM17 Remark 2.

If f is continuous, V (f ) is also the limit as the mesh of σ goes to zero of the above quantity.

In the following, let f : DISPLAYFORM18 This function is of primary importance for our analysis, since in the Quasi Monte-Carlo setting, the bound of Proposition 3.3 makes sense only if the function f has finite Hardy-Krause variation, which is the focus of the following lemma.

Lemma S.5.7.

Assume the boundeness of both scalar kernels k X and k Θ .

Assume moreover that k Θ is C 1 and that its partial derivatives are uniformly bounded by some constant C. Then DISPLAYFORM19 Proof.

It holds that DISPLAYFORM20 The supremum of the integral is smaller than the integral of the supremum, as such DISPLAYFORM21 where f x,y : θ → v(θ, y, h * S (x)(θ)) is the counterpart of the function f at point (x, y).

To bound this quantity, let us first bound locally V (f x,y ).

To that extent, we fix some (x, y) in the following.

Since f x,y is continuous (because k Θ is C 1 ), then using Choquet (1969, Theorem 24.6), it holds that DISPLAYFORM22 Moreover since k ∈ C 1 and ∂k θ = (∂ 1 k)(·, θ) has a finite number of zeros for all θ ∈ ×, one can assume that in the subdivision considered afterhand all the zeros (in θ) of the residuals y − h * S (x)(θ) are present, so that y − h * S (x)(θ i+1 ) and y − h * S (x)(θ i ) are always of the same sign.

Indeed, if not, create a new, finer subdivision with this property and work with this one.

Let us begin the proper calculation: let σ = (θ 1 , θ 2 , . . .

, θ p ) be a subdivision of Θ, it holds that ∀i ∈ { 1, . . .

, p − 1 }: DISPLAYFORM23 We now study the two possible outcomes for the residuals: DISPLAYFORM24 Since k Θ is C 1 , with partial derivatives uniformly bounded by C, |k Θ (θ i+1 , θ i+1 ) − k Θ (θ i+1 , θ i )| ≤ C(θ i+1 − θ i ) and |k Θ (θ i , θ i ) − k Θ (θ i+1 , θ i )| ≤ C(θ i+1 − θ i ) so that |h * S (x)(θ i ) − h * S (x)(θ i+1 )| ≤ κ X 2BC λ θ i+1 − θ i and overall DISPLAYFORM25 • If y − h(x)(θ i+1 ) ≤ 0 and y − h(x)(θ i ) ≤ 0 then |f x,y (θ i+1 ) − f x,y (θ i )| = |(1 − θ i+1 )(y − h * S (x)(θ i+1 )) − (1 − θ i )(y − h * S (x)(θ i ))| ≤ |h * S (x)(θ i ) − h * S (x)(θ i+1 )| + |(θ i+1 − θ i )y| + |(θ i − θ i+1 )h * S (x)(θ i+1 )| + |θ i (h * S (x)(θ i ) − h * S (x)(θ i+1 ))| so that with similar arguments one gets DISPLAYFORM26 Therefore, regardless of the sign of the residuals y − h(x)(θ i+1 ) and y − h(x)(θ i ), one gets Eq. (24).

Since the square root function has Hardy-Kraus variation of 1 on the interval Θ = [0, 1], it holds that DISPLAYFORM27 Combining this with Eq. (23) finally gives DISPLAYFORM28 Lemma S.5.8.

Let R be the risk defined in Eq. (5) for the quantile regression problem.

Assume that (θ) m j=1 have been generated via the Sobol sequence and that k Θ is C 1 and that its partial derivatives are uniformly bounded by some constant C. Then Proof of Proposition 3.3.

Combine Lemma S.5.8 and Corollary S.5.6 to get an asymptotic behaviour as n, m → ∞.

To assess the quality of the estimated model by ∞-OCSVM, we illustrate the θ-property (Schölkopf et al., 2000) : the proportion of inliers has to be approximately 1 − θ (∀θ ∈ (0, 1)).

For the studied datasets (Wilt, Spambase) we used the raw inputs without applying any preprocessing.

Our input kernel was the exponentiated χ 2 kernel k X (x, z):= exp −γ X d k=1 (x k − z k ) 2 /(x k + z k ) with bandwidth γ X = 0.25.

A Gauss-Legendre quadrature rule provided the integral approximation in Eq. FORMULA8 , with m = 100 samples.

We chose the Gaussian kernel for k Θ ; its bandwidth parameter γ Θ was the 0.2−quantile of the pairwise Euclidean distances between the θ j 's obtained via the quadrature rule.

The margin (bias) kernel was k b = k Θ .

As it can be seen in FIG6 , the θ-property holds for the estimate which illustrates the efficiency of the proposed continuum approach for density level-set estimation.

@highlight

We propose an extension of multi-output learning to a continuum of tasks using operator-valued kernels.
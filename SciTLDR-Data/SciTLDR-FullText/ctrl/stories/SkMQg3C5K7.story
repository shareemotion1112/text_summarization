We analyze speed of convergence to global optimum for gradient descent training a deep linear neural network by minimizing the L2 loss over whitened data.

Convergence at a linear rate is guaranteed when the following hold: (i) dimensions of hidden layers are at least the minimum of the input and output dimensions; (ii) weight matrices at initialization are approximately balanced; and (iii) the initial loss is smaller than the loss of any rank-deficient solution.

The assumptions on initialization (conditions (ii) and (iii)) are necessary, in the sense that violating any one of them may lead to convergence failure.

Moreover, in the important case of output dimension 1, i.e. scalar regression, they are met, and thus convergence to global optimum holds, with constant probability under a random initialization scheme.

Our results significantly extend previous analyses, e.g., of deep linear residual networks (Bartlett et al., 2018).

Deep learning builds upon the mysterious ability of gradient-based optimization methods to solve related non-convex problems.

Immense efforts are underway to mathematically analyze this phenomenon.

The prominent landscape approach focuses on special properties of critical points (i.e. points where the gradient of the objective function vanishes) that will imply convergence to global optimum.

Several papers (e.g. Ge et al. (2015) ; Lee et al. (2016) ) have shown that (given certain smoothness properties) it suffices for critical points to meet the following two conditions: (i) no poor local minima -every local minimum is close in its objective value to a global minimum; and (ii) strict saddle property -every critical point that is not a local minimum has at least one negative eigenvalue to its Hessian.

While condition (i) does not always hold (cf.

Safran and Shamir (2018) ), it has been established for various simple settings (e.g. Soudry and Carmon (2016) ; Kawaguchi (2016) ).

Condition (ii) on the other hand seems less plausible, and is in fact provably false for models with three or more layers (cf.

Kawaguchi FORMULA1 ), i.e. for deep networks.

It has only been established for problems involving shallow (two layer) models, e.g. matrix factorization (Ge et al. (2016) ; BID12 ).

The landscape approach as currently construed thus suffers from inherent limitations in proving convergence to global minimum for deep networks.

A potential path to circumvent this obstacle lies in realizing that landscape properties matter only in the vicinity of trajectories that can be taken by the optimizer, which may be a negligible portion of the overall parameter space.

Several papers (e.g. Saxe et al. (2014) ; BID1 ) have taken this trajectory-based approach, primarily in the context of linear neural networks -fully-connected neural networks with linear activation.

Linear networks are trivial from a representational perspective, but not so in terms of optimization -they lead to non-convex training problems with multiple minima and saddle points.

Through a mix of theory and experiments, BID1 argued that such non-convexities may in fact be beneficial for gradient descent, in the sense that sometimes, adding (redundant) linear layers to a classic linear prediction model can accelerate the optimization.

This phenomenon challenges the holistic landscape view, by which convex problems are always preferable to non-convex ones.

Even in the linear network setting, a rigorous proof of efficient convergence to global minimum has proved elusive.

One recent progress is the analysis of BID3 for linear residual networks -a particular subclass of linear neural networks in which the input, output and all hidden dimensions are equal, and all layers are initialized to be the identity matrix (cf.

Hardt and Ma (2016) ).

Through a trajectory-based analysis of gradient descent minimizing 2 loss over a whitened dataset (see Section 2), BID3 show that convergence to global minimum at a linear rateloss is less than > 0 after O(log 1 ) iterations -takes place if one of the following holds: (i) the objective value at initialization is sufficiently close to a global minimum; or (ii) a global minimum is attained when the product of all layers is positive definite.

The current paper carries out a trajectory-based analysis of gradient descent for general deep linear neural networks, covering the residual setting of BID3 , as well as many more settings that better match practical deep learning.

Our analysis draws upon the trajectory characterization of BID1 for gradient flow (infinitesimally small learning rate), together with significant new ideas necessitated due to discrete updates.

Ultimately, we show that when minimizing 2 loss of a deep linear network over a whitened dataset, gradient descent converges to the global minimum, at a linear rate, provided that the following conditions hold: (i) the dimensions of hidden layers are greater than or equal to the minimum between those of the input and output; (ii) layers are initialized to be approximately balanced (see Definition 1) -this is met under commonplace near-zero, as well as residual (identity) initializations; and (iii) the initial loss is smaller than any loss obtainable with rank deficiencies -this condition will hold with probability close to 0.5 if the output dimension is 1 (scalar regression) and standard (random) near-zero initialization is employed.

Our result applies to networks with arbitrary depth and input/output dimensions, as well as any configuration of hidden layer widths that does not force rank deficiency (i.e. that meets condition (i)).

The assumptions on initialization (conditions (ii) and (iii)) are necessary, in the sense that violating any one of them may lead to convergence failure.

Moreover, in the case of scalar regression, they are met with constant probability under a random initialization scheme.

We are not aware of any similarly general analysis for efficient convergence of gradient descent to global minimum in deep learning.

The remainder of the paper is organized as follows.

In Section 2 we present the problem of gradient descent training a deep linear neural network by minimizing the 2 loss over a whitened dataset.

Section 3 formally states our assumptions, and presents our convergence analysis.

Key ideas brought forth by our analysis are demonstrated empirically in Section 4.

Section 5 gives a review of relevant literature, including a detailed comparison of our results against those of BID3 .

Finally, Section 6 concludes.

We denote by v the Euclidean norm of a vector v, and by A F the Frobenius norm of a matrix A.We are given a training set {(x (i) , y (i) )} m i=1 ⊂ R dx × R dy , and would like to learn a hypothesis (predictor) from a parametric family H := {h θ : R dx → R dy | θ ∈ Θ} by minimizing the 2 loss: DISPLAYFORM0 When the parametric family in question is the class of linear predictors, i.e. H = {x → W x | W ∈ R dy×dx }, the training loss may be written as L(W ) = and Y ∈ R dy×m are matrices whose columns hold instances and labels respectively.

Suppose now that the dataset is whitened, i.e. has been transformed such that the empirical (uncentered) covariance matrix for instances -Λ xx := 1 m XX ∈ R dx×dx -is equal to identity.

Standard calculations (see Appendix A) show that in this case: DISPLAYFORM1 where Λ yx := 1 m Y X ∈ R dy×dx is the empirical (uncentered) cross-covariance matrix between instances and labels, and c is a constant (that does not depend on W ).

Denoting Φ := Λ yx for brevity, we have that for linear models, minimizing 2 loss over whitened data is equivalent to minimizing the squared Frobenius distance from a target matrix Φ: DISPLAYFORM2 Our interest in this work lies on linear neural networks -fully-connected neural networks with linear activation.

A depth-N (N ∈ N) linear neural network with hidden widths d 1 , . . . , d N −1 ∈ N corresponds to the parametric family of hypotheses H := {x → W N W N −1 · · · W 1 x | W j ∈ R dj ×dj−1 , j = 1, . . .

, N }, where d 0 := d x , d N := d y .

Similarly to the case of a (directly parameterized) linear predictor (Equation (2)), with a linear neural network, minimizing 2 loss over whitened data can be cast as squared Frobenius approximation of a target matrix Φ: DISPLAYFORM3 Note that the notation L N (·) is consistent with that of Equation FORMULA2 , as a network with depth N = 1 precisely reduces to a (directly parameterized) linear model.

We focus on studying the process of training a deep linear neural network by gradient descent, i.e. of tackling the optimization problem in Equation FORMULA3 by iteratively applying the following updates: DISPLAYFORM4 where η > 0 is a configurable learning rate.

In the case of depth N = 1, the training problem in Equation FORMULA3 is smooth and strongly convex, thus it is known (cf.

BID5 ) that with proper choice of η, gradient descent converges to global minimum at a linear rate.

In contrast, for any depth greater than 1, Equation (3) comprises a fundamentally non-convex program, and the convergence properties of gradient descent are highly non-trivial.

Apart from the case N = 2 (shallow network), one cannot hope to prove convergence via landscape arguments, as the strict saddle property is provably violated (see Section 1).

We will see in Section 3 that a direct analysis of the trajectories taken by gradient descent can succeed in this arena, providing a guarantee for linear rate convergence to global minimum.

We close this section by introducing additional notation that will be used in our analysis.

For an arbitrary matrix A, we denote by σ max (A) and σ min (A) its largest and smallest (respectively) singular values.

2 For d ∈ N, we use I d to signify the identity matrix in R d×d .

Given weights W 1 , . . .

, W N of a linear neural network, we let W 1:N be the direct parameterization of the end-to-end linear mapping realized by the network, i.e. W 1: DISPLAYFORM5 , meaning the loss associated with a depth-N network is equal to the loss of the corresponding end-to-end linear model.

In the context of gradient descent, we will oftentimes use (t) as shorthand for the loss at iteration t: DISPLAYFORM6

In this section we establish convergence of gradient descent for deep linear neural networks (Equations (4) and (3)) by directly analyzing the trajectories taken by the algorithm.

We begin in Subsection 3.1 with a presentation of two concepts central to our analysis: approximate balancedness and deficiency margin.

These facilitate our main convergence theorem, delivered in Subsection 3.2.

We conclude in Subsection 3.3 by deriving a convergence guarantee that holds with constant probability over a random initialization.

In our context, the notion of approximate balancedness is formally defined as follows:2 If A ∈ R d×d , σmin(A) stands for the min{d, d }-th largest singular value.

Recall that singular values are always non-negative.

Definition 1.

For δ ≥ 0, we say that the matrices W j ∈ R dj ×dj−1 , j=1, . . .

, N , are δ-balanced if: DISPLAYFORM0 Note that in the case of 0-balancedness, i.e. W j+1 W j+1 = W j W j , ∀j ∈ {1, . . .

, N − 1}, all matrices W j share the same set of non-zero singular values.

Moreover, as shown in the proof of Theorem 1 in BID1 , this set is obtained by taking the N -th root of each non-zero singular value in the end-to-end matrix W 1:N .

We will establish approximate versions of these facts for δ-balancedness with δ > 0, and admit their usage by showing that if the weights of a linear neural network are initialized to be approximately balanced, they will remain that way throughout the iterations of gradient descent.

The condition of approximate balancedness at initialization is trivially met in the special case of linear residual networks DISPLAYFORM1 Moreover, as Claim 2 in Appendix B shows, for a given δ > 0, the customary initialization via random Gaussian distribution with mean zero leads to approximate balancedness with high probability if the standard deviation is sufficiently small.

The second concept we introduce -deficiency margin -refers to how far a ball around the target is from containing rank-deficient (i.e. low rank) matrices.

Definition 2.

Given a target matrix Φ ∈ R d N ×d0 and a constant c > 0, we say that a matrix W ∈ R d N ×d0 has deficiency margin c with respect to Φ if: DISPLAYFORM2 The term "deficiency margin" alludes to the fact that if Equation (6) holds, every matrix W whose distance from Φ is no greater than that of W , has singular values c-bounded away from zero:Claim 1.

Suppose W has deficiency margin c with respect to Φ. Then, any matrix W (of same size as Φ and W ) for which DISPLAYFORM3 Proof.

Our proof relies on the inequality σ min (A+B) ≥ σ min (A)−σ max (B) -see Appendix D.1.We will show that if the weights W 1 , . . .

, W N are initialized such that (they are approximately balanced and) the end-to-end matrix W 1:N has deficiency margin c > 0 with respect to the target Φ, convergence of gradient descent to global minimum is guaranteed.

4 Moreover, the convergence will outpace a particular rate that gets faster when c grows larger.

This suggests that from a theoretical perspective, it is advantageous to initialize a linear neural network such that the end-to-end matrix has a large deficiency margin with respect to the target.

Claim 3 in Appendix B provides information on how likely deficiency margins are in the case of a single output model (scalar regression) subject to customary zero-centered Gaussian initialization.

It shows in particular that if the standard deviation of the initialization is sufficiently small, the probability of a deficiency margin being met is close to 0.5; on the other hand, for this deficiency margin to have considerable magnitude, a non-negligible standard deviation is required.

Taking into account the need for both approximate balancedness and deficiency margin at initialization, we observe a delicate trade-off under the common setting of Gaussian perturbations around zero: if the standard deviation is small, it is likely that weights be highly balanced and a deficiency margin be met; however overly small standard deviation will render high magnitude for the deficiency margin improbable, and therefore fast convergence is less likely to happen; on the opposite end, large standard deviation jeopardizes both balancedness and deficiency margin, putting the entire convergence at risk.

This trade-off is reminiscent of empirical phenomena in deep learning, by 3 Note that deficiency margin c > 0 with respect to Φ implies σmin(Φ) > 0, i.e. Φ has full rank.

Our analysis can be extended to account for rank-deficient Φ by replacing σmin(Φ) in Equation (6) with the smallest positive singular value of Φ, and by requiring that the end-to-end matrix W1:N be initialized such that its left and right null spaces coincide with those of Φ. Relaxation of this requirement is a direction for future work.

4 In fact, a deficiency margin implies that all critical points in the respective sublevel set (set of points with smaller loss value) are global minima.

This however is far from sufficient for proving convergence, as sublevel sets are unbounded, and the loss landscape over them is non-convex and non-smooth.

Indeed, we show in Appendix C that deficiency margin alone is not enough to ensure convergence -without approximate balancedness, the lack of smoothness can cause divergence.

which small initialization can bring forth efficient convergence, while if exceedingly small, rate of convergence may plummet ("vanishing gradient problem"), and if made large, divergence becomes inevitable ("exploding gradient problem").

The common resolution of residual connections (He et al., 2016) is analogous in our context to linear residual networks, which ensure perfect balancedness, and allow large deficiency margin if the target is not too far from identity.

Using approximate balancedness (Definition 1) and deficiency margin (Definition 2), we present our main theorem -a guarantee for linear convergence to global minimum: Theorem 1.

Assume that gradient descent is initialized such that the end-to-end matrix W 1:N (0) has deficiency margin c > 0 with respect to the target Φ, and the weights DISPLAYFORM0 .

Suppose also that the learning rate η meets: DISPLAYFORM1 Then, for any > 0 and: DISPLAYFORM2 the loss at iteration T of gradient descent -(T ) -is no greater than .

The assumptions made in Theorem 1 -approximate balancedness and deficiency margin at initialization -are both necessary, in the sense that violating any one of them may lead to convergence failure.

We demonstrate this in Appendix C. In the special case of linear residual networks (uniform dimensions and identity initialization), a sufficient condition for the assumptions to be met is that the target matrix have (Frobenius) distance less than 0.5 from identity.

This strengthens one of the central results in BID3 (see Section 5).

For a setting of random near-zero initialization, we present in Subsection 3.3 a scheme that, when the output dimension is 1 (scalar regression), ensures assumptions are satisfied (and therefore gradient descent efficiently converges to global minimum) with constant probability.

It is an open problem to fully analyze gradient descent under the common initialization scheme of zero-centered Gaussian perturbations applied to each layer independently.

We treat this scenario in Appendix B, providing quantitative results concerning the likelihood of each assumption (approximate balancedness or deficiency margin) being met individually.

However the question of how likely it is that both assumptions be met simultaneously, and how that depends on the standard deviation of the Gaussian, is left for future work.

An additional point to make is that Theorem 1 poses a structural limitation on the linear neural network.

Namely, it requires the dimension of each hidden layer (d i , i = 1, . . .

, N −1) to be greater than or equal to the minimum between those of the input (d 0 ) and output (d N ).

Indeed, in order for the initial end-to-end matrix W 1:N (0) to have deficiency margin c > 0, it must (by Claim 1) have full rank, and this is only possible if there is no intermediate dimension DISPLAYFORM0 We make no other assumptions on network architecture (depth, input/output/hidden dimensions).

The cornerstone upon which Theorem 1 rests is the following lemma, showing non-trivial descent whenever σ min (W 1:N ) is bounded away from zero:Lemma 1.

Under the conditions of Theorem 1, we have that for every t = 0, 1, 2, . . .

: DISPLAYFORM0 5 Note that the term dL 1 dW (W1:N (t)) below stands for the gradient of L 1 (·) -a convex loss over (directly parameterized) linear models (Equation FORMULA2 ) -at the point W1:N (t) -the end-to-end matrix of the network at iteration t.

It is therefore (see Equation FORMULA6 ) non-zero anywhere but at a global minimum.

Proof of Lemma 1 (in idealized setting; for complete proof see Appendix D.2).

We prove the lemma here for the idealized setting of perfect initial balancedness (δ = 0): DISPLAYFORM1 and infinitesimally small learning rate (η → 0 + ) -gradient flow: DISPLAYFORM2 where τ is a continuous time index, and dot symbol (inẆ j (τ )) signifies derivative with respect to time.

The complete proof, for the realistic case of approximate balancedness and discrete updates (δ, η > 0), is similar but much more involved, and appears in Appendix D.2.Recall that (t) -the objective value at iteration t of gradient descent -is equal to L 1 (W 1:N (t)) (see Equation FORMULA6 ).

Accordingly, for the idealized setting in consideration, we would like to show: DISPLAYFORM3 We will see that a stronger version of Equation FORMULA1 holds, namely, one without the 1/2 factor (which only appears due to discretization).By (Theorem 1 and Claim 1 in) BID1 , the weights W 1 (τ ), . . .

, W N (τ ) remain balanced throughout the entire optimization, and that implies the end-to-end matrix W 1:N (τ ) moves according to the following differential equation: DISPLAYFORM4 where vec(A), for an arbitrary matrix A, stands for vectorization in column-first order, and P W 1:N (τ ) is a positive semidefinite matrix whose eigenvalues are all greater than or equal to DISPLAYFORM5 Taking the derivative of L 1 (W 1:N (τ )) with respect to time, we obtain the sought-after Equation (10) (with no 1/2 factor): DISPLAYFORM6 The first transition here (equality) is an application of the chain rule; the second (equality) plugs in Equation FORMULA1 ; the third (inequality) results from the fact that the eigenvalues of the symmetric matrix P W 1:N (τ ) are no smaller than σ min (W 1:N (τ )) 2(N −1)/N (recall that · stands for Euclidean norm); and the last (equality) is trivial -A F = vec(A) for any matrix A.With Lemma 1 established, the proof of Theorem 1 readily follows: DISPLAYFORM7 Plugging this into Equation (9) while recalling that (t) = L 1 (W 1:N (t)) (Equation FORMULA6 ), we have (by Lemma 1) that for every t = 0, 1, 2, . . . : DISPLAYFORM8 Published as a conference paper at ICLR 2019Since the coefficients 1 − η · σ min (W 1:N (t)) 2(N −1) N are necessarily non-negative (otherwise would contradict non-negativity of L 1 (·)), we may unroll the inequalities, obtaining: DISPLAYFORM9 Now, this in particular means that for every t = 0, 1, 2, . . .

: DISPLAYFORM10 Deficiency margin c of W 1:N (0) along with Claim 1 thus imply σ min W 1:N (t ) ≥ c, which when inserted back into Equation (12) yields, for every t = 1, 2, 3, . . .

: DISPLAYFORM11 is obviously non-negative, and it is also no greater than 1 (otherwise would contradict non-negativity of L 1 (·)).

We may therefore incorporate the inequality FORMULA1 : DISPLAYFORM12 DISPLAYFORM13 Recalling again that (t) = L 1 (W 1:N (t)) (Equation FORMULA6 ), we conclude the proof.

We define the following procedure, balanced initialization, which assigns weights randomly while ensuring perfect balancedness: DISPLAYFORM0 . .

, N , assigns these weights as follows: DISPLAYFORM1 (ii) Take singular value decomposition A = U ΣV , where DISPLAYFORM2 where the symbol " " stands for equality up to zero-valued padding.

DISPLAYFORM3 The concept of balanced initialization, together with Theorem 1, leads to a guarantee for linear convergence (applicable to output dimension 1 -scalar regression) that holds with constant probability over the randomness in initialization:Theorem 2.

For any constant 0 < p < 1/2, there are constants d 0 , a > 0 8 such that the following holds.

Assume d N = 1, d 0 ≥ d 0 , and that the weights W 1 (0), . . .

, W N (0) are subject to balanced initialization (Procedure 1) such that the entries in W 1:N (0) are independent zero-centered Gaussian perturbations with standard deviation s ≤ Φ 2 / ad 2 0 .

Suppose also that we run gradient 6 These assignments can be accomplished since min{d1, . . .

, dN−1} ≥ min{d0, dN }.

7 By design W1:N = A and W j+1 Wj+1 = WjW j , ∀j ∈ {1, . . .

, N −1} -these properties are actually all we need in Theorem 2, and step (iii) in Procedure 1 can be replaced by any assignment that meets them. .

Then, with probability at least p over the random initialization, we have that for every > 0 and: DISPLAYFORM4 the loss at iteration T of gradient descent -(T ) -is no greater than .Proof.

See Appendix D.3.

Balanced initialization (Procedure 1) possesses theoretical advantages compared with the customary layer-wise independent scheme -it allowed us to derive a convergence guarantee that holds with constant probability over the randomness of initialization (Theorem 2).

In this section we present empirical evidence suggesting that initializing with balancedness may be beneficial in practice as well.

For conciseness, some of the details behind our implementation are deferred to Appendix E.We began by experimenting in the setting covered by our analysis -linear neural networks trained via gradient descent minimization of 2 loss over whitened data.

The dataset chosen for the experiment was UCI Machine Learning Repository's "Gas Sensor Array Drift at Different Concentrations" (Vergara et al., 2012; Rodriguez-Lujan et al., 2014) .

Specifically, we used the dataset's "Ethanol" problem -a scalar regression task with 2565 examples, each comprising 128 features (one of the largest numeric regression tasks in the repository).

Starting with the customary initialization of layer-wise independent random Gaussian perturbations centered at zero, we trained a three layer network (N = 3) with hidden widths (d 1 , d 2 ) set to 32, and measured the time (number of iterations) it takes to converge (reach training loss within = 10 −5 from optimum) under different choices of standard deviation for the initialization.

To account for the possibility of different standard deviations requiring different learning rates (values for η), we applied, for each standard deviation independently, a grid search over learning rates, and recorded the one that led to fastest convergence.

The result of this test is presented in FIG2 (a).

As can be seen, there is a range of standard deviations that leads to fast convergence (a few hundred iterations or less), below and above which optimization decelerates by orders of magnitude.

This accords with our discussion at the end of Subsection 3.3, by which overly small initialization ensures approximate balancedness (small δ; see Definition 1) but diminishes deficiency margin (small c; see Definition 2) -"vanishing gradient problem" -whereas large initialization hinders both approximate balancedness and deficiency margin -"exploding gradient problem".

In that regard, as a sanity test for the validity of our analysis, in a case where approximate balancedness is met at initialization (small standard deviation), we measured its persistence throughout optimization.

As FIG2 (c) shows, our theoretical findings manifest themselves here -trajectories of gradient descent indeed preserve weight balancedness.

In addition to a three layer network, we also evaluated a deeper, eight layer model (with hidden widths identical to the former -N = 8, d 1 = · · · = d 7 = 32).

In particular, using the same experimental protocol as above, we measured convergence time under different choices of standard deviation for the initialization.

FIG2 (a) displays the result of this test alongside that of the three layer model.

As the figure shows, transitioning from three layers to eight aggravated the instability with respect to initialization -there is now a narrow band of standard deviations that lead to convergence in reasonable time, and outside of this band convergence is extremely slow, to the point where it does not take place within the duration we allowed (10 6 iterations).

From the perspective of our analysis, a possible explanation for the aggravation is as follows: under layer-wise independent initialization, the magnitude of the end-to-end matrix W 1:N depends on the standard deviation in a manner that is exponential in depth, thus for large depths the range of standard deviations that lead to moderately sized W 1:N (as required for a deficiency margin) is limited, and within this range, there may not be many standard deviations small enough to ensure approximate balancedness.

The procedure of balanced initialization (Procedure 1) circumvents these difficulties -it assigns W 1:N directly (no exponential dependence on depth), and distributes its content between the individual weights W 1 , . . .

, W N in a perfectly balanced fashion.

Rerunning the experiment of FIG2 (a) with this initialization replacing the customary layer-wise scheme (using same experimental protocol), we obtained the results shown in FIG2 −3 , this plot shows degree of balancedness (minimal δ satisfying W j+1 Wj+1 − WjW j F ≤ δ , ∀j ∈ {1, . . .

, N − 1}) against magnitude of weights (minj=1,...,N WjW j F ) throughout optimization.

Notice that approximate balancedness persists under gradient descent, in line with our theoretical analysis.

(d) Convergence of stochastic gradient descent training the fully-connected non-linear (ReLU) neural network of the MNIST tutorial built into TensorFlow (details in text).

Customary layer-wise independent and balanced initializations -both based on Gaussian perturbations centered at zero -are evaluated, with varying standard deviations.

For each configuration 10 epochs of optimization are run, followed by measurement of the training loss.

Notice that although our theoretical analysis does not cover non-linear activation, softmax-cross-entropy loss and stochastic optimization, the conclusion of balanced initialization leading to improved convergence carries over to this setting.

As a final experiment, we evaluated the effect of balanced initialization in a setting that involves non-linear activation, softmax-cross-entropy loss and stochastic optimization (factors not accounted for by our analysis).

For this purpose, we turned to the MNIST tutorial built into TensorFlow BID0 , 9 which comprises a fully-connected neural network with two hidden layers (width 128 followed by 32) and ReLU activation (Nair and Hinton, 2010), trained through stochastic gradient descent (over softmax-cross-entropy loss) with batch size 100, initialized via customary layer-wise independent Gaussian perturbations centered at zero.

While keeping the learning rate at its default value 0.01, we varied the standard deviation of initialization, and for each value measured the training loss after 10 epochs.

10 We then replaced the original (layer-wise independent) initialization with a balanced initialization based on Gaussian perturbations centered at zero (latter was implemented per Procedure 1, disregarding non-linear activation), and repeated the process.

The results of this experiment are shown in FIG2 .

Although our theoretical analysis does not cover non-linear activation, softmax-cross-entropy loss or stochasticity in optimization, its conclusion of balanced initialization leading to improved (faster and more stable) convergence carried over to such setting.

Theoretical study of gradient-based optimization in deep learning is a highly active area of research.

As discussed in Section 1, a popular approach is to show that the objective landscape admits the properties of no poor local minima and strict saddle, which, by Ge et al. (2015) ; Lee et al. (2016) ; Panageas and Piliouras (2017) , ensure convergence to global minimum.

Many works, both classic (e.g. BID2 ) and recent (e.g. BID9 ; Kawaguchi FORMULA1 FORMULA1 ), have focused on the validity of these properties in different deep learning settings.

Nonetheless, to our knowledge, the success of landscape-driven analyses in formally proving convergence to global minimum for a gradient-based algorithm, has thus far been limited to shallow (two layer) models only (e.g. Ge et al. (2016) ; Du and Lee (2018); BID12 ).An alternative to the landscape approach is a direct analysis of the trajectories taken by the optimizer.

Various papers (e.g. BID6 FORMULA1 ) have recently adopted this strategy, but their analyses only apply to shallow models.

In the context of linear neural networks, deep (three or more layer) models have also been treated -cf.

Saxe et al. FORMULA1 and BID1 , from which we draw certain technical ideas for proving Lemma 1.

However these treatments all apply to gradient flow (gradient descent with infinitesimally small learning rate), and thus do not formally address the question of computational efficiency.

To our knowledge, BID3 is the only existing work rigorously proving convergence to global minimum for a conventional gradient-based algorithm training a deep model.

This work is similar to ours in the sense that it also treats linear neural networks trained via minimization of 2 loss over whitened data, and proves linear convergence (to global minimum) for gradient descent.

It is more limited in that it only covers the subclass of linear residual networks, i.e. the specific setting of uniform width across all layers (d 0 = · · · = d N ) along with identity initialization.

We on the other hand allow the input, output and hidden dimensions to take on any configuration that avoids "bottlenecks" (i.e. admits min{d 1 , . . .

DISPLAYFORM0 , and from initialization require only approximate balancedness (Definition 1), supporting many options beyond identity.

In terms of the target matrix Φ, BID3 treats two separate scenarios:11 (i) Φ is symmetric and positive definite; and (ii) Φ is within distance 1/10e from identity.12 Our analysis does not fully account for scenario (i), which seems to be somewhat of a singularity, where all layers are equal to each other throughout optimization (see proof of Theorem 2 in BID3 ).

We do however provide a strict generalization of scenario (ii) -our assumption of deficiency margin (Definition 2), in the setting of linear residual networks, is met if the distance between target and identity is less than 0.5.

For deep linear neural networks, we have rigorously proven convergence of gradient descent to global minima, at a linear rate, provided that the initial weight matrices are approximately balanced and the initial end-to-end matrix has positive deficiency margin.

The result applies to networks with arbitrary depth, and any configuration of input/output/hidden dimensions that supports full rank, i.e. in which no hidden layer has dimension smaller than both the input and output.

Our assumptions on initialization -approximate balancedness and deficiency margin -are both necessary, in the sense that violating any one of them may lead to convergence failure, as we demonstrated explicitly.

Moreover, for networks with output dimension 1 (scalar regression), we have shown that a balanced initialization, i.e. a random choice of the end-to-end matrix followed by a balanced partition across all layers, leads assumptions to be met, and thus convergence to take place, with constant probability.

Rigorously proving efficient convergence with significant probability under customary layer-wise independent initialization remains an open problem.

The recent work of Shamir (2018) suggests that this may not be possible, as at least in some settings, the number of iterations required for convergence is exponential in depth with overwhelming probability.

This negative result, a theoretical manifestation of the "vanishing gradient problem", is circumvented by balanced initialization.

Through simple experiments we have shown that the latter can lead to favorable convergence in deep learning practice, as it does in theory.

Further investigation of balanced initialization, including development of variants for convolutional layers, is regarded as a promising direction for future research.

The analysis in this paper uncovers special properties of the optimization landscape in the vicinity of gradient descent trajectories.

We expect similar ideas to prove useful in further study of gradient descent on non-convex objectives, including training losses of deep non-linear neural networks.

A 2 LOSS OVER WHITENED DATA Recall the 2 loss of a linear predictor W ∈ R dy×dx as defined in Section 2: DISPLAYFORM0 By definition, when data is whitened, Λ xx is equal to identity, yielding: For approximate balancedness we have the following claim, which shows that it becomes more and more likely the smaller the standard deviation of initialization is: DISPLAYFORM1 Claim 2.

Assume all entries in the matrices W j ∈ R dj ×dj−1 , j = 1, . . .

, N , are drawn independently at random from a Gaussian distribution with mean zero and standard deviation s > 0.

Then, for any δ > 0, the probability of W 1 , . . .

, W N being δ-balanced is at least max{0, 1 − 10δ DISPLAYFORM2 In terms of deficiency margin, the claim below treats the case of a single output model (scalar regression), and shows that if the standard deviation of initialization is sufficiently small, with probability close to 0.5, a deficiency margin will be met.

However, for this deficiency margin to meet a chosen threshold c, the standard deviation need be sufficiently large.

Claim 3.

There is a constant C 1 > 0 such that the following holds.

Consider the case where DISPLAYFORM3 13 and suppose all entries in the matrices W j ∈ R dj ×dj−1 , j = 1, . . .

, N , are drawn independently at random from a Gaussian distribution with mean zero, whose standard deviation s > 0 is small with respect to the target, i.e. DISPLAYFORM4 , the probability of the end-to-end matrix W 1:N having deficiency margin c with respect to Φ is at least 0.49 if: DISPLAYFORM5 Proof.

See Appendix D.5.

13 The requirement d0 ≥ 20 is purely technical, designed to simplify expressions in the claim.

14 The probability 0.49 can be increased to any p < 1/2 by increasing the constant 10 5 in the upper bounds for s and c.15 It is not difficult to see that the latter threshold is never greater than the upper bound for s, thus sought-after standard deviations always exist.

In this appendix we show that the assumptions on initialization facilitating our main convergence result (Theorem 1) -approximate balancedness and deficiency margin -are both necessary, by demonstrating cases where violating each of them leads to convergence failure.

This accords with widely observed empirical phenomena, by which successful optimization in deep learning crucially depends on careful initialization (cf.

Sutskever et al. (2013) ).Claim 4 below shows 16 that if one omits from Theorem 1 the assumption of approximate balancedness at initialization, no choice of learning rate can guarantee convergence:Claim 4.

Assume gradient descent with some learning rate η > 0 is a applied to a network whose depth N is even, and whose input, output and hidden dimensions d 0 , . . .

, d N are all equal to some d ∈ N. Then, there exist target matrices Φ such that the following holds.

For any c with 0 < c < σ min (Φ), there are initializations for which the end-to-end matrix W 1:N (0) has deficiency margin c with respect to Φ, and yet convergence will fail -objective will never go beneath a positive constant.

Proof.

See Appendix D.6.In terms of deficiency margin, we provide (by adapting Theorem 4 in BID3 ) a different, somewhat stronger result -there exist settings where initialization violates the assumption of deficiency margin, and despite being perfectly balanced, leads to convergence failure, for any choice of learning rate:

Claim 5.

Consider a network whose depth N is even, and whose input, output and hidden dimensions d 0 , . . .

, d N are all equal to some d ∈ N. Then, there exist target matrices Φ for which there are non-stationary initializations W 1 (0), . . .

, W N (0) that are 0-balanced, and yet lead gradient descent, under any learning rate, to fail -objective will never go beneath a positive constant.

Proof.

See Appendix D.7.

We introduce some additional notation here in addition to the notation specified in Section 2.

We use A σ to denote the spectral norm (largest singular value) of a matrix A, and sometimes v 2 as an alternative to v -the Euclidean norm of a vector v. Recall that for a matrix A, vec(A) is its vectorization in column-first order.

We let F (·) denote the cumulative distribution function of the standard normal distribution, i.e. F (x) = DISPLAYFORM0 To simplify the presentation we will oftentimes use W as an alternative (shortened) notation for W 1:N -the end-to-end matrix of a linear neural network.

We will also use L(·) as shorthand for L 1 (·) -the loss associated with a (directly parameterized) linear model, i.e. L(W ) := DISPLAYFORM1 F .

Therefore, in the context of gradient descent training a linear neural network, the following expressions all represent the loss at iteration t: DISPLAYFORM2 Also, for weights W j ∈ R dj ×dj−1 , j = 1, . . .

, N of a linear neural network, we generalize the notation W 1:N , and define W j:j := W j W j −1 · · · W j for every 1 ≤ j ≤ j ≤ N .

Note that W j:j = W j W j+1 · · · W j .

Then, by a simple gradient calculation, the gradient descent updates (4) can be written as DISPLAYFORM3 where we define W 1:0 (t) := I d0 and W N +1:N (t) := I d N for completeness.

16 For simplicity of presentation, the claim treats the case of even depth and uniform dimension across all layers.

It can easily be extended to account for arbitrary depth and input/output/hidden dimensions.

17 This statement becomes trivial if one allows initialization at a suboptimal stationary point, e.g. Wj(0) = 0, j = 1, . . .

, N .

Claim 5 rules out such trivialities by considering only non-stationary initializations.

Finally, recall the standard definition of the tensor product of two matrices (also known as the Kronecker product): for matrices A ∈ R m A ×n A , B ∈ R m B ×n B , their tensor product A ⊗ B ∈ R m A m B ×n A n B is defined as DISPLAYFORM4 where a i,j is the element in the i-th row and j-th column of A.

Proof.

Recall that for any matrices A and B of compatible sizes σ min (A + B) ≥ σ min (A) − σ max (B), and that the Frobenius norm of a matrix is always lower bounded by its largest singular value (Horn and Johnson (1990) ).

Using these facts, we have: DISPLAYFORM0

To prove Lemma 1, we will in fact prove a stronger result, Lemma 2 below, which states that for each iteration t, in addition to (9) being satisfied, certain other properties are also satisfied, namely: (i) the weight matrices W 1 (t), . . .

, W N (t) are 2δ-balanced, and (ii) W 1 (t), . . .

, W N (t) have bounded spectral norms.

Lemma 2.

Suppose the conditions of Theorem 1 are satisfied.

Then for all t ∈ N ∪ {0}, DISPLAYFORM0 First we observe that Lemma 1 is an immediate consequence of Lemma 2.Proof of Lemma 1.

Notice that condition B(t) of Lemma 2 for each t ≥ 1 immediately establishes the conclusion of Lemma 1 at time step t − 1.

We next prove some preliminary lemmas which will aid us in the proof of Lemma 2.

The first is a matrix inequality that follows from Lidskii's theorem.

For a matrix A, let Sing(A) denote the rectangular diagonal matrix of the same size, whose diagonal elements are the singular values of A arranged in non-increasing order (starting from the (1, 1) position).

Using Lemma 3, we get: Lemma 4.

Suppose D 1 , D 2 ∈ R d×d are non-negative diagonal matrices with non-increasing values along the diagonal and O ∈ R d×d is an orthogonal matrix.

Suppose that D 1 − OD 2 O F ≤ , for some > 0.

Then: DISPLAYFORM0 Proof.

Since D 1 and OD 2 O T are both symmetric positive semi-definite matrices, their singular values are equal to their eigenvalues.

Moreover, the singular values of D 1 are simply its diagonal elements and the singular values of OD 2 O T are simply the diagonal elements of D 2 .

Thus by Lemma 3 we get that DISPLAYFORM1 , and by the triangle inequality it follows that DISPLAYFORM2 DISPLAYFORM3 and that for some ν > 0, M > 0, the matrices DISPLAYFORM4 and for 1 ≤ j ≤ N , W j σ ≤ M .

Then, for 1 ≤ j ≤ N , DISPLAYFORM5 and DISPLAYFORM6 Moreover, if σ min denotes the minimum singular value of W 1:N , σ 1,min denotes the minimum singular value of W 1 and σ N,min denotes the minimum singular value of W N , then DISPLAYFORM7 Proof.

For 1 ≤ j ≤ N , let us write the singular value decomposition of W j as W j = U j Σ j V j , where U j ∈ R dj ×dj and V j ∈ R dj−1×dj−1 are orthogonal matrices and Σ j ∈ R dj ×dj−1 is diagonal.

We may assume without loss of generality that the singular values of W j are non-increasing along the diagonal of Σ j .

Then we can write (15) as DISPLAYFORM8 Since the Frobenius norm is invariant to orthogonal transformations, we get that DISPLAYFORM9 By Lemma 4, we have that DISPLAYFORM10 We may rewrite the latter of these two inequalities as DISPLAYFORM11 For matrices A, B, we have that AB F ≤ A σ · B F .

Therefore, for j + 1 ≤ i ≤ N , we have that DISPLAYFORM12 We now argue that DISPLAYFORM13 verifying the case k = 1.

To see the general case, since square diagonal matrices commute, we have that DISPLAYFORM14

By the triangle inequality, we then have that DISPLAYFORM0 By an identical argument (formally, by replacing W j with W N −j+1 ), we get that DISPLAYFORM1 (19) and FORMULA2 verify FORMULA1 and FORMULA1 , respectively, so it only remains to verify (18).Letting j = 1 in (19), we get DISPLAYFORM2 Let us write the eigendecomposition of W 1:N W 1:N with an orthogonal eigenbasis as W 1:N W 1:N = U ΣU , where Σ is diagonal with its (non-negative) elements arranged in non-increasing order and U is orthogonal.

We can write the left hand side of FORMULA1 DISPLAYFORM3 By Lemma 4, we have that DISPLAYFORM4 Recall that W ∈ R d N ×d0 .

Suppose first that d N ≤ d 0 .

Let σ min denote the minimum singular value of W 1:N (so that σ 2 min is the element in the (d N , d N DISPLAYFORM5 , and σ N,min denote the minimum singular value (i.e. diagonal element) of Σ N , which lies in the DISPLAYFORM6 By an identical argument using FORMULA2 , we get that, in the case that d 0 ≤ d N , if σ 1,min denotes the minimum singular value of Σ 1 , then DISPLAYFORM7 (Notice that we have used the fact that the nonzero eigenvalues of DISPLAYFORM8 Proof.

For 1 ≤ j ≤ N , let us write the singular value decomposition of W j as W j = U j Σ j V j , where the singular values of W j are decreasing along the main diagonal of Σ j .

By Lemma 4, we have that for DISPLAYFORM9 Write M = max 1≤j≤N W j σ = max 1≤j≤N Σ j σ .

By the above we have that DISPLAYFORM10 Let the singular value decomposition of W 1:N be denoted by W 1:N = U ΣV , so that Σ σ ≤ C. Then by (17) of Lemma 5 and Lemma 4 (see also FORMULA2 , where the same argument was used), we have that DISPLAYFORM11 Now recall that ν is chosen so that ν ≤ C 2/N 30·N 2 .

Suppose for the purpose of contradiction that there is some j such that W j W j σ > 2 1/N C 2/N .

Then it must be the case that DISPLAYFORM12 where we have used that 2 1/N − (5/4) 1/N ≥ 1 30N for all N ≥ 2, which follows by considering the Laurent series exp(1/z) = DISPLAYFORM13 We now rewrite inequality (24) as DISPLAYFORM14 Next, using FORMULA2 and (1 + 1/x) x ≤ e for all x > 0, DISPLAYFORM15 Since DISPLAYFORM16 , we get by combining FORMULA2 and FORMULA2 that DISPLAYFORM17 and since 1 − e/20 > 1/(5/4), it follows that Σ N Σ N σ < (5/4) 1/N C 2/N , which contradicts (24).

It follows that for all 1 ≤ j ≤ N , W j W j σ ≤ 2 1/N C 2/N .

The conclusion of the lemma then follows from the fact that DISPLAYFORM18

Lemma 7 below states that if certain conditions on W 1 (t), . . .

, W N (t) are met, the sought-after descent -Equation (9) -will take place at iteration t. We will later show (by induction) that the required conditions indeed hold for every t, thus the descent persists throughout optimization.

The proof of Lemma 7 is essentially a discrete, single-step analogue of the continuous proof for Lemma 1 (covering the case of gradient flow) given in Section 3.

Lemma 7.

Assume the conditions of Theorem 1.

Moreover, suppose that for some t, the matrices W 1 (t), . . .

, W N (t) and the end-to-end matrix W (t) := W 1:N (t) satisfy the following properties: DISPLAYFORM0 Then, after applying a gradient descent update (4) we have that DISPLAYFORM1 Proof.

For simplicity write M = (4 Φ F ) 1/N and B = Φ F .

We first claim that DISPLAYFORM2 Since c ≤ σ min , for (27) to hold it suffices to have DISPLAYFORM3 which is guaranteed by (7).Next, we claim that DISPLAYFORM4 The second inequality above is trivial, and for the first to hold, since c ≤ Φ F , it suffices to take DISPLAYFORM5 which is guaranteed by the definition of δ in Theorem 1.Next we continue with the rest of the proof.

It follows from (14) that DISPLAYFORM6 where ( ) denotes higher order terms in η.

We now bound the Frobenius norm of ( ).

To do this, note that since DISPLAYFORM7 18 Here, for matrices A1, . . .

, AK such that AK AK−1 · · · A1 is defined, we write DISPLAYFORM8 where the last inequality uses ηM N −2 BN ≤ 1/2, which is a consequence of (27).

Next, by Lemma 5 with ν = 2δ, DISPLAYFORM9 Next, by standard properties of tensor product, we have that DISPLAYFORM10 Let us write eigenvalue decompositions DISPLAYFORM11 If λ D denotes the minimum diagonal element of D and λ E denotes the minimum diagonal element of E, then the minimum diagonal element of Λ is therefore at least λ It follows as a result of the above inequalities that if we write DISPLAYFORM12 DISPLAYFORM13 Then we have DISPLAYFORM14 where the first inequality follows since DISPLAYFORM15 F is 1-smooth as a function of W .

Next, by FORMULA2 and FORMULA3 , DISPLAYFORM16 Thus DISPLAYFORM17 By (27, 28), which bound η, 2δ, respectively, we have that DISPLAYFORM18

Proof of Lemma 2.

We use induction on t, beginning with the base case t = 0.

Since the weights W 1 (0), . . . , W N (0) are δ-balanced, we get that A(0) holds automatically.

To establish B(0), note that since W 1:N (0) has deficiency margin c > 0 with respect to Φ, we must have DISPLAYFORM0 To show that the above implies C(0), we use condition A(0) and Lemma 6 with C = 2 Φ F and ν = 2δ.

By the definition of δ in Theorem 1 and since c ≤ Φ F , we have that DISPLAYFORM1 as required by Lemma 6.

As A(0) and (33) verify the preconditions 1. and 2., respectively, of Lemma 6, it follows that for 1 verifying C(0) and completing the proof of the base case.

DISPLAYFORM2 The proof of Lemma 2 follows directly from the following inductive claims.

To prove this, we use Lemma 7.

We verify first that the preconditions hold.

First, C(t) immediately gives condition 1. of Lemma 7.

By B(t), we have that W (t) − Φ σ ≤ W (t) − Φ F ≤ Φ F , giving condition 2. of Lemma 7.

A(t) immediately gives condition 3. of Lemma 7.

Finally, by B(t), we have that DISPLAYFORM0 , establishing B(t + 1).

(0) , . . . , B(t), C(t) ⇒ A(t + 1), A (t + 1).

To prove this, note that for 1 ≤ j ≤ N − 1, DISPLAYFORM1 DISPLAYFORM2 By B(0), . . .

, B(t), W 1:N (t) − Φ F ≤ Φ F .

By the triangle inequality it then follows that W 1: DISPLAYFORM3 .

By Lemma 6 with C = 2 Φ F , ν = 2δ (so that (34) is satisfied), DISPLAYFORM4 In the first inequality above, we have also used the fact that for matrices A, B such that AB is defined, AB F ≤ A σ B F .

FORMULA3 gives us A (t + 1).We next establish A(t + 1).

By B(i) for 0 ≤ i ≤ t, we have that DISPLAYFORM5 Using A (i) for 0 ≤ i ≤ t and summing over i gives DISPLAYFORM6 Next, by B(0), . . .

, B(t), we have that L(W (i)) ≤ L(W (0)) for i ≤ t. Since W (0) has deficiency margin of c and by Claim 1, it then follows that σ min (W (i)) ≥ c for all i ≤ t.

Therefore, by summing B(0), . . . , B(t), DISPLAYFORM7 Therefore, DISPLAYFORM8 where FORMULA3 follows from the definition of η in FORMULA12 , and the last equality follows from definition of δ in Theorem 1.

By (36), it follows that DISPLAYFORM9 verifying A(t + 1).

We apply Lemma 6 with ν = 2δ and C = 2 Φ F .

First, the triangle inequality and B(t) give DISPLAYFORM0 verifying precondition 2. of Lemma 6.

A(t) verifies condition 1. of Lemma 6, so for DISPLAYFORM1 The proof of Lemma 2 then follows by induction on t.

Theorem 2 is proven by combining Lemma 8 below, which implies that the balanced initialization is likely to lead to an end-to-end matrix W 1:N (0) with sufficiently large deficiency margin, with Theorem 1, which establishes convergence.

DISPLAYFORM0 be a vector.

Suppose that µ is a rotation-invariant distribution 19 over R d with a well-defined density, such that, for some 0 < < 1, DISPLAYFORM1 Then, with probability at least DISPLAYFORM2 , V will have deficiency margin Φ 2 /(b 2 d) with respect to Φ. 19 Recall that a distribution on vectors V ∈ R d is rotation-invariant if the distribution of V is the same as the distribution of OV , for any orthogonal d × d matrix O. If V has a well-defined density, this is equivalent to the statement that for any r > 0, the distribution of V conditioned on V 2 = r is uniform over the sphere centered at the origin with radius r.

The proof of Lemma 8 is postponed to Appendix D.5, where Lemma 8 will be restated as Lemma 16.One additional technique is used in the proof of Theorem 2, which leads to an improvement in the guaranteed convergence rate.

Because the deficiency margin of W 1:N (0) is very small, namely O( Φ 2 /d 0 ) (which is necessary for the theorem to maintain constant probability), at the beginning of optimization, (t) will decrease very slowly.

However, after a certain amount of time, the deficiency margin of W 1:N (t) will increase to a constant, at which point the decrease of (t) will be much faster.

To capture this acceleration, we apply Theorem 1 a second time, using the larger deficiency margin at the new "initialization." From a geometric perspective, we note that the matrices W 1 (0), . . .

, W N (0) are very close to 0, and the point at which W j (0) = 0 for all j is a saddle.

Thus, the increase in (t) − (t + 1) over time captures the fact that the iterates FIG2 , . . .

, W N (t)) escape a saddle point.

Proof of Theorem 2.

Choose some a ≥ 2, to be specified later.

By assumption, all entries of the end-to-end matrix at time 0, W 1:N (0), are distributed as independent Gaussians of mean 0 and standard deviation s ≤ Φ 2 / ad 2 0 .

We will apply Lemma 8 to the vector W 1:N (0) ∈ R d0 .

Since its distribution is obviously rotation-invariant, in remains to show that the distribution of the norm W 1:N (0) 2 is not too spread out.

The following lemma -a direct consequence of the Chernoff bound applied to the χ 2 distribution with d 0 degrees of freedom -will give us the desired result:Lemma 9 (Laurent and Massart FORMULA2 , Lemma 1).

Suppose that d ∈ N and V ∈ R d is a vector whose entries are i.i.d.

Gaussians with mean 0 and standard deviation s.

Then, for any k > 0, DISPLAYFORM0 By Lemma 9 with k = d 0 /16, we have that DISPLAYFORM1 We next use Lemma 8, with DISPLAYFORM2 ; note that since a ≥ 2, b 1 ≥ 1, as required by the lemma.

Lemma 8 then implies that with probability at least DISPLAYFORM3 W 1:N (0) will have deficiency margin s 2 d 0 /2 Φ 2 with respect to Φ. By the definition of balanced initialization (Procedure 1) W 1 (0), . . .

, W N (0) are 0-balanced.

Since 2 4 · 6144 < 10 5 , our assumption on η gives DISPLAYFORM4 so that Equation FORMULA12 holds with c = DISPLAYFORM5 .

The conditions of Theorem 1 thus hold with probability at least that given in Equation (38).

In such a constant probability event, by Theorem 1 (and the fact that a positive deficiency margin implies DISPLAYFORM6 2 ), if we choose DISPLAYFORM7 then DISPLAYFORM8 Moreover, by condition A(t 0 ) of Lemma 2 and the definition of δ in Theorem 1, we have, for DISPLAYFORM9 We now apply Theorem 1 again, verifying its conditions again, this time with the initialization (W 1 (t 0 ), . . .

, W N (t 0 )).

First note that the end-to-end matrix W 1:N (t 0 ) has deficiency margin c = Φ 2 /2 as shown above.

The learning rate η, by Equation (39), satisfies Equation FORMULA12 with c = Φ 2 /2.

Finally, since DISPLAYFORM10 for d 0 ≥ 2, by Equation FORMULA1 , the matrices W 1 (t 0 ), . . .

, W N (t 0 ) are δ-balanced with δ = DISPLAYFORM11 .

Iteration t 0 thus satisfies the conditions of Theorem 1 with deficiency margin Φ 2 /2, meaning that for DISPLAYFORM12 we will have (T ) ≤ .

Therefore, by Equations FORMULA4 and FORMULA2 , to ensure that (T ) ≤ , we may take DISPLAYFORM13 Recall that this entire analysis holds only with the probability given in Equation (38).

As lim d→∞ (1 − 2 exp(−d/16)) = 1 and lim a→∞ (3 − 4F (2 2/a))/2 = 1/2, for any 0 < p < 1/2, there exist a, d 0 > 0 such that for d 0 ≥ d 0 , the probability given in Equation FORMULA3 is at least p.

This completes the proof.

In the context of the above proof, we remark that the expressions 1 − 2 exp(−d 0 /16) and (3 − 4F (2 2/a))/2 converge to their limits of 1 and 1/2, respectively, as d 0 , a → ∞ quite quickly.

For instance, to obtain a probability of greater than 0.25 of the initialization conditions being met, we may take d 0 ≥ 100, a ≥ 100.

We first consider the probability of δ-balancedness holding between any two layers: Lemma 10.

Suppose a, b, d ∈ N and A ∈ R a×d , B ∈ R d×b are matrices whose entries are distributed as i.i.d.

Gaussians with mean 0 and standard deviation s.

Then for k ≥ 1, DISPLAYFORM0 Proof.

Note that for 1 ≤ i, j ≤ d, let X ij be the random variable (A T A − BB T ) ij , so that DISPLAYFORM1 We next note that for a normal random variable Y of variance s 2 and mean 0, DISPLAYFORM2 Then FORMULA3 follows from Markov's inequality.

Now the proof of Claim 2 follows from a simple union bound:Proof of Claim 2.

By (43) of Lemma 10, for each 1 DISPLAYFORM3 By the union bound, DISPLAYFORM4 and the claim follows with δ = ks 2 10d 3 max .

We begin by introducing some notation.

FORMULA1 ).

There is an absolute constant C 0 such that the following holds.

Suppose that h is a multilinear polynomial of K variables X 1 , . . .

, X K and of degree N .

Suppose that X 1 , . . .

, X K are i.i.d.

Gaussian.

Then, for any > 0: DISPLAYFORM0 The below lemma characterizes the norm of the end-to-end matrix W 1:N following zero-centered Gaussian initialization: Lemma 12.

For any constant 0 < C 2 < 1, there is an absolute constant C 1 > 0 such that the following holds.

Let N, d 0 , . . .

, d N −1 ∈ N. Set d N = 1.

Suppose that for 1 ≤ j ≤ N , W j ∈ R dj ×dj−1 are matrices whose entries are i.i.d.

Gaussians of standard deviation s and mean 0.

Then DISPLAYFORM1 2 , so that f is a polynomial of degree 2N in the entries of W 1 , . . .

, W N .

Notice that DISPLAYFORM2 Since each g i0 is a multilinear polynomial in W 1 , . . .

, W N , we have that DISPLAYFORM3 For any constant B 1 (whose exact value will be specified below), it follows that DISPLAYFORM4 Next, by Lemma 11, there is an absolute constant C 0 > 0 such that for any > 0, and any DISPLAYFORM5 for each i 0 , it follows that DISPLAYFORM6 Next, given 0 < C 2 < 1, choose = (1 − C 2 )/(2C 0 N ), and B 1 = 2/(1 − C 2 ).

Then by FORMULA4 and FORMULA4 and a union bound, we have that DISPLAYFORM7 The result of the lemma then follows by taking C 1 = max DISPLAYFORM8 DISPLAYFORM9 i0 , which is a Gaussian with mean 0 and standard deviation s, since O i0 2 = 1.

Since O i0 , O i 0 = 0 for i 0 = i 0 , the covariance between any two distinct entries of W 1 O is 0.

Therefore, the entries of W 1 O are independent Gaussians with mean 0 and standard deviation s, just as are the entries of W 1 .

For a dimension d ∈ N, radius r > 0, and DISPLAYFORM0 Proof.

In BID10 , it is shown that the area of a (d, 1)-hyperspherical cap of height h is given by DISPLAYFORM1 , where Using that C ⊆ D, we continue with the proof.

Notice the fact that C ⊆ D is equivalent to DISPLAYFORM2 DISPLAYFORM3 , by the structure of C and D. Since the probability that V lands in ∂C is at DISPLAYFORM4 , this lower bound applies to V landing in ∂D as well.

Since all V ∈ ∂D have distance at most 1 − 1/(ad) from Φ, and since σ min (Φ) = Φ 2 = 1, it follows that for any V ∈ ∂D, V − Φ 2 ≤ σ min (Φ) − 1/(ad).

Therefore, with probability of at least DISPLAYFORM5 , V has deficiency margin Φ 2 /(ad) with respect to Φ.Lemma 16 (Lemma 8 restated).

Let d ∈ N, d ≥ 20; b 2 > b 1 ≥ 1 be real numbers (possibly depending on d); and Φ ∈ R d be a vector.

Suppose that µ is a rotation-invariant distribution over R d with a well-defined density, such that, for some 0 < < 1, DISPLAYFORM6 Then, with probability at least DISPLAYFORM7 , V will have deficiency margin Φ 2 /(b 2 d) with respect to Φ.Proof.

By rescaling we may assume that Φ 2 = 1 without loss of generality.

Then the deficiency margin of V is equal to 1 − V − Φ 2 .

µ has a well-defined density, so we can setμ to be the probability density function of V 2 .

Since µ is rotation-invariant, we can integrate over spherical coordinates, giving DISPLAYFORM8 where the first inequlaity used Lemma 15 and the fact that the distribution of V conditioned on V 2 = r is uniform on S d (r).

Proof of Claim 3.

We let W ∈ R 1×d0 R d0 denote the random vector W 1:N ; also let µ denote the distribution of W , so that by Lemma 13, µ is rotation-invariant.

Let C 1 be the constant from Lemma 12 for C 2 = 999/1000.

For some a ≥ 10 5 , the standard deviation of the entries of each W j is given by DISPLAYFORM0 Then by Lemma 12, ) with respect to Φ. But a ≥ 10 5 implies that this probability is at least 0.49, and from (48), DISPLAYFORM1 DISPLAYFORM2 Next recall the assumption in the hypothesis that s ≥ C 1 N (c · Φ 2 /(d 1 · · · d N −1 )) 1/2N .

Then the deficiency margin in (49) is at least DISPLAYFORM3 completing the proof.

Proof.

The target matrices Φ that will be used to prove the claim satisfy σ min (Φ) = 1.

We may assume without loss of generality that c ≥ 3/4, the reason being that if a matrix has deficiency margin c with respect to Φ and c < c, it certainly has deficiency margin c with respect to Φ.We first consider the case d = 1, so that the target and all matrices are simply real numbers; we will make a slight abuse of notation in identifying 1 × 1 matrices with their unique entries.

We set Φ = 1.

For all choices of η, we will set the initializations W 1 (0), . . .

, W N (0) so that W 1:N (0) = c. Then A ≤ W j (1) ≤ max{η, 1}A.We prove the following lemma by induction:Lemma 17.

For each t ≥ 1, the real numbers W 1 (t), . . .

, W N (t) all have the same sign and this sign alternates for each integer t. Moreover, there are real numbers 2 ≤ B(t) < C(t) for t ≥ 1 such that for 1 ≤ j ≤ N , B(t) ≤ |W j (t)| ≤ C(t) and ηB(t) 2N −1 ≥ 20C(t).Proof.

First we claim that we may take B(1) = min{η,1} 10A and C(1) = max{η, 1}A. We have shown above that B(1) ≤ W j (1) ≤ C(1) for all j. Next we establish that ηB ( Now set B(t + 1) = 9C(t) and C(t + 1) = ηC(t) 2N −1 .

Since N ≥ 2, we have that ηB(t + 1) 2N −1 = η(9C(t)) 2N −1 ≥ η9 3 C(t) 2N −1 > 20ηC(t) 2N −1 = 20C(t + 1).The case that all W j (t) are negative for 1 ≤ j ≤ N is nearly identical, with the same values for B(t + 1), C(t + 1) in terms of B(t), C(t), except all W j (t + 1) will be positive.

This establishes the inductive step and completes the proof of Lemma 17.

For the general case where d 0 = d 1 = · · · = d N = d for some d ≥ 1, we set Φ = I d , and given c, η, we set W j (0) to be the d × d diagonal matrix where all diagonal entries except the first one are equal to 1, and where the first diagonal entry is given by Equation FORMULA1 , where A is given by Equation (50).

It is easily verified that all entries of W j (t), 1 ≤ j ≤ N , except for the first diagonal element of each matrix, will remain constant for all t ≥ 0, and that the first diagonal elements evolve exactly as in the 1-dimensional case presented above.

Therefore the loss in the d-dimensional case is equal to the loss in the 1-dimensional case, which is always greater than some positive constant.

We remark that the proof of Claim 4 establishes that the loss (t) := L N (W 1 (t), . . .

, W N (t)) grows at least exponentially in t for the chosen initialization.

Such behavior, in which gradients and weights explode, indeed takes place in deep learning practice if initialization is not chosen with care.

Proof.

We will show that a target matrix Φ ∈ R d×d which is symmetric with at least one negative eigenvalue, along with identity initialization (W j (0) = I d , ∀j ∈ {1, . . .

, N }), satisfy the conditions of the claim.

First, note that non-stationarity of initialization is met, as for any 1 ≤ j ≤ N , Lemma 18 BID3 , Lemma 6).

If W 1 (0), . . .

, W N (0) are all initialized to identity, Φ is symmetric, Φ = U DU is a diagonalization of Φ, and gradient descent is performed with any learning rate, then for each t ≥ 0 there is a diagonal matrixD(t) such that W j (t) = UD(t)U for each 1 ≤ j ≤ N .By Lemma 18, for any choice of learning rate η, the end-to-end matrix at time t is given by W 1:N (t) = UD(t) N U .

As long as some diagonal element of D is negative, say equal to −λ < 0, then (t) = L N (W 1 (t), . . .

, W N (t)) = 1 2 DISPLAYFORM0

Below we provide implementation details omitted from our experimental report (Section 4).The platform used for running the experiments is PyTorch (Paszke et al., 2017) .

For compliance with our analysis, we applied PCA whitening to the numeric regression dataset from UCI Machine Learning Repository.

That is, all instances in the dataset were preprocessed by an affine operator that ensured zero mean and identity covariance matrix.

Subsequently, we rescaled labels such that the uncentered cross-covariance matrix Λ yx (see Section 2) has unit Frobenius norm (this has no effect on optimization other than calibrating learning rate and standard deviation of initialization to their conventional ranges).

With the training objective taking the form of Equation FORMULA1 , we then computed c -the global optimum -in accordance with the formula derived in Appendix A.In our experiments with linear neural networks, balanced initialization was implemented with the assignment written in step (iii) of Procedure 1.

In the non-linear network experiment, we added, for each j ∈ {1, . . .

, N − 1}, a random orthogonal matrix to the right of W j , and its transpose to the left of W j+1 -this assignment maintains the properties required from balanced initialization (see Footnote 7).

During all experiments, whenever we applied grid search over learning rate, values between 10 −4 and 1 (in regular logarithmic intervals) were tried.

<|TLDR|>

@highlight

We analyze gradient descent for deep linear neural networks, providing a guarantee of convergence to global optimum at a linear rate.
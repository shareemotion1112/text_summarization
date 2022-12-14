Nesterov SGD is widely used for training modern neural networks and other machine learning models.

Yet, its advantages over SGD have not been theoretically clarified.

Indeed, as we show  in this paper, both theoretically and empirically, Nesterov SGD with any parameter selection does not in general provide acceleration over ordinary SGD.

Furthermore, Nesterov SGD may diverge for step sizes that ensure convergence of ordinary SGD.

This is in contrast to the classical results in the deterministic setting, where the same step size ensures accelerated convergence of the Nesterov's method over optimal gradient descent.



To address the non-acceleration issue, we  introduce a compensation term to Nesterov SGD.

The resulting  algorithm, which we call MaSS, converges  for same step sizes as SGD.

We prove that MaSS obtains an accelerated convergence rates over SGD for any mini-batch size in the linear setting.

For full batch, the convergence rate of MaSS matches the well-known accelerated rate of the Nesterov's method.



We also analyze the  practically important question of the dependence of the convergence rate and  optimal hyper-parameters on the mini-batch size, demonstrating three distinct regimes: linear scaling, diminishing returns and saturation.



Experimental evaluation of MaSS for several standard  architectures of deep networks, including ResNet and convolutional networks, shows improved performance over SGD, Nesterov SGD  and Adam.

Many modern neural networks and other machine learning models are over-parametrized (5) .

These models are typically trained to have near zero training loss, known as interpolation and often have strong generalization performance, as indicated by a range of empirical evidence including (23; 3).

Due to a key property of interpolation -automatic variance reduction (discussed in Section 2.1), stochastic gradient descent (SGD) with constant step size is shown to converge to the optimum of a convex loss function for a wide range of step sizes (12) .

Moreover, the optimal choice of step size ?? * for SGD in that setting can be derived analytically.

The goal of this paper is to take a step toward understanding momentum-based SGD in the interpolating setting.

Among them, stochastic version of Nesterov's acceleration method (SGD+Nesterov) is arguably the most widely used to train modern machine learning models in practice.

The popularity of SGD+Nesterov is tied to the well-known acceleration of the deterministic Nesterov's method over gradient descent (15) .

Yet, has not not theoretically clear whether Nesterov SGD accelerates over SGD.

As we show in this work, both theoretically and empirically, Nesterov SGD with any parameter selection does not in general provide acceleration over ordinary SGD.

Furthermore, Nesterov SGD may diverge, even in the linear setting, for step sizes that guarantee convergence of ordinary SGD.

Intuitively, the lack of acceleration stems from the fact that, to ensure convergence, the step size of SGD+Nesterov has to be much smaller than the optimal step size for SGD.

This is in contrast to the deterministic Nesterov method, which accelerates using the same step size as optimal gradient descent.

As we prove rigorously in this paper, the slow-down of convergence caused by the small step size negates the benefit brought by the momentum term.

We note that a similar lack of acceleration for the stochastic Heavy Ball method was analyzed in (9) .

To address the non-acceleration of SGD+Nesterov, we introduce an additional compensation term to allow convergence for the same range of step sizes as SGD.

The resulting algorithm, MaSS (Momentum-added Stochastic Solver) 1 updates the weights w and u using the following rules (with the compensation term underlined): Figure 1 : Non-acceleration of Nesterov SGD and fast convergence of MaSS.

w t+1 ??? u t ??? ?? 1??? f (u t ), u t+1 ??? (1 + ??)w t+1 ??? ??w t + ?? 2??? f (u t ).

(

Here,??? represents the stochastic gradient.

The step size ?? 1 , the momentum parameter ?? ??? (0, 1) and the compensation parameter ?? 2 are independent of t.

We proceed to analyze theoretical convergence properties of MaSS in the interpolated regime.

Specifically, we show that in the linear setting MaSS converges exponentially for the same range of step sizes as plain SGD, and the optimal choice of step size for MaSS is exactly ?? * which is optimal for SGD.

Our key theoretical result shows that MaSS has accelerated convergence rate over SGD.

Furthermore, in the full batch (deterministic) scenario, our analysis selects ?? 2 = 0, thus reducing MaSS to the classical Nesterov's method (15) .

In this case our convergence rate also matches the well-known convergence rate for the Nesterov's method (15; 4) .

This acceleration is illustrated in Figure 1 .

Note that SGD+Nesterov (as well as Stochastic Heavy Ball) does not converge faster than SGD, in line with our theoretical analysis.

We also prove exponential convergence of MaSS in more general convex setting under additional conditions.

We further analyze the dependence of the convergence rate e ???s(m)t and optimal hyper-parameters on the mini-batch size m. We identify three distinct regimes of dependence defined by two critical values m * 1 and m * 2 : linear scaling, diminishing returns and saturation, as illustrated in Figure 2 .

The convergence speed per iteration s(m), as well as the optimal hyper-parameters, increase linearly as m in the linear scaling regime, sub-linearly in the diminishing returns regime, and can only increase by a small constant factor in the saturation regime.

The critical values m * 1 and m * 2 are derived analytically.

We note that the intermediate "diminishing terurns" regime is new and is not found in SGD (12) .

To the best of our knowledge, this is the first analysis of mini-batch dependence for accelerated stochastic gradient methods.

We also experimentally evaluate MaSS on deep neural networks, which are non-convex.

We show that MaSS outperforms SGD, SGD+Nesterov and Adam (10) both in optimization and generalization, on different architectures of deep neural networks including convolutional networks and ResNet (7) .

The paper is organized as follows: In section 2, we introduce notations and preliminary results.

In section 3, we discuss the non-acceleration of SGD+Nesterov.

In section 4 we introduce MaSS and analyze its convergence and optimal hyper-parameter selection.

In section 5, we analyze the mini-batch MaSS.

In Section 6, we show experimental results.

Over-parameterized models have drawn increasing attention in the literature as many modern machine learning models, especially neural networks, are over-parameterized (5) and show strong generalization performance (16; 23; 2).

Over-parameterized models usually result in nearly perfect fit (or interpolation) of the training data (23; 18; 3).

Exponential convergence of SGD with constant step size under interpolation and its dependence on the batch size is analyzed in (12) .

There are a few works that show or indicate the non-acceleration of existing stochastic momentum methods.

First of all, the work (9) theoretically proves non-acceleration of stochastic Heavy Ball method (SGD+HB) over SGD on certain synthetic data.

Furthermore, these authors provide experimental evidence that SGD+Nesterov also converges at the same rate as SGD on the same data.

The work (22) theoretically shows that, for sufficiently small step-sizes, SGD+Nesterov and SGD+HB is equivalent to SGD with a larger step size.

However, the results in (22) do not exclude the possibility that acceleration is possible when the step size is larger.

The work (11) concludes that "momentum hurts the convergence within the neighborhood of global optima", based on a theoretical analysis of SGD+HB.

These results are consistent with our analysis of the standard SGD+Nesterov.

However, this conclusion does not apply to all momentum methods.

Indeed, we will show that MaSS provably improves convergence over SGD.

There is a large body of work, both practical and theoretical, on SGD with momentum, including (10; 8; 1).

Adam (10) , and its variant AMSGrad (17) , are among the most practically used SGD methods with momentum.

Unlike our method, Adam adaptively adjusts the step size according to a weight-decayed accumulation of gradient history.

In (8) the authors proposed an accelerated SGD algorithm, which can be written in the form shown on the right hand side in Eq.8, but with different hyper-parameter selection.

Their ASGD algorithm also has a tail-averaging step at the final stage.

In the interpolated setting (no additive noise) their analysis yields a convergence rate of O(P oly(??,??) exp(???

)) for our algorithm with batch size 1.

We provide some experimental comparisons between their ASGD algorithm and MaSS in Fig. 4 .

The work (21) proposes and analyzes another first-order momentum algorithm and derives convergence rates under a different set of conditions -the strong growth condition for the loss function in addition to convexity.

As shown in Appendix F.3, on the example of a Gaussian distributed data, the rates obtained in (21) can be slower than those for SGD.

In contrast, our algorithm is guaranteed to never have a slower convergence rate than SGD.

Furthermore, in the same Gaussian setting MaSS matches the optimal accelerated full-gradient Nesterov rate.

Additionally, in our work we consider the practically important dependence of the convergence rate and optimal parameter selection on the mini-batch size, which to the best of our knowledge, has not been analyzed for momentum methods.

, where f i only depends on a single data point (x i , y i ).

Let ???f denote the exact gradient, and??? m f denote the unbiased stochastic gradient evaluated based on a mini-batch of size m. For simplicity, we also denote???f (w) :=??? 1 f (w).

We use the concepts of strong convexity and smoothness of functions, see definitions in Appendix B.1.

For loss function with ??-strong convexity and L-smoothness, the condition number ?? is defined as ?? = L/??.

In the case of the square loss,

, and the Hessian matrix is H :

.

Let L and ?? be the largest and the smallest non-zero eigenvalues of the Hessian respectively.

Then the condition number is then ?? = L/?? (note that zero eigenvalues can be ignored in our setting, see Section 4).

Given a mini-batch size m, we define the m-stochastic condition number as ?? m := L m /??.

Following (8), we introduce the quantity?? (called statistical condition number in (8)), which is the smallest positive real number such that E x 2

Hence, the quadratic loss function is also L m -smooth, for all m ??? 1.

By the definition of ?? m , we also have

Remark 2.

It is important to note that?? ??? ?? 1 , since E x 2

We consider over-parametrized models that have zero training loss solutions on the training data (e.g., (23)).

A solution f i (w) which fits the training data perfectly f i (w) = 0, ???i = 1, 2, ?? ?? ?? , n, is known as interpolating.

In the linear setting, interpolation implies that the linear system {x

has at least one solution.

A key property of interpolation is Automatic Variance Reduction (AVR), where the variance of the stochastic gradient decreases to zero as the weight w approaches the optimal w * .

For a detailed discussion of AVR see Appendix B.2.

Thanks to AVR, plain SGD with constant step size can be shown to converge exponentially for strongly convex loss functions (13; 19; 14; 12) .

The set of acceptable step sizes is (0, 2/L m ), where L m is defined in Eq.2 and m is the mini-batch size.

Moreover, the optimal step size ?? * (m) of SGD that induces fastest convergence guarantee is proven to be 1/L m (12).

In this section we prove that SGD+Nesterov, with any constant hyper-parameter setting, does not generally improve convergence over optimal SGD.

Specifically, we demonstrate a setting where SGD+Nesterov can be proved to have convergence rate of (1 ??? O(1/??)) t , which is same (up to a constant factor) as SGD.

In contrast, the classical accelerated rate for the deterministic Nesterov's method is

We will consider the following two-dimensional data-generating component decoupled model.

Fix an arbitrary w * ??? R 2 and randomly sample z from N (0, 2).

The data points (x, y) are constructed as follow:

where e 1 , e 2 ??? R 2 are canonical basis vectors, The following theorem gives a lower bound for the convergence of SGD+Nesterov, regarding the linear regression problem on the component decoupled data model.

See Appendix C for the proof.

be a dataset generated according to the component decoupled model.

Consider the optimization problem of minimizing quadratic function

2 .

For any step size ?? > 0 and momentum parameter ?? ??? (0, 1) of SGD+Nesterov with random initialization, with probability one, there exists a T ??? N such that ???t > T ,

where C > 0 is a constant.

Compared with the convergence rate (1???1/??) t of SGD (12) , this theorem shows that SGD+Nesterov does not accelerate over SGD.

This result is very different from that in the deterministic gradient scenario, where the classical Nesterov's method has a strictly faster convergence guarantee than gradient descent (15) .

Intuitively, the key reason for the non-acceleration of SGD+Nesterov is a condition on the step size ?? required for non-divergence of the algorithm.

Specifically, when momentum parameter ?? is close to 1, ?? is required to be less than

2 ) (precise formulation is given in Lemma 1 in Appendix C).

The slow-down resulting from the small step size necessary to satisfy that condition cannot be compensated by the benefit of the momentum term.

In particular, the condition on the step-size of SGD+Nesterov excludes ?? * that achieves fastest convergence for SGD.

We show in the following corollary that, with the step size ?? * , SGD+Nesterov diverges.

This is different from the deterministic scenario, where the Nesterov method accelerates using the same step size as gradient descent.

Corollary 1.

Consider the same optimization problem as in Theorem 1.

Let step-size ?? = and acceleration parameter ?? ??? [0.6, 1], then SGD+Nesterov, with random initialization, diverges with probability 1.

We empirically verify the non-acceleration of SGD+Nesterov as well as Corollary 1, in Section 6 and Appendix F.2.

In this section, we propose MaSS, which introduces a compensation term (see Eq.1) onto SGD+Nesterov.

We show that MaSS can converge exponentially for all the step sizes that result in convergence of SGD, i.e., ?? ??? (0, 2/L m ).

Importantly, we derive a convergence rate exp(???t/ ??? ?? 1?? ), where?? ??? ?? 1 , for MaSS which is faster than the convergence rate for SGD exp(???t/?? 1 ).

Moreover, we give an analytical expression for the optimal hyper-parameter setting.

For ease of analysis, we rewrite update rules of MaSS in Eq.1 in the following equivalent form (introducing an additional variable v):

There is a bijection between the hyper-parameters (?? 1 , ?? 2 , ??) and (??, ??, ??), which is given by:

Remark 3 (SGD+Nesterov).

In the literature, the Nesterov's method is sometimes written in a similar form as the R.H.S. of Eq.8.

Since SGD+Nesterov has no compensation term, ?? has to be fixed as ??/??, which is consistent with the parameter setting in (15).

Assumptions.

We first assume square loss function, and later extend the analysis to general convex loss functions under additional conditions.

For square loss function, the solution set W * := {w ??? R d |f (w) = 0} is an affine subspace in the parameter space R d .

Given any w, we denote its closest solution as w * := arg min v???W * w ??? v , and define the error = w ??? w * .

Be aware that different w may correspond to different w * , and that and (stochastic) gradients are always perpendicular to W * (see discussion in Appendix B.3).

Hence, no actual update happens along W * .

For this reason, we can ignore zero eigenvalues of H and restrict our analysis to the span of the eigenvectors of the Hessian with non-zero eigenvalues.

Based on the equivalent form of MaSS in Eq.8, the following theorem shows that, for square loss function in the interpolation setting, MaSS is guaranteed to have exponential convergence when hyper-parameters satisfy certain conditions.

Theorem 2 (Convergence of MaSS).

Consider minimizing a quadratic loss function in the interpolation setting.

Let ?? be the smallest non-zero eigenvalue of the Hessian matrix H. Let L m be as defined in Eq.2.

Denote?? m :=??/m + (m ??? 1)/m.

In MaSS with mini batch of size m, if the positive hyper-parameters ??, ??, ?? satisfy the following two conditions:

then, after t iterations,

Consequently,

for some constant C > 0 which depends on the initialization.

Remark 4.

By condition Eq.10, the admissible step size ?? is (0, 2/L m ), exactly the same as SGD for interpolated setting (12) .

Remark 5.

One can easily check that the hyper-parameter setting of SGD+Nesterov does not satisfy the conditions in Eq.10.

Proof sketch for Theorem 2.

Denote F t := E v t+1 ??? w * 2

we show that, under the update rules of MaSS in Eq.8,

By the condition in Eq.10, c 1 ??? 0, c 2 ??? 0, then the last two terms are non-positive.

Hence,

Using that w t ???w * 2 ??? ??F t /??, we get the final conclusion.

See detailed proof in Appendix D.

Hyper-parameter Selection.

From Theorem 2, we observe that the convergence rate is determined by (1?????) t .

Therefore, larger ?? is preferred for faster convergence.

Combining the conditions in Eq.10, we have

By setting ?? * = 1/L m , which maximizes the right hand side of the inequality, we obtain the optimal selection ?? * = 1/ ??? ?? m??m .

Note that this setting of ?? * and ?? * determines a unique ?? * = ?? * /?? by the conditions in Eq.10.

In summary,

By Eq.9, the optimal selection of (?? 1 , ?? 2 , ??) would be:

?? m is usually larger than 1, which implies that the coefficient ?? * 2 of the compensation term is non-negative.

The non-negative coefficient ?? 2 indicates that the weight u t is "over-descended" in SGD+Nesterov and needs to be compensated along the gradient direction.

It is important to note that the optimal step size for MaSS as in Eq.13 is exactly the same as the optimal one for SGD (12) .

With such hyper-parameter selection given in Eq.14, we have the following theorem for optimal convergence: Theorem 3 (Acceleration of MaSS).

Under the same assumptions as in Theorem 2, if we set hyperparameters in MaSS as in Eq.13, then after t iteration of MaSS with mini batch of size m,

for some constant C > 0 which depends on the initialization.

Remark 6.

With the optimal hyper-parameters in Eq.13, the asymptotic convergence rate of MaSS is

which is faster than the rate O(e ???t/??m ) of SGD (see (12) ), since ?? m ????? m .

Remark 7 (MaSS Reduces to the Nesterov's method for full batch).

In the limit of full batch m ??? ???, we have ?? m ??? ??,?? m ??? 1, the optimal parameter selection in Eq.14 reduces to

It is interesting to observe that, in the full batch (deterministic) scenario, the compensation term vanishes and ?? * 1 and ?? * are the same as those in Nesterov's method.

Hence MaSS with the optimal hyper-parameter selection reduces to Nesterov's method in the limit of full batch.

Moreover, the convergence rate in Theorem 3 reduces to O(e ???t/ ??? ?? ), which is exactly the well-known convergence rate of Nesterov's method (15; 4).

Extension to Convex Case.

First, we extend the definition of L 1 to convex functions,

, for some > 0.

In MaSS, if the hyper-parameters are set to be:

then after t iterations, there exists a constant

t .

Based on our analysis, we discuss the effect of selection of mini-batch size m. We show that the domain of mini-batch size m can be partitioned into three intervals by two critical points:

The three intervals/regimes are depicted in Figure 2 , and the detailed analysis is in Appendix G.

The optimal selection of hyper-parameters is approximated by:

and the convergence rate in Eq. In the linear scaling regime, the hyper-parameter selections follow a Linear Scaling Rule (LSR): When the mini-batch size is multiplied by k, multiply all hyperparameters (??, ??, ??) by k. This parallels the linear scaling rule for SGD which is an accepted practice for training neural networks (6) .

This three regimes partition is different from that for SGD (12) , where only linear scaling and saturation regimes present.

An empirical verification of the dependence of the convergence speed on m is shown in Figure 3 .

See the setup in Appendix G.

Synthetic Data.

We empirically verify the non-acceleration of SGD+Nesterov and the fast convergence of MaSS on synthetic data.

Specifically, we optimize the quadratic function

is generated by the component decoupled model described in Section 3.

We compare the convergence behavior of SGD+Nesterov with SGD, as well as our proposed method, MaSS, and several other methods: SGD+HB, ASGD (8) .

We select the best hyper-parameters from dense grid search for SGD+Nesterov (step-size and momentum parameter), SGD+HB (step-size and momentum parameter) and SGD (step-size).

For MaSS, we do not tune the hyper-parameters but use the hyper-parameter setting suggested by our theoretical analysis in Section 4; For ASGD, we use the setting provided by (8) .

Hyperparameters: we use optimal parameters for SGD+Nesterov, SGD+HB and SGD; the setting in (8) for ASGD; Eq. 13 for MaSS.

9 .

We observe that the fastest convergence of SGD+Nesterov is almost identical to that of SGD, indicating the non-acceleration of SGD+Nesterov.

We also observe that our proposed method, MaSS, clearly outperforms the others.

In Appendix F.2, we provide additional experiments on more settings of the component decoupled data, and Gaussian distributed data.

We also show the divergence of SGD+Nesterov with the same step size as SGD and MaSS in Appendix F.2.

Real data: MNIST and CIFAR-10.

We compare the optimization performance of SGD, SGD+Nesterov and MaSS on the following tasks: classification of MNIST with a fullyconnected network (FCN), classification of CIFAR-10 with a convolutional neural network (CNN) and Gaussian kernel regression on MNIST.

See detailed description of the architectures in Appendix H.1.

In all the tasks and for all the algorithms, we select the best hyper-parameter setting over dense grid search, except that we fix the momentum parameter ?? = 0.9 for both SGD+Nesterov and MaSS, which is typically used in practice.

All algorithms are implemented with mini batches of size 64 for neural network training.

Test Performance.

We show that the solutions found by MaSS have good generalization performance.

We evaluate the classification accuracy of MaSS, and compare with SGD, SGD+Nesterov and Adam, on different modern neural networks: CNN and ResNet (7) .

See description of the architectures in Appendix H.1.

In the training processes, we follow the standard protocol of data augmentation and reduction of learning rate, which are typically used to achieve state-of-the-art results in neural networks.

In each task, we use the same initial learning rate for MaSS, SGD and SGD+Nesterov, and run the same number of epochs (150 epochs for CNN and 300 epochs for ResNet-32).

Detailed experimental settings are deferred to Appendix H.2.

Average of 3 runs that converge.

Some runs diverge.

??? Adam uses initial step size 0.001.

Table 6 compares the classification accuracy of these algorithms on the test set of CIFAR-10 (average of 3 independent runs).

We observe that MaSS produces the best test performance.

We also note that increasing initial learning rate may improves performance of MaSS and SGD, but degrades that of SGD+Nesterov.

Moreover, in our experiment, SGD+Nesterov with large step size ?? = 0.3 diverges in 5 out of 8 runs on CNN and 2 out of 5 runs on ResNet-32 (for random initialization), while MaSS and SGD converge on every run.

Step-size ?? 1 , secondary step-size ?? 2 , acceleration parameter ?? ??? (0, 1).

.

end while Output: weight w t .

Note that the proposed algorithm initializes the variables w 0 and u 0 with the same vector, which could be randomly generated.

As discussed in section 4, MaSS can be equivalently implemented using the following update rules:

In this case, variables u 0 , v 0 and w 0 should be initialized with the same vector.

There is a bijection between the hyper-parameters (?? 1 , ?? 2 , ??) and (??, ??, ??), which is given by:

B ADDITIONAL PRELIMINARIES

).

A differentiable function f : R d ??? R is ??-strongly convex (?? > 0), if f (x) ??? f (z) + ???f (z), x ??? z + ?? 2 x ??? z 2 , ???x, z ??? R d . (21) Definition 2 (Smoothness).

A differentiable function f : R d ??? R is L-smooth (L > 0), if f (x) ??? f (z) + ???f (z), x ??? z + L 2 x ??? z 2 , ???x, z ??? R d .(22)

In the interpolation setting, one can write the square loss as

A key property of interpolation is that the variance of the stochastic gradient of decreases to zero as the weight w approaches an optimal solution w * .

Proposition 1 (Automatic Variance Reduction).

For the square loss function f in the interpolation setting, the stochastic gradient at an arbitrary point w can be written as

Moreover, the variance of the stochastic gradient

Since E[(H m ??? H) 2 ] is independent of w, the above proposition unveils a linear dependence of variance of stochastic gradient on the norm square of error .

This observation underlies exponential convergence of SGD in certain convex settings (20; 13; 19; 14; 12).

Consider the square loss function, f (w) = 1 2n

where the stochastic gradient is computed based on a randomly sampled batch of size m.

Recall that the solution set W * := {w ??? R d |f (w) = 0} is an affine subspace in the parameter space, and that w * is the solution in W * that is closest to w. Hence, w ??? w * is perpendicular to W * , i.e., w ??? w

Hence the (stochastic) gradient is perpendicular to W * .

The key proof technique is to consider the asymptotic behavior of SGD+Nesterov in the decoupled model of data when the condition number becomes large.

Notations and proof setup.

Recall that the square loss function based on the component decoupled data D, define in Eq.6, is in the interpolation regime, then for SGD+Nesterov, we have the recurrence relation

It is important to note that each component of w t evolves independently, due to the fact thatH is diagonal.

With := w ??? w * , we define for each component j = 1, 2 that

where [j] is the j-th component of vector .

The recurrence relation in Eq.26 can be rewritten as

with

For the ease of analysis, we define u := 1 ??? ?? ??? (0, 1] and t j := ???? 2 j , j = 1, 2.

Without loss of generality, we assume ?? 2 1 = 1 in this section.

In this case, t 1 = ?? and t 2 = ??/??, where ?? is the condition number.

[j] :

, where

Proof idea.

For the two-dimensional component decoupled data, we have

By definition of ?? in Eq.27, we can see that the convergence rate is lower bounded by the convergence rates of the sequences { ??

[j] t } t .

By the relation Eq.28, we have that the convergence rate of the sequence { ?? t } t is controlled by the magnitude of the top eigenvalue ?? max of B, if ?? t has nonzero component along the eigenvector of B with eigenvalue ?? max (B).

Specifically, if |?? max | > 1, ??

[j] t grows at a rate of |?? max | t , indicating the divergence of SGD+Nesterov; if |?? max | < 1, then

converges at a rate of |?? max | t .

In the following, We use the eigen-systems of matrices B [j] , especially the top eigenvalue, to analyze the convergence behavior of SGD+Nesterov with any hyper-parameter setting.

We show that, for any choice of hyper-parameters (i.e., step-size and momentum parameter), at least one of the following statements must holds:

??? B [1] has an eigenvalue larger than 1.

??? B [2] has an eigenvalue of magnitude

This is formalized in the following two lemmas.

Lemma 1.

For any u ??? (0, 1], if step size

then, B [1] has an eigenvalue larger than 1.

We will analyze the dependence of the eigenvalues on ??, when ?? is large to obtain Lemma 2.

For any u ??? (0, 1], if step size

then, B [2] has an eigenvalue of magnitude 1 ??? O (1/??).

Finally, we show that ?? t has non-zero component along the eigenvector of B with eigenvalue ?? max , hence the convergence of SGD+Nesterov is controlled by the eigenvalue of B [j] with the largest magnitude.

Lemma 3.

Assume SGD+Nesterov is initialized with w 0 such that both components w 0 ??? w * , e 1 and w 0 ??? w * , e 2 are non-zero.

Then, for all t > 2, ??

[j]

t has a non-zero component in the eigen direction of B

[j] that corresponds to the eigenvalue with largest magnitude.

Remark 8.

When w is randomly initialized, the conditions w 0 ???w * , e 1 = 0 and w 0 ???w * , e 2 = 0 are satisfied with probability 1, since complementary cases form a lower dimensional manifold which has measure 0.

By combining Lemma 1, 2 and 3, we have that SGD+Nesterov either diverges or converges at a rate of (1 ??? O(1/??)) t , and hence, we conclude the non-acceleration of SGD+Nesterov.

In addition, Corollary 1 is a special case of Theorem 1 and is proven by combining Lemma 1 and 3.

In high level, the proof ideas of Lemma 1 and 3 is analogous to those of (9), which proves the non-acceleration of stochastic Heavy Ball method over SGD.

But the proof idea of Lemma 2 is unique.

Proof of Lemma 1.

The characteristic polynomial of B j , j = 1, 2, are:

First note that lim ???????? D j (??) = +??? > 0.

In order to show B 1 has an eigenvalue larger than 1, it suffices to verity that

Replacing ?? by 1 ??? u and ???? 2 1 by t 1 , we have

Solving for the inequality D 1 (1) < 0, we have

, for positive step size ??.

Proof of Lemma 2.

We will show that at least one of the eigenvalues of B [2] is 1 ??? O(1/??), under the condition in Eq.32.

First, we note that t 2 = t 1 /?? = ??/??, which is O(1/??).

We consider the following cases separately:

2 ) and o(1); and 5) u is ??(1), the last of which includes the case where momentum parameter is a constant.

Note that, for cases 1-4, u is o(1).

In such cases, the step size condition Eq.32 can be written as

It is interesting to note that ?? must be o(1) to not diverge, when u is o(1).

This is very different to SGD where a constant step size can result in convergence, see (12) .

2 ).

In this case, the terms u 6 , u 4 t 2 , u 2 t 2 2 and t 3 2 are of the same order.

We find that

2 ).

Hence,

Write t 2 = cu 2 asymptotically for some constant c. If 4t 2 ??? u 2 , i.e., 4c ??? 1 ??? 0, then

If 4t 2 ??? u 2 , i.e., 4c ??? 1 ??? 0, then

In either case, the first-order term is of order u.

Recall that t 2 = ??/??, then we have

Case 2: u is o(t 0.5

2 ) and ??(t 2 ).

In this case,

).

Hence,

2 ).

This implies that t 0.5 2 = o(1/??) and u = o(1/??).

Therefore, all the eigenvalues ?? i are of order 1 ??? O(1/??).

Case 3: u is O(t 2 ).

This case if forbidden by the assumption of this lemma.

This is because, ?? is o(1) ).

This is contradictory to u is O(t 2 ).

Case 4: u is ???(t 0.5 2 ) and o (1) .

In this case, we first consider the terms independent of t 2 , i.e., constant term and u-only terms.

These terms can be obtained by putting t 2 = 0.

In such a setting, the eigenvalues are simplified as:

Note that the u-only terms cancel in ?? 2 , so the first order after the constant term must be of t 2 (could be t 2 /u 2 , t 2 /u etc.).

In the following we are going to analyze the t 2 terms.

Since u is ???(t 0.5

2 ), u 2 has lower order than t 2 , and t 2 /u 2 is o(1).

This allows us to do Taylor expansion:

where f (u) and g(u) are u-terms only, which, by the above analysis in Eq.37, are shown to contribute nothing to ?? 2 .

Hence, we use the first terms of T 1 and T 2 above to analyze the first order term of ?? 2 .

Plugging in these term to the expression of ?? 2 , and keeping the lowest order of t 2 , we find a zero coefficient of the lowest order t 2 -term:

Hence, ?? 2 can be written as:

where c is the coefficient.

On the other hand, by Eq.35 and t 2 = ??/??, we have

Therefore, we can write Eq.38 as ?? 2 = 1 ??? O(1/??).

.

This is the case where the momentum parameter is ??-independent.

Using the same argument as in case 4, we have zero u-only terms.

Then, directly taking Taylor expansion with respect to t 2 results in:

Proof of Lemma 3.

This proof follows the idea of the proof for stochastic Heavy Ball method in (9) .

The idea is to examine the subspace spanned by ?? t , t = 0, 1, 2, ?? ?? ?? , j = 1, 2, and to prove that the eigenvector of B

[j] corresponding to top eigenvalue (i.e., eigenvalue with largest magnitude) is not orthogonal to this spanned subspace.

This in turn implies that there exists a non-zero component of ?? [j] t in the eigen direction of B [j] corresponding to top eigenvalue, and this decays/grows at a rate of ?? t max (B [j] ).

[j]

t has non-zero component in the eigen direction of B [j] with top eigenvalue, then ??

[j]

t should also have non-zero component in the same direction.

Thus, it suffices to show that at least one of ??

3 has non-zero component in the eigen direction with top eigenvalue.

SinceH is diagonal for this two-dimensional decoupled data, w [1] and w [2] evolves independently, and we can analyze each component separately.

In addition, it can be seen that each of the initial values w

, which is non-zero by the assumption of this lemma, just acts as a scale factor during the training.

Hence, without loss of generality, we can assume w

0 ??? (w * )

[j] = 1, for each j. Then, according to the recurrence relation of SGD+Nesterov in Eq.26,

where s

[j]

Denote the vectorized form of ??

t ), which is a 4 ?? 1 column vector.

We stack the vectorized forms of ??

3 to make a 4 ?? 4 matrix, denoted as M [j] :

Note that ??

[j]

t , t = 0, 1, 2, 3, are symmetric tensors, which implies that M [j] contains two identical rows.

Specifically, the second and third row of M [j] are identical.

Therefore, the vector v

T is an eigenvector of M [j] with eigenvalue 0.

In fact, v is also an eigenvector of B

[j] with eigenvalue ??(1 ??? ????

Hence, v is not the eigenvector along top eigenvalue, and therefore, is orthogonal to the eigen space with top eigenvalue.

In order to prove at least one of ?? t , t = 0, 1, 2, 3, has a non-zero component along the eigen direction of top eigenvalue, it suffices to verify that M

[j] is rank 3, i.e., spans a three-dimensional space.

Equivalently, we consider the following matrix

where we omitted the superscript [j] for simplicity of the expression.

If the determinant of M [j] is not zero, then it is full rank, and hence M

[j] spans a three-dimensional space.

Plug in the expressions in Eq.41, then we have

where t j = ????

We note that, for all u ??? [0, 1) both t j are not positive.

This means that, for all u and positive t j , the determinant det(M [j] ) can never be zero.

Therefore, for each j = 1, 2, M [j] is full rank, and M [j] spans a three-dimensional space, which includes the eigenvector with the top eigenvalue of B [j] .

Hence, at least one of ?? t , t ??? {0, 1, 2, 3} has non-zero component in the eigen direction with top eigenvalue.

By ??

t also have non-zero component in the eigen direction with top eigenvalue of B [j] .

Proof of Theorem 1.

Lemma 1 and 2 show that, for any hyper-parameter setting (??, ??) with ?? > 0 and ?? ??? (0, 1), either top eigenvalue of B [1] is larger than 1 or top eigenvalue of B [2] is 1 ??? O(1/??).

Hence, |?? max | is either greater than 1 or is 1 ??? O(1/??).

Lemma 3 shows that ?? t has non-zero component along the eigenvector of B with eigenvalue ?? max (B).

By Eq.28 and Lemma 1 and 2, the sequence { ?? t } t either diverges or converges at a rate of

t .

By definition of ?? in Eq.27, we have that { t } t either diverges or converges at a rate of (1 ??? O(1/??)) t .

Note that, for the two-dimensional component decoupled data, we have

Therefore, the convergence rate of SGD+Nesterov is lower bounded by (1???O(1/??)) t .

Note that the convergence rate of SGD on this data is (1 ??? O(1/??)) t , hence SGD+Nesterov does not accelerate SGD on the two-dimensional component decoupled dataset.

Proof of Corollary 1.

When ?? = 1/L 1 and ?? ??? [0.6, 1], the condition in Eq.31 is satisfied.

By Lemma 1, the top eigenvalue ?? [1] max of B [1] is larger than 1.

By Lemma 3, ?? [1] has non-zero component along the eigenvector with this top eigenvalue ?? [1] max .

Hence, | w t ??? w * , e 1 | grows at a rate of

We first give a lemma that is useful for dealing with the mini-batch scenario:

Lemma 4.

If square loss f is in the interpolated setting, i.e., there exists w * such that f (w

Proof.

onto Eq.53 and add it to Eq.52, then

If the hyper-parameters are selected as:

then the last two terms in Eq.54 are non-positive.

Hence,

which implies

Since f (w t ) ??? L/2 ?? w t ??? w * 2 (by smoothness), then we have the final conclusion

with C being a constant.

Fix an arbitrary w * ??? R 2 and let z be randomly drawn from the zero-mean Gaussian distribution with variance E[z 2 ] = 2, i.e. z ??? N (0, 2).

The data points (x, y) ??? D are constructed as follow:

where e 1 , e 2 ??? R 2 are canonical basis vectors, ?? 1 > ?? 2 > 0.

Note that the corresponding square loss function on D is in the interpolation regime, since f (w * ) = 0.

The Hessian and stochastic Hessian matrices turn out to be

Note thatH is diagonal, which implies that stochastic gradient based algorithms applied on this data evolve independently in each coordinate.

This allows a simplified directional analysis of the algorithms applied.

Here we list some useful results for our analysis.

The fourth-moment of Gaussian variable Gaussian Data.

Suppose the data feature vectors {x i } are zero-mean Gaussian distributed, and y i = w * , x i , ???i, where w * is fixed but unknown.

Then, by the fact that

for zero-mean Gaussian random variables z 1 , z 2 , z 3 and z 4 , we have

In this subsection, we show additional empirical verification for the fast convergence of MaSS, as well as the non-acceleration of SGD+Nesterov, on synthetic data.

In addition, we show the divergence of SGD+Nesterov when using the same step size as SGD and MaSS, as indicated by Corollary 1.

We consider two families of synthetic datasets:

??? Component decoupled: (as defined in Section 3).

Fix an arbitrary w * ??? R 2 with all components non-zero.

x i is drawn from N (0, diag(2??

2 )) with probability 0.5 each.

y i = w * , x i for all i.

??? 3-d Gaussian: Fix an arbitrary w * ??? R 3 with all components non-zero.

x i are independently drawn from N (0, diag(?? Step size: ?? * = 1/L 1 = 1/6, and momentum parameter: ?? is 0.9 or 0.99. performed on either 3-d Gaussian or component decoupled data with fixed ?? 1 and ?? 2 .

For each setting of (?? 1 , ?? 2 ), we randomly select w * , and generate 2000 samples for the dataset.

Batch sizes for all algorithms are set to be 1.

We report the performances of SGD, SGD+Nesterov and SGD+HB using their best hyper-parameter setting selected from dense grid search.

On the other hand, we do not tune hyper-parameters of MaSS, but use the suggested setting by our theoretical analysis, Eq. 14.

Specifically, we use Component decoupled:

3-d Gaussian:

For ASGD, we use the setting suggested by (8) .

Figure 6 (in addition to Fig 4) and Figure 7 show the curves of the compared algorithms under various data settings.

We observe that: 1) SGD+Nesterov with its best hyper-parameters is almost identical to the optimal SGD; 2) MaSS, with the suggested hyper-parameter selections, converges faster than all of the other algorithms, especially SGD.

These observations are consistent with our theoretical results about non-acceleration of SGD+Nesterov, Theorem 1, and accelerated convergence of MaSS, Theorem 3.

Recall that MaSS differs from SGD+Nesterov by only a compensation term, this experiment illustrates the importance of this term.

Note that the vertical axis is log scaled.

Then the linear decrease of log losses in the plots implies an exponential loss decrease, and the slopes correspond to the coefficients in the exponents.

Divergence of SGD+Nesterov with large step size.

As discussed in Corollary 1, SGD+Nesterov diverges with step size ?? * = 1/L 1 (when ?? ??? [0.6, 1]), which is the optimal choice of step size for both SGD and MaSS.

We run SGD+Nesterov, with step size ?? * = 1/L 1 , to optimize the square loss function on component decoupled data mentioned above.

Figure 8 shows the divergence of SGD+Nesterov with two common choices of momentum parameter ??: 0.9 and 0.99.

with the parameter ?? on the loss function they prove convergence rate 1 ??? 1/?? 2 ?? t of their method (called SGD with Nesterov acceleration in their paper), where t is the iteration number.

In the following, we show that, on a simple (zero-mean) Gaussian distributed data, this rate is slower than that of SGD, which has a rate of (1 ??? 1/??) t .

On the other hand, MaSS achieves the accelerated rate 1 ??? 1/ (2 + d)?? t .

We empirically verify the three-regime partition observed in section 5 using zero-mean Gaussian data.

In this evaluation, we set the covariant matrix of the (zero-mean) Gaussian to be: In the experiments, we run MaSS with a variaty of mini-batch size m, ranging from 1 to 160, on this Gaussian dataset.

For each training process, we compute the convergence speed s(m), which is defined to be the inverse of the number of iterations needed to achieve a training error of ??.

Fully-connected Network.

The fully-connected neural network has 3 hidden layers, with 100 ReLU-activated neurons in each layer.

After each hidden layer, there is a dropout layer with keep probability 0.5.

This network takes 784-dimensional vectors as input, and has 10 softmax-activated output neurons.

It has ???99k trainable parameters in total.

Convolutional Neural Network (CNN).

The CNN we considered has three convolutional layers with kernel size of 5 ?? 5 and without padding.

The first two convolutional layers have 64 channels each, while the last one has 128 channels.

Each convolutional layer is followed by a 2 ?? 2 max pooling layer with stride of 2.

On top of the last max pooling layer, there is a fully-connected ReLU-activated layer of size 128 followed by the output layer of size 10 with softmax non-linearity.

A dropout layer with keep probability 0.5 is applied after the full-connected layer.

The CNN has ???576k trainable parameters in total.

Residual Network (ResNet).

We train a ResNet (7) with 32 convolutional layers.

The ResNet-32 has a sequence of 15 residual blocks: the first 5 blocks have an output of shape 32 ?? 32 ?? 16, the following 5 blocks have an output of shape 16??16??32 and the last 5 blocks have an output of shape 8??8??64.

On top of these blocks, there is a 2??2 average pooling layer with stride of 2, followed by a output layer of size 10 with softmax non-linearity.

The ResNet-32 has ???467k trainable parameters in total.

We use the fully-connected network to classify the MNIST dataset, and use CNN and ResNet to classify the CIFAR-10 dataset.

@highlight

This work proves the non-acceleration of Nesterov SGD with any hyper-parameters, and proposes new algorithm which provably accelerates SGD in the over-parameterized setting.
Machine learning (ML) models trained by differentially private stochastic gradient descent (DP-SGD) have much lower utility than the non-private ones.

To mitigate this degradation, we propose a DP Laplacian smoothing SGD (DP-LSSGD) to train ML models with differential privacy (DP) guarantees.

At the core of DP-LSSGD is the Laplacian smoothing, which smooths out the Gaussian noise used in the Gaussian mechanism.

Under the same amount of noise used in the Gaussian mechanism, DP-LSSGD attains the same DP guarantee, but a better utility especially for the scenarios with strong DP guarantees.

In practice, DP-LSSGD makes training both convex and nonconvex ML models more stable and enables the trained models to generalize better.

The proposed algorithm is simple to implement and the extra computational complexity and memory overhead compared with DP-SGD are negligible.

DP-LSSGD is applicable to train a large variety of ML models, including DNNs.

Many released machine learning (ML) models are trained on sensitive data that are often crowdsourced or contain private information (Yuen et al., 2011; Feng et al., 2017; Liu et al., 2017) .

With overparameterization, deep neural nets (DNNs) can memorize the private training data, and it is possible to recover them and break the privacy by attacking the released models (Shokri et al., 2017) .

For example, Fredrikson et al. demonstrated that a model-inversion attack can recover training images from a facial recognition system (Fredrikson et al., 2015) .

Protecting the private data is one of the most critical tasks in ML.

Differential privacy (DP) (Dwork et al., 2006 ) is a theoretically rigorous tool for designing algorithms on aggregated databases with a privacy guarantee.

The idea is to add a certain amount of noise to randomize the output of a given algorithm such that the attackers cannot distinguish outputs of any two adjacent input datasets that differ in only one entry.

For repeated applications of additive noise based mechanisms, many tools have been invented to analyze the DP guarantee for the model obtained at the final stage.

These include the basic and strong composition theorems and their refinements (Dwork et al., 2006; 2010; Kairouz et al., 2015) , the moments accountant (Abadi et al., 2016) , etc.

Beyond the original notion of DP, there are also many other ways to define the privacy, e.g., local DP (Duchi et al., 2014) , concentrated/zeroconcentrated DP (Dwork & Rothblum, 2016; Bun & Steinke, 2016) , and R??nyi-DP (RDP) (Mironov, 2017) .

Differentially private stochastic gradient descent (DP-SGD) reduces the utility of the trained models severely compared with SGD.

As shown in Figure 1 , the training and validation losses of the logistic regression on the MNIST dataset increase rapidly when the DP guarantee becomes stronger.

The convolutional neural net (CNN) 1 trained by DP-SGD has much lower testing accuracy than the non-private one on the MNIST.

We will discuss the detailed experimental settings in Section 4.

A natural question raised from such performance degradations is:

Can we improve DP-SGD, with negligible extra computational complexity and memory cost, such that it can be used to train general ML models with improved utility?

We answer the above question affirmatively by proposing differentially private Laplacian smoothing SGD (DP-LSSGD) to improve the utility in privacy-preserving empirical risk minimization (ERM).

DP-LSSGD leverages the Laplacian smoothing (Osher et al., 2018) as a post-processing to smooth the injected Gaussian noise in the differentially private SGD (DP-SGD) to improve the convergence of DP-SGD in training ML models with DP guarantee.

The main contributions of our work are highlighted as follows:

??? We propose DP-LSSGD and prove its privacy and utility guarantees for convex/nonconvex optimizations.

We prove that under the same privacy budget, DP-LSSGD achieves better utility, excluding a small term that is usually dominated by the other terms, than DP-SGD by a factor that is much less than one for convex optimization.

??? We perform a large number of experiments logistic regression and CNN to verify the utility improvement by using DP-LSSGD.

Numerical results show that DP-LSSGD remarkably reduces training and validation losses and improves the generalization of the trained private models.

In Table 1 , we compare the privacy and utility guarantees of DP-LSSGD and DP-SGD.

For the utility, the notation??(??) hides the same constant and log factors for each bound.

The constants d and n denote the dimension of the model's parameters and the number of training points, respectively.

The numbers ?? and ?? are positive constants that are strictly less than one, and D 0 , D ?? , G are positive constants, which will be defined in Section 3.

?? , we will discuss this in detail in Section 4.

There is a massive volume of research over the past decade on designing algorithms for privacypreserving ML.

Objective perturbation, output perturbation, and gradient perturbation are the three major approaches to perform ERM with a DP guarantee.

Chaudhuri & Monteleoni (2008) ; Chaudhuri et al. (2011) considered both output and objective perturbations for privacy-preserving ERM, and gave theoretical guarantees for both privacy and utility for logistic regression and SVM.

Song et al. (2013) numerically studied the effects of learning rate and batch size in DP-ERM.

Wang et al. (2016) studied stability, learnability and other properties of DP-ERM.

Lee & Kifer (2018) proposed an adaptive per-iteration privacy budget in concentrated DP gradient descent.

Variance reduction techniques, e.g., SVRG, have also been introduced to DP-ERM .

The utility bound of DP-SGD has also been analyzed for both convex and nonconvex smooth objectives (Bassily et al., 2014; .

Jayaraman et al. (2018) analyzed the excess empirical risk of DP-ERM in a distributed setting.

Besides ERM, many other ML models have been made differentially private.

These include: clustering (Su et al., 2015; Y. Wang & Singh, 2015; Balcan et al., 2017) , matrix completion (Jain et al., 2018) , online learning (Jain et al., 2012) , sparse learning (Talwar et al., 2015; Wang & Gu, 2019) , and topic modeling (Park et al., 2016) .

Gilbert & McMillan (2017) exploited the ill-conditionedness of inverse problems to design algorithms to release differentially private measurements of the physical system.

considered sparse linear regression in the local DP models.

Shokri & Shmatikov (2015) proposed distributed selective SGD to train deep neural nets (DNNs) with a DP guarantee in a distributed system, however, the obtained privacy guarantee was very loose.

Abadi et al. (2016) considered applying DP-SGD to train DNNs in a centralized setting.

They clipped the gradient 2 norm to bound the sensitivity and invented the moment accountant to get better privacy loss estimation.

Papernot et al. (2017) proposed Private Aggregation of Teacher Ensembles/PATE based on the semi-supervised transfer learning to train DNNs, and this framework improves both privacy and utility on top of the work by Abadi et al. (2016) .

Recently Papernot et al. (2018) introduced new noisy aggregation mechanisms for teacher ensembles that enable a tighter theoretical DP guarantee.

The modified PATE is scalable to the large dataset and applicable to more diversified ML tasks.

Laplacian smoothing (LS) can be regarded as a denoising technique that performs post-processing on the Gaussian noise injected stochastic gradient.

Denoising has been used in the DP earlier: Postprocessing can enforce consistency of contingency table releases (Barak et al., 2007) and leads to accurate estimation of the degree distribution of private network (Hay et al., 2009 ).

Nikolov et al. (2013) showed that post-processing by projecting linear regression solutions, when the ground truth solution is sparse, to a given 1 -ball can remarkably reduce the estimation error.

Bernstein et al. (2017) used Expectation-Maximization to denoise a class of graphical models' parameters.

showed that in the output perturbation based differentially private algorithm design, denoising dramatically improves the accuracy of the Gaussian mechanism in the high-dimensional regime.

To the best of our knowledge, we are the first to design a denoising technique on the Gaussian noise injected gradient to improve the utility of the trained private ML models.

We use boldface upper-case letters A, B to denote matrices and boldface lower-case letters x, y to denote vectors.

For vectors x and y and positive definite matrix A, we use x 2 and x A to denote the 2 -norm and the induced norm by A, respectively; x, y denotes the inner product of x and y; and ?? i (A) denotes the i-th largest eigenvalue of A. We denote the set of numbers from 1 to n by [n] .

N (0, I d??d ) represents d-dimensional standard Gaussian.

This paper is organized in the following way: In Section 2, we introduce the DP-LSSGD algorithm.

In Section 3, we analyze the privacy and utility guarantees of DP-LSSGD for both convex and nonconvex optimizations.

We numerically verify the efficiency of DP-LSSGD in Section 4.

We conclude this work and point out some future directions in Section 5.

2 PROBLEM SETUP AND ALGORITHM

In this paper, we consider empirical risk minimization problem as follows.

Given a training set S = {(x 1 , y 1 ), . . .

, (x n , y n )} drawn from some unknown but fixed distribution, we aim to find an empirical risk minimizer that minimizes the empirical risk as follows,

where F (w) is the empirical risk (a.k.a., training loss), f i (w) = (w; x i , y i ) is the loss function of a given ML model defined on the i-th training example (x i , y i ), and w ??? R d is the model parameter we want to learn.

Empirical risk minimization serves as the mathematical foundation for training many ML models that are mentioned above.

The LSSGD (Osher et al., 2018) for solving (1) is given by

where ?? is the learning rate, ???f i k denotes the stochastic gradient of F evaluated from the pair of input-output {x i k , y i k }, and B k is a random subset of size b from [n].

Let A ?? = I ??? ??L for ?? ??? 0 being a constant, where I ??? R d??d and L ??? R d??d are the identity and the discrete one-dimensional Laplacian matrix with periodic boundary condition, respectively.

Therefore,

When ?? = 0, LSSGD reduces to SGD.

Note that A ?? is positive definite with condition number 1 + 4?? that is independent of A ?? 's dimension, and LSSGD guarantees the same convergence rate as SGD in both convex and nonconvex optimization.

Moreover, Laplacian smoothing (LS) can reduce the variance of SGD on-thefly, and lead to better generalization in training many ML models including DNNs (Osher et al., 2018) .

T and * is the convolution operator.

By the fast Fourier transform (FFT), we have A

, where the division in the right hand side parentheses is performed in a coordinate wise way.

DP ERM aims to learn a DP model, w, for the problem (1).

A common approach is injecting Gaussian noise into the stochastic gradient, and it resulting in the following DP-SGD

where n is the injected Gaussian noise for DP guarantee.

Note that the LS matrix A ???1 ?? can remove the noise in v. If we assume v is the initial signal, then A ???1 ?? v can be regarded as performing an approximate diffusion step on the initial noisy signal which removes the noise from v. We will provide a detailed argument for the diffusion process in the appendix.

As numerical illustrations, we consider the following two signals:

We reshape v 2 into 1D with row-major ordering and then perform LS.

Figure 2 shows that LS can remove noise efficiently.

This noise removal property enables LSSGD to be more stable to the noise injected stochastic gradient, therefore improves training DP models with gradient perturbations.

We propose the following DP-LSSGD for solving (1) with DP guarantee

In this scheme, we first inject the noise n to the stochastic gradient ???f i k (w k ), and then apply the LS operator A ???1 ?? to denoise the noisy stochastic gradient, ???f i k (w k ) + n, on-the-fly.

We assume that each component function f i in (1) is G-Lipschitz.

The DP-LSSGD for finite-sum optimization is summarized in Algorithm 1.

Compared with LSSGD, the main difference of DP-LSSGD lies in injecting Gaussian noise into the stochastic gradient, before applying the Laplacian smoothing, to guarantee the DP.

initial guess of w, ( , ??): the privacy budget, ??: the step size, T : the total number of iterations.

and ?? is defined in Theorem 1, and

In this section, we present the privacy and utility guarantees for DP-LSSGD.

The technical proofs are provided in the appendix.

Definition 1 (( , ??)-DP). (Dwork et al. (2006) ) A randomized mechanism M : S N ??? R satisfies ( , ??)-DP if for any two adjacent datasets S, S ??? S N differing by one element, and any output subset O ??? R, it holds that

Theorem 1 (Privacy Guarantee).

Suppose that each component function f i is G-Lipschitz.

Given the total number of iterations T , for any ?? > 0 and privacy budget 2 ??? 5T log(1/??)b 2 /n 2 , DP-LSSGD, with injected Gaussian noise N (0, ?? 2 ) for each coordinate, satisfies ( , ??)-DP with ?? 2 = 8T ??G 2 /(n 2 ), where ?? = 2 log(1/??)/ + 1.

Remark 1.

It is straightforward to show that the noise in Theorem 1 is in fact also tight to guarantee the ( , ??)-DP for DP-SGD.

For convex ERM, DP-LSSGD guarantees the following utility in terms of the gap between the ergodic average of the points along the DP-LSSGD path and the optimal solution w * .

Theorem 2 (Utility Guarantee for convex optimization).

Suppose F is convex and each component function

A?? and w * is the global minimizer of F , the DP-LSSGD outputw =

, where ?? = 2??+1??? ??? 4??+1 2?? < 1.

That is, ?? converge to 0 almost exponentially as the dimension, d, increases.

Remark 2.

In the above utility bound for convex optimization, for different ?? (?? = 0 corresponds to DP-SGD), the only difference lies in the term ??(D ?? + G 2 ).

The first part ??D ?? depends on the gap between initialization w 0 and the optimal solution w * .

The second part ??G 2 decrease monotonically as ?? increases.

?? should be selected to get an optimal trade-off between these two parts.

Based on our test on multi-class logistic regression for MNIST classification, ?? = 0 always outperforms the case when ?? = 0.

For nonconvex ERM, DP-LSSGD has the following utility bound measured in gradient norm.

Theorem 3 (Utility Guarantee for nonconvex optimization).

Suppose that F is nonconvex and each component function f i is G-Lipschitz and has L-Lipschitz continuous gradient.

Given

with w * being the global minimum of F , then the DP-LSSGD outputw = T ???1 k=0 w k /T satisfies the following utility

Proposition 2.

In Theorem 3, ?? =

It is worth noting that if we use the 2 -norm instead of the induced norm, we have the following utility guarantee

In the 2 -norm, DP-LSSGD has a bigger utility upper bound than DP-SGD (set ?? = 0 in ??).

However, this does not mean that DP-LSSGD has worse performance.

To see this point, let us consider the following simple nonconvex function

For two points a 1 = (2, 0) and a 2 = (1, ??? 3/2), the distance to the local minima a * = (0, 0) are 2 and ??? 7/2, while ???f (a 1 ) 2 = 1 and ???f (a 2 ) 2 = ??? 13/2.

So a 2 is closer to the local minima a * than a 1 while its gradient has a larger 2 -norm.

In this section, we verify the efficiency of DP-LSSGD in training multi-class logistic regression and CNNs for MNIST and CIFAR10 classification.

We use v ??? v/ max (1, v 2 /C) (Abadi et al., 2016) to clip the gradient 2 -norms of the CNNs to C. The gradient clipping guarantee the Lipschitz condition for the objective functions.

We train all the models below with ( , 10 ???5 )-DP guarantee for different .

For Logistic regression we use the privacy budget given by Theorem 1, and for CNNs we use the privacy budget in the Tensorflow privacy (Andrew & et al., 2019) .

We checked that these two privacy budgets are consistent.

We ran 50 epochs of DP-LSSGD with learning rate scheduled as 1/t with t being the index of the iteration to train the 2 -regularized (regularization constant 10 ???4 ) multi-class logistic regression.

We split the training data into 50K/10K with batch size 128 for cross-validation.

We plot the evolution of training and validation loss over iterations for privacy budgets (0.2, 10 ???5 ) and (0.1, 10 ???5 ) in Figure 3 .

We see that the training loss curve of DP-SGD (?? = 0) is much higher and more oscillatory (log-scale on the y-axis) than that of DP-LSSGD (?? = 1, 3).

Also, the validation loss of the model trained by DP-LSSGD decays faster and has a much smaller loss value than that of the model trained by DP-SGD.

Moreover, when the privacy guarantee gets stronger, the utility improvement by DP-LSSGD becomes more significant.

Next, consider the testing accuracy of the multi-class logistic regression trained with ( , 10 ???5 )-DP guarantee by DP-LSSGD includes ?? = 0, i.e., DP-SGD.

We list the test accuracy of logistic regression trained in different settings in Table 2 .

These results reveal that DP-LSSGD with ?? = 1, 2, 3 can improve the accuracy of the trained private model and also reduce the variance, especially when the privacy guarantee is very strong, e.g., (0.1, 10 ???5 ).

We know that the step size in DP-SGD/DP-LSSGD may affect the accuracy of the trained private models.

We try different step size scheduling of the form {a/t|a = 0.5, 1.0, 1.5, 2.0, 2.5, 3.0}, where t is again the index of iteration, and all the other hyper-parameters are used the same as before.

Figure.

4 plots the test accuracy of the logistic regression model trained with different learning rate scheduling and different privacy budget.

We see that the private logistic regression model trained by DP-LSSGD always outperforms DP-SGD.

In this subsection, we consider training a small CNN 2 with DP-guarantee for MNIST classification.

We implement DP-LSSGD and DP-LSAdam (Kingma & Ba, 2014 ) (simply replace the noisy gradient in DP-Adam in the Tensorflow privacy with the Laplacian smoothed surrogate) into the Tensorflow privacy framework (Andrew & et al., 2019) .

We use the default learning rate 0.15 for DP-(LS)SGD and 0.001 for DP-(LS)Adam and decay them by a factor of 10 at the 10K-th iteration, norm clipping (1), batch size (256), and micro-batches (256).

We vary the noise multiplier (NM), and larger NM guarantees stronger DP.

As shown in Figure 5 , the privacy budget increases at exactly the same speed (dashed red line) for four optimization algorithms.

When the NM is large, i.e., DP-guarantee is strong, DP-SGD performs very well in the initial period.

However, after a few epochs, the validation accuracy gets highly oscillatory and decays.

DP-LSSGD can mitigate the training instability issue of DP-SGD.

DP-Adam outperforms DP-LSSGD, and DP-LSAdam can further improve validation accuracy on top of DP-Adam.

Next, we consider the effects of the LS constant (??) and the learning rate in training the DP-CNN for MNIST classification.

We fixed the NM to be 10, and run 60 epochs of DP-SGD and DP-LSSGD with different ?? and different learning rate.

We show the comparison of DP-SGD with DP-LSSGD with different ?? in the left panel of Figure 6 , and we see that as ?? increases it becomes more stable in training CNNs with DP-guarantee even though initially it becomes slightly slower.

In the middle panel of Figure 6 , we plot the evolution of validation accuracy curves of the DP-CNN trained by DP-SGD and DP-LSSGD with different learning rate, where the solid lines represent results for DP-LSSGD and dashed lines for DP-SGD.

DP-LSSGD outperforms DP-SGD in all learning rates tested, and DP-LSSGD is much more stable than DP-SGD when a larger learning rate is used.

Finally, we go back to the accuracy degradation problem raised in Figure 1 .

As shown in Figure 3 , LS can efficiently reduce both training and validation losses in training multi-class logistic regression for MNIST classification.

Moreover, as shown in the right panel of Figure 6 , DP-LSSGD can improve the testing accuracy of the CNN used above significantly.

In particular, DP-LSSGD improves the testing accuracy of CNN by 3.2% and 5.0% for (0.4, 10 ???5 ) and (0.2, 10 ???5 ), respectively, on top of DP-SGD.

DP-LSAdam can further boost test accuracy.

All the accuracies associated with any given privacy budget in Figure 6 (right panel), are the optimal ones searched over the results obtained in the above experiments with different learning rate, number of epochs, and NM.

Due to page limitation, we put the results of DP-CNN for CIFAR10 classification in the appendix.

In this paper, we integrated Laplacian smoothing with DP-SGD for privacy-presrving ERM.

The resulting algorithm is simple to implement and the extra computational cost compared with the DP-SGD is almost negligible.

We show that DP-LSSGD can improve the utility of the trained private ML models both numerically and theoretically.

It is straightforward to combine LS with other variance reduction technique, e.g., SVRG (Johoson & Zhang, 2013) .

To prove the privacy guarantee in Theorem 1, we first introduce the following 2 -sensitivity.

Definition 2 ( 2 -Sensitivity).

For any given function f (??), the 2 -sensitivity of f is defined by

where S ??? S 1 = 1 means the data sets S and S differ in only one entry.

We will adapt the concepts and techniques of R??nyi DP (RDP) to prove the DP-guarantee of the proposed DP-LSSGD.

Definition 3 (RDP).

For ?? > 1 and ?? > 0, a randomized mechanism M : S n ??? R satisfies (??, ??)-R??nyi DP, i.e., (??, ??)-RDP, if for all adjacent datasets S, S ??? S n differing by one element, we have

where the expectation is taken over M(S ).

Lemma 1. )

Given a function q : S n ??? R, the Gaussian Mechanism M = q(S) + n, where n ??? N (0, ?? 2 I), satisfies (??, ????? 2 (q)/(2?? 2 ))-RDP.

In addition, if we apply the mechanism M to a subset of samples using uniform sampling without replacement, M satisfies

Moreover, the input of the i-th mechanism can be based on outputs of the previous (i ??? 1) mechanisms.

Lemma 3.

If a randomized mechanism M : S n ??? R satisfies (??, ??)-RDP, then M satisfies (?? + log(1/??)/(?? ??? 1), ??)-DP for all ?? ??? (0, 1).

With the definition (Def.

3) and guarantees of RDP (Lemmas 1 and 2), and the connection between RDP and ( , ??)-DP (Lemma 3), we can prove the following DP-guarantee for DP-LSSGD.

Proof of Theorem 1.

Let us denote the update of DP-SGD and DP-LSSGD at the k-th iteration starting from any given points w k andw k , respectively, as

where B k is a mini batch that are drawn uniformly from [n], and |B k | = b is the mini batch size.

We will show that with the aforementioned Gaussian noise N (0, ?? 2 ) for each coordinate of n, the output of DP-SGD,w, after T iterations is ( , ??)-DP.

Let us consider the mechanismM k =

b .

According to Lemma 1, if we add noise with variance

the mechanism M k will satisfy ??, (n 2 /b 2 ) log(1/??)/ 2(?? ??? 1)T -RDP.

By post-processing theorem, we immediately have that under the same noise,

According to Lemma 1,M k will satisfy ??, log(1/??)/(?????1)T -RDP provided that ?? 2 ??? 1/1.25, because ?? = b/n.

Let ?? = 2 log(1/??)/ + 1, we obtain thatM k satisfies 2 log(1/??)/ + 1, /(2T ) -RDP as long as we have

Therefore, the following condition suffices

Therefore, according to Lemma 2, we have w k satisfies 2 log(1/??)/ + 1, k /(2T ) -RDP.

Finally, by Lemma 3, we have w k satisfies k /(2T ) + /2, ?? -DP.

Therefore, the output of DP-SGD,w, is ( , ??)-DP.

Remark 3.

In the above proof, we used the following estimate of the 2 sensitivity

?? g, then according to Osher et al. (2018) we have

where d is the dimension of d, and

Moreover, if we assume the g is randomly sampled from a unit ball in a high dimensional space, then a high probability estimation of the compression ratio of the 2 norm can be derived from Lemma.

5.

, so for the above noise, it can give much stronger privacy guarantee.

To prove the utility guarantee for convex optimization, we first show that the LS operator compresses the 2 norm of any given Gaussian random vector with a specific ratio in expectation.

Lemma 4.

Let x ??? R d be the standard Gaussian random vector.

Then

Proof of Theorem 2.

Recall that we have the following update rule w

, where i k are drawn uniformly from [n], and n ??? N (0, ?? 2 I).

Observe that

Taking expectation with respect to i k and n given w k , we have

where the second inequality is due to the convexity of F , and Lemma 4.

It implies that

Now taking the full expectation and summing up over T iterations, we have

where

According to the definition ofw and the convexity of F , we obtain

To prove the utility guarantee for nonconvex optimization, we need the following lemma, which shows that the LS operator compresses the 2 norms of any given Gaussian random vector with a specific ratio in expectation.

Lemma 5.

Let x ??? R d be the standard Gaussian random vector.

Then

Proof of Lemma 5.

Let the eigenvalue decomposition of A ???1

Proof of Theorem 3.

Recall that we have the following update rule w

, where i k are drawn uniformly from [n], and n ??? N (0, ?? 2 I).

Since F is L-smooth, we have

Taking expectation with respect to i k and n given w k , we have

where the second inequality uses Lemma 5 and the last inequality is due to 1 ??? ?? k L/2 > 1/2.

Now taking the full expectation and summing up over T iterations, we have

2 .

If we choose fix step size, i.e., ?? k = ??, and rearranging the above inequality, and using

which implies that

B CALCULATIONS OF ?? AND ??

To prove Proposition 1, we need the following two lemmas.

Lemma 6 (Residue Theorem).

Let f (z) be a complex function defined on C, then the residue of f around the pole z = c can be computed by the formula

where the order of the pole c is n. Moreover,

where {c i } be the set of pole(s) of f (z) inside {z||z| < 1}.

The proof of Lemma 6 can be found in any complex analysis textbook.

Lemma 7.

For 0 ??? ?? ??? 2??, suppose

has the discrete-time Fourier transform of series f [k].

Then, for integer k,

Proof.

By definition,

We compute (11) by using Residue theorem.

First, note that because

; therefore, it suffices to compute (11)) for nonnegative k. Set z = e i?? .

Observe that cos(??) = 0.5(z + 1/z) and dz = izd??.

Substituting in (11) and simplifying yields that

where the integral is taken around the unit circle, and

are the roots of quadratic ?????z 2 + (2?? + 1)z ??? ??.

Note that ?? ??? lies within the unit circle; whereas, ?? + lies outside of the unit circle.

Therefore, because k is nonnegative, ?? ??? is the only singularity of the integrand in (12) within the unit circle.

A straightforward application of the Residue Theorem, i.e., Lemma 6, yields that

This completes the proof.

Proof of Proposition 1.

First observe that we can re-write ?? as

It remains to show that the above summation is equal to

.

This follows by lemmas 7 and standard sampling results in Fourier analysis (i.e. sampling ?? at points {2??j/d}

).

Nevertheless, we provide the details here for completeness: Observe that that the inverse discrete-time Fourier transform of

is given by

otherwise.

The proof is completed by substituting the result of lemma 7 in the above sum and simplifying.

We list some typical values of ?? in Table 1 . .

Therefore, we have

We list some typical values of ?? in Table 2 .

where v 0 is the discretization of f (x), and v ???t is the numerical solution of (15) at time ???t.

Therefore, we have v ???t = (I ??? ???tL)

which is the LS with ?? = ???t.

In this section, we will show that LS can also improve the utility of the DP-CNN trained by DP-SGD and DP-Adam for CIFAR10 classification.

We simply replace the CNN architecture used above for MNIST classification with the benchmark architecture in the Tensorflow tutorial 3 for CIFAR10 classification.

Also, we use the same set of parameters as that used for training DP-CNN for MNIST classification except we fixed the noise multiplier to be 2.0 and clip the gradient 2 norm to 3.

As shown in Figure 7 , LS can significantly improve the validation accuracy of the model trained by DP-SGD and DP-Adam, and the DP guarantee for all these algorithms are the same (dashed line in Figure 7 ).

@highlight

We propose a differentially private Laplacian smoothing stochastic gradient descent to train machine learning models with better utility and maintain differential privacy guarantees.
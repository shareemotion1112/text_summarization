Robust estimation under Huber's $\epsilon$-contamination model has become an important topic in statistics and theoretical computer science.

Rate-optimal procedures such as Tukey's median and other estimators based on statistical depth functions are impractical because of their computational intractability.

In this paper, we establish an intriguing connection between f-GANs and various depth functions through the lens of f-Learning.

Similar to the derivation of f-GAN, we show that these depth functions that lead to rate-optimal robust estimators can all be viewed as variational lower bounds of the total variation distance in the framework of f-Learning.

This connection opens the door of computing robust estimators using tools developed for training GANs.

In particular, we show that a JS-GAN that uses a neural network discriminator with at least one hidden layer is able to achieve the minimax rate of robust mean estimation under Huber's $\epsilon$-contamination model.

Interestingly, the hidden layers of the neural net structure in the discriminator class are shown to be necessary for robust estimation.

In the setting of Huber's -contamination model (Huber, 1964; 1965) , one has i.i.d observations X 1 , ..., X n ∼ (1 − )P θ + Q,and the goal is to estimate the model parameter θ.

Under the data generating process (1), each observation has a 1 − probability to be drawn from P θ and the other probability to be drawn from the contamination distribution Q. The presence of an unknown contamination distribution poses both statistical and computational challenges to the problem.

For example, consider a normal mean estimation problem with P θ = N (θ, I p ).

Due to the contamination of data, the sample average, which is optimal when = 0, can be arbitrarily far away from the true mean if Q charges a positive probability at infinity.

Moreover, even robust estimators such as coordinatewise median and geometric median are proved to be suboptimal under the setting of (1) (Chen et al., 2018; Diakonikolas et al., 2016a; Lai et al., 2016) .

The search for both statistically optimal and computationally feasible procedures has become a fundamental problem in areas including statistics and computer science.

For the normal mean estimation problem, it has been shown in Chen et al. (2018) that the minimax rate with respect to the squared 2 loss is p n ∨ 2 , and is achieved by Tukey's median (Tukey, 1975) .

Despite the statistical optimality of Tukey's median, its computation is not tractable.

In fact, even an approximate algorithm takes O(e Cp ) in time BID1 Chan, 2004; Rousseeuw & Struyf, 1998) .Recent developments in theoretical computer science are focused on the search of computationally tractable algorithms for estimating θ under Huber's -contamination model (1).

The success of the efforts started from two fundamental papers Diakonikolas et al. (2016a) ; Lai et al. (2016) , where two different but related computational strategies "iterative filtering" and "dimension halving" were proposed to robustly estimate the normal mean.

These algorithms can provably achieve the minimax rate p n ∨ 2 up to a poly-logarithmic factor in polynomial time.

The main idea behind the two methods is a critical fact that a good robust moment estimator can be certified efficiently by higher moments.

This idea was later further extended (Diakonikolas et al., 2017; Du et al., 2017; Diakonikolas et al., 2016b; 2018a; c; b; Kothari et al., 2018) to develop robust and computable procedures for various other problems.

However, many of the computationally feasible procedures for robust mean estimation in the literature rely on the knowledge of covariance matrix and sometimes the knowledge of contamination proportion.

Even though these assumptions can be relaxed, nontrivial modifications of the algorithms are required for such extensions and statistical error rates may also be affected.

Compared with these computationally feasible procedures proposed in the recent literature for robust estimation, Tukey's median (9) and other depth-based estimators (Rousseeuw & Hubert, 1999; Mizera, 2002; Zhang, 2002; Mizera & Müller, 2004; Paindaveine & Van Bever, 2017) have some indispensable advantages in terms of their statistical properties.

First, the depth-based estimators have clear objective functions that can be interpreted from the perspective of projection pursuit (Mizera, 2002) .

Second, the depth-based procedures are adaptive to unknown nuisance parameters in the models such as covariance structures, contamination proportion, and error distributions (Chen et al., 2018; Gao, 2017) .

Last but not least, Tukey's depth and other depth functions are mostly designed for robust quantile estimation, while the recent advancements in the theoretical computer science literature are all focused on robust moments estimation.

Although this is not an issue when it comes to normal mean estimation, the difference is fundamental for robust estimation under general settings such as elliptical distributions where moments do not necessarily exist.

Given the desirable statistical properties discussed above, this paper is focused on the development of computational strategies of depth-like procedures.

Our key observation is that robust estimators that are maximizers of depth functions, including halfspace depth, regression depth and covariance matrix depth, can all be derived under the framework of f -GAN (Nowozin et al., 2016) .

As a result, these depth-based estimators can be viewed as minimizers of variational lower bounds of the total variation distance between the empirical measure and the model distribution (Proposition 2.1).

This observation allows us to leverage the recent developments in the deep learning literature to compute these variational lower bounds through neural network approximations.

Our theoretical results give insights on how to choose appropriate neural network classes that lead to minimax optimal robust estimation under Huber's -contamination model.

In particular, Theorem 3.1 and 3.2 characterize the networks which can robustly estimate the Gaussian mean by TV-GAN and JS-GAN, respectively; Theorem 4.1 is an extension to robust location estimation under the class of elliptical distributions which includes Cauchy distribution whose mean does not exist.

Numerical experiments in Section 5 are provided to show the success of these GANs.

We start with the definition of f -divergence (Csiszár, 1964; BID0 .

Given a strictly convex function f that satisfies f (1) = 0, the f -GAN between two probability distributions P and Q is defined by DISPLAYFORM0 Here, we use p(·) and q(·) to stand for the density functions of P and Q with respect to some common dominating measure.

For a fully rigorous definition, see Polyanskiy & Wu (2017) .

Let f * be the convex conjugate of f .

That is, f * (t) = sup u∈dom f (ut − f (u)).

A variational lower bound of (2) is DISPLAYFORM1 Note that the inequality (3) holds for any class T , and it becomes an equality whenever the class T contains the function f (p/q) (Nguyen et al., 2010) .

For notational simplicity, we also use f for an arbitrary element of the subdifferential when the derivative does not exist.

With i.i.d.

observations X 1 , ..., X n ∼ P , the variational lower bound (3) naturally leads to the following learning method DISPLAYFORM2 The formula (4) is a powerful and general way to learn the distribution P from its i.i.d.

observations.

It is known as f -GAN (Nowozin et al., 2016) , an extension of GAN (Goodfellow et al., 2014) , which stands for generative adversarial networks.

The idea is to find a P so that the best discriminator T in the class T cannot tell the difference between P and the empirical distribution DISPLAYFORM3

Our f -Learning framework is based on a special case of the variational lower bound (3).

That is, DISPLAYFORM0 where q(·) stands for the density function of Q. Note that here we allow the class Q Q to depend on the distribution Q in the second argument of D f (P Q).

Compare (5) with (3), and it is easy to realize that (5) is a special case of (3) with DISPLAYFORM1 Moreover, the inequality (5) becomes an equality as long as P ∈ Q Q .

The sample version of (5) leads to the following learning method DISPLAYFORM2 The learning method (7) will be referred to as f -Learning in the sequel.

It is a very general framework that covers many important learning procedures as special cases.

For example, consider the special case where Q Q = Q independent of Q, Q = Q, and f (x) = x log x. Direct calculations give f (x) = log x + 1 and f * (t) = e t−1 .

Therefore, (7) becomes DISPLAYFORM3 which is the maximum likelihood estimator (MLE).

An important generator f that we will discuss here is f (x) = (x−1) + .

This leads to the total variation distance DISPLAYFORM0 |p − q|.

With f (x) = I{x ≥ 1} and f * (t) = tI{0 ≤ t ≤ 1}, the TV-Learning is given by DISPLAYFORM1 A closely related idea was previously explored by Yatracos (1985) ; Devroye & Lugosi (2012) .

The following proposition shows that when Q Q approaches to Q in some neighborhood, TV-Learning leads to robust estimators that are defined as the maximizers of various depth functions including Tukey's depth, regression depth, and covariance depth.

Proposition 2.1.

The TV-Learning (8) includes the following special cases:1.

Tukey's halfspace depth: DISPLAYFORM2 2.

Regression depth: Take Q = P y,X = P y|X P X : DISPLAYFORM3 and Q η = P y,X = P y|X P X : DISPLAYFORM4 3.

Covariance matrix depth: Take Q = {N (0, Γ) : Γ ∈ E p }, where E p stands for the class of p × p covariance matrices, and Q Γ = N (0, Γ) : DISPLAYFORM5 becomes DISPLAYFORM6 The formula (9) is recognized as Tukey's median, the maximizer of Tukey's halfspace depth.

A traditional understanding of Tukey's median is that (9) maximizes the halfspace depth (Donoho & Gasko, 1992 ) so that θ is close to the centers of all one-dimensional projections of the data.

In the f -Learning framework, N ( θ, I p ) is understood to be the minimizer of a variational lower bound of the total variation distance.

The formula (10) gives the estimator that maximizes the regression depth proposed by Rousseeuw & Hubert (1999) .

It is worth noting that the derivation of (10) does not depend on the marginal distribution P X in the linear regression model.

Finally, (11) is related to the covariance matrix depth (Zhang, 2002; Chen et al., 2018; Paindaveine & Van Bever, 2017) .

All of the estimators (9), (10) and (11) are proved to achieve the minimax rate for the corresponding problems under Huber's -contamination model (Chen et al., 2018; Gao, 2017) .

DISPLAYFORM7 The connection to various depth functions shows the importance of TV-Learning in robust estimation.

However, it is well-known that depth-based estimators are very hard to compute BID1 van Kreveld et al., 1999; Rousseeuw & Struyf, 1998) , which limits their applications only for very low-dimensional problems.

On the other hand, the general f -GAN framework (4) has been successfully applied to learn complex distributions and images in practice (Goodfellow et al., 2014; Radford et al., 2015; Salimans et al., 2016) .

The major difference that gives the computational advantage to f -GAN is its flexibility in terms of designing the discriminator class T using neural networks compared with the pre-specified choice (6) in f -Learning.

While f -Learning provides a unified perspective in understanding various depth-based procedures in robust estimation, we can step back into the more general f -GAN for its computational advantages, and to design efficient computational strategies.

In this section, we focus on the problem of robust mean estimation under Huber's -contamination model.

Our goal is to reveal how the choice of the class of discriminators affects robustness and statistical optimality under the simplest possible setting.

That is, we have i.i.d.

observations X 1 , ..., X n ∼ (1 − )N (θ, I p ) + Q, and we need to estimate the unknown location θ ∈ R p with the contaminated data.

Our goal is to achieve the minimax rate p n ∨ 2 with respect to the squared 2 loss uniformly over all θ ∈ R p and all Q. DISPLAYFORM0 , where b is maximized out for visualization.

Samples are drawn from P = (1 − )N (1, 1) + N (10, 1) with = 0.2.

Left: a surface plot of F (η, w).

The solid curves are marginal functions for fixed η's: F (1, w) (red) and F (5, w) (blue), and the dash curves are marginal functions for fixed w's: F (η, −10) (orange) and F (η, 10) (green).

Right: a heatmap of F (η, w).

It is clear thatF (w) = F (η, w) has two local maxima for a given η, achieved at w = +∞ and w = −∞. In fact, the global maximum for F (w) has a phase transition from w = +∞ to w = −∞ as η grows.

For example, the maximum is achieved at w = +∞ when η = 1 (blue solid) and is achieved at w = −∞ when η = 5 (red solid).

Unfortunately, even if we initialize with η 0 = 1 and w 0 > 0, gradient ascents on η will only increase the value of η (green dash), and thus as long as the discriminator cannot reach the global maximizer, w will be stuck in the positive half space {w : w > 0} and further increase the value of η.

We start with the total variation GAN (TV-GAN) with f (x) = (x − 1) + in (4).

For the Gaussian location family, (4) can be written as DISPLAYFORM0 with T (x) = D(x) in (4).

Now we need to specify the class of discriminators D to solve the classification problem between N (η, I p ) and the empirical distribution 1 n n i=1 δ Xi .

One of the simplest discriminator classes is the logistic regression, DISPLAYFORM1 With FORMULA0 , the procedure (12) can be viewed as a smoothed version of TV-Learning (8).

To be specific, the sigmoid function sigmoid(w T x + b) tends to an indicator function as w → ∞, which leads to a procedure very similar to (9).

In fact, the class (13) is richer than the one used in (9), and thus (12) can be understood as the minimizer of a sharper variational lower bound than that of (9).

DISPLAYFORM2 DISPLAYFORM3 with probability at least 1 − e −C (p+n 2 ) uniformly over all θ ∈ R p and all Q. The constants C, C > 0 are universal.

Though TV-GAN can achieve the minimax rate

Given the intractable optimization property of TV-GAN, we next turn to Jensen-Shannon GAN (JS-GAN) with f (x) = x log x − (x + 1) log x+1 2 .

The estimator is defined by DISPLAYFORM0 with T (x) = log D(x) in (4).

This is exactly the original GAN (Goodfellow et al., 2014) specialized to the normal mean estimation problem.

The advantages of JS-GAN over other forms of GAN have been studied extensively in the literature (Lucic et al., 2017; Kurach et al., 2018) .Unlike TV-GAN, our experiment results show that FORMULA0 with the logistic regression discriminator class FORMULA0 is not robust to contamination.

However, if we replace (13) by a neural network class with one or more hidden layers, the estimator will be robust and will also work very well numerically.

To understand why and how the class of the discriminators affects the robustness property of JS-GAN, we introduce a new concept called restricted Jensen-Shannon divergence.

Let g : R p → R d be a function that maps a p-dimensional observation to a d-dimensional feature space.

The restricted Jensen-Shannon divergence between two probability distributions P and Q with respect to the feature g is defined as DISPLAYFORM1 In other words, P and Q are distinguished by a logistic regression classifier that uses the feature g(X).

It is easy to see that JS g (P, Q) is a variational lower bound of the original Jensen-Shannon divergence.

The key property of JS g (P, Q) is given by the following proposition.

Proposition 3.1.

Assume W is a convex set that contains an open neighborhood of 0.

Then, JS g (P, Q) = 0 if and only if E P g(X) = E Q g(X).The proposition asserts that JS g (·, ·) cannot distinguish P and Q if the feature g(X) has the same expected value under the two distributions.

This generalized moment matching effect has also been studied by Liu et al. (2017) for general f -GANs.

However, the linear discriminator class considered in Liu et al. FORMULA0 is parameterized in a different way compared with the discriminator class here.

When we apply Proposition 3.1 to robust mean estimation, the JS-GAN is trying to match the values of DISPLAYFORM2 for the feature g(X) used in the logistic regression classifier.

This explains what we observed in our numerical experiments.

A neural net without any hidden layer is equivalent to a logistic regression with a linear feature g(X) = (X T , 1) DISPLAYFORM3 , which implies that the sample mean is a global maximizer of (14).

On the other hand, a neural net with at least one hidden layers involves a nonlinear feature function g(X), which is the key that leads to the robustness of (14).We will show rigorously that a neural net with one hidden layer is sufficient to make (14) robust and optimal.

Consider the following class of discriminators, DISPLAYFORM4 The class (15) consists of two-layer neural network functions.

While the dimension of the input layer is p, the dimension of the hidden layer can be arbitrary, as long as the weights have a bounded 1 norm.

The nonlinear activation function σ(·) is allowed to take 1) indicator: DISPLAYFORM5 1+e −x , 3) ramp: σ(x) = max(min(x + 1/2, 1), 0).

Other bounded activation functions are also possible, but we do not exclusively list them.

The rectified linear unit (ReLU) will be studied in Appendix A. Theorem 3.2.

Consider the estimator θ defined by (14) with D specified by (15).

Assume DISPLAYFORM6 with probability at least 1 − e −C (p+n 2 ) uniformly over all θ ∈ R p and all Q. The constants C, C > 0 are universal.

An advantage of Tukey's median (9) is that it leads to optimal robust location estimation under general elliptical distributions such as Cauchy distribution whose mean does not exist.

In this section, we show that JS-GAN shares the same property.

A random vector X ∈ R p follows an elliptical distribution if it admits a representation DISPLAYFORM0 where U is uniformly distributed on the unit sphere {u ∈ R p : u = 1} and ξ ≥ 0 is a random variable independent of U that determines the shape of the elliptical distribution (Fang, 2017) .

The center and the scatter matrix is θ and Σ = AA T .For a unit vector v, let the density function of ξv T U be h. Note that h is independent of v because of the symmetry of U .

Then, there is a one-to-one relation between the distribution of ξ and h, and thus the triplet (θ, Σ, h) fully parametrizes an elliptical distribution.

Note that h and Σ = AA T are not identifiable, because ξA = (cξ)(c −1 A) for any c > 0.

Therefore, without loss of generality, we can restrict h to be a member of the following class DISPLAYFORM1 This makes the parametrization (θ, Σ, h) of an elliptical distribution fully identifiable, and we use EC(θ, Σ, h) to denote an elliptical distribution parametrized in this way.

The JS-GAN estimator is defined as DISPLAYFORM2 where E p (M ) is the set of all positive semi-definite matrix with spectral norm bounded by M .Theorem 4.1.

Consider the estimator θ defined above with D specified by (15).

DISPLAYFORM3 for some sufficiently small constant c > 0, and set DISPLAYFORM4 with probability at least 1 − e DISPLAYFORM5 Remark 4.1.

The result of Theorem 4.1 also holds (and is proved) under the strong contamination model (Diakonikolas et al., 2016a) .

That is, we have i.i.d.

observations X 1 , ..., X n ∼ P for some P satisfying TV(P, EC(θ, Σ, h)) ≤ .

See its proof in Appendix D.2.Note that Theorem 4.1 guarantees the same convergence rate as in the Gaussian case for all elliptical distributions.

This even includes multivariate Cauchy where mean does not exist.

Therefore, the location estimator (16) is fundamentally different from Diakonikolas et al. (2016a); Lai et al. (2016) , which is only designed for robust mean estimation.

We will show such a difference in our numerical results.

To achieve rate-optimality for robust location estimation under general elliptical distributions, the estimator FORMULA0 is different from (14) only in the generator class.

They share the same discriminator class (15).

This underlines an important principle for designing GAN estimators: the overall statistical complexity of the estimator is only determined by the discriminator class.

The estimator (16) also outputs ( Σ, h), but we do not claim any theoretical property for ( Σ, h) in this paper.

This will be systematically studied in a future project.

In this section, we give extensive numerical studies of robust mean estimation via GAN.

After introducing the implementation details in Section 5.1, we verify our theoretical results on minimax estimation with both TV-GAN and JS-GAN in Section 5.2.

Comparison with other methods on robust mean estimation in the literature is given in Section 5.3.

The effects of various network structures are studied in Section 5.4.

Adaptation to unknown covariance is studied in Section 5.5.

In all these cases, we assume i.i.d.

observations are drawn from (1 − )N (0 p , I p ) + Q with and Q to be specified.

Finally, adaptation to elliptical distributions is studied in Section 5.6.

We adopt the standard algorithmic framework of f -GANs (Nowozin et al., 2016) for the implementation of JS-GAN and TV-GAN for robust mean estimation.

In particular, the generator for mean estimation is G η (Z) = Z + η with Z ∼ N (0 p , I p ); the discriminator D is a multilayer perceptron (MLP), where each layer consisting of a linear map and a sigmoid activation function and the number of nodes will vary in different experiments to be specified below.

Details related to algorithms, tuning, critical hyper-parameters, structures of discriminator networks and other training tricks for stabilization and acceleration are discussed in Appendix B.1.

A PyTorch implementation is available at https://github.com/zhuwzh/Robust-GAN-Center.

We verify the minimax rates achieved by TV-GAN (Theorem 3.1) and JS-GAN (Theorem 3.2) via numerical experiments.

Two main scenarios we consider here are p/n < and p/n > , where in both cases, various types of contamination distributions Q are considered.

Specifically, the choice of contamination distributions Q includes N (µ * 1 p , I p ) with µ ranges in {0.2, 0.5, 1, 5}, N (0.5 * 1 p , Σ) and Cauchy(τ * 1 p ).

Details of the construction of the covariance matrix Σ is given in Appendix B.2.

The distribution Cauchy(τ * 1 p ) is obtained by combining p independent one-dimensional standard Cauchy with location parameter τ j = 0.5.

n = 1, 000, = 0.1 and p ranges from 10 to 100) and 1/ √ n (right: p = 50, = 0.1 and n ranges from 50 to 1, 000), respectively.

Net structure: One hidden layer with 20 hidden units (JS-GAN), zero hidden layer (TV-GAN).

The vertical bars indicate ± standard deviations.

The main experimental results are summarized in FIG5 , where the 2 error we present is the maximum error among all choices of Q, and detailed numerical results can be founded in Tables 7, 8 and 9 in Appendix.

We separately explore the relation between the error and one of , √ p and 1/ √ n with the other two parameters fixed.

The study of the relation between the 2 error and is in the regime p/n < so that dominates the minimax rate.

The scenario p/n > is considered in the study of the effects of √ p and 1/ √ n. As is shown in FIG5 , the errors are approximately linear against the corresponding parameters in all cases, which empirically verifies the conclusions of Theorem 3.1 and Theorem 3.2.

We perform additional experiments to compare with other methods including dimension halving (Lai et al., 2016) and iterative filtering (Diakonikolas et al., 2017) under various settings.

We emphasize that our method does not require any knowledge about the nuisance parameters such as the contamination proportion .

Tuning GAN is only a matter of optimization and one can tune parameters based on the objective function only.

Table 1 : Comparison of various robust mean estimation methods.

Net structure: One-hidden layer network with 20 hidden units when n = 50, 000 and 2 hidden units when n = 5, 000.

The number in each cell is the average of 2 error θ − θ with standard deviation in parenthesis estimated from 10 repeated experiments and the smallest error among four methods is highlighted in bold.

Table 1 shows the performances of JS-GAN, TV-GAN, dimension halving, and iterative filtering.

The network structure, for both JS-GAN and TV-GAN, has one hidden layer with 20 hidden units when the sample size is 50,000 and 2 hidden units when sample size is 5,000.

The critical hyper-parameters we apply is given in Appendix and it turns out that the choice of the hyper-parameter is robust against different models when the net structures are the same.

To summarize, our method outperforms other algorithms in most cases.

TV-GAN is good at cases when Q and N (0 p , I p ) are non-separable but fails when Q is far away from N (0 p , I p ) due to optimization issues discussed in Section 3.1 FIG0 ).

On the other hand, JS-GAN stably achieves the lowest error in separable cases and also shows competitive performances for non-separable ones.

DISPLAYFORM0

We further study the performance of JS-GAN with various structures of neural networks.

The main observation is tuning networks with one-hidden layer becomes tough as the dimension grows (e.g. p ≥ 200), while a deeper network can significantly refine the situation perhaps by improving the landscape.

Some experiment results are given in Table 2 .

On the other hand, one-hidden layer performs not worse than deeper networks when dimension is not very large (e.g. p ≤ 100).

More experiments are given in Appendix B.4.

Additional theoretical results for deep neural nets are given in Appendix A. Table 2 : Experiment results for JS-GAN using networks with different structures in high dimension.

Settings: = 0.2, p ∈ {200, 400} and n = 50, 000.

The robust mean estimator constructed through JS-GAN can be easily made adaptive to unknown covariance structure, which is a special case of (16).

We define DISPLAYFORM0 The estimator θ, as a result, is rate-optimal even when the true covariance matrix is not necessarily identity and is unknown (see Theorem 4.1).

Below, we demonstrate some numerical evidence of the optimality of θ as well as the error of Σ in Table 3 .Data generating process Network structure Table 3 : Numerical experiments for robust mean estimation with unknown covariance trained with 50, 000 samples.

The covariance matrices Σ 1 and Σ 2 are generated by the same way described in Appendix B.2.

DISPLAYFORM1

We consider the estimation of the location parameter θ in elliptical distribution EC(θ, Σ, h) by the JS-GAN defined in (16).

In particular, we study the case with i. DISPLAYFORM0 The density function of Cauchy(θ, Σ) is given by p(x; θ, Σ) ∝ |Σ| DISPLAYFORM1 Compared with Algorithm (1), the difference lies in the choice of the generator.

We consider the generator G 1 (ξ, U ) = g ω (ξ)U + θ, where g ω (ξ) is a non-negative neural network parametrized by ω and some random variable ξ.

The random vector U is sampled from the uniform distribution on {u ∈ R p : u = 1}. If the scatter matrix is unknown, we will use the generator G 2 (ξ, U ) = g ω (ξ)AU +θ, with AA T modeling the scatter matrix.

Table 4 shows the comparison with other methods.

Our method still works well under Cauchy distribution, while the performance of other methods that rely on moment conditions deteriorates in this setting.

In this section, we investigate the performance of discriminator classes of deep neural nets with the ReLU activation function.

Since our goal is to learn a p-dimensional mean vector, a deep neural network discriminator without any regularization will certainly lead to overfitting.

Therefore, it is crucial to design a network class with some appropriate regularizations.

Inspired by the work of Bartlett (1997); Bartlett & Mendelson FORMULA1 , we consider a network class with 1 regularizations on all layers except for the second last layer with an 2 regularization.

With G H 1 (B) = g(x) = ReLU(v T x) : v 1 ≤ B , a neural network class with l + 1 layers is defined as DISPLAYFORM0 Combining with the last sigmoid layer, we obtain the following discriminator class, DISPLAYFORM1 Note that all the activation functions are ReLU(·) except that we use sigmoid(·) in the last layer in the feature map g(·).

A theoretical guarantees of the class defined above is given by the following theorem.

DISPLAYFORM2 with probability at least 1 − e −C (p log p+n 2 ) uniformly over all θ ∈ R p such that θ ∞ ≤ √ log p and all Q.The theorem shows that JS-GAN with a deep ReLU network can achieve the error rate p log p n ∨ 2 with respect to the squared 2 loss.

The condition θ ∞ ≤ √ log p for the ReLU network can be easily satisfied with a simple preprocessing step.

We split the data into two halves, whose sizes are log n and n − log n, respectively.

Then, we calculate the coordinatewise median θ using the small half.

It is easy to show that θ − θ ∞ ≤ log p log n ∨ with high probability.

Then, for each X i from the second half, the conditional distribution of X i − θ given the DISPLAYFORM3 and thus we can apply the estimator (14) using the shifted data X i − θ from the second half.

The theoretical guarantee of Theorem A.1 will be DISPLAYFORM4 with high probability.

Hence, we can use θ + θ as the final estimator to achieve the same rate in Theorem A.1.On the other hand, our experiments show that this preprocessing step is not needed.

We believe that the assumption θ ∞ ≤ √ log p is a technical artifact in the analysis of the Rademacher complexity.

It can probably be dropped by a more careful analysis.

The implementation for JS-GAN is given in Algorithm 1, and a simple modification of the objective function leads to that of TV-GAN.

DISPLAYFORM0 , generator network G η (z) = z + η, learning rates γ d and γ g for the discriminator and the generator, batch size m, discriminator steps in each iteration K, total epochs T , average epochs T 0 .

Initialization: Initialize η with coordinatewise median of S. Initialize w with N (0, .05) independently on each element or Xavier (Glorot & Bengio, 2010) .1: for t = 1, . . .

, T do 2: DISPLAYFORM1 Sample mini-batch DISPLAYFORM2 end for 7: DISPLAYFORM3 η ← η − γ g g η 10: end for Return: The average estimate η over the last T 0 epochs.

Several important implementation details are discussed below.• How to tune parameters?

The choice of learning rates is crucial to the convergence rate, but the minimax game is hard to evaluate.

We propose a simple strategy to tune hyper-parameters including the learning rates.

Suppose we have estimators θ 1 , . . .

, θ M with corresponding discriminator networks D w1 ,. . .

, D w M .

Fixing η = θ, we further apply gradient descent to D w with a few more epochs (but not many in order to prevent overfitting, for example 10 epochs) and select the θ with the smallest value of the objective function (14) (JS-GAN) or (12) (TV-GAN).

We note that training discriminator and generator alternatively usually will not suffer from overfitting since the objective function for either the discriminator or the generator is always changing.

However, we must be careful about the overfitting issue when training the discriminator alone with a fixed η, and that is why we apply an early stopping strategy here.

Fortunately, the experiments show if the structures of networks are same (then of course, the dimensions of the inputs are same), the choices of hyper-parameters are robust to different models and we present the critical parameters in Table 5 to reproduce the experiment results in Table 1 and Table 2 .•

When to stop training?

Judging convergence is a difficult task in GAN trainings, since sometimes oscillation may occur.

In computer vision, people often use a task related measure and stop training once the requirement based on the measure is achieved.

In our experiments below, we simply use a sufficiently large T which works well, but it is still interesting to explore an efficient early stopping rule in the future work.• How to design the network structure?

Although Theorem 3.1 and Theorem 3.2 guarantee the minimax rates of TV-GAN without hidden layer and JS-GAN with one hidden layer, one may wonder whether deeper network structures will perform better.

From our preliminary experiments, TV-GAN with one hidden layer is significantly better than TV-GAN without any hidden layer.

Moreover, JS-GAN with deep network structures can significantly improve over shallow networks especially when the dimension is large (e.g. p ≥ 200).

For a network with one hidden layer, the choice of width may depend on the sample size.

If we only have 5,000 samples of 100 dimensions, two hidden units performs better than five hidden units, which performs better than twenty hidden units.

If we have 50,000 samples, networks with twenty hidden units perform the best.• How to stabilize and accelerate TV-GAN?

As we have discussed in Section 3.1, TV-GAN has a bad landscape when N (θ, I p ) and the contamination distribution Q are linearly separable (see FIG0 ).

An outlier removal step before training TV-GAN may be helpful.

Besides, spectral normalization (Miyato et al., 2018 ) is also worth trying since it can prevent the weight from going to infinity and thus can increase the chance to escape from bad saddle points.

To accelerate the optimization of TV-GAN, in all the numerical experiments below, we adopt a regularized version of TV-GAN inspired by Proposition 3.1.

Since a good feature extractor should match nonlinear moments of P = (1 − )N (θ, I p ) + Q and N (η, I p ), we use an additional regularization term that can accelerate training and sometimes even leads to better performances.

Specifically, let D(x) = sigmoid(w T Φ(x)) be the discriminator network with w being the weights of the output layer and Φ D (x) be the corresponding network after removing the output layer from D(x).

The quantity Φ D (x) is usually viewed as a feature extractor, which naturally leads to the following regularization term (Salimans et al., 2016; Mroueh et al., 2017) , defined as DISPLAYFORM4 where DISPLAYFORM5

We introduce the contamination distributions Q used in the experiments.

We first consider Q = N (µ, I p ) with µ ranges in {0.2, 0.5, 1, 5}. Note that the total variation distance between N (0 p , I p ) and N (µ, I p ) is of order 0 p − µ = µ .

We hope to use different levels of µ to test the algorithm and verify the error rate in the worst case.

Second, we consider Q = N (1.5 * 1 p , Σ) to be a Gaussian distribution with a non-trivial covariance matrix Σ. The covariance matrix is generated according to the following steps.

First generate a sparse precision matrix Γ = (γ ij ) with each entry γ ij = z ij * τ ij , i ≤ j, where z ij and τ ij are independently generated from Uniform(0.4, 0.8) and Bernoulli(0.1).

We then define γ ij = γ ji for all i > j andΓ = Γ + (| min eig(Γ)| + 0.05)I p to make the precision matrix symmetric and positive definite, where min eig(Γ) is the smallest eigenvalue of Γ. The covariance matrix is Σ =Γ −1 .

Finally, we consider Q to be a Cauchy distribution with independent component, and the jth component takes a standard Cauchy distribution with location parameter τ j = 0.5.

In Section 5.3, we compare GANs with the dimension halving (Lai et al., 2016) and iterative filtering (Diakonikolas et al., 2017).• Dimension Halving.

Experiments conducted are based on the code from https://github.com/ kal2000/AgnosticMeanAndCovarianceCode.

The only hyper-parameter is the threshold in the outlier removal step, and we take C = 2 as suggested in the file outRemSperical.m.• Iterative Filtering.

Experiments conducted are based on the code from https://github.com/ hoonose/robust-filter.

We assume is known and take other hyper-parameters as suggested in the file filterGaussianMean.m.

The experiments are conducted with i.i.d.

observations drawn from (1 − )N (0 p , I p ) + N (0.5 * 1 p , I p ) with = 0.2.

Table 6 summarizes results for p = 100, n ∈ {5000, 50000} and various network structures.

We observe that TV-GAN that uses neural nets with one hidden layer improves over the performance of that without any hidden layer.

This indicates that the landscape of TV-GAN might be improved by a more complicated network structure.

However, adding one more layer does not improve the results.

For JS-GAN, we omit the results without hidden layer because of its lack of robustness (Proposition 3.1).

Deeper networks sometimes improve over shallow networks, but this is not always true.

We also observe that the optimal choice of the width of the hidden layer depends on the sample size.

Table 8 : Scenario II-a: p/n > .

Setting: n = 1, 000, = 0.1, and p from 10 to 100.

Other details are the same as above.

Table 9 : Scenario II-b: p/n > .

Setting: p = 50, = 0.1, and n from 50 to 1, 000.

Other details are the same as above.

Q Net n = 50 n = 100 n = 200 n = 500 n = 1000 In the first example, consider DISPLAYFORM0 DISPLAYFORM1 In other words, Q is the class of Gaussian location family, and Q η is taken to be a subset in a local neighborhood of N (η, I p ).

Then, with Q = N (η, I p ) and Q = N ( η, I p ), the event q(X)/q(X) ≥ 1 is equivalent to X − η 2 ≤ X − η 2 .

Since η − η ≤ r, we can write η = η + ru for some r ∈ R and u ∈ R p that satisfy 0 ≤ r ≤ r and u = 1.

Then, (8) becomes DISPLAYFORM2 Letting r → 0, we obtain (9), the exact formula of Tukey's median.

The next example is a linear model y|X ∼ N (X T θ, 1).

Consider the following classes DISPLAYFORM3 Here, P y,X stands for the joint distribution of y and X. The two classes Q and Q share the same marginal distribution P X and the conditional distributions are specified by N (X T η, 1) and N (X T η, 1), respectively.

Follow the same derivation of Tukey's median, let r → 0, and we obtain the exact formula of regression depth (10).

It is worth noting that the derivation of (10) does not depend on the marginal distribution P X .The last example is on covariance/scatter matrix estimation.

For this task, we set Q = {N (0, Γ) : Γ ∈ E p }, where E p is the class of all p × p covariance matrices.

Inspired by the derivations of Tukey depth and regression depth, it is tempting to choose Q in the neighborhood of N (0, Γ).

However, a native choice would lead to a definition that is not even Fisher consistent.

We propose a rank-one neighborhood, given by DISPLAYFORM4 Then, a direct calculation gives DISPLAYFORM5 Since lim r→0 DISPLAYFORM6 , depending on whether r tends to zero from left or from right.

Therefore, with the above Q and Q Γ , (8) becomes (11) under the limit r → 0.

Even though the definition of (19) is given by a rank-one neighborhood of the inverse covariance matrix, the formula (11) can also be derived with DISPLAYFORM7 T by applying the Sherman-Morrison formula.

A similar formula to (11) in the literature is given by dN (0,Γ) (X) ≥ 1 , a special case of (6), the formula (21) can be derived directly from TV-GAN with discriminators in the form of I dN (0,β Γ) dN (0,βΓ) (X) ≥ 1 by following a similar rank-one neighborhood argument.

This completes the derivation of Proposition 2.1.

DISPLAYFORM8 To prove Proposition 3.1, we define F (w) = E P log sigmoid(w T g(X)) + E Q log(1 − sigmoid(w T g(X))) + log 4, so that JS g (P, Q) = max w∈W F (w).

The gradient and Hessian of F (w) are given by DISPLAYFORM9 1 + e w T g(X) g(X), DISPLAYFORM10 Therefore, F (w) is concave in w, and max w∈W F (w) is a convex optimization with a convex W. Suppose JS g (P, Q) = 0.

Then max w∈W F (w) = 0 = F (0), which implies ∇F (0) = 0, and thus we have E P g(X) = E Q g(X).

Now suppose E P g(X) = E Q g(X), which is equivalent to ∇F (0) = 0.

Therefore, w = 0 is a stationary point of a concave function, and we have JS g (P, Q) = max w∈W F (w) = F (0) = 0.

In this section, we present proofs of all main theorems in the paper.

We first establish some useful lemmas in Section D.1, and the the proofs of main theorems will be given in Section D.2.

Lemma D.1.

Given i.i.d.

observations X 1 , ..., X n ∼ P and the function class D defined in (13), we have for any δ > 0, DISPLAYFORM0 with probability at least 1 − δ for some universal constant C > 0.

DISPLAYFORM1 .

It is clear that f (X 1 , ..., X n ) satisfies the bounded difference condition.

By McDiarmid's inequality (McDiarmid, 1989) , we have DISPLAYFORM2 , with probability at least 1 − δ.

Using a standard symmetrization technique (Pollard, 2012), we obtain the following bound that involves Rademacher complexity, DISPLAYFORM3 where 1 , ..., n are independent Rademacher random variables.

The Rademacher complexity can be bounded by Dudley's integral entropy bound, which gives DISPLAYFORM4 where N (δ, D, · n ) is the δ-covering number of D with respect to the empirical 2 distance f − Van Der Vaart & Wellner (1996) ).

This leads to the bound DISPLAYFORM5 DISPLAYFORM6 n , which gives the desired result.

Lemma D.2.

Given i.i.d.

observations X 1 , ..., X n ∼ P, and the function class D defined in (15), we have for any δ > 0, DISPLAYFORM7 with probability at least 1 − δ for some universal constant C > 0.

DISPLAYFORM8 Therefore, by McDiarmid's inequality (McDiarmid, 1989) , we have DISPLAYFORM9 with probability at least 1 − δ.

By the same argument of FORMULA1 , it is sufficient to bound the Rademacher complexity E sup D∈D 1 n n i=1 i log(2D(X i )) .

Since the function ψ(x) = log(2sigmoid(x)) has Lipschitz constant 1 and satisfies ψ(0) = 0, we have p log p n + log(1/δ) n , with probability at least 1 − δ for some universal constants C > 0.Proof.

Write f (X 1 , ..., X n ) = sup D∈F H L (κ,τ,B)1 n n i=1 log D(X i ) − E log D(X) .

Then, the inequality (23) holds with probability at least 1 − δ.

It is sufficient to analyze the Rademacher complexity.

Using the fact that the function log(2sigmoid(x)) is Lipschitz and Hölder's inequality, we have 1 n DISPLAYFORM10 Now we use the notation Z i = X i − θ ∼ N (0, I p ) for i = 1, ..., n. We bound E sup g∈G H L−1 (B) 1 n n i=1 i g(Z i + θ) by induction.

Since DISPLAYFORM11 This leads to the desired result under the conditions on τ and θ ∞ .

Proof of Theorem 3.1.

We first introduce some notations.

Define F (P, η) = max w,b F w,b (P, η), where With this definition, we have θ = argmin η F (P n , η), where we use P n for the empirical distribution 1 n n i=1 δ Xi .

We shorthand N (η, I p ) by P η , and then F (P θ , θ) ≤ F ((1 − )P θ + Q, θ) + (24)≤ F (P n , θ) + + C p n + log(1/δ) n (25) ≤ F (P n , θ) + + C p n + log(1/δ) n (26) ≤ F ((1 − )P θ + Q, θ) + + 2C p n + log(1/δ) n (27) ≤ F (P θ , θ) + 2 + 2C p n + log(1/δ) n (28) = 2 + 2C p n + log(1/δ) n .With probability at least 1 − δ, the above inequalities hold.

We will explain each inequality.

Since The inequality FORMULA1 is a direct consequence of the definition of θ.

Finally, it is easy to see that F (P θ , θ) = 0, which gives (29).

In summary, we have derived that with probability at least 1 − δ, F w,b (P θ , θ) ≤ 2 + 2C p n + log(1/δ) n , for all w ∈ R p and b ∈ R. For any u ∈ R p such that u = 1, we take w = u and b = −u T θ, and we have DISPLAYFORM0 where f (t) = 1 1+e z+t φ(z)dz, with φ(·) being the probability density function of N (0, 1).

It is not hard to see that as long as |f (t) − f (0)| ≤ c for some sufficiently small constant c > 0, then |f (t) − f (0)| ≥ c |t| for some constant c > 0.

This implies DISPLAYFORM1 with probability at least 1 − δ.

The proof is complete.

Proof of Theorem 3.2.

We continue to use P η to denote N (η, I p ).

Define DISPLAYFORM2 ≤ F (P n , θ) + 2κ + Cκ p n + log(1/δ) n (31) ≤ F (P n , θ) + 2κ + Cκ p n + log(1/δ) n (32) ≤ F ((1 − )P θ + Q, θ) + 2κ + 2Cκ p n + log(1/δ) n (33) ≤ F (P θ , θ) + 4κ + 2Cκ p n + log(1/δ) n (34) = 4κ + 2Cκ p n + log(1/δ) n .where F w,u,b (P, (η, Γ, g)) = E P log D(X) + E EC(η,Γ,g) log (1 − D(X)) + log 4, with D(x) = sigmoid j≥1 w j σ(u T j x + b j ) .

Let P be the data generating process that satisfies TV(P, P θ,Σ,h ) ≤ , and then there exist probability distributions Q 1 and Q 2 , such that P + Q 1 = P θ,Σ,h + Q 2 .The explicit construction of Q 1 , Q 2 is given in the proof of Theorem 5.1 of Chen et al. (2018) .

This implies that |F (P, (η, Γ, g)) − F (P θ,Σ,h , (η, Γ, g))|≤ sup |E Q2 log(2D(X)) − E Q1 log(2D(X))| ≤ 2κ .

<|TLDR|>

@highlight

GANs are shown to provide us a new effective robust mean estimate against agnostic contaminations with both statistical optimality and practical tractability.
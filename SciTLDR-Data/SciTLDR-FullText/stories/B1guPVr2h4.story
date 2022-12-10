Recent empirical results on over-parameterized deep networks are marked by a striking absence of the classic U-shaped test error curve: test error keeps decreasing in wider networks.

Researchers are actively working on bridging this discrepancy by proposing better complexity measures.

Instead, we directly measure prediction bias and variance for four classification and regression tasks on modern deep networks.

We find that both bias and variance can decrease as the number of parameters grows.

Qualitatively, the phenomenon persists over a number of gradient-based optimizers.

To better understand the role of optimization, we decompose the total variance into variance due to training set sampling and variance due to initialization.

Variance due to initialization is significant in the under-parameterized regime.

In the over-parameterized regime, total variance is much lower and dominated by variance due to sampling.

We provide theoretical analysis in a simplified setting that is consistent with our empirical findings.

Despite a few notable exceptions, such as boosting (Schapire, 1990; BID13 BID5 , the dogma in machine learning has been: "the price to pay for achieving low bias is high variance" BID14 .

This balance between underfitting (high bias) and overfitting (high variance) is commonly known as the biasvariance tradeoff FIG0 .

Statistical learning theory (Vapnik, 1998) identifying a notion of model capacity, understood as the main parameter controlling this tradeoff.

Complex (high capacity) models achieve low prediction bias at the expense of high variance.

In their landmark work that highlighted this dilemma, BID14 suggest that bias decreases and variance increases with network size.

However, there is a growing amount of empirical evidence that wider networks generalize better than their smaller counterparts (Neyshabur et al., 2015; Zagoruyko & Komodakis, 2016; Novak et al., 2018; BID8 BID2 Spigler et al., 2018; Liang et al., 2017; BID6 .

In those cases the U-shaped test error curve is not observed.

Researchers have identified classic measures of complexity as a culprit.

The idea is that, once we have identified the right complexity measure, we will again be able to observe this fundamental tradeoff.

We bypass this important, ongoing discussion by measuring prediction bias and variance directly-something that has not been done in related literature since BID14 , to the best of our knowledge.

These measurements allow us to reason directly about the existence of a tradeoff with respect to network width.

We find evidence that both bias and variance can decrease at the same time as network width increases in common classification and regression settings with deep networks.

We observe this qualitative behavior with a number of gradient-based optimizers.

In order to get a closer look at the role of optimization and sampling, we propose a simple decomposition of total prediction variance.

We use the law of total variance to get a term that corresponds to variance due to training set sampling and another that corresponds to variance due to initialization.

Variance due to initialization is significant in the under-parameterized regime and monotonically decreases with width in the over-parameterized regime.

There, total variance is much lower and dominated by variance due to sampling (Fig. 2) .We provide theoretical analysis, consistent with our empirical findings, in simplified analysis settings: i) prediction variance does not grow arbitrarily in linear models; ii) variance due to initialization diminishes in deep networks under strong assumptions.

On the left is an illustration of the common intuition for the bias-variance tradeoff BID12 .

We find that variance decreases along with bias when increasing network width (right).

These results seem to contradict the traditional intuition.

In concurrent work, Spigler et al. (2018) ; BID2 point out that generalization error decreases with capacity in the over-parameterized setting, with a sharp transition between the under-parameterized and the overparameterized settings.

While this transition can also be seen as the early hump in variance we observe in some of our graphs, we mostly focus on the over-parameterized setting.

Additionally, our work is unique in that we explicitly analyze and experimentally measure the quantities of bias and variance.

We consider the typical supervised learning task of predicting an output y ∈ Y from an input x ∈ X , where the pairs (x, y) are drawn from some unknown joint distribution, D. The learning problem consists of inferring a function h S : X → Y from a finite training dataset S of m i.i.d.

samples from D. The quality of a predictor h can quantified by the expected error, DISPLAYFORM0 In this paper, predictors h θ are parameterized by the weights θ ∈ R N of deep neural networks.

We will consider the average performance over possible training sets (denoted by the random variable S) of size m. This is the same quantity BID14 consider.

While S is the only random quantity focused on in traditional bias-variance decomposition, we also focus on randomness coming from optimization.

We denote the random variable for optimization randomness (e.g. initialization) by I. 1 1 We focus on randomness from initialization and do not focus on randomness from stochastic mini-batching because we found Formally, given a fixed training set S and fixed optimization randomness I, the learning algorithm A produces θ = A(S, I).

Randomness in initialization translates to randomness in A(S, ·).

Given a fixed training set, we encode the randomness due to I in a conditional distribution p(θ|S); marginalizing over the training set S of size m gives a marginal distribution p(θ) = E S p(θ|S) on the weights learned by A from m samples.

In this context, the average performance for the learning algorithm using training sets of size m can be expressed in the following ways: DISPLAYFORM1

We briefly recall the standard bias-variance decomposition in the case of squared-loss.

We work in the context of classification, where each class k ∈ {1 · · · K} is represented by a one-hot vector in R K .

The predictor outputs a score or probability vector in R K .

In this context, the average performance in Eq. (1) decomposes into three sources of error BID14 : DISPLAYFORM0 The first term is an intrinsic error term independent of the predictor; the second is a bias term: DISPLAYFORM1 whereȳ(x) denotes the expectation E[y|x] of y given x. The third term is the expected variance of the output predictions: DISPLAYFORM2 the phenomenon of decreasing variance with width persists when using batch gradient descent (Section 3.1, Appendix B.6).

DISPLAYFORM3 where the expectation over θ can be done as in Eq.(1).

Finally, in the set-up of Section 2.1, the sources of variance are the choice of training set S and the choice of initialization I (encoded into the conditional p(·|S)).

By the law of total variance, we then have the further decomposition: DISPLAYFORM4 We call the first term variance due to initialization and the second term variance due to sampling throughout the paper.

Note that risks computed with classification losses (e.g cross-entropy or 0-1 loss) do not have such a clean biasvariance decomposition BID7 BID18 .

However, it is natural to expect that bias and variance are useful indicators of the performance of the models.

In fact, we show the classification risk can be bounded as 4 times the regression risk in Appendix D.4.

In this section, we experimentally study how variance of fully connected single hidden layer networks varies with width.

We provide evidence against BID14 's claim that "bias falls and variance increases with the number of hidden units."

Experimental details are specified in Appendix A.

In FIG0 , we see that variance decreases (along with bias) as network width increases on MNIST.

Similarly, we also observe this phenomenon in CIFAR10 and SVHN ( Fig. 2 and Appendices B.1 and B.2).

In addition to these classification tasks, we see this in a sinusoid regression task ( FIG2 and Appendix B.7).

In each of these tasks, the same hyperparameters are used across all widths.

Decreasing the size of the dataset can only increase variance.

To study the robustness of the above observation, we decrease the size of the MNIST training set to just 100 examples.

In this small data setting, somewhat surprisingly, we still see that both bias and variance decrease with width ( FIG2 .

The test error behaves similarly FIG2 ).

Test error trends for all of our experiments follow the biasvariance trends (Appendix B), as the bias-variance decomposition would suggest.

Because performance is more sensitive to step size in the small data setting, the step size for each network size is tuned using a validation set (see Appendix B.4 for step sizes).

This protocol allows the bias to decrease with width, indicating effective capacity is, indeed, increasing while variance is decreasing (see Appendix A.1 for more discussion).To see how dependent this phenomenon is on SGD, we also run these experiments using batch gradient descent and PyTorch's version of LBFGS.

Interestingly, we find a decreasing variance trend with those optimizers as well.

These experiments are included in Appendix B.6.

In order to better understand this variance phenomenon in neural networks, we separate the variance due to sampling from the variance due to initialization, according to the law of total variance (Equation 3).

Contrary to what traditional bias-variance tradeoff intuition would suggest, we find variance due to sampling levels with increasingly large width (Fig. 2) .

Furthermore, we find that variance due to initialization decreases with width, causing the joint variance to decrease with width ( Fig. 2) .A body of recent work has provided evidence that over- parameterization (in width) helps gradient descent optimize to global minima in neural networks BID9 BID8 Soltanolkotabi et al., 2017; Livni et al., 2014; Zhang et al., 2018) .

Always reaching a global minimum implies low variance due to initialization on the training set.

Our observation of decreasing variance on the test set shows that the over-parameterization (in width) effect on optimization seems to extend to generalization, on the data sets we consider.

Our empirical results demonstrate that in the practical setting, variance due to initialization decreases with network width while variance due to sampling levels off.

Here, we take inspiration from linear models (Hastie et al., 2009, Section 7.

3) to provide arguments for the behavior of variance in increasingly wide neural networks.

In overparameterized linear models, variance does not grow with the number of parameters.

This is due to the fact that, all learning occurs in rowspace(X) of the design matrix X (no learning in nullspace(X)), and the dimension of the solution space, r = rank(X), is independent of N .

For a complete walk-through of this, see Appendix C. We formalize this in Proposition 1 there.

We will illustrate our arguments in the following simplified setting, where M, M ⊥ , and d(N ) are the more general analogs of rowspace(X), nullspace(X), and r (respectively):Setting.

Let N be the dimension of the parameter space.

The prediction for a fixed example x, given by a trained network parameterized by θ depends on:(i) a subspace of the parameter space, M ∈ R N with relatively small dimension, d(N ), which depends only on the learning task.(ii) parameter components corresponding to directions or- DISPLAYFORM0 , and is essentially irrelevant to the learning task.

We can write the parameter vector as a sum of these two components θ = θ M + θ M ⊥ .

We will further make the following assumptions:Assumption 1 The optimization of the loss function is invariant with respect to θ M⊥ .Assumption 2 Regardless of initialization, the optimization method consistently yields a solution with the same θ M component, (i.e. the same vector when projected onto M).We provide a short discussion on these assumptions in Appendix E. Given the above assumptions, the following result, proved in Appendix D.3, shows that the variance from initialization vanishes as we increase N .Theorem 1 (Decay of variance due to initialization).

For a fixed data set and parameters initialized as θ 0 ∼ N (0, 1 N I), the variance of the prediction satisfies the inequality, DISPLAYFORM1 where L is the Lipschitz constant of the prediction with respect to θ, and for some universal constant C > O.This result guarantees that the variance from initialization decreases to zero as N increases, provided the Lipschitz constant L grows more slowly than the square root of dimension, L = o( √ N ).

First, we provide evidence against BID14 's claim that "the price to pay for achieving low bias is high variance," finding that both bias and variance decrease with width.

Second, we find variance due to sampling (analog of regular variance in simple settings) does not appear to be dependent on width, once sufficiently over-parameterized.

Third, variance due to initialization decreases with width.

We see further theoretical treatment of variance as a fruitful direction for better understanding complexity and generalization abilities of neural networks.

We run experiments on different datasets: MNIST, CIFAR10, SVHN, small MNIST, and a sinusoid regression task.

Averages over data samples are performed by taking the training set S and creating 50 bootstrap replicate training sets S by sampling with replacement from S. We train 50 different neural networks for each hidden layer size using these different training sets.

Then, we estimate E bias 2 and E variance as in Section 2.2, where the population expectation E x is estimated with an average over the test set.

To estimate the two terms from the law of total variance (Equation 3), we use 10 random seeds for the outer expectation and 10 for the inner expectation, resulting in a total of 100 neural networks for each hidden layer size.

Furthermore, we compute 99% confidence intervals for our bias and variance estimates using the bootstrap (Efron, 1979) .The networks are trained using SGD with momentum and generally run for long after 100% training set accuracy is reached (e.g. 500 epochs for full data MNIST and 10000 epochs for small data MNIST).

The overall trends we find are robust to how long the networks are trained after the training error converges.

To make our study as general as possible, we consider networks without regularization bells and whistles such as weight decay, dropout, or data augmentation, which Zhang et al. FORMULA2 found to not be necessary for good generalization.

Hyperparameters: In the full data experiments (all but small MNIST), the same step size is used for all networks for a given dataset (0.1 for MNIST, 0.005 for CIFAR10, and 0.005 for SVHN).

The momentum hyperparameter is always set to 0.9.

In the small data MNIST experiment, the is tuned, using a validation set, for each width.

The training for tuning is stopped after 1000 epochs, whereas the training for the final models is stopped after 10000 epochs.

The chosen step sizes can be found in Appendix B.4.

Because performance is more sensitive to step size in the small data setting, the step size for each network size is tuned using a validation set (see Appendix B.4 for step sizes).Note that because we see decreasing bias with width, effective capacity is, indeed, increasing while variance is decreasing.

One control that motivates the experimental design choice of optimal step size is that it leads to the conventional decreasing bias trend FIG2 that indicates increasing effective capacity.

In fact, in the corresponding experiment where step size is the same 0.01 for all network sizes, we do not see monotonically decreasing bias (Appendix B.5).This sensitivity to step size in the small data setting is evidence that we are testing the limits of our hypothesis.

By looking at the small data setting, we are able to test our hypothesis when the ratio of size of network to dataset size is quite large, and we still find this decreasing trend in variance FIG2 .

Note that the U curve shown in FIG0 when we do not tune the step size is explained by the fact that the constant step chosen is a "good" step size for some networks and "bad" for others.

Results from Keskar et al. FORMULA2 and Smith et al. (2018) show that a step size that corresponds well to the noise structure in SGD is important for achieving good test set accuracy.

Because our networks are different sizes, their stochastic optimization process will have a different landscape and noise structure.

By tuning the step size, we are making the experimental design choice to keep optimality of step size constant across networks, rather than keeping step size constant across networks.

To us, choosing this control makes much more sense than choosing to control for step size.

B.6.

Other optimizers for width experiment on small data mnist

In this section, we review the classic result that the variance of a linear model grows with the number of parameters (Hastie et al., 2009, Section 7.

3) and point out that variance behaves differently in the over-parameterized setting.

We consider least-squares linear regression in a standard setting which assumes a noisy linear mapping y = θ T x + between input feature vectors x ∈ R N and real outputs, where denotes the noise random variable with E[ ] = 0 and Var( ) = σ 2 .

In this context, the over-parameterized setting is when the dimension N of the input space is larger than the number m of examples.

Let X denote the m×N design matrix whose i th row is the training point x T i , let Y denote the corresponding labels, and let Σ = X T X denote the empirical covariance matrix.

We consider the "fixed design" setting where X is fixed, so all of the randomness due to data sampling comes solely from .

A learns weightsθ from (X, Y ), either by a closed-form solution or by gradient descent, using a standard initialization θ 0 ∼ N (0, 1 N I).

The predictor makes a prediction on x ∼ D: h(x) =θ T x. Then, the quantity we care about is E x Var(h(x)).

The case where N ≤ m is standard: if X has maximal rank, Σ is invertible; the solution is independent of the initialization and given byθ = Σ −1 X T Y .

All of the variance is a result of randomness in the noise .

For a fixed x, DISPLAYFORM0 This grows with the number of parameters N .

For example, taking the expected value over the empirical distribution,p, of the sample, we recover that the variance grows with N : DISPLAYFORM1 We provide a reproduction of the proofs in Appendix D.1.

The over-parameterized case where N > m is more interesting: even if X has maximal rank, Σ is not invertible.

This leads to a subspace of solutions, but gradient descent yields a unique solution from updates that belong to the span of the training points x i (row space of X) (LeCun et al., 1991) , which is of dimension r = rank(X) = rank(Σ).

Correspondingly, no learning occurs in the null space of X, which is of dimension N − r. Therefore, gradient descent yields the solution that is closest to initialization:θ = P ⊥ (θ 0 ) + Σ + X T Y , where P ⊥ projects onto the null space of X and + denotes the Moore-Penrose inverse.

The variance has two contributions: one due to initialization and one due to sampling (here, the noise ), as in Eq. (3).

These are made explicit in Proposition 1.

Proposition 1 (Variance in over-parameterized linear models).

Consider the over-parameterized setting where N > m. For a fixed x, the variance decomposition of Eq. (3) yields DISPLAYFORM0 This does not grow with the number of parameters N .

In fact, because Σ −1 is replaced with Σ + , the variance scales as the dimension of the data (i.e the rank of X), as opposed to the number of parameters.

For example, taking the expected value over the empirical distribution,p, of the sample, we obtain DISPLAYFORM1 where r = rank(X).

We provide the proofs for over-parameterized linear models in Appendix D.2.

Here, we reproduce the classic result that variance grows with the number of parameters in a linear model.

This result can be found in BID17 's book, and a similar proof can be found in Gonzalez (2016)'s lecture slides.

Proof.

For a fixed x, we have h(x) = x Tθ .

Takingθ = Σ −1 X T Y to be the gradient descent solution, and using Y = Xθ + , we obtain: DISPLAYFORM0 , and the variance is, DISPLAYFORM1 Taking the expected value over the empirical distribution,p, of the sample, we find an explicit increasing dependence on N : DISPLAYFORM2

Here, we produce a variation on what was done in Appendix D.1 to show that variance does not grow with the number of parameters in over-parameterized linear models.

Recall that we are considering the setting where N > m, where N is the number of parameters and m is the number of training examples.

Proof.

By the law of total variance, DISPLAYFORM0 Here have h(x) = x Tθ , whereθ the gradient descent solutionθ = P ⊥ (θ 0 ) + Σ + X T Y , and θ 0 ∼ N (0, DISPLAYFORM1 Taking the expected value over the empirical distribution,p, of the sample, we find an explicit dependence on r = rank(X), not N : DISPLAYFORM2 = σ 2 r m where I + r denotes the diagonal matrix with 1 for the first r diagonal elements and 0 for the remaining N − r elements.

First we state some known concentration results (Ledoux, 2001 ) that we will use in the proof.

Lemma 1 (Levy).

Let h : S n R → R be a function on the n-dimensional Euclidean sphere of radius R, with Lipschitz constant L; and θ ∈ S n R chosen uniformly at random for the normalized measure.

Then DISPLAYFORM0 for some universal constant C > 0.Uniform measures on high dimensional spheres approximate Gaussian distributions (Ledoux, 2001) .

Using this, Levy's lemma yields an analogous concentration inequality for functions of Gaussian variables: Lemma 2 (Gaussian concentration).

Let h : R n → R be a function on the Euclidean space R n , with Lipschitz constant L; and θ ∼ N (0, σI n ) sampled from an isotropic n-dimensional Gaussian.

Then: DISPLAYFORM1 for some universal constant C > 0.Note that in the Gaussian case, the bound is dimension free.

In turn, concentration inequalities give variance bounds for functions of random variables.

Then Var(h) = Var(g) and DISPLAYFORM0 Now swapping expectation and integral (by Fubini theorem), and by using the identity E1 |g|>t = P(|g| > t), we obtain DISPLAYFORM1 We are now ready to prove Theorem 1.

We first recall our assumptions:Assumption 1.

The optimization of the loss function is invariant with respect to θ M⊥ .Assumption 2.

Along M, optimization yields solutions independently of the initialization θ 0 .We add the following assumptions.

Assumption 3.

The prediction h θ (x) is L-Lipschitz with respect to θ M⊥ .Assumption 4.

The network parameters are initialized as DISPLAYFORM2 We first prove that the Gaussian concentration theorem translates into concentration of predictions in the setting of ??

.Theorem 2 (Concentration of predictions).

Consider the setting of ??

and Assumptions 1 and 4.

Let θ denote the parameters at the end of the learning process.

Then, for a fixed data set, S we get concentration of the prediction, under initialization randomness, DISPLAYFORM3 for some universal constant C > 0.Proof.

In our setting, the parameters at the end of learning can be expressed as DISPLAYFORM4 where θ * M is independent of the initialization θ 0 .

To simplify notation, we will assume that, at least locally around θ * M , M is spanned by the first d(N ) standard basis vectors, and M ⊥ by the remaining N − d(N ).

This will allow us, from now on, to use the same variable names for θ M and θ M ⊥ to denote their lower-dimensional representations of dimension d(N ) and N − d(N ) respectively.

More generally, we can assume that there is a mapping from θ M and θ M ⊥ to those lower-dimensional representations.

From Assumptions 1 and 4 we get DISPLAYFORM5 is L-Lipschitz.

Then, by the Gaussian concentration theorem we get, DISPLAYFORM6 The result of Theorem 1 immediately follows from Theorem 2 and Corollary 1, with σ 2 = 1/N : DISPLAYFORM7 Provided the Lipschitz constant L of the prediction grows more slowly than the square of dimension, L = o( √ N ), we conclude that the variance vanishes to zero as N grows.

In this section we give a bound on classification risk R classif in terms of the regression risk R reg .Notation.

Our classifier defines a map h : X → R k , which outputs probability vectors h(x) ∈ R k , with DISPLAYFORM0 where I(a) = 1 if predicate a is true and 0 otherwise.

Given trained predictors h S indexed by training dataset S, the classification and regression risks are given by, DISPLAYFORM1 where Y denotes the one-hot vector representation of the class y. Proposition 2.

The classification risk is bounded by four times the regression risk, R classif ≤ 4R reg .Proof.

First note that, if h(x) ∈ R k is a probability vector, then DISPLAYFORM2 By taking the expectation over x, y, we obtain the inequality L(h) ≤ L(h) where DISPLAYFORM3 We then have, DISPLAYFORM4 where the last inequality follows from Markov's inequality.

We made strong assumptions, but there is some support for them in the literature.

The existence of a subspace M ⊥ in which no learning occurs was also conjectured by BID0 and shown to hold in linear neural networks under a simplifying assumption that decouples the dynamics of the weights in different layers.

Li et al. (2018) empirically showed the existence of a critical number d(N ) = d of relevant parameters for a given learning task, independent of the size of the model.

Sagun et al. (2017) showed that the spectrum of the Hessian for over-parameterized networks splits into (i) a bulk centered near zero and (ii) a small number of large eigenvalues; and Gur-Ari et al. FORMULA2 recently gave evidence that the small subspace spanned by the Hessian's top eigenvectors is preserved over long periods of training.

These results suggest that learning occurs mainly in a small number of directions.

The problem with classical complexity measures is that they do not take into account optimization and have no notion of what will actually be learned.

BID1 Section 1) define a notion of an effective hypothesis class to take into account what functions are possible to be learned by the learning algorithm.

However, this still has the problem of not taking into account what hypotheses are likely to be learned.

To take into account the probabilistic nature of learning, we define the -hypothesis class for a data distribution D and learning algorithm A, that contains the hypotheses which are at least -likely for some > 0: DISPLAYFORM0 where S is a training set drawn from D m , h(A, S) is a random variable drawn from the distribution over learned functions induced by D and the randomness in A; p is the corresponding density.

Thinking about a model's -hypothesis class can lead to drastically different intuitions for the complexity of a model and its variance FIG0 .

This is at the core of the intuition for why the traditional view of bias-variance as a tradeoff does not hold in all cases.

"Neural Networks and the Bias/Variance Dilemma" from BID14 : "How big a network should we employ?

A small network, with say one hidden unit, is likely to be biased, since the repertoire of available functions spanned by f (x; w) over allowable weights will in this case be quite limited.

If the true regression is poorly approximated within this class, there will necessarily be a substantial bias.

On the other hand, if we overparameterize, via a large number of hidden units and associated weights, then the bias will be reduced (indeed, with enough weights and hidden units, the network will interpolate the data), but there is then the danger of a significant variance contribution to the mean-squared error.

(This may actually be mitigated by incomplete convergence of the minimization algorithm, as we shall see in Section 3.5.5.)" "An Overview of Statistical Learning Theory" from (Vapnik, 1999): "To avoid over fitting (to get a small confidence interval) one has to construct networks with small VC-dimension.""Stability and Generalization" from BID4 : "It has long been known that when trying to estimate an unknown function from data, one needs to find a tradeoff between bias and variance.

Indeed, on one hand, it is natural to use the largest model in order to be able to approximate any function, while on the other hand, if the model is too large, then the estimation of the best function in the model will be harder given a restricted amount of data." Footnote: "We deliberately do not provide a precise definition of bias and variance and resort to common intuition about these notions."Pattern Recognition and Machine Learning from BID3 : "Our goal is to minimize the expected loss, which we have decomposed into the sum of a (squared) bias, a variance, and a constant noise term.

As we shall see, there is a tradeoff between bias and variance, with very flexible models having low bias and high variance, and relatively rigid models having high bias and low variance.""Understanding the Bias-Variance Tradeoff" from Fortmann-Roe (2012): "At its root, dealing with bias and variance is really about dealing with over-and under-fitting.

Bias is reduced and variance is increased in relation to model complexity.

As more and more parameters are added to a model, the complexity of the model rises and variance becomes our primary concern while bias steadily falls.

For example, as more polynomial terms are added to a linear regression, the greater the resulting model's complexity will be.

" FIG0 : Illustration of common intuition for bias-variance tradeoff BID12

@highlight

We provide evidence against classical claims about the bias-variance tradeoff and propose a novel decomposition for variance.
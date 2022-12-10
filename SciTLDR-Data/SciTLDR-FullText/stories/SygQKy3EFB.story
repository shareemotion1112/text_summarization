Stochastic gradient descent (SGD) is the workhorse of modern machine learning.

Sometimes, there are many different potential gradient estimators that can be used.

When so, choosing the one with the best tradeoff between cost and variance is important.

This paper analyzes the convergence rates of SGD as a function of time, rather than iterations.

This results in a simple rule to select the estimator that leads to the best optimization convergence guarantee.

This choice is the same for different variants of SGD, and with different assumptions about the objective (e.g. convexity or smoothness).

Inspired by this principle, we propose a technique to automatically select an estimator when a finite pool of estimators is given.

Then, we extend to infinite pools of estimators, where each one is indexed by control variate weights.

This is enabled by a reduction to a mixed-integer quadratic program.

Empirically, automatically choosing an estimator performs comparably to the best estimator chosen with hindsight.

In stochastic gradient variational inference (SGVI) there are multiple gradient estimators with varying costs and variances.

Estimators may be obtained using the reparameterization trick (Kingma and Welling (2013) ; Rezende et al. (2014) ; Titsias and Lázaro-Gredilla (2014)), the score function method (Williams (1992) ), or other techniques (Titsias and Lázaro-Gredilla (2015) ; Ruiz et al. (2016) ; Agakov and Barber (2004) ).

Also, many control variates can be added to an estimator to reduce variance (Miller et al. (2017) (2018)).

The cost and variance of an estimator significantly affects optimization convergence speed (Bottou et al. (2018) ).

The use of different estimators leads to different optimization performances, and the estimator with optimal cost-variance tradeoff is often situationdependent (for an example see Fig. 1 ).

In settings where multiple estimators with varying costs and variances are available, selecting the optimal one is important.

Rather than rely on the user to manually select one, we propose that estimator selection could be done adaptively.

This paper investigates how, given a pool of gradient estimators, automatically choose one to get the best convergence guarantee for stochastic optimization.

We study cost-variance tradeoffs by analyzing the convergence rates of several variants of SGD.

We express convergence rates in terms of time rather than iterations.

This leads to what we call the "G 2 T principle": A simple rule that predicts, given a pool of gradient estimators, which one results in the best convergence guarantees for optimization.

We use the principle to propose two gradient estimator selection algorithms: One for the case in which a finite pool of estimators is available, and other when the pool contains an infinite number of estimators, each indexed by control variate weights (i.e. control variate selection).

Notation: We use g(w, ξ), where ξ is a random variable, to denote an unbiased estimator of target's gradient, G 2 (g) to denote a bound on g's expected squared norm, and T (g) to denote the computational cost of computing estimator g(w, ξ), measured in seconds.

2 T Principle and Gradient Estimator Selection

Given a set of gradient estimators with varying costs and variances, our goal is to find the one that gives the best convergence guarantee for optimization algorithms.

Convergence guarantees for several variants of SGD are shown in Table 1 .

Given a pool of estimators with known G 2 and T , the one with minimum G 2 T should be used.

In practice, however, G 2 and T are typically not known.

We propose to use estimates.

Assuming that the cost of an estimator g(w, ξ) is independent of w, an estimateT (g) of T (g) can be obtained for each g ∈ G through a single initial profiling phase.

Dealing with G 2 (g) is more involved.

Convergence guarantees assume that E ||g(w, ξ)|| 2 ≤ G 2 (g) for all w. Often (e.g. when w is unbounded) this is not true for any finite G.

We propose an approach that is justified under two assumptions: (i) Optimization starting from a point w 0 will only visit a restricted part of parameter space.

It is sufficient to bound E ||g(w, ξ)|| 2 for the set of w that may actually be encountered. (ii) E g(w, ξ) 2 tends to decrease slowly over time.

When these are true, it makes sense to also form an estimatê G 2 (g) through an initial profiling phase, and to update these estimates a small number of times as optimization proceeds.

The approach is summarized in Alg.

1.

It is possible to use multiple control variates to reduce a gradient estimator's variance.

However, some control variates might be computationally expensive but only result in a small reduction in variance.

It may be better to remove them and accept a noisier but cheaper estimator.

How can we select what control variates to use?

When an unbiased gradient estimator g base and control variates c 1 , ...c J are given, a gradient estimator can be expressed as

The available estimators are G = {g a : a ∈ R J }.

The number of estimators g a ∈ G is infinite, and Alg.

1 cannot be used (cannot measureT andĜ 2 for each estimator g a individually).

We show that, despite having an infinite number of estimators, when estimators are indexed by control variate weights finding the one with minimumĜ 2T can be done efficiently.

This is because two properties hold: (i) EstimatesT (g a ) andĜ 2 (g a ) can be efficiently obtained for all estimators g a ∈ G through the use of shared statistics (a finite number of evaluations of the base estimator and control variates); and (ii) The resulting (combinatorial) optimization problem a * = arg min aĜ 2 (g a )T (g a ) can be reduced to a Mixed Integer Quadratically Constrained Program (MIQCP), which can be solved quickly in practice.

For all g ∈ G measure timeT (g).

for k = 1, 2, · · · do if time to re-select estimator then for each estimator g dô

Algorithm 2 SGD with automatically selected control variates.

Require: Set of estimators, G. Require: Times to re-select estimator.

Require: Number of MC samples M .

For g base measure time t 0 .

This section presents an overview of the experiments.

Full details are in the appendix.

We tackle inference problems using SGVI.

We consider three models: Logistic regression, a hierarchical regression model, and a Bayesian neural network.

For the simple logistic regression model we use a Gaussian with a full rank covariance as variational distribution q w (z).

For the other more complex models we use a factorized Gaussian.

We use SGD with momentum to optimize, and five samples z ∼ q w (z) to form Monte Carlo gradient estimates.

For both Algs.

1 and 2 we update the estimator used (by minimizingĜ 2T ) three times during training.

We first present an empirical validation for Alg.

1.

We compare the results achieved by using three different gradient estimators: (Rep) the plain reparameterization estimator, (Miller) the estimator proposed by Miller et al. (Miller et al. (2017) ), and (STL) the "sticking the landing" estimator (Roeder et al. (2017) ).

We also run Alg.

1 with the set of estimators G = {Rep, Miller, STL}, which uses the estimator g ∈ G with minimumĜ 2T .

We now present an empirical validation for Alg.

2 (control variate selection).

We consider the same three models as above.

The set of candidate estimators is G Auto = {g a : a ∈ R 3 }, where g a is as defined in eq. (2).

The base estimator is plain reparameterization, and there are three candidate control variates (c 1 , c 2 , c 3 ).

The goal is to check if Alg.

2 successfully navigates cost/variance tradeoffs.

We thus compare against using each possible fixed subsets of control variates S ⊆ {c 1 , c 2 , c 3 }, with the weights that minimize the estimator's variance (which can be estimated efficiently (Geffner and Domke (2018) Log.

Regression (a1a) We compare against using different fixed subsets of control variates with the weights that minimize the estimator's variance.

Lines are identified as follows: "Auto" stands for using Alg.

2 to select what control variates to use and their weights, "Base" stands for optimizing using the base gradient alone, "1" stands for using the fixed set of control variates {c 1 } with the minimum variance weights, "12" stands for using the fixed set of control variates {c 1 , c 2 }, and so on.

Appendix A. Appendix

Three different models were considered: a Bayesian neural network, a hierarchical Poisson model, and Bayesian logistic regression.

Bayesian logistic regression: We use the dataset a1a.

The training set is given by

, where y i is binary.

The model is specified by

Hierarchical Poisson model:

By Gelman et al. Gelman et al. (2007) .

The model measures the relative stop-and-frisk events in different precincts in New York city, for different ethnicities.

The model is specified by

In this case, e stands for ethnicity and p for precinct, Y ep for the number of stops in precinct p within ethnicity group e (observed), and N ep for the total number of arrests in precinct p within ethnicity group e (observed).

BNN: As done by Miller et al. (Miller et al. (2017) ) we use a subset of 100 rows from the "Red-wine" dataset (regression).

We implement a neural network with one hidden layer with 50 units and Relu activations.

Let D = {x i , y i } N i=1 be the training set.

The model is specified by log α ∼ N (0, 10 2 ), log τ ∼ N (0, 10 2 ), W ∼ N (0, α 2 I), (weights and biases)

A.2.

Details on the simulations Control variates used: c 1 : Difference between the entropy term computed exactly and estimated using reparameterization: c(w, ξ) = ∇ w log q w (T w (ξ)) − ∇ w Eq w log q w (Z).

c 2 : Control variate by Miller et al. (Miller et al. (2017) ) based on a second order Taylor expansion of log p(x, z).

c 3 : Difference between the prior term computed exactly and estimated using reparameterization: c(w, ξ) = ∇ w log p(T w (ξ)) − ∇ w Eq w log p(Z).

Algorithmic details: For Alg.

2 we use M = 400 to estimateĜ 2 (except for Logistic regression, where we use M = 200).

We re-select the optimal estimator three times during training, initially, after 10% of training is done, and after 50% of training is done.

Optimization details: We use SGD with momentum (β = 0.9) with 5 samples z ∼ q w (z) to form the Monte Carlo gradient estimates.

For all models we find an initial set of parameters by optimizing with the base gradient for 300 steps and a fixed learning rate of 10 −5 .

This initialization was helpful in practice because w tends to change rapidly at the beginning of optimization.

After this brief initialization, E ||g(w, ξ)|| 2 tends to change much more slowly, meaning our technique is more helpful.

The performance of all algorithms depends on the step-size.

To give a fair comparison, Figs. 1 and 2 summarize by showing the results with the best step-size for each estimator.

(12 stepsizes between 10 −6 and 10 −3 were considered.)

Estimators with control variates can be expressed as

An expression forT (g a ) can be obtained by noticing that computing g a only requires computing the base gradient and the control variates with non-zero weights.

Then, for all

2.

Using Tw(ξ) = µ + D 1/2 ξ, where ξ ∼ N (0, I), µ is the mean of qw and D 1/2 is the Cholesky factorization of the covariance of qw.

Thus, we can computeT (g a ) for all g a ∈ G only by profiling the base gradient and each control variate individually.

Similarly,Ĝ 2 (g a , w) is determined by the same set of base gradient and control variate evaluations, regardless of the value of a. Suppose that, at iteration k, we sample ξ k1 , ..., ξ kM .

Then, for all g a ∈ G,

Thus, we can computeĜ 2 (g a , w k ) for all g a ∈ G using only M evaluations of the base gradient g base and each control variate c i .

Equations (3) and (4) characterize the (estimated) cost and variance of the gradient estimator with weights a. We find the weights that result in the optimal cost-variance tradeoff by solving a * (w) = arg min

whereT (g a ) andĜ 2 (g a , w) are as in equations (3) and (4).

The solution a * (w) indicates what control variates to use (those with a * i = 0), and their weights.

Solving the (combinatorial) minimization problem in equation (5) may be challenging.

However, theorem 1 states that it can be reduced to a MIQCP, which can be solved fast using solvers such as Gurobi Gurobi Optimization (2018).

Theorem 1 When different gradient estimators are indexed by a set of J control variate weights, the problem of finding a * (w) as in equation (5) can be reduced to solving a mixed integer quadratically constrained program with 2J + 2 variables, one quadratic constraint, and one linear constraint.

A mixed integer quadratic program is an optimization problem in which the objective function and constraints are quadratic (or linear), and some (or all) variables are restricted to be integers:

Ax + b = 0, where x ∈ R n , Q 0 , ..., Q m ∈ R n×n , and some components of x are restricted to be integers.

We now prove the theorem 1.

Proof GivenT

The final minimization problem shown in equation 14 has the form of a general MIQCP, shown in equation 6, with the exception of the last constraint b i = 1[a i = 0].

Despite not being in the original definition of a MIQCP, several solver accept constraints of this type (Gurobi Gurobi Optimization (2018) , the solver used in our simulation, does).

@highlight

We propose a gradient estimator selection algorithm with the aim on improving optimization efficiency.
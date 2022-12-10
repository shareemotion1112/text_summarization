Weight-sharing—the simultaneous optimization of multiple neural networks using the same parameters—has emerged as a key component of state-of-the-art neural architecture search.

However, its success is poorly understood and often found to be surprising.

We argue that, rather than just being an optimization trick, the weight-sharing approach is induced by the relaxation of a structured hypothesis space, and introduces new algorithmic and theoretical challenges as well as applications beyond neural architecture search.

Algorithmically, we show how the geometry of ERM for weight-sharing requires greater care when designing gradient- based minimization methods and apply tools from non-convex non-Euclidean optimization to give general-purpose algorithms that adapt to the underlying structure.

We further analyze the learning-theoretic behavior of the bilevel optimization solved by practical weight-sharing methods.

Next, using kernel configuration and NLP feature selection as case studies, we demonstrate how weight-sharing applies to the architecture search generalization of NAS and effectively optimizes the resulting bilevel objective.

Finally, we use our optimization analysis to develop a simple exponentiated gradient method for NAS that aligns with the underlying optimization geometry and matches state-of-the-art approaches on CIFAR-10.

Weight-sharing neural architecture search (NAS) methods have achieved state-of-the-art performance while requiring computation training of just a single shared-weights network (Pham et al., 2018; Li and Talwalkar, 2019; .

However, weight-sharing remains poorly understood.

In this work, we present a novel perspective on weight-sharing NAS motivated by the key observation that these methods subsume the architecture hyperparameters as another set of learned parameters of the shared-weights network, in effect extending the hypothesis class.

An important ramification of this insight is that weight-sharing is not NAS-specific and can be used to tune hyperparameters corresponding to parameterized feature maps of the input data.

We refer this larger subset of hyperparameter optimization problems as architecture search, and we study the following two questions associated with weight-sharing applied to the architecture search problem:

1.

How can we efficiently optimize the objective induced by applying weight sharing to architecture search, namely minimizing empirical risk in the joint space of model and architecture parameters?

For large structured search spaces that preclude brute force search, a natural approach to architecture search with weight-sharing is to use gradient-based methods to minimize the empirical risk over a continuous relaxation of the discrete space .

Although this has allowed NAS researchers to apply their preferred optimizers to determine architecture weights, it is far from clear that the success of established methods for unconstrained optimization in training neural networks will naturally extend to these constrained and often non-Euclidean environments.

As we foresee that architecture search spaces will continue to become more complex and multi-faceted, we argue for and develop a more principled, geometry-aware formulation of the optimization problem.

Drawing upon the mirror descent meta-algorithm (Beck and Teboulle, 2003) and successive convex approximation, we give non-asymptotic stationary-point convergence guarantees for the empirical risk minimization (ERM) objective associated with weight-sharing via algorithms that simultaneously connect to the underlying problem structure and handle the alternating-block nature of the architecture search.

Our guarantees inform the design of gradient-based weight-sharing methods by explicitly quantifying the impact of optimizing in the right geometry on convergence rates.

2.

What are the generalization benefits of solving a bilevel optimization for the architecture search problem commonly considered in practice?

At its core, the goal of architecture search is to find a configuration that achieves good generalization performance.

Consequently, a bilevel objective that optimizes the architecture weights using a separate validation loss is commonly used in practice in lieu of the ERM objective naturally induced by weight sharing (Pham et al., 2018; Cai et al., 2019) .

The learning aspects of this approach have generally been studied in settings with much stronger control over the model complexity (Kearns et al., 1997) .

We provide generalization guarantees for this objective over structured hypothesis spaces associated with a finite set of architectures; this leads to meaningful bounds for simple feature map selection problems as well as insightful results for the NAS problem that depend on the size of the space of global optima.

To validate our theoretical results, we conduct empirical studies of weight-sharing in two settings: (1) shallow feature map selection, i.e., tuning the hyperparameters of kernel classification and NLP featurization pipelines, and (2) CNN neural architecture search.

In (1) we demonstrate that weightsharing efficiently optimizes the bilevel objective and achieves low generalization error with respect to the best architecture setting.

For (2), motivated by insights from our convergence analysis, we develop a simple exponentiated gradient version of DARTS called EDARTS that better exploits the geometry of the optimization problem.

We evaluate EDARTS on the design of CNN architectures for CIFAR-10 and demonstrate that EDARTS finds better architectures than DARTS in less than half the time.

We also achieve very competitive results relative to state-of-the-art architectures when using an extended evaluation routine.

Related Work: Our work on optimization for weight-sharing benefits from the literature on firstorder stochastic optimization (Hazan and Kale, 2014; Beck, 2017) and in particular the mirror descent framework (Beck and Teboulle, 2003) .

Specifically, we use successive convex approximation (Razaviyayn et al., 2013; Mairal, 2015) to show convergence of alternating minimization and derive geometry-dependent rates comparable to existing work on non-convex stochastic mirror descent (Dang and Lan, 2015; Zhang and He, 2018) .

Our result generalizes to the constrained, nonEuclidean, and multi-block setting an approach of Agarwal et al. (2019) for obtaining non-convex convergence from strongly convex minimization, which may be of independent interest.

Previous optimization results for NAS have generally only shown bounds on auxiliary quantities such as regret that are not well-connected to the learning objective Carlucci et al., 2019) or have only given monotonic improvement or asymptotic guarantees (Akimoto et al., 2019; Yao et al., 2019) .

However, due to the generality of mirror descent, the approaches in the middle three papers can be seen as special cases of our analysis.

Finally, our analysis of the properties of the bilevel optimization is related to work on model selection (Vuong, 1989; Kearns et al., 1997) , but does not consider the configuration parameters as explicit controls on the model complexity.

Our learning results are broadly related to hyperparameter optimization, although most work focuses on algorithmic and not statistical questions (Li et al., 2018; Kandasamy et al., 2017) .

In this section, we formalize the weight-sharing learning problem, relate it to traditional ERM, and provide examples for the case of NAS and feature map selection that we use for the rest of the paper.

Our main observation is that weight-sharing for architecture search extends the hypothesis space to be further parameterized by a finite set of configurations C. Formally, we have a structured hypothesis space H(C, W) = {h w ∈ H(C, W) with low population error

w (x), y) for loss : Y × Y → R. Hence, we can apply ERM as usual, with optional regularization, to select a hypothesis from the extended hypothesis space; in fact this is done by some NAS methods (e.g., ).

The learning algorithm is then min w∈W,c∈C

for block specific regularizers R W and R C .

Note that in the absence of weight-sharing, we would need to learn a separate hypothesis h (c) wc for each hypothesis subclass H c .

Although a brute force approach to selecting a hypothesis from H(C, W) via ERM would in effect require this as well, our subsequent examples demonstrate how the weight-sharing construct allows us to apply more efficient gradient-based optimization approaches, which we study in Section 3.

Feature Map Selection:

In this setting, the structure is induced by a set of feature transformations C = {φ i : X → R n for i = 1, . . .

, k}, so the hypothesis space is {f w (φ i (·)) : w ∈ W, φ i ∈ C} for some W ⊂ R d .

Examples of feature map selection problems include tuning kernel hyperparameters for kernel ridge classification and tuning NLP featurization pipelines for text classification.

In these cases f w is a linear mapping f w (·) = w, · and W ⊂ R n .

Neural Architecture Search: Weight-sharing methods almost exclusively use micro cell-based search spaces for their tractability and additional structure (Pham et al., 2018; .

These search spaces can be represented as directed acyclic graphs (DAGs) with a set of ordered nodes N and edges E. Each node x (i) in the DAG is a feature representation and each edge o (i,j) is an operation on the feature of node j passed to node i and aggregated with other inputs to form x (j) , with the restriction that a given node j can only receive edges from prior nodes as input.

Hence, the feature at a given node i is

.

Search spaces are then specified by the number of nodes, the number of edges per node, and the set of operations O that can be applied at each edge.

In this case the structure C ⊂ {0, 1}

|E||O| of the hypothesis space is the set of all valid architectures for this DAG encoded by edge and operation decisions.

Treating both weights w ∈ W and architecture decision c ∈ C as parameters, weight-sharing methods train a single shared-weights network h (c) w : X → Y encompassing all possible functions within the search space.

Therefore, the sharedweights network includes all possible edges between nodes and all possible operations per edges.

In addition to the weights w ∈ W corresponding to all the operations, the shared-weights network also takes architecture weights c ∈ C as input, where c

indicates the weight given to operation o on edge (i, j) so that the feature of a given node i is

Gradient-based weight-sharing methods apply continuous relaxations to the architecture search space in order to compute gradients.

Some methods like DARTS and its variants (Chen et al., 2019; Laube and Zell, 2019; Hundt et al., 2019; Noy et al., 2019; relax the search space by considering a mixture of operations per edge and then discretize to a valid architecture in the search space.

With the mixture relaxation, we replace all c ∈ {0, 1} |E||O| in the above expressions by continuous counterparts θ ∈ [0, 1] |E||O| , with the constraint that o∈O θ (i,j) o = 1, i.e., the architecture weights for operations on each edge sum to 1.

Other methods like SNAS , ASNG-NAS (Akimoto et al., 2019) , and ProxylessNAS (Cai et al., 2019 ) assume a parameterized distribution p θ from which architectures are sampled.

By substituting continuous parameters θ ∈ Θ in for discrete parameters c ∈ C, we are able to use gradient-based methods to optimize (1).

We address the question of how to effectively use gradient optimization for weight-sharing in the next section.

While continuous relaxation enables state-of-the-art results, architecture search remains expensive and noisy, with state-of-the-art mixture methods requiring second-order computations and probabilistic methods suffering from high variance policy gradients .

Moreover, while the use of SGD to optimize network weights is a well-tested approach, architecture weights typically lie in constrained, non-Euclidean geometries in which other algorithms may be more appropriate.

Recognizing this, several efforts have attempted to derive a better optimization schemes; however, the associated guarantees for most of them hold for auxiliary objectives, such as regret of local configuration decisions, that are not connected to the optimization objective (1) and ignore the two-block nature of the problem Carlucci et al., 2019) .

While Akimoto et al. (2019) do consider an alternating descent method for the training objective, their results are asymptotic and certainly do not indicate any finite-time convergence.

In this section we address these difficulties by showing that the mirror descent (MD) framework (Beck and Teboulle, 2003) is the right tool for designing algorithms in the block optimization problems that occur in architecture search.

We describe how such geometry-aware gradient algorithms lead to faster stationary-point convergence; as we will show in Section 5, this yields simple, principled, and effective algorithms for large-scale multi-geometry problems such as NAS.

Algorithm 1: Two geometry-aware optimization algorithms for multi-block optimization for a β-strongly-smooth function over

if SBMD (Dang and Lan, 2015)

Problem Geometry: We relax optimization problems of the form (1) to the problem of minimizing a function f : X → R over a convex product space

X i consisting of blocks i, each with an associated norm · (i) .

For example, in typical NAS we can set X 1 to be the set-product of |E| simplices over the operations O and the associated norm to be · 1 while X 2 = W ⊂ R d is the space of network weights and the associated norm is · 2 .

To each block i we further associate a distance-generating function (DGF) ω i : X → R that is 1-strongly-convex w.r.t. (Bregman, 1967) .

For example, in the Euclidean case using ω 2 (·) = yields the usual squared Euclidean distance; over the probability simplex we often use the entropy ω 1 (·) = ·, log(·) , which is 1-strongly-convex w.r.t.

· 1 and for which D ω1 is the KL-divergence.

Given an unbiased gradient estimate g(x t , ζ) = E ζ ∇f (x t ), the (single-block) stochastic MD step is

for some learning rate η > 0.

In the Euclidean setting this reduces to SGD, while with the entropic regularizer the iteration becomes equivalent to exponentiated gradient.

Key to the guarantees for MD is the fact that the dual norm of the problem geometry is used to measure the second moment of the gradient; if every coordinate of g is bounded a.s.

by σ, then under Euclidean geometry the dependence is on E g(x) 2 2, * = E g(x) 2 2 ≤ σ 2 d while using entropic regularization it is on

.

Thus in such constrained 1 geometries mirror descent can yield dimension-free convergence guarantees.

While in this paper we focus on the benefit in this simple case, the MD meta-algorithm can be used for many other geometries of interest in architecture search, such as for optimization over positive-definite matrices (Tsuda et al., 2005) .

Algorithms and Guarantees: We propose two methods for the above multi-block optimization problems: stochastic block mirror descent (SBMD) and alternating successive convex approximation (ASCA).

At each step, both schemes pick a random coordinate i to update; SBMD then performs a mirror descent update similar to (2) but with a batched gradient, while ASCA optimizes a strongly-convex surrogate function using a user-specified solver.

Note that both methods require that f is β-strongly-smooth w.r.t.

each block's norm · (i) to achieve convergence guarantees, a standard assumption in stochastic non-convex optimization (Dang and Lan, 2015; Agarwal et al., 2019) .

This condition holds for the architecture search under certain limitations, such as a restriction to smooth activations.

In the supplement we also show that in the single-block case ASCA converges under the more general relative-weak-convexity criterion (Zhang and He, 2018) .

We first discuss SBMD, for which non-convex convergence guarantees were shown by Dang and Lan (2015) ; this algorithm is the one we implement in Section 5 for NAS.

A first issue is how to measure stationarity in constrained, non-Euclidean geometries.

In the single-block setting we can set a smoothness-dependent constant λ > 0 and measure how far the proximal gradient operator prox∇ λ (x) = arg min u∈X λ ∇f (x), u + D ω (u||x) is from a fixed point, which yields the projected gradient measure

.

Notably, in the unconstrained Euclidean case this yields the standard stationarity measure ∇f (x) 2 2 .

For the multi-block case we replace

an ε-stationary point of f w.r.t.

the projected gradient if G λ (x) ≤ ε.

In the non-Euclidean case the norm of the projected gradient measures how far we are from satisfying a first-order optimality condition, namely how far the negative gradient is from being in the normal cone of f to the set X (Dang and Lan, 2015, Proposition 4.1) .

For this condition, Algorithm 1 has the following guarantee: Theorem 3.1 (Dang and Lan (2015) ).

If f is β-strongly-smooth, F = f (x (1) ) − min u∈X f (u), and

oracle calls to reach an ε-stationary-point x ∈ X as measured by G 1

We next provide a guarantee for ASCA in the form of a reduction to strongly-convex optimization algorithms.

This is accomplished by the construction of the solving of a surrogate function at each iteration, which in the Euclidean case is effectively adding weight-decay.

ASCA is useful in the case when efficient strongly-convex solvers are available for some or all of the blocks; for example, this is frequently the case for feature map selection problems such as our kernel approximation examples, which employ 2 -regularized ERM in the inner optimization.

Taking inspiration from Zhang and He (2018) , for ASCA we analyze a stronger notion of stationarity that upper-bounds G λ (x) 2 , which we term the projected stationarity measure:

.

Note that this is not the same measure used by Zhang and He (2018) , although in the appendix we show that in the single-block case our result holds for their notion also.

Theorem 3.2.

If f is β-strongly-smooth and

where ε > 0 is the solver tolerance and the expectation is over the randomness of the algorithm and associated oracles.

Proof Summary.

The proof generalizes a result of Agarwal et al. (2019) and is in Appendix A.

Thus if we have solvers that return approximate optima of strongly-convex functions on each block then we can converge to a stationary point of the original function.

The convergence rate will depend on the solver used; for concreteness we give a specification for stochastic mirror descent.

Corollary 3.1.

Under the conditions of Theorem 3.1, if on i ASCA uses the Epoch-GD method of Hazan and Kale (2014)

ωi oracle calls suffice to reach an ε-stationary-point x ∈ X as measured by E ∆ 1

This oracle complexity matches that of Theorem 3.1 apart from the extra β 2 L 2 ωi term due to the surrogate function, which we show below is in practice a minor term that does not obviate the benefit of geometry-awareness.

On the other hand, ASCA is much more general, allowing for many different algorithms on individual blocks; for example, many popular neural optimizers such as Adam (Kingma and Ba, 2015) have variants with strongly-convex guarantees (Wang et al., 2019) .

The Benefit of Geometry-Aware Optimization: We conclude this section with a formalization of how convergence of gradient-based architecture-search algorithms can benefit from this optimization strategy.

Recalling from our example, we have the architecture parameter space X 1 consisting of |E| simplices over |O| variables equipped with the 1-norm and the shared weight space X 2 = W equipped with the Euclidean norm.

We suppose that the stochastic gradients along each block i have coordinates bounded a.s.

by σ i > 0.

Then if we run SBMD using SGD ( 2 ) to optimize the shared weights and exponentiated gradient ( 1 ) to update the architecture parameters, Theorem 3.1 implies that we reach an ε-stationary point in O stochastic gradient computations.

The main benefit here is that the first term in the numerator is not σ 2 1 |E||O|, which would be the case if we used SGD; this improvement is critical as the noise σ 2 1 of the architecture gradient can be very high, especially if a policy gradient is used to estimate probabilistic derivatives.

In the case of ASCA, we can get similar guarantees assuming the probabilities for each operation are lower-bounded by some small δ > 0 and that the space of shared weights is bounded by B; then the guarantee will be as above except with an additional O(log

2 ) term (independent of σ 1 , σ 2 ).

While for both SBMD and ASCA the σ 2 2 d term from training the architecture remains, this will be incurred even in single-architecture training using SGD.

Furthermore, in the case of ASCA it may be improved using adaptive algorithms (Agarwal et al., 2019; Wang et al., 2019) .

In Section 2, we described the weight-sharing hypothesis class H(C, W) as a set of functions nondisjointly partitioned by a set of configurations C sharing weights in W and posed the ERM problem associated with selecting a hypothesis from H(C, W).

However, as mentioned in Section 1, the objective solved in practice is a bilevel problem where a separate validation set is used for architecture parameter updates.

Formally, the bilevel optimization problem considered is min w∈W,c∈C

where T, V ⊂ Z is a pair of training/validation sets sampled i.i.d.

from D, the upper objective

w (x), y) is the empirical risk over V , and L T (w, c) is some objective induced by T .

We intentionally differentiate the two losses since training is often regularized.

This setup is closely related to the well-studied problems of model selection and cross-validation.

However, a key difference is that the choice of configuration c ∈ C does not necessarily provide any control over the complexity of the hypothesis space; for example, in NAS as it is often unclear how the hypothesis space changes due to the change in one decision.

By contrast, the theory of model selection is often directly concerned with control of model complexity.

Indeed, in possibly the most common setting the hypothesis classes are nested according to some total order of increasing complexity, forming a structure (Vapnik, 1982; Kearns et al., 1997) .

This is for example the case in most norm-based regularization schemes.

Even in the non-nested case, there is often an explicit tradeoff between parsimony and accuracy (Vuong, 1989) .

With the configuration parameters in architecture search behaving more like regular model parameters rather than as controls on the model complexity, it becomes reasonable to wonder why most NAS practitioners have used the bilevel formulation.

Does the training-validation split exploit the partitioning of the hypothesis space H(C, W) induced by the configurations C?

To see when this might be true, we first note that a key aspect of the optima of the bilevel weight-sharing problem is the restriction on the model weights -that they must be in the set arg min w∈W L T (h (c) w ) of the inner objective L T .

As we will see, under certain assumptions this can reduce the complexity of the hypothesis space without harming performance.

First, for any sample

w )} be the version space (Kearns et al., 1997, Equation 6 ) induced by some configuration c and the objective function.

Second, let N (F, ε) be the L ∞ -covering-number of a set of functions F at scale ε > 0, i.e. the number of L ∞ balls required to construct an ε-cover of F (Mohri et al., 2012, Equation 3 .60).

These two quantities let us define a complexity measure over the shared weight hypothesis space:

The version entropy is a data-dependent quantification of how much the hypothesis class is restricted by the inner optimization.

For finite C, a naive bound shows that Λ(H, ε, T ) is bounded by log |C| + max c∈C log N (H c (T ), ε), so that the second term measures the worst-case complexity of the global minimizers of L T .

In the feature selection problem, L T is usually a strongly-convex loss due to regularization and so all version spaces are singleton sets, making the version entropy log |C|.

In the other extreme case of nested model selection the version entropy reduces to the complexity of the version space of the largest model and so may not be informative.

However, in practical problems such as NAS an inductive bias is often imposed via constraints on the number of input edges.

To bound the excess risk in terms of the version entropy, we first discuss an important assumption that describes cases when we expect the shared weights approach to perform well:

Assumption 4.1.

There exists a good c * ∈ C, i.e. one satisfying (w * , c

w ) for some w * ∈ W, such that w.h.p.

over the drawing of training set T ∼ D m T at least one of the minima of the optimization induced by c * and T has low excess risk, i.e. w.p.

1 − δ there exists

This assumption requires that w.h.p.

the inner optimization objective does not exclude all low-risk classifiers for the optimal configuration.

Note that it asks nothing of either the other configurations in C, which may be arbitrarily bad, nor of the hypotheses found by the procedure.

It does however prevent the case where one knows the optimal configuration but minimizing the provided objective L T does not provide a set of good weights.

Note that if the inner optimization is simply ERM over the training set T , i.e. L T = T , then standard learning-theoretic guarantees will give ε * exc (m T , δ) decreasing in m T and increasing at most poly-logarithmically in 1 δ .

With this assumption, we can show the following guarantee on solutions to the bilevel optimization.

Theorem 4.1.

Letĥ be a hypothesis corresponding to the solution of the bilevel optimization (3).

Then under Assumption 4.1 if is B-bounded we have w.p.

1 − 3δ that

The first difference is bounded by the version entropy usng the constraint onĥ ∈ H c , the second by optimality ofĥ on V , the third by Hoeffding's inequality, and the last by Assumption 4.1.

As shown in the applications below, the significance of this theorem is that a bound on the version entropy guarantees excess risk almost as good as that of the (unknown) optimal configuration without assuming anything about the complexity or behavior of sub-optimal configurations.

Feature Map and Kernel Selection:

In the feature map selection problem introduced in Section 2,

} is a set of feature maps and the inner problem L T is 2 -regularized ERM for linear classification over the resulting feature vectors.

The bilevel problem is then

Due to strong-convexity of L T , each map φ i induces a unique minimizing weight w ∈ W and thus a singleton version space, therefore upper bounding the version entropy by log |C| = log N .

Furthermore, for Lipschitz losses and appropriate choice of regularization coefficient, standard results for 2 -regularized ERM for linear classification (e.g. Sridharan et al. (2008)

In the special case of kernel selection using random Fourier approximation, we can apply associated generalization guarantees (Rudi and Rosasco, 2017 , Theorem 1) to show that we can compete with the optimal RKHS from among those associated with one of the configurations : Corollary 4.2.

In feature map selection suppose each map φ ∈ C is associated with a random Fourier feature approximation of a continuous shift-invariant kernel that approximates an RKHS H φ and is the square loss.

If the number of features

In both cases we are able to get risk bounds almost identical to the excess risk achievable if we knew the optimal configuration beforehand, up to an additional capacity term depending weakly on the number of configurations.

This would not be possible with solving the regular ERM objective instead of the bilevel optimization as we would then have to contend with the possibly high complexity of the hypothesis space induced by the worst configuration.

Neural Architecture Search:

In the case of NAS we do not have a bound on the version entropy, which now depends on all of C. Whether the version space, and thus the complexity, of deep networks is small compared to the number of samples is unclear, although we gather some evidence.

number of critical points is exponential only in the number of layers, which would yield a small version entropy.

It is conceivable that the quantity may be further bounded by the complexity of solutions explored by the algorithm when optimizing L T (Nagarajan and Kolter, 2017; Bartlett et al., 2017) ; indeed, we find that shared-weight optimization leads to models with smaller 2 -norm and distance from initialization than from-scratch SGD on a single network (see Appendix D.4).

On the other hand, Nagarajan and Kolter (2019) argue, with evidence in restricted settings, that even the most stringent implicit regularization cannot lead to a non-vacuous uniform convergence bound; if true more generally this would imply that the NAS version entropy is quite large.

Here we demonstrate how weight-sharing can be used as a tool to speed up general architecture search problems by applying it to two feature map selection problems.

We then validate our optimization analysis with a geometry-aware weight-sharing method to design CNN cells for CIFAR-10.

Feature Map Selection:

Recall that here our configuration space has k feature maps φ i : X → R n with outputs passed to a linear classifier w ∈ R n , which will be the shared weights.

We will approximate the bilevel optimization (3) with the inner minimization over 2 -regularized ERM λ w 2 2 + (x,y)∈T ( w, φ i (x) , y).

Our weight-sharing procedure starts with a vector θ (1) ∈ ∆ N encoding a probability distribution p θ (1) over [N ] and proceeds as follows:

according to (an estimate of) its validation loss (x,y)∈V ( w (t) , φ i (x) , y).

Observe the equivalence to probabilistic NAS: at each step the classifier (shared parameter) is updated using random feature maps (architectures) on the training samples.

The distribution over them is then updated using estimated validation performance.

We consider two schemes for this update of θ (t) : (1) exponentiated gradient using the score-function estimate and (2) successive elimination, where we remove a fraction of the feature maps that perform poorly on validation and reassign their probability among the remainder.

(1) may be viewed as a softer version of (2), with halving also having only one hyperparameter (elimination rate) and not two (learning rate, stopping criterion).

The first problem we consider is kernel ridge regression over random Fourier features (Rahimi and Recht, 2008) on CIFAR-10.

We consider three configuration decisions: data preprocessing, choice of kernel, and bandwidth parameter.

This problem was considered by Li et al. (2018) , except they fixed the Gaussian kernel whereas we also consider Laplacian; however, they also select the regularization parameter λ, which weight-sharing does not handle.

We also study logistic regression for IMDB sentiment analysis of Bag-of-n-Gram (BonG) featurizations, a standard NLP baseline (Wang and Manning, 2012) .

Here there are eight configuration decisions: tokenization method, whether to remove stopwords, whether to lowercase, choice of n, whether to binarize features, type of feature weighting, smoothing parameter, and post-processing.

As some choices affect the feature dimension we hash the BonGs into a fixed number of bins (Weinberger et al., 2009 ).

To test the performance of weight-sharing for feature map selection, we randomly sample 64 configurations each for CIFAR-10 and IMDB and examine whether the above schemes converge to the optimal choice.

The main comparison method here is thus random search, which runs a full sweep over these samples; by contrast successive halving will need to solve 6 = log 2 64 regression problems, while for exponentiated gradient we perform early stopping after five iterations.

Note that weight-sharing can do no better than random search in terms of accuracy because they are picking a configuration from a space that random search sweeps over.

The goal is to see if it consistently returns a good configuration much faster.

As our results in Figures 1 and 2 show, successive halving indeed does almost as well as random search in much less time.

While exponentiated gradient usually does not recover a near-optimal solution, it does on average return a configuration in the top 10%.

We also note the strong benefit of over-parameterization for IMDB -the n-gram vocabulary has size 4 million so the number of bins on the right is much larger than needed to learn in a singleconfiguration setting.

Overall, these experiments show that weight-sharing can also be used as a fast way to obtain signal in regular learning algorithm configuration and not just NAS.

NAS on CIFAR-10: Recall from Section 3 that when the architecture space consists of |E| simplices of dimension |O|, the convergence rate of exponentiated gradient descent to a stationary point of the objective function is independent of the dimension of the space, while SGD has linear dependence.

This result motivates our geometry-aware method called Exponentiated-DARTS (EDARTS).

EDARTS modifies first-order DARTS in two ways.

First, in lieu of the softmax operation used by DARTS on the architecture weights, we use standard normalization so that the weight of operation o on edge (i, j) is u

.

Second, in lieu of Adam, we use exponentiated gradient to update the architecture weights: c t = c t−1 exp(−η∇ c V (h wt−1 (c t−1 )).

While EDARTS resembles XNAS , our justification for using exponentiated gradient comes directly from aligning with the optimization geometry of ERM.

Additionally, EDARTS only requires two straightforward modifications of first-order DARTS, while XNAS relies on a wipeout subroutine and granular gradient-clipping for each edge operation on the cell and data instance level.

1 1 Our own XNAS implementation informed by correspondence with the authors did not produce competitive results.

We still compare to the architecture XNAS reported evaluated by the DARTS training routine in Table 1.

We evaluate EDARTS on the task of designing a CNN cell for CIFAR-10.

We use the standard search space as introduced in DARTS ) for evaluation we use the same three stage process used by DARTS and random search with weight-sharing (Li and Talwalkar, 2019) , with stage 3 results considered the 'final' results.

We provide additional experimental details in Appendix D. Table 1 shows the performance of EDARTS relative to both manually designed and NAS-discovered architectures.

EDARTS finds an architecture that achieves competitive performance with manually designed architectures which have nearly an order-of-magnitude more parameters.

Additionally, not only does EDARTS achieve significantly lower test error than first-order DARTS, it also outperforms second order DARTS while requiring less compute time, showcasing the benefit of geometry-aware optimization.

Finally, EDARTS achieve comparable performance to the reported architecture for state-of-the-art method XNAS when evaluated using the stage 3 training routine of DARTS.

Following XNAS , we also perform an extended evaluation of the best architecture found by EDARTS with AutoAugment, cosine power annealing (Hundt et al., 2019) , cross-entropy with label smoothing , and trains for 1500 epochs.

We evaluated the XNAS architecture using our implementation for a direct comparison and also to serve as a reproducibility check.

EDARTS achieved a test error of 2.18% in the extended evaluation compared to 2.15% for XNAS in our reproduced evaluation; note the published test error for XNAS is 1.81%.

To meet a higher bar for reproducibility we report 'broad reproducibility' results by repeating the entire pipeline from stage 1 to stage 3 for two additional sets of seeds.

Our results in Table 2 (see Appendix) show that EDARTS has lower variance across experiments than random search with weight sharing (Li and Talwalkar, 2019) .

However, we do observe non-negligible variance in the performance of the architecture found by different random seed initializations of the shared-weights network, necessitating running multiple searches before selecting an architecture.

A OPTIMIZATION This section contains proofs and generalizations of the non-convex optimization results in Section 3.

We first gather some necessary definitions and results from convex analysis.

Definition A.1.

Let X be a convex subset of a finite-dimensional real vector space and f : X → R be everywhere sub-differentiable.

1.

For α > 0, f is α-strongly-convex w.r.t.

norm · if ∀ x, y ∈ X we have

Definition A.2.

Let X be a convex subset of a finite-dimensional real vector space.

The Bregman divergence induced by a strictly convex distance-generating function (DGF) ω : X → R is D ω (x||y) = ω(x) − ω(y) − ∇ω(y), x − y ∀ x, y ∈ X By definition, the Bregman divergence satisfies the following properties:

3.

If ω is β-strongly-smooth w.r.t.

norm · then so is D ω (·||y) ∀ y ∈ X .

Furthermore, D ω (x||y) ≤ β 2 x − y 2 ∀ x, y ∈ X .

Lemma A.1 (Three-Points Lemma). (Beck, 2017, Lemma 9.11) For any DGF ω : X → R and all x, y, z ∈ X we have ∇ω(y) − ∇ω(x), z − x = D ω (z||x) + D ω (x||y) − D ω (z||y) Definition A.3.

Let ω : X → R be a 1-strongly-convex DGF.

Then for constant λ > 0 and an everywhere sub-differentiable function f : X → R the proximal operator is defined over x ∈ X as prox λ (x) = arg min

Note that the prox operator is well-defined whenever f is β-strongly-smooth for some β < λ.

We will also use the following notation for the proximal gradient operator:

Note that the prox grad operator is always well-defined.

Theorem A.1. (Beck, 2017, Theorem 9.12) For any λ > 0, 1-strongly-convex DGF ω : X → R, and x ∈ X let f : X → R be an everywhere sub-differentiable function s.t.

λf (·) + D ω (·||x) is convex over X .

Then for x + = prox λ (x) and all u ∈ X we have

Lemma A.2.

For any λ > 0, 1-strongly-convex DGF ω : X → R, and x ∈ X let f : X → R be an everywhere sub-differentiable function s.t.

Proof.

Applying Theorem A.1 followed by Lemma A.1 yields

Corollary A.1.

For any λ > 0, 1-strongly-convex DGF ω : X → R, x ∈ X , and everywhere sub-differentiable function f : X → R we have for

Because we consider constrained non-convex optimization, we cannot measure convergence to a stationary point by the norm of the gradient.

Instead, we analyze the convergence proximal-mappingbased stationarity measures.

The most well-known measure is the norm of the projected gradient (Ghadimi and Lan, 2013) , which in the unconstrained Euclidean case reduces to the norm of the gradient and in the general case measure the distance between the current point and a cone satisfying first-order optimality conditions (Dang and Lan, 2015 , Proposition 4.1).

Our convergence results hold for a stronger measure that we call the projected stationarity and which is inspired by the Bregman stationarity measure of Zhang and He (2018) but using the prox grad operator instead of the prox operator.

Definition A.4.

Let ω : X → R be a 1-strongly-convex DGF and f : X → R be an everywhere sub-differentiable function.

Then for any λ > 0 we define the following two quantities:

The following properties follow:

We can also consider the Bregman stationarity measure of Zhang and He (2018) directly.

As this measure depends on the prox operator, which is not always defined, we first state the notion of non-convexity that Zhang and He (2018) consider.

Definition A.5.

An everywhere sub-differentiable function f : X → R is (γ, ω)-relatively-weaklyconvex ((γ, ω)-RWC) for γ > 0 and ω : X → R a 1-strongly-convex DGF if f (·) + γω(·) is convex over X .

Note that all γ-strongly-smooth functions are (γ, ω)-RWC.

Note that (γ, ω)-RWC is a generalization of γ-strong-smoothness w.r.t.

the norm w.r.t.

which ω is strongly-convex.

Furthermore, for such functions we can always define the prox operator for λ > γ, allowing us to also define the Bregman gradient below.

Similarly to before, bounding the Bregman stationarity measure yields a stronger notion of convergence than the squared norm of the Bregman gradient.

For the relationship between the Bregman stationarity measure and first-order optimality conditions see Zhang and He (2018, Equation 2.11) .

Definition A.6.

Let ω : X → R be a 1-strongly-convex DGF and f : X → R be a (γ, ω)-RWC everywhere sub-differentiable function for some γ > 0.

Then for any λ > γ we define the following two quantities:

Here we prove our main optimization results.

We begin with a descent lemma guaranteeing improvement of a non-convex function due to approximately optimizing a strongly convex surrogate.

Lemma A.3.

Let ω : X → R be a 1-strongly-convex DGF and f : X → R be everywhere subdifferentiable.

For some x ∈ X and ρ > 0 definef x (·) = f (·) + ρD ω (·||x) and letx ∈ X be a point s.t.

Ef x (x) − min u∈Xfx (u) ≤ ε.

Then 1.

If f is β-strongly-smooth, ρ > β, and λ =

Proof.

Generalizing an argument in Agarwal et al. (2019, Theorem A.2) , for x + ∈ X we have by strong-convexity off x that

If f is β-strongly-smooth set x + = prox∇ λ (x), so that by Corollary A.1 we have

In the other case of f being (γ, ω)-RWC set x + = prox λ (x), so that by Lemma A.2 we have

We now turn to formalizing our multi-block setting and assumptions.

Setting A.1.

For i = 1, . . .

, b let X i be a convex subset of a real vector space with an associated DGF ω i : X i → R that is 1-strongly-convex w.r.t.

some norm · (i) over X i .

We have an everywhere sub-differentiable function f : X → R over the product space

Our main results will hold for the case when the following general assumption is satisfied.

We will later show how this assumption can follow from strong smoothness or relative weak convexity and existing algorithmic results.

Assumption A.1.

In Setting A.1, for any given ε > 0 and each i ∈ [b] there exists a constant ρ i > 0 and an algorithm A i : X → X that takes a point x ∈ X and returns a pointx ∈ X satisfyinĝ x −i = x −i and

where the subscript i selects block i, the subscript −i selects all blocks other than block i, and E Ai denotes expectation w.r.t.

the randomness of algorithm A i and any associated stochastic oracles.

Algorithm 2: Generic successive convex approximation algorithm for reaching a stationary point of the non-convex function in Setting A.1.

Input: Point x (1) ∈ X in the product space of Setting A.1.

Algorithms A 1 , . . .

,

Our main result relies on the following simple lemma guaranteeing non-convex convergence for a generic measure satisfying guaranteed expected descent:

Lemma A.4.

In Setting A.1, for some ε > 0 and each

λi (x) be any measure s.t.

for some λ i and some algorithm A i : X → X we have

Then the output x of Algorithm 2 satisfies

where F = f (x (1) ) − arg min u∈X f (x) and the expectation is taken over the sampling at each iteration, the sampling of the output, and the randomness of the algorithms and any associated stochastic oracles.

Proof.

Define Ξ t = {(ξ s , A ξs )} t s=1 and note that x (t+1) = A ξt (x (t) ).

We then have

In the single-block setting, Lemmas A.3 and A.4 directly imply the following guarantee: Theorem A.2.

In Setting A.1 and under Assumption A.1, let b = 1 and ρ satisfy one of the following:

1. f : X → R is β-strongly-smooth and ρ = 2β.

2. f : X → R is (γ, ω)-RWC for some DGF ω : X → R and ρ = 2γ.

Then Algorithm 2 returns a point x ∈ X satisfying one of the following (respectively, w.r.t.

the above settings) for

Here the expectation is taken over the randomness of the algorithm and oracle.

We can apply a known result for the strongly convex case to recover the rate of Zhang and He (2018) for non-convex stochastic mirror descent, up to an additional depending on ω: Corollary A.2.

In Setting A.1 for b = 1 and (γ, ω)-RWC f , suppose we have access to f through a stochastic gradient oracle g(x) = E∇f (x) such that E g 2 * ≤ G 2 .

Let A : X → X be an algorithm that for any x ∈ X runs the Epoch-GD method of Hazan and Kale (2014) with total number of steps N , initial epoch length T 1 = 4 and initial learning rate η 1 = 1 γ onf x (·) = f (·) + 2γD ω (·||x).

Then with N T calls to the stochastic gradient oracle Algorithm 2 returns a point x ∈ X satisfying

and L ω the Lipschitz constant of ω w.r.t.

· over X .

So an expected ε-stationary-point, as measured by

Proof.

Apply Theorem 5 of Hazan and Kale (2014) together with the fact thatf x is γ-stronglyconvex w.r.t.

· and its stochastic gradient is bounded by

For the multi-block case our results hold only the projected stationarity measure: Theorem A.3.

In Setting A.1 and under Assumption A.1 assume f (·, x −i ) is β-strongly-smooth w.r.t.

Here the expectation is taken over the randomness of the algorithm and oracle and the projected stationarity measure ∆ λ is defined w.r.t.

the Bregman divergence of the DGF ω(

where prox∇

To apply Lemma A.4 in the multiblock setting it suffices to show that the sum of the projected stationarity measures on each block is equal to the projected stationarity measure induced by the sum of the DGFs.

For some λ > 0 and any i ∈ [b] we have that

and so

Thus applying Lemma A.2 with λ = 1 4ρ yields the result.

In the following corollary we recover the rate of Dang and Lan (2015) for non-convex blockstochastic mirror descent, up to an additional term depending on ω i : Corollary A.3.

In Setting A.1 for β-strongly-smooth f , suppose we have access to f through a stochastic gradient oracle g(x) = E∇f (x) such that E g i i .

For i ∈ [b] let A i : X → X be an algorithm that for any x ∈ X runs the Epoch-GD method of Hazan and Kale (2014) with total number of steps N , initial epoch length T 1 = 4 and initial learning rate η 1 = 1 γ on surrogate functionf x (·) = f (·, x −i ) + 2βD ωi (·||x i ).

Then with N T calls to the stochastic gradient oracle Algorithm 2 returns a point x ∈ X satisfying

We can specialize this to the architecture search setting where we have a configuration search space contained in the product of simplices induced by having n decisions with c choices each together with a parameter space bounded in Euclidean norm.

Corollary A.4.

Under the assumptions of Corollary A.3, suppose b = 2 and we have the following two geometries:

Suppose the stochastic gradient oracle of f has bounded ∞ -norm σ 1 over X 1 and σ 2 over X 2 .

Then Algorithm 2 will return an expected ε-stationary point of f under the projected stationarity measure in a number of stochastic oracle calls bounded by

This section contains proofs of the generalization results in Section 4.

We first describe the setting for which we prove our general result.

Setting B.1.

Let C be a set of possible architecture/configurations of finite size such that each c ∈ C is associated with a parameterized hypothesis class H c ={h (c)

w : X → Y : w ∈ W} for input space Z = X × Y and fixed set of possible weights W. We will measure the performance of a hypothesis h (c) w on an input z = (x, y) ∈ Z using z (w, c) = (h Finally, we will consider solutions of optimization problems that depend on the training data and architecture.

Specifically, for any configuration c ∈ C and finite subset S ⊂ Z let W c (S) ⊂ W be the set of global minima of some optimization problem induced by S and c and let the associated version space (Kearns et al., 1997) be H c (S) = {h w : X → Y is determined by a choice of architecture c ∈ C and a set of network weights w ∈ W and the loss : Y × Y → {0, 1} is the zero-one loss.

In the simplest case W c (S) is the set of global minima of the ERM problem min

We now state the main assumption we require.

Assumption B.1.

In Setting B.1 there exists a good architecture c * ∈ C, i.e. one satisfying (w * , c * ) ∈ arg min W×C D (w, c) for some weights w * ∈ W, such that w.p.

1 − δ over the drawing of training set T ∼ D m T at least one of the minima of the optimization problem induced by c * and T has low excess risk, i.e. ∃ w ∈ W c * (T ) s.t.

for some error function ε c * .

Clearly, we prefer error functions ε c * that are decreasing in the number of training samples m T and increasing at most poly-logarithmically in 1 δ .

This assumption requires that if we knew the optimal configuration a priori, then the provided optimization problem will find a good set of weights for it.

We will show how, under reasonable assumptions, Assumption B.1 can be formally shown to hold in Settings B.2 and B.3.

Our general result will be stated in terms of covering numbers of certain function classes.

Definition B.1.

Let H be a class of functions from X to Y .

For any ε > 0 the associated L ∞ covering number N (H, ε) of H is the minimal positive integer k such that H can be covered by k balls of L ∞ -radius ε.

The following is then a standard result in statistical learning theory (see e.g. Lafferty et al. (2010, Theorem 7.82

where we use the loss notation from Setting B.1.

Before stating our theorem, we define a final quantity, which measures the log covering number of the version spaces induced by the optimization procedure over a given training set.

Definition B.2.

In Setting B.1, for any sample S ⊂ X × Y define the version entropy to be Λ(H, ε, S) = log N c∈C H c (S), ε .

Theorem B.2.

In Setting B.1 let (ŵ,ĉ) ∈ W × C be obtained as a solution to the following optimization problem:

arg min

Then under Assumption B.1 we have w.p.

1 − 3δ that

each term of which can be bounded as follows:

1.

Sinceŵ ∈ Wĉ(T ) for someĉ ∈ C the hypothesis space can be covered by the union of the coverings of H c (T ) over c ∈ C, so by Theorem B.1 we have that w.

2.

By optimality of the pair (ŵ,ĉ) and the fact that w ∈ W c * (T ) we have

3.

Hoeffding's inequality yields V (w, c

We can then directly apply Theorem B.2 and the fact that the version entropy is bounded by log |C| because the minimizer over the training set is always unique to get the following: Corollary B.2.

In Setting B.2 let (ŵ,ĉ) ∈ W × C be obtained as a solution to the following optimization problem:

arg min

In the special case of kernel selection we can apply generalization results for learning with random features to show that we can compete with the optimal RKHS from among those associated with one of the configurations (Rudi and Rosasco, 2017 , Theorem 1): Corollary B.3.

In Setting B.2, suppose each configuration c ∈ C is associated with a random Fourier feature approximation of a continuous shift-invariant kernel that approximates an RKHS H c .

Suppose is the squared loss so that (ŵ,ĉ) ∈ W × C is obtained as a solution to the following optimization problem:

In the case of neural architecture search we are often solving (unregularized) ERM in the inner optimization problem.

In this case we can make an assumption weaker than Assumption B.1, namely that the set of empirical risk minimizers contains a solution that, rather than having low excess risk, simply has low generalization error; then applying Hoeffding's inequality yields the following: Corollary B.4.

In Setting B.1 let (ŵ,ĉ) ∈ W × C be obtained as a solution to the following optimization problem: arg min Suppose there exists c * ∈ C satisfying (w * , c * ) ∈ arg min W×C D (w, c) for some weights w * ∈ W such that w.p.

1 − δ over the drawing of training set T ∼ D m T at least one of the minima of the optimization problem induced by c * and T has low generalization error, i.e. ∃ w ∈ arg min w ∈W T (w , c * ) s.t.

Solvers for Ridge regression and logistic regression were from scikit-learn (Pedregosa et al., 2011) .

For CIFAR-10 we use the kernel configuration setting from Li et al. (2018) but replacing the regularization parameter by the option to use the Laplace kernel instead of Gaussian.

The regularization was fixed to λ = 1 2 The split is 40K/10K/10K.

For IMDB we consider the following configuration choices:

For stages 2 and 3, we train each architecture for 600 epochs with the same hyperparameter settings as DARTS.

For completeness, we describe the convolutional neural network search space considered.

The set of operations O considered at each node include: (1) 3 × 3 separable convolution, (2) 5 × 5 separable convolution, (3) 3×3 dilated convolution, (4) 5×5 dilated convolution, (5) max pooling, (6) average pooling, (7) identity.

We use the same search space to design a "normal" cell and a "reduction" cell; the normal cells have stride 1 operations that do not change the dimension of the input, while the reduction cells have stride 2 operations that half the length and width dimensions of the input.

In the experiments, for both cell types, we set N = 6 with 2 input nodes and 4 intermediate nodes, after which the output of all intermediate nodes are concatenated to form the output of the cell.

We use EDARTS to train a smaller shared-weights network in the search phase with 8 layers and 24 initial channels instead of the 16 used by DARTS.

Additionally, to more closely mirror the architecture used for evaluation in stage 2 and 3, we use an auxiliary head with weight 0.4 and scheduled path dropout of 0.2.

For the EDARTS architecture updates, we use a learning rate of 0.2 for the normal cell and 0.6 for the reduction cell.

All other hyperparameters are the same as DARTS: 50 training epochs, batch size of 64, gradient clipping of 5 for network weights, SGD with momentum set to 0.9 and learning rate annealed from 0.025 to 0.001 with cosine annealing (Loshchilov and Hutter, 2016) , and weight decay of 0.0003.

We use the same evaluation scheme as DARTS when retraining architectures from scratch.

The larger evaluation network has 20 layers and 36 initial channels and is trained for 600 epochs using SGD with momentum set to 0.9, a batch size of 96, and a learning rate of 0.025 annealed down to 0; the gradient clipping scheduled drop path rate and weight decay are identical to the search phase.

We also use an auxiliary head with a weight of 0.4 and cutout (Devries and Taylor, 2017) .

We investigate whether weight-sharing implicitly regularizes the hypothesis space by examining the 2 norms and distance from initialization of the shared-weights network relative to that observed when training the best EDARTS architecture from scratch.

We use the same network depth and hyperparameters as those used for the shared-weights network to train the fixed architecture.

Figure 4 shows the percent difference in the norms between the fixed architecture and the shared-weights network pruned to just the operations kept for the fixed architecture.

From the chart, we can see that both the 2 distance from initialization and the 2 norm of the shared-weights is smaller than that of a fixed network are higher than that of the shared-weights network by over 40%, suggesting weight-sharing acts as a form of implicit regularization.

Table 2 show that EDARTS has lower variance across experiments than random search with weight sharing (Li and Talwalkar, 2019) .

However, we do observe non-negligible variance in the performance of the architecture found by different random seed initializations of the shared-weights network, necessitating running multiple searches before selecting an architecture.

<|TLDR|>

@highlight

An analysis of the learning and optimization structures of architecture search in neural networks and beyond.
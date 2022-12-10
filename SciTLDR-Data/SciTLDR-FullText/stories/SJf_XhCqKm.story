Driven by the need for parallelizable hyperparameter optimization methods, this paper studies open loop search methods: sequences that are predetermined and can be generated before a single configuration is evaluated.

Examples include grid search, uniform random search, low discrepancy sequences, and other sampling distributions.

In particular, we propose the use of k-determinantal point processes in  hyperparameter optimization via random search.

Compared to conventional uniform random search where hyperparameter settings are sampled independently, a k-DPP promotes diversity.

We describe an approach that transforms hyperparameter search spaces for efficient use with a k-DPP.

In addition, we introduce a novel Metropolis-Hastings algorithm which can sample from k-DPPs defined over any space from which uniform samples can be drawn, including spaces with a mixture of discrete and continuous dimensions or tree structure.

Our experiments show significant benefits  in realistic scenarios with a limited budget for training supervised learners, whether in serial or parallel.

Hyperparameter values-regularization strength, model family choices like depth of a neural network or which nonlinear functions to use, procedural elements like dropout rates, stochastic gradient descent step sizes, and data preprocessing choices-can make the difference between a successful application of machine learning and a wasted effort.

To search among many hyperparameter values requires repeated execution of often-expensive learning algorithms, creating a major obstacle for practitioners and researchers alike.

In general, on iteration (evaluation) k, a hyperparameter searcher suggests a d-dimensional hyperparameter configuration x k ∈ X (e.g., X = R d but could also include discrete dimensions), a worker trains a model using x k , and returns a validation loss of y k ∈ R computed on a hold out set.

In this work we say a hyperparameter searcher is open loop if x k depends only on {x i } k−1 i=1 ; examples include choosing x k uniformly at random BID3 , or x k coming from a low-discrepancy sequence (c.f., BID13 ).

We say a searcher is closed loop if x k depends on both the past configurations and validation losses {(x i , y i )} k−1 i=1 ; examples include Bayesian optimization BID23 and reinforcement learning methods BID30 .

Note that open loop methods can draw an infinite sequence of configurations before training a single model, whereas closed loop methods rely on validation loss feedback in order to make suggestions.

While sophisticated closed loop selection methods have been shown to empirically identify good hyperparameter configurations faster (i.e., with fewer iterations) than open loop methods like random search, two trends have rekindled interest in embarrassingly parallel open loop methods: 1) modern deep learning model are taking longer to train, sometimes up to days or weeks, and 2) the rise of cloud resources available to anyone that charge not by the number of machines, but by the number of CPU-hours used so that 10 machines for 100 hours costs the same as 1000 machines for 1 hour.

This paper explores the landscape of open loop methods, identifying tradeoffs that are rarely considered, if at all acknowledged.

While random search is arguably the most popular open loop method and chooses each x k independently of {x i } k−1 i=1 , it is by no means the only choice.

In many ways uniform random search is the least interesting of the methods we will discuss because we will advocate for methods where x k depends on {x i } k−1 i=1 to promote diversity.

In particular, we will focus on k i=1 from a k-determinantal point process (DPP) BID18 .

We introduce a sampling algorithm which allows DPPs to support real, integer, and categorical dimensions, any of which may have a tree structure, and we describe connections between DPPs and Gaussian processes (GPs).In synthetic experiments, we find our diversity-promoting open-loop method outperforms other open loop methods.

In practical hyperparameter optimization experiments, we find that it significantly outperforms other approaches in cases where the hyperparameter values have a large effect on performance.

Finally, we compare against a closed loop Bayesian optimization method, and find that sequential Bayesian optimization takes, on average, more than ten times as long to find a good result, for a gain of only 0.15 percent accuracy on a particular hyperparameter optimization task.

Open source implementations of both our hyperparameter optimization algorithm (as an extension to the hyperopt package BID4 ) and the MCMC algorithm introduced in Algorithm 2 are available.

1 2 RELATED WORK While this work focuses on open loop methods, the vast majority of recent work on hyperparameter tuning has been on closed loop methods, which we briefly review.

Much attention has been paid to sequential model-based optimization techniques such as Bayesian optimization BID5 BID23 which sample hyperparameter spaces adaptively.

These techniques first choose a point in the space of hyperparameters, then train and evaluate a model with the hyperparameter values represented by that point, then sample another point based on how well previous point(s) performed.

When evaluations are fast, inexpensive, and it's possible to evaluate a large number of points (e.g. k = Ω(2 d ) for d hyperparameters) these approaches can be advantageous, but in the more common scenario where we have limited time or a limited evaluation budget, the sequential nature of closed loop methods can be cumbersome.

In addition, it has been observed that many Bayesian optimization methods with a moderate number of hyperparameters, when run for k iterations, can be outperformed by sampling 2k points uniformly at random , indicating that even simple open loop methods can be competitive.

Parallelizing Bayesian optimization methods has proven to be nontrivial, though many agree that it's vitally important.

While many algorithms exist which can sample more than one point at each iteration BID7 BID8 BID9 BID14 , the sequential nature of Bayesian optimization methods prevent the full parallelization open loop methods can employ.

Even running two iterations (with batches of size k/2) will take on average twice as long as fully parallelizing the evaluations, as you can do with open loop methods like grid search, sampling uniformly, or sampling according to a DPP.One line of research has examined the use of k-DPPs for optimizing hyperparameters in the context of parallelizing Bayesian optimization BID15 BID27 .

At each iteration within one trial of Bayesian optimization, instead of drawing a single new point to evaluate from the posterior, they define a k-DPP over a relevance region from which they sample a diverse set of points.

They found their approach to beat state-of-the-art performance on a number of hyperparameter optimization tasks, and they proved that generating batches by sampling from a k-DPP has better regret bounds than a number of other approaches.

They show that a previous batch sampling approach which selects a batch by sequentially choosing a point which has the highest posterior variance BID7 is just approximating finding the maximum probability set from a k-DPP (an NP-hard problem BID18 ), and they prove that sampling (as opposed to maximization) has better regret bounds for this optimization task.

We use the work of BID15 as a foundation for our exploration of fully-parallel optimization methods, and thus we focus on k-DPP sampling as opposed to maximization.

So-called configuration evaluation methods have been shown to perform well by adaptively allocating resources to different hyperparameter settings BID26 BID19 .

They initially choose a set of hyperparameters to evaluate (often uniformly), then partially train a set of models for these hyperparameters.

After some fixed training budget (e.g., time, or number of training examples observed), they compare the partially trained models against one another and allocate more resources to those which perform best.

Eventually, these algorithms produce one (or a small number) of fully trained, high-quality models.

In some sense, these approaches are orthogonal to open vs. closed loop methods, as the diversity-promoting approach we advocate can be used as a drop-in replacement to the method used to choose the initial hyperparameter assignments.

GPs have long been lauded for their expressive power, and have been used extensively in the hyperparameter optimization literature.

BID10 show that drawing a sample from a k-DPP with kernel K is equivalent to sequentially sampling k times proportional to the (updated) posterior variance of a GP defined with covariance kernel K. This sequential sampling is one of the oldest hyperparameter optimization algorithms, though our work is the first to perform an in-depth analysis.

Additionally, this has a nice information theoretic justification: since the entropy of a Gaussian is proportional to the log determinant of the covariance matrix, points drawn from a DPP have probability proportional to exp(information gain), and the most probable set from the DPP is the set which maximizes the information gain.

With our MCMC algorithm presented in Algorithm 2, we can draw samples with these appealing properties from any space for which we can draw uniform samples.

The ability to draw k-DPP samples by sequentially sampling points proportional to the posterior variance grants us another boon: if one has a sample of size k and wants a sample of size k + 1, only a single additional point needs to be drawn, unlike with the sampling algorithms presented in BID18 .

Using this approach, we can draw samples up to k = 100 in less than a second on a machine with 32 cores.

As discussed above, recent trends have renewed interest in open loop methods.

While there exist many different batch BO algorithms, analyzing these in the open loop regime (when there are no results from function evaluations) is often rather simple.

As there is no information with which to update the posterior mean, function evaluations are hallucinated using the prior or points are drawn only using information about the posterior variance.

For example, in the open loop regime, BID14 's approach without hallucinated observations is equivalent to uniform sampling, and their approach with hallucinated observations (where they use the prior mean in place of a function evaluation, then update the posterior mean and variance) is equivalent to sequentially sampling according to the posterior variance (which is the same as sampling from a DPP).

Similarly, open loop optimization in SMAC BID12 ) is equivalent to first Latin hypercube sampling to make a large set of diverse candidate points, then sampling k uniformly among these points.

Recently, uniform sampling was shown to be competitive with sophisticated closed loop methods for modern hyperparameter optimization tasks like optimizing the hyperparameters of deep neural networks , inspiring other works to explain the phenomenon BID0 .

BID3 offer one of the most comprehensive studies of open loop methods to date, and focus attention on comparing random search and grid search.

A main takeaway of the paper is that uniform random sampling is generally preferred to grid search 2 due to the frequent observation that some hyperparameters have little impact on performance, and random search promotes more diversity in the dimensions that matter.

Essentially, if points are drawn uniformly at random in d dimensions but only d < d dimensions are relevant, those same points are uniformly distributed (and just as diverse) in d dimensions.

Grid search, on the other hand, distributes configurations aligned with the axes so if only d < d dimensions are relevant, many configurations are essentially duplicates.

However, grid search does have one favorable property that is clear in just one dimension.

If k points are distributed on [0, 1] on a grid, the maximum spacing between points is equal to 1 k−1 .

But if points are uniformly at random drawn on [0, 1], the expected largest gap between points scales as DISPLAYFORM0 ) is a point on the grid for ij = 0, 1, . . .

, m for all j, with a total number of grid points equal to (m + 1) d .bad luck, the optimum islocated in this largest gap, this difference could be considerable; we attempt to quantify this idea in the next section.

Quantifying the spread of a sequence x = (x 1 , x 2 , . . . , x k ) (or, similarly, how well x covers a space) is a well-studied concept.

In this section we introduce discrepancy, a quantity used by previous work, and dispersion, which we argue is more appropriate for optimization problems.

Perhaps the most popular way to quantify the spread of a sequence is star discrepancy.

One can interpret the star discrepancy as a multidimensional version of the Kolmogorov-Smirnov statistic between the sequence x and the uniform measure; intuitively, when x contains points which are spread apart, star discrepancy is small.

We include a formal definition in Appendix A.Star discrepancy plays a prominent role in the numerical integration literature, as it provides a sharp bound on the numerical integration error through the the Koksma-Hlawka inequality (given in Appendix B) BID11 .

This has led to wide adoption of low discrepancy sequences, even outside of numerical integration problems.

For example, BID3 analyzed a number of low discrepancy sequences for some optimization tasks and found improved optimization performance over uniform sampling and grid search.

Additionally, low discrepancy sequences such as the Sobol sequence 3 are used as an initialization procedure for some Bayesian optimization schemes BID23 .

Previous work on open loop hyperparameter optimization focused on low discrepancy sequences BID3 BID6 , but optimization performance-how close a point in our sequence is to the true, fixed optimum-is our goal, not a sequence with low discrepancy.

As discrepancy doesn't directly bound optimization error, we turn instead to dispersion DISPLAYFORM0 where ρ is a distance (in our experiments L 2 distance).

Intuitively, the dispersion of a point set is the radius of the largest Euclidean ball containing no points; dispersion measures the worst a point set could be at finding the optimum of a space.

Following BID21 , we can bound the optimization error as follows.

Let f be the function we are aiming to optimize (maximize) with domain B, m(f ) = sup x∈B f (x) be the global optimum of the function, and m k (f ; x) = sup 1≤i≤k f (x i ) be the best-found optimum from the set x. Assuming f is continuous (at least near the global optimum), the modulus of continuity is defined as DISPLAYFORM1 Theorem 1.

BID21 For any point set x with dispersion d k (x), the optimization error is bounded as DISPLAYFORM2 Dispersion can be computed efficiently (unlike discrepancy, D k (x), which is NP-hard BID29 ), and we give an algorithm in Appendix C. Dispersion is at least Ω(k −1/d ), and while low discrepancy implies low BID3 found that the Niederreiter and Halton sequences performed similarly to the Sobol sequence, and that the Sobol sequence outperformed Latin hypercube sampling.

Thus, our experiments include the Sobol sequence (with the Cranley-Patterson rotation) as a representative low-discrepancy sequence.

not hold.

4 Therefore we know that the low-discrepancy sequences evaluated in previous work are also low-dispersion sequences in the big-O sense, but as we will see they may behave quite differently.

Samples drawn uniformly are not low dispersion, as they have rate (ln(k)/k) 1/d BID29 .

DISPLAYFORM3 Optimal dispersion in one dimension is found with an evenly spaced grid, but it's unknown how to get an optimal set in higher dimensions.5 Finding a set of points with the optimal dispersion is as hard as solving the circle packing problem in geometry with k equal-sized circles which are as large as possible.

Dispersion is bounded from below with DISPLAYFORM4 it is unknown if this bound is sharp.

In FIG4 we plot the dispersion of the Sobol sequence, samples drawn uniformly at random, and samples drawn from a k-DPP, in one and two dimensions.

To generate the k-DPP samples, we sequentially drew samples proportional to the (updated) posterior variance (using an RBF kernel, with σ = √ 2/k), as described in Section 2.2.

When d = 1, the regular structure of the Sobol sequence causes it to have increasingly large plateaus, as there are many "holes" of the same size.

For example, the Sobol sequence has the same dispersion for 42 ≤ k ≤ 61, and 84 ≤ k ≤ 125.

Samples drawn from a k-DPP appear to have the same asymptotic rate as the Sobol sequence, but they don't suffer from the plateaus.

When d = 2, the k-DPP samples have lower average dispersion and lower variance.

One other natural surrogate of average optimization performance is to measure the distance from a fixed point, say 2 ) or from the origin, to the nearest point in the length k sequence.

Our experiments (in Appendix D) on these metrics show the k-DPP samples bias samples to the corners of the space, which can be beneficial when the practitioner defined the search space with bounds that are too small.

Note, the low-discrepancy sequences are usually defined only for the [0, 1] d hypecrube, so for hyperparameter search which involves conditional hyperparameters (i.e. those with tree structure) they are not appropriate.

In what follows, we study the k-DPP in more depth and how it performs on real-world hyperparameter tuning problems.

We begin by reviewing DPPs and k-DPPs.

Let B be a domain from which we would like to sample a finite subset.

(In our use of DPPs, this is the set of hyperparameter assignments.)

In general, B could be discrete or continuous; here we assume it is discrete with N values, and we define Y = {1, . . .

, N } to be a a set which indexes B (this index set will be particularly useful in Algorithm 1).

In Section 4.2 we address when B has continuous dimensions.

A DPP defines a probability distribution over 2 Y (all subsets of Y) with the property that two elements of Y are more (less) likely to both be chosen the more dissimilar (similar) they are.

Let random variable Y range over finite subsets of Y.There are several ways to define the parameters of a DPP.

We focus on L-ensembles, which define the probability that a specific subset is drawn (i.e., P (Y = A) for some A ⊂ Y) as: DISPLAYFORM0 As shown in BID18 , this definition of L admits a decomposition to terms representing the quality and diversity of the elements of Y. For any y i , y j ∈ Y, let: DISPLAYFORM1 4 Discrepancy is a global measure which depends on all points, while dispersion only depends on points near the largest "hole".5 In two dimensions a hexagonal tiling finds the optimal dispersion, but this is only valid when k is divisible by the number of columns and rows in the tiling.6 By construction, each individual dimension of the d-dimensional Sobol sequence has these same plateaus.

where q i > 0 is the quality of y i , φ i ∈ R d is a featurized representation of y i , and K : DISPLAYFORM2 is a similarity kernel (e.g. cosine distance). (We will discuss how to featurize hyperparameter settings in Section 4.3.)Here, we fix all q i = 1; in future work, closed loop methods might make use of q i to encode evidence about the quality of particular hyperparameter settings to adapt the DPP's distribution over time.

DPPs have support over all subsets of Y, including ∅ and Y itself.

In many practical settings, one may have a fixed budget that allows running the training algorithm k times, so we require precisely k elements of Y for evaluation.

k-DPPs are distributions over subsets of Y of size k. Thus, DISPLAYFORM0 Sampling from k-DPPs has been well-studied.

When the base set B is a set of discrete items, exact sampling algorithms are known which run in O(N k 3 ) BID18 .

When the base set is a continuous hyperrectangle, a recent exact sampling algorithm was introduced, based on a connection with Gaussian processes (GPs), which runs in O(dk 2 + k 3 ) BID10 .

We are unaware of previous work which allows for sampling from k-DPPs defined over any other base sets.

BID1 present a Metropolis-Hastings algorithm (included here as Algorithm 1) which is a simple and fast alternative to the exact sampling procedures described above.

However, it is restricted to discrete domains.

We propose a generalization of the MCMC algorithm which preserves relevant computations while allowing sampling from any base set from which we can draw uniform samples, including those with discrete dimensions, continuous dimensions, some continuous and some discrete dimensions, or even (conditional) tree structures (Algorithm 2).

To the best of our knowledge, this is the first algorithm which allows for sampling from a k-DPP defined over any space other than strictly continuous or strictly discrete, and thus the first algorithm to utilize the expressive capabilities of the posterior variance of a GP in these regimes.

set Y = Y ∪ {v} \ {u}

with probability p: Y = Y 7: Return B Y Algorithm 1 proceeds as follows: First, initialize a set Y with k indices of L, drawn uniformly.

Then, at each iteration, sample two indices of L (one within and one outside of the set Y), and with some probability replace the item in Y with the other.

When we have continuous dimensions in the base set, however, we can't define the matrix L, so sampling indices from it is not possible.

We propose Algorithm 2, which samples points directly from the base set B instead (assuming continuous dimensions are bounded), and computes only the principal minors of L needed for the relevant computations on the fly.

Algorithm 2 Drawing a sample from a k-DPP defined over a space with continuous and discrete dimensions Input: A base set B with some continuous and some discrete dimensions, a quality function Ψ : compute the quality score for each item, q i = Ψ(β i ), ∀i, and DISPLAYFORM0 DISPLAYFORM1 with probability p: β = β 10: Return β Even in the case where the dimensions of B are discrete, Algorithm 2 requires less computation and space than Algorithm 1 (assuming the quality and similarity scores are stored once computed, and retrieved when needed).

Previous analyses claimed that Algorithm 1 should mix after O(N log(N )) steps.

There are O(N 2 ) computations required to compute the full matrix L, and at each iteration we will compute at most O(k) new elements of L, so even in the worst case we will save space and computation whenever k log(N ) < N .

In expectation, we will save significantly more.

Let φ i be a feature vector for y i ∈ Y, a modular encoding of the attribute-value mapping assigning values to different hyperparameters, in which fixed segments of the vector are assigned to each hyperparameter attribute (e.g., the dropout rate, the choice of nonlinearity, etc.).

For a hyperparameter that takes a numerical value in range [h min , h max ], we encode value h using one dimension (j) of φ and project into the range [0, 1]: , and hence label our approach k-DPP-RBF.

Values for σ 2 lead to models with different properties; when σ 2 is small, points that are spread out interact little with one another, and when σ 2 is large, the increased repulsion between the points encourages them to be as far apart as possible.

DISPLAYFORM0

Many real-world hyperparameter search spaces are tree-structured.

For example, the number of layers in a neural network is a hyperparameter, and each additional layer adds at least one new hyperparameter which ought to be tuned (the number of nodes in that layer).

For a binary hyperparameter like whether or not to use regularization, we use a one-hot encoding.

When this hyperparameter is "on," we set the associated regularization strength as above, and when it is "off" we set it to zero.

Intuitively, with all other hyperparameter settings equal, this causes the off-setting to be closest to the least strong regularization.

One can also treat higher-level design decisions as hyperparameters BID17 , such as whether to train a logistic regression classifier, a convolutional neural network, or a recurrent neural network.

In this construction, the type of model would be a categorical variable (and thus get a one-hot encoding), and all child hyperparameters for an "off" model setting (such as the convergence tolerance for logistic regression, when training a recurrent neural network) would be set to zero.

In this section we present our hyperparameter optimization experiments.

Our experiments consider a setting where hyperparameters have a large effect on performance: a convolutional neural network for text classification BID16 .

The task is binary sentiment analysis on the Stanford sentiment treebank BID25 .

On this balanced dataset, random guessing leads to 50% accuracy.

We use the CNN-non-static model from BID16 , with skip-gram BID20 vectors.

The model architecture consists of a convolutional layer, a max-over-time pooling layer, then a fully connected layer leading to a softmax.

All k-DPP samples are drawn using Algorithm 2.

We begin with a search over three continuous hyperparameters and one binary hyperparameter, with a simple tree structure: the binary hyperparameter indicates whether or not the model will use L 2 regularization, and one of the continuous hyperparameters is the regularization strength.

We assume a budget of k = 20 evaluations by training the convolutional neural net.

L 2 regularization strengths in the range [e −5 , e −1 ] (or no regularization) and dropout rates in [0.0, 0.7] are considered.

We consider three increasingly "easy" ranges for the learning rate:• Hard: [e −5 , e 5 ], where the majority of the range leads to accuracy no better than chance.• Medium: [e −5 , e −1 ], where half of the range leads to accuracy no better than chance.• Easy: [e −10 , e −3 ], where the entire range leads to models that beat chance.

FIG7 shows the accuracy (averaged over 50 runs) of the best model found after exploring 1, 2, . . .

, k hyperparameter settings.

We see that k-DPP-RBF finds better models with fewer iterations necessary than the other approaches, especially in the most difficult case.

FIG7 compares the sampling methods against a Bayesian optimization technique using a tree-structured Parzen estimator (BO-TPE; BID5 .

This technique evaluates points sequentially, allowing the model to choose the next point based on how well previous points performed (a closed loop approach).

It is state-of-the-art on tree-structured search spaces (though its sequential nature limits parallelization).

Surprisingly, we find it performs the worst, even though it takes advantage of additional information.

We hypothesize that the exploration/exploitation tradeoff in BO-TPE causes it to commit to more local search before exploring the space fully, thus not finding hard-to-reach global optima.

Note that when considering points sampled uniformly or from a DPP, the order of the k hyperparameter settings in one trial is arbitrary (though this is not the case with BO-TPE as it is an iterative algorithm).

In all cases the variance of the best of the k points is lower than when sampled uniformly, and the differences in the plots are all significant with p < 0.01.

BID28 analyzed the stability of convolutional neural networks for sentence classification with respect to a large set of hyperparameters, and found a set of six which they claimed had the largest impact: the number of kernels, the difference in size between the kernels, the size of each kernel, dropout, regularization strength, and the number of filters.

We optimized over their prescribed "Stable" ranges for three open loop methods and one closed loop method; average accuracies with 95 percent confidence intervals from 50 trials of hyperparameter optimization are shown in Figure 3 , across k = 5, 10, 15, 20 iterations.

We find that even when optimizing over a space for which all values lead to good models, k-DPP-RBF outperforms the other methods.

Our experiments reveal that, while the hyperparameters proposed by BID28 , can have an effect, the learning rate, which they do not analyze, is at least as impactful.

Here we compare our approach against Spearmint BID23 , perhaps the most popular Bayesian optimization package.

Figure 4 shows wall clock time and accuracy for 25 runs on the "Stable" search space of four hyperparameter optimization approaches: k-DPP-RBF (with k = 20), batch Spearmint with 2 iterations of batch size 10, batch Spearmint with 10 iterations of batch size 2, and sequential Spearmint 7 .

Each point in the plot is one hyperparameter assignment evaluation.

The vertical lines represent how long, on average, it takes to find the best result in one run.

We see that all evaluations for k-DPP-RBF finish quickly, while even the fastest batch method (2 batches of size 10) takes nearly twice as long on average to find a good result.

The final average best-found accuracies are 82.61 for k-DPP-RBF, 82.65 for Spearmint with 2 batches of size 10, 82.7 for Spearmint with 10 batches of size 2, and 82.76 for sequential Spearmint.

Thus, we find it takes on average more than ten times as long for sequential Spearmint to find its best solution, for a gain of only 0.15 percent accuracy.

We have explored open loop hyperparameter optimization built on sampling from a k-DPP.

We described how to define a k-DPP over hyperparameter search spaces, and showed that k-DPPs retain the attractive parallelization capabilities of random search.

In synthetic experiments, we showed k-DPP samples perform well on a number of important metrics, even for large values of k. In hyperprameter optimization experiments, we see k-DPP-RBF outperform other open loop methods.

Additionally, we see that sequential methods, even when using more than ten times as much wall clock time, gain less than 0.16 percent accuracy on a particular hyperparameter optimization problem.

An open-source implementation of our method is available.

A STAR DISCREPANCY DISPLAYFORM0 It is well-known that a sequence chosen uniformly at random from [0, 1] d has an expected star discrepancy of at least 1 k (and is no greater than DISPLAYFORM1 ) BID22 whereas sequences are known to exist with star discrepancy less than BID24 , where both bounds depend on absolute constants.

DISPLAYFORM2 Comparing the star discrepancy of sampling uniformly and Sobol, the bounds suggest that as d grows large relative to k, Sobol starts to suffer.

Indeed, BID2 notes that the Sobol rate is not even valid until k = Ω(2 d ) which motivates them to study a formulation of a DPP that has a star discrepancy between Sobol and random and holds for all k, small and large.

They primarily approached this problem from a theoretical perspective, and didn't include experimental results.

Their work, in part, motivates us to look at DPPs as a solution for hyperparameter optimization.

B KOKSMA-HLAWKA INEQUALITY Let B be the d-dimensional unit cube, and let f have bounded Hardy and Krause variation V ar HK (f ) on B. Let x = (x 1 , x 2 , . . .

, x k ) be a set of points in B at which the function f will be evaluated to approximate an integral.

The Koksma-Hlawka inequality bounds the numerical integration error by the product of the star discrepancy and the variation: DISPLAYFORM3 We can see that for a given f , finding x with low star discrepancy can improve numerical integration approximations.

Find a (bounded) voronoi diagram over the search space for a point set X k .

For each vertex in the voronoi diagram, find the closest point in X k .

The dispersion is the max over these distances.

One natural surrogate of average optimization performance is to define a hyperparameter space on [0, 1] d and measure the distance from a fixed point, say is motivated by a quadratic Taylor series approximation around the minimum of the hypothetical function we wish to minimize.

In the first columns of Figure 5 we plot the smallest distance from the center 1 2 1, as a function of the length of the sequence (in one dimension) for the Sobol sequence, uniform at random, and a DPP.

We observe all methods appear comparable when it comes to distance to the center.

of what low discrepancy sequences attempt to do.

While Sobol and uniformly random sequences will not bias themselves towards the corners, a DPP does.

This happens because points from a DPP are sampled according to how distant they are from the existing points; this tends to favor points in the corners.

This same behavior of sampling in the corners is also very common for Bayesian optimization schemes, which is not surprise due to the known connections between sampling from a DPP and Gaussian processes (see Section 2.2).

In the second column of Figure 5 we plot the distance to the origin which is just an arbitrarily chosen corner of hypercube.

As expected, we observe that the DPP tends to outperform uniform at random and Sobol in this metric.

Figure 5 : Comparison of the Sobol sequence, samples a from k-DPP, and uniform random for two metrics of interest.

These log-log plots show uniform sampling and k-DPP-RBF performs comparably to the Sobol sequence in terms of distance to the center, but on another (distance to the origin) k-DPP-RBF samples outperform the Sobol sequence and uniform sampling.

@highlight

We address fully parallel hyperparameter optimization with Determinantal Point Processes. 
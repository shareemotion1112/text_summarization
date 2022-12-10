Sorting input objects is an important step in many machine learning pipelines.

However, the sorting operator is non-differentiable with respect to its inputs, which prohibits end-to-end gradient-based optimization.

In this work, we propose NeuralSort, a general-purpose continuous relaxation of the output of the sorting operator from permutation matrices to the set of unimodal row-stochastic matrices, where every row sums to one and has a distinct argmax.

This relaxation permits straight-through optimization of any computational graph involve a sorting operation.

Further, we use this relaxation to enable gradient-based stochastic optimization over the combinatorially large space of permutations by deriving a reparameterized gradient estimator for the Plackett-Luce family of distributions over permutations.

We demonstrate the usefulness of our framework on three tasks that require learning semantic orderings of high-dimensional objects, including a fully differentiable, parameterized extension of the k-nearest neighbors algorithm

Learning to automatically sort objects is useful in many machine learning applications, such as topk multi-class classification BID5 , ranking documents for information retrieval (Liu et al., 2009) , and multi-object target tracking in computer vision BID3 .

Such algorithms typically require learning informative representations of complex, high-dimensional data, such as images, before sorting and subsequent downstream processing.

For instance, the k-nearest neighbors image classification algorithm, which orders the neighbors based on distances in the canonical pixel basis, can be highly suboptimal for classification (Weinberger et al., 2006) .

Deep neural networks can instead be used to learn representations, but these representations cannot be optimized end-to-end for a downstream sorting-based objective, since the sorting operator is not differentiable with respect to its input.

In this work, we seek to remedy this shortcoming by proposing NeuralSort, a continuous relaxation to the sorting operator that is differentiable almost everywhere with respect to the inputs.

The output of any sorting algorithm can be viewed as a permutation matrix, which is a square matrix with entries in {0, 1} such that every row and every column sums to 1.

Instead of a permutation matrix, NeuralSort returns a unimodal row-stochastic matrix.

A unimodal row-stochastic matrix is defined as a square matrix with positive real entries, where each row sums to 1 and has a distinct arg max.

All permutation matrices are unimodal row-stochastic matrices.

NeuralSort has a temperature knob that controls the degree of approximation, such that in the limit of zero temperature, we recover a permutation matrix that sorts the inputs.

Even for a non-zero temperature, we can efficiently project any unimodal matrix to the desired permutation matrix via a simple row-wise arg max operation.

Hence, NeuralSort is also suitable for efficient straight-through gradient optimization BID4 , which requires "exact" permutation matrices to evaluate learning objectives.

As the second primary contribution, we consider the use of NeuralSort for stochastic optimization over permutations.

In many cases, such as latent variable models, the permutations may be latent but directly influence observed behavior, e.g., utility and choice models are often expressed as distributions over permutations which govern the observed decisions of agents (Regenwetter et al., 2006; BID7 .

By learning distributions over unobserved permutations, we can account for the uncertainty in these permutations in a principled manner.

However, the challenge with stochastic optimization over discrete distributions lies in gradient estimation with respect to the distribution parameters.

Vanilla REINFORCE estimators are impractical for most cases, or necessitate custom control variates for low-variance gradient estimation (Glasserman, 2013) .In this regard, we consider the Plackett-Luce (PL) family of distributions over permutations (Plackett, 1975; Luce, 1959) .

A common modeling choice for ranking models, the PL distribution is parameterized by n scores, with its support defined over the symmetric group consisting of n! permutations.

We derive a reparameterizable sampler for stochastic optimization with respect to this distribution, based on Gumbel perturbations to the n (log-)scores.

However, the reparameterized sampler requires sorting these perturbed scores, and hence the gradients of a downstream learning objective with respect to the scores are not defined.

By using NeuralSort instead, we can approximate the objective and obtain well-defined reparameterized gradient estimates for stochastic optimization.

Finally, we apply NeuralSort to tasks that require us to learn semantic orderings of complex, highdimensional input data.

First, we consider sorting images of handwritten digits, where the goal is to learn to sort images by their unobserved labels.

Our second task extends the first one to quantile regression, where we want to estimate the median (50-th percentile) of a set of handwritten numbers.

In addition to identifying the index of the median image in the sequence, we need to learn to map the inferred median digit to its scalar representation.

In the third task, we propose an algorithm that learns a basis representation for the k-nearest neighbors (kNN) classifier in an end-to-end procedure.

Because the choice of the k nearest neighbors requires a non-differentiable sorting, we use NeuralSort to obtain an approximate, differentiable surrogate.

On all tasks, we observe significant empirical improvements due to NeuralSort over the relevant baselines and competing relaxations to permutation matrices.

An n-dimensional permutation z = [z 1 , z 2 , . . .

, z n ]T is a list of unique indices {1, 2, . . .

, n}. Every permutation z is associated with a permutation matrix P z ∈ {0, 1} n×n with entries given as: DISPLAYFORM0 Let Z n denote the set of all n!

possible permutations in the symmetric group.

We define the sort : R n → Z n operator as a mapping of n real-valued inputs to a permutation corresponding to a descending ordering of these inputs.

E.g., if the input vector s = [9, 1, 5, 2] T , then sort(s) = [1, 3, 4, 2] T since the largest element is at the first index, second largest element is at the third index and so on.

In case of ties, elements are assigned indices in the order they appear.

We can obtain the sorted vector simply via P sort(s)

s.

The family of Plackett-Luce distributions over permutations is best described via a generative process: Consider a sequence of n items, each associated with a canonical index i = 1, 2, . . .

, n. A common assumption in ranking models is that the underlying generating process for any observed permutation of n items satisfies Luce's choice axiom (Luce, 1959) .

Mathematically, this axiom defines the 'choice' probability of an item with index i as: q(i) ∝ s i where s i > 0 is interpreted as the score of item with index i.

The normalization constant is given by Z = i∈{1,2,...,n} s i .If we choose the n items one at a time (without replacement) based on these choice probabilities, we obtain a discrete distribution over all possible permutations.

This distribution is referred to as the Plackett-Luce (PL) distribution, and its probability mass function for any z ∈ Z n is given by: DISPLAYFORM0 where s = {s 1 , s 2 , . . .

, s n } is the vector of scores parameterizing this distribution (Plackett, 1975) .

Figure 1 : Stochastic computation graphs with a deterministic node z corresponding to the output of a sort operator applied to the scores s. DISPLAYFORM1

The abstraction of stochastic computation graphs (SCG) compactly specifies the forward value and the backward gradient computation for computational circuits.

An SCG is a directed acyclic graph that consists of three kinds of nodes: input nodes which specify external inputs (including parameters), deterministic nodes which are deterministic functions of their parents, and stochastic nodes which are distributed conditionally on their parents.

See Schulman et al. (2015) for a review.

To define gradients of an objective function with respect to any node in the graph, the chain rule necessitates that the gradients with respect to the intermediate nodes are well-defined.

This is not the case for the sort operator.

In Section 3, we propose to extend stochastic computation graphs with nodes corresponding to a relaxation of the deterministic sort operator.

In Section 4, we further use this relaxation to extend computation graphs to include stochastic nodes corresponding to distributions over permutations.

The proofs of all theoretical results in this work are deferred to Appendix B.

Our goal is to optimize training objectives involving a sort operator with gradient-based methods.

Consider the optimization of objectives written in the following form: DISPLAYFORM0 (2) where z = sort(s).Here, s ∈ R n denotes a vector of n real-valued scores, z is the permutation that (deterministically) sorts the scores s, and f (·) is an arbitrary function of interest assumed to be differentiable w.r.t a set of parameters θ and z. For example, in a ranking application, these scores could correspond to the inferred relevances of n webpages and f (·) could be a ranking loss.

Figure 1 shows the stochastic computation graph corresponding to the objective in Eq. 2.

We note that this could represent part of a more complex computation graph, which we skip for ease of presentation while maintaining the generality of the scope of this work.

While the gradient of the above objective w.r.t.

θ is well-defined and can be computed via standard backpropogation, the gradient w.r.t.

the scores s is not defined since the sort operator is not differentiable w.r.t.

s.

Our solution is to derive a relaxation to the sort operator that leads to a surrogate objective with well-defined gradients.

In particular, we seek to use such a relaxation to replace the permutation matrix P z in Eq. 2 with an approximation P z such that the surrogate objective f ( P z ; θ) is differentiable w.r.t.

the scores s.

The general recipe to relax non-differentiable operators with discrete codomains N is to consider differentiable alternatives that map the input to a larger continuous codomain M with desirable properties.

For gradient-based optimization, we are interested in two key properties:1.

The relaxation is continuous everywhere and differentiable (almost-)everywhere with respect to elements in the input domain.

2.

There exists a computationally efficient projection from M back to N .Relaxations satisfying the first requirement are amenable to automatic differentiation for optimizing stochastic computational graphs.

The second requirement is useful for evaluating metrics and losses that necessarily require a discrete output akin to the one obtained from the original, non-relaxed operator.

E.g., in straight-through gradient estimation BID4 Jang et al., 2017) , the Figure 2 : Center: Venn Diagram relationships between permutation matrices (P), doubly-stochastic matrices (D), unimodal row stochastic matrices (U), and row stochastic matrices (R).

Left: A doubly-stochastic matrix that is not unimodal.

Right: A unimodal matrix that is not doublystochastic.

DISPLAYFORM1 non-relaxed operator is used for evaluating the learning objective in the forward pass and the relaxed operator is used in the backward pass for gradient estimation.

The canonical example is the 0/1 loss used for binary classification.

While the 0/1 loss is discontinuous w.r.t.

its inputs (real-valued predictions from a model), surrogates such as the logistic and hinge losses are continuous everywhere and differentiable almost-everywhere (property 1), and can give hard binary predictions via thresholding (property 2).Note: For brevity, we assume that the arg max operator is applied over a set of elements with a unique maximizer and hence, the operator has well-defined semantics.

With some additional bookkeeping for resolving ties, the results in this section hold even if the elements to be sorted are not unique.

See Appendix C.Unimodal Row Stochastic Matrices.

The sort operator maps the input vector to a permutation, or equivalently a permutation matrix.

Our relaxation to sort is motivated by the geometric structure of permutation matrices.

The set of permutation matrices is a subset of doubly-stochastic matrices, i.e., a non-negative matrix such that the every row and column sums to one.

If we remove the requirement that every column should sum to one, we obtain a larger set of row stochastic matrices.

In this work, we propose a relaxation to sort that maps inputs to an alternate subset of row stochastic matrices, which we refer to as the unimodal row stochastic matrices.

Definition 1 (Unimodal Row Stochastic Matrices).

An n × n matrix is Unimodal Row Stochastic if it satisfies the following conditions: DISPLAYFORM2 3.

Argmax Permutation: Let u denote an n-dimensional vector with entries such that u i = arg max j U [i, j] ∀i ∈ {1, 2, . . .

, n}. Then, u ∈ Z n , i.e., it is a valid permuation.

We denote U n as the set of n × n unimodal row stochastic matrices.

All row stochastic matrices satisfy the first two conditions.

The third condition is useful for gradient based optimization involving sorting-based losses.

The condition provides a straightforward mechanism for extracting a permutation from a unimodal row stochastic matrix via a row-wise arg max operation.

Figure 2 shows the relationships between the different subsets of square matrices.

NeuralSort.

Our relaxation to the sort operator is based on a standard identity for evaluating the sum of the k largest elements in any input vector. , we have the sum of the k-largest elements given as: DISPLAYFORM3 The identity in Lemma 2 outputs the sum of the top-k elements.

The k-th largest element itself can be recovered by taking the difference of the sum of top-k elements and the top-(k − 1) elements.

T be a real-valued vector of length n. Let A s denote the matrix of absolute pairwise differences of the elements of s such that A s [i, j] = |s i − s j |.

The permutation matrix P sort(s) corresponding to sort(s) is given by: DISPLAYFORM4 where 1 denotes the column vector of all ones.

E.g., if we set i = (n + 1)/2 then the non-zero entry in the i-th row P sort(s) [i, :] corresponds to the element with the minimum sum of (absolute) distance to the other elements.

As desired, this corresponds to the median element.

The relaxation requires O(n 2 ) operations to compute A s , as opposed to the O(n log n) overall complexity for the best known sorting algorithms.

In practice however, it is highly parallelizable and can be implemented efficiently on GPU hardware.

The arg max operator is non-differentiable which prohibits the direct use of Corollary 3 for gradient computation.

Instead, we propose to replace the arg max operator with soft max to obtain a continuous relaxation P sort(s) (τ ).

In particular, the i-th row of P sort(s) (τ ) is given by: DISPLAYFORM5 where τ > 0 is a temperature parameter.

Our relaxation is continuous everywhere and differentiable almost everywhere with respect to the elements of s. Furthermore, we have the following result.

Theorem 4.

Let P sort(s) denote the continuous relaxation to the permutation matrix P sort(s) for an arbitrary input vector s and temperature τ defined in Eq. 5.

Then, we have:1.

Unimodality: ∀τ > 0, P sort(s) is a unimodal row stochastic matrix.

Further, let u denote the permutation obtained by applying arg max row-wise to P sort(s) .

Then, u = sort(s).

If we assume that the entries of s are drawn independently from a distribution that is absolutely continuous w.r.t.

the Lebesgue measure in R, then the following convergence holds almost surely: DISPLAYFORM0 Unimodality allows for efficient projection of the relaxed permutation matrix P sort(s) to the hard matrix P sort(s) via a row-wise arg max, e.g., for straight-through gradients.

For analyzing limiting behavior, independent draws ensure that the elements of s are distinct almost surely.

The temperature τ controls the degree of smoothness of our approximation.

At one extreme, the approximation becomes tighter as the temperature is reduced.

In practice however, the trade-off is in the variance of these estimates, which is typically lower for larger temperatures.

In many scenarios, we would like the ability to express our uncertainty in inferring a permutation e.g., latent variable models with latent nodes corresponding to permutations.

Random variables that assume values corresponding to permutations can be represented via stochastic nodes in the stochastic computation graph.

For optimizing the parameters of such a graph, consider the following class of objectives: DISPLAYFORM0 where θ and s denote sets of parameters, P z is the permutation matrix corresponding to the permutation z, q(·) is a parameterized distribution over the elements of the symmetric group Z n , and f (·) is an arbitrary function of interest assumed to be differentiable in θ and z. The SCG is shown in FIG0 .

In contrast to the SCG considered in the previous section (Figure 1 ), here we are dealing with a distribution over permutations as opposed to a single (deterministically computed) one.

While such objectives are typically intractable to evaluate exactly since they require summing over a combinatorially large set, we can obtain unbiased estimates efficiently via Monte Carlo.

Monte Carlo estimates of gradients w.r.t.

θ can be derived simply via linearity of expectation.

However, the gradient estimates w.r.t.

s cannot be obtained directly since the sampling distribution depends on s. The REINFORCE gradient estimator (Glynn, 1990; Williams, 1992; Fu, 2006) uses the fact that ∇ s q(z|s) = q(z|s)∇ s log q(z|s) to derive the following Monte Carlo gradient estimates: DISPLAYFORM1

REINFORCE gradient estimators typically suffer from high variance (Schulman et al., 2015; Glasserman, 2013) .

Reparameterized samplers provide an alternate gradient estimator by expressing samples from a distribution as a deterministic function of its parameters and a fixed source of randomness (Kingma & Welling, 2014; Rezende et al., 2014; Titsias & Lázaro-Gredilla, 2014) .

Since the randomness is from a fixed distribution, Monte Carlo gradient estimates can be derived by pushing the gradient operator inside the expectation (via linearity).

In this section, we will derive a reparameterized sampler and gradient estimator for the Plackett-Luce (PL) family of distributions.

Let the score s i for an item i ∈ {1, 2, . . .

, n} be an unobserved random variable drawn from some underlying score distribution (Thurstone, 1927) .

Now for each item, we draw a score from its corresponding score distribution.

Next, we generate a permutation by applying the deterministic sort operator to these n randomly sampled scores.

Interestingly, prior work has shown that the resulting distribution over permutations corresponds to a PL distribution if and only if the scores are sampled independently from Gumbel distributions with identical scales.

Proposition 5.[adapted from Yellott Jr (1977)] Let s be a vector of scores for the n items.

For each item i, sample g i ∼ Gumbel(0, β) independently with zero mean and a fixed scale β.

Lets denote the vector of Gumbel perturbed log-scores with entries such thats i = β log s i + g i .

Then: DISPLAYFORM0 For ease of presentation, we assume β = 1 in the rest of this work.

Proposition 5 provides a method for sampling from PL distributions with parameters s by adding Gumbel perturbations to the logscores and applying the sort operator to the perturbed log-scores.

This procedure can be seen as a reparameterization trick that expresses a sample from the PL distribution as a deterministic function of the scores and a fixed source of randomness ( FIG0 ).

Letting g denote the vector of i.i.d.

Gumbel perturbations, we can express the objective in Eq. 7 as: DISPLAYFORM1 (10) While the reparameterized sampler removes the dependence of the expectation on the parameters s, it introduces a sort operator in the computation graph such that the overall objective is nondifferentiable in s. In order to obtain a differentiable surrogate, we approximate the objective based on the NeuralSort relaxation to the sort operator: DISPLAYFORM2 Accordingly, we get the following reparameterized gradient estimates for the approximation: DISPLAYFORM3 which can be estimated efficiently via Monte Carlo because the expectation is with respect to a distribution that does not depend on s.

The problem of learning to rank documents based on relevance has been studied extensively in the context of information retrieval.

In particular, listwise approaches learn functions that map objects to scores.

Much of this work concerns the PL distribution: the RankNet algorithm BID6 can be interpreted as maximizing the PL likelihood of pairwise comparisons between items, while the ListMLE ranking algorithm in Xia et al. (2008) extends this with a loss that maximizes the PL likelihood of ground-truth permutations directly.

The differentiable pairwise approaches to ranking, such as Rigutini et al. FORMULA1 , learn to approximate the comparator between pairs of objects.

Our work considers a generalized setting where sorting based operators can be inserted anywhere in computation graphs to extend traditional pipelines e.g., kNN.Prior works have proposed relaxations of permutation matrices to the Birkhoff polytope, which is defined as the convex hull of the set of permutation matrices a.k.a.

the set of doubly-stochastic matrices.

A doubly-stochastic matrix is a permutation matrix iff it is orthogonal and continuous relaxations based on these matrices have been used previously for solving NP-complete problems such as seriation and graph matching (Fogel et al., 2013; Fiori et al., 2013; Lim & Wright, 2014) .

BID1 proposed the use of the Sinkhorn operator to map any square matrix to the Birkhoff polytope.

They interpret the resulting doubly-stochastic matrix as the marginals of a distribution over permutations.

Mena et al. (2018) propose an alternate method where the square matrix defines a latent distribution over the doubly-stochastic matrices themselves.

These distributions can be sampled from by adding elementwise Gumbel perturbations.

Linderman et al. FORMULA1 propose a rounding procedure that uses the Sinkhorn operator to directly sample matrices near the Birkhoff polytope.

Unlike Mena et al. (2018) , the resulting distribution over matrices has a tractable density.

In practice, however, the approach of Mena et al. FORMULA1 performs better and will be the main baseline we will be comparing against in our experiments in Section 6.As discussed in Section 3, NeuralSort maps permutation matrices to the set of unimodal rowstochastic matrices.

For the stochastic setting, the PL distribution permits efficient sampling, exact and tractable density estimation, making it an attractive choice for several applications, e.g., variational inference over latent permutations.

Our reparameterizable sampler, while also making use of the Gumbel distribution, is based on a result unique to the PL distribution (Proposition 5).The use of the Gumbel distribution for defining continuous relaxations to discrete distributions was first proposed concurrently by Jang et al. FORMULA1 and Maddison et al. (2017) for categorical variables, referred to as Gumbel-Softmax.

The number of possible permutations grow factorially with the dimension, and thus any distribution over n-dimensional permutations can be equivalently seen as a distribution over n! categories.

Gumbel-softmax does not scale to a combinatorially large number of categories (Kim et al., 2016; Mussmann et al., 2017) , necessitating the use of alternate relaxations, such as the one considered in this work.

We refer to the two approaches proposed in Sections 3, 4 as Deterministic NeuralSort and Stochastic NeuralSort, respectively.

For additional hyperparameter details and analysis, see Appendix D.

Dataset.

We first create the large-MNIST dataset, which extends the MNIST dataset of handwritten digits.

The dataset consists of multi-digit images, each a concatenation of 4 randomly selected individual images from MNIST, e.g., is one such image in this dataset.

Each image is associated with a real-valued label, which corresponds to its concatenated MNIST labels, e.g., the label of is 1810.

Using the large-MNIST dataset, we finally create a dataset of sequences.

Every sequence is this dataset consists of n randomly sampled large-MNIST images.

Setup.

Given a dataset of sequences of large-MNIST images, our goal is to learn to predict the permutation that sorts the labels of the sequence of images, given a training set of ground-truth permutations.

Figure 4 (Task 1) illustrates this task on an example sequence of n = 5 large-MNIST images.

This task is a challenging extension of the one considered by Mena et al. (2018) in sorting scalars, since it involves learning the semantics of high-dimensional objects prior to sorting.

A DISPLAYFORM0 Task 1: Sorting Loss ([3, 5, 1, 4, 2] T , z) DISPLAYFORM1 Figure 4: Sorting and quantile regression.

The model is trained to sort sequences of n = 5 large-MNIST images x 1 , x 2 , . . .

, x 5 (Task 1) and regress the median value (Task 2).

In the above example, the ground-truth permutation that sorts the input sequence from largest to smallest is [3, 5, 1, 4, 2] T , 9803 being the largest and 1270 the smallest.

Blue illustrates the true median image x 1 with ground-truth sorted index 3 and value 2960.

good model needs to learn to dissect the individual digits in an image, rank these digits, and finally, compose such rankings based on the digit positions within an image.

The available supervision, in the form of the ground-truth permutation, is very weak compared to a classification setting that gives direct access to the image labels.

Baselines.

All baselines use a CNN that is shared across all images in a sequence to map each large-MNIST image to a feature space.

The vanilla row-stochastic (RS) baseline concatenates the CNN representations for n images into a single vector that is fed into a multilayer perceptron that outputs n multiclass predictions of the image probabilities for each rank.

The Sinkhorn and GumbelSinkhorn baselines, as discussed in Section 5, use the Sinkhorn operator to map the stacked CNN representations for the n objects into a doubly-stochastic matrix.

For all methods, we minimized the cross-entropy loss between the predicted matrix and the ground-truth permutation matrix.

Following Mena et al. (2018) , our evaluation metric is the the proportion of correctly predicted permutations on a test set of sequences.

Additionally, we evaluate the proportion of individual elements ranked correctly.

TAB1 demonstrates that the approaches based on the proposed sorting relaxation significantly outperform the baseline approaches for all n considered.

The performance of the deterministic and stochastic variants are comparable.

The vanilla RS baseline performs well in ranking individual elements, but is not good at recovering the overall square matrix.

We believe the poor performance of the Sinkhorn baselines is partly because these methods were designed and evaluated for matchings.

Like the output of sort, matchings can also be represented as permutation matrices.

However, distributions over matchings need not satisfy Luce's choice axiom or imply a total ordering, which could explain the poor performance on the tasks considered.

Setup.

In this experiment, we extend the sorting task to regression.

Again, each sequence contains n large-MNIST images, and the regression target for each sequence is the 50-th quantile (i.e., the median) of the n labels of the images in the sequence.

Figure 4 (Task 2) illustrates this task on an example sequence of n = 5 large-MNIST images, where the goal is to output the third largest label.

The design of this task highlights two key challenges since it explicitly requires learning both a suitable representation for sorting high-dimensional inputs and a secondary function that approximates the label itself (regression).

Again, the supervision available in the form of the label of only a single image at an arbitrary and unknown location in the sequence is weak.

Baselines.

In addition to Sinkhorn and Gumbel-Sinkhorn, we design two more baselines.

The Constant baseline always returns the median of the full range of possible outputs, ignoring the input sequence.

This corresponds to 4999.5 since we are sampling large-MNIST images uniformly in the range of four-digit numbers.

The vanilla neural net (NN) baseline directly maps the input sequence of images to a real-valued prediction for the median.

DISPLAYFORM0 Results.

Our evaluation metric is the mean squared error (MSE) and R 2 on a test set of sequences.

Results for n = {5, 9, 15} images are shown in TAB2 .

The Vanilla NN baseline while incurring a large MSE, is competitive on the R 2 metric.

The other baselines give comparable performance on the MSE metric.

The proposed NeuralSort approaches outperform the competing methods on both the metrics considered.

The stochastic NeuralSort approach is the consistent best performer on MSE, while the deterministic NeuralSort is slightly better on the R 2 metric.

Setup.

In this experiment, we design a fully differentiable, end-to-end k-nearest neighbors (kNN) classifier.

Unlike a standard kNN classifier which computes distances between points in a predefined space, we learn a representation of the data points before evaluating the k-nearest neighbors.

We are given access to a dataset D of (x, y) pairs of standard input data and their class labels respectively.

The differentiable kNN algorithm consists of two hyperparameters: the number of training neighbors n, the number of top candidates k, and the sorting temperature τ .

Every sequence of items here consists of a query point x and a randomly sampled subset of n candidate nearest neighbors from the training set, say {x 1 , x 2 , . . .

, x n }.

In principle, we could use the entire training set (excluding the query point) as candidate points, but this can hurt the learning both computationally and statistically.

The query points are randomly sampled from the train/validation/test sets as appropriate but the nearest neighbors are always sampled from the training set.

The loss function optimizes for a representation space h φ (·) (e.g., CNN) such that the top-k candidate points with the minimum Euclidean distance to the query point in the representation space have the same label as the query point.

Note that at test time, once the representation space h φ is learned, we can use the entire training set as the set of candidate points, akin to a standard kNN classifier.

Figure 5 illustrates the proposed algorithm.

Formally, for any datapoint x, let z denote a permutation of the n candidate points.

The uniformlyweighted kNN loss, denoted as kNN (·), can be written as follows: where {y 1 , y 2 , . . .

, y n } are the labels for the candidate points.

Note that when P z is an exact permutation matrix (i.e., temperature τ → 0), this expression is exactly the negative of the fraction of k nearest neighbors that have the same label as x. DISPLAYFORM0 Using Eq. 13, the training objectives for Deterministic and Stochastic NeuralSort are given as: DISPLAYFORM1 Stochastic: min DISPLAYFORM2 where each entry of s is given as DISPLAYFORM3 Datasets.

We consider three benchmark datasetes: MNIST dataset of handwritten digits, Fashion-MNIST dataset of fashion apparel, and the CIFAR-10 dataset of natural images (no data augmentation) with the canonical splits for training and testing.

Baselines.

We consider kNN baselines that operate in three standard representation spaces: the canonical pixel basis, the basis specified by the top 50 principal components (PCA), an autonencoder (AE).

Additionally, we experimented with k = 1, 3, 5, 9 nearest neighbors and across two distance metrics: uniform weighting of all k-nearest neighbors and weighting nearest neighbors by the inverse of their distance.

For completeness, we trained a CNN with the same architecture as the one used for NeuralSort (except the final layer) using the cross-entropy loss.

Results.

We report the classification accuracies on the standard test sets in TAB3 .

On both datasets, the differentiable kNN classifier outperforms all the baseline kNN variants including the convolutional autoencoder approach.

The performance is much closer to the accuracy of a standard CNN.

In this paper, we proposed NeuralSort, a continuous relaxation of the sorting operator to the set of unimodal row-stochastic matrices.

Our relaxation facilitates gradient estimation on any computation graph involving a sort operator.

Further, we derived a reparameterized gradient estimator for the Plackett-Luce distribution for efficient stochastic optimization over permutations.

On three illustrative tasks including a fully differentiable k-nearest neighbors, our proposed relaxations outperform prior work in end-to-end learning of semantic orderings of high-dimensional objects.

In the future, we would like to explore alternate relaxations to sorting as well as applications that extend widely-used algorithms such as beam search (Goyal et al., 2018) .

Both deterministic and stochastic NeuralSort are easy to implement.

We provide reference implementations in Tensorflow BID0 Proof.

For any value of λ, the following inequalities hold: DISPLAYFORM0 This finishes the proof.

Proof.

We first consider at exactly what values of λ the sum in Lemma 2 is minimized.

For simplicity we will only prove the case where all values of s are distinct.

The equality DISPLAYFORM0 .

By Lemma 2, these values of λ also minimize the RHS of the equality.

Replacing λ by −λ and using the definition of t implies that DISPLAYFORM0 It follows that: DISPLAYFORM1 Thus, if s i = s [k] , then i = arg min(2k − 1 − n)s + A s 1.

This finishes the proof.

We prove the two properties in the statement of the theorem independently: DISPLAYFORM0 Proof.

By definition of the softmax function, the entries ef P are positive and sum to 1.

To show that P satisfies the argmax permutation property, .

Formally, for any given row i, we construct the argmax permutation vector u as: DISPLAYFORM1

Proof.

As shown in Gao & Pavel (2017) , the softmax function may be equivalently defined as soft max(z/τ ) = arg max x∈∆ n−1 x, z − τ n i=1 x i log x i .

In particular, lim τ →0 soft max(z/τ ) = arg max x. The distributional assumptions ensure that the elements of s are distinct a.s., so plugging in z = (n + 1 − 2k)s − A s 1 completes the proof.

This result follows from an earlier result by Yellott Jr (1977) .

We give the proof sketch below and refer the reader to Yellott Jr (1977) for more details.

We may prove by induction a generalization of the memoryless property: DISPLAYFORM0 If we assume as inductive hypothesis that q(X 2 ≤ · · · ≤ X n |x + t ≤ min i≥2 X i ) = q(X 2 ≤ · · · ≤ X n |t ≤ min i≥2 X i ), we complete the induction as: DISPLAYFORM1 It follows from a familiar property of argmin of exponential distributions that: DISPLAYFORM2 and by another induction, we have q( DISPLAYFORM3 Finally, following the argument of BID2 , we apply the strictly decreasing function g(x) = −β log x to this identity, which from the definition of the Gumbel distribution implies: DISPLAYFORM4

While applying the arg max operator to a vector with duplicate entries attaining the max value, we need to define the operator semantics for arg max to handle ties in the context of the proposed relaxation.

Definition 6.

For any vector with ties, let arg max set denote the operator that returns the set of all indices containing the max element.

We define the arg max of the i-th in a matrix M recursively:1.

If there exists an index j ∈ {1, 2, . . .

, n} that is a member of arg max set(M [i, :]) and has not been assigned as an arg max of any row k < i, then the arg max is the smallest such index.

This function is efficiently computable with additional bookkeeping.

Lemma 7.

For an input vector s with the sort permutation matrix given as P sort(s) , we have s j1 = s j2 if and only if there exists a row i such that P [i, DISPLAYFORM0 Proof.

From Eq. 5, we have the i-th row of P [i, :] given as: DISPLAYFORM1 .

Therefore, we have the equations: Proof.

Assume without loss of generality that | arg max set( P [i 1 , :])| > 1 for some i. Let j 1 , j 2 be two members of | arg max set( P [i 1 , :])|.

By Lemma 7, s j1 = s j2 , and therefore DISPLAYFORM2 DISPLAYFORM3 Hence if j 1 ∈ arg max set( P [i 2 , :]), then j 2 is also an element.

A symmetric argument implies that if j 2 ∈ arg max set( P [i 2 , :]), then j 1 is also an element for arbitrary j 1 , j 2 ∈ | arg max set( P [i 1 , :])|.

This completes the proof.

T ∈ R n , the vector z defined by z i = arg max j P sort(s) [i, j] is such that z ∈ Z n .

Proof.

From Corollary 3, we know that the row P sort(s) [i, :] attains its maximum (perhaps nonuniquely) at some P sort(s) [i, j] where s j = s [i] .

Note that s [i] is well-defined even in the case of ties.

Consider an arbitrary row P sort(s) [i, :] and let arg max set ( P sort(s) [i, :] Because P [i, :] is one of these rows, there can be up to m − 1 such rows above it.

Because each row above only has one arg max assigned via the tie-breaking protocol, it is only possible for up to m − 1 elements of arg max set(P [i, :] ) to have been an arg max of a previous row k < i. As | arg max set(P [i, :])| = m, there exists at least one element that has not been specified as the arg max of a previous row (pigeon-hole principle).

Thus, the arg max of each row are distinct.

Because each argmax is also an element of {1, . . .

, n}, it follows that z ∈ Z n .

We used Tensorflow BID0 and PyTorch (Paszke et al., 2017) for our experiments.

In Appendix A, we provide "plug-in" snippets for implementing our proposed relaxations in both Tensorflow and PyTorch.

The full codebase for reproducing the experiments can be found at https://github.com/ermongroup/neuralsort.For the sorting and quantile regression experiments, we used standard training/validation/test splits of 50, 000/10, 000/10, 000 images of MNIST for constructing the large-MNIST dataset.

We ensure that only digits in the standard training/validation/test sets of the MNIST dataset are composed together to generate the corresponding sets of the large-MNIST dataset.

For CIFAR-10, we used a split of 45, 000/5000/10, 000 examples for training/validation/test.

With regards to the baselines considered, we note that the REINFORCE based estimators were empirically observed to be worse than almost all baselines for all our experiments.

Architectures.

We control for the choice of computer vision models by using the same convolutional network architecture for each sorting method.

This architecture is as follows: Note that the dimension of a 5-digit large-MNIST image is 140×28.

The primary difference between our methods is how we combine the scores to output a row-stochastic prediction matrix.

For NeuralSort-based methods, we use another fully-connected layer of dimension 1 to map the image representations to n scalar scores.

In the case of Stochastic NeuralSort, we then sample from the PL distribution by perturbing the scores multiple times with Gumbel noise.

Finally, we use the NeuralSort operator to map the set of n scores (or each set of n perturbed scores) to its corresponding unimodal row-stochastic matrix.

For Sinkhorn-based methods, we use a fully-connected layer of dimension n to map each image to an n-dimensional vector.

These vectors are then stacked into an n × n matrix.

We then either map this matrix to a corresponding doubly-stochastic matrix (Sinkhorn) or sample directly from a distribution over permutation matrices via Gumbel perturbations (Gumbel-Sinkhorn).

We implemented the Sinkhorn operator based on code snippets obtained from the open source implementation of Mena et al. (2018) available at https://github.com/google/gumbel sinkhorn.

For the Vanilla RS baseline, we ran each element through a fully-connected n dimensional layer, concatenated the representations of each element and then fed the results through three fullyconnected n 2 -unit layers to output multiclass predictions for each rank.

All our methods yield row-stochastic n × n matrices as their final output.

Our loss is the row-wise cross-entropy loss between the true permutation matrix and the row-stochastic output.

Hyperparameters.

For this experiment, we used an Adam optimizer with an initial learning rate of 10 −4 and a batch size of 20.

Continuous relaxations to sorting also introduce another hyperparameter: the temperature τ for the Sinkhorn-based and NeuralSort-based approaches.

We tuned this hyperparameter on the set {1, 2, 4, 8, 16} by picking the model with the best validation accuracy on predicting entire permutations (as opposed to predicting individual maps between elements and ranks).Effect of temperature.

In FIG8 , we report the log-variance in gradient estimates as a function of the temperature τ .

Similar to the effect of temperature observed for other continuous relaxations to discrete objects such as Gumbel-softmax (Jang et al., 2017; Maddison et al., 2017) , we note that higher temperatures lead to lower variance in gradient estimates.

The element-wise mean squared Table 4 : Element-wise mean squared difference between unimodal approximations and the projected hard permutation matrices for the best temperature τ , averaged over the test set.

difference between unimodal approximations P sort(s) and the projected hard permutation matrices P sort(s) for the best τ on the test set is shown in Table 4 .

Architectures.

Due to resource constraints, we ran the quantile regression experiment on 4-digit numbers instead of 5-digit numbers.

We use the same neural network architecture as previously used in the sorting experiment.

The vanilla NN baseline for quantile regression was generated by feeding the CNN representations into a series of three fully-connected layers of ten units each, the last of which mapped to a singleunit estimate of the median.

In the other experiments, one copy of this network was used to estimate each element's rank through a method like Gumbel-Sinkhorn or NeuralSort that produces a rowstochastic matrix, while another copy was used to estimate each element's value directly.

Point predictions are obtained by multiplying the center row of the matrix with the column vector of estimated values, and we minimize the 2 loss between these point predictions and the true median, learning information about ordering and value simultaneously.

Hyperparameters.

We used the Adam optimizer with an initial learning rate of 10 −4 and a batch size of 5.

The temperature τ was tuned on the set {1, 2, 4, 8, 16} based on the validation loss.

Further Analysis.

In FIG9 , we show the scatter plots for the true vs. predicted medians on 2000 test points from the large-MNIST dataset as we vary n. For stochastic NeuralSort, we average the predictions across 5 samples.

As we increase n, the distribution of true medians concentrates, leading to an easier prediction problem (at an absolute scale) and hence, we observe lower MSE for larger n in TAB2 .

However, the relatively difficulty of the problem increases with increasing n, as the model is trying to learn a semantic sorting across a larger set of elements.

This is reflected in the R 2 values in TAB2 which show a slight dip as n increases.

Architectures.

The baseline kNN implementation for the pixel basis, PCA basis and the autoencoder basis was done using sklearn.

For the autoencoder baselines for kNN, we used the following standard architectures.

The dimension of the encoding used for distance computation in kNN is 50.

For the Fashion-MNIST and CIFAR experiments with NeuralSort, we use the ResNet18 architecture as described in https://github.com/kuangliu/pytorch-cifar.Hyperparameters.

For this experiment, we used an SGD optimizer with a momentum parameter of 0.9, with a batch size of 100 queries and 100 neighbor candidates at a time.

We chose the temperature hyperparameter from the set {1, 16, 64}, the constant learning rate from {10 −4 , 10 −5 }, and the number of nearest neighbors k from the set {1, 3, 5, 9}. The model with the best evaluation loss was evaluated on the test set.

We suspect that accuracy improvements can be made by a more expensive hyperparameter search and a more fine-grained learning rate schedule.

Accuracy for different k. In TAB8 , we show the performance of Deterministic and Stochastic NeuralSort for different choice of the hyperparameter k for the differentiable k-nearest neighbors algorithm.

For each of the experiments in this work, we assume we have access to a finite dataset D = {(x (1) , y (1) ), (x (2) , y (2) ), . . .}.

Our goal is to learn a predictor for y given x, as in a standard supervised learning (classification/regression) setting.

Below, we state and elucidate the semantics of the training objective optimized by Deterministic and Stochastic NeuralSort for the sorting and quantile regression experiments.

We are given a dataset D of sequences of large-MNIST images and the permutations that sort the sequences.

That is, every datapoint in D consists of an input x, which corresponds to a sequence containing n images, and the desired output label y, which corresponds to the permutation that sorts this sequence (as per the numerical values of the images in the input sequence).

For example, Figure 4 shows one input sequence of n = 5 images, and the permutation y = [3, 5, 1, 4, 2] that sorts this sequence.

For any datapoint x, let CE (·) denote the average multiclass cross entropy (CE) error between the rows of the true permutation matrix P y and a permutation matrix P y corresponding to a predicted permutation, say y. where each entry of s is given as s j = h φ (x j ).

E z∼q(z|s) CE (P y , P z ))

where each entry of s is given as s j = h φ (x j ).To ground this in our experimental setup, the score s j for each large-MNIST image x j in any input sequence x of n = 5 images is obtained via a CNN h φ () with parameters φ.

Note that the CNN parameters φ are shared across the different images x 1 , x 2 , . . . , x n in the sequence for efficient learning.

In contrast to the previous experiment, here we are given a dataset D of sequences of large-MNIST images and only the numerical value of the median element for each sequence.

For example, the desired label corresponds to y = 2960 (a real-valued scalar) for the input sequence of n = 5 images in Figure 4 .For any datapoint x, let MSE (·) denote the mean-squared error between the true median y and the prediction, say y. MSE (y, y) = y − y 2 2For the NeuralSort approaches, we optimize the following objective functions.

where each entry of s is given as s j = h φ (x j ).

where each entry of s is given as s j = h φ (x j ).As before, the score s j for each large-MNIST image x j in any input sequence x of n images is obtained via a CNN h φ () with parameters φ.

Once we have a predicted permutation matrix P sort(s) (or P z ) for deterministic (or stochastic) approaches, we extract the median image via P sort(s) x (or P z x).

Finally, we use a neural network g θ (·)with parameters θ to regress this image to a scalar prediction for the median.

@highlight

We provide a continuous relaxation to the sorting operator, enabling end-to-end, gradient-based stochastic optimization.

@highlight

The paper considers how to sort a number of items without explicitly necessarily learning their actual meanings or values and proposes a method to perform the optimization via a continuous relaxation.

@highlight

This work builds on a sum(top k) identity to derive a pathwise differentiable sampler of 'unimodal row stochastic' matrices.

@highlight

Introduces a continuous relaxation of the sorting operator in order to construct an end-to-end gradient-based optimization and introduces a stochastic extension of its method using Placket-Luce distributions and Monte Carlo.
Stochastic gradient descent (SGD), which dates back to the 1950s, is one of the most popular and effective approaches for performing stochastic optimization.

Research on SGD resurged recently in machine learning for optimizing convex loss functions and training nonconvex deep neural networks.

The theory assumes that one can easily compute an unbiased gradient estimator, which is usually the case due to the sample average nature of empirical risk minimization.

There exist, however, many scenarios (e.g., graphs) where an unbiased estimator may be as expensive to compute as the full gradient because training examples are interconnected.

Recently, Chen et al. (2018) proposed using a consistent gradient estimator as an economic alternative.

Encouraged by empirical success, we show, in a general setting, that consistent estimators result in the same convergence behavior as do unbiased ones.

Our analysis covers strongly convex, convex, and nonconvex objectives.

We verify the results with illustrative experiments on synthetic and real-world data.

This work opens several new research directions, including the development of more efficient SGD updates with consistent estimators and the design of efficient training algorithms for large-scale graphs.

Consider the standard setting of supervised learning.

There exists a joint probability distribution P (x, y) of data x and associated label y and the task is to train a predictive model, parameterized by w, that minimizes the expected loss between the prediction and the ground truth y.

Let us organize the random variables as ξ = (x, y) and use the notation (w; ξ) for the loss.

If ξ i = (x i , y i ), i = 1, . . .

, n, are iid training examples drawn from P , then the objective function is either one of the following well-known forms: expected risk f (w) = E[ (w; ξ)]; empirical risk f (w) = 1 n n i=1 (w; ξ i ).

Stochastic gradient descent (SGD), which dates back to the seminal work of Robbins & Monro (1951) , has become the de-facto optimization method for solving these problems in machine learning.

In SGD, the model parameter is updated until convergence with the rule

where γ k is a step size and g k is an unbiased estimator of the gradient ∇f (w k ).

Compared with the full gradient (as is used in deterministic gradient descent), an unbiased estimator involves only one or a few training examples ξ i and is usually much more efficient to compute.

This scenario, however, does not cover all learning settings.

A representative example that leads to costly computation of the unbiased gradient estimator ∇ (w, ξ i ) is graph nodes.

Informally speaking, a graph node ξ i needs to aggregate information from its neighbors.

If information is aggregated across neighborhoods, ξ i must request information from its neighbors recursively, which results in inquiring a large portion of the graph.

In this case, the sample loss for ξ i involves not only ξ i , but also all training examples within its multihop neighborhood.

The worst case scenario is that computing ∇ (w, ξ i ) costs O(n) (e.g., for a complete graph or small-world graph), as opposed to O(1) in the usual learning setting because only the single example ξ i is involved.

In a recent work, Chen et al. (2018) proposed a consistent gradient estimator as an economic alternative to an unbiased one for training graph convolutional neural networks, offering substantial evidence of empirical success.

A summary of the derivation is presented in Section 2.

The subject of this paper is to provide a thorough analysis of the convergence behavior of SGD when g k in (2) is a consistent estimator of ∇f (w k ).

We show that using this estimator results in the same convergence behavior as does using unbiased ones.

Definition 1.

An estimator g N of h, where N denotes the sample size, is consistent if g N converges to h in probability: plim N →∞ g N = h. That is, for any > 0, lim N →∞ Pr( g N − h > ) = 0.

It is important to note that unbiased and consistent estimators are not subsuming concepts (one does not imply the other), even in the limit.

This distinction renders the departure of our convergence results, in the form of probabilistic bounds on the error, from the usual SGD results that bound instead the expectation of the error.

In what follows, we present examples to illustrate the distinctions between unbiasedness and consistency.

To this end, we introduce asymptotic unbiasedness, which captures the idea that the bias of an estimator may vanish in the limit.

Definition 2.

An estimator g N of h, where N denotes the sample size, is asymptotically unbiased

An estimator can be (asymptotically) unbiased but inconsistent.

Consider estimating the mean h = µ of the normal distribution N (µ, σ 2 ) by using N independent samples X 1 , . . .

, X N .

The estimator g N = X 1 (i.e., always use X 1 regardless of the sample size N ) is clearly unbiased because E[X 1 ] = µ; but it is inconsistent because the distribution of X 1 does not concentrate around µ. Moreover, the estimator is trivially asymptotically unbiased.

An estimator can be consistent but biased.

Consider estimating the variance h = σ 2 of the normal distribution N (µ, σ 2 ) by using N independent samples X 1 , . . .

, X N .

The estimator

Hence, it is consistent owing to a straightforward invocation of the Chebyshev inequality, by noting that the mean approaches σ 2 and the variance approaches zero.

However, the estimator admits a nonzero bias σ 2 /N for any finite N .

An estimator can be consistent but biased even asymptotically.

In the preceding example, the bias σ 2 /N approaches zero and hence the estimator is asymptotically unbiased.

Other examples exist for the estimator to be biased even asymptotically.

Consider estimating the quantity h = 0 with an estimator g N that takes the value 0 with probability (N − 1)/N and the value N with probability 1/N .

Then, the probability that g N departs from zero approaches zero and hence it is consistent.

However, E[g N ] = 1 and thus the bias does not vanish as N increases.

To the best of our knowledge, this is the first work that studies the convergence behavior of SGD with consistent gradient estimators, which result from a real-world graph learning scenario that will be elaborated in the next section.

With the emergence of graph deep learning models (Bruna et al., 2014; Defferrard et al., 2016; Li et al., 2016; Kipf & Welling, 2017; Hamilton et al., 2017; Gilmer et al., 2017; Velicković et al., 2018) , the scalability bottleneck caused by the expensive computation of the sample gradient becomes a pressing challenge for training (as well as inference) with large graphs.

We believe that this work underpins the theoretical foundation of the efficient training of a series of graph neural networks.

The theory reassures practitioners of doubts on the convergence of their optimization solvers.

Encouragingly, consistent estimators result in a similar convergence behavior as do unbiased ones.

The results obtained here, including the proof strategy, offer convenience for further in-depth analysis under the same problem setting.

This work opens the opportunity of improving the analysis, in a manner similar to the proliferation of SGD work, from the angles of relaxing assumptions, refining convergence rates, and designing acceleration techniques.

We again emphasize that unbiasedness and consistency are two separate concepts; neither subsumes the other.

One may trace that we intend to write the error bounds for consistent gradient estimators in a manner similar to the expectation bounds in standard SGD results.

Such a resemblance (e.g., in convergence rates) consolidates the foundation of stochastic optimization built so far.

For a motivating application, consider the graph convolutional network model, GCN (Kipf & Welling, 2017) , that learns embedding representations of graph nodes.

The l-th layer of the network is compactly written as

where A is a normalization of the graph adjacency matrix, W (l) is a parameter matrix, and σ is a nonlinear activation function.

The matrix H (l) contains for each row the embedding of a graph node input to the l-th layer, and similarly for the output matrix H (l+1) .

With L layers, the network transforms an initial feature input matrix H (0) to the output embedding matrix H (L) .

For a node v, the embedding H (L) (v, :) may be fed into a classifier for prediction.

Clearly, in order to compute the gradient of the loss for v, one needs the corresponding row of H (L) , the rows of H (L−1) corresponding to the neighbors of v, and further recursive neighbors across each layer, all the way down to H (0) .

The computational cost of the unbiased gradient estimator is rather high.

In the worst case, all rows of H (0) are involved.

To resolve the inefficiency, Chen et al. (2018) proposed an alternative gradient estimator that is biased but consistent.

The simple and effective idea is to sample a constant number of nodes in each layer to restrict the size of the multihop neighborhood.

For notational clarity, the approach may be easier to explain for a network with a single layer; theoretical results for more layers straightforwardly follow that of Theorem 1 below, through induction.

The approach generalizes the setting from a finite graph to an infinite graph, such that the matrix expression (3) becomes an integral transform.

In particular, the input feature vector H (0) (u, :) for a node u is generalized to a feature function X(u), and the output embedding vector H

(1) (v, :) for a node v is generalized to an embedding function Z(v), where the random variables u and v in two sides of the layer reside in different probability spaces, with probability measures P (u) and P (v), respectively.

Furthermore, the matrix A is generalized into a bivariate kernel A(v, u) and the loss is written as a function of the output Z(v).

Then, (1) and (3) become

Such a functional generalization facilitates sampling on all network layers for defining a gradient estimator.

In particular, defining B(v) = A(v, u)X(u) dP (u), simple calculation reveals that the gradient with respect to the parameter matrix W is

Then, one may use t iid samples of u in the input and s iid samples of v in the output to define an estimator of G:

The gradient estimator G st so defined is consistent; see a proof in the supplementary material.

Theorem 1.

If q is continuous and f is finite, then plim s,t→∞ G st = G.

We now settle the notations for SGD.

We are interested in the (constrained) optimization problem

where the feasible region S is convex.

This setting includes the unconstrained case S = R d .

We assume that the objective function f : R d → R is subdifferentiable; and use ∂f (w) to denote the subdifferential at w. When it is necessary to refer to an element of this set, we use the notation h. If f is differentiable, then clearly, ∂f (w) = {∇f (w)}.

The standard update rule for SGD is w k+1 = Π S (w k − γ k g k ), where g k is the negative search direction at step k, γ k is the step size, and Π S is the projection onto the feasible region: Π S (w) := argmin u∈S w − u .

For unconstrained problems, the projection is clearly omitted:

Denote by w * the global minimum.

We assume that w * is an interior point of S, so that the subdifferential of f at w * contains zero.

For differentiable f , this assumption simply means that ∇f (w * ) = 0.

Typical convergence results are concerned with how fast the iterate w k approaches w * , or the function value f (w k ) approaches f (w * ).

Sometimes, the analysis is made convenient through a convexity assumption on f , such that the average of historical function values f (w i ), i = 1, . . .

, k, is lowered bounded by f (w k ), with w k being the cumulative moving average

The following definitions are frequently referenced.

Definition 3.

We say that f is l-strongly convex (with l > 0) if for all w, u ∈ R d and h u ∈ ∂f (u),

Recall that an estimator g N of h is consistent if for any > 0,

In our setting, h corresponds to an element of the subdifferential at step k; i.e., h k ∈ ∂f (w k ), g N corresponds to the negative search direction g k , and N corresponds to the sample size N k .

That g

converges to h k in probability does not imply that g N k k is unbiased.

Hence, a natural question asks what convergence guarantees exist when using g N k k as the gradient estimator.

This section answers that question.

First, note that the sample size N k is associated with not only g

We omit the superscript N k in these vectors to improve readability.

Similar to the analysis of standard SGD, which is built on the premise of the unbiasedness of g k and the boundedness of the gradient, in the following subsection we elaborate the parallel assumptions in this work.

They are stated only once and will not be repeated in the theorems that follow, to avoid verbosity.

The convergence (5) of the estimator does not characterize how fast it approaches the truth.

One common assumption is that the probability in (5) decreases exponentially with respect to the sample size.

That is, we assume that there exists a step-dependent constant C k > 0 and a nonnegative function τ (δ) on the positive axis such that

(6) for all k > 1 and δ > 0.

A similar assumption is adopted by Homem-de-Mello (2008) that studied stochastic optimization through sample average approximation.

In this case, the exponential tail occurs when the individual moment generating functions exist, a simple application of the Chernoff bound.

For the motivating application GCN, the tail is indeed exponential as evidenced by Figure 3 .

Note the conditioning on the history g 1 , . . . , g k−1 in (6).

The reason is that h k (i.e., the gradient ∇f (w k ) if f is differentiable) is by itself a random variable dependent on history.

In fact, a more rigorous notation for the history should be filtration, but we omit the introduction of unnecessary additional definitions here, as using the notion g 1 , . . .

, g k−1 is sufficiently clear.

Assumption 1.

The gradient estimator g k is consistent and obeys (6).

The use of a tail bound assumption, such as (6), is to reverse-engineer the required sample size given the desired probability that some event happens.

In this particular case, consider the setting where T SGD updates are run.

For any δ ∈ (0, 1), define the event

Given (6) and any ∈ (0, 1), one easily calculates that if the sample sizes satisfy

for all k, then,

Hence, all results in this section are established under the event E δ that occurs with probability at least 1 − , a sufficient condition of which is (7).

The sole purpose of the tail bound assumption (6) is to establish the relation between the required sample sizes (as a function of δ and ) and the event E δ , on which convergence results in this work are based.

One may replace the assumption by using other tail bounds as appropriate.

It is out of the scope of this work to quantify the rate of convergence of the gradient estimator for a particular use case.

For GCN, the exponential tail that agrees with (6) is illustrated in Section 5.4.

Additionally, parallel to the bounded-gradient condition for standard SGD analysis, we impose the following assumption.

Assumption 2.

There exists a finite G > 0 such that h ≤ G for all h ∈ ∂f (w) and w ∈ S.

Let us begin with the strongly convex case.

For standard SGD with unbiased gradient estimators, ample results exist that indicate O(1/T ) convergence 2 for the expected error, where T is the number of updates; see, e.g., (2.9)-(2.10) of Nemirovski et al. (2009) and Section 3.1 of Lacoste-Julien et al. (2012) .

We derive similar results for consistent gradient estimators, as stated in the following Theorem 2.

Different from the unbiased case, it is the error, rather than the expected error, to be bounded.

The tradeoff is the introduction of the relative gradient estimator error δ, which relates to the sample sizes as in (7) for guaranteeing satisfaction of the bound with high probability.

Theorem 2.

Let f be l-strongly convex with l ≤ G/ w 1 − w * .

Assume that T updates are run, with diminishing step size γ k = [(l − δ)k]

−1 for k = 1, 2, . . .

, T , where δ = ρ/T and ρ < l is an arbitrary constant independent of T .

Then, for any such ρ, any ∈ (0, 1), and sufficiently large sample sizes satisfying (7), with probability at least 1 − , we have

and

Note the assumption on l in Theorem 2.

This assumption is mild since if f is l-strongly convex, it is also l -strongly convex for all l <

l. The assumption is needed in the induction proof of (8) when establishing the base case w 1 − w * .

One may remove this assumption at the cost of a cumbersome right-hand side of (8), over which we favor a neater expression in the current form.

With an additional smoothness assumption, we may eliminate the logarithmic factor in (9) and obtain a result for the iterate w T rather than the running average w T .

The result is a straightforward consequence of (8).

Theorem 3.

Under the conditions of Theorem 2, additionally let f be L-smooth.

Then, for any ρ satisfying the conditions, any ∈ (0, 1), and sufficiently large sample sizes satisfying (7), with probability at least 1 − , we have

In addition to O(1/T ) convergence, it is also possible to establish linear convergence (however) to a non-vanishing right-hand side, as the following result indicates.

To obtain such a result, we use a constant step size.

Bottou et al. (2016) show a similar result for the function value with an additional smoothness assumption in a different setting; we give one for the iterate error without the smoothness assumption using consistent gradients.

Theorem 4.

Under the conditions of Theorem 2, except that one sets a constant step size γ k = c with 0 < c < (2l − δ) −1 for all k, for any ρ satisfying the conditions, any ∈ (0, 1), and sufficiently large sample sizes satisfying (7), with probability at least 1 − , we have

Compare (11) with (8) in Theorem 2.

The former indicates that in the limit, the squared iterate error is upper bounded by a positive term proportional to G 2 ; the remaining part of this upper bound decreases at a linear speed.

The latter, on the other hand, indicates that the squared iterate error in fact will vanish, although it does so at a sublinear speed O(1/T ).

For convex (but not strongly convex) f , typically O(1/ √ T ) convergence is asserted for unbiased gradient estimators; see., e.g., Theorem 2 of Liu (2015) .

These results are often derived based on an additional assumption that the feasible region is compact.

Such an assumption is not restrictive, because even if the problem is unconstrained, one can always confine the search to a bounded region (e.g., an Euclidean ball).

Under this condition, we obtain a similar result for consistent gradient estimators.

Theorem 5.

Let f be convex and the feasible region S have finite diameter D > 0; that is, sup w,u∈S w − u = D. Assume that T updates are run, with diminishing step size γ k = c/ √ k for k = 1, 2, . . .

, T and for some c > 0.

Let δ = ρ/ √ T where ρ > 0 is an arbitrary constant independent of T .

Then, for any such ρ, any ∈ (0, 1), and sufficiently large sample sizes satisfying (7), with probability at least 1 − , we have

One may obtain a result of the same convergence rate by using a constant step size.

In the case of unbiased gradient estimators, see Theorem 14.8 of Shalev-Shwartz & Ben-David (2014).

For such a result, one assumes that the step size is inversely proportional to √ T .

Such choice of the step size is common and is also used in the next setting.

For the general (nonconvex) case, convergence is typically gauged with the gradient norm.

One again obtains O(1/ √ T ) convergence results for unbiased gradient estimators; see, e.g., Theorem 1 of Reddi et al. (2016) (which is a simplified consequence of the theory presented in Ghadimi & Lan (2013) ).

We derive a similar result for consistent gradient estimators.

Theorem 6.

Let f be L-smooth and S = R d .

Assume that T updates are run, with constant step size is an arbitrary constant.

Then, for any such δ, any ∈ (0, 1), and sufficiently large sample sizes satisfying (7), with probability at least 1 − , we have

All the results in the preceding subsection assert convergence for SGD with the use of a consistent gradient estimator.

As with the use of an unbiased one, the convergence for the strongly convex case is O(1/T ), or linear if one tolerates a non-vanishing upper bound, and the convex and nonconvex cases O(1/ √ T ).

These theoretical results, however, are based on assumptions of the sample size N k and the step size γ k that are practically challenging to verify.

Hence, in a real-life machine learning setting, the sample size and the learning rate (the initial step size) are treated as hyperparameters to be tuned against a validation set.

Nevertheless, these results establish a qualitative relationship between the sample size and the optimization error.

Naturally, to maintain the same failure probability , the relative gradient estimator error δ decreases inversely with the sample size N k .

This intuition holds true in the tail bound condition (6) with (7), when τ (δ) is a monomial or a positive combination of monomials with different degrees.

With this assumption, the larger is N k , the smaller is δ (and also ρ, the auxiliary quantity defined in the theorems); hence, the smaller are the error bounds (8)-(13).

Theorem 4 presents a linear convergence result for the strongly convex case, with a non-vanishing right-hand side.

In fact, it is possible to obtain a result with the same convergence rate but a vanishing right-hand side, if one is willing to additionally assume L-smoothness.

The following theorem departs from the set of theorems in Section 4.2 on the assumption of the sufficient sample size N k and the gradient error δ.

Theorem 7.

Let f be l-strongly convex and L-smooth with l < L. Assume that T updates are run with constant step size γ k = 1/L for k = 1, 2, . . .

, T .

Let δ k , k ≥ 1 be a sequence where lim k→∞ δ k+1 /δ k ≤ 1.

Then, for any positive η < l/L, ∈ (0, 1), and sample sizes

with probability at least 1 − , we have

where

Here, δ k is the step-dependent gradient error.

If it decreases to zero, then so does E T .

Theorem 7 is adapted from Friedlander & Schmidt (2012), who studied unbiased gradients as well as noisy gradients.

We separate Theorem 7 from those in Section 4.2 only for the sake of presentation clarity.

The spirit, however, remains the same.

Namely, consistent estimators result in the same convergence behavior (i.e., rate) as do unbiased ones.

All results require an assumption on sufficient sample size owing to the probabilistic convergence of the gradient estimator.

In this section, we report several experiments to illustrate the convergence behavior of SGD by using consistent gradient estimators.

We base the experiments on the training of the GCN model (Kipf & Welling, 2017) motivated earlier (cf.

Section 2).

The code repository will be revealed upon paper acceptance.

We use three data sets for illustration, one synthetic and two real-world benchmarks.

The purpose of a synthetic data set is to avoid the regularity in the sampling of training/validation/test examples.

The data set, called "Mixture," is a mixture of three overlapping Gaussians.

The points are randomly connected, with a higher probability for those within the same component than the ones straddling across components.

See the supplementary material for details of the construction.

Because of the significant overlap, a classifier trained with independent data points unlikely predicts well the component label, but a graph-based method is more likely to be successful.

Additionally, we use two benchmark data sets, Cora and Pubmed, often seen in the literature.

These graphs are citation networks and the task is to predict the topics of the publications.

We follow the split used in Chen et al. (2018) .

See the supplementary material for a summary of all data sets.

The GCN model is hyperparameterized by the number of layers.

Without any intermediate layer, the model can be considered a generalized linear model and thus the cross-entropy loss function is convex.

Moreover, with the use of an L 2 regularization, the loss becomes strongly convex.

The predictive model reads P = softmax( AXW (0) ), where X is the input feature matrix and P is the output probability matrix, both row-wise.

One easily sees that the only difference between this model and logistic regression P = softmax(XW (0) ) is the neighborhood aggregation AX.

Standard batched training in SGD samples a batch (denoted by the index set I 1 ) from the training set and evaluates the gradient of the loss of softmax( A(I 1 , :)XW (0) ).

In the analyzed consistentgradient training, we additionally uniformly sample the input layer with another index set I 0 and evaluate instead the gradient of the loss of softmax( Figure 1 shows the convergence curves as the iteration progresses.

The plotted quantity is the overall loss on all training examples, rather than the batch loss for only the current batch.

Hence, not surprisingly the curves are generally quite smooth.

We compare standard SGD with the use of consistent gradient estimators, with varying sample size |I 0 |.

Additionally, we compare with the Adam training algorithm (Kingma & Ba, 2015) , which is a stochastic optimization approach predominantly used in practice for training deep neural networks.

One sees that for all data sets, Adam converges faster than does standard SGD.

Moreover, as the sample size increases, the loss curve with consistent gradients approaches that with an unbiased one (i.e., standard SGD).

This phenomenon qualitatively agrees with the theoretical results; namely, larger sample size improves the error bound.

Note that all curves in the same plot result from the same parameter initialization; and all SGD variants apply the same learning rate.

It is important to note that the training loss is only a surrogate measure of the model performance; and often early termination of the optimization acts as a healthy regularization against over-fitting.

In our setting, a small sample size may not satisfy the assumptions of the theoretical results, but it proves to be practically useful.

In Table 1 (left), we report the test accuracy attained by different training algorithms at the epoch where validation accuracy peaks.

One sees that Adam and standard SGD achieves similar accuracies, and that SGD with consistent gradient sometimes surpasses these accuracies.

For Cora, a sample size 400 already yields an accuracy noticeably higher than do Adam and standard SGD.

, and a GCN with more layers is analogous.

We repeat the experiments in the preceding subsection.

The results are reported in Figure 2 and Table 1 (right).

The observation of the loss curve follows the same as that in the convex case.

Namely, Adam converges faster than does unbiased SGD; and the convergence curve with a consistent gradient approaches that with an unbiased one.

On the other hand, compared with 1-layer GCN, 2-layer GCN yields substantially higher test accuracy for the data set Mixture, better accuracy for Cora, and very similar accuracy for Pubmed.

Within each data set, the performances of different training algorithms are on par.

In particular, a small sample size (e.g., 400) suffices for achieving results comparable to the state of the art (cf.

Chen et al. (2018) ).

The nature of a consistent estimator necessitates a characterization of the speed of probability convergence for building further results, such as the ones in this paper.

The speed, however, depends on the neural network architecture and it is out of the scope of this work to quantify it for a particular use case.

Nevertheless, for GCN we demonstrate empirical findings that agree with the exponential tail assumption (6).

In Figure 3 (solid curves), we plot the tail probability as a function of the sample size N at different levels of estimator error δ, for the initial gradient step in 1-layer GCN.

For each N , 10,000 random gradient estimates were simulated for estimating the probability.

Because the probability is plotted in the logarithmic scale, the fact that the curves bend down indicates that the convergence may be faster than exponential.

Additionally, the case of 2-layer GCN is demonstrated by the dashed curves in Figure 3 .

The curves tend to be straight lines in the limit, which indicates an exponential convergence.

To the best of our knowledge, this is the first work that studies the convergence behavior of SGD with consistent gradient estimators, and one among few studies of first-order methods that employ biased (d'Aspremont, 2008; Schmidt et al., 2011) or noisy (Friedlander & Schmidt, 2012; Devolder et al., 2014; Ge et al., 2015) estimators.

The motivation originates from learning with large graphs and the main message is that the convergence behavior is well-maintained with respect to the unbiased case.

While we analyze the classic SGD update formula, this work points to several immediate extensions.

One direction is the design of more efficient update formulas resembling the variance reduction technique for unbiased estimators (Johnson & Zhang, 2013; Defazio et al., 2014; Bottou et al., 2016) .

Another direction is the development of more computation-and memory-efficient training algorithms for neural networks for large graphs.

GCN is only one member of a broad family of message passing neural networks (Gilmer et al., 2017 ) that suffer from the same limitation of neighborhood aggregation.

Learning in these cases inevitably faces the costly computation of the sample gradient.

Hence, a consistent estimator appears to be a promising alternative, whose construction is awaiting more innovative proposals.

We are grateful to an anonymous reviewer who inspired us of an interesting use case (other than GCN).

Learning to rank is a machine learning application that constructs ranking models for information retrieval systems.

In representative methods such as RankNet (Burges et al., 2005) and subsequent improvements (Burges et al., 2007; Burges, 2010) , s i is the ranking function for document i and the learning amounts to minimizing the loss

where the summation ranges over all pairs of documents such that i is ranked higher than j. The pairwise information may be organized as a graph and the loss function may be similarly generalized as a double integral analogous to (4).

Because of nonlinearity, Monte Carlo sampling of each integral will result in a biased but consistent estimator.

Therefore, a new training algorithm is to sample i and j separately (forming a consistent gradient) and apply SGD.

The theory developed in this work offers guarantees of training convergence.

A.9 PROOF OF THEOREM 7

Theorem 2.2 of Friedlander & Schmidt (2012) states that when the gradient error

inequality (14) holds.

It remains to show that the probability that (16) happens is at least 1 − .

The assumption on the sample size N k means that

Then, substituting δ k = δ h k into assumption (6) yields

Hence, the probability that (16) happens is

(1 − /T ) ≥ 1 − , which concludes the proof.

B EXPERIMENT DETAILS , and σ 3 = 0.25 are equally weighted but significantly overlap with each other.

Random connections are made between every pair of points.

For points in the same component, the probability that they are connected is p intra = 1e-3; for points straddle across components, the probability is p inter = 2e-4.

See Figure 4 (a) for an illustration of the Gaussian mixture and Figure 4 (b) for the graph adjacency matrix.

Table 2 for a summary of the data sets used in this work.

Table 3 for the hyperparameters used in the experiments.

For parameter initialization, we use the Glorot uniform initializer (Glorot & Bengio, 2010) .

B.4 RUN TIME See Table 4 for the run time (per epoch).

As expected, a smaller sample size is more computationally efficient.

SGD with consistent gradients runs faster than the standard SGD and Adam, both of which admit approximately the same computational cost.

@highlight

Convergence theory for biased (but consistent) gradient estimators in stochastic optimization and application to graph convolutional networks
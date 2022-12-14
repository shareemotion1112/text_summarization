Permutations and matchings are core building blocks in a variety of latent variable models, as they allow us to align, canonicalize, and sort data.

Learning in such models is difficult, however, because exact marginalization over these combinatorial objects is intractable.

In response, this paper introduces a collection of new methods for end-to-end learning in such models that approximate discrete maximum-weight matching using the continuous Sinkhorn operator.

Sinkhorn iteration is attractive because it functions as a simple, easy-to-implement analog of the softmax operator.

With this, we can define the Gumbel-Sinkhorn method, an extension of the Gumbel-Softmax method (Jang et al. 2016, Maddison2016 et al. 2016) to distributions over latent matchings.

We demonstrate the effectiveness of our method by outperforming competitive baselines on a range of qualitatively different tasks: sorting numbers, solving jigsaw puzzles, and identifying neural signals in worms.

In principle, deep networks can learn arbitrarily sophisticated mappings from inputs to outputs.

However, in practice we must encode specific inductive biases in order to learn accurate models from limit data.

In a variety of recent research efforts, practitioners have provided models with the ability to explicitly manipulate latent combinatorial objects such as stacks BID12 BID23 , memory slots BID16 BID45 , mathematical expressions BID33 , program traces BID13 BID6 , and first order logic .

Operations on these discrete objects can be approximated using differentiable operations on continuous relaxations of the objects.

As such, these operations can be included as modules in neural network models that can be trained end-toend by gradient descent.

Matchings and permutations are a fundamental building block in a variety of applications, as they can be used to align, canonicalize, and sort data.

Prior work has developed learning algorithms for supervised learning where the training data includes annotated matchings BID36 BID46 .

However, we would like to learn models with latent matchings, where the matching is not provided to us as supervision.

This is a common and relevant setting.

For example, BID30 showed a problem from neuroscience involving the identification of neurons from the worm C. elegans can be cast as the inference of latent permutation on a larger hierarchical structure.

Unfortunately, maximizing the marginal likelihood for problems with latent matchings is very challenging.

Unlike for problems with categorical latent variables, we cannot obtain unbiased stochastic gradients of the marginal likelihood using the score function estimator BID54 , as computing the probability of a given matching requires computing an intractable partition function for a structured distribution.

Instead, we draw on recent work that obtains biased stochastic gradients by relaxing the discrete latent variables into continuous random variables that support the reparametrization trick BID22 BID31 .Our contributions are the following: first, in Section 2 we present a theoretical result showing that the non-differentiable parameterization of a permutation can be approximated in terms of a differentiable relaxation, the so-called Sinkhorn operator.

Based on this result, in Section 3 we introduce Sinkhorn networks, which generalize the work of method of BID1 for predicting rankings, and complements the concurrent work by BID9 , by focusing on more fundamental aspects.

Further, in Section 4 we introduce the Gumbel-Sinkhorn, an analog of the Gumbel Softmax distribution BID22 BID31 for permutations.

This enables optimization of the marginal likelihood by the reparametrization trick.

Finally, in Section 5 we demonstrate that our methods outperform strong neural network baselines on the tasks of sorting numbers, solving jigsaw puzzles, and identifying neural signals from C. elegans worms.

One sensible way to approximate a discrete category by continuous values is by using a temperature-dependent softmax function, component-wise defined as softmax ?? (x) i = exp(x i /?? )/ j=1 exp(x j /?? ).

For positive values of ?? , softmax ?? (x) i is a point in the probability simplex.

Also, in the limit ?? ??? 0, softmax ?? (x) i converges to a vertex of the simplex, a one-hot vector corresponding to the largest x i 1 .

This approximation is a key ingredient in the successful implementations by BID22 ; BID31 , and here we extend it to permutations.

To do so, we first state an analog of the normalization implemented by the softmax.

This is achieved through the Sinkhorn operator (or Sinkhorn normalization, or Sinkhorn balancing), which iteratively normalizes rows and columns of a matrix.

Specifically, following BID1 , we define the Sinkhorn operator S(X) over an N dimensional square matrix X as: DISPLAYFORM0 DISPLAYFORM1 where T r (X) = X (X1 N 1 N ), and T c (X) = X (1 N 1 N X) as the row and column-wise normalization operators of a matrix, with denoting the element-wise division and 1 N a column vector of ones.

BID42 proved that S(X) must belong to the Birkhoff polytope, the set of doubly stochastic matrices, that we denote B N 2 .Building on our analogy with categories, notice that choosing a category can always be cast as a maximization problem: the choice arg max i x i is the one that maximizes the function x, v (with v being a one-hot vector), i.e. the maximizing v * indexes the largest x i .

Similarly, one may parameterize the choice of a permutation P through a square matrix X, as the solution to the linear assignment problem BID28 , with P N denoting the set of permutation matrices and A, B F = trace(A B) the (Frobenius) inner product of matrices: DISPLAYFORM2 We call M (??) the matching operator, through which we parameterize the hard choice of a permutation (see FIG0 for an example).

Our theoretical contribution is to show that M (X) can be obtained as the limit of S(X/?? ), meaning that one can approximate M (X) ??? S(X/?? ) with a small ?? .

Theorem 1 summarizes our finding.

We provide a rigorous proof in appendix A; briefly, it is based on showing that S(X/?? ) solves a certain entropy-regularized problem in B n , which in the limit converges to the matching problem in equation 2.

Theorem 1.

For a doubly-stochastic matrix P , define its entropy as h(P ) = ??? i,j P i,j log (P i,j ).

Then, one has, S(X/?? ) = arg max DISPLAYFORM3 Now, assume also the entries of X are drawn independently from a distribution that is absolutely continuous with respect to the Lebesgue measure in R. Then, almost surely, the following convergence holds: DISPLAYFORM4 Finally, we note that Theorem 1 cannot be realized in practice, as it involves a limit on the Sinkhorn iterations l.

Instead, we'll always consider the incomplete version of the Sinkhorn operator BID1 , where we truncate l in (1) to L. FIG0 in appendix A.3 illustrates the dependence of the approximation in ?? and L.

Now we show how to apply the approximation in Theorem 1 in the context of artificial neural networks.

We construct a layer that encodes the representation of a permutation, and show how to train networks containing such layers as intermediate representations.

We define the components of this network through a minimal example: consider the supervised task of learning a mapping from scrambled objectsX to actual, non-scrambled X. Data, then, are M pairs (X i ,X i ) whereX i can be constructed by randomly permuting pieces of X i .

We state this problem as a permutation-valued regression X i = P ???1 ??,XiX i + ?? i , where ?? i is a noise term, and P ??,Xi is the permutation matrix mapping X i toX i , which depends onX i and parameters ??.

We are concerned with minimization of the reconstruction error 3 : DISPLAYFORM0 One way to express a complex parameterization of this kind is through a neural network: this network receivesX i as input, which is then passed through some intermediate, feed-forward computations of the type g h (W h x h + b h ), where g h are nonlinear activation functions, x h is the output of a previous layer, and ?? = {(W h , b h )} h are the network parameters.

To make the final network output be a permutation, we appeal to constructions developed in Section 2: by assuming that the final network output P ??,X can be parameterized as the solution of the assignments problem; i.e., P ??,X = M (g(X, ??)), where g(??, ??) represents the outcome of all operations involving g h .

Unfortunately, the above construction involves a non-differentiable f (in ??).

We use Theorem 1 as a justification for replacing M (g(X, ??)) by the differentiable S(g(X, ??)/?? ) in the computational graph.

The value of ?? must be chosen with caution: if ?? is too small, gradients vanishes almost everywhere, as S(g(X, ??)/?? ) approaches the non-differentiable M (g(X, ??)).

Conversely, if ?? is too large, S(X/?? ) may be far from the vertices of the Birkhoff polytope, and reconstructions P ???1 ??,XX may be nonsensical (see Figure 2a ).

Importantly, we will always add noise to the output layer g(X, ??) as a regularization device: by doing so we ensure uniqueness of M (g(X, ??)), which is required for convergence in Theorem 1.

Among all possible architectures that respect the aforementioned parameterization, we will only consider networks that are permutation equivariant, the natural kind of symmetry arising in this context.

Specifically, we require networks to satisfy: DISPLAYFORM0 where P is an arbitrary permutation.

The underlying intuition is simple: reconstructions of objects should not depend on how pieces were scrambled, but only on the pieces themselves.

We achieve permutation equivariance by using the same network to process each piece ofX, throwing an N dimensional output.

Then, these N outputs (each with N components) are used to create the rows of the matrix g(X, ??), to which we finally apply the (differentiable) Sinkhorn operator (i.e. g stacks the composition of the g h acting locally on each piece).

One can interpret each row as representing a vector of local likelihoods of assignment, but they might be inconsistent.

The Sinkhorn operator, then, mixes those separate representations, and ensures that consistent (approximate) assignment are produced.

With permutation equivariance, the only consideration left to the practitioner is the choice of the particular architecture, which will depend on the particular kind of data.

In Section 5 we illustrate the uses of Sinkhorn networks with three examples, each of them using a different architecture.

Also, in figure 1 we illustrate a network architecture used in one of our examples.

Sinkhorn network is a supervised method for learning to reconstruct a scrambled objectX (input) given several training examples (X i ,X i ).

By applying some non-linear transformations, a Sinkhorn network richly parameterizes the mapping betweenX and the permutation P that once applied t?? X, will allow to reconstruct the original object as X rec = P X (the output).

We note that Sinkhorn networks may be similarly used not only to learn permutations, but also to learn matchings between objects of two sets of the same size.

Figure 1: Schematic of Sinkhorn Network for Jigsaw puzzles.

Each piece of the scrambled digitX is processed with the same (convolutional) network g 1 (arrows with solid circles).

The outputs lying on a latent space (rectangles surroundingX) are then connected through g 2 (arrows with empty circles) to conform the rows of the matrix g(X, ??); g(X, ??) i = g 1 ??? g 2 (X i ).

Rows may be interpreted as unnormalized assignment probabilities, indicating individual unnormalized likelihoods of pieces of X to be at every position in the actual image.

Applying S(??) leads to a 'soft-permutation' P ??,X that resolves inconsistencies in g(X, ??).

P ??,X is then used to recover the actual X at training, although at test time one may use the actual M (g(X, ??)).

Recently, in BID22 and BID31 , the Gumbel-Softmax or Concrete distributions were defined for computational graphs with stochastic nodes; i.e, latent probabilistic representations.

Their choice is guided by the following i) they seek re-parameterizable distributions to enable the re-parameterization trick BID24 , and note that via the Gumbel trick (see below) any categorical distribution is re-parameterizable, ii) since the re-parameterization in i)is not differentiable, they consider instead sampling under the softmax approximation.

This gives rise to the Gumbel-Softmax distribution.

Here we parallel these choices to enable learning of a probabilistic latent representation of permutations.

To this aim, we start by considering a generic distribution on the discrete set Y, with potential function X : Y ??? R: DISPLAYFORM0 Regarding i), the Gumbel trick arises in the context of Perturb and MAP methods BID35 for sampling in discrete graphical models.

This has recently received renewed interest BID2 , as it recasts the a difficult sampling problem as an easier optimization problem.

In detail, sampling from (6), can be achieved by the maximization of random perturbations of each potential X(y), with Gumbel i.i.d.

noise ??(y); i.e., arg max y???Y {X(y) + ??(y)} ??? p(??|X).

Therefore, one can re-parameterize any categorical distribution (corresponding to (6) with X(y) = X, y ) by the choice of a category, after injecting noise.

However, the above scheme is unfeasible in our context, as |Y| = N !.

Nonetheless, we appeal to an interesting result: in cases where DISPLAYFORM1 , the use of rank-one perturbations DISPLAYFORM2 is proposed as a more tractable alternative.

Although ultimately heuristic, they lead to bounds in the partition function BID17 BID2 , and can also be understood as providing approximate or unbiased samples from the true density BID18 BID47 .Guided by this, we say the random permutation P follows the Gumbel-Matching distribution with parameter X, denoted P ??? G.M.(X), if it has the distribution arising by the rank-one perturbation of (6) on permutations, with the linear potential X(P ) = X, P F (replacing y with P ).

One can verify, in a similar line as in BID29 DISPLAYFORM3 Unfortunately, as ii) with the categorical case, Gumbel-Matching distribution samples are not differentiable in X, but by appealing to Theorem 1, we define its relaxation for doubly stochastic matrices as follows: we say P follows the Gumbel-Sinkhorn distribution with parameter X and temperature ?? , denoted P ??? G.S.(X, ?? ), if it has the distribution of S((X + ??)/?? ).

Samples of G.S.(X, ?? ) converge almost surely to samples of the Gumbel-Matching distribution (see FIG0 DISPLAYFORM4 Unlike for the categorical case, neither the Gumbel-Matching nor Gumbel-Sinkhorn distributions have tractable densities.

However, this does not preclude inference: likelihood-free methods have recently been developed to enable learning in such implicitly defined distributions BID37 BID48 .

These methods avoid evaluating the likelihood based on the observation that in many cases inference can be cast as the estimation of a likelihood ratio, which can be obtained from samples BID21 .

Regardless of these useful advances, in the following we develop a solution based on using the likelihoods of random variables whose densities are available.

Consider a latent variable model probabilistic model with observed data Y , and latent Z = {P, W } where P is a permutation and W are other variables.

Here we illustrate how to approximate the posterior probability p({P, W }|Y ) using variational inference .

Specifically, we aim to maximize the ELBO, the r.h.s.

of FORMULA12 : DISPLAYFORM0 We assume that both the prior and variational posteriors decompose as products (mean-field).

That is, q({P, W }|Y ) = q(P )q(W ), p(P, W ) = p(P )

p(W ).

With this assumption, we may focus only on the discrete part of the problem, i.e. without loss of generality we can assume Z = P .We parameterize our variational prior and posteriors on P using the Gumbel-Matching distributions with some parameter X; G.M.(X).

To enable differentiability, we replace them by G.S.(X, ?? ) distributions, leading to a surrogate ELBO that uses relaxed (continuous) variables.

In more detail, Published as a conference paper at ICLR 2018 BID52 .0 .0 .07 1.

1. 1.

Table 1 : Results on the number sorting task measured using Prop.

any wrong.

In the top two rows we compare to BID52 , showing that our approach can sort far more inputs at significantly higher accuracy.

In the bottom rows we evaluate generalization to different intervals on the real line.

DISPLAYFORM1 for our uniform prior over permutations we use the isotropic G.S.(X = 0, ?? prior ) distribution, while for the variational posterior we consider the more generic G.S.(X, ?? ).

FORMULA12 is intractable as there is not closed form expression for the density of G.S. random variables.

As a solution, we use that our prior and posterior are re-parameterizable in terms of matrices ?? of Gumbel i.i.d variables: we have DISPLAYFORM2 DISPLAYFORM3 for the posterior and prior, respectively.

To obtain a tractable expression, we propose to use as 'code' or stochastic node Z, the variable (X + ??)/?? instead.

Then, the KL term substantially simplifies to KL((X + ??)/?? ??/?? prior ).

This term can be computed explicitly, as shown in appendix B.3.This 'trick', however, comes at a cost: the divergence KL(Z 1 Z 2 ) would certainly remain unchanged by applying the same invertible transformation g to both variables Z 1 and Z 2 , but in the general case, for non-invertible transformations, such as S(??), one has KL(Z 1 Z 2 ) ??? KL(g(Z 1 ) g(Z 2 )).

This implies that working in the 'Gumbel space' might entail the optimization of a less tight lower bound.

Nonetheless, through categorical experiments on MNIST (see appendix C.3) we observe this loss of tightness is minimal, suggesting the suitability of our approach on permutations.

Finally, we note that key to to our treatment of the problem is the fact that both the prior and posterior were the same function (S(??)) of a simpler distribution.

This may not be the case in more general models.

To conclude this section, we refer the reader to table 8 in appendix D.2 for a summary of all the constructions on permutations developed in this work.

In this section we perform several experiments comparing to existing methods.

In the first three experiments we explore different Sinkhorn network architectures of increasing complexity, and therefore, they mostly implements section 3.

The fourth experiment relates to the probabilistic constructions described in section 4, and addresses a problem involving marginal inferences over a latent, unobserved permutation.

All experimental details not stated here are in appendix B.

To illustrate the capabilities of Sinkhorn Networks in a simple scenario, we consider the task of sorting numbers using artificial neural networks as in BID52 .

Specifically, we sample uniform random numbersX in the [0, 1] interval and we train our network with pairs (X, X) where X are the sameX but in sorted order.

The network has a first fully connected layer that links a number with an intermediate representation (with 32 units), and a second (also fully connected) layer that turns that representation into a row of the matrix g(X, ??).

Table 1 shows our network learns to sort up to N = 120 numbers.

As an evaluation measure, we report the proportion of sequences where there was at least one error (Prop.

any wrong).

Surprisingly,

Celeba Imagenet 2x2 3x3 4x4 5x5 6x6 2x2 3x3 4x4 5x5 2x2 3x3 BID9 and provide additional results from our experiments.

Randomly guessed permutations of n items have an expected proportion of errors of (n ??? 1)/n.

Note that our model has at least 20x fewer parameters.. the network learns to sort numbers even when test examples are not sampled from U (0, 1), but on a considerably different interval.

This indicates the network is not overfitting.

These results can be compared with those from BID52 , where a much more complex (recurrent) network was used, but performance guarantees were obtained only with at most N = 15 numbers.

In that case, the reported error rate is 0.9, whereas ours starts to degrade only after N ??? 100 for most test intervals.

A more complex scenario for learning permutations arises in the reconstruction of an image X from a collection of scrambled "jigsaw" piecesX BID34 BID9 .

In this example, our network differs from the one in 5.1 in the first layer is a simple CNN (convolution + max pooling), which maps the puzzle pieces to an intermediate representation (see figure 1 for details).For evaluation on test data, we report several measures: first, in addition to Prop.

any wrong we also consider Prop.

wrong, the overall proportion of scrambled pieces that were wrongly assigned to their actual position.

Also, we use l1 and l2 (train) losses and the Kendall tau, a "correlation coefficient" for ranked data.

In Table 2 , we benchmark results for the MNIST, Celeba and Imagenet datasets, with puzzles between 2x2 and 6x6 pieces.

In MNIST we achieve very low l1 and l2 on up to 6x6 puzzles but a high proportion of errors.

This is a consequence of our loss being agnostic to particular permutations, but only caring about reconstruction errors: as the number of black pieces increases with the number of puzzle pieces, many become unidentifiable under this loss.

In Celeba, we are able to solve puzzles of up to 5x5 pieces with only 21% of pieces of faces being incorrectly ordered (see Figure 2a for examples of reconstructions).

For this dataset, we provide additional baselines in TAB3 of appendix C.1: there, we show that performance substantially decreases if the temperature is too small or large, but only slightly decreases if only one Sinkhorn iterations is made.

We observe that temperature does play a relevant role, consistent with the findings of BID31 BID22 .

This might not be obvious a-priori, as one could reason that temperature over-parameterizes the network.

However, results confirm this is not the case.

We hypothesize that different temperatures result in parameter convergence in different phases or regions.

Also, the minor difference for a single iteration suggest that only a few might be necessary, implying potential savings in the memory needed to unroll computations in the graph, during training.

Learning in the Imagenet dataset is much more challenging, as there isn't a sequential structure that generalizes among images, unlike Celeba and MNIST.

In this dataset, our network ties with the .72 Kendall tau score reported in BID9 .

Their network, named DeepPermNet, is based on the stacking of up to the sixth fully connected layer fc6 of AlexNet BID27 , which finally (fully) connects to a Sinkhorn layer through intermediate fc7 and fc8.

We note, however, our network is much simpler, with only two layers and far fewer parameters.

Specifically, the network Figure 2 : (a) Sinkhorn networks can be trained to solve Jigsaw Puzzles.

Given a trained model, 'soft' reconstructions are shown at different ?? using S(X/?? ).

We also show hard reconstructions, made by computing M (X) with the Hungarian algorithm BID32 .

(b) Sinkhorn networks can also be used to learn to transform any MNIST digit into another.

We show hard and soft reconstructions, with ?? = 1.that produced our best results had around 1,050,000 parameters (see appendix B for a derivation), while in DeepPermNet, the layer connecting fc6 with fc7 has 512 ?? 4096 ?? 9 ??? 19, 000, 000 parameters, let alone the AlexNet parameters (also to be learned).

Indeed, we believe there is no reason to consider a complex stacking of convolutions: as the number of pieces increases, each piece is smaller and the convolutional layer eventually becomes fully connected.

In the following experiment we explore this phenomenon in more detail.

We also consider an original application, motivated by the observation that the Jigsaw Puzzle task becomes ill-posed if a puzzle contains too many pieces.

Indeed, consider the binarized MNIST dataset: there, reconstructions are not unique if pieces are sufficiently atomic, and in the limit case of pieces of size 1x1 squared pixels, for a given scrambled MNIST digit there are as many valid reconstructions as there are MNIST digits with the same number of white pixels.

In other words, reconstructions stop being probabilistic and become a multimodal distribution over permutations.

We exploit this intuition to ask whether a neural network can be trained to achieve arbitrary digit reconstructions, given their loose atomic pieces.

To address this question, we slightly changed the network in 5.2, this time stacking several second layers linking an intermediate representation to the output.

We trained the network to reconstruct a particular digit with each layer, by using digit identity to indicate which layer should activate with a particular training example.

Our results demonstrate a positive answer: Figure 2b shows reconstructions of arbitrary digits given 10x10 scrambled pieces.

In general, they can be unambiguously identified by the naked eye.

Moreover, this judgement is supported by the assessment of a neural network.

Specifically, we trained a two-layer CNN 5 on MNIST (achieving a 99.2% accuracy on test set) and evaluated its performance on the test set generated by arbitrary transformations of each digit of the original test set into any other digit.

We found the CNN made an appropriate judgement in 85.1% of the time.

More specific results, regarding specific transformations are presented in Table 5 of appendix C.2.Finally, we note that meaningful assemblies are possible regardless of the original digit: in Figure 4 of appendix C.2 we show arbitrary reconstructions, by this same network, of "digits" from a 'strongly mixed' MNIST dataset.

In detail, these "digits" were crafted by sampling, without replacement, from a bag containing all the small pieces from all original digits.

These reconstructions suggest the possibility of an alternative to generative modeling, based on the (random) assembly of small pieces of noise, instead of the processing of noise through a neural network.

However, this would require training the network without supervision, which is beyond the scope of this work.

Table 3 : Results for the C. elegans neural inference problem.

We illustrate how the G.S. distribution can be used as a continuous relaxation for stochastic nodes in a computational graph.

To this end, we revisit the "C. elegans neural identification problem", originally introduced in Linderman et al. FORMULA0 .

We refer the reader to BID30 for an in-depth introduction, but briefly, C. elegans is a nematode (worm) whose biological neural configuration -the connectome -is stereotypical; i.e. specimens always posses the same number of somatic neurons (282) BID49 , and the ways those neurons connect and interact changes little from worm to worm.

Therefore, its brain can be thought of as a canonical object, and its neurons can unequivocally be identified with names.

The task, then, consists of matching traces from the observed neural dynamics Y to identities (neuron names) in the canonical brain.

This problem is stated in terms of a Bayesian hierarchical model, in order to profit from prior information that may constrain the possibilities.

Specifically, one states a linear dynamical system Y t = P W P T Y t???1 + ?? t , where ?? t is a noise term and W and P are latent variables with respective prior distributions.

W encodes the dynamics, with a prior p(W ) to represent the sparseness of the connectome, etc., and P is a permutation matrix representing the matching between indexes of observed neurons and their canonical counterparts, where we place a flat prior p(P ) over permutations.

Notably, within the framework it is possible to model the simultaneous problem with many worms sharing the same dynamical system, but here we avoid explicit references to individuals for notational ease.

Given this model, we seek the posterior distribution p({P, W }|Y ), a problem that we address with variational inference (Blei et al., 2017) using the constructions developed in 4.1.

In Table 3 (and also in Table 7 of appendix C.4) we show results for this task, using accuracy in matching as the performance measure.

These are broken down by relevant experimental covariates BID30 : different proportion of neurons known beforehand, and by task difficulty.

As baselines, we include i) a simple MCMC sampler that proposes local swipes on permutations ii) the rounding method presented in BID30 , iii) our method, where we also consider the absence of regularization.

Results show our method outperforms the alternatives in most cases.

MCMC fails because mixing is poor, but differences are much subtler with the other baselines.

With them, we see that clear differences with the no-regularization case confirm the stochastic nature of this problem, i.e., that it is truly necessary to represent a latent probabilistic permutation.

We believe our method outperforms the one in Linderman et al. (2017) because theirs, although it provides a explicit density, is a less tight relaxation, in the sense that points can be anywhere in the space, and not only on the Birkhoff polytope.

Therefore, their prior also needs to be defined on the entire space and may not property act as an efficient regularizer.

Learning with matchings has been extensively been studied in the machine learning community; but current applications mostly relate to structured prediction BID36 BID46 .

However, our probabilistic treatment focuses on marginal inference in a model with a latent matching.

This is a more challenging scenario, as standard learning techniques, i.e. the score function estimator or REINFORCE BID54 , are not applicable due to the partition function for non-trivial distributions over matchings.

In the case of latent categories, a recent technique that combines a relaxation and the reparameterization trick BID24 ) was proposed as a competitive alternative to RE-INFORCE for the marginal inference scenario.

Specifically, BID31 ; BID22 use the Gumbel-trick to re-parameterize a discrete density, and then replace it with a relaxed surrogate, the Gumbel Softmax distribution, to enable gradient-descent.

Our work, like the simultaneous work of BID30 , aims to extends the scope of this technique to latent permutations.

We deem our Gumbel Sinkhorn distributions as the most natural tractable extension of the Gumbel Softmax to permutations, as we clearly parallel each of the steps leading to its construction.

A parallel is also presented in Linderman et al. FORMULA0 ; and notably, unlike ours, their framework produces tractable densities.

However, it is less clear how their constructions extend each of the features of the Gumbel Softmax: for example, their rounding-based relaxation also utilizes the Sinkhorn operator, but the limit they consider does not make use of the non-trivial statement of Theorem 1, which naturally extends the categorical case (see appendix A.2 for details).

In practice, we see our results favor the Gumbel Sinkhorn distribution, since it is a tighter relaxation.

Connections between permutations and the Sinkhorn operator have been known for at least twenty years.

Indeed, the limit in Theorem 1 was first presented in BID26 , but their interpretation and motivation were more linked to statistical physics and economics.

However, our approach is different and links to recent developments in optimal transport (OT) BID50 : Theorem 1 draws on the entropy-regularization for OT technique developed inCuturi FORMULA0 , where the entropy-regularized transportation problem is referred to as a 'Sinkhorn distance'.

The extension is sensible as in the case of transportation between two discrete measures (here) the Birkhoff polytope appears naturally as the optimization set BID50 .

Entropy regularization as means to achieve a differentiable version of a loss was first proposed in BID14 in the context of generative modeling.

Although this field may appear separate, recent work BID41 makes explicit the connection to permutations: to compute a (Wasserstein) distance between a batch of dataset samples and one of generative samples of the same size, one needs to solve the matching problem so that the distance between matched samples is minimized.

Finally, we note our work shares with BID41 ; BID14 in that the OT cost function (here, the matrix X) is learned using an artificial neural network.

We understand our work as extending BID1 , which developed neural networks to learn a permutation-like structure; a ranking.

However, there, as in BID19 , the objective function was linear and the Sinkhorn operator was instead used as an approximation of a matrix of the marginals, i.e., S(P ) ??? E(P ).

In consequence, there was no need to introduce a temperature parameter and consider a limit argument, which is critical to our case.

Interestingly, equation FORMULA0 can be understood in terms of approximate marginal inference, justifying the approximation S(P ) ??? E(P ).

We comment on this in appendix D.1.

Note that Sinkhorn iteration can be interpreted as mean-field inference in an associated Gibbs distribution over matchings.

With this in mind, backpropagation through Sinkhorn is an end-to-end learning in an unrolled inference algorithm BID44 BID11 .

In future work, it may be fruitful to unroll alternative algorithms for marginal inference over matchings, such as belief propagation BID20 )

.Sinkhorn networks were also very recently introduced in BID9 , although their work substantially differs from ours.

While their interest lies in the representational aspects of CNN's, we are more concerned with the more fundamental properties.

In their work, they don't consider a temperature parameter ?? , but their network still successfully learns, as ?? = 1 happens to fall within the range of reasonable values.

On the Jigsaw puzzle task, we showed that we achieve equivalent performance with a much simpler network having several times fewer parameters and layers.

Nonetheless, we recognize the need for more complex architectures for the tasks considered in BID9 , and we hope our more general theory; particularly, Theorem 1 and the notion of equivariance, may aid further developments in that direction.

We have demonstrated Sinkhorn networks are able to learn to find the right permutation in the most elementary cases; where all training samples obey the same sequential structure; e.g., in sorted number and in pieces of faces, as we expect parts of faces occupy similar positions from sample to sample.

This is already non-trivial, as indicates one can train a neural network to solve the linear assignment problem.

However, the fact that Imagenet represented a much more challenging scenario indicates there are clear limits to our formulation.

As the most obvious extension we propose to introduce a sequential stage, in which current solutions are kept on a memory buffer, and improved.

One way to achieve this would be by exploring more complex parameterizations for permutations; i.e. replacing M (X) by a quadratic operator that may parameterize a notion of local distance between pieces.

Alternatively, one may resort to reinforcement learning techniques, as suggested in BID3 .

Either sequential improvement would help solve the "Order Matters" problem BID52 , and we deem our elementary work as a significant step in that direction.

We have made available Tensorflow code for Gumbel-Sinkhorn networks featuring an implementation of the number sorting experiment at http://github.com/google/gumbel sinkhorn .

In this section we give a rigorous proof of Theorem 1.

Also, in A.2 we briefly comment on how Theorem 1 extend a perhaps more intuitive results, in the probability simplex.

Before stating Theorem 1 we need some preliminary definitions.

We start by recalling a well-known result in matrix theory, the Sinkhorn theorem.

Theorem (Sinkhorn) .

Let A be an N dimensional square matrix with positive entries.

Then, there exists two diagonal matrices D 1 , D 2 , with positive diagonals, so that P = D 1 AD 2 is a doubly stochastic matrix.

These D 1 , D 2 are unique up to a scalar factor.

Also, P can be obtained through the iterative process of alternatively normalizing the rows and columns of A.Proof.

See BID42 ; BID43 BID25 .For our purposes, it is useful to define the Sinkhorn operator S(??) as follows: Definition 1.

Let X be an arbitrary matrix with dimension N .

Denote T r (X) = X (X1 N 1 N ), T c (X) = X (1 N 1 N X) (with representing the element-wise division and 1 n the n dimensional vector of ones) the row and column-wise normalization operators, respectively.

Then, we define the Sinkhorn operator applied to X; S(X), as follows: DISPLAYFORM0 Here, the exp(??) operator is interpreted as the component-wise exponential.

By Sinkhorn's theorem, S(X) is a doubly stochastic matrix.

Finally, we review some key properties related to the space of doubly stochastic matrices.

First, we need to define a relevant geometric object.

Definition 2.

We denote by B N the N -Birkhoff polytope, i.e., the set of doubly stochastic matrices of dimension N .

Likewise, we denote P n be the set of permutation matrices of size N .

Alternatively, DISPLAYFORM1 Theorem (Birkhoff) .

P N is the set of extremal points of B N .

In other words, the convex hull of B N equals P N .Proof.

See BID4 .

Let's now focus on the standard combinatorial assignment (or matching) problem, for an arbitrary N dimensional matrix X. We aim to maximize a linear functional (in the sense of the Frobenius norm) in the space of permutation matrices.

In this context, let's define the matching operator M (??) as the one that returns the solution of the assignment problem: DISPLAYFORM0 Likewise, we defineM (??) as a related operator, but changing the feasible space by the Birkhoff polytope:M (X) ??? arg max DISPLAYFORM1 Notice that in generalM (X), M (X) might not be unique matrices, but a face of the Birkhoff polytope, or a set of permutations, respectively (see Lemma 2 for details).

In any case, the relation M (X) ???M (X) holds by virtue of Birkhoff's theorem, and the fundamental theorem of linear programming.

Now we state the main theorem of this work:Theorem 1.

For a doubly stochastic matrix P define its entropy as h(P ) = ??? i,j P i,j log (P i,j ).

Then, one has, S(X/?? ) = arg max DISPLAYFORM2 Now, assume also the entries of X are drawn independently from a distribution that is absolutely continuous with respect to the Lebesgue measure in R. Then, almost surely the following convergence holds: DISPLAYFORM3 We divide the proof of Theorem 1 in three steps.

First, in Lemma 1 we state a relation between S(X/?? ) and the entropy regularized problem in equation FORMULA0 .

Then, in Lemma 2 we show that under our stochastic regime, uniqueness of solutions holds.

Finally, in Lemma 3 we show that in this well-behaved regime, convergence of solutions holds.

states that and Lemma 2b endows us with the tools to make a limit argument.

A.1.1 INTERMEDIATE RESULTS FOR THEOREM 1 Lemma 1.

S(X/?? ) = arg max DISPLAYFORM4 Proof.

We first notice that the solution P ?? of the above problem exists, and it is unique.

This is a simple consequence of the strict concavity of the objective (recall the entropy is strictly concave Rao (1984)).

Now, let's state the Lagrangian of this constrained problem DISPLAYFORM5 It is easy to see, by stating the equality ???L/???P = 0 that one must have for each i, j, DISPLAYFORM6 for certain diagonal matrices D 1 , D 2 , with positive diagonals.

By Sinkhorn's theorem, and our definition of the Sinkhorn operator, we must have that S(X/?? ) = P ?? .Lemma 2.

Suppose the entries of X are drawn independently from a distribution that is absolutely continuous with respect to the Lebesgue measure in R. Then, almost surely,M (X) = M (X) is a unique permutation matrix.

Proof.

This is a known result from sensibility analysis on linear programming which we prove for completeness.

Notice first that the problem in (2) is a linear program on a polytope.

As such, by the fundamental theorem of linear program, the optimal solution set must correspond to a face of the polytope.

Let F be a face of B N of dimension ??? 1, and take P 1 , P 2 ??? F, P 1 = P 2 .

If F is an optimal face for a certain X F , then X F ??? {X : P 1 , X F = P 2 , X F }.

Nonetheless, the latter set does not have full dimension, and consequently has measure zero, given our distributional assumption on X. Repeating the argument for every face of dimension ??? 1 and taking a union bound we conclude that, almost surely, the optimal solution lies on a face of dimension 0, i.e, a vertex.

From here uniqueness follows.

Lemma 3.

Call P ?? the solution to the problem in equation 10, i.e. P ?? = P ?? (X) = S(X/?? ).

Under the assumptions of Lemma 2, P ?? ??? P 0 when if ?? ??? 0 + .Proof.

Proof Notice that by Lemmas 1 and 2, P ?? is well defined and unique for each ?? ??? 0.Moreover, at ?? = 0, P 0 = M (X) is the unique solution of a linear program.

Now, let's define f ?? (??) = ??, X F + ?? h(??).

We observe that f 0 (P ?? ) ??? f 0 (P 0 ).

Indeed, one has: DISPLAYFORM7 From which convergence follows trivially.

Moreover, in this case convergence of the values implies the converge of P ?? : suppose P ?? does not converge to P 0 .

Then, there would exist a certain ?? and sequence ?? n ??? 0 such that P ??n ??? P 0 > ??.

On the other hand, since P 0 is the unique maximizer of an LP, there exists ?? > 0 such that f 0 (P 0 ) ??? f 0 (P ) > ?? whenever P ??? P 0 > ??, P ??? B N .

This contradicts the convergence of f 0 (P ??n ).

The first statement is Lemma 1.

Convergence (equation 11) is a direct consequence of Lemma 3, after noticing P ?? = S(X/?? ) and P 0 = M (X).

We note that an alternative approach for the limiting argument is presented in BID8 .

Finally, we notice that all of the above results can be understood as a generalization of the wellknown approximation result arg max i x i = lim ?? ???0 + sof tmax(x/?? ).

To see this, treat a category as a one-hot vector.

Then, one has arg max DISPLAYFORM0 where S n is the probability simplex, the convex hull of the one-hot vectors (denoted H n ).

Again, by the fundamental theorem of linear algebra, the following holds: DISPLAYFORM1 On the other hand, by a similar (but simpler) argument than of the proof of theorem 4 one can easily show that DISPLAYFORM2 where the entropy h(??) is not defined as h(e) = ??? B SUPPLEMENTAL METHODS

All experiments were run on a cluster using Tensorflow BID0 , using several GPU (Tesla K20, K40, K80 and P100) in parallel to enable an efficient exploration of the hyperparameter space: temperature, learning rate, and neural network parameters (dimensions).In all cases, we used L = 20 Sinkhorn Operator Iterations, and a 10x10 batch size: for each sample in the batch we used Gumbel perturbations to generate 10 different reconstructions.

For evaluation, we used the Hungarian Algorithm BID32 to compute M (X) required to infer the predicted matching.

Finally, experiments of section 5.4 were done consistent with model specifications stated in Linderman et al. FORMULA0 B

In the simplest network, the one that sorts number, the number of parameters is given by n u +N ??n u : Indeed, each number is connected with the hidden layer with n u (here, 32) units.

This layer connects with another layer with N units, representing a row of g(X, ??).For images, the first layer is a convolution, composed by n f convolutional filters of receptive field size K s with n c channels (one or three) followed by a ReLU + max-pooling (with stride s) operations.

Then, the number of parameters in the first layer is given by K 2 s ?? n c ?? n f + n f .

The second layers connects the output of a convolution, i.e., the stacked convolved l ?? l images by each of the filters (after max-pooling) and p 2 units, where p is the number of pieces each side was divided by.

Therefore, the number of parameters is given by l 2 /(p 2 s 2 ) ?? n f ?? p 2 = l 2 /s 2 ?? n f , up to rounding and padding subtleties.

Then, the total number of parameters is l 2 /s 2 ?? n f + K 2 s ?? n c ?? n f + n f .

For the 3x3 puzzle on Imagenet, l = 256, p = 3, n c = 3 and the optimal network was such that n f = 64, s = 2, K s = 5.

Then, it had 1,053,440 parameters.

Finally, for arbitrary assembly experiments, as one includes additional fully connected second layers, the total number of parameters is DISPLAYFORM0 where n l is the number of labels (here, n l = 10).

Here we show how to compute KL((X + ??)/?? ??/?? prior ), as defined in 4.1.

We first notice that the density of the variable h = (a + g)/b, where g has a Gumbel distribution and a, b are constants is given by: DISPLAYFORM0 Therefore, the log density ratio LR(z) between each component of h 1 = (x i,j + ?? i,j )/?? and h 2 = ?? i,j /?? prior is (suppressing indexing for simplicity) DISPLAYFORM1 We need to take expectations with respect to the distribution of h 1 .

To compute this expectation, we first express the above ratio in terms of ?? DISPLAYFORM2 Now we appeal to the law of the unconscious statistician, and take the expectation with respect to ??.

??? E(??) = ?? ??? 0.5772 (the Euler-Mascheroni constant)??? Moment generating function E(exp(t??)) = ??(1 ??? t); implying E(exp(?????)) = 1 and DISPLAYFORM0 we have: DISPLAYFORM1 From this, it easily follows (adding all the N 2 components) that DISPLAYFORM2 where S 1 = ?? prior /?? i,j x i,j and S 2 = i,j exp (???x i,j ?? prior /?? ).C SUPPLEMENTAL RESULTS

In table 4 we provide further performance measures for the Jigsaw puzzle task on Celeba, for extreme hyper-parameter values: small temperature, large temperature, and a single Sinkhorn iteration These are worse than the ones in table 2, although surprisingly, one Sinkhorn iteration already provides reasonable performance, as long temperature is chosen in an appropriate range.

In table 5 we show performance of a 2-layer CNN in detecting transformed digits as the ones they are intended to be.

From this we see the most troublesome transformation was to one, as this network most of the times categorized it as a different number.

Also, in figure 4 we show transformations, showing that to reconstruct to arbitrary digits it is not required that the original ones have an actual digit-like structure, but they can be only pieces of 'strokes' or 'dust'.

Figure 4: First column: samples from dataset created by mixing all pieces of digits, and then reassembling them into 'digits'.

Second column: random permutations of first column.

Third column: hard reconstructions using M (X).

Fourth column: soft reconstructions using S(X/?? ) and ?? = 1.

Metaphorically, one is able to reconstruct pieces out of 'dust'.

??? log p(x) Gumbel-Softmax 106.7 Concrete 111.5 Concrete (Gumbel space) 111.9 Table 7 :

Accuracy in the C.elegans neural identification problem, for varying mean number of candidate neurons (10, 30, 45, 60) and number of worms (1 and 4).

Finally, in Table 7 we show additional results for the C.elegans experiment.

The setting is the same as in Figure 4 A second connection between the distribution in (6) (and therefore, the Matching Gumbel distribution) and the Sinkhorn operator arises as a consequence of Theorem 1.

This relates to the estimation of the marginals E ?? (P i,j ), known to be a #P hard problem.

A well known result BID15 BID53 , consequence of Fenchel (conjugate) duality BID39 applied to exponential families, links this problem to optimization in the following way: lets denote by M the marginal polytope, the convex hull of the set of realizable sufficient statistics, that here coincides with B n .

Also, lets call H(??) the entropy of (6) for the parameter ??(??) such that ?? = E ??(??) (P ).

Then, E ?? (P ) = arg max ?????M ??, ?? F + H(??).Notice the only difference between the optimization problems in FORMULA0 and FORMULA0 is the entropy term, after identifying X with ??.

Therefore, one may understand the Sinkhorn operator as providing approximations for the partition function and the marginals, which will be accurate insofar as h(??) is a good approximation for H(??).

In this way, one can understand S(X) as an approximation for E ?? (P ), that may complement more classical ones, as the Bethe and Kituchani's approximations for H(??), and the corresponding approximate inference algorithms that they give rise to BID55 BID51 .

arg max x i = arg max s???S x, s M (X) = arg max P ???B P, X F Approximation arg max i x i = lim ?? ???0 + softmax(x/?? ) M (X) = lim ?? ???0 + S(X/?? ) Entropy h(s) = i ???s i log s i h(P ) = i,j ???P i,j log (P i,j )Entropy regularized linear program softmax(x/?? ) = arg max s???S x, s + ?? h(s) S(X/?? ) = arg max P ???B P, X F + ?? h(P ) Reparameterization Gumbel-max trick arg max i (x i + i )Gumbel-Matching GM (X) M (X + )

Concrete softmax((x + )/?? ) Gumbel-Sinkhorn GS(X, ?? ) S((X + )/?? )

<|TLDR|>

@highlight

A new method for gradient-descent inference of permutations, with applications to latent matching inference and supervised learning of permutations with neural networks

@highlight

The paper utilizes finite approximation of the Sinkhorn operator to describe how one can construct a neural network for learning from permutation valued training data. 

@highlight

The paper proposes a new method that approximates the discrete max-weight for learning latent permutations
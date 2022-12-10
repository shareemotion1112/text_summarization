We develop a mean field theory for batch normalization in fully-connected feedforward neural networks.

In so doing, we provide a precise characterization of signal propagation and gradient backpropagation in wide batch-normalized networks at initialization.

Our theory shows that gradient signals grow exponentially in depth and that these exploding gradients cannot be eliminated by tuning the initial weight variances or by adjusting the nonlinear activation function.

Indeed, batch normalization itself is the cause of gradient explosion.

As a result, vanilla batch-normalized networks without skip connections are not trainable at large depths for common initialization schemes, a prediction that we verify with a variety of empirical simulations.

While gradient explosion cannot be eliminated, it can be reduced by tuning the network close to the linear regime, which improves the trainability of deep batch-normalized networks without residual connections.

Finally, we investigate the learning dynamics of batch-normalized networks and observe that after a single step of optimization the networks achieve a relatively stable equilibrium in which gradients have dramatically smaller dynamic range.

Our theory leverages Laplace, Fourier, and Gegenbauer transforms and we derive new identities that may be of independent interest.

Deep neural networks have been enormously successful across a broad range of disciplines.

These successes are often driven by architectural innovations.

For example, the combination of convolutions BID15 , residual connections BID12 , and batch normalization BID13 has allowed for the training of very deep networks and these components have become essential parts of models in vision (Zoph et al.) , language BID4 , and reinforcement learning (Silver et al., 2017) .

However, a fundamental problem that has accompanied this rapid progress is a lack of theoretical clarity.

An important consequence of this gap between theory and experiment is that two important issues become conflated.

In particular, it is generally unclear whether novel neural network components improve generalization or whether they merely increase the fraction of hyperparameter configurations where good generalization can be achieved.

Resolving this confusion has the promise of allowing researchers to more effectively and deliberately design neural networks.

Recently, progress has been made (Poole et al., 2016; Schoenholz et al., 2016; BID6 BID16 BID11 in this direction by considering neural networks at initialization, before any training has occurred.

In this case, the parameters of the network are random variables which induces a distribution of the activations of the network as well as the gradients.

Studying these distributions is equivalent to understanding the prior over functions that these random neural networks compute.

Picking hyperparameters that correspond to well-conditioned priors ensures that the neural network will be trainable and this fact has been extensively verified experimentally.

However, to fulfill its promise of making neural network design less of a black box, these techniques must be applied to neural network architectures that are used in practice.

Over the past year, this gap has closed significantly and theory for networks with skip connections (Yang & Schoenholz, 2017; , convolutional networks (Xiao et al., 2018) , and gated recurrent networks BID3 BID9 have been developed.

More recently, Yang (2019) devised a formalism that extends this approach to include an even wider range of architectures.

Before state-of-the-art models can be analyzed in this framework, a slowly-decreasing number of architectural innovations must be studied.

One particularly important component that has thus-far remained elusive is batch normalization.

Our Contributions.

In this paper, we develop a theory of fully-connected networks with batch normalization whose weights and biases are randomly distributed.

A significant complication in the case of batch normalization (compared to e.g. layer normalization or weight normalization) is that the statistics of the network depend non-locally on the entire batch.

Thus, our first main result is to recast the theory for random fully-connected networks so that it can be applied to batches of data.

We then extend the theory to include batch normalization explicitly and validate this theory against Monte-Carlo simulations.

We show that as in previous cases we can leverage our theory to predict valid hyperparameter configurations.

In the process of our investigation, we identify a number of previously unknown properties of batch normalization that make training unstable.

In particular, for most nonlinearities used in practice, batchnorm in a deep, randomly initialized network induces high degree of symmetry in the embeddings of the samples in the batch (Thm 3.4) .

Whenever this symmetry takes hold, we show that for any choice of nonlinearity, gradients of fully-connected networks with batch normalization explode exponentially in the depth of the network (Thm 3.9) .

This imposes strong limits on the maximum trainable depth of batch normalized networks.

This limit can be lifted partially but not completely by pushing activation functions to be more linear at initialization.

It might seem that such gradient explosion ought to lead to learning dynamics that are unfavorable.

However, we show that networks with batch normalization causes the scale of the gradients to naturally equilibrate after a single step of gradient descent (provided the initial gradients are not so large as to cause numerical instabilities).

For shallower networks, this equilibrating effect is sufficient to allow adequate training.

Finally, we note that there is a related vein of research that has emerged that leverages the prior over functions induced by random networks to perform exact Bayesian inference BID16 BID7 BID18 BID8 .

One of the natural consequences of this work is that the prior for networks with batch normalization can be computed exactly in the wide network limit.

As such, it is now possible to perform exact Bayesian inference in the case of wide neural networks with batch normalization.

Batch normalization has rapidly become an essential part of the deep learning toolkit.

Since then, a number of similar modifications have been proposed including layer normalization BID0 and weight normalization (Salimans & Kingma, 2016) .

Comparisons of performance between these different schemes have been challenging and inconclusive BID10 .

The original introduction of batchnorm in BID13 proposed that batchnorm prevents "internal covariate shift" as an explanation for its effectiveness.

Since then, several papers have approached batchnorm from a theoretical angle, especially following Ali Rahimi's catalyzing call to action at NIPS 2017.

BID1 found that batchnorm in resnets allow deep gradient signal propagation in contrast to the case without batchnorm.

Santurkar et al. (2018) found that batchnorm does not help covariate shift but helps by smoothing loss landscape.

BID2 reached the opposite conclusion as our paper for residual networks with batchnorm, that batchnorm works in this setting because it induces beneficial gradient dynamics and thus allows a much bigger learning rate.

BID17 explores similar ideas that batchnorm allows large learning rates and likewise uses random matrix theory to support their claims.

BID14 identified situations in which batchnorm can provably induce acceleration in training.

Of the above that mathematically analyze batchnorm, all but Santurkar et al. (2018) make simplifying assumptions on the form of batchnorm and typically do not have gradients flowing through the batch variance.

Even Santurkar et al. (2018) only analyzes a vanilla network which gets added a single batchnorm at a single moment in training.

Our analysis here on the other hand works for networks with arbitrarily many batchnorm layers with very general activation functions, and indeed, this deep stacking of batchnorm is precisely what leads to gradient explosion.

It is an initialization time analysis for networks of infinite width, but we show experimentally that the resulting insights predict both training and test time behavior.

We remark that Philipp & Carbonell (2018) has also empirically noted gradient explosion happens in deep batchnorm networks with various nonlinearities.

In this work, in contrast, we develop a precise theoretical characterization of gradient statistics and, as a result, we are able to make significantly stronger conclusions.

We begin with a brief recapitulation of mean field theory in the fully-connected setting.

In addition to recounting earlier results, we rephrase the formalism developed previously to compute statistics of neural networks over a batch of data.

Later, we will extend the theory to include batch normalization.

We consider a fully-connected network of depth L whose layers have width N l , activation function 1 φ, weights W l ∈ R N l−1 ×N l , and biases b l ∈ R N l .

Given a batch of B inputs 2 {x i : x i ∈ R N0 } i=1,··· ,B , the pre-activations of the network are defined by the recurrence relation, DISPLAYFORM0 At initialization, we choose the weights and biases to be i.i.d.

as W In the following, we use α, β, . . .

for the neuron index, and i, j, . . .

for the batch index.

We will be concerned with understanding the statistics of the pre-activations and the gradients induced by the randomness in the weights and biases.

For ease of exposition we will typically take the network to have constant width N l = N .In the mean field approximation, we iteratively replace the pre-activations in Eq. (2) by Gaussian random variables with matching first and second moments.

In the infinite width limit this approximation becomes exact BID16 BID7 .

Since the weights are i.i.d.

with zero mean it follows that the mean of each pre-activation is zero and the covariance between distinct neurons are zero.

The pre-activation statistics are therefore given by (h are B × B covariance matrices and δ α1,···α B is the Kronecker-δ that is one if α 1 = α 2 = · · · = α B and zero otherwise.

Definition 3.1.

Let V be the operator on functions φ such that V φ (Σ) = E[φ(h)φ(h) T : h ∼ N (0, Σ)] computes the matrix of uncentered second moments of φ(h) for h ∼ N (0, Σ).Using the above notation, we can express the covariance matrices by the recurrence relation, DISPLAYFORM1 At first Eq. (2) may seem challenging since the expectation involves a Gaussian integral in R B .

However, each term in the expectation of V φ involves at most a pair of pre-activations and so the expectation may be reduced to the evaluation of O(B 2 ) two-dimensional integrals.

These integrals can either be performed analytically BID5 Williams, 1997) or efficiently approximated numerically BID16 , and so Eq. (2) defines a computationally efficient method for computing the statistics of neural networks after random initialization.

This theme of dimensionality reduction will play a prominent role in the forthcoming discussion on batch normalization.

Eq.

(2) defines a dynamical system over the space of covariance matrices.

Studying the statistics of random feed-forward networks therefore amounts to investigating this dynamical system and is an enormous simplification compared with studying the pre-activations of the network directly.

As is common in the dynamical systems literature, a significant amount of insight can be gained by investigating the behavior of Eq. (2) in the vicinity of its fixed points.

For most common activation functions, Eq. (2) has a fixed point at some Σ * .

Moreover, when the inputs are non-degenerate, this fixed point generally has a simple structure with Σ * = q * [(1 − c * )I + c * 11T ] owing to permutation symmetry among elements of the batch.

We refer to fixed points with such symmetry as BSB1 (Batch Symmetry Breaking 1 or 1 Block Symmetry Breaking) fixed points.

As we will discuss later, in the context of batch normalization other fixed points with fewer symmetries ("BSBk fixed points") may become preferred.

In the fully-connected setting fixed points may efficiently be computed by solving the fixed point equation induced by Eq. (2) in the special case B = 2.

The structure of this fixed point implies that in asymptotically deep feed-forward neural networks all inputs yield pre-activations of identical norm with identical angle between them.

Neural networks that are deep enough so that their pre-activation statistics lie in this regime have been shown to be untrainable (Schoenholz et al., 2016) .Notation As we often talk about matrices and also linear operators over matrices, we write T {Σ} for an operator T applied to a matrix Σ, and matrix multiplication is still written as juxtaposition.

Composition of matrix operators are denoted with T 1 • T 2 .Local Convergence to BSB1 Fixed Point.

To understand the behavior of Eq. (2) near its BSB1 fixed point we can consider the Taylor series in the deviation from the fixed point, ∆Σ l = Σ l − Σ * .

To lowest order we generically find, DISPLAYFORM2 where DISPLAYFORM3 Jacobian of V φ .

In most prior work where φ was a pointwise non-linearity, Eq. (3) reduces to the case B = 2 which naturally gave rise to linearized dynamics in DISPLAYFORM4 However, in the case of batch normalization we will see that one must consider the evolution of Eq. (3) as a whole.

This is qualitatively reminiscent of the case of convolutional networks studied in Xiao et al. (2018) where the evolution of the entire pixel × pixel covariance matrix had to be evaluated.

The dynamics induced by Eq. (3) will be controlled by the eigenvalues of J: Suppose J has eigenvalues λ i -ordered such that λ 1 ≥ λ 2 ≥ · · · ≥ λ B 2 -with associated eigen"vectors" e i (note that the e i will themselves be B × B matrices).

It follows that if ∆Σ 0 = i c i e i for some choice of constants c i then ∆Σ l = i c i λ l i e i .

Thus, if λ i < 1 for all i, ∆Σ l will approach zero exponentially and the fixed-point will be stable.

The number of layers over which Σ will approach Σ * will be given by −1/ log(λ 1 ).

By contrast if λ i > 1 for any i then the fixed point will be unstable.

In this case, there is typically a different, stable, fixed point that must be identified.

It follows that if the eigenvalues of J can be computed then the dynamics will follow immediately.

While J may appear to be a complicated object at first, a moment's thought shows that each diagonal element J{∆Σ} ii is only affected by the corresponding diagonal element ∆Σ ii , and each off-diagonal element J{∆Σ} ij is only affected by ∆Σ ii , ∆Σ jj , and ∆Σ ij .

Such a Diagonal-Offdiagonal Semidirect (DOS) operator has a simple eigendecomposition with two eigenspaces corresponding to changes in the off-diagonal and changes in the diagonal (Thm E.72).

The associated eigenvalues are precisely those calculated by Schoenholz et al. (2016) in a simplified analysis.

DOS operators are a particularly simple form of a more general operator possessing an abundance of symmetries called ultrasymmetric operators, which will play a prominent role in the analysis of batchnorm below.

Gradient Dynamics.

Similar arguments allow us to develop a theory for the statistics of gradients.

The backpropogation algorithm gives an efficient method of propagating gradients from the end of the network to the earlier layers as, DISPLAYFORM5 Here L is the loss function and δ DISPLAYFORM6 L are N l -dimensional vectors that describe the error signal from neurons in the l'th layer due to the i'th element of the batch.

The preceding discussion gave a precise characterization of the statistics of the h l i that we can leverage to understand the statistics of δ l i .

Assuming gradient independence, that is, that an iid set of weights are used during backpropagation (see Appendix B for more discussions), it is easy to see that E[δ is a covariance matrix and we may once again drop the neuron index.

We can construct a recurrence relation to compute Π l , DISPLAYFORM7 Typically, we will be interested in understanding the dynamics of Π l when Σ l has converged exponentially towards its fixed point.

Thus, we study the approximation, DISPLAYFORM8 Since these dynamics are linear (and in fact componentwise), explosion and vanishing of gradients will be controlled by V φ (Σ * ).

We now extend the mean field formalism to include batch normalization.

Here, the definition for the neural network is modified to be the coupled equations, DISPLAYFORM0 where γ α and β α are parameters, and µ α = or so to prevent division by zero, but in this paper, unless stated otherwise (in the last few sections), is assumed to be 0.

Unlike in the case of vanilla fully-connected networks, here the pre-activations are invariant to σ 2 w and σ 2 b .

Without a loss of generality, we therefore set σ 2 w = 1 and σ 2 b = 0 for the remainder of the text.

In principal, batch normalization additionally yields a pair of hyperparameters γ and β which are set to be constants.

However, these may be incorporated into the nonlinearity and so without a loss of generality we set γ = 1 and β = 0.

In order to avoid degenerate results, we assume B ≥ 4 unless stated otherwise; we shall discuss the small B regime in Appendix J.If one treats batchnorm as a "batchwise nonlinearity", then the arguments from the previous section can proceed identically and we conclude that as the width of the network grows, the pre-activations will be jointly Gaussian with identically distributed neurons.

Thus, we arrive at an analogous expression to Eq. (2), DISPLAYFORM1 Here we have introduced the projection operator DISPLAYFORM2 T which is defined such that Gx = x − µ1 with µ = i x i /B.

Unlike φ, B φ does not act component-wise on h. It is therefore not obvious whether V B φ can be evaluated without performing a B-dimensional Gaussian integral.

Theoretical tools.

In this paper, we present several ways to analyze high dimensional integrals like the above: 1. the Laplace method 2. the Fourier method 3.

spherical integration 4. and the Gegenbauer method.

The former two use the Laplace and Fourier transforms to simplify expressions like the above, where the Laplace method requires that φ be positive homogeneous.

Often the Laplace method will give clean, closed form answers for such φ.

Because batchnorm can be thought of as a linear projection (G) followed by projection to the sphere of radius √ B, spherical integration techniques are often very useful and in fact is typically the most straightforward way of numerically evaluating quantities.

Lastly, the Gegenbauer method expresses objects in terms of the Gegenbauer coefficients of φ.

Briefly, Gegenbauer polynomials {C DISPLAYFORM3 are orthogonal polynomials with respect to the measure (1 − x 2 ) α− Theorem 3.2.

Suppose φ : R → R is degree-α positive homogeneous.

For any positive semi-definite matrix Σ define the projection Σ G = GΣG.

Then DISPLAYFORM4 whenever the integral exists.

denoting the (B − 2)-dimensional sphere, DISPLAYFORM5 Together these theorems provide analytic recurrence relations for random neural networks with batch normalization over a wide range of activation functions.

By analogy to the fully-connected case we would like to study the dynamical system over covariance matrices induced by these equations.

We begin by investigating the fixed point structure of Eq. (8).

As in the case of feed-forward networks, permutation symmetry implies that there exist BSB1 fixed points DISPLAYFORM6 We will see that this fixed point is in fact unique, and a clean expression of q * and c * can be obtained in terms of Gegenbauer basis (see Thm F.13).

Thus the entries of the BSB1 fixed point are diagonal quadratic forms of the Gegenbauer coefficients of φ( √ B − 1x).

Even more concise closed forms are available when the activation functions are degree α positive homogeneous (see Thm F.8).

In particular, for ReLU we arrive at the following Theorem 3.5 (BSB1 fixed point for ReLU).

When φ = relu, then DISPLAYFORM0 where DISPLAYFORM1 is the arccosine kernel BID5 .To determine the eigenvalues of DISPLAYFORM2 dΣ Σ=Σ * it is helpful to consider the action of batch normalization in more detail.

In particular, we notice that B φ can be decomposed into the composition of three separate operations, B φ = φ • n • G. As discussed above, Gh subtracts the mean from h and we introduce the new function n(h) = √ Bh/||h|| which normalizes a centered h by its standard deviation.

Applying the chain rule, we can rewrite the Jacobian as, DISPLAYFORM3 where • denotes composition and G

is the natural extension of G to act on matrices as DISPLAYFORM0 It ends up being advantageous to study DISPLAYFORM1 and to note that the nonzero eigenvalues of this object are identical to the nonzero eigenvalues of the Jacobian (see Lemma F.17).At face value, this is a complicated object since it simultaneously has large dimension and possesses an intricate block structure.

However, the permutation symmetry of the BSB1 Σ * induces strong symmetries inĴ that significantly simplify the analysis (see Appendix F.3).

In particular whilê J ijkl is a four-index object, we haveĴ ijkl =Ĵ π(i)π(j)π(k)π(l) for all permutations π on B and J ijkl =Ĵ jilk .

We call linear operators possessing such symmetries ultrasymmetric (Defn E.53) and show that all ultrasymmetric operators conjugated by G

admit an eigendecomposition that contains three distinct eigenspaces with associated eigenvalues (see Thm E.62).Theorem 3.6.

Let T be an ultrasymmetric matrix operator.

Then on the space of symmetric matrices, DISPLAYFORM0 has the following orthogonal (under trace inner product) eigendecomposition, 1. an eigenspace {Σ : Σ G = 0} with eigenvalue 0.2.

a 1-dimensional eigenspace RG with eigenvalue λ DISPLAYFORM1 M .

The specific forms of the eigenvalues can be obtained as linear functions of the entries of T ; see Thm E.62 for details.

Note that, as the eigenspaces are orthogonal, this implies that DISPLAYFORM2 is self-adjoint (even when T is not).In our context with T =Ĵ, the eigenspaces can be roughly interpreted as follows: The deviation ∆Σ = Σ − Σ * from the fixed point decomposes as a linear combination of components in each of the eigenspaces.

The RG-component captures the average norm of elements of the batch (the trace of ∆Σ), the L-component captures the fluctuation of such norms, and the M-component captures the covariances between elements of the batch.

Because of the explicit normalization of batchnorm, one sees immediately that the RG-component goes to 0 after 1 step.

For positive homogeneous φ, we can use the Laplace method to obtain closed form expressions for the other eigenvalues (see Thm F.33).

The below theorem shows that, as the batch size becomes larger, a deep ReLU-batchnorm network takes more layers to converge to a BSB1 fixed point.

Theorem 3.7.

Let φ = relu and B > 3.

The eigenvalues of DISPLAYFORM3 where DISPLAYFORM4 More generally, we can evaluate them for general nonlinearity using spherical integration (Appendix F.3.1) and, more enlightening, using the Gegenbauer method, is the following DISPLAYFORM5 where the coefficients w B−1,l , u B−1,l ,w B−1,l ,ũ B−1,l , v B−1,l are given in Thms F.22 and F.24.A BSB1 fixed point is not locally attracting if λ DISPLAYFORM6 Thus Thm 3.8 yields insight on the stability of the BSB1 fixed point, which we can interpret heuristically as follows.

The specific forms of the coefficients w B−1,l , u B−1,l ,w B−1,l ,ũ B−1,l , v B−1,l show that λ ↑ M is typically much smaller than λ ↑ L (but there are exceptions like φ = sin), and w B−1,1 < v B−1,1 but w B−1,l ≥ v B−1,l for all l ≥ 2.

Thus one expects that, the larger a 1 is, i.e. the "more linear" and less explosive φ is, the smaller λ ↑ L is and the more likely that Eq. (8) converges to a BSB1 fixed point.

This is consistent with the "winner-take-all" intuition for the emergence of BSB2 fixed point explained above.

See Appendix F.3.2 for more discussion.

With a mean field theory of the pre-activations of feed-forward networks with batch normalization having been developed, we turn our attention to the backpropagation of gradients.

In contrast to the case of networks without batch normalization, we will see that exploding gradients at initialization are a severe problem here.

To this end, one of the main results from this section will be to show that fully-connected networks with batch normalization feature exploding gradients for any choice of nonlinearity such that Σ l converges to a BSB1 fixed point.

Below, by rate of gradient explosion we mean the β such that the gradient norm squared grows as β DISPLAYFORM0 with depth L. As before, all computations below assumes gradient independence (see Appendix B for a discussion).As a starting point we seek an analog of Eq. (6) in the case of batch normalization.

However, because the activation functions no longer act point-wise on the pre-activations, the backpropagation equation becomes, DISPLAYFORM1 , . . .

, h l αB ) and we observe the additional sum over the batch.

Computing the resulting covariance matrix Π l , we arrive at the recurrence relation, DISPLAYFORM2 where we have defined the linear operator DISPLAYFORM3 for any vector-indexed linear operator F h .

As in the case of vanilla feed-forward networks, here we will be concerned with the behavior of gradients when Σ l is close to its fixed point.

We therefore study the asymptotic approximation to Eq. (17) given by DISPLAYFORM4 In this case the dynamics of Π are linear and are therefore naturally determined by the eigenvalues of DISPLAYFORM5 As in the forward case, batch normalization is the composition of three operations DISPLAYFORM6 Applying the chain rule, Eq. (17) can be rewritten as, DISPLAYFORM7 with F (Σ) appropriately defined.

Note that since G

is an idempotent operator, DISPLAYFORM0 , so that it suffices to study the eigende- DISPLAYFORM1 .

Due to the symmetry of Σ * , F (Σ * ) is ultrasymmetric, so that DISPLAYFORM2 has eigenspaces RG, L, M and we can compute its eigenvalues via Thm 3.6.

More illuminating, however, is the Gegenbauer expansion (see Thm G.5).

It requires a new identity Thm E.47 involving Gegenbauer polynomials integrated over a sphere, which may be of independent interest.

Theorem 3.9 (Batchnorm causes gradient explosion).

DISPLAYFORM3 2 ) l (x), then gradients explode at the rate of DISPLAYFORM4 in which case gradients explode at the rate of B−2 B−3 .

This contrasts starkly with the case of non-normalized fully-connected networks, which can use the weight and bias variances to control its mean field network dynamics (Poole et al., 2016; Schoenholz et al., 2016) .

As a corollary, we disprove the conjecture of the original batchnorm paper BID13 that "Batch Normalization may lead the layer Jacobians to have singular values close to 1" in the initialization setting, and in fact prove the exact opposite, that batchnorm forces the layer Jacobian singular values away from 1.Appendix G.1 discusses the numerical evaluation of all eigenvalues, and as usual, the Laplace method yields closed forms for positive homogeneous φ (Thm G.12).

We highlight the result for ReLU.

Theorem 3.10.

In a ReLU-batchnorm network, the gradient norm explodes exponentially at the rate of DISPLAYFORM5 which decreases to π π−1 ≈ 1.467 as B → ∞. In contrast, for a linear batchnorm network, the gradient norm explodes exponentially at the rate of B−2 B−3 , which goes to 1 as B → ∞. FIG4 shows theory and simulation for ReLU gradient dynamics.

Weight Gradient While all of the above only study the gradient with respect to the hidden preactivations, Appendix L shows that the weight gradient norms at layer l is just Π l , µ * G = µ * tr Π l , and thus by Thm 3.9, the weight gradients explode as well at the same rate λ ↓ G .

In practice, is usually treated as small constant and is not regarded as a hyperparameter to be tuned.

Nevertheless, we can investigate its effect on gradient explosion.

A straightforward generalization of the analysis presented above to the case of > 0 suggests somewhat larger values than typically used can ameliorate (but not eliminate) gradient explosion problems.

See FIG8 ).

Forward In addition to analyzing the correlation between preactivations of samples in a batch, we also study the correlation between those of different batches.

The dynamics Eq. (8) can be generalized to simultaneous propagation of k batches (see Eq. (54) and Appendix H).

DISPLAYFORM0 Here the domain of the dynamics is the space of block matrices, with diagonal blocks and offdiagonal blocks resp.

representing within-batch and cross-batch covariance.

We observe empirically In (c) we plot the empirical variance of the diagonal and off-diagonal entries of the covariance matrix which clearly shows a jump at the transition.

In (d) we plot representative covariance matrices for the two phases (BSB1 bottom, BSB2 top).Published as a conference paper at ICLR 2019 DISPLAYFORM1 Figure A.1: Batch norm leads to a chaotic input-output map with increasing depth.

A linear network with batch norm is shown acting on two minibatches of size 64 after random orthogonal initialization.

The datapoints in the minibatch are chosen to form a 2d circle in input space, except for one datapoint that is perturbed separately in each minibatch (leftmost datapoint at input layer 0).

Because the network is linear, for a given minibatch it performs an affine transformation on its inputs -a circle in input space remains an ellipse throughout the network.

However, due to batch norm the coefficients of that affine transformation change nonlinearly as the datapoints in the minibatch are changed.

(a) Each pane shows a scatterplot of activations at a given layer for all datapoints in the minibatch, projected onto the top two PCA directions.

PCA directions are computed using the concatenation of the two minibatches.

Due to the batch norm nonlinearity, minibatches that are nearly identical in input space grow increasingly dissimilar with depth.

Intuitively, this chaotic input-output map can be understood as the source of exploding gradients when batch norm is applied to very deep networks, since very small changes in an input correspond to very large movements in network outputs.

(b) The correlation between the two minibatches, as a function of layer, for the a conference paper at ICLR 2019 DISPLAYFORM2 Batch norm leads to a chaotic input-output map with increasing depth.

A linear batch norm is shown acting on two minibatches of size 64 after random orthogonal .

The datapoints in the minibatch are chosen to form a 2d circle in input space, except oint that is perturbed separately in each minibatch (leftmost datapoint at input layer 0).

etwork is linear, for a given minibatch it performs an affine transformation on its inputs input space remains an ellipse throughout the network.

However, due to batch norm ts of that affine transformation change nonlinearly as the datapoints in the minibatch (a) Each pane shows a scatterplot of activations at a given layer for all datapoints tch, projected onto the top two PCA directions.

PCA directions are computed using ation of the two minibatches.

Due to the batch norm nonlinearity, minibatches that entical in input space grow increasingly dissimilar with depth.

Intuitively, this chaotic map can be understood as the source of exploding gradients when batch norm is applied networks, since very small changes in an input correspond to very large movements in uts.

(b) The correlation between the two minibatches, as a function of layer, for the k. Despite having a correlation near one at the input layer, the two minibatches rapidly ith depth.

See ??

for a theoretical treatment.

The datapoints in the minibatch are chosen to form a 2d circle in input space, except for one datapoint that is perturbed separately in each minibatch (leftmost datapoint at input layer 0).

Because the network is linear, for a given minibatch it performs an affine transformation on its inputs -a circle in input space remains an ellipse throughout the network.

However, due to batch norm the coefficients of that affine transformation change nonlinearly as the datapoints in the minibatch are changed.

(a) Each pane shows a scatterplot of activations at a given layer for all datapoints in the minibatch, projected onto the top two PCA directions.

PCA directions are computed using the concatenation of the two minibatches.

Due to the batch norm nonlinearity, minibatches that are nearly identical in input space grow increasingly dissimilar with depth.

Intuitively, this chaotic input-output map can be understood as the source of exploding gradients when batch norm is applied to very deep networks, since very small changes in an input correspond to very large movements in network outputs.

(b) The correlation between the two minibatches, as a function of layer, for the same network.

Despite having a correlation near one at the input layer, the two minibatches rapidly decorrelate with depth.

See ??

for a theoretical treatment.

A.1 VGG19 WITH BATCHNORM ON CIFAR100Even though at initialization time batchnorm causes gradient explosion, after the first few epochs, the relative gradient norms kr ✓ Lk/k✓k for weight parameters ✓ = W or BN scale parameter ✓ = , equilibrate to about the same magnitude.

See Fig. A .2.

14 a bFigure 2: Batch norm leads to a chaotic input-output map with increasing depth.

A linear network with batch norm is shown acting on two minibatches of size 64 after random orthogonal initialization.

The datapoints in the minibatch are chosen to form a 2d circle in input space, except for one datapoint that is perturbed separately in each minibatch (leftmost datapoint at input layer 0).

Because the network is linear, for a given minibatch it performs an affine transformation on its inputs -a circle in input space remains an ellipse throughout the network.

However, due to batch norm the coefficients of that affine transformation change nonlinearly as the datapoints in the minibatch are changed.

(a) Each pane shows a scatterplot of activations at a given layer for all datapoints in the minibatch, projected onto the top two PCA directions.

PCA directions are computed using the concatenation of the two minibatches.

Due to the batch norm nonlinearity, minibatches that are nearly identical in input space grow increasingly dissimilar with depth.

Intuitively, this chaotic input-output map can be understood as the source of exploding gradients when batch norm is applied to very deep networks, since very small changes in an input correspond to very large movements in network outputs.

(b) The correlation between the two minibatches, as a function of layer, for the same network.

Despite having a correlation near one at the input layer, the two minibatches rapidly decorrelate with depth.

See Appendix H for a theoretical treatment.that for most nonlinearities used in practice like ReLU, or even for the identity function, ie no pointwise nonlinearity, the global fixed point of this dynamics is cross-batch BSB1, with diagonal BSB1 blocks, and off-diagonal entries all equal to the same constant cTo interpret this phenomenon, note that, after mean centering of the preactivations, this covariance matrix becomes a multiple of identity.

Thus, deep embedding of batches loses the mutual information between them in the input space.

Qualitatively, this implies that two batches that are similar at the input to the network will become increasingly dissimilar -i.e.

chaotic -as the signal propagates deep into the network.

This loss in fact happens exponentially fast, as illustrated in FIG16 , and as shown theoretically by Thm H.13, at a rate of λ Thus, a deep batchnorm network loses correlation information between two input batches, exponentially fast in depth, no matter what nonlinearity (that induces fixed point of the form given by Eq. FORMULA0 ).

This again contrasts with the case for vanilla networks which can control this rate of information loss by tweaking the initialization variances for weights and biases Poole et al. (2016); Schoenholz et al. (2016) .

The absence of coordinatewise nonlinearity, i.e. φ = id, maximally suppresses both this loss of information as well as the gradient explosion, and in the ideal, infinite batch scenario, can cure both problems.

Having developed a theory for neural networks with batch normalization at initialization, we now explore the relationship between the properties of these random networks and their learning dynamics.

We will see that the trainability of networks with batch normalization is controlled by gradient explosion.

We quantify the depth scale over which gradients explode by ξ = 1/ log λ ↓ G where, as above, λ ↓ G is the largest eigenvalue of the jacobian.

Across many different experiments we will see strong agreement between ξ and the maximum trainable depth.

We first investigate the relationship between trainability and initialization for rectified linear networks as a function of batch size.

The results of these experiments are shown in FIG6 where in each case we plot the test accuracy after training as a function of the depth and the batch size and overlay 16ξ in white dashed lines.

In FIG6 we consider networks trained using SGD on MNIST where we observe that networks deeper than about 50 layers are untrainable regardless of batch size.

In (b) we compare standard batch normalization with a modified version in which the batch size is Figure 4: Gradients in networks with batch normalization quickly achieve dynamical equilibrium.

Plots of the relative magnitudes of (a) the weights (b) the gradients of the loss with respect to the pre-activations and (c) the gradients of the loss with respect to the weights for rectified linear networks of varying depths during the first 10 steps of training.

Colors show step number from 0 (black) to 10 (green).held fixed but batch statistics are computed over subsets of size B. This removes subtle gradient fluctuation effects noted in Smith & Le (2018) .

In (c) we do the same experiment with RMSProp and in (d) we train the networks on CIFAR10.

In all cases we observe a nearly identical trainable region.

It is counter intuitive that training can occur at intermediate depths, from 10 to 50 layers, where there is significant gradient explosion.

To gain insight into the behavior of the network during learning we record the magnitudes of the weights, the gradients with respect to the pre-activations, and the gradients with respect to the weights for the first 10 steps of training for networks of different depths.

The result of this experiment is shown in Fig. 4 .

Here we see that before learning, as expected, the norm of the weights is constant and independent of layer while the gradients feature exponential explosion.

However, we observe that two related phenomena occur after a single step of learning: the weights grow exponentially in the depth and the magnitude of the gradients are stable up to some threshold after which they vanish exponentially in the depth.

This is as the result of the scaling property of batchnorm, where DISPLAYFORM0 The first-step gradients dominate the weights due to gradient explosion, hence the exponential growth in weight norms, and thereafter, the gradients are scaled down commensurately.

Thus, it seems that although the gradients of batch normalized networks at initialization are ill-conditioned, the gradients appear to quickly reach a stable dynamical equilibrium.

While this appears to be beneficial for shallower networks, in deeper ones, the relative gradient vanishing can in fact be so severe as to cause lower layers to mostly stay constant during training.

Aside from numerical issues, this seems to be the primary mechanism through which gradient explosion causes training problems for networks deeper than 50 layers.

As discussed in the theoretical exposition above, batch normalization necessarily features exploding gradients for any nonlinearity that converges to a BSB1 fixed point.

We performed a number of experiments exploring different ways of ameliorating this gradient explosion.

These experiments are shown in FIG8 with theoretical predictions for the maximum trainable depth overlaid; in all cases we see exceptional agreement.

In FIG8 (a,b) we explore two different ways of tuning the degree to which activation functions in a network are nonlinear.

In FIG8 (a) we tune γ ∈ [0, 2] for networks with tanh-activations and note that in the γ → 0 limit the function is linear.

In FIG8 (b) we tune β ∈ [0, 2] for networks with rectified linear activations and we note, similarly, that in the β → ∞ limit the function is linear.

As expected, we see the maximum trainable depth increase significantly with decreasing γ and increasing β.

In FIG8 (c,d) we vary for tanh and rectified linear networks respectively.

In both cases, we observe a critical point at large where gradients do not explode and very deep networks are trainable.

In this work we have presented a theory for neural networks with batch normalization at initialization.

In the process of doing so, we have uncovered a number of counterintuitive aspects of batch normalization and -in particular -the fact that at initialization it unavoidably causes gradients to explode with depth.

We have introduced several methods to reduce the degree of gradient explosion, enabling the training of significantly deeper networks in the presence of batch normalization.

Finally, this work paves the way for future work on more advanced, state-of-the-art, network architectures and topologies.

: Batch norm leads to a chaotic input-output map with increasing depth.

A linear network with batch norm is shown acting on two minibatches of size 64 after random orthogonal initialization.

The datapoints in the minibatch are chosen to form a 2d circle in input space, except for one datapoint that is perturbed separately in each minibatch (leftmost datapoint at input layer 0).

Because the network is linear, for a given minibatch it performs an affine transformation on its inputs -a circle in input space remains an ellipse throughout the network.

However, due to batch norm the coefficients of that affine transformation change nonlinearly as the datapoints in the minibatch are changed.

(a) Each pane shows a scatterplot of activations at a given layer for all datapoints in the minibatch, projected onto the top two PCA directions.

PCA directions are computed using the concatenation of the two minibatches.

Due to the batch norm nonlinearity, minibatches that are nearly identical in input space grow increasingly dissimilar with depth.

Intuitively, this chaotic input-output map can be understood as the source of exploding gradients when batch norm is applied to very deep networks, since very small changes in an input correspond to very large movements in network outputs.

(b) The correlation between the two minibatches, as a function of layer, for the same network.

Despite having a correlation near one at the input layer, the two minibatches rapidly decorrelate with depth.

See Appendix H for a theoretical treatment.

A VGG19 WITH BATCHNORM ON CIFAR100Even though at initialization time batchnorm causes gradient explosion, after the first few epochs, the relative gradient norms ∇ θ L / θ for weight parameters θ = W or BN scale parameter θ = γ, equilibrate to about the same magnitude.

See Fig. 7 . with batchnorm on CIFAR100 with data augmentation.

We use 8 random seeds for each combination, and assign to each combination the median training/validation accuracy over all runs.

We then aggregate these scores here.

In the first row we look at training accuracy with different learning rate vs β initialization at different epochs of training, presenting the max over .

In the second row we do the same for validation accuracy.

In the third row, we look at the matrix of training accuracy for learning rate vs , taking max over β.

In the fourth row, we do the same for validation accuracy.

Following prior literature Schoenholz et al. (2016); Yang & Schoenholz (2017); Xiao et al. (2018) , in this paper, in regards to computations involving backprop, we assume Assumption 2.

During backpropagation, whenever we multiply by W T for some weight matrix W , we multiply by an iid copy instead.

Fig. 8 , except we don't take the max over the unseen hyperparameter but rather set it to 0 (the default value).As in these previous works, we find excellent agreement between computations made under this assumption and the simulations (see FIG4 ).

Yang (2019) in fact recently rigorously justified this assumption as used in the computation of moments, for a wide variety of architectures, like multilayer perceptron (Schoenholz et al., 2016) , residual networks (Yang & Schoenholz, 2017) , and convolutional networks (Xiao et al., 2018) studied previously, but without batchnorm.

The reason that their argument does not extend to batchnorm is because of the singularity in its Jacobian at 0.

However, as observed in our experiments, we expect that a proof can be found to extend Yang & Schoenholz (2017)'s work to batchnorm.

C NOTATIONS Definition C.1.

Let S B be the space of PSD matrices of size B × B. Given a measurable function DISPLAYFORM0 When φ : R → R and B is clear from context, we also write V φ for V applied to the function acting coordinatewise by φ.

DISPLAYFORM1 be batchnorm (applied to a batch of neuronal activations) followed by coordinatewise applications of φ, DISPLAYFORM2 When φ = id we will also write B = B id .

When B is clear from context, we will suppress the subscript/superscript B. In short, for h ∈ R B , Gh zeros the sample mean of h. G is a projection matrix to the subspace of vectors h ∈ R B of zero coordinate sum.

With the above definitions, we then have DISPLAYFORM3 We use Γ to denote the Gamma function, P (a, b) := Γ(a + b)/Γ(a) to denote the Pochhammer symbol, and Beta(a, b) = DISPLAYFORM4 Γ(a+b) to denote the Beta function.

We use ·, · to denote the dot product for vectors and trace inner product for matrices.

Sometimes when it makes for simpler notation, we will also use · for dot product.

We adopt the matrix convention that, for a multivariate function f : R n → R m , the Jacobian is DISPLAYFORM5 , where x and ∆x are treated as column vectors.

In what follows we further abbreviate DISPLAYFORM6 ∆x.

As discussed in the main text, we are interested in several closely related dynamics induced by batchnorm in a fully-connected network with random weights.

One is the forward propagation equation Eq. (43) DISPLAYFORM0 studied in Appendix F. Another is the backward propagation equation Eq. (51) DISPLAYFORM1 studied in Appendix G. We will also study their generalizations to the simultaneous propagation of multiple batches in Appendices H and I.In general, we discover several ways to go about such analyses, each with their own benefits and drawbacks:• The Laplace Method (Appendix E.1): If the nonlinearity φ is positive homogeneous, we can use Schwinger's parametrization to turn V B φ and V B φ into nicer forms involving only 1 or 2 integrals, as mentioned in the main text.

All relevant quantities, such as eigenvalues of the backward dynamics, can be further simplified to closed forms.

Lemma E.2 gives the master equation for the Laplace method.

Because the Laplace method requires φ to be positive homogeneous to apply, we review relevant results of such functions in Appendices E.1.1 and E.1.2.• The Fourier Method (Appendix E.2): For polynomially bounded and continuous nonlinearity φ (for the most general set of conditions, see Thm E.25), we can obtain similar simplifications using the Fourier expansion of the delta function, with the penalty of an additional complex integral.• Spherical Integration (Appendix E.3): Batchnorm can naturally be interpreted as a linear projection followed by projection to a sphere of radius √ B. Assuming that the forward dynamics converges to a BSB1 fixed point, one can express V B φ and V B φ as spherical integrals, for all measurable φ.

One can reduce this (B −1)-dimensional integral, in spherical angular coordinates, into 1 or 2 dimensions by noting that the integrand depending on φ only depends on 1 or 2 angles.

Thus this method is very suitable for numerical evaluation of quantities of interest for general φ.• The Gegenbauer Method (Appendix E.4): Gegenbauer polynomials are orthogonal polynomials under the weight (1 − x 2 ) α for some α.

They correspond to a special type of spherical harmonics called zonal harmonics (Defn E.44) that has the very useful reproducing property (Fact E.46).

By expressing the nonlinearity φ in this basis, we can see easily that the relevant eigenvalues of the forward and backward dynamics are ratios of two quadratic forms of φ's Gegenbauer coefficients.

In particular, for the largest eigenvalue of the backward dynamics, the two quadratic forms are both diagonal, and in a way that makes apparent the necessity of gradient explosion (under the BSB1 fixed point assumption).

In regards to numerical computations, the Gegenbauer method has the benefit of only requiring 1-dimensional integrals to obtain the coefficients, but if φ has slowlydecaying coefficients, then a large number of such integrals may be required to get an accurate answer.

In each of the following sections, we will attempt to conduct an analysis with each method when it is feasible.

When studying the convergence rate of the forward dynamics and the gradient explosion of the backward dynamics, we encounter linear operators on matrices that satisfy an abundance of symmetries, called ultrasymmetric operators.

We study these operators in Appendix E.5 and contribute several structural results Thms E.61, E.62, E.72 and E.72 on their eigendecomposition that are crucial to deriving the asymptotics of the dynamics.

All of the above will be explained in full in the technique section Appendix E. We then proceed to study the forward and backward dynamics in detail (Appendices F to I), now armed with tools we need.

Up to this point, we only consider the dynamics assuming Eq. FORMULA2 converges to a BSB1 fixed point, and B ≥ 4.

In Appendix J we discuss what happens outside this regime, and in Appendix K we show our current understanding of BSB2 fixed point dynamics.

Note that the backward dynamics studied in these sections is that of the gradient with respect to the hidden preactivations.

Nevertheless, we can compute the moments of the weight gradients from this, which is of course what is eventually used in gradient descent.

We do so in the final section Appendix L.Main Technical Results Corollary F.3 establishes the global convergence of Eq. (43) to BSB1 fixed point for φ = id (as long as the initial covariance is nondegenerate), but we are not able to prove such a result for more general φ.

We thus resort to studying the local convergence properties of BSB1 fixed points.

First, Thms F.5, F.8 and F.13 respectively compute the BSB1 fixed point from the perspectives of spherical integration, the Laplace method, and the Gegenbauer method.

Thms F.22 and F.24 give the Gegenbauer expansion of the BSB1 local convergence rate, and Thm F.33 gives a more succinct form of the same for positive homogeneous φ.

Appendix F.3.1 discusses how to do this computation via spherical integration.

Next, we study the gradient dynamics, and Thms G.5 and G.12 yield the gradient explosion rates with the Gegenbauer method and the Laplace method (for positive homogeneous φ), respectively.

Appendix G.1.1 discusses how to compute this for general φ using spherical integration.

We then turn to the dynamics of cross batch covariances.

Thm H.3 gives the form of the cross batch fixed point via spherical integration and the Gegenbauer method, and Thm H.6 yields a more specific form for positive homogeneous φ via the Laplace method.

The local convergence rate to this fixed point is given by Thms H.13 and H.14, respectively using the Gegenbauer method and the Laplace method.

Finally, the correlation between gradients of the two batches is shown to decrease exponentially in Corollary I.6 using the Gegenbauer method, and the decay rate is given succinctly for positive homogeneous φ in Thm I.3.

We introduce the key techniques and tools in our analysis in this section.

As discussed above, the Laplace method is useful for deriving closed form expressions for positive homogeneous φ.

The key insight here is to apply Schwinger parametrization to deal with normalization.

Lemma E.1 (Schwinger parametrization).

For z > 0 and c > 0, DISPLAYFORM0 The following is the key lemma in the Laplace method.

is well-defined and continuous, and furthermore satisfies DISPLAYFORM0 Proof.

℘(Σ) is well-defined for full rank Σ because the y −2ksingularity at y = 0 is Lebesgueintegrable in a neighborhood of 0 in dimension A > 2k.

We prove Eq. (25) in the case when Σ is full rank and then apply a continuity argument.

Proof of Eq. (25) for full rank Σ. First, we will show that we can exchange the order of integration DISPLAYFORM1 by Fubini-Tonelli's theorem.

Observe that DISPLAYFORM2 as s → ∞, by dominated convergence with dominating function h( DISPLAYFORM3 .

By the same reasoning, the function s → E[ f (y) : y ∼ N (0, Σ(I + 2sΣ) −1 )] is continuous.

In particular this implies that sup 0≤s≤∞ E[ f (y) : y ∼ N (0, Σ(I + 2sΣ) −1 )] < ∞. Combined with the fact that det(I + 2sΣ) DISPLAYFORM4 which is bounded by our assumption that A/2 > k. This shows that we can apply Fubini-Tonelli's theorem to allow exchanging order of integration.

Thus, DISPLAYFORM5 Domain and continuity of ℘(Σ).

The LHS of Eq. FORMULA1 , ℘(Σ), is defined and continuous on DISPLAYFORM6 , where M is a full rank A × C matrix with rank Σ = C ≤ A, then DISPLAYFORM7 This is integrable in a neighborhood of 0 iff C > 2k, while it's always integrable outside a ball around 0 because f by itself already is.

So ℘(Σ) is defined whenever rank Σ > 2k.

Its continuity can be established by dominated convergence.

−1 )] is bounded in s and det(I + 2sΣ) −1/2 = Θ(s − rank Σ/2 ), so that the integral exists iff rank Σ/2 > k.

To summarize, we have proved that both sides of Eq. (25) are defined and continous for rank Σ > 2k.

Because the full rank matrices are dense in this set, by continuity Eq. (25) holds for all rank Σ > 2k.

If φ is degree-α positive homogeneous, i.e. φ(ru) = r α φ(u) for any u ∈ R, r ∈ R + , we can apply Lemma E.2 with k = α, DISPLAYFORM8 where D = Diag(Σ).

Here BID5 .

DISPLAYFORM9

Published as a conference paper at ICLR 2019Matrix simplification.

We can simplify the expression G(I + 2sΣG) −1 ΣG, leveraging the fact that G is a projection matrix.

Definition E.3.

Let e be an B × (B − 1) matrix whose columns form an orthonormal basis of DISPLAYFORM0 an orthogonal matrix.

For much of this paper e can be any such basis, but at certain sections we will consider specific realizations of e for explicit computation.

From easy computations it can be seen that G =ẽ DISPLAYFORM1 , v is a column vector and a is a scalar.

Then Σ = e T Σe and DISPLAYFORM2 T is block lower triangular, and DISPLAYFORM3 where Definition E.4.

For any matrix Σ, write DISPLAYFORM4 Similarly, det(I B + 2sΣG) = det(I B−1 + 2sΣ ) = det(I B + 2sΣ G ).

So, altogether, we have Theorem E.5.

Suppose φ : R → R is degree-α positive homogeneous.

Then for any B × (B − 1) matrix e whose columns form an orthonormal basis of im G := {Gv : v ∈ R B } = {w ∈ R B : DISPLAYFORM5 Now, in order to use Laplace's method, we require φ to be positive homogeneous.

What do these functions look like?

The most familiar example to a machine learning audience is most likely ReLU.

It turns out that 1-dimensional positive homogeneous functions can always be described as a linear combination of powers of ReLU and its reflection across the y-axis.

Below, we review known facts about these functions and their integral transforms, starting with the powers of ReLU in Appendix E.1.1 and then onto the general case in Appendix E.1.2.

Recall that α-ReLUs (Yang & Schoenholz, 2017) are, roughly speaking, the αth power of ReLU.

Definition E.6.

The α-ReLU function ρ α : R → R sends x → x α when x > 0 and x → 0 otherwise.

This is a continuous function for α > 0 but discontinuous at 0 for all other α.

We briefly review what is currently known about the V and W transforms of ρ α BID5 Yang & Schoenholz, 2017) .

DISPLAYFORM0

Published as a conference paper at ICLR 2019 When considering only 1-dimensional Gaussians, V ρα is very simple.

Proposition E.8.

DISPLAYFORM0 To express results of V ρα on S B for higher B, we first need the following Definition E.9.

Define DISPLAYFORM1 (1 − cos θ cos η) 1+α and J α (c) = J α (arccos c) for α > −1/2.Then Proposition E.10.

For any Σ ∈ S B , let D be the diagonal matrix with the same diagonal as Σ. Then DISPLAYFORM2 where J α is applied entrywise.

For example, J α and J α for the first few integral α are DISPLAYFORM3 One can observe very easily that BID6 Yang & Schoenholz, 2017 ) Proposition E.11.

For each α > −1/2, J α (c) is an increasing and convex function on c ∈ [0, 1], and is continous on c ∈ [0, 1] and smooth on c ∈ (0, 1).

DISPLAYFORM4 , and DISPLAYFORM5 Yang & Schoenholz (2017) also showed the following fixed point structure Theorem E.12.

For α ∈ [1/2, 1), J α (c) = c has two solutions: an unstable solution at 1 ("unstable" meaning J α (1) > 1) and a stable solution in c * ∈ (0, 1) ("stable" meaning J α (c * ) < 1).The α-ReLUs satisfy very interesting relations amongst themselves.

For example, Lemma E.13.

Suppose α > 1.

Then DISPLAYFORM6 In additional, surprisingly, one can use differentiation to go from α to α + 1 and from α to α − 1!

Proposition E.14 (Yang & Schoenholz FORMULA0 ).

Suppose α > 1/2.

Then DISPLAYFORM7 Published as a conference paper at ICLR 2019We have the following from Cho & Saul (2009) Proposition E.15 BID5 ).

For all α ≥ 0 and integer n ≥ 1 DISPLAYFORM8 This implies in particular that we can obtain J α from J α and J α+1 .

Proposition E.16.

For all α ≥ 0, DISPLAYFORM9 Proof.

DISPLAYFORM10 Note that we can also obtain this via Lemma E.13 and Proposition E.14.

Suppose for some α ∈ R, φ : R → R is degree α positive-homogeneous, i.e. φ(rx) = r α φ(x) for any x ∈ R, r > 0.

The following simple lemma says that we can always express φ as linear combination of powers of α-ReLUs.

Proposition E.17.

Any degree α positive-homogeneous function φ : R → R with φ(0) = 0 can be written as x → aρ α (x) − bρ α (−x).Proof.

Take a = φ(1) and b = φ(−1).

Then positive-homogeneity determines the value of φ on R \ {0} and it coincides with x → aρ α (x) − bρ α (−x).As a result we can express the V and W transforms of any positive-homogeneous function in terms of those of α-ReLUs.

Proposition E.18.

Suppose φ : R → R is degree α positive-homogeneous.

By Proposition E.17, φ restricted to R \ {0} can be written as x → aρ α (x) − bρ α (−x) for some a and b.

Then for any PSD 2 × 2 matrix M , DISPLAYFORM0 Proof.

We directly compute, using the expansion of φ into ρ α s: DISPLAYFORM1 where in Eq. FORMULA1 we used negation symmetry of centered Gaussians.

The case of V φ (M ) 22 is similar.

DISPLAYFORM2 where in the last equation we have applied Proposition E.10.This then easily generalizes to PSD matrices of arbitrary dimension: Corollary E.19.

Suppose φ : R → R is degree α positive-homogeneous.

By Proposition E.17, φ restricted to R \ {0} can be written as x → aρ α (x) − bρ α (−x) for some a and b. Let Σ ∈ S B .

Then DISPLAYFORM3 where J φ is defined below and is applied entrywise.

Explicitly, this means that for all i, DISPLAYFORM4 Definition E.20.

Suppose φ : R → R is degree α positive-homogeneous.

By Proposition E.17, φ restricted to R \ {0} can be written as x → aρ α (x) − bρ α (−x) for some a and b. Define DISPLAYFORM5 Let us immediately make the following easy but important observations.

DISPLAYFORM6 Proof.

Use the fact that J α (−1) = 0 and DISPLAYFORM7 by Proposition E.11.As a sanity check, we can easily compute that J id (c) = 2J 1 (c) − 2J 1 (−c) = 2c because id(x) = relu(x)−relu(−x).

By Corollary E.19 and c 1 = 1 2 , this recovers the obvious fact that V (id) (Σ) = Σ. We record the partial derivatives of V φ .

Proposition E.22.

Let φ be positive homogeneous of degree α.

Then for all i with Σ ii = 0, DISPLAYFORM8 where c ij = Σ ij / Σ ii Σ jj and J φ denotes its derivative.

DISPLAYFORM9 This proves the first equation.

With Proposition E.16, DISPLAYFORM10 This gives the second equation.

Expanding J α+1 (0) With Proposition E.11, we get DISPLAYFORM11 2 ) Unpacking the definition of Γ(α + 3 2 ) then yields the third equation.

In general, we can factor diagonal matrices out of V φ .Proposition E.24.

For any Σ ∈ S B , D any diagonal matrix, and φ positive-homogeneous with degree α, DISPLAYFORM12 The Laplace method crucially used the fact that we can pull out the norm factor Gh out of φ, so that we can apply Schwinger parametrization.

For general φ this is not possible, but we can apply some wishful thinking and proceed as follows DISPLAYFORM13 Here all steps are legal except for possibly Eq. FORMULA1 has norm 1 and is not integrable.

Finally, in Eq. (31), we need to extend the definition of Gaussian to complex covariance matrices, via complex Gaussian integration.

This last point is no problem, and we will just define DISPLAYFORM14 for general complex Σ, whenever Σ is nonsingular and this integral is exists, and DISPLAYFORM15 T Σe is nonsingular (as in the case above).

The definition of V φ stays the same given the above extension of the definition of Gaussian expectation.

Nevertheless, the above derivation is not correct mathematically due to the other points.

However, its end result can be justified rigorously by carefully expressing the delta function as the limit of mollifiers.

DISPLAYFORM16 exists and is uniformly bounded by some number possibly depending on a and b. DISPLAYFORM17 −1 z |φ(z a /r)φ(z b /r)| exists and is finite.4.

DISPLAYFORM18 exists and is finite for each r > 0.

DISPLAYFORM19 Similarly, DISPLAYFORM20 If G is the mean-centering projection matrix and Σ G = GΣG, then by the same reasoning as in Thm E.5, DISPLAYFORM21 Note that assumption (1) is satisfied if φ is continuous; assumption (4) is satisfied if D ≥ 3 and for all Π, V φ (Π) ≤ Π α for some α ≥ 0; this latter condition, as well as assumptions FORMULA1 and (3), will be satisfied if φ is polynomially bounded.

Thus the common coordinatewise nonlinearities ReLU, identity, tanh, etc all satisfy these assumptions.

Warning: in general, we cannot swap the order of integration as DISPLAYFORM22 because the s-integral in the latter diverges (in a neighborhood of 0).Proof.

We will only prove the first equation; the others follow similarly.

By dominated convergence, DISPLAYFORM23 Let η : R → R be a nonnegative bump function (i.e. compactly supported and smooth) with support [−1, 1] and integral 1 such that η (0) = η (0) = 0.

Then its Fourier transformη(t) decays like DISPLAYFORM24 and pointwise almost everywhere.

Now, we will show that DISPLAYFORM25 by dominated convergence.

Pointwise convergence is immediate, because DISPLAYFORM26 as → 0 (where we used assumption FORMULA0 DISPLAYFORM27 Finally, we construct a dominating integrable function.

Observe DISPLAYFORM28 For small enough then, this is integrable by assumption (2), and yields a dominating integrable function for our application of dominated convergence.

In summary, we have just proven that DISPLAYFORM29 Note that the absolute value of the integral is bounded above by DISPLAYFORM30 for some C, by our construction of η thatη(t) = O(t −2 ) for large |t|.

By assumption (3), this integral exists.

Therefore we can apply the Fubini-Tonelli theorem and swap order of integration.

DISPLAYFORM31 is integrable (where we used the fact thatη is bounded -which is true for all η ∈ C ∞ 0 (R)), so by dominated convergence (applied to 0), DISPLAYFORM32 where we used the fact thatη(0) = R η(x) dx = 1 by construction.

This gives us the desired result after putting back in some constants and changing r 2 →

s.

As mentioned above, the scale invariance of batchnorm combined with the BSB1 fixed point assumption will naturally lead us to consider spherical integration.

Here we review several basic facts.

Definition E.26.

We define the spherical angular coordinates in R DISPLAYFORM0 x 1 = r cos θ 1 x 2 = r sin θ 1 cos θ 2 x 3 = r sin θ 1 sin θ 2 cos θ 3 . . .

DISPLAYFORM1 . . .

DISPLAYFORM2 as the unit norm version of x. The integration element in this coordinate satisfies DISPLAYFORM3 Fact E.27.

If F : S B−2 → R, then its expectation on the sphere can be written as The following integrals will be helpful for simplifying many expressions involving spherical integration using angular coordinates.

DISPLAYFORM4 DISPLAYFORM5 By antisymmetry of cos with respect to DISPLAYFORM6 Proof.

Set x := cos 2 θ =⇒ dx = −2 cos θ sin θ dθ.

So the integral in question is DISPLAYFORM7 As consequences, Lemma E.29.

For j, k ≥ 0, DISPLAYFORM8 ) .

Lemma E.30.

DISPLAYFORM0 Proof.

By Lemma E.29, DISPLAYFORM1 .

Thus this product of integrals is equal to DISPLAYFORM2 Proof.

Apply change of coordinates z = r 2 /2µ. is nonsingular, then DISPLAYFORM3 DISPLAYFORM4 is such that e T Σe is nonsingular (where e is as in Defn E.3) and DISPLAYFORM5 In particular, DISPLAYFORM6 Proof.

We will prove the second statement.

The others follow trivially.

DISPLAYFORM7 As a straightforward consequence, Proposition E.34.

If e T Σe is nonsingular, DISPLAYFORM8 For numerical calculations, it is useful to realize e as the matrix whose columns are e B−m := DISPLAYFORM9 . . .

DISPLAYFORM10 We have DISPLAYFORM11 (ev) 4 = cos θ 1 + sin θ 1 cos θ 2 + sin θ 1 sin θ 2 cos θ 3 − B − 4 B − 3 sin θ 1 sin θ 2 sin θ 3 cos θ 4 (34) so in particular they only depend on θ 1 , . . .

, θ 4 .Angular coordinates In many situations, the sparsity of this realization of e combined with sufficient symmetry allows us to simplify the high-dimensional spherical integral into only a few dimensions, using one of the following lemmas DISPLAYFORM12 where v i = x i / x and r = x , and DISPLAYFORM13 where v i = x i / x and r = x , and DISPLAYFORM14 where v i = x i / x and r = x .We will prove Lemma E.35; those of Lemma E.36 and Lemma E.37 are similar.

Proof of Lemma E.35.

DISPLAYFORM15 Cartesian coordinates.

We can often also simplify the high dimensional spherical integrals without trigonometry, as the next two lemmas show.

Both are proved easily using change of coordinates.

DISPLAYFORM16 where ω n−1 is hypersurface of the (n − 1)-dimensional unit hypersphere.

Lemma E.39.

For any DISPLAYFORM17 where ω n−1 is hypersurface of the (n − 1)-dimensional unit hypersphere.

Definition E.40.

The Gegenbauer polynomials {C DISPLAYFORM0 l (x) = l, are the set of orthogonal polynomials with respect to the weight function DISPLAYFORM1 .

By convention, they are normalized so that DISPLAYFORM2 Here are several examples of low degree Gegenbauer polynomials DISPLAYFORM3 They satisfy a few identities summarized below DISPLAYFORM4 DISPLAYFORM5 DISPLAYFORM6 By repeated applications of Eq. FORMULA2 , Proposition E.43.

DISPLAYFORM7 This proposition is useful for the Gegenbauer expansion of the local convergence rate of Eq. (43).Zonal harmonics.

The Gegenbauer polynomials are closely related to a special type of spherical harmonics called zonal harmonics, which are intuitively those harmonics that only depend on the "height" of the point along some fixed axis.

DISPLAYFORM8 In our applications of the reproducing property, u and v will always beê a,: := e a,: / e a,: = e a,: B B−1 , where e a,: is the ath row of e.

Next, we present a new (to the best of our knowledge) identity showing that a certain quadratic form of φ, reminiscent of Dirichlet forms, depending only on the derivative φ through a spherical integral, is diagonalized in the Gegenbauer basis.

This will be crucial to proving the necessity of gradient explosion under the BSB1 fixed point assumption.

2 ) and φ( √ B − 1x) has Gegenbauer expansion DISPLAYFORM0 This is proved using the following lemmas.

Lemma E.48.

DISPLAYFORM1 where [n] 2 is the sequence {(n%2), (n%2) + 1, . . .

, n − 2, n} and n%2 is the remainder of n/2.Proof.

Apply Eq. (36) and Eq. (37) alternatingly.

Lemma E.49.

DISPLAYFORM2 Proof of Thm E.47.

We have DISPLAYFORM3 where [n, m] 2 is the sequence n, n + 2, . . .

, m. So for any u 1 , u 2 ∈ S B−2 , DISPLAYFORM4

Published as a conference paper at ICLR 2019 For l < m and m − l even, the coefficient of a l a m is DISPLAYFORM0 For each l, the coefficient of a 2 l is c DISPLAYFORM1 DISPLAYFORM2 2 ), we have DISPLAYFORM3 For l < m and m − l even, the coefficient of a l a m is DISPLAYFORM4 By Lemma E.49 and Eqs. FORMULA2 and FORMULA0 , the coefficient of a l a m with m > l in 2 ) l (u 1 · u 2 ).

DISPLAYFORM5

As remarked before, all commonly used nonlinearities ReLU, tanh, and so on induce the dynamics Eq. (43) to converge to BSB1 fixed points, which we formally define as follows.

Definition E.50.

We say a matrix Σ ∈ S B is BSB1 (short for "1-Block Symmetry Breaking") if Σ has one common entry on the diagonal and one common entry on the off-diagonal, i.e. DISPLAYFORM0 We will denote such a matrix as BSB1(a, b).

Note that BSB1(a, b) can be written as (a−b)I +b11T .

Its spectrum is given below.

Lemma E.51.

A B × B matrix of the form µI + ν11 T has two eigenvalues µ and Bν + µ, each with eigenspaces {x : i x i = 0} and {x : x 1 = · · · = x B }.

Equivalently, if it has a on the diagonal and b on the off-diagonal, then the eigenvalues are a − b and (B − 1)b + a.

The following simple lemma will also be very useful Lemma E.52.

DISPLAYFORM1 Proof.

G and BSB1 B (a, b) can be simultaneously diagonalized by Lemma E.51.

Note that G zeros out the eigenspace R1, and is identity on its orthogonal complement.

The result then follows from easy computations.

The symmetry of BSB1 covariance matrices, especially the fact that they are isotropic in a codimension 1 subspace, is crucial to much of our analysis.

Ultrasymmetry.

Indeed, as mentioned before, when investigating the asymptotics of the forward or backward dynamics, we encounter 4-tensor objects, such as Then we say T is ultrasymmetric.

Remark E.54.

In what follows, we will often "normalize" the representation " [ij|kl] " to the unique "[i j |k l ]" that is in the same equivalence class according to Defn E.53 and such that i , j , k , l ∈ [4] and i ≤ j , k ≤ l unless i = l , j = k , in which case the normalization is [12|21].

Explicitly, we have the following equivalence classes and their normalized representations We will study the eigendecomposition of an ultrasymmetric T as well as the projection for the space of symmetric matrices Σ of dimension B such that GΣG = Σ (which is equivalent to saying rows of Σ sum up to 0).

DISPLAYFORM2 As in the case of S, we omit subscript B when it's clear from context.

Published as a conference paper at ICLR 2019 DISPLAYFORM0 is an orthogonal decomposition w.r.t Frobenius inner product.

For the eigenspaces of T , we also need to define Definition E.57.

For any nonzero a, b ∈ R, set DISPLAYFORM1 In general, we say a matrix DISPLAYFORM2 Proof.

GLG can be written as the sum of outer products DISPLAYFORM3 We are now ready to discuss the eigendecomposition of ultrasymmetric operators.

Published as a conference paper at ICLR 2019

Here S B · W denotes the linear span of the orbit of matrix W under simultaneous permutation of its column and rows (by the same permutation), and DISPLAYFORM0 and λ DISPLAYFORM1 and λ DISPLAYFORM2 are the roots to the quadratic DISPLAYFORM3 3.

Eigenspace M (dimension B(B − 3)/2) with eigenvalue λ DISPLAYFORM4 The proof is by careful, but ultimately straightforward, computation.

Proof.

We will use the bracket notation of Defn E.53 to denote entries of T , and implicitly simplify it according to Remark E.54.

Item 1.

Let U ∈ R B×B be the BSB1 matrix.

By ultrasymmetry of T and BSB1 symmetry of A, T {U } is also BSB1.

So we proceed to calculate the diagonal and off-diagonal entries of T {G}.

Thus BSB1(ω 1 , γ 1 ) and BSB1(ω 2 , γ 2 ) are the eigenmatrices of T , where (ω 1 , γ 1 ) and (ω 2 , γ 2 ) are the eigenvectors of the matrix α 11 α 12 α 21 α 22 DISPLAYFORM0 The eigenvalues are the two roots λ T BSB1,1 , λ T BSB1,2 to the quadratic x 2 − (α 11 + α 22 )x + α 11 α 22 − α 12 α 21 and the corresponding eigenvectors are DISPLAYFORM1 Item 2.

We will study the image of L B (a, b) (Defn E.57) under T .

We have DISPLAYFORM2 and λ DISPLAYFORM3 are the roots of the equation DISPLAYFORM4 Similarly, any image of these eigenvectors under simultaneous permutation of rows and columns remains eigenvectors with the same eigenvalue.

This derives Item 2.Item 3.

Let M ∈ M. We first show that T {M } has zero diagonal.

We have DISPLAYFORM5 = 0 + 0 = 0 which follows from M 1 = 0 by definition of M. Similarly T {M } ii = 0 for all i.

Now we show that M is an eigenmatrix.

DISPLAYFORM6 Note that the eigenspaces described above are in general not orthogonal under trace inner product, so T is not self-adjoint (relative to the trace inner product) typically.

However, as we see next, after projection by G

, it is self-adjoint.

Theorem E.62 (Eigendecomposition of a projected ultrasymmetric operator).

Let T : R B×B → R B×B be an ultrasymmetric linear operator.

We write DISPLAYFORM0 has the following eigendecomposition.1.

Eigenspace RG with eigenvalue λ G,T G := B −1 ((B − 1)(α 11 − α 21 ) − (α 12 − α 22 )), where as in Thm E.61, DISPLAYFORM1 2.

Eigenspace L with eigenvalue λ G,T L := B −1 ((B − 2)β 11 + β 12 + 2(B − 2)β 21 + 2β 22 ), where as in Thm E.61, For a = B − 1, b = −1 so that BSB1(B − 1, −1) = BG, we get DISPLAYFORM2 DISPLAYFORM3 DISPLAYFORM4 So with a = B − 2, b = 1, we have DISPLAYFORM5 by Lemma E.60.Item 3.

The proof is exactly the same as in that of Thm E.61.Noting that the eigenspaces of Thm E.62 are orthogonal, we have Proposition E.63.

G ⊗2 • T H G for any ultrasymmetric operator T is self-adjoint.

In some cases, such as the original study of vanilla tanh networks by Schoenholz et al. (2016) , the ultrasymmetric operator involved has a much simpler form: Definition E.64.

Let T : H B → H B be such that for any Σ ∈ H B and any i = j ∈ [B], DISPLAYFORM0 Then we say that T is diagonal-off-diagonal semidirect, or DOS for short.

We write more specifically T = DOS B (u, v, w).Thm E.62 and Thm E.61 still hold for DOS operators, but we can simplify the results and reason about them in a more direct way.

Lemma E.65.

DISPLAYFORM1 Proof.

Let L := L B (a, b).

It suffices to verify the following, each of which is a direct computation.1.

T {L} 3:B,3:B = 0.2.

T {L} 1,2 = T {L} 2,1 = 0 DISPLAYFORM2 and L B (0, 1) are its eigenvectors: DISPLAYFORM3 Proof.

The map (a, b) → (ua, wb − va) has eigenvalues u and w with corresponding eigenvectors (w − u, v) and (0, 1).

By Lemma E.65, this implies the desired results.

Lemma E.67.

DISPLAYFORM4 Proof.

By Lemma E.60 and Lemma E.65, DISPLAYFORM5 .

By permutation symmetry, we also have the general result for any L B (B − 2, 1)-shaped matrix L. Since they span L B , this gives the conclusion we want.

Lemma E.68.

Let T := DOS B (u, v, w).

Then T {BSB1 B (a, b)} = BSB1 B (ua, wb + 2va).Proof.

Direct computation.

Lemma E.69.

Let T := DOS B (u, v, w) .

Then BSB1 B (w − u, wv) and BSB1 B (0, 1) are eigenvectors of T .

DISPLAYFORM6 Proof.

The linear map (a, b) → (ua, wb + 2va) has eigenvalues u and w with corresponding eigenvectors (u − w, 2v) and (0, 1).

The result then immediately follows from Lemma E.68.

DISPLAYFORM7 Proof.

Direct computation with Lemma E.68 and Lemma E.52Definition E.71.

Define M B := {Σ ∈ S B : DiagΣ = 0} (so compared to M, matrices in M do not need to have zero row sums).

In addition, for any a, b ∈ R, set DISPLAYFORM8 ) where P 1i is the permutation matrix that swap the first entry with the ith entry, i.e. L B (a, b) is the span of the orbit of L B (a, b) under permuting rows and columns simultaneously.

a, b) .

Theorem E.72.

Let T := DOS B (u, v, w) .

Suppose w = u. Then T H B has the following eigendecomposition: DISPLAYFORM9 DISPLAYFORM10 Proof.

The case of M B is obvious.

We will show that L B (w−u, v) is an eigenspace with eigenvalue u. Then by dimensionality consideration they are all of the eigenspaces of T .Let L := L B (w − u, v).

Then it's not hard to see T {L} = L B (a, b) for some a, b ∈ R. It follows that a has to be u(w − u) and b has to be −(v(w − u) − wv) = uv, which yields what we want.

DISPLAYFORM11 has the following eigendecomposition:• RG has eigenvalue DISPLAYFORM12 Proof.

The case of M B is obvious.

The case for RG follows from Lemma E.70.

The case for L B follows from Lemma E.67.

By dimensionality considerations these are all of the eigenspaces.

In this section we will be interested in studying the dynamics on PSD matrices of the form DISPLAYFORM0 where Σ l ∈ S B and φ : R → R.

Basic questions regarding the dynamics Eq. (43) are 1) does it converge?

2) What are the limit points?

3) How fast does it converge?

Here we answer these questions definitively when φ = id.

The following is the key lemma in our analysis.

Lemma F.1.

Consider the dynamics DISPLAYFORM0 This convergence is exponential in the sense that, for any full rank Σ 0 , there is a constant DISPLAYFORM1 Here λ 1 (resp.

λ A ) denotes that largest (resp.

smallest) eigenvalue.

DISPLAYFORM2 It's easy to see that i λ i l = 1 for all l ≥ 1.

So WLOG we assume l ≥ 1 and this equality holds.

We will show the "exponential convergence" statement; that lim l→∞ Σ l = 1 A I then follows from the trace condition above.

Proof of Item 2.

For brevity, we write suppress the superscript index l, and use · to denote · l .

We will now compute the eigenvalues λ 1 ≥ · · · ≥ λ A of Σ.First, notice that Σ and Σ can be simultaneously diagonalized, and by induction all of {Σ l } l≥0 can be simultaneous diagonalized.

Thus we will WLOG assume that Σ is diagonal, Σ = Diag(λ 1 , . . .

, λ A ), so that Σ = Diag(γ 1 , . . .

, γ A ) for some {γ i } i .

These {γ i } i form the eigenvalues of Σ but a priori we don't know whether they fall in decreasing order; in fact we will soon see that they do, and DISPLAYFORM3 We have DISPLAYFORM4 Therefore, DISPLAYFORM5 Since the RHS integral is always positive, λ i ≥ λ k =⇒ γ i ≥ γ k and thus γ i = λ i for each i. DISPLAYFORM6

is (strictly) log-convex and hence (strictly) convex in λ.

Furthermore, T (λ) is (strictly) convex because it is an integral of (strictly) convex functions.

Thus T is maximized over any convex region by its extremal points, and only by its extremal points because of strict convexity.

The convex region we are interested in is given by DISPLAYFORM0 The unit sum condition follows from the normalization h/ h of the iteration map.

The extremal points of A are {ω k := ( DISPLAYFORM1 This shows that T (λ) ≤ 1, with equality iff λ = ω k for k = 1, . . .

, A − 1.

In fact, because every point λ ∈ A is a convex combination of ω DISPLAYFORM2 , by convexity of T , we must have DISPLAYFORM3 where the last line follows because ω A is the only point with last coordinate nonzero so that a A = λ A .We now show that the gap λ 1 l − λ A l → 0 as l → ∞. There are two cases: if λ A l is bounded away from 0 infinitely often, then T (λ) < 1 − infinitely often for a fixed > 0 so that the gap indeed vanishes with l. Now suppose otherwise, that λ A l converges to 0; we will show this leads to a contradiction.

Notice that DISPLAYFORM4 where the first lines is Eq. (44) and the 2nd line follows from the convexity of DISPLAYFORM5 as a function of (λ 1 , . . .

, λ A−1 ).

By a simple application of dominated convergence, as λ A → 0, this integral converges to a particular simple form, DISPLAYFORM6 (1 + 2s/(A − 1)) −(A−1)/2 ds = 1 + 2 A − 3 Thus for large enough l, λ A l+1 /λ A l is at least 1 + for some > 0, but this contradicts the convergence of λ A to 0. .

In addition, asymptotically, the gap decreases exponentially as DISPLAYFORM7 l , proving Item 3.

2.

This convergence is exponential in the sense that, for any Σ 0 of rank C, there is a constant

Here λ i denotes the ith largest eigenvalue.

DISPLAYFORM0 Proof.

Note that Σ l can always be simultaneously diagonalized with Σ 0 , so that DISPLAYFORM1 by deleting the dimensions with zero eigenvalues.

The proof then finishes by Lemma F.1.From this it easily follows the following characterization of the convergence behavior.

Corollary F.3.

Consider the dynamics of Eq. (43) for φ = id: DISPLAYFORM2 Suppose GΣG has rank C < B and factors asêDê T where D ∈ R C×C is a diagonal matrix with no zero diagonal entries andê is an B × C matrix whose columns form an orthonormal basis of a subspace of im G. Then DISPLAYFORM3 2.

This convergence is exponential in the sense that, for any GΣ 0 G of rank C, there is a constant K < 1 such that DISPLAYFORM4 Here λ i denotes the ith largest eigenvalue.

DISPLAYFORM5 General nonlinearity.

We don't (currently) have a proof of any characterization of the basin of attraction for Eq. FORMULA2 for general φ.

Thus we are forced to resort to finding its fixed points manually and characterize their local convergence properties.

Batchnorm B φ is permutation-equivariant, in the sense that B φ (πh) = πB φ (h) for any permtuation matrix π.

Along with the case of φ = id studied above, this suggests that we look into BSB1 fixed points Σ * .

What are the BSB1 fixed points of Eq. (43)?

The main result (Thm F.5) of this section is an expression of the BSB1 fixed point diagonal and offdiagonal entries in terms of 1-and 2-dimensional integrals.

This allows one to numerically compute such fixed points.

By a simple symmetry argument, we have the following Lemma F.4.

Suppose X is a random vector in R B , symmetric in the sense that for any permutation matrix π and any subset U of R B measurable with respect to the distribution of X, P (X ∈ U ) = P (X ∈ π(U )).

Let Φ : R B → R B be a symmetric function in the sense that Φ(πx) = πΦ(x) for any permutation matrix π.

Then E Φ(X)Φ(X) T = µI + ν11 T for some µ and ν.

This lemma implies that V B φ (Σ) is BSB1 whenever Σ is BSB1.

In the following, We in fact show that for any BSB1 Σ, V B φ (Σ) is the same, and this is the unique BSB1 fixed point of Eq. (43).

DISPLAYFORM0 where DISPLAYFORM1 Proof.

For any BSB1 Σ, e T Σe = µI B−1 for some µ. Thus we can apply Proposition E.33 to GΣG.

We observe that K(v; GΣG) = 1, so that DISPLAYFORM2 Note that this is now independent of the specific values of Σ. Now, using the specific form of e given by Eq. FORMULA2 , we see that V B φ (Σ) 11 only depends on θ 1 and V B φ (Σ) 12 only depends on θ 1 and θ 2 .

Then applying Proposition E.33 followed by Lemma E.36 and Lemma E.37 yield the desired result.

In the case that φ is positive-homogeneous, we can apply Laplace's method Appendix E.1 and obtain the BSB1 fixed point in a much nicer form.

Note that Proposition F.7.

lim B→∞ K α,B = c α .We give a closed form description of the BSB1 fixed point for a positive homogeneous φ in terms of its J function.

Theorem F.8.

Suppose φ : R → R is degree α positive-homogeneous.

For any BSB1 Σ ∈ S B , V B φ (Σ) is BSB1.

The diagonal entries are K α,B J φ (1) and the off-diagonal entries are DISPLAYFORM0 Here c α is as defined in Defn E.7 and J φ is as defined in Defn E.20.

Thus a BSB1 fixed point of Eq. (43) exists and is unique.

Proof.

Let e be an B × (B − 1) matrix whose columns form an orthonormal basis of im G := {Gv : DISPLAYFORM1 denote a matrix with a on the diagonal and b on the off-diagnals.

Note that V φ is positive homogeneous of degree α, so for any µ, DISPLAYFORM2 by Proposition E.24 DISPLAYFORM3 So by Eq. FORMULA1 , DISPLAYFORM4 Corollary F.9.

Suppose φ : R → R is degree α positive-homogeneous.

If Σ * is the BSB1 fixed point of Eq. (43) as given by Thm F.8, then GΣ * G = µ * I B−1 where µ DISPLAYFORM5 By setting α = 1 and φ = relu, we get Corollary F.10.

For any BSB1 Σ ∈ S B , V B relu (Σ) is BSB1 with diagonal entries 1 2 and off-diagonal entries DISPLAYFORM6 By setting α = 1 and φ = id = x → relu(x) − relu(−x), we get Corollary F.11.

For any BSB1 Σ ∈ S B , V B id (Σ) is BSB1 with diagonal entries 1 and off-diagonal entries DISPLAYFORM7 Remark F.12.

One might hope to tweak the Laplace method for computing the fixed point to work for the Fourier method, but because there is no nice relation between V φ (cΣ) and V φ (Σ) in general, we cannot simplify Eq. (32) as we can Eq. FORMULA1 and Eq. (27).

It turns out that the BSB1 fixed point can be described very cleanly using the Gegenbauer coefficients of φ.

DISPLAYFORM0 , so that by Proposition E.34, DISPLAYFORM1 , independent of the actual values of Σ. Here e a,: is the ath row of e. DISPLAYFORM2 2 ) has Gegenbauer expansion DISPLAYFORM3 Proof.

By the reproducing property Fact E.46, we have We will see later that the rate of gradient explosion also decomposes nicely in terms of Gegenbauer coefficients (Thm G.5).

However, this is not quite true for the eigenvalues of the forward convergence (Thms F.22 and F.24).

DISPLAYFORM4

In this section we consider linearization of the dynamics given in Eq. (43).

Thus we must consider linear operators on the space of PSD linear operators S B .

To avoid confusion, we use the following notation: If T : S B → S B (for example the Jacobian of V B φ ) and Σ ∈ S B , then write T {Σ} for the image of Σ under T .A priori, the Jacobian of V B φ at its BSB1 fixed point may seem like a very daunting object.

But a moment of thought shows that Proposition F.14.

The Jacobian DISPLAYFORM0 Now we prepare several helper reults in order to make progress understanding batchnorm.

Definition F.15.

Define n(x) = √ Bx/ x , i.e. division by sample standard deviation.

Batchnorm B φ can be decomposed as the composition of three steps, φ • n • G, where G is meancentering, n is division by standard deviation, and φ is coordinate-wise application of nonlinearity.

We have, as operators H → H, DISPLAYFORM1 Definition F.16.

With Σ * being the unique BSB1 fixed point, write DISPLAYFORM2 Published as a conference paper at ICLR 2019It turns out to be advantageous to study U first and relate its eigendecomposition back to that of DISPLAYFORM3 DISPLAYFORM4 which are all eigenvectors of BA with nonzero eigenvalues, up to linear combinations within eigenvectors with the same eigenvalue.

DISPLAYFORM5 , Thm E.61 implies that AB and BA can both be diagonalized, and this lemma implies that all nonzero eigenvalues of DISPLAYFORM6 can be recovered from those of U.Proof. (Item 1) Observe rank AB = rank ABAB ≤ rank BA.

By symmetry the two sides are in fact equal.(Item 2) Bv cannot be zero or otherwise ABv = A0 = 0, contradicting the fact that v is an eigenvector with nonzero eigenvalue.

Suppose λ is the eigenvalue associated to v. Then BA(Bv) = B(ABv) = B(λv) = λBv, so Bv is an eigenvector of BA with the same eigenvalue. .

The eigenspaces with different eigenvalues are linearly independent, so it suffices to show that if {Bv ij } j are eigenvectors of the same eigenvalue λ s , then they are linearly independent.

But j a j Bv ij = 0 =⇒ j a j v ij = 0 because B is injective on eigenvectors by Item 2, so that a j = 0 identically.

Hence {Bv j } j is linearly independent.

Since rank BA = k, these are all of the eigenvectors with nonzero eigenvalues of BA up to linearly combinations.

Lemma F.18.

Let f : R B → R A be measurable, and Σ ∈ S B be invertible.

Then for any Λ ∈ R B×B , with ·, · denoting trace inner product, DISPLAYFORM7 If f is in addition twice-differentiable, then DISPLAYFORM8 whenever both sides exist.

Proof.

Let Σ t , t ∈ (− , ) be a smooth path in S B , with Σ 0 = Σ. Write DISPLAYFORM9 (by Lemma F.18) DISPLAYFORM10 Let's extend to all matrices by this formula: Definition F.19.

Definẽ DISPLAYFORM11 Ultimately we will apply Thm E.61 to DISPLAYFORM12 Here we will realize e as the matrix in Eq. (33).

DISPLAYFORM13 If we can evaluate W ij|kl then we can use Thm E.61 to compute the eigenvalues of G ⊗2 • U.

It's easy to see that W ij|kl is ultrasymmetric Thus WLOG we can take i, j, k, l from {1, 2, 3, 4}.By Lemma E.35, and the fact that x → ex is an isometry, DISPLAYFORM14 If WLOG we further assume that k, l ∈ {1, 2} (by ultrasymmetry), then there is no dependence on θ 3 and θ 4 inside φ.

So we can expand (ev) i and (ev) j in trigonometric expressions as in Eq. FORMULA2 and integrate out θ 3 and θ 4 via Lemma E.29.

We will not write out this integral explicitly but instead focus on other techniques for evaluating the eigenvalues.

Now let's compute the local convergence rate via Gegenbauer expansion.

By differentiating Proposition E.34 through a path Σ t ∈ S B , t ∈ (− , ), we get DISPLAYFORM0 where Σ = Σ t = e T Σ t e ∈ S B−1 (as introduced below Defn E.3).

At Σ 0 = Σ * , the BSB1 fixed point, we have Σ 0 = µ * I, and DISPLAYFORM1 If d dt Σ| t=0 = G, then the term in the parenthesis vanishes.

This shows that DISPLAYFORM2 (Proposition E.59), then we know by Thm E.62 that DISPLAYFORM3 .

Because e is an isometry, DISPLAYFORM4 By symmetry, A(a, b; a) = A(a, b; b) and for any c, c ∈ {a, b}, A(a, b; c) = A(a, b; c ).

So we have, for any a, b, c not equal, DISPLAYFORM5 So the eigenvalue associated to L(B − 2, 1) is, So by Lemma E.60, DISPLAYFORM6 DISPLAYFORM7 2 ) has Gegenbauer expansion DISPLAYFORM8 where κ i (l, B − 3 2 ) is understood to be 0 for l < 0 DISPLAYFORM9 We can evaluate, for any a, b (possibly equal), DISPLAYFORM10 By Eq. FORMULA5 , the eigenvalue associated with L(B − 2, 1) is DISPLAYFORM11 where DISPLAYFORM12 Thus,

Published as a conference paper at ICLR 2019 DISPLAYFORM0 2 ) l (x).

Then the eigenvalue of U with respect to the eigenspace L is a ratio of quadratic forms DISPLAYFORM1 Note that v B−1,0 = w B−1,0 = u B−1,0 = 0, so that there is in fact no dependence on a 0 , as expected since batchnorm is invariant under additive shifts of φ.

We see that λ DISPLAYFORM2 This is a quadratic form on the coefficients {a l } l .

We now analyze it heuristically and argue that the eigenvalue is ≥ 1 typically when φ( √ B − 1x) explodes sufficiently as x → 1 or x → −1; in other words, the more explosive φ is, the less likely it is to induce Eq. (43) to converge to a BSB1 fixed point.

DISPLAYFORM3 (1) for sufficiently large B and l, so that DISPLAYFORM4 (1) and is positive.

For small l, we can calculate DISPLAYFORM5 In fact, plotting w B−1,l −v B−1,l for various values of l suggests that for B ≥ 10, w B−1,l −v B−1,l ≥ 0 for all l ≥ 1 FIG4 .Thus, the more "linear" φ is, the larger a 2 1 is compared to the rest of {a l } l , and the more likely that the eigenvalue is < 1.

In the case that a l a l+2 = 0 ∀l, then indeed the eigenvalue is < 1 precisely when a 2 1 is sufficiently large.

Because higher degree Gegenbauer polynomials explodes more violently as x →

±1, this is consistent with our claim.

DISPLAYFORM6 for any permutation π, and DISPLAYFORM7 By Eq. FORMULA5 , we havẽ DISPLAYFORM8 By Thm E.62, G ⊗2 • U has eigenspace M with eigenvalue DISPLAYFORM9 12) −Ã 12 (13) +Ã 12 (34)).

Thus we need to evaluateÃ 12 (12),Ã 12 (13),Ã 12 (34).

We will do so by exploiting linear dependences between different values ofÃ ab (c, d), the computed values ofÃ ab (c, c) = A(a, b; c), and the value ofÃ ab (a, b) computed below in Lemma F.23.

Indeed, we have DISPLAYFORM10 Leveraging the symmmetries ofÃ, we get (B − 3)Ã 12 (34) + 2Ã 12 (13) +Ã 12 (33) = 0 (B − 2)Ã 12 (13) +Ã 12 (11) +Ã 12 (12) = 0.

Expressing in terms ofÃ 12 (11) andÃ 12 (12), we get DISPLAYFORM11 These relations allow us to simplify DISPLAYFORM12 By Eq. FORMULA5 , we know the Gegenbauer expansion ofÃ 12 (11).

The following shows the Gegenbauer expansion ofÃ 12 (12).

DISPLAYFORM13 2 ), and φ( DISPLAYFORM14 where DISPLAYFORM15 Proof.

Let ψ(x) = xφ( √ B − 1x).

Then by Proposition E.43, we have DISPLAYFORM16 Rearranging the sum in terms of a l gives the result.

Combining all of the above, we obtain the following DISPLAYFORM17 2 ) l (x).

Then the eigenvalue of U with respect to the eigenspace M is a ratio of quadratic forms DISPLAYFORM18 2 ) l+1 DISPLAYFORM19 which is much smaller than DISPLAYFORM20 (1) for degree larger than 0.

Thus the eigenvalue for M is typically much smaller than the eigenvalue for L (though there are counterexamples, like sin).

We begin by studying DISPLAYFORM21 .

We will differentiate Eq. (26) directly at G ⊗2 {Σ * } = GΣ * G where Σ * is the BSB1 fixed point given by Thm F.8.

To that end, consider a smooth path Σ t ∈ S G B , t ∈ (− , ) for some > 0, with Σ 0 = GΣ * G. Set Σ t = e T Σ t e ∈ S B−1 , so that Σ t = eΣ t e T and Σ 0 = µ * I B−1 where µ DISPLAYFORM22 B−1 )) as in Thm F.8.

If we writeΣ andΣ for the time derivatives, we have DISPLAYFORM23 −1 e T (apply Lemma F.26 and Lemma F.27) DISPLAYFORM24 2 + 2 > α (precondition for Lemma F.28).

With some trivial simplifications, we obtain the following Lemma F.25.

Let φ be positive-homogeneous of degree α.

Consider a smooth path −1 dΣ /dt).

DISPLAYFORM25 DISPLAYFORM26 Proof.

Straightforward computation.

Lemma F.27.

For any s ∈ R, DISPLAYFORM27 Proof.

Straightforward computation.

Lemma F.28.

For a > b + 1, DISPLAYFORM28 Proof.

Apply change of variables x = 2µs 1+2µs .This immediately gives the following consequence.

Theorem F.29.

Let φ : R → R be any function with finite first and second Gaussian moments.

Then DISPLAYFORM29 has the following eigendecomposition:• RG has eigenvalue DISPLAYFORM30 • L B has eigenvalue DISPLAYFORM31 .Theorem F.30.

Let φ : R → R be positive-homogeneous of degree α.

Then for any p = 0, c ∈ R, DISPLAYFORM32 has the following eigendecomposition: DISPLAYFORM33 • L B has eigenvalue c α p DISPLAYFORM34 • RG has eigenvalue c α p DISPLAYFORM35 Proof.

By Proposition E.22, DISPLAYFORM36 is DOS(u, v, w) with DISPLAYFORM37 With Thm E.73, we can do the computation: DISPLAYFORM38 • RG has eigenvalue DISPLAYFORM39 We record the following consequence which will be used frequently in the sequel. : H B → H B has the following eigendecomposition: DISPLAYFORM40 Proof.

Use Thm E.72 with the computations from the proof of Thm F.30 as well as the following computation DISPLAYFORM41 Theorem F.33.

Let φ be positive-homogeneous with degree α.

Assume DISPLAYFORM42 has 3 distinct eigenvalues.

They are as follows: DISPLAYFORM43 Proof.

Item 1.

The case of λ DISPLAYFORM44 Item 2.

The case of λ ↑ M (B, α).

AssumeΣ 0 ∈ M, Thm F.31 gives (with denoting Hadamard product, i.e. entrywise multiplication) DISPLAYFORM45 Since tr(Σ 0 ) = 0, Eq. FORMULA7 gives DISPLAYFORM46 Since tr(Σ 0 ) = 0, Eq. FORMULA7 gives DISPLAYFORM47 With some routine computation, we obtain Proposition F.34.

With φ and α as above, as B → ∞, DISPLAYFORM48 We can compute, by Proposition E.23, DISPLAYFORM49 where the last part is obtained by optimizing over a and b. On α ∈ (−1/2, ∞), this is greater than α iff α < 1.

Thus for α ≥ 1, the maximum eigenvalue of U is always achieved by eigenspace L for large enough B.

Let's extend the definition of the V operator: DISPLAYFORM0 , which acts on n × n matrices Π by DISPLAYFORM1 Under this definition, V B φ (Σ) is a linear operator R B×B → R

.

Recall the notion of adjoint: DISPLAYFORM0 , its dual space, is defined as the space of linear functionals f : DISPLAYFORM1 If a linear operator is represented by a matrix, with function application represented by matrixvector multiplication (matrix on the left, vector on the right), then the adjoint is represented by matrix transpose.

The backward equation.

In this section we are interested in the backward dynamics, given by the following equation DISPLAYFORM2 where Σ l is given by the forward dynamics.

Particularly, we are interested in the specific case when we have exponential convergence of Σ l to a BSB1 fixed point.

Thus we will study the asymptotic approximation of the above, namely the linear system.

DISPLAYFORM3 where Σ * is the BSB1 fixed point.

Note that after one step of backprop, DISPLAYFORM4 Thus the large L dynamics of Eq. FORMULA0 is given by the eigendecomposition of DISPLAYFORM5 It turns out to be much more convenient to study its adjoint DISPLAYFORM6 , which has the same eigenvalues (in fact, it will turn out that it is self-adjoint).

DISPLAYFORM7 .

As discussed above, we will seek the eigendecomposition of DISPLAYFORM8 We first make some basic calculations.

DISPLAYFORM9 where r = y , v = y/ y = n(y)/ √ B.Proof.

We have DISPLAYFORM10 , and DISPLAYFORM11 By chain rule, this easily gives DISPLAYFORM12 where D = Diag(φ (n(y))), r = y , v = y/ y = n(y)/ √ B.With v = y/ y , r = y , andD = Diag(φ (n(y))), we then have DISPLAYFORM13 DISPLAYFORM14 is the BSB1 fixed point, one can again easily see that F(Σ * ) is ultrasymmetric.

With DISPLAYFORM15 by Lemma E.35, where we assume, WLOG by ultrasymmetry, k, l ∈ {1, 2}; i, j ∈ {1, . . .

, 4}, and DISPLAYFORM16 and e as in Eq. (33).We can integrate out θ 3 and θ 4 symbolically by Lemma E.29 since their dependence only appear outside of φ .

This reduces each entry of F(Σ * ){τ ij } kl to 2-dimensional integrals to evaluate numerically.

The eigenvalues of G ⊗2 • F(Σ * ) can then be obtained from Thm E.62.

We omit details here and instead focus on results from other methods.

Notice that when Λ = G, the expression for F(Σ * ){Λ} significantly simplifies because G acts as identity on v: DISPLAYFORM0 Then the diagonal of the image satisfies DISPLAYFORM1 (by Proposition E.33) DISPLAYFORM2 And the off-diagonal entries satisfy DISPLAYFORM3 (by Proposition E.33) DISPLAYFORM4 Lifting Spherical Expectation.

We first show a way of expressing F(Σ * ) aa in terms of Gegenbauer basis by lifting the spherical integral over S B−2 to spherical integral over S B .

While this technique cannot be extended to F(Σ * ) ab , a = b, we believe it could be of use to future related problems.

DISPLAYFORM5 ea,:(v).

Hereê a,: and v are treated as points on the B-dimensional sphere with last 2 coordinates 0.Thus by Lemmas E.38 and E.39, DISPLAYFORM6 êa,: DISPLAYFORM7 ( FORMULA0 by Lemma E.45 DISPLAYFORM8 Dirichlet-like form of φ.

For F(Σ * ){G} ab , there's no obvious way of lifting the spherical expectation to a higher dimension with the weight (1 + (B − 1)(ê a,: v)(ê b,: v)).

We will instead leverage Thm E.47.

In Thm E.47, setting u 1 =ê a,: = u 2 , we get DISPLAYFORM9 Likewise, setting u 1 =ê a,: , u 2 =ê b,: , we get DISPLAYFORM10 2 ) l (x).

Then for Σ * being the unique BSB1 fixed point of Eq. (43), the eigenvalue of DISPLAYFORM11 This is minimized (over choices of φ) iff φ is linear, in which case λ DISPLAYFORM12

In this section suppose that φ is degree α positive-homogeneous.

Set D = Diag(φ (y)) (and recall v = y/ y , r = y ).

Then D is degree α − 1 positive-homogeneous in x (because φ is).

Consequently we can rewrite Eq. (52) as follows, DISPLAYFORM0 where Σ * is the BSB1 fixed point of Eq. (43), GΣ * G = µ * G, and DISPLAYFORM1 Each term in the sum above is ultrasymmetric, so has the same eigenspaces RG, M, L (this will also become apparent in the computations below without resorting to Thm E.61).

We can thus compute the eigenvalues for each of them in order.

In all computation below, we apply Lemma E.2 first to relate the quantity in question to V φ .Computing A. For two matrices Σ, Λ, write Σ Λ for entrywise multiplication.

We have by Lemma E.2 DISPLAYFORM2 Then Thm E.73 gives the eigendecomposition for DISPLAYFORM3 has the following eigendecomposition.

DISPLAYFORM4 .

This allows us to apply Thm E.73.

We make a further simplification J φ (c) = (2α−1)J φ (c) via Proposition E.22.Computing B. We simplify DISPLAYFORM5 where ψ(y) := yφ (y), which, in this case of φ being degree α positive-homogeneous, is also equal to αφ(y).By Lemma E.2, B{Λ} equals DISPLAYFORM6 This expression naturally leads us to consider the following definition.

Definition G.7.

Let ψ : R → R be measurable and Σ ∈ S B .

Define V (4) DISPLAYFORM7 For two matrices Σ, Λ, write Σ, Λ := tr(Σ T Λ).

We have the following identity.

DISPLAYFORM8 Proof.

We have DISPLAYFORM9 Making the substitution Λ → ΣΛΣ, we get the desired result.ϕ is degree α + 1 positive-homogeneous.

Thus, DISPLAYFORM10 DISPLAYFORM11 has the following eigendecomposition (note that here µ * is still with respect to φ, not ψ = αφ)1.

Eigenspace RG with eigenvalue DISPLAYFORM12 3.

Eigenspace M with eigenvalue DISPLAYFORM13 Proof.

The only thing to justify is the value of λ B G .

We have DISPLAYFORM14 Computing C. By Lemma E.2, DISPLAYFORM15 Lemma G.10.

Suppose φ is degree α positive-homogeneous.

Then for Λ ∈ H B , DISPLAYFORM16 where D = Diag(φ (y)).Proof.

Let Π t , t ∈ (− , ) be a smooth path in H B with Π 0 = I. DISPLAYFORM17 Because for x ∈ R, αφ(x) = xφ (x), for y ∈ R B we can write φ(y) = α −1 Diag(φ (y))y.

Then DISPLAYFORM18 • C has the following eigendecomposition 1.

eigenspace RG with eigenvalue DISPLAYFORM19 3.

eigenspace M with eigenvalue DISPLAYFORM20 Altogether, by Eq. FORMULA2 and Thms G.6, G.9 and G.11, this implies DISPLAYFORM21 has eigenspaces RG, M, L respectively with the following eigenvalues DISPLAYFORM22

In this section, we study the generalization of Eq. (43) to multiple batches.

Definition H.1.

For linear operators DISPLAYFORM0 We also write T ⊕n 1 := n j=1 T 1 for the direct sum of n copies of T 1 .For k ≥ 2, now consider the extended ("k-batch") dynamics on Σ ∈ S kB defined by DISPLAYFORM1

In general, like in the case of Eq. (43), it is difficult to prove global convergence behavior.

Thus we manually look for fixed points and the local convergence behaviors around them.

A very natural extension of the notion of BSB1 matrices is the following Definition H.2.

We say a matrix Σ ∈ S kB is CBSB1 (short for "1-Step Cross-Batch Symmetry Breaking") if Σ in block form (k × k blocks, each of size B × B) has one common BSB1 block on the diagonal and one common constant block on the off-diagonal, i.e.  DISPLAYFORM0 We will study the fixed point Σ * DISPLAYFORM1 Proof.

We will prove for k = 2.

The general k cases follow from the same reasoning.

DISPLAYFORM2 where Σ = BSB1(a, b).

As remarked below Eq. (54), restricting to any diagonal blocks just recovers the dynamics of Eq. FORMULA2 , which gives the claim about the diagonal blocks being Σ * through Thm F.5.We now look at the off-diagonal blocks.

DISPLAYFORM3 where DISPLAYFORM4 (the last step follows from Lemma E.52).

Thus y 1:B is independent from y B+1:2B , and DISPLAYFORM5 By symmetry, DISPLAYFORM6 where e is as in Eq. (33) DISPLAYFORM7 because e is an isometry DISPLAYFORM8 by Lemma E.37.

At the same time, by Proposition E.33, we can obtain the Gegenbauer expansion DISPLAYFORM9 2 ) l DISPLAYFORM10 Proof.

We compute DISPLAYFORM11 So for a positive homogeneous function DISPLAYFORM12 Expanding the beta function and combining with Thm H.3 gives the desired result.

While we don't need to use the Laplace method to compute c * for positive homogeneous functions (since it is already given by Corollary H.4), going through the computation is instructive for the machinery for computing the eigenvalues in a later section.

.

For A, B, C ∈ N, let f : DISPLAYFORM0 Then on {Σ ∈ S A+B : rank Σ > 2(a + b)}, ℘(Σ) is well-defined and continuous, and furthermore satisfies DISPLAYFORM1 .

Published as a conference paper at ICLR 2019Proof.

If Σ is full rank, we can show Fubini-Tonelli theorem is valid in the following computation by the same arguments of the proof of Lemma E.2.

DISPLAYFORM0 where in the last line, (y, z) ∼ N (0, (Σ −1 + 2D 2 ) −1 ).

We recover the equation in question with the following simplifications.

DISPLAYFORM1 The case of general Σ with rank Σ > 2(a + b) is given by the same continuity arguments as in Lemma E.2.

DISPLAYFORM2 where Ω = sΣ DISPLAYFORM3 Theorem H.6 (Rephrasing and adding to Corollary H.4).

Let Σ ∈ S kB and φ be positive homoge- DISPLAYFORM4 where Σ * is the BSB1 fixed point of Thm F.5 and DISPLAYFORM5 Proof.

Like in the proof of Corollary H.4, we only need to compute the cross-batch block, which is given above by Eq. ( .

By Lemma F.17, its nonzero eigenvalues are exactly those DISPLAYFORM6 .

Here DISPLAYFORM7 Thus it suffices to obtain the eigendecomposition ofF.First, notice thatF acts on the diagonal blocks as G ⊗2 • dV φ•n dΣ G , and the off-diagonal components of Λ has no effect on the diagonal blocks ofF{Λ}. We formalize this notion as follows Definition H.7.

Let T : H kB → H kB be a linear operator such that for any block matrix Σ ∈ H kB with k × k blocks Σ ij , i, j ∈ [k], of size B, we have Next, we step back a bit to give a full treatment of the eigen-properties of BDOS operators.

Then we specialize back to our case and give the result forF. (1 + 2sµ * ) (α+1)/2 (1 + 2tµ DISPLAYFORM8 DISPLAYFORM9 Thus the product (I + 2Ω 0 )

0 has zero diagonal blocks so that its trace is 0.

Therefore, only the second term in the sum in Eq. FORMULA7 DISPLAYFORM0 As in the single batch case, we approximate it by taking Σ l to its CBSB1 limit Σ * , so that we analyzẽ This corollary indicates that stacking batchnorm in a deep neural network will always cause chaotic behavior, in the sense that cross batch correlation, both forward and backward, decreases to 0 exponentially with depth.

The φ that can maximally ameliorate the exponential loss of information is linear.

DISPLAYFORM1

In the above exposition of the mean field theory of batchnorm, we have assumed B ≥ 4 and that φ induces BSB1 fixed points in Eq. (8).Small Batch Size What happens when B < 4?

It is clear that for B = 1, batchnorm is not well-defined.

For B = 2, B φ (h) = (±1, ∓1) depending on the signs of h. Thus, the gradient of a batchnorm network with B = 2 is 0.

Therefore, we see an abrupt phase transition from the immediate gradient vanishing of B = 2 to the gradient explosion of B ≥ 4.

We empirically see that B = 3 suffers similar gradient explosion as B = 4 and conjecture that a generalization of Thm 3.9 holds for B = 3.Batch Symmetry Breaking What about other nonlinearities?

Empirically, we observe that if the fixed point is not BSB1, then it is BSB2, like in FIG4 , where a submatrix of Σ (the dominant block) is much larger in magnitude than everything else (see Defn K.1).

If the initial Σ 0 is permutationinvariant, then convergence to this fixed point requires spontaneous symmetry breaking, as the dominant block can appear in any part of Σ along the diagonal.

This symmetry breaking is lost when we take the mean field limit, but in real networks, the symmetry is broken by the network weight randomness.

Because small fluctuation in the input can also direct the dynamics toward one BSB1 fixed point against others, causing large change in the output, the gradient is intuitively large as well.

Additionally, at the BSB2 fixed point, we expect the dominant block goes through a similar dynamics as if it were a BSB1 fixed point for a smaller B, thus suffering from similar gradient explosion.

Appendix K discusses several results on our current understanding of BSB2 fixed points.

A specific form of BSB2 fixed point with a 1 × 1 dominant block can be analyzed much further, and this is done in Appendix K.1.Finite width effect For certain nonlinearities, the favored fixed point can be different between the large width limit and small width.

For example, for φ = C Proof.

We can verify all eigenspaces and their eigenvalues in a straightforward manner.

These then must be all of them by a dimensionality argument.

Specializing to BSB2 matrices, we get • 1-dimensional Rq with eigenvalue For fixed B < B, specializing the forward dynamics Eq. (43) to BSB2 B B fixed points yields a 2-dimensional dynamics on the eigenvalues for the eigenspaces Z 1 and Rq.

This dynamics in general is not degenerate, so that the fixed point seems difficult to obtain analytically.

Moreoever, the Gegenbauer method, essential for proving that gradient explosion is unavoidable when Eq. (43) has a BSB1 fixed point, does not immediately generalize to the BSB2 case, since Z 1 and Rq in general do not have the same eigenvalues so that we cannot reduce the integrals to that on a sphere.

For this reason, at present we do not have any rigorous result on the BSB2 case.

However, we expect that the main block of the BSB2 fixed point should undergo a similar dynamics to that of a BSB1 fixed point, leading to similar gradient explosion.

is the all 1s column vector.

In these situations we can in fact see pathological gradient vanishing, in a way reminiscent of the gradient vanishing for B = 2 batchnorm, as we shall see shortly.

We can extend Corollary K.3 to the B = 1 case.

DISPLAYFORM0 , where for some λ by Thm K.4.

The rest then follows from straightforward computation and another application of Thm K.4.

DISPLAYFORM1

@highlight

Batch normalization causes exploding gradients in vanilla feedforward networks.

@highlight

Develops a mean field theory for batch normalization (BN) in fully-connected networks with randomly initialized weights.

@highlight

Provides a dynamic perspective on deep neural network using the evolution of the covariance matrix along with the layers.
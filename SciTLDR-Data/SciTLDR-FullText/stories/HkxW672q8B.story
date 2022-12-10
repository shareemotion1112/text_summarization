Rectified linear units, or ReLUs, have become a preferred activation function for artificial neural networks.

In this paper we consider the problem of learning a generative model in the presence of nonlinearity (modeled by the ReLU functions).

Given a set of signal vectors $\mathbf{y}^i \in \mathbb{R}^d, i =1, 2, \dots , n$, we  aim to learn the network parameters, i.e., the $d\times k$ matrix $A$, under the model $\mathbf{y}^i = \mathrm{ReLU}(A\mathbf{c}^i +\mathbf{b})$, where $\mathbf{b}\in \mathbb{R}^d$ is a random bias vector, and {$\mathbf{c}^i \in \mathbb{R}^k$ are arbitrary unknown latent vectors}.

We show that it is possible to recover the column space of $A$ within an error of $O(d)$ (in Frobenius norm) under certain conditions on the  distribution of $\mathbf{b}$.

Rectified Linear Unit (ReLU) is a basic nonlinear function defined to be ReLU : R → R + ∪ {0} as ReLU(x) ≡ max(0, x).

For any matrix X, ReLU(X) denotes the matrix obtained by applying the ReLU function on each of the coordinates of the matrix X. ReLUs are building blocks of many nonlinear data-fitting problems based on deep neural networks (see, e.g., [20] for a good exposition).

In particular, [7] showed that supervised training of very deep neural networks is much faster if the hidden layers are composed of ReLUs.

Let Y ⊂ R d be a collection of signal vectors that are of interest to us.

Depending on the application at hand, the signal vectors, i.e., the constituents of Y, may range from images, speech signals, network access patterns to user-item rating vectors and so on.

We assume that the signal vectors satisfy a generative model, where each signal vector can be approximated by a map g : R k → R d from the latent space to the ambient space, i.e., for each y ∈ Y, y ≈ g(c) for some c ∈ R k .

In this paper we consider the following specific model (single layer ReLU-network), with the weight (generator) matrix A ∈ R d×k and bias b ∈ R d :

The generative model in (2) raises multiple interesting questions that play fundamental role in understanding the underlying data and designing systems and algorithms for information processing.

Here, we consider the following network parameter learning problem under the specific generative model of (2) .

Learning the network parameters: Given the n observations {y i } i∈[n] ⊂ R d from the model (cf. (2)), recover the parameters of the model, i.e., A ∈ R d×k such that

with latent vectors {c i } i∈[n] ⊂ R k .

We assume that the bias vector b is a random vector comprising of i.i.d.

coordinates with each coordinate distributed according to the probability density function 33rd Conference on Neural Information Processing Systems (NeurIPS 2019), Vancouver, Canada.

p(·).

This question is closely related to the dictionary-learning problem [16] .

We also note that this question is different from the usual task of training a model (such as, [11] ), in which case the set {c i } i∈[n] is also known (and possibly chosen accordingly) in addition to {y i } i∈[n] .

Related works.

There have been a recent surge of interest in learning ReLUs, and the above question is of basic interest even for a single-layer network (i.e., nonlinearity comprising of a single ReLU function).

It is conceivable that understanding the behavior of a single-layer network would allow one to use some iterative peeling off technique to develop a theory for the generative models comprising of multiple layers.

To the best of our knowledge, the network parameter learning problem, even for single-layer networks has not been studied as such, i.e., theoretical guarantees do not exist.

Only in a very recent paper [22] the unsupervised problem was studied when the latent vectors {c i } i∈ [n] are random Gaussian.

The principled approaches to solve this unsupervised problem in practice reduce this to the 'training' problem, such as the autoencoders [10] that learn features by extensive end-to-end training of encoderdecoder pairs; or use the recently popular generative adversarial networks (GAN) [9] that utilize a discriminator network to tune the generative network.

The method that we are going to propose here can be seen as an alternative to using GANs for this purpose, and can be seen as an isolated 'decoder' learning of the autoencoder.

Note that the problem bears some similarity with matrix completion problems, a fact we greatly exploit.

In matrix completion, a matrix M is visible only partially, and the task is to recover the unknown entries by exploiting some prior knowledge about M .

In the case of (3), we are more likely to observe the positive entries of the matrix M , which, unlike a majority of matrix completion literature, creates the dependence between M and the sampling procedure.

.

With this notion, we can concisely represent the n observation vectors as

where M = AC and 1 ∈ R n denotes the all-ones vector and ⊗ denotes the Kronecker product.

Recall our assumption that b is a random vector comprising of i.i.d.

coordinates with each coordinate distributed according to the probability density function p(·).

Note that this model ensures that the bias corresponding to each coordinate is random, but does not change over different signal vectors.

We employ a natural approach to learn the underlying weight matrix A from the observation matrix Y .

As the network maps a lower dimensional latent vector c ∈ R k to obtain a signal vector y = ReLU(Ac + b) in dimension d > k, the matrix M = AC (cf.

(4)) is a low-rank matrix as long as k < min{d, n}. In our quest of recovering the weight matrix A, we first focus on estimating the matrix M , when given access to Y .

This task can be viewed as estimating a low-rank matrix from its partial (randomized) observations.

One of the main challenges that we face here is that while an entry of the matrix Y is a random variable (since b is a random bias), whether that is being observed or being cut-off by the ReLU function (for being negative) depends on the value of the entry itself.

In general matrix completion literature, the entries of the matrix being observed are sampled independent of the underlying matrix itself (see, e.g., [3, 12, 4] and references therein).

For this reason, we cannot use most of these results off-the-shelf.

However, similar predicament is (partially) present in [5] , where entries are quantized while being observed.

This motivates us to employ a maximum-likelihood method inspired by [5] .

That said, our observation model differs from [5] in a critical way: in our case the bias vector, while random, does not change over observations.

This translates to less freedom during the transformation of the original matrix to the observed matrix, leading to dependence among the elements in a row.

Furthermore, the analysis becomes notably different since the positive observations are not quantized.

Furthermore, although our formulation is close to [5] , because of the aforementioned differences in the observation models, we get much stronger guarantee on the recovery of the matrix.

Indeed, our results are comparable to analogous results of [21, 13] that also study the quantized matrix completion problem.

We show that our method guarantees the recovery of the matrix AC from Y with an error in Frobenius norm at most O( √ d) with high probability (see Theorem 1 for the formal statement).

Then leveraging the well known results on matrix perturbation [23] , it is possible to also recover the column space of A with a similar guarantee.

Extension to multi-layer networks.

Our aim is to use a 'peeling' technique to extend our results to multi-layer networks.

While rigorous theoretical guarantees for the network parameter learning problem do not extend to multi-layer networks as of now, we can still use the peel-off decoding for this case.

Note that, even for the 'training' problem of such network, which is a less demanding task than our problem, no rigorous guarantee exists beyond only two-layers.

For example, the state-of-the art theoretical results for training such as in [11] , hold only for one layer of nonlinearity (the second layer is assumed to have a linear activation function in [11] ).

In fact most recent theoretical guarantees in this domain are restricted to one or two layer networks e.g., [25, 15, 6, 8, 18, 24] .

Given this, obtaining provable guarantees for multi-layer case of network parameter learning problem seems to be a challenging future work.

We now focus on the task of estimating M from the observation matrix Y (cf.

(4)

where M i,j denotes the (i, j)-th entry of M , and

For i ∈ [d] and j ∈ [n], let M i,(j) denote the j-th largest element of the i-th row of M , i.e., for

It is straightforward to verify from (5)

.

Similarly, it follows from (6) that whenever we have (1) ).

Based on these observation, we define the set of matrices X Y,ν,γ ⊂ R d×n as

Recall that, p : R → R denotes the probability density function of each bias RV B. We use the notation F (x 1 , x 2 ) = P(−x 1 ≤ B ≤ −x 2 ) and X * i = max j∈[n]

X i,j .

Thus, the (normalized) log-likelihood of observing Y given that X is the original matrix takes the following form [17] .

In order to recover the matrix M from the observation matrix Y , we employ the following program.

Define ω p,γ,ν to be such that F (x, y) ≥ ω p,γ,ν for all x, y ∈ [−γ, γ] with |x − y| > ν.

In what follows, we simply refer this quantity as ω p given that γ and ν are clear from context.

Further define the following flatness and Lipschitz parameters associated with a function f : R → R:

The following result characterizes the performance of the program proposed in (9).

Theorem 1.

Assume that M ∞ ≤ γ and the observation matrix Y is related to M according to (4) .

Let M be the solution of the program specified in (9) , and the bias density function p(x) is differentiable with bounded derivative.

Then, the following holds with probability at least 1 −

where, C 0 is a constant.

The quantities β γ (p) and L γ (p) depend on the distribution of the bias and are defined in (10) and (11), respectively.

The full proof of Theorem 1 is omitted due to page limit.

In the first step of the proof, we show that given the observation matrix Y , for any X ∈ X Y,ν,γ (cf.

(7)), we have

This fact follows from the regularity assumptions on the probability density p and a sequence of inequalities similar to ones that relate KL divergence and the Hellinger divergence.

In the next step, we show that,

To upper bound the right-hand side we resort to standard techniques to bound the supremum of an empirical process such as the symmetrization and the contraction principle [14] .

The detail can be found in the full version [17] .

Recovering the network parameters.

Let us denote the recovered matrix M as M = M + E, where E is the perturbation matrix that has bounded Frobenius norm (cf. (12)).

Now the task of recovering the parameters of the single-layer ReLU-network is equivalent to solving for A given M = M + E = AC + E. Note that, even without the perturbation E we could only hope to recover the column space of A and not the exact matrix A.

Let U k and U k be the top k left singular vectors of M and M , respectively.

Let σ k , the smallest non-zero singular value of M , is at least δ > 0.

Then, it follows from standard results from matrix-perturbation theory (cf. [23] ) that there exists an orthogonal matrix O ∈ R k×k such that

which is a guarantee that the column space of A is recovered by SVD within an error of O(d) in Frobenius norm by the column space of U k .

Note that σ 1 is the largest singular value of M .

Future directions: Extension to multi-layer networks.

To learn a multi-layer ReLU network, we propose a 'peeling' decoder as defined below.

For better understanding, let us consider a two-layer model as follows:

where

Our 'peeling' decoder will approach this layer-by-layer.

First we use the likelihood-based matrix completion technique outlined in (9) to recover M 2 as in the case of a one-layer network.

Note that we will not be able to recover M 2 exactly, as Theorem 1 guarantees recovery only up to a Frobenius norm error.

This creates a hurdle to recover A 1 C, since our current method does not handle dense bounded norm noise while learning network parameters.

Also it will not be realistic to assume a probabilistic model for this bounded noise, since in the peeling off process the noise is coming from decoding of the previous layer.

Under certain coherence conditions on the matrix to be learned, it is possible to handle such situations (see e.g., [2, 19, 1] ).

Note that, such coherence condition must be satisfied by the parameters of every layer of the network.

An amenable but practical algorithm that is resilient to dense noise, under reasonable assumption on the network parameters, is an immediate area of interest.

Even when this first hurdle can be crossed, we still have a task of factorization of M 2 to find A 2 and ReLU A 1 C + b 1 ⊗ 1 T , and in general we cannot find a unique factorization.

Here as well, with additional reasonable assumptions the factorization can be made unique.

Note that this might be simpler that the factorization step of the one-layer network, since there is already a lot of structure in the latter matrix (such as nonnegativity due to being output of a ReLU and about ∼ 50% sparsity).

Provided this can be made to work analytically for two layers, there should not be any theoretical issue left to extend the process to multiple layers.

@highlight

We show that it is possible to recover the parameters of a 1-layer ReLU generative model from looking at samples generated by it
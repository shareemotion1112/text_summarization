Kronecker-factor Approximate Curvature (Martens & Grosse, 2015) (K-FAC) is a 2nd-order optimization method which has been shown to give state-of-the-art performance on large-scale neural network optimization tasks (Ba et al., 2017).

It is based on an approximation to the Fisher information matrix (FIM) that makes assumptions about the particular structure of the network and the way it is parameterized.

The original K-FAC method was applicable only to fully-connected networks, although it has been recently extended by Grosse & Martens (2016) to handle convolutional networks as well.

In this work we extend the method to handle RNNs by introducing a novel approximation to the FIM for RNNs.

This approximation works by modelling the covariance structure between the gradient contributions at different time-steps using a chain-structured linear Gaussian graphical model, summing the various cross-covariances, and computing the inverse in closed form.

We demonstrate in experiments that our method significantly outperforms general purpose state-of-the-art optimizers like SGD with momentum and Adam on several challenging RNN training tasks.

As neural networks have become ubiquitous in both research and applications the need to efficiently train has never been greater.

The main workhorses for neural net optimization are stochastic gradient descent (SGD) with momentum and various 2nd-order optimizers that use diagonal curvature-matrix approximations, such as RMSprop BID32 and Adam BID1 .

While the latter are typically easier to tune and work better out of the box, they unfortunately only offer marginal performance improvements over well-tuned SGD on most problems.

Because modern neural networks have many millions of parameters it is computationally too expensive to compute and invert an entire curvature matrix and so approximations are required.

While early work on non-diagonal curvature matrix approximations such as TONGA BID17 and the Hessian-free (HF) approach BID20 BID21 BID4 BID30 demonstrated the potential of such methods, they never achieved wide adoption due to issues of scalability (to large models in the case of the former, and large datasets in the case of the latter).Motivated in part by these older results and by the more recent success of centering and normalization methods (e.g. BID31 BID34 BID15 a new family of methods has emerged that are based on non-diagonal curvature matrix approximations the rely on the special structure of neural networks.

Such methods, which include Kronecker-factored approximated curvature (K-FAC) BID24 , Natural Neural Nets BID5 , Practical Riemannian Neural Networks BID18 , and others BID29 , have achieved state-of-the-art optimization performance on various challenging neural network training tasks and benchmarks.

While the original K-FAC method is applicable only to standard feed-forward networks with fully connected layers, it has recently been extended to handle convolutional networks BID8 through the introduction of the "Kronecker Factors for Convolution" (KFC) approximation.

BID2 later developed a distributed asynchronous version which proposed additional approximations to handle very large hidden layers.

In this work we develop a new family of curvature matrix approximations for recurrent neural networks (RNNs) within the same design space.

As in the original K-FAC approximation and the KFC approximation, we focus on the Fisher information matrix (a popular choice of curvature matrix), and show how it can be approximated in different ways through the adoption of various approximating assumptions on the statistics of the network's gradients.

Our main novel technical contribution is an approximation which uses a chain-structured linear Gaussian graphical model to describe the statistical relationship between gradient contributions coming from different time-steps.

Somewhat remarkably, it is possible to sum the required cross-moments to obtain a Fisher approximations which has enough special algebraic structure that it can still be efficiently inverted.

In experiments we demonstrate the usefulness of our approximations on several challenging RNN training tasks.

We denote by f (x, ??) the neural network function associated evaluated on input x, where ?? are the parameters.

We will assume a loss function of the form L(y, z) = ??? log r(y|z), where r is the density function associated with a predictive distribution R. The loss associated with a single training case is then given by L(y, f (x, ??)) ??? ??? log r(y|f (x, ??)).

Throughout the rest of this document we will use the following special notation for derivatives of the single-case loss w.r.t.

some arbitrary variable Z (possibly matrix-valued): DISPLAYFORM0 The objective function which we wish to minimize is the expected loss h(??) = E Q [L(y, f (x, ??))]over the training distribution Q on x and y.

The Fisher information matrix (aka "the Fisher") associated with the model's predictive distribution P y|x (??) is given by F = E D??D?? = cov(D??, D??).Note that here, and for the remainder of this paper, y is taken to be distributed according to the model's predictive conditional distribution P y|x (??), so that E[DZ] = 0 for any variable Z that is conditionally independent of y given the value of f (x, ??) (this includes D??).

All expectations and covariances are defined accordingly.

This is done because the expectation that defines the Fisher information matrix uses P y|x (??).

If we were to instead use the training distribution Q y|x on y, we would essentially be computing to the "empirical Fisher" (or approximations thereof), which as argued by BID23 is a less appropriate choice for a curvature matrix than the true Fisher.

The natural gradient is defined as F ???1 ???h, and is the update direction used in natural gradient descent.

As argued by Amari (1998), natural gradient descent has the two key advantages: it is invariant to the parameterization of the model, and has "Fisher efficient" convergence 1 .

However, as shown by BID23 these two facts have several important caveats.

First, the parameterization invariance only holds approximately in practice when non-infinitesimal step-sizes are used.

Second, Fisher efficiency is actually a weak property possessed by simpler methods like SGD with Polyak/parameter averaging BID28 , and even then will only be achieved when the method converges to a global minimizer and the model is capable of perfectly capturing the true distribution of y given x.

An alternative explanation for the empirical success of the natural gradient method is that it is a 2nd-order method, whose update minimizes the following local quadratic approximation to the objective h(?? + ??): DISPLAYFORM0 This is similar to the 2nd-order Taylor series approximation of h(?? + ??), but with the Fisher substituted in for the Hessian.

This substitution can be justified by the observation that the Fisher is a kind of PSD approximation to the Hessian BID27 BID23 .

And as argued by BID23 , while stochastic 2nd-order methods like natural gradient descent cannot beat the asymptotically optimal Fisher efficient convergence achieved by SGD with Polyak averaging, they can enjoy better pre-asymptotic convergence rates.

Moreover, insofar as gradient noise can be mitigated through the use of large mini-batches -so that stochastic optimization starts to resemble deterministic optimization -the theoretical advantages of 2nd-order methods become further pronounced, which agrees with the empirical observation that the use of large-minibatches speeds up 2nd-methods much more than 1st-order methods BID24 BID2 .In addition to providing an arguably better theoretical argument for the success of natural gradient methods, their interpretation as 2nd-order methods also justifies the common practice of computing the natural gradient as (F + ??I) ???1 ???h instead of F ???1 ???h.

In particular, this practice can be viewed as a type of "update damping/regularization", where one encourages ?? to lie within some region around ?? = 0 where eqn.

1 remains a trustworthy approximation (e.g. BID26 BID22 .

Because modern neural network have millions (or even billions) of parameters it is computationally too expensive to compute and invert the Fisher.

To address this problem, the K-FAC method of BID24 uses a block-diagonal approximation of the Fisher (where the blocks correspond to entire layers/weight matrices), and where the blocks are further approximated as Kronecker products between much smaller matrices.

The details of this approximation are given in the brief derivation below.

Let W be a weight matrix in the network which computes the mapping s = W a, where a and s are vector-valued inputs and outputs respectively and denote g = Ds.

As in the original K-FAC paper we will assume that a includes a homogeneous coordinate with value 1 so that the bias vector may be folded into the matrix W .Here and throughout the rest of this document, F will refer to the block of the Fisher corresponding to this particular weight-matrix W .The Kronecker product of matrices B and C, denoted by B ??? C for matrices B ??? R m??n and C of arbitrary dimensions, is a block matrix defined by DISPLAYFORM0 Note that the Kronecker product has many convenient properties that we will make use of in this paper. (See Van Loan (2000) for a good discussion of the Kronecker product and its properties.)A simple application of the chain rule gives DW = ga .

If we approximate g and a as statistically independent, we can write F as DISPLAYFORM1 where we have defined DISPLAYFORM2 The matrices A and G can be estimated using simple Monte Carlo methods, and averaged over lots of data by taking an exponentially decaying average across mini-batches.

This is the basic Kronecker factored approximation BID12 BID24 BID29 , which is related to the approximation made in the Natural Neural Nets approach BID5 .

It is shown by BID24 that the approximation is equivalent to neglecting the higher-order cumulants of the as and gs, or equivalently, assuming that they are Gaussian distributed.

To see why this approximation is useful, we observe that inversion and multiplication of a vector by F amounts to inverting the factor matrices A and G and performing matrix-matrix multiplications with them, due to the following two basic identities: DISPLAYFORM3 The required inversion and matrix multiplication operations are usually computational feasible because the factor matrices have dimensions equal to the size of the layers, which is typically just a few thousand.

And when they are not, additional approximations can be applied, such as approximate/iterative inversion BID29 , or additional Kronecker-factorization applied to either A or G BID2 .

Moreover, the computation of the inverses can be amortized across iterations of the optimizer at the cost of introducing some staleness into the estimates.

The basic Kronecker-factored approximation to the Fisher block F described in the previous section assumed that the weight matrix W was used to compute a single mapping of the form s = W a. When W is used to compute multiple such mappings, as is often the case for RNNs, or a mapping of a different flavor, as is the case for convolutional networks (CNNs), the approximation is not applicable, strictly speaking.

BID8 recently showed that by making additional approximating assumptions, the basic Kronecker-factored approximation can be extended to convolutional layers.

This new approximation, called "KFC", is derived by assuming that gradient contributions coming from different spatial locations are uncorrelated, and that their intra and inter-location statistics are spatially homogeneous, in the sense that they look the same from all reference locations.

These assumptions are referred to "spatially uncorrelated derivatives" and "spatial homogeneity," respectively.

In this section we give the main technical contribution of this paper, which is a family of Kroneckerbased approximations of F that can be applied to RNNs.

To build this we will apply various combinations of the approximating assumptions used to derive the original K-FAC and KFC approaches, along with several new ones, including an approximation which works by modelling the statistical structure between the gradient contributions from time-steps using a chain-structured linear Gaussian graphical model.

Let W be some weight matrix which is used at T different time-steps (or positions) to compute the mapping DISPLAYFORM0 where t indexes the time-step.

T is allowed to vary between different training cases.

Defining g t = Ds t , the gradient of the single-case loss with respect to W can be written as DISPLAYFORM1 where D t W = g t a t denotes the contribution to the gradient from time-step t. When it is more convenient to work with the vector-representations of the matrix-valued variables D t W we will use the notation DISPLAYFORM2 so that vec(DW ) = T t=1 w t .

Let F T denote the conditional Fisher of DW for a particular value of T .

We have DISPLAYFORM3 Observe that F can be computed from DISPLAYFORM4 To proceed with our goal of obtaining a tractable approximation to F we will make several approximating assumptions, as discussed in the next section.

One simplifying approximation we will make immediately is that T is independent of the w t 's, so DISPLAYFORM0 In this case eqn.

3 can be written as DISPLAYFORM1 where we have defined DISPLAYFORM2 Independence of T and the w t 's is a reasonable approximation assumption to make because 1) for many datasets T is constant (which formally implies independence), and 2) even when T varies substantially, shorter sequences will typically have similar statistical properties to longer ones (e.g. short paragraphs of text versus longer paragraphs).

Another convenient and natural approximating assumption we will make is that the w t 's are temporally homogeneous, which is to say that the statistical relationship between any w t and w s depends only on their distance in time (d = t ??? s).

This is analogous to the "spatial homogeneity" assumption of KFC.

Under this assumption the following single-subscript notation is well-defined: DISPLAYFORM0 Applying this notation to eqn.

4 we have DISPLAYFORM1 where we have used the fact that there are T ??? |d| ways to write d as t ??? s for t, s ??? {1, 2, . . .

, T }.Temporal homogeneity is a pretty mild approximation, and is analogous to the frequently used "steady-state assumption" from dynamical systems.

Essentially, it is the assumption that the Markov chain defined by the system "mixes" and reaches its equilibrium distribution.

If the system has any randomness, and its external inputs reach steady-state, the steady-state assumption is quite accurate for states sufficiently far from the beginning of the sequence (which will be most of them).

If we have that a t and g s are pair-wise independent for each t and s, which is the obvious generalization of the basic approximation used to derive the K-FAC approach, then following a similar derivation to the one from Section 2.3 we have DISPLAYFORM0 where we have defined DISPLAYFORM1 Extending our temporal homogeneity assumption from the w t 's to the a t 's and g t 's (which is natural to do since w t = vec(g t a t )), the following notation becomes well-defined: DISPLAYFORM2 which allows us to write DISPLAYFORM3

Given the approximating assumptions made in the previous subsections we have DISPLAYFORM0 Assuming for the moment that all of the training sequences have the same length, so that F = F T0 for some T 0 , we have that F will be the sum of 2T 0 + 1 Kronecker products.

Without assuming any additional structure, such as a relationship between the various DISPLAYFORM1 there doesn't appear to be any efficient way to invert such a sum.

One can use the elementary identity DISPLAYFORM2 to invert a single Kronecker product, and there exists decomposition-based methods to efficiently invert sums of two Kronecker products (see BID24 ), however there is no known efficient algorithm for inverting sums of three or more Kronecker products.

Thus is appears that we must make additional approximating assumptions in order to proceed.

If we assume that the contributions to the gradient (the w t 's) are independent across time, or at least uncorrelated, this means that V d = 0 for d = 0.

This is analogous to the "spatially uncorrelated derivatives" assumption of KFC.In this case eqn.

5 simplifies to DISPLAYFORM0 Using the identities in eqn.

2, and the symmetry of A 0 and G 0 , we can thus efficiently multiply F ???1 by a vector z = vec(Z) using the formula DISPLAYFORM1 This is, up to normalization by E T [T ], identical to the inverse multiplication formula used in the original K-FAC approximation for fully-connected layers.

We note that E T [T ] = i ?? i T i , where T i are the different values of T , and ?? i 0 are normalized weights (with i ?? i = 1) that measure their proportions in the training set.

As we saw in Section 3.3, the approximation assumptions made in Section 3.2 (independence of T , temporal homogeneity, and independence between the a t 's and the g t 's), aren't sufficient to yield a tractable formula for F ???1 .

And while additionally assuming independence across time of the w t 's is sufficient (as shown in Section 3.4), it seems like an overly severe approximation to make.

In this section we consider a less severe approximation which we will show still produces a tractable F ???1 .

In particular, we will assume that the statistical relationship of the w t 's is described by a simple linear Gaussian graphical model (LGGM) with a compact parameterization (whose size is independent of T ).

Such an approach to computing a tractable Fisher approximations was first explored by BID9 for RBMs, although our use of it here is substantially different, and requires additional mathematical machinery.

The model we will use is a fairly natural one.

It is a linear Gaussian graphical model with a onedimensional chain structure corresponding to time.

The graphical structure of our model is given by the following picture: DISPLAYFORM0 Variables in the model evolve forward in time according to the following equation: DISPLAYFORM1 where ?? is a square matrix and t are i.i.d.

from N (0, ??) for some positive definite matrix ?? (which is the conditional covariance of w t given w t???1 ).Due to the well-known equivalence between directed and undirected Gaussian graphical models for tree-structured graphs like this one, the decision of whether to make the edges directed or undirected, and whether to have them point forwards or backwards in time, are irrelevant from a modeling perspective (and thus to the Fisher approximation we eventually compute).

We will use a directed representation purely for mathematical convenience.

We will assume that our model extends infinitely in both directions, with indices in the range (??????, ???), so that the w t 's are all in their stationary distribution (with respect to time).

For this to yield a well-defined model we require that ?? has spectral radius < 1.The intuition behind this model structure is clear.

The correlations between gradient contributions (the w t 's) at two different time-steps should be reasonably well explained by the gradient contributions made at time-steps between them.

In other words, they should be approximately Markovian.

We know that the gradient computations are generated by a process, Back-prop Through Time (BPTT), where information flows only between consecutive time-steps (forwards through time during the "forward pass", and backwards during the "backwards pass").

This process involves temporal quantities which are external to the w t 's, such as the inputs x and activations for other layers, which essentially act as "hidden variables".

The evolution of these external quantities may be described by their own separate temporal dynamics (e.g. the unknown process which generates the true x's), and thus the w t 's won't be Markovian in general.

But insofar as the w t 's (or equivalently the a t 's and g t 's) encode the relevant information contained in these external variables, they should be approximately Markovian. (If they contained all of the information they would be exactly Markovian.)A similar approximation across consecutive layers was made in the "block-tridiagonal" version of the original K-FAC approach.

It was shown by BID24 that this approximation was a pretty reasonable one.

The linear-Gaussian assumption meanwhile is a more severe one to make, but it seems necessary for there to be any hope that the required expectations remain tractable.

Define the following "transformed" versions of F T and ??: DISPLAYFORM0 As shown in Section A.1 of the appendix we hav?? DISPLAYFORM1 where DISPLAYFORM2 (Note that rational functions can be evaluated with matrix arguments in this way, as discussed in Section A.1.)Our goal is to computeF ???1 , from which we can recover F ???1 via the simple relation DISPLAYFORM3 Unfortunately it doesn't appear to be possible to simplify this formula sufficiently enough to allow for the efficient computation ofF DISPLAYFORM4 ???1 when?? is a Kronecker product (which it will be when V 0 and V 1 are).

The difficulty is due to both the appearance of?? and its transpose (which are not codiagonalizable/commutative in general), and various higher powers of??.To proceed from this point and obtain a formula which can be efficiently evaluated when?? is a Kronecker product, we will make one of two simplifying assumptions/approximations, which we call "Option 1" and "Option 2" respectively.

These are explained in the next two subsections.3.5.2 OPTION 1: DISPLAYFORM5 If V 1 (the cross-moment over time) is symmetric, this implies that?? DISPLAYFORM6 is also symmetric.

Thus by eqn.

7 we hav?? DISPLAYFORM7 Let U diag(??)U =?? be the eigen-decomposition of??. By the above expression forF T we have DISPLAYFORM8 DISPLAYFORM9 We thus hav?? DISPLAYFORM10 Inverting both sides of this yieldsF DISPLAYFORM11 where we have defined DISPLAYFORM12 This expression can be efficiently evaluated when?? is a Kronecker product since the eigendecomposition of a Kronecker product can be easily obtained from the eigendecomposition of the factors.

DISPLAYFORM13 ) and is thus easy to perform.

See Section 3.5.5 for further details.

V 1 is symmetric if and only if?? is symmetric.

And as shown in the proof of Proposition 1 (see Appendix A.1)?? has the interpretation of being the transition matrix of an LGGM which describes the evolution of "whitened" versions of the w t 's (given by?? t = V ???1/2 0 w t ).

Linear dynamical systems with symmetric transition matrices arise frequently in machine learning and related areas BID14 BID11 , particularly because of the algorithmic techniques they enable.

Intuitively, a symmetric transition matrix allows allows one to model exponential decay of different basis components of the signal over time, but not rotations between these components (which are required to model sinusoidal/oscillating signals).Note that the observed/measured V 1 may or may not be exactly symmetric up to numerical precision, even if it well approximated as symmetric.

For these calculations to make sense it must be exactly symmetric, and so even if it turns out to be approximately symmetric one should ensure that it is exactly so by using the symmetrized version (V 1 + V 1 )/2.

If V 1 is not well approximated as symmetric, another option is to approximat?? DISPLAYFORM0 , where we defin?? DISPLAYFORM1 This is essentially equivalent to the assumption that the training sequences are all infinitely long, which may be a reasonable one to make in practice.

We re-scale by the factor T T to achieve the proper scaling characteristics ofF T , and to ensure that the limit actually exists.

As shown in Section A.2 of the appendix this yields the following remarkably simple expression for DISPLAYFORM2 Despite the fact that it includes both?? and?? , this formula can be efficiently evaluated when?? is a Kronecker product due to the existance of decomposition-based techniques for inverting matrices of the form A ??? B + C ??? D. See Section 3.5.5 for further details.

This approximation can break down if some of the linear components of?? t have temporal autocorrelations close to 1 (i.e.[??] i ??? 1 for some i) and T is relatively small.

In such a case we will have that [??]T i is large for some i (despite being raised to the T -th power) so thatF (???) T may essentially "overcount" the amount of temporal correlation that contributes to the sum.

This can be made more concrete by noting that the approximation is essentially equivalent to taking DISPLAYFORM3 We can express the error of this as DISPLAYFORM4 It is easy to see how this expression, when evaluated at x = [??] i , might be large when [??] i is close to 1, and T is relatively small.

The formulae forF ???1 from the previous sections depend on the quantity?? DISPLAYFORM0 , and so it remains to compute ??.

We observe that DISPLAYFORM1 Right-multiplying both sides by V 0 yields ?? = V 1 V ???1 0 .

Thus, given estimates of V 0 and V 1 , we may compute an estimate of?? as?? DISPLAYFORM2 In practice we estimate V 0 and V 1 by forming estimates of their Kronecker factors and taking the product.

The factors themselves are estimated using exponentially decayed averages over mini-batch estimates.

And the mini-batch estimates are in turn computed by averaging over cases and summing across time-steps, before divide by the expected number of time-steps.

For example, for A 0 and A 1 these the mini-batch estimates are averages of DISPLAYFORM3 T t=1 a t a t and DISPLAYFORM4 T ???1 t=1 a t+1 a t , respectively.

Note that as long as V 0 is computed as the 2nd-order moment of some empirical data, and V 1 computed as the 2nd-order moment between that same data and a temporally shifted version, the spectral radius of?? = V ???1/2 0 DISPLAYFORM5 (and similarly ?? = V 1 V ???1 0 ) will indeed be less than or equal to 1, as we prove in Section B.2 of the appendix.

This bound on the spectral radius is a necessary condition for our infinite chain-structured Gaussian graphical model to be well-defined, and for our calculations to make sense.

The sufficient condition that the spectral radius is actually less than 1 will most often be satisfied too, except in the unlikely event that some eigen component remains perfectly constant across time.

But even if this somehow happens, the inclusion within the given V 0 of some damping/regularization term such as ??I will naturally deal with this problem.3.5.5 EFFICIENT IMPLEMENTATION ASSUMING V 0 AND V 1 ARE KRONECKER-FACTORED It remains to show that the approximations developed in Section 3.5 can be combined with the Kronecker-factored approximations for V 0 and V 1 from Section 3.2.3 to yield an efficient algorithm for computing F ???1 z for an arbitrary vector z = vec(Z).

This is a straightforward although very long computation which we leave to Section C of the appendix.

Full pseudo-code for the resulting algorithms is given in Section C.3.

As they only involve symmetric eigen-decomposition and matrix-matrix products with matrices the size of A 0 and G 0 they are only several times more expensive to compute than eqn.

6.

This extra overhead will often be negligible since the gradient computation via BPTT, whose costs scales with the sequence length T , tends to dominate all the other costs.

To demonstrate the benefit of our novel curvature matrix approximations for RNNs, we empirically evaluated them within the standard "distributed K-FAC" framework BID2 on two different RNN training tasks.

The 2nd-order statistics (i.e. the Kronecker factors A 0 , A 1 , G 0 , and G 1 ) are accumulated through an exponential moving average during training.

When computing our approximate inverse Fisher, factored Tikhonov damping BID24 was applied to V 0 = G 0 ??? A 0 .We used a single machine with 16 CPU cores and a Nvidia K40 GPU for all the experiments.

The additional computations required to get the approximate Fisher inverse from these statistics (i.e. the "pre-processing steps" described in Section C.3) are performed asynchronously on the CPUs, while the GPU is used for the usual forward evaluation and back-propagation to compute the gradient.

Updates are computed using the most recently computed values of these (which are allowed to be stale), so there is minimal per-iteration computational overhead compared SGD.

We adopted the step-size selection technique described in Section 5 of BID2 , as we found it let us use larger learning rates without compromising the stability of the optimization.

The hyperparameters of our approach, which include the max learning rate and trust-region size for the aforementioned step-size selection procedure, as well as the momentum, damping constants, and the decay-rate for the second-order statistics, as well as the hyper-parameters of the baseline methods, were tuned using a grid search.

Word-level language model: We start by applying our method to a two-layer RNN based on the well-studied Long Short-Term Memory (LSTM) architecture BID13 for a word-level language modeling task on the Penn-TreeBank (PTB) dataset BID19 following the experimental setup in Zaremba et al. (2014) .

The gradients are computed using a fixed sequence length truncated back-propagation scheme in which the initial states of the recurrent hidden units are inherited from the final state of the preceding sequence.

The truncation length used in the experiments is 35 timesteps.

The learning rate is given by a carefully tuned decaying schedule (whose base value we tune along with the other hyperparamters).In our experiments we simply substitute their optimizer with our modified distributed K-FAC optimizer that uses our proposed RNN Fisher approximations.

We performed experiments on two different sizes of the same architecture, which use two-layer 650 and 1024 LSTM units respectively.

LSTMs have 4 groups of internal units: input gates, output gates, forget gates, and update candidates.

We treat the 4 weight matrices that compute the pre-activations to each of these as distinct for the purposes of defining Fisher blocks (whereas many LSTM implementations treat them as one big matrix).

This results in smaller Kronecker factors that are cheaper to compute and invert.

Because the typical vocabulary size used for PTB is 10,000, the Fisher blocks for the input embedding layer and output layer (computing the logits to the softmax) each contain a 10,000 by 10,000 sizes Kronecker factor, which is too large to be inverted with any reasonable frequency.

Given that the input vector uses a one-hot encoding it is easy to see its associated factor is actually diagonal, and so we can store and invert it as such.

Meanwhile the large factor associated with the output isn't diagonal, but we nonetheless approximate it as such for the sake of efficiency.

In our experiments we found that each parameter update of our method required about 80% more wall-clock time than an SGD update (using mini-batch size of 200) although the updates made more much progress.

In FIG1 , we plot the training progress as a function of the number of parameter updates.

While Adam outperforms SGD in the first few epochs, SGD obtains a lower loss at the end of training.

We found the recent layer-normalization technique BID3 helps speed up Adam considerably, but it hurts the SGD performance.

Such an observation is consistent with previous findings.

In comparison, our proposed method still significantly outperform both the Adam and the SGD baselines even with the help of layer-normalization.

While optimization performance, not generalization performance, is the focus of this paper, we have included validation performance data in the appendix for the sake of completeness. (See FIG10 in Appendix D.) Not surprisingly, we found that the 2nd-order methods, including our approach and diagonal ones like Adam, tended to overfit more than SGD on these tasks.

The tendency for SGD w/ early-stopping to self-regularize is well-documented, and there are many compelling theories about why this happens (e.g. BID6 BID10 .

It is also well-known that 2nd-order methods, including K-FAC and diagonal methods like Adam/RMSprop, dont self-regularize nearly as much (e.g. Wilson et al., 2017; BID16 .

We feel that this problem can likely be addressed through the careful application of additional explicit regularization (e.g. increased weight decay, drop-out, etc) and/or model modifications, but that exploring this is outside of the scope of this paper.

To further investigate the optimization performance of our proposed Fisher approximation, we use a small two layer LSTM with 128 units to model the character sequences on the Penn-TreeBank (PTB) dataset BID19 .

We employ the same data partition in BID25 .

We plotted the bits-per-character vs the number of parameter updates and the wall-clock times in FIG2 .

The K-FAC updates were roughly twice as time-consuming to compute as the Adam updates in our implementation.

Despite this, our results demonstrate that K-FAC has a significant advantage over the Adam baseline in terms of wall-clock time.

To further investigate the potential benefits of using our approach over existing methods, we applied it to the Differentiable Neural Computer (DNC) model BID7 for learning simple algorithmic programs.

Recently, there have been several attempts (Weston et al., 2014; BID7 to extend the existing RNN models to incorporate more long-term memory storage devices in order to help solve problems beyond simple sequence prediction tasks.

Although these extended RNNs could potentially be more powerful than simple LSTMs, they often require thousands of parameter updates to learn simple copy tasks BID7 .

Both the complexity of these models and the difficulty of the learning tasks have posed a significant challenge to commonly used optimization methods.

The DNC model is designed to solve structured algorithmic tasks by using an LSTM to control an external read-write memory.

We applied the Fisher-based precondition to compute the updates for both the weights in the LSTM controller and the read-write weight matrices used to interface with the memory.

We trained the model on a simple repeated copy task in which the DNC needs to recreate a series of two random binary sequences after they are presented as inputs.

The total length of the sequence is fixed to 22 time-steps.

From FIG3 , we see that our method significantly outperforms the Adam baseline in terms of update count, although only provides a modest improvement in wallclock time.

This gap is explained by the fact that the iterations were significantly more time-consuming to compute relative to the gradient computations than they were in previous two experiments on language models.

This is likely due to a different trade-off in terms of the gradient computation vs the overheads specific to our method owing to smallness of the model and dataset.

With more careful engineering to reduce the communication costs, and/or a larger model and dataset, we would expect to see a bigger improvement in wall-clock time.

We have presented a new family of approximations to the Fisher information matrix of recurrent neural networks (RNNs), extending previous work on Kronecker-factored approximations.

With this contribution, recurrent networks can now finally be trained with the K-FAC optimization method.

We have demonstrated that our new approximations substantially reduce the required number of iterations for convergence vs standard baseline optimizers on several realistic tasks.

And we have also shown that in a modern distributed training setup this results in a substantial savings in wallclock time as well.

Jason Weston, Sumit Chopra, and Antoine Bordes.

A SUPPLEMENTARY COMPUTATIONS A.1 PROOFS FOR SECTION 3.5.1Proposition 1 GivenF DISPLAYFORM0 , which can be seen as follows: DISPLAYFORM1 And using DISPLAYFORM2 Setting d = 1 and multiplying both sides by V 0 (which is assumed to be invertible) one can also derive the following simple formula for ??: DISPLAYFORM3 To proceed from here we define a "transformed" version of the original chain-structured linearGaussian graphical model whose variables are?? t = V ???1/2 0 w t .

(Here we assume that V 0 is invertible -it is symmetric by definition.)

All quantities related to the original model have their analogues in the transformed model, which we indicate with the hat symbol??.In the transformed model the 2nd-order moments of the?? t 's are given b?? DISPLAYFORM4 We observe thatV 0 = I.Analogously to the original model, the transformed version obey?? DISPLAYFORM5 =V 1 (usingV 0 = I).

This can be seen by noting that DISPLAYFORM6 It also remains true that the spectral radius of?? is less than 1, which can be seen in at least one of two ways: by noticing that the transformed model is well-defined in the infinite limit if and only if the original one is, or that?? DISPLAYFORM7 is a similar matrix to ?? (in the technical sense) and hence has the same eigenvalues.

As the transformed model is isomorphic to the original one, all of the previously derived relationships which held for it also hold here, simply by replacing each quantity with its transformed version (denoted by the hat symbol??).Given these relations (included the transformed analogue of equation 5)

we can expressF T a?? DISPLAYFORM8 It is a well-known fact that one can evaluate rational functions, and functions that are the limiting values of sequences of rational functions, with matrix arguments.

This is done by replacing scalar multiplication with matrix multiplication, division with matrix inversion, and scalar constants with scalar multiples of the identity matrix, etc.

Note that because sums of powers and inverses of matrices are co-diagonalizable/commutative when the matrices themselves are, there is no issue of ambiguity caused by mixing commutative and non-commutative algebra in this way.

Moreover, the value of some such function f (x), given a matrix argument B, is DISPLAYFORM9 DISPLAYFORM10 undefined from some i, either because of a division by zero, or because the limit which defines f (x) doesn't converge for x = [b] i , then f (B) doesn't exist for that particular B (and otherwise it does).We observe that our above expression for F T can be rewritten a?? DISPLAYFORM11 By Proposition 3 in Appendix B.1, we have for x = 1 that DISPLAYFORM12 Let U diag(??)U ???1 =?? be the eigendecomposition of??. Because?? has a spectral radius less than 1, we have |[??] i | < 1 for each i (so that in particular [??] i = 1), and thus we can evaluate ?? T (??) and ?? T (?? ) according to the above formula for ?? T (x).

Proposition 2 Suppose we approximateF DISPLAYFORM0 ], where we have defined DISPLAYFORM1 Then we haveF DISPLAYFORM2

From eqn.

7 we have that DISPLAYFORM0 To evaluate this we first term note that DISPLAYFORM1 where we have defined DISPLAYFORM2 For |x| < 1 we have that lim T ?????? x T = 0, from which it follows that DISPLAYFORM3 Let U diag(??)U ???1 =?? be the eigendecomposition of??. Using the fact that |[??] i | < 1 (as established in Section A.1) we can use the above expression to evaluate ??(x) at both x =?? and x =?? , which yieldsF DISPLAYFORM4 Pre-multiplying both sides by I ????? , and post-multiplying both sides by I ?????, we have DISPLAYFORM5 Then applying the reverse operation gives DISPLAYFORM6 Taking the expectation over T give?? DISPLAYFORM7 Finally, inverting both sides yield?? DISPLAYFORM8

Proposition 3 Suppose x ??? C, x = 0, and T is a non-negative integer.

We have DISPLAYFORM0 Another way to express this is to use the geometric series formula DISPLAYFORM1 1???x (which holds for x = 0) before computing the derivative.

This gives DISPLAYFORM2 Thus we have DISPLAYFORM3 And so DISPLAYFORM4 where we have again used the geometric series formula DISPLAYFORM5 1???x on the second line.

In what follows all quantities are computed using their defining formulae, starting from the estimated values of A 0 , A 1 , G 0 , and G 1 .First we observe that since?? DISPLAYFORM0 is similar to ?? (in the technical sense of the word), they share the same eigenvalues.

Thus it suffices bound to the spectral radius of DISPLAYFORM1 Because the eigendecomposition of a Kronecker product is the Kronecker product of the decompositions of the factors we have that ??( DISPLAYFORM2 , where ??(X) denotes the spectral radius of a matrix X.Thus it suffices to show that ??(A 1 A ???1 0 ) ??? 1 and ??(G 1 G ???1 0 ) ??? 1 for A i and G i as computed by the estimation scheme outlined in Section 3.5.4.

Recall that this is the exponentially decayed average of mini-batch averages of estimators of the form DISPLAYFORM3 T t=1 a t a t and DISPLAYFORM4 In the remainder of this section we will show that ??( DISPLAYFORM5 Provided that the exponentially decayed averages for A 0 and A 1 are computed in the same way and use the same normalizers (i.e. 1/(mE T [T ])) we thus have that the matrix DISPLAYFORM6 is a positively-weighted linear combination of terms of the form DISPLAYFORM7 where the various a t 's are computed on different data using current and previous model parameters.

It thus follows that A 0.

The inequality ??(A 1 A ???1 0 ) ??? 1 now follows from the following lemma.

Lemma 1 Consider a real, symmetric, positive semi-definite block matrix DISPLAYFORM8 If B is invertible then we have ??(CB ???1 ) ??? 1, and further if the block matrix is positive definite we have ??(CB ???1 ) < 1.Proof Because similarity transformations preserve eigenvalues, the statement is equivalent to DISPLAYFORM9 .

Because any induced matrix norm is an upperbound on the spectral radius, it suffices to show X 2 = ?? max (X) ??? 1, where ?? max (X) denotes the largest singular value of X.By taking Schur complements of the block matrix (11) we have DISPLAYFORM10 Note that the Schur complement is PSD because the original block matrix itself is (e.g. Zhang, 2006) .Using the fact that Z ??? B 1 2 ZB 1 2 maps positive semidefinite matrices to positive semidefinite matrices, this implies DISPLAYFORM11 When the block matrix is positive definite, the inequalities become strict.

DISPLAYFORM12 The ultimate goal of our calculations is to efficiently compute the matrix-vector product of some arbitrary vector z (which will often be the gradient vec(DW )) with our inverse Fisher approximation F ???1 .

That is, we wish to compute F ???1 z.

It will be convenient to assume that z is given as a matrix Z (with the same dimensions as DW ) so that z = vec(Z).

again.

We will suppose that we are given A 0 , A 1 , G 0 , and G 1 such that DISPLAYFORM13 We then have V can be computed using the eigendecompositions of A 0 and G 0 , for example.

The procedure to efficiently multiply a vector z byF ???1 is more involved and depends on which approximation "option" we are using.

However one immediate useful insight we can make before specializing to Option 1 or Option 2 is that?? can be written as a Kronecker product as follows: DISPLAYFORM14 DISPLAYFORM15 , is symmetric.

We note that V 1 = A 1 ??? G 1 will be symmetric if and only if both A 1 and G 1 are.

Our task is to compute the matrix-vector product DISPLAYFORM16 where U diag(??)U is the eigendecomposition of??, and ??(x) is defined as in Section 3.5.3, eqn.

8.

To do this we first multiply the vector by U , then by diag(??(??)), and then finally by U .The eigendecomposition of?? can be computed efficiently using its Kronecker product structure.

In particular, we compute the eigendecompositions of each factor as DISPLAYFORM17 from which we can write the eigendecomposition U diag(??)U ???1 of?? a?? DISPLAYFORM18 In other words, we have U = U A ??? U G and?? = vec(?? G?? A ).To multiply z by U = U A ??? U G we use the identity (C ??? B) vec(X) = vec(B XC ) which gives DISPLAYFORM19 Similarly, the multiplication of z by U = U A ??? U G can be computed as vec(U G ZU A ).

where denotes entry-wise multiplication of matrices.

In summary we have thatF ???1 z can be computed in matrix form as DISPLAYFORM20 We note that computing ??([?? G ] i [?? A ] j ) is trivial since it is just a scalar evaluation of the rational function ??(x).

For Option 2 we must compute the matrix-vector product DISPLAYFORM0 To do this we will first multiply by I ????? , then by (I ????? ?? ) ???1 , and then by I ?????, before finally dividing the result by i ?? i T i .To compute the matrix-vector product (I ????? )z we use the identity (C ??? B) vec(X) = vec(B XC ) while noting that?? = (?? A ????? G ) =?? A ????? G .

This gives The matrix form of this is simply Z ????? G Z?? A .We may similarly compute (I ?????)z in matrix form as Z ????? G Z?? A .The harder task is to compute (I ????? ?? ) ???1 z, which is what we tackle next.

We first observe that DISPLAYFORM1 Given the eigendecompositions E A diag(m A )E A = M A and E G diag(m G )E G = M G we can thus compute the larger eigendecomposition as DISPLAYFORM2 where l is the vector of ones.

Using the eigendecomposition the inverse can then be easily computed as DISPLAYFORM3 Thus (I ????? ?? ) ???1 z may be computed by first multiplying by (E A ??? E G ), then by diag(vec(ll ??? m G m A )) ???1 (which in matrix form corresponds to element-wise division by ll ??? m G m A ), and then by E A ??? E G .

The matrix form of this is DISPLAYFORM4 where B C denotes element-wise division of the matrix B by the matrix C.

@highlight

We extend the K-FAC method to RNNs by developing a new family of Fisher approximations.

@highlight

The authors extends the K-FAC method to RNNs and presents 3 ways of approximating F, showing optimization results on 3 datasets, which outperforms ADAM in both number of updates and computation time.

@highlight

Proposes to extend the Kronecker-factor Appropriate Curvature optimization method to the setting of recurrent neural networks.

@highlight

The authors present a second-order method that is specifically designed for RNNs
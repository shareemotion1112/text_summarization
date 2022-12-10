Adaptive regularization methods pre-multiply a descent direction by a preconditioning matrix.

Due to the large number of parameters of machine learning problems, full-matrix preconditioning methods are prohibitively expensive.

We show how to modify full-matrix adaptive regularization in order to make it practical and effective.

We also provide novel theoretical analysis for adaptive regularization in non-convex optimization settings.

The core of our algorithm, termed GGT, consists of efficient inverse computation of square roots of low-rank matrices.

Our preliminary experiments underscore improved convergence rate of GGT across a variety of synthetic tasks and standard deep learning benchmarks.

Stochastic gradient descent is the workhorse behind the recent deep learning revolution.

This simple and age-old algorithm has been supplemented with a variety of enhancements to improve its practical performance, and sometimes its theoretical guarantees.

Amongst the acceleration methods there are three main categories: momentum, adaptive regularization, and variance reduction.

Momentum (in its various incarnations, like heavy-ball or Nesterov acceleration) is the oldest enhancement.

It has a well-developed theory, and is known to improve practical convergence in a variety of tasks, small and large.

It is also easy to implement.

Variance reduction is the most recent advancement; in theory and practice, it is mostly applicable to convex optimization, and is thus less influential in deep learning.

This brings us to adaptive regularization: the most sophisticated, hard to implement, and debated acceleration method.

While state-of-the-art optimizers such as Adam and AdaGrad (Kingma & Ba, 2014; BID13 do use adaptive regularization, they do so in a very limited form: with diagonal matrices, often marketed as per-coordinate adaptive learning-rate methods.

Despite solid theoretical guarantees, the practical value of diagonal adaptive regularization as compared to "vanilla" SGD has been the subject of much debate BID48 .

However, the efficacy of full-matrix adaptive regularization has been relatively unexplored.

This is due to the prohibitive computational cost associated with full-matrix operations: full AdaGrad requires taking the inverse square root of a large matrix.

In this paper, we present GGT, a practical solution to the computational problems plaguing fullmatrix adaptive regularization, making this technique scalable for modern deep models.

At the heart of our method is a simple, GPU-friendly way to apply the inverse square root of the low-rank second-moment matrix of recent gradients; see FIG0 .

GGT's running time is comparable to state-of-the-art optimizers.

We proceed to show that full-matrix preconditioning allows for much better exploitation of anisotropic curvature in loss landscapes.

First, we show synthetic experiments which demonstate clear benefits of GGT over baselines, especially when the problem is ill-conditioned.

Then, we implement GGT at scale, and show that the benefits translate to faster training on standard deep learning benchmarks.

Our improvement is most salient in complicated landscapes like RNN training.

Our algorithm comes with theoretical guarantees.

We give the first proof of convergence to firstorder critical points for an algorithm with adaptive regularization in a stochastic non-convex setting, featuring a rate which is dependent on an adaptive ratio.

We show examples where our bound is stronger than that for SGD, providing some theoretical basis for our empirical findings.

Since the introduction of AdaGrad BID13 , diagonal adaptive regularization has been a mainstay in the machine learning practitioner's toolbox.

A quick perusal of the literature shows that these methods have continued to thrive in the deep learning era, and appear in all major frameworks BID0 BID39 BID9 .

By citation count (or GitHub search hits), Adam (Kingma & Ba, 2014) is by far the most popular adaptive optimizer for training a variety of modern deep models.

For this reason, this paper's exposition is targeted towards a full-matrix dropin replacement for Adam; however, our techniques extend straightforwardly to a plethora of variants, like RMSprop BID45 , Adadelta (Zeiler, 2012) , Nadam BID12 , etc.

Full-matrix adaptive regularization has existed alongside the more commonly used diagonal-matrix manifestation since their common inception in BID13 ); however, a major obstacle to the scalability of these methods is the need for the storage and inversion of square matrices in the model dimension.

This becomes prohibitively expensive in dimension greater than 10 4 , while state-of-theart models regularly exceed 10 7 parameters.

Matrix sketching has been employed to approximate the AdaGrad preconditioner BID25 BID34 ; however, the sketched estimate for the matrix inverse can be sensitive to noise.

In the former, the authors report a 5-10× overhead over AdaGrad, even with < 10 5 model parameters; we could not find a usable GPU implementation for their requisite rank-1 QR update.

BID19 propose a way to do AdaGrad with Kronecker products of full-matrix preconditioners, a more limited setting which requires knowledge of the model's structure.

Finally, as we argue in Section 3.1, there is intrinsic value of "forgetting" past curvature using an exponential window.

With this, a low-rank preconditioning matrix naturally arises, allowing us to bypass the computational need for sketching in the model dimension or architecture-dependent restriction of the preconditioner.

Our algorithm bears a superficial resemblance to L-BFGS BID28 , a version of BFGS BID6 BID15 BID18 BID43 which uses a sliding window of gradient history.

Although some are viable for large-scale implementation, these quasi-Newton methods, along with (subsampled, online, cubic-regularized) Newton methods BID14 BID2 BID30 BID20 BID1 BID8 exhibit very different dynamics than the standard optimizers in deep learning, and thus have not seen widespread adoption.

We find recent deep learning applications of secondorder methods (e.g. BID32 BID33 ) to be intriguing, though outside the scope of this paper.

Recently, the role of adaptive regularization has been a hotly contested topic.

In BID48 , the authors suggest that properly-tuned SGD exhibits superior generalization to adaptive methods.

In turn, propose switching the optimizer from Adam to SGD at the end of training, to reap the advantages of each.

Influentially, Adam's convergence has been the object of recent scrutiny BID42 .

However, Adam continues to enjoy successful convergence in practice; the problematic construction involves pathological outlier gradients.

We do not use the analyses of Adam or AMSGrad.

Several parallel works BID27 BID51 BID47 BID10 BID40 BID50 have studied the convergence of adaptive methods for non-convex optimization, matching the asymptotic iteration complexity of SGD.

Apart from our algorithmic contribution, our work is (to our knowledge) the first attempt to characterize the advantage of adaptivity in terms of the dimension and geometry of the optimization problem.

Our main algorithmic contribution is GGT, an efficient first-order algorithm for full-matrix adaptive preconditioning.

In brief, GGT uses the preconditioner from full-matrix AdaGrad, with gradient history attenuated exponentially as in Adam, and truncated to a window parameter r. The name GGT acts as a convenient mnemonic for the gradient second-moment matrix GG maintained by full-matrix AdaGrad, even though we never compute this matrix.

The mathematical specification of GGT is given in Algorithm 1, in the usual model of stochastic optimization (see Section 4), with gradients ∇f (x).

Notice that the coordinate-wise scaling of Adam is recovered by zeroing out the off-diagonal entries of GG .Algorithm 1 GGT adaptive optimizer 1: Input: initializer x 1 , window size r, learning rate schedule {η t }, DISPLAYFORM0 Receive stochastic gradient ∇f (x t ).

GGT provides the power of full-matrix adaptive regularization at a cost not much larger than SGD.

This crucially exploits the fact only a small window of historical gradients are used for preconditioning.

The intuition for using a small window, as opposed to the entire history, is clear (and time-tested, by the ubiquity of Adam): the curvature of the loss surface changes, rendering previous gradient information obsolete.

We expand on the benefits of forgetting gradients in section 3.1.The fact that the preconditioning matrix is based on a small window of gradients implies that it has low rank.

GGT exploits this fact by computing the inverse square root of the empirical covariance matrix indirectly, as outlined in FIG0 .

In effect, instead of inverting a full matrix in the dimension of parameters, using the special matrix structure GGT inverts a matrix of dimension window-size.

The remainder of this section will discuss efficient implementation and some heuristics.

GGT has provable guarantees even for non-convex optimization: it is guaranteed to converge to a first-order critical point.

Its rate of convergence is never significantly slower than that of SGD, and in some favorable geometric conditions, can be significantly faster.

These theoretical bounds are made precise in section 4.

The window parameter r should be roughly the number of copies of the model that fit in RAM; in our large-scale experiments, we use r = 200.

A pessimistic but principled choice is r = Θ(1/(1 − β 2 )), which truncates on the time scale of the exponential attenuation.

Our key observation, highlighted in FIG0 , is that the inversion of the large low-rank matrix GG can be performed by diagonalizing the small matrix G G, along with some extremely GPU-friendly matrix-vector operations.

The basic intuition is contained in FIG0 , but it remains to include the εI term.

We derive the full update here.

Let DISPLAYFORM0 , and let Σ r ∈

R r×

r be its top left block.

Let U =: [U r U d−r ], so that the columns of U r ∈ R d×r are an orthonormal basis for the column space of G, and DISPLAYFORM1 The first term is none other than an SGD update step.

The rest can be computed by taking the DISPLAYFORM2 We prefer this to taking the direct SVD of G, which is > 10 times slower on GPU.Using a cyclic buffer to store and update G t , the algorithm takes O(dr 2 + r 3 ) (sequential) time per iteration, and O(dr) memory in total.

Iterating over the model parameters to update G t incurs the same overhead cost as usual adaptive optimizers.

The r × d matrix multiplication and r × r SVD operations benefit from decades of extensive hardware-level optimizations.

In the experiments in Section 3, we observed a ∼ 1.3× (CNN) and ∼ 2× (RNN) running-time overhead over SGD; we note that this ratio could be even smaller in reinforcement learning (where the environment causes the time bottleneck), or universally with a more optimized implementation.

Below, we list some practical suggestions for applying GGT to training large-scale models.

Momentum.

In order to bring GGT closer to a drop-in replacement for Adam, we can add momentum to the gradient steps: let v t ← β 1 v t−1 + ∇f (x t ), and apply the preconditioner to v t to compute the update step.

We use momentum in all large-scale experiments, with the standard β 1 = 0.9.

We also get a small performance boost by using v t instead of the gradients to update G t .

On the other hand, as long as r T , it makes little difference to choose β 2 = 1, letting the window (rather than exponential attenuation) forget stale gradient information.

Interpolation with SGD.

We note the possibility of decoupling the scalars ε and 1/ε which appear in the efficient update step.

Appealingly, this allows the user to tune GGT's behavior to be arbitrarily close to that of SGD.Numerical concerns.

For greater numerical stability, it is possible to add a small multiple of the identity matrix (we suggest 10 −6 ) to G G before computing its eigendecomposition, without noticeable differences in training.

In this section, we present an empirical study of GGT.

We begin with some simple experiments, showing that adaptive methods help in the presence of ill-conditioned optimization problems, as well as the value of limited gradient memory.

Next, we evaluate the performance of GGT on largerscale deep learning tasks (and provide some additional such experiments in Appendix B).

Finally, we present some interesting empirical insights on the training dynamics in deep learning models.

Our visualizations of gradient spectra suggest that adaptive optimizers are indeed correcting for changing anisotropic curvature in the loss landscape.

The original theorems on the behavior of adaptive first-order methods are established from the perspective of online convex optimization BID13 .

The dynamics are less understood on realistic loss landscapes in stochastic optimization.

For this reason, we begin our experimental section with some simple empirical comparisons between full-and diagonal-matrix adaptive optimizers and SGD.

In each synthetic experiment, we generated an ill-conditioned landscape, and compared SGD with adaptive optimizers, excluding the typical accompanying heuristics (i.e. no momentum, regularization, or learning rate schedule).

We tested diagonal-matrix preconditioners with and without exponential gradient attenuation (like Adam and AdaGrad, respectively), and their full-matrix analogues.

The experiments were robust with respect to the choice of ε (we used 10 −4 ) and batch size.

In the first synthetic experiment (left), we exhibit an instance of logistic regression in dimension 10, with 10 3 samples generated from an extremely anisotropic (σ 2 max /σ 2 min ≈ 10 4 ) Gaussian distribution, and binary labels determined by a random hyperplane.

SGD converges the slowest, and diagonal AdaGrad consistently accelerates optimization.

Finally, full-matrix preconditioning (using cubic-time matrix inversion) converges the fastest.

In this setting, adding a window improved convergence, but not drastically; we elaborate below.

Next, we show an optimization problem (right) which accentuates the utility of exponentially decaying gradient memory.

We consider the problem of minimizing the logarithmic barrier function of a randomly generated anisotropic polytope, otherwise known as finding its analytic center: this replaces the logistic loss terms with f i (w) = − log(w x i + c i ), with x i generated the same way as above, and c i generated uniformly from [0, 1].

We observed the same ranking of convergence rates as in the first experiment, but the improvement afforded by the window was much clearer.

The primary conclusion of our synthetic experiments is to demonstrate some small-scale settings in which adaptive regularization ameliorates anisotropy in the optimization landscape.

A subtler point is that the windowed variants can help with changing curvature, even for convex losses.

Note that the curvature of the former landscape is constant (in that its Hessian matrix at different locations w only changes by a scalar factor).

The latter setting, in contrast, features a changing curvature (its Hessians do not commute in general), necessitating "forgetfulness" in adaptive curvature estimation.

In Section 3.4, we will return to these proof-of-concept optimization instances, connecting them to an empirical study of curvature in more realistic landscapes.

We investigated the training dynamics of GGT on a typical deep architecture for computer vision.

For this, we used a 26-layer 3-branch residual network with Shake-Shake regularization, recently proposed in BID16 .

Aside from its ability to reach state-of-the-art classification accuracy, this architecture also features a relatively low parameter count (∼ 3M), enabling the use of a large window parameter (r = 200).In each experiment, we kept the cosine learning rate annealing schedule proposed in the paper, originally from BID29 ; performance degraded consistently and significantly with a fixed learning rate.

For both Adam and GGT, we chose the commonly used parameters β 1 = 0.9, β 2 = 0.999, ε = 10 −8 ; for SGD, we used momentum with parameter 0.9.

With correctly tuned RMSprop and Adadelta, with the same window parameters, training curves were virtually identical to those for Adam.

We used the standard data augmentation techniques of 4-pixel padding + random cropping and horizontal flipping.

Our results are shown in FIG2 .

In terms of training loss, GGT consistently dominated existing optimizers.

We corroborate a number of observations from previous empirical studies of the generalization of optimizers.

Most prominently, we found that SGD generalized slightly better than all others BID48 towards the end of training, including ours.

The gap (< 0.2%) is less dramatic than that seen in (Wilson et al., 2017) for two reasons: we only show curves with a tuned and annealed learning rate; also, we use an architecture with powerful explicit regularization techniques which have gained attention since their publication.

Our preliminary observation is that GGT shrinks this gap slightly (corroborated by another experiment in Appendix B), and expect that there is vastly more empirical work to be done concerning architectures synergistically tuned to existing optimizers.

We also verify the long-held empirical observation that the learning rate decay of AdaGrad is too aggressive (e.g. in (Zeiler, 2012)), resulting in convergence to a poor solution.

Finally, as noted in BID48 , we find that using a sufficiently low learning rate for any optimizer can result in a better training loss curve, but not without significantly degrading generalization (> 3% worse).

Next, we move to recurrent architectures for language modeling.

We train a 3-layer LSTM (Hochreiter & Schmidhuber, 1997) with ∼ 5M parameters for character-level modeling of the Penn Treebank dataset BID31 .

This is the setting in which we observe the most striking improvement over baselines.

The particularities of this optimization task, and why it might be especially amenable to full-matrix regularization, remain a fruitful research direction BID38 .

FIG2 (bottom) shows training and validation perplexities for the first 50 epochs; no optimizer makes significant progress afterwards.

The state of the art for character-level language modeling is less thoroughly documented than its word-level counterpart, though we note that our end-to-end result (validation perplexity 2.42 after 500 epochs) is competitive with those reported for recurrent models, like by BID24 .

In contrast, Adam, AdaGrad, and SGD reach 2.51, 2.65, and 2.76, respectively.

Note that Adam is the de facto standard optimizer for language modeling BID35 .

Even with iterations taking twice the time, we outperform all baselines in wall-clock time throughout training.

We also tried using GGT as a drop-in replacement for Adam in the state-of-the-art word-level language modeling code accompanying BID36 .

Although we were competitive with Adam, we only observed an improvement in the first ∼ 20 epochs.

We hypothesize that the advantage of full-matrix regularization in this setting is more marginal, as the gradients in the embedding layers are naturally sparse in the vocabulary ("one-hot") basis.

On a similar note, we found that Adam outperformed GGT on attention-based architectures for NLP; refer to Appendix B for an experiment and discussion.

In this section, we unify the insights gleaned from the synthetic experiments and deep learning benchmarks.

Along the way, we provide some interesting anecdotal observations on the evolution of the preconditioner matrices' singular values.

We plot the density of the spectrum of the low-rank preconditioner G t G t as training progresses.

Since the fast implementation of GGT takes an eigendecomposition of G t G t , we can read off the distribution of eigenvalues during training at no additional computational cost.

FIG3 visualizes the result of this experiment for the CNN and RNN training settings from the previous two sections.

In each case, we observe that G t G t has a condition number of ∼ 10 3 , noting that this can be visualized as the vertical range in the logarithmic plot.

This visualization affords a new way to see how CNN and RNN landscapes are fundamentally different: their gradient spectra evolve in very distinct ways over the course of training.

Interestingly, the condition number of the CNN landscape surges near the end, which may be related to the the low-rank structure of well-trained nets noted by BID5 , who derive rank-dependent generalization bounds for neural networks.

On recurrent models, the rapidly evolving spectral structure at the early stage of training indicates a possibly more complex landscape.

Intriguingly, the enormous condition number (∼ 10 6 ) correlates with the massive lead of GGT over the others, confirming our intuition that full-matrix preconditioning ameliorates anisotropy.

To our knowledge, this is the first empirical study of this kind, using the covariance matrix of recent gradients as a surrogate to examining the changing curvature of the loss landscape.

In the spirit of recent empirical lenses of this flavor BID41 BID26 , we leave this as a way to visualize deep learning dynamics, possibly of independent exploratory interest.

In this section we outline our analysis of GGT, for which we show convergence to an approximate first-order critical point, in some settings faster than SGD.

To obtain the strongest theory, we analyze GGT with a "hard window" instead of exponentially decaying gradient memory, explained in Section A.2.We work in the usual theoretical framework of stochastic optimization of a differentiable non-convex function f (·), equipped with an unbiased variance-bounded stochastic gradient oracle ∇f (·).

The objective, as is standard in the literature (see, e.g. BID17 ; BID4 ), is to find an ε-approximate stationary point x; that is, ∇f (x) ≤ ε.

We quantify the improvement of adaptive regularization by its advantage over the usual worst-case bound of SGD.

To this end, we define the adaptive ratio µ of an algorithm A as DISPLAYFORM0 , where x A is the output of the A, and x * is a comparator.

For convex optimization problems x * is naturally the global minimum.

For non-convex optimization it is a subtler choice, which we detail in Appendix A.This ratio for the AdaGrad algorithm was shown in BID13 to be always bounded by a quantity independent of T , and potentially much smaller.

Specifically, it was shown to be inversely proportional to the dimension in certain convex optimization problems, providing a theoretical justification for the speedup of adaptive optimizers.

In Section A.4, we show a new, simple, and natural setting illustrating adaptive speedup, even for a strongly convex function f .

We informally state the main theorem below.

We defer the full bound without suppressed smoothness constants, as well as all technical proofs, to Appendix A. Theorem 4.1.

Let f : R d → R be a bounded, Lipschitz, and smooth function with stochastic gradient oracle ∇f (·), whose variance is at most σ 2 .

In expectation, Algorithm 3 outputs an ε- DISPLAYFORM0 This theorem matches and potentially improves the known analysis for stochastic gradient descent with the introduction of the data-dependent adaptivity constant µ into the leading-order term governing the rate of convergence.

Since BID13 bounded µ by a quantity independent of T , our theorem matches the classic O ε −4 rate of convergence.

This work investigates full-matrix adaptive regularization: our main contribution is to make this technique viable for large-scale optimization, by a method for efficient multiplication by the inverse square root of a full second-moment matrix over a short window of gradients.

This leads to a new algorithm, GGT, a truly scalable optimization algorithm with full-matrix adaptive preconditioning.

Through synthetic experiments, we have shown that GGT accelerates optimization in ill-conditioned loss landscapes; this is supported by accompanying adaptive convergence guarantees.

Preliminary experiments show accelerated convergence on standard deep learning benchmarks, with very different training dynamics from existing diagonal adaptive methods.

We accompany our algorithm and experiments with the first theoretical characterization of the benefits of adaptive regularization in a non-convex setting.

We hope that GGT will be the first of a new class of algorithms for the modern large-scale optimization toolbox, and to foster new discussion towards an ever-elusive understanding of loss landscapes in deep learning.

In this section, we give the details on the theoretical treatment of GGT outlined in Section 4.

The overall goal is to develop a theory for adaptive regularization in non-convex stochastic optimization.

After formalizing the setting, we will define a version of GGT that uses a hard gradient memory window.

This will allow us to transfer any insight on the advantage of adaptivity in the convex case to the non-convex case, giving rise to the main theorem.

We will conclude this section by with an example illustrating the advantage of adaptive optimizers in the presence of sparse gradients.

A.1 SETTING: STOCHASTIC NON-CONVEX OPTIMIZATION Theorem A.2 will provide a bound on the number of stochastic gradient calls required by GGT to achieve a first-order critical point.

In particular, the theorem shows that GGT can converge to an approximate first-order critical point faster than SGD, with convergence rate controlled by the adaptive ratio µ, defined in (1).We consider the standard setting of stochastic optimization of a differentiable non-convex function f (·), equipped with a bounded-variance stochastic gradient oracle defined as follows.

Definition A.1 (stochastic gradient oracle).

Given a function f : D → R we call an oracle O f , a σ-bounded stochastic gradient oracle if for any x, O f returns a a random vector ∇f (x) such that DISPLAYFORM0 The objective, as is standard in non-convex optimization, is to find a first-order critical point, i.e. a point x for which ∇f (x) ≤ ε.

We will also assume that f has a Lipschitz gradient; i.e. DISPLAYFORM1 Our algorithm makes a reduction to the case of stochastic convex optimization.

The setting formally is that, given a smooth convex function and a σ-bounded stochastic gradient oracle, the algorithm's aim is to minimize the convex function f .

Given any algorithm A we can now define the adaptive ratio of the algorithm, referred to as µ, as DISPLAYFORM2 where x A is the output of the algorithm A and x * ∈ argmin x f (x), with a total of at most T calls to the stochastic gradient oracle.

µ captures the advantage in convergence rate obtained by the algorithm as compared to the error obtained by vanilla SGD, noting that the denominator is a bound on the error obtained by SGD in the same setting.

A popular algorithm for stochastic (and in general online) convex optimization is AdaGrad BID13 .

Due to adaptive regularization, AdaGrad can often be advantageous over SGD.

We quantify this advantage by the notion of µ defined above.

The bounds of BID13 imply that µ can be as small as DISPLAYFORM3 , depending on the geometry of the optimization problem.

An example of this was provided by BID13 for both the diagonal and the full version of Adagrad.

At the end of this section, we provide a different example which shows the same phenomenon even in the case of strongly convex functions.

In the rest of this section we describe Algorithm 3, which uses AdaGrad (Algorithm 2) as a subroutine during each window.

In this regard, while stating the bounds for our algorithms, we use µ as an upper bound on the advantage of AdaGrad in each iteration.

As mentioned in Section 4, our analysis uses a slightly idealized version of GGT, which replaces the gradient memory mechanism (governed by w and β 2 ) with a hard window; i.e., the gradient buffer is reset every w steps.

This simple modification enables us to develop a more informative theory, in which we benefit directly from the familiar theory of AdaGrad for convex optimization, while capturing the necessity of forgetting past gradient information in adaptive non-convex optimization.

First, for clarity, we restate the definition of the full-matrix AdaGrad algorithm, introduced by BID13 , which accumulates the second-moment matrix of all past gradients:Algorithm 2 AdaGrad for convex optimization BID13 1: Input: initializer x 1 , window length w, stochastic gradient oracle ∇f (·), ε, η > 0.

2: for t = 1, . . . , w do 3:Receive stochastic gradient ∇f (x t ).

Let G t = [g t g t−1 . . .

g 1 ], where g t := ∇f (x t ).

Update x t+1 ← x t − η · εI + (G t G t ) 1/2 −1 g t .

6: end for 7: Output: Average iterate DISPLAYFORM0 The final algorithm we analyze simply runs AdaGrad between restarts.

Algorithm 3 GGT with a hard gradient window 1: Input: initializer x 1 , time horizon T , window length w, λ > 0.

2: for t = 1 to T : do DISPLAYFORM1 Update x t+1 to be the output of Algorithm 2 on f t (x), starting at x t , for w steps.

5: end for 6: Output: Best iterate x t * , where t * := argmin t≤T +1 ∇f (x t ) .The remaining discrepancies between Algorithm 3 and Algorithm 1 from the main paper are standard.

We provide some references below.• Absence of first-moment estimation.

Although it is customary to use nonzero β 1 (otherwise known as momentum) when applying Adam in practice, it is orthogonal to the effect of adaptive regularization in all established theory.

In fact, the convergence rates given by Kingma & Ba (2014) (and fixed by BID42 ) contain only factors of 1/(1 − β 1 ), and are thus strongest when β 1 = 0.• Model averaging.

Theoretical guarantees in online and stochastic convex optimization are most naturally stated on the average iterate; see BID40 BID13 .

Thus, we adopt the convention that Algorithm 2 returns the average iterate.

We note that model averaging is a common regularization technique in practical non-convex settings, though not the default choice for adaptive optimizers in practice.• 2 regularization.

The addition of the λ x − x t 2 term in Algorithm 3 is an artifact we introduce to obtain a tight analysis for hard-window GGT.

It ensures that iterates in each window do not move too far, and allows us to analyze each window as a fixed convex program, so that we can use the convex theory of AdaGrad directly.

The soft-window analogue would simply to be decrease the learning rate.

Interestingly, a similar technique directly appears in the algorithm proposed by BID3 .

Finally, we note that from a σ-bounded stochastic gradient oracle for f , it is trivial to construct one for f t , by adding −2λx t (deterministically).

Theorem A.2.

Consider a non-convex function f , such that for all x, ∇ 2 f (x) 2 ≤ L and a point DISPLAYFORM0 Further, suppose we have access to a σ-bounded DISPLAYFORM1 .

Then the point x returned by Algorithm 3 is such that E ∇f (x ) ≤ ε, where µ = max t∈[T ] µ t and µ t is the adaptive ratio when run on f t (as defined in FORMULA9 ).

Further, note that choosing λ = 3L/2, the total number of stochastic gradient calls to the oracle O f , made by the algorithm is bounded by T · w = FIG2 .

Top: CIFAR-10 classification with a 3-branch ResNet.

Bottom: PTB character-level language modeling with a 3-layer LSTM.

We present some additional large-scale empirical studies in FIG5 .

To demonstrate a vision task with a harder optimization landscape, we use GGT to train a 19-layer "vanilla" convolutional network (VGGNet, BID44 ), without residual connections or batch normalization, on the same CIFAR-10 classification task.

Here, we recover the same insights as found by BID48 , in which diagonal-matrix adaptive methods can fail to train a network dramatically.

Here, unlike diagonal-matrix adaptive optimizers, GGT stays on par with SGD throughout training, with a ∼ 1% gap remaining in generalization at the end.

We use a standard fixed halving learning rate schedule; it is clear here that in the initial epochs after decaying the learning rate, GGT trains the most rapidly.

We leave a careful investigation of leveraging this phenomenon, and tuning GGT's learning rate schedule, to future work.

A recent significant advancement on many NLP tasks, including language modeling, is the introduction of attention-based models.

We investigate the behavior of GGT on a Transformer network BID46 , on the same Penn Treebank character-level language modeling task.

Here, after an initial lead, GGT is outperformed by Adam in training and validation loss.

The value of using gradient correlations to assist in the training of attention models seems to be limited.

<|TLDR|>

@highlight

fast, truly scalable full-matrix AdaGrad/Adam, with theory for adaptive stochastic non-convex optimization
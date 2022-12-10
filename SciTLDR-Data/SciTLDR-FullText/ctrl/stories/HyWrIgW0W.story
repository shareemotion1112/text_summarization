Stochastic gradient descent (SGD) is widely believed to perform implicit regularization when used to train deep neural networks, but the precise manner in which this occurs has thus far been elusive.

We prove that SGD minimizes an average potential over the posterior distribution of weights along with an entropic regularization term.

This potential is however not the original loss function in general.

So SGD does perform variational inference, but for a different loss than the one used to compute the gradients.

Even more surprisingly, SGD does not even converge in the classical sense: we show that the most likely trajectories of SGD for deep networks do not behave like Brownian motion around critical points.

Instead, they resemble closed loops with deterministic components.

We prove that such out-of-equilibrium behavior is a consequence of highly non-isotropic gradient noise in SGD; the covariance matrix of mini-batch gradients for deep networks has a rank as small as 1% of its dimension.

We provide extensive empirical validation of these claims, proven in the appendix.

Our first result is to show precisely in what sense stochastic gradient descent (SGD) implicitly performs variational inference, as is often claimed informally in the literature.

For a loss function f (x) with weights x ∈ R d , if ρ ss is the steady-state distribution over the weights estimated by SGD, DISPLAYFORM0 where H(ρ) is the entropy of the distribution ρ and η and b are the learning rate and batch-size, respectively.

The potential Φ(x), which we characterize explicitly, is related but not necessarily equal to f (x).

It is only a function of the architecture and the dataset.

This implies that SGD implicitly performs variational inference with a uniform prior, albeit of a different loss than the one used to compute back-propagation gradients.

We next prove that the implicit potential Φ(x) is equal to our chosen loss f (x) if and only if the noise in mini-batch gradients is isotropic.

This condition, however, is not satisfied for deep networks.

Empirically, we find gradient noise to be highly non-isotropic with the rank of its covariance matrix being about 1% of its dimension.

Thus, SGD on deep networks implicitly discovers locations where ∇Φ(x) = 0, these are not the locations where ∇ f (x) = 0.

This is our second main result: the most likely locations of SGD are not the local minima, nor the saddle points, of the original loss.

The deviation of these critical points, which we compute explicitly scales linearly with η/b and is typically large in practice.

When mini-batch noise is non-isotropic, SGD does not even converge in the classical sense.

We prove that, instead of undergoing Brownian motion in the vicinity of a critical point, trajectories have a deterministic component that causes SGD to traverse closed loops in the weight space.

We detect such loops using a Fourier analysis of SGD trajectories.

We also show through an example that SGD with non-isotropic noise can even converge to stable limit cycles around saddle points.

Stochastic gradient descent performs the following updates while training a network x k+1 = x k − η ∇ f b (x k ) where η is the learning rate and ∇ f b (x k ) is the average gradient over a mini-batch b, DISPLAYFORM0 We overload notation b for both the set of examples in a mini-batch and its size.

We assume that weights belong to a compact subset Ω ⊂ R d , to ensure appropriate boundary conditions for the evolution of steady-state densities in SGD, although all our results hold without this assumption if the loss grows unbounded as x → ∞, for instance, with weight decay as a regularizer.

.

If a mini-batch is sampled with replacement, we show in Appendix A.1 that the variance of mini-batch gradients is var (∇ f b (x)) =

b where DISPLAYFORM0 Note that D(x) is independent of the learning rate η and the batch-size b.

It only depends on the weights x, architecture and loss defined by f (x), and the dataset.

We will often discuss two cases: isotropic diffusion when D(x) is a scalar multiple of identity, independent of x, and non-isotropic diffusion, when D(x) is a general function of the weights x.

We now construct a stochastic differential equation (SDE) for the discrete-time SGD updates.

Lemma 2 (Continuous-time SGD).

The continuous-time limit of SGD is given by DISPLAYFORM1 where W (t) is Brownian motion and β is the inverse temperature defined as β −1 = η 2b .

The steadystate distribution of the weights ρ(z,t) ∝ P x(t) = z , evolves according to the Fokker-Planck equation BID49 , Ito form): DISPLAYFORM2 where the notation ∇ · v denotes the divergence ∇ · v = ∑ i ∂ x i v i (x) for any vector v(x) ∈ R d ; the divergence operator is applied column-wise to matrices such as D(x).We refer to Li et al. (2017b, Thm. 1) for the proof of the convergence of discrete SGD to (3).

Note that β −1 completely captures the magnitude of noise in SGD that depends only upon the learning rate η and the mini-batch size b.

Assumption 3 (Steady-state distribution exists and is unique).

We assume that the steady-state distribution of the Fokker-Planck equation (FP) exists and is unique, this is denoted by ρ ss (x) and satisfies, DISPLAYFORM3

Let us first implicitly define a potential Φ(x) using the steady-state distribution ρ ss : DISPLAYFORM0 up to a constant.

The potential Φ(x) depends only on the full-gradient and the diffusion matrix; see Appendix C for a proof.

It will be made explicit in Section 5.

We express ρ ss in terms of the potential using a normalizing constant Z(β ) as DISPLAYFORM1 which is also the steady-state solution of DISPLAYFORM2 as can be verified by direct substitution in (FP).The above observation is very useful because it suggests that, if ∇ f (x) can be written in terms of the diffusion matrix and a gradient term ∇Φ(x), the steady-state distribution of this SDE is easily obtained.

We exploit this observation to rewrite ∇ f (x) in terms a term D ∇Φ that gives rise to the above steady-state, the spatial derivative of the diffusion matrix, and the remainder: DISPLAYFORM3 interpreted as the part of ∇ f (x) that cannot be written as D Φ (x) for some Φ .

We now make an important assumption on j(x) which has its origins in thermodynamics.

Assumption 4 (Force j(x) is conservative).

We assume that DISPLAYFORM4 The Fokker-Planck equation (FP) typically models a physical system which exchanges energy with an external environment BID43 BID47 .

In our case, this physical system is the gradient dynamics ∇ · (∇ f ρ) while the interaction with the environment is through the term involving temperature: β −1 ∇ · (∇ · (Dρ)).

The second law of thermodynamics states that the entropy of a system can never decrease; in Appendix B we show how the above assumption is sufficient to satisfy the second law.

We also discuss some properties of j(x) in Appendix C that are a consequence of this.

The most important is that j(x) is always orthogonal to ∇ρ ss .

We illustrate the effects of this assumption in Example 19.This leads us to the main result of this section.

Theorem 5 (SGD performs variational inference).

The functional DISPLAYFORM5 decreases monotonically along the trajectories of the Fokker-Planck equation (FP) and converges to its minimum, which is zero, at steady-state.

Moreover, we also have an energetic-entropic split DISPLAYFORM6 Theorem 5, proven in Appendix F.1, shows that SGD implicitly minimizes a combination of two terms: an "energetic" term, and an "entropic" term.

The first is the average potential over a distribution ρ.

The steady-state of SGD in (6) is such that it places most of its probability mass in regions of the parameter space with small values of Φ. The second shows that SGD has an implicit bias towards solutions that maximize the entropy of the distribution ρ.

Note that the energetic term in (11) has potential Φ(x), instead of f (x).

This is an important fact and the crux of this paper.

Lemma 6 (Potential equals original loss iff isotropic diffusion).

If the diffusion matrix D(x) is isotropic, i.e., a constant multiple of the identity, the implicit potential is the original loss itself DISPLAYFORM7 This is proven in Appendix F.2.

The definition in (8) shows that j = 0 when D(x) is non-isotropic.

This results in a deterministic component in the SGD dynamics which does not affect the functional F(ρ), hence j(x) is called a "conservative force."

The following lemma is proven in Appendix F.3.Lemma 7 (Most likely trajectories of SGD are limit cycles).

The force j(x) does not decrease F(ρ) in (11) and introduces a deterministic component in SGD given bẏ DISPLAYFORM8 The condition ∇ · j(x) = 0 in Assumption 4 implies that most likely trajectories of SGD traverse closed trajectories in weight space.

Theorem 5 applies for a general D(x) and it is equivalent to the celebrated JKO functional BID22 in optimal transportation BID50 BID58 if the diffusion matrix is isotropic.

Appendix D provides a brief overview using the heat equation as an example.

DISPLAYFORM0 Observe that the energetic term contains f (x) in Corollary 8.

The proof follows from Theorem 5 and Lemma 6, see BID51 for a rigorous treatment of Wasserstein metrics.

The JKO functional above has had an enormous impact in optimal transport because results like Theorem 5 and Corollary 8 provide a way to modify the functional F(ρ) in an interpretable fashion.

Modifying the Fokker-Planck equation or the SGD updates directly to enforce regularization properties on the solutions ρ ss is much harder.

Note the absence of any prior in (11).

On the other hand, the evidence lower bound BID27 for the dataset Ξ is, DISPLAYFORM0 where H(q, p) is the cross-entropy of the estimated steady-state and the variational prior.

The implicit loss function of SGD in (11) therefore corresponds to a uniform prior p(x | Ξ).

In other words, we have shown that SGD itself performs variational optimization with a uniform prior.

Note that this prior is well-defined by our hypothesis of x ∈ Ω for some compact Ω.It is important to note that SGD implicitly minimizes a potential Φ(x) instead of the original loss f (x) in ELBO.

We prove in Section 5 that this potential is quite different from f (x) if the diffusion matrix D is non-isotropic, in particular, with respect to its critical points.

Remark 9 (SGD has an information bottleneck).

The functional FORMULA1 is equivalent to the information bottleneck principle in representation learning BID57 .

Minimizing this functional, explicitly, has been shown to lead to invariant representations BID0 .

Theorem 5 shows that SGD implicitly contains this bottleneck and therefore begets these properties, naturally.

Remark 10 (ELBO prior conflicts with SGD).

Working with ELBO in practice involves one or multiple steps of SGD to minimize the energetic term along with an estimate of the KL-divergence term, often using a factored Gaussian prior BID27 BID21 .

As Theorem 5 shows, such an approach also enforces a uniform prior whose strength is determined by β −1 and conflicts with the externally imposed Gaussian prior.

This conflict-which fundamentally arises from using SGD to minimize the energetic term-has resulted in researchers artificially modulating the strength of the KL-divergence term using a scalar pre-factor BID35 .

We will show in Section 5 that the potential Φ(x) does not depend on the optimization process, it is only a function of the dataset and the architecture.

The effect of two important parameters, the learning rate η and the mini-batch size b therefore completely determines the strength of the entropic regularization term.

If β −1 → 0, the implicit regularization of SGD goes to zero.

This implies that DISPLAYFORM0 should not be small is a good tenet for regularization of SGD.Remark 11 (Learning rate should scale linearly with batch-size to generalize well).

In order to maintain the entropic regularization, the learning rate η needs to scale linearly with the batch-size b. This prediction, based on Theorem 5, fits very well with empirical evidence wherein one obtains good generalization performance only with small mini-batches in deep networks BID25 , or via such linear scaling BID16 .Remark 12 (Sampling with replacement is better than without replacement).

The diffusion matrix for the case when mini-batches are sampled with replacement is very close to (2), see Appendix A.2.

However, the corresponding inverse temperature is DISPLAYFORM1 The extra factor of 1 − b N reduces the entropic regularization in (11), as b → N, the inverse temperature β → ∞. As a consequence, for the same learning rate η and batch-size b, Theorem 5 predicts that sampling with replacement has better regularization than sampling without replacement.

This effect is particularly pronounced at large batch-sizes.

Section 4.1 shows that the diffusion matrix D(x) for modern deep networks is highly non-isotropic with a very low rank.

We also analyze trajectories of SGD and detect periodic components using a frequency analysis in Section 4.2; this validates the prediction of Lemma 7.We consider three networks for these experiments: a convolutional network called small-lenet, a twolayer fully-connected network on MNIST BID30 ) and a smaller version of the All-CNN-C architecture of BID54 on the CIFAR-10 and CIFAR-100 datasets BID28 ); see Appendix E for more details.

Figs. 1 and 2 show the eigenspectrum 1 of the diffusion matrix.

In all cases, it has a large fraction of almost-zero eigenvalues with a very small rank that ranges between 0.3% -2%.

Moreover, non-zero eigenvalues are spread across a vast range with a large variance.

Remark 13 (Noise in SGD is largely independent of the weights).

The variance of noise in FORMULA3 is We have plotted the eigenspectra of the diffusion matrix in FIG1 and FIG2 at three different instants, 20%, 40% and 100% training completion; they are almost indistinguishable.

This implies that the variance of the mini-batch gradients in deep networks can be considered a constant, highly non-isotropic matrix.

Remark 14 (More non-isotropic diffusion if data is diverse).

The eigenspectra in FIG2 for CIFAR-10 and CIFAR-100 have much larger eigenvalues and standard-deviation than those in FIG1 , this is expected because the images in the CIFAR datasets have more variety than those in MNIST.

Similarly, while CIFAR-100 has qualitatively similar images as CIFAR-10, it has 10× more classes and as a result, it is a much harder dataset.

This correlates well with the fact that both the mean and standard-deviation of the eigenvalues in FIG2 are much higher than those in FIG2 .

Input augmentation increases the diversity of mini-batch gradients.

This is seen in FIG2 where the standard-deviation of the eigenvalues is much higher as compared to FIG2 . .

The eigenvalues are much larger in magnitude here than those of MNIST in FIG1 , this suggests a larger gradient diversity for CIFAR-10 and CIFAR-100.

The diffusion matrix for CIFAR-100 in FIG2 has larger eigenvalues and is more non-isotropic and has a much larger rank than that of FIG2 ; this suggests that gradient diversity increases with the number of classes.

As FIG2 and FIG2 show, augmenting input data increases both the mean and the variance of the eigenvalues while keeping the rank almost constant.

DISPLAYFORM0 Remark 15 (Inverse temperature scales with the mean of the eigenspectrum).

Remark 14 shows that the mean of the eigenspectrum is large if the dataset is diverse.

Based on this, we propose that the inverse temperature β should scale linearly with the mean of the eigenvalues of D: DISPLAYFORM1 where d is the number of weights.

This keeps the noise in SGD constant in magnitude for different values of the learning rate η, mini-batch size b, architectures, and datasets.

Note that other hyperparameters which affect stochasticity such as dropout probability are implicit inside D.Remark 16 (Variance of the eigenspectrum informs architecture search).

Compare the eigenspectra in Figs. 1a and 1b with those in FIG2 .

The former pair shows that small-lenet which is a much better network than small-fc also has a much larger rank, i.e., the number of non-zero eigenvalues (D(x) is symmetric).

The second pair shows that for the same dataset, data-augmentation creates a larger variance in the eigenspectrum.

This suggests that both the quantities, viz., rank of the diffusion matrix and the variance of the eigenspectrum, inform the performance of a given architecture on the dataset.

Note that as discussed in Remark 15, the mean of the eigenvalues can be controlled using the learning rate η and the batch-size b.

This observation is useful for automated architecture search where we can use the quantity DISPLAYFORM2 to estimate the efficacy of a given architecture, possibly, without even training, since D does not depend on the weights much.

This task currently requires enormous amounts of computational power BID66 BID3 BID6 .

k+1 − x i k where k is the number of epochs and i denotes the index of the weight.

FIG3 shows the auto-correlation of x i k with 99% confidence bands denoted by the dotted red lines.

Both Figs. 3a and 3b show the mean and one standard-deviation over the weight index i; the standard deviation is very small which indicates that all the weights have a very similar frequency spectrum.

Figs.

3a and 3b should be compared with the FFT of white noise which should be flat and the auto-correlation of Brownian motion which quickly decays to zero, respectively.

Figs. 3 and 3a therefore show that trajectories of SGD are not simply Brownian motion.

Moreover the gradient at these locations is quite large FIG3 ).

We train a smaller version of small-fc on 7 × 7 down-sampled MNIST images for 10 5 epochs and store snapshots of the weights after each epoch to get a long trajectory in the weight space.

We discard the first 10 3 epochs of training ("burnin") to ensure that SGD has reached the steady-state.

The learning rate is fixed to 10 −3 after this, up to 10 5 epochs.

Remark 17 (Low-frequency periodic components in SGD trajectories).

Iterates of SGD, after it reaches the neighborhood of a critical point ∇ f (x k ) ≤ ε, are expected to perform Brownian motion with variance var (∇ f b (x)), the FFT in FIG3 would be flat if this were so.

Instead, we see low-frequency modes in the trajectory that are indicators of a periodic dynamics of the force j(x).

These modes are not sharp peaks in the FFT because j(x) can be a non-linear function of the weights thereby causing the modes to spread into all dimensions of x. The FFT is dominated by jittery high-frequency modes on the right with a slight increasing trend; this suggests the presence of colored noise in SGD at high-frequencies.

The auto-correlation (AC) in FIG3 should be compared with the AC for Brownian motion which decays to zero very quickly and stays within the red confidence bands (99%).

Our iterates are significantly correlated with each other even at very large lags.

This further indicates that trajectories of SGD do not perform Brownian motion.

Remark 18 (Gradient magnitude in deep networks is always large).

FIG3 shows that the fullgradient computed over the entire dataset (without burnin) does not decrease much with respect to the number of epochs.

While it is expected to have a non-zero gradient norm because SGD only converges to a neighborhood of a critical point for non-zero learning rates, the magnitude of this gradient norm is quite large.

This magnitude drops only by about a factor of 3 over the next 10 5 epochs.

The presence of a non-zero j(x) also explains this, it causes SGD to be away from critical points, this phenomenon is made precise in Theorem 22.

Let us note that a similar plot is also seen in Shwartz-Ziv and Tishby (2017) for the per-layer gradient magnitude.

This section now gives an explicit formula for the potential Φ(x).

We also discuss implications of this for generalization in Section 5.3.The fundamental difficulty in obtaining an explicit expression for Φ is that even if the diffusion matrix DISPLAYFORM0 We therefore split the analysis into two cases:(i) a local analysis near any critical point ∇ f (x) = 0 where we linearize ∇ f (x) = Fx and ∇Φ(x) = Ux to compute U = G −1 F for some G, and (ii) the general case where ∇Φ(x) cannot be written as a local rotation and scaling of ∇ f (x).Let us introduce these cases with an example from BID39 .

DISPLAYFORM1 Figure 4: Gradient field for the dynamics in Example 19: line-width is proportional to the magnitude of the gradient ∇ f (x) , red dots denote the most likely locations of the steady-state e −Φ while the potential Φ is plotted as a contour map.

The critical points of f (x) and Φ(x) are the same in Fig. 4a , namely (±1, 0), because the force j(x) = 0.

For λ = 0.5 in Fig. 4b , locations where ∇ f (x) = 0 have shifted slightly as predicted by Theorem 22.

The force field also has a distinctive rotation component, see Remark 21.

In Fig. 4c with a large j(x) , SGD converges to limit cycles around the saddle point at the origin.

This is highly surprising and demonstrates that the solutions obtained by SGD may be very different from local minima.

Example 19 (Double-well potential with limit cycles).

Define DISPLAYFORM2 Instead of constructing a diffusion matrix D(x), we will directly construct different gradients ∇ f (x) that lead to the same potential Φ; these are equivalent but the later is much easier.

The dynamics is DISPLAYFORM3 .

We pick j = λ e Φ J ss (x) for some parameter λ > 0 where DISPLAYFORM4 Note that this satisfies (6) and does not change ρ ss = e −Φ .

Fig. 4 shows the gradient field f (x) along with a discussion.

Without loss of generality, let x = 0 be a critical point of f (x).

This critical point can be a local minimum, maximum, or even a saddle point.

We linearize the gradient around the origin and define a fixed matrix F ∈ R d×d (the Hessian) to be ∇ f (x) = Fx.

Let D = D(0) be the constant diffusion matrix matrix.

The dynamics in (3) can now be written as DISPLAYFORM0 Lemma 20 (Linearization).

The matrix F in (15) can be uniquely decomposed into DISPLAYFORM1 D and Q are the symmetric and anti-symmetric parts of a matrix G with GF − FG = 0, to get DISPLAYFORM2 The above lemma is a classical result if the critical point is a local minimum, i.e., if the loss is locally convex near x = 0; this case has also been explored in machine learning before BID35 .

We refer to BID29 for the proof that linearizes around any critical point.

We see from Lemma 20 that, near a critical point, DISPLAYFORM0 up to the first order.

This suggests that the effect of j(x) is to rotate the gradient field and move the critical points, also seen in Fig. 4b .

Note that ∇ · D = 0 and ∇ · Q = 0 in the linearized analysis.

We next give the general expression for the deviation of the critical points ∇Φ from those of the original loss ∇ f .A-type stochastic integration: A Fokker-Planck equation is a deterministic partial differential equation (PDE) and every steady-state distribution, ρ ss ∝ e −β Φ in this case, has a unique such PDE that achieves it.

However, the same PDE can be tied to different SDEs depending on the stochastic integration scheme, e.g., Ito, Stratonovich BID49 BID40 , Hanggi BID18 , α-type etc.

An "A-type" interpretation is one such scheme BID2 BID52 .

It is widely used in non-equilibrium studies in physics and biology BID59 BID65 because it allows one to compute the steady-state distribution easily; its implications are supported by other mathematical analyses such as BID56 ; BID47 .The main result of the section now follows.

It exploits the A-type interpretation to compute the difference between the most likely locations of SGD which are given by the critical points of the potential Φ(x) and those of the original loss f (x).Theorem 22 (Most likely locations are not the critical points of the loss).

The Ito SDE DISPLAYFORM0 is equivalent to the A-type SDE BID2 BID52 DISPLAYFORM1 with the same steady-state distribution ρ ss ∝ e −β Φ(x) and Fokker-Planck equation (FP) if DISPLAYFORM2 The anti-symmetric matrix Q(x) and the potential Φ(x) can be explicitly computed in terms of the gradient ∇ f (x) and the diffusion matrix D(x).

The potential Φ(x) does not depend on β .See Appendix F.4 for the proof.

It exploits the fact that the the Ito SDE (3) and the A-type SDE FORMULA1 should have the same Fokker-Planck equations because they have the same steady-state distributions.

Remark 23 (SGD is far away from critical points).

The time spent by a Markov chain at a state x is proportional to its steady-state distribution ρ ss (x).

While it is easily seen that SGD does not converge in the Cauchy sense due to the stochasticity, it is very surprising that it may spend a significant amount of time away from the critical points of the original loss.

If D(x) + Q(x) has a large divergence, the set of states with ∇Φ(x) = 0 might be drastically different than those with ∇ f (x) = 0.

This is also seen in example Fig. 4c ; in fact, SGD may even converge around a saddle point.

This also closes the logical loop we began in Section 3 where we assumed the existence of ρ ss and defined the potential Φ using it.

Lemma 20 and Theorem 22 show that both can be defined uniquely in terms of the original quantities, i.e., the gradient term ∇ f (x) and the diffusion matrix D(x).

There is no ambiguity as to whether the potential Φ(x) results in the steady-state ρ ss (x) or vice-versa.

Remark 24 (Consistent with the linear case).

Theorem 22 presents a picture that is completely consistent with Lemma 20.

If j(x) = 0 and Q(x) = 0, or if Q is a constant like the linear case in Lemma 20, the divergence of Q(x) in FORMULA1 is zero.

Remark 25 (Out-of-equilibrium effect can be large even if D is constant).

The presence of a Q(x) with non-zero divergence is the consequence of a non-isotropic D(x) and it persists even if D is constant and independent of weights x. So long as D is not isotropic, as we discussed in the beginning of Section 5, there need not exist a function Φ(x) such that ∇Φ(x) = D −1 ∇ f (x) at all x. This is also seen in our experiments, the diffusion matrix is almost constant with respect to weights for deep networks, but consequences of out-of-equilibrium behavior are still seen in Section 4.2.Remark 26 (Out-of-equilibrium effect increases with β −1 ).

The effect predicted by (19) becomes more pronounced if β −1 = η 2b is large.

In other words, small batch-sizes or high learning rates cause SGD to be drastically out-of-equilibrium.

Theorem 5 also shows that as β −1 → 0, the implicit entropic regularization in SGD vanishes.

Observe that these are exactly the conditions under which we typically obtain good generalization performance for deep networks BID25 BID16 .

This suggests that non-equilibrium behavior in SGD is crucial to obtain good generalization performance, especially for high-dimensional models such as deep networks where such effects are expected to be more pronounced.

It was found that solutions of discrete learning problems that generalize well belong to dense clusters in the weight space BID5 BID45 .

Such dense clusters are exponentially fewer compared to isolated solutions.

To exploit these observations, the authors proposed a loss called "local entropy" that is out-of-equilibrium by construction and can find these well-generalizable solutions easily.

This idea has also been successful in deep learning where BID8 modified SGD to seek solutions in "wide minima" with low curvature to obtain improvements in generalization performance as well as convergence rate BID7 .Local entropy is a smoothed version of the original loss given by DISPLAYFORM0 where G γ is a Gaussian kernel of variance γ.

Even with an isotropic diffusion matrix, the steady-state distribution with f γ (x) as the loss function is ρ ss γ (x) ∝ e −β f γ (x) .

For large values of γ, the new loss makes the original local minima exponentially less likely.

In other words, local entropy does not rely on non-isotropic gradient noise to obtain out-of-equilibrium behavior, it gets it explicitly, by construction.

This is also seen in Fig. 4c : if SGD is drastically out-of-equilibrium, it converges around the "wide" saddle point region at the origin which has a small local entropy.

Actively constructing out-of-equilibrium behavior leads to good generalization in practice.

Our evidence that SGD on deep networks itself possesses out-of-equilibrium behavior then indicates that SGD for deep networks generalizes well because of such behavior.

SGD, variational inference and implicit regularization: The idea that SGD is related to variational inference has been seen in machine learning before BID13 BID35 under assumptions such as quadratic steady-states; for instance, see BID36 for methods to approximate steady-states using SGD.

Our results here are very different, we would instead like to understand properties of SGD itself.

Indeed, in full generality, SGD performs variational inference using a new potential Φ that it implicitly constructs given an architecture and a dataset.

It is widely believed that SGD is an implicit regularizer, see BID64 ; BID38 ; Shwartz-Ziv and Tishby (2017) among others.

This belief stems from its remarkable empirical performance.

Our results show that such intuition is very well-placed.

Thanks to the special architecture of deep networks where gradient noise is highly non-isotropic, SGD helps itself to a potential Φ with properties that lead to both generalization and acceleration.

Noise is often added in SGD to improve its behavior around saddle points for non-convex losses, see BID31 ; BID1 ; BID15 .

It is also quite indispensable for training deep networks BID19 BID55 BID26 BID17 BID0 .

There is however a disconnect between these two directions due to the fact that while adding external gradient noise helps in theory, it works poorly in practice BID37 BID10 .

Instead, "noise tied to the architecture" works better, e.g., dropout, or small mini-batches.

Our results close this gap and show that SGD crucially leverages the highly degenerate noise induced by the architecture.

Gradient diversity: BID62 construct a scalar measure of the gradient diversity given by DISPLAYFORM0 , and analyze its effect on the maximum allowed batch-size in the context of distributed optimization.

Markov Chain Monte Carlo: MCMC methods that sample from a negative log-likelihood Φ(x) have employed the idea of designing a force j = ∇Φ − ∇ f to accelerate convergence, see BID34 for a thorough survey, or Pavliotis (2016); BID24 for a rigorous treatment.

We instead compute the potential Φ given ∇ f and D, which necessitates the use of techniques from physics.

In fact, our results show that since j = 0 for deep networks due to non-isotropic gradient noise, very simple algorithms such as SGLD by BID60 also benefit from the acceleration that their sophisticated counterparts aim for BID12 BID11 .

The continuous-time point-of-view used in this paper gives access to general principles that govern SGD, such analyses are increasingly becoming popular BID61 BID9 .

However, in practice, deep networks are trained for only a few epochs with discrete-time updates.

Closing this gap is an important future direction.

A promising avenue towards this is that for typical conditions in practice such as small mini-batches or large learning rates, SGD converges to the steady-state distribution quickly BID48 .

PC would like to thank Adam Oberman for introducing him to the JKO functional.

The authors would also like to thank Alhussein Fawzi for numerous discussions during the conception of this paper and his contribution to its improvement.

This research was supported by ARO W911NF-17-1-0304, ONR N00014-17-1-2072, AFOSR FA9550-15-1-0229.

In this section we denote g k := ∇ f k (x) and g := ∇ f (x) = 1 N ∑ N k=1 g k .

Although we drop the dependence of g k on x to keep the notation clear, we emphasize that the diffusion matrix D depends on the weights x.

A.1 WITH REPLACEMENT Let i 1 , . . .

, i b be b iid random variables in {1, 2, . . .

, N}. We would like to compute DISPLAYFORM0 Note that we have that for any j = k, the random vectors g i j and g i k are independent.

We therefore have covar( DISPLAYFORM1 We use this to obtain DISPLAYFORM2 We will set DISPLAYFORM3 and assimilate the factor of b −1 in the inverse temperature β .

Let us define an indicator random variable 1 i∈b that denotes if an example i was sampled in batch b. We can show that DISPLAYFORM0 Similar to BID32 , we can now compute DISPLAYFORM1 We will again set DISPLAYFORM2 and assimilate the factor of b −1 1 − b N that depends on the batch-size in the inverse temperature β .

Let F(ρ) be as defined in (11).

In non-equilibrium thermodynamics, it is assumed that the local entropy production is a product of the force −∇ δ F δ ρ from (A8) and the probability current −J(x,t) from (FP).

This assumption in this form was first introduced by BID46 based on the works of BID41 BID40 .

See Frank (2005, Sec. 4 .5) for a mathematical treatment and Jaynes (1980) for further discussion.

The rate of entropy (S i ) increase is given by DISPLAYFORM0 This can now be written using (A8) again as DISPLAYFORM1 The first term in the above expression is non-negative, in order to ensure that DISPLAYFORM2 where the second equality again follows by integration by parts.

It can be shown (Frank, 2005, Sec. 4.5.5 ) that the condition in Assumption 4, viz., ∇ · j(x) = 0, is sufficient to make the above integral vanish and therefore for the entropy generation to be non-negative.

C SOME PROPERTIES OF THE FORCE jThe Fokker-Planck equation (FP) can be written in terms of the probability current as DISPLAYFORM3 Since we have ρ ss ∝ e −β Φ(x) , from the observation (7), we also have that DISPLAYFORM4 and consequently, DISPLAYFORM5 In other words, the conservative force is non-zero only if detailed balance is broken, i.e., J ss = 0.

We also have DISPLAYFORM6 which shows using Assumption 4 and ρ ss (x) > 0 for all x ∈ Ω that j(x) is always orthogonal to the gradient of the potential DISPLAYFORM7 Using the definition of j(x) in (8), we have detailed balance when DISPLAYFORM8

As first discovered in the works of Jordan, Kinderleherer and Otto BID23 BID44 , certain partial differential equations can be seen as coming from a variational principle, i.e., they perform steepest descent with respect to functionals of their state distribution.

Section 3 is a generalization of this idea, we give a short overview here with the heat equation.

The heat equation DISPLAYFORM0 can be written as the steepest descent for the Dirichlet energy functional DISPLAYFORM1 However, the same PDE can also be seen as the gradient flow of the negative Shannon entropy in the Wasserstein metric BID51 BID50 , DISPLAYFORM2 More precisely, the sequence of iterated minimization problems DISPLAYFORM3 converges to trajectories of the heat equation as τ → 0.

This equivalence is extremely powerful because it allows us to interpret, and modify, the functional −H(ρ) that PDEs such as the heat equation implicitly minimize.

This equivalence is also quite natural, the heat equation describes the probability density of pure Brownian motion: dx = √ 2 dW (t).

The Wasserstein point-of-view suggests that Brownian motion maximizes the entropy of its state distribution, while the Dirichlet functional suggests that it minimizes the total-variation of its density.

These are equivalent.

While the latter has been used extensively in image processing, our paper suggests that the entropic regularization point-of-view is very useful to understand SGD for machine learning.

We consider the following three networks on the MNIST BID30 and the CIFAR-10 and CIFAR-100 datasets BID28 ).(i)

small-lenet: a smaller version of LeNet (LeCun et al., 1998) on MNIST with batchnormalization and dropout (0.1) after both convolutional layers of 8 and 16 output channels, respectively.

The fully-connected layer has 128 hidden units.

This network has 13, 338 weights and reaches about 0.75% training and validation error.(ii) small-fc: a fully-connected network with two-layers, batch-normalization and rectified linear units that takes 7 × 7 down-sampled images of MNIST as input and has 64 hidden units.

Experiments in Section 4.2 use a smaller version of this network with 16 hidden units and 5 output classes (30, 000 input images); this is called tiny-fc.(iii) small-allcnn: this a smaller version of the fully-convolutional network for CIFAR-10 and CIFAR-100 introduced by BID54 with batch-normalization and 12, 24 output channels in the first and second block respectively.

It has 26, 982 weights and reaches about 11% and 17% training and validation errors, respectively.

We train the above networks with SGD with appropriate learning rate annealing and Nesterov's momentum set to 0.9.

We do not use any data-augmentation and pre-process data using global contrast normalization with ZCA for CIFAR-10 and CIFAR-100.We use networks with about 20, 000 weights to keep the eigen-decomposition of D(x) ∈ R d×d tractable.

These networks however possess all the architectural intricacies such as convolutions, dropout, batch-normalization etc.

We evaluate D(x) using (2) with the network in evaluation mode.

The KL-divergence is non-negative: F(ρ) ≥ 0 with equality if and only if ρ = ρ ss .

The expression in (11) follows after writing log ρ ss = −β Φ − log Z(β ).We now show that dF(ρ) dt ≤ 0 with equality only at ρ = ρ ss when F(ρ) reaches its minimum and the Fokker-Planck equation achieves its steady-state.

The first variation BID50 DISPLAYFORM0 which helps us write the Fokker-Planck equation (FP) as DISPLAYFORM1 Together, we can now write DISPLAYFORM2 As we show in Appendix B, the first term above is zero due to Assumption 4.

Under suitable boundary condition on the Fokker-Planck equation which ensures that no probability mass flows across the boundary of the domain ∂ Ω, after an integration by parts, the second term can be written as DISPLAYFORM3 In the above expression, A : B denotes the matrix dot product A : B = ∑ i j A i j B i j .

The final inequality with the quadratic form holds because D(x) 0 is a covariance matrix.

Moreover, we have from (A7) that dF(ρ ss ) dt = 0.

The forward implication can be checked by substituting ρ ss (x) ∝ e −c β f (x) in the Fokker-Planck equation (FP) while the reverse implication is true since otherwise (A4) would not hold.

The Fokker-Planck operator written as DISPLAYFORM0 from FORMULA9 and (FP) can be split into two operators DISPLAYFORM1 where the symmetric part is DISPLAYFORM2 and the anti-symmetric part is DISPLAYFORM3 We first note that L A does not affect F(ρ) in Theorem 5.

For solutions of ρ t = L A ρ, we have DISPLAYFORM4 by Assumption 4.

The dynamics of the anti-symmetric operator is thus completely deterministic and conserves F(ρ).

In fact, the equation FORMULA1 is known as the Liouville equation BID14 and describes the density of a completely deterministic dynamics given bẏ DISPLAYFORM5 where j(x) = J ss /ρ ss from Appendix C. On account of the trajectories of the Liouville operator being deterministic, they are also the most likely ones under the steady-state distribution ρ ss ∝ e −β Φ .

All the matrices below depend on the weights x; we suppress this to keep the notation clear.

Our original SDE is given by dx = −∇ f dt + 2β −1 D dW (t).We will transform the original SDE into a new SDE DISPLAYFORM0 where S and A are the symmetric and anti-symmetric parts of G −1 , DISPLAYFORM1 Since the two SDEs above are equal to each other, both the deterministic and the stochastic terms have to match.

This gives ∇ f (x) = G ∇Φ(x) DISPLAYFORM2 Using the above expression, we can now give an explicit, although formal, expression for the potential: DISPLAYFORM3 where Γ : [0, 1] → Ω is any curve such that Γ(1) = x and Γ(0) = x(0) which is the initial condition of the dynamics in (3).

Note that Φ(x) does not depend on β because G(x) does not depend on β .We now write the modified SDE (A12) as a second-order Langevin system after introducing a velocity variable p with q x and mass m: DISPLAYFORM4 The key idea in BID63 is to compute the Fokker-Planck equation of the system above and take its zero-mass limit.

The steady-state distribution of this equation, which is also known as the Klein-Kramer's equation, is DISPLAYFORM5 where the position and momentum variables are decoupled.

The zero-mass limit is given by DISPLAYFORM6 We now exploit the fact that Q is defined to be an anti-symmetric matrix.

Note that ∑ i, j ∂ i ∂ j (Q i j ρ) = 0 because Q is anti-symmetric.

Rewrite the third term on the last step ( * ) as DISPLAYFORM7 We now use the fact that (3) has ρ ss ∝ e −β Φ as the steady-state distribution as well.

Since the steady-state distribution is uniquely determined by a Fokker-Planck equation, the two equations (FP) and (A16) are the same.

Let us decompose the second term in (FP): DISPLAYFORM8 Observe that the brown terms are equal.

Moving the blue terms together and matching the drift terms in the two Fokker-Planck equations then gives DISPLAYFORM9 The critical points of Φ are different from those of the original loss f by a term that is β −1 ∇ · (D + Q).

<|TLDR|>

@highlight

SGD implicitly performs variational inference; gradient noise is highly non-isotropic, so SGD does not even converge to critical points of the original loss

@highlight

This paper provides a variational analysis of SGD as a non-equilibrium process.

@highlight

This paper discusses the regularized objective function minimized by standard SGD in the context of neural nets, and provide a variational inference perspective using the Fokker-Planck equation.

@highlight

Develops a theory to study the impact of stocastic gradient noise for SGD, especially for deep neural network models
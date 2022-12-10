We build on auto-encoding sequential Monte Carlo (AESMC): a method for model and proposal learning based on maximizing the lower bound to the log marginal likelihood in a broad family of structured probabilistic models.

Our approach relies on the efficiency of sequential Monte Carlo (SMC) for performing inference in structured probabilistic models and the flexibility of deep neural networks to model complex conditional probability distributions.

We develop additional theoretical insights and introduce a new training procedure which improves both model and proposal learning.

We demonstrate that our approach provides a fast, easy-to-implement and scalable means for simultaneous model learning and proposal adaptation in deep generative models.

We build upon AESMC , a method for model learning that itself builds on variational auto-encoders (VAEs) BID4 BID10 and importance weighted auto-encoders (IWAEs) BID0 .

AESMC is similarly based on maximizing a lower bound to the log marginal likelihood, but uses SMC BID3 as the underlying marginal likelihood estimator instead of importance sampling (IS).

For a very wide array of models, particularly those with sequential structure, SMC forms a substantially more powerful inference method than IS, typically returning lower variance estimates for the marginal likelihood.

Consequently, by using SMC for its marginal likelihood estimation, AESMC often leads to improvements in model learning compared with VAEs and IWAEs.

We provide experiments on structured time-series data that show that AESMC based learning was able to learn useful representations of the latent space for both reconstruction and prediction more effectively than the IWAE counterpart.

AESMC was introduced in an earlier preprint concurrently with the closely related methods of ; BID7 .

In this work we take these ideas further by providing new theoretical insights for the resulting evidence lower bounds (ELBOs), extending these to explore the relative efficiency of different approaches to proposal learning, and using our results to develop a new and improved training procedure.

In particular, we introduce a method for expressing the gap between an ELBO and the log marginal likelihood as a Kullback-Leibler (KL) divergence between two distributions on an extended sampling space.

Doing so allows us to investigate the behavior of this family of algorithms when the objective is maximized perfectly, which occurs only if the KL divergence becomes zero.

In the IWAE case, this implies that the proposal distributions are equal to the posterior distributions under the learned model.

In the AESMC case, it has implications for both the proposal distributions and the intermediate set of targets that are learned.

We demonstrate that, somewhat counter-intuitively, using lower variance estimates for the marginal likelihood can actually be harmful to proposal learning.

Using these insights, we experiment with an adaptation to the AESMC algorithm, which we call alternating ELBOs, that uses different lower bounds for updating the model parameters and proposal parameters.

We observe that this adaptation can, in some cases, improve model learning and proposal adaptation.

State-space models (SSMs) are probabilistic models over a set of latent variables x 1:T and observed variables y 1:T .

Given parameters θ, a SSM is characterized by an initial density µ θ (x 1 ), a series of transition densities f t,θ (x t |x 1:t−1 ), and a series of emission densities g t,θ (y t |x 1:t ) with the joint density being p θ (x 1:T , y 1:T ) = µ θ (x 1 ) T t=2 f t,θ (x t |x 1:t−1 ) T t=1 g t,θ (y t |x 1:t ).

We are usually interested in approximating the posterior p θ (x 1:T |y 1:T ) or the expectation of some test function ϕ under this posterior I(ϕ) := ϕ(x 1:T )p θ (x 1:T |y 1:T ) dx 1:T .

We refer to these two tasks as inference.

Inference in models which are non-linear, non-discrete, and non-Gaussian is difficult and one must resort to approximate methods, for which SMC has been shown to be one of the most powerful approaches BID3 ).We will consider model learning as a problem of maximizing the marginal likelihood p θ (y 1:T ) = p θ (x 1:T , y 1:T ) dx 1:T in the family of models parameterized by θ.

SMC performs approximate inference on a sequence of target distributions (π t (x 1:t )) T t=1 .

In the context of SSMs, the target distributions are often taken to be (p θ (x 1:t |y 1:t )) T t=1 .

Given a parameter φ and proposal distributions q 1,φ (x 1 |y 1 ) and (q t,φ (x t |y 1:t , x 1:t−1 )) T t=2 from which we can sample and whose densities we can evaluate, SMC is described in Algorithm 1.Using the set of weighted particles (x DISPLAYFORM0 at the last time step, we can approximate the posterior as DISPLAYFORM1 is the normalized weight and δ z is a Dirac measure centered on z. Furthermore, one can obtain an unbiased estimator of the marginal likelihood p θ (y 1:T ) using the intermediate particle weights: DISPLAYFORM2 (1)Algorithm 1: Sequential Monte Carlo Data: observed values y 1:T , model parameters θ, proposal parameters φ begin Sample initial particle values x k 1 ∼ q 1,φ (·|y 1 ).

Compute and normalize weights: DISPLAYFORM3 Compute and normalize weights: DISPLAYFORM4 The sequential nature of SMC and the resampling step are crucial in making SMC scalable to large T .

The former makes it easier to design efficient proposal distributions as each step need only target the next set of variables x t .

The resampling step allows the algorithm to focus on promising particles in light of new observations, avoiding the exponential divergence between the weights of different samples that occurs for importance sampling as T increases.

This can be demonstrated both empirically and theoretically (Del Moral, 2004, Chapter 9) .

We refer the reader to BID3 ) for an in-depth treatment of SMC.

Given a dataset of observations (y (n) ) N n=1 , a generative network p θ (x, y) and an inference network q φ (x|y), IWAEs BID0 DISPLAYFORM0 where, for a given observation y, the ELBO IS (with K particles) is a lower bound on log p θ (y) by Jensen's inequality: DISPLAYFORM1 DISPLAYFORM2 Note that for K = 1 particle, this objective reduces to a VAE BID4 BID10 ) objective we will refer to as DISPLAYFORM3 The IWAE optimization is performed using stochastic gradient ascent (SGA) where a sample from DISPLAYFORM4 is obtained using the reparameterization trick BID4 and DISPLAYFORM5 is used to perform an optimization step.

AESMC implements model learning, proposal adaptation, and inference amortization in a similar manner to the VAE and the IWAE: it uses SGA on an empirical average of the ELBO over observations.

However, it varies in the form of this ELBO.

In this section, we will introduce the AESMC ELBO, explain how gradients of it can be estimated, and discuss the implications of these changes.

Consider a family of SSMs {p θ (x 1:T , y 1:T ) : θ ∈ Θ} and a family of proposal distributions {q φ (x 1:T |y 1:T ) = q 1,φ (x 1 |y 1 ) T t=2 q t,φ (x t |x 1:t−1 , y 1:t ) : φ ∈ Φ}. AESMC uses an ELBO objective based on the SMC marginal likelihood estimator (1).

In particular, for a given y 1:T , the objective is defined as DISPLAYFORM0 ELBO SMC forms a lower bound to the log marginal likelihood log p θ (y 1:T ) due to Jensen's inequality and the unbiasedness of the marginal likelihood estimator.

Hence, given a dataset (y DISPLAYFORM1 1:T ).For notational convenience, we will talk about optimizing ELBOs in the rest of this section.

However, we note that the main intended use of AESMC is to amortize over datasets, for which the ELBO is replaced by the dataset average J (θ, φ) in the optimization target.

Nonetheless, rather than using the full dataset for each gradient update, will we instead use minibatches, noting that this forms unbiased estimator.

We describe a gradient estimator used for optimizing ELBO SMC (θ, φ, y 1:T ) using SGA.

The SMC sampler in Algorithm 1 proceeds by sampling x DISPLAYFORM0 . .

until the whole particle-weight trajectory (x 1:T 1:K , a 1:K 1:T −1 ) is sampled.

From this trajectory, using equation (1) , we can obtain an estimator for the marginal likelihood.

Assuming that the sampling of latent variables x 1:K 1:T is reparameterizable, we can make their sampling independent of (θ, φ).

In particular, assume that there exists a set of auxiliary random variables DISPLAYFORM1 We use the resulting reparameterized sample of (x To account for the discrete choices of ancestor indices a k t one could additionally use the REIN-FORCE (Williams, 1992) trick, however in practice, we found that the additional term in the estimator has problematically high variance.

We explore various other possible gradient estimators and empirical assessments of their variances in Appendix A. This exploration confirms that including the additional REINFORCE terms leads to problematically high variance, justifying our decision to omit them, despite introducing a small bias into the gradient estimates.

In this section, we express the gap between ELBOs and the log marginal likelihood as a KL divergence and study implications on the proposal distributions.

We present a set of claims and propositions whose full proofs are in Appendix B.

These give insight into the behavior of AESMC and show the advantages, and disadvantages, of using our different ELBO.

This insight motivates Section 4 which proposes an algorithm for improving proposal learning.

Definition 1.

Given an unnormalized target densityP : X → [0, ∞) with normalizing constant Z P > 0, P :=P /Z P , and a proposal density Q : DISPLAYFORM0 is a lower bound on log Z P and satisfies DISPLAYFORM1 This is a standard identity used in variational inference and VAEs.

In the case of VAEs, applying Definition 1 with P being p θ (x|y),P being p θ (x, y), Z P being p θ (y), and Q being q φ (x|y), we can directly rewrite (4) as ELBO VAE (θ, φ, y) = log p θ (y) − KL (q φ (x|y)||p θ (x|y)).The key observation for expressing such a bound for general ELBOs such as ELBO IS and ELBO SMC is that the target density P and the proposal density Q need not directly correspond to p θ (x|y) and q φ (x|y).

This allows us to view the underlying sampling distributions of the marginal likelihood Monte Carlo estimators such as Q IS in (3) and Q SMC in (6) as proposal distributions on an extended space X .

The following claim uses this observation to express the bound between a general ELBO and the log marginal likelihood as KL divergence from the extended space sampling distribution to a corresponding target distribution.

Claim 1.

Given a non-negative unbiased estimatorẐ P (x) ≥ 0 of the normalizing constant Z P where x is distributed according to the proposal distribution Q(x), the following holds: DISPLAYFORM2 where DISPLAYFORM3 is the implied normalized target density.

In the case of IWAEs, we can apply Claim 1 with Q andẐ P being Q IS andẐ IS respectively as defined in FORMULA7 and Z P being p θ (y).

This yields DISPLAYFORM4 DISPLAYFORM5 Similarly, in the case of AESMC, we obtain DISPLAYFORM6 DISPLAYFORM7 Having expressions for the target distribution P and the sampling distribution Q for a given ELBO allows us to investigate what happens when we maximize that ELBO, remembering that the KL term is strictly non-negative and zero if and only if P = Q. For the VAE and IWAE cases then, provided the proposal is sufficiently flexible, one can always perfectly maximize the ELBO by setting p θ (x|y) = q φ (x|y) for all x. The reverse implication also holds: if ELBO VAE = log Z P then it must be the case that p θ (x|y) = q φ (x|y).

However, for AESMC, achieving ELBO = log Z P is only possible when one also has sufficient flexibility to learn a particular series of intermediate target distributions, namely the marginals of the final target distribution.

In other words, it is necessary to learn a particular factorization of the generative model, not just the correct individual proposals, to achieve P = Q and thus ELBO SMC = Z P .

These observations are formalized in Propositions 1 and 2 below.

Proposition 1.

Q IS (x 1:K ) = P IS (x 1:K ) for all x 1:K if and only if q(x|y) = p(x|y) for all x. 1.

π t (x 1:t ) = p(x 1:T |y 1:T ) dx t+1:T = p(x 1:t |y 1:T ) for all x 1:t and t = 1, . . .

, T , and 2.

q 1 (x 1 |y 1 ) = p(x 1 |y 1:T ) for all x 1 and q t (x t |x 1:t−1 , y 1:t ) = p(x 1:t |y 1:T )/p(x 1:t−1 |y 1:T ) for t = 2, . . .

, T for all x 1:t , where π t (x 1:t ) are the intermediate targets used by SMC.Proposition 2 has the consequence that if the family of generative models is such that the first condition does not hold, we will not be able to make the bound tight.

This means that, except for a very small class of models, then, for most convenient parameterizations, it will be impossible to learn a perfect proposal that gives a tight bound, i.e. there will be no θ and φ such that the above conditions can be satisfied.

However, it also means that ELBO SMC encodes important additional information about the implications the factorization of the generative model has on the inference-the model depends only on the final target π T (x 1:T ) = p θ (x 1:T |y 1:T ), but some choices of the intermediate targets π t (x 1:t ) will lead to much more efficient inference than others.

Perhaps more importantly, SMC is usually a far more powerful inference algorithm than importance sampling and so the AESMC setup allows for more ambitious model learning problems to be effectively tackled than the VAE or IWAE.

After all, even though it is well known in the SMC literature that, unlike for IS, most problems have no perfect set of SMC proposals which will generate exact samples from the posterior BID3 ), SMC still gives superior performance on most problems with more than a few dimensions.

These intuitions are backed up by our experiments that show that using ELBO SMC regularly learns better models than using ELBO IS .

In practice, one is rarely able to perfectly drive the divergence to zero and achieve a perfect proposal.

In addition to the implications of the previous section, this occurs because q φ (x 1:T |y 1:T ) may not be sufficiently expressive to represent p θ (x 1:T |y 1:T ) exactly and because of the inevitable sub-optimality of the optimization process, remembering that we are aiming to learn an amortized inference artifact, rather than a single posterior representation.

Consequently, to accurately assess the merits of different ELBOs for proposal learning, it is necessary to consider their finite-time performance.

We therefore now consider the effect the number of particles K has on the gradient estimators for ELBO IS and ELBO SMC .

Counter-intuitively, it transpires that the tighter bounds implied by using a larger K is often harmful to proposal learning for both IWAE and AESMC.

At a high-level, this is because an accurate estimate forẐ P can be achieved for a wide range of proposal parameters φ and so the magnitude of ∇ φ ELBO reduces as K increases.

Typically, this shrinkage happens faster than increasing K reduces the standard deviation of the estimate and so the standard deviation of the gradient estimate relative to the problem scaling (i.e. as a ratio of true gradient ∇ φ ELBO) actually increases.

This effect is demonstrated in FIG4 which shows a kernel density estimator for the distribution of the gradient estimate for different K and the model given in Section 5.2.

Here we see that as we increase K, both the expected gradient estimate (which is equal to the true gradient by unbiasedness) and standard deviation of the estimate decrease.

However, the former decreases faster and so the relative standard deviation increases.

This is perhaps easiest to appreciate by noting that for K > 10, there is a roughly equal probability of the estimate being positive or negative, such that we are equally likely to increase or decrease the parameter value at the next SGA iteration, inevitably leading to poor performance.

On the other hand, when K = 1, it is far more likely that the gradient estimate is positive than negative, and so there is clear drift to the gradient steps.

We add to the empirical evidence for this behavior in Section 5.

Note the critical difference for model learning is that ∇ θ ELBO does not, in general, decrease in magnitude as K increases.

Note also that using a larger K should always give better performance at test time; it may though be better to learn φ using a smaller K.In simultaneously developed work , we formalized this intuition in the IWAE setting by showing that the estimator of ∇ φ ELBO IS (θ, φ, x) with K particles, denoted by I K , has the following signal-to-noise ratio (SNR): DISPLAYFORM0 We thus see that increasing K reduces the SNR and so the gradient updates for the proposal will degrade towards pure noise if K is set too high.

To address these issues, we suggest and investigate the alternating ELBOs (ALT) algorithm which updates (θ, φ) in a coordinate descent fashion using different ELBOs, and thus gradient estimates, for each.

We pick a θ-optimizing pair and a φ-optimizing pair (A θ , K θ ), (A φ , K φ ) ∈ {IS, SMC} × {1, 2, . . . }, corresponding to an inference type and number of particles.

In an optimization step, we obtain an estimator for ∇ θ ELBO A θ with K θ particles and an estimator for ∇ φ ELBO A φ with K φ particles which we call g θ and g φ respectively.

We use g θ to update the current θ and g φ to update the current φ.

The results from the previous sections suggest that using A θ = SMC and A φ = IS with a large K θ and a small K φ may perform better model and proposal learning than just fixing (A θ , K θ ) = (A φ , K φ ) to (SMC, large) since using A φ = IS with small K φ helps learning φ (at least in terms of the SNR) and using A θ = SMC with large K θ helps learning θ.

We experimentally observe that this procedure can in some cases improve both model and proposal learning.

We now present a series of experiments designed to answer the following questions: 1) Does tightening the bound by using either more particles or a better inference procedure lead to an adverse effect on proposal learning?

2) Can AESMC, despite this effect, outperform IWAE?

3) Can we further improve the learned model and proposal by using ALT?First we investigate a linear Gaussian state space model (LGSSM) for model learning and a latent variable model for proposal adaptation.

This allows us to compare the learned parameters to the optimal ones.

Doing so, we confirm our conclusions for this simple problem.

We then extend those results to more complex, high dimensional observation spaces that require models and proposals parameterized by neural networks.

We do so by investigating the Moving Agents dataset, a set of partially occluded video sequences.

Given the following LGSSM DISPLAYFORM0 DISPLAYFORM1 DISPLAYFORM2 we find that optimizing ELBO SMC (θ, φ, y 1:T ) w.r.t.

θ leads to better generative models than optimizing ELBO IS (θ, φ, y 1:T ).

The same is true for using more particles.

We generate a sequence y 1:T for T = 200 by sampling from the model with θ = (θ 1 , θ 2 ) = (0.9, 1.0).

We then optimize the different ELBOs w.r.t.

θ using the bootstrap proposal q 1 (x 1 |y 1 ) = µ θ (x 1 ) and q t (x t |x 1:t−1 , y 1:t ) = f t,θ (x t |x 1:t−1 ).

Because we use the bootstrap proposal, gradients w.r.t.

to θ are not backpropagated through q.

We use a fixed learning rate of 0.01 and optimize for 500 steps using SGA.

FIG3 shows that the convergence of both log p θ (y 1:T ) to max θ log p θ (y 1:T ) and θ to argmax θ log p θ (y 1:T ) is faster when ELBO SMC and more particles are used.

We now investigate how learning φ, i.e. the proposal, is affected by the the choice of ELBO and the number of particles.

Consider a simple, fixed generative model p(µ)p(x|µ) = Normal(µ; 0, 1 2 )Normal(x; µ, 1 2 ) where µ and x are the latent and observed variables respectively and a family of proposal distributions q φ (µ) = Normal(µ; µ q , σ 2 q ) parameterized by φ = (µ q , log σ 2 q ).

For a fixed observation x = 2.3, we initialize φ = (0.01, 0.01) and optimize ELBO IS with respect to φ.

We investigate the quality of the learned parameter φ as we increase the number of particles K during training.

FIG6 (left) clearly demonstrates that the quality of φ compared to the analytic posterior decreases as we increase K. FIG6 (middle, right) where we optimize ELBO SMC with respect to both θ and φ for the LGSSM described in Section 5.1.

We see that using more particles helps model learning but makes proposal learning worse.

Using our ALT algorithm alleviates this problem and at the same time makes model learning faster as it profits from a more accurate proposal distribution.

We provide more extensive experiments exploring proposal learning with different ELBOs and number of particles in Appendix C.3.

is an marginal mean obtained from the set of 10 SMC particles with learned/bootstrap proposal.

To show that our results are applicable to complex, high dimensional data we compare AESMC and IWAE on stochastic, partially observable video sequences.

FIG22 in Appendix C.2 shows an example of such a sequence.

The dataset consists of N = 5000 sequences of images (y (n) 1:T ) N n=1 of which 1000 are randomly held out as test set.

Each sequence contains T = 40 images represented as a 2 dimensional array of size 32 × 32.

In each sequence there is one agent, represented as circle, whose starting position is sampled randomly along the top and bottom of the image.

The dataset is inspired by BID8 , however with the crucial difference that the movement of the agent is stochastic.

The agent performs a directed random walk through the image.

At each timestep, it moves according to y t+1 ∼ Normal(y t+1 ; y t + 0.15, 0.02 2 ) DISPLAYFORM0 where (x t , y t ) are the coordinates in frame t in a unit square that is then projected onto 32 × 32 pixels.

In addition to the stochasticity of the movement, half of the image is occluded, preventing the agent from being observed.

For the generative model and proposal distribution we use a Variational Recurrent Neural Network (VRNN) BID1 .

It extends recurrent neural networks (RNNs) by introducing a stochastic latent state x t at each timestep t. Together with the observation y t , this state conditions the deterministic transition of the RNN.

By introducing this unobserved stochastic state, the VRNN is able to better model complex long range variability in stochastic sequences.

Architecture and hyperparameter details are given in Appendix C.1.

FIG7 shows max(ELBO IS , ELBO SMC ) for models trained with IWAE and AESMC for different particle numbers.

The lines correspond to the mean over three different random seeds and the shaded areas indicate the standard deviation.

The same number of particles was used for training and testing, additional hyperparameter settings are given in the appendix.

One can see that models trained using AESMC outperform IWAE and using more particles improves the ELBO for both.

In Appendix C.2, we inspect different learned generative models by using them for prediction, confirming the results presented here.

We also tested ALT on this task, but found that while it did occasionally improve performance, it was much less stable than IWAE and AESMC.

We have developed AESMC-a method for performing model learning using a new ELBO objective which is based on the SMC marginal likelihood estimator.

This ELBO objective is optimized using SGA and the reparameterization trick.

Our approach utilizes the efficiency of SMC in models with intermediate observations and hence is suitable for highly structured models.

We experimentally demonstrated that this objective leads to better generative model training than the IWAE objective for structured problems, due to the superior inference and tighter bound provided by using SMC instead of importance sampling.

Additionally, in Claim 1, we provide a simple way to express the bias of objectives induced by log of marginal likelihood estimators as a KL divergence on an extended space.

In Propositions 1 and 2, we investigate the implications of these KLs being zero in the case of IWAE and AESMC.

In the latter case, we find that we can achieve zero KL only if we are able to learn SMC intermediate target distributions corresponding to marginals of the target distribution.

Using our assertion that tighter variational bounds are not necessarily better, we then introduce and test a new method, alternating ELBOs, that addresses some of these issues and observe that, in some cases, this improves both model and proposal learning.

The goal is to obtain an unbiased estimator for the gradient which we can estimate by sampling (x DISPLAYFORM0 Discrete(a where r( In FIG12 , we demonstrate that the estimator in (31) has much higher variance if we include the first term.

Derivation of (9).

DISPLAYFORM0 Proof of Claim 1.

SinceẐ P (x) ≥ 0, Q(x) ≥ 0 and Q(x)Ẑ P (x) dx = Z P , we can let the unnormalized target density in Definition 1 beP (x) = Q(x)Ẑ P (x).

Hence, the normalized target density is P (x) = Q(x)Ẑ P (x)/Z P .

Substituting these quantities into (8) and (9) yields the two equalities in (10).Proof of Proposition 1.

DISPLAYFORM1 Integrating both sides with respect to (x 2 , . . .

, x K ) over the whole support (i.e. marginalizing out everything except x 1 ), we obtain: DISPLAYFORM2 Rearranging gives us q(x 1 |y) = p(x 1 |y) for all x 1 .( ⇐= ) Substituting p(x k |y) = q(x k |y), we obtain DISPLAYFORM3 DISPLAYFORM4 Proof of Proposition 2.

We consider the general sequence of target distributions π t (x 1:t ) (p θ (x 1:t |y 1:t ) in the case of SSMs), their unnormalized versions γ t (x 1:t ) (p θ (x 1:t , y 1:t ) in the case of SSMs), their normalizing constants Z t = γ t (x 1:t ) dx 1:t (p θ (y 1:t ) in the case of SSMs), where Z = Z T = p(y 1:T ).

DISPLAYFORM5 w t (x 1:t ) := γ t (x 1:t ) γ t−1 (x 1:t−1 )q t (x t |x 1:t−1 ) for t = 2, . . . , Tare constant with respect to x 1:t .Pick t ∈ {1, . . .

, T } and distinct k, ∈ {1, . . . , K}. Also, pick x 1:t and x 1:t .

Now, consider two sets of particle sets (x FIG17 , such that The weightsw κ τ andw κ τ for the respective particle sets are identical except when (τ, κ) = (t, k) wherew DISPLAYFORM6 DISPLAYFORM7 DISPLAYFORM8 DISPLAYFORM9 DISPLAYFORM10 DISPLAYFORM11 SinceẐ ( , we have w t (x 1:t ) = w t (x 1:t ).

As this holds for any arbitrary t and x 1:t , it follows that w t (x 1:t ) must be constant with respect to x 1:t for all t = 1, . . .

, T .

DISPLAYFORM12 where w t := w t (x 1:t ) is constant from our previous results.

For this to be a normalized density with respect to x t , we must have DISPLAYFORM13 and for t = 2, . . .

, T : DISPLAYFORM14 DISPLAYFORM15 Since π t+1 (x 1:t+1 ) dx t+1 and π t (x 1:t ) are both normalized densities, we must have π t (x 1:t ) = π t+1 (x 1:t+1 ) dx t+1 for all t = 1, . . .

, T − 1 for all x 1:t .

For a given t ∈ {1, . . . , T − 1} and x 1:t , applying this repeatedly yields DISPLAYFORM16 such that each π t (x 1:t ) must be the corresponding marginal of the final target.

We also have DISPLAYFORM17 DISPLAYFORM18 DISPLAYFORM19 DISPLAYFORM20 ( ⇐= ) To complete the proof, we now simply substitute identities in 1 and 2 of Proposal 2 back to the expression ofẐ(x

In the following we give the details of our VRNN architecture.

The generative model is given by: DISPLAYFORM0 where DISPLAYFORM1 and the proposal distribution is given by DISPLAYFORM2 The functions µ x θ and σ x θ are computed by networks with two fully connected layers of size 128 whose first layer is shared.

ϕ x θ is one fully connected layer of size 128.

For visual input, the encoding ϕ y θ is a convolutional network with conv-4x4-2-1-32, conv-4x4-2-1-64, conv-4x4-2-1-128 where conv-wxh-s-p-n denotes a convolutional network with n filters of size w × h, stride s, padding p. Between convolutions we use leaky ReLUs with slope 0.2 as nonlinearity and batch norms.

The decoding µ y θ uses transposed convolutions of the same dimensions but in reversed order, however with stride s = 1 and padding p = 0 for the first layer.

A Gated Recurrent Unit (GRU) is used as RNN and if not stated otherwise ReLUs are used in between fully connected layers.

For the proposal distribution, the functions µ For the moving agents dataset we use ADAM with a learning rate of 10 −3 .A specific feature of the VRNN architecture is that the proposal and the generative model share the component ϕ y φ,θ .

Consequently, we set φ = θ for the parameters belonging to this module and train it using gradients for both θ and φ.

In FIG22 we investigate the quality of the generative model by comparing visual predictions.

We do so for models learned by IWAE (top) and AESMC (bottom).

The models were learned using ten particles but for easier visualization we only predict using five particles.

The first row in each graphic shows the ground truth.

The second row shows the averaged predictions of all five particles.

The next five rows show the predictions made by each particle individually.

The observations (i.e. the top row) up to t = 19 are shown to the model.

Up to this timestep the latent values x 0:19 are drawn from the proposal distribution q(x t |y t , h t−1 ).

From t = 20 onwards the latent values x 20:37 are drawn from the generative model p(x t |x t−1 ).

Consequently, the model predicts the partially occluded, stochastic movement over 17 timesteps into the future.

We note that most particles predict a viable future trajectory.

However, the model learned by IWAE is not as consistent in the quality of its predictions, often 'forgetting' the particle.

This does not happen in every predicted sequence but the behavior shown here is very typical.

Models learned by AESMC are much more consistent in the quality of their predictions.

We have run experiments where we optimize various ELBO objectives with respect to φ with θ fixed in order to see how various objectives have an effect on proposal learning.

In particular, we train ELBO IS and ELBO SMC with number of particles K ∈ {10, 100, 1000}. Once the training is done, we use the trained proposal network to perform inference using both IS and SMC with number of particles K test ∈ {10, 100, 1000}.In FIG23 , we see experimental results for the LGSSM described in Section 5.1.

We measure the quality of the inference network using a proxy We see that if we train using ELBO SMC with K train = 1000, the performance for inference using SMC (with whichever K test ∈ {10, 100, 1000}) is worse than if we train with ELBO IS with any number of particles K train ∈ {10, 100, 1000}. Examining the other axes of variation:• Increasing K test (moving up in FIG23 (Right)) improves inference.• Increasing K train (moving to the right in FIG23 (Right)) worsens inference.• Among different possible combinations of (training algorithm, testing algorithm), (IS, SMC) (SMC, SMC) (IS, IS) (SMC, IS), where we use "a b" to denote that the combination a results in better inference than combination b. ) 2 which is a proxy for inference quality of φ described in the main text.

The larger the square, the worse the inference.

<|TLDR|>

@highlight

We build on auto-encoding sequential Monte Carlo, gain new theoretical insights and develop an improved training procedure based on those insights.

@highlight

The paper proposes a version of IWAE-style training that uses SMC instead of classical importance sampling.

@highlight

This work proposes auto-encoding sequential Monte Carlo (SMC), extending the VAE framework to a new Monte Carto objective based on SMC. 
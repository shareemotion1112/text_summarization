A framework for efficient Bayesian inference in probabilistic programs is introduced by embedding a sampler inside a variational posterior approximation.

Its strength lies in both ease of implementation and automatically tuning sampler parameters to speed up mixing time.

Several strategies to approximate the evidence lower bound (ELBO) computation are introduced, including a rewriting of the ELBO objective.

Experimental evidence is shown by performing experiments on an unconditional VAE on density estimation tasks; solving an influence diagram in a high-dimensional space with a conditional variational autoencoder (cVAE) as a deep Bayes classifier; and state-space models for time-series data.

We consider a probabilistic program (PP) to define a distribution p(x, z), where x are observations and z, both latent variables and parameters, and ask queries involving the posterior p(z|x).

This distribution is typically intractable but, conveniently, probabilistic programming languages (PPLs) provide inference engines to approximate it using Monte Carlo methods (e.g. particle Markov Chain Monte Carlo (MCMC) (Andrieu et al., 2010) or Hamiltonian Monte Carlo (HMC) (Neal et al., 2011) ) or variational approximations (e.g. Automatic Differentiation Variational Inference (ADVI) (Kucukelbir et al., 2017) ).

Whereas the latter are biased and underestimate uncertainty, the former may be exceedingly slow depending on the target distribution.

For such reason, over the recent years, there has been an increasing interest in developing more efficient posterior approximations (Nalisnick et al., 2016; Salimans et al., 2015; Tran et al., 2015) .

It is known that the performance of a sampling method depends on the parameters used (Papaspiliopoulos et al., 2007) .

Here we propose a framework to automatically adapt the posterior shape and tune the parameters of a posterior sampler with the aim of boosting Bayesian inference in PPs.

Our framework constitutes a principled way to enhance the flexibility of the variational posterior approximation, yet can be seen also as a procedure to tune the parameters of an MCMC sampler.

Our contributions are a new flexible and unbiased variational approximation to the posterior, which improves an initial variational approximation with a (learnable via automatic differentiation) stochastic process.

Appendix A discusses related work.

In standard VI, the variational approximation q φ (z|x) is analytically tractable and typically chosen as a factorized Gaussian distribution.

We propose to use a more flexible approximating posterior by embedding a sampler through: q φ,η (z|x) = Q η,T (z|z 0 )q 0,φ (z 0 |x)dz 0 ,

where q 0,φ (z|x) is the initial and tractable density (i.e., the starting state for the sampler).

We refer to q φ,η (z|x) as the refined variational approximation.

The distribution Q η,T (z|z 0 ) refers to a stochastic process parameterized by η used to evolve the original density q 0,φ (z|x) and achieve greater flexibility; we describe below particular forms of it.

When T = 0, no refinement steps are performed, and the refined variational approximation coincides with the original one, q φ,η (z|x) = q 0,φ (z|x).

As T increases, the variational approximation will be closer to the exact posterior, provided that Q η,T is a valid MCMC sampler.

Next, we maximize a refined ELBO objective, ELBO(q) = E q φ,η (z|x) [log p(x, z) − log q φ,η (z|x)]

to optimize the divergence KL(q φ,η (z|x)||p(z|x)).

The first term of ELBO only requires sampling from q φ,η (z|x); however the second term, −E q φ,η (z|x) [log q φ,η (z|x)] requires also evaluating the evolving density.

Regarding Q η,T (z|z 0 ), we consider the following families of sampling algorithms.

When the latent variables z are continuous (z ∈ R d ), we evolve the original variational density q 0,φ (z|x) through a stochastic diffusion process.

To make it tractable, we discretize the Langevin dynamics using the Euler-Maruyama scheme, arriving at the stochastic gradient Langevin dynamics (SGLD) sampler.

We then follow the process Q η,T (z|z 0 ) (representing T iterations of an MCMC sampler).

As an example, for the SGLD sampler z i = z i−1 + η∇ log p(x, z i−1 ) + ξ i , where i iterates from 1 to T ; in this case, the only parameter of the SGLD sampler is the learning rate η.

The noise for the SGLD is ξ i ∼ N (0, 2ηI).

The initial variational distribution q 0,φ (z|x) is a Gaussian parameterized by a deep neural network (NN).

Then, T iterations of a sampler Q parameterized by η are applied leading to q φ,η .

An alternative may be given by ignoring the noise vector ξ (Mandt et al., 2017), thus refining the initial variational approximation with just stochastic gradient descent (SGD).

Moreover, we can use Stein variational gradient descent (SVGD) (Liu and Wang, 2016) or a stochastic version (Gallego and Insua, 2018) to apply repulsion between particles and promote a more extensive exploration of the latent space.

We propose a set of guidelines for the ELBO optimization using the refined variational approximation.

Particle approximation We can consider the flow Q η,T (z|z 0 ) as a mixture of Dirac deltas (i.e., we approximate it with a finite set of particles).

That is, we sample z 1 , . . .

, z K ∼ Q η,T (z|z 0 ) and useQ η,T (z|z 0 ) = T i=1 q η (z i |z i−1 )q 0,φ (z 0 |x).

The entropy for each factor can be straightforwardly computed, i.e. for the case of SGLD, q η (z i |z i−1 ) = N (z i−1 + η∇ log p(x, z i−1 ), 2ηI).

This approximation keeps track of a better estimate of the entropy than the particle approximation.

Deterministic flows If using a deterministic flow (such as SGD or SVGD), we can keep track of the change in entropy at each iteration using the change of variable formula as done in Duvenaud et al. (2016) .

However, this requires a costly Jacobian computation, making it unfeasible to combine with our backpropagation through the sampler scheme (Sec. 2.3) for moderately complex problems.

In standard VI, the variational approximation q(z|x; φ) is parameterized by φ.

The parameters are learned using SGD or variants such as Adam (Kingma and Ba, 2014) , using ∇ φ ELBO(q).

Since we have shown how to embed a sampler inside the variational guide, it is also possible to compute a gradient of the objective with respect to the sampler parameters η.

For instance, we can compute a gradient with respect to the learning rate η from the SGLD or SGD process from Section 2.1, ∇ η ELBO(q), to search for an optimal step size at every VI iteration.

This is an additional step apart from using the gradient ∇ φ ELBO(q) employed to learn a good initial sampling distribution.

See Appendix D.3 for a discussion on two modes of automatic differentiation that can be used.

Code is released at https://github.com/vicgalle/vis.

The VIS framework was implemented using Pytorch (Paszke et al., 2017) , though we also release a notebook for the first experiment using Jax to highlight its simple implementation.

Appendix B contains additional experiments; Appendix C, implementation details.

Funnel density As a preliminary experiment, we test the VIS framework on a synthetic yet complex target distribution.

The target, bi-dimensional density is defined through:

As a variational approximation we take the usual diagonal Gaussian.

For the VIS case, we refine it for T = 1 steps using SGLD.

Results are in Figure 1 .

Clearly, our refined version achieves a tighter bound, the VIS variant is placed nearer to the mean of the true distribution and is more disperse than the original variational approximation, confirming that the refinement step helps in attaining more flexible posterior approximations.

State-space model (DLM) We now test the VIS framework on the Mauna Loa monthly CO 2 time series data (Keeling, 2005) .

As the training set, we take the first 10 years, and we evaluate over the next 2 years.

We use a dynamic linear model (DLM) composed of a local linear trend plus a seasonality block of periodicity 12.

Full model specification can be checked in Appendix C.1.

As a preprocessing step, we standardize the time series to zero mean and unitary deviation.

To guarantee the same computational budget time, the model without refining is run for 10 epochs, whereas the model with refinement is run for 4 epochs.

We use the particle approximation from Sec. 2.2.

We report mean absolute error (MAE) and predictive entropy in Table 1 .

In addition, we compute the interval score as defined in (Gneiting and Raftery, 2007) , a strictly proper scoring rule.

As can be seen, for similar wall-clock times, the refined model not only achieves lower MAE, but also its predictive intervals are narrower than the non-refined counterpart.

Variational Autoencoder We aim to check whether VIS is competitive with respect to other recent algorithms.

We test our approach in a Variational Autoencoder (VAE) model (Kingma and Welling, 2013) , which is the building block of more complex models and tasks (Chen et al., 2018b; Bouchacourt et al., 2018) .

The VAE defines a conditional distribution p θ (x|z), generating an observation x from a latent variable z. We are interested in modelling two 28 × 28 image distributions, MNIST and fashion-MNIST.

To perform inference (learn parameters θ), the VAE introduces a variational approximation q φ (z|x).

In the standard setting, this is Gaussian; we instead use the refined variational approximation with various values of T .

We used the MC approximation, though achieved similar results using the Gaussian one.

As experimental setup, we reproduce the setting from Titsias and Ruiz (2019) .

Results are reported in Table 2 .

To guarantee a fair comparison, we trained the VIS-5-10 variant for 10 epochs, whereas all the other variants were trained for 15 (fMNIST) or 20 epochs (MNIST), so that the VAE performance is comparable to that in Titsias and Ruiz (2019).

Although VIS is trained for less epochs, by increasing the number of MCMC iterations T , we dramatically improve on test log-likelihood.

In terms of computational complexity, the average time per epoch using T = 5 is 10.46s, whereas with no refinement (T = 0) is 6.10s (hence our decision to train the refined variant for less epochs): a moderate increase in computing time compensates the dramatic increase in log-likelihood while not introducing new parameters, except for the learning rate η.

We also compare our results with the contrastive divergence approach (Ruiz and Titsias, 2019).

Figure 2 displays ten random samples of reconstructed digit images as visual check.

Discussion We have proposed a flexible and efficient framework to perform inference in probabilistic programs defining wide classes of models.

Our framework can be seen as a general way of tuning SG-MCMC sampler parameters, adapting the initial distributions and the learning rate.

Key to the success and applicability of the VIS framework are the approximations introduced for the intractable parts of the refined variational approximations, which are computationally cheap but convenient.

The idea of preconditioning the posterior distribution to speed up the mixing time of an MCMC sampler has recently been explored in (Hoffman et al., 2018) and (Li and Wang, 2018) , where a reparameterization is learned before performing the sampling via HMC.

Both papers extend seminal work of (Parno and Marzouk, 2014) by learning an efficient and expressive deep, non-linear transformation instead of a polynomial regression.

However, they do not account for tuning the parameters of the sampler as we introduce in Section 2, where a fully, end to end differentiable sampling scheme is proposed.

The work of (Rezende and Mohamed, 2015) introduced a general framework for constructing more flexible variational distributions, called normalizing flows.

These transformations are one of the main techniques to improve the flexibility of current VI approaches and have recently pervaded the literature of approximate Bayesian inference with current developments such as continuous-time normalizing flows (Chen et al., 2018a) which extend an initial simple variational posterior with a discretization of Langevin dynamics.

However, they require a generative adversarial network (GAN) (Goodfellow et al., 2014) to learn the posterior, which can be unstable in high-dimensional spaces.

We overcome this issue with the novel formulation stated in Section 2.

Our framework is also compatible with different optimizers, not only those derived from Langevin dynamics.

Other recent proposals to create more flexible variational posteriors are based on implicit approaches, which typically require a GAN (Huszár, 2017) or implicit schema such as UIVI (Titsias and Ruiz, 2019) or SIVI (Yin and Zhou, 2018).

Our variational approximation is also implicit, but we use a sampling algorithm to drive the evolution of the density, combined with a Dirac delta approximation to derive an efficient variational approximation as we report on the extensive experiments in the Section 3.

Closely related to our framework is the work of Hoffman (2017), where a VAE is learned using HMC.

We use a similar compound distribution as the variational approximation, though our framework allows for any SG-MCMC sampler (via the entropy approximation strategies introduced) and also the tuning of sampler parameters via gradient descent.

Our work is also related to the recent idea of amortization of samplers (Feng et al., 2017) .

A common problem with these approaches is that they incur in an additional error, the amortization gap (Cremer et al., 2018) .

We alleviate this by evolving a set of particles z i with a stochastic process in the latent space after learning a good initial distribution.

Hence, the bias generated by the initial approximation is significantly reduced after several iterations of the process.

A recent article related to our paper is (Ruiz and Titsias, 2019), who define a compound distribution similar to our framework.

However, we focus on an efficient approximation using the reverse KL divergence, the standard and well understood divergence used in variational inference, which allows for tuning sampler parameters and achieving competitive results.

With the final experiments we show that the VIS framework can deal with more general probabilistic graphical models.

Influence diagrams (Howard and Matheson, 2005) are one of the most familiar representations of a decision analysis problem.

There is a long history on bridging the gap between influence diagrams and probabilistic graphical models (see (Shachter, 1988) , for instance), so developing better tools for Bayesian inference can be automatically used to solve influence diagrams.

We showcase the flexibility of the proposed scheme to solve inference problems in an experiment with a classification task in a high-dimensional setting.

As dataset, the MNIST (LeCun et al., 1998) handwritten digit classification task is chosen, in which grey-scale 28 × 28 images have to be classified in one of the ten classes Y = {0, 1, . . .

, 9}. More concretely, we extend the VAE model to condition it on a discrete variable y, leading to the conditional VAE (cVAE).

A cVAE defines a decoder distribution p θ (x|z, y) on an input space x ∈ R D given class label y ∈ Y and latent variable z ∈ R d .

To perform inference, a variational posterior is learned as an encoder q φ (z|x, y) from a prior p(z) ∼ N (0, I).

Leveraging the conditional structure on y, we use the generative model as a classifier using Bayes rule:

where we use K Monte Carlo samples z (k) ∼ q φ (z|x, y).

In the experiments we set K = 5.

Given a test sample x, the labelŷ with highest probability p(y|x) is predicted.

Figure 5 in Appendix depicts the corresponding influence diagram.

Additional details regarding the model architecture and hyperparameters can be found in Appendix C. For comparison purposes, we perform various experiments changing T for the transition distribution Q η,T in the refined variational approximation.

Results are in Table 3 .

We report the test accuracy achieved at the end of training.

Note we are comparing different values of T depending on being on the training or testing phases (in the latter, where the model and variational parameters are kept frozen).

The model with T tr = 5 was trained for 10 epochs, whereas the other settings for 15 epochs, in order to give all settings similar training times.

Results are averaged from 3 runs with different random seeds.

From the results it is clear that the effect of using the refined variational approximation (the cases when T > 0) is crucially beneficial to achieve higher accuracy.

The effect of learning a good initial distribution and inner learning rate by using the gradients ∇ φ ELBO(q) and ∇ η ELBO(q) has a highly positive impact in the accuracy obtained.

On a final note, we have not included the case when only using a SGD or SGLD sampler (i.e., without learning an initial distribution q 0,φ (z|x)) since the results were much worse than the ones in Table 3 , for a comparable computational budget.

This strongly suggests that for

We test our variational approximation on two state-space models, one for discrete data and the other for continuous observations.

All the experiments in this subsection use the Fast AD version from Section D.3 since it was not necessary to further tune the sampler parameters to have competitive results.

The model equations are given by

where each conditional is a Categorical distribution which takes 5 different classes and the prior p(θ) = p(θ em )p(θ tr ) are two Dirichlet distributions that sample the emission and transition probabilities, respectively.

We perform inference on the parameters θ.

The model equations are the same as in the HMM case, though the conditional distributions are now Gaussian and the parameters θ refer to the emission and transition variances.

As before, we perform inference over θ.

The full model implementations can be checked in Appendix C.1, based on funsor 1 , a PPL on top of the Pytorch autodiff framework.

For each model, we generate a synthetic dataset, and use the refined variational approximation with T = 0, 1, 2.

As the original variational approximation to the parameters θ we use a Dirac Delta.

Performing VI with this approximation corresponds to MAP estimation using the Kalman filter in the DLM case (Zarchan and Musoff, 2013) and the Baum-Welch algorithm in the HMM case (Rabiner, 1989), since we marginalize out the latent variables z 1:τ .

Model details are given in Appendix C.1.1.

Figure 3 shows the results.

The first row reports the experiments related to the HMM; the second one to the DLM.

While in all graphs we report the evolution of the loglikelihood during inference, in the first column we report the number of ELBO iterations, whereas in the second column we measure wall-clock time as the optimization takes place.

We confirm that VIS (T > 0) achieve better results than regular optimization with VI (T = 0) for a similar amount of time.

With the aim of assessing whether ELBO optimization helps in attaining better auxiliary scores, we also report results on a prediction task.

We generate a synthetic time series of alternating 0 and 1 for τ = 105 timesteps.

We train the HMM model from before on the first 100 points, and report in Table 4 the accuracy of the predictive distribution p(y t ) averaged over the last 5 time-steps.

We also report the predictive entropy since it helps in assessing the confidence of the model in its forecast and is a strictly proper scoring rule (Gneiting and Raftery, 2007) .

To guarantee the same computational budget time and a fair comparison, the model without refining is run with 50 epochs, whereas the model with refinement is run for 20 epochs.

We see that the refined model achieves higher accuracy than its counterpart; in addition it is correctly more confident in its predictions.

x t ∼ N (3.0z t + 0.5, σ em ).

with z 0 = 0.0.

The DLM model is comprised of a linear trend component plus a seasonal block of period 12.

The trend is specified as

With respect to the seasonal component, the main idea is to cycle the state: suppose θ t ∈ R p , with p being the seasonal period.

Then, at each timestep, the model focuses on the first component of the state vector:

Thus, we can specify the seasonal component via:

where F is a p−dimensional vector and G is a p × p matrix such that def encode(self, x): h1_mu = F.relu(self.fc1_mu(x)) h1_cov = F.relu(self.fc1_cov(x)) h1_mu = F.relu(self.fc12_mu(h1_mu)) h1_cov = F.relu(self.fc12_cov(h1_cov)) # we work in the logvar-domain return self.fc2_mu(h1_mu), torch.log(F.softplus(self.fc2_cov(h1_cov))) def decode(self, z): h3 = F.relu(self.fc3(z)) h3 = F.relu(self.fc32(h3)) return torch.sigmoid(self.fc4(h3)) The VAE model is implemented with PyTorch (Paszke et al., 2017) .

The prior distribution p(z) for the latent variables z ∈ R 10 is a standard factorized Gaussian.

The decoder distribution p θ (x|z) and the encoder distribution (initial variational approximation) q 0,φ (z|x) are parameterized by two feed-forward neural networks whose details can be checked in Figure 4 .

The optimizer Adam is used in all experiments, with a learning rate λ = 0.001.

We also set η = 0.001.

We train for 15 epochs (fMNIST) and 20 epochs (MNIST), in order to achieve similar performance to the explicit VAE case in (Titsias and Ruiz, 2019) .

For the VIS-5-10 setting, we train for only 10 epochs, to allow for a fair computational comparison (similar computing times).

The cVAE model is implemented with PyTorch (Paszke et al., 2017) .

The prior distribution p(z) for the latent variables z ∈ R 10 is a standard factorized Gaussian.

The decoder distribution p θ (x|y, z) and the encoder distribution (initial variational approximation) q 0,φ (z|x, y) are parameterized by two feed-forward neural networks whose details can be checked in Figure 6 .

The integral (3) is approximated with 1 MC sample from the variational approximation in all experimental settings.

The optimizer Adam is used in all the experiments, with a learning rate λ = 0.01.

We set the initial η = 5e − 5.

In this Section we study in detail key properties of the proposed VIS framework.

Performing variational inference with the refined variational approximation can be regarded as using the original variational guide while optimizing an alternative, tighter ELBO.

Note that for a refined guide of the form q(z|z 0 )q(z 0 |x), the objective function can be written as

However, using the Dirac Delta approximation for q(z|z 0 ) and noting that z = z 0 + η∇ log p(x, z 0 ) when using SGD with T = 1, we arrive at the modified objective:

which is equivalent to the refined ELBO introduced in (2).

Since we are perturbing the latent variables in the steepest ascent direction, it is straightforward to show that, for moderate η, the previous bound is tighter than the one, for the original variational guide q(z 0 |x), E q(z 0 |x) [log p(x, z 0 ) − log q(z 0 |x)].

This reformulation of ELBO is also convenient since it provides a clear way of implementing our refined variational inference framework in any PPL supporting algorithmic differentiation.

From the result in subsection D.1, we can further restrict to the case when the original variational approximation is also a Dirac point mass.

Then, the original ELBO optimization resorts to the standard maximum likelihood estimation, i.e., max z log p(x, z).

Within the VIS framework, we optimize instead max z log p(x, z + ∆z), where ∆z is one iteration of the sampler, i.e., ∆z = η∇ log p(x, z) in the SGD case.

For notational clarity we resort to the case T = 1, but a similar analysis can be straightforwardly done if more refinement steps are performed.

We may now perform a first-order Taylor expansion of the refined objective as log p(x, z + ∆z) ≈ log p(x, z) + (∆z) ∇ log p(x, z).

Taking gradients of the first order approximation w.r.t.

the latent variables z we arrive at ∇ z log p(x, z) + η∇ z log p(x, z) ∇ 2 z log p(x, z),

where we have not computed the gradient through the ∆z term.

That is, the refined gradient can be deemed as the original gradient plus a second order correction.

Instead of being modulated by a constant learning rate, this correction is adapted by the chosen sampler.

In the experiments in Section B.1 we show that this is beneficial for the optimization as it can take less iterations to achieve lower losses.

By further taking gradients through the ∆z term, we may tune the sampler parameters such as the learning rate as described in Section 2.3.

Consequently, the next subsection describes both modes of differentiation.

Here we describe how to implement two variants of the ELBO objective.

First, we define a stop gradient operator 2 ⊥ that sets the gradient of its operand to zero, i.e., ∇ x ⊥(x) = 0 whereas in the forward pass it acts as the identity function, that is, ⊥(x) = x. Then, the two variants of the ELBO objective are E q [log p(x, z + ∆z) − log q(z + ∆z|x)] (Full AD) and E q [log p(x, z + ⊥(∆z)) − log q(z + ⊥(∆z)|x)] .

(Fast AD)

The Full AD ELBO makes it possible to further compute a gradient wrt sampler parameters inside ∆z at the cost of a slight increase in the computational burden.

Note that for a large class of models (including HMMs and DLMs) we can marginalize out z 1:τ and have reduced variance iterating with:

θ ← θ + ∇ θ log p(x 1:τ |θ) + ξ,

where the latent variables z 1:τ have been marginalized out using the sum-product algorithm.

For linear-Gaussian models we can also compute the exact form of the refined posterior, since all terms in Eq. 5 are linear wrt the latent variables θ.

However, inference in these linear models is exact by using conjugate distributions, so the proposed framework is more fit to the case of state-space models containing non-linear (or non-conjugate) components.

For these families of models, we resort to use just a gradient estimator of the entropy or the Delta approximation in Section 2.1.

@highlight

We embed SG-MCMC samplers inside a variational approximation
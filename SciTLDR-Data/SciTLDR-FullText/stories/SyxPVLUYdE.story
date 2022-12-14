Extending models with auxiliary latent variables is a well-known technique to in-crease model expressivity.

Bachman & Precup (2015); Naesseth et al. (2018); Cremer et al. (2017); Domke & Sheldon (2018) show that Importance Weighted Autoencoders (IWAE) (Burda et al., 2015) can be viewed as extending the variational family with auxiliary latent variables.

Similarly, we show that this view encompasses many of the recent developments in variational bounds (Maddisonet al., 2017; Naesseth et al., 2018; Le et al., 2017; Yin & Zhou, 2018; Molchanovet al., 2018; Sobolev & Vetrov, 2018).

The success of enriching the variational family with auxiliary latent variables motivates applying the same techniques to the generative model.

We develop a generative model analogous to the IWAE bound and empirically show that it outperforms the recently proposed Learned Accept/Reject Sampling algorithm (Bauer & Mnih, 2018), while being substantially easier to implement.

Furthermore, we show that this generative process provides new insights on ranking Noise Contrastive Estimation (Jozefowicz et al.,2016; Ma & Collins, 2018) and Contrastive Predictive Coding (Oord et al., 2018).

Deep generative models with latent variables have seen a resurgence due to the influential work by BID20 ; BID38 and their success at modeling data such as natural images BID37 BID14 , speech and music time-series BID8 BID13 BID22 , and video BID1 BID15 BID10 .

The power of these models lies in the use of auxiliary latent variables to construct complex marginal distributions from tractable conditional distributions.

While directly optimizing the marginal likelihood of latent variable models is intractable, we can instead maximize a variational lower bound on the likelihood such as the evidence lower bound (ELBO) BID17 BID5 .

The tightness of the bound is determined by the expressiveness of the variational family BID43 .Recently, there have been many advances in constructing tighter variational lower bounds for latent variable models (e.g., BID6 ; BID28 ; BID31 ; BID23 ; BID42 ; BID30 ; BID40 ).

Each bound requires a separate derivation and evaluation, however, and the relationship between bounds is unclear.

We show that these bounds can be viewed as specific instances of auxiliary variable variational inference BID0 BID36 BID26 .

In particular, many partition function estimators can be justified from an auxiliary latent variable or extended state space view (e.g., Sequential Monte Carlo BID12 , Hamiltonian Monte Carlo BID34 , Annealed Importance Sampling BID32 ).

Viewed from this perspective, they can be embedded in the variational family as a choice of auxiliary latent variables.

Based on the general results for auxiliary latent variables, this immediately gives rise to a variational lower bound with a characterization of the tightness of the bound.

Furthermore, this view highlights the implicit (potentially suboptimal) choices made and exposes the reusable components that can be combined to form novel auxiliary latent variable schemes.

The success of augmenting variational distributions with auxiliary latent variables motivates investigating a similar augmentation for generative models.

When augmenting the variational distribution, the natural target distribution is the intractable posterior over the latent variables in the model.

With the generative model, this introduces an extra degree of learnable flexibility (i.e., we can learn the unnormalized potential function).

To illustrate this, we develop a latent variable model based on self-normalized importance sampling (Algorithm 1) which can be sampled from exactly and has a tractable lower bound on its log-likelihood.

It interpolates between a tractable proposal distribution and an energy model.

We show that this model is closely related to ranking NCE BID25 and suggests a principled objective for training the noise distribution in NCE.In summary, our contributions are:1.

We view recent tighter variational lower bounds through the lens of auxiliary variable variational inference, unifying their analysis and exposing sub-optimal design choices in algorithms such as IWAE.2.

We apply similar ideas to generative models, developing a new model based on selfnormalized importance sampling which can be fit by maximizing a tractable lower bound on its log-likelihood.3.

We show that the new model generalizes ranking NCE BID25 and provides a proof that the CPC objective BID35 ) is a lower bound on mutual information.4.

We evaluate the proposed model and find it outperforms the recently developed approach in BID4 despite being more computationally efficient and simpler to implement.

In this work, we consider learned probabilistic models of data p(x).

Latent variables z allow us to construct complex distributions by defining the likelihood p(x) = p(x|z)p(z) dz in terms of tractable components p(z) and p(x|z).

While marginalizing z is generally intractable, we can instead optimize a tractable lower bound on log p(x) using the identity DISPLAYFORM0 where q(z|x) is a variational distribution and the positive D KL term can be omitted to form a lower bound commonly referred to as the evidence lower bound (ELBO) BID17 BID5 .

The tightness of the bound is controlled by how accurately q(z|x) models p(z|x), so limited expressivity in the variational family can negatively impact the learned model.

Latent variables can also be used to define complex variational distributions q. As before, we define q(z|x) = q(z|??, x)q(??|x)d?? in terms of tractable conditional distributions q(z|??, x) and q(??|x).

BID0 show that DISPLAYFORM0 where r(??|z, x) is a variational distribution meant to model q(??|z, x), and the identity follows from the fact that q(z|x) = q(z,??|x) q(??|z,x) .

Similar to Eq. (1), Eq. (2) shows the gap introduced by using r(??|z, x) to deal with the intractability of q(z|x).

We can form a lower bound on the original ELBO and thus a lower bound on the log marginal by omitting the positive D KL term.

To tighten the variational bound without explicitly expanding the variational family, BID6 introduced the importance weighted autoencoder (IWAE) bound, DISPLAYFORM0 The IWAE bound reduces to the ELBO when K = 1, is non-decreasing as K increases, and converges to log p(x) as K ??? ??? under mild conditions BID6 .

BID29 developed Monte Carlo Objectives (MCOs), which extend this notion to any unbiased estimator p(x) of p(x) by noting that DISPLAYFORM1 by Jensen's inequality.

IWAE is the special case where the unbiased estimator is the K-sample importance sampling estimator.

Maddison et al. FORMULA0 ; BID31 ; BID23 investigate MCOs in sequential models based on the unbiased estimator produced by Sequential Monte Carlo.

Many unbiased estimators can be justified as performing simple importance sampling on an extended state space (e.g., Hamiltonian Importance Sampling BID33 , Annealed Importance Sampling BID32 , and Sequential Monte Carlo BID12 BID28 ).

In other words, we can define auxiliary variables ?? and distributions q(??|x), q(z|??, x), r(??|z, x) such thatp DISPLAYFORM2 with z, ?? ??? q(z, ??|x).

It immediately follows that the estimator is unbiased and leads to a variational bound Eq. (2).

Viewing recent improvements in variational bounds as augmenting variational families with latent variables allows us to apply the tools of auxiliary variable variational inference to understand the tradeoffs and derivation of these algorithms.

This unified view suggests novel bounds and reveals implicit design choices that may be sub-optimal.

First, we explicitly work through an example with the IWAE bound.

BID2 introduced the idea of viewing IWAE as auxiliary variable variational inference and Naesseth et al. FORMULA0 ; BID9 ; BID11 formalized the notion.

Consider the variational family defined by first sampling a set of K candidate z i s from a proposal distributionq(z i |x), and then sampling z from the empirical distribution composed of atoms located at each z i and weighted proportionally to p(x, z i )/q(z i |x).

In this case, the auxiliary latent variables ?? are the locations of the proposal samples z 1:K and the index of the selected sample, i.

Explicitly, let w i = p(x, z i )/q(z i |x).

Then choosing the generalized densities of q and r as DISPLAYFORM0 DISPLAYFORM1 yields the IWAE bound Eq. (3) when plugged into to Eq. (2) (see Appendix A for details).From Eq. (2), it is clear that IWAE is a lower bound on the standard ELBO for q(z|x) and the gap is due to D KL (q(z 1:K , i|z, x)||r(z 1:K , i|z, x)).

The choice of r(z 1:K , i|z, x) in Eq. FORMULA6 was for convenience and is suboptimal.

The optimal choice of r is DISPLAYFORM2 Compared to the optimal choice, Eq. (5) makes the approximation q(z ???i |i, z, x) ??? j =iq (z j |x) which ignores the influence of z on z ???i and the fact that z ???i are not independent given z. A simple extension could be to learn a factored variational distribution conditional on z DISPLAYFORM3 Learning such an r could improve the tightness of the bound, and we plan to explore this in future work.

More generally, many of the recent improvements in variational bounds (e.g., BID28 BID31 BID23 BID42 BID30 BID40 ) can be viewed as importance sampling on an extended state space.

By making the choice of r explicit, the gap between the bound and the ELBO bound with the marginalized variational distribution is clear and this can reveal novel choices for r.

In Section 3.1, we showed how IWAE uses self-normalized importance sampling to expand the family of q. Analogously, we can develop a generative model based on self-normalized importance sampling.

This model draws samples from a proposal ??(x), weights them according to a potential function U (x), and then draws a sample from the empirical distribution formed by the weighted samples.

We define the self-normalized importance sampling (SNIS) generative process in Algorithm 1 and denote the density of the process by p SN IS (x).

The marginal log-likelihood, log p SN IS (x), can be lower bounded as DISPLAYFORM0 for details see Appendix B. To summarize, p SN IS (x) can be sampled from exactly and has a tractable lower bound on its log-likelihood.

As K ??? ???, p SN IS (x) becomes proportional to ??(x) exp(U (x)).

For finite K, p SN IS (x) interpolates between the tractable ??(x) and the energy model ??(x) exp(U (x)).

Interestingly, log ??(x) only shows up once in the lower bound, and simply lower-bounding it still gives a lower bound on log p SN IS (x).

This expands the class of allowable distributions for the proposal ?? to include Variational Autoencoders (VAEs) BID20 BID38 .To train the SNIS generative model, we can perform stochastic gradient ascent on Eq. (6) with respect to the parameters of the proposal distribution ?? and the potential function U .

When the data

Require: Proposal distribution ??(x) and potential function U (x).

DISPLAYFORM0 Sample x k ??? ??(x).

Compute w(x k ) = exp(U (x k )).

4: end for DISPLAYFORM0 x are continuous, reparameterization gradients can be used to estimate the gradients to the proposal distribution BID38 BID20 .

When the data are discrete, score function gradient estimators such as REINFORCE BID41 or relaxed gradient estimators such as the Gumbel-Softmax BID27 BID16 can be used.

Simple importance sampling scales poorly to high dimensions, so it is natural to consider augmenting the generative model with latent variables from Hamiltonian Monte Carlo or more complex samplers.

We are currently exploring this.

Equation FORMULA9 is closely connected with the ranking NCE loss BID25 , a popular objective for training energy based models.

In fact, if we consider ??(x) as our noise distribution p N (x) and set U (x) =?? (x) ??? log p N (x), then up to a constant, we recover the ranking NCE loss.

The ranking NCE loss is motivated by the fact that it is a consistent objective for any K > 1 when the true data distribution is in our model family.

As a result, it is straightforward to adapt the consistency proof from BID25 to our setting.

Furthermore, our perspective gives a coherent objective for jointly learning the noise distribution and the potential function and shows that the ranking NCE loss can be viewed as a lower bound on the log likelihood of a well-specified model regardless of whether the true data distribution is in our model family.

Moreover, this distribution provides a novel perspective on Contrastive Predictive Coding BID35 , a recent approach to bounding mutual information for representation learning.

Starting from the well-known variational bound on mutual information due to BID3 DISPLAYFORM0 for a variational distribution q(x|y), we can use the self-normalized importance sampling distribution and choose the proposal to be p(x) (i.e., p SN IS(p,U ) ).

Applying the bound in Eq. FORMULA9 , we have DISPLAYFORM1 This recovers the CPC bound and proves that it is indeed a lower bound on mutual information whereas the justification in the original paper relied on approximations.

We evaluated generative models based on self-normalized importance sampling (SNIS) on a small, synthetic dataset as well as the MNIST dataset.

To provide a competitive baseline, we use the recently developed Learned Accept/Reject Sampling (LARS) model BID4 .

LARS trains a proposal distribution and an acceptance function (analogous to our potential function), which are used to perform rejection sampling.

The output of the rejection sampling process is the generated sample.

Such a process is attractive because unbiased gradients of its log likelihood can be easily computed without knowing the normalizing constant.

To ensure a sample can be generated in finite time, LARS truncates the rejection sampling after a set number of steps.

Unfortunately, this change requires estimating a normalizing constant.

In practice, BID4 estimate the normalizing constant using 1024 samples during training and 10 10 samples during evaluation.

Even so, LARS requires additional implementation tricks (e.g., evaluating the target density, using an exponential moving average to estimate the normalizing constant) to ensure successful training, which complicate the implementation and analysis of the algorithm.

On the other hand, SNIS is well-specified and has a tractable log likelihood lower bound for any K. As a result, no implementation tricks are necessary to train SNIS models.

Moreover, SNIS weights and uses all samples instead of choosing a single sample, which we expect to be advantageous.

Comparing the performance of LARS and SNIS on synthetic data.

Both LARS and SNIS achieve comparable data log-likelihood lower bounds, but SNIS does so much faster than LARS.

The results for LARS match previously-reported results in BID4 As a preliminary experiment, we reproduce the synthetic data experiment from BID4 which models a mixture of Gaussian densities.

The target distribution is a mixture of 9 equally-weighted Gaussian densities with variance 0.01 and means (x, y) ??? {???1, 0, 1} 2 .

Both LARS and SNIS used a fixed 2-D N (0, 1) proposal distribution and a learned acceptance/potential function U (x) parameterized by a neural network with 2 hidden layers of size 20 and tanh activations.

For both methods the number of proposal samples drawn, K, was set to 128.

We used batch sizes of 128 and ADAM BID19 with a learning rate of 3 ?? 10 ???4 to fit the models.

We plot the resulting densities and log-likelihood lower bounds in FIG0 .

As expected, SNIS quickly converges to the solution, and the potential function learns to cut out the mass between the mixture modes.

Next, we evaluated SNIS on modeling the MNIST handwritten digit dataset BID24 .

MNIST digits can be either statically or dynamically binarized -for the statically binarized dataset we used the binarization from BID39 , and for the dynamically binarized dataset we sampled images from Bernoulli distributions with probabilities equal to the continuous values of the images in the original MNIST dataset.

We tested two different model configurations: a VAE with an SNIS prior, and an SNIS model with a VAE proposal.

In the first case, the SNIS prior had a Gaussian proposal distribution, and in the second case, the VAE proposal had a Gaussian prior.

We chose hyperparameters to match the MNIST experiments in BID4 .

Specifically, we parameterized the SNIS potential function by a neural network with two hidden layers of size 100 and tanh activations, and parameterized the VAE observation model by neural networks with two layers of 300 units and tanh activations.

The latent spaces of the VAEs were 50-dimensional, and SNIS's K was set to 1024.

We also lin-early annealed the weight of the KL term in the ELBO from 0 to 1 over the first 1 ?? 10 5 steps and dropped the learning rate from 3 ?? 10 ???4 to 1 ?? 10 ???4 on step 1 ?? 10 6 .

All models were trained with ADAM BID19 .In the SNIS model with VAE proposal, we originally used the Straight-Through Gumbel estimator BID16 to estimate gradients through the discrete samples proposed by the VAE, but found that method performed worse than ignoring those gradients altogether.

We suspect that this may be due to bias in the gradients.

Thus, for the SNIS model with VAE proposal, we report numbers on training runs which ignore those gradients, and we plan to investigate unbiased gradient estimators in future work.

We summarize log-likelihood lower bounds on the test set in Table 1 .

We found that SNIS outperformed LARS even though it used only 1024 samples for training and evaluation, whereas LARS used 1024 samples during training and 10 10 samples for evaluation.

In this paper, we viewed recent work on improving variational bounds through the lens of auxiliary variable variational inference.

This perspective allowed us to expose suboptimal choices in existing algorithms such as IWAE, unify analysis of other methods such as ranking NCE and CPC, and derive new methods for generative modeling such as SNIS.

We plan to further develop this view by embedding methods such as Hamiltonian Importance Sampling and Annealed Importance Sampling in generative models which we expect to scale better with dimension of the data space.

Published as a workshop paper at ICLR 2019Then, plugging Eqs. (8) and (9) into Eq. (7) with ?? = (z 1:K , i) gives log p(x) ??? E q(z,??|x) log p(x, z)r(??|z, x) q(z, ??|x) = E q(??|x) log p(x, z i )

@highlight

Monte Carlo Objectives are analyzed using auxiliary variable variational inference, yielding a new analysis of CPC and NCE as well as a new generative model.

@highlight

Proposes a different view on improving variational bounds with auxiliary latent variable models and explores the use of those models in the generative model.
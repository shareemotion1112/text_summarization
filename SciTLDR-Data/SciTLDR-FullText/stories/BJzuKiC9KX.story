Discrete latent-variable models, while applicable in a variety of settings, can often be difficult to learn.

Sampling discrete latent variables can result in high-variance gradient estimators for two primary reasons: 1) branching on the samples within the model, and 2) the lack of a pathwise derivative for the samples.

While current state-of-the-art methods employ control-variate schemes for the former and continuous-relaxation methods for the latter, their utility is limited by the complexities of implementing and training effective control-variate schemes and the necessity of evaluating (potentially exponentially) many branch paths in the model.

Here, we revisit the Reweighted Wake Sleep (RWS; Bornschein and Bengio, 2015) algorithm, and through extensive evaluations, show that it circumvents both these issues, outperforming current state-of-the-art methods in learning discrete latent-variable models.

Moreover, we observe that, unlike the Importance-weighted Autoencoder, RWS learns better models and inference networks with increasing numbers of particles, and that its benefits extend to continuous latent-variable models as well.

Our results suggest that RWS is a competitive, often preferable, alternative for learning deep generative models.

Learning deep generative models with discrete latent variables opens up an avenue for solving a wide range of tasks including tracking and prediction BID28 , clustering BID31 , model structure learning BID0 , speech modeling BID16 , topic modeling BID1 , language modeling BID4 , and concept learning BID17 BID20 .

Furthermore, recent deep-learning approaches addressing counting BID6 , attention BID37 , adaptive computation time BID9 , and differentiable data structures BID9 BID11 , underscore the importance of models with conditional branching induced by discrete latent variables.

Current state-of-the-art methods optimize the evidence lower bound (ELBO) based on the importance weighted autoencoder (IWAE) BID3 by using either reparameterization BID19 BID32 , continuous relaxations of the discrete latents BID15 or the REINFORCE method BID36 with control variates BID24 BID25 BID12 BID35 BID8 .Despite the effective large-scale learning made possible by these methods, several challenges remain.

First, with increasing number of particles, the IWAE ELBO estimator adversely impacts inference-network quality, consequently impeding learning of the generative model .

Second, using continuous relaxations results in a biased gradient estimator, and in models with stochastic branching, forces evaluation of potentially exponential number of branching paths.

For example, a continuous relaxation of the cluster identity in a Gaussian mixture model (GMM) (Section 4.3) forces the evaluation of a weighted average of likelihood parameters over all clusters instead of selecting the parameters based on just one.

Finally, while control-variate methods may be employed to reduce variance, their practical efficacy can be somewhat limited as in some cases they involve designing and jointly optimizing a separate neural network which can be difficult to tune (Section 4.3).To address these challenges, we revisit the reweighted wake-sleep (RWS) algorithm BID2 , comparing it extensively with state-of-the-art methods for learning discrete latent-variable models, and demonstrate its efficacy in learning better generative models and inference networks, and improving the variance of the gradient estimators, over a range of particle budgets.

Going forward, we review the current state-of-the-art methods for learning deep generative models with discrete latent variables (Section 2), revisit RWS (Section 3), and present an extensive evaluation of these methods (Section 4) on (i) the Attend, Infer, Repeat (AIR) model BID6 to perceive and localise multiple MNIST digits, (ii) a continuous latent-variable model on MNIST, and (iii) a pedagogical GMM example, exposing a shortcoming of RWS that we fix using defensive importance sampling BID13 .

Our experiments confirm that RWS is a competitive, often preferable, alternative that unlike IWAE, learns better models and inference networks with increasing particle budgets.

Consider data (x (n) ) N n=1 sampled from a true (unknown) generative model p(x), a family of generative models p θ (z, x) of latent variable z and observation x parameterized by θ and a family of inference networks q φ (z|x) parameterized by φ.

We would like to learn the generative model by maximizing the marginal likelihood over data: DISPLAYFORM0 .

We would simultaneously also like to learn an inference network q φ (z|x) that amortizes inference given observation x; i.e., q φ (z|x) maps an observation x to an approximation of p θ * (z|x).

Amortization ensures that evaluation of this function is cheaper than performing approximate inference of p θ * (z|x) from scratch.

Our focus here is on such joint learning of the generative model and the inference network, here referred to as "learning a deep generative model", although we note that other approaches exist that learn just the generative model BID7 BID26 or the inference network BID29 BID21 in isolation.

We begin with a review of IWAEs BID3 as a general approach for learning deep generative models using stochastic gradient descent (SGD) methods, focusing on generative-model families with discrete latent variables, for which the high variance of the naïve gradient estimator impedes learning.

We will also review control-variate and continuous-relaxation methods for gradient-variance reduction.

The IWAE used alongside such gradient-variance reduction methods is currently the dominant approach for learning deep generative models with discrete latent variables.

BID3 introduced the IWAE which maximizes an average of ELBOs over data,

, where, given number of particles K, DISPLAYFORM0 When K = 1, this reduces to the variational autoencoder (VAE) BID19 BID32 .

BID3 show that ELBO K IS (θ, φ, x) is a lower bound on log p θ (x) and that increasing K leads to a tighter lower bound.

Further, tighter lower bounds arising from increasing K improve learning of the generative model, but worsen learning of the inference network , as the signal-to-noise ratio of θ's gradient estimator is O( DISPLAYFORM1 Moreover, poor learning of the inference network, beyond a certain point (large K), can actually worsen learning of the generative model as well; a finding we explore in Section 4.3.Optimizing the IWAE objective using SGD methods requires unbiased gradient estimators of ELBO K IS (θ, φ, x) with respect to θ and φ BID33 .The θ gradient ∇ θ ELBO K IS (θ, φ, x) is estimated by sampling z 1:K ∼ Q φ (·|x) and evaluating DISPLAYFORM2 is estimated similarly for models with reparameterizable latents, discrete (and other non-reparameterizable) latents require the REINFORCE gradient estimator BID36 DISPLAYFORM3

Since the gradient estimator in Eq. (2) can often suffer from high variance, mainly due to the effect of 1 , a number of approaches have been developed to ameliorate the issue.

When employing continuous relaxation methods, all discrete latent variables in the model can be replaced by the Concrete or Gumbel-Softmax BID15 distribution, whose gradients can be approximated using the reparameterization trick BID19 BID32 .

The main practical difficulty here is tuning the temperature parameter: low temperatures reduce the gradient estimator bias, but rapidly increase its variance.

Moreover, since such relaxations are defined on a simplex, applying them to models that exhibit branching requires the computationally expensive evaluation of all (potentially exponentially many) paths in the generative model.

Variational inference for Monte Carlo objectives (VIMCO) BID25 ) is a method that doesn't require designing an explicit control variate, as it exploits the particle set obtained in IWAE.It replaces 1 with DISPLAYFORM0 where term B is independent of z and highly correlated with term A.Finally, assuming z k is a discrete random variable with C categories 1 , REBAR BID35 ) and RELAX BID8 improve on the methods of BID24 and BID12 , and replaces 1 with DISPLAYFORM1 where g k is a C-dimensional vector of reparameterized Gumbel random variates, z k is a one-hot argmax function of g k , andg k is a vector of reparameterized conditional Gumbel random variates conditioned on z k .

The conditional Gumbel random variates are a form of Rao-Blackwellization used to reduce variance.

The control variate c ρ , parameterized by ρ, is optimized to minimize the gradient variance estimates concurrently with the main ELBO optimization, leading to state-of-theart performance on, for example, sigmoid belief networks BID27 .

The main practical difficulty in using this method is choosing a suitable family of c ρ , as some choices lead to higher variance despite the concurrent gradient variance minimization.

Moreover, the objective for the concurrent optimization requires evaluating a Jacobian-vector product that induces an overhead of O(D φ D ρ ) where D φ , D ρ are number of inference network and control-variate parameters respectively.

Reweighted wake-sleep (RWS) BID2 comes from another family of algorithms for learning deep generative models, eschewing a single objective over parameters θ and φ in favour of individual objectives for each.

We review the RWS algorithm and discuss its advantages and disadvantages.

Reweighted wake-sleep (RWS) BID2 is an extension of the wake-sleep algorithm both of which, like IWAE, jointly learn a generative model and an inference network given data.

While IWAE targets a single objective, RWS alternates between objectives, updating the generative model parameters θ using a wake-phase θ update and the inference network parameters φ using either a sleep-or a wake-phase φ update (or both).Wake-phase θ update.

Given a current value of φ, θ is updated using an unbiased estimate of DISPLAYFORM0 , which can be obtained directly without needing reparameterization or control variates as the sampling distribution Q φ (·|x) is independent of θ.2 Sleep-phase φ update.

Here, φ is updated to maximize the negative Kullback-Leibler (KL) divergence between the posteriors under the generative model and the inference network, averaged over the data distribution of the current generative model DISPLAYFORM1 The gradient of this objective is E p θ (z,x) [∇ φ log q φ (z|x)] and can be estimated by sampling z, x from the generative model p θ (z, x) and evaluating ∇ φ log q φ (z|x).

The variance of such an estimator can be reduced at a standard Monte Carlo rate by increasing the number of samples of z, x.

Wake-phase φ update.

Here, φ is updated to maximize the negative KL divergence between the posteriors under the generative model and the inference network, averaged over the true data distribution DISPLAYFORM2 The outer expectation of the gradient DISPLAYFORM3 can be estimated using a single sample x from the true data distribution p(x), given which, the inner expectation can be estimated using a self-normalized importance sampler with K particles using q φ (z|x) as the proposal distribution.

This results in the following estimator DISPLAYFORM4 where DISPLAYFORM5 , in a similar fashion to Eq. (1).

Note that equation 5 is negative of the second term of the REINFORCE estimator of the IWAE ELBO in equation 2.

The crucial difference of the wake-phase φ update to the sleep-phase φ update is that the expectation in Eq. FORMULA10 is over the true data distribution p(x) unlike the expectation in Eq. FORMULA9 which is under the current model distribution p θ (x).

The former is desirable from the perspective of amortizing inference over data from p(x).

Although the estimator in Eq. FORMULA12 is biased, this bias decreases as K increases.

While BID2 refer to the use of the sleep-phase φ update followed by an equally-weighted mixture of the wake-phase and sleep-phase updates of φ, we here use RWS to refer to the variant that employs just the wake-phase φ update, and not the mixture.

The rationale for our preference will be made clear from the empirical evaluations (Section 4).

While the gradient update of θ targets the same objective as IWAE, the gradient update of φ targets the objective in Eq. (3) in the sleep case and Eq. (4) in the wake case.

This leads to two advantages for RWS over IWAE.

First, since we don't need to use REINFORCE, using RWS leads to much lower variance of gradient estimators of φ.

Second, the φ updates in RWS directly target minimization of the expected KL divergences from true to approximate posteriors.

Increasing the computational budget (using more Monte Carlo samples in the sleep-phase φ update case and higher number of particles K in the wake-phase φ update) results in a better estimator of these expected KL divergences.

This is different to IWAE, where optimizing ELBO K IS targets a KL divergence on an extended sampling space (Le et al., 2018) which for K > 1 doesn't correspond to a KL divergence between true and approximate posteriors (in any order).

Consequently, increasing number of particles in IWAE leads to worse learning of inference networks .

The objective of the sleep-phase φ update in equation 3 is an expectation of KL under the current model distribution p θ (x) rather than the true one p(x).

This makes the sleep-phase φ update suboptimal since the inference network must always follow a "doubly moving" target (both p θ (x) and p θ (z|x) change during the optimization).

The IWAE and RWS algorithms have primarily been applied to problems with continuous latent variables and/or discrete latent variables that do not actually induce branching such as sigmoid belief networks BID27 .

The purpose of the following experiments is to compare RWS to IWAE used alongside the control variate and continuous relaxation methods described in Section 3 on models with conditional branching, where, as we will show, the various control-variate schemes underperform in relation to RWS.

In several ways, including ELBOs achieved and average distance between true and amortized posteriors, we empirically demonstrate that increasing the number of particles K hurts learning in IWAE but improves learning in RWS.The first experiment, using the deep generative model from Attend, Infer, Repeat (AIR) BID6 , demonstrates better learning of the generative model in a model containing both discrete latent variables used for branching as well as continuous latent variables in a complex visual data domain (Section 4.1).

The next experiment on MNIST (Section 4.2) does so in a model with continuous latent variables.

Finally, a GMM experiment (Section 4.3) serves as a pedagogical example to understand sources of advantage for RWS in more detail.

Notationally, the different variants of RWS will be referred to as wake-sleep (WS) and wake-wake (WW).

The wake-phase θ update is always used.

We refer to using it in conjunction with the sleep-phase φ update as WS and using it in conjunction with the wake-phase φ update as WW.

We tried using both wake-and sleep-phase φ updates however, in addition to doubling the amount of sampling, found that doing so only improves performance on the continuous latent variable model.

The number of particles K used for the wake-phase θ and φ updates will always be specified.

The sleep phase φ update will also use K samples from p θ (z, x).

First, we evaluate WW and VIMCO on AIR (Eslami et al., 2016), a structured deep generative model with both discrete and continuous latent variables.

AIR uses the discrete variable to decide how many continuous variables are necessary to explain an image (see supplementary material for details).

The sequential inference procedure of AIR poses a difficult problem, since it implies a sequential decision process with possible branching.

See BID6 for the details of the model (see supplementary material for the model in our notation).We set the maximum number of inference steps in AIR to three and we train it on images of size 50× 50 with zero, one or two MNIST digits.

The training and testing data sets consist of 60000 and 10000 images, respectively, which are generated from the respective MNIST train/test datasets.

Unlike AIR, which used Gaussian likelihood with fixed standard deviation and continuous inputs (i.e., input x ∈ [0, 1] 50×50 ), we use a Bernoulli likelihood and binarized data; the stochastic binarization is the same as in BID3 .

This choice is motivated by initial experiments, which have shown that the original setup is detrimental to the sleep phase of WS -samples from the generative model did not look similar to true data even after the training has converged.

Training is performed over two million iterations by RmsProp BID34 with the learning rate of 10 −5 , which is divided by three after 400k and 1000k training iterations.

We set the glimpse size to 20 × 20.We first evaluate the generative model via the average test log marginal where each log marginal is estimated by a one-sample, 5000-particle IWAE estimate.

The inference network is then evaluated via the average test KL from the inference network to the posterior under the current model where each D KL (q φ (z|x), p θ (z|x)) is estimated as a difference between the log marginal estimate above and a 5000-sample, one-particle IWAE estimate.

Note that this KL estimate is merely a proxy to the desired KL from the inference network to the posterior under the true model.

This experiment confirms FIG1 ) that increasing number of particles hurts learning in VIMCO but improves learning in WW.

Increasing K improves WW monotonically but VIMCO only up to a point.

WW also results in significantly lower variance and better inference networks than VIMCO.

RWS has typically only been considered for learning models with discrete latent variables.

In order to gauge its applicability to a wider class of problems, we evaluate it on a variational autoencoder with normally distributed latent variables for MNIST.

To do this, we use the training procedure and the model with a single stochastic layer of BID3 and their stochastic binarization of data.

with the batch size of 32.

We evaluate performance in the same way as we evaluate the AIR model.

Additionally, we evaluate the number of active latent units BID3 .

This experiment confirms ( FIG2 and TAB0 ) that increasing number of particles hurts learning in IWAE but improves learning in WW.

Increasing K does improve the marginal likelihood attained by WW and IWAE with reparameterization.

However, the latter learns a better model only up to a point (K = 128) -further increase in the number of particles has diminishing returns.

WW also results in better inference networks than IWAE as showed by the KL plot on the right of FIG2 To see what is going on, we study a GMM which branches on a discrete latent variable to select cluster assignments.

The generative model and inference network are defined as DISPLAYFORM0 where z ∈ {0, . . .

, C −1}, C is the number of clusters and µ c , σ Note that KL is between the inference network and the current generative model.

Quality of the generative model: WS and WW improve with larger particle budget thanks to lower variance and lower bias estimators of the gradient respectively.

IWAE methods suffer with a larger particle budget .

WS performs the worst as a consequence of computing the expected KL under model distribution p θ (x) equation 3 instead of the true data distribution p(x) as with WW equation 4.

WW suffers from zero-forcing (described in text) in low-particle regimes, but learns the best model fastest in the many-particle regime; δ-WW additionally learns well in the low-particle regime. (Middle) The quality of the inference network develops identically to that of the generative model. (Bottom) WW and WS have lower-variance gradient estimators of φ than IWAE, as they don't include the high-variance term 1 in equation 2.

This is a necessary, but not sufficient, condition for efficient learning with other important factors being gradient direction and the ability to escape local optima.

under the true model.

The true model is set to p θtrue (x) where softmax(θ true ) c = (c+5)/ C i=1 (i+5) (c = 0, . . .

, C − 1), i.e. the mixture probabilities are linearly increasing with the z FIG0 .

We fix the mixture parameters in order to study the important features of the problem at hand in isolation.

DISPLAYFORM1 IWAE was trained with REINFORCE, RELAX, VIMCO and the Concrete distribution.

We also train using WS and WW.

We fix C = 20 and increase number of particles from K = 2 to 20.

We use the Adam optimizer with the learning rate 10 −3 and default β parameters.

At each iteration, a batch of 100 data points is generated from the true model to train.

Having searched over several temperature schedules for the Concrete distribution, we use the one with the lowest trainable terminal temperature (linearly annealing from 3 to 0.5).

We found that using the control variate c ρ (g 1:K ) =(1 + C)-16-16-1 (with tanh nonlinearity) led to most stable training (see supplementary material for more details).We evaluate the generative model, inference network and the variance of the gradient estimator of φ.

The generative model is evaluated via the L2 distance between the probability mass functions (PMFs) of its prior and true prior as softmax(θ) − softmax(θ true ) .

The inference network is evaluated via the L2 distance between PMFs of the current and true posteriors, averaged over a fixed set (M = 100) of observations (x (m) test ) M m=1 from the true model: DISPLAYFORM2 is the dth element of one of φ's gradient estimators (e.g. equation 2 for REINFORCE) and std(·) is estimated using 10 samples.

Here, we demonstrate that using WS and WW with larger particle budgets leads to a better inference networks whereas this is not the case for IWAE methods FIG3 .

Recall that the former is because using more samples to estimate the gradient of the sleep φ objective equation 3 for WS reduces variance at a standard Monte Carlo rate and that using more particles in equation 5 to estimate the gradient of the wake φ objective results in a lower bias.

The latter is because using more particles results in the signal-to-noise of IWAE's φ gradient estimator to drop at the rate O(1/ √ K) .Learning of the generative model, as a consequence of inference-network learning, is also better for WS and WW, but worse for IWAE methods when the particle budget is increased.

This is because the the θ gradient estimator (common to all methods), ∇ θ ELBO K IS (θ, φ, x) can be seen as an importance sampling estimator whose quality is tied to the proposal distribution (inference network).WW and WS have lower variance gradient estimators than IWAE, even if used with control-variate and continuous-relaxation methods.

This is because φ's gradient estimators for WW and WS don't include the high-variance term 1 in equation 2.

This is a necessary but not sufficient condition for efficient learning with other important factors being gradient direction and the ability to escape local optima (explored below).

Employing the Concrete distribution gives low-variance gradients for φ to begin with, but the model learns poorly due to the high gradients bias (due to high temperature hyperparameter).

We now describe a failure mode, that affects WS, WW, VIMCO and REINFORCE, which we will refer to as zero-forcing.

It is best illustrated by inspecting the generative model and the inference network as the training progresses, focusing on the low-particle (K = 2) regime FIG4 ).

For WS, the generative model p θ (z) peaks at z = 9 and puts zero mass for z > 9; the inference network q φ (z|x) becomes the posterior for this model which, in this model, also has support at most {0, . . .

, 9} for all x. This is a local optimum for WS because (i) the inference network already approximates the posterior of the model p θ (z, x) well, and (ii) the generative model p θ (z), being trained using samples from q φ (z|x), has no samples outside of its current support ({0, . . .

, 9}).

Similar failure mode occurs for WW and VIMCO/REINFORCE although the support of the locally optimal p θ (z) is larger ({0, . . .

, 14} and {0, . . . , 17} respectively).

DISPLAYFORM3 While this failure mode is a particular feature of the GMM, we hypothesize that WS and WW suffer from it more, as they alternate between two different objectives for optimizing θ and φ.

WS attempts to amortize inference for the current model distribution p θ (x) which reinforces the coupling between the generative model and the inference network, making it easier to get stuck in a local optimum.

WW with few particles (say K = 1) on the other hand, results in a highly-biased gradient estimator equation 5 that samples z from q φ (·|x) and evaluates ∇ φ log q φ (z|x); this encourages the inference network to concentrate mass.

This behavior is not seen in WW with many particles where it is the best algorithm at learning both a good generative model and inference network ( FIG3 , right).We propose a simple extension of WW, denoted δ-WW, that mitigates this shortcoming by changing the proposal of the self-normalized importance sampling estimator in Eq. (5) to q φ,δ (z|x) = (1 − δ)q φ (z|x) + δUniform(z).

We use δ = 0.2, noting that the method is robust to a range of values.

Using a different proposal than the inference network q φ (z|x) means that using the lowparticle estimator in Eq. (5) no longer leads to zero-forcing.

This is known as defensive importance sampling BID13 , and is used to better estimate integrands that have long tails using short-tailed proposals.

Using δ-WW outperforms all other algorithms in learning both the generative model and the inference network.

Our experiments suggest that RWS learns both better generative models and inference networks in models that involve discrete latent variables, while performing just as well as state-of-the-art on continuous-variable models as well.

The AIR experiment (Section 4.1) shows that the trained inference networks are unusable when trained with high number of particles.

Moreover, the MNIST experiment (Section 4.2) suggests that RWS is competitive even on models with continuous latent variables, especially for high number of particles where IWAE ELBO starts suffering from worse inference networks.

The GMM experiment (Section 4.3) illustrates that this is at least at least in part due to a lower variance gradient estimator for the inference network and the fact that for RWSunlike the case of optimizing IWAE ELBO )-increasing number of particles actually improves the inference network.

In the low-particle regime, the GMM suffers from zeroforcing of the generative model and the inference network, which is ameliorated using defensive RWS.

Finally, all experiments show that, beyond a certain point, increasing the particle budget starts to affect the quality of the generative model for IWAE ELBO whereas this is not the case for RWS.

As a consequence of our findings, we recommend reconsidering using RWS for learning deep generative models, especially those containing discrete latent variables that induce branching.

@highlight

Empirical analysis and explanation of particle-based gradient estimators for approximate inference with deep generative models.
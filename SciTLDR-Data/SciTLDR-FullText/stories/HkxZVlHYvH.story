Statistical inference methods are fundamentally important in machine learning.

Most state-of-the-art inference algorithms are  variants of Markov chain Monte Carlo (MCMC) or variational inference (VI).

However, both methods struggle with limitations in practice: MCMC methods can be computationally demanding; VI methods may have large bias.

In this work, we aim to improve upon MCMC and VI by a novel hybrid method based on the idea of reducing simulation bias of finite-length MCMC chains using gradient-based optimisation.

The proposed method can generate low-biased samples by increasing the length of MCMC simulation and optimising the MCMC hyper-parameters, which offers attractive balance between approximation bias and computational efficiency.

We show that our method produces promising results on popular benchmarks when compared to recent hybrid methods of MCMC and VI.

Statistical inference methods in machine learning are dominated by two approaches: simulation and optimisation.

Markov chain Monte Carlo (MCMC) is a well-known simulation-based method, which promises asymptotically unbiased samples from arbitrary distributions at the cost of expensive Markov simulations.

Variational inference (VI) is a well-known method using optimisation, which fits a parametric approximation to the target distribution.

VI is biased but offers a computationally efficient generation of approximate samples.

There is a recent trend of hybrid methods of MCMC and VI to achieve a better balance between computational efficiency and bias.

Hybrid methods often use MCMC or VI as an algorithmic component of the other.

In particular, Salimans et al. (2015) proposed a promising modified VI method that reduces approximation bias by using MCMC transition kernels.

Another technique reduces the computational complexity of MCMC by initialising the Markov simulation from a pretrained variational approximation (Hoffman, 2017; Han et al., 2017) .

Levy et al. (2018) proposed to improve MCMC using flexible non-linear transformations given by neural networks and gradientbased auto-tuning strategies.

In this work, we propose a novel hybrid method, called ergodic inference (EI).

EI improves over both MCMC and VI by tuning the hyper-parameters of a flexible finite-step MCMC chain so that its last state sampling distribution converges fast to a target distribution.

EI optimises a tractable objective function which only requires to evaluate the logarithm of the unnormalized target density.

Furthermore, unlike in traditional MCMC methods, the samples generated by EI from the last state of the MCMC chain are independent and have no correlations.

EI offers an appealing option to balance computational complexity vs. bias on popular benchmarks in machine learning.

Compared with previous hybrid methods, EI has following advantages:

• EI's hyperparameter tuning produces sampling distributions with lower approximation bias.

• The bias is guaranteed to decrease as the length of the MCMC chain increases.

• By stopping gradient computations, EI has less computational cost than related baselines.

We also state some disadvantages of our method:

• The initial state distribution in EI's MCMC chain has to have higher entropy than the target.

• The computational complexity per simulated sample of EI is in general higher than in VI.

Monte Carlo (MC) statistical inference approximates expectations under a given distribution using simulated samples.

Given a target distribution π, MC estimations of an expectation E π [f (x)] are defined as empirical average of the evaluation of f on samples from π.

To generate samples from π, we assume that the unnormalized density function π * (x) can be easily computed.

In a Bayesian setting we typically work with π * (x|y) given by the product of the prior p(x) and the likelihood p(y|x), where y denotes observed variables and x denotes the model parameters specifying p(y|x).

Markov chain Monte Carlo (MCMC) casts inference as simulation of ergodic Markov chains that converge to the target π.

The MCMC kernel M (x |x) is characterised by the detailed balance (DB) property: π(x)M (x |x) = π(x )M (x|x ).

Given an unnormalised target density π * , an MCMC kernel can be constructed in three steps: first, sample an auxiliary random variable r from an auxiliary distribution q φ1 with parameters φ 1 ; second, create a new candidate sample as (x , r ) = f φ2 (x t−1 , r), where f φ2 is a deterministic function with parameters φ 2 ; finally, accept the proposal as x t = x with probability p MH = min {0, π * (x )q φ1 (r )/[π * (x t−1 )q φ1 (r)]}, otherwise duplicate the previous sample as x t = x t−1 .

The last step is well known in the literature as the Metropolis-Hastings (M-H) correction step (Robert & Casella, 2005) and it results in MCMC kernels that satisfy the DB condition.

In the following, we denote the joint MCMC parameters (φ 1 , φ 2 ) by φ.

If f φ2 does not preserve volume, then it requires a Jacobian correction factor in the ratio in p M H .

Hamiltonian Monte Carlo (HMC) is a successful MCMC method which has drawn great attention.

A few recent works based on this method are Salimans et al. (2015) ; Hoffman (2017); Levy et al. (2018) .

In HMC, the auxiliary distribution q φ1 is often chosen to be Gaussian with zero-mean and a constant diagonal covariance matrix specified by φ 1 .

The most common f φ2 in HMC is a numeric integrator called the leapfrog algorithm, which simulates Hamiltonian dynamics defined by log π * (Neal, 2010).

The leapfrog integrator requires the gradient of log π * and a step size parameter given by φ 2 .

Given any initial state x 0 , MCMC can generate asymptotically unbiased samples x 1:n .

For this, MCMC iteratively simulates the next sample x t through the application of the MCMC transition kernel to the previous sample x t−1 .

It is well known in the literature that MCMC is computationally demanding (MacKay, 2002; Bishop, 2006) .

In particular, it is often necessary to run sufficiently long burn-in MCMC simulations to reduce simulation bias.

Another drawback of MCMC is sample correlation, which increases the variance of the MC estimator (Neal, 2010) .

To avoid strong sample correlation, the common practice in MCMC to tune hyper-parameters manually using sample quality metrics like effective sample size, (Hoffman, 2017; Robert & Casella, 2005) , which has been developed into automated gradient-based tuning strategies in recent work (Levy et al., 2018) .

Variational inference (VI) is a popular alternative to MCMC for generating approximate samples from π.

Unlike MCMC reducing sample bias by long burn-in simulation, VI casts the sample bias reduction as an optimisation problem, where a parametric approximate sampling distribution P is fit to the target π.

In particular, VI optimises the evidence lower bound (ELBO) given by

where

, also known as the entropy, must be tractable to compute.

L ELBO (P T π * ) is a lower bound on the log normalising constant log Z = log π * (x) dx.

This bound is tight when P = π.

therefore, the approximation bias in VI can be defined as the gap between L ELBO (P π * ) and log Z, that is,

where D KL (P π) denotes the Kullback-Leibler (KL) divergence.

Variational approximations often belong to simple parametric families like the multivariate Gaussian distribution with diagonal covariance matrix.

This results in computationally efficient algorithms for bias reduction and sample generation, but may also produce highly biased samples in cases of over-simplified approximation that ignores correlation.

Designing variational approximation to achieve low bias under the constraint of tractable entropy and efficient sampling procedures is possible using flexible distributions parameterised by neural networks (NNs) (Rezende & Mohamed, 2015; Kingma et al., 2016) .

However, how to design such NNs for VI is still a research challenge.

The balance between computational efficiency and bias is a challenge at the heart of all inference methods.

MCMC represents a family of simulation-based methods that guarantee low-bias samples at cost of expensive simulations; VI represents a family of optimisation-based methods that generate high-bias samples at a low computational cost.

Many recent works seek a better balance between efficiency and bias by combining MCMC and VI.

Salimans et al. (2015) proposed to reduce variational bias by optimising an ELBO specified in terms of the tractable joint density of short MCMC chains.

The idea seems initially promising, but the proposed ELBO becomes looser and looser as the chain grows longer.

Caterini et al. (2018) construct an alternative ELBO for HMC that still has problems since the auxiliary momentum variables are sampled only once at the beginning of the chain, which reduces the empirical performance of HMC.

Inspired by contrastive divergence, Ruiz & Titsias (2019) proposed a novel variational objective function to optimise variational parameters by adding additional term that minimise the KL between a MCMC distribution and variational approximation to reduce variational bias.

Hoffman (2017) and Han et al. (2017) proposed to replace expensive burn-in simulations in MCMC with samples from pre-trained variational approximations.

This approach is effective at finding good initial proposal distributions.

However, it does not offer a solution for tuning HMC parameters (Hoffman, 2017) , which are critical for good empirical performance.

Another line of research has focused on improving inference using flexible distributions, which are transformed from simple parametric distributions by non-linear non-volume preserving (NVP) functions.

Levy et al. (2018) proposed to tune NVP parameterised MCMC w.r.t.

a variant of the expected squared jumped distance (ESJD) loss proposed by Pasarica & Gelman (2010) .

proposed a similar auto-tuning for NVP parameterised MCMC using an adversarial loss.

Ergodic inference (EI) is motivated by the well-known convergence of MCMC chains (Robert & Casella, 2005) : MCMC chains converge in terms of the total variation (TV) distance between the marginal distribution of the MCMC chain and the target π.

Inspired by the convergence property of MCMC chains, we define an ergodic approximation P T to π with T MCMC steps as following.

Given a parametric distribution P 0 with tractable density p 0 (x 0 ; φ 0 ) parameterlized by φ 0 and an MCMC kernel M (x |x; φ) constructed using the unnormalised target density π * and with MCMC hyperparameter φ, an ergodic approximation of π is the marginal distribution of the final state of an T -step MCMC chain initialized from P 0 :

We call φ 0 and φ the ergodic parameters of P T .

Well known in MCMC literature like (Robert & Casella, 2005; Murray & Salakhutdinov, 2008) , the ergodic approximation p T converges to π after every MCMC transition and with sufficiently long chain p T is guaranteed to be arbitrarily close to π with arbitrary φ and φ 0 .

It is important to clarify that ergodic approximation is different from the modified variational methods like (Ruiz & Titsias, 2019) which only optimise the variational parameters φ 0 , but the optimisation objective functions involve MCMC similation.

In the following section, we show how EI can tune the ergodic parameters to minimise the bias of P T as an approximation to the target π with finite T .

To reduce the KL divergence D KL (P T π), one could tune the burn-in parameter φ 0 and the MCMC parameter φ by minimizing equation 2.

However, this is infeasible because we cannot analytically evaluate p T in equation 3.

Instead, we exploit the convergence of ergodic Markov chains and propose to optimise an alternative objective as the following constrained optimisation problem: max

where h is a hyperparameter that should be close to the entropy of the target, that is, h ≈ H(π).

We call the objective in equation 4 the ergodic modified lower bound (EMLBO), denoted by L(φ 0 , φ, π * ).

Note that the EMLBO is similar to L ELBO (P T π * ), with the intractable entropy H(P T ) replaced by the tractable L ELBO (P 0 π * ).

We now give some motivation for this constrained objective.

First, we explain the inclusion of the term L ELBO (P 0 π * ) in equation 4 and its connection to H(P T ).

If we maximised only the first term E p [log π * (x)] with respect to a fully flexible distribution P , the result would be a point probability mass at the mode of the target π.

This degenerate solution is avoided in VI by optimising the sum of E p [log π * (x)] and the entropy term H(P ), which enforces P to move away from a point probability mass.

However, H(P T ) is intractable in ergodic approximation.

Fotunately, we notice that maximising the term L ELBO (P 0 π * ) = E p0 [log π * (x)] + H(P 0 ) has similar effect of maximising H(P T ) for preventing P 0 from collapsing to the mode of π.

It is easy to show that P T cannot be a delta unless H(P T ) = −∞, which also implies L ELBO (P T π * ) does not exist.

Since the KL divergence D KL (P t π) never increases after each MCMC transition step (Murray & Salakhutdinov, 2008) ,

The constraint in equation 5 is necessary to eliminate the following pathology.

If P 0 does not satisfy

, we will favor P T to stay close to P 0 instead of making it converge to π faster.

This is illustrated by the plot in the right part of Figure 2 .

To avoid this pathological case, note that

It is interesting to compare the EMLBO with the objective function optimised by Salimans et al. (2015) , that is, the ELBO given by

where p(x 0:T −1 |x T ) denotes the conditional density of the first T states of the MCMC chain given the last one x T and r(x 0:T −1 |x T ) is an auxiliary variational distribution that approximates p(x 0:T −1 |x T ).

Note that the negative KL term in equation 6 will increase as T increases.

This makes the ELBO in equation 6 become looser and looser as the chain length increases.

In this case, the optimisation of equation 6 results in an MCMC sampler that fits well the biased inverse model r(x 0:T −1 |x T ) but whose marginal distribution for x T does not approximate π well.

This limits the effectiveness of this method in chains with multiple MCMC transitions.

By contrast, the EMLBO does not have this problem and its optimisation will produce a more and more accurate P T as T increases.

EI combines the benefits of MCMC and VI and avoids their drawbacks, as shown in Table 1 .

In particular, the bias in EI is reduced by using longer chains, as in MCMC, and EI generates independent samples, as in VI.

Futhermore, EI optimises an objective that directly quantifies the bias of the generated samples, as in VI.

Methods for tuning MCMC do not satisfy the latter and optimise instead indirect proxies for mixing speed, e.g. expected squared jumped distance (Levy et al., 2018) .

Importantly, EI can use gradients to tune different MCMC parameters at each step of the chain, as suggested by Salimans et al. (2015) .

This gives EI an extra flexibility which existing MCMC methods do not have.

Finally, EI is different from parallel-chain MCMC: while EI generates independent samples, parallel-chain MCMC draws correlated samples from several chains running in parallel.

We now show how to maximise the ergodic objective using gradient-based optimisation.

The gradient ∂ φ0,φ L(φ 0 , φ, π * ) is equal to the sum of two gradient terms.

The first one ∂ φ0 L ELBO (P 0 π * ) is affected by the constraint H(P 0 ) > h, while the second term

is not.

If we ignore the constraint, the first gradient term can be estimated by Monte Carlo using the reparameterization trick proposed in (D.P. Kingma, 2014; Rezende & Mohamed, 2015) :

where f φ0 (·) is a deterministic function that maps the random variable i sampled from a simple distribution, e.g. a factorized standard Gaussian, into the random variable x i 0 sampled from p 0 (·; φ 0 ).

To guarantee that our gradient-based optimiser yields a solution satisfying the constraint, we first initialize φ 0 so that H(P 0 ) > h and, afterwards, we force the gradient descent optimiser to leave φ 0 unchanged if H(P 0 ) is to get lower than h during the optimisation process.

The Monte Carlo estimation of ∂ φ E p T [log π * (x)] can also be computed using the reparameterization trick.

For this, the Metropolis-Hastings (M-H) correction step in the MCMC transitions, as described in Section 2.2, can be reformulated as applying the following transformation to x t−1 :

where

is an indicator function that takes value one if p MH > u and zero otherwise.

In Hamiltonian Monte Carlo (HMC), f φ is the leapfrog integrator of Hamiltonian dynamics with the leapfrog step size φ.

We define the T -times composition of g φ , given in equation 8, as the transformation x T = g T φ (x 0 , r 1:T ; u 1:T ).

Then, the second gradient term can be estimated by Monte Carlo as follows: T t=1 q(r t )Unif(u t ; 0, 1).

Note that the gradient term equation 9 is correct under the assumption f φ is volume-preserving in the joint space of (x t−1 , r t ), otherwise additional gradient term of the Jacobian of f φ w.r.t.

φ is required.

However, it is not a concern for many popular MCMC kernels.

For example, the leapfrog integrator in HMC f φ guarantees the preservation of volume as shown in (Neal, 2010) .

It is worth to mention that the indicator function in equation 8 is not continuous but differentiable almost everywhere.

Therefore, the gradient in equation 9 can be computed conveniently using standard autodifferentiation tools.

The gradient in equation 9 requires computing ∂ φ g T φ (x 0 , r 1:T ; u 1:T ), which can be done easily by using auto-differentiation and gradient backpropagation through the transfromations g φ (·, r t ; u t ) with t = T, . . .

, 1.

However, backpropagation in deep compositions can be computationally demanding.

We discovered a trick to accelerate the gradient computation by stopping the backpropagation of the gradient at the input x t−1 of g φ (x t−1 , r t ; u t ), for t = 1, . . .

, T .

Empirically this trick has almost no impact on the convergence speed of the ergodic approximation, as shown in Figure 2.

3.3 THE ENTROPY CONSTRAINT AND HYPERPARAMETER TUNING As mentioned previously, ignoring the constraint H(P 0 ) > H(π) may lead to pathological results when optimising the ergodic objective.

To illustrate this, we consider fitting an ergodic approximation given by a Hamilton Monte Carlo (HMC) transition kernel with T = 9.

P 9 denotes the initial ergodic approximation before traing and P * 9 denotes the same approximation after training.

The target distribution is a correlated bivariate Gaussian given by π = N (0, (2.0, 1.5; 1.5; 1.6)).

Samples from this distribution are shown in plot (a) in Figure 1 .

We optimise different a separate HMC parameter φ t , as described in Section 2.2, for each HMC step t. We consider two initial distributions.

The first one is P 0 = N (0, 3I) which satisfies the assumption H(P 0 ) > H(π).

The second one is P 0 = N (0, I) with the entropy H(P 0 ) < H(π), which violates the assumption.

In this latter case, we perform the unconstrained optimisation of equation 4.

Plots (b) and (c) in Figure 1 show samples from P 9 and P * 9 for the valid P 0 .

In this first example, maximising the ergodic objective under equation 5 significantly accelerates the chain convergence as further shown by the left plot in Figure  2 .

Plots (d) and (e) in Figure 1 show samples from P 9 and P * 9 for the invalid initial distribution P 0 .

In the second example, E p0 [log π * (x)] is higher than E π [log π * (x)] and, consequently, maximising the unconstrained ergodic objective actually deteriorates the quality of the resulting approximation.

This is further illustrated by the right plot in Figure 2 which shows how the convergence of E pt [log π * (x)] to E π [log π * (x)] is significantly slowed down by the optimisation under the invalid P 0 .

Fortunately, it is straightforward to prevent this type of failure cases by appropriately tuning the scalar hyperparameter h in equation 5.

A value of h that is too low may result in higher bias of P T after optimisation as illustrated by the convergence of E pt [log π * (x)] in the blue and orange curves in Plot (b) in Figure 2 .

Furthermore, in many cases, estimating an upper bound on H(π) is feasible.

For example, in Bayesian inference, the entropy of the prior distribution p(x) is often higher than the entropy of the posterior p(x|y).

Therefore, the prior entropy can be used as a reference for tuning h. Figure 1: Histograms of samples from ergodic inference using HMC transition kernels.

P 9 denotes the ergodic approximation before traing; P * 9 denotes the ergodic approximation after training.

] as a function of the length of the chain T using 10000 samples: Left: with the valid P 0 as H(P 0 ) > H(π); Right: with invalid P 0 as H(P 0 ) < H(π).

SG training means the stop gradient is applied to the x from previous HMC step in equation 9.

We first describe the general configuration of the ergodic inference method used in our experiments.

Our ergodic approximation is constructed using HMC, one of the most successful MCMC methods in machine learning literature.

We use T HMC transitions, each one involving 5 steps of the vanilla leapfrog integrator which was implemented following Neal (2010).

The leapfrog pseudocode can be found in the appendix.

In each HMC transition, the auxiliary variables are sampled from a zero-mean Gaussian distribution with diagonal covariance matrix.

We tune the following HMC parameters: the variance of the auxiliary variables and the leapfrog step size, as mentioned in Section 2.2.

We use and optimise a different value of the HMC parameters for each of the T HMC transitions considered.

We call our ergodic inference method Hamiltonian ergodic inference (HEI).

The burn-in model P 0 is factorized Gaussian.

The initial entropy of P 0 is chosen to be the same as the entropy of the prior.

The stocastic optimisation algorithm is Adam (Kingma & Ba, 2015) with TensorFlow implemtation Abadi et al. (2015) and the optimiser hyperparameter setting is (β 1 = 0.9, β 2 = 0.999, = 10 −8 ).

The initial HMC leapfrog step sizes are sampled uniformly between 0.01 and 0.025.

Additional experiment on Bayesian neural networks is included in Appendix 6.3.

We first compare Hamiltonian ergodic inference (HEI) with previous related methods on 6 synthetic bivariate benchmark distributions.

Histograms of ground truth samples from each target distribution using rejection sampling are shown in Figure 3 .

The baselines considered include: 1) Hamiltonian variational inference (HVI) (Salimans et al., 2015) ; 2) generalized Hamiltonian Monte Carlo (GHMC) using an NVP parameterized HMC kernel and gradient-based auto-tuning of MCMC parameters w.r.t.

sample correlation loss (Levy et al., 2018) ; 3) Hamiltonian annealed importance sampling (HAIS) (Sohl-Dickstein & Culpepper, 2012) .

It is worth to mention that we do not consider other hybrid inference methods like (Ruiz & Titsias, 2019; Hoffman, 2017) in our experiment, because these methods only combines MCMC simulation with VI but not optimise the parameters of MCMC kernel using the gradient-based approach like EI.

HVI is the most similar method to HEI among all three baselines, because both HEI and HVI methods generate samples from the last state of MCMC chains and use gradient-based MCMC hyperparameter tuning to reduce bias.

For a fair comparison between HVI and HEI, we consider the HMC chains with exactly the same setting in both methods: the initial state follows a standard Gaussian distribution and the length of HMC chain is T = 10.

The key difference between HVI and HEI is the hyperparameter tuning objective, as mentioned in Section 3.1.

We trained HVI for 1000 iterations and verified the ELBO converges to a (local) minimum (plots of the training ELBO values are included in Appendix 6.2).

We trained HEI for 50 iterations.

Following the setting of HAIS by , we used 1,000 intermediate distributions with 5 leapfrog steps per HMC transition and manually tuned the HMC parameters to have acceptance rate around 70%.

GHMC 1 was run using 100 parallel chains with 5 leapfrog steps per GHMC transition, 100 burn-in steps and 1000 auto-tuned training iterations Levy et al. (2018) .

The verification of the convergence of E p T [log π * (x)] to E π [log π * (x)] for HEI is shown in plot (a) of Figure 5 .

We generate 100,000 samples with each method and evaluate sample quality using two metrics: 1) the histogram of simulated samples for visual inspection; 2) the MC estimation of E π [log π * (x)].

Effective sample size (ESS) is a popular sample correlation based evaluation metric in recent MCMC literature (Levy et al., 2018) .

However, we do not consider ESS in this experiment, because GHMC is the only method among all methods generating correlated samples.

Therefore, the ESS of GHMC is guaranteed to be lower than HVI and HEI.

To generate ground truth samples from benchmark distributions, we use .

The resulting sample histograms of the ground truth using rejection sampling are shown in figures 3 and considered approximated sampling methods are shown in 4.

Table  2 shows the resulting estimates of −E π [log π * (x)] together with the wall-clock simulation time for generating 100,000 samples.

The left part of Table 3 shows the training time of the MCMC parameter optimisation for all methods except HAIS, which does not support gradient-based HMC hyperparameter tuning.

HEI is faster than HVI and GHMC.

Note, however, that the acceleration of HEI over HSVI is due to the stopping gradient trick described in Section 3.2.

The histograms and the estimates of −E π [log π * (x)] generated by HEI are consistent with the results of the more expensive unbiased samplers GHMC and HAIS, which are close to the ground truth.

By contrast, HVI exhibits a clear bias in all benchmarks.

Regarding the sampling time, HVI and HEI simulate HMC chains with the same length and, consequently, perform similarly in this case while sample simulation from HAIS and GHMC is much more expensive.

6.00 3000 -83.57 50 HVI(T =1, 16LF, n h =800, ConvNet encoder) 6.00 360 -83.68 48 HVAE(T =1, 16LF, n h =500, ConvNet encoder) 6.00 360 -84.22 48 HEI(T =30, 5LF, n h =500, no neural net encoder)

1.65 54 -83.17 48 HEI(T =30, 5LF, n h =500, no neural net encoder) 3.00 100 -82.76 46 HEI(T =30, 5LF, n h =500, no neural net encoder) 6.00 200 -82.65 45 HEI(T =30, 5LF, n h =500, no neural net encoder) 12.00 400 -81.43 38 HEI(T =15, 5LF, n h =500, no neural net encoder) 8.00 540 -83.30 48 Table 4 : Comparisons in terms of compuational efficiency and test log-likelihood in the training of deep generative models on the MNIST dataset.

We implemented the deconvolutional decoder network in Salimans et al. (2015) to test HVI.

In Salimans et al. (2015) , the test likelihood is estimated using importence-weighted samples from the encoder network.

In our experiment, we use Hamiltonian annealled importance sampling and report the effective sample size (ESS).

We now evaluate HEI in the task of training deep generative models.

MNIST is a standard benchmark problem in this case with 60,000 grey level 28 × 28 images of handwritten digits.

For fair comparison with previous works, we use the 10,000 prebinarised MNIST test images 2 used by Burda et al. (2015) .

The architecture of the generative model considered follows the deconvolutional network from Salimans et al. (2015) .

In particular, the unnormalised target p θ (x, y) consists of 32 dimensional latent variables x with Gaussian prior p(x) = N (0, I) and a deconvolutional network p θ (y|x) from top to bottom including a single fully-connected layer with 500 RELU hidden units, then three deconvolutional layers with 5 × 5 filters, (16, 32, 32) feature maps, RELU activations and a logistic output layer.

We consider a baseline given by a standard VAE with a factorised Gaussian approximate Salimans et al. (2015) 4 Table 3 : Left.

The training time of MCMC parameter optimisation in seconds for 100 iterations for all candidate methods to produce the results in Figure 4 .

The training time of HEI is lower than HVI because of the stop gradient trick mentioned in Section 3.2.

We do not report the training time for HAIS, because HAIS requires manual tuning of MCMC hyperparameters which is not directly comparable to the gradient-based autotuning used by the other methods.

Right.

The training time in seconds per epoch for the experiments with deep generative models (DGM).

posterior generated by an encoder network q(x|y) which mirrors the architecture of the decoder (Salimans et al., 2015) .

The code for HVI Salimans et al. (2015) is not publicly available.

Nevertheless, we reimplemented their convolutional VAE and were able to reproduce the marginal likelihood reported by Salimans et al. (2015) , as shown in Table 4 .

This verifies that our implementation of the generation network is correct.

We implemented HVI in (Salimans et al., 2015) using an auxiliary reverse model in the ELBO parameterized by a single hidden layer network with 640 hidden units and RELU activations.

We also implemented the Hamiltonian variational encoder (HVAE) method (Caterini et al., 2018) , which is similar to HVI but without the reverse model.

Unlike in the original HVAE, our implementation does not use tempering but still produces results similar to those from Caterini et al. (2018) .

For the HEI encoder, we use T = 30 HMC steps, each with 5 leapfrog steps.

The initial approximation P 0 is kept fixed to be the prior p(x).

We optimise the decoder and the HEI encoder jointly using Adam.

Table 4 shows the marginal test log-likelihood for HEI and the other methods, as estimated with 1,000 HAIS samples (Sohl-Dickstein & Culpepper, 2012) .

Following Li et al. (2017) , we also include the effective sample size (ESS) of HAIS samples for the purpose of verifying the reliability of the reported test log-likelihoods.

Overall, HEI outperforms HVI, HVAE and the standard VAE in test log-likelihood when the training time of all methods is fixed to be 6 hours.

HEI still produces significant gains when the training time is extended to 12 hours and, with only 1.6 hours of training, HEI can already outperform the convolutional VAE of Salimans et al. (2015) with 6 hours of training.

To verify the convergence of HEI, we show in plot (b) of Figure 5 estimates of

. . , 10 on five randomly chosen test images, where the ground truth E π [log π * (x)] is estimated by HAIS, after HMC hyper-parameter tuning in HEI (blue) and without hyper-parameter tuning in HEI (green), i.e. just using the initial hyper-parameter values.

Plot (c) in Figure 5 shows similar results, but using the maximum mean discrepancy (MMD) score (Gretton et al., 2012) to quantify the similarity of samples from p T to samples from π, where the latter ground truth samples are generated by HAIS.

These plots suggests that shortening the HEI chain to T = 10 HMC steps will have a negligible effect on final simulation accuracy.

Finally, the right part of Table 3 shows the training time of HEI with and without the stopping gradient trick.

These resuls show that the former method is up to 5 times faster.

:

a: the targets are 2D benchmarks with the ground truth of E π [log π * (x)]; b: the target π is the VAE posterior p(x|y) each curve represents one random chosen test MNIST image y with the ground truth of E π [log π * (x)] estimated by HAIS using 100 samples; c: MMD score between HEI samples and HAIS samples.

We have proposed Ergodic Inference (EI), a novel hybrid inference method that bridges MCMC and VI.

EI a) reduces the approximation bias by increasing the number of MCMC steps, b) generates independent samples and c) tunes MCMC hyperparameters by optimising an objective function that directly quantifies the bias of the resulting samples.

The effectiveness of EI was verified on synthetic examples and on popular benchmarks for deep generative models.

We have shown that we can generate samples much closer to a gold standard sampling method than similar hybrid inference methods and at a low computational cost.

However, one disadvantage of EI is that it requires the entropy of the first MCMC step to be larger than the entropy of the target distribution.

Here is the code for the vanilla leapfrog algorithm we used in HVI, HEI and HAIS.

Algorithm 1: Leapfrog Input: x: state, r: momenta, φ 1 : r variance, φ 2 : step size, m: number of steps Result: x : new state, r : new momentum x = x; r = r; for t ← 1 to m dō r =r − 0.5φ 2 ∂ x U (x); x =x + φ 2 /φ 1r ; r =r − 0.5φ 2 ∂ x U (x); end x =x; r =r; return x and r ;

The plots in Figure 6 show training loss (negative ELBO) of HVI and the training expected log likelihood E p T [log π * (x)] with T = 10 HMC steps with Adam with hyperparameter setting described in Section 4.

It is clear that HVI is well trained but the approximation is biased, because E p T [log π * (x)] does not converge to the true loss (the red line on the right plots).

In comparison, in Figure 6 (Left) in our paper, E p T [log π * (x)] of HEI converges to the ground true by optimising our ergodic loss.

In this additional experiment we approximate the posterior distribution of Bayesian neural networks with standard Gaussian priors.

We consider four UCI datasets and compare HEI with the stochastic gradient Hamilton Monte Carlo (SGHMC) method from Springenberg et al. (2016) .

The networks used in this experiment have 50 hidden layers and 1 real valued output unit, as stated in Springenberg et al. (2016) .

The HEI chain contains 50 HMC transformation with 3 Leapfrog steps each.

The initial proposal distribution P 0 is a factorised Gaussian distribution with mean values obtained by running standard mean-field VI using Adam for 200 iterations.

We do not use in P 0 the variance values returned by VI because these are unlikely to result in higher entropy than the exact posterior since VI tends to understimate uncertainty.

Instead, we choose the marginal variances to be n −0.5 where n is the number of inputs to the neural network layer for the weight.

To reduce computational cost, we use in this case stochastic gradients in the leapfrog integrator.

For this, we split the training data into 19 mini-batches and only use one random sampled mini-batch for computing the gradient in each leapfrog iteration.

We train our HEI for 10 epochs and the stationary distribution is chosen as approximate posterior on a random sampled mini-batch.

The resulting test log-likelihoods are shown in Table 5 .

Overall, HEI produce significantly better results than SGHMC.

We also show in the right plot of Figure 7 estimates of E pt [log p(x, y)] for t = 1, . . .

, 50 after HMC hyper-parameter tuning and without hyper-parameter tuning.

@highlight

In this work, we aim to improve upon MCMC and VI by a novel hybrid method based on the idea of reducing simulation bias of finite-length MCMC chains using gradient-based optimisation.
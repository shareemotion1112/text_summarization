Inverse problems are ubiquitous in natural sciences and refer to the challenging task of inferring complex and potentially multi-modal posterior distributions over hidden parameters given a set of observations.

Typically, a model of the physical process in the form of differential equations is available but leads to intractable inference over its parameters.

While the forward propagation of parameters through the model simulates the evolution of the system, the inverse problem of finding the parameters given the sequence of states is not unique.

In this work, we propose a generalisation of the Bayesian optimisation framework to approximate inference.

The resulting method learns approximations to the posterior distribution by applying Stein variational gradient descent on top of estimates from a Gaussian process model.

Preliminary results demonstrate the method's performance on likelihood-free inference for reinforcement learning environments.

We consider the problem of estimating parameters θ of a physical system according to observed data y.

The forward model of the system is approximated by a computational model that generates dataŷ θ based on the given parameter settings θ.

In many cases, the corresponding likelihood function p(ŷ θ |θ) is not available, and one resorts to likelihoodfree methods, such as approximate Bayesian computation (ABC) (Robert, 2016) , conditional density estimation (Papamakarios and Murray, 2016) , etc.

For certain applications in robotics and reinforcement learning, however, the number of simulations might be limited by resource constraints, imposing challenges to current approaches.

Recent methods address the problem of efficiency in the use of simulations by either constructing conditional density estimators from joint data {θ i ,ŷ i } N i=1 , using, for example, mixture density networks (Papamakarios and Murray, 2016; Ramos et al., 2019) , or by sequentially learning approximations to the likelihood function (Gutmann and Corander, 2016; Papamakarios et al., 2019) and then running Markov chain Monte Carlo (MCMC).

In particular, Gutmann and Corander (2016) derive an active learning approach using Bayesian optimisation (BO) (Shahriari et al., 2016) to propose parameters for simulations.

Their approach reduces the number of simulator runs from the typical thousands to a few hundreds.

This paper investigates an approach to combine the flexible representative power of variational inference methods (Liu and Wang, 2016) with the data efficiency of Bayesian optimisation.

We present a Thompson sampling strategy (Russo and Van Roy, 2016) to sequentially refine variational approximations to a black-box posterior.

Parameters for new simulations are proposed by running Stein variational gradient descent (SVGD) (Liu and Wang, 2016) over samples from a Gaussian process (GP) (Rasmussen and Williams, 2006) .

The approach is also equipped with a method to optimally subsample the variational approximations for batch evaluations of the simulator models at each round.

In the following, we present the derivation of our approach and preliminary experimental results.

Our goal is to estimate a distribution q that approximates a posterior distribution p(θ|y) over simulator parameters θ ∈ Θ ⊂ R d given observations y from a target system.

We assume no access to a likelihood function p(y|θ), but only to a discrepancy measure 1 between simulator outputs and observations ∆ θ , as in Gutmann and Corander (2016) .

We take a Bayesian optimisation approach to find the optimal q * by minimising a discrepancy between q and the target p:

where S represents the kernelised Stein discrepancy (KSD) (Liu et al., 2016) .

2 We solve Equation 1 via a black-box approach which does not require gradients of the target distribution p nor its availability in closed form.

The resulting BO algorithm is composed of a GP model to form an approximate likelihood, a Thompson sampling acquisition function to select candidate distributions and a kernel herding procedure to optimally select samples of simulator parameters.

A standard BO approach would place a GP to model the map from q's parameters to the corresponding KSD.

However, such parameter space holds a weak connection with the original Θ and is possibly higher-dimensional.

We choose to bypass this step by learning q directly via Stein variational gradient descent (SVGD) (Liu and Wang, 2016) .

Applying SVGD directly to Equation 1 would require gradients of the target log p.

In our case, we have that:

As p(y|θ) is unavailable, we use a GP to model g : θ → −∆ θ , which defines a synthetic likelihood function (Gutmann and Corander, 2016) , i.e.:

1.

The ABC literature offers a plenitude of choices for ∆ θ .

For a review, we refer the reader to Gutmann and Corander (2016) and Robert (2016) .

Our choice for experiments is given in Section 3.

2.

Background details on the KSD are presented in the appendix (Section A.1).

The simulations-observations discrepancy ∆ θ is possibly expensive to evaluate and not differentiable, due to the need of running a black-box simulator.

The GP then provides an approximation which is cheap to evaluate and whose sample functions are differentiable for smooth kernels, allowing us to apply SVGD in the BO loop.

We propose selecting candidate distributions q n ∈ Q based on a GP posterior sampling approach known as Thompson sampling (Russo and Van Roy, 2016) , which has been successfully applied to BO problems in the case of selecting point candidates θ ∈ Θ (Chowdhury and Gopalan, 2017; Kandasamy et al., 2018; Mutný and Krause, 2018) .

Thompson sampling accounts for uncertainty in the model by sampling functions from the GP posterior.

For models based on finite feature maps, such as sparse spectrum Gaussian processes (SSGPs) (Lázaro-Gredilla et al., 2010) , the Thompson sampling approach resumes to sampling weights w n from a multivariate Gaussian (Appendix A.2), so that:

constitutes a sample from the posterior of a SSGP with mean function µ 0 and feature map φ.

Recalling the objective in Equation 1, we can now define the acquisition function as:

wherep n (θ) ∝ p(θ)e gn(θ) corresponds to an approximation to the target posterior p(θ|y) based on g n .

SVGD represents the variational distribution q as a set of particles {θ i } M i=1 forming an empirical distribution.

The particles are initialised as i.i.d.

samples from the prior p(θ) and optimised via a sequence of smooth perturbations:

where k(θ, θ ) = φ(θ) T φ(θ ) corresponds to the SSGP kernel, and η t is a small step size.

Intuitively, the first term in the definition of ζ guides the particles to the local maxima of logp n , i.e. the modes ofp n , while the second term encourages diversification by repelling nearby particles.

In contrast to the true posterior, the gradients of logp n are available as:

Gradients of sample functions are always defined for SSGP models with differentiable mean functions, since the feature maps are smooth.

For a uniform prior, which we use in experiments, also note that ∇ θ log p(θ) = 0 almost everywhere.

Having selected a distribution q n , we need to run evaluations of ∆ θ from samples θ ∼ q n to update the GP model with.

Representing q by a large number of particles M improves Algorithm 1: DBO Input: f , Q, N , S for n ∈ {1, . . .

N } do q n ∈ argmax q∈Q h(q|D n−1 ) # Maximise acquisition function via SVGD {θ n,i } S i=1 ∼ Herding(q n , D n−1 ) # Sample simulator parameters for i ∈ {1, . . .

, S} do z n,i := −∆ θ n,i # Collect observation end

end exploration of the approximate posterior surface, allowing SVGD to find distant modes.

However, we should not use the large number of particles directly as sample parameters to run the simulator with, since simulations are expensive.

Therefore, we select S M query parameters {θ n,j } S j=1 ⊂ Θ by optimally subsampling the candidate q n .

Kernel herding (Chen et al., 2010 ) constructs a set of samples which minimises the error on empirical estimates for expectations under a given distribution q.

This error is bounded by the maximum mean discrepancy (MMD) between the kernel embedding of q and its subsampled version (Muandet et al., 2016) .

In the case of SSGPs, the kernel herding procedure resumes to the following algorithm:

for j ∈ {0, . . .

, S − 1} and α 0 = ψ q = E θ∼q [φ(θ)].

However, instead of naively herding with the original feature map φ, we make use of the information encoded by the GP to select samples which will be the most informative for the model.

Such information is encoded by the GP posterior kernel:

where

N is the covariance matrix of the GP weights posterior (defined in Appendix A.2).

The posterior kernel provides an embedding for q given by:

which accounts for the previously observed locations in the GP data.

Replacing ψ q by ψ n q in Equation 8 yields the sampling scheme we use.

The distributional Bayesian optimisation (DBO) algorithm is summarised in Algorithm 1.

In this section, we present experimental results evaluating DBO in synthetic data scenarios.

As a baseline we compare the method against mixture density networks (MDNs), as in Ramos et al. (2019) , which were learnt from a dataset of parameters sampled from the prior p(θ) and the corresponding simulator outputsŷ θ .

The experiment evaluates the proposed method on OpenAI Gym's 3 cart-pole environment.

We fix a given setting for its physics parameters θ real and generate a dataset y of 10 trajectories by executing randomly sampled actions.

Summary statistics γ were the same as Ramos et al. (2019) .

The discrepancy was set to ∆ θ := γ θ − γ real 2 /σ 2 .

We place a uniform prior p(θ) with bounds specific for the environment.

Further details on the experimental setup are described in Appendix B. An open-source implementation can be found online 4 .

The results in Figure 1 show that the mehtod is able to recover the target system's curve-shaped posterior and is able to obtain better approximations to the posterior when compared to the MDN approach.

We can also see that in terms of MMD, DBO is able to provide a better overall approximation than the MDN.

This paper presented a Bayesian optimisation approach to inverse problems on simulator parameters.

Preliminary results demonstrated the potential of the method for reinforcement learning applications.

In particular, results show that distributional Bayesian optimisation is able to provide a more sample-efficient approach than other likelihood-free inference methods when inferring parameters of a classical reinforcement learning environment.

Future work includes further scalability and theoretical analysis of the method.

3.

OpenAI Gym: https://gym.openai.com 4.

Code available at: https://github.com/rafaol/dbo-aabi2019 , instead.

, we can represent any function g sampled from the SSGP posterior as g(θ) = µ 0 (θ) + w T φ(θ), θ ∈ Θ, where:

with Φ N = [φ(θ 1 ), . . .

, φ(θ N )] ∈ R 2M ×N .

The posterior over g is then determined by:

µ N (θ) := µ 0 (θ) + φ(θ)

where µ N and σ 2 N denote the GP posterior mean and variance functions, respectively.

Fast incremental updates: To reduce the time complexity in the update of the GP posterior when given a new observation pair (θ N +1 , z N +1 ), Gijsberts and Metta (2013) propose using the decomposition:

To avoid recomputing A −1 N +1 , one can instead keep track of its Cholesky factors.

The latter allows us to update the GP posterior with time complexity O(M 2 ) (Gijsberts and Metta, 2013) , which is constant with respect to the number of data points N .

@highlight

An approach to combine variational inference and Bayesian optimisation to solve complicated inverse problems
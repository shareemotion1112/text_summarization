We present a general-purpose method to train Markov chain Monte Carlo kernels, parameterized by deep neural networks, that converge and mix quickly to their target distribution.

Our method generalizes Hamiltonian Monte Carlo and is trained to maximize expected squared jumped distance, a proxy for mixing speed.

We demonstrate large empirical gains on a collection of simple but challenging distributions, for instance achieving a 106x improvement in effective sample size in one case, and mixing when standard HMC makes no measurable progress in a second.

Finally, we show quantitative and qualitative gains on a real-world task: latent-variable generative modeling.

Python source code will be open-sourced with the camera-ready paper.

High-dimensional distributions that are only analytically tractable up to a normalizing constant are ubiquitous in many fields.

For instance, they arise in protein folding BID41 , physics simulations BID37 , and machine learning BID1 .

Sampling from such distributions is a critical task for learning and inference BID31 , however it is an extremely hard problem in general.

Markov Chain Monte Carlo (MCMC) methods promise a solution to this problem.

They operate by generating a sequence of correlated samples that converge in distribution to the target.

This convergence is most often guaranteed through detailed balance, a sufficient condition for the chain to have the target equilibrium distribution.

In practice, for any proposal distribution, one can ensure detailed balance through a Metropolis-Hastings BID20 accept/reject step.

Despite theoretical guarantees of eventual convergence, in practice convergence and mixing speed depend strongly on choosing a proposal that works well for the task at hand.

What's more, it is often more art than science to know when an MCMC chain has converged ("burned-in"), and when the chain has produced a new uncorrelated sample ("mixed").

Additionally, the reliance on detailed balance, which assigns equal probability to the forward and reverse transitions, often encourages random-walk behavior and thus slows exploration of the space BID24 .For densities over continuous spaces, Hamiltonian Monte Carlo (HMC; BID12 BID36 introduces independent, auxiliary momentum variables, and computes a new state by integrating Hamiltonian dynamics.

This method can traverse long distances in state space with a single Metropolis-Hastings test.

This is the state-of-the-art method for sampling in many domains.

However, HMC can perform poorly in a number of settings.

While HMC mixes quickly spatially, it struggles at mixing across energy levels due to its volume-preserving dynamics.

HMC also does not work well with multi-modal distributions, as the probability of sampling a large enough momentum to traverse a very low-density region is negligibly small.

Furthermore, HMC struggles with ill-conditioned energy landscapes BID14 and deals poorly with rapidly changing gradients BID44 .Recently, probabilistic models parameterized by deep neural networks have achieved great success at approximately sampling from highly complex, multi-modal empirical distributions BID27 BID39 BID16 .

Building on these successes, we present a method that, given an analytically described distribution, automatically returns an exact sampler with good convergence and mixing properties, from a class of highly expressive parametric models.

The proposed family of samplers is a generalization of HMC; it transforms the HMC trajectory using parametric functions (deep networks in our experiments), while retaining theoretical guarantees with a tractable Metropolis-Hastings accept/reject step.

The sampler is trained to minimize a variation on expected squared jumped distance (similar in spirit to BID38 ).

Our parameterization reduces easily to standard HMC.

It is further capable of emulating several common extensions of HMC such as withintrajectory tempering BID34 and diagonal mass matrices BID4 .We evaluate our method on distributions where HMC usually struggles, as well as on a the real-world task of training latent-variable generative models.

Our contributions are as follows:• We introduce a generic training procedure which takes as input a distribution defined by an energy function, and returns a fast-mixing MCMC kernel.•

We show significant empirical gains on various distributions where HMC performs poorly.• We finally evaluate our method on the real-world task of training and sampling from a latent variable generative model, where we show improvement in the model's log-likelihood, and greater complexity in the distribution of posterior samples.

Adaptively modifying proposal distributions to improve convergence and mixing has been explored in the past BID0 .

In the case of HMC, prior work has reduced the need to choose step size BID36 or number of leapfrog steps BID22 by adaptively tuning those parameters.

BID40 proposed an alternate scheme based on variational inference.

We adopt the much simpler approach of BID38 , who show that choosing the hyperparameters of a proposal distribution to maximize expected squared jumped distance is both principled and effective in practice.

Previous work has also explored applying models from machine learning to MCMC tasks.

Kernel methods have been used both for learning a proposal distribution BID42 and for approximating the gradient of the energy BID47 .

In physics, Restricted and semiRestricted Boltzmann machines have been used both to build approximations of the energy function which allow more rapid sampling BID30 BID23 , and to motivate new hand-designed proposals BID50 .Most similar to our approach is recent work from BID46 , which uses adversarial training of a volume-preserving transformation, which is subsequently used as an MCMC proposal distribution.

While promising, this technique has several limitations.

It does not use gradient information, which is often crucial to maintaining high acceptance rates, especially in high dimensions.

It also can only indirectly measure the quality of the generated sample using adversarial training, which is notoriously unstable, suffers from "mode collapse" (where only a portion of a target distribution is covered), and often requires objective modification to train in practice BID2 .

Finally, since the proposal transformation preserves volume, it can suffer from the same difficulties in mixing across energy levels as HMC, as we illustrate in Section 5.To compute the Metropolis-Hastings acceptance probability for a deterministic transition, the operator must be invertible and have a tractable Jacobian.

Recent work BID11 , introduces RNVP, an invertible transformation that operates by, at each layer, modifying only a subset of the variables by a function that depends solely on the remaining variables.

This is exactly invertible with an efficiently computable Jacobian.

Furthermore, by chaining enough of these layers, the model can be made arbitrarily expressive.

This parameterization will directly motivate our extension of the leapfrog integrator in HMC.

Let p be a target distribution, analytically known up to a constant, over a space X .

Markov chain Monte Carlo (MCMC) methods BID33 aim to provide samples from p. To that end, MCMC methods construct a Markov Chain whose stationary distribution is the target distribution p. Obtaining samples then corresponds to simulating a Markov Chain, i.e., given an initial distribution π 0 and a transition kernel K, constructing the following sequence of random variables: DISPLAYFORM0 In order for p to be the stationary distribution of the chain, three conditions must be satisfied: K must be irreducible and aperiodic (these are usually mild technical conditions) and p has to be a fixed point of K. This last condition can be expressed as:

DISPLAYFORM1 This condition is most often satisfied by satisfying the stronger detailed balance condition, which can be written as: DISPLAYFORM2 Given any proposal distribution q, satisfying mild conditions, we can easily construct a transition kernel that respects detailed balance using Metropolis-Hastings BID20 accept/reject rules.

More formally, starting from x 0 ∼ π 0 , at each step t, we sample x ∼ q(·|X t ), and with probability A(x |x t ) = min 1, DISPLAYFORM3 , accept x as the next sample x t+1 in the chain.

If we reject x , then we retain the previous state and x t+1 = x t .

For typical proposals this algorithm has strong asymptotic guarantees.

But in practice one must often choose between very low acceptance probabilities and very cautious proposals, both of which lead to slow mixing.

For continuous state spaces, Hamiltonian Monte Carlo (HMC; Neal, 2011) tackles this problem by proposing updates that move far in state space while staying roughly on iso-probability contours of p.

Without loss of generality, we assume p (x) to be defined by an energy function U (x), s.t.

p(x) ∝ exp(−U (x)), and where the state x ∈ R n .

HMC extends the state space with an additional momentum vector v ∈ R n , where v is distributed independently from x, as p(v) DISPLAYFORM0 (i.e., identity-covariance Gaussian).

From an augmented state ξ (x, v), HMC produces a proposed state ξ = (x , v ) by approximately integrating Hamiltonian dynamics jointly on x and v, with U (x) taken to be the potential energy, and 1 2 v T v the kinetic energy.

Since Hamiltonian dynamics conserve the total energy of a system, their approximate integration moves along approximate iso-probability contours of p(x, v) = p(x)p(v).The dynamics are typically simulated using the leapfrog integrator BID19 BID29 , which for a single time step consists of: DISPLAYFORM1 Following Sohl-Dickstein et al. FORMULA0 , we write the action of the leapfrog integrator in terms of an operator L: Lξ L(x, v) (x , v ), and introduce a momentum flip operator F: F(x, v) (x, −v).

It is important to note two properties of these operators.

First, the transformation FL is an involution, i.e. FLFL( DISPLAYFORM2 2 ), and from (x , v 1 2 ) to (x , v ) are all volume-preserving shear transformations i.e., only one of the variables (x or v) changes, by an amount determined by the other one.

The determinant of the Jacobian, DISPLAYFORM3 , is thus easy to compute.

For vanilla HMC DISPLAYFORM4 = 1, but we will leave it in symbolic form for use in Section 4.

The Metropolis-HastingsGreen BID20 BID17 acceptance probability for the HMC proposal is made simple by these two properties, and is DISPLAYFORM5

In this section, we describe our proposed method L2HMC (for 'Learning To Hamiltonian Monte Carlo').

Given access to only an energy function U (and not samples), L2HMC learns a parametric leapfrog operator L θ over an augmented state space.

We begin by describing what desiderata we have for L θ , then go into detail on how we parameterize our sampler.

Finally, we conclude this section by describing our training procedure.

HMC is a powerful algorithm, but it can still struggle even on very simple problems.

For example, a two-dimensional multivariate Gaussian with an ill-conditioned covariance matrix can take arbitrarily long to traverse (even if the covariance is diagonal), whereas it is trivial to sample directly from it.

Another problem is that HMC can only move between energy levels via a random walk BID36 , which leads to slow mixing in some models.

Finally, HMC cannot easily traverse low-density zones.

For example, given a simple Gaussian mixture model, HMC cannot mix between modes without recourse to additional tricks, as illustrated in FIG1 .

These observations determine the list of desiderata for our learned MCMC kernel: fast mixing, fast burn-in, mixing across energy levels, and mixing between modes.

While pursuing these goals, we must take care to ensure that our proposal operator retains two key features of the leapfrog operator used in HMC: it must be invertible, and the determinant of its Jacobian must be tractable.

The leapfrog operator satisfies these properties by ensuring that each sub-update only affects a subset of the variables, and that no sub-update depends nonlinearly on any of the variables being updated.

We are free to generalize the leapfrog operator in any way that preserves these properties.

In particular, we are free to translate and rescale each sub-update of the leapfrog operator, so long as we are careful to ensure that these translation and scale terms do not depend on the variables being updated.

As in HMC, we begin by augmenting the current state x ∈ R n with a continuous momentum variable v ∈ R n drawn from a standard normal.

We also introduce a binary direction variable d ∈ {−1, 1}, drawn from a uniform distribution.

We will denote the complete augmented state as ξ (x, v, d), with probability density p(ξ) = p(x)p(v)p(d).

Finally, to each step t of the operator L θ we assign a fixed random binary mask m t ∈ {0, 1} n that will determine which variables are affected by each sub-update.

We draw m t uniformly from the set of binary vectors satisfying DISPLAYFORM0 , that is, half of the entries of m t are 0 and half are 1.

For convenience, we writem t = 1 − m t and x m t = x m t ( denotes element-wise multiplication, and 1 the all ones vector).

We now describe the details of our augmented leapfrog integrator L θ , for a single time-step t, and for direction d = 1.We first update the momenta v. This update can only depend on a subset ζ 1 (x, ∂ x U (x), t) of the full state, which excludes v. It takes the form DISPLAYFORM0 We have introduced three new functions of ζ 1 : T v , Q v , and S v .

T v is a translation, exp(Q v ) rescales the gradient, and exp( 2 S v ) rescales the momentum.

The determinant of the Jacobian of this transformation is exp 2 1 · S v (ζ 1 ) .

Note that if T v , Q v , and S v are all zero, then we recover the standard leapfrog momentum update.

We now update x. As hinted above, to make our transformation more expressive, we first update a subset of the coordinates of x, followed by the complementary subset.

The first update, which yields x and affects only x m t , depends on the state subset ζ 2 (xmt, v, t).

Conversely, with x defined below, the second update only affects x m t and depends only on ζ 3 (x m t , v, t): DISPLAYFORM1 Again, T x is a translation, exp(Q x ) rescales the effect of the momenta, exp( S x ) rescales the positions x, and we recover the original leapfrog position update if T x = Q x = S x = 0.

The determinant of the Jacobian of the first transformation is exp ( m t · S x (ζ 2 )), and the determinant of the Jacobian of the second transformation is exp ( m t · S x (ζ 3 )).Finally, we update v again, based on the subset ζ 4 (x , ∂ x U (x ), t): DISPLAYFORM2 This update has the same form as the momentum update in equation 4.To give intuition into these terms, the scaling applied to the momentum can enable, among other things, acceleration in low-density zones, to facilitate mixing between modes.

The scaling term applied to the gradient of the energy may allow better conditioning of the energy landscape (e.g., by learning a diagonal inertia tensor), or partial ignoring of the energy gradient for rapidly oscillating energies.

The corresponding integrator for d = −1 is given in Appendix A; it essentially just inverts the updates in equations 4, 5 and 6.

For all experiments, the functions Q, S, T are implemented using multi-layer perceptrons, with shared weights.

We encode the current time step in the MLP input.

Our leapfrog operator L θ corresponds to running M steps of this modified leapfrog, DISPLAYFORM3 , and our flip operator F reverses the direction variable d, Fξ = (x, v, −d).

Written in terms of these modified operators, our proposal and acceptance probability are identical to those for standard HMC.

Note, however, that this parameterization enables learning non-volume-preserving transformations, as the determinant of the Jacobian is a function of S x and S v that does not necessarily evaluate to 1.

This quantity is derived in Appendix B.

For convenience, we denote by R an operator that re-samples the momentum and direction.

I.e., given DISPLAYFORM0 .

Sampling thus consists of alternating application of the FL θ and R, in the following two steps each of which is a Markov transition that satisfies detailed balance with respect to p: DISPLAYFORM1 This parameterization is effectively a generalization of standard HMC as it is non-volume preserving, with learnable parameters, and easily reduces to standard HMC for Q, S, T = 0.

We need some criterion to train the parameters θ that control the functions Q, S, and T .

We choose a loss designed to reduce mixing time.

Specifically, we aim to minimize lag-one autocorrelation.

This is equivalent to maximizing expected squared jumped distance BID38 .

For ξ, ξ in the extended state space, we define DISPLAYFORM0 However, this loss need not encourage mixing across the entire state space.

Indeed, maximizing this objective can lead to regions of state space where almost no mixing occurs, so long as the average squared distance traversed remains high.

To optimize both for typical and worst case behavior, we include a reciprocal term in the loss, DISPLAYFORM1 where λ is a scale parameter, capturing the characteristic length scale of the problem.

The second term encourages typical moves to be large, while the first term strongly penalizes the sampler if it is ever in a state where it cannot move effectively -δ(ξ, ξ ) being small resulting in a large loss value.

We train our sampler by minimizing this loss over both the target distribution and initialization distribution.

Formally, given an initial distribution π 0 over X , we define q(ξ) = π 0 (x)N (v; 0, I)p(d), and minimize DISPLAYFORM2 The first term of this loss encourages mixing as it considers our operator applied on draws from the distribution; the second term rewards fast burn-in; λ b controls the strength of the 'burn-in' regularization.

Given this loss, we exactly describe our training procedure in Algorithm 1.

It is important to note that each training iteration can be done with only one pass through the network and can be efficiently batched.

We further emphasize that this training procedure can be applied to any learnable operator whose Jacobian's determinant is tractable, making it a general framework for training MCMC proposals.

Input: Energy function U : X → R and its gradient ∇ x U : X → X , initial distribution over the augmented state space q, number of iterations n iters , number of leapfrogs M , learning rate schedule (α t ) t≤niters , batch size N , scale parameter λ and regularization strength λ b .

Initialize the parameters of the sampler θ.

Initialize {ξ DISPLAYFORM0

We present an empirical evaluation of our trained sampler on a diverse set of energy functions.

We first present results on a collection of toy distributions capturing common pathologies of energy landscapes, followed by results on a task from machine learning: maximum-likelihood training of deep generative models.

For each, we compare against HMC with well-tuned step length and show significant gains in mixing time.

Code implementing our algorithm is available online 1 .

We evaluate our L2HMC sampler on a diverse collection of energy functions, each posing different challenges for standard HMC.Ill-Conditioned Gaussian (ICG): Gaussian distribution with diagonal covariance spaced loglinearly between 10 −2 and 10 2 .

This demonstrates that L2HMC can learn a diagonal inertia tensor.

.

This is an extreme version of an example from Neal (2011).

This problem shows that, although our parametric sampler only applies element-wise transformations, it can adapt to structure which is not axis-aligned.

Mixture of Gaussians (MoG): Mixture of two isotropic Gaussians with σ 2 = 0.1, and centroids separated by distance 4.

The means are thus about 12 standard deviations apart, making it almost impossible for HMC to mix between modes.

Rough Well: Similar to an example from BID44 , for a given η > 0, U (x) = 1 2 x T x + η i cos( xi η ).

For small η the energy itself is altered negligibly, but its gradient is perturbed by a high frequency noise oscillating between −1 and 1.

In our experiments, we choose η = 10 −2 .For each of these distributions, we compare against HMC with the same number of leapfrog steps and a well-tuned step-size.

To compare mixing time, we plot auto-correlation for each method and report effective sample size (ESS).

We compute those quantities in the same way as BID44 .

We observe that samplers trained with L2HMC show greatly improved autocorrelation and ESS on the presented tasks, providing more than 106× improved ESS on the SCG task.

In addition, for the MoG, we show that L2HMC can easily mix between modes while standard HMC gets stuck in a mode, unable to traverse the low density zone.

Experimental details, as well as a comparison with LAHMC BID44 , are shown in Appendix C.Comparison to A-NICE-MC BID46 In addition to the well known challenges associated with adversarial training BID2 , we note that parameterization using a volume-preserving operator can dramatically fail on simple energy landscapes.

We build off of the mog2 experiment presented in BID46 , which is a 2-d mixture of isotropic Gaussians separated by a distance of 10 with variances 0.5.

We consider that setup but increase the ratio of variances: σ 2 1 = 3, σ 2 2 = 0.05.

We show in FIG1 sample chains trained with L2HMC and A-NICE-MC; A-NICE-MC cannot effectively mix between the two modes as only a fraction of the volume of the large mode can be mapped to the small one, making it highly improbable to traverse.

This is also an issue for HMC.

On the other hand, L2HMC can both traverse the low-density region between modes, and map a larger volume in the left mode to a smaller volume in the right mode.

It is important to note that the distance between both clusters is less than in the mog2 case, and it is thus a good diagnostic of the shortcomings of volume-preserving transformations.

We apply our learned sampler to the task of training, and sampling from the posterior of, a latentvariable generative model.

The model consists of a latent variable z ∼ p(z), where we choose p(z) = N (z; 0, I), and a conditional distribution p(x|z) which generates the image x. Given a family of parametric 'decoders' {z → p(x|z; φ), φ ∈ Φ}, and a set of samples D = {x (i) } i≤N , training involves finding φ * = arg max φ∈Φ p(D; φ).

However, the log-likelihood is intractable as p(x; φ) = p(x|z; φ)p(z)dz.

To remedy that problem, BID27 proposed jointly training an approximate posterior q ψ that maximizes a tractable lower-bound on the log-likelihood: DISPLAYFORM0 where q ψ (z|x) is a tractable conditional distribution with parameters ψ, typically parameterized by a neural network.

Recently, to improve upon well-known pitfalls like over-pruning BID7 of the VAE, Hoffman FORMULA0 proposed HMC-DLGM.

For a data sample x (i) , after obtaining a sample from the approximate posterior q ψ (·|x (i) ), Hoffman (2017) runs a MCMC algorithm with DISPLAYFORM1 to obtain a more exact posterior sample from p(z|x (i) ; φ).

Given that better posterior sample z , the algorithm maximizes log p(x (i) |z ; φ).To show the benefits of L2HMC, we borrow the method from Hoffman (2017), but replace HMC by jointly training an L2HMC sampler to improve the efficiency of the posterior sampling.

We call this model L2HMC-DLGM.

A diagram of our model and a formal description of our training procedure are presented in Appendix D. We define, for ξ = {z, v, d}, r(ξ|x; ψ) q ψ (z|x)N (v; 0, I) U (d; {−1, 1}).In the subsequent sections, we compare our method to the standard VAE model from BID27 and HMC-DGLM from Hoffman (2017).

It is important to note that, since our sampler is trained jointly with p φ and q ψ , it performs exactly the same number of gradient computations of the energy function as HMC.

We first show that training a latent variable generative model with L2HMC results in better generative models both qualitatively and quantitatively.

We then show that our improved sampler enables a more expressive, non-Gaussian, posterior.

Implementation details: Our decoder (p φ ) is a neural network with 2 fully connected layers, with 1024 units each and softplus non-linearities, and outputs Bernoulli activation probabilities for each pixel.

The encoder (q ψ ) has the same architecture, returning mean and variance for the approximate posterior.

Our model was trained for 300 epochs with Adam BID26 ) and a learning rate α = 10 −3 .

All experiments were done on the dynamically binarized MNIST dataset (LeCun).

We first present samples from decoders trained with L2HMC, HMC and the ELBO (i.e. vanilla VAE).

Although higher log likelihood does not necessarily correspond to better samples BID48 , we can see in Figure 5 , shown in the Appendix, that the decoder trained with L2HMC generates sharper samples than the compared methods.

We now compare our method to HMC in terms of log-likelihood of the data.

As we previously stated, the marginal likelihood of a data point x ∈ X is not tractable as it requires integrating p(x, z) over a high-dimensional space.

However, we can estimate it using annealed importance sampling (AIS; Neal FORMULA0 ).

Following BID51 , we evaluate our generative models on both training and held-out data.

In FIG2 , we plot the data's log-likelihood against the number of gradient computation steps for both HMC-DGLM and L2HMC-DGLM.

We can see that for a similar number of gradient computations, L2HMC-DGLM achieves higher likelihood for both training and held-out data.

This is a strong indication that L2HMC provides significantly better posterior samples.

In the standard VAE framework, approximate posteriors are often parametrized by a Gaussian, thus making a strong assumption of uni-modality.

In this section, we show that using L2HMC to sample from the posterior enables learning of a richer posterior landscape.

Block Gibbs Sampling To highlight our ability to capture more expressive posteriors, we in-paint the top of an image using Block Gibbs Sampling using the approximate posterior or L2HMC.

Formally, let x 0 be the starting image.

We denote top or bottom-half pixels as x top 0 and x bottom 0.

At each step t, we sample z (t) ∼ p(z|x t ; θ), samplex ∼ p(x|z t ; θ).

We then set x .

We compare the results obtained by sampling from p(z|x; θ) using q ψ (i.e. the approximate posterior) vs. our trained sampler.

The results are reported in FIG3 .

We can see that L2HMC easily mixes between modes (3, 5, 8, and plausibly 9 in the figure) while the approximate posterior gets stuck on the same reconstructed digit (3 in the figure).Visualization of the posterior After training a decoder with L2HMC, we randomly choose an element x 0 ∈ D and run 512 parallel L2HMC chains for 20, 000 Metropolis-Hastings steps.

We then find the direction of highest variance, project the samples along that direction and show a histogram in FIG3 .

This plot shows non-Gaussianity in the latent space for the posterior.

Using our improved sampler enables the decoder to make use of a more expressive posterior, and enables the encoder to sample from this non-Gaussian posterior.

The loss in Section 4.2 targets lag-one autocorrelation.

It should be possible to extend this to also target lag-two and higher autocorrelations.

It should also be possible to extend this loss to reward fast decay in the autocorrelation of other statistics of the samples, for instance the sample energy as well as the sample position.

These additional statistics could also include learned statistics of the samples, combining benefits of the adversarial approach of BID46 with the current work.

Our learned generalization of HMC should prove complementary to several other research directions related to HMC.

It would be interesting to explore combining our work with the use of HMC in a minibatch setting BID9 ; with shadow Hamiltonians BID25 ; with gradient pre-conditioning approaches similar to those used in Riemannian HMC BID15 BID6 ; with the use of alternative HMC accept-reject rules BID44 BID5 ; with the use of non-canonical Hamiltonian dynamics BID49 ; with variants of AIS adapted to HMC proposals BID43 ; with the extension of HMC to discrete state spaces BID52 ; and with the use of alternative Hamiltonian integrators BID10 BID8 .Finally, our work is also complementary to other methods not utilizing gradient information.

For example, we could incorporate the intuition behind Multiple Try Metropolis schemes BID32 by having several parametric operators and training each one when used.

In addition, one could draw inspiration from the adaptive literature BID18 BID0 or component-wise strategies BID13 .

In this work, we presented a general method to train expressive MCMC kernels parameterized with deep neural networks.

Given a target distribution p, analytically known up to a constant, our method provides a fast-mixing sampler, able to efficiently explore the state space.

Our hope is that our method can be utilized in a "black-box" manner, in domains where sampling constitutes a huge bottleneck such as protein foldings BID41 or physics simulations BID37 ....

DISPLAYFORM0 Figure 4: Diagram of our L2HMC-DGLM model.

Nodes are functions of their parents.

Round nodes are deterministic, diamond nodes are stochastic and the doubly-circled node is observed.

Let (x τ ) τ ≤T be a set of correlated samples converging to the distribution p with mean µ and covariance Σ. We define auto-correlation at time t as: DISPLAYFORM0 We can now define effective sample size (ESS) as: DISPLAYFORM1 Similar to Hoffman & Gelman (2014), we truncate the sum when the auto-correlation goes below 0.05.

We compare our trained sampler with LAHMC BID44 .

Results are reported in Table 1 .

L2HMC largely outperforms LAHMC on all task.

LAHMC is also unable to mix between modes for the MoG task.

We also note that L2HMC could be easily combined with LAHMC, by replacing the leapfrog integrator of LAHMC with the learned one of L2HMC.

In this section, we present our training algorithm as well as a diagram explaining L2HMC-DGLM.

For conciseness, given our operator L θ , we denote by K θ (·|x) the distribution over next state given sampling of a momentum and direction and the Metropolis-Hastings step.

Similar to our L2HMC training on unconditional sampling, we share weights across Q, S and T .

In addition, the auxiliary variable x (here the image from MNIST) is first passed through a 2-layer neural network, with softplus non-linearities and 512 hidden units.

This input is given to both

@highlight

General method to train expressive MCMC kernels parameterized with deep neural networks. Given a target distribution p, our method provides a fast-mixing sampler, able to efficiently explore the state space.

@highlight

Proposes a generalized HMC by modifying the leapfrog integrator using neural networks to make the sampler to converge and mix quickly. 
The extended Kalman filter (EKF) is a classical signal processing algorithm which performs efficient approximate Bayesian inference in non-conjugate models by linearising the local measurement function, avoiding the need to compute intractable integrals when calculating the posterior.

In some cases the EKF outperforms methods which rely on cubature to solve such integrals, especially in time-critical real-world problems.

The drawback of the EKF is its local nature, whereas state-of-the-art methods such as variational inference or expectation propagation (EP) are considered global approximations.

We formulate power EP as a nonlinear Kalman filter, before showing that linearisation results in a globally iterated algorithm that exactly matches the EKF on the first pass through the data, and iteratively improves the linearisation on subsequent passes.

An additional benefit is the ability to calculate the limit as the EP power tends to zero, which removes the instability of the EP-like algorithm.

The resulting inference scheme solves non-conjugate temporal Gaussian process models in linear time, $\mathcal{O}(n)$, and in closed form.

Temporal Gaussian process (GP, Rasmussen and Williams, 2006 ) models can be solved in linear computational scaling, O(n), in the number of data n (Hartikainen and Särkkä, 2010) .

However, non-conjugate (i.e., non-Gaussian likelihood) GP models introduce a computational problem in that they generally involve approximating intractable integrals in order to update the posterior distribution when data is observed.

The most common numerical method used in such scenarios is sigma-point integration (Kokkala et al., 2016) , with Gauss-Hermite cubature being a popular way to choose the sigma-point locations and weights.

A drawback of this method is that the number of cubature points scales exponentially with the dimensionality d. Lower-order sigma-point methods allow accuracy to be traded off for scalability, for example the unscented transform (which forms the basis for the unscented Kalman filter, see Särkkä, 2013) requires only 2d + 1 cubature points.

One significant alternative to cubature methods is linearisation.

Although such an approach has gone out of fashion lately, García-Fernández et al. (2015) showed that a globally iterated version of the statistically linearised filter (SLF, Särkkä, 2013) , which performs linearisation w.r.t.

the posterior rather than the prior, performs in line with expectation propagation (EP, Minka, 2001 ) in many modelling scenarios, whilst also providing local convergence guarantees (Appendix D explains the connection to our proposed method).

Crucially, linearisation guarantees that the integrals required to calculate the posterior have a closed form solution, which results in significant computational savings if d is large.

Motivated by these observations, and with the aim of illustrating the connections between classical filtering methods and EP, we formulate power EP (PEP, Minka, 2004) as a Gaussian filter parametrised by a set of local likelihood approximations.

The linearisations used to calculate these approximations are then refined during multiple passes through the data.

We show that a single iteration of our approach is identical to the extended Kalman filter (EKF, Jazwinski, 1970) , and furthermore that we are able to calculate exactly the limit as the EP power tends to zero, since there are no longer any intractable integrals that depend on the power.

The result is a global approximate inference algorithm for temporal GPs that is efficient and stable, easy to implement, scales to problems with large data and high-dimensional latent states, and consistently outperforms the EKF.

We consider non-conjugate (i.e., non-Gaussian likelihood) Gaussian process models with one-dimensional inputs t (i.e., time) which have a dual kernel (left) and discrete state space (right) form ,

R s is the latent state vector containing the GP dynamics.

Each x (i) k contains the state dynamics for one latent GP, for example a Matérn-5 /2 GP prior is modelled with x

The hyerparameters θ of the kernel K θ determine the state transition matrix A θ,k and the process noise q k ∼ N(0, Q θ,k ).

The measurement model h(x k , r k ) is a (nonlinear) function of x k and the observation noise r k ∼ N(0, R k ).

Our aim is to calculate the posterior over the latent states, p(x k | y 1 , . . .

, y n ) for k < n, otherwise known as the smoothing solution, which can be obtained via application of a Gaussian filter (to obtain the filtering solution p(x k | y 1 , . . .

, y k )) followed by a Gaussian smoother.

If h(·) is linear then the Kalman filter and Rauch-Tung-Striebel (RTS, Särkkä, 2013) smoother return the optimal solution.

Gaussian filtering and smoothing As with most approximate inference methods, we approximate the filtering distributions with Gaussians, p(x k | y 1:k ) ≈ N(x k ; m k , P k ).

The prediction step remains the same as in the standard Kalman filter, with the resulting distribution acting as the EP cavity on the forward (filtering) pass: m

To account for the non-Gaussian likelihood in the update step we follow Nickisch et al. (2018) , introducing an intermediary step in which the parameters of the approximate likelihoods, N(x k ; m

, are set via a moment matching procedure and stored before continuing with the Kalman updates.

This PEP formulation, with power α, makes use of the fact that the required moments can be calculated via the derivatives of the log-normaliser, Z k , of the tilted distribution (see Seeger, 2005) , giving

After the mean and covariance of our new likelihood approximation have been calculated, we can proceed with a modified set of linear Kalman filter updates,

As in Wilkinson et al. (2019) , we augment the standard RTS smoother with another moment matching step where the cavity distribution is calculated by removing (a fraction α of) the local likelihood from the marginal smoothing distribution p( R a×s and

∈ R a×a are the Jacobian of h(·) evaluated at the mean w.r.t.

x k and r k respectively.

This new Gaussian form means the moment matching step becomes,

where

is zero (see Deisenroth and Mohamed, 2012 , for discussion).

Therefore,

Now we update the approximate likelihood in closed form (Appendix B gives the derivation),

The result when we use Eq. (7) (with α = 1) to modify the filter updates, Eq. (3), is exactly the EKF (see Appendix C for the proof).

Additionally, since these updates are now available in closed form, a variational free energy method (α → 0, see Bui et al., 2017 ) is simple to implement and doesn't require any matrix subtractions and inversions in Eq. (4), which can be costly and unstable.

Taking α → 0 prior to linearisation is not possible because the intractable integrals also depend on α.

Appendix A describes our full iterative algorithm.

In Fig. 2 , we compare our approach (EKF-PEP, α = 1) to EP and the EKF on two nonconjugate GP tasks (see Appendix E for the full formulations).

Whilst our method is suited to large datasets, we focus here on small time series for ease of comparison.

In the left-hand plot, a log-Gaussian Cox process (approximated with a Poisson model for 200 equal time interval bins) is used to model the intensity of coal mining accidents.

EKF-PEP and the EKF match the EP posterior well, with EKF-PEP obtaining an even tighter match to both the mean and marginal variances.

The right-hand plot shows a similar comparison for 133 accelerometer readings in a simulated motorcycle crash, using a heteroscedastic noise model.

Linearisation in this model is a crude approximation to the true likelihood, but we observe that iteratively refining the linearisation vastly improves the posterior is some regions.

This new perspective on linearisation in approximate inference unifies the PEP and EKF paradigms for temporal data, and provides an improvement to the EKF that requires no additional implementation effort.

Key areas for further exploration are the effect of adjusting α (i.e., changing the cavity and the linearisation point), and the use of statistical linearisation as an alternative method for obtaining the local approximations.

Appendix A. The proposed globally iterated EKF-PEP algorithm Algorithm 1 Globally iterated extended Kalman filter with power EP-style updates

and discretised state space model h, H, J x , J r , α measurement model, Jacobian and EP power m 0 ← 0, P 0 ← P ∞ , e 1:n = 0 initial state while not converged do iterated EP-style loop for k = 1 to n do forward pass (FILTERING)

evaluate Jacobian

Here we derive in full the closed form site updates after linearisation.

Plugging the derivatives from Eq. (6) into the updates in Eq. (2) we get,

By the matrix inversion lemma, and withR

so that

where

Applying the matrix inversion lemma for a second time we obtain

We can also write

Together the above calculations give the approximate site mean and covariance as

Appendix C. Analytical linearisation in EP (α = 1) results in an iterated version of the EKF

Here we prove that a single pass of the proposed EP-style algorithm with linearisation is exactly equivalent to the EKF.

Plugging the closed form site updates, Eq. (7), with α = 1 (since the filter predictions can be interpreted as the cavity with the full site removed), into our modified Kalman filter update equations, Eq. (3), we get a new set of Kalman updates in which the latent noise terms are determined by scaling the observation noise with the Jacobian of the state:

This can be rewritten to explicitly show that there are two innovation covariance terms, S k andŜ k , which act on the state mean and covariance separately:

Now we calculate the inverse ofŜ k :

and the inverse of S k :

which shows thatŜ

and hence, recalling thatR k = J r k R k J r k , Eq. (15) simplifies to give exactly the extended Kalman filter updates: EKF update step:

Posterior linearisation (García-Fernández et al., 2015) is a filtering algorithm that iteratively refines local posterior approximations based on statistical linear regression (SLR), and can be seen as a globally iterated extension of the SLR filter (Särkkä, 2013) .

The idea is that the measurement function is linearised with respect to the posterior, rather than the prior, which is particularly beneficial when the measurement noise is small, such that the prior and posterior can have very different locations and variance.

One drawback of using SLR is that it does not generally result in closed form updates, however it does provide local convergence guarantees.

We have shown in Section 2 that on the first filtering pass our proposed algorithm is equivalent to the EKF.

However, the power EP formulation of the smoothing pass, Eq. (4), iteratively refines the approximate likelihood parameters in the context of the posterior (with a fraction of the local likelihood removed).

Letting α → 0 during the cavity calculation in Eq. (4) implies that the expectations are now with respect to the full marginal posterior.

This shows that PLF is a version of our algorithm in which α = 0 and analytical linearisation is replaced with SLR.

This motivates the following observation: posterior linearisation is a variational free energy method in which the intractable integrals required for posterior calculation are solved via linearisation of the likelihood mean function.

This is intuitive since the formulation of the PLF is based on minimizing local KL divergences.

The local convergence analysis in García-Fernández et al. (2015) depends on using SLR as the linearisation method and initialising the state sufficiently close to a fixed point.

However, it now becomes apparent why both the PLF and our algorithm are generally more stable than EP: no covariance subtractions and inversions are necessary in calculating the cavity distribution, which avoids the possibility of negative-definite covariance matrices.

Log-Gaussian Cox process The coal mining dataset contains the dates of 191 coal mine explosions in Britain between the years 1851-1962, discretised into n = 200 equal time interval bins.

We use a log-Gaussian Cox process to model this count data.

Assuming the process has locally constant intensity in the subregions allows a Poisson likelihood to be used for each bin,

, where we define f

However, the Poisson is a discrete probability distribution and the EKF applies to continuous observations.

Therefore we use a Gaussian approximation to the Poisson likelihood, noticing that the first two moments of the Poisson distribution are equal to the intensity λ k = exp f

Heteroscedastic noise model The motorcycle crash experiment consists of 131 simulated readings from an accelerometer on a motorcycle helmet during impact.

A single GP is not a good model for this data due to the heteroscedasticity of the observation noise, therefore it is common to model the noise separately.

We model the process with one GP for the mean and another for the time varying observation noise.

Letting r k ∼ N(0, 1), we place a GP prior over f (1) and f (2) , both with Matern-3 /2 kernels, f (1) (t) ∼ GP 0, κ θ 1 (t, t ) , f (2) (t) ∼ GP 0, κ θ 2 (t, t ) ,

k ) = N(f

h(x k , r k ) = f

(1)

where φ(z) = log(1 + exp(z)).

In practice a problem arises when linearising this likelihood model.

Since the mean of r k = 0, the Jacobian of the noise term disappears when evaluated at the mean regardless of the value of f (2) .

Hence we reformulate the model to improve identifiability,h (x k , r k ) = (y k − f Left is the EKF-PEP method and right is the PEP equivalent.

The top plots are the posterior for f (1) (t) (the mean process), the middle plots show the posterior for f (2) (t) (the observation noise process), and the bottom plots are the full model.

<|TLDR|>

@highlight

We unify the extended Kalman filter (EKF) and the state space approach to power expectation propagation (PEP) by solving the intractable moment matching integrals in PEP via linearisation. This leads to a globally iterated extension of the EKF.
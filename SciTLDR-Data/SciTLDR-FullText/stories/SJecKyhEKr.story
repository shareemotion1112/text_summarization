Deterministic models are approximations of reality that are often easier to build and interpret than stochastic alternatives.

Unfortunately, as nature is capricious, observational data can never be fully explained by deterministic models in practice.

Observation and process noise need to be added to adapt deterministic models to behave stochastically, such that they are capable of explaining and extrapolating from noisy data.

Adding process noise to deterministic simulators can induce a failure in the simulator resulting in no return value for certain inputs -- a property we describe as ``brittle.''

We investigate and address the wasted computation that arises from these failures, and the effect of such failures on downstream inference tasks.

We show that performing inference in this space can be viewed as rejection sampling, and train a conditional normalizing flow as a proposal over noise values such that there is a low probability that the simulator crashes, increasing computational efficiency and inference fidelity for a fixed sample budget when used as the proposal in an approximate inference algorithm.

In order to compensate for epistemic uncertainty due to modelling approximations and unmodeled aleatoric uncertainty, deterministic simulators are often "converted" to "stochastic" simulators by randomly perturbing the state at each time step.

In practice, models adapted in this way often provide better inferences (Møller et al., 2011; Saarinen et al., 2008; Lv et al., 2008; Pimblott and LaVerne, 1990; Renard et al., 2013) .

State-independent white noise with heuristically tuned variance is often used to perturb the state (Adhikari and Agrawal, 2013; Brockwell and Davis, 2016; Fox, 1997; Reddy and Clinton, 2016; Du and Sam, 2006; Allen, 2017; Mbalawata et al., 2013) .

However, naively adding noise to the state will, in many applications, render the perturbed input state "invalid," inducing failure (Razavi et al., 2019; Lucas et al., 2013; Sheikholeslami et al., 2019) .

These failures waste computational resources and reduce sample diversity, worsening inference performance.

Examples of failure modes include ordinary differential equation (ODE) solvers not converging to the required tolerance in the allocated time, or, the state crossing into an unhandled configuration, such as solid bodies overlapping.

Establishing the cause of failure is non-trivial and hence, the simulation artifact can be sensitive to seemingly inconsequential alterations to the state -a property we describe as "brittle."

The principal contribution of this paper is a technique for minimizing this failure rate.

We proceed by first framing sampling from brittle simulators as rejection sampling.

We then eliminate rejections by learning the state-dependent density over perturbations that do not induce failure, using conditional autoregressive flows (Papamakarios et al., 2017) .

Doing so renders the joint distribution unchanged and retains the interpretability afforded by the simulator, but improves sample efficiency.

We show that using the learned proposal increases the fidelity of the inference results attainable on a range of examples.

We denote the brittle deterministic simulator as f : X → {X , ⊥} , X = R D , where a return value of ⊥ denotes failure.

Over the whole support, f defines a many-to-one function, as many states map to ⊥, however we only require that f is one-to-one in the accepted region, a condition satisfied by ODE models.

A stochastic, additive perturbation to state, denoted z t ∈ X , is proposed such that x t ← f (x t−1 + z t ), z t ∼ p(·|x t−1 ), although this is often state independent.

We include more detailed derivations in Supplementary Materials Section B.

The naive approach to iterate the perturbed system is to repeatedly sample from the proposal distribution and evaluate f until the simulator successfully exists.

We begin by showing that this process defines a rejection sampler.

The use of f and p(z t |·) implicitly specifies a distribution over successfully iterated states, p(x t |x t−1 ); and consequently a second distribution over accepted perturbations, denoted p(z t |x t−1 ), which, under the process outlined above, can be written as: p(z t |x t−1 ) = 1 Mp p(z t |x t−1 ), if f (x t−1 + z t ) = ⊥ 0, otherwise

where the normalizing constant M p is the acceptance rate under p. In regions that fail the sample is rejected with certainty.

In the accepted region,p ∝ p, which is a sufficient condition for a rejection sampler to be valid, without needing to evaluate M p orp.

This represents a rejection sampler with an acceptance rule of I [f (x t−1 + z t ) = ⊥] targetinḡ p(x t |x t−1 ).

We now seek to learn a proposal distribution over z t values conditioned on the current state x t−1 , denoted q φ , parameterized by φ, to replace p but placing no mass on regions that are rejected, resulting in an acceptance rate tending to unity.

We denote q φ as the proposal we train, which, coupled with the simulator, implicitly defines a proposal over accepted samples, denoted q φ .

We wish to minimize the distance between joint distribution implicitly specified over accepted iterated states using p, f and q φ , amortized across state space (Le et al., 2016) :

Expanding the Kullback-Leibler divergence (D KL ), applying a change of variables yields, and noting that the Jacobian terms can be cancelled yields:

However, q φ is defined implicitly after rejection sampling, and so we can adapt (1) for q φ and substitute back into (3).

Differentiation of the acceptance rate (M q φ ) is intractable.

However, noting that φ * is a maximizer for both q φ andq φ , we can instead optimize q φ (z t |x t−1 ):

By removing the rejection sampler as we have, we have implicitly specified that the proposal distribution must have an acceptance rate of one.

This term is differentiable with respect to φ and so we can maximize this quantity using stochastic gradients.

Importantly the flow is not explicitly trained to maximize the acceptance rate.

The flow is trained to minimize the KL divergence between the implicitly specified distribution over accepted samples and the learned proposal distribution.

Accordingly the flow retains the shape of the proposal distribution in regions of state space that do not yield failure (this can be seen by comparing red and green contours in the interior of the dashed lines in Figure  1a ) and hence the learned distribution cannot collapse to add trivially small perturbations, as would be the case if we had directly optimized for high acceptance rates.

By exploiting change of variables we are able to "project" back through the rejection sampling procedure and hence we can optimize q φ as we do not need to compute the derivative of the rejection rate as we would have otherwise needed to do.

Finally, we note that generation of data training data and learning of the autoregressive flow is a computationally intensive procedure.

However simulators can take on the order of seconds to iterate and so the intention of this work is to create a technique that maximizes computational efficiency when deployed.

Sampling from the flow takes on the order of milliseconds, can be accelerated using GPUs and scale favourably in the number of samples being produced.

Furthermore, the training procedure is performed once and hence represents an offline, one-off cost exchanged for higher efficiency deployment.

The training data can also be generated using large-scale distributed computing (as the mini-batching process is inherently embarrassingly parallelizable) that may not be available or practical for use at deployment time.

Implementation We use a conditional masked autoregressive flow (Papamakarios et al., 2017) as the structure of q φ , with 5 single-layer MADE blocks (Germain et al., 2015) , 256 hidden units per layer and batch normalization layers at the input to each intermediate MADE block.

The dimensionality of the flow is the number of states perturbed in the original model.

To introduce conditioning we use the current state vector, x t−1 , as input to a hypernetwork (Ha et al., 2016) that outputs the parameters for each layer in the flow.

The networks are implemented in PyTorch (Paszke et al., 2017) , and optimized using ADAM (Kingma and Ba, 2014).

We demonstrate on two examples here, and an additional two experiments in the appendix.

In these experiments we first aim to demonstrate that learning the required conditional autoregressive flow is tractable and faithfully represents the conditional distribution over accepted perturbations.

We then use the learned proposal in a particle-based sequential (c) Figure 1 : Results for the annulus problem introduced in Section 3.1.

1a indicates the permissible region as a black dashed band, where p and q φ is shown in red and green respectively.

The shape of q φ is the same as p inside the band, with little mass outside of the band.

This shows the flow has learnedp effectively with a low rejection rate.

1b confirms q φ all-but eliminates rejection.

1c shows the reduction in the variance of the evidence across 100 independent sequential Monte Carlo sweeps of 100 independent datasets.

Monte Carlo state-space inference scheme and show that lower-variance inference results can be obtained for a fixed sample budget.

In this example, the (unknown) true generative model of the observed data is a constant speed circular orbit around the origin in the x-y plane.

We perform inference using a misspecified model that only simulates constant velocity forward motion, such that x t ∈ R 4 , with Gaussian perturbations to position and velocity.

We impose a failure constraint limiting the change in the distance of the point from the origin to a fixed threshold.

This condition mirrors the notion that states in brittle simulators have large allowable covariances in particular directions, but very narrow permissible perturbations in other directions.

Figure 1a and Figure 1b shows q φ has effectively learnedp(z t |x t−1 ), reducing rejection rate under q φ to less than 4% compared to approximately 75% under p.

We then use the learned q φ as the proposal in a particle filter (Doucet et al., 2001) , an approximate inference algorithm often applied to posterior inference in time-series models.

We use a fixed sample budget and hence failed samples are discarded, without retrying a new sample from the proposal.

The results in Figure 1c show that we are able to recover lower variance evidence approximations using q φ compared to p, achieving a paired t-test score of < 0.0001.

This experiment confirms we are able to learn a proposal that incurs lower rejection, and that reducing the rejection rate increases fidelity of inference (for a fixed computational budget).

We now apply our method to the robotics simulator MuJoCo (Todorov et al., 2012) , using the built-in example "tosser." MuJoCo allows some overlap between solid objects to simulate the contact dynamics.

This is an example of model misspecification borne out of the requirements of reasonably writing a simulator.

We therefore place a hard limit on the amount objects are allowed to overlap.

We add Gaussian perturbations to the state.

Results of the "tosser" experiment introduced in Section 3.2.

2a shows a typical state evolution.

2b shows the conditional autoregressive flow we learn markedly reduces the number of rejections.

2c shows the results of performing sequential Monte Carlo using p and q φ .

2d shows the results of performing hypothesis testing, where the correct hypothesis (3) not selected using p, but is using q φ .

Figure 2 shows the results of this experiment.

Collisions are generally rare events and hence the rejection rate of p is just 10%.

Figure 2b shows that the autoregressive flow learns a proposal with a significantly lower rejection rate, reaching 3% rejection.

However these rejections are concentrated in the critical regions of state-space and so this reduction yields an large reduction in the variance of the evidence approximation, as shown in Figure 2c .

We conclude by applying our method to hypothesis testing, selecting the mass of the capsule.

Shown in Figure 2d , using p results in higher variance evidence approximations than when q φ is used, causing p to select the wrong model, with a reasonable level of significance (p = 0.125), while using q φ selects the correct hypothesis with p = 0.0127.

In this paper we have tackled reducing simulator failures caused by naively perturbing the input state.

We achieve this by defining these simulators as rejection samplers and learning a conditional autoregressive flow to estimate the state-dependent proposal distribution conditioned on acceptance.

We show that using this learned proposal reduces the variance of inference results when used as the proposal in a subsequent approximate inference scheme.

This work has readily transferable practical contributions in the scientific community where naively modified simulation platforms are widely deployed.

Journal of basic Engineering, 82 (1) Appendix A. Background

Deterministic simulators are often stochastically perturbed to increase the diversity of the achievable simulations and to fit data more effectively.

White noise perturbation to time series systems is common, such as the widely used ARMA models (Adhikari and Agrawal, 2013; Brockwell and Davis, 2016) .

The most straightfoward example of this however is the widely used Kalman filter (Kalman et al., 1960) .

The Kalman filter, at its core, is determinstic transition model which is then perturbed with additive Gaussian noise.

The form of the process and noise kernels are chosen such the system has a closed form representation.

Without the additive process noise, the Kalman filter is deterministic and would be unable to represent the variability in the real-world.

More complex systems cannot be analyzed in closed form like the Kalman filter.

Accordingly deterministic simulators of the dynamics with stochastic perturbations and numerical methods are used in practice.

Specific examples of such systems that are: stochastic Hodgkin Huxley models of neural dynamics (Fox, 1997; Coutin et al., 2018; Goldwyn and Shea-Brown, 2011; Saarinen et al., 2008) , computational finance analysis of asset prices (Gamba, 2003; Reddy and Clinton, 2016; Kalogeropoulos et al., 2010) , predator-prey dynamics (Du and Sam, 2006) , epidemiology (Allen, 2017) and mobile robotics (Thrun et al., 2001; Fallon et al., 2012) .

As simulators become more complex, guaranteeing robustness is more difficult, and individual function evaluations are more expensive.

Lucas et al. (2013) and Edwards et al. (2011) establish the sensitivity of earth science models to static input parameter values by building a discriminative classifer for parameters that induce failure.

Sheikholeslami et al. (2019) take an alternative approach instead treating simulator failure as an imputation problem, fitting a function regressor to predict the outcome of the failed experiment given the neighbouring experiments that successfully terminated.

However these methods are limited by the lack of clear probabilistic interpretation in terms of the originally specified joint distribution in time series models and their ability to scale to high dimensions.

Autoregressive flows (AFs) (Papamakarios et al., 2017 ) are a flexible class of density estimators.

AFs define a density, q φ (x), trainable using stochastic gradient descent to approximate the target distribution p(x), by minimizing the KL-divergence between the target distribution and the approximation:

AFs operate by transforming samples from a "base distribution" through a series of learned warpings, interpretable as a change of variables, into samples distributed according to the target distribution.

The flow layers are designed such that the required Jacobians and inverses are cheaply computable.

A popular flow structure is the masked autoencoder for distribution estimation (Germain et al., 2015) , or MADE, that facilitates GPU-based parallelization.

Multiple MADE blocks are used in masked autoregressive flows (MAF) (Papamakarios et al., 2017) overcoming the ordering dependency of autoregressive flows.

AFs are also capable of learning conditional distributions by making the parameters of the flow dependent on the data using hypernetworks (Ha et al., 2016) .

We include here a more complete derivation of the results presented in the main text.

The overarching aim of this work is to develop a flexible proposal over perturbations that places minimal mass on perturbations that cause the simulator to not return a value, while not changing the originally specified model.

Doing so reduces the wasted computational cost incurred by simulations failing, and also increases the effective sample size for a given sample budget.

We consider deterministic models, expressed as simulators, describing the time-evolution of a state x t ∈ X , where we denote application of the simulator iterating the state as x t ← f (x t−1 ).

However, brittle simulators fail for "invalid" inputs, which we denote as a return value of ⊥ (read as "bottom") from the simulator.

Hence the complete definition of f is f : X → {X , ⊥}. We denote the region of valid inputs as X A ⊂ X , and the region of invalid inputs as X R ⊂ X , such that X A ∪ X R = X , where the boundary between these regions is unknown.

Over the whole support, f defines a many-to-one function, as all X R maps to ⊥. However, the algorithm we go on to derive only requires that f is one-to-one in the accepted region.

This is not uncommon in real simulators, and is satisfied by, for example, ODE models.

A stochastic, additive perturbation to state, denoted z t ∈ X , is applied to induce a distribution over states.

The distribution of this perturbation is denoted p(z t |x t−1 ), although, in practice, this distribution is often state independent.

The iterated state is therefore calculated as x t ← f (x t−1 + z t ).

We define the random variable A t ∈ {0, 1} to denote whether the perturbation (as x t−1 is being conditioned on) is accepted.

The naive approach to sampling from the perturbed system, shown in Algorithm 1, is to repeatedly sample from the proposal distribution and evaluate f until the simulator successfully exists.

This procedure defines A t = I [f (x t−1 + z t ) = ⊥] , z t ∼ p(z t |x t−1 ), i.e. successfully iterated samples are accepted with certainty.

This approach incurs significant wasted computation as the simulator must be called repeatedly, with failed iterations being discarded.

Therefore the objective of this work is to derive a more efficient sampling mechanism.

We begin by showing that Algorithm 1 defines a rejection sampler, with a specific form, targeting the space of successfully iterated states.

This reasoning is illustrated in Figure 3 .

The behavior of f and the distribution p(z t |·) implicitly define a distribution over successfully iterated states.

We denote this "target" distribution as p(x t |x t−1 ) = p(x t |x t−1 , A t = 1), where the bar indicates that the sample was accepted, and hence places no probability mass on failures.

Note there is no bar on p(z t |x t−1 ) above, indicating that it is defined Algorithm 1 Sampling from a brittle simulator.

Data: Current state x t−1 , brittle simulator f , perturbation proposal p(z t |x t−1 ).

Result: Iterated state x t and perturbation z t .

Figure 3: Graphical representation of how a brittle deterministic simulator acts as a rejection sampler, targetingp(z t |x t−1 ).

For clarity here we assume x t=1 = 0 and z t is independent of x t .

The simulator, f (z t ), returns ⊥ for some unknown input regions, shown in green.

The proposal over z t is shown in blue.

The target distribution,p(z t ), shown in orange, is implicitly defined asp(z t ) = 1 Mp p(z t )I [f (z t ) = ⊥], where M p is the normalizing constant from p, equal to the acceptance rate.

Accordingly the proposal distribution, scaled by M p , is exactly equal top(z t ) in the accepted region.

Therefore sampling from p until f successfully exits, as in Algorithm 1, can be seen as constructing a rejection sampler with proposal p(z t ), and acceptance ratio,p with no knowledge of the accept/reject behaviors of f and hence probability mass may be placed on regions that yield failure.

The functional form ofp is unavailable, and the density cannot be evaluated for any input value.

Importantly,p is the distribution specified a-priori by the modeler, sampled from by the entire simulation pipeline, and hence any algorithm we develop must also targetp(x t |x t−1 ).

The existence of p(x t |x t−1 ) implies the existence of a second distribution: the distribution over accepted perturbations, denoted p(z t |x t−1 ).

Note that this distribution is also conditioned on acceptance under the chosen simulator indicated be the presence of a bar.

We assume f is one-to-one in the accepted regime, and so the change of variables rule can be applied to directly relate this to p(x t |x t−1 ).

Under our initial algorithm for sampling from a brittle simulator we can therefore write the following identity:

where the normalizing constant M p is the acceptance rate under p. By inspecting (6), accepting with certainty perturbations that exit successfully can be seen as proportionally shifting mass from regions of p where the simulator does not exit to regions where it does.

In a rejection sampler, the probability of accepting a proposed sample is proportional to the ratio between the target distribution and the proposal, scaled by a constant such that:

As we have already stated, we cannot evaluate the target density, but we can establish if the density is non-zero (indicated by the simulator not failing).

A sufficient condition to ensure the correctness of a rejection sampler in this scenario is that the proposal density is proportional to the target density wherever the target density has support.

Applying this condition to our scenario implies that if the simulator fails, the density under the target distribution is known to be zero and the sample should be rejected with certainty, regardless of the density under the proposal distribution.

In the accepted region, the sample should be accepted with probability p(z t |·)/M p(z t |·), where M is selected to satisfy (7).

However, from (6), it can be seen p ∝ p hence proposal and target are proportional irrespective of the choice of M , and the value of M p , satisfying the above criteria.

The acceptance rule of the rejection sampler is therefore reduced to I [f (x t−1 + z t ) = ⊥].

Importantly, we do not need to evaluate M p , M , orp to use Algorithm 1 as a valid rejection sampler.

This simple probabilistic interpretation of the behavior of the simulation process enables us to establish (6) as a definition ofp valid across the entire input domain of f -a definition we now exploit to learn an efficient proposal.

We now derive how we can learn the proposal distribution, denoted q φ parameterized by φ, to replace p, such that the acceptance rate under q φ (denoted M q φ ) tends towards unity, minimizing wasted computation, while also retaining the same joint distribution as the originally specified model.

We denote q φ as the proposal we train, which, coupled with the simulator, implicitly define a proposal over accepted samples, denoted q φ .

Expressing this mathematically, we wish to minimize the distance between joint distribution implicitly specified over accepted iterated states using the a-priori specified proposal distribution p, and the joint distribution defined implicitly as q φ :

where we select the Kullback-Leibler (KL) divergence as the metric of distance between distributions.

The outer expectation defines this objective as amortized across state space.

As is standard in amortized and compiled inference methods we can generate the samples by directly sampling from the model (Le et al., 2016; Gershman and Goodman, 2014) .

We eventually perform this minimization using stochastic gradient descent, and so this expectation defines the distribution from which we sample the minibatches used, and so we drop this expectation for compactness.

Expanding the KL term yields:

Noting that q φ and p are defined only on accepted samples, where f is one-to-one, we can apply a change of variables defined for q φ as:

and likewise for p. This transforms the distribution over x t into a distribution over z t and a Jacobian term:

Noting that the same Jacobian terms appear in the numerator and denominator we are able to cancel these:

taking care to also apply the change variables in the distribution we are sampling from in (9).

We can now discard the remaining p term as it is independent of φ, and noting that f −1 (x t ) = x t−1 + z t we can write:

It can now be read off that minimizing the KL stated in (9) can be performed by setting q φ (z t |x t−1 ) equal to p(z t |x t−1 ).

Had we have discarded p a step earlier, we would have been unable to eliminate the Jacobian terms inside the logarithm.

However, this distribution is defined after rejection sampling, and can only be defined as in (6):

denoting M q φ as the acceptance rate under q φ .

Note again that q φ is not dependent on the accept/reject characteristics of f .

Differentiation of M q φ is intractable.

Further, there is an infinite family of q φ proposals that yield p = q φ , that have non-zero rejection rates.

However, we observe that φ * is a maximizer for both q φ andq φ in the limit of zero rejection, and so we can instead optimize q φ (z t |x t−1 ):

with no consideration of the rejection behavior.

Additionally, by removing the rejection sampler as we have, we have implicitly specified that the proposal distribution must have an acceptance rate of one.

This term is differentiable with respect to φ and so we can maximize this quantity using stochastic gradients, where samples from the outer expectation over x t−1 defines the distribution from which we sample minibatches from.

This expression shows that we can learn the distribution over accepted x t values by learning the distribution over z t , without needing to calculate the Jacobian or inverse of the transformation defined by f .

We can now perform density estimation on the accepted samples from the a-priori specified rejection sampler to learn a proposal for accepted x t samples, thus minimizing wasted computation, targeting the same overall joint distribution, and retaining interpretability by utilizing the simulator.

In this section we include additional results figures and experimental details for the annulus and MuJoCo experiment presented in the main text, along with an additional two experiments.

Our first additional example uses a simulator of balls elastically bouncing in a square enclosure, as shown in Figure 4a .

The dimensionality of the state vector, x t , is four times the number of balls -the x-y coordinate and velocity of the centre of mass, per ball.

We add a small amount of Gaussian noise at each iteration to the position and velocity of each ball.

This perturbation induces the possibility that two balls overlap, or, a ball intersects with the wall.

Both of these represent invalid physical arrangements and so the simulator returns ⊥ for such configurations.

We note that here, we are conditioning on the state of both balls simultaneously, and proposing the perturbation to the state jointly.

Figure 4 displays the results of this experiment.

Figure 5 shows the distribution over x-y perturbations of a single ball, conditioned on the other ball being static and stationary.

Green contours show the perturbations learned by autoregressive flow such that failure is not induced.

In Figure 4c we plot the rejection rate under p and q φ as a function of the position of the first ball, with the second ball fixed in the position shown, showing that rejection has been all but eliminated.

We again see a reduction in the variance of the evidence approximation when comparing p and q in an SMC scheme, as shown in Figure 4d .

This example demonstrates the applicability of the autoregressive flow; but also demonstrates how a seemingly simple simulator becomes brittle when naively perturbed.

shows the rejection rate as a function of the position of the first ball, with the second ball in the position shown.

The trained proposal (right) has all but eliminated rejection in the permissible space compared to the a-priori specified proposal (left).

The rejection rate under p is much higher in the interior as the second ball may also leave the enclosure, whereas q φ has practically eliminated rejection by jointly proposing perturbations.

4d shows the reduction in variance achieved by using q φ .

Although the reduction appears more modest compared to, say, the annulus example, it still achieves a paired t-test score of < 0.0001, indicating a strong level of statistical significance.

We conclude by applying our algorithm to a simulator for the widely studied Caenorhabditis elegans roundworm.

WormSim, presented by Boyle et al. (2012) , is a simulator of the body of the worm, driven by a surrogate for the true neural architecture of Caenorhabditis elegans, and uses a 510 dimensional state representation.

We apply perturbations to the 98 dimensional subspace defining the 49 x-y coordinate pairs physical position of the worm, while conditioning on the full 510 dimensional state vector.

The expected rate of failure increases sharply as a function of the scale of the perturbation applied, as shown in Figure  6a , as the integrator used in WormSim is unable to integrate highly perturbed states.

We then train a autoregressive flow targetingp, where the rejection rate during training is shown in Figure 6b .

We see that we are able to learn an autoregressive flow with lower rejection rates, reaching approximately 53% rejection, for a p with approximately 75% rejection.

Although the rejection rate is higher than ultimately desired, we include this example to show how rejections occur in real simulators through integrator failure.

We believe larger flows with regularized parameters can reduce the rejection rate further.

Figure 5: Shown is the a-priori specified proposal distribution, p, over the perturbation to the position of the first ball specified in the model, and the learned proposal, q φ in green, for the bouncing balls experiment introduced in Section C.1.

The edge of the permissible region of the enclosure is shown as a black dashed line.

The second ball is fixed at [25, 15] , and the resulting invalid region induced shaded.

The flow has learned to deflect away from the disallowed regions.

x t ← x t−1 + δt ×ẋ t−1 + z xt , z xt ∼ N (0, 0.1), y t ← y t−1 + δt ×ẏ t−1 + z yt , z xt ∼ N (0, 0.1),

x t ←ẋ t−1 + zẋ t , z xt ∼ N (0, 0.1), y t ←ẏ t−1 + zẏ t , z yt ∼ N (0, 0.1),

Failure is defined as the change in radius being greater than 0.03.

To compute the variances of the SMC sweep we generate 100 random traces.

We then perform 50 SMC sweeps per trace, using 100 particles, and compute the evidence.

We use the configuration "tosser" included in MuJoCo Todorov et al. (2012) , only modifying it by removing the second unused bucket.

We use completely standard simulation configuration.

We introduce the limit on overlap leveraging MuJoCos in-built collision detection, rejecting overlaps above 0.005.

Typical overlaps in the standard execution of MuJoCo are below this limit.

An integration time of 0.002 is used.

We observe only the x-y position of the capsule with Gaussian distributed noise, with standard deviation 0.1.

We perturb the x-y position and velocity of the capsule with Gaussian distributed noise, with standard deviation 0.005 and 0.1 respectively.

We perturb the angle and angular velocity of the capsule with Gaussian distributed noise, with standard deviation 0.05 and 0.05 respectively.

These values were chosen to be in line with typical simulated values in the tosser example.

We place a prior over the initial position and velocity with standard deviation 0.01 for positions and 0.1 for velocities, and mean equal to their true position.

In this, the state input to the normalizing flow is the position and angle, and derivatives, of the capsule, as well as the state of the actuator.

The actuators state is unobserved and is not perturbed under the model.

We also input time into the normalizing flow as the control dynamics are not constant with time.

To compute the variances of the SMC sweep we generate 50 random traces.

We then perform 20 SMC sweeps per trace, using 100 particles, and compute the evidence.

@highlight

We learn a conditional autoregressive flow to propose perturbations that don't induce simulator failure, improving inference performance.
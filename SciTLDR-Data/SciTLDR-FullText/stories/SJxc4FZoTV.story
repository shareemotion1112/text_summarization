While recent developments in autonomous vehicle (AV) technology highlight substantial progress, we lack tools for rigorous and scalable testing.

Real-world testing, the de facto evaluation environment, places the public in danger, and, due to the rare nature of accidents, will require billions of miles in order to statistically validate performance claims.

We implement a simulation framework that can test an entire modern autonomous driving system, including, in particular, systems that employ deep-learning perception and control algorithms.

Using adaptive sampling methods to accelerate rare-event probability evaluation, we estimate the probability of an accident under a base distribution governing standard traffic behavior.

We demonstrate our framework on a highway scenario.

Several fatal accidents involving autonomous vehicles (AVs) underscore the importance of testing whether AV perception and control pipelines-when considered as a whole systemcan safely interact with other human traffic participants.

Unfortunately, testing AVs in real environments, the most straightforward validation framework for system-level inputoutput behavior, requires prohibitive amounts of time due to the rare nature of serious accidents BID22 .

Concretely, a recent study BID8 argues that AVs need to drive "hundreds of millions of miles, and, under some scenarios, hundreds of billions of miles to create enough data to clearly demonstrate their safety."

On the other hand, formally verifying an AV algorithm's "correctness" BID11 BID0 BID21 BID13 is inherently difficult because all driving policies are subject to crashes caused by other drivers BID22 .

Ruling out scenarios where the AV should not be blamed for such accidents is a task subject to logical inconsistency and subjective assignment of fault.

Motivated by the challenges underlying real-world testing and formal verification, we consider a probabilistic paradigmwhich we describe as a risk-based framework BID14 -where the goal is to evaluate the probability of an accident under a base distribution representing standard traffic behavior.

By assigning learned probabilities to environmental states and agent behaviors, our risk-based framework considers performance of the AV policy under a data-driven model of the world.

Formally, we let P 0 denote the base distribution that models standard traffic behavior and X ∼ P 0 be a realization of the simulation (e.g. weather conditions and driving policies of other agents).

For an objective function f : X → R that measures "safety"-so that low values of f (x) correspond to dangerous scenarios-our goal is to evaluate the probability of a dangerous event p γ := P 0 (f (X) ≤ γ) for some threshold γ.

Our riskbased framework is agnostic to the complexity of the ego-policy and views it as a black-box module.

Importantly, this approach allows deep-learning based perception systems that make formal verification methods intractable.

For control algorithms which approach or exceed humanlevel performance, an adverse event will be rare, and the probability p γ close to 0.

Thus, estimating p γ is a rare event simulation problem (see BID1 for an overview of this topic).

For rare probabilities p γ , the naive Monte Carlo method can be prohibitively inefficient.

For a sample X i DISPLAYFORM0 DISPLAYFORM1 In order to achieve -relative accuracy, we need N 1−pγ pγ 2 rollouts from our simulator.

For light-tailed f (X), then p γ ∝ exp(−γ) so that the required sample size for naive Monte Carlo grows exponentially in γ.

In the next section, we use adaptive sampling techniques that sample unsafe events more frequently to make the evaluation of p γ tractable.

To address the shortcomings of naive Monte Carlo for estimating rare event probabilities, we use a multilevel splitting method BID7 BID3 BID5 BID24 ] that decomposes the rare-event probability p γ into conditional probabilities with interim threshold levels DISPLAYFORM0 This decomposition introduces a product of non-rare probabilities that are easy to estimate individually.

Markov chain Monte Carlo (MCMC) is used to estimate each term, which can be accurately estimated as long as consecutive levels are close and the conditional probability is therefore large.

Intuitively, the splitting method iteratively steers samples X i to the rare set {X|f (X) < γ} through a series of supersets {X|f (X) DISPLAYFORM1 Since it is a priori unclear how to choose the levels ∞ =: The discarding fraction δ trades off two dueling objectives; for small values, each term in the product (1) is large and hence easy to estimate by MCMC; for large values, the number of total iterations K until convergence is reduced and more samples (δN ) at each iteration can be simulated in parallel.

DISPLAYFORM2 The AMS approach complements other procedures such as adaptive importance sampling (AIS) methods.

AIS methods require computation of likelihood ratios between different distributions and become numerically unstable in high dimensions.

AMS does not require computation of likelihood ratios, nor does it need to postulate models for the form of the optimal importance sampling distribution P 0 (·|f (·) < γ).

On the other hand, the "modes" of failure AMS discovers are limited by the number of samples and the mixing properties of the MCMC sampler employed.

Contrary to model-based AIS methods such as the cross-entropy method BID19 , AMS has several convergence guarantees including those for bias, variance, and runtime (see BID4 for details).

Notably, AMS is unbiased and has relative variance which scales as log(1/p γ ) as opposed to 1/p γ for naive Monte Carlo (cf.

Section II).

Intuitively, AMS computes O(log(1/p γ )) independent probabilities, each of which has variance independent of p γ .

To implement our risk-based framework, we first learn a base distribution P 0 of nominal traffic behavior.

Using videos of highway traffic in the NGSim BID12 dataset, we train policies of human drivers via imitation learning BID20 BID17 BID18 BID6 BID2 .

It has recently been observed that supervised approaches to imitationlearning-where expert data is used to predict actions given vehicle states-suffer from poor performance in regions of the state space not encountered in data BID17 BID18 .

Reinforcementlearning techniques such as generative adversarial imitation learning (GAIL) BID6 improve generalization performance, as the imitation agent explores novel regions of the state space during training.

We use the model-based variant of GAIL (MGAIL) BID2 that allows end-to-end differentiation.

GAIL has been validated by Kuefler et al. BID10 to realistically mimic human-like driving behavior from the NGSim dataset across multiple metrics.

These include the similarity of low-level actions (speeds, accelerations, turn-rates, jerks, and time-to-collision), as well as higher-level behaviors (lane change rate, collision rate, hard-brake rate, etc).We consider a scenario consisting of six agents, five of which are considered part of the environment.

The environment vehicles' policies follow the distribution learned via GAIL.

All vehicles are constrained to start within a set of possible initial configurations consisting of pose and velocity, and each vehicle has a goal of reaching the end of the approximately 2 km stretch of road.

We created a photorealistic simulator of the portion of I-80 in Emeryville, CA where the traffic data was collected BID12 (see Appendix B for details).

FIG1 details the performance of the AMS algorithm on the scenario, where the risk metric f (X) is the minimum time-to-collision (TTC) over a rollout (cf.

Appendix C).

For events with probability 10 −5 or less, AMS outperforms naive Monte Carlo, and the variance of the failure probability is reduced by up to 56×.

We can also combine AMS with importance sampling.

Since AMS outputs particles sampled from the desired failure region(s), we simply fit a model to this empirical distribution.

FIG2 shows the output from 10 5 samples from a normalizing-flow-based importance sampler (cf.

Appendix D) built upon the output of AMS.

It is significantly more efficient than naive sampling.

A fundamental tradeoff emerges when comparing the requirements of our risk-based framework to other testing paradigms.

Real-world testing endangers the public but is still in some sense a gold standard.

Verified subsystems provide evidence that the AV should drive safely in all specified scenarios; they are limited by computational intractability and require both white-box models and a complete specifications for assigning blame (e.g. BID22 ).

In turn, our risk-based framework is most useful when the base distribution P 0 is accurate.

Although an estimate of p γ is not informative when P 0 is misspecified, our adaptive sampling techniques still efficiently identify dangerous scenarios in this case; such dangerous scenarios are independent of potentially subjective assignments of blame.

Principled techniques for building and validating the model of the environment P 0 represent an open research question.

Rigorous safety evaluation of AVs necessitates benchmarks based on adaptive adversarial conditions rather than standard nominal conditions.

Importantly, our framework only requires black-box access to the driving policy and simulation environment.

Our approach offers significant speedups over realworld testing and allows efficient evaluation of black-box AV input/output behavior, providing a powerful tool to aid in the design of safe AVs.

DISPLAYFORM0 Evaluate and sort f (X i ) in decreasing order BID5 : DISPLAYFORM1 Discard X (1) , . . .

, X (δN ) and reinitialize by resampling with replacement from X (δN +1) , . . .

, X (N )

Apply T MCMC transitions separately to each of X (1) , . . .

, X (δN ) 11: end while

Our simulator is a distributed, modular framework, which is necessary to support the inclusion of new AV systems and updates to the environment-vehicle policies.

A benefit of this design is that simulation rollouts are simple to parallelize.

In particular, we allow instantiation of multiple simulations simultaneously, without requiring that each include the entire set of components.

For example, a desktop may support only one instance of Unreal Engine (for perception pipelines) but could be capable of simulating 10 physics simulations in parallel; it would be impossible to fully utilize the compute resource with a monolithic executable wrapping all engines together.

Our architecture enables instances of components to be distributed on heterogeneous GPU clusters while maintaining the ability to perform meaningful analysis locally on commodity desktops.

Using the asynchronous messaging library ZeroMQ, our implementation is fully-distributed among available CPUs and GPUs; our rollouts are up to 30P times faster than real time, where P is the number of processors.

A video of a rollout from the simulator is available at https://youtu.be/CLXJ0CitDck and a snapshot from this rollout is shown in FIG4 .

In our implementation the safety measure is minimum timeto-collision (TTC).

TTC is defined as the time it would take for two vehicles to intercept one another given that they each maintain their current heading and velocity BID23 .

The TTC between the ego-vehicle and vehicle i is given by DISPLAYFORM0 where r i is the distance between the ego vehicle and vehicle i, andṙ i the time derivative of this distance (which is simply computed by projecting the relative velocity of vehicle i onto the vector between the vehicles' poses).

The operator [·] + is defined as [x] + := max(x, 0).

We define T T C i (t) = ∞ foṙ r i (t) ≥ 0.

In this paper, vehicles are described as oriented rectangles in the 2D plane.

Since we are interested in the time it would take for the ego-vehicle to intersect the polygonal boundary of another vehicle on the road, we utilize a finite set of range and range measurements in order to approximate the TTC metric.

For a given configuration of vehicles, we compute N uniformly spaced angles θ 1 , . . .

, θ N in the range [0, 2π] with respect to the ego vehicle's orientation and cast rays outward from the center of the ego vehicle.

For each direction we compute the distance which a ray could travel before intersecting one of the M other vehicles in the environment.

These form N range measurements s 1 , . . .

, s N .

Further, for each ray s i , we determine which vehicle (if any) that ray hit; projecting the relative velocity of this vehicle with respect to ego vehicle gives the range-rate measurementṡ i .

Finally, we approximate the minimum TTC for a given simulation rollout X of length T discrete time steps by: DISPLAYFORM1 where we again define the approximate instantaneous TTC as ∞ forṡ i (t) ≥ 0.

Note that this measure can approximate the true TTC arbitrarily well via choice of N and the discretization of time used by the simulator.

Furthermore, note that our definition of TTC is with respect to the center of the ego vehicle touching the boundary of another vehicle.

Crashing, on the other hand, is defined in our simulation as the intersection of boundaries of two vehicles.

Thus, TTC values we evaluate in our simulation are nonzero even during crashes, since the center of the ego vehiclehas not yet collided with the boundary of another vehicle.

Normalizing flows are used to describe classes of distributions using multi-layer neural networks, which are more expressive than typical analytical parameterizations such as exponential families.

First a base distribution (usually something easy to sample from e.g. a standard normal distribution) is chosen.

Then a series of transformations are applied to the samples from this distribution.

Note that if the transformations are invertible, it is possible to work backwards from new samples to determine their likelihood using standard methods.

Suppose a non-linear function y = f (x) is applied to x ∈ X then we can determine the density p(y) as follows: DISPLAYFORM0 where J(·) denotes the Jacobian.

Composing a sequence of such transforms f 1 (x), f 2 (f 1 (x)) . . .

allows expressive transformations to the base density.

Rezende and Mohamed BID16 describe a modern version of the approach which ensures that the Jacobian is upper triangular, rendering the determinant computation efficient.

Given that each transform is parameterized by trainable weights, the architecture can be used to learn a density by maximizing the log-probability of the observed data in the transformed distribution p(y).

Further enhancements to this architecture for fitting distributions can be found in BID9 BID15 .

@highlight

Using adaptive sampling methods to accelerate rare-event probability evaluation, we estimate the probability of an accident under a base distribution governing standard traffic behavior. 
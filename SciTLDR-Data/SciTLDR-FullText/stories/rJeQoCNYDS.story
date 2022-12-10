Transfer and adaptation to new unknown environmental dynamics is a key challenge for reinforcement learning (RL).

An even greater challenge is performing near-optimally in a single attempt at test time, possibly without access to dense rewards, which is not addressed by current methods that require multiple experience rollouts for adaptation.

To achieve single episode transfer in a family of environments with related dynamics, we propose a general algorithm that optimizes a probe and an inference model to rapidly estimate underlying latent variables of test dynamics, which are then immediately used as input to a universal control policy.

This modular approach enables integration of state-of-the-art algorithms for variational inference or RL.

Moreover, our approach does not require access to rewards at test time, allowing it to perform in settings where existing adaptive approaches cannot.

In diverse experimental domains with a single episode test constraint, our method significantly outperforms existing adaptive approaches and shows favorable performance against baselines for robust transfer.

One salient feature of human intelligence is the ability to perform well in a single attempt at a new task instance, by recognizing critical characteristics of the instance and immediately executing appropriate behavior based on experience in similar instances.

Artificial agents must do likewise in applications where success must be achieved in one attempt and failure is irreversible.

This problem setting, single episode transfer, imposes a challenging constraint in which an agent experiences-and is evaluated on-only one episode of a test instance.

As a motivating example, a key challenge in precision medicine is the uniqueness of each patient's response to therapeutics (Hodson, 2016; Bordbar et al., 2015; Whirl-Carrillo et al., 2012) .

Adaptive therapy is a promising approach that formulates a treatment strategy as a sequential decision-making problem (Zhang et al., 2017; West et al., 2018; Petersen et al., 2019) .

However, heterogeneity among instances may require explicitly accounting for factors that underlie individual patient dynamics.

For example, in the case of adaptive therapy for sepsis (Petersen et al., 2019) , predicting patient response prior to treatment is not possible.

However, differences in patient responses can be observed via blood measurements very early after the onset of treatment (Cockrell and An, 2018) .

As a first step to address single episode transfer in reinforcement learning (RL), we propose a general algorithm for near-optimal test-time performance in a family of environments where differences in dynamics can be ascertained early during an episode.

Our key idea is to train an inference model and a probe that together achieve rapid inference of latent variables-which account for variation in a family of similar dynamical systems-using a small fraction (e.g., 5%) of the test episode, then deploy a universal policy conditioned on the estimated parameters for near-optimal control on the new instance.

Our approach combines the advantages of robust transfer and adaptation-based transfer, as we learn a single universal policy that requires no further training during test, but which is adapted to the new environment by conditioning on an unsupervised estimation of new latent dynamics.

In contrast to methods that quickly adapt or train policies via gradients during test but assume access to multiple test rollouts and/or dense rewards (Finn et al., 2017; Killian et al., 2017; Rakelly et al., 2019) , we explicitly optimize for performance in one test episode without accessing the reward function at test time.

Hence our method applies to real-world settings in which rewards during test are highly delayed or even completely inaccessible-e.g., a reward that depends on physiological factors that are accessible only in simulation and not from real patients.

We also consider computation time a crucial factor for real-time application, whereas some existing approaches require considerable computation during test (Killian et al., 2017) .

Our algorithm builds on variational inference and RL as submodules, which ensures practical compatibility with existing RL workflows.

Our main contribution is a simple general algorithm for single episode transfer in families of environments with varying dynamics, via rapid inference of latent variables and immediate execution of a universal policy.

Our method attains significantly higher cumulative rewards, with orders of magnitude faster computation time during test, than the state-of-the-art model-based method (Killian et al., 2017) , on benchmark high-dimensional domains whose dynamics are discontinuous and continuous in latent parameters.

We also show superior performance over optimization-based meta-learning and favorable performance versus baselines for robust transfer.

Our goal is to train a model that performs close to optimal within a single episode of a test instance with new unknown dynamics.

We formalize the problem as a family (S, A, T , R, γ), where (S, A, R, γ) are the state space, action space, reward function, and discount of an episodic Markov decision process (MDP).

Each instance of the family is a stationary MDP with transition function T z (s |s, a) ∈ T .

When a set Z of physical parameters determines transition dynamics (Konidaris and Doshi-Velez, 2014) , each T z has a hidden parameter z ∈ Z that is sampled once from a distribution P Z and held constant for that instance.

For more general stochastic systems whose modes of behavior are not easily attributed to physical parameters, Z is induced by a generative latent variable model that indirectly associates each T z to a latent variable z learned from observed trajectory data.

We refer to "latent variable" for both cases, with the clear ontological difference understood.

Depending on application, T z can be continuous or discontinuous in z. We strictly enforce the challenging constraint that latent variables are never observed, in contrast to methods that use known values during training (Yu et al., 2017) , to ensure the framework applies to challenging cases without prior knowledge.

This formulation captures a diverse set of important problems.

Latent space Z has physical meaning in systems where T z is a continuous function of physical parameters (e.g., friction and stiffness) with unknown values.

In contrast, a discrete set Z can induce qualitatively different dynamics, such as a 2D navigation task where z ∈ {0, 1} decides if the same action moves in either a cardinal direction or its opposite (Killian et al., 2017) .

Such drastic impact of latent variables may arise when a single drug is effective for some patients but causes serious side effects for others (Cockrell and An, 2018) .

Training phase.

Our training approach is fully compatible with RL for episodic environments.

We sample many instances, either via a simulator with controllable change of instances or using off-policy batch data in which demarcation of instances-but not values of latent variables-is known, and train for one or more episodes on each instance.

While we focus on the case with known change of instances, the rare case of unknown demarcation can be approached either by preprocessing steps such as clustering trajectory data or using a dynamic variant of our algorithm (Appendix C).

Single test episode.

In contrast to prior work that depend on the luxury of multiple experience rollouts for adaptation during test time (Doshi-Velez and Konidaris, 2016; Killian et al., 2017; Finn et al., 2017; Rakelly et al., 2019) , we introduce the strict constraint that the trained model has access to-and is evaluated on-only one episode of a new test instance.

This reflects the need to perform near-optimally as soon as possible in critical applications such as precision medicine, where an episode for a new patient with new physiological dynamics is the entirety of hospitalization.

We present Single Episode Policy Transfer (SEPT), a high-level algorithm for single episode transfer between MDPs with different dynamics.

The following sections discuss specific design choices in SEPT, all of which are combined in synergy for near-optimal performance in a single test episode.

Our best theories of natural and engineered systems involve physical constants and design parameters that enter into dynamical models.

This physicalist viewpoint motivates a partition for transfer learning in families of MDPs: 1. learn a representation of latent variables with an inference model that rapidly encodes a vectorẑ of discriminative features for a new instance; 2. train a universal policy π(a|s, z) to perform near-optimally for dynamics corresponding to any latent variable in Z; 3. immediately deploy both the inference model and universal policy on a given test episode.

To build on the generality of model-free RL, and for scalability to systems with complex dynamics, we do not expend computational effort to learn a model of T z (s |s, a), in contrast to model-based approaches (Killian et al., 2017; Yao et al., 2018) .

Instead, we leverage expressive variational inference models to represent latent variables and provide uncertainty quantification.

In domains with ground truth hidden parameters, a latent variable encoding is the most succinct representation of differences in dynamics between instances.

As the encodingẑ is held constant for all episodes of an instance, a universal policy π(a|s, z) can either adapt to all instances when Z is finite, or interpolate between instances when T z is continuous in z (Schaul et al., 2015) .

Estimating a discriminative encoding for a new instance enables immediate deployment of π(a|s, z) on the single test episode, bypassing the need for further fine-tuning.

This is critical for applications where further training complex models on a test instance is not permitted due to safety concerns.

In contrast, methods that do not explicitly estimate a latent representation of varied dynamics must use precious experiences in the test episode to tune the trained policy (Finn et al., 2017) .

In the training phase, we generate an optimized

of short trajectories, where each

Tp ) is a sequence of early state-action pairs at the start of episodes of instance T i ∈ T (e.g. T p = 5).

We train a variational auto-encoder, comprising an approximate posterior inference model q φ (z|τ ) that produces a latent encodingẑ from τ and a parameterized generative model p ψ (τ |z).

The dimension chosen forẑ may differ from the exact true dimension when it exists but is unknown; domain knowledge can aid the choice of dimensionality reduction.

Because dynamics of a large variety of natural systems are determined by independent parameters (e.g., coefficient of contact friction and Reynolds number can vary independently), we consider a disentangled latent representation where latent units capture the effects of independent generative parameters.

To this end, we bring β-VAE (Higgins et al., 2017) into the context of families of dynamical systems, choosing an isotropic unit Gaussian as the prior and imposing the constraint D KL (q φ (z|τ i ) p(z)) < .

The β-VAE is trained by maximizing the variational lower bound L(ψ, φ; τ i ) for each τ i across D:

This subsumes the VAE (Kingma and Welling, 2014) as a special case (β = 1), and we refer to both as VAE in the following.

Since latent variables only serve to differentiate among trajectories that arise from different transition functions, the meaning of latent variables is not affected by isometries and hence the value ofẑ by itself need not have any simple relation to a physically meaningful z even when one exists.

Only the partition of latent space is important for training a universal policy.

Earlier methods for a family of similar dynamics relied on Bayesian neural network (BNN) approximations of the entire transition function s t+1 ∼T (BNN) z (s t , a t ), which was either used to perform computationally expensive fictional rollouts during test time (Killian et al., 2017) or used indirectly to further optimize a posterior over z (Yao et al., 2018) .

Our use of variational inference is more economical: the encoder q φ (z|τ ) can be used immediately to infer latent variables during test, while the decoder p ψ (τ |z) plays a crucial role for optimized probing in our algorithm (see Section 3.3).

In systems with ground truth hidden parameters, we desire two additional properties.

The encoder should produce low-variance encodings, which we implement by minimizing the entropy of q φ (z|τ ):

under a diagonal Gaussian parameterization, where σ 2 d = Var(q φ (z|τ )) and dim(z) = D. We add −H(q φ (z|τ )) as a regularizer to equation 1.

Second, we must capture the impact of z on higher-order dynamics.

While previous work neglects the order of transitions (s t , a t , s t+1 ) in a trajectory (Rakelly et al., 2019) , we note that a single transition may be compatible with multiple instances whose differences manifest only at higher orders.

In general, partitioning the latent space requires taking the ordering of a temporally-extended trajectory into account.

Therefore, we parameterize our encoder q φ (z|τ ) using a bidirectional LSTM-as both temporal directions of (s t , a t ) pairs are informative-and we use an LSTM decoder p ψ (τ |z) (architecture in Appendix E.2).

In contrast to embedding trajectories from a single MDP for hierarchical learning (Co-Reyes et al., 2018), our purpose is to encode trajectories from different instances of transition dynamics for optimal control.

We train a single universal policy π(a|s, z) and deploy the same policy during test (without further optimization), for two reasons: robustness against imperfection in latent variable representation and significant improvement in scalability.

Earlier methods trained multiple optimal policies {π *

on training instances with a set {z i } N i=1 of hidden parameters, then employed either behavioral cloning (Yao et al., 2018) or off-policy Q-learning (Arnekvist et al., 2019) to train a final policy π(a|s, z) using a dataset {(s t ,ẑ i ; a t ∼ π * i (a|s t ))}.

However, this supervised training scheme may not be robust (Yu et al., 2017) : if π(a|s, z) were trained only using instance-specific optimal state-action pairs generated by π * i (a|s) and posterior samples ofẑ from an optimal inference model, it may not generalize well when faced with states and encodings that were not present during training.

Moreover, it is computationally infeasible to train a collection {π

-which is thrown away during test-when faced with a large set of training instances from a continuous set Z. Instead, we interleave training of the VAE and a single policy π(a|s, z), benefiting from considerable computation savings at training time, and higher robustness due to larger effective sample count.

To execute near-optimal control within a single test episode, we first rapidly computeẑ using a short trajectory of initial experience.

This is loosely analogous to the use of preliminary medical treatment to define subsequent prescriptions that better match a patient's unique physiological response.

Our goal of rapid inference motivates two algorithmic design choices to optimize this initial phase.

First, the trajectory τ used for inference by q φ (z|τ ) must be optimized, in the sense of machine teaching (Zhu et al., 2018) , as certain trajectories are more suitable than others for inferring latent variables that underlie system dynamics.

If specific degrees of freedom are impacted the most by latent variables, an agent should probe exactly those dimensions to produce an informative trajectory for inference.

Conversely, methods that deploy a single universal policy without an initial probing phase (Yao et al., 2018) can fail in adversarial cases, such as when the initial placeholderẑ used in π θ (a|s, ·) at the start of an instance causes failure to exercise dimensions of dynamics that are necessary for inference.

Second, the VAE must be specifically trained on a dataset D of short trajectories consisting of initial steps of each training episode.

We cannot expend a long trajectory for input to the encoder during test, to ensure enough remaining steps for control.

Hence, single episode transfer motivates the machine teaching problem of learning to distinguish among dynamics: our algorithm must have learned both to generate and to use a short initial trajectory to estimate a representation of dynamics for control.

Our key idea of optimized probing for accelerated latent variable inference is to train a dedicated probe policy π ϕ (a|s) to generate a dataset D of short trajectories at the beginning of all training episodes, such that the VAE's performance on D is optimized 2 .

Orthogonal to training a meta-policy for faster exploration during standard RL training (Xu et al., 2018) , our probe and VAE are trained for the purpose of performing well on a new test MDP.

For ease of exposition, we discuss the case with access to a simulator, but our method easily allows use of off-policy batch data.

We start each training episode using π ϕ for a probe phase lasting T p steps, record the probe trajectory τ p into D, train the VAE using minibatches from D, then use τ p with the encoder to generateẑ for use by π θ (a|s,ẑ) to complete the remainder of the episode (Algorithm 1).

At test time, SEPT only requires lines 5, 8, and 9 in Algorithm 1 (training step in 9 removed; see Algorithm 2).

The reward function for π ϕ is defined as the VAE objective, approximated by the variational lower bound (1): Initialize encoder φ, decoder ψ, probe policy ϕ, control policy θ, and trajectory buffer D 3:

for each instance Tz with transition function sampled from T do 4:

for each episode on instance Tz do 5:

Execute πϕ for Tp steps and store trajectory τp into D 6:

Use variational lower bound (1) as the reward to train πϕ by descending gradient (3) 7:

Train VAE using minibatches from D for gradient ascent on (1) and descent on (2) 8:

Estimateẑ from τp using encoder q φ (z|τ ) 9:

Execute π θ (a|s, z) withẑ for remaining time steps and train it with suitable RL algorithm 10:

end for 11:

end for 12: end procedure probe to help the VAE's inference of latent variables that distinguish different dynamics (Figure 1 ).

We provide detailed justification as follows.

First we state a result derived in Appendix A: Proposition 1.

Let p ϕ (τ ) denote the distribution of trajectories induced by π ϕ .

Then the gradient of the entropy H(p ϕ (τ )) is given by

Noting that dataset D follows distribution p ϕ and that the VAE is exactly trained to maximize the log probability of D, we use L(ψ, φ; τ ) as a tractable lowerbound on log p ϕ (τ ).

Crucially, to generate optimal probe trajectories for the VAE, we take a minimum-entropy viewpoint and descend the gradient (3).

This is opposite of a maximum entropy viewpoint that encourages the policy to generate diverse trajectories (Co-Reyes et al., 2018), which would minimize log p ϕ (τ ) and produce an adversarial dataset for the VAE-hence, optimal probing is not equivalent to diverse exploration.

The degenerate case of π ϕ learning to "stay still" for minimum entropy is precluded by any source of environmental stochasticity: trajectories from different instances will still differ, so degenerate trajectories result in low VAE performance.

Finally we observe that equation 3 is the defining equation of a simple policy gradient algorithm (Williams, 1992) for training π ϕ , with log p ϕ (τ ) interpreted as the cumulative reward of a trajectory generated by π ϕ .

This completes our justification for defining reward R p (τ ) := L(ψ, φ; τ ).

We also show empirically in ablation experiments that this reward is more effective than choices that encourage high perturbation of state dimensions or high entropy (Section 6).

Figure 1: π ϕ learns to generate an optimal dataset for the VAE, whose performance is the reward for π ϕ .

Encodingẑ by the VAE is given to control policy π θ .

The VAE objective function may not perfectly evaluate a probe trajectory generated by π ϕ because the objective value increases due to VAE training regardless of π ϕ .

To give a more stable reward signal to π ϕ , we can use a second VAE whose parameters slowly track the main VAE according to ψ ← αψ + (1 − α)ψ for α ∈ [0, 1], and similarly for φ .

While analogous to target networks in DQN (Mnih et al., 2015) , the difference is that our second VAE is used to compute the reward for π ϕ .

Transfer learning in a family of MDPs with different dynamics manifests in various formulations (Taylor and Stone, 2009 ).

Analysis of -stationary MDPs and -MDPs provide theoretical grounding by showing that an RL algorithm that learns an optimal policy in an MDP can also learn a near-optimal policy for multiple transition functions (Kalmár et al., 1998; Szita et al., 2002) .

Imposing more structure, the hidden-parameter Markov decision process (HiP-MDP) formalism posits a space of hidden parameters that determine transition dynamics, and implements transfer by model-based policy training after inference of latent parameters (Doshi-Velez and Konidaris, 2016; Konidaris and Doshi-Velez, 2014) .

Our work considers HiP-MDP as a widely applicable yet special case of a general viewpoint, in which the existence of hidden parameters is not assumed but rather is induced by a latent variable inference model.

The key structural difference from POMDPs (Kaelbling et al., 1998) is that given fixed latent values, each instance from the family is an MDP with no hidden states; hence, unlike in POMDPs, tracking a history of observations provides no benefit.

In contrast to multi-task learning (Caruana, 1997) , which uses the same tasks for training and test, and in contrast to parameterized-skill learning (Da Silva et al., 2012) , where an agent learns from a collection of rewards with given task identities in one environment with fixed dynamics, our training and test MDPs have different dynamics and identities of instances are not given.

Prior latent variable based methods for transfer in RL depend on a multitude of optimal policies during training (Arnekvist et al., 2019) , or learn a surrogate transition model for model predictive control with real-time posterior updates during test (Perez et al., 2018) .

Our variational model-free approach does not incur either of these high computational costs.

We encode trajectories to infer latent representation of differing dynamics, in contrast to state encodings in (Zhang et al., 2018) .

Rather than formulating variational inference in the space of optimal value functions (Tirinzoni et al., 2018), we implement transfer through variational inference in a latent space that underlies dynamics.

Previous work for transfer across dynamics with hidden parameters employ model-based RL with Gaussian process and Bayesian neural network (BNN) models of the transition function (Doshi-Velez and Konidaris, 2016; Killian et al., 2017) , which require computationally expensive fictional rollouts to train a policy from scratch during test time and poses difficulties for real-time test deployment.

DPT uses a fully-trained BNN to further optimize latent variable during a single test episode, but faces scalability issues as it needs one optimal policy per training instance (Yao et al., 2018) .

In contrast, our method does not need a transition function and can be deployed without optimization during test.

Methods for robust transfer either require access to multiple rounds from the test MDP during training (Rajeswaran et al., 2017) , or require the distribution over hidden variables to be known or controllable (Paul et al., 2019) .

While meta-learning (Finn et al., 2017; Rusu et al., 2019; Zintgraf et al., 2019; Rakelly et al., 2019) in principle can take one gradient step during a single test episode, prior empirical evaluation were not made with this constraint enforced, and adaptation during test is impossible in settings without dense rewards.

We conducted experiments on three benchmark domains with diverse challenges to evaluate the performance, speed of reward attainment, and computational time of SEPT versus five baselines in the single test episode.

We evaluated four ablation and variants of SEPT to investigate the necessity of all algorithmic design choices.

For each method on each domain, we conducted 20 independent training runs.

For each trained model, we evaluate on M independent test instances, all starting with the same model; adaptations during the single test episode, if done by any method, are not preserved across the independent test instances.

This means we evaluate on a total of 20M independent test instances per method per domain.

Hyperparameters were adjusted using a coarse coordinate search on validation performance.

We used DDQN with prioritized replay (Van Hasselt et al., 2016; Schaul et al., 2016) as the base RL component of all methods for a fair evaluation of transfer performance; other RL algorithms can be readily substituted.

Domains.

We use the same continuous state discrete action HiP-MDPs proposed by Killian et al. (2017) for benchmarking.

Each isolated instance from each domain is solvable by RL, but it is highly challenging, if not impossible, for naïve RL to perform optimally for all instances because significantly different dynamics require different optimal policies.

In 2D navigation, dynamics are discontinuous in z ∈ {0, 1} as follows: location of barrier to goal region, flipped effect of actions (i.e., depending on z, the same action moves in either a cardinal direction or its opposite), and direction of a nonlinear wind.

In Acrobot (Sutton et al., 1998) , the agent applies {+1, 0, −1} torques to swing a two-link pendulum above a certain height.

Dynamics are determined by a vector z = (m 1 , m 2 , l 1 , l 2 ) of masses and lengths, centered at 1.0.

We use four unique instances in training and validation, constructed by sampling ∆z uniformly from {−0.

extrapolation.

In HIV, a patient's state dynamics is modeled by differential equations with high sensitivity to 12 hidden variables and separate steady-state regions of health, such that different patients require unique treatment policies (Adams et al., 2004) .

Four actions determine binary activation of two drugs.

We have M = 10, 5, 5 for 2D navigation, Acrobot, and HIV, respectively.

Baselines.

First, we evaluated two simple baselines that establish approximate bounds on test performance of methods that train a single policy: as a lower bound, Avg trains a single policy π(a|s) on all instances sampled during training and runs directly on test instances; as an upper bound in the limit of perfect function approximation for methods that use latent variables as input, Oracle π(a|s, z) receives the true hidden parameter z during both training and test.

Next we adapted existing methods, detailed in Appendix E.1, to single episode test evaluation: 1.

we allow the model-based method BNN (Killian et al., 2017) to fine-tune a pre-trained BNN and train a policy using BNN-generated fictional episodes every 10 steps during the test episode; 2.

we adapted the adversarial part of EPOpt (Rajeswaran et al., 2017) , which we term EPOpt-adv, by training a policy π(a|s) on instances with the lowest 10-percentile performance; 3.

we evaluate MAML as an archetype of meta-learning methods that require dense rewards or multiple rollouts (Finn et al., 2017) .

We allow MAML to use a trajectory of the same length as SEPT's probe trajectory for one gradient step during test.

We used the same architecture for the RL module of all methods (Appendix E.2).

To our knowledge, these model-free baselines are evaluated on single-episode transfer for the first time in this work.

To investigate the benefit of our optimized probing method for accelerated inference, we designed an ablation called SEPT-NP, in which trajectories generated by the control policy are used by the encoder for inference and stored into D to train the VAE.

Second, we investigated an alternative reward function for the probe, labeled TotalVar and defined as R(τ ) := 1/T p Tp−1 t=1

|s t+1,i − s t,i | for probe trajectory τ .

In contrast to the minimum entropy viewpoint in Section 3.3, this reward encourages generation of trajectories that maximize total variation across all state space dimensions.

Third, we tested the maximum entropy viewpoint on probe trajectory generation, labeled MaxEnt, by giving negative lowerbound as the probe reward: R p (τ ) := −L(ψ, φ; τ ).

Last, we tested whether DynaSEPT, an extension that dynamically decides to probe or execute control (Appendix C), has any benefit for stationary dynamics.

2D navigation and Acrobot are solved upon attaining terminal reward of 1000 and 10, respectively.

SEPT outperforms all baselines in 2D navigation and takes significantly fewer number of steps to solve (Figures 2a and 2c) .

While a single instance of 2D navigation is easy for RL, handling multiple instances is highly non-trivial.

EPOpt-adv and Avg almost never solve the test instance-we set "steps to solve" to 50 for test episodes that were unsolved-because interpolating between instance-specific optimal policies in policy parameter space is not meaningful for any task instance.

MAML did not perform well despite having the advantage of being provided with rewards at test time, unlike SEPT.

The gradient adaptation step was likely ineffective because the rewards are sparse and delayed.

BNN requires significantly more steps than SEPT, and it uses four orders of magnitude longer computation time (Table 4) , due to training a policy from scratch during the test episode.

Training times of all algorithms except BNN are in the same order of magnitude (Table 3) .

In Acrobot and HIV, where dynamics are continuous in latent variables, interpolation within policy space can produce meaningful policies, so all baselines are feasible in principle.

SEPT is statistically significantly faster than BNN and Avg, is within error bars of MAML, while EPOpt-adv outperforms the rest by a small margin (Figures 2b and 2d ).

Figure 5 shows that SEPT is competitive in terms of percentage of solved instances.

As the true values of latent variables for Acrobot test instances were interpolated and extrapolated from the training values, this shows that SEPT is robust to out-oftraining dynamics.

BNN requires more steps due to simultaneously learning and executing control during the test episode.

On HIV, SEPT reaches significantly higher cumulative rewards than all methods.

Oracle is within the margin of error of Avg.

This may be due to insufficient examples of the high-dimensional ground truth hidden parameters.

Due to its long computational time, we run three seeds for BNN on HIV, shown in Figure 4b , and find it was unable to adapt within one test episode.

Comparing directly to reported results in DPT (Yao et al., 2018) , SEPT solves 2D Navigation at least 33% (>10 steps) faster, and the cumulative reward of SEPT (mean and standard error) are above DPT's mean cumulative reward in Acrobot (Table 2) .

Together, these results show that methods that explicitly distinguish different dynamics (e.g., SEPT and BNN) can significantly outperform methods that implicitly interpolate in policy parameter space (e.g., Avg and EPOpt-adv) in settings where z has large discontinuous effect on dynamics, such as 2D navigation.

When dynamics are continuous in latent variables (e.g., Acrobot and HIV), interpolation-based methods fare better than BNN, which faces the difficulty of learning a model of the entire family of dynamics.

SEPT worked the best in the first case and is robust to the second case because it explicitly distinguishes dynamics and does not require learning a full transition model.

Moreover, SEPT does not require rewards at test time allowing it be useful on a broader class of problems than optimization-based meta-learning approaches like MAML.

Appendix D contains training curves.

Figures 2f, 2g and 2j show that the probe phase is necessary to solve 2D navigation quickly, while giving similar performance in Acrobot and significant improvement in HIV.

SEPT significantly outperformed TotalVar in 2D navigation and HIV, while TotalVar gives slight improvement in Acrobot, showing that directly using VAE performance as the reward for probing in certain environments can be more effective than a reward that deliberately encourages perturbation of state dimensions.

The clear advantage of SEPT over MaxEnt in 2D navigation and HIV supports our hypothesis in Section 3.3 that the variational lowerbound, rather than its negation in the maximum entropy viewpoint, should be used as the probe reward, while performance was not significantly differentiated in Acrobot.

SEPT outperforms DynaSEPT on all problems where dynamics are stationary during each instance.

On the other hand, DynaSEPT is the better choice in a non-stationary variant of 2D navigation where the dynamics "switch" abruptly at t = 10 (Figure 4c) .

Figure 3 shows that SEPT is robust to varying the probe length T p and dim(z).

Even with certain suboptimal probe length and dim(z), it can outperform all baselines on 2D navigation in both steps-to-solve and final reward; it is within error bars of all baselines on Acrobot based on final cumulative reward; and final cumulative reward exceeds that of baselines in HIV.

Increasing T p means foregoing valuable steps of the control policy and increasing difficulty of trajectory reconstruction for the VAE in high dimensional state spaces; T p is a hyper-parameter that should be validated for each application.

Appendix D.5 shows the effect of β on latent variable encodings.

We propose a general algorithm for single episode transfer among MDPs with different stationary dynamics, which is a challenging goal with real-world significance that deserves increased effort from the transfer learning and RL community.

Our method, Single Episode Policy Transfer (SEPT), trains a probe policy and an inference model to discover a latent representation of dynamics using very few initial steps in a single test episode, such that a universal policy can execute optimal control without access to rewards at test time.

Strong performance versus baselines in domains involving both continuous and discontinuous dependence of dynamics on latent variables show the promise of SEPT for problems where different dynamics can be distinguished via a short probing phase.

The dedicated probing phase may be improved by other objectives, in addition to performance of the inference model, to mitigate the risk and opportunity cost of probing.

An open challenge is single episode transfer in domains where differences in dynamics of different instances are not detectable early during an episode, or where latent variables are fixed but dynamics are nonstationary.

Further research on dynamic probing and control, as sketched in DynaSEPT, is one path toward addressing this challenge.

Our work is one step along a broader avenue of research on general transfer learning in RL equipped with the realistic constraint of a single episode for adaptation and evaluation.

A DERIVATIONS Proposition 1.

Let p ϕ (τ ) denote the distribution of trajectories induced by π ϕ .

Then the gradient of the entropy H(p ϕ (τ )) is given by

Proof.

Assuming regularity, the gradient of the entropy is

For trajectory τ := (s 0 , a 0 , s 1 , . . . , s t ) generated by the probe policy π ϕ :

Since p(s 0 ) and p(s i+1 |s i , a i ) do not depend on ϕ, we get

Substituting this into the gradient of the entropy gives equation 3.

Restore trained decoder ψ, encoder φ, probe policy ϕ, and control policy θ 3:

Run probe policy π ϕ for T p time steps and record trajectory τ p

Use τ p with decoder q φ (z|τ ) to estimateẑ 5:

Useẑ with control policy π θ (a|s, z) for the remaining duration of the test episode 6: end procedure C DYNASEPT In our problem formulation, it is not necessary to computeẑ at every step of the test episode, as each instance is a stationary MDP and change of instances is known.

However, removing the common assumption of stationarity leads to time-dependent transition functions T z (s |s, a), which introduces problematic cases.

For example, a length T p probing phase would fail if z leads to a switch in dynamics at time t > T p , such as when poorly understood drug-drug interactions lead to abrupt changes in dynamics during co-medication therapies (Kastrin et al., 2018) .

Here we describe an alternative general algorithm for non-stationary dynamics, which we call DynaSEPT.

We train a single policy π θ (a|s, z, η) that dynamically decides whether to probe for better inference or act to maximize the MDP reward R env , based on a scalar-valued function η : R → [0, 1] representing the degree of uncertainty in posterior inference, which is updated at every time step.

The total reward is R tot (τ ) := ηR p (τ ) + (1 − η)R env (τ f ), where τ is a short sliding-window trajectory of length T p , and τ f is the final state of τ .

The history-dependent term R p (τ ) is equivalent to a delayed reward given for executing a sequence of probe actions.

Following the same reasoning for SEPT, one choice for R p (τ ) is L(φ, ψ; τ ).

Assuming the encoder outputs variance σ 2 i of each latent dimension, one choice for η is a normalized standard deviation over all dimensions of the latent variable, i.e.

, where σ i,max is a running max of σ i .

Despite its novelty, we consider DynaSEPT only for rare nonstationary dynamics and merely as a baseline in the predominant case of stationary dynamics, where SEPT is our primary contribution.

DynaSEPT does not have any clear advantage over SEPT when each instance T z is a stationary MDP.

DynaSEPT requires η to start at 1.0, representing complete lack of knowledge about latent variables, and it still requires the choice of hyperparameter T p .

Only after T p steps can it use the uncertainty of q φ (z|τ ) to adapt η and continue to generate the sliding window trajectory to improveẑ.

By this time, SEPT has already generated an optimized sequence using π ϕ for the encoder to estimateẑ.

If a trajectory of length T p is sufficient for computing a good estimate of latent variables, then SEPT is expected to outperform DynaSEPT.

Steps to solve Table 1 reports the number of steps in a test episode required to solve the MDP.

Average and standard deviation were computed across all test instances and across all independently trained models.

If an episode was not solved, the maximum allowed number of steps was used (50 for 2D navigation and 200 for Acrobot).

Table 2 shows the mean and standard error of the cumulative reward over test episodes on Acrobot.

The reported mean cumulative value for DPT in Acrobot is -27.7 (Yao et al., 2018) .

Step 0 Figure 6 shows training curves on all domains by all methods.

None of the baselines, except for Oracle, converge in 2D navigation, because it is meaningless for Avg and EPOpt-adv to interpolate between optimal policies for each instance, and MAML cannot adapt due to lack of informative rewards for almost the entire test episode.

Hence these baselines cannot work for a new unknown test episode, even in principle.

We allowed the same number of training episodes for HIV as in Killian et al. (2017) , and all baselines except MAML show learning progress.

Figure 7: Two-dimensional encodings generated for four instances of Acrobot (represented by four ground-truth colors), for different values of β.

We chose β = 1 for Acrobot.

There is a tradeoff between reconstruction and disentanglement as β increases (Higgins et al., 2017) .

Increasing β encourages greater similarity between the posterior and an isotropic Gaussian.

Figure 7 gives evidence that this comes at a cost of lower quality of separation in latent space.

E EXPERIMENTAL DETAILS For 2D navigation, Acrobot, and HIV, total number of training episodes allowed for all methods are 10k, 4k, and 2.5k, respectively.

We switch instances once every 10, 8 and 5 episodes, respectively.

There are 2, 8 and 5 unique training instances, and 2, 5, and 5 validation instances, respectively.

For each independent training run, we tested on 10, 5, and 5 test instances, respectively.

The simple baselines Average and Oracle can be immediately deployed in a single test episode after training.

However, the other methods for transfer learning require modification to work in the setting of single episode test, as they were not designed specifically for this highly constrained setting.

We detail the necessary modifications below.

We also describe the ablation SEPT-NP in more detail.

BNN.

In Killian et al. (2017) , a pre-trained BNN model was fine-tuned using the first test episode and then used to generate fictional episodes for training a policy from scratch.

More episodes on the same test instance were allowed to help improve model accuracy of the BNN.

In the single test episode setting, all fine-tuning and policy training must be conducted within the first test episode.

We fine-tune the pre-trained BNN every 10 steps and allow the same total number of fictional episodes as reported in (Killian et al., 2017) for policy training.

We measured the cumulative reward attained by the policy-while it is undergoing training-during the single real test episode.

EPOpt.

EPOpt trains on the lowest -percentile rollouts from instances sampled from a source distribution, then adapts the source distribution using observations from the target instance (Rajeswaran et al., 2017) .

Since we do not allow observation from the test instance, we only implemented the adversarial part of EPOpt.

To run EPOpt with off-policy DDQN, we generated 100 rollouts per iteration and stored the lowest 10-percentile into the replay buffer, then executed the same number of minibatch training steps as the number that a regular DDQN would have done during rollouts.

MAML.

While MAML uses many complete rollouts per gradient step (Finn et al., 2017) , the single episode test constraint mandates that it can only use a partial episode for adaptation during test, and hence the same must be done during meta-training.

For both training and test, we allow MAML to take one gradient step for adaptation using a trajectory of the same length as the probe trajectory of SEPT, starting from the initial state of the episode.

We implemented a first-order approximation that computes the meta-gradient at the post-update parameters but omits second derivatives.

This was reported to have nearly equal performance as the full version, due to the use of ReLU activations.

SEPT-NP.

π θ (a|s, z) begins with a zero-vector for z at the start of training.

When it has produced a trajectory τ p of length T p , we store τ p into D for training the VAE, and use τ p with the VAE to estimate z for the episode.

Later training episodes begin with the rolling mean of all z estimated so far.

For test, we give the final rolling mean of z at the end of training as initial input to π θ (a|s, z).

Encoder.

For all experiments, the encoder q φ (z|τ ) is a bidirectional LSTM with 300 hidden units and tanh activation.

Outputs are mean-pooled over time, then fully-connected to two linear output layers of width dim(z), interpreted as the mean and log-variance of a Gaussian over z.

Decoder.

For all experiments, the decoder p ψ (τ |z) is an LSTM with 256 hidden units and tanh activation.

Given input [s t , a t ,ẑ] at LSTM time step t, the output is fully-connected to two linear output layers of width |S| + |A|, and interpreted as the mean and log-variance of a Gaussian decoder for the next state-action pair (s t+1 , a t+1 ).

Q network.

For all experiments, the Q function is a fully-connected neural network with two hidden layers of width 256 and 512, ReLU activation, and a linear output layer of size |A|.

For SEPT and Oracle, the input is the concatenation [s t , z], where z is estimated in the case of SEPT and z is the ground truth in for the Oracle.

For all other methods, the input is only the state s.

Probe policy network.

For all experiments, π ϕ (a|s) is a fully-connected neural network with 3 hidden layers, ReLU activation, 32 nodes in all layers, and a softmax in the output layer.

E.3 HYPERPARAMETERS VAE learning rate was 1e-4 for all experiments.

Size of the dataset D of probe trajectories was limited to 1000, with earliest trajectories discarded.

10 minibatches from D were used for each VAE training step.

We used β = 1 for the VAE.

Probe policy learning rate was 1e-3 for all experiments.

DDQN minibatch size was 32, one training step was done for every 10 environment steps, end = 0.15, learning rate was 1e-3, gradient clip was 2.5, γ = 0.99, and target network update rate was 5e-3.

Exploration decayed according n+1 = c n every episode, where c satisfies end = c N start and N is the total number of episodes.

Prioritized replay used the same parameters in (Killian et al., 2017) .

@highlight

Single episode policy transfer in a family of environments with related dynamics, via optimized probing for rapid inference of latent variables and immediate execution of a universal policy.
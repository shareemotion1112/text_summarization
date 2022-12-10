Model-based reinforcement learning (RL) is considered to be a promising approach to reduce the sample complexity that hinders model-free RL.

However, the theoretical understanding of such methods has been rather limited.

This paper introduces a novel algorithmic framework for designing and analyzing model-based RL algorithms with theoretical guarantees.

We design a meta-algorithm with a theoretical guarantee of monotone improvement to a local maximum of the expected reward.

The meta-algorithm iteratively builds a lower bound of the expected reward based on the estimated dynamical model and sample trajectories, and then maximizes the lower bound jointly over the policy and the model.

The framework extends the optimism-in-face-of-uncertainty principle to non-linear dynamical models in a way that requires no explicit uncertainty quantification.

Instantiating our framework with simplification gives a  variant of model-based RL algorithms Stochastic Lower Bounds Optimization (SLBO).

Experiments demonstrate that SLBO achieves the state-of-the-art performance when only 1M or fewer samples are permitted on a range of continuous control benchmark tasks.

In recent years deep reinforcement learning has achieved strong empirical success, including superhuman performances on Atari games and Go (Mnih et al., 2015; BID21 and learning locomotion and manipulation skills in robotics BID33 BID18 Lillicrap et al., 2015) .

Many of these results are achieved by model-free RL algorithms that often require a massive number of samples, and therefore their applications are mostly limited to simulated environments.

Model-based deep reinforcement learning, in contrast, exploits the information from state observations explicitly -by planning with an estimated dynamical model -and is considered to be a promising approach to reduce the sample complexity.

Indeed, empirical results BID14 Deisenroth et al., 2013; BID33 Nagabandi et al., 2017; Kurutach et al., 2018; Pong et al., 2018a) have shown strong improvements in sample efficiency.

Despite promising empirical findings, many of theoretical properties of model-based deep reinforcement learning are not well-understood.

For example, how does the error of the estimated model affect the estimation of the value function and the planning?

Can model-based RL algorithms be guaranteed to improve the policy monotonically and converge to a local maximum of the value function?

How do we quantify the uncertainty in the dynamical models?It's challenging to address these questions theoretically in the context of deep RL with continuous state and action space and non-linear dynamical models.

Due to the high-dimensionality, learning models from observations in one part of the state space and extrapolating to another part sometimes 0 * indicates equal contribution 1 The source code of this work is available at https://github.com/roosephu/slbo involves a leap of faith.

The uncertainty quantification of the non-linear parameterized dynamical models is difficult -even without the RL components, it is an active but widely-open research area.

Prior work in model-based RL mostly quantifies uncertainty with either heuristics or simpler models (Moldovan et al., 2015; BID33 BID13 .Previous theoretical work on model-based RL mostly focuses on either the finite-state MDPs (Jaksch et al., 2010; BID4 Fruit et al., 2018; Lakshmanan et al., 2015; Hinderer, 2005; Pirotta et al., 2015; 2013) , or the linear parametrization of the dynamics, policy, or value function BID0 BID22 BID11 BID27 BID29 , but not much on non-linear models.

Even with an oracle prediction intervals 2 or posterior estimation, to the best of our knowledge, there was no previous algorithm with convergence guarantees for model-based deep RL.Towards addressing these challenges, the main contribution of this paper is to propose a novel algorithmic framework for model-based deep RL with theoretical guarantees.

Our meta-algorithm (Algorithm 1) extends the optimism-in-face-of-uncertainty principle to non-linear dynamical models in a way that requires no explicit uncertainty quantification of the dynamical models.

Let V π be the value function V π of a policy π on the true environment, and let V π be the value function of the policy π on the estimated model M .

We design provable upper bounds, denoted by D π, M , on how much the error can compound and divert the expected value V π of the imaginary rollouts from their real value V π , in a neighborhood of some reference policy.

Such upper bounds capture the intrinsic difference between the estimated and real dynamical model with respect to the particular reward function under consideration.

The discrepancy bounds D π, M naturally leads to a lower bound for the true value function: DISPLAYFORM0 (1.1)Our algorithm iteratively collects batches of samples from the interactions with environments, builds the lower bound above, and then maximizes it over both the dynamical model M and the policy π.

We can use any RL algorithms to optimize the lower bounds, because it will be designed to only depend on the sample trajectories from a fixed reference policy (as opposed to requiring new interactions with the policy iterate.)We show that the performance of the policy is guaranteed to monotonically increase, assuming the optimization within each iteration succeeds (see Theorem 3.1.)

To the best of our knowledge, this is the first theoretical guarantee of monotone improvement for model-based deep RL.Readers may have realized that optimizing a robust lower bound is reminiscent of robust control and robust optimization.

The distinction is that we optimistically and iteratively maximize the RHS of (1.1) jointly over the model and the policy.

The iterative approach allows the algorithms to collect higher quality trajectory adaptively, and the optimism in model optimization encourages explorations of the parts of space that are not covered by the current discrepancy bounds.

To instantiate the meta-algorithm, we design a few valid discrepancy bounds in Section 4.

In Section 4.1, we recover the norm-based model loss by imposing the additional assumption of a Lipschitz value function.

The result suggests a norm is preferred compared to the square of the norm.

Indeed in Section 6.2, we show that experimentally learning with 2 loss significantly outperforms the mean-squared error loss ( 2 2 ).

In Section 4.2, we design a discrepancy bound that is invariant to the representation of the state space.

Here we measure the loss of the model by the difference between the value of the predicted next state and the value of the true next state.

Such a loss function is shown to be invariant to one-to-one transformation of the state space.

Thus we argue that the loss is an intrinsic measure for the model error without any information beyond observing the rewards.

We also refine our bounds in Section A by utilizing some mathematical tools of measuring the difference between policies in χ 2 -divergence (instead of KL divergence or TV distance).Our analysis also sheds light on the comparison between model-based RL and on-policy model-free RL algorithms such as policy gradient or TRPO BID17 .

The RHS of equation (1.1) is likely to be a good approximator of V π in a larger neighborhood than the linear approximation of V π used in policy gradient is (see Remark 4.5.)Finally, inspired by our framework and analysis, we design a variant of model-based RL algorithms Stochastic Lower Bounds Optimization (SLBO).

Experiments demonstrate that SLBO achieves state-of-the-art performance when only 1M samples are permitted on a range of continuous control benchmark tasks.

We denote the state space by S, the action space by A. A policy π(·|s) specifies the conditional distribution over the action space given a state s. A dynamical model M (·|s, a) specifies the conditional distribution of the next state given the current state s and action a. We will use M globally to denote the unknown true dynamical model.

Our target applications are problems with the continuous state and action space, although the results apply to discrete state or action space as well.

When the model is deterministic, M (·|s, a) is a dirac measure.

In this case, we use M (s, a) to denote the unique value of s and view M as a function from S × A to S. Let M denote a (parameterized) family of models that we are interested in, and Π denote a (parameterized) family of policies.

Unless otherwise stated, for random variable X, we will use p X to denote its density function.

Let S 0 be the random variable for the initial state.

Let S π,M t to denote the random variable of the states at steps t when we execute the policy π on the dynamic model M stating with S 0 .

Note that S π,M 0 = S 0 unless otherwise stated.

We will omit the subscript when it's clear from the context.

We use A t to denote the actions at step t similarly.

We often use τ to denote the random variable for the trajectory (S 0 , A 1 , . . .

, S t , A t , . . . ).

Let R(s, a) be the reward function at each step.

We assume R is known throughout the paper, although R can be also considered as part of the model if unknown.

Let γ be the discount factor.

Let V π,M be the value function on the model M and policy π defined as: DISPLAYFORM0 as the expected reward-to-go at Step 0 (averaged over the random initial states).

Our goal is to maximize the reward-to-go on the true dynamical model, that is, V π,M , over the policy π.

For simplicity, throughout the paper, we set κ = γ(1 − γ) −1 since it occurs frequently in our equations.

Every policy π induces a distribution of states visited by policy π: Definition 2.1.

For a policy π, define ρ π,M as the discounted distribution of the states visited by π on M .

Let ρ π be a shorthand for ρ π,M and we omit the superscript M throughout the paper.

Concretely,we have DISPLAYFORM1 As mentioned in the introduction, towards optimizing V π,M , 3 our plan is to build a lower bound for V π,M of the following type and optimize it iteratively: DISPLAYFORM2 where D( M , π) ∈ R ≥0 bounds from above the discrepancy between V π, M and V π,M .

Building such an optimizable discrepancy bound globally that holds for all M and π turns out to be rather difficult, if not impossible.

Instead, we shoot for establishing such a bound over the neighborhood of a reference policy π ref .

DISPLAYFORM3 Here d(·, ·) is a function that measures the closeness of two policies, which will be chosen later in alignment with the choice of D. We will mostly omit the subscript δ in D for simplicity in the rest of the paper.

We will require our discrepancy bound to vanish when M is an accurate model: DISPLAYFORM4 The third requirement for the discrepancy bound D is that it can be estimated and optimized in the sense that DISPLAYFORM5 where f is a known differentiable function.

We can estimate such discrepancy bounds for every π in the neighborhood of π ref by sampling empirical trajectories τ (1) , . . .

, τ (n) from executing policy π ref on the real environment M and compute the average of f ( M , π, τ (i) )'s.

We would have to insist that the expectation cannot be over the randomness of trajectories from π on M , because then we would have to re-sample trajectories for every possible π encountered.

For example, assuming the dynamical models are all deterministic, one of the valid discrepancy bounds (under some strong assumptions) that will prove in Section 4 is a multiple of the error of the prediction of M on the trajectories from π ref : DISPLAYFORM6 Suppose we can establish such an discrepancy bound D (and the distance function d) with properties (R1), (R2), and (R3), -which will be the main focus of Section 4 -, then we can devise the following meta-algorithm (Algorithm 1).

We iteratively optimize the lower bound over the policy π k+1 and the model M k+1 , subject to the constraint that the policy is not very far from the reference policy π k obtained in the previous iteration.

For simplicity, we only state the population version with the exact computation of D πref ( M , π), though empirically it is estimated by sampling trajectories.

For k = 0 to T : DISPLAYFORM7 We first remark that the discrepancy bound D π k (M, π) in the objective plays the role of learning the dynamical model by ensuring the model to fit to the sampled trajectories.

For example, using the discrepancy bound in the form of equation (3.2), we roughly recover the standard objective for model learning, with the caveat that we only have the norm instead of the square of the norm in MSE.

Such distinction turns out to be empirically important for better performance (see Section 6.2).Second, our algorithm can be viewed as an extension of the optimism-in-face-of-uncertainty (OFU) principle to non-linear parameterized setting: jointly optimizing M and π encourages the algorithm to choose the most optimistic model among those that can be used to accurately estimate the value function.

(See (Jaksch et al., 2010; BID4 Fruit et al., 2018; Lakshmanan et al., 2015; Pirotta et al., 2015; 2013) and references therein for the OFU principle in finite-state MDPs.)

The main novelty here is to optimize the lower bound directly, without explicitly building any confidence intervals, which turns out to be challenging in deep learning.

In other words, the uncertainty is measured straightforwardly by how the error would affect the estimation of the value function.

Thirdly, the maximization of V π,M , when M is fixed, can be solved by any model-free RL algorithms with M as the environment without querying any real samples.

Optimizing V π,M jointly over π, M can be also viewed as another RL problem with an extended actions space using the known "extended MDP technique".

See (Jaksch et al., 2010 , section 3.1) for details.

Our main theorem shows formally that the policy performance in the real environment is nondecreasing under the assumption that the real dynamics belongs to our parameterized family M.

Theorem 3.1.

Suppose that M ∈ M, that D and d satisfy equation (R1) and (R2), and the optimization problem in equation (3.3) is solvable at each iteration.

Then, Algorithm 1 produces a sequence of policies π 0 , . . .

, π T with monotonically increasing values: DISPLAYFORM0 Moreover, as k → ∞, the value V π k ,M converges to some Vπ ,M , whereπ is a local maximum of DISPLAYFORM1 The theorem above can also be extended to a finite sample complexity result with standard concentration inequalities.

We show in Theorem G.2 that we can obtain an approximate local maximum in O(1/ε) iterations with sample complexity (in the number of trajectories) that is polynomial in dimension and accuracy ε and is logarithmic in certain smoothness parameters.

Proof of Theorem 3.1.

Since D and d satisfy equation (R1), we have that DISPLAYFORM2 By the definition that π k+1 and M k+1 are the optimizers of equation (3. 3), we have that DISPLAYFORM3 Combing the two equations above we complete the proof of equation (3.5).For the second part of the theorem, by compactness, we have that a subsequence of π k converges to someπ.

By the monotonicity we have DISPLAYFORM4 For the sake of contradiction, we assumeπ is a not a local maximum, then in the neighborhood ofπ there exists π such that DISPLAYFORM5 (Here the last inequality uses equation (R1) with π t as π ref .)

The fact (π , M ) is a strictly better solution than (π t+1 , M t+1 ) contradicts the fact that (π t+1 , M t+1 ) is defined to be the optimal solution of (3.3) .

Thereforeπ is a local maximum and we complete the proof.

In this section, we design discrepancy bounds that can provably satisfy the requirements (R1), (R2), and (R3).

We design increasingly stronger discrepancy bounds from Section 4.1 to Section A.

In this subsection, we assume the dynamical model M is deterministic and we also learn with a deterministic model M .

Under assumptions defined below, we derive a discrepancy bound D of the form M (S, A) − M (S, A) averaged over the observed state-action pair (S, A) on the dynamical model M .

This suggests that the norm is a better metric than the mean-squared error for learning the model, which is empirically shown in Section 6.2.

Through the derivation, we will also introduce a telescoping lemma, which serves as the main building block towards other finer discrepancy bounds.

We make the (strong) assumption that the value function V π, M on the estimated dynamical model is L-Lipschitz w.r.t to some norm · in the sense that DISPLAYFORM0 In other words, nearby starting points should give reward-to-go under the same policy π.

We note that not every real environment M has this property, let alone the estimated dynamical models.

However, once the real dynamical model induces a Lipschitz value function, we may penalize the Lipschitz-ness of the value function of the estimated model during the training.

We start off with a lemma showing that the expected prediction error is an upper bound of the discrepancy between the real and imaginary values.

DISPLAYFORM1 However, in RHS in equation 4.2 cannot serve as a discrepancy bound because it does not satisfy the requirement (R3) -to optimize it over π we need to collect samples from ρ π for every iterate π -the state distribution of the policy π on the real model M .

The main proposition of this subsection stated next shows that for every π in the neighborhood of a reference policy π ref , we can replace the distribution ρ π be a fixed distribution ρ πref with incurring only a higher order approximation.

We use the expected KL divergence between two π and π ref to define the neighborhood: DISPLAYFORM2 Proposition 4.2.

In the same setting of Lemma 4.1, assume in addition that π is close to a reference policy DISPLAYFORM3 and that the states in S are uniformly bounded in the sense that s ≤ B, ∀s ∈ S. Then, DISPLAYFORM4 In a benign scenario, the second term in the RHS of equation (4.4) should be dominated by the first term when the neighborhood size δ is sufficiently small.

Moreover, the term B can also be replaced by max S,A M (S, A) − M (S, A) (see the proof that is deferred to Section C.).

The dependency on κ may not be tight for real-life instances, but we note that most analysis of similar nature loses the additional κ factor BID17 ; BID1 , and it's inevitable in the worst-case.

A telescoping lemma.

Towards proving Propositions 4.2 and deriving stronger discrepancy bound, we define the following quantity that captures the discrepancy between M and M on a single state-action pair (s, a).

DISPLAYFORM5 We give a telescoping lemma that decompose the discrepancy between V π,M and V π,M into the expected single-step discrepancy G. DISPLAYFORM6 The proof is reminiscent of the telescoping expansion in Kakade & Langford (2002 ) (c.f.

Schulman et al. (2015a ) for characterizing the value difference of two policies, but we apply it to deal with the discrepancy between models.

The detail is deferred to Section B. With the telescoping Lemma 4.3, Proposition 4.1 follows straightforwardly from Lipschitzness of the imaginary value function.

Proposition 4.2 follows from that ρ π and ρ πref are close.

We defer the proof to Appendix C.

The main limitation of the norm-based discrepancy bounds in previous subsection is that it depends on the state representation.

Let T be a one-to-one map from the state space S to some other space S , and for simplicity of this discussion let's assume a model M is deterministic.

Then if we represent every state s by its transformed representation T s, then the transformed model DISPLAYFORM0 is equivalent to the original set of the model, reward, and policy in terms of the performance (Lemma C.1).

Thus such transformation T is not identifiable from only observing the reward.

However, the norm in the state space is a notion that depends on the hidden choice of the transformation T .

Another limitation is that the loss for the model learning should also depend on the state itself instead of only on the difference M (S, A) − M (S, A).

It is possible that when S is at a critical position, the prediction error needs to be highly accurate so that the model M can be useful for planning.

On the other hand, at other states, the dynamical model is allowed to make bigger mistakes because they are not essential to the reward.

We propose the following discrepancy bound towards addressing the limitations above.

Recall the definition of ) ) which measures the difference between M (s, a)) and M (s, a) according to their imaginary rewards.

We construct a discrepancy bound using the absolute value of G. Let's define ε 1 and ε max as the average of |G π,M | and its maximum: DISPLAYFORM0 DISPLAYFORM1 .

We will show that the following discrepancy bound D The proof follows from the telescoping lemma (Lemma 4.3) and is deferred to Section C. We remark that the first term κε 1 can in principle be estimated and optimized approximately: the expectation be replaced by empirical samples from ρ πref , and G π,M is an analytical function of π and M when they are both deterministic, and therefore can be optimized by back-propagation through time (BPTT).

(When π and M and are stochastic with a re-parameterizable noise such as Gaussian distribution Kingma & Welling (2013) , we can also use back-propagation to estimate the gradient.)

The second term in equation (4.7) is difficult to optimize because it involves the maximum.

However, it can be in theory considered as a second-order term because δ can be chosen to be a fairly small number.

(In the refined bound in Section A, the dependency on δ is even milder.)

Remark 4.5.

Proposition 4.4 intuitively suggests a technical reason of why model-based approach can be more sample-efficient than policy gradient based algorithms such as TRPO or PPO BID17 .

The approximation error of V π, M in model-based approach decreases as the model error ε 1 , ε max decrease or the neighborhood size δ decreases, whereas the approximation error in policy gradient only linearly depends on the the neighborhood size BID17 .

In other words, model-based algorithms can trade model accuracy for a larger neighborhood size, and therefore the convergence can be faster (in terms of outer iterations.)

This is consistent with our empirical observation that the model can be accurate in a descent neighborhood of the current policy so that the constraint (3.4) can be empirically dropped.

We also refine our bonds in Section A, where the discrepancy bounds is proved to decay faster in δ.

DISPLAYFORM2

Model-based reinforcement learning is expected to require fewer samples than model-free algorithms (Deisenroth et al., 2013) and has been successfully applied to robotics in both simulation and in the real world BID14 Morimoto & Atkeson, 2003; BID15 ) using dynamical models ranging from Gaussian process BID14 Ko & Fox, 2009 ), time-varying linear models (Levine & Koltun, 2013; Lioutikov et al., 2014; Levine & Abbeel, 2014; BID34 , mixture of Gaussians (Khansari-Zadeh & Billard, 2011) , to neural networks (Hunt et al., 1992; Nagabandi et al., 2017; Kurutach et al., 2018; BID30 Sanchez-Gonzalez et al., 2018; Pascanu et al., 2017) .

In particular, the work of Kurutach et al. (2018) uses an ensemble of neural networks to learn the dynamical model, and significantly reduces the sample complexity compared to model-free approaches.

The work of BID7 makes further improvement by using a probabilistic model ensemble.

Clavera et al. BID8 extended this method with meta-policy optimization and improve the robustness to model error.

In contrast, we focus on theoretical understanding of model-based RL and the design of new algorithms, and our experiments use a single neural network to estimate the dynamical model.

Our discrepancy bound in Section 4 is closely related to the work (Farahmand et al., 2017) on the value-aware model loss.

Our approach differs from it in three details: a) we use the absolute value of the value difference instead of the squared difference; b) we use the imaginary value function from the estimated dynamical model to define the loss, which makes the loss purely a function of the estimated model and the policy; c) we show that the iterative algorithm, using the loss function as a building block, can converge to a local maximum, partly by cause of the particular choices made in a) and b).

BID3 also study the discrepancy bounds under Lipschitz condition of the MDP.Prior work explores a variety of ways of combining model-free and model-based ideas to achieve the best of the two methods BID26 BID25 Racanière et al., 2017; Mordatch et al., 2016; BID24 .

For example, estimated models (Levine & Koltun, 2013; Gu et al., 2016; Kalweit & Boedecker, 2017) On the control theory side, BID12 provide strong finite sample complexity bounds for solving linear quadratic regulator using model-based approach.

BID5 provide finitedata guarantees for the "coarse-ID control" pipeline, which is composed of a system identification step followed by a robust controller synthesis procedure.

Our method is inspired by the general idea of maximizing a low bound of the reward in BID11 .

By contrast, our work applies to non-linear dynamical systems.

Our algorithms also estimate the models iteratively based on trajectory samples from the learned policies.

Strong model-based and model-free sample complexity bounds have been achieved in the tabular case (finite state space).

We refer the readers to (Kakade et al., 2018; BID10 BID28 Kearns & Singh, 2002; Jaksch et al., 2010; BID2 and the reference therein.

Our work focus on continuous and high-dimensional state space (though the results also apply to tabular case).Another line of work of model-based reinforcement learning is to learn a dynamic model in a hidden representation space, which is especially necessary for pixel state spaces (Kakade et al., 2018; BID10 BID28 Kearns & Singh, 2002; Jaksch et al., 2010) .

BID23 shows the possibility to learn an abstract transition model to imitate expert policy.

Oh et al. We design with simplification of our framework a variant of model-based RL algorithms, Stochastic Lower Bound Optimization (SLBO).

First, we removed the constraints (3.4).

Second, we stop the gradient w.r.t M (but not π) from the occurrence of M in V π,M in equation (3.3) (and thus our practical implementation is not optimism-driven.)Extending the discrepancy bound in Section 4.1, we use a multi-step prediction loss for learning the models with 2 norm.

For a state s t and action sequence a t:t+h , we define the h-step predictionŝ t+h asŝ t = s t , and for h ≥ 0,ŝ t+h+1 = M φ (ŝ t+h , a t+h ), The H-step loss is then defined as DISPLAYFORM0 A similar loss is also used in Nagabandi et al. (2017) for validation.

We note that motivation by the theory in Section 4.1, we use 2 -norm instead of the square of 2 norm.

The loss function we attempt to optimize at iteration k is thus DISPLAYFORM1 where λ is a tunable parameter and sg denotes the stop gradient operation.

We note that the term V π θ ,sg( M φ ) depends on both the parameter θ and the parameter φ but there is no gradient passed through φ, whereas L (H) φ only depends on the φ.

We optimize equation (6.2) by alternatively maximizing V π θ ,sg( M φ ) and minimizing L for n inner iterations do optimize (6.2) with stochastic alternating updates 6:for n model iterations do for n policy iterations do

D ← { collect n trpo samples using M φ as dynamics } 10:optimize π θ by running TRPO on D Power of stochasticity and connection to standard MB RL: We identify the main advantage of our algorithms over standard model-based RL algorithms is that we alternate the updates of the model and the policy within an outer iteration.

By contrast, most of the existing model-based RL methods only optimize the models once (for a lot of steps) after collecting a batch of samples (see Algorithm 3 for an example).

The stochasticity introduced from the alternation with stochastic samples seems to dramatically reduce the overfitting (of the policy to the estimated dynamical model) in a way similar to that SGD regularizes ordinary supervised training.

8 Another way to view the algorithm is that the model obtained from line 7 of Algorithm 2 at different inner iteration serves as an ensemble of models.

We do believe that a cleaner and easier instantiation of our framework (with optimism) exists, and the current version, though performing very well, is not necessarily the best implementation.

Entropy regularization: An additional component we apply to SLBO is the commonly-adopted entropy regularization in policy gradient method BID32 Mnih et al., 2016) , which was found to significantly boost the performance in our experiments (ablation study in Appendix F.5).

Specifically, an additional entropy term is added to the objective function in TRPO.

We hypothesize that entropy bonus helps exploration, diversifies the collected data, and thus prevents overfitting.6 This is technically not a well-defined mathematical objective.

The sg operation means identity when the function is evaluated, whereas when computing the update, sg(M φ ) is considered fixed.7 In principle, to balance the number of steps, it suffices to take one of nmodel and npolicy to be 1.

However, empirically we found the optimal balance is achieved with larger nmodel and npolicy, possibly due to complicated interactions between the two optimization problem.8 Similar stochasticity can potentially be obtained by an extreme hyperparameter choice of the standard MB RL algorithm: in each outer iteration of Algorithm 3, we only sample a very small number of trajectories and take a few model updates and policy updates.

We argue our interpretation of stochastic optimization of the lower bound (6.2) is more natural in that it reveals the regularization from stochastic optimization.

We evaluate our algorithm SLBO (Algorithm 2) on five continuous control tasks from rllab (Duan et al., 2016) , including Swimmer, Half Cheetah, Humanoid, Ant, Walker.

All environments that we test have a maximum horizon of 500, which is longer than most of the existing model-based RL work (Nagabandi et al., 2017; Kurutach et al., 2018) .

(Environments with longer horizons are commonly harder to train.)

More details can be found in Appendix F.1.Baselines.

We compare our algorithm with 3 other algorithms including: (1) Soft Actor-Critic (SAC) (Haarnoja et al., 2018) , the state-of-the-art model-free off-policy algorithm in sample efficiency; (2) Trust-Region Policy Optimization (TRPO) BID17 , a policy-gradient based algorithm; and (3) Model-Based TRPO, a standard model-based algorithm described in Algorithm 3.

Details of these algorithms can be found in Appendix F.4.

The result is shown in FIG0 .

FIG0 shows superior convergence rate (in number of samples) than all the baseline algorithms while achieving better final performance with 1M samples.

Specifically, we mark model-free TRPO performance after 8 million steps by the dotted line in FIG0 and find out that our algorithm can achieve comparable or better final performance in one million steps.

For ablation study, we also add the performance of SLBO-MSE, which corresponds to running SLBO with squared 2 model loss instead of 2 .

SLBO-MSE performs significantly worse than SLBO on four environments, which is consistent with our derived model loss in Section 4.1.

We also study the performance of SLBO and baselines with 4 million training samples in F.5.

Ablation study of multi-step model training can be found in Appendix F.5.

We devise a novel algorithmic framework for designing and analyzing model-based RL algorithms with the guarantee to convergence monotonically to a local maximum of the reward.

Experimental results show that our proposed algorithm (SLBO) achieves new state-of-the-art performance on several mujoco benchmark tasks when one million or fewer samples are permitted.

A compelling (but obvious) empirical open question then given rise to is whether model-based RL can achieve near-optimal reward on other more complicated tasks or real-world robotic tasks with fewer samples.

We believe that understanding the trade-off between optimism and robustness is essential to design more sample-efficient algorithms.

Currently, we observed empirically that the optimism-driven part of our proposed meta-algorithm (optimizing V π, M over M ) may lead to instability in the optimization, and therefore don't in general help the performance.

It's left for future work to find practical implementation of the optimism-driven approach.

In our theory, we assume that the parameterized model class contains the true dynamical model.

Removing this assumption is also another interesting open question.

It would be also very interesting if the theoretical analysis can be applied other settings involving model-based approaches (e.g., model-based imitation learning).

The theoretical limitation of the discrepancy bound D G ( M , π) is that the second term involving ε max is not rigorously optimizable by stochastic samples.

In the worst case, there seem to exist situations where such infinity norm of G π, M is inevitable.

In this section we tighten the discrepancy bounds with a different closeness measure d, χ 2 -divergence, in the policy space, and the dependency on the ε max is smaller (though not entirely removed.)

We note that χ 2 -divergence has the same second order approximation as KL-divergence around the local neighborhood the reference policy and thus locally affects the optimization much.

We start by defining a re-weighted version β π of the distribution ρ π where examples in later step are slightly weighted up.

We can effectively sample from β π by importance sampling from ρ π Definition A.1.

For a policy π, define β π as the re-weighted version of discounted distribution of the states visited by π on M .

Recall that p S π t is the distribution of the state at step t, we define DISPLAYFORM0 Then we are ready to state our discrepancy bound.

Let DISPLAYFORM1 where We defer the proof to Section C so that we can group relevant proofs with similar tools together.

Some of these tools may be of independent interests and used for better analysis of model-free reinforcement learning algorithms such as TRPO Schulman et al. (2015a) , PPO Schulman et al. (2017) and CPO Achiam et al. (2017) .

DISPLAYFORM2

Proof of Lemma 4.3.

Let W j be the cumulative reward when we use dynamical model M for j steps and then M for the rest of the steps, that is, DISPLAYFORM0 .

Then, we decompose the target into a telescoping sum, DISPLAYFORM1 Now we re-write each of the summands W j+1 − W j .

Comparing the trajectory distribution in the definition of W j+1 and W j , we see that they only differ in the dynamical model applied in j-th step.

Concretely, W j and W j+1 can be rewritten DISPLAYFORM2 where R denotes the reward from the first j steps from policy π and model M .

Canceling the shared term in the two equations above, we get DISPLAYFORM3

Towards proving the second part of Proposition 4.4 regarding the invariance, we state the following lemma:Lemma C.1.

Suppose for simplicity the model and the policy are both deterministic.

For any oneto-one transformation from S to S , let DISPLAYFORM0 , and π T (s) π(T −1 s) be a set of transformed model, reward and policy.

Then we have that DISPLAYFORM1 where the value function V DISPLAYFORM2 Proof of Proposition A.2.

Let µ be the distribution of the initial state S 0 , and let P and P be the state-to-state transition kernel under policy π and π ref .

DISPLAYFORM3 Under these notations, we can re-write ρ πref =Ḡµ and ρ π =Ḡ µ. Moreover, we observe that β πref =ḠPḠµ. DISPLAYFORM4 (P , P ) 1/2 by the χ 2 divergence between P and P , measured with respect to distributionsḠµ = ρ πref andḠPḠµ = β πref .

By Lemma D.1, we have that the χ 2 -divergence between the states can be bounded by the χ 2 -divergence between the actions in the sense that: DISPLAYFORM5 Therefore we obtain that DISPLAYFORM6 .

By Lemma D.4, we can control the difference between ρ πref , f and ρ π , f by DISPLAYFORM7 Proof of Proposition 4.1 and 4.2 .

By definition of G and the Lipschitzness of V π, M , we have that DISPLAYFORM8 Then, by Lemma 4.3 and triangle inequality, we have that DISPLAYFORM9 Next we prove the main part of the proposition.

Thus we proved Proposition 4.1.

Note that for any distribution ρ and ρ and function f , we have ES∼ρ DISPLAYFORM10 Thus applying this inequality with f (S) = EA∼π(·|S) M (S, A) − M (S, A) , we obtain that DISPLAYFORM11 where the last inequality uses the inequalities (see Corollary E.7) that ρ π − ρ DISPLAYFORM12 Proof.

By definition, we have that Y |S = s, A = a has the same density as Y |S = s, A = a for any a and s.

Therefore by Theorem E.4 (setting X, X , Y, Y in Theorem E.4 by A|S = s, A |S = s, Y |S = s, Y |S = s respectively), we have DISPLAYFORM13 Taking expectation over the randomness of S we complete the proof.

In this subsection, we consider bounded the difference of the distributions induced by two markov process starting from the same initial distributions µ. Let P, P be two transition kernels.

Let DISPLAYFORM0 Define G andḠ similarly.

Therefore we have thatḠµ is the discounted distribution of states visited by the markov process starting from distribution µ. In other words, if µ is the distribution of S 0 , and P is the transition kernel induced by some policy π, then Gµ = ρ π .First of all, let ∆ = γ(P − P ) and we note that with simple algebraic manipulation, DISPLAYFORM1 Let f be some function.

We will mostly interested in the difference between E S∼Ḡµ [f ] and DISPLAYFORM2 , which can be rewritten as (Ḡ − G)µ, f .

We will bound this quantity from above by some divergence measure between P and P .We start off with a simple lemma that controls the form p − q, f by the χ 2 divergence between p and q. With this lemma we can reduce our problem of bounding (Ḡ − G)µ, f to characterizing the χ 2 divergence betweenḠ µ andḠµ.Lemma D.2.

Let p and q be probability distributions.

Then we have DISPLAYFORM3 Proof.

By Cauchy-Schwartz inequality, we have DISPLAYFORM4 The following Lemma is a refinement of the lemma above.

It deals with the distributions p and q with the special structure p = W P µ and q = W P µ.Lemma D.3.

Let W, P , P be transition kernels and µ be a distribution.

Then, DISPLAYFORM5 where χ 2 µ (P , P ) is a divergence between transitions defined in Definition E.3.Proof.

By Lemma D.2 with p = W P µ and q = W P µ, we conclude that DISPLAYFORM6 By Theorem E.4 and Theorem E.5 we have that χ 2 (W P µ, W P µ) ≤ χ 2 (P µ, P µ) ≤ χ 2 µ (P , P ), plugging this into the equation above we complete the proof.

Now we are ready to state the main result of this subsection.

Lemma D.4.

LetḠ,Ḡ , P , P, f as defined in the beginning of this section.

Let DISPLAYFORM7 By Holder inequality and the fact that Ḡ 1→1 = 1, Ḡ 1→1 = 1 and P 1→1 = 1, we have DISPLAYFORM8 Combining equation (D.3) and (D.5) we complete the proof of equation (D.2).Next we bound Ḡ PḠµ, f 2 1/2 in a more refined manner.

By equation (D.1), we have DISPLAYFORM9 By Lemma D.3 again, we have that DISPLAYFORM10 By Holder inequality and the fact that Ḡ 1→1 = 1, Ḡ 1→1 = 1 and P 1→1 = 1, we have DISPLAYFORM11 Then, combining equation (D.3), (D.4), (D.9), we have DISPLAYFORM12 The following Lemma is a stronger extension of Lemma D.4, which can be used to future improve Proposition A.2, and may be of other potential independent interests.

We state it for completeness.

Lemma D.5.

LetḠ,Ḡ , P , P, f as defined in the beginning of this section.

DISPLAYFORM13 (P , P ) 1/2 , then we have that for any K, DISPLAYFORM14 Proof.

We first use induction to prove that: DISPLAYFORM15 By the first equation of Lemma D.4, we got the case for K = 1.

Assuming we have proved the case for K, then applying DISPLAYFORM16 By Cauchy-Schwartz inequality, we obtain that DISPLAYFORM17 Plugging the equation above into equation (D.10), we provide the induction hypothesis for the case with K + 1.

distance between two distributions p and q is defined as DISPLAYFORM18 DISPLAYFORM19 For notational simplicity, suppose two random variables X and Y has distributions p X and p Y , we often write χ 2 (X, Y ) as a simplification for χ 2 (p X , p Y ). (2016) ).

The Kullback-Leibler (KL) divergence between two distributions p, q is bounded from above by the χ 2 distance:

Proof.

Since log is a concave function, by Jensen inequality we have DISPLAYFORM0 Definition E.3 (χ 2 distance between transitions).

Given two transition kernels P, P .

For any distribution µ, we define χ Proof.

With algebraic manipulation, we obtain, DISPLAYFORM1 It follows that DISPLAYFORM2 Replacing G in the RHS of the equation (E.2) by G = G + G ∆G, and doing this recursively gives DISPLAYFORM3 .

Let π and π be two policies and let ρ π be defined as in Definition 2.1.

Then, DISPLAYFORM4 Proof.

Let P and P be the state-state transition matrix under policy π and π and ∆ = γ(P − P ) By Claim E.6, we have that DISPLAYFORM5 We benchmark our algorithm on six tasks based on physics simulator Mujoco BID31 .

We use rllab's implementation (Duan et al., 2016) 12 to interact with Mujoco.

All the environments we use have a maximum horizon of 500 steps.

We remove all contact information from observation.

To compute reward from states, we put the velocity of center of mass into the states.

We use the same reward function as in rllab, except that all the coefficients C contact in front of the contact force s are set to 0 in our case.

We refer the readers to (Duan et al., 2016 ) Supp Material 1.2 for more details.

All actions are projected to the action space by clipping.

We normalize all observations by s = s−µ σ where µ, σ ∈ R dobservation are computed from all observations we collect from the real environment.

Note that µ, σ may change as we collect new data.

Our policy will always produce an action a in [−1, 1] daction and the action a , which is fed into the environment, is scaled linearly by a = 1−a 2 a min + 1+a 2 a max , where a min , a max are the min or max values allowed at each entry.

The dynamical model is represented by a feed-forward neural network with two hidden layers, each of which contains 500 hidden units.

The activation function at each layer is ReLU.

We use Adam to optimize the loss function with learning rate 10 −3 and L 2 regularization 10 −5 .

The network does not predict the next state directly; instead, it predicts the normalized difference of s t+1 − s t .

The normalization scheme and statistics are the same as those of observations: We maintain µ, σ from collected data in the real environment and may change them as we collect more, and the normalized difference is DISPLAYFORM0 The policy network is a feed-forward network with two hidden layers, each of which contains 32 hidden units.

The policy network uses tanh as activation function and outputs a Gaussian distribution N (µ(s), σ 2 ) where σ a state-independent trainable vector.

During our evaluation, we use H = 2 for multi-step model training and the batch size is given by 256 H = 128, i.e., we enforce the model to see 256 transitions at each batch.

We run our algorithm n outer = 100 iterations.

We collect n train = 10000 steps of real samples from the environment at the start of each iteration using current policy with Ornstein-Uhlunbeck noise (with parameter θ = 0.15, σ = 0.3) for better exploration.

At each iteration, we optimize dynamics model and policy alternatively for n inner = 20 times.

At each iteration, we optimize dynamics model for n model = 100 times and optimize policy for n policy = 40 times.

TRPO.

TRPO hyperparameters are listed at TAB1 , which are the same as OpenAI Baselines' implementation.

These hyperparameters are fixed for all experiments where TRPO is used, including ours, MB-TRPO and MF-TRPO.

We do not tune these hyperparameters.

We also normalize observations as our algorithm and OpenAI Baselines do.

We use a neural network as the value function to reduce variance, which has 2 hidden layers of units 64 and uses tanh as activation functions.

We use Generalized Advantage Estimator (GAE) BID18 to estimate advantages.

Both TRPO used in our algorithm and that in model-free algorithm share the same set of hyperparameters.

D ← D ∪ { collect n collect samples from real environment using π θ with noises }

for n model iterations do 6:optimize (6.1) over φ with sampled data from D by one step of Adam 7:for n policy iterations do 8:D ← { collect n trpo samples using M φ as dynamics } 9:optimize π θ by running TRPO on D SLBO.

We tune multi-step model training parameter H ∈ {1, 2, 4, 8}, entropy regularization coefficient λ ∈ {0, 0.001, 0.003, 0.005} and n policy ∈ {10, 20, 40} on Ant and find H = 2, λ = 0.005, n policy = 40 work best, then we fix them in all environments, though environment-specific hyperparameters may work better.

The other hyperparameters, including n inner , n model and network architecture, are never tuned.

We observe that at the first several iterations, the policy overfits to the learnt model so a reduction of n policy at the beginning can further speed up convergence but we omit this for simplicity.

The most important hyperparameters we found are n policy and the coefficient in front of the entropy regularizer λ.

It seems that once n model is large enough we don't see any significant changes.

We did have a held-out set for model prediction (with the same distribution as the training set) and found out the model doesn't overfit much.

As mentioned in F.3, we also found out normalizing the state helped a lot since the raw entries in the state have different magnitudes; if we don't normalize them, the loss will be dominated by the loss of some large entries.

Multi-step model training.

We compare multi-step model training with single-step model training and the results are shown on FIG8 .

Note that H = 1 means we use single-step model training.

We observe that small H (e.g., 2 or 4) can be beneficial, but larger H (e.g., 8) can hurt.

We hypothesize that smaller H can help the model learn the uncertainty in the input and address the error-propagation issue to some extent.

Pathak et al. (2018) uses an auto-regressive recurrent model to predict a multi-step loss on a trajectory, which is closely related to ours.

However, theirs differs from ours in the sense that they do not use the predicted output x t+1 as the input for the prediction of x t+2 , and so on and so forth.

DISPLAYFORM0 Figure 2: Ablation study on multi-step model training.

All the experiments are average over 10 random seeds.

The x-axis shows the total amount of real samples from the environment.

The y-axis shows the averaged return from execution of our learned policy.

The solid line is the mean of the total rewards from each seed.

The shaded area is one-standard deviation.

Entropy regularization.

FIG12 shows that entropy reguarization can improve both sample efficiency and final performance.

More entropy regularization leads to better sample efficiency and higher total rewards.

We observe that in the late iterations of training, entropy regularization may hurt the performance thus we stop using entropy regularization in the second half of training.

SLBO with 4M training steps.

FIG2 shows that SLBO is superior to SAC and MF-TRPO in Swimmer, Half Cheetah, Walker and Humanoid when 4 million samples or fewer samples are allowed.

For Ant environment , although SLBO with less than one million samples reaches the performance of MF-TRPO with 8 million samples, SAC's performance surpasses SLBO after 2 million steps of training.

Since model-free TRPO almost stops improving after 8M steps and our algorithms uses TRPO for optimizing the estimated environment, we don't expect SLBO can significantly outperform the reward of TRPO at 8M steps.

The result shows that SLBO is also satisfactory in terms of asymptotic convergence (compared to TRPO.)

It also indicates a better planner or policy optimizer instead of TRPO might be necessary to further improve the performance.

In this section, we extend Theorem 3.1 to a final sample complexity result.

For simplicity, let L πref,δ to denotes its empirical estimates.

Namely, we replace the expectation in equation (R3) by empirical samples τ(1) , . . .

, τ (n) .

In other words, we optimize DISPLAYFORM0 instead of equation (3.

3).Let p be the total number of parameters in the policy and model parameterization.

We assume that we have a discrepancy bound D πref (π, M ) satisfying (R3) with a function f that is bounded with [−B f , B f ] and that is L f -Lipschitz in the parameters of π and M .

That is, suppose π is parameterized by θ and M is parameterized by φ, then we require DISPLAYFORM1 2 ) for all τ , θ, θ , φ, φ .

We note that L f is likely to be exponential in dimension due to the recursive nature of the problem, but our bounds only depends on its logarithm.

We also restrict our attention to parameters in an Euclidean ball {θ : θ 2 ≤ B} and {φ : φ 2 ≤ B}. Our bounds will be logarithmic in B.We need the following definition of approximate local maximum since with sampling error we cannot hope to converge to the exact local maximum.

Definition G.1.

We say π is a (δ, ε)-local maximum of V π,M with respect to the constraint set Π and metric d, if for any π ∈ Π with d(π, π ) ≤ δ, we have V π,M ≥ V π ,M − ε.

We show a sample complexity bounds that scales linearly in p and logarithmically in L f , B and B f .

Theorem G.2.

Let ε > 0.

In the setting of Theorem 3.1, under the additional assumptions above, suppose we use n = O(B f p log(BL f /ε)/ε 2 ) trajectories to estimate the discrepancy bound in Algorithm 1.

Then, for any t, if π t is not a (δ, ε)-local maximum, then the total reward will increase in the next step: with high probability, DISPLAYFORM2 As a direct consequence, suppose the maximum possible total reward is B R and the initial total reward is 0, then for some T = O(B R /ε), we have that π T is a (δ, ε)-local maximum of the V π,M .Proof.

By Hoeffiding's inequality, we have for fix π and M , with probability 1 − n O(1) over the randomness of τ (1) , . . .

, τ (n) , DISPLAYFORM3 In more succinct notations, we have DISPLAYFORM4 B f log n n , and therefore DISPLAYFORM5 By a standard ε-cover + union bound argument, we can prove the uniform convergence: with high probability (at least 1 − n O(1) ) over the choice of τ (1) , . . .

, τ (n) , for all policy and model, for all policy π and dynamics M , DISPLAYFORM6 Suppose at iteration t, we are at policy π t which is not a (δ, ε)-local maximum of V π,M .

Then, there exists π such that d(π , π t ) ≤ δ and DISPLAYFORM7 Then, we have that Note that the total reward can only improve by ε/2 for at most O(B R /ε) steps.

Therefore, in the first O(B R /ε) iterations, we must have hit a solution that is a (δ, ε)-local maximum.

This completes the proof.

DISPLAYFORM8

<|TLDR|>

@highlight

We design model-based reinforcement learning algorithms with theoretical guarantees and achieve state-of-the-art results on Mujuco benchmark tasks when one million or fewer samples are permitted.

@highlight

The paper proposed a framework to design model-based RL algorithms based on OFU that achieves SOTA performance on MuJoCo tasks.
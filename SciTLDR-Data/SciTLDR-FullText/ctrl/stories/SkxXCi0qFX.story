Credit assignment in Meta-reinforcement learning (Meta-RL) is still poorly understood.

Existing methods either neglect credit assignment to pre-adaptation behavior or implement it naively.

This leads to poor sample-efficiency during meta-training as well as ineffective task identification strategies.

This paper provides a theoretical analysis of credit assignment in gradient-based Meta-RL.

Building on the gained insights we develop a novel meta-learning algorithm that overcomes both the issue of poor credit assignment and previous difficulties in estimating meta-policy gradients.

By controlling the statistical distance of both pre-adaptation and adapted policies during meta-policy search, the proposed algorithm endows efficient and stable meta-learning.

Our approach leads to superior pre-adaptation policy behavior and consistently outperforms previous Meta-RL algorithms in sample-efficiency, wall-clock time, and asymptotic performance.

A remarkable trait of human intelligence is the ability to adapt to new situations in the face of limited experience.

In contrast, our most successful artificial agents struggle in such scenarios.

While achieving impressive results, they suffer from high sample complexity in learning even a single task, fail to generalize to new situations, and require large amounts of additional data to successfully adapt to new environments.

Meta-learning addresses these shortcomings by learning how to learn.

Its objective is to learn an algorithm that allows the artificial agent to succeed in an unseen task when only limited experience is available, aiming to achieve the same fast adaptation that humans possess (Schmidhuber, 1987; Thrun & Pratt, 1998) .Despite recent progress, deep reinforcement learning (RL) still relies heavily on hand-crafted features and reward functions as well as engineered problem specific inductive bias.

Meta-RL aims to forego such reliance by acquiring inductive bias in a data-driven manner.

Recent work proves this approach to be promising, demonstrating that Meta-RL allows agents to obtain a diverse set of skills, attain better exploration strategies, and learn faster through meta-learned dynamics models or synthetic returns BID8 Xu et al., 2018; BID14 Saemundsson et al., 2018) .Meta-RL is a multi-stage process in which the agent, after a few sampled environment interactions, adapts its behavior to the given task.

Despite its wide utilization, little work has been done to promote theoretical understanding of this process, leaving Meta-RL grounded on unstable foundations.

Although the behavior prior to the adaptation step is instrumental for task identification, the interplay between pre-adaptation sampling and posterior performance of the policy remains poorly understood.

In fact, prior work in gradient-based Meta-RL has either entirely neglected credit assignment to the pre-update distribution BID9 or implemented such credit assignment in a naive way BID10 Stadie et al., 2018) .To our knowledge, we provide the first formal in-depth analysis of credit assignment w.r.t.

preadaptation sampling distribution in Meta-RL.

Based on our findings, we develop a novel Meta-RL algorithm.

First, we analyze two distinct methods for assigning credit to pre-adaptation behavior.of MAML, was first introduced by BID9 .

We refer to it as formulation I which can be expressed as maximizing the objective DISPLAYFORM0 In that U denotes the update function which depends on the task T , and performs one VPG step towards maximizing the performance of the policy in T .

For national brevity and conciseness we assume a single policy gradient adaptation step.

Nonetheless, all presented concepts can easily be extended to multiple adaptation steps.

Later work proposes a slightly different notion of gradient-based Meta-RL, also known as E-MAML, that attempts to circumvent issues with the meta-gradient estimation in MAML BID10 Stadie et al., 2018) : DISPLAYFORM1 R(τ ) with θ := U (θ, τ 1: DISPLAYFORM2 Formulation II views U as a deterministic function that depends on N sampled trajectories from a specific task.

In contrast to formulation I, the expectation over pre-update trajectories τ is applied outside of the update function.

Throughout this paper we refer to π θ as pre-update policy, and π θ as post-update policy.

This section analyzes the two gradient-based Meta-RL formulations introduced in Section 3.

The red arrows depict how credit assignment w.r.t the pre-update sampling distribution P T (τ |θ) is propagated.

Formulation I (left) propagates the credit assignment through the update step, thereby exploiting the full problem structure.

In contrast, formulation II (right) neglects the inherent structure, directly assigning credit from post-update return R to the pre-update policy π θ which leads to noisier, less effective credit assignment.

Both formulations optimize for the same objective, and are equivalent at the 0 th order.

However, because of the difference in their formulation and stochastic computation graph, their gradients and the resulting optimization step differs.

In the following, we shed light on how and where formulation II loses signal by analyzing the gradients of both formulations, which can be written as (see Appendix A for more details and derivations) ∇ θ J post (τ , τ ) simply corresponds to a policy gradient step on the post-update policy π θ w.r.t θ , followed by a linear transformation from post-to pre-update parameters.

It corresponds to increasing the likelihood of the trajectories τ that led to higher returns.

However, this term does not optimize for the pre-update sampling distribution, i.e., which trajectories τ led to better adaptation steps.

The credit assignment w.r.t.

the pre-updated sampling distribution is carried out by the second term.

In formulation II, ∇ θ J II pre can be viewed as standard reinforcement learning on π θ with R(τ ) as reward signal, treating the update function U as part of the unknown dynamics of the system.

This shifts the pre-update sampling distribution to better adaptation steps.

Formulation I takes the causal dependence of P T (τ |θ ) on P T (τ |θ) into account.

It does so by maximizing the inner product of pre-update and post-update policy gradients (see Eq. 4).

This steers the pre-update policy towards 1) larger post-updates returns 2) larger adaptation steps α∇ θ J inner , 3) better alignment of pre-and post-update policy gradients (Li et al., 2017; Nichol et al., 2018) .

When combined, these effects directly optimize for adaptation.

As a result, we expect the first meta-policy gradient formulation, J I , to yield superior learning properties.

In the previous section we show that the formulation introduced by BID9 results in superior meta-gradient updates, which should in principle lead to improved convergence properties.

However, obtaining correct and low variance estimates of the respective meta-gradients proves challenging.

As discussed by BID10 , and shown in Appendix B.3, the score function surrogate objective approach is ill suited for calculating higher order derivatives via automatic differentiation toolboxes.

This important fact was overlooked in the original RL-MAML implementation BID9 leading to incorrect meta-gradient estimates 1 .

As a result, ∇ θ J pre does not appear in the gradients of the meta-objective (i.e. ∇ θ J = ∇ θ J post ).

Hence, MAML does not perform any credit assignment to pre-adaptation behavior.

But, even when properly implemented, we show that the meta-gradients exhibit high variance.

Specifically, the estimation of the hessian of the RL-objective, which is inherent in the metagradients, requires special consideration.

In this section, we motivate and introduce the low variance curvature estimator (LVC): an improved estimator for the hessian of the RL-objective which promotes better meta-policy gradient updates.

As we show in Appendix A.1, we can write the gradient of the meta-learning objective as DISPLAYFORM0 H−1 t =t r(s t , a t )|s t , a t denotes the expected state-action value function under policy π θ at time t.

Computing the expectation of the RL-objective is in general intractable.

Typically, its gradients are computed with a Monte Carlo estimate based on the policy gradient theorem (Eq. 82).

In practical implementations, such an estimate is obtained by automatically differentiating a surrogate objective (Schulman et al., 2015b) .

However, this results in a highly biased hessian estimate which just computes H 2 , entirely dropping the terms H 1 and H 12 +H 12 .

In the notation of the previous section, it leads to neglecting the ∇ θ J pre term, ignoring the influence of the pre-update sampling distribution.

The issue can be overcome using the DiCE formulation, which allows to compute unbiased higherorder Monte Carlos estimates of arbitrary stochastic computation graphs BID10 .

The DiCE-RL objective can be rewritten as follows DISPLAYFORM1 DISPLAYFORM2 In that, ⊥ denotes the "stop gradient" operator, i.e., DISPLAYFORM3 The sequential dependence of π θ (a t |s t ) within the trajectory, manifesting itself through the product of importance weights in FORMULA4 , results in high variance estimates of the hessian DISPLAYFORM4 As noted by BID12 , H 12 is particularly difficult to estimate, since it involves three nested sums along the trajectory.

In section 7.2 we empirically show that the high variance estimates of the DiCE objective lead to noisy meta-policy gradients and poor learning performance.

To facilitate a sample efficient meta-learning, we introduce the low variance curvature (LVC) estimator: DISPLAYFORM5 By removing the sequential dependence of π θ (a t |s t ) within trajectories, the hessian estimate neglects the term H 12 + H 12 which leads to a variance reduction, but makes the estimate biased.

The choice of this objective function is motivated by findings in BID12 : under certain conditions the term H 12 + H 12 vanishes around local optima θ DISPLAYFORM6 Hence, the bias of the LVC estimator becomes negligible close to local optima.

The experiments in section 7.2 underpin the theoretical findings, showing that the low variance hessian estimates obtained through J LVC improve the sample-efficiency of meta-learning by a significant margin when compared to J DiCE .

We refer the interested reader to Appendix B for derivations and a more detailed discussion.6 PROMP: PROXIMAL META-POLICY SEARCH Building on the previous sections, we develop a novel meta-policy search method based on the low variance curvature objective which aims to solve the following optimization problem: DISPLAYFORM7 Prior work has optimized this objective using either vanilla policy gradient (VPG) or TRPO (Schulman et al., 2015a) .

TRPO holds the promise to be more data efficient and stable during the learning process when compared to VPG.

However, it requires computing the Fisher information matrix (FIM).

Estimating the FIM is particularly problematic in the meta-learning set up.

The meta-policy gradients already involve second order derivatives; as a result, the time complexity of the FIM estimate is cubic in the number of policy parameters.

Typically, the problem is circumvented using finite difference methods, which introduce further approximation errors.

The recently introduced PPO algorithm (Schulman et al., 2017) achieves comparable results to TRPO with the advantage of being a first order method.

PPO uses a surrogate clipping objective which allows it to safely take multiple gradient steps without re-sampling trajectories.

for step n = 0, ..., N − 1 do DISPLAYFORM8

if n = 0 then 6: DISPLAYFORM0 for all DISPLAYFORM1 Sample pre-update trajectories D i = {τ i } from T i using π θ 9:Compute adapted parameters DISPLAYFORM2 Sample post-update trajectories DISPLAYFORM3 11: DISPLAYFORM4 In case of Meta-RL, it does not suffice to just replace the post-update reward objective with J CLIP T. In order to safely perform multiple meta-gradient steps based on the same sampled data from a recent policy π θo , we also need to 1) account for changes in the pre-update action distribution π θ (a t |s t ), and 2) bound changes in the pre-update state visitation distribution (Kakade & Langford, 2002) .We propose Proximal Meta-Policy Search (ProMP) which incorporates both the benefits of proximal policy optimization and the low variance curvature objective (see Alg.

1.)

In order to comply with requirement 1), ProMP replaces the "stop gradient" importance weight DISPLAYFORM5 , which results in the following objective DISPLAYFORM6 An important feature of this objective is that its derivatives w.r.t θ evaluated at θ o are identical to those of the LVC objective, and it additionally accounts for changes in the pre-update action distribution.

To satisfy condition 2) we extend the clipped meta-objective with a KL-penalty term between π θ and π θo .

This KL-penalty term enforces a soft local "trust region" around π θo , preventing the shift in state visitation distribution to become large during optimization.

This enables us to take multiple meta-policy gradient steps without re-sampling.

Altogether, ProMP optimizes DISPLAYFORM7 ProMP consolidates the insights developed throughout the course of this paper, while at the same time making maximal use of recently developed policy gradients algorithms.

First, its meta-learning formulation exploits the full structural knowledge of gradient-based meta-learning.

Second, it incorporates a low variance estimate of the RL-objective hessian.

Third, ProMP controls the statistical distance of both pre-and post-adaptation policies, promoting efficient and stable meta-learning.

All in all, ProMP consistently outperforms previous gradient-based meta-RL algorithms in sample complexity, wall clock time, and asymptotic performance (see Section 7.1).

In order to empirically validate the theoretical arguments outlined above, this section provides a detailed experimental analysis that aims to answer the following questions: (i) How does ProMP perform against previous Meta-RL algorithms? (ii) How do the lower variance but biased LVC gradient estimates compare to the high variance, unbiased DiCE estimates? (iii) Do the different formulations result in different pre-update exploration properties? (iv) How do formulation I and formulation II differ in their meta-gradient estimates and convergence properties?To answer the posed questions, we evaluate our approach on six continuous control Meta-RL benchmark environments based on OpenAI Gym and the Mujoco simulator BID5 Todorov et al., 2012) .

A description of the experimental setup is found in Appendix D. In all experiments, the reported curves are averaged over at least three random seeds.

Returns are estimated based on sampled trajectories from the adapted post-update policies and averaged over sampled tasks.

The source code and the experiment data are available on our supplementary website.

We compare our method, ProMP, in sample complexity and asymptotic performance to the gradientbased meta-learning approaches MAML-TRPO BID9 ) and E-MAML-TRPO (see FIG2 ).

Note that MAML corresponds to the original implementation of RL-MAML by BID9 where no credit assignment to the pre-adaptation policy is happening (see Appendix B.3 for details).

Moreover, we provide a second study which focuses on the underlying meta-gradient estimator.

Specifically, we compare the LVC, DiCE, MAML and E-MAML estimators while optimizing meta-learning objective with vanilla policy gradient (VPG) ascent.

This can be viewed as an ablated version of the algorithms which tries to eliminate the influences of the outer optimizers on the learning performance (see Fig. 3 )

.These algorithms are benchmarked on six different locomotion tasks that require adaptation: the half-cheetah and walker must switch between running forward and backward, the high-dimensional agents ant and humanoid must learn to adapt to run in different directions in the 2D-plane, and the hopper and walker have to adapt to different configuration of their dynamics.

The results in FIG2 highlight the strength of ProMP in terms of sample efficiency and asymptotic performance.

In the meta-gradient estimator study in Fig. 3 , we demonstrate the positive effect of the LVC objective, as it consistently outperforms the other estimators.

In contrast, DiCE learns only slowly when compared to the other approaches.

As we have motivated mathematically and substantiate empirically in the following experiment, the poor performance of DiCE may be ascribed to the high variance of its meta-gradient estimates.

The fact that the results of MAML and E-MAML are comparable underpins the ineffectiveness of the naive pre-update credit assignment (i.e. formulation II), as discussed in section 4.Results for four additional environments are displayed in Appendix D along with hyperparameter settings, environment specifications and a wall-clock time comparison of the algorithms.

In Section 5 we discussed how the DiCE formulation yields unbiased but high variance estimates of the RL-objective hessian and served as motivation for the low variance curvature (LVC) estimator.

Here we investigate the meta-gradient variance of both estimators as well as its implication on the learning performance.

Specifically, we report the relative standard deviation of the metapolicy gradients as well as the average return throughout the learning process in three of the metaenvironments.

The results, depicted in Figure 4 , highlight the advantage of the low variance curvature estimate.

The trajectory level dependencies inherent in the DiCE estimator leads to a meta-gradient standard deviation that is on average 60% higher when compared to LVC.

As the learning curves indicate, the noisy gradients may be a driving factor for the poor performance of DiCE, impeding sample efficient meta-learning.

Meta-policy search based on the LVC estimator leads to substantially better sample-efficiency and asymptotic performance.

In case of HalfCheetahFwdBack, we observe some unstable learning behavior of LVC-VPG which is most likely caused by the bias of LVC in combination with the naive VPG optimizer.

However, the mechanisms in ProMP that ensure proximity w.r.t.

to the policys KL-divergence seem to counteract these instabilities during training, giving us a stable and efficient meta-learning algorithm.

Here we evaluate the effect of the different objectives on the learned pre-update sampling distribution.

We compare the low variance curvature (LVC) estimator with TRPO (LVC-TRPO) against MAML BID9 ) and E-MAML-TRPO (Stadie et al., 2018) in a 2D environment on which the exploration behavior can be visualized.

Each task of this environment corresponds to reaching a different corner location; however, the 2D agent only experiences reward when it is sufficiently close to the corner (translucent regions of FIG5 ).

Thus, to successfully identify the task, the agent must explore the different regions.

We perform three inner adaptation steps on each task, allowing the agent to fully change its behavior from exploration to exploitation.

functions.

Through its superior credit assignment, the LVC objective learns a pre-update policy that is able to identify the current task and respectively adapt its policy, successfully reaching the goal (dark green circle).The different exploration-exploitation strategies are displayed in FIG5 .

Since the MAML implementation does not assign credit to the pre-update sampling trajectory, it is unable to learn a sound exploration strategy for task identification and thus fails to accomplish the task.

On the other hand, E-MAML, which corresponds to formulation II, learns to explore in long but random paths: because it can only assign credit to batches of pre-update trajectories, there is no notion of which actions in particular facilitate good task adaptation.

As consequence the adapted policy slightly misses the task-specific target.

The LVC estimator, instead, learns a consistent pattern of exploration, visiting each of the four regions, which it harnesses to fully solve the task.

To shed more light on the differences of the gradients of formulation I and formulation II, we evaluate the meta-gradient updates and the corresponding convergence to the optimum of both formulations in a simple 1D environment.

In this environment, the agent starts in a random position in the real line and has to reach a goal located at the position 1 or -1.

In order to visualize the convergence, we parameterize the policy with only two parameters θ 0 and θ 1 .

We employ formulation I by optimizing the DiCE objective with VPG, and formulation II by optimizing its (E-MAML) objective with VPG.

Figure 6 depicts meta-gradient updates of the parameters θ i for both formulations.

Formulation I (red) exploits the internal structure of the adaptation update yielding faster and steadier convergence to the optimum.

Due to its inferior credit assignment, formulation II (green) produces noisier gradient estimates leading to worse convergence properties.

In this paper we propose a novel Meta-RL algorithm, proximal meta-policy search (ProMP), which fully optimizes for the pre-update sampling distribution leading to effective task identification.

Our method is the result of a theoretical analysis of gradient-based Meta-RL formulations, based on which we develop the low variance curvature (LVC) surrogate objective that produces low variance meta-policy gradient estimates.

Experimental results demonstrate that our approach surpasses previous meta-reinforcement learning approaches in a diverse set of continuous control tasks.

Finally, we underpin our theoretical contributions with illustrative examples which further justify the soundness and effectiveness of our method.

In this section we discuss two different gradient-based meta-learning formulations, derive their gradients and analyze the differences between them.

The first meta-learning formulation, known as MAML BID9 , views the inner update rule U (θ, T ) as a mapping from the pre-update parameter θ and the task T to an adapted policy parameter θ .

The update function can be viewed as stand-alone procedure that encapsulates sampling from the task-specific trajectory distribution P T (τ |π θ ) and updating the policy parameters.

Building on this concept, the meta-objective can be written as DISPLAYFORM0 The task-specific gradients follow as DISPLAYFORM1 DISPLAYFORM2 DISPLAYFORM3 In order to derive the gradients of the inner update ∇ θ θ = ∇ θ U (θ, T ) it is necessary to know the structure of U .

The main part of this paper assumes the inner update rule to be a policy gradient descent step DISPLAYFORM4 DISPLAYFORM5 Thereby the second term in (19) is the local curvature (hessian) of the inner adaptation objective function.

The correct hessian of the inner objective can be derived as follows: DISPLAYFORM6

The second meta-reinforcement learning formulation views the the inner update θ = U (θ, τ 1:N ) as a deterministic function of the pre-update policy parameters θ and N trajectories τ 1:N ∼ P T (τ 1:N |θ) sampled from the pre-update trajectory distribution.

This formulation was introduced in BID10 and further discussed with respect to its exploration properties in Stadie et al. (2018) .Viewing U as a function that adapts the policy parameters θ to a specific task T given policy rollouts in this task, the corresponding meta-learning objective can be written as DISPLAYFORM0 Since the first part of the gradient derivation is agnostic to the inner update rule U (θ, τ 1:N ), we only assume that the inner update function U is differentiable w.r.t.

θ.

First we rewrite the meta-objective J(θ) as expectation of task specific objectives J II T (θ) under the task distribution.

This allows us to express the meta-policy gradients as expectation of task-specific gradients: DISPLAYFORM1 The task specific gradients can be calculated as follows DISPLAYFORM2 As in A.1 the structure of U (θ, τ 1:N ) must be known in order to derive the gradient ∇ θ θ .

Since we assume the inner update to be vanilla policy gradient, the respective gradient follows as DISPLAYFORM3 The respective gradient of U (θ, τ 1:N ) follows as DISPLAYFORM4 DISPLAYFORM5

In the following we analyze the differences between the gradients derived for the two formulations.

To do so, we begin with ∇ θ J I T (θ) by inserting the gradient of the inner adaptation step (19) into FORMULA4 : DISPLAYFORM0 We can substitute the hessian of the inner objective by its derived expression from FORMULA26 and then rearrange the terms.

Also note that ∇ θ log P T (τ |θ) = ∇ θ log π θ (τ ) = H−1 t=1 log π θ (a t |s t ) where H is the MDP horizon.

DISPLAYFORM1 Next, we rearrange the gradient of J II into a similar form as ∇ θ J I T (θ).

For that, we start by inserting (28) for ∇ θ θ and replacing the expectation over pre-update trajectories τ 1:N by the expectation over a single trajectory τ .

DISPLAYFORM2 While the first part of the gradients match ( (32) and (34) ), the second part ( (33) and FORMULA35 ) differs.

Since the second gradient term can be viewed as responsible for shifting the pre-update sampling distribution P T (τ |θ) towards higher post-update returns, we refer to it as ∇ θ J pre (τ , τ ) .

To further analyze the difference between ∇ θ J I pre and ∇ θ J II pre we slightly rearrange (33) and put both gradient terms next to each other: DISPLAYFORM3 In the following we interpret and and compare of the derived gradient terms, aiming to provide intuition for the differences between the formulations:The first gradient term J post that matches in both formulations corresponds to a policy gradient step on the post-update policy π θ .

Since θ itself is a function of θ, the term I + αR(τ )∇ 2 θ log π θ (τ )) can be seen as linear transformation of the policy gradient update R(τ )∇ θ log π θ (τ ) from the post-update parameter θ into θ.

Although J post takes into account the functional relationship between θ and θ, it does not take into account the pre-update sampling distribution P T (τ |θ).This is where ∇ θ J pre comes into play: ∇ θ J I pre can be viewed as policy gradient update of the preupdate policy π θ w.r.t.

to the post-update return R(τ ).

Hence this gradient term aims to shift the pre-update sampling distribution so that higher post-update returns are achieved.

However, ∇ θ J II pre does not take into account the causal dependence of the post-update policy on the pre-update policy.

Thus a change in θ due to ∇ θ J II pre may counteract the change due to ∇ θ J II post .

In contrast, ∇ θ J I pre takes the dependence of the the post-update policy on the pre-update sampling distribution into account.

Instead of simply weighting the gradients of the pre-update policy ∇ θ log π θ (τ ) with R(τ ) as in ∇ θ J I post , ∇

θ J I post weights the gradients with inner product of the pre-update and post-update policy gradients.

This inner product can be written as DISPLAYFORM4 wherein δ denotes the angle between the the inner and outer pre-update and post-update policy gradients.

Hence, ∇ θ J I post steers the pre-update policy towards not only towards larger post-updates returns but also towards larger adaptation steps α∇ θ J inner , and better alignment of pre-and postupdate policy gradients.

This directly optimizes for maximal improvement / adaptation for the respective task.

See Li et al. (2017); Nichol et al. (2018) for a comparable analysis in case of domain generalization and supervised meta-learning.

Also note that (38) allows formulation I to perform credit assignment on the trajectory level whereas formulation II can only assign credit to entire batches of N pre-update trajectories τ 1:N .As a result, we expect the first meta-policy gradient formulation to learn faster and more stably since the respective gradients take the dependence of the pre-update returns on the pre-update sampling distribution into account while this causal link is neglected in the second formulation.

When employing formulation I for gradient-based meta-learning, we aim maximize the loss DISPLAYFORM0 by performing a form of gradient-descent on J(θ).

Note that we, from now on, assume J := J I and thus omit the superscript indicating the respective meta-learning formulation.

As shown in A.2 the gradient can be derived as DISPLAYFORM1 where DISPLAYFORM2 ] denotes hessian of the inner adaptation objective w.r.t.

θ.

This section concerns the question of how to properly estimate this hessian.

Since the expectation over the trajectory distribution P T (τ |θ) is in general intractable, the score function trick is typically used to used to produce a Monte Carlo estimate of the policy gradients.

Although the gradient estimate can be directly defined, when using a automatic-differentiation toolbox it is usually more convenient to use an objective function whose gradients correspond to the policy gradient estimate.

Due to the Policy Gradient Theorem (PGT) Sutton et al. (2000) such a "surrogate" objective can be written as: DISPLAYFORM0 While (41) and FORMULA41 are equivalent (Peters & Schaal, 2006) , the more popular formulation formulation (41) can be seen as forward looking credit assignment while (42) can be interpreted as backward looking credit assignment BID10 .

A generalized procedure for constructing "surrogate" objectives for arbitrary stochastic computation graphs can be found in Schulman et al. (2015a).

Estimating the the hessian of the reinforcement learning objective has been discussed in BID12 and BID4 with focus on second order policy gradient methods.

In the infinite horizon MDP case, BID4 derive a decomposition of the hessian.

In the following, we extend their finding to the finite horizon case.

Proof.

As derived in (24), the hessian of J inner (θ) follows as: DISPLAYFORM0 DISPLAYFORM1 The term in (50) is equal to H 2 .

We continue by showing that the remaining term in (51) is equivalent to H 1 + H 12 + H 12 .

For that, we split the inner double sum in (51) into three components: DISPLAYFORM2 DISPLAYFORM3 By changing the backward looking summation over outer products into a forward looking summation of rewards, (53) can be shown to be equal to H 1 : DISPLAYFORM4 DISPLAYFORM5 By simply exchanging the summation indices t and h in FORMULA45 it is straightforward to show that FORMULA45 is the transpose of (54).

Hence it is sufficient to show that (54) is equivalent to H 12 .

However, instead of following the direction of the previous proof we will now start with the definition of H 12 and derive the expression in (54).

DISPLAYFORM6 The gradient of Q π θ t can be expressed recursively: DISPLAYFORM7 By induction, it follows that DISPLAYFORM8 When inserting FORMULA51 into FORMULA48 and swapping the summation, we are able to show that H 12 is equivalent to (54).

DISPLAYFORM9 This concludes the proof that the hessian of the expected sum of rewards under policy π θ and an MDP with finite time horizon H can be decomposed into H 1 + H 2 + H 12 + H 12 .

As pointed out by BID10 Stadie et al. (2018) and BID10 , simply differentiating through the gradient of surrogate objective J PGT as done in the original MAML version BID9 leads to biased hessian estimates.

Specifically, when compared with the unbiased estimate, as derived in FORMULA26 and decomposed in Appendix B.2, both H 1 and H 12 + H 12 are missing.

Thus, ∇ θ J pre does not appear in the gradients of the meta-objective (i.e. ∇ θ J = ∇ θ J post ).

Only performing gradient descent with ∇ θ J post entirely neglects influences of the pre-update sampling distribution.

This issue was overseen in the RL-MAML implementation of BID9 .

As discussed in Stadie et al. (2018) this leads to poor performance in meta-learning problems that require exploration during the pre-update sampling.

Addressing the issue of incorrect higher-order derivatives of monte-carlo estimators, BID10 propose DICE which mainly builds upon an newly introduced MagicBox( ) operator.

This operator allows to formulate monte-carlo estimators with correct higher-order derivatives.

A DICE formulation of a policy gradient estimator reads as: DISPLAYFORM0 DISPLAYFORM1 In that, ⊥ denotes a "stop gradient" operator (i.e. DISPLAYFORM2 Note that → denotes a "evaluates to" and does not necessarily imply equality w.r.t.

to gradients.

Hence, J DICE (θ) evaluates to the sum of rewards at 0th order but produces the unbiased gradients ∇ n θ J DICE (θ) when differentiated n-times (see BID10 for proof).

To shed more light on the maverick DICE formulation, we rewrite (67) as follows: DISPLAYFORM3 Interpreting this novel formulation, the MagicBox operator θ ({a t ≤t }) can be understood as "dry" importance sampling weight.

At 0th order it evaluates to 1 and leaves the objective function unaffected, but when differentiated once it yields an estimator for the marginal rate of return due to a change in the policy-implied trajectory distribution.

In addition to the expected reward J(π) under policy π, we will use the state value function V π , the state-action value function Q π as well as the advantage function A π : DISPLAYFORM4 with a t ∼ π(a t |s t ) and s t+1 ∼ p(s t+1 |s t , a t ).The expected return under a policyπ can be expressed as the sum of the expected return of another policy π and the expected discounted advantage ofπ over π (see Schulman et al. (2015a) for proof).

DISPLAYFORM5 Let d π denote the discounted state visitation frequency: DISPLAYFORM6 We can use d π to express the expectation over trajectories τ ∼ p π (τ ) in terms of states and actions: DISPLAYFORM7 Local policy search aims to find a policy update π →π in the proximity of π so that J(π) is maximized.

Since J(π) is not affected by the policy update π →π, it is sufficient to maximize the expected advantage underπ.

However, the complex dependence of dπ(s) onπ makes it hard to directly maximize the objective in (102).

Using a local approximation of (102) where it is assumed that the state visitation frequencies d π and dπ are identical, the optimization can be phrased as DISPLAYFORM8 In the following we refer toJ(π) as surrogate objective.

It can be shown that the surrogate objectivẽ J matches J to first order when π =π (see Kakade & Langford (2002) ).

If π θ is a parametric and differentiable function with parameter vector θ, this means that for any θ o : DISPLAYFORM9 When π =π, an approximation error of the surrogate objectiveJ w.r.t.

to the true objective J is introduced.

BID0 derive a lower bound for the true expected return ofπ: DISPLAYFORM10 DISPLAYFORM11 Trust region policy optimization (TPRO) (Schulman et al., 2015a) attempts to approximate the bound in (105) by phrasing local policy search as a constrained optimization problem: DISPLAYFORM12 Thereby the KL-constraint δ induces a local trust region around the current policy π θo .

A practical implementation of TPRO uses a quadratic approximation of the KL-constraint which leads to the following update rule: DISPLAYFORM13 with g := ∇ θ E s∼dπ θo (s) DISPLAYFORM14 π θo (a|s) A π θo (s, a) being the gradient of the objective and F = ∇ 2 θD KL [π θo ||π θ ] the Fisher information matrix of the current policy π θo .

In order to avoid the cubic time complexity that arise when inverting F , the Conjugate Gradient (CG) algorithm is typically used to approximate the Hessian vector product F −1 g.

While TPRO is framed as constrained optimization, the theory discussed in Appendix C.1 suggest to optimize the lower bound.

Based on this insight, Schulman et al. (2017) propose adding a KL penalty to the objective and solve the following unconstrained optimization problem: DISPLAYFORM0 However, they also show that it is not sufficient to set a fixed penalty coefficient β and propose two alternative methods, known as Proximal Policy Optimization (PPO) that aim towards alleviating this issue:1) Adapting the KL coefficient β so that a desired target KL-divergenceD KL [π θo ||π θ ] between the policy before and after the parameter update is achieved 2) Clipping the likelihood ratio so that the optimization has no incentive to move the policy π θ too far away from the original policy π θo .

A corresponding optimization objective reads as: DISPLAYFORM1 Empirical results show that the latter approach leads to better learning performance (Schulman et al., 2017) .Since PPO objective keeps π θ in proximity of π θo , it allows to perform multiple gradient steps without re-sampling trajectories from the updated policy.

This property substantially improves the data-efficiency of PPO over vanilla policy gradient methods which need to re-estimate the gradients after each step.

The optimal hyperparameter for each algorithm was determined using parameter sweeps.

Table 1 contains the hyperparameter settings used for the different algorithms.

Any environment specific modifications are noted in the respective paragraph describing the environment.

PointEnv (used in the experiment in 7.3)• Trajectory Length : 100• Num Adapt Steps : 3 In this environment, each task corresponds to one corner of the area.

The point mass must reach the goal by applying directional forces.

The agent only experiences a reward when within a certain radius of the goal, and the magnitude of the reward is equal to the distance to the goal.

HalfCheetahFwdBack, AntFwdBack, WalkerFwdBack, HumanoidFwdBack• Trajectory Length : 100 (HalfCheetah, Ant); 200 (Humanoid, Walker)• Num Adapt Steps: 1The task is chosen between two directions -forward and backward.

Each agent must run along the goal direction as far as possible, with reward equal to average velocity minus control costs.

• Trajectory Length : 100 (Ant); 200 (Humanoid)• Num Adapt Steps: 1Each task corresponds to a random direction in the XY plane.

As above, each agent must learn to run in that direction as far as possible, with reward equal to average velocity minus control costs.

• Trajectory Length : 200• Num Adapt Steps:

2In this environment, each task is a location randomly chosen from a circle in the XY plane.

The goal is not given to the agent -it must learn to locate, approach, and stop at the target.

The agent receives a penalty equal to the distance from the goal.

• Trajectory Length : 200• Inner LR : 0.05• Num Adapt Steps: 1The agent must move forward as quickly as it can.

Each task is a different randomization of the simulation parameters, including friction, joint mass, and inertia.

The agent receives a reward equal to its velocity.

Published as a conference paper at ICLR 2019

In addition to the six environments displayed in 2, we ran experiments on the other four continuous control environments described above.

The results are displayed in 7.

In addition to the improved sample complexity and better asymptotic performance, another advantage of ProMP is its computation time.

FIG8 shows the average time spent per iteration throughout the learning process in the humanoid environment differences of ProMP, LVC-VPG, and MAML-TRPO.

Due to the expensive conjugate gradient steps used in TRPO, MAML takes far longer than either first order method.

Since ProMP takes multiple stochastic gradient descent steps per iteration, it leads to longer outer update times compared to VPG, but in both cases the update time is a fraction of the time spent sampling from the environment.

The difference in sampling time is due to the reset process: resetting the environment when the agent "dies" is an expensive operation.

ProMP acquires better performance quicker, and as a result the agent experiences longer trajectories and the environment is reset less often.

In our setup, instances of the environment are run in parallel and performing a reset blocks all environments.

<|TLDR|>

@highlight

A novel and theoretically grounded meta-reinforcement learning algorithm
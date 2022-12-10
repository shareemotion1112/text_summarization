Hierarchical Reinforcement Learning is a promising approach to long-horizon decision-making problems with sparse rewards.

Unfortunately, most methods still decouple the lower-level skill acquisition process and the training of a higher level that controls the skills in a new task.

Treating the skills as fixed can lead to significant sub-optimality in the transfer setting.

In this work, we propose a novel algorithm to discover a set of skills, and continuously adapt them along with the higher level even when training on a new task.

Our main contributions are two-fold.

First, we derive a new hierarchical policy gradient, as well as an unbiased latent-dependent baseline.

We introduce Hierarchical Proximal Policy Optimization (HiPPO), an on-policy method to efficiently train all levels of the hierarchy simultaneously.

Second, we propose a method of training time-abstractions that improves the robustness of the obtained skills to environment changes.

Code and results are available at sites.google.com/view/hippo-rl.

Reinforcement learning (RL) has made great progress in a variety of domains, from playing games such as Pong and Go BID5 BID20 to automating robotic locomotion BID11 BID20 Florensa et al., 2018b) , dexterous manipulation (Florensa et al., 2017b; BID1 , and perception BID7 Florensa et al., 2018a ).

Yet, most work in RL is still learning a new behavior from scratch when faced with a new problem.

This is particularly inefficient when dealing with tasks that are hard to solve due to sparse rewards or long horizons, or when solving many related tasks.

A promising technique to overcome this limitation is Hierarchical Reinforcement Learning (HRL) BID17 Florensa et al., 2017a) .

In this paradigm, policies have several modules of abstraction, so the reuse of a subset of the modules becomes easier.

The most common case consists of temporal abstraction BID9 Dayan & Hinton, 1993 ), where a higher-level policy (manager) takes actions at a lower frequency, and its actions condition the behavior of some lower level skills or sub-policies.

When transferring knowledge to a new task, most prior works fix the skills and train a new manager on top.

Despite having a clear benefit in kick-starting the learning in the new task, having fixed skills can considerably cap the final performance on the new task (Florensa et al., 2017a) .

Little work has been done on adapting pre-trained sub-policies to be optimal for a new task.

In this paper, we develop a new framework for adapting all levels of temporal hierarchies simultaneously.

First, we derive an efficient approximated hierarchical policy gradient.

Our key insight is that, under mild assumptions, the manager's decisions can be considered part of the observation from the perspective of the sub-policies.

This decouples the gradient with respect to the manager and the sub-policies parameters and provides theoretical justification for a technique used in other prior works (Frans et al., 2018) .

Second, we introduce an unbiased sub-policy specific baseline for our hierarchical policy gradient.

Our experiments reveal faster convergence, suggesting efficient gradient variance reduction.

Then we introduce a more stable way of using this gradient, Hierarchical Proximal Policy Optimization (HiPPO).

This helps us take more conservative steps in our policy space BID12 , necessary in hierarchies because of the interdependence of each layer.

Finally we also evaluate the benefit of varying the time-commitment to the sub-policies, and show it helps both in terms of final performance and zero-shot adaptation to similar tasks.

We define a discrete-time finite-horizon discounted Markov decision process (MDP) by a tuple M = (S, A, P, r, ρ 0 , γ, H), where S is a state set, A is an action set, P : S × A × S → R + is the transition probability distribution, γ ∈ [0, 1] is a discount factor, and H Figure 1 .

Temporal hierarchy studied in this paper.

A latent code zt is sampled from the manager policy π θ h (zt|st) every p time-steps, using the current observation s kp .

The actions at are sampled from the sub-policy π θ l (at|st, z kp ) conditioned on the same latent code from timestep t = kp to timestep (k + 1)p − 1 the horizon.

Our objective is to find a stochastic policy π θ that maximizes the expected discounted reward within the MDP, η( DISPLAYFORM0 We denote by τ = (s 0 , a 0 , ..., ) the entire state-action trajectory, where s 0 ∼ ρ 0 (s 0 ), a t ∼ π θ (a t |s t ), and s t+1 ∼ P(s t+1 |s t , a t ).

Prior works have been focused on learning a manager that combines provided sub-policies, but they do not further train the sub-policies when learning a new task.

However, preventing the skills from learning results in sub-optimal behavior in new tasks.

This effect is exacerbated when the skills were learned in a task agnostic way or in a different environment.

In this paper, we present a HRL method that learns all levels of abstraction in the hierarchical policy: the manager learns to make use of the low-level skills, while the skills are continuously adapted to attain maximum performance in the given task.

We derive a policy gradient update for hierarchical policies that monotonically improves the performance.

Furthermore, we demonstrate that our approach prevents sub-policy collapse behavior, when the manager ends up using just one skill, observed in previous approaches.

When using a hierarchical policy, the intermediate decision taken by the higher level is not directly applied in the environment.

This consideration makes it unclear how it should be incorporated into the Markovian framework of RL: should it be treated as an observed variable, like an action, or as a latent?In this section, we first prove that one framework is an approximation of the other under mild assumptions.

Then, we derive an unbiased baseline for the HRL setup that reduces its variance.

Thirdly, we introduce the notion of information bottleneck and trajectory compression, which proves critical for learning reusable skills.

Finally, with these findings, we present our method, Hierarchical Proximal Policy Optimization (HiPPO), an on-policy algorithm for hierarchical policies that monotonically improves the RL objective, allowing learning at all levels of the policy and preventing sub-policy collapse.

Policy gradient algorithms are based on the likelihood ratio trick BID21 to estimate the gradient of returns with respect to the policy parameters as DISPLAYFORM0 In the context of HRL, a hierarchical policy with a manager π θ h (z t |s t ) selects every p time-steps one of n sub-policies to execute.

These sub-policies, indexed by z ∈ [n], can be represented as a single conditional probability distribution over actions π θ l (a t |z t , s t ).

This allows us to also leverage skills learned with Stochastic Neural Networks (SNNs) (Florensa et al., 2017a) .

Under this framework, the probability of a trajectory τ = (s 0 , a 0 , s 1 , . . .

, s H ) can be written as DISPLAYFORM1 The mixture action distribution, which presents itself as an additional summation over skills, prevents the additive factorization when taking the logarithm, as in Eq. 1.

This can yield considerable numerical instabilities due to the product of the p sub-policy probabilities.

For instance, in the case where all the skills are distinguishable all the sub-policies probabilities but one will have small values, resulting in an exponentially small value.

In the following Lemma, we derive an approximation of the policy gradient, whose error tends to zero as the skills become more diverse, and draw insights on the interplay of the manager actions.

Lemma 1.

If the skills are sufficiently differentiated, then the latent variable can be treated as part of the observation to compute the gradient of the trajectory probability.

Let π θ h (z|s) and π θ l (a|s, z) be Lipschitz functions w.r.t.

their parameters, and assume that 0 < π θ l (a|s, z j ) < ∀j = kp, DISPLAYFORM2 Proof.

See Appendix.

Our assumption is that the skills are diverse.

Namely, for each action there is just one sub-policy that gives it high probability.

In this case, the latent variable can be treated as part of the observation to compute the gradient of the trajectory probability.

Many algorithms to extract lowerlevel skills are based on promoting diversity among the skills (Florensa et al., 2017a; Eysenbach et al., 2019) , so our assumption usually holds.

We further empirically analyze this assumption in the appendix.

The REINFORCE policy gradient estimate is known to have large variance.

A very common approach to mitigate this issue without biasing the estimate is to subtract a baseline from the returns BID8 .

We show how, under the assumptions of Lemma 1, we can formulate an unbiased latent dependent baseline for the approximate gradient (Eq. 4).Lemma 2.

For any functions b h : S → R and b l : S ×Z → R we have: DISPLAYFORM0 Proof.

See Appendix.

Now we apply Lemma 1 and Lemma 2 to Eq. 1.

By using the corresponding value functions as the function baseline, the return can be replaced by the Advantage function BID11 , and we obtain the following gradient expression: DISPLAYFORM1 This hierarchical policy gradient estimate has lower variance than without baselines, but using it for policy optimization through stochastic gradient descent still yields an unstable algorithm.

In the next section, we further improve the stability and sample efficiency of the policy optimization by incorporating techniques from Proximal Policy Optimization BID12 .

Using an appropriate step size in policy space is critical for stable policy learning.

We adopt the approach used by Proximal Policy Optimization (PPO) BID12 , which modifies the cost function in a way that prevents large changes to the policy while only requiring the computation of the likelihood.

Letting r h,kp (θ) = DISPLAYFORM0 , and using the super-index clip to denote the clipped objective version, we obtain the new surrogate objective: DISPLAYFORM1 We call this algorithm Hierarchical Proximal Policy Optimization (HiPPO).

Next, we introduce two critical additions: a switching of the time-commitment between skills, and an information bottleneck at the lower-level.

Both are detailed in the following subsections.

Most hierarchical methods either consider a fixed timecommitment to the lower level skills (Florensa et al., 2017a; Frans et al., 2018) , or implement the complex options framework BID9 BID2 .

In this work we propose an in-between, where the time-commitment to the skills is a random variable sampled from a fixed distribution Categorical(T min , T max ) just before the manager takes a decision.

This modification does not hinder final performance, and we show it improves zero-shot adaptation to a new task.

This approach to sampling rollouts is detailed given in Algorithm 1 in the appendix.

If we apply the above HiPPO algorithm in the general case, there is little incentive to either learn or maintain a diverse set of skills.

We claim this can be addressed via two simple additions:• Let z only take a finite number of values • Provide a masked observation to the skills DISPLAYFORM0 The masking function f restricts the information about the task, such that a single skill cannot perform the full task.

We use a hard-coded agent-space and problem-space split (Konidaris & Barto, 2007; Florensa et al., 2017a) that hides all task-related information and only allows the sub-policies to see proprioceptive information.

With this setup, all the missing information needed to perform the task must come from the sequence of latent codes passed to the skills.

We can interpret this as a lossy compression, whereby the manager encodes the relevant problem information into log n bits sufficient for the next p timesteps.

We design the experiments to answer the following questions: 1) How does HiPPO compare against a flat policy when learning from scratch?

2) Does it lead to more robust policies?

3) How well does it adapt already learned skills?

and 4) Does our skill diversity assumption hold in practice?

Figure 4 .

Benefit of adapting some given skills when the preferences of the environment are different from those of the environment where the skills were originally trained.

In this section, we study the benefit of using the HiPPO algorithm instead of standard PPO on a flat policy BID12 .

The results, shown in FIG0 , demonstrate that training from scratch with HiPPO leads faster learning and better performance than flat PPO.

Furthermore, the benefit of HiPPO does not just come from having temporally correlated exploration, as PPO with action repeat converges at a performance level well below our method.

Finally, FIG1 shows the effectiveness of using the presented baseline.

For this task, we take 6 pre-trained subpolicies encoded by a Stochastic Neural Network BID18 that were trained in a diversity-promoting environment (Florensa et al., 2017a) .

We fine-tune them with HiPPO on the Gather environment, but with an extra penalty on the velocity of the Center of Mass. This can be understood as a preference for cautious behavior.

This requires adjustment of the sub-policies, which were trained with a proxy reward encouraging them to move as far as possible (and hence quickly).

Fig. 4 shows the difference between fixing the sub-policies and only training a manager with PPO vs using HiPPO to simultaneously train a manager and fine-tune the skills.

The two initially learn at the same rate, but HiPPO's ability to adjust to the new dynamics allows it to reach a higher final performance.

In this paper, we examined how to effectively adapt hierarchical policies.

We began by deriving a hierarchical policy gradient and approximation of it.

We then proposed a new method, HiPPO, that can stably train multiple layers of a hierarchy.

The adaptation experiments suggested that we can optimize pretrained skills for downstream environments, and learn emergent skills without any unsupervised pre-training.

We also explored hierarchy from an information bottleneck point of view, demonstrating that HiPPO with randomized period can learn from scratch on sparsereward and long time horizon tasks, while outperforming non-hierarchical methods on zero-shot transfer.

There are many enticing avenues of future work.

For instance, replacing the manually designed bottleneck with a variational autoencoder with an information bottleneck could further improve HiPPO's performance and extend the gains seen here to other tasks.

Also, as HiPPO provides a policy architecture and gradient expression, we could explore using meta-learning on top of it in order to learn better skills that are more useful on a distribution of different tasks.

The key points in HRL are how the different levels of the hierarchy are defined, trained, and then re-used.

In this work, we are interested in approaches that allow us to build temporal abstractions by having a higher level taking decisions at a slower frequency than a lower-level.

There has been growing interest in HRL for the past few decades BID17 BID9 , but only recently has it been applied to high-dimensional continuous domains as we do in this work (Kulkarni et al., 2016; Daniel et al., 2016 ).To obtain the lower level policies, or skills, most methods exploit some additional assumptions, like access to demonstrations (Le et al., 2018; BID4 BID10 BID13 , policy sketches BID0 , or task decomposition into sub-tasks (Ghavamzadeh & Mahadevan, 2003; BID16 .

Other methods use a different reward for the lower level, often constraining it to be a "goal reacher" policy, where the signal from the higher level is the goal to reach BID6 BID3 BID20 .

These methods are very promising for state-reaching tasks, but might require access to goal-reaching reward systems not defined in the original MDP, and are more limited when training on tasks beyond state-reaching.

Our method does not require any additional supervision, and the obtained skills are not constrained to be goal-reaching.

When transferring skills to a new environment, most HRL methods keep them fixed and simply train a new higher-level on top (Hausman et al., 2018; Heess et al., 2016) .

Other work allows for building on previous skills by constantly supplementing the set of skills with new ones BID14 , but they require a hand-defined curriculum of tasks, and the previous skills are never fine-tuned.

Our algorithm allows for seamless adaptation of the skills, showing no trade-off between leveraging the power of the hierarchy and the final performance in a new task.

Other methods use invertible functions as skills (Haarnoja et al., 2018) , and therefore a fixed skill can be fully over-written when a new layer of hierarchy is added on top.

This kind of "fine-tuning" is promising, although they do not apply it to temporally extended skills as we are interested in here.

One of the most general frameworks to define temporally extended hierarchies is the options framework BID17 , and it has recently been applied to continuous state spaces BID2 .

One of the most delicate parts of this formulation is the termination policy, and it requires several regularizers to avoid skill collapse BID2 BID19 .

This modification of the objective may be difficult to tune and affects the final performance.

Instead of adding such penalties, we propose having skills of a random length, not controlled by the agent during training of the skills.

The benefit is two-fold: no termination policy to train, and more stable skills that transfer better.

Furthermore, these works only used discrete action MDPs.

We lift this assumption, and show good performance of our algorithm in complex locomotion tasks.

The closest work to ours in terms of final algorithm is the one proposed by Frans et al. (2018) .

Their method can be included in our framework, and hence benefits from our new theoretical insights.

We also introduce two modifications that are shown to be highly beneficial: the random time-commitment explained above, and the notion of an information bottleneck to obtain skills that generalize better.

Algorithm 1 Collect Rollout 1: Input: skills π θ l (a|s, z), manager π θ h (z|s), time-commitment bounds P min and P max , horizon H, and bottleneck function o = f (s) 2: Reset environment: s 0 ∼ ρ 0 , t = 0.

3: while t < H do

Sample time-commitment p ∼ Cat([P min , P max ])

Sample skill z t ∼ π θ h (·|s t ) 6: DISPLAYFORM0 Observe new state s t +1 and reward r t 9: end for 10:t ← t + p 11: end while 12: Output: DISPLAYFORM1

Input: skills π θ l (a|s, z), manager π θ h (z|s), horizon H, learning rate α while not done do for actor = 1, 2, ..., N do Obtain trajectory with Collect Rollout Estimate advantagesÂ(a t , o t , z t ) and To answer the posed questions, we evaluate our new algorithms on a variety of robotic navigation tasks.

Each task is a different robot trying to solve the Gather environment (Duan et al., 2016) , depicted in FIG2 , in which the agent must collect apples (green balls, +1 reward) while avoiding bombs (red balls, -1 reward).

This is a challenging hierarchical task with sparse rewards that requires agents to simultaneously learn perception, locomotion, and higher-level planning capabilities.

We use 2 different types of robots within this environment.

Snake is a 5-link robot with a 17-dimensional observation space and 4-dimensional action space; and Ant a quadrupedal robot with a 27-dimensional observation space and 8-dimensional action space.

Both can move and rotate in all directions, and Ant faces the added challenge of avoiding falling over irrecoverably.

DISPLAYFORM0

Gather Algorithm Initial Mass Dampening Inertia Friction Snake Flat PPO 2.72 3.16 (+16%) 2.75 (+1%) 2.11 (-22%) 2.75 (+1%) HiPPO, p = 10 4.38 3.28 (-25%) 3.27 (-25%) 3.03 (-31%) 3.27 (-25%) HiPPO random p 5.11 4.09 (-20%) 4.03 (-21%) 3.21 (-37%) 4.03 (-21%) Ant Flat PPO 2.25 2.53 (+12%) 2.13 (-5%) 2.36 (+5%) 1.96 (-13%) HiPPO, p = 10 3.84 3.31 (-14%) 3.37 (-12%) 2.88 (-25%) 3.07 (-20%) HiPPO random p 3.22 3.37 (+5%) 2.57 (-20%) 3.36 (+4%) 2.84 (-12%) Table 1 .

Zero-shot transfer performance of flat PPO, HiPPO, and HiPPO with randomized period.

The performance in the initial environment is shown, as well as the average performance over 25 rollouts in each new modified environment.

We try several different modifications to the base Snake Gather and Ant Gather environments.

One at a time, we change the body mass, dampening of the joints, body inertia, and friction characteristics of both robots.

The results, presented in Table 1 , show that HiPPO with randomized period Categorical([T min , T max ]) not only learns faster initially on the original task, but it is also able to better handle these dynamics changes.

In terms of the percent change in policy performance between the training environment and test environment, it is able to outperform HiPPO with fixed period on 6 out of 8 related tasks without even taking any gradient steps.

Our hypothesis is that the randomized period teaches the policy to adapt to wide variety of scenarios, while its information bottleneck is able to keep separate its representations for planning and locomotion, so changes in dynamics aren't able to simultaneously affect both.

Gather Algorithm Cosine Similarity max z =z kp π θ l (a t |o t , z ) Table 2 .

Empirical evaluation of Lemma 1.

On the right column we evaluate the quality of our assumption by computing what is the average largest probability of a certain action under other skills.

On the left column we report cosine similarity between our approximate gradient and the gradient computed using Eq. 2 without approximation.

In Lemma 1, we assumed that the sub-policies present ought to be diverse.

This allowed us to derive a more efficient and numerically stable gradient.

In this section, we empirically test the validity of our assumption, as well as the quality of our approximation.

For this we run, on Snake Gather and Ant Gather, the HiPPO algorithm both from scratch and on some pretrained skills as described in the previous section.

In Table 2 , we report the average maximum probability under other sub-policies, corresponding to from the assumption.

We observe that in all settings this is on the order of magnitude of 0.1.

Therefore, under the p = 10 that we use in our experiments, the term we neglect has a factor p−1 = 10 −10 .

It is not surprising then that the average cosine similarity between the full gradient and the approximated one is almost 1, as also reported in Table 2 .

We only ran two random seeds of these experiments, as the results seemed pretty consistent, and they are more computationally challenging to run.

For all experiments, both PPO and HiPPO used learning rate 3 × 10 −3 , clipping parameter = 0.1, 10 gradient updates per iteration, a batch size of 100,000, and discount γ = 0.999.

HiPPO used n = 6 sub-policies.

Ant Gather has a horizon of 5000, while Snake Gather has a horizon of 8000 due to its larger size.

All runs used three random seeds.

HiPPO uses a manager network with 2 hidden layers of 32 units, and a skill network with 2 hidden layers of 64 units.

In order to have roughly the same number of parameters for each algorithm, flat PPO uses a network with 2 hidden layers with 256 and 64 units respectively.

For HiPPO with randomized period, we resample p ∼ Uniform{5, 15} every time the manager network outputs a latent, and provide the number of timesteps until the next latent selection as an input into both the manager and skill networks.

The single baselines and skill-dependent baselines used a MLP with 2 hidden layers of 32 units to fit the value function.

The skill-dependent baseline receives, in addition to the full observation, the active latent code and the time remaining until the next skill sampling.

Lemma 1.

If the skills are sufficiently differentiated, then the latent variable can be treated as part of the observation to compute the gradient of the trajectory probability.

Concretely, if π θ h (z|s) and π θ l (a|s, z) are Lipschitz in their parameters, and 0 < π θ l (a t |s t , z j ) < ∀j = kp, then DISPLAYFORM0 Proof.

From the point of view of the MDP, a trajectory is a sequence τ = (s 0 , a 0 , s 1 , a 1 , . . .

, a H−1 , s H ).

Let's assume we use the hierarchical policy introduced above, with a higher-level policy modeled as a parameterized discrete distribution with n possible outcomes π θ h (z|s) = Categorical θ h (n).

We can expand P (τ ) into the product of policy and environment dynamics terms, with z j denoting the jth possible value out of the n choices, DISPLAYFORM1 Taking the gradient of log P (τ ) with respect to the policy parameters θ = [θ h , θ l ], the dynamics terms disappear, leaving: DISPLAYFORM2 The sum over possible values of z prevents the logarithm from splitting the product over the p-step sub-trajectories.

This term is problematic, as this product quickly approaches 0 as p increases, and suffers from considerable numerical instabilities.

Instead, we want to approximate this sum of products by a single one of the terms, which can then be decomposed into a sum of logs.

For this we study each of the terms in the sum: the gradient of a sub-trajectory probability under a specific latent ∇ θ π θ h (z j |s kp ) (k+1)p−1 t=kp π θ l (a t |s t , z j ) .

Now we can use the assumption that the skills are easy to distinguish, 0 < π θ l (a t |s t , z j ) < ∀j = kp.

Therefore, the probability of the sub-trajectory under a latent different than the one that was originally sampled z j = z kp , is upper bounded by p .

Taking the gradient, applying the product rule, and the Lipschitz continuity of the policies, we obtain that for all z j = z kp , Thus, we can across the board replace the summation over latents by the single term corresponding to the latent that was sampled at that time.

DISPLAYFORM3 ∇ θ log P (τ ) = Interestingly, this is exactly ∇ θ P (s 0 , z 0 , a 0 , s 1 , . . . ).

In other words, it's the gradient of the probability of that trajectory, where the trajectory now includes the variables z as if they were observed.

Lemma 2.

For any functions b h : S → R and b l : S × Z → R we have: Then, we can write out the definition of the expectation and undo the gradient-log trick to prove that the baseline is unbiased.

∇ θ log π s,θ (a t |s t , z kp )b(s t , z kp )] = 0 DISPLAYFORM4 We'll follow the same strategy to prove the second equality: apply the same law of iterated expectations trick, express the expectation as an integral, and undo the gradient-log trick.

@highlight

We propose HiPPO, a stable Hierarchical Reinforcement Learning algorithm that can train several levels of the hierarchy simultaneously, giving good performance both in skill discovery and adaptation.
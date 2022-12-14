We propose Support-guided Adversarial Imitation Learning (SAIL), a generic imitation learning framework that unifies support estimation of the expert policy with the family of Adversarial Imitation Learning (AIL) algorithms.

SAIL addresses two important challenges of AIL, including the implicit reward bias and potential training instability.

We also show that SAIL is at least as efficient as standard AIL.

In an extensive evaluation, we demonstrate that the proposed method effectively handles the reward bias and achieves better performance and training stability than other baseline methods on a wide range of benchmark control tasks.

The class of Adversarial Imitation Learning (AIL) algorithms learns robust policies that imitate an expert's actions from a small number of expert trajectories, without further access to the expert or environment signals.

AIL iterates between refining a reward via adversarial training, and reinforcement learning (RL) with the learned adversarial reward.

For instance, Generative Adversarial Imitation Learning (GAIL) (Ho & Ermon, 2016) shows the equivalence between some settings of inverse reinforcement learning and Generative Adversarial Networks (GANs) (Goodfellow et al., 2014) , and recasts imitation learning as distribution matching between the expert and the RL agent.

Similarly, Adversarial Inverse Reinforcement Learning (AIRL) (Fu et al., 2017) modifies the GAIL discriminator to learn a reward function robust to changes in dynamics or environment properties.

AIL mitigates the issue of distributional drift from behavioral cloning (Ross et al., 2011) , a classical imitation learning algorithm, and demonstrates good performance with only a small number of expert demonstrations.

However, AIL has several important challenges, including implicit reward bias (Kostrikov et al., 2019) , potential training instability (Salimans et al., 2016; Brock et al., 2018) , and potential sample inefficiency with respect to environment interaction (Sasaki et al., 2019) .

In this paper, we propose a principled approach towards addressing these issues.

Wang et al. (2019) demonstrated that imitation learning is also feasible by constructing a fixed reward function via estimating the support of the expert policy.

Since support estimation only requires expert demonstrations, the method sidesteps the training instability associated with adversarial training.

However, we show in Section 4.2 that the reward learned via support estimation deteriorates when expert data is sparse, and leads to poor policy performances.

Support estimation and adversarial reward represent two different yet complementary RL signals for imitation learning, both learnable from expert demonstrations.

We unify both signals into Supportguided Adversarial Imitation Learning (SAIL), a generic imitation learning framework.

SAIL leverages the adversarial reward to guide policy exploration and constrains the policy search to the estimated support of the expert policy.

It is compatible with existing AIL algorithms, such as GAIL and AIRL.

We also show that SAIL is at least as efficient as standard AIL.

In an extensive evaluation, we demonstrate that SAIL mitigates the implicit reward bias and achieves better performance and training stability against baseline methods over a series of benchmark control tasks.

We briefly review the Markov Decision Process (MDP), the context of our imitation learning task, followed by related works on imitation learning.

We consider an infinite-horizon discounted MDP (S, A, P, r, p 0 , ??), where S is the set of states, A the set of actions, P : S ?? A ?? S ??? [0, 1] the transition probability, r : S ?? A ??? R the reward function, p 0 : S ??? [0, 1] the distribution over initial states, and ?? ??? (0, 1) the discount factor.

Let ?? be a stochastic policy ?? :

, and s t+1 ??? P (??|s t , a t ) for t ??? 0.

We denote ?? E the expert policy.

Behavioral Cloning (BC) learns a policy ?? :

S ??? A directly from expert trajectories via supervised learning.

BC is simple to implement, and effective when expert data is abundant.

However, BC is prone to distributional drift: the state distribution of expert demonstrations deviates from that of the agent policy, due to accumulation of small mistakes during policy execution.

Distributional drift may lead to catastrophic errors (Ross et al., 2011) .

While several methods address the issue (Ross & Bagnell, 2010; Sun et al., 2017) , they often assume further access to the expert during training.

Inverse Reinforcement Learning (IRL) first estimates a reward from expert demonstrations, followed by RL using the estimated reward (Ng & Russell, 2000; Abbeel & Ng, 2004) .

Building upon a maximum entropy formulation of IRL (Ziebart et al., 2008) , Finn et al. (2016) and Fu et al. (2017) explore adversarial IRL and its connection to Generative Adversarial Imitation Learning (Ho & Ermon, 2016) .

Imitation Learning via Distribution Matching Generative Adversarial Imitation Learning (GAIL) (Ho & Ermon, 2016) frames imitation learning as distribution matching between the expert and the RL agent.

The authors show the connection between IRL and GANs.

Specifically, GAIL imitates the expert by formulating a minimax game: min

where the expectations E ?? and E ?? E denote the joint distributions over state-actions of the RL agent and the expert, respectively.

GAIL is able to achieve expert performance with a small number of expert trajectories on various benchmark tasks.

However, GAIL is relatively sample inefficient with respect to environment interaction, and inherits issues associated with adversarial learning, such as vanishing gradients, training instability and overfitting to expert demonstrations (Arjovsky & Bottou, 2017; Brock et al., 2018) .

Recent works have improved the sample efficiency and stability of GAIL.

For instance, Generative Moment Matching Imitation Learning (Kim & Park, 2018) replaces the adversarial reward with a non-parametric maximum mean discrepancy estimator to sidestep adversarial learning.

Baram et al. (2017) improve sample efficiency with a model-based RL algorithm.

Kostrikov et al. (2019) and Sasaki et al. (2019) demonstrate significant gain in sample efficiency with off-policy RL algorithms.

In addition, Generative Predecessor Models for Imitation Learning (Schroecker et al., 2019) imitates the expert policy using generative models to reason about alternative histories of demonstrated states.

Our proposed method is closely related to the broad family of AIL algorithms including GAIL and adversarial IRL.

It is also complementary to many techniques for improving the algorithmic efficiency and stability, as discussed above.

In particular, we focus on improving the quality of the learned reward by constraining adversarial reward to the estimated support of the expert policy.

Imitation Learning via Support Estimation Alternative to AIL, Wang et al. (2019) demonstrate the feasibility of using a fixed RL reward via estimating the support of the expert policy from expert demonstrations.

Connecting kernel-based support estimation (De Vito et al., 2014) to Random Network Distillation (Burda et al., 2018) , the authors propose Random Expert Distillation (RED) to learn a reward function based on support estimation.

Specifically, RED learns the reward parameter?? by minimizing:

where f ?? : S ?? A ??? R K projects (s, a) from expert demonstrations to some embedding of size K, with randomly initialized ??.

The reward is then defined as:

where ?? is a hyperparameter.

As optimizing Eq. (2) only requires expert data, RED sidesteps adversarial learning, and casts imitation learning as a standard RL task using the learned reward.

While RED works well given sufficient expert data, we show in the experiments that its performance suffers in the more challenging setting of sparse expert data.

Formally, we consider the task of learning a reward functionr(s, a) from a finite set of trajectories

, sampled from the expert policy ?? E within a MDP.

Each trajectory is a sequence of stateaction tuples in the form of ?? i = {s 1 , a 1 , s 2 , a 2 , ..., s T , a T }.

Assuming that the expert trajectories are consistent with some latent reward function r * (s, a), we aim to learn a policy that achieves good performance with respect to r * (s, a) by applying RL on the learned reward functionr(s, a).

In this section, we first discuss the advantages and shortcomings of AIL to motivate our method.

We then introduce Support-guided Adversarial Learning (SAIL), and present a theoretical analysis that compares SAIL with the existing methods, specifically GAIL.

A clear advantage of AIL resides in its low sample complexity with respect to expert data.

For instance, GAIL requires as little as 200 state-action tuples from the expert to achieve imitation.

The reason is that the adversarial reward may be interpreted as an effective exploration mechanism for the RL agent.

To see this, consider the learned reward function under the optimality assumption.

With the optimal discriminator to Eq.

Eq. (4) shows that the adversarial reward only depends on the ratio ??(s, a) =

p??(s,a) .

Intuitively, r gail incentivizes the RL agent towards under-visited state-actions, where ??(s, a) > 1, and away from over-visited state-actions, where ??(s, a) < 1.

When ?? E and ?? match exactly, r gail converges to an indicator function for the support of ?? E , since ??(s, a) = 1 ??? (s, a) ??? supp(?? E ) (Goodfellow et al., 2014) .

In practice, the adversarial reward is unlikely to converge, as p ?? E is estimated from a finite set of expert demonstrations.

Instead, the adversarial reward continuously drives the agent to explore by evolving the reward landscape.

However, AIL also presents several challenges.

Kostrikov et al. (2019) demonstrated that the reward ??? log D(s, a) suffers from an implicit survival bias, as the non-negative reward may lead to suboptimal behaviors in goal-oriented tasks where the agent learns to move around the goal to accumulate rewards, instead of completing the tasks.

While the authors resolve the issue by introducing absorbing states, the solution assumes extra RL signals from the environment, including access to the time limit of an environment to detect early termination of training episodes.

In Section 4.1, we empirically demonstrate the survival bias on Lunar Lander, a common RL benchmark, by showing that agents trained with GAIL often hover over the goal location 1 .

We also show that our proposed method is able to robustly imitate the expert.

Another challenge with AIL is potential training instability.

Wang et al. (2019) demonstrated empirically that the adversarial reward could be unreliable in regions where the expert data is sparse, causing the agent to diverge from the intended behavior.

When the agent policy is substantially different from the expert policy, the discriminator could differentiate them with high confidence, resulting in very low rewards and significant slow down in training, similar to the vanishing gradient problem in GAN training (Arjovsky & Bottou, 2017) .

We propose a novel reward function by combining the standard adversarial reward r gail with the corresponding support guidance r red .

SAIL is designed to leverage the exploration mechanism offered by the adversarial reward, and to constrain the agent to the estimated support of the expert policy.

Despite being a simple modification, support guidance provides strong reward shaping to address the challenges discussed in the previous

, ?? function models, initial policy ?? ??0 , initial discriminator parameters w 0 , learning rate l D .

2: r red = RED(??, ?? E ) 3: for i = 0, 1, . . .

sample a trajectory ?? i ??? ?? 5:

?? ??i+1 = TRPO(r red ?? r gail , ?? ??i ).

Sample ?? ??? ?? 10:?? =MINIMIZE(f??, f ?? , ?? ) 11:

section.

As both support guidance and adversarial reward are learnable from expert demonstrations, our method requires no further assumptions that standard AIL.

SAIL addresses the survival bias in goal-oriented tasks by encouraging the agent to stop at the goal and complete the task.

In particular, r red shapes the adversarial reward by favoring stopping at the goal against all other actions, as stopping at the goal is on the support of the expert policy, while other actions are not.

We demonstrate empirically that SAIL assigns significantly higher reward towards completing the task and corrects for the bias in Section 4.1.

To improve training stability, SAIL constrains the RL agent to the estimated support of the expert policy, where r gail provides a more reliable RL signal (Wang et al., 2019) .

As r red tends to be very small (ideally zero) for (s, a) ??? supp(?? E ), r sail discourages the agent from exploring those state-actions by masking away the rewards.

This is a desirable property as the quality of the RL signals beyond the support of the expert policy can't be guaranteed.

We demonstrate in Section 4.2 the improved training stability on the Mujoco benchmark tasks .

We provide the pseudocode implementation of SAIL in Algorithm 1.

The algorithm computes r red by estimating the support of the expert policy, followed by iterative updates of the policy and r gail .

We apply the Trust Region Policy Optimization (TRPO) algorithm (Schulman et al., 2015) with the reward r sail for policy updates.

Reward Variants In practice, we observe that constraining the range of the adversarial reward generally produces lower-variance policies.

Specifically, we transform r gail in Eq. (5)

For ease of notation, we refer to the bounded variant as SAIL-b, and the unbounded variant as SAIL.

Similarly, we denote the bounded GAIL reward as GAIL-b.

We include the comparison between the reward variants in the experiments.

In this section, we show that SAIL is at least as efficient as GAIL in its sample complexity for expert data, and provide comparable RL signals on the expert policy's support.

We note that our analysis could be similarly applied to other AIL methods, suggesting the broad applicability of our approach.

We begin from the asymptotic setting, where the number of expert trajectories tends to infinity.

In this case, both GAIL's, RED's and SAIL's discriminators ultimately recover the expert policy's support at convergence (see Ho & Ermon (2016) for GAIL and Wang et al. (2019) for RED; SAIL follows from their combination).

Moreover, for both GAIL and SAIL, the expert and agent policy distributions match exactly at convergence, implying a successful imitation learning.

Therefore, it is critical to characterize the rates of convergence of the two methods, namely their relative sample complexity with respect to the number of expert demonstrations.

Formally, let (s, a) ??? supp(?? E ).

Prototypical learning bounds for an estimator of the support r ??? 0 provide high probability bounds in the form of P(r(s, a) ??? c log(1/??)n ????? ) > 1 ??? ?? for any confidence ?? ??? (0, 1], with c a constant not depending on ?? or the number n of samples (i.e., expert state-actions).

Here, ?? > 0 represents the learning rate, namely how fast the estimator is converging to the support.

By choosing the reward in Eq. (5), we are leveraging the faster learning rates between ?? red and ?? gail , with respect to support estimation.

At the time being, no results are available to characterize the sample complexity of GAIL (loosely speaking, the ?? and c introduced above).

Therefore, we proceed by focusing on a relative comparison with SAIL.

In particular, we show the following (see appendix for a proof).

Proposition 1.

Assume that for any (s, a) ??? supp(?? E ) the rewards for RED and GAIL have the following learning rates in estimating the support

Then, for any ?? ??? (0, 1] and any (s, a) ??? supp(?? E ), the following holds

with probability at least 1 ??? ??, where R red and R gail are the upper bounds for r red and r gail , respectively.

Eq. (7) shows that SAIL is at least as fast as the faster among RED and GAIL with respect to support estimation, implying that SAIL is at least as efficient as GAIL in the sample complexity for expert data.

Eq. (7) also indicates the quality of the learned reward, as state-actions outside the expert's support should be assigned minimum reward.

Proposition 2.

For any (s, a) ??? supp(?? E ) and any ?? ??? (0, 1], we assume that

The following event holds with probability at least 1 ??? ?? that

Eq. (9) shows that on the expert policy's support, r sail is close to r gail up to a precision that improves with the number of expert state-actions.

SAIL thus provides RL signals comparable to GAIL on the expert policy's support.

It is also worth noting that the analysis could explain why r red + r gail is a less viable approach for combining the two RL signals.

The analogous bound to Eq. (7) would be the sum of errors from the two methods, implying the slower of the two learning rates, while Eq. (9) would improve only by a constant, as R gail would be absent from Eq. (9).

Our preliminary experiments indicated that r red + r gail performed noticeably worse than Eq. (5).

Lastly, we comment on whether the assumptions in Eqs. (6) and (8) are satisfied in practice.

Following the kernel-based version of RED (Wang et al., 2019) , we can borrow previous results from the set learning literature, which guarantee RED to have a rate of ?? red = 1/2 (De Vito et al., 2014; Rudi et al., 2017) .

These rates have been shown to be optimal.

Any estimator of the support cannot have faster rates than n ???1/2 , unless additional assumptions are imposed.

Learning rates for distribution matching with GANs are still an active area of research, and conclusive results characterizing the convergence rates of these estimators are not available.

We refer to Singh et al. (2018) for an in-depth analysis of the topic.

We evaluate the proposed method against BC, GAIL and RED on Lunar Lander and six Mujoco control tasks including Hopper, Reacher, HalfCheetah, Walker2d, Ant, and Humanoid.

We omit evaluation against methods using off-policy RL algorithms, as they are not the focus of this work.

We also note that support guidance is complementary to such methods.

We demonstrate that SAIL variants mitigate the survival bias in Lunar Lander (Fig. 1 ) from OpenAI Gym (Brockman et al., 2016) , while other baseline methods imitate the expert inconsistently.

In this task, the agent is required to control a spacecraft to safely land between the flags.

A human expert provided 10 demonstrations for this task as an imitation target.

We observe that even without the environment reward, Lunar Lander provides a natural RL signal by terminating episodes early when crashes are detected, thus encouraging the agent to avoid crashing.

Consequently, all methods are able to successfully imitate the expert and land the spacecraft appropriately.

SAIL variants perform slightly better than GAIL variants on the average reward, and achieve noticeably lower standard deviation.

The average performances and the standard deviations evaluated over 50 runs are presented in Table 1 .

To construct a more challenging task, we disable all early termination feature of the environment, thus removing the environment RL signals.

In this no-terminal environment, a training episode only ends after the time limit.

We present each algorithm's performance for the no-terminal setting in Table 1 .

SAIL variants outperform GAIL variants.

Specifically, we observe that GAIL learns to land for some initial conditions, while exhibit survival bias in other scenarios by hovering at the goal.

In contrast, SAIL variants are still able to recover the expert policy.

To visualize the shaping effect from support guidance, we plot the average learned reward for GAIL, SAIL-b and RED at goal states.

The goal states are selected from the expert trajectories and satisfy two conditions: 1) touching the ground (the state vector has indicator variables for ground contact), and 2) has "no op" as the corresponding action.

As the adversarial reward functions are dynamic, we snapshot the learned rewards when the algorithms obtain their best policies, respectively.

Fig. 3 shows the average rewards for each available action, averaged across all the goal states.

Compared against the other algorithms, SAIL-b assigns a significantly higher reward to "no op", which facilitates the agent learning.

Though GAIL and RED still favor "no op" to other actions, the differences in reward are much smaller, causing less consistent landing behaviors.

We further observe that all evaluated AIL methods oscillate between partially hovering behavior and landing behavior during policy learning.

The observation suggests that our method only partially addresses the survival bias, a limitation we will tackle in future works.

This is likely caused by SAIL's non-negative reward, despite the beneficial shaping effect from support estimation.

For additional experiment results and discussion on Lunar Lander, please refer to the appendix.

Mujoco control tasks have been commonly used as the standard benchmark for AIL.

We evaluate SAIL against GAIL, RED and BC on Hopper, Reacher, HalfCheetah, Walker2d, Ant and Humanoid.

We adopt the same experimental setup presented in Ho & Ermon (2016) by sub-sampling the expert trajectories every 20 samples.

Consistent with the observation from Kostrikov et al. (2019) , our preliminary experiments show that sub-sampling presents a more challenging setting, as BC is competitive with AIL when full trajectories are used.

In our experiments, we also adopt the minimum 1056.5 ?? 0.5 -9.1 ?? 4.1 -0.2 ?? 0.7 2372.8 ?? 8.8 1005.5 ?? 8.6 6012.0 ?? 434.9 GAIL 3826.5 ?? 3.2 -9.1 ?? 4.4 4604.7 ?? 77.6 5295.4 ?? 44.1 1013.3 ?? 16.0 8781.2 ?? 3112.6 GAIL-b 3810.5 ?? 8.1 -8.3 ?? 2.5 4510.0 ?? 68.0 5388.1 ?? 161.2 3413.1 ?? 744.7 10132.5 ?? 1859.3 SAIL 3824.7 ?? 6.6 -7.5 ?? 2.7 4747.5 ?? 43.4 5293.0 ?? 590.9 3330.4 ?? 729.4 9292.8 ?? 3190.0 SAIL-b 3811.6 ?? 3.8 -7.4 ?? 2.5 4632.2 ?? 59.1 5438.6 ?? 18.4 4176.3 ?? 203.1 10589.6 ?? 52.2 Table 2 : Episodic reward and standard deviation on the Mujoco tasks by different methods evaluated over 50 runs.

SAIL-b achieves overall the best performance, with significantly lower standard deviation, indicating the robustness of the learned policies.

number of expert trajectories specified in Ho & Ermon (2016) for each task.

More details on experiment setup are available in the appendix.

We apply each algorithm using 5 different random seeds in all Mujoco tasks.

Table 2 shows the performance comparison between the evaluated algorithms.

We report the mean performance and standard deviation for each algorithm over 50 evaluation runs, choosing the best policies obtained for each algorithm out of the 5 random seeds.

The results show that SAIL-b is comparable to GAIL on Hopper, and outperform the other methods on all other tasks.

We note that RED significantly underperforms in the sub-sampling setting, while Wang et al. (2019) used full trajectories in their experiments.

Across all tasks, SAIL-b generally achieves lower standard deviation compared to other algorithms, in particular for Humanoid, indicating the robustness of the learned policies.

We stress that standard deviation is also a critical metric, as it indicates the robustness of the learned policies when presented with different states.

For instance, the large standard deviations in Humanoid are caused by occasional crashes, which may be highly undesirable depending on the intended applications.

To illustrate robustness of the learned policies, we plot the histogram of all 50 evaluations in Humanoid for RED, GAIL-b and SAIL-b in Fig. 2 .

The figure shows that SAIL-b performs consistently with expert performance.

Though GAIL-b appears to be only slightly worse in average performance, the degradation is caused by occasional and highly undesirable crashes, suggesting incomplete imitation of the expert.

RED performs the worst in average performance, but is consistent with no failure modes detected.

The result suggests that the proposed method combines the advantages of both support guidance and adversarial learning.

Comparing SAIL against SAIL-b, we observe that the bounded variant generally produces policies with smaller standard deviations and better performances, especially for Ant and Humanoid.

This is likely due to the fact that SAIL-b receives equal contribution from both support guidance and adversarial learning, as r red and r gail have the same range in this formulation.

In addition, we note that GAIL fails to imitate the expert in Ant, while GAIL-b performs significantly better.

The results suggest that restricting the range of the adversarial reward could improve performance.

To assess the sensitivity with respect to random seeds, we plot the training progress against number of iterations for the evaluated algorithms in Fig. 4 , Each iteration consists of 1000 environment steps.

The figure reports mean and standard deviation of each algorithm, across the 5 random seeds.

Fig. 4 shows that SAIL-b is more sample efficient and stable in Reacher, Ant and Humanoid tasks; and is comparable to the other algorithms in the remaining tasks.

Consistent with our analysis in Section 3.3, SAIL-b appears at least as efficient as GAIL even when the support guidance (i.e., the performance of RED) suffers from insufficient expert data in Hopper, HalfCheetah and Walker2d.

In Reacher, Ant and Humanoid, SAIL-b benefits from the support guidance and achieves better performance and training stability.

In particular, we note that without support guidance, GAIL fails to imitate the expert in Ant (Fig. 4e) .

Similar failures were also observed in Kostrikov et al. (2019) .

GAIL is also more sensitive to initial conditions: in Humanoid, GAIL converged to sub-optimal policies in 2 out 5 seeds.

Lastly, while RED improves noticeably faster during early training in Humanoid, it converged to a sub-optimal policy eventually.

In this paper, we propose Support-guided Adversarial Imitation Learning by combining support guidance with adversarial imitation learning.

Our approach is complementary to existing adversarial imitation learning algorithms, and addresses several challenges associated with them.

More broadly, our results show that expert demonstrations contain rich sources of information for imitation learning.

Effectively combining different sources of reinforcement learning signals from the expert demonstrations produces more efficient and stable algorithms by constraining the policy search space; and appears to be a promising direction for future research.

10413.1 ?? 47.0 RED and SAIL use RND Burda et al. (2018) for support estimation.

We use the default networks from RED 4 .

We set ?? following the heuristic in Wang et al. (2019) that (s, a) from the expert trajectories mostly have reward close to 1.

For fair comparisons, all algorithms shared hyperparameters for each task.

We present them in the table below, including discriminator learning rate l D , discount factor ??, number of policy steps per iteration n G , and whether the policy has fixed variance.

All other hyperparameters are set to their default values from OpenAI's baselines.

To compare our method with the technique of introducing virtual absorbing state (AS) (Kostrikov et al., 2019) , we also construct a goal-terminal environment where the only terminal state is successful landing at the goal, because the AS technique cannot be directly applied in the no-terminal environment.

We present the results in Appendix C.

The results suggest that AS overall improves both the mean performance and standard deviations for both GAIL and SAIL.

Specifically, the technique is able to mitigates the survival bias in GAIL significantly.

However, SAIL still compares favorably to the technique in the goal-terminal environment.

Further, since AS and support guidance is not mutually exclusive, we also combine them and report the performances.

The results suggest that support guidance is compatible with AS, and achieves overall the best performance with low standard deviations.

The results also suggest that both AS and support guidance partially mitigate the reward bias, but don't fully solve it.

We will further explore this issue in future work.

<|TLDR|>

@highlight

We unify support estimation with the family of Adversarial Imitation Learning algorithms into Support-guided Adversarial Imitation Learning, a more robust and stable imitation learning framework.
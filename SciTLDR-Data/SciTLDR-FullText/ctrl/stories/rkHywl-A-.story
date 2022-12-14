Reinforcement learning provides a powerful and general framework for decision making and control, but its application in practice is often hindered by the need for extensive feature and reward engineering.

Deep reinforcement learning methods can remove the need for explicit engineering of policy or value features, but still require a manually specified reward function.

Inverse reinforcement learning holds the promise of automatic reward acquisition, but has proven exceptionally difficult to apply to large, high-dimensional problems with unknown dynamics.

In this work, we propose AIRL, a practical and scalable inverse reinforcement learning algorithm based on an adversarial reward learning formulation that is competitive with direct imitation learning algorithms.

Additionally, we show that AIRL is able to recover portable reward functions that are robust to changes in dynamics, enabling us to learn policies even under significant variation in the environment seen during training.

While reinforcement learning (RL) provides a powerful framework for automating decision making and control, significant engineering of elements such as features and reward functions has typically been required for good practical performance.

In recent years, deep reinforcement learning has alleviated the need for feature engineering for policies and value functions, and has shown promising results on a range of complex tasks, from vision-based robotic control BID12 to video games such as Atari BID13 and Minecraft BID16 .

However, reward engineering remains a significant barrier to applying reinforcement learning in practice.

In some domains, this may be difficult to specify (for example, encouraging "socially acceptable" behavior), and in others, a naïvely specified reward function can produce unintended behavior BID2 .

Moreover, deep RL algorithms are often sensitive to factors such as reward sparsity and magnitude, making well performing reward functions particularly difficult to engineer.

Inverse reinforcement learning (IRL) BID19 BID14 refers to the problem of inferring an expert's reward function from demonstrations, which is a potential method for solving the problem of reward engineering.

However, inverse reinforcement learning methods have generally been less efficient than direct methods for learning from demonstration such as imitation learning BID10 , and methods using powerful function approximators such as neural networks have required tricks such as domain-specific regularization and operate inefficiently over whole trajectories BID6 .

There are many scenarios where IRL may be preferred over direct imitation learning, such as re-optimizing a reward in novel environments BID7 or to infer an agent's intentions, but IRL methods have not been shown to scale to the same complexity of tasks as direct imitation learning.

However, adversarial IRL methods BID6 a) hold promise for tackling difficult tasks due to the ability to adapt training samples to improve learning efficiency.

Part of the challenge is that IRL is an ill-defined problem, since there are many optimal policies that can explain a set of demonstrations, and many rewards that can explain an optimal policy BID15 .

The maximum entropy (MaxEnt) IRL framework introduced by BID24 handles the former ambiguity, but the latter ambiguity means that IRL algorithms have difficulty distinguishing the true reward functions from those shaped by the environment dynamics.

While shaped rewards can increase learning speed in the original training environment, when the reward is deployed at test-time on environments with varying dynamics, it may no longer produce optimal behavior, as we discuss in Sec. 5.

To address this issue, we discuss how to modify IRL algorithms to learn rewards that are invariant to changing dynamics, which we refer to as disentangled rewards.

In this paper, we propose adversarial inverse reinforcement learning (AIRL), an inverse reinforcement learning algorithm based on adversarial learning.

Our algorithm provides for simultaneous learning of the reward function and value function, which enables us to both make use of the efficient adversarial formulation and recover a generalizable and portable reward function, in contrast to prior works that either do not recover a reward functions BID10 , or operates at the level of entire trajectories, making it difficult to apply to more complex problem settings BID6 a) .

Our experimental evaluation demonstrates that AIRL outperforms prior IRL methods BID6 on continuous, high-dimensional tasks with unknown dynamics by a wide margin.

When compared to GAIL BID10 , which does not attempt to directly recover rewards, our method achieves comparable results on tasks that do not require transfer.

However, on tasks where there is considerable variability in the environment from the demonstration setting, GAIL and other IRL methods fail to generalize.

In these settings, our approach, which can effectively disentangle the goals of the expert from the dynamics of the environment, achieves superior results.

Inverse reinforcement learning (IRL) is a form of imitation learning and learning from demonstration BID3 .

Imitation learning methods seek to learn policies from expert demonstrations, and IRL methods accomplish this by first inferring the expert's reward function.

Previous IRL approaches have included maximum margin approaches BID0 BID18 , and probabilistic approaches such as BID24 ; BID4 .

In this work, we work under the maximum causal IRL framework of BID23 .

Some advantages of this framework are that it removes ambiguity between demonstrations and the expert policy, and allows us to cast the reward learning problem as a maximum likelihood problem, connecting IRL to generative model training.

Our proposed method most closely resembles the algorithms proposed by BID21 ; BID10 BID5 .

Generative adversarial imitation learning (GAIL) BID10 differs from our work in that it is not an IRL algorithm that seeks to recover reward functions.

The critic or discriminator of GAIL is unsuitable as a reward since, at optimality, it outputs 0.5 uniformly across all states and actions.

Instead, GAIL aims only to recover the expert's policy, which is a less portable representation for transfer.

BID21 does not interleave policy optimization with reward learning within an adversarial framework.

Improving a policy within an adversarial framework corresponds to training an amortized sampler for an energy-based model, and prior work has shown this is crucial for performance BID6 .

BID22 also consider learning cost functions with neural networks, but only evaluate on simple domains where analytically solving the problem with value iteration is tractable.

Previous methods which aim to learn nonlinear cost functions have used boosting BID17 and Gaussian processes BID11 , but still suffer from the feature engineering problem.

Our IRL algorithm builds on the adversarial IRL framework proposed by BID5 , with the discriminator corresponding to an odds ratio between the policy and exponentiated reward distribution.

The discussion in BID5 is theoretical, and to our knowledge no prior work has reported a practical implementation of this method.

Our experiments show that direct implementation of the proposed algorithm is ineffective, due to high variance from operating over entire trajectories.

While it is straightforward to extend the algorithm to single state-action pairs, as we discuss in Section 4, a simple unrestricted form of the discriminator is susceptible to the reward ambiguity described in BID15 , making learning the portable reward functions difficult.

As illustrated in our experiments, this greatly limits the generalization capability of the method: the learned reward functions are not robust to environment changes, and it is difficult to use the algo-rithm for the purpose of inferring the intentions of agents.

We discuss how to overcome this issue in Section 5.

BID1 consider learning reward functions which generalize to new tasks given multiple training tasks.

Our work instead focuses on how to achieve generalization within the standard IRL formulation.

Our inverse reinforcement learning method builds on the maximum causal entropy IRL framework BID23 , which considers an entropy-regularized Markov decision process (MDP), defined by the tuple (S, A, T , r, γ, ρ 0 ).

S, A are the state and action spaces, respectively, γ ∈ (0, 1) is the discount factor.

The dynamics or transition distribution T (s |a, s), the initial state distribution ρ 0 (s), and the reward function r(s, a) are unknown in the standard reinforcement learning setup and can only be queried through interaction with the MDP.The goal of (forward) reinforcement learning is to find the optimal policy π * that maximizes the expected entropy-regularized discounted reward, under π, T , and ρ 0 : DISPLAYFORM0 where τ = (s 0 , a 0 , ...s T , a T ) denotes a sequence of states and actions induced by the policy and dynamics.

It can be shown that the trajectory distribution induced by the optimal policy π * (a|s) takes the form π * (a|s) ∝ exp{Q * soft (s t , a t )} BID23 BID9 , where DISPLAYFORM1 Inverse reinforcement learning instead seeks infer the reward function r(s, a) given a set of demonstrations D = {τ 1 , ..., τ N }.

In IRL, we assume the demonstrations are drawn from an optimal policy π * (a|s).

We can interpret the IRL problem as solving the maximum likelihood problem: DISPLAYFORM2 Where st,at) parametrizes the reward function r θ (s, a) but fixes the dynamics and initial state distribution to that of the MDP.

Note that under deterministic dynamics, this simplifies to an energy-based model where for feasible trajectories, p θ (τ ) ∝ e T t=0 γ t r θ (st,at) BID24 .

BID5 propose to cast optimization of Eqn.

1 as a GAN BID8 optimization problem.

They operate in a trajectory-centric formulation, where the discriminator takes on a particular form (f θ (τ ) is a learned function; π(τ ) is precomputed and its value "filled in"): DISPLAYFORM3 DISPLAYFORM4 and the policy π is trained to maximize DISPLAYFORM5 Updating the discriminator can be viewed as updating the reward function, and updating the policy can be viewed as improving the sampling distribution used to estimate the partition function.

If trained to optimality, it can be shown that an optimal reward function can be extracted from the optimal discriminator as f * (τ ) = R * (τ )+const, and π recovers the optimal policy.

We refer to this formulation as generative adversarial network guided cost learning (GAN-GCL) to discriminate it from guided cost learning (GCL) BID5 .

This formulation shares similarities with GAIL BID10 , but GAIL does not place special structure on the discriminator, so the reward cannot be recovered.

In practice, using full trajectories as proposed by GAN-GCL can result in high variance estimates as compared to using single state, action pairs, and our experimental results show that this results in very poor learning.

We could instead propose a straightforward conversion of Eqn.

2 into the single state and action case, where: DISPLAYFORM0 .As in the trajectory-centric case, we can show that, at optimality, f * (s, a) = log π * (a|s) = A * (s, a), the advantage function of the optimal policy.

We justify this, as well as a proof that this algorithm solves the IRL problem in Appendix A .This change results in an efficient algorithm for imitation learning.

However, it is less desirable for the purpose of reward learning.

While the advantage is a valid optimal reward function, it is a heavily entangled reward, as it supervises each action based on the action of the optimal policy for the training MDP.

Based on the analysis in the following Sec. 5, we cannot guarantee that this reward will be robust to changes in environment dynamics.

In our experiments we demonstrate several cases where this reward simply encourages mimicking the expert policy π * , and fails to produce desirable behavior even when changes to the environment are made.

We now discuss why IRL methods can fail to learn robust reward functions.

First, we review the concept of reward shaping.

BID15 describe a class of reward transformations that preserve the optimal policy.

Their main theoretical result is that under the following reward transformation, DISPLAYFORM0 the optimal policy remains unchanged, for any function Φ : S → R. Moreover, without prior knowledge of the dynamics, this is the only class of reward transformations that exhibits policy invariance.

Because IRL methods only infer rewards from demonstrations given from an optimal agent, they cannot in general disambiguate between reward functions within this class of transformations, unless the class of learnable reward functions is restricted.

We argue that shaped reward functions may not be robust to changes in dynamics.

We formalize this notion by studying policy invariance in two MDPs M, M which share the same reward and differ only in the dynamics, denoted as T and T , respectively.

Suppose an IRL algorithm recovers a shaped, policy invariant rewardr(s, a, s ) under MDP M where Φ = 0.

Then, there exists MDP pairs M, M where changing the transition model from T to T breaks policy invariance on MDP M .

As a simple example, consider deterministic dynamics T (s, a) → s and state-action rewardsr(s, a) = r(s, a) + γΦ(T (s, a)) − Φ(s).

It is easy to see that changing the dynamics T to T such that T (s, a) = T (s, a) means thatr(s, a) no longer lies in the equivalence class of Eqn.

3 for M .

First, let the notation Q * r,T (s, a) denote the optimal Q-function with respect to a reward function r and dynamics T , and π * r,T (a|s) denote the same for policies.

We first define our notion of a "disentangled" reward.

Definition 5.1 (Disentangled Rewards).

A reward function r (s, a, s ) is (perfectly) disentangled with respect to a ground-truth reward r(s, a, s ) and a set of dynamics T such that under all dynamics T ∈ T , the optimal policy is the same: π * r ,T (a|s) = π * r,T (a|s)We could also expand this definition to include a notion of suboptimality.

However, we leave this direction to future work.

Under maximum causal entropy RL, the following condition is equivalent to two optimal policies being equal, since Q-functions and policies are equivalent representations (up to arbitrary functions of state f (s)): DISPLAYFORM0 To remove unwanted reward shaping with arbitrary reward function classes, the learned reward function can only depend on the current state s. We require that the dynamics satisfy a decomposability Collect trajectories τ i = (s 0 , a 0 , ..., s T , a T ) by executing π.

Train D θ,φ via binary logistic regression to classify expert data τ E i from samples τ i .6: DISPLAYFORM0 Update π with respect to r θ,φ using any policy optimization method.

8: end for condition where functions over current states f (s) and next states g(s ) can be isolated from their sum f (s) + g(s ).

This can be satisfied for example by adding self transitions at each state to an ergodic MDP, or any of the environments used in our experiments.

The exact definition of the condition, as well as proof of the following statements are included in Appendix B. Theorem 5.1.

Let r(s) be a ground-truth reward, and T be a dynamics model satisfying the decomposability condition.

Suppose IRL recovers a state-only reward r (s) such that it produces an optimal policy in T : DISPLAYFORM1 DISPLAYFORM2 Then r is only a function of state.

In the traditional IRL setup, where we learn the reward in a single MDP, our analysis motivates learning reward functions that are solely functions of state.

If the ground truth reward is also only a function of state, this allows us to recover the true reward up to a constant.

In the method presented in Section 4, we cannot learn a state-only reward function, r θ (s), meaning that we cannot guarantee that learned rewards will not be shaped.

In order to decouple the reward function from the advantage, we propose to modify the discriminator of Sec. 4 with the form: DISPLAYFORM0 where f θ,φ is restricted to a reward approximator g θ and a shaping term h φ as DISPLAYFORM1 The additional shaping term helps mitigate the effects of unwanted shaping on our reward approximator g θ (and as we will show, in some cases it can account for all shaping effects).

The entire training procedure is detailed in Algorithm 1.

Our algorithm resembles GAIL BID10 and GAN-GCL BID5 , where we alternate between training a discriminator to classify expert data from policy samples, and update the policy to confuse the discriminator.

The advantage of this approach is that we can now parametrize g θ (s) as solely a function of the state, allowing us to extract rewards that are disentangled from the dynamics of the environment in which they were trained.

In fact, under this restricted case, we can show the following under deterministic environments with a state-only ground truth reward (proof in Appendix C): DISPLAYFORM2 where r * is the true reward function.

Since f * must recover to the advantage as shown in Sec. 4, h recovers the optimal value function V * , which serves as the reward shaping term.

To be consistent with Sec. 4, an alternative way to interpret the form of Eqn.

4 is to view f θ,φ as the advantage under deterministic dynamics DISPLAYFORM3 In stochastic environments, we can instead view f (s, a, s ) as a single-sample estimate of A * (s, a).

In our experiments, we aim to answer two questions:1.

Can AIRL learn disentangled rewards that are robust to changes in environment dynamics?2.

Is AIRL efficient and scalable to high-dimensional continuous control tasks?To answer 1, we evaluate AIRL in transfer learning scenarios, where a reward is learned in a training environment, and optimized in a test environment with significantly different dynamics.

We show that rewards learned with our algorithm under the constraint presented in Section 5 still produce optimal or near-optimal behavior, while naïve methods that do not consider reward shaping fail.

We also show that in small MDPs, we can recover the exact ground truth reward function.

To answer 2, we compare AIRL as an imitation learning algorithm against GAIL BID10 ) and the GAN-based GCL algorithm proposed by BID5 , which we refer to as GAN-GCL, on standard benchmark tasks that do not evaluate transfer.

Note that BID5 does not implement or evaluate GAN-GCL and, to our knowledge, we present the first empirical evaluation of this algorithm.

We find that AIRL performs on par with GAIL in a traditional imitation learning setup while vastly outperforming it in transfer learning setups, and outperforms GAN-GCL in both settings.

It is worth noting that, except for BID6 , our method is the only IRL algorithm that we are aware of that scales to high dimensional tasks with unknown dynamics, and although GAIL BID10 resembles an IRL algorithm in structure, it does not recover disentangled reward functions, making it unable to re-optimize the learned reward under changes in the environment, as we illustrate below.

For our continuous control tasks, we use trust region policy optimization BID20 as our policy optimization algorithm across all evaluated methods, and in the tabular MDP task, we use soft value iteration.

We obtain expert demonstrations by training an expert policy on the ground truth reward, but hide the ground truth reward from the IRL algorithm.

In this way, we simulate a scenario where we wish to use RL to solve a task but wish to refrain from manual reward engineering and instead seek to learn a reward function from demonstrations.

Our code and additional supplementary material including videos will be available at https://sites.google.com/view/ adversarial-irl, and hyper-parameter and architecture choices are detailed in Appendix D.

We first consider MaxEnt IRL in a toy task with randomly generated MDPs.

The MDPs have 16 states, 4 actions, randomly drawn transition matrices, and a reward function that always gives a reward of 1.0 when taking an action from state 0.

The initial state is always state 1.The optimal reward, learned reward with a state-only reward function, and learned reward using a state-action reward function are shown in FIG1 .

We subtract a constant offset from all reward functions so that they share the same mean for visualization -this does not influence the optimal policy.

AIRL with a state-only reward function is able to recover the ground truth reward, but AIRL with a state-action reward instead recovers a shaped advantage function.

We also show that in the transfer learning setup, under a new transition matrix T , the optimal policy under the state-only reward achieves optimal performance (it is identical to the ground truth reward) whereas the state-action reward only improves marginally over uniform random policy.

The learning curve for this experiment is shown in Learning curve for the transfer learning experiment on tabular MDPs.

Value iteration steps are plotted on the x-axis, against returns for the policy on the y-axis.

To evaluate whether our method can learn disentangled rewards in higher dimensional environments, we perform transfer learning experiments on continuous control tasks.

In each task, a reward is learned via IRL on the training environment, and the reward is used to reoptimize a new policy on a test environment.

We train two IRL algorithms, AIRL and GAN-GCL, with state-only and stateaction rewards.

We also include results for directly transferring the policy learned with GAIL, and an oracle result that involves optimizing the ground truth reward function with TRPO.

Numerical results for these environment transfer experiments are given in TAB2 The first task involves a 2D point mass navigating to a goal position in a small maze when the position of the walls are changed between train and test time.

At test time, the agent cannot simply mimic the actions learned during training, and instead must successfully infer that the goal in the maze is to reach the target.

The task is shown in FIG3 .

Only AIRL trained with state-only rewards is able to consistently navigate to the goal when the maze is modified.

Direct policy transfer and state-action IRL methods learn rewards which encourage the agent to take the same path taken in the training environment, which is blocked in the test environment.

We plot the learned reward in FIG4 In our second task, we modify the agent itself.

We train a quadrupedal "ant" agent to run forwards, and at test time we disable and shrink two of the front legs of the ant such that it must significantly change its gait.

We find that AIRL is able to learn reward functions that encourage the ant to move forwards, acquiring a modified gait that involves orienting itself to face the forward direction and crawling with its two hind legs.

Alternative methods, including transferring a policy learned by GAIL (which achieves near-optimal performance with the unmodified agent), fail to move forward at all.

We show the qualitative difference in behavior in FIG5 .We have demonstrated that AIRL can learn disentangled rewards that can accommodate significant domain shift even in high-dimensional environments where it is difficult to exactly extract the true reward.

GAN-GCL can presumably learn disentangled rewards, but we find that the trajectorycentric formulation does not perform well even in learning rewards in the original task, let alone transferring to a new domain.

GAIL learns successfully in the training domain, but does not acquire a representation that is suitable for transfer to test domains.

Illustration of the shifting maze task, where the agent (blue) must reach the goal (green).

During training the agent must go around the wall on the left side, but during test time it must go around on the right.

Bottom row: Behavior acquired by optimizing a state-only reward learned with AIRL on the disabled ant environment.

Note that the ant must orient itself before crawling forward, which is a qualitatively different behavior from the optimal policy in the original environment, which runs sideways.

Finally, we evaluate AIRL as an imitation learning algorithm against the GAN-GCL and the stateof-the-art GAIL on several benchmark tasks.

Each algorithm is presented with 50 expert demonstrations, collected from a policy trained with TRPO on the ground truth reward function.

For AIRL, we use an unrestricted state-action reward function as we are not concerned with reward transfer.

Numerical results are presented in TAB3 .These experiments do not test transfer, and in a sense can be regarded as "testing on the training set," but they match the settings reported in prior work BID10 .We find that the performance difference between AIRL and GAIL is negligible, even though AIRL is a true IRL algorithm that recovers reward functions, while GAIL does not.

Both methods achieve close to the best possible result on each task, and there is little room for improvement.

This result goes against the belief that IRL algorithms are indirect, and less efficient that direct imitation learning algorithms BID10 .

The GAN-GCL method is ineffective on all but the simplest Pendulum task when trained with the same number of samples as AIRL and GAIL.

We find that a discriminator trained over trajectories easily overfits and provides poor learning signal for the policy.

Our results illustrate that AIRL achieves the same performance as GAIL on benchmark imitation tasks that do not require any generalization.

On tasks that require transfer and generalization, illustrated in the previous section, AIRL outperforms GAIL by a wide margin, since our method is able to recover disentangled rewards that transfer effectively in the presence of domain shift.

We presented AIRL, a practical and scalable IRL algorithm that can learn disentangled rewards and greatly outperforms both prior imitation learning and IRL algorithms.

We show that rewards learned with AIRL transfer effectively under variation in the underlying domain, in contrast to unmodified IRL methods which tend to recover brittle rewards that do not generalize well and GAIL, which does not recover reward functions at all.

In small MDPs where the optimal policy and reward are unambiguous, we also show that we can exactly recover the ground-truth rewards up to a constant.

In this section, we show that the objective of AIRL matches that of solving the maximum causal entropy IRL problem.

We use a similar method as BID5 , which shows the justification of GAN-GCL for the trajectory-centric formulation.

For simplicity we derive everything in the undiscounted case.

A.1 SETUP As mentioned in Section 3, the goal of IRL can be seen as training a generative model over trajectories as: max st,at) .

We can compute the gradient with respect to θ as follows: DISPLAYFORM0 DISPLAYFORM1 Let p θ,t (s t , a t ) = s t =t ,a t =t p θ (τ ) denote the state-action marginal at time t. Rewriting the above equation, we have: DISPLAYFORM2 As it is difficult to draw samples from p θ , we instead train a separate importance sampling distribution µ(τ ).

For the choice of this distribution, we follow BID5 and use a mixture policy µ(a|s) = 1 2 π(a|s) + 1 2p (a|s), wherep(a|s) is a rough density estimate trained on the demonstrations.

This is justified as reducing the variance of the importance sampling estimate when the policy π(a|s) has poor coverage over the demonstrations in the early stages of training.

Thus, our new gradient is: DISPLAYFORM3 We additionally wish to adapt the importance sampler π to reduce variance, by min- DISPLAYFORM4 The policy trajectory distribution factorizes as π(τ ) = p(s 0 )T −1 t=0 p(s t+1 |s t , a t )π(a t |s t ).

The dynamics and initial state terms inside π(τ ) and p θ (τ ) cancel, leaving the entropy-regularized policy objective: DISPLAYFORM5 In AIRL, we replace the cost learning objective with training a discriminator of the following form: DISPLAYFORM6 The objective of the discriminator is to minimize cross-entropy loss between expert demonstrations and generated samples: DISPLAYFORM7 We replace the policy optimization objective with the following reward: DISPLAYFORM8 A.2 DISCRIMINATOR OBJECTIVE First, we show that training the gradient of the discriminator objective is the same as Eqn.

5.

We write the negative loss to turn the minimization problem into maximization, and use µ to denote a mixture between the dataset and policy samples.

DISPLAYFORM9 Taking the derivative w.r.t.

θ, DISPLAYFORM10 Multiplying the top and bottom of the fraction in the second expectation by the state marginal π(s t ) = a π t (s t , a t ), and grouping terms we get: DISPLAYFORM11 Where we have writtenp θ,t (s t , a t ) = exp{f θ (s t , a t )}π t (s t ), andμ to denote a mixture between p θ (s, a) and policy samples.

This expression matches Eqn.

5, with f θ (s, a) serving as the reward function, when π maximizes the policy objective so thatp θ (s, a) = p θ (s, a).

Next, we show that the policy objective matches that of the sampler of Eqn.

6.

The objective of the policy is to maximize with respect to the rewardr t (s, a).

First, note that: DISPLAYFORM0 Thus, whenr(s, a) is summed over entire trajectories, we obtain the entropy-regularized policy objective DISPLAYFORM1 Where f θ serves as the reward function.

DISPLAYFORM2 The global minimum of the discriminator objective is achieved when π = π E , where π denotes the learned policy (the "generator" of a GAN) and π E denotes the policy under which demonstrations were collected BID8 .

At this point, the output of the discriminator is 1 2 for all values of s, a, meaning we have exp{f θ (s, a)} = π E (a|s), or f * (s, a) = log π E (a|s) = A * (s, a).

In this section we include proofs for Theorems 5.1 and 5.2, and the condition on the dynamics necessary for them to hold.

Definition B.1 (Decomposability Condition).

Two states s 1 , s 2 are defined as "1-step linked" under a dynamics or transition distribution T (s |a, s) if there exists a state s that can reach s 1 and s 2 with positive probability in one time step.

Also, we define that this relationship can transfer through transitivity: if s 1 and s 2 are linked, and s 2 and s 3 are linked, then we also consider s 1 and s 3 to be linked.

A transition distribution T satisfies the decomposability condition if all states in the MDP are linked with all other states.

The key reason for needing this condition is that it allows us to decompose the functions state dependent f (s) and next state dependent g(s ) from their sum f (s) + g(s ), as stated below: Lemma B.1.

Suppose the dynamics for an MDP satisfy the decomposability condition.

Then, for functions a(s), b(s), c(s), d(s), if for all s, s : DISPLAYFORM0 Then for for all s, a(s) = c(s) + const DISPLAYFORM1 Proof.

Rearranging, we have: DISPLAYFORM2 for some function only dependent on s. In order for this to be representable, the term b(s ) − d(s ) must be equal for all successor states s from s. Under the decomposability condition, all successor states must therefore be equal in this manner through transitivity, meaning we have b(s ) − d(s ) must be constant with respect to s. Therefore, a(s) = c(s) + const.

We can then substitute this expression back in to the original equation to derive b(s) = d(s) + const.

We consider the case when the ground truth reward is state-only.

We now show that if the learned reward is also state-only, then we guarantee learning disentangled rewards, and vice-versa (sufficiency and necessity).

Theorem 5.1.

Let r(s) be a ground-truth reward, and T be a dynamics model satisfying the decomposability condition.

Suppose IRL recovers a state-only reward r (s) such that it produces an optimal policy in T : Q * r ,T (s, a) = Q * r,T (s, a) − f (s) Then, r (s) is disentangled with respect to all dynamics.

Proof.

We show that r (s) must equal the ground-truth reward up to constants (modifying rewards by constants does not change the optimal policy).

Let r (s) = r(s) + φ(s) for some arbitrary function of state φ(s).

We have: Proof.

We show the converse, namely that if r (s, a, s ) can depend on a or s , then there exists a dynamics model T such that the optimal policy is changed, i.e. Q * r,T (s, a) = Q * r ,T (s, a) + f (s) ∀s, a.

Consider the following 3-state MDP with deterministic dynamics and starting state S: We denote the action with a small letter, i.e. taking the action a from S brings the agent to state A, receiving a reward of 0.

For simplicity, assume the discount factor γ = 1.

The optimal policy here takes the a action, returns to s, and repeat for infinite positive reward.

An action-dependent reward which induces the same optimal policy would be to move the reward from the action returning to s to the action going to a or s: Optimizing r on this new MDP results in a different policy than optimizing r, as the agent visits B, resulting in infinite negative reward.

In this section, we prove that AIRL can recover the ground truth reward up to constants if the ground truth is only a function of state r(s).

For simplicity, we consider deterministic environments, so that s is uniquely defined by s, a, and we restrict AIRL's reward estimator g to only be a function of state.

<|TLDR|>

@highlight

We propose an adversarial inverse reinforcement learning algorithm capable of learning reward functions which can transfer to new, unseen environments.
We consider a problem of learning the reward and policy from expert examples under unknown dynamics.

Our proposed method builds on the framework of generative adversarial networks and introduces the empowerment-regularized maximum-entropy inverse reinforcement learning to learn near-optimal rewards and policies.

Empowerment-based regularization prevents the policy from overfitting to expert demonstrations, which advantageously leads to more generalized behaviors that result in learning near-optimal rewards.

Our method simultaneously learns empowerment through variational information maximization along with the reward and policy under the adversarial learning formulation.

We evaluate our approach on various high-dimensional complex control tasks.

We also test our learned rewards in challenging transfer learning problems where training and testing environments are made to be different from each other in terms of dynamics or structure.

The results show that our proposed method not only learns near-optimal rewards and policies that are matching expert behavior but also performs significantly better than state-of-the-art inverse reinforcement learning algorithms.

Reinforcement learning (RL) has emerged as a promising tool for solving complex decision-making and control tasks from predefined high-level reward functions BID23 .

However, defining an optimizable reward function that inculcates the desired behavior can be challenging for many robotic applications, which include learning social-interaction skills BID17 , dexterous manipulation BID5 , and autonomous driving BID10 .Inverse reinforcement learning (IRL) BID14 addresses the problem of learning reward functions from expert demonstrations, and it is often considered as a branch of imitation learning BID2 ).

The prior work in IRL includes maximum-margin BID0 BID18 and maximum-entropy BID24 formulations.

Currently, maximum entropy (MaxEnt) IRL is a widely used approach towards IRL, and has been extended to use non-linear function approximators such as neural networks in scenarios with unknown dynamics by leveraging sampling-based techniques BID3 BID5 BID9 .

However, designing the IRL algorithm is usually complicated as it requires, to some extent, hand engineering such as deciding domain-specific regularizers BID5 .Rather than learning reward functions and solving the IRL problem, imitation learning (IL) learns a policy directly from expert demonstrations.

Prior work addressed the IL problem through behavior cloning (BC), which learns a policy from expert trajectories using supervised learning BID15 .

Although BC methods are simple solutions to IL, these methods require a large amount of data because of compounding errors induced by covariate shift BID19 .

To overcome BC limitations, a generative adversarial imitation learning (GAIL) algorithm BID8 was proposed.

GAIL uses the formulation of Generative Adversarial Networks (GANs) BID7 , i.e., a generator-discriminator framework, where a generator is trained to generate expert-like trajectories while a discriminator is trained to distinguish between generated and expert trajectories.

Although GAIL is highly effective and efficient framework, it does not recover transferable/portable reward functions along with the policies, thus narrowing its use cases to similar problem instances in similar environments.

Reward function learning is ultimately preferable, if possible, over direct imitation learning as rewards are portable functions that represent the most basic and complete representation of agent intention, and can be re-optimized in new environments and new agents.

Reward learning is challenging as there can be many optimal policies explaining a set of demonstrations and many reward functions inducing an optimal policy BID14 BID24 .

Recently, an adversarial inverse reinforcement learning (AIRL) framework BID6 , an extension of GAIL, was proposed that offers a solution to the former issue by exploiting the maximum entropy IRL method BID24 whereas the latter issue is addressed through learning disentangled reward functions by modeling the reward as a function of state only instead of both state and action.

However, AIRL fails to recover the ground truth reward when the ground truth reward is a function of both state and action.

For example, the reward function in any locomotion or ambulation tasks contains a penalty term that discourages actions with large magnitudes.

This need for action regularization is well known in optimal control literature and limits the use cases of a state-only reward function in most practical real-life applications.

A more generalizable and useful approach would be to formulate reward as a function of both states and actions, which induces action-driven reward shaping that has been shown to play a vital role in quickly recovering the optimal policies BID13 .In this paper, we propose the empowerment-regularized adversarial inverse reinforcement learning (EAIRL) algorithm 1 .

Empowerment BID20 ) is a mutual information-based theoretic measure, like state-or action-value functions, that assigns a value to a given state to quantify the extent to which an agent can influence its environment.

Our method uses variational information maximization BID12 to learn empowerment in parallel to learning the reward and policy from expert data.

Empowerment acts as a regularizer to policy updates to prevent overfitting the expert demonstrations, which in practice leads to learning robust rewards.

Our experimentation shows that the proposed method recovers not only near-optimal policies but also recovers robust, transferable, disentangled, state-action based reward functions that are near-optimal.

The results on reward learning also show that EAIRL outperforms several state-of-the-art IRL methods by recovering reward functions that leads to optimal, expert-matching behaviors.

On policy learning, results demonstrate that policies learned through EAIRL perform comparably to GAIL and AIRL with non-disentangled (state-action) reward function but significantly outperform policies learned through AIRL with disentangled reward (state-only) and GAN interpretation of Guided Cost Learning (GAN-GCL) BID4 .

We consider a Markov decision process (MDP) represented as a tuple (S, A, P, R, ?? 0 , ??) where S denotes the state-space, A denotes the action-space, P represents the transition probability distribution, i.e., P : S ?? A ?? S ??? [0, 1], R(s, a) corresponds to the reward function, ?? 0 is the initial state distribution ?? 0 : S ??? R, and ?? ??? (0, 1) is the discount factor.

Let q(a|s, s ) be an inverse model that maps current state s ??? S and next state s ??? S to a distribution over actions A, i.e., q : S ?? S ?? A ??? [0, 1].

Let ?? be a stochastic policy that takes a state and outputs a distribution over actions such that ?? : S ?? A ??? [0, 1].

Let ?? and ?? E denote a set of trajectories, a sequence of state-action pairs (s 0 , a 0 , ?? ?? ?? s T , a T ), generated by a policy ?? and an expert policy ?? E , respectively, where T denotes the terminal time.

Finally, let ??(s) be a potential function that quantifies a utility of a given state s ??? S, i.e., ?? : S ??? R. In our proposed work, we use an empowerment-based potential function ??(??) to regularize policy update under MaxEnt-IRL framework.

Therefore, the following sections provide a brief background on MaxEnt-IRL, adversarial reward and policy learning, and variational information-maximization approach to learn the empowerment.

MaxEnt-IRL BID24 ) models expert demonstrations as Boltzmann distribution using parametrized reward r ?? (?? ) as an energy function, i.e., DISPLAYFORM0 where r ?? (?? ) = T t=0 r ?? (s t , a t ) is a commutative reward over given trajectory ?? , parameterized by ??, and Z is the partition function.

In this framework, the demonstration trajectories are assumed to be sampled from an optimal policy ?? * , therefore, they get the highest likelihood whereas the suboptimal trajectories are less rewarding and hence, are generated with exponentially decaying probability.

The main computational challenge in MaxEnt-IRL is to determine Z. The initial work in MaxEnt-IRL computed Z using dynamic programming BID24 whereas modern approaches BID5 a; BID6 present importance sampling technique to approximate Z under unknown dynamics.

This section briefly describes Adversarial Inverse Reinforcement Learning (AIRL) BID6 algorithm which forms a baseline of our proposed method.

AIRL is the current state-of-the-art IRL method that builds on GAIL BID8 , maximum entropy IRL framework BID24 and GAN-GCL, a GAN interpretation of Guided Cost Learning BID5 a) .

GAIL is a model-free adversarial learning framework, inspired from GANs BID7 , where the policy ?? learns to imitate the expert policy behavior ?? E by minimizing the JensenShannon divergence between the state-action distributions generated by ?? and the expert state-action distribution by ?? E through following objective DISPLAYFORM0 where D is the discriminator that performs the binary classification to distinguish between samples generated by ?? and ?? E , ?? is a hyper-parameter, and H(??) is an entropy regularization term E ?? [log ??].

Note that GAIL does not recover reward; however, BID4 shows that the discriminator can be modeled as a reward function.

Thus AIRL BID6 ) presents a formal implementation of BID4 and extends GAIL to recover reward along with the policy by imposing a following structure on the discriminator: DISPLAYFORM1 where f ??,?? (s, a, s ) = r ?? (s) + ??h ?? (s ) ??? h ?? (s) comprises a disentangled reward term r ?? (s) with training parameters ??, and a shaping term F = ??h ?? (s ) ??? h ?? (s) with training parameters ??. The entire D ??,?? (s, a, s ) is trained as a binary classifier to distinguish between expert demonstrations ?? E and policy generated demonstrations ?? .

The policy is trained to maximize the discriminative reward DISPLAYFORM2 consists of free-parameters as no structure is imposed on h ?? (??), and as mentioned in BID6 , the reward function r ?? (??) and function F are tied upto a constant (?? ??? 1)c, where c ??? R; thus the impact of F , the shaping term, on the recovered reward r is quite limited and therefore, the benefits of reward shaping are not fully realized.

Mutual information (MI), an information-theoretic measure, quantifies the dependency between two random variables.

In intrinsically-motivated reinforcement learning, a maximal of mutual information between a sequence of K actions a and the final state s reached after the execution of a, conditioned on current state s is often used as a measure of internal reward BID12 , known as Empowerment ??(s), i.e., DISPLAYFORM0 where p(s |a, s) is a K-step transition probability, w(a|s) is a distribution over a, and p(a, s |s) is a joint-distribution of K actions a and final state s 2 .

Intuitively, the empowerment ??(s) of a state s quantifies an extent to which an agent can influence its future.

Thus, maximizing empowerment induces an intrinsic motivation in the agent that enforces it to seek the states that have the highest number of future reachable states.

Empowerment, like value functions, is a potential function that has been previously used in reinforcement learning but its applications were limited to small-scale cases due to computational intractability of MI maximization in higher-dimensional problems.

Recently, however, a scalable method BID12 was proposed that learns the empowerment through the moreefficient maximization of variational lower bound, which has been shown to be equivalent to maximizing MI BID1 .

The lower bound was derived (for complete derivation see Appendix A.1) by representing MI in term of the difference in conditional entropies H(??) and utilizing the non-negativity property of KL-divergence, i.e., DISPLAYFORM1 where DISPLAYFORM2 is a variational distribution with parameters ?? and w ?? (??) is a distribution over actions with parameters ??.

Finally, the lower bound in Eqn.

5 is maximized under the constraint H(a|s) < ?? (prevents divergence, see BID12 ) to compute empowerment as follow: DISPLAYFORM3 where ?? is ?? dependent temperature term.

BID12 also applied the principles of Expectation-Maximization (EM) BID1 to learn empowerment, i.e., alternatively maximizing Eqn.

6 with respect to w ?? (a|s) and q ?? (a|s , s).

Given a set of training trajectories ?? , the maximization of Eqn.

6 w.r.t q ?? (??) is shown to be a supervised maximum log-likelihood problem whereas the maximization w.r.t w ?? (??) is determined through the functional derivative ???I/???w = 0 under the constraint a w(a|s) = 1.

The optimal w * that maximizes Eqn.

6 turns out to be 1 DISPLAYFORM4 , where Z(s) is a normalization term.

Substituting w * in Eqn.

6 showed that the empowerment ??(s) = 1 ?? log Z(s) (for full derivation, see Appendix A.2).Note that w * (a|s) is implicitly unnormalized as there is no direct mechanism for sampling actions or computing Z(s).

BID12 introduced an approximation w * (a|s) ??? log ??(a|s) + ??(s) where ??(a|s) is a normalized distribution which leaves the scalar function ??(s) to account for the normalization term log Z(s).

Finally, the parameters of policy ?? and scalar function ?? are optimized by minimizing the discrepancy, l I (s, a, s ), between the two approximations (log ??(a|s) + ??(s)) and ?? log q ?? (a|s , s)) through either absolute (p = 1) or squared error (p = 2), i.e., DISPLAYFORM5 3 EMPOWERED ADVERSARIAL INVERSE REINFORCEMENT LEARNINGWe present an inverse reinforcement learning algorithm that learns a robust, transferable reward function and policy from expert demonstrations.

Our proposed method comprises (i) an inverse model q ?? (a|s , s) that takes the current state s and the next state s to output a distribution over actions A that resulted in s to s transition, (ii) a reward r ?? (s, a), with parameters ??, that is a function of both state and action, (iii) an empowerment-based potential function ?? ?? (??) with parameters ?? that determines the reward-shaping function F = ???? ?? (s ) ??? ?? ?? (s) and also regularizes the policy update, and (iv) a policy model ?? ?? (a|s) that outputs a distribution over actions given the current state s. All these models are trained simultaneously based on the objective functions described in the following sections to recover optimal policies and generalizable reward functions concurrently.3.1 INVERSE MODEL q ?? (a|s, s ) OPTIMIZATION As mentioned in Section 2.3, learning the inverse model q ?? (a|s, s ) is a maximum log-likelihood supervised learning problem.

Therefore, given a set of trajectories ?? ??? ??, where a single trajectory is a sequence states and actions, i.e., ?? i = {s 0 , a 0 , ?? ?? ?? , s T , a T } i , the inverse model q ?? (a|s , s) is trained to minimize the mean-square error between its predicted action q(a|s , s) and the action a taken according to the generated trajectory ?? , i.e., DISPLAYFORM6 Empowerment will be expressed in terms of normalization function Z(s) of optimal w * (a|s), i.e., DISPLAYFORM7 .

Therefore, the estimation of empowerment ?? ?? (s) is approximated by minimizing the loss function l I (s, a, s ), presented in Eqn.

7, w.r.t parameters ??, and the inputs (s, a, s ) are sampled from the policy-generated trajectories ?? .

To train the reward function, we first compute the discriminator as follow: DISPLAYFORM0 where r ?? (s, a) is the reward function to be learned with parameters ??.

We also maintain the target ?? and learning ?? parameters of the empowerment-based potential function.

The target parameters ?? are a replica of ?? except that the target parameters ?? are updated to learning parameters ?? after every n training epochs.

Note that keeping a stationary target ?? ?? stabilizes the learning as also mentioned in BID11 .

Finally, the discriminator/reward function parameters ?? are trained via binary logistic regression to discriminate between expert ?? E and generated ?? trajectories, i.e., DISPLAYFORM1 3.4 POLICY OPTIMIZATION POLICY ?? ?? (a|s)We train our policy ?? ?? (a|s) to maximize the discriminative rewardr(s, a, s ) = log(D(s, a, s ) ??? log(1 ??? D(s, a, s ))) and to minimize the loss function l I (s, a, s ) = ?? log q ?? (a|s, s ) ??? (log ?? ?? (a|s) + ?? ?? (s)) p which accounts for empowerment regularization.

Hence, the overall policy training objective is: DISPLAYFORM2 where policy parameters ?? are updated using any policy optimization method such as TRPO BID21 or an approximated step such as PPO BID22 .Algorithm 1 outlines the overall training procedure to train all function approximators simultaneously.

Note that the expert samples ?? E are seen by the discriminator only, whereas all other models are trained using the policy generated samples ?? .

Furthermore, the discriminating rewardr(s, a, s ) boils down to the following expression (Appendix B.1): DISPLAYFORM3 where f (s, a, s ) = r ?? (s, a) + ???? ?? (s ) ??? ?? ?? (s).

Thus, an alternative way to express our policy training objective is E ?? [log ?? ?? (a|s)r ?? (s, a, s )], where r ?? (s, a, s ) =r(s, a, s ) ??? ?? I l I (s, a, s ), DISPLAYFORM4 Update ?? i to ?? i+1 using natural gradient update rule (i.e., TRPO/PPO) with the gradient: DISPLAYFORM5 After every n epochs sync ?? with ?? Fig (b) represents a problem where environment structure is modified during testing, i.e., a reward learned on a maze with left-passage is transferred to a maze with right-passage to the goal (green).which would undoubtedly yield the same results as Eqn.

11, i.e., maximize the discriminative reward and minimize the loss l I .

The analysis of this alternative expression is given in Appendix B to highlight that our policy update rule is equivalent to MaxEnt-IRL policy objective BID4 except that it also maximizes the empowerment, i.e., DISPLAYFORM6 where, ?? and ?? are hyperparameters, and??(??) is the entropy-regularization term depending on ??(??) and q(??).

Hence, our policy is regularized by the empowerment which induces generalized behavior rather than locally overfitting to the limited expert demonstrations.

Our proposed method, EAIRL, learns both reward and policy from expert demonstrations.

Thus, for comparison, we evaluate our method against both state-of-the-art policy and reward learning techniques on several control tasks in OpenAI Gym.

In case of policy learning, we compare our method against GAIL, GAN-GCL, AIRL with state-only reward, denoted as AIRL(s), and an augmented version of AIRL we implemented for the purposes of comparison that has state-action reward, denoted as AIRL(s, a).

In reward learning, we only compare our method against AIRL(s) and AIRL(s, a) as GAIL does not recover rewards, and GAN-GCL is shown to exhibit inferior performance than AIRL BID6 .

Furthermore, in the comparisons, we also include the expert The performance of policies obtained from maximizing the learned rewards in the transfer learning problems.

It can be seen that our method performs significantly better than AIRL BID6 and exhibits expert-like performance in all five randomly-seeded trials which imply that our method learns near-optimal, transferable reward functions.performances which represents a policy learned by optimizing a ground-truth reward using TRPO BID21 .

The performance of different methods are evaluated in term of mean and standard deviation of total rewards accumulated (denoted as score) by an agent during the trial, and for each experiment, we run five randomly-seeded trials.

To evaluate the learned rewards, we consider a transfer learning problem in which the testing environments are made to be different from the training environments.

More precisely, the rewards learned via IRL in the training environments are used to re-optimize a new policy in the testing environment using standard RL.

We consider two test cases shown in the FIG0 .In the first test case, as shown in FIG0 , we modify the agent itself during testing.

We trained a reward function to make a standard quadruped ant to run forward.

During testing, we disabled the front two legs (indicated in red) of the ant (crippled-ant), and the learned reward is used to reoptimize the policy to make a crippled-ant move forward.

Note that the crippled-ant cannot move sideways (Appendix C.1).

Therefore, the agent has to change the gait to run forward.

In the second test case, shown in FIG0 , we change the environment structure.

The agent learns to navigate a 2D point-mass to the goal region in a simple maze.

We re-position the maze central-wall during testing so that the agent has to take a different path, compared to the training environment, to reach the target (Appendix C.2).

TAB0 summarizes the means and standard deviations of the scores over five trials.

It can be seen that our method recovers near-optimal reward functions as the policy scores almost reach the expert scores in all five trials even after transfering to unseen testing environments.

Furthermore, our method performs significantly better than both

Next, we considered the performance of the learned policy specifically for an imitation learning problem in various control tasks.

The tasks, shown in FIG3 , include (i) making a 2D halfcheetah robot to run forward, (ii) making a 3D quadruped robot (ant) to move forward, (iii) making a 2D swimmer to swim, and (iv) keeping a friction less pendulum to stand vertically up.

For each algorithm, we provided 20 expert demonstrations generated by a policy trained on a ground-truth reward using TRPO BID21 .

TAB2 presents the means and standard deviations of policy learning performance scores, over the five different trials.

It can be seen that EAIRL, AIRL(s, a) and GAIL demonstrate similar performance and successfully learn to imitate the expert policy, whereas AIRL(s) and GAN-GCL fails to recover a policy.

This section highlights the importance of empowerment-regularized MaxEnt-IRL and modeling rewards as a function of both state and action rather than restricting to state-only formulation on learning rewards and policies from expert demonstrations.

In the scalable MaxEnt-IRL framework BID4 BID6 , the normalization term is approximated by importance sampling where the importance-sampler/policy is trained to minimize the KL-divergence from the distribution over expert trajectories.

However, merely minimizing the divergence between expert demonstrations and policy-generated samples leads to localized policy behavior which hinders learning generalized reward functions.

In our proposed work, we regularize the policy update with empowerment i.e., we update our policy to reduce the divergence from expert data distribution as well as to maximize the empowerment (Eqn.12).

The proposed regularization prevents premature convergence to local behavior which leads to robust state-action based rewards learning.

Furthermore, empowerment quantifies the extent to which an agent can control/influence its environment in the given state.

Thus the agent takes an action a on observing a state s such that it has maximum control/influence over the environment upon ending up in the future state s .Our experimentation also shows the importance of modeling discriminator/reward functions as a function of both state and action in reward and policy learning under GANs framework.

The re-ward learning results show that state-only rewards (AIRL(s)) does not recover the action dependent terms of the ground-truth reward function that penalizes high torques.

Therefore, the agent shows aggressive behavior and sometimes flips over after few steps (see the accompanying video), which is also the reason that crippled-ant trained with AIRL's disentangled reward function reaches only the half-way to expert scores as shown in TAB0 .

Therefore, the reward formulation as a function of both states and actions is crucial to learning action-dependent terms required in most real-world applications, including any autonomous driving, robot locomotion or manipulation task where large torque magnitudes are discouraged or are dangerous.

The policy learning results further validate the importance of the state-action reward formulation.

TAB2 shows that methods with state-action reward/discriminator formulation can successfully recover expert-like policies.

Hence, our empirical results show that it is crucial to model reward/discriminator as a function of state-action as otherwise, adversarial imitation learning fails to learn ground-truth rewards and expert-like policies from expert data.

We present an approach to adversarial reward and policy learning from expert demonstrations by regularizing the maximum-entropy inverse reinforcement learning through empowerment.

Our method learns the empowerment through variational information maximization in parallel to learning the reward and policy.

We show that our policy is trained to imitate the expert behavior as well to maximize the empowerment of the agent over the environment.

The proposed regularization prevents premature convergence to local behavior and leads to a generalized policy that in turn guides the reward-learning process to recover near-optimal reward.

We show that our method successfully learns near-optimal rewards, policies, and performs significantly better than state-of-the-art IRL methods in both imitation learning and challenging transfer learning problems.

The learned rewards are shown to be transferable to environments that are dynamically or structurally different from training environments.

In our future work, we plan to extend our method to learn rewards and policies from diverse human/expert demonstrations as the proposed method assumes that a single expert generates the training data.

Another exciting direction would be to build an algorithm that learns from sub-optimal demonstrations that contains both optimal and non-optimal behaviors.

For completeness, we present a derivation of presenting mutual information (MI) as variational lower bound and maximization of lower bound to learn empowerment.

As mentioned in section 2.3, the variational lower bound representation of MI is computed by defining MI as a difference in conditional entropies, and the derivation is formalized as follow.

??? ???E w(a|s) log w(a|s) + E p(s |a,s)w(a|s) [log q(a|s , s)]

DISPLAYFORM0

The empowerment is a maximal of MI and it can be formalized as follow by exploiting the variational lower bound formulation (for details see BID12 ).

w,q E p(s |a,s)w(a|s) [??? 1 ?? log w(a|s) + log q(a|s , s)]As mentioned in section 2.3, given a training trajectories, the maximization of Eqn.

13 w.r.t inverse model q(a|s , s) is a supervised maximum log-likelihood problem.

The maximization of Eqn.

13 w.r.t w(a|s) is derived through a functional derivative ???I w,q /???w = 0 under the constraint a w(a|s) = 1.

For simplicity, we consider discrete state and action spaces, and the derivation is as follow: By using the constraint a w(a|s) = 1, it can be shown that the optimal solution w * (a|s) = 1 Z(s) exp(u(s, a)), where u(s, a) = ??E p(s |a,s) [log q(a|s , s)] and Z(s) = a u(s, a).

This solution maximizes the lower bound since ??? 2 I w (s)/???w 2 = ??? a 1 w(a|s) < 0.

DISPLAYFORM0

In this section we derive the Empowerment-regularized formulation of maximum entropy IRL.

Let ?? be a trajectory sampled from expert demonstrations D and p ?? (?? ) ??? p(s 0 )?? T ???1 t=0 p(s t+1 |s t , a t ) exp r ?? (st,at) be a distribution over ?? .

As mentioned in Section 2, the IRL objective is to maximize the likelihood: DISPLAYFORM0 Furthermore, as derived in BID6 , the gradient of above equation w.r.t ?? can be written as: DISPLAYFORM1 where r ?? (??) is a parametrized reward to be learned, and p ??,t = s t =t,a t =t p ?? (?? ) denotes marginalization of state-action at time t. Since, it is unfeasible to draw samples from p ?? , BID4 proposed to train an importance sampling distribution ??(?? ) whose varience is reduced by defining ??(?? ) as a mixture of polices, i.e., ??(a|s) = 1 2 (??(a|s) +p(a|s)), wherep is a rough density estimate over demonstrations.

Thus the above gradient becomes: DISPLAYFORM2 We train our importance-sampler/policy ?? to maximize the empowerment ??(??) for generalization and to reduce divergence from true distribution by minimizing DISPLAYFORM3 , the matching terms of ??(?? ) and p ?? (?? ) cancel out, resulting into entropy-regularized policy update.

Furthermore, as we also include the empowerment ??(??) in the policy update to be maximized, hence the overall objective becomes: DISPLAYFORM4 Our discriminator is trained to minimize cross entropy loss as mention in Eqn.

10, and for the proposed structure of our discriminator Eqn.

9, it can be shown that the discriminator's gradient w.r.t its parameters turns out to be equal to Equation 14 (for more details, see BID6 ).

On the other hand, our policy training objective is DISPLAYFORM5 In the next section, we show that the above policy training objective is equivalent to Equation 15.

We train our policy to maximize the discriminative rewardr(s, a, s ) = log(D(s, a, s ) ??? log(1 ??? D(s, a, s ))) and minimize the information-theoretic loss function l I (s, a, s ).

The discriminative rewardr(s, a, s ) simplifies to: DISPLAYFORM0 where f (s, a, s ) = r(s, a) + ????(s ) ??? ??(s).

The entropy-regularization is usually scaled by the hyperparameter, let say ?? h ??? R, thusr(s, a, s ) = f (s, a, s ) ??? ?? h log ??(a|s).

Hence, assuming single-sample (s, a, s ), absolute-error for l I (s, a, s ) = | log q ?? (a|s, s ) ??? (log ??(a|s) + ??(s))|, and l i > 0, the policy is trained to maximize following: DISPLAYFORM1 = r(s, a) + ????(s ) ??? ??(s) ??? ?? h log ??(a|s) ??? log q(a|s, s ) + log ??(a|s) + ??(s) = r(s, a) + ????(s ) ??? ?? h log ??(a|s) ??? log q(a|s, s ) + log ??(a|s)Note that, the potential function ??(s) cancels out and we scale the leftover terms of l I with a hyperparameter ?? I .

Hence, the above equation becomes: r ?? (s, a, s ) = r(s, a, s ) + ????(s ) + (?? I ??? ?? h ) log ??(a|s) ???

?? I log q(a|s, s )We combine the log terms together as: r ?? (s, a, s ) = r(s, a) + ?? I ??(s ) + ????(??)where ?? is a hyperparameter, and??(??) is an entropy regularization term depending on q(a|s, s ) and ??(a|s).

Therefore, it can be seen that the Eqn.

17 is equivalent/approximation to Eqn.

15.

The following figures show the difference between the path profiles of standard and crippled Ant.

It can be seen that the standard Ant can move sideways whereas the crippled ant has to rotate in order to move forward.

The following figures show the path profiles of a 2D point-mass agent to reach the target in training and testing environment.

It can be seen that in the testing environment the agent has to take the opposite route compared to the training environment to reach the target.

We use two-layer ReLU network with 32 units in each layer for the potential function h ?? (??) and ?? ?? (??), reward function r ?? (??), discriminators of GAIL and GAN-GCL.

Furthermore, policy ?? ?? (??) of Figure 5 : The top and bottom rows show the path followed by a 2D point-mass agent (yellow) to reach the target (green) in training and testing environment, respectively.all presented models and the inverse model q ?? (??) of EAIRL are presented by two-layer RELU network with 32 units in each layer, where the network's output parametrizes the Gaussian distribution, i.e., we assume a Gaussian policy.

For all experiments, we use the temperature term ?? = 1.

We evaluated both mean-squared and absolute error forms of l I (s, a, s ) and found that both lead to similar performance in reward and policy learning.

We set entropy regularization weight to 0.1 and 0.001 for reward and policy learning, respectively.

The hyperparameter ?? I was set to 1.0 for reward learning and 0.001 for policy learning.

The target parameters of the empowerment-based potential function ?? ?? (??) were updated every 5 and 2 epochs during reward and policy learning respectively.

Although reward learning hyperparameters are also applicable to policy learning, we decrease the magnitude of entropy and information regularizers during policy learning to speed up the policy convergence to optimal values.

Furthermore, we set the batch size to 2000-and 20000-steps per TRPO update for the pendulum and remaining environments, respectively.

For the methods BID6 BID8 presented for comparison, we use their suggested hyperparameters.

We also use policy samples from previous 20 iterations as negative data to train the discriminator of all IRL methods presented in this paper to prevent the parametrized reward functions from overfitting the current policy samples.

@highlight

Our method introduces the empowerment-regularized maximum-entropy inverse reinforcement learning to learn near-optimal rewards and policies from expert demonstrations.
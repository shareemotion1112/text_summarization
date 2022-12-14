Building upon the recent success of deep reinforcement learning methods, we investigate the possibility of on-policy reinforcement learning improvement by reusing the data from several consecutive policies.

On-policy methods bring many benefits, such as ability to evaluate each resulting policy.

However, they usually discard all the information about the policies which existed before.

In this work, we propose adaptation of the replay buffer concept, borrowed from the off-policy learning setting, to the on-policy algorithms.

To achieve this, the proposed algorithm generalises the Q-, value and advantage functions for data from multiple policies.

The method uses trust region optimisation, while avoiding some of the common problems of the algorithms such as TRPO or ACKTR: it uses hyperparameters to replace the trust region selection heuristics, as well as  the trainable covariance matrix instead of the fixed one.

In many cases, the method not only improves the results comparing to the state-of-the-art trust region on-policy learning algorithms such as ACKTR and TRPO, but also with respect to their off-policy counterpart DDPG.

The past few years have been marked by active development of reinforcement learning methods.

Although the mathematical foundations of reinforcement learning have been known long before BID23 , starting from 2013, the novel deep learning techniques allowed to solve vision based discrete control tasks such as Atari 2600 games BID15 as well as continuous control problems BID12 .

Many of the leading state-of-the-art reinforcement learning methods share the actor-critic architecture BID5 .

Actorcritic methods separate the actor, providing a policy, and the critic, providing an approximation for the expected discounted cumulative reward or some derived quantities such as advantage functions BID2 .

However, despite improvements, state-of-the-art reinforcement learning still suffers from poor sample efficiency and extensive parameterisation.

For most real-world applications, in contrast to simulations, there is a need to learn in real time and over a limited training period, while minimising any risk that would cause damage to the actor or the environment.

Reinforcement learning algorithms can be divided into two groups: on-policy and off-policy learning.

On-policy approaches (e. g., SARSA BID18 , ACKTR BID28 ) evaluate the target policy by assuming that future actions will be chosen according to it, hence the exploration strategy must be incorporated as a part of the policy.

Off-policy methods (e. g., Qlearning BID27 , DDPG BID12 ) separate the exploration strategy, which modifies the policy to explore different states, from the target policy.

The off-policy methods commonly use the concept of replay buffers to memorise the outcomes of the previous policies and therefore exploit the information accumulated through the previous iterations BID13 .

BID15 combined this experience replay mechanism with Deep Q-Networks (DQN), demonstrating end-to-end learning on Atari 2600 games.

One limitation of DQN is that it can only operate on discrete action spaces.

BID12 proposed an extension of DQN to handle continuous action spaces based on the Deep Deterministic Policy Gradient (DDPG).

There, exponential smoothing of the target actor and critic weights has been introduced to ensure stability of the rewards and critic predictions over the subsequent iterations.

In order to improve the variance of policy gradients, BID20 proposed a Generalised Advantage Function.

combined this advantage function learning with a parallelisation of exploration using differently trained actors in their Asynchronous Advantage Actor Critic model (A3C); however, BID26 demonstrated that such parallelisation may also have negative impact on sample efficiency.

Although some work has been performed on improvement of exploratory strategies for reinforcement learning BID8 , but it still does not solve the fundamental restriction of inability to evaluate the actual policy, neither it removes the necessity to provide a separate exploratory strategy as a separate part of the method.

In contrast to those, state-of-the-art on-policy methods have many attractive properties: they are able to evaluate exactly the resulting policy with no need to provide a separate exploration strategy.

However, they suffer from poor sample efficiency, to a larger extent than off-policy reinforcement learning.

TRPO method BID19 has introduced trust region policy optimisation to explicitly control the speed of policy evolution of Gaussian policies over time, expressed in a form of Kullback-Leibler divergence, during the training process.

Nevertheless, the original TRPO method suffered from poor sample efficiency in comparison to off-policy methods such as DDPG.

One way to solve this issue is by replacing the first order gradient descent methods, standard for deep learning, with second order natural gradient (Amari, 1998).

BID28 used a Kroneckerfactored Approximate Curvature (K-FAC) optimiser BID14 in their ACKTR method.

PPO method proposes a number of modifications to the TRPO scheme, including changing the objective function formulation and clipping the gradients.

BID26 proposed another approach in their ACER algorithm: in this method, the target network is still maintained in the off-policy way, similar to DDPG BID12 , while the trust region constraint is built upon the difference between the current and the target network.

Related to our approach, recently a group of methods has appeared in an attempt to get the benefits of both groups of methods.

BID7 propose interpolated policy gradient, which uses the weighted sum of both stochastic BID24 and deterministic policy gradient BID22 .

BID17 propose an off-policy trust region method, Trust-PCL, which exploits off-policy data within the trust regions optimisation framework, while maintaining stability of optimisation by using relative entropy regularisation.

While it is a common practice to use replay buffers for the off-policy reinforcement learning, their existing concept is not used in combination with the existing on-policy scenarios, which results in discarding all policies but the last.

Furthermore, many on-policy methods, such as TRPO BID19 , rely on stochastic policy gradient BID24 , which is restricted by stationarity assumptions, in a contrast to those based on deterministic policy gradient BID22 , like DDPG BID12 .

In this article, we describe a novel reinforcement learning algorithm, allowing the joint use of replay buffers with trust region optimisation and leading to sample efficiency improvement.

The contributions of the paper are given as follows:1.

a reinforcement learning method, enabling replay buffer concept along with on-policy data; 2.

theoretical insights into the replay buffer usage within the on-policy setting are discussed; 3.

we show that, unlike the state-of-the-art methods as ACKTR BID28 , PPO (Schulman et al., 2017) and TRPO BID19 , a single non-adaptive set of hyperparameters such as the trust region radius is sufficient for achieving better performance on a number of reinforcement learning tasks.

As we are committed to make sure the experiments in our paper are repeatable and to further ensure their acceptance by the community, we will release our source code shortly after the publication.

Consider an agent, interacting with the environment by responding to the states s t , t ??? 0, from the state space S, which are assumed to be also the observations, with actions a t from the action space A chosen by the policy distribution ?? ?? (??|s t ), where ?? are the parameters of the policy.

The initial state distribution is ?? 0 : S ??? R. Every time the agent produces an action, the environment gives back a reward r(s t , a t ) ??? R, which serves as a feedback on how good the action choice was and switches to the next state s t+1 according to the transitional probability P (s t+1 |s t , a t ).

Altogether, it can be formalised as an infinite horizon ??-discounted Markov Decision Process (S, A, P, r, ?? 0 , ??), ?? ??? [0, 1) BID28 BID19 .

The expected discounted return BID3 ) is defined as per BID19 : DISPLAYFORM0 The advantage function A ?? BID2 , the value function V ?? and the Q-function Q ?? are defined as per ; BID19 : DISPLAYFORM1 DISPLAYFORM2 DISPLAYFORM3 In all above definitions s 0 ??? ?? 0 (s 0 ), a t ??? ??(a t |s t ), s t+1 ??? P (s t+1 |s t , a t ), and the policy ?? = ?? ?? is defined by its parameters ??.

A straightforward approach for learning a policy is to perform unconstrained maximisation ??(?? ?? ) with respect to the policy parameters ??.

However, for the state-of-the-art iterative gradient-based optimisation methods, this approach would lead to unpredictable and uncontrolled changes in the policy, which would impede efficient exploration.

Furthermore, in practice the exact values of ??(?? ?? ) are unknown, and the quality of its estimates depends on approximators which tend to be correct only in the vicinity of parameters of observed policies. (2015a) mention that in practice the algorithm's convergence rate and the complexity of maximum KL divergence computations makes it impractical to apply this method directly.

Therefore, they proposed to replace the unconstrained optimisation with a similar constrained optimisation problem, the Trust Region Policy Optimisation (TRPO) problem: DISPLAYFORM0 where D KL is the KL divergence between the old and the new policy ?? ?? old and ?? ?? respectively, and ?? is the trust region radius.

Despite this improvement, it needs some further enhancements to solve this problem efficiently, as we will elaborate in the next section.

Many of the state-of-the-art trust region based methods, including TRPO BID19 and ACKTR BID28 , use second order natural gradient based actor-critic optimisation BID1 BID10 .

The motivation behind it is to eliminate the issue that gradient descent loss, calculated as the Euclidean norm, is dependent on parametrisation.

For this purpose, the Fisher information matrix is used, which is, as it follows from Amari (1998) and BID10 , normalises per-parameter changes in the objective function.

In the context of actor-critic optimisation it can be written as BID28 BID10 , where p(?? ) is the trajectory distribution p(s 0 ) T t=0 ??(a t |s t )p(s t+1 |s t , a t ): DISPLAYFORM0 However, the computation of the Fisher matrix is intractable in practice due to the large number of parameters involved; therefore, there is a need to resort to approximations, such as the Kroneckerfactored approximate curvature (K-FAC) method BID14 , which has been first proposed for ACKTR in BID28 .

In the proposed method, as it is detailed in Algorithm 1, this optimisation method is used for optimising the policy.

While the original trust regions optimisation method can only use the samples from the very last policy, discarding the potentially useful information from the previous ones, we make use of samples over several consecutive policies.

The rest of the section contains definition of the proposed replay buffer concept adaptation, and then formulation and discussion of the proposed algorithm.3.1 USAGE OF REPLAY BUFFERS BID15 suggested to use replay buffers for DQN to improve stability of learning, which then has been extended to other off-policy methods such as DDPG BID12 .

The concept has not been applied to on-policy methods like TRPO BID19 or ACKTR BID28 , which do not use of previous data generated by other policies.

Although based on trust regions optimisation, ACER BID26 uses replay buffers for its off-policy part.

In this paper, we propose a different concept of the replay buffers, which combines the on-policy data with data from several previous policies, to avoid the restrictions of policy distribution stationarity for stochastic policy gradient BID24 .

Such replay buffers are used for storing simulations from several policies at the same time, which are then utilised in the method, built upon generalised value and advantage functions, accommodating data from these policies.

The following definitions are necessary for the formalisation of the proposed algorithm and theorems.

We define a generalised Q-function for multiple policies {?? 1 , . . .

, ?? n , . . .

, ?? N } as DISPLAYFORM0 DISPLAYFORM1 |s n t , a n t ), a n t ??? ?? n (a n t |s n t ).(9) We also define the generalised value function and the generalised advantage function as DISPLAYFORM2 DISPLAYFORM3 DISPLAYFORM4 P (s ??? x, k, ??), as in BID24 , is the probability of transition from the state s to the state x in k steps using policy ??.

Theorem 1.

For the set of policies {?? 1 , . . .

, ?? N } the following equality will be true for the gradient: DISPLAYFORM5 where ?? are the joint parameters of all policies {?? n } and b ??n (s) is a bias function for the policy.

The proof of Theorem 1 is given in Appendix B. Applying a particular case of the bias function b ??n (s) = ???V ?? (s) and using the likelihood ratio transformation, one can get DISPLAYFORM6

The proposed approach is summarised in Algorithm 1.

The replay buffer R p contains data collected from several subsequent policies.

The size of this buffer is RBP CAPACITY. , increase i by the total number of timesteps in all new paths.

{Stage 2} Put recorded paths into the policy paths replay buffer R p ??? P .

{Stage 3}

For every path in R p compute the targets for the value function regression using equation FORMULA0 .

?? = Update the value function estimator parameters {Stage 4} For every path in R p , estimate the advantage function using Equation (23).{Stage 5} Update parameters of the policy ?? for N ITER PL UPDATE iterations using the gradient from Equation FORMULA1 and a barrier function defined in Equation (26). end while During Stage 1, the data are collected for every path until the termination state is received, but at least TIMESTEPS PER BATCH steps in total for all paths.

The policy actions are assumed to be sampled from the Gaussian distribution, with the mean values predicted by the policy estimator along with the covariance matrix diagonal.

The covariance matrix output was inspired, although the idea is different, by the EPG paper BID4 .At Stage 2, the obtained data for every policy are saved in the policy replay buffer R p .At Stage 3, the regression of the value function is trained using Adam optimiser BID11 with step size VF STEP SIZE for N ITER VF UPDATE iterations.

For this regression, the sum-of-squares loss function is used.

The value function target values are computed for every state s t for every policy in the replay buffer using the actual sampled policy values, where t max is the maximum policy step index:V DISPLAYFORM0 During Stage 4, we perform the advantage function estimation.

BID20 proposed the Generalised Advantage Estimator for the advantage function A ?? (s t , a t ) as follows: DISPLAYFORM1 DISPLAYFORM2 Here k > 0 is a cut-off value, defined by the length of the sequence of occured states and actions within the MDP, ?? ??? [0, 1] is an estimator parameter, and??? ?? (s t ) is the approximation for the value function V ?? (s t ), with the approximation targets defined in Equation (15).

As proved in BID20 , after rearrangement this would result in the generalised advantage function estimator DISPLAYFORM3 For the proposed advantage function (see Equation 11), the estimator could be defined similarly to BID20 as DISPLAYFORM4 DISPLAYFORM5 However, it would mean the estimation of multiple value functions, which diminishes the replay buffer idea.

To avoid it, we modify this estimator for the proposed advantage function as DISPLAYFORM6 Theorem 2.

The difference between the estimators FORMULA1 and FORMULA1 is DISPLAYFORM7 The proof of Theorem 2 is given in Appendix C.

It shows that the difference between two estimators is dependent of the difference in the conventional and the generalised value functions; given the continuous value function approximator it reveals that the closer are the policies, within a few trust regions radii, the smaller will be the bias.

During Stage 5, the policy function is approximated, using the K-FAC optimiser BID14 with the constant step size PL STEP SIZE.

As one can see from the description, and differently from ACKTR, we do not use any adaptation of the trust region radius and/or optimisation algorithm parameters.

Also, the output parameters include the diagonal of the (diagonal) policy covariance matrix.

The elements of the covariance matrix, for the purpose of efficient optimisation, are restricted to universal minimum and maximum values MIN COV EL and MAX COV EL.As an extention from BID20 and following Theorem 1 with the substitution of likelihood ratio, the policy gradient estimation is defined as DISPLAYFORM8 To practically implement this gradient, we substitute the parameters ?? ?? , derived from the latest policy for the replay buffer, instead of joint ?? parameters assuming that the parameters would not deviate far from each other due to the trust region restrictions; it is still possible to calculate the estimation of?? ??n (s n t , a n t ) for each policy using Equation (23) as these policies are observed.

For the constrained optimisation we add the linear barrier function to the function ??(??): DISPLAYFORM9 where ?? > 0 is a barrier function parameter and ?? old are the parameters of the policy on the previous iteration.

Besides of removing the necessity of heuristical estimation of the optimisation parameters, it also conforms with the theoretical prepositions shown in and, while our approach is proposed independently, pursues the similar ideas of using actual constrained optimisation method instead of changing the gradient step size parameters as per BID19 .The networks' architectures correspond to OpenAI Baselines ACKTR implementation ,which has been implemented by the ACKTR authors BID28 .

The only departure from the proposed architecture is the diagonal covariance matrix outputs, which are present, in addition to the mean output, in the policy network.

In order to provide the experimental evidence for the method, we have compared it with the on-policy ACKTR BID28 , PPO and TRPO BID19 ) methods, as well as with the off-policy DDPG BID12 method on the MuJoCo BID25 robotic simulations.

The technical implementation is described in Appendix A. BID25 : comparison with TRPO BID19 , ACKTR BID28 and PPO BID25 : comparison between the proposed algorithm and DDPG BID12 In contrast to those methods, the method shows that the adaptive values for trust region radius can be advantageously replaced by a fixed value in a combination with the trainable policy distribution covariance matrix, thus reducing the number of necessary hyperparameters.

The results for ACKTR for the tasks HumanoidStandup, Striker and Thrower are not included as the baseline ACKTR implementation diverged at the first iterations with the predefined parameterisation.

PPO results are obtained from baselines implementation PPO1 .

Figure 2 compares results for different replay buffer sizes; the size of the replay buffers reflects the number of policies in it and not actions (i.e. buffer size 3 means data from three successive policies in the replay buffer).

We see that in most of the cases, the use of replay buffers show performance improvement against those with replay buffer size 1 (i.e., no replay buffer with only the current policy used for policy gradient); substantial improvements can be seen for HumanoidStandup task.

Figure 3 shows the performance comparison with the DDPG method BID12 .

In all the tasks except HalfCheetah and Humanoid, the proposed method outperforms DDPG.

For HalfCheetah, the versions with a replay buffer marginally overcomes the one without.

It is also remarkable that the method demonstrates stable performance on the tasks HumanoidStandup, Pusher, Striker and Thrower, on which DDPG failed (and these tasks were not included into the DDPG article).

The paper combines replay buffers and on-policy data for reinforcement learning.

Experimental results on various tasks from the MuJoCo suite BID25 show significant improvements compared to the state of the art.

Moreover, we proposed a replacement of the heuristically calculated trust region parameters, to a single fixed hyperparameter, which also reduces the computational expences, and a trainable diagonal covariance matrix.

The proposed approach opens the door to using a combination of replay buffers and trust regions for reinforcement learning problems.

While it is formulated for continuous tasks, it is possible to reuse the same ideas for discrete reinforcement learning tasks, such as ATARI games.

The parameters of Algorithm 1, used in the experiment, are given in Table 1 ; the parameters were initially set, where possible, to the ones taken from the state-of-the-art trust region approach implementation BID28 , and then some of them have been changed based on the experimental evidence.

As the underlying numerical optimisation algorithms are out of the scope of the paper, the parameters of K-FAC optimiser from have been used for the experiments; for the Adam algorithm BID11 , the default parameters from Tensorflow BID0 implementation (?? 1 = 0.9, ?? 2 = 0.999, = 1 ?? 10 ???8 ) have been used.

The method has been implemented in Python 3 using Tensorflow BID0 as an extension of the OpenAI baselines package .

The neural network for the control experiments consists of two fully connected layers, containing 64 neurons each, following the OpenAI ACKTR network implementation Proof.

Extending the derivation from Sutton et al. FORMULA1 , one can see that: DISPLAYFORM0 Then, Proof.

The difference between the two k-th estimators is given as DISPLAYFORM1 By substituting this into the GAE estimator difference one can obtain ????? ??n (s t , a t ) = (1 ??? ??)(?????V 1 + ???? 2 ???V 2 + ?? 2 ?? 3 ???V 3 + . . .

DISPLAYFORM2

<|TLDR|>

@highlight

We investigate the theoretical and practical evidence of on-policy reinforcement learning improvement by reusing the data from several consecutive policies.
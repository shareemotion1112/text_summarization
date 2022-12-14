The goal of imitation learning (IL) is to enable a learner to imitate expert behavior given expert demonstrations.

Recently, generative adversarial imitation learning (GAIL) has shown significant progress on IL for complex continuous tasks.

However, GAIL and its extensions require a large number of environment interactions during training.

In real-world environments, the more an IL method requires the learner to interact with the environment for better imitation, the more training time it requires, and the more damage it causes to the environments and the learner itself.

We believe that IL algorithms could be more applicable to real-world problems if the number of interactions could be reduced.

In this paper, we propose a model-free IL algorithm for continuous control.

Our algorithm is made up mainly three changes to the existing adversarial imitation learning (AIL) methods – (a) adopting off-policy actor-critic (Off-PAC) algorithm to optimize the learner policy, (b) estimating the state-action value using off-policy samples without learning reward functions, and (c) representing the stochastic policy function so that its outputs are bounded.

Experimental results show that our algorithm achieves competitive results with GAIL while significantly reducing the environment interactions.

Recent advances in reinforcement learning (RL) have achieved super-human performance on several domains BID20 BID21 .

On most of such domains with the success of RL, the design of reward, that explains what agent's behavior is favorable, is obvious for humans.

Conversely, on domains where it is unclear how to design the reward, agents trained by RL algorithms often obtain poor policies and behave worse than what we expect them to do.

Imitation learning (IL) comes in such cases.

The goal of IL is to enable the learner to imitate expert behavior given the expert demonstrations without the reward signal.

We are interested in IL because we desire an algorithm that can be applied to real-world problems for which it is often hard to design the reward.

In addition, since it is generally hard to model a variety of real-world environments with an algorithm, and the state-action pairs in a vast majority of realworld applications such as robotics control can be naturally represented in continuous spaces, we focus on model-free IL for continuous control.

A wide variety of IL methods have been proposed in the last few decades.

The simplest IL method among those is behavioral cloning (BC) BID23 which learns an expert policy in a supervised fashion without environment interactions during training.

BC can be the first IL option when enough demonstration is available.

However, when only a limited number of demonstrations are available, BC often fails to imitate the expert behavior because of the problem which is referred to compounding error BID25 -inaccuracies compound over time and can lead the learner to encounter unseen states in the expert demonstrations.

Since it is often hard to obtain a large number of demonstrations in real-world environments, BC is often not the best choice for real-world IL scenarios.

Another widely used approach, which overcomes the compounding error problem, is Inverse Reinforcement Learning (IRL) BID27 BID22 BID0 BID33 .

Recently, BID15 have proposed generative adversarial imitation learning (GAIL) which is based on prior IRL works.

Since GAIL has achieved state-of-the-art performance on a variety of continuous control tasks, the adversarial IL (AIL) framework has become a popular choice for IL BID1 BID11 BID16 .

It is known that the AIL methods are more sample efficient than BC in terms of the expert demonstration.

However, as pointed out by BID15 , the existing AIL methods have sample complexity in terms of the environment interaction.

That is, even if enough demonstration is given by the expert before training the learner, the AIL methods require a large number of state-action pairs obtained through the interaction between the learner and the environment 1 .

The sample complexity keeps existing AIL from being employed to real-world applications for two reasons.

First, the more an AIL method requires the interactions, the more training time it requires.

Second, even if the expert safely demonstrated, the learner may have policies that damage the environments and the learner itself during training.

Hence, the more it performs the interactions, the more it raises the possibility of getting damaged.

For the real-world applications, we desire algorithms that can reduce the number of interactions while keeping the imitation capability satisfied as well as the existing AIL methods do.

The following three properties of the existing AIL methods which may cause the sample complexity in terms of the environment interactions:(a) Adopting on-policy RL methods which fundamentally have sample complexity in terms of the environment interactions.(b) Alternating three optimization processes -learning reward functions, value estimation with learned reward functions, and RL to update the learner policy using the estimated value.

In general, as the number of parameterized functions which are related to each other increases, the training progress may be unstable or slower, and thus more interactions may be performed during training.(c) Adopting Gaussian policy as the learner's stochastic policy, which has infinite support on a continuous action space.

In common IL settings, we observe action space of the expert policy from the demonstration where the expert action can take on values within a bounded (finite) interval.

As BID3 suggests, the policy which can select actions outside the bound may slow down the training progress and make the problem harder to solve, and thus more interactions may be performed during training.

In this paper, we propose an IL algorithm for continuous control to improve the sample complexity of the existing AIL methods.

Our algorithm is made up mainly three changes to the existing AIL methods as follows:(a) Adopting off-policy actor-critic (Off-PAC) algorithm BID5 to optimize the learner policy instead of on-policy RL algorithms.

Off-policy learning is commonly known as the promising approach to improve the complexity.(b) Estimating the state-action value using off-policy samples without learning reward functions instead of using on-policy samples with the learned reward functions.

Omitting the reward learning reduces functions to be optimized.

It is expected to make training progress stable and faster and thus reduce the number of interactions during training.(c) Representing the stochastic policy function of which outputs are bounded instead of adopting Gaussian policy.

Bounding action values may make the problem easier to solve and make the training faster, and thus reduce the number of interactions during training.

Experimental results show that our algorithm enables the learner to imitate the expert behavior as well as GAIL does while significantly reducing the environment interactions.

Ablation experimental results show that (a) adopting the off-policy scheme requires about 100 times fewer environment interactions to imitate the expert behavior than the one on-policy IL algorithms require, (b) omitting the reward learning makes the training stable and faster, and (c) bounding action values makes the training faster.

We consider a Markov Decision Process (MDP) which is defined as a tuple {S, A, T , R, d 0 , γ}, where S is a set of states, A is a set of possible actions agents can take, T : S×A×S → [0, 1] is a transition probability, R : S×A → R is a reward function, d 0 : S → [0, 1] is a distribution over initial states, and γ ∈ [0, 1) is a discount factor.

The agent's behavior is defined by a stochastic policy π : S×A → [0, 1] and Π denotes a set of the stochastic policies.

We denote S E ⊂ S and A E ⊂ A as sets of states and actions observed in the expert demonstration, and S π ⊂ S and A π ⊂ A as sets of those observed in rollouts following a policy π.

We will use π E , π θ , β ∈ Π to refer to the expert policy, the learner policy parameterized by θ, and a behavior policy, respectively.

Given a policy π, performance measure of π is defined as J (π, R) = E ∞ t=0 γ t R(s t , a t )|d 0 , T , π where s t ∈ S is a state that the agent receives at discrete time-step t, and a t ∈ A is an action taken by the agent after receiving s t .

The performance measure indicates expectation of the discounted return ∞ t=0 γ t R(s t , a t ) when the agent follows the policy π in the MDP.

Using discounted state visitation distribution denoted by ρ π (s) = ∞ t=0 γ t P(s t = s|d 0 , T , π) where P is a probability that the agent receives the state s at time-step t, the performance measure can be rewritten as J (π, R) = E s∼ρπ,a∼π R(s, a) .

The state-action value function for the agent following π is defined as DISPLAYFORM0 |T , π , and Q π,ν denotes its approximator parameterized by ν.

We briefly describe objectives of RL, IRL, and AIL below.

We refer the readers to BID15 for details.

The goal of RL is to find an optimal policy that maximizes the performance measure.

Given the reward function R, the objective of RL with parameterized stochastic policies π θ : S×A → [0, 1] is defined as follows: DISPLAYFORM0 (1) The goal of IRL is to find a reward function based on an assumption that the discounted returns earned by the expert behavior are greater than or equal to those earned by any non-experts behavior.

Technically, the objective of IRL is to find reward functions R ω : S × A → R parameterized by ω that satisfies J (π E , R ω ) ≥ J (π, R ω ) where π denotes the non-expert policy.

The existing AIL methods adopt max-margin IRL BID0 of which objective can be defined as follows: DISPLAYFORM1 2) The objective of AIL can be defined as a composition of the objectives (1) and (2) as follows: DISPLAYFORM2

The objective of Off-PAC to train the learner can be described as follows: DISPLAYFORM0 The learner policy is updated by taking the gradient of the state-action value.

BID5 proposed the gradient as follows: provided another formula of the gradient using "re-parameterization trick" in the case that the learner policy selects the action as a = π θ (s, z) with random variables z ∼ P z generated by a distribution P z : DISPLAYFORM1 DISPLAYFORM2 3 ALGORITHMAs mentioned in Section.1, our algorithm (a) adopts Off-PAC algorithms to train the learner policy, (b) estimates state-action value without learning the reward functions, and (c) represents the stochastic policy function so that its outputs are bounded.

In this section, we first introduce (b) in 3.1 and describe (c) in 3.2, then present how to incorporate (b) and (c) into (a) in 3.3.

In this subsection, we introduce a new IRL objective to learn the reward function in 3.1.1 and a new objective to learn the value function approximator in 3.1.2.

Then, we show that combining those objectives derives a novel objective to learn the value function approximator without reward learning in 3.1.3.

We define the parameterized reward function as R ω (s, a) = log r ω (s, a), with a function r ω : S×A → [0, 1] parameterized by ω.

r ω (s, a) represents a probability that the state-action pairs (s, a) belong to S E × A E .

In other words, r ω (s, a) explains how likely the expert executes the action a at the state s. With this reward, we can also define a Bernoulli distribution p ω : Π×S×A → [0, 1] such that p ω (π E |s, a) = r ω (s, a) for the expert policy π E and p ω (π|s, a) = 1 − r ω (s, a) for any other policies π ∈ Π \ {π E } which include π θ and β.

A nice property of this definition of the reward is that the discounted return for a trajectory {(s 0 , a 0 ), (s 1 , a 1 ), ...} can be written as a log likelihood with p ω (π E |s t , a t ): DISPLAYFORM0 Here, we assume Markov property in terms of p ω such that p ω (π E |s t , a t ) for t ≥ 1 is independent of p ω (π E |s t−u , a t−u ) for u ∈ {1, ..., t}. Under this assumption, the return naturally represents how likely a trajectory is the one the expert demonstrated.

The discount factor γ plays a role to make sure the return is finite as in standard RL.The IRL objective (2) can be said to aim at assigning r ω = 1 for state-action pairs (s, a) ∈ S E × A E and r ω = 0 for (s, a) ∈ S π × A π when the same definition of the reward R ω (s, a) = log r ω (s, a) is used.

Following this fashion easily leads to a problem where the return earned by the non-expert policy becomes −∞, since log r ω (s, a) = −∞ if r ω (s, a) = 0 and thus log DISPLAYFORM1 The existing AIL methods seem to mitigate this problem by trust region optimization for parameterized value function approximator BID29 , and it works somehow.

However, we think this problem should be got rid of in a fundamental way.

We propose a different approach to evaluate state-action pairs (s, a) ∈ S π × A π .

Intuitively, the learner does not know how the expert behaves in the states s ∈ S \ S E -that is, it is uncertain which actions the expert executes in the states the expert has not visited.

We thereby define a new IRL objective as follows: DISPLAYFORM2 where H denotes entropy of Bernoulli distribution such that: DISPLAYFORM3 Unlike the existing AIL methods, our IRL objective is to assign p ω (π E |s, a) = p ω (π|s, a) = 0.5 for (s, a) ∈ S π × A π .

This uncertainty p ω (π E |s, a) = 0.5 explicitly makes the return earned by the non-expert policy finite.

On the other hand, the objective is to assign r ω = 1 for (s, a) ∈ S E × A E as do the existing AIL methods.

The optimal solution for the objective (8) satisfies the assumption of IRL : DISPLAYFORM4 , even though the objective does not aim at discriminating between (s, a) ∈ S E × A E and (s, a) ∈ S π × A π ,

As we see in Equation FORMULA7 , the discounted return can be represented as a log likelihood.

Therefore, a value function approximator Q π θ following the learner policy π θ can be formed as a log probability.

We introduce a function q π θ ,ν : S×A → [0, 1] parameterized by ν to represent the approximator Q π θ ,ν as follows: DISPLAYFORM0 The optimal value function following a policy π satisfies the Bellman equation Q π (s t , a t ) = R(s t , a t ) + γE st+1∼T ,at+1∼π Q π (s t+1 , a t+1 ) .

Substituting π θ for π, log r ω (s t , a t ) for R(s t , a t ), and log q π θ ,ν (s t , a t ) for Q π (s t , a t ), the Bellman equation for the learner policy π θ can be written as follows: DISPLAYFORM1 We introduce additional Bernoulli distributions P ν : Π × S × A :→ [0, 1] and P ωνγ : Π × S × A × S × A :→ [0, 1] as follows: DISPLAYFORM2 Using P ν and P ωνγ , the loss to satisfy Equation FORMULA13 can be rewritten as follows: DISPLAYFORM3 We use Jensen's inequality with the concave property of logarithm in Equation FORMULA4 .

Now we see that the loss L(ω, ν, θ) is bounded by the log likelihood ratio between the two Bernoulli distributions P ν and P ωνγ , and L(ω, ν, θ) = 0 if P ν (π E |s t , a t ) = E st+1∼T ,at+1∼π θ P ωνγ (π E |s t , a t , s t+1 , a t+1 ) .In the end, learning the approximator Q π θ ,ν turns out to be matching the two Bernoulli distributions.

A natural way to measure the difference between two probability distributions is divergence.

We choose Jensen-Shannon (JS) divergence to measure the difference because we empirically found it works better, and thereby the objective to optimize Q π θ ,ν can be written as follows: DISPLAYFORM4 where D JS denotes JS divergence between two Bernoulli distributions.

Suppose the optimal reward function R ω * (s, a) = log r ω * (s, a) for the objective (8) can be obtained, the Bellman equation FORMULA13 can be rewritten as follows: DISPLAYFORM0 Recall that IRL objective (8) aims at assigning r ω * (s t , a t ) = 1 for (s t , a t ) ∈ S E × A E and r ω * (s t , a t ) = 0.5 for (s t , a t ) ∈ S π × A π where π ∈ Π \ {π E }.

Therefore, the objective FORMULA9 is rewritten as the following objective using the Bellman equation FORMULA13 : DISPLAYFORM1 Thus, r ω * can be obtained by the Bellman equation FORMULA6 as long as the solution for the objective (17) can be obtained.

We optimize q π θ ,ν (s t , a t ) in the same way of objective (15) as follows: DISPLAYFORM2 We use P γ ν instead of P ωνγ in objective (18) unlike the objective (15).

Thus, we omit reward learning that the existing AIL methods require, while learning q π θ ,ν (s t , a t ) to obtain r ω * .

Initialize time-step t = 0 and receive initial state s0 5:while not terminate condition do 6:Execute an action at = π θ (st, z) with z ∼ Pz and observe new state st+1 7:Store a state-action triplet (st, at, st+1) in B β .

8: DISPLAYFORM3 end while 10:for u = 1, t do 11:Sample mini-batches of triplets (s t ′ +1 ).

end for 16: end for

Recall that the aim of IL is to imitate the expert behavior.

It can be summarized that IL attempts to obtain a generative model the expert has over A conditioned on states in S. We see that the aim itself is equivalent to that of conditional generative adversarial networks (cGANs) BID19 .

The generator of cGANs can generate stochastic outputs of which range are bounded.

As mentioned in Section 1, bounding action values is expected to make the problem easier to solve and make the training faster.

In the end, we adopt the form of the conditional generator to represent the stochastic learner policy π θ (s, z).

The typical Gaussian policy and the proposed policy representations with neural networks are described in FIG0

Algorithm.1 shows the overview of our off-policy actor-critic imitation learning algorithm.

To learn the value function approximator Q π θ ,ν , we adopt a behavior policy β as π in the second term in objective (18) We employ a mixture of the past learner policies as β and a replay buffer B β BID20 ) to perform sampling s t ∼ ρ π , a t ∼ π and s t+1 ∼ T .

The buffer B β is a finite cache and stores the (s t , a t , s t+1 ) triplets in a first-in-first-out manner while the learner interacts with the environment.

The approximator Q π θ ,ν (s t , a t ) = log q π θ ,ν (s t , a t ) takes (−∞, 0].

With the approximator, using the gradient (5) to update the learner policy always punish (or ignore) the learner's actions.

Instead, we adopt the gradient (6) which directly uses Jacobian of Q π θ ,ν .As do off-policy RL methods such as BID20 and , we use the target value function approximator, of which parameters are updated to track ν , to optimize Q π θ ,ν .

We update Q π θ ,ν and π θ at the end of each episode rather than following each step of interaction.

In recent years, the connection between generative adversarial networks (GAN) BID10 and IL has been pointed out BID15 BID6 .

BID15 show that IRL is a dual problem of RL which can be deemed as a problem to match the learner's occupancy measure BID31 to that of the expert, and that a choice of regularizer for the cost function yields an objective which is analogous to that of GAN.

Their algorithm, namely GAIL, has become a popular choice for IL and some extensions of GAIL have been proposed BID1 BID11 BID16 .

However, those extensions have never addressed reducing the number of interactions during training.

There has been a few attempts that try to improve the sample complexity in IL literatures, such as Guided Cost Learning (GCL) BID7 .

However, those methods have worse imitation capability in comparison with GAIL, as reported by BID8 .

As detailed in section 5, our algorithm have comparable imitation capability to GAIL while improving the sample complexity.

Hester & Osband FORMULA7 proposed an off-policy algorithm using the expert demonstration.

They address problems where both demonstration and hand-crafted rewards are given.

Whereas, we address problems where only the expert demonstration is given.

There is another line of IL works where the learner can ask the expert which actions should be taken during training, such as DAgger BID24 , SEARN (Daumé & Marcu, 2009 ), SMILe BID25 , and AggreVaTe BID26 .

As opposed to those methods, we do not suppose that the learner can query the expert during training.

In our experiments, we aim to answer the following three questions: Q1.

Can our algorithm enable the learner to imitate the expert behavior?

Q2.

Is our algorithm more sample efficient than BC in terms of the expert demonstration?

Q3.

Is our algorithm more efficient than GAIL in terms of the training time?

To answer the questions above, we use five physics-based control tasks that are simulated with MuJoCo physics simulator BID32 .

See Appendix A for the description of each task.

In the experiments, we compare the performance of our algorithm, BC, GAIL, and GAIL initialized by BC 23 .

The implementation details can be found in Appendix B.

We train an agent on each task by TRPO BID28 using the rewards defined in the OpenAI Gym BID2 , then we use the resulting agent with a stochastic policy as the expert for the IL algorithms.

We store (s t , a t , s t+1 ) triplets during the expert demonstration, then the triplets are used as training samples in the IL algorithms.

In order to study the sample efficiency of the IL algorithms, we arrange two setups.

The first is sparse sampling setup, where we randomly sample 100 (s t , a t , s t+1 ) triplets from each trajectory which contains 1000 triplets.

Then we perform the IL algorithms using datasets that consist of several 100s triplets.

Another setup is dense sampling setup, where we use full (s t , a t , s t+1 ) triplets in each trajectory, then train the learner using datasets that consist of several trajectories.

If an IL algorithm succeeds to imitate the expert behavior in the dense sampling setup whereas it fails in the sparse sampling setup, we evaluate the algorithm as sample inefficient in terms of the expert demonstration.

The performance of the experts and the learners are measured by cumulative reward they earned in a trajectory.

We run three experiments on each task, and measure the performance during training.

Figure 2 shows the experimental results in both sparse and dense sampling setup.

In comparison with GAIL, our algorithm marks worse performance on Walker2d-v1 and Humanoid-v1 with the datasets of the smallest size in sparse sampling setup, better performance on Ant-v1 in both setups, and competitive performance on the other tasks in both setups.

Overall, we conclude that our algorithm is competitive with GAIL with regards to performance.

That is, our algorithm enables the learner to imitate the expert behavior as well as GAIL does.

BC imitates the expert behavior successfully on all tasks in the dense sampling setup.

However, BC often fails to imitate the expert behavior in the sparse sampling setup with smaller datasets.

Our algorithm achieves better performance than BC does all over the tasks.

It shows that our algorithm is more sample efficient than BC in terms of the expert demonstration.

Figure 3 shows the performance plot curves over validation rollouts during training in the sparse sampling setup.

The curves on the top row in Figure 3 show that our algorithm denoted by Ours trains the learner more efficiently than GAIL does in terms of training time.

In addition, the curves on the bottom row in Figure 3 show that our algorithm trains the learner much more efficiently than GAIL does in terms of the environment interaction.

As opposed to BID15 suggestion, GAIL initialized by BC (BC+GAIL) does not improve the sample efficiency, but rather harms the leaner's performance significantly.

We conducted additional ablation experiments to demonstrate that our proposed method described in Section.3 improves the sample efficiency.

FIG2 shows the ablation experimental results on Antv1 task.

Ours+OnP, which denotes an on-policy variant of Ours, requires 100 times more interactions than Ours.

The result with Ours+OnP suggests that adopting off-policy learning scheme instead of on-policy one significantly improves the sample efficiency.

Ours+IRL(D) and Ours+IRL(E) are variants of Ours that learn value function approximators using the learned reward function with the objective (2) and FORMULA9 , respectively.

The result with Ours+IRL(D) and Ours+IRL(E) suggests that omitting the reward learning described in 3.1 makes the training stable and faster.

The result with Ours+GP, which denotes a variant of Ours that adopts the Gaussian policy, suggests that bounding action values described in 3.2 makes the training faster and stable.

The result with Ours+DP, which denotes a variant of Ours that has a deterministic policy with fixed input noises, fails to imitate the expert behavior.

It shows that the input noise variable z in our algorithm plays a role to obtain stochastic policies.

In this paper, we proposed a model-free IL algorithm for continuous control.

Experimental results showed that our algorithm achieves competitive performance with GAIL while significantly reducing the environment interactions.

A DETAILED DESCRIPTION OF EXPERIMENT TAB0 summarizes the description of each task, the performance of an agent with random policy, and the performance of the experts.

We implement our algorithm using two neural networks with two hidden layers.

Each network represents π θ and q ν .

For convenience, we call those networks for π θ and q ω as policy network (PN) and Q-network (QN), respectively.

PN has 100 hidden units in each hidden layer, and its final output is followed by hyperbolic tangent nonlinearity to bound its action range.

QN has 500 hidden units in each hidden layer and a single output is followed by sigmoid nonlinearity to bound the output between [0,1].

All hidden layers are followed by leaky rectified nonlinearity BID18 .

The parameters in all layers are initialized by Xavier initialization BID9 .

The input of PN is given by concatenated vector representations for the state s and noise z. The noise vector, of which dimensionality corresponds to that of the state vector, generated by zero-mean normal distribution so that z ∼ P z = N (0, 1).

The input of QN is given by concatenated vector representations for the state s and action a. We employ RMSProp BID14 for learning parameters with a decay rate 0.995 and epsilon 10 −8 .

The learning rates are initially set to 10 −3 for QN and 10 −4 for PN, respectively.

The target QN with parameters ν ′ are updated so that ν ′ = 10 −3 * ν+(1−10 −3 ) * ν ′ at each update of ν.

We linearly decrease the learning rates as the training proceeds.

We set minibatch size of (s t , a t , s t+1 ) triplets 64, the replay buffer size |B β | = 15000, and the discount factor γ = 0.85.

We sample 128 noise vectors for calculating empirical expectation E z∼Pz of the gradient (6).

We use publicly available code (https://github.com/openai/imitation) for the implementation of GAIL and BC.

Note that, the number of hidden units in PN is the same as that of networks for GAIL.

All experiments are run on a PC with a 3.30 GHz Intel Core i7-5820k Processor, a GeForce GTX Titan GPU, and 32GB of RAM.

<|TLDR|>

@highlight

In this paper, we proposed a model-free, off-policy IL algorithm for continuous control. Experimental results showed that our algorithm achieves competitive results with GAIL while significantly reducing the environment interactions.
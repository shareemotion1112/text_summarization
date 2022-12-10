The goal of imitation learning (IL) is to enable a learner to imitate an expert’s behavior given the expert’s demonstrations.

Recently, generative adversarial imitation learning (GAIL) has successfully achieved it even on complex continuous control tasks.

However, GAIL requires a huge number of interactions with environment during training.

We believe that IL algorithm could be more applicable to the real-world environments if the number of interactions could be reduced.

To this end, we propose a model free, off-policy IL algorithm for continuous control.

The keys of our algorithm are two folds: 1) adopting deterministic policy that allows us to derive a novel type of policy gradient which we call deterministic policy imitation gradient (DPIG), 2) introducing a function which we call state screening function (SSF) to avoid noisy policy updates with states that are not typical of those appeared on the expert’s demonstrations.

Experimental results show that our algorithm can achieve the goal of IL with at least tens of times less interactions than GAIL on a variety of continuous control tasks.

Recent advances in reinforcement learning (RL) have achieved super-human performance on several domains BID16 BID17 BID13 .

However, on most of domains with the success of RL, the design of reward, that explains what agent's behavior is favorable, is clear enough for humans.

Conversely, on domains where it is unclear how to design the reward, agents trained by RL algorithms often obtain poor policies and their behavior is far from what we want them to do.

Imitation learning (IL) comes in such cases.

The goal of IL is to enable the learner to imitate the expert's behavior given the expert's demonstrations but the reward signals.

We are interested in IL because we desire an algorithm that can be applied in real-world environments where it is often hard to design the reward.

Besides, since it is generally hard to model a variety of the real-world environments with an algorithm, and the state-action pairs in a vast majority of the real-world applications such as robotics control can be naturally represented in continuous spaces, we focus on model free IL for continuous control.

A widely used approach of existing model free IL methods is the combination of Inverse Reinforcement Learning (IRL) BID22 BID18 BID1 BID32 and RL.

Recently, BID10 has proposed generative adversarial imitation learning (GAIL) on the line of those works.

GAIL has achieved state-of-the art performance on a variety of continuous control tasks.

However, as pointed out by BID10 , a crucial drawback of GAIL is requirement of a huge number of interactions between the learner and the environments during training 1 .

Since the interactions with environment can be too much time-consuming especially in the real-world environments, we believe that model free IL could be more applicable to the real-world environments if the number could be reduced while keeping the imitation capability satisfied as well as GAIL.To reduce the number of interactions, we propose a model free, off-policy IL algorithm for continuous control.

As opposed to GAIL and its variants BID2 BID30 BID8 BID12 those of which adopt a stochastic policy as the learner's policy, we adopt a deterministic policy while following adversarial training fashion as GAIL.

We show that combining the deterministic policy into the adversarial off-policy IL objective derives a novel type of policy gradient which we call deterministic policy imitation gradient (DPIG).

Because DPIG only integrates over the state space as deterministic policy gradient (DPG) algorithms BID25 , the number of the interactions is expected to be less than that for stochastic policy gradient (PG) which integrates over the state-action space.

Besides, we introduce a function which we call state screening function (SSF) to avoid noisy policy update with states that are not typical of those appeared on the experts demonstrations.

In order to evaluate our algorithm, we used 6 physics-based control tasks that were simulated with MuJoCo physics simulator BID29 .

The experimental results show that our algorithm enables the learner to achieve the same performance as the expert does with at least tens of times less interactions than GAIL.

It indicates that our algorithm is more applicable to the real-world environments than GAIL.

We consider a Markov Decision Process (MDP) which is defined as a tuple {S, A, T , r, d 0 , γ}, where S is a set of states, A is a set of possible actions agents can take, T : S×A×S → [0, 1] is a transition probability, r : S×A → R is a reward function, d 0 : S → [0, 1] is a distribution over initial states, and γ ∈ (0, 1] is a discount factor.

The agent's behavior is defined by a stochastic policy π : S×A → [0, 1].

Performance measure of the policy is defined as J (π, r) = E ∞ t=0 γ t r(s t , a t )|d 0 , T , π where s t ∈ S is a state that the agent received at discrete time-step t, and a t ∈ A is an action taken by the agent after receiving s t .

Using discounted state visitation distribution (SVD) denoted by DISPLAYFORM0 where P is a probability that the agent receives the state s at time-step t, the performance measure can be rewritten as J (π, r) = E s∼ρπ,a∼π r(s, a) .

In IL literature, the agent indicates both the expert and the learner.

In this paper, we consider that the states and the actions are represented in continuous spaces S = R ds and A = R da respectively.

The goal of RL is to find an optimal policy that maximizes the performance measure.

In this paper, we consider the policy-based methods rather than the value-based method such as Q-learning.

Given the reward function r, the objective of RL with parameterized stochastic policies π θ is defined as follows.

DISPLAYFORM0 Typically, as do REINFORCE BID31 family of algorithms, the update of θ is performed by gradient ascent with estimations of ∇ θ J (π θ , r) which is called PG BID27 .

There are several different expressions of PG which can be represented as follows.

DISPLAYFORM1 Where ζ denotes trajectories generated by the agent with current policy π θ , and Ψ t (s t , a t ) can be chosen from a set of the expressions for expected discounted cumulative rewards E ∞ t =t γ t −t r(s t , a t )|s t , a t .

We refer the readers to BID24 for the expressions in detail.

As mentioned in Section 1, we consider deterministic policy µ θ : S → A which is a special case of stochastic policy.

In the RL literature, gradient estimation of the performance measure for deterministic policy which is called DPG BID25 can be represented as follows.

DISPLAYFORM2 Where Q : S×A → R denotes Q-function.

Note that, both PG and DPG require calculation or approximation of the expected discounted cumulative rewards, and do not suppose that the reward function is accessible.

That is, as opposed to IRL algorithms, RL algorithms using PG or DPG do not suppose that the reward function is parameterized.

The goal of IRL is to find a reasonable reward function based on an assumption that the cumulative rewards earned by the expert's behavior are greater than or equal to those earned by any non-experts' behavior.

The objective of IRL with parameterized reward functions r ω can be defined as follows.

DISPLAYFORM0 Where π E and π denote the expert's and the non-expert's policies respectively.

The update of ω is typically performed by gradient ascent with estimations of ∇ ω J (π E , r ω ) − ∇ ω J (π, r ω ).

Likewise IRL the reward function is unknown in IL setting.

Hence, the decision process turns to be characterized by MDP\r = {S, A, T , d 0 , γ} and the performance measure becomes unclear unless the reward function can be found.

Given the parameterized reward functions r ω learned by IRL, we suppose the decision process becomes MDP ∪ r ω \ r = {S, A, T , r ω , d 0 , γ}. We consider the widely used IL approach that combines IRL and RL, and its objective can be defined as a composition of FORMULA1 and FORMULA4 as follows.

DISPLAYFORM0 It is common to alternate IRL and RL using the gradient estimation of FORMULA5 with respect to ω and to θ respectively.

We can see this IL process as adversarial training as pointed out in BID10 BID5 .

We define the parameterized reward functions as r ω (s, a) = log R ω (s, a), where DISPLAYFORM0 represents a probability that the state-action pair (s, a) belong to the trajectories demonstrated by the expert, and thus r ω (s, a) represents its log probability.

Besides, we introduce a stochastic behavior policy BID20 BID4 DISPLAYFORM1 We thereby define the objective of IRL in our algorithm as follows.

DISPLAYFORM2 Although we adopt deterministic policy µ θ : S → A as mentioned in Section 1, a well-known concern for the deterministic policy is about its exploration in the state space.

To ensure the adequate exploration with the deterministic policy, we introduce an off-policy learning scheme with the stochastic behavior policy β introduced above.

We assume that the stochastic behavior policy β is a mixture of the past learner's policies, and approximate the performance measure for µ θ as DISPLAYFORM3 as off-policy DPG algorithms BID25 .

Therefore, the objective of the learner to imitate the expert's behavior can be approximated as follows.

DISPLAYFORM4 We alternate the optimization for (6) and FORMULA10 as GAIL.

In practice, as do common off-policy RL methods BID16 BID13 , we perform the state sampling s ∼ ρ β using a replay buffer B β .

The replay buffer is a finite cache and stores the state-action pairs in first-in first-out manner while the learner interacts with the environment.

Whereas RL algorithms using DPG do not suppose that the reward function is parameterized as mentioned in Section 2.2, the reward function in FORMULA10 is parameterized.

Hence, our algorithm can obtain gradients of the reward function with respect to action executed by the deterministic policy, and apply the chain rule for the gradient estimation of (7) with respect to θ as follows.

DISPLAYFORM5 Thus, combining the deterministic policy into the adversarial IL objective with the parameterized reward function enables us to derive this novel gradient estimation (8) which we call deterministic policy imitation gradient (DPIG).

Let S β ⊂ S and S E ⊂ S be subsets of S explored by the learner and the expert respectively, let S β * E = S β ∩ S E and S β * E = S \ S β * E be intersection of the two subsets and its complement, and let U ω : S → ∆θ be updates of the learner's policy with DPIG.

If there exist states s ∈ S β * E , we would say the mapping U ω (s) for the states is reasonable to get r ω (s, µ θ (s)) close to r ω (s, a) where a ∼ π E (·|s).

In other words, since r ω can reason desirable actions a ∼ π E (·|s) and undesirable actions µ θ (s) for the states s ∈ S β * E through comparisons between (s, a) and (s, µ θ (s)) in IRL process, U ω (s) for the states could reasonably guide the leaner to right directions to imitate the expert's behavior.

Conversely, if there exist states s ∈ S β * E , the mapping U ω (s) for the states would be unreasonable, since the comparisons for the states are not performed in IRL process.

As a result, U ω (s) for the states s ∈ S β * E , which we call noisy policy updates, often guide the leaner to somewhere wrong without any supports.

In IL setting, S β * E is typically small due to a limited number of demonstrations, and S β * E is greatly lager than S β * E .

Thus, the noisy policy updates could frequently be performed in IL and make the learner's policy poor.

From this observation, we assume that preventing the noisy policy updates with states that are not typical of those appeared on the expert's demonstrations benefits to the imitation.

Based on the assumption above, we introduce an additional function v η : S → R , which we call state screening function (SSF), parameterized by η.

We empirically found that SSF works with form v η (s) = 1/(−log V η (s) + ) where V η : S → [0, 1] and > 0.V η (s) represents a probability that the state s belongs to S E .

Hence, the values of v η (s) are expected to be much smaller values for the states s ∈ S β * E ∩ S β than those for s ∈ S β * E ⊆ S E .

We show below the final objective functions K r (ω), K v (η), and K µ (θ) to be maximized in our algorithm.

DISPLAYFORM0 DISPLAYFORM1 DISPLAYFORM2 The update of parameters ω, η and θ uses the gradient estimations DISPLAYFORM3 respectively, where ∇ θ K µ (θ) follows DPIG (7).

The overview of our algorithm is described in Algorithm 1.

Note that, the parameter θ is updated while accessing states s ∈ S β = (S β * E ∩ S β ) ∪ S β * E , and SSF works as weighted sampling coefficients of which amounts for the states in S β * E are greater than those for the states in S β * E ∩ S β .

Thus, effects of the noisy policy updates to the learner's policy can be reduced.

One may think that the application of SSF in (11) makes policy updates more similar to applying generative adversarial training over the actions while sampling states from ρ E as true distribution.

However, a notable difference of IL from generative adversarial training such as GANs BID7 is that sampling states from the SVDs of current learner's policy or behavioral policy are essential for estimating policy gradients that improve the learner's behavior.

Whereas, sampling states from distributions, such as ρ E or uniform distribution on S, would not make sense for the improvement theoretically.

We think that sampling states from ρ β with SSF in equation FORMULA1 can also be interpreted as sampling states from SVDs of current learner's policy only on S β * E if V η was well approximated.

In order to evaluate our algorithm, we used 6 physics-based control tasks that were simulated with MuJoCo physics simulator BID29 .

We trained agents on each task by an RL algorithm, namely trust region policy optimization (TRPO) BID23 , using the rewards defined in the OpenAI Gym BID3 , then we used the resulting agents as the experts for IL algorithms.

Using a variety number of trajectories generated by the expert's policy as datasets, we trained learners by IL algorithms.

See Appendix A for description of each task, an agents performance with random policy, and the performance of the experts.

The performance of the experts and the learners were measured by cumulative reward they earned in an episode.

To measure how well the learner imitates expert's behavior, we introduced a criterion which we call performance recovery rate Initialize time-step t = 0 7:while not terminate condition do 8:Select an action at = µ θ (st) according to the current policy.

9:Execute the action at and observe new state st+1 10:Store a state-action pair (st+1, at+1) in B β .

Sample a random mini-batch of N state-action pairs (si, ai) from BE.

Sample a random mini-batch of N state-action pairs (sj, aj) from B β .

13:Update ω and η using the sampled gradients ∇ωKr(ω) and ∇ωKr(ω) respectively.

14: DISPLAYFORM0 Sample a random mini-batch of N states s k from B β .

17:Update θ using the sampled gradient ∇ θ Kµ(θ): 18: DISPLAYFORM1 end while 21: end for (PRR).

PRR was calculated by (C − C 0 )/(C E − C 0 ) where C denotes the performance of the learner, C 0 denotes the performance of agents with random policies, and C E denotes the performance of the experts.

We tested our algorithm against four algorithms: behavioral cloning (BC) (Pomerleau, 1991), GAIL, MGAIL which is model based extension of GAIL, Ours\SSF which denotes our algorithm without using SSF (v η is always one), and Ours\int which denotes our algorithm sampling states from ρ E instead of ρ β without using SSF in (11).

We ran three experiments on each task, and measure the PRR during training the learner.

We implemented our algorithm with three neural networks with two hidden layers.

Each network represented the functions µ θ , R ω , and V η .

All neural networks were implemented using TensorFlow BID0 .

See Appendix B for the implementation details.

For GAIL and MGAIL, we used publicly available code which are provided the authors 23 .

Note that number of hidden layers and number of hidden units in neural network for representing policy are the same over all methods tested.

All experiments were run on a PC with a 3.30 GHz Intel Core i7-5820k Processor, a GeForce GTX Titan GPU, and 32GB of RAM.

FIG0 shows curves of PRR measured during training with BC, GAIL, Ours\SSF, and Ours\int.

The PRR of our algorithm and GAIL achieved were nearly 1.0 over all tasks.

It indicates that our algorithm enables the learner to achieve the same performance as the expert does over a variety of tasks as well as GAIL.

We also observe that the number of interactions our algorithm required to achieve the PPR was at least tens of times less than those for GAIL over all tasks.

It show that our algorithm is much more sample efficient in terms of the number of interactions than GAIL, while keeping imitation capability satisfied as well as GAIL.

Table 1 shows that comparison of CPU-time required to achieve the expert's performance between our algorithm and GAIL.

We see that our algorithm is more sample efficient even in terms of computational cost.

We observe that SSF yielded better performance over all tasks compared to Ours\SSF.

The difference of the PRR is obvious for Ant-v1 and Humanoid-v1 of which state spaces are comparatively greater than those of the other tasks.

We also observe that SSF made difference with relatively smaller number of trajectories given as dataset in Reacher-v1 and HalfCheetah-v1.

It shows that, as we noted in Section 3.2, SSF is able to reduce the effect of the noisy policy updates in the case where S β * E is greatly lager than S β * E .

Reacher-v1, .

Each column represents different number of the expert's trajectories given as dataset (ds denotes the number).

Table 1 : Summary of how much times more CPU-time GAIL required to achieve the expert's performance than that of our algorithm.

The number of the expert's trajectories given as dataset is denoted by ds.

The CPU-time was measured when satisfying either two criteria -(a) PRR achieved to be greater than or equals to 1.0.

(b) PRR marked the highest number.

Over all tasks, both BC and Ours\int performed worse than our algorithm.

It indicates that our algorithm is less affected by the compounding error as described in Section 5, and sampling states from any distribution other than ρ β does not make sense for improving the learner's performance as mentioned in Section 3.2.

FIG1 depicts the experimental results comparing our algorithm with MGAIL on Hopper-v1 4 .

As mentioned by BID2 , the number of the interactions MGAIL required to achieve the expert's performance was relatively less than GAIL.

However, our algorithm is more sample efficient in terms of the number of interactions than MGAIL.

A wide variety of IL methods have been proposed in these last few decades.

The simplest IL method among those is BC (Pomerleau, 1991) which learns a mapping from states to actions in the expert's demonstrations using supervised learning.

Since the learner with the mapping learned by BC does not interacts with the environments, inaccuracies of the mapping are never corrected once the training has done, whereas our algorithm corrects the learner's behavior through the interactions.

A noticeable point in common between BC and our algorithm is that the both just consider the relationship between single time-step state-action pairs but information over entire trajectories of the behavior in the optimizations.

In other words, the both assume that the reward structure is dense and the reasonable rewards for the states s ∈ S \ S E can not be defined.

A drawback of BC due to ignorance of the information over the trajectories is referred to as the problem of compounding error BID21 ) -the inaccuracies compounds over time and can lead the learner to encounter unseen states in th expert's demonstrations.

For the state s ∈ S β * E , it is assumed in our algorithm that the immediate reward is greater if the learner's behavior is more likely to the expert's behavior, and the expert's behavior yields the greatest cumulative reward.

That is, maximizing the immediate reward for the state s ∈ S β * E ⊂ S E implies maximizing the cumulative reward for trajectories stating from the state in our algorithm, and thus the information over the trajectories is implicitly incorporated in log R ω of the objective (11).

Therefore, our algorithm is less affected by the compounding error than BC.Another widely used approach for IL the combination of IRL and RL that we considered in this paper.

The concept of IRL was originally proposed by BID22 , and a variety of IRL algorithms have been proposed so far.

Early works on IRL BID18 BID1 BID32 represented the parameterized reward function as a linear combination of hand-designed features.

Thus, its capabilities to represent the reward were limited in comparison with that of nonlinear functional representation.

Indeed, applications of the early works were often limited in small discrete domains.

The early works were extended to algorithms that enable to learn nonlinear functions for representing the reward BID11 BID6 .

A variety of complex tasks including continuous control in the real-world environment have succeeded with the nonlinear functional representation BID6 .

As well as those methods, our algorithm can utilize the nonlinear functions if it is differentiable with respect to the action.

In recent years, the connection between GANs and the IL approach has been pointed out BID10 BID5 .

BID10 showed that IRL is a dual problem of RL which can be deemed as a problem to match the learner's occupancy measure BID28 to that of the expert, and found a choice of regularizer for the cost function yields an objective which is analogous to that of GANs.

After that, their algorithm, namely GAIL, has become a popular choice for IL and some extensions of GAIL have been proposed BID2 BID30 BID8 BID12 .

However, those extensions have never addressed reducing the number of interactions during training whereas we address it, and our algorithm significantly reduce the number while keeping the imitation capability as well as GAIL.The way of deriving policy gradients using gradients of the parameterized reward function with respect to actions executed by the current learner's policy is similar to DPG BID25 and deep DPG BID13 .

However, they require parameterized Q-function approximator with known reward function whereas our algorithm does not use Q-function besides the parameterized reward function learned by IRL.

In IL literature, MGAIL BID2 uses the gradients derived from parameterized discriminator to update the learner's policy.

However, MGAIL is modelbased method and requires to train parameterized forward-model to derive the gradients whereas our algorithm is model free.

Although model based methods have been thought to need less environment interactions than model free methods in general, the experimental results showed that MGAIL needs more interactions than our model free algorithm.

We think that the reasons are the need for training the forward model besides that for the policy, and lack of care for the noisy policy updates issue that MGAIL essentially has.

In this paper, we proposed a model free, off-policy IL algorithm for continuous control.

The experimental results showed that our algorithm enables the learner to achieve the same performance as the expert does with several tens of times less interactions than GAIL.Although we implemented shallow neural networks to represent the parameterized functions in the experiment, deep neural networks can also be applied to represent the functions in our algorithm.

We expect that the that advanced techniques used in deep GANs enable us to apply our algorithm to more complex tasks.

A DETAILED DESCRIPTION OF EXPERIMENT TAB2 summarizes the description of each task, an agents performance with random policy, and the performance of the experts.

We implemented our algorithm with three neural networks each of which represents function µ θ , R ω , and V η .

For convenience, we call the networks as deterministic policy network (DPN) for µ θ , reward network (RN) for R ω , and state screening network (SSN) for V η .

All networks were two layer perceptrons that used leaky rectified nonlinearity (Maas et al., 2013a) except for the final output.

DPN had 100 hidden units in each hidden layer, and its final output was followed by hyperbolic tangent nonlinearity to bound its action range.

Note that the number of hidden units in DPN was the same as that of networks used in BID10 .

RN had 1000 hidden units in each hidden layer and a single output was followed by sigmoid nonlinearity to bound the output between [0,1].

The input of RN was given by concatenated vector representations for the state-action pairs.

The SSN had the same architecture as RN had except for its input which was just vector representations for the state.

The choice of hidden layer width in both RN and SSN was based on preliminary experimental results.

We employed a sampling strategy which was almost the same as deep deterministic policy gradient (DDPG) algorithm BID13 does.

When updating RN and SSN, the state-action pairs were sampled from the replay buffers B β and a buffer B E that stores the datasets.

When updating DPN, the states were sampled from the replay buffers B β .We employed RMSProp BID9 for learning parameters of RN, DPN and SSN with a learning rate 10 −4 , a decay rate 0.99, and epsilon 10 −10 for all tasks.

The weights and biases in all layers were initialized by Xavier initialization BID15 .

We used coefficient = 10 −2 for v η , mini-batch size N = 64, size of the replay buffer |B β | = 5 × 10 5 .

Regarding to the maximum number of episodes for training, we used M = 10000 for Humanoid-v1, and M = 2000 for the other tasks.

<|TLDR|>

@highlight

We propose a model free imitation learning algorithm that is able to reduce number of interactions with environment in comparison with state-of-the-art imitation learning algorithm namely GAIL.

@highlight

Proposes to extend the determinist policy gradient algorithm to learn from demonstrations, while combined with a type of density estimation of the expert.

@highlight

This paper considers the problem of model-free imitation learning and proposes an extension of the generative adversarial imitation learning algorithm by replacing the stochastic policy of the learner with a deterministic one.

@highlight

The paper combines IRL, adversarial training, and ideas from deterministic policy gradients with the goal of decreasng sample complexity
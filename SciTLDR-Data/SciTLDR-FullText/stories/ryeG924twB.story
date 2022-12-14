Existing works in deep Multi-Agent Reinforcement Learning (MARL) mainly focus on coordinating cooperative agents to complete certain tasks jointly.

However, in many cases of the real world, agents are self-interested such as employees in a company and clubs in a league.

Therefore, the leader, i.e., the manager of the company or the league, needs to provide bonuses to followers for efficient coordination, which we call expensive coordination.

The main difficulties of expensive coordination are that i) the leader has to consider the long-term effect and predict the followers' behaviors when assigning bonuses and ii) the complex interactions between followers make the training process hard to converge, especially when the leader's policy changes with time.

In this work, we address this problem through an event-based deep RL approach.

Our main contributions are threefold.

(1) We model the leader's decision-making process as a semi-Markov Decision Process and propose a novel multi-agent event-based policy gradient to learn the leader's long-term policy.

(2) We exploit the leader-follower consistency scheme to design a follower-aware module and a follower-specific attention module to predict the followers' behaviors and make accurate response to their behaviors.

(3) We propose an action abstraction-based policy gradient algorithm to reduce the followers' decision space and thus accelerate the training process of followers.

Experiments in resource collections, navigation, and the predator-prey game reveal that our approach outperforms the state-of-the-art methods dramatically.

Deep Multi-Agent Reinforcement Learning (MARL) has been widely used in coordinating cooperative agents to jointly complete certain tasks where the agent is assumed to be selfless (fully cooperative), i.e., the agent is willing to sacrifice itself to maximize the team reward.

However, in many cases of the real world, the agents are self-interested, such as taxi drivers in a taxi company (fleets) and clubs in a league.

For instance, in the example of taxi fleets (Miao et al., 2016) , drivers may prefer to stay in the area with high customer demand to gain more reward.

It is unfair and not efficient to compel the taxi driver to selflessly contribute to the company, e.g., to stay in the low customer demand area.

Forcing the drivers to selflessly contribute may increase the income for the company in a short-term but it will finally causes the low efficient and unsustainable of that company in the long run because the unsatisfied drivers may be demotivated and even leave the company.

Another important example is that the government wants some companies to invest on the poverty area to achieve the fairness of the society, which may inevitably reduce the profits of companies.

Similar to previous example, the companies may leave when the government forces them to invest.

A better way to achieve coordination among followers and achieve the leader's goals is that the manager of the company or the government needs to provide bonuses to followers, like the taxi company pays extra bonuses for serving the customers in rural areas and the government provides subsidies for investing in the poverty areas, which we term as expensive coordination.

In this paper, we solve the large-scale sequential expensive coordination problem with a novel RL training scheme.

There are several lines of works related to the expensive coordination problem, including mechanism design (Nisan & Ronen, 2001 ) and the principal-agent model (Laffont & Martimort, 2009 ).

However, these works focus more on static decisions (each agent only makes a single decision).

To consider sequential decisions, the leader-follower MDP game (Sabbadin & Viet, 2013; 2016) and the RL-based mechanism design (Tang, 2017; Shen et al., 2017) are introduced but most of their works only focus on matrix games or small-scale Markov games, which cannot be applied to the case with the large-scale action or state space.

The most related work is M 3 RL (Shu & Tian, 2019) where the leader assigns goals and bonuses by using a simple attention mechanism (summing/averaging the features together) and mind (behaviors) tracking to predict the followers' behaviors and makes response to the followers' behaviors.

But they only consider the rule-based followers, i.e., followers with fixed preference, and ignore the followers' behaviors responding to the leader's policy, which significantly simplifies the problem and leads the unreasonability of the model.

In the expensive coordination problem, there are two critical issues which should be considered: 1) the leader's long-term decision process where the leader has to consider both the long-term effect of itself and long-term behaviors of the followers when determining his action to incentivise the coordination among followers, which is not considered in (Sabbadin & Viet, 2013; Mguni et al., 2019) ; and 2) the complex interactions between the leader and followers where the followers will adapt their policies to maximize their own utility given the leader's policy, which makes the training process unstable and hard, if not unable, to converge in large-scale environment, especially when the leader changes his actions frequently, which is ignored by (Tharakunnel & Bhattacharyya, 2007; Shu & Tian, 2019) .

In this work, we address these two issues in the expensive coordination problem through an abstraction-based deep RL approach.

Our main contributions are threefold.

(1) We model the leader's decision-making process as a semiMarkov Decision Process (semi-MDP) and propose a novel event-based policy gradient to learn the leader's policy considering the long-term effect (leader takes actions at important points rather than at each step to avoid myopic decisions.) (Section 4.1).

(2) A well-performing leader's policy is also highly dependent on how well the leader knows the followers.

To predict the followers' behaviors precisely, we show the leader-follower consistency scheme.

Based on the scheme, the follower-aware module, the follower-specific attention module, and the sequential decision module are proposed to capture these followers' behaviors and make accurate response to their behaviors (Section 4.2).

(3) To accelerate the training process, we propose an action abstraction-based policy gradient algorithm for the followers.

This approach is able to reduce followers' decision space and thus simplifies the interaction between the leader and followers as well as accelerates the training process of followers (Section 4.3).

Experiments in resource collections, navigation and predatorprey show that our method outperforms the state-of-the-art methods dramatically.

Our works are closely related to leader-follower RL, temporal abstraction RL, and event-based RL.

Leader-follower RL.

The leader-follower RL targets at addressing the issue of expensive coordination where the leader wants to maximize the social benefit (or the leader's self-benefit) by coordinating non-cooperative followers through providing them bonuses.

Previous works have investigated different approaches to solve the expensive coordination, including the vanilla leader-follower MARL (Sabbadin & Viet, 2013; Laum??nier & Chaib-draa, 2005) , leader semi-MDP (Tharakunnel & Bhattacharyya, 2007) , multiple followers and sub-followers MARL (Cheng et al., 2017) , followers abstraction (Sabbadin & Viet, 2016) , and Bayesian optimization (Mguni et al., 2019) .

But most of them focus on simple tabular games or small-scale Markov games.

The most related work (Shu & Tian, 2019) leverages the deep RL approach to compute the leader's policy of assigning goals Figure 1 : Overview of our framework.

The details of the leader's module and the follower's module can be found in Section 4.2 and Section 4.3, respectively.

The implement details of each module can be found in Appendix D.2.1. and bonuses to rule-based followers.

But their method performs poorly when the followers are RLbased.

In this work, we aim to compute the leader's policy against the RL-based followers in the complex and sequential scenarios.

Temporal abstraction RL.

Our methods are also related to temporal abstraction method (Sutton et al., 1998; Daniel et al., 2016; Bacon et al., 2017; Smith et al., 2018; Zhang & Whiteson, 2019; Vezhnevets et al., 2016) .

The basic idea of temporal abstraction is to divide the original one-level decision process into a two-level decision process where the high-level part is to decide the meta goal while the low-level policy is to select the primitive actions.

Our leader's decision process is different from those methods mentioned above because the leader's policy can naturally form as an intermittent (temporal abstraction) decision process (semi-MDP) (Tharakunnel & Bhattacharyya, 2007) and it is unnecessary to design the two-level decision process for the leader (since the low-level decision process is the follower).

Based on the nature of the leader, a novel training method is introduced.

Event-based RL & Planning.

Previous studies also focus on using events to capture important elements (e.g., whether agent reaches a goal) during the whole episode.

Upadhyay et al. (2018) regard the leader's action and the environment feedback as events in the continuous time environment.

Becker et al. (2004); Gupta et al. (2018) leverage events to capture the fact that an agent has accomplished some goals.

We adopt this idea by depicting the event as the actions taken by the leader at some time steps and design a novel event-based policy gradient to learn the long-term leader's policy.

Our research focuses on single-leader multi-follower Stackelberg Markov Games (SMG) (Mguni et al., 2019; Sabbadin & Viet, 2013) , which can be formulated as a tuple G = N , S, A, ???, P, R, ?? .

N is the set of N followers, i.e., |N | = N .

S is the set of states.

s 0 ??? S 0 ??? S is an initial state and S 0 is the set of initial states.

A = ?? k???N A k is the set of joint actions for followers where a k ??? A k is an action for the k-th follower.

?? ??? ??? = ?? k???N ??? k is an action for the leader and

k is a goal and a bonus that the leader assigns to the k-th follower.

P : S ?? A ??? ???(S) is the transition function 1 and R = ?? k???N r k ?? r l is the reward function set where r k : S ?? A ?? ??? ??? R is the reward function for the k-th follower and r l : S ?? A ?? ??? ??? R is the reward function for the leader.

?? is the discount factor and a is a joint action of followers.

The leader's policy is defined as ?? = ?? k k???N where ?? k : ??? ?? S ??? ???(??? k ) is the leader's action to the k-th follower given the leader's action in the previous timestep ?? t???1 and the current state s t .

???(??) is a probability distribution.

The followers' joint policy is defined as ?? = ?? k where

is the k-th follower policy given the leader's action ?? k t and the current state s t .

Given the policy profile of the leader and followers ??, ?? , the follower's utility is defined as at, ??t) and the leader's utility is J(??, ??) = E T t=0 ?? t r l t (st, at, ??t) .

We assume that the leader and followers aim to maximize their own utilities.

We define the trajectory ?? as a sequence of state, leader's action, and followers' actions ?? ???1 , (s t , a t , ?? t ) T t=0 where ?? ???1 is the first step leader's action and is set to zero.

(a) A simple example for the illustration of AT .

Suppose that the whole step is 4, the AT = {e

(b) The probabilistic graphical model of the proposed framework.

Dotted line means that ?? affects the final result of ?? indirectly.

?????1 is set to be zero.

In this section, we propose a novel training scheme to train a well-performing leader policy against both rule-based and RL-based followers in the expensive coordination problem.

We address the two issues, the leader's long-term decision process and the complex interactions between the leader and followers, with three key steps: (a) we model the leader's decision-making process as a semiMarkov Decision Process (semi-MDP) and propose a novel event-based policy gradient to take actions only at important time steps to avoid myopic policy; (b) to accurately predict followers' behaviors, we construct a follower-aware module based on the leader-follower consistency, including a novel follower-specific attention mechanism, and a sequential decision module to predict followers' behaviors precisely and make accurate response to these behaviors; and (c) an action abstractionbased policy gradient method for followers is proposed to simplify the decision process for the followers and thus simplify the interaction between leader and followers, and accelerate the convergence of the training process.

We first describe the event-based trajectory optimization for the leader.

As we mentioned above, the leader's decision process can be naturally formulated as a semi-MDP (Tharakunnel & Bhattacharyya, 2007) .

Therefore, we firstly describe the basic ideas of semi-MDP using the modified option structure.

We define the modified option as a tuple:

where ?? is the leader's policy as we defined above and ?? k (s t , ?? t???1 ) : S ?? ??? ??? [0, 1] is the termination function for the k-th follower, to indicate the probability whether the leader's action to the k-th follower changes (?? k t???1 = ?? k t ).

Based on these definitions, we formulate the one-step option-state transition function with decay as: (Bacon et al., 2017) .

Differently, we do not have the low-level policy here (the low-level policy is the follower) and since we only focus on the finite time horizon, ?? is set to be 1.

Our modified option is used to depict the long-term decision process for the leader as shown in Fig. 2 .

Now we start to discuss our leader's policy gradient.

In fact, it is not easy to directly optimize the leader's utility based on this multi-agent option-state transition function since this form includes leader's different action stages to different followers.

Notice that for a sampled trajectory, the occurrence of the leader actions is deterministic.

Therefore, we can regard the time step and the action the leader takes at that step as an event and define the (universal) event set

We use the notation e k i = t i , ?? k ti to represent the leader's action to the k-th follower at step t i , i is the index of the event.

Since we focus on the change of the actions from the leader, we further define a set that represents a collection of new actions (?? k t = ?? k t???1 ) taken by the leader within that trajectory:

where t i ??? 1 is the previous time step.

A T represents when and how the leader commits to a new action (an example can be found in Fig. 2a ).

For brevity, e k j ??? A T means e k j ??? U T \A T .

The probability of A T can be represented as:

where t j ??? 1 is the previous time step for t j .

This equation illustrates that the probability of the occurrence of a certain leader's event set within a trajectory.

Concretely, the leader changes action to the k-th follower at t i ??? e k i while maintaining the same action within the interval from t i ??? 1 ??? e k i???1

.

Similarly, we can further define the probability of the whole trajectory ?? as:

Comparing with P (A T ), P (?? ) includes the probability of the followers as well as the state transition.

Do note that our goal is to maximize max

, indicating that the leader is required to select an action that can maximize the accumulated reward, where R ?? (T ) = T t=0 ?? t r l t is the accumulated reward and ?? is to stress that its accumulated reward is from the trajectory ?? .

Following the REINFORCE trick (Sutton & Barto, 1998) , the policy gradient for the termination function and the leader's policy function can be formulated under the following proposition: Proposition 1.

The policy gradients for the termination function ?? k (s ti , ?? ti ) and leader's policy function ?? k ?? k ti |s ti , ?? ti???1 can be written as:

where ?? and ?? are the parameters for the termination function ?? k ?? and leader's policy ?? k ?? .

I(??) and I (??) are the piece-wise functions:

All the proofs can be found in Appendix A. Proposition 1 implies that under the event-based method, whether the leader's commitment to a new action will induce different policy gradients for both termination function and the policy function.

However, from the empirical results, we find that the leader's policy function updates rarely during the whole episode because the policy only updates when the leader commits to a new action, which causes the sample inefficiency.

Notice that in fact the leader commits to the same action when e k i / ??? A T .

Therefore, the policy indication function I (??) can be formulated in an alternative way:

This form considers both committing to a new action and maintaining the same actions (Details can be found in Remark 2), which we call the Event-Based Policy Gradient (EBPG) and the previous one as the sparse EBPG respectively.

Intuitively, the dense EBPG is better than the sparse EBPG because it updates the leader's policy function more frequently than the sparse one.

For example, in time step t, supposing that the leader chooses a wrong action for follower k and receives a negative reward.

Then, the leader should learn to diminish the action chosen that state by EBPG.

The sparse EBPG only do one PG during before terminating the action (at the committing action step) while the dense one does PG in each step before terminating the action.

The latter can provide more signal to correct the wrong action.

Experiments also reveal that the dense one is better (Sec. D.3.3).

The EBPG approach is able to improve leader's performance.

However, it is still very hard for the leader to choose actions considering long-term effect only based on the current state information.

This is because the followers change their behaviors over time according to the leader's policy.

Therefore, we introduce new modules and training schemes so as to capture the change of the followers' behaviors as well as the global state.

To abstract the complicated state information, we use neural networks to learn the state representation.

To capture the followers' behaviors and make accurate response to their behaviors, we design three modules: (1) we exploit the leader-follower consistency under game regularization and policy bound conditions, (2) based on the consistency, a follower-aware module is introduced and (3) based on the follower-aware module, a novel attention mechanism, and sequential decision making module is designed to make accurate response to these followers' behaviors as shown in Fig. 1 .

Leader-Follower Consistency.

In previous works, a surge of researches focus on predicting other agents' behaviors through historical information, where the other agents are assumed to be opponents of that agent, which is only suitable for zero-sum games (Zheng et al., 2018; Foerster et al., 2018; He et al., 2016) .

However, these methods cannot be directly applied to our case because SMG is not zero-sum.

We note that Shu & Tian (2019) This assumption is inspired by (Antos et al., 2008) .

We only extend it into the multi-agent forms.

This assumption indicates that the action and states space should be limited and the reward function for the leader action should be smooth.

Assumption 2. (Policy Bound) For any agent k, reward function r k and policy is consistency, i.e.,

This assumption is inspired by (Mguni et al., 2019) .

?? ???k indicates the joint policy without the k-th agent's.

This assumption indicates that the change of the leader causes only slightly changes on each followers policy.

Based on these two assumptions, we propose a proposition here: Proposition 2.

(Leader-Follower Consistency.)

If both the assumptions of game regularization and policy bound are satisfied, for ??? > 0, k ??? N , there exists ?? > 0, such that |?? ??? ?? | ??? implies ?? k ??? ?? k ??? ??, where ?? and ?? k are the new policies for the leader and the k-th follower respectively.

These methods mentioned above are fully implemented can enhance the performance dramatically.

But when facing the RL-based followers, the SMG is still hard to converge.

This is because in SMG, the policies of the leader and followers are always changing depending on other agents' performance.

To guarantee convergence, the leader can only update its policy when the followers reach (or are near to) the best response policy (Fiez et al., 2019) .

However, when the followers are RL-based agents, there is no way to ensure the followers' policies are (near) the best response policies in large-scale SMG and the commonly-seen idea is to provide enough training time but it is unbearable in practice due to the limitation of computing power (Mguni et al., 2019) .

To accelerate the training process, inspired by the action abstraction approach which is commonlyseen in Poker (Brown & Sandholm, 2019; Tuyls et al., 2018) and action abstraction RL (Chandak et al., 2019), we collect the followers' primitive actions sharing the same properties together as a meta policy.

Then, the followers only need to select the meta action to make a decision.

Therefore, the original game is converted into a meta game, which is easy to solve.

Specifically, we define the policy for the k-th follower as:

k is the augmented state for the follower (the combination of current state and the leader's action).

?? k meta (z|??) is the meta policy for the k-th follower and z is the high-level (meta) action.

We hypothesize that the lower-level policy (the policy to choose the primitive actions) is already known (rule-based) and deterministic, i.e., ?? k lower (a k |??, z) = 1.

For instance, given the example of the navigation task, the ?? k meta can be the selection to which landmark to explore while ?? k lower is a specific route planning algorithm (such as Dijkstra Algorithm).

Based on this assumption, we can design a novel policy gradient to train the meta policy:

where ?? k is the parameter for meta-policy ?? k meta (Details can be found in Lemma 3).

In this section, we discuss how to design the leader's and followers' loss functions.

Loss Functions for the Leaders.

The basic structure for the leader is the actor-critic structure (Sutton & Barto, 1998) .

We find that adding regularizers can enhance the leader's performance and we implement the maximum entropy for the leader's policy function as well as the L2 regularization for the termination function, i.e.,

) and L reg = ?? 2 .

We also use imitation learning to learn the predicted action function p k .

Following the same logic of (Shu & Tian, 2019) , two baseline functions ?? g (c t ) and ?? b (c t ) are also introduced to further reduce the variance.

Details can be found in Appendix B.

Loss Functions for the RL-Based Followers.

The basic structure for each follower is also based on the actor-critic structure.

We leverage the action abstraction policy gradient as we mentioned above.

The learning rate between the leader and follower should satisfy the two time-scale principle (Roughly speaking, the leader learns slower than the follower(s)), similar to (Borkar, 1997) .

Details can be found in Appendix B and the pseudo-code can be found in Appendix C.

Resource Collection Multi-bonus Resource Collection Navigation Predator-prey

We evaluate the following tasks to testify the performance of our proposed method.

All of these tasks are based on SMG mentioned above.

(1) resource collections: each follower collects three types of resources including its preferred one and the leader can choose two bonuses levels (Shu & Tian, 2019); (2) multi-bonus resources collections: based on (1), the leader can choose four bonuses levels; (3) modified navigation: followers are required to navigate some landmarks and after one of the landmarks is reached, the reached landmark disappears and new landmark will appear randomly.

(4) modified predator-prey: followers are required to capture some randomly moving preys, prizes will be given after touching them.

Both (3) and (4) are based on (Lowe et al., 2017) and we modify them into our SMG setting.

Moreover, to increase the difficulty, in each episode, the combinations of the followers will change, i.e., in each task, there are 40 different followers and at each episode, we randomly choose some followers to play the game.

More details can be found in Appendix D.

Baselines & Ablations.

To evaluate our method, we compare a recently proposed method as our baseline: M 3 RL (Shu & Tian, 2019).

We do not include other baselines because other methods cannot be used in our problems, as justified in (Shu & Tian, 2019) .

For the ablations of the leader part, we choose: (1) ours: the full implementation of our method.

(2) ours w/o EBPG: removing the event-based policy gradient part; (3) ours w/o Attention: replacing follower-specified attention model by the original attention model mentioned in (Shu & Tian, 2019) .

For the follower part, we choose (a) with rule-based follower (b) with vanilla RL-based follower, and (c) with action abstraction RL-based follower to testify the ability of our methods when facing different followers.

Hyper-Parameters.

Our code is implemented in Pytorch (Paszke et al., 2017) .

If no special mention, the batch size is 1 (online learning).

Similar to (Shu & Tian, 2019), we set the learning rate as 0.001 for the leader's critic and followers while 0.0003 for the leader's policy.

The optimization algorithm is Adam (Kingma & Ba, 2014) .

Our method takes less than two days to train on a NVIDIA Geforce GTX 1080Ti GPU in each experiment.

For the loss function, we set the ?? 1 = 0.01 and ?? 2 = 0.001.

The total training episode is 250, 000 for all the tasks (including both the rule-based followers and the RL-based followers).

To encourage exploration, we use the ??-greedy 2 .

For the leader, the exploration rate is set to 0.1 and slightly decreases to zero (5000 episode).

For the followers, the exploration rate for each agent is always 0.3 (except for the noise experiments).

The quantitative results with different tasks are shown in Figs. 3 & 4.

For the rule-based followers, from Fig. 3 , we find that our method outperforms the state-of-the-art method in all the tasks, showing that our method is sample efficient and fast to coverage.

There is an interesting phenomenon that in the task of multi-bonus resource collections and navigation, only our method obtains a positive reward, indicating that our method can work well in complicated environments.

For ablations, we can see that ours w/o attention and ours w/o EBPG are worse than ours, representing these components do enhance the performance.

For the RL-based followers, from Fig. 4 , we observe that when facing the RL-based method with action abstraction, our approach outperforms the baseline method in all the tasks (in predator-prey game, the reward for ours is twice as that of the state-of-the-art).

We also find that without action abstraction, the reward is less than zero, revealing that the abstraction does play a crucial role in stabilizing training.

This experiment is to evaluate whether our method is robust to the noise, i.e., the follower randomly takes actions.

We make this experiment by introducing noise into the follower decision.

From Table 1 , we can find that our method reaches a higher total reward (more than 5) among all the environment with noise than the state-of-the-art, indicating that our method is robust to the noise.

We also observe that the total reward for the baseline method becomes lower with the increase of the noise while our method is more robust to the change.

Moreover, for the incentive (the total gain), we find that our method gains much more incentive than the state-of-the-art method, showing that our method coordinates have a better coordination the followers than the state-of-the-art method.

We also do a substantial number of experiments.

However, due to the space limitation, we can only provide some results here: (1) The total incentives: incentive can reveal the performance of successful rate interacting with the followers.

Our method outperforms the state-of-the-art method, indicating that our method has a better ability to interact with the followers.

(2) Sparse EBPG: we compare the performance gap between sparse EBPG and (dense) EBPG.

This results show that the sparse one is worse than the dense one, supporting the assumption that the dense signal can improve the sample efficiency.

(3) Visualizing attention: We visualize the attention module to find what it actually learns and the result indicates that our attention mechanism does capture the followers whom the leader needs to assign bonuses to.

(4) Two time-scale training: We testify whether our two timescale training scheme works and the ablation shows that this scheme does play an important role in improving the performance of both the leader and the followers.

(5) The committing interval: We observe that the dynamic committing interval (our method) performs better than the one with fixed committing intervals.

(6) Reward for RL-based followers: we show the reward for the followers, which can provide the situation of the followers.

The result represents that our method aids the followers to gain more than the state-of-the-art method.

(7) Number of RL-based followers: finally, we testify our method in cases with different number of RL-based followers.

The result shows that our method always performs well.

The full results can be found in Appendix D.

This paper proposes a novel RL training scheme for Stackelberg Markov Games with single leader and multiple self-interested followers, which considers the leader's long-term decision process and complicated interaction between followers with three contributions.

1) To consider the long-term effect of the leader's behavior, we develop an event-based policy gradient for the leader's policy.

2) To predict the followers' behaviors and make accurate response to their behaviors, we exploit the leader-follower consistency to design a novel follower-aware module and follower-specific attention mechanism.

3) We propose an action abstraction-based policy gradient algorithm to accelerate the training process of followers.

Experiments in resource collections, navigation, and predator-prey game reveal that our method outperforms the state-of-the-art methods dramatically.

We are willing to highlight that SMGs contribute to the RL (especially MARL) community with three key aspects: 1).

As we mentioned in the Introduction, most of the existing MARL methods assume that all the agents are willing to sacrifice themselves to maximize the total rewards, which is not true in many real-world non-cooperative scenarios.

On the contrary, our proposed method realistically assumes that agents are self-interested.

Thus, SMGs provide a new scheme focusing more on the self-interested agents.

We think this aspect is the most significant contribution to the RL community.

2).

The SMGs can be regarded as the multi-agent system with different roles (the leader and the followers) (Wilson et al., 2008) and our method provides a solution to that problem.

3).

Our methods also contribute to the hierarchical RL, i.e., it provides a non-cooperative training scheme between the high-level policy (the leaders) and the low-level policy (the followers), which plays an important role when the followers are self-interested.

Moreover, our EBPG also propose an novel policy gradient method for the temporal abstraction structure.

There are several directions we would like to investigate to further extend our SMG model: i) we will consider multiple cooperative/competitive leaders and multiple self-interested followers, which is the case in the labor market, ii) we will consider multi-level leaders, which is the case in the hierarchical organizations and companies and iii) we will consider the adversarial attacks to our SMG model, which may induce extra cost to the leader for efficient coordination.

We believe that our work is a preliminary step towards a deeper understanding of the leader-follower scheme in both research and the application to society.

We would like to thank Tianming Shu, Suming Yu, Enrique Munoz de Cote, and Xu He for their kind suggestions and helps.

We also appreciate the anonymous reviewers for their hard work.

where ?? and ?? are the parameters for the termination function ?? k ?? and the leader's policy ?? k ?? .

I(??) and I (??) are the piece-wise functions:

Proof.

First recall the utility for the leader:

Where m is the number of the times taking new action, m ??? T .

If m = T , implying that the leader has taken new action at each time.

P(|A T | = m) means the probability of times taking new action within an episode.

Take derivatives on both LHS and RHS, we get:

For brevity, we use P (?? ) to represent P (s 0 ) k???N

P (s t+1 |s t , a t ), the trajectory probability.

And we use i ??? e k i and j ??? e k j to represent i ??? e k i ??? A T to j ??? e k j / ??? A T with a slight abuse of notation.

Thus the equation mentioned above can be further simplified as:

The equation above is exactly the REINFORCE trick (Sutton & Barto, 1998) and the rule of derivations.

The approximation indicates that one trajectory only has one A T 3 .

Also based on the definition of e k i and e k j , the equation can be rewritten in a more compact form:

Where I(??) is the piece-wise function:

This is the first part of the proof (the policy gradient for the termination function).

Here, we start proving the second part (the policy gradient for the leader's action).

The proof of the second part is similar to the first part.

We rewrite it to a more compact form:

Remark 1.

Some researches also focus on event-based RL but either on single-agent continuous time (Upadhyay et al., 2018) or reward representation (Gupta et al., 2018) .

We are the first to develop and implement the event-based policy gradient into the multi-agent system.

Remark 2.

In fact, the policy gradient for the leader actions might be somewhat sparse, i.e., we only update the policy when the leader changes its actions.

Notice that the leader commits to the same action when e k i / ??? A T .

Therefore, the probability of leader's action P (A T ) can also represented as:

Then the policy gradient for leader's policy

, where

For any agent k, the corresponding reward function r k w.r.t ?? is CLipschitz continuous.

Where C = (1 ??? ??) ???1 C.The last equation is drawn form the Assumption 1 and the inequality of a geometric series: Proof.

By combining Lemma 2 and Assumption 2, we can draw that:

If there exist |?? ??? ?? | < and we have:

And we set ?? ??? (1 ??? ??)C, the consistency is established.

Lemma 3.

(Action Abstraction Policy Gradient.)

Under the assumption that the low-level follower policy ?? When ?? k lower (a|??, z) is fixed and deterministic, the equation can be:

lower (a t |?? t , z t ) = 1 is the partition function.

For brevity, with a slight abuse of notation, we omit the superscript for variables a and z which represents the index of an agent.

We add a baseline function to reduce the variance of the event-based policy gradient for the leader.

We adopt the idea of successor representation (Rabinowitz et al., 2018; Shu & Tian, 2019) as two expected baseline functions:?? g (c t ) and?? b (c t ).

For the gain baseline function:

For the bonus-based baseline function:

Two baseline neural network functions with parameters?? g and?? b are trained through minimizing the mean square error:

Where c t is the attention-based latent variable.

To this end, the gradient for the leader can be formulated as:

Where

are the baseline policy gradients.

We also leverage the imitation learning to learn the action probability function p k a

The illustration of experimental scenarios can be found in Figure 5 .

Here we give some details about these environments:

Resource Collections.

This task is similar to (Shu & Tian, 2019) , which is based on the scene that the leader and the followers collect some resources.

Each follower has its own preference which might be the same (or against) to the leader's preference.

In order to make the followers obey the leader's instruction, the leader should pay the followers bonuses.

There are total 4 types of resources and for different resources each agent has different preferences.

The leader owns two type of bonus (1 or 2) and 4 types of goals (each resource is a goal).

The number of leader is 1 while the number of followers is 4.

Multi-Bonus Resource Collections.

This task is similar to Resource Collections.

Except that the leader can take 4 level bonuses (a bonus from 1 to 4) while each agent owns one skill.

The number of leader is 1 while the number of followers is 4.

Modified Navigation.

This task is original from (Lowe et al., 2017) .

We make some modifications here to make it suitable our SMG: the leader and the followers are going to navigate some landmarks.

Each follower has its own preference which might be the same (or against) to the leader's preference.

When a landmark has been navigated, it disappears immediately and a new landmark will appear.

There are total 6 types of landmarks and for different landmarks, each agent has different preferences.

The leader owns two type of bonuses and 6 types of goals (each landmark is a goal).

The number of leader is 1 while the number of followers is 8 and the number of the landmarks is 6.

Modified Predator-Prey.

This task is also original from (Lowe et al., 2017) .

We make some modification here to make it suitable our SMG: the leader and the followers are going to catch some preys.

Each follower has its own preference which might be the same (or against) to the leader's preference.

In each step, whether a prey has been caught, it randomly chooses a direction to go.

Catching a prey will not make it disappear, which means that the preys can exist until the game ends.

There are total 8 types of preys and for different preys, each agent has different preferences.

The leader owns two type of bonuses and 8 types of goals (each prey is a goal).

The followers are faster than the preys.

The number of leader is 1 while the number of followers is 10 and the number of the landmarks is 8.

Reward Design.

The rewards mentioned in Section 3 are the general forms.

Here, we define two specified forms of the leader and followers reward function in our experiments:

Leader Reward.

We define v g as the prize (utility) for finishing task g. We set the reward function for the leader at step t as: r emphasize that our leader reward is total different from the (Shu & Tian, 2019) : in their approaches, the leader changes its mind after signing a contract will not be punished.

To make it suitable to the real world, we modify the reward as the leader should pay the followers bonuses immediately after signing the contract and cannot get back if it gives up the contract.

Follower Reward.

For the followers, we set the reward for the k-th follower as:

where u k,g reveals the payoff of the k-th follower when finishing task g (the preference).

Specifically, r k t indicates that the follower can either follow the leader's instruction or just do what it prefers to.

The followers will receive reward immediately after signing the contract (the leader and the followers achieve an agreement).

I s k t = s g means that the follower finishes the task g at step t. A penalty is added to the followers if the followers betray the leader (the followers and the leader sign the contract but the followers do not agree to work).

Our network is based on (Shu & Tian, 2019) .

Some do not suit our method.

We do some modification here: (1) We change the vanilla attention mechanism (sum/average all the history and action of each follower together) to a follower-specified one: each follower has a weight which indicates how important the follower is at the current step.

(2) The output for g and b are changed into the sequential form, i.e., we first calculate p(g k t can reveal how well the leader interacts with the followers; the higher the R in is, the more successful the coordination between the leader and the followers.

From Figure 6 and 7, comparing with the state-of-the-art method, we can see that our method far outperforms the state-of-the-art method, which reveals that our method does have a better ability to coordinate with the followers.

In all of the scenarios, without the EBPG, the performance of our method is worse than ours with EBPG.

Specifically, in some scenarios (e.g., multi-bonus resource collections, navigation), without the EBPG, the performance of our method is (or nearly) similar to the performance of M 3 RL, showing the effectiveness of our novel policy gradient.

Moreover, we can notice that in navigation environment, without follower-specified attention, the performance of our method diminishes rapidly, which implies that in some scenarios, attention does play an important role.

Figure 9: The ablation study of sparse EBPG in the predator-prey task.

In this section, we are going to testify the robust of our method.

Specifically, we evaluate whether our method is robust to the noise.

We make this experiment by introducing noise into the follower decision.

For example, if we set the noise function as 30%, indicating that there is 30% probability that the followers will choose action randomly.

This experiment testifies whether the dense event-based policy gradient increases the leader's performance comparing with the sparse event-based policy gradient.

We make ablations here: (1) Ours: the full structure of our method; (2) Sparse Event-Based Policy Gradient (sparse EBPG): the fully structure of ours except that the EBPG is replaced by sparse event-based policy gradient; (3) sparse EBPG w/o attention: replacing the follower-specified attention mechanism by averaging the input features.

From Figure 8 and 9 we can find that if the policy gradient is sparse, its performance is worse than the dense one, implying that the dense method does improve the leader's performance.

There is also an interesting phenomenon that sparse EBPG with follower-specified attention mechanism performs better than that without, revealing that the attention can stabilize training when the training signal is sparse.

Figure 11: The ablation study of reward curves for two time-scale update method in resource collections task.

Following the same logic of (Iqbal & Sha, 2019) , we visualize the weight of the attention when the leader takes actions.

From Figure 10 , we find that the attention mechanism does learn to strongly attend to the followers that the leader needs to take actions.

The followers with leader's commitment obtain much higher attention weight than others, showing that the attention module actually learn to identify the important followers while leader committing new action.

Thus, the attention mechanism does play an important role in improving the performance.

In order to evaluate the performance of our two time-scale update scheme (TTSU), we do an ablation study as shown in Fig 11.

We can find that the performance where the followers' learning rate ?? (1 ?? 10 ???3 ) is much larger than the leader's ?? (3 ?? 10 ???4 ) is better than the performance where the leader's learning rate is similar to the followers (1 ?? 10 ???3 ).

Moreover, without TTSU, the reward curves of training methods become unstable, revealing that TTSU can stabilize the training process.

In fact, TTSU improves the rate of convergence and play an important role in improving performance.

We evaluate the leader's performance between static committing interval and our dynamic committing interval.

As shown in Figure 12 , we observe that all the different fixed committing intervals only change the rate of convergence and do not enhance the leader's performance.

All the fixed committing intervals are much worse than our dynamic committing approach, revealing the fact that our dynamic committing approach aids a lot in improving the leader's performance.

We are interesting in the reward for the RL-based follower(s).

Intuitively, a well-performing leader can make the follower gain more.

As shown in Figure 13 , the reward for RL follower is higher than M 3 RL follower in all the tasks.

This represents the leader can coordinate the followers better and make them gain more reward than other methods, which forms a win-win strategy.

Finally, we evaluate the leader's performance with different number of RL-based followers.

As shown in Figure 15 , we find that our method outperforms the state-of-the-art method when facing different number of RL-based workers.

To further illustrate the performance of different combinations of our methods, we make an extra ablation here.

We choose the resource collections as the environments.

Our analysis is as follows:

For the RL based followers scenario, As shown in Table 2 , we find that the action abstraction policy gradient is very important to converge.

Additionally, adding different modules can improve the performance and the method with all the modules reach the highest reward than other combinations.

The contribution ranking for each module is: Action abstraction policy gradient > EBPG > Leaderfollower consistency.

@highlight

We propose an event-based policy gradient  to train the leader and an action abstraction policy gradient to train the followers in leader-follower Markov game.
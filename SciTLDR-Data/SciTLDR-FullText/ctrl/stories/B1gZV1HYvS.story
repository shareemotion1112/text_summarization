In multi-agent systems, complex interacting behaviors arise due to the high correlations among agents.

However, previous work on modeling multi-agent interactions from demonstrations is primarily constrained by assuming the independence among policies and their reward structures.

In this paper, we cast the multi-agent interactions modeling problem into a multi-agent imitation learning framework with explicit modeling of correlated policies by approximating opponentsâ€™ policies, which can recover agents' policies that can regenerate similar interactions.

Consequently, we develop a Decentralized Adversarial Imitation Learning algorithm with Correlated policies (CoDAIL), which allows for decentralized training and execution.

Various experiments demonstrate that CoDAIL can better regenerate complex interactions close to the demonstrators and outperforms state-of-the-art multi-agent imitation learning methods.

Our code is available at \url{https://github.com/apexrl/CoDAIL}.

Modeling complex interactions among intelligent agents from the real world is essential for understanding and creating intelligent multi-agent behaviors, which is typically formulated as a multiagent learning (MAL) problem in multi-agent systems.

When the system dynamics are agnostic and non-stationary due to the adaptive agents with implicit goals, multi-agent reinforcement learning (MARL) is the most commonly used technique for MAL.

MARL has recently drawn much attention and achieved impressive progress on various non-trivial tasks, such as multi-player strategy games (OpenAI, 2018; Jaderberg et al., 2018) , traffic light control (Chu et al., 2019) , taxi-order dispatching etc.

A central challenge in MARL is to specify a good learning goal, as the agents' rewards are correlated and thus cannot be maximized independently (Bu et al., 2008) .

Without explicit access to the reward signals, imitation learning could be the most intuitive solution for learning good policies directly from demonstrations.

Conventional solutions such as behavior cloning (BC) (Pomerleau, 1991) learn the policy in a supervised manner by requiring numerous data while suffering from compounding error (Ross & Bagnell, 2010; Ross et al., 2011) .

Inverse reinforcement learning (IRL) (Ng et al., 2000; Russell, 1998) alleviates these shortcomings by recovering a reward function but is always expensive to obtain the optimal policy due to the forward reinforcement learning procedure in an inner loop.

Generative adversarial imitation learning (GAIL) (Ho & Ermon, 2016 ) leaves a better candidate for its model-free structure without compounding error, which is highly effective and scalable.

However, real-world multi-agent interactions could be much challenging to imitate because of the strong correlations among adaptive agents' policies and rewards.

Consider if a football coach wants to win the league, he must make targeted tactics against various opponents, in addition to the situation of his team.

Moreover, the multi-agent environment tends to give rise to more severe compounding errors with more expensive running costs.

Motivated by these challenges, we investigate the problem of modeling complicated multi-agent interactions from a pile of off-line demonstrations and recover their on-line policies, which can regenerate analogous multi-agent behaviors.

Prior studies for multi-agent imitation learning typically limit the complexity in demonstrated interactions by assuming isolated reward structures (Barrett et al., 2017; Le et al., 2017; Lin et al., 2014; Waugh et al., 2013) and independence in per-agent policies that overlook the high correlations among agents (Song et al., 2018; Yu et al., 2019) .

In this paper, we cast the multi-agent interactions modeling problem into a multi-agent imitation learning framework with correlated policies by approximating opponents' policies, in order to reach inaccessible opponents' actions due to concurrently execution of actions among agents when making decisions.

Consequently, with approximated opponents model, we develop a Decentralized Adversarial Imitation Learning algorithm with Correlated policies (CoDAIL) suitable for learning correlated policies under our proposed framework, which allows for decentralized training and execution.

We prove that our framework treats the demonstrator interactions as one of -Nash Equilibrium ( -NE) solutions under the recovered reward.

In experiments, we conduct multi-dimensional comparisons for both the reward gap between learned agents and demonstrators, along with the distribution divergence between demonstrations and regenerated interacted trajectories from learned policies.

Furthermore, the results reveal that CoDAIL can better recover correlated multi-agent policy interactions than other state-of-the-art multi-agent imitation learning methods in several multi-agent scenarios.

We further illustrate the distributions of regenerated interactions, which indicates that CoDAIL yields the closest interaction behaviors to the demonstrators.

2.1 MARKOV GAME AND -NASH EQUILIBRIUM Markov game (MG), or stochastic game (Littman, 1994) , can be regarded as an extension of Markov Decision Process (MDP).

Formally, we define an MG with N agents as a tuple N, S, A

(1) , . . .

, A (N ) , P, r (1) , . . .

, r (N ) , Ď? 0 , Îł , where S is the set of states, A (i) represents the action space of agent i, where i â?? {1, 2, . . .

, N }, P : S Ă— A

(1) Ă— A (2) Ă— Â· Â· Â· Ă— A (N ) Ă— S â†’ R is the state transition probability distribution, Ď? 0 : S â†’ R is the distribution of the initial state s 0 , and Îł â?? [0, 1] is the discounted factor.

Each agent i holds its policy

] to make decisions and receive rewards defined as r

We use â?’i to represent the set of agents except i, and variables without superscript i to denote the concatenation of all variables for all agents (e.g., Ď€ represents the joint policy and a denotes actions of all agents).

For an arbitrary function f : s, a â†’ R, there is a fact that

The objective of agent i is to maximize its own total expected return R

In Markov games, however, the reward function for each agent depends on the joint agent actions.

Such a fact implies that one's optimal policy must also depend on others' policies.

For the solution to the Markov games, -Nash equilibrium ( -NE) is a commonly used concept that extends Nash equilibrium (NE) (Nash, 1951) .

where

) is the value function of agent i under state s, and Î  (i) is the set of policies available to agent i.

-NE is weaker than NE, which can be seen as sub-optimal NE.

Every NE is equivalent to an -NE where = 0.

Imitation learning aims to learn the policy directly from expert demonstrations without any access to the reward signals.

In single-agent settings, such demonstrations come from behavior trajectories sampled with the expert policy, denoted as Ď„ E = {(s t , a

.

However, in multi-agent settings, demonstrations are often interrelated trajectories, that is, which are sampled from the interactions of policies among all agents, denoted as â„¦ E = {(s t , a

.

For simplicity, we will use the term interactions directly as the concept of interrelated trajectories, and we refer to trajectories for a single agent.

Typically, behavior cloning (BC) and inverse reinforcement learning (IRL) are two main approaches for imitation learning.

Although IRL theoretically alleviates compounding error and outperforms to BC, it is less efficient since it requires resolving an RL problem inside the learning loop.

Recent proposed work aims to learn the policy without estimating the reward function directly, notably, GAIL (Ho & Ermon, 2016) , which takes advantage of Generative Adversarial Networks (GAN (Goodfellow et al., 2014)) , showing that IRL is the dual problem of occupancy measure matching.

GAIL regards the environment as a black-box, which is non-differentiable but can be leveraged through Monte-Carlo estimation of policy gradients.

Formally, its objective can be expressed as min

where D is a discriminator that identifies the expert trajectories with agents' sampled from policy Ď€, which tries to maximize its evaluation from D; H is the causal entropy for the policy and Î» is the hyperparameter.

In multi-agent learning tasks, each agent i makes decisions independently while the resulting reward

) depends on others' actions, which makes its cumulative return subjected to the joint policy Ď€.

One common joint policy modeling method is to decouple the Ď€ with assuming conditional independence of actions from different agents (Albrecht & Stone, 2018 ):

However, such a non-correlated factorization on the joint policy is a vulnerable simplification which ignores the influence of opponents .

And the learning process of agent i lacks stability since the environment dynamics depends on not only the current state but also the joint actions of all agents (Tian et al., 2019) .

To solve this, recent work has taken opponents into consideration by decoupling the joint policy as a correlated policy conditioned on state s and a

where

) is the conditional policy, with which agent i regards all potential actions from its opponent policies Ď€ (â?’i) (a (â?’i) |s), and makes decisions through the marginal policy

In multi-agent settings, for agent i with policy Ď€ (i) , it seeks to maximize its cumulative reward against demonstrator opponents who equip with demonstrated policies Ď€ (â?’i) E via reinforcement learning:

where H(Ď€ (i) ) is the Îł-discounted entropy (Bloem & Bambos, 2014; Haarnoja et al., 2017) of policy Ď€ (i) and Î» is the hyperparameter.

By coupling with Eq. (5), we define an IRL procedure to find a reward function r (i) such that the demonstrated joint policy outperforms all other policies, with the

It is worth noting that we cannot obtain the demonstrated policies from the demonstrations directly.

To solve this problem, we first introduce the occupancy measure, namely, the unnormalized distribution of s, a pairs correspond to the agent interactions navigated by joint policy Ď€:

With the definition in Eq. (7), we can further formulate Ď? Ď€ from agent i's perspective as

where

.

Furthermore, with the support of Eq. (8), we have

In analogy to the definition of occupancy measure of that in a single-agent environment, we follow the derivation from Ho & Ermon (2016) and state the conclusion directly 1 .

Proposition 1.

The IRL regarding demonstrator opponents is a dual form of following occupancy measure matching problem with regularizer Ď?, and the induced optimal policy is the primal optimum:

With setting the regularizer Ď? = Ď? GA similar to Ho & Ermon (2016) , we can obtain a GAIL-like imitation algorithm to learn Ď€

by introducing the adversarial training procedures of GANs which lead to a saddle point (

where D (i) denotes the discriminator for agent i, which plays a role of surrogate cost function and guides the policy learning.

However, such an algorithm is not practical, since we are unable to access the policies of demonstrator opponents Ď€

because the demonstrated policies are always given through sets of interactions data.

To alleviate this deficiency, it is necessary to deal with accessible counterparts.

Thereby we propose Proposition 2.

Proposition 2.

Let Âµ be an arbitrary function such that Âµ holds a similar form as Ď€ (â?’i) , then

Proof.

Substituting Ď€ (â?’i) with Âµ in Eq. (9) by importance sampling.

Proposition 2 raises an important point that a term of importance weight can quantify the demonstrator opponents.

By replacing Ď€

where

is the importance sampling weight.

In practice, it is challenging to estimate the densities and the learning methods might suffer from large variance.

Thus, we fix Î± = 1 in our implementation, and as the experimental results have shown, it has no significant influences on performance.

Besides, a similar approach can be found in Kostrikov et al. (2018) .

So far, we've built a multi-agent imitation learning framework, which can be easily generalized to correlated or non-correlated policy settings.

No prior has to be considered in advance since the discriminator is able to learn the implicit goal for each agent.

With the objective shown in Eq. (11), demonstrated interactions can be imitated by updating discriminators to offer surrogate rewards and learning their policies alternately.

Formally, the update of discriminator for each agent i can be expressed as:

and the update of policy is:

where discriminator D (i) is parametrized by Ď‰, and the policy Ď€ (i) is parametrized by Î¸.

It is worth noting that the agent i considers opponents' action a (â?’i) while updating its policy and discriminator, with integrating all its possible decisions to find the optimal response.

However, it is unrealistic to have the access to opponent joint policy Ď€(a (â?’i) |s) for agent i. Thus, it is possible to estimate opponents' actions via approximating Ď€ (â?’i) (a (â?’i) |s) using opponent modeling.

To that end, we construct a function

, as the approximation of opponents for each agent i.

Then we rewrite Eq. (13) and Eq. (14) as:

and

respectively.

Therefore, each agent i must infer the opponents model Ď? (i) to approximate the unobservable policies Ď€ (â?’i) , which can be achieved via supervised learning.

Specifically, we learn in discrete action space by minimizing a cross-entropy (CE) loss, and a mean-square-error (MSE) loss in continuous action space:

With opponents modeling, agents are able to be trained in a fully decentralized manner.

We name our algorithm as Decentralized Adversarial Imitation Learning with Correlated policies (Correlated DAIL, a.k.a.

CoDAIL) and present the training procedure in Appendix Algo.

1, which can be easily scaled to a distributed algorithm.

As a comparison, we also present a non-correlated DAIL algorithm with non-correlated policy assumption in Appendix Algo.

2.

In this section, we prove that the reinforcement learning objective against demonstrator counterparts shown in the last section is essentially equivalent to reaching an -NE.

Since we fix the policies of agents â?’i as Ď€

E , the RL procedure mentioned in Eq. (5) can be regarded as a single-agent RL problem.

Similarly, with a fixed Ď€ (â?’i) E , the IRL process of Eq. (6) is cast to a single-agent IRL problem, which recovers an optimal reward function r (i) * which achieves the best performance following the joint action Ď€ E .

Thus we have

We can also rewrite Eq. (18) as

Given the value function defined in Eq.

(1) for each agent i, for H(

, then we finally obtain

which is exactly the -NE defined in Definition 1.

We can always prove that is bounded in small values such that the -NE solution concept is meaningful.

Generally, random policies that keep vast entropy are not always considered as sub-optimal solutions or demonstrated policies Ď€

E in most reinforcement learning environments.

As we do not require those random policies, we can remove them from the candidate policy set Î  (i) , which indicates that H(Ď€ (i) ) is bounded in small values, so as .

Empirically, we adopt a small Î», and attain the demonstrator policy Ď€ E with an efficient learning algorithm to become a close-to-optimal solution.

Thus, we conclude that the objective of our CoDAIL assumes that demonstrated policies institute an -NE solution concept (but not necessarily unique) that can be controlled the hyperparameter Î» under some specific reward function, from which the agent learns a policy.

It is worth noting that Yu et al. (2019) claimed that NE is incompatible with maximum entropy inverse reinforcement learning (MaxEnt IRL) because NE assumes that the agent never takes sub-optimal actions.

Nevertheless, we prove that given demonstrator opponents, the multi-agent MaxEnt IRL defined in Eq. (6) is equivalent to finding an -NE.

Albeit non-correlated policy learning guided by a centralized critic has shown excellent properties in couple of methods, including MADDPG (Lowe et al., 2017) , COMA (Foerster et al., 2018) , MA Soft-Q (Wei et al., 2018) , it lacks in modeling complex interactions because its decisions making relies on the independent policy assumption which only considers private observations while ignores the impact of opponent behaviors.

To behave more rational, agents must take other agents into consideration, which leads to the studies of opponent modeling (Albrecht & Stone, 2018) where an agent models how its opponents behave based on the interaction history when making decisions (Claus & Boutilier, 1998; Greenwald et al., 2003; Tian et al., 2019) .

For multi-agent imitation learning, however, prior works fail to learn from complicated demonstrations, and many of them are bounded with particular reward assumptions.

For instance, Bhattacharyya et al. (2018) proposed Parameter Sharing Generative Adversarial Imitation Learning (PS-GAIL) that adopts parameter sharing trick to extend GAIL to handle multi-agent problems directly, but it does not utilize the properties of Markov games with strong constraints on the action space and the reward function.

Besides, there are many works built-in Markov games that are restricted under tabular representation and known dynamics but with specific prior of reward structures, as fully cooperative games (Barrett et al., 2017; Le et al., 2017; Ĺ oĹˇic et al., 2016; Bogert & Doshi, 2014) , two-player zero-sum games (Lin et al., 2014) , two-player general-sum games (Lin et al., 2018) , and linear combinations of specific features (Reddy et al., 2012; Waugh et al., 2013) .

Recently, some researchers take advantage of GAIL to solve Markov games.

Inspired by a specific choice of Lagrange multipliers for a constraint optimization problem (Yu et al., 2019) , Song et al. (2018) derived a performance gap for multi-agent from NE.

It proposed multi-agent GAIL (MA-GAIL), where they formulated the reward function for each agent using private actions and observations.

As an improvement, Yu et al. (2019) presented a multi-agent adversarial inverse reinforcement learning (MA-AIRL) based on logistic stochastic best response equilibrium and MaxEnt IRL.

However, both of them are inadequate to model agent interactions with correlated policies with independent discriminators.

By contrast, our approach can generalize correlated policies to model the interactions from demonstrations and employ a fully decentralized training procedure without to get access to know the specific opponent policies.

Except for the way of modeling multi-agent interactions as recovering agents' policies from demonstrations, which can regenerate similar interacted data, some other works consider different effects of interactions.

Grover et al. (2018) proposed to learn a policy representation function of the agents based on their interactions and sets of generalization tasks using the learned policy embeddings.

They regarded interactions as the episodes that contain only k (in the paper they used 2 agents), which constructs an agent-interaction graph.

Different from us, they focused on the potential relationships among agents to help characterize agent behaviors.

Besides, Kuhnt et al. (2016) and Gindele et al. (2015) proposed to use the Dynamic Bayesian Model that describes physical relationships among vehicles and driving behaviors to model interaction-dependent behaviors in autonomous driving scenario.

Correlated policy structures that can help agents consider the influence of other agents usually need opponents modeling (Albrecht & Stone, 2018) to infer others' actions.

Opponent modeling has a rich history in MAL (Billings et al., 1998; Ganzfried & Sandholm, 2011) , and lots of researches have recently worked out various useful approaches for different settings in deep MARL, e.g., DRON (He et al., 2016) and ROMMEO (Tian et al., 2019) .

In this paper, we focus on imitation learning with correlated policies, and we choose a natural and straightforward idea of opponent modeling that learning opponents' policies in the way of supervised learning with historical trajectories.

Opponent models are used both in the training and the execution stages.

Environment Description We test our method on the Particle World Environments (Lowe et al., 2017) , which is a popular benchmark for evaluating multi-agent algorithms, including several cooperative and competitive tasks.

Specifically, we consider two cooperative scenarios and two com- petitive ones as follows: 1) Cooperative-communication, with 2 agents and 3 landmarks, where an unmovable speaker knowing the goal, cooperates with a listener to reach a particular landmarks who achieves the goal only through the message from the speaker; 2) Cooperative-navigation, with 3 agents and 3 landmarks, where agents must cooperate via physical actions and it requires each agent to reach one landmark while avoiding collisions; 3) Keep-away, with 1 agent, 1 adversary and 1 landmark, where the agent has to get close to the landmark, while the adversary is rewarded by pushing away the agent from the landmark without knowing the target; 4) Predator-prey, with 1 prey agent with 3 adversary predators, where the slower predactor agents must cooperate to chase the prey agent that moves faster and try to run away from the adversaries.

Experimental Details We aim to compare the quality of interactions modeling in different aspects.

To obtain the interacted demonstrations sampled from correlated policies, we train the demonstrator agent via a MARL learning algorithm with opponents modeling to regard others' policies into one's decision making, since the ground-truth reward in those simulated environments is accessible.

Specifically, we modify the multi-agent version ACKTR Song et al., 2018) , an efficient model-free policy gradient algorithm, by keeping an auxiliary opponents model and a conditioned policy for each agent, which can transform the original centralized on-policy learning algorithm to be decentralized.

Note that we do not necessarily need experts that can do well in our designated environments.

Instead, any demonstrator will be treated as it is from an -NE strategy concept under some unknown reward functions, which will be recovered by the discriminator.

In our training procedure, we first obtain demonstrator policies induced by the ground-truth rewards and then generate demonstrations, i.e., the interactions data for imitation training.

Then we train the agents through the surrogate rewards from discriminators.

We compare CoDAIL with MA-AIRL, MA-GAIL, non-correlated DAIL (NC-DAIL) (the only difference between MA-GAIL and NC-DAIL is whether the reward function depends on joint actions or individual action) and a random agent.

We do not apply any prior to the reward structure for all tasks to let the discriminator learn the implicit goals.

All training procedures are pre-trained via behavior cloning to reduce the sample complexity, and we use 200 episodes of demonstrations, each with a maximum of 50 timesteps.

Tab.

1 and Tab.

2 show the averaged absolute differences of reward for learned agents compared to the demonstrators in cooperative and competitive tasks, respectively.

The learned interactions are considered superior if there are smaller reward gaps.

Since cooperative tasks are reward-sharing, we show only a group reward for each task in Tab.

1.

Compared to the baselines, CoDAIL achieves smaller gaps in both cooperative and competitive tasks, which suggests that our algorithm has a robust imitation learning capability of modeling the demonstrated interactions.

It is also worth noting that CoDAIL achieves higher performance gaps in competitive tasks than cooperative ones, for which we think that conflict goals motivate more complicated interactions than a shared goal.

Besides, MA-GAIL and NC-DAIL are about the same, indicating that less important is the surrogate reward structure on these multi-agent scenarios.

To our surprise, MA-AIRL does not perform well in some environments, and even fails in Predator-prey.

We list the raw obtained rewards in Appendix C, and we provide more hyperparameter sensitivity results in Appendix D.

Since we aim to recover the interactions of agents generated by the learned policies, it is proper to evaluate the relevance between distributions of regenerated interactions and demonstration data.

Specifically, we collect positions of agents over hundreds of state-action tuples, which can be regarded as the low-dimension projection of the state-action interactions.

We start each episode from a different initial state but the same for each algorithm in one episode.

We run all the experiments under the same random seed, and collect positions of each agent in the total 100 episodes, each with a maximum of 50 timesteps.

We first estimate the distribution of position (x, y) via Kernel Density Estimation (KDE) (Rosenblatt, 1956 ) with Gaussian kernel to compute the Kullback-Leibler (KL) divergence between the generated interactions with the demonstrated ones, shown in Tab.

3.

It is evident that in terms of the KL divergence between regenerated interactions with demonstrator interactions, CoDAIL generates the interaction data that obtains the minimum gap with the demonstration interaction, and highly outperforms other baseline methods.

Besides, MA-GAIL and NC-DAIL reflect about-thesame performance to model complex interactions, while MA-AIRL behaves the worst, even worse than random agents on Predator-prey.

To further understand the interactions generated by learned policies compared with the demonstrators, we visualize the interactions for demonstrator policies and all learned ones.

We plot the density distribution of positions, (x, y) and marginal distributions of x-position and y-position.

We illustrate the results conducted on Keep-away in Fig. 1 , other scenarios can be found in the Appendix E. Higher frequency positions in collected data are colored darker in the plane, and higher the value with respect to its marginal distributions.

As shown in Fig. 1 , the interaction densities of demonstrators and CoDAIL agents are highly similar (and with the smallest KL divergence), which tend to walk in the right-down side.

In contrast, other learned agents fail to recover the demonstrator interactions.

It is worth noting that even different policies can interact to earn similar rewards, but still keep vast differences among their generated interactions.

Furthermore, such a result reminds us that the real reward is not the best metric to evaluate the quality of modeling the demonstrated interactions or imitation learning (Li et al., 2017) .

In this paper, we focus on modeling complex multi-agent interactions via imitation learning on demonstration data.

We develop a decentralized adversarial imitation learning algorithm with correlated policies (CoDAIL) with approximated opponents modeling.

CoDAIL allows for decentralized training and execution and is more capable of modeling correlated interactions from demonstrations shown by multi-dimensional comparisons against other state-of-the-art multi-agent imitation learning methods on several experiment scenarios.

In the future, we will consider covering more imitation learning tasks and modeling the latent variables of policies for diverse multi-agent imitation learning.

We list the raw obtained rewards of all algorithms in each scenarios.

We evaluate how the stability of our algorithm when the hyperparameters change during our experiments on Communication-navigation.

Tab.

6 shows the total reward difference between learned agents and demonstrators when we modify the training frequency of D and G (i.e., the policy), which indicates that the frequencies of D and G are more stable when D is trained slower than G, and the result reaches a relative better performance when the frequency is 1:2 or 1:1.

Fig. 2 illustrates that the choice of Î» has little effect on the total performance.

The reason may be derived from the discrete action space in this environment, where the policy entropy changes gently.

The density and marginal distribution of agents' positions, (x, y), in 100 repeated episodes with different initialized states, generated from different learned policies upon Cooperative-navigation.

Experiments are done under the same random seed.

The top of each sub-figure is drawn from state-action pairs of all agents while the below explain for each one.

KL is the KL divergence between generated interactions (top figure) with the demonstrators.

The density and marginal distributions of agents' positions, (x, y), in 100 repeated episodes with different initialized states, generated from different learned policies upon Predator-prey.

Experiments are conducted under the same random seed.

The top of each sub-figure is drawn from state-action pairs of all agents while the below explains for each one.

The KL term means the KL divergence between generated interactions (top figure) with the demonstrators.

<|TLDR|>

@highlight

Modeling complex multi-agent interactions under multi-agent imitation learning framework with explicit modeling of correlated policies by approximating opponentsâ€™ policies. 
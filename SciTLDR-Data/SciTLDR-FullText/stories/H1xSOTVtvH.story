Producing agents that can generalize to a wide range of environments is a significant challenge in reinforcement learning.

One method for overcoming this issue is domain randomization, whereby at the start of each training episode some parameters of the environment are randomized so that the agent is exposed to many possible variations.

However, domain randomization is highly inefficient and may lead to policies with high variance across domains.

In this work, we formalize the domain randomization problem, and show that minimizing the policy's Lipschitz constant with respect to the randomization parameters leads to low variance in the learned policies.

We propose a method where the agent only needs to be trained on one variation of the environment, and its learned state representations are regularized during training to minimize this constant.

We conduct experiments that demonstrate that our technique leads to more efficient and robust learning than standard domain randomization, while achieving equal generalization scores.

Deep Reinforcement Learning (RL) has proven very successful on complex high-dimensional problems ranging from games like Go (Silver et al., 2017) and Atari games (Mnih et al., 2015) to robot control tasks .

However, one prominent issue is that of overfitting, illustrated in figure 1: agents trained on one domain fail to generalize to other domains that differ only in small ways from the original domain (Sutton, 1996; Cobbe et al., 2018; Zhang et al., 2018b; Packer et al., 2018; Zhang et al., 2018a; Witty et al., 2018; Farebrother et al., 2018) .

Good generalization is essential for problems such as robotics and autonomous vehicles, where the agent is often trained in a simulator and is then deployed in the real world where novel conditions will certainly be encountered.

Transfer from such simulated training environments to the real world is known as crossing the reality gap in robotics, and is well known to be difficult, thus providing an important motivation for studying generalization.

We focus on the problem of generalizing between environments that visually differ from each other, for example in color or texture, but where the underlying dynamics are the same.

In reinforcement learning, prior work to address this topic has studied both domain adaptation and domain randomization.

Domain adaptation techniques aim to update the data distribution in simulation to match the real distribution through some form of canonical mapping or using regularization methods (James et al., 2018; Bousmalis et al., 2017; Gamrian & Goldberg, 2018) .

Alternatively, domain randomization, in which the visual and physical properties of the training domains are randomized at the start of each episode during training, has also been shown to lead to improved generalization and transfer to the real world with little or no real world data (Tobin et al., 2017; Sadeghi & Levine, 2016; Antonova et al., 2017; Peng et al., 2017; Mordatch et al., 2015; Rajeswaran et al., 2016; OpenAI, 2018) .

However, domain randomization has been empirically shown to often lead to suboptimal policies with high variance in performance over different randomizations (Mehta et al., 2019) .

This issue can cause the learned policy to underperform in any given target domain.

We propose a regularization method for learning policies that are robust to irrelevant visual changes in the environment.

Our work combines aspects from both domain adaptation and domain randomization, in that we maintain the notion of randomized environments but use a regularization method to achieve good generalization over the randomization space.

Our contributions are the following:

• We formalize the visual domain randomization problem, and show that the Lipschitz constant of the agent's policy over visual variations provides an upper bound on the agent's robustness to these variations.

• We propose an algorithm whereby the agent is only trained on one variation of the environment but its learned representations are regularized so that the Lipschitz constant is minimized.

• We experimentally show that our method is more efficient and leads to lower-variance policies than standard domain randomization, while achieving equal or better returns and generalization ability.

This paper is structured as follows.

We first review related work, formalize the visual generalization problem, and present our theory contributions.

We then describe our regularization method, and illustrate its application to a toy gridworld problem.

Finally, we compare our method with standard domain randomization and other regularization techniques in complex visual environments.

Figure 1: Illustration of the visual generalization challenge in reinforcement learning.

In this cartpole domain, the agent must learn to keep the pole upright.

However, changes in the background color can completely throw off a trained agent.

Generalization to novel samples is well studied in supervised learning, where evaluating generalization through train/test splits is ubiquitous.

However, evaluating for generalization to novel conditions through such train/test splits is not common practice in Deep RL.

Zhang et al. (2018b) show that Deep RL algorithms are shown to suffer from overfitting to training configurations and to memorize training scenarios in discrete maze tasks.

Packer et al. (2018) study performance under train-test domain shift by modifying environmental parameters such as robot mass and length to generate new domains.

Farebrother et al. (2018) propose using different game modes of Atari games to measure generalization.

They turn to supervised learning for inspiration, finding that both L2 regularization and dropout can help agents learn more generalizable features.

These works all show that standard Deep RL algorithms tend to overfit to the environment used during training, hence the urgent need for designing agents that can generalize better.

We distinguish between two types of domain randomization: visual randomization, in which the variability between domains should not affect the agent's policy, and dynamics randomization, in which the agent should learn to adjust its behavior to achieve its goal.

Visual domain randomization, which we focus on in this work, has been successfully used to directly transfer RL agents from simulation to the real world without requiring any real images (Tobin et al., 2017; Sadeghi & Levine, 2016; Kang et al., 2019) .

These approaches used low fidelity rendering and randomized scene properties such as lighting, textures, camera position, and colors, which led to improved generalization.

Other work has also combined domain randomization and domain adaptation techniques (James et al., 2018; Chebotar et al., 2018; Gamrian & Goldberg, 2018) .

These approaches both randomize the simulated environment and penalize the gap between the trajectories in the simulations and the real world, either by adding a term to the loss, or learning a mapping between the states of the simulation and the real world.

However, these methods require a large number of samples of real world trajectories, which can be expensive to collect.

Prior work has, however, noted the inefficiency of domain randomization.

Mehta et al. (2019) show that domain randomization may lead to suboptimal policies that vary a lot between domains, and propose to train on the most informative environment variations within the given randomization ranges.

Zakharov et al. (2019) also guide the domain randomization procedure by training a DeceptionNet, that learns which randomizations are actually useful to bridge the domain gap for image classification tasks.

Learning domain-invariant features has emerged as a promising approach for taking advantage of the commonalities between domains.

For instance, in the semi-supervised context, Bachman et al. (2014) ; Sajjadi et al. (2016) ; Coors et al. (2018) ; Miyato et al. (2018); Xie et al. (2019) enforce that predictions of their networks be similar for original and augmented data points, with the objective of reducing the required amount of labelled data for training.

Our work extends such methods to reinforcement learning.

In the reinforcement learning context, several other papers have also explored this topic.

Tzeng et al. (2015) and Gupta et al. (2017) add constraints to encourage networks to learn similar embeddings for samples from both a simulated and a target domain.

Daftry et al. (2016) apply a similar approach to transfer policies for controlling aerial vehicles to different environments.

Bousmalis et al. (2017) compare different domain adaptation methods in a robot grasping task, and show that they improve generalization.

Wulfmeier et al. (2017) use an adversarial loss to train RL agents in such a way that similar policies are learned in both a simulated domain and the target domain.

While promising, these methods are designed for cases when simulated and target domains are both known, and cannot straightforwardly be applied when the target domain is only known to be within a distribution of domains.

Concurrently and independently of our work, Aractingi et al. (2019) also propose a regularization scheme to learn policies that are invariant to randomized visual changes in the environment without any real world data.

Our work differs from theirs in that we propose a theoretical justification for this regularization and an analysis of the effects of this regularization on the learned representations.

Crucially, whereas Aractingi et al. (2019) propose regularizing the network outputs, we regularize intermediate layers instead.

In the appendix, we experimentally compare their regularization to ours and show that regularizing the network outputs leads to an undesirable trade-off between agent performance and generalization.

We consider Markov decision processes (MDP) defined by (S, A, R, T, γ), where S is the state space, A the action space, R : S × A → R the reward function, T : S × A → P r(S) the transition dynamics, and γ the discount factor.

In reinforcement learning, an agent's objective is to find a policy π that maps states to distributions over actions such that the cumulative discounted reward yielded by its interactions with the environment is maximized.

We consider a framework in which we are given a set of N parameters that can be changed to visually modify the environment, defined within a randomization space Ξ ⊂ R N .

These parameters can for example control textures, colors, or lighting.

Denoting J(π, ξ) the cumulative returns of a policy π, the goal is to solve the optimization problem defined by J(π

Standard domain randomization, in which parameters ξ are randomly sampled at the start of each training episode, empirically produces policies with strongly varying performance over different regions of the randomization space, as demonstrated by Mehta et al. (2019) .

This high variance can cause the learned policy to underperform in any given target domain.

To yield insight into the robustness of policies learned by domain randomization, we start by formalizing the notion of a visually randomized MDP.

Definition 1 Let M = (S, A, R, T, γ) be an MDP.

A randomizer function of M is a mapping φ : S → S where S is a new set of states.

The randomized MDP M φ = (S φ , A φ , R φ , T φ , γ φ ) is defined as, for s, s ∈ S, a ∈ A :

Given a policy π on MDP M and a randomization M φ , we also define the agent's policy on M φ as π φ (·|s) = π(·|φ(s)).

Despite all randomized MDPs sharing the same underlying rewards and transitions, the agent's policy can vary between domains.

For example, in policy-based algorithms (Williams, 1992) , if there are several optimal policies then the agent may adopt different policies for different φ.

Furthermore, for value-based algorithms such as DQN (Mnih et al., 2015) , two scenarios can lead to there being different policies for different φ.

First, the (unique) optimal Q-function may correspond to several possible policies.

Second, imperfect function approximation can lead to different value estimates for different randomizations and thus to different policies.

To compare the ways in which policies can differ between randomized domains, we introduce the notion of Lipschitz continuity of a policy over a set of randomizations.

Definition 2 We assume the state space is equipped with a distance metric.

A policy π is Lipschitz continuous over a set of randomizations {φ} if for all randomizations φ 1 and φ 2 in {φ},

is the total variation distance between distributions (given by 1 2 a∈A |P (a) − Q(a)| when the action space is discrete).

The following inequality shows that this Lipschitz constant is crucial in quantifying the robustness of RL agents over a randomization space.

The smaller the Lipschitz constant, the less a policy is affected by different randomization parameters.

Informally, if a policy is Lipschitz continuous over randomized MDPs, then small changes in the background color in an environment will have a small impact on the policy.

We consider an MDP M and a set of randomizations {φ} of this MDP.

Let π be a K-Lipschitz policy over {φ}. Suppose the rewards are bounded by r max such that ∀a ∈ A, s ∈ S, |r(s, a)| ≤ r max .

Then for all φ 1 and φ 2 in {φ}, the following inequalities hold :

Where η i is the expected cumulative return of policy π φi on MDP M φi , for i ∈ {1, 2}, and

Proof.

See appendix.

These inequalities shows that the smaller the Lipschitz constant, the smaller the maximum variations of the policy over the randomization space can be.

In the following, we present a regularization technique that produces low-variance policies over the randomization space by minimizing the Lipschitz constant of the policy.

We propose a simple regularization method to produce an agent with policies that vary little over randomized environments, despite being trained on only one environment.

We start by choosing one variation of the environment on which to train an agent with a policy π parameterized by θ, and during training we minimize the loss

where λ is a regularization parameter, L RL is the loss corresponding to the chosen reinforcement learning algorithm, the first expectation is taken over the distribution of states visited by the current policy which we assume to be fixed when optimizing this loss, and f θ is a feature extractor used by the agent's policy.

In our experiments, we choose the output of the last hidden layer of the value or policy network as our feature extractor.

Minimizing the second term in this loss function minimizes the Lipschitz constant as defined above over the states visited by the agent, and causes the agent to learn representations of states that ignore variations caused by the randomization.

Our method can be applied to many RL algorithms, since it involves simply adding an additional term to the learning loss.

In the following, we experimentally demonstrate applications to both value-based and policy-based reinforcement learning algorithms.

Implementation details can be found in the appendix, and the code will be made available online.

We first conduct experiments on a simple gridworld to illustrate the theory described above.

Same path probability Randomized 86% Regularized 100% (ours) Figure 2 :

Left: a simple gridworld, in which the agent must make its way to the goal while avoiding the fire.

Center: empirical differences between regularized agents' policies on two randomizations of the gridworld compared to our theoretical bound in equation 1 (the dashed line).

Each point corresponds to one agent, and 20 training seeds per value of λ are shown here.

Right: probability that different agents choose the same path for two randomizations of this domain.

Our regularization method leads to more consistent behavior.

The environment we use is the 3 × 3 gridworld shown in figure 2, in which two optimal policies exist.

The agent starts in the bottom left of the grid and must reach the goal while avoiding the fire.

The agent can move either up or right, and in addition to the rewards shown in figure 2 receives -1 reward for invalid actions that would case it to leave the grid.

We set a time limit of 10 steps and γ = 1.

We introduce randomization into this environment by describing the state observed by the agent as a tuple (x, y, ξ), where (x, y) is the agent's position and ξ is a randomization parameter with no impact on the underlying MDP.

For this toy problem, we consider only two possible values for ξ: +5 and −5.

The agents we consider use the REINFORCE algorithm (Sutton et al., 2000) with a baseline (see appendix), and a multi-layer perceptron as the policy network.

First, we observe that even in a simple environment such as this one, a randomized agent regularly learns different paths for different randomizations (figure 2).

An agent trained only on ξ = 5 and regularized with our technique, however, consistently learns the same path regardless of ξ.

Although both agents easily solve the problem, the variance of the randomized agent's policy can be problematic in more complex environments in which identifying similarities between domains and ignoring irrelevant differences is important.

Next, we compare the measured difference between the policies learned by regularized agents on the two domains to the smallest of our theoretical bounds in equation 1, which in this simple environment can be directly calculated.

For a given value of λ, we train a regularized agent on the reference domain.

We then measure the difference in returns obtained by this agent on the reference and on the regularized domain, and this return determines the agent's position along the x axis.

We then numerically calculate the Lipschitz constant from the agent's action distribution over all states, and use this constant to calculate the bound in proposition 1.

This bound determines the agent's position along the y axis.

Our results for different random seeds and values of λ are shown in figure 2.

We observe that increasing λ does lead to decreases in both the empirical difference in returns and in the theoretical bound.

We compare standard visual domain randomization to our regularization method on a more challenging visual environment, in terms of 1) training stability, 2) returns and variance of the learned policies, and 3) state representations learned by the agents.

To run domain randomization experiments, we use a visual Cartpole environment shown in figure  1 , where the states consist of raw pixels of the images.

The agent must keep a pole upright as long as possible on a cart that can move left or right.

The episode terminates either after 200 time steps, if the cart leaves the track, or if the pole falls over.

The randomization consists of changing the color of the background.

Each randomized domain ξ ∈ Ξ corresponds to a color (r, g, b) , where 0 ≤ r, g, b ≤ 1.

Our implementation of this environment is based on the OpenAI Gym (Brockman et al., 2016) .

For training, we use the DQN algorithm with a CNN architecture similar to that used by Mnih et al. (2015) .

In principle, such a value-based algorithm should learn a unique value function independently of the randomization parameters we consider.

However, as we will show function approximation errors cause different value functions to be learned for different background colors.

We compare the performance of three agents.

The Normal agent is trained on only one domain (with a white background).

The Randomized agent is trained on a chosen randomization space Ξ. The Regularized agent is trained on a white background using our regularization method with respect to randomization space Ξ. The training of all three agents is done using the same hyperparameters, and over the same number of steps.

We first compare the performance of our agents during training.

We train all three agents over two randomization spaces (environments with different background colors), having the following sizes : We obtain the training curves shown in figure 3 .

We find that the normal and regularized agents have similar training curves and are not affected by the size of the randomization space.

However, the randomized agent learns more slowly on the small randomization space Ξ small (left), and also achieves worse performance on the bigger randomization space Ξ big (right).

In high-dimensional problems, we would like to pick the randomization space Ξ to be as large as possible to increase the chances of transferring to the target domain.

We find that standard domain randomization scales poorly with the size of the randomization space Ξ, whereas our regularization method is more robust to a larger randomization space.

,g,b) cube in Ξ big , where g = 1 is fixed, averaged over 1000 steps.

The training domain for both the regularized and normal agents is located at the top right.

The regularized agent learns more stable policies than the randomized agent over these domains.

• Ξ small = {(r,

We compare the returns of the policies learned by the agents in different domains within the randomization space.

We select a plane within Ξ big obtained by varying only the R and B channels but keeping G fixed.

We plot the scores obtained on this plane in figure 4 .

We see that despite having only been trained on one domain, the regularized agent achieves consistently high scores on the other domains.

On the other hand, the randomized agent's policy exhibits returns with high variance between domains, which indicates that different policies were learned for different domains.

Standard Deviations Normal 10.1 Randomized 6.2 Regularized (ours) 3.7 Figure 5 : Left:

Visualization of the representations learned by the agents for pink and green background colors and for the same set of states.

We observe that the randomized agent learns different representations for the two domains.

Right: Standard deviation of estimated value functions over randomized domains, averaged over 10 training seeds.

To understand what causes this difference in behavior between the two agents, we study the representations learned by the agents by analyzing the activations of the final hidden layer.

We consider the agents trained on Ξ big , and a sample of states obtained by performing a greedy rollout on a white background (which is included in Ξ big ).

For each of these states, we calculate the representation corresponding to that state for another background color in Ξ big .

We then visualize these representations using t-SNE plots, where each color corresponds to a domain.

A representative example of such a plot is shown in figure 5 .

We see that the regularized agent learns a similar representation for both backgrounds, whereas the randomized agent clearly separates them.

This result indicates that the regularized agent learns to ignore the background color, whereas the randomized agent is likely to learn a different policy for a different background color.

Further experiments comparing the representations of both agents can be found in the appendix.

To further study the effect of our regularization method on the representations learned by the agents, we compare the variations in the estimated value function for both agents over Ξ big .

Figure 5 shows the standard deviation of the estimated value function over different background colors, averaged over 10 training seeds and a sample of states obtained by the same procedure as described above.

We observe that our regularization technique successfully reduces the variance of the value function over the randomization domain.

Figure 6 : Left: frames from the reference and a randomized CarRacing environment.

Right: training curves of our agents, averaged over 5 seeds.

Shaded areas indicate the 95% confidence interval of the mean.

To demonstrate the applicability of our regularization method to other domains and algorithms, we also perform experiments with the PPO algorithm (Schulman et al., 2017) on the CarRacing environment (Brockman et al., 2016) , in which an agent must drive a car around a racetrack.

An example state from this environment and a randomized version in which part of the background changes color are shown in figure 6 .

We start by training 3 agents on this domain: a normal agent on the original background, a randomized agent, and a regularized agent with λ = 50.

Randomization in this experiment occurs over the entire RGB cube, which is larger than for the cartpole experiments.

Training curves are shown in figure 6 .

We see that the randomized agent fails to learn a successful policy on this large randomization space, whereas the other agents successfully learn.

We also compare the generalization ability of these agents to other agents trained with different randomization and regularization methods.

On the reference domain, we train a regularized agent with a smaller value of λ = 10, and two agents respectively with dropout 0.1 and l2 weight decay of 10 −4 , as in Cobbe et al. (2018) and Aractingi et al. (2019) .

On the randomized domain, we train an agent with the EPOpt-PPO algorithm Rajeswaran et al. (2016) , where in our implementation the agent only trains on the randomized domains on which its score is worse than average.

Scores on both the reference domain and its randomizations are shown in 1.

These results confirm that our regularization leads to agents that are both successful in training and successfully generalize to a wide range of backgrounds.

Moreover, a larger value of λ yields higher generalization scores.

Of the other regularization schemes that we tested, we find that although they do improve learning on the reference domain, only dropout leads to improvement in generalization over the randomization space compared to our baseline.

In this paper we studied generalization to visually diverse environments in deep reinforcement learning.

We formalized the problem, illustrated the inefficiencies of standard domain randomization, and proposed a theoretically grounded method that leads to robust, low-variance policies that generalize well.

We conducted several experiments in different environments of differing complexities using both on-policy and off-policy algorithms to support our claims.

The proof presented in the following applies to MDPs with a discrete action space.

However, it can straightforwardly be generalized to continuous action spaces by replacing sums over actions with integrals over actions.

The proof uses the following lemma :

Lemma 1 For two distributions p(x, y) = p(x)p(y|x) and q(x, y) = q(x)q(y|x), we can bound the total variation distance of the joint distribution :

Proof of the Lemma.

We have that :

Proof of the proposition.

We still have to bound

.

For s ∈ S we have that :

Summing over s we have that

But by marginalizing over actions : p φ1 (s|s ) = a π φ1 (a|s )p φ1 (s|a, s ), and using the fact that

And using s p(s|a, s ) = 1 we have that :

Thus, by induction, and assuming

Our second, looser bound can now be achieved as follows, Aractingi et al. (2019) .

Shaded errors correspond to 95% confidence intervals of the mean, calculated from 10 training seeds.

Bottom: scores obtained by trained agents for different regularization strengths on a plane within the RGB cube.

Concurrently and independently of our work, Aractingi et al. (2019) propose a similar regularization scheme on randomized visual domains, which they experimentally demonstrate with the PPO algorithm on the VizDoom environment Kempka et al. (2016) with randomized textures.

As opposed to the regularization scheme proposed in our work in which we regularize the final hidden layer of the network, they propose regularizing the output of the policy network.

Regularizing the last hidden layer as in our scheme more clearly separates representation learning and policy learning, since the final layer of the network is only affected by the RL loss.

We hypothesized that regularizing the output of the network directly could lead to the regularization loss and the RL loss competing against each other, such that a tradeoff between policy performance and generalization would be necessary.

To test this hypothesis, we performed experiments on the visual cartpole domain with output regularization with different values of regularization parameter λ.

Our results are shown in figure 7.

We find that increasing the regularization strength adversely affects training.

However, agents trained with higher values of λ do achieve more consistent results over the randomization space.

This shows that there is indeed a tradeoff between generalization and policy performance when regularizing the network output as in Aractingi et al. (2019) .

In our experiments, however, we have found that changing the value of λ only affects generalization ability and not agent performance on the reference domain.

All code used for our experiments will be made available online.

For our implementation of the visual cartpole environment, each image consists of 84 × 84 pixels with RGB channels.

To include momentum information in our state description, we stack k = 3 frames, so the shape of the state that is sent to the agent is 84 × 84 × 9.

We note that because of this preprocessing, agents trained until convergence achieve average returns of about 175 instead of the maximum achievable score of 200.

Since the raw pixels do not contain momentum information, we stack three frames as input to the network.

When the environment is reset, two random actions are thus taken before the agent is allowed to make a decision.

For some initializations, this causes the agent to start in a situation it cannot recover from.

Moreover, due to the low image resolution the agent may sometimes struggle to correctly identify momentum and thus may make mistakes.

In CarRacing, each state consists of 96×96 pixels with RGB channels.

We introduce frame skipping as is often done for Atari games (Mnih et al. (2015) ), with a skip parameter of 5.

This restricts the length of an episode to 200 action choices.

We then stack 2 frames to include momentum information into the state description.

The shape of the state that is sent to the agent is thus 96 × 96 × 6.

We note that although this preprocessing makes training agents faster, it also causes trained agents to not attain the maximum achievable score on this environment.

C.2 VISUAL CARTPOLE C.2.1 EXTRAPOLATION Figure 8 : Generalization scores, with 95% confidence intervals obtained over 10 training seeds.

The normal agent is trained on white (1, 1, 1), corresponding to a distance to train= 0.

The rest of the domains correspond to (x, x, x), for x = 0.9, 0.8, . . .

, 0.

Given that regularized agents are stronger in interpolation over their training domain, it is natural to wonder what the performance of these agents is in extrapolation to colors not within the range of colors sampled within training.

For this purpose, we consider randomized and regularized agents trained on Ξ big , and test them on the set {(x, x, x), 0 ≤ x ≤ 1}. None of these agents was ever exposed to x ≤ 0.5 during training.

Our results are shown in figure 8.

We find that although the regularized agent consistently outperforms the randomized agent in interpolation, both agents fail to extrapolate well outside the train domain.

Since we only regularize with respect to the training space, there is indeed no guarantee that our regularization method can produce an agent that extrapolates well.

Since the objective of domain randomization often is to achieve good transfer to an a priori unknown target domain, this result suggests that it is important that the target domain lie within the randomization space, and that the randomization space be made as large as possible during training.

We perform further experiments to demonstrate that the randomized agent learns different representations for different domains, whereas the regularized agent learns similar representations.

We consider agents trained on Ξ split = [0, 0.2] 3 ∪[0.8, 1] 3 , the union of darker, and lighter backgrounds.

We then rollout each agent on a single episode of the domain with a white background and, for each state in this episode, calculate the representations learned by the agent for other background colors.

We visualize these representations using the t-SNE plot shown in figure 9 .

We observe that the randomized agent clearly separates the two training domains, whereas the regularized agent learns similar representations for both domains.

We are interested in how robust our agents are to unseen values ξ ∈ Ξ split .

To visualize this, we rollout both agents in domains having different background colors : {(x, x, x), 0 ≤ x ≤ 1}, i.e ranging from black to white, and collect their features over an episode.

We then plot the t-SNEs of these features for both agents in figure 10 , where each color corresponds to a domain.

We observe once again that the regularized agent has much lower variance over unseen domains, whereas the randomized agent learns different features for different domains.

This shows that the regularized agent is more robust to domain shifts than the randomized agent.

is the randomization that had been selected when the transition had been observed.

We first compare the training speed of the agents.

The training curves averaged over 100 seeds are plotted in figure 11 .

We observe once again that the randomized agent is significantly slower than the regularized one, and is more unstable.

Figure 12: Generalization scores averaged over 5 training seeds and 4 test episodes per seed.

Red dots correspond to training environments Next, we examine the agents' generalization ability.

We test the agents on environments having values of pole length l and gravity g unseen during training.

We plot their scores in figure 12 .

The randomized agent clearly specializes on the two different training domains, corresponding to the two clearly distinguishable regions where high scores are achieved, whereas the regularized agents achieves more consistent scores across domain.

This result can be understood as follows.

Although the different dynamics between the two domains lead to there being different sets of optimal policies, our regularization method forces the agent to only learn policies that do not depend on the specific values of the randomized dynamics parameters.

These policies are therefore more likely to also work when those dynamics are different.

We analyze the representations learned by each agent in our dynamics randomization experiment.

Once the agents are trained, we rollout their policies in both randomized environments with angreedy strategy, where we use = 0.2 to reach a larger number of states of the MDP, over 10000 steps.

We collect the representations (the activations of the last hidden layer) corresponding to the visited states.

These features are 100-dimensional, so in order to visualize them, we use the t-SNE plots shown in figure 13 .

We emphasize that although this figure corresponds to a single training seed, the general aspect of these results is repeatable.

Figure 13 : t-sne of the representations learned by the regularized and randomized agents on the two training environments.

The randomized agent learns completely different representations for the two randomized environments.

This explains its high variance during the training, since it tries to learn a different strategy for each domain.

On the other hand, our regularized agent has the same representation for both domains, which allows it to learn much faster, and to learn policies that are robust to changes in the environment's dynamics.

@highlight

We produce reinforcement learning agents that generalize well to a wide range of environments using a novel regularization technique.

@highlight

The paper introduces the high variance policies challenge in domain randomization for reinforcement learning and mainly focuses on the problem of visual randomization, where the different randomized domains differ only in state space and the underlying rewards and dynamics are the same.

@highlight

To improve the generalization ability of deep RL agents across the tasks with different visual patterns, this paper proposed a simple regularization technique for domain randomization.
Deep reinforcement learning has achieved great success in many previously difficult reinforcement learning tasks, yet recent studies show that deep RL agents are also unavoidably susceptible to adversarial perturbations, similar to deep neural networks in classification tasks.

Prior works mostly focus on model-free adversarial attacks and agents with discrete actions.

In this work, we study the problem of continuous control agents in deep RL with adversarial attacks and propose the first two-step algorithm based on learned model dynamics.

Extensive experiments on various MuJoCo domains (Cartpole, Fish, Walker, Humanoid) demonstrate that our proposed framework is much more effective and efficient than model-free based attacks baselines in degrading agent performance as well as driving agents to unsafe states.

Deep reinforcement learning (RL) has revolutionized the fields of AI and machine learning over the last decade.

The introduction of deep learning has achieved unprecedented success in solving many problems that were intractable in the field of RL, such as playing Atari games from pixels and performing robotic control tasks (Mnih et al., 2015; Lillicrap et al., 2015; Tassa et al., 2018) .

Unfortunately, similar to the case of deep neural network classifiers with adversarial examples, recent studies show that deep RL agents are also vulnerable to adversarial attacks.

A commonly-used threat model allows the adversary to manipulate the agent's observations at every time step, where the goal of the adversary is to decrease the agent's total accumulated reward.

As a pioneering work in this field, (Huang et al., 2017) show that by leveraging the FGSM attack on each time frame, an agent's average reward can be significantly decreased with small input adversarial perturbations in five Atari games. (Lin et al., 2017) further improve the efficiency of the attack in (Huang et al., 2017) by leveraging heuristics of detecting a good time to attack and luring agents to bad states with sample-based Monte-Carlo planning on a trained generative video prediction model.

Since the agents have discrete actions in Atari games (Huang et al., 2017; Lin et al., 2017) , the problem of attacking Atari agents often reduces to the problem of finding adversarial examples on image classifiers, also pointed out in (Huang et al., 2017) , where the adversaries intend to craft the input perturbations that would drive agent's new action to deviate from its nominal action.

However, for agents with continuous actions, the above strategies can not be directly applied.

Recently, (Uesato et al., 2018) studied the problem of adversarial testing for continuous control domains in a similar but slightly different setting.

Their goal was to efficiently and effectively find catastrophic failure given a trained agent and to predict its failure probability.

The key to success in (Uesato et al., 2018) is the availability of agent training history.

However, such information may not always be accessible to the users, analysts, and adversaries.

Therefore, in this paper we study the robustness of deep RL agents in a more challenging setting where the agent has continuous actions and its training history is not available.

We consider the threat models where the adversary is allowed to manipulate an agent's observations or actions with small perturbations, and we propose a two-step algorithmic framework to find efficient adversarial attacks based on learned dynamics models.

Experimental results show that our proposed modelbased attack can successfully degrade agent performance and is also more effective and efficient than model-free attacks baselines.

The contributions of this paper are the following: Figure 1: Two commonly-used threat models.

• To the best of our knowledge, we propose the first model-based attack on deep RL agents with continuous actions.

Our proposed attack algorithm is a general two-step algorithm and can be directly applied to the two commonly-used threat models (observation manipulation and action manipulation).

• We study the efficiency and effectiveness of our proposed model-based attack with modelfree attack baselines based on random searches and heuristics (rand-U, rand-B, flip, see Section 4).

We show that our model-based attack can degrade agent performance more significantly and efficiently than model-free attacks, which remain ineffective in numerous MuJoCo domains ranging from Cartpole, Fish, Walker, and Humanoid.

Adversarial attacks in reinforcement learning.

Compared to the rich literature of adversarial examples in image classifications (Szegedy et al., 2013) and other applications (including natural language processing (Jia & Liang, 2017) , speech (Carlini & Wagner, 2018) , etc), there is relatively little prior work studying adversarial examples in deep RL.

One of the first several works in this field are (Huang et al., 2017) and (Lin et al., 2017) , where both works focus on deep RL agent in Atari games with pixels-based inputs and discrete actions.

In addition, both works assume the agent to be attacked has accurate policy and the problem of finding adversarial perturbation of visual input reduces to the same problem of finding adversarial examples on image classifiers.

Hence, (Huang et al., 2017 ) applied FGSM (Goodfellow et al., 2015 to find adversarial perturbations and (Lin et al., 2017) further improved the efficiency of the attack by heuristics of observing a good timing to attack -when there is a large gap in agents action preference between most-likely and leastlikely action.

In a similar direction, (Uesato et al., 2018) study the problem of adversarial testing by leveraging rejection sampling and the agent training histories.

With the availability of training histories, (Uesato et al., 2018) successfully uncover bad initial states with much fewer samples compared to conventional Monte-Carlo sampling techniques.

Recent work by (Gleave et al., 2019) consider an alternative setting where the agent is attacked by another agent (known as adversarial policy), which is different from the two threat models considered in this paper.

Finally, besides adversarial attacks in deep RL, a recent work (Wang et al., 2019 ) study verification of deep RL agent under attacks, which is beyond the scope of this paper.

Learning dynamics models.

Model-based RL methods first acquire a predictive model of the environment dynamics, and then use that model to make decisions (Atkeson & Santamaria, 1997) .

These model-based methods tend to be more sample efficient than their model-free counterparts, and the learned dynamics models can be useful across different tasks.

Various works have focused on the most effective ways to learn and utilize dynamics models for planning in RL (Kurutach et al., 2018; Chua et al., 2018; Chiappa et al., 2017; Fu et al., 2016) .

In this section, we first describe the problem setup and the two threat models considered in this paper.

Next, we present an algorithmic framework to rigorously design adversarial attacks on deep RL agents with continuous actions.

Let s i ∈ R N and a i ∈ R M be the observation vector and action vector at time step i, and let π : R N → R M be the deterministic policy (agent).

Let f : R N × R M → R N be the dynamics model of the system (environment) which takes current state-action pair (s i , a i ) as inputs and outputs the next state s i+1 .

We are now in the role of an adversary, and as an adversary, our goal is to drive the agent to the (un-safe) target states s target within the budget constraints.

We can formulate this goal into two optimization problems, as we will illustrate shortly below.

Within this formalism, we can consider two threat models:

Threat model (i): Observation manipulation.

For the threat model of observation manipulation, an adversary is allowed to manipulate the observation s i that the agent perceived within an budget:

where ∆s i ∈ R N is the crafted perturbation and U s ∈ R N , L s ∈ R N are the observation limits.

Threat model (ii): Action manipulation.

For the threat model of action manipulation, an adversary can craft ∆a i ∈ R M such that

M are the limits of agent's actions.

Our formulations.

Given an initial state s 0 and a pre-trained policy π, our (adversary) objective is to minimize the total distance of each state s i to the pre-defined target state s target up to the unrolled (planning) steps T .

This can be written as the following optimization problems in Equations 3 and 4 for the Threat model (i) and (ii) respectively:

A common choice of d(x, y) is the squared 2 distance x − y 2 2 and f is the learned dynamics model of the system, and T is the unrolled (planning) length using the dynamics models.

In this section, we propose a two-step algorithm to solve Equations 3 and 4.

The core of our proposal consists of two important steps: learn a dynamics model f of the environment and deploy optimization technique to solve Equations 3 and 4.

We first discuss the details of each factor, and then present the full algorithm by the end of this section.

Step 1: learn a good dynamics model f .

Ideally, if f is the exact (perfect) dynamics model of the environment and assuming we have an optimization oracle to solve Equations 3 and 4, then the solutions are indeed the optimal adversarial perturbations that give the minimal total loss with -budget constraints.

Thus, learning a good dynamics model can conceptually help on developing a strong attack.

Depending on the environment, different forms of f can be applied.

For example, if the environment of concerned is close to a linear system, then we could let f (s, a) = As + Bu, where A and B are unknown matrices to be learned from the sample trajectories (s i , a i , s i+1 ) pairs.

For a more complex environment, we could decide if we still want to use a simple linear model (the next state prediction may be far deviate from the true next state and thus the learned dynamical model is less useful) or instead switch to a non-linear model, e.g. neural networks, which usually has better prediction power but may require more training samples.

For either case, the model parameters A, B or neural network parameters can be learned via standard supervised learning with the sample trajectories pairs (s i , a i , s i+1 ).

Step 2: solve Equations 3 and 4.

Once we learned a dynamical model f , the next immediate task is to solve Equation 3 and 4 to compute the adversarial perturbations of observations/actions.

When the planning (unrolled) length T > 1, Equation 3 usually can not be directly solved by off-theshelf convex optimization toolbox since the deel RL policy π is usually a non-linear and non-convex neural network.

Fortunately, we can incorporate the two equality constraints of Equation 3 into the objective and with the remaining -budget constraint (Equation 1), Equation 3 can be solved via projected gradient descent (PGD) 1 .

Similarly, Equation 4 can be solved via PGD to get ∆a i .

We note that, similar to the n-step model predictive control, our algorithm could use a much larger planning (unrolled) length T when solving Equations 3 and 4 and then only apply the first n (≤ T ) adversarial perturbations on the agent over n time steps.

Besides, with the PGD framework, f is not limited to feed-forward neural networks.

Our proposed attack is summarized in Algorithm 2 for

Step 1, and Algorithm 3 for Step 2.

Algorithm 1 Collect trajectories 1: Input: pre-trained policy π, MaxSampleSize n s , environment env 2:

Output: a set of trajectory pairs

k ← k + 1 10: end while 11: Return S Algorithm 2 learn dynamics 1: Input: pre-trained policy π, MaxSampleSize n s , environment env, trainable parameters W 2: Output: learned dynamical model f (s, a; W ) 3: S agent ← Collect trajectories(π, n s , env) 4: S random ← Collect trajectories(random policy, n s , env) 5: f (s, a; W ) ← supervised learning algorithm(S agent ∪ S random , W ) 6: Return f (s, a; W ) Algorithm 3 model based attack 1: Input: pre-trained policy π, learned dynamical model f (s, a; W ), threat model, maximum perturbation magnitude , unroll length T , apply perturbation length n (≤ T ) 2: Output: a sequence of perturbation δ 1 , . . .

, δ n 3: if threat model is observation manipulation (Eq. 1) then

In this section, we conduct experiments on standard reinforcement learning environment for continuous control (Tassa et al., 2018) .

We demonstrate results on 4 different environments in MuJoCo Tassa et al. (2018) and corresponding tasks: Cartpole-balance/swingup, Fish-upright, Walkerstand/walk and Humanoid-stand/walk.

For the deep RL agent, we train a state-of-the-art D4PG Evaluations.

We conduct experiments for 10 different runs, where the environment is reset to different initial states in different runs.

For each run, we attack the agent for one episode with 1000 time steps (the default time intervals is usually 10 ms) and we compute the total loss and total return reward.

The total loss calculates the total distance of current state to the unsafe states and the total return reward measures the true accumulative reward from the environment based on agent's action.

Hence, the attack algorithm is stronger if the total return reward and the total loss are smaller.

Baselines.

We compare our algorithm with the following model-free attack baselines with random searches and heuristics:

• rand-U: generate m randomly perturbed trajectories from Uniform distribution with interval [− , ] and return the trajectory with the smallest loss (or reward), • rand-B: generate m randomly perturbed trajectories from Bernoulli distribution with probability 1/2 and interval [− , ], and return the trajectory with the smallest loss (or reward), • flip: generate perturbations by flipping agent's observations/actions within the budget in ∞ norm.

For rand-U and rand-B, they are similar to Monte-Carlo sampling methods, where we generate m sample trajectories from random noises and report the loss/reward of the best trajectory (with minimum loss or reward among all the trajectories).

We set m = 1000 throughout the experiments.

Our algorithm.

A 4-layer feed-forward neural network with 1000 hidden neurons per layer is trained as the dynamics model f respectively for the domains of Cartpole, Fish, Walker and Humanoid.

We use standard 2 loss (without regularization) to learn a dynamics model f .

Instead of using recurrent neural network to represent f , we found that the 1-step prediction for dynamics with the 4-layer feed-forward network is already good for the MuJoCo domains we are studying.

Specifically, for the Cartpole and Fish, we found that 1000 episodes (1e6 training points) are sufficient to train a good dynamics model (the mean square error for both training and test losses are at the order of 10 −5 for Cartpole and 10 −2 for Fish), while for the more complicated domain like Walker and Humanoid, more training points (5e6) are required to achieve a low test MSE error (at the order of 10 −1 and 10 0 for Walker and Humanoid respectively).

Consequently, we use larger planning (unrolled) length for Cartpole and Fish (e.g. T = 10, 20), while a smaller T (e.g. 3 or 5) is used for Walker and Humanoid.

Meanwhile, we focus on applying projected gradient descent (PGD) to solve Equation 3 and 4.

We use Adam as the optimizer with optimization steps equal to 30 and we report the best result for each run from a combination of 6 learning rates, 2 unroll length {T 1 , T 2 } and n steps of applying PGD solution with n ≤ T i .

For observation manipulation, we report the results on Walker, Humanoid and Cartpole domains with tasks (stand, walk, balance, swingup) respectively.

The unsafe states s target for Walker and Humanoid are set to be zero head height, targeting the situation of falling down.

For Cartpole, the unsafe states are set to have 180

• pole angle, corresponding to the cartpole not swinging up and nor balanced.

For the Fish domain, the unsafe states for the upright task target the pose of swimming fish to be not upright, e.g. zero projection on the z-axis.

The full results of both two threat models on observation manipulation and action manipulation are shown in Table 1a , b and c, d respectively.

Since the loss is defined as the distance to the target (unsafe) state, the lower the loss, the stronger the attack.

It is clear that our proposed attack achieves much lower loss in Table 1a & c than the other three model-free baselines, and the averaged ratio is also listed in 1b & d. Notably, over the 10 runs, our proposed attack always outperforms baselines for the threat model of observation perturbation and the Cartpole domain for the threat model of action perturbation, while still superior to the baselines despite losing two times to the flip baseline on the Fish domain.

Only our proposed attack can constantly make the Walker fall down (since we are minimizing its head height to be zero).

To have a better sense on the numbers, we give some quick examples below.

For instance, as shown in Table 1a and b, we show that the average total loss of walker head height is almost unaffected for the three baselines -if the walker successfully stand or walk, its head height usually has to be greater than 1.2 at every time step, which is 1440 for one episode -while our attack can successfully lower the walker head height by achieving an average of total loss of 258 (468), which is roughly 0.51(0.68) per time step for the stand (walk) task.

Similarly, for the humanoid results, a successful humanoid usually has head height greater than 1.4, equivalently a total loss of 1960 for one episode, and Table 1a shows that the d4pg agent is robust to the perturbations generated from the three modelfree baselines while being vulnerable to our proposed attack.

Indeed, as shown in Figure 2 , the walker and humanoid falls down quickly (head height is close to zero) under our specially-designed attack while remaining unaffected for all the other baselines.

Evaluating on the total reward.

Often times, the reward function is a complicated function and its exact definition is often unavailable.

Learning the reward function is also an active research field, which is not in the coverage of this paper.

Nevertheless, as long as we have some knowledge of unsafe states (which is often the case in practice), then we can define unsafe states that are related to low reward and thus performing attacks based on unsafe states (i.e. minimizing the total loss of distance to unsafe states) would naturally translate to decreasing the total reward of agent.

As demonstrated in Table 2 , the results have the same trend of the total loss result in Table 1 , where our proposed attack significantly outperforms all the other three baselines.

In particular, our method can lower the average total reward up to 4.96× compared to the baselines result, while the baseline results are close to the perfect total reward of 1000.

Evaluating on the efficiency of attack.

We also study the efficiency of the attack in terms of sample complexity, i.e. how many episodes do we need to perform an effective attack?

Here we adopt the convention in control suite (Tassa et al., 2018) where one episode corresponds to 1000 time steps (samples), and we learn the neural network dynamical model f with different number of episodes.

Figure 3 plots the total head height loss of the walker (task stand) for the three baselines and our method with dynamical model f trained with three different number of samples: {5e5, 1e6, 5e6}, or equivalently {500, 1000, 5000} episodes.

We note that the sweep of hyper parameters is the same for all the three models, and the only difference is the number of training samples.

The results show that for the baselines rand-U and flip, the total losses are roughly at the order of 1400-1500, while (21) 809 (85) 959 (5) 193 (114) walk 934 (22) 913 (21) 966 (6) 608 ( a stronger baseline rand-B still has total losses of 900-1200.

However, if we solve Equation 3 with f trained by 5e5 or 1e6 samples, the total losses can be decreased to the order of 400-700 and are already winning over the three baselines by a significant margin.

Same as our expectation, if we use more samples (e.g. 5e6, which is 5-10 times more), to learn a more accurate dynamics model, then it is beneficial to our attack method -the total losses can be further decreased by more than 2× and are at the order of 50-250 over 10 different runs.

Here we also give a comparison between our model-based attack to existing works (Uesato et al., 2018; Gleave et al., 2019) on the sample complexity.

In (Uesato et al., 2018) , 3e5 episodes of training data is used to learn the adversarial value function, which is roughly 1000× more data than even our strongest adversary (with 5e3 episodes).

Similarly, (Gleave et al., 2019) use roughly 2e4 episodes to train an adversary via deep RL, which is roughly 4× more data than ours 2 .

In this paper, we study the problem of adversarial attacks in deep RL with continuous control for two commonly-used threat models (observation manipulation and action manipulation).

Based on the threat models, we proposed the first model-based attack algorithm and showed that our formulation can be easily solved by off-the-shelf gradient-based solvers.

Through extensive experiments on 4 MuJoCo domains (Cartpole, Fish, Walker, Humanoid), we show that our proposed algorithm outperforms all the model-free based attack baselines by a large margin.

There are several interesting future directions can be investigated based on this work and is detailed in Appendix.

A.1 MORE ILLUSTRATION ON FIGURE 3

The meaning of Fig 3 is to show how the accuracy of the learned models affects our proposed technique:

1.

we first learned 3 models with 3 different number of samples: 5e5, 1e6, 5e6 and we found that with more training samples (e.g. 5e6, equivalently 5000 episodes), we are able to learn a more accurate model than the one with 5e5 training samples;

2.

we plot the attack results of total loss for our technique with 3 learned models (denoted as PGD, num train) as well as the baselines (randU, randB, Flip) on 10 different runs (initializations).

We show with the more accurate learned model (5e6 training samples), we are able to achieve a stronger attack (the total losses are at the order of 50-200 over 10 different runs) than the less accurate learned model (e.g. 5e5 training samples).

However, even with a less accurate learned model, the total losses are on the order of 400-700, which already outperforms the best baselines by a margin of 1.3-2 times.

This result in Fig 3 also suggest that a very accurate model isn't necessarily needed in our proposed method to achieve effective attack.

Of course, if the learned model is more accurate, then we are able to degrade agent's performance even more.

For the baselines (rand-U and rand-B), the adversary generates 1000 trajectories with random noise directly and we report the best loss/reward at the end of each episode.

The detailed steps are listed below:

Step 1: The perturbations are generated from a uniform distribution or a bernoulli distribution within the range [-eps, eps] for each trajectory, and we record the total reward and total loss for each trajectory from the true environment (the MuJoCo simulator)

Step 2: Take the best (lowest) total reward/loss among 1000 trajectories and report in Table 1 and 2.

We note that here we assume the baseline adversary has an "unfair advantage" since they have access to the true reward (and then take the best attack result among 1000 trials), whereas our techniques do not have access to this information.

Without this advantage, the baseline adversaries (rand-B, rand-U) may be weaker if they use their learned model to find the best attack sequence.

In any case, Table 1 and 2 demonstrate that our proposed attack can successfully uncover vulnerabilities of deep RL agents while the baselines cannot.

For the baseline 'flip', we add the perturbation (with the opposite sign and magnitude ) on the original state/action and project the perturbed state/action are within its limits.

We use default total timesteps = 1000, and the maximum total reward is 1000.

We report the total reward of the d4pg agents used in this paper below.

The agents are well-trained and have total reward close to 1000.

There are several interesting future directions can be investigated based on this work, including learning reward functions to facilitate a more effective attack, extending our current approach to develop effective black-box attacks, and incorporating our proposed attack algorithm to adversarial training of the deep RL agents.

In particular, we think there are three important challenges that need to be addressed to study adversarial training of RL agents along with our proposed attacks:

1.

The adversary and model need to be jointly updated.

How do we balance these two updates, and make sure the adversary is well-trained at each point in training?

2.

How do we avoid cycles in the training process due to the agent overfitting to the current adversary?

3.

How do we ensure the adversary doesn't overly prevent exploration / balance unperturbed vs. robust performance?

<|TLDR|>

@highlight

We study the problem of continuous control agents in deep RL with adversarial attacks and proposed a two-step algorithm based on learned model dynamics. 
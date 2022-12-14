High-dimensional sparse reward tasks present major challenges for reinforcement learning agents.

In this work we use imitation learning to address two of these challenges:  how to learn a useful representation of the world e.g.  from pixels, and how to explore efficiently given the rarity of a reward signal?

We show that adversarial imitation can work well even in this high dimensional observation space.

Surprisingly the adversary itself, acting as the learned reward function, can be tiny, comprising as few as 128 parameters, and can be easily trained using the most basic GAN formulation.

Our approach removes limitations present in most contemporary imitation approaches: requiring no demonstrator actions (only video), no special initial conditions or warm starts, and no explicit tracking of any single demo.

The proposed agent can solve a challenging robot manipulation task of block stacking from only video demonstrations and sparse reward, in which the non-imitating agents fail to learn completely.

Furthermore, our agent learns much faster than competing approaches that depend on hand-crafted, staged dense reward functions, and also better compared to standard GAIL baselines.

Finally, we develop a new adversarial goal recognizer that in some cases allows the agent to learn stacking without any task reward, purely from imitation.

minute differences between the agent and expert for a deep neural network discriminator to exploit.

Furthermore, optimizing deep adversarial networks is notoriously difficult, although there is progress towards stabilizing this problem; see e.g. ; .In this work, we show that if GAIL is provided the right kind of features, it can actually easily handle high-dimensional pixel observations using a single layer discriminator network.

We can also improve its efficiency in terms of environment interactions by using a Deep Distributed Deterministic Policy Gradients (D4PG) agent , which is a state-of-the-art off-policy method for control, that can take advantage of a replay buffer to store past experiences.

We show that several types of features can be used successfully with a tiny, single-layer adversary, in cases where a deep adversary on pixels would fail completely.

Specifically: self-supervised embeddings, e.g. Contrastive Predictive Coding (van den BID35 ; surprisingly, random projections through a deep residual network; and value network features from the D4PG agent itself.

Concurrently with BID21 , we additionally show how to modify GAIL for the off-policy case of D4PG agents with experience replay.

In our experiments, we demonstrate that our proposed approach is able to, from pixels, solve a challenging simulated robotic block stacking task using only demonstrations and a sparse binary reward indicating whether or not the stack has completed.

Previous imitation approaches to this task have all used dense, staged task rewards crafted by humans and/or true state instead of pixels.

In addition to reducing the dependency on hand-crafted rewards, our approach learns to stack faster than the dense staged reward baseline agent with the same amount of actor processes.

The main contributions of this paper are the following:??? A 6-DoF Jaco robot arm agent that learns block stacking from demonstration videos and sparse reward.

On a simulated Jaco arm it achieves 94% success rate, compared to ??? 29% using a behavior cloning agent with an equivalent number of demonstrations.??? An adversary-based early termination method for actor processes, that improves task performance and learning speed by creating a natural curriculum for the agent.??? An agent that learns with no task reward using an auxiliary goal recognizer adversary, achieving 55% stacking success from video imitation only.??? Ablation experiments on Jaco stacking as well as a 2D planar walker benchmark BID33 , to understand the specific reasons for improvement in our agent.

We find that random projections with a linear discriminator work surprisingly well in some cases, and that using value network features is even better.

Following the notation of BID32 , a Markov Decision Process (MDP) is a tuple (S, A, R, P, ??) with states S, actions A, reward function R(s, a), transition distribution P (s |s, a), and discount ??.

An agent in state s ??? S takes action a ??? A according to its policy ?? and moves to state s ??? S according to the transition distribution.

The goal of training an agent is to find a policy that maximizes the expected sum of discounted rewards, represented by the action value function DISPLAYFORM0 , where E ?? is an expectation over trajectories starting from s 0 = s and taking action a 0 = a and thereafter running the policy ??.

DDPG BID23 is an actor-critic method in which the actor, or policy ??(s|??) and the critic, or action-value function Q(s, a|??) are represented by neural networks with parameters ?? and ??, respectively.

New transitions (s, a, r, s ) are added to a replay buffer B by sampling from the policy according to a = ??(s|??) + , with ??? N added for better exploration.

The action-value function is trained to match the 1-step returns by minimizing where the transition is sampled from the replay buffer B, and ?? , Q are target actor and critic networks parameterized by ?? and ?? respectively.

To improve the stability of weight updates, the target networks are updated every K learning steps.

DISPLAYFORM0 The policy network is trained via gradient descent to produce actions that maximize the action-value function using the deterministic policy gradient BID31 : DISPLAYFORM1 Building on top of the basic DDPG agent, we also leverage several subsequent improvements following , called D4PG.

We summarize these in section 7.6.

In GAIL, a reward function is learned by training a discriminator network D(s, a) to distinguish between agent transitions and expert transitions.

The GAIL objective is formulated as follows: DISPLAYFORM0 where ?? is the agent policy, ?? E the expert policy, and H(??) an entropy regularizer.

GAIL is closely related to MaxEnt inverse reinforcement learning BID38 BID11 .

To make use of all the available training data, we use a D4PG agent that does off-policy training with experience replay with buffer B. The actor and critic updates are the same as in D4PG, but in addition we jointly train the reward function using a modified equation 3 for D: DISPLAYFORM0 As pointed out by BID21 , the use of a replay buffer changes the original expectation over the agent policy in equation 3, so that the discriminator must distinguish expert transitions from the transitions produced by all previous agents.

This could be corrected by importance sampling, but in practice, we also find this not to be necessary.

Note that we do not use actions a in the discriminator because we only assume access to videos of the expert.

Our reward function interpolates imitation reward and a sparse task reward: DISPLAYFORM1 where s is the state reached after taking action a in state s.

Note that by D(.) we mean the sigmoid of the logits, ??(x) = 1/(1 + exp(???x)), which bounds the imitation reward between 0 and 1.

This is convenient because it allows us to choose intuitive values for early termination of episodes in the actor process, e.g. if the discriminator score is too low, and it is scaled similarly to the sparse reward that we use in our block stacking experiments.

Above we include pseudocode for the actor and learner processes.

In practice we use many CPU actor processes in parallel (128 in our experiments) and a single learner process on GPU.

The actor processes receive updated network parameters every 100 acting steps.

An important detail in the actor is the use of early termination of the episode when the discriminator score is below a threshold ??, which is a hyperparameter.

This prevents the agent from drifting too far from the expert trajectories, and avoids wasting computation time.

In practice we set ?? = 0.1.

We include a plot of the training dynamics induced by this early stopping mechanism in figure 7b.

A critical design choice in our agent is the type of network used in the discriminator, whose output is used as the reward function.

If the network has too much capacity and direct access to highdimensional observations, it may be too easy for it to distinguish agent from expert, and too difficult for the agent to fool the discriminator.

If the discriminator has too little capacity, it may not capture the salient differences between agent and expert, and so cannot teach the agent how to solve the task.

FIG1 illustrates the discriminator architectures that we study in this work.

Expert demonstrations are a useful source of data for feature learning, because by construction they have a sufficient coverage of regions of state space that the agent needs to observe in order to solve the task, at least for the problem instances in the training set.

We do not assume access to expert actions, so behavior cloning is not an option for feature learning.

Also, we assume that the images are high resolution relative to what contemporary generative models are capable of generating realistically and efficiently, so we decided not to learn features by predicting in pixel space.

Furthermore, pixel prediction objectives may not encourage the learning of long term structure in the data, which we expect to be most helpful for imitation learning.

Based on these desiderata, contrastive predictive coding (CPC, van den Oord et al. (2018) ) is an appealing option.

CPC is a representation learning technique that maps a sequence of observations into a latent space such that a simple autoregressive model can easily make long-term predictions over latents.

Crucially CPC uses a probabilistic contrastive loss using negative sampling.

This allows the encoder and autoregressive model to be trained jointly and without having to need a decoder model to make predictions at the observation level.

We describe CPC in more detail in section 7.3

Beyond moving from hand-crafted dense staged rewards to sparse rewards, we can also remove task rewards entirely.

One straightforward way to achieve this could be to swap out the "success or failure" sparse reward with a neural network goal recognizer, trained on expert trajectories.

The problem here is, such a network would be frozen during agent training, and the agent could adversarially find blind spots of the goal recognizer, and get imitation reward without solving this task.

In fact, this is what we observe in practice (see failure cases in the appendix).To overcome this issue, we can replace the sparse task reward with another discriminator, whose job is to detect whether an agent has reached a goal state or not.

A goal state can be defined for our purposes as a state in the latter 1/M proportion of the expert demonstration.

In our experiments we used M = 3.

The modified reward function then becomes DISPLAYFORM0 where D goal is the secondary goal discriminator network.

It does not share weights with D, but it is also a single-layer network that operates on the same feature space as D. Training D goal is the same as for D, except that the expert states are only sampled from the latter 1/M portion of each demonstration trajectory.

Typically GAIL is viewed as purely an imitation learning method, which bounds the agent performance by how well the demonstrator performs.

However, by training a second discriminator to recognize whether a goal state has been reached, it is possible for an agent to surpass the demonstrator by learning to reach the goal faster, which has already observed when agents are trained with combined imitation and sparse task rewards.

Our environments are visualized in FIG2 .

The first consists of a Kinova Jaco arm, and two blocks on a tabletop.

The arm has 9 degrees of freedom: six arm joints and three actuated fingers.

Policies control the arm by setting the joint velocity commands, producing 9-dimensional continuous velocities in the range of [-1, 1] at 20Hz.

The observations are 128x128 RGB images.

The hand-crafted reward functions (sparse and dense staged) are described in section 7.4.To collect demonstrations we use a SpaceNavigator 3D motion controller.

A human operator controlled the jaco arm with a position controller, and gathers 500 episodes of demonstration for each task including observations, actions, and physical states.

Another 500 trajectories (used for validation purposes) were gathered by a different human demonstrator.

A dataset of 30 "non-expert" trajectories (used for CPC diagnostics) were collected by performing behaviors other than stacking, such as random arm motions, lifting and dropping blocks, and stacking in an incorrect orientation.

The second environment is a 2D walker from the DeepMind control suite BID33 .

To collect demonstrations, we trained a D4PG agent from proprioceptive states to match a target velocity.

Our agents use 64 ?? 64 pixel observations of 200 expert demonstrations of length 300 steps.

For all of the D4PG agents we used the hyperparameter settings listed in section 7.5.

In this section we compare our imitation method to a comparable D4PG agent on dense and sparse reward, and to comparable GAIL agents with discriminator networks operating on pixels directly.

Figure 4 shows that our proposed method using a tiny adversary compares favorably.

The i th row and j th column in the matrices display the CPC's probability of the j th frame being j steps away from the i th frame.

Visualizations are averaged over all the expert and non-expert trajectories.

Note that CPC models the future observations well for expert sequences,but not nonexpert.

(c) shows that conditioning on k-step predictions improves the performance of our method on stacking tasks, when the discriminator also uses CPC embeddings.

In figure 4 , we see that D4PG with sparse rewards never takes off due to the complexity of exploration in this task, whereas with dense rewards its learning pace is very slow.

However our imitation methods learn very quickly with superior performance despite the fact that they only utilize sparse rewards.

Using our method, the agent using value network features takes off more quickly than with CPC features, though they reach to a comparable performance towards the end.

The conventional GAIL from pixels perform very poorly, and GAIL with tiny adversaries on random projections achieves limited success.

Note that with CPC features, which are of dimension 128, the discriminator network has only 128 parameters.

The value network features are 2048-dimensional.

One possible reason that GAIL value features worked while pixel features did not could be due to the regularizing effect of norm clipping applied in the critic optimizer.

To check for this, we evaluated another agent indicated by "GAIL -pixels + clip" which performs norm clipping via tf.clip_by_global_norm(..., 40)}as is done in the critic.

We find that as in the pixel case, this does not result in success in either Jaco (figure 4) or Walker2D (figure 7a).In addition to using CPC features as input to the discriminator, we can also try to make use of the temporal predictions made by CPC.

At each step, one can query CPC about what the expert state would look like several steps in the future, starting from the current state.

Figure 5 visualizes CPC features learned from the 500 training trajectories on the held out validation trajectories.(a) (b) Figure 6 : Jaco stacking ablation experiments.

Left: We find that adding layers to the discriminator network does not improve performance, and not doing early termination hurts performance.

Right: In the case of 120 demonstrations one of the three seeds failed to take off.

However, even with 60 demonstrations, our agent can learn stacking equivalently well as with 500.

In the previous Jaco arm experiments we showed that learning a discriminator directly on the pixels resulted in poor performance.

In the first ablation experiment we determine if a tiny (linear) discriminator is the optimal choice for imitation learning, or if a deeper network on these features can improve the results.

Figure 6(a) shows the effect of the number of layers: as the discriminator becomes more powerful the performance actually degrades, indicating the advantage of a small discriminator on a meaningful representation.

In Section 3 we introduced an early termination criterion to stop an episode when the discriminator score becomes too low.

Figure 6 (a) shows that that when early stopping is disabled, the model learns a lot slower.

To understand better why this helps learning we plot the average episode length during training in Figure 7 .

In the beginning of training the discriminator is not good at distinguishing expert trajectories from agent trajectories yet which is why the episode length is high.

After a while (2000+ episodes) most of the trajectories get stopped early on in the episode.

From 6000 episodes onwards the agent becomes better at imitating the expert and the episodes take longer.

The same figure shows the task and imitation reward (which are scaled between 0 and 1).In a third ablation experiment we evaluate the data efficiency of the proposed method in terms of expert demonstrations.

Figure 6 (b) visualizes the performance with 60, 120, 240 and 500 demonstrations, showing that even 60 demonstrations is enough to get good performance.

We believe the result with 120 demos is an outlier: one of the three random seeds did not take off.

Figure 7 a shows results on the planar walker.

As in the Jaco experiments, the conventional GAIL on pixels with and without norm clipping did not learn.

Both our proposed method using value network features and using random projections learn to run.

Videos of the trained agent are included in the supplementary videos anonymously linked in the appendix.

In this section we show results for agents trained without any rewards, as described in section 3.3.

We used expert states in the final 1/3rd of each sequence as positive examples for D goal , and set ?? = 0.5 in the imitation reward function.

Figure 8 (left) shows different runs (each with a different random seeds), showing that two out of five runs were able to learn without providing any task reward.

The best no-task-reward agent seed achieved a success rate of 55%.

In the top row, the agent learns to stack in a more efficient way than the demonstrator, taking under 2s while the human teloperator takes up to 30s.

In the bottom row, we see an agent exploit, in which the top block is rolled to the background to give the appearance of a completed stack, without actually stacking.

The idea of leveraging expert demonstrations to improve agent performance has a long history in robotics BID5 BID19 BID24 .

Similar in spirit to very recent work, BID28 show that by priming a Q-function on expert demonstrations, their Q-learning agent can perform cart-pole swing up after a single episode of training.

However, our task is different in that we do not assume access to both states and actions, only pixel observations, so we cannot prime the value function on expert demonstrations in the same way.

A large amount of work in the past several years has tried to extend the success of deep learning beyond discriminative tasks in computer vision, towards taking actions to interact with simulated or real environments.

Imitation learning is an attractive setting from this perspective, because the inputs and outputs to the learning problem strongly resemble those found in classification and regression problems at which deep networks already excel in solving.

A simple yet effective approach is supervised imitation BID27 , also called behavioral cloning.

BID10 extend this to one-shot imitation, in which a behaviors are inferred from single demonstrations via an encoder network, and a state-to-action decoder with an attention mechanism replicates the desired behavior on a new problem instance.

This approach is able stack blocks into target arrangements from scripted demonstrations on a simulated Fetch robot.

Instead of using attention, BID12 ) use a gradient-based meta learning approach to perform one-shot learning of observed behaviors.

This approach learns to pick and place novel objects into containers given video demonstrations on a PR2 robot.

Our approach is different mainly in that we aim for the agent to learn by interacting with the environment rather than supervised learning.

A major downside of behavioral cloning is that, if the agent ventures into very different states than were observed in the expert trajectories, this results in cascading failures.

This necessitates a large number of demonstrations and limits how far the agent can generalize.

It also requires access to demonstrator actions, which may not always be available, and tend to be tightly coupled to the particular robot and teloperation setup used to gather demonstrations.

Instead of using behavior cloning, BID38 BID25 ; BID0 propose inverse reinforcement learning (IRL), in which they learn a reward function from demonstrations, and then use reinforcement learning to optimize that learned reward.

developed deep Q-Learning from demonstration (DQfD), in which expert trajectories are added to experience replay and jointly used to train agents along with their own experiences.

This was later extended by BID26 to better handle sparse-exploration Atari games.

BID36 develop deterministic policy gradients from demonstration (DPGfD), and similarly populate a replay buffer with both expert data and agent experience, and show that through imitation it can solve a peg insertion task on a real robot without access to any dense shaped reward.

However, both of these methods still require access to the expert actions in order to learn.

Following the success of Generative Adversarial Networks BID14 in image generation, GAIL BID16 applies adversarial learning to the problem of imitation.

Although many variants are introduced in the literature BID22 BID13 BID37 BID6 , making GAIL work for high-dimensional input spaces, particularly for hard exploration problems with sparse rewards, remains a challenge.

Our major contribution, which is complementary, is that through the use of minimal adversaries on top of learned features we can successfully solve sparse reward tasks with high-dimensional input spaces.

Another line of work is learning compact representations for imitation learning using expert observations (i.e. no actions).

Both BID30 and BID3 learn self-supervised features from third person observations in order to mitigate the domain gap between first and third person views.

At the end, they both utilize these feature spaces for closely tracking a single expert trajectory, whereas we use our features for learning the task using all available expert trajectories through GAIL.

Our target is not to track a single trajectory, but to generalize all possible initializations of a hard exploration task.

We utilize both static self-supervised features such as contrastive predictive coding BID35 , and dynamic value network features which constantly change during the learning process, and show that both can be used to successfully train block stacking agents from sparse rewards on pixels.

7.1 SUPPLEMENTARY VIDEOS Videos of our learned agents can be viewed at the following anonymized web site: https://sites.google.com/view/iclr2019-visual-imitation/home

In figure 4 we include a dashed line labeled "BC" indicating the average performance of a pure supervised baseline model that regresses expert actions from pixel observations.

The behavior cloning model consists of the same residual network pixel encoder architecture that we use for D4PG, followed by a 128-dimension LSTM, followed by a final 512-dimensional linear layer, ELU, then the output actions.

The stacking accuracy is approximately 15%.

Figure 9 (left) shows a visualization of CPC on video data.

The model consists of two parts: the encoder which maps every observation x t to a latent representation z t = g enc (x t ) (or target vector) and the autoregressive model which summarizes the past latents into a context vector c t = g ar (z ???t ).

Both optimize the same loss:

, where z 1 , z 2 . . .

z N are negative samples, which in the case of CPC are usually drawn from other examples or timesteps in the minibatch.

The weights W k for the bilinear mapping z t+k W k c t are also learned and depend on k, the number of latent steps the model is predicting in the future.

By optimizing L CPC the mutual information between z t+k and c t is maximized, which results in the variables that the context and target have in common being linearly embedded into compact representations.

This is especially useful for extracting slow features, for example when z t+k and c t are far apart in time.

Figure 9 : Overview of our proposed approach.

Left: model learning via contrastive predictive coding (CPC).

Right: After training and freezing CPC expert model, we train the agent, which makes use of CPC future predictions.

Note that we never need to predict in pixel space.

Our reward functions are slightly modified from BID37 .

For evaluation each episode lasts 500 time steps; we do not use early stopping.

Dense staged reward: We define five stages and their rewards to be initial (0), reaching the orange block (0.125), lifting the orange block (0.25), stacking the orange block onto the pink block (1.0), and releasing the orange block and lifting the arm (1.25).Sparse reward: We define two stages and their rewards to be initial (0) and stacking the orange block onto the pink block (1.0).

There are no rewards for reaching, lifting or releasing.

Actor and critic share a residual network with twenty convolutional layers (3x3 convolutions, four 5-layer blocks with 64, 128, 256 and 512 channels), instance normalization (Ulyanov et al.) and exponential linear units BID9 between layers and a fully connected layer with layer normalization BID4 .

Both actor and critic use independent three-layer fully connected networks (2x1024 and an output layer) with exponential linear unit between layers.

Instead of using a scalar state-action value function, we adopt Distributional Q fuctions where Q(s, a|??) = EZ(s, a|??) for some random variable Z. In this paper, we adopt a categorical representation of Z such that Z(s, a|??) = z i w.p.

p i ??? exp(?? i (s, a|??)) for i ??? {0, ?? ?? ?? , l ??? 1}.

The z i 's are fixed atoms bounded between V min and V max such that z i = V min + i

Vmax???Vmin l???1 .Again following (Barth-Maron et al., 2018), we compute our bootstrap target with N-step returns.

Given sub-sequence s t , a t , {r t , ?? ?? ?? , r t+N ???1 }, s t+N , we construct a bootstrap target Z such that Z = z i w.p.

p i ??? exp(?? i (s t+K , ??(s t+K |?? )|?? )) where z i = N ???1 n=0 ?? j r t+n + ?? N z i .

Notice Z is not likely have the same support as Z. We therefore adopt the categorical projection ?? proposed in BID8 .

The loss function for training our distributional value functions is L N (??) = E (st,at,{rt,?????? ,r t+N ???1 },s t+N )???B [H(??(Z ), Z(s t , a t |??))] ,where H represents cross entropy.

Finally, we use distributed prioritized experience replay BID29 to further increase stability and learning efficiency.

@highlight

Imitation from pixels, with sparse or no reward, using off-policy RL and a tiny adversarially-learned reward function.

@highlight

The paper proposes to use a "minimal adversary" in generative adversarial imitation learning under high-dimensional visual spaces.

@highlight

This paper aims at solving the problem of estimating sparse rewards in a high-dimensional input setting.
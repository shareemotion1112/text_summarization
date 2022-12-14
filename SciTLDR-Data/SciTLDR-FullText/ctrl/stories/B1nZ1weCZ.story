One of the long-standing challenges in Artificial Intelligence for learning goal-directed behavior is to build a single agent which can solve multiple tasks.

Recent progress in multi-task learning for goal-directed sequential problems has been in the form of distillation based learning wherein a student network learns from multiple task-specific expert networks by mimicking the task-specific policies of the expert networks.

While such approaches offer a promising solution to the multi-task learning problem, they require supervision from large expert networks which require extensive data and computation time for training.

In this work, we propose an efficient multi-task learning framework which solves multiple goal-directed tasks in an on-line setup without the need for expert supervision.

Our work uses active learning principles to achieve multi-task learning by sampling the harder tasks more than the easier ones.

We propose three distinct models under our active sampling framework.

An adaptive method with extremely competitive multi-tasking performance.

A UCB-based meta-learner which casts the problem of picking the next task to train on as a multi-armed bandit problem.

A meta-learning method that casts the next-task picking problem as a full Reinforcement Learning problem and uses actor-critic methods for optimizing the multi-tasking performance directly.

We demonstrate results in the Atari 2600 domain on seven multi-tasking instances: three 6-task instances, one 8-task instance, two 12-task instances and one 21-task instance.

Deep Reinforcement Learning (DRL) arises from the combination of the representation power of Deep learning (DL) BID10 BID3 ) with the use of Reinforcement Learning (RL) BID28 objective functions.

DRL agents can solve complex visual control tasks directly from raw pixels BID6 BID12 BID24 BID11 BID23 BID13 BID29 BID30 BID2 BID26 BID7 .

However, models trained using such algorithms tend to be task-specific because they train a different network for different tasks, however similar the tasks are.

This inability of the AI agents to generalize across tasks motivates the field of multi-task learning which seeks to find a single agent (in the case of DRL algorithms, a single deep neural network) which can perform well on all the tasks.

Training a neural network with a multi-task learning (MTL) algorithm on any fixed set of tasks (which we call a multi tasking instance (MTI)) leads to an instantiation of a multi-tasking agent (MTA) (we use the terms Multi-Tasking Network (MTN) and MTA interchangeably).

Such an MTA would possess the ability to learn task-agnostic representations and thus generalize learning across different tasks.

Successful DRL approaches to the goal-directed MTL problem fall into two categories.

First, there are approaches that seek to extract the prowess of multiple task-specific expert networks into a single student network.

The Policy Distillation framework BID20 and Actor-Mimic Networks BID16 fall into this category.

These works train k task-specific expert networks (DQNs ) and then distill the individual task-specific policies learned by the expert networks into a single student network which is trained using supervised learning.

While these approaches eventually produce a single a network that solves multiple tasks, individual expert networks must first be trained, and this training tends to be extremely computation and data intensive.

The second set of DRL approaches to multi-tasking are related to the field of transfer learning.

Many recent DRL works BID16 BID21 BID18 BID4 attempt to solve the transfer learning problem.

Progressive networks BID21 ) is one such framework which can be adapted to the MTL problem.

Progressive networks iteratively learn to solve each successive task that is presented.

Thus, they are not a truly on-line learning algorithm.

Progressive Networks instantiate a task-specific column network for each new task.

This implies that the number of parameters they require grows as a large linear factor with each new task.

This limits the scalability of the approach with the results presented in the work being limited to a maximum of four tasks only.

Another important limitation of this approach is that one has to decide the order in which the network trains on the tasks.

In this work we propose a fully on-line multi-task DRL approach that uses networks that are comparable in size to the single-task networks.

In particular, our contributions are the following: 1) We propose the first successful on-line multi-task learning framework which operates on MTIs that have many tasks with very visually different high-dimensional state spaces (See FIG0 for a visual depiction of the 21 tasks that constitute our largest multi-tasking instance).

2) We present three concrete instantiations of our MTL framework: an adaptive method, a UCB-based meta-learning method and a A3C based meta-learning method.

3) We propose a family of robust evaluation metrics for the multi-tasking problem and demonstrate that they evaluate a multi-tasking algorithm in a more sensible manner than existing metrics.

4) We provide extensive analyses of the abstract features learned by our methods and argue that most of the features help in generalization across tasks because they are task-agnostic.

5) We report results on seven distinct MTIs: three 6-task instances, one 8-task instance, two 12-task instances and one 21-task instance.

Previous works have only reported results on a single MTI.

Our largest MTI has more than double the number of tasks present in the largest MTI on which results have been published in the Deep RL literature BID20 .

6) We hence demonstrate how hyper-parameters tuned for an MTI (an instance with six tasks) generalize to other MTIs (with up to 21 tasks).

In this section, we introduce the various concepts needed to explain our proposed framework and the particular instantiations of the framework.

A large class of decision problems can be cast as multi-armed bandit problem wherein the goal is to select the arm (or action) that gives the maximum expected reward.

An efficient class of algorithms for solving the bandit problem are the UCB algorithms BID1 BID0 .

The UCB algorithms carefully track the uncertainty in estimates by giving exploration bonuses to the agent for exploring the lesser explored arms.

Such UCB algorithms often maintain estimates for the average reward that an arm gives, the number of times the arm has been pulled and other exploration factors required to tune the exploration.

In the case of non-stationary bandit problems, it is required for such average estimates to be non-stationary as well.

For solving such non-stationary bandit problems, Discounted UCB style algorithms are often used BID8 BID22 BID5 .

In our UCB-based meta-learner experiments, we use the Discounted UCB1-Tuned+ algorithm BID8 .

One of the ways of learning optimal control using RL is by using (parametric) actor critic algorithms BID9 .

These approaches consist of two components: an actor and a critic.

The actor is a parametric function: ?? ??a (a t |s t ) mapping from the states to the actions according to which the RL agent acts (?? a are the parameters of the policy/actor).

A biased but low-variance sample estimate for the policy gradient is: ??? ??a log ?? ??a (a t |s t )(Q(s t , a t ) ??? b(s t )) where a t is the action executed in state s t .

Q(s t , a t ) is the action value function.

b(s t ) is a state-dependent baseline used for reducing the variance in policy gradient estimation.

The critic estimates Q(s t , a t ) and possibly the state dependent baseline b(s t ).

Often, b(s t ) is chosen to be the value function of the state, V (s t ).

We can also get an estimate for Q(s t , a t ) as r t+1 + ??V (s t+1 ) where ?? is the discounting factor.

The critic is trained using temporal difference learning BID27 algorithms like TD(0) BID28 .

The objective function for the critic is: DISPLAYFORM0 2 .

In all our experiments, we use the Asynchronous Advantage Actor-Critic Algorithm (A3C) as our base RL algorithm.

In the domain of Multi-Task Learning, the goal is to obtain a single agent which can perform well on all the k tasks in a given fixed MTI.

The performance metrics we use are presented in Section 3.1.

While there might be transfer happening between the instances while learning, it is assumed that the agent at every point of time has access to all the tasks in the multi-task instance.

The MTA acts in an action space that is the union of the action spaces of the individual tasks.

We assume that the input to the MTA is such that the state features are the same, or at the least the same feature learning mechanism will work across all the tasks.

In this work, we demonstrate the effectiveness of our MTA on games from Arcade Learning Environment BID32 .

While these games are visually distinct, the same feature learning mechanism, namely a Convolutional Neural Network, works well across all the games.

It is also important to note that the identity of the current task is not part of the input to the MTN during training on a task.

In contrast, existing methods such as BID20 give the identity of the task that the MTN is being trained on as an input.

Thus, our MTN must implicitly figure out the identity of the task just from the input features and the dynamics of the task.

Previous works BID16 define the performance of an MTA on an MTI as the arithmetic mean (p am ) of the normalized game-play scores of the MTA on the various tasks in the MTI.

Let ?? i be the game-play score of an MTA in task i, h i be the target score in task i (potentially obtained from other published work).

We argue that p am is not a robust evaluation metric.

An MTA can be as good as the target on all tasks and achieves p am = 1.

However, a bad MTA can achieve p am = 1 by being k (total number of tasks) times better than the target on one of the tasks and being as bad as getting 0 score in all the other tasks.

We define a better performance metric: q am (Equation 1).

It is better because the MTA needs to be good in all the tasks in order to get a high q am .

We also define, q gm , the geometric-mean based and q hm , the harmonic-mean based performance metrics.

DISPLAYFORM0 We evaluate the MTAs in our work on p am , q am , q gm , q hm .

TAB1 reports the evaluation on q am .

Evaluations on the other metrics have been reported in Appendix E. In this section, we introduce our framework for MTL by first describing a naive framework for on-line MTL in the first subsection and then presenting our approach as extension to this framework in the second subsection.

To avoid the computational costs of training single-task expert networks, we assume that the MTA does not have access to expert networks' predictions.

Previous approaches to MTL: BID16 BID20 have been off-line in nature.

Before we describe the frameworks, we outline how an on-line algorithm for MTL takes inputs from different tasks.

When a MTN is trained using an on-line MTL algorithm, it must be trained by interleaving data/observations from all the tasks in the MTI.

An on-line MTL algorithm must decide once every few time steps, the next task on which the MTN is to be trained .

We call such decision steps as task decision steps.

Note that task decision steps can be event driven (eg: at the end of every episode) or time driven (eg: once every k time steps).

The BA3C is a simple on-line MTL algorithm.

The MTN is a single A3C network which is trained by interleaving observations from k tasks in an on-line fashion.

The task decision steps in BA3C occur at the end of every episode of training for the MTN.

The next task for the MTN to train on is decided uniformly at random.

The full training algorithm for BA3C is given as Algorithm 2 in Appendix C. We believe that the lackluster performance of BA3C (this has been reported in BID16 as well) is because of the probability distribution according to which the agent decides which task to train on next.

In BA3C's case, this is the uniform probability distribution.

We posit that this distribution is an important factor in determining the multi-tasking abilities of a trained MTA.

We demonstrate our framework with the help of the LSTM BID33 version of the A3C algorithm .

Our framework is inspired by active learning principles BID17 BID31 BID25 .

We call our framework A4C-Active sampling A3C.

The overarching idea of our work is simple and effective: A multi-task learning algorithm can achieve better performance with fewer examples if it is allowed to decide which task to train on at every task decision step (thus "actively sampling" tasks) as opposed to uniformly sampling.

More precisely, it is better if the agent decides to train on tasks which it is currently bad at.

This decision can be made based on a heuristic or using another meta-learner.

We explore two different approaches for the meta-learner -posing the meta-learning problem as an multi-arm bandit problem and as a full RL problem.

DISPLAYFORM0 n ??? Number of episodes used for estimating current performance in any task T i

s i ??? List of last n scores that the multi-tasking agent scored during training on task T i

p i ??? Probability of training on task T i next 6:amta ??? The Active Sampling multi-tasking agent 7:meta_decider ??? An instantiation of our active learning based task decision framework 8:for train_steps:0 to MaxSteps do 9:for i in {1, ?? ?? ?? , |T |} do 10: DISPLAYFORM0 DISPLAYFORM1 architecture is the same as that of a single-task network.

The important improvement is in the way the next task for training is selected.

Instead of selecting the next task to train on uniformly at random, our framework maintains for each task T i , an estimate of the MTN's current performance (?? i ) as well as the target performance (h i ).

These numbers are then used to actively sample tasks on which the MTA's current performance is poor.

In all the methods we ensure that all tasks continue to be selected with non-zero probability during the learning.

We emphasize that no single-task expert networks need to be trained for our framework; published scores from other papers (such as BID26 for Atari) or even Human performance can be used as target performance.

In case the task-decision problem is cast as a full RL problem, there are various definitions of state and reward that can be chosen.

In what follows, we present 3 different instantiations of our A4C framework with particular choices of states and rewards.

We experimented with other choices for state and reward definitions and we report the ones with the best performance in our experiments.

There could be other agents under the A4C framework, some potentially better than our instantiations with other choices of state and reward functions and design of such agents are left as future work.

We refer to this method as A5C (Adaptive Active-sampling A3C).

The task decision steps in A5C occur at the end of every episode of training of the MTN.

Among the methods we propose, this is the only method which does not learn the sampling distribution p (Line 15, Algorithm 1).

It computes an estimate of how well the MTN can solve task T i by calculating m i = hi?????i hi for each of the tasks.

The probability distribution for sampling next tasks (at task decision steps) is then computed as: DISPLAYFORM0 , where ?? is a temperature hyper-parameter.

Intuitively, m i is a task-invariant measure of how much the current performance of the MTN lags behind the target performance on task T i .

A higher value for m i means that the MTN is bad in Task T i .

By actively sampling from a softmax probability distribution with m i as the evidence, our adaptive method is able to make smarter decisions about where to allocate the training resources next (which task to train on next).

This method is referred to as UA4C (UCB Active-sampling A3C).

The task decision steps in UA4C occur at the end of every episode of training for the MTN.

In this method, the problem of picking the next task to train on is cast as a multi-armed bandit problem with the different arms corresponding to the various tasks in the MTI being solved.

The reward for the meta-learner is defined as: r = m i where i is the index of the latest task that was picked by the meta-learner, and that the MTN was trained on.

The reason for defining the reward in this way is that it allows the meta-learner to directly optimize for choosing those tasks that the MTN is bad at.

For our experiments, we used the Discounted-UCB1-tuned+ algorithm BID8 .

We used a discounted-UCB algorithm because the bandit problem is non-stationary (the more a task is trained on, the smaller the rewards corresponding to the choice of that task become).

We also introduced a tunable hyperparameter ?? which controls the relative importance of the bonus term and the average reward for a given task.

Using the terminology from BID8 , the upper confidence bound that the meta-learner uses for selecting the next task to train on is: DISPLAYFORM0

We refer to this method as EA4C (Episodic meta-learner Active-sampling A3C).

The task decision steps in EA4C occur at the end of every episode of training for the MTN (see Appendix A for a version of EA4C which makes task-decision steps every few time steps of training).

EA4C casts the problem of picking the next task to train on as a full RL problem.

The idea behind casting it in this way is that by optimizing for the future sum of rewards (which are defined based on the multi-tasking performance) EA4C can learn the right sequence in which to sample tasks and hence learn a good curriculum for training MTN.

The EA4C meta-learner consists of an LSTM-A3C-based controller that learns a task-sampling policy over the next tasks as a function of the previous sampling decisions and distributions that the meta-learner has used for decison making.

Reward definition: Reward for picking task T j at (meta-learner time step t) is defined as: DISPLAYFORM0 where m i was defined in Section 4.2.1.

L is the set of worst three tasks, according to 1 ??? m i = ??i hi (normalized task performance).

?? is a hyper-parameter.

First part of the reward function is similar to that defined for the UCB meta-learner in Section 4.2.2.

The second part of the reward function ensures that the performance of the MTN improves on worst three tasks and thus increases the multi-tasking performance in general.

State Definition:

The state for the meta-learner is designed to be descriptive enough for it to learn the optimal policy over the choice of which task to train the MTN on next.

To accomplish this, we pass a 3k length vector to the meta-learner (where k is the number of tasks in the MTI) which is a concatenation of 3 vectors of length k. The first vector is a normalized count of the number of times each of the tasks has been sampled by the meta-learner since the beginning of training.

The second vector is the identity of the task sampled at the previous task decision step, given as a one-hot vector.

The third vector is the previous sampling distribution over the tasks that the meta learner had used to select the task on which the MTN was trained, at the last task decision step.

Our definition of the meta-learner's state is just one of the many possible definitions.

The first and the third vectors have been included to make the state descriptive.

We included the identity of the latest task on which the MTN was trained so that the meta-learner is able to learn policies which are conditioned on the actual counts of the number of times a task was sampled.

The A3C MTNs we use have the same size and architecture as a single-task network (except when our MTL algorithms need to solve MT7, which has 21 tasks) and these architectural details are the same across different MTL algorithms that we present.

The experimental details are meticulously documented in Appendix B. All our MTAs used a single 18-action shared output layer across all the different tasks in an MTI, instead of different output head agents per task as used in BID20 .

Appendix I contains our empirical argument against using such different-output head MTAs.

It is important to note that previous works BID16 BID20 have results only on a single MTI.

TAB3 (in Appendix B) contains the description of the seven MTIs presented to MTAs in this work.

Hyper-parameters for all multi-tasking algorithms in this work were tuned on only one MTI: MT1.

If an MTI consists of k tasks, then all MTNs in this work were trained on it for only k ?? 50 million time steps, which is half of the combined training time for all the k tasks put together (task-specific agents were trained for 100 million time steps in BID26 ).

All the target scores in this work were taken from TAB5 of BID26 .

We reiterate that for solving the MTL problem, it is not necessary to train single-task expert networks for arriving at the target scores; one can use scores published in other works.

We conducted experiments on seven different MTIs with number of constituent tasks varying from 6 to 21.

All hyper-parameters were tuned on MT1, an MTI with 6 tasks.

We demonstrate the robustness of our framework by testing all our the algorithms on seven MTIs including a 21-task MTI (MT7) which is more than double the number of tasks any previous work BID20 has done multi-tasking on.

Description of the MTIs used in this work is provided in TAB3 in Appendix B. We have performed three experiments to demonstrate the robustness of our method to the target scores chosen.

These experiments and the supporting details have been documented in Appendix G. We now describe our findings from the general game-play experiments as demonstrated in TAB1 .

We observe that all our proposed models beat the BA3C agent by a large margin and obtain a performance of more than double the performance obtained by the BA3C agent.

Among the proposed models, on MT1 (where the hyperparameters were tuned), A5C performs the best.

However, the performance of UA4C and EA4C is only slightly lower than that of A5C.

We accredit this relatively higher performance to the fact that there are many hyper-parameters to tune in the UA4C and the EA4C methods unlike A5C where only the temperature hyperparameter had to be tuned.

We tuned all the important hyperparameters for UA4C and EA4C.

However, our granularity of tuning was perhaps not very fine.

This could be the reason for the slightly lower performance.

The UA4C agent, however, generalizes better than A5C agent on the larger MTIs (MT5 & MT6).

Also, the performance obtained by EA4C is close to that of A5C and UA4C in all the multitasking instances.

The MTI MT4 has been taken from BID16 .

On MT4, many of our agents are consistently able to obtain a performance close to q am = 0.9.

It is to be noted that Actor Mimic networks are only able to obtain q am = 0.79 on the same MTI.

The most important test of generalization is the 21-task instance (MT7).

EA4C is by far the best performing method for this instance.

This clearly demonstrates the hierarchy of generalization capabilities demonstrated by our proposed methods.

At the first level, the EA4C MTA can learn task-agnostic representations which help it perform well on even large scale MTIs like MT7.

Note that the hyper-parameters for all the algorithms were tuned on MT1, which is a 6-task instance.

That the proposed methods can perform well on much larger instances with widely visually different constituent tasks without retuning hyperparameters is proof for a second level of generalization: the generalization of the hyper-parameter setting across multi-tasking instances.

An important component of our framework are the target scores for the different tasks.

There are two concerns that one might have regarding the use of target scores: 1) Access to target scores implies access to trained single-task agents which defeats the purpose of online multi-task learning.2) The method of training such an active-sampling based agent on new tasks where the tasks have never been solved.

We aim to address both the concerns regarding the use of target scores in our proposed framework.

We reiterate that the access to target scores does not imply access to trained single-task agents.

We would expect that any researcher who uses our framework would also use published resources as the source of target scores, rather than training single-task networks for each of the tasks.

In some cases, one might want to build an MTA prior to the existence of agents that can solve each of the single tasks.

In such a case, it would be impossible to access target scores because the tasks in question have never been solved.

In such cases, we propose to use a doubling of targets paradigm (demonstrated using Doubling UCB-based Active-sampling A3C (DUA4C) in Algorithm 7) to come up with rough estimates for the target scores and demonstrate that our doubling-target paradigm can result in impressive performances.

The doubling target paradigm maintains an estimate of target scores for each of the tasks that the MTA needs to solve.

As soon as the MTA achieves a performance that is greater than or equal to the estimated target, the estimate for the target is doubled.

The idea is that in some sense, the agent can keep improving until it hits a threshold, and then the threshold is doubled.

All the hyper-parameters found by tuning UA4C on MT1 were retained.

None of the hyper-parameters were retuned.

This thus represents a setup which isn't very favorable for DUA4C.

Figure 4 depicts the evolution of the raw performance (game-score) of the DUA4C agent trained with doubling target estimates instead of single-task network's scores.

The performance of DUA4C on different MTIs is contained in TAB2 .

Results on other metrics along with training curves on various MTIs are shown in Appendix K. We observe that even in this unfavorable setup, the performance of DUA4C is impressive.

The performance could possibly improve if hyper-parameters were tuned for this specific paradigm/framework.

This section analyses the reasons as to why our MTL framework A4C performs much better than the baseline (BA3C).

Based on the experiments that follow, we claim that it is the task-agnostic nature of the abstract features that are learned in this work which allow our proposed algorithms to perform very well.

An MTA can potentially perform well at the different tasks in an MTI due to the sheer representational power of a deep neural network by learning task-specific features without generalizing across tasks.

We empirically demonstrate that this is not the case for the agents proposed in our work.

The experiments in this section analyze the activation patterns of the output of the LSTM controller.

We call a neuron task-agnostic if it is as equally responsible for the performance on many of the tasks.

Before we show the task agnostic nature of the neurons in our A4C agents, we present an intuition as to how our agents are able to overcome the problem of catastrophic forgetting.

We first note that in all the agents defined under the A4C framework, a task has a higher probability to get sampled if the m i for the task is higher.

Forgetting is avoided in our agents by the virtue of the sampling procedure used by the meta-learners.

Say m 1 is largest among all m i '

s. This causes task 1 to get sampled more.

Since the agent is training on task 1, it gets better at it.

This leads to m 1 getting smaller.

At some point if m 2 (some other task) becomes larger than m 1 , task 2 will start getting sampled more.

At some later point, if performance on task 1 degrades due to the changes made to the network, then m 1 will again become larger and thus it'll start getting sampled more.

It can now be argued that performance estimates (m i ) could be stale for some tasks if they don't get sampled.

While it is true that we don't update the score of a task till it is sampled again, we need to keep in mind that the sampling of the tasks is done from a distribution across tasks.

As a result, there is still finite probability of every task getting sampled.

This is analogous to exploration in RL.

Note that if the probability of sampling such tasks was so low that it would practically be impossible to sample it again, this would imply that performance on the task was great.

What we have observed through comprehensive experimentation is that once such good performance has been achieved on some task, degradation does not happen.

In this set of experiments, our agents trained on M T 1 are executed on each of the constituent tasks for 10 episodes.

A neuron is said to fire for a time step if its output has an absolute value of 0.3 or more.

Let f ij denote the fraction of time steps for which neuron j fires when tested on task i. Neuron j fires for the task i if f ij ??? 0.01.

We chose this low threshold because there could be important neurons that detect rare events.

FIG4 demonstrates that for A4C, a large fraction of the neurons fire for a large subset of tasks and are not task-specific.

It plots neuron index versus the set of fraction of time steps that neuron fires in, for each task.

The neurons have been sorted first by |{i : f ij ??? 0.01}| and then by i f ij .

Neurons to the left of the figure fire for many tasks whereas those towards the right are task-specific.

The piece-wise constant line in the figure counts the number of tasks in which a particular neuron fires with the leftmost part signifying 6 tasks and the rightmost part signifying zero tasks.

Appendix H contains the analysis for all MTIs and methods.

We introduce a way to analyze multitasking agents without using any thresholds.

We call this method the turnoff-analysis.

Here, we force the activations of one of the neurons in LSTM output to 0 and then observe the change in the performances on individual tasks with the neuron switched off.

This new score is then compared with the original score of the agent when none of the neurons were switched off and an absolute percentage change in the scores is computed.

These percentage changes are then normalized for each neuron and thus a tasks versus neuron matrix A is obtained.

The variance of column i of A gives a score for the task-specificity of the neuron i.

We then sort the columns of A in the increasing order of variance and plot a heatmap of the matrix A. We conclude from Figure 6 that A4C agents learn many non task-specific abstract features which help them perform well across a large range of tasks.

Our experiments demonstrate that A4C agents learn many more task-agnostic abstract features than the BA3C agent.

Specifically, observe how uniformly pink the plot corresponding to the UA4C agent is, compared to the BA3C plot.

We propose a framework for training MTNs which , through a form of active learning succeeds in learning to perform on-line multi-task learning.

The key insight in our work is that by choosing the task to train on, an MTA can choose to concentrate its resources on tasks in which it currently performs poorly.

While we do not claim that our method solves the problem of on-line multi-task reinforcement learning definitively, we believe it is an important first step.

Our method is complementary to many Figure 6 : Turn Off analysis heap-maps for the all agents.

For BA3C since the agent scored 0 on one of the games, normalization along the neuron was done only across the other 5 games.of the existing works in the field of multi-task learning such as: BID20 and BID16 .

These methods could potentially benefit from our work.

Another possible direction for future work could be to explicitly force the learned abstract representations to be task-agnostic by imposing objective function based regularizations.

One possible regularization could be to force the average firing rate of a neuron to be the same across the different tasks.

In the EA4C method introduced in Section 4.2.3, the task-decision steps, which also correspond to one training step for the meta-learner, happen at the end of one episode of training on one of the tasks.

For three of the multi-tasking instances (MT1, MT2 and MT3) that we experimented with, the total number of training steps was 300 million.

Also, an average episode length of tasks in these instances is of the order of 1000 steps.

Hence, the number of training steps for the meta-learner in EA4C is of the order of 3 ?? 10 5 .

This severely restricts the size of the neural network which is used to represent the policy of the meta-learner.

To alleviate this problem we introduce a method called FA4C: Fine-grained meta-learner Activesampling A3C.

The same architecture and training procedure from EA4C is used for FA4C, except for the fact that task decision steps happen after every N steps of training the multi-tasking network, instead of at the end of an episode.

The value of N was fixed to be 20.

This is the same as the value of n used for n-step returns in our work as well as BID26 .

Observe that when the number of training steps for the multi-tasking network is 300 million, the number of training steps for meta-learner is now of the order of 15 million.

This allows the use of larger neural networks for the meta-learner policy as compared to EA4C.

Since we used an LSTM in the neural network representing the multitasking agent's policy, we stored the state of the LSTM cells at the end of these n = 20 training steps for each of the tasks.

This allows us to resume executing any of the tasks after training on one of them for just 20 steps using these cached LSTM state cells.

We now describe the reward function and state construction for FA4C:Reward Function: Since the task decision steps for this method happen after every 20 steps of training the multi-tasking network, the meta-learner needs to be rewarded in a way that evaluates its 20-step task selection policy.

It makes sense to define this reward to be proportional to the performance of the MTN during those 20 time steps, and inversely proportional to the target performance during those 20 time steps.

These target scores have to be computed differently from those used by other methods introduced in this paper since the scores now correspond to performance over twenty time steps and not over the entire episode.

The target scores for a task in FA4C can be obtained by summing the score of a trained single-task agent over twenty time steps and finding the value of this score averaged over the length of the episode.

Concretely, if the single-task agent is executed for k episodes and each episode is of length l i , 1 ??? i ??? k and r i,j denotes the reward obtained by the agent at time step j in episode i where 1 ??? i ??? k, 1 ??? j ??? l i then the averaged 20-step target score is given by (let x i = li 20 ): DISPLAYFORM0 Published as a conference paper at ICLR 2018This design of the target score has two potential flaws: 1) A task could be very rewarding in certain parts of the state space(and hence during a particular period of an episode) whereas it could be inherently sparsely rewarding over other parts.

It would thus make sense to use different target scores for different parts of the episode.

However, we believe that in an expected sense our design of the target score is feasible.2) Access to such fine grained target scores is hard to get.

While the target scores used in the rest of the paper are simple scalars that we took from other published work BID26 , for getting these h f g 's we had to train single-task networks and get these fine grained target scores.

Hopefully such re-training for targets would not be necessary once a larger fraction of the research starts open-sourcing not only their codes but also their trained models.

The overall reward function is the same as that for EA4C (defined in Equation 2) except one change, m i is now defined as: DISPLAYFORM1 where h i,f g is the target score defined in Equation 3 for task T i and ?? i,f g is the score obtained by multi-tasking instance in task T i over a duration of twenty time steps.

State Function: The state used for the fine-grained meta-learner is the same as that used by the episodic meta-learner.

Our experimental results show that while FA4C is able to perform better than random on some multi-tasking instances, on others, it doesn't perform very well.

This necessitates a need for better experimentation and design of fine-grained meta controllers for multi-task learning.

We first describe the seven multi-tasking instances with which we experiment.

We then describe the hyper-parameters of the MTA which is common across all the 5 methods (BA3C, A5C, UA4C, EA4C, FA4C) that we have experimented with, in this paper.

In the subsequent subsections we describe the hyper-parameter choices for A5C, UA4C, EA4C, FA4C.

The seven multi-tasking instances we experimented with have been documented in TAB3 .

The first three instances are six task instances, meant to be the smallest instances.

MT4 is an 8-task instance.

It has been taken from BID16 and depicts the 8 tasks on which BID16 experimented.

We experimented with this instance to ensure that some kind of a comparison can be carried out on a set of tasks on which other results have been reported.

MT5 and MT6 are 12-task instances and demonstrate the generalization capabilities of our methods to medium-sized multi-tasking instances.

Note that even these multi-tasking instances have two more tasks than any other multi-tasking result (Policy distillation BID20 reports results on a 10-task instance.

However, we decided not to experiment with that set of tasks because the result has been demonstrated with the help of a neural network which is 4 times the size of a single task network.

In comparison, all of our results for 6, 8 and 12 task instances use a network which has same size as a single-task network).

Our last set of experiments are on a 21-task instance.

This is in some sense a holy grail of multi-tasking since it consists of 21 extremely visually different tasks.

The network used for this set of experiments is only twice the size of a single-task network.

Hence, the MTA still needs to distill the prowess of 10.5 tasks into the number of parameters used for modeling a single-task network.

Note that this multi-tasking instance is more than twice the size of any other previously published work in multi-tasking.

In this sub-section we document the experimental details regarding the MTN that we used in our experiments.

We used the LSTM version of the network proposed in and trained it using the async-rms-prop algorithm.

The initial learning rate was set to 10 ???3 (found after hyper-parameter tuning over the set {7 ?? 10 ???4 , 10 ???3 }) and it was decayed linearly over the entire training period to a value of 10 ???4 .

The value of n in the n-step returns used by A3C was set to 20.

This was found after hyper-parameter tuning over the set {5, 20}. The discount factor ?? for the discounted returns was set to be ?? = 0.99.

Entropy-regularization was used to encourage exploration, similar to its use in .

The hyper-parameter which trades-off optimizing for the entropy and the policy improvement is ?? (introduced in .

?? = 0.02 was separately found to give the best performance for all the active sampling methods (A5C, UA4C, EA4C, FA4C) after hyper-parameter tuning over the set {0.003, 0.01, 0.02, 0.03, 0.05}. The best ?? for BA3C was found to be 0.01.The six task instances (MT1, MT2 and MT3) were trained for 300 million steps.

The eight task instance (MT4) was trained over 400 million steps.

The twelve task instances (MT5 and MT6) were trained for 600 million steps.

The twenty-one task instance was trained for 1.05 billion steps.

Note that these training times were chosen to ensure that each of our methods was at least 50% more data efficient than competing methods such as off-line policy distillation.

All the models on all the instances ,except the twenty-one task instance were trained with 16 parallel threads.

The models on the twenty-one task instance were trained with 20 parallel threads.

Training and evaluation were interleaved.

It is to be noted that while during the training period active sampling principles were used in this work to improve multi-tasking performance, during the testing/evaluation period, the multi-tasking network executed on each task for the same duration of time (5 episodes, each episode capped at length 30000).

For the smaller multi-tasking instances (MT1, MT2, MT3 and MT4), after every 3 million training steps, the multi-tasking network was made to execute on each of the constituent tasks of the multi-tasking instance it is solving for 5 episodes each.

Each such episode was capped at a length of 30000 to ensure that the overall evaluation time was bounded above.

For the larger multi-tasking instances (MT5, MT6 and MT7) the exact same procedure was carried out for evaluation, except that evaluation was done after every 5 million training steps.

The lower level details of the evaluation scheme used are the same as those described in BID26 .

The evolution of this average game-play performance with training progress has been demonstrated for MT1 in FIG3 .

Training curves for other multi-tasking instances are presented in Appendix D.

We used a low level architecture similar to BID26 which in turn uses the same low level architecture as .

The first three layers of are convolutional layers with same filter sizes, strides, padding as BID26 .

The convolutional layers each have 64 filters.

These convolutional layers are followed by two fully connected (FC) layers and an LSTM layer.

A policy and a value function are derived from the LSTM outputs using two different output heads.

The number of neurons in each of the FC layers and the LSTM layers is 256.Similar to ) the Actor and Critic share all but the final layer.

Each of the two functions: policy and value function are realized with a different final output layer, with the value function outputs having no non-linearity and with the policy having a softmax-non linearity as output non-linearity, to model the multinomial distribution.

We will now describe the hyper-parameters of the meta-task-decider used in each of the methods proposed in the paper:

The algorithm for A5C has been specified in Algorithm 3.

The temperature parameter ?? in the softmax function used for the task selection was tuned over the set {0.025, 0.033, 0.05, 0.1}. The best value was found to be 0.05.

Hyper-parameter n was set to be 10.

The hyper-parameter l was set to be 4 million.

The discounted UCB1-tuned + algorithm from BID8 was used to implement the meta-task-decider.

The algorithm for training UA4C agents has been demonstrated in Algorithm 4.

We hyper-parameter tuned for the discount factor ?? used for the meta-decider (tuned over the set {0.8, 0.9, 0.99}) and the scaling factor for the bonus ?? (tuned over the set {0.125, 0.25, 0.5, 1}).

The best hyper-parameters were found to be ?? = 0.99 and ?? = 0.25.

The meta-learner network was also a type of A3C network, with one meta-learner thread being associated with one multi-task learner thread.

The task that the MTN on thread i trained on was sampled according to the policy of the meta-learner M i where M i denotes the meta-learner which is executing on thread i.

The meta-learner was also trained using the A3C algorithm with asyncrms-prop.

The meta-learner used 1-step returns instead of the 20-step returns that the usual A3C algorithm uses.

The algorithm for training EA4C agents has been demonstrated in Algorithm 5.

We tuned the ?? meta for entropy regularization for encouraging exploration in the meta-learner's policy over the set {0, 0.003, 0.01} and found the best value to be ?? meta = 0.

We also experimented with the ?? meta , the discounting factor for the RL problem that the meta-learner is solving.

We tuned it over the set {0.5, 0.8, 0.9} and found the best value to be ?? meta = 0.8.

The initial learning rate for the meta learner was tuned over the set {5 ?? 10 ???4 , 10 ???3 , 3 ?? 10 ???3 } and 10 ???3 was found to be the optimal initial learning rate.

Similar to the multi-tasking network, the learning rate was linearly annealed to 10 ???4 over the number of training steps for which the multi-tasking network was trained.

We extensively experimented with the architecture of the meta-learner.

We experimented with feedforward and LSTM versions of EA4C and found that the LSTM versions comprehensively outperform the feed-forward versions in terms of the multi-tasking performance (q am ).

We also comprehensively experimented with wide, narrow, deep and shallow networks.

We found that increasing depth beyond a point (??? 3 fully connected layers) hurt the multi-tasking performance.

Wide neural networks (both shallow and deep ones) were unable to perform as well as their narrower counter-parts.

The number of neurons in a layer was tuned over the set {50, 100, 200, 300, 500} and 100 was found to be the optimal number of neurons in a layer.

The number of fully-connected layers in the meta-learner was tuned over the set {1, 2, 3} was 2 was found to be the optimal depth of the meta-controller.

The best-performing architecture of the meta-learner network consists of: two fully-connected layers with 100 neurons each, followed by an LSTM layer with 100 LSTM cells, followed by one linear layer each modeling the meta-learner's policy and its value function.

We experimented with dropout layers in meta-learner architecture but found no improvement and hence did not include it in the final architecture using which all experiments were performed.

All the hyper-parameters for FA4C were tuned in exactly the same way that they were tuned for EA4C.

The task decision steps for FA4C were time-driven (taken at regular intervals of training the multi-tasking network) rather than being event-driven (happening at the end of an episode, like in the EA4C case).

While the interval corresponding to the task decision steps can in general be different from the n to be used for n-step returns, we chose both of them to be the same with n = 20.

This was done to allow for an easier implementation of the FA4C method.

Also, 20 was large enough so that one could find meaningful estimates of 20-step cumulative returns without the estimate having a high variance and also small enough so that the FA4C meta-learner was allowed to make a large number of updates (when the multi-tasking networks were trained for 300 million steps (like in MT1, MT2 and MT3) The FA4C meta-learner was trained for roughly 15 million steps.)

Algorithm 1 contains a pseudo-code for training a generic active sampling method proposed in this work.

This appendix contains specific instantiations of that algorithm for the all the methods proposed in this work.

It also contains an algorithm for training the baseline MTA proposed in this work.

Algorithm FORMULA7 for i in {1, ?? ?? ?? , k} do 6: DISPLAYFORM0 for train_steps:0 to t do score j ??? bsmta.train_for_one_episode(T j )

Algorithm 3 A5C 1: function MULTITASKING ( SetOfTasks T ) 2: DISPLAYFORM0 h i ??? Target score in task T i .

This could be based on expert human performance or even published scores from other technical works 4:n ??? Number of episodes which are used for estimating current average performance in any task T i

l ??? Number of training steps for which a uniformly random policy is executed for task selection.

At the end of l training steps, the agent must have learned on ??? n episodes ??? tasks T i ??? T

t ??? Total number of training steps for the algorithm 7:s i ??? List of last n scores that the multi-tasking agent scored during training on task T i .

p i ??? Probability of training on an episode of task T i next.

?? ??? Temperature hyper-parameter of the softmax task-selection non-parametric policy 10:amta ???

The Active Sampling multi-tasking agent 11:for i in {1, ?? ?? ?? , k} do 12: DISPLAYFORM0 for train_steps:0 to t do

if train_steps ??? l then X i ??? Discounted sum of rewards for task i 8:X i ??? Mean of discounted sum of rewards for task i score ??? amta.train_for_one_episode(T j )19: DISPLAYFORM0 hj ???score hj , 0 21:

X i ??? Discounted sum of rewards for task i 8:X i ??? Mean of discounted sum of rewards for task i X i ??? 0 ???i 13: DISPLAYFORM1 DISPLAYFORM2 for train_steps:0 to t do 17: DISPLAYFORM3 DISPLAYFORM4 X i ??? ??X i ???i 22: DISPLAYFORM5 hj ???score hj , 0 23: DISPLAYFORM6 n j ??? n j + 1 25:X i ??? X i /n i ???i 26: DISPLAYFORM7 Comparison of performance of BA3C, A5C, UA4C, EA4C and FA4C agents along with task-specific A3C agents for MT2 (6 tasks).

Agents in these experiments were trained for 300 million time steps and required half the data and computation that would be required to train the task-specific agents (STA3C) for all the tasks.

Figure 9 : Comparison of performance of BA3C, A5C, UA4C, EA4C and FA4C agents along with task-specific A3C agents for MT3 (6 tasks).

Agents in these experiments were trained for 300 million time steps and required half the data and computation that would be required to train the task-specific agents (STA3C) for all the tasks.

This multi-tasking instance has 12 tasks.

Although this set of tasks is medium-sized, the multi-tasking network has the same size as those used for MT1, MT2 and MT3 as well as a single-task network.

Figure 10: Comparison of performance of BA3C, A5C, UA4C, EA4C and FA4C agents along with task-specific A3C agents for MT4 (8 tasks).

Agents in these experiments were trained for 400 million time steps and required half the data and computation that would be required to train the task-specific agents (STA3C) for all the tasks.

FIG0 : Comparison of performance of BA3C, A5C, UA4C, EA4C and FA4C agents along with task-specific A3C agents for MT5 (12 tasks).

Agents in these experiments were trained for 600 million time steps and required half the data and computation that would be required to train the task-specific agents (STA3C) for all the tasks.

This multi-tasking instance has 12 tasks as well.

FIG0 : Comparison of performance of BA3C, A5C, UA4C, EA4C and FA4C agents along with task-specific A3C agents for MT6 (12 tasks).

Agents in these experiments were trained for 600 million time steps and required half the data and computation that would be required to train the task-specific agents (STA3C) for all the tasks.

This multi-tasking instance has 21 tasks.

This is a large-sized set of tasks.

Since a single network now needs to learn the prowess of 21 visually different Atari tasks, we roughly doubled the number of parameters in the network, compared to the networks used for MT1, MT2 and MT3 as well as a single-task network.

We believe that this is a fairer large-scale experiment than those done in BID20 wherein for a multi-tasking instance with 10 tasks, a network which has four times as many parameters as a single-task network is used.

Published as a conference paper at ICLR 2018 FIG0 : Comparison of performance of BA3C, A5C, UA4C, EA4C and FA4C agents along with task-specific A3C agents for MT7 (21 tasks).

Agents in these experiments were trained for 1.05 billion time steps and required half the data and computation that would be required to train the task-specific agents (STA3C) for all the tasks.

In this appendix, we document the performance of our methods on the all the four performance metrics (p am , q am , q gm , q hm ) that have been proposed in Section 4.1.q am is a robust evaluation metric because the agent needs to be good in all the tasks in order to get a high score on this metric.

In TAB5 we can observe a few important trends:1.

The adaptive method is a hard baseline to beat.

The very fact that tasks are being sampled in accordance with the lack of performance of the multi-tasking on them, means that the MTA benefits directly from such a strategy.2.

The UCB-based meta-learner generalizes fairly well to medium-sized instances but fails to generalize to the largest of our multi-tasking instances: MT7.

3.

It is our meta-learning method EA4C which generalizes the best to the largest multi-tasking instance MT7.

This could be because the UCB and adaptive controllers are more rigid compared to the learning method.

TAB6 demonstrates the need for the evaluation metrics that we have proposed.

specifically, it can be seen that in case of MT4, the non-clipped average performance is best for BA3C.

However, this method is certainly not a good MTL algorithm.

This happens because the uniform sampling ensures that the agent trains on the task of Enduro a lot (can be seen in the corresponding training curves).

Owing to high performance on a single task, p am ends up concluding that BA3C is the best multi-tasking network.

We defined the q gm and q hm metrics because in some sense, the q am metric can still get away with being good on only a few tasks and not performing well on all the tasks.

In this limited sense, q hm is probably the best choice of metric to understand the multi-tasking performance of an agent.

We can observe that while A5C performance was slightly better than EA4C performance for MT4 according to the q am metric, the agents are much more head to head as evaluated by the q hm metric.

To demonstrate that our framework is robust to the use of different target scores, we performed two targeted experiments.

In this first experiment, we swapped out the use of single-task scores as target scores with the use of scores obtained by Human testers.

These human scores were taken from .

We experimented with UA4C on MT1 in this subsection.

Consequently we refer to the use of human scores in UA4C as HUA4C.

FIG0 depicts the evolution of the raw performance (game-score) FIG0 : Training curve for HUA4C: when human scores are used as target for calculating the rewards.of HUA4C agent trained with human scores as targets instead of single-task network's scores.

The performance of HUA4C on all the metrics proposed in this paper is contained in TAB9 .

All the hyper-parameters found by tuning UA4C on MT1 were retained.

None of the hyper-parameters were re-tuned.

This represents a setup which isn't very favorable for HUA4C.

We observe that even in this setup, the performance of UA4C is impressive.

However, it is unable to learn at all for two of the tasks and has at best mediocre performance in three others.

We believe that the performance could possibly improve if hyper-parameters were tuned for this specific paradigm/framework.

To demonstrate that the impressive performance of our methods is not conditioned on the use of singletask performance as target scores, we decided to experiment with twice the single-task performance as the target scores.

In some sense, this twice the single-task performance score represents a very optimistic estimate of how well an MTA can perform on a given task.

All experiments in this sub-section are performed with A5C.

Since the hyper-parameters for all the methods were tuned on M T 1 , understandably, the performance of our agents is better on M T 1 than M T 2 or M T 3 .

Hence we picked the multi-tasking instances M T 2 and M T 3 to demonstrate the effect of using twice the target scores which were used by A5C.

We chose the twice-single-task-performance regime arbitrarily and merely wanted to demonstrate that a change in the target scores does not adversely affect our methods' performance.

Note that we did not tune the hyper-parameters for experiments in this sub-section.

Such a tuning could potentially improve the performance further.

It can be seen that in every case, the use of twice-the-single-task-performance as target scores improves the performance of our agents.

In some cases such as M T 3 there was a large improvement.

In this section, we present the results from Firing Analyses done for all the MTIs in this work.

The method used to generate the following graphs has been described in section 7.2.

It can be seen from the following graphs that the active sampling methods(A5C,UA4C and EA4C) have a large fraction of neurons that fire for a large fraction of time in atleast half the number of tasks in the MTI, whereas BA3C has a relatively higher fraction of task-specific neurons.

This alludes to the fact that the active sampling methods have been successful in learning useful features that generalize across different tasks, hence leading to better performance.

Neuron-Firing Analysis on MT1: FIG1 : Training Curves for the DUA4C agent on MT1 (6 tasks).

The horizontal line represents the Single Task Agent's score.

Agents in these experiments were trained for 300 million time steps and required half the data and computation that would be required to train the task-specific agents (STA3C) for all the tasks.

Figure 27: Training Curves for the DUA4C agent on MT2 (6 tasks).

The horizontal line represents the Single Task Agent's score.

Agents in these experiments were trained for 300 million time steps and required half the data and computation that would be required to train the task-specific agents (STA3C) for all the tasks. .

The horizontal line represents the Single Task Agent's score.

Agents in these experiments were trained for 400 million time steps and required half the data and computation that would be required to train the task-specific agents (STA3C) for all the tasks.

Figure 29: Training Curves for the DUA4C agent on MT5 (12 tasks).

The horizontal line represents the Single Task Agent's score.

Agents in these experiments were trained for 600 million time steps and required half the data and computation that would be required to train the task-specific agents (STA3C) for all the tasks.

<|TLDR|>

@highlight

Letting a meta-learner decide the task to train on for an agent in a multi-task setting improves multi-tasking ability substantially
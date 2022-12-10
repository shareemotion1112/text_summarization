We present an adversarial exploration strategy, a simple yet effective imitation learning scheme that incentivizes exploration of an environment without any extrinsic reward or human demonstration.

Our framework consists of a deep reinforcement learning (DRL) agent and an inverse dynamics model contesting with each other.

The former collects training samples for the latter, and its objective is to maximize the error of the latter.

The latter is trained with samples collected by the former, and generates rewards for the former when it fails to predict the actual action taken by the former.

In such a competitive setting, the DRL agent learns to generate samples that the inverse dynamics model fails to predict correctly, and the inverse dynamics model learns to adapt to the challenging samples.

We further propose a reward structure that ensures the DRL agent collects only moderately hard samples and not overly hard ones that prevent the inverse model from imitating effectively.

We evaluate the effectiveness of our method on several OpenAI gym robotic arm and hand manipulation tasks against a number of baseline models.

Experimental results show that our method is comparable to that directly trained with expert demonstrations, and superior to the other baselines even without any human priors.

Over the past decade, imitation learning (IL) has been successfully applied to a wide range of domains, including robot learning BID7 BID20 , autonomous navigation BID4 BID19 , manipulation tasks BID11 BID17 , and self-driving cars BID5 .

Traditionally, IL aims to train an imitator to learn a control policy π only from expert demonstrations.

The imitator is typically presented with multiple demonstrations during the training phase, with an aim to distill them into π.

To learn π effectively and efficiently, a large set of high-quality demonstrations are necessary.

This is especially prevalent in current state-of-the-art IL algorithms, such as dataset aggregation (DAgger) BID18 and generative adversarial imitation learning (GAIL) BID9 .

Although these approaches have been the dominant algorithms in IL, a major bottleneck for them is their reliance on high-quality demonstrations, which often require extensive supervision from human experts.

In addition, a serious flaw in the learned policy π is its tendency to overfit to demonstration data, preventing it from generalizing to new ones.

To overcome the aforementioned challenges in IL, a number of methods have been investigated to enhance the generalizability and data efficiency, or reduce the degree of human supervision.

Initial efforts in this direction were based on the idea of meta learning BID6 BID8 Yu et al., 2018) , in which the imitator is trained from a meta learner that is able to quickly learn a new task with only a few set of demonstrations.

However, such schemes still require training the meta-learner with tremendous amount of time and demonstration data, leaving much room for improvement.

Thus, a rapidly-growing body of literature based on the concept of using forward/inverse dynamics models to learn π within an environment in a self-supervised fashion BID0 BID11 BID13 has emerged in the past few years.

One key advantage of the concept is that it provides an autonomous way for preparing training data, removing the need of human intervention.

In this paper, we call it self-supervised IL.Self-supervised IL allows an imitator to collect training data by itself instead of using predefined extrinsic reward functions or expert supervision during training.

It only needs demonstration during inference, drastically decreasing the time and effort required from human experts.

Although the core principles of self-supervised IL are straightforward and have been exploited in many fields BID0 BID11 BID12 , recent research efforts have been dedicated to addressing the challenges of multi-modality and multi-step planning.

For example, the use of forward consistency loss and forward regularizer have been extensively investigated to enhance the task performance of the imitator BID0 BID13 .

This becomes especially essential when the lengths of trajectories grow and demonstration samples are sparse, as multiple paths may co-exist to lead the imitator from its initial observation to the goal observation.

The issue of multi-step planning has also drawn a lot of attention from researchers, and is usually tackled by recurrent neural networks (RNNs) and step-by-step demonstrations BID11 BID13 .

The above self-supervised IL approaches report promising results, however, most of them are limited in applicability due to several drawbacks.

First, traditional methods of data collection are usually inefficient and time-consuming.

Inefficient data collection results in poor exploration, giving rise to a degradation in robustness to varying environmental conditions (e.g., noise in motor control) and generalizability to difficult tasks.

Second, human bias in data sampling range tailored to specific interesting configurations is often employed BID0 BID11 .

Although a more general exploration strategy called curiosity-driven exploration was later proposed in BID12 , it focuses only on exploration in states novel to the forward dynamics model, rather than those directly influential to the inverse dynamics model.

Furthermore, it does not discuss the applicability to continuous control domains, and fails in high dimensional action spaces according to our experiments in Section 4.

Unlike the approaches discussed above, we do not propose to deal with multi-modality or multi-step planning.

Instead, we focus our attention on improving the overall quality of the collected samples in the context of self-supervised IL.

This motivates us to equip the model with the necessary knowledge to explore the environment in an efficient and effective fashion.

In this paper, we propose a straightforward and efficient self-supervised IL scheme, called adversarial exploration strategy, which motivates exploration of an environment in a self-supervised manner (i.e., without any extrinsic reward or human demonstration).

Inspired by Pinto et al. (2017) ; BID23 ; BID24 , we implement the proposed strategy by jointly training a deep reinforcement learning (DRL) agent and an inverse dynamics model competing with each other.

The former explores the environment to collect training data for the latter, and receives rewards from the latter if the data samples are considered difficult.

The latter is trained with the training data collected by the former, and only generates rewards when it fails to predict the true actions performed by the former.

In such an adversarial setting, the DRL agent is rewarded only for the failure of the inverse dynamics model.

Therefore, the DRL agent learns to sample hard examples to maximize the chances to fail the inverse dynamics model.

On the other hand, the inverse dynamics model learns to be robust to the hard examples collected by the DRL agent by minimizing the probability of failures.

As a result, as the inverse dynamics model becomes stronger, the DRL agent is also incentivized to search for harder examples to obtain rewards.

Overly hard examples, however, may lead to biased exploration and cause instability of the learning process.

In order to stabilize the learning curve of the inverse dynamics model, we further propose a reward structure such that the DRL agent is encouraged to explore moderately hard examples for the inverse dynamics model, but refraining from too difficult ones for the latter to learn.

The self-regulating feedback structure between the DRL agent and the inverse dynamics model enables them to automatically construct a curriculum for exploration.

We perform extensive experiments to validate adversarial exploration strategy on multiple OpenAI gym BID3 robotic arm and hand manipulation task environments simulated by the MuJoCo physics engine (Todorov et al., 2012) , including FetchReach, FetchPush, FetchPickAndPlace, FetchSlide, and HandReach.

These environments are intentionally selected by us for evaluating the performance of inverse dynamics model, as each of them allows only a very limited set of chained actions to transition the robotic arms and hands to target observations.

We examine the effectiveness of our method by comparing it against a number of self-supervised IL schemes.

The experimental results show that our method is more effective and data-efficient than the other self-supervised IL schemes for both low-and high-dimensional observation spaces, as well as in environments with high-dimensional action spaces.

We also demonstrate that in most of the cases the performance of the inverse dynamics model trained by our method is comparable to that directly trained with expert demonstrations.

The above observations suggest that our method is superior to the other self-supervised IL schemes even in the absence of human priors.

We further evaluate our method on environments with action space perturbations, and show that our method is able to achieve satisfactory success rates.

To justify each of our design decisions, we provide a comprehensive set of ablative analysis and discuss their implications.

The contributions of this work are summarized as follows:• We introduce an adversarial exploration strategy for self-supervised IL.

It consists of a DRL agent and an inverse dynamics model developed for efficient exploration and data collection.• We employ a competitive scheme for the DRL agent and the inverse dynamics model, enabling them to automatically construct a curriculum for exploration of observation space.• We introduce a reward structure for the proposed scheme to stabilize the training process.•

We demonstrate the proposed method and compare it with a number of baselines for multiple robotic arm and hand manipulation tasks in both low-and high-dimensional state spaces.•

We validate that our method is generalizable to tasks with high-dimensional action spaces.

The remainder of this paper is organized as follows.

Section 2 introduces background material.

Section 3 describes the proposed adversarial exploration strategy in detail.

Section 4 reports the experimental results, and provides an in-depth ablative analysis of our method.

Section 5 concludes.

In this section, we briefly review DRL, policy gradient methods, as well as inverse dynamics model.

DRL trains an agent to interact with an environment E. At each timestep t, the agent receives an observation x t ∈ X , where X is the observation space of E. It then takes an action a t from the action space A based on its current policy π, receives a reward r, and transitions to the next observation x .

The policy π is represented by a deep neural network with parameters θ, and is expressed as π(a|x, θ).

The goal of the agent is to learn a policy to maximize the discounted sum of rewards G t : DISPLAYFORM0 where t is the current timestep, γ ∈ (0, 1] the discount factor, and T the horizon.

Policy gradient methods BID10 BID25 Williams, 1992) are a class of RL techniques that directly optimize the parameters of a stochastic policy approximator using policy gradients.

Although these methods have achieved remarkable success in a variety of domains, the high variance of gradient estimates has been a major challenge.

Trust region policy optimization (TRPO) BID21 circumvented this problem by applying a trust-region constraint to the scale of policy updates.

However, TRPO is a second-order algorithm, which is relatively complicated and not compatible with architectures that embrace noise or parameter sharing BID22 .

In this paper, we employ a more recent family of policy gradient methods, called proximal policy optimization (PPO) BID22 .

PPO is an approximation to TRPO, which similarly prevents large changes to the policy between updates, but requires only first-order optimization.

PPO is superior in its generalizability and sample complexity while retaining the stability and reliability of TRPO 1 .

An inverse dynamics model I takes as input a pair of observations (x, x ), and predicts the actionâ required to reach the next observation x from the current observation x.

It is formally expressed as: DISPLAYFORM0 where (x, x ) are sampled from the collected data, and θ I represents the trainable parameters of I. During the training phase, θ I is iteratively updated to minimize the loss function L I , expressed as: DISPLAYFORM1 where d is a distance metric, and a the ground truth action.

During the testing phase, a sequence of observations {x 0 ,x 1 , · · · ,x T } is first captured from an expert demonstration.

A pair of observations (x t ,x t+1 ) is then fed into I at each timestep t. Starting fromx 0 , the objective of I is to predict a sequence of actions {â 0 ,â 1 , · · · ,â T −1 } and transition the final observationx T as close as possible.

In this section, we first describe the proposed adversarial exploration strategy.

We then explain the training methodology in detail.

Finally, we discuss a technique for stabilizing the training process.

FIG0 shows a framework that illustrates the proposed adversarial exploration strategy, which includes a DRL agent P and an inverse dynamics model I. Assume that sequence of observations and actions generated by P as it explores E using a policy π.

At each timestep t, P collects a 3-tuple training sample (x t , a t , x t+1 ) for I, while I predicts an actionâ t and generates a reward r t for P .

In this work, I is modified from Eq. FORMULA1 to include an additional hidden vector h t , which recurrently encodes the information of the past observations.

I is thus expressed as:

where f (·) denotes the recurrent function.

θ I is iteratively updated to minimize L I , formulated as: DISPLAYFORM0 where β is a scaling constant.

We employ mean squared error β||a t −â t || 2 as the distance metric d(a t ,â t ), since we only consider continuous control domains in this paper.

It can be replaced with a cross-entropy loss for discrete control tasks.

We directly use L I as the reward r t for P , expressed as: DISPLAYFORM1 Our method targets at improving both the quality and efficiency of the data collection process performed by P , as well as the performance of I. Therefore, the goal of the proposed framework is twofold.

First, P has to learn an adversarial policy π adv (a t |x t ) such that its cumulated discounted rewards DISPLAYFORM2 ) is maximized.

Second, I requires to learn an optimal θ I such that Eq. (6) is minimized.

Minimizing L I (i.e., r t ) leads to decreased G t|π adv , forcing P to enhance π adv to explore more difficult samples to increase G t|π adv .

This implies that P is motivated to focus on I's weak points, instead of randomly collecting ineffective training samples.

Training I with hard samples not only accelerates its learning progress, but also helps to boost its performance.

We describe the training methodology of our adversarial exploration strategy by a pseudocode presented in Algorithm 1.

Assume that P 's policy π adv is parameterized by a set of trainable parameters θ P , and is represented as π adv (a t |x t , θ P ).

We create two buffers Z P and Z I for storing the training samples of P and I, respectively.

In the beginning, Z P , Z I , E, θ P , θ I , π adv , as well as a timestep cumulative counter c are initialized.

A number of hyperparameters are set to appropriate values, including the number of iterations N iter , the number of episodes N episode , the horizon T , as well as the update period T P of θ P .

At each timestep t, P perceives the current observation x t from E, takes an action a t according to π adv (a t |x t , θ P ), and receives the next observation x t+1 and a termination indicator ξ (lines 9-11).

ξ is set to 1 only when t equals T , otherwise it is set to 0.

We then store (x t , a t , x t+1 , ξ) and (x t , a t , x t+1 ) in Z P and Z I , respectively.

We update θ P every T P timesteps using the samples stored in Z P , as shown in (lines 13-21).

At the end of each episode, we update θ I with samples drawn from Z I according to the loss function L I defined in Eq. (5) (line 23).

Although adversarial exploration strategy is effective in collecting hard samples, it requires additional adjustments if P becomes too strong such that the collected samples are too difficult for I to learn.

Overly difficult samples lead to a large variance in gradients derived from L I , which in turn cause a performance drop in I and instability in its learning process.

We analyze this phenomenon in greater detail in Section 4.5.

To tackle the issue, we propose a training technique that reshapes r t as follows: DISPLAYFORM0 Algorithm 1 Adversarial exploration strategy 1: Initialize Z P , Z I , E, and model parameters θ P & θ I 2: Initialize π adv (at|xt, θ P ) 3: Initialize the timestep cumulative counter c = 0 4: Set Niter, N episode , T , and T P 5: for iteration i = 1 to Niter do 6:for episode e = 1 to N episode do

for timestep t = 0 to T do 8:P perceives xt from E, and predicts an action at according to π adv (at|xt, θ P )

xt+1 = E(xt, at)10: DISPLAYFORM0 Store (xt, at, xt+1, ξ) in Z P

Store (xt, at, xt+1) in Z I

if (c % T P ) == 0 then

Initialize an empty batch B

Initialize a recurrent state ht

for (xt, at, xt+1, ξ) in Z P do 17: Evaluateât = I(xt, xt+1|ht, θ I ) (calculated from Eq. FORMULA4 18:Evaluate rt(xt, at, xt+1) = L I (at,ât|θ I ) (calculated from Eq. FORMULA6 19: DISPLAYFORM0 Update θ P with the gradient calculated from the samples of B

Reset Z P 22: DISPLAYFORM0

Update θ I with the gradient calculated from the samples of Z I (according to Eq. FORMULA5 24: end where δ is a pre-defined threshold value.

This technique poses a restriction on the range of r t , driving P to gather moderate samples instead of overly hard ones.

Note that the value of δ affects the learning speed and the final performance.

We plot the impact of δ on the learning curve of I in Section 4.5.

We further provide an example in our supplementary material to visualize the effect of this technique.

In this section, we present experimental results for a series of robotic tasks, and validate that (i) our method is effective in both low-and high-dimensional observation spaces; (ii) our method is effective in environments with high-dimensional action spaces; (iii) our method is more data efficient than the baseline methods; and (iv) our method is robust against action space perturbations.

We first introduce our experimental setup.

Then, we report experimental results of robotic arm and hand manipulation tasks.

Finally, we present a comprehensive set of ablative analysis to validate our design decisions.

We first describe the environments and tasks.

Next, we explain the evaluation procedure and the method for collecting expert demonstrations.

We then walk through the baselines used for comparison.

We evaluate our method on a number of robotic arm and hand manipulation tasks via OpenAI gym BID3 environments simulated by the MuJoCo (Todorov et al., 2012) physics engine.

We use the Fetch and Shadow Dexterous Hand BID16 for the arm and hand manipulation tasks, respectively.

For the arm manipulation tasks, which include FetchReach, FetchPush, FetchPickAndPlace, and FetchSlide, the imitator (i.e., the inverse dynamic model I) takes as inputs the positions and velocities of a gripper and a target object.

It then infers the gripper's action in 3-dimensional space to manipulate it.

For the hand manipulation task HandReach, the imitator takes as inputs the positions and velocities of the fingers of a robotic hand, and determines the velocities of the joints to achieve the goal.

In addition to low-dimensional observations (i.e., position, velocity, and gripper state), we further perform experiments for the above tasks using visual observations (i.e., high-dimensional observations) in the form of camera images taken from a third-person perspective.

The detailed description of the above tasks is specified in BID16 .

For the detailed configurations of these tasks, please refer to our supplementary material.

The primary objective of our experiments is to demonstrate the efficiency of the proposed adversarial exploration strategy in collecting training data (in a self-supervised manner) for the imitator.

We compare our strategy against a number of self-supervised data collection methods (referred to as "baselines" or "baseline methods") described in Section 4.1.4.

As different baseline methods employ different data collection strategies, the learning curve of the imitator also varies for different cases.

For a fair comparison, the model architecture of the imitator and the amount of training data are fixed for all cases.

All of the experimental results are evaluated and averaged over 20 trials, corresponding to 20 different random initial seeds.

In each trial, we train an imitator by the training data collected by a single self-supervised data collection method.

At the beginning of each episode, the imitator receives a sequence of observations {x 0 ,x 1 , · · · ,x T } from a successful expert demonstration.

At each timestep t, the imitator infers an actionâ t from an expert observationx t+1 and its current observation x t by Eq. (4).

We periodically evaluate the imitator every 10K timesteps.

The evaluation is performed by averaging the success rates of reachingx T over 500 episodes.

The configuration of the imitator and the hyperparameters of the baselines are summarized in the supplementary material.

For each task mentioned in Section 4.1.1, we first randomly configure task-relevant settings (e.g., goal position, initial state, etc.).

We then collect demonstrations from non-trivial and successful episodes performed by a pre-trained expert agent .

Please note that the collected demonstrations only contain sequences of observations.

The implementation details of the expert agent and the method for filtering out trivial episodes are presented in our supplementary material.

We compare our proposed methodology with the following four baseline methods in our experiments.• Random: This method collects training samples by random exploration.

We consider it to be an important baseline because of its simplicity and prevalence in a number of research works on self-supervised IL BID0 BID11 BID13 ).•

Demo: This method trains the imitator directly with expert demonstrations.

It serves as the performance upper bound, as the training data is the same as the testing data for this method.• Curiosity: This method trains a DRL agent via curiosity BID12 to collect training samples.

Unlike the original implementation, we replace its DRL algorithm with PPO, as training should be done on a single thread for a fair comparison with the other baselines.

This is alo an important baseline due to its effectiveness in BID13 .•

Noise BID15 : In this method, noise is injected to the parameter space of a DRL agent to encourage exploration BID15 .

Please note that its exploratory behavior relies entirely on parameter space noise, instead of using any extrinsic reward.

We include this method due to its superior performance and data efficiency in many DRL tasks.

We compare the performance of the proposed method and the baselines on the robotic arm manipulation tasks described in Section 4.1.1.

As opposed to discrete control domains, these tasks are especially challenging, as the sample complexity grows in continuous control domains.

Furthermore, the imitator may not have the complete picture of the environment dynamics, increasing its difficulty to learn an inverse dynamics model.

In FetchSlide, for instance, the movement of the object on the slippery surface is affected by both friction and the force exerted by the gripper.

It thus motivates us to investigate whether the proposed method can help overcome the challenge.

In the subsequent paragraphs, we discuss the experimental results in both low-and high-dimensional observation spaces, and plot them in Figs. 2 and 3, respectively.

All of the results are obtained by following the procedure described in Section 4.1.2.

The shaded regions in Figs. 2 and 3 represent the confidence intervals.

Low-dimensional observation spaces.

FIG1 plots the learning curves for all of the methods in low-dimensional observation spaces.

In all of the tasks, our method yields superior or comparable performance to the baselines except for Demo, which is trained directly with expert demonstrations.

In FetchReach, it can be seen that every method achieves a success rate of 1.0.

This implies that it does not require a sophisticated exploration strategy to learn an inverse dynamics model in an environment where the dynamics is relatively simple.

It should be noted that although all methods reach the same final success rate, ours learns significantly faster than Demo.

In contrast, in FetchPush, our method is comparable to Demo, and demonstrates superior performance to the other baselines.

Our method also learns drastically faster than all the other baselines, which confirms that the proposed strategy does improve the performance and efficiency of self-supervised IL.

Our method is particularly effective in tasks that require an accurate inverse dynamics model.

In FetchPickAndPlace, for example, our method surpasses all the other baselines.

However, all methods including Demo fail to learn a successful inverse dynamics model in FetchSlide, which suggests that it is difficult to train an imitator when the outcome of an action is not completely dependent on the action itself.

It is worth noting that Curiosity loses to Random in FetchPush and FetchSlide, and Noise performs even worse than these two methods in all of the tasks.

We therefore conclude that Curiosity is not suitable for continuous control tasks, and the parameter space noise strategy cannot be directly applied to self-supervised IL.

In addition to the quantitative results presented above, we further discuss the empirical results qualitatively.

Please refer our supplementary material for a description of the qualitative results.

High-dimensional observation spaces.

FIG2 plots the learning curves of all methods in highdimensional observation spaces.

It can be seen that our method performs significantly better than the other baseline methods in most of the tasks, and is comparable to Demo.

In FetchPickAndPlace, our method is the only one that learns a successful inverse dynamics model.

Similar to the results in FIG1 , Curiosity is no better than Random in high-dimensional observation spaces.

Please note that we do not include Noise in FIG2 as it performs worse enough already in low-dimensional settings.

FIG1 plots the learning curves for each of the methods considered.

Please note that Curiosity, Noise and our method are pre-trained with 30K samples collected by random exploration, as we observe that these methods on their own suffer from large errors in an early stage during training, which prevents them from learning at all.

After the first 30K samples, they are trained with data collected by their exploration strategy instead.

From the results in FIG1 , it can be seen that Demo easily stands out from the other methods as the best-performing model, surpassing them all by a considerable extent.

Although our method is not as impressive as Demo, it significantly outperforms all of the other baseline methods, achieving a success rate of 0.4 while the others are still stuck at around 0.2.

The reason that the inverse dynamics models trained by the self-supervised data-collection strategies discussed in this paper (including ours and the other baselines) are not comparable to the Demo baseline in the HandReach task is primarily due to the high-dimensional action space.

It is observed that the data collected by the self-supervised data-collection strategies only cover a very limited range of the state space in the HandReach environment.

Therefore, the inverse dynamics models trained with these data only learn to imitate trivial poses, leading to the poor success rates presented in FIG1 .

We evaluate the performance of the imitator trained in an environment with action space perturbations to validate the robustness of our adversarial exploration strategy.

In such an environment, every action taken by the DRL agent is perturbed by a Gaussian random noise, such that the training samples collected by the DRL agent are not inline with its actual intentions.

Please note that we only inject noise during the training phase, as we aim to validate the robustness of the proposed data collection strategy.

The scale of the injected noise is specified in the supplementary material.

We report the performance change rates of various methods for different tasks in Table.

1.

The performance change rate is defined as: DISPLAYFORM0 , where P r perturb and P r orig represent the highest success rates with and without action space perturbations, respectively.

From Table.

1, it can be seen that our method retains the performance for most of the tasks, indicating that our method is robust to action space perturbations during the training phase.

Please note that although Curiosity and Noise also achieve a change rate of 0% in HandReach and FetchSlide, they are not considered robust due to their poor performance in the original environment FIG1 .

Another interesting observation is that our method even gains some performance from action space perturbations in FetchPush and HandReach, which we leave as one of our future directions.

We thus conclude that our method is robust to action space perturbations during the training phase, making it a practical option in real-world settings.

In this section, we provide a set of ablative analysis.

We examine the effectiveness of our method by an investigation of the training loss distribution, the stabilization technique, and the influence of δ.

Please note that the value of δ is set to 1.5 by default, as described in our supplementary material.

Training loss distribution.

FIG3 plots the probability density function (PDF) of L I (derived from Eq. FORMULA5 ) by kernel density estimation (KDE) for the first 2K training batches during the training phase.

The vertical axis corresponds to the probability density, while the horizontal axis represents the scale of L I .

The curves Ours (w stab) and Ours (w/o stab) represent the cases where the stabilization technique described in Section 3.3 is employed or not, respectively.

We additionally plot the curve Random in FIG3 to highlight the effectiveness of our method.

It can be observed that both Ours (w stab) and Ours (w/o stab) concentrate on notably higher loss values than Random.

This observation implies that adversarial exploration strategy does explore hard samples for inverse dynamics model.

We validate the proposed stabilization technique in terms of the PDF of L I and the learning curve of the imitator, and plot the results in FIG3 and 5, respectively.

FIG3 , it can be observed that the modes of Ours (w stab) are lower than those of Ours (w/o stab) in most cases, implying that the stabilization technique indeed motivates the DRL agents to favor those moderately hard samples.

We also observe that for each of the five cases, the mode of Ours (w stab) is close to the value of δ (plotted in a dotted line), indicating that our reward structure presented in Eq. (7) does help to regulate L I (and thus r t ) to be around δ.

To further demonstrate the effectiveness of the stabilization technique, we compare the learning curves of Ours (w stab) and Ours (w/o stab) in FIG4 .

It is observed that for the initial 10K samples of the five cases, the success rates of Ours (w/o stab) are comparable to those of Ours (w stab).

However, their performance degrade drastically during the rest of the training phase.

This observation confirms that the stabilization technique does contribute significantly to our adversarial exploration strategy.

Although most of the DRL works suggest that the rewards should be re-scaled or clipped within a range (e.g., from -1 to 1), the unbounded rewards do not introduce any issues during the training process of our experiments.

The empirical rationale is that the rewards received by the DRL agent are regulated by Eq. FORMULA8 to be around δ, as described in Section 4.5 and depicted in FIG3 .

Without the stabilization technique, however, the learning curves of the inverse dynamics model degrade drastically (as illustrated in FIG1 , even if the reward clipping technique is applied.

Influence of δ.

FIG5 compares the learning curves of the imitator for different values of δ.

For instance, Ours(0.1) corresponds to δ = 0.1.

It is observed that for most of the tasks, the success rates drop when δ is set to an overly high or low value (e.g., 100.0 or 0.0), suggesting that a moderate value of δ is necessary for the stabilization technique.

The value of δ can be adjusted dynamically by the adaptive scaling technique presented in BID15 , which is left as our future direction.

From the analysis presented above, we conclude that the proposed adversarial exploration strategy is effective in collecting difficult training data for the imitator.

The analysis also validates that our stabilization technique indeed leads to superior performance, and is capable of guiding the DRL agent to collect moderately hard samples.

This enables the imitator to pursue a stable learning curve.

In this paper, we presented an adversarial exploration strategy, which consists of a DRL agent and an inverse dynamics model competing with each other for self-supervised IL.

The former is encouraged to adversarially collect difficult training data for the latter, such that the training efficiency of the latter is significantly enhanced.

Experimental results demonstrated that our method substantially improved the data collection efficiency in multiple robotic arm and hand manipulation tasks, and boosted the performance of the inverse dynamics model in both low-and high-dimensional observation spaces.

In addition, we validated that our method is generalizable to environments with high-dimensional action spaces.

Moreover, we showed that our method is robust to action space perturbations.

Finally, we provided a set of ablative analysis to validate the effectiveness for each of our design decisions.

In addition to the quantitative results presented above, we further discuss the empirical results qualitatively.

Through visualizing the training progress, we observe that our method initially acts like Random, but later focuses on interacting with the object in FetchPush, FetchSlide, and FetchPickAndPlace.

This phenomenon indicates that adversarial exploration strategy naturally gives rise to a curriculum that improves the learning efficiency, which resembles curriculum learning BID2 .

Another benefit that comes with the phenomenon is that data collection is biased towards interactions with the object.

Therefore, the DRL agent concentrates on collecting interesting samples that has greater significance, rather than trivial ones.

For instance, the agent prefers pushing the object to swinging the robotic arm.

On the other hand, although Curiosity explores the environment very thoroughly in the beginning by stretching the arm into numerous different poses, it quickly overfits to one specific pose.

This causes its forward dynamics model to keep maintaining a low error, making it less curious about the surroundings.

Finally, we observe that the exploratory behavior of Noise does not change as frequently as ours, Random, and Curiosity.

We believe that the method's success in the original paper BID15 ) is largely due to extrinsic rewards.

In the absence of extrinsic rewards, however, the method becomes less effective and unsuitable for data collection, especially in self-supervised IL.

We employ PPO BID22 as the RL agent responsible for collecting training samples because of its ease of use and good performance.

PPO computes an update at every timestep that minimizes the cost function while ensuring the deviation from the previous policy is relatively small.

One of the two main variants of PPO is a clipped surrogate objective expressed as: DISPLAYFORM0 whereÂ is the advantage estimate, and a hyperparameter.

The clipped probability ratio is used to prevent large changes to the policy between updates.

The other variant employs an adaptive penalty on KL divergence, given by: DISPLAYFORM1 where β is an adaptive coefficient adjusted according to the observed change in the KL divergence.

In this work, we employ the former objective due to its better empirical performance.

In the experiments, the inverse dynamics model I(x t , x t+1 |h t , θ I ) of all methods employs the same network architecture.

For low-dimensional observation setting, we use 3 Fully-Connected (FC) layers with 256 hidden units followed by tanh activation units.

For high-dimensional observation setting, we use 3-layer Convolutional Neural Network (CNN) followed by relu activation units.

The CNNs are configured as (32, 8, 4), (64, 4, 2) , and (64, 3, 1), with each element in the 3-tuple denoting the number of output features, width/height of the filter, and stride.

The features extracted by stacked CNNs are then fed forward to a FC with 512 hidden units followed by relu activation units.

For both low-and high-dimensional observation settings, we use the architecture proposed in BID22 .

During training, we periodically update the DRL agent with a batch of transitions as described in Algorithm.

1.

We split the batch into several mini-batches, and update the RL agent with these mini-batches iteratively.

The hyperparameters are listed in Table.

2 (our method).

Our baseline Curiosity is implemented based on the work BID13 .

The authors in BID13 propose to employ a curiosity-driven RL agent BID12 efficiency of data collection.

The curiosity-driven RL agent takes curiosity as intrinsic reward signal, where curiosity is formulated as the error in an agents ability to predict the consequence of its own actions.

This can be defined as a forward dynamics model: DISPLAYFORM0 whereφ(x ) is the predicted feature encoding at the next timestep, φ(x) the feature vector at the current timestep, a the action executed at the current timestep, and θ F the parameters of the forward model f .

The network parameters θ F is optimized by minimizing the loss function L F : DISPLAYFORM1 For low-and high-dimensional observation settings, we use the architecture proposed in BID22 .

The implementation of φ depends on the model architecture of the RL agent.

For low-dimensional observation setting, we implement φ with the architecture of low-dimensional observation PPO.

Note that φ does not share parameters with the RL agent in this case.

For highdimensional observation setting, we share the features extracted by the CNNs of the RL agent, then feed these features to φ which consists of a FC with 512 hidden units followed by relu activation.

The hyperparameters settings can be found in Table.

2(Curiosity).

We directly apply the same architecture in BID15 without any modification.

Please refer to BID15 for more detail.set of demonstrations in FetchReach is relatively difficult with only 100 episodes of demonstrations.

A huge performance gap is observed when the number of episodes is increased to 1,000.

Consequently, Demo(1,000) is selected as our Demo baseline for the presentation of the experimental results in Section 4.

Another advantage is that Demo(1,000) demands less memory than Demo(10,000).

To test the robustness of our method to noisy actions, we add noise to the actions in the training stage.

Letâ t denote the predicted action by the imitator.

The actual noisy action to be executed by the robot is defined as:â t :=â t + N (0, σ), where σ is set as 0.01.

Note thatâ t will be clipped in the range defined by each environment.

In this section, we visualize the effects of our stabilization technique with a list of rewards r in FIG7 .

The rows of Before and After represent the rewards before and after reward shaping, respectively.

The bar on the right-hand side indicates the scale of the reward.

It can be observed in FIG7 that after reward shaping, the rewards are transformed to the negative distance to the specified δ (i.e., 2.5 in this figure) .

As a result, our stabilization technique is able to encourage the DRL agent to pursue rewards close to δ, where higher rewards can be received.

<|TLDR|>

@highlight

A simple yet effective imitation learning scheme that incentivizes exploration of an environment without any extrinsic reward or human demonstration.
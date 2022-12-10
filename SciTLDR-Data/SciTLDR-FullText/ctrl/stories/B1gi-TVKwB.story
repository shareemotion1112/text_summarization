An algorithm is introduced for learning a predictive state representation with off-policy temporal difference (TD) learning that is then used to learn to steer a vehicle with reinforcement learning.

There are three components being learned simultaneously:  (1) the off-policy predictions as a compact representation of state, (2) the behavior policy distribution for estimating the off-policy predictions, and (3) the deterministic policy gradient for learning to act.

A behavior policy discriminator is learned and used for estimating the important sampling ratios needed to learn the predictive representation off-policy with general value functions (GVFs).

A linear deterministic policy gradient method is used to train the agent with only the predictive representations while the predictions are being learned.

All three components are combined, demonstrated and evaluated on the problem of steering the vehicle from images in the TORCS racing simulator environment.

Steering from only images is a challenging problem where evaluation is completed on a held-out set of tracks that were never seen during training in order to measure the generalization of the predictions and controller.

Experiments show the proposed method is able to steer smoothly and navigate many but not all of the tracks available in TORCS with performance that exceeds DDPG using only images as input and approaches the performance of an ideal non-vision based kinematics model.

Predicting the future is an important topic in machine learning and is believed to be an important part of how humans process and interact with the world, cf Clark (2013) .

Study of the brain shows that it is highly predictive of future events and outcomes.

Despite these advances, there is still much work needed to bridge the worlds of predictive learning and control.

Most predictive control approaches learn either a forward model or a backward model Lesort et al. (2018) however these next-step models suffer from compounding errors Sutton (1988) .

This paper introduces a predictive control architecture using one kind of off-policy predictive learning, called general value functions (GVFs) White (2015) Modayil et al. (2012 Schaul & Ring (2013) , that learns to predict the relevant aspects of the environment, decided by an expert, from raw sensor data such as pixel data captured from a camera.

GVFs answer the predictive question, "if I follow policy τ , how much total cumulant will I see in the future?"

The value of the GVF framework is not yet fully understood and realized despite the connections to neuroscience; but some early work has investigated its advantages for predictive representations and found that the representations are compact and general Schaul & Ring (2013) .

An objective of this research is to better understand the value that GVFs have to offer in real-world applications.

Our work is based on the hypothesis that predictive representations are good for generalization Rafols et al. (2005) Schaul & Ring (2013) .

We are motivated by the belief that GVFs, like RL, could allow for behavior that is anticipative of future consequences rather than reactive to the current state.

General value functions (GVFs) are an understudied topic of interest in AI research fields and applications.

There is a considerable focus on understanding how to learn these predictions but limited efforts on understanding how to use them in real applications.

This is unfortunate, as todate, research into applications of GVFs suggest they have potential in real world robotics and its applications Günther et al. (2016) Pilarski et al. (2013) )White (2015 Modayil et al. (2012) .

However, several elements have been missing to apply these predictions to a larger scale problem such as autonomous driving: (1) how to characterize the behavior policy to achieve off-policy learning when it is unknown, (2) what predictions are useful, and (3) how to use those predictions to control the vehicle.

Our objective is two-fold: (1) introduce a novel architecture combining elements of predictive learning, adversarial learning and reinforcement learning, and (2) demonstrate how this architecture can be used to steer a vehicle in a racing simulator.

Steering a vehicle is a challenging problem where the bicycle model is the classical approach Paden et al. (2016) .

However, the bicycle model requires knowing the angle of the vehicle with respect to the road direction in order to compute the desired steering angle.

Steering directly from images has been a long desired goal in autonomous driving where approaches like Salvucci & Gray (2004) advocate for a two point model which inspired the multi-point predictive representation proposed in this paper.

In comparison, learning to regress an image directly to the steering angle in an end-to-end manner has been a recent hot topic Bojarski et al. (2016) Garimella et al. (2017) Chen & Huang (2017 )Sallab et al. (2017 .

However, a serious challenge is ensuring robustness of the controller when learning end-to-end Bojarski et al. (2016) .

In particular, the agent is not typically trained on recovery mode scenarios and so there are generalization and data coverage issues; for this reason, authors Bojarski et al. (2016) introduced augmented images in training by artificially shifting and rotating them to help the network learn to recover with some limited success.

The approach in Chen et al. (2015) learns to predict the current road angle directly from images and then uses a classical steering controller to control the vehicle.

The proposed approach is similar except we predict future road angles and lane centeredness at different temporal horizons which is then passed to a controller module to choose steering angles.

Policy gradient with the predictive state representation is the approach used in this paper but this can also be replaced with other controllers.

This architecture allows for a degree of interpretability in the controller that is not easily achieved with end-to-end approaches Bojarski et al. (2016) despite work on understanding and improving its robustness.

We consider an environment described by a set of states S, a set of actions A, and Markov transition dynamics with probability P (s |s, a) of transitioning to next state s after taking action a from state s.

This setting is nearly identical to a Markov Decision Process (MDP) where the only difference is the absence of a reward signal to maximize.

The goal is to learn an estimator that predicts the return G t of a cumulant c t defined by

where c t is a cumulant signal to be predicted, and 0 ≤ γ t < 1 is the continuation function.

The general value function is defined as

where τ (a|s), γ(s, a, s ), and c(s, a, s ) are the policy, continuation and cumulant functions, respectively, that make up the predictive question where V τ (s) represents the total discounted cumulant starting from state s and acting under policy τ .

Unfortuantely, there are currently no algorithms to learn the predictive question through interaction with the environment; thus, τ , γ, and c are typically defined by an expert.

Cumulants are commonly scaled by a factor of 1 − γ when γ is a constant in non-episodic predictions.

A GVF can be approximated with a function approximator, such as a neural network, parameterized by θ to predict equation 1.

The agent usually collects experience under a different behavior policy µ(a|s) where off-policy policy evaluation methods are needed to learn the GVF.

The parameters θ are optimized with gradient descent minimizing the following loss function

where δ = E[y −v τ (s; θ)|s, a] is the TD error and ρ = τ (a|s)

µ(a|s) is the importance sampling ratio to correct for the difference between the target policy distribution τ and behavior distribution µ. Note that only the behavior policy distribution is corrected rather than the state distribution d µ .

The target y is produced by bootstrapping a prediction Sutton (1988) of the value of the next state following target policy τ given by

where y is a bootstrap prediction using recent parameters θ that are assumed constant in the gradient computation.

Some approaches use older parameters θ of the network to make a bootstrapped prediction to improve stability in the learning Mnih et al. (2013) .

However, this was not found to be necessary when learning GVFs since the target policy is fixed and the learning is simply off-policy policy evaluation.

d µ is the state distribution of the behavior policy µ and the time subscript on c and γ has been dropped to simplify notation.

The gradient of the loss function equation 3 is given by

An alternative approach to using importance sampling ratios ρ is to apply importance resampling Schlegel et al. (2019) .

With importance resampling, a replay buffer D of size N is required and the gradient is multiplied with the average importance sampling ratio of the samples in the buffer

.

The importance resampling gradient is given by

where the transitions in the replay buffer are sampled according to

for transition with state s i and action a i in replay buffer D. This approach is proven to have lower variance than equation 5 with linear function approximation Schlegel et al. (2019) .

An efficient data structure for the replay buffer is the SumTree used in prioritized experience replay Schaul et al. (2016) .

This is a natural approach to learning predictions with deep reinforcement learning since sampling a mini-batch from the replay buffer helps to decorrelate sample updates in deep function approximation Mnih et al. (2013) .

A behavior policy needs to be defined to adequately explore the environment when learning GVFs.

This may be an evolving policy that is learned by RL, a random policy for exploring the environment, or a human driver collecting data safely.

It is common, especially in the case of human drivers, for the behavior policy distribution µ(a|s) of the agent to be unknown.

We propose an algorithm using the density ratio trick to learn the behavior policy distribution in an adversarial way.

It is well suited for problems with low dimensional action spaces like autonomous driving.

The ratio of two probability densities can be expressed as a ratio of discriminator class probabilities that distinguish samples from the two distributions.

Let us define a probability density function η(a|s) for the distribution to compare to the behavior distribution µ(a|s) and class labels y = +1 and y = −1 that denote the class of the distribution that the state action pair was sampled from: µ(a|s) or η(a|s) respectively.

A discriminator g(a, s) is learned that distinguishes state action pairs from these two distributions using the cross-entropy loss.

The ratio of the densities can be computed using only the discriminator g(a, s).

Here we assume that p(y = +1) = p(y = −1).

From this result, we can estimate µ(a|s) withμ(a|s) as followsμ

where η(a|s) is a known distribution over action conditioned on state.

The uniform distribution over the action is independent of state and has the advantage of being effective and easy to implement.

The algorithm for training a GVF off-policy with an unknown behavior distribution is given by Algorithm 1 Off-policy GVF training algorithm with unknown µ(a|s) Compute cumulant c t+1 = c(s t , a t , s t+1 ) 7:

Compute behavior density valueμ(a t |s t ) according to equation 8 9: Compute importance sampling ratio

Sample random minibatch A of transitions (s i , a i , c i+1 , γ i+1 , s i+1 ) from D according to probability Compute

Perform gradient descent step on (y i −v τ (s i ; θ)) 2 according to equation 6 for minibatch A

Sample random minibatch B of state action pairs (s i , a i ) from D according to a uniform probability and assign label y = +1 to each pair 15: Randomly select half the samples in the minibatch B and temporarily replace the label with y = −1 and action with a t ∼ η(a|s)

Update behavior discriminator g(a, s) with modified minibatch B

Let us consider an MDP with a predictive representation φ(s) mapping state s to predictions φ(s).

The reward for the problem is denoted as r. The problem is to find a policy π(a|φ(s)) that maximizes future return or accumulated discounted reward.

We hypothesize that this approach should be easier to train than learning π(a|s) directly for the following reasons:

• the target policy of the predictions τ is fixed making for faster learning • the compact abstraction φ(s) allows for simple (possibly even linear) policy and action-value functions • the cumulant signal c may only be available during training or is expensive to obtain The last advantage is particularly important in autonomous driving where localization techniques often require a collection of expensive sensors and high definition map data that is not always available or easily scalable to a fleet of autonomous vehicles.

In this way, one can train a neural network to map images captured by inexpensive cameras to predictions of lane centeredness and road angle captured by any number of highly accurate but expensive localization approaches with the hopes of generalizing features for lane control.

The agent learns to steer with deterministic policy gradient (DPG) Silver et al. (2014) using the predictions as the state of the agent.

When linear policy function approximation is used, the controller learned is essentially equivalent to a prediction-based PID controller only where there is a deep mapping from images captured by a camera to predictions of the future used to control the vehicle.

Using predictions for PID control is not new; this approach can be used to tackle problems with high temporal delay between the error signal and the corrective actions.

One can also add integral and derivative terms of the predictions to the state space representation of the agent.

Action-value Q π (φ(s), a) and policy π(φ(s)) networks are trained according to DPG Silver et al. (2014) where the action value approximates the expected discounted return, i.e.

.

In addition, a policy network π(φ(s)) produces an action according to the current predictive state representation φ(s) that maximizes the expected discounted return.

Because of the interesting connection between DPG and PID control when linear function approximation is used, the policy network is parameterized as

where ψ is a matrix denoting parameters to be learned by DPG.

They also represent the gain coefficients for a proportional controller which allows for interpretability of the learned parameters.

The action-value network Q π (φ(s), a) is given by a small neural network that maps the predictive state representations φ(s) and action a to an estimate of the action-value in the current state s if a is taken and the optimal policy π followed thereafter.

In autonomous driving, knowing the road curvature ahead can be informative for making decisions to maintain lane centeredness in a tight turn.

In Figure 1 , it is demonstrated how future off-policy predictions of lane centeredness can be used to predict the deviation (or error) between the true center of lane and the projected lane centeredness along the current direction of the vehicle.

These predictions must be off-policy because if they were on-policy they would tell us no information to inform the agent how much adjustment is needed to make corrective actions to stay in the center of the lane.

The lane centeredness α and road angle β are two kinds of predictions that are useful in steering the vehicle as depicted in Figure 2 .

We represent the road curvature as a set of predictions of future lane centeredness α and road angles β at different time horizons.

The predictive state representation is given by the feature vector φ(s)

where V τ α (s t ; γ = γ 0 ) is a GVF prediction under target policy τ , cumulant α (lane centeredness) and continuation function γ 0 while V τ β (s t ; γ = γ 0 ) is a GVF prediction of cumulant β (road angle).

There are m α number of γ functions for predicting lane centeredness and m β number of γ functions for predicting road angle at different temporal horizons.

Because the predictions represent deviations from the desired lane centeredness and road angle, the policy network of DPG can be linear.

The predictive learning approach is applied to the challenging problem of learning to steer a vehicle in the TORCS Wymann et al. (2013) racing environment.

A kinematic-based steering approach Paden et al. (2016) is used as a baseline for all the experiments.

The reward in the TORCS environment is given by r t = v t cos β t where v t is the speed of the vehicle in km/h, β t is the angle between the road direction and the vehicle direction.

A simple scaling factor of 0.01 was applied to the reward in order to reduce the variance of the action-value.

Notice that this reward doesn't force the agent to stay in the center of the lane; however, this strategy is likely a good idea to achieve high total reward on all the test tracks.

The target speed of all the agents is 50 km/h where vehicle speed was controlled by a separate manually tuned PID controller.

The agents were trained on 85% of the 40 tracks available in TORCS.

The rest of the tracks were used for testing (6 in total); all of the results in this section are on the testing tracks which were never presented to the agents during training.

This was done to measure the generalization performance of the policies learned by the agents which can be a serious problem in RL, cf.

Zhao et al. (2019 )Farebrother et al. (2018 .

Tracks where the road was not level were excluded since they proved to be challenging likely because a non-zero action was required to keep the vehicle centered.

In all the figures, blue corresponds to DDPG-Image with only image and speed as input, green corresponds to DDPG-ImageLowDim with images, current speed, α and β provided as input, orange corresponds to the new GVF-DPG approach with image, current speed and last two actions since the target policy of the prediction depends on the last action taken, and red corresponds to the classical front wheel steering model.

A history of two images were provided to each method.

A supervised method of predicting the current lane centeredness and road angle directly from the image was attempted with negative results: the controller learned was consistently unstable.

Results were repeated over 5 runs.

The total score achieved on the test tracks by the agents is plotted in Figure 3 over 1M training iterations.

It is clear that DDPG-ImageLowDim performs best of the learned methods on all test tracks.

This low dimensional information provides ground truth to the agent which may not always be available in all locations; however, it makes a good baseline target for what we hope our proposed method could achieve through generalization.

With DDPG-Image, it is clear that it does not learn to steer from images very well.

Figure 4 where DDPG-Image consistently converges to a solution that oscillates between the extreme left and right steering actions very rapidly.

This oscillation is so extreme that the agent is unable to achieve the target speed of 50 km/h; instead it travels at 40 km/h on average for all the test tracks.

The performance gap also suggests that DDPG-ImageLowDim may be relying more on the low dimensional lane information rather than the image.

To highlight how uncomfortable the two DDPG agents drive compared to the GVF-DPG agent, Figure  4 shows the standard deviation of the change in action during 1M iterations of training.

On most tracks, it is apparent that the GVF-DPG approach controls the vehicle more smoothly than the other learned methods.

The performance of the individual test tracks is given in the following Figure 5 .

The GVF-DPG approach does not steer successfully on all test tracks: it fails immediately on wheel-2, part-way through on a-wheelway, and drives well on most of alpine-2.

The DDPG-Image fails to complete dirt-4, wheel-2, spring, and a-speedway.

Finally, DDPG-ImageLowDim successfully completes all the test tracks; however, the agent has a strong bias to the left side of the track.

The GVF-DPG agent often follows the classical controller relatively well except on the tracks where the agent fails.

This suggests that using a predictive representation of lane centeredness α and road angle β achieves closer performance to a classical controller than an end-to-end learned approach.

However, more work is needed to improve the generalization abilities of the approach.

The learning curves for the predictors and the policy gradient agents is given in the following Figure 6 .

It is interesting that the learning curves of the action-value function estimator of the GVF-DPG agent is much smaller than the other agents and quite smooth.

The reason is believed to be because the predictive state representation is constrained to values between [−1, +1] acting as a sort of regularizer to the state representation of the agent.

The learning curve of the DDPG-ImageLowDim however eventually approaches the low error of the GVF-DPG agent.

The predictors converge relatively quickly as shown in Figure 6 (b).

The behavior estimator in Figure 6 (c) stabilizes relatively quickly during learning as well; it is postulated that the error does not decrease further since the behavior policy is changing slowly over time and the behavior estimator must track this change.

A method of learning a predictive representation off-policy is presented where the behavior policy distribution is estimated via an adversarial method employing the density ratio trick.

It is demonstrated that deep off-policy predictions can be learned with a deep behavior policy estimation to predict future lane centeredness and road angles from images.

The predictive representation is learned with linear deterministic policy gradient.

All of these components are combined together in a framework called GVF-DPG and learned simultaneously on the challenging problem of steering a vehicle in TORCS from only images.

The results show that the GVF-DPG is able to steer smoothly with less change in action and achieve better performance than DDPG from only images and similar performance to the kinematics model in several but not all of the test tracks.

This work is also a demonstration that we can learn off-policy predictions, characterize the behavior policy and learn the controller all at the same time despite the challenges of the behavior policy evolving with the agent and the predictive state representation changing over time.

Our work demonstrates that a learned prediction-based vision-only steering controller could potentially be viable with more work on improving the generalizability of the off-policy predictions.

This work supports the predictive state representation hypothesis in Rafols et al. (2005) that deep predictions can improve the generalization of RL to new road environments when using only images as input.

For future work, we hope to study how to learn the question for the predictive state representation: τ , γ, and c. Moreover, because the behavior policy is unknown and estimated, our results suggest that collecting real-world human driving to train predictions off-policy without the need for a simulator could be a viable approach to steering a vehicle from images.

This is potentially advantageous since the human driver can explore the road safely.

The predictive learning approach presented in this paper is called GVF-DPG (general value function deterministic policy gradient).

Exploration followed the same approach as Lillicrap et al. (2016) where an Ornstein Uhlenbeck process Uhlenbeck & Ornstein (1930) is used to explore the track where the parameters of the process (θ = 0.01, σ = 0.01) were tuned to provide a gradual wandering behavior on the track without excessive oscillations in the action.

The reason is to improve the learning of the off-policy predictions for GVF-DPG since the behavior policy µ(a|s) is closer to the target policy τ (a|s) of the predictions.

The GVF-DPG approach learned 8 predictions: 4 predictions of lane centeredness α, and 4 predictions of road angle β.

Each of the 4 predictions had different values of γ for different temporal horizons: 0.5, 0.9, 0.95, 0.97.

This allowed the agent to make short-term, medium term and long term predictions of lane centeredness α and road angle β.

The GVF predictors all share the same deep convolutional neural network where the convolutional layers are identical to the architecture in Bojarski et al. (2016) followed by three fully connected layers of 512, 384 and 8 outputs, respectively.

The behavior estimator µ(a|s) also uses a very similar neural network with identical convolutional layers followed by three fully connected layers of 512, 256, and 1 output, respectively.

The GVF predictors and behavior estimator are both given the current image, previous image, current speed and the last two actions taken.

The policy network of the DPG agent is linear with respect to the predictions.

The action-value network is a small 4 layer network of 64x64x32x1 and outputs the change in the action from the previous action.

The predictions are supplied to the agent and the last action is needed to compute the next action.

All networks use rectified linear unit activations except for the last layers which are linear.

The learning rates for the linear policy network, action-value network, predictors and behavior policy network were 1e −4 , 1e −8 , 1e −4 , and 1e −4 respectively.

It was noted that the action-value learning rate needed to be small in order to faciliate two time-scale learning of the predictive representation and the action-values.

The GVF-DPG approach is compared to two DDPG (deep deterministic policy gradient) Lillicrap et al. (2016) baselines where the differences are given by only the information provided in the state.

The first method is a vision-based approach where the image and current speed is provided to the agent; the only information available to the agent about the road is supplied via images.

This proved challenging for the DDPG agent.

Thus, a second agent was trained with the image and the lane centeredness α and road angle β that the GVF-DPG agent had access to during training.

This is cheating in a sense because the agent must have the lane centeredness and road angle available at all times including at test time, whereas the GVF-DPG agent is hopefully learning features that are generalizable to unseen roads where this information may not be always available.

However, it provides a good target for evaluating the GVF-DPG agent.

An Ornstein Uhlenbeck process Uhlenbeck & Ornstein (1930) is used to explore the track (θ = 0.05, σ = 0.05).

In previous works with DDPG, target networks were utilized for both the action-value and the policy networks.

The target networks are copies of the action-value and policy networks that are updated more slowly in order make the bootstrapped prediction of the action-values more stable Lillicrap et al. (2016) .

However, in our experiments it was found that removing the target networks improved learning and so no target networks were deployed in any of the following experiments.

Identical network architectures were used for the policy and action-value networks of the DDPG agents as was used for the behavior network for GVF-DPG.

The low dimensional state information was feed through a separate branch of 12 neurons that were then merged with the fully connected layer of 512 neurons.

This method of merging can present challenges due to mismatching statistical properties of the branches but that did not seem to present a significant challenge in learning.

Future work would be to find better ways to bridge these two different pieces of information.

The hyperbolic tangent activation was used for output layer of the policy network, the linear activation was used for the output layer of the action-value network and rectified linear activation was used everywhere else.

During training, a track is selected randomly according to a priority sampling method, since the tracks were not balanced, and the agent is allowed to interact with the environment and learn until termination.

Termination occurs when either the agent leaves the lane or the maximum number of steps has been reached (1200 steps = 120 seconds).

A priority sampling method was chosen so that tracks that were more difficult were selected more often.

The probability of sampling a track i is given by e − n i κ N j=1 e − n j κ (11) where n i is the number of steps that the agent was able to achieve in the last episode for that track and κ controls the spread of the distribution.

A value of κ = 1 N N j=1 n j was found to perform well.

The initial probabilities are equal for all tracks.

The TORCS environment was modified to provide higher resolution images in grayscale rather than RGB with most of the image above the horizon cropped out of the image.

The grayscale images were 128 pixels wide by 64 pixels high.

This allowed the agent to see more detail farther away which is very helpful in making long term predictions which is beneficial to both policy gradient methods and predictive learning.

<|TLDR|>

@highlight

An algorithm to learn a predictive state representation with general value functions and off-policy learning is applied to the problem of vision-based steering in autonomous driving.
As deep reinforcement learning is being applied to more and more tasks, there is a growing need to better understand and probe the learned agents.

Visualizing and understanding the decision making process can be very valuable to comprehend and identify problems in the learned behavior.

However, this topic has been relatively under-explored in the reinforcement learning community.

In this work we present a method for synthesizing states of interest for a trained agent.

Such states could be situations (e.g. crashing or damaging a car) in which specific actions are necessary.

Further, critical states in which a very high or a very low reward can be achieved (e.g. risky states) are often interesting to understand the situational awareness of the system.

To this end, we learn a generative model over the state space of the environment and use its latent space to optimize a target function for the state of interest.

In our experiments we show that this method can generate insightful visualizations for a variety of environments and reinforcement learning methods.

We explore these issues in the standard Atari benchmark games as well as in an autonomous driving simulator.

Based on the efficiency with which we have been able to identify significant decision scenarios with this technique, we believe this general approach could serve as an important tool for AI safety applications.

Humans can naturally learn and perform well at a wide variety of tasks, driven by instinct and practice; more importantly, they are able to justify why they would take a certain action.

Artificial agents should be equipped with the same capability, so that their decision making process is interpretable by researchers.

Following the enormous success of deep learning in various domains, such as the application of convolutional neural networks (CNNs) to computer vision BID19 BID18 BID20 BID33 , a need for understanding and analyzing the trained models has arisen.

Several such methods have been proposed and work well in this domain, for example for image classification BID35 BID44 BID8 , sequential models BID12 or through attention BID41 .Deep reinforcement learning (RL) agents also use CNNs to gain perception and learn policies directly from image sequences.

However, little work has been so far done in analyzing RL networks.

We found that directly applying common visualization techniques to RL agents often leads to poor results.

In this paper, we present a novel technique to generate insightful visualizations for pre-trained agents.

Currently, the generalization capability of an agent is-in the best case-evaluated on a validation set of scenarios.

However, this means that this validation set has to be carefully crafted to encompass as many potential failure cases as possible.

As an example, consider the case of a self-driving agent, where it is near impossible to exhaustively model all interactions of the agent with other drivers, pedestrians, cyclists, weather conditions, even in simulation.

Our goal is to extrapolate from the training scenes to novel states that induce a specified behavior in the agent.

In our work, we learn a generative model of the environment as an input to the agent.

This allows us to probe the agent's behavior in novel states created by an optimization scheme to induce specific actions in the agent.

For example we could optimize for states in which the agent sees the only option as being to slam on the brakes; or states in which the agent expects to score exceptionally low.

Visualizing such states allows to observe the agent's interaction with the environment in critical scenarios to understand its shortcomings.

Furthermore, it is possible to generate states based on an objective function specified by the user.

Lastly, our method does not affect and does not depend on the training of the agent and thus is applicable to a wide variety of reinforcement learning algorithms.

We divide prior work into two parts.

First we discuss the large body of visualization techniques developed primarily for image recognition, followed by related efforts in reinforcement learning.

In the field of computer vision, there is a growing body of literature on visualizing features and neuron activations of CNNs.

As outlined in BID10 , we differentiate between saliency methods, that highlight decision-relevant regions given an input image, methods that synthesize an image (pre-image) that fulfills a certain criterion, such as activation maximization BID7 or input reconstruction, and methods that are perturbation-based, i.e. they quantify how input modification affects the output of the model.

Saliency methods typically use the gradient of a prediction or neuron at the input image to estimate importance of pixels.

Following gradient magnitude heatmaps BID35 and class activation mapping BID46 , more sophisticated methods such as guided backpropagation BID36 BID24 , excitation backpropagation BID45 , GradCAM BID34 and GradCAM++ BID4 have been developed.

BID47 distinguish between regions in favor and regions speaking against the current prediction.

BID37 distinguish between sensitivity and implementation invariance.

An interesting observation is that such methods seem to generate believable saliency maps even for networks with random weights BID0 .

BID14 show that saliency methods do not produce analytically correct explanations for linear models and further discuss reliability issues in BID13 .Perturbation Methods Perturbation methods modify a given input to understand the importance of individual image regions.

BID44 slide an occluding rectangle across the image and measure the change in the prediction, which results in a heatmap of importance for each occluded region.

This technique is revisited by BID8 who introduce blurring/noise in the image, instead of a rectangular occluder, and iteratively find a minimal perturbation mask that reduces the classifier's score, while BID5 train a network for masking salient regions.

Input Reconstruction As our method synthesizes inputs to the agent, the most closely related work includes input reconstruction techniques.

BID21 reconstruct an image from an average of image patches based on nearest neighbors in feature space.

BID23 propose to reconstruct images by inverting representations learned by CNNs, while BID6 train a CNN to reconstruct the input from its encoding.

When maximizing the activation of a specific class or neuron, regularization is crucial because the optimization procedure-starting from a random noise image and maximizing an output-is vastly under-constrained and often tends to generate fooling examples that fall outside the manifold of realistic images BID29 .

In BID24 total variation (TV) is used for regularization, while BID1 propose an update scheme based on Sobolev gradients.

In BID29 Gaussian filters are used to blur the pre-image or the update computed in every iteration.

Since there are usually multiple input families that excite a neuron, BID30 propose an optimization scheme for the distillation of these clusters.

BID38 show that even CNNs with random weights can be used for regularization.

More variations of regularization can be found in BID31 BID17 .

Instead of regularization, BID27 use a denoising autoencoder and optimize in latent space to reconstruct pre-images for image classification.

In deep reinforcement learning however, feature visualization is to date relatively unexplored.

BID43 apply t-SNE BID22 on the last layer of a deep Q-network (DQN) to cluster states of behavior of the agent.

BID25 also use t-SNE embeddings for visualization, while BID9 examine how the current state affects the policy in a vision-based approach using saliency methods.

BID39 use saliency methods from BID35 to visualize the value and advantage function of their dueling Q-network.

Interestingly, we could not find prior work using activation maximization methods for visualization.

In our experiments we show that the typical methods fail in the case of RL networks and generate images far outside the manifold of valid game states, even with all typical forms of regularization.

In the next section, we will show how to overcome these difficulties.

We will first introduce the notation and definitions that will be used through out the remainder of the paper.

We formulate the reinforcement learning problem as a discounted, infinite horizon Markov decision process (S, A, γ, P, r), where at every time step t the agent finds itself in a state s t ∈ S and chooses an action a t ∈ A following its policy π θ (a|s t ).

Then the environment transitions from state s t to state s t+1 given the model P (s t+1 |s t , a t ).

Our goal is to visualize RL agents given a user-defined objective function, without adding constraints on the optimization process of the agent itself, i.e. assuming that we are given a previously trained agent with fixed parameters θ.

We approach visualization via a generative model over the state space S and synthesize states that lead to an interesting, user-specified behavior of the agent.

This could be, for instance, states in which the agent expresses high uncertainty regarding which action to take or states in which it sees no good way out.

This approach is fundamentally different than saliency-based methods as they always need an input for the test-set on which the saliency maps can be computed.

The generative model constrains the optimization of states to induce specific agent behavior.

Often in feature visualization for CNNs, an image is optimized starting from random noise.

However, we found this formulation too unconstrained, often ending up in local minima or fooling examples ( Figure 4a ).

To constrain the optimization problem we learn a generative model on a set S of states generated by the given agent that is acting in the environment.

The model is inspired by variational autoencoders (VAEs) BID16 and consists of an encoder f (s) = (µ, σ) ∈ R 2×n that maps inputs to a Gaussian distribution in latent space and a decoder g(µ, σ, z) =ŝ that reconstructs the input.

The training of our generator has three objectives.

First, we want the generated samples to be close to the manifold of valid states s. To avoid fooling examples, the samples should also induce correct behavior in the agent and lastly, sampling states needs to be efficient.

We encode these goals in three corresponding loss terms.

DISPLAYFORM0 ( 1) The role of L p (s) is to ensure that the reconstruction g(f (s), z) is close to the input s such that g(f (s), z) − s 2 2 is minimized.

We observe that in the typical reinforcement learning benchmarks, such as Atari games, small details-e.g.

the ball in Pong or Breakout-are often critical for the decision making of the agent.

However, a typical VAE model tends to yield blurry samples that are not able to capture such details.

To address this issue, we model the reconstruction error L p (s) with an attentive loss term, which leverages the saliency of the agent to put focus on critical regions of the reconstruction.

The saliency maps are computed by guided backpropagation of the policy's gradient with respect to the state.

DISPLAYFORM1 Since we are interested in the actions of the agent on synthesized states, the second objective L a (s) is used to model the perception of the agent: DISPLAYFORM2 where A is a generic formulation of the output of the agent.

For a DQN for example, π(s) = max a A(s) a , i.e. the final action is the one with the maximal Q-value.

This term encourages the reconstructions to be interpreted by the agent the same way as the original inputs s. The last term KL( f (s), N (0, I n ) ) ensures that the distribution predicted by the encoder f stays close to a Gaussian distribution.

This allows us to initialize the optimization with a reasonable random vector later and forms the basis of a regularizer.

Thus, after training, the model approximates the distribution of states p(s) by sampling z from N (0, I n ).

We will now use the generator inside an optimization scheme to generate state samples that satisfy a user defined target objective.

Training a generator with the objective function of Equation 1 allows us to sample states that are not only visually close to the real ones, but which the agent can also interpret and act upon as if they were states from a real environment.

We can further exploit this property and formulate an energy optimization scheme to generate samples that satisfy a specified objective.

The energy operates on the latent space of the generator and is defined as the sum of a target function T on agent's policy and a regularizer R DISPLAYFORM0 The target function can be defined freely by the user and depends on the agent that is being visualized.

For a DQN, one could for example define T as the Q-value of a certain action, e.g. pressing the brakes of a car.

In section 3.3, we show several examples of targets that are interesting to analyze.

The regularizer R can again be chosen as the KL divergence between x and the normal distribution: DISPLAYFORM1 forcing the samples that are drawn from the distribution x to be close to the Gaussian distribution that the generator was trained with.

We can optimize Equation 4 with gradient descent on x = (σ, µ).

Depending on the agent, one can define several interesting target functions T .

For a DQN the previously discussed action maximization is interesting to find situations in which the agent assigns a high value to a certain action e.g. T lef t (s) = −A lef t (s).

Other states of interest are those to which the agent assigns a low (or high) value for all possible actions A(s) = q = (q 1 , . . .

, q m ).

Consequently, one can optimize towards a low Q-value for the highest valued action with the following objective: DISPLAYFORM0 where β > 0 controls the sharpness of the soft maximum formulation.

Analogously, one can maximize the lowest Q-value with T + (q) = −T − (−q).

We can also optimize for interesting situations in which one action is of very high value and another is of very low value by defining DISPLAYFORM1 4 EXPERIMENTSIn this section we thoroughly evaluate and analyze our method on Atari games BID2 using the OpenAI Gym BID3 ) and a driving simulator.

We present qualitative results for three different reinforcement learning algorithms, show examples on how the method helps finding flaws in an agent, analyze the loss contributions and compare to previous techniques.

Implementation details In all our experiments we use the same factors to balance the loss terms in Equation 6: λ = 10 −4 for the KL divergence and η = 10 −3 for the agent perception loss.

The generator is trained on 10, 000 frames (using the agent and an -greedy policy with = 0.1).

Optimization is done with Adam BID15 with a learning rate of 10 −3 and a batch size of 16 for 2000 epochs.

Training takes approximately four hours on a Titan Xp.

Our generator uses a latent space of 100 dimensions, and consists of four encoder stages comprised of a 3 × 3 convolution with stride 2, batch-normalization BID11 and ReLU layer.

The starting number of + generates high reward and T − low reward states; T ± generates states in which one action is highly beneficial and another is bad.filters is 32 and is doubled at every stage.

A fully connected layer is used for mean and log-variance prediction.

Decoding is inversely symmetric to encoding, using deconvolutions and halving the number of channels at each of the four steps.

For the experiments on the Atari games we train a double DQN BID39 for two million steps with a reward discount factor of 0.95.

The input size is 84 × 84 pixels.

Therefore, our generator performs up-sampling by factors of 2, up to a 128 × 128 output, which is then center cropped to 84 × 84 pixels.

The agents are trained on grayscale images, for better visual quality however, our generator is trained with color frames and convert to grayscale using a differentiable, weighted sum of the color channels.

In the interest of reproducibility we will make the visualization code available.

In FIG0 , we show qualitative results from various Atari games using different target functions T , as described in Section 3.3.

From these images we can validate that the general visualizations that are obtained from the method are of good quality and can be interpreted by a human.

T + generates generally high value states independent of a specific action (first row of FIG0 ), while T − generates low reward situations, such as close before losing the game in Seaquest FIG0 .e) or when there are no points to score FIG0 .

Critical situations can be found by maximizing the difference Figure 2 : Seaquest with ACKTR.

Visualization results for a network trained with ACKTR on Seaquest.

The objective is T ± indicating situations that can be rewarding but also have a low scoring outcome.

The generated states show low oxygen or close proximity to enemies.

between lowest and highest estimated Q-value with T ± .

In those cases, there is clearly a right and a wrong action to take.

In Name This Game FIG0 this occurs when close to the air refill pipe, which prevents suffocating under water; in Kung Fu Master when there are enemies coming from both sides FIG0 ), the order of attack is critical, especially since the health of the agent is low (yellow/blue bar on top).

An example of maximizing the value of a single action (similar to maximizing the confidence of a class when visualizing image classification CNNs) can be seen in FIG0 .f) where the agent sees moving left and avoiding the enemy as the best choice of action.

To show that this visualization technique generalizes over different RL algorithms, we also visualize ACKTR BID40 .

We use the code and pretrained models from a public repository BID17 and train our generative model with the same hyperparameters as above and without any modifications on the agent.

We present the T ± objective for the ACKTR agent in Figure 2 to visualize states with both high and low rewards, for example low oxygen (surviving vs. suffocating) or close proximity to enemies (earning points vs. dying).

Compared to the DQN visualizations the ACKTR visualizations, are almost identical in terms of image quality and interpretability.

This supports the notion that our proposed approach is independent of the specific RL algorithm.

Analyzing the visualizations on Seaquest, we make an interesting observation.

When maximizing the Q-value for the actions, in many samples we see a low or very low oxygen meter.

In these cases the submarine would need to ascend to the surface to avoid suffocation.

Although the up action is the only sensible choice in this case, we also obtain visualized low oxygen states for all other actions.

This implies that the agent has not understood the importance of resurfacing when the oxygen is low.

We then run several roll outs of the agent and see that the major cause of death is indeed suffocation and not collision with enemies.

This shows the impact of visualization, as we are able to understand a flaw of the agent.

Although it would be possible to identify this flawed behavior directly by analyzing the 10, 000 frames of training data for our generator, it is significantly easier to review a handful of samples from our method.

Further, as the generator is a generative model, we can synthesize states that are not part of its training set.

In this section we analyze the three loss terms of our generative model.

The human perceptual loss is weighted by the (guided) gradient magnitude of the agent in Equation 2.

In Figure 3 we visualize this mask for a DQN agent for random frames from the dataset.

The masks are blurred with an averaging filter of kernel size 5.

We observe that guided backpropagation results in precise saliency maps focusing on player and enemies that then focus the reconstructions on what is important for the agent.

To study the influence of the loss terms we perform an experiment in which we evaluate the agent not on the real frames but on their reconstructions.

If the reconstructed frames are perfect, the agent with generator goggles achieves the same score as the original agent.

We can use this metric to understand the quantitative influence of the loss terms.

In Pong, the ball is the most important visual aspect of Figure 3: Weight Visualization.

We visualize the weighting (second row) of the reconstruction loss from Equation 2 for eight randomly drawn samples (first row) of the dataset.

Most weight lies on the player's submarine and close enemies, supporting their importance for the decision making.the game for decision making.

In TAB0 we see that the VAE baseline scores much lower than our model.

Since the ball is very small, it is mostly ignored by the reconstruction loss of a VAE.

Our formulation is built to regain the original performance of the agent.

Overall, we see that our method always improves over the baseline but does not always match the original performance.

For image classification tasks, activation maximization works well when optimizing the pre-image directly BID23 BID1 ).

However we find that for reinforcement learning, the features learned by the network are not complex enough to reconstruct meaningful pre-images, even with sophisticated regularization techniques.

The pre-image converges to a fooling example maximizing the class but being far away from the manifold of states of the environment.

In Figure 4 .a we compare our results with the reconstructions generated using the method of BID1 for a DQN agent.

We obtain similarly bad pre-images with TV-regularization BID24 , Gaussian blurring BID29 and other regularization tricks such as random jitter, rotations, scaling and cropping BID31 .

One explanation for the low performance of standard methods for activation maximization can be found when visualizing the first layer filters of Atari agents.

We show Conv1 of a DQN in Figure 5 .

We stack the representations for the temporal component vertically and the 32 filters horizontally.

Looking at the filters of Pong, we make two observations.

The agent only needs five distinct filters to play the game and due to the strong temporal changes in patterns, it is mostly focused on moving parts.

This aligns with the reactive game play of pong and its visual simplicity.

Instead, a complex game such as Seaquest uses all available filters.

These observations bring up interesting points.

Indeed the CNN architecture of BID26 contains enough filters for the most visually complex games that are not needed for the simpler environments such as Pong.

Also, the temporal component seems to only be important for some of the environments.

This, and the strong visible differences between the weights of different environments leaves the open question whether a common feature extractor exists that could work for all games.

However, poor Conv1 weights are an indicator that it is easier to distract the model by activating "unused" filters with imperceptible noise which can be the cause for the poor visualizations with classical activation maximization techniques.

We have created a 3D driving simulation environment and trained an A2C agent maximizing speed while avoiding pedestrians that are crossing the road.

The agent is trained with four temporal semantic segmentation frames (128 × 128 pixels) as input ( Figure 6 ).

With this safety-critical application we can assess multiple points.

First, driving is a continuous task in a much more complex environment than Atari games.

Second, we use the simulator to build two custom environments and validate that we can identify problematic behavior in the agent.

Specifically, we train the agent in a "reasonable pedestrians" environment, where pedestrians cross the road carefully, when no car is coming or at traffic lights.

With these choices, we model data collected in the real world, where it is unlikely that people unexpectedly run in front of the car.

We visualize states in which the agent expects a low future return (T − objective) in Figure 6 .

It shows that the agent is aware of other cars, traffic lights and intersections.

However, there are no generated states in which the car is about to collide with a person, meaning that the agent does not recognize the criticality of pedestrians.

To verify our suspicion, we test this agent in a "distracted pedestrians" environment where people cross the road looking at their phones without paying attention to approaching cars.

We find that the agent does indeed run over humans.

With this experiment, we show that our visualization technique can identify biases in the training data just by critically analyzing the sampled frames.

While one could simply examine the experience replay buffer to find scenarios of interest, our approach allows unseen scenarios to be synthesized.

To quantitatively evaluate the assertion that our generator is capable of generating novel states, we sample states and compare them to their closest frame in the training set under an MSE metric.

We count a pixel as different if the relative difference in a channel exceeds 25% and report the histogram in Table 2 .

The results show that there are very few samples that are very close to the training data.

On average a generated state is different in 25% of the pixels, which is high, considering the overall common layout of the road, buildings and sky.

Figure 6 : Driving simulator.

We show 16 samples for the T − objective of an agent trained in the reasonable pedestrians environment.

From these samples one can infer that the agent is aware of traffic lights (red) and other cars (blue) but has very likely not understood the severity of hitting pedestrians (yellow).

Deploying this agent in the distracted pedestrians environment shows that the agent indeed collides with people that cross the road in front of the agent.

Table 2 : Synthesizing unseen states.

We compare generated samples to their closest neighbor in the training set and compute the percentage of pixels whose values differ by at least 25%, e.g. 73% of the synthesized samples differ in more than 20% pixels in comparison to their closest training sample.

DISPLAYFORM0

We have presented a method to synthesize inputs to deep reinforcement learning agents based on generative modeling of the environment and user-defined objective functions.

Training the generator to produce states that the agent perceives as those from the real environment enables optimizing its latent space to sample states of interest.

We believe that understanding and visualizing agent behavior in safety critical situations is a crucial step towards creating safer and more robust agents using reinforcement learning.

We have found that the methods explored here can indeed help accelerate the detection of problematic situations for a given learned agent.

As such we intend to build upon this work.

To show an unbiased and wide variety of results, in the following, we will show four random samples generated by our method for a DQN agent trained on many of the Atari benchmark environments.

We show visualizations optimized for a meaningful objective for each game (e.g. not optimizing for unused buttons).

All examples were generated with the same hyperparameter settings.

Please note that for some games better settings can be found.

Some generators on visually more complex games would benefit from longer training to generate sharper images.

Our method is able to generate reasonable images even when the DQN was unable to learn a meaningful policy such as for Montezuma's revenge.

We show two additional objectives maximizing/minimizing the expected reward of the state under a random action: S + (q) = m i=1 q i and S − (q) = −S + (q).

Results in alphabetical order.

<|TLDR|>

@highlight

We present a method to synthesize states of interest for reinforcement learning agents in order to analyze their behavior. 

@highlight

This paper proposes a generative model of visual observations in RL that is capable of generating observations of interests.

@highlight

An approach for visualizing states of interest that involves a variational autoencoder that learns to reconstruct state space and an optimization step that finds conditioning parameters to generate synthetic images.
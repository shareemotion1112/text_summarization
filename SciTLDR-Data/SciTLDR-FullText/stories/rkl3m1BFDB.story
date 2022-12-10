Saliency maps are often used to suggest explanations of the behavior of deep rein- forcement learning (RL) agents.

However, the explanations derived from saliency maps are often unfalsifiable and can be highly subjective.

We introduce an empirical approach grounded in counterfactual reasoning to test the hypotheses generated from saliency maps and show that explanations suggested by saliency maps are often not supported by experiments.

Our experiments suggest that saliency maps are best viewed as an exploratory tool rather than an explanatory tool.

Saliency map methods are a popular visualization technique that produce heatmap-like output highlighting the importance of different regions of some visual input.

They are frequently used to explain how deep networks classify images in computer vision applications (Simonyan et al., 2014; Springenberg et al., 2014; Shrikumar et al., 2017; Smilkov et al., 2017; Selvaraju et al., 2017; Zhang et al., 2018; Zeiler & Fergus, 2014; Ribeiro et al., 2016; Dabkowski & Gal, 2017; Fong & Vedaldi, 2017) and to explain how agents choose actions in reinforcement learning (RL) applications (Bogdanovic et al., 2015; Wang et al., 2015; Zahavy et al., 2016; Greydanus et al., 2017; Iyer et al., 2018; Sundar, 2018; Yang et al., 2018; Annasamy & Sycara, 2019) .

Saliency methods in computer vision and reinforcement learning use similar procedures to generate these maps.

However, the temporal and interactive nature of RL systems presents a unique set of opportunities and challenges.

Deep models in reinforcement learning select sequential actions whose effects can interact over long time periods.

This contrasts strongly with visual classification tasks, in which deep models merely map from images to labels.

For RL systems, saliency maps are often used to assess an agent's internal representations and behavior over multiple frames in the environment, rather than to assess the importance of specific pixels in classifying images.

Despite their common use to explain agent behavior, it is unclear whether saliency maps provide useful explanations of the behavior of deep RL agents.

Some prior work has evaluated the applicability of saliency maps for explaining the behavior of image classifiers (Adebayo et al., 2018; Kindermans et al., 2019; Samek et al., 2016) , but there is not a corresponding literature evaluating the applicability of saliency maps for explaining RL agent behavior.

In this work, we develop a methodology grounded in counterfactual reasoning to empirically evaluate the explanations generated using saliency maps in deep RL.

Specifically, we:

C1 Survey the ways in which saliency maps have been used as evidence in explanations of deep RL agents.

C2 Describe a new interventional method to evaluate the inferences made from saliency maps.

C3 Experimentally evaluate how well the pixel-level inferences of saliency maps correspond to the semantic-level inferences of humans.

(a) (b) (c) Figure 1 : (a) A perturbation saliency map from a frame in Breakout, (b) a saliency map from the same model and frame with the brick pattern reflected across the vertical axis, and (c) a saliency map from the same model and frame with the ball, paddle and brick pattern reflected across the vertical axis.

The blue and red regions represent their importance in action selection and reward estimation from the current state, respectively.

The pattern and intensity of saliency around the channel is not symmetric in either reflection intervention.

Consider the saliency maps generated from a deep RL agent trained to play the Atari game Breakout.

The goal of Breakout is to use the paddle to keep the ball in play so it hits bricks, eliminating them from the screen.

Figure 1a shows a sample frame with its corresponding saliency.

The red regions represent the importance of these pixels for action selection.

Note the high salience on the missing section of bricks ("tunnel") in Figure 1a .

Creating a tunnel to target bricks at the top layers is one of the most high-profile examples of agent behavior being explained according to semantic, human-understandable concepts (Mnih et al., 2015) .

Given the intensity of saliency on the tunnel in 1a, it may seem reasonable to infer that this saliency map provides evidence that the agent has learned to aim at tunnels.

If this is the case, moving the horizontal position of the tunnel should lead to similar saliency patterns on the new tunnel.

However, Figures 1b and 1c show that the salience pattern is not preserved.

Neither the presence of the tunnel, nor the relative positioning of the ball, paddle, and tunnel, are responsible for the intensity of the saliency observed in Figure 1a .

Examining how some of the technical details of reinforcement learning interact with saliency maps can help understand both the potential utility and the potential pitfalls of interpreting saliency maps.

RL methods enable agents to learn how to act effectively within an environment by repeated interaction with that environment.

Certain states in the environment give the agent positive or negative reward.

The agent learns a policy, a mapping between states and actions according to these reward signals.

The goal is to learn a policy that maximizes the discounted sum of rewards received while acting in the environment (Sutton et al., 1998) .

Deep reinforcement learning uses deep neural networks to represent policies.

These models enable interaction with environments requiring high-dimensional state inputs (e.g., Atari games).

Consider the graphical model in Figure 2a representing the deep RL system for a vision-based game environment.

The environment maintains some (usually latent) game state.

Some function F produces a high-dimensional pixel representation of game state ("Pixels").

The learned network takes this pixel image and produces logits used to select an action.

Temporally extended sequences of this action selection procedure result in observed agent behavior.

Saliency maps are produced by performing some kind of intervention M on this system and calculating the difference in logits produced by the original and modified images.

The interventions used to calculate saliency for deep RL are performed at the pixel level (red node and arrow in Figure 2a ).

These interventions change the conditional probability distribution of "Pixels" by giving it another parent (Pearl, 2000) .

Functionally, this can be accomplished through a variety of means, including changing the color of the pixel (Simonyan et al., 2014) , adding a gray mask (Zeiler & Fergus, 2014) , blurring a small region (Greydanus et al., 2017) , or masking objects with the background color (Iyer et al., 2018) .

The interventions M are used to simulate the effect of the absence of the pixel(s) on the network's output.

Note however that these interventions change the image in a way that is inconsistent with the generative process F .

They are not "naturalistic" modifications.

This type of intervention produces images for which the learned network function may not be well-defined.

To form explanations of agent behavior, human observers combine information from saliency maps, agent behavior, and semantic concepts.

Figure 2b shows a system diagram of how these components interact.

Hypotheses about the semantic features identified by the learned policy are proposed by reasoning backwards about what representation might jointly produce the observed saliency pattern and agent behavior.

Counterfactual reasoning has been identified as a particularly effective way to present explanations of the decision boundaries of deep models (Mittelstadt et al., 2019) .

Humans use counterfactuals to reason about the enabling conditions of particular outcomes, as well as to identify situations where the outcome would have occurred even in the absence of some action or condition (Byrne, 2019) .

Saliency maps provide a kind of pixel-level counterfactual, but if the goal is to explain agent behavior according to semantic concepts, interventions at the pixel level may not be sufficient.

Since many semantic concepts may map to the same set of pixels, it may be difficult to identify the functional relationship between changes in pixels and changes in network output according to semantic concepts or game state (Chalupka et al., 2015) .

Researchers may be interpreting differences in network outputs as evidence of differences in semantic concepts.

However, changes in pixels do not guarantee changes in semantic concepts or game state.

In terms of changes to pixels, semantic concepts, and game state, we distinguish among three classes of interventions: distortion, semantics-preserving, and fat-hand (see Table 1 ).

Semantics-preserving and fat-hand interventions are defined with respect to a specific set of semantic concepts.

Fat-hand interventions change game state in such a way that the semantic concepts of interest are also altered.

The pixel-level manipulations used to produce saliency maps primarily result in distortion interventions, though some saliency methods (e.g., object-based) may conceivably produce semanticspreserving or fat-hand interventions as well.

As pixel-level interventions are not guaranteed to produce changes in semantic concepts, counterfactual evaluations applying semantics-preserving interventions may be a more appropriate approach for precisely testing hypotheses of behavior.

To assess how saliency maps are used to make inferences regarding agent behavior in practice, we surveyed recent conference papers in deep RL.

We focused our pool of papers on those that use saliency maps to generate explanations or make claims regarding agent behavior.

Our search criteria consisted of examining papers that cited work that first described any of the following four types of saliency maps:

Jacobian Saliency.

Wang et al. (2015) extended gradient-based saliency maps to deep RL by computing the Jacobian of the output logits with respect to a stack of input images.

Perturbation Saliency.

Greydanus et al. (2017) generate saliency maps by perturbing the original input image using a Gaussian blur of the image and measure changes in policy from removing information from a region.

Object Saliency.

Iyer et al. (2018) use template matching, a common computer vision technique (Brunelli, 2009) , to detect (template) objects within an input image and measure salience through changes in Q-values for masked and unmasked objects.

Attention Saliency.

Most recently, attention-based saliency mapping methods have been proposed to generate interpretable saliency maps (Mott et al., 2019; Nikulin et al., 2019) .

From a set of 90 papers, we found 46 claims drawn from 11 papers that cited and used saliency maps as evidence in their explanations of agent behavior.

The full set of claims are given in Appendix C.

We found three categories of saliency map usage, summarized in Table 2 .

First, all claims interpret salient areas as a proxy for agent focus.

For example, a claim about a Breakout agent notes that the network is focusing on the paddle and little else (Greydanus et al., 2017) .

Second, 87% of the claims in our survey propose hypotheses about the features of the learned policy by reasoning backwards about what representation might jointly produce the observed saliency pattern and agent behavior.

These types of claims either develop an a priori explanation of behavior and verify it using saliency, or they propose an ad hoc explanation after observing saliency to reason about how the agent is using salient areas.

One a priori claim notes that the displayed score is the only differing factor between two states and verify from saliency that the agent is focusing on these pixels (Zahavy et al., 2016 ).

An ad hoc claim about a racing game notes that the agent is recognizing a time-of-day cue from the background color and acting to prepare for a new race (Yang et al., 2018) .

Finally, only 7% (3 out of 46) of the claims attempt to empirically verify the explanations generated from saliency maps.

One of these attempts to corroborate the interpreted saliency behavior by obtaining additional saliency samples from multiple runs of the game.

The other two attempt to manipulate semantics in the pixel input to assess the agent's response by, for example, adding an additional object to verify a hypothesis about memorization (Annasamy & Sycara, 2019).

In the course of the survey, we also observed several more qualitative characteristics of how saliency maps are routinely used.

Subjectivity.

Recent critiques of machine learning have already noted a worrying tendency to conflate speculation and explanation (Lipton & Steinhardt, 2018) .

Saliency methods are not designed for formalizing an abstract human-understandable concept such as "aiming" in Breakout, and they do not provide a means to quantitatively compare semantically meaningful consequences of agent behavior.

This leads to subjectivity in the conclusions drawn from saliency maps.

Unfalsiability.

One hallmark of a scientific hypothesis or claim is that it is falsifiable (Popper, 1959) .

If a claim is false, its falsehood should be identifiable from some conceivable experiment or observation.

One of the most disconcerting practices identified in the survey is the presentation of unfalsifiable interpretations of saliency map patterns.

An example: "A diver is noticed in the saliency map but misunderstood as an enemy and being shot at" (Zahavy et al., 2016) .

It is unclear how we might falsify an abstract concept such as "misunderstanding".

Assessment of Learned Representations.

Evidence indicates that humans rely on causal relationships between objects and their categorization into abstract classes to learn complex processes, such as video games (Tenenbaum & Niyogi, 2003; Dubey et al., 2018) .

Our survey suggests researchers infer that: (1) salient regions map to learned representations of semantic concepts (e.g., ball, paddle), and (2) the relationships between the salient regions map to high-level behaviors (e.g., channel-building, aiming).

The researchers expectations impose a strong bias on both the existence and nature of these mappings.

Our survey indicates that many researchers use saliency maps as an explanatory tool to infer the representations and processes behind an agent's behavior.

However, the extent to which such inferences are valid has not been empirically evaluated under controlled conditions.

In this section, we show how to generate falsifiable hypotheses from saliency maps and propose an intervention-based approach to verify the hypotheses generated from saliency maps.

We intervene on game state to produce counterfactual semantic conditions.

This provides a medium through which we might assess the relationship between saliency and learned semantic representations.

Building Falsifiable Hypotheses from Saliency Maps.

Though saliency maps may not relate directly to semantic concepts, they may still be an effective tool for exploring hypotheses about agent behavior.

Claims or explanations informed by saliency maps have three components: semantic concepts, learned representations, and behavior.

Let X be a subset of the semantic concepts that can be inferred from the input image.

Semantic concepts are often identified visually from the pixel output as the game state is typically latent.

Let B represent behavior, or aggregate actions, over temporally extended sequences of frames, and let R be a representation that is a function of some pixels that the agent learns during training.

As noted in Section 3, researchers often attempt to infer representations from saliency patterns and agent behavior.

To create scientific claims from saliency maps, we recommend using a relatively standard pattern which facilitates objectivity and falsifiability:

Concept set X is salient =⇒ Agent has learned representation R resulting in behavior B.

Consider the Breakout brick reflection example presented in Section 2.

The hypothesis introduced ("the agent has learned to aim at tunnels") can be reconstructed as: bricks are salient =⇒ agent has learned to identify a partially complete tunnel resulting in maneuvering the paddle to hit the ball toward that region.

Forming hypotheses in this format constructs falsifiable claims amenable to empirical analysis.

Counterfactual Evaluation of Claims.

As previously established in Figure 2 , the learned representation and pixel input share a causal relationship with saliency maps generated over a sequence of frames.

Given that the representation learned is static, the relationship between the learned representation and saliency should be invariant under different manipulations of pixel input.

We use this property to assess saliency under counterfactual conditions.

We generate counterfactual conditions by intervening on the RL environment.

Prior work has focused on manipulating the pixel input.

However, this does not modify the underlying latent game state.

Instead, we intervene directly on game state.

In the do-calculus formalism (Pearl, 2000) , this shifts the intervention node in Figure 2a to game state, which leaves the generative process of the pixel image F intact.

We employ TOYBOX, a set of fully parameterized implementation of Atari games (Foley et al., 2018) , to generate interventional data under counterfactual conditions.

The interventions are dependent on the mapping between semantic concepts and learned representations in the hypotheses.

Given a mapping between concept set X and a learned representation R, any intervention would require meaningfully manipulating the state in which X resides to assess the saliency on X under the semantic treatment applied.

Saliency on x ∈ X is defined as the average saliency over a bounding-box around x.

1 Since the learned policies should be semantically invariant under manipulations of the RL environment, by intervening on state, we can verify whether the counterfactual states produce expected patterns of saliency on the associated concept set X. If the counterfactual saliency maps reflect similar saliency patterns, this provides stronger evidence that the observed saliency indicates the agent has learned representation R corresponding to semantic concept set X.

We conduct three case studies to evaluate hypotheses about the relationship between semantic concepts and semantic processes formed from saliency maps.

Each case study uses observed saliency maps to identify hypotheses in the format described in Section 4.

The hypotheses were generated by watching multiple episodes and noting atypical, interesting or popular behaviors from saliency maps.

Using TOYBOX allows us to produce counterfactual states and to generate saliency maps in these altered states.

The case studies are conducted on two Atari games, Breakout and Amidar.

2 The deterministic nature of both games allows some stability in the way we interpret the network's action selection.

Each map is produced from an agent trained with A2C (Mnih et al., 2016) using a CNN-based (Mnih et al., 2015) OpenAI Baselines implementation (Dhariwal et al., 2017) with default hyperparameters (see Appendix B for more details).

Our choice of model is arbitrary.

The emphasis of this work is on methods of explanation, not the explanations themselves.

Case Study 1: Breakout Brick Translation.

Here we evaluate the behavior from Section 2:

Hypothesis 1: bricks are salient =⇒ agent has learned to identify a partially complete tunnel resulting in maneuvering the paddle to hit the ball toward that region

We intervene on the state by translating the brick configurations horizontally.

We expect salience will be nearly invariant to the horizontal translation of the brick configuration.

Figure 3a depicts saliency after intervention.

Salience on the tunnel is less pronounced under left translation, and more pronounced under right translation.

Since the paddle appears on the right, we additionally move the ball and paddle to the far left ( Figure 3b ).

Temporal association (e.g. formation of a tunnel followed by higher saliency) does not generally imply causal dependence.

In this case at least, tunnel formation and salience appear to be confounded by location or, at least, the dependence of these phenomena are highly dependent on location.

Case Study 2: Amidar Score.

Amidar is a Pac-Man-like game in which an agent attempts to completely traverse a series of passages while avoiding enemies.

The yellow sprite that indicates the location of the agent is almost always salient in Amidar.

Surprisingly, the displayed score is salient as often as the yellow sprite throughout the episode with varying levels of intensity.

This can lead to multiple hypotheses about the agent's learned representation: (1) the agent has learned to associate increasing score with higher reward; (2) due to the deterministic nature of Amidar, the agent has created a lookup table that associates its score and its actions.

We can summarize these hypotheses as follows:

Hypothesis 2: score is salient =⇒ agent has learned to {use score as a guide to traverse the board} resulting in {successfully following similar paths in games}.

To evaluate hypothesis 2, we designed four interventions on score:

• intermittent reset: modify the score to 0 every x ∈ [5, 20] timesteps.

• random varying: modify the score to a random number between [1, 200] [5, 20] timesteps.

• fixed: select a score from [0, 200] and fix it for the whole game.

• decremented: modify score to be 3000 initially and decrement score by d ∈ [1, 20] at every timestep.

Figures 4a and 4b show the result of intervening on displayed score on reward and saliency intensity, measured as the average saliency over a 25x15 bounding box, respectively for the first 1000 timesteps of an episode.

The mean is calculated over 50 samples.

If an agent died before 1000 timesteps, the last reward was extended for the remainder of the timesteps and saliency was set to zero.

Using reward as a summary of agent behavior, different interventions on score produce different agent behavior.

Total accumulated reward differs over time for all interventions, typically due to early agent death.

However, salience intensity patterns of all interventions follow the original trajectory very closely.

Different interventions on displayed score cause differing degrees of degraded performance ( Figure 4a ) despite producing similar saliency maps (Figure 4b ), indicating that agent behavior is underdetermined by salience.

Specifically, the salience intensity patterns are similar for the Interventions on displayed score result in differing levels of degraded performance but produce similar saliency maps, suggesting that agent behavior as measured by rewards is underdetermined by salience.

control, fixed, and decremented scores, while the non-ordered score interventions result in degraded performance.

Figure 4c indicates only very weak correlations between the difference-in-reward and difference-in-saliency-under-intervention as compared to the original trajectory.

Correlation coefficients range from 0.041 to 0.274, yielding insignificant p-values for all but one intervention.

See full results in Appendix E.1, Table 6 .

Similar trends are noted for Jacobian and perturbation saliency methods in Appendix E.1.

The existence of a high correlation between two processes (e.g., incrementing score and persistence of saliency) does not imply causation.

Interventions can be useful in identifying the common cause leading to the high correlation.

Case Study 3: Amidar Enemy Distance.

Enemies are salient in Amidar at varying times.

From visual inspection, we observe that enemies close to the player tend to have higher saliency.

Accordingly, we generate the following hypothesis:

Hypothesis 3: enemy is salient =⇒ agent has learned to {look for enemies close to it} resulting in {successful avoidance of enemy collision}.

Without directly intervening on the game state, we can first identify whether the player-enemy distance and enemy saliency is correlated using observational data.

We collect 1000 frames of an episode of Amidar and record the Manhattan distance between the midpoints of the player and enemies, represented by 7x7 bounding boxes, along with the object salience of each enemy.

Figure  5a shows the distance of each enemy to the player over time with saliency intensity represented by the shaded region.

Figure 5b shows the correlation between the distance to each enemy and the corresponding saliency.

Correlation coefficients and significance values are reported in Table 3 .

It is clear that there is no correlation between saliency and distance of each enemy to the player.

Given that statistical dependence is almost always a necessary pre-condition for causation, we expect that there will not be any causal dependence.

To further examine this, we intervene on enemy positions of salient enemies at each timestep by moving the enemy closer and farther away from the player.

Figure 5c contains these results.

Given Hypothesis 3, we would expect to see an increasing trend in saliency for enemies closer to the player.

However, the size of the effect is close to 0 (see Table 3 ).

In addition, we find no correlation in the enemy distance experiments for the Jacobian or perturbation saliency methods (included in Appendix E.2).

Conclusion.

Spurious correlations, or misinterpretations of existing correlation, can occur between two processes (e.g. correlation between player-enemy distance and saliency), and human observers are susceptible to identifying spurious correlations (Simon, 1954) .

Spurious correlations can sometimes be identified from observational analysis without requiring interventional analysis.

Thinking counterfactually about the explanations generated from saliency maps facilitates empirical evaluation of those explanations.

The experiments above show some of the difficulties in drawing conclusions from saliency maps.

These include the tendency of human observers to incorrectly infer association between observed processes, the potential for experimental evidence to contradict seemingly obvious observational conclusions, and the challenges of potential confounding in temporal processes.

One of the main conclusions from this evaluation is that saliency maps are an exploratory tool rather than an explanatory tool.

Saliency maps alone cannot be reliably used to infer explanations and instead require other supporting tools.

This can include combining evidence from saliency maps with other explanation methods or employing a more experimental approach to evaluation of saliency maps such as the approach demonstrated in the case studies above.

The framework for generating falsifiable hypotheses suggested in Section 4 can assist with designing more specific and falsifiable explanations.

The distinction between the components of an explanation, particularly the semantic concept set X, learned representation R and observed behavior B, can further assist in experimental evaluation.

Generalization of Proposed Methodology.

The methodology presented in this work can be easily extended to other vision-based domains in deep RL.

Particularly, the framework of the graphical model introduced in Figure 2a applies to all domains where the input to the network is image data.

An extended version of the model for Breakout can be found in Appendix 7.

We propose intervention-based experimentation as a primary tool to evaluate the hypotheses generated from saliency maps.

Yet, alternative methods can identify a false hypothesis even earlier.

For instance, evaluating statistical dependence alone can help identify some situations in which causation is absent (e.g., Case Study 3).

We also employ TOYBOX in this work.

However, limited forms of evaluation may be possible in non-intervenable environments, though they may be more tedious to implement.

For instance, each of the interventions conducted in Case Study 1 can be produced in an observation-only environment by manipulating the pixel input (Chalupka et al., 2015; Brunelli, 2009 ).

Developing more experimental systems for evaluating explanations is an open area of research.

This work analyzes explanations generated from feed-forward deep RL agents.

Yet, given that the proposed methodology is not model dependent, aspects of the approach will carry over to recurrent deep RL agents.

The proposed methodology would not work for repeated interventions on recurrent deep RL agents due to their capacity for memorization.

Prior work has proposed alternatives to the use of saliency maps to support explanation of deep RL agents.

Toybox was developed to support behavioral experiments for deep RL systems (Tosch et al., 2019) .

Evaluation and Critiques of Saliency Maps.

Prior work in the deep network literature has evaluated and critiqued saliency maps.

Kindermans et al. (2019) and Adebayo et al. (2018) demonstrate the utility of saliency maps by adding random variance in input.

Seo et al. (2018) provide a theoretical justification of saliency and hypothesize that there exists a correlation between gradients-based saliency methods and model interpretation.

Samek et al. (2017) and Hooker et al. (2018) present evaluation of existing saliency methods for image classification.

Others have critiqued network attention as a means of explanation (Jain & Wallace, 2019; Brunner et al., 2019) ).

We conduct a survey of uses of saliency maps, propose a methodology to evaluate saliency maps, and examine the extent to which the agent's learned representations can be inferred from saliency maps.

We investigate how well the pixel-level inferences of saliency maps correspond to the semantic concept-level inferences of human-level interventions.

Our results show saliency maps cannot be trusted to reflect causal relationships between semantic concepts and agent behavior.

We recommend saliency maps to be used as an exploratory tool, not explanatory tool.

A SALIENCY METHODS Figure 6 shows example saliency maps for Amidar of the three saliency methods evaluated in this work, namely perturbation, object and Jacobian.

Figure 6: Examples of (a) perturbation saliency method (Greydanus et al., 2017) ; (b) object saliency method (Iyer et al., 2018) ; and (c) Jacobian saliency method (Wang et al., 2015) , on Amidar.

We use the OpenAI Baselines' implementation (Dhariwal et al., 2017 ) of an A2C model (Mnih et al., 2016) to train the RL agents on Breakout and Amidar.

The model uses the CNN architecture proposed by Mnih et al. (2015) .

Each agent is trained for 40 million iterations using RMSProp with default hyperparameters ( We conducted a survey of recent literature to assess how saliency maps are used to interpret agent behavior in deep RL.

We began our search by focusing on work citing the following four types of saliency maps: Jacobian (Wang et al., 2015) , perturbation (Greydanus et al., 2017) , object (Iyer et al., 2018) and attention (Mott et al., 2019) .

Papers were selected if they employed saliency maps to create explanations regarding agent behavior.

This resulted in selecting 46 claims from 11 papers.

These 11 papers have appeared at ICML (3), NeurIPS (1), AAAI (2), ArXiv (3), OpenReview (1) and a thesis (1).

There are several model-specific saliency mapping methods that we excluded from our survey.

Perturbation Y N "Notice that both actor and critic tend to monitor the area above the ship.

This may be useful for determining whether the ship is protected from enemy fire or has a clear shot at enemies."

Perturbation Y N "We found that the agent enters and exits a 'tunneling mode' over the course of a single frame.

Once the tunneling location becomes salient, it remains so until the tunnel is finished." "Once the agent finished the first screen it is presented with another one, distinguished only by the score that was accumulated in the first screen.

Therefore, an agent might encounter problems with generalizing to the new screen if it over-fits the score pixels.

Figure 15 shows the saliency maps of different games supporting our claim that DQN is basing its estimates using these pixels.

We suggest to further investigate this, for example, we suggest to train an agent that does not receive those pixels as input."

There is a very notable difference between the policy saliency between the two models, where the former one only pays limited attention to the road and almost no attention to the engine indicator, the opposite from fA3C-LSTM.

Explicitly,it means masking any regions from the input does not cause much perturbation to the policy when trained with continuous space as targets, likely because the real consequence from a small change in action, e.g. no braking (a3= 0) versus braking (a3= 0.3), can be very substantial but numerically too subtle for the network to capture during optimization on the continuous spectrum.

"

Rupprecht et al. (2018) "Analyzing the visualizations on Seaquest, we make an interesting observation.

When maximizing the Q-value for the actions, in many samples we see a low or very low oxygen meter.

In these cases the submarine would need to ascend to the surface to avoid suffocation.

Although the up action is the only sensible choice in this case, we also obtain visualized low oxygen states for all other actions.

This implies that the agent has not understood the importance of resurfacing when the oxygen is low.

We then run several roll outs of the agent and see that the major cause of death is indeed suffocation and not collision with enemies.

"

The most dominant pattern we observe is that the model learns to attend to task-relevant things in the scene.

In most ATARI games that usually means that the player is one of the foci of attention, as well as enemies, power-ups and the score itself (which is an important factor in the calculating the value function).

"

N N " Figure 4 shows a examples of this in Ms Pacman and Alien in the both games the model scans through possible paths, making sure there are no enemies or ghosts ahead.

We observe that when it does see a ghost, another path is produced or executed in order to avoid it."

Pacman Attention Y N "In many games we observe that the agent learns to place trip-wires at strategic points in space such that if a game object crosses them a specific action is taken.

For example, in Space Invaders two such trip wires are following the player ship on both sides such that if a bullet crosses one of them the agent immediately evades them by moving towards the opposite direction."

Space Invaders Attention Y N "Another example is Breakout where we can see it working in two stages.

First the attention is spread out around the general area of the ball, then focuses into a localized line.

Once the ball crosses that line the agent moves towards the ball." Breakout Attention Y N "As can be seen, the system uses the two modes to make its decisions, some of the heads are content specific looking for opponent cars.

Some are mixed, scanning the horizon for incoming cars and when found, tracking them, and some are location based queries, scanning the area right in front of the player for anything the crosses its path (a trip-wire which moves with the player)."

Comparing the attention agent to the baseline agent, we see that the attention agent is sensitive to more focused areas along the possible future trajectory.

The baseline agent is more focused on the area immediately in front of the player (for the policy saliency) and on the score, while the attention agent focuses more specifically on the path the agent will follow (for the policy) and on possible future longer term paths (for the value).

" Note how it focuses attention on the upper part of the screen after destroying multiple bricks from the top.

This attention does not go away after the ball moves elsewhere (not shown in the images).We speculate that this is how the agent models tunneling:rather than having a high-level concept of digging a tunnel,it simply strikes wherever it has managed to strike already."

The causal graphical model in Figure 2a can be extended to different domains in RL where the input to the model is an image.

This requires extending game state to include the underlying state variables.

Figure 7 shows the extension for Breakout using an A2C model.

Note, game state includes the underlying state variables for Breakout and logits were split into action logits and value as outputted by A2C.

Table 6 : Numeric results representing Pearson's correlation coefficient and p-value for the differences in reward and saliency (object, perturbation and Jacobian) from the original trajectory for each intervention.

Results show small correlation coefficients (r) suggesting that there is a weak relationship between the differences in reward and saliency for the interventions.

We also evaluated Pearson's correlation between the differences in reward and saliency with the original trajectory for all three methods (see Table 6 ).

The results support the correlation plots in Figures 4c, 8c and 9c.

We further evaluated the relationship between player-enemy distance and saliency (Section 5) in Amidar on perturbation and Jacobian saliency maps.

These results are presented in Figures 10  and 11 , respectively.

Jacobian saliency performed the worst for the intervention-based experiment, suggesting that there is no impact of player-enemy distance on saliency.

Regression analysis between distance and perturbation and Jacobian saliency can be found in Tables  7 and 8 respectively.

The results support the lack of correlation between the observational and interventional distributions.

Note, enemy 1 is more salient throughout the game compared to the other four enemies resulting in a larger interventional sample size.

Figures 11b and c .

The results indicate a very small strength of effect (slope) for both observational and interventional data and a small correlation coefficient (r), suggesting that there is, at best, only a very weak causal dependence of saliency on distance-to-player.

@highlight

Proposing a new counterfactual-based methodology to evaluate the hypotheses generated from saliency maps about deep RL agent behavior. 
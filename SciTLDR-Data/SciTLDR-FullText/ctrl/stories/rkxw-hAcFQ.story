We study the problem of training sequential generative models for capturing coordinated multi-agent trajectory behavior, such as  offensive basketball gameplay.

When modeling such settings, it is often beneficial to design hierarchical models that can capture long-term coordination using intermediate variables.

Furthermore, these intermediate variables should capture interesting high-level behavioral semantics in an interpretable and manipulable way.

We present a hierarchical framework that can effectively learn such sequential generative models.

Our approach is inspired by recent work on leveraging programmatically produced weak labels, which we extend to the spatiotemporal regime.

In addition to synthetic settings, we show how to instantiate our framework to effectively model complex interactions between basketball players and generate realistic multi-agent trajectories of basketball gameplay over long time periods.

We validate our approach using both quantitative and qualitative evaluations, including a user study comparison conducted with professional sports analysts.

The ongoing explosion of recorded tracking data is enabling the study of fine-grained behavior in many domains: sports BID25 BID44 BID46 BID20 , video games BID34 , video & motion capture BID36 BID38 BID43 , navigation & driving BID48 BID45 BID22 , laboratory animal behaviors BID16 BID7 , and tele-operated robotics BID0 BID23 .

However, it is an open challenge to develop sequential generative models leveraging such data, for instance, to capture the complex behavior of multiple cooperating agents.

FIG1 shows an example of offensive players in basketball moving unpredictably and with multimodal distributions over possible trajectories.

FIG1 depicts a simplified Boids model from BID31 for modeling animal schooling behavior in which the agents can be friendly or unfriendly.

In both cases, agent behavior is highly coordinated and non-deterministic, and the space of all multi-agent trajectories is naively exponentially large.

When modeling such sequential data, it is often beneficial to design hierarchical models that can capture long-term coordination using intermediate variables or representations BID21 BID46 ).

An attractive use-case for these intermediate variables is to capture interesting highlevel behavioral semantics in an interpretable and manipulable way.

For instance, in the basketball setting, intermediate variables can encode long-term strategies and team formations.

Conventional approaches to learning interpretable intermediate variables typically focus on learning disentangled latent representations in an unsupervised way (e.g., BID22 BID42 ), but it is challenging for such approaches to handle complex sequential settings BID4 .

To address this challenge, we present a hierarchical framework that can effectively learn such sequential generative models, while using programmatic weak supervision.

Our approach uses a labeling function to programmatically produce useful weak labels for supervised learning of interpretable intermediate representations.

This approach is inspired by recent work on data programming BID29 , which uses cheap and noisy labeling functions to significantly speed up learning.

In this work, we extend this approach to the spatiotemporal regime.

Our contributions can be summarized as follows:??? We propose a hierarchical framework for sequential generative modeling.

Our approach is compatible with many existing deep generative models.???

We show how to programmatically produce weak labels of macro-intents to train the intermediate representation in a supervised fashion.

Our approach is easy to implement and results in highly interpretable intermediate variables, which allows for conditional inference by grounding macro-intents to manipulate behaviors.??? Focusing on multi-agent tracking data, we show that our approach can generate highquality trajectories and effectively encode long-term coordination between multiple agents.

In addition to synthetic settings, we showcase our approach in an application on modeling team offense in basketball.

We validate our approach both quantitatively and qualitatively, including a user study comparison with professional sports analysts, and show significant improvements over standard baselines.

Deep generative models.

The study of deep generative models is an increasingly popular research area, due to their ability to inherit both the flexibility of deep learning and the probabilistic semantics of generative models.

In general, there are two ways that one can incorporate stochastics into deep models.

The first approach models an explicit distribution over actions in the output layer, e.g., via logistic regression BID3 BID26 BID41 BID46 BID7 .

The second approach uses deep neural nets to define a transformation from a simple distribution to one of interest BID10 BID18 BID33 and can more readily be extended to incorporate additional structure, such as a hierarchy of random variables BID28 or dynamics BID16 BID6 BID19 BID9 .

Our framework can incorporate both variants.

Structured probabilistic models.

Recently, there has been increasing interest in probabilistic modeling with additional structure or side information.

Existing work includes approaches that enforce logic constraints BID1 , specify generative models as programs , or automatically produce weak supervision via data programming BID29 .

Our framework is inspired by the latter, which we extend to the spatiotemporal regime.

Imitation Learning.

Our work is also related to imitation learning, which aims to learn a policy that can mimic demonstrated behavior BID37 BID0 BID47 BID12 .

There has been some prior work in multi-agent imitation learning BID20 BID35 and learning stochastic policies BID12 BID22 , but no previous work has focused on learning generative polices while simultaneously addressing generative and multi-agent imitation learning.

For instance, experiments in BID12 all lead to highly peaked distributions, while BID22 captures multimodal distributions by learning unimodal policies for a fixed number of experts.

BID14 raise the issue of learning stochastic multi-agent behavior, but their solution involves significant feature engineering.3 BACKGROUND: SEQUENTIAL GENERATIVE MODELING Let x t ??? R d denote the state at time t and x ???T = {x 1 , . . .

, x T } denote a sequence of states of length T .

Suppose we have a collection of N demonstrations D = {x ???T }.

In our experiments, all sequences have the same length T , but in general this does not need to be the case.

The goal of sequential generative modeling is to learn the distribution over sequential data D. A common approach is to factorize the joint distribution and then maximize the log-likelihood: DISPLAYFORM0 where ?? are the learn-able parameters of the model, such as a recurrent neural network (RNN).Stochastic latent variable models.

However, RNNs with simple output distributions that optimize Eq.(1) often struggle to capture highly variable and structured sequential data.

For example, an RNN with Gaussian output distribution has difficulty learning the multimodal behavior of the green player moving to the top-left/bottom-left in FIG1 .

Recent work in sequential generative models address this issue by injecting stochastic latent variables into the model and optimizing using amortized variational inference to learn the latent variables BID9 BID11 .In particular, we use a variational RNN (VRNN BID6 ) as our base model (Figure 3a ), but we emphasize that our approach is compatible with other sequential generative models as well.

A VRNN is essentially a variational autoencoder (VAE) conditioned on the hidden state of an RNN and is trained by maximizing the (sequential) evidence lower-bound (ELBO): DISPLAYFORM1 Eq. (2) is a lower-bound of the log-likelihood in Eq.(1) and can be interpreted as the VAE ELBO summed over each timestep t. We refer to appendix A for more details of VAEs and VRNNs.

In our problem setting, we assume that each sequence x ???T consists of the trajectories of K coordinating agents.

That is, we can decompose each x ???T into K trajectories: DISPLAYFORM0 For example, the sequence in FIG1 can be decomposed into the trajectories of K = 5 basketball players.

Assuming conditional independence between the agent states x k t given state history x <t , we can factorize the maximum log-likelihood objective in Eq. (1) even further: DISPLAYFORM1 Naturally, there are two baseline approaches in this setting:1.

Treat the data as a single-agent trajectory and train a single model: DISPLAYFORM2 Train independent models for each agent: ?? = {?? 1 , . . .

, ?? K }.As we empirically verify in Section 5, VRNN models using these two approaches have difficulty learning representations of the data that generalize well over long time horizons, and capturing the coordination inherent in multi-agent trajectories.

Our solution introduces a hierarchical structure of macro-intents obtained via labeling functions to effectively learn low-dimensional (distributional) representations of the data that extend in both time and space for multiple coordinating agents.

Defining macro-intents.

We assume there exists shared latent variables called macro-intents that: 1) provide a tractable way to capture coordination between agents; 2) encode long-term intents of agents and enable long-term planning at a higher-level timescale; and 3) compactly represent some low-dimensional structure in an exponentially large multi-agent state space.

Figure 2: Macro-intents (boxes) for two players.

For example, Figure 2 illustrates macro-intents for two basketball players as specific areas on the court (boxes).

Upon reaching its macro-intent in the top-right, the blue player moves towards its next macro-intent in the bottom-left.

Similarly, the green player moves towards its macro-intents from bottom-right to middle-left.

These macro-intents are visible to both players and capture the coordination as they describe how the players plan to position themselves on the court.

Macro-intents provide a compact summary of the players' trajectories over a long time.

Macro-intents do not need to have a geometric interpretation.

For example, macro-intents in the Boids model in FIG1 can be a binary label indicating friendly vs. unfriendly behavior.

The goal is for macro-intents to encode long-term intent and ensure that agents behave more cohesively.

Our modeling assumptions for macro-intents are: DISPLAYFORM3 are conditioned on some shared macro-intent g t , ??? the start and end times [t 1 , t 2 ] of episodes can vary between trajectories, ??? macro-intents change slowly over time relative to the agent states: dg t /dt 1, ??? and due to their reduced dimensionality, we can model (near-)arbitrary dependencies between macro-intents (e.g., coordination) via black box learning.

Labeling functions for macro-intents.

Obtaining macro-intent labels from experts for training is ideal, but often too expensive.

Instead, our work is inspired by recent advances in weak supervision settings known as data programming, in which multiple weak and noisy label sources called labeling functions can be leveraged to learn the underlying structure of large unlabeled datasets BID30 BID2 .

These labeling functions often compute heuristics that allow users to incorporate domain knowledge into the model.

For instance, the labeling function we use to obtain macro-intents for basketball trajectories computes the regions on the court in which players remain stationary; this integrates the idea that players aim to set up specific formations on the court.

In general, labeling functions are simple scripts/programs that can parse and label data very quickly, hence the name programmatic weak supervision.t is the hidden state of an RNN that summarizes the trajectory up to time t, and g t is the shared macro-intent at time t. Figure 3b shows our hierarchical model, which samples macro-intents during generation rather than using only ground-truth macro-intents.

Here, we train an RNN-model to sample macro-intents: DISPLAYFORM4 where ?? g maps to a distribution over macro-intents and h g,t???1 summarizes the history of macrointents up to time t.

We condition the macro-intent model on previous states x t???1 in Eq. (5) and generate next states by first sampling a macro-intent g t , and then sampling x k t conditioned on g t (see Figure 3b ).

Note that all agent-models for generating x k t share the same macro-intent variable g t .

This is core to our approach as it induces coordination between agent trajectories (see Section 5).We learn our agent-models by maximizing the VRNN objective from Eq (2) conditioned on the shared g t variables while independently learning the macro-intent model via supervised learning by maximizing the log-likelihood of macro-intent labels obtained programmatically.

Circles are stochastic and diamonds are deterministic.

macro-intent g t is shared across agents.

In principle, any generative model can be used in our framework.

We first apply our approach on generating offensive team basketball gameplay (team with possession of the ball), and then on a synthetic Boids model dataset.

We present both quantitative and qualitative experimental results.

Our quantitative results include a user study comparison with professional sports analysts, who significantly preferred basketball rollouts generated from our approach to standard baselines.

Examples from the user study and videos of generated rollouts can be seen in our demo video.2 Our qualitative results demonstrate the ability of our approach to generate high-quality rollouts under various conditions.

Training data.

Each demonstration in our data contains trajectories of K = 5 players on the left half-court, recorded for T = 50 timesteps at 6 Hz.

The offensive team has possession of the ball for the entire sequence.

x k t are the coordinates of player k at time t on the court (50 ?? 94 feet).

We normalize and mean-shift the data.

Players are ordered based on their relative positions, similar to the role assignment in BID24 .

There are 107,146 training and 13,845 test examples.

We ignore the defensive players and the ball to focus on capturing the coordination and multimodality of the offensive team.

In principle, we can provide the defensive positions as conditional input for our model and update the defensive positions using methods such as BID20 .

We leave the task of modeling the ball and defense for future work.

Figure 5: Rollouts from baselines and our model starting from black dots, generated for 40 timesteps after an initial burn-in period of 10 timesteps (marked by dark shading).

An interactive demo of our hierarchical model is available at: http://basketball-ai.com/.1.

RNN-gauss: RNN without latent variables using 900 2-layer GRU cells as hidden state.2.

VRNN-single: VRNN in which we concatenate all player positions together (K = 1) with 900 2-layer GRU cells for the hidden state and a 80-dimensional latent variable.3.

VRNN-indep: VRNN for each agent with 250 2-layer GRUs and 16-dim latent variables.4.

VRNN-mixed: Combination of VRNN-single and VRNN-indep.

Shared hidden state of 600 2-layer GRUs is fed into decoders with 16-dim latent variables for each agent.5.

VRAE-mi: VRAE-style architecture BID8 ) that maximizes the mutual information between x ???T and macro-intent.

We refer to appendix C for details.

Log-likelihood.

TAB1 reports the average log-likelihoods on the test data.

Our approach outperforms RNN-gauss and is comparable with other baselines.

However, higher log-likelihoods do not necessarily indicate higher quality of generated samples BID39 .

As such, we also assess using other means, such as human preference studies and auxiliary statistics.

Human preference study.

We recruited 14 professional sports analysts as judges to compare the quality of rollouts.

Each comparison animates two rollouts, one from our model and another from a baseline.

Both rollouts are burned-in for 10 timesteps with the same ground-truth states from the test set, and then generated for the next 40 timesteps.

Judges decide which of the two rollouts looks more realistic.

TAB2 shows the results from the preference study.

We tested our model against two baselines, VRNN-single and VRNN-indep, with 25 comparisons for each.

All judges preferred our model over the baselines with 98% statistical significance.

These results suggest that our model generates rollouts of significantly higher quality than the baselines.

Domain statistics.

Finally, we compute several basketball statistics (average speed, average total distance traveled, % of frames with players out-of-bounds) and summarize them in TAB3 : Domain statistics of 1000 basketball trajectories generated from each model: average speed, average distance traveled, and % of frames with players out-of-bounds (OOB).

Trajectories from our models using programmatic weak supervision match the closest with the ground-truth.

See appendix D for labeling function pseudocode.(a) 10 rollouts of the green player ( ) with a burn-in period of 20 timesteps.

Left: The model generates macro-intents.

Right: We ground the macro-intents at the bottom-left.

In both, we observe a multimodal distribution of trajectories.(b) The distribution of macro-intents sampled from 20 rollouts of the green player changes in response to the change in red trajectories and macro-intents.

This suggests that macro-intents encode and induce coordination between multiple players.

model generates trajectories that are most similar to ground-truth trajectories with respect to these statistics, indicating that our model generates significantly more realistic behavior than all baselines.

Choice of labeling function.

In addition to LF-stationary, we also assess the quality of our approach using macro-intents obtained from different labeling functions.

LF-window25 and LF-window50 labels macro-intents as the last region a player resides in every window of 25 and 50 timesteps respectively (pseudocode in appendix D).

TAB3 shows that domain statistics from our models using programmatic weak supervision match closer to the ground-truth with more informative labeling functions (LF-stationary > LF-window25 > LF-window50).

This is expected, since LF-stationary provides the most information about the structure of the data.

We next conduct a qualitative visual inspection of rollouts.

Figure 5 shows rollouts generated from VRNN-single, VRNN-indep, and our model by sampling states for 40 timesteps after an initial burnin period of 10 timesteps with ground-truth states from the test set.

An interactive demo to generate more rollouts from our hierarchical model can be found at: http://basketball-ai.com/.Common problems in baseline rollouts include players moving out of bounds or in the wrong direction (Figure 5a ).

These issues tend to occur at later timesteps, suggesting that the baselines do not perform well over long horizons.

One possible explanation is due to compounding errors BID34 : if the model makes a mistake and deviates from the states seen during training, it is likely to make more mistakes in the future and generalize poorly.

On the other hand, generated rollouts from our model are more robust to the types of errors made by the baselines (Figure 5b ).Macro-intents induce multimodal and interpretable rollouts.

Generated macro-intents allow us to intepret the intent of each individual player as well as a global team strategy (e.g. setting up a specific formation on the court).

We highlight that our model learns a multimodal generating distribution, as repeated rollouts with the same burn-in result in a dynamic range of generated trajectories, as seen in FIG3 Left.

Furthermore, FIG3 Right demonstrates that grounding macro-intents during generation instead of sampling them allows us to control agent behavior.

counts) of average distance to an agent's closest neighbor in 5000 rollouts.

Our hierarchical model more closely captures the two distinct modes for friendly (small distances, left peak) vs. unfriendly (large distances, right peak) behavior compared to baselines, which do not learn to distinguish them.

Macro-intents induce coordination.

FIG3 illustrates how the macro-intents encode coordination between players that results in realistic rollouts of players moving cohesively.

As we change the trajectory and macro-intent of the red player, the distribution of macro-intents generated from our model for the green player changes such that the two players occupy different areas of the court.

To illustrate the generality of our approach, we apply our model to a simplified version of the Boids model BID31 that produces realistic trajectories of schooling behavior.

We generate trajectories for 8 agents for 50 frames.

The agents start in fixed positions around the origin with initial velocities sampled from a unit Gaussian.

Each agent's velocity is then updated at each timestep: DISPLAYFORM0 Full details of the model can be found in Appendix B. We randomly sample the sign of c 1 for each trajectory, which produces two distinct types of behaviors: friendly agents (c 1 > 0) that like to group together, and unfriendly agents (c 1 < 0) that like to stay apart (see FIG1 .

We also introduce more stochasticity into the model by periodically updating ?? randomly.

Our labeling function thresholds the average distance to an agent's closest neighbor (see last plot in FIG4 ).

This is equivalent to using the sign of c 1 as our macro-intents, which indicates the type of behavior.

Note that unlike our macro-intents for the basketball dataset, these macro-intents are simpler and have no geometric interpretation.

All models have similar average log-likelihoods on the test set in TAB1 , but our hierarchical model can capture the true generating distribution much better than the baselines.

For example, FIG4 depicts the histograms of average distances to an agent's closest neighbor in trajectories generated from all models and the ground-truth.

Our model more closely captures the two distinct modes in the ground-truth (friendly, small distances, left peak vs. unfriendly, large distances, right peak) whereas the baselines fail to distinguish them.

Output distribution for states.

The outputs of all models (including baselines) sample from a multivariate Gaussian with diagonal covariance.

We also experimented with sampling from a mixture of 2, 3, 4, and 8 Gaussian components, but discovered that the models would always learn to assign all the weight on a single component and ignore the others.

The variance of the active component is also very small.

This is intuitive because sampling with a large variance at every timestep would result in noisy trajectories and not the smooth ones that we see in FIG3 .Choice of macro-intent model.

In principle, we can use more expressive generative models, like a VRNN, to model macro-intents over richer macro-intent spaces in Eq. (5).

In our case, we found that an RNN was sufficient in capturing the distribution of macro-intents shown in Figure 4 .

The RNN learns multinomial distributions over macro-intents that are peaked at a single macro-intent and relatively static through time, which is consistent with the macro-intent labels that we extracted from data.

Latent variables in a VRNN had minimal effect on the multinomial distribution.

Maximizing mutual information isn't effective.

The learned macro-intents in our fully unsupervised VRAE-mi model do not encode anything useful and are essentially ignored by the model.

In particular, the model learns to match the approximate posterior of macro-intents from the encoder with the discriminator from the mutual information lower-bound.

This results in a lack of diversity in rollouts as we vary the macro-intents during generation.

We refer to appendix C for examples.

The macro-intents labeling functions used in our experiments are relatively simple.

For instance, rather than simply using location-based macro-intents, we can also incorporate complex interactions such as "pick and roll".

Another future direction is to explore how to adapt our method to different domains, e.g., defining a macro-intent representing "argument" for a dialogue between two agents, or a macro-intent representing "refrain" for music generation for "coordinating instruments" BID40 .

We have shown that weak macro-intent labels extracted using simple domain-specific heuristics can be effectively used to generate high-quality coordinated multi-agent trajectories.

An interesting direction is to incorporate multiple labeling functions, each viewed as noisy realizations of true macro-intents, similar to BID29 BID2 .

Recurrent neural networks.

A RNN models the conditional probabilities in Eq.(1) with a hidden state h t that summarizes the information in the first t ??? 1 timesteps: DISPLAYFORM0 where ?? maps the hidden state to a probability distribution over states and f is a deterministic function such as LSTMs BID13 or GRUs BID5 .

RNNs with simple output distributions often struggle to capture highly variable and structured sequential data.

Recent work in sequential generative models address this issue by injecting stochastic latent variables into the model and using amortized variational inference to infer latent variables from data.

Variational Autoencoders.

A variational autoencoder (VAE) BID18 ) is a generative model for non-sequential data that injects latent variables z into the joint distribution p ?? (x, z) and introduces an inference network parametrized by ?? to approximate the posterior q ?? (z | x).

The learning objective is to maximize the evidence lower-bound (ELBO) of the log-likelihood with respect to the model parameters ?? and ??: DISPLAYFORM1 The first term is known as the reconstruction term and can be approximated with Monte Carlo sampling.

The second term is the Kullback-Leibler divergence between the approximate posterior and the prior, and can be evaluated analytically (i.e. if both distributions are Gaussian with diagonal covariance).

The inference model q ?? (z | x), generative model p ?? (x | z), and prior p ?? (z) are often implemented with neural networks.

Variational RNNs.

VRNNs combine VAEs and RNNs by conditioning the VAE on a hidden state h t (see Figure 3a) : DISPLAYFORM2 VRNNs are also trained by maximizing the ELBO, which in this case can be interpreted as the sum of VAE ELBOs over each timestep of the sequence: DISPLAYFORM3 Note that the prior distribution of latent variable z t depends on the history of states and latent variables (Eq. (9) ).

This temporal dependency of the prior allows VRNNs to model complex sequential data like speech and handwriting BID6 .

We generate 32,768 training and 8,192 test trajectories.

Each agent's velocity is updated as: DISPLAYFORM0 ??? v coh is the normalized cohesion vector towards an agent's local neighborhood (radius 0.9)??? v sep is the normalized vector away from an agent's close neighborhood (radius 0.2)??? v ali is the average velocity of other agents in a local neighborhood??? v ori is the normalized vector towards the origin DISPLAYFORM1 ??? ?? is sampled uniformly at random every 10 frames in range [0.8, 1.4]

We ran experiments to see if we can learn meaningful macro-intents in a fully unsupervised fashion by maximizing the mutual information between macro-intent variables and trajectories x ???T .

We use a VRAE-style model from BID8 in which we encode an entire trajectory into a latent macro-intent variable z, with the idea that z should encode global properties of the sequence.

The corresponding ELBO is: DISPLAYFORM0 where p ?? (z) is the prior, q ?? (z | x ???T ) is the encoder, and p ?? k (x k t | x <t , z) are decoders per agent.

It is intractable to compute the mutual information between z and x ???T exactly, so we introduce a discriminator q ?? (z | x ???T ) and use the following variational lower-bound of mutual information: DISPLAYFORM1 We jointly maximize L 1 + ??L 2 wrt.

model parameters (??, ??, ??), with ?? = 1 in our experiments.

Categorical vs. real-valued macro-intent z. When we train an 8-dimensional categorical macrointent variable with a uniform prior (using gumbel-softmax trick BID15 ), the average distribution from the encoder matches the discriminator but not the prior FIG5 ).

When we train a 2-dimensional real-valued macro-intent variable with a standard Gaussian prior, the learned model generates trajectories with limited variability as we vary the macro-intent variable FIG6 ).

We define macro-intents in basketball by segmenting the left half-court into a 10 ?? 9 grid of 5ft ??5ft boxes (Figure 2 ).

Algorithm 1 describes LF-window25, which computes macro-intents based on last positions in 25-timestep windows (LF-window50 is similar).

Algorithm 2 describes LF-stationary, which computes macro-intents based on stationary positions.

For both, Label-macro-intent(x for t = T ??? 1 . . .

1 do return g

<|TLDR|>

@highlight

We blend deep generative models with programmatic weak supervision to generate coordinated multi-agent trajectories of significantly higher quality than previous baselines.

@highlight

Proposes multi-agent sequential generative models.

@highlight

The paper proposes training generative models that produce multi-agent trajectories using heuristic functions that label variables that would otherwise be latent in training data
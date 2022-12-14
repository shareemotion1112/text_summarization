In visual planning (VP), an agent learns to plan goal-directed behavior from observations of a dynamical system obtained offline, e.g., images obtained from self-supervised robot interaction.

VP algorithms essentially combine data-driven perception and planning, and are important for robotic manipulation and navigation domains, among others.

A recent and promising approach to VP is the semi-parametric topological memory (SPTM) method, where image samples are treated as nodes in a graph, and the connectivity in the graph is learned using deep image classification.

Thus, the learned graph represents the topological connectivity of the data, and planning can be performed using conventional graph search methods.

However, training SPTM necessitates a suitable loss function for the connectivity classifier, which requires non-trivial manual tuning.

More importantly, SPTM is constricted in its ability to generalize to changes in the domain, as its graph is constructed from direct observations and thus requires collecting new samples for planning.

In this paper, we propose Hallucinative Topological Memory (HTM), which overcomes these shortcomings.

In HTM, instead of training a discriminative classifier we train an energy function using contrastive predictive coding.

In addition, we learn a conditional VAE model that generates samples given a context image of the domain, and use these hallucinated samples for building the connectivity graph, allowing for zero-shot generalization to domain changes.

In simulated domains, HTM outperforms conventional SPTM and visual foresight methods in terms of both plan quality and success in long-horizon planning.

For robots to operate in unstructured environments such as homes and hospitals, they need to manipulate objects and solve complex tasks as they perceive the physical world.

While task planning and object manipulation have been studied in the classical AI paradigm [20, 9, 30, 10] , most successes have relied on a human-designed state representation and perception, which can be challenging to obtain in unstructured domains.

While high-dimensional sensory input such as images can be easy to acquire, planning using raw percepts is challenging.

This has motivated the investigation of datadriven approaches for robotic manipulation.

For example, deep reinforcement learning (RL) has made impressive progress in handling high-dimensional sensory inputs and solving complex tasks in recent years [7, 4, 15, 23] .

One of the main challenges in deploying deep RL methods in human-centric environment is interpretability.

For example, before executing a potentially dangerous task, it would be desirable to visualize what the robot is planning to do step by step, and intervene if necessary.

Addressing both data-driven modeling and interpretability, the visual planning (VP) paradigm seeks to learn a model of the environment from raw perception and then produce a visual plan of solving a task before actually executing a robot action.

Recently, several studies in manipulation and navigation [13, 29, 5, 22] have investigated VP approaches that first learn what is possible to do in a particular environment by self-supervised interaction, and then use the learned model to generate a visual plan from the current state to the goal, and finally apply visual servoing to follow the plan.

One particularly promising approach to VP is the semi-parametric topological memory (SPTM) method proposed by Savinov et al. [22] .

In SPTM, images collected offline are treated as nodes in a graph and represent the possible states of the system.

To connect nodes in this graph, an image classifier is trained to predict whether pairs of images were 'close' in the data or not, effectively learning which image transitions are feasible in a small number of steps.

The SPTM graph can then be used to generate a visual plan -a sequence of images between a pair of start and goal images -by directly searching the graph.

SPTM has several advantages, such as producing highly interpretable visual plans and the ability to plan long-horizon behavior.

However, since SPTM builds the visual plan directly from images in the data, when the environment changes -for example, the lighting varies, the camera is slightly moved, or other objects are displaced -SPTM requires recollecting images in the new environment; in this sense, SPTM does not generalize in a zero-shot sense.

Additionally, similar to [5] , we find that training the graph connectivity classifier as originally proposed by [22] requires extensive manual tuning.

Figure 1 : HTM illustration.

Top left: data collection.

In this illustration, the task is to move a green object between gray obstacles.

Data consists of multiple obstacle configurations (contexts), and images of random movement of the object in each configuration.

Bottom left: the elements of HTM.

A CVAE is trained to hallucinate images of the object and obstacles conditioned on the obstacle image context.

A connectivity energy model is trained to score pairs of images based on the feasibility of their transition.

Right: HTM visual planning.

Given a new context image and a pair of start and goal images, we first use the CVAE to hallucinate possible images of the object and obstacles.

Then, a connectivity graph (blue dotted lines) is computed based on the connectivity energy, and we plan for the shortest path from start to goal on this graph (orange solid line).

For executing the plan, a visual servoing controller is later used to track the image sequence.

In this work, we propose to improve both the robustness and zero-shot generalization of SPTM.

To tackle the issue of generalization, we assume that the environment is described using some context vector, which can be an image of the domain or any other observation data that contains enough information to extract a plan (see Figure 1 top left) .

We then train a conditional generative model that hallucinates possible states of the domain conditioned on the context vector.

Thus, given an unseen context, the generative model hallucinates exploration data without requiring actual exploration.

When building the connectivity graph with these hallucinated images, we replace the vanilla classifier used in SPTM with an energy-based model that employs a contrastive loss.

We show that this alteration drastically improves planning robustness and quality.

Finally, for planning, instead of connecting nodes in the graph according to an arbitrary threshold of the connectivity classifier, as in SPTM, we cast the planning as an inference problem, and efficiently search for the shortest path in a graph with weights proportional to the inverse of a proximity score from our energy model.

Empirically, we demonstrate that this provides much smoother plans and barely requires any hyperparameter tuning.

We term our approach Hallucinative Topological Memory (HTM).

A visual overview of our algorithm is presented in Figure 1 .

We evaluate our method on a set of simulated VP problems of moving an object between obstacles, which require long-horizon planning.

In contrast with prior work, which only focused on the success of the method in executing a task, here we also measure the interpretability of visual planning, through mean opinion scores of features such as image fidelity and feasibility of the image sequence.

In both measures, HTM outperforms state-of-the-art data-driven approaches such as visual foresight [4] and the original SPTM.

Context-Conditional Visual Planning and Acting (VPA) Problem.

We consider the contextconditional visual planning problem from [13, 29] .

Consider deterministic and fully-observable environments E 1 , ..., E N that are sampled from an environment distribution P E .

Each environment E i can be described by a context vector c i that entirely defines the dynamics o Figure 1 , the context could represent an image of the obstacle positions, which is enough to predict the possible movement of objects in the domain.

1 As is typical in VP problems, we assume our data D = {o

Ti , c i } i???{1,...,N } is collected in a self-supervised manner, and that in each environment E i , the observation distribution is defined as P o (??|c i ).

At test time, we are presented with a new environment, its corresponding context vector c, and a pair of start and goal observations o start , o goal .

Our goal is to use the training data to build a planner Q h (o start , o goal , c) and an h-horizon policy ?? h .

The planner's task is to generate a sequence of observations between o start and o goal , in which any two consecutive observations are reachable within h time steps.

The policy takes as input the image sequence and outputs a control policy that transitions the system from o start to o goal .

As the problem requires a full plan given only a context image in the new environment, the planner must be capable of zero-shot generalization.

Note that the planner and policy form an interpretable planning method that allows us to evaluate their performance separately.

For simplicity we will omit the subscript h for the planner and the policy.

Semi-Parametric Topological Memory (SPTM) [22] is a visual planning method that can be used to solve a special case of VPA.

where there is only a single training environment, E and no context image.

SPTM builds a memory-based planner and an inverse-model controller.

At training, a classifier R is trained to map two observation images o i , o j to a score ??? [0, 1] representing the feasibility of the transition, where images that are ??? h steps apart are labeled positive and images that are ??? l are negative.

The policy is trained as an inverse model L, mapping a pair of observation images o i , o j to an appropriate action a that transitions the system from o i to o j .

Given an unseen environment E * , new observations are manually collected and organized as nodes in a graph G. Edges in the graph connect observations o i , o j if R(o i , o j ) ??? s shortcut , where s shortcut is a manually defined threshold.

To plan, given start and goal observations o start and o goal , SPTM first uses R to localize, i.e., find the closest nodes in G to o start and o goal .

A path is found by running Dijkstra's algorithm, and the method then selects a waypoint o wi on the path which represents the farthest observation that is still feasible under R. Since both the current localized state o i and its waypoint o wi are in the observation space, we can directly apply the inverse model and take the action a i where a i = L(o i , o wi ).

After localizing to the new observation state reached by a i , SPTM repeats the process until the node closest to o goal is reached. [25] is a deep generative model that can be used for learning a high-dimensional conditional distribution P o (??|c).

The CVAE is trained by maximizing the evidence lower bound (ELBO):

, where q ?? (z|o, c) is the encoder that maps observations and contexts to the latent distribution, p ?? (o|z, c) is the decoder that maps latents and contexts to the observation distribution, and r ?? (z|c) is the prior that maps contexts to latent prior distributions.

Together p ?? , q ?? , r ?? are trained to maximize the variational lower bound above.

We assume that the prior and the encoder are Gaussian, which allows the D KL term to be computed in closed-form.

Monte-Carlo sampling and the reparametrization trick [12] are used to approximate the gradient of the loss.

Contrastive Predictive Coding (CPC) [17] extracts compact representations that maximize the causal and predictive aspects of high-dimensional sequential data.

A non-linear encoder g enc encodes the observation o t to a latent representation z t = g enc (o t ).

We maximize the mutual information between the latent representation z t and future observation o t+k with a log-

.

This model is trained to be proportional to the density ratio p(o t+k |z t )/p(o t+k ) by the CPC loss function: the cross entropy loss of correctly classifying a positive sample from a set X = {o 1 , ..., o N } of N random samples with 1 positive sample from p(o t+k |z t ) and N ??? 1 negative samples sampled from p(o t+k ):

SPTM has been shown to solve long-horizon planning problems such as navigation from first-person view [22] .

However, SPTM is not zero-shot: even a small change to the training environment requires collecting substantial exploration data for building the planning graph.

This can be a limitation in practice, especially in robotic domains, as any interaction with the environment requires robot time, and exploring a new environment can be challenging (indeed, [22] applied manual exploration).

In addition, similarly to [5] , we found that training the connectivity classifier as proposed in [22] requires extensive hyperparameter tuning.

In this section, we propose an extension of SPTM to overcome these two challenges by employing three ideas -(1) using a CVAE [25] to hallucinate samples in a zero-shot setting, (2) using contrastive loss for a more robust score function and planner, and (3) planning based on an approximate maximum likelihood formulation of the shortest path under uniform state distribution.

We call this approach Hallucinative Topological Memory (HTM), and next detail each component in our method.

We propose a zero-shot learning solution for automatically building the planning graph using only a context vector of the new environment.

Our idea is that, after seeing many different environments and corresponding states of the system during training, given a new environment we should be able to effectively hallucinate possible system states.

We can then use these hallucinations in lieu of real samples from the system in order to build the planning graph.

To generate images conditioned on a context, we implement a CVAE as depicted in Figure 1 .

During training, we learn the prior latent distribution r ?? (z|c), modeled as a Gaussian with mean ??(c) and covariance matrix ??(c), where ??(??) and ??(??) are learned non-linear neural network transformations.

During testing, when prompted with a new context vector c, we can sample latent vectors z 1 , ..., z N | c ??? N (??(c), ??(c)), and pass them through the decoder p ?? (x|z, c) for hallucinating samples in replacement of exploration data.

A critical component in the SPTM method is the connectivity classifier that decides which image transitions are feasible.

False positives may result in impossible short-cuts in the graph, while false negatives can make the plan unnecessarily long.

In [22] , the classifier was trained discriminatively, using observations in the data that were reached within h steps as positive examples, and more than l steps as negative examples, where h and l are chosen arbitrarily.

In practice, this leads to three important problems.

First, this method is known to be sensitive to the choice of positive and negative labeling [5] .

Second, training data are required to be long, non-cyclic trajectories for a high likelihood of sampling 'true' negative samples.

However, self-supervised interaction data often resembles random walks that repeatedly visit a similar state, leading to inconsistent estimates on what constitutes negative data.

Third, since the classifier is only trained to predict positively for temporally nearby images and negatively for temporally far away images, its predictions of medium-distance images can be arbitrary.

This creates both false positives and false negatives, thereby increasing shortcuts and missing edges in the graph.

To solve these problems, we propose to learn a connectivity score using contrastive predictive loss [17] .

Similar to CVAE, we initialize a CPC encoder g enc that takes in both observation and context, and a density-ratio model f k that does not depend on the context.

Through optimizing the CPC objective, f k of positive pairs are encouraged to be distinguishable from that of negative pairs.

Thus, it serves as a proxy for the temporal distance between two observations, leading to a connectivity score for planning.

Theoretically, CPC loss is better motivated than the classification loss in SPTM as it structures the latent space on a clear objective: maximize the mutual information between current and future observations.

In practice, this results in less hyperparameter tuning and a smoother distance manifold in the representation space.

Finally, instead of only sampling from the same trajectory as done in SPTM, our negative data are collected by sampling from the latent space of a trained CVAE or the replay buffer.

Without this trick, we found that the SPTM classifier fails to handle self-supervised data.

l) ).

This score reflects the difficulty in transitioning to the next state from the current state by self-supervised exploration.

The learned connectivity graph G can be viewed as a topological memory upon which we can use conventional graph planning methods to efficiently perform visual planning.

In the third step, we find the shortest path using Dijkstra's algorithm on the learned connectivity graph G between the start and end node.

In the fourth step, we apply our policy to follow the visual plan, reaching the next node in our shortest path and replan every fixed number of steps until we reach?? goal .

For the policy, we train an inverse model which predicts actions given two observations that are within h steps apart.

Maximum likelihood trajectory with Dijkstra's.

We show that the CPC loss can be utilized to cast the planning problem as an inference problem, and results in an effective planning algorithm.

After training the CPC objective to convergence, we have

To estimate p(o t+k |o t )/p(o t+k ), we compute the normalizing factor o ???V [f k (o , o t )] for each o t by averaging over all nodes in the graph.

Let's define our non-negative weight from o t to o t+k as

A shortest-path planning algorithm finds T, o 0 , ..., o T that minimizes

, Thus, assuming that the self-supervised data distribution is approximately uniform, the shortest path algorithm with proposed weight ?? maximizes a lower bound on the trajectory likelihood given the start and goal states.

In practice, this leads to a more stable planning approach and yields more feasible plans.

Reinforcement Learning.

Most of the study of data-driven planning has been under the model-free RL framework [23, 15, 24] .

However, the need to design a reward function, and the fact that the learned policy does not generalize to tasks that are not defined by the specific reward, has motivated the study of model-based approaches.

Recently, [11, 8] investigated model-based RL from pixels on Mujoco and Atari domains, but did not study generalization to a new environment. [6, 4] explored model-based RL with image-based goals using visual model predictive control (visual MPC).

These methods rely on video prediction, and are limited in the planning horizon due to accumulating errors.

In comparison, our method does not predict full trajectories but only individual images, mitigating this problem.

Our method can also use visual MPC as a replacement for the visual servoing policy.

Self-supervised learning.

Several studies investigated planning goal directed behavior from data obtained offline, e.g., by self-supervised robot interaction [1, 18] .

Nair et al. [16] used an inverse model to reach local sub-goals, but require human demonstrations of long-horizon plans.

Wang et al. [29] solve the visual planning problem using a conditional version of Causal InfoGAN [13] .

However, as training GAN is unstable and requires tedious model selection [21] , we opted for the CVAE-based approach, which is much more robust.

Classical planning and representation learning.

In classical planning literature, task and motion planning also separates the high-level planning and the low-level controller [30, 27, 10] .

In these works, domain knowledge is required to specify preconditions and effects at the task level.

Our approach only requires data collected through self-supervised interaction.

Other studies that bridge between classical planning and representation learning include [13, 3, 2, 5] .

These works, however, do not consider zero-shot generalization.

While Srinivas et al. [26] and Qureshi et al. [19] learn representations that allow goal-directed planning to unseen environments, they require expert training trajectories.

Ichter and Pavone [8] also generalizes motion planning to new environments, but require a collision checker and valid samples from test environments.

Recent work in visual planning (e.g., [13, 29, 4] ) focused on real robotic tasks with visual input.

While impressive, such results can be difficult to reproduce or compare.

For example, it is not clear whether manipulating a rope with the PR2 robot [29] is more or less difficult than manipulating a rigid object among many visual distractors [4] .

In light of this difficulty, we propose a suite of simulated tasks with an explicit difficulty scale and clear evaluation metrics.

Our domains consider moving a rigid object between obstacles using Mujoco [28] , and by varying the obstacle positions, we can control the planning difficulty.

For example, placing the object in a cul-de-sac would require non-trivial planning compared to simply moving around an obstacle along the way to the goal.

We thus create two domains, as seen in Figure 2: 1.

Block wall:: A green block navigates around a static red obstacle, which can vary in position.

2.

Block wall with complex obstacle: Similar to the above, but here the wall is a 3-link object which can vary in position, joint angles, and length, making the task significantly harder.

With these domains, we aim to asses the following attributes:

??? Does HTM improve visual plan quality over state-of-the-art VP methods [22, 4] ?

??? How does HTM execution success rate compare to state-of-the-art VP methods?

??? How well does HTM generalize its planning to unseen contexts?

We discuss our evaluation metrics for these attributes in Section 5.1.

To fully assess success of HTM relative to other state-of-the-art VP methods, we run these evaluation metrics on SPTM [22] and Visual Foresight [4] .

In the first baseline, since vanilla SPTM cannot plan in a new environment, we use the same samples generated by the same CVAE as HTM, and then build the graph by assigning edge weights in the graph proportional to their exponentiated SPTM classifier score.

3 We also give it the same negative sampling proceedure as HTM.

The same low-level controller is also used to follow the plans.

In the second baseline, Visual Foresight trains a video prediction model, and then performs model predictive control (MPC) which finds the optimal action sequence through random shooting.

For the random shooting, we used 3 iterations of the cross-entropy method with 200 sample sequences.

The MPC acts for 10 steps and replans, where the planning horizon T is 15.

We use the state-of-the-art video predictor as proposed by Lee et al. [14] and the public code provided by the authors.

For evaluating trajectories in random shooting, we studied two cost functions that are suitable for our domains: pixel MSE loss and green pixel distance.

The pixel MSE loss computes the pixel distance between the predicted observations and the goal image.

This provides a sparse signal when the object pixels in the plan can overlap with those of the goal.

We also investigate a cost function that uses prior knowledge about the task -the position of the moving green block, which is approximated by calculating the center of mass of the green pixels.

As opposed to pixel MSE, the green pixel distance provides a smooth cost function which estimates the normalized distance between the estimated block positions of the predicted observations and the goal image.

Note that this assumes additional domain knowledge compared to HTM.

We design a set of tests that measure both qualitative and quantitative performance of an algorithm.

To motivate the need for qualitative metrics, we reiterate the importance of planning interpretability; it is highly desirable that the generated plan visually make sense so as to allow a human to approve of the plan prior to execution.

Qualitative Visual plans have the essential property of being intuitive, in that the imagined trajectory is perceptually sensible.

Since these qualities are highly subjective, we devised a set of tests to evaluate plans based on human visual perception.

For each domain, we asked 5 participants to visually score 5 randomly generated plans from each model by answering the following questions: (1) Fidelity: Does the pixel quality of the images resemble the training data?; (2) Feasibility: Is each transition in the generated plan executable by a single action step?; and (3) Completeness: Is the goal reachable from the last image in the plan using a single action?

Answers were in the range [0,1], where 0 denotes No to the proposed question and 1 means Yes.

The mean opinion score were calculated for each model.

Quantitative In addition to generating visually sensible trajectories, a planning algorithm must also be able to successfully navigate towards a predefined goal.

Thus, for each domain, we selected 20 start and goal images, each with an obstacle configuration unseen during training.

Success was measured by the ability to get within some L2 distance to the goal in a n steps or less, where the distance threshold and n varied on the domain but was held constant across all models.

A controller specified by the algorithm executed actions given an imagined trajectory, and replanning occurred every r steps.

Specific details can be found in the Appendix D.

As shown in Table 5 .2, HTM outperforms all baselines in both qualitative and quantitative measurements across all domains.

In the simpler block wall domain, Visual Foresight with green pixel distance only succeeds under the assumption of additional state information of the object's location.

the other algorithms do not have.

However, in the complex obstacle domain, Visual Foresight fails to perform comparably to our algorithm, regardless of the additional assumption.

We also compared our method with SPTM, using the same inverse model and CVAE to imagine testing samples.

However, without a robust classification loss and improved method of weighting the graph's edges, SPTM often fails to find meaningful transitions.

In regards to perceptual evaluation, Visual Foresight generates realistic transitions, as seen by the high participant scores for feasibility.

However, the algorithm is limited in creating a visual plan within the optimal T = 15 timesteps.

4 Thus, when confronted with a challenging task of navigating around a convex shape where the number of timesteps required exceeds T , Visual Foresight fails to construct a reliable plan (see Figure 3) , and thus lacks plan completeness.

Conversely, SPTM is able to imagine some trajectory that will reach the goal state.

However, as mentioned above and was confirmed in the perceptual scores, SPTM fails to select feasible transitions, such as imagining a trajectory where the block will jump across the wall or split into two blocks.

Our approach, on the other hand, received the highest scores of fidelity, feasibility, and completeness.

Finally, we show in Figure 3 the results of our two proposed improvements to SPTM in isolation.

The results clearly show that a classifier using contrastive loss outperforms that which uses Binary Cross Entropy (BCE) loss, and furthermore that the inverse of the score function for edge weighting is more successful than the best tuned version of binary edge weights.

Table 1 : Qualitative and quantitative evaluation for the the block wall (1) and block wall with complex obstacle (2) domains.

Qualitative data also displays the 95% confidence interval.

Note HTM (1) refers to edge weighting using the energy model, and (2) is weighting using the density ratio, as described in 3.3.

For the score function, we denote the energy model structured with contrastive loss as CPC and the classifier as proposed in [22] with BCE loss as SPTM.

For the edge weighting function, we test the binary edge weighting from the original SPTM paper, the inverse of the score function, and the inverse of the normalized score function.

We propose a method that is visually interpretable and modular -we first hallucinate possible configurations, then compute a connectivity between them, and then plan.

Our HTM can generalize to unseen environments and improve visual plan quality and execution success rate over state-of-the-art VP methods.

Our results suggest that combining classical planning methods with data-driven perception can be helpful for long-horizon visual planning problems, and takes another step in bridging the gap between learning and planning.

In future work, we plan to combine HTM with Visual MPC for handling more complex objects, and use object-oriented planning for handling multiple objects.

Another interesting aspect is to improve planning by hallucinating samples conditioned on the start and goal configurations, which can help reduce the search space during planning.

In this section, we assume the dataset as described in VPA, D = {o

.

There are two ways of learning a model to distinguish the positive from the negative transitions.

Classifier: As noted above, SPTM first trains a classifier which distinguishes between an image pair that is within h steps apart, and the images that are far apart using random sampling.

The classifier is used to localize the current image and find possible next images for planning.

In essence, the classifier contains the encoder g ?? that embeds the observation x and the the score function f that takes the embedding of each image and output the logit for a sigmoid function.

The binary cross entropy loss of the classifier can be written as follows:

, where z ??? t is a random sample from D. Energy model: Another form of discriminating the the positive transition out of negative transitions is through an energy model.

Oord et al. [17] learn the embeddings of the current states that are predictive of the future states.

Let g be an encoder of the input x and z = g ?? (x) be the embedding.

The loss function can be described as a cross entropy loss of predicting the correct sample from N + 1 samples which contain 1 positive sample and N negative samples:

, where f ?? (u, v) = exp (u T ??v) and z Figure 6: Sample observations (top) and contexts (bottom).

In this domain, an object can be translated and rotated (SE(2)) slightly per timestep.

The data are collected from 360 different object shapes with different number of building blocks between 3 to 7.

Each object is randomly initialized 50 times and each episode has length 30.

The goal is to plan a manipulation of an unseen object through the narrow gap between obstacles in zero-shot.

@highlight

We propose Hallucinative Topological Memory (HTM), a visual planning algorithm that can perform zero-shot long horizon planning in new environments. 
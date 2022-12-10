Object-based factorizations provide a useful level of abstraction for interacting with the world.

Building explicit object representations, however, often requires supervisory signals that are difficult to obtain in practice.

We present a paradigm for learning object-centric representations for physical scene understanding without direct supervision of object properties.

Our model, Object-Oriented Prediction and Planning (O2P2), jointly learns a perception function to map from image observations to object representations, a pairwise physics interaction function to predict the time evolution of a collection of objects, and a rendering function to map objects back to pixels.

For evaluation, we consider not only the accuracy of the physical predictions of the model, but also its utility for downstream tasks that require an actionable representation of intuitive physics.

After training our model on an image prediction task, we can use its learned representations to build block towers more complicated than those observed during training.

Consider the castle made out of toy blocks in Figure 1a .

Can you imagine how each block was placed, one-by-one, to build this structure?

Humans possess a natural physical intuition that aids in the performance of everyday tasks.

This physical intuition can be acquired, and refined, through experience.

Despite being a core focus of the earliest days of artificial intelligence and computer vision research BID19 BID28 , a similar level of physical scene understanding remains elusive for machines.

Cognitive scientists argue that humans' ability to interpret the physical world derives from a richly structured apparatus.

In particular, the perceptual grouping of the world into objects and their relations constitutes core knowledge in cognition BID24 .

While it is appealing to apply such an insight to contemporary machine learning methods, it is not straightforward to do so.

A fundamental challenge is the design of an interface between the raw, often high-dimensional observation space and a structured, object-factorized representation.

Existing works that have investigated the benefit of using objects have either assumed that an interface to an idealized object space already exists or that supervision is available to learn a mapping between raw inputs and relevant object properties (for instance, category, position, and orientation).Assuming access to training labels for all object properties is prohibitive for at least two reasons.

The most apparent concern is that curating supervision for all object properties of interest is difficult to scale for even a modest number of properties.

More subtly, a representation based on semantic a) b)Figure 1: (a) A toy block castle.

(b) Our method's build of the observed castle, using its learned object representations as a guide during planning.

The second uses ground-truth labels of object properties to supervise a learning algorithm that can map to the space of a traditional or learned physics engine.

(c) O2P2, like (b), employs an object factorization and the functional structure of a physics engine, but like (a), does not assume access to supervision of object properties.

Without object-level supervision, we must jointly learn a perception function to map from images to objects, a physics engine to simulate a collection of objects, and a rendering engine to map a set of objects back to a single composite image prediction.

In all three approaches, we highlight the key supervision in orange.attributes can be limiting or even ill-defined.

For example, while the size of an object in absolute terms is unambiguous, its orientation must be defined with respect to a canonical, class-specific orientation.

Object categorization poses another problem, as treating object identity as a classification problem inherently limits a system to a predefined vocabulary.

In this paper, we propose Object-Oriented Prediction and Planning (O2P2), in which we train an object representation suitable for physical interactions without supervision of object attributes.

Instead of direct supervision, we demonstrate that segments or proposal regions in video frames, without correspondence between frames, are sufficient supervision to allow a model to reason effectively about intuitive physics.

We jointly train a perception module, an object-factorized physics engine, and a neural renderer on a physics prediction task with pixel generation objective.

We evaluate our learned model not only on the quality of its predictions, but also on its ability to use the learned representations for tasks that demand a sophisticated physical understanding.

In this section, we describe a method for learning object-based representations suitable for planning in physical reasoning tasks.

As opposed to much prior work on object-factorized scene representations (Section 4), we do not supervise the content of the object representations directly by way of labeled attributes (such as position, velocity, or orientation).

Instead, we assume access only to segments or region proposals for individual video frames.

Since we do not have labels for the object representations, we must have a means for converting back and forth between images and object representations for training.

O2P2 consists of three components, which are trained jointly:• A perception module that maps from an image to an object encoding.

The perception module is applied to each object segment independently.• A physics module to predict the time evolution of a set of objects.

We formulate the engine as a sum of binary object interactions plus a unary transition function.• A rendering engine that produces an image prediction from a variable number of objects.

We first predict an image and single-channel heatmap for each object.

We then combine all of the object images according to the weights in their heatmaps at every pixel location to produce a single composite image.

A high-level overview of the model is shown in FIG0 .

Below, we give details for the design of each component and their subsequent use in a model-based planning setting.

The perception module is a four-layer convolutional encoder that maps an image observation to object representation vectors O = {o k } k=1...N .

We assume access to a segmentation of the input image S = {s k } k=1...N and apply the encoder individually to each segment.

The perception module is not supervised directly to predict semantically meaningful properties such as position or orientation; instead, its outputs are used by the physics and rendering modules to make image predictions.

In this way, the perception module must be trained jointly with the other modules.

The physics module predicts the effects of simulating a collection of object representations O forward in time.

As in Chang et al. (2016) ; BID27 , we consider the interactions of all pairs of object vectors.

The physics engine contains two learned subcomponents: a unary transition function f trans applied to each object representation independently, and a binary interaction function f interact applied to all pairs of object representations.

LettingŌ = {ō k } k=1...

N denote the output of the physics predictor, the k th object is given DISPLAYFORM0 where both f trans and f interact are instantiated as two-layer MLPs.

Much prior work has focused on learning to model physical interactions as an end goal.

In contrast, we rely on physics predictions only insofar as they affect action planning.

To that end, it is more important to know the resultant effects of an action than to make predictions at a fixed time interval.

We therefore only need to make a single prediction,Ō = f physics (O), to estimate the steady-state configuration of objects as a result of simulating physics indefinitely.

This simplification avoids the complications of long-horizon sequential prediction while retaining the information relevant to planning under physical laws and constraints.

Because our only supervision occurs at the pixel level, to train our model we learn to map all objectvector predictions back to images.

A challenge here lies in designing a function which constructs a single image from an entire collection of objects.

The learned renderer consists of two networks, both instantiated as convolutional decoders.

The first network predicts an image independently for each input object vector.

Composing these images into a single reconstruction amounts to selecting which object is visible at every pixel location.

In a traditional graphics engine, this would be accomplished by calculating a depth pass at each location and rendering the nearest object.

To incorporate this structure into our learned renderer, we use the second decoder network to produce a single-channel heatmap for each object.

The composite scene image is a weighted average of all of the object-specific renderings, where the weights come from the negative of the predicted heatmaps.

In effect, objects with lower heatmap predictions at a given pixel location will be more visible than objects with higher heatmap values.

This encourages lower heatmap values for nearer objects.

Although this structure is reminiscent of a depth pass in a traditional renderer, the comparison should not be taken literally; the model is only supervised by composite images and no true depth maps are provided during training.

We train the perception, physics, and rendering modules jointly on an image reconstruction and prediction task.

Our training data consists of image pairs (I 0 , I 1 ) depicting a collection of objects on a platform before and after a new object has been dropped.

(I 0 shows one object mid-air, as if being held in place before being released.

We refer to Section 3 for details about the generation of training data.)

We assume access to a segmentation S 0 for the initial image I 0 .Given the observed segmented image S 0 , we predict object representations using the perception module O = f percept (S 0 ) and their time-evolution using the physics moduleŌ = f physics (O).

The rendering engine then predicts an image from each of the object representations: DISPLAYFORM0 We compare each image predictionÎ t to its ground-truth counterpart using both L 2 distance and a perceptual loss L VGG .

As in BID12 , we use L 2 distance in the feature space of a

Input perception, physics, and rendering modules fpercept, fphysics, frender Input goal image I goal with N segments S goal = {s DISPLAYFORM0 Segment the objects that have already been placed to yield S curr 4: DISPLAYFORM1 Sample action am of the form (shape, position, orientation, color) from uniform distribution 6:Observe action am as a segment sm by moving object to specified position and orientation 7:Concatenate the observation and segments of existing objects S m = {sm} ∪ S BID23 as a perceptual loss function.

The perception module is supervised by the reconstruction of I 0 , the physics engine is supervised by the reconstruction of I 1 , and the rendering engine is supervised by the reconstruction of both images.

DISPLAYFORM2

We now describe the use of our perception, physics, and rendering modules in the representative planning task depicted in Figure 1 , in which the goal is to build a block tower to match an observed image.

Here, matching a tower does not refer simply to producing an image from the rendering engine that looks like the observation.

Instead, we consider the scenario where the model must output a sequence of actions to construct the configuration.

This setting is much more challenging because there is an implicit sequential ordering to building such a tower.

For example, the bottom cubes must be placed before the topmost triangle.

O2P2 was trained solely on a pixel-prediction task, in which it was never shown such valid action orderings (or any actions at all).

However, these orderings are essentially constraints on the physical stability of intermediate towers, and should be derivable from a model with sufficient understanding of physical interactions.

Although we train a rendering function as part of our model, we guide the planning procedure for constructing towers solely through errors in the learned object representation space.

The planning procedure, described in detail in Algorithm 1, can be described at a high level in four components:1.

The perception module encodes the segmented goal image into a set of object representations O goal .2.

We sample actions of the form (shape, position, orientation, color), where shape is categorical and describes the type of block, and the remainder of the action space is continuous and describes the block's appearance and where it should be dropped.

3.

We evaluate the samples by likewise encoding them as object vectors and comparing them with O goal .

We view action sample a m as an image segment s m (analogous to observing a block held in place before dropping it) and use the perception module to produce object vectors O m .

Because the actions selected should produce a stable tower, we run these object representations through the physics engine to yieldŌ m before comparing with O goal .

The cost is the L 2 distance between the objectō ∈Ō m corresponding to the most recent action and the goal object in O goal that minimizes this distance.

4. Using the action sampler and evaluation metric, we select the sampled action that minimizes L 2 distance.

We then execute that action in MuJoCo BID25 .

We continue this procedure, iteratively re-planning and executing actions, until there are as many actions in the DISPLAYFORM0 Figure 3: Given an observed segmented image I 0 at t = 0, our model predicts a set of object representations O, simulates the objects with a learned physics engine to produceŌ = f physics (O), and renders the resulting predictionsÎ = f render (Ō), the scene's appearance at a later time.

We use the convention (in all figures) that observations are outlined in green, other images rendered with the ground-truth renderer are outlined in black, and images rendered with our learned renderer are outlined in blue.executed sequence as there are objects in the goal image.

In the simplest case, the distribution from which actions are sampled may be uniform, as in Algorithm 1.

Alternatively, the crossentropy method (CEM) BID20 ) may be used, repeating the sampling loop multiple times and fitting a Gaussian distribution to the lowest-cost samples.

In practice, we used CEM starting from a uniform distribution with five iterations, 1000 samples per iteration, and used the top 10% of samples to fit the subsequent iteration's sampling distribution.

In our experimental evaluation, we aim to answer the following questions, (1) After training solely on physics prediction tasks, can O2P2 reason about physical interactions in an actionable and useful way?

(2) Does the implicit object factorization imposed by O2P2's structure provide a benefit over an object-agnostic black-box video prediction approach?

(3) Is an object factorization still useful even without supervision for object representations?

3.1 IMAGE RECONSTRUCTION AND PREDICTION We trained O2P2 to reconstruct observed objects and predict their configuration after simulating physics, as described in Section 2.4.

To generate training data, we simulated dropping a block on top of a platform containing up to four other blocks.

We varied the position, color, and orientation of three block varieties (cubes, rectangular cuboids, and triangles).

In total, we collected 60,000 training images using the MuJoCo simulator.

Since our physics engine did not make predictions at every timestep (Section 2.2), we only recorded the initial and final frame of a simulation.

For this synthetic data, we used ground truth segmentations corresponding to visible portions of objects.

Representative predictions of our model for image reconstruction (without physics) and prediction (with physics) on held-out random configurations are shown in Figure 3 .

Even when the model's predictions differed from the ground truth image, such as in the last row of the figure, the physics engine produced a plausible steady-state configuration of the observed scene.

3.2 BUILDING TOWERS After training O2P2 on the random configurations of blocks, we fixed its parameters and employed the planning procedure as described in Section 2.5 to build tower configurations observed in images.

We also evaluated the following models as comparisons:• No physics is an ablation of our model that does not run the learned physics engine, but instead simply setsŌ = O • Stochastic adversarial video prediction (SAVP), a block-box video prediction model which does not employ an object factorization BID15 .

The cost function of samples is evaluated directly on pixels.

The sampling-based planning routine is otherwise the same as in ours.

Figure 4: Qualitative results on building towers using planning.

Given an image of the goal tower, we can use the learned object representations and predictive model in O2P2 for guiding a planner to place blocks in the world and recreate the configuration.

We compare with an ablation, an objectagnostic video prediction model, and two 'oracles' with access to the ground-truth simulator.

• Oracle (pixels) uses the MuJoCo simulator to evaluate samples instead of our learned physics and graphics engines.

The cost of a block configuration is evaluated directly in pixel space using L 2 distance.• Oracle (objects) also uses MuJoCo, but has access to segmentation masks on input images while evaluating the cost of proposals.

Constraining proposed actions to account for only a single object in the observation resolves some of the inherent difficulties of using pixel-wise loss functions.

Qualitative results of all models are shown in Figure 4 and a quantitative evaluation is shown in TAB1 .

We evaluated tower stacking success by greedily matching the built configuration to the ground-truth state of the goal tower, and comparing the maximum object error (defined on its position, identity, and color) to a predetermined threshold.

Although the threshold is arbitrary in the sense that it can be chosen low enough such that all builds are incorrect, the relative ordering of the models is robust to changes in this value.

All objects must be of the correct shape for a built tower to be considered correct, meaning that our third row prediction in Figure 4 was incorrect because a green cube was mistaken for a green rectangular cuboid.

While SAVP made accurate predictions on the training data, it did not generalize well to these more complicated configurations with more objects per frame.

As such, its stacking success was low.

Physics simulation was crucial to our model, as our No-physics ablation failed to stack any towers correctly.

We explored the role of physics simulation in the stacking task in Section 3.3.

The 'oracle' model with access to the ground-truth physics simulator was hampered when making comparisons in pixel space.

A common failure mode of this model was to drop a single large block on the first step to cover the visual area of multiple smaller blocks in the goal image.

This scenario was depicted by the blue rectangular cuboid in the first row of Figure 4 in the Oracle (pixels) column.

FIG2 depicts the entire planning and execution procedure for O2P2 on a pyramid of six blocks.

At each step, we visualize the process by which our model selects an action by showing a heatmap of

First action Execution Figure 6 : Heatmaps showing sampled action scores for the initial action given a goal block tower.

O2P2's scores reflect that the objects resting directly on the platform must be dropped first, and that they may be dropped from any height because they will fall to the ground.

The No-physics ablation, on the other hand, does not implicitly represent that the blocks need to be dropped in a stable sequence of actions because it does not predict the blocks moving after being released.scores (negative MSE) for each action sample according to the sample's (x, y) position FIG2 ).

Although the model is never trained to produce valid action decisions, the planning procedure selects a physically stable sequence of actions.

For example, at the first timestep, the model scores three x-locations highly, corresponding to the three blocks at the bottom of the pyramid.

It correctly determines that the height at which it releases a block at any of these locations does not particularly matter, since the block will drop to the correct height after running the physics engine.

FIG2 shows the selected action at each step, and FIG2 shows the model's predictions about the configuration after releasing the sampled block.

Similar heatmaps of scored samples are shown for the No-physics ablation of our model in Figure 6 .

Because this ablation does not simulate the effect of dropping a block, its highly-scored action samples correspond almost exactly to the actual locations of the objects in the goal image.

Further, without physics simulation it does not implicitly select for stable action sequences; there is nothing to prevent the model from selecting the topmost block of the tower as the first action.

Planning for alternate goals.

By implicitly learning the underlying physics of a domain, our model can be used for various tasks besides matching towers.

In Figure 7a , we show our model's representations being used to plan a sequence of actions to maximize the height of a tower.

There is no observation for this task, and the action scores are calculated based on the highest non-zero pixels after rendering samples with the learned renderer.

In Figure 7b , we consider a similar sampling procedure as in the tower-matching experiments, except here only a single unstable block is shown.

Matching a free-floating block requires planning with O2P2 for multiple steps at once.

Figure 7: O2P2 being used to plan for the alternate goals of (a) maximizing the height of a tower and (b) making an observed block stable by use of any other blocks.

Figure 8: Ten goal images alongside the result of the Sawyer's executed action sequence using O2P2 for planning.

The seven action sequences counted as correct are outlined in solid black; the three counted as incorrect are outlined in dashed lines.

We refer the reader to Appendix B for more evaluation examples and people.eecs.berkeley.edu/∼janner/o2p2 for videos of the evaluation.

We evaluated O2P2 on a Sawyer robotic arm using real image inputs.

We deployed the same perception, physics, and rendering modules used on synthetic data with minor changes to the planning procedure to make real-world evaluation tractable.

Instead of evaluating a sampled action by moving an appropriate block to the specified position and inferring object representations with the perception module, we trained a separate two-layer MLP to map directly from actions to object representations.

We refer to this module as the embedder: o m = f embedder (a m ).Mapping actions to object representations removed the need to manually move every sampled block in front of the camera, which would have been prohibitively slow on a real robot.

The embedder was supervised by the predicted object representations of the perception module on real image inputs; we collected a small dataset of the Sawyer gripper holding each object at one hundred positions and recorded the ground truth position of the gripper along with the output of the perception module for the current observation.

The embedder took the place of lines 6-8 of Algorithm 1.

We also augmented the objective used to select actions in line 11.

In addition to L 2 distance between goal and sampled object representations, we used a pixelwise L 2 distance between the observed and rendered object segments and between the rendered object segments before and after use of the physics module.

The latter loss is useful in a real setting because the physical interactions are less predictable than their simulated counterparts, so by penalizing any predicted movement we preferentially placed blocks directly in a stable position.

By using end-effector position control on the Sawyer gripper, we could retain the same action space as in synthetic experiments.

Because the position component of the sampled actions referred to the block placement location, we automated the picking motion to select the sampled block based on the shape and color components of an action.

Real-world evaluation used colored wooden cubes and rectangular cuboids.

Real image object segments were estimated by applying a simple color filter and finding connected components of sufficient size.

To account for shading and specularity differences, we replaced all pixels within an object segment by the average color within the segment.

To account for noisy segment masks, we replaced each mask with its nearest neighbor (in terms of pixel MSE) in our MuJoCo-rendered training set.

We tested O2P2 on twenty-five goal configurations total, of which our model correctly built seventeen.

Ten goal images, along with the result of our model's executed action sequence, are shown in Figure 8 .

The remainder of the configurations are included in Appendix B.

Our work is situated at the intersection of two distinct paradigms.

In the first, a rigid notion of object representation is enforced via supervision of object properties (such as size, position, and identity).In the second, scene representations are not factorized at all, so no extra supervision is required.

These two approaches have been explored in a variety of domains.

Image and video understanding.

The insight that static observations are physically stable configurations of objects has been leveraged to improve 3D scene understanding algorithms.

For example, ; BID9 ; BID22 ; BID11 build physically-plausible scene representations using such stability constraints.

We consider a scenario in which the physical representations are learned from data instead of taking on a predetermined form.

BID30 a) encode scenes in a markup-style representation suitable for consumption by off-the-shelf rendering engines and physics simulators.

In contrast, we do not assume access to supervision of object properties (only object segments) for training a perception module to map into a markup language.

There has also been much attention on inferring object-factorized, or otherwise disentangled, representations of images BID5 BID8 BID26 .

In contrast to works which aim to discover objects in a completely unsupervised manner, we focus on using object representations learned with minimal supervision, in the form of segmentation masks, for downstream tasks.

Object-centric scene decompositions have also been considered as a potential state representation in reinforcement learning BID3 BID21 BID2 BID7 BID13 .

We are specifically concerned with the problem of predicting and reasoning about physical phenomena, and show that a model capable of this can also be employed for decision making.

Learning and inferring physics.

BID6 ; BID27 BID1 have shown approaches to learning a physical interaction engine from data.

BID10 use a traditional physics engine, performing inference over object parameters, and show that such a model can account for humans' physical understanding judgments.

We consider a similar physics formulation, whereby update rules are composed of sums of pairwise object-interaction functions, and incorporate it into a training routine that does not have access to ground truth supervision in the form of object parameters (such as position or velocity).An alternative to using a traditional physics engine (or a learned object-factorized function trained to approximate one) is to treat physics prediction as an image-to-image translation or classification problem.

In contrast to these prior methods, we consider not only the accuracy of the predictions of our model, but also its utility for downstream tasks that are intentionally constructed to evaluate its ability to acquire an actionable representation of intuitive physics.

Comparing with representative video prediction BID15 BID0 and physical prediction BID4 BID18 BID17 BID16 ) methods, our approach achieves substantially better results at tasks that require building structures out of blocks.

We introduced a method of learning object-centric representations suitable for physical interactions.

These representations did not assume the usual supervision of object properties in the form of position, orientation, velocity, or shape labels.

Instead, we relied only on segment proposals and a factorized structure in a learned physics engine to guide the training of such representations.

We demonstrated that this approach is appropriate for a standard physics prediction task.

More importantly, we showed that this method gives rise to object representations that can be used for difficult planning problems, in which object configurations differ from those seen during training, without further adaptation.

We evaluated our model on a block tower matching task and found that it outperformed object-agnostic approaches that made comparisons in pixel-space directly.

@highlight

We present a framework for learning object-centric representations suitable for planning in tasks that require an understanding of physics.

@highlight

The paper presents a platform for predicting images of objects interacting with each other under the effect of gravitational forces.

@highlight

The paper presents a method that learns to reproduce 'block towers' from a given image.

@highlight

Proposes a method which learns to reason on physical interaction of different objects with no supervison of object properties.
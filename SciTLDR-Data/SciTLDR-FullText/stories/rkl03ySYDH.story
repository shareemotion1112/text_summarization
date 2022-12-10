The ability to decompose complex multi-object scenes into meaningful abstractions like objects is fundamental to achieve higher-level cognition.

Previous approaches for unsupervised object-oriented scene representation learning are either based on spatial-attention or scene-mixture approaches and limited in scalability which is a main obstacle towards modeling real-world scenes.

In this paper, we propose a generative latent variable model, called SPACE, that provides a uniﬁed probabilistic modeling framework that combines the best of spatial-attention and scene-mixture approaches.

SPACE can explicitly provide factorized object representations for foreground objects while also decomposing background segments of complex morphology.

Previous models are good at either of these, but not both.

SPACE also resolves the scalability problems of previous methods by incorporating parallel spatial-attention and thus is applicable to scenes with a large number of objects without performance degradations.

We show through experiments on Atari and 3D-Rooms that SPACE achieves the above properties consistently in comparison to SPAIR, IODINE, and GENESIS.

Results of our experiments can be found on our project website: https://sites.google.com/view/space-project-page

One of the unsolved key challenges in machine learning is unsupervised learning of structured representation for a visual scene containing many objects with occlusion, partial observability, and complex background.

When properly decomposed into meaningful abstract entities such as objects and spaces, this structured representation brings many advantages of abstract (symbolic) representation to areas where contemporary deep learning approaches with a global continuous vector representation of a scene have not been successful.

For example, a structured representation may improve sample efficiency for downstream tasks such as a deep reinforcement learning agent (Mnih et al., 2013) .

It may also enable visual variable binding (Sun, 1992) for reasoning and causal inference over the relationships between the objects and agents in a scene.

Structured representations also provide composability and transferability for better generalization.

Recent approaches to this problem of unsupervised object-oriented scene representation can be categorized into two types of models: scene-mixture models and spatial-attention models.

In scenemixture models (Greff et al., 2017; Burgess et al., 2019; Engelcke et al., 2019) , a visual scene is explained by a mixture of a finite number of component images.

This type of representation provides flexible segmentation maps that can handle objects and background segments of complex morphology.

However, since each component corresponds to a full-scale image, important physical features of objects like position and scale are only implicitly encoded in the scale of a full image and further disentanglement is required to extract these useful features.

Also, since it does not explicitly reflect useful inductive biases like the locality of an object in the Gestalt principles (Koffka, 2013) , the resulting component representation is not necessarily a representation of a local area.

Moreover, to obtain a complete scene, a component needs to refer to other components, and thus inference is inherently performed sequentially, resulting in limitations in scaling to scenes with many objects.

In contrast, spatial-attention models (Eslami et al., 2016; Crawford & Pineau, 2019) can explicitly obtain the fully disentangled geometric representation of objects such as position and scale.

Such features are grounded on the semantics of physics and should be useful in many ways (e.g., sample efficiency, interpretability, geometric reasoning and inference, transferability).

However, these models cannot represent complex objects and background segments that have too flexible morphology to be captured by spatial attention (i.e. based on rectangular bounding boxes).

Similar to scene-mixture models, previous models in this class show scalability issues as objects are processed sequentially.

In this paper, we propose a method, called Spatially Parallel Attention and Component Extraction (SPACE), that combines the best of both approaches.

SPACE learns to process foreground objects, which can be captured efficiently by bounding boxes, by using parallel spatial-attention while decomposing the remaining area that includes both morphologically complex objects and background segments by using component mixtures.

Thus, SPACE provides an object-wise disentangled representation of foreground objects along with explicit properties like position and scale per object while also providing decomposed representations of complex background components.

Furthermore, by fully parallelizing the foreground object processing, we resolve the scalability issue of existing spatial attention methods.

In experiments on 3D-room scenes and Atari game scenes, we quantitatively and qualitatively compare the representation of SPACE to other models and show that SPACE combines the benefits of both approaches in addition to significant speed-ups due to the parallel foreground processing.

The contributions of the paper are as follows.

First, we introduce a model that unifies the benefits of spatial-attention and scene-mixture approaches in a principled framework of probabilistic latent variable modeling.

Second, we introduce a spatially parallel multi-object processing module and demonstrate that it can significantly mitigate the scalability problems of previous methods.

Lastly, we provide an extensive comparison with previous models where we illustrate the capabilities and limitations of each method.

In this section, we describe our proposed model, Spatially Parallel Attention and Component Extraction (SPACE).

The main idea of SPACE, presented in Figure 1 , is to propose a unified probabilistic generative model that combines the benefits of the spatial-attention and scene-mixture models.

SPACE assumes that a scene x is decomposed into two independent latents: foreground z fg and background z bg .

The foreground is further decomposed into a set of independent foreground objects z fg = {z fg i } and the background is also decomposed further into a sequence of background segments z bg = z bg 1:K .

While our choice of modeling the foreground and background independently worked well empirically, for better generation, it may also be possible to condition one on the other.

The image distributions of the foreground objects and the background components are combined together with a pixel-wise mixture model to produce the complete image distribution:

Here, the foreground mixing probability α is computed as α = f α (z fg ).

This way, the foreground is given precedence in assigning its own mixing weight and the remaining is apportioned to the background.

The mixing weight assigned to the background is further sub-divided among the K background components.

These weights are computed as π k = f π k (z bg 1:k ) and k π k = 1.

With these notations, the complete generative model can be described as follows.

Figure 1: Illustration of the SPACE model.

SPACE consists of a foreground module and a background module.

In the foreground module, the input image is divided into a grid of H × W cells (4 × 4 in the figure).

An image encoder is used to compute the z where , z depth , and z pres for each cell in parallel.

z where is used to identify proposal bounding boxes and a spatial transformer is used to attend to each bounding box in parallel, computing a z what encoding for each cell.

The model selects patches using the bounding boxes and reconstructs them using a VAE from all the foreground latents z fg .

The background module segments the scene into K components (4 in the figure) using a pixel-wise mixture model.

Each component consists of a set of latents z bg = (z m , z c ) where z m models the mixing probability of the component and z c models the RGB distribution of the component.

The components are combined to reconstruct the background using a VAE.

The reconstructed background and foreground are then combined using a pixel-wise mixture model to generate the full reconstructed image.

We now describe the foreground and background models in more detail.

Foreground.

SPACE implements z fg as a structured latent.

In this structure, an image is treated as if it were divided into H × W cells and each cell is tasked with modeling at most one (nearby) object in the scene.

This type of structuring has been used in (Redmon et al., 2016; Santoro et al., 2017; Crawford & Pineau, 2019) .

Similar to SPAIR, in order to model an object, each cell i is associated with a set of latents (z ).

In this notation, z pres is a binary random variable denoting if the cell models any object or not, z where denotes the size of the object and its location relative to the cell, z depth denotes the depth of the object to resolve occlusions and z what models the object appearance and its mask.

These latents may then be used to compute the foreground image component p(x|z fg ) which is modeled as a Gaussian distribution N (µ fg , σ 2 fg ).

In practice, we treat σ 2 fg as a hyperparameter and decode only the mean image µ fg .

In this process, SPACE reconstructs the objects associated to each cell having z SPACE imposes a prior distribution on these latents as follows:

Here, only z pres i is modeled using a Bernoulli distribution while the remaining are modeled as Gaussian.

Since we cannot analytically evaluate the integrals in equation 2 due to the continuous latents z fg and z bg 1:K , we train the model using a variational approximation.

The true posterior on these variables is approximated as follows.

This is used to derive the following ELBO to train the model using the reparameterization trick and SGD (Kingma & Welling, 2013) .

See Appendix B for the detailed decomposition of the ELBO and the related details.

Parallel Inference of Cell Latents.

SPACE uses mean-field approximation when inferring the cell latents, so z

) for each cell does not depend on other cells.

As shown in Figure 1 , this allows each cell to act as an independent object detector, spatially attending to its own local region in parallel.

This is in contrast to inference in SPAIR, where each cell's latents auto-regressively depend on some or all of the previously traversed cells in a row-major order i.e., q(z

However, this method becomes prohibitively expensive in practice as the number of objects increases.

While Crawford & Pineau (2019) claim that these lateral connections are crucial for performance since they model dependencies between objects and thus prevent duplicate detections, we challenge this assertion by observing that 1) due to the bottom-up encoding conditioning on the input image, each cell should have information about its nearby area without explicitly communicating with other cells, and 2) in (physical) spatial space, two objects cannot exist at the same position.

Thus, the relation and interference between objects should not be severe and the mean-field approximation is a good choice in our model.

In our experiments, we verify empirically that this is indeed the case and observe that SPACE shows comparable detection performance to SPAIR while having significant gains in training speeds and efficiently scaling to scenes with many objects.

Preventing Box-Splitting.

If the prior for the bounding box size is set to be too small, then the model could split a large object by multiple bounding boxes and when the size prior is too large, the model may not capture small objects in the scene, resulting in a trade-off between the prior values of the bounding box size.

To alleviate this problem, we found it helpful to introduce an auxiliary loss which we call the boundary loss.

In the boundary loss, we construct a boundary of thickness b pixels along the borders of each glimpse.

Then, we restrict an object to be inside this boundary and penalize the model if an object's mask overlaps with the boundary area.

Thus, the model is penalized if it tries to split a large object by multiple smaller bounding boxes.

A detailed implementation of the boundary loss is mentioned in Appendix C.

Our proposed model is inspired by several recent works in unsupervised object-oriented scene decomposition.

The Attend-Infer-Repeat (AIR) (Eslami et al., 2016) framework uses a recurrent neural network to attend to different objects in a scene and each object is sequentially processed one at a time.

An object-oriented latent representation is prescribed that consists of 'what', 'where', and 'presence' variables.

The 'what' variable stores the appearance information of the object, the 'where' variable represents the location of the object in the image, and the 'presence' variable controls how many steps the recurrent network runs and acts as an interruption variable when the model decides that all objects have been processed.

Since the number of steps AIR runs scales with the number of objects it attends to, it does not scale well to images with many objects.

Spatially Invariant Attend, Infer, Repeat (SPAIR) (Crawford & Pineau, 2019) attempts to address this issue by replacing the recurrent network with a convolutional network.

Similar to YOLO (Redmon et al., 2016) , the locations of objects are specified relative to local grid cells rather than the entire image, which allow for spatially invariant computations.

In the encoder network, a convolutional neural network is first used to map the image to a feature volume with dimensions equal to a pre-specified grid size.

Then, each cell of the grid is processed sequentially to produce objects.

This is done sequentially because the processing of each cell takes as input feature vectors and sampled objects of nearby cells that have already been processed.

SPAIR therefore scales with the pre-defined grid size which also represents the maximum number of objects that can be detected.

Our model uses an approach similar to SPAIR to detect foreground objects, but importantly we make the foreground object processing fully parallel to scale to large number of objects without performance degradation.

Works based on Neural Expectation Maximization (Van Steenkiste et al., 2018; Greff et al., 2017 ) do achieve unsupervised object detection but do not explicitly model the presence, appearance, and location of objects.

These methods also suffer from the problem of scaling to images with a large number of objects.

For unsupervised scene-mixture models, several recent models have shown promising results.

MONet (Burgess et al., 2019) leverages a deterministic recurrent attention network that outputs pixel-wise masks for the scene components.

A variational autoencoder (VAE) (Kingma & Welling, 2013) is then used to model each component.

IODINE (Greff et al., 2019) approaches the problem from a spatial mixture model perspective and uses amortized iterative refinement of latent object representations within the variational framework.

GENESIS (Engelcke et al., 2019) also uses a spatial mixture model which is encoded by component-wise latent variables.

Relationships between these components are captured with an autoregressive prior, allowing complete images to be modeled by a collection of components.

We evaluate our model on two datasets: 1) an Atari (Bellemare et al., 2013) dataset that consists of random images from a pretrained agent playing the games, and 2) a generated 3D-room dataset that consists of images of a walled enclosure with a random number of objects on the floor.

In order to test the scalability of our model, we use both a small 3D-room dataset that has 4-8 objects and a large 3D-room dataset that has 18-24 objects.

Each image is taken from a random camera angle and the colors of the objects, walls, floor, and sky are also chosen at random.

Additional details of the datasets can be found in the Appendix E.

Baselines.

We compare our model against two scene-mixture models (IODINE and GENESIS) and one spatial-attention model (SPAIR).

Since SPAIR does not have an explicit background component, we add an additional VAE for processing the background.

Additionally, we test against two implementations of SPAIR: one where we train on the entire image using a 16 × 16 grid and another where we train on random 32 × 32 pixel patches using a 4 × 4 grid.

We denote the former model as SPAIR and the latter as SPAIR-P. SPAIR-P is consistent with the SPAIR's alternative training regime on Space Invaders demonstrated in Crawford & Pineau (2019) to address the slow training of SPAIR on the full grid size because of its sequential inference.

Lastly, for performance reasons, unlike the original SPAIR implementation, we use parallel processing for rendering the objects from their respective latents onto the canvas 1 for both SPAIR and SPAIR-P. Thus, because of these improvements, our SPAIR implementation can be seen as a stronger baseline than the original SPAIR.

The complete details of the architecture used is given in Appendix D.

In this section, we provide a qualitative analysis of the generated representations of the different models.

For each model, we performed a hyperparameter search and present the results for the best settings of hyperparameters for each environment.

Figure 2 shows sample scene decompositions of our baselines from the 3D-Room dataset and Figure 3 shows the results on Atari.

Note that SPAIR does not use component masks and IODINE and GENESIS do not separate foreground from background, hence the corresponding cells are left empty.

Additionally, we only show a few representative components for IODINE and GENESIS since we ran those experiments with larger K than can be displayed.

More qualitative results of SPACE can be found in Appendix A.

In the 3D-Room environment, IODINE is able to segment the objects and the background into separate components.

However, it occasionally does not properly decompose objects (in the Large 3D-room results, the orange sphere on the right is not reconstructed) and may generate blurry objects.

GENESIS is able to segment the background walls, floor, and sky into multiple components.

It is able to capture blurry foreground objects in the Small 3D-Room, but is not able to cleanly capture foreground objects with the larger number of objects in the Large 3D-Room.

In Atari, for all games, both IODINE and GENESIS fail to capture the foreground properly.

We believe this is because the objects in Atari games are smaller, less regular and lack the obvious latent factors like color and shape as in the 3D dataset, which demonstrates that detection-based approaches are more appropriate in this case.

SPAIR & SPAIR-P. SPAIR is able to detect tight bounding boxes in both 3D-Room and most Atari games (it does not work as well on dynamic background games, which we discuss below).

SPAIR-P, however, often fails to detect the foreground objects in proper bounding boxes, frequently uses multiple bounding boxes for one object and redundantly detects parts of the background as foreground objects.

This is a limitation of the patch training as the receptive field of each patch is limited to a 32 × 32 glimpse, hence prohibiting it to detect objects larger than that and making it difficult to distinguish the background from foreground.

These two properties are illustrated well in Space Invaders, where it is able to detect the small aliens, but it detects the long piece of background ground on the bottom of the image as foreground objects.

SPACE.

In 3D-Room, SPACE is able to accurately detect almost all objects despite the large variations in object positions, colors, and shapes, while producing a clean segmentation of the background walls, ground, and sky.

This is in contrast to the SPAIR model, while being able to provide similar foreground detection quality, encodes the whole background into a single component, which makes the representation less disentangled and the reconstruction more blurry.

Similarly in Atari, SPACE consistently captures all foreground objects while producing clean background segmentation across many different games.

Dynamic Backgrounds.

SPACE and SPAIR exhibit some very interesting behavior when trained on games with dynamic backgrounds.

For the most static game -Space Invaders, both SPACE and SPAIR work well.

For Air Raid, in which the background building moves, SPACE captures all objects accurately while providing a two-component segmentation, whereas SPAIR and SPAIR-P produce splitting and heavy re-detections.

In the most dynamic games, SPAIR completely fails because of the difficulty to model dynamic background with a single VAE component, while SPACE is able to perfectly segment the blue racing track while accurately detecting all foreground objects.

Foreground vs Background.

Typically, foreground is the dynamic local part of the scene that we are interested in, and background is the relatively static and global part.

This definition, though intuitive, is ambiguous.

Some objects, such as the red shields in Space Invaders and the key in Montezuma's Revenge ( Figure 5 ) are detected as foreground objects in SPACE, but are considered background in SPAIR.

Though these objects are static 2 , they are important elements of the games and should be considered as foreground objects.

Similar behavior is observed in Atlantis (Figure 7) , where SPACE detects some foreground objects from the middle base that is above the water.

We believe this is an interesting property of SPACE and could be very important for providing useful representations for downstream tasks.

By using a spatial broadcast network (Watters et al., 2019) which is much weaker when compared to other decoders like sub-pixel convolutional nets (Shi et al. (2016)), we limit the capacity of background module, which favors modeling static objects as foreground rather than background.

Boundary Loss.

We notice SPAIR sometimes splits objects into two whereas SPACE is able to create the correct bounding box for the objects (for example, see Air Raid).

This may be attributed to the addendum of the auxiliary boundary loss in the SPACE model that would penalize splitting an object with multiple bounding boxes.

In this section we compare SPACE with the baselines in several quantitative metrics 3 .

We first note that each of the baseline models has a different decomposition capacity (C), which we define as the capability of the model to decompose the scene into its semantic constituents such as the foreground objects and the background segmented components.

For SPACE, the decomposition capacity is equal to the number of grid cells H × W (which is the maximum number of foreground objects that can be detected) plus the number of background components K. For SPAIR, the decomposition capacity is equal to the number of grid cells H × W plus 1 for background.

For IODINE and GENESIS, it is equal to the number of components K.

For each experiment, we compare the metrics for each model with similar decomposition capacities.

This way, each model can decompose the image into the same number of components.

For a setting in SPACE with a grid size of H × W with K SPACE components, the equivalent settings in IODINE and GENESIS would be with C = (H × W ) + K SPACE .

The equivalent setting in SPAIR would be a grid size of H × W .

Step Latency.

The leftmost chart of Figure 4 shows the time taken to complete one gradient step (forward and backward propagation) for different decomposition capacities for each of the models.

We see that SPAIR's latency grows with the number of cells because of the sequential nature of its latent inference step.

Similarly GENESIS and IODINE's latency grows with the number of components K because each component is processed sequentially in both the models.

IODINE is

Training Latency Plot

Figure 4: Quantitative performance comparison between SPACE , SPAIR, IODINE and GENESIS in terms of batch-processing time during training, training convergence and converged pixel MSE.

Convergence plots showing pixel-MSE were computed on a held-out set during training.

the slowest overall with its computationally expensive iterative inference procedure.

Furthermore, both IODINE and GENESIS require storing data for each of the K components, so we were unable to run our experiments on 256 components or greater before running out of memory on our 22GB GPU.

On the other hand, SPACE employs parallel processing for the foreground which makes it scalable to large grid sizes, allowing it to detect a large number of foreground objects without any significant performance degradation.

Although this data was collected for gradient step latency, this comparison implies a similar relationship exists with inference time which is a main component in the gradient step.

Time for Convergence.

The remaining three charts in Figure 4 show the amount of time each model takes to converge in different experimental settings.

We use the pixel-wise mean squared error (MSE) as a measurement of how close a model is to convergence.

We see that not only does SPACE achieve the lowest MSE, it also converges the quickest out of all the models.

Average Precision and Error Rate.

In order to assess the quality of our bounding box predictions and the effectiveness of boundary loss, we measure the Average Precision and Object Count Error Rate of our predictions.

Our results are shown in Table 1 .

We only report these metrics for 3D-Room since we have access to the ground truth bounding boxes for each of the objects in the scene.

All three models have very similar average precision and error rate.

Despite being parallel in its inference, SPACE has a comparable count error rate to that of SPAIR.

SPACE also achieves better average precision and count error rate compared to its variant without the boundary loss (SPACE-WB), which shows the efficacy of our proposed loss.

From our experiments, we can assert that SPACE can produce similar quality bounding boxes as SPAIR while 1) having orders of magnitude faster inference and gradient step time, 2) converging more quickly, 3) scaling to a large number of objects without significant performance degradation, and 4) providing complex background segmentation.

We propose SPACE, a unified probabilistic model that combines the benefits of the object representation models based on spatial attention and the scene decomposition models based on component mixture.

SPACE can explicitly provide factorized object representation per foreground object while also decomposing complex background segments.

SPACE also achieves a significant speed-up and thus makes the model applicable to scenes with a much larger number of objects without performance degradation.

Besides, the detected objects in SPACE are also more intuitive than other methods.

We show the above properties of SPACE on Atari and 3D-Rooms.

Interesting future directions are to replace the sequential processing of background by a parallel one and to improve the model for natural images.

Our next plan is to apply SPACE for object-oriented model-based reinforcement learning.

Joseph Redmon, Santosh Divvala, Ross Girshick, and Ali Farhadi.

You only look once: Unified, real-time object detection. : Object detection and background segmentation using SPACE on 3D-Room data set with large number of objects.

In this section, we derive the ELBO for the log-likelihood log p(x).

KL Divergence for the Foreground Latents Under the SPACE 's approximate inference, the

inside the expectation can be evaluated as follows.

KL Divergence for the Background Latents Under our GENESIS-like modeling of inference for the background latents, the KL term inside the expectation for the background is evaluated as follows.

Relaxed treatment of z pres In our implementation, we model the Bernoulli random variable z pres i using the Gumbel-Softmax distribution (Jang et al., 2016) .

We use the relaxed value of z pres in the entire training and use hard samples only for the visualizations.

In this section we elaborate on the implementation details of the boundary loss.

We construct a kernel of the size of the glimpse, gs × gs (we use gs = 32) with a boundary gap of b = 6 having negative uniform weights inside the boundary and a zero weight in the region between the boundary and the glimpse.

This ensures that the model is penalized when the object is outside the boundary.

This kernel is first mapped onto the global space via STN Jaderberg et al. (2015) to obtain the global kernel.

This is then multiplied element-wise with global object mask α to obtain the boundary loss map.

The objective of the loss is to minimize the mean of this boundary loss map.

In addition to the ELBO, this loss is also back-propagated via RMSProp (Tieleman & Hinton. (2012) ).

This loss, due to the boundary constraint, enforces the bounding boxes to be less tight and results in lower average precision, so we disable the loss and optimize only the ELBO after the model has converged well.

D.1 ALGORITHMS Algorithm 1 and Algorithm 2 present SPACE's inference for foreground and background.

Algorithm 3 show the details of the generation process of the background module.

For foreground generation, we simply sample the latent variables from the priors instead of conditioning on the input.

Note that, for convenience the algorithms for the foreground module and background module are presented with for loops, but inference for all variables of the foreground module are implemented as parallel convolution operations and most operations of the background module (barring the LSTM module) are parallel as well. (2014)) optimizer with a learning rate of 1 × 10 −3 for the background module.

We use gradient clipping with a maximum norm of 1.0.

For Atari games, we find it beneficial to set α to be fixed for the first 1000-2000 steps, and vary the actual value and number of steps for different games.

This allows both the foreground as well as the background module to learn in the early stage of training.

Atari.

For each game, we sample 60,000 random images from a pretrained agent (Wu et al., 2016) .

We split the images into 50,000 for the training set, 5,000 for the validation set, and 5,000 for the testing set.

Each image is preprocessed into a size of 128 × 128 pixels with BGR color channels.

We present the results for the following games: Space Invaders, Air Raid, River Raid, Montezuma's Revenge.

We also train our model on a dataset of 10 games jointly, where we have 8,000 training images, 1,000 validation images, and 1,000 testing images for each game.

We use the following games: Asterix, Atlantis, Carnival, Double Dunk, Kangaroo, Montezuma Revenge, Pacman, Pooyan, Qbert, Space Invaders.

Room 3D.

We use MuJoCo (Todorov et al., 2012) to generate this dataset.

Each image consists of a walled enclosure with a random number of objects on the floor.

The possible objects are randomly sized spheres, cubes, and cylinders.

The small 3D-Room dataset has 4-8 objects and the large 3D-Room dataset has 18-24 objects.

The color of the objects are randomly chosen from 8 different colors and the colors of the background (wall, ground, sky) are chosen randomly from 5 different colors.

The angle of the camera is also selected randomly.

We use a training set of 63,000 images, a validation set of 7,000 images, and a test set of 7,000 images.

We use a 2-D projection from the camera to determine the ground truth bounding boxes of the objects so that we can report the average precision of the different models.

@highlight

We propose a generative latent variable model for unsupervised scene decomposition that provides factorized object representation per foreground object while also decomposing background segments of complex morphology.
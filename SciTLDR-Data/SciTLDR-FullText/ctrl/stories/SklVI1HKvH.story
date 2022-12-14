Point clouds are a flexible and ubiquitous way to represent 3D objects with arbitrary resolution and precision.

Previous work has shown that adapting encoder networks to match the semantics of their input point clouds can significantly improve their effectiveness over naive feedforward alternatives.

However, the vast majority of work on point-cloud decoders are still based on fully-connected networks that map shape representations to a fixed number of output points.

In this work, we investigate decoder architectures that more closely match the semantics of variable sized point clouds.

Specifically, we study sample-based point-cloud decoders that map a shape representation to a point feature distribution, allowing an arbitrary number of sampled features to be transformed into individual output points.

We develop three sample-based decoder architectures and compare their performance to each other and show their improved effectiveness over feedforward architectures.

In addition, we investigate the learned distributions to gain insight into the output transformation.

Our work is available as an extensible software platform to reproduce these results and serve as a baseline for future work.

Point clouds are an important data type for deep learning algorithms to support.

They are commonly used to represent point samples of some underlying object.

More generally, the points may be extended beyond 3D space to capture additional information about multi-sets of individual objects from some class.

The key distinction between point clouds and the more typical tensor data types is that the information content is invariant to the ordering of points.

This implies that the spatial relationships among points is not explicitly captured via the indexing structure of inputs and outputs.

Thus, standard convolutional architectures, which leverage such indexing structure to support spatial generalization, are not directly applicable.

A common approach to processing point clouds with deep networks is voxelization, where point clouds are represented by one or more occupancy-grid tensors (Zhou & Tuzel (2018) , Wu et al. (2018) ).

The grids encode the spatial dimensions of the points in the tensor indexing structure, which allows for the direct application of convolutional architectures.

This voxelization approach, however, is not appropriate in many use cases.

In particular, the size of the voxelized representation depends on the spatial extent of the point cloud relative to the spatial resolution needed to make the necessary spatial distinctions (such as distinguishing between different objects in LIDAR data).

In many cases, the required resolution will be unknown or result in enormous tensors, which can go beyond the practical space and time constraints of an application.

This motivates the goal of developing architectures that support processing point cloud data directly, so that processing scales with the number of points rather than the required size of an occupancy grid.

One naive approach, which scales linearly in the size of the point cloud, is to 'flatten' the point cloud into an arbitrarily ordered list.

The list can then be directly processed by standard convolutional or fully-connected (MLP) architectures directly.

This approach, however, has at least two problems.

First, the indexing order in the list carries no meaningful information, while the networks do not encode this as a prior.

Thus, the networks must learn to generalize in a way that is invariant to ordering, which can be data inefficient.

Second, in some applications, it is useful for point clouds to consist of varying numbers of points, while still representing the same underlying objects.

However, the number of points that can be consumed by the naive feedforward architecture is fixed.

PointNet (Qi et al., 2017) and Deepsets Zaheer et al. (2017) exhibit better performance over the MLP baseline with a smaller network by independently transforming each point into a high-dimensional representation with a single shared MLP that is identically applied to each individual point.

This set of derived point features is then mapped to a single, fixed-sized dense shape representation using a symmetric reduction function.

As such the architectures naturally scale to any number of input points and order invariance is built in as an architectural bias.

As a result, these architectures have been shown to yield significant advantages in applications in which point clouds are used as input, such as shape classification.

The success of PointNet and DeepSet style architectures in this domain shows that designing a network architecture to match the semantics of a point cloud results in a more efficient, and better performing network.

Since point clouds are such a useful object representation, it's natural to ask how we should design networks to decode point clouds from some provided shape representation.

This would allow for the construction of point cloud auto-encoders, which could serve a number of applications, such as anomaly detection and noise smoothing.

Surprisingly, the dominant approach to designing such a differentiable point cloud decoder is to feed the dense representation of the desired object through a single feedforward MLP whose result is then reshaped into the appropriate size for the desired point cloud.

This approach has similar issues as the flat MLP approach to encoding point clouds; the decoder can only produce a fixed-sized point cloud while point clouds are capable of representing objects at low or high levels of detail; the decoder only learns a single deterministic mapping from a shape representation to a point cloud while we know that point clouds are inherently random samples of the underlying object.

The primary goal and contribution of this paper is to study how to apply the same lessons learned from the PointNet encoder's semantic congruence with point clouds to a point cloud decoder design.

As such, we build on PointNet's principles to present the 'NoiseLearn' algorithm-a novel, simple, and effective point cloud decoding approach.

The simplicity of the decoding architectures and the increase in performance are strong indicators that sample-based decoders should be considered as a default in future studies and systems.

In addition, we investigate the operation of the decoders to gain insight into how the output point clouds are generated from a latent shape representation.

Point cloud decoders are a relatively unexplored area of research.

Among the works which describe an algorithm that produces a point cloud, the majority focus their efforts on learning a useful latent shape representation that is then passed to a MLP decoder.

PU-Net (Yu et al., 2018 ) is one such example, in which they design a novel point cloud upsampling network which uses a hierarchical approach to aggregating and expanding point features into a meaningful latent shape representation.

To decode the learned shape representation into a point cloud, the latent vector is then passed through a feedforward MLP to produce a fixed number of points.

This implies that the network would need to be retrained to allow for a different upsampling rate, which unlikely to be a desired property of an upsampling algorithm.

TopNet (Tchapmi et al., 2019) recognizes the data inefficiency of using a single MLP to decode a point cloud and instead reorganizes their MLP into a hierarchical tree structure in which MLPs at the same level share the same parameters.

Their results show that addressing this inefficiency allows for better performance with a smaller parameter count.

Similarly, in "Learning Localized Generative Models for 3D Point Clouds via Graph Convolution" Valsesia et al. (2019) augments their decoder by assuming a graph structure over the decoded point cloud and employing graph convolutions.

However, despite improved performance neither approach addresses the other issues that come with using MLPs to decode entire point clouds, namely the fixed-size output.

"Point Cloud GAN" and PointFlow (Yang et al., 2019 ) take a different approach to producing a point set in a generative setting.

Instead of learning a single mapping from any latent vector directly to its decoded point cloud, they learn a function parameterized by the latent vector which transforms low-dimensional Gaussian noise to a 3D point on the surface of the object described by the latent shape representation.

This sampling based approach is more in line with the semantics of point clouds.

First, an arbitrary number of points can be drawn from the Gaussian noise to produce a point cloud consisting of that number of points without requiring any changes to or retraining of the algorithm.

Second, every individual point is decoded independently and identically, which avoids the data inefficiency issues that come with using MLPs to process set data.

While this sampling approach has several desirable properties and appears promising, it's unclear whether the algorithm is applicable outside of the GAN settings these two papers inhabit, if they require specific bespoke loss functions to be trained effectively, or if they are capable of outperforming the baseline MLP approach according to other metrics.

A point cloud is a set of n 3D points C = {p 1 , . . .

, p N }, where each p i ??? R 3 .

In general, each p i may have additional auxiliary information associated with it via non-spatial dimensions.

While all of our architectures easily generalize to include such information, in this paper, we focus on point clouds that exclusively encode shapes without auxiliary information.

A point cloud auto-encoder takes a point cloud C with n points and outputs a point cloud?? with m points that is intended to represent the shape described by C. While often n = m, we are interested in the general case when n and m may be different, which corresponds to up-scaling or downscaling C. Each auto-encoder will be comprised of an encoder E(C), which takes an input point cloud and outputs a latent shape representation h in R l , and a decoder D(h) which maps a latent representation to an output point cloud of the appropriate size.

Thus, given an input point cloud C, the auto-encoder output is given by?? = D(E(C)).

In this paper, we focus on the Chamfer distance as the measure of auto-encoder quality.

Intuitively this loss function measures how well?? matches C in terms of the nearest neighbor in?? to each point in C and vice versa.

Specifically, if dist(p,??) gives the distance between point p and the nearest neighbor in point cloud??, our loss function is defined by

Since the focus of this paper is on point-cloud decoders, all of our architectures use the same pointcloud encoder architecture, while varying the decoder architecture.

Below, we first overview the common PointNet-style encoder used followed by a description of the four decoders considered in our experimental analysis, which include three sample-based decoders.

PointNet (Qi et al., 2017) handles unordered input by recognizing that a symmetric function g (such as element-wise max or sum) produces the same result regardless of the order of its inputs.

PointNet thus learns a single transformation function f that maps individual points to an l-dimensional representation and then combines those representations via g. That is, the latent encoding produced by PointNet for a point cloud C = {p 1 , . . .

, p n } is the l dimensional vector

As desired, E(C) is invariant to the ordering of points in C and applies to any number of points.

We learn an MLP representation of f , with input space R 3 , encoding points, and output space R l , encoding the latent representation or point feature.

We use max as the reduction function g to map the arbitrary number of resulting point features to a single fixed-size latent shape representation.

The hidden layers and size of the latent shape representation for each instantiation of this encoder architecture can be found in Table 1 .

Most prior work has used MLP decoders, which we consider here as a baseline approach.

An MLP decoder is a fully connected network that takes the latent shape representation as input and outputs an m ?? 3 output vector, which represents the m output points.

Accordingly, MLP decoders are parameterized by the number and size of their fully connected layers.

In our experiments, each fully connected layer consists of parameterized ReLU units with a batch normalization layer.

Our main focus is on sample-based decoders, which allow for an arbitrary number of outputs points to be produced for a latent shape representation.

In particular, given a latent shape representation h, each of our decoders is defined in terms of a point feature sampling distribution S(h), where the decoder produces a point-cloud output by sampling m point features from S(h).

Once we have a set of M independently sampled point features from our sampling distribution S(h) we need to transform each one into a triple representing that point's location.

Note that we are now in an identical but opposite situation as the point cloud encoder.

Whereas the encoder had to transform independent point samples of some underlying object into corresponding high-dimensional representations, our decoder now has to transform independently sampled high-dimensional point representations into a point in space on the surface of the target object.

Therefore, we can simply apply the same style of PointNet encoding mechanism with different input and output tensor sizes to implement an effective point feature decoder.

The sizes of the hidden layers in our decoder network can be seen in Table 1 .

By applying the shared MLP point decoder to each sampled point feature, we can directly decode point clouds of arbitrary size.

Below we describe three architectures for S, which are compared to each other and the baseline MLP decoder in our experiments.

NoiseAppend Decoder.

NoiseAppend is similar to the sampling approach described in "Point Cloud GAN" by .

They sample point features by simply sampling from a multivariate Gaussian distribution with zero mean and unit variance before appending the sampled noise to the latent shape vector.

That is, S(h) = concat (h, N (0, I)).

However, this requires us to decide how many elements of noise should be appended to the latent shape representation.

state that the size of the appended noise vector should be 'much smaller than' the size of the latent shape representation, but it's not clear how much noise is necessary to allow the decoder to fully represent the shape.

Ultimately this is an additional hyperparameter that needs to be investigated and tuned.

NoiseAdd Decoder.

NoiseAdd builds on the concept of adding unit Gaussian noise to the latent shape vector with the goal of avoiding the additional hyperparameter that NoiseAppend introduces.

This can be easily accomplished by treating the entire latent vector as the mean of a Gaussian distribution with unit variance.

That is, S(h) = N (h, I).

However, this violates the claim by that the amount of noise introduced to the resulting point feature samples should be much smaller than the size of the latent shape representation itself.

Therefore, it may be the case that uniformly adding noise to every element of the latent vector obscures the crucial information it represents.

NoiseLearn Decoder.

NoiseLearn attempts to instead learn a small separate function V (h) which predicts the log-variance of the desired point feature distribution.

Specifically, S(h) = N h, e V (h)/2 I .

We define V (h) as a small MLP, the size of which can be seen in Table 1 .

By allowing the network to choose the amount and location of noise to be added to the latent shape vector, we hope that it will learn both to add an appropriate amount of noise for the target shape while

Figure 2: Diagrams of the different approaches to deriving a distribution from the latent shape representation h. conserving the information necessary to accurately reconstruct it without introducing any additional hyperparameters.

We evaluated each decoding architecture by training several instantiations of each architecture on a point cloud auto-encoding problem derived from the ModelNet40 dataset, which consists of over 12,000 3D models of 40 different common object classes.

The dataset has a prescribed train/test split, with approximately 9800 models in the training dataset and 2500 in the test dataset.

We randomly select 10% of the training data to use for validation during training.

Before training, each object model in the ModelNet40 dataset is used to generate a uniformlysampled point cloud with 4096 points which is then scaled to fit within the unit sphere.

For all autoencoder network models, at each iteration of training, the point clouds are randomly downsampled to 1024 points before being used to update the network parameters.

The helps reduce the computational cost of training and also encouraging better generalization.

During training, each decoded point cloud consists of 1024 points.

We use the Chamfer distance as the loss function due to its relative speed and capability to directly compare point clouds of unequal sizes without modification.

Each network is trained for 100 epochs using the ADAM optimizer with an initial learning rate of 10 ???3 , where each epoch performs a parameter update on each training example.

The learning rate is decreased by a factor of 10 at epoch 50 and epoch 80.

We trained five instantiations of each of the four network architectures with each instantiation varying the number of parameters as shown in Table 1 (note that we were not able to scale down the MLP for the smallest parameter setting).

For each instantiation we ran the entire training process 15 times and all results show average performance across the 15 runs.

All code and infrastructure for "push-button" replication of our experiments open-source (Github/Gitlab location removed for anonymous review -code will be privately made available to reviewers through a comment approximately a week after submission).

Quantitative Results.

Figure 3 shows the validation loss along the learning curves for the 2M parameter instantiation of each architecture.

The relative ordering of the architectures is consistent after the initial phase of training, with all curves flattening out by 100 epochs.

First, the large jumps in the MLP training (due to unstable training runs) show that it was much less stable to training compared to the sample-based architectures.

While effort was spent trying to specialize the training configuration for the MLP, stability remained an issue.

1 In contrast the runs for each sample based architecture were stable and similar.

Ignoring the MLP stability, it performs similarly to worst performing sample-based architectureby the end of the learning curve.

The three sample based architectures show rapid progress early in learning and then settle into a consistent ordering with NoiseLearn performing best, followed by NoiseAppend, and then NoiseAdd.

This suggests that NoiseAdd's approach of adding uniform noise to the latent representation may be obscuring information needed for accurate reconstruction, compared to NoiseAppend, which separates noise from the shape representation.

On the other hand, we see that while NoiseLearn also adds noise to the latent representation, it is able to outperform NoiseAppend.

This indicates the importance of being able to intelligently select how much noise to add to different components of the representation.

Apparently, this allows NoiseLearn to avoid obscuring critical information in the latent representation needed for accurate reconstruction.

Figure 4 shows the average test set performance after 100 epochs of training of each size instantiation of the four architectures (note the log scale).

The appendix also shows more detailed results broken down for each of 5 selected object classes.

The three sample based architectures show relatively consistent improvement in performance as the sizes grow by orders of magnitude.

Rather, the MLP shows initial improvement, but then performance decreases significantly past 100K parameters.

We looked further into the behavior of the MLP architecture for the larger parameter sets.

We observed that the larger MLPs showed a similar decrease in performance on even the training data, indicating that the problem is not necessarily overfitting but also difficulty of the optimization.

It is possible that with substantially more epochs the MLP performance would improve, but at great cost.

This indicates that the MLP architecture is much less efficient at exploiting larger network sizes than the more structured sample-based architectures.

It is possible that the architecture and training hyperparameters could be tweaked to improve the large MLP networks' performance, such as by adding additional regularization via weight decay or other mechanisms.

However, we consider this tweaking to be outside the scope of this work, and note that none of the sampling based architectures required any such tweaking to achieve competitive performance at all parameter counts.

NoiseParam-2m NoiseAdd-2m NoiseAppend-2m

Figure 5: Examples of networks' auto-encoding results on several previously unseen objects.

Overall the results give strong evidence that the sample-based architectures encode a bias that is much better matched to decoding the class of shapes in ModelNet40 compared to MLPs.

The structured sample-based architectures, compared to the MLP, result in more stable learning and the ability to continually improve as the architectures grow in size.

Further, we see that the NoiseLearn architecture, which avoids the need to specify hyperparameters to control the amount of noise performs the best, or near best, across all network sizes and number of epochs.

Illustrative Qualitative Results.

Figure 5 shows each network's performance three test objects not seen in the training data.

The point cloud decoded by the MLP network appears to be more evenly distributed spatially, while the sampling-based approaches are better able to capture finer detail in the target shape, such as the stool's thin legs and crossbars.

Among the sample-based approaches, no single approach is clearly dominant in terms of visual quality across the objects.

It is interesting that all of the sample-based architectures tend to miss the same type of object details, e.g. the jets on the plane or the leg cross bars on the chair, which may be due to limitations of the PointNet encoders sized and/or architecture.

Nevertheless, it is quite interesting that a single relatively small latent vector representation is able to encode the level of detail exibited in these results.

Each sampling architecture defines a function from the latent shape representation to a point feature distribution.

The underlying latent representation inherently defines the manifold of the encoded shape.

Rather, the injected noise (either via appending or addition) can be viewed as playing the role of indexing locations on the manifold for the generated point.

Effectively, the primary difference between the sample-based architectures is how they use the noise to index locations and traverse the manifolds.

Below we aim to better understand this relationship between noise and spatial indexing and how the architectures differ in that respect.

In Figure 6 we demonstrate how each architecture uses noise by controlling the variance introduced to a trained network in two different ways.

To examine how the decoder's output is influenced by individual elements of noise we show the output of these networks when all but one of the noise elements is held at a constant near-zero value.

In the lower plots, we show the decoder's behavior when it only receives the union of the noise elements above.

This demonstrates both how the network learns to exploit individual elements of noise and how the decoder combines those elements to produce a point cloud that spans the entire shape.

For NoiseAppend all of the noise is of equal magnitude, so we just examine the first five elements of noise in its noise vector.

NoiseLearn predicts individual variances for each element in the dense shape encoding, enabling us to select the five elements of noise with the highest variance, and therefore presumably the biggest contribution to the decoded point cloud.

The appendix contains additional examples of noise manipulation.

The plots shown in Figure 6 give us some insight into how the networks use noise to complete the decoded shape.

Each individual element of noise appears to correspond to a learned path along the surface of the learned shape.

The final point cloud then seems to be produced by 'extruding' along those paths.

NoiseLearn's use of only four significant elements of noise suggests that in this domain only three or four elements of noise is sufficient to achieve good coverage of the target shape.

Figure 8 shows how individual noise channels change when the NoiseAppend architecture is modified to only append one, two, and three noise elements.

With only one element of noise, we can see that the network effectively has to learn a single path that spans as much of the target shape as possible.

With two elements of noise, the network instead seems to learn individual 'loops' around the object which are transformed and rotated as necessary.

Once the network has access to three elements of noise, we see the same behavior as the functional networks of learning small paths on the object's surface.

If too little noise can seriously hurt NoiseLearn's performance, does adding too much noise do the same?

Figure 7 shows the NoiseAppend architecture trained with different amounts of added noise to see if the same performance dropoff is present at both extremes.

It appears that even when the noise vector is much larger than the dense shape representation, the decoder's overall performance is not impacted.

However, note that adding large amounts of noise does significantly increase the parameter count, so there is a nontrivial cost to doing this.

In this work, we evaluated and compared several realizations of a sample-based point cloud decoder architecture.

We show that these sampling approaches are competitive with or outperform the MLP approach while using fewer parameters and providing better functionality.

These advantages over the baseline suggest that sample based point cloud decoders should be the default approach when a network needs to produce independent point samples of some underlying function or object.

To further this this area of research, we provide a complete open-source implementation of our tools used to train and evaluate these networks.

The tables below show each architecture's average loss on each individual class in the ModelNet40 dataset.

The best-performing network is bolded for each object class.

<|TLDR|>

@highlight

We present and evaluate sampling-based point cloud decoders that outperform the baseline MLP approach by better matching the semantics of point clouds.
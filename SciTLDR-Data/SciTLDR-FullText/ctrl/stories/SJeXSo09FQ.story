Point clouds are an important type of geometric data and have widespread use in computer graphics and vision.

However, learning representations for point clouds is particularly challenging due to their nature as being an unordered collection of points irregularly distributed in 3D space.

Graph convolution, a generalization of the convolution operation for data defined over graphs, has been recently shown to be very successful at extracting localized features from point clouds in supervised or semi-supervised tasks such as classification or segmentation.

This paper studies the unsupervised problem of a generative model exploiting graph convolution.

We focus on the generator of a GAN and define methods for graph convolution when the graph is not known in advance as it is the very output of the generator.

The proposed architecture learns to generate localized features that approximate graph embeddings of the output geometry.

We also study the problem of defining an upsampling layer in the graph-convolutional generator, such that it learns to exploit a self-similarity prior on the data distribution to sample more effectively.

Convolutional neural networks are at the core of highly successful models in image generation and understanding.

This success is due to the ability of the convolution operation to exploit the principles of locality, stationarity and compositionality that hold true for many data of interest.

In particular, feature locality and weight sharing across the data domain greatly reduce the number of parameters in the model, simplifying training and countering overfitting.

However, while images are defined on an underlying regular grid structure, several other types of data naturally lie on irregular or nonEuclidean domains .

Examples include problems in 3D models BID3 BID17 , computational biology BID1 BID7 or social network graphs BID14 .

Defining convolutional architectures on these domains is key to exploit useful priors on the data to obtain more powerful representations.

Graph convolution is emerging as one of the most successful approaches to deal with data where the irregular domain can be represented as a graph.

In this case, the data are defined as vectors on the nodes of a graph.

Defining a convolution-like operation for this kind of data is not trivial, as even simple notions such as shifts are undefined.

The literature has identified two main approaches to define graph convolution, namely spectral or spatial.

In the former case BID13 BID6 BID14 , the convolution operator is defined in the spectral domain through the graph Fourier transform BID24 .

Fast polynomial approximations BID6 exist that allow an efficient implementation of the operation.

This spectral approach has been successfully used in semi-supervised classification BID14 and link prediction BID23 .

However, the main drawback of these techniques is that the structure of the graph is supposed to be fixed and it is not clear how to handle the case where the graph structure varies.

The latter class of methods BID25 BID26 defines the convolution operator using a spatial approach by means of local aggregations, i.e., weighted combinations of the vectors restricted to a neighborhood.

Since this kind of convolution is defined at a neighborhood level, the operation remains well defined even when the graph varies.

Point clouds are a challenging data type due to the irregular positioning of the points and the fact that a point cloud is an unordered set of points, and therefore any permutation of its members, while changing the representation, does not change its semantic meaning.

Some works have addressed supervised problems on point clouds such as classification or segmentation, either through voxelization BID19 BID27 , where the irregular point structure is approximated with a regular 3D grid, or by networks like PointNet BID21 b) that address the problem of permutation invariance by processing each point identically and independently before applying a globally symmetric operation.

The most recent approaches BID25 BID26 build graphs in the Euclidean space of the point cloud and use graph convolution operations.

This approach has shown multiple advantages in i) reducing the degrees of freedom in the learned models by enforcing some kind of weight sharing, ii) extracting localized features that successfully capture dependencies among neighboring points.

Generative models are powerful tools in unsupervised learning aiming at capturing the data distribution.

However, so far little work has been done on generative models for point clouds.

Generative models of point clouds can be useful for many tasks that range from data augmentation to shape completion or inpainting partial data thanks to the features learned by the model.

Generative Adversarial Networks (GANs) have been shown on images to provide better approximations of the data distribution than variational autoencoders (VAEs) BID15 , being able to generate sharper images and to capture semantic properties in their latent space.

For this reason, it is interesting to study them for unordered point sets.

In the first work on the topic, BID0 studied some GAN architectures to generate point clouds.

Such architectures use the PointNet approach to deal with the permutation problem at the discriminator and employ a dense generator.

However, this means that they are unable to learn localized features or exploit weight sharing.

This paper studies a generative model for point clouds based on graph convolution.

In particular, we focus on the GAN generator which is not well explored by the graph convolution literature.

This poses a unique challenge: how can one apply a localized operation (the graph convolution) without knowing the domain (the graph) in advance because it is the very output of the generator?

We show that the proposed architecture learns domain and features simultaneously and promotes the features to be graph embeddings, i.e. representations in a vector space of the local dependencies between a point and its neighbors.

Such localized features learned by the generator provide a flexible and descriptive model.

Moreover, we address the problem of upsampling at the generator.

While downsampling based on graph coarsening is a staple in (semi-)supervised problems using graph convolution, it is not obvious how to properly upsample the intermediate layers of a graph-convolutional GAN generator.

We propose a method exploiting non-local self-similarities in the data distribution.

2.1 GRAPH-CONVOLUTIONAL GAN GANs BID9 are state-of-the-art generative models composed of a generator and a discriminator network.

The generator learns a function mapping a latent vector z to a sample x from the data distribution.

In the original formulation, the discriminator worked as a classifier trained to separate real samples from generated ones.

Recently, the Wasserstein GAN addressed the instability and mode collapse issues of the original formulation by modifying the loss function to be a dual formulation of an optimal transport problem using the Wasserstein metric: DISPLAYFORM0 with a discriminator D and a generator G. In this paper, we use the Wasserstein GAN with the gradient penalty method BID12 to enforce the Lipschitz constraint at the discriminator.

In the proposed generative model, we use the Edge-Conditioned Convolution BID25 which falls under the category of spatial approaches to graph convolution and is suitable for dealing with multiple arbitrary graphs.

Given a layer l with N

The focus of this paper is to design a GAN generator that uses localized operations in the form of graphs convolutions.

Notice that such operations are able to deal with data in the form of unordered sets, such as points clouds, because they are by design invariant to permutations.

However, there are some issues peculiar to the generative problem to be addressed.

First, while in supervised problems BID25 BID26 or in unsupervised settings involving autoencoders BID28 ) the point cloud is known in advance, the intermediate layers of the GAN generator do not know it in advance as it is the very result of the generation operation.

It is therefore not obvious how to define an operation that is localized to neighborhoods of a graph that is not known in advance.

The solution to this problem is to exploit the pairwise distances (�h DISPLAYFORM0 between node features of the preceding layer to build a k-nearest neighbor graph.

FIG0 shows a block diagram of a graph-based generator where each graph convolution block uses the graph constructed from the input features of the block itself.

The intuition behind this solution is that this promotes the features to become graph embeddings, i.e. representations in a high-dimensional metric space of relationships between points.

Going through the generator network from the latent space towards the point cloud output, these embeddings are assembled hierarchically and their associated graphs represent better and better approximations of the graph of the output point cloud.

According to the definition of graph convolution in (2), the new features of a node are a weighted combination of the features of the node itself and of the neighbors as determined by the graph construction.

Notice that this localized approach differs from the one in BID0 where the generator of the r-GAN model is a fully-connected network, therefore unable to provide any localized interpretation of its hidden layers.

It also differs from the PointNet BID21 and PointNet++ (Qi et al., 2017b) architectures.

PointNet processes each point independently with the same weights and then aggregates them using a globally symmetric operation to deal with the permutation invariance problem.

PointNet++ extends this work using some localized operations.

However, the key difference with the work in this paper is that PointNet and PointNet++ are not generative models, but are used in supervised problems such as classification or segmentation.

Other works explore likelihood-based generative models, typically in the form of variational autoencoders BID8 BID20 BID16 .

The most similar approach to the method of this paper is the one in BID11 , with the key difference being that a distribution over adjacency matrices of graphs is learned using a spectral graph-convolutional VAE.

DISPLAYFORM1

The previous section presented the basic outline of a graph-based generator in a GAN.

However, one evident shortcoming is the fixed number of points throughout the generator, which is determined by the number of output points.

Many data of interest typically display some kind of regularity in the form of multi-resolution or other kinds of compositionality whereby points can be predicted from a smaller number of neighboring points.

In the case of 2D images, lower resolutions provide a prediction of higher resolutions by supplying the low-frequency content and the upsampling operation is straightforward.

In fact, convolutional GANs for image generation are composed of a sequence of upsampling and convolutional layers.

Extending upsampling to deal with the generation of sets of points without a total ordering is not a trivial task.

Many works have addressed the problem of upsampling 3D point clouds, e.g., by creating grids in the 3D space BID18 .

Notice, however, that introducing upsampling to interleave the graph-convolutional layers outlined in the previous section is a more complex problem because the high dimensionality of the feature vectors makes the gridding approach unfeasible.

If we consider the l-th generator layer, we want to define an upsampling operation that, starting from the graph convolution output DISPLAYFORM0 l .

Then, these new feature vectors are concatenated to H l in order to obtain the output H l,up ∈ R 2N l ×d l .

We propose to define an upsampling operation using local aggregations.

In this case, the upsampling operation becomes similar to a graph convolution.

Given a feature vector h It is important to note that, differently from the graph convolution described in 2.1 where Θ l,ij andW l are dense matrices, in this case we use diagonal matrices.

This means that during the upsampling operation the local aggregation treats each feature independently.

This also reduces the number of parameters.

DISPLAYFORM1

Graph embeddings BID10 are representations of graphs in a vector space where a feature vector is associated to each node of the graph.

For what concerns this paper we consider the following definition of graph embedding, focused on predicting edges from the feature vectors.

Definition 1 Given a graph G = (V, E), a graph embedding is a mapping f : i → h i ∈ R d , ∀i ∈ V, such that d � |V| and the function f is defined such that if we consider two pairs of nodes (i, j) and (i, k) where DISPLAYFORM0 The graph-convolutional generator presented in this paper can be interpreted as generating graph embeddings of the nearest-neighbor graph of the output point cloud at each hidden layer, thus creating features that are able to capture some properties of the local topology.

In order to see why this is the case, we analyze the architecture in FIG0 backwards from the output to the input.

The final output x is the result of a graph convolution aggregating features localized to the nearest-neighbor graph computed from the features of the preceding layer.

Since the GAN objective is to match the distribution of the output with that of real data, the neighborhoods identified by the last graph must be a good approximation of the neighborhoods in the true data.

Therefore, we say that features H L are a graph embedding in the sense that they allow to predict the edges of the output graph from their pairwise distances.

Proceeding backwards, there is a hierarchy of graph embeddings as the other graphs are constructed from higher-order features.

Notice that the upsampling operation in the architecture of FIG1 affects this chain of embeddings by introducing new points.

While the graph convolution operation promotes the features of all the points after upsampling to be graph embeddings, the upsampling operation affects which points are generated.

In the experiments we show that the upsampling method approximately maintains the neighborhood shape but copies it elsewhere in the point cloud.

This suggests a generation mechanism exploiting self-similarities between the features of the point cloud at different locations.

We tested the proposed architecture by using three classes of point clouds taken from the ShapeNet repository (Chang et al., 2015): "chair", "airplane" and "sofa".

A class-specific model is trained for the desired class of point clouds.

Since the focus of this paper is the features learned by the generator, the architecture for the discriminator is the same as the one of the r-GAN in BID0 , with 4 layers with weights shared across points (number of output features: 64, 128, 256, 512) followed by a global maxpool and by 3 dense layers.

The generator architecture is reported in TAB0 .

The graph is built by selecting the 20 nearest neighbors in terms of Euclidean distance in the feature space.

We use Leaky ReLUs as nonlinearities and RMSProp as optimization method with a learning rate equal to 10 −4for both generator and discriminator.

Batch normalization follows every graph convolution.

The gradient penalty parameter of the WGAN is 1 and the discriminator is optimized for 5 iterations for each generator step.

The models have been trained for 1000 epochs.

For the "chair" class this required about 5 days without upsampling and 4 days with upsampling.

In this section we perform qualitative and quantitative comparisons with the generated point clouds.

We first visually inspect the generated point clouds from the classes "chair" and "airplane", as shown in Fig. 3 .

The results are convincing from a visual standpoint and the variety of the generated objects is high, suggesting no mode collapse in the training process.

The distribution of points on the object is quite uniform, especially for the method with upsampling.

To the best of our knowledge this is the first work addressing GANs for point clouds learning localized features.

We compare the proposed GAN for point cloud generation with other GANs able to (2017) has a dense generator, which is unable to generate localized representations because there is no mapping between points and feature vectors.

As an additional baseline variant, dubbed "r-GANconv", we study the use of a generator having as many feature vectors as the points in the point cloud and using a size-1 convolution across the points.

Notice that the graph convolution we use can be seen as a generalization of this model, aggregating the features of neighboring points instead of processing each point independently.

We point out that we cannot compare the proposed method in a fair way with the variational autoencoders mentioned in Sec. 2.1: BID8 generate point clouds conditioned on an input image; BID20 use object segmentation labels to generate point clouds by parts; BID16 focus on generating vertices on meshes with a fixed and given topology.

In order to perform a quantitative evaluation of the generated point clouds we use the evaluation metrics proposed in BID0 , employing three different metrics to compare a set of generated samples with the test set.

The first one is the Jensen-Shannon divergence (JSD) between marginal distributions defined in the 3D space.

Then, we also evaluate the coverage (COV) and the minimum matching distance (MMD), as defined in BID0 , using two different point-set distances, the earth mover's distance (EMD) and the Chamfer distance (CD).

TAB1 shows the obtained results.

As can be seen, the proposed methods achieve better values for the metrics under consideration.

In particular, the method with upsampling operations is consistently the better.

Notice that BID0 report that the Chamfer distance is often unreliable as it fails to penalize non-uniform distributions of points.

FIG4 visually shows that the proposed methods generate point clouds with better-distributed points, confirming the quantitative results.

In particular, the r-GAN-dense shows clusters of points, while the r-GAN-conv also exhibits noisy shapes.

In this section we quantitatively study the properties of the features in the layers of the generator.

Referring to TAB0 , the output of each layer is a matrix where every point is associated to a feature vector.

In Sec. 2.3 we claimed that these features learned by the generator are graph embeddings.

We tested this hypothesis by measuring how much the adjacency matrix of the final point cloud, constructed as a nearest-neighbor graph in 3D, is successfully predicted by the nearest-neighbor adjacency matrix computed from hidden features.

This is shown in FIG6 which reports the percentage of edges correctly predicted as function of the number of neighbors considered for the graph of the output point cloud and a fixed number of 20 neighbors in the feature space.

Notice that layers closer to the output correctly predict a higher percentage of edges and in this sense are better graph embeddings of the output geometry.

FIG5 shows another experiment concerning localization of features.

We applied k-means with 6 clusters to the features of intermediate layers and represented the cluster assignments onto the final point cloud.

This experiment confirms that the features are highly localized and progressively more so in the layers closer to the output.

We further investigated the effective receptive field of the convolution operation in FIG8 .

This figure reports histograms of Euclidean distances measured on the output point cloud between neighbors as determined by the nearest neighbor graph in one of the intermediate layers.

We can see that layers closer to the output aggregate points which are very close in the final point cloud, thus implementing a highly localized operation.

Conversely, layers close to the latent space perform more global operations.

The main drawback of the model without upsampling is the unnecessarily large number of parameters in the first dense layer.

This is solved by the introduction of the upsampling layers which aim at exploiting hierarchical priors to lower the number of parameters by starting with a lower number of points and progressively predicting new points from the generated features.

The proposed upsampling technique based on local aggregations computes a new point as a weighted aggregation of neighboring points.

The weights of the aggregation are learned by the network, thus letting the network decide the best method to create a new point from a neighborhood, at the expense of an increased number of total parameters.

The experiment in Figs. 7b and 8 shows an interesting behavior.

First, the generated points are not close to the original point: FIG8 shows the ratio between the generator-generated distance and the average neighborhood distance (neighborhoods are defined in the feature space, while distances are measured as Euclidean distances on the output 3D point cloud) and since it is usually significantly larger than 1, we can conclude that the generated point is far from the original generating neighborhood.

Then, the clusters in FIG9 show that the points in the first layers are not uniformly distributed over the point cloud, but rather form parts of it.

The mechanism learned by the network to generate new points is essentially to apply some mild transformation to a neighborhood and copy it in a different area of the feature space.

The generated points will no longer be close to their generators, but the structure of the neighborhood resembles the one of the generating neighborhood.

This notion is similar to the second-order proximity in the graph embedding literature BID10 and it seems that this operation is exploiting the inherent self-similarities between the data features at distant points.

To validate this hypothesis we measured two relevant quantities.

First, we considered a point i, its neighbors N TAB2 .

The result shows that the neighborhood of a generated point is almost entirely generated by the points that were neighbors of the generator, and that the new points are not neighbors of the original ones.

This behavior is consistent over different layers.

Then, we measured the Euclidean distances in the feature space between point i and its neighbors N .

TAB2 reports the correlation coefficient between those distance vectors, which suggests that the shape of the neighborhood is fairly conserved.

We presented a GAN using graph convolutional layers to generate 3D point clouds.

In particular, we showed how constructing nearest neighbor graphs from generator features to implement the graph convolution operation promotes the features to be localized and to approximate a graph embedding of the output geometry.

We also proposed an upsampling scheme for the generator that exploits self-similarities in the samples to be generated.

The main drawback of the current method is the rather high complexity of the graph convolution operation.

Future work will focus on reducing the overall complexity, e.g., in the graph construction operation, and study new upsampling schemes.

<|TLDR|>

@highlight

A GAN using graph convolution operations with dynamically computed graphs from hidden features

@highlight

The paper proposes a version of GANs specifically designed for generating point clouds with the core contribution of the work the upsampling operation.

@highlight

This paper proposes graph-convolutional GANs for irregular 3D point clouds that learn domain and features at the same time.
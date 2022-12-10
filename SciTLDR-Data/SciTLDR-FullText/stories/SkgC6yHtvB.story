Computer vision tasks such as image classification, image retrieval and few-shot learning are currently dominated by Euclidean and spherical embeddings, so that the final decisions about class belongings or the degree of similarity are made using linear hyperplanes, Euclidean distances, or spherical geodesic distances (cosine similarity).

In this work, we demonstrate that in many practical scenarios hyperbolic embeddings provide a better alternative.

Figure 1: An example of two-dimensional Poincaré embeddings computed by a hyperbolic neural network trained on MNIST, and evaluated additionally on Omniglot.

Ambiguous and unclear images from MNIST, as well as most of the images from Omniglot are embedded near the center, while samples with clear class labels (or characters from Omniglot similar to one of the digits) lie near the boundary.

High-dimensional embeddings are ubiquitous in modern computer vision.

Many, perhaps most, modern computer vision systems learn non-linear mappings (in the form of deep convolutional networks) from the space of images or image fragments into high-dimensional spaces.

The operations at the end of deep networks imply a certain type of geometry of the embedding spaces.

For example, image classification networks (Krizhevsky et al., 2012; LeCun et al., 1989) use linear operators (matrix multiplication) to map embeddings in the penultimate layer to class logits.

The class boundaries in the embedding space are thus piecewise-linear, and pairs of classes are separated by Euclidean hyperplanes.

The embeddings learned by the model in the penultimate layer, therefore, live in the Euclidean space.

The same can be said about systems where Euclidean distances are used to perform image retrieval (Oh Song et al., 2016; Sohn, 2016; Wu et al., 2017) , face recognition (Parkhi et al., 2015; Wen et al., 2016) or one-shot learning (Snell et al., 2017) .

Alternatively, some few-shot learning (Vinyals et al., 2016) , face recognition (Schroff et al., 2015) and person re-identification methods (Ustinova & Lempitsky, 2016; Yi et al., 2014) learn spherical embeddings, so that sphere projection operator is applied at the end of a network that computes the embeddings.

Cosine similarity (closely associated with sphere geodesic distance) is then used by such architectures to match images.

Euclidean spaces with their zero curvature and spherical spaces with their positive curvature have certain profound implications on the nature of embeddings that existing computer vision systems can learn.

In this work, we argue that hyperbolic spaces with negative curvature might often be more appropriate for learning embedding of images.

Towards this end, we add the recently-proposed hyperbolic network layers to the end of several computer vision networks, and present a number of experiments corresponding to image classification, one-shot, and few-shot learning and person re-identification.

We show that in many cases, the use of hyperbolic geometry improves the performance over Euclidean or spherical embeddings.

Motivation for hyperbolic image embeddings.

The use of hyperbolic spaces in natural language processing (Nickel & Kiela, 2017; Tifrea et al., 2018; Dhingra et al., 2018 ) is motivated by their natural ability to embed hierarchies (e.g., tree graphs) with low distortion (Sarkar, 2011) .

Hierarchies are ubiquitous in natural language processing.

First, there are natural hierarchies corresponding to, e.g., biological taxonomies and linguistic ontologies.

Likewise, a more generic short phrase can have many plausible continuations and is therefore semantically-related to a multitude of long phrases that are not necessarily closely related to each other (in the semantic sense).

The innate suitability of hyperbolic spaces to embedding hierarchies (Sala et al., 2018a; Sarkar, 2011) explains the success of such spaces in natural language processing (Nickel & Kiela, 2017) .

Here, we argue that similar hierarchical relations between images are common in computer vision tasks (Figure 2 ).

One can observe the following example cases:

• In image retrieval, an overview photograph is related to many images that correspond to the close-ups of different distinct details.

Likewise, for classification tasks in-the-wild, an image containing the representatives of multiple classes is related to images that contain representatives of the classes in isolation.

Embedding a dataset that contains composite images into continuous space is therefore similar to embedding a hierarchy.

•

In some tasks, more generic images may correspond to images that contain less information and are therefore more ambiguous.

E.g., in face recognition, a blurry and/or low-resolution face image taken from afar can be related to many high-resolution images of faces that clearly belong to distinct people.

Again natural embeddings for image datasets that have widely varying image quality/ambiguity calls for retaining such hierarchical structure.

In order to build deep learning models which operate on the embeddings to hyperbolic spaces, we capitalize on recent developments , which construct the analogues of familiar layers (such as a feed-forward layer, or a multinomial regression layer) in hyperbolic spaces.

We show that many standard architectures used for tasks of image classification, and in particular in the few-shot learning setting can be easily modified to operate on hyperbolic embeddings, which in many cases also leads to their improvement.

Formally, n-dimensional hyperbolic space denoted as H n is defined as the homogeneous, simply connected n-dimensional Riemannian manifold of constant negative sectional curvature.

The property of constant negative curvature makes it analogous to the ordinary Euclidean sphere (which has constant positive curvature), however, the geometrical properties of the hyperbolic space are very different.

It is known that hyperbolic space cannot be isometrically embedded into Euclidean space (Krioukov et al., 2010; Linial et al., 1998) , but there exist several well-studied models of hyperbolic geometry.

In every model a certain subset of Euclidean space is endowed with a hyperbolic metric, however, all these models are isomorphic to each other and we may easily move from one to another base on where the formulas of interest are easier.

We follow the majority of NLP works and use the Poincaré ball model.

Investigating the alternative models that might provide better numerical stability remain future work (though already started in the NLP community (Nickel & Kiela, 2018; Sala et al., 2018b) ).

Here, we provide a very short summary of the model.

1− x 2 is the conformal factor and g E is the Euclidean metric tensor g E = I n .

In this model the geodesic distance between two points is given by the following expression:

In order to define the hyperbolic average, we will make use of the Klein model of hyperbolic space.

Similarly to the Poincaré model, it is defined on the set K n = {x ∈ R n : x < 1}, however, with a different metric, not relevant for further discussion.

In Klein coordinates, the hyperbolic average (generalizing the usual Euclidean mean) takes the most simple form, and we present the necessary formulas in Section 4.

From the viewpoint of hyperbolic geometry, all points of Poincaré ball are equivalent.

The models that we consider below are, however, hybrid in the sense that most layers use Euclidean operators, such as standard generalized convolutions, while only the final layers operate within the hyperbolic geometry framework.

The hybrid nature of our setups makes the origin a special point, since from the Euclidean viewpoint the local volumes in Poincare ball expand exponentially from the origin to the boundary.

This leads to the useful tendency of the learned embeddings to place more generic/ambiguous objects closer to the origin, while moving more specific objects towards the boundary.

The distance to the origin in our models therefore provides a natural estimate of uncertainty, that can be used in several ways, as we show below.

Hyperbolic language embeddings Hyperbolic embeddings in the natural language processing field have recently been very successful (Nickel & Kiela, 2017; .

They are motivated by the innate ability of hyperbolic spaces to embed hierarchies (e.g., tree graphs) with low distortion (Sala et al., 2018b; Sarkar, 2011) .

The main result in this area states that any tree can be embedded into (two dimensional) hyperbolic space with arbitrarily low distortion.

Another direction of research, more relevant to the present work is based on imposing hyperbolic structure on activations of neural networks Gulcehre et al., 2019) .

The task of few-shot learning, which has recently attracted a lot of attention, is concerned with the overall ability of the model to generalize to unseen data during training.

A body of papers devoted to few-shot classification that focuses on metric learning methods includes Siamese Networks (Koch et al., 2015) , Matching Networks (Vinyals et al., 2016) , Prototypical Networks (Snell et al., 2017) , Relation Networks (Sung et al., 2018) .

In contrast, other models apply meta-learning to few-shot learning: e.g., MAML by (Finn et al., 2017) , Meta-Learner LSTM by (Ravi & Larochelle, 2016) , SNAIL by (Mishra et al., 2018) .

While these methods employ either Euclidean or spherical geometries (like in (Vinyals et al., 2016) ), there is no model extension to hyperbolic space.

Person re-identification The task of person re-identification is to match pedestrian images captured by possibly non-overlapping surveillance cameras.

Papers (Ahmed et al., 2015; Guo & Cheung, 2018; adopt the pairwise models that accept pairs of images and output their similarity scores.

The resulting similarity scores are used to classify the input pairs as being matching or non-matching.

Another popular direction of work includes approaches that aim at learning a mapping of the pedestrian images to the Euclidean descriptor space.

Several papers, e.g., (Suh et al., 2018; Yi et al., 2014) use verification loss functions based on the Euclidean distance or cosine similarity.

A number of methods utilize a simple classification approach for training (Chang et al., 2018; Su et al., 2017; Kalayeh et al., 2018; Zhao et al., 2017) , and Euclidean distance is used in test time.

In our work we strongly rely on the apparatus of hyperbolic neural networks developed in .

Hyperbolic networks are extensions of conventional neural networks in a sense that they generalize typical neural network operations to those in hyperbolic space using the formalism of Möbius gyrovector spaces.

In this paper, the authors present the hyperbolic versions of feedforward networks, multinomial logistic regression, and recurrent neural networks.

In Appendix A we discuss the hyperbolic functions and layers used in hyperbolic neural networks.

Similarly to the paper , we use an additional hyperparameter c corresponding to the radius of the Poincaré ball, which is then defined in the following manner: D n c = {x ∈ R n : c x 2 < 1, c ≥ 0}. The corresponding conformal factor is then modified as λ c x = 2 1−c x 2 .

In practice, the choice of c allows one to balance between hyperbolic and Euclidean geometries, which is made precise by noting that with c → 0 all the formulas discussed below take their usual Euclidean form.

Hyperbolic averaging One important operation common in image processing is averaging of feature vectors, used, e.g., in prototypical networks for few-shot learning (Snell et al., 2017) .

In the Euclidean setting this operation takes the form (x 1 , . . .

, x N ) →

1 N i x i .

Extension of this operation to hyperbolic spaces is called the Einstein midpoint and takes the most simple form in Klein coordinates:

where

are the Lorentz factors.

Recall from the discussion in Section 2 that the Klein model is supported on the same space as the Poincaré ball, however the same point has different coordinate representations in these models.

Let x D and x K denote the coordinates of the same point in the Poincaré and Klein models correspondingly.

Then the following transition formulas hold.

Thus, given points in the Poincaré ball we can first map them to the Klein model, compute the average using Equation (2), and then move it back to the Poincaré model.

Practical aspects of implementation While implementing most of the formulas described above is straightforward, we employ some tricks to make the training more stable.

• To ensure numerical stability we perform clipping by norm after applying the exponential map, which constrains the norm to not exceed

(1 − 10 −3 ).

• Some of the parameters in the aforementioned layers are naturally elements of D c n .

While in principle it is possible to apply Riemannian optimization techniques to them (e.g., previously proposed Riemannian Adam optimizer (Becigneul & Ganea, 2019) ), we did not observe any significant improvement.

Instead, we parametrized them via ordinary Euclidean parameters which were mapped to their hyperbolic counterparts with the exponential map and used the standard Adam optimizer.

Gromov's δ-hyperbolicity A necessary parameter for embedding to Poincaré disk is its radius.

In hyperbolic neural networks, one has a curvature parameter c, which is inversed squared disk radius:

.

For the Euclidean case, i.e., c = 0, the corresponding radius would be equal to infinity.

The disk radius is closely related to the notion of Gromov's δ-hyperbolicity (Gromov, 1987) , as we will show later in this section.

Intuitively, this δ value shows 'how hyperbolic is a metric space'.

For example, for graphs, δ represents how 'far' the graph is from a tree, which is known to be hyperbolic (Fournier et al., 2015) .

Hence, we can compute the corresponding δ-hyperbolicity value to find the right Poincaré disk radius for an accurate embedding.

Formally, δ-hyperbolicity is defined as follows; we emphasize that this notion is defined for any metric space (X, d).

First, we need to define Gromov product for points x, y, z ∈ X:

Then, the δ is the minimal value such that the following four-point condition holds for all points x, y, z, w ∈ X:

In practice, it suffice to find the δ for some fixed point w 0 .

A more computational friendly way to define δ is presented in (Fournier et al., 2015) .

Having a set of points, we first compute the matrix A of pairwise Gromov products (5).

After that, the δ value is simply the largest coefficient in the matrix (A ⊗ A) − A, where ⊗ denotes the min-max matrix product

Relation between δ-hyperbolicity and Poincaré disk radius It is known (Tifrea et al., 2018 ) that the standard Poincaré ball is δ-hyperbolic with δ P = log(1 + √ 2) ∼ 0.88.

Using this constant we can estimate the radius of Poincaré disk suitable for an embedding of a specific dataset.

Suppose that for some dataset X we have found that its natural Gromov's δ is equal to δ X .

Then we can estimate c(X) as follows.

Estimating hyperbolicity of a dataset In order to verify our hypothesis on hyperbolicity of visual datasets we compute the scale-invariant metric, defined as δ rel (X) = 2δ(X) diam(X) , where diam(X) denotes the set diameter (Borassi et al., 2015) .

By construction, δ rel (X) ∈ [0, 1] and specifies how close is the dataset to a perfect hyperbolic space.

For instance, trees which are discrete analogues of a hyperbolic space (under the natural shortest path metric) have δ rel equal to 0.

We computed δ rel for various datasets we used for experiments.

As a natural distance between images we used the standard Euclidean distance between the features extracted with VGG16 (Simonyan & Zisserman, 2014) .

Our results are summarized in Table 1 .

We observe that degree of hyperbolicity in image datasets is quite high, as the obtained δ rel are significantly closer to 0 than to 1 (which corresponds to total non-hyperbolicity), which supports our hypothesis.

Embeddings are computed by a hyperbolic neural network trained for the MNIST classification task.

We observe a significant difference between these distributions: embeddings of the Omniglot images are much closer to the origin.

Table 2 provides the KS distances between the distributions.

In our further experiments, we concentrate on the few-shot classification and person re-identification tasks.

The experiments on the Omniglot dataset serve as a starting point, and then we move towards more complex datasets.

Afterwards, we consider two datasets, namely: MiniImageNet (Ravi & Larochelle, 2016) and Caltech-UCSD Birds-200-2011 (CUB) (Wah et al., 2011) .

Here, for each dataset, we train four models: for one-shot five-way and five-shot five-way classification tasks both in the Euclidean and hyperbolic spaces.

Finally, we provide the re-identification results for the two popular datasets: Market-1501 (Zheng et al., 2015) and DukeMTMD (Ristani et al., 2016; Zheng et al., 2017) .

Further in this section, we provide a thorough description of each experiment.

Our code is available at github 1 .

In this subsection, we validate our hypothesis which claims that if one trains a hyperbolic classifier, then a distance of the Poincaré ball embedding of an image can serve as a good measure of confidence of a model.

We start by training a simple hyperbolic convolutional neural network on the MNIST dataset.

The output of the last hidden layer was mapped to the Poincaré ball using the exponential map (10) and was followed by the hyperbolic MLR layer.

After training the model to ∼ 99% test accuracy, we evaluate it on the Omniglot dataset (by resizing images to 28 × 28 and normalizing them to have the same background color as MNIST).

We then evaluate the hyperbolic distance to the origin of embeddings produced by the network on both datasets.

The closest Euclidean analogue to this approach would be comparing distributions of p max , maximum class probability predicted by the network.

For the same range of dimensions we train ordinary Euclidean classifiers on MNIST, and compare these distributions for the same sets.

Our findings are summarized in Figure 3 and Table 2 .

We observe that distances to the origin present a more statistically significant indicator of the dataset dissimilarity in 3 cases.

We have visualized the learned MNIST and Omniglot embeddings on Figure 1 .

We observe that more 'unclear' images are located near the center, while the images that are easy to classify are located closer to the boundary.

Table 2 : Kolmogorov-Smirnov distances between the distributions of distance to the origin of the MNIST and Omniglot datasets embedded into the Poincaré ball with the hyperbolic classifier trained on MNIST, and between the distributions of p max (maximum probablity predicted for a class) for the Euclidean classifier trained on MNIST and evaluated on the same sets.

See further description in Subsection 5.1 and visualization on Figure 3 .

We observe that distance to the origin mostly presents a more statistically significant indicator of the dataset dissimilarity.

We hypothesize that a certain class of problems -namely the few-shot classification task can benefit from hyperbolic embeddings.

The starting point for our analysis is the experiments on the Omniglot dataset for few-shot classification.

This dataset consists of the images of 1623 characters sampled from 50 different alphabets; each character is supported by 20 examples.

We test several fewshot learning algorithms to see how hyperbolic embeddings affect them.

In order to validate if hyperbolic embeddings can improve models performing on the state-of-the-art level, for the baseline architecture, we choose the prototype network (ProtoNet) introduced in the paper (Snell et al., 2017) with four convolutional blocks in a backbone.

The specifics of the experimental setup can be found in B.

In ProtoNet, one uses a so-called prototype representation of a class, which is defined as a mean of the embedded support set of a class.

Generalizing this concept to hyperbolic space, we substitute the Euclidean mean operation by HypAve, defined earlier in the Equation (2).

Results are presented in Table 3 .

We can see that in some scenarios, in particular for one-shot learning, hyperbolic embeddings are more beneficial, while in other cases results are slightly worse.

Relative simplicity of this dataset may explain why have not observed significant benefit of hyperbolic embeddings.

We further test our approach on more advanced datasets.

MiniImageNet dataset is the subset of ImageNet dataset (Russakovsky et al., 2015) , which contains of 100 classes represented by 600 examples per class.

We use the following split provided in the paper (Ravi & Larochelle, 2016) : training dataset consists of 64 classes, validation dataset is represented by 16 classes, and the remaining 20 classes serve as a test dataset.

As a baseline model, we again use prototype network (ProtoNet).

We test the models on tasks for one-shot and five-shot classifications; the number of query points in each batch always equals to 15.

All implementation details can be found in Appendix B. Table 4 illustrates the obtained results on MiniImageNet dataset.

For MiniImageNet dataset, the results of the other models are available for the same classification tasks (i.e., for one-shot and fiveshot learning).

Therefore, we can compare our obtained results to those that were reported in the original papers.

From these experimental results, we may observe a slight gain in model accuracy.

The CUB dataset consists of 11, 788 images of 200 bird species and was designed for fine-grained classification.

We use the split introduced in (Triantafillou et al., 2017) : 100 classes out of 200 were used for training, 50 for validation and 50 for testing.

Also, following (Triantafillou et al., 2017) , we make the same pre-processing step by resizing each image to the size of 64×64.

The implementation details can be found in B. Our findings on the experiments on the CUB dataset are summarized in Table 4 .

Interestingly, for this dataset, the hyperbolic version significantly outperforms its Euclidean counterpart.

The DukeMTMC-reID dataset contains 16, 522 training images of 702 identities, 2228 query images of 702 identities and 17, 661 gallery images.

Market1501 contains 12936 training images of 751 identities, 3368 queries of 750 identities and 15913 gallery images respectively.

We report Rank1 of the Cumulative matching Characteristic Curve and Mean Average Precision for both datasets.

We refer the reader to B for a more detailed description of the experimental setting.

The results are reported after the 300 training epochs.

As we can see in the Table 5 , hyperbolic version generally performs better than the baseline, while the gap between the baseline and hyperbolic versions' results is decreasing for larger dimensionalities.

We have investigated the use of hyperbolic spaces for image embeddings.

The models that we have considered use Euclidean operations in most layers, and use the exponential map to move from the Euclidean to hyperbolic spaces at the end of the network (akin to the normalization layers that are used to map from the Euclidean space to Euclidean spheres).

The approach that we investigate here is thus compatible with existing backbone networks trained in Euclidean geometry.

At the same time, we have shown that across a number of tasks, in particular in the few-shot image classification, learning hyperbolic embeddings can result in a substantial boost in accuracy.

We speculate that the negative curvature of the hyperbolic spaces allows for embeddings that are better conforming to the intrinsic geometry of at least some image manifolds with their hierarchical structure.

Future work may include several potential modifications of the approach.

We have observed that the use of hyperbolic embeddings improves performance for some problems and datasets, while not helping others.

A better understanding of when and why the use of hyperbolic geometry is justified is therefore needed.

Also, we note that while all hyperbolic geometry models are equivalent in the continuous setting, fixed-precision arithmetic used in real computers breaks this equivalence.

In practice, we observed that care should be taken about numeric precision effects (following , we clip the embeddings to minimize numerical errors during learning).

Using other models of hyperbolic geometry may result in more favourable floating point performance.

The resulting formula for hyperbolic MLR for K classes is written below; here p k ∈ D n c and a k ∈ T p k D n c \ {0} are learnable parameters.

For a more thorough discussion of hyperbolic neural networks, we refer the reader to the paper ).

Omniglot As a baseline model, we consider the prototype network (ProtoNet).

Each convolutional block consists of 3 × 3 convolutional layer followed by batch normalization, ReLU nonlinearity and 2 × 2 max-pooling layer.

The number of filters in the last convolutional layer corresponds to the value of the embedding dimension, for which we choose 64.

The hyperbolic model differs from the baseline in the following aspects.

First, the output of the last convolutional block is embedded into the Poincaré ball of dimension 64 using the exponential map.

The initial value of learning rate equals to 10 −3 and is multiplied by 0.5 every 20 epochs out of total 60 epochs.

miniImageNet For this task we again considered ProtoNet as a baseline model.

Similarly, number of filters the last convolutional layer corresponds to the varying value of the embedding dimension.

In our experiments we set this value to 1024.

We test the models on tasks for one-shot and fiveshot classifications; the number of query points in each batch always equals to 15.

We consider the following learning rate decay scheme: the initial learning rate equals to 10 −3 and is further multiplied by 0.2 every 10 epochs (out of total 200 epochs).

The hyperbolic model differs from the baseline in the following aspects.

First, the output of the last convolutional block is embedded into Poincaré ball of dimension 1024 using the exponential map defined in Equation (10).

In ProtoNet, one uses a so-called prototype representation of a class, which is defined as a mean of the embedded support set of a class.

Generalizing this concept to hyperbolic space, we substitute the Euclidean mean operation by HypAve, defined earlier in the Equation (2).

The initial learning rate equals to 10 −3 and is further multiplied by 0.2 every 10 epochs (out of total 200 epochs).

Caltech-UCSD Birds Likewise, we use ProtoNet mentioned above with the following modifications.

Here, we fix the embedding dimension to 512 and use a slightly different setup for learning rate scheduler: the initial learning rate of value 10 −3 is multiplied by 0.7 every 20 epochs out of total 100 epochs.

Remaining architecture and parameters both in baseline and hyperbolic models are identical to those in the experiments on the MiniImageNet dataset.

Person re-identification We use ResNet-50 (He et al., 2016) architecture with one fully connected embedding layer following the global average pooling.

Three embedding dimensionalities are used in our experiments: 32, 64 and 128.

For the baseline experiments, we add the additional classification linear layer, followed by the cross-entropy loss.

For the hyperbolic version of the experiments, we map the descriptors to the Poincaré ball and apply multiclass logistic regression as described in Section 4.

We found that in both cases the results are very sensitive to the learning rate schedules.

We tried four schedules for learning 32-dimensional descriptors for both baseline and hyperbolic versions.

Two best performing schedules were applied for the 64 and 128-dimensional descriptors.

In these experiments, we also found that smaller c values give better results.

We finally set c to 10 −5 .

Therefore, based on the discussion in 4, our hyperbolic setting is quite close to Euclidean.

The results are compiled in Table 5 .

We set starting learning rates to 3 · 10 −4 and 6 · 10 −4

for sch#1 and sch#2 correspondingly and multiply them by 0.1 after each of the epochs 200 and 270.

@highlight

We show that hyperbolic embeddings are useful for high-level computer vision tasks, especially for few-shot classification.
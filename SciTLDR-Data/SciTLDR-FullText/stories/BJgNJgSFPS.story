Capsule networks are constrained by the parameter-expensive nature of their layers, and the general lack of provable equivariance guarantees.

We present a variation of capsule networks that aims to remedy this.

We identify that learning all pair-wise part-whole relationships between capsules of successive layers is inefficient.

Further, we also realise that the choice of prediction networks and the routing mechanism are both key to equivariance.

Based on these, we propose an alternative framework for capsule networks that learns to projectively encode the manifold of pose-variations, termed the space-of-variation (SOV), for every capsule-type of each layer.

This is done using a trainable, equivariant function defined over a grid of group-transformations.

Thus, the prediction-phase of routing involves projection into the SOV of a deeper capsule using the corresponding function.

As a specific instantiation of this idea, and also in order to reap the benefits of increased parameter-sharing, we use type-homogeneous group-equivariant convolutions of shallower capsules in this phase.

We also introduce an equivariant routing mechanism based on degree-centrality.

We show that this particular instance of our general model is equivariant, and hence preserves the compositional representation of an input under transformations.

We conduct several experiments on standard object-classification datasets that showcase the increased transformation-robustness, as well as general performance, of our model to several capsule baselines.

The hierarchical component-structure of visual objects motivates their description as instances of class-dependent spatial grammars.

The production-rules of such grammars specify this structure by laying out valid type-combinations for components of an object, their inter-geometry, as well as the behaviour of these with respect to transformations on the input.

A system that aims to truly understand a visual scene must accurately learn such grammars for all constituent objects -in effect, learning their aggregational structures.

One means of doing so is to have the internal representation of a model serve as a component-parsing of an input across several semantic resolutions.

Further, in order to mimic latent compositionalities in objects, such a representation must be reflective of detected strengths of possible spatial relationships.

A natural structure for such a representation is a parse-tree whose nodes denote components, and whose weighted parent-child edges denote the strengths of detected aggregational relationships.

Capsule networks (Hinton et al., 2011) , (Sabour et al., 2017) are a family of deep neural networks that aim to build such distributed, spatially-aware representations in a multi-class setting.

Each layer of a capsule network represents and detects instances of a set of components (of a visual scene) at a particular semantic resolution.

It does this by using vector-valued activations, termed 'capsules'.

Each capsule is meant to be interpreted as being representative of a set of generalised pose-coordinates for a visual object.

Each layer consists of capsules of several types that may be instantiated at all spatial locations depending on the nature of the image.

Thus, given an image, a capsule network provides a description of its components at various 'levels' of semantics.

In order that this distributed representation across layers be an accurate component-parsing of a visual scene, and capture meaningful and inherent spatial relationships, deeper capsules are constructed from shallower capsules using a mechanism that combines backpropagation-based learning, and consensus-based heuristics.

Briefly, the mechanism of creating deeper capsules from a set of shallower capsules is as follows.

Each deeper capsule of a particular type receives a set of predictions for its pose from a local pool of shallower capsules.

This happens by using a set of trainable neural networks that the shallower capsules are given as input into.

These networks can be interpreted as aiming to capture possible part-whole relationships between the corresponding deeper and shallower capsules.

The predictions thus obtained are then combined in a manner that ensures that the result reflects agreement among them.

This is so that capsules are activated only when their component-capsules are in the right spatial relationship to form an instance of the object-type it represents.

The agreement-based aggregation described just now is termed 'routing'.

Multiple routing algorithms exist, for example dynamic routing (Sabour et al., 2017) , EM-routing (Hinton et al., 2018) , SVD-based routing (Bahadori, 2018) , and routing based on a clustering-like objective function (Wang & Liu, 2018) .

Based on their explicit learning of compositional structures, capsule networks can be seen as an alternative (to CNNs) for better learning of compositional representations.

Indeed, CNN-based models do not have an inherent mechanism to explicitly learn or use spatial relationships in a visual scene.

Further, the common use of layers that enforce local transformation-invariance, such as pooling, further limit their ability to accurately detect compositional structures by allowing for relaxations in otherwise strict spatial relations (Hinton et al., 2011) .

Thus, despite some manner of hierarchical learning -as seen in their layers capturing simpler to more complex features as a function of depth -CNNs do not form the ideal representational model we seek.

It is our belief that capsule-based models may serve us better in this regard.

This much said, research in capsule networks is still in its infancy, and several issues have to be overcome before capsule networks can become universally applicable like CNNs.

We focus on two of these that we consider as fundamental to building better capsule network models.

First, most capsule-network models, in their current form, do not scale well to deep architectures.

A significant factor is the fact that all pair-wise relationships between capsules of two layers (upto a local pool) are explicitly modelled by a unique neural network.

Thus, for a 'convolutional capsule' layer -the number of trainable neural networks depends on the product of the spatial extent of the windowing and the product of the number of capsule-types of each the two layers.

We argue that this design is not only expensive, but also inefficient.

Given two successive capsule-layers, not all pairs of capsule-types have significant relationships.

This is due to them either representing object-components that are part of different classes, or being just incompatible in compositional structures.

The consequences of this inefficiency go beyond poor scalability.

For example, due to the large number of prediction-networks in this design, only simple functions -often just matrices -are used to model part-whole relationships.

While building deep capsule networks, such a linear inductive bias can be inaccurate in layers where complex objects are represented.

Thus, for the purpose of building deeper architectures, as well as more expressive layers, this inefficiency in the prediction phase must be handled.

The second issue with capsule networks is more theoretical, but nonetheless has implications in practice.

This is the lack, in general, of theoretical guarantees on equivariance.

Most capsule networks only use intuitive heuristics to learn transformation-robust spatial relations among components.

This is acceptable, but not ideal.

A capsule network model that can detect compositionalities in a provablyinvariant manner are more useful, and more in line with the basic motivations for capsules.

Both of the above issues are remedied in the following description of our model.

First, instead of learning pair-wise relationships among capsules, we learn to projectively encode a description of each capsule-type for every layer.

This we do by associating each capsule-type with a vector-valued function, given by a trainable neural network.

This network assumes the role of the prediction mechanism in capsule networks.

We interpret the role of this network as a means of encoding the manifold of legal pose-variations for its associated capsule-type.

It is expected that, given proper training, shallower capsules that have no relationship with a particular capsule-type will project themselves to a vector of low activation (for example, 2-norm), when input to the corresponding network.

As an aside, it is this mechanism that gives the name to our model.

We term this manifold the 'space-of-variation' of a capsule-type.

Since, we attempt to learn such spaces at each layer, we name our model 'space-of-variation' networks (SOVNET).

In this design, the number of trainable networks for a given layer depend on the number of capsule-types of that layer.

As mentioned earlier, the choice of prediction networks and routing algorithm is important to having guarantees on learning transformation-invariant compositional relationships.

Thus, in order to ensure equivariance, which we show is sufficient for the above, we use group-equivariant convolutions (GCNN) (Cohen & Welling, 2016) in the prediction phase.

Thus, shallower capsules of a fixed type are input to a GCNN associated with a deeper capsule-type to obtain predictions for it.

Apart from ensuring equivariance to transformations, GCNNs also allow for greater parameter-sharing (across a set of transformations), resulting in greater awareness of local object-structures.

We argue that this could potentially improve the quality of predictions when compared to isolated predictions made by convolutional capsule layers, such as those of (Hinton et al., 2018) .

The last contribution of this paper is an equivariant degree-centrality based routing algorithm.

The main idea of this method is to treat each prediction for a capsule as a vertex of a graph, whose weighted edges are given by a similarity measure on the predictions themselves.

Our method uses the softmaxed values of the degree scores of the affinity matrix of this graph as a set of weights for aggregating predictions.

The key idea being that predictions that agree with a majority of other predictions for the same capsule get a larger weight -following the principle of routing-by-agreement.

While this method is only heuristic in the sense of optimality, it is provably equivariant and preserves the capsule-decomposition of an input.

We summarise the contributions of this paper in the following:

1.

A general framework for a scalable capsule-network model.

A particular instantiation of this model that uses equivariant convolutions, and an equivariant, degree-centrality-based routing algorithm.

3.

A graph-based framework for studying the representation of a capsule network, and the proof of the sufficiency of equivariance for the (qualified) preservation of this representation under transformations of the input.

4.

A set of proof-of-concept, evaluative experiments on affinely transformed variations of MNIST, FASHIONMNIST, and CIFAR10, as well as separate experiments on KMNIST and SVHN that showcase the superior adapatability of SOVNET architectures to train and test-time geometric perturbations of the data, as well as their general performance.

We begin with essential definitions for a layer of SOVNET, and the properties we wish to guarantee.

Given a group (G, •), we formally describe the l th layer of a SOVNET architecture as the set of We model each capsule-type as a function over a group of transformations so as to allow for formal guarantees on transformation-equivariance.

Thus, we also model images as function from a group to a representation-space.

The main assumption being that the translation-group is a subgroup of the group in question.

This is similar in approach to (Cohen & Welling, 2016) .

We wish for each capsule-type, both pose and activation-wise, to display equivariance.

We present a formal definition of this notion.

Consider a group (G, •) and vector spaces V , W .

Let T and T be two group-representations for elements of G over V and W , respectively.

Φ: V → W is said to be equivariant with respect to T and

This definition translates to a preservation on transformations in the input-space to the output-space -something that allows no loss of information in compositional structures.

As in (Cohen & Welling, 2016) , we restrict the notion of equivariance in our model by using the operator L g in place of the group-representation.

The operator ⊗ describes the change in representation space, and is dependent on the nature of the deep learning model.

In the case of capsule networks (and SOVNET), this change is given by routing among capsules as described in subsection 2.1.

We define the capsule-types of a particular layer as an output of an agreement-based aggregation of predictions made by the preceding layer.

A recursive application of this definition is enough to define a SOVNET architecture, given an initial set of capsules.

A means of obtaining this initial set is given in section 3.

We provide a general framework for the summation-based family of routing procedures in Algorithm 1.

Algorithm 1 A general summation-based routing algorithm for SOVNET.

The weighted-sum family of routing algorithms builds deeper capsules using a weighted sum of predictions made for them by shallower capsules.

To ensure that the predictions are combined in a meaningful manner, different methods can be used to obtain the weights.

The role of the function GetW eights is to represent any such mechanism.

The activation of a capsule, representative of the probability of existence of the object it represents, is determined by the extent of the consensus among its predictions.

This is based on the routing-by-agreement principle of capsule networks.

The Agreement function represents any means of evaluating such consensus.

We instantiate the above algorithm to a specific model, as given in Algorithm 2.

In this model, the Ψ l j are group-equivariant convolutional filters, and the operator · is the corresponding groupequivariant correlation operator .

The weights c l+1 ij (g) are, in this routing method, the softmaxed degree-scores of the affinities among predictions for the same deeper capsule.

Further, like in dynamic routing (Sabour et al., 2017) , we also assume that the activation of a capsule is given by its 2-norm.

To ensure that this value is in [0, 1], we use the 'squash' function of dynamic routing.

Thus, we do not mention it explicitly.

Note that we have used the subscript notation to also denote that a variable is part of a vector, for example S l+1 ijp (g) denotes the p th element of the

This new routing algorithm is meant to serve as an alternative to existing iterative routing strategies such as dynamic routing.

An important strength of our method being that there is no hyperparameter, like that of the number of iterations in dynamic routing or EM routing.

The SOVNET layer we introduced in Algorithm 2 is group-equivariant with respect to the group action L g , where g ∈ G -the set of transformations over which the group-convolution is defined.

For notational convenience, we define ⊗ to be an operator that encapsulates the degree-routing procedure with prediction networks Ψ l+1 j .

Thus, the j th capsule-type of the l + 1 th layer is functionally depicted

.

The formal statement of this result is given below; the proof is presented in the appendix.

Theorem 2.1.

The SOVNET layer defined in Algorithm 2, and denoted by the operator ⊗ as given above, satisfies

, where g belongs to the underlying group of the equivariant convolution.

Proof.

The proof is given in the appendix.

Algorithm 2 The degree-centrality based routing algorithm for SOVNET.

Equivariance is widely considered a desirable inductive bias for a variety of reasons.

First, equivariance mirrors natural label-invariance under transformations.

Second, it lends predictability to the output of a network under (fixed) transformations of the input.

These, of course, lead to a greater robustness in handling transformations of the data.

We aim at adding to this list by showing that equivariance guarantees the preservation of detected compositionalities in a SOVNET architecture.

This is of course quite unsurprising, and has been a significant undercurrent of the capsule-network idea.

Our work completes this intuition with a formal result.

We begin by first defining the notion of a capsule-decomposition graph.

This graph is formed from the activations and the routing weights of a SOVNET.

Specifically, given an input to a SOVNET model, each capsule of every type is a vertex in this graph.

We construct an edge between capsules that are connected by routing, with the direction from the shallower capsule to the deeper capsule.

Each of these edges are weighted by the corresponding routing coefficient.

Capsules not related to each other by routing are not connected by an edge.

This graph is a direct formalisation of the various detected compositionalities with their strengths.

What should the ideal behaviour of this graph be under the change-of-viewpoint of an input?

The answer to this lies in the expected behaviour of natural compositionalities.

Thus, while the pose of objects, and their components, is changed under transformations of the input, the relative geometry is constant.

Thus, it is desirable that the capsule-decomposition graphs of a particular input (and its transformed variations) be isomorphic to each other.

We show that a SOVNET model that is equivariant with respect to a set of transformations satisfies the above property for that set.

A more formal description of the capsule-decomposition graph, and the statement for the above theorem are given below.

Consider an L-layer SOVNET model, whose routing procedure belongs to the family of methods given by Algorithm 1.

Let us consider a fixed input x : G → R c .

We define the capsule-decomposition graph of such a model, for this input x, as G(x) = (V (x), E(x)).

Here, V (x) and E(x) denote the vertex-set and the edge-set, respectively.

denotes the pool of grid-positions at layer l that route to the deeper capsule of type j of layer l + 1 at g 2 .

A more formal definition is given the appendix.

We also use the notation L hf

Theorem 2.2.

Consider an L-layer SOVNET whose activations are routed according to a procedure belonging to the family given by Algorithm 1.

Further, assume that this routing procedure is equivariant with respect to the group G. Then, given an input x and ∀g ∈ G,

are isomorphic.

Proof.

The proof is given in the appendix.

Based on above theorem, and the fact that degree-centrality based routing is equivariant, the above result applies to SOVNET models that use Algorithm 2 .

This section presents a description of the experiments we performed.

We conducted two sets of experiments; the first to compare SOVNET architectures to other capsule network baselines with respect to transformation robustness on classification, and the second to compare SOVNET to certain capsule as well as convolutional baselines based on classification performance.

Before we present the details of these experiments, we briefly describe some details of the SOVNET architecture we used.

We only present an outline -the complete details, both architecture-wise and about the training, can be found in the anonymised github repository https://github.com/sairaamVenkatraman/ SOVNET.

The first detail of the architecture pertains to the construction of the first layer of capsules.

While many approaches are possible, we used the following methodology that is similar in spirit to other capsule network models.

The first layer of the SOVNET architectures we constructed use a modified residual block that uses the SELU activation, along with group-equivariant convolutions.

This is so as to allow a meaningful set of equivariant feature maps to be used for the creation of the first set of capsules.

Intuition and some literature, for example Rosario et al. (2019) , suggest that the construction of primary capsules plays a significant role in the performance of the capsule network.

Thus, it is necessary to build a sufficiently expressive layer that yields the first set of meaningful capsule-activations.

To this end, each capsule-type in the primary capsule layer is associated with a group-convolution layer followed by a modified residual block.

The convolutional feature-maps from the preceding layer passes through each of these sub-networks to yield the primary capsules.

No routing is performed in this layer.

We now describe the SOVNET blocks.

Since the design of SOVNET significantly reduces the number of prediction networks, and thereby the number of trainable parameters, we are able to build architectures whose each layer uses more expressive prediction mechanisms than a simple matrix.

Specifically, each hidden layer of the SOVNET architectures we consider uses a (group-equivariant) modified residual block as the prediction mechanism.

We use a SOVNET architecture that uses 5 hidden layers for MNIST, FashionMNIST, KMNIST, and SVHN, and a model that uses 6 hidden layers for CIFAR-10.

Unlike DeepCaps -another capsule network whose predictions use (regular) convolution, each of the hidden layers of our SOVNET models use degree-routing.

The hidden layers of DeepCaps (excepting the last), in contrast, are not strictly capsule-based -being just convolutions whose outputs are reshaped to a capsule-form.

The output capsule-layer of SOVNET is designed similar to the hidden capsule-layers, with the difference that the prediction-mechanism is a group-convolutional implementation of a fullyconnected layer.

In order to make a prediction for the class of an input, the maximum across the rotational (and reflectional) positions of the two-norm of the capsule-activations of this layer are taken for each class-type.

This is an equivariant operation, as it corresponds to the subgroup-pooling of Cohen & Welling (2016) .

The predictions that this layer yields is the type of the capsule with the maximum 2-norm.

In order to guarantee the robustness to translations and rotations, we used the p4-convolutions (Cohen & Welling, 2016) for the prediction mechanism in all the networks used in the first set of experiments.

For the second set, we used the p4m-convolution (Cohen & Welling, 2016) , that is equivariant to rotations, translations and reflections -for greater ability to learn from augmentations.

The architectures, however are identical but for this difference.

As in (Sabour et al., 2017) , we used a margin loss and a regularising reconstruction loss to train the networks.

The positive and negative margins for half of the training epochs were set to 0.9 and 0.1, respectively.

Further, the negative margin-loss was weighted by 0.5, as in (Sabour et al., 2017) .

These values were used for the first half of the training epochs.

In order to facilitate better predictions, these values were changed to 0.95, 0.05, and 0.8, respectively for the second half of the training.

We adopt this from .

The reconstruction loss was computed by masking the incorrect classes, and by feeding the 'true' class-capsule to a series of transposed convolutions to reconstruct the image.

The mean square loss was computed for the reconstruction and original image.

The main idea being that this loss guides the capsule network to build meaningful capsules.

This loss was weighed by 0.0005 as in (Sabour et al., 2017) .

We used the Adam optimiser and an exponential learning rate scheduler that reduced the learning rate by a factor of 0.9 each epoch.

With this outline of the architecture and details of the training, we now describe the first set of experiments we conducted on SOVNET.

The preservation of detected compositionalities under transformations in SOVNET leads us to the expectation that SOVNET models, when properly trained, will display greater robustness to changes in viewpoint of the input.

Apart from handling test-time transformations, as is the commonly held notion of transformation robustness, a robust model must also effectively learn from train-time perturbations of the data.

Based on these ideas, we designed a set of experiments that compare SOVNET architectures to other capsule networks on their ability to handle train and test-time affine transformations of the data.

Specifically, we perform experiments on MNIST (LeCun & Cortes, 2010), FashionMNIST (Xiao et al., 2017) , and CIFAR-10 ( Krizhevsky & Hinton, 2009) .

For each of these datasets, we created 5 variations of the train and test-splits by randomly transforming data according to the extents of the transformations given in Table 1 .

We train a given model on each transformed version of the training-split, and test each model on each of the versions of the test-split.

Thus we obtain, for a single model, 25 accuracies per dataset -each corresponding to a pair of train and test-splits.

There is a single modification to these transformations for the case of CIFAR-10.

In order to compare SOVNET against the closest competitor DeepCaps, we use their strategy of first resizing CIFAR-10 images to 64×64, followed by translations and rotations.

We tested SOVNET against four capsule network baselines, namely Capsnet (Sabour et al., 2017) , EMcaps (Hinton et al., 2018), DeepCaps , and GCaps (Lenssen et al., 2018) .

The results of these experiments are given in Tables 2 to 4.

In the majority of the cases, SOVNET obtains the highest accuracy -showing that it is more robust to transformations of the data.

Note that we had to conduct these experiments as such a robustness study was not done in the original papers for the baselines.

We used, and modified, code from the following github sources for the implementation of the baselines: (Li, 2019) for CAPSNET; (Yang, 2019) for EMCAPS; (Rajasegaran, 2019) and (HopefulRational, 2019) for DeepCaps, and (Lenssen, 2019) for GCaps.

We also tested against a group-equivariant convolution network (GCNN).

The second set of experiments we conducted, tested SOVNET against several capsule as well as convolutional baselines.

We trained and tested SOVNET on KMNIST (Clanuwat et al., 2018) and SVHN (Netzer et al., 2011) .

With fairly standard augmentation -mild translations (and resizing for SVHN to 64×64) -the SOVNET architecture with p4m-convolutions was able to achieve on-par, or above, comparative performance.

The results of this experiment are in Table 5 .

In order to compare the performance of SOVNET architectures against more sophisticated CNN-baselines, we also trained ResNet-18, ResNet-34 on the most extreme transformation -translation by up to ± 2 pixels, and rotation by up to ± 180°.

The results of these experiments are presented in the appendix.

A number of insights can be drawn from an observation of the accuracies obtained from the experiments.

First, the most obvious, is that SOVNET is significantly more robust to train and test-time geometric transformations of the input.

Indeed, SOVNET learns to use even extreme transformations of the training data and generalises better to test-time transformations in a majority of the cases.

However, in certain splits, some baselines perform better than SOVNET.

These cases are briefly discussed below.

On the CIFAR-10 experiments, DeepCaps performs significantly better than SOVNET on the untransformed case -generalising to test-time transformations better.

However, SOVNET learns from train-time transformations better than DeepCaps -outperforming it in a large majority of the other cases.

We hypothesize that the first observation is due to the increased (almost double) number of parameters of DeepCaps that allows it to learn features that generalise better to transformations.

Further, as p4-convolutions (the prediction-mechanisms used) are equivariant only to rotations in multiples of 90°, its performance is significantly lower for test-time transformations of 30°and 60°for the untransformed case.

However, the equivariance of SOVNET allows it to learn better from train-time geometric transforms than DeepCaps, explaining the second observation.

The second case is that GCaps outperforms SOVNET on generalising to extreme transformations on (mainly) MNIST, and once on FashionMNIST, under mild train-time conditions.

However, it is unable to sustain this under more extreme train-time perturbations.

We infer that this is caused largely by the explicit geometric parameterisation of capsules in G-Caps.

While under mild-tomoderate train-time conditions, and on simple datasets, this approach could yield better results, this parameterisation, especially with very simple prediction-mechanisms, can prove detrimental.

Thus, the convolutional nature of the prediction-mechanisms, which can capture more complex features, and also the greater depth of SOVNET allows it to learn better from more complex training scenarios.

This makes the case for deeper models with more expressive and equivariant prediction-mechanisms.

A related point of interest is that G-Caps performs very poorly on the CIFAR-10 dataset -achieving the least accuracy on most cases on this dataset -despite provable guarantees on equivariance.

We argue that this is significantly due to the nature of the capsules of this model itself.

In GCaps, each capsule is explicitly modelled as an element of a Lie group.

Thus, capsules capture exclusively geometric information, and use only this information for routing.

In contrast, other capsule models have no such parameterisation.

In the case of CIFAR-10, where non-geometric features such as texture are important, we see that purely spatio-geometric based routing is not effective.

This observation allows us to make a more general hypothesis that could deal with the fundamentals of capsule networks.

We propose a trade-off in capsule networks, based on the notion of equivariance.

To appreciate this, some background is necessary on both equivariance and capsule networks.

As the body of literature concerning equivariance is quite vast, we only mention a relevant selection of papers.

Equivariance can be seen as a desirable, if not fundamental, inductive bias for neural networks used in computer vision.

Indeed, the fact that AlexNet (Krizhevsky et al., 2012) automatically learns representation that are equivariant to flips, rotation and scaling shows the importance of equivariance as well as its natural necessity (Lenc & Vedaldi, 2015) .

Thus, a neural network model that can formally guarantee this property is essential.

An early work in this regard is the group-equivariant convolution proposed in (Cohen & Welling, 2016) .

There, the authors proposed a generalisation of the 2-D spatial convolution operation to act on a general group of symmetry transforms -increasing the parameter-sharing and, thereby, improving performance.

Since then, several other models exhibiting equivariance to certain groups of transformations have been proposed, for example (Cohen et al., 2018b) , where a spherical correlation operator that exhibits rotationequivariance was introduced; (Carlos Esteves & Daniilidis, 2017) , where a network equivariant to rotation and scale, but invariant to translations was presented, and Worrall & Brostow (2018) , where a model equivariant to translations and 3D right-angled rotations was developed.

A general theory of equivariant CNNs was developed in (Cohen et al., 2018a) .

In their paper, they show that convolutions with equivariant kernels are the most general class of equivariant maps between feature spaces.

A fundamental issue with group-equivariant convolutional networks is the fact that the grid the convolution works with increases exponentially with the type of the transformations considered.

This was pointed out in (Sabour et al., 2017) ; capsules were proposed as an efficient alternative.

In a general capsule network model, each capsule is supposed to represent the pose-coordinates of an object-component.

Thus, to increase the scope of equivariance, only a linear increase in the dimension of each capsule is necessary.

This was however not formalised in most capsule architectures, which focused on other aspects such as routing (Hinton et al., 2018) , (Bahadori, 2018) , (Wang & Liu, 2018) ; general architecture , (Deliège et al., 2018) , (Rawlinson et al., 2018) , Jeong et al. (2019) , (Phaye et al., 2018) , Rosario et al. (2019) ; or application Afshar et al. (2018) .

It was only in group-equivariant capsules (Lenssen et al., 2018 ) that this idea of efficient equivariance was formalised.

Indeed, in that paper, equivariance changed from preserving the action of a group on a vector space to preserving the group-transformation on an element.

While such models scale well to larger transformation groups in the sense of preserving equivariance guarantees, we argue that they cannot efficiently handle compositionalities that involve more than spatial geometry.

The direct use of capsules as geometric pose-coordinates could lead to exponential representational inefficiencies in the number of capsules.

This is the tradeoff we referred to.

We do not attempt a formalisation of this, and instead make the observation given next.

While SOVNET (using GCNNs) lacks in transformational efficiency, the use of convolutions allows it to capture non-geometric structures well.

Further, SOVNET still retains the advantage of learning compositional structures better than CNN models due to the use of routing, placing it in a favourable position between two extremes.

We presented a scalable, equivariant model for capsule networks that uses group-equivariant convolutions and degree-centrality routing.

We proved that the model preserves detected compositionalities under transformations.

We presented the results of experiments on affine variations of various classification datasets, and showed that our model performs better than several capsule network baselines.

A second set of experiments showed that our model performs comparably to convolutional baselines on two other datasets.

We also discussed a possible tradeoff between efficiency in the transformational sense and efficiency in the representation of non-geometric compositional relations.

As future work, we aim at understanding the role of the routing algorithm in the optimality of the capsule-decomposition graph, and various other properties of interest based on it.

We also note that SOVNET allows other equivariant prediction mechanisms -each of which could result in a wider application of SOVNET to different domains.

A tuple (G, •) , where G is a non-empty set and • defines a binary operation on G, is said to form a group if the following properties are satisfied:

Existence of the identity element:

Existence of an inverse:

A.2 GROUP ACTION AND GROUP REPRESENTATION Given a group (G, •) and a vector space V , a group action is a function f : G × V → V satisfying the following properties.

A group representation is a group action by invertible linear maps.

More formally, a group representation of a group (G, •) with respect to a vector space V is a homomorphism from G to GL(V ) -the set of linear, invertible maps from V to V .

Consider a one-layer GCNN-convolutional prediction network Ψ l+1 j for a SOVNET layer l + 1, and for the d l+1 -dimensional j th capsule-type.

Intuitively, P ool l+1 j (g) is defined by the extent of the support of the g-transformed filter Ψ l+1 j .

More formally,

For a general L-layer GCNN prediction-network, P ool l+1 j (g) is defined by recursively applying the above definition through all the layers of the prediction network.

The 2-norm of a vector x = (x 0 , ..., x n−1 ) T ∈ R n , and denoted by x 2 , is defined as

We present proofs for the theorems mentioned in the main body.

Theorem B.1.

The SOVNET layer defined in Algorithm 2, and denoted by the operator ⊗ as given above, satisfies

, where g belongs to the underlying group of the equivariant convolution.

Proof.

For the theorem to be true, we must show that each step of Algorithm 2 is equivariant.

We do this step-wise.

The predictions S l+1 ij made in the first step are group-equivariant.

This follows from the fact that (Cohen & Welling, 2016) .

We now show that the DegreeScore procedure is equivariant.

We see that Degree

From the equivariance of ,

Moreover, the 2-norm of an equivariant map is also equivariant -from the equivariance of the post-composition of non-linearities over equivariant maps (Cohen & Welling, 2016) .

Also, the division of two (non-zero) equivariant maps is also equivariant.

Thus, obtaining the degree-scores is equivariant.

Again, the softmax function preserves the equivariance as it is a point-wise non-linearity.

The proof is concluded by pointing out that the product and sum of equivariant maps is also equivariant.

Theorem B.2.

Consider an L-layer SOVNET whose activations are routed according to a procedure belonging to the family given by Algorithm 1.

Further, assume that this routing procedure is equivariant with respect to the group G. Then, given an input x and ∀g ∈ G, G(x) and G([L g x]) are isomorphic.

Proof.

Consider a fixed L-layer SOVNET that is equivariant to transformations from a group G, and an input x : G → R c .

Let G(x) be the capsule-decomposition graph corresponding to x. Then G(L h x) denotes the the capsule-decomposition graph of the transformed input L h x.

We show that the mapf

.

This is from the definition of the vertex set of a capsule-decomposition graph and the fact that the map g → h −1 • g is a bijection.

We now show that (f

First, let us assume (f

However, due to the assumed equivariance of the model,f

The converse of this result is proved in the same way by considering

, and applying the above result to E(L h x) and E(L h −1 L h x).

We performed two experiments to verify that the capsule decomposition-graphs of the transformed and untransformed images are isomorphic.

For the first of these, we trained a p4-convolution based SOVNET architecture on untransformed images of MNIST and FashionMNIST.

We then considered four variations of the two test-datasetsuntransformed, and three versions rotated exactly by multiples of 90 degrees: 90, 180, and 270.

Our experiment verifies that the mapping defined in the proof of Theorem 2.2 is indeed an isomorphism.

To this end, we considered the capsule-activations as well as the degree-scores, obtained across all the capsule-layers, for each image of all the variations of the test split of the corresponding dataset.

We then mapped the activations and the degree-scores for the untransformed images by the aforesaid mapping for each of the transformations.

This corresponds to 'rotating' the activations and degree-scores by each transformation.

We then computed the squared error of these with each of the activations and degree-scores obtained from the correspondingly transformed image, respectively.

A successful verification would result in zero error (up to machine precision).

The results in Table 6 show that this happens.

The second of our experiments is an empirical verification that the test-accuracies remain unchanged under transformations for which SOVNET exhibits equivariance.

We use the same trained architecture as above, and verify that the accuracy remains unchanged under exact transformations of the images.

The results are presented in Table 7 .

The accuracies presented below are only for the purpose of veryfying the isomorphism of the of the graph.

19% 77.19% 77.19% 77.19% C.2 RESULTS ON TESTING ON UNSEEN TRANSFORMS: AFFNIST We trained a SOVNET architecture on MNIST images that are padded to size 40x40 -the size of AFFNIST images.

We augment these images by translation, as is the standard approach.

Note that the changed size of the images necessitates a different architecture.

The result of this experiment is given in Table 8 .

We see that our SOVNET architecture obtains the highest accuracy when compared to other recent capsule network models.

Method Accuracy (Sabour et al., 2017) 79.0% (Hinton et al., 2018) 93.1% (Lenssen et al., 2018) 89.10% (Jeong et al., 2019) 87.8% (Choi et al., 2019) 91.6% SOVNET 97.01%

We also trained the above SOVNET architecture on MNIST with translations in the range of [-6,6 ] pixels and rotations from [-30 ,30] degrees.

While this increases the extent of train-time augmentation, there are several test-time transformations that are unseen.

With this scheme, we achieve state-of-the-art accuracy of 99.20%.

This improves over the best, to our knowledge, accuracy of 98.3% obtained by (Tai et al., 2019) .

We considered an implementation of the CAPSNET model (Sabour et al., 2017) .

Unlike (Sabour et al., 2017) , that uses one prediction-network per connection between capsules, this model uses one prediction-network per class-capsule.

The result of this model on augmented versions of MNIST and FashionMNIST are presented in Table 9 , with corresponding accuracies of capsnet.

We have trained a SOVNET architecture on CIFAR100.

Our model has achieved an accuracy of 71.55%, an almost 4 percentage improvement over a recent capsule network model -STARCAPS (Karim Ahmed, 2019) which achieved 67.66%.

In order to compare SOVNET with more sophisticated CNN models, we performed a limited set of experiments on MNIST and FashionMNIST.

We trained ResNet18 and ResNet34 on the train split of MNIST and FashionMNIST transformed by random translations of up to ± 2 pixels, and random rotations of up to ± 180°.

The models were tested on various transformed versions of the test-splits.

The results of these experiments are given in Table 10 .

As can be seen in the table, SOVNET compares with the two much deeper CNN models.

More testing on more complex datasets, as well as deeper SOVNET models must be done, however, to obtain a better understanding of the relative performance of these two kinds of models.

Consider Algorithm 1, which is given below for convenience.

The role of the GetW eights and Agreement procedures is to evaluate the relative importances of predictions for a deeper capsule, and the extent of consensus among them, respectively.

The second of these is interpreted as a measure of the activation of the corresponding deeper capsule.

A formalisation of these concepts to a general framework for even summation-based routing so as to cover all possible notions of relative importance, and consensus is not within the scope of this paper.

Indeed, to the best of our knowledge, such a formalisation has not been successfully completed.

Thus, instead of a formal description of a general routing procedure, we provide examples to better understand the role of these two functions.

We first explain GetW eights, and then Agreement.

Algorithm A general weighted-summation routing algorithm for SOVNET.

The first example of GetW eights we provide is from the proposed degree-centrality based routing.

The algorithm is given below, again.

In this case, GetW eights is instantiated by the DegreeScore procedure, which assigns weights to predictions based on their normalised degree centrality scores.

Thus, a prediction that agrees with a significant number of its peers obtains a higher importance than one that does not.

This scheme follows the principle of routing-by-agreement, that aims to activate a deeper capsule only when its predicting shallower, component-capsules are in an acceptable spatial configuration (Hinton et al., 2011) .

The above form for the summation-based routing procedure generalises for several existing routing algorithms.

As an example, we present the dynamic routing algorithm of (Sabour et al., 2017) .

This differs with our proposed algorithm in that it is a "attention-based", rather than "agreement-based" routing algorithm.

That is, the relative importance of a prediction with respect to a fixed deeper capsule is not a direct measure of the extent of its consensus with its peers, but rather a measure of the relative attention it offers to the deeper capsule.

Thus, the weight associated with a prediction for a fixed deeper capsule by a fixed shallower capsule depends on other deeper capsules.

In order to accomodate such methods into a general procedure, we modify our formalism by having GetW eights take all the predictions as parameters, and return all the routing weights.

This modified general procedure is given in Algorithm 5.

Consider the dynamic routing algorithm of (Sabour et al., 2017) , given in Algorithm 6 -modified to our notation and also the use of group-equivariant convolutions.

The procedure DynamicRouting is the instantiation for GetW eights.

Note that the weights c ij (g) depend on the routing weights for the deeper capsules.

Due to the formulation of capsules in our paper, as in (Sabour et al., 2017) , we use the 2-norm of a capsule to denote its activation.

Thus, our degree-centrality based procedure, and also dynamic routing, do not use a separate value for this.

However, examples of algorithms that use a separate activation value exist; for example, spectral routing (Bahadori, 2018) computes the activation score from the sigmoid of the first singular value of the matrix of stacked predictions.

Our theoretical results and algorithms admit a generalisation to other groups -as long as an appropriate group-convolution is defined.

The equivariance and the preservation of detected compositionality is preserved under the condition that the group-convolution is equivariant.

As an example, consider the discrete translation group Z 2 and the regular correlation operation defined for an input with d channels by (f Ψ)(x) = t∈Z 2 d−1 k=0 f k (t)Ψ k (x − t).

The translationequivariance of this operation is proved in (Cohen & Welling, 2016) .

The general n-dimensional correlation defined on Z n is given by (f Ψ)(x) = t∈Z n d−1 k=0 f k (t)Ψ k (x − t).

This operation is equivariant to translations in n-dimensions.

The proof for this is given below.

Theorem E.1.

The n-dimensional correlation operator is equivariant with respect to Z n and the group representation L.

Proof.

Consider x, y, t ∈ Z n , and f :

Our degree-centrality based algorithm, with its use of discrete convolutions, can be used in its current form with the above convolution.

The proof of equivariance and the preservation of compositionality holds from a direct application of the above result to Theorem 2.1 and Theorem 2.2, using the underlying group as Z n .

For continuous groups such as SO(n), the degree-centrality based algorithm must use equivariant convolutions defined over it to remain equivariant.

We consider the specific case of SO(3) below.

The correlation of two functions f, Ψ : SO(3) → R d is given by:

This correlation is equivariant to transformations in SO(3), with respect to the group representation L R defined by [L r f (Q)] = f (R −1 Q), as proved in (Cohen et al., 2018b) .

It is to be noted that due to approximations introduced by the sampling of continuous functions in implementations, exact equivariance is not preserved.

However, our routing algorithm can still be used with such convolutions and does not contribute to any reduction of equivariance by itself.

This is due to the equivariance of the dot-product and the post-composition operators.

The equivariance of the post-composition operator was proved in (Cohen & Welling, 2016) .

We formally prove the equivariance of dot-product for the SO(3) group.

Theorem E.2.

The dot-product between two equivariant functions f, g : SO(3) → R d is equivariant with respect to the group representation L. That is,

The proof for the preservation of compositionality also holds by considering the infinite graph G(x).

The definition for this is the same as before.

The proof follows by using the same mapping between vertices, and from the equivariance of the routing procedure.

@highlight

A new scalable, group-equivariant model for capsule networks that preserves compositionality under transformations, and is empirically more transformation-robust to older capsule network models.
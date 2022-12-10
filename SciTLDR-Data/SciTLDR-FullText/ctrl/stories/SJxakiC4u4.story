We first pose the Unsupervised Continual Learning (UCL) problem: learning salient representations from a non-stationary stream of unlabeled data in which the number of object classes varies with time.

Given limited labeled data just before inference, those representations can also be associated with specific object types to perform classification.

To solve the UCL problem, we propose an architecture that involves a single module, called Self-Taught Associative Memory (STAM), which loosely models the function of a cortical column in the mammalian brain.

Hierarchies of STAM modules learn based on a combination of Hebbian learning, online clustering, detection of novel patterns and forgetting outliers, and top-down predictions.

We illustrate the operation of STAMs in the context of learning handwritten digits in a continual manner with only 3-12 labeled examples per class.

STAMs suggest a promising direction to solve the UCL problem without catastrophic forgetting.

Introduction.

Unsupervised Continual Learning (UCL) involves learning from a stream of unlabeled data in which the data distribution or number/type of object classes vary with time.

UCL is motivated by recent advances in Continual Learning (CL) BID3 BID11 but also differs in that it is completely unsupervised and there are no priors on the data stream.

In UCL the data stream includes unlabeled instances of both previously learned classes and, occasionally, new classes.

This setting mirrors the natural world where known object types keep re-appearing -if they do not, it makes sense to forget them.

Many CL methods involve some sort of "replay" -we argue that observing instances of known classes (perhaps infrequently) is equivalent to replaying previous instances.

To evaluate whether a given architecture can solve the UCL problem, we partition the time axis in distinct learning phases.

During each phase, the data stream includes unlabeled examples from a constant set of classes (unknown to the architecture).

At the end of each phase, we evaluate the architecture with a simple classification task.

To do so, we provide a limited number of labeled instances per class.

This labeled dataset is not available during the learning phase and it is only used to associate the class-agnostic representations that the architecture has learned with the specific classes that are present in the labeled dataset.

This is different than Semi-Supervised Learning (SSL) methods BID12 BID10 BID9 because SSL requires both labeled and unlabeled data during the training process.

We have found one SSL method compatible with the UCL problem, the latent-feature discriminate model (M1) , and we present a variation of that method in the experimental section.

To solve the UCL problem, we have developed a neuro-inspired architecture based on a model of cortical-columns, referred to as Self-Taught Associative Memory (STAM).

The connection between STAMs and cortical models is described in another reference.

Due to space constraints, we only present the STAM architecture from a functional perspective here.

The architecture is a layered hierarchy of STAM modules that involve forward, feedback, and lateral connections.

The hierarchy learns salient features through online clustering.

Each feature is a cluster centroid.

Figure 1 : Illustration of a 3-layer STAM hierarchy.

At each layer, the input is broken up into several overlapping patches (Receptive Fields), which are clustered using online k-means (Euclidean distance).

If an input RF is flagged to be novel, a new cluster is created and its centroid is initialized based on that patch.

Each layer reconstructs its output image based on the selected centroid for each RF, and that image becomes the input to the next layer.

Feedback connections are used to control the creation of new centroids based on higher-layer predictions over wider RFs.field sizes).

STAMs learn in an online manner through mechanisms that include novelty detection, forgetting outlier patterns, intrinsic dimensionality reduction, and top-down predictions.

STAMs have some superficial similarities with Convolutional Neural Networks (CNN) BID6 in that they are both layered and have increasing receptive field sizes.

However, the STAM architecture learns in a Hebbian manner without the task-specific optimization requirement of CNNs.

Further, the features learned by STAMs are highly interpretable (they are basically common patterns at different spatial resolutions), and they adapt to non-stationarities in the data distribution.

The STAM architecture also differs from previous hierarchical clustering schemes such as BID2 BID1 in that STAMs rely on online clustering (to support continual learning), novelty detection (to detect new classes), limited memory (to forget outlier centroids), and intrinsic dimensionality reduction (to generalize across instances of the same class).

In general, we do not consider iterative algorithms, such as the "deep clustering" architecture BID0 , to be compatible with UCL because they require repetitive training epochs through the same data.

Self-Taught Associative Memory (STAM) architecture.

A STAM architecture (illustrated in Figure 1 ) is composed of L layers.

The functional unit at each layer is a STAM module.

Layer i consists of M i STAM modules.

In the context of object recognition in static images, each STAM processes a Receptive Field (RF) of the input image in that layer.

The RF size gradually increases along the hierarchy (i.e., M i gradually decreases).The feedforward input to the m'th STAM module of layer i at time t is denoted by x i,m (t).

The set C i (t) of clusters at layer i is shared among all STAMs of that layer.

The j'th centroid of layer i is denoted by the vector w i,j (t).

We drop the time variable t when it is not necessary.

Given the set of C i clusters, each STAM module of layer i selects the nearest centroid to its input based on Euclidean distance: DISPLAYFORM0 The input of layer i + 1 is the output of the previous layer.

The output of layer i, denoted by Y i , is of the same (extrinsic) dimensionality with the input X i in that layer.

Y i is constructed by the sequence of selected centroids, first replacing the input RF x i,m with the corresponding centroid c(x i,m ), and averaging the overlapping segments.

Consequently, the intrinsic dimensionality of Y i is much lower than that of X i : Y i can take |C i | Mi distinct values, and M i decreases along the hierarchy as the RFs get larger.

A STAM learns in an online manner by updating the centroid that has been selected by its input vector.

If the m'th STAM module selected centroid j at layer i for its input vector x i,m , we update that centroid as follows: DISPLAYFORM1 where the constant α is a learning rate parameter 0 < α < 1.

The higher the α, the faster the learning process becomes, potentially resulting in lower accuracy.

In the rest of this paper, α=0.05.Centroids are created and initialized dynamically, based on a novelty detection algorithm.

To detect novel patterns, we estimate in an online manner the mean distance µ j and standard differenceσ j between a centroid j and its assigned inputs: DISPLAYFORM2 Based on the previous two online estimates, an input x i,m is flagged as "novel" if its distance from the nearest centroid j is significantly larger than the centroid's mean distance µ j estimate, DISPLAYFORM3 If x i,m is flagged as novel, a new centroid is created at layer i and it is initialized based on that input.

The number of centroids learned at each layer is fixed: layer i cannot remember more than C i centroids.

When that number is exceeded, the centroid that has been Least Recently Used (LRU) is forgotten.

To help differentiate between patterns that are outliers and true novelties, we leverage top-down connections.

Suppose that y i+1,m is the portion of Y i+1 that corresponds to the m'th RF at the ith layer, and let c i (y i+1,m ) be the layer i centroid that is nearest to to y i+1,m .

This centroid represents the prediction of layer i + 1 for the m'th RF at layer i. If the corresponding input x i,m at layer i was flagged as novel but y i+1,m does not pass the "novelty" criterion of equation 5, then we do not create a new centroid for that input at layer i.

Classification.

In principle, we can use any classifier to evaluate the representations (centroids) that the STAM architecture has learned at the end of a learning phase.

Here, we describe a simple classifier that first associates each output-layer centroid with a class by calculating the "allegiance" of each labeled input vector x n to centroid w j relative to the nearest-neighbor centroid: DISPLAYFORM4 The allegiance of centroid w j to class m is simply the average s wj ,xn across all labeled inputs x n that belong to class m: DISPLAYFORM5 where N m is the number of labeled examples of class m, and y n is the class of input x n .

It is possible that a centroid at the output layer does not have strong allegiance to any class.

For this reason, we remove centroids for which the maximum allegiance max m (S wj ,m ) is less than 70%.The classification of an input x is based on the distance between x and each centroid as well as the allegiance of each centroid to every class.

Specifically, x is assigned to the class m that maximizes the following sum across all centroids w j , DISPLAYFORM6 Experiments.

We divide the time axis into five learning phases.

In each learning phase, the data stream includes two additional classes (digits) from the MNIST dataset, i.e., the first learning phase includes only 0s and 1s, while the fifth learning phase includes all ten digits.

In each learning phase, the architecture has access to N old unlabeled examples per class of previously learned classes and N new unlabeled examples per class of newly introduced classes.

At the end of each phase, we introduce a limited amount of labeled data per class to evaluate classification accuracy.

Together with the STAM architecture, we also train a Convolutional AutoEncoder (CAE) BID8 in an unsupervised manner, and then create a classifier using latent representations of the labeled data for each evaluation period.

The CAE architecture was designed specifically for the MNIST dataset, using three convolution and max pooling layers in the encoder and three convolution and upsampling layers in the decoder.

We optimize binary cross-entropy loss using the Adam method BID4 .

As another baseline, we simply consider a single-layer STAM, which can be interpreted as a non-hierarchical version of the STAM architecture.

For both STAMs and the CAE network, we use two classifiers: nearest-neighbor (NN) and the classifier of (equation 8) -referred to as EQ8 in the results.

We apply EQ8 on the CAE latent representations as centroids with allegiance only to the class corresponding to the input instance's label.

The STAM architecture is described in TAB1 .

We present results (Figures 2-3) for two experiments on the MNIST dataset BID7 based on 10 trials, evaluating accuracy on 10,000 images that were not seen during training.

TAB1 For the first experiment, we compare classification accuracy for various amounts of unlabeled data.

We consider N old = N new = {1, 000, 10, 000} and provide 10 labeled examples per class for classification.

We observe that the performance of the CAE and singlelayer baselines strongly fall off when reducing the unlabeled data to 1, 000, whereas the STAM architecture shows less catastrophic forgetting.

For the second experiment, we repeat the first experiment varying the amount of labeled data per class and we report only the classification accuracy at the last learning phase.

As expected, STAMs and CAE both see large benefits from increasing the number of labeled examples per class.

However, we see that STAM can perform reasonably well with fewer labeled examples compared to the CAE baseline.

<|TLDR|>

@highlight

We introduce unsupervised continual learning (UCL) and a neuro-inspired architecture that solves the UCL problem.

@highlight

Proposes using hierachies of STAM modules to solve the UCL problem, providing evidence that the representations the modules learn are well-suited for few-shot classification.
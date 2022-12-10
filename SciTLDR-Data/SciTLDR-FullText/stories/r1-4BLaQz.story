Theories in cognitive psychology postulate that humans use similarity as a basis for object categorization.

However, work in image classification generally as- sumes disjoint and equally dissimilar classes to achieve super-human levels of performance on certain datasets.

In our work, we adapt notions of similarity using weak labels over multiple hierarchical levels to boost classification performance.

Instead of pitting clustering directly against classification, we use a warm-start based evaluation to explicitly provide value to a clustering representation by its ability to aid classification.

We evaluate on CIFAR10 and a fine-grained classifi- cation dataset to show improvements in performance with the procedural addition of intermediate losses and weak labels based on multiple hierarchy levels.

Further- more, we show that pretraining AlexNet on hierarchical weak labels in conjunc- tion with intermediate losses outperforms a classification baseline by over 17% on a subset of Birdsnap dataset.

Finally, we show improvement over AlexNet trained using ImageNet pre-trained weights as initializations which further supports our  claim of the importance of similarity.

Similarity is one of the bases of object categorization in humans BID14 .

Theories of perceptual categorization in cognitive psychology postulate that humans construct categories by grouping similar stimuli to construct one or several prototypes.

New instances are then labeled as the category that they are most similar to.

For instance, when one categorizes a specific animal as a dog, they are saying that it is more similar to previously observed dogs than it is to all other objects.

This view of categorization better explains the often fuzzy boundaries between real-life classes, where a new object may be equally similar to multiple prototypes.

For example, while a dolphin is technically a mammal, its visual appearance is similar to fish, resulting in a lot of people misclassifying it.

While research in cognitive psychology on object categorization might indicate that similarity should play a central role in object classification in humans, image classification in computer vision seems to do pretty well without using it.

The task is commonly interpreted as one of classifying an image into one of multiple classes that are assumed to be disjoint and equally dissimilar.

The use of softmax-based loss functions assumes that the classes are disjoint, while the use of one-hot labels assumes that classes are equally dissimilar.

Despite those strong assumptions, which seem to violate human notions of categories, image classification has shown remarkable progress, achieving superhuman performance on multiple datasets BID6 BID3 .

Far from being an anomaly, state-of-the-art models across image classification benchmark datasets use losses, such as cross-entropy loss, that make the explicit assumption of disjoint classes.

However, BID4 notes that the predictions of ensembles of image classification models produces soft labels that "define a rich similarity structure over the data" despite the predictions of individual models not capturing this structure.

Can similarity-based metrics improve classification performance in convolutional neural networks?

Previous work has tried to answer this question in different problems such as metric learning BID0 , clustering BID16 , and hierarchical classification BID11 .

We focus on applications of similarity in the context of convolutional neural networks such as BID5 who use a contrastive loss to perform end-to-end clustering using weak labels.

While they show impressive performance on MNIST BID10 and CIFAR-10 (Krizhevsky & Hinton, 2009) , their method does not scale well to more complex datasets.

BID13 propose a new loss function, magnet loss, to learn a representational space where distance corresponds to similarity.

They show that clustering in this space allows them to achieve state-of-the-art performance on multiple fine-grained classification datasets.1 However, they initialize their model with pre-trained weights on ImageNet BID15 , so it is not clear whether their model is learning a good representation, or if it is finding a good mapping between already learned representations and the labels.

BID20 pose the problem as one of hierarchical classification and propose a loss based on an ultra-metric tree representation of a hierarchy over the labels.

While their loss has interesting properties, it only outperforms classification losses for datasets with a small number of instances per class.

In this work, we use a contrastive loss to improve the classification performance of randomly initialized convolutional neural networks on a fine-grained classification task.

What measures of similarity should we teach our model?

We represent the relations between our classes in a hierarchy where the labels are all leaf nodes.

Therefore, there are two kinds of similarities that we want our model to capture.

The first is intra-class similarity-all cats are similar to each other and they are dissimilar to other animals.

The second is inter-class similarity-dogs and cats are more similar to each other than they are to non-living objects.

For a more complex hierarchy, our model should learn similarity at different levels of the class hierarchy.

BID20 use their hierarchical loss function to learn those similarities, however, they observe that reducing the similarity between two classes across all levels of the hierarchy to a single value biases the model towards correct classification at the higher levels of the hierarchy resulting in poor classification performance.

BID13 also observe that applying classification losses to the final layer reduces the entire representation of each class to a single scalar value which destroys intra-and inter-class variation.

However, these are the same variations that we want our model to capture.

We overcome those limitations by explicitly training the model to capture different grains of similarity at different levels of the network through applying an intermediate pairwise contrastive loss at those levels.

In this manner, we can use hierarchical information, such as species of birds having the same coarse-grained category of bird while having different fine-grained labels.

Despite being able to encode different levels of similarity, a contrastive loss does not require an explicit hierarchy; we only need weak labels for pairs of instances.

How can we evaluate the representations learned by a clustering algorithm?

Previous work has always tried to compete against classification using metrics biased towards the latter task.

While some researchers have used hierarchical metrics BID20 or qualitative measures BID13 to evaluate the quality of their representations, their primary evaluation criteria has consistently been the classification accuracy of their model.

We propose another way to evaluate the clustering representations by using the clustering model as a "warm start" from which they can train on classification.

The intuition is that if the clustering model learned a representations that captures similarity in a space, it would be be at an advantage when it is fine-tuned for classification.

We propose the use of a contrastive loss at different layers of a convolutional neural network to learn a notion of similarity across a set of labels.

We also propose the use of the accuracy of a model pre-trained for clustering and fine-tuned for classification as an evaluation metric for clustering algorithms.

Pairwise contrastive losses were first proposed by BID2 to learn a function that maps high dimensional inputs to lower dimensional outputs such that distances in the lower dimensional space approximate relationships in the input space.

Relationships in the input space are not restricted to simple distance measures.

In the context of image classification, the resulting function would have to learn the complex invariances that make two images of the same class similar, such that they are both mapped to points that are close to each other in the output space.

This loss can be calculated using the representation of two points in the same embedding space and a binary label for whether or not they are similar.

Hence, the loss only requires weak labels in the form of pairwise relationships.

DISPLAYFORM0 ( 1) The first half of Eq. (1) penalizes any divergence between the two vectors for similar instances while the second half of the loss rewards divergence up to a margin.

Following the formulation proposed by BID5 , we use Kullback-Leibler (KL) divergence as our distance metric and a hinge loss for dissimilar cases.

The final formulation of the loss we use is shown in Eq. (2): DISPLAYFORM1 DISPLAYFORM2 DISPLAYFORM3 We have found that a value of two for the margin works well.

Since we apply the loss to activations at intermediate layers of the network, we first apply a softmax operation on the activations to obtain the two vectors, P and Q, used in Eq. (2).

Given that one of our objectives is to encode different levels of similarity into the representations learned by the network, we explore different ways of using the intermediate loss to achieve that goal as shown in FIG1 .

We establish a baseline network where the losses are all applied to the output of the last layer.

We then use three different variants by applying intermediate losses at different levels of the network.

In variant B, we apply the intermediate loss at the penultimate layer.

This allows us to apply the contrastive-loss to a higher dimensional space where it could potentially learn a richer representation.

In variants C and D, we provide the network with two different kinds of weak labels based on fine-grained and coarse-grained.

Since we expect that a weaker form of similarity would be more beneficial earlier on in the network pipeline, we apply the coarse-grained loss to the secondto-last fully-connected layer in Network C. It should be noted that many instance pairs will have contradicting losses since they may match at the coarse-grained level but conflict at the fine-grained level.

To tackle this, we augment the network with a skip layer around the second-to-last layer and apply the coarse-grained loss to the skip layer only.

Hence, the coarse-grained loss will only provide additional information to the network without directly detracting from the representations passed to subsequent layers.

While we show the network variants on AlexNet, the same ideas are applied to a variant of LeNet that has three fully-connected layers.

We follow the efficient implementation regime proposed by BID5 in order to train a single network based on pairwise constraints.

The original network formulation to handle pairwise constraints is a Siamese network.

For every pairwise constraint, at least two data passes need to be performed which results in redundancies.

Instead, we feed the softmax outcomes from a single network and the pairwise similarity constraints to the contrastive loss function which externally handles all possible pairwise constraints that are available within a mini-batch.

We further extend the application of contrastive loss functions to multiple intermediate layers with a minimal overhead of extra processing and pairwise constraints, if alternate hierarchical labels are used.

Details regarding the exact parameters and setup used to perform experiments are furnished in the Appendices.

PyTorch 2 is used for all implementations discussed within this paper.

We evaluate our approach on two different datasets using two different convolutional neural networks: LeNet on CIFAR10 and AlexNet on Birdsnap34-a 34-class subset of the Birdsnap dataset BID1 .

We generate coarse-grained labels for CIFAR10 by separating the ten classes into six animal classes and four vehicle classes.

Below is a discussion of how we constructed Birdsnap34.

Birdsnap is a popular fine-grained classification dataset that has 500 different classes and over 40,000 images.

We annotated Birdsnap using the scientific classification of each bird to create a six-level hierarchy over all the labels.

In this work, we use the birds' subfamily and family as our fine-grained and coarse-grained labels, respectively.

Using a pairwise loss with a large number of classes poses a significant implementation challenge.

For a uniform dataset with n classes, the ratio of similar pairs to dissimilar pairs is roughly 1 : (n − 1).

Hence, for a large number of classes, most of the pairs will be dissimilar resulting in a very weak signal to the model for clustering instances together.

This challenge is analogous to that of imbalanced datasets BID19 , which is a major problem that lies beyond the scope of our paper.

Instead of abandoning Birdsnap, we curate a 34-class subset, which we will refer to as Birdsnap34, and apply our approach to it.

The classes are chosen such that each class comes from a unique biological subfamily of birds.

Through sampling form unique subfamilies, we increase the interclass variance between our classes.

The resulting dataset has 2982 training images and 158 test images that are divided into 34 fine-grained and 18 coarse-grained classes.

We plan on extending our model to the entire Birdsnap dataset in future work.

All the network architectures used have randomly initialized weights unless specified otherwise.

The models are evaluated using accuracy, purity, and Normalized Mutual Information (NMI) BID17 .

We train each model for a fixed number of epochs, and we report the top performing results out of 5 runs.

Details regarding the exact parameters and setups used to perform the experiments are available in Appendix B.

In our first experiment, we show the value of evaluating the representations learned by a clustering loss through using it as a warm start from training on classification.

We replicate the setup used by BID5 by using LeNet on CIFAR10, and extend their framework to AlexNet on Birdsnap34.

We use the Hungarian algorithm BID9 to evaluate clustering accuracy.

This method of evaluation assumes that the clusters will be centered across the different dimensions of the softmax output of the final layer.

The Hungarian algorithm is then used to find the assignment of dimensions to classes that would achieve the highest accuracy.

We use the Hungarian algorithm to evaluate clustering performance in all of our experiments.

From TAB0 we observe that for CIFAR10, the application of the loss to the final layer results in clustering performance almost matching classification performance.

However, in more complex dataset such as Birdsnap34, this pattern disappears.

This supports the pattern shown in BID5 where the gap between clustering and classification performance increases when they move from MNIST to CIFAR10.

The Cross-Entropy loss formulation, as mentioned in BID5 , is extremely harsh and seeks to segment each and every sample into one absolute class.

The tSNE BID18 outcomes from the last fully connected layer of LeNet in FIG2

The previous experiment indicates that the representations learned from end-to-end clustering do not extend beyond simple datasets.

We posit that this is a result of how the clustering loss is applied.

Most machine learning models only apply losses to their last layer.

This is generally a reasonable thing to do since we only have labels for the output of the last layer.

On the other hand, pairwise losses can be applied at any level of the network since they only require a pair of vectors in the same embedding space.

In this experiment we explore the potential of applying losses at intermediate layers on the learned representations.

We train two variants of LeNet and AlexNet using the pairwise contrastive loss.

Variant A follows the setup used by BID5 , while Variant B applies the same loss at both the final and penultimate layer of the convolutional neural network.

We also train the same network using CrossEntropy loss to establish baseline performance.

As shown in TAB2 , performance on CIFAR10 is almost the same across all three settings.

Meanwhile, we see clear differences on Birdsnap34.

There is a significant increase across all metrics for the clustering models for the intermediate loss version, however, this improvement does not carry over to the classification accuracy.

Previous work on embedding spaces has shown that adding structure to the embedding space or learning it using a hierarchical structure often validates its integrity as well as improves performance BID12 .

In an effort to understand the impact of adding a hierarchical-labelbased contrastive loss to our baseline networks, we apply it on variants C and D of LeNet and AlexNet.

Since the restriction on dimensionality is removed by using a contrastive loss with KL divergence, we can directly apply the same loss to an alternate set of weak labels and representations.

Our ultimate expectation is that adding hierarchical structure should aid in improving classification, given their potency to improve embedding spaces as shown in previous works.

TAB3 shows that the experimental results align with our expectations in terms of overall improvement in classification.

As seen in other experiments, the performance on CIFAR10 is almost the same across all metrics.

Meanwhile, there is a spike in performance for Birdsnap34 for both AlexNet C and D with the metrics far exceeding classification performance.

Given that Birdsnap34 has a more well-defined hierarchy, it would be expected that using the hierarchical labels to learn similarity would result in learning good representations.

Another thing worth noting is that the dataset is curated by having subfamilies as fine-grained labels and biological families as coarsegrained labels.

This structure results in similarity along this tree corresponding to distinct visual features.

This might be indicating that choice of hierarchy can have an impact on performance.

A surprising finding is that AlexNet C outperforms AlexNet D. One would expect AlexNet D to be able to extract features for both the coarse and fine levels of the hierarchy.

We plan on investigating this in future work.

Given the ability of standard end-to-end clustering networks to perform fine-tuned classification reasonably well, we test their sensitivity towards network size in order to understand if they would be able to achieve or outperform their simpler baseline classification counterparts.

In order to do so, we double the number of convolutional filters and the sizes of fully-connected layers within AlexNet.

We use their doubled networks to perform clustering and fine-tune them for classification.

We denote these modified networks with a suffix of "Double." From TAB4 , we clearly observe that clustering is highly sensitive to network capacity.

Compared to AlexNet A, its doubled counterpart allows clustering to learn better representations which leads to classificaiton performance being higher than the baseline classification model.

However, our AlexNet C variant outperforms the double capacity network.

This points to the efficacy of the combination of weak labels over multiple levels of heirarchy and the use of intermediate losses.

Furthermore, TAB5 shows that even we use ImageNet pre-trained weights, variant C outperforms the ImageNet pre-trained classification baseline.

This points to a tractable and viable approach to extending performance over more complex datasets without the need to increase the capacity of neural networks.

In this work, we argue that similarity-based losses can be applied as a warm start to boost the performance of image classification models.

We show that applying a pairwise contrastive loss to intermediate layers of the network results in better clustering performance, as well as better finetuned classification performance.

Furthermore, we demonstrate how the pairwise loss can be used to train a model using weak hierarchical labels.

We find that training with hierarchical labels results in the highest performance beating models with more parameters and approaches the performance of models pre-trained with ImageNet weights.

This is a very significant finding since ImageNet contains multiple bird classes, so a model pretrained with ImageNet has seen many more birds than a model pre-trained with pairwise constraints on Birdsnap34.

Nevertheless, it only outperforms a cluster-based warm-start model by just 10%.

Regardless, we also show that applying our approach to ImageNet weights still results in a boost of 1.27%.

This supports our claim that similarity-based metrics can improve classification performance, but it suggests that the gains we expect decrease if we start of with a good representational space.

We hope to expand this in future work in multiple ways.

We plan on extending our approach to the entire Birdsnap dataset, as well as to other fine-grained classification datasets.

We also hope to perform more extensive analysis of the quality of the embedding spaces learned as it is obvious that the use of the Hungarian algorithm does not accurately reflect the performance of the clustering model.

We use two primary datasets, CIFAR10 and Birdsnap34, in experiments that substantiate our claims of improving on the classification task by means of using clustering representations as a warm-start.

For the CIFAR10 dataset, we follow the preprocessing steps used by BID5 .

They are listed below,• Convert the images to YUV format.• Calculate the mean and standard deviation of U and V channels over the entire training set.• For each image, normalize by its Y channel mean and standard deviation while normalizing over the remaining channels using the aggregate mean and standard deviation over the entire training set.

NOTE:

The mean and standard deviation for U and V channel as per our calculations are (0.001, 0.003) and (0.227, 0.105).For the Birdsnap34 dataset, we perform the standard normalization using mean and standard deviations for R, G and B channels as (0.485, 0.229), (0.456, 0.224) and (0.406, 0.225).B EXPERIMENTAL SETUP

@highlight

Cluster before you classify; using weak labels to improve classification 

@highlight

Proposes using a clustering based loss function at multiple levels of a deepnet as well as using hierarchical structure of the label space to train better representations.

@highlight

This paper uses hierarchical label information to impose additional losses on intermediate representations in neural network training.
In this work, we approach one-shot and few-shot learning problems as methods for finding good prototypes for each class, where these prototypes are generalizable to new data samples and classes.

We propose a metric learner that learns a Bregman divergence by learning its underlying convex function.

Bregman divergences are a good candidate for this framework given they are the only class of divergences with the property that the best representative of a set of points is given by its mean.

We propose a flexible extension to prototypical networks to enable joint learning of the embedding and the divergence, while preserving computational efficiency.

Our preliminary results are comparable with the prior work on the Omniglot and Mini-ImageNet datasets, two standard benchmarks for one-shot and few-shot learning.

We argue that our model can be used for other tasks that involve metric learning or tasks that require approximate convexity such as structured prediction and data completion.

Deep learning methods have shown tremendous performance on many tasks involving large-scale data.

However, collecting large amounts of data is costly or even infeasible for many applications BID8 BID0 .

The few-shot learning problem aims to achieve good performance on adapting to novel classes where only a small number of examples per novel class are available.

In scenarios with few examples, classical classification, fine-tuning, or retraining methods fail due to severe overfitting, catastrophic forgetting, or inflexibility to adapt to new samples and categories BID3 .This problem has been of increasing interest to researchers, some of whom have been inspired by humans' ability to 1 Boston University, MA, USA.

Correspondence to: Kubra Cilingir <kubra@bu.edu>, Brian Kulis <bkulis@bu.edu>. recognize novel classes very successfully with very few examples.

The most recent approaches to solve the few-shot learning problem involve meta learning, which attempts to learn transferable knowledge between classes and tasks at training time, in order to help generalization and adaptivity at test time.

Information is stored either in the initialization of the weights BID4 , in a recurrent memory unit BID16 , in the optimization strategy BID13 , or in an embedded space BID17 .

In this work, we focus on the last approach due to its simplicity and compelling results, whereas the other methods require complex training mechanisms, complex inference, or the gathering of many similar tasks.

In particular, we based our approach on prototype networks BID17 , which learn an embedding of the input data, and then construct prototypes for classes via averages or weighted averages over points in each class.

A single vector representation per class is assumed to be sufficient to contain class-specific features BID14 .

In BID17 , the Euclidean distance is used to measure distance between a query point and a class prototype.

In contrast to existing work, we treat the problem as a joint embedding and metric learning problem.

Because prototypes are typically represented by means of points, for the metric learning function we choose to learn a Bregman divergence as the underlying divergence.

This class of divergences has the key property that the best representative of a set of points (in terms of the sum of divergences between the points and the representative) is given by the mean, which we argue makes it appropriate for constructing prototypes of classes for our problem.

Compared to existing methods such as relation networks BID18 , ours is a more flexible approach since we focus on Bregman divergences, of which squared Euclidean distances are a special case.

We may favor Bregman divergences over Euclidean distances since symmetry and the triangle inequality may not be necessary for data in a few-shot learning problem.

In FIG1 we show a possible scenario to demonstrate this.

Suppose image A and image B have the same shape, and image B and image C have the same color.

Image A and image C need not be similar, but the triangle inequality forces a resemblance between A and C. Similarly, representations and similarity measures of class members' may not be desired to be symmetric.

Proto- types share the abstract representative features with the data points, but each point has idiosyncratic features, that may break symmetry when interpreting the embedding space.

Each Bregman divergence is parametrized by a convex function; furthermore, this relationship is a surjection.

We design the metric learning function of our deep learning model with a convexity constraint with respect to the embedding space.

This convex function is used to calculate the Bregman divergence as a learned metric.

Our formulation also provides flexibility for the architectural design of the convex function, with a regularization term to improve generalizability.

We empirically measure a convexity score by drawing random points from the convex hull of the data samples to verify our claim.

Overall, we propose a model that has a learnable embedding and a learnable Bregman divergence that can be trained simultaneously.

Compared with the state-of-art, our initial results are promising.

Other than improving the results with the current model and testing on other datasets, our preliminary work has two clear future directions: (i) Taking our convex framework to other sets of problems such as semi-supervised learning, similarity learning, structured prediction, etc.(ii) Following different approaches to satisfy and measure convexity such as modifying the constraint or the optimization algorithm itself.

Few shot learning methods have received increasing interest given the recent success of discriminative models.

Many of these effective methods fall under meta learning, where these approaches store transferable information in different ways to remedy overfitting issues and provide adaptivity for new samples and classes.

The most well-known method under this category is MAML BID4 .

These methods attempt to learn an initialization over the weights, such that a similar few shot learning problem can be adapted by fine-tuning BID6 , BID11 .

Many related target tasks are employed in order to train a model which can later be fine-tuned for each task.

The need for fine-tuning and many tasks limits the efficiency of these methods.

Meta learning by a recurrent memory: In these methods, such as MANN BID16 , Meta Nets BID12 , and Memory Matching Networks BID2 , the useful knowledge required to solve the tasks are stored in a recurrent manner using a memory unit.

Existing information in the memory and new information is compared to update the model and perform the task.

However, this type of algorithm suffers from inherent issues in RNNs such as instability and difficulty in properly storing long-term dependency.

Meta learning by optimizer: This category of methods aims at training an optimizer to provide gradients and to be used in fine tuning.

The LSTM-based optimizer BID13 is an example of this approach.

These methods also require fine-tuning while currently-proposed optimizers add unnecessary complexity to the training phase compared to the benefit for their performance.

Meta learning by embedding: These approaches are based on metric learning methods that aim to learn an embedded space to store the relations between the samples and classes.

A comprehensive overview on metric learning can be found in BID10 .

The task is then performed with a classifier that uses the embedding and a fixed metric.

This type of method is free from the various complexity issues that the other approaches have.

Siamese Networks BID9 , Matching Networks BID19 , Relation Nets BID18 , and Prototypical Nets BID17 are examples from this class which efficiently represent each class by its mean.

Our work resembles this setting, with the distinction that our work jointly learns the embedding and the metric from a set of distance families called Bregman divergences.

Another approach for fewshot learning is to leverage additional data to eliminate overfitting issues BID1 , BID20 .

Learned transformations are applied or new data is generated to enrich prior information.

We will not consider this approach separately since they can be combined with the above methods.

We now define our problem setup and notations.

Assume we are given a set of examples coming from N classes.

The training data is split into a training set, and a separate validation set V .

Both sets are divided into a support set S with K samples per class and a query set Q with the remaining samples for each class.

The tasks are defined as N -way K-shot learning, which corresponds to having K samples for each one of N classes in S. We consider the cases where K is 1 or 5 and N is 5 for our experiments.

We refer to the embedding function of our model as f w f , and the subsequent convexified layers as φ w φ where w f and w φ are the trainable weights for these functions, respectively.

For simplicity we drop the weight variables from the notation, and continue with f and φ.

We use x and x to represent a random pair from the space of interest.

Before diving into details of our model, we briefly review Bregman divergences and their properties to better clarify our choice for the class of divergences.

Bregman divergences are derived from a strictly convex and differentiable function, denoted as φ.

The Bregman divergence between two points x and x is defined as: DISPLAYFORM0 Bregman divergences are not metrics, but they satisfy sufficient properties for our problem, namely non-negativity and having a unique 0.

We previously discussed why symmetry and the triangle inequality may not be desired for our setting.

Since we are exploring the case where we represent classes as their means in the embedded space, our choice of distance should have the mean as the minimizer for the distances within a class.

Mean Minimization Property:

Assume we have an ndimensional random variable X defined on a convex set Ω ∈ R n .

d : R n × R n → R is a continuous function and d(X, X) ≥ 0 with continuous derivatives.

It is known that DISPLAYFORM1 , if, and only if d ∈ D φ where φ is the corresponding strictly convex function BID5 .For the cases where all samples are not available, or if the distribution is discrete, the expectation can simply be replaced by the sample averages.

Assume we have M observed samples DISPLAYFORM2 .

The inequality can be rewritten as the following: DISPLAYFORM3 Thus, given a set of points, the best representative is given by the mean, under any Bregman divergence.

Each Bregman divergence is identified by a φ such as Euclidean distance is a Bregman divergence identified with φ = x 2 .

We overview our model in FIG2 .

Our model consists of two learnable functions: (i) the embedding function f , and (ii) the metric learning function φ.

The embedding function f brings in non-linearity to the framework, which provides flexibility when integrating with the metric learning function.

The metric learning function is a neural network φ that is trained via a convexity constraint to output a convex function with respect to the embedded features f (x i ), followed by Bregman divergence using the learned φ.

We impose convexity by using midpoint convexity characterization with the continuity of f , which is equivalent to the standard convexity definition.

Midpoint Convexity: φ is midpoint convex if and only if DISPLAYFORM0 ≥ 0 for all x ,x ∈ Ω. If φ is continuous and satisfies midpoint convexity, φ is convex BID7 .At a high level the problem can be expressed as: DISPLAYFORM1 where µ f n represents the n th class's mean on the embedded space induced by f .

x and y represent the sample and target pairs coming from T .

L p represents a classification loss, e.g., cross entropy loss or mean squared error loss.

Midpoint convexity implies that for any input pairs, the midpoint value of the function is not greater than the function value of the midpoint of the pairs.

The integral turns into a summation since we have a finite number of samples.

This definition naturally integrates to our framework without additional significant computation since pairs are already used to determine the similarities.

We obtain an approximately convex network by feeding a sufficient number of samples.

We reformulate the indicator function with a clamping function to impose the convexity inequality, which penalizes any pair of points that violates the midpoint convexity constraint.

This formulation also gives flexibility in architectural design for the convex function.

We use a regularization term in order to further control overfitting and convexity in hard tasks by controlling the gradient change, i.e, the Lipschitz constant of the convex function.

DISPLAYFORM2 We combine L p loss and L r loss with a weighting term, and relax the LHS term in the constraint inequality in 2, denoted as L c , to train our model.

We follow a joint training approach; however, it is of value to note that alternating training between the embedding function and the metric function would be another option suitable for our setting.

We applied our model on two commonly used datasets for our experiments: Omniglot and mini-Imagenet.

The Omniglot dataset consists of 1,623 handwritten letters coming from 60 different alphabets.

Each letter image has dimensions 28x28 and has 20 samples.

We applied rotations to increase the number of classes.

The data is then divided into training, validation and test sets with 4112, 688 and 423 classes similar to a previous approach BID17 .

The Mini-Imagenet dataset is derived from ILSVRC12 BID15 .

It contains 1000 classes with 600 84x84 images each.

We split the data into training, validation and test cases by using a standard method proposed in BID13 .We choose our model architecture to be comparable with existing methods.

We use 3 convnet blocks for the embedding function, where each block consists of a 3x3 convolutional layer followed by a batch normalization, ReLU and pooling layers.

Our embedding layer is 128 dimensional for the Omniglot and 512 dimensional for the mini Imagenet.

Our convex network also contains 2 fully connected layers with sigmoid activations.

The fully connected layers followed by a linear layer output a scalar for each input pair.

Then the Bregman divergence term is calculated to classify samples according to their distance to the class means.

It is worth mentioning that different kinds of layers other than fullyconnected layers for the metric function are adaptable to our scheme.

We test our model and compare with existing algorithms for 1-way 5-shot and 5-way 5-shot problems.

We use metavalidation set to determine the best model to use in the test case.

The accuracies for each task are given in TAB0 .

Despite the fact that these are our preliminary results, they are comparable with the previous models.

We define a way to measure the convexity we achieve for φ.

We select random pairs and divide the line connecting the pairs into 100 segments with 101 points.

We then re-sample new pairs from these points and record how much L c deviates from the convexity constraint.

We take the overall average deviation of all pairs.

Our results for Omniglot dataset are: (i) For training, we obtain 8, 4x10 −8 ± 2x10 −8 . (ii) For testing, we obtain 6, 7x10 −3 ± 5x10 −1 .

We will run more tests for the convexity measure and analyze its behavior under different problem configurations and various modifications to the network, e.g., sensitivity to hyperparameters and different architectures.

In this paper we proposed an alternative method for few shot learning.

Our method is based on two jointly learnable functions: a nonlinear embedding function followed by a metric learning function.

Our metric function learns a suitable Bregman divergence via approximating a convex function, with a motivation that using Bregman divergences allows the mean to be the most representative point for the relevant class.

We achieve comparable preliminary results sufficient to validate their potential.

We plan to further investigate our model and do more extensive parameter and architecture search to improve our results.

We can utilize different constraints or optimization methods to satisfy convexity, or apply an alternating training between embedding and metric function.

We can also carry our method to other problems that contains convexity such as semi-supervised learning and structured prediction.

<|TLDR|>

@highlight

Bregman divergence learning for few-shot learning. 
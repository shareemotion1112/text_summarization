We study many-class few-shot (MCFS) problem in both supervised learning and meta-learning scenarios.

Compared to the well-studied many-class many-shot and few-class few-shot problems, MCFS problem commonly occurs in practical applications but is rarely studied.

MCFS brings new challenges because it needs to distinguish between many classes, but only a few samples per class are available for training.

In this paper, we propose ``memory-augmented hierarchical-classification network (MahiNet)'' for MCFS learning.

It addresses the ``many-class'' problem by exploring the class hierarchy, e.g., the coarse-class label that covers a subset of fine classes, which helps to narrow down the candidates for the fine class and is cheaper to obtain.

MahiNet uses a convolutional neural network (CNN) to extract features, and integrates a memory-augmented attention module with a multi-layer perceptron (MLP) to produce the probabilities over coarse and fine classes.

While the MLP extends the linear classifier, the attention module extends a KNN classifier, both together targeting the ''`few-shot'' problem.

We design different training strategies of MahiNet for supervised learning and meta-learning.

Moreover, we propose two novel benchmark datasets ''mcfsImageNet'' (as a subset of ImageNet) and ''mcfsOmniglot'' (re-splitted Omniglot) specifically for MCFS problem.

In experiments, we show that MahiNet outperforms several state-of-the-art models on MCFS classification tasks in both supervised learning and meta-learning scenarios.

The representation power of deep neural networks (DNN) has dramatically improved in recent years, as deeper, wider and more complicated DNN architectures BID5 BID6 have emerged to match the increasing computation power of new hardwares.

Although this brings hope for complex tasks that could be hardly solved by previous shallow models, more training data is usually required.

Hence, the scarcity of annotated data has become a new bottleneck for training more powerful DNNs.

For example, in image classification, the number of candidate classes can easily range from hundreds to tens of thousands (i.e., many-class), but the training samples available for each class can be less than 100 (i.e., few-shot).

Additionally, in life-long learning, models are always updated once new training data becomes available, and those models are expected to quickly adapt to new classes with a few training samples.

This "many-class few-shot" problem is very common in various applications, such as image search, robot navigation and video surveillance.

Although enormous previous works have shown the remarkable power of DNN when "many-class many-shot" training data is available, their performance degrades dramatically when each class only has a few samples available for training.

In practical applications, acquiring samples of rare species is usually difficult and often expensive.

In these few-shot scenarios, the model's capacity cannot be fully utilized, and it becomes much harder to generalize the model to unseen data.

Recently, several approaches have been proposed to address the few-shot learning problem.

Most of them are based on the idea of "meta-learning", which trains a meta-learner that can generalize to different tasks.

For classification, each task targets a different set of classes.

Meta-learning can be categorized into two types: methods based on "learning to optimize", and methods based on metric learning.

The former type adaptively modifies the optimizer (or some parts of it) applied to the training process.

It includes methods that incorporate an RNN meta-learner BID0 BID13 BID16 , and model-agnostic meta-learning (MAML) methods aiming to learn a … Figure 1 : The MCFS problem with class hierarchy information.

There are a few coarse classes (blue), but each coarse class contains a large number of fine classes (red), and the total number of fine classes is large.

Only a few training samples are available for each fine class.

The goal is to train a classifier to generate a prediction over all fine classes.

In meta-learning, each task is an MCFS problem sampled from a certain distribution.

The meta-learner's goal is to help train a classifier for any sampled task with better adaptation to few-shot data.generally compelling initialization BID4 .

The latter type learns a similarity/distance metric BID22 or a support set of samples BID20 ) that can be generally used to build KNN classifiers for different tasks.

Instead of using meta-learning, some other approaches, such as BID2 , address the few-shot learning problem through data augmentation by generating artificial samples for each class.

However, most existing few-shot learning approaches only focus on "few-class" case (e.g., 5 or 10) per task, and performance usually collapses when the number of classes grows to hundreds or thousands.

This is because the samples per class no longer provide enough information to distinguish them from other possible samples within a large number of other classes.

And, in real-world problems, tasks are usually complicated involving many classes.

Fortunately, in practice, class hierarchy is usually available or cheaper to obtain.

As shown in Figure 1, coarse class labels might reveal the relationships among the targeted fine classes.

Moreover, the samples per coarse class are sufficient to train a reliable coarse classifier, whose predictions are able to narrow down the candidates for fine classes.

For example, a sheepdog with long hair could be easily mis-classified as mop when training samples of sheepdog are insufficient.

However, if we could train a reliable dog classifier, it would be much simpler to predict an image as a sheepdog than a mop given a correct prediction of the coarse class as "dog".

Hence, class hierarchy might provide weakly supervised information to help solve the "many-class few-shot (MCFS)" problem.

In this paper, we study how to explore the class hierarchy to solve MCFS problem in both traditional supervised learning and in meta-learning.

We develop a DNN architecture "memory-augmented hierarchical-classification networks (MahiNet)" that can be applied to both learning scenarios.

MahiNet uses a CNN, i.e., ResNet by BID5 , as a backbone network to extract features from raw images.

The CNN feeds features into coarse-class and fine-class classifiers, and the results are combined to produce the final prediction according to fine classes as probabilities.

In this way, both the coarse-class and the fine-class classifiers mutually help each other within MahiNet: the former helps to narrow down the candidates for the latter, while the latter provides multiple attributes per coarse class that can regularize the former.

This design leverages the relationship between fine classes, and mitigates the difficulty caused by "many class" problem.

To the best of our knowledge, we are the first to successfully employ the class hierarchy information to improve few-shot learning.

Previous works BID17 ) cannot achieve improvement after using the same information.

To address the "few-shot" problem, we apply two types of classifiers in MahiNet, i.e., MLP and Knearest neighbor (KNN), which have advantages in many-shot and few-shot situations, respectively.

We always use MLP for coarse classification, and KNN for fine classification.

With a sufficient amount of data in supervised learning, MLP is combined with KNN for fine classification; and in meta-learning when less data is available, we also use KNN for coarse classification to assist MLP.

In TAB0 , we provide a brief comparison of MahiNet with other popular models on the learning scenarios they excel.

To make the KNN learnable and more adaptive to classes with few-shot data, we use an attention module to learn the similarity/distance metric used in KNN, and a re-writable memory of limited size to store and update KNN support set during training.

In supervised learning, it is necessary to maintain and update a relatively small memory (7.2% of the dataset) by selecting a few samples, because conducting a KNN search over all available training samples is too computationally expensive in computation.

In meta-learning, the attention module can be treated as a meta-learner that learns a universal similarity metric for different tasks.

We extract a large subset of ImageNet BID1 ) "mcfsImageNet" as a benchmark dataset specifically designed for MCFS problem.

It contains 139,346 images from 77 non-overlapping coarse classes composed of 754 randomly sampled fine classes, each has only ∼ 180 images.

Imbalance between the different classes are preserved to reflect the imbalance in practical problems.

We further extract "mcfsOmniglot" from Omniglot BID10 for the same purpose.

We will make them publicly available later.

In experiments on these two datasets, MahiNet outperforms the widely used ResNet BID5 in supervised learning.

In meta-learning scenario where each task convers many classes, it shows more promising performance than popular few-shot methods including prototypical networks BID20 and relation networks BID26 .2 MEMORY-AUGMENTED HIERARCHICAL-CLASSIFICATION NETWORK

We study supervised learning and meta-learning given a training set of n samples DISPLAYFORM0 , where each sample x i ∈ X is associated with a fine-class label y i ∈ Y and a coarse-class label z i ∈ Z, and is sampled from a data distribution D, i.e., (x i , y i , z i ) ∼ D. Here, Y denotes the set of all the fine classes, and Z denotes the set of all the coarse classes.

To define a class hierarchy for Y and Z, we further assume that each coarse class z ∈ Z covers a subset of fine classes Y z , and that distinct coarse classes are associated with disjoint subsets of fine classes, i.e., for any z 1 , z 2 ∈ Z, we have Y z1 ∩ Y z2 = ∅. Our goal is fine-class classification by using the class hierarchy information.

In particular, the supervised learning in this case can be formulated as:min DISPLAYFORM1 where Θ is the model parameters.

In practice, we solve the corresponding empirical risk minimization (ERM) during training, i.e., DISPLAYFORM2 In contrast, meta-learning aims to maximize the expectation of the prediction likelihood of a task drawn from a distribution of tasks.

Specifically, we assume that the subset of fine classes T for each task is sampled from a distribution T , and the problem is formulated as min DISPLAYFORM3 where D T refers to the distribution of samples with label y i ∈ T .

The corresponding ERM is DISPLAYFORM4 where T is a task (defined by a subset of fine classes) sampled from distribution T , and D T is a training set sampled from D T .To leverage the coarse class information of z, we write Pr(y|x; Θ) in Eq.(1) and Eq. (3) as DISPLAYFORM5 where Θ f and Θ c are the model parameters for fine classifier and coarse classifier, respectively 1 .

Accordingly, given a specific sample (x i , y i , z i ) with its ground truth labels for coarse and fine Step 1.

Clustering: r clusters per class (row) DISPLAYFORM6 Step 2. replace the r memory slots (with the smallest utility rates) by the r cluster centroids memory utility rate Figure 2 : Left: MahiNet.

The final fine-class prediction combines predictions based on both fine classes and coarse classes, each of which is produced by an MLP classifier or/and an attention-based KNN classifier.

Top right: KNN classifier with learnable similarity metric and updatable support set.

Attention provides a similarity metric a j,k between each input sample fi and a small support set per class stored in memory M j,k .

The learning of KNN classifier aims to optimize 1) the similarity metric parameterized by the attention, detailed in Sec. 2.3; and 2) a small support set of feature vectors per class stored in memory, detailed in Sec. 2.4.

Bottom right: The memory update mechanism.

In meta-learning, the memory stores the features of all training samples of a task.

In supervised learning, the memory is updated during training as follows: for each sample xi within an epoch, if the KNN classifier produces correct prediction, fi will be merged into the memory; otherwise, fi will be written into a "cache".

At the end of each epoch, we apply clustering to the samples per class stored in the cache, and use the resultant centroids to replace r slots of the memory with the smallest utility rate.classes, we can write Pr(y i |x i ; Θ) in Eq. FORMULA2 and Eq. (4) as follows.

DISPLAYFORM7 (6) Suppose that a DNN model already produces a logit a y for each fine class y, and a logit b z for each coarse class z, the two probabilities in the right hand side of Eq. (6) are computed by applying softmax function to the logit values in the following way.

DISPLAYFORM8 Therefore, we integrate both the fine-class label and coarse-class label in an ERM, whose goal is to maximize the likelihood of the ground truth fine-class label.

Given a DNN that can produce two logit vectors a and b for fine class and coarse class, we can train it for supervised learning or meta-learning by solving the ERM problems in Eq. FORMULA2 or Eq. (4) (with Eq. (6) and Eq. (7) plugged in).

To address MCFS problem in both supervised learning and meta-learning scenarios, we developed a universal model, MahiNet, as in Figure 2 .

MahiNet uses a CNN to extract features from raw inputs, and then applies two modules to produce coarse-class prediction and fine-class prediction, respectively.

Each module includes one or two classifiers: either an MLP or an attention-based KNN classifier or both.

Intuitively, MLP performs better when data is sufficient, while the KNN classifier is more stable in few-shot scenario.

Hence, we always apply MLP to coarse prediction and apply KNN to fine prediction.

In addition, we use KNN to assist MLP for coarse module in metalearning, and use MLP to assist KNN for fine module in supervised learning.

In the attention-based KNN classifier, an attention module is trained to compute the similarity between two samples, and a re-writable memory is maintained with a highly representative support set for KNN prediction.

Our method for learning a KNN classifier combines the ideas from two popular meta-learning methods, i.e., matching networks BID22 ) that aims to learn a similarity metric, and prototypical networks BID20 ) that aims to find a representative center per class for NN search.

However, our method relies on an augmented memory rather than a bidirectional RNN for retrieving of NN in matching networks.

In contrast to prototypical networks, that only have one prototype per class, we allow multiple prototypes as long as they can fit in the memory budget.

Together these two mechanisms prevent the confusion caused by subtle differences between classes in many-class scenario.

Notably, MahiNet can also be extended to "life-long learning" given this memory updat-

Input: DISPLAYFORM0 and θ

; Hyper-parameters: memory update parameters r, γ, µ and η; learning rate and its scheduler; 1: while no converge do 2:for mini-batch {(xi, yi, zi)}i∈B in D do 3:Compute fine-class logits a and coarse-class logits b from the outputs of MLP/KNN classifiers; 4:Apply one step of mini-batch SGD for ERM in Eq. (2) (with Eq. (6) and Eq. (7) plugged in); 5:for sample in the mini-batch do 6:Update the memory M according to Eq. (11); 7:Update the utility rate U according to Eq. (13); 8:Expand the feature cache C according to Eq. (12); 9:end for 10:end for 11:for each fine class j in Y do 12:Fine the indexes of the r smallest values in Uj, denoted as {k1, k2, ..., kr}; 13:Clustering of the feature vectors within cache Cj to r clusters with centroids {c1, c2, ..., cr}; 14:Replace the r memory slots by centroids: DISPLAYFORM0 end for 16: end while ing mechanism.

We do not adopt the architecture used in BID14 since it requires the representations of all historical data to be stored.

In MahiNet, we train an attention module to compute the similarity used in the KNN classifier.

The attention module learns a distance metric between the feature vector f i of a given sample x i and any feature vector from the support set stored in the memory.

Specifically, we use the dot product attention similar to the one adopted in BID21 for supervised learning, and use an Euclidean distance based attention for meta-learning, following the instruction from BID20 .

Given a sample x i , we compute a feature vector f i ∈ R d by applying a backbone CNN to x i .

In the memory, we maintain a support set of m feature vectors for each class, i.e., M ∈ R C×m×d , where C is the number of classes.

The KNN classifier produces the class probabilities of x i by first calculating the attention scores between f i and each feature vector in the memory, as follows.

DISPLAYFORM0 where g and h are learnable transformations for f i and the feature vectors in the memory.

We select the K nearest neighbors of f i among the m feature vectors for each class j, and compute the sum of their similarity scores as the attention score of f i to class j, i.e., DISPLAYFORM1 We usually find K = 1 is sufficient in practice.

The predicted class probability is derived by applying a softmax function to the attention scores of f i over all C classes, i.e., DISPLAYFORM2

Ideally, the memory M ∈ R C×m×d can store all available training samples as the support set of the KNN classifier.

In meta learning, in each episode, we sample a task with C classes and m training samples per class, and store them in the memory.

Due to the small amount of training data for each task, we can store all data in the memory.

In supervised learning, we only focus on one task, which is possible to have a large training set that cannot be entirely stored in the memory.

Hence, we set up a budget hyper-parameter m for each class.

m is the maximal number of feature vectors to be stored for one class.

Moreover, we develop a memory update mechanism to maintain a small memory with diverse and representative feature vectors (t-SNE visualization can be found in Figure 4 in Appendix E).

Intuitively, it can choose to forget or merge feature vectors that are no longer representative, and select new important feature vectors into memory.

Sample a task T ∼ T as a subset of fine classes T ⊆ Y.

for class j in T do 4:Randomly sample ns data points of class j from D to be the support set Sj of class j.

Randomly sample nq data points of class j from D\Sj to be the query set Qj of class j. 6: end for 7:for mini-batch from Q do 8:Compute fine-class logits a and coarse-class logits b from the outputs of MLP/KNN classifiers; 9:Apply one step of mini-batch SGD for ERM in Eq. (4) (with Eq. (6) and Eq. (7) plugged in); 10:end for 11: end whileWe will show later in experiments that a small memory can result in sufficient improvement, while the time cost of memory updating is negligible.

During training, for the data that can be correctly predicted by the KNN classifier, we merge its feature with corresponding slots in the memory by computing their convex combination, i.e., DISPLAYFORM0 where y i is the ground truth label, and γ = 0.95 is a combination weight that works well in most of our empirical studies; for input feature vector that cannot be correctly predicted, we write it to a cache C = {C 1 , ..., C C } that stores the candidates written into the memory for the next epoch, i.e., DISPLAYFORM1 Concurrently, we record the utility rate of the feature vectors in the memory, i.e., how many times each feature vector being selected into the K nearest neighbor during the epoch.

The rates are stored in a matrix U ∈ R C×m , and we update it as follows.

DISPLAYFORM2 where µ ∈ (1, 2) and η ∈ (0, 1) are hyper-parameters.

At the end of each epoch, we cluster the feature vectors per class in the cache, and obtain r cluster centroids as the candidates for the memory update in the next epoch.

Then, for each class, we replace r feature vectors in the memory that have the smallest utility rate with the r cluster centroids.

As shown in the network structure in Figure 2 , in supervised learning and meta learning, we use different combinations of MLP and KNN to produce fine-class and coarse-class predictions.

The classifiers are combined by summing up their logits for each class, and a softmax function is used to generate the class probabilities.

Assume the MLP classifiers for the coarse classes and the fine classes are φ(·; θ According to Sec. 2.1, we train MahiNet for supervised learning by solving the ERM problems in Eq. (2) and by solving Eq. (4) for meta-learning.

As previously mentioned, the logits (for either fine classes or coarse classes) used in those ERM problems are obtained by summing up the logits produced by the corresponding combination of classifiers.

Training MahiNet for Supervised learning.

In supervised learning, the memory update relies heavily on the clustering of the merged feature vectors in the cache.

To achieve relatively highquality feature vectors, We first pretrain the CNN+MLP model by using the standard backpropagation to minimize the sum of cross entropy loss on both coarse-classes and fine-classes and then fine-tune the whole model (including the fine-class KNN classifier) with memory updates.

The training procedure of the fine-tune stage is explained in Alg.

1.

Training MahiNet for Meta-learning.

In meta learning, the memory is constant and stores features extracted from the support set for KNN classifier.

The detailed training procedure can be found in Alg.

2.

In summary, we sample each training task by randomly sampling a subset of fine classes, and then randomly sample a support set S and a query set Q. We store the CNN feature vectors of S in the memory, and train MahiNet to produce correct predictions for the samples in Q. When sampling the training/test tasks, we allow new fine classes that were not covered in any training task to appear as test tasks, but the ground set of the coarse classes is fixed for both training and test.

Hence, every coarse class appearing in any test task has been seen in previous training, but the corresponding fine classes belonging to this coarse class in training and test tasks can vary.

We propose two benchmark datasets specifically for MCFS Problem: mcfsImageNet & mcfsOmniglot, and compare them with several existing datasets in TAB2 .

Please see more details in Appendix A. Our following experimental study focuses on these two datasets.

Experiments on mcfsImageNet.

We use ResNet18 for the backbone CNN.

The transformations g and h in the attention module are two fully connected layers followed by group normalization BID25 ) with a residual connection.

See more detailed parameter choices in Appendix B. TAB3 compares MahiNet with the supervised learning model (i.e., ResNet18) and meta learning model (i.e., prototypical networks).

The results show that MahiNet outperforms the specialized models, such as ResNet18 in MCFS scenario.

Prototypical Net is a meta-learning model designed to solve few-shot classification problems.

We train it in a supervised learning manner (i.e., on a single task with many classes and relatively much more samples per class), and include it in the comparison to test its performance on MCFS problem.

Prototypical network fails to solve MCFS problem in the supervised learning scenario.

To separately measure the contribution of the class hierarchy and the attention-based KNN classifier, we conduct an ablation study that removes the KNN classifier from MahiNet.

The results show that MahiNet outperforms ResNet18 even when only using the extra coarse-label information during training, and that using a KNN classifier further improves the performance.

For each epoch, the average clustering time is 30s and is only 7.6% of the total epoch time (393s).

Within an epoch, the memory update time (0.02s) is only 9% of the total iteration time (0.22s).

Experiments on mcfsImageNet.

We use the same backbone CNN, g, and h as in supervised learning.

In each task, we sample the same number of classes for training and test, and follow the training procedure in BID20 .

More detailed parameters can be found in Appendix B. Test accuracy is reported as the averaged over 600 test episodes along with the corresponding 95% confidence intervals are reported.

In the first row, "n-k" represents n-way (class) k-shot.

Mem-1, Mem-2, and Mem-3 indicate 3 different kinds of memory.

In 50-way experiments, Relation Net stops to improve after the first few iterations and fails to achieve comparable performance (more details in Appendix D).

TAB4 shows that MahiNet outperforms the supervised learning baseline (ResNet18) and the metalearning baseline (Prototypical Net).

For ResNet18, we follow the fine-tune baseline in BID4 .

To evaluate the contributions of each component in MahiNet, we show results of several variants in TAB4 .

"Attention" indicates parametric functions for g and h, otherwise using identity mapping.

"Hierarchy" indicates the assist of class hierarchy.

For a specific task, "Mem-1" stores the average feature of all training samples for each class; "Mem-2" stores all features of the training samples; "Mem-3" is the union of "Mem-1" and "Mem-2".

TAB4 implies: (1) Class hierarchy information can incur steady performance across all tasks; (2) Combining "Mem-1" and "Mem-2" outperforms using either of them independently; (3) Attention should be learned with class hierarchy in MCFS problem.

Because the data is usually insufficient to train a reliable similarity metric to distinguish all fine classes, but distinguishing the fine classes in each coarse class is much easier.

We conduct experiments on the secondary benchmark mcfsImageNet.

We use the same training setting as for mcfsImageNet.

Following BID18 , mcfsOmniglot is augmented with rotations by multiples of 90 degrees.

We do not use ResNet18 on mcfsOmniglot, since mcfsOmniglot is a 28 × 28 small dataset, which would be easy for ResNet18 to overfit.

Therefore, we use four consecutive convolutional layers as the backbone CNN and compare MahiNet with prototypical networks as in TAB6 .

We do ablation study on MahiNet with/without hierarchy and MahiNet with different kinds of memory.

"Mem-3", i.e., the union of "Mem-1" and "Mem-2", outperforms "Mem-1", and "Attention" mechanism can improve the performance.

Additionally, MahiNet outperforms other compared methods, which indicates the class hierarchy assists to make more accurate predictions.

In summary, experiments on the small-scale and large-scale datasets show that class hierarchy brings a stable improvement.

ImageNet BID1 ) may be the most widely used large-scale benchmark dataset for image classification.

However, although it provides hierarchical information about class labels, it cannot be directly used to test the performance of MCFS learning methods.

Because a fine class may belong to multiple coarse classes, and in MCFS problem, each sample has only one unique coarse-class label.

In addition, ImageNet does not satisfy the criteria of "few-shot" per class.

miniImageNet BID22 has been widely used as a benchmark dataset in meta-learning community to test the performance on few-shot learning task.

miniImageNet is a subset of data extracted from ImageNet; however, its data are sourced from only 80 fine classes, which is not "many-class" nor does this carry a class hierarchy.

Hence, to develop a benchmark dataset specifically for the purpose of testing the performance of MCFS learning methods, we extracted a subset of images from ImageNet and created a dataset called "mcfsImageNet".

TAB2 compares the statistics of mcfsImageNet with several benchmark datasets, and more details of the class hierarchies in mcfsImageNet are given in the Appendix F.Comparing to the original ImageNet, we avoided selecting the samples that belong to more than one coarse classes into mcfsImageNet to meet the class hierarchy requirements of MCFS problem, i.e., each fine class only belongs to one coarse class.

Compared to miniImageNet, mcfsImageNet is about 5× larger, and covers 754 fine classes -many more than the 80 fine classes in miniImageNet.

Moreover, on average, each fine class only has ∼ 185 images for training and test, which is typical MCFS scenarios.

Additionally, the number of coarse classes in mcfsImageNet is 77, which is many less than 754 of the fine classes.

This is consistent with the data properties found in many practical applications, where the coarse-class labels can only provide weak supervision, but each coarse class has sufficient training samples.

Further, we avoided selecting coarse classes which were too broad or contained too many very different fine classes.

For example, the "Misc" class in ImageNet has 20400 sub-classes, and includes both animal (3998 sub-classes) and plant (4486 sub-classes).Omniglot BID10 ) is a small hand-written character dataset with two levels.

However, new coarse classes appear in the test set, which is inconsistent with our MCFS settings (all the coarse classes are exposed in training, but new fine classes can appear during test).

As a result, we re-split Omniglot to fulfill the MCFS problem requirement and the class hierarchy information are listed in Appendix G.

Setup for the supervised learning.

We use ResNet18 BID5 for the backbone CNN.

The transformation functions g and h in the attention module are two fully connected layers followed by group normalization BID25 ) with a residual connection.

We set the memory size to m = 12 and the number of clusters to r = 3, which can achieve a better trade-off between memory cost and performance.

Batch normalization BID7 is applied after each convolution and before activation.

During pre-training, we apply the cross entropy loss on the probability predictions in Eq. (7).

During fine-tuning, we fix the θ CN N , θ

, and θ

to ensure the fine-tuning process is stable.

We use SGD with a mini-batch size of 128 and a cosine learning rate scheduler with an initial learning rate 0.1.

µ = 1.05, η = 0.95, a weight decay of 0.0001, and a momentum of 0.9 are used.

We train the model for 100 epochs during pre-training and 90 epochs for the fine-tuning.

Setup for the meta learning.

We use the same backbone CNN, g, and h as in supervised learning.

In each task, we sample the same number of classes for training and test, and follow the training procedure in BID20 .

We set an initial learning rate to 10 −3 and reduce it by a factor 2× every 10k iterations.

Our model is trained by Adam (Kingma & Ba, 2015) with a mini-batch size of 128, a weight decay of 0.0001, and a momentum of 0.9.

We train the model for 25k iterations in total.

For class hierarchy, the objective function is the sum of the softmax with cross entropy losses on the coarse class and on the fine class, respectively.

Few-shot learning has a long history.

Before deep learning, generative models BID3 are trained to provide a global prior knowledge for solving the one-shot learning problem.

However, with the advent of deep learning techniques, some recent approaches BID24 BID11 use generative models to encode specific prior knowledge, such as strokes and patches.

More recently, BID2 and BID23 have applied hallucinations to training images and to generate more training samples, which converts a few-shot problem to a many-shot problem.

Meta-learning has been used in attempts to solve the few-shot learning problems.

Meta learning was first proposed in the last century BID15 BID19 , and ) but has recently seen some significant improvements.

For example, BID12 proposed a dataset of characters for meta-learningn while BID9 extended this idea into a Siamese network.

A more challenging dataset BID16 BID22 was introduced later.

Researchers have also studied RNN and attention based method to overcome the few-shot problem.

More recently, BID20 is proposed based on a metric learning equipped KNN.

In contrast, BID4 based their approach on the second order optimization.

BID14 uses temporal convolution to address the few-shot image recognition.

However, unlike above methods, our model leverages the class hierarchy information, and can be easily applied to both the supervised learning and meta-learning scenarios.

Relation network with class hierarchy.

We train relation network with class hierarchy in the similar manner as in MahiNet.

The results are shown in TAB7 .

It demonstrates that the class hierarchy also improves the accuracy of relation network by more than 1%, which verifies the advantage of using class hierarchy in other models besides MahiNet.

Relation network in high way setting.

For relation network in high way settings, we found that the network is easy to be stuck into a suboptimal solution.

After first few iterations, the training loss stays in a high level and the training accuracy stays in a low level.

We demonstrate the training loss and training accuracy for the first 100 iterations under different learning rate as Figure 3 .

The training loss and accuracy keep the same value after 100 iterations.

Visualization.

In order to show how representative and diverse the feature vectors selected into memory slots are, we visualize feature vectors in the memory and the rest image feature vectors in t-SNE in Figure 4 .

In particular, we randomly sample 50 fine classes marked by different colors.

Within every class, we show both the selected feature vectors in memory and feature vectors of other images from the same class.

It shows that the small number of highly selected feature vectors in memory are diverse and sufficiently representative of the whole class.

Memory Cost.

In experiments of supervised learning, the memory size required by MahiNet is only 754 × 12/125321 = 7.2% (12 samples per class for all the 754 fine classes, while the training set includes 125, 321 images in total) of the memory needed to store the whole training set.

We also tried to increase the memory size to about 10%, but the resultant improvement on performance is negligible.

In each task of meta learning, since every class only has few-shot samples, the memory

The hierarchy of coarse and fine classes is shown below.

Every key (marked in bold) in the dictionary is a coarse class, the value is a list of the fine classes of this coarse classes.

<|TLDR|>

@highlight

A memory-augmented neural network that addresses many-class few-shot problem by leveraging class hierarchy in both supervised learning and meta-learning.

@highlight

This paper presents methods for adding inductive bias to a classifier through coarse-to-fine prediction along a class hierarchy and learning a memory-based KNN classifier that keeps track of mislabeled instances during learning.

@highlight

This paper formulates the many-class-few-shot classification problem from a supervised learning perspective and a meta-learning perspective.
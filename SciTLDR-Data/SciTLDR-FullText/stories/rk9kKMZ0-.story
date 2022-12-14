Determining the optimal order in which data examples are presented to Deep Neural Networks during training is a non-trivial problem.

However, choosing a non-trivial scheduling method may drastically improve convergence.

In this paper, we propose a Self-Paced Learning (SPL)-fused Deep Metric Learning (DML) framework, which we call Learning Embeddings for Adaptive Pace (LEAP).

Our method parameterizes mini-batches dynamically based on the \textit{easiness} and \textit{true diverseness} of the sample within a salient feature representation space.

In LEAP, we train an \textit{embedding} Convolutional Neural Network (CNN) to learn an expressive representation space by adaptive density discrimination using the Magnet Loss.

The \textit{student} CNN classifier dynamically selects samples to form a mini-batch based on the \textit{easiness} from cross-entropy losses and \textit{true diverseness} of examples from the representation space sculpted by the \textit{embedding} CNN.

We evaluate LEAP using deep CNN architectures for the task of supervised image classification on MNIST, FashionMNIST, CIFAR-10, CIFAR-100, and SVHN.

We show that the LEAP framework converges faster with respect to the number of mini-batch updates required to achieve a comparable or better test performance on each of the datasets.

The standard method to train Deep Neural Networks (DNNs) is stochastic gradient descent (SGD) which employs backpropagation to compute gradients.

It typically relies on fixed-size mini-batches of random samples drawn from a finite dataset.

However, the contribution of each sample during model training varies across training iterations and configurations of the model's parameters BID15 .

This raises the importance of data scheduling for training DNNs, that is, searching for an optimal ordering of training examples which are presented to the model.

Previous studies on Curriculum Learning (Bengio et al., 2009, CL) show that organizing training samples based on the ascending order of difficulty can favour model training.

However, in CL, the curriculum remains fixed over the iterations and is determined without any knowledge or introspection of the model's learning.

Self-Paced Learning BID14 ) presents a method for dynamically generating a curriculum by biasing samples based on their easiness under the current model parameters.

This can lead to a highly imbalanced selection of samples, i.e. very few instances of some classes are chosen, which negatively affects the training process due to overfitting.

BID19 propose a simple batch selection strategy based on the loss values of training data for speeding up neural network training.

However, their results are limited and the approach is time-consuming, as it achieves high performance on MNIST, but fails on CIFAR-10.

Their work reveals that selecting the examples to present to a DNN is non-trivial, yet the strategy of uniformly sampling the training data set is not necessarily the optimal choice.

BID12 show that partitioning the data into groups with respect to diversity and easiness in their Self-Paced Learning with Diversity (SPLD) framework, can have substantial effect on training.

Rather than constraining the model to limited groups and areas, they propose to spread the sample selection as wide as possible to obtain diverse samples of similar easiness.

However, their use of K-Means and Spectral Clustering to partition the data into groups can lead to sub-optimal clustering results when learning non-linear feature representations.

Therefore, learning an appropriate metric by which to capture similarity among arbitrary groups of data is of great practical importance.

Deep Metric Learning (DML) approaches have recently attracted considerable attention and have been the focus of numerous studies BID1 ; BID23 ).

The most common methods are supervised, in which a feature space in which distance corresponds to class similarity is obtained.

The Magnet Loss BID21 presents state-of-the-art performance on fine-grained classification tasks.

BID26 show that it achieves state-of-the-art on clustering and retrieval tasks.

This paper makes two key contributions toward scheduling data examples in the mini-batch setting:??? We propose a general sample selection framework called Learning Embeddings for Adap- tive Pace (LEAP) that is independent of model architecture or objective, and learns when to introduce certain samples to the DNN during training.??? To our knowledge, we are the first to leverage metric learning to improve self-paced learning.

We exploit a new type of knowledge -similar instance-level samples are discovered through an embedding network trained by DML in concert with the self-paced learner.2 RELEVANT WORK

The perspective of "starting small and easy" for structuring the learning regimen of neural networks dates back decades to BID4 .

Recent studies show that selecting a subset of good samples for training a classifier can lead to better results than using all the samples BID17 BID15 .

Pioneering work in this direction is Curriculum Learning BID2 , which introduced a heuristic measure of easiness to determine the selection of samples from the training data.

By comparison, SPL BID14 quantifies the easiness by the current sample loss.

The training instances with loss values larger than a threshold, ??, are neglected during training and ?? dynamically increases in the training process to include more complex samples, until all training instances are considered.

This theory has been widely applied to various problems, including dictionary learning for image classification BID29 , object detection BID22 , multimedia event detection (Jiang et al., 2014a) , long-term tracking BID28 , visual tracking BID10 and medical imaging analysis .

In SPLD BID12 , training data are pre-clustered in order to balance the selection of the easiest samples with a sufficient inter-cluster diversity.

However, the clusters and the feature space are fixed: they do not depend on the current self-paced training iteration.

Adaptation of this method to a deep-learning scenario, where the feature space changes during learning, is non-trivial.

Our self-paced sample selection framework aims at a similar goal but the diversity of samples is obtained with a DML approach to adaptively sculpt a representation space by autonomously identifying and respecting intra-class variation and inter-class similarity.

Deep metric learning has gained much popularity in recent years, along with the success of deep learning.

The objective of DML is to learn a distance metric consistent with a given set of constraints, which usually aim to minimize the distances between pairs of data points from the same class and maximize the distances between pairs of data points from different classes.

DML approaches have shown promising results on various tasks, such as semantic segmentation BID5 , visual product search BID13 , face recognition BID23 , feature matching BID3 , fine-grained image classification BID34 , zero-shot learning BID6 and collaborative filtering BID9 .

DML can also be used for challenging, extreme classification settings, where the number of classes is very large and the number of examples per class becomes scarce.

Most of the current methods define the loss in terms of pairs BID27 BID30 , triplets BID23 or n-pair tuples BID25 inside the training mini-batch.

These methods require a separate data preparation stage which has very expensive time and space cost.

Also, they do not take the global structure of the embedding space into consideration, which can result in reduced clustering.

An alternative is the Magnet Loss BID21 and DML via Facility Location BID26 which do not require the training data to be preprocessed in rigid paired format and are aware of the global structure of the embedding space.

Our work employs the Magnet loss to learn a representation space, where we compute centroids on the raw features and then update the learned representation continuously.

To our knowledge, the concept of employing DML for SPL-based DNN training has not been investigated.

Effectively, an end-to-end DML can be constructed to be a feature extractor using a deep CNN which can learn to sculpt an expressive representation space by metric embedding.

We can use this feature representation space which maintains intra-class variations and inter-class similarity to select samples based on the true diverseness and easiness for the student model we want to train.

Our architecture combines the strength of adaptive sampling with that of mini-batch online learning and adaptive representation learning to formulate a representative self-paced strategy in an end-to-end DNN training protocol.

The Learning Embeddings for Adaptive Pace (LEAP) framework consists of a dual DNN setup.

An embedding DNN learns a salient representation space, then transfers its knowledge to the selfpaced selection strategy to train the second DNN, called the student.

In this work, but without loss of generality, we focus on training deep Convolutional Neural Networks (CNNs) for the task of supervised image classification.

More specifically, an embedding CNN is trained alongside the student CNN of ultimate interest ( Figure 1 ).

In this framework, we want to form mini-batches using the easiness and true diverseness as sample importance priors for the selection of training samples.

Given that we are learning the representation space adaptively alongside the student as training progresses, this has negligible computational cost compared to the actual training of the student CNN (see Section 4).FIGURE 1: The LEAP framework, consisting of an embedding CNN that learns a representation for the student CNN to create a self-paced strategy based on easiness and true diverseness as sample importance priors.

We adopt the Magnet loss to learn a representation space because it proved to achieve a higher accuracy on classification tasks, in comparison to margin-based Triplet loss and softmax regression BID21 .

Assuming we have a training set with N input-label pairs D = {x n , y n } N n=1 , the Magnet loss learns the distribution of distances for each example, from K clusters assigned for each class, c, denoted as {I DISPLAYFORM0 The mapping of inputs to representation space are parametrized by f(??; ??), where their representations are defined as r n = {f(x n ; ??)} N n=1 .

The approach then repositions the different cluster assignments using an intermediate K-Means++ clustering BID0 .

Therefore, for each class c, we have, {I The class of representation r and its assigned cluster center are defined as C(r) and ??(r), respectively.

The mini-batches used to train the embedding CNN are constructed iteratively with neighbourhood sampling.

Moreover, a seed cluster is sampled DISPLAYFORM1 .

The losses of each example is stored and the average loss L I of each cluster I is computed during training.

This results in the following stochastic approximation of the Magnet Loss objective: DISPLAYFORM2 where {??} + is the hinge function, ?? ??? R is a scalar, the cluster means approximation DISPLAYFORM3 , and the variance of all samples from their respective centers is given by?? = DISPLAYFORM4

The aim of LEAP can be formally described as follows.

Let us assume that a training set D consisting of N samples, D = {x n } N n=1 is grouped into K clusters for C classes using Algorithm 1.

Therefore, we have DISPLAYFORM0 , where D k corresponds to the k th cluster, n k is the number of samples in each cluster and DISPLAYFORM1 , where DISPLAYFORM2 Non-zero weights of W are assigned to samples that the student model considers "easy" and non-zero elements are distributed across more clusters to increase diversity.

This leads to an objective similar to the one presented in SPLD: DISPLAYFORM3 where ??, ?? are the two pacing parameters for sampling based on easiness and the true diverseness as sample importance priors, respectively.

The negative l 1 -norm: ??? W 1 is used to select easy samples over hard samples, as seen in conventional SPL.

The negative l 2 -norm inherited from the original SPLD algorithm is used to disperse non-zero elements of W across a large number of clusters to obtain a diverse selection of training samples.

The student CNN receives a diverse cluster of samples, the up-to-date model parameters ??, ??, ?? and outputs the optimal of W of min W E(??, W; ??, ??) for extracting the global optimum of this optimization problem.

The detailed algorithm to train the student CNN with LEAP is presented in Algorithm 2.

Step 18 of Algorithm 2 selects the "easy" samples for training when L(y DISPLAYFORM4 DISPLAYFORM5 , ??)) < ?? + ??, which represents the "hard" samples with higher losses.

We select other samples by ranking a sample w.r.t to its loss value within its cluster, denoted by i.

Then, we compare the losses to a threshold ?? + ?? DISPLAYFORM6 Step 18 penalizes samples repeatedly selected from the same cluster, seeing as this threshold decreases as the sample's rank i grows.

All experiments were conducted using the PyTorch framework, while leveraging containerized multi-GPU training on NVIDIA P100 Pascal GPUs through Docker.

We compared our LEAP framework against the original SPLD algorithm and Random sampling on MNIST, FashionMNIST and CIFAR-10.

The ?? and ?? pace parameters are kept consistent between the SPLD strategy in LEAP and the original SPLD algorithm to ensure a fair evaluation.

The embedding CNN is trained asynchronously in parallel with the student CNN.

The computational requirement of training the embedding CNN can be mitigated by leveraging multiprocessing for parallel computing to share data between processes locally using arrays and values.

As a result, at every epoch, the student CNN adaptively selects "easy" samples from K cluster representations generated by the embedding CNN.

In our experiments, we mainly compare convergence in terms of number of mini-batches required to achieve a comparable or better state-of-the-art test performance.

We also visualize the original high-dimensional representations using t-SNE (van der BID31 , where the different colours correspond to different classes and the values to density estimates.

The following sections discuss the experimental setups and results in more detail.

In the experiments for MNIST, we extract the feature embeddings from a LeNet BID16 , as the embedding CNN, and learn a representation space using the Magnet Loss.

The fullyconnected layer of the LeNet is replaced with an embedding layer for compacting the distribution of the learned features for feature similarity comparison using the Magnet Loss.

The student CNN (classifier) was also a LeNet which we then trained with our LEAP framework.

The embedding CNN was trained with randomly sampled mini-batches of size 64 and optimized using Adam with a learning rate of 0.0001.

The results in FIG1 show that the test performance is comparable to that of Random sampling and shows better convergence than the standard SPLD algorithm.

The learned representation space of the MNIST dataset using LeNet is presented in Figure 3b and the training loss in Figure 3a .

The MNIST experiments were primarily carried out to show that the LEAP framework can be deployed as an end-to-end DNN training protocol.

The experiments for FashionMNIST had a similar setup to MNIST, however we use a ResNet-18 BID7 for the classifier.

The embedding CNN remains the same LeNet for feature extraction, and is trained using an identical setup to the MNIST experiments.

The classifier is trained using SGD with Nesterov momentum of 0.9, weight decay of 0.0005 and a learning rate of 0.001.

The FashionMNIST dataset is considered a direct drop-in replacement for the original MNIST dataset, with a training set of 60,000 examples and a test set of 10,000 examples.

Each example is a 28??28 grayscale image, associated with a label from 10 classes.

We performed data augmentation on the training set with normalization, random horizontal flip, random vertical flip, random translation, random crop and random rotation.

In comparison to SPLD and Random sampling, the results in Figure 4 reveal that LEAP converges to a higher test accuracy with a fewer number of mini-batch updates before saturating.

We ran two sets of experiments on CIFAR-10, one with a fixed learning rate and the other with a learning rate scheduler identical to that of the WideResNet BID33 training scheme.

The CIFAR-10 training set was augmented with normalization, random horizontal flip and random crop.

We used VGG-16 as our embedding CNN and ResNet-18 as our classifier in the CIFAR-10 experiments.

We chose ResNet-18 over other architectures because it is faster to train and achieves good performance on CIFAR-10.

Our experiments revealed that VGG-16 BID24 ) learned strong and rich feature representations which yielded the best convergence on the Magnet Loss.

Therefore, we treat the VGG-16 model as a feature extraction engine and use it for feature extraction from CIFAR-10 images, without any fine-tuning.

In the first experiment, the classifier is trained using SGD with a momentum of 0.9, weight decay of 0.0005, a fixed learning rate of 0.001 and batch size of 128.

In the second experiment, the classifier is trained with batch sizes of 128 as well, using SGD with a momentum of 0.9, weight decay of 0.0005 and a starting learning rate of 0.1 which is dropped by a factor of 0.1 at 60, 120 and 160 epochs.

In our experiments, this would translate to 23,400, 46,800, and 62,400 mini-batch updates.

In both experiments, VGG-16 is trained with randomly sampled mini-batches of size 64 and optimized using Adam at a learning rate of 0.0001.At a fixed learning rate of 0.001, training ResNet-18 with the LEAP framework results in a faster convergence to achieve higher test accuracy than either Random sampling or SPLD FIG4 ).

Interestingly, SPLD and Random sampling show comparable results on an average of 5 runs.

This is because SPLD uses K-Means to partition its data into K clusters at the start of training, which would lead to sub-optimal clustering results containing samples that are not of similar-instance level.

As a result, the SPLD would not be selecting diverse samples of similar-instance level when traversing through the different clusters.

On the other hand, our LEAP framework ensures that optimal clustering is achieved using the Magnet Loss and the classifier is able to use the learned representation space adaptively as training progresses.

This ensures that during early stages of training, the minibatches being fed into the student CNN are parameterized with truly diverse and easy samples.

As the model matures, the mini-batches maintain diversity, as well as a mix of easy and hard samples.

The second set of experiments we ran on CIFAR-10 with the learning rate scheduler shows that LEAP converges faster earlier on in training compared to SPLD and Random sampling ( Figure 7 ).

As the learning rate drops by a factor of 0.1 at 23,400, 46,800, and 62,400 mini-batch updates, the classifier under the LEAP training protocol eventually achieves a higher test accuracy than SPLD and Random sampling.

Also, the test loss of LEAP is lower than that of SPLD and Random.

This is expected because LEAP is designed to sample for heterogeneity, thus maximizing the diversity relevant to the learning stage of the student CNN, which also helps prevent overfitting.

The learned feature representation space of the augmented CIFAR-10 data is shown in Figure 8b and the training loss of the embedding CNN is presented in Figure 8a .

We evaluated our LEAP framework on CIFAR-100 dataset with a WideResNet (student CNN) that has a fixed depth of 28, a fixed widening factor of 10, and dropout probability of 0.3.

The embedding CNN was a VGG-16 setup similar to our CIFAR-10 experiments.

The optimizer and learning rate scheduler used for training the WideResNet was identical to the CIFAR-10 experiments.

A data augmentation scheme was not applied on the CIFAR-100 dataset.

The results in FIG7 reveal that on CIFAR-100, a dataset that requires a lot more fine-grained recognition, there is noticeable gain in improvement over Random and SPLD.

We can expect this gain on a more fine-grained dataset because the Magnet loss is a metric embedding technique which performs really well on classification tasks for fine-grained visual recognition BID21 .

This further shows that the combination of a dynamic representation space and self-paced strategy can lead to a noticeable improvement on more complex fine-grained datasets.

The Street View House Numbers (SVHN) dataset BID20 ) is a real-world image dataset with 630,420 RGB images of 32??32 pixels in size, where each image consists of digits that are from one of ten different classes.

A single image may contain multiple digits, and the task is to classify the digit located at the center of the image.

The SVHN dataset is split into the training set, testing set and extra set with 73,257, 26,032, and 531,131 images, respectively.

The student CNN used to train on the SVHN dataset was a WideResNet with a fixed depth of 16, a fixed widening factor of 8, and dropout probability of 0.4.

The SVHN training set and extra set were combined for a total of 604,388 images to train the student CNN for 65 epochs and no data augmentation scheme was applied.

The student model was optimized using SGD with Nesterov momentum of 0.9, weight decay of 0.0005 and batch-size of 128.

The embedding CNN was a VGG-16 setup similar to our CIFAR-10 and CIFAR-100 experiments.

The results in Figure 10 show that LEAP is able to converge faster to a higher test accuracy than Random and SPLD.

The learned feature representation space of the SVHN data is presented in Figure 11b and the training loss of the embedding CNN is presented in Figure 11a .

An important finding is that fusing a salient non-linear representation space with a dynamic learning strategy can help a DNN converge towards an optimal solution.

A random curriculum or a dynamic learning strategy without a good representation space was found to achieve a lower test accuracy or converge more slowly than LEAP.

Biasing samples based on the easiness and true diverseness to select mini-batches shows improvement in convergence to achieve classification performance comparable or better than the baselines, Random and SPLD.

As shown in TAB2 , the student CNN models show increased accuracy on MNIST, Fashion-MNIST, CIFAR-10, CIFAR-100 and SVHN with our LEAP sampling method.

It is to be noted that the LEAP framework improves the performance of complex convolutional architectures which already leverage regularization techniques such as batch normalization, dropout, and data augmentation.

We see that the improvements on coarse-grain datasets such as MNIST, Fashion-MNIST, CIFAR-10, and SVHN are between 0.11 and 0.81 percentage points.

On a fine-grained dataset like CIFAR-100, it is more challenging to obtain a high classification accuracy.

This is because there are a 100 fine-grained classes but the number of training instances for each class is small.

We have only 500 training images and 100 testing images per class.

In addition, the dataset contains images of low quality and images where only part of the object is visible (i.e. for a person, only head or only body).

However, we show that with LEAP, we can attain a significant increase in accuracy by 4.50 and 3.72 percentage points over the baselines SPLD and Random, respectively.

The mix of easy and diverse samples from a more accurate representation space of the data helps select appropriate samples during different stages of training and guide the network to achieve a higher classification accuracy, especially for more difficult fine-grained classifcation tasks.

Experimental results across all datasets (MNIST, Fashion-MNIST, CIFAR-10, CIFAR-100, and SVHN) and sampling methods (LEAP, SPLD, and Random).

The test accuracy (%) results are averaged over five runs, with the exception of CIFAR-100 and SVHN which had four runs each.

"*" indicates that no data augmentation scheme was applied on the dataset.

In cases where the classification dataset is balanced and the classes are clearly identifiable, we showed that our end-to-end LEAP training protocol is practical.

An interesting line of work would be to apply LEAP on more complex real-world classification datasets such as iNaturalist BID8 , where there are imbalanced classes with a lot of diversity and require fine-grained visual recognition.

Another interesting area of application would be learning representations using DML for different computer vision tasks (e.g. human pose estimation, human activity recognition, semantic segmentation, etc.) and fusing a representative SPL strategy to train the student CNN.

We introduced LEAP, an end-to-end representation learning SPL strategy for adaptive mini-batch formation.

Our method uses an embedding CNN for learning an expressive representation space through a DML technique called the Magnet Loss.

The student CNN is a classifier which can exploit this new knowledge from the representation space to place the true diverseness and easiness as sample importance priors during online mini-batch selection.

The computational overhead of training two CNNs can be mitigated by training the embedding CNN and student CNN in parallel.

LEAP achieves good convergence speed and higher test performance on MNIST, FashionMNIST, CIFAR-10, CIFAR-100 and SVHN using a combination of two deep CNN architectures.

We hope this will help foster progress of end-to-end SPL fused DML strategies for DNN training, where a number of potentially interesting directions can be considered for further exploration.

Our framework is implemented in PyTorch and will be released as open-source on GitHub following the review process.

@highlight

LEAP combines the strength of adaptive sampling with that of mini-batch online learning and adaptive representation learning to formulate a representative self-paced strategy in an end-to-end DNN training protocol. 

@highlight

Introduces a method for creating mini batches for a student network by using a second learned representation space to dynamically select examples by their 'easiness and true diverseness'.

@highlight

Experiments the classification accuracy on MNIST, FashionMNIST, and CIFAR-10 datasets to learn a representation with curriculum learning style minibatch selection in an end-to-end framework.
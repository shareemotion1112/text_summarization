Few-shot learning is the process of learning novel classes using only a few examples and it remains a challenging task in machine learning.

Many sophisticated few-shot learning algorithms have been proposed based on the notion that networks can easily overfit to novel examples if they are simply fine-tuned using only a few examples.

In this study, we show that in the commonly used low-resolution mini-ImageNet dataset, the fine-tuning method achieves higher accuracy than common few-shot learning algorithms in the 1-shot task and nearly the same accuracy as that of the state-of-the-art algorithm in the 5-shot task.

We then evaluate our method with more practical tasks, namely the high-resolution single-domain and cross-domain tasks.

With both tasks, we show that our method achieves higher accuracy than common few-shot learning algorithms.

We further analyze the experimental results and show that: 1) the retraining process can be stabilized by employing a low learning rate, 2) using adaptive gradient optimizers during fine-tuning can increase test accuracy, and 3) test accuracy can be improved by updating the entire network when a large domain-shift exists between base and novel classes.

Previous studies have shown that high image classification performance can be achieved by using deep networks and big datasets (Krizhevsky et al., 2012; Simonyan & Zisserman, 2015; He et al., 2016; Szegedy et al., 2015) .

However, the performances of these algorithms rely heavily on extensive manually annotated images, and considerable cost is often incurred in preparing these datasets.

To avoid this problem, few-shot learning, which is a task of learning novel classes using only a few examples, has been actively researched.

However, few-shot learning remains a considerably challenging task in machine learning, and classification accuracy in few-shot tasks is much lower than that of the many-shot regime.

This is because a network pretrained using base classes must adapt to novel classes using only a few examples.

The simplest means of overcoming this difficulty is to fine-tune the network using novel classes.

However, the number of trainable parameters of deep networks is so large that we believe that networks can easily overfit to novel classes if we simply fine-tune the networks using only a few examples.

For example, the number of trainable parameters in the ResNet-152 (He et al., 2016 ) is approximately 60 M, which is much greater than the number of novel examples (e.g., 25 for 5-way 5-shot learning), and this leads us to the idea of overfitting.

Using various sophisticated methods, numerous studies have been conducted to prevent networks from overfitting.

However, the performance of a naive fine-tuning method has not been well investigated, and Chen et al. (2019) has pointed out that performance of this method had been underestimated in previous studies.

Therefore, in this study, we analyze the performance of a fine-tuning method and show that it can achieve higher classification accuracy than common few-shot learning methods and, in some cases, can achieve an accuracy approximating that of the state-of-the-art algorithm.

We also experimentally show that: 1) a low learning rate stabilizes the retraining process, 2) using an adaptive gradient optimizer when fine-tuning the network increases test accuracy, and 3) updating the entire network increases test accuracy when a large domain shift occurs between base and novel classes.

To evaluate accuracy in few-shot image classification tasks, the mini-ImageNet dataset (Vinyals et al., 2016) has been used in many previous studies.

This is a subset of the ImageNet dataset (Deng et al., 2009) in which each image is resized to 84 × 84 to reduce computational cost.the high-resolution mini-ImageNet dataset and cross-domain dataset.

Both datasets contain higherresolution images than the original mini-ImageNet dataset, and the cross-domain dataset represents a greater challenge because base and novel classes are sampled from different datasets.

Thus, a larger domain shift occurs between these classes.

In this study, we evaluate the performance of our method using the high-resolution mini-ImageNet dataset (high-resolution single-domain task) and cross-domain dataset (cross-domain task) as well as the common low-resolution mini-ImageNet dataset (low-resolution single-domain task).

Details of these datasets are provided in Section 2.3.

The main contributions of this study are as follows: 1) We show that in the common low-resolution single-domain task, our fine-tuning method achieves higher accuracy than common few-shot learning algorithms in the 1-shot task and nearly the same accuracy as that of the state-of-the-art method in the 5-shot task.

We also show that our method achieves higher accuracy than common few-shot learning methods both in the high-resolution single-domain and cross-domain tasks.

Note that we do not compare the performance of our method with the state-of-the-art algorithm in the high-resolution single-domain and cross-domain tasks because the performances for these tasks are not reported in the corresponding papers.

2) We further analyze the experimental results and show that a low learning rate stabilizes the relearning process, that test accuracy can be increased by using an adaptive gradient optimizer such as the Adam optimizer, and that updating the entire network can increase test accuracy when a large domain shift occurs.

2 OVERVIEW OF FEW-SHOT LEARNING 2.1 NOTATION Few-shot learning is a task of learning novel classes using only a few labeled examples.

This task is also called N -way K-shot learning, where N denotes the number of novel classes and K is the number of labeled examples per class.

We focus on the 5-way learning task such as in previous studies (Chen et al., 2019; Schwartz et al., 2018) .

Labeled and unlabeled examples of novel classes are called support and query sets, respectively.

A network is pretrained using base classes, which contain numerous labeled examples.

Base and novel classes are mutually exclusive.

Base classes are used for pretraining, and novel classes are used for retraining and testing.

Validation classes are used to determine a learning rate and the number of epochs required to retrain the network.

To date, numerous few-shot learning algorithms have been proposed, and these methods can be roughly classified into three categories: learning discriminative embedding using metric-based classification, learning to learn novel classes, and data-augmentation using synthetic data.

Metric-learning approaches such as MatchingNet (Vinyals et al., 2016) and ProtoNet (Snell et al., 2017) tackle few-shot classification tasks by training an embedding function and applying a differentiable nearest-neighbor method to the feature space using the Euclidean metric.

RelationNet (Sung et al., 2018) was developed to replace the nearest-neighbor method with a trainable metric using convolutional and fully connected (FC) layers and has achieved higher few-shot classification accuracy.

Qi et al. (2018) proposed a method called weight imprinting.

They showed that including normalized feature vectors of novel classes in the final layer weight provides effective initialization for novel classes.

We use the weight-imprinting method to initialize the last FC layer before finetuning the network.

These conventional methods successfully retrain networks using novel classes while preventing overfitting, but we show that few-shot classification performance can be further improved by fine-tuning networks.

Meta-learning-based approaches address the few-shot learning problem by training networks to learn to learn novel classes.

Ravi & Larochelle (2017) focused on the similarity between gradient descent methods and long short-term memory (LSTM) (Hochreiter & Schmidhuber, 1997) , and they achieved a fast adaptation to novel classes by using LSTM to update network weights.

Finn et al. (2017) proposed a method to train a network to obtain easily adaptable parameters so that the network can adapt to novel classes by means of a few gradient steps using only a few examples.

In these algorithms, networks are explicitly trained to learn how to adapt to novel classes.

However, we show that networks pretrained without explicit meta-learning methods can also learn novel classes and achieve high few-shot classification accuracy.

Data-augmentation-based approaches overcome data deficiencies by generating synthetic examples of novel classes.

Some methods synthesize examples of novel classes by applying withinclass differences of base classes to real examples of novel classes (Hariharan & Girshick, 2017; Schwartz et al., 2018) .

Wang et al. (2018) integrated a feature generator using a few-shot learning process and succeeded in generating synthetic data using only a few novel examples.

These methods succeeded in improving the performance of few-shot learning by using synthetic examples for retraining.

Nevertheless, we show that networks can adapt to novel classes by using only naive data-augmentation methods such as image flipping and image jittering.

The mini-ImageNet dataset is a well-known dataset used to evaluate few-shot learning methods.

The dataset was first proposed by Vinyals et al. (2016) , but the train/validation/test split proposed by Ravi & Larochelle (2017) is often used instead.

Therefore, we used this split in this study.

This dataset is a subset of the ImageNet dataset (Deng et al., 2009 ) and contains 100 classes with 600 examples of each class.

The classes are split into 64 base, 16 validation, and 20 novel classes.

Images in this dataset are resized to 84 × 84 to reduce computational cost.

Recently, Chen et al. (2019) used a higher-resolution mini-ImageNet dataset with an image resolution of 224 × 224 to employ deeper networks.

They also revealed that the domain shift that occurs between base and novel classes in the mini-ImageNet dataset is small because the classes are sampled in the same dataset.

The authors proposed the cross-domain dataset, which has a larger domain shift between these classes.

In this dataset, the whole mini-ImageNet dataset is used as a set of base classes, and randomly sampled 50 and 50 classes from the CUB-200-2011 dataset (Wah et al., 2011) are used as validation and novel classes, respectively.

These datasets are more practical because they use high-resolution images, and the cross-domain dataset is more challenging because the domain shift that occurs between base and novel classes is larger.

Therefore, we use the highresolution mini-ImageNet dataset and cross-domain dataset for evaluation as well as the common low-resolution mini-ImageNet dataset.

The ResNet-18/34/50/101/152 (He et al., 2016) and VGG-16 (Simonyan & Zisserman, 2015) without FC layers are used as feature extractors in this study.

Note that the last MaxPool2d layer of the VGG-16 is replaced by the GlobalAveragePool2d layer to support different resolutions of input images.

We also use the simple classifier (i.e., common FC layer) and the normalized classifier.

The technique known as weight imprinting (Qi et al., 2018 ) is used in the normalized classifier; the normalized classifier is illustrated in Figure 1 .

Before fine-tuning the normalized network for novel classes, we initialize classifier weight W by deleting the weight and inserting columns for novel classes, as shown in Figure 1 .

Regarding the simple classifier, the initial weight for novel classes can be obtained by applying the multi-class linear SVM to feature vectors of novel classes.

We evaluated our method using the low-resolution mini-ImageNet dataset as a common evaluation dataset.

In addition, we used the high-resolution mini-ImageNet and cross-domain datasets as more practical datasets.

Details of these datasets are provided in Section 2.3.

In this study, we identify tasks that use these datasets as low-resolution single-domain task, the high-resolution single-domain task, and cross-domain tasks.

Qi et al. (2018) .

Each column w i ∈ R d of the classifier weight is normalized so that w i has a norm of 1 (i.e., ∥w i ∥ = 1), and classification is performed by taking the inner products between w i and normalized feature vectorẑ ∈ R d .

Note that variable d is the dimension of the feature space.

Before the network is fine-tuned, the initial weight for a novel class can be obtained by including feature vectorẑ of the novel class in the classifier weight.

When multiple novel examples per class are available (i.e., K-shot learning with K > 1), the initial weight for the novel class can be obtained by normalizing class mean 1/K ∑ K j=1ẑ j again.

Because the output range of Wẑ is [−1, 1], ensuring that the probability of the correct label approximates 1 using softmax activation is difficult.

This problem can be avoided by applying scale factor s ∈ R to the output, as discussed by Qi et al. (2018) .

The networks were pretrained by using the base classes of the datasets for 600 epochs.

We used the Adam optimizer (Kingma & Ba, 2014) with a learning rate of 0.001 in the same manner as Chen et al. (2019) .

These parameters for pretraining are normally optimized using validation classes, but we fixed these parameters to reduce computational cost.

Input images were preprocessed by random-resized cropping with a size of 224 × 224.

We also performed color jittering and random-horizontal flipping, and we subtracted channel-wise means of the ImageNet dataset (0.485, 0.456, 0.406).

In addition, division by channel-wise standard deviations of the ImageNet dataset (0.229, 0.224, 0.225) was performed in the same manner as Chen et al. (2019) .

Note that in the low-resolution single-domain task, the size of the random-resized cropping was set to 84 × 84.

In this study, we compared three fine-tuning methods in which: 1) the entire network is updated, 2) the classifier weight and batch-normalization (BN) statistics are updated, and 3) only the classifier weight is updated.

The third method is a common fine-tuning method to prevent overfitting.

The second method is based on a previous study (Noguchi & Harada, 2019 ) that successfully fine-tuned an image generator to a novel class without overfitting by updating only the BN statistics (i.e., γ and β of BN layers).

A similar approach is known as meta-transfer learning (MTL) (Sun et al., 2019) .

The authors who proposed MTL showed that updating only scales and biases of network parameters prevents them from overfitting while achieving efficient adaptation to unseen tasks.

Although the methods proposed by Noguchi & Harada (2019) and Sun et al. (2019) presented similar ideas, we chose the former because of its simplicity in implementation.

Initial classifier weights for novel classes were obtained before we fine-tuned the networks, as discussed in Section 3.1.

The networks were retrained with mini-batch-based learning with a batch size of N K in the N -way K-shot learning scenario.

The learning rate and number of epochs for finetuning were determined by using validation classes.

We evaluated few-shot classification accuracy by calculating the mean accuracy of 600 trials using randomly sampled classes and examples in the novel classes.

We also calculated the 95% confidence interval of the mean accuracy.

In the validation, test, and network initialization phases for the novel classes, input images were preprocessed by resizing to 256 × 256, center-cropping to a size of 224 × 224, subtracting channel-wise means of the ImageNet dataset (0.485, 0.456, 0.406), and dividing by channel-wise standard-deviations of the ImageNet dataset (0.229, 0.224, 0.225) in the same manner as Chen et al. (2019) .

The input preprocessing for fine-tuning phase was the same as discussed in Section 3.3.

Table 1 : Performance of our method in the 5-way low-resolution single-domain task.

"Normalized" and "Simple" mean that the normalized and simple classifiers are used, respectively.

"All", "BN & FC", and "FC" mean the following: the entire network was updated; the BN and FC layer were updated; only the FC layers were updated.

"w/o FT" refers to performance without fine-tuning using novel classes.

Values with the † mark refer to classification accuracy without fine-tuning, as validation accuracy was not increased by fine-tuning the network.

The * mark means that the classification accuracy for novel classes was not available because the loss value did not decrease in the pretraining phase.

The -mark means that we did not conduct an experiment because the network did not have BN layers.

Table 2 : Performance of our method in the 5-way high-resolution single-domain task.

"Normalized" and "Simple" mean that the normalized and simple classifiers were used, respectively.

"All", "BN & FC", and "FC" mean the following: the entire network was updated; the BN and FC layer were updated; only the FC layers were updated.

"w/o FT" refers to performance without fine-tuning using novel classes.

Values with the † mark refer to classification accuracy without fine-tuning, as validation accuracy was not increased by fine-tuning the network.

The * mark means that classification accuracy for novel classes was not available because the loss value did not decrease in the pretraining phase.

The -mark means that we did not conduct an experiment because the network did not have BN layers.

Few-shot classification accuracies for the low-resolution single-domain task, high-resolution singledomain task, and cross-domain task are listed in Tables 1, 2 , and 3, respectively.

Table 1 shows that the classification accuracy could be increased by approximately 6% when the VGG-16 and normalized classifier were used in the 1-shot learning task.

However, the accuracy could not be further improved by updating the entire network in the 1-shot learning task.

However, classification accuracy could be further improved by updating the entire network in the 5-shot task.

We assume that this was because the within-class difference could be reduced by fine-tuning the feature extractor when multiple novel examples were available.

By comparing the results for the high-resolution single-domain (Table 2 ) and low-resolution (Table  1) tasks, it could be argued that the robustness against low-resolution inputs differs depending on the feature extractor.

For example, by comparing the results for 5-shot "Normalized all" in Table 1 and 2, we can see that the classification accuracy of the ResNet-152 decreased by 11.0% whereas that of the VGG-16 decreased by only 4.3%.

This implies that the robustness against low-resolution inputs should also be considered and that evaluating only few-shot learning performance is difficult.

Although the low-resolution mini-ImageNet dataset is extremely useful for doing fast experiments, we must reconsider the validity of the dataset for evaluation of few-shot learning performance.

Table 3 : Performance of our method in the 5-way cross-domain task.

"Normalized" and "Simple" mean that the normalized and simple classifiers were used, respectively.

"All", "BN & FC", and "FC" mean the following: the entire network was updated; the BN and FC layer were updated; only the FC layers were updated.

"w/o FT" refers to performance without fine-tuning using novel classes.

Values with the † mark refer to classification accuracy without fine-tuning, as validation accuracy was not increased by fine-tuning the network.

The * mark means that classification accuracy for novel classes was not available because the loss value did not decrease in the pretraining phase.

The -mark means that we did not conduct an experiment because the network did not have BN layers.

Table 4 : Comparison between our method and conventional methods.

We show several results from different networks in Section 3.5, and therefore in this table show the highest accuracy for each task.

More specifically, we use the results from "VGG-16 Normalized FC", "VGG-16 Normalized All", "VGG-16 Normalized FC", "ResNet-50 Normalized All", and "ResNet-50 Normalized All" from left to right in this table.

Values with the ‡ marks were reported by Chen et al. (2019) , and other values were reported in the original studies.

The -mark means that the classification accuracy for the task was not reported. (Schwartz et al., 2018)

59.9 69.7 ---

The comparison of the results from the cross-domain task (Table 3) and high-resolution singledomain task shows that the performance decreased in the cross-domain task.

This can be explained by the larger domain shift that occurs between base and novel classes, as indicated by Chen et al. (2019) .

In addition, the difference in classification accuracy between the cross-domain task and high-resolution single-domain task was decreased by fine-tuning the entire network.

For example, in the 5-shot learning task using the VGG-16, the difference in classification accuracy between the single-domain and cross-domain tasks was 15.9% without fine-tuning, but it could be reduced to 6.9% by fine-tuning the entire network.

This means that the network could adapt to a large domain shift by having the entire network updated.

We discuss this further in greater detail in Section 3.7.

3.6 COMPARATIVE EVALUATION Table 4 shows comparative results of ours and previous methods.

Note that we use the best result for each task as given in Section 3.5 because we obtained several results from different networks.

In the 1-shot low-resolution single-domain task, the classification accuracy was lower than that of the state-of-the-art algorithm, but it was still higher than those of other common few-shot learning methods such as MatchingNet and ProtoNet.

It is interesting to note that our method achieved nearly the same classification accuracy as that of the state-of-the-art method in the 5-shot task.

The reason for the higher performance in the 5-shot task may be that the within-class variance could be reduced by fine-tuning the entire network using several examples per class.

In addition, we achieved higher classification accuracy than the reported values of conventional methods both in the high-resolution single-domain and cross-domain tasks.

The difference in classification accuracy between the 5-shot high-resolution single-domain and cross-domain tasks was only approximately 5% with our method, whereas the performance of the conventional methods decreased by more than 10% in the cross-domain task.

This shows that our method successfully We used the ResNet-18, normalized classifier, and Adam optimizer.

We chose the 5-shot cross-domain task for visualization because the transition of validation accuracy is clearer than in other tasks.

We set the learning rates as 0.01, 0.001, and 0.0001, and conducted four trials with randomly selected validation classes and support sets.

Note that classification accuracy can be significantly changed by the randomly selected classes and samples.

Therefore, we focused on the transition of the validation accuracy rather than the validation accuracy itself.

reduced the effect of a large domain shift in the cross-domain task by updating the entire network for novel classes.

We revealed in Section 3.5 and 3.6 that the fine-tuning method achieved high few-shot classification accuracy in many cases.

In this section, we discuss the means of improving the performance of the fine-tuning method.

We experimentally show that:

• Using a low learning rate for fine-tuning stabilizes the retraining process.

• Using adaptive gradient optimizers such as Adam increases the classification accuracy.

• Higher performance can be obtained by updating the entire network when a large domain shift occurs.

A learning rate is a critical parameter in training a network; this is also true for fine-tuning for fewshot learning.

Figure 2 shows that the retraining process can be stabilized by using a lower learning rate.

For example, the transition of the validation accuracy was unstable when the learning rate was set as 0.01 and 0.001.

However, the validation accuracy increased in a stable manner when the learning rate was set as 0.0001, which is lower than that used in the pretraining phase (lr = 0.001).

This means that the learning rate for few-shot fine-tuning should be set low.

Here, we show that optimizers affect few-shot classification performance when the network is (Duchi et al., 2011) , RMSprop (Graves, 2013) , Momentum-SGD, and ASGD (Polyak & Juditsky, 1992) .

Of these, Adam, Adamax, Adadelta, Adagrad, and RMSprop are known as adaptive gradient methods.

Figure 3 shows the classification accuracies for a 5-shot high-resolution single-domain task using the ResNet-18 with different optimizers.

The results show that higher classification accuracies could be obtained by using adaptive gradient optimizers, particularly when the normalized classifier was used.

Although Wilson et al. (2017) revealed that local minima obtained by the Adam optimizer lack a generalization ability, our experimental results show that this was not necessarily true for few-shot learning.

Why this occurs in few-shot learning is interesting, but this is beyond the scope of this study.

Therefore, we leave this interesting direction for future works.

Updating the Entire Network for Adaptation to a Large Domain Shift Figure 4 shows the relationship between test accuracy and the updated parts of the network.

This shows that updating the entire network achieves higher accuracy, particularly when the normalized classifier is used.

Considering the results for the high-resolution single-domain task, when test accuracy was not further increased by updating the entire network, it could be argued that updating the entire network when a large domain shift occurs between base and novel classes is preferable.

In this study, we showed that in the low-resolution single-domain task, our fine-tuning method achieved higher accuracy than common few-shot learning methods in the 1-shot task and nearly the same accuracy as the state-of-the-art method in the 5-shot task.

We also evaluated our method with more practical tasks, such as the high-resolution single-domain and cross-domain tasks.

In both tasks, our method achieved higher accuracy than common few-shot learning methods.

We then experimentally showed that: 1) a low learning rate stabilizes the retraining process, 2) adaptive gradient optimizers such as Adam improve test accuracy, and 3) updating the entire network results in higher accuracy when a large domain shift occurs.

We believe that these insights into fine-tuning for few-shot learning tasks will help our community tackle this challenging task.

<|TLDR|>

@highlight

An empirical study that provides a novel perspective on few-shot learning, in which a fine-tuning method shows comparable accuracy to more complex state-of-the-art methods in several classification tasks.
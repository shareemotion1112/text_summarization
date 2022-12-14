In this paper, we empirically investigate the training journey of deep neural networks relative to fully trained shallow machine learning models.

We observe that the deep neural networks (DNNs) train by learning to correctly classify shallow-learnable examples in the early epochs before learning the harder examples.

We build on this observation this to suggest a way for partitioning the dataset into hard and easy subsets that can be used for improving the overall training process.

Incidentally, we also found evidence of a subset of intriguing examples across all the datasets we considered, that were shallow learnable but not deep-learnable.

In order to aid reproducibility, we also duly release our code for this work at https://github.com/karttikeya/Shallow_to_Deep/

Analyzing the temporal journey taken by deep neural networks (DNNs) during training has elicited a lot of attention recently.

The authors in BID0 suggested that DNNs learn simple patterns first, before memorizing.

More specifically, they posit that real world datasets are littered with easy examples characterized by simple patterns that are learned in the initial epoch(s) before the conquest of hard examples in the training dataset.

Tishby et al BID25 conjectured that DNN training was characterized by two distinct phases consisting of an initial fitting phase (memorization) and a subsequent compression phase.

While this claim was questioned in BID24 , the authors do remark that when an input domain consists of a subset of task-relevant and task-irrelevant information, hidden representations do compress the task-irrelevant information.

These works do suggest that the easy-vs-hard dichotomy in real-world datasets does influence the learn- ing in DNNs and goad a data-dependent approach towards understanding the capacity of DNNs.

Taking cue from this, we strive to contribute to this growing body of literature by bringing in another viewpoint: The dichotomy between shallow learnable examples and deep learnable examples in the dataset.

More specifically we try to address the questions:1.

Is the notion of easiness same for models with as different parameterizations and architectures as shallow machine learning models and deep networks the same?

And hence is attached to the example independently of a model?2.

If we are to investigate the examples that a DNN learns to correctly classify over the training batches, do we observe a shallow learnable to deep learnable regime change?3.

Are there examples that are shallow learnable but somehow a DNN with a far better overall accuracy fails to classify?

At the heart of this quest is to understand if shallow learnability is a good proxy for the easiness of an example.

We'd like to reiterate that the motivation behind this work is to obtain insights into the changing scenery of the conquest of the training dataset experienced by deep neural networks and not to delineate the nature of compositional functions that DNNs can learn and shallow algorithms cannot or comment on the amount of training data required to do so.

In BID19 , the authors have already shown how DNNs can approximate the class of compositional functions as well as shallow networks but with exponentially lower number of training parameters and sample complexity.

The rest of the paper is organized as follows.

In section 2, we present the quantitative methodology we used to answer these questions raised above.

In section 3, we showcase our empirical experiments with the results covered in section 4.

We conclude the paper in section 5.

In this section we delineate our proposed method to study the learning process of a deep learning model D relative to a shallow machine learning model M.

We propose to measure the generalization capability of a deep learning model D on unknown data using a custom DISPLAYFORM0

Several useful metrics can be derived from T that analyze different aspects of the learning process.

Naively, accuracies of models D and M can be recovered simply by DISPLAYFORM0 Note that since we're tracking the learning dynamics for D ( i) and model M is kept the same, Accuracy(M) actually remains constant throughout.

More interesting metrics can be derived such as the accuracy of D (i) on subsets of the training data that M classifies correctly (R i + ) and those that M classifies wrongly (R i ??? ).

DISPLAYFORM1 Also, to measure how many times more accurate D (i) is on one subset versus the other, we study the ratio of the above two accuracies DISPLAYFORM2

We perform our experiments under three different regimes divided on the basis of relative performance of shallow Machine Learning models to Deep Networks to check the robustness of our observations.

We experiment with (a) The MNIST dataset, where the ML models tend to perform competitively with the ConvNets (b) The CIFAR10 dataset where ML models perform worse than neural networks but not are still quite good and lastly (c) the CIFAR100 dataset, where the deep networks far outperform the shallow Machine Learning models.

We preprocess all the datasets to center and normalize the images before training models but do not augment data to avoid other potential confounding factors in our observations.

MNIST: MNIST (Mixed National Institute of Standards and Technology) (LeCun, 1998) is a very popular toy dataset consisting of images of handwritten digits and their numeric value as their labels.

It consists of a total of 70, 000 grayscale 28 ?? 28 labelled images divided in training set of 60, 000 and another test set of 10, 000 images.

It is widely used to examine image recognition techniques and considered a relatively simple starter dataset.

The CIFAR database BID15 ) refers to two related but different datasets namely, the CIFAR10 and CIFAR100 datasets.

CIFAR10 consists of 60, 000 color 32 ?? 32 images divided into 10 object categories like airplane, bird, cat etc.

It consists of a training set of 50, 000 and a test set of 10, 000 images.

CIFAR100 also has the same characterstics and size except it consists of 100 image classes rather than just 10.

Thus, it consists of 600 examples of each class which are grouped proportionally into 500 in the training set and 100 in the test set.

The dataset also contains the 100 classes grouped into 20 super classes but in this work we use fine grained classes for comparing models on CIFAR100.

We train two types of machine learning models to study the learning process -Support Vector Machines BID4 and Random Forests BID2 ).Random Forest: Random Forest are another family of successful Machine Learning models which have been applied to a large number of classification and regression problems BID18 like real time face detection BID3 , Gene Selection (D??az-Uriarte & De Andres, 2006), Remote sensing BID20 and several other applications.

In this paper, we train the Random Forests using 20 estimators and the gini index criterion BID21 .

We flatten the preprocessed images into a 784 dimensional vector in case of MNIST and into a 1024 dimensional vector for CIFAR to facilitate training the random forest.

Ratio of Accuracies R i plotted against i with M being a Support Vector Machine.

Values of the contingency matrix T i plotted against i (smoothed and scaled for plotting).

Referring to TAB0 , the entries T00, T01, T10 and T11 are denoted by red, orange, green and blue respectively.

Figure 1 .

Graphs showing evolution of various metrics on the test set (defined in Section 3) as training progresses for MNIST (left columns), CIFAR10 (middle column) and CIFAR100 (right column) datasets.

Each X-tick represents: 640 images for MNIST, 128 images for CIFAR10 and 25, 600 images for CIFAR100 dataset.(SVM) are extremely popular ML models shown to achieve good results in several limited data domains like hand written character recognition BID6 , face recognition BID11 ), hypertext classification BID23 and several biological applications BID5 BID26 .

We train a SVM with slacks under the Radial Basis Kernel BID21 with hyperparameters C = 1.0 and ?? = 0.1.

Similar to Random Forests, we flatten out the images for training an SVM.

We train a different deep network on each dataset because of two reasons.

First, some networks we train have a much larger capacity than others and hence are not suited for a small dataset like MNIST.

Likewise, the architecture suited for MNIST is too small for a more complicated dataset like CIFAR and would not be a optimal choice for good performance.

Second, we want to check robustness of our observations across different model sizes and architectures and hence experiment with very popular but very different ontologies like DenseNet and ResNet.

We train a small Convolution Network with the architecture : DISPLAYFORM0 The network is trained on the MNIST dataset with SGD with a learning rate 0.01, momentum set to 0.5 and a batch size of 64.DenseNet121: DenseNet 121 BID12 ) is a deep network that has been demonstrated to achieve a very good performance on that task of Image recognition on a variety of benchmarks.

In a nutshell, DensetNet contains several DenseBlocks which inside themselves are a constant feature depth stack of ConvNets; connected in a fully connected fashion.

We train DenseNet 121 on the CIFAR10 dataset with SGD with learning rate 0.1, momentum 0.5 and a batch size of 128.

Further training details are in the appendix.

Resnet 101: ResNet 101 belongs to the very popular family of Residual Networks BID10 and is widely used as a backbone in a variety of computer vision tasks like Image Recognition BID10 , Action Classification BID8 , Image to Image Translation BID27 etc.

We use ResNet 101 for image recognition on the CIFAR100 dataset where it is trained with SGD under same parameters as mentioned for DenseNet121 but with an additional weight decay of 5 ?? 10 ???4 .

Here we discuss our observations while studying the training process under the lens of machine learning models.

TAB2 reports the maximum accuracy achieved with the models discussed in Section 3.2 and 3.3.

While the accuracies are reported for completeness, this work studies the learning process in the early stages and hence is not concerned with the maximum accuracy.

Moreover, in cases of CIFAR10 and CIFAR100 datasets the operating regime for DenseNet121 and ResNet101 are far from their optimal performance.

Referring to the top row in FIG1 , we observe how the ratio of accuracies (R + ) changes during training.

Note that if the two subsets, M-correct and M-incorrect were completely irrelevant and similar for training process of D, R + would remain identically 1.

However, we observe that across datasets and (M, D) pairs, the curve has a right skewed unimodal shape with a sharp hump.

We also note that this happens in very early stages of training, sometimes as early as just after 1/20 th epoch over the training set.

Also, the accuracies can be very different on the two subsets with D being upto 8 times more accurate on M-correct subsets than on M-correct subsets in some cases.

Furthermore, we observe that the curve shows a long tail as the ratio R i returns back to 1.

This observation is the cornerstone in confirming out hypothesis that deep networks training starts from quickly learning shallow classifiable easy examples and then slowly extends to the hard ones.

The second row of Fig. 1 and FIG1 depicts test accuracies as training progresses.

Note that the overall trend is increasing as expected from a network in early stages of training, however there are huge gaps in accuracy on the two different subsets.

For example, in the case of CIFAR10 when the M -incorrect subset is ??? 60% of the total set, it weighs down the overall accuracy by over as much as 20% at times during training.

Thus, identifying the hard examples with M incorrectness can help in training procedures like curriculum learning BID1 , teacher forcing BID13 BID22 and professor forcing BID16 .

Furthermore, weighing the training set examples on basis of whether they are correctly classified by M can provide a more balanced dataset for training.

Observe the long tail decay of T 00 and T 11 and the very fast rise of T 11 and T 10 (bottom row Figure 1 ).

This shows that the the slow learning of M-incorrect examples is the major factor of slump in accuracy growth for deep networks after the often observed initial fast ascent.

This observation can be used for iterating more on M-incorrect harder data points after the initial phase and achieve faster convergence.

In this work, we track the training of DNNs relative to shallow machine learning models.

We showcase some results on analyzing the training trajectory of the DNNs relative to SVM and RF on three different datasets.

Empirically, we observe that the during training the Deep Network quickly learns shallow classifiable easy examples first and then learns the hard examples in the later epochs.

Furthermore, we find that the notion of hardness of an example is largely independent of the model being used and can be evaluated reliably using a shallow learning model.

This observation allows for a procedural slicing of the training set into easy and hard categories that can improve network training.

We also report a slightly surprising finding pertaining to the existence of a subset of examples in all the datasets considered that were shallow-classifiable but not deep-classifiable.

We are currently extending this work along the following two paths.

The first entails using the influence functions framework BID14 to understand the distribution of the influence of the training examples and juxtaposing this with respect to their shallow/deep learnability.

The second path entails understanding the nature of this conquest of the training space of the deep and shallow classifiers from the viewpoint of complexity and interestingness of images BID9 .

We conclude with a conjecture that complexity of images as measured by, say, it's JPEG compressibility will have strong correlations with it's shallow learnability.

<|TLDR|>

@highlight

We analyze the training process for Deep Networks and show that they start from rapidly learning shallow classifiable examples and slowly generalize to harder data points.
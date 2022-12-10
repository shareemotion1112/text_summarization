The capability of reliably detecting out-of-distribution samples is one of the key factors in deploying a good classifier, as the test distribution always does not match with the training distribution in most real-world applications.

In this work, we propose a deep generative classifier which is effective to detect out-of-distribution samples as well as classify in-distribution samples, by integrating the concept of Gaussian discriminant analysis into deep neural networks.

Unlike the discriminative (or softmax) classifier that only focuses on the decision boundary partitioning its latent space into multiple regions, our generative classifier aims to explicitly model class-conditional distributions as separable Gaussian distributions.

Thereby, we can define the confidence score by the distance between a test sample and the center of each distribution.

Our empirical evaluation on multi-class images and tabular data demonstrate that the generative classifier achieves the best performances in distinguishing out-of-distribution samples, and also it can be generalized well for various types of deep neural networks.

Out-of-distribution (OOD) detection, also known as novelty detection, refers to the task of identifying the samples that differ in some respect from the training samples.

Recently, deep neural networks (DNNs) turned out to show unpredictable behaviors in case of mismatch between the training and testing data distributions; for example, they tend to make high confidence prediction for the samples that are drawn from OOD or belong to unseen classes (Szegedy et al., 2014; Moosavi-Dezfooli et al., 2017) .

For this reason, accurately measuring the distributional uncertainty (Malinin & Gales, 2018) of DNNs becomes one of the important challenges in many real-world applications where we can hardly control the testing data distribution.

Several recent studies have tried to simply detect OOD samples using the confidence score defined by softmax probability (Hendrycks & Gimpel, 2017; Liang et al., 2018) or Mahalanobis distance from class means (Lee et al., 2018) , and they showed promising results even without re-training the model.

However, all of them employ the DNNs designed for a discriminative (or softmax) classifier, which has limited power to locate OOD samples distinguishable with in-distribution (ID) samples in their latent space.

To be specific, the softmax classifier is optimized to learn the discriminative latent space where the training samples are aligned along their corresponding class weight vectors, maximizing the softmax probability for the target classes.

As pointed out in (Hendrycks & Gimpel, 2017) , OOD samples are more likely to have small values of the softmax probability for all known classes, which means that their latent vectors get closer to the origin.

As a result, there could be a large overlap between two sets of ID and OOD samples in the latent space (Figure 1 ), which eventually reduces the gap between their confidence scores and degrades the performance as well.

In addition, most of existing confidence scores adopt additional calibration techniques Hinton et al., 2015) to enhance the reliability of the detection, but they include several hyperparameters whose optimal values vary depending on the testing data distribution.

In this situation, they utilized a small portion of each test set (containing both ID and OOD samples) for validation, and reported the results evaluated on the rest by using the optimal hyperparameter values for each test case.

Considering the motivation of OOD detection that prior knowledge of test distributions is not available before we encounter them, such process of tuning the hyperparameters for each test case is not practical when deploying the DNNs in practice.

In this paper, we propose a novel objective to train DNNs with a generative (or distance) classifier which is capable of effectively identifying OOD test samples.

The main difference of our deep generative classifier is to learn separable class-conditional distributions in the latent space, by explicitly modeling them as a DNN layer.

The generative classifier places OOD samples further apart from the distributions of all given classes, without utilizing OOD samples for its validation.

Thus, based on the Euclidean distance between a test sample and the centers of the obtained class-conditional distributions, we can calculate how likely and how confidently the sample belongs to each class.

This can be interpreted as a multi-class extension of unsupervised anomaly detection (Ruff et al., 2018) , and Gaussian discriminant analysis provides the theoretical background for incorporating the generative classifier into the DNNs.

Our extensive experiments on images and tabular data demonstrate that the proposed classifier distinguishes OOD samples more accurately than the state-of-the-art method, while maintaining the classification accuracy for ID samples.

We introduce a novel objective for training deep neural networks (DNNs) with a generative classifier, which is able to effectively detect out-of-distribution samples as well as classify in-distribution samples into known classes.

We first derive the learning objective from the Gaussian discriminant analysis, and propose the distance-based confidence score for out-of-distribution sample detection.

Metric learning objective for classification.

The key idea of our objective is to optimize the deep learning model so that the latent representations (i.e., the outputs of the last layer) of data samples in the same class gather together thereby form an independent sphere.

In other words, it aims to learn each class-conditional distribution in the latent space to follow a normal distribution that is entirely separable from other class-conditional distributions.

Using the obtained distributions, we can calculate the class-conditional probabilities that indicate how likely an input sample is generated from each distribution, and this probability can serve as a good measure of the confidence.

We define the two terms based on the Euclidean distance between the data representations obtained by the DNNs, denoted by f (x), and the center of each class-conditional distribution, denoted by c k .

Given N training samples {(x 1 , y 1 ), . . .

, (x N , y N )} from K different classes, the objective is described as follows.

The objective includes three types of trainable parameters: the weights of DNNs W, the class centers c 1 , . . .

, c K and biases b 1 , . . .

, b K .

All of them can be effectively optimized by stochastic gradient descent (SGD) and back-propagation, which are widely used in deep learning.

Note that we directly optimize the latent space induced by the DNNs using Euclidean distance, similarly to other metric learning objectives.

Existing deep metric learning based on the triplet loss (Hoffer & Ailon, 2015; Schroff et al., 2015) learns the distance among training samples utilizing their label information to capture their similarities into the metric space for a variety of retrieval tasks.

On the other hand, our objective focuses on the distance between the samples and their target class centers for the accurate modeling of class-conditional distributions.

Derivation from Gaussian discriminant analysis.

Our objective for the generative classifier can be understood from the perspective of Gaussian discriminant analysis (GDA) (Murphy, 2012) .

The generative classifier defines the posterior distribution P (y|x) by using the class-conditional distribution P (x|y) and class prior P (y).

In case of GDA, each class-conditional distribution is assumed to follow the multivariate Gaussian distribution (i.e., P (x|y = k) = N (x|µ k , Σ k )) and the class prior is assumed to follow the Bernoulli distribution (i.e., P (y = k) =

To simply fuse GDA with DNNs, we further fix all the class covariance matrices to the identity matrix (i.e., Σ k = I).

Then, the posterior probability that a sample f (x) belongs to the class k is described as

Considering µ k and log β k as the class center c k and bias b k respectively, the first term of our objective (1) is equivalent to the negative log posterior probability.

That is, the objective eventually trains the classifier by maximizing the posterior probability for training samples.

However, the direct optimization of the DNNs and other parameters by its gradient does not guarantee that the class-conditional distributions become the Gaussian distributions and the class centers are the actual class means of training samples.

Thus, to enforce our GDA assumption, we minimize the Kullback-Leibler (KL) divergence between the k-th empirical class-conditional distribution and the Gaussian distribution whose mean and covariance are c k and I, respectively.

The empirical class-conditional distribution is represented by the average of the dirac delta functions for all training samples of a target class, i.e.,

where N k is the number of the training samples of the class k. Then, the KL divergence is formulated as

The entropy term of the empirical class-conditional distribution can be calculated by using the definition of the dirac measure (Murphy, 2012) .

By minimizing this KL divergence for all the classes, we can approximate the K class-conditional Gaussian distributions.

Finally, we complete our objective by combining this KL term with the posterior term using the λ-weighted sum in order to control the effect of the regularization.

We remark that λ is the hyperparameter used for training the model, which depends on only ID, not OOD; thus it does not need to be tuned for different test distributions.

In-distribution classification.

Since our objective maximizes the posterior probability for the target class of each sample P (y = y i |x), we can predict the class label of an input sample to the class that has the highest posterior probability as follows.

In terms of DNNs, our proposed classifier replaces the fully-connected layer (fc-layer) computing the final classification score by w k · f (x) + b k with the distance metric layer (dm-layer) computing the distance from each center by − f (x) − c k 2 + b k .

In other words, the class label is mainly predicted by the distance from each class center, so we use the terms "distance classifier" and "generative classifier" interchangeably in the rest of this paper.

The dm-layer contains the exactly same number of model parameters with the fully-connected layer, because only the weight matrix

K×d is replaced with the class center matrix Out-of-distribution detection.

Using the trained generative classifier (i.e., class-conditional distributions obtained from the classifier), the confidence score of each sample can be computed based on the class-conditional probability P (x|y = k).

Taking the log of the probability, we simply define the confidence score D(x) using the Euclidean distance between a test sample and the center of the closest class-conditional distribution in the latent space,

This distance-based confidence score yields discriminative values between ID and OOD samples.

In the experiment section, we show that the Euclidean distance in the latent space of our distance classifier is more effective to detect the samples not belonging to the K classes, compared to the Mahalanobis distance in the latent space of the softmax classifier.

Moreover, it does not require further computation to obtain the class means and covariance matrix, and the predictive uncertainty can be measured by a single DNN inference.

Relationship to deep one-class classifier.

Recent studies on one-class classification, which have been mainly applied to anomaly detection, try to employ DNNs in order to effectively model the normality of a single class.

Inspired by early work on one-class classification including one-class support vector machine (OC-SVM) (Schölkopf et al., 2001 ) and support vector data description (SVDD) (Tax & Duin, 2004) , Ruff et al. (2018; 2019) proposed a simple yet powerful deep learning objective, DeepSVDD.

It trains the DNNs to map samples of the single known class close to its class center in the latent space, showing that it finds a hypersphere of minimum volume with the center c:

Our DNNs with the distance classifier can be interpreted as an extension of DeepSVDD for multiclass classification, which incorporates K one-class classifiers into a single network.

In the proposed objective (1), the first term makes the K classifiers distinguishable for the multi-class setting, and the second term learns each classifier by gathering the training samples into their corresponding center, as done in DeepSVDD.

The purpose of the one-class classifier is to determine whether a test sample belong to the target class or not, thus training it for each class is useful for detecting out-of-distribution samples in our task as well.

In this section, we present experimental results that support the superiority of the proposed model.

Using tabular and image datasets, we compare the performance of our distance classifier (i.e., DNNs with dm-layer) with that of the softmax classifier (i.e., DNNs with fc-layer) in terms of both ID classification and OOD detection.

We also provide empirical analysis on the effect of our regularization term.

Our code and preprocessed datasets will be publicly available for reproducibility.

Experimental settings.

We first evaluate our distance classifier using four multi-class tabular datasets with real-valued attributes: GasSensor, Shuttle, DriveDiagnosis, and MNIST.

They are downloaded from UCI Machine Learning repository 1 , and we use them after preprocessing all the attributes using z-score normalization.

Table 1 summarizes the details of the datasets.

To simulate the scenario that the test distribution includes both ID and OOD samples, we build the training and test set by regarding one of classes as the OOD class and the rest of them as the ID classes.

We exclude the samples of the OOD class from the training set, then train the DNNs using only the ID samples for classifying inputs into the K-1 classes.

The test set contains all samples of the OOD class as well as the ID samples that are left out for testing.

The evaluations are repeated while alternately changing the OOD class, thus we consider K scenarios for each dataset.

For all the scenarios, we perform 5-fold cross validation and report the average results.

The multi-layer perceptron (MLP) with three hidden layers is chosen as the DNNs for training the tabular data.

For fair comparisons, we employ the same architecture of MLP (# Input attributes ×128 × 128 × 128× # Classes) for both the softmax classifier and the distance classifier.

We use the Adam optimizer (Kingma & Ba, 2014) with the initial learning rate η = 0.01, and set the maximum number of epochs to 100.

In case of tabular data, we empirically found that the regularization coefficient λ hardly affects the performance of our model, so fix it to 1.0 without further hyperparameter tuning.

We consider two competing methods using the DNNs optimized for the softmax classifier: 1) the baseline method (Hendrycks & Gimpel, 2017 ) uses a maximum value of softmax posterior probability as the confidence score, max k

, and 2) the state-of-the-art method (Lee et al., 2018) defines the score based on the Mahalanobis distance using empirical class meansμ k and covariance matrixΣ, which is max

Note that any OOD samples are not available at training time, so we do not consider advanced calibration techniques for all the methods; for example, temperature scaling, input perturbation (Liang et al., 2018) , and regression-based feature ensemble (Lee et al., 2018) .

We measure the classification accuracy for ID test samples 2 , as well as three performance metrics for OOD detection: the true negative rate (TNR) at 85% true positive rate (TPR), the area under the receiver operating characteristic curve (AUROC), and the detection accuracy.

Experimental results.

In Table 2 , our proposed method (i.e., distance-based confidence score) using the distance classifier considerably outperforms the other competing methods using the softmax classifier in most scenarios.

Compared to the baseline method, the Mahalanobis distance-based confidence score sometimes performs better, and sometimes worse.

This strongly indicates that the empirical data distribution in the latent space does not always take the form of Gaussian distribution for each class, in case of the softmax classifier.

For this reason, our explicit modeling of class-conditional Gaussian distributions using the dm-layer guarantees the GDA assumption, and it eventually helps to distinguish OOD samples from ID samples.

Moreover, the distance classifier shows almost the same classification accuracy with the softmax classifier; that is, it improves the performance of OOD detection without compromising the performance of ID classification.

For qualitative comparison on the latent spaces of the softmax classifier and distance classifier, we plot the 2D latent space after training the DNNs whose size of latent dimension is set to 2.

Figure 1 illustrates the training and test distributions of the GasSensor dataset, where the class 3 (i.e., Ammonia) is considered as the OOD class.

Our DNNs successfully learn the latent space so that ID and OOD samples are separated more clearly than the DNNs of the softmax classifier.

Notably, in case of the softmax classifier, the covariance matrices of all the classes are not identical, which violates the necessary condition for the Mahalanobis distance-based confidence score to be effective in detecting OOD samples.

4 In this sense, the proposed score does not require such assumption any longer, because our objective makes the latent space satisfy the GDA assumption.

Experimental settings.

We validate the effectiveness of the distance classifier on OOD image detection as well.

Two types of deep convolutional neural networks (CNNs) are utilized: ResNet (He et al., 2016) with 100 layers and DenseNet (Huang et al., 2017) with 34 layers.

Specifically, we train ResNet and DenseNet for classifying three image datasets: CIFAR-10, CIFAR-100 (Krizhevsky et al., 2009) , and SVHN (Netzer et al., 2011) .

Each dataset used for training the models is considered as ID samples, and the others are considered as OOD samples.

To consider a variety of OOD samples at test time, we measure the performance by additionally using TinyImageNet (randomly cropped image patches of size 32 × 32 from ImageNet dataset) (Deng et al., 2009 ) and LSUN (Yu et al., 2015) as test OOD samples.

All CNNs are trained with stochastic gradient descent with Nesterov momentum (Duchi et al., 2011) , and we follow the training configuration (e.g., the number of epochs, batch size, learning rate and its scheduling, and momentum) suggested by (Lee et al., 2018; Liang et al., 2018) .

The regularization coefficient λ of the distance classifier is set to 0.1.

Experimental results.

Table 3 shows that our distance classifier also can be generalized well for deeper and more complicated models such as ResNet and DenseNet.

Similarly to tabular data, our confidence score achieves the best performance for most test cases, and significantly improves the detection performance over the state-of-the-art method.

Interestingly, the distance classifier achieves better ID classification accuracy than the softmax classifier in Table 4 .

These results show the possibility that any existing DNNs can improve their classification power by adopting the dmlayer, which learns the class centers instead of the class weights.

From the experiments, we can conclude that our proposed objective is helpful to accurately classify ID samples as well as identify OOD samples from unknown test distributions.

We further investigate the effects of our regularization term on the performance and the data distributions in the latent space.

We first evaluate the distance classifier, using the DNNs trained with different λ values from 10 −3 to 10 3 .

Figure 2 presents the performance changes with respect to the λ value.

In terms of ID classification, the classifier cannot be trained properly when λ grows beyond 10 2 , because the regularization term is weighted too much compared to the log posterior term in our objective which learns the decision boundary.

On the other hand, we observe that the OOD detection performances are not much affected by the regularization coefficient, unless we set λ too small or too large; any values in the range (0.1, 10) are fine enough to obtain the model working well.

We also visualize the 2D latent space where the training distribution of MNIST are represented, varying the value of λ ∈ {0.01, 0.1, 1, 10}. In Figure 3 , even with a small value of λ, we can find the decision boundary that partitions the space into K regions, whereas the class centers (plotted as black circles) do not match with the actual class means and the samples are spread over the entire space.

As λ increases, the class centers approach to the actual class means, and simultaneously the samples get closer to its corresponding class center thereby form multiple spheres.

As discussed in Section 2, the regularization term enforces the empirical class-conditional distributions to approximate the Gaussian distribution with the mean c k .

In conclusion, the proper value of λ makes the DNNs place the class-conditional Gaussian distributions far apart from each other, so the OOD samples are more likely to be located in the rest of the space.

As DNNs have become the dominant approach to a wide range of real-world applications and the cost of their errors increases rapidly, many studies have been carried out on measuring the uncertainty of a model's prediction, especially for non-Bayesian DNNs (Gal, 2016; Teye et al., 2018) .

Recently, Malinin & Gales (2018) defined several types of uncertainty, and among them, distributional uncertainty occurs by the discrepancy between the training and test distributions.

In this sense, the OOD detection task can be understood as modeling the distributional uncertainty, and a variety of approaches have been attempted, including the parameterization of a prior distribution over predictive distributions (Malinin & Gales, 2018) and training multiple classifiers for an ensemble method (Shalev et al., 2018; Vyas et al., 2018) .

The baseline method (Hendrycks & Gimpel, 2017 ) is the first work to define the confidence score by the softmax probability based on a given DNN classifier.

To enhance the reliability of detection, ODIN (Liang et al., 2018 ) applies two calibration techniques, i.e., temperature scaling (Hinton et al., 2015) and input perturbation , to the baseline method, which can push the softmax scores of ID and OOD samples further apart from each other.

Lee et al. (2018) uses the Mahalanobis distance from class means instead of the softmax score, assuming that samples of each class follows the Gaussian distribution in the latent space.

However, all of them utilize the DNNs for the discriminative (i.e, softmax) classifier, only optimized for classifying ID samples.

Our approach differs from the existing methods in that it explicitly learns the class-conditional Gaussian distributions and computes the score based on the Euclidean distance from class centers.

This paper introduces a deep learning objective to learn the multi-class generative classifier, by fusing the concept of Gaussian discriminant analysis with DNNs.

Unlike the conventional softmax classifier, our generative (or distance) classifier learns the class-conditional distributions to be separated from each other and follow the Gaussian distribution at the same time, thus it is able to effectively distinguish OOD samples from ID samples.

We empirically show that our confidence score beats other competing methods in detecting both OOD tabular data and OOD images, and also the distance classifier can be easily combined with various types of DNNs to further improve their performances.

@highlight

This paper proposes a deep generative classifier which is effective to detect out-of-distribution samples as well as classify in-distribution samples, by integrating the concept of Gaussian discriminant analysis into deep neural networks.
State-of-the-art Unsupervised Domain Adaptation (UDA) methods learn transferable features by minimizing the feature distribution discrepancy between the source and target domains.

Different from these methods which do not model the feature distributions explicitly, in this paper, we explore explicit feature distribution modeling for UDA.

In particular, we propose Distribution Matching Prototypical Network (DMPN) to model the deep features from each domain as Gaussian mixture distributions.

With explicit feature distribution modeling, we can easily measure the discrepancy between the two domains.

In DMPN, we propose two new domain discrepancy losses with probabilistic interpretations.

The first one minimizes the distances between the corresponding Gaussian component means of the source and target data.

The second one minimizes the pseudo negative log likelihood of generating the target features from source feature distribution.

To learn both discriminative and domain invariant features, DMPN is trained by minimizing the classification loss on the labeled source data and the domain discrepancy losses together.

Extensive experiments are conducted over two UDA tasks.

Our approach yields a large margin in the Digits Image transfer task over state-of-the-art approaches.

More remarkably, DMPN obtains a mean accuracy of 81.4% on VisDA 2017 dataset.

The hyper-parameter sensitivity analysis shows that our approach is robust w.r.t hyper-parameter changes.

Recent advances in deep learning have significantly improved state-of-the-art performance for a wide range of applications.

However, the improvement comes with the requirement of a massive amount of labeled data for each task domain to supervise the deep model.

Since manual labeling is expensive and time-consuming, it is therefore desirable to leverage or reuse rich labeled data from a related domain.

This process is called domain adaptation, which transfers knowledge from a label rich source domain to a label scarce target domain (Pan & Yang, 2009 ).

Domain adaptation is an important research problem with diverse applications in machine learning, computer vision (Gong et al., 2012; Gopalan et al., 2011; Saenko et al., 2010) and natural language processing (Collobert et al., 2011; Glorot et al., 2011) .

Traditional methods try to solve this problem via learning domain invariant features by minimizing certain distance metric measuring the domain discrepancy, for example Maximum Mean Discrepancy (MMD) (Gretton et al., 2009; Pan et al., 2008; and correlation distance (Sun & Saenko, 2016) .

Then labeled source data is used to learn a model for the target domain.

Recent studies have shown that deep neural networks can learn more transferable features for domain adaptation (Glorot et al., 2011; Yosinski et al., 2014) .

Consequently, adaptation layers have been embedded in the pipeline of deep feature learning to learn concurrently from the source domain supervision and some specially designed domain discrepancy losses Long et al., 2015; Sun & Saenko, 2016; Zellinger et al., 2017) .

However, none of these methods explicitly model the feature distributions of the source and target data to measure the discrepancy.

Inspired from the recent works by Wan et al. (2018) and Yang et al. (2018) , which have shown that modeling feature distribution of a training set improves classification performance, we explore explicit distribution modeling for UDA.

We model the feature distributions as Gaussin mixture distributions, which facilitates us to measure the discrepancy between the source and target domains.

Our proposed method, i.e., DMPN, works as follows.

We train a deep network over the source domain data to generate features following a Gaussian mixture distribution.

The network is then used to assign pseudo labels to the unlabeled target data.

To learn both discriminative and domain invariant features, we fine-tune the network to minimize the cross-entropy loss on the labeled source data and domain discrepancy losses.

Specifically, we propose two new domain discrepancy losses by exploiting the explicit Gaussian mixture distributions of the deep features.

The first one minimizes the distances between the corresponding Gaussian component means between the source and target data.

We call it Gaussian Component Mean Matching (GCMM).

The second one minimizes the negative log likelihood of generating the target features from the source feature distribution.

We call it Pseudo Distribution Matching (PDM).

Extensive experiments on Digits Image transfer tasks and synthetic-to-real image transfer task demonstrate our approach can provide superior results than state-of-the-art approaches.

We present our proposed method in Section 3, extensive experiment results and analysis in Section 4 and conclusion in Section 5.

Domain adaptation is an important research problem with diverse applications in machine learning, computer vision (Gong et al., 2012; Gopalan et al., 2011; Saenko et al., 2010) and natural language processing (Collobert et al., 2011; Glorot et al., 2011) .

According to the survey Pan & Yang (2009) , traditional domain adaptation methods can be organized into two categories: feature matching and instance re-weighting.

Feature matching aims to reduce the domain discrepancy via learning domain invariant features by minimizing certain distance metric, for example Maximum Mean Discrepancy (MMD) (Gretton et al., 2009; Pan et al., 2008; , correlation distance (Sun & Saenko, 2016) , Central Moment Discrepancy (CMD) Zellinger et al. (2017) and et al. Then labeled source data is used to learn a model for the target domain.

Instance reweighting aims to reduce the domain discrepancy via re-weighting the source instances according to their importance weights with respect to the target distribution (Huang et al., 2007) .

In the era of deep learning, studies have shown that deep neural networks can learn more transferable features for domain adaptation (Glorot et al., 2011; Yosinski et al., 2014) , therefore, domain adaptation layers have been embedded in the pipeline of deep feature learning to learn concurrently from the source domain supervision and some specially designed domain discrepancy losses Long et al., 2015; Sun & Saenko, 2016; Zellinger et al., 2017) .

Some recent works Ganin & Lempitsky (2014) , Tzeng et al. (2017) , Long et al. (2018) add a domain discriminator into the deep feature learning pipeline, where a feature generator and a domain discriminator are learned adversarially to generate domain invariant features.

All these works can be categorized as the feature matching type of domain adaptation method.

However, none of them models the feature distributions of the source and target data for distribution matching.

In this paper, we show that explicitly modeling the feature distributions enables us to measure the domain discrepancy more easily and helps us to propose new domain discrepancy losses.

Prototypical network (PN) was first proposed in Snell et al. (2017) for few shot learning, which shows that learning PN is equivalent to performing mixture density estimation on the deep features with an exponential density.

Recently, in Wan et al. (2018)'s and Yang et al. (2018) 's works, it has been shown that modeling the deep feature distribution of a training set as Gaussian mixture distribution improves classification performance.

As Gaussian density belongs to one type of exponential density, the models proposed in Wan et al. (2018)'s and Yang et al. (2018) 's works are variants of PN.

However, the two works study the classification problem in a single domain, which is different from our work on the problem of domain adaptation.

In Pan et al. (2019) , prototypical networks are first applied for domain adaptation.

Multi-granular domain discrepancy minimization at both class-level and sample-level are employed in Pan et al. (2019) to reduce the domain difference and achieves state-of-the-art results in various domain adaptation tasks.

However, in Pan et al. (2019) 's work, the deep feature distribution is modeled implicitly when they apply PN for UDA, in our work, we explicitly model the deep feature distribution as Gaussian mixture distribution for UDA.

In Unsupervised Domain Adaptation (UDA), we are given N s labeled samples

in the source domain and N t unlabeled samples

in the target domain.

The source and target samples share the same set of labels and are sampled from probability distributions P s and P t respectively with P s = P t .

The goal is to transfer knowledge learnt from the labeled source domain to the unlabeled target domain.

We model the deep embedded features of the source data as a Gaussian mixture distribution where the Gaussian component means act as the prototypes for each class.

Let {?? be the Gaussian component means and covariance matrices of the Gaussian mixture distribution, then the posterior distribution of a class y given the embedded feature f can be expressed as in Eqn.

1 where f = F (x, ??), F : X ??? R d is the embedding function with parameter ?? and d is the dimension of the embedded feature, p(c) is the prior probability of class c and C is the total number of classes.

With labeled source data

, a classification loss L cls can be computed as the cross-entropy between the posterior probability distribution and the one-hot class label as shown in Eq. 2 and following Wan et al. (2018) , a log likelihood regularization term L lkd can be defined as in Eq. 3, where f

The final loss function L GM for training a network with Gaussian mixture feature distribution is defined as

, where ?? is a non-negative weighting coefficient.

Notice, the distribution parameters {?? are learned automatically from data.

To match the deep feature distributions between the source and target data, we propose to match the corresponding Gaussian component means between them.

We utilize the network learnt on the labeled source data to assign pseudo labels to target samples.

As such, we denote the target samples with pseudo labels asD t = {(

.

We empirically estimate the Gaussian component means {??

where D s c andD t c denote the sets of source/target samples from class c, f

where || ?? || is the L 2 norm between two vectors.

Intuitively, if the source features and target features follow the same Gaussian mixture distribution, then the Gaussian component means of the same class from the two domains will be the same.

Thus minimizing L GCM M helps to reduce the domain discrepancy.

Better illustrated in Fig. 1 1 {??

, as the latter are learned directly from data and are used to assign pseudo labels for target data.

Figure 1: Illustration of the overall training objective.

This figure displays the model after we finish pre-training it with the labeled source data on L GM .

Different colors represent different classes.

Dotted ellipses represent Gaussian mixture distribution of the source embedded features.

The amorphous shapes represent pseudo labeled target feature distribution before we optimize the network further on the overall objective function in Eqn.

7.

GCMM loss tries to bring the corresponding Gaussian component means between the source data and pseudo labeled target data closer, represented by the black two-way arrows.

Minimizing GCMM brings the feature distributions of the source and target domains closer, thus reducing the domain discrepancy.

PDM loss tries to match the pseudo target feature distribution to the source Gaussian mixture distribution, represented by the colored one-way arrow.

Minimizing PDM increases the likelihood of target features on the source feature distribution, thus reducing the domain discrepancy.

Best viewed in color.

On the pseudo labeled target dataD t , we further propose to match the embedded target feature distribution with the source Gaussian mixture feature distribution via minimizing the following pseudo negative log likelihood loss, which we denoted as L P DM :

Minimizing L P DM 2 maximizes the likelihood of the pseudo labeled target features on the source Gaussian mixture feature distribution.

To achieve that, the network is enforced to learn an embedding function which produces similar embedded feature distributions between the source data and target data.

Otherwise, this term will induce a large loss value and dominate the overall objective function to be minimized.

Therefore, minimizing L P DM helps to reduce the domain discrepancy.

As we are using pseudo labeled target data to calculate this domain discrepancy loss function, we term it as Pseudo Distribution Matching (PDM) loss.

Furthermore, while minimizing GCMM loss brings the source and target feature distribution closer, minimizing PDM loss shapes the target feature distribution to be similar as the source Gaussian mixture distribution.

Thus, these two loss functions complement each other to reduce the distribution discrepancy.

Better illustrated in Fig. 1.

The overall training objective of DMPN can be written as follows:

where minimizing the first two terms of the objective function helps the model to learn discriminative features with the supervision from the labeled source data, and minimizing the last two terms helps to match the embedded feature distributions between the source and target domains so that the learned classifier from the labeled source data can be directly applied in the target domain.

The whole model is illustrated in Fig. 1 .

Training Procedure.

To train DMPN, we first pre-train a network with labeled source data on L GM .

Then mini-batch gradient descent algorithm is adopted for further optimization of the network on 2 Notice, gradient from LP DM does not back-propagate to update {?? .

We learn source distribution parameters only from labeled source data.

Eqn.

7, where half of the samples in the mini-batch are from labeled source data D s and the other half are from unlabeled target data D t .

To obtain pseudo labels for the unlabeled target data, we use the learned source distribution parameters to calculate the class probabilities for each target data point as in Eqn.

1 and assign the class with the largest probability as the pseudo label.

To remedy the error of the self-labeling, we took similar approach as in French et al. (2018) and Pan et al. (2019) to filter unlabeled target data points whose maximum predicted class probability is smaller than some threshold.

Apart from that, we also propose to weight the contribution of each sample to the discrepancy loss based on the predicted probability.

In this way, less confidently predicted target samples will make smaller contributions in the training process.

Inference.

For inference, we first apply the learned embedding function F on the target data, then we will use the learned distribution parameters to calculate the class probabilities for each target data point as in Eqn.

1. Finally, we output the class with the largest probability for each target data point as our prediction.

There is another type of domain adaptation problem, called Supervised Domain Adaptation (SDA) in the literature.

In SDA, we are provided with a large amount of labeled source data and a small amount of labeled target data, the goal is to find a hypothesis that works well in the target domain.

By employing pseudo labeled target data in the training process, our method can be considered as working on a generalized problem of SDA, where the labeled target data is noisy.

Ben-David et al. (2010) has proved that we can bound the target error of a domain adaptation algorithm that minimizes a convex combination of empirical source and target error in SDA as follows:

where

is the convex combination of the source and target error with ??

, f s and f t are the labeling function in the source and target domains respectively, h is a hypothesis in class H,

| measures the domain discrepsancy in the hypothesis space H and ?? = s (h * ) + t (h * ) is the combined error in two domains of the joint ideal hypothesis h * = arg min h???H s (h) + t (h).

Denote the noise ratio of the target labeling function to be ??, the convex combination of the source and noisy target error as?? ?? (h) = ?? t (h) + (1 ??? ??) s (h), where t (h) is the target error on the noisy target labeling function, then we can bound the target error as follows:

In summary, this bound is decomposed into three parts: the domain discrepancy d H???H , the error ?? of the ideal joint hypothesis and the noise ratio ?? of the pseudo labels.

In DMPN, we minimize the first term through minimizing the domain discrepancy losses, as d H???H is small when the source features and target features have similar distribution and minimizing the domain discrepancy losses makes the source and target feature to distribute similarly.

The second term is assumed to be small, as otherwise there is no classifier that performs well on both domains.

Finally, during training, as we continuously improve the accuracy of the classifier for target data, we get more and more accurate predictions, thus reducing the noise ratio ??.

We empirically verify that ?? is decreasing in Section 4.2.

digits from '0' to '9'.

The MNIST dataset consists of 70k images and the USPS dataset has 9.3k images.

Unlike MNIST and USPS, the SVHN (S) dataset is a real-world Digits dataset of house numbers in Google street view images and contains 100k cropped Digits images.

We follow the standard evaluation protocol (Tzeng et al., 2017; Pan et al., 2019) .

We consider three directions of adaptation: M ??? U, U ??? M and S ??? M. For the transfer between MNIST and USPS, we sample 2k images from MNIST training set (60,000) and 1.8k images from USPS training set (7,291) for adaptation and evaluation is reported on the standard test sets: MNIST (10,000), USPS (2,007).

For S ??? M, we use the whole training set SVHN (73,257) and MNIST (60,000) for adaptation and evaluation is reported on the standard test set MNIST (10,000).

In addition, we use the same CNN architecture, namely a simple modified version of LeCun et al. (1998) to be diagonal and the prior probability to be p(c) = 1/C when pre-training the network on the labeled source data.

The three trade-off parameters ??, ?? and ?? in Eqn.

7 are simply set to be 0.1, 1, 0.1.

We strictly follow Pan et al. (2019) and set the embedding dimension d as 10/512 for Digits/synthetic-to-real image transfer.

We implement DMPN with Pytorch.

We use ADAM with 0.0005 weight decay and 0.9/0.999 momentum for training and set the mini-batch size to be 128/120 in Digits/synthetic-to-real image transfer.

We train the network for 350 epochs for the Digits Image transfer tasks.

The learning rate is initially set to be 1e-5 for the covariance matrices and 1e-3 for the other parameters 4 and is decayed by 0.1 at epoch 150 and 250.

For the synthetic-to-real image transfer, we fix the learning rate to be 1e-6 and train the network for 100 epochs 4 .

Finally, for the Digits Image transfer tasks, we apply weighted PDM loss to remedy the labeling error, where each sample is weighted by the maximum predicted class probability.

For the synthetic-to-real image transfer task, we apply filtering to remedy the labeling error, where only target examples with maximum predicted probability over 0.8 is used for training.

Following the standard, for Digits Image transfer tasks, we adopt the classification accuracy on target domain as evaluation metric and for synthetic-to-real image transfer, we use the average per class accuracy for evaluation metric.

We will publish our code upon acceptance.

Compared Methods.

To demonstrate the benefits of our proposed method, we compare it with the following approaches: (1) Source-only directly exploits the classification model trained on source domain to classify target samples. (Tzeng et al., 2017) separates the source feature learning and target feature learning using different networks and use a domain discriminator to learn domain invariant features.

(7) JAN (Long et al., 2017) aligns the joint distribution of the network activation of multiple layers across domains. (8) MCD (Saito et al., 2018) employs task-specific decision boundaries to align the distributions of source and target domains.

(9) CDAN+E (Long et al., 2018) adds a conditional adversarial classifier on the deep feature learning pipeline to learn domain invariant features. (10) S-En+Mini-aug (French et al., 2018) modifies the mean teacher variant of temporal ensembling for UDA. (11) TPN (Pan et al., 2019) is the first work to apply PN for UDA.

TPN gen is the variant trained only with general-purpose domain discrepancy loss. (12) DMPN is our proposed method.

DMPN GCM M and DMPN P DM are trained only with GCMM loss and PDM loss respectively. (13) Train-on-target is an oracle that trained on labeled target samples.

Table 1 shows the results of all methods for the two tasks.

Overall, our proposed method achieves superior results than all the existing methods.

For the Digits Image transfer tasks, DMPN has improved the accuracy for M ??? U, U ??? M and S ??? M by 2.6%, 0.7% and 3.8% respectively compared to the second best.

We have made great advancement considering the second best accuracy results are already quite high.

For the task S ??? M, due to convergence reasons, we have added batch normalization layers to the original CNN architectures.

For fair comparison, we have re-run some experiments on other methods by adding batch normalization layers to them.

For methods whose public code are not available, we simply report the accuracy results with the original CNN architecture.

For ADDA, adding batch normalization layer has improved its accuracy result from 76.0% to 83.6%, which has an increase of 7.6% of accuracy.

However, we doubt adding batch normalization layers will have the same effect on TPN, as TPN already has a quite high accuracy.

Nonetheless, we think our accuracy result of 96.8% on this task will be difficult for the other methods to surpass even with batch normalization layers.

For the Synthetic-to-real image transfer task, we only compare with methods without extensive data augmentations and our method has increased the state-of-the-art single model mean accuracy by 1.0%.

TPN gen reduces the domain discrepancy via minimizing the pairwise Reproducing Kernel Hilbert Space (RKHS) distances among the corresponding prototypes of the source-only, target-only and source-target data.

In DMPN GCMM we minimize the L 2 distance between the corresponding Gaussian component means of the source and target data.

The L 2 distance can be viewed as the distance in a Linear RKHS space.

The calculation of our proposed GCMM loss is much simpler than the general purpose loss, yet with explicitly modeling of the feature distributions, DMPN GCMM has a gain of accuracy of 3.0%, 1.1%, 6.3% and 6.6% Ablation Analysis.

In Table 1 , combining GCMM loss and PDM loss helps to increase the accuracy results, showing that the two domain discrepancy losses are compatible to each other.

DMPN GCMM performs better than or similar to almost all other domain adaptation methods and DMPN PDM performs better than most of them.

Convergence Analysis.

Figure 2 (a) shows the training progress of DMPN.

The GCMM loss and PDM loss keep decreasing with more training epochs.

The prediction accuracy on the unlabeled target data keeps increasing.

And the noise ratio ?? decreases along the training process, from the initial value of 38.6% decreases to 22.9%, which supports our theoretical analysis in Section 3.5.

Figure 3 shows the t-SNE visualizations of the source and target embedded features during training, which shows that target classes are becoming increasingly well discriminated by the source classifier.

shows the sensitivity analysis on the hyper-parameters ??, ?? and ?? with the other hyper-parameters fixed.

Overall, the experiment results show that we can get similar accuracy results or even better when changing the hyper-parameters in a certain range, demonstrating that our method is robust against hyper-parameter changes.

The sensitivity analysis on the confidence threshold is in the Appendix A.2, which shows our method is robust against threshold value.

In this paper, we propose Distribution Matching Prototypical Network (DMPN) for Unsupervised Domain Adaptation (UDA) where we explicitly model and match the deep feature distribution of the source and target data as Gaussian mixture distributions.

Our work fills the gap in UDA where stateof-the-art methods assume the deep feature distributions of the source and target data are unknown when minimizing the discrepancy between them.

We propose two new domain discrepancy losses based on the Figure 4 : Sensitivity analysis on confidence threshold.

Fig. 4 shows the sensitivity analysis of our method on different values of confidence threshold on VisDA 2017 dataset.

The experiment results show that we can get similar accuracy results or even better when changing the confidence threshold in a certain range, demonstrating that our method is robust against hyper-parameter changes.

A.3 OFFICE-HOME TRANSFER Table 3 presents experiment results of state-of-the-art UDA methods and our method on OfficeHome dataset.

Our method gives the best accuracy results in all transfer tasks, showing the effectiveness of our method.

In this experiment, we train the network for 100 epochs.

The learning rate is initially set to be 1e-5 for all the parameters and is decayed by 0.1 at epoch 60 and 80.

1+e ?????p respectively, where ?? is set to be the default value 10, p is the training process changing from 0 to 1.

<|TLDR|>

@highlight

We propose to explicitly model deep feature distributions of source and target data as Gaussian mixture distributions for Unsupervised Domain Adaptation (UDA) and achieve superior results in multiple UDA tasks than state-of-the-art methods.
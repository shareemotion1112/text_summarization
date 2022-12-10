Supervised deep learning methods require cleanly labeled large-scale datasets, but collecting such data is difficult and sometimes impossible.

There exist two popular frameworks to alleviate this problem: semi-supervised learning and robust learning to label noise.

Although these frameworks relax the restriction of supervised learning, they are studied independently.

Hence, the training scheme that is suitable when only small cleanly-labeled data are available remains unknown.

In this study, we consider learning from bi-quality data as a generalization of these studies, in which a small portion of data is cleanly labeled, and the rest is corrupt.

Under this framework, we compare recent algorithms for semi-supervised and robust learning.

The results suggest that semi-supervised learning outperforms robust learning with noisy labels.

We also propose a training strategy for mixing mixup techniques to learn from such bi-quality data effectively.

Learning from imperfect data is essential for applying machine learning, especially data-hungry deep learning, to real-world problems.

One approach to handling this problem is semi-supervised learning (SSL), where training data consist of a small amount of labeled data and a large amount of unlabeled data.

Another approach is robust learning to label noise (RLL), wherein all data are labeled, but some of them are mislabeled.

SSL leverages large unlabeled data to improve the performance of supervised learning on a limited number of labeled data.

In the context of deep SSL, one effective method is to train neural networks to maintain consistency for a small perturbation of unlabeled inputs BID7 ; BID10 ; BID11 ).

BID8 refers these methods as consistency regularization.

In the RLL setting, learners need to enhance their performance using corrupted labels and avoid the performance deterioration caused by such data.

This requirement is particularly important for deep neural networks because they have ample capacity to remember whole samples even if their labels are completely random BID0 BID14 ).

To tackle this problem, some methods use a small amount of clean data to estimate noise transition matrix BID12 ; BID2 ) or to learn to select possibly correctly-labeled samples BID4 ; BID3 ).Although both SSL and RLL aim to alleviate the limited-data problem, they have been studied independently and evaluated using different benchmarks.

However, if only a small amount of clean data is available, they can be regarded as similar problems.

In such as situation, can RLL outperform SSL under the same settings?

This question was our initial motivation to unify these two lines of research.

In this paper, we introduce a generalization of SSL and RLL, based on the concept of trusted data BID1 ; BID2 ) in the literature of RLL.

More precisely, we assumed that some labels are guaranteed to be clean, and the rest are noisy.

The two learning frameworks can be unified by controlling the ratio of corrupted labels to all labels and the noisiness of label corruption.

Using the shared evaluation procedure in BID8 , we compared recent SSL and RLL algorithms using image classification task and found that the existing RLL methods using a small amount of clean data cannot outperform SSL under this setting.

This finding suggests that such RLL algorithms cannot use noisy labels effectively.

Therefore, it is necessary to adaptively use SSL and RLL in a data-driven manner.

As a baseline learning algorithm, we propose combining the mixup losses for SSL BID11 ) and RLL BID15 ); the results obtained are comparable to those of SSL-and RLL-specific methods and indicate the effective use of useful information from noisy labels.

In this section, we describe the setting of learning from bi-quality data, which is a generalization of SSL and RLL.

In this formulation, we assume that the given data consist of two parts: trusted data D T (for which the labels are always correct) and untrusted data D U ( for which some labels might be wrong).

Learners are allowed to access the information irrespective of whether each sample is from D T or D U .Let us denote the ratio of trusted and untrusted data to the entire data p as follows: DISPLAYFORM0 We also introduce quality as DISPLAYFORM1 where p T (y|x) and p U (y|x) are the conditional probabilities of labels when inputs of trusted data and untrusted data, respectively, are given, and D is a divergence between two probability distributions (e.g., Kullback-Leibler divergence).

In equation 2, q ∈ [0, 1].

Obviously, q = 0 if the labels are completely independent of the input features (i.e., completely random), and q = 1 if the labels are clean.

We assume that the quality of untrusted data q U is in [0, 1] , and that of trusted data q T is 1.We believe that this setting is realistic.

Suppose that there are unlabeled data.

Under budget constraint, one strategy to label this dataset as training data is to divide the data to two parts and spend most of the budget on high-quality labeling to acquire D T (e.g., using experts) and the rest on lower-quality labeling to obtain D U (e.g., using crowdsourcing), where usually |D T | ≪ |D U |.Under this generalized framework, SSL is equivalent to the particular case where q = 0, that is, labels of untrusted data are entirely random and give no useful information.

In SSL, such labels are usually ignored, and the data are treated as unlabeled.

Meanwhile, RLL without trustable data is another special case, where p = 0 and 0 q < 1.

In some studies on RLL, the setting p > 0 is used explicitly BID1 ; BID6 ; BID2 ) and implicitly BID4 ; BID3 ).Intuitively, when q is relatively high, we can expect some performance gain by utilizing untrusted, but somewhat informative, labels with RLL.

On the contrary, when q is almost zero, untrusted data are not informative anymore, and better generalization may be obtained using only standard SSL.

A critical problem in practice, however, is that we cannot know the exact quality of untrusted data, and, therefore, we cannot decide the best learning strategy in advance.

To handle such data, we need an adaptive mechanism to properly fuse SSL and RLL in a data-driven manner.

To realize this goal, we propose to combine techniques for SSL and RLL.

We use the convex combination of loss functions for SSL L semi and RLL L robust .

That is, with an additional hyperparameter γ, the loss function for untrusted data is defined as follows: DISPLAYFORM2 For L robust and L semi , any loss functions for SSL and RLL can be used 1 .

In this study, as a baseline of this setting, we combine mixup techniques for SSL BID11 ) and RLL BID15 ).1 Here, each L· is cross entropy loss.

DISPLAYFORM3 3 MIXMIXUP mixup is a regularization technique, where neural network models are trained on virtual training pairs (x,ỹ), wherex DISPLAYFORM4 Here, (x i , y i ) and (x j , y j ) are sampled from training data, and λ ∈ [0, 1] is sampled from Beta distribution.

BID15 showed that this method alleviates the performance decrease under label corruption.

In BID11 , mixup is used for SSL as consistency regularization BID8 ).

Here, for unlabeled inputs x We show the details in Algorithm 1.

The parameters of neural networks are updated with a combination of losses on trusted data L T and untrusted data L U as L T + σ(k)L U .

We use sigmoid scheduling σ(k) as BID11 .Because this method mixes the mixup losses, hereinafter we refer to our method as mixmixup.

We use CIFAR-10 (Krizhevsky FORMULA1 , which has 50,000 of 10-category images.

Following the common protocol of SSL in BID8 , we split the training dataset into 45,000 images for training and 5,000 for validation.

To simulate bi-quality data, we sample 4,000 examples from the training data as trusted data and the rest as untrusted data.

We randomly replace each label of the untrusted samples with another one with a given probability, which is a common protocol in RLL BID15 BID4 ).

We use the non-corrupted validation data and test data for hyperparameter tuning and final evaluation, respectively.

Following BID8 and BID11 , we use WRN-28-2 (Wide ResNet 28-2, proposed in BID13 ) as the image classifier.

To optimize the network, we use SGD with a learning rate of 0.1, a momentum of 0.9, a weight decay of 1.0 × 10 −4 and a minibatch size of 256.

We train networks for 1.6 × 10 5 iterations, following BID11 .

We implement the model with PyTorch v1.0 BID9 ) and tune hyperparameters α, β and γ in Algorithm 1 with a Bayesian optimization algorithm in Optuna v0.8 2 .4.2 RESULTS 4.2.1 LEARNING FROM BI-QUALITY DATA According to the settings in Section 4.1, we trained WRN-28-2 with bi-quality data of different quality.

We randomly replace 40% and 100% of the labels of the untrusted data to simulate label BID15 4,000 N/A 0.78 B Basic 4,000 41,000 (q = 0.0) 0.29 Basic 4,000 41,000 (q = 0.6) 0.78 C input mixup BID15 ) 4,000 41,000 (q = 0.0) 0.43 input mixup BID15 4,000 41,000 (q = 0.6) 0.89 D mixmixup (ours) 4,000 41,000 (q = 0.0) 0.88 mixmixup (ours) 4,000 41,000 (q = 0.6) 0.90 BID4 ) 4,000 41,000 (q = 0.6) 0.87 GLC † † BID2 ) 4,000 41,000 (q = 0.6) 0.84 C mixmixup (ours) 4,000 41,000 (q = 0.0) 0.88 mixmixup (ours) 4,000 41,000 (q = 0.6) 0.90 corruption.

From the definition of quality (equation 2), the quality of these data is 0.6 and 0.0, respectively.

The former corresponds to RLL with trusted data, and the latter to SSL.

TAB1 presents the test accuracy of SSL methods BID11 ; BID10 ; BID7 , TAB1 A) and RLL methods with trusted data BID4 ; BID2 , TAB1 B) with the results obtained using mixmixup TAB1 .

In all the experiments, the same network (WRN-28-2) and the same dataset split mentioned above are used.

Note that SSL methods in TAB1 A does not use labels of untrusted data.

Surprisingly, our results suggest that SSL TAB1 A) can provide better performance than RLL TAB1 with trusted data for identical settings.

This result suggests the difficulty of learning robustly from partially corrupted labels preventing overfitting.

Further, mixmixup can handle both SSL and RLL.

Moreover, the accuracy when the quality is 0.6 (corresponding to RLL) is superior to that for quality 0.0 (corresponding to SSL), indicating that mixmixup can effectively use information from corrupted labels.

In this paper, we introduce a novel framework of weakly supervised learning by unifying SSL and RLL, which have been independently studied.

To handle this problem, we propose to mix mixup for SSL and RLL.

This method empirically works well and achieves competitive results with semisupervised and robust learning specific methodologies.

In addition, our experiments indicate that the performance of some RLL with trusted data might be inferior to that of SSL under identical settings.

This result suggests that the existing RLL methods cannot effectively exploit the information which should be extracted from noisy labels.

Our proposed method does not use the estimated quality; instead some hyperparameters are introduced.

The use of quality estimation may ease hyperparameter tuning, but is still an open question.

This work was supported by JSPS KAKENHI Grant Number JP19H04166.

<|TLDR|>

@highlight

We propose to compare semi-supervised and robust learning to noisy label under a shared setting

@highlight

The authors propose a strategy based on mixup for training a model in a formal setting that includes the semi-supervised and the robust learning tasks as special cases.
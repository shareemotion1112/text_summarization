Semi-supervised learning (SSL) is a study that efficiently exploits a large amount of unlabeled data to improve performance in conditions of limited labeled data.

Most of the conventional SSL methods assume that the classes of unlabeled data are included in the set of classes of labeled data.

In addition, these methods do not sort out useless unlabeled samples and use all the unlabeled data for learning, which is not suitable for realistic situations.

In this paper, we propose an SSL method called selective self-training (SST), which selectively decides whether to include each unlabeled sample in the training process.

It is also designed to be applied to a more real situation where classes of unlabeled data are different from the ones of the labeled data.

For the conventional SSL problems which deal with data where both the labeled and unlabeled samples share the same class categories, the proposed method not only performs comparable to other conventional SSL algorithms but also can be combined with other SSL algorithms.

While the conventional methods cannot be applied to the new SSL problems where the separated data do not share the classes, our method does not show any performance degradation even if the classes of unlabeled data are different from those of the labeled data.

Recently, machine learning has achieved a lot of success in various fields and well-refined datasets are considered to be one of the most important factors (Everingham et al., 2010; Krizhevsky et al., 2012; BID6 .

Since we cannot discover the underlying real distribution of data, we need a lot of samples to estimate it correctly (Nasrabadi, 2007) .

However, creating a large amount of dataset requires a huge amount of time, cost and manpower BID3 .Semi-supervised learning (SSL) is a method relieving the inefficiencies in data collection and annotation process, which lies between the supervised learning and unsupervised learning in that both labeled and unlabeled data are used in the learning process (Chapelle et al., 2009; BID3 .

It can efficiently learn a model from fewer labeled data using a large amount of unlabeled data BID15 .

Accordingly, the significance of SSL has been studied extensively in the previous literatures BID18 BID5 Kingma et al., 2014; BID4 BID2 .

These results suggest that SSL can be a useful approach in cases where the amount of annotated data is insufficient.

However, there is a recent research discussing the limitations of conventional SSL methods BID3 .

They have pointed out that conventional SSL algorithms are difficult to be applied to real applications.

Especially, the conventional methods assume that all the unlabeled data belong to one of the classes of the training labeled data.

Training with unlabeled samples whose class distribution is significantly different from that of the labeled data may degrade the performance of traditional SSL methods.

Furthermore, whenever a new set of data is available, they should be trained from the scratch using all the data including out-of-class 1 data.

In this paper, we focus on the classification task and propose a deep neural network based approach named as selective self-training (SST) to solve the limitation mentioned above.

Unlike the conventional self-training methods in (Chapelle et al., 2009) , our algorithm selectively utilizes the unlabeled data for the training.

To enable learning to select unlabeled data, we propose a selection network, which is based on the deep neural network, that decides whether each sample is to be added or not.

Different from BID12 , SST does not use the classification results for the data selection.

Also, we adopt an ensemble approach which is similar to the co-training method BID0 ) that utilizes outputs of multiple classifiers to iteratively build a new training dataset.

In our case, instead of using multiple classifiers, we apply a temporal ensemble method to the selection network.

For each unlabeled instance, two consecutive outputs of the selection network are compared to keep our training data clean.

In addition, we have found that the balance between the number of samples per class is quite important for the performance of our network.

We suggest a simple heuristics to balance the number of selected samples among the classes.

By the proposed selection method, reliable samples can be added to the training set and uncertain samples including out-of-class data can be excluded.

SST is a self-training framework, which iteratively adopts the newly annotated training data (details in Section 2.1).

SST is also suitable for the incremental learning which is frequently used in many real applications when we need to handle gradually incoming data.

In addition, the proposed SST is suitable for lifelong learning which makes use of more knowledge from previously acquired knowledge BID10 Carlson et al., 2010; Chen & Liu, 2018) .

Since SSL can be learned with labeled and unlabeled data, any algorithm for SSL may seem appropriate for lifelong learning.

However, conventional SSL algorithms are inefficient when out-of-class samples are included in the additional data.

SST only add samples having high relevance in-class data and is suitable for lifelong learning.

The main contributions of the proposed method can be summarized as follows:??? For the conventional SSL problems, the proposed SST method not only performs comparable to other conventional SSL algorithms but also can be combined with other algorithms.??? For the new SSL problems, the proposed SST does not show any performance degradation even with the out-of-class data.??? SST requires few hyper-parameters and can be easily implemented.??? SST is more suitable for lifelong learning compared to other SSL algorithms.

To prove the effectiveness of our proposed method, first, we conduct experiments comparing the classification errors of SST and several other state-of-the-art SSL methods (Laine & Aila, 2016; BID9 Luo et al., 2017; Miyato et al., 2017) in conventional SSL settings.

Second, we propose a new experimental setup to investigate whether our method is more applicable to realworld situations.

The experimental setup in BID3 samples classes among in-classes and out-classes.

In the experimental setting in this paper, we sample unlabeled instances evenly in all classes. (details in Section 6.6 of the supplementary material).

We evaluate the performance of the proposed SST using three public benchmark datasets: CIFAR-10, CIFAR-100 BID8 Hinton, 2009), and SVHN (Netzer et al., 2011) .

In this section, we introduce the background of our research.

First, we introduce some methods of self-training (McLachlan, 1975; BID16 BID17 ) on which our work is based.

Then we describe consistency regularization-based algorithms such as temporal ensembling (Laine & Aila, 2016) .

Self-training method has long been used for semi-supervised learning (McLachlan, 1975; BID5 BID16 BID17 .

It is a resampling technique that repeatedly labels unlabeled training samples based on the confidence scores and retrains itself with the selected pseudo-annotated data.

Our proposed method can also be categorized as a self-training method.

FIG0 shows an overview of our SSL system.

Since our proposed algorithm is based on the selftraining, we follow its learning process.

This process can be formalized as follows.

However, most self-training methods assume that the labeled and unlabeled data are generated from the identical distribution.

Therefore, in real-world scenarios, some instances with low likelihood according to the distribution of the labeled data are likely to be misclassified inevitably.

Consequently, these erroneous samples significantly lead to worse results in the next training step.

To alleviate this problem, we adopt the ensemble and balancing methods to select reliable samples.

Consistency regularization is one of the popular SSL methods and has been referred to many recent researches (Laine & Aila, 2016; Miyato et al., 2017; BID9 .

Among them, ?? model and temporal ensembling are widely used (Laine & Aila, 2016) .

They have defined new loss functions for unlabeled data.

The ?? model outputs f (x) andf (x) for the same input x by perturbing the input with different random noise and using dropout BID8 , and then minimizes DISPLAYFORM0 2 ) between these output values.

Temporal ensembling does not make different predictions f (x) andf (x), but minimizes the difference ( DISPLAYFORM1 2 ) between the outputs of two consecutive iterations for computational efficiency.

In spite of the improvement in performance, they require lots of things to consider for training.

These methods have various hyperparameters such as 'ramp up', 'ramp down', 'unsupervised loss weight' and so on.

In addition, customized settings for training such as ZCA preprocessing and mean-only batch normalization BID7 are also very important aspects for improving the performance BID3 .Algorithm 1 Training procedure of the proposed SST Require: x i , y i : training data and label Require: L, U: labeled and unlabeled datasets Require: I U : set of unlabeled sample indices Require: f n (??; ?? n ), f cl (??; ?? c ) and f sel (??; ?? s ): trainable SST model Require: ??, , K, K re : hyper-parameters, 0 ??? ?? < 1, 0 ??? < 1 1: randomly initialize ?? n , ?? c , ?? s 2: train f n (??; ?? n ), f cl (??; ?? c ) and f sel (??; ?? s ) for K epochs using L 3: repeat 4:initialize r DISPLAYFORM2 for each i ??? I U do 6: DISPLAYFORM3 end if 10: DISPLAYFORM4 if z i > 1 ??? then

I S ??? I S ??? {i} DISPLAYFORM0 retrain f n (??; ?? n ), f cl (??; ?? c ) and f sel (??; ?? s ) for K re epochs using T 19: until stopping criterion is true

In this section, we introduce our selective self-training (SST) method.

The proposed model consists of three networks as shown in the bottom part of FIG0 .

The output of the backbone network is fed into two sibling fully-connected layers -a classification network f cl (??; ?? c ) and a selection network f sel (??; ?? s ), where ?? c and ?? s are learnable parameters for each of them.

In this paper, we define the classification result and the selection score as r i = f cl (f n (x i ; ?? n ); ?? c ) and DISPLAYFORM0 respectively, where f n (??; ?? n ) denotes the backbone network with learnable parameters ?? n .

Note that we define r i as the resultant label and it belongs to one of the class labels r i ??? Y = {1, 2, ?? ?? ?? , C}. The network architecture of the proposed model is detailed in Section 6.2 in the supplementary material.

As shown in FIG0 , the proposed SST method can be represented in the following four steps.

First, SST trains the network using a set of the labeled data L = {(x i , y i ) | i = 1, ?? ?? ?? , L}, where x i and y i ??? {1, 2, ?? ?? ?? , C} denote the data and the ground truth label respectively, which is a standard supervised learning method.

The next step is to predict all the unlabeled data U = {x i | i = L + 1, ?? ?? ?? , N } and select a subset of the unlabeled data {x i |i ??? I S } whose data have high selection scores with the current trained model, where I S denotes a set of selected sample indices from I U = {L + 1, ?? ?? ?? , N }.

Then, we annotate the selected samples with the pseudo-categories evaluated by the classification network and construct a new training dataset T composed of L and U S = {(x i ,?? i )|i ??? I S }.

After that, we retrain the model with T and repeat this process iteratively.

The overall process of the SST is described in Algorithm 1 and the details of each of the four steps will be described later.

The SST algorithm first trains a model with supervised learning.

At this time, the entire model (all three networks) is trained simultaneously.

The classification network is trained using the softmax function and the cross-entropy loss as in the ordinary supervised classification learning task.

In case of the selection network, the training labels are motivated by discriminator of generative adversarial networks (GAN) (Goodfellow et al., 2014; BID13 .

When i-th sample x i with the class label y i is fed into the network, the target for the selection network is set as: DISPLAYFORM0 where I L = {1, ?? ?? ?? , L} represents a set of labeled sample indices.

The selection network is trained with the generated target g i .

Especially, we use the sigmoid function for the final activation and the binary cross-entropy loss to train the selection network.

Our selection network does not utilize the softmax function because it produces a relative value and it can induce a high value even for an out-of-class sample.

Instead, our selection network is designed to estimate an absolute confidence score using the sigmoid activation function.

Consequently, our final loss function is a sum of the classification loss L cl and the selection loss L sel : DISPLAYFORM1

After learning the model in a supervised manner, SST takes all instances of the unlabeled set U as input and predicts classification result r i and the selection score s i , for all i ??? I U .

We utilize the classification result and selection score (r i and s i ) to annotate and choose unlabeled samples, respectively.

In the context of self-training, removing erroneously annotated samples is one of the most important things for the new training dataset.

Thus, we adopt temporal co-training and ensemble methods for selection score in order to keep our training set from contamination.

First, let r t i and r t???1 i be the classification results of the current and the previous iterations respectively and we utilize the temporal consistency of these values.

If these values are different, we set the ensemble score z i = 0 to reduce uncertainty in selecting unlabeled samples.

Second, inspired by (Laine & Aila, 2016), we also utilize multiple previous network evaluations of unlabeled instances by updating the ensemble score z i = ??z i + (1 ??? ??)s i , where ?? is a momentum weight for the moving average of ensemble scores.

However, the aim of our ensembling approach is different from (Laine & Aila, 2016).

They want to alleviate different predictions for the same input, which are resulted from different augmentation and noise to the input.

However, our aim differs from theirs in that we are interested in selecting reliable (pseudo-)labeled samples.

After that, we select unlabeled samples with high ensemble score z i .

It is very important to set an appropriate threshold because it decides the quality of the added unlabeled samples for the next training.

If the classification network is trained well on the labeled data, the training accuracy would be very high.

Since the selection network is trained with the target g i generated from the classification score r i , the selection score s i will be close to 1.0.

We set the threshold to 1 ??? and control it by changing .

In this case, if the ensemble score z i exceeds 1 ??? , the pseudo-label of the unlabeled sample?? i is set to the classification result r i .

When we construct a new training dataset, we keep the number of samples of each class the same.

The reason is that if one class dominates the others, the classification performance is degraded by the imbalanced distribution (Fern??Ndez et al., 2013) .

We also empirically found that naively creating a new training dataset fails to yield good performance.

In order to fairly transfer the selected samples to the new training set, the amount of migration in each class should not exceed the number of the class having the least selected samples.

We take arbitrary samples in every class as much as the maximum number satisfying this condition.

The new training set T is composed of both a set of labeled samples L and a set of selected unlabeled samples U S .

The number of selected unlabeled samples is the same for all classes.

After combining the labeled and selected pseudo-labeled data, the model is retrained with the new dataset for K re epochs.

In this step, the label for the selection network is obtained by a process similar to Eq. (1).

Above steps (except for Section 3.1) are repeated for M iterations until (near-) convergence.

Table 1 : Ablation study with 5 runs on the CIFAR-10 dataset.

'balance' denotes the usage of data balancing scheme during data addition as described in Sec. 3.3, 'ensemble' is for the usage of previous selection scores as in the 10th line of Algorithm 1, and 'multiplication' is the scheme of multiplying top-1 softmax output of the classifier network to the selection score and use it as a new selection score.

To evaluate our proposed SST algorithm, we conduct two types of experiments.

First, we evaluate the proposed SST algorithm for the conventional SSL problem where all unlabeled data are in-class.

Then, SST is evaluated with the new SSL problem where some of the unlabeled data are out-of-class.

In the case of in-class data, gradually gathering highly confident samples in U can help improve the performance.

On the other hand, in the case of out-of-class data, a strict threshold is preferred to prevent uncertain out-of-class data from being involved in the new training set.

Therefore, we have experimented with decay mode that decreases the threshold in log-scale and fixed mode that fixes the threshold in the way described in Section 4.2.

We have experimented our method with 100 iterations and determined epsilon by cross-validation in decay modes.

In case of fixed modes, epsilon is fixed and the number of iteration is determined by cross-validation.

The details about the experimental setup and the network architecture are presented in Section 6.1, 6.2 of the supplementary material.

We experiment with a couple of simple synthetic datasets (two moons, four spins) and three popular datasets which are SVHN, CIFAR-10, and CIFAR-100 (Netzer et al., 2011; BID8 .

The settings of labeled versus unlabeled data separation for each dataset are the same with (Laine & Aila, 2016; Miyato et al., 2017; BID9 .

More details are provided in Section 6.3 in the supplementary material.

The experimental results of the synthetic datasets can be found in Section 6.4 of the supplementary material.

We have performed experiments on CIFAR-10 dataset with the combination of three types of components.

As described in Table 1 , these are whether to use data balancing scheme described in Section 3.3 (balance), whether to use selection score ensemble in the 10th line of Algorithm 1 (ensemble) and whether to multiply the selection score with the top-1 softmax output of the classifier network to set a new selection score for comparison with the threshold (multiplication).

First, when SST does not use all of these, the error 21.44% is higher than that of the supervised learning which does not use any unlabeled data.

This is due to the problem of unbalanced data mentioned in subsection 3.3.

When the data balance is used, the error is 14.43%, which is better than the baseline 21.44%.

Adding the ensemble scheme results in 11.82% error, and the multiplication scheme shows a slight drop in performance.

Since all of the experiments use the same threshold, the number of candidate samples to be added is reduced by the multiplication with the top-1 softmax output and the variation becomes smaller because only confident data are added.

However, we have not used the multiplication scheme in what follows because the softmax classification output is dominant in multiplication.

Therefore, we have used only balance and ensemble schemes in the following experiments.

TAB2 shows the experiment results of supervised learning, conventional SSL algorithms and the proposed SST on CIFAR-10, SVHN and CIFAR-100 datasets.

Our baseline model with supervised learning performs slightly better than what has been reported in other papers (Laine & Aila, 2016; BID9 Luo et al., 2017) because of our different settings such as Gaussian noise Figure 2 : SST result on CIFAR-10, SVHN, and CIFAR-100 datasets with 5 runs.

The x-axis is the iteration, the blue circle is the average of the number of data used for training, and the red diamond is the average accuracy.

BID9 12.31 ?? 0.28% 3.95 ?? 0.21% -?? model (Laine & Aila, 2016) 12.36 ?? 0.31% 4.82 ?? 0.17% 39.19 ?? 0.36% TempEns (Laine & Aila, 2016) 12.16 ?? 0.24% 4.42 ?? 0.16% 38.65 ?? 0.51% TempEns + SNTG (Luo et al., 2017) 10.93 ?? 0.14% 3.98 ?? 0.21% 40.19 ?? 0.51%* VAT (Miyato et al., 2017) 11.36 ?? 0.34% 5.42 ?? 0.22% -VAT + EntMin (Miyato et al., 2017) 10.55 ?? 0.05% 3.86 ?? 0.11% -pseudo-label (Lee, 2013; BID3 17.78 ?? 0.57% 7.62 ?? 0.29% -Proposed method (SST)* 11.82 ?? 0.40% 6.88 ?? 0.59% 34.89 ?? 0.75% SST + TempEns + SNTG* 9.99 ?? 0.31% 4.74 ?? 0.19% 34.94 ?? 0.54% on inputs, optimizer selection, the mean-only batch normalizations and the learning rate parameters.

For all the datasets, we have also performed experiments with a model of SST combined with the temporal ensembling (TempEns) and SNTG, labeled as SST+TempEns+SNTG in the table.

For the model, the pseudo-labels of SST at the last iteration is considered as the true class label.

Figure 2 shows the number of samples used in the training and the corresponding accuracy on the test set for each dataset.

The baseline network yields the test error of 18.97% and 5.57% when trained with 4,000 (sampled) and 50,000 (all) labeled images respectively.

The test error of our SST method reaches 11.82% which is comparable to other algorithms while SST+TempEns+SNTG model results 1.83% better than the SST-only model.

The baseline model for SVHN dataset is trained with 1,000 labeled images and yields the test error of 13.45%.

Our proposed method has an error of 6.88% which is relatively higher than those of other SSL algorithms.

Performing better than SST, SST+TempEns+SNTG reaches 4.74% of error which is worse than that of TempEns+SNTG model.

We suspect two reasons for this.

The first is that SVHN dataset is not well balanced, and the second is that SVHN is a relatively easy dataset, so it seems to be easily added to the hard labels.

With data balancing, the SST is still worse than other algorithms.

More details are provided in Section 6.5 in the supplementary material.

We think this phenomenon owes to the use of hard labels in SST where incorrectly estimated samples deteriorate the performance.

conjectured that the hyper-parameter in the current temporal ensembling and SNTG may not have been optimized.

We have experimented with the following settings for real-world applications.

The dataset is categorized into six animal and four non-animal classes as similarly done in BID3 .

In CIFAR-10, 400 images per animal class are used as the labeled data (total 2,400 images for 6 animal classes) and a pool of 20,000 images with different mixtures of both animal and non-animal classes are experimented as an unlabeled dataset.

In CIFAR-100, 5,000 labeled data (100 images per animal class) and a total of 20,000 unlabeled images of both classes with different mixed ratios are utilized.

Unlike the experimental setting in BID3 , we have experimented according to the ratio (%) of the number of out-of-class data in the unlabeled dataset.

More details are provided in Section 6.6 in the supplementary material.

As mentioned in Section 4, in the presence of out-of-class samples, a strict threshold is required.

If all of the unlabeled data is assumed to be in-class, the decay mode may be a good choice.

However, in many real-applications, out-of-class unlabeled data is also added to the training set in the decay mode and causes poor performance.

In avoidance of such matter, we have experimented on a fixed mode of criterion threshold on adding the unlabeled data.

Unlike the decay mode that decrements the threshold value, SST in the fixed mode sets a fixed threshold at a reasonably high value throughout the training.

Our method in the fixed mode should be considered more suitable for real-applications but empirically shows lower performances in FIG1 and TAB4 than when running in the decay mode.

The difference between the decay mode and the fixed mode are an unchangeable and the initial ensemble.

Setting a threshold value for the fixed mode is critical for a feasible comparison against the decay mode.

FIG1 shows the average of the results obtained when performing SST five times for each ratio in CIFAR-10.

As shown in FIG1 , as the number of iteration increases, the threshold in the decay mode decreases and the number of additional unlabeled data increases.

Obviously, while the different percentage of the non-animal data inclusion show different trends of training, in the cases of 0 ??? 75% of non-animal data included in the unlabeled dataset, the additionally selected training data shows an initial increase at 30 th ??? 40 th iteration.

On the other hand, when the unlabeled dataset is composed of only the out-of-class data, selective data addition of our method initiates at 55 th ??? 65 th training iteration.

This tendency has been observed in previous researches on classification problems and we have set the threshold value fixed at a value between two initiating points of data addition as similarly done in the works of BID11 BID14 .

We have set the fixed threshold based on 47th iteration (between 40 and 55).

For a more reliable selection score, we have not added any unlabeled data to the new training set and have trained our method with the labeled data only for 5 iterations.

As it can be seen in TAB4 , in the case of SST in the decay mode, the performance has been improved when the unlabeled dataset consists only in-class animal data, but when the unlabeled pool is filled with only out-of-class data, the performance is degraded.

For the case of SST with a fixed threshold value, samples are not added and the performance was not degraded at 100% nonanimal ratio as shown in FIG1 (c).

Furthermore, at 0% of out-of-class samples in the pool, there is a more improvement in the performance than at 100 % of out-of-class samples while still being inferior to the improvement than the decay mode.

Because less but stable data samples are added by SST with a fixed threshold, the performance is improved for all the cases compared to that of supervised learning.

Therefore, it is more suitable for real applications where the origin of data is usually unknown.

We proposed selective self-training (SST) for semi-supervised learning (SSL) problem.

Unlike conventional methods, SST selectively samples unlabeled data and trains the model with a subset of the dataset.

Using selection network, reliable samples can be added to the new training dataset.

In this paper, we conduct two types of experiments.

First, we experiment with the assumption that unlabeled data are in-class like conventional SSL problems.

Then, we experiment how SST performs for out-of-class unlabeled data.

For the conventional SSL problems, we achieved competitive results on several datasets and our method could be combined with conventional algorithms to improve performance.

The accuracy of SST is either saturated or not depending on the dataset.

Nonetheless, SST has shown performance improvements as a number of data increases.

In addition, the results of the combined experiments of SST and other algorithms show the possibility of performance improvement.

For the new SSL problems, SST did not show any performance degradation even if the model is learned from in-class data and out-of-class unlabeled data.

Decreasing the threshold of the selection network in new SSL problem, performance degrades.

However, the output of the selection network shows different trends according to in-class and out-of-class.

By setting a threshold that does not add out-of-class data, SST has prevented the addition of out-of-class samples to the new training dataset.

It means that it is possible to prevent the erroneous data from being added to the unlabeled dataset in a real environment.

6 SUPPLEMENTARY MATERIAL

The basic settings of our experiments are as follows.

Different from (Laine & Aila, 2016; Luo et al., 2017) , we use stochastic gradient descent (SGD) with a weight decay of 0.0005 as an optimizer.

The momentum weight for the ensemble of selection scores is set to ?? = 0.5.

Also, we do not apply mean-only batch normalization layer BID7 and Gaussian noise.

We follow the same data augmentation scheme in (Laine & Aila, 2016) consisting of horizontal flips and random translations.

However, ZCA whitening is not used.

In the supervised learning phase, we train our model using batch size 100 for 300 epochs.

After that, in the retraining phase, we train using the same batch size for 150 epochs with the new training dataset.

The learning rate starts from 0.1.

In the supervised learning phase, it is divided by 10 at the 150-th and 225-th epoch.

In the retraining phase, it is divided by 10 at the 75-th and 113-th epoch.

The number of training iteration and thresholding are very important parameters in our algorithm and have a considerable correlation with each other.

In the first experiment, the iteration number remains fixed and the growth rate of is adjusted so that the validation accuracy saturates near the settled iteration number.

While the validation accuracy is evaluated using the cross-validation, we set the number of training iteration to be 100 so that the model is trained enough until it saturates.

is increased in log-scale and begins at a very small value (10 ???5 ) where no data is added.

The growth rate of is determined according to when the validation accuracy saturates.

The stopping criterion is that the accuracy of the current iteration reaches the average accuracy of the previous 20 steps.

If the stopping iteration is much less than 100 times, the growth rate should be reduced so that the data is added more slowly.

If the stopping iteration significantly exceeds 100 iterations, the growth rate should be increased so that the data is added more easily.

We allow 5 iterations as a deviation from 100 iterations and the growth rate of is left unchanged in this interval.

As a result, the is gradually increased in log-scale by 10 times every 33 iterations in CIFAR-10 and SVHN.

In the case of CIFAR-100, the is increased by 10 times in log-scale every 27 iterations.

In the second experiment, we leave the fixed and simply train the model until the stopping criteria are satisfied.

Other details are the same as those of the first experiment.

We used two types of networks.

The network for training the synthetic dataset is shown in TAB7 and consists of two hidden layers with 30 nodes.

The network structure for CIFAR-10, SVHN, and CIFAR-100 consists of convolutions, and its structure is shown in TAB6 .

We used standard batch normalization (Ioffe & Szegedy, 2015) and Leaky ReLU (Maas et al., 2013) with 0.1.

We have experimented with CIFAR-10, SVHN, and CIFAR-100 datasets that consist of 32 ?? 32 pixel RGB images.

CIFAR-10 and SVHN have 10 classes and CIFAR-100 has 100 classes.

Overall, standard data normalization and augmentation scheme are used.

For data augmentation, we used random horizontal flipping and random translation by up to 2 pixels.

In the case of SVHN, random horizontal flipping is not used.

To show that the SST algorithm is comparable to the conventional SSL algorithms, we experimented with the popular setting (Laine & Aila, 2016; Miyato et al., 2017; BID9 .

The validation set in the cross-validation to obtain the reduction rate of epsilon is extracted from the training set by 5000 images.

After the epsilon is obtained, all the training datasets are used.

The following is the standard labeled/unlabeled split.

CIFAR-10 : 4k labeled data ( 400 images per class ), 46k unlabeled data ( 4,600 images per class ), As synthetic datasets, two moons and 4 spins were tested in the same manner as SNTG (Luo et al., 2017) .

Each dataset has 6,000 training and 6,000 test samples.

In the case of two moons, there are two classes y ??? {0, 1}, and in case of 4 spins, y ??? {0, 1, 2, 3}. In 6,000 training data, there are 12 labeled data and 5,988 unlabeled data.

Thus, for two moons, each class has 6 points and for 4 spins, each class has 3 points.

Because the number of labeled datapoints are too small, random sampling can lead to sample similar points.

Therefore, we randomly sampled the labeled data with a constraint that the Euclidian distance of each data point is greater than 0.7.

For these datasets, total iteration was performed 50 times, and the was increased from 10 ???7 to 10 ???4.5 on a log scale.

FIG3 shows the basic setting of the synthetic dataset, and Figure 5 and 6 show the progress of the SST algorithm.

The SST algorithm improves performance by gradually expanding certain data in a synthetic dataset.

CIFAR-10 : When the network were trained with 1k and 2k images, the test error were 38.71% and 26.99% respectively.

The test errors in the SST algorithm were 23.15% and 15.72%, the SST has better performance than ?? model but worse than Mean Teacher in 1k test.

In 2k test, the SST has better performance than ?? model and similar with Mean Teacher.

Table 6 : Classification error on CIFAR-10 (1k and 2k Labels) with 5 runs using in-class unlabeled data Method CIFAR-10 (1k) CIFAR-10 (2k) supervised (sampled)38.71 ?? 0.47% 26.99 ?? 0.79% ?? model (Laine & Aila, 2016) 27.36 ?? 1.20% 18.02 ?? 0.60% Mean Teacher BID9 21.55 ?? 1.48% 15, 73 ?? 0.31%Proposed method (SST) 23.15 ?? 0.61% 15.72 ?? 0.50%

For the balancing experiments, in SVHN, 1,000 images are used as the labeled data and 45,000 balanced unlabeled images are used.

As a result, the SST is still worse than other algorithms.

As mentioned in Section 4.1, we think that incorrectly estimated samples by SST deteriorate the performance.

BID3 adds only four unlabeled classes and tests according to the radio of unlabeled class.

For example, at 50%, two classes are in-class, and two classes are out-of-class.

However, we experimented with the ratio of the number of non-animal data.

Thus at 50% in CIFAR-10, unlabeled data consists of 50% in-class and 50% out-of-class.

The data for each ratio are shown in TAB8 , and the data category for animal and non-animal is shown in TAB9 .

TAB10 shows the results of a general test on other algorithms.

First, self-training (McLachlan, 1975; BID16 BID17 ) without threshold does not improve performance even at 0%, and performance at 100% is degraded.

When SST is applied to the softmax output as a threshold without selection network, the performance is improved at 0%, but the performance is degraded at 100%.

Although the threshold was 0.9999, unlabeled data was added in 100% of the non-animal data.

In the new SSL problem, the experiment in decay mode is to find a gap between two initiating points of data addition in 0 ??? 75% of non-animal data and 100% of non-animal data.

In our experiment, the growth rate of epsilon in CIFAR-10 is applied to CIFAR-100.

(The smaller the growth rate of epsilon, the less the difference in between iterations.

Therefore, although the difference in the between intervals is the same, depending on the growth rate of , the difference in the iteration can be greater.)

In the case of 0 ??? 75%, the number of data shows a slight increase from about 30 iterations.

On the other hand, in the case of 100%, selected samples are added from about 40 iterations.

The fixed threshold set to the threshold of 35 iterations.

In the decay mode, the performance is much improved at 0%, and at 100%, the performance is degraded.

On the other hand, in the fixed mode, there was no performance degradation from 0% to 100%.

In CIFAR-100, the difference between 0% and 100 % is less than CIFAR-10, because the gap between animal and non-animal is small and additional data is small.

FIG5 shows the experimental results.

<|TLDR|>

@highlight

Our proposed algorithm does not use all of the unlabeled data for the training, and it rather uses them selectively.
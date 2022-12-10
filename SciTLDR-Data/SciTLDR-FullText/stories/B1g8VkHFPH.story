Fine-tuning from pre-trained ImageNet models has become the de-facto standard for various computer vision tasks.

Current practices for fine-tuning typically involve selecting an ad-hoc choice of hyper-parameters and keeping them fixed to values normally used for training from scratch.

This paper re-examines several common practices of setting hyper-parameters for fine-tuning.

Our findings are based on extensive empirical evaluation for fine-tuning on various transfer learning benchmarks.

(1) While prior works have thoroughly investigated learning rate and batch size, momentum for fine-tuning is a relatively unexplored parameter.

We find that picking the right value for momentum is critical for fine-tuning performance and connect it with previous theoretical findings.

(2) Optimal hyper-parameters for fine-tuning in particular the effective learning rate are not only dataset dependent but also sensitive to the similarity between the source domain and target domain.

This is in contrast to hyper-parameters for training from scratch.

(3) Reference-based regularization that keeps models close to the initial model does not necessarily apply for "dissimilar" datasets.

Our findings challenge common practices of fine- tuning and encourages deep learning practitioners to rethink the hyper-parameters for fine-tuning.

Many real-world applications often have limited number of training instances, which makes directly training deep neural networks hard and prone to overfitting.

Transfer learning with the knowledge of models learned on a similar task can help to avoid overfitting.

Fine-tuning is a simple and effective approach of transfer learning and has become popular for solving new tasks in which pre-trained models are fine-tuned with the target dataset.

Specifically, fine-tuning on pre-trained ImageNet classification models (Simonyan & Zisserman, 2015; He et al., 2016b) has achieved impressive results for tasks such as object detection (Ren et al., 2015) and segmentation (He et al., 2017; Chen et al., 2017) and is becoming the de-facto standard of solving computer vision problems.

It is believed that the weights learned on the source dataset with a large number of instances provide better initialization for the target task than random initialization.

Even when there is enough training data, fine-tuning is still preferred as it often reduces training time significantly (He et al., 2019) .

The common practice of fine-tuning is to adopt the default hyperparameters for training large models while using smaller initial learning rate and shorter learning rate schedule.

It is believed that adhering to the original hyperparameters for fine-tuning with small learning rate prevents destroying the originally learned knowledge or features.

For instance, many studies conduct fine-tuning of ResNets (He et al., 2016b) with these default hyperparameters: learning rate 0.01, momentum 0.9 and weight decay 0.0001.

However, the default setting is not necessarily optimal for fine-tuning on other tasks.

While few studies have performed extensive hyperparameter search for learning rate and weight decay (Mahajan et al., 2018; Kornblith et al., 2018) , the momentum coefficient is rarely changed.

Though the effectiveness of the hyperparameters has been studied extensively for training a model from scratch, how to set the hyperparameters for fine-tuning is not yet fully understood.

In addition to using ad-hoc hyperparameters, commonly held beliefs for fine-tuning also include:

• Fine-tuning pre-trained networks outperforms training from scratch; recent work (He et al., 2019) has already revisited this.

• Fine-tuning from similar domains and tasks works better (Ge & Yu, 2017; Cui et al., 2018; Achille et al., 2019; Ngiam et al., 2018) .

• Explicit regularization with initial models matters for transfer learning performance (Li et al., 2018; 2019) .

Are these practices or beliefs always valid?

From an optimization perspective, the difference between fine-tuning and training from scratch is all about the initialization.

However, the loss landscape of the pre-trained model and the fine-tuned solution could be much different, so as their optimization strategies and hyperparameters.

Would the hyperparameters for training from scratch still be useful for fine-tuning?

In addition, most of the hyperparameters (e.g., batch size, momentum, weight decay) are frozen; will the conclusion differ when some of them are changed?

With these questions in mind, we re-examined the common practices for fine-tuning.

We conducted extensive hyperparameter search for fine-tuning on various transfer learning benchmarks with different source models.

The goal of our work is not to obtain state-of-the-art performance on each fine-tuning task, but to understand the effectiveness of each hyperparameter for fine-tuning, avoiding unnecessary computations.

We explain why certain hyperparameters work so well on certain datasets while fail on others, which can guide future hyperparameter search for fine-tuning.

Our main findings are as follows:

• Optimal hyperparameters for fine-tuning are not only dataset dependent, but also depend on the similarity between the source and target domains, which is different from training from scratch.

Therefore, the common practice of using optimization schedules derived from ImageNet training cannot guarantee good performance.

It explains why some tasks are not achieving satisfactory results after fine-tuning because of inappropriate hyperparameter selection.

Specifically, as opposed to the common practice of rarely tuning the momentum value beyond 0.9, we verified that zero momentum could work better for fine-tuning on tasks that are similar with the source domain, while nonzero momentum works better for target domains that are different from the source domain.

• Hyperparameters are coupled together and it is the effective learning rate-which encapsulates the learning rate, momentum and batch size-that matters for fine-tuning performance.

While effective learning rate has been studied for training from scratch, to the best of our knowledge, no previous work investigates effective learning rate for fine-tuning and is less used in practice.

Our observation of momentum can be explained as small momentum actually decreases the effective learning rate, which is more suitable for fine-tuning on similar tasks.

We show that the optimal effective learning rate actually depends on the similarity between the source and target domains.

• We find regularization methods that were designed to keep models close to the initial model does not apply for "dissimilar" datasets, especially for nets with Batch Normalization.

Simple weight decay can result in as good performance as the reference based regularization methods for fine-tuning with better search space.

In transfer learning for image classification, the last layer of a pre-trained network is usually replaced with a randomly initialized fully connected layer with the same size as the number of classes in the target task (Simonyan & Zisserman, 2015) .

It has been shown that fine-tuning the whole network usually results in better performance than using the network as a static feature extractor (Yosinski et al., 2014; Huh et al., 2016; Mormont et al., 2018; Kornblith et al., 2018) .

Ge & Yu (2017) select images that have similar local features from source domain to jointly fine-tune pre-trained networks.

Cui et al. (2018) estimate domain similarity with ImageNet and demonstrate that transfer learning benefits from pre-training on a similar source domain.

Besides image classification, many object detection frameworks also rely on fine-tuning to improve over training from scratch (Girshick et al., 2014; Ren et al., 2015) .

Many researchers re-examined whether fine-tuning is a necessity for obtaining good performance.

Ngiam et al. (2018) find that when domains are mismatched, the effectiveness of transfer learning is negative, even when domains are intuitively similar.

Kornblith et al. (2018) examine the fine-tuning performance of various ImageNet models and find a strong correlation between ImageNet top-1 accuracy and the transfer accuracy.

They also find that pre-training on ImageNet provides minimal benefits for some fine-grained object classification dataset.

He et al. (2019) questioned whether ImageNet pre-training is necessary for training object detectors.

They find the solution of training from scratch is no worse than the fine-tuning counterpart as long as the target dataset is large enough.

Raghu et al. (2019) find that transfer learning has negligible performance boost on medical imaging applications, but speed up the convergence significantly.

There is much literature on the hyperparameter selection for training neural networks from scratch, mostly on batch size, learning rate and weight decay (Goyal et al., 2017; Smith et al., 2018; Smith & Topin, 2019) .

There are few works on the selection of momentum (Sutskever et al., 2013) .

Zhang & Mitliagkas (2017) proposed an automatic tuner for momentum and learning rate in SGD and empirically show that it converges faster than Adam (Kingma & Ba, 2014) .

There are also studies on the correlations of the hyperparameters, such as linear scaling rule between batch size and learning (Goyal et al., 2017; Smith et al., 2018; Smith, 2017) .

However, most of these advances on hyperparameter tuning are designed for training from scratch, but not examined on fine-tuning tasks for computer vision problems.

Most work on fine-tuning just choose fixed hyperparameters for all fine-tuning experiments (Cui et al., 2018) or use dataset dependent learning rates in their experiments (Li et al., 2018) .

Due to the huge computational cost for hyperparameter search, only a few works (Kornblith et al., 2018; Mahajan et al., 2018 ) performed large-scale grid search of learning rate and weight decay for obtaining the best performance.

In this section, we first introduce the notations and experimental settings, and then present our observations on momentum, effective learning rate and regularization.

The fine-tuning process is not different from learning from scratch except for the weights initialization.

The goal of the process is still to minimize the loss function L =

, where is the loss function, N is the number of samples, x i is the input data, y i is its label, f is the neural network and θ is the model parameters.

Momentum is widely used for accelerating and smoothing the convergence of SGD by accumulating a velocity vector in the direction of persistent loss reduction (Sutskever et al., 2013; Goh, 2017) .

The commonly used Nesterov momentum SGD (Nesterov, 1983 ) iteratively updates the model in the following form:

where θ t indicates the model parameter at iteration t. The hyperparameters include the learning rate η t , batch size n, momentum coefficient m ∈ [0, 1), and the weight decay λ.

We evaluate fine-tuning on seven widely used image classification datasets, which covers tasks for fine-grained object recognition, scene recognition and general object recognition.

Detailed statistics of each dataset can be seen in Table 1 .

We use ImageNet (Russakovsky et al., 2015) , Place365 (Zhou et al., 2018) and iNaturalist (Van Horn et al., 2018) as source domains for pre-trained models.

We resize the input images such that the aspect ratio is preserved and the shorter side is 256 pixels.

The images are normalized with mean and std values calculated over ImageNet.

For data augmentation, we adopt the common practices used for training ImageNet models (Szegedy et al., 2015) : random mirror, random scaled cropping with scale and aspect variations, and color jittering.

The augmented images are resized to 224×224.

Note that state-of-the-art results could achieve even better performance by using higher resolution images (Cui et al., 2018) or better data augmentation (Cubuk et al., 2018) .

We mainly use ResNet-101-V2 (He et al., 2016a) as our base network, which is pre-trained on ImageNet (Russakovsky et al., 2015) .

Similar observations are also observed on DenseNets (Huang et al., 2017) and MobileNet (Howard et al., 2017 ) (see Appendix B).

The hyperparameters to be tuned (and ranges) are: learning rate (0.1, 0.05, 0.01, 0.005, 0.001, 0.0001), momentum (0.9, 0.99, 0.95, 0.9, 0.8, 0.0) and weight decay (0.0, 0.0001, 0.0005, 0.001).

We set the default hyperparameter to be batch size 256 1 , learning rate 0.01, momentum 0.9 and weight decay 0.0001.

To avoid insufficient training and observe the complete convergence behavior, we use 300 epochs for fine-tuning and 600 epochs for scratch-training , which is long enough for the training curves to converge.

The learning rate is decayed by a factor of 0.1 at epoch 150 and 250.

We report the Top-1 validation error at the end of fine-tuning.

The total computation time for the experiments is more than 10K GPU hours.

Momentum 0.9 is the most widely adopted value for training from scratch (Krizhevsky et al., 2012; Simonyan & Zisserman, 2015; He et al., 2016b) , and is also widely adopted in fine-tuning (Kornblith et al., 2018) .

To the best of our knowledge, it is rarely changed, regardless of the network architectures or target tasks.

To check the influence of momentum on fine-tuning, we first search the best momentum values for fine-tuning on the Birds dataset with different weight decay and learning rate.

Figure 1(a) shows the performance of fine-tuning with and without weight decays.

Surprisingly, momentum zero actually outperforms the nonzero momentum.

We also noticed that the optimal learning rate increases when the momentum is disabled (Figure 1(b) and Appendix A).

To verify this observation, we further compare momentum 0.9 and 0.0 on other datasets.

Table 2 shows the performance of 8 hyperparameter settings on seven datasets.

We find a clear pattern that disabling momentum works better for Dogs, Caltech, Indoor datasets, while momentum 0.9 works better for Cars, Aircrafts and Flowers.

Interestingly, datasets such as Dogs, Caltech, Indoor and Birds are known to have high overlap with ImageNet dataset 2 , while Cars/Aircrafts are identified to be difficult to benefit from fine-tuning from pre-trained ImageNet models (Kornblith et al., 2018) .

According to Cui et al. (2018) , in which the Earth Mover's Distance (EMD) is used to calculate the distance between a dataset with ImageNet, the similarity to Birds and Dogs are 0.562 and 0.620, while the similarity to Cars, Aircrafts and Flowers are 0.560 and 0.555, 0.525 3 .

The relative order of similarity to ImageNet is Dogs, Birds, Cars, Aircrafts and Flowers which aligns well with the transition of optimal momentum value from 0.0 to 0.9.

To verify this dependency on domain similarity, we fine-tune from pre-trained models of different source domains.

It is reported that Place365 and iNaturalist are better source domains than ImageNet for fine-tuning on Indoor and Birds dataset (Cui et al., 2018) .

We can expect that fine-tuning from iNaturalist works well for Birds with m = 0 and similarly, Places365 for Indoor.

Indeed, as shown in Table 3 , disabling momentum improves the performance when the source and target domains are similar, such as Places for Indoor and iNaturalist for Birds.

Large momentum works better for fine-tuning on different domains but not for tasks that are close to source domains Our explanation for the above observations is that because the Dogs dataset is very close to ImageNet, the pre-trained ImageNet model is expected to be close to the fine-tuned solution on the Dogs dataset.

In this case, momentum may not help much as the gradient direction around the minimum could be much random and accumulating the momentum direction could be meaningless.

Whereas, for faraway target domains (e.g., Cars and Aircrafts) where the pre-trained ImageNet model could be much different with the fine-tuned solution, the fine-tuning process is more similar with training from scratch, where large momentum stabilizes the decent directions towards the minimum.

An illustration of the difference can be found in Figure 2 .

Connections to early observations on decreasing momentum Early work (Sutskever et al., 2013) actually pointed out that reducing momentum during the final stage of training allows finer convergence while aggressive momentum would prevent this.

They recommended reducing momentum from 0.99 to 0.9 in the last 1000 parameter updates but not disabling it completely.

Recent work (Liu et al., 2018; Smith, 2018) showed that a large momentum helps escape saddle points but can hurt the final convergence within the neighborhood of the optima, implying that momentum should be reduced at the end of training.

Liu et al. (2018) find that a larger momentum introduces higher variance of noise and encourages more exploration at the beginning of optimization, and encourages more aggressive exploitation at the end of training.

They suggest that at the final stage of the step size annealing, momentum SGD should use a much smaller step size than that of vanilla SGD.

When applied to fine-tuning, we can interpret that if the pre-trained model lies in the neighborhood of the optimal solution on the target dataset, the momentum should be small.

Our work identifies the empirical evidence of disabling momentum helps final convergence, and fine-tuning on close domains seems to be a perfect case. (Hertz et al., 1991; Smith & Le, 2018) for SGD with momentum is follows:

which was shown to be more closely related with training dynamics and final performance rather than η (Smith et al., 2018; Smith & Le, 2018) .

The effective learning rate with m = 0.9 is 10× higher than the one with m = 0.0 if other hyperparameters are fixed, which is probably why we see an increase in optimal learning rate when momentum is disabled in Figure 1 (b) and Appendix A.

Because learning rate and momentum are coupled, looking at the performance with only one hyperparameter varied can give a misleading understanding of the effect of hyperparameters.

Therefore, we report the best result with and without momentum.

which does not affect the maximum accuracy obtainable with and without momentum, as long as the hyperparameters explored are sufficiently close to the optimal parameters.

We review previous experiments that demonstrated the importance of momentum tuning when the effective learning rate η = η/(1−m) is held fixed instead of the learning rate η.

Figure 3 shows that when η is constant, momentum 0.0 and 0.9 are actually equivalent.

In addition, the best performance obtained by momentum 0.9 and momentum 0 is equivalent when other hyperparameters are allowed to change.

However, different effective learning rates results in different performance, which indicates that it is effective learning rate that matters for the best performance.

It explains why the common practice of changing only learning rate generally works, though changing momentum may results in the same effect.

They both change the effective learning rate.

Optimal effective learning rate and weight decay depend on the similarity between source domain and target domain.

Now that we have shown ELR is critical for the performance of fine-tuning, we are interested in the factors that determine the optimal ELR affected.

Smith & Le (2018) found that there is an optimum fluctuation scale which maximizes the test set accuracy (at constant learning rate).

However, the relationship between ELR and domain distance is unknown, which is important for fine-tuning.

Effective learning rate encapsulates the effect of learning rate and momentum for fine-tuning.

We varied other hyperparameters and report the best performance for each η .

As shown in Figure 4 , a smaller η works better if source and target domains are similar, such as Dogs for ImageNet and Birds for iNaturalist.

On the other hand, the ELR for training from scratch is large and relative stable.

We made similar observations on DenseNets (Huang et al., 2017) and MobileNet (Howard et al., 2017 ) (see Appendix B).

The relationship between weight decay and effective learning rate are also well-studied (Loshchilov & Hutter, 2018; van Laarhoven; Zhang et al., 2018) .

It was shown that the optimal weight decay value λ is inversely related with learning rate η.

The 'effective' weight decay is λ = λ/η.

We show in Figure 5 that the optimal effective weight decay is larger when the source domain is similar with the target domain.

Domain Similarity and Hyperparameter Selection Now we have made qualitative observations about the relationship between domain similarity and optimal ELR, which can help for reducing the hyperparameter search ranges.

Note the original similarity score is based on models pre-trained on large scale dataset Sun et al. (2017) , which are not public available.

We revisited the domain similarity score calculation in Appendix C and propose to use the source model as feature extractor We find there is a good correlation between our own domain similarity and the scale of optimal ELR.

Based on the correlation between the similarity score and the optimal ELR, we propose a simple strategy for ELR selection in Appendix C.

L 2 regularization or weight decay is widely used for constraining the model capacity (Hanson & Pratt, 1989; Krogh & Hertz, 1992) .

Recent work (Li et al., 2018; 2019) pointed that standard L 2 regularization, which drives the parameters towards the origin, is not adequate in transfer learning.

To retain the knowledge learned by the pre-trained model, reference based regularization was used to regularize the distance between fine-tuned weights and the pre-trained weights, so that the finetuned model is not too different from the initial model.

Li et al. (2018) propose L 2 -SP norm, i.e.,

, where θ refers to the part of network that shared with the source network, and θ refers to the novel part, e.g., the last layer with different number of neurons.

While the motivation is intuitive, there are several issues for adopting reference based regularization for fine-tuning: (1) Many applications actually adopt fine-tuning on target domains that are quite different from source domain, such as fine-tuning ImageNet models for medical imaging (Mormont et al., 2018; Raghu et al., 2019) .

The fine-tuned model does not necessarily have to be close with the initial model.

(2) The scale invariance introduced by Batch Normalization (BN) (Ioffe & Szegedy, 2015) layers enable models with different parameter scales to function the same, i.e., f (θ) = f (αθ).

Therefore, when L 2 regularization drives θ 2 2 towards zeros, it could still have the same functionality as the initial model.

On the contrary, a model could still be different even when the L 2 -SP norm is small.

(3) It has been shown that the effect of weight decay on models with BN layers is equivalent to increasing the effective learning rate by shrinking the weights scales (van Laarhoven; Zhang et al., 2018) .

Regularizing weights with L 2 -SP norm would constrain the scale of weights to be close to the original one, therefore not increasing the effective learning rate, during fine-tuning.

As a small effective learning rate is beneficial for fine-tuning from similar domains, which may explain why L 2 -SP provides better performance.

If this is true, then by decreasing the effective learning rate, L 2 regularization would functions the same.

To examine these conjectures, we revisited the work of (Li et al., 2018) with additional experiments.

To show the effectiveness of L 2 -SP norm, Li et al. (2018) conducted experiments on datasets such as Dogs, Caltech and Indoor, which are all datasets close to the source domain (ImageNet or Place-365) according to previous sections.

We extend their experiments on other datasets that are relatively "far" away from ImageNet, such as Birds, Cars, Aircrafts and Flowers.

We use the source code of Li et al. (2018) to fine-tune on these datasets with both L 2 and L 2 -SP regularization.

For fair comparison, we performed the same hyperparameter search for both methods (see experimental settings in Appendix E).

As expected, Table 4 shows that L 2 regularization is very close to if not better than L 2 -SP on Birds, Cars, Aircrafts and Flowers, which indicates that reference based regularization methods may not be able to generalize for fine-tuning on dissimilar domains.

The two extreme ways for selecting hyperparameters-performing exhaustive hyperparameter search or taking ad-hoc hyperparameters from scratch training-could be either too computationally expensive or yield inferior performance.

Different with training from scratch, the default hyperparameter setting may work well for random initialization, the choice of hyperparameters for fine-tuning is not only dataset dependent but is also influenced by the similarity between the target domain and the source domains.

The rarely tuned momentum value could impede the performance when the target domain and source domain are close.

These observations connect with previous theoretical works on decreasing momentum at the end of training and effective learning rate.

We further identify the optimal effective learning rate depends on the similarity of source domain and target domain.

With this understanding, one can significant reduce the hyperparameter search space.

We hope these findings could be one step towards better hyperparameter selection strategies for fine-tuning.

To check the influence of momentum on fine-tuning, we first search the best momentum values for fine-tuning on the Birds dataset with fixed learning rate but different batch size and weight decay.

Figure 6 provides the convergence curves for the results shown in Figure 1 (a), which shows the learning curves of fine-tuning with 6 different batch sizes and weight decay combinations.

Zero momentum outperforms the nonzero momentum in 5 of the 6 configurations.

Optimal learning rate increases after disabling momentum.

Figure 7 compares the performance of turning on/off momentum for each datasets with different learning.

For datasets that are "similar" to ImageNet (Figure 7 (a-h)) and fixed learning rate (e.g., 0.01), the Top-1 validation error decreases significantly after disabling momentum.

On the other hand, for datasets that are "dissimilar" to ImageNet (Figure 7(g-n) ) and fixed learning rate, disabling momentum hurts the top-1 accuracy.

We can also observe that the optimal learning rate generally increase 10x after changing from 0.9 to 0.0, which is coherrent with the rule of effective learning rate.

We also verified our observations on DenseNet-121 (Huang et al., 2017) and MobileNet-1.0 (Howard et al., 2017) with similar settings.

We made similar observations among the three architectures: the optimal effective learning rate is related with the similarity to source domain.

As seen in Figure 8 (b) and (c), the optimal effective learning rates for Dogs/Caltech/Indoor datasets are much smaller than these for Aircrafts/Flowers/Cars when fine-tuned from ImageNet, similar with ResNet-101.

This verified our findings on ResNet-101 are pretty consistent on a variety of architectures.

In Appendix C, we will show the the correlation between EMD based domain similarity and optimal ELR is quite consistent across different architectures.

The domain similarity calculation based on Earth Mover Distance (EMD) is introduced in the section 4.1 of (Cui et al., 2018) 4 .

Here we briefly introduce the steps.

In (Cui et al., 2018) , the authors first train ResNet-101 on the large scale JFT dataset (Sun et al., 2017) and use it as a feature extractor.

They extracted features from the penultimate layer of the model for each image of the training set of the source domain and target domain.

For ResNet-101, the length of the feature vector is 2048.

The features of images belonging to the same category are averaged and g(s i ) denotes the average feature vector of ith label in source domain S, similarly, g(t j ) denotes the average feature vector of jth label in target domain T .

The distance between the averaged features of two labels is d i,j = g(s i ) − g(t j ) .

Each label is associated with a weight w ∈ [0, 1] corresponding to the percentage of images with this label in the dataset.

So the source domain S with m labels and the target domain T with n labels can be represented as

The EMD between the two domains is defined as

where the optimal flow f i,j corresponds to the least amount of total work by solving the EMD optimization problem.

The domain similarity is defined as

where γ is 0.01.

The domain similarities based on this pre-trained model for the seven dataset is listed in Table 5 .

Cui et al. (2018) , which use JFT pretrained ResNet-101 as the feature extractor.

Note that the ResNet-101 pre-trained on JFT is not released and therefore we cannot calculate its similarities for datasets such as Caltech and Indoor.

On the right three columns, we use three public available ImageNet pre-trained models as the feature extractor.

All features are extracted from the training set of each dataset.

The 1st, 2nd, 3rd and 4th scores are color coded, and the smallest three scores are marked in brown.

The corresponding optimal ELR η is listed, which is the same as shown in Figure

Due to the unavailability of the large-scale JFT dataset (300x larger than ImageNet) and its pre-trained ResNet-101 model, we cannot use it for extracting features for new datasets, such as Caltech256 and MIT67-Indoor.

Instead of using the powerful feature representation, we use the pre-trained source model directly as the feature extractor for both source domain and target domains.

We believe the similarity based on features extracted from source models captures the transfer learning process better for different source models.

In Table 5 , we compared the domain similarities calculated by three different ImageNet pre-trained models.

We find some consistent patterns across different architectures:

• The 1st and 2nd highest similarity scores are Caltech and Dogs across architectures.

• The 3rd and 4th highest similarity scores refers to Birds and Indoor.

• The most dissimilar datasets are Cars, Aircrafts and Flowers.

Note the relative orders for the dissimilar datasets (Cars, Aircrafts and Flowers) are not consistent across difference architectures.

As we can see from Table 5 , the domain similarity score has some correlation with the scale of optimal ELR.

Generally, the more similarity between two domains, the smaller the optimal ELR.

Though the optimal ELR is not strictly corresponding to the domain similarity score, the scores provide reasonable predictions about the scale of optimal ELR, such as and therefore can reduce the search ranges of ELR.

One can calculate the domain similarities and perform exhaustive hyperparameter searches for the first few target datasets, including similar and dissimilar datasets, such as Dogs and Cars and we refer these datasets as reference datasets.

Then for the given new dataset to fine-tune, one can calculate its domain similarity and compare with the scores of reference datasets, and choose the range of ELRs with the closest domain similarity.

D BN MOMENTUM Kornblith et al. (2018) referred to the fact that the momentum parameter of BN is essential for fine-tuning.

They found it useful to decrease the batch normalization momentum parameter from its ImageNet value to max(110/s, 0.9) where s is the number of steps per epoch.

This will change the default BN momentum value (0.9) when s is larger than 100, but only applies when the dataset size is larger than 25.6K with batch size 256.

The maximum data size used in our experiments is Caltech-256, which is 15K, so this strategy is not applicable to our experiments.

We further explore the effect of BN momentum by perform similar study as to ELR.

We want to identify whether there is an optimal BN momentum for each dataset.

For each dataset, we finetune the pre-trained model using previously obtained best hyperparameters and only tune the BN momentum.

In addition to the default 0.9, we searched 0.0, 0.95 and 0.99.

If BN mommentum is critical, we can expect significant performance differences.

The result is shown in Figure 9 We see m bn = 0.99 slightly improves the performance for some datasets, however, we did not see the significant performance difference among values greater than 0.9.

We use the code 5 provided by the authors.

The base network is ImageNet pretrained ResNet-101-V1.

The model is fine-tuned with batch size 64 in 9000 iterations, and the learning rate is decayed at iteration 6000.

Following the original setting, we use momentum 0.9.

We performed grid search on learning rate and weight decay, with the range of η : {0.02, 0.01, 0.005, 0.001, 0.0001} and λ 1 : {0.1, 0.01, 0.001, 0.0001}, and report the best average error for both methods.

For L 2 -SP norm, we follow the authors' setting to use constant λ 2 = 0.01.

Different with the original setting for L 2 regularization, we set λ 2 = λ 1 to simulate the normal L 2 -norm.

Data augmentation is an important way of increasing data quantity and diversity to make models more robust.

It is even critical for transfer learning with few instances.

The effect of data augmentation can be viewed as a regularization method and the choice of data augmentation method is also a hyperparameter.

Most current widely used data augmentation methods have verified their effectiveness on training ImageNet models, such as random mirror flipping, random rescaled cropping 6 , color jittering and etc Szegedy et al. (2015) ; Xie et al. (2018) and they are also widely used for fine-tuning.

Do these methods transfer for fine-tuning on other datasets?

Here we compare three settings for data augmentation: 1) random resized cropping: our default data augmentation; 2) random crop: the same as standard data augmentation except that we use random cropping with fixed size; 3) random flip: simply random horizontal flipping.

The effect of data augmentation is dataset dependent and has big impact on the convergence time The training and validation errors of fine-tuning with different data augmentation strategies are illustrated in Figure 10 .

We find that advanced cropping works significantly better on datasets like Cars, Aircrafts and Flowers but performs worse on Dogs.

The choice of data augmentation methods has dramatic influence to the convergence behaviour.

Simpler data augmentation usually converge very quickly (e.g., in 20 epochs), while the training error for random resized cropping converges much slower.

We see that default hyperparemter and data augmentation method lead to overfitting on Dogs dataset.

This can be solved by disabling momentum as we can see in Table 2 , and result in better performance than random cropping.

We can expect that random resized cropping adds extra variance to the gradient direction and the effect of disabling momentum is more obvious for this case.

Disabling momentum improves performance on datasets that are close to source domains Here we compare data augmentation methods with different momentum settings.

As can be seen in Table 6 , random resized cropping consistently outperforms random cropping on datasets like Cars, Aircrafts and Flowers.

Using momentum improves the performance significantly for both methods.

We see that advanced data augmentation method with default hyperparameters (m = 0.9 and η = 0.01) leads to overfitting on Dogs and Caltech dataset (Figure 11 (a) and (c)).

Random resized cropping with zero momentum solves this problem and results in better performance than random cropping.

When momentum is disabled for random cropping, the performance is still better for Dogs, but decreases for other datasets.

This can be expected as random cropping produces images with less variation and noise than random resized cropping, the gradients variation is less random and momentum can still point to the right direction.

This can be further verified as we increase the learning rate for random cropping, which adds variation to the gradients, and disabling momentum shows better performance that nonzero momentum on datasets that are close.

Transfer learning from similar source domains helps but does not guarantee good performance We consider two ImageNet subsets: 449 Natural objects and 551 Man-made objects, following the splits of (Yosinski et al., 2014) (supplementary materials).

From the bottom of Table 7 , we can see that fine-tuning from ImageNet-Natural pre-trained models performs better on Birds and Dogs dataset, whereas Caltech-256 and Indoor benefit more from ImageNet-Manmade pretrained models.

The performance gap between ImageNet-Manmad and ImageNet-Natural on Cars and Flowers are not as significant as for Birds and Dogs.

It is surprising to see that fine-tuning from ImageNetManmade subset yields worse performance than ImageNet-Natural on the Cars and Indoor dataset.

The fine-tuning results on both subsets do not exceed the pre-trained models with full ImageNet.

Scratch training can outperform fine-tuning with better hyperparameters We further reexamine the default hyperparameters for scratch training.

For most tasks, training from scratch with default hyperparameters is much worse than fine-tuning from ImageNet.

However, after slight hyperparameter tuning on learning rates, momentum and weight decay, the performance of training from scratch gets close to the default fine-tuning result (e.g., Cars and Aircrafts).

Scratch training HPO on Cars and Aircrafts even surpasses the default fine-tuning result.

Previous studies Kornblith et al. (2018) ; Cui et al. (2018) also identified that datasets like Cars, Aircrafts do not benefit too much from fine-tuning.

FT ImageNet Default is the fine-tuning result with the default hyperparameters.

Scratch Train use similar hyperparameters as default fine-tuning, which are η = 0.1, n = 256, λ = 0.0001 and m = 0.9 with doubled length of training schedules.

HPO refers to the best results with hyperparameter grid search.

Note our Indoor dataset result is fine-tuned from ImageNet.

Results of ResNet-101 DELTA refers to (Li et al., 2019) , and Inception-v3 refers to (Cui et al., 2018 Pre-trained models on ImageNet-Natural and ImageNet-Manmade We train ResNet-101 from scratch on each subset using standard hyperparameters, i.e., initial learning rate 0.1, batch size 256, momentum 0.9.

We train 180 epochs, learning rate is decayed at epoch 60 and 120 by a factor of 10.

Table 8 illustrates the Top-1 errors of training ResNet-101 on each source datasets.

Scratch Training HPO Figure 12 shows the training/validation errors of training from scratch on each dataset with different learning rate and weight decay.

We use initial learning rate 0.1, batch size 256.

For most dataset, we train 600 epochs, and decay the learning rate at epoch 400 and 550 by a factor of 10.

The parameters to search is η ∈ [0.1, 0.2, 0.5] and λ ∈ [0.0001, 0.0005] with fixed momentum 0.9 and batch size 256.

We observe weight decay 0.0005 consistently performs better than 0.0001.

@highlight

This paper re-examines several common practices of setting hyper-parameters for fine-tuning.
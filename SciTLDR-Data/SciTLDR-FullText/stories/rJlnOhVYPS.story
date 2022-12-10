Person re-identification (re-ID) aims at identifying the same persons' images across different cameras.

However, domain diversities between different datasets pose an evident challenge for adapting the re-ID model trained on one dataset to another one.

State-of-the-art unsupervised domain adaptation methods for person re-ID transferred the learned knowledge from the source domain by optimizing with pseudo labels created by clustering algorithms on the target domain.

Although they achieved state-of-the-art performances, the inevitable label noise caused by the clustering procedure was ignored.

Such noisy pseudo labels substantially hinders the model's capability on further improving feature representations on the target domain.

In order to mitigate the effects of noisy pseudo labels, we propose to softly refine the pseudo labels in the target domain by proposing an unsupervised framework, Mutual Mean-Teaching (MMT), to learn better features from the target domain via off-line refined hard pseudo labels and on-line refined soft pseudo labels in an alternative training manner.

In addition, the common practice is to adopt both the classification loss and the triplet loss jointly for achieving optimal performances in person re-ID models.

However, conventional triplet loss cannot work with softly refined labels.

To solve this problem, a novel soft softmax-triplet loss is proposed to support learning with soft pseudo triplet labels for achieving the optimal domain adaptation performance.

The proposed MMT framework achieves considerable improvements of 14.4%, 18.2%, 13.1% and 16.4% mAP on Market-to-Duke, Duke-to-Market, Market-to-MSMT and Duke-to-MSMT unsupervised domain adaptation tasks.

:

Person image A 1 and A 2 belong to the same identity while B with similar appearance is from another person.

However, clustering-generated pseudo labels in state-of-the-art Unsupervised Domain Adaptation (UDA) methods contain much noise that hinders feature learning.

We propose pseudo label refinery with on-line refined soft pseudo labels to effectively mitigate the influence of noisy pseudo labels and improve UDA performance on person re-ID.

To effectively address the problem of noisy pseudo labels in clustering-based UDA methods (Song et al., 2018; Zhang et al., 2019b; (Figure 1 ), we propose an unsupervised Mutual Mean-Teaching (MMT) framework to effectively perform pseudo label refinery by optimizing the neural networks under the joint supervisions of off-line refined hard pseudo labels and on-line refined soft pseudo labels.

Specifically, our proposed MMT framework provides robust soft pseudo labels in an on-line peer-teaching manner, which is inspired by the teacher-student approaches (Tarvainen & Valpola, 2017; Zhang et al., 2018b) to simultaneously train two same networks.

The networks gradually capture target-domain data distributions and thus refine pseudo labels for better feature learning.

To avoid training error amplification, the temporally average model of each network is proposed to produce reliable soft labels for supervising the other network in a collaborative training strategy.

By training peer-networks with such on-line soft pseudo labels on the target domain, the learned feature representations can be iteratively improved to provide more accurate soft pseudo labels, which, in turn, further improves the discriminativeness of learned feature representations.

The classification and triplet losses are commonly adopted together to achieve state-of-the-art performances in both fully-supervised and unsupervised (Zhang et al., 2019b; person re-ID models.

However, the conventional triplet loss (Hermans et al., 2017) cannot work with such refined soft labels.

To enable using the triplet loss with soft pseudo labels in our MMT framework, we propose a novel soft softmax-triplet loss so that the network can benefit from softly refined triplet labels.

The introduction of such soft softmax-triplet loss is also the key to the superior performance of our proposed framework.

Note that the collaborative training strategy on the two networks is only adopted in the training process.

Only one network is kept in the inference stage without requiring any additional computational or memory cost.

The contributions of this paper could be summarized as three-fold.

(1) We propose to tackle the label noise problem in state-of-the-art clustering-based UDA methods for person re-ID, which is mostly ignored by existing methods but is shown to be crucial for achieving superior final performance.

The proposed Mutual Mean-Teaching (MMT) framework is designed to provide more reliable soft labels.

(2) Conventional triplet loss can only work with hard labels.

To enable training with soft triplet labels for mitigating the pseudo label noise, we propose the soft softmax-triplet loss to learn more discriminative person features.

(3) The MMT framework shows exceptionally strong performances on all UDA tasks of person re-ID.

Compared with state-of-the-art methods, it leads to significant improvements of 14.4%, 18.2%, 13.4%, 16.4% mAP on Market-to-Duke, Duke-to-Market, Market-to-MSMT, Duke-to-MSMT re-ID tasks.

Unsupervised domain adaptation (UDA) for person re-ID.

UDA methods have attracted much attention because their capability of saving the cost of manual annotations.

There are three main categories of methods.

The first category of clustering-based methods maintains state-of-the-art performance to date. (Fan et al., 2018) proposed to alternatively assign labels for unlabeled training samples and optimize the network with the generated targets.

proposed a bottom-up clustering framework with a repelled loss.

introduced to assign hard pseudo labels for both global and local features.

However, the training of the neural network was substantially hindered by the noise of the hard pseudo labels generated by clustering algorithms, which was mostly ignored by existing methods.

The second category of methods learns domain-invariant features from style-transferred source-domain images.

SPGAN (Deng et al., 2018) and PTGAN transformed source-domain images to match the image styles of the target domain while maintaining the original person identities.

The style-transferred images and their identity labels were then used to fine-tune the model.

HHL (Zhong et al., 2018) learned camera-invariant features with camera style transferred images.

However, the retrieval performances of these methods deeply relied on the image generation quality, and they did not explore the complex relations between different samples in the target domain.

The third category of methods attempts on optimizing the neural networks with soft labels for target-domain samples by computing the similarities with reference images or features.

ENC (Zhong et al., 2019) assigned soft labels by saving averaged features with an exemplar memory module.

MAR conducted multiple soft-label learning by comparing with a set of reference persons.

However, the reference images and features might not be representative enough to generate accurate labels for achieving advanced performances.

Generic domain adaptation methods for close-set recognition.

Generic domain adaptation methods learn features that can minimize the differences between data distributions of source and target domains.

Adversarial learning based methods (Zhang et al., 2018a; Tzeng et al., 2017; Ghifary et al., 2016; Bousmalis et al., 2016; Tzeng et al., 2015) adopted a domain classifier to dispel the discriminative domain information from the learned features in order to reduce the domain gap.

There also exist methods (Tzeng et al., 2014; Long et al., 2015; Yan et al., 2017; Saito et al., 2018; Ghifary et al., 2016) that minimize the Maximum Mean Discrepancy (MMD) loss between source-and target-domain distributions.

However, these methods assume that the classes on different domains are shared, which is not suitable for unsupervised domain adaptation on person re-ID.

Teacher-student models have been widely studied in semi-supervised learning methods and knowledge/model distillation methods.

The key idea of teacher-student models is to create consistent training supervisions for labeled/unlabeled data via different models' predictions.

Temporal ensembling (Laine & Aila, 2016) maintained an exponential moving average prediction for each sample as the supervisions of the unlabeled samples, while the mean-teacher model (Tarvainen & Valpola, 2017) averaged model weights at different training iterations to create the supervisions for unlabeled samples.

Deep mutual learning (Zhang et al., 2018b ) adopted a pool of student models instead of the teacher models by training them with supervisions from each other.

However, existing methods with teacher-student mechanisms are mostly designed for close-set recognition problems, where both labeled and unlabeled data share the same set of class labels and could not be directly utilized on unsupervised domain adaptation tasks of person re-ID.

Generic methods for handling noisy labels can be classified into four categories.

Loss correction methods (Patrini et al., 2017; Vahdat, 2017; Xiao et al., 2015) tried to model the noise transition matrix, however, such matrix is hard to estimate in real-world tasks, e.g. unsupervised person re-ID with noisy pseudo labels obtained via clustering algorithm. (Veit et al., 2017; Lee et al., 2018; Han et al., 2019) attempted to correct the noisy labels directly, while the clean set required by such methods limits their generalization on real-world applications.

Noise-robust methods designed robust loss functions against label noises, for instance, Mean Absolute Error (MAE) loss (Ghosh et al., 2017) , Generalized Cross Entropy (GCE) loss (Zhang & Sabuncu, 2018) and Label Smoothing Regularization (LSR) (Szegedy et al., 2016) .

However, these methods did not study how to handle the triplet loss with noisy labels, which is crucial for learning discriminative feature representations on person re-ID.

The last kind of methods which focused on refining the training strategies is mostly related to our method.

Co-teaching (Han et al., 2018) trained two collaborative networks and conducted noisy label detection by selecting on-line clean data for each other, Co-mining further extended this method on the face recognition task with a re-weighting function for Arc-Softmax loss (Deng et al., 2019) .

However, the above methods are not designed for the open-set person re-ID task and could not achieve state-of-the-art performances under the more challenge unsupervised settings.

We propose a novel Mutual Mean-Teaching (MMT) framework for tackling the problem of noisy pseudo labels in clustering-based Unsupervised Domain Adaptation (UDA) methods.

The label noise has important impacts to the domain adaptation performance but was mostly ignored by those methods.

Our key idea is to conduct pseudo label refinery in the target domain by optimizing the neural networks with off-line refined hard pseudo labels and on-line refined soft pseudo labels in a collaborative training manner.

In addition, the conventional triplet loss cannot properly work with soft labels.

A novel soft softmax-triplet loss is therefore introduced to better utilize the softly refined pseudo labels.

Both the soft classification loss and the soft softmax-triplet loss work jointly to achieve optimal domain adaptation performances.

3.1 CLUSTERING-BASED UDA METHODS REVISIT State-of-the-art UDA methods (Fan et al., 2018; Zhang et al., 2019b; follow a similar general pipeline.

They generally pre-train a deep neural network F (·|θ) on the source domain, where θ denotes current network parameters, and the network is then transferred to learn from the images in the target domain.

The source-domain images' and target-domain images' features encoded by the network are denoted as {F (x (2) The network parameters θ and a learnable target-domain classifier C t : f t → {1, · · · , M t } are then optimized with respect to an identity classification (crossentropy) loss L t id (θ) and a triplet loss (Hermans et al., 2017

where || · || denotes the L 2 -norm distance, subscripts i,p and i,n indicate the hardest positive and hardest negative feature index in each mini-batch for the sample x t i , and m = 0.5 denotes the triplet distance margin.

Such two operations, pseudo label generation by clustering and feature learning with pseudo labels, are alternated until the training converges.

However, the pseudo labels generated in step (1) inevitably contain errors due to the imperfection of features as well as the errors of the clustering algorithms, which hinder the feature learning in step (2).

To mitigate the pseudo label noise, we propose the Mutual Mean-Teaching (MMT) framework together with a novel soft softmax-triplet loss to conduct the pseudo label refinery.

3.2.1 SUPERVISED PRE-TRAINING FOR SOURCE DOMAIN UDA task on person re-ID aims at transferring the knowledge from a pre-trained model on the source domain to the target domain.

A deep neural network is first pre-trained on the source domain.

Given the training data D s , the network is trained to model a feature transformation function F (·|θ) that transforms each input sample x s i into a feature representation F (x s i |θ).

Given the encoded features, the identification classifier C s outputs an M s -dimensional probability vector to predict the identities in the source-domain training set.

The neural network is trained with a classification loss L s id (θ) and a triplet loss L s tri (θ) to separate features belonging to different identities.

The overall loss is therefore calculated as

where L Overall framework of the proposed Mutual Mean-Teaching (MMT) with two collaborative networks jointly optimized under the supervisions of off-line refined hard pseudo labels and on-line refined soft pseudo labels.

A soft identity classification loss and a novel soft softmax-triplet loss are adopted.

(c) One of the average models with better validated performance is adopted for inference as average models perform better than models with current parameters.

Our proposed MMT framework is based on the clustering-based UDA methods with off-line refined hard pseudo labels as introduced in Section 3.1, where the pseudo label generation and refinement are conducted alternatively.

However, the pseudo labels generated in this way are hard (i.e., they are always of 100% confidences) but noisy.

In order to mitigate the pseudo label noise, apart from the off-line refined hard pseudo labels, our framework further incorporates on-line refined soft pseudo labels (i.e., pseudo labels with < 100% confidences) into the training process.

Our MMT framework generates soft pseudo labels by collaboratively training two same networks with different initializations.

The overall framework is illustrated in Figure 2 (b).

The pseudo classes are still generated the same as those by existing clustering-based UDA methods, where each cluster represents one class.

In addition to the hard and noisy pseudo labels, our two collaborative networks also generate on-line soft pseudo labels by network predictions for training each other.

The intuition is that, after the networks are trained even with hard pseudo labels, they can roughly capture the training data distribution and their class predictions can therefore serve as soft class labels for training.

However, such soft labels are generally not perfect because of the training errors and noisy hard pseudo labels in the first place.

To avoid two networks collaboratively bias each other, the past temporally average model of each network instead of the current model is used to generate the soft pseudo labels for the other network.

Both off-line hard pseudo labels and on-line soft pseudo labels are utilized jointly to train the two collaborative networks.

After training, only one of the past average models with better validated performance is adopted for inference (see Figure 2 (c)).

We denote the two collaborative networks as feature transformation functions F (·|θ 1 ) and F (·|θ 2 ), and denote their corresponding pseudo label classifiers as C t 1 and C t 2 , respectively.

To simultaneously train the coupled networks, we feed the same image batch to the two networks but with separately random erasing, cropping and flipping.

Each target-domain image can be denoted by x t i and x t i for the two networks, and their pseudo label confidences can be predicted as C t 1 (F (x t i |θ 1 )) and C t 2 (F (x t i |θ 2 )).

One naïve way to train the collaborative networks is to directly utilize the above pseudo label confidence vectors as the soft pseudo labels for training the other network.

However, in such a way, the two networks' predictions might converge to equal each other and the two networks lose their output independences.

The classification errors as well as pseudo label errors might be amplified during training.

In order to avoid error amplification, we propose to use the temporally average model of each network to generate reliable soft pseudo labels for supervising the other network.

Specifically, the parameters of the temporally average models of the two networks at current iteration T are denoted as

respectively, which can be calculated as

where

indicate the temporal average parameters of the two networks in the previous iteration (T −1), the initial temporal average parameters are

and α is the ensembling momentum to be within the range [0, 1).

The robust soft pseudo label supervisions are then generated by the two temporal average models as

respectively.

The soft classification loss for optimizing θ 1 and θ 2 with the soft pseudo labels generated from the other network can therefore be formulated as

The two networks' pseudo-label predictions are better dis-related by using other network's past average model to generate supervisions and can therefore better avoid error amplification.

Generalizing classification cross-entropy loss to work with soft pseudo labels has been well studied (Hinton et al., 2015) , (Müller et al., 2019) .

However, optimizing triplet loss with soft pseudo labels poses a great challenge as no previous method has investigated soft labels for triplet loss.

For tackling the difficulty, we propose to use softmax-triplet loss, whose hard version is formulated as

where

Here L bce (·, ·) denotes the binary cross-entropy loss, F (x and its positive sample x t i,p to measure their similarity, and "1" denotes the ground-truth that the positive sample x t i,p should be closer to the sample x t i than its negative sample x t i,n .

Given the two collaborative networks, we can utilize the one network's past temporal average model to generate soft triplet labels for the other network with the proposed soft softmax-triplet loss,

where

are the soft triplet labels generated by the two networks' past temporally average models.

Such soft triplet labels are fixed as training supervisions.

By adopting the soft softmax-triplet loss, our MMT framework overcomes the limitation of hard supervisions by the conventional triple loss (equation 2).

It can be successfully trained with soft triplet labels, which are shown to be important for improving the domain adaptation performance in our experiments.

Note that such a softmax-triplet loss was also studied in (Zhang et al., 2019a) .

However, it has never been used to generate soft labels and was not designed to work with soft pseudo labels before.

Our proposed MMT framework is trained with both off-line refined hard pseudo labels and on-line refined soft pseudo labels.

The overall loss function L(θ 1 , θ 2 ) simultaneously optimizes the coupled networks, which combines equation 1 We evaluate our proposed MMT on three widely-used person re-ID datasets, i.e., Market-1501 (Zheng et al., 2015) , DukeMTMC-reID (Ristani et al., 2016) , and MSMT17 .

The Market-1501 (Zheng et al., 2015) dataset consists of 32,668 annotated images of 1,501 identities shot from 6 cameras in total, for which 12,936 images of 751 identities are used for training and 19,732 images of 750 identities are in the test set.

DukeMTMC-reID (Ristani et al., 2016) contains 16,522 person images of 702 identities for training, and the remaining images out of another 702 identities for testing, where all images are collected from 8 cameras.

MSMT17 is the most challenging and large-scale dataset consisting of 126,441 bounding boxes of 4,101 identities taken by 15 cameras, for which 32,621 images of 1,041 identities are spitted for training.

For evaluating the domain adaptation performance of different methods, four domain adaptation tasks are set up, i.e., Duke-to-Market, Market-to-Duke, Duke-to-MSMT and Market-to-MSMT, where only identity labels on the source domain are provided.

Mean average precision (mAP) and CMC top-1, top-5, top-10 accuracies are adopted to evaluate the methods' performances.

4.2.1 TRAINING DATA ORGANIZATION For both source-domain pre-training and target-domain fine-tuning, each training mini-batch contains 64 person images of 16 actual or pseudo identities (4 for each identity).

Note that the generated hard pseudo labels for the target-domain fine-tuning are updated after each epoch, so the mini-batch of target-domain images needs to be re-organized with updated hard pseudo labels after each epoch.

All images are resized to 256 × 128 before being fed into the networks.

All the hyper-parameters of the proposed MMT framework are chosen based on a validation set of the Duke-to-Market task with M t = 500 pseudo identities and IBN-ResNet-50 backbone.

The same hyper-parameters are then directly applied to the other three domain adaptation tasks.

We propose a two-stage training scheme, where ADAM optimizer is adopted to optimize the networks with a weight decay of 0.0005.

Randomly erasing (Zhong et al., 2017b ) is only adopted in target-domain fine-tuning.

Stage 1: Source-domain pre-training.

We adopt ResNet-50 (He et al., 2016) or IBN-ResNet-50 (Pan et al., 2018) as the backbone networks, where IBN-ResNet-50 achieves better performances by integrating both IN and BN modules.

Two same networks are initialized with ImageNet (Deng et al., 2009 ) pre-trained weights.

Given the mini-batch of images, network parameters θ 1 , θ 2 are updated independently by optimizing equation 3 with λ s = 1.

The initial learning rate is set to 0.00035 and is decreased to 1/10 of its previous value on the 40th and 70th epoch in the total 80 epochs.

Stage 2: End-to-end training with MMT.

Based on pre-trained weights θ 1 and θ 2 , the two networks are collaboratively updated by optimizing equation 9 with the loss weights λ t id = 0.5, λ t tri = 0.8.

The temporal ensemble momentum α in equation 4 is set to 0.999.

The learning rate is fixed to 0.00035 for overall 40 training epochs.

We utilize k-means clustering algorithm and the number M t of pseudo classes is set as 500, 700, 900 for Market-1501 and DukeMTMC-reID, and 500, 1000, 1500, 2000 for MSMT17.

Note that actual identity numbers in the target-domain training Table 1 : Experimental results of the proposed MMT and state-of-the-art methods on Market-1501 (Zheng et al., 2015) , DukeMTMC-reID (Ristani et al., 2016) , and MSMT17 datasets, where MMT-M t represents the result with M t pseudo classes.

Note that none of M t values equals the actual number of identities but our method still outperforms all state-of-the-arts.

sets are different from M t .

We test different M t values that are either smaller or greater than actual numbers.

We compare our proposed MMT framework with state-of-the-art methods on the four domain adaptation tasks, Market-to-Duke, Duke-to-Market, Market-to-MSMT and Duke-to-MSMT.

The results are shown in Table 1 .

Our MMT framework significantly outperforms all existing approaches with both ResNet-50 and IBN-ResNet-50 backbones, which verifies the effectiveness of our method.

Moreover, we almost approach fully-supervised learning performances (Sun et al., 2018; Ge et al., 2018) without any manual annotations on the target domain.

No post-processing technique, e.g. re-ranking (Zhong et al., 2017a) or multi-query fusion (Zheng et al., 2015) , is adopted.

Specifically, by adopting the ResNet-50 (He et al., 2016 ) backbone, we surpass the state-of-theart clustering-based SSG by considerable margins of 11.7% and 12.9% mAP on Market-to-Duke and Duke-to-Market tasks with simpler network architectures and lower output feature dimensions.

Furthermore, evident 9.7% and 10.2% mAP gains are achieved on Market-to-MSMT and Duke-to-MSMT tasks.

Recall that M t is the number of clusters or number of hard pseudo labels manually specified.

More importantly, we achieve state-of-the-art performances on all tested target datasets with different M t , which are either fewer or more than the actual number of identities in the training set of the target domain.

Such results prove the necessity and effectiveness of our proposed pseudo label refinery for hard pseudo labels with inevitable noises.

Table 2 : Ablation studies of our proposed MMT on Duke-to-Market and Market-to-Duke tasks with M t of 500.

Note that the actual numbers of identities are not equal to 500 for both datasets but our MMT method still shows significant improvements.

To compare with relevant methods for tackling general noisy label problems, we implement Coteaching (Han et al., 2018) on unsupervised person re-ID task with 500 pseudo identities on the target domain, where the noisy labels are generated by the same clustering algorithm as our MMT framework.

The hard classification (cross-entropy) loss is adopted on selected clean batches.

All the hyper-parameters are set as the same for fair comparison, and the experimental results are denoted as "Co-teaching (Han et al., 2018) -500" with both ResNet-50 and IBN-ResNet-50 backbones in Table 1 .

Comparing "Co-teaching (Han et al., 2018 )-500 (ResNet-50)" with "Proposed MMT-500 (ResNet-50)", we observe significant 7.4% and 6.1% mAP drops on Market-to-Duke and Duketo-Market tasks respectively, since Co-teaching (Han et al., 2018) is designed for general close-set recognition problems with manually generated label noise, which could not tackle the real-world challenges in unsupervised person re-ID.

More importantly, it does not explore how to mitigate the label noise for the triplet loss as our method does.

In this section, we evaluate each component in our proposed framework by conducting ablation studies on Duke-to-Market and Market-to-Duke tasks with both ResNet-50 (He et al., 2016) and IBN-ResNet-50 (Pan et al., 2018) backbones.

Results are shown in Table 2 .

Effectiveness of the soft pseudo label refinery.

To investigate the necessity of handling noisy pseudo labels in clustering-based UDA methods, we create baseline models that utilize only off-line refined hard pseudo labels, i.e., optimizing equation 9 with λ t id = λ t tri = 0 for the two-step training strategy in Section 3.1.

The baseline model performances are present in Table 2 as "Baseline (only L t id & L t tri )".

Considerable drops of 17.7% and 14.9% mAP are observed on ResNet-50 for Duketo-Market and Market-to-Duke tasks.

Similarly, 13.8% and 10.7% mAP decreases are shown on the IBN-ResNet-50 backbone.

Stable increases achieved by the proposed on-line refined soft pseudo labels on different datasets and backbones demonstrate the necessity of soft pseudo label refinery and the effectiveness of our proposed MMT framework.

Effectiveness of the soft softmax-triplet loss.

We also verify the effectiveness of soft softmaxtriplet loss with softly refined triplet labels in our proposed MMT framework.

Experiments of removing the soft softmax-triplet loss, i.e., λ Specifically, the mAP drops are 5.3% on ResNet-50 and 4.8% on IBN-ResNet-50 when evaluating on the target dataset Market-1501.

As for the Market-to-Duke task, similar mAP drops of 3.6% and 4.0% on the two network structures can be observed.

An evident improvement of up to 5.3% mAP demonstrates the usefulness of our proposed soft softmax-triplet loss.

Effectiveness of Mutual Mean-Teaching.

We propose to generate on-line refined soft pseudo labels for one network with the predictions of the past average model of the other network in our MMT framework, i.e., the soft labels for network 1 are output from the average model of network 2 and vice versa.

We observe that the soft labels generated in such manner are more reliable due to the better decoupling between the past temporally average models of the two networks.

Such a framework could effectively avoid bias amplification even when the networks have much erroneous outputs in the early training epochs.

There are two possible simplification our MMT framework with less de-coupled structures.

The first one is to keep only one network in our framework and use its past temporal average model to generate soft pseudo labels for training itself.

Such experiments are denoted as "Baseline+MMT-500 (w/o θ 2 )".

The second simplification is to naïvely use one network's current-iteration predictions as the soft pseudo labels for training the other network and vice versa, i.e., α = 0 for equation 4.

This set of experiments are denoted as "Baseline+MMT-500 (w/o E[θ])".

Significant mAP drops compared to our proposed MMT could be observed in the two sets of experiments, especially when using the ResNet-50 backbone, e.g. the mAP drops by 8.9% on Duke-to-Market task when removing past average models.

This validates the necessity of employing the proposed mutual mean-teaching scheme for providing more robust soft pseudo labels.

In despite of the large margin of performance declines when removing either the peer network or the past average model, our proposed MMT outperforms the baseline model significantly, which further demonstrates the importance of adopting the proposed on-line refined soft pseudo labels.

Necessity of hard pseudo labels in proposed MMT.

Despite the robust soft pseudo labels bring significant improvements, the noisy hard pseudo labels are still essential to our proposed framework, since the hard classification loss L The initial network usually outputs uniform probabilities for each identity, which act as soft labels for soft classification loss, since it could not correctly distinguish between different identities on the target domain.

Directly training with such smooth and noisy soft pseudo labels, the networks in our framework would soon collapse due to the large bias.

One-hot hard labels for classification loss are critical for learning discriminative representations on the target domain.

In contrast, the hard triplet loss L t tri is not absolutely necessary in our framework, as experiments without L t tri , denoted as "Baseline+MMT-500 (w/o L t tri )" with λ t tri = 1.0, show similar performances as our final results with λ t tri = 0.8.

It is much easier to learn to predict robust soft labels for the soft softmax-triplet loss in equation 8 even at early training epochs, which has only two classes, i.e., positive and negative.

In this work, we propose an unsupervised Mutual Mean-Teaching (MMT) framework to tackle the problem of noisy pseudo labels in clustering-based unsupervised domain adaptation methods for person re-ID.

The key is to conduct pseudo label refinery to better model inter-sample relations in the target domain by optimizing with the off-line refined hard pseudo labels and on-line refined soft pseudo labels in a collaborative training manner.

Moreover, a novel soft softmax-triplet loss is proposed to support learning with softly refined triplet labels for optimal performances.

Our method significantly outperforms all existing person re-ID methods on domain adaptation task with up to 18.2% improvements.

Two temporal average models are introduced in our proposed MMT framework to provide more complementary soft labels and avoid training error amplification.

Such average models are more de-coupled by ensembling the past parameters and provide more independent predictions, which is ignored by previous methods with peer-teaching strategy (Han et al., 2018; Zhang et al., 2018b ).

Despite we have verified the effectiveness of such design in Table 2 by removing the temporal average model, denoted as "Baseline+MMT-500 (w/o E[θ])", we would like to visualize the training process by plotting the KL divergence between peer networks' predictions for further comparison.

As illustrated in Figure 3 , the predictions by two temporal average models ("Proposed MMT-500") always keep a larger distance than predictions by two ordinary networks ("Proposed MMT-500 (w/o E[θ])"), which indicates that the temporal average models could prevent the two networks in our MMT from converging to each other soon under the collaborative training strategy.

We utilize weighting factors of λ t tri = 0.8, λ t id = 0.5 in all our experiments by tuning on Duketo-Market task with IBN-ResNet-50 backbone and 500 pseudo identities.

To further analyse the impact of different λ t tri and λ t id on different tasks, we conduct comparison experiments by varying the value of one parameter and keep the others fixed.

Our MMT framework is robust and insensitive to different parameters except when the hard classification loss is eliminated with λ t id = 1.0.

The weighting factor of hard and soft triplet losses λ t tri .

In Figure 4 (a-b) , we investigate the effect of the weighting factor λ t tri in equation 9, where the weight for soft softmax-triplet loss is λ t tri and the weight for hard triplet loss is (1 − λ t tri ).

We test our proposed MMT-500 with both ResNet-50 and IBN-ResNet-50 backbones when λ t tri is varying from 0.0, 0.3, 0.5, 0.8 and 1.0.

Specifically, the soft softmax-triplet loss is removed from the final training objective (equation 9) when λ t tri is equal to 0.0, and the hard triplet loss is eliminated when λ t tri is set to 1.0.

We observe

@highlight

A framework that conducts online refinement of pseudo labels with a novel soft softmax-triplet loss for unsupervised domain adaptation on person re-identification.
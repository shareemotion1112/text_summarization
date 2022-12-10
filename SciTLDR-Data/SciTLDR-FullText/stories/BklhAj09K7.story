Unsupervised domain adaptation is a promising avenue to enhance the performance of deep neural networks on a target domain, using labels only from a source domain.

However, the two predominant methods, domain discrepancy reduction learning and semi-supervised learning, are not readily applicable when source and target domains do not share a common label space.

This paper addresses the above scenario by learning a representation space that retains discriminative power on both the (labeled) source and (unlabeled) target domains while keeping representations for the two domains well-separated.

Inspired by a theoretical analysis, we first reformulate the disjoint classification task, where the source and target domains correspond to non-overlapping class labels, to a verification one.

To handle both within and cross domain verifications, we propose a Feature Transfer Network (FTN) to separate the target feature space from the original source space while aligned with a transformed source space.

Moreover, we present a non-parametric multi-class entropy minimization loss to further boost the discriminative power of FTNs on the target domain.

In experiments, we first illustrate how FTN works in a controlled setting of adapting from MNIST-M to MNIST with disjoint digit classes between the two domains and then demonstrate the effectiveness of FTNs through state-of-the-art performances on a cross-ethnicity face recognition problem.

Despite strong performances on facial analysis using deep neural networks BID17 BID15 Schroff et al., 2015; Parkhi et al., 2015) , learning a model that generalizes across variations in attributes like ethnicity, gender or age remains a challenge.

For example, it is reported by BID5 that commercial engines tend to make mistakes at detecting gender for images of darker-skinned females.

Such biases have enormous social consequences, such as conscious or unconscious discrimination in law enforcement, surveillance or security (WIRED, 2018a; b; NYTimes, 2018; GIZMODO, 2018) .

A typical solution is to collect and annotate more data along the underrepresented dimension, but such efforts are laborious and time consuming.

This paper proposes a novel deep unsupervised domain adaptation approach to overcome such biases in face verification and identification.

Deep domain adaptation (Long et al., 2013; BID22 BID12 Sohn et al., 2017; Haeusser et al., 2017; Luo et al., 2017) allows porting a deep neural network to a target domain without extensive labeling efforts.

Currently, there are two predominant approaches to deep domain adaptation.

The first approach, domain divergence reduction learning, is motivated by the works of BID0 BID1 .

It aims to reduce the source-target domain divergence using domain adversarial training BID12 Sohn et al., 2017; BID19 or maximum mean discrepancy minimization BID22 Long et al., 2015; , while leveraging supervised loss from labeled source examples to maintain feature space discriminative power.

Since the theoretical basis of this approach BID0 assumes a common task between domains, it is usually applied to a classification problem where the source and target domains share the same label space and task definition.

The second approach considers domain adaptation as a semi-supervised learning problem and applies techniques such as entropy minimization (Grandvalet & Bengio, 2005) or self-ensembling (Laine & Aila, 2017; BID18 BID11 on target examples to encourage decisive and consistent predictions.

However, neither of those are applicable if the label spaces of source and target domains do not align.

As a motivating example, consider a cross-ethnicity generalization of face recognition problem, where the source ethnicity (e.g., Caucasian) contains labeled examples and the target ethnicity (e.g., African-American) contains only unlabeled examples.

When it is cast as a classification problem, the tasks of the two domains are different due to disjoint label spaces.

Moreover, examples from different ethnicity domains almost certainly belong to different identity classes.

To satisfy such additional label constraints, representations of examples from different domains should ideally be distant from each other in the embedding space, which conflicts with the requirements of domain divergence reduction learning as well as entropy minimization on target examples with source domain class labels.

In this work, we aim at learning a shared representation space between a source and target domain with disjoint label spaces that not only remains discriminative over both domains but also keep representations of examples from different domains well-separated, when provided with additional label constraints.

Firstly, to overcome the limitation of domain adversarial neural network (DANN) BID12 , we propose to convert disjoint classification tasks (i.e., the source and target domains correspond to non-overlapping class labels) into a unified binary verification task.

We term adaptation across such source and target domains as cross-domain distance metric adaptation (CD2MA).

We demonstrate a generalization of the theory of domain adaptation BID0 to our setup, which bounds the empirical risk for within-domain verification of two examples drawn from the unlabeled target domain.

While the theory does not guarantee verification between examples from different domains, we propose approaches that also address such cross-domain verification tasks.

To this end, we introduce a Feature Transfer Network (FTN) that separates the target features from the source features while simultaneously aligning them with an auxiliary domain of transformed source features.

Specifically, we learn a shared feature extractor that maps examples from different domains to representations far apart.

Simultaneously, we learn a feature transfer module that transforms the source representation space to another space used to align with the target representation space through a domain adversarial loss.

By forging this alignment, the discriminative power from the augmented source representation space would ideally be transferred to the target representation space.

The verification setup also allows us to introduce a novel entropy minimization loss in the form of N -pair metric loss (Sohn, 2016) , termed multi-class entropy minimization (MCEM), to further leverage unlabeled target examples whose label structure is not known.

MCEM samples pairs of examples from a discovered label structure within the target domain using an offline hierarchical clustering algorithm such as HDBSCAN BID6 , computes the N -pair metric loss among these examples (Sohn, 2016) , and backpropagates the resulting error derivatives.

In experiments, we first perform on a controlled setting by adapting between disjoint sets of digit classes.

Specifically, we adapt from 0-4 of MNIST-M BID12 dataset to 5-9 of MNIST dataset and demonstrate the effectiveness of FTN in learning to align and separate domains.

Then, we assess the impact of our proposed unsupervised CD2MA method on a challenging cross-ethnicity face recognition task, whose source domain contains face images of Caucasian identities and the target domain of non-Caucasian identities, such as African-American or East-Asian.

This is an important problem since existing face recognition datasets show significant label biases towards Caucasian ethnicity, leading to sub-optimal recognition performance for other ethnicities.

The proposed method demonstrates significant improvement in face verification and identification compared to a source-only baseline model and a standard DANN.

Our proposed method also closely matches the performance upper bounds obtained by training with fully labeled source and target domains.

Research efforts in deep domain adaptation have explored a proper metric to measure the variational distance between two domains and subsequently regularize neural networks to minimize this distance.

For example, maximum mean discrepancy (Long et al., 2013; BID21 BID9 BID22 BID14 estimates the domain difference based on kernels.

As another example, domain adversarial neural networks BID12 BID3 Sohn et al., 2017; Luo et al., 2017; BID19 , measuring the distance using a trainable and flexible discriminator often parameterized by an MLP, have been successfully adopted for several computer vision applications, such as semantic segmentation (Hoffman et al., 2016; BID20 BID31 and object detection BID7 .

Most of those works assume a common classification task between two domains, whereas we tackle a cross-domain distance metric adaptation problem where label spaces of source and target domains are different.

Moreover, our problem setting, an adaptation from labeled source to unlabeled target with disjoint label spaces, contains flavors from both domain adaptation (DA) and transfer learning (TL), following the nomenclature of (Pan et al., 2010) .

The difference in input distribution between source and target domains and the lack of labels in the target domain are similar to that of DA or transductive TL (Pan et al., 2010) , while the difference in label distribution and task definitions between two domains is akin to inductive TL (Pan et al., 2010; BID8 .

In our work, we formalize this problem in domain adaptation framework using verification as a common task.

This is a key contribution that allows theoretical analysis on the generalization bound as presented in Section 3 and Appendix A, while allowing novel applications like cross-ethnicity face recognition.

In terms of task objective, (Hu et al., 2015; BID12 Sohn et al., 2017) also deal with domain adaptation in distance metric learning, but neither learns a representation space capable of separating the source and target domains.

Resembling CD2MA, Luo et al. (2017) considers domain adaptation with disjoint label spaces, but the problem is still cast as classification with an assumption that the target label space is known and a few labeled target examples are provided for training.

In terms of network design, residual transfer network (Long et al., 2016) , which learns two classifiers differ by a residual function for the source and the target domain, is closely related.

However, it only tackles the scenario where source and target domains share a common label space for classification.

Under the domain adaptation assumption, BID0 show that the empirical risk on the target domain X T is bounded by the empirical risk on the source domain X S and the variational distance between the two domains, provided that the source and the target domains share the classifiers.

Therefore, this bound is not applicable to our CD2MA setup where the label spaces of two domains are often different.

To generalize those theoretical results to our setting, we reformulate the verification task as a binary classification task shared across two domains.

This new binary classification task takes a pair of images as an input and predicts the label of 1 if the pair of images shares the same identity and 0 otherwise.

Furthermore, if we now define the new source domain to be pairs of source images and the new target domain to be pairs of target images, then Theorem 1 and 2 from BID0 can be directly carried over to bound the new target domain binary classification error in the same manner.

That is, the empirical with-in target domain verification loss is bounded by with-in source domain verification loss and the variational distance between X S × X S and X T × X T .1 Note that inputs to the binary classifier are pairs of images from the same domain.

Thus, this setup only addresses adaptation of within-domain verification to unlabeled target domains.

There are two implications from the theoretical insights on domain adaptation using verification as a shared classification task.

Firstly, domain adversarial training, reducing the discrepancy between the source and the target product spaces, coupled with supervised source domain binary classification loss (i.e., verification loss using source domain labels) can yield target representations with high discriminative power when performing within-domain verification.

Note that in practice we approximately reduce the product space discrepancy by generic adversarial learning as done in BID12 Sohn et al., 2017) .

Secondly, there is no guarantee that the aligned source and target feature spaces possess any discriminative power for cross-domain verification task.

Thus, additional actions in the form of a feature transfer module and domain separation objective are required to address this issue.

These two consequences together motivate the design of our proposed framework, which is introduced in the next section.

In this section, we first define the CD2MA problem setup and motivate our proposed feature transfer network (FTN).

Then we elaborate on the training objectives that help our model achieve its desired properties.

Lastly, we provide practical considerations to implement our proposed algorithm.

Recall the description of CD2MA, given source and target domain data distributions X S and X T , our goal is to verify whether two random samples x, x drawn from either of the two distributions (and we do not know which distribution x or x come from a priori) belong to the same class.

There are 3 scenarios of constructing a pair: x, x ∈ X S , x, x ∈ X T , or x ∈ X S , x ∈ X T .

We refer the task of the first two cases as within-domain verification and the last as cross-domain verification.

DISPLAYFORM0 Figure 1: Training of Feature Transfer Network (FTN) for verification, composed of feature generation module (Gen; f ), feature transfer module (Tx; g), and two domain discriminators D1 and D2.

Verification objective Lvrf's are applied to source (fs) pairs and transformed source (g(fs))) pairs.

Our FTN applies domain adversarial objective Ladv for domain alignment between transformed source and target domains by D1 and applies Lsep to distinguish source domain from both target and transformed source domains by D2.If x, x ∈ X S (or X T ), we need a source (or target) domain classifier 2 .

For the source domain, we are provided with adequate labeled training examples to learn a competent classifier.

For the target domain, we are only given unlabeled examples.

However, with our extension of Theorem 1 and 2 from BID0 , discriminative power of the classifier can be transferred to the target domain by adapting the representation spaces of X T × X T and X S × X S , that is, we can utilize the same competent classifier from the source domain to verify target domain pairs if two domains are well-aligned.

For the third scenario where x ∈ X S but x ∈ X T , we assume that the two examples cannot be of the same class, which is true for problems such as cross-ethnicity face verification.

Our proposed framework, Feature Transfer Network (FTN), is designed to solve all these verification scenarios in an unified framework.

FTN is composed of multiple modules as illustrated in Figure 1 .

First, a feature generation module f : X → Z denoted as "Gen" in Figure 1 ideally maps X S and X T to distinguishable representation spaces, that is, f (X S ) and f (X T ) are far apart.

To achieve this, we introduce a domain separation objective.3 Next, the feature transfer module g : Z → Z denoted as "Tx" in Figure 1 transforms f (X S ) to g(f (X S )) for it to be aligned with f (X T ).

To achieve this, we introduce a domain adversarial objective.

Finally, we apply verification losses on f (X S ) and g(f (X S )) using classifiers h f , h g : Z × Z → {0, 1}. During testing, we compare the metric distance between f (x) and f (x ).

Overall, we achieve the following desired capabilities:• If x, x are from different domains, f (x) and f (x ) will be far away due to the functionality of the feature generation module.• If x, x ∈ X S , then f (x) and f (x ) will be close if they belong to the same class and far away otherwise, due to the discriminative power acquired from optimizing h f .•

If x, x ∈ X T , then f (x) and f (x ) will be close if they belong to the same class and far otherwise, due to the discriminative power acquired by optimizing h g with domain adversarial training.

We first define individual learning objectives of the proposed Feature Transfer Network and then present overall training objectives of FTN.

For ease of exposition, all objectives are to be maximized.

Verification Objective.

For a pair of source examples, we evaluate the verification losses at two representations spaces f (X S ) and g(f (X S )) using classifiers h f and h g as follows: DISPLAYFORM0 where DISPLAYFORM1 ) and y 12 = 1 if x 1 and x 2 are from the same class and 0 otherwise.

While classifiers h f , h g can be parameterized by neural networks, we aim to learn a generator f and g whose embeddings can be directly used as a distance metric.

Therefore, we use non-parameteric DISPLAYFORM2 As mentioned earlier, D 1 is trained to discriminate distributions f (X T ) and g(f (X S )) and then produces gradient for them to be indistinguishable.

The learning objectives are written as follows: DISPLAYFORM3 Note that when feature transform module is an identity mapping, i.e., g(f (x)) = f (x), Equation FORMULA4 defines the training objective of standard DANN.Domain Separation Objective.

The goal of this objective is to distinguish between source and target at representation spaces of generation module.

To this end, we formulate the objective using another domain discriminator D 2 : Z → (0, 1): DISPLAYFORM4 Note that, in L sep , the source space f (X S ) is not only pushed apart from the target space f (X T ) but also from the augmented source space g(f (X S )) to ensure that g learns meaningful transformation of source domain representation beyond identity transformation.

Training FTN.

Now we are ready to present the overall training objectives L f and L g : DISPLAYFORM5 with λ 1 for domain adversarial objective and λ 2 for domain separation objective.

We use L D1 in Equation FORMULA4 for DISPLAYFORM6 We alternate updating between D 1 and (f, g, D 2 ).

Preventing Mode Collapse via Feature Reconstruction Loss.

The mode collapsing phenomenon with generative adversarial networks (GANs) (Goodfellow et al., 2014) has received much attention (Salimans et al., 2016) .

In the context of domain adaptation, we also find it critical to treat the domain adversarial objective with care to avoid similar optimization instability.

In this work, we prevent the mode collapse issue for domain adversarial learning with an additional regularization method similar to (Sohn et al., 2017) .

Assuming the representation of the source domain is already close to optimal, we regularize the features of source examples to be similar to those from the reference network f ref : X → Z, which is pretrained on labeled source data and fixed during the training of f .

Furthermore, we add a similar but less emphasized (λ 4 < λ 3 ) regularization to target examples, simultaneously avoiding collapsing and allowing more room for target features to diverge from the original representations.

Finally, the feature reconstruction loss is written as follows: DISPLAYFORM0 We empirically find that without the feature reconstruction loss, the training would become unstable, reach an early local optimum and lead to suboptimal performance (see Section 6 and Appendix C).

Thus, we always include the feature reconstruction loss to train DANN or FTN models unless stated otherwise.

Replacing Verification Loss with N -pair Loss.

Our theoretical analysis in Section 3 (and Appendix A) suggests to use a verification loss that compares similarity between a pair of images.

In practice, however, the pairwise verification loss is too weak to learn a good deep distance metric.

Following (Sohn, 2016) , we propose to replace the verification loss with an N -pair loss, defined as follows: DISPLAYFORM1 where x n and x + n are from the same class and x n and x + k , n = k, are from different classes.

Replacing L vrf into L N , the training objective of FTN with N -pair loss is written as follows: DISPLAYFORM2

Entropy minimization (Grandvalet & Bengio, 2005 ) is a popular training objective in unsupervised domain adaptation: unlabeled data is trained to minimize entropy of a class prediction distribution so as to form features that convey confident decision rules.

However, it is less straightforward how to apply entropy minimization when label spaces for source and target are disjoint.

Motivated from Section 3, we extend entropy minimization for distance metric adaptation using verification as a common task for both domains: DISPLAYFORM0 where DISPLAYFORM1 .

This formulation encourages a more confident prediction for verifying two unlabeled images, whether or not coming from the same class.

However, recall that for the source domain, we use N -pair loss instead of pair-wise verification loss for better representation learning.

Therefore, we would like to similarly incorporate the concept of Npair loss on the target domain by forging a multi-class entropy minimization (MCEM) objective.

This demands N pair examples to be sampled from the target domain.

As the target domain is unlabeled, we ought to first discover a plausible label structure, which is done off-line via HDBSCAN BID6 McInnes et al., 2017) , a fast and scalable density-based hierarchical clustering algorithm.

The returned clusters provide pseudo-labels to individual examples of the target domain, allowing us to sample N pair examples to evaluate the following MCEM objective: DISPLAYFORM2 where x n and x + n are from the same cluster and x n and x + k , n = k are from different clusters.

The objective can be combined with L f in Equation FORMULA10 to optimize f .

In this section, we first experiment on digit datasets as a proof of concept and compare our proposed FTN to DANN.

Then, we tackle the problem of cross-ethnicity generalization in the context of face recognition to demonstrate the effectiveness of FTN.

In all experiments, we use N -pair loss as defined in Equation FORMULA10 to update f and g for better convergence and improved performance.

We also use the same learning objectives for DANN while fixing g to the identity mapping and λ 2 = 0.

To provide insights on the functionality of FTN, we conduct an experiment adapting the digits 0-4 from MNIST-M BID12 to 5-9 from MNIST.

In other words, the two domains in our setting not only differ in foreground and background patterns but also contain non-overlapping digit classes, contrasting the usual adaptation setup with a shared label space.

Our goal is to learn a feature space that separates the digit classes not only within each domain, but also across the two.

We construct a feature generator f composed of a CNN encoder followed by two fully-connected (FC) layers and a feature transfer module g composed of MLP with residual connections.

Outputs of f and g are then fed to discriminators D 1 and D 2 parameterized by MLPs to induce domain adversarial and domain separation losses respectively.

We provide more architecture details in Appendix B.1.

FIG0 .

Without an adaptation FIG0 ), features of digits from the target domain are heavily mixed with those from the source domain as well as one another.

The model reaches 1.3% verification error in the source domain but as high as 27.3% in the target domain.

Though DANN in FIG0 (b) shows better separation with a reduced target verification error of 2.2%, there still exists significant overlap between digit classes across two domains, such as 3/5, 4/9, 0/6 and 2/8.

As a result, a domain classifier trained to distinguish source and target on top of generator features can only attain 11.5% classification error.

In contrast, the proposed FTN in FIG0 (c) shows 10 clean clusters without any visual overlap among 10 digits classes from either source or target domain, implying that it not only separates digits within the target domain (2.1% verification error), but also differentiates them across domains (0.3% domain classification error).

The performances of face recognition engines have significantly improved thanks to recent advances in deep learning for image recognition (Krizhevsky et al., 2012; Simonyan & Zisserman, 2015; BID16 He et al., 2016) and publicly available large-scale face recognition datasets Guo et al., 2016) .

However, most public datasets are collected from the web by querying celebrities, with significant label bias towards Caucasian ethnicity.

For example, more than 85% of identities are Caucasian for CASIA Web face dataset .

Similarly, 82% are Caucasian (CAU) for MS-Celeb-1M (MS-1M) dataset (Guo et al., 2016) , while there are only 9.7% African-American (AA), 6.4% East-Asian (EA) and less than 2% Latino and South-Asian combined.

Such imbalance across ethnicity in labeled training data can result in significant drop in identification performance on data-scarce minorities: the second row of Table 1 shows a model trained on Caucasian dominated dataset performs poorly on the other ethnicities.

As expected, if the training data is composed of only Caucasian identities as source domain, the performance over the target domains consisting of the other ethnicities further deteriorates (see row 1 of Table 1 ).

Provided the available labeled source domain contains only Caucasian identities, we subsequently demonstrate that our method can effectively leverage unlabeled data from the non-Caucasian target ethnicity to substantially improve their face verification performances.

Experimental Setup.

We perform an adaptation from CAU to a mixture of AA and EA.

Our experiments use the MS-1M dataset.

We first remove identities that both appear in the training and testing sets.

The resulting training set consists of 4.04M images from 60K CAU identities, 398K images from 7K AA identities, and 308K images from 4.6K EA identities.

For domain adaptation experiments, we use labeled CAU images and unlabeled AA, EA images for training.

For supervised experiments to obtain performance lower and upper bound, we use labeled CAU images to train Sup C and labeled CAU, AA, EA images to train Sup C,A,E .We adopt a 38-layer ResNet (He et al., 2016) for the feature generation module.

Feature transfer module and discriminators are parameterized with MLPs similarly to Section 6.1.

We use 4096-pair loss for training, including for the supervised CNNs.

It is worth mentioning that our network architecture and training scheme result in strongly competitive face recognition performance, comparing to other state-of-the-art methods such as FaceNet (Schroff et al., 2015) on YouTube Faces BID26 (97.32% (ours) vs 95.12%) and Neural Aggregation Network BID28 on IJB-A (see row 2 of Table 3 ).

The complete network architecture and training details are provided in Appendix B.2.Evaluation.

We report the performance of the baseline and our proposed models on two standard face recognition benchmarks LFW (Huang et al., 2007) and IJB-A (Klare et al., 2015) .

Note that these datasets also exhibit significant ethnicity bias.

To highlight the effectiveness of the proposed adaptation approach, we construct individual test set for CAU, AA, EA, each of which contains 10 face images from 200 identities.

We refer to our testing set as the Cross-Ethnicity Faces (CEF) dataset.

We apply two evaluation metrics on CEF dataset, verification accuracy and identification accuracy.

For verification, following the standard protocol (Huang et al., 2007) , we construct 10 splits, each containing 900 positive and 900 negative pairs, and compute the accuracy on each split using the threshold found from the other 9 splits.

For identification, a pair composed of the reference and the query images from the same identity is considered correct if there is no image from different identity that has higher similarity to the reference image than the query image.

We evaluate identification accuracy per ethnicity (200-way) as well as across all ethnicities (600-way).Results.

The results on CEF are summarized in Table 1 .

Cross domain identification accuracy is reported in TAB1 , where we use AA and EA as negative classes when evaluating accuracy on CAU and vice versa, as a measure to indicate domain discrepancy.

Among adaptation models, DANN without feature reconstruction loss (DANN\L recon ) shows unstable training and easily degenerate, which leads to only marginal improvement upon Sup C .

Similar trend is observed while training FTN.

Therefore, to ensure training stability, we impose L recon as a regularization term for all adaptation models.

More analysis on the effectiveness of L recon is provided in Appendix C.When testing on AA and EA with model trained on only the labeled source CAU domain (Sup C ), we observe significant performance drops in Table 1 .

Meanwhile, in TAB1 , cross domain identification accuracy is much higher than within domain identification accuracy, i.e., 96.14% of AA vs. CAU is much higher than 71.92% of AA identification in Table 1 , indicating 1) significant discrepancy between the feature spaces of the source and target domains and 2) lack of discriminative power for within domain verification task on target ethnicity.

Comparing to Sup C , both DANN and FTN show moderate improvement when testing on AA and EA from CEF (Table 1) , demonstrating the effectiveness of domain adversarial learning in transferring within domain verification capability from labeled source domain to unlabeled target domain.

Despite the improvement, DANN suffers a notable drawback from adversarial objective which attempts to align identities from different domains, resulting a poor cross domain identification accuracy as shown in TAB1 .

In contrast, the proposed FTN achieves much higher cross domain identification accuracy, demonstrating both within and cross domain discriminative power.

Additionally, in combination with the multi-class entropy minimization (FTN+MCEM), we further boost the verification and identification accuracy over FTN on AA and EA as well as approach the accuracy of Sup C,A,E , the performance upper bound.

This indicates that the HDBSCAN-based hierarchical clustering provides high quality pseudo-class labels for MCEM to be effective.

Indeed, the clustering algorithm achieves F-score as high as 96.31% and 96.34% on AA and EA.

We provide more in-depth analysis on the clustering strategy in Appendix D.Finally, Table 3 reports the performance of face recognition models on standard verification and recognition benchmarks.

We observe similar improvements with our proposed distance metric adaptation when only using labeled CAU, i.e., source domain, as training data.

Once the task becomes more challenging thus demands more discriminative power, the advantage of our method becomes more evident, such as in the case of open-set recognition and verification at low FAR.

We address the challenge of unsupervised domain adaptation when the source and the target domains have disjoint label spaces by formulating the classification problem into a verification task.

We propose a Feature Transfer Network, allowing simultaneous optimization of domain adversarial loss and domain separation loss, as well as a variant of N -pair metric loss for entropy minimization on the target domain where the ground-truth label structure is unknown, to further improve the adaptation quality.

Our proposed framework excels at both within-domain and cross-domain verification tasks.

As an application, we demonstrate cross-ethnicity face verification that overcomes label biases in training data, achieving high accuracy even for unlabeled ethnicity domains, which we believe is a result with vital social significance.

Vinod Nair and Geoffrey E Hinton.

Following (Haeusser et al., 2017) , we preprocess the data by subtracting a channel-wise pixel mean and dividing by channel-wise standard deviation of pixel values.

For MNIST examples, we also apply color-intensity inversion.

All images are resized into 32×32 with 3 channels.

Our feature generator module is composed of 6 convolution layers and 3 max-pooling layers followed by 2 fully-connected layers.

We use ReLU (Nair & Hinton, 2010) after convolution layers.

The output dimension of the feature generator module is 128 and is normalized to have L2-norm of 2.

The full description of the generator module is in TAB5 .The feature transfer module maps 128 dimensional vector into the same dimensional vector using two fully-connected layers (128 − 256 − 256 − 128) and residual connection as in Figure 1 (a).

Discriminator architectures are similar to that in Figure 1 (b) but with fully-connected layers whose output dimensions are 128 instead of 256.We use Adam stochastic optimizer with learning rate of 0.0003, λ 1 = 0.3 and λ 2 = 0.03 to train FTN.

Our experimental protocols, such as data preprocessing and network architecture, closely follow those of (Sohn et al., 2017) .

We preprocess face images by detecting BID27 , aligning BID30 , and cropping to provide face images of size 110 × 110.

The data is prepared for network training by random cropping into 100 × 100 with horizontal flip with a 50% chance and converting into gray-scale.

Our feature generation module contains 38 layers of convolution with several residual blocks and max pooling layers.

We use ReLU (Nair & Hinton, 2010) for most of the layers in combination with maxout nonlinearities (Goodfellow et al., 2013) .

We add 7 × 7 average pooling layer on top of the last convolution layer.

The output of the feature generation module is 320 dimensional vector and is normalized to have L2-norm of size 12.

The full description of the model is in TAB1 .The feature transfer module maps 320 dimensional output vector from feature generation module into the same dimensional vector using two fully-connected layers and residual connection.

The architecture of feature transfer module is described in Figure 1 (a).

Discriminators have similar network architecture besides different numbers of neurons and omitted residual connection.

All models, including supervised CNNs (Sup C , Sup C,A,E ), are trained with 4096-pair loss.

For Sup C and Sup C,A,E , we use Adam stochastic optimizer (Kingma & Ba, 2015) with the learning rate of 0.0003 for the first 12K updates and 0.0001 and 0.00003 for the next two subsequent 3K updates.

Our feature generation module is initialized with the Sup C model, which is also used as a reference network for feature reconstruction loss as described in Section 4.3.

Other modules of our model, such as feature generation module and discriminators, are initialized randomly.

All modules are then updated with the learning rate of 0.00003.

Hyperparameters of different models are summarized in Table S3 .

We demonstrate the effectiveness of feature reconstruction loss in stabilizing the domain adversarial training in DANN framework.

We train four different DANN models with different configurations of λ 3 and λ 4 .

We visualize in FIG0 the performance curves of identification accuracy evaluated on the AA, EA, and CAU ethnicities of CEF dataset.

Note that we stop training early on when the performance start to degrade significantly.

Therefore, x-axis, the number of training epoch, of different curves are different.

y-axis represents the identification accuracy.

As we see in FIG0 , the performance of all models on the target ethnicities start to improve in the beginning of training from those of the pretrained reference network.

Soon after, however, the accuracy starts to drop when values of either λ 3 or λ 4 are set to 0.

Note that even in that situation the performance on the CAU set still remains high, which implies the failure of discriminative information transfer.

On the other hand, our proposed feature reconstruction loss with non-zero values of λ 3 and λ 4 FIG0 ) shows much more stable performance curve.

Nonetheless, values of λ 3 and λ 4 should be carefully selected since the feature generation module of DANNs or FTNs will remain almost the same to the reference network when they are set too strong and the effectiveness of the domain adversarial loss will be reduced.

In our experiment, we use λ 3 = 0.1 and λ 4 = 0.01 for DANN, λ 3 = 0.03 and λ 4 = 0.01 for FTN.

For FTN with entropy minimization we further reduce λ 4 = 0.003 to give more flexibility in updating model parameters based on entropy loss.

DISPLAYFORM0

In this section, we provide analysis on the performance of our clustering strategy by measuring the clustering accuracy.

Specifically, we measure the verification precision and recall as follows: DISPLAYFORM0 where y i is the ground-truth class label of an example x i , andŷ i is an index of an assigned cluster.

Precision computes the proportion of positive pairs among pairs assigned to the same cluster, i.e., purity of returned clusters, and recall computes the proportion of positive pairs assigned to the same cluster.

Ideally, we expect high precision and high recall, i.e., high F-score, to ensure examples with the same class labels are assigned to the same cluster.

Note that we only use clusters of size 5 or larger as new target classes and discard examples assigned to a cluster whose size is less than 5.Here, in addition to our proposed clustering strategy, we also evaluate the clustering performance that clusters target examples by finding a nearest classes or examples from the source domain, which are shown to be effective for zero-shot learning BID23 or semi-supervised domain adaptation with disjoint source and target classes (Luo et al., 2017) .

In this case, we call two examples from the target domain are assigned to the same cluster if the nearest source examples are the same.

We also measure the clustering performance by matching the nearest source classes.

The summary result is provided in Table S4 .

Firstly, we observe extremely low precision when using source domain examples or clusters as a proxy to relate target examples.

We believe that this idea of "clustering by finding the nearest source classes" works under a cross-category similarity assumption between disjoint classes of source and target domains.

In other words, it assumes that there exists a certain source class closer to examples from certain target class, so that those examples from the same target class can be clustered around that source class, even though those matching source and target classes are indeed different (e.g., 3/5, 2/8, 4/9, and 0/6 in Section 6.1).

Unfortunately, such an assumption does not hold for our problem, maybe due to the huge number of identity classes (60K) in the source domain.

On the other hand, using hierarchical clustering on target features achieves significantly higher precision and recall.

Especially, when using embedding vectors of Sup C , we achieve 100% precision, which means that all clusters are pure even though some ground-truth classes might be separated into multiple clusters.

We observe slightly lower precision using FTN features but much higher recall, achieving higher F-score overall.

Further, the number of examples returned with FTN feature (253K and 195K for AA and EA, respectively) is higher than with Sup C feature (217K and 165K).

Repeating the process using feature of FTN+MCEM model improves the F-score while returning more target examples that are with cluster assignment (276K and 214K Table S4 : Verification precision and recall of clustering methods, such as projection to source example or source class center, or hierarchical clustering using embeddings of Sup C (HDBSCAN) or our proposed FTN model.

Furthermore, we repeat the clustering using the FTN with multi-class entropy minimization model (FTN+MCEM) and report the clustering accuracy.improved discriminative quality of features by FTNs, but also suggests a potential tool for automatic labeling of unlabeled data by iterative training of FTN model and hierarchical clustering.

We visualize few images from each ethnicity subset in FIG1 for annotation quality assurance.

@highlight

A new theory of unsupervised domain adaptation for distance metric learning and its application to face recognition across diverse ethnicity variations.

@highlight

Proposes a novel feature transfer network that optimizes domain adversarial loss and domain separation loss.
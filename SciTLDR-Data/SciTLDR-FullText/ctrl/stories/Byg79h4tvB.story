This paper presents a generic framework to tackle the crucial class mismatch problem in unsupervised domain adaptation (UDA) for multi-class distributions.

Previous adversarial learning methods condition domain alignment only on pseudo labels, but noisy and inaccurate pseudo labels may perturb the multi-class distribution embedded in probabilistic predictions, hence bringing insufficient alleviation to the latent mismatch problem.

Compared with pseudo labels, class prototypes are more accurate and reliable since they summarize over all the instances and are  able  to  represent  the  inherent  semantic  distribution  shared  across  domains.

Therefore, we propose a novel Prototype-Assisted Adversarial Learning (PAAL) scheme, which incorporates instance probabilistic predictions and class prototypes together  to  provide  reliable  indicators  for  adversarial  domain  alignment.

With the PAAL scheme,  we align both the instance feature representations and class prototype  representations  to  alleviate  the  mismatch  among  semantically  different classes.

Also,  we exploit the class prototypes as proxy to minimize the within-class variance in the target domain to mitigate the mismatch among semantically similar classes.

With these novelties, we constitute a Prototype-Assisted Conditional Domain Adaptation (PACDA) framework which well tackles the class mismatch problem.

We demonstrate the good performance and generalization ability of the PAAL scheme and also PACDA framework on two UDA tasks, i.e., object recognition (Office-Home,ImageCLEF-DA, andOffice) and synthetic-to-real semantic segmentation (GTA5→CityscapesandSynthia→Cityscapes).

Unsupervised domain adaptation (UDA) aims to leverage the knowledge of a labeled data set (source domain) to help train a predictive model for a unlabeled data set (target domain).

Deep UDA methods bring noticeable performance gain to many tasks (Long et al., 2015; Saito et al., 2017; Richter et al., 2016; Tsai et al., 2018; Lee et al., 2019; Vu et al., 2019a) by exploiting supervision from heterogeneous sources.

Some methods exploit maximum mean discrepancy (MMD) (Gretton et al., 2008; Long et al., 2015) or other distribution statistics like central moments (Sun & Saenko, 2016; Zellinger et al., 2017; Koniusz et al., 2017) for domain adaptation.

Recently, generative adversarial learning (Goodfellow et al., 2014) provides a promising alternative solution to UDA problem.

Since the labels of the target instances are not given in UDA, adversarial learning scheme for adaptation (Ganin & Lempitsky, 2015) suffers from the cross-domain misalignment, where the target instances from a class A are potentially misaligned with source instances from another class B. Inspired by the pseudo-labeling strategy from semi-supervised learning, previous methods either used the pseudo labels in the target domain to perform joint distribution discrepancy minimization (Long et al., 2013; or developed conditional adversarial learning methods that involve one high-dimensional domain discriminator or multiple discriminators (Chen et al., 2017b; Pei et al., 2018) .

Though effective, these conditional domain adversarial learning methods align different instances from different domains relying only on their own predictions.

Simple probabilistic predictions or pseudo labels may not accurately represent the semantic information of input instances, misleading the alignment.

A toy example is given in Fig. 1(a) .

The pseudo label of the chosen instance x is inclined to be class 'square' while the ground truth label is class 'circle'.

Only guided by the instance prediction, the 'circle' class in the target domain and the 'square' class in the source domain are easily confused, causing the misalignment in the adversarial domain adaptation.

To remedy the misalignment, we propose to exploit the class prototypes for adversarial domain alignment, instead of using only the possibly inaccurate predictions.

Prototypes are global feature representations of different classes and are relevant to the inherent semantic structures shared across (a) conditional adversarial learning (b) prototype-assisted adversarial learning

Figure 1: Illustration of two adversarial learning schemes.

Different from class-agnostic adversarial learning that pursues the marginal distribution alignment but ignores the semantic consistency, (a) conditional adversarial learning relies heavily on the instance-level pseudo labels to perform conditional distribution alignment, while (b) our prototype-assisted adversarial learning integrates the instance-level pseudo labels and global class prototypes to make the conditional indicators more reliable.

Class information is denoted in different shapes with source in solid and target in hollow.

domains.

As shown in Fig. 1(b) , class prototypes are expected to remedy the negative effects of inaccurate probabilistic predictions.

Motivated by this, we propose a Prototype-Assisted Adversarial Learning (PAAL) scheme which complements instance predictions with class prototypes to obtain more reliable conditional information for guiding the source-target feature representation alignment.

Specifically, we summarize the class prototypes from all instances according to their predictions.

In this way, on one hand, we lower the dependence of class prototypes on instance predictions which may be inaccurate, and on the other hand, we encourage the instances with greater certainty to contribute more to their corresponding class prototypes.

The prototypes are updated dynamically through a moving average strategy to make them more accurate and reliable.

Then by broadcasting class prototypes to each instance according to its probability prediction, the inaccurate semantic distribution depicted by instance predictions can be alleviated.

Based on reliable prototype-based conditional information, we align both the instance feature representations and the class prototypes through the proposed PAAL scheme to relieve the alignment among semantically dissimilar instances.

However, such a conditional domain alignment may promote the confusion among semantically similar instances across domains to some degree.

To further alleviate it, we introduce an intra-class objective in the target domain to pursue the class compactness.

Built on the proposed PAAL scheme and this intra-class compactness objective, we develop a Prototype-Assisted Conditional Domain Adaptation (PACDA) framework for solving UDA problems.

Extensive experimental evaluations on both object recognition and semantic segmentation tasks clearly demonstrate the advantages of our approaches over previous state-of-the-arts Xu et al., 2019; Tsai et al., 2019) .

The contributions of this work can be summarized into three folds: 1) To the best of our knowledge, we are the first to leverage the class prototypes in conditional adversarial learning to prevent the misalignment in UDA; 2) We propose a simple yet effective domain adversarial learning framework PACDA to remedy the misalignment among semantically similar instances as well as semantically dissimilar instances; 3) The proposed PAAL scheme and PACDA framework are generic, and our framework achieves the state-of-the-art results on several unsupervised domain adaptation tasks including object recognition and semantic segmentation.

Unsupervised Domain Adaptation.

UDA is first modeled as the covariate shift problem (Shimodaira, 2000) where marginal distributions of different domains are different but their conditional distributions are the same.

To address it, (Dudík et al., 2006; Huang et al., 2007) exploit a nonparametric instance re-weighting scheme.

Another prevailing paradigm (Pan et al., 2010; Long et al., 2013; Herath et al., 2017) aims to learn feature transformation with some popular cross-domain metrics, e.g., the empirical maximum mean discrepancy (MMD) statistics.

Recently, a large number of deep UDA works (Long et al., 2015; Haeusser et al., 2017; Saito et al., 2018; Tsai et al., 2018) have been developed and boosted the performance of various vision tasks.

Generally, they can be divided into discrepancy-based and adversarial-based methods.

Discrepancy-based methods (Tzeng et al., 2014; Long et al., 2017) address the dataset shift by mitigating specific discrepancies defined on different layers of a shared model between domains, e.g. resembling shallow feature transforma-tion by matching higher moment statistics of features from different domains (Zellinger et al., 2017; Koniusz et al., 2017) .

Recently, adversarial learning has become a dominantly popular solution to domain adaptation problems.

It leverages an extra domain discriminator to promote domain confusion. (Ganin & Lempitsky, 2015) designs a gradient reversal layer inside the classification network and (Tzeng et al., 2017) utilizes an inverted label GAN loss to fool the discriminator.

Pseudo-labeling.

UDA can be regarded as a semi-supervised learning (SSL) task where unlabeled data are replaced by the target instances.

Therefore, some popular SSL strategies, e.g., entropy minimization (Grandvalet & Bengio, 2005; Vu et al., 2019b) , mean-teacher (Tarvainen & Valpola, 2017; French et al., 2018) , and virtual adversarial training (Miyato et al., 2018; Shu et al., 2018) , have been successfully applied to UDA.

Pseudo-labeling is favored by most UDA methods due to its convenience.

For example, (Saito et al., 2017; exploit the intermediate pseudo-labels with tri-training and self-training, respectively. (Pan et al., 2019) obtains target-specific prototypes with the help of pseudo labels and aligns prototypes across domains at different levels.

Recently, curriculum learning (Choi et al., 2019) , self-paced learning (Zou et al., 2018) and re-weighting schemes ) are further leveraged to tackle possible false pseudo-labels.

Conditional Domain Adaptation.

Apart from the explicit integration with the last classifier layer, pseudo-labels can also be incorporated into adversarial learning to enhance the feature-level domain alignment.

Concerning shallow methods (Long et al., 2013; Zhang et al., 2017) , pseudo-labels can help mitigate the joint distribution discrepancy via minimizing multiple class-wise MMD measures. (Long et al., 2017) proposes to align the joint distributions of multiple domain-specific layers across domains based on a joint maximum mean discrepancy criterion.

Recently, (Chen et al., 2017b; Pei et al., 2018) leverages the probabilities with multiple domain discriminators to enable fine-grained alignment of different data distributions in an end-to-end manner.

In contrast, conditions the adversarial domain adaptation on discriminative information via the outer product of feature representation and classifier prediction.

Motivated by the semantically-consistent GAN, (Cicek & Soatto, 2019 ) imposes a multi-way adversarial loss instead of a binary one on the domain alignment.

However, these methods all highly rely on the localized pseudo-labels to align labelconditional feature distributions and ignore the global class-level semantics.

As far as we know, we are the first to exploit class prototypes to guide the domain adversarial learning.

Compared with (Pei et al., 2018; , our PACDA framework complements the original feature representations with reliable semantic features and merely involves two low-dimensional domain discriminators, making the domain alignment process simple, conditional, and reliable.

In this section, we first begin with the basic settings of UDA and then give detailed descriptions on the proposed PAAL scheme and the PACDA framework.

Though proposed for image classification, they can also be easily applied to semantic segmentation.

In a vanilla UDA task, we are given label-rich source domain data {(

sampled from the joint distribution P s (x s , y s ) and unlabeled target domain data {x

, where x i s ∈ X S and y i s ∈ Y S denote an image and its corresponding label from the source domain dataset, x i t ∈ X T denotes an image from the target domain dataset and P s = Q t .

The goal of UDA is to learn a discriminative model from X S , Y S , and X T to predict labels for unlabeled target samples X T .

As described in (Ganin et al., 2016) , a vanilla domain adversarial learning framework consists of a feature extractor network G, a classifier network F , and a discriminator network D. Given an image x, we denote the feature representation vector extracted by G as f = G(x) ∈ R d and the probability prediction obtained by F as p = F (f ) ∈ R c where d means the feature dimension and c means the number of classes.

The vanilla domain adversarial learning method in (Ganin et al., 2016) can be formulated as optimizing the following minimax optimization problem: min .

M ema represents the global class prototype matrix while M s,t is computed by source or target instances within current batch.

where the binary domain classifier D :

predicts the domain assignment probability over the input features, L y (G, F ) is the cross-entropy loss of source domain data as for the classification task, and λ adv is the trade-off parameter.

The misalignment in UDA of multi-class distributions challenges the popular vanilla adversarial learning.

In previous works (Long et al., 2017; Pei et al., 2018; , target domain data are conditioned only on corresponding pseudo labels predicted by the model for adversarial domain alignment.

The general optimization process of these methods is the same as aforementioned vanilla domain adversarial learning, except that feature representations jointly with predictions are considered by the discriminator D:

is the conditional adversarial loss that leverages the classification predictions p s and p t .

A classic previous work implicitly conditions the feature representation on the prediction through the outer product f ⊗ p, and uses one shared discriminator to align the conditioned feature representations.

further proves that using the outer product can perform much better than simple concatenation f ⊕ p. Different from , (Chen et al., 2017b; Pei et al., 2018) explicitly utilize multiple class-wise domain discriminators to align the feature representations relying on the corresponding predictions.

However, the pseudo labels may be inaccurate due to the domain shift.

Therefore, only conditioning the alignment on pseudo labels can not safely remedy the misalignment.

Compared with the pseudo labels, the class prototypes are more robust and reliable in terms of representing the shared semantic structures .

To acquire more reliable and accurate conditional information for domain adversarial learning, we propose to complement instance predictions with class prototypes and reformulate the adversarial loss to:

c×d denotes the global class prototype matrix in our prototype-assisted adversarial learning loss L paal adv (G, D).

In reality, the reliable conditional information is obtained through broadcasting the global class prototypes to each independent instance according to its prediction p.

We propose to summarize feature representations of the instances within the same class as the corresponding prototype.

Then the probability prediction is leveraged to obtain accurate class prototypes.

Using predictions as weights can adaptively control the contributions of typical and non-typical instances to the class prototype, making class prototypes more reliable.

Specifically, we first gather the feature representation of each instance relying on its prediction to generate the batch-level class prototypes.

Then the global class prototypes can be obtained by virtue of an averaging strategy such as exponential moving average (ema) on the batch ones.

This process can be formulated as

Here n means the batch size, p k,i represents the probability of the i-th instance belonging to the k-th semantic class, λ ema is an empirical weight, M ∈ R c×d is the batch-level class prototype matrix and M ema is the global one computed by certain source domain data and contributes to more reliable conditional information exploited by discriminators.

Similarly, batch-level class prototypes are broadcast to each instance in this batch through M T a p a which can be denoted as f a , a ∈ {s, t}.

3.3 PROTOTYPE-ASSISTED CONDITIONAL DOMAIN ADAPTATION (PACDA) FRAMEWORK With our prototype-based conditional information, we further propose a Prototype-Assisted Conditional Domain Adaptation (PACDA) framework.

This framework aligns both instance-level and prototype-level feature representations through PAAL and promotes the intra-class compactness in target domain such that the misalignment can be substantially alleviated even though no supervision is available in the target domain.

Its overall architecture is shown in Fig. 2 .

Besides the backbone feature extractor G and the task classifier F , there are two discriminators in our framework PACDA, i.e., the instance-level feature discriminator D f and the prototype-level feature discriminator D p .

We can formulate our general objective function as (

where λ denotes balance factors among different loss functions, L y is the supervised classification loss on source domain data described by Eq. (3), L f adv is the adversarial loss to align instance feature representations across domains, L p adv is the adversarial loss to align class prototype representations across domains, and L t is the loss to promote the intra-class compactness in target domain.

Instance-Level Alignment Conditioning the instance feature representation on our prototype-based conditional information, we seek to align feature representations across domains at the instance-level through discriminator D f .

With the assistance of the accurate semantic structures embedded in class prototypes, misalignment among semantically dissimilar instances can be effectively alleviated.

We can define the instance-level adversarial loss L adv f as

(8) Prototype-Level Alignment Instance-level alignment only implicitly aligns the multi-class distribution across domains, which may not ensure the semantic consistency between two domains.

Besides, since in practice global class prototypes are collected from only source domain data, which possibly cannot accurately represent inherent semantic structures in the target domain due to the domain shift.

Taking into account these two causes, we perform the prototype-level alignment with discriminator D p to explicitly align the class prototype representations across domains.

The specific loss function

(9) Intra-Class Compactness Although adversarial alignment based on PAAL can relieve the misalignment among obviously semantically different instances, it cannot well handle the misalignment among semantically similar instances.

Specifically, incorporating class prototypes into instance predictions would confuse semantically similar instances during domain alignment and result in the misalignment among them.

To solve this problem, our framework further promotes the intra-class compactness in the target domain to enlarge the margin between instances of semantically similar classes.

Taking the prototypes as proxy, we minimize the following loss for target domain samples to encourage the intra-class compactness:

Thus, the complete minimax optimization problem of our PACDA framework can be formulated as

With only two low-dimensional (2 × d) discriminators added, we effectively remedy the misalignment in domain adversarial learning.

Some theoretical insights with the help of domain adaptation theory (Ben-David et al., 2010 ) is discussed in the Appendix.

We conduct experiments to verify the effectiveness and generalization ability of our methods, i.e., PACDA (full) in Eq. (11) and PAAL (λ p adv = λ t = 0) on two different UDA tasks, including cross-domain object recognition on ImageCLEF-DA 1 , Office31 (Saenko et al., 2010) and OfficeHome (Venkateswara et al., 2017) , and synthetic-to-real semantic segmentation for GTA5 (Richter et al., 2016) →Cityscapes (Cordts et al., 2016) and Synthia (Ros et al., 2016 )→ Cityscapes.

Datasets.

Office-Home is a new challenging dataset that consists of 65 different object categories found typically in 4 different Office and Home settings, i.e., Artistic (Ar) images, Clip Art (Ca), Product images (Pr), and Real-World (Re) images.

ImageCLEF-DA is a standard dataset built for the 'ImageCLEF2014:domain-adaptation' competition.

We follow (Long et al., 2015) to select 3 subsets, i.e., C, I, and P, which share 12 common classes.

Office31 is a popular dataset that includes 31 object categories taken from 3 domains, i.e., Amazon (A), DSLR (D), and Webcam (W).

Cityscapes is a realistic dataset of pixel-level annotated urban street scenes.

We use its original training split and validation split as the training target data and testing target data respectively.

GTA5 consists of 24,966 densely labeled synthetic road scenes annotated with the same 19 classes as Cityscapes.

For Synthia, we take the SYNTHIA-RAND-CITYSCAPES set as the source domain, which is composed of 9,400 synthetic images compatible with annotated classes of Cityscapes.

Implementation Details.

For object recognition, we follow the standard protocol (Ganin & Lempitsky, 2015) , i.e. using all the labeled source instances and all the unlabeled target instances for UDA, and report the average accuracy based on three random trials for fair comparisons.

Following Xu et al., 2019) , we experiment with ResNet-50 model pretrained on ImageNet.

Specifically, we follow to choose the network parameters, and all convolutional layers and the classifier layer are trained through backpropagation, where λ t =5e-3, λ ema =5e-1, λ f adv and λ p adv increase from 0 to 1 with the same strategy as (Ganin & Lempitsky, 2015) .

Regarding the domain discriminator, we design a simple two-layer classifier (256→1024→1) for both D f and D p .

Empirically, we fix the batch size to 36 with the initial learning rate being 1e-4.

For semantic segmentation, we adopt DeepLab-V2 (Chen et al., 2017a ) based on ResNet-101 (He et al., 2016) as done in (Tsai et al., 2018; Vu et al., 2019b; Tsai et al., 2019) .

Following DCGAN (Radford et al., 2015) , the discriminator network consists of three 4 × 4 convolutional layers with stride 2 and channel numbers {256, 512, 1}. In training, we use SGD (Bottou, 2010) to optimize the network with momentum (0.9), weight decay (5e-4), and initial learning rate (2.5e-4).

We use the same learning rate policy as in (Chen et al., 2017a) .

Discriminators are optimized by Adam (Kingma & Ba, 2015) with momentum (β 1 = 0.9, β 2 = 0.99), initial learning rate (1e-4) along with the same decreasing strategy as above.

For both tasks, λ f adv is set to 1e-3 following (Tsai et al., 2018) and λ ema is set to 0.7.

For GTA5→Cityscapes, λ p adv =1e-3 and λ t =1e-5.

For Synthia→Cityscapes, λ p adv =1e-4 and λ t =1e-4.

All experiments are implemented via PyTorch on a single Titan X GPU.

The total iteration number is set as 10k for object recognition and 100k for semantic segmentation.

For objection recognition tasks, we choose the hyper-parameters which have the minimal mean entropy of target data (Morerio et al., 2018) on Ar→Cl for convenience.

For semantic segmentation tasks, training split of Cityscapes is used for the hyper-parameters selection.

Data augmentation skills like random scale or random flip and ten-crop ensemble evaluation are not adopted.

Cross-Domain Object Recognition.

The comparison results between our methods (i.e., PAAL and PACDA) and state-of-the-art (SOTA) approaches (Xu et al., 2019; on Office-Home, Office31, and ImageCLEF-DA are shown in Tables 1 and 2, respectively.

As indicated in these tables, PACDA improves previous approaches in the average accuracy for all three benchmarks (e.g., 67.3%→68.7% for Office-Home, 88.1%→88.8% for ImageCLEF-DA, and 87.7%→89.3% for Office31).

Generally, PACDA performs the best for most transfer tasks.

Taking 89.9 78.5 94.7 79.5 92.0 89.7 87.4 90.1 92.5 72.1 98.8 69.9 100.

87.2 CAT (Deng et al., 2019) 91 Table 3 : Comparison results of synthetic-to-real semantic segmentation using the same architecture with NonAdapt and AdaptSeg (Tsai et al., 2018) , AdvEnt (Vu et al., 2019b) , CLAN and AdaptPatch (Tsai et al., 2019) .

Top: GTA5 → Cityscapes.

Bottom: Synthia → Cityscapes.

a careful look at PAAL, we find that it always beats CDAN and achieves competitive performance with SOTA methods like CAT (Deng et al., 2019) .

Synthetic-to-real Semantic Segmentation.

We compare PAAL and PACDA with SOTA methods (Tsai et al., 2018; Vu et al., 2019b; Tsai et al., 2019) on synthetic-to-real semantic segmentation.

Following (Chen et al., 2017b) , we evaluate models on all 19 classes for GTA5→Cityscapes while on only 13 classes for Synthia→Cityscapes.

As shown in Table 3 , without bells and whistles, our PAAL method outperforms all of those methods and our PACDA framework further achieves new SOTA results on both tasks, i.e., 43.8%→46.6% for GTA5→Cityscapes and 47.8%→49.2% for Synthia→Cityscapes in terms of the mean IoU (mIoU) value.

Quantitative Analysis.

To verify the effectiveness of each component in Eq. (11), we introduce a variant named PAAL f,p that merely ignores the intra-class objective (λ t = 0).

The empirical convergence curves about Ar→Cl in Fig. (3)(a) imply that all of our variants tend to converge after 10k iterations, and the second term can help accelerate the convergence.

Fig. 3(b) shows that all terms in the PACDA framework, i.e., PAAL alignment at different levels and the intra-class objective, can bring evident improvement on both tasks.

As shown in Fig. ( 3)(c), we provide the proxy A-distances (Ganin et al., 2016) of different methods for Ar→Cl and C→I. The A-distance Dist A =2(1 − 2 ) is a popular measure for domain discrepancy, where is the test error of a binary classifier trained on the learned features.

All the UDA methods have smaller distances than 'source only' by aligning different domains.

Besides, our PACDA has the minimum distance for both tasks, implying that it can learn better features to bridge the domain gap between domains.

To testify the sensitivity of our PACDA, in Fig. (3) Qualitative Analysis.

For object recognition, we study the t-SNE visualizations of aligned features generated by different UDA methods in Fig. 4 .

As expected, conditional methods including CDAN and PAAL can semantically align multi-class distributions much better than DANN.

Besides, PAAL learns slightly better features than CDAN due to less misalignment.

Once considering the intra-class objective, PACDA further enhances PAAL by pushing away semantically confusing classes, which achieves the best adaptation performance.

For semantic segmentation, we present some qualitative results in Fig. 5 .

Similarly, PAAL effectively improves the adaptation performance and PAAL f,p as well as PACDA can further improve the segmentation results.

In this work, we developed the prototype-assisted adversarial learning scheme to remedy the misalignment for UDA tasks.

Unlike previous conditional ones whose performance is vulnerable to inaccurate instance predictions, our proposed scheme leverages the reliable and accurate class prototypes for aligning multi-class distributions across domains and is demonstrated to be more effective to prevent the misalignment.

Then we further augment this scheme by imposing the intra-class compactness with the prototypes as proxy.

Extensive evaluations on both object recognition and semantic segmentation tasks clearly justify the effectiveness and superiority of our UDA methods over well-established baselines.

We try to explain why our PAAL works well for UDA according to the domain adaptation theory proposed in (Ben-David et al., 2010) .

Denote by P (F ) = E (f,y)∈P [F (f ) = y] the risk of a classifier model F ∈ H w.r.t.

the distribution P , and by P (F 1 , F 2 ) = E (f,y)∈P [F 1 (f ) = F 2 (f )] the disagreement between hypotheses F 1 , F 2 ∈ H. Particularly, (Ben-David et al., 2010) gives a well-known upper bound on the target risk Q (F ) of classifier F in the following,

where F * is the ideal classifier induced from F * = arg min F ∈H [ P (F ) + Q (F )], and the last term is related to the classical H-divergence d H∆H (P, Q) = 2 sup F,F * ∈H | P (F, F * ) − Q (F, F * )|.

Besides, according to (Ben-David et al., 2010) , the empirical H-divergence calculated by m respective samples from distributions P and Q converges uniformly to the true H-divergence for classifier classes H of finite VC dimension d, which is expressed as d H∆H (P, Q) ≤d H∆H (P, Q) + 4 d log(2m) + log(2/δ) m .

The work (Ganin & Lempitsky, 2015) introduces a binary domain discriminator to minimize the empirical H-divergenced H∆H (P, Q), which aligns the marginal distributions well.

However, if two multi-class distributions P and Q are not semantically aligned, there may not be any classifier with low risk in both domains, which means the second term of the upper bound in Eq. (12) is very large.

The proposed PAAL scheme leverages reliable conditional information in the adversarial learning module so that semantically similar samples from different domains are implicitly aligned, thus it has a high possibility of decreasing the second term.

Compared with , the input to domain the adversarial learning module is much more compact (2 × d c × d), which helps decrease the second term in Eq. (13).

<|TLDR|>

@highlight

We propose a reliable conditional adversarial learning scheme along with a simple, generic yet effective framework for UDA tasks.
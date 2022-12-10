There is a strong incentive to develop versatile learning techniques that can transfer the knowledge of class-separability from a labeled source domain to an unlabeled target domain in the presence of a domain-shift.

Existing domain adaptation (DA) approaches are not equipped for practical DA scenarios as a result of their reliance on the knowledge of source-target label-set relationship (e.g. Closed-set, Open-set or Partial DA).

Furthermore, almost all the prior unsupervised DA works require coexistence of source and target samples even during deployment, making them unsuitable for incremental, real-time adaptation.

Devoid of such highly impractical assumptions, we propose a novel two-stage learning process.

Initially, in the procurement-stage, the objective is to equip the model for future source-free deployment, assuming no prior knowledge of the upcoming category-gap and domain-shift.

To achieve this, we enhance the model’s ability to reject out-of-source distribution samples by leveraging the available source data, in a novel generative classifier framework.

Subsequently, in the deployment-stage, the objective is to design a unified adaptation algorithm capable of operating across a wide range of category-gaps, with no access to the previously seen source samples.

To achieve this, in contrast to the usage of complex adversarial training regimes, we define a simple yet effective source-free adaptation objective by utilizing a novel instance-level weighing mechanism, named as Source Similarity Metric (SSM).

A thorough evaluation shows the practical usability of the proposed learning framework with superior DA performance even over state-of-the-art source-dependent approaches.

Deep learning models have proven to be highly successful over a wide variety of tasks (Krizhevsky et al., 2012; Ren et al., 2015) .

However, a majority of these remain heavily dependent on access to a huge amount of labeled samples to achieve a reliable level of generalization.

A recognition model trained on a certain distribution of labeled samples (source domain) often fails to generalize when deployed in a new environment (target domain) in the presence a discrepancy in the input distribution (Shimodaira, 2000) .

Domain adaptation (DA) algorithms seek to minimize this discrepancy either by learning a domain invariant feature representation (Long et al., 2015; Kumar et al., 2018; Ganin et al., 2016; Tzeng et al., 2015) , or by learning independent domain transformations (Long et al., 2016) to a common latent representation through adversarial distribution matching (Tzeng et al., 2017; Nath Kundu et al., 2018) , in the absence of target label information.

Most of the existing approaches (Zhang et al., 2018c; Tzeng et al., 2017 ) assume a common label-set shared between the source and target domains (i.e. C s = C t ), which is often regarded as Closed-Set DA (see Fig. 1 ).

Though this assumption helps to analyze various insights of DA algorithms, such an assumption rarely holds true in real-world scenarios.

Recently researchers have independently explored two broad adaptation settings by partly relaxing the above assumption.

In the first kind, Partial DA (Zhang et al., 2018b; Cao et al., 2018a; b) , the target label space is considered as a subset of the source label space (i.e. C t ⊂ C s ).

This setting is more suited for large-scale universal source datasets, which will almost always subsume the label-set of a wide range of target domains.

However, the availability of such a universal source is highly questionable for a wide range of input domains and tasks.

In the second kind, regarded as Open-set DA (Baktashmotlagh et al., 2019; Ge et al., 2017) , the target label space is considered as a superset of the source label space (i.e. C t ⊃ C s ).

The major challenge in this setting is attributed to detection of target samples from the unobserved categories in a fully-unsupervised scenario.

Apart from the above two extremes, certain works define a partly mixed scenario by allowing "private" label-set for both source and target domains (i.e. C s \ C t = ∅ and C t \ C s = ∅) but with extra supervision such as few-shot labeled data (Luo et al., 2017) or access to the knowledge of common categories (Panareda Busto & Gall, 2017) .

Most of the prior approaches consider each scenario in isolation and propose independent solutions.

Thus, they require access to the knowledge of label-set relationship (or category-gap) to carefully choose a DA algorithm, which would be suitable for the problem in hand.

Furthermore, all the prior unsupervised DA works require coexistence of source and target samples even during deployment, hence not source-free.

This is highly impractical, as labeled source data may not be accessible after deployment due to several reasons such as, privacy concerns, restricted access to proprietary data, accidental loss of source data or other computational limitations in real-time deployment scenarios.

Acknowledging the aforementioned shortcomings, we propose one of the most convenient DA frameworks which is ingeniously equipped to address source-free DA for all kinds of label-set relationships, without any prior knowledge of the associated category-gap (i.e. universal-DA).

We not only focus on identifying the key complications associated with the challenging problem setting, but also devise insightful ideas to tackle such complications by adopting learning techniques much different from the available DA literature.

This leads us to realize a holistic solution which achieves superior DA performance even over prior source-dependent approaches.

We briefly review the available domain adaptation methods under the three major divisions according to the assumption on label-set relationship.

a) Closed-set DA.

The cluster of previous works under this setting focuses on minimizing the domain gap at some intermediate feature level either by minimizing well-defined statistical distance functions (Wang & Schneider, 2014; Duan et al., 2012; Zhang et al., 2013; Saenko et al., 2010) or by formalizing it as an adversarial distribution matching problem (Tzeng et al., 2017; Kang et al., 2018; Long et al., 2018; Hu et al., 2018; inspired from the Generative Adversarial Nets (Goodfellow et al., 2014) .

Certain prior works (Sankaranarayanan et al., 2018; Zhu et al., 2017; use GAN framework to explicitly generate target-like images translated from the source image samples, which is also regarded as pixel-level adaptation (Bousmalis et al., 2017 ) in contrast to other feature level adaptation works (Nath Kundu et al., 2018; Tzeng et al., 2017; Long et al., 2015; .

b) Partial DA.

Focusing on Partial DA, Cao et al. (2018a) proposed to achieve adversarial class-level matching by utilizing multiple domain discriminators furnishing class-level and instance-level weighting for individual data samples.

Zhang et al. (2018b) proposed to utilize importance weights for source samples depending on their similarity to the target domain data using an auxilliary discriminator.

To effectively address the problem of negative-transfer (Wang et al., 2019), Cao et al. (2018b) employed a single discriminator to achieve both adversarial adaptation and class-level weighting of source samples.

c) Open-set DA.

Saito et al. (2018b) proposed a more general open-set adaptation setting without accessing the knowledge of source private labels set in contrast to the prior work (Panareda Busto & Gall, 2017) .

They extended the source classifier to accommodate an additional "unknown" class, which is trained adversarially against the other source classes.

Universal DA.

You et al. (2019) proposed Universal DA, which requires no prior knowledge of label-set relationship similar to the proposed setting, but considers access to both source and target samples during adaptation.

The problem setting for source-free domain adaptation is broadly divided into a two stage process.

a) Procurement stage.

In this stage, we are given full access to the labeled samples of source domain,

where p is the distribution of source samples and C s denotes the label-set of the source domain.

Here, the objective is to equip the model for the second stage, i.e. the Deployment stage, in the presence of a discrepancy in the distribution of input target samples.

To achieve this we rely on an artificially generated negative dataset, D n = {(x n , y n ) : x n ∼ p n , y n ∈ C n }, where p n is the distribution of negative source samples such that C n ∩ C s = ∅. Figure 2: Latent space cluster arrangement during adaptation (see Section 3.1.1).

b) Deployment stage.

After obtaining a trained model from the Procurement stage, the model will have its first encounter with the unlabeled target domain samples from the deployed environment.

We denote the unlabeled target data by D t = {x t : x t ∼ q}, where q is the distribution of target samples.

Note that, access to the source dataset D s from the previous stage is fully restricted during adaptation in the Deployment stage.

Suppose that, C t is the "unknown" label-set of the target domain.

We define the common label space between the source and target domain as C = C s ∩ C t .

The private label-set for the source and the target domains is represented as C s = C s \ C t and C t = C t \ C s respectively.

3.1.1 Challenges.

The available DA techniques heavily rely on the adversarial discriminative (Tzeng et al., 2017; Saito et al., 2018a) strategy.

Thus, they require access to the source samples to reliably characterize the source domain distribution.

Moreover, these approaches are not equipped to operate in a source-free setting.

Though a generative model can be used as a memory-network (Sankaranarayanan et al., 2018; Bousmalis et al., 2017) to realize source-free adaptation, such a solution is not scalable for large-scale source datasets (e.g. ImageNet (Russakovsky et al., 2015) ), as it introduces unnecessary extra parameters in addition to the associated training difficulties (Salimans et al., 2016) .

This calls for a fresh analysis of the requirements beyond the solutions found in literature.

In a general DA scenario, with access to source samples in the Deployment stage (specifically for Open-set or Partial DA), a widely adopted approach is to learn domain invariant features.

In such approaches the placement of source category clusters is learned in the presence of unlabeled target samples which obliquely provides a supervision regarding the relationship between C s and C t .

For instance, in case of Open-set DA, the source clusters may have to disperse to make space for the clusters from target private C t (see Fig. 2a to 2b) .

Similarly, in partial DA, the source clusters may have to rearrange themselves to keep all the target shared clusters (C = C t ) separated from the source private C s (see Fig. 2a to 2c ).

However in a complete source-free framework, we do not have the liberty to leverage such information as source and target samples never coexist together during training.

Motivated by the adversarial discriminative DA technique (Tzeng et al., 2017) , we hypothesize that, inculcating the ability to reject samples that are out of the source data distribution can facilitate future source-free domain alignment using this discriminatory knowledge.

Therefore, in the Procurement stage the overarching objective is two-fold.

• Firstly, we must aim to learn a certain placement of source clusters best suited for all kinds of category-gap scenarios acknowledging the fact that, a source-free scenario does not allow us to modify the placement in the presence of target samples during adaptation (see Fig. 2d ).

• Secondly, the learned embedding must have the ability to reject out-of-distribution samples, which is an essential requirement for unsupervised adaptation in the presence of domain-shift.

3.1.2 Solution.

In the presence of source data, we aim to restrain the model's domain and category bias which is generally inculcated as a result of the over-confident supervised learning paradigms (see Fig. 4A ).

To achieve this goal, we adopt two regularization strategies viz.

i) regularization via generative modeling and ii) utilization of a labeled simulated negative source dataset to generalize for the latent regions not covered by the given positive source samples (see Fig. 4C ).

How to configure the negative source dataset?

While configuring D n , the following key properties have to be met.

Firstly, latent clusters formed by the negative categories must lie in-between the latent clusters of positive source categories to enable a higher degree of intra-class compactness with interclass separability (Fig. 4C) .

Secondly, the negative source samples must enrich the source domain n .

One of the key characteristics shared between the samples from source and unknown target domain is the semantics of the local part-related features specifically for image-based object recognition tasks.

Relying on this assumption, we propose a systematic procedure to simulate the samples of D n by randomly compositing local regions between a pair of images drawn from the positive source dataset D s (see Fig. 3A and appendix, Algo.

2).

Intuitively, composite samples x n created on image pairs from different source categories are expected to lie in-between the two positive source clusters in the latent space, thereby introducing a combinatorial amount of new class labels i.e.

n .

As an alternative approach, in the absence of domain knowledge (e.g. non-image datasets, or for tasks beyond image-recognition such as pose estimation), we propose to sample virtual negative instances, u n from the latent space which are away from the high confidence regions (3-sigma) of positive source clusters (Fig. 4B) .

For each negative sample, we assign a negative class label (one of |C n | = |Cs| C 2 ) corresponding to the pair of most confident source classes predicted by the classifier.

Thus, we obtain D

is the distribution of negative samples in the latent u-space (more details in appendix Algo.

3).

Training procedure.

The generative source classifier is divided into three stages; i) backbone-model M , ii) feature extractor F s , and iii) classifier D (see Fig. 3B ).

Output of the backbone-model is denoted as v = M (x), where x is drawn from either D s or D n .

Following this, the output of F s and D are represented as u and d respectively.

.

Additionally, we define priors of only positive source classes as P (u s |c i ) = N (u s |µ ci , Σ ci ) for i = 1, 2...|C s | at Algorithm 1 Training algorithm in the Procurement stage 1: input: (xs, ys) ∈ Ds, (xn, yn) ∈ Dn; θF s , θD, θG: Parameters of Fs, D and G respectively.

2: initialization: pretrain {θF s , θD} using cross-entropy loss on (xs, ys) followed by initialization of the sample mean µc i and covariance Σc i (at u-space) of Fs • M (xs) for xs from class ci; i = 1, 2, ...|Cs| 3: for iter < M axIter do 4:

where ks and kn are the index of ground-truth label ys and yn respectively.

6:

; Lv = |vs −vs|; Lu = |ur −ûr| 7:

Update θF s , θD, θG by minimizing LCE, Lv, Lu, and Lp alternatively using separate optimizers.

9:

if (iter % U pdateIter == 0) then 10:

Recompute the sample mean (µc i ) and covariance (Σc i ) of Fs • M (xs) for xs from class ci;

n : generate fresh latent-simulated negative samples using the updated priors) the intermediate embedding

Here, parameters of the normal distributions are computed during training as shown in line-10 of Algo.

1.

A cross-entropy loss over these prior distributions is defined as L p (line-7 in Algo.

1), to effectively enforce intra-class compactness with inter-class separability (progression from Fig. 4B to 4C).

Motivated by generative variational auto-encoder (VAE) setup (Kingma & Welling, 2013), we introduce a feature decoder G, which aims to minimize the cyclic reconstruction loss selectively for the samples from positive source categories v s and randomly drawn samples u r from the corresponding class priors (i.e. L v and L u , line-6 in Algo.

1).

This along with a lower weightage α for the negative source categories (i.e. at the cross-entropy loss L CE , line-6 in Algo.

1) is incorporated to deliberately bias F s towards the positive source samples, considering the level of unreliability of the generated negative dataset.

3.2.1 Challenges.

We hypothesize that, the large number of negative source categories along with the positive source classes i.e. C s ∪ C n can be interpreted as a universal source dataset, which can subsume label-set C t of a wide range of target domains.

Moreover, we seek to realize a unified adaptation algorithm, which can work for a wide range of category-gaps.

However, a forceful adaptation of target samples to positive source categories will cause target private samples to be classified as an instance of the source private or the common label-set, instead of being classified as "unknown", i.e. one of the negative categories in C n .

In contrast to domain agnostic architectures (You et al., 2019; Cao et al., 2018a; Saito et al., 2018a) , we resort to an architecture supporting domain specific features (Tzeng et al., 2017) , as we must avoid disturbing the placement of source clusters obtained from the Procurement stage.

This is an essential requirement to retain the task-dependent knowledge gathered from the source dataset.

Thus, we introduce a domain specific feature extractor denoted as F t , whose parameters are initialized from the fully trained F s (see Fig. 3B ).

Further, we aim to exploit the learned generative classifier from the Procurement stage to complement for the purpose of separate ad-hoc networks (critic or discriminator) as utilized by the prior works (You et al., 2019; Cao et al., 2018b ).

We define a weighting factor (SSM) for each target sample x t , as w(x t ).

A higher value of this metric indicates x t 's similarity towards the positive source categories, specifically inclined towards the common label space C. Similarly, a lower value of this metric indicates x t 's similarity towards the negative source categories C n , showing its inclination towards the private target labels C t .

Let, ps, qt be the distribution of source and target samples with labels in C s and C t respectively.

We define, p c and q c to denote the distribution of samples from source and target domains belonging to the shared label-set C. Then, the SSM for the positive and negative source samples should lie on the two extremes, forming the following inequality:

To formalize the SSM criterion we rely on the class probabilities defined at the output of source model only for the positive class labels, i.e.ŷ (k) for k = 1, 2...|C s |.

Note that,ŷ (k) is obtained by performing softmax over |C s | + |C n | categories as discussed in the Procurement stage.

Finally, the SSM and its complement are defined as,

We hypothesize that, the above definition will satisfy Eq. 1, as a result of the generative learning strategy adopted in the Procurement stage.

In Eq. 2 the exponent is used to further amplify separation between target samples from the shared C and those from the private C t label-set (see Fig. 5A ).

b) Source-free domain adaptation.

To perform domain adaptation, the objective function aims to move the target samples with higher SSM value towards the clusters of positive source categories and vice-versa at the frozen source embedding, u-space (from the Procurement stage).

To achieve this, parameters of only F t network are allowed to be trained in the Deployment stage.

However, the decision of weighting the loss on target samples towards the positive or negative source clusters is computed using the source feature extractor F s i.e. the SSM in Eq. 2.

We define, the deployment model as h = D • F t • M (x t ) using the target feature extractor, with softmax predictions over K categories obtained asẑ

.

Thus, the primary loss function for adaptation is defined as,

Additionally, in the absence of label information, there would be uncertainty in the predictionsẑ

as a result of distributed class probabilities.

This leads to a higher entropy for such samples.

Entropy minimization (Grandvalet & Bengio, 2005; Long et al., 2016 ) is adopted in such scenarios to move the target samples close to the highly confident regions (i.e. positive and negative cluster centers from the Procurement stage) of the classifier's feature space.

However, it has to be done separately for positive and negative source categories based on the SSM values of individual target samples to effectively distinguish the target-private set from the full target dataset.

To achieve this, we define two different class probability vectors separately for the positive and negative source classes denoted as, z Fig. 3B ).

Entropy of the target samples in the positive and negative regimes of the source classifier is obtained as

n respectively.

Consequently, the entropy minimization loss is formalized as,

Thus, the final loss function for adapting the parameters of F t is presented as

Here β is a hyper-parameter controlling the importance of entropy minimization during adaptation.

We perform a thorough evaluation of the proposed source-free, universal domain adaptation framework against prior state-of-the-art models across multiple datasets.

We also provide a comprehensive ablation study to establish generalizability of the approach across a variety of label-set relationships and justification of the various model components.

Datasets.

For all the following datasets, we resort to the experimental settings inline with the recent work by You et al. (2019) (UAN) .

Office-Home (Venkateswara et al., 2017) dataset consists of images from 4 different domains -Artistic (Ar), Clip-art (Cl), Product (Pr) and Real-world (Rw).

Alphabetically, the first 10 classes are selected as C, the next 5 classes as C s , and the rest 50 as C t .

VisDA2017 (Peng et al., 2018) dataset comprises of 12 categories with synthetic images as the source domain and natural images as the target domain, out of which, the first 6 are chosen as C, the next 3 as C s and the rest as C t .

Office-31 (Saenko et al., 2010) dataset contains images from 3 distinct domains -Amazon (A), DSLR (D) and Webcam (W).

We use the 10 classes shared by Office-31 and Caltech-256 (Gong et al., 2012) to construct the shared label-set C and alphabetically select the next 10 as C s , with the remaining 11 classes contributing to C t .

To evaluate scalability, ImageNet-Caltech is also considered with 84 common classes inline with the setting in You et al. (2019) .

Simulation of labeled negative samples.

To simulate negative labeled samples for training in the Procurement stage, we first sample a pair of images, each from different categories of C s , to create unique negative classes in C n .

Note that, we impose no restriction on how the hypothetical classes are created (e.g. one can composite non-animal with animal).

A random mask is defined which splits the images into two complementary regions using a quadratic spline passing through a central image region (see Appendix Algo.

2).

Then, the negative image is created by merging alternate mask regions as shown in Fig. 3A .

For the I→C task of ImageNet-Caltech, the source domain (ImageNet), consisting of 1000 classes, results in a large number of possible negative classes (i.e. |C n | = |Cs| C 2 ).

We address this by randomly selecting only 600 of these negative classes for ImageNet(I), and 200 negative classes for Caltech(C) in the task C→I. In a similar fashion, we generate latent-simulated negative samples only for the selected negative classes in these datasets.

Consequently, we compare two models with different Procurement stage training -(i) USFDA-a: using image-composition as negative dataset , and (ii) USFDA-b: using latent-simulated negative samples as the negative dataset.

We use USFDA-a for most of our ablation experiments unless mentioned explicitly.

Average accuracy on Target dataset, T avg .

We resort to the evaluation protocol proposed in the VisDA2018 Open-Set Classification challenge.

Accordingly, all the target private classes are grouped into a single "unknown" class and the metric reports the average of per-class accuracy over |C s | + 1 classes.

In the proposed framework a target sample is marked as "unknown", if it is classified (argmax kẑ (k) ) into any of the negative |C n | classes out of total |C s | + |C n | categories.

In contrast, UAN (You et al., 2019) relies on a sensitive hyperparameter, as a threshold on the sample-level weighting, to mark a target sample as "unknown".

Also note that, our method is completely source-free during the Deployment stage, while all other methods have access to the full source-data.

Accuracy on Target-Unknown data, T unk .

We evaluate the target unknown accuracy, T unk , as the proportion of actual target private samples (i.e. {(x t , y t ) : y t ∈ C t }) being classified as "unknown" after adaptation.

Note that, UAN (You et al., 2019) does not report T unk which is a crucial metric to evaluate the vulnerability of the model after its deployment in the target environment.

The T avg metric fails to capture this as a result of class-imbalance in the Open-set scenario (Saito et al., 2018b) .

Hence, to realize a common evaluation ground, we train the UAN implementation provided by the authors (You et al., 2019) and denote it as UAN* in further sections of this paper.

We observe that, the UAN(You et al., 2019) training algorithm is often unstable with a decreasing trend of T unk and T avg over increasing training iterations.

We thus report the mean and standard deviation of the peak values of T unk and T avg achieved by UAN*, over 5 separate runs on Office-31 dataset (see Table 7 ).

Implementation Details.

We implement our network in PyTorch and use ResNet-50 (He et al., 2016) as the backbone-model M , pre-trained on ImageNet (Russakovsky et al., 2015) inline with UAN (You et al., 2019) .

The complete architecture of other components with fully-connected layers is provided in the Supplementary.

A sensitivity analysis of the major hyper-parameters used in the proposed framework is provided in Fig. 5B -C, and Appendix Fig. 8B .

In all our ablations across the datasets, we fix the hyperparameters values as α = 0.2 and β = 0.1.

We utilize Adam optimizer (Kingma & Ba, 2014 ) with a fixed learning rate of 0.0001 for training in both Procurement and Deployment stage (see Appendix for the code).

For the implementation of UAN*, we use the hyper-parameter value w 0 = −0.5, as specified by the authors for the task A→D in Office-31 dataset.

a) Comparison with prior arts.

We compare our approach with UAN You et al. (2019) , and other prior methods.

The results are presented in Table 1 and Table 2 .

Clearly, our framework achieves state- Relative freq.

x t from target-private x t from target-shared P iter =100 P iter =500 of-the-art results even in a source-free setting on several tasks.

Particularly in Table 2 , we present the target-unknown accuracy T unk on various dataset.

It also holds the mean and standard-deviation for both the accuracy metrics computed over 5 random initializations in the Office-31 dataset (the last six rows).

Our method is able to achieve much higher T unk than UAN* (You et al., 2019), highlighting our superiority as a result of the novel learning approach incorporated in both Procurement and Deployment stages.

Note that, both USFDA-a and USFDA-b yield similar performance across a wide range of standard benchmarks.

We also perform a characteristic comparison of algorithm complexity in terms of the amount of learnable parameters and training time.

In contrast to UAN, the proposed framework offers a much simpler adaptation algorithm devoid of utilization of ad-hoc networks like adversarial discriminator and additional finetuning of the b) Does SSM satisfy the expected inequality?

Effectiveness of the proposed learning algorithm, in case of source-free deployment, relies on the formulation of SSM, which is expected to satisfy Eq. 1.

Fig. 5A shows a histogram of the SSM separately for samples from target-shared (blue) and target-private (red) label space.

The success of this metric is attributed to the generative nature of Procurement stage, which enables the source model to distinguish between the marginally more negative target-private samples as compared to the samples from the shared label space.

c) Sensitivity to hyper-parameters.

As we tackle DA in a source-free setting simultaneously intending to generalize across varied category-gaps, a low sensitivity to hyperparameters would further enhance our practical usability.

To this end, we fix certain hyperparameters for all our ablations (also in Fig. 6C ) even across datasets (i.e. α = 0.2, β = 0.1).

Thus, one can treat them as global-constants with |C n | being the only hyperparameter, as variations in one by fixing the others yield complementary effect on regularization in the Procurement stage.

A thorough analysis reported in the appendix Fig. 8 , clearly demonstrates the low-sensitivity of our model to these hyperparameters.

Figure 6 : Comparison across varied label-set relationships for the task A→D in Office-31 dataset.

A) Visual representation of label-set relationships and T avg at the corresponding instances for B) UAN* (You et al., 2019) and C) ours source-free model.

Effectively, the direction along x-axis (blue horizontal arrow) characterizes increasing Open-set complexity.

The direction along y-axis (red vertical arrow) shows increasing complexity of Partial DA scenario.

The pink diagonal arrow denotes the effect of decreasing shared label space.

the most compelling manner, we propose a tabular form shown in Fig. 6A .

We vary the number of private classes for target and source along x and y axis respectively, with a fixed |C s ∪ C t | = 31.

We compare the T avg metric at the corresponding table instances, shown in Fig. 6B -C.

The results clearly highlight superiority of the proposed framework specifically for the more practical scenarios (close to the diagonal instances) as compared to the unrealistic Closed-set setting (|C s | = |C t | = 0).

e) DA in absence of shared categories.

In universal adaptation, we seek to transfer the knowledge of "class-separability criterion" obtained from the source domain to the deployed target environment.

More concretely, it is attributed to the segregation of data samples based on some expected characteristics, such as classification of objects according to their pose, color, or shape etc.

To quantify this, we consider an extreme case where C s ∩ C t = ∅ (A→D in Office-31 with |C s | = 15, |C t | = 16).

Allowing access to a single labeled target sample from each category in C t = C t , we aim to obtain a one-shot recognition accuracy (assignment of cluster index or class label using the one-shot samples as the cluster center at F t • M (x t )) to quantify the above metric.

We obtain 64.72% accuracy for the proposed framework as compared to 13.43% for UAN* (You et al., 2019) .

This strongly validates our superior knowledge transfer capability as a result of the generative classifier with labeled negative samples complementing for the target-private categories.

f) Dependency on the simulated negative dataset.

Conceding that a combinatorial amount of negative labels can be created, we evaluate the scalability of the proposed approach, by varying the number of negative classes in the Procurement stage by selecting 0, 4, 8, 64, 150 and 190 negative classes as reported in the X-axis of Fig. 5C .

For the case of 0 negative classes, denoted as |C n | * = 0 in Fig. 5C , we synthetically generate random negative features at the intermediate level u, which are at least 3-sigma away from each of the positive source priors P (u s |c i ).

We then make use of these feature samples along with positive image samples, to train a (|C s | + 1) class Procurement model with a single negative class.

The results are reported in Fig. 5C on the A→D task of Office-31 dataset with category relationship inline with the setting in Table 7 .

We observe an acceptable drop in accuracy with decrease in number of negative classes, hence validating scalability of the approach for large-scale classification datasets (such as ImageNet).

Similarly, we also evaluated our framework by combining three or more images to form such negative classes.

An increasing number of negative classes ( |Cs| C 3 > |Cs| C 2 ) attains under-fitting on positive source categories (similar to Fig. 5C , where accuracy reduces beyond a certain limit because of over regularization).

We have introduced a novel source-free, universal domain adaptation framework, acknowledging practical domain adaptation scenarios devoid of any assumption on the source-target label-set relationship.

In the proposed two-stage framework, learning in the Procurement stage is found to be highly crucial, as it aims to exploit the knowledge of class-separability in the most general form with enhanced robustness to out-of-distribution samples.

Besides this, success in the Deployment stage is attributed to the well-designed learning objectives effectively utilizing the source similarity criterion.

This work can be served as a pilot study towards learning efficient inheritable models in future.

In this section, we describe the architecture and the training process used for the Procurement and Deployment stages of our approach.

a) Design of classifier D used in the Procurement stage.

Keeping in mind the possibility of an additional domain shift after performing adaptation (e.g. encountering domain W after performing the adaptation A → D in Office-31 dataset), we design the classifier's architecture in a manner which allows for dynamic modification in the number of negative classes post-procurement.

We achieve this by maintaining two separate classifiers during Procurement -D src , that operates on the positive source classes, and, D neg that operates on the negative source classes (see architecture in Table 5 ).

The final classification score is obtained by computing softmax over the concatenation of logit vectors produced by D src and D neg .

Therefore, the model can be retrained on a different number of negative classes post deployment (using another negative class classifier D neg ), thus preparing it for a subsequent adaptation step to another domain.

b) Negative dataset generation.

We propose two methods to generate negative samples for the Procurement stage, and name the models trained subsequently as USFDA-a and USFDA-b.

Here, we describe the two processes:

n (USFDA-a).

In the presence of domain knowledge (knowledge of the task at hand, i.e. object recognition using images), we generate the negative dataset D n by compositing images taken from different classes, as described in Algo.

2.

We generate random masks using quadratic splines passing through a central image region (lines 3-9).

Using these masks, we merge alternate regions of the images, both horizontally and vertically, resulting in 4 negative images for each pair of images (lines 10-13).

To effectively cover the inter-class negative region, we randomly sample image pairs from D s belonging to different classes, however we do not impose any constraint on how the classes are selected (for e.g. one can composite images from an animal and a non-animal class).

We choose 5000 pairs for tasks on Office-31, Office-Home and VisDA datasets, and 12000 for ImageNet-Caltech.

Since the input source distribution (p) is fixed we first synthesize a negative dataset offline (instead of creating them on the fly) to ensure finiteness of the training set.

The training algorithm for USFDA-a is given in Algo.

1.

horizontal splicing 7: s2 ← − quadratic_interpolation ([(x1, 0), (dx, dy), (x2, 223) ]) vertical splicing 8: m1 ← − mask region below s1 9: m2 ← − mask region to the left of s2 10: Ia ← − m1 * I1 + (1 − m1) * I2 11: Let λ cj and l cj be the maximum eigen value and the corresponding eigen vector of Σ cj , for each class c j 5:ũ r ∼ N (µ, Σ) n (USFDA-b): Here, we perform rejection sampling as given in Algorithm 3.

Here, we obtain a sample from the global source prior P (u s ) = N (u s |µ, Σ), where µ and Σ are the mean and covariance computed at u-space over all the positive source image samples.

We reject the sample if it lies within the 3-sigma bound of any class (i.e. we keep the sample if it is far away from all source class-priors, N (µ ci , Σ ci )), as shown in lines 6 to 11 in Algo.

3.

A sample selected in this fashion is expected to lie in an intermediate region between the source class priors.

The two classes in the vicinity of the sample are then determined by obtaining the two most confident class predictions given by the classifier D src (lines 7 and 8).

Using this pair of classes, we assign a unique negative class label to the sample which corresponds to the intermediate region between the pair of classes.

Note, to learn the arrangement of positive and negative clusters, the feature extractor F s must be trained using negative samples.

We do this by passing the sampled latent-simulated negative instance (ũ r ) through the decoder-encoder pair, (i.e. D • F s • G(ũ r )), and enforcing the cross-entropy loss to classify them into the respective negative class.

The training algorithm for USFDA-b is given in Algo.

4.

c) Justification of L p .

The cross-entropy loss on the likelihoods (referred as L p in the paper) not only enforces intra-class compactness but also ensures inter-class separability in the embedding space, u. Since the negative samples are only an approximation of future target private classes expected to be encountered during deployment, we choose not to employ this loss for them.

Such a training procedure, eventually results in a natural development of bias towards the confident positive source classes.

This subsequently leads to the placement of source clusters in a manner which enables source-free adaptation (See Fig. 4) . (ũr,ỹr) = sample latent-simulated negative instances from D

where ks and kn are the index of ground-truth label ys and yn respectively, and σ is the softmax activation.

7:

; Lv = |vs −vs|; Lu = |ur −ûr| 8:

Update θF s , θD, θG by minimizing LCE, Lv, Lu, and Lp alternatively using separate optimizers.

10:

if (iter % U pdateIter == 0) then 11:

Recompute µc i , Σc i for each source class ci ; Generate D e) Use of multiple optimizers for training.

In the presence of multiple loss terms, we subvert a time-consuming loss-weighting scheme search by making use of multiple Adam optimizers during training.

Essentially, we define a separate optimizer for each loss term, and optimize only one of the losses (chosen in a round robin fashion) in each iteration of training.

We use a learning rate of 0.0001 during training.

Intuitively, the higher order moment parameters in the Adam optimizer adaptively scale the gradients as required by the loss landscape.

f) Label-Set Relationships.

For Office-31 dataset in the UDA setting, we use the 10 classes shared by Office-31 and Caltech-256 as the shared label-set C. These classes are: back_pack, calculator, keyboard, monitor, mouse, mug, bike, laptop_computer, headphones, projector.

From the remaining classes, in alphabetical order, we choose the first 10 classes as source-private (C s ) classes, and the rest 11 as target-private (C t ) classes.

For VisDA, alphabetically, the first 6 classes are considered C, the next 3 as C s and the last 3 comprise C t .

The Office-Home dataset has 65 categories, of which we use the first 10 classes as C, the next 5 for C s , and the rest 50 classes as C t .

The details of the architecture used during the Deployment stage are given in Table 7 .

Note that the Feature Decoder G used during the Procurement stage, is not available during the Deployment stage, restricting complete access to the source data.

Training during the Deployment stage.

The only trainable component is the Feature Extractor F t , which is initialized from F s at Deployment.

Here, the SSM is calculated by passing the target images through the network trained on source data (source model), i.e for each image x t , we calculateŷ = softmax(D • F s • M (x t )).

Note that the softmax is calculated over all |C s | + |C n | classes.

This is done by concatenating the outputs of D src and D neg , and then calculating softmax.

Then, the SSM is determined by the exponential confidence of a target sample, where confidence is the highest softmax value in the categories in |C s |.

We find that widely adopted standard domain adaptation datasets such as Office-31 and VisDA often share a part or all of their label-set with ImageNet.

Therefore, to validate our method's applicability when initialized from a network pretrained on an unrelated dataset, we attempt to solve the adaptation task A→D in Office-31 dataset by pretraining the ResNet-50 backbone on Places dataset (Zhou et al., 2017) .

In Table 3 it can be observed that our method outperforms even source-dependent methods (e.g. UAN (You et al., 2019) , which is also initialized a ResNet-50 backbone pretrained on Places dataset).

In contrast to our method, the algorithm in UAN involves ResNet-50 finetuning.

Therefore, we also compare against a variant of UAN with a frozen backbone network, by inserting an additional feature extractor that operates on the features extracted from ResNet-50 (similar to F s in the proposed method).

The architecture of the feature extractor used for this variant of UAN is outlined in Table 6 .

We observe that our method significantly outperforms this variant of UAN with lesser number of trainable parameters (see Table 3 ).

C.2 SPACE AND TIME COMPLEXITY ANALYSIS.

On account of keeping the weights of the backbone network frozen throughout the training process, and devoid of ad-hoc networks such as adversarial discriminator our method makes use of significantly lesser trainable parameters when compared to previous methods such as UAN (See Table 3 ).

Devoid of adversarial training, the proposed method also has a significantly lesser total training time for adaptation: 44 sec versus 280 sec in UAN (for the A→D task of Office-31 and batch size of 32).

Therefore, the proposed framework offers a much simpler adaptation pipeline, with a superior time and space complexity and at the same time achieves state-of-the-art domain adaptation performance across different datasets, even without accessing labeled source data at the time of adaptation (See Table 3 ).

This corroborates the superiority of our method in real-time deployment scenarios.

In addition to the T avg reported in Fig. 6 in the paper, we also compare the target-unknown accuracy T unk for UAN* and our pipeline.

The results are presented in Figure 7 .

Refer the link to the code provided in the submission for details of the chosen class labels for each adaptation scenario shown in Figure 7 .

Clearly, our method achieves a statistically significant improvement on most of the label-set VisDA, S→ R =20, =20, relationships over UAN.

This demonstrates the capability of our algorithm to detect outlier classes more efficiently than UAN, which can be attributed to the ingeniously developed Procurement stage.

In all our experiments (across datasets as in Tables 1 and 2 and across varied label-set relationships as in Fig. 6 ), we fix the hyperparameters as, α = 0.2, β = 0.1, |C n | = |Cs| C 2 and b +ve /b −ve = 1.

As mentioned in Section 4.3, one can treat these hyperparameters as global constants.

In Fig. 8 we demonstrate the sensitivity of the model to these hyperparameters.

Specifically, in Fig. 8A we show the sensitivity of the adaptation performance, to the choice of |C n | during the Procurement stage, across a spectrum of label-set relationships.

In Fig. 8B we show the sensitivity of the model to α and the batch-size ratio b +ve /b −ve .

Sensitivity to β is shown in Fig. 5 .

Clearly, the model achieves a reasonably low sensitivity to the hyperparameters, even in the challenging source-free scenario.

We additionally evaluate our method in the unsupervised closed set adaptation scenario.

In Table 4 we compare with the closed set domain adaptation methods DAN (Long et al., 2015) , ADDA (Tzeng et al., 2017) , CDAN (Long et al., 2018) and the universal domain adaptation method UAN (You et al., 2019) .

Note that, DAN, ADDA and CDAN rely on the assumption of a shared label space between the source and the target, and hence are not suited for a universal setting.

Furthermore, all other methods require an explicit retraining on the source data during adaptation to perform well, even in the closed-set scenario.

This clearly establishes the superiority of our method in the source-free setting.

We observe in our experiments that the accuracy on the source samples does not drop as a result of the partially generative framework.

For the experiments conducted in Fig. 5C , we observe similar classification accuracy on the source validation set, on increasing the number of negative classes from 0 to 190.

This effect can be attributed to a carefully chosen α = 0.2, which is deliberately biased towards positive source samples to help maintain the discriminative power of the model even in the presence of class imbalance (i.e. |C n | |C s |).

This enhances the model's generative ability without compromising on the discriminative capacity on the positive source samples.

In universal adaptation, we seek to transfer the knowledge of "class separability" obtained from the source domain to the deployed target environment.

More concretely, it is attributed to the segregation of data samples based on an expected characteristics, such as classification of objects according to their pose, color, or shape etc.

To quantify this, we consider an extreme case where C s ∩ C t = ∅ (A→D in Office-31 with |C s | = 15, |C t | = 16).

Considering access to a single labeled target sample from each target category in C t = C t , which are denoted as x cj t , where j = 1, 2, .., |C t |, we perform one-shot Nearest-Neighbour based classification by obtaining the predicted class label asĉ t = argmin cj ||F t • M (x t ) − F t • M (x cj t )|| 2 .

Then, the classification accuracy for the entire target set is computed by comparingĉ t with the corresponding ground-truth category.

We obtain 64.72% accuracy for the proposed framework as compared to 13.43% for UAN* (You et al., 2019) .

A higher accuracy indicates that, the samples are inherently clustered in the intermediate feature level M • F t (x t ) validating an efficient transfer of "class separability" in a fully unsupervised manner.

We obtain a t-SNE plot at the intermediate feature level u for both target and source samples (see Figure 9 ), where the embedding for the target samples is obtained as u t = F t • M (x t ) and the same for the source samples is obtained as u s = F s • M (x s ).

This is because we aim to learn domain-specific features in contrast to domain-agnostic features as a result of the restriction imposed by the source-free scenario ("cannot disturb placement of source clusters").

Firstly we obtain compact clusters for the source-categories as a result of the partially generative Procurement stage.

Secondly, the target-private clusters are placed away from the source-shared and source-private as expected as a result of the carefully formalized SSM weighting scheme in the Deployment stage.

This plot clearly validates our hypothesis.

For both Procurement and Deployment stages, we make use of the machine with the specifications mentioned in Table 8 .

The architecture is developed and trained in Python 2.7 with PyTorch 1.0.0.

<|TLDR|>

@highlight

A novel unsupervised domain adaptation paradigm - performing adaptation without accessing the source data ('source-free') and without any assumption about the source-target category-gap ('universal').
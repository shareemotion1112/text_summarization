Unsupervised domain adaptation has received significant attention in recent years.

Most of existing works tackle the closed-set scenario, assuming that the source and target domains share the exactly same categories.

In practice, nevertheless, a target domain often contains samples of classes unseen in source domain (i.e., unknown class).

The extension of domain adaptation from closed-set to such open-set situation is not trivial since the target samples in unknown class are not expected to align with the source.

In this paper, we address this problem by augmenting the state-of-the-art domain adaptation technique, Self-Ensembling, with category-agnostic clusters in target domain.

Specifically, we present Self-Ensembling with Category-agnostic Clusters (SE-CC) --- a novel architecture that steers domain adaptation with the additional guidance of category-agnostic clusters that are specific to target domain.

These clustering information provides domain-specific visual cues, facilitating the generalization of Self-Ensembling for both closed-set and open-set scenarios.

Technically, clustering is firstly performed over all the unlabeled target samples to obtain the category-agnostic clusters, which reveal the underlying data space structure peculiar to target domain.

A clustering branch is capitalized on to ensure that the learnt representation preserves such underlying structure by matching the estimated assignment distribution over clusters to the inherent cluster distribution for each target sample.

Furthermore, SE-CC enhances the learnt representation with mutual information maximization.

Extensive experiments are conducted on Office and VisDA datasets for both open-set and closed-set domain adaptation, and superior results are reported when comparing to the state-of-the-art approaches.

Convolutional Neural Networks (CNNs) have driven vision technologies to reach new state-ofthe-arts.

The achievements, nevertheless, are on the assumption that large quantities of annotated data are accessible for model training.

The assumption becomes impractical when cost-expensive and labor-intensive manual labeling is required.

An alternative is to recycle off-the-shelf learnt knowledge/models in source domain for new domain(s).

Unfortunately, the performance often drops significantly on a new domain, a phenomenon known as "domain shift." One feasible way to alleviate this problem is to capitalize on unsupervised domain adaptation, which leverages labeled source samples and unlabeled target samples to generalize a target model.

One of the most critical limitations is that most existing models simply align data distributions between source and target domains.

As a consequence, these models are only applicable in closed-set scenario (Figure 1(a) ) under the unrealistic assumption that both domains should share exactly the same set of categories.

This adversely hinders the generalization of these models in open-set scenario to distinguish target samples of unknown class (unseen in source domain) from the target samples of known classes (seen in source domain).

The difficulty of open-set domain adaptation mainly originates from two aspects: 1) how to distinguish the unknown target samples from known ones while classifying the known target samples correctly?

2) how to learn a hybrid network for both closed-set and open-set domain adaptation?

One straightforward way (Figure 1(b) ) to alleviate the first issue is by employing an additional binary classifier for assigning known/unknown label to each target sample Panareda Busto & Gall (2017) .

All the unknown target samples are further taken as outlier and will be discarded during the adaptation from source to target.

As the unknown target samples are holistically grouped as one generic class, the inherent data structure is not fully exploited.

In the case when the distribution of these target samples is diverse or the semantic labels between known and unknown classes are ambiguous, the performance of binary classification is suboptimal.

Instead, we novelly perform clustering over all unlabeled target samples to explicitly model the diverse semantics of both known and unknown classes in target domain, as depicted in Figure 1 (c).

All target samples are firstly decomposed into clusters, and the learnt clusters, though category-agnostic, convey the discriminative knowledge of unknown and known classes specific to target domain.

As such, by further steering domain adaptation with category-agnostic clusters, the learnt representations are expected to be domain-invariant for known classes, and discriminative for unknown and known classes in target domain.

To address the second issue, we remould Self-Ensembling French et al. (2018) with an additional clustering branch to estimate the assignment distribution over all clusters for each target sample, which in turn refines the learnt representations to preserve inherent structure of target domain.

To this end, we present a new Self-Ensembling with Category-agnostic Clusters (SE-CC), as shown in Figure 2 .

Specifically, clustering is firstly implemented to decompose all the target samples into a set of category-agnostic clusters.

The underlying structure of each target sample is thus formulated as its inherent cluster distribution over all clusters, which is initially obtained by utilizing a softmax over the cosine similarities between this sample and each cluster centroid.

With this, an additional clustering branch is integrated into student model of Self-Ensembling to predict the cluster assignment distribution of each target sample.

For each target sample, the KL-divergence is exploited to model the mismatch between its estimated cluster assignment distribution and the inherent cluster distribution.

By minimizing the KL-divergence, the learnt feature is enforced to preserve the underlying data structure in target domain.

Moreover, we uniquely maximize the mutual information among the input intermediate feature map, the output classification distribution and cluster assignment distribution of target sample in student to further enhance the learnt feature representation.

The whole SE-CC framework is jointly optimized.

Unsupervised Domain Adaptation.

One common solution for unsupervised domain adaptation in closed-set scenario is to learn transferrable feature in CNNs by minimizing domain discrepancy through Maximum Mean Discrepancy (MMD) Gretton et al. (2012) .

Tzeng et al. (2014) is one of early works that integrates MMD into CNNs to learn domain invariant representation.

Long et al. (2016) additionally incorporates a residual transfer module into the MMD-based adaptation of classifiers.

Inspired by Goodfellow et al. (2014) , another direction of unsupervised domain adaptation is to encourage domain confusion across different domains via a domain discriminator, which is devised to predict the domain (source/target) of each input sample.

In particular, a domain confusion loss Tzeng et al. (2015) in domain discriminator is devised to enforce the learnt representation to be domain invariant.

Ganin & Lempitsky (2015) formulates domain confusion as a task of binary classification and utilizes a gradient reversal algorithm to optimize domain discriminator.

Open-Set Domain Adaptation.

The task of open-set domain adaptation goes beyond the traditional domain adaptation to tackle a realistic open-set scenario, in which the target domain includes numerous samples from completely new and unknown classes not present in source domain.

Panareda Busto & Gall (2017) is one of the early attempts to tackle the realistic open-set scenario.

Busto et al. additionally exploit the assignments of target samples as know/unknown classes when learning the mapping of known classes from source to target domain.

Later on, Saito et al. (2018b) utilizes adversarial training to learn feature representations that could separate the target samples of unknown class from the known target samples.

Furthermore, Baktashmotlagh et al. (2019) factorizes the source and target data into the shared and private subspace.

The shared subspace models the target and source samples from known classes, while the target samples from unknown class are modeled with a private subspace, tailored to the target domain.

t and x T t , before injected into student and teacher models separately.

Conditional entropy is applied to x S t in student pathway and self-ensembling loss is adopted to align the classification predictions between teacher and student.

To further exploit the underlying data structure of target domain, we perform clustering to decompose the whole unlabeled target samples into a set of category-agnostic clusters (top right), which will be incorporated into Self-Ensembling to facilitate both closed-set and open-set scenarios.

Specifically, an additional clustering branch is integrated into student to infer the assignment distribution over all clusters for each target sample x S t .

By aligning the estimated cluster assignment distribution to the inherent cluster distribution learnt from original clusters via minimizing their KL-divergence, the feature representation is enforced to preserve the underlying data structure in target domain.

Furthermore, the feature representation of student is enhanced by maximizing the mutual information among its feature map, classification and cluster assignment distributions (bottom right).

The maximization is conducted at both global and local levels as detailed in Appendix A.

Summary.

In summary, similar in spirit as previous methods Baktashmotlagh et al. (2019); Panareda Busto & Gall (2017) , SE-CC utilizes unlabeled target samples for learning task-specific classifiers in the open-set scenario.

Different from these approaches, SE-CC leverages categoryagnostic clusters for representation learning.

The learnt feature is driven to preserve the target data structure during domain adaption.

The structure preservation enables effective alignment of sample distributions within known and unknown classes, and discrimination of samples between known and unknown classes.

As a by-product, the preservation, which is represented as a cluster probability distribution, is exploited to further enhance representation learning.

This is achieved through maximizing the mutual information among input feature, its cluster and class probability distributions.

To the best of our knowledge, there is no study yet to fully explore the advantages of category-agnostic clusters for open-set domain adaptation.

In this paper, we remold Self-Ensembling to suit both closed-set and open-set scenarios by integrating category-agnostic clusters into domain adaptation procedure.

An overview of our SelfEnsembling with Category-agnostic Clusters (SE-CC) model is depicted in Figure 2 .

In open-set domain adaptation, we are given the labeled samples X s = {(x s , y s )} in source domain and the unlabeled samples X t = {x t } in target domain belonging to N classes, where y s is the class label of sample x s .

The set of N classes is denoted as C, which consists of N − 1 known classes shared between two domains and an additional unknown class that aggregates all samples of unlabeled classes.

The goal of open-set domain adaptation is to learn the domain-invariant representations and classifiers for recognizing the N − 1 known classes in target domain and meanwhile distinguishing the unknown target samples from known ones.

We first briefly recall the method of Self-Ensembling French et al. (2018) .

Self-Ensembling mainly builds upon the Mean Teacher Tarvainen & Valpola (2017) for semi-supervised learning, which consists of a student model and a teacher model with the same network architecture.

The main idea behind Self-Ensembling is to encourage consistent classification predictions between teacher and student under small perturbations of the input image.

In other words, despite of different augmentations imposed on a target sample, both teacher and student models should predict similar classification probability distribution over all classes.

Specifically, given two perturbed target samples x S t and x T t augmented from an unlabeled sample x t , the self-ensembling loss penalizes the difference between the classification predictions of student and teacher:

where

N denote the predicted classification distribution over N classes via the classification branch in student and teacher.

During training, the student is trained using gradient descent, while the weights of the teacher are directly updated as the exponential moving average of the student weights.

Inspired by Shu et al. (2018) , we additionally adopt the unsupervised conditional entropy loss to train the classification branch in student, aiming to drive the decision boundaries of the classifier far away from high-density regions in target domain.

Accordingly, the overall training loss of our Self-Ensembling is composed of supervised cross entropy loss (L CSE ) on source data, unsupervised self-ensembling loss (L SE ) and conditional entropy loss (L CDE ) of unlabeled target data, balanced with two tradeoff parameters (λ 1 and λ 2 ):

Open-set is more difficult than closed-set domain adaptation because it is required to classify not only inliers but also outliers into N − 1 known and one unknown classes.

The most typical way is by learning a binary classifier to recognize each target sample as known/unkown class.

Nevertheless, such recipe oversimplifies the problem by assuming that all unknown samples belong to one class, while leaving the inherent data distribution among them unexploited.

The robustness of this approach is questionable when the unknown samples span across multiple unknown classes and may not be properly grouped as one generic class.

To alleviate this issue, we perform clustering to explicitly model the diverse semantics in target domain as the distilled category-agnostic clusters, which are further integrated into Self-Ensembling to guide domain adaptation.

Specifically, we design an additional clustering branch in student of Self-Ensembling to align its estimated cluster assignment distribution with the inherent cluster distribution among category-agnostic clusters.

Hence, the learnt feature representations are enforced to be domain-invariant for known classes and meanwhile more discriminative for unknown and known classes in target domain.

Category-agnostic Clusters.

Clustering is an essential data analysis technique for grouping unlabeled data in unsupervised machine learning Jain et al. (1999) .

Here we utilize k-means MacQueen et al. (1967) , the most popular clustering method, to decompose all unlabeled target samples X t into a set of

, where C k represents the set of target samples from the k-th cluster.

Accordingly, the obtained clusters {C k } K k=1 , though category-agnostic, is still able to reveal the underlying structure tailored to target domain, where the target samples with similar semantics stay closer with local discrimination.

In our implementations, we directly represent each target sample x t as the output feature (x t ) of CNNs pre-trained on ImageNet Russakovsky et al. (2015) for clustering.

We also tried to refresh the clusters according to learnt features periodically (e.g., every 5 training epoches), but that did not make a major difference.

We encode the underlying structure of each target sample x t as the joint relations between this sample and all category-agnostic clusters, i.e., the inherent cluster distribution over all clusters.

Specifically, for each target sample x t , we measure its inherent cluster distributionP clu (x t ) ∈ R K through a softmax over the cosine similarities between this sample and each cluster centroid.

The k-th element represents the cosine similarity between x t and the centroid µ k of k-th cluster:

where cos (·) is cosine similarity function and ρ is the temperature parameter of softmax for scaling.

The centroid of each cluster µ k is defined as the average of all samples belonging to that cluster.

Clustering Branch.

An additional branch in student, named as clustering branch, is especially designed to predict the distribution over all category-agnostic clusters for cluster assignment of each target sample x S t .

Concretely, we denote the feature of target sample x S t along student pathway as x S t ∈ R M .

Hence, depending on the input feature x S t , clustering branch infers its cluster assignment distribution P clu (x S t ) ∈ R K over all K clusters via a modified softmax layer Liu et al. (2017) :

where

is the k-th element in P clu representing the probability of assigning target sample x S t into the k-th cluster.

W k is the k-th row of the parameter matrix W ∈ R K×M in the modified softmax layer, which denotes the cluster assignment parameter matrix for the k-th cluster.

KL-divergence Loss.

The clustering branch is trained with the supervision from the inherent cluster distribution of each target sample.

To measure the mismatch between the estimated cluster assignment distribution and the inherent cluster distribution, a KL-divergence loss is defined as

By minimizing the KL-divergence loss, the learnt representation is enforced to preserve the underlying data structure of target domain, pursuing to be more discriminative for both unknown and known classes.

Moreover, we incorporate the inter-cluster relationship into the KL-divergence loss as a constraint to preserve the inherent relations among the cluster assignment parameter matrices.

The spirit behind follows the philosophy that the cluster assignment parameter matrices of two semantically similar clusters should be similar.

Hence, the KL-divergence loss with the constraint of inter-cluster relationships is formulated as

The KL-divergence loss in Eq. (6) is further relaxed as:

Given the input feature of a target sample, the student in our SE-CC produces both classification and cluster assignment distributions via the two parallel branches in a multi-task paradigm.

To further strengthen the learnt target feature in an unsupervised manner, we leverage Mutual Information Maximization (MIM) Hjelm et al. (2019) in student to maximize the mutual information among the input feature and the two output distributions.

The rationale behind follows the philosophy that the global/local mutual information between input feature and output high-level features can be used to tune the feature's suitability for downstream tasks.

As a result, we design a MIM module in student to simultaneously estimate and maximize the local and global mutual information among input feature map, the output classification distribution, and cluster assignment distribution.

Global Mutual Information.

Technically, let x S t ∈ R H×H×D0 be the output feature map of the last convolutional layer in student model for the input target sample x S t (H: the size of height and width; D 0 : the number of channels).

We encode this feature map into a global feature vector G(x

via a convolutional layer (kernel size: 3 × 3; stride size: 1; filter number: D 1 ) plus an average pooling layer.

Next, we concatenate the global feature vector G(x S t ) with the conditioning classification distribution P S cls (x S t ) and cluster assignment distribution P clu (x S t ).

The concatenated feature will be fed into the global Mutual information discriminator for discriminating whether the input global feature vector is aligned with the given classification and cluster assignment distributions.

Here the global Mutual information discriminator is implemented with three stacked fully-connected network plus nonlinear activation.

The final output score of global Mutual information discriminator is

, which represents the probability of discriminating the real input feature with matched classification and cluster assignment distributions.

As such, the global Mutual Information is estimated via Jensen-Shannon MI estimator Nowozin et al. (2016) :

where ϕ (·) is softplus function and G(x S t ) denotes the global feature of a different target imagex S t .

Local Mutual Information.

In addition, we exploit the local Mutual Information among the local input feature at every spatial location, and the output classification and cluster assignment distributions.

In particular, we spatially replicate the two distributions P S cls (x S t ) and P clu (x S t ) to construct H × H × N and H × H × K feature maps respectively, and then concatenate them with the input feature map x S t along the channel dimension.

The concatenated feature map

H×H×(D0+N +K) will be fed into the local Mutual information discriminator for discriminating whether each input local feature is matched with the given classification and cluster assignment distributions.

The local Mutual information discriminator is constructed with three stacked convolutional layer (kernel size: 1 × 1) plus nonlinear activation.

Hence the final output score map of local Mutual information discriminator is

) in score map denotes the probability of discriminating the real input local feature at the i-th spatial location with matched classification and cluster assignment distributions.

As such, the local Mutual Information is estimated as:

Accordingly, the final objective for MIM module is measured as the combination of local and global Mutual Information estimations, balanced with tradeoff parameter α:

Appendix A conceptually depicts the process of both local and global mutual information estimation.

The overall training objective of our SE-CC integrates the cross entropy loss on source data, unsupervised self-ensembling loss, conditional entropy loss, KL-divergence loss of clustering branch in Eq. (7), and the local & global Mutual Information estimation in Eq.(10) on target data:

where λ 3 and λ 4 are tradeoff parameters.

We empirically verify the merit of our SE-CC by conducting experiments on Office Saenko et al. VisDA is a large-scale dataset for the challenging synthetic-real image transfer, consisting of 280k images from three domains.

The synthetic images generated from 3D CAD models are taken as the training domain.

The validation domain contains real images from COCO Lin et al. (2014) and the testing domain includes video frames in YTBB Real et al. (2017) .

Given the fact that the ground truth of testing set are not publicly available, the synthetic images in training domain are taken as source and the COCO images in validation domain are taken as target for evaluation.

In particular, for open-set adaptation, we follow the open-set setting in Peng et al. (2018) and take the 12 classes as the known classes for source & target domains, the 33 background classes as the unknown classes in source, and the other 69 COCO categories as the unknown classes in target.

The known-to-unknown ratio of samples in target domain is strictly set as 1:10.

Three metrics, i.e., Knwn, Mean, and Overall, are adopted for evaluation.

Here Knwn denotes the accuracy averaged over all known classes, Mean is the accuracy averaged over all known & unknown classes, and Overall is the accuracy over all target samples.

For closed-set adaptation, we report the accuracy of all the 12 classes for adaptation, as in the closed-set setting of Peng et al. (2018) .

We utilize ResNet152 as the backbone of CNNs for clustering and adaptation in both closed-set and open-set scenarios.

Open-Set Domain Adaptation on Office.

The performances of different models on Office for open-set adaptation are shown in Table 1 .

It is worth noting that AODA adopts a different open-set setting where unknown source samples are absent.

For fair comparison with AODA, we additionally include a variant of our SE-CC (dubbed as SE-CC ♦ ) which learns classifier without unknown source samples.

Specifically, the classifier in SE-CC ♦ is naturally able to recognize only the N-1 known classes and the target samples will be recognized as unknown if the predicted probability is lower than the threshold for any class as performed in open set SVM Jain et al. (2014) .

Overall, the results across two metrics consistently indicate that our SE-CC obtains better performances against other state-of-the-art closed-set adaptation models (RTN and RevGrad) and open-set adaptation methods (AODA, ATI-λ, and FRODA) on most transfer directions.

Please also note that our SE-CC improves the classification accuracy evidently on the harder transfers, e.g., D → A and W → A, where the two domains are substantially different.

The results generally highlight the key advantage of exploiting underlying target data structure implicit in category-agnostic clusters for open-set domain adaptation.

Such design makes the learnt feature representation to be domaininvariant for known classes while discriminative enough to segregate target samples from known and unknown classes.

Specifically, by aligning the data distributions between source and target domains, RTN and RevGrad exhibit better performance than Source-only that trains classifier only on source data while leaving unlabeled target data unexploited.

By rejecting unknown target samples as outliers and aligning data distributions only for inliers, the open-set adaptation techniques (AODA, ATI-λ, and FRODA) outperform RTN and RevGrad.

This confirms the effectiveness of excluding unknown target samples from the known target samples during domain adaptation in open-set scenario.

Nevertheless, AODA, ATI-λ, and FRODA are still inferior to our SE-CC which steers the domain adaptation by injecting the distribution of category-agnostic clusters as a constraint for feature learning and alignment.

RTN Long et al. (2016) 77 Closed-Set Domain Adaptation on Office and VisDA.

To further verify the generality of our proposed SE-CC, we additionally conduct experiments for domain adaptation in closed-set scenario.

Tables 3 and 4 show the performance comparisons on Office and VisDA datasets for closed-set domain adaptation.

Similar to the observations for open-set domain adaptation task on these two datasets, our SE-CC achieves better performances than other state-of-the-art closed-set adaptation techniques.

The results basically demonstrate the advantage of exploiting the underlying data structure in target domain via category-agnostic clusters, for domain adaptation, even on closed-set scenario without any diverse and ambiguous unknown samples.

Ablation Study.

Here we investigate how each design in our SE-CC influences the overall performance.

Conditional Entropy (CE) incorporates an unsupervised conditional entropy loss into SE to drive the classifier's decision boundaries away from high-density target data regions in student model.

KL-divergence Loss (KL) aligns the estimated cluster assignment distribution to the inherent cluster distribution for each target sample, targeting for refining feature to preserve the underlying structure of target domain.

Mutual Information Maximization (MIM) further enhances the feature's suitability for downstream tasks by maximizing the mutual information among the input feature, the output classification and cluster assignment distributions.

Table 5 details the performance improvements on VisDA by considering different designs and their contributions for open-set domain adaptation in our SE-CC.

CE is a general way to enhance classifier for target domain irrespective of any domain adaptation architectures.

In our case, CE improves the Mean accuracy from 65.2% to 66.3%, which demonstrates that CE is an effective choice.

KL and MIM are two specific designs in our SE-CC and the performance gain of each is 3.0% and 1.2% in Mean metric.

In other words, our SE-CC leads to a large performance boost of 4.2% in total in terms of Mean metric.

The results verify the idea of exploiting underlying target data structure and mutual information maximization for open-set adaptation.

We have presented Self-Ensembling with Category-agnostic Clusters (SE-CC), which exploits the category-agnostic clusters in target domain for domain adaptation in both open-set and closed-set scenarios.

Particularly, we study the problem from the viewpoint of how to separate unknown target samples from known ones and how to learn a hybrid network that nicely integrates category-agnostic clusters into Self-Ensembling.

We initially perform clustering to decompose all target samples into a set of category-agnostic clusters.

Next, an additional clustering branch is integrated into student model to align the estimated cluster assignment distribution to the inherent cluster distribution implicit in category-agnostic clusters.

That enforces the learnt feature to preserve the underlying data structure in target domain.

Moreover, the mutual information among the input feature, the outputs of classification and clustering branches is exploited to further enhance the learnt feature.

Experiments conducted on Office and VisDA for both open-set and closed-set adaptation tasks verify our proposal.

Performance improvements are observed when comparing to state-of-the-art techniques.

In this section, we illustrate the detailed frameworks for global and local mutual information estimation in Figure 3

The implementation of our SE-CC is mainly developed with PyTorch and the network weights are optimized with SGD.

We set the learning rate and mini-batch size as 0.001 and 56 for all experiments.

The maximum training iteration is set as 300 and 25 epochs on Office and VisDA, respectively.

The dimension D 1 of global feature for global Mutual Information estimation is set as 128/1,024 in the backbone of AlexNet/ResNet.

Table 6 details the settings of cluster number K, the tradeoff parameters λ 1 , λ 2 , λ 3 , λ 4 in Eq. (11) and α in Eq.(10) on two datasets for open-set and closed-set adaptation tasks.

In particular, the number of clusters (K) is determined using Gap statistics method.

λ 1 = 10 is fixed based on French et al. (2018) for all the experiments, and the other four parameters are tuned as in Hjelm et al. (2019) ; Shu et al. (2018) .

We restrict the hyper-parameter search for each transfer in range of λ 2 = {10 −2 , 10 −1 , 1}, λ 3 = {10 −2 , 10 −1 , 1}, λ 4 = {10 −4 , 10 −3 , 10 −2 }, and α = {1, 5, 10}.

Evaluation of Clustering Branch.

To study how the design of loss function in clustering branch affects the performance, we compare the use of KL-divergence in our proposed SE-CC with L 1 and L 2 distance.

The results in Table 7 (a) verify that KL-divergence is a better measure of mismatch between the classification and cluster assignment distributions than L 1 and L 2 distance, which yield inferior performance.

Evaluation of Mutual Information Maximization.

Next, we evaluate different variants of MIM module in our SE-CC by estimating mutual information between input feature and different outputs, as shown in Table 7 (b).

CLS, CLU and CLS+CLU estimates the local and global mutual information between input feature and the output of classification branch, the output of clustering branch, and the combined output of two branches, respectively.

Compared to our SE-CC without MIM module (Knwn: 69.3%, Mean: 69.3%, and Overall: 69.1%), CLS and CLU slightly improves the performances by additionally exploiting the mutual information between input feature and the output of each branch.

Furthermore, CLS+CLU obtains a larger performance boost, when combining the outputs from both branches for mutual information estimation.

The results demonstrate the merit of exploiting the mutual information among the input feature and the combined outputs of two downstream tasks (i.e., classification and cluster assignment) in our MIM module.

Figure 5 (a)-(c).

Compared to Source-only without domain adaptation, SE brings the two distributions of source and target closer, leading to domain-invariant representation.

However, in SE, all target samples including unknown samples are enforced to match source samples, making it difficult to recognize unknown target samples with ambiguous semantics.

Through the preservation of underlying target data structure for both known and unknown classes by SE-CC, the unknown target samples are separated from known target samples, and meanwhile the known samples in two domains are indistinguishable.

<|TLDR|>

@highlight

We present a new design, i.e., Self-Ensembling with Category-agnostic Clusters, for both closed-set and open-set domain adaptation.

@highlight

A new approach to open set domain adaptation, where the source domain categories are contained in the target domain categories in order to filter out outlier categories and enable adaptation within the shared classes.
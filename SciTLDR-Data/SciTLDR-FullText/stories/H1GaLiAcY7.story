This paper studies the problem of domain division which aims to segment instances drawn from different probabilistic distributions.

This problem exists in many previous recognition tasks, such as Open Set Learning (OSL) and Generalized Zero-Shot Learning (G-ZSL), where the testing instances come from either seen or unseen/novel classes with different probabilistic distributions.

Previous works only calibrate the conﬁdent prediction of classiﬁers of seen classes (WSVM Scheirer et al. (2014)) or taking unseen classes as outliers Socher et al. (2013).

In contrast, this paper proposes a probabilistic way of directly estimating and ﬁne-tuning the decision boundary between seen and unseen classes.

In particular, we propose a domain division algorithm to split the testing instances into known, unknown and uncertain domains, and then conduct recognition tasks in each domain.

Two statistical tools, namely, bootstrapping and KolmogorovSmirnov (K-S) Test, for the ﬁrst time, are introduced to uncover and ﬁne-tune the decision boundary of each domain.

Critically, the uncertain domain is newly introduced in our framework to adopt those instances whose domain labels cannot be predicted conﬁdently.

Extensive experiments demonstrate that our approach achieved the state-of-the-art performance on OSL and G-ZSL benchmarks.

This paper discusses the problem of learning to separate two domains which include the instances sampled from different distributions.

This is a typical and general research topic that can be potentially used in various recognition tasks, such as Open Set Learning (OSL) and Generalized Zero-Shot Learning (G-ZSL).

Particularly, OSL can break the constraints of the closed set in supervised learning, and aim at recognizing the testing instances from one of the seen classes (i.e., known domain), and the novel class (i.e., unknown domain).

The novel classes include the testing instances which have different distributions from that of the seen ones.

In contrast, G-ZSL targets at distinguishing the labels of instances from the seen and unseen classes.

Only the seen classes have the training instances, but unseen classes do not.

Note that OSL does not explicitly give the class labels for those instances categorized as the novel class, but G-ZSL requires predicting the class labels of unseen classes.

To address G-ZSL, semantic attributes or vectors are introduced as the intermediate representations; each (seen/unseen) class has one semantic prototype that contains class level information.

Specifically, a reasonable solution of OSL and G-ZSL is via dividing the known and unknown domains.

For training classes, the predictors are constructed to map visual features to the class label space (OSL), (or semantic space (G-ZSL)).

Testing is performed on each separated domain to identify seen classes and the novel class (OSL), or both seen and unseen classes (G-ZSL).The key question of OSL and ZSL is how to deal with the newly introduced novel class/unseen classes efficiently in the testing time.

This is different from the conventional Zero-Shot Learning (ZSL) task which assumes that, in the testing stage, seen classes would not be misclassified as unseens, and vice versa; ZSL only uses the unseen classes for testing.

Unfortunately, the predictors learned on training classes will inevitably make OSL or G-ZSL approaches tend to be biased towards the seen classes, and thus leading to very poor classification results for the novel class (OSL) or unseen classes (G-ZSL) BID39 ; .

We show an example in Fig. 1 .

On aPY dataset (described in Sec. 6.1) BID10 ), t-SNE van der Maaten & Hinton (2008 is The initial boundary of the known domain is estimated by bootstrapping.

We can further divide an uncertain domain by K-S Test.

Then we can recognize instances in each domain.

(b) The distribution of pairwise intraclass and interclass distances: We compute the empirical density of the pairwise distance in aPY dataset (described in Sec. 6.1).

There is a large overlapping of the distribution of the intraclass and interclass distances.employed to visualize the distributions of the testing instances of the ResNet-101 features in BID39 (Fig. 1 (a) ), and semantic features learned by SAE Kodirov et al. (2017) (Fig. 1 (b) ).

We categorize the SAE prediction as known or unknown domain labels and compare with the groundtruth in Fig. 1(c) .

We show that a large portion of unseen instances being predicted as one of the known classes.

A natural recipe for addressing this problem is to learn to separate domains by the distributions of instances; and different classifiers can be directly applied in each domain.

However, there are still two key problems.

First, visual features alone are not discriminative enough to help to distinguish the seen and unseen/novel classes.

As Fig. 2 (a) , bicycle and motorbike, respectively, are one of the seen and unseen classes 1 in aPY dataset (described in Sec. 6.1).

We can observe that there is a large overlapping region between their t-SNE visualized feature distributions.

That is, the visual features may not be representative enough to differentiate these two classes; the instances of motorbike (circled as the uncertain domain) may be taken as the bicycle, or vice versa; Second, the predictors trained on seen classes may be not trustworthy.

A not well-trained predictor may negatively affect the recognition algorithms.

Third and even worse, the performance of classifiers in each domain is still very sensitive to the results of domain separation: should the domain of one testing instance be wrongly divided, it would never be correctly categorized by the classifiers.

To tackle the aforementioned issues, our key insight (see Fig. 2(a) ) is to introduce a novel domain -uncertain domain that accounts for the overlapping regions of testing instances from seen or novel/unseen classes.

Thus, the visual or semantic space can be learned to be divided into known, unknown and uncertain domains.

The recognition algorithms will be directly employed in each domain.

Nonetheless, how to divide the domains based on known information is also a non-trivial task.

Though the supervised classifiers can learn the patterns of known classes, not all classes encountered during testing are known.

Formally, we propose exploiting the distribution information of seen and novel/unseen classes to efficiently learn to divide the domains from a probabilistic perspective.

Our domain separation algorithm has two steps: the initial division of domains by bootstrapping, and fine-tuning by the Kolmogorov-Smirnov test.

Specifically, according to extreme value theory BID30 , the maximum/minimum confidence scores predicted by the classifier of each class can be taken as an extreme value distribution.

Since we do not have the prior knowledge of the underlying data distributions of each class; bootstrapping is introduced here as an asymptotically consistent method in estimating an initial boundary of known classes.

Nevertheless, the initial boundary estimated by bootstrapping is too relaxed to include novel testing instances as is illustrated in Fig. 2(b) .

To finetune the boundary, we exploit the K-S Test to validate whether the learned predictors are trustworthy in a specific region.

The uncertain domain introduced thus accounts for those testing instances whose labels are hard to be judged.

Recognition models can be conducted in each domain.

The main contribution is to present a systematic framework of learning to separate domains by probabilistic distributions of instances, which is capable of addressing various recognition tasks, including OSL and G-ZSL.

Towards this goal, two simple, most widely used, and very effective tools -bootstrapping and the Kolmogorov-Smirnov test, are employed to firstly initially estimate and then fine-tune the boundary.

In particular, we introduce an uncertain domain, which encloses the instances which can hardly be classified into known or unknown with high confidence.

We extensively evaluate the importance of domain division on several zero-shot learning benchmarks and achieved significant improvement over existing ZSL approaches.

One-Class Classification (OCC).

It is also known as the unary classification or class-modeling.

The OCC assumes that the training set contains only the positive samples of one specific class.

By learning from such positive instances, OCC aims at identifying the instances belonging to that class.

The common algorithms of OCC include One-class Support Vector Machine (OCSVM) BID26 , Local Outlier Factor (LOF) BID5 .

OCSVM leverages Support Vector Data Description (SVDD) to get a spherical boundary in feature space.

It regularizes the volume of hypersphere so that the effects of outliers can be minimized.

The LOF measures the local deviation of the density of a given instance comparing to other instances, namely locality.

The locality represents the density of the area.

The instances in low-density parts can be taken as outliers.

Note that all OCC algorithms just considered and build a boundary for the positive instances of one class.

Open Set Learning (OSL).

It judges whether the instances belong to known/seen classes BID25 ; BID30 ; BID3 or a novel unknown class.

Both the OCC and OSL are able to divide the instances into known and unknown domains and recognize the known classes from the known domain.

OSL aims at discriminating the instances into seen classes and instances beyond these classes are categorized into a single novel class.

Critically OSL does not have the semantic prototypes of unseen classes to further give the class label of those instances in the novel class.

Both the OCC and OSL are able to divide the instances into known and unknown domains and recognize the known classes in the known domain.

Intrinsically, their key difference lies in whether leveraging the information of different seen classes in building the classifiers.

Specifically, OCC only utilizes the instances of one class to learn its class boundary, whilst OSL can use the instances of different seen classes.

.

ZSL aims at recognizing the novel instances which have never been trained before.

It transfers the knowledge learned from known source classes to recognize the testing instances from unknown target classes.

The knowledge can be formulated as semantic attributes However, ZSL usually assumes that the unseen classes cannot be mis-classified as seen classes and vice versa.

This has greatly simplified the learning task.

Generalized Zero-Shot Learning.

Chao et al. realized that it is nontrivial to directly utilize the existing Zero-Shot Learning algorithms in a more general setting, i.e., G-ZSL.

In such a setting, the testing instances can come from either the seen or unseen classes.

A thorough evaluation of G-ZSL is further conducted in BID39 .

Their results show that the existing ZSL algorithms do not perform well if directly applied to G-ZSL.

The predicted results are inclined to be biased towards seen classes.

3.1 PROBLEM SETUP In learning tasks, we are given the training dataset, i.e., seen classes, of n s instances, DISPLAYFORM0 n is the feature of i th instance with the class label l i ∈ C s , where C s is the source class set; n c s is the number of instances in seen class c. Analogous to standard ZSL setting, we introduce the target label classes C t with C s C t = ∅ and the total class label set C = C s ∪ C t .

y i is the semantic attribute vector of instance x i .

In general, the y i of instances in one class should be the same BID18 .

We simplify y c as the semantic prototype for all the instances in class c. Given one test instance x i , our goal is to predict its class label c i .

We discuss two tasks: (1) Open set recognition: c i ∈ {C s , novel class }; (2) Generalized zero-shot learning: c i ∈ {C s , C t }.

The semantic prototype is predefined for each class in C t .

The 'novel class' is an umbrella term referring to any class not in C s .

We firstly introduce the background of modeling the extreme values (i.e., minimum/maximum scores) computed from one supervised classifier as the extreme value distributions.

In particular, by using each source class c, we can train a binary predictor function, e.g. SVM, z c = f c (x) : R n → R where z is the confidence score of instance x belonging to the class c. In Extreme Value Theory (EVT) , the extreme values (i.e., maximum / minimum confidence scores) of the score distribution computed by the predictor function f c (·) can be modeled by an EVT distribution.

Specifically, for instance set {x} that belong to class c; the minimum score z c min = minf c ({x}) follows the Weibull distribution, DISPLAYFORM0 where DISPLAYFORM1 .

Critically, Eq (1) models the lower boundary distribution of instance confidence scores belonging to class c. On the other hand, for instance set {x} NOT belonging to class c, the maximum score z c max = maxf c ({x}) should follow the reverse Weibull distribution BID27 , DISPLAYFORM2 where rG (·) is the CDF of reverse Weibull distribution: DISPLAYFORM3 Eq (2) models the upper boundary distribution of confidence scores NOT belonging to class c. The scale parameters λ c , λ c , shape parameters κ c , κ c , and location parameters ν c , ν c are estimated by Maximum Likelihood Estimator fitted from the training data.

Critically, Eq. (2) models the upper boundary distribution of instance confidence scores NOT belonging to class c. The distributions of extreme values defined in Eq (1) and Eq (2) can actually give us the boundary of each event above happened in a probabilistic perspective.

Thus we can have the probability

W-SVM introduces a threshold δ c to determine whether the instance i belongs to the class c as, DISPLAYFORM0 where δ c is a fixed value Scheirer et al. (2014) .

The instance x i rejected by all the seen classes by Eq FORMULA7 is labeled as the unknown domain.

Generalizing to C s class is straightforward by training multiple prediction functions {f c (x)}, c = 1, · · · , |C s |.However, there are several key limitations in directly utilizing Eq (4) and Eq (5) of learning the division of domains: (1) Eq (4) directly multiplies two terms and this indicates that there is a potential hypothesis that no correlation exists between E 1 and ¬E 2 , which is generally not the case.

FORMULA3 In multiple seen classes, the instances may derive from many different classes.

It is hard to determine a single fixed δ in Eq (5) for each class.

(3) Furthermore, we give an illustration of the non-negligible overlapping between the intra-class and inter-class distances of each pair of instances on the aPY dataset (described in Sec. 6.1, BID10 ).

As shown in the feature space of Fig. 2 (b) , we compute the pairwise L 2 distances over (1) the instances within the same classes (intra-class), and (2) the instance from different classes (inter-class) on aPY dataset BID10 .

We use the empirical density to show the results in Fig. 2 (c) .

Practically, it is hard to predict the class labels of instances of the overlapped region in the known/unknown domain.

Due to the large overlapped region, the instances (e.g., CMT in BID32 ) whose domains are wrongly labeled will never be correctly categorized.

The W-SVM in Eq (5) and Eq (4) estimates the confidence scores by a fixed threshold empirically per-defined for any data distributions in the known domain.

However, intrinsically, it can be taken as a model selection task in estimating the boundary by Eq (4).

In this paper, we tackle the question of constructing the boundary of the known domain via the bootstrap approach BID8 .

The bootstrapping 2 is a strategy of estimating the standard errors and the confidence intervals of parameters when the underlying distributions are unknown.

Its procedures are closely related to the other methods such as cross-validation, and jackknife.

Bootstrapping is the most widely used tool in approximating the sampling distributions of test statistics and estimators.

To facilitate the discussion, we denote the training set of class c as {x its confidence score f c (x i ), c ∈ C s .

To determine whether the class of a instance x i is seen or unseen, we calculate the statistic m c (x i ) in Eq (4) and Eq (5) with the threshold δ c estimated by the bootstrapping algorithm in Alg.

1.

The instances computed in the known and unknown domain will be categorized by supervised, or zero-shot classifiers respectively.

There are still two difficulties in the above framework.

(1) The whole framework relies on the classifier f c (·), c ∈ C s which is supposed to be robust and well-trained.

However, empirically, we can not always train good classifiers for all classes.

For example, some class has small number of labeled training instances which are insufficient in training the classifier; some outliers may affect the predictor; the hyper-parameters of the classifiers are wrongly tuned.

(2) The naive bootstrapping in Alg.

1 generally provides the bad approximations of the distributions of empirical quantiles in practice BID9 .

Practically, in our tasks, we observe that the estimated δ c may be consistently too relaxed to determine the boundary of the known domain.

We illustrate such a phenomenon in Figure 2(a) : the low-density of seen class bicycle instances (blue points) in the northwest part extends the decision boundary.

The relaxed boundary could inadvertently classify unseen instances (red points) as the false positives.

Unfortunately, in the framework above, once one testing instance in unseen class is wrongly labeled as the known domain, this instance will never be correctly classified.

To address these two problems, we suggest a shrinking step in updating the initial boundary of bootstrapping in the next subsection.

TEST The key idea of updating initial boundary of bootstrapping is to validate whether the learned classifier f c (·), c ∈ C s is trustworthy.

Generally, assume the instances of class c independent and identically distributed and provided training samples sufficient, a ideal classifier f c (·) should produce the similar confidence score distributions of training and testing instances of class c.

The Kolmogorov-Smirnov (K-S) test is an efficient, straightforward, and qualified choice method for comparing distributions Massey Jr (1951); Miller (1956); BID37 .

Remarkably, K-S test is a distribution free test, and the statistics of K-S test is effortless to compute.

We define the null and alternative hypothesis as DISPLAYFORM0 When H 0 is accepted, it indicates that the f c (·) is trustworthy, and the confidence scores of training and testing instances in class c come from the same distribution.

We are certain that a large portion of testing instances {z H 0 is rejected, we are not sure whether f c (·) is well learned; and the class labels of these testing instances are uncertain.

To this end, we introduce a new domain -uncertain domain to include these instances.

Uncertain Domain.

The labels of instances in the uncertain domain should be labeled as the most likely seen class, or one of unseen classes.

Specifically, we can compute the {z c = f c (x)} |Cs| c=1 over all C s classes; and we can obtain, {c , z } = argmax c∈Cs {z c } .The mapping function g (·) is learned on the known domain from features x i to its corresponding semantic attributes y i .

Given one testing instance x i : if z i is very high, we can confidently predict x i belonging to one of seen classes; otherwise, the label of x i is either in the uncertain or unknown domain.

We thus have, DISPLAYFORM1 where y c is semantic prototype of class c; c is the most likely known class to which x i belongs to.

Note that in OSL, we only know the y c (c ∈ C s ) of seen classes; We can dynamically construct a C t set by randomly generating y i by making sure y i − y j > (∀y j ∈ C s ), and = min yi,yj ∈Cs;i =j y i − y j .

The sample size is usually the same with the number of target classes.

We can apply different recognition algorithms in each domain.

In known domain, the standard supervised classifiers can be learned and applied.

In unknown and uncertain domains, we propose a simple yet effective feature prototype embedding recognition algorithm as our plain implement.

Feature prototype embedding.

Once the domain is well separated, we can use the ZSL algorithms to set up the mapping from feature space to semantic/attribute space.

In order to confirm that our main contribution is the domain division part, we do not use very complicated ZSL algorithms.

Only the simplest linear predictor is utilized here to recognize the unseen classes.

Particularly, we use feature prototypes to replace all the instances of each class to avoid the unbalance sample size among classes.

We learn a linear predictor to predict the attribute/word vector g (x) = w T ·

x. The feature prototype embedding is computed as, DISPLAYFORM0 where DISPLAYFORM1 is the feature prototype of class c; y c is the semantic prototype of class c.

When we tackle an instance in the unknown or uncertain domain, we need to embed features into semantic space with g, which can infer the class labels of instances: DISPLAYFORM2 where c is the most likely seen class for x i which is computed by the supervised classifier and y c is the semantic prototype.

Also, the experiment in each domain can be done with ANY other ZSL.

For instance, we report the implement with f-CLSWGAN (f-C) BID40 so that G-ZSL can be done with f-CLSWGAN within a single domain.

Experimental settings.

Our model is validated in OSL and G-ZSL settings.

OSL identifies whether an image belongs to the one of seen classes or the novel class.

G-ZSL gives the class label of testing instances either from seen or unseen classes.

We set the significance level α = 0.05 to tolerate 5% Type-I error.

By default, we use SVM with RBF kernel with parameter cross-validated, unless otherwise specified.

We compare against the competitors, including Attribute Baseline (Attr-B), W-SVM Scheirer et al.(2014), One-class SVM SchÄlkopf et al. FORMULA1 , Binary SVM, OSDN Bendale & Boult (2016) and LOF Breunig et al. (2000) .

The attribute baseline is the variant of our task without using domain division algorithm.

Particularly, the Attr-B uses the same semantic space and embedding as our model, but does not leverage domain division step, i.e., use negative samples and prototypes to identify projected instances directly ( Fig. 1 (c) ).We use the metric -F1-measure, which is defined as the harmonic mean of seen class accuracy (specific class) and unseen prediction accuracy (unnecessary to predict the specific class).

The results are compared in Tab.

1.

Significant performance gain over existing approaches has been observed, in particular for AwA, aPY and ImageNet.

This validates the effectiveness of our framework.

We attribute the improvement to the newly introduced uncertain domain which help better differentiate whether testing instances derive from known or unknown domain.

6.3 RESULTS OF GENERALIZED ZERO-SHOT LEARNING Settings: We first compare the experiments on G-ZSL by using the settings in BID39 .

The results are summarized in Tab.

2.

In particular, we further compare the separate settings; and top-1 accuracy in (%) is reported here: (1) S → T: Test instances from seen classes, the prediction candidates include both seen and unseen classes; (2) U → T: Test instances from unseen classes, the prediction candidates include both seen and unseen classes.

FORMULA4 We employ the harmonic mean as the main evaluation metric to further combine the results of both S → T and U → T, as DISPLAYFORM0 Competitors.

We compare several competitors.(1) DAP Lampert et al. FORMULA1 , trains a probabilistic attribute classifier and utilizes the joint probability to predict labels; (2) ConSE BID22 , maps features into the semantic space by convex combination of attributes; (3) CMT Socher et al. 2 Implement with f-C) Results.

As seen in Tab.

2, our harmonic mean results are significantly better than all the competitors on almost all the datasets.

This shows that ours can effectively address the G-ZSL tasks.

Particularly, DISPLAYFORM1 (1) Our plain results can beat other competitors by a large margin on AwA and aPY dataset, due to the efficacy of our domain division algorithm.

Also, thanks to the power of f-CLSWGAN (f-C), our implement with it on both CUB and AwA dataset are impressive.

FORMULA3 The key advantage of the proposed framework is learning to better divide the testing instances into known, uncertain and unknown domains.

In the known domain, we use the standard SVM classifier.

In unknown/uncertain domains, we directly embed feature prototypes into semantic space and match the most likely class in the candidate pool.

This is the most simple and straightforward recognition method.

Thus our good harmonic mean performance on G-ZSL largely comes from the good domain division algorithm.

Additionally, we also highlight that the other advanced supervised or zero-shot algorithms are orthogonal and potentially be useful in each domain if we want to further improve the performance of G-ZSL.

(3) Our framework is also applied to large-scale datasets in Tab.

3.

We compare several state-of-the-art methods that address G-ZSL on the large-scale dataset.

We use the SVM with the linear kernel on this dataset, due to the large data scale.

Our harmonic mean results surpass the other competitors with a very significant margin.

We notice that other algorithms have very poor performance on U → T. This indicates the intrinsic difficulty of G-ZSL on large-scale dataset.

In contrast, our domain division algorithm can better separate the testing instances into different domains; thus achieving better recognition performance.

Additionally, we found that the prediction of ConSE BID22 is heavily biased towards known classes which is consistent with the results in small datasets.

This is due to the probability of unseen classes are expressed as the convex combination of seen classes.

Usually, there is no higher probability would be assigned to unseen classes than the most probable seen class, especially for large datasets.

DISPLAYFORM2

In the ablation study, we report the F1-measure and Harmonic mean for OSL and G-ZSL respectively with our plain implement.

As is illustrated in Fig. 2 , we notice that although the distance statistic shows the different histogram patterns in feature space, the overlapping part is not negligible.

Importance of bootstrapping the initial threshold.

We introduce a variant A of our framework by replacing bootstrapping step (Sec. 4.1) by using Eq (4) and Eq (5) to fix the threshold (i.e., W-SVM Scheirer et al. FORMULA1 ), i.e., K-S test ( √ ), and Bootstrap (×).

As in Tab.

4, the results of variant A are significantly lower than our framework on all three datasets.

This actually directly validates the importance of determining the initial threshold by bootstrapping.

Improvements of fine-tuning the threshold by K-S test.

We define the variant B is to only use step without fine-tuning the boundary by K-S Test (in Sec. 4.2).

TAB5 directly shows the improvement with/without fine-tuning the threshold, i.e., K-S test (×), and Bootstrap ( √ ).

In particular, we note that variant B has significant lower results on OSL and G-ZSL than variant A and our framework.

One reason is that our bootstrapping step actually learns to determine a very wide boundary of the known domain, to make sure the good results in labeling testing instances as unknown domain samples.

The fine-tuning threshold step will further split the individual known domain into known/uncertain domain by shrinking the threshold.

Without such a fine-tuning step, variant B may wrongly categorize many instances from unseen classes as one of the known classes.

Thus, we can show that the two steps of our framework are very complementary to each other and they work as a whole to enable the good performance on OSL and G-ZSL.

Finally, we introduce the variant C in Tab.

4, by using W-SVM to do OSL, and then use our ZSL model for G-ZSL, i.e., K-S test (×), and Bootstrap (×).

The performance of variant C is again significantly lower than that of ours, and this demonstrates the efficacy of our model.

This paper learns to divide the instances into known, unknown and uncertain domains for the recognition tasks from a probabilistic perspective.

The domain division procedure consists of bootstrapping and K-S Test steps.

The bootstrapping is used to set an initial threshold for each class; we further employ the K-S test to fine-tune the boundary.

Such a domain division algorithm can be used for OSL and G-ZSL tasks, and achieves remarkable results.

@highlight

 This paper studies the problem of domain division by segmenting instances drawn from different probabilistic distributions.  

@highlight

This paper deals with the problem of novelty recognition in open set learning and generalized zero-shot learning and proposes a possible solution

@highlight

An approach to domain separation based on bootstrapping to identify similarity cutoff thresholds for known classes, followed by a Kolmogorov-Smirnoff test to refine the bootstrapped in-distribution zones.

@highlight

Proposes to introduce a new domain, the uncertain domain, to better handle the division between seen/unseen domains in open-set and generalized zero-shot learning
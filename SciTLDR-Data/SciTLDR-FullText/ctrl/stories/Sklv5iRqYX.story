Multi-domain learning (MDL) aims at obtaining a model with minimal average risk across multiple domains.

Our empirical motivation is automated microscopy data, where cultured cells are imaged after being exposed to known and unknown chemical perturbations, and each dataset displays significant experimental bias.

This paper presents a multi-domain adversarial learning approach, MuLANN, to leverage multiple datasets with overlapping but distinct class sets, in a semi-supervised setting.

Our contributions include: i) a bound on the average- and worst-domain risk in MDL, obtained using the H-divergence; ii) a new loss to accommodate semi-supervised multi-domain learning and domain adaptation; iii) the experimental validation of the approach, improving on the state of the art on two standard image benchmarks, and a novel bioimage dataset, Cell.

Advances in technology have enabled large scale dataset generation by life sciences laboratories.

These datasets contain information about overlapping but non-identical known and unknown experimental conditions.

A challenge is how to best leverage information across multiple datasets on the same subject, and to make discoveries that could not have been obtained from any individual dataset alone.

Transfer learning provides a formal framework for addressing this challenge, particularly crucial in cases where data acquisition is expensive and heavily impacted by experimental settings.

One such field is automated microscopy, which can capture thousands of images of cultured cells after exposure to different experimental perturbations (e.g from chemical or genetic sources).

A goal is to classify mechanisms by which perturbations affect cellular processes based on the similarity of cell images.

In principle, it should be possible to tackle microscopy image classification as yet another visual object recognition task.

However, two major challenges arise compared to mainstream visual object recognition problems BID51 .

First, biological images are heavily impacted by experimental choices, such as microscope settings and experimental reagents.

Second, there is no standardized set of labeled perturbations, and datasets often contain labeled examples for a subset of possible classes only.

This has limited microscopy image classification to single datasets and does not leverage the growing number of datasets collected by the life sciences community.

These challenges make it desirable to learn models across many microscopy datasets, that achieve both good robustness w.r.t.

experimental settings and good class coverage, all the while being robust to the fact that datasets contain samples from overlapping but distinct class sets.

Multi-domain learning (MDL) aims to learn a model of minimal risk from datasets drawn from distinct underlying distributions BID20 , and is a particular case of transfer learning BID46 .

As such, it contrasts with the so-called domain adaptation (DA) problem BID7 BID5 BID22 BID46 .

DA aims at learning a model with minimal risk on a distribution called "target" by leveraging other distributions called "sources".

Notably, most DA methods assume that target classes are identical to source classes, or a subset thereof in the case of partial DA BID77 .The expected benefits of MDL, compared to training a separate model on each individual dataset, are two-fold.

First, MDL leverages more (labeled and unlabeled) information, allowing better generalization while accommodating the specifics of each domain BID20 BID72 .

Thus, MDL models have a higher chance of ab initio performing well on a new domain − a problem referred to as domain generalization BID44 or zero-shot domain adaptation BID74 .

Second, MDL enables knowledge transfer between domains: in unsupervised and semi-supervised settings, concepts learned on one domain are applied to another, significantly reducing the need for labeled examples from the latter BID46 .

Learning a single model from samples drawn from n distributions raises the question of available learning guarantees regarding the model error on each distribution.

BID32 introduced the notion of H-divergence to measure the distance between source and target marginal distributions in DA.

BID4 have shown that a finite sample estimate of this divergence can be used to bound the target risk of the learned model.

The contributions of our work are threefold.

First, we extend the DA guarantees to MDL (Sec. 3.1), showing that the risk of the learned model over all considered domains is upper bounded by the oracle risk and the sum of the H-divergences between any two domains.

Furthermore, an upper bound on the classifier imbalance (the difference between the individual domain risk, and the average risk over all domains) is obtained, thus bounding the worst-domain risk.

Second, we propose the approach Multi-domain Learning Adversarial Neural Network (MULANN), which extends Domain Adversarial Neural Networks (DANNs) BID22 to semi-supervised DA and MDL.

Relaxing the DA assumption, MULANN handles the so-called class asymmetry issue (when each domain may contain varying numbers of labeled and unlabeled examples of a subset of all possible classes), through designing a new loss (Sec. 3.2).

Finally, MULANN is empirically validated in both DA and MDL settings (Sec. 4), as it significantly outperforms the state of the art on three standard image benchmarks BID52 BID35 , and a novel bioimage benchmark, CELL, where the state of the art involves extensive domain-dependent pre-processing.

Notation.

Let X denote an input space and Y = {1, . . .

, L} a set of classes.

For i = 1, . . .

, n, dataset S i is an iid sample drawn from distribution D i on X × Y. The marginal distribution of D i on X is denoted by D X i .

Let H be a hypothesis space; for each h in H (h : X → Y) we define the risk under distribution D i as i (h) = P x,y∼Di (h(x) = y).

h i (respectively h ) denotes the oracle hypothesis according to distribution D i (resp.

with minimal total risk over all domains): DISPLAYFORM0 In the semi-supervised setting, the label associated with an instance might be missing.

In the following, "domain" and "distribution" will be used interchangeably, and the "classes of a domain" denote the classes for which labeled or unlabeled examples are available in this domain.

Machine learning classically relies on the iid setting: when training and test samples are independently drawn from the same joint distribution P (X, Y ) BID71 .

Two other settings emerged in the 1990s, "concept drift" and "covariate shift".

They respectively occur when conditional data distributions P (Y |X) and marginal data distributions P (X) change, either continuously or abruptly, across training data or between train and test data BID56 .

Since then, transfer learning has come to designate methods to learn across drifting, shifting or distinct distributions, or even distinct tasks (Pratt et al., 1991; BID46 In MDL, the different domains can be taken into account by maintaining shared and domain-specific parameters BID20 , or through a domain-specific use of shared parameters.

The domaindependent use of these parameters can be learned, e.g. using domain-guided dropout BID72 , or based on prior knowledge about domain semantic relationships BID74 .Early DA approaches leverage source examples to learn on the target domain in various ways, e.g. through reweighting source datapoints BID41 BID26 BID24 , or defining an extended representation to learn from both source and target BID19 .

Other approaches proceed by aligning the source and target representations with PCA-based correlation alignment , or subspace alignment BID21 .

In the field of computer vision, a somewhat related way of mapping examples in one domain onto the other is image-to-image translation, possibly in combination with a generative adversarial network (see references in Appendix A).Intuitively, the difficulty of DA crucially depends on the distance between source and target distribution.

Accordingly, a large set of DA methods proceed by reducing this distance in the original input space X , e.g. via importance sampling BID7 or by modifying the source representation using optimal transport BID17 BID18 .

Another option is to map source and target samples on a latent space where they will have minimal distance.

Neural networks have been intensively exploited to build such latent spaces, either through generative adversarial mechanisms BID67 BID23 , or through combining task objective with an approximation of the distance between source(s) and target.

Examples of used distances include the Maximum Mean Discrepancy due to BID25 BID66 BID10 , some of its variants BID38 , the L 2 contrastive divergence BID43 , the Frobenius norm of the output feature correlation matrices , or the H-divergence BID4 BID22 BID47 BID40 ) (more in Sec. 3).

Most DA methods assume that source(s) and target contain examples from the same classes; in particular, in standard benchmarks such as OFFICE BID52 , all domains contain examples from the same classes.

Notable exceptions are partial DA methods, where target classes are expected to be a subset of source classes e.g. BID77 .

DA and partial DA methods share two drawbacks when applied to semi-supervised MDL with non-identical domain class sets.

First, neither generic nor partial DA methods try to mitigate the impact of unlabeled samples from a class without any labeled counterparts.

Second, as they focus on target performance, (partial) DA methods do not discuss the impact of extra labeled source classes on source accuracy.

However, as shown in Sec. 4.3, class asymmetry can heavily impact model performance if not accounted for.

Bioinformatics is increasingly appreciating the need for domain adaptation methods BID55 BID73 BID68 .

Indeed, experimentalists regularly face the issues of concept drift and covariate shift.

Most biological experiments that last more than a few days are subject to technical variations between groups of samples, referred to as batch effects.

Batch effects in image-based screening data are usually tackled with specific normalization methods BID8 ).

More recently, work by Ando et al. (2017) applied CorAl for this purpose, aligning each batch with the entire experiment.

DA has been applied to image-based datasets for improving or accelerating image segmentation tasks BID3 BID70 BID6 BID30 .

However, to our knowledge, MDL has not yet been used in Bioimage Informatics, and this work is the first to leverage distinct microscopy screening datasets using MDL.

The H-divergence has been introduced to bound the DA risk BID4 BID22 .

This section extends the DA theoretical results to the MDL case (Sec. 3.1), supporting the design of the MULANN approach (Sec. 3.2).

The reader is referred to Appendix B for formal definitions and proofs.

The distance between source and target partly governs the difficulty of DA.

The H-divergence has been introduced to define such a distance which can be empirically estimated with proven guarantees BID2 BID32 .

This divergence measures how well one can discriminate between samples from two marginals.

It inspired an adversarial approach to DA BID22 , through the finding of a feature space in which a binary classification loss between source and target projections is maximal, and thus their H-divergence minimal.

Furthermore, the target risk is upper-bounded by the empirical source risk, the empirical H-divergence between source(s) and target marginals, and the oracle DA risk BID4 BID76 .Bounding the MDL loss using the H-divergence.

A main difference between DA and MDL is that MDL aims to minimize the average risk over all domains while DA aims to minimize the target risk only.

Considering for simplicity a binary classification MDL problem and taking inspiration from BID42 BID5 , the MDL loss can be formulated as an optimal convex combination of domain risks.

A straightforward extension of Ben-David et al. FORMULA0 (Theorem 2 in Appendix B.2) establishes that the compound empirical risk is upper bounded by the sum of: i) the oracle risk on each domain; ii) a statistical learning term involving the VC dimension of H; iii) the divergence among any two domains as measured by their H-divergence and summed oracle risk.

This result states that, assuming a representation in which domains are as indistinguishable as possible and on which every 1-and 2-domain classification task is well addressed, then there exists a model that performs well on all of them.

In the 2-domain case, the bound is minimized when one minimizes the convex combination of losses in the same proportion as samples.

Bounding the worst risk.

The classifier imbalance w.r.t.

the i-th domain is defined as DISPLAYFORM0 The extent to which marginal D i can best be distinguished by a classifier from H (i.e., the Hdivergence), and the intrinsic difficulty i of the i-th classification task, yield an upper-bound on the classifier imbalance (proof in Appendix B.3): Proposition 1.

Given an input space X , n distributions D i over X × {0, 1} and hypothesis class H on X , for any h ∈ H, let i (h) (respectively¯ (h)) denote the classification risk of h w.r.t.

distribution D i (resp.

its average risk over all D i ).

The risk imbalance | i (h) −¯ (h)| is upper bounded as: DISPLAYFORM1 Accordingly, every care taken to minimize H-divergences or ∆ ij (e.g. using the class-wise contrastive losses BID43 ) improves the above upper bound.

An alternative bound of the classifier imbalance can be obtained by using the H∆H-divergence (proposition 3, and corollaries 4, 5 for the 2-domain case in Appendix).

As pointed out by e.g. BID47 , when minimizing the H-divergence between two domains, a negative transfer can occur in the case of class asymmetry, when domains involve distinct sets of classes.

For instance, if a domain has unlabeled samples from a class which is not present in the other domains, both global BID22 and class-wise BID47 ) domain alignments will likely deteriorate at least one of the domain risks by putting the unlabeled samples close to labeled ones from the same domain.

A similar issue arises if a domain has no (labeled or unlabeled) samples in classes which are represented in other domains.

In general, unlabeled samples are only subject to constraints from the domain discriminator, as opposed to labeled samples.

Thus, in the case of class asymmetry, domain alignment will tend to shuffle unlabeled samples more than labeled ones.

This limitation is addressed in MULANN by defining a new discrimination task referred to as Known Unknown Discrimination (KUD).

Let us assume that, in each domain, a fraction p of unlabeled samples comes from extra classes, i.e. classes with no labeled samples within the domain.

KUD aims at discriminating, within each domain, labeled samples from unlabeled ones that most likely belong to such extra classes.

More precisely, unlabeled samples of each domain are ranked according to the entropy of their classification according to the current classifier, restricted to their domain classes.

Introducing the hyper-parameter p, the top p% examples according to this classification entropy are deemed "most likely unknown", and thus discriminated from the labeled ones of the same domain.

The KUD module aims at repulsing the most likely unknown unlabeled samples from the labeled ones within each domain ( FIG1 , thus resisting the contractive effects of global domain alignment.

DISPLAYFORM0 Overall, MULANN involves 3+n interacting modules, where n is the number of domains with unlabeled data.

The first module is the feature extractor with parameters θ f , which maps the input space X to some latent feature space Ω. 2+n modules are defined on Ω: the classifier module, the domain discriminator module, and the n KUD modules, with respective parameters θ c , θ d and (θ u,i ) i .

All modules are simultaneously learned by minimizing loss DISPLAYFORM1 where ζ and λ are hyper-parameters, DISPLAYFORM2 is the domain discrimination loss (multi-class cross-entropy loss of classifying examples from S i in class i), and L i u (θ f , θ u,i ) is the KUD loss (binary cross-entropy loss of discriminating labelled samples from S i from the "most likely unknown" unlabelled samples from S i ).The loss minimization aims to find a saddle point (θ f ,θ y ,θ d ,θ u ), achieving an equilibrium between the classification performance, the discrimination among domains (to be prevented) and the discrimination among labeled and some unlabeled samples within each domain (to be optimized).

The sensitivity w.r.t.

hyperparameter p will be discussed in Sec. 4.3.

This section reports on the experimental validation of MULANN in DA and MDL settings on three image datasets (Sec. 4.2), prior to analyzing MULANN and investigating the impact of class asymmetry on model performances (Sec. 4.3).

Datasets The DA setting considers three benchmarks: DIGITS, including the well-known MNIST and MNIST-M BID35 BID22 ; Synthetic road signs and German traffic sign benchmark BID14 BID60 and OFFICE BID52 .

The MDL setting considers the new CELL benchmark, which is made of fluorescence microscopy images of cells (detailed in Appendix C).

Each image contains tens to hundreds of cells that have been exposed to a given chemical compound, in three domains: California (C), Texas (T) and England (E).

There are 13 classes across the three domains (Appendix FIG5 ; a drug class is a group of compounds targeting a similar known biological process, e.g. DNA replication.

Four domain shifts are considered: C↔T, T↔E, E↔C and C↔T↔E.Baselines and hyperparameters.

In all experiments, MULANN is compared to DANN BID22 and its extension MADA BID47 (that involves one domain discriminator module per class rather than a single global one).

For DANN, MADA and MULANN, the same pre-trained VGG-16 architecture BID59 from Caffe BID28 ) is used for OFFICE and CELL 2 ; the same small convolutional network as BID22 is used for DIGITS (see Appendix D.1 for details).

The models are trained in Torch BID16 using stochastic gradient descent with momentum (ρ = 0.9).

As in BID22 , no hyper-parameter grid-search is performed for OFFICE results -double cross-validation is used for all other benchmarks.

Hyper-parameter ranges can be found in Appendix D.2.Semi-supervised setting.

For OFFICE and CELL, we follow the experimental settings from BID52 .

A fixed number of labeled images per class is used for one of the domains in all cases (20 for Amazon, 8 for DSLR and Webcam, 10 in CELL).

For the other domain, 10 labeled images per class are used for half of the classes (15 for OFFICE, 4 for CELL).

For DIGITS and RoadSigns, all labeled source train data is used, whereas labeled target data is used for half of the classes only (5 for DIGITS, 22 for RoadSigns).

In DA, the evaluation is performed on all target images from the unlabeled classes.

In MDL, the evaluation is performed on all source and target classes (considering labeled and unlabeled samples).Evaluation goals.

A first goal is to assess MULANN performance comparatively to the baselines.

A second goal is to assess how the experimental setting impacts model performance.

As domain discriminator and KUD modules can use both labeled and unlabeled images, a major question regards the impact of seeing unlabeled images during training.

Two experiments are conducted to assess this impact: a) the same unlabeled images are used for training and evaluation (referred to as fully transductive setting, noted FT) ; b) some unlabeled images are used for training, and others for evaluation (referred to as non-fully transductive setting, noted NFT). (The case where no unlabeled images are used during training is discarded due to poor results).

DA on DIGITS, RoadSigns and OFFICE.

BID43 ) (legend CCSA), that uses a contrastive loss to penalizes large (resp.

small) distances between same (resp.

different) classes and different domains in the feature space; Published results from BID65 , an extension of DANN that adds a loss on target softmax values ("soft label loss"; legend Tseng15).

Overall, MULANN yields the best results, significantly improving upon the former best results on the most difficult cases, i.e., D→A, A→D or W→A. As could be expected, the fully transductive results match or significantly outperform the non-fully transductive ones.

Notably, MADA performs similarly to DANN on DIGITS and RoadSigns, but worse on OFFICE; a potential explanation is that MADA is hindered as the number of classes, and thus domain discriminators, increases (respectively 10, 32 and 43 classes).MDL on CELL.

A state of the art method for fluorescence microscopy images relies on tailored approaches for quantifying changes to cell morphology BID31 .

Objects (cells) are segmented in each image, and circa 650 shape, intensity and texture features are extracted for each object in each image.

The profile of each image is defined as the vector of its Kolmogorov-Smirnov statistics, computed for each feature by comparing its distribution to that of the same feature from pooled negative controls of the same plate 3 .

Classification in profile space is realized using linear discriminant analysis, followed by k-nearest neighbor (LDA+k-NN) ("Baseline P" in Table 2 ).

As a state of the art shallow approach to MDL to be applied in profile space, CORAL was chosen ("P + CORAL" in Table 2 ).

A third baseline corresponds to fine-tuning VGG-16 without any transfer loss ("Baseline NN").

Table 2 compares DANN, MADA and MULANN to the baselines, where columns 4-7 (resp.

8-9) consider raw images (resp.

the profile representations).

4 The fact that a profile-based baseline generally outperforms an image-based baseline was expected, as profiles are designed to reduce the impact of experimental settings (column 4 vs. 8).

The fact that standard deviations tend to be larger Table 2 : CELL test classification accuracy results on all domains (average and stdev on 5 folds), in the fully transductive setting (see TAB10 in Appendix for non-transductive ones, and sections C.4, C.5 for details about image and class selection).Shift Image set # classes Baseline NN DANN MADA MULANN Baseline P P+Coral E-C E 7 63.7 (7.0) 62.9 (7.6) 59.5 (9.5) 64.4 (8.0) 74.1 (3.9) 58.4 (6.1) C lab.

here than for OFFICE, RoadSigns or DIGITS is explained by a higher intra-class heterogeneity; some classes comprise images from different compounds with similar but not identical biological activity.

Most interestingly, MULANN and P+CORAL both improve classification accuracy on unlabeled classes at the cost of a slighty worse classification accuracy for the labeled classes (in all cases but one).

This is explained as reducing the divergence between domain marginals on the latent feature space prevents the classifier from exploiting dataset-dependent biases.

Overall, MULANN and P+CORAL attain comparable results on two-domain cases, with MULANN performing significantly better in the three-domain case.

Finally, MULANN matches or significantly outperforms DANN and MADA.

Sensitivity w.r.t.

the fraction p of "known unknowns".

MULANN was designed to counter the negative transfer that is potentially caused by class asymmetry.

This is achieved through the repulsion of labeled examples in each domain from the fraction p of unlabeled examples deemed to belong to extra classes (not represented in the domain).

The sensitivity of MULANN performance to the value of p and its difference to the ground truth p is investigated on MNIST↔MNIST-M. A first remark is that discrepancies between p and p has no influence on the accuracy on a domain without unlabeled datapoints (Fig. 4 in Appendix) .

FIG1 , right, displays the error depending on p for various values of p .

As could have been expected, it is better to underestimate than to overestimate p ; it is even better to slightly underestimate it than to get it right, as the entropy ranking of unlabeled examples can be perturbed by classifier errors.

Impact of class/domain asymmetry.

Section 4.2 reports on the classification accuracy when all classes are represented in all domains of a given shift.

In the general case however, the classes represented by the unlabeled examples are unknown, hence there might exist "orphan" classes, with labeled or unlabeled samples, unique to a single domain.

The impact of such orphan classes, referred to as class asymmetry, is investigated in the 2-domain case.

Four types of samples are considered TAB5 : A class might have labeled examples in both domains (α), labeled in one domain and unlabeled in the other domain (β), labeled in one domain and absent in the other one (orphan γ), and finally unlabeled in one domain and absent in the other one (orphan δ).

The impact of the class asymmetry is displayed on FIG3 , reporting the average classification accuracy of α, β classes on domain 1 on the x-axis, and classification accuracy of unlabeled β classes on domain 2 on the y-axis, for MULANN, DANN and MADA on OFFICE (on CELL in Fig. 5, Appendix) .A clear trend is that adding labeled orphans γ (case "2", FIG3 ) entails a loss of accuracy for all algorithms compared to the no-orphan reference (case "1").

This is explained as follows: on the one hand, the γ samples are subject to the classifier pressure as all labeled samples; on the other hand, they must be shuffled with samples from domain 2 due to the domain discriminator(s) pressure.

Thus, the easiest solution is to shuffle the unlabeled β samples around, and the loss of accuracy on these β samples is very significant (the "2" is lower on the y-axis compared to "1" for all algorithms).

The perturbation is less severe for the labeled (α, β) samples in domain 1, which are preserved by the classifier pressure (x-axis).The results in case "3" are consistent with the above explanation: since the unlabeled δ samples are only seen by the discriminator(s), their addition has little impact on either the labeled or unlabeled data classification accuracy FIG3 .

Finally, there is no clear trend in the impact of both labeled and unlabeled orphans (case "4"): labeled (α, β) (resp.

unlabeled β) are only affected for MADA on CELL (resp.

MULANN on OFFICE).

Overall, these results show that class asymmetry matters for practical applications of transfer learning, and can adversely affect all three adversarial methods FIG3 , with asymmetry in labeled class content ("2") being the most detrimental to model performance.

This paper extends the use of domain adversarial learning to multi-domain learning, establishing how the H-divergence can be used to bound both the risk across all domains and the worst-domain risk (imbalance on a specific domain).

The stress is put on the notion of class asymmetry, that is, when some domains contain labeled or unlabeled examples of classes not present in other domains.

Showing the significant impact of class asymmetry on the state of the art, this paper also introduces MULANN, where a new loss is meant to resist the contractive effects of the adversarial domain discriminator and to repulse (a fraction of) unlabeled examples from labeled ones in each domain.

The merits of the approach are satisfactorily demonstrated by comparison to DANN and MADA on DIGITS, RoadSigns and OFFICE, and results obtained on the real-world CELL problem establish a new baseline for the microscopy image community.

A perspective for further study is to bridge the gap between the proposed loss and importance sampling techniques, iteratively exploiting the latent representation to identify orphan samples and adapt the loss while learning.

Further work will also focus on how to identify and preserve relevant domain-specific behaviours while learning in a domain adversarial setting (e.g., if different cell types have distinct responses to the same class of perturbations).

This work was supported by NIH RO1 CA184984 (LFW), R01GM112690 (SJA) and the Institute of Computational Health Sciences at UCSF (SJA and LFW).

We thank the Shoichet lab (UCSF) for access to their GPUs and Theresa Gebert for suggestions and feedback.

In the field of computer vision, another way of mapping examples in one domain onto the other domain is image-to-image translation.

In the supervised case (the true pairs made of an image and its translation are given), Pic2Pix trains a conditional GAN to discriminate true pairs from fake ones.

In the unsupervised case, another loss is designed to enforce cycle consistency (simultaneously learning the mapping φ from domain A to B, ψ from B to A, and requiring φoψ =Id) BID75 .

Note that translation approaches do not per se address domain adaptation as they are agnostic w.r.t.

the classes.

Additional losses are used to overcome this limitation: Domain transfer network (DTN) BID64 uses an auto-encoder-like loss in the latent space; GenToAdapt BID53 ) uses a classifier loss in the latent space; UNIT BID36 ) uses a VAE loss.

StarGAN BID15 combines image-to-image translation with a GAN, where the discriminator is trained to discriminate true from fake pairs on the one hand, and the domain on the other hand.

ComboGAN BID1 learns two networks per domain, an encoder and a decoder.

DIRT-T BID57 ) uses a conditional GAN and a classifier in the latent space, with two additional losses, respectively enforcing the cluster assumption (the classifier boundary should not cross high density region) and a virtual adversarial training (the hypothesis should be invariant under slight perturbations of the input).

Interestingly, DA and MDL (like deep learning in general) tend to combine quite some losses; two benefits are expected from using a mixture of losses, a smoother optimization landscape and a good stability of the representation BID11 .

Definition.

BID32 BID4 Given a domain X , two distributions D and D over that domain and a binary hypothesis class H on X , the H-divergence between D and D is defined as: DISPLAYFORM0

Theorem 2.

Given an input space X , we consider n distributions D i over X ×{0; 1} and a hypothesis class H on X of VC dimension d. Let α and γ be in the simplex of dimension n. If S is a sample of size m which contains γ i m samples from D i , andĥ is the empirical minimizer of i α iˆ i on (S i ) i , then for any δ > 0, with probability at least 1 − δ, the compound empirical error is upper bounded as: DISPLAYFORM0 with DISPLAYFORM1 A tighter bound can be obtained by BID5 operates on the symmetric difference hypothesis space H∆H. However, divergence H∆H does not lend itself to empirical estimation: even BID5 fall back on H-divergence in their empirical validation.

DISPLAYFORM2

the n-dimensional simplex and h ∈ H, we note α (h) = i α i i (h).We have for α in the simplex of dimension n, h ∈ H and j ∈ {1, . . .

, m}, using the triangle inequality (similarly to the proof of Theorem 4 in BID5 ) DISPLAYFORM0 The last line follows from the definitions of β i,j and H-divergence.

Thus using lemma 6 in (Ben-David et al., 2010) DISPLAYFORM1 Hence the result.

Proof of proposition 1 We have for h ∈ H and j ∈ [1, . . .

, m], using the triangle inequality and the definition of i (similarly to the proof of Theorem 1 in BID4 ) DISPLAYFORM0 We have for i DISPLAYFORM1 The second line follows from the triangle inequality and the definition of the H-divergence.

Thus DISPLAYFORM2 By symmetry we obtain 1 n DISPLAYFORM3 Thus the result.

Proposition 3.

Given a domain X , m distributions D i over X × {0; 1} and a hypothesis class H on X , we have for h ∈ H and j ∈ [1, . . .

, m] DISPLAYFORM4 The second line follows from Lemma 3 from BID5 , and the third from the triangle inequality.

From this and proposition 1 we obtain the result.

Corollaries for the 2-domain case Corollary 4.

Given a domain X , two distributions D S and D T over X × {0, 1} and a hypothesis class H on X , we have for DISPLAYFORM5 Corollary 5.

Given a domain X , two distributions D S and D T over X × {0; 1} and a hypothesis class H on X , we have for DISPLAYFORM6 This dataset is extracted from that published in BID31 .

It contains 455 biologically active images, in 11 classes, on four 384-well plates, in three channels: H2B-CFP, XRCC5-YFP and cytoplasmic-mCherry.

Our analysis used 10 classes: 'Actin', 'Aurora', 'DNA', 'ER', 'HDAC', 'Hsp90', 'MT', 'PLK', 'Proteasome', 'mTOR'.On top of the quality control from the original paper, a visual quality control was implemented to remove images with only apoptotic cells, and XRCC5-YFP channel images were smoothed using a median filter of size 2 using SciPy BID29 .

This dataset is designed to be similar to the Texas domain BID31 , generated using the same cell line, but in a different laboratory, by a different biologist, and using different equipment.

It contains 1,077 biologically active images, in 10 classes, on ten 384-well plates, in three channels: H2B-CFP, XRCC5-YFP and cytoplasmic-mCherry.

The classes are: 'Actin', 'Aurora', 'DNA', 'ER', 'HDAC', 'Hsp90', 'MT', 'PLK', 'Proteasome', 'mTOR'.Cell culture, drug screening and image acquisition Previously BID31 , retroviral transduction of a marker plasmid "pSeg" was used to stably express H2B-CFP and cytoplasmicmCherry tags in A549 human lung adenocarcinoma cells.

A CD-tagging approach BID58 was used to add an N-terminal YFP tag to endogenous XRCC5.Cells were maintained in RPMI1640 media containing 10% FBS, 2 mM glutamine, 50 units/ml penicillin, and 50 µg/ml streptomycin (all from Life Technologies, Inc.), at 37 • C, 5% CO 2 and 100% humidity.

24h prior to drug addition, cells were seeded onto 384-well plate at a density of 1200 cells/well.

Following compound addition, cells were incubated at 37• C for 48 hours.

Images were then acquired using a GE InCell Analyzer 2000.

One image was acquired per well using a 10x objective lens with 2x2 binning.

Image processing Uneven illumination was corrected as described in BID61 .

Background noise was removed using the ImageJ RollingBall plugin BID54 .

Images were segmented, object features extracted and biological activity determined as previously described BID31 .

A visual quality control was implemented to remove images with obvious anomalies (e.g. presence of a hair or out-of-focus image) and images with only apoptotic cells.

YFP-XRCC5 channel images were smoothed using a median filter of size 2.

This dataset was published by BID12 and retrieved from BID37 .

It contains 879 biologically active images of MCF7 breast adenocarcinoma cells, in 15 classes on 55 96-well plates, in 3 channels: Alexa Fluor 488 (Tubulin), Alexa Fluor 568 (Actin) and DAPI (nuclei).

Classes with fewer than 15 images and absent from the other datasets ("Calcium regulation", "Cholesterol", "Epithelial", "MEK", "mTOR") were not used, which leaves 10 classes: 'Actin', 'Aurora', 'DNA', 'ER', 'Eg5 inhibitor', 'HDAC', 'Kinase', 'MT', 'Proteasome', 'Protein synthesis'.Image processing As the images were acquired using a 20X objective, they were stitched using ImageJ plugin BID50 ) and down-scaled 2 times.

Cells thus appear the same size as in the other domains.

Images were segmented, object features extracted and biological activity obtained as previously described BID31 .

A visual quality control was implemented to remove images with obvious anomalies and images with only apoptotic cells.

Images with too few cells were also removed: an Otsu filter BID45 was used to estimate the percentage of pixels containing nuclei in each image, and images with less than 1% nuclear pixels were removed.

Tubulin channel images were smoothed using a median filter of size 2.

Images which were not significantly distinct from negative controls were identified as previously BID31 and excluded from our analysis.

Previous work on the England dataset further focused on images which "clearly [have] one of 12 different primary mechanims of action" BID37 .

We chose not to do so, since it results in a simpler problem (90% accuracy easy to reach) with much less room for improvement.

Images from all domains were down-scaled 4 times and flattened to form RGB images.

Images were normalized by subtracting the intensity values from negative controls (DMSO) of the same plate in each channel.

England, Texas and California share images for cell nucleus and cytoplasm, but their third channel differs: Texas and California shows the protein XRCC5, whereas England shows the Actin protein.

Therefore, the experiments which combine Texas and England, and California and England used only the first two channels, feeding an empty third channel into the network.

Similarly, profiles contain 443 features which are related to the first two channels, and 202 features which are related to the third channel.

Only the former were used in experiments which involve the England dataset.

Shift Dom.

2, labeled classes Domain 2, unlabeled classes E-C HDAC, Proteasome, Actin, Aurora DNA, MT, ER C-T DNA, HDAC, MT, ER, Aurora, mTOR, PLK Actin, Proteasome, Hsp90 T-E DNA, MT, Proteasome, Actin, ER Aurora, HDAC, Actin C-T-E DNA, MT, Proteasome, Actin, ER Aurora, HDAC, Actin BID22 BID66 , a bottleneck fully connected layer is added after the last dense layer of VGG-16.

Learning rates on weights (resp.

biases) from "from scratch" layers is ten (resp, twenty) times that on parameters of fine-tuned layers.

Instance normalization is used on DIGITS, whereas global normalization is used on OFFICE and CELL.

Exponentially increasing, constant ζ 0.1, 0.8 TAB10 : Range of hyper-parameters which were evaluated in cross-validation experiments.

Exponentially decreasing schedule, exponentially increasing schedule, indiv.

lr (learning rates from layers which were trained from scratch are multiplied by 10), as in BID22 ).E ADDITIONAL RESULTS E.1 3-DOMAIN RESULTS ON OFFICE

We use tSNE BID69 to visualize the common feature space in the example of Webcam → Amazon.

FIG3 shows that classes are overall better separated with MULANN.

In particular, when using MULANN, unlabeled examples (blue) are both more grouped and closer to labeled points from the other domain.

E.3 SEMI-SUPERVISED MDL ON THE BIO DATASET

<|TLDR|>

@highlight

Adversarial Domain adaptation and Multi-domain learning: a new loss to handle multi- and single-domain classes in the semi-supervised setting.
Domain adaptation refers to the problem of leveraging labeled data in a source domain to learn an accurate model in a target domain where labels are scarce or unavailable.

A recent approach for finding a common representation of the two domains is via domain adversarial training (Ganin & Lempitsky, 2015), which attempts to induce a feature extractor that matches the source and target feature distributions in some feature space.

However, domain adversarial training faces two critical limitations: 1) if the feature extraction function has high-capacity, then feature distribution matching is a weak constraint, 2) in non-conservative domain adaptation (where no single classifier can perform well in both the source and target domains), training the model to do well on the source domain hurts performance on the target domain.

In this paper, we address these issues through the lens of the cluster assumption, i.e., decision boundaries should not cross high-density data regions.

We propose two novel and related models: 1) the Virtual Adversarial Domain Adaptation (VADA) model, which combines domain adversarial training with a penalty term that punishes the violation the cluster assumption; 2) the Decision-boundary Iterative Refinement Training with a Teacher (DIRT-T) model, which takes the VADA model as initialization and employs natural gradient steps to further minimize the cluster assumption violation.

Extensive empirical results demonstrate that the combination of these two models significantly improve the state-of-the-art performance on the digit, traffic sign, and Wi-Fi recognition domain adaptation benchmarks.

The development of deep neural networks has enabled impressive performance in a wide variety of machine learning tasks.

However, these advancements often rely on the existence of a large amount of labeled training data.

In many cases, direct access to vast quantities of labeled data for the task of interest (the target domain) is either costly or otherwise absent, but labels are readily available for related training sets (the source domain).

A notable example of this scenario occurs when the source domain consists of richly-annotated synthetic or semi-synthetic data, but the target domain consists of unannotated real-world data BID28 Vazquez et al., 2014) .

However, the source data distribution is often dissimilar to the target data distribution, and the resulting significant covariate shift is detrimental to the performance of the source-trained model when applied to the target domain BID27 .Solving the covariate shift problem of this nature is an instance of domain adaptation BID2 .

In this paper, we consider a challenging setting of domain adaptation where 1) we are provided with fully-labeled source samples and completely-unlabeled target samples, and 2) the existence of a classifier in the hypothesis space with low generalization error in both source and target domains is not guaranteed.

Borrowing approximately the terminology from BID2 , we refer to this setting as unsupervised, non-conservative domain adaptation.

We note that this is in contrast to conservative domain adaptation, where we assume our hypothesis space contains a classifier that performs well in both the source and target domains.

To tackle unsupervised domain adaptation, BID9 proposed to constrain the classifier to only rely on domain-invariant features.

This is achieved by training the classifier to perform well on the source domain while minimizing the divergence between features extracted from the source versus target domains.

To achieve divergence minimization, BID9 employ domain adversarial training.

We highlight two issues with this approach: 1) when the feature function has high-capacity and the source-target supports are disjoint, the domain-invariance constraint is potentially very weak (see Section 3), and 2) good generalization on the source domain hurts target performance in the non-conservative setting.

BID24 addressed these issues by replacing domain adversarial training with asymmetric tri-training (ATT), which relies on the assumption that target samples that are labeled by a sourcetrained classifier with high confidence are correctly labeled by the source classifier.

In this paper, we consider an orthogonal assumption: the cluster assumption BID5 , that the input distribution contains separated data clusters and that data samples in the same cluster share the same class label.

This assumption introduces an additional bias where we seek decision boundaries that do not go through high-density regions.

Based on this intuition, we propose two novel models: 1) the Virtual Adversarial Domain Adaptation (VADA) model which incorporates an additional virtual adversarial training BID20 and conditional entropy loss to push the decision boundaries away from the empirical data, and 2) the Decision-boundary Iterative Refinement Training with a Teacher (DIRT-T) model which uses natural gradients to further refine the output of the VADA model while focusing purely on the target domain.

We demonstrate that 1.

In conservative domain adaptation, where the classifier is trained to perform well on the source domain, VADA can be used to further constrain the hypothesis space by penalizing violations of the cluster assumption, thereby improving domain adversarial training.2.

In non-conservative domain adaptation, where we account for the mismatch between the source and target optimal classifiers, DIRT-T allows us to transition from a joint (source and target) classifier (VADA) to a better target domain classifier.

Interestingly, we demonstrate the advantage of natural gradients in DIRT-T refinement steps.

We report results for domain adaptation in digits classification (MNIST-M, MNIST, SYN DIGITS, SVHN), traffic sign classification (SYN SIGNS, GTSRB), general object classification (STL-10, CIFAR-10), and Wi-Fi activity recognition (Yousefi et al., 2017) .

We show that, in nearly all experiments, VADA improves upon previous methods and that DIRT-T improves upon VADA, setting new state-of-the-art performances across a wide range of domain adaptation benchmarks.

In adapting MNIST → SVHN, a very challenging task, we out-perform ATT by over 20%.

Given the extensive literature on domain adaptation, we highlight several works most relevant to our paper.

BID27 ; BID17 proposed to correct for covariate shift by re-weighting the source samples such that the discrepancy between the target distribution and reweighted source distribution is minimized.

Such a procedure is problematic, however, if the source and target distributions do not contain sufficient overlap.

BID13 BID16 ; BID9 proposed to instead project both distributions into some feature space and encourage distribution matching in the feature space.

BID9 in particular encouraged feature matching via domain adversarial training, which corresponds approximately to Jensen-Shannon divergence minimization BID11 .

To better perform nonconservative domain adaptation, BID24 proposed to modify tri-training (Zhou & Li, 2005) for domain adaptation, leveraging the assumption that highly-confident predictions are correct predictions (Zhu, 2005) .

Several of aforementioned methods are based on BID1 's theoretical analysis of domain adaptation, which states the following, Theorem 1 BID1 ) Let H be the hypothesis space and let (X s , s ) and (X t , t ) be the two domains and their corresponding generalization error functions.

Then for any h ∈ H, DISPLAYFORM0 where d H∆H denotes the H∆H-distance between the domains X s and X t , DISPLAYFORM1 Intuitively, d H∆H measures the extent to which small changes to the hypothesis in the source domain can lead to large changes in the target domain.

It is evident that d H∆H relates intimately to the complexity of the hypothesis space and the divergence between the source and target domains.

For infinite-capacity models and domains with disjoint supports, d H∆H is maximal.

A critical component to our paper is the cluster assumption, which states that decision boundaries should not cross high-density regions BID5 .

This assumption has been extensively studied and leveraged for semi-supervised learning, leading to proposals such as conditional entropy minimization BID12 and pseudo-labeling BID15 .

More recently, the cluster assumption has led to many successful deep semi-supervised learning algorithms such as semi-supervised generative adversarial networks , virtual adversarial training BID20 , and self/temporal-ensembling BID14 BID29 .

Given the success of the cluster assumption in semi-supervised learning, it is natural to consider its application to domain adaptation.

Indeed, BID0 formalized the cluster assumption through the lens of probabilistic Lipschitzness and proposed a nearest-neighbors model for domain adaptation.

Our work extends this line of research by showing that the cluster assumption can be applied to deep neural networks to solve complex, high-dimensional domain adaptation problems.

Independently of our work, BID8 demonstrated the application of selfensembling to domain adaptation.

However, our work additionally considers the application of the cluster assumption to non-conservative domain adaptation.

Before describing our model, we first highlight that domain adversarial training may not be sufficient for domain adaptation if the feature extraction function has high-capacity.

Consider a classifier h θ , parameterized by θ, that maps inputs to the (K − 1)-simplex (denote as C), where K is the number of classes.

Suppose the classifier h = g • f can be decomposed as the composite of an embedding function f θ : X → Z and embedding classifier g θ : Z → C. For the source domain, let D s be the joint distribution over input x and one-hot label y and let X s be the marginal input distribution.

DISPLAYFORM0 where the supremum ranges over discriminators D : Z → (0, 1).

Then L y is the cross-entropy objective and D is a domain discriminator.

Domain adversarial training minimizes the objective DISPLAYFORM1 where λ d is a weighting factor.

Minimization of L d encourages the learning of a feature extractor f for which the Jensen-Shannon divergence between f (X s ) and f (X t ) is small.

2 BID9 suggest that successful adaptation tends to occur when the source generalization error and feature divergence are both small.

It is easy, however, to construct situations where this suggestion fails.

In particular, if f has infinitecapacity and the source-target supports are disjoint, then f can employ arbitrary transformations to the target domain so as to match the source feature distribution (see Appendix E for formalization).We verify empirically that, for sufficiently deep layers, jointly achieving small source generalization error and feature divergence does not imply high accuracy on the target task TAB6 .

Given the limitations of domain adversarial training, we wish to identify additional constraints that one can place on the model to achieve better, more reliable domain adaptation.

In this paper, we apply the cluster assumption to domain adaptation.

The cluster assumption states that the input distribution X contains clusters and that points in the same cluster come from the same class.

This assumption has been extensively studied and applied successfully to a wide range of classification tasks (see Section 2).

If the cluster assumption holds, the optimal decision boundaries should occur far away from data-dense regions in the space of X BID5 .

Following BID12 , we achieve this behavior via minimization of the conditional entropy with respect to the target distribution,

Intuitively, minimizing the conditional entropy forces the classifier to be confident on the unlabeled target data, thus driving the classifier's decision boundaries away from the target data BID12 .

In practice, the conditional entropy must be empirically estimated using the available data.

However, BID12 note that this approximation breaks down if the classifier h is not locally-Lipschitz.

Without the locally-Lipschitz constraint, the classifier is allowed to abruptly change its prediction in the vicinity of the training data points, which 1) results in a unreliable empirical estimate of conditional entropy and 2) allows placement of the classifier decision boundaries close to the training samples even when the empirical conditional entropy is minimized.

To prevent this, we propose to explicitly incorporate the locally-Lipschitz constraint via virtual adversarial training BID20 and add to the objective function the additional term DISPLAYFORM0 which enforces classifier consistency within the norm-ball neighborhood of each sample x. Note that virtual adversarial training can be applied with respect to either the target or source distributions.

We can combine the conditional entropy minimization objective and domain adversarial training to yield min.

DISPLAYFORM1 a basic combination of domain adversarial training and semi-supervised training objectives.

We refer to this as the Virtual Adversarial Domain Adaptation (VADA) model.

Empirically, we observed that the hyperparameters (λ d , λ s , λ t ) are easy to choose and work well across multiple tasks (Appendix B).H∆H-Distance Minimization.

VADA aligns well with the theory of domain adaptation provided in Theorem 1.

Let the loss, DISPLAYFORM2 denote the degree to which the target-side cluster assumption is violated.

Modulating λ t enables VADA to trade-off between hypotheses with low target-side cluster assumption violation and hypotheses with low source-side generalization error.

Setting λ t > 0 allows rejection of hypotheses with high target-side cluster assumption violation.

By rejecting such hypotheses from the hypothesis space H, VADA reduces d H∆H and yields a tighter bound on the target generalization error.

We verify empirically that VADA achieves significant improvements over existing models on multiple domain adaptation benchmarks (Table 1) .

VADA DIRT-T In non-conservative domain adaptation, we assume the following inequality, DISPLAYFORM0 where ( s , t ) are generalization error functions for the source and target domains.

This means that, for a given hypothesis class H, the optimal classifier in the source domain does not coincide with the optimal classifier in the target domain.

We assume that the optimality gap in Eq. FORMULA0 results from violation of the cluster assumption.

In other words, we suppose that any source-optimal classifier drawn from our hypothesis space necessarily violates the cluster assumption in the target domain.

Insofar as VADA is trained on the source domain, we hypothesize that a better hypothesis is achievable by introducing a secondary training phase that solely minimizes the target-side cluster assumption violation.

Under this assumption, the natural solution is to initialize with the VADA model and then further minimize the cluster assumption violation in the target domain.

In particular, we first use VADA to learn an initial classifier h θ0 .

Next, we incrementally push the classifier's decision boundaries away from data-dense regions by minimizing the target-side cluster assumption violation loss L t in Eq. (9).

We denote this procedure Decision-boundary Iterative Refinement Training (DIRT).

Stochastic gradient descent minimizes the loss L t by selecting gradient steps ∆θ according to the following objective, min.

DISPLAYFORM0 DISPLAYFORM1 which defines the neighborhood in the parameter space.

This notion of neighborhood is sensitive to the parameterization of the model; depending on the parameterization, a seemingly small step ∆θ may result in a vastly different classifier.

This contradicts our intention of incrementally and locally pushing the decision boundaries to a local conditional entropy minimum, which requires that the decision boundaries of h θ+∆θ stay close to that of h θ .

It is therefore important to define a neighborhood that is parameterization-invariant.

Following BID21 , we instead select ∆θ using the following objective, min.

DISPLAYFORM2 Each optimization step now solves for a gradient step ∆θ that minimizes the conditional entropy, subject to the constraint that the Kullback-Leibler divergence between h θ (x) and h θ+∆θ (x) is small for x ∼ X t .

The corresponding Lagrangian suggests that one can instead minimize a sequence of optimization problems min.

DISPLAYFORM3 that approximates the application of a series of natural gradient steps.

In practice, each of the optimization problems in Eq. FORMULA0 can be solved approximately via a finite number of stochastic gradient descent steps.

We denote the number of steps taken to be the refinement interval B. Similar to BID29 , we use the Adam Optimizer with Polyak averaging BID22 .

We interpret h θn−1 as a (sub-optimal) teacher for the student model h θn , which is trained to stay close to the teacher model while seeking to reduce the cluster assumption violation.

As a result, we denote this model as Decision-boundary Iterative Refinement Training with a Teacher (DIRT-T).Weakly-Supervised Learning.

This sequence of optimization problems has a natural interpretation that exposes a connection to weakly-supervised learning.

In each optimization problem, the teacher model h θn−1 pseudo-labels the target samples with noisy labels.

Rather than naively training the student model h θn on the noisy labels, the additional training signal L t allows the student model to place its decision boundaries further from the data.

If the clustering assumption holds and the initial noisy labels are sufficiently similar to the true labels, conditional entropy minimization can improve the placement of the decision boundaries BID23 .

Adaptation.

An alternative interpretation is that DIRT-T is the recursive extension of VADA, where the act of pseudo-labeling of the target distribution constructs a new "source" domain (i.e. target distribution X t with pseudo-labels).

The sequence of optimization problems can then be seen as a sequence of non-conservative domain adaptation problems in which DISPLAYFORM0 is the true conditional label distribution in the target domain.

Since d H∆H is strictly zero in this sequence of optimization problems, domain adversarial training is no longer necessary.

Furthermore, if L t minimization does improve the student classifier, then the gap in Eq. (10) should get smaller each time the source domain is updated.

In principle, our method can be applied to any domain adaptation tasks so long as one can define a reasonable notion of neighborhood for virtual adversarial training BID19 .

For comparison against BID24 and BID8 , we focus on visual domain adaptation and evaluate on MNIST, MNIST-M, Street View House Numbers (SVHN), Synthetic Digits (SYN DIGITS), Synthetic Traffic Signs (SYN SIGNS), the German Traffic Signs Recognition Benchmark (GTSRB), CIFAR-10, and STL-10.

For non-visual domain adaptation, we evaluate on Wi-Fi activity recognition.

Architecture We use a small CNN for the digits, traffic sign, and Wi-Fi domain adaptation experiments, and a larger CNN for domain adaptation between CIFAR-10 and STL-10.

Both architectures are available in Appendix A. For fair comparison, we additionally report the performance of source-only baseline models and demonstrate that the significant improvements are attributable to our proposed method.

Replacing gradient reversal.

In contrast to BID9 , which proposed to implement domain adversarial training via gradient reversal, we follow BID11 and instead optimize via alternating updates to the discriminator and encoder (see Appendix C).Instance normalization.

We explored the application of instance normalization as an image preprocessing step.

This procedure makes the classifier invariant to channel-wide shifts and rescaling of pixel intensities.

A discussion of instance normalization for domain adaptation is provided in Appendix D. We show in Figure 3 the effect of applying instance normalization to the input image.

Figure 3: Effect of applying instance normalization to the input image.

In clockwise direction: MNIST-M, GTSRB, SVHN, and CIFAR-10.

In each quadrant, the top row is the original image, and the bottom row is the instance-normalized image.

Hyperparameters.

For each task, we tuned the four hyperparameters (λ d , λ s , λ t , β) by randomly selecting 1000 labeled target samples from the training set and using that as our validation set.

We observed that extensive hyperparameter-tuning is not necessary to achieve state-of-the-art performance.

In all experiments with instance-normalized inputs, we restrict our hyperparameter search for each task to λ d = {0, 10 −2 }, λ s = {0, 1}, λ t = {10 −2 , 10 −1 }.

We fixed β = 10 −2 .

Note that the decision to turn (λ d , λ s ) on or off that can often be determined a priori.

A complete list of the hyperparameters is provided in Appendix B. Table 1 : Test set accuracy on visual domain adaptation benchmarks.

In all settings, both VADA and DIRT-T achieve state-of-the-art performance in all settings.

MNIST → MNIST-M. We first evaluate the adaptation from MNIST to MNIST-M. MNIST-M is constructed by blending MNIST digits with random color patches from the BSDS500 dataset.

MNIST ↔ SVHN.

The distribution shift is exacerbated when adapting between MNIST and SVHN.

Whereas MNIST consists of black-and-white handwritten digits, SVHN consists of crops of colored, street house numbers.

Because MNIST has a significantly lower intrinsic dimensionality that SVHN, the adaptation from MNIST → SVHN is especially challenging when the input is not pre-processed via instance normalization.

When instance normalization is applied, we achieve a strong state-ofthe-art performance 76.5% and an equally impressive margin-of-improvement over source-only of 35.6%.

Interestingly, by reducing the refinement interval B and taking noisier natural gradient steps, we were occasionally able to achieve accuracies as high as 87%.

However, due to the high-variance associated with this, we omit reporting this configuration in Table 1 .SYN DIGITS → SVHN.

The adaptation from SYN DIGITS → SVHN reflect a common adaptation problem of transferring from synthetic images to real images.

The SYN DIGITS dataset consist of 500000 images generated from Windows fonts by varying the text, positioning, orientation, background, stroke color, and the amount of blur.

SYN SIGNS → GTSRB.

This setting provides an additional demonstration of adapting from synthetic images to real images.

Unlike SYN DIGITS → SVHN, SYN SIGNS → GTSRB contains 43 classes instead of 10.STL ↔ CIFAR.

Both STL-10 and CIFAR-10 are 10-class image datasets.

These two datasets contain nine overlapping classes.

Following the procedure in BID8 , we removed the non-overlapping classes ("frog" and "monkey") and reduce to a 9-class classification problem.

We achieve state-of-the-art performance in both adaptation directions.

In STL → CIFAR, we achieve a 11.7% margin-of-improvement and a performance accuracy of 73.3%.

Note that because STL-10 contains a very small training set, it is difficult to estimate the conditional entropy, thus making DIRT-T unreliable for CIFAR → STL.

Wi-Fi Activity Recognition.

To evaluate the performance of our models on a non-visual domain adaptation task, we applied VADA and DIRT-T to the Wi-Fi Activity Recognition Dataset (Yousefi et al., 2017) .

The Wi-Fi Activity Recognition Dataset is a classification task that takes the WiFi Channel State Information (CSI) data stream as input x to predict motion activity within an indoor area as output y. Domain adaptation is necessary when the training and testing data are collected from different rooms, which we denote as Rooms A and B. TAB2 shows that VADA significantly improves classification accuracy compared to Source-Only and DANN by 17.3% and 15% respectively.

However, DIRT-T does not lead to further improvements on this dataset.

We perform experiments in Appendix F which suggests that VADA already achieves strong clustering in the target domain for this dataset, and therefore DIRT-T is not expected to yield further performance improvement.

Table 3 : Additional comparison of the margin of improvement computed by taking the reported performance of each model and subtracting the reported source-only performance in the respective papers.

W.I.N.I. indicates "with instance-normalized input."Overall.

We achieve state-of-the-art results across all tasks.

For a fairer comparison against ATT and the Π-model, Table 3 provides the improvement margin over the respective source-only performance reported in each paper.

In four of the tasks (MNIST → MNIST-M, SVHN → MNIST, MNIST → SVHN, STL → CIFAR), we achieve substantial margin of improvement compared to previous models.

In the remaining three tasks, our improvement margin over the source-only model is competitive against previous models.

Our closest competitor is the Π-model.

However, unlike the Π-model, we do not perform data augmentation.

It is worth noting that DIRT-T consistently improves upon VADA.

Since DIRT-T operates by incrementally pushing the decision boundaries away from the target domain data, it relies heavily on the cluster assumption.

DIRT-T's empirical success therefore demonstrates the effectiveness of leveraging the cluster assumption in unsupervised domain adaptation with deep neural networks.6.3 ANALYSIS OF VADA AND DIRT-T

To study the relative contribution of the virtual adversarial training in the VADA and DIRT-T objectives (Eq. FORMULA6 and Eq. (14) respectively), we perform an extensive ablation analysis in Table 4 .

The removal of the virtual adversarial training component is denoted by the "no-vat" subscript.

Our results show that VADA no-vat is sufficient for out-performing DANN in all but one task.

The further ability for DIRT-T no-vat to improve upon VADA no-vat demonstrates the effectiveness of conditional entropy minimization.

Ultimately, in six of the seven tasks, both virtual adversarial training and conditional entropy minimization are essential for achieving the best performance.

The empirical importance of incorporating virtual adversarial training shows that the locally-Lipschitz constraint is beneficial for pushing the classifier decision boundaries away from data.

Table 4 : Test set accuracy in ablation experiments, starting from the DANN model.

The "no-vat" subscript denote models where the virtual adversarial training component is removed.

When considering Eq. FORMULA0 , it is natural to ask whether defining the neighborhood with respect to the classifier is truly necessary.

In FIG2 , we demonstrate in SVHN → MNIST and STL → CIFAR that removal of the KL-term negatively impacts the model.

Since the MNIST data manifold is low-dimensional and contains easily identifiable clusters, applying naive gradient descent (Eq. FORMULA0 ) can also boost the test accuracy during initial training.

However, without the KL constraint, the classifier can sometimes deviate significantly from the neighborhood of the previous classifier, and the resulting spikes in the KL-term correspond to sharp drops in target test accuracy.

In STL → CIFAR, where the data manifold is much more complex and contains less obvious clusters, naive gradient descent causes immediate decline in the target test accuracy.

We further analyze the behavior of VADA and DIRT-T by showing T-SNE embeddings of the last hidden layer of the model trained to adapt from MNIST → SVHN.

In FIG3 , source-only training shows strong clustering of the MNIST samples (blue) and performs poorly on SVHN (red).

VADA offers significant improvement and exhibits signs of clustering on SVHN.

DIRT-T begins with the VADA initialization and further enhances the clustering, resulting in the best performance on MNIST → SVHN.

In TAB6 , we applied domain adversarial training to various layers of a Domain Adversarial Neural Network BID9 trained to adapt MNIST → SVHN.

With the exception of layers L − 2 and L − 0, which experienced training instability, the general observation is that as the layer gets deeper, the additional capacity of the corresponding embedding function allows better matching of the source and target distributions without hurting source generalization accuracy.

This demonstrates that the combination of low divergence and high source accuracy does not imply better adaptation to the target domain.

Interestingly, when the classifier is regularized to be locally-Lipschitz via VADA, the combination of low divergence and high source accuracy appears to correlate more strongly with better adaptation.

In this paper, we presented two novel models for domain adaptation inspired by the cluster assumption.

Our first model, VADA, performs domain adversarial training with an added term that penalizes violations of the cluster assumption.

Our second model, DIRT-T, is an extension of VADA that recursively refines the VADA classifier by untethering the model from the source training signal and applying approximate natural gradients to further minimize the cluster assumption violation.

Our experiments demonstrate the effectiveness of the cluster assumption: VADA achieves strong performance across several domain adaptation benchmarks, and DIRT-T further improves VADA performance.

Our proposed models open up several possibilities for future work.

One possibility is to apply DIRT-T to weakly supervised learning; another is to improve the natural gradient approximation via K-FAC BID18 and PPO BID25 .

Given the strong performance of our models, we also recommend them for other downstream domain adaptation applications.

DISPLAYFORM0 Gaussian noise, σ = 1 DISPLAYFORM1 Gaussian noise, σ = 1 DISPLAYFORM2

We observed that extensive hyperparameter-tuning is not necessary to achieve state-of-the-art performance.

To demonstrate this, we restrict our hyperparameter search for each task to λ d = {0, 10 −2 }, λ s = {0, 1}, λ t = {10 −2 , 10 −1 }, in all experiments with instance-normalized inputs.

We fixed β = 10 −2 .

Note that the decision to turn (λ d , λ s ) on or off that can often be determined a priori based on prior belief regarding the extent to covariate shift.

In the absence of such prior belief, a reliable choice is (λ d = 10 −2 , λ s = 1, λ t = 10 −2 , β = 10 −2 ).

When the target domain is MNIST/MNIST-M, the task is sufficiently simple that we only allocate B = 500 iterations to each optimization problem in Eq. (14) .

In all other cases, we set the refinement interval B = 5000.

We apply Adam Optimizer (learning rate = 0.001, β 1 = 0.5, β 2 = 0.999) with Polyak averaging (more accurately, we apply an exponential moving average with momentum = 0.998 to the parameter trajectory).

VADA was trained for 80000 iterations and DIRT-T takes VADA as initialization and was trained for {20000, 40000, 60000, 80000} iterations, with number of iterations chosen as hyperparameter.

We note from BID11 that the gradient of ∇ θ ln(1 − D(f θ (x))) is tends to have smaller norm than −∇ θ ln D(f θ (x)) during initial training since the latter rescales the gradient by 1/D(f θ (x)).

Following this observation, we replace the gradient reversal procedure with alternating minimization of DISPLAYFORM0 The choice of using gradient reversal versus alternating minimization reflects a difference in choice of approximating the mini-max using saturating versus non-saturating optimization BID7 .

In some of our initial experiments, we observed the replacement of gradient reversal with alternating minimization stabilizes domain adversarial training.

However, we encourage practitioners to try either optimization strategy when applying VADA.

Theorem 1 suggests that we should identify ways of constraining the hypothesis space without hurting the global optimal classifier for the joint task.

We propose to further constrain our model by introducing instance normalization as an image pre-processing step for the input data.

Instance normalization was proposed for style transfer BID30 and applies the operation DISPLAYFORM0 where x (i) ∈ R H×W ×C denotes the i th sample with (H, W, C) corresponding to the height, width, and channel dimensions, and where µ, σ : R H×W ×C → R C are functions that compute the mean and standard deviation across the spatial dimensions.

A notable property of instance normalization is that it is invariant to channel-wide scaling and shifting of the input elements.

Formally, consider scaling and shift variables γ, β ∈ R C .

If γ 0 and σ( DISPLAYFORM1 For visual data the application of instance normalization to the input layer makes the classifier invariant to channel-wide shifts and scaling of the pixel intensities.

For most visual tasks, sensitivity to channel-wide pixel intensity changes is not critical to the success of the classifier.

As such, instance normalization of the input may help reduce d H∆H without hurting the globally optimal classifier.

Interestingly, Figure 3 shows that input instance normalization is not equivalent to gray-scaling, since color is partially preserved.

To test the effect of instance normalization, we report results both with and without the use of instance-normalized inputs.

We denote the source and target distributions respectively as p s (x, y) and p t (x, y).

Let the source covariate distribution p s (x) define the random variable X s that have support supp(X s ) = X s and let (X t , X t ) be analogously defined for the target domain.

Both X s and X t are subsets of R n .

Let p s (y) and p t (y) define probabilities over the support Y = {1, . . .

, K}. We consider any embedding function f : R n → R m , where R m is the embedding space, and any embedding classifier g : R m → C, where C is the (K − 1)-simplex.

We denote a classifier h = g • f has the composite of an embedding function with an embedding classifier.

For simplicity, we restrict our analysis to the simple case where K = 2, i.e. where Y = {0, 1}. Furthermore, we assume that for any δ ∈ [0, 1], there exists a subset Ω ⊆ R n where p s (x ∈ Ω) = δ.

We impose a similar condition on p t (x).For a joint distribution p(x, y), we denote the generalization error of a classifier as DISPLAYFORM0 Note that for a given classifier h : R n → [0, 1], the corresponding hard classifier is k(x) = 1{h(x) > 0.5}. We further define the set Ω ⊆ R n such that DISPLAYFORM1 In a slight abuse of notation, we define the generalization error (Ω) with respect to Ω as DISPLAYFORM2 An optimal Ω * p is a partitioning of DISPLAYFORM3 such that generalization error under the distribution p(x, y) is minimized.

E.1 GOOD TARGET-DOMAIN ACCURACY IS NOT GUARANTEED Domain adversarial training seeks to find a single classifier h used for both the source p s and target p t distributions.

To do so, domain adversarial training sets up the objective DISPLAYFORM4 DISPLAYFORM5 where F and G are the hypothesis spaces for the embedding function and embedding classifier.

Intuitively, domain adversarial training operates under the hypothesis that good source generalization error in conjunction with source-target feature matching implies good target generalization error.

We shall see, however, that if X s ∩ X t = ∅ and F is sufficiently complex, this implication does not necessarily hold.

Let F contain all functions mapping R n → R m , i.e. F has infinite capacity.

Suppose G contains the function g(z) = 1{z = 1 m } and X s ∩ X t = ∅. We consider the set DISPLAYFORM6 Such a set of classifiers satisfies the feature-matching constraint while achieving source generalization error no worse than the optimal source-domain hard classifier.

It suffices to show that H * includes hypotheses that perform poorly in the target domain.

We first show H * is not an empty set by constructing an element of this set.

Choose a partitioning Ω where DISPLAYFORM7 Consider the embedding function DISPLAYFORM8 Let g(z) = 1{z = 1 m }.

It follows that the composite classifier DISPLAYFORM9 Next, we show that a classifier h ∈ H * does not necessarily achieve good target generalization error.

Consider the partitioningΩ which solves the following optimization problem DISPLAYFORM10 DISPLAYFORM11 Such a partitioningΩ is the worst-case partitioning subject to the probability mass constraint.

It follows that worse case h ∈ H * has generalization error pt (h ) = max h∈H * pt (h) ≥ pt (hΩ).To provide intuition that pt (h ) is potentially very large, consider hypothetical source and target domains where X s ∩ X t = ∅ and p t (x ∈ Ω * pt ) = p s (x ∈ Ω * ps ) = 0.5.

The worst-case partitioning subject to the probability mass constraint is simplyΩ = R n \ Ω Define the embedding functions DISPLAYFORM12 f (x) = 1 m if (x ∈ X s ∩ Ω s ) ∨ (x ∈ X t ∩ (R n \ Ω t )) 0 m otherwise.

Let g (z) = g(z) = 1{z = 1 m }.

It follows that the composite classifiers h = g • f and h = g • f are elements ofH.From the definition of d H∆H , we see that DISPLAYFORM13 TheH∆H-divergence thus achieves the maximum value of 2.

Our analysis assumes infinite capacity embedding functions and the ability to solve optimization problems exactly.

The empirical success of domain adversarial training suggests that the use of finite-capacity convolutional neural networks combined with stochastic gradient-based optimization provides the necessary regularization for domain adversarial training to work.

The theoretical characterization of domain adversarial training in the case finite-capacity convolutional neural networks and gradient-based learning remains a challenging but important open research problem.

To evaluate the performance of our models on a non-visual domain adaptation task, we applied VADA and DIRT-T to the Wi-Fi Activity Recognition Dataset (Yousefi et al., 2017) .

The Wi-Fi Activity Recognition Dataset is a classification task that takes the Wi-Fi Channel State Information (CSI) data stream as input x to predict motion activity within an indoor area as output y. The dataset collected the CSI data stream samples associated with seven activities, denoted as "bed", "fall", "walk", "pick up", "run", "sit down", and "stand up".However, the joint distribution over the CSI data stream and motion activity changes depending on the room in which the data was collected.

Since the data was collected for multiple rooms, we selected two rooms (denoted here as Room A and Room B) and constructed the unsupervised domain adaptation task by using Room A as the source domain and Room B as the target domain.

We compare the performance of DANN, VADA, and DIRT-T on the Wi-Fi domain adaptation task in TAB2 , using the hyperparameters (λ d = 0, λ s = 0, λ t = 10 −2 , β = 10 −2 ).

TAB2 shows that VADA significantly improves classification accuracy compared to Source-Only and DANN.

However, DIRT-T does not lead to further improvements on this dataset.

We believe this is attributable to VADA successfully pushing the decision boundary away from data-dense regions in the target domain.

As a result, further application of DIRT-T would not lead to better decision boundaries.

To validate this hypothesis, we visualize the t-SNE embeddings for VADA and DIRT-T in FIG4 and show that VADA is already capable of yielding strong clustering in the target domain.

To verify that the decision boundary indeed did not change significantly, we additionally provide the confusion matrix between the VADA and DIRT-T predictions in the target domain (Fig. 7) .

@highlight

SOTA on unsupervised domain adaptation by leveraging the cluster assumption.
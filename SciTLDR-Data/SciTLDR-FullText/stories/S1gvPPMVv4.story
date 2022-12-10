Domain adaptation addresses the common problem when the target distribution generating our test data drifts from the source (training) distribution.

While absent assumptions, domain adaptation is impossible, strict conditions, e.g. covariate or label shift, enable principled algorithms.

Recently-proposed domain-adversarial approaches consist of aligning source and target encodings, often motivating this approach as minimizing two (of three) terms in a theoretical bound on target error.

Unfortunately, this minimization can cause arbitrary increases in the third term, e.g. they can break down under shifting label distributions.

We propose asymmetrically-relaxed distribution alignment, a new approach that overcomes some limitations of standard domain-adversarial algorithms.

Moreover, we characterize precise assumptions under which our algorithm is theoretically principled and demonstrate empirical benefits on both synthetic and real datasets.

Despite breakthroughs in supervised deep learning across a variety of challenging tasks, current techniques depend precariously on the i.i.d.

assumption.

Unfortunately, real-world settings often demand not just generalization to unseen examples but robustness under a variety of shocks to the data distribution.

Ideally, our models would leverage unlabeled test data, adapting in real time to produce improved predictions.

Unsupervised domain adaptation formalizes this problem as learning a classifier from labeled source domain data and unlabeled data from a target domain, to maximize performance on the target distribution.

Without further assumptions, guarantees of target-domain accuracy are impossible BID3 .

However, well-chosen assumptions can make possible algorithms with non-vacuous performance guarantees.

For example, under the covariate shift assumption BID7 BID11 , although the input marginals can vary between source and target (p S (x) = p T (x)), the conditional distribution of the labels (given features) exhibits invariance across domains (p S (y|x) = p T (y|x)).

Traditional approaches to the covariate shift problem require the source distributions' support to cover the target support, estimating adapted classifiers via importanceweighted risk minimization BID11 BID8 BID6 BID13 BID9 .Problematically, assumptions of contained support are violated in practice.

A recent sequence of deep learning papers have proposed empirically-justified adversarial training schemes aimed at practical problems with non-overlapping supports BID5 BID12 .

Example problems include generalizing from gray-scale images to colored images or product images on white backgrounds to photos of products in natural settings.

While importance-weighting solutions are useless here (with non-overlapping support, weights are unbounded), domain-adversarial networks BID5 and subsequently-proposed variants report strong empirical results on a variety of image recognition challenges.

The key idea of domain-adversarial networks is to simultaneously minimize the source error and align the two distributions in representation space.

The scheme consists of an encoder, a label classifier, and a domain classifier.

During training, the domain classifier is optimized to predict each image's domain given its encoding.

The label classifier is optimized to predict labels from encodings (for source images).

The encoder weights are optimized for the twin objectives of accurate label classification (of source data) and fooling the domain classifier (for all data).Although BID5 motivate their idea via theoretical results due to BID2 , the theory is insufficient to justify their method.

Put simply, BID2 bound the test error by a sum of three terms.

The domain-adversarial objective minimizes two among these, but this minimization may cause the third term to increase.

This is guaranteed to happen when the label distribution shifts between source and target ( FIG0 ).In this paper, we propose asymmetrically-relaxed distribution alignment, a relaxed distance for aligning data across domains that can be minimized without requiring latent-space distributions to match exactly.

The new distance is minimized whenever the density ratios in representation space from target to source are upper bounded by a certain constant, such that the target representation support is contained in the source representation's.

The relaxed distribution alignment need not lead to a poor classifier on the target domain under label distribution mismatch FIG0 ).

We demonstrate theoretically that the relaxed alignment is sufficient for a good target domain performance under a concrete set of assumptions on the data distributions.

Further, we propose several practical ways to achieve the relaxed distribution alignment, translating the new distance into adversarial learning objectives.

Empirical results on synthetic and real datasets show that incorporating our relaxed distribution alignment loss into adversarial domain adaptation gives better classification performance on the target domain.

Due to space constraints, we only briefly state our results in the main text and append the full version of our paper after references.

Unsupervised domain adaptation with representations For simplicity, we address the binary classification scenario.

Let X be the input space and f : X → {0, 1} be the (domain-invariant) ground truth labeling function.

Let p S and p T be the input distributions over X for source and target domain respectively.

Let Z be a latent space and Φ denote a class of mappings from X to Z. Define H to be a class of predictors over the latent space Z, i.e., each h ∈ H maps from Z to {0, 1}. Given a representation mapping φ ∈ Φ, classifier h ∈ H, and input x ∈ X , our prediction is h(φ(x)).

In the unsupervised domain adaptation setting, we have access to labeled source data (x, f (x)) for x ∼ p S and unlabeled target data x ∼ p T .

We are interested in bounding the classification risk of a (φ, h)-pair on the target domain: DISPLAYFORM0 where r is the risk function in the latent space.

Domain-adversarial learning Domain-adversarial approaches focus on minimizing the first and third term in (1) jointly.

Informally, these approaches minimize the source domain classification risk and the distance between the two distributions in the latent space: DISPLAYFORM1 DISPLAYFORM2 This problem happens because although D(p φ S , p φ T ) = 0 is a sufficient condition for the third term of (1) to be zero, it is not a necessary condition.

We now examine the third term of FORMULA0 : DISPLAYFORM3 ).

This expression shows that if the source error E S (φ, h) is zero then it is sufficient to say the third term of FORMULA0 is zero when the density ratio p φ T (z)/p φ S (z) is upper bounded by some constant for all z, as shown in FIG0 .

Given this motivation, we propose relaxing from exact distribution matching to bounding the density ratio in the domain-adversarial learning objective (2).

We call this asymmetrically-relaxed distribution alignment.

More specifically, our proposed approach is to replace the typical distribution distance D in the domain-adversarial objective (2) with a β-admissible distance D β so that minimizing the new objective does not necessarily lead to a failure under label distribution shift.

We bound the target domain error under our proposed asymmetrically-relaxed distribution alignment.

Our theoretical result makes distinct contribution to the domain adaptation literature: We provide a risk bound that explains the behavior of domain-adversarial methods with model-independent assumptions on data distributions.

Existing theories without assumptions of contained support (BenDavid et al., 2007; BID2 BID0 BID10 BID4 do not exhibit this property.

Construction 3.1.

The following statements hold simultaneously: (1) (Lipschitzness of representa- DISPLAYFORM0 Assumption 3.2.

(Connectedness from target domain to source domain.)

Given constants (L, β, ∆, δ 1 , δ 2 , δ 3 ), assume that, for any B S , B T ⊂ X with p S (B S ) ≥ 1 − δ 2 and p T (B T ) ≥ 1 − δ 1 − (1 + β)δ 2 , there exists C T ⊂ B T that satisfies the following conditions: (1) For any x ∈ C T , there exists x ∈ C T ∩ B S such that one can find a sequence of points x 0 , x 1 , ..., x m ∈ C T with DISPLAYFORM1 Given a L-Lipschitz mapping φ ∈ Φ and a binary classifier h ∈ H, if φ satisfies the properties in Construction 3.1 with constants (L, β, ∆, δ 1 , δ 2 ), and Assumption 3.2 holds with the same set of constants plus δ 3 , then the target domain error can be bounded as DISPLAYFORM2

In this section, we derive several β-admissible distance metrics that can be practically minimized with adversarial training.f -divergence We propose a general approach to make any f -divergence β-admissible by partially linearizing the function f .

Plugging in the corresponding f for JS-divergence gives DISPLAYFORM0 Wasserstein distance The idea behind modifying the Wasserstein distance is to model the optimal transport from p to the region where distributions have 1 + β maximal density ratio with respect to q. Following the dual-form derivation for the original Wasserstein distance gives DISPLAYFORM1 DISPLAYFORM2 Reweighting distance Given any distance metric D, a generic way to make it β-admissible is to allow reweighting for one of the distances within a β-dependent range: Given a distribution q over Z and a reweighting function w : DISPLAYFORM3 .

Define W β,q to be a set of β-qualified reweighting with respect to q: DISPLAYFORM4 .

Then the relaxed distance can be defined as DISPLAYFORM5

To evaluate our approach, we implement Domain Adversarial Neural Networks (DANN), BID5 replacing the JS-divergence with several β-admissible distances.

Table 3 : Classification accuracy on target domain with/without label distribution shift on USPS-MNIST.

@highlight

Instead of strict distribution alignments in traditional deep domain adaptation objectives, which fails when target label distribution shifts, we propose to optimize a relaxed objective with new analysis, new algorithms, and experimental validation.

@highlight

This paper suggests relaxed metrics for domain adaptation which give new theoretical bounds on the target error.
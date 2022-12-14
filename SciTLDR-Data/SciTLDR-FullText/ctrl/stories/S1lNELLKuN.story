The goal of unpaired cross-domain translation is to learn useful mappings between two domains, given unpaired sets of datapoints from these domains.

While this formulation is highly underconstrained, recent work has shown that it is possible to learn mappings useful for downstream tasks by encouraging approximate cycle consistency in the mappings between the two domains [Zhu et al., 2017].

In this work, we propose AlignFlow, a framework for unpaired cross-domain translation that ensures exact cycle consistency in the learned mappings.

Our framework uses a normalizing flow model to specify a single invertible mapping between the two domains.

In contrast to prior works in cycle-consistent translations, we can learn AlignFlow via adversarial training, maximum likelihood estimation, or a hybrid of the two methods.

Theoretically, we derive consistency results for AlignFlow which guarantee recovery of desirable mappings under suitable assumptions.

Empirically, AlignFlow demonstrates significant improvements over relevant baselines on image-to-image translation and unsupervised domain adaptation tasks on benchmark datasets.

Given data from two domains, cross-domain translation refers to the task of learning a mapping from one domain to another, such as translating text across two languages or image colorization.

This ability to learn a meaningful alignment between two domains has a broad range of applications across machine learning, including relational learning BID1 , domain adaptation BID2 BID4 , image and video translation for computer vision BID6 , and machine translation for natural language processing BID7 .Broadly, there are two learning paradigms for cross-domain translation: paired and unpaired.

In paired cross-domain translation, we assume access to pairs of datapoints across the two domains, e.g., black and white images and their respective colorizations.

However, paired data can be expensive to obtain or may not even exist, as in neural style transfer BID8 where the goal is to translate across the works of two artists that typically do not exhibit a direct correspondence.

Unpaired cross-domain translation tackles this regime where paired data is not available and learns an alignment between two domains given only unpaired sets of datapoints from the domains.

Formally, we seek to learn a joint distribution over two domains, say A and B, given samples only from the marginal distributions over A and B. CycleGAN BID0 , a highly successful approach to this problem, learns a pair of conditional generative models, say G A???B and G B???A , to match the marginal distributions over A and B via an adversarial objective BID9 .

The marginal matching constraints alone are insufficient to learn the desired joint distribution, both in theory and practice.

To further constrain the problem, an additional desideratum is imposed in the form of cycle-consistency.

That is, given any datapoint A = a, the cycle-consistency term in the learning objective prefers mappings G A???B and G B???A such that G B???A (G A???B (a)) ??? a. Symmetrically, cycle-consistency in the reverse direction implies G A???B (G B???A (b))

??? b for all datapoints B = b. Intuitively, this encourages the learning of approximately bijective mappings.

While empirically effective, the CycleGAN objective only imposes a soft cycle-consistency penalty and provides no guarantee that G A???B and G B???A are true inverses of each other.

A natural question, then, is whether the cycle-consistency objective can be replaced with a single, invertible model G A???B .

Drawing inspiration from the literature on invertible generative models (Rezende and BID10 BID11 BID13 , we propose AlignFlow, a learning framework for cross-domain translations which uses normalizing flow models to represent the mappings.

In AlignFlow, we compose a pair of invertible flow models G Z???A and G Z???B , to represent the mapping G A???B = G Z???B ??? G ???1 Z???A .

Here, Z is a shared latent space between the two domains.

Since composition of invertible mappings preserves invertibility, the mapping G A???B is invertible and the reverse mapping from B ??? A is simply given as G B???A = G ???1 A???B .

Hence, AlignFlow guarantees exact cycle-consistency by design and simplifies the standard CycleGAN learning objective by learning a single, invertible mapping.

Furthermore, AlignFlow provides flexibility in specifying the training objective.

In addition to adversarial training, we can also specify a prior distribution over the latent variables Z and train the two component models G Z???B and G Z???A via maximum likelihood estimation (MLE).

MLE is statistically efficient, exhibits stable training dynamics, and can have a regularizing effect when used in conjunction with adversarial training of invertible generative models BID14 .

In this section, we discuss the necessary background and notation on generative adversarial networks, normalizing flows, and cross-domain translations using CycleGANs.

Unless explicitly stated otherwise, we assume probability distributions admit absolutely continuous densities on a suitable reference measure.

We use uppercase notation X, Y, Z to denote random variables, and lowercase notation x, y, z to denote specific values in the italicized corresponding sample spaces X , Y, Z.

A generative adversarial network (GAN) is a latent variable model which specifies a deterministic mapping h : Z ??? X between a set of latent variables Z and a set of observed variables X BID9 .

In order to sample from GANs, we need a prior density over Z that permits efficient sampling.

A GAN generator can also be conditional, where the conditioning is on another set of observed variables (and optionally the latent variables Z as before) BID15 .A GAN is trained via adversarial training, wherein the generator h plays a minimax game with an auxiliary critic C. The goal of the critic C : X ??? R is to distinguish real samples from the observed dataset with samples generated via h. The generator, on the other hand, tries to generate samples that can maximally confuse the critic.

Many learning objectives have been proposed for adversarial training, including those based on f-divergences BID16 , Wasserstein Distance BID17 , and maximum mean discrepancy BID18 .

The generator and the critic are both parameterized by deep neural networks and learned via alternating gradient-based optimization.

Because adversarial training only requires samples from the generative model, it can be used to train generative models with intractable or ill-defined likelihoods BID19 .

In practice, such likelihood-free methods give excellent performance on sampling-based tasks unlike the alternative maximum likelihood estimation-based training criteria for learning generative models.

However, these models are harder to train due to the alternating minimax optimization and suffer from issues such as mode collapse BID20 .

Normalizing flows represent a latent variable generative model that specifies an invertible mapping h : Z ??? X between a set of latent variables Z and a set of observed variables X. Let p X and p Z denote the marginal densities defined by the model over X and Z respectively.

Using the change-of-variables formula, the marginal densities can be related as: DISPLAYFORM0 where z = h ???1 (x) due to the invertibility constraints.

Here, the second term on the RHS corresponds to the absolute value of the determinant of the Jacobian of the inverse transformation and signifies the shrinkage/expansion in volume when translating across the two sample spaces.

For evaluating likelihoods via the change-of-variables formula, we require efficient and tractable evaluation of the prior density, the inverse transformation h ???1 , and the determinant of its Jacobian of h ???1 .

To draw a sample from this model, we perform ancestral sampling, i.e., we first sample a latent vector z ??? p Z (z) and obtain the sampled vector as given by x = h(z).

This requires the ability to efficiently: (1) sample from the prior density and (2) evaluate the forward transformation h. Many transformations parameterized by deep neural networks that satisfy one or more of these criteria have been proposed in the recent literature on normalizing flows, e.g., NICE BID11 and Autoregressive Flows BID22 .

By suitable design of transformations, both likelihood evaluation and sampling can be performed efficiently, as in Real-NVP BID12 .

Consequently, a flow model can be trained efficiently via maximum likelihood estimation as well as likelihood-free adversarial training BID14 .

Consider two multi-variate random variables A and B with domains specified as A ??? R n and B ??? R n respectively.

Let p * A,B denote the joint distribution over these two variables.

In the unpaired cross-domain translation setting, we are given access to a finite datasets D A and and D B , sampled independently from the two unknown corresponding (marginal) data distributions p * A and p * B respectively.

Using these datasets, the goal is to learn the conditional distributions p * A|B and p * B|A .

Without any paired data, the problem is underconstrained (even in the limit of infinite paired data) since the conditionals can only be derived from p * A,B , but we only have data sampled from the marginal densities.

To address this issue, CycleGAN introduced additional constraints that have proven to be empirically effective in learning mappings that are useful for downstream tasks.

We now proceed by describing the CycleGAN framework.

If we assume the conditional distributions for A|B and B|A are deterministic, the conditionals can alternatively be represented as cross-domain mappings G A???B : A ??? B and G B???A : B ??? A. A CycleGAN uses a pair of conditional GANs to translate data from two domains BID0 .

It consists of the following components:1.

A conditional GAN G A???B : A ??? B that takes as input data from domain A and maps it to domain B. The mapping G A???B is learned adversarially with the help of a critic C B : B ??? R trained to distinguish between real and synthetic data (generated via G A???B ) from domain B.2.

Symmetrically, a conditional GAN G B???A : B ??? A and a critic C A : A ??? R for adversarial learning of the reverse mapping from B to A.Any suitable GAN loss can be substituted in the above objective, e.g., Wasserstein GAN BID17 .

For the standard cross-entropy based GAN loss, the critic outputs a probability of a datapoint being real and optimizes the following objective: DISPLAYFORM0 Additionally, semantically meaningful mappings can be learned via a pair of conditional GANs G A???B and G B???A that are encouraged to be cycle consistent.

Cycle consistency encourages the data translated from domain A to B via G A???B to be mapped back to the original datapoints in A via G B???A .

That is, G B???A (G A???B (a)) ??? a for all a ??? A. Formally, the cycle-consistency loss for translation from A to B and back is defined as: DISPLAYFORM1 Symmetrically, an additional cycle consistency term The full objective optimized by a CycleGAN is given as: DISPLAYFORM2 DISPLAYFORM3 where ?? A???B and ?? B???A are hyperparameters controlling the relative strength of the cycle consistent terms.

The objective is minimized w.r.

The use of cycle consistency has indeed been shown empirically to be a good inductive bias for learning cross-domain translations.

However, it necessitates a careful design of the loss function that could involve a trade-off between the adversarial training and cycle consistency terms in the objective in Eq. 4.

To stabilize training and achieve good empirical performance, BID0 proposes a range of techniques such as the use of an identity loss in the above objective.

In this section, we present the AlignFlow framework for learning cross-domain translations between two domains A and B. We will first discuss the model representation, followed by the learning and inference procedures for AlignFlow.

Finally, we will present a theoretical result analyzing the proposed framework.

We will use a graphical model to represent the relationships between the domains to be translated.

Consider a Bayesian network between two sets of observed random variables A and B with domains A and B respectively along with a parent set of unobserved random variable Z with domain Z. The network is illustrated in FIG0 .The latent variables Z indicate a shared feature space between the observed variables A and B, which will be exploited later for efficient learning and inference.

While Z is unobserved, we assume a prior density p Z over these variables, such as an isotropic Gaussian.

The marginal densities over A and B are not known, and will be learned using the unpaired data from the two domains.

Finally, to specify the joint distribution between these sets of variables, we constrain the relationship between A and Z, and B and Z to be invertible.

That is, we specify mappings G Z???A and G Z???B such that the respective inverses DISPLAYFORM0 Z???B exist.

In the proposed AlignFlow framework, we specify the cross-domain mappings as the composition of two invertible mappings: DISPLAYFORM1 Since composition of invertible mappings is invertible, both G A???B and G B???A are invertible.

In fact, it is straightforward to observe that G A???B and G B???A are inverses of each other: DISPLAYFORM2 Hence, AlignFlow only needs to specify the forward mapping from one domain to another.

The corresponding mapping in the reverse direction is simply given by the inverse of the forward mapping.

Such a choice permits increased flexibility in specifying learning objectives and performing efficient inference, which we discuss next.

B???Z that is exactly cycle-consistent, represents a shared latent space Z between the two domains, and can be trained via both adversarial training and exact maximum likelihood estimation.

Double-headed arrows in AlignFlow denote invertible mapping.

Y A and Y B are random variables denoting the output of the critics used for adversarial training.

DISPLAYFORM3

From a probabilistic standpoint, the cross-domain translation problem requires us to learn a conditional distribution p * A|B over A and B given data sampled from the corresponding marginals p * A and p * B .

We now discuss two methods to learn a mapping from B ??? A such that the resulting marginal distribution over A, denoted as p A is close to p * A .

Unless mentioned otherwise, all our results that hold for a particular domain A will have a natural counterpart for the domain B, by the symmetrical nature of the problem setup and the AlignFlow framework.

Adversarial Training.

A flow model representation permits efficient ancestral sampling.

Hence, a likelihood-free framework to learn the conditional mapping from B to A is to perform adversarial training similar to a GAN.

That is, we introduce a critic C A that plays a minimax game with the generator mapping G B???A .

The critic C A distinguishes real samples a ??? p * A with the generated samples G B???A (b) for b ??? p * B .

An example GAN loss is illustrated in Eq. 2.

Alternatively if our goal is to only learn a generative model with the marginal density close to p * A , then we can choose to simply learn the mapping G Z???A .

As shown in BID14 , the mapping G Z???A along with an easy-to-sample prior density p Z itself specifies a latent variable model that can learned via an adversarial training objective, similar to the one illustrated in Eq. 2 or any other GAN loss.

Maximum Likelihood Estimation.

Flow models can also be trained via maximum likelihood estimation (MLE).

Hence, an MLE objective for learning the mapping G Z???A maximizes the likelihood of the dataset D A : DISPLAYFORM0 where DISPLAYFORM1 As in the previous cases, the expectation w.r.t.

p * A is approximated via Monte Carlo averaging over the dataset D A .

Besides efficient evaluation of the inverse transformations and its Jacobian, this objective additionally requires a prior with a tractable density, e.g. an isotropic Gaussian.

Cycle-consistency.

So far, we have only discussed objectives for modeling the marginal density over A (and symmetrical learning objectives exist for B).

However, as discussed previously, the marginal densities alone do not guarantee learning a mapping that is useful for downstream tasks.

Cycle consistency, as proposed in CycleGAN BID0 , is a highly effective learning objective that encourages learning of meaningful cross-domain mappings.

For AlignFlow, we observe that cycle consistency is exactly satisfied.

Formally, we have the following result: Proposition 1.

Let G denote the class of invertible mappings represented by an arbitrary AlignFlow architecture.

For any G B???A ??? G, we have: DISPLAYFORM2 where DISPLAYFORM3 B???A by design.

The proposition follows directly from the invertible design of the AlignFlow framework (Eq. 7).Overall objective.

In AlignFlow, we optimize a combination of the adversarial learning objective and the maximum likelihood objective.

DISPLAYFORM4 where ?? A ??? 0 and ?? B ??? 0 are hyperparameters that reflect the strength of the MLE terms for domains A and B respectively.

The AlignFlow objective is minimized w.r.t.

the parameters of the generator G A???B and maximized w.r.t.

parameters of the critics C A and C B .

Notice that we have expressed L AlignFlow as a function of the critics C A , C B and only G B???A since the latter also encompasses the other parametric functions appearing in the objective (G A???B , G Z???A , G Z???B ) via the invertibility constraints in Eqs. 5-7.

For different choices of ?? A and ?? B , we cover the following three cases:1.

Adversarial training only: For ?? A = ?? B = 0, we recover the CycleGAN objective in Eq. 4, with the additional benefits of exact cycle consistency and a single invertible generator.

In this case, the prior over Z plays no role in learning.

On the other extreme for large values of ?? A , ?? B such that ?? A = ?? B ??? ???, we can perform pure maximum likelihood training to learn the invertible generator.

Here, the critics C A , C B play no role since the adversarial training terms are ignored in Eq. 11.3.

Hybrid:

For any finite, non-zero value of ?? A , ?? B , we obtain a hybrid objective where both the adversarial and MLE terms are accounted for during learning.

AlignFlow can be used for both conditional and unconditional sampling at test time.

For conditional sampling, we are given a datapoint b ??? B and we can draw the corresponding cross-domain translation in domain A via the mapping G B???A .For unconditional sampling, we require ?? A = 0 since doing so will activate the use of the prior p Z via the MLE terms in the learning objective.

Thereafter, we can obtain samples by first drawing z ??? p Z and then applying the mapping G Z???A to z. Furthermore, the same z can be mapped to domain B via G Z???B .

Hence, we can sample paired data (G Z???A (z), G Z???B (z) given z ??? p Z .

AlignFlow differs from CycleGAN with respect to the model family as well as the learning algorithm and inference capabilities.

We illustrate and compare both models in FIG1 .

CycleGAN parameterizes two independent mappings G A???B and G B???A , whereas AlignFlow only specifies a single, invertible mapping.

Learning in a CycleGAN is restricted to an adversarial training objective along with a cycle-consistent loss term, whereas AlignFlow is exactly consistent and can be trained via adversarial learning, MLE, or a hybrid.

Finally, inference in CycleGAN is restricted to conditional sampling since it does not involve any latent variables Z with easy-to-sample prior densities.

As described previously, AlignFlow permits both conditional and unconditional sampling.

For finite non-zero values of ?? A and ?? B , the AlignFlow objective consists of three parametric models: one generator G B???A ??? G, and two critics C A ??? C A , C B ??? C B .

Here, G, C A , C B denote model families specified e.g., via deep neural network based architectures.

In this section, we analyze the optimal solutions to these parameterized models within well-specified model families.

Our first result characterizes the conditions under which the optimal generators exhibit marginalconsistency for the data distributions defined over the domains A and B. Definition 1.

Let p X,Y denote the joint distribution between two domains X and Y. An invertible mapping G Y???X : Y ??? X is marginally-consistent w.r.t.

two arbitrary distributions (p X , p Y ) iff for all x ??? X , y ??? Y: DISPLAYFORM0 Next, we show that AlignFlow is marginally-consistent for well-specified model families.

Note that marginally-consistent mappings w.r.t.

a target data distribution and a prior density need not be unique.

While an invertible model family mitigates the underconstrained nature of the problem, it does not provably eliminate it.

We provide some non-identifiable constructions in Appendix A.3 and leave the exploration of additional constraints that guarantee identifiability to future work.

Unlike standard adversarial training of an unconditional normalizing flow model BID14 BID23 , the AlignFlow model involves two critics.

Here, we are interested in characterizing the dependence of the optimal critics for a given invertible mapping G A???B .

Consider the AlignFlow framework where the GAN loss terms in Eq. 11 are specified via the cross-entropy objective in Eq. 2.

For this model, we can relate the optimal critics using the following result.

Theorem 2.

Let p * A and p * B denote the true data densities for domains A and B respectively.

Let C * A and C * B denote the optimal critics for the AlignFlow objective with the cross-entropy GAN loss for any fixed choice of the invertible mapping G A???B .

Then, we have for any a ??? A: DISPLAYFORM0 where b = G A???B (a).Proof.

See Appendix A.2.In essence, the above result shows that the optimal critic for one domain, w.l.o.g. say A, can be directly obtained via the optimal critic of another domain B for any choice of the invertible mapping G A???B , assuming one were given access to the data marginals p * A and p * B .

89.4 ?? 0.

In this section, we empirically evaluate AlignFlow for image-to-image translation and unsupervised domain adaptation.

For both these tasks, the most relevant baseline is CycleGAN.

Extensions to CycleGAN that are complementary to our work are excluded for comparison to ensure a controlled evaluation.

We discuss these extensions in detail in Section 6.

In all our experiments, we specify the AlignFlow architecture based on the invertible transformations introduced in Real-NVP BID12 .

For experimental details beyond those stated below, we refer the reader to Appendix B.

We evaluate AlignFlow on three image-to-image translation datasets used by BID0 : Facades, Maps, and CityScapes BID25 .

These datasets are chosen because they provide aligned image pairs, so one can quantitatively evaluate unpaired image-to-image translation models via a distance metric such as mean squared error (MSE) between generated examples and the corresponding ground truth.

Note that we restrict ourselves to unpaired translation, so the pairing information is omitted during training and only used for evaluation.

While MSE can have limitations, we follow prior evaluation protocols and report the MSE for translations on the test sets after cross-validation of hyperparameters in TAB1 .

For hybrid models, we set ?? A = ?? B .

We observe that while learning AlignFlow via adversarial training or MLE alone is not as competitive as CycleGAN, hybrid training of AlignFlow significantly outperforms CycleGAN in almost all cases.

Specifically, we observe that MLE alone typically performs worse than adversarial training, but together both these objectives seem to have a regularizing effect on each other.

Qualitative evaluation of the reconstructions for all datasets is deferred to Appendix B.

The setup for unsupervised domain adaptation BID26 ) is as follows.

We are given data from two related domains: a source and a target domain.

For the source, we have access to both the input datapoints and their labels.

For the target, we are only provided with input datapoints without any labels.

Using the available data, the goal is to learn a classifier for the target domain.

A variety of algorithms have been proposed for the above task which seek to match pixel-level or feature-level distributions across the two domains.

One such model relevant to this experiment is Cycle-Consistent Domain Adaptation (CyCADA) .

CyCADA first learns a cross-domain translation mapping from source to target domain via CycleGAN.

This mapping is used to stylize the source dataset into the target domain, which is then subject to additional feature-level and semantic consistency losses for learning the target domain classifier BID27 .

A full description of CyCADA is beyond the scope of discussion of this work; we direct the reader to for further details.

In this experiment, we seek to assess the usefulness of AlignFlow for domain adaptation in the CyCADA framework.

We evaluate the same pairs of source and target datasets as in : MNIST (LeCun et al., 1998) , USPS BID29 , SVHN BID30 , which are all image datasets of handwritten digits with 10 classes.

Instead of training a source-to-target and a target-to-source generator with a cycle-consistency loss term, we train AlignFlow with only the GAN-based loss in the target direction.

In TAB2 , we see that CyCADA based models perform better in two out of three adaptation settings when used in conjunction with AlignFlow.

A key assumption in unsupervised domain alignment is the existence of a deterministic or stochastic mapping G A???B such that the distribution of B matches that of G A???B (A), and vice versa.

This assumption can be incorporated as a marginal distribution-matching constraint into the objective using an adversarially-trained GAN critic BID9 .

However, this objective is under-constrained.

To partially mitigate this issue, CycleGAN BID0 , DiscoGAN BID1 , and DualGAN BID31 added an approximate cycle-consistency constraint, by encouraging G B???A ??? G A???B and G A???B ??? G B???A to behave like identity functions on domains A and B respectively.

While cycle-consistency is empirically very effective, alternatives based on variational autoencoders that do not require either cycles or adversarial training have also been proposed recently BID32 BID33 .In a parallel line of work, CoGAN BID34 and UNIT BID35 demonstrated the efficacy of adding a shared-space constraint, where two decoders (decoding into domains A and B respectively) share the same latent space.

These works have since been extended to enable one-tomany mappings BID36 BID37 as well as multi-domain alignment BID38 .

Our work focuses on the one-to-one unsupervised domain alignment setting.

In contrast to previous models, AlignFlow leverages both a shared latent space and exact cycle-consistency.

To our knowledge, AlignFlow provides the first demonstration that invertible models can be used successfully in lieu of the cycle-consistency objective.

Furthermore, AlignFlow allows the incorporation of exact maximum likelihood training, which we demonstrated to induce a meaningful shared latent space that is amenable to interpolation.

To enforce exact cycle-consistency, we leverage the growing literature on invertible generative models.

BID11 proposed a class of volume-preserving invertible neural networks (NICE) that uses the change of variables formulation to enable exact maximum likelihood training.

Real-NVP BID12 and Flow++ BID39 extend this line of work by allowing volume transformations and additional architectural considerations.

Glow BID13 further builds upon this by incorporating invertible 1 ?? 1 convolutions.

We note that additional lines of work based on autoregressive flows BID22 BID40 , ordinary differential equations-based flows BID42 , and planar flows BID43 have shown improvements in specific scenarios.

For fast inversion, our work makes use of the Real-NVP model, and we leave extensions of this model in the unsupervised domain alignment setting as future work.

In this work, we presented AlignFlow, a learning framework for cross-domain translations based on normalizing flow models.

The use of normalizing flow models is an attractive choice for several reasons we highlight: it guarantees exact cycle-consistency via a single cross-domain mapping, learns a shared latent space across two domains, and permits a flexible training objective which is a hybrid of terms corresponding to adversarial training and exact maximum likelihood estimation.

Theoretically, we derived conditions under which the AlignFlow model learns marginals that are consistent with the underlying data distributions.

Finally, our empirical evaluation demonstrated significant gains on the tasks of image-to-image translation and unsupervised domain adaptation, along with an increase in inference capabilities due to the use of invertible models, e.g., paired interpolations in the latent space for two domains.

In the future, we would like to consider extensions of AlignFlow to learning stochastic, multimodal mappings BID37 and translations across more than two domains BID38 .

In spite of strong empirical results in domain alignments in the last few years, a well-established theory explaining such results is lacking.

With a handle on model likelihoods and exact invertibility for inference, we are optimistic that AlignFlow can potentially aid the development of such a theory and characterize structure that leads to provably identifiable recovery of cross-domain mappings.

Exploring the latent space of AlignFlow from a manifold learning perspective to domain alignment BID44 is also an interesting direction for future research.

A.1 PROOF OF THEOREM 1Proof.

Since the maximum likelihood estimate minimizes the KL divergence between the data and model distributions, the optimal value for L MLE (G Z???A ) is attained at a marginally-consistent mapping, say G * Z???A .

Symmetrically, there exists a marginally-consistent mapping G * Z???B that optimizes L MLE (G Z???B )

.From Theorem 1 of BID9 , we know that the cross-entropy GAN objective L GAN (C A , G B???A ) is globally minimized when p A = p * A and critic is Bayes optimal.

Further, from Lemma 1, we know that G * B???A is marginally-consistent w.r.t.

DISPLAYFORM0 Z???B globally optimizes all the individual loss terms in the AlignFlow objective in Eq. 11, it globally optimizes the overall objective for any value of ?? A ??? 0, ?? B ??? 0.

Proof.

First, we note that only the GAN loss terms depend on C A and C B .

Hence, the MLE terms are constants for a fixed G B???A and hence, can be ignored for deriving the optimal critics.

Next, for any GAN trained with the cross-entropy loss as specified in Eq 2, we know that the Bayes optimal critic C * A prediction for any a ??? A is given as: DISPLAYFORM0 See Proposition 1 in BID9 for a proof.

We can relate the densities p A (a) and p B (b) via the change of variables as: DISPLAYFORM1 where b = G A???B (a).Substituting the expression for density of p A (a) from Eq. 15 in Eq. 14, we get: DISPLAYFORM2 where b = G A???B (a).Symmetrically, using Proposition 1 in BID9 we have the Bayes optimal critic C * B for any b ??? B given as: DISPLAYFORM3 Rearranging terms in Eq. 17, we have: DISPLAYFORM4 for any b ??? B.Substituting the expression for density of p B (b) from Eq. 18 in Eq. 16, we get: DISPLAYFORM5 where b = G A???B (a).

As discussed, marginal consistency along with invertibility can only reduce the underconstrained nature of the unpaired cross-domain translation problem, but not completely eliminate it.

In the following result, we identify one such class of non-identifiable model families for the MLE-only objective of AlignFlow (?? A = ???, ?? B = ???).

We will need the following definitions.

Definition 2.

Let S n denotes the symmetric group on n dimensional permutation matrices.

A function class for the cross-domain mappings G is closed under permutations iff for all G B???A ??? G, S ??? S n , we have G B???A ??? S ??? G.Definition 3.

A density p X is symmetric iff for all x ??? X ??? R n , S ??? S n , we have p X (x) = p X (Sx).Examples of distributions with symmetric densities include the isotropic Gaussian and Laplacian distributions.

Proposition 2.

Consider the case where G * B???A ??? G, and G is closed under permutations.

For a symmetric prior p Z (e.g., isotropic Gaussian), there exists an optimal solution G ??? B???A ??? G to the AlignFlow objective (Eq. 11) for ?? A = ?? B = ??? such that G ??? B???A = G * B???A .Proof.

We will prove the proposition via contradiction.

That is, let's assume that G * B???A is a unique solution for the AlignFlow objective for ?? A = ?? B = ??? (Eq. 11).

Now, consider an alternate mapping G ??? B???A = G * B???A S for an arbitrary non-identity permutation matrix S = I in the symmetric group.

Z???B due to the invertibility constraints in Eqs. 5-7.

Since permutation matrices are invertible and so is G * B???A , their composition given by G ??? B???A is also invertible.

Further, since G is closed under permutation and G * B???A ??? G, we also have G ??? B???A ??? G. Next, we note that the inverse of a permutation matrix is also a permutation matrix.

Since the prior is assumed to be symmetric and a a transformation specified by a permutation matrix is volumepreserving (i.e., det(S) = 1 for all S ??? S n ), we can use the change-of-variables formula in Eq. 1 to get: DISPLAYFORM0 DISPLAYFORM1 Noting that G * DISPLAYFORM2 Z???B due to the invertibility constraints in Eqs. 5-7, we can substitute the above equations in Eq. 11.

When ?? A = ?? B = ???, for any choice of C A , C B we have: DISPLAYFORM3 The above equation implies that G ??? B???A is also an optimal solution to the AlignFlow objective in Eq. 11 for ?? A = ?? B = ???. Thus, we arrive at a contradiction since G * B???A is not the unique maximizer.

Hence, proved.

The above construction suggests that MLE-only training can fail to identify the optimal mapping corresponding to the joint distribution p * A,B even if it lies within the mappings represented via the family represented via the AlignFlow architecture.

Failure modes due to non-identifiability could also potentially arise for adversarial and hybrid training.

Empirically, we find that while MLE-only training gives poor performance for cross-domain translations, the hybrid and adversarial training objectives are much more effective, which suggests that these objectives are less susceptible to identifiability issues in recovering the true mapping.

<|TLDR|>

@highlight

We propose a learning framework for cross-domain translations which is exactly cycle-consistent and can be learned via adversarial training, maximum likelihood estimation, or a hybrid.

@highlight

Proposes AlignFlow, an efficient way of implementing cycle consistency principle using invertible flows.

@highlight

Flow models for unpaired image to image translation
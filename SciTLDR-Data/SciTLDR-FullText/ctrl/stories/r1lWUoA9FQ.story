A wide range of defenses have been proposed to harden neural networks against adversarial attacks.

However, a pattern has emerged in which the majority of adversarial defenses are quickly broken by new attacks.

Given the lack of success at generating robust defenses, we are led to ask a fundamental question:  Are adversarial attacks inevitable?

This paper analyzes adversarial examples from a theoretical perspective, and identifies fundamental bounds on the susceptibility of a classifier to adversarial attacks.

We show that, for certain classes of problems, adversarial examples are inescapable.

Using experiments, we explore the implications of theoretical guarantees for real-world problems and discuss how factors such as dimensionality and image complexity limit a classifier's robustness against adversarial examples.

A number of adversarial attacks on neural networks have been recently proposed.

To counter these attacks, a number of authors have proposed a range of defenses.

However, these defenses are often quickly broken by new and revised attacks.

Given the lack of success at generating robust defenses, we are led to ask a fundamental question: Are adversarial attacks inevitable?In this paper, we identify a broad class of problems for which adversarial examples cannot be avoided.

We also derive fundamental limits on the susceptibility of a classifier to adversarial attacks that depend on properties of the data distribution as well as the dimensionality of the dataset.

Adversarial examples occur when a small perturbation to an image changes its class label.

There are different ways of measuring what it means for a perturbation to be "small"; as such, our analysis considers a range of different norms.

While the ∞ -norm is commonly used, adversarial examples can be crafted in any p -norm (see FIG0 ).

We will see that the choice of norm can have a dramatic effect on the strength of theoretical guarantees for the existence of adversarial examples.

Our analysis also extends to the 0 -norm, which yields "sparse" adversarial examples that only perturb a small subset of image pixels FIG2 ).

BID19 on Resnet50, along with the distance between the base image and the adversarial example, and the top class label.

2As a simple example result, consider a classification problem with n-dimensional images with pixels scaled between 0 and 1 (in this case images live inside the unit hypercube).

If the image classes each occupy a fraction of the cube greater than 1 2 exp(−π 2 ), then images exist that are susceptible to adversarial perturbations of 2 -norm at most .

Note that = 10 was used in FIG0 , and larger values are typical for larger images.

Finally, in Section 8, we explore the causes of adversarial susceptibility in real datasets, and the effect of dimensionality.

We present an example image class for which there is no fundamental link between dimensionality and robustness, and argue that the data distribution, and not dimensionality, is the primary cause of adversarial susceptibility.

Adversarial examples, first demonstrated in BID36 and BID3 , change the label of an image using small and often imperceptible perturbations to its pixels.

A number of defenses have been proposed to harden networks against attacks, but historically, these defenses have been quickly broken.

Adversarial training, one of the earliest defenses, successfully thwarted the fast gradient sign method (FGSM) BID13 , one of the earliest and simplest attacks.

However, adversarial training with FGSM examples was quickly shown to be vulnerable to more sophisticated multi-stage attacks BID16 BID39 .

More sophisticated defenses that rely on network distillation BID26 and specialized activation functions BID44 were also toppled by strong attacks BID25 BID40 BID6 .The ongoing vulnerability of classifiers was highlighted in recent work by BID2 and BID1 that broke an entire suite of defenses presented in ICLR 2018 including thermometer encoding BID5 , detection using local intrinsic dimensionality BID18 , input transformations such as compression and image quilting BID14 , stochastic activation pruning BID10 , adding randomization at inference time BID42 , enhancing the confidence of image labels BID33 , and using a generative model as a defense BID29 .Rather than hardening classifiers to attacks, some authors have proposed sanitizing datasets to remove adversarial perturbations before classification.

Approaches based on auto-encoders BID22 and GANs BID30 were broken using optimization-based attacks BID8 a) .A number of "certifiable" defense mechanisms have been developed for certain classifiers.

BID27 harden a two-layer classifier using semidefinite programming, and BID32 propose a convex duality-based approach to adversarial training that works on sufficiently small adversarial perturbations with a quadratic adversarial loss.

BID15 consider training a robust classifier using the convex outer adversarial polytope.

All of these methods only consider robustness of the classifier on the training set, and robustness properties often fail to generalize reliably to test examples.

One place where researchers have enjoyed success is at training classifiers on low-dimensional datasets like MNIST BID19 BID32 .

The robustness achieved on more complicated datasets such DISPLAYFORM0 Figure 2: Sparse adversarial examples perturb a small subset of pixels and can hide adversarial "fuzz" inside highfrequency image regions.

The original image (left) is classified as an "ox."

Under ∞-norm perturbations, it is classified as "traffic light", but the perturbations visibly distort smooth regions of the image (the sky).

These effects are hidden in the grass using 0-norm (sparse) perturbations limited to a small subset of pixels.as CIFAR-10 and ImageNet are nowhere near that of MNIST, which leads some researchers to speculate that adversarial defense is fundamentally harder in higher dimensions -an issue we address in Section 8.This paper uses well-known results from high-dimensional geometry, specifically isoperimetric inequalities, to provide bounds on the robustness of classifiers.

Several other authors have investigated adversarial susceptibility through the lens of geometry.

BID11 study adversarial susceptibility of datasets under the assumption that they are produced by a generative model that maps random Gaussian vectors onto images.

BID12 do a detailed case study, including empirical and theoretical results, of classifiers for a synthetic dataset that lies on two concentric spheres.

BID31 show that the Lipschitz constant of untrained networks with random weights gets large in high dimensions.

Shortly after the original appearance of our work, BID20 presented a study of adversarial susceptibility that included both evasion and poisoning attacks.

Our work is distinct in that it studies adversarial robustness for arbitrary data distributions, and also that it rigorously looks at the effect of dimensionality on robustness limits.

We use [0, 1] n to denote the unit hypercube in n dimensions, and vol(A) to denote the volume (i.e., ndimensional Lebesgue measure) of a subset A ⊂ [0, 1] n .

We use S n−1 = {x ∈ R n | x 2 = 1} to denote the unit sphere embedded in R n , and s n−1 to denote its surface area.

The size of a subset A ∈ S n−1 can be quantified by its (n − 1 dimensional) measure µ[A], which is simply the surface area the set covers.

Because the surface area of the unit sphere varies greatly with n, it is much easier in practice to work with the normalized measure, which we denote µ 1 [A] = µ[A]/s n−1 .

This normalized measure has the property that µ 1 [S n−1 ] = 1, and so we can interpret µ 1 [A] as the probability of a uniform random point from the sphere lying in A. When working with points on a sphere, we often use geodesic distance, which is always somewhat larger than (but comparable to) the Euclidean distance.

In the cube, we measure distance between points using p -norms, which are denoted DISPLAYFORM0 Note that · p is not truly a norm for p < 1, but rather a semi-norm.

Such metrics are still commonly used, particularly the " 0 -norm" which counts the number of non-zero entries in a vector.

We consider the problem of classifying data points that lie in a space Ω (either a sphere or a hypercube) into m different object classes.

The m object classes are defined by probability density functions {ρ c } m c=1 , where ρ c : Ω → R. A "random" point from class c is a random variable with density ρ c .

We assume ρ c to be bounded (i.e., we don't allow delta functions or other generalized functions), and denote its upper bound by U c = sup x ρ c (x).We also consider a "classifier" function C : Ω → {1, 2, . . .

, m} that partitions Ω into disjoint measurable subsets, one for each class label.

The classifier we consider is discrete valued -it provides a label for each data point but not a confidence level.

With this setup, we can give a formal definition of an adversarial example.

Definition 1.

Consider a point x ∈ Ω drawn from class c, a scalar > 0, and a metric d. We say that x admits an -adversarial example in the metric d if there exists a pointx ∈ Ω with C(x) = c, and d(x,x) ≤ .In plain words, a point has an -adversarial example if we can sneak it into a different class by moving it at most units in the distance d.

We consider adversarial examples with respect to different p -norm metrics.

These metrics are written d p (x,x) = x −x p .

A common choice is p = ∞, which limits the absolute change that can be made to any one pixel.

However, 2 -norm and 1 -norm adversarial examples are also used, as it is frequently easier to create adversarial examples in these less restrictive metrics.

We also consider sparse adversarial examples in which only a small subset of pixels are manipulated.

This corresponds to the metric d 0 , in which case the constraint x −x 0 ≤ means that an adversarial example was crafted by changing at most pixels, and leaving the others alone.

We begin by looking at the case of classifiers for data on the sphere.

While this data model may be less relevant than the other models studied below, it provides a straightforward case where results can be proven using simple, geometric lemmas.

The more realistic case of images with pixels in [0, 1] will be studied in Section 4.The idea is to show that, provided a class of data points takes up enough space, nearly every point in the class lies close to the class boundary.

To show this, we begin with a simple definition.

Definition 2.

The -expansion of a subset A ⊂ Ω with respect to distance metric d, denoted A( , d), contains all points that are at most units away from A. To be precise DISPLAYFORM0 We sometimes simply write A( ) when the distance metric is clear from context.

Our result provides bounds on the probability of adversarial examples that are independent of the shape of the class boundary.

This independence is a simple consequence of an isoperimetric inequality.

The classical isoperimetric inequality states that, of all closed surfaces that enclose a unit volume, the sphere has the smallest surface area.

This simple fact is intuitive but famously difficult to prove.

For a historical review of the isoperimetric inequality and its variants, see BID24 .

We will use a special variant of the isoperimetric inequality first proved by BID18 and simplified by BID37 .

Lemma 1 (Isoperimetric inequality).

Consider a subset of the sphere A ⊂ S n−1 ⊂ R n with normalized measure µ 1 (A) ≥ 1/2.

When using the geodesic metric, the -expansion A( ) is at least as large as the -expansion of a half sphere.

The classical isoperimetric inequality is a simple geometric statement, and frequently appears without absolute bounds on the size of the -expansion of a half-sphere, or with bounds that involve unspecified constants BID41 .

A tight bound derived by BID23 is given below.

The asymptotic blow-up of the -expansion of a half sphere predicted by this bound is shown in FIG1 .

Lemma 2 ( -expansion of half sphere).

The geodesic -expansion of a half sphere has normalized measure at least DISPLAYFORM1 Lemmas 1 and 2 together can be taken to mean that, if a set is not too small, then in high dimensions almost all points on the sphere are reachable within a short jump from that set.

These lemmas have immediate implications for adversarial examples, which are formed by mapping one class into another using small perturbations.

Despite its complex appearance, the result below is a consequence of the (relatively simple) isoperimetric inequality.

Theorem 1 (Existence of Adversarial Examples).

Consider a classification problem with m object classes, each distributed over the unit sphere S n−1 ⊂ R n with density functions {ρ c } m c=1 .

Choose a classifier function C : S n−1 → {1, 2, . . .

, m} that partitions the sphere into disjoint measurable subsets.

Define the following scalar constants:• Let V c denote the magnitude of the supremum of ρ c relative to the uniform density.

This can be written V c := s n−1 · sup x ρ c (x).• Let f c = µ 1 {x|C(x) = c} be the fraction of the sphere labeled as c by classifier C.Choose some class c with f c ≤ 1 2 .

Sample a random data point x from ρ c .

Then with probability at least DISPLAYFORM2 one of the following conditions holds: 1.

x is misclassified by C, or 2.

x admits an -adversarial example in the geodesic distance.

Proof.

Choose a class c with f c ≤ 1 2 .

Let R = {x|C(x) = c} denote the region of the sphere labeled as class c by C, and let R be its complement.

R( ) is the -expansion of R in the geodesic metric.

Because R covers at least half the sphere, the isoperimetric inequality (Lemma 1) tells us that the epsilon expansion is at least as great as the epsilon expansion of a half sphere.

We thus have DISPLAYFORM3 Now, consider the set S c of "safe" points from class c that are correctly classified and do not admit adversarial perturbations.

A point is correctly classified only if it lies inside R, and therefore outside of R. To be safe from adversarial perturbations, a point cannot lie within distance from the class boundary, and so it cannot lie within R( ).

It is clear that the set S c of safe points is exactly the complement of R( ).

This set has normalized measure DISPLAYFORM4 The probability of a random point lying in S c is bounded above by the normalized supremum of ρ c times the normalized measure µ 1 [S c ].

This product is given by DISPLAYFORM5 We then subtract this probability from 1 to obtain the probability of a point lying outside the safe region, and arrive at equation 1.In the above result, we measure the size of adversarial perturbations using the geodesic distance.

Most studies of adversarial examples measure the size of perturbation in either the 2 (Euclidean) norm or the ∞ (max) norm, and so it is natural to wonder whether Theorem 1 depends strongly on the distance metric.

Fortunately (or, rather unfortunately) it does not.

It is easily observed that, for any two points x and y on a sphere, DISPLAYFORM6 where d ∞ (x, y), d 2 (x, y), and d g (x, y) denote the l ∞ , Euclidean, and geodesic distance, respectively.

From this, we see that Theorem 1 is actually fairly conservative; any -adversarial example in the geodesic metric would also be adversarial in the other two metrics, and the bound in Theorem 1 holds regardless of which of the three metrics we choose (although different values of will be appropriate depending on the norm).

The above result about the sphere is simple and easy to prove using classical results.

However, real world images do not lie on the sphere.

In a more typical situation, images will be scaled so that their pixels lie in [0, 1], and data lies inside a high-dimensional hypercube (but, unlike the sphere, data is not confined to its surface).

The proof of Theorem 1 makes extensive use of properties that are exclusive to the sphere, and is not applicable to this more realistic setting.

Are there still problem classes on the cube where adversarial examples are inevitable?This question is complicated by the fact that geometric isoperimetric inequalities do not exist for the cube, as the shapes that achieve minimal -expansion (if they exist) depend on the volume they enclose and the choice of BID28 .

Fortunately, researchers have been able to derive "algebraic" isoperimetric inequalities that provide lower bounds on the size of the -expansion of sets without identifying the shape that achieves this minimum BID38 BID23 .

The result below about the unit cube is analogous to Proposition 2.8 in BID17 , except with tighter constants.

For completeness, a proof (which utilizes methods from Ledoux) is provided in Appendix A. Lemma 3 (Isoperimetric inequality on a cube).

Consider a measurable subset of the cube A ⊂ [0, 1] n , and DISPLAYFORM0 2 /2 dt, and let α be the scalar that satisfies DISPLAYFORM1 where p * = min(p, 2).

In particular, if vol(A) ≥ 1/2, then we simply have DISPLAYFORM2 Using this result, we can show that most data samples in a cube admit adversarial examples, provided the data distribution is not excessively concentrated.

n → {1, 2, . . .

, m} that partitions the hypercube into disjoint measurable subsets.

Define the following scalar constants:• Let U c denote the supremum of ρ c .• Let f c be the fraction of hypercube partitioned into class c by C.Choose some class c with f c ≤ 1 2 , and select an p -norm with p > 0.

Define p * = min(p, 2).

Sample a random data point x from the class distribution ρ c .

Then with probability at least DISPLAYFORM3 one of the following conditions holds:1.

x is misclassified by C, or 2.

x has an adversarial examplex, with x −x p ≤ .When adversarial examples are defined in the 2 -norm (or for any p ≥ 2), the bound in equation 4 becomes DISPLAYFORM4 Provided the class distribution is not overly concentrated, equation 5 guarantees adversarial examples with relatively "small" relative to a typical vector.

In n dimensions, the 2 diameter of the cube is √ n, and so it is reasonable to choose = O( √ n) in equation 5.

In FIG0 , we chose = 10.

A similarly strong bound of DISPLAYFORM5 DISPLAYFORM6 2 /2 (for z > 0), and α = Φ −1 (1 − f c ).

For this bound to be meaningful with < 1, we need f c to be relatively small, and to be roughly f c or smaller.

This is realistic for some problems; ImageNet has 1000 classes, and so f c < 10 −3 for at least one class.

Interestingly, under ∞ -norm attacks, guarantees of adversarial examples are much stronger on the sphere (Section 3) than on the cube.

One might wonder whether the weakness of Theorem 4 in the ∞ case is fundamental, or if this is a failure of our approach.

One can construct examples of sets with ∞ expansions that nearly match the behavior of equation 5, and so our theorems in this case are actually quite tight.

It seems to be inherently more difficult to prove the existence of adversarial examples in the cube using the ∞ -norm.

A number of papers have looked at sparse adversarial examples, in which a small number of image pixels, in some cases only one BID34 , are changed to manipulate the class label.

To study this case, we would like to investigate adversarial examples under the 0 metric.

The 0 distance is defined as DISPLAYFORM0 If a point x has an -adversarial example in this norm, then it can be perturbed into a different class by modifying at most pixels (in this case is taken to be a positive integer).Theorem 2 is fairly tight for p = 1 or 2.

However, the bound becomes quite loose for small p, and in particular it fails completely for the important case of p = 0.

For this reason, we present a different bound that is considerably tighter for small p (although slightly looser for large p).The case p = 0 was studied by Milman & Schechtman (1986) (Section 6.2) and BID21 , and later by BID37 BID38 .

The proof of the following theorem (appendix B) follows the method used in Section 5 of BID38 , with modifications made to extend the proof to arbitrary p. Lemma 4 (Isoperimetric inequality on the cube: small p).

Consider a measurable subset of the cube A ⊂ [0, 1] n , and a p-norm distance metric d(x, y) = x − y p for any p ≥ 0.

We have DISPLAYFORM1 Using this result, we can prove a statement analogous to Theorem 2, but for sparse adversarial examples.

We present only the case of p = 0, but the generalization to the case of other small p using Lemma 4 is straightforward.

Theorem 3 (Sparse adversarial examples).

Consider the problem setup of Theorem 2.

Choose some class c with f c ≤ 1 2 , and sample a random data point x from the class distribution ρ c .

Then with probability at least DISPLAYFORM2 one of the following conditions holds: 1.

x is misclassified by C, or 2.

x can be adversarially perturbed by modifying at most pixels, while still remaining in the unit hypercube.

Tighter bounds can be obtained if we only guarantee that adversarial examples exist for some data points in a class, without bounding the probability of this event.

Theorem 4 (Condition for existence of adversarial examples).

Consider the setup of Theorem 2.

Choose a class c that occupies a fraction of the cube f c < 1 2 .

Pick an p norm and set p * = min(p, 2).Let supp(ρ c ) denote the support of ρ c .

Then there is a point x with ρ c (x) > 0 that admits an -adversarial example if DISPLAYFORM0 The bound for the case p = 0 is valid only if ≥ n log 2/2.It is interesting to consider when Theorem 4 produces non-vacuous bounds.

When the 2 -norm is used, the bound becomes vol[supp(ρ c )]

≥ exp(−π 2 )/2.

The diameter of the cube is √ n, and so the bound becomes active for = √ n. Plugging this in, we see that the bound is active whenever the size of the support satisfies DISPLAYFORM1 2e πn .

Remarkably, this holds for large n whenever the support of class c is larger than (or contains) a hypercube of side length at least e −π ≈ 0.043.

Note, however, that the bound being "active" does not guarantee adversarial examples with a "small" .

There are a number of ways to escape the guarantees of adversarial examples made by Theorems 1-4.

One potential escape is for the class density functions to take on extremely large values (i.e., exponentially large U c ); the dependence of U c on n is addressed separately in Section 8.Unbounded density functions and low-dimensional data manifolds In practice, image datasets might lie on low-dimensional manifolds within the cube, and the support of these distributions could have measure zero, making the density function infinite (i.e., U c = ∞).

The arguments above are still relevant (at least in theory) in this case; we can expand the data manifold by adding a uniform random noise to each image pixel of magnitude at most 1 .

The expanded dataset has positive volume.

Then, adversarial examples of this expanded dataset can be crafted with perturbations of size 2 .

This method of expanding the manifold before crafting adversarial examples is often used in practice.

BID39 proposed adding a small perturbation to step off the image manifold before crafting adversarial examples.

This strategy is also used during adversarial training BID19 .Adding a "don't know" class The analysis above assumes the classifier assigns a label to every point in the cube.

If a classifier has the ability to say "I don't know," rather than assign a label to every input, then the region of the cube that is assigned class labels might be very small, and adversarial examples could be escaped even if the other assumptions of Theorem 4 are satisfied.

In this case, it would still be easy for the adversary to degrade classifier performance by perturbing images into the "don't know" class.

Feature squeezing If decreasing the dimensionality of data does not lead to substantially increased values for U c (we see in Section 8 that this is a reasonable assumption) or loss in accuracy (a stronger assumption), measuring data in lower dimensions could increase robustness.

This can be done via an auto-encoder BID22 BID30 , JPEG encoding BID9 , or quantization BID43 .Computational hardness It may be computationally hard to craft adversarial examples because of local flatness of the classification function, obscurity of the classifier function, or other computational difficulties.

Computational hardness could prevent adversarial attacks in practice, even if adversarial examples still exist.

In this section, we discuss the relationship between dimensionality and adversarial robustness, and explore how the predictions made by the theorems above are reflected in experiments.

It is commonly thought that high-dimensional classifiers are more susceptible to adversarial examples than low-dimensional classifiers.

This perception is partially motivated by the observation that classifiers on highresolution image distributions like ImageNet are more easily fooled than low resolution classifiers on MNIST BID39 .

Indeed, Theorem 2 predicts that high-dimensional classifiers should be much easier to fool than low-dimensional classifiers, assuming the datasets they classify have comparable probability density limits U c .

However, this is not a reasonable assumption; we will see below that high dimensional distributions may be more concentrated than their low-dimensional counterparts.

We study the effects of dimensionality with a thought experiment involving a "big MNIST" image distribution.

Given an integer expansion factor b, we can make a big MNIST distribution, denoted b-MNIST, by replacing each pixel in an MNIST image with a b × b array of identical pixels.

This expands an original 28 × 28 image into a 28b × 28b image.

FIG3 shows that, without adversarial training, a classifier on big MNIST is far more susceptible to attacks than a classifier trained on the original MNIST 1 .However, each curve in FIG3 only shows the attack susceptibility of one particular classifier.

In contrast, Theorems 1-4 describe the fundamental limits of susceptibility for all classifiers.

These limits are an inherent property of the data distribution.

The theorem below shows that these fundamental limits do not depend in a non-trivial way on the dimensionality of the images in big MNIST, and so the relationship between dimensionality and susceptibility in FIG3 results from the weakness of the training process.

Theorem 5.

Suppose and p are such that, for all MNIST classifiers, a random image from class c has an -adversarial example (in the 2 -norm) with probability at least p.

Then for all classifiers on b-MNIST, with integer b ≥ 1, a random image from c has a b -adversarial example with probability at least p.

Likewise, if all b-MNIST classifiers have b -adversarial examples with probability p for some b ≥ 1, then all classifiers on the original MNIST distribution have -adversarial examples with probability p.

Theorem 5 predicts that the perturbation needed to fool all 56 × 56 classifiers is twice that needed to fool all 28 × 28 classifiers.

This is reasonable since the 2 -norm of a 56 × 56 image is twice that of its 28 × 28 counterpart.

Put simply, fooling big MNIST is just as hard/easy as fooling the original MNIST regardless of resolution.

This also shows that for big MNIST, as the expansion factor b gets larger and is expanded to match, the concentration bound U c grows at exactly the same rate as the exponential term in equation 2 shrinks, and there is no net effect on fundamental susceptibility.

Also note that an analogous result could be based on any image classification problem (we chose MNIST only for illustration), and any p ≥ 0.We get a better picture of the fundamental limits of MNIST by considering classifiers that are hardened by adversarial training 2 FIG3 ).

These curves display several properties of fundamental limits predicted by our theorems.

As predicted by Theorem 5, the 112 × 112 classifer curve is twice as wide as the 56 × 56 curve, which in turn is twice as wide as the 28×28 curve.

In addition, we see the kind of "phase transition" behavior predicted by Theorem 2, in which the classifier suddenly changes from being highly robust to being highly susceptible as passes a critical threshold.

For these reasons, it is reasonable to suspect that the adversarially trained classifiers in FIG3 are operating near the fundamental limit predicted by Theorem 2.Theorem 5 shows that increased dimensionality does not increase adversarial susceptibility in a fundamental way.

But then why are high-dimensional classifiers so easy to fool?

To answer this question, we look at the concentration bound U c for object classes.

The smallest possible value of U c is 1, which only occurs when images are "spread out" with uniform, uncorrelated pixels.

In contrast, adjacent pixels in MNIST (and especially big MNIST) are very highly correlated, and images are concentrated near simple, low-dimensional manifolds, resulting in highly concentrated image classes with large U c .

Theory predicts that such highly concentrated datasets can be relatively safe from adversarial examples.

Under review as a conference paper at ICLR 2019 We can reduce U c and dramatically increase susceptibility by choosing a more "spread out" dataset, like CIFAR-10, in which adjacent pixels are less strongly correlated and images appear to concentrate near complex, higher-dimensional manifolds.

We observe the effect of decreasing U c by plotting the susceptibility of a 56 × 56 MNIST classifier against a classifier for CIFAR-10 FIG3 , right).

The former problem lives in 3136 dimensions, while the latter lives in 3072, and both have 10 classes.

Despite the structural similarities between these problems, the decreased concentration of CIFAR-10 results in vastly more susceptibility to attacks, regardless of whether adversarial training is used.

The theory above suggests that this increased susceptibility is caused at least in part by a shift in the fundamental limits for CIFAR-10, rather than the weakness of the particular classifiers we chose.

Informally, the concentration limit U c can be interpreted as a measure of image complexity.

Image classes with smaller U c are likely concentrated near high-dimensional complex manifolds, have more intra-class variation, and thus more apparent complexity.

An informal interpretation of Theorem 2 is that "high complexity" image classes are fundamentally more susceptible to adversarial examples, and FIG3 suggests that complexity (rather than dimensionality) is largely responsible for differences we observe in the effectiveness of adversarial training for different datasets.

The question of whether adversarial examples are inevitable is an ill-posed one.

Clearly, any classification problem has a fundamental limit on robustness to adversarial attacks that cannot be escaped by any classifier.

However, we have seen that these limits depend not only on fundamental properties of the dataset, but also on the strength of the adversary and the metric used to measure perturbations.

This paper provides a characterization of these limits and how they depend on properties of the data distribution.

Unfortunately, it is impossible to know the exact properties of real-world image distributions or the resulting fundamental limits of adversarial training for specific datasets.

However, the analysis and experiments in this paper suggest that, especially for complex image classes in high-dimensional spaces, these limits may be far worse than our intuition tells us.

A PROOF OF LEMMA 3We now prove Lemma 3.

To do this, we begin with a classical isoperimetric inequality for random Gaussian variables.

Unlike the case of a cube, tight geometric isoperimetric inequalities exist in this case.

We then prove results about the cube by creating a mapping between uniform random variables on the cube and random Gaussian vectors.

In the lemma below, we consider the standard Gaussian density in R n given by p(x) = 1 (2π) n/2 e −nx 2 /2 and corresponding Gaussian measure µ. We also define DISPLAYFORM0 which is the cumulative density of a Gaussian curve.

The following Lemma was first proved in BID35 , and an elementary proof was given in BID4 .

Lemma 5 (Gaussian Isoperimetric Inequality).

Of all sets with the same Gaussian measure, the set with 2 -expansion of smallest measure is a half space.

Furthermore, for any measurable set A ⊂ R n , and scalar constant a such that DISPLAYFORM1 Using this result we can now give a proof of Lemma 3.This function Φ maps a random Guassian vector z ∈ N (0, I) onto a random uniform vector in the unit cube.

To see why, consider a measurable subset B ⊂ R n .

If µ is the Gaussian measure on R n and σ is the uniform measure on the cube, then DISPLAYFORM2 .

DISPLAYFORM3 , we also have DISPLAYFORM4 for any z, w ∈ R n .

From this, we see that for p * = min(p, 2) DISPLAYFORM5 where we have used the identity u p ≤ n 1/ min(p,2)−1/2 u 2 .

Now, consider any set A in the cube, and let B = Φ −1 (A).

From equation 10, we see that DISPLAYFORM6 It follows from equation 10 that DISPLAYFORM7 Applying Lemma 5, we see that DISPLAYFORM8 where DISPLAYFORM9 To obtain the simplified formula in the theorem, we use the identity DISPLAYFORM10 which is valid for x > 0, and can be found in BID0 .B PROOF OF LEMMA 4Our proof emulates the method of Talagrand, with minor modifications that extend the result to other p norms.

We need the following standard inequality.

Proof can be found in BID37 BID38 .

Lemma 6 (Talagrand).

Consider a probability space Ω with measure µ. For g : Ω → [0, 1], we have DISPLAYFORM11 Our proof of Lemma 3 follows the three-step process of Talagrand illustrated in BID37 .

We begin by proving the bound DISPLAYFORM12 where DISPLAYFORM13 is a measure of distance from A to x, and α, t are arbitrary positive constants.

Once this bound is established, a Markov bound can be used to obtain the final result.

Finally, constants are tuned in order to optimize the tightness of the bound.

We start by proving the bound in equation 12 using induction on the dimension.

The base case for the induction is n = 1, and we have DISPLAYFORM14 We now prove the result for n dimensions using the inductive hypothesis.

We can upper bound the integral by integrating over "slices" along one dimension.

Let A ⊂ [0, 1] n .

Define DISPLAYFORM15 Clearly, the distance from (ω, z) to A is at most the distance from z to A ω , and so DISPLAYFORM16 We also have that the distance from x to A is at most one unit greater than the distance from x to B. This gives us .

DISPLAYFORM17 Now, we apply lemma 6 to equation 13 with g(ω) = α[A ω ]/α[B] to arrive at equation 12.The second step of the proof is to produce a Markov inequality from equation 12.

For the bound in equation 12 to hold, we need DISPLAYFORM18 The third step is to optimize the bound by choosing constants.

We minimize the right hand side by choosing t = Now, we can simply choose α = 1 to get the simple bound DISPLAYFORM19 or we can choose the optimal value of α = 2 2p n log(1/σ) − 1, which optimizes the bound in the case 2p ≥ n 2 log(1/σ(A)).

We arrive at DISPLAYFORM20 This latter bound is stronger than we need to prove Lemma 3, but it will come in handy later to prove Theorem 4.C PROOF OF THEOREMS 2 AND 3We combine the proofs of these results since their proofs are nearly identical.

The proofs closely follow the argument of Theorem 1.Choose a class c with f c ≤ 1 2 and let R = {x|C(x) = c} denote the subset of the cube lying in class c according to the classifier C. Let R be the complement, who's p expansion is denoted R( ; d p ).

Because R covers at least half the cube, we can invoke Lemma 3.

We have that vol[R( ; h)] ≥ 1 − δ, where δ = exp(−πn 1−2/p * 2 ) 2πn 1/2−1/p * , for Theorem 2 and 2U c exp(− 2 /n), for Theorem 3.The set R( ; h) contains all points that are correctly classified and safe from adversarial perturbations.

This region has volume at most δ, and the probability of a sample from the class distribution ρ c lying in this region is at most U c δ.

We then subtract this from 1 to obtain the mass of the class distribution lying in the "unsafe" region R c .

Let A denote the support of p c , and suppose that this support has measure vol[A] = η.

We want to show that, for large enough , the expansion A( , d p ) is larger than half the cube.

Since class c occupies less than half the cube, this would imply that A( , d p ) overlaps with other classes, and so there must be data points in A with -adversarial examples.

We start with the case p > 0, where we bound A( , d p ) using equation 2 of Lemma 3.

To do this, we need to approximate Φ −1 (η).

This can be done using the inequality Φ(α) = 1 2π The quantity on the left will be greater than This can be re-arranged to obtain the desired result.

In the case p = 0, we need to use equation 17 from the proof of Lemma 3 in Appendix B, which we restate here

<|TLDR|>

@highlight

This paper identifies classes of problems for which adversarial examples are inescapable, and derives fundamental bounds on the susceptibility of any classifier to adversarial examples. 
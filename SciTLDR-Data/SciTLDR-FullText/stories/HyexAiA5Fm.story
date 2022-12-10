Generative adversarial networks (GANs) are an expressive class of neural generative models with tremendous success in modeling high-dimensional continuous measures.

In this paper, we present a scalable method for unbalanced optimal transport (OT) based on the generative-adversarial framework.

We formulate unbalanced OT as a problem of simultaneously learning a transport map and a scaling factor that push a source measure to a target measure in a cost-optimal manner.

We provide theoretical justification for this formulation, showing that it is closely related to an existing static formulation by Liero et al. (2018).

We then propose an algorithm for solving this problem based on stochastic alternating gradient updates, similar in practice to GANs, and perform numerical experiments demonstrating how this methodology can be applied to population modeling.

We consider the problem of unbalanced optimal transport: given two measures, find a cost-optimal way to transform one measure to the other using a combination of mass variation and transport.

Such problems arise, for example, when modeling the transformation of a source population into a target population ( Figure 1a ).

In this setting, one needs to model mass transport to account for the features that are evolving, as well as local mass variations to allow sub-populations to become more or less prominent in the target population BID34 .Classical optimal transport (OT) considers the problem of pushing a source to a target distribution in a way that is optimal with respect to some transport cost without allowing for mass variations.

Modern approaches are based on the Kantorovich formulation BID24 , which seeks the optimal probabilistic coupling between measures and can be solved using linear programming methods for discrete measures.

Recently, BID13 showed that regularizing the objective using an entropy term allows the dual problem to be solved more efficiently using the Sinkhorn algorithm.

Stochastic methods based on the dual objective have been proposed for the continuous setting BID19 .

Optimal transport has been applied to many areas, such as computer graphics BID17 BID36 and domain adaptation (Courty et al., 2014; .In many applications where a transport cost is not available, transport maps can also be learned using generative models such as generative adversarial networks (GANs) BID20 , which push a source distribution to a target distribution by training against an adversary.

Numerous transport problems in image translation BID30 BID38 BID37 , natural language translation BID22 , domain adaptation BID5 and biological data integration (Amodio & (a) (b)Figure 1: (a) Illustration of the problem of modeling the transformation of a source population µ to a target population ν.

In this example, one sub-population is growing more rapidly than the others.

(b) Schematic of Monge-like formulations of unbalanced optimal transport.

The objective is to learn a transport map T (for transporting mass) and scaling factor ξ (for mass variation) to push the source µ to the target ν, using a deterministic transport map (top) BID7 or a stochastic transport map (bottom).Krishnaswamy, 2018) have been tackled using variants of GANs, with strategies such as conditioning or cycle-consistency employed to enforce correspondence between original and transported samples.

However, all these methods conserve mass between the source and target and therefore cannot handle mass variation.

Several formulations have been proposed for extending the theory of OT to the setting where the measures can have unbalanced masses BID7 BID26 BID27 BID18 .

In terms of numerical methods, a class of scaling algorithms BID8 ) that generalize the Sinkhorn algorithm for balanced OT have been developed for approximating the solution to optimal entropy-transport problems; this formulation of unbalanced OT by BID27 corresponds to the Kantorovich OT problem in which the hard marginal constraints are relaxed using divergences to allow for mass variation.

In practice, these algorithms have been used to approximate unbalanced transport plans between discrete measures for applications such as computer graphics BID8 , tumor growth modeling BID6 and computational biology BID34 .

However, while optimal entropy-transport allows mass variation, it cannot explicitly model it, and there are currently no methods that can perform unbalanced OT between continuous measures.

Contributions.

Inspired by the recent successes of GANs for high-dimensional transport problems, we present a novel framework for unbalanced optimal transport that directly models mass variation in addition to transport.

Concretely, our contributions are the following:• We propose to solve a Monge-like formulation of unbalanced OT, in which the goal is to learn a stochastic transport map and scaling factor to push a source to a target measure in a cost-optimal manner.

This generalizes the unbalanced Monge OT problem by BID7 .•

By relaxing this problem, we obtain an alternative form of the optimal entropy-transport problem by BID27 , which confers desirable theoretical properties.• We develop scalable methodology for solving the relaxed problem.

Our derivation uses a convex conjugate representation of divergences, resulting in an alternating gradient descent method similar to GANs BID20 ).•

We demonstrate in practice how our methodology can be applied towards population modeling using the MNIST and USPS handwritten digits datasets, the CelebA dataset, and a recent single-cell RNA-seq dataset from zebrafish embrogenesis.

In addition to these main contributions, for completeness we also propose a new scalable method (Algorithm 2) in the Appendix for solving the optimal-entropy transport problem by BID27 in the continuous setting.

The algorithm extends the work of to unbalanced OT and is a scalable alternative to the algorithm of BID8 for very large or continuous datasets.

Notation.

Let X , Y ⊆ R n be topological spaces and let B denote the Borel σ-algebra.

Let M 1 + (X ), M + (X ) denote respectively the space of probability measures and finite non-negative measures over X .

For a measurable function T , let T # denote its pushforward operator: if µ is a measure, then T # µ is the pushforward measure of µ under T .

Finally, let π X , π Y be functions that project onto X and Y; for a joint measure γ ∈ M + (X × Y), π X # γ and π Y # γ are its marginals with respect to X and Y respectively.

Optimal transport (OT) addresses the problem of transporting between measures in a cost-optimal manner.

Monge (1781) formulated this problem as a search over deterministic transport maps.

Specifically, given DISPLAYFORM0 subject to the constraint T # µ = ν.

While the optimal T has an intuitive interpretation as an optimal transport map, the Monge problem is non-convex and not always feasible depending on the choices of µ and ν.

The Kantorovich OT problem is a convex relaxation of the Monge problem that formulates OT as a search over probabilistic transport plans.

DISPLAYFORM1 Note that the conditional probability distributions γ y|x specify stochastic maps from X to Y and can be considered a "one-to-many" version of the deterministic map from the Monge problem.

In terms of numerical methods, the relaxed problem is a linear program that is always feasible and can be solved in O(n 3 ) time for discrete µ, ν.

BID13 recently showed that introducing entropic regularization results in a simpler dual optimization problem that can be solved efficiently using the Sinkhorn algorithm.

Based on the entropyregularized dual problem, BID19 and proposed stochastic algorithms for computing transport plans that can handle continuous measures.

Unbalanced OT.

Several formulations that extend classical OT to handle mass variation have been proposed BID7 BID26 .

Existing numerical methods are based on a Kantorovichlike formulation known as optimal-entropy transport BID27 .

This formulation is obtained by relaxing the marginal constraints of (2) using divergences as follows: given two positive measures µ ∈ M + (X ) and ν ∈ M + (Y) and a cost function c : X × Y → R + , optimal entropy-transport finds a measure γ ∈ M + (X × Y) that minimizes DISPLAYFORM2 where DISPLAYFORM3 where ψ ∞ := lim s→∞ ψ(s) s and dP dQ Q + P ⊥ is the Lebesgue decomposition of P with respect to Q. Note that mass variation is allowed since the marginals of γ are not constrained to be µ and ν.

In terms of numerical methods, the state-of-the-art in the discrete setting is a class of iterative scaling algorithms BID8 that generalize the Sinkhorn algorithm for computing regularized OT plans BID13 .

There are no practical algorithms for unbalanced OT between continuous measures, especially in high-dimensional spaces.

In this section, we propose the first algorithm for unbalanced OT that directly models mass variation and can be applied towards transport between high-dimensional continuous measures.

The starting point of our development is the following Monge-like formulation of unbalanced OT, in which the goal is to learn a stochastic transport map and scaling factor to push a source to a target measure in a cost-optimal manner.

Unbalanced Monge OT.

Let c 1 : X × Y → R + be the cost of transport and c 2 : R + → R + the cost of mass variation.

Let the probability space (Z, B(Z), λ) be the source of randomness in the transport map T .

Given two positive measures µ ∈ M + (X ) and ν ∈ M + (Y), we seek a transport map T : X × Z → Y and a scaling factor ξ : DISPLAYFORM0 subject to the constraint T # (ξµ × λ) = ν.

Concretely, the first and second terms of (5) penalize the cost of mass transport and variation respectively, and the equality constraint ensures that (T, ξ) pushes µ to ν exactly.

A special case of (5) is the unbalanced Monge OT problem by BID7 , which employs a deterministic transport map ( Figure 1b) .

We consider the more general case of stochastic (i.e. one-to-many) maps because it is a more suitable model for many practical problems.

For example, in cell biology, it is natural to think of one cell in a source population as potentially giving rise to multiple cells in a target population.

In practice, one can take Z = R n and λ to be the standard Gaussian measure if a stochastic map is desired; otherwise λ can be set to a deterministic distribution.

The following are examples of problems that can be modeled using unbalanced Monge OT.

Example 3.1 FIG0 ).

Suppose the objective is to model the transformation from a source measure (Column 1) to the target measure (Column 2), which represent a population of interest at two distinct time points.

The transport map T models the transport/movement of points from the source to the target, while the scaling factor ξ models the growth (replication) or shrinkage (death) of these points.

Different models of transformation are optimal depending on the relative costs of mass transport and variation (Columns 3-6).

Example 3.2 FIG0 ).

Suppose the objective is to transport points from a source measure (1st panel, color) to a target measure (1st panel, grey) in the presence of class imbalances.

A pure transport map would muddle together points from different classes, while an unbalanced transport map with a scaling factor is able to ameliorate the class imbalance (2nd panel).

In this case, the scaling factor tells us explicitly how to downweigh or upweigh samples in the source distribution to balance the classes with the target distribution (3rd panel).Relaxation.

From an optimization standpoint, it is challenging to satisfy the constraint T # (ξµ × λ) = ν.

We hence consider the following relaxation of (5) using a divergence penalty in place of the equality constraint: DISPLAYFORM1 using an appropriate choice of ψ that satisfies the requirements of Lemma C.2 in the Appendix 1 .This relaxation is the Monge-like version of the optimal-entropy transport problem (3) by BID27 .

Specifically, (T, ξ) specifies a joint measure γ ∈ M + (X × Y) given by DISPLAYFORM2 and by reformulating (6) in terms of γ instead of (T, ξ), one obtains the objective function for optimal-entropy transport.

The main difference between the formulations is their search space, since not all joint measures γ ∈ M + (X × Y) can be specified by some choice of (T, ξ).

For example, if T is a deterministic transport map, then γ is necessarily restricted to the set of deterministic couplings.

Even if T is sufficiently random, it is generally not possible to specify all joint measures γ ∈ M + (X × Y): in the asymmetric Monge formulation (6), all the mass transported to Y must come from somewhere within the support of µ, since the scaling factor ξ allows mass to grow but not to materialize outside of its original support.

Therefore equivalence can be established in general only when restricting the support of γ to supp(µ) × Y as described in the following lemma, whose proof is given in the Appendix.

Lemma 3.3.

Let G be the set of joint measures supported on supp(µ) × Y, and definẽ DISPLAYFORM3 Based on the relation between (3) and (6), several theoretical results for (6) follow from the analysis of optimal entropy-transport by BID27 .

Importantly, one can show the following theorem, namely that for an appropriate and sufficiently large choice of divergence penalty, solutions of the relaxed problem (6) converge to solutions of the original problem (5).

The proof is given in the Appendix.

Theorem 3.4.

Suppose c 1 , c 2 , ψ satisfy the existence assumptions of Proposition B.1 in the Appendix, and let (Z, B(Z), λ) be an atomless probability space.

Furthermore, let ψ be uniquely minimized at ψ(1) = 0.

Then for a sequence 0 < ζ DISPLAYFORM4 .

Additionally, let γ k be the joint measure specified by a minimizer of L ζ k ψ (µ, ν).

If L(µ, ν) < ∞, then up to extraction of a subsequence, γ k converges weakly to γ, the joint measure specified by a minimizer of L(µ, ν).Algorithm.

Using the relaxation of unbalanced Monge OT in (6), we now show that the transport map and scaling factor can be learned by stochastic gradient methods.

While the divergence term cannot easily be minimized using the definition in (4), we can write it as a penalty witnessed by an adversary function f : Y → (−∞, ψ ∞ ] using the convex conjugate representation (see Lemma B.2): DISPLAYFORM5 where ψ * is the convex conjugate of ψ.

The objective in (6) can now be optimized using alternating stochastic gradient updates after parameterizing T , ξ, and f with neural networks; see Algorithm 1 2 .

The optimization procedure is similar to GAN training and can be interpreted as an adversarial game between (T, ξ) and f :• T takes a point x ∼ µ and transports it from X to Y by generating T (x, z) where z ∼ λ.• ξ determines the importance weight of each transported point.• Their shared objective is to minimize the divergence between transported samples and real samples from ν that is measured by the adversary f .•

Additionally, cost functions c 1 and c 2 encourage T, ξ to find the most cost-efficient strategy.

Input: Initial parameters θ, φ, ω; step size η; normalized measuresμ,ν, constants c µ , c ν .

DISPLAYFORM0 Update ω by gradient descent on − (θ, φ, ω).

Update θ, φ by gradient descent (θ, φ, ω). end while Table 1 in the Appendix provides some examples of divergences with corresponding entropy functions and convex conjugates that can be plugged into (7).

Further practical considerations for implementation and training are discussed in Appendix C.Relation to other approaches.

The probabilistic Monge-like formulation (6) is similar to the Kantorovichlike entropy-transport problem (3) in theory, but they result in quite different numerical methods in practice.

Algorithm 1 solves the non-convex formulation (6) and learns a transport map T and scaling factor ξ parameterized by neural networks, enabling scalable optimization using stochastic gradient descent.

The networks are immediately useful for many practical applications; for instance, it only requires a single forward pass to compute the transport and scaling of a point from the source domain to the target.

Furthermore, the neural architectures of T, ξ imbue their function classes with a particular structure, and when chosen appropriately, enable effective learning of these functions in high-dimensional settings.

Due to the non-convexity of the optimization problem, however, Algorithm 1 is not guaranteed to find the global optimum.

In contrast, the scaling algorithm of BID8 based on (3) solves a convex optimization problem and is proven to converge, but is currently only practical for discrete problems and has limited scalability.

For completeness, in Section A of the Appendix, we propose a new stochastic method based on the same dual objective as BID8 that can handle transport between continuous measures (Algorithm 2 in the Appendix).

This method generalizes the approach of for handling transport between continuous measures and overcomes the scalability limitations of BID8 .

However, the output is in the form of the dual solution, which is less interpretable for practical applications compared to the output of Algorithm 1.

In particular, while one can compute a deterministic transport map known as a barycentric projection from the dual solution, it is unclear how best to obtain a scaling factor or a stochastic transport map that can generate samples outside of the target dataset.

In the numerical experiments of Section 4, we show the advantage of directly learning a transport map and scaling factor using Algorithm 1.The problem of learning a scaling factor (or weighting factor) that "balances" measures µ and ν also arises in causal inference.

Generally, µ is the distribution of covariates from a control population and ν is the distribution from a treated population.

The goal is to scale the importance of different members from the DISPLAYFORM1 Figure 3: Learning weights on MNIST data using unbalanced OT.control population based on how likely they are to be present in the treated population, in order to eliminate selection biases in the inference of treatment effects.

BID23 proposed a generative-adversarial method for learning the scaling factor, but they do not consider transport.

In this section, we illustrate in practice how Algorithm 1 performs unbalanced OT, with applications geared towards population modeling.

MNIST-to-MNIST.

We first apply Algorithm 1 to perform unbalanced optimal transport between two modified MNIST datasets.

The source dataset consists of regular MNIST digits with the class distribution shown in column 1 of FIG4 .

The target dataset consists of either regular (for the experiment in FIG4 ) or dimmed (for the experiment in FIG4 ) MNIST digits with the class distribution shown in column 2 of FIG4 .The class imbalance between the source and target datasets imitates a scenerio in which certain classes (digits 0-3) become more popular and others (6-9) become less popular in the target population, while the change in brightness is meant to reflect population drift.

We evaluated Algorithm 1 on the problem of transporting the source distribution to the target distribution, enforcing a high cost of transport (w.r.t.

Euclidean distance).

In both cases, we found that the scaling factor over each of the digit classes roughly reflects its ratio of imbalance between the source and target distributions FIG4 .

These experiments validate that the scaling factor learned by Algorithm 1 reflects the class imbalances and can be used to model growth or decline of different classes in a population.

FIG4 is a schematic illustrating the reweighting that occurs during unbalanced OT.MNIST-to-USPS.

Next, we apply unbalanced OT from the MNIST dataset to the USPS dataset.

As before, these two datasets are meant to imitate a population sampled at two different time points, this time with a large degree of evolution.

We use Algorithm 1 to model the evolution of the MNIST distribution to the USPS distribution, taking as transport cost the Euclidean distance between the original and transported images.

A summary of the unbalanced transport is visualized in FIG1 .

Each arrow originates from a real MNIST image and points towards the predicted appearance of this image in the USPS dataset.

The size of the image reflects the scaling factor of the original MNIST image, i.e. whether it is relatively increasing or decreasing in prominence in the USPS dataset compared to the MNIST dataset according to the unbalanced OT model.

Even though the Euclidean distance is not an ideal measure of correspondence between MNIST and USPS digits, many MNIST digits were able to preserve their likeness during the transport FIG1 ).

We analyzed which MNIST digits were considered as increasing or decreasing in prominence by the model.

The MNIST digits with higher scaling factors were generally brighter ( FIG1 ) and covered a larger area of pixels FIG1 ) compared to the MNIST digits with lower scaling factors.

These results are consistent with the observation that the target USPS digits are generally brighter and contain more pixels.

CelebA-Young-to-CelebA-Aged.

We applied Algorithm 1 on the CelebA dataset to perform unbalanced OT from the population of young faces to the population of aged faces.

This synthetic problem imitates a real application of interest, which is modeling the transformation of a population based on samples taken from two timepoints.

Since the Euclidean distance between two faces is a poor measure of semantic similarity, we first train a variational autoencoder (VAE) (Kingma & Welling, 2013) on the CelebA dataset and encode all samples into the latent space.

We then apply Algorithm 1 to perform unbalanced OT from the encoded young to the encoded aged faces, taking the transport cost to be the Euclidean distance in the latent space.

A summary of the unbalanced transport is visualized in FIG2 .

Each arrow originates from a real face from the young population and points towards the predicted appearance of this face in the aged population.

Generally, the transported faces retain the most salient features of the original faces FIG2 ), although there are exceptions (e.g. gender swaps) which reflects that some features are not prominent components of the VAE encodings.

Interestingly, the young faces with higher scaling factors were significantly enriched for males compared to young faces with lower scaling factors; 9.6% (9,913/103,287) of young female faces had a high scaling factor as compared to 18.5% (8,029/53,447) for young male faces FIG2 , top, p = 0).

In other words, our model predicts growth in the prominence of male faces compared to female faces as the CelebA population evolves from young to aged.

After observing this phenomenon, we confirmed based on checking the ground truth labels that there was indeed a strong gender imbalance between the young and aged populations: while the young population is predominantly female, the aged population is predominantly male FIG2 , bottom).Zebrafish embroygenesis.

A problem of great interest in biology is lineage tracing of cells between different developmental stages or during disease progression.

This is a natural application of transport in which the source and target distributions are unbalanced: some cells in the earlier stage are more poised to develop into cells seen in the later stage.

To showcase the relevance of learning the scaling factor, we apply Algorithm 1 to recent single-cell gene expression data from two stages of zebrafish embryogenesis BID16 .

The source population is from a late stage of blastulation and the target population from an early stage of gastrulation ( FIG3 ).

The results of the transport are plotted in FIG3 -c after dimensionality reduction by PCA and T-SNE BID28 .

To assess the scaling factor, we extracted the cells from the blastula stage with higher scaling factors (i.e. over 90th percentile) and compared them to the remainder of the cells using differential gene expression analysis, producing a ranked list of upregulated genes.

Using the GOrilla tool BID15 , we found that the cells with higher scaling factors were significantly enriched for genes associated with differentiation and development of the mesoderm FIG3 ).

This experiment shows that analysis of the scaling factor can be applied towards interesting and meaningful biological discovery.

In this section, we present a stochastic method for unbalanced OT based on the regularized dual formulation of BID7 , which can be considered a natural generalization of .

The dual formulation of FORMULA2 is given by DISPLAYFORM0 where the supremum is taken over functions u : DISPLAYFORM1 .

This is a constrained optimization problem that is challenging to solve.

A standard technique for making the dual problem unconstrained is to add a strongly convex regularization term to the primal objective , such as an entropic regularization term BID13 : DISPLAYFORM2 where > 0.

Concretely, this term has a "smoothing" effect on the transport plan, in the sense that it encourages plans with high entropy.

By the Fenchel-Rockafellar theorem, the dual of the regularized problem is given by, DISPLAYFORM3 where the supremum is taken over functions u : DISPLAYFORM4 , and the relationship between the primal optimizer γ * and dual optimizer (u * , v * ) is given by DISPLAYFORM5 Next, we rewrite (9) in terms of expectations.

We assume that one has access to samples from µ, ν, and in the setting where µ, ν are not normalized, then samples to the normalized measuresμ,ν as well as the normalization constants.

Based on these assumptions, we have DISPLAYFORM6 If ψ * 1 , ψ * 2 are differentiable, we can parameterize u, v with neural networks u θ , v φ and optimize θ, φ using stochastic gradient descent.

This is described in Algorithm 2.

Note that this algorithm is a generalization of the algorithm of from classical OT to unbalanced OT.

Indeed, taking ψ 1 , ψ 2 to be equality constraints, (9) becomes DISPLAYFORM7 which is the dual of the entropy-regularized classical OT problem.

Algorithm 2 SGD for Unbalanced OT Input: Initial parameters θ, φ; step size η; regularization parameter ; constants c µ , c ν and normalized measuresμ,ν Output: Updated parameters θ, φ while (θ, φ) not converged do DISPLAYFORM8 Update θ, φ by gradient descent on (θ, φ) end whileThe dual solution (u * , v * ) learned from Algorithm 2 can be used to reconstruct the primal solution γ * based on the relation in (10).

Concretely, γ * is a transport map that indicates the amount of mass transported between every pair of points in X and Y. Note that the marginals of γ * with respect to X and Y are not necessarily µ and ν, which is where mass variation is implicitly built into the problem.

Given γ * , it is possible to also learn an "averaged" deterministic mapping from X to Y. A standard approach is to take the barycentric projection T : X → Y, defined as, proposed a stochastic algorithm for learning such a map from the dual solution, which we reproduce in Algorithm 3.

DISPLAYFORM9 DISPLAYFORM10

Input: Learned functions u, v; initial DISPLAYFORM0 Update T θ by gradient descent on (θ) end while B SUPPLEMENT TO SECTION 3

The objectives in (6) and (3) are equivalent if one reformulates (6) in terms of γ instead of (T, ξ), where γ ∈ M + (X × Y) is a joint measure given by DISPLAYFORM0 Furthermore, the formulations are equivalent if one restricts the search space of (3) to contain only those joint measures that can be specified by some (T, ξ).

This relation between the formulations is formalized by Lemma 3.3.Proof of Lemma 3.3.

First we show L ψ (µ, ν) ≥W c1,c2,ψ (µ, ν).

If L ψ (µ, ν) = ∞, this is trivial, so assume L ψ (µ, ν) < ∞. Let (T, ξ) be any solution and define γ by (12).

Note by this definition that DISPLAYFORM1 i.e. ξ is the Radon-Nikodym derivative of π X # γ with respect to µ. Also DISPLAYFORM2 It follows that DISPLAYFORM3 (by definition of γ, linearity and monotone convergence) DISPLAYFORM4 (since ξ is the Radon-Nikodym derivative) DISPLAYFORM5 ≥W c,ψ1,ψ2 (µ, ν).

Since this inequality holds for any (T, ξ), taking the infimum over the left-hand side yields DISPLAYFORM6 To show L ψ (µ, ν) ≤W c,ψ1,ψ2 (µ, ν), assumeW c,ψ1,ψ2 (µ, ν) < ∞ and let γ be any solution.

By the disintegration theorem, there exists a family of probability measures {γ y|x DISPLAYFORM7 Since (Z, B(Z), λ) is atomless, it follows from Proposition 9.1.2 and Theorem 13.1.1 in BID14 ) that there exists a family of measurable functions {T x : Z → Y} x∈X such that γ y|x is the pushforward measure of λ under T x for all x ∈ X .

Denoting T (x, z) : (x, z) → T x (z), then by a change of variables, DISPLAYFORM8 By hypothesis, π X # γ is restricted to the support of µ, i.e. π X # γ ≪ µ. Let ξ be the Radon-Nikodym derivative DISPLAYFORM9 dµ .

It follows from the Radon-Nikodym theorem that (T, ξ) satisfy DISPLAYFORM10 which is the same relation as in (12).

Same as before, DISPLAYFORM11 It then follows that DISPLAYFORM12 (by Fubini's Theorem for Markov kernels) DISPLAYFORM13 (by change of variables) DISPLAYFORM14 Since this inequality holds for any γ, this implies thatW c1,c2,ψ (µ, ν) ≥ L ψ (µ, ν), which completes the proof.

Due to the near equivalence of the formulations, several theoretical results for (6) follow from the analysis of optimal entropy-transport by BID27 , such as the following existence and uniqueness result: DISPLAYFORM15 DISPLAYFORM16 If c 2 , ψ are strictly convex, ψ ∞ = ∞ and c 1 satisfies Corollary 3.6 BID27 , then the joint measure γ specified by any minimizer of L ψ (µ, ν) is unique.

Proof of Proposition B.1.

Note thatW c1,c2,ψ (µ, ν) is equivalent to W c1,c2,ψ (µ, ν) when X is restricted to the support of µ. If c 1 , c 2 , ψ satisfy (i) or (ii), they also satisfy (i) and (ii) when X is restricted to the support of µ. By Theorem 3.3 of BID27 ,W c1,c2,ψ (µ, ν) has a minimizer.

It follows from the construction of the proof of Lemma 3.3 that a minimizer of L ψ (µ, ν) also exists.

For uniqueness, if ψ ∞ = ∞, then it follows from Lemma 3.5 of BID27 and the fact that minimizers are restricted to G that the marginals π X # γ, π Y # γ are uniquely determined for any solution γ ofW c1,c2,ψ (µ, ν).

The uniqueness of γ then follows from the proof of Corollary 3.6 in BID27 .

It follows from the construction of the proof of Lemma 3.3 that the product measure generated by the minimizers of L ψ (µ, ν) is unique, which completes the proof.

For certain cost functions and divergences, it can be shown that L ψ defines a proper metric between positive measures µ and ν, i.e. taking c 2 , ψ to be entropy functions corresponding to the KL-divergence and c 1 = log cos 2 + (d(x, y)), then L ψ (µ, ν) corresponds to the Hellinger-Kantorovich BID27 or the Wasserstein-Fisher-Rao BID9 metric between positive measures µ and ν.

Based on Lemma 3.3, the theoretical analysis of BID27 , and standard results on constrained optimization, it can be shown that for an appropriate and sufficiently large choice of divergence penalty, solutions of the relaxed problem (6) converge to solutions of the original problem (5) (Theorem 3.4).Proof of Theorem 3.4.

Since ζ k ψ(s) converges pointwise to the equality constraint ι = (s), which is 0 for s = 1 and ∞ otherwise, by Lemma 3.9 in BID27 , we have that lim inf k→∞Wc 1,c2 ,ζ k ψ (µ, ν) ≥ W c1,c2,ι= (µ, ν).

Additionally,W c1,c2,ζ k ψ (µ, ν) ≤W c1,c2,ι= (µ, ν) for any value of k since for any minimizer γ ofW c1,c2,ι= (µ, ν), it holds that π DISPLAYFORM17 for all k. Therefore, lim k→∞Wc 1 ,c2,ζ k ψ (µ, ν) =W c1,c2,ι= (µ, ν), which then by Lemma 3.3 implies the first part of the proposition.

For the second part, by the hypothesis we have thatW c1,c2,ι= (µ, ν) = C < ∞ and as a consequencẽ W c1,c2,ζ k ψ (µ, ν) ≤ C for all k. Hence, by Proposition 2.10 in BID27 , the sequence of minimizers γ k is bounded.

If the assumptions of Proposition B.1 are satisfied, then the sequence γ k is equally tight.

For assumption (ii) this follows by Proposition 2.10 in BID27 and for assumption (i) this follows by the Markov inequality: for any λ > 0, DISPLAYFORM18 Since γ k are bounded and equally tight, by an extension of Prokhorov's theorem (Theorem 2.2 of BID27 ), there exists a subsequence of γ k that is weakly convergent to someγ.

Then by lower semicontinuity, we obtain that DISPLAYFORM19 |ν) = 0 by lower semicontinuity.

Therefore,γ is a minimizer ofW c1,c2,ι= (µ, ν).

By construction of the proof of Lemma 3.3, γ k is equivalent to the product measure induced by minimizers of L ζ k ψ (µ, ν), which implies the second part of the proposition.

In this section, we present the convex conjugate form of ψ-divergence used to rewrite the main objective as a min-max problem.

Lemma B.2.

For non-negative finite measures P, Q over T ⊂ R d , it holds that DISPLAYFORM0 where F is a subset of measurable functions {f : T → (−∞, ψ ∞ ]}.

Equality holds if and only if ∃f ∈ F such that (i) the restriction of f to the support of Q belongs to the subdifferential of ψ( dP dQ ), i.e. the Radon-Nikodym derivative of P with respect to Q and (ii) f = ψ ∞ over the support of P ⊥ .We provide a simple proof of this result.

A similar result under stronger assumptions was shown in BID32 and used by BID33 for generative modeling.

A rigorous proof can be found in BID27 .Proof of Lemma B.2.

Note that DISPLAYFORM1 (by defintion of convex conjugate) DISPLAYFORM2 By first-order optimality conditions, the optimal f over the support of Q is obtained when dP dQ belongs to the subdifferential of ψ * (f ), or equivalently when f belongs to the subdifferential of ψ( dP dQ ).

It is straightforward to see that the optimal f over the support of P ⊥ is equal to ψ ∞ , which completes the proof.

Proof.

DISPLAYFORM3

Choice of cost functions.

Proposition B.1 gives sufficient conditions on c 1 , c 2 for the problem to be wellposed.

In practice, it is often convenient the cost of transport, c 1 , to be some measurement of correspondence between X and Y. For example, we can take c 1 (x, y) to be the Euclidean distance between x and y after mapping them to some common feature space.

For the cost of mass adjustment, c 2 , it is generally sensible to choose some convex function that vanishes at 1 (i.e. no mass adjustment) and such that lim x→0 c 2 (x) = lim x→∞ c 2 (x) = ∞ to prevent ξ from becoming too small or too large.

Any of the entropy functions shown in Table 1 ) − log(2 − e s ) log 2 log(2) − log(1 + e −s )) Table 1: Table of some common ψ-divergences, associated entropy functions ψ, and convex conjugates ψ * for Algorithm 1, partly adapted from BID33 .Choice of ψ.

In BID33 , it was shown that any ψ-divergence could be used to train generative models, i.e. to match a generated distribution P to a true data distribution Q.

This is due to Jensen's inequality: for any convex lower semi-continous entropy function ψ, D ψ (P |Q) is uniquely minimized when P = Q, where P, Q are probability measures.

However, this does not generally hold when P, Q are not probability measures, as illustrated by the following example.

Example C.1.

In the original GAN paper, the discrminative objective, sup f log f (x)dP (x) − log(1 − f (x))dQ(x), corresponds to D ψ (P |Q) with ψ(s) = s log s − (s + 1) log(s + 1) BID33 .

If P, Q are probability measures, this divergence is equivalent to the Jensen-Shannon divergence and is minimized when P = Q. If P, Q are non-negative measures with unconstrained total mass, the divergence is minimized when P = ∞ and Q = 0.When P, Q are not probability measures, we require an additional constraint on ψ to ensure that divergence minimization matchces P to Q: Lemma C.2.

Suppose P, Q are non-negative finite measures over T ⊆ R n .

If ψ(s) attains a unique minimum at s = 1 with ψ(1) = 0 and ψ ∞ > 0, then D ψ (P |Q) = 0 ⇒ P = Q. Otherwise, then P = Q in general when D ψ (P |Q) is minimized.

Proof.

Suppose ψ(s) attains a unique minimum at s = 1 with ψ(1) = 0, ψ ∞ > 0, and P = Q over a region with positive measure.

It is straightforward to see by the definition in (4) that D ψ (P |Q) > 0, since at least one of the two terms will be strictly positive.

Therefore, the first statement holds.

For the second statement, suppose either ψ(s) does not attain a unique minimum at s = 1 or ψ ∞ ≤ 0.

If ψ(s) attains a minimum at some s = 1, then taking P = s Q results in a divergence that is equal to or less than P = Q. If ψ ∞ ≤ 0, then letting P = Q + P ⊥ where P ⊥ is a positive measure orthogonal to Q results in a divergence that is equal to or less than P = Q. Table 1 provides some examples of ψ corresponding to common divergences that can be used for unbalanced OT.Choice of f .

According to Lemma B.2, f should belong to a class of functions that maps from Y to (−∞, ψ ∞ ].

In practice, this can be enforced by parameterizing f using a neural network with a final layer that maps to the correct range, also known as an output activation layer BID33 .

Table 1 provides some examples of activation layers that can be used.

Choice of neural architectures.

For our experiments in Section 4, we used fully-connected feedforward networks with 3 hidden layers and ReLU activations.

For T , the output activation layer was a sigmoid function to map the final pixel brightness to the range (0, 1).

For ξ, the output activation layer was a softplus function to map the scaling factor weight to the range (0, ∞).

@highlight

We propose new methodology for unbalanced optimal transport using generative adversarial networks.

@highlight

The authors consider the unbalanced optimal transport problem between two measures with different total mass using a stochastic min-max algorithm and local scaling

@highlight

The authors propose an approach to estimate unbalanced optimal transport between sampled measures that scales well in the dimension and in the number of samples.

@highlight

The paper introduces a static formulation for unbalanced optimal transport by learning simultaneously a transport map T and scaling factor xi.
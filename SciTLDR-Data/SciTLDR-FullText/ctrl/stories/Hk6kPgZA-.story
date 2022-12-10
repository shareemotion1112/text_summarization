Neural networks are vulnerable to adversarial examples and researchers have proposed many heuristic attack and defense mechanisms.

We address this problem through the principled lens of distributionally robust optimization, which guarantees performance under adversarial input perturbations.

By considering a Lagrangian penalty formulation of perturbing the underlying data distribution in a Wasserstein ball, we provide a training procedure that augments model parameter updates with worst-case perturbations of training data.

For smooth losses, our procedure provably achieves moderate levels of robustness with little computational or statistical cost relative to empirical risk minimization.

Furthermore, our statistical guarantees allow us to efficiently certify robustness for the population loss.

For imperceptible perturbations, our method matches or outperforms heuristic approaches.

Consider the classical supervised learning problem, in which we minimize an expected loss E P0 [ (θ; Z)] over a parameter θ ∈ Θ, where Z ∼ P 0 , P 0 is a distribution on a space Z, and is a loss function.

In many systems, robustness to changes in the data-generating distribution P 0 is desirable, whether they be from covariate shifts, changes in the underlying domain BID2 , or adversarial attacks BID22 BID29 .

As deep networks become prevalent in modern performance-critical systems (perception for self-driving cars, automated detection of tumors), model failure is increasingly costly; in these situations, it is irresponsible to deploy models whose robustness and failure modes we do not understand or cannot certify.

Recent work shows that neural networks are vulnerable to adversarial examples; seemingly imperceptible perturbations to data can lead to misbehavior of the model, such as misclassification of the output BID22 BID40 BID29 BID36 .

Consequently, researchers have proposed adversarial attack and defense mechanisms BID41 BID53 BID47 BID12 BID23 BID33 BID51 .

These works provide an initial foundation for adversarial training, but it is challenging to rigorously identify the classes of attacks against which they can defend (or if they exist).

Alternative approaches that provide formal verification of deep networks BID24 BID26 are NP-hard in general; they require prohibitive computational expense even on small networks.

Recently, researchers have proposed convex relaxations of the NP-hard verification problem with some success BID28 BID45 , though they may be difficult to scale to large networks.

In this context, our work is situated between these agendas: we develop efficient procedures with rigorous guarantees for small to moderate amounts of robustness.

We take the perspective of distributionally robust optimization and provide an adversarial training procedure with provable guarantees on its computational and statistical performance.

We postulate a class P of distributions around the data-generating distribution P 0 and consider the problem minimize DISPLAYFORM0 The choice of P influences robustness guarantees and computability; we develop robustness sets P with computationally efficient relaxations that apply even when the loss is non-convex.

We provide an adversarial training procedure that, for smooth , enjoys convergence guarantees similar to non-robust approaches while certifying performance even for the worst-case population loss sup P ∈P E P [ (θ; Z)].

On a simple implementation in Tensorflow, our method takes 5-10× as long as stochastic gradient methods for empirical risk minimization (ERM), matching runtimes for other adversarial training procedures BID22 BID29 BID33 .

We show that our procedure-which learns to protect against adversarial perturbations in the training dataset-generalizes, allowing us to train a model that prevents attacks to the test dataset.

We briefly overview our approach.

Let c : Z × Z → R + ∪ {∞}, where c(z, z 0 ) is the "cost" for an adversary to perturb z 0 to z (we typically use c(z, z 0 ) = z − z 0 2 p with p ≥ 1).

We consider the robustness region P = {P : W c (P, P 0 ) ≤ ρ}, a ρ-neighborhood of the distribution P 0 under the Wasserstein metric W c (·, ·) (see Section 2 for a formal definition).

For deep networks and other complex models, this formulation of problem FORMULA0 is intractable with arbitrary ρ.

Instead, we consider its Lagrangian relaxation for a fixed penalty parameter γ ≥ 0, resulting in the reformulation minimize θ∈Θ F (θ) := sup DISPLAYFORM1 where φ γ (θ; z 0 ) := sup z∈Z { (θ; z) − γc(z, z 0 )} .(See Proposition 1 for a rigorous statement of these equalities.)

Here, we have replaced the usual loss (θ; Z) by the robust surrogate φ γ (θ; Z); this surrogate (2b) allows adversarial perturbations of the data z, modulated by the penalty γ.

We typically solve the penalty problem (2) with P 0 replaced by the empirical distribution P n , as P 0 is unknown (we refer to this as the penalty problem below).The key feature of the penalty problem (2) is that moderate levels of robustness-in particular, defense against imperceptible adversarial perturbations-are achievable at essentially no computational or statistical cost for smooth losses .

Specifically, for large enough penalty γ (by duality, small enough robustness ρ), the function z → (θ; z) − γc(z, z 0 ) in the robust surrogate (2b) is strongly concave and hence easy to optimize if (θ, z) is smooth in z. Consequently, stochastic gradient methods applied to problem (2) have similar convergence guarantees as for non-robust methods (ERM).

In Section 3, we provide a certificate of robustness for any ρ; we give an efficiently computable data-dependent upper bound on the worst-case loss sup P :Wc(P,P0)≤ρ E P [ (θ; Z)].

That is, the worst-case performance of the output of our principled adversarial training procedure is guaranteed to be no worse than this certificate.

Our bound is tight when ρ = ρ n , the achieved robustness for the empirical objective.

These results suggest advantages of networks with smooth activations rather than ReLU's.

We experimentally verify our results in Section 4 and show that we match or achieve state-of-the-art performance on a variety of adversarial attacks.

Robust optimization and adversarial training The standard robust-optimization approach minimizes losses of the form sup u∈U (θ; z + u) for some uncertainty set U BID46 BID3 BID54 .

Unfortunately, this approach is intractable except for specially structured losses, such as the composition of a linear and simple convex function BID3 BID54 BID55 .

Nevertheless, this robust approach underlies recent advances in adversarial training BID49 BID22 BID42 BID12 BID33 , which heuristically perturb data during a stochastic optimization procedure.

One such heuristic uses a locally linearized loss function (proposed with p = ∞ as the "fast gradient sign method" BID22 ): DISPLAYFORM2 One form of adversarial training trains on the losses (θ; (x i + ∆ xi (θ), y i )) BID22 BID29 , while others perform iterated variants BID42 BID12 BID33 BID51 .

BID33 observe that these procedures attempt to optimize the objective E P0 [sup u p ≤ (θ; Z + u)], a constrained version of the penalty problem (2).

This notion of robustness is typically intractable: the inner supremum is generally non-concave in u, so it is unclear whether model-fitting with these techniques converges, and there are possibly worst-case perturbations these techniques do not find.

Indeed, it is NP-hard to find worst-case perturbations when deep networks use ReLU activations, suggesting difficulties for fast and iterated heuristics (see Lemma 2 in Appendix B).

Smoothness, which can be obtained in standard deep architectures with exponential linear units (ELU's) BID15 , allows us to find Lagrangian worst-case perturbations with low computational cost.

Distributionally robust optimization To situate the current work, we review some of the substantial body of work on robustness and learning.

The choice of P in the robust objective (1) affects both the richness of the uncertainty set we wish to consider as well as the tractability of the resulting optimization problem.

Previous approaches to distributional robustness have considered finitedimensional parametrizations for P, such as constraint sets for moments, support, or directional deviations BID13 BID16 BID21 , as well as non-parametric distances for probability measures such as f -divergences BID4 BID5 BID30 BID34 , and Wasserstein distances BID48 BID7 .

In constrast to f -divergences (e.g. χ 2 -or Kullback-Leibler divergences) which are effective when the support of the distribution P 0 is fixed, a Wasserstein ball around P 0 includes distributions Q with different support and allows (in a sense) robustness to unseen data.

Many authors have studied tractable classes of uncertainty sets P and losses .

For example, BID4 and BID38 use convex optimization approaches for fdivergence balls.

For worst-case regions P formed by Wasserstein balls, , BID48 , and BID7 show how to convert the saddle-point problem (1) to a regularized ERM problem, but this is possible only for a limited class of convex losses and costs c. In this work, we treat a much larger class of losses and costs and provide direct solution methods for a Lagrangian relaxation of the saddle-point problem (1).

One natural application area is in domain adaptation BID31 ; concurrently with this work, Lee & Raginsky provide guarantees similar to ours for the empirical minimizer of the robust saddle-point problem (1) and give specialized bounds for domain adaptation problems.

In contrast, our approach is to use the distributionally robust approach to both defend against imperceptible adversarial perturbations and develop efficient optimization procedures.

Our approach is based on the following simple insight: assume that the function z → (θ; z) is smooth, meaning there is some L for which ∇ z (θ; ·) is L-Lipschitz.

Then for any c : Z × Z → R + ∪ {∞} 1-strongly convex in its first argument, a Taylor expansion yields DISPLAYFORM0 Thus, whenever the loss is smooth enough in z and the penalty γ is large enough (corresponding to less robustness), computing the surrogate (2b) is a strongly-concave optimization problem.

We leverage the insight (4) to show that as long as we do not require too much robustness, this strong concavity approach (4) provides a computationally efficient and principled approach for robust optimization problems (1).

Our starting point is a duality result for the minimax problem (1) and its Lagrangian relaxation for Wasserstein-based uncertainty sets, which makes the connections between distributional robustness and the "lazy" surrogate (2b) clear.

We then show (Section 2.1) how stochastic gradient descent methods can efficiently find minimizers (in the convex case) or approximate stationary points (when is non-convex) for our relaxed robust problems.

Wasserstein robustness and duality Wasserstein distances define a notion of closeness between distributions.

Let Z ⊂ R m , and let (Z, A, P 0 ) be a probability space.

Let the transportation cost c : Z × Z → [0, ∞) be nonnegative, lower semi-continuous, and satisfy c(z, z) = 0.

For example, for a differentiable convex h : Z → R, the Bregman divergence c(z, z 0 ) = h(z) − h(z 0 ) − ∇h(z 0 ), z − z 0 satisfies these conditions.

For probability measures P and Q supported on Z, let Π(P, Q) denote their couplings, meaning measures M on Z 2 with M (A, Z) = P (A) and DISPLAYFORM1 For ρ ≥ 0 and distribution P 0 , we let P = {P : W c (P, P 0 ) ≤ ρ}, considering the Wasserstein form of the robust problem (1) and its Lagrangian relaxation (2) with γ ≥ 0.

The following duality result BID7 gives the equality (2) for the relaxation and an analogous result for the problem (1).

We give an alternative proof in Appendix C.1 for convex, continuous cost functions.

Proposition 1.

Let : Θ × Z → R and c : Z × Z → R + be continuous.

Let φ γ (θ; z 0 ) = sup z∈Z { (θ; z) − γc(z, z 0 )} be the robust surrogate (2b).

For any distribution Q and any ρ > 0, DISPLAYFORM2 and for any γ ≥ 0, we have DISPLAYFORM3 Leveraging the insight (4), we give up the requirement that we wish a prescribed amount ρ of robustness (solving the worst-case problem (1) for P = {P : W c (P, P 0 ) ≤ ρ}) and focus instead on the Lagrangian penalty problem (2) and its empirical counterpart DISPLAYFORM4

We now develop stochastic gradient-type methods for the relaxed robust problem FORMULA8 , making clear the computational benefits of relaxing the strict robustness requirements of formulation (5).

We begin with assumptions we require, which roughly quantify the amount of robustness we can provide.

Assumption A. The function c : Z ×Z → R + is continuous.

For each z 0 ∈ Z, c(·, z 0 ) is 1-strongly convex with respect to the norm · .To guarantee that the robust surrogate (2b) is tractably computable, we also require a few smoothness assumptions.

Let · * be the dual norm to · ; we abuse notation by using the same norm · on Θ and Z, though the specific norm is clear from context.

Assumption B. The loss : Θ × Z → R satisfies the Lipschitzian smoothness conditions DISPLAYFORM0 These properties guarantee both (i) the well-behavedness of the robust surrogate φ γ and (ii) its efficient computability.

Making point (i) precise, Lemma 1 shows that if γ is large enough and Assumption B holds, the surrogate φ γ is still smooth.

Throughout, we assume Θ ⊆ R d .Lemma 1. Let f : Θ×Z → R be differentiable and λ-strongly concave in z with respect to the norm · , and definef DISPLAYFORM1 , and assume g θ and g z satisfy Assumption B with (θ; z) replaced with f (θ, z).

Thenf is differentiable, and letting z (θ) = argmax z∈Z f (θ, z), we have ∇f (θ) = g θ (θ, z (θ)).

Moreover, DISPLAYFORM2 See Section C.2 for the proof.

Fix z 0 ∈ Z and focus on the 2 -norm case where c(z, z 0 ) satisfies Assumption A with · 2 .

Noting that DISPLAYFORM3 -Lipschitz gradients, and for t = 0, . . .

, T − 1 do Sample z t ∼ P 0 and find an -approximate maximizer z t of (θ DISPLAYFORM4 DISPLAYFORM5 This motivates Algorithm 1, a stochastic-gradient approach for the penalty problem (7).

The benefits of Lagrangian relaxation become clear here: for (θ; z) smooth in z and γ large enough, gradient ascent on (θ t ; z)−γc(z, z t ) in z converges linearly and we can compute (approximate) z t efficiently (we initialize our inner gradient ascent iterations with the sampled natural example z t ).Convergence properties of Algorithm 1 depend on the loss .

When is convex in θ and γ is large enough that z → ( (θ; z) − γc(z, z 0 )) is concave for all (θ, z 0 ) ∈ Θ × Z, we have a stochastic monotone variational inequality, which is efficiently solvable BID25 BID14 with convergence rate 1/ √ T .

When the loss is nonconvex in θ, the following theorem guarantees convergence to a stationary point of problem FORMULA8 at the same rate when γ ≥ L zz .

Recall that F (θ) = E P0 [φ γ (θ; Z)] is the robust surrogate objective for the Lagrangian relaxation (2).

Theorem 2 (Convergence of Nonconvex SGD).

Let Assumptions A and B hold with the 2 -norm and let DISPLAYFORM6 See Section C.3 for the proof.

We make a few remarks.

First, the condition DISPLAYFORM7 2 holds (to within a constant factor) whenever ∇ θ (θ, z) 2 ≤ σ for all θ, z. Theorem 2 shows that the stochastic gradient method achieves the rates of convergence on the penalty problem (7) achievable in standard smooth non-convex optimization BID20 .

The accuracy parameter has a fixed effect on optimization accuracy, independent of T : approximate maximization has limited effects.

Key to the convergence guarantee of Theorem 2 is that the loss is smooth in z: the inner supremum (2b) is NP-hard to compute for non-smooth deep networks (see Lemma 2 in Section B for a proof of this for ReLU's).

The smoothness of is essential so that a penalized version (θ, z) − γc(z, z 0 ) is concave in z (which can be approximately verified by computing Hessians ∇ 2 zz (θ, z) for each training datapoint), allowing computation and our coming certificates of optimality.

Replacing ReLU's with sigmoids or ELU's BID15 allows us to apply Theorem 2, making distributionally robust optimization tractable for deep learning.

In supervised-learning scenarios, we are often interested in adversarial perturbations only to feature vectors (and not labels).

Letting Z = (X, Y ) where X denotes the feature vector (covariates) and Y the label, this is equivalent to defining the Wasserstein cost function c : DISPLAYFORM8 where c x : X × X → R + is the transportation cost for the feature vector X. All of our results suitably generalize to this setting with minor modifications to the robust surrogate (2b) and the above assumptions (see Section D).

Similarly, our distributionally robust framework (2) is general enough to consider adversarial perturbations to only an arbitrary subset of coordinates in Z. For example, it may be appropriate in certain applications to hedge against adversarial perturbations to a small fixed region of an image BID11 .

By suitably modifying the cost function c(z, z ) to take value ∞ outside this small region, our general formulation covers such variants.

From results in the previous section, Algorithm 1 provably learns to protect against adversarial perturbations of the form (7) on the training dataset.

Now we show that such procedures generalize, allowing us to prevent attacks on the test set.

Our subsequent results hold uniformly over the space of parameters θ ∈ Θ, including θ WRM , the output of the stochastic gradient descent procedure in Section 2.1.

Our first main result, presented in Section 3.1, gives a data-dependent upper bound on the population worst-case objective sup P :Wc(P,P0)≤ρ E P [ (θ; Z)] for any arbitrary level of robustness ρ; this bound is optimal for ρ = ρ n , the level of robustness achieved for the empirical distribution by solving (7).

Our bound is efficiently computable and hence certifies a level of robustness for the worst-case population objective.

Second, we show in Section 3.2 that adversarial perturbations on the training set (in a sense) generalize: solving the empirical penalty problem (7) guarantees a similar level of robustness as directly solving its population counterpart (2).

Our main result in this section is a data-dependent upper bound for the worst-case population objective: DISPLAYFORM0 with high probability.

To make this rigorous, fix γ > 0, and consider the worst-case perturbation, typically called the transportation map or Monge map BID53 ), DISPLAYFORM1 Under our assumptions, T γ is easily computable when γ ≥ L zz .

Letting δ z denote the point mass at z, Proposition 1 shows the empirical maximizers of the Lagrangian formulation (6) are attained by DISPLAYFORM2 δ Tγ (θ,Zi) and DISPLAYFORM3 Our results imply, in particular, that the empirical worst-case loss DISPLAYFORM4 gives a certificate of robustness to (population) Wasserstein perturbations up to level ρ n .

E P * n (θ) [ (θ; Z)] is efficiently computable via (10), providing a data-dependent guarantee for the worst-case population loss.

Our bound relies on the usual covering numbers for the model class { (θ; ·) : θ ∈ Θ} as the notion of complexity (e.g. van der BID52 , so, despite the infinite-dimensional problem (7), we retain the same uniform convergence guarantees typical of empirical risk minimization.

Recall that for a set V , a collection v 1 , . . .

, v N is an -cover of V in norm · if for each v ∈ V, there exists DISPLAYFORM5 To ease notation, we let DISPLAYFORM6 where b 1 , b 2 are numerical constants.

We are now ready to state the main result of this section.

We first show from the duality result (6) that we can provide an upper bound for the worst-case population performance for any level of robustness ρ.

For ρ = ρ n (θ) and θ = θ WRM , this certificate is (in a sense) tight as we see below.

Theorem 3.

Assume | (θ; z)| ≤ M for all θ ∈ Θ and z ∈ Z. Then, for a fixed t > 0 and numerical constants b 1 , b 2 > 0, with probability at least 1 − e −t , simultaneously for all θ ∈ Θ, ρ ≥ 0, γ ≥ 0, DISPLAYFORM7 In particular, if ρ = ρ n (θ) then with probability at least 1 − e −t , for all θ ∈ Θ sup DISPLAYFORM8 See Section C.4 for its proof.

We now give a concrete variant of Theorem 3 for Lipschitz functions.

DISPLAYFORM9 , Theorem 3 provides a robustness guarantee scaling linearly with d despite the infinite-dimensional Wasserstein penalty.

Assuming there exist θ 0 ∈ Θ, M θ0 < ∞ such that | (θ 0 ; z)| ≤ M θ0 for all z ∈ Z, we have the following corollary (see proof in Section C.5).Corollary 1.

Let (·; z) be L-Lipschitz with respect to some norm · for all z ∈ Z. Assume that DISPLAYFORM10 Then, the bounds (11) and (12) hold with DISPLAYFORM11 A key consequence of the bound FORMULA0 is that γρ + E Pn [φ γ (θ; Z)] certifies robustness for the worstcase population objective for any ρ and θ.

For a given θ, this certificate is tightest at the achieved level of robustness ρ n (θ), as noted in the refined bound (12) which follows from the duality result DISPLAYFORM12 (See Section C.4 for a proof of these equalities.)

We expect θ WRM , the output of Algorithm 1, to be close to the minimizer of the surrogate loss E Pn [φ γ (θ; Z)] and therefore have the best guarantees.

Most importantly, the certificate FORMULA0 is easy to compute via expression (10): as noted in Section 2.1, the mappings T (θ, Z i ) are efficiently computable for large enough γ, and DISPLAYFORM13 The bounds FORMULA0 - FORMULA0 may be too large-because of their dependence on covering numbers and dimension-for practical use in security-critical applications.

With that said, the strong duality result, Proposition 1, still applies to any distribution.

In particular, given a collection of test examples Z test i, we may interrogate possible losses under perturbations for the test examples by noting that, if P test denotes the empirical distribution on the test set (say, with putative assigned labels), then DISPLAYFORM14 for all γ, ρ ≥ 0.

Whenever γ is large enough (so that this is tight for small ρ), we may efficiently compute the Monge-map (9) and the test loss (15) to guarantee bounds on the sensitivity of a parameter θ to a particular sample and predicted labeling based on the sample.

We can also show that the level of robustness on the training set generalizes.

Our starting point is Lemma 1, which shows that T γ (·; z) is smooth under Assumptions A and B: DISPLAYFORM0 for all θ 1 , θ 2 , where we recall that L zz is the Lipschitz constant of ∇ z (θ; z).

Leveraging this smoothness, we show that ρ n (θ) = E Pn [c(T γ (θ; Z), Z)], the level of robustness achieved for the empirical problem, concentrates uniformly around its population counterpart.

DISPLAYFORM1 If Assumptions A and B hold, then with probability at least 1 − e −t , DISPLAYFORM2 See Section C.6 for the proof.

For DISPLAYFORM3 that the bound (30) gives the usual d/n generalization rate for the distance between adversarial perturbations and natural examples.

Another consequence of Theorem 4 is that ρ n (θ WRM ) in the certificate (12) is positive as long as the loss is not completely invariant to data.

To see this, note from the optimality conditions for DISPLAYFORM4 surely, and hence for large enough n, we have ρ n (θ) > 0 by the bound (30).

Our technique for distributionally robust optimization with adversarial training extends beyond supervised learning.

To that end, we present empirical evaluations on supervised and reinforcement learning tasks where we compare performance with empirical risk minimization (ERM) and, where appropriate, models trained with the fast-gradient method (3) (FGM) BID22 , its iterated variant (IFGM) BID29 , and the projected-gradient method (PGM) BID33 .

PGM augments stochastic gradient steps for the parameter θ with projected gradient ascent over x → (θ; x, y), iterating (for data point x i , y i ) DISPLAYFORM0 for t = 1, . . .

, T adv , where Π denotes projection onto DISPLAYFORM1 The adversarial training literature (e.g. BID22 ) usually considers · ∞ -norm attacks, which allow imperceptible perturbations to all input features.

In most scenarios, however, it is reasonable to defend against weaker adversaries that instead perturb influential features more.

We consider this setting and train against · 2 -norm attacks.

Namely, we use the squared Euclidean cost for the feature vectors c x (x, x ) := x − x 2 2 and define the overall cost as the covariate-shift adversary (8) for WRM (Algorithm 1), and we use p = 2 for FGM, IFGM, PGM training in all experiments; we still test against adversarial perturbations with respect to the norms p = 2, ∞. We use T adv = 15 iterations for all iterative methods (IFGM, PGM, and WRM) in training and attacks.

In Section 4.1, we visualize differences between our approach and ad-hoc methods to illustrate the benefits of certified robustness.

In Section 4.2 we consider a supervised learning problem for MNIST where we adversarially perturb the test data.

Finally, we consider a reinforcement learning problem in Section 4.3, where the Markov decision process used for training differs from that for testing.

WRM enjoys the theoretical guarantees of Sections 2 and 3 for large γ, but for small γ (large adversarial budgets), WRM becomes a heuristic like other methods.

In Appendix A.4, we compare WRM with other methods on attacks with large adversarial budgets.

In Appendix A.5, we further compare WRM-which is trained to defend against · 2 -adversaries-with other heuristics trained to defend against · ∞ -adversaries.

WRM matches or outperforms other heuristics against imperceptible attacks, while it underperforms for attacks with large adversarial budgets.

For our first experiment, we generate synthetic data DISPLAYFORM0 , where X ∈ R 2 and I 2 is the identity matrix in R 2 .

Furthermore, to create a wide margin separating the classes, we remove data with X 2 ∈ ( √ 2/1.3, 1.3 √ 2).

We train a small neural network with 2 hidden layers of size 4 and 2 and either all ReLU or all ELU activations between layers, comparing our approach (WRM) with ERM and the 2-norm FGM.

For our approach we use γ = 2, and to make fair comparisons with FGM we use DISPLAYFORM1 for the fast-gradient perturbation magnitude , where θ WRM is the output of Algorithm 1.1 FIG0 illustrates the classification boundaries for the three training procedures over the ReLUactivated FIG0 ) and ELU-activated FIG0 ) models.

Since 70% of the data are of the blue class ( X 2 ≤ √ 2/1.3), distributional robustness favors pushing the classification boundary outwards; intuitively, adversarial examples are most likely to come from pushing blue points outwards across the boundary.

ERM and FGM suffer from sensitivities to various regions of the data, as evidenced by the lack of symmetry in their classification boundaries.

For both activations, WRM pushes the classification boundaries further outwards than ERM or FGM.

However, WRM with ReLU's still suffers from sensitivities (e.g. radial asymmetry in the classification surface) due to the lack of robustness guarantees.

WRM with ELU's provides a certified level of robustness, yielding an axisymmetric classification boundary that hedges against adversarial perturbations in all directions.

Recall that our certificates of robustness on the worst-case performance given in Theorem 3 applies for any level of robustness ρ.

In Figure 2 (a), we plot our certificate (11) against the out-of-sample (test) worst-case performance sup P :Wc(P,P0)≤ρ E P [ (θ; Z)] for WRM with ELU's.

Since the worstcase loss is hard to evaluate directly, we solve its Lagrangian relaxation (6) for different values of γ adv .

For each γ adv , we consider the distance to adversarial examples in the test dataset DISPLAYFORM2 where P test is the test distribution, c(z, z ) := x − x 2 2 + ∞ · 1 {y = y } as before, and T γ adv (θ, Z) = argmax z { (θ; z) − γ adv c(z, Z)} is the adversarial perturbation of Z (Monge map) for the model θ.

The worst-case losses on the test dataset are then given by DISPLAYFORM3 As anticipated, our certificate is almost tight near the achieved level of robustness ρ n (θ WRM ) for WRM (10) and provides a performance guarantee even for other values of ρ.

We now consider a standard benchmark-training a neural network classifier on the MNIST dataset.

The network consists of 8 × 8, 6 × 6, and 5 × 5 convolutional filter layers with ELU activations followed by a fully connected layer and softmax output.

We train WRM with γ = 0.04E Pn [ X 2 ], and for the other methods we choose as the level of robustness achieved by WRM (19).

2 In the figures, we scale the budgets 1/γ adv and adv for the adversary with Figure 2 (b) we again illustrate the validity of our certificate of robustness FORMULA0 for the worstcase test performance for arbitrary level of robustness ρ.

We see that our certificate provides a performance guarantee for out-of-sample worst-case performance.

DISPLAYFORM0 We now compare adversarial training techniques.

All methods achieve at least 99% test-set accuracy, implying there is little test-time penalty for the robustness levels ( and γ) used for training.

It is thus important to distinguish the methods' abilities to combat attacks.

We test performance of the five methods (ERM, FGM, IFGM, PGM, WRM) under PGM attacks (18) with respect to 2-and ∞-norms.

In FIG2 (a) and (b), all adversarial methods outperform ERM, and WRM offers more robustness even with respect to these PGM attacks.

Training with the Euclidean cost still provides robustness to ∞-norm fast gradient attacks.

We provide further evidence in Appendix A.1.Next we study stability of the loss surface with respect to perturbations to inputs.

We note that small values of ρ test (θ), the distance to adversarial examples (20), correspond to small magnitudes of ∇ z (θ; z) in a neighborhood of the nominal input, which ensures stability of the model.

FIG4 shows that ρ test differs by orders of magnitude between the training methods (models θ = θ ERM , θ FGM , θ IFGM , θ PGM , θ WRM ); the trend is nearly uniform over all γ adv , with θ WRM being the most stable.

Thus, we see that our adversarial-training method defends against gradientexploiting attacks by reducing the magnitudes of gradients near the nominal input.

In FIG4 (b) we provide a qualitative picture by adversarially perturbing a single test datapoint until the model misclassifies it.

Specifically, we again consider WRM attacks and we decrease γ adv until each model misclassifies the input.

The original label is 8, whereas on the adversarial examples IFGM predicts 2, PGM predicts 0, and the other models predict 3.

WRM's "misclassifications" appear consistently reasonable to the human eye (see Appendix A.2 for examples of other digits); WRM defends against gradient-based exploits by learning a representation that makes gradients point towards inputs of other classes.

Together, FIG4 and (b) depict our method's defense mechanisms to gradient-based attacks: creating a more stable loss surface by reducing the magnitude of gradients and improving their interpretability.

For our final experiments, we consider distributional robustness in the context of Q-learning, a model-free reinforcement learning technique.

We consider Markov decision processes (MDP's) (S, A, P sa , r) with state space S, action space A, state-action transition probabilities P sa , and rewards r : S → R. The goal of a reinforcement-learning agent is to maximize (discounted) cumulative rewards t λ t E[r(s t )] (with discount factor λ); this is analogous to minimizing E P [ (θ; Z)] in supervised learning.

Robust MDP's consider an ambiguity set P sa for state-action transitions.

The goal is maximizing the worst-case realization inf P ∈Psa t λ t E P [r(s t )], analogous to problem (1).

such that argmax a Q(s, a) is (eventually) the optimal action in state s to maximize cumulative reward.

In scenarios where the underlying environment has a continuous state-space and we represent Q with a differentiable function (e.g. BID35 ), we can modify the update (21) with an adversarial state perturbation to incorporate distributional robustness.

Namely, we draw the nominal state-transition update s t+1 ∼ p sa (s t , a t ), and proceed with the update (21) using the perturbation DISPLAYFORM0 For large γ, we can again solve problem (22) efficiently using gradient descent.

This procedure provides robustness to uncertainties in state-action transitions.

For tabular Q-learning, where we represent Q only over a discretized covering of the underlying state-space, we can either neglect the second term in the update FORMULA46 and, after performing the update, round s t+1 as usual, or we can perform minimization directly over the discretized covering.

In the former case, since the update (22) simply modifies the state-action transitions (independent of Q), standard results on convergence for tabular Q-learning (e.g. BID50 ) apply under these adversarial dynamics.

We test our adversarial training procedure in the cart-pole environment, where the goal is to balance a pole on a cart by moving the cart left or right.

The environment caps episode lengths to 400 steps and ends the episode prematurely if the pole falls too far from the vertical or the cart translates too far from its origin.

We use reward r(β) := e −|β| for the angle β of the pole from the vertical.

We use a tabular representation for Q with 30 discretized states for β and 15 for its time-derivativeβ (we perform the update (22) without the Q-dependent term).

The action space is binary: push the cart left or right with a fixed force.

Due to the nonstationary, policy-dependent radius for the Wasserstein ball, an analogous for the fast-gradient method (or other variants) is not well-defined.

Thus, we only compare with an agent trained on the nominal MDP.

We test both models with perturbations to the physical parameters: we shrink/magnify the pole's mass by 2, the pole's length by 2, and the strength of gravity g by 5.

The system's dynamics are such that the heavy, short, and strong-gravity cases are more unstable than the original environment, whereas their counterparts are less unstable.

Table 1 shows the performance of the trained models over the original MDP and all of the perturbed MDPs.

Both models perform similarly over easier environments, but the robust model greatly outperforms in harder environments.

Interestingly, as shown in FIG5 , the robust model also learns more efficiently than the nominal model in the original MDP.

We hypothesize that a potential sideeffect of robustness is that adversarial perturbations encourage better exploration of the environment.

Explicit distributional robustness of the form (5) is intractable except in limited cases.

We provide a principled method for efficiently guaranteeing distributional robustness with a simple form of adversarial data perturbation.

Using only assumptions about the smoothness of the loss function , we prove that our method enjoys strong statistical guarantees and fast optimization rates for a large class of problems.

The NP-hardness of certifying robustness for ReLU networks, coupled with our empirical success and theoretical certificates for smooth networks in deep learning, suggest that using smooth networks may be preferable if we wish to guarantee robustness.

Empirical evaluations indicate that our methods are in fact robust to perturbations in the data, and they match or outperform less-principled adversarial training techniques.

The major benefit of our approach is its simplicity and wide applicability across many models and machine-learning scenarios.

There remain many avenues for future investigation.

Our optimization result (Theorem 2) applies only for small values of robustness ρ and to a limited class of Wasserstein costs.

Furthermore, our statistical guarantees (Theorems 3 and 4) use · ∞ -covering numbers as a measure of model complexity, which can become prohibitively large for deep networks.

In a learning-theoretic context, where the goal is to provide insight into convergence behavior as well as comfort that a procedure will "work" given enough data, such guarantees are satisfactory, but this may not be enough in security-essential contexts.

This problem currently persists for most learning-theoretic guarantees in deep learning, and the recent works of BID1 , BID18 , and BID39 attempt to mitigate this shortcoming.

Replacing our current covering number arguments with more intricate notions such as margin-based bounds BID1 would extend the scope and usefulness of our theoretical guarantees.

Of course, the certificate (15) still holds regardless.

More broadly, this work focuses on small-perturbation attacks, and our theoretical guarantees show that it is possible to efficiently build models that provably guard against such attacks.

Our method becomes another heuristic for protection against attacks with large adversarial budgets.

Indeed, in the large-perturbation regime, efficiently training certifiably secure systems remains an important open question.

We believe that conventional · ∞ -defense heuristics developed for image classification do not offer much comfort in the large-perturbation/perceptible-attack setting: · ∞ -attacks with a large budget can render images indiscernible to human eyes, while, for example, · 1 -attacks allow a concerted perturbation to critical regions of the image.

Certainly · ∞ -attack and defense models have been fruitful in building a foundation for security research in deep learning, but moving beyond them may be necessary for more advances in the large-perturbation regime.

In Figure 7 , we repeat the illustration in FIG4 (b) for more digits.

WRM's "misclassifications" are consistently reasonable to the human eye, as gradient-based perturbations actually transform the original image to other labels.

Other models do not exhibit this behavior with the same consistency (if at all).

Reasonable misclassifications correspond to having learned a data representation that makes gradients interpretable.

In FIG7 , we choose a fixed WRM adversary (fixed γ adv ) and perturb WRM models trained with various penalty parameters γ.

As the bound (11) with η = γ suggests, even when the adversary has more budget than that used for training (1/γ < 1/γ adv ), degradation in performance is still smooth.

Further, as we decrease the penalty γ, the amount of achieved robustness-measured here by test error on adversarial perturbations with γ adv -has diminishing gains; this is again consistent with our theory which says that the inner problem (2b) is not efficiently computable for small γ.

Figure 7 .

Visualizing stability over inputs.

We illustrate the smallest WRM perturbation (largest γ adv ) necessary to make a model misclassify a datapoint.

Published as a conference paper at ICLR 2018 DISPLAYFORM0

Figures 9 and 10 repeat Figures 2(b) , 3, and 6 for a larger training adversarial budget (γ = 0.02C 2 ) as well as larger test adversarial budgets.

The distinctions in performance between various methods are less apparent now.

For our method, the inner supremum is no longer strongly concave for over 10% of the data, indicating that we no longer have guarantees of performance.

For large adversaries (i.e. large desired robustness values) our approach becomes a heuristic just like the other approaches.

We consider training FGM, IFGM, and PGM with p = ∞. We first compare with WRM trained in the same manner as before-with the squared Euclidean cost.

Then, we consider a heuristic Lagrangian approach for training WRM with the squared ∞-norm cost.

A.5.1 COMPARISON WITH STANDARD WRM Our method (WRM) is trained to defend against · 2 -norm attacks by using the cost function c((x, y), (x 0 , y 0 )) = x − x 0 2 2 + ∞ · 1 {y = y 0 } with the convention that 0 · ∞ = 0.

Standard adversarial training methods often train to defend against · ∞ -norm attacks, which we compare our method against in this subsection.

Direct comparison between these approaches is not immediate, as we need to determine a suitable to train FGM, IFGM, and PGM in the ∞-norm that corresponds to the penalty parameter γ for the · 2 -norm that we use.

Similar to the expression (19), we use DISPLAYFORM0 as the adversarial training budget for FGM, IFGM and PGM with · ∞ -norms.

Because 2-norm adversaries tend to focus budgets on a subset of features, the resulting ∞-norm perturbations are relatively large.

In FIG0 we show the results trained with a small training adversarial budget.

In this regime, (large γ, small ), WRM matches the performance of other techniques.

In FIG0 we show the results trained with a large training adversarial budget.

In this regime (small γ, large ), performance between WRM and other methods diverge.

WRM, which provably defends against small perturbations, outperforms other heuristics against imperceptible attacks for both Euclidean and ∞ norms.

Further, it outperforms other heuristics on natural images, showing that it consistently achieves a smaller price of robustness.

On attacks with large adversarial budgets (large adv ), however, the performance of WRM is worse than that of the other methods (especially in the case of ∞-norm attacks).

These findings verify that WRM is a practical alternative over existing heuristics for the moderate levels of robustness where our guarantees hold.

DISPLAYFORM1 Our computational guarantees given in Theorem 2 does not hold anymore when we consider ∞-norm adversaries: DISPLAYFORM2 Optimizing the Lagrangian formulation (2b) with the ∞-norm is difficult since subtracting a multiple of the ∞-norm does not add (negative) curvature in all directions.

In Appendix E, we propose a heuristic algorithm for solving the inner supremum problem (2b) with the above cost function (24).

Our approach is based on a variant of proximal algorithms.

We compare our proximal heuristic introduced in Appendix E with other adversarial training procedures that were trained against ∞-norm adversaries.

Results are shown in FIG0 for a small training adversary and FIG0 for a large training adversary.

We observe that similar trends as in Section A.5.1 hold again.

We show that computing worst-case perturbations sup u∈U (θ; z + u) is NP-hard for a large class of feedforward neural networks with ReLU activations.

This result is essentially due to BID26 .

In the following, we use polynomial time to mean polynomial growth with respect to m, the dimension of the inputs z.

An optimization problem is NPO (NP-Optimization) if (i) the dimensionality of the solution grows polynomially, (ii) the language {u ∈ U } can be recognized in polynomial time (i.e. a deterministic algorithm can decide in polynomial time whether u ∈ U), and (iii) can be evaluated in polynomial time.

We restrict analysis to feedforward neural networks with ReLU activations such that the cor-Published as a conference paper at ICLR 2018 responding worst-case perturbation problem is NPO.

4 Furthermore, we impose separable structure on U, that is, U := {v ≤ u ≤ w} for some v < w ∈ R m .Lemma 2.

Consider feedforward neural networks with ReLU's and let U := {

v ≤ u ≤ w}, where v < w such that the optimization problem max u∈U (θ; z + u) is NPO.

Then there exists θ such that this optimization problem is also NP-hard.

Proof First, we introduce the decision reformulation of the problem: for some b, we ask whether there exists some u such that (θ; z + u) ≥ b. The decision reformulation for an NPO problem is in NP, as a certificate for the decision problem can be verified in polynomial time.

By appropriate scaling of θ, v, and w, BID26 show that 3-SAT Turing-reduces to this decision problem: given an oracle D for the decision problem, we can solve an arbitrary instance of 3-SAT with a polynomial number of calls to D. The decision problem is thus NP-complete.

Now, consider an oracle O for the optimization problem.

The decision problem Turing-reduces to the optimization problem, as the decision problem can be solved with one call to O. Thus, the optimization problem is NP-hard.

For completeness, we provide an alternative proof to that given in BID7 using convex analysis.

Our proof is less general, requiring the cost function c to be continuous and convex in its first argument.

The below general duality result gives Proposition 1 as an immediate special case.

Recalling Rockafellar & Wets (1998, Def.

14.27 and Prop.

14.33), we say that a function g : X × Z → R is a normal integrand if for each α, the mapping DISPLAYFORM0 is closed-valued and measurable.

We recall that if g is continuous, then g is a normal integrand (Rockafellar & Wets, 1998, Cor.

14.34) ; therefore, g(x, z) = γc(x, z) − (θ; x) is a normal integrand.

We have the following theorem.

Theorem 5.

Let f, c be such that for any γ ≥ 0, the function g(x, z) = γc(x, z) − f (x) is a normal integrand.

(For example, continuity of f and closed convexity of c is sufficient.)

For any ρ > 0 we have DISPLAYFORM1 Proof First, the mapping P → W c (P, Q) is convex in the space of probability measures.

As taking P = Q yields W c (Q, Q) = 0, Slater's condition holds and we may apply standard (infinite dimensional) duality results (Luenberger, 1969, Thm. 8.7 .1) to obtain sup P :Wc (P,Q) f (x)dP (x) = sup DISPLAYFORM2 Now, noting that for any M ∈ Π(P, Q) we have f dP = f (x)dM (x, z), we have that the rightmost quantity in the preceding display satisfies DISPLAYFORM3 That is, we have DISPLAYFORM4 Now, we note a few basic facts.

First, because we have a joint supremum over P and measures M ∈ Π(P, Q) in expression (25), we have that DISPLAYFORM5 We would like to show equality in the above.

To that end, we note that if P denotes the space of regular conditional probabilities (Markov kernels) from Z to X, then DISPLAYFORM6 Recall that a conditional distribution P (· | z) is regular if P (· | z) is a distribution for each z and for each measurable A, the function z → P (A | z) is measurable.

Let X denote the space of all measurable mappings z → x(z) from Z to X. Using the powerful measurability results of Rockafellar & Wets (1998, Theorem 14 .60), we have DISPLAYFORM7 because f − c is upper semi-continuous, and the latter function is measurable.

Now, let x(z) be any measurable function that is -close to attaining the supremum above.

Define the conditional distribution P (· | z) to be supported on x(z), which is evidently measurable.

Then using the preceding display, we have DISPLAYFORM8 As > 0 is arbitrary, this gives DISPLAYFORM9 as desired, which implies both equality (6) and completes the proof.

First, note that z (θ) is unique and well-defined by the strong convexity of f (θ, ·).

For Lipschitzness of z (θ), we first argue that z (θ) is continuous in θ.

For any θ, optimality of z (θ) implies that DISPLAYFORM0 By strong concavity, for any θ 1 , θ 2 and z 1 = z (θ 1 ) and DISPLAYFORM1 Summing these inequalities gives DISPLAYFORM2 where the last inequality follows because g z (θ 1 , z 1 ) T (z 2 − z 1 ) ≤ 0.

Using a cross-Lipschitz condition from above and Holder's inequality, we obtain DISPLAYFORM3 that is, DISPLAYFORM4 To see the second inequality, we show thatf is differentiable with ∇f (θ) = g θ (θ, z (θ)).

By using a variant of the envelope (or Danskin's) theorem, we first show directional differentiability off .

Recall that we say f is inf-compact if for all θ 0 ∈ Θ, there exists α > 0 and a compact set C ⊂ Θ such that DISPLAYFORM5 for all θ in some neighborhood of θ 0 BID8 .

See Bonnans & Shapiro (2013, Theorem 4.13 ) for a proof of the following result.

Lemma 3.

Suppose that f (·, z) is differentiable in θ for all z ∈ Z, and f , ∇ z f are continuous on Θ × Z. If f is inf-compact, thenf is directionally differentiable with DISPLAYFORM6 Now, note that from Assumption B, we have DISPLAYFORM7 from which it is easy to see that f is inf-compact.

Applying Lemma 3 tof and noting that S(θ) is unique by strong convexity of f (θ, ·), we have thatf is directionally differentiable with ∇f (θ) = g θ (θ, z (θ)).

Since g θ is continuous by Assumption B and z (θ) is Lipschitz (26), we conclude that f is differentiable.

Finally, we have DISPLAYFORM8 where we have used inequality FORMULA7 again.

This is the desired result.

Our proof is based on that of BID20 .

For shorthand, let f (θ, z; z 0 ) = (θ; z) − γc(z, z 0 ), noting that we perform gradient steps with DISPLAYFORM0 in the rest of the proof, which is satisfied for the constant stepsize α = DISPLAYFORM1 .

By a Taylor expansion using the L φ -smoothness of the objective F , we have DISPLAYFORM2 (27) DISPLAYFORM3 where the latter inequality holds since DISPLAYFORM4 gives the result.

We first show the bound (11).

From the duality result (5), we have the deterministic result that DISPLAYFORM0 for all ρ > 0, distributions Q, and γ ≥ 0.

Next, we show that E Pn [φ γ (θ; Z)] concentrates around its population counterpart at the usual rate BID9 .

DISPLAYFORM1 Thus, the functional θ → F n (θ) satisfies bounded differences (Boucheron et al., 2013, Thm. 6 .2), and applying standard results on Rademacher complexity BID0 and entropy integrals (van der Vaart & Wellner, 1996, Ch.

2.2) gives the result.

To see the second result (12), we substitute ρ = ρ n in the bound (11).

Then, with probability at least 1 − e −t , we have DISPLAYFORM2 Since we have DISPLAYFORM3 from the strong duality in Proposition 1, our second result follows.

The result is essentially standard BID52 , which we now give for completeness.

Note that for DISPLAYFORM0 First, we show that P * (θ) and P * n (θ) are attained for all θ ∈ Θ. We omit the dependency on θ for notational simplicity and only show the result for P * (θ) as the case for P * n (θ) is symmetric.

Let P be an -maximizer, so that DISPLAYFORM1 As Z is compact, the collection {P 1/k } k∈N is a uniformly tight collection of measures.

By Prohorov's theorem (Billingsley, 1999, Ch 1.1, p. 57) , (restricting to a subsequence if necessary), there exists some distribution P * on Z such that P 1/k d → P * as k → ∞. Continuity properties of Wasserstein distances (Villani, 2009, Corollary 6.11 ) then imply that lim k→∞ W c (P 1/k , P 0 ) = W c (P * , P 0 ).Combining (29) and the monotone convergence theorem, we obtain E P * [ (θ; Z)] − γW c (P * , P 0 ) = lim k→∞ E P 1/k [ (θ; Z)] − γW c (P 1/k , P 0 )≥ sup P {E P [ (θ; Z)] − γW c (P, P 0 )} .We conclude that P * is attained for all P 0 .Next, we show the concentration result (30).

Recall the definition (9) of the transportation mapping T (θ, z) := argmax z ∈Z { (θ; z ) − γc(z , z)} , which is unique and well-defined under our strong concavity assumption that γ > L zz , and smooth (recall Eq. FORMULA0 ) in θ.

Then by Proposition 1 (or by using a variant of Kantorovich duality BID53 , Chs.

9-10)), we have under both cases (i), that c is Lipschitz, and (ii), that is Lipschitz in z, using a covering argument on Θ. Recall inequality (16) (i.e. Lemma 1), which is that DISPLAYFORM2 DISPLAYFORM3 We have the following lemma.

Lemma 4.

Assume the conditions of Theorem 4.

Then for any θ 1 , θ 2 ∈ Θ, DISPLAYFORM4 Proof In the first case, that c is L c -Lipschitz in its first argument, this is trivial: we have DISPLAYFORM5 by the smoothness inequality (16) for T .In the second case, that z → (θ, z) is L c -Lipschitz, let z i = T (θ i ; z) for shorthand.

Then we have γc(z 2 , z) − γc(z 1 , z) = γc(z 2 , z) − (θ 2 , z 2 ) + (θ 2 , z 2 ) − γc(z 1 , z) ≤ γc(z 1 , z) − (θ 2 , z 1 ) + (θ 2 , z 2 ) − γc(z 1 , z) = (θ 2 , z 2 ) − (θ 2 , z 1 ), and similarly, γc(z 2 , z) − γc(z 1 , z) = γc(z 2 , z) − (θ 1 , z 1 ) + (θ 1 , z 1 ) − γc(z 1 , z) ≥ γc(z 2 , z) − (θ 1 , z 1 ) + (θ 1 , z 2 ) − γc(z 2 , z) = (θ 1 , z 2 ) − (θ 1 , z 1 ).Combining these two inequalities and using that DISPLAYFORM6 DISPLAYFORM7 Similarly, an analogous result to Theorem 4 holds.

Define the transport map for the covariate shift DISPLAYFORM8

First, note that β → i:vi>β (v i − β) − αλβ =: h(β) is decreasing.

Noting that v 1 > 0 and −αλ v ∞ < 0, there exists β such that h(β ) = 0 and β ∈ (0, v ∞ ).

Since v i 's are decreasing and nonnegative, there exists j such that v j > β ≥ v j +1 (we abuse notation and let v m+1 := 0).

Then, we have DISPLAYFORM0 That is, j = j .

Solving for β in 0 = h(β ) =

<|TLDR|>

@highlight

We provide a fast, principled adversarial training procedure with computational and statistical performance guarantees.
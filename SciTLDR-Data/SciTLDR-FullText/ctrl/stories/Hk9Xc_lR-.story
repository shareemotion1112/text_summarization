Generative adversarial training can be generally understood as minimizing certain moment matching loss defined by a set of discriminator functions, typically  neural networks.

The discriminator set should be large enough to be able to uniquely identify the true distribution (discriminative), and also be small enough to go beyond memorizing samples (generalizable).

In this paper, we show that a discriminator set is guaranteed to be discriminative whenever its linear span is dense in the set of bounded continuous functions.

This is a very mild condition satisfied even by neural networks with a single neuron.

Further, we develop generalization bounds between the learned distribution and true distribution under different evaluation metrics.

When evaluated with neural distance, our bounds show that generalization is guaranteed as long as the discriminator set is small enough, regardless of the size of the generator or hypothesis set.

When evaluated with KL divergence, our bound provides an explanation on the counter-intuitive behaviors of testing likelihood in GAN training.

Our analysis sheds lights on understanding the practical performance of GANs.

Generative adversarial networks (GANs) BID14 and their variants can be generally understood as minimizing certain moment matching loss defined by a set of discriminator functions.

Mathematically, GANs minimize the integral probability metric (IPM) BID31 , that is, DISPLAYFORM0 whereμ m is the empirical measure of the observed data, and F and G are the sets of discriminators and generators, respectively.1.

Wasserstain GAN (W-GAN) BID1 .

F = Lip 1 (X) := {f : ||f || Lip ≤ 1}, corresponding to the Wasserstain-1 distance.

2.

MMD-GAN BID27 BID13 BID25 .

F is taken as the unit ball in a Reproducing Kernel Hilbert Space (RKHS), corresponding to the Maximum Mean Discrepency (MMD).

3. Energy-based GANs BID45 .

F is taken as the set of continuous functions bounded between 0 and M for some constant M > 0, corresponding to the total variation distance BID1 .4.

When the KL divergence is used as the evaluation metric, our bound (Corollary 3.5) suggests that the generator and discriminator sets have to be compatible in that the log density ratios of the generators and the true distributions should exist and be included inside the linear span of the discriminator set.

The strong condition that log-density ratio should exist partially explains the counter-intuitive behavior of testing likelihood in flow GANs (e.g., BID11 BID15 .

5.

We extend our analysis to study neural f -divergences that are the learning objective of f -GANs, and establish similar results on the discrimination and generalization properties of neural fdivergences; see Appedix B. Different from neural distance, a neural f -divergence is discriminative if linear span of its discriminators without the output activation function is dense in the bounded continuous function space.

We use X to denote a subset of R d .

For each continuous function f : X → R, we define the maximum norm as f ∞ = sup x∈X |f (x)|, and the Lipschitz norm f Lip = sup{|f (x) − f (y)|/ x − y : x, y ∈ X, x = y}, and the bounded Lipschitz (BL) norm f BL = max{ f Lip , f ∞ }.

The set of continuous functions on X is denoted by C(X), and the Banach space of bounded continuous function is C b (X) = {f ∈ C(X) : f ∞ < ∞}.The set of Borel probability measures on X is denoted by P B (X).

In this paper, we assume that all measures involved belong to P B (X), which is sufficient in all practical applications.

We denote by E µ [f ] the integral of f with respect to probability measure µ.

The weak convergence, or convergence in distribution, is denoted by ν n ν.

Given a base measures τ (e.g., Lebesgue measure), the density of µ ∈ P B (X), if it exists, is denoted by ρ µ = dµ dτ .

We do not assume density exists in our main theoretical results, except the cases when discussing KL divergence.

As listed in the introduction, many variants of GAN can be viewed as minimizing the integral probability metric (1).

Without loss of generality, we assume that the discriminator set F is even, i.e., f ∈ F implies −f ∈ F. Intuitively speaking, minimizing (1) towards zero corresponds to matching the moments E µ [f ] = E ν [f ] for all discriminators f ∈ F. In their original formulation, all those discriminator sets are non-parametric, infinite dimensional, and large enough to guarantee that d F (µ, ν) = 0 implies µ = ν.

In practice, however, the discriminator set is typically restricted to parametric function classes of form F nn = {f θ : θ ∈ Θ}. When f θ is a neural network, we call d Fnn (µ, ν) a neural distance following .

Neural distances are the actual object function that W-GAN optimizes in practice because they can be practically optimized and can leverage the representation power of neural networks.

Therefore, it is of great importance to directly study neural distances, instead of Wasserstein metric, in order to understand practical performance of GANs.

Because the parameter function set F nn is much smaller than the non-parametric sets like Lip 1 (X), a key question is whether F nn is large enough so that moment matching on F nn (i.e., d Fnn (µ, ν) = 0) implies µ = ν.

It turns out the answer is affirmative once F nn is large enough so that its linear span (instead of F nn itself) forms a universal approximator.

This is a rather weak condition, which is satisfied even by very small sets such as neural networks with a single neuron.

We make this concrete in the following.

Definition 2.1.

Let (X, d X ) be a metric space and F be a set of functions on X. We say that d F (µ, ν) (and F) is discriminative if DISPLAYFORM0 for any two Borel probability measures µ, ν ∈ P B (X).

In other words, F is discriminative if the moment matching on F, i.e., E µ [f ] = E ν [f ] for any f ∈ F, implies µ = ν.

The key observation is that E µ [f ] = E ν [f ] for any f ∈ F implies the same holds true for all f in the linear span of F. Therefore, it is sufficient to require the linear span of F, instead of F itself, to be large enough to well approximate all the indicator test functions.

Theorem 2.2.

For a given function set F ⊂ C b (X), define DISPLAYFORM1 α i f i : α i ∈ R,f i ∈ F, n ∈ N}.(3)Then d F (µ, ν) is discriminative if spanF is dense in the space of bounded continuous functions C b (X) under the uniform norm || · || ∞ , that is, for any f ∈ C b (X) and > 0, there exists an f ∈ spanF such that f − f ∞ ≤ .

An equivalent way to put is that C b (X) is included in the closure of spanF, that is, DISPLAYFORM2 (4) Further, (4) is a necessary condition for d F (µ, ν) to be discriminative if X is a compact space.

Remark 2.1.

The basic idea of characterizing probability measures using functions in C b (X) is closely related to the concept of weak convergence.

Recall that a sequence ν n weakly converges to µ, i.e., ν n µ, if and only if DISPLAYFORM3 Proof of the sufficient part of Theorem 2.2 is standard and the same with proof of the uniqueness of weak convergence; see, e.g., Lemma 9.3.2 in BID12 .

Remark 2.2.

We obtain similar results for neural f -divergence d φ,F (µ||ν) in Theorem B.1.

The difficulty in analyzing neural f -divergence is that moment matching on the discriminator set is only a sufficient condition for minimizing neural f -divergence, i.e., DISPLAYFORM4 Consequently, cl(spanF) ⊇ C b (X) is only necessary but not sufficient for a neural f -divergence to be discriminative.

When the discriminators are neural networks, we show that moment matching on the function set that consists of discriminators without their output activation function, denoted as F 0 , is a necessary condition for minimizing neural f -divergence, i.e., DISPLAYFORM5 We refer to Theorem B.1 (ii) for a precise statement.

Therefore, a neural f -divergence is discriminative if linear span of its discriminators without the output activation function is dense in the bounded continuous function space.

Remark 2.3.

Because the set of bounded Lipschitz functions BL(X) = {f ∈ C b (X) : ||f || Lip < ∞} is dense in C b (X), the condition in (4) can be replaced by a weaker condition cl(spanF) ⊇ BL(X).

One can define a norm · BL for functions in BL(X) by f BL = max{ f Lip , f ∞ }.

This defines the bounded Lipschitz (BL) distance, DISPLAYFORM6 The BL distance is known to metrize weak convergence in sense that d BL (µ, ν n ) → 0 is equivalent to ν n µ for all Borel probability measures on R d ; see section 8.3 in Bogachev (2007) .Neural distances are discriminative.

The key message of Theorem 2.2 is that it is sufficient to require cl(spanF) ⊇ C b (X) (Condition (4)), which is a much weaker condition than the perhaps more straightforward condition cl(F) ⊇ C b (X).

In fact, (4) is met by function sets that are much smaller than what we actually use in practice.

For example, it is satisfied by the neural networks with only a single neuron, i.e., DISPLAYFORM7 This is because its span spanF nn includes neural networks with infinite numbers of neurons, which are well known to be universal approximators in C b (X) according to classical theories (e.g., BID10 BID19 BID18 BID24 BID5 .

We recall the following classical result.

Theorem 2.3 (Theorem 1 in BID24 ).

Let σ : R → R be a continuous activation function and X ⊂ R d be any compact set.

Let F nn be the set of neural networks with a single neuron as defined in (5), then spanF nn is dense in C(X) if and only if σ is not a polynomial.

The above result requires that the parameters [w, b] take values in R d+1 .

In practice, however, we can only efficiently search in bounded parameter sets of [w, b] using local search methods like gradient descent.

We observe that it is sufficient to replace R d+1 with a bounded parameter set Θ for non-decreasing homogeneous activation functions such as σ(u) = max{u, 0} α with α ∈ N; note that α = 1 is the widely used rectified linear unit (ReLU).Corollary 2.4.

Let X ⊂ R d be any compact set, and σ(u) = max{u, 0} α (α ∈ N), and DISPLAYFORM8 For the case when Θ = {θ ∈ R d+1 : θ 2 ≤ 1}, Bach (2017) not only proves that spanF nn is dense in Lip 1 (X) (and thus dense in C b (X)), but also gives the convergence rate.

Therefore, for ReLU activation functions, F nn with bounded parameter sets, like {θ : θ ≤ 1} or {θ : θ = 1} for any norm on R d+1 , is sufficient to discriminate any two Borel probability measures.

Note that this is not true for some other activation functions such as tanh or sigmoid, because there is an approximation gap between span{σ(w DISPLAYFORM9 d+1 is bounded; see e.g., Barron (1993) (Theorem 3).

From this perspective, homogeneous activation functions such as ReLU are preferred as discriminators.

One advantage of using bounded parameter set Θ is that it makes F nn have a bounded Lipschitz norm, and hence the corresponding neural distance is upper bounded by Wasserstein distance.

In fact, W-GAN uses weight clipping to explicitly enforce θ ∞ ≤ δ.

However, we should point out that the Lipschitz constraint does not help in making F discriminative since the constraint decreases, instead of enlarges, the function set F. Instead, the role of the Lipschitz constraint should be mostly in stabilizing the training BID1 and assuring a generalization bound as we discuss in Section 3.

Another related way to justify the Lipschitz constraint is its relation to metrizing weak convergence, as we discuss in the sequel.

Neural distance and weak convergence.

If F is discriminative, then d F (µ, ν) = 0 implies µ = ν.

In practice, however, we often cannot achieve d F (µ, ν) = 0 strictly.

Instead, we often have d F (µ, ν n ) → 0 for a sequence of ν n and want to establish the weak convergence ν n µ.Theorem 2.5.

Let (X, d X ) be any metric space.

If spanF is dense in C b (X), we have lim n→∞ d F (µ, ν n ) = 0 implies ν n weakly converges to µ.Additionally, if F is contained in a bounded Lipchitz function space, i.e., there exists 0 < C < ∞ such that ||f || BL ≤ C for all f ∈ F, then ν n weakly converges to µ implies lim n→∞ d F (µ, ν n ) = 0.Theorem 10 of BID28 states a similar result for generic adversarial divergences, but does not obtain the specific weak convergence result for neural distances due to lacking of Theorem 2.2.

Another difference is that Theorem 10 of BID28 heavily relies on the compactness assumption of X, while our result does not need this assumption.

We provide the proof for Theorem 2.5 in Appendix C.When X is compact, Wasserstein distance and the BL distance are equivalent and both metrize weak convergence.

As we discussed earlier, the condition cl(spanF) = C b (X) and F ⊆ Lip K (X) are satisfied by neural networks F nn with ReLU activation function and bounded parameter set Θ. Therefore, the related neural distance d Fnn is topologically equivalent to the Wasserstein and BL distance, because all of them metrize the weak convergence.

This does not imply, however, that they are equivalent in the metric sense (or strongly equivalent) since the ratio d BL (µ, ν)/d Fnn (µ, ν) can be unbounded.

In general, the neural distances are weaker than the BL distance because of smaller F. In Section 2.1 (and particularly Corollary 2.8), we draw more discussions on the bounds between BL distance and neural distances.

Theorem 2.2 characterizes the condition under which a neural distance is discriminative, and shows that even neural networks with a single neuron are sufficient to be discriminative.

This does not explain, however, why it is beneficial to use larger and deeper networks as we do in practice.

What is missing here is to frame and understand how discriminative or strong a neural distance is.

This is because even if d F (µ, ν) is discriminative, it can be relatively weak in that d F (µ, ν) may be small when µ and ν are very different under standard metrics (e.g., BL distance).

Obviously, a larger F yields a stronger neural distance, that is, if DISPLAYFORM0 For example, because it is reasonable to assume that neural networks are bounded Lipschitz when X and Θ are bounded, we can control a neural distance with the BL distance: DISPLAYFORM1 where C := sup f ∈F {||f || BL } < ∞. A more difficult question is if we can establish inequalities in the other direction, that is, controlling d BL (µ, ν), or in general a stronger d F (µ, ν), with a weaker d F (µ, ν) in some way.

In this section, we characterize conditions under which this is possible and develop bounds that allow us to use neural distances to control stronger distances such as BL distance, and even KL divergence.

These bounds are used in Section 3 to translate generalization bounds in d F (µ, ν) to that in BL distance and KL divergence.

The core of the discussion involves understanding how d F (µ, ν) can be used to control the difference of the moment | E µ g − E ν g| for g outside of F. We address this problem by two steps: first controlling functions in spanF, and then functions in cl(spanF) that is large enough to include C b (X) for neural networks.

Controlling functions in spanF. We start with understanding how DISPLAYFORM2 This can be characterized by introducing a notion of norm on spanF. Proposition 2.6.

For each g ∈ spanF that can be decomposed into g = n i=1 w i f i + w 0 as we define in (3), the F-variation norm ||g|| F ,1 of g is the infimum of DISPLAYFORM3 Intuitively speaking, ||g|| F ,1 denotes the "minimum number" of functions in F needed to represent g. As F becomes larger, ||g|| F ,1 decreases and DISPLAYFORM4 .

Therefore, although adding more neurons in F may not necessarily enlarge spanF, it decreases ||g|| F ,1 and yields a stronger neural distance.

A more critical question is how the neural distance d F (µ, ν) can also control the discrepancy E µ g − E ν g for functions outside of spanF but inside cl(spanF).

The bound in this case is characterized by a notion of error decay function defined as follows.

Proposition 2.7.

Given a function g, we say that g is approximated by F with error decay function (r) if for any r ≥ 0, there exists an f r ∈ spanF with ||f r || F ,1 ≤ r such that ||f − f r || ∞ ≤ (r).

Therefore, g ∈ cl(spanF) if and only if inf r≥0 (r) = 0.

We have DISPLAYFORM0 It requires further efforts to derive the error decay function for specific F and g. For example, Proposition 6 of Bach (2017) allows us to derive the decay rate of approximating bounded Lipschitz functions with rectified neurons, yielding a bound between BL distance and neural distance.

Corollary 2.8.

Let X be the unit ball of R DISPLAYFORM1 whereÕ denotes the big-O notation ignoring the logarithm factor.

The result in (6) shows that d F (µ, ν) gives an increasingly weaker bound when the dimension d increases.

This is expected because we approximate a non-parametric set with a parametric one.

Likelihood and KL divergence.

Maximum likelihood has been the predominant approach in statistical learning, and testing likelihood forms a standard criterion for testing unsupervised models.

The recent advances in deep unsupervised learning, however, make it questionable whether likelihood is the right objective for training and evaluation (e.g., BID40 .

For example, some recent empirical studies (e.g., BID11 BID15 showed a counter-intuitive phenomenon that both the testing and training likelihood (assuming generators with valid densities are used) tend to decrease, instead of increase, as the GAN loss is minimized.

A hypothesis for explaining this is that the neural distances used in GANs are too weak to control the KL divergence properly.

Therefore, from the theoretical perspective, it is desirable to understand under what conditions (even if it is a very strong one), the neural distance can be strong enough to control KL divergence.

This can be done by the following simple result.

Proposition 2.9.

Assume µ and ν have positive density functions ρ µ (x) and ρ ν (x), respectively.

Then DISPLAYFORM2 If log(ρ µ /ρ ν ) ∈ cl(spanF) with an error decay function (r) = O(r −κ ), then DISPLAYFORM3 This result shows that we require that the density ratio log(ρ µ /ρ ν ) should exist and behave nicely in spanF or cl(spanF) in order to bound KL divergence with d F (µ, ν).

If either µ or ν is an empirical measure, the bound is vacuum since DISPLAYFORM4 Obviously, this strong condition is hard to satisfy in practice, because practical data distributions and generators in GANs often have no densities or at least highly peaky densities.

We draw more discussions in Corollary 3.5.

Section 2 suggests that it is better to use larger discriminator set F in order to obtain stronger neural distance.

However, why do regularization techniques, which effectively shrink the discriminator set, help GAN training in practice?

The answer has to do with the fact that we observe the true model µ only through an i.i.d.

sample of size m (whose empirical measure is denoted byμ m ), and hence can only optimize the empirical loss d F (μ m , ν), instead of the exact loss d F (µ, ν).

Therefore, generalization bounds are required to control the exact loss d F (µ, ν) when we can only minimize its empirical version d F (μ m , ν).

Specifically, let G be a class of generators that may or may not include the unknown true distribution µ. Assume ν m minimizes the GAN loss DISPLAYFORM0 We are interested in bounding the difference between ν m and the unknown µ under certain evaluation metric.

Depending on what we care about, we may be interested in the generalization error in terms of the neural distance d F (µ, ν m ), or other standard quantities of interest such as BL distance d BL (µ, ν m ) and KL divergence KL(µ, ν m ) or the testing likelihood.

In this section, we adapt the standard Rademacher complexity argument to establish generalization bounds for GANs.

We show that the discriminator set F should be small enough to be generalizable, striking a tradeoff with the other requirement that it should be large enough to be discriminative.

We first present the generalization bound under neural distance, which purely depends on the Rademacher complexity of the discriminator set F and is independent of the generator set G. Then using the results in Section (2.1), we discuss the generalization bounds under other standard metrics, like BL distance and KL divergence.

Using the standard derivation and the optimality condition (9), we have (see Appendix D) DISPLAYFORM0 This reduces the problem to bounding the discrepancy DISPLAYFORM1 | between the true model µ and its empirical versionμ m .

This can be achieved by the uniform concentration bounds developed in statistical learning theory (e.g., BID43 and empirical process (e.g., BID42 .

In particular, the concentration property related to DISPLAYFORM2 where the expectation is taken w.r.t.

X i ∼ µ, and Rademacher random variable DISPLAYFORM3 m (F) characterizes the ability of overfitting with pure random labels using functions in F and hence relates to the generalization bounds.

Standard results in learning theory show that DISPLAYFORM4 where ∆ = sup f ∈F ||f || ∞ .

Combining this with FORMULA0 , we obtain the following result.

Theorem 3.1.

Assume that the discriminator set F is even, i.e., f ∈ F implies −f ∈ F, and that all discriminators are bounded by ∆, i.e., f ∞ ≤ ∆ for any f ∈ F. Letμ m be an empirical measure of an i.i.d.

sample of size m drawn from µ. DISPLAYFORM5 Then with probability at least 1 − δ, we have DISPLAYFORM6 where R (µ) m (F) is the Rademacher complexity of F defined in (11).We obtain nearly the same generalization bound for neural f -divergence in Theorem B.3.

Theorem 3.1 relates the generalization error of GANs to the Rademacher complexity of the discriminator set F. The smaller the discriminator set F is, the more generalizable the result is.

Therefore, the choice of F should strike a subtle balance between the generalizability and the discriminative power: F should be large enough to make d F (µ, ν) discriminative as we discuss in Section 2.1, and simultaneously should be small enough to have a small generalization error in (12).

It turns out parametric neural discriminators strike a good balance for this purpose, given that it is both discriminative as we show in Section 2.1, and give small generalization bound as we show in the following.

DISPLAYFORM7 Assume that F is neural networks with a single rectified linear unit (ReLU) DISPLAYFORM8 Then with probability at least 1 − δ, DISPLAYFORM9 and DISPLAYFORM10 where C = 4 √ 2 + 4 log(1/δ)

andÕ denotes the big-O notation ignoring the logarithm factor.

Note that the three terms in Eqn.

FORMULA0 DISPLAYFORM11 p } is a parametric function class with p parameters in a bounded set Θ and that (2) every f θ is L-Lipschitz continuous with respect to the parameters θ, i.e., f θ − f θ ∞ ≤ L θ − θ 2 .

Then with probability at least 1 − δ, we have DISPLAYFORM12 where DISPLAYFORM13 This result can be easily applied to neural discriminators, since neural networks f θ (x) are generally Lipschitz w.r.t.

the parameter θ, once the input domain X is bounded.

For neural discriminators, we also apply the bound on the Rademacher complexity of DNNs recently derived in BID7 , which gives a sharper bound than that in Corollary 3.3; see Appendix A.1.With the basic result in Theorem 3.1, we can also discuss the learning bounds of GANs with choices of non-parametric discriminators.

Making use of Rademacher complexity of bounded sets in a RKHS (e.g., Lemma 22 in Bartlett & Mendelson FORMULA44 ), we give the learning bound of MMDbased GANs BID27 BID13 as follows.

We present the results for Wasserstein distance and total variance distance in Appendix A.2, and highlight the advantages of using parametric neural discriminators.

Corollary 3.4.

Under the condition of Theorem 3.1, we further assume that F = {f ∈ H : f H ≤ 1} where H is a RKHS whose positive definite kernel k(x, x ) satisfies k(x, x) ≤ C k < +∞ for all x ∈ X. Then with probability at least 1 − δ, DISPLAYFORM14 where C = 2 2 + 2 log(1/δ) √ C k .Remark 3.1 (Comparisons with results in ).

also discussed the generalization properties of GANs under a similar framework.

In particular, they developed bounds of form |d F (µ, ν) − d F (μ m ,ν m )| whereμ m andν m are empirical versions of the target distribution µ and ν with sample size m. Our framework is similar, but considers bounding the quantity DISPLAYFORM15 , which is of more direct interest.

In fact, our Eqn. (10) shows that our generalization error can be bounded by the generalization error studied in .

Another difference is that we adapt the Rademacher complexity argument to derive the bound, while made use of the -net argument.

Bounding the KL divergence and testing likelihood.

The above results depend on the evaluation metric we use, which is d F (µ, ν) or d BL (µ, ν).

If we are interested in evaluating the model using even stronger metrics, such as KL divergence or equivalently testing likelihood, then the generator set G enters the scene in a more subtle way, in that a larger generator set G should be companioned with a larger discriminator set F in order to provide meaningful bounds on KL divergence.

This is illustrated in the following result obtained by combining Theorem 3.1 and Proposition 2.9.

Corollary 3.5.

Assume both the true µ and all the generators ν ∈ G have positive densities ρ µ and ρ ν , respectively.

Assume F consists of bounded functions with ∆ := sup f ∈F ||f || ∞ < ∞.Further, assume the discriminator set F is compatible with the generator set G in the sense that log(ρ ν /ρ µ ) ∈ spanF, ∀ν ∈ G, with a compatible coefficient defined as DISPLAYFORM16 Different from the earlier bounds, the bound in (17) depends on the compatibility coefficient Λ F ,G that casts a more interesting trade-off on the choice of the generator set G: the generator set G should be small and have well-behaved density functions to ensure a small Λ F ,G , while should be large enough to have a small modeling error inf ν∈G KL(µ, ν).

Related, the discriminator set should be large enough to include all density ratios log(ρ µ /ρ ν ) in a ball of radius Λ F ,G of spanF, and should also be small to have a low Rademacher complexity R (µ) m (F).

Obviously, one can also extend Corollary 3.5 using (8) in Proposition 2.7, to allow log(ρ µ /ρ ν ) ∈ cl(spanF) in which case the compatibility of G and F should be mainly characterized by the error decay function (r).

DISPLAYFORM17 is the difference between the testing likelihood E µ [log p νm ] of estimated model ν m and the optimal testing likelihood E µ [log p µ ].

Therefore, Corollary 3.5 also provides a bound for testing likelihood.

Unfortunately, the condition in Corollary 3.5 is rather strong, in that it requires that both the true distribution µ and the generators ν have positive densities and that the log-density ratio log(ρ µ /ρ ν ) be well-behaved.

In practical applications of computer vision, however, both µ and ν tend to concentrate on local regions or sub-manifolds of X, with very peaky densities, or even no valid densities; this causes the compatibility coefficient Λ F ,G very large, or infinite, making the bound in (17) loose or vacuum.

This provides a potential explanation for some of the recent empirical findings (e.g., BID11 BID15 ) that the negative testing likelihood is uncorrelated with the GAN loss functions, or even increases during the GAN training progress.

The underlying reason here is that the neural distance is not strong enough to provide meaningful bound for KL divergence.

See Appendix E for an illustration using toy examples.

There is a surge of research interest in GANs; however, most of the work has been empirical in nature.

There has been some theoretical literature on understanding GANs, including the discrimination and generalization properties of GANs.

The discriminative power of GANs is typically justified by assuming that the discriminator set F has enough capacity.

For example, BID14 assumes that F contains the optimal discriminator DISPLAYFORM0 .

Similar capacity assumptions have been made in nearly all other GANs to prove their discriminative power; see, e.g., BID45 BID33 ; BID1 .

However, discriminators are in practice taken as certain parametric function class, like neural networks, which violates these capacity assumptions.

The universal approximation property of neural networks is used to justify the discriminative power empirically.

In this work, we show that the GAN loss is discriminative if spanF can approximate any continuous functions.

This condition is very weak and can be satisfied even when none of the discriminators is close to the optimal discriminator.

The MMD-based GANs BID27 BID13 BID25 avoid the parametrization of discriminators by taking advantage of the close-form solution of the optimal discriminator in the non-parametric RKHS space.

Therefore, the capacity assumption is satisfied in MMD-based GANs, and their discriminative power is easily justified.

BID28 defines a notion of adversarial divergences that include a number of GAN objective functions.

They show that if the objective function is an adversarial divergence with some additional conditions, then using a restricted discriminator family has a moment-matching effect.

Our treatment of the neural divergence is directly inspired by them.

We refer to Remark B.1 for a detailed comparison.

BID28 also shows that for objective functions that are strict adversarial divergence, convergence in the objective function implies weak convergence.

However, they do not provide a condition under which an adversarial divergence is strict.

A major contribution of our work is to fill this gap, and to provide such a condition that is sufficient and necessary.

FORMULA0 is the definition of generalization error; see more discussions in Remark 3.1.

Moreover, allows only polynomial number of samples from the generated distribution because the training algorithm should run in polynomial time.

We do not consider this issue because in this work we only study the statistical properties of the objective functions and do not touch the optimization method.

Finally, shows that the GAN loss can approach its optimal value even if the generated distribution has very low support, and provides empirical evidence for this problem.

Our result is consistent with their results because our generalization error is measured by the neural distance/divergence.

Finally, there are some other lines of research on understanding GANs.

BID26 studies the dynamics of GAN's training and finds that: a GAN with an optimal discriminator provably converges, while a first order approximation of the discriminator leads to unstable dynamics and mode collapse.

BID23 studies WGAN and optimal transportation by convex geometry and provides a close-form formula for the optimal transportation map.

BID20 provides a new formulation of GANs and variational autoencoders (VAEs), and thus unifies the most two popular methods to train deep generative models.

We'd like to mention other recent interesting research on GANs, e.g., BID16 BID37 BID32 BID29 BID41 BID17 .

We studied the discrimination and generalization properties of GANs with parameterized discriminator class such as neural networks.

A neural distance is guaranteed to be discriminative whenever the linear span of its discriminator set is dense in the bounded continuous function space.

On the other hand, a neural divergence is discriminative whenever the linear span of features defined by the last linear layer of its discriminators is dense in the bounded continuous function space.

We also provided generalization bounds for GANs in different evaluation metrics.

In terms of neural distance, our bounds show that generalization is guaranteed as long as the discriminator set is small enough, regardless of the size of the generator or hypothesis set.

This raises an interesting discriminationgeneralization balance in GANs.

Fortunately, several GAN methods in practice already choose their discriminator set at the sweet point, where both the discrimination and generalization hold.

Finally, our generalization bound in KL divergence provides an explanation on the counter-intuitive behaviors of testing likelihood in GAN training.

There are several directions that we would like to explore in the future.

First of all, in this paper, we do not talk about methods to compute the neural distance/divergence.

This is typically a non-concave maximization problem and is extremely difficult to solve.

Many methods have been proposed to solve this kind of minimax problems, but both stable training methods and theoretical analysis of these algorithms are still missing.

Secondly, our generalization bound depends purely on the discriminator set.

It is possible to obtain sharper bounds by incorporating structural information from the generator set.

Finally, we would like to extend our analysis to conditional GANs (see, e.g., BID30

For neural discriminators, we can use the following bound on the Rademacher complexity of DNNs, which was recently proposed in BID7 .

Theorem A.1.

Let fixed activation functions (σ 1 , . . . , σ L ) and reference matrices (M 1 , . . .

, M L ) be given, where σ i is ρ i -Lipschitz and σ i (0) = 0.

Let spectral norm bounds (s 1 , . . .

, s L ) and matrix (2,1) norm bounds (b 1 , . . .

, b L ) be given.

Let F denote the discriminator set consisting of all choices of neural network f A : DISPLAYFORM0 where A σ := σ max (A) and A 2,1 := ( A :,1 2 , . . .

, A :,m 2 ) 1 are the matrix spectral norm and (2, 1) norm, respectively, and DISPLAYFORM1 is the neural network associated with weight matrices (A 1 , . . .

, A L ).

Moreover, assume that each matrix in (A 1 , . . .

, A L ) has dimension at most W along each axis and define the spectral normalized complexity R as DISPLAYFORM2 Let data matrix X ∈ R m×d be given, where the m rows correspond to data points.

When the sample size m ≥ 3 X F R, the empirical Rademacher complexity satisfieŝ DISPLAYFORM3 where τ = (τ 1 , . . . , τ m ) are the Rademacher random variables which are iid with Pr[τ i = 1] = Pr[τ i = −1] = 1/2, X F is the Frobenius norm of X.Proof.

The proof is the same with the proof of Lemma A.8 in BID7 .

When m ≥ 3 X F R, we use the optimal α = 3 X F R/ √ m to obtain the above result.

Combined with our Theorem 3.1, we obtain the following generalization bound for the neural discriminator set defined in (18).

Corollary A.2.

Suppose that the discriminator set F nn is taken as (18) and that f ∞ ≤ ∆ for any f ∈ F nn .

Let data matrix X ∈ R m×d be the m data points that define the empirical distribution µ m .

Then with probability at least 1 − δ, we have DISPLAYFORM4 where R is the spectral normalized complexity defined in (19) and is the optimization error defined in (9).Proof.

In the proof of Theorem 3.1, instead of using DISPLAYFORM5 log(2/δ) 2m to revise the generalization bound (12) as DISPLAYFORM6 Combining the revised bound with Equation FORMULA44 , we conclude the proof.

Compared to Corollary 3.3, the bound in FORMULA47 gets rid of the number of parameters p, which can be prohibitively large in practice.

Moreover, Corollary A.2 can be directly applied to the spectral normalized GANs BID0 , and may give an explanation of the empirical success of the spectral normalization technique.

With the basic result in Theorem 3.1, we can also discuss the learning bounds of GANs with other choices of non-parametric discriminator sets F. This allows us to highlight the advantages of using parametric neural discriminators.

For simplicity, we assume zero model error and optimization so that the bound is solely based on the generalization error d F (µ,μ m ) between µ and its empirical versionμ m .1.

Bounded Lipschitz distance, F = {f ∈ C(X) : ||f || BL ≤ 1}, which is equivalent to Wasserstein distance when X is compact.

When X is a convex bounded set in R d , we have Corollary 12 in Sriperumbudur et al. FORMULA21 , and hence DISPLAYFORM0 DISPLAYFORM1 .

This is comparable with Corollary 3.2.

This bound is tight.

Assume that µ is the uniform distribution on X. A simple derivation (similar to Lemma 1 in DISPLAYFORM2 for some constant only depending on X. Therefore, one must need at least m = exp(Ω(d)) samples to reduce d F (µ,μ m ), and hence the generalization bound, to O( ).

2.

Total variation (TV) distance, F = {f ∈ C(X) : f ≤ min{1, ∆}}. It is easy to verify that R (µ) m (F) = 2.

Therefore, Eqn.

FORMULA0 cannot guarantee generalization even when we have infinite number of samples, i.e., m → ∞. The estimate given in Eqn.

FORMULA0 is tight.

Assume that µ is the uniform distribution on X. It is easy to see that d TV (µ,μ m ) = 2 almost surely.

Therefore, ν m is close toμ m implies that it is order 1 away from µ, which means that generalization does not hold in this case.

With the statement that training with the TV distance does not generalize, we mean that training with TV distance does not generalize in TV distance.

More precisely, even if the training loss on empirical samples is very small, i.e., TV(μ m , ν m ) = O( ), the TV distance to the unknown target distribution can be large, i.e., d T V (µ, ν m ) = O(1).

However, this does not imply that training with TV distance is useless, because it is possible that training with a stronger metric leads to asymptotic vanishing in a weaker metric.

For example, DISPLAYFORM3 and thus a small d Fnn (µ, ν m ).Take the Wasserstein metric as another example, even though we only establish d W (µ, ν m ) = O(m −1/d ) (assuming zero model error (µ ∈ G) and optimization = 0), it does not eliminate the possibility that the weaker neural distance has a faster convergence rate d Fnn (µ, ν m ) = O(m −1/2 ).

From the practical perspective, however, TV and Wasserstein distances are less clearly favorable than neural distance because the difficulty of calculating and optimizing them.

f -GAN is another broad family of GANs that are based on minimizing f -divergence (also called φ-divergence) BID33 , which includes the original GAN by BID14 .

1 However, φ-divergence has substantially different properties from IPM (see e.g., Sriperumbudur et al. FORMULA21 , and is not defined as the intuitive moment matching form as IPM.

In this Appendix, we extend our analysis to φ-divergence by interpreting it as a form of penalized moment matching.

Similar to the case of IPM, we analyze the neural φ-divergence that restricts the discriminators to parametric function set F for practical computability, and establish its discrimination and generalization properties under mild conditions that practical f -GANs satisfy.

Assume that µ and ν are two distributions on X. Given a convex, lower-semicontinuous univariate function φ that satisfies φ(1) = 0, the related φ-divergence is DISPLAYFORM0 .

If φ is strictly convex, then a standard derivation based on Jensen's inequality shows that φ-divergence is nonnegative and discriminative: d φ (µ || ν) ≥ φ(1) = 0 and the equality holds iff µ = ν.

Different choices of φ recover popular divergences as special cases.

For example, φ(t) = (t − 1) 2 recovers Pearson χ 2 divergence, and φ(t) = (u + 1) log((u + 1)/2) + u log u gives the Jensen-Shannon divergence used in the vanilla GAN Goodfellow et al. (2014) .

In this work, we find it helps to develop intuition by introducing another convex function ψ(t) := φ(t + 1), defined by shifting the input variable of φ by +1; the φ-divergence becomes DISPLAYFORM0 where we should require that ψ(0) = 0; in right hand side of FORMULA53 , we assume ρ µ and ρ ν are the density functions of µ and ν, respectively, under a base measure τ .

The key advantage of introducing ψ is that it gives a suggestive variational representation that can be viewed as a regularized moment matching.

Specially, assume ψ * is the convex conjugate of ψ, that is, ψ * (t) = sup y {yt − ψ(y)}. By standard derivation, we can show that DISPLAYFORM1 where A is the class of all functions f : X → dom(ψ * ) where dom(ψ * ) = {t : ψ * (t) ∈ R}, and the equality holds if ϕ * (ρµ (x) ρν (x) − 1) ∈ A where ϕ is the inverse function of ψ * .

In (24), the term DISPLAYFORM2 , as we show in Lemma B.1 in sequel, can be viewed as a type of complexity penalty on f that ensures the supreme is finite.

This is in contrast with the IPM d F (µ, ν) in which the complexity constraint is directly imposed using the function class F, instead of a regularization term.

Proof.

i) It is obvious that Ψ ν,ψ * [f ] is convex given that f * is convex.

By the convex conjugate, we have ψ(t) = sup y ty − ψ * (y) .

Take t = 0 and note that ψ(0) = 0, then we have ψ DISPLAYFORM3 ii) If ψ is strictly convex, then ψ * is also strictly convex.

This implies there exists at most a single value b 0 such that ψ * (c) = 0.

Given that ψ * (y) ≥ 0 for ∀y, we arrive that E x∼ν [ψ * (f (x))] = 0 implies ψ * (f (x)) = 0 almost surely under x ∼ ν, which then implies f (x) = b 0 almost surely.

In practice, it is impossible to numerically optimize over the class of all functions in (24).

Instead, practical f -GANs restrict the optimization to a parametric set F of neural networks, yielding the following neural φ-divergence: DISPLAYFORM4 Note that this can be viewed as a generalization of the F-related IPM d F (µ, ν) by considering ψ * = 0.

However, the properties of the neural φ-divergence can be significantly different from that of d F (µ, ν).

For example, d φ,F (µ || ν) is not even guaranteed to be non-negative for arbitrary discriminator sets F because of the negative regularization term.

Fortunately, we can still establish the non-negativity and discriminative property of d φ,F (µ || ν) under certain weak conditions on F. Moreover, the property that d F (µ, ν) = 0 implies moment matching on F, which is the key step to establish the discriminative power, is not necessarily true for neural divergence.

Fortunately, it turns out that d φ,F (µ || ν) = 0 implies moment matching on features defined by the last linear layer of discriminators.

Theorem B.1.

Assume F includes the constant function b 0 ∈ R, which satisfies ψ * (b 0 ) = 0 as defined in Lemma B.1.

We have DISPLAYFORM5 In other words, moment matching on F is a sufficient condition of zero neural φ-divergence.ii) Further, we assume F has the following form: DISPLAYFORM6 where F 0 is any function set, and α f0 > 0 is positive number associated with each f 0 ∈ F 0 , and c 0 is a constant and σ : R → R is any function that satisfies σ(c 0 ) = b 0 and σ (c 0 ) > 0.

Here σ can be viewed as the output activation function of a deep neural network whose previous layers are specified by DISPLAYFORM7 In other words, moment matching on F 0 is a necessary condition of zero neural φ-divergence.

DISPLAYFORM8 Condition (26) defines a commonly used structure of F that naturally satisfied by the f -GANs used in practice; in particular, the output activation function σ plays the role of ensuring the output of F respects the input domain of the convex function ψ * .

For example, the vanilla GAN has ψ * = − log(1 − exp(t)) − t with an input domain of (−∞, 0), and activation function is taken to be σ(t) = − log(1 + exp(−t)).

See Table 2 of BID33 for the list of output activation functions related to commonly used ψ.

Proof of Theorem B.1.

i) because b 0 ∈ F and ψ DISPLAYFORM9

By the differentiability assumptions, DISPLAYFORM0 where we used the fact that ψ * (σ(c 0 )) = ψ * (b 0 ) = 0 and ψ * (b 0 ) = 0 because b 0 is a differentiable minimum point of ψ * .

Taking the limit of α → 0 on both sides of (27), we get DISPLAYFORM1 .

The same argument applies to −f 0 , and we thus we finally obtain DISPLAYFORM2 iii) Combining Theorem 2.2 and the last point, we directly get the result.

Remark B.1.

Our results on neural φ-divergence can in general extended to the more unified framework of BID28 in which divergences of form max f E (x,y)∼µ⊗ν [f (x, y)] are studied.

We choose to focus on φ-divergence because of its practical importance.

Our Theorem B.1 i) can be viewed as a special case of Theorem 4 of BID28 and our Theorem B.1 ii) is related to Theorem 5 of BID28 .

However, Theorem 5 of BID28 requires a rather counterintuitive condition, while our condition in Theorem B.1 ii) is clear and satisfied by all φ-divergence listed in BID33 .

DISPLAYFORM3 Further, if there exists C > 0 such that F ⊂ {f ∈ C(X) : f Lip ≤ C}, we have DISPLAYFORM4 Notice that we assume that (X, d X ) be a compact metric space here for simplicity.

A non-compact result is available but its proof is messy and non-intuitive.

Proof.

The first half is a direct application of Theorem B.1 and Theorem 10 in BID28 .For the second half, we have DISPLAYFORM5 where we use Theorem B.1 i) in the first inequality and the Lipschitz condition of F in the second ineqaulity.

Since d W metrizes the weak convergence for compact X, we obtain DISPLAYFORM6

Similar to the case of neural distance, we can establish generalization bounds for neural φ-divergence.

Theorem B.3.

Assume that f ∞ ≤ ∆ for any f ∈ F.μ m is an empirical distribution with m samples from µ, and DISPLAYFORM0 Then with probability at least 1 − 2δ, we have DISPLAYFORM1 where R (µ) m (F) is the Rademacher complexity of F.Notice that the only difference between Theorem 3.1 and Theorem B.3 is that the failure probability change from δ to 2δ.

This comes from the fact that F is typically not even in the neural divergence case.

For example, the vanilla GAN takes σ(t) = − log(1 + exp(−t)) as the output activation function, and thus f ≤ 0 for all f ∈ F.Proof of Theorem B.3.

With the same argument in Equation FORMULA0 , we obtain DISPLAYFORM2 Although F is not even, we have DISPLAYFORM3 Standard argument on Rademacher complexity (same in the proof of Theorem 3.1) gives with probalibity at least 1 − δ, DISPLAYFORM4 With the same argument, we obtain that with probalibity at least 1 − δ, DISPLAYFORM5 Combining all the results above, we conclude the proof.

With Theorem B.3, we obtain generalization bounds for difference choices of F, as we had in section 3.

For example, we have an analog of Corollary 3.3 in the neural divergence setting as follows.

Corollary B.4.

Under the condition of Theorem B.3, we further assume that (1) F = F nn = {f θ : θ ∈ Θ ⊂ [−1, 1] p } is a parametric function class with p parameters in a bounded set Θ and that (2) every f θ is L-Lipschitz continuous with respect to the parameters θ.

Then with probability at least 1 − 2δ, we have DISPLAYFORM6 where C = 16 √ 2πpL + 2∆ 2 log(1/δ).

Proof of Theorem 2.2.

For the sufficient part, the proof is standard and the same as that of the uniqueness of weak convergence.

We refer to Lemma 9.3.2 in BID12 for a complete proof.

For the necessary part, suppose that F ⊂ C b (X) is discriminative in P B (X).

Assume that cl(span(F ∪ {1})) is a strictly closed subspace of C b (X).

Take g ∈ C b (X)\cl(span(F)) and DISPLAYFORM0 .

By the Hahn-Banach theorem, there exists a bounded linear functional L : DISPLAYFORM1 ) and L = 0.

Thanks to the Riesz representation theorem for compact metric spaces, there exists a signed, regular Borel measure DISPLAYFORM2 Suppose m = µ − ν are the Hahn decomposition of m, where µ and ν are two nonnegative Borel measures.

Then we have L(f ) = µ f − ν f for any f ∈ C b (X).

Thanks to L(1) = 0, we have 0 < µ(X) = ν(X) < ∞. We can assume that µ and ν are Borel probability measures. (Otherwise, we can use the normalized nonzero linear functional L/µ(X) whose Hahn decomposition consists of two Borel probability measures.)

Since L(f ) = 0 for any f ∈ cl(span(F)), we have µ f = ν f for any f ∈ F. Since F ⊂ C b (X) is discriminative, we have µ = ν and thus L = 0, which leads to a contradiction.

Proof of Corollary 2.4.

Thanks to {λθ : DISPLAYFORM3 where we used σ(u) = max{u, 0} α in the last step.

Therefore, we have DISPLAYFORM4 Thanks to Theorem 2.3, we know that spanF nn is dense in C b (X).Proof of Theorem 2.5.

Given a function g ∈ C b (X), we say that g is approximated by F with error decay function (r) if for any r ≥ 0, there exists f r ∈ spanF with ||f r || F ,1 ≤ r such that ||f − f r || ∞ ≤ (r).

Obviously, (r) is an non-increasing function w.r.t.

r. Thanks to cl(spanF) = C b (X), we have lim r→∞ (r) = 0.

Now denote r n := d F (µ, ν n ) −1/2 and correspondingly f n := f rn .

We have DISPLAYFORM5 If lim n→∞ d F (µ, ν n ) = 0, we have lim n→∞ r n = ∞. Thanks to lim r→∞ (r) = 0, we prove that lim n→∞ | E µ g − E νn g| = 0.

Since this holds true for any g ∈ C b (X), we conclude that ν n weakly converges to µ.If F ⊆ BL C (X) for some C > 0, we have d F (µ, ν) ≤ Cd BL (µ, ν) for any µ, ν.

Because the bounded Lipschitz distance (also called FortetMourier distance) metrizes the weak convergence, we obtain that ν n µ implies d BL (µ, ν n ) → 0, and thus DISPLAYFORM6 The result is obtain by taking infimum over all possible w i .Proof of Proposition 2.7.

For any r ≥ 0, we have DISPLAYFORM7 Taking the infimum on r > 0 on the right side gives the result.

Proof of Corollary 2.8.

Proposition 5 of BID4 shows that for any bounded Lipschitz function g that satisfies ||g|| BL : = max{||g|| ∞ , ||g|| Lip } ≤ η, we have (r) = O(η(r/η) −1/(α+(d−1)/2) log(r/η)).

Using Proposition 2.7, we get DISPLAYFORM8 The result follows BL(µ, ν) = sup g {| E µ g − E ν g| : ||g|| BL ≤ 1}.

Proof of Equation (10) Using the standard derivation and the optimality condition (9), we have DISPLAYFORM0 Therefore, we obtain DISPLAYFORM1 Combining with the definition (1), we obtain DISPLAYFORM2 Proof of Theorem 3.1.

First of all, since F is even, we have DISPLAYFORM3 Since f takes values in [−∆, ∆], changing X i to another independent copy X i can change h by no more than 2∆/m.

McDiarmid's inequality implies that with probability at least 1 − δ, DISPLAYFORM4 Standard argument on Rademacher complexity gives DISPLAYFORM5 Combining the two estimates above and Eqn. (10), we conclude the proof.

Proof of Corollary 3.4.

Lemma 22 in BID6 shows that if DISPLAYFORM6 Combined with Theorem 3.1, we conclude the proof.

Proof of Corollary 3.5.

Use Proposition 3.1 and note that KL(µ, DISPLAYFORM7

In this section, we will test our analysis of the consistency of GAN objective and likelihood objective on two toy datasets, e.g., a 2D Gaussian dataset and a 2D 8-Gaussian mixture dataset.

The underlying ground-truth distribution is a 2D Gaussian with mean (0.5, −0.5) and covariance matrix 1 12817 15 15 17 .

We take 10 5 samples for training, and 1000 samples for testing.

For a 2D Gaussian distribution, we use the following generator DISPLAYFORM0 where DISPLAYFORM1 is a standard 2D normal random vector, and l ∈ R, s = DISPLAYFORM2 2 are trainable parameters in the generator.

We train the generative model by WGAN with weight clipping.

In the first experiment, the discriminator set is a neural network with one hidden layer and 500 hidden neurons, i.e., Motivated by Corollary 3.5, in the second experiment, we take the discriminators to be the logdensity ratio between two Gaussian distributions, which are quadratic polynomials: We plot their results in FIG2 .

We can see that both discriminators behave well: The training loss (the neural distance) converge to zero, and the testing log likelihood increases monotonically during the training.

However, the quadratic polynomial discriminators F quad yields higher testing log likelihood and better generative model at the convergence.

This is expected because Corollary 3.5 guarantees that the testing log likelihood is bounded by the GAN loss (up to a constant), while it is not true for F nn .

DISPLAYFORM3 We can also maximize the likelihood (MLE) on the training dataset to train the model, and we show its result in FIG7 .

We can see that both MLE and Q-GAN (refers to WGAN with the quadratic discriminator F quad ) yield similar results.

However, directly maximizing the likelihood converges much faster than the WGAN in this example.

In this simple Gaussian example, the WGAN loss and the testing log likelihood are consistent.

We indeed observe that by carefully choosing the discriminator set (as suggested in Corollary 3.5), the testing log likelihood can be simultaneously optimized as we optimize the GAN objective.

The underlying ground truth distribution is a 2D Gaussian mixture with 8 Gaussians and with equal weights.

Their centers are distributed equally on the circle centered at the origin and with radius √ 2, and their standard deviations are all 0.01414.

We take 10 5 samples as training dataset, and 1000 samples as testing dataset.

We show one batch (256) of training dataset and the testing dataset in FIG8 .

Note that that the density of the ground-truth distribution is highly singular.

We still use Eqn. (33) as the generator for a single Gaussian component.

Our generator assume that there are 8 Gaussian components and they have equal weights, and thus our generator does not have any modeling error.

The training parameters are eight sets of scaling and biasing parameters in Eqn. (33), each for one Gaussian component.

We first train the model by WGAN with clipping.

We use an MLP with 4 hidden layers and relu activations as the discriminator set.

We show the result in FIG9 .

We can see that the generator's samples are nearly indistinguishable from the real samples.

However, the GAN loss and the log likelihood are not consistent.

In the initial stage of training, both the negative GAN loss and log likelihood are increasing.

As the training goes on, the generator's density gets more and more singular, the log likelihood behaves erratically in the latter stage of training.

Although the negative GAN loss is still increasing, the log likelihood oscillates a lot, and in fact over half of time the log likelihood is −∞. We show the generated samples at intermediate steps in FIG10 , and we indeed see that the likelihood starts to oscillate violently when the generator's distribution gets singular.

This inconsistency between GAN loss and likelihood is observed by other works as well.

The reason for this consistency is that the neural discriminators are not a good approximation of the singular density ratios.

We also train the model by maximizing likelihood on the training dataset.

We show the result in FIG11 .

We can see that the maximal likelihood training got stuck in a local minimum, and failed to exactly recover all 8 components.

The log likelihood on training and testing datasets are consistent as expected.

Although the log likelihood (≈ 2.7) obtained by maximizing likelihood is higher than Published as a conference paper at ICLR 2018 that (≈ 2.0) obtained by WGAN training, its generator is obviously worse than what we obtained in WGAN training.

The reason for this is that the negative log-likelihood loss has many local minima, and maximizing likelihood is easy to get trapped in a local minimum.

The FlowGAN BID15 proposed to combine the WGAN loss and the log likelihood to solve the inconsistency problem.

We showed the FlowGAN result on this dataset in FIG12 .

We can see that training by FlowGAN indeed makes the training loss and log likelihood consistent.

However, FlowGAN got stuck in a local minimum as maximizing likelihood did, which is not desirable.

<|TLDR|>

@highlight

This paper studies the discrimination and generalization properties of GANs when the discriminator set is a restricted function class like neural networks.

@highlight

Balances capacities of generator and discriminator classes in GANs by guaranteeing that induced IPMs are metrics and not pseudo metrics

@highlight

This paper provides a mathematical analysis of the role of the size of the adversary/discriminator set in GANs
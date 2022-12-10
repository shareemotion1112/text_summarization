While Generative Adversarial Networks (GANs) have empirically produced impressive results on learning complex real-world distributions, recent works have shown that they suffer from lack of diversity or mode collapse.

The theoretical work of Arora et al. (2017a) suggests a dilemma about GANs’ statistical properties: powerful discriminators cause overfitting, whereas weak discriminators cannot detect mode collapse.

By contrast, we show in this paper that GANs can in principle learn distributions in Wasserstein distance (or KL-divergence in many cases) with polynomial sample complexity, if the discriminator class has strong distinguishing power against the particular generator class (instead of against all possible generators).

For various generator classes such as mixture of Gaussians, exponential families, and invertible and injective neural networks generators, we design corresponding discriminators (which are often neural nets of specific architectures) such that the Integral Probability Metric (IPM) induced by the discriminators can provably approximate the Wasserstein distance and/or KL-divergence.

This implies that if the training is successful, then the learned distribution is close to the true distribution in Wasserstein distance or KL divergence, and thus cannot drop modes.

Our preliminary experiments show that on synthetic datasets the test IPM is well correlated with KL divergence or the Wasserstein distance, indicating that the lack of diversity in GANs may be caused by the sub-optimality in optimization instead of statistical inefficiency.

In the past few years, we have witnessed great empirical success of Generative Adversarial Networks (GANs) (Goodfellow et al., 2014) in generating high-quality samples in many domains.

Various ideas have been proposed to further improve the quality of the learned distributions and the stability of the training. (See e.g., BID0 Odena et al., 2016; Huang et al., 2017; Radford et al., 2016; Tolstikhin et al., 2017; Salimans et al., 2016; Jiwoong Im et al., 2016; Durugkar et al., 2016; Xu et al., 2017) and the reference therein.)

However, understanding of GANs is still in its infancy.

Do GANs actually learn the target distribution?

Recent work (Arora et al., 2017a; b; Dumoulin et al., 2016) has both theoretically and empirically brought the concern to light that distributions learned by GANs suffer from mode collapse or lack of diversity -the learned distribution tends to miss a significant amount of modes of the target distribution (elaborated in Section 1.1).

The main message of this paper is that the mode collapse can be in principle alleviated by designing proper discriminators with strong distinguishing power against specific families of generators such as special subclasses of neural network generators (see Section 1.2 and 1.3 for a detailed introduction.)

We mostly focus on the Wasserstein GAN (WGAN) formulation BID0 in this paper.

Define the F-Integral Probability Metric (F-IPM) (Müller, 1997) between distributions p, q as W F (p, q) := sup DISPLAYFORM0 Given samples from distribution p, WGAN sets up a family of generators G, a family of discriminators F, and aims to learn the data distribution p by solving DISPLAYFORM1 wherep n denotes "the empirical version of the distribution p", meaning the uniform distribution over a set of n i.i.d samples from p (and similarlyq m .)When F = {all 1-Lipschitz functions}, IPM reduces to the Wasserstein-1 distance W 1 .

In practice, parametric families of functions F such as multi-layer neural networks are used for approximating Lipschitz functions, so that we can empirically optimize this objective eq. (2) via gradient-based algorithms as long as distributions in the family G have parameterized samplers. (See Section 2 for more details.)One of the main theoretical and empirical concerns with GANs is the issue of "modecollapse" (Arora et al., 2017a; Salimans et al., 2016) -the learned distribution q tends to generate high-quality but low-diversity examples.

Mathematically, the problem apparently arises from the fact that IPM is weaker than W 1 , and the mode-dropped distribution can fool the former (Arora et al., 2017a) : for a typical distribution p, there exists a distribution q such that simultaneously the followings happen:W F (p, q) ε and W 1 (p, q) 1.where , hide constant factors.

In fact, setting q =p N with N = R(F)/ε 2 , where R(F) is a complexity measure of F (such as Rademacher complexity), q satisfies eq. (3) but is clearly a mode-dropped version of p when p has an exponential number of modes.

Reasoning that the problem is with the strength of the discriminator, a natural solution is to increase it to larger families such as all 1-Lipschitz functions.

However, Arora et al. (Arora et al., 2017a) points out that Wasserstein-1 distance doesn't have good generalization properties: the empirical Wasserstein distance used in the optimization is very far from the population distance.

Even for a spherical Gaussian distribution p = N(0, 1 d I d×d ) (or many other typical distributions), when the distribution q is exactly equal to p, lettingq m andp n be two empirical versions of q and p with m, n = poly(d), we have with high probability, W 1 (p n ,q m ) 1 even though W 1 (p, q) = 0.Therefore even when learning succeeds (p = q), it cannot be gleaned from the empirical version of W 1 .The observations above pose a dilemma in establishing the theories of GANs: powerful discriminators cause overfitting, whereas weak discriminators result in diversity issues because IPM doesn't approximate the Wasserstein distance The lack of diversity has also been observed empirically by (Srivastava et al., 2017; Di & Yu, 2017; Borji, 2018; Arora et al., 2017b ).

This paper proposes a resolution to the conundrum by designing a discriminator class F that is particularly strong against a specific generator class G. We say that a discriminator class F (and its IPM W F ) has restricted approximability w.r.t.

a generator class G and the data distribution p, if F can distinguish p and any q ∈ G approximately as well as all 1-Lipschitz functions can do:W F has restricted approximability w.r.t.

G and p ∀q ∈ G, γ L (W 1 (p, q)) W F (p, q) γ U (W 1 (p, q)),where γ L (·) and γ U (·) are two monotone nonnegative functions with γ L (0) = γ U (0) = 0.

The paper mostly focuses on γ L (t) = t α with 1 ≤ α ≤ 2 and γ U (t) = t, although we use the term "restricted approximability" more generally for this type of result (without tying it to a concrete definition of γ).

In other words, we are looking for discriminators F so that F-IPM can approximate the Wasserstein distance W 1 for the data distribution p and any q ∈ G.Throughout the rest of this paper, we will focus on the realizable case, that is, we assume p ∈ G, in which case we say W F has restricted approximability with respect to G if eq. (5) holds for all p, q ∈ G. We note, however, that such a framework allows the non-realizible case p / ∈ G in full generality (for example, results can be established through designing F that satisfies the requirement in Lemma 4.3).A discriminator class F with restricted approximability resolves the dilemma in the following way.

First, F avoids mode collapse -if the IPM between p and q is small, then by the left hand side of eq. (5), p and q are also close in Wasserstein distance and therefore significant mode-dropping cannot happen.

1 Second, we can pass from population-level guarantees to empirical-level guarantees -as shown in Arora et al. (2017a) , classical capacity bounds such as the Rademacher complexity of F relate W F (p, q) to W F (p n ,q m ).

Therefore, as long as the capacity is bounded, we can expand on eq. (5) to get a full picture of the statistical properties of Wasserstein GANs: DISPLAYFORM0 Here the first inequality addresses the diversity property of the distance W F , and the second approximation addresses the generalization of the distance, and the third inequality provides the reverse guarantee that if the training fails to find a solution with small IPM, then indeed p and q are far away in Wasserstein distance.

2 To the best of our knowledge, this is the first theoretical framework that tackles the statistical theory of GANs with polynomial samples.

The main body of the paper will develop techniques for designing discriminator class F with restricted approximability for several examples of generator classes including simple classes like mixtures of Gaussians, exponential families, and more complicated classes like distributions generated by invertible neural networks.

In the next subsection, we will show that properly chosen F provides diversity guarantees such as inequalities eq. (5).

We start with relatively simple families of distributions G such as Gaussian distributions and exponential families, where we can directly design F to distinguish pairs of distribution in G. As we show in Section 3, for Gaussians it suffices to use one-layer neural networks with ReLU activations as discriminators, and for exponential families to use linear combinations of the sufficient statistics.

In Section 4, we study the family of distributions generated by invertible neural networks.

We show that a special type of neural network discriminators with one additional layer than the generator has restricted approximability 3 .

We show this discriminator class guarantees that DISPLAYFORM0 where here we hide polynomial dependencies on relevant parameters (Theorem 4.2).

We remark that such networks can also produce an exponentially large number of modes due to the non-linearities, and our results imply that if W F (p, q) is small, then most of these exponential modes will show up in the learned distribution q.

One limitation of the invertibility assumption is that it only produces distributions supported on the entire space.

The distribution of natural images is often believed to reside approximately on a low-dimensional manifold.

When the distribution p have a Lebesgue measure-zero support, the KLdivergence (or the reverse KL-divergence) is infinity unless the support of the estimated distribution coincides with the support of p. 4 Therefore, while our proof makes crucial use of the KL-divergence in the invertible case, the KL-divergence is fundamentally not the proper measurement of the statistical distance for the cases where both p and q have low-dimensional supports.

The crux of the technical part of the paper is to establish the approximation of Waserstein distance by IPMs for generators with low-dimensional supports.

We will show that a variant of an IPM can still be sandwiched by Wasserstein distance as in form of eq. (5) without relating to KL-divergence (Theorem 4.5).

This demonstrates the advantage of GANs over MLE approach on learning distributions with low-dimensional supports.

As the main proof technique, we develop tools for approximating the log-density of a smoothed neural network generator.

We demonstrate in synthetic and controlled experiments that the IPM correlates with the Wasserstein distance for low-dimensional distributions with measure-zero support and correlates with KLdivergence for the invertible generator family (where computation of KL is feasible) (Section 5 and Appendix G.) The theory suggests the possibility that when the KL-divergence or Wasserstein distance is not measurable in more complicated settings, the test IPM could serve as a candidate alternative for measuring the diversity and quality of the learned distribution.

We also remark that on real datasets, often the optimizer is tuned to carefully balance the learning of generators and discriminators, and therefore the reported training loss is often not the test IPM (which requires optimizing the discriminator until optimality.)

Anecdotally, the distributions learned by GANs can often be distinguished by a well-trained discriminator from the data distribution, which suggests that the IPM is not well-optimized (See Lopez-Paz & Oquab (2016) for analysis of for the original GANs formulation.)

We conjecture that the lack of diversity in real experiments may be caused by sub-optimality of the optimization, rather than statistical inefficiency.

Various empirical proxy tests for diversity, memorization, and generalization have been developed, such as interpolation between images (Radford et al., 2016) , semantic combination of images via arithmetic in latent space (Bojanowski et al., 2017) , classification tests (Santurkar et al., 2017) , etc.

These results by and large indicate that while "memorization" is not an issue with most GANs, lack of diversity frequently is.

As discussed thoroughly in the introduction, Arora et al. (2017a; b) formalized the potential theoretical sources of mode collapse from a weak discriminator, and proposed a "birthday paradox" that convincingly demonstrates this phenomenon is real.

Many architectures and algorithms have been proposed to remedy or ameliorate mode collapse (Dumoulin et al., 2016; Srivastava et al., 2017; Di & Yu, 2017; Borji, 2018; Lin et al., 2017) with varying success.

Feizi et al. (2017) showed provable guarantees of training GANs with quadratic discriminators when the generators are Gaussians.

However, to the best of our knowledge, there are no provable solutions to this problem in more substantial generality.

The inspiring work of Zhang et al. (Zhang et al., 2017) shows that the IPM is a proper metric (instead of a pseudo-metric) under a mild regularity condition.

Moreover, it provides a KL-divergence bound with finite samples when the densities of the true and estimated distributions exist.

Our Section 4.1 can be seen as an extension of (Zhang et al., 2017, Proposition 2.9 and Corollary 3.5) .

The strength in our work is that we develop statistical guarantees in Wasserstein distance for distributions such as injective neural network generators, where the data distribution resides on a low-dimensional manifold and thus does not have proper density.

Liang (2017) considers GANs in a non-parametric setup, one of the messages being that the sample complexity for learning GANs improves with the smoothness of the generator family.

However, the rate they derive is non-parametric -exponential in the dimension -unless the Fourier spectrum of the target family decays extremely fast, which can potentially be unrealistic in practical instances.

The invertible generator structure was used in Flow-GAN (Grover et al., 2018) , which observes that GAN training blows up the KL on real dataset.

Our theoretical result and experiments show that successful GAN training (in terms of the IPM) does imply learning in KL-divergence when the data distribution can be generated by an invertible neural net.

This suggests, along with the message in (Grover et al., 2018) , that the real data cannot be generated by an invertible neural network.

In addition, our theory implies that if the data can be generated by an injective neural network (Section 4.2), we can bound the closeness between the learned distribution and the true distribution in Wasserstein distance (even though in this case, the KL divergence is no longer an informative measure for closeness.)

The notion of IPM (recall the definition in eq. (1)) includes a number of statistical distances such as TV (total variation) and Wasserstein-1 distance by taking F to be 1-bounded and 1-Lipschitz functions respectively.

When F is a class of neural networks, we refer to the F-IPM as the neural net IPM.

There are many distances of interest between distributions that are not IPMs, two of which we will particularly focus on: the KL divergence D kl (p q) = E p [log p(X) − log q(X)] (when the densities exist), and the Wasserstein-2 distance, defined as DISPLAYFORM0 Π be the set of couplings of (p, q).

We will only consider distributions with finite second moments, so that W 1 and W 2 exist.

For any distribution p, we letp n be the empirical distribution of n i.i.d.

samples from p. The Rademacher complexity of a function class DISPLAYFORM1 .

and ε i ∼ {±1} are independent.

We define R n (F, G) = sup p∈G R n (F, p) to be the largest Rademacher complexity over p ∈ G. The training IPM loss (over the entire dataset) for the Wasserstein GAN, assuming discriminator reaches optimality, is Eqn [W F (p n ,q n )] 6 .

Generalization of the IPM is governed by the quantity R n (F, G), as stated in the following result (see Appendix A.1 for the proof): Theorem 2.1 (Generalization, c.f. (Arora et al., 2017a) ).

For any p ∈ G, we have that DISPLAYFORM2 Miscellaneous notation.

We let N(µ, Σ) denote a (multivariate) Gaussian distribution with mean µ and covariance Σ. For quantities a, b > 0 a b denotes that a ≤ Cb for a universal constant C > 0 unless otherwise stated explicitly.

As a warm-up, we design discriminators with restricted approximability for relatively simple parameterized distributions such Gaussian distributions, exponential families, and mixtures of Gaussians.

We first prove that one-layer neural networks with ReLU activation are strong enough to distinguish Gaussian distributions with the restricted approximability guarantees.

We consider the set of Gaussian distributions with bounded mean and well-conditioned covariance DISPLAYFORM0 Here D, σ min and σ max are considered as given hyper-parameters.

We will show that the IPM W F induced by the following discriminators has restricted approximability w.r.t.

G: DISPLAYFORM1 Theorem 3.1.

The set of one-layer neural networks (F defined in eq. (6)) has restricted approximability w.r.t.

the Gaussian distributions in G in the sense that for any p, q ∈ G DISPLAYFORM2 .

DISPLAYFORM3 Apart from absolute constants, the lower and upper bounds differ by a factor of 1/ √ d. 7 We point out that the 1/ √ d factor is not improvable unless using functions more sophisticated than Lipschitz functions of one-dimensional projections of x. Indeed, W F (p, q) is upper bounded by the maximum Wasserstein distance between one-dimensional projections of p, q, which is on the order of W 1 (p, q)/ √ d when p, q have spherical covariances.

The proof is deferred to Section B.1.Extension to mixture of Gaussians.

Discriminator family F with restricted approximability can also be designed for mixture of Gaussians.

We defer this result and the proof to Appendix C.

Now we consider exponential families and show that the linear combinations of the sufficient statistics are a family of discriminators with restricted approximability.

Concretely, let G = {p θ : θ ∈ Θ ⊂ R k } be an exponential family, where DISPLAYFORM0 k is the vector of sufficient statistics, and Z(θ) is the partition function.

Let the discriminator family be all linear functionals over the features T (x): F = {x → v, T (x) : v 2 ≤ 1}. Theorem 3.2.

Let G be the exponential family and F be the discriminators defined above.

Assume that the log partition function log Z(θ) satisfies that γI ∇ 2 log Z(θ) βI. Then we have for any DISPLAYFORM1 If we further assume X has diameter D and T (x) is L-Lipschitz in X .

Then, DISPLAYFORM2 Moreover, F has Rademacher complexity bound R n (F, G) ≤ DISPLAYFORM3 We note that the log partition function log Z(θ) is always convex, and therefore our assumptions only require in addition that the curvature (i.e. the Fisher information matrix) has a strictly positive lower bound and a global upper bound.

For the bound eq. (8), some geometric assumptions on the sufficient statistics are necessary because the Wasserstein distance intrinsically depends on the underlying geometry of x, which are not specified in exponential families by default.

The proof of eq. FORMULA15 follows straightforwardly from the standard theory of exponential families.

The proof of eq. (8) requires machinery that we will develop in Section 4 and is therefore deferred to Section B.2.

In this section, we design discriminators with restricted approximability for neural net generators, a family of distributions that are widely used in GANs to model real data.

In Section 4.1 we consider the invertible neural networks generators which have proper densities.

In Section 4.2, we extend the results to the more general and challenging setting of injective neural networks generators, where the latent variables are allowed to have lower dimension than the observable dimensions (Theorem 4.5) and the distributions no longer have densities.

In this section, we consider the generators that are parameterized by invertible neural networks 8 .

Concretely, let G be a family of neural networks G = {G θ : θ ∈ Θ}. Let p θ be the distribution of DISPLAYFORM0 where G θ is a neural network with parameters θ and γ ∈ R d standard deviation of hidden factors.

By allowing the variances to be non-spherical, we allow each hidden dimension to have a different impact on the output distribution.

In particular, the case γ = [1 k , δ1 d−k ] for some δ 1 has the ability to model data around a "k-dimensional manifold" with some noise on the level of δ.

We are interested in the set of invertible neural networks G θ .

We let our family G consist of standard -layer feedforward nets x = G θ (z) of the form DISPLAYFORM1 where W i ∈ R d×d are invertible, b i ∈ R d , and σ : R → R is the activation function, on which we make the following assumption:Assumption 1 (Invertible generators).

Let R W , R b , κ σ , β σ > 0, δ ∈ (0, 1] be parameters which are considered as constants (that may depend on the dimension).

We consider neural networks G θ that are parameterized by parameters θ = (W i , b i ) i∈[ ] belonging to the set DISPLAYFORM2 The activation function σ is twice-differentiable with DISPLAYFORM3 The standard deviation of the hidden factors satisfy γ i ∈ [δ, 1].Clearly, such a neural net is invertible, and its inverse is also a feedforward neural net with activation σ −1 .

We note that a smoothed version of Leaky ReLU (Xu et al., 2015) satisfies all the conditions on the activation functions.

Further, it is necessary to impose some assumptions on the generator networks because arbitrary neural networks are likely to be able to implement pseudo-random functions which can't be distinguished from random functions by even any polynomial time algorithms.

Lemma 4.1.

For any θ ∈ Θ, the function log p θ can be computed by a neural network with at most + 1 layers, O( d 2 ) parameters, and activation function among {σ DISPLAYFORM4 where DISPLAYFORM5 As a direct consequence, the following family F of neural networks with activation functions above of at most + 2 layers contains all the functions {log p − log q : p, q ∈ G} : DISPLAYFORM6 We note that the exact form of the parameterized family F is likely not very important in practice, since other family of neural nets also possibly contain good approximations of log p − log q (which can be seen partly from experiments in Section G.)The proof builds on the change-of-variable formula log p θ (x) = log φ γ (G −1 DISPLAYFORM7 | (where φ γ is the density of Z ∼ N(0, diag(γ 2 ))) and the observation that G −1 θ is a feedforward neural net with layers.

Note that the log-det of the Jacobian involves computing the determinant of the (inverse) weight matrices.

A priori such computation is non-trivial for a given G θ .

However, it's just some constant that does not depend on the input, therefore it can be representable by adding a bias on the final output layer.

This frees us from further structural assumptions on the weight matrices (in contrast to the architectures in flow-GANs (Gulrajani et al., 2017) ).

We defer the proof of Lemma 4.1 to Section D.2.

Theorem 4.2.

Suppose G = {p θ : θ ∈ Θ} is the set of invertible-generator distributions as defined in eq. (9) satisfying Assumption 1.

Then, the discriminator class F defined in Lemma 4.1 has restricted approximability w.r.t.

G in the sense that for any p, q ∈ G, DISPLAYFORM8 The proof of Theorem 4.2 uses the following lemma that relates the KL divergence to the IPM when the log densities exist and belong to the family of discriminators.

Lemma 4.3 (Special case of (Zhang et al., 2017, Proposition 2.9)).

Let ε > 0.

Suppose F satisfies that for every q ∈ G, there exists f ∈ F such that f − (log p − log q) ∞ ≤ , and that all the functions in F are L-Lipschitz.

Then, DISPLAYFORM9 We outline a proof sketch of Theorem 4.2 below and defer the full proof to Appendix D.3.

As we choose the discriminator class as in Lemma 4.1 which implements log p − log q for any p, q ∈ G, DISPLAYFORM10 It thus suffices to (1) lower bound this quantity by the Wasserstein distance and (2) upper bound W F (p, q) by the Wasserstein distance.

To establish (1), we will prove in Lemma D.3 that for any p, q ∈ G, DISPLAYFORM11 Such a result is the simple implication of transportation inequalities by Bobkov-Götze and Gozlan (Theorem D.1), which state that if X ∼ p (or q) and f is 1-Lipschitz implies that f (X) is subGaussian, then the inequality above holds.

In our invertible generator case, we have X = G θ (Z) where Z are independent Gaussians, so as long as G θ is suitably Lipschitz, f (X) = f (G θ (Z)) is a sub-Gaussian random variable by the standard Gaussian concentration result (Vershynin, 2010).The upper bound (2) would have been immediate if functions in F are Lipschitz globally in the whole space.

While this is not strictly true, we give two workarounds -by either doing a truncation argument to get a W 1 bound with some tail probability, or a W 2 bound which only requires the Lipschitz constant to grow at most linearly in x 2 .

This is done in Theorem D.2 as a straightforward extension of the result in (Polyanskiy & Wu, 2016) .Combining the restricted approximability and the generalization bound, we immediately obtain that if the training succeeds with small expected IPM (over the randomness of the learned distributions), then the estimated distribution q is close to the true distribution p in Wasserstein distance.

Corollary 4.4.

In the setting of Theorem 4.2, with high probability over the choice of training datâ p n , we have that if the training process returns a distribution DISPLAYFORM12 We note that the training error is measured by Eqm [W F (p n ,q m )], the expected IPM over the randomness of the learned distributions, which is a measurable value because one can draw fresh samples from q to estimate the expectation.

It's an important open question to design efficient algorithms to achieve a small training error according to this definition, and this is left for future work.

In this section we consider injective neural network generators (defined below) which generate distributions residing on a low dimensional manifold.

This is a more realistic setting than Section 4.1 for modeling real images, but technically more challenging because the KL divergence becomes infinity, rendering Lemma 4.3 useless.

Nevertheless, we design a novel divergence between two distributions that is sandwiched by Wasserstein distance and can be optimized as IPM.

9 Therefore, G θ is invertible only on the image of G θ , which is a kdimensional manifold in R d .

Let G be the corresponding family of distributions generated by neural nets in G.Our key idea is to design a variant of the IPM, which provably approximates the Wasserstein distance.

Let p β denote the convolution of the distribution p with a Gaussian distribution N(0, β 2 I).

We define a smoothed F-IPM between p, q as DISPLAYFORM0 Clearlyd F can be optimized as W F with an additional variable β introduced in the optimization.

We show that for certain discriminator class (see Section E for the details of the construction) such thatd F approximates the Wasserstein distance.

Theorem 4.5 (Informal version of Theorem E.1).

Let G be defined as above.

There exists a discriminator class F such that for any pair of distributions p, q ∈ G, we have DISPLAYFORM1 Furthermore, when n poly(d), we have the generalization bound DISPLAYFORM2 Here poly(d) hides polynomial dependencies on d and several other parameters that will be defined in the formal version (Theorem E.1.)The direct implication of the theorem is that ifd(p n ,q n ) is small for n poly(d), then W (p, q) is guaranteed to be also small and thus we don't have mode collapse.

Our theoretical results on neural network generators in Section 4 convey the message that mode collapse will not happen as long as the discriminator family F has restricted approximability with respect to the generator family G. In particular, the IPM W F (p, q) is upper and lower bounded by the Wasserstein distance W 1 (p, q) given the restricted approximability.

We design certain specific discriminator classes in our theory to guarantee this, but we suspect it holds more generally in GAN training in practice.

We perform two sets of synthetic experiments to confirm that the practice is indeed consistent with our theory.

We design synthetic datasets, set up suitable generators, and train GANs with either our theoretically proposed discriminator class with restricted approximability, or vanilla neural network discriminators of reasonable capacity.

In both cases, we show that IPM is well correlated with the Wasserstein / KL divergence, suggesting that the restricted approximability may indeed hold in practice.

This suggests that the difficulty of GAN training in practice may come from the optimization difficulty rather than statistical inefficiency, as we observe evidence of good statistical behaviors on "typcial" discriminator classes.

We briefly describe the experiments here and defer details of the second experiment to Appendix G.(a) We learn synthetic 2D datasets with neural net generators and discriminators and show that the IPM is well-correlated with the Wasserstein distance (Section 5.1).

(b) We learn invertible neural net generators with discriminators of restricted approximability and vanilla architectures (Appendix G).

We show that the IPM is well-correlated with the KL divergence, both along training and when we consider two generators that are perturbations of each other (the purpose of the latter being to eliminate any effects of the optimization).

In this section, we perform synthetic experiments with WGANs that learn various curves in two dimensions.

In particular, we will train GANs that learn the unit circle and a "swiss roll" curve (Gulrajani et al., 2017) -both distributions are supported on a one-dimensional manifold in R 2 , therefore the KL divergence does not exist, but one can use the Wasserstein distance to measure the quality of the learned generator.

We show that WGANs are able to learn both distributions pretty well, and the IPM W F is strongly correlated with the Wasserstein distance W 1 .

These ground truth distributions are not covered in our Theorems 4.2 and 4.5, but our results show evidence that restricted approximability is still quite likely to hold here.

We set the ground truth distribution to be a unit circle or a Swiss roll curve, sampled from DISPLAYFORM0 Generators and discriminators We use standard two-hidden-layer ReLU nets as both the generator class and the discriminator class.

The generator architecture is 2-50-50-2, and the discriminator architecture is 2-50-50-1.

We use the RMSProp optimizer (Tieleman & Hinton, 2012) as our update rule, the learning rates are 10 −4 for both the generator and discriminator, and we perform 10 steps on the discriminator in between each generator step.

Metric We compare two metrics between the ground truth distribution p and the learned distribution q along training:(1) The neural net IPM W F (p, q), computed on fresh batches from p, q through optimizing a separate discriminator from cold start.(2) The Wasserstein distance W 1 (p, q), computed on fresh batches from p, q using the POT package 10 .

As data are in two dimensions, the empirical Wasserstein distance W 1 (p,q) does not suffer from the curse of dimensionality and is a good proxy of the true Wasserstein distance W 1 (p, q) (Weed & Bach, 2017).

Result See FIG1 for the Swiss roll experiment and FIG4 (in Appendix F) for the unit circle experiment.

On both datasets, the learned generator is very close to the ground truth distribution at iteration 10000.

Furthermore, the neural net IPM and the Wasserstein distance are well correlated.

At iteration 500, the generators have not quite learned the true distributions yet (by looking at the sampled batches), and the IPM and Wasserstein distance are indeed large.

We present the first polynomial-in-dimension sample complexity bounds for learning various distributions (such as Gaussians, exponential families, invertible neural networks generators) using GANs with convergence guarantees in Wasserstein distance (for distributions with low-dimensional supports) or KL divergence.

The analysis technique proceeds via designing discriminators with restricted approximability -a class of discriminators tailored to the generator class in consideration which have good generalization and mode collapse avoidance properties.

We hope our techniques can be in future extended to other families of distributions with tighter sample complexity bounds.

This would entail designing discriminators that have better restricted approximability bounds, and generally exploring and generalizing approximation theory results in the context of GANs.

We hope such explorations will prove as rich and satisfying as they have been in the vanilla functional approximation settings.

DISPLAYFORM0 Taking expectation overp n on the above bound yields DISPLAYFORM1 So it suffices to bound Epn [W F (p,p n )] by 2R n (F, G) and the same bound will hold for q. Let X i be the samples inp n .

By symmetrization, we have DISPLAYFORM2 Adding up this bound and the same bound for q gives the desired result.

B PROOFS FOR SECTION 3 B.1 PROOF OF THEOREM 3.1Recall that our discriminator family is DISPLAYFORM3 Restricted approximability The upper bound W F (p 1 , p 2 ) ≤ W 1 (p 1 , p 2 ) follows directly from the fact that functions in F are 1-Lipschitz.

We now establish the lower bound.

First, we recover the mean distance, in which we use the following simple fact: a linear discriminator is the sum of two ReLU discriminators, or mathematically t = σ(t) − σ(−t).

Taking v = µ1−µ2 µ1−µ2 2, we have DISPLAYFORM4 Therefore at least one of the above two terms is greater than µ 1 − µ 2 2 /2, which shows that DISPLAYFORM5 For the covariance distance, we need to actually compute DISPLAYFORM6 (Defining R(a) = E[max {W + a, 0}] for W ∼ N(0, 1).)

Therefore, the neuron distance between the two Gaussians is DISPLAYFORM7 As a → max {a + w, 0} is strictly increasing for all w, the function R is strictly increasing.

It is also a basic fact that R(0) = 1/ √ 2π.

Consider any fixed v. By flipping the sign of v, we can let v µ 1 ≥ v µ 2 without changing Σ DISPLAYFORM8 As R is strictly increasing, for this choice of (v, b) we have DISPLAYFORM9 Ranging over v 2 ≤ 1 we then have DISPLAYFORM10 The quantity in the supremum can be further bounded as DISPLAYFORM11 .

DISPLAYFORM12 Now, using the perturbation bound (cf. (Schmitt, 1992 , Lemma 2.2)), we get DISPLAYFORM13 DISPLAYFORM14 Combining the above bound with the bound in the mean difference, we get DISPLAYFORM15 The last equality following directly from the closed-form expression of the W 2 distance between two Gaussians (Masarotto et al., 2018, Proposition 3) .

Thus the claimed lower bound holds with c = 1/(2 √ 2π).

We use the W 2 distance to bridge the KL and the F-distance, which uses the machinery developed in Section D. Let p 1 , p 2 be two Gaussians distributions with parameters θ i = (µ i , Σ i ) ∈ Θ. By the equality DISPLAYFORM0 it suffices to upper bound the term only involving log p 1 (X) (the other follows similarly), which by Theorem D.2 requires bounding the growth of ∇ log p 1 (x) 2 .

We have DISPLAYFORM1 2 for i = 1, 2, therefore by (a trivial variant of) Theorem D.2(c)

we get DISPLAYFORM2 The same bound holds for log p 2 .

Adding them up and substituting the bound appendix B.1 gives that DISPLAYFORM3 As σ : R → R is 1-Lipschitz, by the Rademacher contraction inequality (Ledoux & Talagrand, 2013) , we have DISPLAYFORM4 The right hand side can be bounded directly as DISPLAYFORM5 B.2 PROOF OF THEOREM 3.2 KL bounds Recall the basic property of exponential family that DISPLAYFORM6 By the assumption on ∇ 2 A we have that DISPLAYFORM7 17) Moreover, the exponential family also satisfies that DISPLAYFORM8 where ρ = θ 1 − θ 2 .

Using the assumption we have that γ θ 1 − θ 2 2 ≤ ρ ∇ 2 A(θ 2 + tρ)ρ ≤ β θ 1 − θ 2 2 and therefore DISPLAYFORM9 Combining this with eq. (17) we complete the proof.

Wasserstein bounds We show eq. (8).

As diam(X ) = D, there exists x 0 ∈ X such that x − x 0 ≤ D for all x ∈ X .

Hence for any 1-Lipschitz function f : DISPLAYFORM10 2 /4-sub-Gaussian.

Applying Theorem D.1(a), we get that for any p, q ∈ G, DISPLAYFORM11 Generalization For any θ ∈ Θ we compute the Rademacher complexity DISPLAYFORM12

We consider mixture of k identity-covariance Gaussians on R d : DISPLAYFORM0 We will use a one-hidden-layer neural network that implements (a slight modification of) log p θ : DISPLAYFORM1 Theorem C.1.

The family F is suitable for learning mixture of k Gaussians.

Namely, we have that(1) (Restricted approximability) For any θ 1 , θ 2 ∈ Θ, we have DISPLAYFORM2 (2) (Generalization) We have for some absolute constant C > 0 that DISPLAYFORM3

The Gaussian concentration result (Vershynin, 2010, Proposition 5.34) will be used here and in later proofs, which we provide for convenience.

DISPLAYFORM0

Restricted approximability For the upper bound, it suffices to show that each DISPLAYFORM0 is D-Lipschitz.

Indeed, we have DISPLAYFORM1 This further shows that every discriminator f 1 − f 2 ∈ F is at most 2D-Lipschitz, so by Theorem D.2(a)

we get the upper bound.

We now establish the lower bound.

As F implements the KL divergence, for any two p 1 , p 2 ∈ P, we have DISPLAYFORM2 We consider regularity properties of the distributions p 1 , p 2 in the Bobkov-Gotze sense (Theorem D.1(a)).

Suppose DISPLAYFORM3 Letting X j ∼ N(µ j , I d ) be the mixture components.

By the Gaussian concentration (Lemma C.2), each f (X j ) is 1-sub-Gaussian, so we have for any DISPLAYFORM4

Therefore f (X) is at most (D 2 +1)-sub-Gaussian, and thus X satisfies the Bobkov-Gozlan condition with σ 2 = D 2 + 1.

Applying Theorem D.1(a)

we get DISPLAYFORM0 Generalization Reparametrize the one-hidden-layer neural net eq. FORMULA0 as DISPLAYFORM1 It then suffices to bound the Rademacher complexity of f θ for θ DISPLAYFORM2 and the Rademacher process DISPLAYFORM3 we show that Y θ is suitably Lipschitz in θ (in the ρ metric) and use a one-step discretization bound.

Indeed, we have DISPLAYFORM4 Therefore, for any ε > 0 we have DISPLAYFORM5 for some constant C > 0.We now bound the expected supremum of the max over a covering set.

Let N (Θ, ρ, ε) be a ε-covering set of Θ under ρ, and N (Θ, ρ, ε) be the covering number.

As ρ looks at each µ i , c j separately, its covering number can be upper bounded by the product of each separate covering: DISPLAYFORM6 Now, for each invididual process Y θ is the i.i.d.

average of random variables of the form ε i log k j=1 exp(µ j X+c j ).

The log-sum-exp part is D-Lipschitz in X, so we can reuse the analysis done precedingly (in the Bobkov-Gotze part) to get that log DISPLAYFORM7 This shows that the term ε i log DISPLAYFORM8 1)-subGaussian, and thus we have by sub-Gaussian maxima bounds that DISPLAYFORM9 By the 1-step discretization bound and combining eq. (19) and appendix C.2, we get DISPLAYFORM10 Choosing ε = c/n for sufficiently small c (depending on D 2 , B w ) gives that DISPLAYFORM11 DISPLAYFORM12 Theorem D.2 (Upper bounding f -contrast by Wasserstein).

Let p, q be two distributions on R d with positive densities and denote their probability measures by P, Q. Let f : DISPLAYFORM13 (b) (Truncated W 1 bound) Let D > 0 be any diameter of interest.

Suppose for any p ∈ {p, q} we have DISPLAYFORM14 then we have DISPLAYFORM15 (c) (W 2 bound) Suppose ∇f (x) ≤ c 1 x + c 2 for all x ∈ R d , then we have DISPLAYFORM16 Proof.

(a) This follows from the dual formulation of W 1 .(b) We do a truncation argument.

We have DISPLAYFORM17 Term II has the followng bound by Cauchy-Schwarz: DISPLAYFORM18 We now deal with term I. By definition of the Wasserstein distance, there exists a coupling DISPLAYFORM19 .

On this coupling, we have DISPLAYFORM20 Above, inequality (i) used the Lipschitzness of f in the D-ball, and (ii) used Cauchy-Schwarz.

Putting terms I and II together we get DISPLAYFORM21 (c) This part is a straightforward extension of (Polyanskiy & Wu, 2016, Proposition 1).

For completeness we present the proof here.

For any x, y ∈ R d we have DISPLAYFORM22 By definition of the W 2 distance, there exists a coupling (X, Y ) ∼ π such that X ∼ P , Y ∼ Q, and DISPLAYFORM23 On this coupling, taking expectation of the above bound, we get DISPLAYFORM24 Finally, the triangle inequality gives DISPLAYFORM25 so the left hand side is also bounded by the preceding quantity.

It is straightforward to see that the inverse of x = G θ (z) can be computed as DISPLAYFORM0 θ is also a -layer feedforward net with activation σ −1 .We now consider the problem of representing log p θ (x) by a neural network.

Let φ γ be the density of Z ∼ N(0, diag(γ 2 )).

Recall that the log density has the formula DISPLAYFORM1 First consider the inverse network that implements G −1 θ .

By eq. FORMULA0 , this network has layers ( − 1 hidden layers), d2 + d parameters in each layer, and σ −1 as the activation function.

Now, as log φ γ has the form log φ γ (z) = a(γ) − i z 2 i /(2γ 2 i ), we can add one more layer on top of z with the square activation and the inner product with −γ −2 /2 to get this term.

Second, we show that by adding some branches upon this network, we can also compute the log determinant of the Jacobian.

Define h = W −1 (x − b ) and backward recursively h k−1 = DISPLAYFORM2 Taking the log determinant gives DISPLAYFORM3 As (h , . . .

, h 2 ) are exactly the (pre-activation) hidden layers of the inverse network, we can add one branch from each layer, pass it through the log σ −1 activation, and take the inner product with 1.Finally, by adding up the output of the density branch and the log determinant branch, we get a neural network that computes log p θ (x) with no more than + 1 layers and O( d 2 ) parameters, and choice of activations within {σ DISPLAYFORM4 We state a similar restricted approximability bound here in terms of the W 2 distance, which we also prove.

DISPLAYFORM5 The theorem follows by combining the following three lemmas, which we show in sequel.

Lemma D.3 (Lower bound).

There exists a constant c = c(R W , R b , ) > 0 such that for any θ 1 , θ 2 ∈ Θ, we have DISPLAYFORM6

We show that p θ satisfies the Gozlan condition for any θ ∈ Θ and apply Theorem D.1.

DISPLAYFORM0 .

By definition, we can write DISPLAYFORM1 .

DISPLAYFORM2 the last inequality following from γ i ≤ 1.

Therefore G θ is also L-Lipschitz.

DISPLAYFORM3 Therefore the mapping ( DISPLAYFORM4 Hence by Lemma C.2, the random variable DISPLAYFORM5 is L 2 -sub-Gaussian, and thus the Gozlan condition is satisfied with σ 2 = L 2 .

By definition of the network G θ we have DISPLAYFORM6 Now, for any θ 1 , θ 2 ∈ Θ, we can apply Theorem D.1(b) and get DISPLAYFORM7 and the same holds with p θ1 and p θ2 swapped.

As log p θ1 − log p θ2 ∈ F, by Lemma 4.3, we obtain DISPLAYFORM8 The last bound following from the fact that W 2 ≥ W 1 .

We are going to upper bound W F by the Wasserstein distances through Theorem D.2.

Fix θ 1 , θ 2 ∈ Θ. By definition of F, it suffices to upper bound the Lipschitzness of log p θ (x) for all θ ∈ Θ. Recall that DISPLAYFORM0 where h 1 , . . .

, h (= z) are the hidden-layers of the inverse network z = G −1 θ (x), and C(θ) is a constant that does not depend on x.

We first show the W 2 bound.

Clearly log p θ (x) is differentiable in x. As θ ∈ Θ has norm bounds, each layer h k is C(R W , R b , k)-Lipschitz in x, so term II is altogether DISPLAYFORM1 For term I, note that h is C-Lipschitz in x, so we have DISPLAYFORM2 Putting together the two terms gives DISPLAYFORM3 Further, under either p θ1 or p θ2 (for example p θ1 ), we have DISPLAYFORM4 Therefore we can apply Theorem D.2(c) and get DISPLAYFORM5 We now turn to the W 1 bound.

The bound eq. (22) already implies that for X 2 ≤ D, DISPLAYFORM6 Choosing D = K √ d, for a sufficiently large constant K, by the bound X 2 ≤ C( Z 2 + 1) we have the tail bound P( X 2 ≥ D) ≤ exp(−20d).

On the other hand by the bound | log p θ (x)| ≤ C(( x 2 + 1) 2 /δ 2 + √ d( x 2 + 1)) we get under either p θ1 or p θ2 (for example p θ1 ) we have DISPLAYFORM7 Thus we can substitute DISPLAYFORM8 For any log-density neural network F θ (x) = log p θ (x), reparametrize so that (W i , b i ) represent the weights and the biases of the inverse network z = G −1 θ (x).

By eq. FORMULA0 , this has the form DISPLAYFORM9 Consequently the reparametrized θ = (W i , b i ) i∈[ ] belongs to the (overloading Θ) DISPLAYFORM10 As F = {F θ1 − F θ2 : θ 1 , θ 2 ∈ Θ}, the Rademacher complexity of F is at most two times the quantity DISPLAYFORM11 We do one additional re-parametrization.

Note that the log-density network F θ (x) = log p θ (x) has the form DISPLAYFORM12 (24) The constant C(θ) is the sum of the normalizing constant for Gaussian density (which is the same across all θ, and as we are taking subtractions of two log p θ , we can ignore this) and the sum of log det(W i ), which is upper bounded by d R W .

We can additionally create a parameter K = K(θ) ∈ [0, d R W ] for this term and let θ ← (θ, K).For any (reparametrized) θ, θ ∈ Θ, define the metric DISPLAYFORM13 Then we have, letting Y θ = 1 n n i=1 ε i F θ (X i ) denote the Rademacher process, the one-step discretization bound (Wainwright, 2018, Section 5).

DISPLAYFORM14 We deal with the two terms separately in the following two lemmas.

Lemma D.6 (Discretization error).

There exists a constant C = C(R W , R b , ) such that, for all θ, θ ∈ Θ such that ρ(θ, θ ) ≤ ε, we have DISPLAYFORM15 Lemma D.7 (Expected max over a finite set).

There exists constants λ 0 , C (depending on DISPLAYFORM16 Published as a conference paper at ICLR 2019Substituting the above two Lemmas into the bound eq. (25), we get that for all ε ≤ min {R W , R b } and λ ≤ λ 0 δ 2 n, DISPLAYFORM17 the last bound holding if n ≥ d. Choosing λ = n log n/δ 4 , which will be valid if n/ log n ≥ δ DISPLAYFORM18 This term dominates term I and is hence the order of the generalization error.

DISPLAYFORM19 Fix θ, θ such that ρ(θ, θ ) ≤ ε.

As Y θ is the empirical average over n samples and |ε i | ≤ 1, it suffices to show that for any DISPLAYFORM20 For the inverse network G −1 DISPLAYFORM21 denote the k-th hidden layer: DISPLAYFORM22 Let h k (x) denote the layers of G −1 θ (x) accordingly.

Using this notation, we have DISPLAYFORM23 Lipschitzness of hidden layers We claim that for all k, we have DISPLAYFORM24 and consequently when ρ(θ, θ ) ≤ ε, we have DISPLAYFORM25 (27) We induct on k to show these two bounds.

For eq. FORMULA1 , note that h 0 = x 2 and DISPLAYFORM26 so an induction on k shows the bound.

For eq. (27), note that DISPLAYFORM27 so the base case holds.

Now, suppose the claim holds for the (k − 1)-th layer, then for the k-th layer we have DISPLAYFORM28 ' verifying the result for layer k.

Dealing with (·) 2 and log σ −1For the log σ −1 term, note that |(log σ −1 ) | = |σ −1 /σ −1 | ≤ β σ by assumption.

So we have the Lipschitzness DISPLAYFORM29 For the quadratic term, let DISPLAYFORM30 Putting together Combining the preceding two bounds and that |K − K | ≤ ε, we get DISPLAYFORM31 Tail decay at a single θ Fixing any θ ∈ Θ, we show that the random variable DISPLAYFORM32 is suitably sub-exponential.

To do this, it suffices to look at a single x and then use rules for independent sums.

DISPLAYFORM33 ) is sub-Gaussian, with mean and sub-Gaussianity parameter O(Cd).

Indeed, we have DISPLAYFORM34 Hence the above term is a C √ dLipschitz function of a standard Gaussian, so is Cd-sub-Gaussian by Gaussian concentration C.2.

To bound the mean, use the bound DISPLAYFORM35 As we have − 1 terms of this form, their sum is still Cd-sub-Gaussian with a O(Cd) mean (absorbing into C).Second, the term h , A γ h is a quadratic function of a sub-Gaussian random vector, hence is subexponential.

Its mean is bounded by E[ A γ op h 2 2 ] ≤ Cd/δ 2 .

Its sub-exponential parameter is 1/δ 2 times the sub-Gaussian parameter of h , hence also Cd/δ 2 .

In particular, there exists a constant DISPLAYFORM36 (See for example (Vershynin, 2010) for such results.)

Also, the parameter K is upper bounded by DISPLAYFORM37 Putting together, multiplying by ε i (which addes up the squared mean onto the sub-Gaussianity / sub-exponentiality and multiplies it by at most a constant) and summing over n, we get that Y θ is mean-zero sub-exponential with the MGF bound DISPLAYFORM38 Bounding the expected maximum We use the standard covering argument to bound the expected DISPLAYFORM39 Hence, the covering number of Θ is bounded by the product of independent covering numbers, which further by the volume argument is DISPLAYFORM40 Using Jensen's inequality and applying the bound appendix D.6.2, we get that for any λ ≤ λ 0 δ 2 n, DISPLAYFORM41 Towards stating the theorem more quantitatively, we will need to specify a few quantities of the generator class that will be relevant for us.

First, for notational simplicity, we override the definition of p β θ by a truncated version of the convolution of p and a Gaussian distribution.

Concretely, let D z = {z : z ≤ √ d log 2 d, z ∈ R k } be a truncated region in the latent space (which contains an overwhelming large part of the probability mass), and the let DISPLAYFORM42 Then, let p β θ (x) be the distribution obtained by adding Gaussian noise with variance β 2 to a sample from G θ , and truncates the distribution to a very high-probability region (both in the latent variable and observable domain.)

Formally, let p β θ be a distribution over R d , s.t.

DISPLAYFORM43 For notational convenience, denote by f : DISPLAYFORM44 , and denote by z * a maximum of f .

Furthermore, whenever clear from the context, we will drop θ from p θ and G θ .We introduce several regularity conditions for the family of generators G: Assumption E.1.

We assume the following bounds on the partial derivatives of f : we denote S := max z∈Dz: z−z * ≤δ ∇ 2 ( G θ (z)−x 2 ) , and λ min := max z∈Dz: DISPLAYFORM45 Similarly, we denote t(z) := k 3 max |I|=3 DISPLAYFORM46 (z).

and T = max z:z∈Dz |t(z)|.

We will denote by R an upper bound on the quantity DISPLAYFORM47 .

Finally, we assume the inverse activation function is Lipschitz, namely |σ DISPLAYFORM48 Note on asymptotic notation: For notational convenience, in this section, , , as well as the Big-Oh notation will hide dependencies on R, L G , S, T (in the theorem statements we intentionally emphasize the polynomial dependencies on d.) The main theorem states that for certain F,d F approximates the Wasserstein distance.

Theorem E.1.

Suppose the generator class G satisfies the assumption E.1 and let F be the family of functions as defined in Theorem E.2.

Then, we have that for every p, q ∈ G, DISPLAYFORM49 Furthermore, when n poly(d) we have R n (F, G) poly FORMULA59 log n n .

Here hides dependencies on R, L G , S, and T .The main ingredient in the proof will be the theorem that shows that there exists a parameterized family F that can approximate the log density of p β for every p ∈ G.Theorem E.2.

Let G satisfy the assumptions in Assumption E.1.

For β = O(poly(1/d)), there exists a family of neural networks F of size poly( 1 β , d) such that for every distribution p ∈ G, there exists N ∈ F satisfying:(1) N approximates log p for typical x: given input x = G(z * ) + r, for r ≤ 10β √ d log d, and DISPLAYFORM50 (2) N is globally an approximate lower bound of p: on any input x, N outputs N (x) ≤ log p DISPLAYFORM51 (3) N approximates the entropy in the sense that: the output DISPLAYFORM52 The approach will be as follows: we will approximate p β (x) essentially by a variant of Laplace's method of integration, using the fact that DISPLAYFORM53 for a normalization constant C that can be calculated up to an exponentially small additive factor.

When x is typical (in case (1) of Theorem E.2), the integral will mostly be dominated by it's maximum value, which we will approximately calculate using a greedy "inversion" procedure.

When x is a atypical, it turns out that the same procedure will give a lower bound as in (2).We are ready to prove Theorem E.1, assuming the correctness of Theorem E.2:Proof of Theorem E.1.

By Theorem E.2, we have that there exist neural networks N 1 , N 2 ∈ F that approximate log p β and log q β respectively in the sense of bullet FORMULA0 - FORMULA2 in Theorem E.2.

Thus we have that by bullet (2) for distribution q β , and bullet (3) for distribution q β , we have DISPLAYFORM54 Similarly, we have DISPLAYFORM55 Combining the equations above, setting f = N 1 (x) − N 2 (x), we obtain that DISPLAYFORM56 Therefore, by definition, and Bobkov-Götze theorem ( DISPLAYFORM57 Thus we prove the lower bound.

Proceeding to the upper bound, notice that DISPLAYFORM58 Having this, we'd be done: namely, we simply set β = W 1/6 to get the necessary bound.

Proceeding to the claim, consider the optimal coupling C of p, q, and consider the induced coupling C z on the latent variable z in p, q. Then, DISPLAYFORM59 Consider the couplingC z on the latent variables of p β , q DISPLAYFORM60 2 .

The couplingC of p β , q β specified by coupling z's according toC z and the (truncated) Gaussian noise to be the same in p β , q β , we have that DISPLAYFORM61 The generalization claim follows completely analogously to Lemma D.5, using the Lipschitzness bound of the generators in Theorem E.2.The rest of the section is dedicated to the proof of Theorem E.2, which will be finally in Section E.3.

First, we prove several helper lemmas: DISPLAYFORM0 Proof.

The proof proceeds by reverse induction on l. We will prove that DISPLAYFORM1 Published as a conference paper at ICLR 2019 The claim trivial holds for i = 0, so we proceed to the induction.

Suppose the claim holds for i. Then, DISPLAYFORM2 which recovers aẑ, s. DISPLAYFORM3 Proof.

N will iteratively produce estimatesĥ i , s.t.

DISPLAYFORM4 We will prove by induction that DISPLAYFORM5 .

The claim trivial holds for i = 0, so we proceed to the induction.

Suppose the claim holds for i. Then, DISPLAYFORM6 where the last inequality holds by the inductive hypothesis, and the next-to-last one due to Lipschitzness of σ −1 .

DISPLAYFORM7 , we have DISPLAYFORM8 This implies that DISPLAYFORM9 which in turns means DISPLAYFORM10 which completes the claim.

Turning to the size/Lipschitz constant of the neural network: all we need to notice is thatĥ i = σ −1 (W The integral on the right is nothing more than the (unnormalized) cdf of a Gaussian with covariance DISPLAYFORM11 is positive definite with smallest eigenvalue bounded by DISPLAYFORM12 where G i is the i-th coordinate of G. We claim DISPLAYFORM13 The latter follows from the bound on r and Cauchy-Schwartz.

For the former, note that we have DISPLAYFORM14 up to a multiplicative factor of 1 ± O poly(d) (β log(1/β)), since the normalizing factor satisfies DISPLAYFORM15 We will first present the algorithm, then prove that it:(1) Approximates the integral as needed.(2) Can be implemented by a small, Lipschitz network as needed.

The algorithm is as follows:Algorithm 1 Discriminator family with restricted approximability for degenerate manifold 1: Parameters: DISPLAYFORM16 and let S be the trivial β 2 -net of the matrices with spectral norm bounded by O(1/β 2 ).

3: Letẑ = N inv (x) be the output of the "invertor" circuit of Lemma E.4.

4: Calculate g = ∇f (ẑ), H = ∇ 2 f (ẑ) by the circuit implied in Lemma E.7.

5: Let M be the nearest matrix in S to H and E i , i ∈ [r] be s.t.

M + E i has Ω(β)-separated eigenvalues. (If there are multiple E i that satisfy the separation condition, pick the smallest i.) 6: Let (e i , λ i ) be approximate eigenvector/eigenvalue pairs of H + E i calculated by the circuit implied in Lemma E.6.

First, we will show (1), namely that the Algorithm 1 approximates the integral of interest.

We'll use an approximate version of Lemma E.8 -with a slightly different division of where the "bulk" of the integral is located.

As in Algorithm 1, letẑ = N inv (x) be the output of the "invertor" circuit of Lemma E.4.

and let δ = 100d L G β log(1/β) R 2 and denote by B the set B = {z : | z −ẑ, e i | ≤ δ}.

Furthermore, let's define how the matrices E i , i ∈ [r] are to be chosen.

Let S be an β 2 -net of the matrices with spectral norm bounded by O(1/β 2 ).

We claim that there exist matrices E 1 , E 2 , . . .

, E r , r = Ω(d log(1/β)), s.t.

if M ∈ S, at least one of the matrices M + E i , i ∈ [r] has eigenvalues that are Ω(β)-separated and E i 2 ≤ up to a multiplicative factor of 1 − O(β log(1/β)).

However, if we consider the proof of Theorem E.9, we notice that the approximation consider there indeed serves our purpose: Taylorexpanding same as there, we have DISPLAYFORM17 This integral can be evaluated in the same manner as in Theorem E.9, as our bound on T β holds universally on neighborhood of radius D x .Finally, part (3) follows easily from FORMULA0 and FORMULA1 : DISPLAYFORM18

We further perform synthetic WGAN experiments with invertible neural net generators (cf.

Section 4.1) and discriminators designed with restricted approximability (Lemma 4.1).

In this case, the invertibility guarantees that the KL divergence can be computed, and our goal is to demonstrate that the empirical IPM W F (p, q) is well correlated with the KL-divergence between p and q on synthetic data for various pairs of p and q (The true distribution p is generated randomly from a ground-truth neural net, and the distribution q is learned using various algorithms or perturbed version of p.)

Data The data is generated from a ground-truth invertible neural net generator (cf.

Section 4.1), i.e. X = G θ (Z), where DISPLAYFORM0 is a -layer layer-wise invertible feedforward net, and Z is a spherical Gaussian.

We use the Leaky ReLU with negative slope 0.5 as the activation function σ, whose derivative and inverse can be very efficiently computed.

The weight matrices of the layers are set to be well-conditioned with singular values in between 0.5 to 2.We choose the discriminator architecture according to the design with restricted approximability guarantee (Lemma 4.1, eq. (10) eq. (11)).

As log σ −1 is a piecewise constant function that is not differentiable, we instead model it as a trainable one-hidden-layer neural network that maps reals to reals.

We add constraints on all the parameters in accordance with Assumption 1.Training To train the generator and discriminator networks, we generate stochastic batches (with batch size 64) from both the ground-truth generator and the trained generator, and solve the min-max problem in the Wasserstein GAN formulation.

We perform 10 updates of the discriminator in between each generator step, with various regularization methods for discriminator training (specified later).

We use the RMSProp optimizer (Tieleman & Hinton, 2012) as our update rule.

Evaluation metric We evaluate the following metrics between the true and learned generator.(1) The KL divergence.

As the density of our invertible neural net generator can be analytically computed, we can compute their KL divergence from empirical averages of the difference of the log densities: DISPLAYFORM1 where p and p are the densities of the true generator and the learned generator.

We regard the KL divergence as the "correct" and rather strong criterion for distributional closeness.(2) The training loss (IPM W F train).

This is the (unregularized) GAN loss during training.

Note: as typically in the training of GANs, we balance carefully the number of steps for discriminator and generators, the training IPM is potentially very far away from the true W F (which requires sufficient training of the discriminators).(3) The neural net IPM (W F eval).

We report once in a while a separately optimized WGAN loss in which the learned generator is held fixed and the discriminator is trained from scratch to optimality.

Unlike the training loss, here the discriminator is trained in norm balls but with no other regularization.

By doing this, we are finding f ∈ F that maximizes the contrast and we regard the f found by stochastic optimization an approximate maximizer, and the loss obtained an approximation of W F .Our theory shows that for our choice of G and F, WGAN is able to learn the true generator in KL divergence, and the F-IPM (in evaluation instead of training) should be indicative of the KL divergence.

We test this hypothesis in the following experiments.

In our first experiment, G is a two-layer net in d = 10 dimensions.

Though the generator is only a shallow neural net, the presence of the nonlinearity makes the estimation problem non-trivial.

We train a discriminator with the architecture specified in Lemma 4.1), using either Vanilla WGAN (clamping the weight into norm balls) or WGAN-GP (Gulrajani et al., 2017) (adding a gradient penalty).

We fix the same ground-truth generator and run each method from 6 different random initializations.

Results are plotted in FIG5 .Our main findings are two-fold:(1) WGAN training with discriminator design of restricted approximability is able to learn the true distribution in KL divergence.

Indeed, the KL divergence starts at around 10 -30 and the best run gets to KL lower than 1.

As KL is a rather strong metric between distributions, this is strong evidence that GANs are finding the true distribution and mode collapse is not happening.(2) The W F (eval) and the KL divergence are highly correlated with each other, both along each training run and across different runs.

In particular, adding gradient penalty improves the optimization significantly (which we see in the KL curve), and this improvement is also reflected by the W F curve.

Therefore the quantity W F can serve as a good metric for monitoring convergence and is at least much better than the training loss curve.

To test the necessity of the specific form of the discriminator we designed, we re-do the same experiment with vanilla fully-connected discriminator nets.

Results (in Appendix G.4) show that IPM with vanilla discriminators also correlate well with the KL-divergence.

This is not surprising from The left-most figure shows the KL-divergence between the true distribution p and learned distribution q at different steps of training, the middle the estimated IPM (evaluation) between p and q, and the right one the training loss.

We see that the estimated IPM in evaluation correlates well with the KL-divergence.

Moving average is applied to all curves.a theoretical point of view because a standard fully-connected discriminator net (with some overparameterization) is likely to be able to approximate the log density of the generator distributions (which is essentially the only requirement of Lemma 4.3.)For this synthetic case, we can see that the inferior performance in KL of the WGAN-Vanilla algorithm doesn't come from the statistical properties of GANs, but rather the inferior training performance in terms of the convergence of the IPM.

We conjecture similar phenomenon occurs in training GANs with real-life data as well.

In this section, we remove the effect of the optimization and directly test the correlation between p and its perturbations.

We compare the KL divergence and neural net IPM on pairs of perturbed generators.

In each instance, we generate a pair of generators (G, G ) (with the same architecture as above), where G is a perturbation of G by adding small Gaussian noise.

We compute the KL divergence and the neural net IPM between G and G .

To denoise the unstable training process for computing the neural net IPM, we optimize the discriminator from 5 random initializations and pick the largest value as the output.

As is shown in FIG6 , there is a clear positive correlation between the (symmetric) KL divergence and the neural net IPM.

In particular, majority of the points fall around the line W F = 100D kl , which is consistent with our theory that the neural net distance scales linearly in the KL divergence.

Note that there are a few outliers with large KL.

This happens mostly due to the perturbation being accidentally too large so that the weight matrices become poorly conditioned -in the context of our theory, they fall out of the good constraint set as defined in Assumption 1.

We re-do the experiments of Section G.2 with vanilla fully-connected discriminator nets.

We use a three-layer net with hidden dimensions 50-10, which has more parameters than the architecture with restricted approximability.

Results are plotted in FIG7 .

We find that the generators also converge well in the KL divergence, but the correlation is slightly weaker than the setting with restricted approximability (correlation still presents along each training run but weaker across different runs).

This suggests that vanilla discriminator structures might be practically quite satisfying for getting a good generator, though specific designs may help improve the quality of the distance W F .

The left-most figure shows the KLdivergence between the true distribution p and learned distribution q at different steps of training, the middle the estimated IPM (evaluation) between p and q, and the right one the training loss.

We see that the estimated IPM in evaluation correlates well with the KL-divergence.

Moving average is applied to all curves.

Correlation between KL and neural net IPM is computed with vanilla fully-connected discriminators and plotted in FIG8 .

The correlation (0.7489) is roughly the same as for discriminators with restricted approximability (0.7315).

@highlight

GANs can in principle learn distributions sample-efficiently, if the discriminator class is compact and has strong distinguishing power against the particular generator class.

@highlight

Proposes the notion of restricted approximability, and provides a sample complexity bound, polynomial in the dimension, which is useful in investigating lack of diversity in GANs.

@highlight

Analyzes that the Integral Probability Metric can be a good approximation of Wasserstein distance under some mild assumptions.
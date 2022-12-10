The importance weighted autoencoder (IWAE) (Burda et al., 2016) is a popular variational-inference method which achieves a tighter evidence bound (and hence a lower bias) than standard variational autoencoders by optimising a multi-sample objective, i.e. an objective that is expressible as an integral over $K > 1$ Monte Carlo samples.

Unfortunately, IWAE crucially relies on the availability of reparametrisations and even if these exist, the multi-sample objective leads to inference-network gradients which break down as $K$ is increased (Rainforth et al., 2018).

This breakdown can only be circumvented by removing high-variance score-function terms, either by heuristically ignoring them (which yields the 'sticking-the-landing' IWAE (IWAE-STL) gradient from Roeder et al. (2017)) or through an identity from Tucker et al. (2019) (which yields the 'doubly-reparametrised' IWAE (IWAE-DREG) gradient).

In this work, we argue that directly optimising the proposal distribution in importance sampling as in the reweighted wake-sleep (RWS) algorithm from Bornschein & Bengio (2015) is preferable to optimising IWAE-type multi-sample objectives.

To formalise this argument, we introduce an adaptive-importance sampling framework termed adaptive importance sampling for learning (AISLE) which slightly generalises the RWS algorithm.

We then show that AISLE admits IWAE-STL and IWAE-DREG (i.e. the IWAE-gradients which avoid breakdown) as special cases.

Let x be some observation and let z be some latent variable taking values in some space Z. These are modeled via the generative model p θ (z, x) = p θ (z)p θ (x|z) which gives rise to the marginal likelihood p θ (x) = Z p θ (z, x) dz of the model parameters θ.

In this work, we analyse algorithms for variational inference, i.e. algorithms which aim to 1. learn the generative model, i.e. find a value θ which is approximately equal to the maximum-likelihood estimate (MLE) θ ml := arg max θ p θ (x); 2. construct a tractable variational approximation q φ,x (z) of p θ (z|x) = p θ (z, x)/p θ (x), i.e. find the value φ such that q φ ,x (z) is as close as possible to p θ (z|x) in some suitable sense.

A few comments about this setting are in order.

Firstly, as is common in the literature, we restrict our presentation to a single latent representation-observation pair (z, x) to avoid notational clutter -the extension to multiple independent observations is straightforward.

Secondly, we assume that no parameters are shared between the generative model p θ (z, x) and the variational approximation q φ,x (z).

This is common in neural-network applications but could be relaxed.

Thirdly, our setting is general enough to cover amortised inference.

For this reason, we often refer to φ as the parameters of an inference network.

Two main classes of stochastic gradient-ascent algorithms for optimising ψ := (θ, φ) which employ K ≥ 1 Monte Carlo samples ('particles') to reduce errors have been proposed. (Roeder et al., 2017) heuristically drops the problematic score-function terms from the IWAE φ-gradient.

This induces bias for the IWAE objective.

-IWAE-DREG.

The 'doubly-reparametrised' IWAE (IWAE-DREG) φ-gradient (Tucker et al., 2019) unbiasedly removes the problematic score-function terms from the IWAE φ-gradient using a formal identity.

• RWS.

The reweighted wake-sleep (RWS) algorithm (Bornschein & Bengio, 2015) optimises two separate objectives for θ and φ.

Its gradients are approximated by self-normalised importance sampling with K particles: this induces a bias which vanishes as K → ∞. RWS can be viewed as an adaptive importance-sampling approach which iteratively improves its proposal distribution while simultaneously optimising θ via stochastic approximation.

Crucially, the RWS φ-gradients do not degenerate as K → ∞.

Of these two methods, the IWAE is the most popular and Tucker et al. (2019) demonstrated empirically that RWS can break down, conjecturing that this is due to the fact that RWS does not optimise a joint objective (for θ and φ).

Meanwhile, the IWAE-STL gradient performed consistently well despite lacking a firm theoretical footing.

Yet, IWAE suffers from the above-mentioned φ-gradient breakdown and exhibited inferior empirical performance to RWS (Le et al., 2019) .

Thus, it is not clear whether the multi-sample objective approach of IWAE or the adaptive importance-sampling approach of RWS is preferable.

In this work, we show that directly optimising the proposal distribution, e.g. as done by RWS, is preferable to optimising the IWAE multi-sample objective because (a) the multi-sample objective typically relies on reparametrisations and, even if these are available, leads to the φ-gradient breakdown, (b) modifications of the IWAE φ-gradient which avoid this breakdown (i.e. IWAE-STL and IWAE-DREG) can be justified in a more principled manner by taking an RWS-type adaptive importance-sampling view.

This conclusion was already reached by Le et al. (2019) based on numerical experiments.

They demonstrated that the need for reparametrisations can make IWAE inferrior to RWS e.g. for discrete latent variables.

Our work complements theirs by formalising this argument.

To this end, we slightly generalise the RWS algorithm to obtain a generic adaptive importance-sampling framework for variational inference which we term adaptive importance sampling for learning (AISLE) for ease of reference.

We then show that AISLE admits not only RWS but also the IWAE-DREG and IWAE-STL gradients as special cases.

Novel material is presented in Section 3, where we introduce the AISLEframework.

From this, most of the previously proposed gradient estimators can be naturally derived in a principled manner.

Importantly, the derived gradient estimators are guaranteed to not degenerate as K → ∞. Specifically, we establish the following connections.

• We prove that the IWAE-STL gradient can be recovered as a special case of AISLE via a principled and novel application of the 'double-reparametrisation' identity from Tucker et al. (2019) .

This indicates that the breakdown of RWS observed in Tucker et al. (2019) may not be due to its lack of a joint objective as previously conjectured (since IWAE-STL avoided this breakdown despite having the same idealised objective as RWS).

Our work also provides a theoretical foundation for IWAE-STL which was hitherto only heuristically justified as a biased IWAE-gradient.

• We prove that AISLE also admits the IWAE-DREG gradient as a special case.

Our derivation also makes it clear that the learning rate should be scaled as O(K) for the IWAE φ-gradient (and its modified version IWAE-DREG) unless the gradients are normalised as implicitly done by popular optimisers such as ADAM (Kingma & Ba, 2015) .

In contrast, the learning rate for AISLE need not be scaled up with of K.

• When applied to the family of α-divergences, AISLE leads to a new family of gradient estimators that generalises some previously derived in the literature.

• In the supplementary materials, we provide insights into the impact of the selfnormalisation bias on some of the importance-sampling based gradient approxima-tions (Appendix A) and empirically compare the main algorithms discussed in this work (Appendix B).

We stress that the focus of our work is not necessarily to derive new algorithms nor to establish which of the various special cases of AISLE is preferable.

Indeed, while we compare all algorithms discussed in this work empirically on Gaussian models in the supplementary materials, we refer the reader to Tucker et al. (2019) ; Le et al. (2019) for an extensive empirical comparisons of all the algorithms discussed in this work.

Notation.

We repeatedly employ the shorthand p(f ) :

To keep the notation concise, we hereafter suppress dependence on the observation x, i.e. we write q φ (z) := q φ,x (z) as well as

where γ θ (z) := p θ (z, x) and where

2 Background

The expectation q φ (f ) of a test function f : Z → R can be unbiasedly estimated by the

φ , which are independent and identically distributed (IID) according to q φ .

Similarly, expectations of the type π θ (f ) can be approximated by the self-normalised importance sampling estimatê

The notation φ, z stresses the dependence of this estimator on φ and z. The quantity w ψ (z k ) are called the kth importance weight and s w k ψ is its self-normalised version.

For readability, we have dropped the dependence of s w k ψ on z ∈ Z K from the notation.

Remark 2.

The self-normalised estimateπ θ φ, z (f ) is typically not unbiased.

Under mild assumptions (e.g. if sup w ψ < ∞), its bias vanishes at rate O(K −1 ), its standard deviation vanishes at Monte-Carlo rate

Objective.

The importance weighted autoencoder (IWAE), introduced by Burda et al. (2016) , seeks to find a value θ of the generative-model parameters θ which maximises a lower bound L K ψ on the log-marginal likelihood ('evidence').

This bound depends on the inference-network parameters φ and the number of samples, K ≥ 1:

where the expectation is w.r.t.

z ∼ q ⊗K φ .

For any finite K, optimisation of the inferencenetwork parameters φ tightens the evidence bound.

Burda et al. (2016) prove that for any φ we have that L K ψ ↑ log Z θ as K → ∞. If K = 1, the IWAE reduces to the variational autoencoder (VAE) from Kingma & Welling (2014) .

However, for K > 1, as pointed out in Cremer et al. (2017) ; Domke & Sheldon (2018) , the IWAE also constitutes another VAE on an extended space based on an auxiliary-variable construction developed in Andrieu & Roberts (2009) ; Andrieu et al. (2010) ; Lee (2011 ) (see, e.g. Finke, 2015 , for a review).

The gradient of the IWAE objective from (1):

The intractable quantity E G ψ (z) can be approximated unbiasedly via a vanilla Monte Carlo approach using a single (

Unfortunately, this approximation typically has such a large variance that it becomes impracticably noisy (Paisley et al., 2012) .

To remove this high-variance term, the well known reparametrisation trick (Kingma & Welling, 2014 ) is usually employed.

It requires the following assumption.

(R1) There exists a distribution q on some space E and a family of differentiable mappings

, the gradient can be expressed as

Here, the notation ψ indicates that one does not differentiate w ψ w.r.t.

ψ.

The IWAE then uses a vanilla Monte Carlo estimate of (2),

Before proceeding, we state the following lemma, proved in Tucker et al. (2019, Section 8.1), which generalises of the well-known identity q φ (∇ φ log q φ ) = 0.

Lemma 1 (Tucker et al. (2019)).

Under R1, for suitably integrable f ψ : Z → R, we have

We now exclusively focus on the φ-portion of the IWAE gradient, ∇ iwae φ θ, z .

Remark 3 (drawbacks of the IWAE φ-gradient).

The gradient ∇ iwae φ θ, z has three drawbacks.

The last two of these are attributable to the 'score-function' terms ∇ φ log q φ (z) in the φ-gradient portion of (3).

• Reliance on reparametrisations.

A reparametrisation à la R1 is necessary to remove the high-variance term G ψ (z).

For, e.g. discrete, models that violate R1, control-variate approaches (Mnih & Rezende, 2016) or continuous relaxations have been proposed but these incur additional implementation, tuning and computation costs whilst not necessarily reducing the variance (Le et al., 2019).

• Vanishing signal-to-noise ratio.

The φ-gradient breaks down in the sense that its signal-to-noise ratio vanishes as Rainforth et al., 2018) .

This is because ∇ iwae φ θ, z constitutes a self-normalised importance-sampling approximation of π θ ( ψ − ∇ φ log q φ ) = 0, an identity which directly follows from Lemma 1 with f ψ = w ψ .

• Inability to achieve zero variance.

As pointed out in Roeder et al. (2017) , Two modifications of ∇ iwae φ θ, z have been proposed which (under R1) avoid the scorefunction terms in (3) and hence (a) exhibit a stable signal-to-noise ratio as K → ∞ and (b) can achieve zero variance if q φ = π θ (because then ψ ≡ 0 since w ψ is constant).

• IWAE-STL.

The 'sticking-the-landing' IWAE (IWAE-STL) gradient proposed by Roeder et al. (2017) heuristically ignores the score function terms,

As shown in Tucker et al. (2019)), this introduces an additional bias whenever K > 1.

• IWAE-DREG.

The 'doubly-reparametrised' IWAE (IWAE-DREG) gradient proposed by Tucker et al. (2019) removes the score-function terms through Lemma 1,

The quantities ∇ iwae-dreg φ θ, z and ∇ iwae φ φ, z are equal in expectation.

The reweighted wake-sleep (RWS) algorithm was proposed in Bornschein & Bengio (2015) .

The θ-and φ-gradients read

These quantities are usually intractable and therefore approximated by replacing π θ by the self-normalised importance sampling approximationπ θ φ, z (this does not require R1):

Since (7) relies on self-normalised importance sampling, Remark 2 shows that its bias relative to (6) is of order O(1/K).

Appendix A discusses the impact of this bias on the φ-gradient in more detail.

The optimisation of both θ and φ is carried out simultaneously, allowing both gradients to share the same particles and weights.

Nonetheless, the lack of a joint objective (for both θ and φ) is often viewed as the main drawback of RWS.

rws φ θ, z in expectation and is derived by applying Lemma 1 to the latter.

It reads

where the function F(w) := w(1 − w) is used to transform the self-normalised importance weights s w k ψ .

In high-dimensional settings, it is typically the case that the ordered selfnormalised importance weights s w

are then mainly supported on the two particles with the largest self-normalised weights.

and φ simultaneously is that (a) Monte Carlo samples used to approximate the θ-gradient can be re-used to approximate the φ-gradient and (b) optimising φ typically reduces the error (both in terms of bias and variance) of the θ-gradient approximation.

However, adapting the proposal distribution q φ in importance-sampling schemes need not necessarily be based on minimising the (inclusive) KL-divergence.

Numerous other techniques exist in the literature (e.g. Geweke, 1989; Evans, 1991; Oh & Berger, 1992; Richard & Zhang, 2007; Cornebise et al., 2008) and may sometimes be preferable.

Indeed, another popular approach with strong theoretical support is based on minimising the χ 2 -divergence (see, e.g., Deniz Akyildiz & Míguez, 2019).

Based on this insight, we slightly generalise the RWS-objective as θ := arg max θ log Z θ , φ := arg min φ Dƒ(πθ q φ ).

Here, Dƒ(p q) := Z ƒ(p(z)/q(z))q(z) dz is some ƒ-divergence from p to q. We reiterate that alternative approaches for optimising φ (which do not minimise ƒ-divergences) could be used.

However, we state (9) for concreteness as it suffices for the remainder of this work; we call the resulting algorithm adaptive importance sampling for learning (AISLE).

As will become clear below, this unified framework permits a straightforward and principled derivation of robust φ-gradient estimators that do not degenerate as K → ∞.

Optimisation is again performed via a stochastic gradient-ascent.

The intractable θ-gradient

The θ-gradient is thus the same for all algorithms discussed in this work although the IWAEparadigm views it as an unbiased gradient of a (biased) lower-bound to the evidence, while AISLE (and RWS) interpret it as a self-normalised importance-sampling (and consequently biased) approximation of the gradient ∇ θ log Z θ for the 'exact' objective.

In the derivations to follow, integrals of the form π θ ([F • w ψ ]∇ φ log q φ ) naturally appear.

These can also be expressed as Z

Approximating the expectation as well as the normalising constant Z θ on the r.h.s.

with the vanilla Monte Carlo method with

Remark 2 shows that this approximation has a bias of order O(K −1 ) and a standarddeviation of order O(K −1/2 ).

Now, most of the ƒ-divergences used for variational inference in intractable models are such that there exists a functionf :

for an exponent κ ∈ R and constant C(θ) independent of φ.

In other words, for a given value of θ, the optimization of the ƒ-divergence as a function of φ can be carried out without relying on the knowledge of Z θ .

Writing g(y) :=f (y) −f (y)/y, simple algebra then directly shows that

Since the integral in (11) is an expectation with respect to π θ , it can be approximated with selfimportance sampling, possibly multiplied an additional importance-sampling approximation Z θ φ, z of Z θ raised to some power.

This leads to,

Indeed, Equation (10) applies to (11), leading to the reparametrised estimator

where h(y) = g(y)y and g : R → R given immediately above (11).

We now describe several particular cases.

We have KL(

In that case, with the notations of Section 3.3.1, we have g(y) = 1 and h (y) = 1.

• AISLE-KL-NOREP/RWS.

Without relying on any reparametrisation, Equation (12) yields the following gradient, which clearly equals ∇ rws φ θ, z :

• AISLE-KL.

Using reparametrisation, Equation (13) yields the gradient:

We thus arrive at the following result which demonstrates that IWAE-STL can be derived in a principled manner from AISLE, i.e. without the need for a multi-sample objective.

θ, z .

Proposition 1 is notable because it shows that IWAE-STL (which avoids the breakdown highlighted in Rainforth et al. (2018) and which can also achieve zero variance) can be derived in a principled manner from AISLE, i.e. without relying on a multi-sample objective.

Proposition 1 thus provides a theoretical basis for IWAE-STL which was previously viewed as an alternative gradient for IWAE for which it is biased and only heuristically justified.

Furthermore, the fact that IWAE-STL exhibited good empirical performance in Tucker et al. (2019) even in an example in which RWS broke down, suggests that this breakdown may not be due to RWS' lack of optimising a joint objective as previously conjectured. (8) by first replacing the exact (but intractable) φ-gradient by the self-normalised importance-sampling approximation ∇ rws φ θ, z and then applying the identity from Lemma 1.

Note that this may result in a variance reduction but does not change the bias of the gradient estimator.

In contrast, AISLE-KL is derived by first applying Lemma 1 to the exact (RWS) φ-gradient and then approximating the resulting expression.

This can potentially reduce both bias and variance.

Up to some irrelevant additive constant, the α-divergence between two distributions p and q is given by Z (p(z)/q(z)) α q(z) dz for some α > 1.

This can also be expressed as Z κ θ Zf (w ψ (z))q φ (z) dz with κ = −α andf (y) = y α .

In this case, with the notation from Section 3.3.1, we have g(y) = (α − 1)y α−1 and h (y) = α(α − 1) y α−1 .

Note that the case α = 2 is equivalent, up to an irrelevant additive constant, to a standard χ 2 -divergence.

Minimising this divergence is natural in importance sampling since χ 2 (π θ q φ ) = var z∼q φ [w ψ /Z θ ] is the variance of the importance weights.

• AISLE-α-NOREP.

Without relying on any reparametrisation, Equation (13) yields

with the following special case which is also proportional to the 'score gradient' from Dieng et al. (2017, Appendix G):

• AISLE-α.

Using reparametrisation, Equation (12) becomes

again with the special case ∇

This demonstrates that IWAE-DREG can be derived (up to the proportionality factor 2K) in a principled manner from AISLE, i.e. without the need for a multi-sample objective.

θ, z .

Note that if the implementation normalises the gradients, e.g. as effectively done by ADAM (Kingma & Ba, 2015) , the constant factor cancels out and AISLE-χ 2 becomes equivalent to IWAE-DREG.

Otherwise (e.g. in plain stochastic gradient-ascent) this shows that the learning rate needs to be scaled as O(K) for the IWAE or IWAE-DREG φ-gradients.

For the 'exclusive' KL-divergence, we have KL(q φ π θ ) = f (w ψ (z))q φ (z) dz + C(θ) with f (y) = log(y).

In that case, with the notation from Section 3.3.1, we have h (y) = 1/y.

This directly leads to the following approximation,

This can be recognised as a simple average over K independent replicates of the 'stickingthe-landing' estimator for VAEs proposed in Roeder et al. (2017, Equation 8 ).

As we discuss in Appendix A, optimising this 'exclusive' KL-divergence can sometimes lead to faster convergence of φ than optimising the 'inclusive' KL-divergence KL(π θ q φ ).

However, care must be taken because minimising the exclusive divergence does not necessarily lead to well behaved or even well-defined importance weights and thus can negatively affect learning of θ (whose gradient is a self-normalised importance-sampling approximation which makes use of those weights).

We have shown that the adaptive-importance sampling paradigm of the reweighted wake-sleep (RWS) (Bornschein & Bengio, 2015) is preferable to the multi-sample objective paradigm of importance weighted autoencoders (IWAEs) (Burda et al., 2016) because the former achieves all the goals of the latter whilst avoiding its drawbacks.

A On the rôle of the self-normalisation bias within RWS/AISLE

Within the self-normalised importance-sampling approximation, the number of particles, K, interpolates between two extremes:

• As K ↑ ∞,π θ φ, z (f ) becomes an increasingly accurate approximation of π θ (f ).

• For K = 1, however,π θ φ, z (f ) = f (z 1 ) reduces to a vanilla Monte Carlo approximation of q φ (f ) (because the single self-normalised importance weight is always equal to 1).

This leads to the following insight about the estimators ∇ aisle-kl φ θ, z and ∇ aisle-χ 2 φ θ, z .

• As K ↑ ∞, these two estimators become increasingly accurate approxi-

, respectively.

• For K = 1, however, these two estimators reduce to vanilla Monte Carlo ap-

This is similar to the standard IWAE φ-gradient which also represents a vanilla Monte Carlo approximation of −∇ φ KL(q φ π θ ) if K = 1 as IWAE reduces to a VAE in this case.

Characterising the small-K self-normalisation bias of the reparametrisation-free AISLE φ gradients, AISLE-KL-NOREP and AISLE-χ 2 -NOREP, is more difficult because if K = 1, they constitute vanilla Monte Carlo approximations of q φ (∇ φ log q φ ) = 0.

Nonetheless, Le et al. (2019, Figure 5 ) lends some support to the hypothesis that the small-K self-normalisation bias of these gradients also favours a minimisation of the exclusive KL-divergence.

Recall that the main motivation for use of IWAEs (instead of VAEs) was the idea that we could use self-normalised importance-sampling approximations with K > 1 particles to reduce the bias of the θ-gradient relative to ∇ θ log Z θ .

The error of such (self-normalised) importance-sampling approximations can be controlled by ensuring that q φ is close to π θ (in some suitable sense) in any part of the space Z in which π θ has positive probability mass.

For instance, it is well known that the error will be small if the 'inclusive' KL-divergence KL(π θ q φ ) is small as this implies well-behaved importance weights.

In contrast, a small 'exclusive' KL-divergence KL(q φ π θ ) is not sufficient for well-behaved importance weights because the latter only ensures that q φ is close to π θ in those parts of the space Z in which q φ has positive probability mass.

Let Q := {q φ } (which is indexed by φ) be the family of proposal distributions/the variational family.

Then we can distinguish two scenarios.

1.

Sufficiently expressive Q.

For the moment, assume that the family Q is flexible ('expressive') enough in the sense that it contains a distribution q φ which is (at least approximately) equal to π θ and that our optimiser can reach the value φ of φ.

In this case, minimising the exclusive KL-divergence can still yield well-behaved importance weights because in this case, φ := arg min φ KL(π θ q φ ) is (at least approximately) equal to arg min φ KL(q φ π θ ).

2.

Insufficiently expressive Q. In general, the family Q is not flexible enough in the sense that all of its members are 'far away' from π θ , e.g. if the

is fully factorised.

In this case, minimising the exclusive KL-divergence could lead to poorly-behaved importance weights and we should optimise φ := arg min φ KL(π θ q φ ) as discussed above.

Remark 4.

In Scenario 1 above, i.e. for a sufficiently flexible Q, using a gradient-descent algorithm which seeks to minimise the exclusive divergence can sometimes be preferable to a gradient-descent algorithm which seeks to minimise the inclusive divergence.

This is because both find (approximately) the same optimum but the latter may exhibit faster convergence in some applications.

In such scenarios, the discussion in Subsection A.1 indicates that a smaller number of particles, K, could then be preferable for some of the φ-gradients because (a) the O(K −1 ) self-normalisation bias outweighs the O(K −1/2 ) standard deviation and (b) the direction of this bias may favour faster convergence.

Unfortunately, simply setting K = 1 for the approximation of the φ-gradients 2 is not necessarily optimal because

• even in the somewhat idealised scenario 1 above and even if the direction of the self-normalisation bias encourages faster convergence, increasing K is still desirable to reduce the variance of the gradient approximations and furthermore, even in this scenario, seeking to optimise the exclusive KL-divergence could lead to poorly behaved importance-sampling approximations of the θ-gradient whenever φ is still far away from optimal;

• not using the information contained in all K particles and weights (which have already been sampled/calculated to approximate the θ-gradient) seems wasteful;

• if K = 1, the reparametrisation-free AISLE φ-gradients, AISLE-KL-NOREP and AISLE-χ 2 -NOREP are simply vanilla Monte Carlo estimates of 0 and the RWS-DREG φ-gradient is then equal to 0.

In these supplementary materials, we illustrate the different φ-gradient estimators (recall that all algorithms discussed in this work share the same θ-gradient estimator).

Specifically, we compare the following approximations.

• AISLE-KL-NOREP.

The gradient for AISLE based on the KL-divergence without any further reparametrisation from (14) i.e. this coincides with the standard RWSgradient from (7).

This gradient does not require R1 but does not achieve zero variance even if q φ = π θ .

• AISLE-KL.

The gradient for AISLE based on the KL-divergence after reparametrising and exploiting the identity from Lemma 1; it is given by (15) and coincides with the IWAE-STL-gradient from (4).

• AISLE-χ 2 -NOREP.

The gradient for AISLE based on the χ 2 -divergence without any reparametrisation given in (16).

This gradient again does not require R1 but does not achieve zero variance even if q φ = π θ .

• AISLE-χ 2 .

The gradient for AISLE based on the χ 2 -divergence after reparametrising and exploiting the identity from Lemma 1; it is given by (17) and is alsow proportional to IWAE-DREG from Tucker et al. (2019) which was stated in (5).

When normalising the gradients (as, e.g. implicitly done by optimisers such as ADAM Kingma & Ba, 2015) the proportionality constant cancels out so that both these gradient approximations lead to computationally the same algorithm.

• IWAE.

The gradient for IWAE employing the reparametrisation trick from Kingma & Welling (2014) .

Its sampling approximation is given in (3).

Recall that this is the φ-gradient whose signal-to-noise ratio degenerates with K as pointed out in Rainforth et al. (2018) (and which also cannot achieve zero variance even if q φ = π θ ).

• IWAE-DREG.

The 'doubly-reparametrised' IWAE gradient from (5) which was proposed in Tucker et al. (2019) .

It is proportional to AISLE-χ 2 .

• RWS-DREG.

The 'doubly-reparametrised' RWS φ-gradient from (8) Hereafter, wherever necessary, we add an additional subscript to make the dependence on the observations explicit.

The joint law (the 'generative model'), parametrised by θ, of the observations and latent variables then factorises as

We model each latent variable-observation pair (z, x) as

..,D} ∈ R D×D is assumed to be known and where I denotes the D × D-identity matrix.

For any θ,

with P := (Σ −1 + I) −1 and ν θ,x := P (Σ −1 µ + x).

In particular, (18) implies that

Proposal/variational approximation.

We take the proposal distributions as a fullyfactored Gaussian:

where

.

The parameters to optimise are thus

where

denotes the column vector formed by the elements in the dth row of A. Furthermore, for the reparametrisation trick, we take q( ) := N( ; 0, I), where 0 ∈ R D is a vector whose elements are all 0, so that

Note that the mean of the proposal in (20) coincides with the mean of the posterior in (19) if A = P and b

This model is similar to the one used as a benchmark in Rainforth et al. (2018, Section 4) and also in Tucker et al. (2019, Section 6.1) who specified both the generative model and the variational approximation to be isotropic Gaussians.

Specifically, their setting can be recovered by taking Σ := I and fixing c d = log(2/3)/2 so that C = 2 3 I throughout.

Here, in order to investigate a slightly more realistic scenario, we also allow for the components of the latent vectors z to be correlated/dependent under the generative model.

However, as the variational approximation remains restricted to being fully factored, it may fail to fully capture the uncertainty about the latent variables.

φ,x (z), we then have

Note that the only source of randomness in this expression is the multivariate normal random variable .

Thus, by (23) and (24), for any values of A and b and any K ≥ 1, the variance of the A-and b-gradient portion of AISLE-KL/IWAE-STL and AISLE-χ 2 /IWAE-DREG goes to zero as C → C = 1 2 I. In other words, in this model, these 'score-function free' φ-gradients achieve (near) zero variance for the parameters governing the proposal mean as soon as the variance-parameters fall within a neighbourhood of their optimal values.

Furthermore, (25) combined with (26) shows that for any K ≥ 1, the variance of the C-gradient portion also goes to zero as (A, b, C) → (A , b , C ) .

A more thorough analysis of the benefits of reparametrisation-trick gradients in Gaussian settings is carried out in Xu et al. (2019) .

Setup.

We end this section by empirically comparing the algorithms from Subsection B.1.

We run each of these algorithms for a varying number of particles, K ∈ {1, 10, 100}, and varying model dimensions, D ∈ {2, 5, 10}. Each of these configurations is repeated independently 100 times.

Each time using a new synthetic data set consisting of N = 25 observations sampled from the generative model after generating a new 'true' prior mean vector as µ ∼ N(0, I).

Since all the algorithms share the same θ-gradient, we focus only on the optimisation of φ and thus simply fix θ := θ ml throughout.

We show results for the following model settings.

• Figure 1 .

The generative model is specified via Σ = I. In this case, there exists a value φ of φ such that q φ,x (z) = π θ,x (z).

Note that this corresponds to Scenario 1 in Subsection A.2.

• Figure 2 .

The generative model is specified via Σ = (0.95 |d−d |+1 ) (d,d )∈{1,...,D} 2 .

Note that in this case, the fully-factored variational approximation cannot fully mimic the dependence structure of the latent variables under the generative model.

That is, in this case, q φ,x (z) = π θ,x (z) for any values of φ.

Note that this corresponds to Scenario 2 in Subsection A.2.

To initialise the gradient-ascent algorithm, we draw each component of the initial values φ 0 of φ IID according to a standard normal distribution.

We use both plain stochastic gradient-ascent with the gradients normalised to have unit L 1 -norm (Figures 1a, 2a) and ADAM (Kingma & Ba, 2015) with default parameter values (Figures 1b, The total number of iterations is 10, 000; in each case, the learning-rate parameters at the ith step are i −1/2 .

Figure 1 except that here, the covariance matrix Σ = (0.95 |d−e|+1 ) (d,e)∈{1,...,D} 2 is not a diagonal matrix.

Again, note the logarithmic scaling on the second axis.

@highlight

We show that most variants of importance-weighted autoencoders can be derived in a more principled manner as special cases of adaptive importance-sampling approaches like the reweighted-wake sleep algorithm.
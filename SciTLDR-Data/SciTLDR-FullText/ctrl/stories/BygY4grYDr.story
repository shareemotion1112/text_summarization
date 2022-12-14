Interpreting generative adversarial network (GAN) training as approximate divergence minimization has been theoretically insightful, has spurred discussion, and has lead to theoretically and practically interesting extensions such as f-GANs and Wasserstein GANs.

For both classic GANs and f-GANs, there is an original variant of training and a "non-saturating" variant which uses an alternative form of generator gradient.

The original variant is theoretically easier to study, but for GANs the alternative variant performs better in practice.

The non-saturating scheme is often regarded as a simple modification to deal with optimization issues, but we show that in fact the non-saturating scheme for GANs is effectively optimizing a reverse KL-like f-divergence.

We also develop a number of theoretical tools to help compare and classify f-divergences.

We hope these results may help to clarify some of the theoretical discussion surrounding the divergence minimization view of GAN training.

Generative adversarial networks (GANs) (Goodfellow et al., 2014) have enjoyed remarkable progress in recent years, producing images of striking fidelity, resolution and coherence (Karras et al., 2018; Miyato et al., 2018; Brock et al., 2018; Karras et al., 2019) .

There has been much progress in both theoretical and practical aspects of understanding and performing GAN training (Nowozin et al., 2016; Mescheder et al., 2018; Gulrajani et al., 2017; Sønderby et al., 2017; Miyato et al., 2018; Karras et al., 2018; Brock et al., 2018; Karras et al., 2019) .

One of the key considerations for GAN training is the scheme used to update the generator and critic.

A rich avenue of developments has come from viewing GAN training as divergence minimization.

Goodfellow et al. (2014) showed the conventional GAN training can be viewed as approximately minimizing the Jensen-Shannon divergence.

f-GANs (Nowozin et al., 2016) approximately minimize f-divergences such as reverse KL in a principled way.

Wasserstein GANs approximately minimize the Wasserstein metric, and combine solid theoretical underpinnings with strong practical results.

Nevertheless a relatively unprincipled "non-saturating" scheme (Goodfellow et al., 2014) has continued to obtain groundbreaking results (Karras et al., 2019) and remains a state-of-the-art approach (Lucic et al., 2018) .

The effect of the non-saturating scheme on training dynamics, and in particular whether it can be viewed as divergence minimization, has been source of discussion and some confusion since the original formulation of GAN training (Goodfellow et al., 2014) .

The main result of this paper is to show that the non-saturating scheme approximately minimizes the f-divergence 4 KL( 1 2 p + 1 2 q p), which we refer to as the softened reverse KL divergence ( §6).

This puts non-saturating training on a similar footing to Wasserstein GANs as a theoretically sound approach with strong empirical results.

We also discuss how our results relate to previous attempts at this problem and attempt to clarify some of the confusion surrounding the divergence minimization view of non-saturating training.

In order to better understand the qualitative behavior of different divergences such as softened reverse KL, we develop several tools.

We show how to write f-divergences in a symmetry-preserving way, allowing easy visual comparison of f-divergences in a way that reflects their qualitative properties ( §7).

We develop a rigorous formulation of tail weight which generalizes the notions of modeseeking and covering behavior ( §8).

Using these tools we show that the softened reverse KL divergence is fairly similar to the reverse KL but very different to the Jensen-Shannon divergence approximately minimized by the original GAN training scheme.

The precise practical effect of the non-saturating scheme and whether it can be motivated in a principled way have been a source of discussion and some confusion.

In this section we review previous attempts to view non-saturating gradients as a form of divergence minimization.

The original GAN paper claims that, compared to the saturating training scheme based on the Jensen-Shannon divergence, the non-saturating training scheme "results in the same fixed point of the dynamics of G and D but provides much stronger gradients early in learning." (Goodfellow et al., 2014, Section 3) .

It is true that the original and non-saturating generator gradients give the same final result in the non-parametric case where q is unrestricted, but this is fairly trivial since both gradients lead to q = p, as do all divergences.

It is even true that the dynamics of training are essentially the same for the original and non-saturating gradients when q ≈ p, but again this is fairly trivial since all f-divergences agree in this regime, as discussed in §3.

However the "fixed point of the dynamics" is certainly not the same in the general case of parametric q (see §G for an empirical demonstration).

Our results provide a precise way to view the relationship between saturating and non-saturating generator gradients: They are optimizing different f-divergences.

The original f-GAN paper presents a simple argument that the "non-saturating" training scheme has the same fixed points and that the original and non-saturating generator gradients have the same direction (Nowozin et al., 2016, Section 3.2) 1 .

However this argument is erroneous.

It is true that if p ≈ q then (f * ) (f (u)) is approximately 1 everywhere, and so the original and non-saturating generator gradients are approximately equal, but this is true of any f-divergence.

There is no guarantee that the regime p ≈ q will ever be approached in the general case where q belongs to a parametric family, it is not the case that the original and non-saturating generator gradients point in approximately the same direction in general (see §G for an empirical demonstration).

In fact, the non-saturating form of generator gradient can have completely different qualitative behavior.

For example, we show that the non-saturating KL scheme in fact optimizes reverse KL.

A recent paper showed experimentally that the non-saturating generator gradient can successfully learn a distribution in a case where optimizing Jensen-Shannon divergence should fail, and used this to argue that perhaps it is not particularly helpful to view GANs as optimizing Jensen-Shannon divergence (Fedus et al., 2018) .

The divergence optimized in practice for parametric critics is not exactly the divergence which would be optimized by the theoretically optimal critic, and this distinction seems particularly important in the situation where p and q initially have non-overlapping support.

However the fact that non-saturating training is not optimizing Jensen-Shannon is also highly relevant to this discussion, since the gradient in the limit of zero noise is zero for Jensen-Shannon but sizeable for softened reverse KL.

Thus the success of non-saturating GAN training in practice may be as much due to its optimizing a different divergence as it is to using an inexact critic.

Arjovsky and Bottou correctly recognize that the non-saturating generator gradient results in approximately minimizing a different objective function and derive the function for classic GANs (Arjovsky & Bottou, 2017, Section 2.2.2) .

The objective function there is expressed as KL(q p) − 2 JS(p, q) (1) which is a slightly convoluted form of the expression 2 KL( 1 2 p+ 1 2 q p) we derive below.

The paper suggests the negative sign of the second term is "pushing for the distributions to be different, which seems like a fault in the update", whereas our expression for the divergence makes it clear that this is not an issue.

Poole et al. (2016) present a very similar view to that presented in this paper, including recognizing that the generator and critic may be trained to optimize different f-divergences and interpreting the classic non-saturating generator gradient as a hybrid scheme of this form where the generator gradient is based on a new f-divergence (Poole et al., 2016) .

However the f-divergence derived there is f (u) = log(1 + u −1 ), which differs from (50) by a factor of u + 1.

We refer to this as the improved generator objectives for GANs (IGOG) divergence.

It can be written as

(1+u) 2 u 2 , and has (2, 0) tail weights.

Figure 3 shows that this divergence is qualitatively quite similar to the softened reverse KL but is not identical.

The source of the discrepancy between our results and theirs is matching the value instead of the gradient, and is described in detail in §A.

We start by reviewing the definition of an f-divergence (Ali & Silvey, 1966) and establishing some basic properties.

These properties are described in more detail in §B. Throughout the paper we use the convention that p is the "true" distribution and q is a model intended to approximate p.

Given a strictly convex twice continuously differentiable function f : R >0 → R with f (1) = 0, the f -divergence between probability distributions with densities 2 p and q over R K is defined as:

f-divergences satisfy several mathematical properties.

Firstly D f is linear in f .

Secondly D f (p, q) ≥ 0 for all distributions p and q with equality iff p = q. This justifies referring to D f as a divergence.

D f is completely determined by f .

As we will see, the algebraic form of f is often simpler than that of f .

All f-divergences agree up to an overall scale factor on the divergence between nearby distributions:

.

This can also be seen in Figure 2 , where all f-divergences approximately overlap near zero.

If f (1) = 0 and f (1) = 1 then we say f is in canonical form.

We can always find such an f by appropriately scaling D f .

Using canonical form removes a superficial difference in scaling between different f-divergences, making them easier to compare, e.g. in Figure 2 .

The definition (2) appears to be quite asymmetric in how it treats p and q, but it obeys a particular symmetry (Reid & Williamson, 2011) .

This is more explicitly symmetric than (2) in the role of p and q. We refer to A as the set of left mismatches (q > p), and B as the set of right mismatches (q < p).

At each point in A, the two distributions p and q are somewhat mismatched, and the penalty paid for this mismatch in terms of the overall divergence D f is governed by the behavior of f (u) for 0 < u < 1 (the "left" of the graph of f ).

Similarly the penalty paid for right mismatches is governed by f (u) for u > 1.

Note from (3) that a left mismatch can only be heavily penalized if the point is plausible under q, i.e. q(x) is not tiny.

Similarly a right mismatch can only be heavily penalized for points plausible under p.

f-GANs are based on an elegant way to estimate the f-divergence between two distributions given only samples from the two distributions (Nguyen et al., 2010) .

In this section we review this approach to variational divergence estimation.

See §E for details on how our derivation and notation relates to that of Nowozin et al. (2016) .

There is an elegant variational bound on the f-divergence D f (p, q) between two densities p and q. Since f is strictly convex, its graph lies at or above any of its tangent lines and only touches in one place.

That is, for k, u > 0,

with equality iff k = u. This inequality is illustrated in the appendix in Figure 4 .

Substituting p(x)/q(x) for k and u(x) for u, for any continuously differentiable function u :

with equality iff u = u * , where u * (x) = p(x)/q(x).

The function u is referred to as the critic.

It will be helpful to have a concise notation for this bound.

Writing u(x) = exp(d(x)) without loss of generality, for any continuously differentiable function d :

with equality iff d = d * , where

Note that both a f and b f are linear in f .

Their derivatives a f (log u) = uf (u) and b f (log u) = u 2 f (u) depend on f only through f .

The bound (6) leads naturally to variational divergence estimation.

The f -divergence between p and q can be estimated by maximizing E f with respect to d (Nguyen et al., 2010) .

Conveniently E f is expressed in terms of expectations and may be approximately computed and maximized with respect to d using only samples from p and q. If we parameterize d as a neural net d ν with parameters ν then we can approximate the divergence by maximizing E f (p, q, d ν ) with respect to ν.

This does not compute the exact divergence because there is no guarantee that the optimal function d * lies in the family {d ν : ν} of functions representable by the neural net, but we hope that for sufficiently flexible neural nets the approximation will be close.

Here we briefly summarize the three main f-divergences we consider.

The Kullback-

It has (1, 2) tail weights and is left-bounded and right-unbounded.

The reverse KL divergence KL(q p) has Consider the task of estimating a probabilistic model from data using an f-divergence.

Here p is the true distribution and the goal is to minimize l(λ) = D f (p, q λ ) with respect to λ, where λ → q λ is a parametric family of densities over R K .

We refer to q λ as the generator.

For implicit generative models such as typical GAN generators, the distribution q λ is the result of a deterministic transform x λ (z) of a stochastic latent variable z. However we do not need to assume this specific form for most of our discussion.

We first note that the variational divergence bound E f satisfies a convenient gradient matching property.

This is not made explicit in the original f-GAN paper.

Denote the optimal d given p and

They also match gradients: From the definitions of D f and E f we can verify that they have the same gradient with respect to the generator parameters λ:

We can minimize

with respect to ν while minimizing it with respect to λ.

Adversarial optimization such as this lies at the heart of all flavors of GAN training.

Define λ and ν as

To perform the adversarial optimization, we can feed λ and ν (or in practice, stochastic approximation to them) as the gradients into any gradient-based optimizer designed for minimization, e.g. stochastic gradient descent or ADAM.

There is a simple generalization of the above training procedure, which is to base the generator gradients on E f but the critic gradients on E g for a possibly different function g (Poole et al., 2016, Section 2.3).

We refer to this as using hybrid (f, g) gradients.

This also approximately minimizes D f .

See §F for more details.

When training classic GANs in practice, an alternative non-saturating loss is used as the basis for the generator gradient, and is found to perform much better in practice (Goodfellow et al., 2014) .

This issue has been discussed in detail previously, so we just give a summary here and discuss in more detail in §F. Early on in training, the generator and data distribution are typically not well matched, with samples from p being very unlikely under q and vice versa.

This means most of the probability mass of p and q is in regions where d has large magnitude, corresponding to the positive and negative tails in Figure 2 and (19).

In this regime Jensen-Shannon has very flat gradient, and it is not too surprising that this might lead to optimization issues.

Similar concerns do not apply to other f-divergences such as KL or reverse KL, but an alternative "non-saturating" generator gradient has still been suggested for use in f-GANs (Nowozin et al., 2016) .

For both GANs and f-GANs the specific change is to replace b f by a f in the definition of λ in (12).

We are not aware of a particular motivation for this procedure in the case of f-GANs other than that it yields the traditional non-saturating GAN scheme in the case of Jensen-Shannon.

We now discuss the effect of the non-saturating generator gradient on training.

We show that, for an optimal critic, the non-saturating generator gradient is the gradient of a globally coherent objective function, that this objective function is an f-divergence, and that this f-divergence is not the same as the one optimized by using the original "saturating" gradient.

We explicitly derive the divergences optimized by the "non-saturating" KL, reverse KL and Jensen-Shannon training schemes.

We first establish our main result: "Non-saturating" training based on g is precisely equivalent to a hybrid (f, g) scheme for some f .

Consider the f-divergence D f defined by

where the constant k ∈ R does not affect the (reparameterized) gradients.

Since the non-saturating gradient uses b instead of a in the definition of its generator gradient λ, an original generator gradient using f is the same as a non-saturating generator gradient using g. Since the critic gradient is still based on g, the overall scheme is a hybrid (f, g) one, and so approximately minimizes D f .

We now explicitly compute the corresponding f for some common choices of g. It is easy to show that if D g has (R, S) tails then D f has (R+1, S −1) tails, so the divergence effectively optimized by non-saturating training penalizes left mismatches more strongly and right mismatches less strongly than the original divergence.

For the KL divergence, g (u) = u −1 , so f (u) = u −2 .

We already saw in §4 that this is the reverse KL divergence.

Thus "non-saturating" training based on the KL divergence is a hybrid (reverse KL, KL) scheme, and so in fact approximately minimizes the reverse KL.

This equivalence also follows directly from the equality of KL's a g to reverse KL's b f .

For the reverse KL divergence, g (u) = u −2 , so f (u) = u −3 .

The corresponding f may be obtained by integrating twice, choosing constants of integration such that f is canonical.

We show in §D that D f is the canonicalized Pearson χ 2 divergence.

It has (3, 0) tail weights and is left-unbounded and right-bounded.

Thus "non-saturating" training based on the reverse KL divergence is a hybrid ( 1 2 χ 2 , reverse KL) scheme, and so approximately minimizes the Pearson χ 2 divergence.

For the canonicalized Jensen-Shannon divergence,

It has (2, 0) tail weights and is left-unbounded and right-bounded.

Thus the non-saturating training scheme described by Goodfellow et al. (2014) is a hybrid (SRKL, JS) scheme, and so approximately minimizes the softened reverse KL.

Having derived our main result that the typical non-saturating GAN training scheme effectively optimizes the softened reverse KL divergence, we focus on understanding the qualitative properties of this divergence.

We do this by developing some analytic tools applicable to any f-divergence.

While f-divergences unify many divergences, just plotting the function f is often not informative.

The symmetric relationship between divergences such as KL and reverse KL is obfuscated, and f may grow quickly even when the divergence is well-behaved.

In this section we develop a straightforward and intuitive way to compare f-divergences visually through a symmetry-preserving divergence plot.

Our perspective also allows a simple summary of the prevalence of mismatches between p and q, through a pushforward plot.

Firstly note that for x ∼ q(x), p(x)/q(x) is a random variable with some distribution.

In fact, since (2) is the expected value of some function of this random variable, D f (p, q) must depend only on the one-dimensional distribution of this random variable and not on the detailed distribution of p and q in space.

Formally the distribution of this random variable may be described as the pushforward measure of q through the function u * (x) = p(x)/q(x).

To obtain more intuitive plots, we will work in terms of d * (x) = log p(x) − log q(x) instead of u * .

We denote the density of the pushforward of

Rewriting the expectation in (2), we obtain

As above we can write this more symmetrically.

Define

By considering expectations of an arbitrary function of d expressed in x-space and d-space, we can show thatq

Thus, using (3) and (17), we can write the f-divergence as

An f-divergence D f (p, q) involves an interaction between the distributions p, q and the function f , and (19) nicely decomposes this interaction in terms of something that only depends on p and q (the pushforwards) and something that only depends on f (the function s f ), connected via a onedimensional integral.

By plotting s f and imagining integrating against various pushforwards, we can see the properties of different f-divergences in a very direct way.

By plotting the pushforwards, we can get a feel for what types of mismatch between p and q are present in multidimensional x-space, and understand at a glance how badly these mismatches would be penalized for a given f-divergence.

Examples of pushforwards for the simple case where p and q are multidimensional Gaussians with common covariance are shown in Figure 1 .

In this case the pushforwardsq d * andp d * are themselves one-dimensional Gaussians (since d * is linear), with densities N (− 1 2 σ 2 , σ 2 ) and N ( 1 2 σ 2 , σ 2 ) respectively, for some σ (this follows from (17)).

Examples of s f for various f-divergences are shown in Figure 2 .

We refer to s f as a symmetry-preserving representation of f .

Note that as long as f is in Figure 1 using (19) .

Symmetries such as that between KL and reverse KL are evident.

canonical form, s f is twice continuously differentiable at zero.

Figure 2 directly expresses several facts about divergences.

It shows that left mismatches (regions of space where q(x) > p(x), corresponding to d < 0) are penalized by reverse KL much more severely than right mismatches (regions of space where q(x) > p(x), corresponding to d > 0).

The symmetry between KL and reverse KL is evident.

We see that Jensen-Shannon and the Jeffreys divergence (the average of KL and reverse KL) are both symmetric in how they penalize left and right mismatches, but differ greatly in how much they penalize small versus large mismatches.

Applying the tools developed in this section to analyze the non-saturating variant of GAN training, Figure 3 shows the symmetry-preserving representation s f (d) for the Jensen-Shannon and softened reverse KL divergences, as well as the reverse KL for comparison.

The qualitative behavior of softened reverse KL is quite similar to reverse KL.

As discussed in §C, softening has the potential to make large right mismatches much less severely penalized, thus making the divergence more modeseeking.

Here softening increases the slope of the left tail and changes the right tail behavior slightly, but these changes are relatively minor modifications.

The Jensen-Shannon is extremely different to the reverse KL and softened reverse KL.

In this section we introduce a classification scheme for f-divergences in terms of their behavior for large left and right mismatches.

While different f-divergences differ in details, this classification determines many aspects of their qualitative behavior.

First we define the notion of tail weight and examine some of its consequences.

If f (u) ∼ Cu −R as u → 0 for C > 0 and f (u) ∼ Du S−3 as u → ∞ for D > 0 then we say that D f has (Cu −R , Du S−3 ) tails and (R, S) tail weights.

Here we have used the notation g(u) ∼ h(u) as u → a to mean g(u)/h(u) → 1 as u → a. Note that, since f R (u) = u −3 f (u −1 ), f having a u S−3 right tail is equivalent to f R having a u −S left tail.

Thus tail weights interact simply with symmetry: If D f has (R, S) tail weights then D fR has (S, R) tail weights.

Intuitively, the left tail weight R determines how strongly large left mismatches are penalized compared to small mismatches (which are penalized the same amount by every canonical f-divergence), whereas the right tail weight S determines how strongly large right mismatches are penalized compared to small mismatches.

Some f-divergences such as Jensen-Shannon are bounded, while others such as KL are unbounded, and it is useful to have a characterization of when boundedness occurs.

We say D f is bounded if there is an M ∈ R such that D f (p, q) ≤ M for all densities p and q. We say f is left-bounded if f is bounded on (0, 1), and right-bounded if f R is bounded on (0, 1), or equivalently if f (u)/u is bounded on u > 1.

From (3) it is easy to see that if f is left-bounded and right-bounded then D f is bounded.

The converse is also true: If f is left-unbounded or right-unbounded then we can find p and q with arbitrarily large divergence D f (p, q).

This can be seen for example by partitioning R K into two sets A and B and considering densities p and q which are constant on A and constant on B, or strictly speaking smooth approximations thereof.

Tail weight determines boundedness.

It can be checked by integrating and bounding that a divergence with (R, S) tail weights is left-bounded iff R < 2 and right-bounded iff S < 2.

Thus D f is bounded iff R, S < 2.

The tail weights and boundedness properties of various f-divergences considered in this paper are summarized in Table 1 .

Boundedness properties can also be seen in Figure 2 .

Left and right boundedness of f is trivially equivalent to left and right boundedness of s f .

Thus we can see that reverse KL is left unbounded but right bounded, for example.

The unbounded tails in this plot are all asymptotically linear in d.

Tail weights provide an extension of the typical classification of divergences as mode-seeking or covering (Bishop, 2006, Section 10.1.2).

Models trained with reverse KL tend to have distributions which are more compact than the true distribution, sometimes only successfully modeling certain modes (density peaks) of a multi-modal true distribution.

Models trained with KL tend to have distributions which are less compact than the true distribution, "covering" the true distribution entirely even if it means putting density in regions which are very unlikely under the true distribution (Bishop, 2006, Figure 10.3) .

However there are important qualitative aspects of divergence behavior that are not captured by these labels.

For example, Jensen-Shannon is neither mode-seeking nor covering: It would be more accurate to say that a model trained using Jensen-Shannon tries to match very closely when it matches, but doesn't worry overly about large mismatches in either direction.

The Jeffreys divergence is also symmetric and so neither mode-seeking nor covering, but has very different behavior from Jensen-Shannon.

Tail weights capture these distinctions in a straightforward but precise way.

Tail weights and boundedness provide an extremely concise way to see the qualitative effect of using the non-saturating variant of GAN training.

The softened reverse KL divergence effectively optimized by conventional non-saturating GAN training has tail weights (2, 0), and so is unbounded, is likely to have strong gradients starting from a random initialization where large mismatches are present, and penalizes left mismatches strongly but tolerates large right mismatches and so is modeseeking.

In contrast the Jensen-Shannon divergence effectively optimized by saturating GAN training has tail weights (1, 1), and so is bounded, is likely to have weak gradients in the presence of large mismatches, and tolerates large left and right mismatches.

As mentioned in §2, Poole et al. (2016) present a very similar view to that presented in this paper, including recognizing that the generator and critic may be trained to optimize different f-divergences and interpreting the classic non-saturating generator gradient as a hybrid scheme of this form where the generator gradient is based on a new f-divergence (Poole et al., 2016) .

We now discuss the discrepancy between our result and theirs.

In the language of the present paper, Poole et al. (2016, Equation (8)) define the approximation

and show that the gradients ofẼ f for this particular f match the non-saturating GAN gradients.

This is a valid approximation of the value, since

However the gradients are not the same: The partial derivative of the left side of (20) with respect to the parameters of q involves two terms, one for each occurrence of q(x) in the integrand, and the partial derivative of the right side only includes one of these.

Thus it is not the case that optimizingẼ f using gradient descent (while continually keeping the critical optimal) optimizes D f .

In this section we go into more detail about some of the properties of f-divergences which were briefly covered in the main text.

and g(1) = 0 then (f + g)(1) = 0 and (kf )(1) = 0, so D f +g and D kf are valid f-divergences.

Secondly note that adding an affine term to f (u) does not affect

Any affine term added must be of the form k − ku in order to respect the f (1) = 0 constraint.

Thus the second derivative f determines the divergence completely.

This property is also true of the various bounds and finite sample approximations 4 derived in this paper, so we may legitimately consider f rather than f as the essential quantity of interest for a given divergence.

Working with f has the added advantage that for many common f-divergences f has a simpler algebraic form than f .

For any densities p and q we have D f (p, q) ≥ 0 with equality iff p = q, as can be seen by plugging the constant function u(x) = 1 into (5).

If f (1) = 0 and f (1) = 1 then we say f is in canonical form.

We can put any f in canonical form by scaling and adding a suitable affine term, and this corresponds to a scaling of D f .

Each f-divergence has a unique canonical form.

Different f-divergences may behave very differently when p and q are far apart but are essentially identical when q ≈

p.

In fact the divergence between nearby distributions belonging to some family is given by f (1) times the Fisher metric of the family.

Specifically

where ε ∈ R, v ∈ R K , and

T is the Fisher information matrix for the parametric family of distributions specified by q λ .

Alternatively this may be stated in the non-parametric form

where v :

Informally we may state this as:

Thus all f-divergences agree up to a constant factor on the divergence between two nearby distributions, and they are all just scaled versions of the Fisher metric in this regime.

This can also be seen in Figure 2 , where all f-divergences approximately overlap near zero.

We can apply some simple operations to a divergence to obtain another divergence.

In this section we consider the effect of reversing, symmetrizing and softening operations on f-divergences.

Many common f-divergences can be obtained from others in this way, and this provides a unified way of concisely describing many f-divergences based on KL, for example.

Consider applying an operation to a divergence D(p, q) to obtain another divergenceD(p, q).

We already saw the reversing operationD(p, ).

The factor of 4 above is to ensure that the divergence remains canonical after softening, i.e. f (1) = 1.

Softening has the potential to make large right mismatches much less severely penalized, since in regions of space where p(x)/q(x) was large because p(x) was moderate and q(x) was tiny, p(x)/m(x) is now approximately 2, so a large right mismatch is only penalized by the softened divergence as much as a moderate right mismatch is penalized by the original divergence.

This is reflected in the tail weights: It is easy to show using the tools we have developed above that if the original divergence has (R, S) tail weights then the softened divergence has (R, 0) tail weights.

Many f-divergences can be written concisely as a series of these operations.

For example reverse KL is Reverse(KL), Jeffreys is Symmetrize(KL), the canonicalized K-divergence 4 KL(p m) (Cha, 2007) is Soften(KL) and canonicalized Jensen-Shannon is Symmetrize(Soften(KL)).

In this terminology, the main claim of this paper is that the non-saturating procedure for GAN training is in fact effectively minimizing the softened reverse KL divergence 4 KL(m p) given by Soften(Reverse(KL)).

In this section we give more details of the f-divergences considered in §4 and §6.

The expressions for D f and E f are obtained by plugging the chosen f into (2) and (7) respectively.

The KL divergence satisfies:

The KL divergence has (u −1 , u −1 ) tails, (1, 2) tail weights, and is left-bounded and rightunbounded.

The reverse KL divergence satisfies:

The reverse KL divergence has (u −2 , u −2 ) tails, (2, 1) tail weights, and is left-unbounded and rightbounded.

The Jensen-Shannon divergence JS(p, q) has f (1) = 1/4 and so is not canonical.

In most of the paper we therefore consider the canonicalized Jensen-Shannon divergence 4 JS(p, q).

This satisfies:

f (u) = 2u log u − 2(u + 1) log(u + 1) + 4 log 2 (36)

= 2 KL(p

The canonicalized Jensen-Shannon divergence has (2u −1 , 2u −2 ) tails, (1, 1) tail weights, and is both left-bounded and right-bounded and so bounded overall.

The Pearson χ 2 (or Kagan) divergence has f (1) = 2 as so is not canonical.

The canonicalized Pearson χ 2 divergence satisfies:

The canonicalized Pearson χ 2 divergence has (u −3 , u −3 ) tails, (3, 0) tail weights, and is leftunbounded and right-bounded.

The expression for f here corrects a swapped definition in the original f-GAN paper 5 (according to the definitions of the Pearson and Neyman divergences given in the paper, the expression given for the Pearson f is actually the Neyman f and vice versa) (Nowozin et al., 2016) .

In §6 we discussed the equality of the non-saturating reverse KL generator gradient to the conventional canonicalized Pearson χ 2 generator gradient.

This can be seen from (14), as we did in §6, or directly by noting that a f for the reverse KL divergence is equal to b f for the Pearson χ 2 divergence.

The softened reverse KL divergence satisfies:

5 In the the arxiv preprint, not the final NIPS version of the paper.

The SRKL divergence has (2u −2 , 2u −3 ) tails, (2, 0) tail weights, and is left-unbounded and rightbounded.

In §6 we discussed the equality of the non-saturating canonicalized Jensen-Shannon generator gradient to the conventional softened reverse KL generator gradient.

This can be seen from (14), as we did in §6, or directly by noting that a f for the canonicalized Jensen-Shannon divergence is equal to b f for the softened reverse KL divergence.

The original f-GAN paper (Nowozin et al., 2016) phrases the results presented in §4 in terms of the Legendre transform f * of f .

The two descriptions are equivalent, as can be seen by setting T (x) = f (u(x)) and using the result f * (f (u)) = uf (u) − f (u).

We find our description helpful since it avoids having to explicitly match the domain of f * , ensures the optimal d is the same for all f -divergences, and because the Legendre transform is complicated for one of the divergences we consider.

An "output activation" was used in the original f-GAN paper to adapt the output d of the neural net to the domain of f * .

This is equal to f (exp(d)), up to irrelevant additive constants, for all the divergences we consider, and so our description also matches the original description in this respect.

The gradient matching property shows that performing very many critic updates followed by a single generator update is a sensible learning strategy which, assuming the critic is sufficiently flexible and amenable to optimization, essentially performs very slow gradient-based optimization on the true divergence D f with respect to λ.

However in practice performing a few critic updates for each generator update, or simultaneous generator and critic updates, performs well, and it is easy to see that these approaches at least have the correct fixed points in terms of Nash equilibria of E f and optima of D f , subject as always to the assumption that the critic is sufficiently richly parameterized.

Convergence properties of these schemes are investigated much more thoroughly elsewhere, for example (Nagarajan & Kolter, 2017; Gulrajani et al., 2017; Mescheder et al., 2017; Balduzzi et al., 2018; Peng et al., 2019) , and are not the main focus here.

A similar discussion applies to hybrid schemes.

Subject as always to the assumption of a richly parameterized critic, if we perform very many critic updates for each generator update, then the d used to compute the generator gradient will still be close to d * , and so the generator gradient will be close to the gradient of D f , even though the path d took to approach d * was governed by g rather than f .

The fixed points of the two gradients are also still correct, and so it seems reasonable to again use more general update schemes and we might hope for similar convergence results (not analyzed here).

For an implicit generative model x λ (z) where z ∼ P(z), we have

Thus there is a b f (d) factor in the generator gradient, and in fact this is the only way the choice of f-divergence affects the generator gradient.

For reverse KL, b f (d) = 1, allowing the gradients from the other factors to pass freely.

Most of the contribution to the initial gradient for reverse KL is likely to come from regions in space with large negative d due to the P(z) factor.

For canonicalized Jensen-Shannon, b f (d) = 2σ(d), which tends to zero exponentially quickly as d → −∞ and tends to 2 as d → ∞. Regions of space with large positive d have a tiny contribution to the gradient due to the P(z) factor, while regions with large negative d are exponentially suppressed by b f (d).

Based on these considerations it might be tempting to conclude that left-unboundedness is the most important factor in being able to learn from a random initialization.

A divergence with left tail weight R has b f (d) ∼ exp(−d(R − 2)) so R ≥ 2 ensures that b f (d) does not decay exponentially as d → −∞.

However the case of KL shows that right-unboundedness is also capable of allowing learning.

For KL, b f (d) = exp d, and the situation is complicated, since it exponentially magnifies gradients from regions with large positive d, which are extremely unlikely under P(z).

We know the overall gradient can sometimes be a reasonable learning signal, since training models such as a multivariate Lines show the progression of SGD-based JS training based on the original, saturating gradient and based on the non-saturating gradient (solid for learned critic; dotted for optimal critic).

The original scheme converges to the JS divergence minimum.

The non-saturating scheme, which by the results of this paper is equivalent to a hybrid (SRKL, JS) scheme, converges to the SRKL divergence minimum as expected.

Gaussian using KL divergence works well.

However even if the expected gradient allows learning, the stochastic approximation obtained by sampling from q is likely to have extremely large variance.

The saturation issue is sometimes presented as being specific to the loss E f used for classic GAN training, but the gradient matching property presented in §5 shows it is fundamental to the JensenShannon divergence.

The more critic updates we perform initially, the more saturated d is on samples from q, and the more closely the gradient of E f with respect to λ approximates the gradient of the true divergence D f .

The typical fix to the saturation issue is to use the non-saturating generator gradient

Since the gradient of log σ(d) tends to 1 as d tends to −∞, the gradient used for training is now larger.

In order to validate our mathematical conclusions we conducted a simple experiment.

Training behavior using the original and non-saturating gradients on a toy problem is shown in Figure 5 .

We see that the two cases minimize different divergences, as expected based on the theoretical arguments presented above.

<|TLDR|>

@highlight

Non-saturating GAN training effectively minimizes a reverse KL-like f-divergence.

@highlight

This paper proposes a useful expression of the class of f-divergences, investigates theoretical properties of popular f-divergences from newly developed tools, and investigates GANs with the non-saturating training scheme.
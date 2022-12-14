We investigate the properties of multidimensional probability distributions in the context of latent space prior distributions of implicit generative models.

Our work revolves around the phenomena arising while decoding linear interpolations between two random latent vectors -- regions of latent space in close proximity to the origin of the space are oversampled, which restricts the usability of linear interpolations as a tool to analyse the latent space.

We show that the distribution mismatch can be eliminated completely by a proper choice of the latent probability distribution or using non-linear interpolations.

We prove that there is a trade off between the interpolation being linear, and the latent distribution having even the most basic properties required for stable training, such as finite mean.

We use the multidimensional Cauchy distribution as an example of the prior distribution, and also provide a general method of creating non-linear interpolations, that is easily applicable to a large family of commonly used latent distributions.

Generative latent variable models have grown to be a very popular research topic, with Variational Auto-Encoders (VAEs) BID8 and Generative Adversarial Networks (GANs) BID4 gaining a lot of interest in the last few years.

VAEs use a stochastic encoder network to embed input data in a typically lower dimensional space, using a conditional probability distribution p(z|x) over possible latent space codes z ∈ R D .

A stochastic decoder network is then used to reconstruct the original sample.

GANs, on the other hand, use a generator network that creates data samples from noise z ∼ p(z), where p(z) is a fixed prior distribution, and train a discriminator network jointly to distinguish between real and generated data.

Both of these model families require a probability distribution to be defined on the latent space.

The most popular variants are the multidimensional normal distribution and the uniform distribution on the zero-centred hypercube.

Given a trained model, studying the structure of the latent space is a common way to measure generator capabilities.

There are various methods used to analyse the latent space.

Locally, one can sample and decode points in close neighbourhood of a given latent vector to investigate a small region in the space.

On the other hand, global methods are designed to capture long-distance relationships between points in the space, e.g. latent arithmetics, latent directions analysis, and interpolations (see e.g. BID10 ; BID7 ; BID12 ; BID15 ; BID0 ).The main advantage of using interpolations is the interpretability that comes with dealing with onedimensional curves, instead of high-dimensional Euclidean space.

For example, if the model has managed to find a meaningful representation, one would expect the latent space to be organised in a way that reflects the internal structure of the training dataset.

In that case, decoding an interpolation will show a gradual transformation of one endpoint into the other.

Contrarily, if the model memorises the data, the latent space might consist of regions corresponding to particular training examples, divided by boundaries with unnatural, abrupt changes in generated data BID1 .

We * These two authors contributed equally This work was supported by National Science Centre, Poland (grants no. 2015/19/B/ST6/01819).need to note that this notion of "meaningful representation" is not enforced by the training objective.

However, it is not contradicting the objective, making it necessary to use additional tools to evaluate whether the learned manifold is coherently structured and equipped with desirable qualities.

What distinguishes interpolations from other low-dimensional methods is the shortest path property.

In absence of any additional knowledge about the latent space, it feels natural to use the Euclidean metric.

In that case, the shortest path between two points is defined as a segment.

This is, probably the most popular, linear interpolation, formally defined as f L (x 1 , x 2 , λ) = (1−λ)x 1 +λx 2 , for λ ∈ [0, 1], where x 1 , x 2 are the endpoints.

Other definitions of shortest path might yield different interpolations, we will study some of them later on.

While traversing the latent space along the shortest path between two points, a well-trained model should transform the samples in a sensible way.

For example, if the modelled data has a natural hierarchy, we would expect the interpolation to reflect it, i.e. an image of a truck should not arise on a path between images of a cat and a dog.

Also, if the data can be described with a set of features, then an interpolation should maintain any features shared by the endpoints along the path.

For example, consider a dataset of images of human faces, with features such as wearing sunglasses, having a long beard, etc.

Again, this is not enforced by the training objective.

If one would desire such property, it is necessary to somehow include the information about the trained manifold in the interpolation scheme.

There has been an amount of work done on equipping the latent space with a stochastic Riemannian metric BID1 that additionally depends on the generator function.

The role of the shortest paths is fulfilled by the geodesics, and the metric is defined precisely to enforce some of the properties mentioned above.

This approach is somewhat complementary to the one we are concerned with -instead of analysing the latent space using simple tools, we would need to find a more sophisticated metric that describes the latent space comprehensively, and then analyse the metric itself.

If our goal was solely the quality of generated interpolation samples, the aforementioned approach would be preferable.

However, in this work we are concerned with evaluating the properties directly connected with the model's objective.

With that in mind, we criticise the broad use of linear interpolations in this particular context.

In this work we shall theoretically prove that linear interpolations are an incorrect tool for the stated task, and propose a simple, suitable interpolation variant.

While considered useful, the linear interpolation used in conjunction with the most popular latent distributions results in a distribution mismatch (also defined in BID0 ; BID7 ).

That is, if we fix the λ coefficient and interpolate linearly between two endpoints sampled from the latent space distribution, the probability distribution of the resulting vectors will differ significantly from the latent distribution.

This can be partially explained by the well-known fact that in high dimensions the norms of vectors drawn from the latent distribution are concentrated around a certain value.

As a consequence, the midpoints of sampled pairs of latent vectors will have, on average, significantly smaller norm.

Thus, the linear interpolation oversamples regions in close proximity of the origin of the latent space.

A thorough analysis of this phenomenon will be conducted in section 2.1.Such behaviour raises questions about the applicability of the linear interpolation to study the latent space.

Indeed, changing the latent distribution after the model was trained may have unexpected consequences.

In BID7 , experiments conducted using a DCGAN model BID12 on the celebA dataset BID9 showed flawed data generation near the latent space origin.

Other works concerning the traversal of latent space do not mention this effect, e.g. BID0 .

We recreated this experiment, and concluded that it might be caused by stopping the training process too early (see Appendix C figure 6 for a visualisation).

This may explain the apparent disagreement in the literature.

Nevertheless, with either a midpoint decoding to a median face, or a non-sensible sample, the interpolation is not informative -we would like to see smooth change of features, and not a transition through the same, homogeneous region.

The solution is, either, to change the latent distribution so that the linear interpolation will not cause a distribution mismatch, or redefine the shortest path property.

A simple well-known compromise is to use spherical interpolations BID13 BID15 .

As the latent distribution is concentrated around a sphere, replacing segments with arcs causes relatively small distribution mismatch (see section 3.2).

Nonetheless, reducing the consequences of the distribution mismatch is still a popular research topic BID0 BID7 BID1 ).

In section 2.1 we show that if the linear interpolation does not change the latent probability distribution, then it must be trivial or "pathological" (with undefined expected value).

Then, in section 2.2, we give an example of such an invariant distribution, namely the Cauchy distribution, thus proving its existence.

We also discuss the negative consequences of choosing a heavy-tailed probability distribution as the latent prior.

In section 3 we relax the Euclidean shortest path property of interpolations, and investigate nonlinear interpolations that do not cause the latent distribution mismatch.

We describe a general framework for creating such interpolations, and give two concrete examples in sections 3.4 and 3.5.

We find these interpolations to be appropriate for evaluating the model's objective induced properties in contrast to the linear interpolations.

The experiments conducted using the DCGAN model on the CelebA dataset are presented solely to illustrate the problem, not to study the DCGAN itself, theoretically or empirically.

In this section we will tackle the problem of distribution mismatch by selecting a proper latent distribution.

Let us assume that we want to train a generative model which has a D-dimensional latent space and a fixed latent probability distribution, defined by a random variable Z. We denote by X ∼ X that the random variable X has distribution X .

X n X represents the fact that the sequence of random variables {X n } n∈N converges weakly to a random variable with distribution X as n tends to infinity.

By X n X n we mean that lim n→∞ sup x∈R |CDF Xn (x) − CDF Xn (x)| = 0, where CDF X denotes the cumulative distribution function of X. The index n will usually be omitted for readability.

In other words, by X X we mean, informally, that X has distribution similar to X .

Property 2.1 (Linear Interpolation Invariance).

If Z defines a distribution on the D-dimensional latent space, Z(1) and Z (2) are independent and distributed identically to Z, and for every λ DISPLAYFORM0 is distributed identically to Z, then we will say that Z has the linear interpolation invariance property, or that linear interpolation does not change the distribution of Z.The most commonly used latent probability distributions Z are products of D independent random variables.

That is, Z = (Z 1 , Z 2 , . . . , Z D ), where Z 1 , Z 2 , . . . , Z D are the independent marginals distributed identically to Z. If the norms of Z concentrate around a certain value, then the latent distribution resembles sampling from a zero-centred sphere and the linear interpolation oversamples regions in the proximity of the origin of the latent space.

As a consequence, Z does not have the linear interpolation invariance property.

The following observation will shed light upon this problem.

Let N (µ, σ 2 ) denote the normal distribution with mean µ and variance σ 2 .Observation 2.1.

Let us assume that Z 2 has finite mean µ and finite variance σ DISPLAYFORM1 The proof of this and all further observations is presented in the appendix B.For example, if Z ∼ N (0, 1), then Z is distributed according to the D-dimensional normal distribution with mean 0 and identity covariance matrix I. Z 2 has moments µ = 1, D .

In that case, DISPLAYFORM2 DISPLAYFORM3 It is worth noting that the variance of the approximated probability distribution of Z , the thickness of the sphere, does not change as D tends to infinity -only the radius of the sphere is affected.

On the other hand, if the latent distribution is normalised (divided by the expected value of Z ), then the distribution concentrates around the unit sphere (not necessarily uniformly), and we observe the so-called soap bubble phenomenon BID2 .One might think that the factorisation of the latent probability distribution is the main reason why the linear interpolation changes the distribution.

Unfortunately, this is not the case.

Let DISPLAYFORM4 are two independent samples from Z. Therefore, Z is the distribution of the middle points of a linear interpolation between two vectors drawn independently from Z. Observation 2.2.

If Z has a finite mean, and Z is distributed identically to Z, then Z must be concentrated at a single point.

If a probability distribution is not heavy-tailed, then its tails are bounded by the exponential distribution, which in turn means that it has a finite mean.

Therefore, all distributions having undefined expected value must be heavy-tailed.

We will refer to this later on, as the heavy tails may have strong negative impact on the training procedure.

There have been attempts to find Z, with finite mean, such that Z is at least similar to Z. BID7 managed to reduce the distribution mismatch by defining the latent distribution as DISPLAYFORM5 where U(S D−1 ) is the uniform distribution on the unit sphere, and Γ( 1 2 , θ) is the gamma distribution.

We extend this idea by using a distribution that has no finite mean, namely the Cauchy distribution.

The standard Cauchy distribution is denoted by C(0, 1), and its density function is defined as 1/ π(1 + x 2 ) .

The most important property of the Cauchy distribution is the fact that if C (1) , . . .

, C (n) are independent samples from the standard Cauchy distribution, and DISPLAYFORM0 is also distributed according to the standard Cauchy distribution.

In case of n = 2 it means that the Cauchy distribution satisfies the distribution matching property.

On the other hand, as a consequence of observation 2.2, the Cauchy distribution cannot have finite mean.

In fact, all of its moments of order greater than or equal to one are undefined.

See Siegrist (2017) for further details.

There are two ways of using the Cauchy distribution in high dimensional spaces while retaining the distribution matching property.

The multidimensional Cauchy distribution is defined as a product of independent standard Cauchy distributions.

Then, the linear interpolation invariance property can be simply proved by applying the above formulas coordinate-wise.

In the case of vectors drawn from the multidimensional Cauchy distribution we may expect that some of the coordinates will be sufficiently larger, by absolute value, than the others BID5 , thus making the latent distribution similar to coordinate-wise sampling.

In contrast, the multivariate Cauchy distribution comes with the isotropy property at the cost of the canonical directions becoming statistically dependent.

There are multiple ways of defining it, and further analysis is out of the scope of this paper.

We tested both variants as latent distributions with similar results.

From now on, we shall concentrate on the non-isotropic Cauchy distribution.

The Cauchy distribution is a member of the family of stable distributions, and has been previously used to model heavy-tailed data BID11 .

However, according to our best knowledge, the Cauchy distribution has never been used as the latent distribution in generative models.

FIG1 presents a decoded linear interpolations between random latent vectors using a DCGAN model trained on the CelebA dataset for the Cauchy distribution and the distribution from BID7 .

It should be noted that if D is large enough, the distribution of the norms of vectors sampled from the D-dimensional Cauchy distribution has a low density near zero -similarly to the normal and uniform distributions -but linear interpolations do not oversample this part of the latent space, due to the heavy-tailed nature of the Cauchy distribution.

Comparison of the distributions of norms is given in FIG0 .

The distribution-interpolation trade off states that if the probability distribution has the linear interpolation invariance property, then it must be trivial or heavy-tailed.

In case of the Cauchy distribution we observed issues with generating images if the norm of the sampled latent vector was relatively large (the probability distribution of the norms is also heavy-tailed).

Some of those faulty examples are presented in the appendix C. This is consistent with the known fact, that artificial networks perform poorly if their inputs are not normalised (see e.g. BID3 ).A probability distribution having the linear interpolation invariance property cannot be normalised using linear transformations.

For example, the batch normalisation technique BID6 would be highly ineffective, as the mean of a batch of samples is, in fact, a single sample from the distribution.

On the other hand, using a non-linear normalisation (e.g., clipping the norm of the latent vectors in subsequent layers), is mostly equivalent to changing the latent probability distribution and making the interpolation non-linear.

This idea will be explored in the next section.

In this section we review the most popular variants of interpolations, with an emphasis on the distribution mismatch analysis.

We also present two new examples of interpolations stemming from a general scheme, that perform well with the popular latent priors.

An interpolation on the latent space R D is formally defined as a function DISPLAYFORM0 For brevity, we will represent f (x 1 , x 2 , λ) by f x1,x2 (λ).Property 3.1 (Distribution Matching Property).

If Z defines a distribution on the D-dimensional latent space, Z (1) and Z (2) are independent and distributed identically to Z, and for every λ ∈ [0, 1] the random variable f Z (1) ,Z (2) (λ) is distributed identically to Z, then we will say that the interpolation f has the distribution matching property in conjunction with Z, or that the interpolation f does not change the distribution of Z.

The linear interpolation is defined as f L x1,x2 (λ) = (1 − λ)x 1 + λx 2 .

This interpolation does not satisfy the distribution matching property for the most commonly used probability distributions, as they have a finite mean.

A notable exception is the Cauchy distribution.

This was discussed in details in the previous section.

As in BID13 ; BID15 , the spherical linear interpolation is defined as DISPLAYFORM0 where Ω is the angle between vectors x 1 and x 2 .

Note that this interpolation is undefined for parallel endpoint vectors, and the definition cannot be extended without losing the continuity.

Also, if vectors x 1 and x 2 have the same length R, then the interpolation corresponds to a geodesic on the sphere of radius R. In this regard, it might be said that the spherical linear interpolation is defined as the shortest path on the sphere.

The most important fact is that this interpolation can have the distribution matching property.

Observation 3.1.

If Z has uniform distribution on the zero-centred sphere of radius R > 0, then f SL does not change the distribution of Z.

Introduced in BID0 , the normalised interpolation is defined as DISPLAYFORM0 Observation 3.2.

If Z ∼ N (0, I), then f N does not change the distribution of Z.If vectors x 1 and x 2 are orthogonal and have equal length, then the curve defined by this interpolation is equal to the one of the spherical linear interpolation.

On the other hand, the normalised interpolation behaves poorly if x 1 is close to x 2 .

In the extreme case of x 1 = x 2 the interpolation is not constant with respect to λ, which violates any sensible definition of the shortest path.

Here we present a general way of designing interpolations that have the distribution matching property in conjunction with a given probability distribution Z. This method requires some additional assumptions about Z, but it works well with the most popular latent distributions.

Let L be the D-dimensional latent space, Z define the probability distribution on the latent space, C be distributed according to the D-dimensional Cauchy distribution on L, K be a subset of L such that Z is concentrated on this set, and g : L → K be a bijection such that g(C) is distributed identically to Z on K. Then for x 1 , x 2 ∈ K we define the Cauchy-linear interpolation as DISPLAYFORM0 In other words, for endpoints x 1 , x 2 ∼ Z:1.

Transform x 1 and x 2 using g −1 .

This step changes the latent distribution to the D-dimensional Cauchy distribution.

The transformed latent distribution remains unchanged.

Originally referred to as distribution matched.3.

Transform x λ back to the original space using g. We end up with the original latent distribution.

Observation 3.3.

With the above assumptions about g the Cauchy-linear interpolation does not change the distribution of Z.Finding an appropriate function g might seem hard, but in practice it usually is fairly straightforward.

For example, if Z is distributed identically to the product of D independent one-dimensional distributions Z, then we can define g as CDF −1 C • CDF Z applied to every coordinate.

We might want the interpolation to have some other desired properties.

For example, to behave exactly as the spherical linear interpolation if only the endpoints have equal norm.

For that purpose, we need to make additional assumptions.

Let Z be isotropic, C be distributed according to the onedimensional Cauchy distribution, and g : R → (0, +∞) be a bijection such that g(C) is distributed identically as Z on (0, +∞).

Then we can modify the spherical linear interpolation formula to define what we call the spherical Cauchy-linear interpolation DISPLAYFORM0 where Ω is the angle between vectors x 1 and x 2 .

In other words:1.

Interpolate the directions of latent vectors using the spherical linear interpolation.2.

Interpolate the norms using the Cauchy-linear interpolation.

Observation 3.4.

With the above assumptions about g, the spherical Cauchy-linear interpolation does not change the distribution of Z if the Z distribution is isotropic.

The simplest candidate for the g function is CDF −1 C • CDF Z , but we usually need to know more about Z to check if the assumptions hold.

For example, let Z be a D-dimensional normal distribution with zero mean and identity covariance matrix.

Then Z ∼ χ 2 D and DISPLAYFORM1 where Γ denotes the gamma function, and γ is the lower incomplete gamma function.

Thus we set FIG4 shows comparison of the Cauchy-linear and the spherical Cauchy-linear interpolations on a two-dimensional plane for pairs of vectors sampled from different probability distributions.

It illustrates how these interpolations manage to keep the distributions unchanged.

FIG5 is an illustration of distribution matching property for Cauchy-linear interpolation.

We also compare the data samples generated by the DCGAN model trained on the CelebA dataset; the results are shown in figure 5.

DISPLAYFORM2

We investigated the properties of multidimensional probability distributions in the context of generative models.

We found out that there is a certain trade-off: it is impossible to define a latent probability distribution with a finite mean and the linear interpolation invariance property.

The D-dimensional Cauchy distribution serves as an example of a latent probability distribution that remains unchanged by linear interpolation, at the cost of poor model performance, due to the heavytailed nature.

Instead of using the Cauchy distribution as the latent distribution, we propose to use it to define nonlinear interpolations that have the distribution matching property.

The assumption of the shortest path being a straight line must be relaxed, but our scheme is general enough to provide a way of incorporating other desirable properties.

We observe that there are three different goals when using interpolations for studying a generative model.

Firstly, to check whether the training objective was fulfilled, one must use an interpolation that does not cause the distribution mismatch.

This is, in our opinion, a necessary step before performing any further evaluation of the trained model.

Secondly, if one is interested in the manifold convexity, linear interpolations are a suitable method provided the above analysis yields positive results.

Finally, to perform a complete investigation of the learned manifold one can employ methods that incorporate some information about the trained model, e.g. the approach of BID1 mentioned in section 1.1.We do not propose to completely abandon the use of linear interpolations, as the convexity of the learned manifold is still an interesting research topic.

For instance, we have observed that generative models are capable of generating sensible images from seemingly out-of-distribution regions, e.g. the emergence of the median face mentioned in the introduction.

In our opinion, this is a promising direction for future research.

All experiments were conducted using a DCGAN model BID12 , in which the generator network consisted of a linear layer with 8192 neurons, followed by four convolution transposition layers, each using 5 × 5 filters and strides of 2, with number of filters in order of layers: 256, 128, 64, 3.

Except for the output layer, where tanh activation function was used, all previous layers used ReLU.

Discriminator's architecture mirrored the one from the generator, with a single exception of using leaky ReLU instead of vanilla ReLU function for all except the last layer.

No batch normalisation was used in both networks.

Adam optimiser with learning rate of 2e −4 and momentum set to 0.5 was used.

Batch size 64 was used throughout all experiments.

If not explicitly stated otherwise, latent space dimension was set to 100.

For the CelebA dataset we resized the input images to 64 × 64.

Observation 2.1.

Let us assume that Z 2 has finite mean µ and finite variance σ DISPLAYFORM0 Proof.

Recall that Z, Z 1 , . . .

, Z D are independent and identically distributed.

Therefore DISPLAYFORM1 D are also independent and identically distributed.

Z = (Z 1 , . . . , Z D ) and DISPLAYFORM2 almost everywhere, Z = 0 almost everywhere, and finally Z = 0 almost everywhere.

From now on we will assume that µ > 0.Using the central limit theorem we know that DISPLAYFORM3 The convergence of cumulative distribution functions is uniform, because the limit is continuous everywhere DISPLAYFORM4 Additionally, DISPLAYFORM5 and now we have DISPLAYFORM6 Finally, the function DISPLAYFORM7 is a bijection (again, because D > 0), so we may substitute Dµ + x √ D with x and the innermost statement will hold for every x ∈ R DISPLAYFORM8 Before taking square root of the normal distribution we must deal with negative values.

Let N + (ν, τ ) be defined by its cumulative distribution function: DISPLAYFORM9 The idea is to take all negative values of N (ν, τ ) and concentrate them at zero.

Now we can modify (1) DISPLAYFORM10 for x ≥ 0 we simply use (1), for x < 0 the inequality simplifies to |0 − 0| < .Since Z 2 and N + (Dµ, Dσ 2 ) are non-negative, we are allowed to take the square root of these random variables.

The square root is a strictly increasing function, thus for x ≥ 0 we have DISPLAYFORM11 therefore we can approximate the variable Z DISPLAYFORM12 for x ≥ 0 we substitute x 2 for x in (2), for x < 0 the inequality simplifies, again, to |0 − 0| < .This paragraph is a summary of the second part of the proof.

To calculate N + (Dµ, Dσ 2 ) we observe that, informally, in proximity of Dµ the square root behaves approximately like scaling with constant (2 √ Dµ) −1 .

Additionally, N (Dµ, Dσ 2 ) has width proportional to √ D, which is infinitesimally smaller than Dµ, so we expect the result to be DISPLAYFORM13 Let us define DISPLAYFORM14 Here b is defined so that the probability of x drawn from N √ Dµ, σ 2 4µ being at least b far from the mean is equal to 2 .

Also, note that b does not depend on D. For now we will assume that √ Dµ − b > 0 -this is always true for sufficiently large D, as µ > 0 DISPLAYFORM15 Now let us assume that we have a fixed > 0.

For x ∈ [−b , b ] we write the following inequalities DISPLAYFORM16 which are equivalent to 0 ≤ x 2 ≤ b 2 , thus true.

Every cumulative distribution function is weakly increasing, therefore DISPLAYFORM17 Because we assumed that DISPLAYFORM18 We transform the outer distributions using basic properties of the normal distribution.

We also take square root of the middle distribution and obtain DISPLAYFORM19 is continuous, thus we have uniform convergence DISPLAYFORM20 Using (5) we get DISPLAYFORM21 Now we will extend this result to all x ∈ R. For > 0 we have DISPLAYFORM22 DISPLAYFORM23 Substituting −b and b for x in (6), and using FORMULA37 and FORMULA38 respectively, we obtain DISPLAYFORM24 DISPLAYFORM25 Cumulative distribution functions are increasing functions with values in [0, 1], thus combining FORMULA37 and (9) DISPLAYFORM26 Analogically, using (8) and (10) DISPLAYFORM27 Thus, DISPLAYFORM28 because for any > 0 we may define DISPLAYFORM29 are taken from (6), FORMULA23 and (12).To simplify, DISPLAYFORM30 because for any > 0 we may define D := max{D 1 , D 2 }, where D 1 , D 2 are taken from (4) and (13), making the antecedent true.

We also replaced √ Dµ + x with x, since now the statement holds for all x ∈ R.Finally, we combine (3) and (14) using the triangle inequality DISPLAYFORM31 because for any > 0 we may define D := max{D 1 , D 2 }, where D 1 , D 2 are taken from FORMULA27 and FORMULA23 , and since it is true for any positive , we replace 3 with DISPLAYFORM32 because for any > 0 we may define D := D 1 , where D 1 is taken from (15), substituting 3 for .Observation 2.2.

If Z has a finite mean, and Z is distributed identically to Z, then Z must be concentrated at a single point.

DISPLAYFORM33 . .

be an infinite sequence of independent and identically distributed random variables.

Using induction on n we can show that DISPLAYFORM34 is distributed identically to Z. Indeed, for n = 1 this is one of the theorem's assumptions.

To prove the inductive step let us define DISPLAYFORM35 A and B are independent -they are defined as functions of independent variables -and, by the inductive hypothesis, distributed identically to Z. Finally, it is sufficient to observe that DISPLAYFORM36 Z has finite mean -let us denote it by µ.

Let also N + be the set of strictly positive natural numbers.

By the law of large numbers the sequence { 1 n (Z (1) + . . .

+ Z (n) )} n∈N+ converges in probability to µ. The same is true for any infinite subsequence, in particular for { 1 2 n (Z (1) + . . .

+ Z (2 n ) )} n∈N+ , but we have shown that all elements of this subsequence are distributed identically to Z, thus Z must be concentrated at µ.Observation 3.1.

If Z has uniform distribution on the zero-centred sphere of radius R > 0, then f SL does not change the distribution of Z.Proof.

Let Z, Z (1) , Z (2) be independent and identically distributed.

Let λ ∈ [0, 1] be a fixed real number.

The random variable f SL Z (1) ,Z (2) (λ) is defined almost everywhere (with the exception of parallel samples from Z (1) , Z (2) ) and is also concentrated on the zero-centred sphere of radius R (because if x 1 = x 2 , then f SL x1,x2 (λ) = x 1 = x 2 ).

Let iso be any linear isometry of the latent space.

iso(x) = x , thus iso is also an isometry of the zero-centred sphere of radius R. Additionally, we have DISPLAYFORM37 ,iso(x2) (λ) and the last equality holds because the isometry does not change the angle Ω between x 1 and x 2 .

DISPLAYFORM38 (λ), and this is distributed identically to f DISPLAYFORM39 , both uniform distributions, are invariant to iso.

In that case, f SL Z (1) ,Z (2) (λ) is concentrated on the zero-centred sphere of radius R and invariant to all linear isometries of the latent space.

The only distribution having these properties is the uniform distribution on the sphere.

Proof.

Let Z, Z (1) , Z (2) be independent and identically distributed.

Let λ ∈ [0, 1] be a fixed real number.

The random variables Z(1) and Z (2) are both distributed according to N (0, I).

Using the definition of f N and elementary properties of the normal distribution we conclude DISPLAYFORM40 Observation 3.3.

With the above assumptions about g the Cauchy-linear interpolation does not change the distribution of Z.Proof.

Let Z, Z (1) , Z (2) be independent and identically distributed.

Let λ ∈ [0, 1] be a fixed real number.

First observe that g −1 (Z (1) ) and g −1 (Z (2) ) are independent (because Z(1) , Z (2) are independent) and distributed identically to C (property of g).

Likewise, (1 − λ)g −1 (Z (1) ) + λg −1 (Z (2) ) ∼ C (property of the Cauchy distribution).

Therefore, g((1 − λ)g −1 (Z (1) ) + λg −1 (Z (1) )) ∼ Z (property of g).Observation 3.4.

With the above assumptions about g, the spherical Cauchy-linear interpolation does not change the distribution of Z if the Z distribution is isotropic.

Proof.

Let Z, Z (1) , Z (2) be independent and identically distributed.

Let λ ∈ [0, 1] be a fixed real number.

The following statements are straightforward consequences of Z (1) , Z (2) being isotropic (and also independent).

(1) DISPLAYFORM0 Z (2) , Z (1) , Z (2) are independent, 2.

Z (1) and Z (2) are both distributed identically to Z , 3.

Z

<|TLDR|>

@highlight

We theoretically prove that linear interpolations are unsuitable for analysis of trained implicit generative models. 

@highlight

Studies the problem of when the linear interpolant between two random variables follows the same distribution, related to prior distribution of an implicit generative model

@highlight

This work asks how to interpolate in the latent space given a latent variable model.
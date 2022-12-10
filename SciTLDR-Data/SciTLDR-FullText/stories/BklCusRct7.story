Generative models such as Variational Auto Encoders (VAEs) and Generative Adversarial Networks (GANs) are typically trained for a fixed prior distribution in the latent space, such as uniform or Gaussian.

After a trained model is obtained, one can sample the Generator in various forms for exploration and understanding, such as interpolating between two samples, sampling in the vicinity of a sample or exploring differences between a pair of samples applied to a third sample.

However, the latent space operations commonly used in the literature so far induce a distribution mismatch between the resulting outputs and the prior distribution the model was trained on.

Previous works have attempted to reduce this mismatch with heuristic modification to the operations or by changing the latent distribution and re-training models.

In this paper, we propose a framework for modifying the latent space operations such that the distribution mismatch is fully eliminated.

Our approach is based on optimal transport maps, which adapt the latent space operations such that they fully match the prior distribution, while minimally modifying the original operation.

Our matched operations are readily obtained for the commonly used operations and distributions and require no adjustment to the training procedure.

Generative models such as Variational Autoencoders (VAEs) BID7 and Generative Adversarial Networks (GANs) BID3 have emerged as popular techniques for unsupervised learning of intractable distributions.

In the framework of Generative Adversarial Networks (GANs) BID3 , the generative model is obtained by jointly training a generator G and a discriminator D in an adversarial manner.

The discriminator is trained to classify synthetic samples from real ones, whereas the generator is trained to map samples drawn from a fixed prior distribution to synthetic examples which fool the discriminator.

Variational Autoencoders (VAEs) BID7 are also trained for a fixed prior distribution, but this is done through the loss of an Autoencoder that minimizes the variational lower bound of the data likelihood.

For both VAEs and GANs, using some data X we end up with a trained generator G, that is supposed to map latent samples z from the fixed prior distribution to output samples G(z) which (hopefully) have the same distribution as the data.

In order to understand and visualize the learned model G(z), it is a common practice in the literature of generative models to explore how the output G(z) behaves under various arithmetic operations on the latent samples z. However, the operations typically used so far, such as linear interpolation BID3 , spherical interpolation BID20 , vicinity sampling and vector arithmetic BID12 , cause a distribution mismatch between the latent prior distribution and the results of the operations.

This is problematic, since the generator G was trained on a fixed prior and expects to see inputs with statistics consistent with that distribution.

To address this, we propose to use distribution matching transport maps, to obtain analogous latent space operations (e.g. interpolation, vicinity sampling) which preserve the prior distribution of samples from prior linear matched (ours) spherical (a) Uniform prior: Trajectories of linear interpolation, our matched interpolation and the spherical interp.

BID20 .

(e) Spherical midpoint distribution BID20 Figure 1: We show examples of distribution mismatches induced by the previous interpolation schemes when using a uniform prior in two dimensions.

Our matched interpolation avoids this with a minimal modification to the linear trajectory, traversing through the space such that all points along the path are distributed identically to the prior.

the latent space, while minimally changing the original operation.

In Figure 1 we showcase how our proposed technique gives an interpolation operator which avoids distribution mismatch when interpolating between samples of a uniform distribution.

The points of the (red) matched trajectories are obtained as minimal deviations (in expectation of l 1 distance) from the the points of the (blue) linear trajectory.

In the literature there are dozens of papers that use sample operations to explore the learned models BID0 ; BID3 ; BID2 ; BID13 ; BID1 ; BID13 to name a few), but most of them have ignored the problem of distribution mismatch.

BID7 and BID10 sidestep the problem when visualizing their models, by not performing operations on latent samples, but instead restrict the latent space to 2-d and uniformly sample the percentiles of the distribution on a 2-d grid.

This way, the samples have statistics that are consistent with the prior distribution.

However, this approach does not scale up to higher dimensions -whereas the latent spaces used in the literature can have hundreds of dimensions.

BID20 experimentally observe that there is a distribution mismatch between the norm for points drawn from uniform or Gaussian distribution and points obtained with linear interpolation (SLERP), and (heuristically) propose to use a so-called spherical linear interpolation to reduce the mismatch, obtaining higher quality interpolated samples.

While SLERP has been subjectively observed to produce better looking samples than linear interpolation and is now commonly, its heuristic nature has limited it from fully replacing the linear interpolation.

Furthermore, while perhaps possible it is not obvious how to generalize it to other operations, such as vicinity sampling, n-point interpolation and random walk.

In Section 2 we show that for interpolation, in high dimensions SLERP tends to approximately perform distribution matching the approach taken by our framework which can explain why it works well in practice.

BID6 further analyze the (norm) distribution mismatch observed by BID20 (in terms of KL-Divergence) for the special case of Gaussian priors, and propose an alternative prior distribution with dependent components which produces less (but still nonzero) distribution mismatch for linear interpolation, at the cost of needing to re-train and re-tune the generative models.

In contrast, we propose a framework which allows one to adapt generic operations, such that they fully preserve the original prior distribution while being faithful to the original operation.

Thus the KL-Divergence between the prior and the distribution of the results from our operations is zero.

The approach works as follows: we are given a 'desired' operation, such as linear interpolation y = tz 1 + (1 − t)z 2 , t ∈ [0, 1].

Since the distribution of y does not match the prior distribution of z, we search for a warping f : DISPLAYFORM0 has the same distribution as z. In order to have the modificationỹ as faithful as possible to the original operation y, we use optimal transform Published as a conference paper at ICLR 2019 Operation Expression 2-point interpolation maps BID17 BID18 BID19 to find a minimal modification of y which recovers the prior distribution z. Figure 1a , where each pointỹ of the matched curve is obtained by warping a corresponding point y of the linear trajectory, while not deviating too far from the line.

DISPLAYFORM1

With implicit models such as GANs BID3 and VAEs BID7 , we use the data X , drawn from an unknown random variable x, to learn a generator G : DISPLAYFORM0 with respect to a fixed prior distribution p z , such that G(z) approximates x. Once the model is trained, we can sample from it by feeding latent samples z through G.We now bring our attention to operations on latent samples DISPLAYFORM1 We give a few examples of such operations in TAB0 .Since the inputs to the operations are random variables, their output y = κ(z 1 , · · · , z k ) is also a random variable (commonly referred to as a statistic).

While we typically perform these operations on realized (i.e. observed) samples, our analysis is done through the underlying random variable y. The same treatment is typically used to analyze other statistics over random variables, such as the sample mean, sample variance and test statistics.

In TAB0 we show example operations which have been commonly used in the literature.

As discussed in the Introduction, such operations can provide valuable insight into how the trained generator G changes as one creates related samples y from some source samples.

The most common such operation is the linear interpolation, which we can view as an operation DISPLAYFORM2 where z 1 , z 2 are latent samples from the prior p z and y t is parameterized by t ∈ [0, 1].

Now, assume z 1 and z 2 are i.i.d, and let Z 1 , Z 2 be their (scalar) first components with distribution p Z .

Then the first component of y t is Y t = tZ 1 + (1 − t)Z 2 , and we can compute: DISPLAYFORM3 Since (1 + 2t(t − 1)) = 1 for all t ∈ [0, 1] \ {0, 1}, it is in general impossible for y t to have the same distribution as z, which means that distribution mismatch is inevitable when using linear interpolation.

A similar analysis reveals the same for all of the operations in TAB0 .This leaves us with a dilemma: we have various intuitive operations (see TAB0 ) which we would want to be able to perform on samples, but their resulting distribution p yt is inconsistent with the distribution p z we trained G for.

Due to the curse of dimensionality, as empirically observed by BID20 , this mismatch can be significant in high dimensions.

We illustrate this in FIG1 , where we plot the distribution of the squared norm y t 2 for the midpoint t = 1/2 of linear interpolation, compared to the prior distribution z 2 .

With d = 100 (a typical dimensionality for the latent space), the distributions are dramatically different, having almost no common support.

BID6 quantify this mismatch for Gaussian priors in terms of KL-Divergence, and show that it grows linearly with the dimension d. In Appendix A (see Supplement) we expand this analysis and show that this happens for all prior distributions with i.i.d.

entries (i.e. not only Gaussian), both in terms of geometry and KL-Divergence.

In order to address the distribution mismatch, we propose a simple and intuitive framework for constructing distribution preserving operators, via optimal transport: Published as a conference paper at ICLR 2019 BID20 .

Both linear and spherical interpolation introduce a distribution mismatch, whereas our proposed matched interpolation preserves the prior distribution for both priors.

Strategy 1 (Optimal Transport Matched Operations).

DISPLAYFORM0 2.

We analytically (or numerically) compute the resulting (mismatched) distribution p y 3.

We search for a minimal modificationỹ = f (y) (in the sense that E y [c(ỹ, y)] is minimal with respect to a cost c), such that distribution is brought back to the prior, i.e. pỹ = p z .The cost function in step 3 could e.g. be the euclidean distance c(x, y) = x − y , and is used to measure how faithful the modified operator DISPLAYFORM1 Finding the map f which gives a minimal modification can be challenging, but fortunately it is a well studied problem from optimal transport theory.

We refer to the modified operationỹ as the matched version of y, with respect to the cost c and prior distribution p z .For completeness, we introduce the key concepts of optimal transport theory in a simplified setting, i.e. assuming probability distributions are in euclidean space and skipping measure theoretical formalism.

We refer to BID18 BID19 and BID17 for a thorough and formal treatment of optimal transport.

The problem of step (3) above was first posed by Monge (1781) and can more formally be stated as: Problem 1 (Santambrogio (2015) Problem 1.1).

Given probability distributions p x , p y , with domains X , Y respectively, and a cost function c : X × Y → R + , we want to minimize DISPLAYFORM2 We refer to the minimizer f * X → Y of (MP) (if it exists), as the optimal transport map from p x to p y with respect to the cost c.

However, the problem remained unsolved until a relaxed problem was studied by BID5 :

Problem 2 (Santambrogio (2015) Problem 1.2).

Given probability distributions p x , p y , with domains X , Y respectively, and a cost function c : X × Y → R + , we want to minimize DISPLAYFORM3 where (x, y) ∼ p x,y , x ∼ p x , y ∼ p y denotes that (x, y) have a joint distribution p x,y which has (previously specified) marginals p x and p y .We refer to the joint p x,y which minimizes (KP) as the optimal transport plan from p x to p y with respect to the cost c.

The key difference is to relax the deterministic relationship between x and f (x) to a joint probability distribution p x,y with marginals p x and p y for x and y. In the case of Problem 1, the minimization Published as a conference paper at ICLR 2019 might be over the empty set since it is not guaranteed that there exists a mapping f such that f (x) ∼ y. In contrast, for Problem 2, one can always construct a joint density p x,y with p x and p y as marginals, such as the trivial construction where x and y are independent, i.e. p x,y (x, y) := p x (x)p y (y).Note that given a joint density p x,y (x, y) over X × Y, we can view y conditioned on x = x for a fixed x as a stochastic function f (x) from X to Y, since given a fixed x do not get a specific function value f (x) but instead a random variable f (x) that depends on x, with f (x) ∼ y|x = x with density p y (y|x = x) := px,y(x,y)px (x) .

In this case we have (x, f (x)) ∼ p x,y , so we can view the Problem KP as a relaxation of Problem MP where f is allowed to be a stochastic mapping.

While the relaxed problem of Kantorovich (KP) is much more studied in the optimal transport literature, for our purposes of constructing operators it is desirable for the mapping f to be deterministic as in (MP) (see Appendix C for a more detailed discussion on deterministic vs stochastic operations).To this end, we will choose the cost function c such that the two problems coincide and where we can find an analytical solution f or at least an efficient numerical solution.

In particular, we note that the operators in TAB0 are all pointwise, such that if the points z i have i.i.d.

components, then the result y will also have i.i.d.

components.

If we combine this with the constraint for the cost c to be additive over the components of x, y, we obtain the following simplification: Theorem 1.

Suppose p x and p y have i.i.d components and c over DISPLAYFORM4 Consequently, the minimization problems (MP) and (KP) turn into d identical scalar problems for the distributions p X and p Y of the components of x and y: DISPLAYFORM5 such that an optimal transport map T for (MP-1-D) gives an optimal transport map f for (MP) by pointwise application of T , i.e. f (x) (i) := T (x (i) ), and an optimal transport plan p X,Y for (KP-1-D)gives an optimal transport plan p x,y (x, y) : DISPLAYFORM6 Proof.

See Appendix.

Fortunately, under some mild constraints, the scalar problems have a known solution: Theorem 2 (Theorem 2.9 in Santambrogio FORMULA3 ).

Let h : R → R + be convex and suppose the cost C takes the form C(x, y) = h(x − y).

Given an continuous source distribution p X and a target distribution p Y on R having a finite optimal transport cost in (KP-1-D), then DISPLAYFORM7 defines an optimal transport map from p X to p Y for (MP-1-D), where DISPLAYFORM8 is the Cumulative Distribution Function (CDF) of X and F DISPLAYFORM9 ≥ y} is the pseudo-inverse of F Y .

Furthermore, the joint distribution of (X, T mon X→Y (X)) defines an optimal transport plan for (KP-1-D).The mapping T mon X→Y (x) in Theorem 2 is non-decreasing and is known as the monotone transport map from X to Y .

It is easy to verify that T mon X→Y (X) has the distribution of Y , in particular DISPLAYFORM10 Now, combining Theorems 1 and 2, we obtain a concrete realization of the Strategy 1 outlined above.

We choose the cost c such that it admits to Theorem 1, such as c(x, y) := x − y 1 , and use an operation that is pointwise, so we just need to compute the monotone transport map in (5).

That is, if DISPLAYFORM11 0.8 1 ỹ y t = 0.05 t = 0.25 t = 0.5 −3 −2.5 −2 −1.5 −1 −0.5 0 0.5 1 1.5 2 2.5 3 −3 −2 −1 0 1 2 3 ỹ y t = 0.05 t = 0.25 t = 0.5 (a) Uniform prior (b) Gaussian prioras the component-wise modification of y, i.e.ỹ DISPLAYFORM12 In FIG3 we show the monotone transport map for the linear interpolation y = tz 1 + (1 − t)z 2 for various values of t. The detailed calculations and examples for various operations are given in Appendix B, for both Uniform and Gaussian priors.

To validate the correctness of the matched operators computed in Appendix B, we numerically simulate the distributions for toy examples, as well as prior distributions typically used in the literature.

Priors vs. interpolations in 2-D For Figure 1 , we sample 1 million pairs of points in two dimension, from a uniform prior (on [−1, 1] 2 ), and estimate numerically the midpoint distribution of linear interpolation, our proposed matched interpolation and the spherical interpolation of BID20 .

It is reassuring to see that the matched interpolation gives midpoints which are identically distributed to the prior.

In contrast, the linear interpolation condenses more towards the origin, forming a pyramidshaped distribution (the result of convolving two boxes in 2-d).

Since the spherical interpolation of BID20 follows a great circle with varying radius between the two points, we see that the resulting distribution has a "hole" in it, "circling" around the origin for both priors.

FIG1 , we sample 1 million pairs of points in d = 100 dimensions, using either i.i.d.

uniform components on [−1, 1] or Gaussian N (0, 1) and compute the distribution of the squared norm of the midpoints.

We see there is a dramatic difference between vector lengths in the prior and the midpoints of linear interpolation, with only minimal overlap.

We also see that the spherical interpolation (SLERP) is approximately matching the prior (norm) distribution, having a matching first moment, but otherwise also induces a distribution mismatch.

In contrast, our matched interpolation, fully preserves the prior distribution and perfectly aligns.

We note that this setting (d = 100, uniform or Gaussian) is commonly used in the literature.

Setup We used DCGAN BID12 generative models trained on LSUN bedrooms BID21 , CelebA BID8 and LLD BID14 , an icon dataset, to qualitatively evaluate.

For LSUN, the model was trained for two different output resolutions, providing 64 × 64 pixel and a 128 × 128 pixel output images (where the latter is used in figures containing larger sample images).

The models for LSUN and the icon dataset where both trained on a uniform latent prior distribution, while for CelebA a Gaussian prior was used.

The dimensionality of the latent space is 100 for both LSUN and CelebA, and 512 for the model trained on the icon model.

Furthermore we use improved Wasserstein GAN (iWGAN) with gradient penalty (Gulrajani et Table 3 : We measure over the average (normalized) perturbation ỹ −

y p / y p incurred by our matched interpolation for the latent spaces used in TAB2 , for p = 1, 2.2017) trained on CIFAR-10 at 32 × 32 pixels with a 128-dimensional Gaussian prior to compute inception scores.

To measure the effect of the distribution mismatch, we quantitatively evaluate using the Inception score BID16 .

In TAB2 we compare the Inception score of our trained models (i.e. using random samples from the prior) with the score when sampling midpoints from the 2-point and 4-point interpolations described above, reporting mean and standard deviation with 50,000 samples, as well as relative change to the original model scores if they are significant.

Compared to the original scores of the trained models (random samples), our matched operations are statistically indistinguishable (as expected) while the linear interpolation gives a significantly lower score in all settings (up to 29% lower).However, this is not surprising, since our matched operations are guaranteed to produce samples that come from the same distribution as the random samples.

To quantify the effect our matching procedure has on the original operation, in Table 3 we compute the perturbation incurred when warping the linear interpolation y to the matched counterpartỹ for 2-point interpolation on the latent spaces used in TAB2 .

We compute the normalized perturbation ỹ t − y t p /

y t p (with p = 1 corresponding to l 1 distance and p = 2 to l 2 distance), over N = 100000 interpolation points y t = tz 1 + (1 − t)z 2 where z 1 , z 2 are sampled from the prior and t ∈ [0, 1] sampled uniformly.

We observe that for all priors and both metrics, the perturbation is in the range 0.23 − 0.25, i.e. less than a one fourth of y t .

In the following, we will qualitatively show that our matched operations behave as expected, and that there is a visual difference between the original operations and the matched counterparts.

To this end, the generator output for latent samples produced with linear interpolation, SLERP (spherical linear interpolation) of BID20 and our proposed matched interpolation will be compared.

We begin with the classic example of 2-point interpolation: Figure 4 shows three examples per dataset for an interpolation between 2 points in latent space.

Each example is first done via linear interpolation, then SLERP and finally matched interpolation.

It is immediately obvious in Figures 4a and 4b that linear interpolation produces inferior results with generally more blurry, less saturated and less detailed output images.

The SLERP heuristic and matched interpolation are slightly different visually, but we do not observe a difference in visual quality.

However, we stress that the goal of this work is to construct operations in a principled manner, whose samples are consistent with the generative model.

In the case of linear Published as a conference paper at ICLR 2019 interpolation (our framework generalizes to more operations, see below and Appendix), the SLERP heuristic tends to work well in practice but we provide a principled alternative.

4-point interpolation An even stronger effect can be observed when we do 4-point interpolation, showcased in Figure 5 (LSUN) and Figure 8 (LLD icons).

The higher resolution of the LSUN output highlights the very apparent loss of detail and increasing prevalence of artifacts towards the midpoint in the linear version, compared to SLERP and our matched interpolation.

Midpoints (Appendix) In all cases, the point where the interpolation methods diverge the most, is at the midpoint of the interpolation where t = 0.5.

Thus we provide 25 such interpolation midpoints in Figures 11 (LLD icons) and 12 (LSUN) in the Appendix for direct comparison.

Vicinity sampling (Appendix) Furthermore we provide two examples for vicinity sampling in Figures 9 and 10 in the Appendix.

Analogous to the previous observations, the output under a linear operator lacks definition, sharpness and saturation when compared to both spherical and matched operators.

Random walk An interesting property of our matched vicinity sampling is that we can obtain a random walk in the latent space by applying it repeatedly: we start at a point y 0 = z drawn from the prior, and then obtain point y i by sampling a single point in the vicinity of y i−1 , using some fixed 'step size' .

We show an example of such a walk in FIG5 , using = 0.5.

As a result of the repeated application of the vicinity sampling operation, the divergence from the prior distribution in the non-matched case becomes stronger with each step, resulting in completely unrecognizable output images on the LSUN and LLD icon models.

We proposed a framework that fully eliminates the distribution mismatch in the common latent space operations used for generative models.

Our approach uses optimal transport to minimally modify (in l 1 distance) the operations such that they fully preserve the prior distribution.

We give analytical formulas of the resulting (matched) operations for various examples, which are easily implemented.

The matched operators give a significantly higher quality samples compared to the originals, having the potential to become standard tools for evaluating and exploring generative models.

This work was partly supported by ETH Zurich General Fund (OK) and Nvidia through a hardware grant.

Published as a conference paper at ICLR 2019

We note that the analysis here can bee seen as a more rigorous version of an observation made by BID20 , who experimentally show that there is a significant difference between the average norm of the midpoint of linear interpolation and the points of the prior, for uniform and Gaussian distributions.

Suppose our latent space has a prior with DISPLAYFORM0 In this case, we can look at the squared norm DISPLAYFORM1 From the Central Limit Theorem (CLT), we know that as d → ∞, DISPLAYFORM2 in distribution.

Thus, assuming d is large enough such that we are close to convergence, we can approximate the distribution of z 2 as N (dµ Z 2 , dσ 2 Z 2 ).

In particular, this implies that almost all points lie on a relatively thin spherical shell, since the mean grows as O(d) whereas the standard deviation grows only as O( DISPLAYFORM3 We note that this property is well known for i.i.d Gaussian entries (see e.g. Ex.

6.14 in MacKay FORMULA4 ).

For Uniform distribution on the hypercube it is also well known that the mass is concentrated in the corner points (which is consistent with the claim here since the corner points lie on a sphere).Now consider an operator such as the midpoint of linear interpolation, y = In this case, we can compute: DISPLAYFORM4 Thus, the distribution of y 2 can be approximated with N ( DISPLAYFORM5 .

Therefore, y also mostly lies on a spherical shell, but with a different radius than z. In fact, the shells will intersect at regions which have a vanishing probability for large d. In other words, when looking at the squared norm y 2 , y 2 is a (strong) outlier with respect to the distribution of z 2 .This can be quantified in terms of KL-Divergence: DISPLAYFORM6 so D KL ( z 2 , y 2 ) grows linearly with the dimensions d.

Proof.

We will show it for the Kantorovich problem, the Monge version is similar.

Published as a conference paper at ICLR 2019Starting from (KP), we compute DISPLAYFORM0 where the inequality in (17) is due to each term being minimized separately.

DISPLAYFORM1 where p X,Y has marginals p X and p Y .

In this case P d (X, Y ) is a subset of all joints p x,y with marginals p x and p y , where the pairs ( DISPLAYFORM2 where the inequality in (21) is due to minimizing over a smaller set.

Since the two inequalities above are in the opposite direction, equality must hold for all of the expressions above, in particular: DISPLAYFORM3 Thus, (KP) and (KP-1-D) equal up to a constant, and minimizing one will minimize the other.

Therefore the minimization of the former can be done over p X,Y with p x,y (x, y) = DISPLAYFORM4

In the next sections, we illustrate how to compute the matched operations for a few examples, in particular for linear interpolation and vicinity sampling, using a uniform or a Gaussian prior.

We picked the examples where we can analytically compute the uniform transport map, but note that it is also easy to compute F DISPLAYFORM0 and (F Y (y)) numerically, since one only needs to estimate CDFs in one dimension.

Since the components of all random variables in these examples are i.i.d, for such a random vector x we will implicitly write X for a scalar random variable that has the distribution of the components of x.

When computing the monotone transport map T mon X→Y , the following Lemma is helpful.

Lemma 1 (Theorem 2.5 in BID17 ).

Suppose a mapping g(x) is non-decreasing and maps a continuous distribution p X to a distribution p Y , i.e. DISPLAYFORM1 then g is the monotone transport map T mon X→Y .According to Lemma 1, an alternative way of computing T mon X→Y is to find some g that is nondecreasing and transforms p X to p Y .

Suppose z has uniform components Z ∼ Uniform(−1, 1).

In this case, p Z (z) = 1/2 for −1 < z < 1.

Now let y t = tz 1 + (1 − t)z 2 denote the linear interpolation between two points z 1 , z 2 , with component distribution p Yt .

Due to symmetry we can assume that t > 1/2, since p Yt = p Y1−t .

We then obtain p Yt as the convolution of p tZ and p (1−t)Z , i.e. p Yt = p tZ * p (1−t)Z .

First we note that p tZ = 1/(2t) for −t < z < t and p (1−t)Z = 1/(2(1 − t)) for −(1 − t) < z < 1 − t. We can then compute:p Yt (y) = (p tZ * p (1−t)Z )(y)= 1 2(1 − t)(2t) DISPLAYFORM0 if y < −1 y + 1 if − 1 < y < −t + (1 − t) 2 − 2t if − t + (1 − t) < y < t − (1 − t) −y + 1 if t − (1 − t) < y < 1 0 if 1 < yThe CDF F Yt is then obtained by computing if 1 − 2t < y < 2t − 1 2(1 − t)(3t − 1) + (− 1 2 y 2 + y + 1 2 (2t − 1) 2 − (2t − 1)) if 2t − 1 < y < 1 2(1 − t)(2t) if 1 < ySince p Z (z) = 1/2 for |z| < 1, we have F Z (z) =

@highlight

We propose a framework for modifying the latent space operations such that the distribution mismatch between the resulting outputs and the prior distribution the generative model was trained on is fully eliminated.
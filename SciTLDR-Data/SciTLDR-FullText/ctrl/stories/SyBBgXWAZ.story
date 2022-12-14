Generative models such as Variational Auto Encoders (VAEs) and Generative Adversarial Networks (GANs) are typically trained for a fixed prior distribution in the latent space, such as uniform or Gaussian.

After a trained model is obtained, one can sample the Generator in various forms for exploration and understanding, such as interpolating between two samples, sampling in the vicinity of a sample or exploring differences between a pair of samples applied to a third sample.

In this paper, we show that the latent space operations used in the literature so far induce a distribution mismatch between the resulting outputs and the prior distribution the model was trained on.

To address this, we propose to use distribution matching transport maps to ensure that such  latent space operations preserve the prior distribution, while minimally modifying the original operation.

Our experimental results validate that the proposed operations give higher quality samples compared to the original operations.

Generative models such as Variational Autoencoders (VAEs) BID6 and Generative Adversarial Networks (GANs) BID3 have emerged as popular techniques for unsupervised learning of intractable distributions.

In the framework of Generative Adversarial Networks (GANs) BID3 , the generative model is obtained by jointly training a generator G and a discriminator D in an adversarial manner.

The discriminator is trained to classify synthetic samples from real ones, whereas the generator is trained to map samples drawn from a fixed prior distribution to synthetic examples which fool the discriminator.

Variational Autoencoders (VAEs) BID6 are also trained for a fixed prior distribution, but this is done through the loss of an Autoencoder that minimizes the variational lower bound of the data likelihood.

For both VAEs and GANs, using some data X we end up with a trained generator G, that is supposed to map latent samples z from the fixed prior distribution to output samples G(z) which (hopefully) have the same distribution as the data.

In order to understand and visualize the learned model G(z), it is a common practice in the literature of generative models to explore how the output G(z) behaves under various arithmetic operations on the latent samples z. In this paper, we show that the operations typically used so far, such as linear interpolation BID3 , spherical interpolation (White, 2016) , vicinity sampling and vector arithmetic BID12 , cause a distribution mismatch between the latent prior distribution and the results of the operations.

This is problematic, since the generator G was trained on a fixed prior and expects to see inputs with statistics consistent with that distribution.

We show that this, somewhat paradoxically, is also a problem if the support of resulting (mismatched) distribution is within the support of a uniformly distributed prior, whose points all have equal likelihood during training.

To address this, we propose to use distribution matching transport maps, to obtain analogous latent space operations (e.g. interpolation, vicinity sampling) which preserve the prior distribution of the latent space, while minimally changing the original operation.

In Figure 1 we showcase how our proposed technique gives an interpolation operator which avoids distribution mismatch when interpolating between samples of a uniform distribution.

The points of the (red) matched trajectories samples from prior linear matched (ours) spherical (a) Uniform prior: Trajectories of linear interpolation, our matched interpolation and the spherical interpolation (White, 2016) . (White, 2016) Figure 1: We show examples of distribution mismatches induced by the previous interpolation schemes when using a uniform prior in two dimensions.

Our matched interpolation avoids this with a minimal modification to the linear trajectory, traversing through the space such that all points along the path are distributed identically to the prior.are obtained as minimal deviations (in expectation of l 1 distance) from the the points of the (blue) linear trajectory.

In the literature there are dozens of papers that use sample operations to explore the learned models.

BID0 use linear interpolation between neighbors in the latent space to study how well deep vs shallow representations can disentangle the latent space of Contractive Auto Encoders (CAEs) BID14 .In the seminal GAN paper of BID3 , the authors use linear interpolation between latent samples to visualize the transition between outputs of a GAN trained on MNIST.

BID2 linearly interpolate the latent codes of an auto encoder trained on a synthetic chair dataset.

BID12 also linearly interpolate between samples to evaluate the quality of the learned representation.

Furthermore, motivated by the semantic word vectors of Mikolov et al. (2013) , they explore using vector arithmetic on the samples to change semantics such as adding a smile to a generated face.

BID13 use linear interpolation to explore their proposed GAN model which operates jointly in the visual and textual domain.

BID1 combine GANs and VAEs for a neural photo editor, using masked interpolations to edit an embedded photo in the latent space.

While there are numerous works performing operations on samples, most of them have ignored the problem of distribution mismatch, such as the one presented in Figure 1d .

BID6 and BID9 sidestep the problem when visualizing their models, by not performing operations on latent samples, but instead restrict the latent space to 2-d and uniformly sample the percentiles of the distribution on a 2-d grid.

This way, the samples have statistics that are consistent with the prior distribution.

However, this approach does not scale up to higher dimensions -whereas the latent spaces used in the literature can have hundreds of dimensions.

Related to our work, White (2016) experimentally observe that there is a distribution mismatch between the distance to origin for points drawn from uniform or Gaussian distribution and points obtained with linear interpolation, and propose to use a so-called spherical linear interpolation to reduce the mismatch, obtaining higher quality interpolated samples.

However, the proposed approach has no theoretical guarantees.

DISPLAYFORM0 Vicinity sampling Table 1 : Examples of interesting sample operations which need to be adapted if we want the distribution of the result y to match the prior distribution.

If the prior is Gaussian, our proposed matched operation simplifies to a proper re-scaling factor (see third column) for additive operations.

DISPLAYFORM1 In this work, we propose a generic method to fully preserve the desired prior distribution when using sample operations.

The approach works as follows: we are given a 'desired' operation, such as linear interpolation y = tz 1 + (1 ??? t)z 2 , t ??? [0, 1].

Since the distribution of y does not match the prior distribution of z, we search for a warping f : DISPLAYFORM2 has the same distribution as z. In order to have the modification??? as faithful as possible to the original operation y, we use optimal transform maps BID17 Villani, 2003; 2008) to find a minimal modification of y which recovers the prior distribution z. Figure 1a , where each point??? of the matched curve is obtained by warping a corresponding point y of the linear trajectory, while not deviating too far from the line.

With implicit models such as GANs BID3 and VAEs BID6 , we use the data X , drawn from an unknown random variable x, to learn a generator G : DISPLAYFORM0 with respect to a fixed prior distribution p z , such that G(z) approximates x. Once the model is trained, we can sample from it by feeding latent samples z through G.We now bring our attention to operations on latent samples DISPLAYFORM1 We give a few examples of such operations in Table 1 .Since the inputs to the operations are random variables, their output y = ??(z 1 , ?? ?? ?? , z k ) is also a random variable (commonly referred to as a statistic).

While we typically perform these operations on realized (i.e. observed) samples, our analysis is done through the underlying random variable y. The same treatment is typically used to analyze other statistics over random variables, such as the sample mean, sample variance and test statistics.

In Table 1 we show example operations which have been commonly used in the literature.

As discussed in the Introduction, such operations can provide valuable insight into how the trained generator G changes as one creates related samples y from some source samples.

The most common such operation is the linear interpolation, which we can view as an operation DISPLAYFORM2 where z 1 , z 2 are latent samples from the prior p z and y t is parameterized by t ??? [0, 1].

Now, assume z 1 and z 2 are i.i.d, and let Z 1 , Z 2 be their (scalar) first components with distribution p Z .

Then the first component of y t is Y t = tZ 1 + (1 ??? t)Z 2 , and we can compute: DISPLAYFORM3 Since (1 + 2t(t ??? 1)) = 1 for all t ??? [0, 1] \ {0, 1}, it is in general impossible for y t to have the same distribution as z, which means that distribution mismatch is inevitable when using linear interpolation.

A similar analysis reveals the same for all of the operations in Table 1 .This leaves us with a dilemma: we have various intuitive operations (see Table 1 ) which we would want to be able to perform on samples, but their resulting distribution p yt is inconsistent with the distribution p z we trained G for.

Due to the curse of dimensionality, as empirically observed by White (2016) , this mismatch can be significant in high dimensions.

We illustrate this in FIG1 , where we plot the distribution of the squared norm y t 2 for the midpoint t = 1/2 of linear interpolation, compared to the prior

In order to address the distribution mismatch, we propose a simple and intuitive strategy for constructing distribution preserving operators, via optimal transport: Strategy 1 (Optimal Transport Matched Operations).

2.

We analytically (or numerically) compute the resulting (mismatched) distribution p y 3.

We search for a minimal modification??? = f (y) (in the sense that E y [c(???, y)] is minimal with respect to a cost c), such that distribution is brought back to the prior, i.e. p??? = p z .The cost function in step 3 could e.g. be the euclidean distance c(x, y) = x ??? y , and is used to measure how faithful the modified operator,??? = f (??(z 1 , ?? ?? ?? , z k )) is to the original operator k. Finding the map f which gives a minimal modification can be challenging, but fortunately it is a well studied problem from optimal transport theory.

We refer to the modified operation??? as the matched version of y, with respect to the cost c and prior distribution p z .For completeness, we introduce the key concept of optimal transport theory in a simplified setting, i.e. assuming probability distributions are in euclidean space and skipping measure theoretical formalism.

We refer to Villani (2003; 2008) and BID17 for a thorough and formal treatment of optimal transport.

The problem of step (3) above was first posed by Monge (1781) and can more formally be stated as: Problem 1 (Santambrogio (2015) Problem 1.1).

Given probability distributions p x , p y , with domains X , Y respectively, and a cost function c : X ?? Y ??? R + , we want to minimize DISPLAYFORM0 We refer to the minimizer f * X ??? Y of (MP) (if it exists), as the optimal transport map from p x to p y with respect to the cost c.

However, the problem remained unsolved until a relaxed problem was studied by Kantorovich (1942):Problem 2 (Santambrogio (2015) Problem 1.2).

Given probability distributions p x , p y , with domains X , Y respectively, and a cost function c : X ?? Y ??? R + , we want to minimize DISPLAYFORM1 where (x, y) ??? p x,y , x ??? p x , y ??? p y denotes that (x, y) have a joint distribution p x,y which has (previously specified) marginals p x and p y .We refer to the joint p x,y which minimizes (KP) as the optimal transport plan from p x to p y with respect to the cost c.

The key difference is to relax the deterministic relationship between x and f (x) to a joint probability distribution p x,y with marginals p x and p y for x and y. In the case of Problem 1, the minimization might be over the empty set since it is not guaranteed that there exists a mapping f such that f (x) ??? y. In contrast, for Problem 2, one can always construct a joint density p x,y with p x and p y as marginals, such as the trivial construction where x and y are independent, i.e. DISPLAYFORM2 Note that given a joint density p x,y (x, y) over X ?? Y, we can view y conditioned on x = x for a fixed x as a stochastic function f (x) from X to Y, since given a fixed x do not get a specific function value f (x) but instead a random variable f (x) that depends on x, with f (x) ??? y|x = x with density p y (y|x = x) := px,y(x,y)px (x) .

In this case we have (x, f (x)) ??? p x,y , so we can view the Problem KP as a relaxation of Problem MP where f is allowed to be a stochastic mapping.

While the relaxed problem of Kantorovich (KP) is much more studied in the optimal transport literature, for our purposes of constructing operators it is desirable for the mapping f to be deterministic as in (MP).To this end, we will choose the cost function c such that the two problems coincide and where we can find an analytical solution f or at least an efficient numerical solution.

In particular, we note that most operators in Table 1 are all pointwise, such that if the points z i have i.i.d.

components, then the result y will also have i.i.d.

components.

If we combine this with the constraint for the cost c to be additive over the components of x, y, we obtain the following simplification: Theorem 1.

Suppose p x and p y have i.i.d components and c over DISPLAYFORM3 Consequently, the minimization problems (MP) and (KP) turn into d identical scalar problems for the distributions p X and p Y of the components of x and y: DISPLAYFORM4 such that an optimal transport map T for (MP-1-D) gives an optimal transport map f for (MP) by pointwise application of T , i.e. f (x) (i) := T (x (i) ), and an optimal transport plan p X,Y for DISPLAYFORM5 Proof.

See Appendix.

Fortunately, under some mild constraints, the scalar problems have a known solution: Theorem 2 (Theorem 2.9 in BID17 ).

Let h : R ??? R + be convex and suppose the cost C takes the form C(x, y) = h(x ??? y).

Given an continuous source distribution p X and a target distribution p Y on R having a finite optimal transport cost in (KP-1-D), then defines an optimal transport map from DISPLAYFORM6 DISPLAYFORM7 is the Cumulative Distribution Function (CDF) of X and F DISPLAYFORM8 ??? y} is the pseudo-inverse of F Y .

Furthermore, the joint distribution of (X, T mon X???Y (X)) defines an optimal transport plan for (KP-1-D).The mapping T mon X???Y (x) in Theorem 2 is non-decreasing and is known as the monotone transport map from X to Y .

It is easy to verify that T mon X???Y (X) has the distribution of Y , in particular DISPLAYFORM9 Now, combining Theorems 1 and 2, we obtain a concrete realization of the Strategy 1 outlined above.

We choose the cost c such that it admits to Theorem 1, such as c(x, y) := x ??? y 1 , and use an operation that is pointwise, so we just need to compute the monotone transport map in (5).

That is, if z has i.i.d components with distribution p Z , we just need to compute the component distribution p Y of the result y of the operation, the CDFs F Z , F Y and obtain DISPLAYFORM10 as the component-wise modification of y, i.e.??? DISPLAYFORM11 In FIG2 we show the monotone transport map for the linear interpolation y = tz 1 + (1 ??? t)z 2 for various values of t. The detailed calculations and examples for various operations are given in Appendix 5.3, for both Uniform and Gaussian priors.

The Gaussian case has a particularly simple resulting transport map for additive operations, where it is just a linear transformation through a scalar multiplication, summarized in the third column of Table 1 .

To validate the correctness of the matched operators obtained above, we numerically simulate the distributions for toy examples, as well as prior distributions typically used in the literature.

Priors vs. interpolations in 2-D For Figure 1 , we sample 1 million pairs of points in two dimension, from a uniform prior (on [???1, 1] 2 ), and estimate numerically the midpoint distribution of linear interpolation, our proposed matched interpolation and the spherical interpolation of White (2016).

It is reassuring to see that the matched interpolation gives midpoints which are identically distributed to the prior.

In contrast, the linear interpolation condenses more towards the origin, forming a pyramid-shaped distribution (the result of convolving two boxes in 2-d).

Since the spherical interpolation of White (2016) follows a great circle with varying radius between the two points, we see that the resulting distribution has a "hole" in it, "circling" around the origin for both priors.

distribution of the squared norm of the midpoints.

We see there is a dramatic difference between vector lengths in the prior and the midpoints of linear interpolation, with only minimal overlap.

We also show the spherical (SLERP) interpolation of White (2016) which has a matching first moment, but otherwise also induces a distribution mismatch.

In contrast, our matched interpolation, fully preserves the prior distribution and perfectly aligns.

We note that this setting (d = 100, uniform or Gaussian) is commonly used in the literature.

In this section we will present some concrete examples for the differences in generator output dependent on the exact sample operation used to traverse the latent space of a generative model.

To this end, the generator output for latent samples produced with linear interpolation, SLERP (spherical linear interpolation) of White (2016) and our proposed matched interpolation will be compared.

Please refer to Table 1 for an overview of the operators used in this Section.

Setup We used DCGAN BID12 generative models trained on LSUN bedrooms (Yu et al., 2015) , CelebA BID7 and LLD BID15 , an icon dataset, to qualitatively evaluate.

For LSUN, the model was trained for two different output resolutions, providing 64 ?? 64 pixel and a 128??128 pixel output images (where the latter is used in figures containing larger sample images).

The models for LSUN and the icon dataset where both trained on a uniform latent prior distribution, while for CelebA a Gaussian prior was used.

The dimensionality of the latent space is 100 for both LSUN and CelebA, and 512 for the model trained on the icon model.

Furthermore we use improved Wasserstein GAN (iWGAN) with gradient penalty BID4 ) trained on CIFAR-10 at 32 ?? 32 pixels with a 128-dimensional Gaussian prior to produce the inception scores presented in Section 3.3.

We begin with the classic example of 2-point interpolation: FIG3 shows three examples per dataset for an interpolation between 2 points in latent space.

Each example is first done via linear interpolation, then SLERP and finally matched interpolation.

In FIG5 in the Appendix we show more densely sampled examples.

FIG3 that linear interpolation produces inferior results with generally more blurry, less saturated and less detailed output images.

SLERP and matched interpolation are slightly different, however it is not visually obvious which one is superior.

Differences between the various interpolation methods for CelebA FIG3 ) are much more subtle to the point that they are virtually indistinguishable when viewed side-by-side.

This is not an inconsistency though: while distribution mismatch can cause large differences, it can also happen that the model generalizes well enough that it does not matter.

In all cases, the point where the interpolation methods diverge the most, is at the midpoint of the interpolation where t = 0.5.

Thus we provide 25 such interpolation midpoints in Figures 5 (LLD icons) and 6 (LSUN) for direct comparison.

3.69 ?? 0.10 3.91 ?? 0.10 2.04 ?? 0.04 Table 2 : Inception scores on LLD-icon, LSUN, CIFAR-10 and CelebA for the midpoints of various interpolation operations.

Scores are reported as mean ?? standard deviation (relative change in %).

highlights the very apparent loss of detail and increasing prevalence of artifacts towards the midpoint in the linear version, compared to SLERP compared and our matched interpolation.

Vicinity sampling Furthermore we provide two examples for vicinity sampling in Figures 9 and 10.

Analogous to the previous observations, the output under a linear operator lacks definition, sharpness and saturation when compared to both spherical and matched operators.

Random walk An interesting property of our matched vicinity sampling is that we can obtain a random walk in the latent space by applying it repeatedly: we start at a point y 0 = z drawn from the prior, and then obtain point y i by sampling a single point in the vicinity of y i???1 , using some fixed 'step size' .We show an example of such a walk in FIG8 , using = 0.5.

As a result of the repeated application of the vicinity sampling operation, the divergence from the prior distribution in the non-matched case becomes stronger with each step, resulting in completely unrecognizable output images on the LSUN and LLD icon models.

Even for the CelebA model where differences where minimal before, they are quite apparent in this experiment.

The random walk thus perfectly illustrates the need for respecting the prior distribution when performing any operation in latent space, as the adverse effects can cumulate through the repeated application of operators that do not comply to the prior distribution.

We quantitatively confirm the observations of the previous section by using the Inception score BID16 .

In Table 2 we compare the Inception score of our trained models (i.e. using random samples from the prior) with the score when sampling midpoints from the 2-point and 4-point interpolations described above, reporting mean and standard deviation with 50,000 samples, as well as relative change to the original model scores if they are significant.

Compared to the original scores of the trained models, our matched operations are statistically indistinguishable (as expected) while the linear interpolation gives a significantly lower score in all settings (up to 29% lower).

As observed for the quality visually, the SLERP heuristic gives similar scores to the matched operations.

We have shown that the common latent space operations used for Generative Models induce distribution mismatch from the prior distribution the models were trained for.

This problem has been mostly ignored by the literature so far, partially due to the belief that this should not be a problem for uniform priors.

However, our statistical and experimental analysis shows that the problem is real, with the operations used so far producing significantly lower quality samples compared to their inputs.

To address the distribution mismatch, we propose to use optimal transport to minimally modify (in l 1 distance) the operations such that they fully preserve the prior distribution.

We give analytical formulas of the resulting (matched) operations for various examples, which are easily implemented.

The matched operators give a significantly higher quality samples compared to the originals, having the potential to become standard tools for evaluating and exploring generative models.

We note that the analysis here can bee seen as a more rigorous version of an observation made by White (2016) , who experimentally show that there is a significant difference between the average norm of the midpoint of linear interpolation and the points of the prior, for uniform and Gaussian distributions.

Suppose our latent space has a prior with DISPLAYFORM0 In this case, we can look at the squared norm DISPLAYFORM1 From the Central Limit Theorem (CLT), we know that as d ??? ???, DISPLAYFORM2 in distribution.

Thus, assuming d is large enough such that we are close to convergence, we can approximate the distribution of z 2 as N (d?? Z 2 , d?? 2 Z 2 ).

In particular, this implies that almost all points lie on a relatively thin spherical shell, since the mean grows as O(d) whereas the standard deviation grows only as O( DISPLAYFORM3 We note that this property is well known for i.i.d Gaussian entries (see e.g. Ex.

6.14 in MacKay FORMULA5 ).

For Uniform distribution on the hypercube it is also well known that the mass is concentrated in the corner points (which is consistent with the claim here since the corner points lie on a sphere).Now consider an operator such as the midpoint of linear interpolation, y = DISPLAYFORM4 In this case, we can compute: DISPLAYFORM5 Thus, the distribution of y 2 can be approximated with N ( DISPLAYFORM6 .

Therefore, y also mostly lies on a spherical shell, but with a different radius than z. In fact, the shells will intersect at regions which have a vanishing probability for large d. In other words, when looking at the squared norm y 2 , y 2 is a (strong) outlier with respect to the distribution of z 2 .

Proof.

We will show it for the Kantorovich problem, the Monge version is similar.

Starting from (KP), we compute DISPLAYFORM0 where the inequality in (14) is due to each term being minimized separately.

DISPLAYFORM1 where p X,Y has marginals p X and p Y .

In this case P d (X, Y ) is a subset of all joints p x,y with marginals p x and p y , where the pairs ( DISPLAYFORM2 ) are constrained to be i.i.d.

Starting again from (13) can compute: DISPLAYFORM3 where the inequality in (18) is due to minimizing over a smaller set.

Since the two inequalities above are in the opposite direction, equality must hold for all of the expressions above, in particular: DISPLAYFORM4 Thus, (KP) and (KP-1-D) equal up to a constant, and minimizing one will minimize the other.

Therefore the minimization of the former can be done over p X,Y with p x,y (x, y) = DISPLAYFORM5

In the next sections, we illustrate how to compute the matched operations for a few examples, in particular for linear interpolation and vicinity sampling, using a uniform or a Gaussian prior.

We picked the examples where we can analytically compute the uniform transport map, but note that it is also easy to compute F DISPLAYFORM0 and (F Y (y)) numerically, since one only needs to estimate CDFs in one dimension.

Since the components of all random variables in these examples are i.i.d, for such a random vector x we will implicitly write X for a scalar random variable that has the distribution of the components of x.

When computing the monotone transport map T mon X???Y , the following Lemma is helpful.

Lemma 1 (Theorem 2.5 in BID17 ).

Suppose a mapping g(x) is non-decreasing and maps a continuous distribution p X to a distribution p Y , i.e. DISPLAYFORM1 then g is the monotone transport map T mon X???Y .According to Lemma 1, an alternative way of computing T mon X???Y is to find some g that is nondecreasing and transforms p X to p Y .

Suppose z has uniform components Z ??? Uniform(???1, 1).

In this case, p Z (z) = 1/2 for ???1 < z < 1.

Now let y t = tz 1 + (1 ??? t)z 2 denote the linear interpolation between two points z 1 , z 2 , with component distribution p Yt .

Due to symmetry we can assume that t > 1/2, since p Yt = p Y1???t .

We then obtain p Yt as the convolution of p tZ and p (1???t)Z , i.e. p Yt = p tZ * p (1???t)Z .

First we note that p tZ = 1/(2t) for ???t < z < t and p (1???t)Z = 1/(2(1 ??? t)) for ???(1 ??? t) < z < 1 ??? t. We can then compute: DISPLAYFORM0 if ??? t + (1 ??? t) < y < t ??? (1 ??? t) ???y + 1 if t ??? (1 ??? t) < y < 1 0 if 1 < yThe CDF F Yt is then obtained by computing Since p Z (z) = 1/2 for |z| < 1, we have F Z (z) = .

For vicinity sampling, we want to obtain new points z 1 , ??, z k which are close to z. We thus define DISPLAYFORM1 where u i also has uniform components, such that each coordinate of z i differs at most by from z. By identifying tZ i = tZ + (1 ??? t)U i with t = 1/(1 + ), we see that tZ i has identical distribution to the linear interpolation Y t in the previous example.

Thus g t (Z i ) := T mon Yt???Z (tZ i ) will have the distribution of Z, and by Lemma1 is then the monotone transport map from Z i to Z.

Suppose z has components Z ??? N (0, ?? 2 ).

In this case, we can compute linear interpolation as before, y t = tz 1 + (1 ??? t)z 2 .

Since the sum of Gaussians is Gaussian, we get, Y t ??? N (0, t 2 ?? 2 + (1 ??? t) 2 ?? 2 ).

Now, it is easy to see that with a proper scaling factor, we can adjust the variance of Y t back to ?? 2 .

That is, By adjusting the vicinity sampling operation to DISPLAYFORM0 where e i ??? N (0, 1), we can similarly find the monotone transport map g (y) = 1 ??? 1+ 2 y. Another operation which has been used in the literature is the "analogy", where from samples z 1 , z 2 , z 3 , one wants to apply the difference between z 1 and z 2 , to z 3 .

The transport map is then g(y) =

<|TLDR|>

@highlight

Operations in the GAN latent space can induce a distribution mismatch compared to the training distribution, and we address this using optimal transport to match the distributions. 
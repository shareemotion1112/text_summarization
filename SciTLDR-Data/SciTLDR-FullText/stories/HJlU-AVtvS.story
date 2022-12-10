Are neural networks biased toward simple functions?

Does depth always help learn more complex features?

Is training the last layer of a network as good as training all layers?

These questions seem unrelated at face value, but in this work we give all of them a common treatment from the spectral perspective.

We will study the spectra of the *Conjugate Kernel, CK,* (also called the *Neural Network-Gaussian Process Kernel*), and the *Neural Tangent Kernel, NTK*. Roughly, the CK and the NTK tell us respectively ``"what a network looks like at initialization" and "``what a network looks like during and after training."

Their spectra then encode valuable information about the initial distribution and the training and generalization properties of neural networks.

By analyzing the eigenvalues, we lend novel insights into the questions put forth at the beginning, and we verify these insights by extensive experiments of neural networks.

We believe the computational tools we develop here for analyzing the spectra of CK and NTK serve as a solid foundation for future studies of deep neural networks.

We have open-sourced the code for it and for generating the plots in this paper at github.com/jxVmnLgedVwv6mNcGCBy/NNspectra.

Understanding the behavior of neural networks and why they generalize has been a central pursuit of the theoretical deep learning community.

Recently, Valle-Pérez et al. (2018) observed that neural networks have a certain "simplicity bias" and proposed this as a solution to the generalization question.

One of the ways with which they argued that this bias exists is the following experiment: they drew a large sample of boolean functions by randomly initializing neural networks and thresholding the output.

They observed that there is a bias toward some "simple" functions which get sampled disproportionately more often.

However, their experiments were only done for relu networks.

Can one expect this "simplicity bias" to hold universally, for any architecture?

A priori, this seems difficult, as the nonlinear nature seems to present an obstacle in reasoning about the distribution of random networks.

However, this question turns out to be more easily treated if we allow the width to go to infinity.

A long line of works starting with Neal (1995) and extended recently by ; ; Yang (2019) have shown that randomly initialized, infinite-width networks are distributed as Gaussian processes.

These Gaussian processes also describe finite width random networks well (Valle-Pérez et al., 2018) .

We will refer to the corresponding kernels as the Conjugate Kernels (CK), following the terminology of Daniely et al. (2016) .

Given the CK K, the simplicity bias of a wide neural network can be read off quickly from the spectrum of K: If the largest eigenvalue of K accounts for most of tr K, then a typical random network looks like a function from the top eigenspace of K.

In this paper, we will use this spectral perspective to probe not only the simplicity bias, but more generally, questions regarding how hyperparameters affect the generalization of neural networks.

Via the usual connection between Gaussian processes and linear models with features, the CK can be thought of as the kernel matrix associated to training only the last layer of a wide randomly initialized network.

It is a remarkable recent advance (Jacot et al., 2018; Allen-Zhu et al., 2018a; c; Du et al., 2018) that, under a certain regime, a wide neural network of any depth evolves like a linear model even when training all parameters.

The associated kernel is call the Neural Tangent Kernel, which is typically different from CK.

While its theory was initially derived in the infinite width setting, Lee et al. (2019) confirmed with extensive experiment that this limit is predictive of finite width neural networks as well.

Thus, just as the CK reveals information about what a network looks like at Next, we examine how hyperparameters affect the performance of neural networks through the lens of NTK and its spectrum.

To do so, we first need to understand the simpler question of how a kernel affects the accuracy of the function learned by kernel regression.

A coarse-grained theory, concerned with big-O asymptotics, exists from classical kernel literature (Yao et al., 2007; Raskutti et al., 2013; Lin and Rosasco; Schölkopf and Smola, 2002) .

However, the fine-grained details, required for discerning the effect of hyperparameters, have been much less studied.

We make a first attempt at a heuristic, fractional variance (i.e. what fraction of the trace of the kernel does an eigenspace contribute), for understanding how a minute change in kernel effects a change in performance.

Intuitively, if an eigenspace has very large fractional variance, so that it accounts for most of the trace, then a ground truth function from this eigenspace should be very easy to learn.

Using this heuristic, we make two predictions about neural networks, motivated by observations in the spectra of NTK and CK, and verify them with extensive experiments.

• Deeper networks learn more complex features, but excess depth can be detrimental as well.

Spectrally, depth can increase fractional variance of an eigenspace, but past an optimal depth, it will also decrease it. (Section 5) Thus, deeper is not always better.

• Training all layers is better than training just the last layer when it comes to more complex features, but the opposite is true for simpler features.

Spectrally, fractional variances of more "complex" eigenspaces for the NTK are larger than the correponding quantities of the CK. (Section 6) Finally, we use our spectral theory to predict the maximal nondiverging learning rate ("max learning rate") of SGD (Section 7).

In general, we will not only verify our theory with experiments on the theoretically interesting distributions, i.e. uniform measures over the boolean cube and the sphere, or the standard Gaussian, but also confirm these findings on real data like MNIST and CIFAR10 1 .

For space concerns, we review relevant literature along the flow of the main text, and relegate a more complete discussion of the related research landscape in Appendix A.

As mentioned in the introduction, we now know several kernels associated to infinite width, randomly initialized neural networks.

The most prominent of these are the neural tangent kernel (NTK) (Jacot et al., 2018) and the conjugate kernel (CK) (Daniely et al., 2016) , which is also called the NNGP kernel .

We briefly review them below.

First we introduce the following notation that we will repeatedly use.

Definition 2.1.

For φ : R → R, write V φ for the function that takes a PSD (positive semidefinite) kernel function to a PSD kernel of the same domain by the formula

Conjugate Kernel Neural networks are commonly thought of as learning a high-quality embedding of inputs to the latent space represented by the network's last hidden layer, and then using its final linear layer to read out a classification given the embedding.

The conjugate kernel is just the kernel associated to the embedding induced by a random initialization of the neural network.

Consider an MLP with widths {n l } l , weight matrices {W l ∈ R n l ×n l−1 } l , and biases {b l ∈ R n l } l , l = 1, . . .

, L.

For simplicity of exposition, in this paper, we will only consider scalar output n L = 1.

Suppose it is parametrized by the NTK parametrization, i.e. its computation is given recursively as

with some hyperparameters σ w , σ b that are fixed throughout training 2 .

At initialization time, suppose W l αβ , b .

It can be shown that, for each α ∈ [n l ], h l α is a Gaussian process with zero mean and kernel function Σ l in the limit as all hidden layers become infinitely wide (n l → ∞, l = 1, . . .

, L − 1), where Σ l is defined inductively on l as

The kernel Σ L corresponding the the last layer L is the network's conjugate kernel, and the associated Gaussian process limit is the reason for its alternative name Neural Network-Gaussian process kernel.

In short, if we were to train a linear model with features given by the embedding x → h L−1 (x) when the network parameters are randomly sampled as above, then the CK is the kernel of this linear model.

See Daniely et al. (2016) ; and Appendix F for more details.

Neural Tangent Kernel On the other hand, the NTK corresponds to training the entire model instead of just the last layer.

Intuitively, if we let θ be the entire set of parameters {W l } l ∪ {b l } l of Eq. (MLP), then for θ close to its initialized value θ 0 , we expect

via a naive first-order Taylor expansion.

In other words, h L (x; θ) − h L (x; θ 0 ) behaves like a linear model with feature of x given by the gradient taken w.r.t.

the initial network, ∇ θ h L (x; θ 0 ), and the weights of this linear model are the deviation θ − θ 0 of θ from its initial value.

It turns out that, in the limit as all hidden layer widths tend to infinity, this intuition is correct (Jacot et al., 2018; Yang, 2019) , and the following inductive formula computes the corresponding infinite-width kernel of this linear model:

Computing CK and NTK While in general, computing V φ and V φ requires evaluating a multivariate Gaussian expectation, in specific cases, such as when φ = relu or erf, there exists explicit, efficient formulas that only require pointwise evaluation of some simple functions (see Facts F.1 and F.2).

This allows us to evaluate CK and NTK on a set X of inputs in only time O(|X | 2 L).

What Do the Spectra of CK and NTK Tell Us?

In summary, the CK governs the distribution of a randomly initialized neural network and also the properties of training only the last layer of a network, while the NTK governs the dynamics of training (all parameters of) a neural network.

A study of their spectra thus informs us of the "implicit prior" of a randomly initialized neural network as well as the "implicit bias" of GD in the context of training neural networks.

In regards to the implicit prior at initialization, we know from that a randomly initialized network as in Eq. (MLP) is distributed as a Gaussian process N (0, K), where K is the corresponding CK, in the infinite-width limit.

If we have the eigendecomposition

with eigenvalues λ i in decreasing order and corresponding eigenfunctions u i , then each sample from this GP can be obtained as

If, for example, λ 1 i≥2 λ i , then a typical sample function is just a very small perturbation of u 1 .

We will see that for relu, this is indeed the case (Section 4), and this explains the "simplicity bias" in relu networks found by Valle-Pérez et al. (2018) .

Training the last layer of a randomly initialized network via full batch gradient descent for an infinite amount of time corresponds to Gaussian process inference with kernel K 2019) .

A similar intuition holds for NTK: training all parameters of the network (Eq. (MLP)) for an infinite amount of time yields the mean prediction of the GP N (0, NTK) in expectation; see Lee et al. (2019) and Appendix F.4 for more discussion.

Thus, the more the GP prior (governed by the CK or the NTK) is consistent with the ground truth function f * , the more we expect the Gaussian process inference and GD training to generalize well.

We can measure this consistency in the "alignment" between the eigenvalues λ i and the squared coefficients a 2 i of f * 's expansion in the {u i } i basis.

The former can be interpreted as the expected magnitude (squared) of the u i -component of a sample f ∼ N (0, K), and the latter can be interpreted as the actual magnitude squared of such component of f * .

In this paper, we will investigate an even cleaner setting where f * = u i is an eigenfunction.

Thus we would hope to use a kernel whose ith eigenvalue λ i is as large as possible.

Neural Kernels From the forms of the equation Eqs. (CK) and (NTK) and the fact that V φ (K)(x, x ) only depends on K(x, x), K(x, x ), and K(x , x ), we see that CK or NTK of MLPs takes the form

for some function Φ :

R 3 → R. We will refer to this kind of kernel as Neural Kernel in this paper.

We will consider input spaces of various forms X ⊆ R d equipped with some probability measure.

Then a kernel function K acts as an integral operator on functions

We will use the "juxtaposition syntax" Kf to denote this application of the integral operator.

3 Under certain assumptions, it then makes sense to speak of the eigenvalues and eigenfunctions of the integral operator K. While we will appeal to an intuitive understanding of eigenvalues and eigenfunctions in the main text below, we include a more formal discussion of Hilbert-Schmidt operators and their spectral theory in Appendix G for completeness.

In the next section, we investigate the eigendecomposition of neural kernels as integral operators over different distributions.

3 THE SPECTRA OF NEURAL KERNELS

We first consider a neural kernel K on the boolean cube X = d def = {±1} d , equipped with the uniform measure.

In this case, since each x ∈ X has the same norm, K(x, y) = Φ

x,y x y ,

effectively only depends on x, y , so we will treat Φ as a single variate function in this section, Φ(c) = Φ(c, 1, 1).

Brief review of basic Fourier analysis on the boolean cube (2014)).

The space of real functions on

Any such function has a unique expansion into a multilinear polynomial (polynomials whose monomials do not contain x p i , p ≥ 2, of any variable x i ).

For example, the majority function over 3 bits has the following unique multilinear expansion

In the language of Fourier analysis, the 2 d multilinear monomial functions

form a Fourier basis of the function space

Thus, any function f :

d → R can be always written as

for a unique set of coefficients {f (S)} S⊆ [d] .

It turns out that K is always diagonalized by this Fourier basis {χ S } S⊆ [d] .

Theorem 3.1.

On the d-dimensional boolean cube

where 1 = (1, . . . , 1) ∈ d .

This definition of µ |S| does not depend on the choice S, only on the cardinality of S. These are all of the eigenfunctions of K by dimensionality considerations.

Define T ∆ to be the shift operator on functions over [−1, 1] that sends Φ(·) to Φ(· − ∆).

Then we can re-express the eigenvalue as follows.

Lemma 3.2.

With µ k as in Thm 3.1,

where

Eq. (5) will be important for computational purposes, and we will come back to discuss this more in Section 3.5.

It also turns out µ k affords a pretty expression via the Fourier series coefficients of Φ.

As this is not essential to the main text, we relegate its exposition to Appendix H.1.

Now let's consider the case when X = √ dS d−1 is the radius-√ d sphere in R d equipped with the uniform measure.

Again, because x ∈ X all have the same norm, we will treat Φ as a univariate function with K(x, y) = Φ( x, y / x y ) = Φ( x, y /d).

As is long known (Schoenberg, 1942; Gneiting, 2013; Xu and Cheney, 1992; Smola et al., 2001) , K is diagonalized by spherical harmonics, and the eigenvalues are given by the coefficients of Φ against a system of orthogonal polynomials called Gegenbuaer polynomials.

We relegate a complete review of this topic to Appendix H.2.

Now let's consider X = R d equipped with standard isotropic Gaussian N (0, I), so that K behaves like

for any f ∈ L 2 (N (0, I)).

In contrast to the previous two sections, K will essentially depend on the effect of the norms x and y on Φ.

Nevertheless, because an isotropic Gaussian vector can be obtained by sampling its direction uniformly from the sphere and its magnitude from a chi distribution, K can still be partially diagonalized into a sum of products between spherical harmonics and kernels on R equipped with a chi distribution (Thm H.14).

In certain cases, we can obtain complete eigendecompositions, for example when Φ is positive homogeneous.

See Appendix H.3 for more details.

The reason we have curtailed a detailed discussion of neural kernels on the sphere and on the standard Gaussian is because, in high dimension, the kernel behaves the same under these distributions as under uniform distribution over the boolean cube.

Indeed, by intuition along the lines of the central limit theorem, we expect that uniform distribution over a high dimension boolean cube should approximate high dimensional standard Gaussian.

Similarly, by concentration of measure, most of the mass of a Gaussian is concentrated around a thin shell of radius √ d. Thus, morally, we expect the same kernel function K induces approximately the same integral operator on these three distributions in high dimension, and as such, their eigenvalues should also approximately coincide.

We verify empirically and theoretically this is indeed the case in Appendix H.4.

As the eigenvalues of K over the different distributions are very close, we will focus in the rest of this paper on eigenvalues over the boolean cube.

This has the additional benefit of being much easier to compute.

Each eigenvalue over the sphere and the standard Gaussian requires an integration of Φ against a Gegenbauer polynomial.

In high dimension d, these Gegenbauer polynomials varies wildly in a sinusoidal fashion, and blows up toward the boundary (see Fig. 15 in the Appendix).

As such, it is difficult to obtain a numerically stable estimate of this integral in an efficient manner when d is large.

In contrast, we have multiple ways of computing boolean cube eigenvalues, via Eqs. (5) and (6).

In either case, we just take some linear combination of the values of Φ at a grid of points on [−1, 1], spaced apart by ∆ = 2/d.

While the coefficients C d−k,k r (defined in Eq. (7)) are relatively efficient to compute, the change in the sign of C d−k,k r makes this procedure numerically unstable for large d. Instead, we use Eq. (5) to isolate the alternating part to evaluate in a numerically stable way:

Since

k Φ via k finite differences, and then , we randomly initialize a network of 2 hidden layers, 40 neurons each.

Then we threshold the function output to a boolean output, and obtain a boolean function sample.

We repeat this for 10 4 random seeds to obtain all samples.

Then we sort the samples according to their empirical probability (this is the x-axis, rank), and plot their empirical probability (this is the y-axis, probability).

The high values at the left of the relu curve indicates that a few functions get sampled repeatedly, while this is less true for erf.

For erf and σ , and depth, we study the eigenvalues of the corresponding CK.

Each CK has 8 different eigenvalues µ 0 , . . .

, µ 7 corresponding to homogeneous polynomials of degree 0, . . .

, 7.

We plot them in log scale against the degree.

Note that for erf and σ b = 0, the even degree µ k vanish.

See main text for explanations.

When Φ arises from the CK or the NTK of an MLP, all derivatives of Φ at 0 are nonnegative (Thm I.3).

Thus intuitively, the finite differenceΦ should be also all nonnegative, and this sum can be evaluated without worry about floating point errors from cancellation of large terms.

A slightly more clever way to improve the numerical stability when 2k ≤ d is to note that

So an improved algorithm is to first compute the kth finite difference (I − T 2∆ ) k with the larger step size 2∆, then compute the sum (I + T ∆ ) d−2k as in Eq. (8).

4 CLARIFYING THE "SIMPLICITY BIAS" OF RANDOM NEURAL NETWORKS As mentioned in the introduction, Valle-Pérez et al. (2018) claims that neural networks are biased toward simple functions.

We show that this phenomenon depends crucially on the nonlinearity, the sampling variances, and the depth of the network.

In Fig. 1(a) , we have repeated their experiment for 10 4 random functions obtained by sampling relu neural networks with 2 hidden layers, 40 neurons each, following Valle-Pérez et al. (2018) 's architectural choices 5 .

We also do the same for erf networks of the same depth and width, varying as well the sampling variances of the weights and biases, as shown in the legend.

As discussed in Valle-Pérez et al. (2018) , for relu, there is indeed this bias, where a single function gets sampled more than 10% of the time.

However, for erf, as we increase σ 2 w , we see this bias disappear, and every function in the sample gets sampled only once.

This phenomenon can be explained by looking at the eigendecomposition of the CK, which is the Gaussian process kernel of the distribution of the random networks as their hidden widths tend to infinity.

In Fig. 1(b) , we plot the normalized eigenvalues {µ k / 7 i=0 7 i µ i } 7 k=0 for the CKs corresponding to the networks sampled in Fig. 1(a) .

Immediately, we see that for relu and σ 2 w = σ 2 b = 2, the degree 0 eigenspace, corresponding to constant functions, accounts for more than 80% of the variance.

This means that a typical infinite-width relu network of 2 layers is expected to be almost constant, and this should be even more true after we threshold the network to be a boolean function.

On the other hand, for erf and σ b = 0, the even degree µ k s all vanish, and most of the variance comes from degree 1 components (i.e. linear functions).

This concentration in degree 1 also lessens as σ 2 w increases.

But because this variance is spread across a dimension 7 eigenspace, we don't see duplicate function samples nearly as much as in the relu case.

As σ w increases, we also see the eigenvalues become more equally distributed, which corresponds to the flattening of , so we take a closer look at this slice by plotting heatmaps of fractional variance of various degrees versus depth for relu (c) and erf (d) NTK, with bright colors representing high variance.

Clearly, we see the brightest region of each column, corresponding to a fixed degree, moves up as we increase the degree, barring for the even/odd degree alternating pattern for erf NTK.

The pattern for CKs are similar and their plots are omitted.

the probability-vs-rank curve in Fig. 1(a) .

Finally, we observe that a 32-layer erf network with σ 2 w = 4 has all its nonzero eigenvalues (associated to odd degrees) all equal (see points marked by * in Fig. 1(b) ).

This means that its distribution is a "white noise" on the space of odd functions, and the distribution of boolean functions obtained by thresholding the Gaussian process samples is the uniform distribution on odd functions.

This is the complete lack of simplicity bias modulo the oddness constraint.

However, from the spectral perspective, there is a weak sense in which a simplicity bias holds for all neural network-induced CKs and NTKs.

Even though it's not true that the fraction of variance contributed by the degree k eigenspace is decreasing with k, the eigenvalue themselves will be in a nonincreasing pattern across even and odd degrees.

In fact, if we fix k and let d → ∞, then we can show that (Thm I.6)

Of course, as we have seen, this is a very weak sense of simplicity bias, as it doesn't prevent "white noise" behavior as in the case of erf CK with large σ 2 w and large depth.

In the rest of this work, we compute the eigenvalues µ k over the 128-dimensional boolean cube ( d , with d = 128) for a large number of different hyperparameters, and analyze how the latter affect the former.

We vary the degree k ∈ [0, 8] , the nonlinearity between relu and erf, the depth (number of hidden layers) from 1 to 128, and σ 2 b ∈ [0, 4].

We fix σ 2 w = 2 for relu kernels, but additionally vary σ 2 w ∈ [1, 5] for erf kernels.

Comprehensive contour plots of how these hyperparameters affect the kernels are included in Appendix D, but in the main text we summarize several trends we see.

We will primarily measure the change in the spectrum by the degree k fractional variance, which is just

This terminology comes from the fact that, if we were to sample a function f from a Gaussian process with kernel K, then we expect that r% of the total variance of f comes from degree k components of f , where r% is the degree k fractional variance.

If we were to try to learn a homogeneous degree-k Verifying best depths and NTK complexity bias, varying degree of ground truth Figure 3 : (a) We train relu networks of different depths against a ground truth polynomial on 128 of different degrees k. We either train only the last layer (marked "ck") or all layers (marked "ntk"), and plot the degree k fractional variance of the corresponding kernel against the best validation loss over the course of training.

We see that the best validation loss is in general inversely correlated with fraction variance, as expected.

However, their precise relationship seems to change depending on the degree, or whether training all layers or just the last.

See Appendix E for experimental details.

(b) Same experimental setting as (a), with slightly different hyperparameters, and plotting depth against best validation loss (solid curves), as well as the corresponding kernel's (1− fractional variance) (dashed curves).

We see that the loss-minimizing depth increases with the degree, as predicted by Fig. 2 .

Note that we do not expect the dashed and solid curves to match up, just that they are positively correlated as shown by (a).

In higher degrees, the losses are high across all depths, and the variance is large, so we omit them.

See Appendix E for experimental details.

(c) Similar experimental setting as (a), but with more hyperparameters, and now comparing training last layer vs training all layers.

The color of each dot indicates the degree of the ground truth polynomial.

Below the identity line, training all layers is better than training last layer.

We see that the only nontrivial case where this is not true is when learning degree 0 polynomials, i.e. constant functions.

See Appendix E for experimental details.

We also replicate (b) for MNIST and CIFAR10, and moreover both (b) and (c) over the input distributions of standard Gaussian and the uniform measure over the sphere.

See Figs. 6 to 8.

polynomial using a kernel K, intuitively we should try to choose K such that its µ k is maximized, relative to other eigenvalues.

Fig. 3 (a) shows that this is indeed the case even with neural networks: over a large number of different hyperparameter settings, degree k fractional variance is inversely related to the validation loss incurred when learning a degree k polynomial.

However, this plot also shows that there does not seem like a precise, clean relationship between fractional variance and validation loss.

Obtaining a better measure for predicting generalization is left for future work.

Before we continue, we remark that the fractional variance of a fixed degree k converges to a fixed value as the input dimension d → ∞: Theorem 5.1 (Asymptotic Fractional Variance).

Let K be the CK or NTK of an MLP on a boolean cube d .

Then K can be expressed as K(x, y) = Φ( x, y /d) for some analytic function Φ : R → R.

If we fix k and let the input dimension d → ∞, then the fractional variance of degree k converges to

where Φ (k) denotes the kth derivative of Φ.

For the fractional variances we compute in this paper, their values at d = 128 are already very close to their d → ∞ limit, so we focus on the d = 128 case experimentally.

If K were to be the CK or NTK of a relu or erf MLP, then we find that for higher k, the depth of the network helps increase the degree k fractional variance.

In Fig. 2 (a) and (b), we plot, for each degree k, the depth that (with some combination of other hyperparameters like σ 2 b ) achieves this maximum, for respectively relu and erf kernels.

Clearly, the maximizing depths are increasing with k for relu, and also for erf when considering either odd k or even k only.

The slightly differing behavior between even and odd k is expected, as seen in the form of Thm 4.1.

Note the different scales of y-axes for relu and erf -the depth effect is much stronger for erf than relu.

For relu NTK and CK, σ : Across nonlinearities and hyperparameters, NTK tends to have higher fraction of variance attributed to higher degrees than CK.

In (a), we give several examples of the fractional variance curves for relu CK and NTK across several representative hyperparameters.

In (b), we do the same for erf CK and NTK.

In both cases, we clearly see that, while for degree 0 or 1, the fractional variance is typically higher for CK, the reverse is true for larger degrees.

In (c), for each degree k, we plot the fraction of hyperparameters where the degree k fractional variance of NTK is greater than that of CK.

Consistent with previous observations, this fraction increases with the degree.

Brighter color indicates higher variance, and we see the optimal depth for each degree k clearly increases with k for relu NTK, and likewise for odd degrees of erf NTK.

However, note that as k increases, the difference between the maximal fractional variance and those slightly suboptimal becomes smaller and smaller, reflected by suppressed range of color moving to the right.

The heatmaps for relu and erf CKs look similar and are omitted.

We verify this increase of optimal depth with degree in Fig. 3 (b).

There we have trained relu networks of varying depth against a ground truth multilinear polynomial of varying degree.

We see clearly that the optimal depth is increasing with degree.

We also verify this phenomenon when the input distribution changes to the standard Gaussian or the uniform distribution over the sphere

Note that implicit in our results here is a highly nontrivial observation: Past some point (the optimal depth), high depth can be detrimental to the performance of the network, beyond just the difficulty to train, and this detriment can already be seen in the corresponding NTK or CK.

In particular, it's not true that the optimal depth is infinite.

We confirm the existence of such an optimal depth even in real distributions like MNIST and CIFAR10; see Fig. 7 .

This adds significant nuance to the folk wisdom that "depth increases expressivity and allows neural networks to learn more complex features."

We generally find the degree k fractional variance of NTK to be higher than that of CK when k is large, and vice versa when k is small, as shown in Fig. 4 .

This means that, if we train only the last layer of a neural network (i.e. CK dynamics), we intuitively should expect to learn simpler features better, while, if we train all parameters of the network (i.e. NTK dynamics), we should expect to learn more complex features better.

Similarly, if we were to sample a function from a Gaussian process with the CK as kernel (recall this is just the distribution of randomly initialized infinite width MLPs ), this function is more likely to be accurately approximated by low degree polynomials than the same with the NTK.

We verify this intuition by training a large number of neural networks against ground truth functions of various homogeneous polynomials of different degrees, and show a scatterplot of how training the last layer only measures against training all layers ( Fig. 3(c) ).

This phenomenon remains true over the standard Gaussian or the uniform distribution on the sphere (Fig. 8) .

Consistent with our theory, the only place training the last layer works meaningfully better than training all layers is when the ground truth is a constant function.

However, we reiterate that fractional variance is an imperfect indicator of performance.

Even though for erf neural networks and k ≥ 1, degree k fractional variance of NTK is not always greater than that of the CK, we do not see any instance where training the last layer of an erf network is better than training all layers.

We leave an investigation of this discrepancy to future work.

In any setup that tries to push deep learning benchmarks, learning rate tuning is a painful but indispensable part.

In this section, we show that our spectral theory can accurately predict the maximal nondiverging learning rate over real datasets as well as toy input distributions, which would help set the correct upper limit for a learning rate search.

By Jacot et al. (2018) , in the limit of large width and infinite data, the function g : X → R represented by our neural network evolves like

when trained under full batch GD (with the entire population) with L2 loss

2 , ground truth g * , and learning rate α, starting from randomly initialization.

If we train only the last layer, then K is the CK; if we train all layers, then K is the NTK.

Given an eigendecomposition of K as in Eq. (1)

Consequently, we must have α < (max i λ i ) −1 in order for Eq. (10) When the input distribution is the uniform distribution over d , the maximum learning rate is max(µ 0 , µ 1 ) by Thm 4.1.

By Thm 5.1, as long as the Φ function corresonding to K has Φ(0) = 0,

Therefore, we should predict

for the maximal learning rate when training on the boolean cube.

However, as Fig. 5 shows, this prediction is accurate not only for the boolean cube, but also over the sphere, the standard Gaussian, and even MNIST and CIFAR10!

In this work, we have taken a first step at studying how hyperparameters change the initial distribution and the generalization properties of neural networks through the lens of neural kernels and their spectra.

We obtained interesting insights by computing kernel eigenvalues over the boolean cube and relating them to generalization through the fractional variance heuristic.

While it inspired valid predictions that are backed up by experiments, fractional variance is clearly just a rough indicator.

We hope future work can refine on this idea to produce a much more precise prediction of test loss.

Nevertheless, we believe the spectral perspective is the right line of research that will not only shed light on mysteries in deep learning but also inform design choices in practice.

Boolean cube theory predicts max learning rate for real datasets Figure 5 : Spectral theory of CK and NTK over boolean cube predicts max learning rate for SGD over real datasets MNIST and CIFAR10 as well as over boolean cube 128 , the sphere √ 128S 128−1 , and the standard Gaussian N (0, I 128 ).

In all three plots, for different depth, nonlinearity, σ 2 w , σ 2 b of the MLP, we obtain its maximal nondiverging learning rate ("max learning rate") via binary search.

We center and normalize each image of MNIST and CIFAR10 to the

sphere, where d = 28 2 = 784 for MNIST and d = 3 × 32 2 = 3072 for CIFAR10.

See Appendix E.2 for more details.

(a) We empirically find max learning rate for training only the last layer of an MLP.

Theoretically, we predict 1/Φ(0) where Φ corresponds to the CK of the MLP.

We see that our theoretical prediction is highly accurate.

Note that the Gaussian and Sphere points in the scatter plot coincide with and hide behind the BoolCube points.

(b) and (c) We empirically find max learning rate for training all layers.

Theoretically, we predict 1/Φ(0) where Φ corresponds to the NTK of the MLP.

The points are identical between (b) and (c), but the color coding is different.

Note that the Gaussian points in the scatter plots coincide with and hide behind the Sphere points.

In (b) we see that our theoretical prediction when training all layers is not as accurate as when we train only the last layer, but it is still highly correlated with the empirical max learning rate.

It in general underpredicts, so that half of the theoretical learning rate should always have SGD converge.

This is expected, since the NTK limit of training dynamics is only exact in the large width limit, and larger learning rate just means the training dynamics diverges from the NTK regime, but not necessarily that the training diverges.

In (c), we see that deeper networks tend to accept higher learning rate than our theoretical prediction.

If we were to preprocess MNIST and CIFAR10 differently, then our theory is less accurate at predicting the max learning rate; see Fig. 9

The Gaussian process behavior of neural networks was found by Neal (1995) for shallow networks and then extended over the years to different settings and architectures (Williams, 1997; Le Roux and Bengio, 2007; Hazan and Jaakkola, 2015; Daniely et al., 2016; Matthews et al., 2018; .

This connection was exploited implicitly or explicitly to build new models (Cho and Saul, 2009; Lawrence and Moore, 2007; Damianou and Lawrence, 2013; Wilson et al., 2016a; b; Bradshaw et al., 2017; van der Wilk et al., 2017; Kumar et al., 2018; Blomqvist et al., 2018; Borovykh, 2018; Garriga-Alonso et al., 2018; .

The Neural Tangent Kernel is a much more recent discovery by Jacot et al. (2018) (2018) came upon the same reasoning independently.

Like CK, NTK has also been applied toward building new models or algorithms (Arora et al., 2019a; Achiam et al., 2019) .

Closely related to the discussion of CK and NTK is the signal propagation literature, which tries to understand how to prevent pathological behaviors in randomly initialized neural networks when they are deep (Poole et al., 2016; Yang and Schoenholz, 2017; Hanin, 2018; Hanin and Rolnick, 2018; Chen et al., 2018; Pennington et al., 2017a; Hayou et al., 2018; Philipp and Carbonell, 2018) .

This line of work can trace its original at least to the advent of the Glorot and He initialization schemes for deep networks (Glorot and Bengio, 2010; He et al., 2015) .

The investigation of forward signal propagation, or how random neural networks change with depth, corresponds to studying the infinite-depth limit of CK, and the investigation of backward signal propagation, or how gradients of random networks change with depth, corresponds to studying the infinite-depth limit of NTK.

Some of the quite remarkable results from this literature includes how to train a 10,000 layer CNN (Xiao et al., 2017) and that, counterintuitively, batch normalization causes gradient explosion .

This signal propagation perspective can be refined via random matrix theory (Pennington et al., 2017a; .

In these works, free probability is leveraged to compute the singular value distribution of the input-output map given by the random neural network, as the input dimension and width tend to infinity together.

Other works also investigate various questions of neural network training and generalization from the random matrix perspective (Pennington and Worah, 2017; Pennington and Worah, 2018 ).

Yang (2019) presents a common framework, known as Tensor Programs, unifying the GP, NTK, signal propagation, and random matrix perspectives, as well as extending them to new scenarios, like recurrent neural networks.

It proves the existence of and allows the computation of a large number of infinite-width limits (including ones relevant to the above perspectives) by expressing the quantity of interest as the output of a computation graph and then manipulating the graph mechanically.

Several other works also adopt a spectral perspective on neural networks (Candès, 1999; Sonoda and Murata, 2017; Eldan and Shamir, 2016; Barron, 1993; Xu, 2018) ; here we highlight a few most relevant to us.

Rahaman et al. (2018) studies the real Fourier frequencies of relu networks and perform experiments on real data as well as synthetic ones.

They convincingly show that relu networks learn low frequencies components first.

They also investigate the subtleties when the data manifold is low dimensional and embedded in various ways in the input space.

In contrast, our work focuses on the spectra of the CK and NTK (which indirectly informs the Fourier frequencies of a typical network).

Nevertheless, our results are complementary to theirs, as they readily explain the low frequency bias in relu that they found.

Karakida et al. (2018) studies the spectrum of the Fisher information matrix, which share the nonzero eigenvalues with the NTK.

They compute the mean, variance, and maximum of the eigenvalues Fisher eigenvalues (taking the width to infinity first, and then considering finite amount of data sampled iid from a Gaussian).

In comparison, our spectral results yield all eigenvalues of the NTK (and thus also all nonzero eigenvalues of the Fisher) as well as eigenfunctions.

Finally, we note that several recent works (Xie et al., 2016; Bietti and Mairal, 2019; Basri et al., 2019; Ghorbani et al., 2019 ) studied one-hidden layer neural networks over the sphere, building on Smola et al. (2001) 's observation that spherical harmonics diagonalize dot product kernels, with the latter two concurrent to us.

This is in contrast to the focus on boolean cube here, which allows us to study the fine-grained effect of hyperparameters on the spectra, leading to a variety of insights into neural networks' generalization properties.

Using the spectral theory we developed in this paper, we made three observations, that can be roughly summarized as follows: 1) the simplicity bias noted by Valle-Pérez et al. (2018) is not universal; 2) for each function of fixed "complexity" there is an optimal depth such that networks shallower or deeper will not learn it as well; 3) training last layer only is better than training all layers when learning "simpler" features, and the opposite is true for learning "complex" features.

In this section, we discuss the applicability of these observations to distributions that are not uniform over the boolean cube: in particular, the uniform distribution over the sphere

, as well as realistic data distributions such as MNIST and CIFAR10.

Simplicity bias The simplicity bias noted by Valle-Pérez et al. (2018) , in particular Fig. 1 , depends on the finiteness of the boolean cube as a domain, so we cannot effectively test this on the distributions above, which all have uncountable support.

Optimal depth With regard to the second observation, we can test whether an optimal depth exists for learning functions over the distributions above.

Since polynomial degrees remain the natural indicator of complexity for the sphere and the Gaussian (see Appendices H.2 and H.3 for the relevant spectral theory), we replicated the experiment in Fig. 3 (b) for these distributions, using the same ground truth functions of polynomials of different degrees.

The results are shown in Fig. 6 .

We see the same phenomenon as in the boolean cube case, with an optimal depth for each degree, and with the optimal depth increasing with degree.

For MNIST and CIFAR10, the notion of "feature complexity" is less clear, so we will not test the hypothesis that "optimal depth increases with degree" for these distributions but only test for the existence of the optimal depth for the ground truth marked by the labels of the datasets.

We do so by training a large number of MLPs of varying depth on these datasets until convergence, and plot the results in Fig. 7 .

This figure clearly shows that such an optimal depth exists, such that shallower or deeper networks do monotonically worse as the depth diverge away from this optimal depth.

Again, the existence of the optimal depth is not obvious at all, as conventional deep learning wisdom would have one believe that adding depth should always help.

Training last layer only vs training all layers Finally, we repeat the experiment in Fig. 3(c) for the sphere and the standard Gaussian, with polynomials of different degrees as ground truth functions.

The results are shown in Fig. 8 .

We see the same phenomenon as in the boolean cube case: for degree 0 polynomials, training last layer works better in general, but for higher degree polynomials, training all layers fares better.

Note that, unlike the sphere and the Gaussian, whose spectral theory tells us that (harmonic) polynomial degree is a natural notion of complexity, for MNIST and CIFAR10 we have much less clear idea of what a "complex" or a "simple" feature is.

Therefore, we did not attempt a similar experiment on these datasets.

In the main text Fig. 5 , on the MNIST and CIFAR10 datasets, we preprocessed the data by centering and normalizing to the sphere (see Appendix E.2 for a precise description).

With this preprocessing, our theory accurately predicts the max learning rate in practice.

In general, if we go by another preprocessing, such as PCA or ZCA, or no preprocessing, our theoretical max learning rate 1/Φ(0) is less accurate but still correlated in general.

The only exception seems to be relu networks on PCA-or ZCA-preprocessed CIFAR10.

See Fig. 9 . (dashed lines), where d = 128.

We also compare against the results over the boolean cube from Fig. 3(b) , which are drawn with dotted lines.

Colors indicate the degrees of the ground truth polynomial functions.

The best validation loss for degree 0 to 2 are all very close no matter which distribution the input is sampled from, such that the curves all sit on top of each other.

For degree 3, there is less precise agreement between the validation loss over the different distributions, but the overall trend is unmistakably the same.

We see that for networks deeper or shallower than the optimal depth, the loss monotonically increases as the depth moves away from the optimum.

= 0 for all depths from 0 to 10.

We used SGD with learning rate 10 and batch size 256, and trained until convergence.

We record the best test error throughout the training procedure for each depth.

For each configuration, we repeat the randomly initialization and training for 10 random seeds to estimate the variance of the best test error.

The rows demonstrate the best test error over the course of training on CIFAR10 and MNIST, and the columns demonstrate the same for training only the last layer or training all layers.

As one can see, the best depth when training only the last layer is 1, for both CIFAR10 and MNIST.

The best depth when training all layers is around 5 for both CIFAR10 and MNIST.

Performance monotically decreases for networks shallower or deeper than the optimal depth.

Note that we have reached the SOTA accuracy for MLPs reported in on CIFAR10, and within 1 point of their accuracy on MNIST.

on MNIST and CIFAR10 preprocessed in different ways.

See Appendix E.2 for experimental details.

The first row compares the theoretical and empirical max learning rates when training only the last layer.

The second row compares the same when training all layers (under NTK parametrization (Eq. (MLP))).

The three columns correspond to the different preprocessing procedures: no preprocessing, PCA projection to the first 128 components (PCA128), and ZCA projection to the first 128 components (ZCA128).

In general, the theoretical prediction is less accurate (compared to preprocessing by centering and projecting to the sphere, as in Fig. 5 ), though still well correlated with the empirical max learning rate.

The most blatant caveat is the relu networks trained on PCA128-and ZCA128-processed CIFAR10. , log 2 (depth))-space achieving this value in the corresponding color.

The closer to blue the color, the higher the value.

Note that the contour for the highest values in higher degree plots "floats in mid-air", implying that there is an optimal depth for learning features of that degree that is not particularly small nor particularly big.

Fig. 3(a) , (b) and (c) differ in the set of hyperparameters they involve (to be specified below), but in all of them, we train relu networks against a randomly generated ground truth multilinear polynomial, with input space

Training We perform SGD with batch size 1000.

In each iteration, we freshly sample a new batch, and we train for a total of 100,000 iterations, so the network potentially sees 10 8 different examples.

At every 1000 iterations, we validate the current network on a freshly drawn batch of 10,000 examples.

We thus record a total of 100 validation losses, and we take the lowest to be the "best validation loss."

Generating the Ground Truth Function The ground truth function f * (x) is generated by first sampling 10 monomials m 1 , . . .

, m 10 of degree k, then randomly sampling 10 coefficients a 1 , . . .

, a 10 for them.

The final function is obtained by normalizing {a i } such that the sum of their squares is 1:

Hyperparameters for Fig. 3 (a)

• The learning rate is half the theoretical maximum learning rate • Ground truth degree k ∈ {0, 1, 2, 3}

• Depth ∈ {0, . . . , 10}

• activation = relu

• 10 random seeds per hyperparameter combination

• training last layer (marked "ck"), or all layers (marked "ntk").

In the latter case, we use the NTK parametrization of the MLP (Eq. (MLP)).

Fig. 3(b)

• The learning rate is half the theoretical maximum learning rate

• Ground truth degree k ∈ {0, 1, 2, 3}

• Depth ∈ {0, . . . , 10}

• activation = relu

• 100 random seeds per hyperparameter combination

• training last layer weight and bias only 7 Note that, because the L2 loss here is

2 , the maximum learning rate is λ

then the maximum learning rate would be 2λ Fig. 3(c)

• The learning rate ∈ {0.05, 0.1, 0.5} • Ground truth degree k ∈ {0, 1, . . . , 6}

• Depth ∈ {1, . . . , 5}

• activation ∈ {relu, erf} • σ Theoretical max learning rate For a fixed setup, we compute Φ according to Eq. (CK) (if only last layer is trained) or Eq. (NTK) (if all layers are trained).

For ground truth problems where the output is n-dimensional, the theoretical max learning rate is nΦ(0) −1 ; in particular, the max learning rates for MNIST and CIFAR10 are 10 times those for boolean cube, sphere, and Gaussian.

This is because the kernel for an multi-output problem effectively becomes

where the 1 n factor is due to the 1 n factor in the scaled square loss L(f, f

n times the top eigenvalue for K.

Empirical max learning rate For a fixed setup, we perform binary search for the empirical max learning rate as in Algorithm 1.

Preprocessing In Fig. 5 , for MNIST and CIFAR10, we center and project each image onto the sphere

, where d = 28 × 28 = 784 for MNIST and d = 3 × 32 × 32 = 3072 for CIFAR10.

More precisely, we compute the average imagex over the entire dataset, and we preprocess each image x as √ d

x−x x−x .

In Fig. 9 , there are three different preprocessing schemes.

For "no preprocessing," we load the MNIST and CIFAR10 data as is.

In "PCA128," we take the top 128 eigencomponents of the data, so that the data has only 128 dimensions.

In "ZCA128," we take the top 128 eigencomponents but rotate it back to the original space, so that the data still has dimension d, where d = 28×28 = 784 for MNIST and d = 3 × 32 × 32 = 3072 for CIFAR10.

• Target function: For boolean cube, sphere, and standard Gaussian, we randomly sample a degree 1 polynomial as in Eq. (11).

For MNIST and CIFAR10, we just use the label in the dataset, encoded as a one-hot vector for square-loss regression.

• Depth ∈ {1, 2, 4, 8, 16}

• activation ∈ {relu, erf} • σ 2 w = 2 for relu, but σ 2 w ∈ {1, 2, . . .

, 5} for erf • σ 2 b ∈ {1, . . . , 4}

• width = 1000 • 1 random seed per hyperparameter combination • Training last layer (CK) or all layers (NTK).

In the latter case, we use the NTK parametrization of the MLP (Eq. (MLP)).

Conjugate Kernel Via a central-limit-like intuition, each unit h l (x) α of Eq. (MLP) should behave like a Gaussian as width n l−1 → ∞, as it is a sum of a large number of roughly independent random variables (Poole et al., 2016; Yang and Schoenholz, 2017) .

The devil, of course, is in what "roughly independent" means and how to apply the central limit theorem (CLT) to this setting.

It can be done, however, Matthews et al., 2018; , and in the most general case, using a "Gaussian conditioning" technique, this result can be rigorously generalized to almost any architecture Yang (2019) .

In any case, the consequence is that, for any finite set S ⊆ X ,

as min{n 1 , . . .

, n l−1 } → ∞, where Σ l is the CK as given in Eq. (CK).

Neural Tangent Kernel By a slightly more involved version of the "Gaussian conditioning" technique, Yang (2019) also showed that, for any x, y ∈ X ,

as the widths tend to infinity, where Θ l is the NTK as given in Eq. (NTK).

For certain φ like relu or erf, V φ and V φ can be evaluated very quickly, so that both the CK and NTK can be computed in O(|X | 2 L) time, where X is the set of points we want to compute the kernel function over, and L is the number of layers.

Fact F.1 (Cho and Saul (2009) Neal (1995) ).

For any kernel K,

Under review as a conference paper at ICLR 2020

Remarkably, the NTK governs the evolution of the neural network function under gradient descent in the infinite-width limit.

First, let's consider how the parameters θ and the neural network function f evolve under continuous time gradient flow.

Suppose f is only defined on a finite input space X = {x 1 , . . .

, x k }.

We will visualize

(best viewed in color).

Then under continuous time gradient descent with learning rate η,

where

is of course the (finite width) NTK.

These equations can be visualized as

Thus f undergoes kernel gradient descent with (functional) loss L(f ) and kernel Θ t .

This kernel Θ t of course changes as f evolves, but remarkably, it in fact stays constant for f being an infinitely wide MLP (Jacot et al., 2018) :

where Θ is the infinite-width NTK corresponding to f .

A similar equation holds for the CK Σ if we train only the last layer,

If L is the square loss against a ground truth function

, and the equations above become linear differential equations.

However, typically we only have a training set X train ⊆ X of size far less than |X |.

In this case, the loss function is effectively

with functional gradient

where D train is a diagonal matrix of size k × k whose diagonal is 1 on x ∈ X train and 0 else.

Then our function still evolves linearly

where K is the CK or the NTK depending on which parameters are trained.

Recall that the initial f 0 in Eq. (13) is distributed as a Gaussian process N (0, Σ) in the infinite width limit.

As Eq. (13) is a linear differential equation, the distribution of f t will remain a Gaussian process for all t, whether K is CK or NTK.

Under suitable conditions, it can be shown that (Lee et al., 2019) , in the limit as t → ∞, if we train only the last layer, then the resulting function f ∞ is distributed as a Gaussian process with meanf ∞ given bȳ

and kernel Var f ∞ given by

These formulas precisely described the posterior distribution of f given prior N (0, Σ) and data

If we train all layers, then similarly as t → ∞, the function f ∞ is distributed as a Gaussian process with meanf ∞ given by (Lee et al., 2019)

This is, again, the mean of the Gaussian process posterior given prior N (0, Θ) and the training data {(x, f * (x))} x∈X train .

However, the kernel of f ∞ is no longer the kernel of this posterior, but rather is an expression involving both the NTK Θ and the CK Σ; see Lee et al. (2019) .

In any case, we can make the following informal statement in the limit of large width Training the last layer (resp.

all layers) of an MLP infinitely long, in expectation, yields the mean prediction of the GP inference given prior N (0, Σ) (resp.

N (0, Θ)).

In this section, we briefly review the theory of Hilbert-Schmidt kernels, and more importantly, to properly define the notion of eigenvalues and eigenfunctions.

A function K :

HS is known as the Hilbert-Schmidt norm of K. K is called symmetric if K(x, y) = K(y, x) and positive definite (resp.

semidefinite) if

A spectral theorem (Mercer's theorem) holds for Hilbert-Schmidt operators.

Fact G.1.

If K is a symmetric positive semidefinite Hilbert-Schmidt kernel, then there is a sequence of scalars λ i ≥ 0 (eigenvalues) and functions f i ∈ L 2 (X ) (eigenfunctions), for i ∈ N, such that

where the convergence is in L 2 (X × X ) norm.

This theorem allows us to speak of the eigenfunctions and eigenvalues, which are important for training and generalization considerations when K is a kernel used in machine learning, as discussed in the main text.

A sufficient condition for K to be a Hilbert-Schmidt kernel in our case (concerning only probability measure on X ) is just that K is bounded.

All Ks in this paper satisfy this property.

From the Fourier Series Perspective.

We continue from the discussion of the boolean cube in the main text.

Recall that T ∆ is the shift operator on functions that sends Φ(·) to Φ(· − ∆).

Notice that, if we let Φ(t) = e κt for some κ ∈ C, then T ∆ Φ(s) = e −κ∆ · e κt .

Thus Φ is an "eigenfunction" of the operator T ∆ with eigenvalue e −κ∆ .

In particular, this implies that

2 , as in the case when K is the CK or NTK of a 1-layer neural network with nonlinearity exp(·/σ), up to multiplicative constant (Fact F.3).

Then the eigenvalue µ k over the boolean cube

where ∆ = 2/d.

It would be nice if we can express any Φ as a linear combination of exponentials, so that Eq. (5) simplifies in the fashion of Prop H.1 -this is precisely the idea of Fourier series.

We will use the theory of Fourier analysis on the circle, and for this we need to discuss periodic functions.

LetΦ : [−2, 2] → R be defined as

where the supremum is taken over all partitions P of the interval [a, b], P = {x 0 , . . .

, x n P },

Intuitively, a function of bounded variation has a graph (in [a, b] × R) of finite length.

Fact H.3 (Katznelson (2004) ).

A bounded variation function f : [−2, 2] → R that is periodic (i.e. f (−2) = f (2)) has a pointwise-convergent Fourier series: whenever both sides are well defined.

If Ψ is continuous and has bounded variation then T ∆ Ψ is also continuous and has bounded variation, and thus its Fourier series, the RHS above, converges pointwise to T ∆ Ψ.

Expressing the LHS in Fourier basis, we obtain Theorem H.5.

(t)e Recovering the values of Φ given the eigenvalues µ 0 , . . .

, µ d .

Conversely, given eigenvalues µ 0 , . . .

, µ d corresponding to each monomial degree, we can recover the entries of the matrix K. If x and y differ on a set T ⊆

[d], then we can simplify the inner sum

Remark H.7.

If we let T be the operator that sends µ • → µ •+1 , then we have the following operator expression

Remark H.8.

The above shows that the matrix

H.2 SPHERE Now let's consider the case when X = √ dS d−1 is the radius-√ d sphere in R d equipped with the uniform measure.

Again, because x ∈ X all have the same norm, we will consider Φ as a univariate function with K(x, y) = Φ( x, y / x y ) = Φ( x, y /d).

As is long known (Schoenberg, 1942; Gneiting, 2013; Xu and Cheney, 1992; Smola et al., 2001) , K is diagonalized by spherical harmonics.

We review these results briefly below, as we will build on them to deduce spectral information of K on isotropic Gaussian distributions.

Review: spherical harmonics and Gegenbauer polynomials.

Spherical harmonics are L 2 functions on S d−1 that are eigenfunctions of the Laplace-Beltrami operator ∆ S d−1 of S d−1 .

They can be described as the restriction of certain homogeneous polynomials in

the space of spherical harmonics of degree l on sphere

There is a special class of spherical harmonics called zonal harmonics that can be represented as x → p( x, y ) for specific polynomials p : R → R, and that possess a special reproducing property which we will describe shortly.

Intuitively, the value of any zonal harmonics only depends on the "height" of x along some fixed axis y, so a typical zonal harmonics looks like Fig. 16 .

The polynomials p must be one of the Gegenbauer polynomials.

Gegenbauer polynomials {C Fig. 15 for examples), and here we adopt the convention that

Then for each (oriented) axis y ∈ S d−1 and degree l, there is a unique zonal harmonic Fact H.9 (Reproducing property (Suetin) ).

For any f ∈ H d−1,(m) ,

We also record a useful fact about Gegenbauer polynomials.

Fact H.10 (Suetin).

By a result of Schoenberg (1942) , we have the following eigendecomposition of K on the sphere.

Theorem H.11 (Schoenberg) .

2 −1 ), so that it has the Gegenbauer expansion

For completeness, we include the proof of this theorem in Appendix I.

By Bezubik et al. (2008), we can express the Gegenbauer coefficients, and thus equivalently the eigenvalues, via derivatives of Φ: Theorem H.12 (Bezubik et al. (2008)).

If the Taylor expansion of Φ at 0,

is absolutely convergent on the closed interval [−1, 1], then the Gegenbauer coefficients a l in Thm H.11 in dimension d is equal to the absolute convergent series

As the dimension d of the sphere tends to ∞, the eigenvalues in fact simplify to the derivatives of Φ:

Theorem H.13.

Let K be the CK or NTK of an MLP on the sphere This theorem is the same as Thm I.6 except that it concerns the sphere rather than the boolean cube.

Proof.

By Thm I.3, Φ's Taylor expansion around 0 is absolutely convergent on [−1, 1], so that the condition of Thm H.12 is satisfied.

Therefore, Eq. (15) holds and is absolutely convergent.

By dominated convergence theorem, we can exchange the limit and the summation, and get

as desired.

Now let's consider X = R d equipped with standard isotropic Gaussian N (0, I), so that K behaves like

for any f ∈ L 2 (N (0, I)).

In contrast to the previous two sections, K will essentially depend on the effect of the norms x and y on Φ.

Note that an isotropic Gaussian vector z ∼ N (0, I) can be sampled by independently sampling its direction v uniformly from the sphere S d−1 and sampling its magnitude r from a chi distribution χ d with d degrees of freedom.

Proceeding along this line of logic yields the following spectral theorem: Theorem H.14.

forms a positive semidefinite Hilbert-Schmidt operator on L 2 (N (0, I)) iff Φ can be decomposed as

where

2 ) l (t) are Gegenbauer polynomials as in Appendix H.2, with C

2 −1 ).

• and A l are positive semidefinite Hilbert-Schmidt kernels on

, the L 2 space over the probability measure of a χ 2 d -variable divided by d, and with A l denoting the HilbertSchmidt norm of A l .

In addition, K is positive definite iff all A l are.

See Appendix I for a proof.

As a consequence, K has an eigendecomposition as follows under the standard Gaussian measure in d dimensions.

Corollary H.15.

Suppose K and Φ are as in Thm H.14 and K is a positive semidefinite HilbertSchmidt operator, so that Eq. (16) holds, with Hilbert-Schmidt kernels A l .

Let A l have eigendecomposition

for eigenvalues a li ≥ 0 and eigenfunctions

eigenvalues {a li : l, i ∈ [0, ∞)}, and each eigenvalue a li corresponds to the eigenspace

where

is the space of degree l spherical harmonics on the unit sphere

For certain simple F , we can obtain {A l } l≥0 explicitly.

For example, suppose K is degree-s positive-homogeneous, in the sense that, for a, b > 0,

This happens when K is the CK or NTK of an MLP with degree-s positive-homogeneous.

Then it's easy to see that Φ(t, q, q ) = (qq ) sΦ (t) for someΦ : [−1, 1] → R, and

where {a l } l are the Gegenbauer coefficients ofΦ,

We can then conclude with the following theorem.

Theorem H.16.

be the Gegenbauer expansion ofΦ. Also define

Then over the standard Gaussian in R d , K has the following eigendecomposition

is an eigenspace with eigenvalue λ 2 a l .

• For any S ∈ L 2 (

Proof.

The A l in Eq. (16) for K are all equal to

This is a rank 1 kernel (on

, with eigenfunction R/λ and eigenvalue λ 2 .

The rest then follows straightforwardly from Thm H.11.

A common example where Thm H.16 applies is when K is the CK of an MLP with relu, or more generally degree-s positive homogeneous activation functions, so that the R in Thm H.16 is a polynomial.

In general, we cannot expect K can be exactly diagonalized in a natural basis, as {A l } l≥0 cannot even be simultaneously diagonalizable.

We can, however, investigate the "variance due to each degree of spherical harmonics" by computing

which is the coefficient of Gegenbauer polynomials in

Proposition H.17.

Assume that Φ(t, q, q ) is continuous in q and q .

Suppose for any d and any t ∈ [−1, 1], the random variable Φ(t, q, q ) with q,

Proof.

By the strong law of large number, d −1 χ 2 d converges to 1 almost surely.

Because Φ(t, q, q ) is continuous in q and q , almost surely we have Φ(t, q, q ) → Φ(t, 1, 1) almost surely.

Since Φ is bounded by Y , by dominated convergence, we havê

In the final part of this section, we show that in the limit of large input dimension d, the top eigenvalues of K over the standard Gaussian can be easily described (Thm H.19) .

First, we need to specify some conditions on Φ. Definition H.18.

Φ :

• There is a Taylor expansion in t,

that is absolutely convergent on (t, q, q ) ∈ [−1, 1] × R + × R + , and such that each B is smooth (C ∞ ) on (q, q ) ∈ R + × R + .

• for all (t, q, q ) ∈ [−1, 1] × R + × R + , and for any l ∈ [0, ∞), we have |B l (t, q, q )| ≤ C(1 + |q| r + |q | r ) for some constants C, r > 0 that may depend on l but not on t, q, q .

Theorem H.19.

Suppose K is a the CK or NTK of an MLP with polynomially bounded activation function.

For every degree l, K over N (0, I d ) has an eigenvalue a l0 at spherical harmonics degree l (in the sense of Cor H.15) with

where A l is as in Eq. (18).

Furthermore,

Here Φ (l) is the lth derivative of Φ(t, q, q ) against t.

Proof.

By Lem H.20 below, Φ is reasonable.

Let (19) , and let

.

Let a li be the ith largest eigenvalue of A l as in Eq. (18), with a l0 being the largest.

Note that all of these quantities a l , b l , a li depend on d, but we suppress this notationally.

We seek to prove the following claims:

Claim 1: a l0 ≤ a l .

First note that a l is the trace of the operator A l , so that a l = ∞ i=0 a li .

Thus, any eigenvalue a li is at most a l .

Claim 2: a l0 ≥ b l .

Now, by Min-Max theorem, the largest eigenvalue a l0 of A l is equal to

Claims 3 and 4.

Now note that we have the following equalities in the Hilbert space

2 −1 ), because of the absolute convergence in Eq. (17):

2 ) (t).

Thus, by Eq. (15),

By the absolute convergence of Eq. (21), differentiation commutes with expectation:

where B l+2k is as in Eq. (21).

Furthermore, as d → ∞, because Φ (l+2k) (t, q, q ) is smooth and polynomially bounded in q and q , while 10 -6 Figure 18 : In high dimension d, the eigenvalues are very close for the kernel over the boolean cube, the sphere, and standard Gaussian.

We plot the eigenvalues µ k of the erf CK, with σ 2 w = 2, σ 2 b = 0.001, depth 2, over the boolean cube, the sphere, as well as kernel on the sphere induced bŷ Φ d (Eq. (20) ).

We do so for each degree k ≤ 5 and for dimensions d = 16, 32, 64, 128.

We see that by dimension d = 128, the eigenvalues shown are already very close to each other.

If we fix k and let the input dimension d → ∞, then the fractional variance of degree k converges to

Practically speaking, only the top eigenvalues matter.

Observe that, in the empirical and theoretical results above, we only verify that the top eigenvalues (µ k , a k , or a k0 for k small compared to d) are close when d is large.

While this result may seem very weak at face value, in practice, the closeness of these top eigenvalues is the only thing that matters.

Indeed, in machine learning, we will only ever have a finite number, say N , of training samples to work with.

Thus, we can only use a finite N × N submatrix of the kernel K. This submatrix, of course, has only N eigenvalues.

Furthermore, if these samples are collected in an iid fashion (as is typically assumed), then these eigenvalues approximate the largest N eigenvalues (top N counting multiplicity) of the kernel K itself (Tropp, 2015) .

As such, the smaller eigenvalues of K can hardly be detected in the training sample, and cannot affect the machine learning process very much.

Let's discuss a more concrete example: Fig. 18 shows that the boolean cube eigenvalues µ k are very close to the sphere eigenvalues a k for all k ≤ 5.

Over the boolean cube, µ 0 , . . . , µ 5 cover eigenspaces of total dimension

, which is 275,584,033 when d = 128.

We need at least that many samples to be able to even detect the eigenvalue µ 6 and the possible difference between it and the sphere eigenvalue a 6 .

But note in comparison, Imagenet, one of the most common large datasets in use today, has only about 15 million samples, 10 times less than the number above.

Additionally, in this same comparison, d = 128 dramatically pales compared to Imagenet's input dimension 3 × 256 2 = 196608, and even to the those of the smaller common datasets like CIFAR10 (d = 3 × 32 2 = 3072) and MNIST (d = 24 2 = 576) -if we were to even use the input dimension of MNIST above, then µ 0 , . . .

, µ 5 would cover eigenspaces of 523 billion total dimensions!

Hence, it is quite practically relevant to consider the effect of large d on the eigenvalues, while keeping k small.

Again, we remark that even when one fixes k and increases d, the dimension of eigenspaces affected by our limit theorems Thms H.13, H.19 and I.6 increases like Θ(d k ), which implies one needs an increasing number Θ(d k ) of training samples to see the difference of eigenvalues in higher degrees k.

Finally, from the perspective of fractional variance, we also can see that only the top k spectral closeness matters: By Cor H.21, for any > 0, there is a k such that the total fractional variance of degree 0 to degree k (corresponding to eigenspaces of total dimension Θ(d k )) sums up to more than 1 − , for the cube, the sphere, and the standard Gaussian simultaneously, when d is sufficiently large.

This is because the asymptotic fractional variance is completely determined by the derivatives of Φ at t = 0.

where 1 = (1, . . . , 1) ∈ d .

This definition of µ |S| does not depend on the choice S, only on the cardinality of S. These are all of the eigenfunctions of K by dimensionality considerations.

Proof.

We directly verify Kχ S = µ |S| χ S .

Notice first that K(x, y) = Φ( x, y ) = Φ( x y, 1 ) = K(x y, 1)

where is Hadamard product.

We then calculate Kχ S (y) = E

by orthogonality = a n f (x/ √ d) by reproducing property.

Lemma 3.2.

With µ k as in Thm 3.1,

where

Proof.

Because i x i /d only takes on values {− Thus Φ resides in the tensor product space

and therefore can be expanded as Φ(t, q, q ) = Using the notation of Lem I.4, we can express this as

Substituting the Taylor expansion of andφ are nonnegative as well.

In addition, plugging in 1 for c in this Taylor series shows that the sum of the coefficients equalφ(1) = 1, so the series is absolutely convergent for c ∈ [−1, 1].

This proves the inductive step.

For the NTK, the proof is similar, except we now also have a product step where we multiply V φ (Σ l−1 ) with Θ l−1 , and we simply just need to use the fact that product of two Taylor series with nonnegative coefficients is another Taylor series with nonnegative coefficients, and that the resulting radius of convergence is the minimum of the original two radii of convergence.

J THE {0, 1} d VS THE {±1} d BOOLEAN CUBE Valle-Pérez et al. (2018) actually did their experiments on the {0, 1} d boolean cube, whereas here, we have focused on the {±1} d boolean cube.

As datasets are typically centered before feeding into a neural network (for example, using Pytorch's torchvision.transform.Normalize), {±1}

d is much more natural.

In comparison, using the {0, 1} d cube is equivalent to adding a bias in the input of a network and reducing the weight variance in the input layer, since any x ∈ {±1} d corresponds to 1 2 (x + 1) ∈ {0, 1}

d .

As such, one would expect there is more bias toward low frequency components with inputs from {0, 1}

d .

Nevertheless, here we verify that our observations of Section 4 above still holds over the {0, 1} d cube by repeating the same experiments as Fig. 1 in this setting (Fig. 19) .

Just like over the {±1} d cube, the relu network biases significantly toward certain functions, but with erf, and with increasing σ

@highlight

Eigenvalues of Conjugate (aka NNGP) and Neural Tangent Kernel can be computed in closed form over the Boolean cube and reveal the effects of hyperparameters on neural network inductive bias, training, and generalization.

@highlight

This paper gives a spectral analysis on neural networks' conjugate kernel and neural tangent kernel on boolean cube to resolve why deep networks are biased towards simple functions.
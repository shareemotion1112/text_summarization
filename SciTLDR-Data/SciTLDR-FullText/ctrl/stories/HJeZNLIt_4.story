Flow based models such as Real NVP are an extremely powerful approach to density estimation.

However, existing flow based models are restricted to transforming continuous densities over a continuous input space into similarly continuous distributions over continuous latent variables.

This makes them poorly suited for modeling and representing discrete structures in data distributions, for example class membership or discrete symmetries.

To address this difficulty, we present a normalizing flow architecture which relies on domain partitioning using locally invertible functions, and possesses both real and discrete valued latent variables.

This Real and Discrete (RAD) approach retains the desirable normalizing flow properties of exact sampling, exact inference, and analytically computable probabilities, while at the same time allowing simultaneous modeling of both continuous and discrete structure in a data distribution.

Latent generative models are one of the prevailing approaches for building expressive and tractable generative models.

The generative process for a sample x can be expressed as DISPLAYFORM0 where z is a noise vector, and g a parametric generator network (typically a deep neural network).

This paradigm has several incarnations, including variational autoencoders (Kingma & Welling, 2014; Rezende et al., 2014) , generative adversarial networks (Goodfellow et al., 2014) , and flow based models BID0 BID9 BID5 Kingma & Dhariwal, 2018; BID3 Grathwohl et al., 2019) .The training process and model architecture for many existing latent generative models, and for all published flow based models, assumes a unimodal smooth distribution over latent variables z. Given the parametrization of g as a neural network, the mapping to x is a continuous function.

This imposed structure makes it challenging to model data distributions with discrete structure -for instance, multi-modal distributions, distributions with holes, distributions with discrete symmetries, or distributions that lie on a union of manifolds (as may approximately be true for natural images, see BID11 .

Indeed, such cases require the model to learn a generator whose input Jacobian has highly varying or infinite magnitude to separate the initial noise source into different clusters.

Such variations imply a challenging optimization problem due to large changes in curvature.

This shortcoming can be critical as several problems of interest are hypothesized to follow a clustering structure, i.e. the distributions is concentrated along several disjoint connected sets (Eghbal-zadeh et al., 2018) .A standard way to address this issue has been to use mixture models BID16 Richardson & Weiss, 2018; Eghbal-zadeh et al., 2018) or structured priors (Johnson et al., 2016) .

In order to efficiently parametrize the model, mixture models are often formulated as a discrete latent variable models (Hinton & Salakhutdinov, 2006; BID4 Mnih & Gregor, 2014 ; van den Oord model (1c, 1d) .

Note the dependency of K on Z in 1d.

While this is not necessary, we will exploit this structure as highlighted later in the main text and in Figure 4 .

et al., 2017) , some of which can be expressed as a deep mixture model BID10 BID14 BID13 .

Although the resulting exponential number of mixture components with depth in deep mixture models is an advantage in terms of expressivity, it is an impediment to inference, evaluation, and training of such models, often requiring as a result the use of approximate methods like hard-EM or variational inference (Neal & Hinton, 1998) .In this paper we combine piecewise invertible functions with discrete auxiliary variables, selecting which invertible function applies, to describe a deep mixture model.

This framework enables a probabilistic model's latent space to have both real and discrete valued units, and to capture both continuous and discrete structure in the data distribution.

It achieves this added capability while preserving the exact inference, exact sampling, exact evaluation of log-likelihood, and efficient training that make standard flow based models desirable.

We aim to learn a parametrized distribution p X (x) on the continuous input domain R d by maximizing log-likelihood.

The major obstacle to training an expressive probabilistic model is typically efficiently evaluating log-likelihood.

If we consider a mixture model with a large number |K| of components, the likelihood takes the form DISPLAYFORM0 In general, evaluating the likelihood requires computing probabilities for all |K| components.

However, following a strategy similar to Rainforth et al. (2018) , if we partition the domain DISPLAYFORM1 , we can write the likelihood as DISPLAYFORM2 This transforms the problem of summation to a search problem x → f K (x).

This can be seen as the inferential converse of a stratified sampling strategy BID8 .

The proposed approach will be a direct extension of flow based models BID6 BID5 Kingma & Dhariwal, 2018) .

Flow based models enable log-likelihood (a) An example of a trimodal distribution pX , sinusoidal distribution.

The different modes are colored in red, green, and blue.(b) The resulting unimodal distribution pZ , corresponding to the distribution of any of the initial modes in pX .(c) An example fZ (x) of a piecewise invertible function aiming at transforming pZ into a unimodal distribution.

The red, green, and blue zones corresponds to the different modes in input space.

Figure 2: Example of a trimodal distribution (2a) turned into a unimodal distribution (2b) using a piecewise invertible function (2c).

Note that the initial distribution p X correspond to an unfolding of DISPLAYFORM0 evaluation by relying on the change of variable formula DISPLAYFORM1 with f Z a parametrized bijective function from R d onto R d and ∂f Z ∂x T the absolute value of the determinant of its Jacobian.

As also proposed in Falorsi et al. (2019) , we relax the constraint that f Z be bijective, and instead have it be surjective onto R d and piecewise invertible.

That is, we require f Z|A k (x) be an invertible function, where DISPLAYFORM2 we can define the following generative process: DISPLAYFORM3 If we use the set identification function f K associated with A k , the distribution corresponding to this stochastic inversion can be defined by a change of variable formula DISPLAYFORM4 Because of the use of both Real and Discrete stochastic variables, we call this class of model RAD.The particular parametrization we use on is depicted in Figure 2 .

We rely on piecewise invertible functions that allow us to define a mixture model of repeated symmetrical patterns, following a Figure 4: Illustration of the expressive power the gating distribution p K|Z provides.

By capturing the structure in a sine wave in p K|Z , the function z, k → x can take on an extremely simple form, corresponding only to a linear function with respect to z. DISPLAYFORM5 method of folding the input space.

Note that in this instance the function f K is implicitly defined by f Z , as the discrete latent corresponds to which invertible component of the piecewise function x falls on.

So far, we have defined a mixture of |K| components with disjoint support.

However, if we factorize p Z,K as p Z · p K|Z , we can apply another piecewise invertible map to Z to define p Z as another mixture model.

Recursively applying this method results in a deep mixture model (see FIG2 ).Another advantage of such factorization is in the gating network p K|Z , as also designated in (van den BID13 .

It provides a more constrainted but less sample wasteful approach than rejection sampling BID1 by taking into account the untransformed sample z before selecting the mixture component k. This allows the model to exploit the distribution p Z in different regions A k in more complex ways than repeating it as a patternm as illustrated in Figure 4 .The function from the input to the discrete variables, f K (x), contains discontinuities.

This presents the danger of introducing discontinuities into log p X (x), making optimization more difficult.

However, by carefully imposing boundary conditions on the gating network, we are able to exactly counteract the effect of discontinuities in f K , and cause log p X (x) to remain continuous with respect to the parameters.

This is discussed in detail in Appendix A.

We conduct a brief comparison on six two-dimensional toy problems with REAL NVP to demonstrate the potential gain in expressivity RAD models can enable.

Synthetic datasets of 10, 000 points each are constructed following the manifold hypothesis and/or the clustering hypothesis.

We designate these problems as: grid Gaussian mixture, ring Gaussian mixture, two moons, two circles, spiral, and many moons (see FIG3 ).

For the RAD model implementation, we use the piecewise linear activations defined in Appendix A in a coupling layer architecture BID5 for f Z where, instead of a conditional linear transformation, the conditioning variable x 1 determines the parameters of the piecewise linear activation on x 2 to obtain z 2 and k 2 , with z 1 = x 1 (see FIG4 ).

For the gating network p K|Z , the gating logit neural network s (z) take as input z = (z 1 , z 2 ).

We compare with a REAL NVP model using only affine coupling layers.

p Z is a standard Gaussian distribution.

As both these models can easily approximately solve these generative modeling tasks provided enough capacity, we study these model in a relatively low capacity regime, where we can showcase the potential expressivity RAD may provide.

Each of these models uses six coupling layers, and each coupling layer uses a one-hidden-layer rectified network with a tanh output activation scaled by a scalar parameter as described in Dinh et al. (2017) .

For RAD, the logit network s (·) also uses a one-hidden-layer rectified neural network, but with linear output.

In order to fairly compare with respect to number of parameters, we provide REAL NVP seven times more hidden units per (e) REAL NVP on spiral.(f) REAL NVP on many moons.(g) RAD on grid Gaussian mixture.(h) RAD on ring Gaussian mixture.(i) RAD on two moons.(j) RAD on two circles.(k) RAD on spiral.(l) RAD on many moons.

Figure 7: Comparison of samples from trained REAL NVP (top row) (a-f) and RAD (bottow row) (g-l) models.

REAL NVP fails in a low capacity setting by attributing probability mass over spaces where the data distribution has low density.

Here, these spaces often connect data clusters, illustrating the challenges that come with modeling multimodal data as one continuous manifold.hidden layer than RAD, which uses 8 hidden units per hidden layer.

For each level, p K|Z and f Z are trained using stochastic gradient ascent with ADAM (Kingma & Ba, 2015) on the log-likelihood with a batch size of 500 for 50, 000 steps.

In each of these problems, RAD is consistently able to obtain higher log-likelihood than REAL NVP.

We plot the samples (Figure 7 ) of the described RAD and REAL NVP models trained on these problems.

In the described low capacity regime, REAL NVP fails by attributing probability mass over spaces where the data distribution has low density.

This is consistent with the mode covering behavior of maximum likelihood.

However, the particular inductive bias of REAL NVP is to prefer modeling the data as one connected manifold.

This results in the unwanted probability mass being distributed along the space between clusters.

Flow-based models often follow the principle of Gaussianization (Chen & Gopinath, 2001), i.e. transforming the data distribution into a Gaussian.

The inversion of that process on a Gaussian distribution would therefore approximate the data distribution.

We plot in FIG7 the inferred Gaussianized variables z (5) for both models trained on the ring Gaussian mixture problem.

The Gaussianization from REAL NVP leaves some area of the standard Gaussian distribution unpopulated.

These unattended areas correspond to unwanted regions of probability mass in the input space.

RAD suffers significantly less from this problem.

An interesting feature is that RAD seems also to outperform REAL NVP on the spiral dataset.

One hypothesis is that the model successfully exploits some non-linear symmetries in this problem.

In RAD several points which were far apart in the input space become neighbors in z (5) .

This is not the case for REAL NVP.

We take a deeper look at the Gaussianization process involved in both models.

In FIG8 we plot the inference process of z (5) from x for both models trained on the two moons problem.

As a result of a folding process similar to that in Montufar et al. (2014) , several points which were far apart in the input space become neighbors in z (5) in the case of RAD.We further explore this folding process using the visualization described in FIG1 .

We verify that the non-linear folding process induced by RAD plays at least two roles: bridging gaps in the distribution of probability mass, and exploiting symmetries in the data.

We observe that in the case of the ring Gaussian mixture FIG1 ), RAD effectively uses foldings in order to bridge the different modes of the distribution into a single mode, primarily in the last layers of the transformation.

We contrast this with REAL NVP FIG1 ) which struggles to combine these modes under the standard Gaussian distribution using bijections.

In the spiral problem FIG1 ), RAD decomposes the spiral into three different lines to bridge FIG1 ) instead of unrolling the manifold fully, which REAL NVP struggles to do FIG1 ).In both cases, the points remain generally well separated by labels, even after being pushed through a RAD layer FIG1 .

This enables the model to maximize the conditional log-probability log(p K|Z ).

We introduced an approach to tractably evaluate and train deep mixture models using piecewise invertible maps as a folding mechanism.

This allows exact inference, exact generation, and exact evaluation of log-likelihood, avoiding many issues in previous discrete variables models.

This method can easily be combined with other flow based architectural components, allowing flow based models to better model datasets with discrete as well as continuous structure.

Figure 11: RAD and REAL NVP inference processes on the ring Gaussian mixture problem.

Each column correspond to a RAD or affine coupling layer.

RAD effectively uses foldings in order to bridge the multiple modes of the distribution into a single mode, primarily in the last layers of the transformation, whereas REAL NVP struggles to bring together these modes under the standard Gaussian distribution using continuous bijections.

A CONTINUITYThe standard approach in learning a deep probabilistic model has been stochastic gradient descent on the negative log-likelihood.

Although the model enables the computation of a gradient almost everywhere, the log-likelihood is unfortunately discontinuous.

Let's decompose the log-likelihood DISPLAYFORM0 There are two sources of discontinuity in this expression: f K is a function with discrete values (therefore discontinuous) and ∂f Z ∂x T is discontinuous because of the transition between the subsets A k , leading to the expression of interest DISPLAYFORM1 which takes a role similar to the log-Jacobian determinant, a pseudo log-Jacobian determinant.

Let's build from now on the simple scalar case and a piecewise linear function DISPLAYFORM2 In this case, s(z) = log p K|Z k | z k≤N + C1 |K| can be seen as a vector valued function.

We can attempt at parametrizing the model such that the pseudo log-Jacobian determinant becomes continuous with respect to β by expressing the boundary condition at x = β DISPLAYFORM3 ⇒s(−α 2 β) 2 + log(α 2 ) = s(−α 2 β) 3 + log(α 3 ).

DISPLAYFORM4 − log(α 1 ), log(α 2 ), log(α 3 ) + β 2 1 + cos (zα DISPLAYFORM5 Another type of boundary condition can be found at between the non-invertible area and the invertible area z = α 2 β, as ∀z > α 2 β, p 3|Z (3 | z) = 1, therefore DISPLAYFORM6 Since the condition ∀k < 3, p K|Z k | z) → 0 when z → (α 2 β) − will lead to an infinite loss barrier at x = −β, another way to enforce this boundary condition is by adding linear pieces FIG1 ): DISPLAYFORM7 The inverse is defined as DISPLAYFORM8 In order to know the values of s at the boundaries ±α 2 β, we can use the logit function DISPLAYFORM9 Given those constraints, the model can then be reliably learned through gradient descent methods.

Note that the resulting tractability of the model results from the fact that the discrete variables k is only interfaced during inference with the distribution p K|Z , unlike discrete variational autoencoders approaches (Mnih & Gregor, 2014; BID15 where it is fed to a deep neural network.

Similar to BID7 , the learning of discrete variables is achieved by relying on the the continuous component of the model, and, as opposed as other approaches (Jang et al., 2017; BID12 Grathwohl et al., 2018; BID12 , this gradient signal extracted is exact and closed form.

We plot the remaining inference processes of RAD and REAL NVP on the remaining problems not plotted previously: grid Gaussian mixture FIG1 , two circles FIG1 ), two moons FIG1 , and many moons FIG1 ).

We also compare the final results of the Gaussianization processes on both models on the different toy problems in FIG1 .

(e) REAL NVP on spiral.(f) REAL NVP on many moons.(g) RAD on grid Gaussian mixture.(h) RAD on ring Gaussian mixture.(i) RAD on two moons.(j) RAD on two circles.(k) RAD on spiral.(l) RAD on many moons.

FIG1 : Comparison of the Gaussianization from the trained REAL NVP (top row) (a-f) and RAD (bottow row) (g-l).

REAL NVP fails in a low capacity setting by leaving unpopulated areas where the standard Gaussian attributes probability mass.

Here, these spaces as often ones separating clusters, showing the failure in modeling the data as one manifold.

<|TLDR|>

@highlight

Flow based models, but non-invertible, to also learn discrete variables
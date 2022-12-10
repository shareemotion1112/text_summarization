This work presents the Poincaré Wasserstein Autoencoder, a reformulation of the recently proposed Wasserstein autoencoder framework on a non-Euclidean manifold, the Poincaré ball model of the hyperbolic space H n .

By assuming the latent space to be hyperbolic, we can use its intrinsic hierarchy to impose structure on the learned latent space representations.

We show that for datasets with latent hierarchies, we can recover the structure in a low-dimensional latent space.

We also demonstrate the model in the visual domain to analyze some of its properties and show competitive results on a graph link prediction task.

Variational Autoencoders (VAE) (17; 28) are an established class of unsupervised machine learning models, which make use of amortized approximate inference to parametrize the otherwise intractable posterior distribution.

They provide an elegant, theoretically sound generative model used in various data domains.

Typically, the latent variables are assumed to follow a Gaussian standard prior, a formulation which allows for a closed form evidence lower bound formula and is easy to sample from.

However, this constraint on the generative process can be limiting.

Real world datasets often possess a notion of structure such as object hierarchies within images or implicit graphs.

This notion is often reflected in the interdependence of latent generative factors or multimodality of the latent code distribution.

The standard VAE posterior parametrizes a unimodal distribution which does not allow structural assumptions.

Attempts at resolving this limitation have been made by either "upgrading" the posterior to be more expressive (27) or imposing structure by using various structured priors (34) , (36) .

Furthermore, the explicit treatment of the latent space as a Riemannian manifold has been considered.

For instance, the authors of (5) show that the standard VAE framework fails to model data with a latent spherical structure and propose to use a hyperspherical latent space to alleviate this problem.

Similarly, we believe that for datasets with a latent tree-like structure, using a hyperbolic latent space, which imbues the latent codes with a notion of hierarchy, is beneficial.

There has recently been a number of works which explicitly make use of properties of non-Euclidean geometry in order to perform machine learning tasks.

The use of hyperbolic spaces in particular has been shown to yield improved results on datasets which either present a hierarchical tree-like structure such as word ontologies (24) or feature some form of partial ordering (4) .

However, most of these approaches have solely considered deterministic hyperbolic embeddings.

In this work, we propose the Poincaré Wasserstein Autoencoder (PWA), a Wasserstein autoencoder (33) model which parametrizes a Gaussian distribution in the Poincaré ball model of the hyperbolic space H n .

By treating the latent space as a Riemannian manifold with constant negative curvature, we can use the norm ranking property of hyperbolic spaces to impose a notion of hierarchy on the latent space representation, which is better suited for applications where the dataset is hypothesized to possess a latent hierarchy.

We demonstrate this aspect on a synthetic dataset and evaluate it using a distortion measure for Euclidean and hyperbolic spaces.

We derive a closed form definition of a Gaussian distribution in hyperbolic space H n and sampling procedures for the prior and posterior distributions, which are matched using the Maximum Mean Discrepancy (MMD) objective.

We also compare the PWA to the Euclidean VAE visually on an MNIST digit generation task as well quantitatively on a semi-supervised link prediction task.

The rest of this paper is structured as follows: we review related work in Section 2, give an overview of the mathematical tools required to work with Riemannian manifolds as well as define the notion of probability distributions on Riemannian manifolds in Section 3.

Section 4 describes the model architecture as well as the intuition behind the Wasserstein autoencoder approach.

Furthermore, we derive a method to obtain samples from prior and posterior distributions in order to estimate the PWA objective.

We present the performed experiments in and discuss the observed results in Section 5 and a summary of our results in Section 6.

Amortized variational inference There has been a number of extensions to the original VAE framework (17) .

These extensions address various problematic aspects of the original model.

The first type aims at improving the approximation of the posterior by selecting a richer family of distributions.

Some prominent examples include the Normalizing Flow model (27) as well as its derivates (20) , (16) , (7) .

A second direction aims at imposing structure on the latent space by selecting structured priors such as the mixture prior (6), (34) , learned autoregressive priors (36) or imposing informational constraints on the objective (12), (38) .

The use of discrete latent variables has been explored in a number of works (13) (36) .

The approach conceptually most similar to ours but with a hyperspherical latent space and a von-Mises variational distribution has been presented in (5) .

Hyperbolic geometry The idea of graph generation in hyperbolic space and analysis of complex network properties has been studied in (19) .

The authors of (24) have recently used both the Poincaré model and the Lorentz model (25) of the hyperbolic space to develop word ontology embeddings which carry hierarchical information encoded by the embedding norm.

The general idea of treating the latent space as a Riemannian manifold has been explored in (2) .

A model for Bayesian inference for Riemannian manifolds relying on particle approximations has been proposed in (21) .

Finally the natural gradient method is a prime example for using the underlying information geometry imposed by the Fisher information metric to enhance learning performance (1).

Three concurrent works have explored an idea similar to ours. (22) propose to train a VAE with a hyperbolic latent space using the traditional evidence lower bound (ELBO) formulation.

They approximate the ELBO using MCMC samples as opposed to our approach, which uses a Wasserstein formulation of the problem. (23) propose to use a wrapped Gaussian distribution to obtain samples on the Lorentz model of hyperbolic latent space.

The samples are generated in Euclidean space using classical methods and then projected onto the manifold under a concatenation of a parallel transport and the exponential map at the mean.

The authors of (10) also propose a similar approach but use an adversarial autoencoder model in their work instead.

In this section, we briefly outline some of the concepts from differential geometry, which are necessary to formally define our model.

A Riemannian manifold is defined as a the tuple (M, g), where for every point x belonging to the manifold M, a tangent space T x M is defined, which corresponds to a first order local linear approximation of M at point x. The Riemannian metric g is a collection of inner products ·|· x : T x M × T x M → R on the tangent spaces T x M. We denote by α(t) ∈ M to be smooth curves on the manifold.

By computing the speed vectorα(t) at every point of the curve, the Riemannian metric allows the computation of the curve length:

Given a smooth curve α(a, b) → M, the distance is defined by the infimum over α(

The smooth curves of shortest distance between two points on a manifold are called geodesics.

Given a point x ∈ M, the exponential map exp x (v) : T x M → M gives a way map a vector v in the tangent space T x M at point x to the corresponding point on the manifold M. For the Poincaré ball model of the hyperbolic space, which is geodesically complete, this map is well defined on the whole tangent space T x M. The logarithmic map log x (v) is the inverse mapping from the manifold to the tangent space.

The parallel transport P x0→x : T x0 M → T x M defines a linear isometry between two tangent spaces of the manifold and allows to move tangent vectors along geodesics.

Hyperbolic spaces are one of three existing types of isotropic spaces: the Euclidean spaces with zero curvature, the spherical spaces with constant positive curvature and the hyperbolic spaces which feature constant negative curvature.

The Poincaré ball is one of the five isometric models of the hyperbolic space.

The model is defined by the tuple (B n , g H ) where B n is the open ball of radius 1, 1 g H is the hyperbolic metric and g E = I n is the Riemannian metric on the flat Euclidean manifold.

The geodesic distance on the Poincaré ball is given by

In order to perform arithmetic operations on the Poincaré ball model, we rely on the concept of gyrovector spaces, which is a generalization of Euclidean vector spaces to models of hyperbolic space based on Möbius transformations.

First proposed by (35) , they have been recently used to describe typical neural network operations in the Poincaré ball model of hyperbolic space (9) .

In order to perform the reparametrization in hyperbolic space, we use the gyrovector addition and Hadamard product defined as a diagonal matrix-gyrovector multiplication.

Furthermore, we make use of the exponential exp µ and logarithm log µ map operators in order to map points onto the manifold and perform the inverse mapping back to the tangent space.

The Gaussian decoder network is symmetric to the encoder network.

The Gaussian distribution is a common choice of prior for VAE style models.

Similarly to the VAE, we can select a generalization of the Gaussian distribution in the hyperbolic space as prior for our model.

In particular, we choose the maximum entropy generalization of the Gaussian distribution (26) on the Poincaré ball model.

The Gaussian probability density function in hyperbolic space is defined via the Fréchet mean µ and dispersion parameter σ > 0, analogously to the density in the Euclidean space.

The main difference compared to Euclidean space is the use of the geodesic distance d(x, µ) in the exponent and a different dispersion dependent normalization constant Z(σ) which accounts for the underlying geometry.

In order to compute the normalization constant, we use hyperbolic 1 this can be generalized to radius

for curvature c. Throughout this paper, we assume the Poincaré ball radius to be c = 1 and omit it from the notation.

polar coordinates where r = d(x, µ) is the geodesic distance between the x and µ. This allows the decomposition of Z(σ) into radial and angular components.

We derive the closed form of the normalization constant in appendix A. For a two-dimensional space, the normalization constant is given by (30) :

Dispersion representation The closed form of the hyperbolic Gaussian distribution (2) is only defined for a scalar dispersion value.

This can be a limitation on the expressivity of the learned representations.

However, the variational family which is implicitly given by the hyperbolic reparametrization allows for vectorial or even full covariance matrix representations, which can be more expressive.

Since the maximum mean discrepancy can be estimated via samples, we do not require a closed form definition of the posterior density as is the case with training using the evidence lower bound.

This allows the model to learn richer latent space representations.

Our model mimics the general architecture of a variational autoencoder.

The encoder parametrizes the posterior variational distribution q φ (z|x) and the decoder parametrizes the unit variance Gaussian likelihood p θ (x|z).

In order to accomodate the change in the underlying geometry of the latent space, we introduce the maps into hyperbolic space and back to the tangent space.

Both the encoder and decoder network consist of three fully-connected layers with ReLU activations.

We use the recently proposed hyperbolic feedforward layer (9) for the encoding of the variational family parameters (µ H , σ).

For the decoder f θ (x|z), we use the logarithm map at the origin log 0 (z) to map the posterior sample z back into the tangent space.

Mean and variance parametrization In order to obtain posterior samples in hyperbolic space, the parametrization of the mean uses a hyperbolic feedforward layer (W, b H ) as the last layer of the encoder network (proposed in (9)).

The weight matrix parameters are Euclidean and are subject to standard Euclidean optimization procedures (we use Adam (15)) while the bias parameters are hyperbolic, requiring the use of Riemannian stochastic gradient descent (RSGD) (3).

The outputs of the underlying Euclidean network h are projected using the exponential map at the origin and transformed using the hyperbolic feedforward layer map where ϕ h is the hyperbolic nonlinearity 2 :

The reparametrization trick is a common method to make the sampling operation differentiable by using a differentiable function g( , θ) to obtain a reparametrization gradient for backpropagation through the stochastic layer of the network.

For the location-scale family of distributions, the reparametrization function g( , θ) can be written as z = µ + σ in the Euclidean space where ∼ N (0, I).

We adapt the reparametrization trick for the Gaussian distribution in the hyperbolic space by using the framework of gyrovector operators.

We obtain the posterior samples for the parametrized mean µ H (x) and dispersion σ(x) using the following relation:

We can motivate the reparametrization (3) with the help of Fig. 1 , which depicts the reparametrization in a graphical fashion.

In a first step, we sample from the hyperbolic standard prior ∼ N H (0, 1) using a rejection sampling procedure we describe in Algorithm 1.

The samples are projected to the tangent space using the logarithm map log 0 at the origin, where they are scaled using the dispersion parameter.

The scaled samples are then projected back to the manifold using the exponential map exp 0 and translated using µ. We choose the hyperbolic standard prior N H (0, I) as prior p(z).

In order to generate samples from the standard prior, we use an approach based on the volume ratio of spheres in H d to obtain the quasi-uniform samples on the Poincaré disk (19) and subsequently use a rejection sampling procedure to obtain radius samples.

We use the quasi-uniform distribution

as a proposal distribution for the radius.

Using the decomposition into radial and angular components, we can sample a direction from the unit sphere uniformly and simply scale using the sampled radius to obtain the samples from the prior.

An alternative choice of prior is the wrapped Gaussian distribution.

Evidence Lower Bound The variational autoencoder relies on the evidence lower bound (ELBO) reformulation in order to perform tractable optimization of the Kullback-Leibler divergence (KLD) between the true and approximate posteriors.

In the Euclidean VAE formulation, the KLD integral has a closed-form expression, which simplifies the optimization procedure considerably.

The definition of the evidence lower bound can be extended to non-Euclidean spaces by using the following formulation with the volume element of the manifold dvol g H induced by the Riemannian metric g H .

By substituting the hyperbolic Gaussian (2) into (4) we obtain the following expressions for E q φ (z ) log q φ (z|x):

Due to the nonlinearity of the geodesic distance in the exponent, we cannot derive a closed form solution of the expectation expression E q φ (z) [log q φ (z)].

One possibility is to use a Taylor expansion of the first two moments of the expectation of the squared logarithm.

This is however problematic from a numerical standpoint due to the small convergence radius of the Taylor expansion.

The ELBO can be approximated using Monte-Carlo samples, as is done in (22) .

We have considered this approach to be suboptimal due to large variance associated with one-sample MC approximations of the integral.

Wasserstein metric In order to circumvent the high variance associated with the MC approximation we propose to use a Wasserstein Autoencoder (WAE) formulation of the variational inference problem.

The authors of the WAE framework propose to solve the optimal transport problem for matching distributions in the latent space instead of the more difficult problem of matching the data distribution p(x) to the distribution generated by the model p y (z) as is done in the generative adversarial network (GAN) literature.

Kantorovich's formulation of the optimal transport problem is given by:

Γ∈p(x∼px,y∼py)

where c(x, y) is the cost function, p(x ∼ p x , y ∼ p y ) is the set of joint distributions of the variables x ∼ p x and y ∼ p y .

Solving this problem requires a search over all possible couplings Γ of the two distributions which is very difficult from an optimization perspective.

The issue is circumvented in a WAE model as follows.

The generative model of a variational autoencoder is defined by two steps.

First we sample a latent variable z from the latent space distribution p(z).

In a second step, we map it to the output space using a deterministic parametric decoder f θ (x|z).

The resulting density is given by:

Under this model, the optimal transport cost (5) takes the following simpler form due to the fact that the transportation plan factors through the map f θ .

The optimization procedure is over the encoders q φ (x) instead of the couplings between p x and p y .

The WAE objective is derived from the optimal transport cost (5) by relaxing the constraint on the posterior q. The constraint is relaxed by using a Lagrangian multiplier and an appropriate divergence measure.

The Maximum Mean Discrepancy (MMD) metric with an appropriate positive definite RKHS 3 kernel is an example of such a divergence measure.

MMD is known to perform well when matching high-dimensional standard normal distributions (11) .

MMD is a metric on the space of probability distributions under the condition that the selected RKHS kernel is characteristic.

Geodesic kernels are generally not positive definite, however it has been shown that the Laplacian kernel k(x, y) = exp(−λ(d H (x, y))) is positive definite if the metric of the underlying space is conditionally negative definite (8) .

In particular, this holds for hyperbolic spaces (14) .

In practice, there is a high probability that a geodesic RBF kernel is also positive definite depending on the dataset topology (8) .

We choose the Laplacian kernel as it also features heavier tails than the Gaussian RBF kernel, which has a positive effect on outlier gradients (33) .

The MMD loss function is defined over two probability measures p and q in an RKHS unit ball F as follows:

There exists an unbiased estimator for D MMD (p, q φ ).

A finite sample estimate can be computed based on minibatch samples from the prior z ∼ p(z) via the rejection sampling procedure described

Parameter updates The hyperbolic geometry of the latent space requires us to perform Riemannian stochastic gradient descent (RSGD) updates for a subset of the model parameters, specifically the bias parameters of µ.

We perform full exponential map updates using gyrovector arithmetic for the gradients with respect to the hyperbolic parameters similar to (9) instead of using a retraction approximation as in (24) .

In order to avoid numerical problems at the origin and far away from the origin of the Poincaré ball, we perturb the operands if the norm is close to 0 or 1 respectively.

The Euclidean parameters are updated in parallel using the Adam optimization procedure (15).

To determine the capability of the model to retrieve an underlying hierarchy, we have setup two experiments in which we measure the average distortion of the respective latent space embeddings.

We measure the distortion between the input and latent spaces using the following distortion metric, where subscript U denotes the distances in the input space and V the distances in the latent space.

Noisy trees The first dataset is a set of synthetically generated noisy binary trees.

The vertices of the main tree are generated from a normal distribution where the mean of the child nodes corresponds to the parent sample x i = N (x p(i) , σ i ) and p(i) denotes the index of the parent node.

In addition to the main tree, we add

To encourage a good embedding in a hyperbolic space, we enforce the norms of the tree vertices to grow monotonously with the depth of the tree by rejecting samples whose norms are smaller than the norm of the parent vertices.

We have trained our model on 100 generated trees for 100 epochs.

The tree vertex variance was set to σ i = 1 and the noise variance to σ j = 0.1.

We have also normalized the generated vertices to zero mean and unit variance.

Table 1 compares the distortion values of the test set latent space embeddings obtained by using the Euclidean VAE model compared to the PWA model.

We can see that the PWA model shows less distortion when embedding trees into the latent space of dimension d = 2, which confirms our hypothesis that a hyperbolic latent space is better suited to data with latent hierarchies.

As a reference, we provide the distortion scores obtained by the classical T-SNE (37) dimensionality reduction technique.

In this experiment, we apply our model to the task of generating MNIST digits in order to get an intuition for the properties of the latent hyperbolic geometry.

In particular, we are interested in the visual distibution of the latent codes in the Poincaré disk latent space.

While the MNIST latent space is not inherently hierarchically structured -there is no obvious norm ranking that can be imposed -we can use it to compare our model to the Euclidean VAE approach.

We train the models on dynamically binarized MNIST digits and evaluate the generated samples qualitatively as well as quantitatively via the reconstruction error scores.

We can observe in Appendix B that the samples present a deteriorating quality as the dimensionality increases despite the lower reconstruction error.

This can be explained by the issue of dimension mismatch between the selected latent space dimensionality d z and the intrinsic latent space dimensionality d I documented in (29) and can be alleviated by an additional p-norm penalty on the variance.

We have not observed a significant improvement by applying the L2-penalty for higher dimensions.

We have also performed an experiment using a two-dimensional latent space.

We can observe that the structure imposed by the Poincaré disk pushes the samples towards the outside of the disk.

This observation can be explained by the fact that hyperbolic spaces grow exponentially.

In order to generate quality samples using the prior, some overlap is required with the approximate posterior in the latent space.

The issue is somewhat alleviated in higher dimensions as the distribution shifts towards the ball surface.

In this experiment, we aim at exploring the advantages of using a hyperbolic latent space on the task of predicting links in a graph.

We train our model on three different citation network datasets: Cora, Citeseer and Pubmed (32) .

We use the Variational Graph Auto-Encoder (VGAE) framework (18) and train the model in an unsupervised fashion using a subset of the links.

The performance is measured in terms of average precision (AP) and area under curve (AUC) on a test set of links that were masked during training.

Table 1 shows a comparison to the baseline with a Euclidean latent space (N -VGAE), showing improvements on the Cora and Citeseer datasets.

We also compare our results to the results obtained using a hyperspherical autoencoder (S-VGAE) (5).

It should be noted that we have used a smaller dimensionality for the hyperbolic latent space (16 vs 64 and 32 for the Euclidean and hyperspherical cases respectively), which could be attributed to the fact that a dataset with a hierarchical latent manifold requires latent space embeddings of smaller dimensionality to efficiently encode the information (analogously to the results of (24)).

We can observe that the PWA outperforms the Euclidean VAE on two of the three datasets.

The hyperspherical graph autoencoder (S-VGAE) outperforms our model.

One hypothesis which explains this is the fact that the structure of the citation networks has a tendency towards a positive curvature rather than a negative one.

It is worth noting that it is not entirely transparent whether the use of Graph Convolutional Networks (18) , which present a very simple local approximation of the convolution operator on graphs, allows to preserve the curvature of the input data.

We have presented an algorithm to perform amortized variational inference on the Poincaré ball model of the hyperbolic space.

The underlying geometry of the hyperbolic space allows for an improved performance on tasks which exhibit a partially hierarchical structure.

We have discovered certain issues related to the use of the MMD metric in hyperbolic space.

Future work will aim to circumvent these issues as well as extend the current results.

In particular, we hope to demonstrate the capabilities of our model on more tasks hypothesized to have a latent hyperbolic manifold and explore this technique for mixed curvature settings.

A PRIOR REJECTION SAMPLING H (r|0, 1) Result: n samples from prior p(z) while i < n do sampleφ ∼ N (0, I d ); compute direction on the unit sphereφ =φ ||φ|| ; sample u ∼ U(0, 1); get uniform radius samples r i ∈ [0, r max ] via ratio of hyperspheres;

where erfc is the complementary error function.

In this list of gyrovector operations and throughout this paper, we assume the Poincaré ball radius to be c = 1 and omit it from the notation.

Gyrovector addition:

x ⊕ y = (1 + 2 x, y + ||y|| 2 )x + (1 − ||x|| 2 )y 1 + 2 x, y + ||x|| 2 ||y|| 2 ]

Matrix-gyrovector product:

<|TLDR|>

@highlight

Wasserstein Autoencoder with hyperbolic latent space
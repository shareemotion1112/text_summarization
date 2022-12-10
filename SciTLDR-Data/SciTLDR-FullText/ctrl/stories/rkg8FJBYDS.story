Variational inference (VI) methods and especially variational autoencoders (VAEs) specify scalable generative models that enjoy an intuitive connection to manifold learning --- with many default priors the posterior/likelihood pair $q(z|x)$/$p(x|z)$ can be viewed as an approximate homeomorphism (and its inverse) between the data manifold and a latent Euclidean space.

However, these approximations are well-documented to become degenerate in training.

Unless the subjective prior is carefully chosen, the topologies of the prior and data distributions often will not match.

Conversely, diffusion maps (DM) automatically \textit{infer} the data topology and enjoy a rigorous connection to manifold learning, but do not scale easily or provide the inverse homeomorphism.

In this paper, we propose \textbf{a)} a principled measure for recognizing the mismatch between data and latent distributions and \textbf{b)} a method that combines the advantages of variational inference and diffusion maps to learn a homeomorphic generative model.

The measure, the \textit{locally bi-Lipschitz property}, is a sufficient condition for a homeomorphism and easy to compute and interpret.

The method, the \textit{variational diffusion autoencoder} (VDAE), is a novel generative algorithm that first infers the topology of the data distribution, then models a diffusion random walk over the data.

To achieve efficient computation in VDAEs, we use stochastic versions of both variational inference and manifold learning optimization.

We prove approximation theoretic results for the dimension dependence of VDAEs, and that locally isotropic sampling in the latent space results in a random walk over the reconstructed manifold.

Finally, we demonstrate the utility of our method on various real and synthetic datasets, and show that it exhibits performance superior to other generative models.

Recent developments in generative models such as variational auto-encoders (VAEs, Kingma & Welling (2013) ) and generative adversarial networks (GANs, Goodfellow et al. (2014) ) have made it possible to sample remarkably realistic points from complex high dimensional distributions at low computational cost.

While their methods are very different -one is derived from variational inference and the other from game theory -their ends both involve learning smooth mappings from a user-defined prior distribution to the modeled distribution.

These maps are closely tied to manifold learning when the prior is supported over a Euclidean space (e.g. Gaussian or uniform priors) and the data lie on a manifold (also known as the Manifold Hypothesis, see Narayanan & Mitter (2010) ; Fefferman et al. (2016) ).

This is because manifolds themselves are defined by sets that have homeomorphisms to such spaces.

Learning such maps is beneficial to any machine learning task, and may shed light on the success of VAEs and GANs in modeling complex distributions.

Furthermore, the connection to manifold learning may explain why these generative models fail when they do.

Known as posterior collapse in VAEs (Alemi et al., 2017; Zhao et al., 2017; He et al., 2019; Razavi et al., 2019) and mode collapse in GANs (Goodfellow, 2017) , both describe cases where the forward/reverse mapping to/from Euclidean space collapses large parts of the input to a single output.

This violates the bijective requirement of the homeomorphic mapping.

It also results in degenerate latent spaces and poor generative performance.

A major cause of such failings is when Figure 1 : A diagram depicting one step of the diffusion process modeled by the variational diffusion autoencoder (VDAE).

The diffusion and inverse diffusion maps ψ, ψ −1 , as well as the covariance C of the random walk on M Z , are all approximated by neural networks.

the geometries of the prior and target data do not agree.

We explore this issue of prior mismatch and previous treatments of it in Section 3.

Given their connection to manifold learning, it is natural to look to classical approaches in the field for ways to improve VAEs.

One of the most principled methods is spectral learning (Schölkopf et al., 1998; Roweis & Saul, 2000; Belkin & Niyogi, 2002) which involves describing data from a manifold X ⊂ M X by the eigenfunctions of a kernel on M X .

We focus specifically on DMs, where Coifman & Lafon (2006) show that normalizations of the kernel approximate a very specific diffusion process, the heat kernel over M X .

A crucial property of the heat kernel is that, like its physical analogue, it defines a diffusion process that has a uniform stationary distribution -in other words, drawing from this stationary distribution draws uniformly from the data manifold.

Moreover, Jones et al. (2008) established another crucial property of DMs, namely that distances in local neighborhoods in the eigenfunction space are nearly isometric to corresponding geodesic distances on the manifold.

However, despite its strong theoretical guarantees, DMs are poorly equipped for large scale generative modeling as they are not easily scalable and do not provide an inverse mapping from the intrinsic feature space.

In this paper we address issues in variational inference and manifold learning by combining ideas from both.

Theory in manifold learning allows us to better recognize prior mismatch, whereas variational inference provides a method to learn the difficult to approximate inverse diffusion map.

Our contributions: 1) We introduce the locally bi-Lipschitz property, a sufficient condition for a homeomorphism, for measuring the stability of a mapping between latent and data distributions.

2) We introduce VDAEs, a class of variational autoencoders whose encoder-decoder feedforward pass approximates the diffusion process on the data manifold with respect to a user-defined kernel k. 3) We show that deep neural networks are capable of learning such diffusion processes, and 4) that networks approximating this process produce random walks that have certain desirable properties, including well defined transition and stationary distributions.

5) Finally, we demonstrate the utility of the VDAE framework on a set of real and synthetic datasets, and show that they have superior performance and satisfy the locally bi-Lipschitz property where GANs and VAEs do not.

Variational inference (VI, Jordan et al. (1999); Wainwright et al. (2008) ) is a machine learning method that combines Bayesian statistics and latent variable models to approximate some probability density p(x).

VI assumes and exploits a latent variable structure in the assumed data generation process, that the observations x ∼ p(x) are conditionally distributed given unobserved latent vari-ables z. By modeling the conditional distribution, then marginalizing over z, as in

we obtain the model evidence, or likelihood that x could have instead been drawn from p θ (x).

Maximizing Eq. 1 leads to an algorithm for finding likely approximations of p(x).

As the cost of computing this integral scales exponentially with the dimension of z, we instead maximize the evidence lower bound (ELBO):

where q(z|x) is usually an approximation of p θ (z|x).

Optimizing the ELBO is sped up by taking stochastic gradients (Hoffman et al., 2013) , and further accelerated by learning a global function approximator q φ in an autoencoding structure (Kingma & Welling, 2013) .

Diffusion maps (DMs, Coifman & Lafon (2006) ) on the other hand, are a class of kernel methods that perform non-linear dimensionality reduction on a set of observations X ⊆ M X , where M X is the data manifold.

Given a symmetric and positive kernel k, DM considers the induced random walk on the graph of X, where given x, y ∈ X, the transition probabilities p(y|x) = p(x, y) are row normalized versions of k(x, y).

Moreover, the diffusion map ψ embeds the data X ∈ R m into the Euclidean space R D so that the diffusion distance is approximated by Euclidean distance.

This is a powerful property, as it allows the arbitrarily complex random walk induced by k on M X to become an isotropic Gaussian random walk on ψ(M X ).

SpectralNet is an algorithm introduced by algorithm in Shaham et al. (2018b) to speed up the diffusion map.

Until recently, the method ψ k could only be computed via the eigendecomposition of K. As a result, DMs were only be tractable for small datasets, or on larger datasets by combining landmark-based estimates and Nystrom approximation techniques.

However, Shaham et al. (2018b) propose approximations of the function ψ itself in the case that the kernel k is symmetric.

In particular, we will leverage SpectralNet to enforce our diffusion embedding prior.

Locally bi-lipschitz coordinates by kernel eigenfunctions. (Jones et al. (2008) ) analyzed the construction of local coordinates of Riemannian manifolds by Laplacian eigenfunctions and diffusion map coordinates.

They establish, for all x ∈ X, the existence of some neighborhood U (x) and d spectral coordinates given U (x) that define a bi-Lipschitz mapping from U (x) to R d .

With a smooth compact Riemannian manifold, U (x) can be chosen to be a geodesic ball with radius a constant multiple of the inradius (the radius of the largest possible ball around x without intersecting with the manifold boundary), where the constant is uniform for all x, but the indices of the d spectral coordinates as well as the local bi-Lipschitz constants may depend on x. Specifically, the Lipschitz constants involve inverse of the inradius at x multiplied again by some global constants.

For completeness we give a simplified statement of the Jones et al. (2008) result in the supplementary material.

Using the compactness of the manifold, one can always cover the manifold with m many neighborhoods (geodesic balls) on which the bi-Lipschitz property in Jones et al. (2008) holds.

As a result, there are a total of D spectral coordinates, D ≤ md (in practice D is much smaller than md, since the selected spectral coordinates in the proof of Jones et al. (2008) tend to be low-frequency ones, and thus the selection on different neighborhoods tend to overlap), such that on each of the m neighborhoods, there exists a subset of d spectral coordinates out of the D ones which are bi-Lipschitz on the neighborhood, and the Lipschitz constants can be bounded uniformly from below and above.

Our proposed measure and model is motivated by degenerate latent spaces and poor generative performance in a variational inference framework arising from prior mismatch: when the topologies of the data and prior distributions do not agree.

In real world data, this is usually due to two factors: first, when the dimensionalities of the distributions do not match, and second, when the geometries do not match.

It is easy to see that homeomorphisms between the distributions will not exist in either case: pointwise correspondences cannot be established, thus the bijective condition cannot be met.

As a result, the model has poor generative performance -for each point not captured in the pointwise correspondence, the latent or generated distribution loses expressivity.

Though the default choice of Gaussian distribution for p(z) is mathematically elegant and computationally expedient, there are many datasets, real and synthetic, for which this distribution is ill-suited.

It is well known that spherical distributions are superior for modeling directional data (Fisher et al., 1993; Mardia, 2014) , which can be found in fields as diverse as bioinformatics (Hamelryck et al., 2006) , geology (Peel et al., 2001) , material science (Krieger Lassen et al., 1994) , natural image processing (Bahlmann, 2006) , and simply preprocessed datasets 1 .

Additionally observe that no homeomorphism exists between R k and S 1 for any k. For data distributed on more complex manifolds, the literature is sparse due to the difficult nature of such study.

However, the manifold hypothesis is well-known and studied (Narayanan & Mitter, 2010; Fefferman et al., 2016) .

Previous research on alleviating prior mismatch exists.

Davidson et al. (2018) ; Xu & Durrett (2018) consider VAEs with the von-Mises Fisher prior, a geometrically hyperspherical prior. (Rey et al., 2019) further model arbitrarily complex manifolds as priors, but require explicit knowledge of the manifold (i.e. its projection map, scalar curvature, and volume).

Finally, Tomczak & Welling (2017) consider mixtures of any pre-existing priors.

But while these methods increase the expressivity of the priors available, they do not prescribe a method for choosing the prior itself.

That responsibility still lies with the user.

Convserly, our method chooses the best prior automatically.

To our knowledge, ours is the first to take a data-driven approach to prior selection.

By using some data to inform the prior, we not only guarantee the existence of a homeomorphism between data and prior distributions, we explicitly define it by the learned diffusion mapψ.

In this section we propose VDAEs, a variational inference method that, given the data manifold M X , observations X ⊂ M X , and a kernel k, models the geometry of X by approximating a random walk over the latent diffusion manifold M Z := ψ(M X ).

The model is trained by maximizing the local evidence: the evidence (i.e. log-likelihood) of each point given its random walk neighborhood.

Points are generated from the trained model by sampling from π, the stationary distribution of the resulting random walk.

Starting from some point x ∈ X, we can roughly describe one step of the walk as the composition of three functions: 1) the approximate diffusion mapψ Θ : M X → M Z , 2) a sampling procedure from the learned diffusion process z ∼ q φ (z |x) = N (ψ Θ (x),C φ ) on M Z , and 3) the learned inverse diffusion mapψ

where the constant c is user-defined and fixed.

We rely crucially on three advantages of our latent spaceψ Θ (X): a) that it is well-defined (given the first D eigenvalues of k are distinct), b) well-approximated (given SpectralNet) and c) that Euclidean distances in M Z approximate single-step random walk distances on M X (see Section 2 and Coifman & Lafon (2006)).

Thus the transition probabilities induced by k can be approximated by Gaussian kernels 2 in M Z .

Therefore, to model a diffusion random walk over M Z , we must learn the functionsψ Θ ,ψ −1 θ ,C φ that approximate the diffusion map, its inverse, and the covariance of the random walk on M Z , at all points z ∈ M Z .

SpectralNet gives usψ Θ .

To learnψ −1 θ andC φ , we use variational inference.

Formally, let us define U x := B d (x, δ) ∩ M X , where B d (x, δ) as the δ-ball around x with respect to d(·, ·), the diffusion distance on M Z .

For each x ∈ X, we define the local evidence of x as

where p(x |x)| Ux is the restriction of p(x |x) to U x .

The resulting local evidence lower bound is:

Note that the neighborhood reconstruction error should be differentiated from the self reconstruction error that is in VAEs.

Eq. 4 produces the empirical loss function:

where

is the deterministic, differentiable function, depending onψ Θ andC φ , that generates q φ by the reparameterization trick 3 (Kingma & Welling, 2013) .

Algorithm 1 VDAE training Θ, φ, θ ← Initialize parameters Obtain parameters Θ for the approximate diffusion mapψ Θ by Shaham et al. (2018b) while not converged do

Take one step of the diffusion random walk

Compute gradients of the loss, i.e. Eq. equation 4

Update φ, θ using g

Here we discuss the algorithm for generating data points from p(x).

Composing q φ (z |x)(≈ p θ (z |x)) with p θ (x |z ) gives us an approximation of p θ (x |x).

Then the simple, parallelizable, and fast random walk based sampling procedure naturally arises: initialize with an arbitrary point on the manifold x 0 ∈ M X , then pick suitably large N and for n = 1, . . .

, N draw x n ∼ p(x|x n−1 ).

Eventually, our diffusion random walk converges on its stationary distribution π.

By Coifman & Lafon (2006), this is guaranteed to be the uniform distribution on the data manifold.

See Section 6.2 for examples of points drawn from this procedure.

We now introduce a practical implementation VDAEs, considering the case whereψ Θ (x), q φ (z |x) and p θ (x |z ) are neural network functions, as they are in VAEs and SpectralNet, respectively.

The neighborhood reconstruction error.

Since q φ (z |x) models the neighborhood ofψ Θ (x), we may sample q φ to obtain z (the neighbor of x in the latent space).

This gives p θ (x |x) ≈ ψ −1 θ (q φ (z |x)), where ψ −1 exists due to the bi-Lipschitz property.

We can efficiently approximate x ∈ M X by considering the closest embedded data pointψ Θ (x) ∈ M Z to z =ψ Θ (x ).

This is because Euclidean distance on M Z approximates the diffusion distance on M X .

In other words, x ∼ p θ (x |x) ≈ψ −1 θ (q φ (z |x)) which we approximate empirically by

where A ⊆ X is the training batch.

On the other hand, the divergence of random walk distributions −D KL (q φ (z |x)||p θ (z |x)) can be modeled simply as the divergence of two Gaussian kernels defined on M Z .

Though p θ (z |x) is intractable, the diffusion map ψ gives us the diffusion embedding Z, which is an approximation of the true distribution of p θ (z |x) in a neighborhood around z = ψ(x).

We estimate the first and second moments of this distribution in R D by computing the local Mahalanobis distance of points in the neighborhood.

Then, by minimizing the KL divergence between q φ (z |x) and the one implied by this Mahalanobis distance, we obtain the loss:

where

is the covariance of the points in a neighborhood of z = ψ(x) ∈ Z, and α is a scaling parameter.

Note that C φ (x) does not have to be diagonal, and in fact is most likely not.

Combining Eqs. 6 and 7 we obtain Algorithm 1.

Now we consider the sampling procedure.

Since we use neural networks to approximate q φ (z |x) and p θ (x |z ), the generation procedure is highly parallelizable.

We empirically observe the random walk enjoys rapid mixing properties -it does not take many iterations of the random walk to sample from all of M Z 4 .

This leads to Algorithm 2.

Algorithm 2 VDAE sampling

Take one step of the diffusion random walk

Map back into input space t ← t + 1

We theoretically prove that the desired inverse map ψ −1 from spectral coordinate codes back to the manifold can be approximated by a decoder network, where the network complexity is bounded by quantities related to the intrinsic geometry of the manifold.

This section relies heavily on the known bi-Lipschitz property of DMs Jones et al. (2008), which we are approximating with the VDAE latent space without the need for regularization.

The theory for the capacity of the encoder to map M to the diffusion map space ψ(M) has already been considered in Shaham et al. (2018a) and Mishne et al. (2017) .

We instead focus on the decoder, which requires a different treatment.

The following theorem is proved in Appendix A.3, based upon the result in Jones et al. (2008) .

to have a subset of coordinates that are locally bi-Lipschitz.

Let X = [X 1 , ..., X m ] be the set of all m extrinsic coordinates of the manifold.

Then there exists a sparsely-connected ReLU network f N , with 4DC M X nodes in the first layer, 8dmN nodes in the second layer, and 2mN nodes in the third layer, and m nodes in the output layer, such that

where the norm is interpreted as

Here C ψ depends on how sparsely X(ψ(x)) Ui can be represented in terms of the ReLU wavelet frame on each neighborhood U i , and C M X on the curvature and dimension of the manifold M X .

Theorem 1 is complementary to the theorem in Shaham et al. (2018a) , which provides guarantees for the encoder, as Theorem 1 demonstrates a similar approximation theoretic argument for the decoder.

The proof is built on two properties of ReLU neural networks: 1) their ability to split curved domains into small, almost Euclidean patches, 2) their ability to build differences of bump functions VDAE SVAE VAE GAN Figure 2 : Reconstructed images from the rotating bulldog example plotted in the latent space of VDAE (left), Spherical VAE (SVAE, left-middle) and VAE (right-middle), and GAN (right) on each patch, which allows one to borrow approximation results from the theory of wavelets on spaces of homogeneous type.

The proof also crucially uses the bi-Lipschitz property of the diffusion embedding Jones et al. (2008) .

The key insight of Theorem 1 is that, because of the bi-Lipschitz property, the coordinates of the manifold in the ambient space R m can be thought of as functions of the diffusion coordinates.

We show that because each of coordinates function X i is a Lipschitz function, the ReLU wavelet coefficients of X i are necessarily

1 .

This allows us to use the existing guarantees of Shaham et al. (2018a) to complete the desired bound.

We also discuss the connections between the distribution at each point in diffusion map space, q φ (z|x), and the result of this distribution after being decoded through the decoder network f N (z) for z ∼ q φ (z|X).

Similar to Singer & Coifman (2008) , we characterize the covariance matrix

The following theorem is proved in Appendix A.3.

Theorem 2.

Let f N be a neural network approximation to X as in Theorem 1, such that it approximates the extrinsic manifold coordinates.

Let C ∈ R m×m be the covariance matrix

, Σ) with small enough Σ that there exists a patch U z0 ⊂ M around z 0 satisfying the bi-Lipschitz property of Jones et al. (2008), and such that P r(z ∼ q φ (z|x) ∈ ψ(U z0 )) < .

Then the number of eigenvalues of C greater than is at most d, and C = J z0 ΣJ 6 EXPERIMENTAL RESULTS

We consider the problem of generating new frames from a video of rigid movement.

We take 200 frames of a color video (each frame is 100 × 80 × 3) of a spinning bulldog Lederman & Talmon (2018) .

Due to the spinning of figure and the fixed background, this creates a low-dimensional approximately circular manifold.

We compare our method to VAE, the Wasserstein GAN Gulrajani et al. (2017) (with a bi-lipchitz constraint on the critic), and the hyperspherical VAE Davidson et al. (2018) .

For the VAE, we use a two dimensional Gaussian prior p θ (z), such that z ∼ N (0, I 2 ).

The noise injected to the GAN is drawn from a two dimensional uniform distribution p θ (z), such that z i ∼ U (0, 1), i = 1, 2.

For the spherical VAE, we use a latent dimension of D = 2, which highlights the dimension mismatch issue that occurs with a spherical prior.

This is a benefit of VDAE, even if we choose D > d the latent embedding will still only be locally d dimensional.

We use the same architecture for all networks which consists of one hidden layer with 512 neurons, activation function for all networks are tanh.

In Fig. 2 , we present 300 generated samples, by displaying them on a scatter plot with coordinates corresponding to their latent dimensions z 1 and z 2 .

In this series of experiments, we visualize the results of the sampling procedure in Algorithm 2 on three synthetic manifolds.

As discussed in 4.2, we randomly select an initial seed point, then recursively sample from p θ (x |x) many times to simulate a random walk on the manifold.

In the top row of Fig. 3 , we highlight the location of the initial seed point, take 20 steps of the random walk, and display the resulting generated points on three learned manifolds.

Clearly after a large number of resampling iterations, the algorithm continues to generate points on the manifold, and the distribution of sampled points converges to a uniform stationary distribution on the manifold.

Moreover, this stationary distribution is reached very quickly.

In the bottom row of the same Fig. 3 , we show p θ (x |x) by sampling a large number of points sampled from the single seed point.

As can be seen, a single step of p θ (x |x) covers a large part of the latent space.

The architecture also uses one hidden layer of 512 neurons and tanh activations.

In this section, we deal with the problem of generating samples from data with multiple clusters in an unsupervised fashion (i.e. no a priori knowledge of the cluster structure).

Clustered data creates a problem for many generative models, as the topology of the latent space (i.e. normal distribution) differs from the topology of the data space with multiple clusters.

In our first experiment, we show that our method is capable of generating new points from a particular cluster given an input point from that cluster.

This generation is done in an unsupervised fashion, which is a different setting from the approach of conditional VAEs Sohn et al. (2015) that require training labels.

We demonstrate this property on MNIST in Figure 4 , and show that the newly generated points after short diffusion time remain in the equivalent class to the seeded image.

Here the architecture is a standard fully convolutional architecture.

Details can be found in Appendix A.4.

Figure 4 : An example of cluster conditional sampling with our method, given a seed point (top left of each image grid).

The DVAE is able to produce examples via the random walk that stay approximately within the cluster of the seed point, without any supervised knowledge of the cluster.

The problem of addressing difference in topologies between the latent space of a generative model and the output data has been acknowledged in recent works about rejection sampling (Azadi et al., 2018; Turner et al., 2018) .

Rejection sampling of neural networks consists of generating a large collection of samples using a standard GAN, and then designing a probabilistic algorithm to decide in a post-hoc fashion whether the points were truly in the support of the data distribution p(x).

In the following experiment, we compare to the standard example in the generative model literature.

The data consists of nine bounded spherical densities with significant minimal separation, lying on a 5 × 5 grid.

A standard GAN or VAE struggles to avoid generating points in the gaps between Figure 5 : Comparison between GAN, DRS-GAN, and our samples on a 5 × 5 Gaussian grid.

GAN and DRS-GAN samples taken from Azadi et al. (2018) .

Shown from left-right are Original, GAN, DRS-GAN, and our method.

these densities, and thus requires the post-sampling rejection analysis.

On the other hand, our model creates a latent space that separates each of these clusters into their own features and only generates points that exist in the neighborhood of training data.

Figure 5 clearly shows that this results in significantly fewer points generated in the gaps between clusters, as well as eliminating the need to generate additional points that are not in final generated set.

Our VDAE architecture here uses one hidden layer of 512 neurons and tanh activations.

GAN and DRS-GAN architectures are as described in Azadi et al. (2018) .

Here we describe a practical method for computing the local bi-Lipschitz property, then use it to evaluate several methods on the MNIST dataset.

Let Z and X be metric spaces and f : Z → X. We define, for each z ∈ Z and k ∈ N, the function bilip k (z):

where Z := f −1 (X) is the latent embedding of our dataset X 5 , d X and d Z are metrics on X and Z, and U z,k is the k-nearest neighborhood of z. Intuitively, increasing values of K can be thought of as an increasing tendency of the learned map to stretch or compress regions of space.

By analyzing various statistics of the local bi-Lipschitz measure evaluated at all points of a latent space Z, we can gain insight into how well-behaved a homeomorphism f is.

In Table 1 we report the mean and standard deviation, over 10 runs, of the local bi-Lipschitz property for several methods trained on the MNIST dataset.

The comparison is between the Wassertein GAN (WGAN), the VAE, the hyperspherical VAE (SVAE), and our method.

We use standard architectures prescribed by their respective papers to train the methods.

For our method we use a single 500 unit hidden layer network architecture with ReLU nonlinearities for both the encoder and decoder.

By constraining our latent space to be the diffusion embedding of the data, our method finds a mapping that automatically enjoys the homeomorphic properties of an ideal mapping, and this is reflected in the low values of the local bi-Lipschitz constant.

Conversely, other methods do not consider the topology of the data in the prior distribution.

This is especially appparent in the VAE and SVAE, which must generate from the entirety of the input distribution X since they minimize a reconstruction loss.

Interestingly, the mode collapse tendency of GANs alleviate the pathology of the bi-Lipschitz constant by allowing the GAN to focus on a subset of the distribution -but this comes at the cost, of course, of collapsing to a few modes of the dataset.

Our method is able to reconstruct the entirety of X while simultaneously maintaining a low local bi-Lipschitz constant.

We begin with taking the log of the random walk transition likelihood,

where q(z ) is an arbitrary distribution.

We let q(z ) to be the conditional distribution q(z |x).

Furthermore, if we make the simplifying assumption that p θ (x |z , z) = p θ (x |z ), then we obtain Eq. 4

To state the result in Jones et al. (2008), we need the following set-up:

(C1) M is a d-dimensional smooth compact manifold, possibly having boundary, equipped with a smooth (at least C 2 ) Riemannian metric g;

We denote the geodesic distance by d M , and the geodesic ball centering at x with radius r by B M (x, r).

Under (C1), for each point x ∈ M, there exists r M (x) which is the inradius, that is, r is the largest number s.t.

B M (x, r) is contained M.

Let M be the Laplacian-Beltrami operator on M with Neumann boundary condition, which is self-adjoint on L 2 (M, µ), µ being the Riemannian volume given by g. Suppose that M is re-scaled to have volume 1.

The next condition we need concerns the spectrum of the manifold Laplacian (C2) M has discrete spectrum, and the eigenvalues λ 0 ≤ λ 1 ≤ · · · satisfy the Weyl's estimate, i.e. exists constant C which only depends on M s.t.

Let ψ j be the eigenfunction associated with λ j , {ψ j } j form an orthonormal bases of L 2 (M, µ).

The last condition is (C3) The heat kernel (defined by the heat equation on M) has the spectral representation as

That is, Ψ is bi-Lipschitz on the neighborhood B(x, c 1 r M (x)) with the Lipschitz constants indicated as above.

The subscript x in Ψ x emphasizes that the indices j 1 , · · · , j d may depend on x.

Proof of Theorem 1.

The proof of Theorem 1 is actually a simple extension of the following theorem, Theorem 4, which needs to be proved for each individual extrinsic coordinate X k , hence the additional factor of m coming from the L2 norm of m functions.

Theorem 4.

Let M ⊂ R m be a smooth d-dimensional manifold, ψ(M) ⊂ R D be the diffusion map for D ≥ d large enough to have a subset of coordinates that are locally bi-Lipschitz.

Let one of the m extrinsic coordinates of the manifold be denoted X(ψ(x)) for x ∈ M. Then there exists a sparsely-connected ReLU network f N , with 4DC M nodes in the first layer, 8dN nodes in the second layer, and 2N nodes in the third layer, such that

where C ψ depends on how sparsely X(ψ(x)) Ui can be represented in terms of the ReLU wavelet frame on each neighborhood U i , and C M on the curvature and dimension of the manifold M.

Proof of Theorem 4.

The proof borrows from the main theorem of Shaham et al. (2018a) .

We adopt this notation and summarize the changes in the proof here.

For a full description of the theory and guarantees for neural networks on manifolds, see Shaham et al. (2018a) .

Let C M be the number of neighborhoods U i = B(x i , δ) ∩ M needed to cover M such that ∀x, y ∈ U i , (1 First, we note that as in Shaham et al. (2018a) , the first layer of a neural network is capable of using 4D units to select the subset of d coordinates ψ(x) from ψ(x) for x ∈ U i and zeroing out the other D−d coordinates with ReLU bump functions.

Then we can define X( ψ(x)) = X(ψ(x)) on x ∈ U i .

Now to apply the theorem from Shaham et al. (2018a) , we must establish that X Ui : ψ(U i ) → R can be written efficiently in terms of ReLU functions.

Because of the manifold and diffusion metrics being bi-Lipschitz, we know at a minimum that ψ is invertible on ψ(U i ).

Because of this invertibility, we will slightly abuse notation and refer to X(ψ(x)) = X(x), where this is understood to be the extrinsic coordinate of the manifold at the point x that cooresponds to ψ(x).

we also know that ∀x, y ∈ U i , |X( ψ(x)) − X( ψ(y))| = |X(x) − X(y)| ≤ max

where ∇X(z) is understood to be the gradient of X(z) at the point z ∈ M.

This means X( ψ(x)) is a Lipschitz function w.r.t.

ψ(x).

Because X( ψ(x)) Lipschitz continuous, it can be approximated by step functions on a ball of radius 2 − to an error that is at most with the fact that ψ(U i ) is compact, gives the fact that on ψ(U i ), set of ReLU wavelet coefficients is in 1 .

And from Shaham et al. (2018a) , if on a local patch the function is expressible in terms of ReLU wavelet coefficients in 1 , then there is an approximation rate of 1 √ N for N ReLU wavelet terms.

<|TLDR|>

@highlight

We combine variational inference and manifold learning (specifically VAEs and diffusion maps) to build a generative model based on a diffusion random walk on a data manifold; we generate samples by drawing from the walk's stationary distribution.
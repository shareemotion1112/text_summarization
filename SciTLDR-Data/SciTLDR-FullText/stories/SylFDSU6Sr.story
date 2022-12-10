A disentangled representation of a data set should be capable of recovering the underlying factors that generated it.

One question that arises is whether using Euclidean space for latent variable models can produce a disentangled representation when the underlying generating factors have a certain geometrical structure.

Take for example the images of a car seen from different angles.

The angle has a periodic structure but a 1-dimensional representation would fail to capture this topology.

How can we address this problem?

The submissions presented for the first stage of the  NeurIPS2019 Disentanglement Challenge consist of a Diffusion Variational Autoencoder ($\Delta$VAE) with a hyperspherical latent space which can for example recover periodic true factors.

The training of the $\Delta$VAE is enhanced by incorporating a modified version of the Evidence Lower Bound (ELBO) for tailoring the encoding capacity of the posterior approximate.

Variational Autoencoders (VAEs) proposed by BID4 are an unsupervised learning method that can estimate the underlying generative model that produced a data set in terms of the so-called latent variables.

In the context of VAEs, a disentangled representation is obtained when the latent variables represent the true independent underlying factors, which usually have a semantic meaning, that generated the data set.

VAEs assume that a data set X = {x i } N i=1 consists of N independent and identically distributed data points belonging to a set X. A set Z of unobserved latent variables is proposed and the main goal is to maximize the log-likelihood via variational inference using an approximate to the posterior distribution Q X|z with parameters a, b calculated by neural networks.

A prior distribution P Z is selected before training such that the training of the VAE is carried out by maximizing for each data point the Evidence Lower Bound (ELBO) w.r.t.

the neural network weights that calculate a, b given by DISPLAYFORM0 To accomplish the disentanglement of latent variables BID3 proposed to weight the contribution of both terms in the ELBO by using a parameter β ∈ R + to change the capacity of encoding of the posterior distribution.

The idea of changing the capacity of the encoding distribution was further explored in BID0 where the KullbackLeibler divergence term is pushed towards a certain value C ∈ R + in each training step.

The combination of both approaches led to a to the following training objective to be maximized, DISPLAYFORM1 The value of β is fixed before training and C is increased linearly each epoch of training from a minimum value C min to C max .

We refer to this procedure as capacity annealing.

In some cases the underlying factors that generated a data set have a certain geometrical/topological structure that cannot be captured with the traditional Euclidean latent variables as has been mentioned in and in .

This problem is referred to as manifold mismatch.

For the NeurIPS2019 Disentanglement challenge, datasets for local evaluation are provided based on the paper by BID5 .

It is important to note that in such datasets there is at least one underlying factor that has a periodic structure.

Take for example the Cars3D dataset consisting of images of cars.

In particular, one factor of variation is the azimuthal angle of rotation of the car.

The geometrical structure of this factor is circular and thus it is better represented with a periodical latent variable.

The Diffusion Variational Autoencoders ∆VAE presented by Pérez Rey et al. FORMULA0 provide a versatile method that can be used to implement arbitrary closed manifolds for a latent space, in particular, hyperspheres.

We propose the use of a ∆VAE with hyperspherical latent space coupled with the capacity annealing procedure from Equation 2.

In has described that for high dimensional latent spaces, the vanilla VAE from Kingma and Welling (2014) behaves similarly to the VAE with a high dimensional hyperspherical latent space.

Thus, we have chosen to use a high dimensional hyperspherical latent space of dimension d, i.e. Z = S d since it can provide better representations for periodical latent variables while still maintaining the properties of the vanilla implementation.

The Diffusion VAE from Pérez Rey et al. FORMULA0 with hyperspherical latent space consists of the following elements:• Hyperspherical latent space embedded in Euclidean latent space Z = S d ⊆

R d+1• Uniform prior P Z over the hypersphere.• Posterior distribution Q µ Z ,t Z from a family of solutions to the heat equation over the hypersphere parameterized by location µ Z ∈ S d+1 and scale t ∈ R + .•

Decoder distribution P µ X ,σ X X from a family of normal distributions parametrized by location µ X ∈ X (covariance is chosen to be the identity).• Neural networks to calculate parameters µ Z : DISPLAYFORM0 The encoding neural network µ Z is a composition of a multi layer perceptron into R d+1 with a projection function P into the hypersphere.• Projection map corresponds to P : DISPLAYFORM1 During training, there are two key procedures that need to be taken into account: the reparameterization trick for sampling the posterior approximate in order to calculate the first term of Equation 2 and the calculation of the Kullback-Leibler divergence between the posterior approximate and the uniform prior for the second term of Equation 2.Reparameterization trick In order to approximate the first term of the ELBO, BID4 proposed the reparameterization trick.

In the hypersphere the procedure for sampling z ∼ Q (µ Z ,t) Z|x described in Pérez Rey et al. FORMULA0 was implemented.

It consists of a random walk of L steps over the hypersphere which approximates to the transition kernel of the Brownian motion over the manifold.

DISPLAYFORM2 Given a data point x ∈ X in the data set.1.

Calculate the parameters for the posterior distribution with the corresponding neural networks t = t(x) and DISPLAYFORM3 2.

Repeat for l ∈ {0, 1, 2, . . .

, L − 1} steps• Sample an auxiliary variable ∼ N (0, I) from a d + 1 dimensional standard normal distribution.• Calculate the l + 1 step in the random walk z (l+1) = P z (l) + t

The sampled latent variables z is then used to estimate the first term of the ELBO and is passed to the decoding neural network.

The Kullback-Leibler divergence between the prior and the posterior is approximated using the formula in Pérez BID6 where Vol(S d ) corresponds to the volume of the hypersphere and is given by DISPLAYFORM0

The hyperparameter values were chosen based on basic implementations described in the corresponding papers: β from BID3 , capacity annealing BID5 and Diffusion VAE Pérez Rey et al. (2019) .

The exact values used are presented in the Appendix A.

@highlight

Description of submission to NeurIPS2019 Disentanglement Challenge based on hyperspherical variational autoencoders
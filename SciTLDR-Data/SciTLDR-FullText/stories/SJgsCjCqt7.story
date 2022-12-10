We propose a method for learning the dependency structure between latent variables in deep latent variable models.

Our general modeling and inference framework combines the complementary strengths of deep generative models and probabilistic graphical models.

In particular, we express the latent variable space of a variational autoencoder (VAE) in terms of a Bayesian network with a learned, flexible dependency structure.

The network parameters, variational parameters as well as the latent topology are optimized simultaneously with a single objective.

Inference is formulated via a sampling procedure that produces expectations over latent variable structures and incorporates top-down and bottom-up reasoning over latent variable values.

We validate our framework in extensive experiments on MNIST, Omniglot, and CIFAR-10.

Comparisons to state-of-the-art structured variational autoencoder baselines show improvements in terms of the expressiveness of the learned model.

@highlight

We propose a method for learning latent dependency structure in variational autoencoders.

@highlight

Uses a matrix of binary random variables to capture dependencies between latent variables in a hierarchical deep generative model.

@highlight

This paper presents a VAE approach in which a dependency structure on the latent variable is learned during training.

@highlight

The authors propose to augment the latent space of a VAE with an auto-regressive structure, to improve the expressiveness of both the inference network and the latent prior
Many models based on the Variational Autoencoder are proposed to achieve disentangled latent variables in inference.

However, most current work is focusing on designing powerful disentangling regularizers, while the given number of dimensions for the latent representation at initialization could severely inﬂuence the disentanglement.

Thus, a pruning mechanism is introduced, aiming at automatically seeking for the intrinsic dimension of the data while promoting disentangled representations.

The proposed method is validated on MPI3D and MNIST to be advancing state-of-the-art methods in disentanglement, reconstruction, and robustness.

The code is provided on the https://github.com/WeyShi/FYP-of-Disentanglement.

To advance disentanglement, models based on the Variational Autoencoder (VAE) (Kingma and Welling, 2014) are proposed in terms of additional disentangling regularizers.

However, in this paper, we introduce an orthogonal mechanism that is applicable to most state-of-theart models, resulting in higher disentanglement and robustness for model configurationsespecially the choice of dimensionality for the latent representation.

Intuitively, both excessive and deficient latent dimensions can be detrimental to achieving the best disentangled latent representations.

For excessive dimensions, powerful disentangling regularizers, like the β-VAE (Higgins et al., 2017) , can force information to be split across dimensions, resulting in capturing incomplete features.

On the other hand, having too few dimensions inevitably leads to an entangled representation, such that each dimension could capture enough information for the subsequent reconstruction.

In this paper, we introduce an approximated L 0 regularization (Louizos et al., 2018) to prune the dimension of the latent representation vector.

Consequently, our Pruning Variational Autoencoders (PVAE) framework is applicable to most state-of-the-art VAE-based models due to its orthogonality with current approaches.

But in this challenge, we choose to put the pruning mechanism onto the DIP-VAE (for Disentangled Inferred Prior VAE) (Kumar et al., 2018) due to its decent performance on MPI3D (Gondal et al., 2019) .

In the context of pruning, the aim of L 0 is to compress the network, while here the goal is seeking for the intrinsic dimension for the latent representation, which is achieved by the balance between several terms.

2.1.

The Masked Base Model: Masked DIP-VAE Basically, we desire to achieve binary masks m, depending on some learnable parameters α, to control each dimension.

Thus, the DIP-VAE loss term with masks can be formulated as follow:

where x, p(x), z, and p(z) are the input images, the data distribution, the latent variables (the output of the encoder), and their prior, respectively, and µ φ (·), and p θ (·), q φ (z|·) denote the function of the encoder's mean path, the decoder, and the encoder.

Meanwhile,

] denotes the covariance matrix of the pruned mean representations.

There are two points to note about the Kullback-Leibler (KL) divergence terms.

Firstly, they decompose across dimensions (z i ) because we assumed factorized prior and variational posterior distributions.

Secondly, the KL term for each dimension is multiplied by the mask for consistency when that dimension is forced to zero, which can be understood in terms of inference with spike-and-slab distributions (see Louizos et al., 2018 , Appendix A).

With a second term denoting L 0 regularization over e, the samples drawn from the q φ (z|x), the total loss can be formulated as

where

, which is the sum of the probability of m j being positive.

where γ < 0 and ζ > 1 are the lower and upper bounds for the stretched range, and β is the temperature coefficient of the masks generation process introduced in Section 2.3.

The given formulation is slightly different from Louizos et al. (2018) for clarity.

The mask vector m is clamped such that m j ∈ [0, 1].

The binary masks m are modelled as following Bernoulli distributions with parameters α: Louizos et al. (2018) proposed to obtain these masks in a differentiable fashion, feeding uniform random variables through a sigmoid-like function whose location depends on α.

Furthermore, to ensure that masks are likely to be exactly 0 or 1, they stretch the value range of the sigmoid-like function to be [ζ, γ] and then clamp it to be [0, 1].

This process can be formulated as below, and is illustrated in Appendix A:

To align it with VAE, we need the encoder to output means µ φ (x) and variances σ 2 φ (x) of q(z|x) instead of means µ φ (x) and log σ 2 φ (x) such that after pruning we have a N (0, 0) rather than N (0, 1) for a specific dimension.

In detail, a mask is multiplied with each pair of mean and variance and the KL divergence for the corresponding dimension, such that dimensions can effectively be 'switched off' and not affect training.

To avoid numerical instability in the KL divergence, we add a small positive constant to σ 2 φ (x).

Given the outputs of the last layer of the original encoder, the L0Pair layer can be expressed as

In terms of the structure of the encoder and the decoder, we adopt the default settings given in the starter kit 1 , which is based on row 3 of Table 1 on page 13 of Higgins et al. (2017) .

We list our choices of hyperparameters in Appendix B. The L 0 regularization in the pruning mechanism facilitates the performance and the robustness of vanilla DIP-VAE on MPI3D (Gondal et al., 2019) by approaching the intrinsic dimension during training.

In Appendix B, we additionally present results on MNIST (LeCun and Cortes, 2010) with a JointVAE (Dupont, 2018) extension of the proposed PVAE (PJVAE), which further validates the disentanglement benefits of pruning.

A pruning mechanism that is complementary to most current state-of-the-art VAE-based disentangling models is introduced and validated on MPI3D and MNIST.

The approximated L 0 regularization facilitates the model to capture better-disentangled representations with optimal size and increases the robustness to initialization.

Moreover, with the same hyperparameters, the model approaches the intrinsic dimension for several datasets including MNIST and MPI3D, even with an extra-large number of dimensions at initialization.

Even given the intrinsic dimension, the PVAE still outperforms other SOTA methods in terms of disentanglement and reconstruction.

The default parameters are given in Table 1 .

As for the optimizer and its Learning rate, we select Adam optimizer with 10 −4 .

Moreover, the τ = 0.1 generalizes well on both MNIST and CelebA 2 .

To capture the discrete features like Digits (Dupont, 2018) , we adopt one additional discrete variable and the model becomes Pruning Joint VAE (PJVAE).

Since there is only one discrete variable, it is unnecessary to impose further disentanglement on it (the disentanglement on discrete variables is beyond the scope of this report).

In Figure 3 , we can see the advantage of pruning on MNIST, especially when the initialization far deviates from the intrinsic dimension (which is still unknown for MNIST, but is estimated to be around 10 by several methods).

However, the PJVAE is robust to the initialization as long as it is given enough latent space at initialization.

Surprisingly, with appropriate initialization, its reconstruction occasionally becomes better than the VAE, with consistent higher disentanglement performance.

Furthermore, on this dataset PJVAE outperforms DIP-VAE in both metrics.

Inspecting the variation between different initialization, we can validate the robustness of PJVAE versus the other two methods.

In general, in terms of TC, PJVAE possesses obvious advantages.

And reconstruction performance is the same, PJVAE also showing a consistent lower error.

Note that both VAE and DIP-VAE are initialized with one additional 10-value categorical (discrete) variable for a fair comparison.

The only difference between this DIP-VAE (actually, DIP-JointVAE) and PJVAE, is the approximated L 0 .

MNIST.

The number denotes the total dimensionality of the latent variables at initialization.

TC stands for Total Correlation.

<|TLDR|>

@highlight

The Pruning VAE is proposed to search for disentangled variables with intrinsic dimension.
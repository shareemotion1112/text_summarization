Modern generative models are usually designed to match target distributions directly in the data space, where the intrinsic dimensionality of data can be much lower than the ambient dimensionality.

We argue that this discrepancy may contribute to the difficulties in training generative models.

We therefore propose to map both the generated and target distributions to the latent space using the encoder of a standard autoencoder, and train the generator (or decoder) to match the target distribution in the latent space.

The resulting method, perceptual generative autoencoder (PGA), is then incorporated with maximum likelihood or variational autoencoder (VAE) objective to train the generative model.

With maximum likelihood, PGA generalizes the idea of reversible generative models to unrestricted neural network architectures and arbitrary latent dimensionalities.

When combined with VAE, PGA can generate sharper samples than vanilla VAE.

Recent years have witnessed great interest in generative models, mainly due to the success of generative adversarial networks (GANs) BID7 BID12 BID1 .

Despite the prevalence, the adversarial nature of GANs can lead to a number of challenges, such as unstable training dynamics and mode collapse.

Since the advent of GANs, substantial efforts have been devoted to addressing these challenges BID21 BID9 BID19 , while non-adversarial approaches that are free of these issues have also gained attention.

Examples include variational autoencoders (VAEs) BID13 , reversible generative models BID5 BID14 , and Wasserstein autoencoders (WAEs) BID22 .However, non-adversarial approaches often have significant limitations.

For instance, VAEs tend to generate blurry samples, while reversible generative models require restricted neural network architectures or solving neural differential equations BID8 .

Furthermore, to use the change of variable formula, the latent space of a reversible model must have the same dimensionality as the data space, which is unreasonable considering that real-world, high-dimensional data (e.g., images) tends to lie on low-dimensional manifolds, and thus results in redundant latent dimensions and variability.

Intriguingly, recent research BID3 suggests that the discrepancy between the intrinsic and ambient dimensionalities of data also contributes to the difficulties in training GANs and VAEs.

In this work, we present a novel framework for training autoencoder-based generative models, with non-adversarial losses and unrestricted neural network architectures.

Given a standard autoencoder and a target data distribution, instead of matching the target distribution in the data space, we map both the generated and target distributions to the latent space using the encoder, and train the generator (or decoder) to minimize the divergence between the mapped distributions.

We prove, under mild assumptions, that by minimizing a form of latent reconstruction error, matching the target distribution in the latent space implies matching it in the data space.

We call this framework perceptual generative autoencoder (PGA).

We show that PGA enables training generative autoencoders with maximum likelihood, without restrictions on architectures or latent dimensionalities.

In addition, when combined with VAE, PGA can generate sharper samples than vanilla VAE.

1 2 METHODS 2.1 PERCEPTUAL GENERATIVE MODEL Let f : R D → R H be the encoder parameterized by φ, and g : R H → R D be the decoder parameterized by θ.

Our goal is to obtain a generative model, which maps a simple prior distribution to the data distribution, D. Throughout this paper, we use N (0, I) as the prior distribution.

For z ∈ R H , the output of the decoder, g (z), lies in a manifold that is at most H-dimensional.

Therefore, if we train the autoencoder to minimize DISPLAYFORM0 DISPLAYFORM1 , thenx can be seen as a projection of the input data, x, onto the manifold of g (z).

LetD denote the distribution ofx.

Given enough capacity of the encoder,D is the best approximation to D (in terms of L2 distance), that we can obtain from the decoder, and thus can serve as a surrogate target for training the generator.

Due to the difficulty of directly training the generator to matchD, we seek to mapD to the latent space, and train the generator to match the mapped distribution,Ĥ, in the latent space.

To this end, we reuse the encoder for mappingD toĤ, and train the generator such that h (·) = f (g (·)) maps N (0, I) toĤ. In addition, to ensure that g maps N (0, I) toD, we minimize the following latent reconstruction loss with respect to (w.r.t.) φ: DISPLAYFORM2 Formally, let Z (x) be the set of all z's that are mapped to the same x by the decoder, we have the following theorem: Theorem 1.

Assuming the convexity of Z (x) for all x ∈ R D , and sufficient capacity of the encoder; for z ∼ N (0, I), if Eq. (2) is minimized and h (z) ∼Ĥ, then g (z) ∼D.Proof.

We first show that any different x's generated by the decoder are mapped to different z's by the encoder.

Let x 1 = g (z 1 ), x 2 = g (z 2 ), and x 1 = x 2 .

Since the encoder has sufficient capacity and Eq. (2) is minimized, we have f ( DISPLAYFORM3 For z ∼ N (0, I), denote the distributions of g (z) and h (z), respectively, by D and H. We then consider the case where D andD are discrete distributions.

If g (z) D , then there exists an x, DISPLAYFORM4 The result still holds when D andD approach continuous distributions.

Note that the two distributions compared in Theorem 1, D andD, are mapped respectively from N (0, I) and H. While N (0, I) is supported on the whole R H , there can be z's with low probabilities in N (0, I), but with high probabilities in H, which are not well covered by Eq. (2).

Therefore, it is sometimes helpful to minimize another latent reconstruction loss on H: DISPLAYFORM5 By Theorem 1, the problem of training the generative model reduces to training h to map N (0, I) tô H, which we refer to as the perceptual generative model.

In the subsequent subsections, we present a maximum likelihood approach, as well as a VAE-based approach, to train the perceptual generative model.

We first assume the invertibility of h. Forx ∼D, letĤ be the distribution of f (x).

We can train h directly with maximum likelihood using the change of variable formula as DISPLAYFORM0 Ideally, we would like to maximize Eq. (4) w.r.t.

the parameters of the generator (or decoder), θ.

However, directly optimizing the first term in Eq. (4) requires computing z = h −1 (ẑ), which is usually unknown.

Nevertheless, forẑ ∼Ĥ, we have h −1 (ẑ) = f (x) and x ∼ D, and thus we can minimize the following loss function w.r.t.

φ instead: DISPLAYFORM1 To avoid computing the Jacobian for the second term in Eq. (4), which is slow for unrestricted architectures, we approximate the Jacobian determinant and derive a loss function for the decoder as DISPLAYFORM2 where S ( ) is a uniform distribution on a small hypersphere of radius .

When → 0, the approximation forms an upper bound on the right-hand side (r.h.s.) of Eq. FORMULA8 , and becomes tight if h is close to the identity function.

Intuitively, Eq. (5) attracts the latent representations of data samples to the origin, while Eq. (6) expands the volume occupied by each sample in the latent space.

The above discussion relies on the assumption that h is invertible, which is not necessarily true for unrestricted architectures.

If h (z) is not invertible for some z, the logarithm of the Jacobian determinant at z becomes infinite, in which case Eq. (4) cannot be optimized.

Nevertheless, since DISPLAYFORM3 2 is unlikely to be zero if the model is properly initialized, the approximation in Eq. (6) remains finite, and thus can be optimized regardless.

To summarize, we train the autoencoder to obtain a generative model by minimizing the following loss function: DISPLAYFORM4 where α, β, and γ are hyperparameters to be tuned.

We refer to this approach as maximum likelihood PGA (LPGA).

The original VAE is trained by maximizing the evidence lower bound on log p (x) as DISPLAYFORM0 where p (x | z) is modeled with the decoder, and q (z | x) is modeled with the encoder.

In our case, we would like to modify Eq. (8) in a way that helps maximize log p (ẑ).

Therefore, we replace p (x | z) on the r.h.s.

of Eq. (8) with p (ẑ | z), and derive a lower bound on log p (ẑ) as DISPLAYFORM1 Similar to the original VAE, we make the assumption that q (z | x) and p (ẑ | z) are Gaussian; i.e., DISPLAYFORM2 , and σ > 0 is a tunable scalar.

The VAE variant is trained by minimizing DISPLAYFORM3 Note that we slightly abuse the notation, since z is deterministic for the losses in Eqs. FORMULA2 and FORMULA5 , but is stochastic (with additive Gaussian noise) in Eq. (10).

Accordingly, the overall loss function is given by DISPLAYFORM4 We refer to this approach as variational PGA (VPGA).

In this section, we evaluate the performance of LPGA and VPGA on three image datasets, MNIST BID17 , CIFAR-10 (Krizhevsky & Hinton, 2009) , and CelebA BID18 .

For CelebA, we employ the discriminator and generator architecture of DCGAN for the encoder and decoder of PGA.

We half the number of filters (i.e., 64 filters for the first convolutional layer) for faster experiments, while more filters are observed to improve performance.

Due to smaller input sizes, we reduce the number of convolutional layers accordingly for MNIST and CIFAR-10, and add a fully-connected layer of 1024 units for MNIST, as done in .

SGD with a momentum of 0.9 is used to train all models.

The training process of PGA is stable in general, given the non-adversarial losses.

However, stability issues can occur when batch normalization BID11 is introduced, since both the encoder and decoder are fed with multiple batches drawn from different distributions.

In our experiments, we only use batch normalization when it does not cause stability issues, in which case it is observed to substantially accelerate convergence.

As shown in FIG1 , the visual quality of the PGA-generated samples is significantly improved over that of VAE.

In particular, VPGA generates much sharper samples on CIFAR-10 and CelebA compared to vanilla VAE.

For CelebA, we further show latent space interpolations in FIG1 .

In addition, we use the Fréchet Inception Distance (FID) BID10 to evaluate LPGA, VPGA, and VAE.

For each model and each dataset, we take 5,000 generated samples to compute the FID score.

The results are summarized in Table.

1.

Compared to other non-adversarial generative models BID22 BID15 BID6 , where similar but larger architectures are used, we obtain substantially better FID scores on CIFAR-10 and CelebA.

We proposed a framework, PGA, for training autoencoder-based generative models, with nonadversarial losses and unrestricted neural network architectures.

By matching target distributions in the latent space, PGA trained with maximum likelihood generalizes the idea of reversible generative models to unrestricted neural network architectures and arbitrary latent dimensionalities.

In addition, it improves the performance of VAE when combined together.

In principle, PGA can be combined with any method that can train the perceptual generative model.

While we have only considered two non-adversarial approaches, an interesting future work would be to combine PGA with an adversarial discriminator trained on latent representations.

Moreover, the compatibility issue with batch normalization deserves further investigation.

@highlight

A framework for training autoencoder-based generative models, with non-adversarial losses and unrestricted neural network architectures.

@highlight

This paper uses autoencoders to do distribution matching in high dimensional space.
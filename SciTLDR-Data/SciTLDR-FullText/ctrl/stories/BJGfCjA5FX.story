We propose a novel autoencoding model called Pairwise Augmented GANs.

We train a generator and an encoder jointly and in an adversarial manner.

The generator network learns to sample realistic objects.

In turn, the encoder network at the same time is trained to map the true data distribution to the prior in  latent space.

To ensure good reconstructions, we introduce an augmented adversarial reconstruction loss.

Here we train a discriminator to distinguish two types of pairs: an object with its augmentation and the one with its reconstruction.

We show that such adversarial loss compares objects based on the content rather than on the exact match.

We experimentally demonstrate that our model generates samples and reconstructions of quality competitive with state-of-the-art on datasets MNIST, CIFAR10, CelebA and achieves good quantitative results on CIFAR10.

Deep generative models are a powerful tool to sample complex high dimensional objects from a low dimensional manifold.

The dominant approaches for learning such generative models are variational autoencoders (VAEs) and generative adversarial networks (GANs) BID12 .

VAEs allow not only to generate samples from the data distribution, but also to encode the objects into the latent space.

However, VAE-like models require a careful likelihood choice.

Misspecifying one may lead to undesirable effects in samples and reconstructions (e.g., blurry images).

On the contrary, GANs do not rely on an explicit likelihood and utilize more complex loss function provided by a discriminator.

As a result, they produce higher quality images.

However, the original formulation of GANs BID12 lacks an important encoding property that allows many practical applications.

For example, it is used in semi-supervised learning , in a manipulation of object properties using low dimensional manifold BID7 and in an optimization utilizing the known structure of embeddings BID11 .VAE-GAN hybrids are of great interest due to their potential ability to learn latent representations like VAEs, while generating high-quality objects like GANs.

In such generative models with a bidirectional mapping between the data space and the latent space one of the desired properties is to have good reconstructions (x ≈ G(E(x))).

In many hybrid approaches BID30 BID34 BID36 BID3 BID33 as well as in VAE-like methods it is achieved by minimizing L 1 or L 2 pixel-wise norm between x and G(E(x)).

However, the main drawback of using these standard reconstruction losses is that they enforce the generative model to recover too many unnecessary details of the source object x. For example, to reconstruct a bird picture we do not need an exact position of the bird on an image, but the pixel-wise loss penalizes a lot for shifted reconstructions.

Recently, improved ALI model BID8 by introducing a reconstruction loss in the form of a discriminator which classifies pairs (x, x) and (x, G(E(x))).

However, in such approach, the discriminator tends to detect the fake pair (x, G(E(x))) just by checking the identity of x and G(E(x)) which leads to vanishing gradients.

In this paper, we propose a novel autoencoding model which matches the distributions in the data space and in the latent space independently as in BID36 .

To ensure good reconstructions, we introduce an augmented adversarial reconstruction loss as a discriminator which classifies pairs (x, a(x)) and (x, G(E(x))) where a(·) is a stochastic augmentation function.

This enforces the DISPLAYFORM0 discriminator to take into account content invariant to the augmentation, thus making training more robust.

We call this approach Pairwise Augmented Generative Adversarial Networks (PAGANs).Measuring a reconstruction quality of autoencoding models is challenging.

A standard reconstruction metric RMSE does not perform the content-based comparison.

To deal with this problem we propose a novel metric Reconstruction Inception Dissimilarity (RID) which is robust to content-preserving transformations (e.g., small shifts of an image).

We show qualitative results on common datasets such as MNIST BID19 , CIFAR10 BID17 and CelebA BID21 .

PAGANs outperform existing VAE-GAN hybrids in Inception Score BID31 and Fréchet Inception Distance BID14 except for the recently announced method PD-WGAN BID10 on CIFAR10 dataset.

Let us consider an adversarial learning framework where our goal is to match the true distribution p * (x) to the model distribution p θ (x).

As it was proposed in the original paper BID12 , the model distribution p θ (x) is induced by the generator G θ : z − → x where z is sampled from a prior p(z).

To match the distributions p * (x) and p θ (x) in an adversarial manner, we introduce a discriminator DISPLAYFORM0 It takes an object x and predicts the probability that this object is sampled from the true distribution p * (x).

The training procedure of GANs BID12 ) is based on the minimax game of two players: the generator G θ and the discriminator D ψ .

This game is defined as follows DISPLAYFORM1 where V (θ, ψ) is a value function for this game.

The optimal discriminator D ψ * given fixed generator G θ is DISPLAYFORM2 DISPLAYFORM3

The solution of the reconstruction matching problem ensures that reconstructions G θ (E ϕ (x)) correspond to the source object x up to defined random augmentations a(x).

In PAGANs model we introduce the minimax game for training the adversarial distance between the reconstructions and augmentations of the source object x. We consider the discriminator D ψ which takes a pair (x, y) and classifies it into one of the following classes:• the real class: pairs (x, y) from the distribution p * (x)r(y|x), i.e. the object x is taken from the true distribution p * (x) and the second y is obtained from the x by the random augmentation a(x); • the fake class: pairs (x, y) from the distribution DISPLAYFORM0 i.e. x is sampled from p * (x) then z is generated from the conditional distribution q ϕ (z|x) by the encoder E ϕ (x) and y is produced by the generator G ϕ (z) from the conditional model distribution p θ (y|z).Then the minimax problem is DISPLAYFORM1 where DISPLAYFORM2 Let us prove that such minimax game will match the distributions r(y|x) and p θ,ϕ (y|x).

At first, we find the optimal discriminator: Proposition 1.

Given a fixed generator G θ and a fixed encoder E ϕ , the optimal discriminator D ψ * is DISPLAYFORM3 Proof.

Given in Appendix A.1.Then we can prove that given an optimal discriminator the value function V (θ, ϕ, ψ) is equivalent to the expected Jensen-Shanon divergence between the distributions r(y|x) and p θ,ϕ (y|x).

Proposition 2.

The minimization of the value function V under an optimal discriminator D ψ * is equivalent to the minimization of the expected Jensen-Shanon divergence between r(y|x) and p θ,ϕ (y|x), i.e. DISPLAYFORM4 Proof.

Given in Appendix A.2.If r(y|x) = δ x (y) then the optimal discriminator D ψ * (x, y) will learn an indicator I{x = y} as was proved in .

As a consequence, the objectives of the generator and the encoder are very unstable and have vanishing gradients in practice.

On the contrary, if the distribution r(y|x) is non-degenerate as in our model then the value function V (θ, ϕ, ψ) will be well-behaved and much more stable which we observed in practice.

We obtain that for the generator and the encoder we should optimize the sum of two value functions:• the generator's objective: DISPLAYFORM0 • the encoder's objective: DISPLAYFORM1 Draw N samples from the dataset and the prior DISPLAYFORM2 Compute generator loss DISPLAYFORM3 Gradient update on discriminator networks DISPLAYFORM4 Gradient update on generator-encoder networks until convergenceIn practice in order to speed up the training we follow BID12 and use more stable objectives replacing log(1 − D(·)) with − log(D(·)).

See Figure 1 for the description of our model and Algorithm 1 for an algorithmic illustration of the training procedure.

We can straightforwardly extend the definition of PAGANs model to f -PAGANs which minimize the f -divergence and to WPAGANs which optimize the Wasserstein-1 distance.

More detailed analysis of these models is placed in Appendix C.

Recent papers on VAE-GAN hybrids explore different ways to build a generative model with an encoder part.

One direction is to apply adversarial training in the VAE framework to match the variational posterior distribution q(z|x) and the prior distribution p(z) BID23 or to match the marginal q(z) and p(z) BID22 BID33 .

Another way within the VAE model is to introduce the discriminator as a part of a data likelihood BID18 BID3 .

Within the GANs framework, a common technique is to regularize the model with the reconstruction loss term BID5 BID30 BID34 .Another principal approach is to train the generator and the encoder BID8 simultaneously in a fully adversarial way.

These methods match the joint distributions p * (x)q(z|x) and p θ (x|z)p(z) by training the discriminator which classifies the pairs (x, z).

ALICE model introduces an additional entropy loss for dealing with the non-identifiability issues in ALI model.

approximated the entropy loss with the cycle-consistency term which is equivalent to the adversarial reconstruction loss.

The model of BID27 puts ALI to the VAE framework where the same joint distributions are matched in an adversarial manner.

As an alternative, BID34 train generator and encoder by optimizing the minimax game without the discriminator.

Optimal transport approach is also explored, BID10 introduce an algorithm based on primal and dual formulations of an optimal transport problem.

In PAGANs model the marginal distributions in the data space p * (x) and p θ (x) and in the latent space p(z) and q(z) are matched independently as in BID36 .

Additionally, the augmented adversarial reconstruction loss is minimized by fooling the discriminator which classifies the pairs (x, a(x)) and (x, G(E(x))).

In this section, we validate our model experimentally.

At first, we compare PAGAN with other similar methods that allow performing both inference and generation using Inception Score and Fréchet Inception Distance.

Secondly, to measure reconstruction quality, we introduce Reconstruction Inception Dissimilarity (RID) and prove its usability.

In the last two experiments we show the importance of the adversarial loss and augmentations.

For the architecture choice we used deterministic DCGAN 1 generator and discriminator networks provided by pfnet-research 2 , the encoder network has the same architecture as the discriminator except for the output dimension.

The encoder's output is a factorized normal distribution.

Thus DISPLAYFORM0 ϕ (x)I) where µ ϕ , σ ϕ are outputs of the encoder network.

The discriminator D(z) architecture is chosen to be a 2 layer MLP with 512, 256 hidden units.

We also used the same default hyperparameters as provided in the repository and applied a spectral normalization following BID24 .

For the augmentation a(x) defined in Section 3 we used a combination of reflecting 10% pad and the random crop to the same image size.

The prior distribution p(z) is chosen to be a standard distribution N (0, I).

To evaluate Inception Score and Fréchet Inception Distance we used the official implementation provided in tensorflow 1.10.1 BID0 .To optimize objectives FORMULA2 , FORMULA2 , we need to have a discriminator working on pairs (x, y).

This can be done using special network architectures like siam networks BID4 or via an image concatenation.

The latter approach can be implemented in two concurrent ways: concatenating channel or widthwise.

Empirically we found that the siam architecture does not lead to significant improvement and concatenating width wise to be the most stable.

We use this configuration in all the experiments.

To see whether our method provides good quality samples from the prior, we compared our model to related works that allow an inverse mapping.

We performed our evaluations on CIFAR10 dataset since quantitative metrics are available there.

Considering Fréchet Inception Distance (FID), our model outperforms all other methods.

Inception Score shows that PAGANs significantly better than others except for recently announced PD-WGAN.

Quantitative results are given in TAB0 .

For S-VAE we report IS that is reproduced using officially provided code and hyperparameters 3 .

Plots with samples and reconstructions for CIFAR10 dataset are provided in Figure 2 .

Additional visual results for more datasets can be found in Appendix E.3.

BID33 87.7 4.18 ± 0.04 ALI 5.34 ± 0.04 AGE BID34 39.51 5.9 ± 0.04 ALICE 6.02 ± 0.03 S-VAE BID6 6.055 α-GANs BID30 6.2 AS-VAE BID28 6.3 PD-WGAN, λ mix = 0 BID10 33.0 6.70 ± 0.09 PAGAN (ours) 32.84 6.56 ± 0.06

The traditional approach to estimate the reconstruction quality is to compute RMSE distance from source images to reconstructed ones.

However, this metric suffers from focusing on exact reconstruction and is not content aware.

RMSE penalizes content-preserving transformations while allows such undesirable effect as blurriness which degrades visual quality significantly.

We propose a novel metric Reconstruction Inception Dissimilarity (RID) which is based on a pre-trained classification network and is defined as follows: DISPLAYFORM0 where p(y|x) is a pre-trained classifier that estimates the label distribution given an image.

Similar to BID31 we use a pre-trained Inception Network BID32 to calculate softmax outputs.

Low RID indicates that the content did not change after reconstruction.

To calculate standard deviations, we use the same approach as for IS and split test set on 10 equal parts 4 .

Moreover RID is robust to augmentations that do not change the visual content and in this sense is much better than RMSE.

To compare new metric with RMSE, we train a vanilla VAE with resnet-like architecture on CI-FAR10.

We compute RID for its reconstructions and real images with the augmentation (mirror 10% pad + random crop).

In TAB1 we show that RMSE for VAE is better in comparison to augmented images (AUG), but we are not satisfied with its reconstructions (see Figure 12 in Appendix E.4), Figure 3 provides even more convincing results.

RID allows a fair comparison, for VAE it is dramatically higher (44.33) than for AUG (1.57).

Value 1.57 for AUG says that KL divergence is close to zero and thus content is almost not changed.

We also provide estimated RID and RMSE for AGE that was publicly available 5 .

From TAB1 we see that PAGANs outperform AGE which reflects that our model has better reconstruction quality.

To prove the importance of an adversarial loss, we experiment replacing adversarial loss with the 4 Split is done sequentially without shuffling 5 Pretrained AGE: https://github.com/DmitryUlyanov/AGE TAB2 .

IS and FID results suggest that our model without adversarial loss performed worse in generation.

Reconstruction quality significantly dropped considering RID.

Visual results in Appendix E.1 confirm our quantitative findings.

In ALICE model an adversarial reconstruction loss was implemented without an augmentation.

As we discussed in Section 1 its absence leads to undesirable effects.

Here we run an experiment to show that our model without augmentation performs worse.

Quantitative results provided in TAB2 illustrate that our model without an augmentation fails to recover both good reconstruction and generation properties.

Visual comparisons can be found in Appendix E.2.

Using the results obtained from the last two experiments we conclude that adversarial reconstruction loss works significantly better with augmentation.

Experiments checking augmentation effects (see Appendix B for details) conclude the following.

A good augmentation: 1) is required to be non-deterministic, 2) should preserve the content of source image, 3) should be hard to use pixel-wise comparison for discriminator.

In this paper, we proposed a novel framework with an augmented adversarial reconstruction loss.

We introduced RID to estimate reconstructions quality for images.

It was empirically shown that this metric could perform content-based comparison of reconstructed images.

Using RID, we proved the value of augmentation in our experiments.

We showed that the augmented adversarial loss in this framework plays a key role in getting not only good reconstructions but good generated images.

Some open questions are still left for future work.

More complex architectures may be used to achieve better IS and RID.

The random shift augmentation may not the only possible choice, and other smart choices are also possible.

A.1 PROOF OF PROPOSITION 1 (OPTIMAL DISCRIMINATOR) Proposition 1.

Given a fixed generator G θ and a fixed encoder E ϕ , the optimal discriminator D ψ * is DISPLAYFORM0 Proof.

For fixed generator and encoder, the value function V (ψ) with respect to the discriminator is DISPLAYFORM1 Let us introduce new variables and notations DISPLAYFORM2 Then DISPLAYFORM3 Using the results of the paper BID12 we obtain DISPLAYFORM4 A.2 PROOF OF PROPOSITION 2Proposition 2.

The minimization of the value function V under an optimal discriminator D ψ * is equivalent to the minimization of the expected Jensen-Shanon divergence between r(y|x) and p θ,ϕ (y|x), i.e. DISPLAYFORM5 Proof.

As in the paper BID12 we rewrite the value function V (θ, ϕ) for the optimal discriminator D ψ * as follows DISPLAYFORM6 APPENDIX B CHOICE OF AUGMENTATION Augmentation choice might be problem specific.

Thereby we additionally study different augmentations and provide an intuition how to choose the right transformation.

Theory suggests to pick up a stochastic augmentation.

The practical choice should take into account the desired properties of reconstructions.

A random shift of an image by a small margin is sufficient to create good quality reconstructions.

However, this shift should not be large because it may inherit augmentation artifacts.

This can be spotted beforehand just looking at pairs (x, a(x)).

Once these pairs are not satisfactory, model reconstructions would be bad as well.

In this experiment, we investigate the effects caused by the padding and random crop augmentation.

We choose different padding size (comparatively to the original image size) and plot FID, RID and IS metrics.

The results provide the intuition to choose padding size (see Figure 4) .

Padding should be chosen to maintain visual content while making impossible to compare augmented and original images by nearly element-wise comparison.

Larger padding cause undesirable effects in reconstructions that are captured by RID (see Figure 5 ).

Visual quality of samples, on the other hand is slightly better with more aggressive augmentation considering FID metric, what is explained by more robust training due to less mode collapse problem.

We also checked two different augmentation types: Gaussian blur and random contrast (see Figures 6, 7) .

Both augmentations led to highly unstable training and did not yield satisfactory results (IS was 2.15 and 4.18 respectively).

Therefore we conclude that a good augmentation is better to change spatial image structure preserving content (as padding does) what will force the discriminator to take content into account.

• is required to be non-deterministic • should preserve the content of source image • should be hard to use pixel-wise comparison for discriminator APPENDIX C EXTENDING PAGANS C.1 f -DIVERGENCE PAGANS f -GANs BID26 are the generalization of GAN approach.

BID26 introduces the model which minimizes the f -divergence D f BID1 between the true distribution p * (x) and the model distibution p θ (x), i.e. it solves the optimization problem DISPLAYFORM0 where f : R + − → R is a convex, lower-semicontinuous function satisfying f (1) = 0.The minimax game for f -GANs is defined as DISPLAYFORM1 where V (θ, ψ) is a value function and f * is a Fenchel conjugate of f BID25 .

For fixed parameters θ, the optimal T ψ * (x) is f p * (x) DISPLAYFORM2 .

Then the value function V (θ, ψ * ) for optimal parameters ψ * equals to f -divergence between the distributions p * and p θ BID25 , i.e. DISPLAYFORM3

<|TLDR|>

@highlight

We propose a novel autoencoding model with augmented adversarial reconstruction loss. We intoduce new metric for content-based assessment of reconstructions. 
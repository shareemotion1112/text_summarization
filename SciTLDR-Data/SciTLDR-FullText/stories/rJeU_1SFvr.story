Training generative adversarial networks requires balancing of delicate adversarial dynamics.

Even with careful tuning, training may diverge or end up in a bad equilibrium with dropped modes.

In this work, we introduce a new form of latent optimisation inspired by the CS-GAN and show that it improves adversarial dynamics by enhancing interactions between the discriminator and the generator.

We develop supporting theoretical analysis from the perspectives of differentiable games and stochastic approximation.

Our experiments demonstrate that latent optimisation can significantly improve GAN training, obtaining state-of-the-art performance for the ImageNet (128 x 128) dataset.

Our model achieves an Inception Score (IS) of 148 and an Frechet Inception Distance (FID) of 3.4, an improvement of 17% and 32% in IS and FID respectively, compared with the baseline BigGAN-deep model with the same architecture and number of parameters.

Generative Adversarial Nets (GANs) are implicit generative models that can be trained to match a given data distribution.

GANs were originally proposed and demonstrated for images by Goodfellow et al. (2014) .

As the field of generative modelling has advanced, GANs have remained at the frontier, generating high-fidelity images at large scale (Brock et al., 2018) .

However, despite growing insights into the dynamics of GAN training, most recent advances in large-scale image generation come from architectural improvements (Radford et al., 2015; Zhang et al., 2019) , or regularisation focusing on particular parts of the model (Miyato et al., 2018; Miyato & Koyama, 2018) .

Inspired by the compressed sensing GAN (CS-GAN; Wu et al., 2019) , we further exploit the benefit of latent optimisation in adversarial games using natural gradient descent to optimise the latent variable z at each step of training, presenting a scalable and easy to implement approach to improve the dynamical interaction between the discriminator and the generator.

For clarity, we unify these approaches as latent optimised GANs (LOGAN).

To summarise our contributions:

1.

We present a novel analysis of latent optimisation in GANs from the perspective of differentiable games and stochastic approximation (Balduzzi et al., 2018; Heusel et al., 2017) , arguing that latent optimisation can improve the dynamics of adversarial training.

2.

Motivated by this analysis, we improve latent optimisation by taking advantage of efficient second-order updates.

3.

Our algorithm improves the state-of-the-art BigGAN-deep model (Brock et al., 2018) by a significant margin, without introducing any architectural change or additional parameters, resulting in higher quality images and more diverse samples (Figure 1 and 2).

We use θ D and θ G to denote the vectors representing parameters of the generator and discriminator.

We use x for images, and z for the latent source generating an image.

The prime is used to denote .

p(x) and p(z) denote the data distribution and source distribution respectively.

E p(x) [f (x)] indicates taking the expectation of function f (x) over the distribution p(x).

A GAN consists of a generator that generates image x = G(z; θ G ) from a latent source z ∼ p(z), and a discriminator that scores the generated images as D(x; θ D ) (Goodfellow et al., 2014) .

Training GANs involves an adversarial game: while the discriminator tries to distinguish generated samples x = G (z; θ G ) from data x ∼ p(x), the generator tries to fool the discriminator.

This procedure can be summarised as the following min-max game: The exact form of h(·) depends on the choice of loss function (Goodfellow et al., 2014; Arjovsky et al., 2017; Nowozin et al., 2016) .

To simplify our presentation and analysis, we use the Wasserstein loss (Arjovsky et al., 2017) , so that h D (t) = −t and h G (t) = t. Our experiments with BigGANdeep uses the hinge loss (Lim & Ye, 2017; Tran et al., 2017) , which is identical to this form in its linear regime.

Our analysis can be generalised to other losses as in previous theoretical work (e.g., Arora et al. 2017) .

To simplify notation, we abbreviate

, which may be further simplified as f (z) when the explicit dependency on θ D and θ G can be omitted.

Training GANs requires carefully balancing updates to D and G, and is sensitive to both architecture and algorithm choices (Salimans et al., 2016; Radford et al., 2015) .

A recent milestone is BigGAN (and BigGAN-deep, Brock et al. 2018) , which pushed the boundary of high fidelity image generation by scaling up GANs to an unprecedented level.

BigGANs use an architecture based on residual blocks (He et al., 2016) , in combination with regularisation mechanisms and self-attention (Saxe et al., 2014; Miyato et al., 2018; Zhang et al., 2019 ).

Here we aim to improve the adversarial dynamics during training.

We focus on the second term in eq. 1 which is at the heart of the min-max game, with adversarial losses for D and G, which can be written as

Computing the gradients with respect to θ D and θ G obtains the following gradient, which cannot be expressed as the gradient of any single function (Balduzzi et al., 2018) :

The fact that g is not the gradient of a function implies that gradient updates in GANs can exhibit cycling behaviour which can slow down or prevent convergence.

In Balduzzi et al. (2018) , vector fields of this form are referred to as the simultaneous gradient.

Although many GAN models use alternating update rules (e.g., Goodfellow et al. 2014; Brock et al. 2018) , following the gradient with respect to θ D and θ G alternatively in each step, they can still suffer from cycling, so we use the simpler simultaneous gradient (eq. 3) for our analysis.

Inspired by compressed sensing (Candes et al., 2006; Donoho, 2006) , Wu et al. (2019) introduced latent optimisation for GANs.

We call this type of model latent-optimised GANs (LOGAN).

Latent optimization has been shown to improve the stability of training as well as the final performance for medium-sized models such as DCGANs and Spectral Normalised GANs (Radford et al., 2015; Miyato et al., 2018) .

Latent optimisation exploits knowledge from D to guide updates of z. Intuitively, the gradient

∂z points in the direction that satisfies the discriminator D, which implies better samples.

Therefore, instead of using the randomly sampled z ∼ p(z), Wu et al. (2019) uses the optimised latent

in eq. 1 for training 1 .

The general algorithm is summarised in Algorithm 1 and illustrated in Figure 3 a. We develop the natural gradient descent form of latent update in Section 4.

and use it to obtain ∆z from eq. 4 (GD) or eq. 12 (NGD) Optimise the latent z ← [z + ∆z], [·] indicates clipping the value between −1 and 1 Compute generator loss L

Update θ D and θ G with the gradients

until reaches the maximum training steps 3 ANALYSIS OF THE ALGORITHM To understand how latent optimisation improves GAN training, we analyse LOGAN as a 2-player differentiable game following Balduzzi et al. (2018); Gemp & Mahadevan (2018) ; Letcher et al. (2019) .

The appendix provides a complementary analysis that relates LOGAN to unrolled GANs (Metz et al., 2016) and stochastic approximation (Heusel et al., 2017; Borkar, 1997) .

An important problem with gradient-based optimization in GANs is that the vector-field generated by the losses of the discriminator and generator is not a gradient vector field.

It follows that gradient descent is not guaranteed to find a local optimum and can cycle, which can slow down convergence or lead to phenomena like mode collapse and mode hopping.

Balduzzi et al. (2018) ; Gemp & Mahadevan (2018) proposed Symplectic Gradient Adjustment (SGA) to improve the dynamics of gradient-based methods in adversarial games.

For a game with gradient g (eq. 3), define the Hessian as the second order derivatives with respect to the parameters, H = ∇ θ g. SGA uses the adjusted gradient g * = g + λ A T g where λ is a positive constant (5) and A = 1 2 (H − H T ) is the anti-symmetric component of the Hessian.

Applying SGA to GANs yields the adjusted updates (see Appendix B.1 for details):

Compared with g in eq. 3, the adjusted gradient g * has second-order terms reflecting the interactions between D and G. SGA has been shown to significantly improve GAN training in basic examples (Balduzzi et al., 2018) , allowing faster and more robust convergence to stable fixed points (local Nash equilibria).

Unfortunately, SGA is expensive to scale because computing the second-order derivatives with respect to all parameters is expensive.

Explicitly computing the gradients for the discriminator and generator at z after one step of latent optimisation (eq. 4) obtains

In both equations, the first terms represent how f (z ) depends on the parameters directly and the second terms represent how f (z ) depends on the parameters via the optimised latent source.

For the second equality, we substitute ∆z = α

∂z as the gradient-based update of z and use

∂z .

The original GAN's gradient (eq. 3) does not include any second-order term, since ∆z = 0 without latent optimisation.

In LOGAN, these extra terms are computed by automatic differentiation when back-propagating through the latent optimisation process (see Algorithm 1).

The SGA updates in eq. 6 and the LOGAN updates in eq. 8 are strikingly similar, suggesting that the latent step used by LOGAN reduces the negative effects of cycling by introducing a symplectic gradient adjustment into the optimization procedure.

The role of the latent step can be formalized in terms of a third player, whose goal is to help the generator, see appendix B for details.

Crucially, latent optimisation approximates SGA using only second-order derivatives with respect to the latent z and parameters of the discriminator and generator separately.

The second order terms involving parameters of both the discriminator and the generator -which are extremely expensive to compute -are not used.

For latent z's with dimensions typically used in GANs (e.g., 128-256, orders of magnitude less than the number of parameters), these can be computed efficiently.

In short, latent optimisation efficiently couples the gradients of the discriminator and generator, as prescribed by SGA, but using the much lower-dimensional latent source z which makes the adjustment scalable.

An important consequence of reducing the rotational aspect of GAN dynamics is that it is possible to use larger step sizes during training which suggests using stronger optimisers to fully take advantage of latent optimisation.

Latent optimisation can improve GAN training dynamics further by allowing larger single steps ∆z towards the direction of ∂f (z) ∂z without overshooting.

Appendix B further explains how LOGAN relates to unrolled GANs (Metz et al., 2016) and stochastic approximation.

Our main finding is that latent optimisation accelerates the speed of updating D relative to that of G, facilitating convergence according to Heusel et al. (2017)

In particular, the generator requires less update compared with D to achieve the same reduction of loss, because latent optimisation "helps" G.

Although our analysis suggests using strong optimisers for optimising z, Wu et al. (2019) only used basic gradient descent (GD) with a fixed step-size.

This choice limits the size ∆z can take: in order not to overshoot when the curvature is large, the step size would be too conservative when the curvature is small.

We hypothesis that GD is more detrimental for larger models, which have complex loss surfaces with highly varying curvatures.

Consistent with this hypothesis, we observed only marginal improvement over the baseline using GD (section 5.3, Table In this work, we instead use natural gradient descent (NGD, Amari 1998) for latent optimisation.

NGD can be seen as an approximate second-order optimisation method (Pascanu & Bengio, 2013; Martens, 2014) , and has been applied successfully in many domains.

By using the positive semidefinite (PSD) Gauss-Newton matrix to approximate the (possibly negative definite) Hessian, NGD often works even better than exact second-order methods.

NGD is expensive in high dimensional parameter spaces, even with approximations (Martens, 2014) .

However, we demonstrate it is efficient for latent optimisation, even in very large models.

Given the gradient of z, g = ∂f (z) ∂z , NGD computes the update as

where the Fisher information matrix F is defined as

The log-likelihood function ln p(t|z) typically corresponds to commonly used error functions such as cross entropy loss.

This correspondence is not necessary when NGD is interpreted as an approximate second-order method, as has long been used in practice (Martens, 2014) .

Nevertheless, Appendix C provides a Poisson log-likelihood interpretation for the hinge loss commonly used in GANs (Lim & Ye, 2017; Tran et al., 2017 ).

An important difference between latent optimisation and commonly seen senarios using NGD is that the expectation over the condition (z) is absent.

Since each z is only responsible for generating one image, it only minimises the loss L G (z) for this particular instance.

Computing per-sample Fisher this way is necessary to approximate SGA (see Appendix B.1 for details).

More specifically, we use the empirical Fisher F with Tikhonov damping, as in TONGA (Roux et al., 2008 ) F = g · g T + β I (11) F is cheaper to compute compared with the full Fisher, since g is already available.

The damping factor β regularises the step size, which is important when F only poorly approximates the Hessian or when the Hessian changes too much across the step.

Using the Sherman-Morrison formula, the NGD update can be simplified into the following closed form:

which does not involve any matrix inversion.

Thus, NGD adapts the step size according to the curvature estimate c = 1 Figure 4 a illustrates the scaling for different values of β.

NGD automatically smooths the scale of updates by down-scaling the gradients as their norm grows, which also contributes to the smoothed norms of updates (Figure 4 b) .

Since the NGD update remains proportional to g, our analysis based on gradient descent in section 3 still holds.

We focus on large scale GANs based on BigGAN-deep (Brock et al., 2018 ) trained on 128 × 128 size images from the ImageNet dataset (Deng et al., 2009) .

In Appendix E, we present results from applying our algorithm on Spectral Normalised GANs trained with CIFAR dataset (Krizhevsky et al., 2009) , which obtains state-of-the-art scores on this model.

We used the standard BigGAN-deep architecture with three minor modifications: 1.

We increased the size of the latent source from 128 to 256, to compensate the randomness of the source lost when optimising z. 2.

We use the uniform distribution U(−1, 1) instead of the standard normal distribution N (0, 1) for p(z), to be consistent with the clipping operation (Algorithm 1).

3.

We use leaky ReLU instead of ReLU as the non-linearity for smoother gradient flow for

∂z .

Consistent with detailed findings in Brock et al. (2018) that these changes have limited effect, our experiment with this baseline model obtains only slightly better scores compared with those in Brock et al. (2018) (Table 1 , see also Figure 8 ).

The FID and IS are computed as in Brock et al. (2018) , and IS values are computed from checkpoints with the lowest FIDs.

The means and standard deviations are computed from 5 models with different random seeds.

To apply latent optimisation, we use a damping factor β = 5.0 combined with a large step size of α = 0.9.

As an additional way of damping, we only optimise 50% of z's dimensions.

Optimising the entire population of z was unstable in our experiments.

Similar to Wu et al. (2019), we found it was helpful to regularise the Euclidean norm of weight-change ∆z, with a regulariser weight of 300.0.

All other hyper-parameters, including learning rates and a large batch size of 2048, remain the same as in BigGAN-deep; we did not optimise these hyper-parameters.

We call this model LOGAN (NGD).

Employing the same architecture and number of parameters as the BigGAN-deep baseline, LOGAN (NGD) achieved better FID and IS (Table 1) .

As observed by Brock et al. (2018) , BigGAN training always eventually collapsed.

Training with LOGAN also collapsed, perhaps due to higher-order dynamics beyond the scope we have analysed, but took significantly longer (600k steps versus 300k steps with BigGAN-deep).

During training, LOGAN was 2 − 3 times slower per step compared with BigGAN-deep because of the additional forward and backward pass.

We found that optimising z during evaluation did not improve sample scores (even up to 10 steps), so we do not optimise z for evaluation.

Therefore, LOGAN has the same evaluation cost as original BigGAN-deep.

To help understand this behaviour, we plot the change from ∆z during training in Figure 5 a. Although the movement in Euclidean space ∆z grew until training collapsed, the movement in D's output space, measured as f (z + ∆z) − f (z) , remained unchanged (see Appendix D for details).

As shown in our analysis, optimising z improves the training dynamics, so LOGANs work well after training without requiring latent optimisation.

We verify our theoretical analysis in section 3 by examining key components of Algorithm 1 via ablation studies.

First, we experimented with using basic GD to optimising z, as in Wu et al. (2019), and call this model LOGAN (GD).

A smaller step size of α = 0.0001 was required; larger values were unstable and led to premature collapse of training.

As shown in Table 1 , the scores from LOGAN (GD) were worse than LOGAN (NGD) and similar to the baseline model.

We then evaluate the effects of removing those terms depending on ∂f (z) ∂z in eq. 8, which are not in the ordinary gradient (eq. 3).

Since these terms were computed when back-propagating through the latent optimisation procedure, we removed them by selectively blocking back-propagation with "stop gradient" operations (e.g., in TensorFlow Abadi et al. 2016 ).

T ∂f (z ) ∂z and removing both terms.

As predicted by our analysis (section 3), both terms help stabilise training; training diverged early for all three ablations.

Truncation is a technique introduced by Brock et al. (2018) to illustrate the trade-off between the FID and IS in a trained model.

For a model trained with z ∼ p(z) from a source distribution symmetric around 0, such as the standard normal distribution N (0, 1) and the uniform distribution U(−1, 1), down-scaling (truncating) the sourcez = s · z with 0 ≤ s ≤ 1 gives samples with higher visual quality but reduced diversity.

This observation is quantified as higher IS and lower FID when evaluating samples from truncated distributions.

Figure 2 b show higher quality compared with those in a (e.g., the interfaces between the elephants and ground, the contours around the pandas).

In this work we present the LOGAN model which significantly improves the state-of-the-art on large scale GAN training for image generation by online optimising the latent source z. Our results illustrate improvements in quantitative evaluation and samples with higher quality and diversity.

Moreover, our analysis suggests that LOGAN fundamentally improves adversarial training dynamics.

We therefore expect our method to be useful in other tasks that involve adversarial training, including representation learning and inference (Donahue et al., 2017; Dumoulin et al., 2017 ), text generation (Zhang et al., 2019) , style learning (Zhu et al., 2017; Karras et al., 2019) , audio generation and video generation (Vondrick et al., 2016; Clark et al., 2019 A ADDITIONAL SAMPLES AND RESULTS Figure 6 and 7 provide additional samples, organised similarly as in Figure 1 and 2.

Figure 8 shows additional truncation curves.

In this section we present three complementary analyses of LOGAN.

In particular, we show how the algorithm brings together ideas from symplectic gradient adjustment, unrolled GANs and stochastic approximation with two time scales.

To analyse LOGAN as a differentiable game we treat the latent step ∆z as adding a third player to the original game played by the discriminator and generator.

The third player's parameter, ∆z, is optimised online for each z ∼ p(z).

Together the three players (latent player, discriminator, and generator) have losses averaged over a batch of samples:

where η = 1 N (N is the batch size) reflects the fact that each ∆z is only optimised for a single sample z, so its contribution to the total loss across a batch is small compared with θ D and θ G which are directly optimised for batch losses.

This choice of η is essential for the following derivation, and has important practical implication.

It means that the per-sample loss L G (z ), instead of the loss summed over a batch N n=1 L G (z n ), should be the only loss function guiding latent optimisation.

Therefore, when using natural gradient descent (Section 4), the Fisher information matrix should only be computed using the current sample z.

The resulting simultaneous gradient is

Following Balduzzi et al. (2018) , we can write the Hessian of the game as:

The presence of a non-zero anti-symmetric component in the Hessian

implies the dynamics have a rotational component which can cause cycling or slow down convergence.

Since η 1 for typical batch sizes (e.g., Symplectic gradient adjustment (SGA) counteracts the rotational force by adding an adjustment term to the gradient to obtain g * ← g + λ A T g, which for the discriminator and generator has the form:

The gradient with respect to ∆z is ignored since the convergence of training only depends on θ D and θ G .

If we drop the last terms in eq.17 and 18, which are expensive to compute for large models with high-dimensional θ D and θ G , and use

∂z , the adjusted updates can be rewritten as

Because of the third player, there are still the terms depend on

to adjust the gradients.

Efficiently computing

is non-trivial (e.g., Pearlmutter 1994).

However, if we introduce the local approximation

then the adjusted gradient becomes identical to 8 from latent optimisation.

In other words, automatic differentiation by commonly used machine learning packages can compute the adjusted gradient for θ D and θ G when back-propagating through the latent optimisation process.

Despite the approximation involved in this analysis, both our experiments in section 5 and the results from Wu et al. (2019) verified that latent optimisation can significantly improve GAN training.

Latent optimisation can be seen as unrolling GANs (Metz et al., 2016) in the space of the latent, rather than the parameters.

Unrolling in the latent space has the advantages that:

1.

LOGAN is more scalable than Unrolled GANs because it avoids second-order derivatives over a potentially very large number of parameters.

2.

While unrolling the update of D only affects the parameters of G (as in Metz et al. 2016) , latent optimisation effects both D and G as shown in eq. 8.

We next formally present this connection by showing that SGA can be seen as approximating Unrolled GANs (Metz et al., 2016) .

For the update θ D = θ D + ∆θ D , we have the Taylor expansion approximation at θ D :

Here p(t = 1; z, D, G) is the probability that the generated image G(z) can fool the discriminator D.

The original GAN's discriminator can be interpreted as outputting a Bernoulli distribution

In this case, if we parameterise β G = D (G(z)), the generator loss is the negative log-likelihood

Bernoulli, however, is not the only valid choice as the discriminator's output distribution.

Instead of sampling "1" or "0", we assume that there are many identical discriminators that can independently vote to reject an input sample as fake.

The number of votes k in a given interval can be described by a Poisson distribution with parameter λ with the following PMF:

The probability that a generated image can fool all the discriminators is the probability of G(z) receiving no vote for rejection

(40) Therefore, we have the following negative log-likelihood as the generator loss if we parameterise λ = −D (G(z)):

This interpretation has a caveat that when D (G(z)) > 0 the Poisson distribution is not well defined.

However, in general the discriminator's hinge loss

pushes D (G(z)) < 0 via training.

For a temporal sequence x 1 , x 2 , . . . , x T (changes of z or f (z) at each training step in this paper), to normalise its variance while accounting for the non-stationarity, we process it as follows.

We first compute the moving average and standard deviation over a window of size N :

Then normalise the sequence as:

The result in Figure 5 a is robust to the choice of window size.

Our experiments with N from 10 to 50 yielded visually similar plots.

To test if latent optimisation works with models at more moderate scales, we applied it on SN-GANs (Miyato et al., 2018) .

Although our experiments on this model are less thorough than in the main paper with BigGAN-deep, we hope to provide basic guidelines for researchers interested in applying latent optimisation on smaller models.

The experiments follows the same basic setup and hyper-parameter settings as the CS-GAN in Wu et al. (2019) .

There is no class conditioning in this model.

For NGD, we found a smaller damping factor β = 0.1, a z regulariser weight of 3.0 (the same as in Wu et al. 2019) , combined with optimising 70% of the latent source (instead of 50% for BigGAN-deep) worked best for SN-GANs.

In addition, we found running extra latent optimisation steps benefited evaluation, so we use ten steps of latent optimisation in evaluation for results in this section, although the models were still trained with a single optimisation step.

We reckon that smaller models might not be "over-parametrised" enough to fully amortise the computation from optimising z, which can then further exploit the architecture in evaluation time.

On the other hand, the overhead from running multiple iterations of latent optimisation is relatively small at this scale.

We aim to further investigate this difference in future studies.

Table 2 shows the FID and IS alongside SN-GAN and CS-CAN which used the same architecture.

Here we observe similarly significant improvement over the baseline SN-GAN model, with an improvement of 16.8% in IS and 39.6% in FID.

Figure 9 shows random samples from these two models.

Overall, samples from LOGAN (NGD) have higher contrasts and sharper contours.

@highlight

Latent optimisation improves adversarial training dynamics. We present both theoretical analysis and state-of-the-art image generation with ImageNet 128x128.
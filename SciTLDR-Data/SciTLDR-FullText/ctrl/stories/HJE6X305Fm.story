Generative Adversarial Networks are one of the leading tools in generative modeling, image editing and content creation.

However, they are hard to train as they require a delicate balancing act between two deep networks fighting a never ending duel.

Some of the most promising adversarial models today minimize a Wasserstein objective.

It is smoother and more stable to optimize.

In this paper, we show that the Wasserstein distance is just one out of a large family of objective functions that yield these properties.

By making the discriminator of a GAN robust to adversarial attacks we can turn any GAN objective into a smooth and stable loss.

We experimentally show that any GAN objective, including Wasserstein GANs, benefit from adversarial robustness both quantitatively and qualitatively.

The training additionally becomes more robust to suboptimal choices of hyperparameters, model architectures, or objective functions.

Generative adversarial networks (GANs) BID4 are at the forefront of generative modeling.

They cast generative modeling as a never ending duel between two networks: a generator network produces synthetic samples from random noise and a discriminator network distinguishes synthetic from real data samples.

GANs produce visually appealing samples, but are often hard to train.

Much recent work, tries to stabilize GAN training through novel architecture BID25 BID3 or training objectives BID17 BID6 .In this paper, we show that GAN objectives significantly stabilize, if the discriminator is robust to adversarial attacks.

Specifically, we show that a robust discriminator leads to a robust minimax objective for the generator irrespective of the training objective used.

In addition, any training objective is smooth and Lipschitz as long as the discriminator is Lipschitz.

Finally, we show that this robustness does not need to be enforced for every single input of the discriminator, but rather just in expectation over the generated samples.

We present two new regularization terms, borrowed from the adversarial training literature.

Both terms ensure the robustness of the discriminator.

They are easy to optimize and can be added to any existing GAN objective.

Our experiments show that adversarial robustness both improves the visual quality of the results, as well as stabilizes the training procedure across a wide range of architectures, hyper-parameters and training objectives.

We will publish the code and data used to perform our experiments upon acceptance.

The scope of generative modeling changed significantly with the emergence of photo realistic image generation like Generative Adversarial Networks (GANs) BID4 , Variational AutoEncoders BID13 ), or Pixel Convolutional Networks (Oord et al., 2016 .

The visual quality of these generative models enabled applications such as, generative image processing BID24 , image editing BID29 , and image translation .

They also started the race for ever prettier generated images.

Two major areas of improvement are architectures BID25 BID3 and loss functions BID17 standard discriminator robust discriminator Figure 1 : Results from four generative adversarial models.

The leftmost model is trained with a standard discriminator, while the models on the right use an increasingly more robust discriminator.

We show that a robust discriminator leads to a smoother and more stable training objective, resulting in a better generative model.

Best viewed on screen.

BID6 .

BID3 showed that a Laplacian pyramid generator greatly improves the quality of the generated images, while BID25 showed impressive results by carefully balancing the expressive power of the generator and discriminator networks.

In this work, we use Radford et al. BID25 architecture, and solely focus on the loss function.

GANs come in three popular loss functions: the original Jensen-Shanon divergence objective BID4 , least squares GANs (LSGAN) BID17 , and Wasserstein distance (WGAN) .

Each objectives minimizes special case of a f-divergence between the generative and data distribution BID20 .

augment the GAN objective with a smoothness condition in the form of a Lipschitz-1 constraint on the discriminator.

However, enforcing a Lipschitz-1 discriminator is difficult.

clip the weights of the discriminator to enforce the Lipschitz condition, while BID6 apply a penalty on the gradient of the discriminator.

The gradient penalty (WGAN-GP) generally yields better results, however it looses some of the appealing theoretical properties of WGANs.

Arjovsky et al. show that in theory most GAN objective are non-continuous and non-Lipschitz, while the Wasserstein objective is Lipschitz, smooth, and more stable to optimize.

In this paper, we extend their analysis and show that any GAN objective can be made smooth as long as the discriminator is robust to adversarial perturbations.

In addition, the discriminator only needs to be robust to adversarial attacks in expectation over all generated samples.

A simple penalty function is sufficient to carry the theoretical smoothness results of WGAN other GAN objectives.

Our theoretical analysis is related to BID1 , who study the geometry to common distance measures for generative modeling.

We build on their analysis and show that any GAN objective is Lipschitz and smooth, as long as the generator and discriminator are Lipschitz.

Adversarial Examples Despite success of deep networks, their differentiable nature makes them vulnerabilities to adversarial attacks.

Small perturbation of the input can significantly change the output of a network BID27 .

The initial work of BID27 set off an arms race between novel attack methods BID5 BID2 BID22 BID19 and defenses BID11 BID23 ) against these attacks.

The goal of an attacker is to find an perturbation in an attack region that yields a large change in output.

Some of the fastest attacks simply perturb the input in the gradient direction of the network BID5 , while more complex attacks optimize for an attack vector BID2 .

Many attacks can be directly used in a defense through dataset augmentation BID23 .

In this work, we primarily use the fast gradient methods of BID5 , in particular we use the normalized gradient attack.

Let z ∼ P Z be a sample from a noise distribution P Z , e.g. uniform noise.

Let x ∼ P R be a sample form a data distribution P R .

The goal of sampling based generative modeling is to find a function G(z) that maps samples from P Z to P R .

More specifically, we want P Y = {y = G(z)|z ∼ P Z } to closely resemble P R .Generative Adversarial Networks A generative adversarial network (GAN) optimizes a two player game between a generative model G and a discriminator D. The generator G maps noise to data samples.

A discriminator D then judges if the transformed noise is close enough to the true data distribution P R .

A GAN jointly optimizes both generator and discriminator in a minimax game.

The GAN objective L minimizes a generator G and maximizes a discriminator D over DISPLAYFORM0 where f is differentiable, usually concave, loss function.

The generator minimizes DISPLAYFORM1 In practice, both generator G(z; θ) and discriminator D(x; φ) are deep neural networks with parameters θ and φ respectively.

The discriminator D produces a single scalar output, which is then passed through the loss function f .

Different choices of f lead to different GAN models.

The original GAN objective uses a sigmoid log likelihood loss for f (x) = − log(1 + exp(−x)), and corresponds to the Jensen-Shannon divergence between data and generator BID4 .

A Euclidean loss f (x) = − 1 2 (1 + x) 2 leads to least squares GAN (LSGAN) BID17 .

A Lipschitz-1 discriminator D and identity loss functions f (x) = x corresponds to the Wasserstein distance in WGAN .

In all these examples f is concave, and for GAN and LSGAN non-positive.

Adversarial attacks We follow the standard adversarial attack definition in literature BID2 BID11 .

|h(x) − h(x + ∆)| < ε for all ∆ p < δ, where the p-norm · p defines the local attack region.

For simplicity of the analysis, we focus on distance norms | · | instead of general distance functions of BID2 .

Our definition includes logit pairing of BID11 .Definition 3.1 focuses on individual inputs x drawn from an empirical data distribution of training or testing images.

We extend this definition to generative distributions G(z).

Definition 3.2 (expected robustness).

A function h, e.g. a discriminator, is robust to adversarial perturbations for a generative distribution G(z) if and only if DISPLAYFORM0 where the p-norm · p defines the local attack region, and d measures the distance between the network outputs.

This definition only requires robustness in expectation, but not for every single sample DISPLAYFORM1 For notational simplicity, we define the additive combination of a generator G and a perturbation function u as G + u where (G + u)(z) = G(z) + u(z).

In this section, we show that the GAN objective is robust to adversarial perturbations, as long as the discriminator is robust.

This property even holds as the discriminator adapts with the perturbation applied.

We first analyze the case when the loss of the discriminator is robust to an adversarial attack, and then study the robustness of the discriminator directly.

then the adversarial objective is robust DISPLAYFORM0 Proof.

Let D G be the optimal discriminator for generator G. For this discriminator we have DISPLAYFORM1 by definition.

This allows us to bound the difference in the objective for a generator G and its perturbation G + u: DISPLAYFORM2 We can equivalently bound (G + u) − (G) < ε, and obtain | (G) − (G + u)| < ε.

The above proof uses the (global) optimality of the discriminator in Equation (3).

However, the bound holds for any discriminator that improves the objective, without reaching a local or global minima, and could thus be generalized.

Next, we focus our attention to robust discriminators.

Here, we rely on an additional assumption that all loss functions f we consider are concave.

This is the case for the three most popular architectures: original GAN, LSGAN and WGAN.

Theorem 4.2 (robust discriminator).

For a concave loss f and a discriminator D that is robust to perturbations u(z) p < δ in expectation DISPLAYFORM3 Proof.

The proof follows directly from the definition of concavity f (a) − f (b) ≤ f (a)(a − b) and the fairly conservative bound |f (a) − f (b)| ≤ max(|f (a)|, |f (b)|)|a − b|.

In expectation, this bound reduces the robust discriminator to a robust loss DISPLAYFORM4 The rest of the proof follows Theorem 4.1.For the original GAN C < 1, for WGAN C = 1, and for DISPLAYFORM5 where |1 + D(G(z))| is directly minimized in the objective and is close to zero.

Theorem 4.1 and Theorem 4.2 have some interesting implications for general GANs.

First, any GAN that is trained with a robust discriminator or loss has a robust and hence smooth objective.

This directly extends the theoretical properties of WGANs ) to other GAN model.

It further relaxes the strict Lipschitz constraints to adversarial robustness in expectation.

This is much easier to enforce in practice as we will show in the next section.

Second, Theorem 4.2 allows us to analyze the continuity and Lipschitzness of any GAN objective.

In particular, we can show any GAN objective is continuous or Lipschitz in its parameters as long as the discriminator and generator are continuous or Lipschitz.

This is in direct contradiction to , that show only WGAN to be continuous.

However, their counter example relies on a discontinuous discriminator with infinitely large weights.

See supplemental material for a detailed derivation and proof.

Robust discriminators or losses can be hard to train in practice, as they require constrained optimization or a carefully tuned architecture.

Next, we show how a regularized discriminator objective can lead to a robust discriminator without the need for any constraints.

Our goal is to train a discriminator D such that it is robust to adversarial perturbations in expectation.

We do this by augmenting the original discriminator training objective with an additional adversarial regularization DISPLAYFORM0 In adversarial defense, this is known as distillation BID23 or logit pairing BID11 , where v is the attack vector in an attack region v p < δ.

For the main evaluation, we find the attack vector using the fast normalized gradient attack BID5 : DISPLAYFORM1 .

Additional results using other attack methods are in the supplementary material.

This adversarial regularization is sufficient to ensure a robust discriminator.

Theorem 4.3 (regularized robustness).

For a non-positive loss f and a regularized discriminator objective maximize DISPLAYFORM2 the optimal discriminator D * is robust DISPLAYFORM3 Proof.

We know that the objective value of the optimal discriminator is larger than the objective of a constant zero discriminator: DISPLAYFORM4 where the last inequality holds due to the non-positivity of the objective.

Using Jensen's inequality we can further reduce DISPLAYFORM5 As we increase the weight λ of the robustness term, the bound in Theorem 4.3 tightens and the discriminator is more robust to adversarial perturbations.

A similar bound can be derived for an absolute instead of squared loss ρ.

While the adversarial regularization ρ yields all the nice theoretical benefits of a robust discriminator, in practice it often does not provide enough regularization.

A much stronger regularization is to match the features φ(x) of the penultimate layer of the discriminator network D(x) = w φ(x).

We call this robust feature matching (RFM).

Our robust feature matching minimizes DISPLAYFORM6 where v is the attack direction, w is the weight of the last linear layer, and α ∈ R + is a hyperparameter.

We use δ = 0.05 and α = 10 −4 in all our experiments.

Similar to Theorem 4.3, robust feature matching ensures the robustness of the optimal discriminator DISPLAYFORM7 For a detailed proof see supplement.

Next, we will give some intuition on how adversarial defense changes the loss landscape of a GAN on a small toy example.

We verify the above robustness properties on the simple one dimensional Dirac-GAN of BID18 .

It allows us to separate practical optimization issues from the form and regularity of the loss function.

We can easily optimize the discriminator of Dirac-GAN to convergence with or without robustness constraints or penalties.

In the Dirac-GAN, the true data distribution p D is represented by a Dirac distribution centered at zero.

The generator produces samples from another Dirac distribution centered at θ, where θ is the only learnable generator parameter.

The discriminator is a linear function D(x) = ψx with a single scalar parameter ψ.

This experimental setup is analogous to the two dimensional toy example of .

The GAN objective of Dirac-GAN reduces to L(G) = max ψ f (ψθ) + f (0).

We plot this objective in FIG0 for unconstrained GAN, LSGAN and WGAN respectively.

Note, that all objectives are zero if the data and generative distributions match and constant otherwise.

Without any regularization, neither one of these objectives will provide a meaningful gradient signal.

Adding a robust loss FIG0 , Theorem 4.1, or a robust discriminator FIG0 , Theorem 4.2, will smooth out all loss functions and provide a meaningful gradient signal throughout training.

Finally, the regularized loss shows an equally smooth loss curve.

This serves as an illustrative example of the effect of robustness on the overall loss function of a GAN.

We perform all our experiments on the MNIST BID15 , CIFAR10 BID14 and CelebA BID16 datasets.

We use a slightly modified DCGAN BID25 architecture.

All convolutional blocks are replaced with residual blocks BID7 , the generator employs batch normalization BID9 and ReLU nonlinearities, while the discriminator uses instance normalization BID28 and Leaky ReLU.

We train using a weight decay term λ = 10 −4 , with batch size n = 64 and optimize using ADAM (Kingma & Ba, 2014) with h = 2 · 10 −4 and β 0 = 0, β 1 = 0.9 for 50 epochs on CIFAR10 and 25 epochs on CelebA. We use a latent vector of dimension z = 128 and use a unit Gaussian for our sampling distribution.

For complete training and architecture details see supplement.

Robust GAN training We compare original Jenson-Shannon divergence loss (GAN) BID4 , the least squares loss (LSGAN) BID17 , and the Wasserstein distance (WGAN) on CIFAR10 and CelebA. Along with these losses, we add a regularization method: Instance Noise (IN) BID26 , Gradient Penalty (GP) BID6 , Adversarial Regularization (AR) ρ, and Robust Feature Matching (RFM) ρ r .

We tuned the hyper-parameters for each combination of loss functions and robustness terms (or none).We present quantitative results in terms of Fréchet Inception Distance (FID) BID8 , as well as qualitative results.

The FID score measures the distance between two Gaussian distributions that are estimated from the deep feature statistics from the real and generated distributions.

Distributions that are close have a lower FID score, distributions that are large have a higher score.

Heusel et al. FORMULA0 showed that the FID metric corresponds well with human judgment of image quality.

In our experiments, we use FID as the primary quantitative measure of image quality.

As we are mainly interested in the relative improvements of different methods we did not extensively tune the architecture.

See the supplementary for additional details on experimental setup.

TAB1 shows the quantitative results.

Without any regularization the original and least squares objective work best, while the linear (WGAN) objective does not train well.

This is not surprising, as an unconstrained linear objective easily large discriminator outputs and unstable gradients.

Adding instance noise (IN) exacerbates this problem for CIFAR-10, but slightly helps for Celeb-A. The gradient penalty (GP) fixes some of the instabilities for both CIFAR-10 and Celeb-A, and helps most for the linear (WGAN) model.

However, the gradient penalty does not nearly perform as well as our robust regularizations (AR and RFM), both of which improve the quality of the synthesized images throughout different GAN objectives.

FIG1 show the corresponding qualitative results.

More generated images as well as additional results measuring the effect of different adversarial attacks can be found in supplementary material.

Stability across architectures/hyperparameters We follow BID6 and evaluate the robustness of our regularization across a variety of experimental setups on CIFAR-10.

We randomly sample 55 different experiment setups from a large pool of commonly used losses (JS, LSGAN, WGAN), batch size (8, 64), learning rate (10 −3 , 10 −4 ), network blocks (vanilla convolution, ResNet), filter size (32, 128), nonlinearities (ReLU, LeakyReLU, Tanh) and normalizations (BatchNorm, InstanceNorm, none).

For each random setup, we train four separate models: without regularization, with additive instance noise, with gradient penalty, and with RFM.

We rank them from 1 to 4 in order of increasing FID score.

We then count how many times a given method performs the best and worst out of the batch.

Table 2 shows the results.

Our robust feature matching consistently outperform other regularization, performing best in over two thirds of the experimental setups.

Experimentally, a robust loss or penalty leads to better convergence, see supplement.

However, the theoretical implications of adversarial robustness on local convergence is not yet well understood and warrants further investigation.

None Noise GP RFM # Top Perf.

1 6 11 37 # Worst Perf.

39 9 4 3 Average Rank 3.6 2.7 2.2 1.5 Table 2 :

Performance under different experimental setups.

In this paper, we established a clear connection between robust discriminators in generative adversarial networks and the overall smoothness of the optimization and the quality of the results.

To our knowledge, we are the first to show that a robustness regularization guarantees a smooth and robust loss function for any GAN objective.

Finally, our results suggest that robust regularization leads to better training and visual results than standard gradient penalties.

We start by showing that any GAN with a continuous generator and disciminator has itself a continuous objective.

Corollary A.1.

If generator, discriminator and loss function f F in a GAN are continuous, then the GAN objective (G(θ)) is continuous in the generator parameters θ.

Proof.

By definition of continuity we have G(z; θ) − G(z; θ + ∆ θ ) < ε G for all ∆ θ < δ θ and all noise samples z, and |f DISPLAYFORM0 The same properties hold for Lipschitz functions.

Corollary A.2.

In a GAN, if the generator is K-Lipschitz and the discriminator is L-Lipschitz, then the GAN objective is (KLC)-Lipschitz in the generator parameters θ, where C is the expected slope of the loss function.

Proof.

By definition of Lipschitz continuity any perturbation u(z) = G(z; θ + ∆ θ ) − G(z; θ) is bounded u(z) < K ∆ θ .

Furthermore the composition of Lipschitz functions is bounded by DISPLAYFORM1 Hence the GAN objective is (KLC)-Lipschitz.

Here we show that for a optimal discriminator D * (x) = w φ(x) in robust feature matching DISPLAYFORM0 is robust.

Theorem B.1.

For a non-positive loss f and a robust feature matching objective DISPLAYFORM1 Proof.

We know that the objective value of the optimal discriminator is larger than the objective of a constant zero discriminator: DISPLAYFORM2 where the last inequality holds due to the non-positively of the objective.

For every element of the RFM objective DISPLAYFORM3

To better understand the effects that a robust loss or discriminator have on the GAN objective function, we use the Dirac-GAN setup.

Each example in FIG3 shows the vector field of the given GAN objective as a function of (θ, ψ), where the initial points θ 0 = ψ 0 = 1, and the global optimum θ * = ψ * = 0.

We ran simultanious updates, for an unconstrained GAN, a GAN with hard robustness constraints, and with a robustness regularization.

The unregularized GAN does not converge.

The hard constraint GAN comes within distance d of the global optimum, where d depends on the strength of the robustness.

The smaller ε the smaller is d. Finally, adversarial regularization leads to a convergent Dirac-GAN objective.

However, a thorough theoretical analysis is warranted to fully understand this local convergence property.

In our main experiments we use an FGM attack (scaled normalized gradient) for simplicity, as other attacks have more hyperparameters such as number of iterations and constants.

To show, that this is a result not exclusive to FGM, we provide additional experiments using defenses against other popular attacks -Fast Gradient Sign Method (FGSM), Projected Gradient Descent (PGD) and Carlini Wagner (CW) using robust feature matching (RFM) and adversarial regularization (AR).

As expected, these defense techniques also provide improvements to the FID score.

Finally, we present additional generations for various models.(a) Robust discriminator using WGAN+RFM.(b) Robust loss using GAN+DS.

<|TLDR|>

@highlight

A discriminator that is not easily fooled by adversarial example makes GAN training more robust and leads to a smoother objective.

@highlight

This paper proposes a new way to stabilize the training process of GAN by regularizing the Discriminator to be robust to adversarial examples.

@highlight


The paper proposes a systematic way of training GANs with robustness regularization terms, allowing for smoother training of GANs. 

@highlight

Presents idea that making a discriminator robust to adversarial perturbations the GAN objective can be made smooth which results in better results both visually and in terms of FID.
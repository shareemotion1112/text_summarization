We propose a new form of an autoencoding model which incorporates the best properties of variational autoencoders (VAE) and generative adversarial networks (GAN).

It is known that GAN can produce very realistic samples while VAE does not suffer from mode collapsing problem.

Our model optimizes λ-Jeffreys divergence between the model distribution and the true data distribution.

We show that it takes the best properties of VAE and GAN objectives.

It consists of two parts.

One of these parts can be optimized by using the standard adversarial training, and the second one is the very objective of the VAE model.

However, the straightforward way of substituting the VAE loss does not work well if we use an explicit likelihood such as Gaussian or Laplace which have limited flexibility in high dimensions and are unnatural for modelling images in the space of pixels.

To tackle this problem we propose a novel approach to train the VAE model with an implicit likelihood by an adversarially trained discriminator.

In an extensive set of experiments on CIFAR-10 and TinyImagent datasets, we show that our model achieves the state-of-the-art generation and reconstruction quality and demonstrate how we can balance between mode-seeking and mode-covering behaviour of our model by adjusting the weight λ in our objective.

Variational autoencoder (VAE) (Kingma et al., 2014; Rezende et al., 2014; Titsias & Lázaro-Gredilla, 2014 ) is one of the most popular approaches for modeling complex high-dimensional distributions.

It has been applied successfully to many practical problems.

It has several nice properties such as learning low-dimensional representations for the objects and ability to conditional generation.

Due to an explicit reconstruction term in its objective, one may ensure that VAE can generate all objects from the training set.

These advantages, however, come at a price.

It is a known fact that VAE tends to generate unrealistic objects, e.g., blurred images.

Such behaviour can be explained by the properties of a maximum likelihood estimation (MLE) which is used to fit a restricted VAE model p θ (x) in data that comes from a complex distribution p with an equiprobable mixture of two Gaussians with learnable location and scale.

Plots a)-c) show pairwise comparisons of optimal log-densities, the plot d) compares optimal densities themselves.

).

This way, we encourage our model to be mode-seeking while still having relatively high values of p θ (x) on all objects from a training set, thus preventing the mode-collapse.

We note that J λ (p θ (x) p * (x)) is not symmetric with respect to p θ (x) and p * (x) and by the weight λ we can balance between mode-seeking and mass-covering behaviour.

However, the straightforward way of substituting each KL term with GAN and VAE losses does not work well in practice if we use an explicit likelihood for object reconstruction in VAE objective.

Such simple distributions as Gaussian or Laplace that are usually used in VAE have limited flexibility and are unnatural for modelling images in the space of pixels.

To tackle this problem we propose a novel approach to train the VAE model in an adversarial manner.

We show how we can estimate the implicit likelihood in our loss function by an adversarially trained discriminator.

We theoretically analyze the introduced loss function and show that under assumptions of optimal discriminators, our model minimizes the λ-Jeffreys divergence J λ (p θ (x) p * (x)) and we call our method as Implicit λ-Jeffreys Autoencoder (λ-IJAE).

In an extensive set of experiments, we evaluate the generation and reconstruction ability of our model on CIFAR10 (Krizhevsky et al., 2009) and TinyImagenet datasets.

It shows the state-of-the-art trade-off between generation and reconstruction quality.

We demonstrate how we can balance between the ability of generating realistic images and the reconstruction ability by changing the weight λ in our objective.

Based on our experimental study we derive a default choice for λ that establishes a reasonable compromise between mode-seeking and mass-covering behaviour of our model and this choice is consistent over these two datasets.

Relation to forward KL-based methods.

We can say that all VAE-based models minimize the upper bound on the forward KL D KL (p * (x) p θ ).

In recent years there have been many extensions and improvements of the standard VAE.

One direction of research is to inroduce the discriminator as a part of data likelihood (Larsen et al., 2015; Brock et al., 2017) to leverage its intermediate layers for measuring similarity between objects.

However, these models do not have a sound theoretical justification about what distance between p θ (x) and p * (x) they optimize.

The other way is to consider more complex variational distribution q ϕ (z|x).

One can either use better variational bounds (Agakov & Barber, 2004; Maaløe et al., 2016; Ranganath et al., 2016; Molchanov et al., 2018; Sobolev & Vetrov, 2019) or apply the adversarial training to match q ϕ (z|x) and the prior distribution p(z) (Mescheder et al., 2017) or to match the marginals q ϕ (z) and p(z) (Makhzani et al., 2016) .

Although these methods improve approximate inference in VAE model, they remain in the scope of MLE framework.

As we discussed above within this framework the model with a limited capacity is going to have the mass-covering behaviour.

Relation to reverse KL-based methods.

The vanilla GAN framework is equivalent under the assumption of optimal discriminator to minimization of Jensen-Shanon divergence JSD(p * (x) p θ (x)) (Goodfellow et al., 2014) .

With a minor modification of a generator loss we can obtain the equivalence to minimization of the reverse KL D KL (p θ (x) p * (x)) (Huszar, 2016; Arjovsky & Bottou, 2017) .

There have been proposed many autoencoder models which utilize one of these two divergences JSD(p

One approach is to minimize the divergence between joint distributions p * (x)q(z|x) and p θ (x|z)p(z) in a GAN framework (Donahue et al., 2017; Dumoulin et al., 2017) .

ALICE model (Li et al., 2017) introduces an additional entropy loss for dealing with the non-identifiability issues in previous works.

Other methods (Chen et al., 2018; Pu et al., 2017a; Rosca et al., 2017; Ulyanov et al., 2018; Zhu et al., 2017) use the reverse KL D KL (p θ (x) p * (x)) as an additional term to encourage mode-seeking behaviour.

Relation to Jeffreys divergence-based methods.

To the best of our knowledge, there are only two other autoencoder models which minimize λ-Jeffreys divergence for λ = 0.5.

It is an important case when λ-Jeffreys divergence equals symmetric KL divergence.

These methods are AS-VAE (Pu et al., 2017a) and SVAE (Chen et al., 2018) and they are most closely related to our work.

AS-VAE is a special case of SVAE method therefore further we will consider only SVAE.

There are two most crucial differences between SVAE and λ-IJAE models.

The first one is that SVAE minimizes

) between marginal distributions p * (x) and p θ (x) for arbitrary λ.

The second difference is that the SVAE's loss J λ (p * (x)q(z|x) p θ (x|z)p(z)) solely did not give good reconstructions in experiments.

Therefore, authors introduced additional data-fit terms E p * (x)q θ (z|x) log p θ (x|z) + E p θ (x|z)p(z) log q ϕ (z|x) where p θ (x|z) and q ϕ (z|x) are explicit densities.

In contrast, λ-IJAE model achieves good generation and reconstruction quality as it is and allows training implicit p θ (x|z) and q ϕ (z|x) distributions.

These two differences make SVAE and λ-IJAE models significantly distinct, and we observe it in practice.

Consider samples x ∈ X from the true data distribution p * (x).

The aim of generative models is to fit a model distribution p θ (x) to p * (x).

Most popular models are GAN and VAE.

In practice, we observe that they have significantly different properties.

VAE tends to cover all modes of p * (x) at the cost of capturing low probability regions as well.

As a result, it often generates unspecific and/or blurry images.

On the other hand, GAN is highly mode-seeking, i.e. it tends to concentrate most of its probability mass in a small number of modes of p * (x).

Therefore it may not cover significant part of p * (x) which is also known as a mode collapse problem.

Such radical contrast between VAE and GAN can be explained by the fact that they optimize different divergences between p θ (x) and p * (x).

Variational Inference.

VAE is trained by MLE: max θ E p * (x) log p θ (x).

The distribution p θ (x) is defined as an integral over a latent variable z: p θ (x) = p θ (x|z)p(z)dz, and in practice it is typically intractable.

Variational inference (Hinton & Van Camp, 1993) sidesteps this issue by introducing an encoder model (also known as a variational distribution) q ϕ (z|x) and replacing the intractable log p θ (x) with a tractable evidence lower bound (ELBO):

Then we maximize ELBO L ELBO (θ, ϕ) with respect to θ and ϕ. One can easily derive that MLE is equivalent to optimizing the forward KL D KL (p * p θ ):

Adversarial Training.

The adversarial framework is based on a game between a generator G θ (z) and a discriminator D ψ (x) which classifies objects from p * (x) versus ones from p θ (x):

Goodfellow et al. (2014) showed that the loss of the generator (3) is equivalent to the Jensen-Shanon

It is easy to recognize this as an instance of classification-based Density Ratio Estimation (DRE) (Sugiyama et al., 2012) .

Following this framework, one can consider different generator's objectives while keeping the same objective (3) for the discriminator.

DRE relies on the fact that

.

By this approach we can obtain a likelihood-free estimator for the reverse D KL (p θ p * ) (Huszar, 2016):

(Un)Biased Gradients in Adversarial Training.

Since in practice the discriminator D ψ (x) is only trained to work for one particular set of generator parameters θ, we need to be cautious regarding validity of gradients obtained by DRE approach.

For example, consider the forward KL D KL (p * p θ ).

If we apply DRE, we will arrive at E p * (x) log

.

However, we can notice that in practice this expression does not depend on θ in any way, i.e.

.

This is because the forward KL depends on θ only through the ratio of densities, which is replaced by a point estimate using a discriminator which has no idea regarding p θ 's local behaviour.

This shows we need to be careful when designing adversarial learning objective as to ensure unbiased gradients.

Luckily, JSD(p θ p * ) and D KL (p θ p * ) are not affected by this problem:

for any x. Then

Proof.

Given in Appendix, section A.

VAE provide a theoretically sound way to learn generative models with a natural and coherent encoder.

However, they are known to generate blurry and unspecific samples that have inferior perceptual quality compared to generative models based on adversarial learning.

The main cause for that is that the root principle VAEs are built upon -MLE framework -is equivalent to minimization of the forward KL D KL (p * p θ ).

While D KL (p * p θ ) recovers the true data-generating process p * (x) if the model p θ (x) has enough capacity, in a more realistic case of an insufficiently expressive model p θ (x) it is known to be mass-covering.

As a result, the model is forced to cover all modes of p * (x) even at the cost of covering low-probability regions as well.

This in turn might lead to blurry samples as the model does not have the capacity to concentrate inside the modes.

On the other hand, the reverse KL D KL (p θ p * ) has mode-seeking behavior that penalizes covering low-probability regions and thus the model p θ (x) tends to cover only a few of the modes of p * (x).

Following this reasoning, we propose a more balanced divergence -one that seeks modes, but still does a decent job covering all modes of p * (x) to prevent mode collapse.

We chose λ-Jeffreys divergence (Jeffreys, 1998)

We illustrate the advantage of λ-Jeffreys divergence for λ = 0.5 over Forward KL, Reverse KL and JSD divergences in the case of a model with limited capacity in Figure 1 .

In this figure we compared divergences in a simple task (see Appendix, section B) of approximating a mixture of 4 Gaussians with a mixture of just two: both Reverse KL and JSD exhibit mode-seeking behavior, completely dropping side modes, whereas the Forward KL assigns much more probability to tails and does poor job capturing the central modes.

On a contrast, λ-Jeffreys divergence uses one mixture component to capture the most probable mode, and the other to ensure mass-covering.

The optimization of λ-Jeffreys divergence consists of two parts.

The first one is the minimization of the reverse KL D KL (p θ (x) p * (x)) which can be implemented as a standard GAN optimization as we discussed in Section 3.

The second part is the optimization of the forward KL D KL (p * (x) p θ (x)) and we tackle it by maximization of the ELBO L ELBO (θ, ϕ) as in VAE.

So, we obtain an upper bound on λ-Jeffreys divergence by incorporating GAN and VAE objectives:

The ELBO term L ELBO (θ, ϕ) can be decomposed into two parts: (i) a reconstruction term

.

While both terms are easy to deal with in cases of explicit p(x|z) and q(z|x), an implicit formulation poses some challenges.

In the next two sections we address them.

Typically to optimize the reconstruction term E p * (x) E qϕ(z|x) log p θ (x|z) the conditional likelihood p θ (x|z) is defined explicitly as a fully factorized Gaussian or Laplace distribution (Kingma et al., 2014; Rezende et al., 2014; Titsias & Lázaro-Gredilla, 2014; Pu et al., 2017b; Chen et al., 2018; Rosca et al., 2017; Mescheder et al., 2017) .

While convenient, such choice might limit the expressivity of the generator G θ (z).

As we discussed previously, optimization of the forward KL(p * p θ ) leads to a mass-covering behavior.

The undesired properties of this behavior such as sampling unrealistic and/or blurry images can be more significant if a capacity of our model p θ (x) is limited.

Therefore we propose a technique which allows to extend the class of possible likelihoods for p θ (x|z) to implicit ones.

We note that typically in VAE the decoder p θ (x|z) first maps the latent code z to the space X , which is then used to parametrize the distribution of z's decodings x|z.

For example, this is the case for N (x|G θ (z), σI) or Laplace(x|G θ (z), σI).

We also use the output of the generator G θ (z) ∈ X to parametrize an implicit likelihood.

In particular, we assume p θ (x|z) = r(x|G θ (z)) for some symmetric likelihood r(x|y):

(ii) r(x = a|y = b) has a mode at a = b.

While the Gaussian and Laplace likelihoods are symmetric and explicit, in general we do not require r(x|y) to be explicit, only being able to generate samples from r(x|y) is required.

The idea is to introduce a discriminator D τ (x, z, y) which classifies two types of triplets:

• fake class: (x, z, y) ∼ p * (x)q ϕ (z|x)r (y|G θ (z)).

We note that r(y|x) and r (y|x) can be different and we will utilize this possibility in practice.

Then we train the discriminator D τ (x, z, y) using the standard binary cross-entropy objective:

If we apply the straightforward way to obtain an objective for the generator G θ (z) we will derive that we should minimize E p * (x)qϕ(z|x) log Dτ (x,z,x) 1−Dτ (x,z,x) .

Indeed, given an optimal discriminator for (4) D τ * (x, z, y) = r(y|x) r(y|x)+r (y|G θ (z)) , we obtain: Dτ (x,z,x) given the optimal D τ * (x, z, y) is equivalent to maximizing the reconstruction term with p θ (x|z) = r (x|G θ (z)).

However, we face in practice the same issue as we discussed in Section 3 that ∇ θ E p * (x)qϕ(z|x) log Dτ (x,z,x) 1−Dτ (x,z,x) = 0 because D τ (x, z, x) does not depend on θ explicitly even for optimal τ = τ * .

We can overcome this issue by exploiting the properties of symmetric likelihoods if we minimize a slightly different loss for the generator

The following theorem guarantees the gradients will be unbiased in the optimal discriminator case: Theorem 1.

Let D τ * (x, z, y) be the optimal solution for the objective (4) and r(y|x) and r (y|x) are symmetric likelihoods.

Then

Proof.

Given in Appendix, section A.

So, we obtain that we can maximize the reconstruction term E p * (x) E qϕ(z|x) log r(x|G θ (z)) by minimizing −E p * (x) E qϕ(z|x) log Dτ (x,z,G θ (z)) 1−Dτ (x,z,G θ (z)) and optimize it using gradient based methods.

We note again that we do not require an access to an analytic form of r(y|G θ (z)).

It is an open question what is the best choice for the r(y|G θ (z)).

Our expectations from r(y|G θ (z)) are that it should encourage realistic reconstructions and highly penalize for visually distorted images.

In experiments, as r(y|x) we use a distribution over cyclic shifts in all directions of an image x.

This distribution is symmetric with respect to all directions and has a mode in x, therefore it is the symmetric likelihood (see Definition 1 for details).

Although in practice we use r(y|x) which has an explicit form due to non-optimality of D τ (x, z, y) (that is always the case when training on finite datasets) the ratio log Dτ (x,z,G θ (z)) 1−Dτ (x,z,G θ (z)) sets implicit likelihood of reconstructions.

We can think of the non-optimality of D τ (x, z, y) as a form of regularization that allows us to convert explicit function r(y|x) into implicit likelihood that has desirable properties, i.e. encourages realistic reconstructions of x and penalizes unrealistic ones.

Implicit Encoder.

The KL term from L ELBO (θ, ϕ) can be computed either analytically, using the Monte Carlo estimation or by the adversarial manner.

We chose the latter approach proposed by Mescheder et al. (2017) because it enables implicit variational distribution q ϕ (z|x) defined by a neural sampler (encoder) E ϕ (x, ξ) where ξ ∼ N (·|0, I).

For this purpose we should train a discriminator D ζ (x, z) which tries to distinguish pairs (x, z) from p * (x)q ϕ (z|x) versus the ones from p

) is a reverse KL with respect to parameters ϕ, therefore we can substitute it by the expression −E qϕ(z|x) log

Putting it all together we arrive at the following objective:

In practice, discriminators are not optimal therefore we train our model by alternating gradients.

We maximize objectives (3), (4), (5)

In experiments, we evaluate generation and reconstruction ability of our model on datasets CIFAR-10 and TinyImageNet.

We used a standard ResNet architecture (Gulrajani et al., 2017) for the encoder E ϕ (x, ξ), the generator G θ (z) and for all three discriminators D ψ (x), D τ (x, z, y), D ζ (x, z).

The complete architecture description for all networks and hyperparameters used in λ-IJAE can be found in Appendix, section D. To compare our method to other autoencoding methods in the best way, we also used official and publicly available code for baselines.

For AGE 1 we use a pretrained model.

For SVAE 2 , TwoStage-VAE (2SVAE) 3 we report metrics reproduced using officially provided code and hyperparameters.

For α-GAN we also use public implementation 4 with same architecture as in λ-IJAE.

In experiments, for symmetric likelihoods r(y|x) and r (y|x) we use the following: r(y|x) is a continuous distribution over cyclic shifts in all directions of an image x. In practice, we discretize this distribution.

To sample from it: (i) we sample one of four directions (top, bottom, right, left) equally probable; (ii) then sample the size of a shift (maximum size S = 5 pixels) from 0 to S

(iii) as a result, we shift an image x to the selected direction in a size which is sampled.

For r (y|x) in practice we observe that the best choice is when r (y|x) is close to a delta function δ x (y).

Therefore, we use r (y|x) = N (y|x, σI) which is clearly a symmetric likelihood.

We set σ = 10 −8 .

For r(y|x) as an implicit likelihood we also studied a distribution over small rotations of x, however, we observed that cyclic shifts achieve better results.

Evaluation.

We evaluate our model on both generation and reconstruction tasks.

The quality of the former is assessed using Inception Score (IS) (Salimans et al., 2016) .

To calculate these metrics we used the official implementation provided in tensorflow 1.13 (Abadi et al., 2015) .

The reconstruction quality is evaluated using LPIPS, proposed by (Zhang et al., 2018) .

LPIPS compares images based on high-level features obtained by the pre-trained network.

It was show by Zhang et al. (2018) that LPIPS is a good metric which captures perceptual similarity between images.

We use the official implementation (LPIPS github) to compute LPIPS.

Ablation Study.

To show the importance of the implicit conditional likelihood r(y|x) we compare λ-IJAE with its modification which has instead of implicit r(y|x) a standard Gaussian or Laplace distribution.

We call such models λ-IJAE-L 2 and λ-IJAE-L 1 respectively.

In Figure 2 we compare λ-IJAE with λ-IJAE-L 2 and λ-IJAE-L 1 in terms of IS (generation quality) and LPIPS (reconstruction quality).

We see that λ-IJAE significantly outperforms these baselines and allows to achieve paretooptimal results for different choice of λ.

Comparison with Baselines.

We assess generation and reconstruction quality of λ-IJAE on CIFAR-10 and TinyImageNet datasets.

We compare the results to closest baselines with publicly available code.

We provide visual results in Appendix, section C. Quantitative results are given in Figure 3 and in Table 1 .

In Figure 3 we compare the methods with respect to IS and LPIPS.

Considering both metrics λ-IJAE achieves a better trade-off between reconstruction and generation quality within these Reconstruction and generation quality on CIFAR10 and TinyImagenet for models that allow reconstructions.

Baseline models were trained using publicly available code, if possible, to fill reconstruction quality metrics.

↓ -lower is better, ↑ -higher is better, best is marked with bold.

WAE (Tolstikhin et al., 2017) 4.18 ± 0.04 ALI (Dumoulin et al., 2017) ) 5.34 ± 0.04 ALICE (Li et al., 2017) 6.02 ± 0.03 AS-VAE (Pu et al., 2017b) 6.3 VAE (resnet) 3.45 ± 0.02 0.09 ± 0.03 2Stage-VAE (Dai & Wipf, 2019) 3.85 ± 0.03 0.06 ± 0.03 α-GAN (Rosca et al., 2017) 5.20 ± 0.08 0.04 ± 0.02 AGE (Ulyanov et al., 2018) 5.90 ± 0.04 0.06 ± 0.02 SVAE (Chen et al., 2018) 6.56 ± 0.07 0.19 ± 0.08

6.98 ± 0.1 0.07 ± 0.03

TinyImagenet AGE (Ulyanov et al., 2018) 6.75 ± 0.09 0.27 ± 0.09 SVAE (Chen et al., 2018) 5.09 ± 0.05 0.28 ± 0.08 2Stage-VAE (Dai & Wipf, 2019) 4.22 ± 0.05

6.87 ± 0.09 0.09 ± 0.03 datasets.

We see that small values of λ give a good IS score while remain the decent reconstruction quality in terms of LPIPS.

However, if decrease λ further LPIPS will start to degrade.

Therefore, we chose the λ = 0.3 as a reasonable trade-off between generation and reconstruction ability of λ-IJAE.

For this choice of λ we compute the results for Table 1 .

From these Table 1 we see that λ-IJAE achieves the state-of-the-art trade-off between generation and reconstruction quality.

It confirms our justification about λ-Jeffreys divergence that it takes the best properties of both KL divergences.

In the paper, we considered a fusion of VAE and GAN models that takes the best of two worlds: it has sharp and coherent samples and can encode observations into low-dimensional representations.

We provide a theoretical analysis of our objective and show that it is equivalent to the Jeffreys divergence.

In experiments, we demonstrate that our model achieves a good balance between generation and reconstruction quality.

It confirms our assumption that the Jeffreys divergence is the right choice for learning complex high-dimensional distributions in the case of the limited capacity of the model.

Proof.

Now we will show the second term is equal to zero given our assumptions:

Where we have used the (1) and (2) properties of the likelihoods r(x|y) (Definition 1):

To generate the plot 1 we considered the following setup: a target distribution was a mixture:

While the model as an equiprobable mixture of two learnable Gaussians:

The optimal θ was found by making 10,000 stochastic gradient descent iterations on Monte Carlo estimations of the corresponding divergences with a batch size of 1000.

We did 50 independent runs for each method to explore different local optima and chose the best one based on a divergence estimate with 100,000 samples Monte Carlo samples.

@highlight

We propose a new form of an autoencoding model which incorporates the best properties of variational autoencoders (VAE) and generative adversarial networks (GAN)
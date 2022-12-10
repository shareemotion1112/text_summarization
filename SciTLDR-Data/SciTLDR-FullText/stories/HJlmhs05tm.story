Unsupervised learning is about capturing dependencies between variables and is driven by the contrast between the probable vs improbable configurations of these variables, often either via a generative model which only samples probable ones or with an energy function (unnormalized log-density) which is low for probable ones and high for improbable ones.

Here we consider learning both an energy function and  an efficient approximate sampling mechanism for the corresponding distribution.

Whereas the critic (or discriminator) in generative adversarial networks (GANs) learns to separate data and generator samples, introducing an entropy maximization regularizer on the generator can turn the interpretation of the critic into an energy function, which separates the training distribution from everything else, and thus can be used for tasks like anomaly or novelty detection.



This paper is motivated by the older idea of sampling in latent space rather than data space because running a Monte-Carlo Markov Chain (MCMC) in latent space has been found to be easier and more efficient, and because a GAN-like generator can convert latent space samples to data space samples.

For this purpose, we show how a Markov chain can be run in latent space whose samples can be mapped to data space, producing better samples.

These samples are also used for the negative phase gradient required to estimate the log-likelihood gradient of the data space energy function.

To maximize entropy at the output of the generator, we take advantage of recently introduced neural estimators of mutual information.

We find that in addition to producing a useful scoring function for anomaly detection, the resulting approach produces sharp samples (like GANs) while covering the modes well, leading to high Inception and Fréchet scores.

The early work on deep learning relied on unsupervised learning BID13 BID2 BID17 ) to train energy-based models BID18 , in particular Restricted Boltzmann Machines, or RBMs.

However, it turned out that training energy-based models without an analytic form for the normalization constant is very difficult, because of the challenge of estimating the gradient of the partition function, also known as the negative phase part of the log-likelihood gradient (described in more details below, Sec. 2).

Several algorithms were proposed for this purpose, such as Contrastive Divergence BID12 and Stochastic Maximum Likelihood BID28 BID26 , relying on Monte-Carlo Markov Chains (MCMC) to iteratively sample from the energy-based model.

However, because they appear to suffer from either high bias or high variance (due to long mixing times), training of RBMs and other Boltzmann machines has not remained competitive after the introduction of variational auto-encoders BID16 ) and generative adversarial networks or GANs .In this paper, we revisit the question of training energy-based models, taking advantage of recent advances in GAN-related research, and propose a novel approach to training energy functions and sampling from them, called EnGAN.

The main inspiration for the proposed solution is the earlier observation BID4 made on stacks of auto-encoders that sampling in latent space (and then applying a decoder to map back to data space) led to faster mixing and more efficient sampling.

The authors observed that whereas the data manifold is generally very complex and curved, the corresponding distribution in latent space tends to be much simpler and flatter.

This was verified visually by interpolating in latent space and projecting back to data space through the decoder, observing that the resulting samples look like data samples (i.e., the latent space manifold is approximately convex, with most points interpolated between examples encoded in latent space also having high probability).

We propose a related approach, EnGAN, which also provides two energy functions, one in data space and one in latent space.

A key ingredient of the proposed approach is the need to regularize the generator (playing the role of the decoder in auto-encoders, but with no need for an encoder) so as to increase its entropy.

This is needed to make sure to produce negative examples that can kill off spurious minima of the energy function.

This need was first identified by BID15 , who showed that in order for an approximate sampler to match the density associated with an energy function, a compromise must be reached between sampling low energy configurations and obtaining a high-entropy distribution.

However, estimating and maximizing the entropy of a complex high-dimensional distribution is not trivial, and we take advantage for this purpose of very recently proposed GAN-based approaches for maximizing mutual information BID1 BID24 , since the mutual information between the input and the output of the generator is equal to the entropy at the output of the generator.

In this context, the main contributions of this paper are the following:• proposing EnGAN, a general architecture, sampling and training framework for energy functions, taking advantage of an estimator of mutual information between latent variables and generator output and approximating the negative phase samples with MCMC in latent space, • showing that the resulting energy function can be successfully used for anomaly detection, improving on recently published results with energy-based models, • showing that EnGAN produces sharp images -with competitive Inception and Frechet scores -and which also better cover modes than standard GANs and WGAN-GPs, while not suffering from the common blurriness issue of many maximum likelihood generative models.

Let x denote a sample in the data space X and E θ : X → R an energy function corresponding to minus the logarithm of an unnormalized density density function DISPLAYFORM0 where Z θ := e −E θ (x) dx is the partition function or normalizing constant of the density sample in the latent space.

Let p D be the training distribution, from which the training set is drawn.

Towards optimizing the parameters θ of the energy function, the maximum likelihood parameter gradient is DISPLAYFORM1 where the second term is the gradient of log Z θ , and the sum of the two expectations is zero when training has converged, with expected energy gradients in the positive phase (under the data p D ) matching those under the negative phase (under p θ (x)).

Training thus consists in matching the shape of two distributions: the positive phase distribution (associated with the data) and the negative phase distribution (where the model is free-running and generating configurations by itself).

This observation has motivated the pre-GAN idea presented by BID3 that "model samples are negative examples" and a classifier could be used to learn an energy function if it separated the data distribution from the model's own samples.

Shortly after introducing GANs, Goodfellow (2014) also made a similar connection, related to noise-contrastive estimation BID11 .

One should also recognize the similarity between Eq. 2 and the objective function for Wasserstein GANs or WGAN .

In the next section, we examine a way to train what appears to be a particular form of WGAN that makes the discriminator compute an energy function.

The main challenge in Eq. 2 is to obtain samples from the distribution p θ associated with the energy function E θ .

Although having an energy function is convenient to obtain a score allowing to compare the relative probability of different x's, it is difficult to convert an energy function into a generative function.

The commonly studied approaches for this are based on Monte-Carlo Markov chains, in which one iteratively updates a candidate configuration, until these configurations converge in distribution to the desired distribution p θ .

For the RBM, the most commonly used algorithms have been Contrastive Divergence BID12 and Stochastic Maximum Likelihood BID28 BID26 , relying on the particular structure of the RBM to perform Gibbs sampling.

Although these MCMC-based methods are appealing, RBMs (and their deeper form, the deep Boltzmann machine) have not been competitive in recent years compared to autoregressive models BID27 ), variational auto-encoders (Kingma & Welling, 2014 ) and generative adversarial networks or GANs .What has been hypothesized as a reason for the poorer results obtained with energy-based models trained with an MCMC estimator for the negative phase gradient is that running a Markov chain in data space is fundamentally difficult when the distribution is concentrated (e.g, near manifolds) and has many modes separated by vast areas of low probability.

This mixing challenge is discussed by BID4 who argue that a Markov chain is very likely to produce only sequences of highly probable configurations.

If two modes are far from each other and only local moves are possible (which is typically the case with MCMCs), it becomes exponentially unlikely to traverse the 'desert' of low probability which can separate two modes.

This makes mixing between modes difficult in high-dimensional spaces with strong concentration of probability mass in some places (e.g. corresponding to different categories) and very low probability elsewhere.

In the same papers, the authors propose a heuristic method for jumping between modes, based on performing the random walk not in data space but in the latent space of an auto-encoder.

Data samples can then be obtained by mapping the latent samples to data space via the decoder.

They argue that auto-encoders tend to flatten the data distribution and bring the different modes closer to each other.

The EnGAN sampling method proposed here is highly similar but leads to learning both an energy function in data space and one in latent space, from which we find that better samples are obtain.

The energy function can be used to perform the appropriate Metropolis-Hastings rejection.

Having an efficient way to approximately sample from the energy function also opens to the door to estimating the log-likelihood gradient with respect to the energy function according to Eq. 2, as outlined below.

Turning a GAN discriminator into an energy function has been studied in the past BID15 BID31 BID6 but in order to turn a GAN discriminator into an energy function, a crucial and difficult requirement is the maximization of entropy at the output of the generator.

Let's see why.

In Eq. 2, we can replace the difficult to sample p θ by another generative process, say p G , such as the generative distribution associated with a GAN generator: DISPLAYFORM0 where Ω is a regularizer which we found necessary to avoid numerical problems in the scale (temperature) of the energy.

In this paper we use a gradient norm regularizer BID10 ) DISPLAYFORM1 2 for this purpose.

This is similar to the training objective of a WGAN as to Eq. 2, but this interpretation allows to train the energy function only to the extent that p G is sufficiently similar to p θ .

To make them match, consider optimizing G to minimize the KL divergence KL(p G ||p θ ), which can be rewritten in terms of minimizing the energy of the samples from the generator while maximizing the entropy at the output of the generator: DISPLAYFORM2 as already shown by BID15 .

When taking the gradient of KL(p G ||p θ ) with respect to the parameters w of the generator, the partition function of p G disappears and we equivalently can optimize w to minimize DISPLAYFORM3 where p z is the prior distribution of the latent variable of the generator.

In order to maximize the entropy at the output of the generator, we propose to exploit another GANderived framework in order to estimate and maximize mutual information between the input and output of the generator network.

The entropy at the output of a deterministic function (the generator in our case) can be computed using an estimator of mutual information between the input and output of that function, since the conditional entropy term is 0 because the function is deterministic.

With x = G(z) the function of interest: DISPLAYFORM4 Hence, any neural mutual information maximization method such as MINE BID1 , noise constrastive estimation BID24 and DeepINFOMAX (Hjelm et al., 2018) can be applied to estimate and maximize the entropy of the generator.

All these estimators are based on training a discriminator which separates the joint distribution p(X, Z) from the product of the corresponding marginals p(X)p(Z).

As proposed by BID5 in the context of using a discriminator to minimize statistical dependencies between the outputs of an encoder, the samples from the marginals can be obtained by creating negative examples pairing an X and a Z from different samples of the joint, e.g., by independently shuffling each column of a matrix holding a minibatch with one row per example.

The training objective for the discriminator can be chosen in different ways.

In this paper, we used the Deep INFOMAX (DIM) estimator , which is based on maximizing the Jensen-Shannon divergence between the joint and the marginal (see Nowozin et al. for the original F-GAN formulation).

DISPLAYFORM5 where s+(a) = log(1+e a ) is the softplus function.

The discriminator T used to increase entropy at the output of the generator is trained by maximizing I JSD (X, Z) with respect to the parameters of T .

With X = G(Z) the output of the generator, IJSD (G(Z), Z) is one of the terms to be minimized the objective function for training G, with the effect of maximizing the generator's output entropy H(G(Z)).

The overall training objective for G is DISPLAYFORM6 where Z ∼ p z , the latent prior (typically a N (0, I) Gaussian).

Figure 1: EnGAN model overview where G ω is the Generator network, T φ is the Statistics network used for MI estimation and E θ is the energy network One option to generate samples is simply to use the usual GAN approach of sampling a z ∼ p z from the latent prior and then output x = G(z), i.e., obtain a sample x ∼ p G .

Since we have an energy function, another option is to run an MCMC in data space, and we have tried this with both Metropolis-Hastings (with a Gaussian proposal) and adjusted Langevin (detailed below, which does a gradient step down the energy and adds noise, then rejects high-energy samples).

However, we have interestingly obtained the best samples by considering E θ • G as an energy function in latent space and running an adjusted Langevin in that space (compare Fig. 4 with Fig. 7.1 ).

Then, in order to produce a data space sample, we apply G. For performing the MCMC sampling, we use the Metropolis-adjusted Langevin algorithm (MALA), with Langevin dynamics producing a proposal distribution in the latent space as follows: DISPLAYFORM0 Next, the proposedz t+1 is accepted or rejected using the Metropolis Hastings algorithm, by computing the acceptance ratio DISPLAYFORM1 and accepting (setting z t+1 =z t+1 ) with probability r.

The overall training procedure for EnGAN is detailed in Algorithm 1, with MALA referring to the above procedure for sampling by MCMC, with n mcmc steps.

When n mcmc =0, we recover the base case where z is only sampled from the prior and passed through G, and no MCMC is done to clean up the sample.

Require: Score penalty coefficient λ, number of energy function updates n ϕ per generator updates, number of MCMC steps n mcmc , number of training iterations T , Adam hyperparameters α, β 1 and β 2 .

Require: Energy function E θ with parameters θ, entropy statistics network T φ with parameters φ, generator network G ω with parameters ω, minibatch size m for t = 1, ..., T do for 1, ..., n ϕ do Sample minibatch of real data {x (1) , ...,

Sample minibatch of latent variables {z DISPLAYFORM0 Per-dimension shuffle of the minibatch z of latent variables, obtaining {z DISPLAYFORM1 The gradient-based updates can be performed with any gradient-based learning rule.

We used Adam in our experiments.

Generative models trained with maximum likelihood often suffer from the problem of spurious modes and excessive entropy of the trained distribution, where the model incorrectly assigns high probability mass to regions not present in the data manifold.

Typical energy-based models such as RBMs suffer from this problem partly because of the poor approximation of the negative phase gradient, as discussed above.

To check if EnGAN suffers from spurious modes, we train the energy-based model on synthetic 2D datasets (swissroll, 25gaussians and 8gaussians) similar to BID10 and visualize the energy function.

From the probaility density plots on Figure 1 , we can see that the energy model doesn't suffer from spurious modes and learns a sharp energy distribution.

Bottom: Corresponding probabiltiy density visualizations.

Density was estimated using a sample based approximation of the partition function.

GANs have been notoriously known to have issues with mode collapse, by which certain modes of the data distribution are not at all represented by the generative model.

Similar to the mode dropping issue that occurs in GANs, our generator is prone to mode dropping as well, since it is matched with the energy model's distribution using a reverse KL penalty D KL [P G || P E ].

Although the entropy maximization term attempts to fix this issue by maximizing the entropy of the generator's distribution, it is important to verify this effect experimentally.

For this purpose, we follow the same experimental setup as BID21 and BID25 .

We train our generative model on the StackedMNIST dataset, which is a synthetic dataset created by stacking MNIST on different channels.

The number of modes can be counted using a pretrained MNIST classifier, and the KL divergence can be calculated empirically between the mode count distribution produced by the generative model and true data (assumed to be uniform).

Table 1 : Number of captured modes and Kullblack-Leibler divergence between the training and samples distributions for ALI BID7 , Unrolled GAN BID21 , Vee-GAN BID25 , PacGAN BID20 , WGAN-GP BID10 .

Numbers except our model and WGAN-GP are borrowed from BID1 Table 1 , we can see that our model naturally covers all the modes in that data, without dropping a single mode.

Apart from just representing all the modes of the data distribution, our model also better matches the data distribution as evidenced by the very low KL divergence scores as compared to the baseline WGAN-GP.We noticed empirically that modeling 10 3 modes was quite trivial for benchmark methods such as WGAN-GP BID10 .

Hence, we also try evaluating our model on a new dataset with 10 4 modes (4 stacks).

The 4-StackedMNIST was created to have similar statistics to the original 3-StackedMNIST dataset.

We randomly sample and fix 128 × 10 4 images to train the generative model and take 26 × 10 4 samples for evaluations.

Generative models trained with maximum likelihood have often been found to produce more blurry samples.

Our energy model is trained with maximum likelihood to match the data distribution and the generator is trained to match the energy model's distribution with a reverse KL penalty.

To evaluate if our generator exhibits blurriness issues, we train our EnGAN model on the standard benchmark 32x32 CIFAR10 dataset for image modeling.

We additionally train our models on the 64x64 cropped CelebA -celebrity faces dataset to report qualitative samples from our model.

Similar to recent GAN works BID22 , we report both Inception Score (IS) and Frchet Inception Distance (FID) scores on the CIFAR10 dataset and compare it with a competitive WGAN-GP baseline.

From TAB1 , we can see that in addition to learning an energy function, EnGAN trains generative model producing samples comparable to recent adversarial methods such as WGAN-GP BID10 widely known for producing samples of high perceptual quality.

Additionally, we attach samples from the generator trained on the CelebA dataset and the 3-StackedMNIST dataset for qualitative inspection.

As shown below in Fig. 4 , the visual quality of the samples can be further improved by using the proposed MCMC sampler.

Figure 3: Left: 64x64 samples from the CelebA dataset Right: 28x28 samples from the 3-StackedMNIST dataset.

All samples are produced by the generator in a single step, without MCMC fine-tuning (see Fig. 4 for that).

Apart from the usefulness of energy estimates for relative density estimation (up to the normalization constant), energy functions can also be useful to perform unsupervised anomaly detection.

Unsupervised anomaly detection is a fundamental problem in machine learning, with critical applications in many areas, such as cybersecurity, complex system management, medical care, etc.

Density estimation is at the core of anomaly detection since anomalies are data points residing in low probability density areas.

We test the efficacy of our energy-based density model for anomaly detection using two popular benchmark datasets: KDDCUP and MNIST.KDDCUP We first test our generative model on the KDDCUP99 10 percent dataset from the UCI repository BID19 .Our baseline for this task is Deep Structured Energy-based Model for Anomaly Detection (DSEBM) BID30 , which trains deep energy models such as Convolutional and Recurrent EBMs using denoising score matching instead of maximum likelihood, for performing anomaly detection.

We also report scores on the state of the art DAGMM BID32 , which learns a Gaussian Mixture density model (GMM) over a low dimensional latent space produced by a deep autoencoder.

We train our model on the KDD99 data and use the score norm ||∇ x E θ (x)|| 2 2 as the decision function, similar to BID30 .

BID32 .

Values for our model are derived from 5 runs.

For each individual run, the metrics are averaged over the last 10 epochs.

+0.1990 F1 score) and is comparable to the current SOTA model (DAGMM) specifically designed for anomaly detection.

MNIST Next we evaluate our generative model on anomaly detection of high dimensional image data.

We follow the same experiment setup as BID29 and make each digit class an anomaly and treat the remaining 9 digits as normal examples.

We also use the area under the precision-recall curve (AUPRC) as the metric to compare models.

Table 4 : Performance on the unsupervised anomaly detection task on MNIST measured by area under precision recall curve.

Numbers except ours are obtained from BID29 .

Results for our model are averaged over last 10 epochs to account for the variance in scores.

Table 4 , it can be seen that our energy model outperforms VAEs for outlier detection and is comparable to the SOTA BiGAN-based anomaly detection methods for this dataset BID29 which train bidirectional GANs to learn both an encoder and decoder (generator) simultaneously.

An advantage with our method is that it has theoretical justification for the usage of energy function as a decision function, whereas the BiGAN-σ model lacks justification for using a combination of the reconstruction error in output space as well as the discriminator's cross entropy loss for the decision function.

To show that the Metropolis Adjusted Langevin Algorithm (MALA) performed in latent space produced good samples in observed space, we attach samples from the beginning (with z sampled from a Gaussian) and end of the chain for visual inspection.

From the attached samples, it can be seen that the MCMC sampler appears to perform a smooth walk on the image manifold, with the initial and final images only differing in a few latent attributes such as hairstyle, background color, face orientation, etc.

Figure 4: Left:

Samples at the beginning of the chain (i.e. simply from the ordinary generator, z ∼ N (0, I)).

Right: Generated samples after 100 iterations of MCMC using the MALA sampler.

We see how the chain is smoothly walking on the image manifold and changing semantically meaningful and coherent aspects of the images.

We proposed EnGAN, an energy-based generative model that produces energy estimates using an energy model and a generator that produces fast approximate samples.

This takes advantage of novel methods to maximize the entropy at the output of the generator using a GAN-like technique.

We have shown that our energy model learns good energy estimates using visualizations in toy 2D data and through performance in unsupervised anomaly detection.

We have also shown that our generator produces samples of high perceptual quality by measuring Inception and Frchet scores and shown that EnGAN is robust to the respective weaknesses of GAN models (mode dropping) and maximumlikelihood energy-based models (spurious modes).

We found that running an MCMC in latent space rather than in data space (by composing the generator and the data-space energy to obtain a latentspace energy) works substantially better than running the MCMC in data-space.

7.1 MCMC IN DATA SPACE Figure 5 : Samples from the beginning, middle and end of the chain performing MCMC sampling in visible space.

Initial sample is from the generator (p G ) but degrades as we follow MALA directly in data space.

Compare with samples obtained by running the chain in latent space and doing the MH rejection according to the data space energy (Fig. 4) .

It can be seen that MCMC in data space has poor mixing and gets attracted to spurious modes.

For all experiments we use Adam as the optimizer with α = 0.0001, β 1 = 0.5, β 2 = 0.9.

We used n mcmc = 0 (no MCMC steps during training) for all scores reported in the paper.

Toy Data: The generator, energy-model and the statistics network are simple 3-hidden layer MLPs with dimensionality 512.

The input to the statistics network is a conatenation of the inputs x and latents z. For these experiments, we use the energy norm co-efficient λ = 0.1StackedMNIST: : In line with previous work, we adopt the same architectural choices for the generator and energy-model / discriminator as VeeGAN BID25 .

The statistics network is modeled similar to the energy-model, except with the final MLP which now takes as input both the latents z and reduced feature representation of x produced by the CNN.

For the CIFAR10 experiments, we adopt the same 'Standard CNN' architecture as in SpectralNorm BID22 .

We adapt the architecture for the Statistics Network similar to the StackedMNIST experiments as mentioned above.

For these experiments, we use the energy norm co-efficient λ = 10Anomaly Detection: For the KDD99 dataset, we adopt the same architecture as BID29 .

We noticed that using n ψ = 1 and λ = 10 5 worked best for these experiments.

A large energy norm coefficient was specifically necessary since the energy model overfit to some artifacts in the data and exploded in value.

For the MNIST anomaly detection experiments, we use the same architecture as the StackedMNIST experiments.

@highlight

We introduced entropy maximization to GANs, leading to a reinterpretation of the critic as an energy function.
We propose a rejection sampling scheme using the discriminator of a GAN to approximately correct errors in the GAN generator distribution.

We show that under quite strict assumptions, this will allow us to recover the data distribution exactly.

We then examine where those strict assumptions break down and design a practical algorithm—called Discriminator Rejection Sampling (DRS)—that can be used on real data-sets.

Finally, we demonstrate the efficacy of DRS on a mixture of Gaussians and on the state of the art SAGAN model.

On ImageNet, we train an improved baseline that increases the best published Inception Score from 52.52 to 62.36 and reduces the Frechet Inception Distance from 18.65 to 14.79.

We then use DRS to further improve on this baseline, improving the Inception Score to 76.08 and the FID to 13.75.

Generative Adversarial Networks (GANs) BID5 are a powerful tool for image synthesis.

They have also been applied successfully to semi-supervised and unsupervised learning BID25 BID20 BID11 , image editing BID31 BID12 , and image style transfer BID2 .

Informally, the GAN training procedure pits two neural networks against each other, a generator and a discriminator.

The discriminator is trained to distinguish between samples from the target distribution and samples from the generator.

The generator is trained to fool the discriminator into thinking its outputs are real.

The GAN training procedure is thus a two-player differentiable game, and the game dynamics are largely what distinguishes the study of GANs from the study of other generative models.

These game dynamics have well-known and heavily studied stability issues.

Addressing these issues is an active area of research BID17 BID7 .However, we are interested in studying something different: Instead of trying to improve the training procedure, we (temporarily) accept its flaws and attempt to improve the quality of trained generators by post-processing their samples using information from the trained discriminator.

It's well known that (under certain very strict assumptions) the equilibrium of this training procedure is reached when sampling from the generator is identical to sampling from the target distribution and the discriminator always outputs 1/2.

However, these assumptions don't hold in practice.

In particular, GANs as presently trained don't learn to reproduce the target distribution BID1 .

Moreover, trained GAN discriminators aren't just identically 1/2 -they can even be used to perform chess-type skill ratings of other trained generators .We ask if the information retained in the weights of the discriminator at the end of the training procedure can be used to "improve" the generator.

At face value, this might seem unlikely.

After all, if there is useful information left in the discriminator, why doesn't it find its way into the generator via the training procedure?

Further reflection reveals that there are many possible reasons.

First, the assumptions made in various analyses of the training procedure surely don't hold in practice (e.g. the discriminator and generator have finite capacity and are optimized in parameter space rather than density-space).

Second, due to the concrete realization of the discriminator and the generator as neural networks, it may be that it is harder for the generator to model a given distribution than it is for the discriminator to tell that this distribution is not being modeled precisely.

Finally, we may simply not train GANs long enough in practice for computational reasons.

In this paper, we focus on using the discriminator as part of a probabilistic rejection sampling scheme.

In particular, this paper makes the following contributions:• We propose a rejection sampling scheme using the GAN discriminator to approximately correct errors in the GAN generator distribution.• We show that under quite strict assumptions, this scheme allows us to recover the data distribution exactly.• We then examine where those strict assumptions break down and design a practical algorithm -called DRS -that takes this into account.• We conduct experiments demonstrating the effectiveness of DRS.

First, as a baseline, we train an improved version of the Self-Attention GAN, improving its performance from the best published Inception Score of 52.52 up to 62.36, and from a Fréchet Inception Distance of 18.65 down to 14.79.

We then show that DRS yields further improvement over this baseline, increasing the Inception Score to 76.08 and decreasing the Fréchet Inception Distance to 13.75.

A generative adversarial network (GAN) BID5 consists of two separate neural networks -a generator, and a discriminator -trained in tandem.

The generator G takes as input a sample from the prior z ∈ Z ∼ p z and produces a sample G(z) ∈ X. The discriminator takes an observation x ∈ X as input and produces a probability D(x) that the observation is real.

The observation is sampled either according to the density p d (the data generating distribution) or p g (the implicit density given by the generator and the prior).

Using the standard non-saturating variant, the discriminator and generator are then trained using the following loss functions: DISPLAYFORM0

The two most popular techniques for evaluating GANs on image synthesis tasks are the Inception Score and the Fréchet Inception Distance.

The Inception Score BID24 ) is given by exp(E x KL(p(y|x)||p(y))), where p(y|x) is the output of a pre-trained Inception classifier BID27 .

This measures the ability of the GAN to generate samples that the pre-trained classifier confidently assigns to a particular class, and also the ability of the GAN to generate samples from all classes.

The Fréchet Inception Distance (FID) BID9 , is computed by passing samples through an Inception network to yield "semantic embeddings", after which the Fréchet distance is computed between Gaussians with moments given by these embeddings.

We use a Self-Attention GAN (SAGAN) BID32 in our experiments.

We do so because SAGAN is considered state of the art on the ImageNet conditional-image-synthesis task (in which images are synthesized conditioned on class identity).

SAGAN differs from a vanilla GAN in the following ways: First, it uses large residual networks BID8 instead of normal convolutional layers.

Second, it uses spectral normalization in the generator and the discriminator and a much lower learning rate for the generator than is conventional BID9 .

Third, SAGAN makes use of self-attention layers (Wang et al.) , in order to better model DISPLAYFORM0 Append(X, samples); end end Figure 1 : Left:

For a uniform proposal distribution and Gaussian target distribution, the blue points are the result of rejection sampling and the red points are the result of naively throwing out samples for which the density ratio ( DISPLAYFORM1 ) is below a threshold.

The naive method underrepresents the density of the tails.

Right: the DRS algorithm.

KeepTraining continues training using early stopping on the validation set.

BurnIn computes a large number of density ratios to estimate their maximum.

DISPLAYFORM2 .M is an empirical estimate of the true maximum M .long range dependencies in natural images.

Finally, this whole model is trained using a special hinge version of the adversarial loss BID14 BID28 : DISPLAYFORM3

Rejection sampling is a method for sampling from a target distribution p d (x) which may be hard to sample from directly.

Samples are instead drawn from a proposal distribution p g (x), which is easier to sample from, and which is chosen such that there exists a finite value DISPLAYFORM0 .

A given sample y drawn from p g is kept with acceptance probability DISPLAYFORM1 , and rejected otherwise.

See the blue points in Figure 1 (Left) for a visualization.

Ideally, p g (x) should be close to p d (x), otherwise many samples will be rejected, reducing the efficiency of the algorithm BID16 .In Section 3, we explain how to apply this rejection sampling algorithm to the GAN framework: in brief, we draw samples from the trained generator, p g (x), and then reject some of those samples using the discriminator to attain a closer approximation to the true data distribution, p d (x).

An independent rejection sampling approach was proposed by BID6 in the latent space of variational autoencoders for improving samples from the variational posterior.

In this section we introduce our proposed rejection sampling scheme for GANs (which we call Discriminator Rejection Sampling, or DRS).

We'll first derive an idealized version of the algorithm that will rely on assumptions that don't necessarily hold in realistic settings.

We'll then discuss the various ways in which these assumptions might break down.

Finally, we'll describe the modifications we made to the idealized version in order to overcome these challenges.

Suppose that we have a GAN and our generator has been trained to the point that p g and p d have the same support.

That is, for all x ∈ X, p g (x) = 0 if and only if p d (x) = 0.

If desired, we can make p d and p g have support everywhere in X if we add low-variance Gaussian noise to the observations.

Now further suppose that we have some way to compute DISPLAYFORM0 for all x, so we can perform rejection sampling with p g as the proposal distribution and p d as the target distribution as long as we can evaluate the quantity DISPLAYFORM1 1 .

In this case, we can exactly sample from p d BID3 , though we may have to reject many samples to do so.

But how can we evaluate p d (x)/M p g (x)?

p g is defined only implicitly.

One thing we can do is to borrow an analysis from the original GAN paper BID5 , which assumes that we can optimize the discriminator in the space of density functions rather than via changing its parameters.

If we make this assumption, as well as the assumption that the discriminator is defined by a sigmoid applied to some function of x and trained with a cross-entropy loss, then by Proposition 1 of that paper, we have that, for any fixed generator and in particular for the generator G that we have when we stop training, training the discriminator to completely minimize its own loss yields DISPLAYFORM2 We will discuss the validity of these assumptions later, but for now consider that this allows us to solve for p d (x)/p g (x) as follows: As noted above, we can assume the discriminator is defined as: DISPLAYFORM3 where D(x) is the final discriminator output after the sigmoid, and D(x) is the logit.

Thus, DISPLAYFORM4 Now suppose one last thing, which is that we can tractably compute DISPLAYFORM5 3.2 DISCRIMINATOR REJECTION SAMPLING: THE PRACTICAL SCHEMEAs we hinted at, the above analysis has a number of practical issues.

In particular:1.

Since we can't actually perform optimization over density functions, we can't actually compute D * .

Thus, our acceptance probability won't necessarily be proportional to DISPLAYFORM6 2.

At least on large datasets, it's quite obvious that the supports of p g and p d are not the same.

If the support of p g and p d has a low volume intersection, we may not even want to compute DISPLAYFORM7 would just evaluate to 0 most places.3.

The analysis yielding the formula for D * also assumes that we can draw infinite samples from p d , which is not true in practice.

If we actually optimized D all the way given a finite data-set, it would give nonzero results on a set of measure 0.1 Why go through all this trouble when we could instead just pick some threshold T and throw out x when D * (x) < T ?

This doesn't allow us to recover p d in general.

If, for example, there is x s.t.

pg(x ) > p d (x ) > 0, we still want some probability of observing x .

See the red points in Figure 1 (Left) for a visual explanation.4.

In general it won't be tractable to compute M .

5.

Rejection sampling is known to have too low an acceptance probability when the target distribution is high dimensional BID16 .This section describes the Discriminator Rejection Sampling (DRS) procedure, which is an adjustment of the idealized procedure, meant to address the above issues.

On the difficulty of actually computing D * : Given that items 2 and 3 suggest we may not want to compute D * exactly, we should perhaps not be too concerned with item 1, which suggests that we can't.

The best argument we can make that it is OK to approximate D * is that doing so seems to be successful empirically.

We speculate that training a regularized D with SGD gives a final result that is further from D * but perhaps is less over-fit to the finite sample from p d used for training.

We also hypothesize that the D we end up with will distinguish between "good" and "bad" samples, even if those samples would both have zero density under the true p d .

We qualitatively evaluate this hypothesis in FIG1 .

We suspect that more could be done theoretically to quantify the effect of this approximation, but we leave this to future work.

On the difficulty of actually computing M : It's nontrivial to compute M , at the very least because we can't compute D * .

In practice, we get around this issue by estimating M from samples.

We first run an estimation phase, in which 10,000 samples are used to estimate D * M .

We then use this estimate in the sampling phase.

Throughout the sampling phase we update our estimate of D * M if a larger value is found.

It's true that this will result in slight overestimates of the acceptance probability for samples that were processed before a new maximum was found, but we choose not to worry about this too much, since we don't find that we have to increase the maximum very often in the sampling phase, and the increase is very small when it does happen.

Dealing with acceptance probabilities that are too low: Item 5 suggests that we may end up with acceptance probabilities that are too low to be useful when performing this technique on realistic data-sets.

If D * M is very large, the acceptance probability e D * (x)− D * M will be close to zero, and almost all samples will be rejected, which is undesirable.

One simple way to avoid this problem is to compute some F (x) such that the acceptance probability can be written as follows: DISPLAYFORM8 If we solve for F (x) in the above equation we can then perform the following rearrangement: DISPLAYFORM9 In practice, we instead computê DISPLAYFORM10 where is a small constant added for numerical stability and γ is a hyperparameter modulating overall acceptance probability.

For very positive γ, all samples will be rejected.

For very negative γ, all samples will be accepted.

See Figure 2 for an analysis of the effect of adding γ.

A summary of our proposed algorithm is presented in Figure 1 (Right).

In this section we justify the modifications made to the idealized algorithm.

We do this by conducting two experiments in which we show that (according to popular measures of how well a GAN has Figure 2 : (A) Histogram of the sigmoid inputs,F (x) (left plot), and acceptance probabilities, σ(F (x)) (center plot), on 20K fake samples before (purple) and after (green) adding the constant γ to all F (x).

Before adding gamma, 98.9% of the samples had an acceptance probability < 1e-4.

(B) Histogram of max j p(y j |x i ) from a pre-trained Inception network where p(y j |x i ) is the predicted probability of sample x i belonging to the y j category (from 1, 000 ImageNet categories).

The green bars correspond to 25, 000 accepted samples and the red bars correspond to 25, 000 rejected samples.

The rejected images are less recognizable as belonging to a distinct class.

learned the target distribution) Discriminator Rejection Sampling yields improvements for actual GANs.

We start with a toy example that yields insight into how DRS can help, after which we demonstrate DRS on the ImageNet dataset BID23 .

We investigate the impact of DRS on a low-dimensional synthetic data set consisting of a mixture of twenty-five 2D isotropic Gaussian distributions (each with standard deviation of 0.05) arranged in a grid BID4 BID26 BID15 .

We train a GAN model where the generator and discriminator are neural networks with four fully connected layers with ReLu activations.

The prior is a 2D Gaussian with mean of 0 and standard deviation of 1 and the GAN is trained using the standard loss function.

We generate 10,000 samples from the generator with and without DRS.

The target distribution and both sets of generated samples are depicted in FIG0 .

Here, we have set γ dynamically for each batch, to the 95 th percentile ofF (x) for all x in the batch.

To measure performance, we assign each generated sample to its closest mixture component.

As in BID26 , we define a sample as "high quality" if it is within four standard deviations of its assigned mixture component.

As shown in Table 1 , DRS increases the fraction of high-quality samples from 70% to 90%.

As in BID4 and BID26 we call a mode "recovered" if at least one high-quality sample was assigned to it.

Table 1 shows that DRS does not reduce the number of recovered modes -that is, it does not trade off quality for mode coverage.

It does reduce the standard deviation of the high-quality samples slightly, but this is a good thing in this case (since the standard deviation of the target Gaussian distribution is 0.05).

It also confirms that DRS does not accept samples only near the center of each Gaussian but near the tails as well.

Table 1 : Results with and without DRS on 10,000 generated samples from a model of a 2D grid of Gaussian components.

# of recovered modes % "high quality" std of "high quality" samples Without DRS 24.8 ± 0.4 70 ± 9 0.11 ± 0.01 With DRS 24.8 ± 0.4 90 ± 2 0.10 ± 0.01 61.44 ± 0.09 17.14 ± 0.09 76.08 ± 0.30 13.57 ± 0.13

Since it is presently the state-of-the-art model on the conditional ImageNet synthesis task, we have reimplemented the Self-Attention GAN BID32 as a baseline.

After reproducing the results reported by BID32 (with the learning rate of 1e −4 ), we fine-tuned a trained SAGAN with a much lower learning rate (1e −7 ) for both generator and discriminator.

This improved both the Inception Score and FID significantly as can be seen in the Improved-SAGAN column in TAB0 .

Plots of Inception score and FID during training are given in FIG2 (A).Since SAGAN uses a hinge loss and DRS requires a sigmoid output, we added a fully-connected layer "on top of" the trained discriminator and trained it to distinguish real images from fake ones using the binary cross-entropy loss.

We trained this extra layer with 10,000 generated samples from the model and 10,000 examples from ImageNet.

We then generated 50,000 samples from normal SAGAN and Improved SAGAN with and without DRS, repeating the sampling process 4 times.

We set γ dynamically to the 80 th percentile of the F (x) values in each batch.

The averages of Inception Score and FID over these four trials are presented in TAB0 .

Both scores were substantially improved for both models, indicating that DRS can indeed be useful in realistic settings involving large data-sets and sophisticated GAN variants.

Qualitative Analysis of ImageNet results: From a pool of 50,000 samples, we visualize the "best" and the "worst" 100 samples based on their acceptance probabilities.

FIG1 shows that the subjective visual quality of samples with high acceptance probability is considerably better.

Figure 2 (B) also shows that the accepted images are on average more recognizable as belonging to a distinct class.

We also study the behavior of the discriminator in another way.

We choose an ImageNet category randomly, then generate samples from that category until we have found two images G(z 1 ), G(z 2 ) such that G(z 1 ) appears visually realistic and G(z 2 ) appears visually unrealistic.

Here, z 1 and z 2 are the input latent vectors.

We then generate many images by interpolating in latent space between the two images according to z = αz 1 + (1 − α)z 2 with α ∈ {0, 0.1, 0.2, . . .

, 1}. In FIG2 , the first and last columns correspond with α = 1 and α = 0, respectively.

The color bar in the figure represents the acceptance probability assigned to each sample.

In general, acceptance probabilities decrease from left to right.

There is no reason to expect a priori that the acceptance probability should decrease monotonically as a function of the interpolated z, so it says something interesting about the discriminator that most rows basically follow this pattern.

We have proposed a rejection sampling scheme using the GAN discriminator to approximately correct errors in the GAN generator distribution.

We've shown that under strict assumptions, we can recover the data distribution exactly.

We've also examined where those assumptions break down and Each row shows images synthesized by interpolating in latent space.

The color bar above each row represents the acceptance probabilities for each sample: red for high and white for low.

Subjective visual quality of samples with high acceptance probability is considerably better: objects are more coherent and more recognizable as belonging to a specific class.

There are fewer indistinct textures, and fewer scenes without recognizable objects.

• There's no reason that our scheme can only be applied to GAN generators.

It seems worth investigating whether rejection sampling can improve e.g. VAE decoders.

This seems like it might help, because VAEs may have trouble with "spreading mass around" too much.• In one ideal case, the critic used for rejection sampling would be a human.

Can we use better proxies for the human visual system to improve rejection sampling's effect on image synthesis models?• It would be interesting to theoretically characterize the efficacy of rejection sampling under the breakdown-of-assumptions that we have described earlier.

In addition, we represent Inception score as a function of acceptance rate in FIG5 -left.

Different acceptance rates are achieved by changing γ from the 0 th percentile of F (x) (acceptance rate = 100%) to its 90 th percentile (acceptance rate = 14%).

Decreasing the acceptance rate filters more non-realistic samples and increases the final Inception score.

After an specific rate, rejecting more samples does not gain any benefit in collecting a better pool of samples.

Moreover, FIG5 -right shows the correlation between the acceptance probabilities that DRS assigns to the synthesized samples and the recognizability of those samples from the view-point of a pre-trained Inception network.

The latter is measured by computing max j p(y j |x i ) which is the probability of sample x i belonging to the category y j from the 1,000 ImageNet classes.

As expected, there is a large mass of the recognizable images accepted with high acceptance probabilities on the top right corner.

The small mass of images which cannot be easily classified into one of the 1,000 categories while having high acceptance probability scores (the top left corner of the graph) can be due to the non-optimal GAN discriminator in practice.

Therefore, we expect that improving the discriminator performance boosts the final inception score even more substantially.

, and the acceptance probability assigned to each sample x i by DRS versus the maximum probability of belonging to one of the 1K categories based on a pre-trained Inception network, max j p(y j |x i ) (right).

To confirm that our Discriminator Rejection Sampling is not duplicating the training samples, we show the nearest neighbor of a few visually-realistic generated samples in the ImageNet training data in FIG2 .

The nearest neighbors are found based on their fc7 features from the pre-trained VGG16 model.

@highlight

We use a GAN discriminator to perform an approximate rejection sampling scheme on the output of the GAN generator.

@highlight

 Proposes a rejection sampling algorithm for sampling from the GAN generator.

@highlight

This paper proposed a post-processing rejection sampling scheme for GANs, named Discriminator Rejection Sampling, to help filter ‘good’ samples from GANs’ generator.
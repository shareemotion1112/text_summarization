In this paper, we describe the "implicit autoencoder" (IAE), a generative autoencoder in which both the generative path and the recognition path are parametrized by implicit distributions.

We use two generative adversarial networks to define the reconstruction and the regularization cost functions of the implicit autoencoder, and derive the learning rules based on maximum-likelihood learning.

Using implicit distributions allows us to learn more expressive posterior and conditional likelihood distributions for the autoencoder.

Learning an expressive conditional likelihood distribution enables the latent code to only capture the abstract and high-level information of the data, while the remaining information is captured by the implicit conditional likelihood distribution.

For example, we show that implicit autoencoders can disentangle the global and local information, and perform deterministic or stochastic reconstructions of the images.

We further show that implicit autoencoders can disentangle discrete underlying factors of variation from the continuous factors in an unsupervised fashion, and perform clustering and semi-supervised learning.

Deep generative models have achieved remarkable success in recent years.

One of the most successful models is the generative adversarial network (GAN) BID7 , which employs a two player min-max game.

The generative model, G, samples the noise vector z ∼ p(z) and generates the sample G(z).

The discriminator, D(x), is trained to identify whether a point x comes from the data distribution or the model distribution; and the generator is trained to maximally confuse the discriminator.

The cost function of GAN is DISPLAYFORM0 GANs can be viewed as a general framework for learning implicit distributions BID18 BID12 .

Implicit distributions are probability distributions that are obtained by passing a noise vector through a deterministic function that is parametrized by a neural network.

In the probabilistic machine learning problems, implicit distributions trained with the GAN framework can learn distributions that are more expressive than the tractable distributions trained with the maximum-likelihood framework.

Variational autoencoders (VAE) BID13 BID20 are another successful generative models that use neural networks to parametrize the posterior and the conditional likelihood distributions.

Both networks are jointly trained to maximize a variational lower bound on the data log-likelihood.

One of the limitations of VAEs is that they learn factorized distributions for both the posterior and the conditional likelihood distributions.

In this paper, we propose the "implicit autoencoder" (IAE) that uses implicit distributions for learning more expressive posterior and conditional likelihood distributions.

Learning a more expressive posterior will result in a tighter variational bound; and learning a more expressive conditional likelihood distribution will result in a global vs. local decomposition of information between the prior and the conditional likelihood.

This enables the latent code to only capture the information that we care about such as the high-level and abstract information, while the remaining low-level information of data is separately captured by the noise vector of the implicit decoder.

Implicit distributions have been previously used in learning generative models in works such as adversarial autoencoders (AAE) BID16 , adversarial variational Bayes (AVB) (Mescheder et al., 2017) , ALI (Dumoulin et al., 2016) , BiGAN BID5 and other works such as BID12 BID22 .

The global vs. local decomposition of information has also been studied in previous works such as PixelCNN autoencoders (van den Oord et al., 2016) , PixelVAE BID9 , variational lossy autoencoders BID4 , PixelGAN autoencoders BID15 , or other works such as BID2 BID8 BID0 .

In the next section, we first propose the IAE and then establish its connections with the related works.

Let x be a datapoint that comes from the data distribution p data (x).

The encoder of the implicit autoencoder ( FIG0 ) defines an implicit variational posterior distribution q(z|x) with the function z = f φ (x, ) that takes the input x along with the input noise vector and outputsẑ.

The decoder of the implicit autoencoder defines an implicit conditional likelihood distribution p(x|z) with the functionx = g θ (ẑ, n) that takes the codeẑ along with the latent noise vector n and outputs a reconstruction of the imagex.

In this paper, we refer toẑ as the latent code or the global code, and refer to the latent noise vector n as the local code.

Let p(z) be a fixed prior distribution, p(x, z) = p(z)p(x|z) be the joint model distribution, and p(x) be the model distribution.

The variational distribution q(z|x) induces the joint data distribution q(x, z), the aggregated posterior distribution q(z), and the inverse posterior/encoder distribution q(x|z) as follows: DISPLAYFORM0 Maximum likelihood learning is equivalent to matching the model distribution p(x) to the data distribution p data (x); and learning with variational inference is equivalent to matching the joint model distribution p(x, z) to the joint data distribution q(x, z).

The entropy of the data distribution H data (x), the entropy of the latent code H(z), the mutual information I(x; z), and the conditional entropies H(x|z) and H(z|x) are all defined under the joint data distribution q(x, z) and its marginals p data (x) and q(z).

Using the aggregated posterior distribution q(z), we can define the joint reconstruction distribution r(x, z) and the aggregated reconstruction distribution r(x) as follows: DISPLAYFORM1 Note that in general we have r(x, z) = q(x, z) = p(x, z), q(z) = p(z), and r(x) = p data (x) = p(x).We now use different forms of the aggregated evidence lower bound (ELBO) to describe the IAE and establish its connections with VAEs and AAEs.

DISPLAYFORM2 Mutual Info.(5) DISPLAYFORM3 Entropy of Data DISPLAYFORM4 Entropy of Data (7) See Appendix A for the proof.

The standard formulation of the VAE (Equation 4) only enables us to learn factorized Gaussian posterior and conditional likelihood distributions.

The AAE BID16 ( Equation 5 ) and the AVB BID17 enable us to learn implicit posterior distributions, but their conditional likelihood distribution is still a factorized Gaussian distribution.

However, the IAE enables us to learn implicit distributions for both the posterior and the conditional likelihood distributions.

Similar to VAEs and AAEs, the IAE (Equation 6) has a reconstruction cost function and a regularization cost function, but trains each of them with a GAN.

The IAE reconstruction cost is E z∼q(z) KL(q(x|z) p(x|z)) .

The standard VAE uses a factorized decoder, which has a very limited stochasticity.

Thus, the standard VAE performs almost deterministic reconstructions by learning to invert the deterministic mapping of the encoder.

The IAE, however, uses a powerful implicit decoder to perform stochastic reconstructions, by learning to match the expressive decoder distribution p(x|z) to the inverse encoder distribution q(x|z).

We note that there are other variants of VAEs that can also learn expressive decoder distributions by using autoregressive neural networks.

We will discuss these models later in this section.

Equation 8 contrasts the reconstruction cost of standard autoencoders that is used in VAEs/AAEs, with the reconstruction cost of IAEs.

DISPLAYFORM5 We can see from Equation 8 that similar to IAEs, the reconstruction cost of the autoencoder encourages matching the decoder distribution to the inverse encoder distribution.

But in autoencoders, the cost function also encourages minimizing the conditional entropy H(x|z), or maximizing the mutual information I(x, z).

Maximizing the mutual information in autoencoders enforces the latent code to capture both the global and local information.

In contrast, in IAEs, the reconstruction cost does not penalize the encoder for losing the local information, as long as the decoder can invert the encoder distribution.

In order to minimize the reconstruction cost function of the IAE, we re-write it in the form of a distribution matching cost function between the joint data distribution and the joint reconstruction distribution KL(q(x, z) r(x, z)) (Equation 7 ).

This KL divergence is approximately minimized with the reconstruction GAN.

The IAE has also a regularization cost function KL(q(z) p(z)) that matches the aggregated posterior distribution with a fixed prior distribution.

This is the same regularization cost function used in AAEs (Equation 5), and is approximately minimized with the regularization GAN.

Note that the last term in FIG4 is the entropy of the data distribution that is fixed.

Training Process.

We now describe the training process.

We pass a given point x ∼ p data (x) through the encoder and the decoder to obtainẑ ∼ q(z) andx ∼ r(x).

We now train the discriminator of the reconstruction GAN to identify the positive example (x,ẑ) from the negative example (x,ẑ).

Suppose this discriminator function at its optimality is D * (x, z).

We try to confuse this discriminator by backpropagating through the negative example (x,ẑ) 1 and updating the encoder and decoder weights.

More specifically, the generative loss of the reconstruction GAN is T * (x, z) = − log D * (x, z), which defines the reconstruction cost of the IAE.

We use the re-parametrization trick to update the encoder and decoder weights by computing the unbiased Monte Carlo estimate of the gradient of the reconstruction cost T * (x, z) with respect to (φ, θ) as follows: DISPLAYFORM6 We call this process the adversarial reconstruction.

Similarly, we train the discriminator of the regularization GAN to identify the positive example z ∼ p(z) from the negative exampleẑ ∼ q(z).

This discriminator now defines the regularization cost function, which can provide us with a gradient to update only the encoder weights.

We call this process the adversarial regularization.

Optimizing the adversarial regularization and reconstruction cost functions encourages p(x|z) = q(x|z) and p(z) = q(z), which results in the model distribution capturing the data distribution p(x) = p data (x).We note that in this work, we use the original formulation of GANs BID7 to match the distributions.

As a result, the gradient that we obtain from the adversarial training, only approximately follows the gradient of the variational bound on the data log-likelihood.

However, as shown in BID19 , the objective of the GAN can be modified to optimize any f -divergence including the KL divergence.

Bits-Back Interpretation of the IAE Objective.

In Appendix B, we describe an information theoretic interpretation of the ELBO of IAEs (Equation 7) using the Bits-Back coding argument BID10 BID4 BID8 .

In IAEs, the dimension of the latent vector along with its prior distribution defines the capacity of the latent code, and the dimension of the latent noise vector along with its distribution defines the capacity of the implicit decoder.

By adjusting these dimensions and distributions, we can have a full control over the decomposition of information between the latent code and the implicit decoder.

In one extreme case, by removing the noise vector, we can have a fully deterministic autoencoder that captures all the information by its latent code.

In the other extreme case, we can remove the global latent code and have an unconditional implicit distribution that can capture the whole data distribution by itself.

The global vs. local decomposition of information in IAEs is further discussed in Appendix C from an information theoretic perspective.

In IAEs, we can choose to only optimize the reconstruction cost or both the reconstruction and the regularization costs.

In the following, we discuss four special cases of the IAE and establish connections with the related methods.

In this case, we remove the noise vectors from the IAE, which makes both q(z|x) and p(x|z)deterministic.

We then only optimize the reconstruction cost E z∼q(z) KL(q(x|z) p(x|z)) .

As a result, similar to the standard autoencoder, the deterministic decoder p(x|z) learns to match to the inverse deterministic encoder q(x|z), and thus the IAE learns to perform exact and deterministic reconstruction of the original image, while the latent code is learned in an unconstrained fashion.

In other words, in standard autoencoders, the Euclidean cost explicitly encouragesx to reconstruct x, and in case of uncertainty, performs mode averaging by blurring the reconstructions; however, in IAEs, the adversarial reconstruction implicitly encouragesx to reconstruct x, and in case of uncertainty, captures this uncertainty by the local noise vector (Case 3), which results in sharp reconstructions.

In the previous case, the latent code was learned in an unconstrained fashion.

We now keep the decoder deterministic and add the regularization term which matches the aggregated posterior distribution to a fixed prior distribution.

In this case, the IAE reduces to the AAE with the difference that the IAE performs adversarial reconstruction rather than Euclidean reconstruction.

This case of the IAE defines a valid generative model where the latent code captures all the information of the data distribution.

In order to sample from this model, we first sample from the imposed prior p(z) and then pass this sample through the deterministic decoder.

In this case of the IAE, we only optimize KL(q(x, z) r(x, z)), while p(x|z) is a stochastic implicit distribution.

Matching the joint distribution q(x, z) to r(x, z) ensures that their marginal distributions would also match; that is, the aggregated reconstruction distribution r(x) matches the data distribution p data (x).

This model by itself defines a valid generative model in which both the prior, which in this case is q(z), and the conditional likelihood p(x|z) are learned at the same time.

In order to sample from this generative model, we initially sample from q(z) by first sampling a point x ∼ p data (x) and then passing it through the encoder to obtain the latent codeẑ ∼ q(z).

Then we sample from the implicit decoder distribution conditioned onẑ to obtain the stochastic reconstructionx ∼ r(x).

If the decoder is deterministic (Case 1), the reconstructionx would be the same as the original image x. But if the decoder is stochastic, the global latent code only captures the abstract and high-level information of the image, and the stochastic reconstructionx only shares this high-level information with the original x.

This case of the IAE is related to the PixelCNN autoencoder BID23 , where the decoder is parametrized by an autoregressive neural network which can learn expressive distributions, while the latent code is learned in an unconstrained fashion.

In the previous case, we showed that even without the regularization term, r(x) will capture the data distribution.

But the main drawback of the previous case is that its prior q(z) is not a parametric distribution that can be easily sampled from.

One way to fix this problem is to fit a parametric prior p(z) to q(z) once the training is complete, and then use p(z) to sample from the model.

However, a better solution would be to consider a fixed and pre-defined prior p(z), and impose it on q(z) during the training process.

Indeed, this is the regularization term that the ELBO suggests in Equation 7 .

By adding the adversarial regularization cost function to match q(z) to p(z), we ensure that r(x) = p data (x) = p(x).

Now sampling from this model only requires first sampling from the pre-defined prior z ∼ p(z), and then sampling from the conditional implicit distribution to obtain x ∼ r(x).

In this case, the information of data distribution is captured by both the fixed prior and the learned conditional likelihood distribution.

Similar to the previous case, the latent code captures the high-level and abstract information, while the remaining local and low-level information is captured by the implicit decoder.

We will empirically show this decomposition of information on different datasets in Section 2.1.1 and Section 2.1.2.

This decomposition of information has also been studied in other works such as PixelVAE BID9 , variational lossy autoencoders BID4 , PixelGAN autoencoders BID15 and variational Seq2Seq autoencoders BID2 .

However, the main drawback of these methods is that they all use autoregressive decoders which are not parallelizable, and are much more computationally expensive to scale up than the implicit decoders.

Another advantage of implicit decoders to autoregressive decoders is that in implicit decoders, the local statistics is captured by the local code representation; but in autoregressive decoders, we do not learn a vector representation for the local statistics.

Connections with ALI and BiGAN.

In ALI BID6 and BiGAN BID5 models, there are two separate networks that define the joint data distribution q(x, z) and the joint model distribution p(x, z).

The parameters of these networks are trained using the gradient that comes from a single GAN that tries to match these two distributions.

However, in the IAE, similar to VAEs or AAEs, the encoder and decoder are stacked on top of each other and trained jointly.

So the gradient that the encoder receives comes through the decoder and the conditioning vector.

In other words, in the ALI model, the input to the conditional likelihood is the samples of the prior distribution, whereas in the IAE, the input to the conditional likelihood is the samples of the variational posterior distribution, while the prior distribution is separately imposed on the aggregated posterior distribution by the regularization GAN.

This makes the training dynamic of IAEs similar to that of autoencoders, which encourages better reconstructions.

Recently, many variants of ALI have been proposed for improving its reconstruction performance.

For example, the HALI BID1 ) uses a Markovian generator to achieve better reconstructions, and ALICE BID14 augments the ALI's cost by a joint distribution matching cost function between (x,x) and (x, x), which is different from our reconstruction cost.

In this section, we show that the IAE can learn a global vs. local decomposition of information between the latent code and the implicit decoder.

We use the Gaussian distribution for both the global and local codes, and show that by adjusting the dimensions of the global and local codes, we can have a full control over the decomposition of information.

FIG1 shows the performance of the IAE on the MNIST dataset.

By removing the local code and using only a global code of size 20D FIG1 , the IAE becomes a deterministic autoencoder.

In this case, the global code of the IAE captures all the information of the data distribution and the IAE achieves almost perfect reconstructions.

By decreasing the global code size to 10D and using a 100D local code FIG1 ), the global code retains the global information of the digits such as the label information, while the local code captures small variations in the style of the digits.

By using a smaller global code of size 5D FIG1 ), the encoder loses more local information and thus the global code captures more abstract information.

For example, we can see from FIG1 that the encoder maps visually similar digits such as {3, 5, 8} or {4, 9} to the same global code, while the implicit decoder learns to invert this mapping and generate stochastic reconstructions that share the same high-level information with the original images.

Note that if we completely remove the global code, the local code captures all the information, similar to the standard unconditional GAN.

Figure 3 shows the performance of the IAE on the SVHN dataset.

When using a 150D global code with no local code (Figure 3b ), similar to the standard autoencoder, the IAE captures all the information by its global code and can achieve almost perfect reconstructions.

However, when using a 75D global code along with a 1000D local code (Figure 3c ), the global code of the IAE only captures the middle digit information as the global information, and loses the left and right digit information.

At the same time, the implicit decoder learns to invert the encoder distribution by keeping the middle digit and generating synthetic left and right SVHN digits with the same style of the middle digit.

Figure 4 shows the performance of the IAE on the CelebA dataset.

When using a 150D global code with no local code (Figure 4b ), the IAE achieves almost perfect reconstructions.

But when using a 50D global code along with a 1000D local code (Figure 4c ), the global code of the IAE only retains the global information of the face such as the general shape of the face, while the local code captures the local attributes of the face such as eyeglasses, mustache or smile.

In IAEs, by using a categorical global code along with a Gaussian local code, we can disentangle the discrete and continuous factors of variation, and perform clustering and semi-supervised learning.

Clustering.

In order to perform clustering with IAEs, we change the architecture of FIG0 by using a softmax function in the last layer of the encoder, as a continuous relaxation of the categorical global code.

The dimension of the categorical code is the number of categories that we wish the data to be clustered into.

The regularization GAN is trained directly on the continuous output probabilities of the softmax simplex, and imposes the categorical distribution on the aggregated posterior distribution.

This adversarial regularization imposes two constraints on the encoder output.

The first constraint is that the encoder has to make confident decisions about the cluster assignments.

The second constraint is that the encoder must distribute the points evenly across the clusters.

As a result, the global code only captures the discrete underlying factors of variation such as class labels, while the rest of the structure of the image is separately captured by the Gaussian local code of the implicit decoder.

Figure 5 shows the samples of the standard GAN and the IAE trained on the mixture of Gaussian data.

Figure 5b shows the samples of the GAN, which takes a 7D categorical and a 10D Gaussian noise vectors as the input.

Each sample is colored based on the one-hot noise vector that it was generated from.

We can see that the GAN has failed to associate the categorical noise vector to different mixture components, and generate the whole data solely by using its Gaussian noise vector.

Ignoring the categorical noise forces the GAN to do a continuous interpolation between different mixture components, which results in reducing the quality of samples.

Figure 5c shows the samples of the IAE whose implicit decoder architecture is the same as the GAN.

The IAE has a 7D categorical global code (inferred by the encoder) and a 10D Gaussian noise vector.

In this case, the inference network of the IAE learns to cluster the data in an unsupervised fashion, while its generative path learns to condition on the inferred cluster labels and generate each mixture component using the stochasticity of the Gaussian noise vector.

This example highlights the importance of using discrete latent variables for improving generative models.

A related work is the InfoGAN BID3 , which uses a reconstruction cost in the code space to prevent the GAN from ignoring the categorical noise vector.

The relationship of InfoGANs with IAEs is discussed in details in Section 3.

The style (local noise vector) is drawn from a Gaussian distribution and held fixed across each row.

FIG3 shows the clustering performance of the IAE on the MNIST dataset.

The IAE has a 30D categorical global latent code and a 10D Gaussian local code.

Each column corresponds to the conditional samples from one of the learned clusters (only 20 are shown).

The local code is sampled from the Gaussian distribution and held fixed across each row.

We can see that the discrete global latent code of the network has learned discrete factors of variation such as the digit identities, while the writing style information is separately captured by the continuous Gaussian noise vector.

This network obtains about 5% error rate in classifying digits in an unsupervised fashion, just by matching each cluster to a digit type.

Semi-Supervised Learning.

The IAE can be used for semi-supervised classification.

In order to incorporate the label information, we set the number of clusters to be the same as the number of class labels and additionally train the encoder weights on the labeled mini-batches to minimize the cross-entropy cost.

On the MNIST dataset with 100 labels, the IAE achieves the error rate of 1.40%.

In comparison, the AAE achieves 1.90%, and the Improved-GAN BID21 achieves 0.93%.

On the SVHN dataset with 1000 labels, the IAE achieves the error rate of 9.80%.

In comparison, the AAE achieves 17.70%, and the Improved-GAN achieves 8.11%.

In this section, we describe the "Flipped Implicit Autoencoder" (FIAE), which is a generative model that is very closely related to IAEs.

Let z be the latent code that comes from the prior distribution p(z).

The encoder of the FIAE FIG4 ) parametrizes an implicit distribution that uses the noise vector n to define the conditional likelihood distribution p(x|z).

The decoder of the FIAE parametrizes an implicit distribution that uses the noise vector to define the variational posterior distribution q(z|x).

In addition to the distributions defined in Section 2, we also define the joint latent reconstruction distribution s(x, z), and the aggregated latent reconstruction distribution s(z) as follows: DISPLAYFORM0 The objective of the standard variational inference is minimizing KL(q(x, z) p(x, z)), which is the variational upper-bound on KL(p data (x) p(x)).

The objective of FIAEs is the reverse KL divergence KL(p(x, z) q(x, z)), which is the variational upper-bound on KL(p(x) p data (x)).

The FIAE optimizes this variational bound by splitting it into a reconstruction term and a regularization term as follow: DISPLAYFORM1 Cond.

Entropy DISPLAYFORM2 where the conditional entropy H(z|x) is defined under the joint model distribution p(x, z).

Similar to the IAE, the FIAE has a regularization term and a reconstruction term (Equation 14 and Equation 15 ).

The regularization cost uses a GAN to train the encoder (conditional likelihood) such that the model distribution p(x) matches the data distribution p data (x).

The reconstruction cost uses a GAN to train both the encoder (conditional likelihood) and the decoder (variational posterior) such that the joint model distribution p(x, z) matches the joint latent reconstruction distribution s(x, z).Connections with ALI and BiGAN.

In ALI BID6 and BiGAN BID5 ) models, the input to the recognition network is the samples of the real data p data (x); however, in FIAEs, the recognition network only gets to see the synthetic samples that come from the simulated data p(x), while at the same time, the regularization cost ensures that the simulated data distribution is close the real data distribution.

Training the recognition network on the simulated data in FIAEs is in spirit similar to the "sleep" phase of the wake-sleep algorithm BID11 , during which the recognition network is trained on the samples that the network "dreams" up.

One of the flaws of training the recognition network on the simulated data is that early in the training, the simulated data do not look like the real data, and thus the recognition path learns to invert the generative path in part of the data space that is far from the real data distribution.

As the result, the reconstruction GAN might not be able to keep up with the moving simulated data distribution and get stuck in a local optimum.

However, in our experiments with FIAEs, we did not find this to be a major problem.

Connections with InfoGAN.

InfoGANs BID3 , similar to FIAEs, train the variational posterior network on the simulated data; however, as shown in Equation 13, InfoGANs use an explicit reconstruction cost function (e.g., Euclidean cost) on the code space for learning the variational posterior.

In order to compare FIAEs and InfoGANs, we train them on a toy dataset with four data-points and use a 2D Gaussian prior (Figure 8 and Figure 9 ).

Each colored cluster corresponds to the posterior distribution of one data-point.

In InfoGANs, using the Euclidean cost to reconstruct the code corresponds to learning a factorized Gaussian variational posterior distribution (Figure 8b) 2 .

This constraint on the variational posterior restricts the family of the conditional likelihoods that the model can learn by enforcing the generative path to learn a conditional likelihood whose true posterior could fit to the factorized Gaussian approximation of the posterior.

For example, we can see in Figure 8a that the model has learned a conditional likelihood whose true posterior is axis-aligned, so that it could better match the factorized Gaussian variational posterior (Figure 8b ).

In contrast, the FIAE can learn an arbitrarily expressive variational posterior distribution (Figure 9b) , which enables the generative path to learn a more expressive conditional likelihood and true posterior (Figure 9a) .One of the main flaws of optimizing the reverse KL divergence is that the variational posterior will have the mode-covering behavior rather than the mode-picking behavior.

For example, we can see from Figure 8b that the Gaussian posteriors of different data-points in InfoGAN have some overlap; but this is less of a problem in the FIAE (Figure 9b ), as it can learn a more expressive q(z|x).

This mode-averaging behavior of the posterior can be also observed in the wake-sleep algorithm, in which during the sleep phase, the recognition network is trained using the reverse KL divergence objective.

The FIAE objective is not only an upper-bound on KL(p(x) p data (x)), but is also an upper-bound on KL(p(z) q(z)) and KL(p(z|x) q(z|x)).

As a result, the FIAE matches the variational posterior q(z|x) to the true posterior p(z|x), and also matches the aggregated posterior q(z) to the prior p(z).

For example, we can see in Figure 9b that q(z) is very close to the Gaussian prior.

However, the InfoGAN objective is theoretically not an upper-bound on KL(p(x) p data (x)), KL(p(z) q(z)) or KL(p(z|x) q(z|x)).

As a result, in InfoGANs, the variational posterior q(z|x) need not be close to the true posterior p(z|x), or the aggregated posterior q(z) does not have to match the prior p(z).

Reconstruction.

In this section, we show that the variational posterior distribution of the FIAE can invert its conditional likelihood function by showing that the network can perform reconstructions of the images.

We make both the conditional likelihood and the variational posterior deterministic by removing both noise vectors n and .

FIG0 shows the performance of the FIAE with a code size of 15 on the test images of the MNIST dataset.

The reconstructions are obtained by first passing the image through the recognition network to infer its latent code, and then using the inferred latent code at the input of the conditional likelihood to generate the reconstructed image.

Clustering.

Similar to IAEs, we can use FIAEs for clustering.

We perform an experiment on the MNIST dataset by choosing a discrete categorical latent code z of size 10, which captures the digit identity; and a continuous Gaussian noise vector n of size 10, which captures the style of the digit.

The variational posterior distribution q(z|x) is also parametrized by an implicit distribution with a Gaussian noise vector of size 20, and performs inference only over the digit identity z. Once the network is trained, we can use the variational posterior to cluster the test images of the MNIST dataset.

This network achieves the error rate of about 2% in classifying digits in an unsupervised fashion by matching each categorical code to a digit type.

We observed that when there is uncertainty in the digit identity, different draws of the noise vector results in different one-hot vectors at the output of the recognition network, showing that the implicit decoder can efficiently capture the uncertainty.

In this paper, we proposed the implicit autoencoder, which is a generative autoencoder that uses implicit distributions to learn expressive variational posterior and conditional likelihood distributions.

We showed that in IAEs, the information of the data distribution is decomposed between the prior and the conditional likelihood.

When using a low dimensional Gaussian distribution for the global code, we showed that the IAE can disentangle high-level and abstract information from the low-level and local statistics.

We also showed that by using a categorical latent code, we can learn discrete factors of variation and perform clustering and semi-supervised learning.

In this section, we describe an information theoretic interpretation of the ELBO of IAEs (Equation 7 ) using the Bits-Back coding argument BID10 BID4 BID8 .

Maximizing the variational lower bound is equivalent to minimizing the expected description length of a source code for the data distribution p data (x) when the code is designed under the model distribution p(x).

In order to transmit x, the sender uses a two-part code.

It first transmits z, which ideally would only require H(z) bits; however, since the code is designed under p(z), the sender has to pay the penalty of KL(q(z) p(z)) extra bits to compensate for the mismatch between q(z) and p(z).

After decoding z, the receiver now has to resolve the uncertainty of q(x|z) in order to reconstruct x, which ideally requires the sender to transmit the second code of the length H(x|z) bits.

However, since the code is designed under p(x|z), the sender has to pay the penalty of E z∼q(z) KL(q(x|z) p(x|z)) extra bits on average to compensate for the fact that the conditional decoder p(x|z) has not perfectly captured the inverse encoder distribution q(x|z); i.e., the autoencoder has failed to achieve perfect stochastic reconstruction.

But for a given x, the sender could use the stochasticity of q(z|x) to encode other information.

Averaged over the data distribution, this would get the sender H(z|x) "bits back" that needs to be subtracted in order to find the true cost for transmitting x: DISPLAYFORM0 From Equation 25, we can see that the IAE only minimizes the extra number of bits required for transmitting x, while the VAE minimizes the total number of bits required for the transmission.

Continuous Variables.

The Bits-Back argument is also applicable to continuous random variables.

Suppose x and z are real-valued random variables.

Let h(x) and h(z) be the differential entropies of x and z; and H(x) and H(z) be the discrete entropies of the quantized versions of x and z, with the quantization interval of ∆x and ∆z.

We have DISPLAYFORM1 The sender first transmits the real-valued random variable z, which requires transmission of H(z) = h(z) − log ∆z bits, as well as KL(q(z) p(z)) extra bits.

As ∆z → 0, we will have H(z) → ∞, which, as expected, implies that the sender would need infinite number of bits to source code and send the real-valued random variable z. However, as we shall see, we are going to get most of these bits back from the receiver at the end.

After the first message, the sender then sends the second message, which requires transmission of h(x|z) − log ∆x bits, as well as E z∼q(z) KL(q(x|z) p(x|z)) extra bits.

Once the receiver decodes z, and form that decodes x, it can decode a secondary message of the average length H(z|x) = h(z|x) − log ∆z, which needs to be subtracted in order to find the true cost for transmitting x: DISPLAYFORM2 From Equation 28, we can interpret the IAE cost as the extra number of bits required for the transmission of x.

In IAEs, the global code (prior) captures the global information of data, while the remaining local information is captured by the local noise vector (conditional likelihood).

In this section, we describe the global vs. local decomposition of information from an information theoretic perspective.

In order to transmit x, the sender first transmits z and then transmits the residual bits required for reconstructing x, using a source code that is designed based on p(x, z).

If p(z) and p(x|z) are powerful enough, in theory, they can capture any q(x, z), and thus regardless of the decomposition of information, the sender would only need to send H data (x) bits.

In this case, the ELBO does not prefer one decomposition of information to another.

But if the capacities of p(z) and p(x|z) are limited, the sender will have to send extra bits due to the distribution mismatch, resulting in the regularization and reconstruction errors.

But now different decompositions of information will result in different numbers of extra bits.

So the sender has to decompose the information in a way that is compatible with the source codes that are designed based on p(z) and p(x|z).

The prior p(z) that we use in this work is a low-dimensional Gaussian or categorical distribution.

So the regularization cost encourages the sender to encode low-dimensional or simple concepts in z that is consistent with p(z); otherwise, the sender would need to pay a large cost for KL(q(z) p(z)).

The choice of the information encoded in z would also affect the extra number of bits of E z∼q(z) KL(q(x|z) p(x|z)) , which is the reconstruction cost.

This is because the conditional decoder p(x|z) with its limited capacity is supposed to capture the inverse encoder distribution q(x|z).

So the sender must encode the kind of information in z that after being observed, can maximally remove the stochasticity of q(x|z) so as to lower the burden on p(x|z) for matching to q(x|z).

So the reconstruction cost encourages learning the kind of concepts that can remove as much uncertainty as possible from the data distribution.

By balancing the regularization and reconstruction costs, the latent code learns global concepts which are low-dimensional or simple concepts that can maximally remove uncertainty from data.

Examples of global concepts are digit identities in the MNIST dataset, objects in natural images or topics in documents.

There are two methods to implement how the reconstruction GAN conditions on the global code.

Location-Dependent Conditioning.

Suppose the size of the first convolutional layer of the discriminator is (batch, width, height, channels).

We use a one layer neural network with 1000 ReLU hidden units to transform the global code of size (batch, global_code_size) to a spatial tensor of size (batch, width, height, 1).

We then broadcast this tensor across the channel dimension to get a tensor of size (batch, width, height, channels), and then add it to the first layer of the discriminator as an adaptive bias.

In this method, the latent vector has spatial and location-dependent information within the feature map.

This is the method that we used in deterministic and stochastic reconstruction experiments.

Location-Invariant Conditioning.

Suppose the size of the first convolutional layer of the discriminator is (batch, width, height, channels).

We use a linear mapping to transform the global code of size (batch, global_code_size) to a tensor of size (batch, channels).

We then broadcast this tensor across the width and height dimensions, and then add it to the first layer of the discriminator as an adaptive bias.

In this method, the global code is encouraged to learn the global information that is location-invariant such as the class label information.

We used this method in all the clustering and semi-supervised learning experiments.

The regularization discriminator in all the experiments is a two-layer neural network, where each layer has 2000 hidden units with the ReLU activation function.

The architecture of the encoder, the decoder and the reconstruction discriminator for each dataset is as follows.

@highlight

We propose a generative autoencoder that can learn expressive posterior and conditional likelihood distributions using implicit distributions, and train the model using a new formulation of the ELBO.